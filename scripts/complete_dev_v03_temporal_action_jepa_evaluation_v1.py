#!/usr/bin/env python3
"""Complete the temporal JEPA evaluation: parity, shared mask, full battery.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only with respect to training -- no
arm is retrained and no checkpoint is modified.

Three things this adds over the first evaluation pass:

1. A float32-versus-float16 parity check on a fixed subset, so that no small
   difference reported below can be an artefact of the repaired cache and
   batched-prediction path.

2. **One** changed-token threshold and boolean mask, derived from the FROZEN
   train representation and applied unchanged to both arms and to every action
   arm.  The first pass derived them per arm, which scored the two arms on
   different token subsets; that is corrected here.

3. Frozen-probe spatial performance on *predicted* future tokens against
   persistence, scored on the future frame's own raster labels, plus predicted
   occupied fractions so diffuse over-prediction cannot masquerade as
   improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import eval_dev_v03_temporal_action_jepa_v1 as V  # noqa: E402

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
OUT = CACHE / "temporal_action_jepa_v1" / "completion"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
CHANGED_QUANTILE = 0.75
DERANGEMENT_SEEDS = (11, 23, 37)
PARITY_ROWS = 24
EPOCH = 5

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def future_labels(rows) -> np.ndarray:
    """raster_labels for the t+240 endpoint of each retained sequence."""
    endpoints = {
        e["endpoint_identity_sha256"]: e
        for e in (json.loads(l) for l in (SUP / "endpoints.jsonl").read_text().splitlines() if l.strip())
    }
    by_pair = {}
    for line in (SUP / "pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        by_pair[p["content_sha256"]] = endpoints[p["next_endpoint_sha256"]]
    shards, out = {}, np.empty((len(rows), 64, 64), dtype=np.uint8)
    for i, row in enumerate(rows):
        e = by_pair[row["pair_sha256"]]
        shard = str(SUP / Path(e["scene_shard"]).parent)
        if shard not in shards:
            shards[shard] = np.fromfile(Path(shard) / "raster_labels.u1",
                                        dtype=np.uint8).reshape(-1, 64, 64)
        out[i] = shards[shard][int(e["shard_row"])]
    return out


def load_cache(name: str, count: int, arm) -> torch.Tensor:
    shape = (count, arm.token_grid[0] * arm.token_grid[1], arm.token_dim)
    path = EVAL / name
    return torch.from_numpy(
        np.ascontiguousarray(np.memmap(path, dtype=np.float16, mode="r", shape=shape))
    )


@torch.no_grad()
def predict_tokens(predictor, context, actions, mask, device, batch_size, dtype):
    out = []
    for start in range(0, len(context), batch_size):
        stop = start + batch_size
        out.append(
            predictor(
                context[start:stop].to(device=device, dtype=torch.float32),
                actions[start:stop].to(device),
                mask[start:stop].to(device),
            ).to(dtype).cpu()
        )
    return torch.cat(out, 0)


def score_against(pred, current, future, mask):
    pred = pred.float()
    cos = F.cosine_similarity(pred, future, dim=-1)[mask]
    err = (pred - future).pow(2).mean(-1)[mask]
    base = (current - future).pow(2).mean(-1)[mask]
    return {
        "changed_cosine": float(cos.mean()),
        "normalised_error_vs_persistence": float(err.mean() / base.mean().clamp_min(1e-12)),
        "tokens": int(cos.numel()),
    }


def derangement(n, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    while bool((order == torch.arange(n)).any()):
        order = torch.randperm(n, generator=generator)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    arm = E.VJepa21CroppedV03Arm()
    grid, dim = arm.token_grid, arm.token_dim
    n_train, n_sel = len(train_rows), len(sel_rows)
    labels_future_train = future_labels(train_rows)
    labels_future_sel = future_labels(sel_rows)
    labels_current = V.load_targets(ordered)

    record = {
        "status": STATUS, "claim_bearing": False,
        "read_only_with_respect_to_training": True,
        "checkpoints": {
            a: sorted(p.name for p in (CACHE / "temporal_action_jepa_v1" / f"arm_{a}").glob("checkpoint_epoch*.pt"))
            for a in ("frozen", "moving")
        },
        "evaluated_epoch": EPOCH,
    }
    record["checkpoint_note"] = (
        "every epoch checkpoint exists for both arms, but no epoch selection was "
        "performed: both arms are evaluated at the final epoch of the fixed "
        "six-epoch recipe, so the conclusion is bounded to that recipe"
    )

    # ---------------------------------------------------------------- parity
    predictors = {}
    for name in ("frozen", "moving"):
        checkpoint = torch.load(
            CACHE / "temporal_action_jepa_v1" / f"arm_{name}" / f"checkpoint_epoch{EPOCH}.pt",
            map_location="cpu",
        )
        predictor = T.Predictor().to(device)
        predictor.load_state_dict(checkpoint["predictor"])
        predictor.eval()
        predictors[name] = (predictor, checkpoint)

    parity = {}
    for name in ("frozen", "moving"):
        predictor, checkpoint = predictors[name]
        module, moved = V.build_arm_encoder(arm, checkpoint, device)
        pick = list(range(0, n_sel, max(1, n_sel // PARITY_ROWS)))[:PARITY_ROWS]
        paths = [sel_rows[i]["context_paths"][2] for i in pick]
        with torch.no_grad():
            batch = torch.stack([arm.preprocess(p) for p in paths]).to(device, torch.float32)
            fresh32 = module(batch.unsqueeze(2)).float().cpu()
        cached16 = load_cache(f"{name}_current.f16", len(ordered), arm)[
            [n_train + i for i in pick]
        ].float()
        feature_abs = (fresh32 - cached16).abs()

        ctx32 = []
        with torch.no_grad():
            for k in range(3):
                b = torch.stack(
                    [arm.preprocess(sel_rows[i]["context_paths"][k]) for i in pick]
                ).to(device, torch.float32)
                ctx32.append(module(b.unsqueeze(2)).float().cpu())
        ctx32 = torch.stack(ctx32, dim=1)
        ctx16 = torch.stack(
            [load_cache(f"{name}_ctx{k}.f16", n_sel, arm)[pick].float() for k in range(3)], dim=1
        )
        actions = T.action_tensor([sel_rows[i]["primitive"] for i in pick], torch.device("cpu"))
        mask = torch.ones(len(pick), grid[0] * grid[1], dtype=torch.bool)
        with torch.no_grad():
            # reference: float32 context, whole subset in one call
            ref = predictor(
                T.normalise(ctx32).to(device), actions.to(device), mask.to(device)
            ).float().cpu()
        # repaired path: float16 cache, batched, half-precision output
        rep = predict_tokens(
            predictor, T.normalise(ctx16.float()).half(), actions, mask, device,
            args.batch, torch.float16,
        ).float()
        pred_abs = (ref - rep).abs()

        fut32 = None
        with torch.no_grad():
            b = torch.stack(
                [arm.preprocess(sel_rows[i]["target_path"]) for i in pick]
            ).to(device, torch.float32)
            fut32 = module(b.unsqueeze(2)).float().cpu()
        fut16 = load_cache(f"{name}_sel_future.f16", n_sel, arm)[pick].float()
        m = torch.ones(len(pick), grid[0] * grid[1], dtype=torch.bool)
        s32 = score_against(ref, T.normalise(fresh32), T.normalise(fut32), m)
        s16 = score_against(rep, T.normalise(cached16), T.normalise(fut16), m)
        parity[name] = {
            "rows": len(pick),
            "feature_max_abs_diff": float(feature_abs.max()),
            "feature_mean_abs_diff": float(feature_abs.mean()),
            "feature_relative_mean": float(feature_abs.mean() / fresh32.abs().mean()),
            "prediction_max_abs_diff": float(pred_abs.max()),
            "prediction_mean_abs_diff": float(pred_abs.mean()),
            "float32_path": s32,
            "float16_batched_path": s16,
            "changed_cosine_delta": s16["changed_cosine"] - s32["changed_cosine"],
            "normalised_error_delta": (
                s16["normalised_error_vs_persistence"] - s32["normalised_error_vs_persistence"]
            ),
            "moved_encoder_tensors": moved,
        }
        del module
        torch.cuda.empty_cache()
    record["parity_check"] = parity

    # ------------------------------------------- ONE shared changed-token mask
    frozen_train_future = load_cache("frozen_train_future.f16", n_train, arm)
    frozen_current = load_cache("frozen_current.f16", len(ordered), arm)
    frozen_current_train = frozen_current[:n_train]      # the cache spans train+selection
    chunks = []
    for start in range(0, n_train, 256):
        stop = min(start + 256, n_train)
        chunks.append(
            (T.normalise(frozen_train_future[start:stop].float())
             - T.normalise(frozen_current_train[start:stop].float())).pow(2).mean(-1)
        )
    threshold = float(torch.quantile(torch.cat(chunks, 0).flatten().float(), CHANGED_QUANTILE))
    frozen_sel_future = load_cache("frozen_sel_future.f16", n_sel, arm)
    frozen_now = T.normalise(frozen_current[n_train:].float())
    frozen_future = T.normalise(frozen_sel_future.float())
    shared_mask = (frozen_future - frozen_now).pow(2).mean(-1) >= threshold
    record["changed_token_mask"] = {
        "derived_from": "frozen arm train representation only",
        "quantile": CHANGED_QUANTILE,
        "threshold": threshold,
        "selection_changed_tokens": int(shared_mask.sum()),
        "selection_total_tokens": int(shared_mask.numel()),
        "applied_unchanged_to": ["frozen", "moving", "correct", "shuffled", "persistence"],
        "first_pass_defect": (
            "the first evaluation pass derived the threshold and mask separately "
            "per arm, scoring the two arms on different token subsets; corrected here"
        ),
    }
    record["invariants"] = {
        "single_mask_both_arms_and_all_action_arms": True,
        "context_identical_across_correct_shuffled_persistence": True,
        "only_the_action_tensor_differs_in_the_sensitivity_comparison": True,
        "predictor_mask_argument_is_unused_in_forward": (
            "Predictor.forward emits all 768 target tokens and ignores its mask "
            "argument; the mask selects which positions enter the training loss and "
            "the evaluation score, so it cannot differ between action arms by "
            "construction. Context frames are fully visible -- the masking is on the "
            "target positions, not on the context."
        ),
    }

    # ------------------------------------------------------ spatial probes
    # probe trained on NORMALISED frozen true-future tokens, future labels;
    # applied unchanged to true future, persistence and predicted tokens.
    future_probe_path = OUT / "future_token_probe.pt"
    train_future_normalised = torch.cat(
        [T.normalise(frozen_train_future[i : min(i + 256, n_train)].float()).half()
         for i in range(0, n_train, 256)], 0
    )
    stacked = torch.cat([train_future_normalised, frozen_future.half()], 0).to(device)
    all_future_labels = np.concatenate([labels_future_train, labels_future_sel], 0)
    tr = np.arange(n_train)
    se = np.arange(n_train, n_train + n_sel)
    if future_probe_path.is_file():
        future_probe = S.SharedTokenToBev(dim).to(device)
        future_probe.load_state_dict(torch.load(future_probe_path, map_location=device))
        future_probe.eval()
    else:
        future_probe, _, _ = S.train_probe(
            stacked, all_future_labels, tr, se, grid, dim, device, "future_token_probe"
        )
        torch.save(future_probe.state_dict(), future_probe_path)
    true_future_pred = S.predict(future_probe, stacked, se, grid, device)
    del stacked
    torch.cuda.empty_cache()

    def spatial(tokens):
        gpu = tokens.half().to(device)
        out = S.predict(future_probe, gpu, np.arange(len(tokens)), grid, device)
        del gpu
        torch.cuda.empty_cache()
        return out

    fixed_probe = S.SharedTokenToBev(dim).to(device)
    fixed_probe.load_state_dict(torch.load(EVAL / "fixed_probe.pt", map_location=device))
    fixed_probe.eval()

    previous = json.loads((EVAL / "result.json").read_text())
    scenes = [r["scene"] for r in sel_rows]
    families = [r["family"] for r in sel_rows]
    actions = T.action_tensor([r["primitive"] for r in sel_rows], torch.device("cpu"))
    orders = [derangement(n_sel, s) for s in DERANGEMENT_SEEDS]

    record["arms"] = {}
    for name in ("frozen", "moving"):
        predictor, checkpoint = predictors[name]
        current = load_cache(f"{name}_current.f16", len(ordered), arm)
        ctx = torch.stack([load_cache(f"{name}_ctx{k}.f16", n_sel, arm) for k in range(3)], dim=1)
        fut = load_cache(f"{name}_sel_future.f16", n_sel, arm)
        now = T.normalise(current[n_train:].float())
        future = T.normalise(fut.float())
        context = T.normalise(ctx.float()).half()
        del ctx

        predicted = predict_tokens(predictor, context, actions,
                                   torch.ones(n_sel, grid[0]*grid[1], dtype=torch.bool),
                                   device, args.batch, torch.float16)
        shuffled_scores = [
            score_against(
                predict_tokens(predictor, context, actions[o],
                               torch.ones(n_sel, grid[0]*grid[1], dtype=torch.bool),
                               device, args.batch, torch.float16),
                now, future, shared_mask)
            for o in orders
        ]
        prediction = {
            "correct_action": score_against(predicted, now, future, shared_mask),
            "shuffled_action": {
                k: float(np.mean([s[k] for s in shuffled_scores])) for k in shuffled_scores[0]
            },
            "persistence": score_against(now, now, future, shared_mask),
        }
        prediction["correct_minus_shuffled"] = (
            prediction["correct_action"]["changed_cosine"]
            - prediction["shuffled_action"]["changed_cosine"]
        )
        prediction["correct_minus_persistence"] = (
            prediction["correct_action"]["changed_cosine"]
            - prediction["persistence"]["changed_cosine"]
        )

        per_scene = {}
        for scene in sorted(set(scenes)):
            pick = torch.tensor([i for i, s in enumerate(scenes) if s == scene])
            c = score_against(predicted[pick], now[pick], future[pick], shared_mask[pick])
            sh = float(np.mean([
                score_against(
                    predict_tokens(predictor, context[pick], actions[o][pick],
                                   torch.ones(len(pick), grid[0]*grid[1], dtype=torch.bool),
                                   device, args.batch, torch.float16),
                    now[pick], future[pick], shared_mask[pick])["changed_cosine"]
                for o in orders
            ]))
            per_scene[scene] = {
                "family": families[int(pick[0])],
                "correct_changed_cosine": c["changed_cosine"],
                "shuffled_changed_cosine": sh,
                "correct_minus_shuffled": c["changed_cosine"] - sh,
                "changed_tokens": c["tokens"],
            }

        # spatial on TRUE encoder tokens (current frame, current labels)
        gpu_current = current.half().to(device)
        fixed_selection = S.predict(fixed_probe, gpu_current, se, grid, device)
        fresh_probe, _, _ = S.train_probe(
            gpu_current, labels_current, tr, se, grid, dim, device, f"{name}/fresh_recheck"
        )
        fresh_selection = S.predict(fresh_probe, gpu_current, se, grid, device)
        del gpu_current
        torch.cuda.empty_cache()

        # spatial on PREDICTED future tokens vs persistence, future labels
        predicted_map = spatial(predicted)
        persistence_map = spatial(now)

        def block(pred, truth):
            s = S.summarise(pred, truth)
            return {
                "observable_occupied_iou": s["observable_occupied_iou"],
                "observable_occupied_precision": s["observable_occupied_precision"],
                "observable_occupied_recall": s["observable_occupied_recall"],
                "predicted_occupied_fraction_all_cells":
                    s["predicted_class_fraction_over_all_cells"]["occupied"],
                "target_occupied_fraction_all_cells":
                    s["target_class_fraction_over_all_cells"]["occupied"],
                "observable_free_iou": s["observable_free_iou"],
                "all_free_baseline_free_iou": s["all_free_baseline"]["observable_free_iou"],
            }

        record["arms"][name] = {
            "prediction": prediction,
            "per_scene": per_scene,
            "spatial_true_tokens": {
                "fixed_probe": block(fixed_selection, labels_current[se]),
                "fresh_probe": block(fresh_selection, labels_current[se]),
                "fixed_probe_per_family": S.grouped(fixed_selection, labels_current[se], families),
                "fresh_probe_per_family": S.grouped(fresh_selection, labels_current[se], families),
            },
            "spatial_predicted_future_tokens": {
                "note": "probe trained once on normalised FROZEN true-future tokens, applied unchanged",
                "predicted": block(predicted_map, labels_future_sel),
                "persistence": block(persistence_map, labels_future_sel),
                "true_future_upper_reference": block(true_future_pred, labels_future_sel),
                "predicted_per_family": S.grouped(predicted_map, labels_future_sel, families),
                "persistence_per_family": S.grouped(persistence_map, labels_future_sel, families),
            },
            "token_health": V.token_stats(now, future),
        }
        del current, context, predicted, now, future
        torch.cuda.empty_cache()
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    f, m = record["arms"]["frozen"], record["arms"]["moving"]
    record["decision_inputs"] = {
        "margin_frozen": f["prediction"]["correct_minus_shuffled"],
        "margin_moving": m["prediction"]["correct_minus_shuffled"],
        "margin_delta": m["prediction"]["correct_minus_shuffled"] - f["prediction"]["correct_minus_shuffled"],
        "scenes_where_moving_margin_higher": sum(
            1 for s in f["per_scene"]
            if m["per_scene"][s]["correct_minus_shuffled"] > f["per_scene"][s]["correct_minus_shuffled"]
        ),
        "scenes_total": len(f["per_scene"]),
        "fresh_probe_iou_frozen": f["spatial_true_tokens"]["fresh_probe"]["observable_occupied_iou"],
        "fresh_probe_iou_moving": m["spatial_true_tokens"]["fresh_probe"]["observable_occupied_iou"],
        "predicted_future_iou_frozen": f["spatial_predicted_future_tokens"]["predicted"]["observable_occupied_iou"],
        "predicted_future_iou_moving": m["spatial_predicted_future_tokens"]["predicted"]["observable_occupied_iou"],
    }
    d = record["decision_inputs"]
    record["DECISION"] = (
        "ACCEPT V-JEPA-INITIALISED TEMPORAL ACTION JEPA"
        if d["margin_delta"] > 0 and d["fresh_probe_iou_moving"] >= d["fresh_probe_iou_frozen"]
        else "REJECT ENCODER-MOVING RECIPE"
    )
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"parity": parity, "decision_inputs": d,
                      "DECISION": record["DECISION"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
