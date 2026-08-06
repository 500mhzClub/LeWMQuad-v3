#!/usr/bin/env python3
"""Evaluate the matched temporal JEPA arms against the frozen v03 reference.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The moving arm is accepted only if it improves action-conditioned future
prediction **while preserving** the matched frozen v03 spatial representation.
A lower JEPA loss on its own is not evidence of success, so the JEPA loss is not
an acceptance criterion here.

Reported for every arm, on the scene-disjoint ``checkpoint_selection`` rows:

  * correct action vs shuffled actions, on changed tokens
  * prediction vs persistence
  * occupied precision / recall / IoU under a FIXED probe -- trained once on the
    frozen reference features and applied unchanged to every arm
  * fresh-probe spatial recoverability -- retrained per arm
  * raw token variance, effective rank and temporal change
  * per-family, with ``open_obstacle_field`` explicit
"""
from __future__ import annotations

import argparse
import collections
import copy
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

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
REFERENCE = CACHE / "frozen_spatial_reference"
OUT = CACHE / "temporal_action_jepa_v1" / "evaluation"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
CHANGED_QUANTILE = 0.75
DERANGEMENT_SEEDS = (11, 23, 37)


def load_targets(rows):
    cache = {}
    out = np.empty((len(rows), 64, 64), dtype=np.uint8)
    for i, row in enumerate(rows):
        shard = row["shard_dir"]
        if shard not in cache:
            cache[shard] = np.fromfile(
                Path(shard) / "raster_labels.u1", dtype=np.uint8
            ).reshape(-1, 64, 64)
        out[i] = cache[shard][row["shard_row"]]
    return out


@torch.no_grad()
def encode_frames(module, paths, arm, device, batch_size=16, cache: Path | None = None):
    """Dense image tokens, kept in float16 -- 491x768x1024 in float32 will not fit."""
    shape = (len(paths), arm.token_grid[0] * arm.token_grid[1], arm.token_dim)
    if cache is not None and cache.is_file() and cache.stat().st_size == int(
        np.prod(shape) * 2
    ):
        return torch.from_numpy(
            np.ascontiguousarray(np.memmap(cache, dtype=np.float16, mode="r", shape=shape))
        )
    out = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack(
            [arm.preprocess(p) for p in paths[start : start + batch_size]]
        ).to(device=device, dtype=torch.float32)
        out.append(module(batch.unsqueeze(2)).half().cpu())
    tokens = torch.cat(out, 0)
    if cache is not None:
        memory = np.memmap(cache, dtype=np.float16, mode="w+", shape=shape)
        memory[:] = tokens.numpy()
        memory.flush()
    return tokens


def build_arm_encoder(arm, checkpoint, device):
    """Official weights, with the moved tensors applied for the moving arm."""
    module = arm.build(device, torch.float32)
    moved = 0
    if checkpoint is not None and checkpoint.get("encoder_trainable"):
        state = module.state_dict()
        for name, tensor in checkpoint["encoder_trainable"].items():
            if name not in state:
                raise RuntimeError(f"unknown moved tensor {name}")
            state[name] = tensor.to(state[name].dtype)
            moved += 1
        module.load_state_dict(state, strict=True)
    module.eval().requires_grad_(False)
    return module, moved


def token_stats(tokens_now, tokens_future):
    now = tokens_now.reshape(-1, tokens_now.shape[-1]).float()
    centred = now - now.mean(0, keepdim=True)
    cov = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    p = eig / eig.sum().clamp_min(1e-12)
    return {
        "raw_token_variance": float(centred.pow(2).mean()),
        "raw_effective_rank": float((-(p * (p + 1e-12).log()).sum()).exp()),
        "raw_temporal_delta": float((tokens_future - tokens_now).pow(2).mean()),
    }


@torch.no_grad()
def run_predictor(predictor, context, actions, mask_tokens, device, batch_size=8):
    """Batched: the whole selection split at once exhausts 32 GiB."""
    out = []
    for start in range(0, len(context), batch_size):
        stop = start + batch_size
        out.append(
            predictor(
                context[start:stop].to(device=device, dtype=torch.float32),
                actions[start:stop].to(device),
                mask_tokens[start:stop].to(device),
            ).half().cpu()
        )
    return torch.cat(out, 0)


def prediction_metrics(predictor, context, current, future, actions, mask_tokens, device):
    """Correct vs shuffled action vs persistence, on the changed-token set."""
    results = {}
    with torch.no_grad():
        predicted = run_predictor(predictor, context, actions, mask_tokens, device)

        def score(pred):
            pred = pred.float()
            cos = F.cosine_similarity(pred, future, dim=-1)[mask_tokens]
            err = (pred - future).pow(2).mean(-1)[mask_tokens]
            base = (current - future).pow(2).mean(-1)[mask_tokens]
            return {
                "changed_cosine": float(cos.mean()),
                "mse": float(err.mean()),
                "normalised_error_vs_persistence": float(err.mean() / base.mean().clamp_min(1e-12)),
                "tokens": int(cos.numel()),
            }

        results["correct_action"] = score(predicted)
        shuffled = []
        for seed in DERANGEMENT_SEEDS:
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(actions), generator=generator)
            while bool((order == torch.arange(len(actions))).any()):
                order = torch.randperm(len(actions), generator=generator)
            shuffled.append(
                score(run_predictor(predictor, context, actions[order], mask_tokens, device))
            )
        results["shuffled_action"] = {
            k: float(np.mean([s[k] for s in shuffled])) for k in shuffled[0]
        }
        results["persistence"] = score(current)
    results["correct_minus_shuffled_changed_cosine"] = (
        results["correct_action"]["changed_cosine"] - results["shuffled_action"]["changed_cosine"]
    )
    results["correct_minus_persistence_changed_cosine"] = (
        results["correct_action"]["changed_cosine"] - results["persistence"]["changed_cosine"]
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--arms", default="frozen,moving")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    labels = load_targets(ordered)
    train_idx = np.arange(len(train_rows))
    sel_idx = np.arange(len(train_rows), len(ordered))
    arm_spec = E.VJepa21CroppedV03Arm()
    grid = arm_spec.token_grid

    # ---- FIXED probe: trained once on the frozen reference features --------
    receipt = json.loads((REFERENCE / "vjepa2_1_vitl_384_v03crop.json").read_text())
    reference = torch.from_numpy(
        np.ascontiguousarray(
            np.memmap(REFERENCE / "vjepa2_1_vitl_384_v03crop.f16", dtype=np.float16,
                      mode="r", shape=tuple(receipt["token_shape"]))
        )
    ).to(device)
    fixed_path = OUT / "fixed_probe.pt"
    if fixed_path.is_file():
        fixed = S.SharedTokenToBev(arm_spec.token_dim).to(device)
        fixed.load_state_dict(torch.load(fixed_path, map_location=device))
        fixed.eval()
    else:
        fixed, _, _ = S.train_probe(
            reference, labels, train_idx, sel_idx, grid, arm_spec.token_dim, device, "fixed_probe"
        )
        torch.save(fixed.state_dict(), fixed_path)
    reference_selection = S.predict(fixed, reference, sel_idx, grid, device)
    del reference
    torch.cuda.empty_cache()

    record = {
        "status": STATUS,
        "claim_bearing": False,
        "acceptance": (
            "the moving arm succeeds only if it improves correct-versus-shuffled future "
            "prediction while preserving the matched frozen v03 spatial representation, "
            "including occupied precision/IoU and open_obstacle_field; a lower JEPA loss "
            "alone is not evidence of success"
        ),
        "rows": {"train": len(train_rows), "checkpoint_selection": len(sel_rows)},
        "frozen_v03_reference": {
            "fixed_probe_on_reference_features": S.summarise(
                reference_selection, labels[sel_idx]
            ),
        },
        "arms": {},
    }

    for name in (a for a in args.arms.split(",") if a):
        arm_dir = CACHE / "temporal_action_jepa_v1" / f"arm_{name}"
        checkpoint = torch.load(arm_dir / f"checkpoint_epoch{args.epoch}.pt", map_location="cpu")
        module, moved = build_arm_encoder(arm_spec, checkpoint, device)

        blob = OUT / f"{name}_current.f16"
        current = encode_frames(
            module, [r["frames"][2]["path"] for r in ordered], arm_spec, device, cache=blob
        )

        # fixed probe, applied unchanged
        features = current.to(device)
        fixed_selection = S.predict(fixed, features, sel_idx, grid, device)
        # fresh probe, retrained on this arm's representation
        fresh, _, fresh_epoch = S.train_probe(
            features, labels, train_idx, sel_idx, grid, arm_spec.token_dim, device, f"{name}/fresh"
        )
        fresh_selection = S.predict(fresh, features, sel_idx, grid, device)
        del features
        torch.cuda.empty_cache()

        # prediction battery on the selection rows
        predictor = T.Predictor().to(device)
        predictor.load_state_dict(checkpoint["predictor"])
        predictor.eval()
        sel_context = torch.stack(
            [
                encode_frames(module, [r["context_paths"][k] for r in sel_rows], arm_spec,
                              device, cache=OUT / f"{name}_ctx{k}.f16")
                for k in range(3)
            ],
            dim=1,
        )
        sel_future_raw = encode_frames(
            module, [r["target_path"] for r in sel_rows], arm_spec, device,
            cache=OUT / f"{name}_sel_future.f16",
        )
        train_context_now = current[train_idx]
        train_future = encode_frames(
            module, [r["target_path"] for r in train_rows], arm_spec, device,
            cache=OUT / f"{name}_train_future.f16",
        )
        chunks = []
        for start in range(0, len(train_future), 256):
            stop = start + 256
            chunks.append(
                (T.normalise(train_future[start:stop].float())
                 - T.normalise(train_context_now[start:stop].float())).pow(2).mean(-1)
            )
        change = torch.cat(chunks, 0)
        threshold = float(torch.quantile(change.flatten().float(), CHANGED_QUANTILE))
        now = T.normalise(current[sel_idx].float())
        future = T.normalise(sel_future_raw.float())
        changed = (future - now).pow(2).mean(-1) >= threshold
        actions = T.action_tensor([r["primitive"] for r in sel_rows], torch.device("cpu"))

        context_normalised = T.normalise(sel_context.float()).half()
        del sel_context
        prediction = prediction_metrics(
            predictor, context_normalised, now, future, actions, changed, device
        )
        per_family_pred = {}
        families = [r["family"] for r in sel_rows]
        for family in sorted(set(families)):
            pick = torch.tensor([i for i, f in enumerate(families) if f == family])
            per_family_pred[family] = prediction_metrics(
                predictor, context_normalised[pick], now[pick], future[pick],
                actions[pick], changed[pick], device,
            )

        record["arms"][name] = {
            "checkpoint_epoch": args.epoch,
            "moved_encoder_tensors": moved,
            "changed_token_threshold": {
                "quantile": CHANGED_QUANTILE, "value": threshold,
                "frozen_on": "train rows", "selection_changed_tokens": int(changed.sum()),
                "selection_total_tokens": int(changed.numel()),
            },
            "prediction": prediction,
            "prediction_per_family": per_family_pred,
            "fixed_probe": {
                "note": "trained once on the frozen reference features, applied unchanged",
                "checkpoint_selection": S.summarise(fixed_selection, labels[sel_idx]),
                "per_family": S.grouped(fixed_selection, labels[sel_idx], families),
            },
            "fresh_probe": {
                "selected_epoch": fresh_epoch,
                "checkpoint_selection": S.summarise(fresh_selection, labels[sel_idx]),
                "per_family": S.grouped(fresh_selection, labels[sel_idx], families),
            },
            "token_health": token_stats(now, future),
        }
        del module, predictor, context_normalised, sel_future_raw, train_future
        torch.cuda.empty_cache()
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    # ---- comparison ---------------------------------------------------------
    if {"frozen", "moving"} <= set(record["arms"]):
        f, m = record["arms"]["frozen"], record["arms"]["moving"]
        record["comparison"] = {
            "action_margin_frozen": f["prediction"]["correct_minus_shuffled_changed_cosine"],
            "action_margin_moving": m["prediction"]["correct_minus_shuffled_changed_cosine"],
            "action_margin_delta": (
                m["prediction"]["correct_minus_shuffled_changed_cosine"]
                - f["prediction"]["correct_minus_shuffled_changed_cosine"]
            ),
            "fixed_probe_occupied_iou_frozen": f["fixed_probe"]["checkpoint_selection"][
                "observable_occupied_iou"],
            "fixed_probe_occupied_iou_moving": m["fixed_probe"]["checkpoint_selection"][
                "observable_occupied_iou"],
            "fresh_probe_occupied_iou_frozen": f["fresh_probe"]["checkpoint_selection"][
                "observable_occupied_iou"],
            "fresh_probe_occupied_iou_moving": m["fresh_probe"]["checkpoint_selection"][
                "observable_occupied_iou"],
            "open_obstacle_field_fresh_frozen": (
                f["fresh_probe"]["per_family"].get("open_obstacle_field") or {}
            ).get("observable_occupied_iou"),
            "open_obstacle_field_fresh_moving": (
                m["fresh_probe"]["per_family"].get("open_obstacle_field") or {}
            ).get("observable_occupied_iou"),
        }
        c = record["comparison"]
        c["ACCEPTED"] = bool(
            c["action_margin_delta"] > 0
            and c["fresh_probe_occupied_iou_moving"] >= c["fresh_probe_occupied_iou_frozen"]
            and (c["open_obstacle_field_fresh_moving"] or 0)
            >= (c["open_obstacle_field_fresh_frozen"] or 0)
        )
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record.get("comparison", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
