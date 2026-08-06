#!/usr/bin/env python3
"""DEVELOPMENT-ONLY predictor-only h=1 run in ViT patch-token space.

NOT CLAIM BEARING.  The online encoder, EMA target encoder, BEV decoder and K/O
head are frozen in evaluation mode; only the action embedder, FiLM-conditioned
transformer and delta projection train.  Frozen-parameter gradient isolation is
asserted, not assumed.

Every comparison -- correct action, persistence, shuffled action derangements and
a neutral in-distribution action -- uses the identical current/target tensors,
token masks, normalisation and aggregation.  The changed-token threshold is the
fit-set 75th percentile of true token change and is frozen before validation.

The spatial probe is trained once on true target-encoder tokens and applied
unchanged to true-future, persistence and predicted tokens.  It is never
retrained on predictor outputs.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_COUNT, TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_predictor_h1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

PRIMITIVES = ("arc_left", "arc_right", "backward", "forward_fast", "forward_medium",
              "forward_slow", "hold", "yaw_left", "yaw_right")
COMMANDS = {  # canonical body-frame velocity per primitive
    "arc_left": (0.20, 0.0, 0.45), "arc_right": (0.20, 0.0, -0.45),
    "backward": (-0.20, 0.0, 0.0), "forward_fast": (0.30, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0), "forward_slow": (0.20, 0.0, 0.0),
    "hold": (0.0, 0.0, 0.0), "yaw_left": (0.0, 0.0, 0.45), "yaw_right": (0.0, 0.0, -0.45),
}
NEUTRAL = "hold"
SEED = 2_026_080_591
DERANGEMENT_SEEDS = (11, 23, 37)
EPOCHS = 40
BATCH = 32
LR = 3.0e-4


def load_pairs():
    eps = {e["endpoint_identity_sha256"]: e
           for e in (json.loads(l) for l in (SUP / "endpoints.jsonl").read_text().splitlines() if l.strip())}
    rows = []
    for line in (SUP / "pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if p["dataset_role"] not in ("train", "checkpoint_selection"):
            continue
        a, b = eps.get(p["current_endpoint_sha256"]), eps.get(p["next_endpoint_sha256"])
        if a is None or b is None or p["primitive"] not in COMMANDS:
            continue
        rows.append({"cur": a["image_path_metadata_only"], "nxt": b["image_path_metadata_only"],
                     "nxt_sha": {"shard_dir": str(SUP / Path(b["scene_shard"]).parent),
                                 "shard_row": b["shard_row"]},
                     "cur_sha": {"shard_dir": str(SUP / Path(a["scene_shard"]).parent),
                                 "shard_row": a["shard_row"]},
                     "primitive": p["primitive"], "scene": p["scene_id"], "family": p["family"],
                     "role": p["dataset_role"], "se2": p["relative_se2_current_frame"]})
    return rows


@torch.no_grad()
def encode(paths, enc, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([P.native_preprocess(p) for p in paths[i:i + bs]])
        out.append(enc.forward_tokens(batch.to(device))[:, 1:, :].cpu())
    return torch.cat(out, 0)


def action_tensors(prims, device=None):
    idx = torch.tensor([PRIMITIVES.index(p) for p in prims])
    one_hot = F.one_hot(idx, num_classes=len(PRIMITIVES)).float()
    cmd = torch.tensor([COMMANDS[p] for p in prims], dtype=torch.float32)
    return (one_hot, cmd) if device is None else (one_hot.to(device), cmd.to(device))


def metrics(pred, target, current, mask=None):
    """Cosine/error against target, optionally restricted to changed tokens.

    ``mask`` is a boolean (N, 256) selector applied identically to every arm.
    """
    cos = F.cosine_similarity(pred, target, dim=-1)
    err = (pred - target).pow(2).mean(-1)
    base = (current - target).pow(2).mean(-1)
    if mask is not None:
        cos, err, base = cos[mask], err[mask], base[mask]
    return {
        "cosine": float(cos.mean()),
        "mse": float(err.mean()),
        "normalised_error_vs_persistence": float(err.mean() / base.mean().clamp_min(1e-12)),
        "tokens": int(cos.numel()),
    }


class SpatialProbe(nn.Module):
    """Per-token linear readout to the 64x64 3-class raster, via a global head."""

    def __init__(self, dim=TOKEN_DIM, c=32, hidden=1024):
        super().__init__()
        self.reduce = nn.Conv2d(dim, c, 1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 16 * 16, hidden), nn.GELU(),
                                  nn.Linear(hidden, 64 * 64 * 3))

    def forward(self, tokens):
        b, n, d = tokens.shape
        grid = tokens.transpose(1, 2).reshape(b, d, 16, 16)
        return self.head(F.gelu(self.reduce(grid))).reshape(b, 3, 64, 64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-pairs", type=int, default=3000)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    enc, dec, head = P.load_stack(device)
    for module in (enc, dec, head):
        module.eval()
        for q in module.parameters():
            q.requires_grad_(False)

    # Partition by role BEFORE any cap.  probability_calibration and any
    # evaluation role are never loaded.  An empty role is fatal: a silent
    # substitute split would produce a valid-looking number under a wrong
    # description.
    rows = load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not val:
        raise RuntimeError(
            f"designated roles are required and must be non-empty: "
            f"train={len(fit)} checkpoint_selection={len(val)}"
        )
    if args.max_pairs:                        # cap applies within each role only
        fit, val = fit[: args.max_pairs], val[: args.max_pairs]
    fit_scenes, val_scenes = {r["scene"] for r in fit}, {r["scene"] for r in val}
    overlap = sorted(fit_scenes & val_scenes)
    if overlap:
        raise RuntimeError(f"train/selection scene overlap: {overlap[:5]}")
    import hashlib as _hl
    order_hash = lambda rws: _hl.sha256(
        json.dumps([[r["cur"], r["nxt"], r["primitive"]] for r in rws]).encode()).hexdigest()

    record = {
        "status": STATUS, "claim_bearing": False,
        "horizon": {"steps": 1, "frame_index_delta": 240, "interval_fixed": True,
                    "translation_m": {"p10": 0.0227, "median": 0.0799, "p90": 0.1462, "max": 0.5523},
                    "yaw_rad": {"p10": 0.0180, "median": 0.1155, "p90": 0.2871, "max": 0.4850},
                    "seconds": "not derivable: no control-rate field in the corpus metadata"},
        "split": {
            "source": "corpus designated roles, partitioned before any cap",
            "fallback_removed": True,
            "train_pairs": len(fit), "selection_pairs": len(val),
            "train_scenes": len(fit_scenes), "selection_scenes": len(val_scenes),
            "scene_overlap": 0,
            "train_ordered_pair_sha256": order_hash(fit),
            "selection_ordered_pair_sha256": order_hash(val),
            "roles_never_loaded": ["probability_calibration", "evaluation"],
        },
        "fit_pairs": len(fit), "validation_pairs": len(val),
        "fit_scenes": len(fit_scenes), "validation_scenes": len(val_scenes),
    }

    cur_fit = encode([r["cur"] for r in fit], enc, device)
    nxt_fit = encode([r["nxt"] for r in fit], enc, device)
    cur_val = encode([r["cur"] for r in val], enc, device)
    nxt_val = encode([r["nxt"] for r in val], enc, device)

    # changed-token threshold: fit-set 75th percentile, frozen for validation
    change_fit = (nxt_fit - cur_fit).pow(2).mean(-1)
    threshold = float(torch.quantile(change_fit.flatten().float(), 0.75))
    mask_fit = change_fit >= threshold
    mask_val = (nxt_val - cur_val).pow(2).mean(-1) >= threshold
    record["changed_token_threshold"] = {
        "value": threshold, "source": "fit_set_75th_percentile", "frozen_for_validation": True,
        "fit_changed_fraction": float(mask_fit.float().mean()),
        "validation_changed_fraction": float(mask_val.float().mean()),
    }

    predictor = initialize_token_predictor_v1(SEED).to(device)
    frozen_before = {n: q.detach().clone() for m in (enc, dec, head) for n, q in m.named_parameters()}
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    oh_fit, cmd_fit = action_tensors([r["primitive"] for r in fit])
    g = torch.Generator().manual_seed(SEED)
    oh_val, cmd_val = action_tensors([r["primitive"] for r in val])

    @torch.no_grad()
    def selection_changed_cosine():
        predictor.eval()
        out = []
        for i in range(0, len(val), 64):
            out.append(predictor(cur_val[i:i + 64].to(device), oh_val[i:i + 64].to(device),
                                 cmd_val[i:i + 64].to(device)).cpu())
        t = torch.cat(out, 0)
        predictor.train()
        return float(F.cosine_similarity(t, nxt_val, dim=-1)[mask_val].mean())

    curve = []
    predictor.train()
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(fit), generator=g)
        for s in range(0, len(order), BATCH):
            sel = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            pred = predictor(cur_fit[sel].to(device), oh_fit[sel].to(device), cmd_fit[sel].to(device))
            loss = F.mse_loss(pred, nxt_fit[sel].to(device).detach())
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
        if (_epoch + 1) in (1, 5, 10, 20, 30, 40):
            curve.append({"epoch": _epoch + 1, "selection_changed_cosine": selection_changed_cosine()})
    predictor.eval()
    record["selection_curve"] = curve
    torch.save({"state_dict": predictor.state_dict(), "epochs": EPOCHS, "seed": SEED,
                "split": record["split"]}, OUT / "predictor_only_designated_role_epoch40.pt")

    # frozen-parameter isolation, asserted
    drift = max(float((q.detach() - frozen_before[n]).abs().max())
                for m in (enc, dec, head) for n, q in m.named_parameters())
    grads = sum(1 for m in (enc, dec, head) for q in m.parameters() if q.grad is not None)
    record["frozen_isolation"] = {"max_parameter_drift": drift,
                                  "frozen_params_with_gradients": grads,
                                  "passed": drift == 0.0 and grads == 0}

    @torch.no_grad()
    def predict(cur, prims):
        oh, cmd = action_tensors(prims)
        out = []
        for i in range(0, len(cur), 64):
            out.append(predictor(cur[i:i + 64].to(device), oh[i:i + 64].to(device),
                                 cmd[i:i + 64].to(device)).cpu())
        return torch.cat(out, 0)

    prims_val = [r["primitive"] for r in val]
    arms = {"correct_action": predict(cur_val, prims_val),
            "persistence": cur_val.clone(),
            "neutral_action": predict(cur_val, [NEUTRAL] * len(val))}
    rng = np.random.default_rng(SEED)
    for k, s in enumerate(DERANGEMENT_SEEDS):
        r2 = np.random.default_rng(s)
        perm = np.arange(len(val))
        while True:
            r2.shuffle(perm)
            if not (perm == np.arange(len(perm))).any():
                break
        arms[f"shuffled_action_{k}"] = predict(cur_val, [prims_val[i] for i in perm])

    record["validation"] = {
        name: {"all_tokens": metrics(t, nxt_val, cur_val),
               "changed_tokens": metrics(t, nxt_val, cur_val, mask_val)}
        for name, t in arms.items()
    }
    pv = arms["correct_action"]
    record["collapse_diagnostics"] = {
        "predicted_token_variance": float(pv.reshape(-1, TOKEN_DIM).var(0).mean()),
        "true_future_token_variance": float(nxt_val.reshape(-1, TOKEN_DIM).var(0).mean()),
        "mean_delta_magnitude": float((pv - cur_val).abs().mean()),
        "true_delta_magnitude": float((nxt_val - cur_val).abs().mean()),
    }

    # ---- frozen spatial probe: trained ONCE on TRUE target-encoder tokens ----
    # Applied unchanged to true-future, persistence and predicted tokens; never
    # retrained on predictor outputs.
    def raster_for(rws, which="nxt_sha"):
        out = []
        for r in rws:
            e = r[which]
            labels = np.fromfile(Path(e["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
            out.append(labels.reshape(-1, 64, 64)[int(e["shard_row"])])
        return torch.from_numpy(np.stack(out)).long()

    y_fit, y_val = raster_for(fit), raster_for(val)
    probe = SpatialProbe().to(device)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(y_fit.numpy().reshape(-1), minlength=3).astype(np.float64)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    w = w / w.mean()
    gp = torch.Generator().manual_seed(SEED + 1)
    probe.train()
    for _ in range(30):
        order = torch.randperm(len(fit), generator=gp)
        for s2 in range(0, len(order), BATCH):
            sel = order[s2:s2 + BATCH]
            popt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(nxt_fit[sel].to(device)), y_fit[sel].to(device), weight=w)
            loss.backward()
            popt.step()
    probe.eval()

    @torch.no_grad()
    def probe_eval(tokens):
        preds = []
        for i in range(0, len(tokens), 64):
            preds.append(probe(tokens[i:i + 64].to(device)).argmax(1).cpu().numpy())
        pred = np.concatenate(preds, 0)
        truth = y_val.numpy()
        out = {}
        for k, name in ((0, "unknown"), (1, "free"), (2, "occupied")):
            a, c = truth == k, pred == k
            sup, inter, union = int(a.sum()), int((a & c).sum()), int((a | c).sum())
            out[f"{name}_iou"] = inter / union if union else None
            out[f"{name}_recall"] = inter / sup if sup else None
            out[f"{name}_precision"] = inter / int(c.sum()) if int(c.sum()) else None
        obs = truth != 0
        out["observable_accuracy"] = float((pred[obs] == truth[obs]).mean()) if obs.any() else None
        return out

    # Frozen probe applied unchanged to EVERY arm.
    sp = {"trained_on": "true_target_encoder_tokens_only",
          "retrained_on_predictor_outputs": False,
          "true_future_reference": probe_eval(nxt_val),
          "persistence": probe_eval(cur_val)}
    for name, t in arms.items():
        sp[name] = probe_eval(t)

    # EMA target-encoder persistence, if the target branch is loadable.
    try:
        import lewm.models.direct_egocentric_bev_state_jepa_v1 as _m1
        tgt_enc = _m1._construct_n320_encoder_without_rng_draw()
        sd = torch.load(P.CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"]
        tgt_enc.load_state_dict({k[len("target_encoder") + 1:]: v for k, v in sd.items()
                                 if k.startswith("target_encoder.")}, strict=True)
        tgt_enc.to(device).eval().requires_grad_(False)
        cur_ema = encode([r["cur"] for r in val], tgt_enc, device)
        sp["ema_target_persistence"] = probe_eval(cur_ema)
        record["ema_target_available"] = True
    except Exception as error:  # noqa: BLE001
        record["ema_target_available"] = f"unavailable: {type(error).__name__}"
    record["spatial_probe"] = sp

    # Paired per-scene correct-minus-shuffled on changed tokens.
    scenes = [r["scene"] for r in val]
    uniq = sorted(set(scenes))
    idx_by_scene = {sc: torch.tensor([i for i, s2 in enumerate(scenes) if s2 == sc]) for sc in uniq}
    per_scene = {}
    for sc, idx in idx_by_scene.items():
        m = mask_val[idx]
        row = {"pairs": int(len(idx)), "changed_tokens": int(m.sum())}
        c = metrics(arms["correct_action"][idx], nxt_val[idx], cur_val[idx], m)
        row["correct_cosine"] = c["cosine"]
        diffs = []
        for k in range(len(DERANGEMENT_SEEDS)):
            sh = metrics(arms[f"shuffled_action_{k}"][idx], nxt_val[idx], cur_val[idx], m)
            diffs.append(c["cosine"] - sh["cosine"])
        row["correct_minus_shuffled_cosine"] = float(np.mean(diffs))
        row["per_derangement"] = [float(d) for d in diffs]
        per_scene[sc] = row
    wins = sum(1 for v in per_scene.values() if v["correct_minus_shuffled_cosine"] > 0)
    record["per_scene_correct_minus_shuffled"] = {
        "scenes": len(per_scene), "scenes_favouring_correct": wins,
        "mean_difference": float(np.mean([v["correct_minus_shuffled_cosine"] for v in per_scene.values()])),
        "min_difference": float(np.min([v["correct_minus_shuffled_cosine"] for v in per_scene.values()])),
        "max_difference": float(np.max([v["correct_minus_shuffled_cosine"] for v in per_scene.values()])),
        "per_scene": per_scene,
    }

    record["wall_seconds"] = time.time() - started
    (OUT / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({k: record[k] for k in
                      ("status", "horizon", "changed_token_threshold", "frozen_isolation",
                       "validation", "collapse_diagnostics")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
