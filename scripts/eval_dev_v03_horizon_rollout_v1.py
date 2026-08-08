#!/usr/bin/env python3
"""Read-only H = 1,2,3,4 autoregressive rollout of the frozen selected models.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Strictly read-only: no model is trained, no
checkpoint is written, no frozen model is modified.  Every checkpoint is verified
against its recorded SHA-256 before use.

At each horizon h the predictor consumes its OWN previous prediction, never the
true intermediate token:

    p_1 = f([c0, c1, c2],        a_1)
    p_h = f([c_{h-1}, c_h, p_{h-1}], a_h)     h >= 2

with the same fixed sliding three-frame context and per-token normalisation used
in training.  Reported per horizon:

  * changed-token cosine and normalised error against the true tokens at that
    horizon, versus the persistence baseline (the current frame, held);
  * correct action sequence versus a fully shuffled sequence;
  * per-scene paired differences between models;
  * degradation from h=1;
  * canonical fixed-probe spatial metrics wherever native rasters exist.

The changed-token mask at each horizon is defined on the TRAIN split of the
one-step corpus and frozen; it is never fitted on the selection rows.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts import complete_dev_v03_temporal_action_jepa_evaluation_v1 as C  # noqa: E402
from scripts import audit_dev_v03_predicted_token_alignment_v1 as A  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
TWO = CACHE / "two_step"
HOR = CACHE / "horizons"
OUT = HOR / "evaluation"
MAX_H = 4
DERANGEMENT_SEED = 11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def encode(paths, device, blob: Path, batch=16):
    arm = E.VJepa21CroppedV03Arm()
    shape = (len(paths), R.TOKENS, R.DIM)
    if blob.is_file() and blob.stat().st_size == int(np.prod(shape) * 2):
        return
    module = arm.build(device, torch.float32)
    memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
    for s in range(0, len(paths), batch):
        chunk = paths[s : s + batch]
        pixels = torch.stack([arm.preprocess(p) for p in chunk]).to(device, torch.float32)
        memory[s : s + len(chunk)] = module(pixels.unsqueeze(2)).half().cpu().numpy()
    memory.flush()
    del module
    torch.cuda.empty_cache()


def load(blob: Path, count: int) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(
        np.memmap(blob, dtype=np.float16, mode="r", shape=(count, R.TOKENS, R.DIM))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frozen = json.loads((TWO / "selected_frozen_models.json").read_text())
    rows = [json.loads(l) for l in (HOR / "horizon_rows.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["max_horizon"] >= MAX_H]
    families = [r["family"] for r in rows]
    scenes = [r["scene"] for r in rows]
    n = len(rows)

    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    base_sel = [r for r in base if r["role"] == "checkpoint_selection"]
    pos = {r["pair_sha256"]: i for i, r in enumerate(base_sel)}
    idx = np.array([pos[r["pair_sha256"]] for r in rows])
    n_bt, n_bs = len(base_train), len(base_sel)

    ctx = torch.stack([R.load_cache(EVAL / f"frozen_ctx{k}.f16", n_bs)[idx] for k in range(3)], 1)
    current = R.load_cache(EVAL / "frozen_current.f16", n_bt + n_bs)[n_bt:][idx]
    context0 = T.normalise(ctx.float()).half()
    now = T.normalise(current.float())
    del ctx, current

    # true tokens at each horizon
    targets = {}
    for h in range(1, MAX_H + 1):
        blob = HOR / f"target_h{h}.f16"
        encode([r["horizon_frames"][h]["path"] for r in rows], device, blob)
        targets[h] = T.normalise(load(blob, n).float())

    # frozen changed-token masks, one per horizon, from the ONE-STEP TRAIN split
    matched = json.loads((TWO / "evaluation" / "MATCHED_24_EPOCH_result_epochs_0_23.json").read_text())
    masks = {1: (targets[1] - now).pow(2).mean(-1) >= matched["masks"]["step1_threshold"],
             2: (targets[2] - now).pow(2).mean(-1) >= matched["masks"]["step2_threshold"]}
    # h=3,4 reuse the h=2 threshold: no new threshold is fitted on selection rows
    for h in (3, 4):
        masks[h] = (targets[h] - now).pow(2).mean(-1) >= matched["masks"]["step2_threshold"]

    fixed = S.SharedTokenToBev(R.DIM).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()
    labels1 = C.future_labels(rows)

    generator = torch.Generator().manual_seed(DERANGEMENT_SEED)
    order = torch.randperm(n, generator=generator)
    while bool((order == torch.arange(n)).any()):
        order = torch.randperm(n, generator=generator)
    actions = [T.action_tensor([r["horizon_actions"][h - 1]["primitive"] for r in rows],
                               torch.device("cpu")) for h in range(1, MAX_H + 1)]

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "read_only": "no model trained, no checkpoint written, no frozen model modified",
        "rows": n, "max_horizon": MAX_H,
        "rows_by_family": {f: families.count(f) for f in sorted(set(families))},
        "mask_policy": ("h=1 and h=2 use the existing frozen thresholds; h=3 and h=4 reuse the "
                        "h=2 threshold. No threshold is fitted on the selection rows."),
        "masks": {str(h): int(masks[h].sum()) for h in masks},
        "models": {}, "comparison": {},
    }

    predictions = {}
    for name, meta in frozen["models"].items():
        path = Path(meta["path"])
        digest = sha256(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{name}: checkpoint hash mismatch; refusing to evaluate")
        predictor = T.Predictor(**R.PRED).to(device)
        predictor.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
        predictor.eval()

        def unroll(action_list):
            outs = []
            for s in range(0, n, args.batch):
                e = min(s + args.batch, n)
                m = torch.ones(e - s, R.TOKENS, dtype=torch.bool, device=device)
                c = context0[s:e].to(device=device, dtype=torch.float32)
                frame_a, frame_b = c[:, 1], c[:, 2]
                step = []
                with torch.no_grad():
                    prev = T.normalise(predictor(c, action_list[0][s:e].to(device), m))
                    step.append(prev.half().cpu())
                    for h in range(1, MAX_H):
                        ctx_h = torch.stack([frame_a, frame_b, prev], dim=1)
                        prev = T.normalise(predictor(ctx_h, action_list[h][s:e].to(device), m))
                        step.append(prev.half().cpu())
                        frame_a, frame_b = frame_b, step[-2].to(device).float()
                outs.append(step)
            return [torch.cat([o[h] for o in outs], 0) for h in range(MAX_H)]

        correct = unroll(actions)
        shuffled = unroll([a[order] for a in actions])
        predictions[name] = correct

        per_h = {}
        for h in range(1, MAX_H + 1):
            p, q, t, m = correct[h - 1].float(), shuffled[h - 1].float(), targets[h], masks[h]
            base_err = (now - t).pow(2).mean(-1)[m]
            entry = {
                "changed_cosine": float(F.cosine_similarity(p, t, dim=-1)[m].mean()),
                "normalised_error_vs_persistence": float(
                    (p - t).pow(2).mean(-1)[m].mean() / base_err.mean().clamp_min(1e-12)),
                "persistence_changed_cosine": float(F.cosine_similarity(now, t, dim=-1)[m].mean()),
                "shuffled_sequence_changed_cosine": float(F.cosine_similarity(q, t, dim=-1)[m].mean()),
                "changed_tokens": int(m.sum()),
            }
            entry["margin_vs_shuffled_sequence"] = (
                entry["changed_cosine"] - entry["shuffled_sequence_changed_cosine"])
            entry["advantage_over_persistence"] = (
                entry["changed_cosine"] - entry["persistence_changed_cosine"])
            entry["per_scene_changed_cosine"] = {
                sc: float(F.cosine_similarity(
                    p[[i for i, v in enumerate(scenes) if v == sc]],
                    t[[i for i, v in enumerate(scenes) if v == sc]], dim=-1)[
                        m[[i for i, v in enumerate(scenes) if v == sc]]].mean())
                for sc in sorted(set(scenes))}
            if h == 1:
                entry["spatial_fixed_probe"] = A.spatial_block(
                    fixed, correct[0], labels1, families, (24, 32), device)
            per_h[str(h)] = entry
        for h in range(2, MAX_H + 1):
            per_h[str(h)]["degradation_from_h1"] = (
                per_h["1"]["changed_cosine"] - per_h[str(h)]["changed_cosine"])
        record["models"][name] = {
            "epoch": meta["epoch"], "checkpoint_sha256": digest,
            "hash_verified": True, "objective": meta["objective"],
            "per_horizon": per_h,
        }
        del predictor, shuffled
        torch.cuda.empty_cache()
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    names = list(frozen["models"].keys())
    if len(names) == 2:
        a, b = names
        cmp = {}
        for h in range(1, MAX_H + 1):
            pa = record["models"][a]["per_horizon"][str(h)]
            pb = record["models"][b]["per_horizon"][str(h)]
            scenes_better = sum(1 for sc in pa["per_scene_changed_cosine"]
                                if pb["per_scene_changed_cosine"][sc] > pa["per_scene_changed_cosine"][sc])
            cmp[str(h)] = {
                f"{a}_changed_cosine": pa["changed_cosine"],
                f"{b}_changed_cosine": pb["changed_cosine"],
                "difference": pb["changed_cosine"] - pa["changed_cosine"],
                "normalised_error_difference": (pb["normalised_error_vs_persistence"]
                                                - pa["normalised_error_vs_persistence"]),
                f"scenes_where_{b}_better": scenes_better,
                "scenes_total": len(pa["per_scene_changed_cosine"]),
            }
        record["comparison"] = {"models": names, "per_horizon": cmp}
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"rows": n, "masks": record["masks"],
                      "comparison": record.get("comparison", {}).get("per_horizon")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
