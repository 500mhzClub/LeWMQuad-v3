#!/usr/bin/env python3
"""Quantify out-of-frame steering robustness to onboard-odometry noise/drift.

The committed result uses zero-drift recovered egomotion. Real onboard odometry
(IMU + leg odometry) has per-step error that random-walks into drift over the
out-of-frame horizon. This perturbs the exact solved egomotion with per-step iid
yaw noise (rad) and relative translation noise, which accumulate to ~sqrt(gap)
drift, and reports out-of-frame steering vs noise level, stratified by in-cone
gap (steps since the landmark was last seen).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    _COLOR_RGB,
    _finite_float,
    _landmark_by_id,
    _object_color,
    _propagate_vectors,
    _seq_key,
    _steering_index,
)
from train_go2_hidden_target_memory_probe import _load_image  # noqa: E402
from audit_go2_ranged_memory_steering_mechanism import _mask_stats, _ranged_write  # noqa: E402
from add_exact_odometry_to_go2_dataset import solve_egomotion  # noqa: E402

# (yaw_sigma rad/step, trans_rel_sigma) levels: yaw sweep, then realistic combos.
LEVELS = [
    (0.0, 0.0),
    (0.01, 0.0),
    (0.02, 0.0),
    (0.05, 0.0),
    (0.10, 0.0),
    (0.20, 0.0),
    (0.03, 0.10),   # realistic: yaw ~1.7deg/step, leg-odom ~10% translation
    (0.05, 0.15),   # pessimistic
]
GAP_BUCKETS = [(2, 4), (5, 8), (9, 16), (17, 10**9)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-glob",
        default=".generated/go2_hidden_target_memory/observed_memory_gate_20260622/cv/val_holdout_*.jsonl",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    crgb = {c: torch.tensor(v, dtype=torch.float32) for c, v in _COLOR_RGB.items()}
    scale = float(args.range_scale_m)

    def steer(vec) -> int:
        return _steering_index(math.atan2(float(vec[1]), float(vec[0])))

    # gather queries: (area, xc, clean delta chain [(dx,dy,dyaw)...], true_cls, gap)
    queries = []
    for f in sorted(glob.glob(args.val_glob)):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            by[_seq_key(r)].append(r)
        for sk, rws in by.items():
            rws.sort(key=lambda r: int(r.get("episode_step", 0)))
            imgs = [_load_image(Path(r["rgb_path"]), image_size=args.image_size) for r in rws]
            lbids = [_landmark_by_id(r) for r in rws]
            ostat = []
            for pos in range(len(rws)):
                st = {}
                for oid, lm in lbids[pos].items():
                    c = _object_color(oid)
                    if c in crgb and c != "unknown":
                        a, xc = _mask_stats(imgs[pos], crgb[c], 0.20, 0.55, 0.08)
                        st[oid] = (a > 0.006, a, xc)
                ostat.append(st)
            for pos, row in enumerate(rws):
                for ev in row.get("go2_causal_memory_pair_selection", ()):
                    if not str(ev.get("pair_role", "")).startswith("current_"):
                        continue
                    if not bool(ev.get("seen_before", False)):
                        continue
                    oid = str(ev.get("object_id", ""))
                    lm = lbids[pos].get(oid)
                    if lm is None:
                        continue
                    bq = _finite_float(lm.get("bearing_body_rad"), float("nan"))
                    if math.isnan(bq):
                        continue
                    priors = [i for i in range(pos) if oid in ostat[i] and ostat[i][oid][0]]
                    if not priors:
                        continue
                    src = max(priors)
                    if src == pos:
                        continue
                    deltas = [solve_egomotion(lbids[j - 1], lbids[j]) for j in range(src + 1, pos + 1)]
                    if any(d is None for d in deltas):
                        continue
                    _, area, xc = ostat[src][oid]
                    queries.append((area, xc, deltas, _steering_index(bq), pos - src))

    def acc_at(yaw_sig, trans_sig, gap_lo=0, gap_hi=10**9):
        correct = total = 0
        for area, xc, deltas, true_cls, gap in queries:
            if not (gap_lo <= gap <= gap_hi):
                continue
            for _ in range(args.noise_draws):
                v = _ranged_write(area, xc, scale)
                for dx, dy, dyaw in deltas:
                    ndyaw = dyaw + float(rng.normal(0, yaw_sig)) if yaw_sig > 0 else dyaw
                    f = (1.0 + float(rng.normal(0, trans_sig))) if trans_sig > 0 else 1.0
                    v = _propagate_vectors(
                        v.reshape(1, 2),
                        torch.tensor([dx * f / scale, dy * f / scale, ndyaw]),
                    )[0]
                correct += int(steer(v) == true_cls)
                total += 1
        return correct / total if total else float("nan"), total

    report = {"n_queries": len(queries), "noise_draws": args.noise_draws, "levels": []}
    print(f"queries={len(queries)}  noise_draws={args.noise_draws}\n")
    header = "yaw_sig trans_sig |  overall  | " + " ".join(f"gap{lo}-{hi if hi<1000 else '+'}" for lo, hi in GAP_BUCKETS)
    print(header)
    for yaw_sig, trans_sig in LEVELS:
        overall, _ = acc_at(yaw_sig, trans_sig)
        per_gap = {f"{lo}-{hi if hi<1000 else '+'}": acc_at(yaw_sig, trans_sig, lo, hi)[0] for lo, hi in GAP_BUCKETS}
        report["levels"].append({"yaw_sigma": yaw_sig, "trans_rel_sigma": trans_sig,
                                 "overall": overall, "by_gap": per_gap})
        print(f"{yaw_sig:7.2f} {trans_sig:9.2f} |  {overall:.3f}   | "
              + " ".join(f"{v:.2f}" for v in per_gap.values()))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
