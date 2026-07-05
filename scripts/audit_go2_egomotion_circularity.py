#!/usr/bin/env python3
"""Integrity check: does the recovered egomotion leak the steering answer through
the queried landmark?

`exact_body_motion` is recovered as a 2D rigid solve over the static-landmark
constellation, which at the query frame INCLUDES the queried landmark's true body
position. This probe re-solves the full src->query egomotion chain EXCLUDING the
query's own object_id at every pair, propagates the RGB ranged write, and compares
out-of-frame steering to the all-landmark baseline. If accuracy holds, the
egomotion is genuinely the robot's motion (recoverable from other landmarks =
what onboard proprioception provides), and the headline result is leak-free.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-glob",
        default=".generated/go2_hidden_target_memory/observed_memory_gate_20260622/cv/val_holdout_*.jsonl",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    crgb = {c: torch.tensor(v, dtype=torch.float32) for c, v in _COLOR_RGB.items()}
    scale = float(args.range_scale_m)
    variants = {"all_landmark": [0, 0], "query_excluded": [0, 0]}
    n_total = 0
    n_excl_infeasible = 0  # query has <2 other landmarks at some pair in the chain

    def steer(vec) -> int:
        return _steering_index(math.atan2(float(vec[1]), float(vec[0])))

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
                        continue  # in-frame now; not an out-of-frame test
                    _, area, xc = ostat[src][oid]
                    true_cls = _steering_index(bq)
                    # build both egomotion chains
                    deltas_all = [solve_egomotion(lbids[j - 1], lbids[j]) for j in range(src + 1, pos + 1)]
                    deltas_ex = [
                        solve_egomotion(lbids[j - 1], lbids[j], exclude_oid=oid)
                        for j in range(src + 1, pos + 1)
                    ]
                    if any(d is None for d in deltas_all):
                        continue
                    n_total += 1
                    # all-landmark
                    v = _ranged_write(area, xc, scale)
                    for d in deltas_all:
                        v = _propagate_vectors(v.reshape(1, 2), torch.tensor([d[0] / scale, d[1] / scale, d[2]]))[0]
                    variants["all_landmark"][0] += int(steer(v) == true_cls)
                    variants["all_landmark"][1] += 1
                    # query-excluded
                    if any(d is None for d in deltas_ex):
                        n_excl_infeasible += 1
                        continue
                    v = _ranged_write(area, xc, scale)
                    for d in deltas_ex:
                        v = _propagate_vectors(v.reshape(1, 2), torch.tensor([d[0] / scale, d[1] / scale, d[2]]))[0]
                    variants["query_excluded"][0] += int(steer(v) == true_cls)
                    variants["query_excluded"][1] += 1

    report = {
        "n_out_of_frame_queries": n_total,
        "query_excluded_infeasible_pairs": n_excl_infeasible,
        "out_of_frame_steering_acc": {
            k: {"correct": v[0], "n": v[1], "acc": (v[0] / v[1] if v[1] else float("nan"))}
            for k, v in variants.items()
        },
    }
    print(json.dumps(report, indent=2))
    a = report["out_of_frame_steering_acc"]["all_landmark"]["acc"]
    e = report["out_of_frame_steering_acc"]["query_excluded"]["acc"]
    print(f"\nall_landmark={a:.3f}  query_excluded={e:.3f}  "
          f"=> {'LEAK-FREE' if e >= 0.95 else 'POSSIBLE LEAK (investigate)'}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
