#!/usr/bin/env python3
"""Isolate the systematic gap=2-4 out-of-frame steering error: odometry/propagation
convention bug vs RGB range-estimate error.

For each just-left-frame steering query (a `current_*` seen_before event whose
color mask last fired `gap` positions ago), we propagate the TRUE landmark
position at the sighting (no range error) through the recorded body odometry to
the query position, under several hypotheses, and compare the resulting steering
class to the truth:

  true_write + block odometry  (as the controller propagates)
  true_write + 0.5*translation/rotation  (tests fixed-window double-counting)
  true_write + negated dyaw    (tests a yaw sign-convention flip)
  true_write + rotation only   (drops translation)
  rgb_write  + block odometry  (adds back the area->range estimate error)

If true_write propagation is already wrong, the bug is odometry/convention; if
only rgb_write is wrong, it is the monocular range estimate.
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
    _vector_target,
)
from train_go2_hidden_target_memory_probe import _load_image  # noqa: E402
from audit_go2_ranged_memory_steering_mechanism import (  # noqa: E402
    _mask_stats,
    _ranged_write,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-glob",
        default=".generated/go2_hidden_target_memory/observed_memory_gate_20260622/cv/val_holdout_*.jsonl",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--area-threshold", type=float, default=0.006)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--motion-field", default="integrated_body_motion_block")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    crgb = {c: torch.tensor(v, dtype=torch.float32) for c, v in _COLOR_RGB.items()}
    scale = max(1e-6, float(args.range_scale_m))
    variants = ["true_block", "true_half", "true_negyaw", "true_rotonly", "rgb_block"]
    acc = {v: [0, 0] for v in variants}
    n_examples = 0

    def steer(vec: torch.Tensor) -> int:
        return _steering_index(math.atan2(float(vec[1]), float(vec[0])))

    for f in sorted(glob.glob(args.val_glob)):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        by_seq = collections.defaultdict(list)
        for row in rows:
            by_seq[_seq_key(row)].append(row)
        for sk, rws in by_seq.items():
            rws.sort(key=lambda r: int(r.get("episode_step", 0)))
            imgs = [_load_image(Path(r["rgb_path"]), image_size=args.image_size) for r in rws]
            # per position: object_id -> (fires, area, x_centroid, landmark dict)
            obj_stat = []
            deltas = []
            for pos, row in enumerate(rws):
                lbid = _landmark_by_id(row)
                stat = {}
                for oid, lm in lbid.items():
                    color = _object_color(oid)
                    if color not in crgb or color == "unknown":
                        continue
                    a, xc = _mask_stats(
                        imgs[pos], crgb[color], args.sigma, args.threshold, args.temperature
                    )
                    stat[oid] = (a > args.area_threshold, a, xc, lm)
                obj_stat.append(stat)
                vals = [float(v) for v in row.get(args.motion_field, ())[:3]]
                while len(vals) < 3:
                    vals.append(0.0)
                deltas.append([vals[0] / scale, vals[1] / scale, vals[2]])

            for pos, row in enumerate(rws):
                lbid = _landmark_by_id(row)
                for ev in row.get("go2_causal_memory_pair_selection", ()):
                    if not str(ev.get("pair_role", "")).startswith("current_"):
                        continue
                    if not bool(ev.get("seen_before", False)):
                        continue
                    oid = str(ev.get("object_id", ""))
                    lm = lbid.get(oid)
                    if lm is None:
                        continue
                    bearing = _finite_float(lm.get("bearing_body_rad"), float("nan"))
                    if math.isnan(bearing):
                        continue
                    priors = [
                        i for i in range(pos)
                        if oid in obj_stat[i] and obj_stat[i][oid][0]
                    ]
                    if not priors:
                        continue
                    src = max(priors)
                    gap = pos - src
                    if gap < 2 or gap > args.max_gap:
                        continue
                    true_cls = _steering_index(bearing)
                    _, area, xc, src_lm = obj_stat[src][oid]
                    true_src = torch.tensor(_vector_target(src_lm, range_scale_m=scale))
                    rgb_src = _ranged_write(area, xc, scale)
                    vecs = {
                        "true_block": true_src.clone(),
                        "true_half": true_src.clone(),
                        "true_negyaw": true_src.clone(),
                        "true_rotonly": true_src.clone(),
                        "rgb_block": rgb_src.clone(),
                    }
                    for j in range(src + 1, pos + 1):
                        dx, dy, dyaw = deltas[j]
                        d_block = torch.tensor([dx, dy, dyaw])
                        d_half = torch.tensor([dx * 0.5, dy * 0.5, dyaw * 0.5])
                        d_neg = torch.tensor([dx, dy, -dyaw])
                        d_rot = torch.tensor([0.0, 0.0, dyaw])
                        vecs["true_block"] = _propagate_vectors(vecs["true_block"].reshape(1, 2), d_block)[0]
                        vecs["true_half"] = _propagate_vectors(vecs["true_half"].reshape(1, 2), d_half)[0]
                        vecs["true_negyaw"] = _propagate_vectors(vecs["true_negyaw"].reshape(1, 2), d_neg)[0]
                        vecs["true_rotonly"] = _propagate_vectors(vecs["true_rotonly"].reshape(1, 2), d_rot)[0]
                        vecs["rgb_block"] = _propagate_vectors(vecs["rgb_block"].reshape(1, 2), d_block)[0]
                    n_examples += 1
                    for v in variants:
                        acc[v][0] += int(steer(vecs[v]) == true_cls)
                        acc[v][1] += 1
                    if args.verbose:
                        cum_yaw = sum(deltas[j][2] for j in range(src + 1, pos + 1))
                        print(
                            f"gap={gap} true_cls={true_cls} "
                            f"src_true=[{float(true_src[0]):+.2f},{float(true_src[1]):+.2f}] "
                            f"qry_true_bearing={math.degrees(bearing):+.0f} cum_dyaw={math.degrees(cum_yaw):+.0f} "
                            f"| pred true_block={steer(vecs['true_block'])} "
                            f"negyaw={steer(vecs['true_negyaw'])} rot={steer(vecs['true_rotonly'])} "
                            f"rgb={steer(vecs['rgb_block'])}"
                        )

    print(f"\ngap 2-{args.max_gap} out-of-frame queries: n={n_examples}")
    for v in variants:
        c, t = acc[v]
        print(f"  {v:14s}: {c}/{t} = {c / t:.2f}" if t else f"  {v}: n=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
