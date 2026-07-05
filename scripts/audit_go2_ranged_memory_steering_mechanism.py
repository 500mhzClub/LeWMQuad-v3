#!/usr/bin/env python3
"""Deterministic proof that a ranged RGB memory + odometry propagation recovers
out-of-frame steering at short horizons (the 2D claim-task regime).

No training, no learned head. For each scored steering query (a `current_*`
event with `seen_before=True`), we find the most recent prior sequence position
where that color's soft mask fired in-frame, write a *calibrated ranged* relative
position there ([range*cos(bearing), range*sin(bearing)] / range_scale, bearing
from the mask centroid, range from the mask area), propagate it forward through
the recorded per-step body odometry to the query position with the controller's
own ``_propagate_vectors``, and read the steering class from the propagated
vector. This isolates the explicit-memory mechanism from the controller's
learned head (which the steering investigation found collapses to the per-scene
majority class because the landmark is out of frame at query time).

Reported per position-gap bucket and compared against:
  * the crude fixed-range write ([0.75, -x_centroid]) the controller ships today;
  * rotation-only propagation (ignore translation) to isolate the range term;
  * the per-scene majority-class baseline.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

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

# Pooled fits from audit_go2_rgb_bearing_range_calibration.py on broad_clean.jsonl.
BEARING_A = -0.7412162764485124
BEARING_B = 0.01266205992118909
RANGE_LOGLOG_M = -0.25799125815180496
RANGE_LOGLOG_C = -0.7229343594424763


def _mask_stats(image: torch.Tensor, color_rgb: torch.Tensor, sigma, threshold, temperature):
    distance = ((image - color_rgb.reshape(3, 1, 1)) ** 2).mean(dim=0)
    similarity = torch.exp(-distance / (2.0 * sigma**2))
    soft_mask = torch.sigmoid((similarity - threshold) / temperature)
    area = float(soft_mask.mean().clamp_min(1e-8))
    width = soft_mask.shape[-1]
    x_coords = torch.linspace(-1.0, 1.0, width)
    denom = soft_mask.sum().clamp_min(1e-6)
    x_centroid = float((soft_mask * x_coords.reshape(1, width)).sum() / denom)
    return area, x_centroid


def _ranged_write(area: float, x_centroid: float, range_scale_m: float) -> torch.Tensor:
    bearing = BEARING_A * x_centroid + BEARING_B
    range_m = math.exp(RANGE_LOGLOG_M * math.log(max(area, 1e-8)) + RANGE_LOGLOG_C)
    range_m = max(0.0, min(range_scale_m, range_m))
    r = range_m / range_scale_m
    return torch.tensor([r * math.cos(bearing), r * math.sin(bearing)], dtype=torch.float32)


def _crude_write(area: float, x_centroid: float) -> torch.Tensor:
    # the controller's _rgb_color_readout, with --rgb-vector-scale 2 + tanh.
    vec = torch.tensor([0.75, -max(-1.0, min(1.0, x_centroid))], dtype=torch.float32)
    return torch.tanh(2.0 * vec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-glob",
        default=".generated/go2_hidden_target_memory/observed_memory_gate_20260622/cv/val_holdout_*.jsonl",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--rgb-evidence-sigma", type=float, default=0.20)
    parser.add_argument("--rgb-evidence-threshold", type=float, default=0.55)
    parser.add_argument("--rgb-evidence-temperature", type=float, default=0.08)
    parser.add_argument("--rgb-evidence-area-threshold", type=float, default=0.006)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--motion-field", default="integrated_body_motion_block")
    parser.add_argument("--gap-buckets", default="1,2,4,8,16,1000")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    crgb = {c: torch.tensor(v, dtype=torch.float32) for c, v in _COLOR_RGB.items()}
    scale = max(1e-6, float(args.range_scale_m))
    buckets = [int(x) for x in args.gap_buckets.split(",")]

    # variant -> gap_bucket -> [correct(0/1)], plus sign-only and majority
    per_variant = {
        v: collections.defaultdict(list)
        for v in ("ranged", "crude", "ranged_rot_only")
    }
    majority = collections.defaultdict(list)
    per_scene = collections.defaultdict(lambda: collections.defaultdict(list))
    noprior = 0
    total = 0

    for f in sorted(glob.glob(args.val_glob)):
        scene = Path(f).name.replace("val_holdout_", "").replace(".jsonl", "")
        rows = [json.loads(l) for l in open(f) if l.strip()]
        by_seq = collections.defaultdict(list)
        for row in rows:
            by_seq[_seq_key(row)].append(row)
        # per-scene true class counts for the majority baseline
        scene_classes = []
        for sk, rws in by_seq.items():
            rws.sort(key=lambda r: int(r.get("episode_step", 0)))
            imgs = [_load_image(Path(r["rgb_path"]), image_size=args.image_size) for r in rws]
            # per position, per color: (fires, area, x_centroid)
            stats = []
            for img in imgs:
                pos_stat = {}
                for c in crgb:
                    if c == "unknown":
                        continue
                    a, xc = _mask_stats(
                        img, crgb[c], args.rgb_evidence_sigma,
                        args.rgb_evidence_threshold, args.rgb_evidence_temperature,
                    )
                    pos_stat[c] = (a > args.rgb_evidence_area_threshold, a, xc)
                stats.append(pos_stat)
            # per-position odometry delta [dx/scale, dy/scale, dyaw]
            deltas = []
            for row in rws:
                vals = [float(v) for v in row.get(args.motion_field, ())[:3]]
                while len(vals) < 3:
                    vals.append(0.0)
                deltas.append(torch.tensor([vals[0] / scale, vals[1] / scale, vals[2]]))

            for pos, row in enumerate(rws):
                lbid = _landmark_by_id(row)
                for ev in row.get("go2_causal_memory_pair_selection", ()):
                    if not str(ev.get("pair_role", "")).startswith("current_"):
                        continue
                    if not bool(ev.get("seen_before", False)):
                        continue
                    oid = str(ev.get("object_id", ""))
                    color = _object_color(oid)
                    lm = lbid.get(oid)
                    if lm is None or color not in crgb:
                        continue
                    bearing = _finite_float(lm.get("bearing_body_rad"), float("nan"))
                    if math.isnan(bearing):
                        continue
                    true_cls = _steering_index(bearing)
                    total += 1
                    scene_classes.append(true_cls)
                    priors = [i for i in range(pos) if stats[i][color][0]]
                    if not priors:
                        noprior += 1
                        continue
                    src = max(priors)
                    gap = pos - src
                    _, area, xc = stats[src][color]
                    ranged = _ranged_write(area, xc, scale)
                    crude = _crude_write(area, xc)
                    rot_only = ranged.clone()
                    vecs = {"ranged": ranged, "crude": crude, "ranged_rot_only": rot_only}
                    for j in range(src + 1, pos + 1):
                        d = deltas[j]
                        for key in vecs:
                            if key == "ranged_rot_only":
                                drot = torch.tensor([0.0, 0.0, d[2]])
                                vecs[key] = _propagate_vectors(vecs[key].reshape(1, 2), drot)[0]
                            else:
                                vecs[key] = _propagate_vectors(vecs[key].reshape(1, 2), d)[0]
                    b = next(bk for bk in buckets if gap <= bk)
                    for key, vec in vecs.items():
                        pred = _steering_index(math.atan2(float(vec[1]), float(vec[0])))
                        per_variant[key][b].append(int(pred == true_cls))
                    per_scene[scene][b].append(
                        int(
                            _steering_index(
                                math.atan2(float(vecs["ranged"][1]), float(vecs["ranged"][0]))
                            )
                            == true_cls
                        )
                    )
        # majority baseline: predict per-scene most-common class
        if scene_classes:
            cnt = collections.Counter(scene_classes)
            maj = cnt.most_common(1)[0][0]
            for c in scene_classes:
                majority[scene].append(int(c == maj))

    def _summ(d):
        return {
            str(b): {"n": len(v), "acc": (float(np.mean(v)) if v else float("nan"))}
            for b, v in sorted(d.items())
        }

    report = {
        "total_steering_queries": total,
        "no_prior_incone_sighting": noprior,
        "motion_field": args.motion_field,
        "fits": {"bearing_a": BEARING_A, "bearing_b": BEARING_B,
                 "range_loglog_m": RANGE_LOGLOG_M, "range_loglog_c": RANGE_LOGLOG_C},
        "accuracy_by_gap": {v: _summ(per_variant[v]) for v in per_variant},
        "per_scene_ranged_by_gap": {s: _summ(d) for s, d in per_scene.items()},
        "majority_class_acc": {s: float(np.mean(v)) for s, v in majority.items()},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
