#!/usr/bin/env python3
"""Calibrate the Go2 RGB color-mask write against true landmark geometry.

Decides whether the explicit-memory steering gap to the 2D demo is a missing
*range* / mis-calibrated *bearing* in the write-time vector, rather than a
metric-poor JEPA latent.

For every landmark observation whose soft color mask actually fires in the RGB
frame (area > area_threshold, i.e. the landmark is inside the forward camera
cone), we measure how well the mask centroid predicts true ego bearing and how
well the mask area predicts true range. We also report the observation-cone
width and the query-time bearing distribution, because `visible` in this dataset
means line-of-sight (median |bearing| ~63deg), not in-frame.

The write currently used by the controller is the fixed-range proxy
``[forward=0.75, lateral=-x_centroid]`` from ``_rgb_color_readout``; this probe
quantifies the error that proxy introduces and what a fitted calibration buys.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    _COLOR_RGB,
    _finite_float,
    _object_color,
    _seq_key,
    _steering_index,
)
from train_go2_hidden_target_memory_probe import _load_image  # noqa: E402


def _soft_mask_stats(
    image: torch.Tensor,
    color_rgb: torch.Tensor,
    *,
    sigma: float,
    threshold: float,
    temperature: float,
    value_normalized: bool = False,
    value_norm_floor: float = 0.15,
) -> tuple[float, float]:
    """Replicates ColorVectorMemoryController._rgb_color_readout for one color."""
    if value_normalized:
        # Value-normalize each pixel (divide by its max channel) and compare hue to
        # the (value-normalized) pure color: a desaturated/shadowed target still fires
        # while a near-gray background tint normalizes far from the hue and is rejected.
        mx = image.amax(dim=0, keepdim=True).clamp_min(value_norm_floor)
        norm = image / mx
        cn = color_rgb / float(color_rgb.max())
        distance = ((norm - cn.reshape(3, 1, 1)) ** 2).mean(dim=0)
    else:
        distance = ((image - color_rgb.reshape(3, 1, 1)) ** 2).mean(dim=0)
    similarity = torch.exp(-distance / (2.0 * sigma**2))
    soft_mask = torch.sigmoid((similarity - threshold) / temperature)
    area = float(soft_mask.mean().clamp_min(1e-8))
    width = soft_mask.shape[-1]
    x_coords = torch.linspace(-1.0, 1.0, width)
    denom = soft_mask.sum().clamp_min(1e-6)
    x_centroid = float((soft_mask * x_coords.reshape(1, width)).sum() / denom)
    return area, x_centroid


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """y ~= a*x + b. Returns a, b, R^2, residual std."""
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    resid_std = float(np.sqrt(((y - pred) ** 2).mean()))
    return a, b, r2, resid_std


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--rgb-evidence-sigma", type=float, default=0.20)
    parser.add_argument("--rgb-evidence-threshold", type=float, default=0.55)
    parser.add_argument("--rgb-evidence-temperature", type=float, default=0.08)
    parser.add_argument("--rgb-evidence-area-threshold", type=float, default=0.006)
    parser.add_argument("--rgb-evidence-value-normalized", action="store_true",
                        help="Value-normalize each pixel (divide by its max channel) before "
                             "the Euclidean readout so it compares hue, not brightness.")
    parser.add_argument("--value-norm-floor", type=float, default=0.15)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]

    color_rgb = {c: torch.tensor(v, dtype=torch.float32) for c, v in _COLOR_RGB.items()}

    # Per-observation records of every visible landmark.
    records: list[dict[str, Any]] = []
    # Session firing history for propagation feasibility: seq_key -> color -> [steps fired]
    session_fire_steps: dict[Any, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    image_cache: dict[str, torch.Tensor] = {}

    for row in rows:
        rgb_path = row.get("rgb_path")
        if not rgb_path:
            continue
        if rgb_path not in image_cache:
            image_cache[rgb_path] = _load_image(Path(rgb_path), image_size=args.image_size)
            if len(image_cache) > 4096:
                image_cache.clear()
                image_cache[rgb_path] = _load_image(Path(rgb_path), image_size=args.image_size)
        image = image_cache[rgb_path]
        seq_key = _seq_key(row)
        step = int(row.get("episode_step", 0))
        scene = str(row.get("scene_id", ""))
        for landmark in row.get("landmarks", ()):
            color = _object_color(str(landmark.get("object_id", "")))
            if color not in color_rgb or color == "unknown":
                continue
            if not bool(landmark.get("visible", False)):
                continue
            bearing = _finite_float(landmark.get("bearing_body_rad"), float("nan"))
            range_m = _finite_float(landmark.get("range_m"), float("nan"))
            if math.isnan(bearing) or math.isnan(range_m):
                continue
            area, x_centroid = _soft_mask_stats(
                image,
                color_rgb[color],
                sigma=args.rgb_evidence_sigma,
                threshold=args.rgb_evidence_threshold,
                temperature=args.rgb_evidence_temperature,
                value_normalized=bool(args.rgb_evidence_value_normalized),
                value_norm_floor=float(args.value_norm_floor),
            )
            fires = area > args.rgb_evidence_area_threshold
            if fires:
                session_fire_steps[seq_key][color].append(step)
            records.append(
                {
                    "scene": scene,
                    "color": color,
                    "bearing": bearing,
                    "range_m": range_m,
                    "area": area,
                    "x_centroid": x_centroid,
                    "fires": fires,
                }
            )

    n_visible = len(records)
    fired = [r for r in records if r["fires"]]
    n_fired = len(fired)

    def _block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        if not recs:
            return {"n": 0}
        bearings = np.array([r["bearing"] for r in recs])
        ranges = np.array([r["range_m"] for r in recs])
        xc = np.array([r["x_centroid"] for r in recs])
        areas = np.array([r["area"] for r in recs])
        abs_bear_deg = [abs(math.degrees(b)) for b in bearings]

        # bearing calibration: bearing ~= a*x_centroid + b
        a, b, r2, resid = _fit_linear(xc, bearings)
        # current crude write bearing vs calibrated, scored as 3-class steering
        true_cls = np.array([_steering_index(float(b_)) for b_ in bearings])
        crude_cls = np.array(
            [_steering_index(math.atan2(-float(np.clip(x, -1, 1)), 0.75)) for x in xc]
        )
        calib_cls = np.array([_steering_index(a * float(x) + b) for x in xc])
        crude_acc = float((crude_cls == true_cls).mean())
        calib_acc = float((calib_cls == true_cls).mean())
        # sign agreement (left/right ignoring forward deadzone)
        nonfwd = true_cls != 1
        crude_sign = (
            float((crude_cls[nonfwd] == true_cls[nonfwd]).mean()) if nonfwd.any() else float("nan")
        )

        # range from area: log(range) ~= m*log(area) + c   (inverse-square ~ -0.5)
        valid = (areas > 0) & (ranges > 0)
        m = c = rr2 = range_mae = float("nan")
        if valid.sum() >= 3:
            la = np.log(areas[valid])
            lr = np.log(ranges[valid])
            m, c, rr2, _ = _fit_linear(la, lr)
            range_pred = np.exp(m * la + c)
            range_mae = float(np.abs(range_pred - ranges[valid]).mean())

        return {
            "n": len(recs),
            "obs_cone_abs_bearing_deg": {
                "median": float(np.median(abs_bear_deg)),
                "p90": _pct(abs_bear_deg, 0.9),
                "max": float(max(abs_bear_deg)),
                "frac_within_45deg": float(np.mean([d < 45 for d in abs_bear_deg])),
            },
            "bearing_fit": {"a": a, "b": b, "r2": r2, "resid_rad": resid},
            "steering_3class_acc": {"crude_write": crude_acc, "calibrated": calib_acc},
            "leftright_sign_acc_crude": crude_sign,
            "range_fit_loglog": {"slope": m, "intercept": c, "r2": rr2, "range_mae_m": range_mae},
        }

    # propagation feasibility: of fired observations, the within-session
    # firing density tells whether a wide-bearing query can be reached by
    # dead-reckoning a prior in-frame write.
    fire_gaps = []
    for _, by_color in session_fire_steps.items():
        for _, steps in by_color.items():
            steps = sorted(steps)
            for i in range(1, len(steps)):
                fire_gaps.append(steps[i] - steps[i - 1])

    per_scene = {}
    by_scene: dict[str, list] = defaultdict(list)
    for r in fired:
        by_scene[r["scene"]].append(r)
    for scene, recs in sorted(by_scene.items()):
        per_scene[scene] = _block(recs)

    per_color = {}
    by_color: dict[str, list] = defaultdict(list)
    for r in fired:
        by_color[r["color"]].append(r)
    for color, recs in sorted(by_color.items()):
        per_color[color] = _block(recs)

    report = {
        "dataset": str(args.dataset),
        "mask_params": {
            "sigma": args.rgb_evidence_sigma,
            "threshold": args.rgb_evidence_threshold,
            "temperature": args.rgb_evidence_temperature,
            "area_threshold": args.rgb_evidence_area_threshold,
        },
        "n_visible_landmark_obs": n_visible,
        "n_fired_in_frame": n_fired,
        "fire_rate_of_visible": float(n_fired / max(1, n_visible)),
        "visible_obs_cone_abs_bearing_deg": {
            "median": float(np.median([abs(math.degrees(r["bearing"])) for r in records]))
            if records
            else float("nan"),
        },
        "within_session_fire_gap_steps": {
            "n": len(fire_gaps),
            "median": float(np.median(fire_gaps)) if fire_gaps else float("nan"),
            "p90": _pct([float(g) for g in fire_gaps], 0.9),
        },
        "pooled_fired": _block(fired),
        "per_scene_fired": per_scene,
        "per_color_fired": per_color,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
