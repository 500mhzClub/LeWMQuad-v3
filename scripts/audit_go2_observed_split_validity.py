#!/usr/bin/env python3
"""Pre-training validity guard for a Go2 observed-memory split.

A held-out observed-memory split is only a *valid* clean test if:

1. it has enough observed-then-hidden positives per color, and
2. its kept negatives are genuinely no-prior under the SAME RGB-evidence
   parameters the controller uses at inference.

Point 2 is the failure that invalidated the `04f670` (yellow) held-out test: the
observable filter
(`scripts/filter_go2_rgb_observable_memory_dataset.py`) dropped negatives whose
queried color had prior RGB at sim 0.65 / area 0.001, but the controller
(`scripts/train_go2_rgb_jepa_vector_memory_controller.py`) fires its color-mask
evidence at sim 0.55 / area 0.006 — so a "no-prior" yellow negative in a
yellow-saturated scene still triggers a false claim. This audit recomputes
prior color RGB evidence under the *controller* parameters (defaults below) and
flags scenes/colors where the negatives are contaminated.

Reuses the RGB-area logic from `scripts/audit_go2_rgb_memory_observability.py`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Defaults mirror the controller's RGB-evidence inference parameters
# (train_go2_rgb_jepa_vector_memory_controller.py:
#  --rgb-evidence-threshold 0.55, --rgb-evidence-area-threshold 0.006,
#  --rgb-evidence-sigma 0.20, --rgb-evidence-temperature 0.08).
COLOR_RGB = {
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--label", type=str, default="split")
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--area-threshold", type=float, default=0.006)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument(
        "--min-observed-positives", type=int, default=10,
        help="Per-scene minimum observed-then-hidden positives to be a valid test.")
    parser.add_argument(
        "--max-contaminated-negative-rate", type=float, default=0.05,
        help="Per-(scene,color) max fraction of kept negatives that fire the gate.")
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)
    for seq in sequences.values():
        seq.sort(key=lambda r: int(r.get("episode_step", 0)))

    # per (scene, color) tallies
    sc: dict[tuple[str, str], Counter] = defaultdict(Counter)
    img_cache: dict[tuple[str, str], float] = {}

    def area_for(path: str, color: str) -> float:
        key = (path, color)
        if key not in img_cache:
            img_cache[key], _ = _rgb_color_area(
                Path(path), color=color, image_size=int(args.image_size),
                sigma=float(args.sigma), similarity_threshold=float(args.similarity_threshold),
                temperature=float(args.temperature))
        return img_cache[key]

    for seq in sequences.values():
        # Faithful to the controller: color evidence is the per-frame full-RGB
        # color-mask area (NOT gated on landmark visibility), accumulated over
        # prior frames. `prior_col_area` = max color-mask area over frames before
        # the current one; that is what the controller's read gate sees.
        prior_col_area: dict[str, float] = defaultdict(float)
        for row in seq:
            scene = str(row.get("scene_id", ""))
            for color, target, object_id in _queries(row):
                key = (scene, color)
                fired = prior_col_area.get(color, 0.0) >= float(args.area_threshold)
                if target:
                    sc[key]["positives"] += 1
                    sc[key]["observed_positives"] += int(fired)
                else:
                    sc[key]["negatives"] += 1
                    sc[key]["contaminated_negatives"] += int(fired)
            # accumulate prior RGB evidence from the full frame, every color
            rgb_path = str(row.get("rgb_path", ""))
            if rgb_path:
                for color in COLOR_RGB:
                    a = area_for(rgb_path, color)
                    if a > prior_col_area[color]:
                        prior_col_area[color] = a

    # aggregate per scene
    per_scene_color = []
    scene_obs: Counter = Counter()
    scene_contam_flag: dict[str, list[str]] = defaultdict(list)
    for (scene, color), c in sorted(sc.items()):
        neg = c["negatives"]
        contam_rate = c["contaminated_negatives"] / neg if neg else 0.0
        per_scene_color.append({
            "scene_id": scene, "color": color,
            "positives": c["positives"], "observed_positives": c["observed_positives"],
            "negatives": neg, "contaminated_negatives": c["contaminated_negatives"],
            "contaminated_negative_rate": round(contam_rate, 4),
        })
        scene_obs[scene] += c["observed_positives"]
        if neg and contam_rate > float(args.max_contaminated_negative_rate):
            scene_contam_flag[scene].append(f"{color}({contam_rate:.2f})")

    per_scene = []
    for scene in sorted({k[0] for k in sc}):
        obs = scene_obs[scene]
        flags = scene_contam_flag.get(scene, [])
        valid = obs >= int(args.min_observed_positives) and not flags
        per_scene.append({
            "scene_id": scene, "observed_positives": obs,
            "contaminated_colors": flags, "valid": bool(valid),
        })

    report = {
        "schema": "lewm_go2_observed_split_validity_v0",
        "label": args.label,
        "datasets": [str(p) for p in args.datasets],
        "params": {
            "image_size": int(args.image_size), "sigma": float(args.sigma),
            "similarity_threshold": float(args.similarity_threshold),
            "area_threshold": float(args.area_threshold),
            "temperature": float(args.temperature),
            "min_observed_positives": int(args.min_observed_positives),
            "max_contaminated_negative_rate": float(args.max_contaminated_negative_rate),
        },
        "per_scene": per_scene,
        "per_scene_color": per_scene_color,
    }
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[{args.label}] params: sim={args.similarity_threshold} area={args.area_threshold}")
    for s in per_scene:
        verdict = "VALID" if s["valid"] else "INVALID"
        flags = (" contaminated=" + ",".join(s["contaminated_colors"])) if s["contaminated_colors"] else ""
        print(f"  {verdict:>7}  {s['scene_id'][-12:]:>12}  observed_pos={s['observed_positives']:>3}{flags}")
    if args.report_output is not None:
        print(f"  report -> {args.report_output}")
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id", "")), int(row.get("env_idx", 0)), int(row.get("episode_id", 0)))


def _queries(row: dict[str, Any]) -> list[tuple[str, bool, str]]:
    result = []
    seen = set()
    for event in row.get("go2_causal_memory_pair_selection", ()):
        if not str(event.get("pair_role", "")).startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        color = _object_color(object_id)
        if color not in COLOR_RGB:
            continue
        target = bool(event.get("seen_before", False))
        key = (color, target)
        if key in seen:
            continue
        seen.add(key)
        result.append((color, target, object_id))
    return result


def _object_color(object_id: str) -> str:
    lowered = str(object_id).lower()
    for color in COLOR_RGB:
        if color in lowered:
            return color
    return "unknown"


def _rgb_color_area(path: Path, *, color: str, image_size: int, sigma: float,
                    similarity_threshold: float, temperature: float) -> tuple[float, float]:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.asarray(COLOR_RGB[color], dtype=np.float32).reshape(1, 1, 3)
    distance = ((pixels - rgb) ** 2).mean(axis=2)
    similarity = np.exp(-distance / (2.0 * float(sigma) ** 2))
    soft_mask = 1.0 / (1.0 + np.exp(-(similarity - float(similarity_threshold)) / float(temperature)))
    return float(soft_mask.mean()), float(similarity.max())


if __name__ == "__main__":
    raise SystemExit(main())
