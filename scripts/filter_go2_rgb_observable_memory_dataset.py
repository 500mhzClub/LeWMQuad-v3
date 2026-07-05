#!/usr/bin/env python3
"""Filter Go2 memory query labels to an RGB-observable contract.

This keeps all rows/context frames intact but removes current_* memory-query
events that are not supportable from rendered RGB history:

- positive current_seen_before events require prior rendered RGB evidence for
  the same object/color;
- negative current_unseen_before events require no prior rendered RGB evidence
  for that queried color;
- optionally, current query objects must be hidden in the current frame.

The resulting dataset is still offline-supervised, but the remaining query
labels are learnable from RGB memory rather than geometry-only visibility.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COLOR_RGB = {
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--area-threshold", type=float, default=0.001)
    # Negative-cleanliness check. A negative must have no prior FULL-RGB color
    # evidence for the queried color, computed exactly as the controller computes
    # its color-mask evidence at inference (not gated on landmark visibility).
    # Defaults mirror train_go2_rgb_jepa_vector_memory_controller.py's
    # --rgb-evidence-threshold / --rgb-evidence-area-threshold so "no prior color
    # RGB" at filter time means "controller will not fire" at inference time.
    parser.add_argument("--neg-similarity-threshold", type=float, default=0.55)
    parser.add_argument("--neg-area-threshold", type=float, default=0.006)
    parser.add_argument(
        "--allow-current-visible-query",
        action="store_true",
        help="Keep current_* queries even when the queried object is visible now.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: int(row.get("episode_step", 0)))

    output_rows = []
    counts: Counter[str] = Counter()
    by_color: dict[str, Counter[str]] = defaultdict(Counter)
    for seq_key in sorted(sequences):
        prior_observed_objects: set[str] = set()
        prior_observed_colors: set[str] = set()
        prior_fullrgb_colors: set[str] = set()
        for row in sequences[seq_key]:
            landmark_by_id = _landmark_by_id(row)
            filtered_events = []
            for event in row.get("go2_causal_memory_pair_selection", ()):
                role = str(event.get("pair_role", ""))
                if not role.startswith("current_"):
                    filtered_events.append(event)
                    continue
                counts["current_event_in"] += 1
                object_id = str(event.get("object_id", ""))
                color = _object_color(object_id)
                if color not in COLOR_RGB:
                    counts["drop_unknown_color"] += 1
                    continue
                landmark = landmark_by_id.get(object_id)
                current_visible = bool(landmark.get("visible", False)) if landmark else False
                if current_visible and not bool(args.allow_current_visible_query):
                    counts["drop_current_visible"] += 1
                    by_color[color]["drop_current_visible"] += 1
                    continue
                target = bool(event.get("seen_before", False))
                if target:
                    if object_id not in prior_observed_objects:
                        counts["drop_positive_no_prior_object_rgb"] += 1
                        by_color[color]["drop_positive_no_prior_object_rgb"] += 1
                        continue
                    counts["keep_positive"] += 1
                    by_color[color]["keep_positive"] += 1
                    filtered_events.append(event)
                    continue
                if color in prior_fullrgb_colors:
                    counts["drop_negative_prior_color_rgb"] += 1
                    by_color[color]["drop_negative_prior_color_rgb"] += 1
                    continue
                counts["keep_negative"] += 1
                by_color[color]["keep_negative"] += 1
                filtered_events.append(event)

            out_row = dict(row)
            out_row["go2_causal_memory_pair_selection"] = filtered_events
            output_rows.append(out_row)

            for landmark in row.get("landmarks", ()):
                if not bool(landmark.get("visible", False)):
                    continue
                object_id = str(landmark.get("object_id", ""))
                color = _object_color(object_id)
                if color not in COLOR_RGB:
                    continue
                area, _ = _rgb_color_area(
                    Path(row["rgb_path"]),
                    color=color,
                    image_size=int(args.image_size),
                    sigma=float(args.sigma),
                    similarity_threshold=float(args.similarity_threshold),
                    temperature=float(args.temperature),
                )
                if area >= float(args.area_threshold):
                    prior_observed_objects.add(object_id)
                    prior_observed_colors.add(color)

            # Full-RGB color evidence (every color, NOT gated on landmark
            # visibility) at the controller's inference parameters. This is what
            # decides negative cleanliness: a color seen anywhere in the prior
            # frames -- even with no geometrically-visible landmark of that color
            # -- will fire the controller's color-mask gate.
            rgb_path = row.get("rgb_path")
            if rgb_path:
                for color in COLOR_RGB:
                    if color in prior_fullrgb_colors:
                        continue
                    area, _ = _rgb_color_area(
                        Path(rgb_path),
                        color=color,
                        image_size=int(args.image_size),
                        sigma=float(args.sigma),
                        similarity_threshold=float(args.neg_similarity_threshold),
                        temperature=float(args.temperature),
                    )
                    if area >= float(args.neg_area_threshold):
                        prior_fullrgb_colors.add(color)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    summary = {
        "schema": "lewm_go2_rgb_observable_memory_filter_v0",
        "datasets": [str(path) for path in args.datasets],
        "out": str(args.out),
        "sequence_count": len(sequences),
        "row_count": len(output_rows),
        "params": {
            "image_size": int(args.image_size),
            "sigma": float(args.sigma),
            "similarity_threshold": float(args.similarity_threshold),
            "temperature": float(args.temperature),
            "area_threshold": float(args.area_threshold),
            "allow_current_visible_query": bool(args.allow_current_visible_query),
        },
        "counts": dict(sorted(counts.items())),
        "by_color": {key: dict(sorted(value.items())) for key, value in sorted(by_color.items())},
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): landmark
        for landmark in row.get("landmarks", ())
        if str(landmark.get("object_id", ""))
    }


def _object_color(object_id: str) -> str:
    lowered = str(object_id).lower()
    for color in COLOR_RGB:
        if color in lowered:
            return color
    return "unknown"


def _rgb_color_area(
    path: Path,
    *,
    color: str,
    image_size: int,
    sigma: float,
    similarity_threshold: float,
    temperature: float,
) -> tuple[float, float]:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.asarray(COLOR_RGB[color], dtype=np.float32).reshape(1, 1, 3)
    distance = ((pixels - rgb) ** 2).mean(axis=2)
    similarity = np.exp(-distance / (2.0 * float(sigma) ** 2))
    soft_mask = 1.0 / (
        1.0 + np.exp(-(similarity - float(similarity_threshold)) / float(temperature))
    )
    return float(soft_mask.mean()), float(similarity.max())


if __name__ == "__main__":
    raise SystemExit(main())
