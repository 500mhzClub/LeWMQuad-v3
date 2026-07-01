#!/usr/bin/env python3
"""Audit strict Go2 memory-query buckets by RGB observability.

This is a label/contract diagnostic for the unfiltered strict split. It does
not evaluate a model. It answers whether a current_* query is supported by
prior rendered RGB evidence, current geometry visibility, or neither.
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
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--area-threshold", type=float, default=0.001)
    parser.add_argument(
        "--deduplicate-like-trainer",
        action="store_true",
        help="Collapse repeated (color,target) current queries per frame like the trainer.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: int(row.get("episode_step", 0)))

    counts: Counter[str] = Counter()
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for seq_key, sequence in sorted(sequences.items()):
        prior_observed_objects: set[str] = set()
        prior_observed_colors: set[str] = set()
        for row in sequence:
            landmark_by_id = _landmark_by_id(row)
            seen_query_keys: set[tuple[str, bool]] = set()
            for event in row.get("go2_causal_memory_pair_selection", ()):
                role = str(event.get("pair_role", ""))
                if not role.startswith("current_"):
                    continue
                object_id = str(event.get("object_id", ""))
                color = _object_color(object_id)
                landmark = landmark_by_id.get(object_id)
                if landmark is None or color not in COLOR_RGB:
                    counts["drop_unknown_or_missing"] += 1
                    continue
                target = bool(event.get("seen_before", False))
                if args.deduplicate_like_trainer:
                    query_key = (color, target)
                    if query_key in seen_query_keys:
                        counts["drop_trainer_duplicate"] += 1
                        continue
                    seen_query_keys.add(query_key)
                current_visible = bool(landmark.get("visible", False))
                prior_object = object_id in prior_observed_objects
                prior_color = color in prior_observed_colors
                steering = _steering_name(float(landmark.get("bearing_body_rad", 0.0)))
                target_name = "positive" if target else "negative"
                bucket = _bucket_name(
                    target=target,
                    prior_object=prior_object,
                    prior_color=prior_color,
                    current_visible=current_visible,
                )
                counts[target_name] += 1
                counts[bucket] += 1
                bucket_counts[bucket][color] += 1
                bucket_counts[bucket][steering] += 1
                if len(examples[bucket]) < 8:
                    examples[bucket].append(
                        {
                            "seq_key": list(seq_key),
                            "episode_step": int(row.get("episode_step", 0)),
                            "object_id": object_id,
                            "color": color,
                            "steering": steering,
                            "range_m": float(landmark.get("range_m", 0.0)),
                            "bearing_body_rad": float(
                                landmark.get("bearing_body_rad", 0.0)
                            ),
                            "current_visible": current_visible,
                            "prior_object_rgb": prior_object,
                            "prior_color_rgb": prior_color,
                            "rgb_path": str(row.get("rgb_path", "")),
                        }
                    )

            for landmark in row.get("landmarks", ()):
                if not bool(landmark.get("visible", False)):
                    continue
                object_id = str(landmark.get("object_id", ""))
                color = _object_color(object_id)
                if color not in COLOR_RGB:
                    continue
                area = _rgb_color_area(
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

    report = {
        "schema": "lewm_go2_strict_memory_query_buckets_v0",
        "datasets": [str(path) for path in args.datasets],
        "sequence_count": len(sequences),
        "row_count": len(rows),
        "deduplicate_like_trainer": bool(args.deduplicate_like_trainer),
        "rgb_area_params": {
            "image_size": int(args.image_size),
            "sigma": float(args.sigma),
            "similarity_threshold": float(args.similarity_threshold),
            "temperature": float(args.temperature),
            "area_threshold": float(args.area_threshold),
        },
        "counts": dict(sorted(counts.items())),
        "bucket_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(bucket_counts.items())
        },
        "examples": dict(sorted(examples.items())),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", -1)),
        int(row.get("episode_id", -1)),
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


def _bucket_name(
    *,
    target: bool,
    prior_object: bool,
    prior_color: bool,
    current_visible: bool,
) -> str:
    prefix = "positive" if target else "negative"
    if prior_object:
        support = "prior_object_rgb"
    elif prior_color:
        support = "prior_color_rgb"
    elif current_visible:
        support = "current_visible_no_prior_rgb"
    else:
        support = "no_prior_current_hidden"
    return f"{prefix}_{support}"


def _steering_name(bearing_rad: float) -> str:
    if bearing_rad <= -0.1:
        return "right"
    if bearing_rad >= 0.1:
        return "left"
    return "forward"


def _rgb_color_area(
    path: Path,
    *,
    color: str,
    image_size: int,
    sigma: float,
    similarity_threshold: float,
    temperature: float,
) -> float:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.asarray(COLOR_RGB[color], dtype=np.float32).reshape(1, 1, 3)
    distance = ((pixels - rgb) ** 2).mean(axis=2)
    similarity = np.exp(-distance / (2.0 * float(sigma) ** 2))
    soft_mask = 1.0 / (
        1.0
        + np.exp(
            -(similarity - float(similarity_threshold)) / float(temperature)
        )
    )
    return float(soft_mask.mean())


if __name__ == "__main__":
    raise SystemExit(main())
