#!/usr/bin/env python3
"""Audit whether Go2 hidden-target memory labels are observable in rendered RGB.

The pure RGB/JEPA memory contract requires a positive query to have actual
rendered evidence in the preceding context. Geometry labels can mark a landmark
visible even when the camera image has no useful target-color pixels; those
cases are not learnable from RGB memory alone.
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
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--area-threshold", type=float, default=0.001)
    parser.add_argument("--report-output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: int(row.get("episode_step", 0)))

    counts: Counter[str] = Counter()
    by_color: dict[str, Counter[str]] = defaultdict(Counter)
    missing_examples = []
    observed_examples = []
    for seq_key, sequence in sequences.items():
        prior_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        prior_by_color: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sequence:
            for color, target in _queries(row):
                bucket = "positive" if target else "negative"
                counts[f"{bucket}_query"] += 1
                by_color[color][f"{bucket}_query"] += 1
                if target:
                    object_id = _query_object_id(row, color)
                    object_evidence = prior_by_object.get(object_id, [])
                    color_evidence = prior_by_color.get(color, [])
                    max_object_area = max(
                        (float(item["area"]) for item in object_evidence),
                        default=0.0,
                    )
                    max_color_area = max(
                        (float(item["area"]) for item in color_evidence),
                        default=0.0,
                    )
                    observed = max_object_area >= float(args.area_threshold)
                    counts["positive_with_rendered_object_color_evidence"] += int(observed)
                    counts["positive_without_rendered_object_color_evidence"] += int(
                        not observed
                    )
                    by_color[color]["positive_with_rendered_object_color_evidence"] += int(
                        observed
                    )
                    by_color[color][
                        "positive_without_rendered_object_color_evidence"
                    ] += int(not observed)
                    example = {
                        "seq_key": list(seq_key),
                        "episode_step": int(row.get("episode_step", 0)),
                        "object_id": object_id,
                        "color": color,
                        "prior_object_label_visible_count": len(object_evidence),
                        "prior_color_label_visible_count": len(color_evidence),
                        "max_object_rgb_area": max_object_area,
                        "max_color_rgb_area": max_color_area,
                        "query_rgb_path": str(row.get("rgb_path", "")),
                        "prior_examples": object_evidence[:2] + object_evidence[-2:],
                    }
                    if observed and len(observed_examples) < 10:
                        observed_examples.append(example)
                    if not observed and len(missing_examples) < 20:
                        missing_examples.append(example)

            for landmark in row.get("landmarks", ()):
                if not bool(landmark.get("visible", False)):
                    continue
                object_id = str(landmark.get("object_id", ""))
                color = _object_color(object_id)
                if color not in COLOR_RGB:
                    continue
                area, max_similarity = _rgb_color_area(
                    Path(row["rgb_path"]),
                    color=color,
                    image_size=int(args.image_size),
                    sigma=float(args.sigma),
                    similarity_threshold=float(args.similarity_threshold),
                    temperature=float(args.temperature),
                )
                evidence = {
                    "episode_step": int(row.get("episode_step", 0)),
                    "rgb_path": str(row.get("rgb_path", "")),
                    "area": float(area),
                    "max_similarity": float(max_similarity),
                }
                prior_by_object[object_id].append(evidence)
                prior_by_color[color].append(evidence)

    report = {
        "schema": "lewm_go2_rgb_memory_observability_audit_v0",
        "datasets": [str(path) for path in args.datasets],
        "sequence_count": len(sequences),
        "row_count": len(rows),
        "color_area_params": {
            "image_size": int(args.image_size),
            "sigma": float(args.sigma),
            "similarity_threshold": float(args.similarity_threshold),
            "temperature": float(args.temperature),
            "area_threshold": float(args.area_threshold),
        },
        "counts": dict(sorted(counts.items())),
        "by_color": {color: dict(sorted(counter.items())) for color, counter in by_color.items()},
        "missing_positive_examples": missing_examples,
        "observed_positive_examples": observed_examples,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(text, encoding="utf-8")
    print(text, end="")
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


def _queries(row: dict[str, Any]) -> list[tuple[str, bool]]:
    result = []
    seen = set()
    for event in row.get("go2_causal_memory_pair_selection", ()):
        if not str(event.get("pair_role", "")).startswith("current_"):
            continue
        color = _object_color(str(event.get("object_id", "")))
        if color not in COLOR_RGB:
            continue
        target = bool(event.get("seen_before", False))
        key = (color, target)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _query_object_id(row: dict[str, Any], color: str) -> str:
    for event in row.get("go2_causal_memory_pair_selection", ()):
        if not str(event.get("pair_role", "")).startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        if _object_color(object_id) == color and bool(event.get("seen_before", False)):
            return object_id
    return ""


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
