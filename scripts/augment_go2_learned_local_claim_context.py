#!/usr/bin/env python3
"""Augment Go2 learned-local rows with synthetic claimed-color context.

This is a training-time domain-randomization utility. It does not add new
runtime features; it only edits the explicit claimed-color mask that already
exists in the learned-local feature vector and mirrors that change in metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dim", type=int, default=1600)
    parser.add_argument(
        "--color-vocab",
        default="blue,green,red,yellow",
        help=(
            "Color order used by the learned-local feature vector. The current "
            "Go2 vector-memory controller checkpoint stores blue,green,red,yellow."
        ),
    )
    parser.add_argument(
        "--claimed-colors",
        action="append",
        required=True,
        help="Comma-separated claimed colors for one augmented copy. Repeat for variants.",
    )
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()

    color_vocab = [item.strip() for item in str(args.color_vocab).split(",") if item.strip()]
    if not color_vocab:
        raise SystemExit("--color-vocab must not be empty")
    claim_variants = [
        [item.strip() for item in str(raw).split(",") if item.strip()]
        for raw in args.claimed_colors
    ]
    known = set(color_vocab)
    for variant in claim_variants:
        unknown = [color for color in variant if color not in known]
        if unknown:
            raise SystemExit(f"unknown claimed color(s): {unknown}")

    row_arrays: dict[str, list[np.ndarray]] = {}
    static_arrays: dict[str, np.ndarray] = {}
    reports: list[dict[str, Any]] = []
    feature_variant = "base"
    source_rows = 0
    output_rows = 0

    for input_path in args.input:
        with np.load(input_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            meta = [json.loads(str(item)) for item in np.asarray(data["meta_json"]).tolist()]
            if features.ndim != 2:
                raise SystemExit(f"{input_path} features must be rank-2")
            if features.shape[0] != labels.shape[0] or labels.shape[0] != len(meta):
                raise SystemExit(f"{input_path} features/labels/meta row counts do not match")
            if features.shape[1] < int(args.base_dim):
                raise SystemExit(
                    f"{input_path} feature_dim={features.shape[1]} < base_dim={args.base_dim}"
                )
            source_rows += int(labels.shape[0])

            chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            if bool(args.include_original):
                chunks.append((features, labels, np.asarray([json.dumps(m, sort_keys=True) for m in meta])))
            for variant in claim_variants:
                aug_features = features.copy()
                claim_mask = np.asarray(
                    [1.0 if color in set(variant) else 0.0 for color in color_vocab],
                    dtype=np.float32,
                )
                claimed_start = int(args.base_dim) - (len(color_vocab) + 8 + 24)
                claimed_end = claimed_start + len(color_vocab)
                aug_features[:, claimed_start:claimed_end] = claim_mask.reshape(1, -1)
                aug_meta = []
                for row in meta:
                    out = dict(row)
                    out["claimed_count"] = int(len(variant))
                    out["claimed_colors_augmented"] = list(variant)
                    out["claim_context_augmentation"] = True
                    aug_meta.append(json.dumps(out, sort_keys=True))
                chunks.append((aug_features, labels, np.asarray(aug_meta)))

            for feat_chunk, label_chunk, meta_chunk in chunks:
                row_arrays.setdefault("features", []).append(feat_chunk)
                row_arrays.setdefault("labels", []).append(label_chunk)
                row_arrays.setdefault("meta_json", []).append(meta_chunk)
                output_rows += int(label_chunk.shape[0])

            for key in data.files:
                if key in {"features", "labels", "meta_json", "result_json", "filter_report_json", "relabel_report_json"}:
                    continue
                value = np.asarray(data[key])
                if key not in static_arrays:
                    static_arrays[key] = value
                elif not np.array_equal(static_arrays[key], value):
                    raise SystemExit(f"static array mismatch for key {key!r} in {input_path}")
            source_report = _source_report(data, input_path)
            reports.append(source_report)
            if str(source_report.get("feature_variant", "base")) != "base":
                feature_variant = str(source_report.get("feature_variant"))

    arrays = dict(static_arrays)
    for key, chunks in row_arrays.items():
        arrays[key] = np.concatenate(chunks, axis=0)
    report = {
        "schema": "lewm_go2_claim_context_augmentation_v0",
        "inputs": [str(path) for path in args.input],
        "source_rows": int(source_rows),
        "output_rows": int(output_rows),
        "include_original": bool(args.include_original),
        "base_dim": int(args.base_dim),
        "color_vocab": list(color_vocab),
        "claimed_color_variants": claim_variants,
        "sources": reports,
        "wall_metrics": {
            "learned_local_policy_feature_variant": feature_variant,
            "claim_context_augmentation": True,
        },
    }
    arrays["result_json"] = np.asarray([json.dumps(report, sort_keys=True)])
    arrays["claim_context_augmentation_report_json"] = np.asarray(
        [json.dumps(report, sort_keys=True)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _source_report(data: np.lib.npyio.NpzFile, input_path: Path) -> dict[str, Any]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    feature_variant = "base"
    if "result_json" in data and len(data["result_json"]) > 0:
        try:
            result = json.loads(str(data["result_json"][0]))
            metrics = result.get("wall_metrics", {})
            if isinstance(metrics, dict):
                feature_variant = str(
                    metrics.get("learned_local_policy_feature_variant", feature_variant)
                )
        except json.JSONDecodeError:
            pass
    return {
        "path": str(input_path),
        "rows": int(labels.shape[0]),
        "feature_dim": int(np.asarray(data["features"]).shape[1]),
        "feature_variant": feature_variant,
    }


if __name__ == "__main__":
    raise SystemExit(main())
