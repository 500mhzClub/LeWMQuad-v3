#!/usr/bin/env python3
"""Filter a Go2 learned-local policy dataset without changing its schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--input-list",
        action="append",
        type=Path,
        default=[],
        help="Optional newline-delimited file of dataset paths to filter.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-states", default="")
    parser.add_argument(
        "--include-labels",
        default="",
        help="Optional comma-separated primitive labels to keep.",
    )
    parser.add_argument(
        "--exclude-labels",
        default="",
        help="Optional comma-separated primitive labels to drop.",
    )
    parser.add_argument("--require-meta-bool", action="append", default=[])
    parser.add_argument(
        "--meta-min",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose numeric metadata KEY is at least VALUE.",
    )
    parser.add_argument(
        "--meta-max",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose numeric metadata KEY is at most VALUE.",
    )
    parser.add_argument(
        "--meta-abs-max",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose absolute numeric metadata KEY is at most VALUE.",
    )
    parser.add_argument(
        "--meta-eq",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose metadata KEY stringifies exactly to VALUE.",
    )
    parser.add_argument(
        "--meta-not-eq",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Drop rows whose metadata KEY stringifies exactly to VALUE.",
    )
    parser.add_argument("--max-examples-per-label", type=int, default=0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each kept source row N times before optional per-label capping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inputs = _expand_input_paths(list(args.input), list(args.input_list))
    if not inputs:
        parser.error("at least one --input or --input-list dataset is required")
    repeat = max(1, int(args.repeat))
    row_arrays: dict[str, list[np.ndarray]] = {}
    static_arrays: dict[str, np.ndarray] = {}
    source_reports: list[dict[str, Any]] = []
    source_examples = 0
    source_kept = 0
    states = {item.strip().upper() for item in args.include_states.split(",") if item.strip()}
    include_labels = _label_filter(str(args.include_labels))
    exclude_labels = _label_filter(str(args.exclude_labels))
    meta_min = _numeric_filters(args.meta_min, flag="--meta-min")
    meta_max = _numeric_filters(args.meta_max, flag="--meta-max")
    meta_abs_max = _numeric_filters(args.meta_abs_max, flag="--meta-abs-max")
    meta_eq = _string_filters(args.meta_eq, flag="--meta-eq")
    meta_not_eq = _string_filters(args.meta_not_eq, flag="--meta-not-eq")

    for input_path in inputs:
        with np.load(input_path, allow_pickle=False) as data:
            arrays, report = _filtered_arrays(
                data,
                input_path=input_path,
                states=states,
                include_labels=include_labels,
                exclude_labels=exclude_labels,
                require_meta_bool=list(args.require_meta_bool),
                meta_min=meta_min,
                meta_max=meta_max,
                meta_abs_max=meta_abs_max,
                meta_eq=meta_eq,
                meta_not_eq=meta_not_eq,
            )
            labels = np.asarray(data["labels"], dtype=np.int64)
            source_examples += int(labels.shape[0])
            source_kept += int(arrays["labels"].shape[0])
            if report:
                source_reports.append(report)

            for _ in range(repeat):
                for key, value in arrays.items():
                    row_arrays.setdefault(key, []).append(value)
            for key in data.files:
                if key in arrays or key in {
                    "result_json",
                    "filter_report_json",
                    "relabel_report_json",
                }:
                    continue
                value = data[key]
                if key not in static_arrays:
                    static_arrays[key] = np.asarray(value)
                elif not np.array_equal(static_arrays[key], value):
                    raise SystemExit(f"static array mismatch for key {key!r} in {input_path}")

    if not row_arrays or "labels" not in row_arrays:
        raise SystemExit("no source rows kept")

    arrays = dict(static_arrays)
    for key, chunks in row_arrays.items():
        arrays[key] = np.concatenate(chunks, axis=0)

    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if int(args.max_examples_per_label) > 0:
        keep = _per_label_cap(
            labels,
            np.ones(labels.shape[0], dtype=bool),
            cap=int(args.max_examples_per_label),
            seed=int(args.seed),
        )
        for key, value in list(arrays.items()):
            if value.shape[:1] == labels.shape[:1]:
                arrays[key] = value[keep]
        labels = np.asarray(arrays["labels"], dtype=np.int64)

    report_payload = {
        "inputs": [str(path) for path in inputs],
        "kept_examples": int(labels.shape[0]),
        "source_examples": int(source_examples),
        "source_kept_before_repeat": int(source_kept),
        "repeat": int(repeat),
        "include_states": sorted(states),
        "include_labels": sorted(include_labels),
        "exclude_labels": sorted(exclude_labels),
        "require_meta_bool": list(args.require_meta_bool),
        "meta_min": dict(sorted(meta_min.items())),
        "meta_max": dict(sorted(meta_max.items())),
        "meta_abs_max": dict(sorted(meta_abs_max.items())),
        "meta_eq": dict(sorted(meta_eq.items())),
        "meta_not_eq": dict(sorted(meta_not_eq.items())),
        "max_examples_per_label": int(args.max_examples_per_label),
    }
    arrays["result_json"] = np.asarray([json.dumps(_merged_result(source_reports), sort_keys=True)])
    arrays["filter_report_json"] = np.asarray([json.dumps(report_payload, sort_keys=True)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        f"wrote {args.output} kept={int(labels.shape[0])}/"
        f"{int(source_examples)} repeat={int(repeat)} inputs={len(inputs)}"
    )
    return 0


def _filtered_arrays(
    data: np.lib.npyio.NpzFile,
    *,
    input_path: Path,
    states: set[str],
    include_labels: set[str],
    exclude_labels: set[str],
    require_meta_bool: list[str],
    meta_min: dict[str, float],
    meta_max: dict[str, float],
    meta_abs_max: dict[str, float],
    meta_eq: dict[str, str],
    meta_not_eq: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "labels" not in data:
        raise SystemExit(f"{input_path} has no labels array")
    labels = np.asarray(data["labels"], dtype=np.int64)
    keep = np.ones(labels.shape[0], dtype=bool)
    meta = _meta_rows(data, rows=labels.shape[0])
    primitive_vocab = [str(item) for item in np.asarray(data["primitive_vocab"]).tolist()]

    if states:
        keep &= np.asarray(
            [str(row.get("state", "")).upper() in states for row in meta],
            dtype=bool,
        )
    if include_labels:
        include_ids = _label_ids(include_labels, primitive_vocab, input_path=input_path)
        keep &= np.isin(labels, np.asarray(sorted(include_ids), dtype=np.int64))
    if exclude_labels:
        exclude_ids = _label_ids(exclude_labels, primitive_vocab, input_path=input_path)
        keep &= ~np.isin(labels, np.asarray(sorted(exclude_ids), dtype=np.int64))
    for key in require_meta_bool:
        keep &= np.asarray([bool(row.get(key, False)) for row in meta], dtype=bool)
    for key, threshold in meta_min.items():
        keep &= np.asarray(
            [
                (value := _meta_float(row.get(key))) is not None and value >= threshold
                for row in meta
            ],
            dtype=bool,
        )
    for key, threshold in meta_max.items():
        keep &= np.asarray(
            [
                (value := _meta_float(row.get(key))) is not None and value <= threshold
                for row in meta
            ],
            dtype=bool,
        )
    for key, threshold in meta_abs_max.items():
        keep &= np.asarray(
            [
                (value := _meta_float(row.get(key))) is not None and abs(value) <= threshold
                for row in meta
            ],
            dtype=bool,
        )
    for key, expected in meta_eq.items():
        keep &= np.asarray(
            [_meta_string(row.get(key)) == expected for row in meta],
            dtype=bool,
        )
    for key, forbidden in meta_not_eq.items():
        keep &= np.asarray(
            [_meta_string(row.get(key)) != forbidden for row in meta],
            dtype=bool,
        )

    arrays: dict[str, Any] = {}
    for key in data.files:
        value = data[key]
        if value.shape[:1] == labels.shape[:1]:
            arrays[key] = value[keep]

    report = {}
    if "result_json" in data and len(data["result_json"]) > 0:
        try:
            parsed = json.loads(str(data["result_json"][0]))
        except json.JSONDecodeError:
            parsed = {}
        report = parsed if isinstance(parsed, dict) else {}
    return arrays, report


def _expand_input_paths(paths: list[Path], list_paths: list[Path]) -> list[Path]:
    out = list(paths)
    for list_path in list_paths:
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(Path(line))
    return out


def _label_filter(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def _numeric_filters(items: list[str], *, flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key, sep, raw_value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        try:
            out[key] = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"{flag} value must be numeric: {item!r}") from exc
    return out


def _string_filters(items: list[str], *, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, raw_value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        out[key] = raw_value.strip()
    return out


def _meta_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _meta_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _label_ids(labels: set[str], primitive_vocab: list[str], *, input_path: Path) -> set[int]:
    known = {name: idx for idx, name in enumerate(primitive_vocab)}
    missing = sorted(label for label in labels if label not in known)
    if missing:
        raise SystemExit(f"{input_path} has no primitive labels: {', '.join(missing)}")
    return {int(known[label]) for label in labels}


def _meta_rows(data: np.lib.npyio.NpzFile, *, rows: int) -> list[dict[str, Any]]:
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return [{} for _ in range(rows)]
    out: list[dict[str, Any]] = []
    for raw in data["meta_json"].tolist():
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            parsed = {}
        out.append(parsed if isinstance(parsed, dict) else {})
    return out


def _per_label_cap(labels: np.ndarray, keep: np.ndarray, *, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    capped = np.zeros_like(keep, dtype=bool)
    for label in sorted({int(item) for item in labels[keep].tolist()}):
        idx = np.flatnonzero(keep & (labels == label))
        if idx.shape[0] > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        capped[idx] = True
    return capped


def _merged_result(source_reports: list[dict[str, Any]]) -> dict[str, Any]:
    variant = "base"
    source_scenes = []
    claimed_all = True
    for report in source_reports:
        scene = report.get("scene")
        if scene:
            source_scenes.append(str(scene))
        claimed = {str(item) for item in report.get("claimed_colors", [])}
        claimed_all = claimed_all and {"red", "yellow", "blue", "green"}.issubset(claimed)
        metrics = report.get("wall_metrics", {})
        if isinstance(metrics, dict):
            current = str(metrics.get("learned_local_policy_feature_variant", ""))
            if current:
                variant = current
    return {
        "scene": "merged",
        "success": bool(claimed_all),
        "claimed_colors": ["red", "yellow", "blue", "green"] if claimed_all else [],
        "wall_metrics": {
            "learned_local_policy_feature_variant": variant,
            "merged_source_count": int(len(source_reports)),
            "merged_source_scenes": sorted(set(source_scenes)),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
