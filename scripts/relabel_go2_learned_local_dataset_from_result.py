#!/usr/bin/env python3
"""Relabel a learned-local dataset from its paired closed-loop result log."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label-source",
        choices=(
            "final_primitive",
            "learned_wall_follow",
            "wall_guard_selected",
            "visual_servo",
        ),
        default="final_primitive",
    )
    parser.add_argument(
        "--include-states",
        default="",
        help="Optional comma-separated controller states to keep from dataset metadata.",
    )
    parser.add_argument(
        "--min-claimed-count",
        type=int,
        default=0,
        help="Drop rows with fewer prior claims in dataset metadata.",
    )
    parser.add_argument(
        "--visual-min-area-logit",
        type=float,
        default=-1.25,
        help="For visual_servo labels, require the active target area logit to be at least this value.",
    )
    parser.add_argument(
        "--visual-min-mem-conf",
        type=float,
        default=0.0,
        help="For visual_servo labels, require active target memory confidence to be at least this value.",
    )
    parser.add_argument(
        "--visual-forward-bearing",
        type=float,
        default=0.35,
        help="For visual_servo labels, drive forward when abs(bearing) is below this value.",
    )
    parser.add_argument(
        "--visual-low-area-arc-logit",
        type=float,
        default=None,
        help=(
            "For visual_servo labels, optionally use translational arc labels when "
            "area is at or below this logit and abs(bearing) is large. Omitted "
            "preserves the default yaw-only off-bearing rule."
        ),
    )
    parser.add_argument(
        "--visual-arc-bearing",
        type=float,
        default=None,
        help=(
            "For visual_servo labels, abs(bearing) threshold for the optional "
            "low-area arc rule."
        ),
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    log_rows = payload.get("log", [])
    if not isinstance(log_rows, list):
        raise SystemExit(f"{args.result} has no list log")
    by_tick = {
        int(row.get("tick")): row
        for row in log_rows
        if isinstance(row, dict) and row.get("tick") is not None
    }

    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels_in = np.asarray(data["labels"], dtype=np.int64)
        primitive_vocab = [str(item) for item in np.asarray(data["primitive_vocab"]).tolist()]
        meta_raw = [str(item) for item in np.asarray(data["meta_json"]).tolist()]
        schema = np.asarray(data["schema"])
        source_result_json = (
            [str(item) for item in np.asarray(data["result_json"]).tolist()]
            if "result_json" in data
            else []
        )

    if features.ndim != 2:
        raise SystemExit(f"{args.dataset} features must be rank-2, got {features.shape}")
    if int(features.shape[0]) != len(meta_raw) or int(labels_in.shape[0]) != len(meta_raw):
        raise SystemExit("dataset features/labels/meta row counts do not match")

    kept_features: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_meta: list[str] = []
    skipped = Counter()
    label_counts = Counter()
    include_states = {
        item.strip().upper()
        for item in str(args.include_states).split(",")
        if item.strip()
    }
    for idx, raw in enumerate(meta_raw):
        meta = json.loads(raw)
        state_name = str(meta.get("state", "")).upper()
        if include_states and state_name not in include_states:
            skipped["state_filter"] += 1
            continue
        if int(meta.get("claimed_count", 0)) < int(args.min_claimed_count):
            skipped["claimed_count_filter"] += 1
            continue
        tick = int(meta.get("tick", -1))
        row = by_tick.get(tick)
        if row is None:
            skipped["missing_tick"] += 1
            continue
        if str(args.label_source) == "visual_servo":
            primitive, skip_reason = _select_visual_servo_label(
                row,
                min_area_logit=float(args.visual_min_area_logit),
                min_mem_conf=float(args.visual_min_mem_conf),
                forward_bearing=float(args.visual_forward_bearing),
                low_area_arc_logit=args.visual_low_area_arc_logit,
                arc_bearing=args.visual_arc_bearing,
            )
            if primitive is None:
                skipped[skip_reason or "visual_servo_filter"] += 1
                continue
        else:
            primitive = _select_label(row, source=str(args.label_source))
        label_name = _label_primitive(primitive)
        if label_name is None or label_name not in primitive_vocab:
            skipped["unmapped_label"] += 1
            continue
        out_meta = dict(meta)
        out_meta["label"] = label_name
        out_meta["relabel_source"] = str(args.label_source)
        out_meta["relabel_source_primitive"] = None if primitive is None else str(primitive)
        if str(args.label_source) == "visual_servo":
            out_meta["visual_servo_area"] = _safe_float(row.get("area"))
            out_meta["visual_servo_bearing"] = _safe_float(row.get("bearing"))
            out_meta["visual_servo_mem_conf"] = _safe_float(row.get("mem_conf"))
            out_meta["visual_servo_in_cone"] = bool(row.get("in_cone", False))
        out_meta["source_dataset"] = str(args.dataset)
        out_meta["source_result"] = str(args.result)
        kept_features.append(features[idx])
        kept_labels.append(int(primitive_vocab.index(label_name)))
        kept_meta.append(json.dumps(out_meta, sort_keys=True))
        label_counts[label_name] += 1

    if not kept_features:
        raise SystemExit("no rows remained after relabeling")

    report = {
        "schema": "lewm_go2_learned_local_relabel_report_v0",
        "dataset": str(args.dataset),
        "result": str(args.result),
        "output": str(args.output),
        "label_source": str(args.label_source),
        "include_states": sorted(include_states),
        "min_claimed_count": int(args.min_claimed_count),
        "visual_min_area_logit": float(args.visual_min_area_logit),
        "visual_min_mem_conf": float(args.visual_min_mem_conf),
        "visual_forward_bearing": float(args.visual_forward_bearing),
        "visual_low_area_arc_logit": (
            None if args.visual_low_area_arc_logit is None else float(args.visual_low_area_arc_logit)
        ),
        "visual_arc_bearing": None if args.visual_arc_bearing is None else float(args.visual_arc_bearing),
        "input_rows": int(features.shape[0]),
        "output_rows": int(len(kept_features)),
        "skipped": dict(sorted(skipped.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "source_result_json_count": int(len(source_result_json)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=schema,
        features=np.stack(kept_features).astype(np.float32),
        labels=np.asarray(kept_labels, dtype=np.int64),
        primitive_vocab=np.asarray(primitive_vocab),
        meta_json=np.asarray(kept_meta),
        result_json=np.asarray(source_result_json[:1] or [json.dumps(report, sort_keys=True)]),
        relabel_report_json=np.asarray([json.dumps(report, sort_keys=True)]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _select_label(row: dict[str, Any], *, source: str) -> str | None:
    if source == "final_primitive":
        return None if row.get("primitive") is None else str(row.get("primitive"))
    if source == "learned_wall_follow":
        item = row.get("learned_wall_follow")
        if isinstance(item, dict) and item.get("selected") is not None:
            return str(item.get("selected"))
        return None if row.get("primitive") is None else str(row.get("primitive"))
    if source == "wall_guard_selected":
        item = row.get("wall_guard")
        if isinstance(item, dict) and item.get("selected") is not None:
            return str(item.get("selected"))
        return None if row.get("primitive") is None else str(row.get("primitive"))
    raise ValueError(source)


def _select_visual_servo_label(
    row: dict[str, Any],
    *,
    min_area_logit: float,
    min_mem_conf: float,
    forward_bearing: float,
    low_area_arc_logit: float | None,
    arc_bearing: float | None,
) -> tuple[str | None, str | None]:
    area = _safe_float(row.get("area"))
    bearing = _safe_float(row.get("bearing"))
    mem_conf = _safe_float(row.get("mem_conf"))
    if area is None:
        return None, "missing_area"
    if bearing is None:
        return None, "missing_bearing"
    if mem_conf is not None and mem_conf < float(min_mem_conf):
        return None, "low_mem_conf"
    if area < float(min_area_logit):
        return None, "low_area"
    if abs(bearing) <= float(forward_bearing):
        return "forward_medium", None
    if (
        low_area_arc_logit is not None
        and arc_bearing is not None
        and area <= float(low_area_arc_logit)
        and abs(bearing) >= float(arc_bearing)
    ):
        if bearing > 0.0:
            return "arc_left", None
        return "arc_right", None
    if bearing > 0.0:
        return "yaw_left", None
    return "yaw_right", None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _label_primitive(primitive: str | None) -> str | None:
    if primitive is None:
        return None
    if primitive == "forward_slow":
        return "forward_medium"
    return primitive


if __name__ == "__main__":
    raise SystemExit(main())
