#!/usr/bin/env python3
"""Relabel a learned-local dataset from its paired closed-loop result log."""

from __future__ import annotations

import argparse
import json
import math
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
            "teacher_nn_guard_compatible",
            "teacher_pose_guard_compatible",
            "target_xy_servo",
            "target_xy_guard_compatible",
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
    parser.add_argument(
        "--target-forward-bearing",
        type=float,
        default=0.25,
        help="For target_xy_servo labels, drive forward when abs(target bearing) is below this value.",
    )
    parser.add_argument(
        "--target-arc-bearing",
        type=float,
        default=0.9,
        help="For target_xy_servo labels, use arc labels below this abs(target bearing), yaw above it.",
    )
    parser.add_argument(
        "--target-yaw-to-arc-bearing",
        type=float,
        default=None,
        help=(
            "For target_xy_guard_compatible labels, when target_xy_servo would yaw "
            "but abs(target bearing) is at or below this value, prefer the same-side "
            "arc if the runtime wall/action guard marked that arc compatible."
        ),
    )
    parser.add_argument(
        "--target-opposite-arc-to-yaw",
        action="store_true",
        help=(
            "For target_xy_guard_compatible labels, if an arc toward the target "
            "would be reranked/guard-selected into the opposite arc, emit the "
            "same-side yaw primitive when it is guard-compatible."
        ),
    )
    parser.add_argument(
        "--teacher-dataset",
        type=Path,
        default=None,
        help=(
            "For teacher_nn_guard_compatible labels, use this successful "
            "learned-local dataset as nearest-neighbor behavior targets."
        ),
    )
    parser.add_argument(
        "--teacher-result",
        type=Path,
        default=None,
        help=(
            "For teacher_pose_guard_compatible labels, use this successful "
            "closed-loop result log as privileged pose/yaw behavior targets."
        ),
    )
    parser.add_argument(
        "--teacher-policy-feature-slot",
        default="post_claim",
        help=(
            "For teacher_nn_guard_compatible labels, keep teacher rows whose "
            "policy_feature_slot matches this value. Empty keeps all teacher rows."
        ),
    )
    parser.add_argument(
        "--teacher-min-claimed-count",
        type=int,
        default=None,
        help=(
            "For teacher_nn_guard_compatible labels, keep teacher rows with at "
            "least this claimed_count. Defaults to --min-claimed-count."
        ),
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    result_payload = payload.get("result", payload)
    target_xy_by_color = result_payload.get("target_xy", {})
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

    teacher_index = None
    if str(args.label_source) == "teacher_nn_guard_compatible":
        if args.teacher_dataset is None:
            raise SystemExit("--teacher-dataset is required for teacher_nn_guard_compatible")
        teacher_index = _load_teacher_nn_index(
            args.teacher_dataset,
            expected_feature_dim=int(features.shape[1]),
            expected_primitive_vocab=primitive_vocab,
            min_claimed_count=(
                int(args.min_claimed_count)
                if args.teacher_min_claimed_count is None
                else int(args.teacher_min_claimed_count)
            ),
            policy_feature_slot=str(args.teacher_policy_feature_slot),
        )
    elif str(args.label_source) == "teacher_pose_guard_compatible":
        if args.teacher_result is None:
            raise SystemExit("--teacher-result is required for teacher_pose_guard_compatible")
        teacher_index = _load_teacher_pose_index(args.teacher_result)

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
        teacher_match = None
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
        elif str(args.label_source) == "target_xy_servo":
            primitive, skip_reason = _select_target_xy_servo_label(
                row,
                target_xy_by_color=target_xy_by_color,
                forward_bearing=float(args.target_forward_bearing),
                arc_bearing=float(args.target_arc_bearing),
            )
            if primitive is None:
                skipped[skip_reason or "target_xy_servo_filter"] += 1
                continue
        elif str(args.label_source) == "target_xy_guard_compatible":
            primitive, skip_reason = _select_target_xy_guard_compatible_label(
                row,
                target_xy_by_color=target_xy_by_color,
                forward_bearing=float(args.target_forward_bearing),
                arc_bearing=float(args.target_arc_bearing),
                yaw_to_arc_bearing=args.target_yaw_to_arc_bearing,
                opposite_arc_to_yaw=bool(args.target_opposite_arc_to_yaw),
            )
            if primitive is None:
                skipped[skip_reason or "target_xy_guard_compatible_filter"] += 1
                continue
        elif str(args.label_source) == "teacher_nn_guard_compatible":
            assert teacher_index is not None
            primitive, skip_reason, teacher_match = _select_teacher_nn_guard_compatible_label(
                row,
                features[idx],
                teacher_index=teacher_index,
            )
            if primitive is None:
                skipped[skip_reason or "teacher_nn_guard_compatible_filter"] += 1
                continue
        elif str(args.label_source) == "teacher_pose_guard_compatible":
            assert teacher_index is not None
            primitive, skip_reason, teacher_match = _select_teacher_pose_guard_compatible_label(
                row,
                teacher_index=teacher_index,
            )
            if primitive is None:
                skipped[skip_reason or "teacher_pose_guard_compatible_filter"] += 1
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
        out_meta["relabel_source_post_xy"] = row.get("post_xy")
        out_meta["relabel_source_post_yaw"] = row.get("post_yaw")
        post_xy = row.get("post_xy")
        if isinstance(post_xy, (list, tuple)) and len(post_xy) >= 2:
            out_meta["relabel_source_post_x"] = _safe_float(post_xy[0])
            out_meta["relabel_source_post_y"] = _safe_float(post_xy[1])
        if str(args.label_source) == "visual_servo":
            out_meta["visual_servo_area"] = _safe_float(row.get("area"))
            out_meta["visual_servo_bearing"] = _safe_float(row.get("bearing"))
            out_meta["visual_servo_mem_conf"] = _safe_float(row.get("mem_conf"))
            out_meta["visual_servo_in_cone"] = bool(row.get("in_cone", False))
        elif str(args.label_source) == "target_xy_servo":
            out_meta["target_xy_servo_bearing"] = _target_bearing(row, target_xy_by_color)
            out_meta["target_xy_servo_post_xy"] = row.get("post_xy")
        elif str(args.label_source) == "target_xy_guard_compatible":
            guard_meta = _target_xy_guard_compatible_meta(
                row,
                target_xy_by_color,
                forward_bearing=float(args.target_forward_bearing),
                arc_bearing=float(args.target_arc_bearing),
                yaw_to_arc_bearing=args.target_yaw_to_arc_bearing,
                opposite_arc_to_yaw=bool(args.target_opposite_arc_to_yaw),
                emitted_label=label_name,
            )
            out_meta.update(guard_meta)
        elif str(args.label_source) == "teacher_nn_guard_compatible":
            out_meta.update(_teacher_nn_guard_compatible_meta(row, teacher_match, label_name))
        elif str(args.label_source) == "teacher_pose_guard_compatible":
            out_meta.update(_teacher_pose_guard_compatible_meta(row, teacher_match, label_name))
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
        "target_forward_bearing": float(args.target_forward_bearing),
        "target_arc_bearing": float(args.target_arc_bearing),
        "target_yaw_to_arc_bearing": (
            None if args.target_yaw_to_arc_bearing is None else float(args.target_yaw_to_arc_bearing)
        ),
        "target_opposite_arc_to_yaw": bool(args.target_opposite_arc_to_yaw),
        "teacher_dataset": None if args.teacher_dataset is None else str(args.teacher_dataset),
        "teacher_result": None if args.teacher_result is None else str(args.teacher_result),
        "teacher_policy_feature_slot": str(args.teacher_policy_feature_slot),
        "teacher_min_claimed_count": (
            int(args.min_claimed_count)
            if args.teacher_min_claimed_count is None
            else int(args.teacher_min_claimed_count)
        ),
        "teacher_rows": _teacher_index_row_count(teacher_index),
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


def _teacher_index_row_count(teacher_index: dict[str, Any] | None) -> int:
    if teacher_index is None:
        return 0
    if "features" in teacher_index:
        return int(teacher_index["features"].shape[0])
    if "pose" in teacher_index:
        return int(teacher_index["pose"].shape[0])
    return 0


def _load_teacher_nn_index(
    path: Path,
    *,
    expected_feature_dim: int,
    expected_primitive_vocab: list[str],
    min_claimed_count: int,
    policy_feature_slot: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        primitive_vocab = [str(item) for item in np.asarray(data["primitive_vocab"]).tolist()]
        meta_raw = [str(item) for item in np.asarray(data["meta_json"]).tolist()]
    if features.ndim != 2:
        raise SystemExit(f"{path} teacher features must be rank-2, got {features.shape}")
    if int(features.shape[1]) != int(expected_feature_dim):
        raise SystemExit(
            f"{path} teacher feature_dim={features.shape[1]} does not match "
            f"dataset feature_dim={expected_feature_dim}"
        )
    if primitive_vocab != list(expected_primitive_vocab):
        raise SystemExit(f"{path} teacher primitive vocab does not match dataset vocab")
    keep: list[int] = []
    teacher_meta: list[dict[str, Any]] = []
    slot_filter = str(policy_feature_slot)
    for idx, raw in enumerate(meta_raw):
        meta = json.loads(raw)
        if int(meta.get("claimed_count", 0)) < int(min_claimed_count):
            continue
        if slot_filter and str(meta.get("policy_feature_slot", "")) != slot_filter:
            continue
        keep.append(idx)
        teacher_meta.append(meta)
    if not keep:
        raise SystemExit(f"{path} produced no teacher rows after filters")
    kept_features = features[np.asarray(keep, dtype=np.int64)]
    kept_labels = labels[np.asarray(keep, dtype=np.int64)]
    scale = np.std(kept_features, axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-4, scale, 1.0).astype(np.float32)
    center = np.mean(kept_features, axis=0).astype(np.float32)
    teacher_z = ((kept_features - center) / scale).astype(np.float32)
    return {
        "features": kept_features,
        "labels": kept_labels,
        "primitive_vocab": primitive_vocab,
        "meta": teacher_meta,
        "center": center,
        "scale": scale,
        "z": teacher_z,
        "path": str(path),
    }


def _select_teacher_nn_guard_compatible_label(
    row: dict[str, Any],
    feature: np.ndarray,
    *,
    teacher_index: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    feature_z = (np.asarray(feature, dtype=np.float32) - teacher_index["center"]) / teacher_index["scale"]
    diff = teacher_index["z"] - feature_z.astype(np.float32)
    distances = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
    nearest = int(np.argmin(distances))
    teacher_label = str(teacher_index["primitive_vocab"][int(teacher_index["labels"][nearest])])
    teacher_label = _label_primitive(teacher_label)
    if teacher_label is None:
        return None, "teacher_unmapped_label", None
    emitted, fallback_reason = _teacher_guard_compatible_fallback(row, teacher_label)
    teacher_meta = teacher_index["meta"][nearest]
    match = {
        "teacher_dataset": teacher_index["path"],
        "teacher_index": nearest,
        "teacher_distance": float(distances[nearest]),
        "teacher_label": teacher_label,
        "teacher_tick": teacher_meta.get("tick"),
        "teacher_state": teacher_meta.get("state"),
        "teacher_target_color": teacher_meta.get("target_color"),
        "teacher_policy_feature_slot": teacher_meta.get("policy_feature_slot"),
        "fallback_reason": fallback_reason,
    }
    if emitted is None:
        return None, fallback_reason or "teacher_guard_no_label", match
    return emitted, None, match


def _load_teacher_pose_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    log_rows = payload.get("log", [])
    if not isinstance(log_rows, list):
        raise SystemExit(f"{path} has no list log")
    blue_claim_tick = None
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("state")) == "CLAIM" and str(row.get("target_color", "")).lower() == "blue":
            blue_claim_tick = int(row.get("tick", -1))
            break
    rows: list[dict[str, Any]] = []
    pose: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        tick = row.get("tick")
        if tick is None or (blue_claim_tick is not None and int(tick) <= int(blue_claim_tick)):
            continue
        if str(row.get("state", "")).upper() == "CLAIM":
            continue
        if str(row.get("target_color", "")).lower() != "yellow":
            continue
        post_xy = row.get("post_xy")
        yaw = _safe_float(row.get("post_yaw"))
        if post_xy is None or yaw is None:
            continue
        try:
            x, y = float(post_xy[0]), float(post_xy[1])
        except (TypeError, ValueError, IndexError):
            continue
        primitive = row.get("requested_primitive", row.get("primitive"))
        label = _label_primitive(None if primitive is None else str(primitive))
        if label is None:
            continue
        rows.append(row)
        pose.append((x, y, float(yaw)))
        labels.append(label)
    if not rows:
        raise SystemExit(f"{path} produced no teacher pose rows")
    return {
        "rows": rows,
        "pose": np.asarray(pose, dtype=np.float32),
        "labels": labels,
        "path": str(path),
        "blue_claim_tick": blue_claim_tick,
    }


def _select_teacher_pose_guard_compatible_label(
    row: dict[str, Any],
    *,
    teacher_index: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    post_xy = row.get("post_xy")
    yaw = _safe_float(row.get("post_yaw"))
    if post_xy is None or yaw is None:
        return None, "missing_pose", None
    try:
        x, y = float(post_xy[0]), float(post_xy[1])
    except (TypeError, ValueError, IndexError):
        return None, "missing_pose", None
    teacher_pose = np.asarray(teacher_index["pose"], dtype=np.float32)
    xy_delta = teacher_pose[:, :2] - np.asarray([x, y], dtype=np.float32)
    yaw_delta = np.asarray([_wrap_pi(float(item) - float(yaw)) for item in teacher_pose[:, 2]])
    distances = np.einsum("ij,ij->i", xy_delta, xy_delta, dtype=np.float32) + (
        0.35 * yaw_delta.astype(np.float32)
    ) ** 2
    nearest = int(np.argmin(distances))
    teacher_label = str(teacher_index["labels"][nearest])
    emitted, fallback_reason = _teacher_guard_compatible_fallback(row, teacher_label)
    teacher_row = teacher_index["rows"][nearest]
    match = {
        "teacher_result": teacher_index["path"],
        "teacher_index": nearest,
        "teacher_distance": float(distances[nearest]),
        "teacher_label": teacher_label,
        "teacher_tick": teacher_row.get("tick"),
        "teacher_state": teacher_row.get("state"),
        "teacher_target_color": teacher_row.get("target_color"),
        "teacher_post_xy": teacher_row.get("post_xy"),
        "teacher_post_yaw": teacher_row.get("post_yaw"),
        "fallback_reason": fallback_reason,
    }
    if emitted is None:
        return None, fallback_reason or "teacher_pose_guard_no_label", match
    return emitted, None, match


def _teacher_guard_compatible_fallback(
    row: dict[str, Any],
    teacher_label: str,
) -> tuple[str | None, str | None]:
    if _wall_guard_primitive_compatible(row, teacher_label):
        return teacher_label, "teacher_compatible"
    if teacher_label in ("arc_left", "arc_right"):
        same_side_yaw = "yaw_left" if teacher_label == "arc_left" else "yaw_right"
        if _wall_guard_primitive_compatible(row, same_side_yaw):
            return same_side_yaw, "arc_to_same_side_yaw"
    if teacher_label in ("yaw_left", "yaw_right"):
        same_side_arc = "arc_left" if teacher_label == "yaw_left" else "arc_right"
        if _wall_guard_primitive_compatible(row, same_side_arc):
            return same_side_arc, "yaw_to_same_side_arc"
    selected = _wall_guard_selected(row)
    if selected is not None:
        return selected, "wall_guard_selected_fallback"
    return teacher_label, "teacher_unchecked_fallback"


def _teacher_nn_guard_compatible_meta(
    row: dict[str, Any],
    teacher_match: dict[str, Any] | None,
    emitted_label: str,
) -> dict[str, Any]:
    selected = _wall_guard_selected(row)
    out = {
        "teacher_nn_guard_selected_fallback": selected,
        "teacher_nn_guard_emitted_label": str(emitted_label),
        "teacher_nn_guard_emitted_compatible": _wall_guard_primitive_compatible(row, emitted_label),
    }
    if teacher_match is not None:
        out.update({f"teacher_nn_guard_{key}": value for key, value in teacher_match.items()})
        out["teacher_nn_guard_used_selected_fallback"] = bool(
            emitted_label == selected and emitted_label != teacher_match.get("teacher_label")
        )
    return out


def _teacher_pose_guard_compatible_meta(
    row: dict[str, Any],
    teacher_match: dict[str, Any] | None,
    emitted_label: str,
) -> dict[str, Any]:
    selected = _wall_guard_selected(row)
    out = {
        "teacher_pose_guard_selected_fallback": selected,
        "teacher_pose_guard_emitted_label": str(emitted_label),
        "teacher_pose_guard_emitted_compatible": _wall_guard_primitive_compatible(row, emitted_label),
    }
    if teacher_match is not None:
        out.update({f"teacher_pose_guard_{key}": value for key, value in teacher_match.items()})
        out["teacher_pose_guard_used_selected_fallback"] = bool(
            emitted_label == selected and emitted_label != teacher_match.get("teacher_label")
        )
    return out


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


def _select_target_xy_servo_label(
    row: dict[str, Any],
    *,
    target_xy_by_color: Any,
    forward_bearing: float,
    arc_bearing: float,
) -> tuple[str | None, str | None]:
    bearing = _target_bearing(row, target_xy_by_color)
    if bearing is None:
        return None, "missing_target_bearing"
    if abs(bearing) <= float(forward_bearing):
        return "forward_medium", None
    if abs(bearing) <= float(arc_bearing):
        if bearing > 0.0:
            return "arc_left", None
        return "arc_right", None
    if bearing > 0.0:
        return "yaw_left", None
    return "yaw_right", None


def _select_target_xy_guard_compatible_label(
    row: dict[str, Any],
    *,
    target_xy_by_color: Any,
    forward_bearing: float,
    arc_bearing: float,
    yaw_to_arc_bearing: float | None,
    opposite_arc_to_yaw: bool,
) -> tuple[str | None, str | None]:
    intended, skip_reason = _select_target_xy_servo_label(
        row,
        target_xy_by_color=target_xy_by_color,
        forward_bearing=forward_bearing,
        arc_bearing=arc_bearing,
    )
    if intended is None:
        return None, skip_reason
    bearing = _target_bearing(row, target_xy_by_color)
    if (
        bearing is not None
        and yaw_to_arc_bearing is not None
        and intended in ("yaw_left", "yaw_right")
        and abs(float(bearing)) <= float(yaw_to_arc_bearing)
    ):
        arc = "arc_left" if intended == "yaw_left" else "arc_right"
        if _wall_guard_primitive_compatible(row, arc):
            yaw = "yaw_left" if arc == "arc_left" else "yaw_right"
            if (
                bool(opposite_arc_to_yaw)
                and _wall_guard_selected(row) == _opposite_arc(arc)
                and _wall_guard_primitive_compatible(row, yaw)
            ):
                return yaw, None
            return arc, None
    if (
        bool(opposite_arc_to_yaw)
        and intended in ("arc_left", "arc_right")
        and _wall_guard_selected(row) == _opposite_arc(intended)
    ):
        yaw = "yaw_left" if intended == "arc_left" else "yaw_right"
        if _wall_guard_primitive_compatible(row, yaw):
            return yaw, None
    if _wall_guard_primitive_compatible(row, intended):
        return intended, None
    fallback = _wall_guard_selected(row)
    if fallback is not None:
        return fallback, None
    return intended, None


def _target_xy_guard_compatible_meta(
    row: dict[str, Any],
    target_xy_by_color: Any,
    *,
    forward_bearing: float,
    arc_bearing: float,
    yaw_to_arc_bearing: float | None,
    opposite_arc_to_yaw: bool,
    emitted_label: str,
) -> dict[str, Any]:
    bearing = _target_bearing(row, target_xy_by_color)
    intended = None
    arc_preference = None
    if bearing is not None:
        intended, _ = _select_target_xy_servo_label(
            row,
            target_xy_by_color=target_xy_by_color,
            forward_bearing=forward_bearing,
            arc_bearing=arc_bearing,
        )
        if (
            yaw_to_arc_bearing is not None
            and intended in ("yaw_left", "yaw_right")
            and abs(float(bearing)) <= float(yaw_to_arc_bearing)
        ):
            arc_preference = "arc_left" if intended == "yaw_left" else "arc_right"
    selected = _wall_guard_selected(row)
    intended_compatible = (
        False if intended is None else _wall_guard_primitive_compatible(row, intended)
    )
    arc_preference_compatible = (
        None
        if arc_preference is None
        else _wall_guard_primitive_compatible(row, arc_preference)
    )
    return {
        "target_xy_guard_bearing": bearing,
        "target_xy_guard_post_xy": row.get("post_xy"),
        "target_xy_guard_intended": intended,
        "target_xy_guard_intended_compatible": bool(intended_compatible),
        "target_xy_guard_arc_preference": arc_preference,
        "target_xy_guard_arc_preference_compatible": arc_preference_compatible,
        "target_xy_guard_opposite_arc_to_yaw": bool(opposite_arc_to_yaw),
        "target_xy_guard_opposite_arc": (
            None
            if intended not in ("arc_left", "arc_right")
            else _opposite_arc(str(intended))
        ),
        "target_xy_guard_selected_fallback": selected,
        "target_xy_guard_emitted_label": str(emitted_label),
        "target_xy_guard_used_selected_fallback": bool(
            emitted_label == selected
            and emitted_label != intended
            and emitted_label != arc_preference
        ),
    }


def _wall_guard_selected(row: dict[str, Any]) -> str | None:
    item = row.get("wall_guard")
    if isinstance(item, dict) and item.get("selected") is not None:
        return str(item.get("selected"))
    primitive = row.get("primitive")
    return None if primitive is None else str(primitive)


def _wall_guard_primitive_compatible(row: dict[str, Any], primitive: str) -> bool:
    item = row.get("wall_guard")
    if not isinstance(item, dict):
        return True
    requested = item.get("requested")
    selected = item.get("selected")
    if selected is not None and str(selected) == str(primitive):
        return True
    for candidate in item.get("candidates", ()):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("primitive", "")) != str(primitive):
            continue
        if bool(candidate.get("blocked", False)):
            return False
        return True
    if requested is not None and str(requested) == str(primitive):
        return not bool(item.get("requested_blocked", False))
    return False


def _opposite_arc(primitive: str) -> str | None:
    if primitive == "arc_left":
        return "arc_right"
    if primitive == "arc_right":
        return "arc_left"
    return None


def _target_bearing(row: dict[str, Any], target_xy_by_color: Any) -> float | None:
    color = str(row.get("target_color", "")).lower()
    if not color or not isinstance(target_xy_by_color, dict):
        return None
    target_xy = target_xy_by_color.get(color)
    post_xy = row.get("post_xy")
    yaw = _safe_float(row.get("post_yaw"))
    if target_xy is None or post_xy is None or yaw is None:
        return None
    try:
        tx, ty = float(target_xy[0]), float(target_xy[1])
        x, y = float(post_xy[0]), float(post_xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    return _wrap_pi(math.atan2(ty - y, tx - x) - float(yaw))


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


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
