#!/usr/bin/env python3
"""Evaluate a hybrid Go2 memory selector plus target-geometry command extractor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_go2_causal_memory_target_gate import (  # noqa: E402
    _hidden_states_by_sequence as _selector_hidden_states_by_sequence,
    _select_object,
)
from evaluate_go2_memory_geometry_command_extractor import (  # noqa: E402
    _primitive_from_geometry,
    _steering_bucket_for_primitive,
)
from train_go2_causal_memory_query_probe import (  # noqa: E402
    QueryConditionedGo2MemoryProbe,
    _build_frames as _build_selector_frames,
    _current_role,
    _max_landmark_slot,
    _scrub_command_aux,
)
from train_go2_hidden_target_memory_probe import _load_rows, _resolve_device  # noqa: E402
from train_go2_memory_target_geometry import (  # noqa: E402
    QueryGeometryMemoryProbe,
    _build_frames as _build_geometry_frames,
    _hidden_states_by_sequence as _geometry_hidden_states_by_sequence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selector-threshold", type=float, default=None)
    parser.add_argument("--arc-threshold-rad", type=float, default=0.35)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.0)
    args = parser.parse_args()

    rows_raw = _load_rows(args.datasets)
    if not rows_raw:
        raise SystemExit("no rows")
    selector_checkpoint = _load_checkpoint(
        args.selector_checkpoint,
        expected_schema="lewm_go2_causal_memory_query_probe_checkpoint_v0",
    )
    geometry_checkpoint = _load_checkpoint(
        args.geometry_checkpoint,
        expected_schema="lewm_go2_memory_target_geometry_checkpoint_v0",
    )
    selector_threshold = (
        _checkpoint_threshold(selector_checkpoint)
        if args.selector_threshold is None
        else float(args.selector_threshold)
    )

    selector_rows = (
        _scrub_command_aux(rows_raw)
        if bool(dict(selector_checkpoint.get("args", {})).get("scrub_command_aux", False))
        else rows_raw
    )
    geometry_rows = (
        _scrub_command_aux(rows_raw)
        if bool(geometry_checkpoint.get("scrubbed_command_aux", True))
        else rows_raw
    )
    selector_sequences = _build_selector_frames(
        selector_rows,
        primitive_vocab=list(selector_checkpoint["primitive_vocab"]),
        color_vocab=list(selector_checkpoint["color_vocab"]),
        max_slot=_max_landmark_slot(rows_raw),
        feature_stats={
            "mean": np.asarray(selector_checkpoint["feature_mean"], dtype=np.float32),
            "std": np.asarray(selector_checkpoint["feature_std"], dtype=np.float32),
        },
        image_size=int(selector_checkpoint["image_size"]),
        include_object_slot=bool(
            dict(selector_checkpoint.get("args", {})).get("include_object_slot", False)
        ),
        include_privileged_landmark_geometry=bool(
            dict(selector_checkpoint.get("args", {})).get(
                "include_privileged_landmark_geometry",
                False,
            )
        ),
    )
    geometry_sequences = _build_geometry_frames(
        geometry_rows,
        primitive_vocab=list(geometry_checkpoint["primitive_vocab"]),
        color_vocab=list(geometry_checkpoint["color_vocab"]),
        max_slot=_max_landmark_slot(rows_raw),
        feature_stats={
            "mean": np.asarray(geometry_checkpoint["feature_mean"], dtype=np.float32),
            "std": np.asarray(geometry_checkpoint["feature_std"], dtype=np.float32),
        },
        image_size=int(geometry_checkpoint["image_size"]),
        range_scale_m=float(geometry_checkpoint["range_scale_m"]),
        include_object_slot=bool(
            dict(geometry_checkpoint.get("args", {})).get("include_object_slot", False)
        ),
    )
    row_index = _row_index(rows_raw)

    device = _resolve_device(str(args.device))
    selector_model = QueryConditionedGo2MemoryProbe(
        aux_dim=int(selector_checkpoint["aux_dim"]),
        query_dim=int(selector_checkpoint["query_dim"]),
        hidden_dim=int(selector_checkpoint["hidden_dim"]),
    ).to(device)
    selector_model.load_state_dict(selector_checkpoint["model_state_dict"])
    selector_model.eval()

    geometry_model = QueryGeometryMemoryProbe(
        aux_dim=int(geometry_checkpoint["aux_dim"]),
        query_dim=int(geometry_checkpoint["query_dim"]),
        hidden_dim=int(geometry_checkpoint["hidden_dim"]),
    ).to(device)
    geometry_model.load_state_dict(geometry_checkpoint["model_state_dict"])
    geometry_model.eval()

    ablations = (
        "normal",
        "reset_recurrent_state",
        "reverse_input_history",
        "shuffle_hidden_states",
    )
    evaluations = {
        ablation: _evaluate(
            selector_model,
            geometry_model,
            selector_sequences,
            geometry_sequences,
            row_index,
            device=device,
            selector_threshold=float(selector_threshold),
            range_scale_m=float(geometry_checkpoint["range_scale_m"]),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
            ablation=ablation,
        )
        for ablation in ablations
    }
    normal = evaluations["normal"]
    report = {
        "schema": "lewm_go2_hybrid_memory_geometry_controller_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "selector_checkpoint": str(args.selector_checkpoint),
        "geometry_checkpoint": str(args.geometry_checkpoint),
        "device": str(device),
        "selector_threshold": float(selector_threshold),
        "config": {
            "arc_threshold_rad": float(args.arc_threshold_rad),
            "yaw_threshold_rad": float(args.yaw_threshold_rad),
            "hold_range_m": float(args.hold_range_m),
        },
        "row_count": sum(len(sequence) for sequence in selector_sequences.values()),
        "sequence_count": len(selector_sequences),
        "evaluations": evaluations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": (
            normal["target_steering_pipeline_success"]
            - max(
                evaluations[name]["target_steering_pipeline_success"]
                for name in (
                    "reset_recurrent_state",
                    "reverse_input_history",
                    "shuffle_hidden_states",
                )
            )
        ),
        "normal_minus_best_corrupted_target_recall": (
            normal["target_recall"]
            - max(
                evaluations[name]["target_recall"]
                for name in (
                    "reset_recurrent_state",
                    "reverse_input_history",
                    "shuffle_hidden_states",
                )
            )
        ),
        "claim_boundary": (
            "Offline hybrid of learned target selection and learned "
            "target-relative geometry. This reports command-direction proxy "
            "success, not closed-loop robot execution."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_hybrid_memory_geometry_controller:"
        f" report={args.out}"
        f" target_recall={normal['target_recall']:.3f}"
        f" false_claim_rate={normal['false_claim_rate']:.3f}"
        f" target_steer_success={normal['target_steering_pipeline_success']:.3f}"
        f" delta={report['normal_minus_best_corrupted_target_steering_pipeline_success']:.3f}"
    )
    return 0


def _load_checkpoint(path: Path, *, expected_schema: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != expected_schema:
        raise SystemExit(f"unsupported checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


def _checkpoint_threshold(checkpoint: dict[str, Any]) -> float:
    return float(dict(checkpoint.get("args", {})).get("threshold", 0.5))


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, int, int], int], dict[str, Any]]:
    return {
        (
            (
                str(row.get("scene_id", "")),
                int(row.get("env_idx", 0)),
                int(row.get("episode_id", 0)),
            ),
            int(row.get("episode_step", 0)),
        ): row
        for row in rows
    }


def _evaluate(
    selector_model: QueryConditionedGo2MemoryProbe,
    geometry_model: QueryGeometryMemoryProbe,
    selector_sequences: dict[tuple[str, int, int], list[Any]],
    geometry_sequences: dict[tuple[str, int, int], list[Any]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    device: torch.device,
    selector_threshold: float,
    range_scale_m: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
    ablation: str,
) -> dict[str, Any]:
    selector_hidden = _selector_hidden_states_by_sequence(
        selector_model,
        selector_sequences,
        device=device,
        ablation=ablation,
    )
    geometry_hidden = _geometry_hidden_states_by_sequence(
        geometry_model,
        geometry_sequences,
        device=device,
        ablation=ablation,
    )
    geometry_frame_index = _frame_index(geometry_sequences)
    positive_frames = 0
    negative_frames = 0
    selected_frames = 0
    correct_target = 0
    missed_positive = 0
    false_claim = 0
    wrong_object = 0
    target_steer_success = 0
    target_primitive_success = 0
    route_steer_success = 0
    route_primitive_success = 0
    selected_records = []

    with torch.no_grad():
        for key, sequence in selector_sequences.items():
            hidden = selector_hidden[key]
            for step_idx, frame in enumerate(sequence):
                current_queries = [query for query in frame.queries if _current_role(query)]
                if not current_queries:
                    continue
                object_scores, object_colors, object_targets = _selector_scores(
                    selector_model,
                    hidden[step_idx],
                    current_queries,
                    device=device,
                )
                positive_objects = {
                    object_id
                    for object_id, target in object_targets.items()
                    if float(target) >= 0.5
                }
                if positive_objects:
                    positive_frames += 1
                else:
                    negative_frames += 1
                selected_object = _select_object(
                    object_scores,
                    object_colors=object_colors,
                    default_threshold=selector_threshold,
                    color_thresholds={},
                )
                if selected_object is None:
                    if positive_objects:
                        missed_positive += 1
                    continue
                selected_frames += 1
                if selected_object not in positive_objects:
                    if positive_objects:
                        wrong_object += 1
                    else:
                        false_claim += 1
                    continue
                correct_target += 1
                geometry_frame, geometry_step_idx = geometry_frame_index.get(
                    (frame.seq_key, int(frame.episode_step)),
                    (None, None),
                )
                if geometry_frame is None:
                    continue
                geometry_query = _geometry_query_by_object(geometry_frame, selected_object)
                if geometry_query is None:
                    continue
                route_primitive = str(
                    (row_index.get((frame.seq_key, int(frame.episode_step)), {}).get("command") or {}).get(
                        "primitive_name",
                        "",
                    )
                )
                record = _geometry_record(
                    geometry_model,
                    geometry_hidden[frame.seq_key][int(geometry_step_idx)],
                    geometry_query,
                    route_primitive=route_primitive,
                    device=device,
                    range_scale_m=range_scale_m,
                    arc_threshold_rad=arc_threshold_rad,
                    yaw_threshold_rad=yaw_threshold_rad,
                    hold_range_m=hold_range_m,
                )
                selected_records.append(record)
                if record["predicted_steering"] == record["target_steering"]:
                    target_steer_success += 1
                if record["predicted_primitive"] == record["target_primitive"]:
                    target_primitive_success += 1
                if record["predicted_steering"] == record["route_steering"]:
                    route_steer_success += 1
                if record["predicted_primitive"] == record["route_primitive"]:
                    route_primitive_success += 1

    return {
        "ablation": ablation,
        "positive_frame_count": float(positive_frames),
        "negative_frame_count": float(negative_frames),
        "selected_frame_count": float(selected_frames),
        "correct_target_count": float(correct_target),
        "missed_positive_count": float(missed_positive),
        "false_claim_count": float(false_claim),
        "wrong_object_count": float(wrong_object),
        "target_recall": correct_target / max(1, positive_frames),
        "false_claim_rate": false_claim / max(1, negative_frames),
        "target_selection_precision": correct_target / max(1, selected_frames),
        "target_steering_success_count": float(target_steer_success),
        "target_steering_pipeline_success": target_steer_success / max(1, positive_frames),
        "target_primitive_pipeline_success": target_primitive_success / max(1, positive_frames),
        "route_steering_pipeline_success": route_steer_success / max(1, positive_frames),
        "route_primitive_pipeline_success": route_primitive_success / max(1, positive_frames),
        "selected_record_counts": _record_counts(selected_records),
    }


def _selector_scores(
    model: QueryConditionedGo2MemoryProbe,
    hidden: torch.Tensor,
    current_queries: list[Any],
    *,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    query_features = torch.stack([query.features for query in current_queries]).to(device)
    hidden_rows = hidden.repeat(len(current_queries), 1)
    probs = torch.sigmoid(model.score_queries(hidden_rows, query_features)).detach().cpu().numpy()
    object_scores: dict[str, float] = {}
    object_colors: dict[str, str] = {}
    object_targets: dict[str, float] = {}
    for query, prob in zip(current_queries, probs):
        object_scores[query.object_id] = max(
            object_scores.get(query.object_id, -1.0),
            float(prob),
        )
        object_colors[query.object_id] = query.color
        object_targets[query.object_id] = max(
            object_targets.get(query.object_id, 0.0),
            float(query.target),
        )
    return object_scores, object_colors, object_targets


def _frame_index(
    sequences: dict[tuple[str, int, int], list[Any]],
) -> dict[tuple[tuple[str, int, int], int], tuple[Any, int]]:
    result = {}
    for sequence in sequences.values():
        for idx, frame in enumerate(sequence):
            result[(frame.seq_key, int(frame.episode_step))] = (frame, idx)
    return result


def _geometry_query_by_object(frame: Any, object_id: str) -> Any | None:
    for query in frame.queries:
        if query.role.startswith("current_") and query.object_id == object_id:
            return query
    return None


def _geometry_record(
    model: QueryGeometryMemoryProbe,
    hidden: torch.Tensor,
    query: Any,
    *,
    route_primitive: str,
    device: torch.device,
    range_scale_m: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> dict[str, Any]:
    query_features = query.features.unsqueeze(0).to(device)
    _, geom_pred = model.score_queries(hidden.unsqueeze(0), query_features)
    pred = geom_pred.squeeze(0).detach().cpu().numpy()
    pred_bearing = math.atan2(float(pred[0]), float(pred[1]))
    pred_range = max(0.0, float(pred[2]) * float(range_scale_m))
    target_primitive = _primitive_from_geometry(
        query.bearing_rad,
        query.range_m,
        arc_threshold_rad=arc_threshold_rad,
        yaw_threshold_rad=yaw_threshold_rad,
        hold_range_m=hold_range_m,
    )
    predicted_primitive = _primitive_from_geometry(
        pred_bearing,
        pred_range,
        arc_threshold_rad=arc_threshold_rad,
        yaw_threshold_rad=yaw_threshold_rad,
        hold_range_m=hold_range_m,
    )
    return {
        "object_id": query.object_id,
        "color": query.color,
        "target_primitive": target_primitive,
        "predicted_primitive": predicted_primitive,
        "route_primitive": route_primitive,
        "target_steering": _steering_bucket_for_primitive(target_primitive),
        "predicted_steering": _steering_bucket_for_primitive(predicted_primitive),
        "route_steering": _steering_bucket_for_primitive(route_primitive),
    }


def _record_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for field in (
        "color",
        "target_steering",
        "predicted_steering",
        "route_steering",
        "target_primitive",
        "predicted_primitive",
        "route_primitive",
    ):
        counts: dict[str, int] = {}
        for record in records:
            key = str(record.get(field, ""))
            counts[key] = counts.get(key, 0) + 1
        result[field] = dict(sorted(counts.items()))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
