#!/usr/bin/env python3
"""Evaluate command extraction from a Go2 target-geometry memory checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_causal_memory_query_probe import _scrub_command_aux  # noqa: E402
from train_go2_hidden_target_memory_probe import _load_rows, _resolve_device  # noqa: E402
from train_go2_memory_target_geometry import (  # noqa: E402
    QueryGeometryMemoryProbe,
    _build_frames,
    _hidden_states_by_sequence,
    _landmark_slot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arc-threshold-rad", type=float, default=0.35)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.35)
    parser.add_argument("--seen-threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    rows_raw = _load_rows(args.datasets)
    if not rows_raw:
        raise SystemExit("no rows")
    rows_for_memory = (
        _scrub_command_aux(rows_raw)
        if bool(checkpoint.get("scrubbed_command_aux", True))
        else rows_raw
    )
    row_index = _row_index(rows_raw)
    feature_stats = {
        "mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
    }
    checkpoint_args = dict(checkpoint.get("args", {}))
    sequences = _build_frames(
        rows_for_memory,
        primitive_vocab=list(checkpoint["primitive_vocab"]),
        color_vocab=list(checkpoint["color_vocab"]),
        max_slot=_max_landmark_slot(rows_raw),
        feature_stats=feature_stats,
        image_size=int(checkpoint["image_size"]),
        range_scale_m=float(checkpoint["range_scale_m"]),
        include_object_slot=bool(checkpoint_args.get("include_object_slot", False)),
    )
    if not sequences:
        raise SystemExit("no evaluable sequences")

    device = _resolve_device(str(args.device))
    model = QueryGeometryMemoryProbe(
        aux_dim=int(checkpoint["aux_dim"]),
        query_dim=int(checkpoint["query_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ablations = (
        "normal",
        "reset_recurrent_state",
        "reverse_input_history",
        "shuffle_hidden_states",
    )
    evaluations = {
        ablation: _evaluate(
            model,
            sequences,
            row_index,
            device=device,
            range_scale_m=float(checkpoint["range_scale_m"]),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
            seen_threshold=float(args.seen_threshold),
            ablation=ablation,
        )
        for ablation in ablations
    }
    baselines = _baselines(evaluations["normal"]["oracle_target_examples"])
    report = {
        "schema": "lewm_go2_memory_geometry_command_extractor_report_v0",
        "checkpoint": str(args.checkpoint),
        "datasets": [str(path) for path in args.datasets],
        "device": str(device),
        "scrubbed_command_aux": bool(checkpoint.get("scrubbed_command_aux", True)),
        "config": {
            "arc_threshold_rad": float(args.arc_threshold_rad),
            "yaw_threshold_rad": float(args.yaw_threshold_rad),
            "hold_range_m": float(args.hold_range_m),
            "seen_threshold": float(args.seen_threshold),
        },
        "sequence_count": len(sequences),
        "row_count": sum(len(sequence) for sequence in sequences.values()),
        "baselines": baselines,
        "evaluations": {
            key: _strip_examples(value) for key, value in evaluations.items()
        },
        "normal_minus_best_corrupted_target_steering_accuracy": (
            evaluations["normal"]["oracle_target"]["target_steering_accuracy"]
            - max(
                evaluations[name]["oracle_target"]["target_steering_accuracy"]
                for name in (
                    "reset_recurrent_state",
                    "reverse_input_history",
                    "shuffle_hidden_states",
                )
            )
        ),
        "normal_minus_best_corrupted_route_steering_accuracy": (
            evaluations["normal"]["oracle_target"]["route_steering_accuracy"]
            - max(
                evaluations[name]["oracle_target"]["route_steering_accuracy"]
                for name in (
                    "reset_recurrent_state",
                    "reverse_input_history",
                    "shuffle_hidden_states",
                )
            )
        ),
        "claim_boundary": (
            "Offline command extraction from predicted target-relative geometry. "
            "This is not closed-loop Go2 execution; it only tests whether the "
            "memory geometry has enough directional structure to choose command "
            "families under memory ablations."
        ),
    }

    out = args.out or args.checkpoint.with_suffix(".geometry_command_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    normal = report["evaluations"]["normal"]["oracle_target"]
    pipeline = report["evaluations"]["normal"]["learned_seen_gate"]
    print(
        "go2_memory_geometry_command_extractor:"
        f" report={out}"
        f" target_steer={normal['target_steering_accuracy']:.3f}"
        f" route_steer={normal['route_steering_accuracy']:.3f}"
        f" route_primitive={normal['route_primitive_accuracy']:.3f}"
        f" pipeline_success={pipeline['target_steering_pipeline_success']:.3f}"
    )
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "lewm_go2_memory_target_geometry_checkpoint_v0":
        raise SystemExit(f"unsupported checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, int, int], int], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (
            (
                str(row.get("scene_id", "")),
                int(row.get("env_idx", 0)),
                int(row.get("episode_id", 0)),
            ),
            int(row.get("episode_step", 0)),
        )
        result[key] = row
    return result


def _max_landmark_slot(rows: list[dict[str, Any]]) -> int:
    slots = [
        _landmark_slot(str(landmark.get("object_id", "")))
        for row in rows
        for landmark in row.get("landmarks", ())
    ]
    return max(slots) if slots else 0


def _evaluate(
    model: QueryGeometryMemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    device: torch.device,
    range_scale_m: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
    seen_threshold: float,
    ablation: str,
) -> dict[str, Any]:
    hidden_by_key = _hidden_states_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
    )
    oracle_examples = []
    pipeline = _PipelineMetrics()
    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                row = row_index.get((frame.seq_key, int(frame.episode_step)))
                if row is None:
                    continue
                route_primitive = str((row.get("command") or {}).get("primitive_name", ""))
                current_queries = [
                    query for query in frame.queries if query.role.startswith("current_")
                ]
                if not current_queries:
                    continue
                query_features = torch.stack([query.features for query in current_queries]).to(device)
                hidden_rows = hidden[step_idx].repeat(len(current_queries), 1)
                seen_logits, geom_pred = model.score_queries(hidden_rows, query_features)
                seen_probs = torch.sigmoid(seen_logits).detach().cpu().numpy()
                geom_np = geom_pred.detach().cpu().numpy()
                positive_indices = [
                    idx
                    for idx, query in enumerate(current_queries)
                    if float(query.seen_before) >= 0.5
                ]
                selected_idx = _select_seen_index(seen_probs, threshold=seen_threshold)
                selected_positive = (
                    selected_idx is not None and selected_idx in set(positive_indices)
                )
                selected_record = None
                if selected_idx is not None:
                    selected_record = _prediction_record(
                        current_queries[selected_idx],
                        geom_np[selected_idx],
                        route_primitive=route_primitive,
                        range_scale_m=range_scale_m,
                        arc_threshold_rad=arc_threshold_rad,
                        yaw_threshold_rad=yaw_threshold_rad,
                        hold_range_m=hold_range_m,
                    )
                if positive_indices:
                    for idx in positive_indices:
                        oracle_examples.append(
                            _prediction_record(
                                current_queries[idx],
                                geom_np[idx],
                                route_primitive=route_primitive,
                                range_scale_m=range_scale_m,
                                arc_threshold_rad=arc_threshold_rad,
                                yaw_threshold_rad=yaw_threshold_rad,
                                hold_range_m=hold_range_m,
                            )
                        )
                pipeline.add(
                    has_positive=bool(positive_indices),
                    selected=selected_idx is not None,
                    selected_positive=selected_positive,
                    selected_record=selected_record if selected_positive else None,
                )
    return {
        "ablation": ablation,
        "oracle_target": _example_metrics(oracle_examples),
        "learned_seen_gate": pipeline.to_dict(),
        "oracle_target_examples": oracle_examples,
    }


def _select_seen_index(seen_probs: np.ndarray, *, threshold: float) -> int | None:
    if len(seen_probs) <= 0:
        return None
    index = int(np.argmax(seen_probs))
    if float(seen_probs[index]) < float(threshold):
        return None
    return index


def _prediction_record(
    query: Any,
    pred: np.ndarray,
    *,
    route_primitive: str,
    range_scale_m: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> dict[str, Any]:
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
        "true_bearing_rad": float(query.bearing_rad),
        "pred_bearing_rad": float(pred_bearing),
        "true_range_m": float(query.range_m),
        "pred_range_m": float(pred_range),
        "target_primitive": target_primitive,
        "predicted_primitive": predicted_primitive,
        "route_primitive": route_primitive,
        "target_steering": _steering_bucket_for_primitive(target_primitive),
        "predicted_steering": _steering_bucket_for_primitive(predicted_primitive),
        "route_steering": _steering_bucket_for_primitive(route_primitive),
    }


def _primitive_from_geometry(
    bearing_rad: float,
    range_m: float,
    *,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> str:
    if float(range_m) <= float(hold_range_m):
        return "hold"
    if bearing_rad >= yaw_threshold_rad:
        return "yaw_left"
    if bearing_rad <= -yaw_threshold_rad:
        return "yaw_right"
    if bearing_rad >= arc_threshold_rad:
        return "arc_left"
    if bearing_rad <= -arc_threshold_rad:
        return "arc_right"
    return "forward_medium"


def _steering_bucket_for_primitive(primitive: str) -> str:
    if primitive in {"yaw_left", "arc_left", "lateral_left"}:
        return "left"
    if primitive in {"yaw_right", "arc_right", "lateral_right"}:
        return "right"
    if primitive in {"hold", "recovery_stand"}:
        return "hold"
    return "forward"


def _example_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {
            "example_count": 0.0,
            "target_steering_accuracy": 0.0,
            "target_primitive_accuracy": 0.0,
            "route_steering_accuracy": 0.0,
            "route_primitive_accuracy": 0.0,
        }
    target_steer_correct = sum(
        1 for item in examples if item["predicted_steering"] == item["target_steering"]
    )
    target_primitive_correct = sum(
        1 for item in examples if item["predicted_primitive"] == item["target_primitive"]
    )
    route_steer_correct = sum(
        1 for item in examples if item["predicted_steering"] == item["route_steering"]
    )
    route_primitive_correct = sum(
        1 for item in examples if item["predicted_primitive"] == item["route_primitive"]
    )
    return {
        "example_count": float(len(examples)),
        "target_steering_accuracy": target_steer_correct / max(1, len(examples)),
        "target_primitive_accuracy": target_primitive_correct / max(1, len(examples)),
        "route_steering_accuracy": route_steer_correct / max(1, len(examples)),
        "route_primitive_accuracy": route_primitive_correct / max(1, len(examples)),
        "target_steering_counts": _counts(examples, "target_steering"),
        "predicted_steering_counts": _counts(examples, "predicted_steering"),
        "route_steering_counts": _counts(examples, "route_steering"),
        "target_primitive_counts": _counts(examples, "target_primitive"),
        "predicted_primitive_counts": _counts(examples, "predicted_primitive"),
        "route_primitive_counts": _counts(examples, "route_primitive"),
    }


class _PipelineMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.selected_frames = 0
        self.correct_selected_target = 0
        self.false_claim = 0
        self.missed_positive = 0
        self.target_steer_success = 0
        self.route_steer_success = 0
        self.route_primitive_success = 0

    def add(
        self,
        *,
        has_positive: bool,
        selected: bool,
        selected_positive: bool,
        selected_record: dict[str, Any] | None,
    ) -> None:
        if has_positive:
            self.positive_frames += 1
        else:
            self.negative_frames += 1
        if selected:
            self.selected_frames += 1
        if has_positive and not selected:
            self.missed_positive += 1
            return
        if selected and not selected_positive:
            self.false_claim += 1
            return
        if not selected_positive or selected_record is None:
            return
        self.correct_selected_target += 1
        if selected_record["predicted_steering"] == selected_record["target_steering"]:
            self.target_steer_success += 1
        if selected_record["predicted_steering"] == selected_record["route_steering"]:
            self.route_steer_success += 1
        if selected_record["predicted_primitive"] == selected_record["route_primitive"]:
            self.route_primitive_success += 1

    def to_dict(self) -> dict[str, float]:
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "selected_frame_count": float(self.selected_frames),
            "correct_selected_target_count": float(self.correct_selected_target),
            "false_claim_count": float(self.false_claim),
            "missed_positive_count": float(self.missed_positive),
            "target_recall": self.correct_selected_target / max(1, self.positive_frames),
            "target_steering_pipeline_success": self.target_steer_success
            / max(1, self.positive_frames),
            "route_steering_pipeline_success": self.route_steer_success
            / max(1, self.positive_frames),
            "route_primitive_pipeline_success": self.route_primitive_success
            / max(1, self.positive_frames),
        }


def _baselines(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {}
    target_steering_majority = _majority_value(examples, "target_steering")
    route_steering_majority = _majority_value(examples, "route_steering")
    route_primitive_majority = _majority_value(examples, "route_primitive")
    return {
        "target_steering_majority": target_steering_majority,
        "target_steering_majority_accuracy": _constant_accuracy(
            examples,
            field="target_steering",
            value=target_steering_majority,
        ),
        "route_steering_majority": route_steering_majority,
        "route_steering_majority_accuracy": _constant_accuracy(
            examples,
            field="route_steering",
            value=route_steering_majority,
        ),
        "route_primitive_majority": route_primitive_majority,
        "route_primitive_majority_accuracy": _constant_accuracy(
            examples,
            field="route_primitive",
            value=route_primitive_majority,
        ),
        "route_vs_target_steering_accuracy": sum(
            1 for item in examples if item["route_steering"] == item["target_steering"]
        )
        / max(1, len(examples)),
        "route_vs_target_primitive_accuracy": sum(
            1 for item in examples if item["route_primitive"] == item["target_primitive"]
        )
        / max(1, len(examples)),
    }


def _majority_value(examples: list[dict[str, Any]], field: str) -> str:
    return Counter(str(item[field]) for item in examples).most_common(1)[0][0]


def _constant_accuracy(examples: list[dict[str, Any]], *, field: str, value: str) -> float:
    return sum(1 for item in examples if str(item[field]) == str(value)) / max(1, len(examples))


def _counts(examples: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item[field]) for item in examples).items()))


def _strip_examples(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "oracle_target_examples"}


if __name__ == "__main__":
    raise SystemExit(main())
