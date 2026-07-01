#!/usr/bin/env python3
"""Replay Go2 command blocks from a direct frozen-JEPA memory controller.

This is the execution-facing step after the offline selector+geometry proxy:
the learned memory selects a hidden target, the geometry head predicts a
target-relative primitive, and the primitive is expanded through the same Go2
command registry/safety adapter used by the Genesis/ROS command-block contract.

It is still replayed/offline: the emitted command blocks are not fed back into
Genesis physics here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_go2_direct_gate_geometry_controller import (  # noqa: E402
    _frame_index,
    _geometry_query_by_object,
    _load_checkpoint,
    _record_counts,
    _row_index,
)
from evaluate_go2_memory_geometry_command_extractor import (  # noqa: E402
    _primitive_from_geometry,
    _steering_bucket_for_primitive,
)
from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402
from train_go2_causal_memory_query_probe import (  # noqa: E402
    _build_frames as _build_selector_frames,
    _max_landmark_slot,
    _scrub_command_aux,
    _scrub_runtime_aux,
)
from train_go2_frozen_jepa_target_gate import (  # noqa: E402
    DirectGo2TargetGate,
    _candidate_batch,
    _hidden_states_by_sequence as _selector_hidden_states_by_sequence,
    _select_index,
)
from train_go2_hidden_target_memory_probe import _load_rows, _resolve_device  # noqa: E402
from train_go2_memory_target_geometry import (  # noqa: E402
    QueryGeometryMemoryProbe,
    _build_frames as _build_geometry_frames,
    _hidden_states_by_sequence as _geometry_hidden_states_by_sequence,
)


@dataclass(frozen=True)
class _PrimitiveRegistry:
    block_size: int
    command_dt_s: float
    primitives: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class _SafetyLimits:
    min_vx_mps: float
    max_vx_mps: float
    min_vy_mps: float
    max_vy_mps: float
    max_yaw_rate_radps: float
    max_delta_vx_mps: float
    max_delta_vy_mps: float
    max_delta_yaw_rate_radps: float


ABLATIONS = (
    "normal",
    "memory_off_abstain",
    "reset_recurrent_state",
    "reverse_input_history",
    "shuffle_hidden_states",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--command-jsonl-out", type=Path, default=None)
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument(
        "--platform-manifest",
        type=Path,
        default=ROOT / "config/go2_platform_manifest.yaml",
    )
    parser.add_argument("--selection-margin", type=float, default=None)
    parser.add_argument("--arc-threshold-rad", type=float, default=0.35)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-target-steering-success", type=float, default=0.70)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.25)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.15)
    args = parser.parse_args()

    rows_raw = _load_rows(args.datasets)
    if not rows_raw:
        raise SystemExit("no rows")
    selector_checkpoint = _load_checkpoint(
        args.selector_checkpoint,
        expected_schema="lewm_go2_frozen_jepa_target_gate_checkpoint_v0",
    )
    geometry_checkpoint = _load_checkpoint(
        args.geometry_checkpoint,
        expected_schema="lewm_go2_memory_target_geometry_checkpoint_v0",
    )
    selection_margin = (
        float(selector_checkpoint.get("selection_margin", 0.0))
        if args.selection_margin is None
        else float(args.selection_margin)
    )
    registry = _load_primitive_registry(args.primitive_registry)
    limits = _load_safety_limits(args.platform_manifest)

    selector_args = dict(selector_checkpoint.get("args", {}))
    if bool(selector_checkpoint.get("scrubbed_runtime_aux", False)) or bool(
        selector_args.get("scrub_runtime_aux", False)
    ):
        selector_rows = _scrub_runtime_aux(rows_raw)
    elif bool(selector_checkpoint.get("scrubbed_command_aux", False)) or bool(
        selector_args.get("scrub_command_aux", False)
    ):
        selector_rows = _scrub_command_aux(rows_raw)
    else:
        selector_rows = rows_raw
    geometry_args = dict(geometry_checkpoint.get("args", {}))
    if bool(geometry_checkpoint.get("scrubbed_runtime_aux", False)) or bool(
        geometry_args.get("scrub_runtime_aux", False)
    ):
        geometry_rows = _scrub_runtime_aux(rows_raw)
    elif bool(geometry_checkpoint.get("scrubbed_command_aux", True)):
        geometry_rows = _scrub_command_aux(rows_raw)
    else:
        geometry_rows = rows_raw
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
        include_object_slot=bool(selector_args.get("include_object_slot", False)),
        include_privileged_landmark_geometry=False,
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
        include_object_slot=bool(geometry_args.get("include_object_slot", False)),
    )
    if not selector_sequences:
        raise SystemExit("no selector sequences")
    if not geometry_sequences:
        raise SystemExit("no geometry sequences")

    device = _resolve_device(str(args.device))
    selector_encoder, selector_jepa = load_go2_jepa_encoder(
        selector_checkpoint["frozen_jepa_checkpoint"],
        device=device,
        freeze=True,
    )
    selector_model = DirectGo2TargetGate(
        encoder=selector_encoder,
        encoder_output_dim=int(selector_jepa.get("latent_dim", selector_checkpoint["hidden_dim"])),
        aux_dim=int(selector_checkpoint["aux_dim"]),
        query_dim=int(selector_checkpoint["query_dim"]),
        hidden_dim=int(selector_checkpoint["hidden_dim"]),
    ).to(device)
    selector_model.load_state_dict(selector_checkpoint["model_state_dict"])
    selector_model.eval()

    geometry_encoder = None
    geometry_encoder_dim = None
    if geometry_checkpoint.get("frozen_jepa_checkpoint"):
        geometry_encoder, geometry_jepa = load_go2_jepa_encoder(
            geometry_checkpoint["frozen_jepa_checkpoint"],
            device=device,
            freeze=True,
        )
        geometry_encoder_dim = int(
            geometry_jepa.get("latent_dim", geometry_checkpoint["hidden_dim"])
        )
    geometry_model = QueryGeometryMemoryProbe(
        aux_dim=int(geometry_checkpoint["aux_dim"]),
        query_dim=int(geometry_checkpoint["query_dim"]),
        hidden_dim=int(geometry_checkpoint["hidden_dim"]),
        encoder=geometry_encoder,
        encoder_output_dim=geometry_encoder_dim,
        freeze_encoder=geometry_encoder is not None,
        predict_steering=bool(geometry_checkpoint.get("has_steering_head", False)),
    ).to(device)
    geometry_model.load_state_dict(geometry_checkpoint["model_state_dict"])
    geometry_model.eval()

    row_index = _row_index(rows_raw)
    geometry_frame_index = _frame_index(geometry_sequences)
    evaluations: dict[str, Any] = {}
    command_records_by_ablation: dict[str, list[dict[str, Any]]] = {}
    for ablation in ABLATIONS:
        evaluation, command_records = _evaluate_replay(
            selector_model,
            geometry_model,
            selector_sequences,
            geometry_sequences,
            geometry_frame_index,
            row_index,
            registry=registry,
            limits=limits,
            device=device,
            selection_margin=selection_margin,
            range_scale_m=float(geometry_checkpoint["range_scale_m"]),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
            ablation=ablation,
        )
        evaluations[ablation] = evaluation
        command_records_by_ablation[ablation] = command_records

    normal = evaluations["normal"]
    corrupted_names = (
        "memory_off_abstain",
        "reset_recurrent_state",
        "reverse_input_history",
        "shuffle_hidden_states",
    )
    best_corrupted_target_steer = max(
        float(evaluations[name]["target_steering_pipeline_success"])
        for name in corrupted_names
    )
    best_corrupted_target_recall = max(
        float(evaluations[name]["target_recall"]) for name in corrupted_names
    )
    target_steer_gap = (
        float(normal["target_steering_pipeline_success"]) - best_corrupted_target_steer
    )
    target_recall_gap = float(normal["target_recall"]) - best_corrupted_target_recall
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and target_steer_gap >= float(args.min_corrupted_gap)
    )

    report = {
        "schema": "lewm_go2_frozen_jepa_command_replay_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "selector_checkpoint": str(args.selector_checkpoint),
        "geometry_checkpoint": str(args.geometry_checkpoint),
        "primitive_registry": str(args.primitive_registry),
        "platform_manifest": str(args.platform_manifest),
        "device": str(device),
        "selection_margin": float(selection_margin),
        "config": {
            "arc_threshold_rad": float(args.arc_threshold_rad),
            "yaw_threshold_rad": float(args.yaw_threshold_rad),
            "hold_range_m": float(args.hold_range_m),
            "min_target_steering_success": float(args.min_target_steering_success),
            "max_false_claim_rate": float(args.max_false_claim_rate),
            "min_corrupted_gap": float(args.min_corrupted_gap),
        },
        "row_count": sum(len(sequence) for sequence in selector_sequences.values()),
        "sequence_count": len(selector_sequences),
        "evaluations": evaluations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": target_steer_gap,
        "normal_minus_best_corrupted_target_recall": target_recall_gap,
        "execution_facing_replay_gate_pass": bool(gate_pass),
        "claim_boundary": (
            "Replayed command-block evaluation: primitives are expanded through "
            "the Go2 command registry and safety adapter, but the resulting "
            "blocks are not executed in Genesis physics or on hardware."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.command_jsonl_out is not None:
        args.command_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.command_jsonl_out.open("w", encoding="utf-8") as handle:
            for ablation in ABLATIONS:
                for record in command_records_by_ablation[ablation]:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        "go2_frozen_jepa_command_replay:"
        f" report={args.out}"
        f" target_recall={normal['target_recall']:.3f}"
        f" false_claim_rate={normal['false_claim_rate']:.3f}"
        f" target_steer_success={normal['target_steering_pipeline_success']:.3f}"
        f" delta={target_steer_gap:.3f}"
        f" pass={bool(gate_pass)}"
    )
    return 0


def _evaluate_replay(
    selector_model: DirectGo2TargetGate,
    geometry_model: QueryGeometryMemoryProbe,
    selector_sequences: dict[tuple[str, int, int], list[Any]],
    geometry_sequences: dict[tuple[str, int, int], list[Any]],
    geometry_frame_index: dict[tuple[tuple[str, int, int], int], tuple[Any, int]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    registry: _PrimitiveRegistry,
    limits: _SafetyLimits,
    device: torch.device,
    selection_margin: float,
    range_scale_m: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
    ablation: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selector_ablation = "normal" if ablation == "memory_off_abstain" else ablation
    selector_hidden = _selector_hidden_states_by_sequence(
        selector_model,
        selector_sequences,
        device=device,
        ablation=selector_ablation,
    )
    geometry_hidden = _geometry_hidden_states_by_sequence(
        geometry_model,
        geometry_sequences,
        device=device,
        ablation=selector_ablation,
    )

    metrics = _ReplayMetrics()
    command_records: list[dict[str, Any]] = []
    previous_by_sequence: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    sequence_ids: dict[tuple[str, int, int], int] = {}

    with torch.no_grad():
        for key, sequence in selector_sequences.items():
            hidden = selector_hidden[key]
            previous = previous_by_sequence.get(key, (0.0, 0.0, 0.0))
            for step_idx, frame in enumerate(sequence):
                candidate_batch = _candidate_batch(frame, device=device)
                if candidate_batch is None:
                    continue
                query_features, object_ids, positive_mask = candidate_batch
                positive_objects = {
                    object_id
                    for object_id, is_positive in zip(
                        object_ids,
                        positive_mask.detach().cpu().numpy(),
                    )
                    if bool(is_positive)
                }

                selected_object = None
                abstained_by_ablation = ablation == "memory_off_abstain"
                if not abstained_by_ablation:
                    abstain_logit, candidate_logits = selector_model.score_candidates(
                        hidden[step_idx],
                        query_features,
                    )
                    selected_index = _select_index(
                        abstain_logit,
                        candidate_logits,
                        selection_margin=selection_margin,
                    )
                    selected_object = None if selected_index is None else object_ids[selected_index]

                row = row_index.get((frame.seq_key, int(frame.episode_step)), {})
                route_primitive = str((row.get("command") or {}).get("primitive_name", ""))
                selected_record = None
                primitive_name = "hold"
                geometry_frame, geometry_step_idx = geometry_frame_index.get(
                    (frame.seq_key, int(frame.episode_step)),
                    (None, None),
                )
                if selected_object is not None and geometry_frame is not None:
                    geometry_query = _geometry_query_by_object(geometry_frame, selected_object)
                    if geometry_query is not None:
                        selected_record = _geometry_record(
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
                        primitive_name = str(selected_record["predicted_primitive"])

                requested_block = _expand_primitive_to_block(registry, primitive_name)
                requested = [tuple(map(float, row_)) for row_ in requested_block.tolist()]
                executed, clipped = _apply_safety_limits_single(
                    requested,
                    previous,
                    limits,
                    enforce_rate_limits=True,
                )
                previous = executed[-1] if executed else previous

                sequence_id = sequence_ids.setdefault(key, len(sequence_ids))
                command_record = _command_record(
                    ablation=ablation,
                    sequence_id=sequence_id,
                    frame=frame,
                    positive_objects=positive_objects,
                    selected_object=selected_object,
                    selected_record=selected_record,
                    primitive_name=primitive_name,
                    route_primitive=route_primitive,
                    requested=requested,
                    executed=executed,
                    clipped=clipped,
                    command_dt_s=float(registry.command_dt_s),
                )
                command_records.append(command_record)
                metrics.add(
                    positive_objects=positive_objects,
                    selected_object=selected_object,
                    selected_record=selected_record,
                    primitive_name=primitive_name,
                    clipped=clipped,
                )
            previous_by_sequence[key] = previous

    return metrics.to_dict(command_records), command_records


def _load_primitive_registry(path: Path) -> _PrimitiveRegistry:
    text = path.read_text(encoding="utf-8").splitlines()
    block_size = 5
    command_dt_s = 0.10
    primitives: dict[str, tuple[float, float, float]] = {}
    in_primitives = False
    current_name: str | None = None
    current_type = ""
    current_command: dict[str, float] = {}
    in_command = False

    def flush() -> None:
        nonlocal current_name, current_type, current_command
        if current_name is not None and current_type == "velocity_block":
            primitives[current_name] = (
                float(current_command.get("vx_body_mps", 0.0)),
                float(current_command.get("vy_body_mps", 0.0)),
                float(current_command.get("yaw_rate_radps", 0.0)),
            )
        current_name = None
        current_type = ""
        current_command = {}

    for raw in text:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if raw.startswith("block_size:"):
            block_size = int(stripped.split(":", 1)[1].strip())
            continue
        if raw.startswith("command_dt_s:"):
            command_dt_s = float(stripped.split(":", 1)[1].strip())
            continue
        if raw.startswith("primitives:"):
            in_primitives = True
            continue
        if not in_primitives:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            flush()
            current_name = stripped[:-1]
            in_command = False
            continue
        if current_name is None:
            continue
        if indent == 4 and stripped.startswith("type:"):
            current_type = stripped.split(":", 1)[1].strip()
            continue
        if indent == 4 and stripped == "command:":
            in_command = True
            continue
        if in_command and indent == 6 and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_command[key.strip()] = float(value.strip())
    flush()
    if not primitives:
        raise SystemExit(f"no velocity primitives parsed from {path}")
    return _PrimitiveRegistry(
        block_size=int(block_size),
        command_dt_s=float(command_dt_s),
        primitives=primitives,
    )


def _load_safety_limits(path: Path) -> _SafetyLimits:
    values: dict[str, float] = {}
    in_delta = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if stripped.startswith("max_command_delta_per_tick:"):
            in_delta = True
            continue
        if in_delta and indent <= 4 and not stripped.startswith("max_command_delta_per_tick:"):
            in_delta = False
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if in_delta:
            values[f"delta_{key}"] = parsed
        else:
            values[key] = parsed
    return _SafetyLimits(
        min_vx_mps=float(values.get("min_vx_mps", -np.inf)),
        max_vx_mps=float(values.get("max_vx_mps", np.inf)),
        min_vy_mps=float(values.get("min_vy_mps", -np.inf)),
        max_vy_mps=float(values.get("max_vy_mps", np.inf)),
        max_yaw_rate_radps=float(values.get("max_yaw_rate_radps", np.inf)),
        max_delta_vx_mps=float(values.get("delta_vx_mps", np.inf)),
        max_delta_vy_mps=float(values.get("delta_vy_mps", np.inf)),
        max_delta_yaw_rate_radps=float(values.get("delta_yaw_rate_radps", np.inf)),
    )


def _expand_primitive_to_block(
    registry: _PrimitiveRegistry,
    primitive_name: str,
) -> np.ndarray:
    if primitive_name not in registry.primitives:
        raise KeyError(f"unknown primitive '{primitive_name}'")
    return np.full(
        (registry.block_size, 3),
        registry.primitives[primitive_name],
        dtype=np.float32,
    )


def _apply_safety_limits_single(
    requested: list[tuple[float, float, float]],
    previous: tuple[float, float, float],
    limits: _SafetyLimits,
    *,
    enforce_rate_limits: bool,
) -> tuple[list[tuple[float, float, float]], bool]:
    prev = np.asarray(previous, dtype=np.float32)
    abs_lo = np.asarray(
        [limits.min_vx_mps, limits.min_vy_mps, -limits.max_yaw_rate_radps],
        dtype=np.float32,
    )
    abs_hi = np.asarray(
        [limits.max_vx_mps, limits.max_vy_mps, limits.max_yaw_rate_radps],
        dtype=np.float32,
    )
    delta = np.asarray(
        [
            limits.max_delta_vx_mps,
            limits.max_delta_vy_mps,
            limits.max_delta_yaw_rate_radps,
        ],
        dtype=np.float32,
    )
    clipped_any = False
    executed = []
    for item in requested:
        req = np.asarray(item, dtype=np.float32)
        bounded = np.clip(req, abs_lo, abs_hi)
        clipped = bool(np.any(bounded != req))
        if enforce_rate_limits:
            after_rate = np.clip(bounded, prev - delta, prev + delta)
            clipped = clipped or bool(np.any(after_rate != bounded))
            bounded = after_rate
        clipped_any = clipped_any or clipped
        executed.append(tuple(float(value) for value in bounded.tolist()))
        prev = bounded
    return executed, bool(clipped_any)


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
    _seen_logits, geom_pred, steering_logits = model.score_queries_with_steering(
        hidden.unsqueeze(0),
        query_features,
    )
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
    predicted_steering_source = "geometry"
    predicted_steering = _steering_bucket_for_primitive(predicted_primitive)
    if steering_logits is not None:
        steering_index = int(torch.argmax(steering_logits.squeeze(0)).detach().cpu())
        predicted_steering = _STEERING_CLASSES[max(0, min(steering_index, 2))]
        predicted_primitive = _primitive_from_steering_prediction(
            predicted_steering,
            pred_bearing,
            pred_range,
            arc_threshold_rad=arc_threshold_rad,
            yaw_threshold_rad=yaw_threshold_rad,
            hold_range_m=hold_range_m,
        )
        predicted_steering_source = "steering_head"
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
        "predicted_steering": predicted_steering,
        "predicted_steering_source": predicted_steering_source,
        "route_steering": _steering_bucket_for_primitive(route_primitive),
    }


def _command_record(
    *,
    ablation: str,
    sequence_id: int,
    frame: Any,
    positive_objects: set[str],
    selected_object: str | None,
    selected_record: dict[str, Any] | None,
    primitive_name: str,
    route_primitive: str,
    requested: list[tuple[float, float, float]],
    executed: list[tuple[float, float, float]],
    clipped: bool,
    command_dt_s: float,
) -> dict[str, Any]:
    classification = "abstain"
    if selected_object is not None and not positive_objects:
        classification = "false_claim"
    elif selected_object is not None and selected_object in positive_objects:
        classification = "correct_target"
    elif selected_object is not None:
        classification = "wrong_object"
    elif positive_objects:
        classification = "missed_positive"

    result = {
        "ablation": ablation,
        "sequence_id": int(sequence_id),
        "seq_key": list(frame.seq_key),
        "episode_step": int(frame.episode_step),
        "positive_objects": sorted(positive_objects),
        "selected_object": selected_object,
        "classification": classification,
        "primitive_name": primitive_name,
        "route_primitive": route_primitive,
        "block_size": len(executed),
        "command_dt_s": float(command_dt_s),
        "requested_vx_body_mps": [float(item[0]) for item in requested],
        "requested_vy_body_mps": [float(item[1]) for item in requested],
        "requested_yaw_rate_radps": [float(item[2]) for item in requested],
        "executed_vx_body_mps": [float(item[0]) for item in executed],
        "executed_vy_body_mps": [float(item[1]) for item in executed],
        "executed_yaw_rate_radps": [float(item[2]) for item in executed],
        "clipped": bool(clipped),
    }
    if selected_record is not None:
        result.update(
            {
                "color": selected_record.get("color"),
                "true_bearing_rad": selected_record.get("true_bearing_rad"),
                "pred_bearing_rad": selected_record.get("pred_bearing_rad"),
                "true_range_m": selected_record.get("true_range_m"),
                "pred_range_m": selected_record.get("pred_range_m"),
                "target_primitive": selected_record.get("target_primitive"),
                "predicted_primitive": selected_record.get("predicted_primitive"),
                "target_steering": selected_record.get("target_steering"),
                "predicted_steering": selected_record.get("predicted_steering"),
                "predicted_steering_source": selected_record.get(
                    "predicted_steering_source"
                ),
                "route_steering": selected_record.get("route_steering"),
            }
        )
    return result


class _ReplayMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.selected_frames = 0
        self.correct_target = 0
        self.missed_positive = 0
        self.false_claim = 0
        self.wrong_object = 0
        self.target_steer_success = 0
        self.target_primitive_success = 0
        self.route_steer_success = 0
        self.route_primitive_success = 0
        self.non_hold_commands = 0
        self.negative_non_hold_commands = 0
        self.clipped_commands = 0

    def add(
        self,
        *,
        positive_objects: set[str],
        selected_object: str | None,
        selected_record: dict[str, Any] | None,
        primitive_name: str,
        clipped: bool,
    ) -> None:
        has_positive = bool(positive_objects)
        if has_positive:
            self.positive_frames += 1
        else:
            self.negative_frames += 1
        if primitive_name != "hold":
            self.non_hold_commands += 1
            if not has_positive:
                self.negative_non_hold_commands += 1
        if clipped:
            self.clipped_commands += 1
        if selected_object is None:
            if has_positive:
                self.missed_positive += 1
            return
        self.selected_frames += 1
        if selected_object not in positive_objects:
            if has_positive:
                self.wrong_object += 1
            else:
                self.false_claim += 1
            return
        self.correct_target += 1
        if selected_record is None:
            return
        if selected_record["predicted_steering"] == selected_record["target_steering"]:
            self.target_steer_success += 1
        if selected_record["predicted_primitive"] == selected_record["target_primitive"]:
            self.target_primitive_success += 1
        if selected_record["predicted_steering"] == selected_record["route_steering"]:
            self.route_steer_success += 1
        if selected_record["predicted_primitive"] == selected_record["route_primitive"]:
            self.route_primitive_success += 1

    def to_dict(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        command_count = len(records)
        selected_records = [
            record
            for record in records
            if record.get("classification") == "correct_target"
            and record.get("predicted_primitive") is not None
        ]
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "command_frame_count": float(command_count),
            "selected_frame_count": float(self.selected_frames),
            "correct_target_count": float(self.correct_target),
            "missed_positive_count": float(self.missed_positive),
            "false_claim_count": float(self.false_claim),
            "wrong_object_count": float(self.wrong_object),
            "target_recall": self.correct_target / max(1, self.positive_frames),
            "false_claim_rate": self.false_claim / max(1, self.negative_frames),
            "target_selection_precision": self.correct_target / max(1, self.selected_frames),
            "target_steering_success_count": float(self.target_steer_success),
            "target_steering_pipeline_success": self.target_steer_success
            / max(1, self.positive_frames),
            "target_primitive_pipeline_success": self.target_primitive_success
            / max(1, self.positive_frames),
            "route_steering_pipeline_success": self.route_steer_success
            / max(1, self.positive_frames),
            "route_primitive_pipeline_success": self.route_primitive_success
            / max(1, self.positive_frames),
            "non_hold_command_count": float(self.non_hold_commands),
            "non_hold_command_rate": self.non_hold_commands / max(1, command_count),
            "negative_non_hold_command_count": float(self.negative_non_hold_commands),
            "negative_non_hold_command_rate": self.negative_non_hold_commands
            / max(1, self.negative_frames),
            "clipped_command_count": float(self.clipped_commands),
            "clipped_command_rate": self.clipped_commands / max(1, command_count),
            "primitive_counts": dict(sorted(Counter(record["primitive_name"] for record in records).items())),
            "classification_counts": dict(
                sorted(Counter(record["classification"] for record in records).items())
            ),
            "selected_record_counts": _record_counts(selected_records),
        }


_STEERING_CLASSES = ["right", "forward", "left"]


def _primitive_from_steering_prediction(
    steering: str,
    pred_bearing: float,
    pred_range: float,
    *,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> str:
    if float(hold_range_m) > 0.0 and float(pred_range) <= float(hold_range_m):
        return "hold"
    if steering == "left":
        return "yaw_left" if pred_bearing >= yaw_threshold_rad else "arc_left"
    if steering == "right":
        return "yaw_right" if pred_bearing <= -yaw_threshold_rad else "arc_right"
    return "forward_medium"


if __name__ == "__main__":
    raise SystemExit(main())
