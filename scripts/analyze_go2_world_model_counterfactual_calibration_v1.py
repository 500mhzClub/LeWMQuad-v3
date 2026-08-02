#!/usr/bin/env python3
"""Receipt-only analyzer for the 160-branch Go2 counterfactual calibration.

The analyzer never opens an RGB leaf, simulator input, or checkpoint.  It
starts from one caller-bound, checker-validated physics collection and derives
the physical-rank tolerances and terminal calibration decision.  The resulting
receipt is an input to the separate pilot join; it grants no runtime authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker  # noqa: E402


CALIBRATION_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_receipt_v1"
)
PHYSICAL_RANK_CONTRACT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_physical_rank_contract_v1"
)
TOLERANCE_DERIVATION_SCHEMA = (
    "lewm_go2_world_model_counterfactual_tolerance_derivation_v1"
)
RESOURCE_MEASUREMENTS_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_resource_measurements_v1"
)
ANALYZER_SOURCE_NAME = "calibration_analyzer"
JOINER_SOURCE_NAME = "pilot_joiner"
CHECKER_SOURCE_NAME = "checker"
NUMERICAL_FLOOR_M = 1.0e-6
ACTION_COUNT = 9
CALIBRATION_STATE_COUNT = 16
CALIBRATION_SCENE_COUNT = 8
CALIBRATION_BRANCH_COUNT = 160
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CalibrationAnalysisError(RuntimeError):
    """Raised before an invalid collection can mint a calibration receipt."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _binding(path: Path) -> dict[str, object]:
    selected = Path(path).resolve(strict=True)
    if selected.is_symlink() or not selected.is_file():
        raise CalibrationAnalysisError(f"bound input is not a regular file: {selected}")
    raw = selected.read_bytes()
    return {
        "path": str(selected),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _validate_binding(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise CalibrationAnalysisError(f"{name} binding is malformed")
    path = value["path"]
    digest = value["file_sha256"]
    byte_count = value["byte_count"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or str(Path(path)) != path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise CalibrationAnalysisError(f"{name} binding is malformed")
    return {
        "path": path,
        "file_sha256": digest,
        "byte_count": byte_count,
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationAnalysisError(f"{name} must be finite")
    return result


def _strict_json_loads(raw: bytes, *, name: str) -> dict[str, object]:
    def reject_constant(token: str) -> None:
        raise CalibrationAnalysisError(f"{name} contains non-finite token {token}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationAnalysisError(f"{name} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationAnalysisError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise CalibrationAnalysisError(f"{name} must be a JSON object")
    return value


def validate_calibration_receipt_v1(
    value: object,
    *,
    verify_external_bindings: bool,
) -> dict[str, object]:
    """Validate the complete analyzer receipt and optionally re-open its sources."""

    expected = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "calibration_id",
        "role",
        "train_eval_scenes_accessed",
        "decision",
        "calibration_collection_receipt",
        "calibration_contract",
        "repeatability_analysis",
        "physics_validation",
        "visual_validation",
        "resource_measurements",
        "analyzer_binding",
        "checker_binding",
        "source_bindings",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CalibrationAnalysisError("calibration receipt field set changed")
    if (
        value["schema"] != CALIBRATION_RECEIPT_SCHEMA
        or value["status"] != "COMPLETE"
        or value["citable_as_scientific_evidence"] is not False
        or value["authorizes_retry_or_resume"] is not False
        or not isinstance(value["calibration_id"], str)
        or not value["calibration_id"]
        or value["role"] != "calibration"
        or value["train_eval_scenes_accessed"] is not False
        or value["decision"] not in {
            "FREEZE_PILOT_CONTRACT",
            "STOP_SOURCE_REDESIGN",
        }
    ):
        raise CalibrationAnalysisError("calibration receipt status/identity changed")
    external = _validate_binding(
        value["calibration_collection_receipt"],
        name="calibration collection",
    )
    analyzer = _validate_binding(value["analyzer_binding"], name="analyzer")
    checker_source = _validate_binding(value["checker_binding"], name="checker")
    contract = value["calibration_contract"]
    contract_fields = {
        "schema",
        "excluded_scene_ids",
        "progress_tolerance_m",
        "path_length_tolerance_m",
        "quantization_rule",
        "lexicographic_key",
        "proxy_fields_excluded",
        "tolerance_derivation",
    }
    if not isinstance(contract, Mapping) or set(contract) != contract_fields:
        raise CalibrationAnalysisError("calibration contract field set changed")
    excluded = contract["excluded_scene_ids"]
    if (
        contract["schema"] != PHYSICAL_RANK_CONTRACT_SCHEMA
        or not isinstance(excluded, list)
        or len(excluded) != CALIBRATION_SCENE_COUNT
        or excluded != sorted(excluded)
        or len(set(excluded)) != len(excluded)
        or any(not isinstance(item, str) or not item for item in excluded)
        or _finite(
            contract["progress_tolerance_m"], name="progress tolerance"
        ) <= 0.0
        or _finite(
            contract["path_length_tolerance_m"], name="path-length tolerance"
        ) <= 0.0
        or contract["quantization_rule"]
        != "sign(x)*floor(abs(x)/t+0.5)"
        or contract["lexicographic_key"] != [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ]
        or contract["proxy_fields_excluded"] is not True
    ):
        raise CalibrationAnalysisError("calibration contract semantics changed")
    derivation = contract["tolerance_derivation"]
    if (
        not isinstance(derivation, Mapping)
        or set(derivation) != {
            "schema",
            "method",
            "minimum_numerical_resolution_m",
            "repeat_controls",
            "repeated_action_ids",
            "all_requested_primitives_covered",
            "deterministic_repeat_gate_passed",
            "empirical_noise_scale_estimated",
        }
        or derivation["schema"] != TOLERANCE_DERIVATION_SCHEMA
        or derivation["method"]
        != "fixed_numerical_floor_after_exact_deterministic_repeat_gate"
        or derivation["minimum_numerical_resolution_m"] != NUMERICAL_FLOOR_M
        or derivation["repeat_controls"] != CALIBRATION_STATE_COUNT
        or derivation["repeated_action_ids"]
        != [index % ACTION_COUNT for index in range(CALIBRATION_STATE_COUNT)]
        or derivation["all_requested_primitives_covered"] is not True
        or derivation["deterministic_repeat_gate_passed"] is not True
        or derivation["empirical_noise_scale_estimated"] is not False
    ):
        raise CalibrationAnalysisError("calibration tolerance derivation changed")
    if (
        contract["progress_tolerance_m"] != NUMERICAL_FLOOR_M
        or contract["path_length_tolerance_m"] != NUMERICAL_FLOOR_M
    ):
        raise CalibrationAnalysisError("calibration numerical floor changed")
    repeatability = value["repeatability_analysis"]
    physics = value["physics_validation"]
    visual = value["visual_validation"]
    resources = value["resource_measurements"]
    if (
        not isinstance(repeatability, Mapping)
        or repeatability.get("repeat_controls") != CALIBRATION_STATE_COUNT
        or repeatability.get("repeated_action_ids")
        != [index % ACTION_COUNT for index in range(CALIBRATION_STATE_COUNT)]
        or repeatability.get("all_requested_primitives_covered") is not True
        or repeatability.get("interpretation")
        != "deterministic_replay_gate_not_empirical_noise_estimate"
        or repeatability.get("empirical_noise_scale_estimated") is not False
        or repeatability.get("executed_command_tapes_exact") is not True
        or repeatability.get("physical_trajectories_exact") is not True
        or repeatability.get("stored_rgb_exact") is not True
        or not isinstance(physics, Mapping)
        or physics.get("receipt_checker_passed") is not True
        or physics.get("common_prefix_exact") is not True
        or physics.get("nine_unique_executed_tapes_per_state") is not True
        or physics.get("physics_validated_for_branch_outcomes") is not True
        or not isinstance(visual, Mapping)
        or visual.get("camera_quality_receipts_passed") is not True
        or visual.get("endpoint_pose_replay_bound") is not True
        or visual.get("visual_domain_fidelity_claimed") is not False
        or visual.get("eligible_for_physical_branch_evaluation") is not True
        or visual.get("eligible_for_visual_domain_parity_claim") is not False
    ):
        raise CalibrationAnalysisError("calibration validation evidence changed")
    if not isinstance(resources, Mapping) or set(resources) != {
        "schema",
        "stored_rgb_png",
        "stage_wall_seconds",
        "outcome_counts",
        "gpu_peak_memory_measurement_scope",
    }:
        raise CalibrationAnalysisError("calibration resource measurements changed")
    stored_rgb = resources["stored_rgb_png"]
    stages = resources["stage_wall_seconds"]
    outcomes = resources["outcome_counts"]
    if (
        resources["schema"] != RESOURCE_MEASUREMENTS_SCHEMA
        or resources["gpu_peak_memory_measurement_scope"]
        != "external_terminal_required_not_observed_by_analyzer"
        or not isinstance(stored_rgb, Mapping)
        or set(stored_rgb) != {
            "context_frames",
            "context_bytes",
            "target_frames",
            "target_bytes",
            "total_frames",
            "total_bytes",
            "raw_uncompressed_rgb_ceiling_bytes",
        }
        or stored_rgb["context_frames"] != 48
        or stored_rgb["target_frames"] != 160
        or stored_rgb["total_frames"] != 208
        or stored_rgb["total_frames"]
        != stored_rgb["context_frames"] + stored_rgb["target_frames"]
        or any(
            type(stored_rgb[name]) is not int or stored_rgb[name] < 0
            for name in ("context_bytes", "target_bytes", "total_bytes")
        )
        or stored_rgb["total_bytes"]
        != stored_rgb["context_bytes"] + stored_rgb["target_bytes"]
        or stored_rgb["raw_uncompressed_rgb_ceiling_bytes"] != 208 * 224 * 224 * 3
        or not isinstance(stages, Mapping)
        or set(stages) != {
            "collection_external_wall_seconds",
            "physics_scene_build_wall_seconds",
            "render_scene_build_wall_seconds",
            "common_prefix_step_wall_seconds",
            "branch_step_wall_seconds",
            "native_render_wall_seconds",
            "camera_quality_resize_wall_seconds",
            "png_encode_write_hash_wall_seconds",
            "post_lockstep_receipt_wall_seconds",
            "summed_scene_total_wall_seconds",
        }
        or any(_finite(number, name=f"stage {name}") < 0.0 for name, number in stages.items())
        or not isinstance(outcomes, Mapping)
        or set(outcomes) != {
            "complete_all_nine_action_groups",
            "executed_tape_distinct_groups",
            "prebranch_exact_groups",
            "clipped_candidate_branches",
            "fallen_candidate_branches",
            "tipped_candidate_branches",
            "camera_invalid_frames",
            "incomplete_states",
        }
        or outcomes["complete_all_nine_action_groups"] != 16
        or outcomes["executed_tape_distinct_groups"] != 16
        or outcomes["prebranch_exact_groups"] != 16
        or outcomes["camera_invalid_frames"] != 0
        or outcomes["incomplete_states"] != 0
        or any(type(number) is not int or number < 0 for number in outcomes.values())
    ):
        raise CalibrationAnalysisError("calibration resource evidence is invalid")
    minimum_classes = physics.get("minimum_physical_rank_classes_per_state")
    expected_decision = (
        "FREEZE_PILOT_CONTRACT"
        if type(minimum_classes) is int and minimum_classes >= 2
        else "STOP_SOURCE_REDESIGN"
    )
    if value["decision"] != expected_decision:
        raise CalibrationAnalysisError("calibration decision disagrees with evidence")
    sources = value["source_bindings"]
    expected_names = [CHECKER_SOURCE_NAME, ANALYZER_SOURCE_NAME, JOINER_SOURCE_NAME]
    if (
        not isinstance(sources, list)
        or [entry.get("name") if isinstance(entry, Mapping) else None for entry in sources]
        != expected_names
    ):
        raise CalibrationAnalysisError("calibration source closure changed")
    normalized_sources = []
    for entry in sources:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "binding"}:
            raise CalibrationAnalysisError("calibration source binding changed")
        normalized_sources.append({
            "name": entry["name"],
            "binding": _validate_binding(
                entry["binding"], name=f"source {entry['name']}"
            ),
        })
    if normalized_sources[0]["binding"] != checker_source or normalized_sources[1][
        "binding"
    ] != analyzer:
        raise CalibrationAnalysisError("calibration source aliases changed")
    if verify_external_bindings:
        for name, binding in (
            ("calibration collection", external),
            *(
                (f"source {entry['name']}", entry["binding"])
                for entry in normalized_sources
            ),
        ):
            if _binding(Path(str(binding["path"]))) != binding:
                raise CalibrationAnalysisError(f"{name} changed after calibration")
    return dict(value)


def load_bound_calibration_receipt_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    if _SHA256.fullmatch(expected_sha256) is None or expected_byte_count <= 0:
        raise CalibrationAnalysisError("caller calibration binding is malformed")
    actual = _binding(path)
    if (
        actual["file_sha256"] != expected_sha256
        or actual["byte_count"] != expected_byte_count
    ):
        raise CalibrationAnalysisError("caller calibration binding changed")
    raw = Path(actual["path"]).read_bytes()
    receipt = validate_calibration_receipt_v1(
        _strict_json_loads(raw, name="calibration receipt"),
        verify_external_bindings=True,
    )
    if _binding(path) != actual:
        raise CalibrationAnalysisError("calibration receipt changed while loaded")
    return receipt, actual, raw


def _quantize(value: float, tolerance: float) -> int:
    magnitude = math.floor(abs(value) / tolerance + 0.5)
    return magnitude if value >= 0.0 else -magnitude


def _physical_key(branch: Mapping[str, Any], tolerances: Mapping[str, float]) -> tuple[int, ...]:
    return (
        int(bool(branch["physical_fell"])),
        int(bool(branch["physical_tipped"])),
        -_quantize(
            _finite(branch["physical_target_progress_m"], name="target progress"),
            tolerances["progress_tolerance_m"],
        ),
        _quantize(
            _finite(branch["physical_path_length_m"], name="path length"),
            tolerances["path_length_tolerance_m"],
        ),
    )


def derive_calibration_receipt_v1(
    collection: Mapping[str, Any],
    *,
    collection_binding: Mapping[str, object],
    analyzer_binding: Mapping[str, object],
    checker_binding: Mapping[str, object],
    joiner_binding: Mapping[str, object],
) -> dict[str, object]:
    """Derive a deterministic calibration receipt from validated receipts."""

    normalized_collection_binding = _validate_binding(
        collection_binding, name="calibration collection"
    )
    normalized_analyzer_binding = _validate_binding(
        analyzer_binding, name="calibration analyzer"
    )
    normalized_checker_binding = _validate_binding(
        checker_binding, name="receipt checker"
    )
    normalized_joiner_binding = _validate_binding(
        joiner_binding, name="pilot joiner"
    )
    if collection.get("purpose") != "sizing_calibration_only":
        raise CalibrationAnalysisError(
            "calibration analyzer requires a sizing_calibration_only collection"
        )
    counts = collection.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("scenes") != CALIBRATION_SCENE_COUNT
        or counts.get("states") != CALIBRATION_STATE_COUNT
        or counts.get("total_branches") != CALIBRATION_BRANCH_COUNT
        or counts.get("candidate_branches") != CALIBRATION_STATE_COUNT * ACTION_COUNT
        or counts.get("sentinel_branches") != CALIBRATION_STATE_COUNT
        or counts.get("roles") != {"calibration": CALIBRATION_STATE_COUNT}
    ):
        raise CalibrationAnalysisError("calibration collection size contract changed")
    states = collection.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise CalibrationAnalysisError("calibration states are absent")

    progress_repeat_deltas: list[float] = []
    path_repeat_deltas: list[float] = []
    endpoint_position_repeat_deltas: list[float] = []
    endpoint_quaternion_repeat_deltas: list[float] = []
    calibration_scene_ids: set[str] = set()
    repeated_action_ids: list[int] = []
    clipped_candidate_count = 0
    for state_index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise CalibrationAnalysisError(f"calibration state {state_index} is malformed")
        state_document = state.get("state")
        branches = state.get("branches")
        if (
            not isinstance(state_document, Mapping)
            or not isinstance(branches, Sequence)
            or isinstance(branches, (str, bytes))
            or len(branches) != ACTION_COUNT + 1
        ):
            raise CalibrationAnalysisError("calibration branch grid changed")
        calibration_scene_ids.add(str(state_document.get("scene_id")))
        sentinel = branches[-1]
        if not isinstance(sentinel, Mapping):
            raise CalibrationAnalysisError("calibration repeat branch is malformed")
        duplicate_action_id = sentinel.get("action_id")
        if type(duplicate_action_id) is not int or not 0 <= duplicate_action_id < ACTION_COUNT:
            raise CalibrationAnalysisError("calibration repeat action ID is invalid")
        repeated_action_ids.append(duplicate_action_id)
        candidate = branches[duplicate_action_id]
        if not isinstance(candidate, Mapping):
            raise CalibrationAnalysisError("calibration repeated candidate is malformed")
        progress_repeat_deltas.append(abs(
            _finite(candidate["physical_target_progress_m"], name="candidate progress")
            - _finite(sentinel["physical_target_progress_m"], name="sentinel progress")
        ))
        path_repeat_deltas.append(abs(
            _finite(candidate["physical_path_length_m"], name="candidate path length")
            - _finite(sentinel["physical_path_length_m"], name="sentinel path length")
        ))
        candidate_endpoint = candidate.get("endpoint_state")
        sentinel_endpoint = sentinel.get("endpoint_state")
        if not isinstance(candidate_endpoint, Mapping) or not isinstance(
            sentinel_endpoint, Mapping
        ):
            raise CalibrationAnalysisError("calibration endpoint state is absent")
        for field, output in (
            ("base_pos_world", endpoint_position_repeat_deltas),
            ("base_quat_wxyz", endpoint_quaternion_repeat_deltas),
        ):
            left = candidate_endpoint.get(field)
            right = sentinel_endpoint.get(field)
            if (
                not isinstance(left, Sequence)
                or isinstance(left, (str, bytes))
                or not isinstance(right, Sequence)
                or isinstance(right, (str, bytes))
                or len(left) != len(right)
            ):
                raise CalibrationAnalysisError(f"repeat endpoint {field} changed shape")
            output.append(max(
                abs(_finite(a, name=field) - _finite(b, name=field))
                for a, b in zip(left, right, strict=True)
            ))
        clipped_candidate_count += sum(
            int(branch.get("clipped") is True) for branch in branches[:ACTION_COUNT]
        )

    if len(calibration_scene_ids) != CALIBRATION_SCENE_COUNT:
        raise CalibrationAnalysisError("calibration scene identities repeat")
    if set(repeated_action_ids) != set(range(ACTION_COUNT)):
        raise CalibrationAnalysisError(
            "the 16 repeat controls do not cover all nine requested primitives"
        )
    max_progress_repeat = max(progress_repeat_deltas)
    max_path_repeat = max(path_repeat_deltas)
    max_endpoint_position_repeat = max(endpoint_position_repeat_deltas)
    max_endpoint_quaternion_repeat = max(endpoint_quaternion_repeat_deltas)
    if any(
        value != 0.0
        for value in (
            max_progress_repeat,
            max_path_repeat,
            max_endpoint_position_repeat,
            max_endpoint_quaternion_repeat,
        )
    ):
        raise CalibrationAnalysisError(
            "repeat controls are an exact deterministic gate and must have zero delta"
        )
    tolerances = {
        "progress_tolerance_m": NUMERICAL_FLOOR_M,
        "path_length_tolerance_m": NUMERICAL_FLOOR_M,
    }
    class_counts = [
        len({
            _physical_key(branch, tolerances)
            for branch in state["branches"][:ACTION_COUNT]
        })
        for state in states
    ]
    decision = (
        "FREEZE_PILOT_CONTRACT"
        if min(class_counts) >= 2
        else "STOP_SOURCE_REDESIGN"
    )
    frame_receipts = collection.get("frame_receipts")
    if not isinstance(frame_receipts, Mapping) or len(frame_receipts) != 208:
        raise CalibrationAnalysisError("calibration frame receipt grid changed")
    context_bytes = 0
    target_bytes = 0
    context_frames = 0
    target_frames = 0
    for frame in frame_receipts.values():
        if not isinstance(frame, Mapping):
            raise CalibrationAnalysisError("calibration frame receipt is malformed")
        identity = frame.get("frame_identity")
        byte_count = frame.get("byte_count")
        if (
            not isinstance(identity, str)
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise CalibrationAnalysisError("calibration frame byte receipt changed")
        if ":context:" in identity:
            context_frames += 1
            context_bytes += byte_count
        else:
            target_frames += 1
            target_bytes += byte_count
    if context_frames != 48 or target_frames != 160:
        raise CalibrationAnalysisError("calibration context/target frame split changed")

    collection_document = collection.get("document")
    if not isinstance(collection_document, Mapping):
        raise CalibrationAnalysisError("calibration collection document is absent")
    scene_metrics = collection_document.get("scene_metrics")
    if not isinstance(scene_metrics, list) or len(scene_metrics) != 8:
        raise CalibrationAnalysisError("calibration scene timing panel changed")
    timing_fields = (
        "physics_build_wall_seconds",
        "render_scene_build_wall_seconds",
        "common_prefix_step_wall_seconds",
        "branch_step_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "scene_total_wall_seconds",
    )
    timing_sums = {name: 0.0 for name in timing_fields}
    for scene_metric in scene_metrics:
        if not isinstance(scene_metric, Mapping):
            raise CalibrationAnalysisError("calibration scene timing row changed")
        for name in timing_fields:
            timing_sums[name] += _finite(
                scene_metric.get(name), name=f"scene timing {name}"
            )
    collection_wall_seconds = _finite(
        collection_document.get("collection_wall_seconds"),
        name="collection wall seconds",
    )
    if collection_wall_seconds < 0.0:
        raise CalibrationAnalysisError("collection wall time cannot be negative")
    resource_measurements = {
        "schema": RESOURCE_MEASUREMENTS_SCHEMA,
        "stored_rgb_png": {
            "context_frames": context_frames,
            "context_bytes": context_bytes,
            "target_frames": target_frames,
            "target_bytes": target_bytes,
            "total_frames": context_frames + target_frames,
            "total_bytes": context_bytes + target_bytes,
            "raw_uncompressed_rgb_ceiling_bytes": 208 * 224 * 224 * 3,
        },
        "stage_wall_seconds": {
            "collection_external_wall_seconds": collection_wall_seconds,
            "physics_scene_build_wall_seconds": timing_sums[
                "physics_build_wall_seconds"
            ],
            "render_scene_build_wall_seconds": timing_sums[
                "render_scene_build_wall_seconds"
            ],
            "common_prefix_step_wall_seconds": timing_sums[
                "common_prefix_step_wall_seconds"
            ],
            "branch_step_wall_seconds": timing_sums["branch_step_wall_seconds"],
            "native_render_wall_seconds": timing_sums[
                "native_render_wall_seconds"
            ],
            "camera_quality_resize_wall_seconds": timing_sums[
                "camera_quality_resize_wall_seconds"
            ],
            "png_encode_write_hash_wall_seconds": timing_sums[
                "png_encode_write_hash_wall_seconds"
            ],
            "post_lockstep_receipt_wall_seconds": timing_sums[
                "post_lockstep_receipt_wall_seconds"
            ],
            "summed_scene_total_wall_seconds": timing_sums[
                "scene_total_wall_seconds"
            ],
        },
        "outcome_counts": {
            "complete_all_nine_action_groups": CALIBRATION_STATE_COUNT,
            "executed_tape_distinct_groups": CALIBRATION_STATE_COUNT,
            "prebranch_exact_groups": CALIBRATION_STATE_COUNT,
            "clipped_candidate_branches": clipped_candidate_count,
            "fallen_candidate_branches": sum(
                int(branch["physical_fell"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "tipped_candidate_branches": sum(
                int(branch["physical_tipped"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "camera_invalid_frames": 0,
            "incomplete_states": 0,
        },
        "gpu_peak_memory_measurement_scope": (
            "external_terminal_required_not_observed_by_analyzer"
        ),
    }
    calibration_contract = {
        "schema": PHYSICAL_RANK_CONTRACT_SCHEMA,
        "excluded_scene_ids": sorted(calibration_scene_ids),
        **tolerances,
        "quantization_rule": "sign(x)*floor(abs(x)/t+0.5)",
        "lexicographic_key": [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ],
        "proxy_fields_excluded": True,
        "tolerance_derivation": {
            "schema": TOLERANCE_DERIVATION_SCHEMA,
            "method": "fixed_numerical_floor_after_exact_deterministic_repeat_gate",
            "minimum_numerical_resolution_m": NUMERICAL_FLOOR_M,
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": True,
            "deterministic_repeat_gate_passed": True,
            "empirical_noise_scale_estimated": False,
        },
    }
    return {
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "calibration_id": str(collection["document"]["attempt_id"]),
        "role": "calibration",
        "train_eval_scenes_accessed": False,
        "decision": decision,
        "calibration_collection_receipt": normalized_collection_binding,
        "calibration_contract": calibration_contract,
        "repeatability_analysis": {
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": True,
            "interpretation": (
                "deterministic_replay_gate_not_empirical_noise_estimate"
            ),
            "empirical_noise_scale_estimated": False,
            "progress_max_abs_delta_m": max_progress_repeat,
            "path_length_max_abs_delta_m": max_path_repeat,
            "endpoint_position_max_abs_delta_m": max_endpoint_position_repeat,
            "endpoint_quaternion_max_abs_delta": max_endpoint_quaternion_repeat,
            "executed_command_tapes_exact": True,
            "physical_trajectories_exact": True,
            "stored_rgb_exact": True,
        },
        "physics_validation": {
            "receipt_checker_passed": True,
            "common_prefix_exact": True,
            "nine_unique_executed_tapes_per_state": True,
            "minimum_physical_rank_classes_per_state": min(class_counts),
            "maximum_physical_rank_classes_per_state": max(class_counts),
            "clipped_candidate_branches": clipped_candidate_count,
            "physics_validated_for_branch_outcomes": True,
        },
        "visual_validation": {
            "camera_quality_receipts_passed": True,
            "endpoint_pose_replay_bound": True,
            "visual_domain_fidelity_claimed": False,
            "eligible_for_physical_branch_evaluation": True,
            "eligible_for_visual_domain_parity_claim": False,
        },
        "resource_measurements": resource_measurements,
        "analyzer_binding": normalized_analyzer_binding,
        "checker_binding": normalized_checker_binding,
        "source_bindings": [
            {"name": CHECKER_SOURCE_NAME, "binding": normalized_checker_binding},
            {"name": ANALYZER_SOURCE_NAME, "binding": normalized_analyzer_binding},
            {"name": JOINER_SOURCE_NAME, "binding": normalized_joiner_binding},
        ],
    }


def _write_exclusive(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    selected = Path(path)
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration receipt: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        selected,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _binding(selected)


def analyze_calibration(
    *,
    collection_path: Path,
    expected_collection_sha256: str,
    expected_collection_byte_count: int,
    output_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    collection = checker.load_bound_collection_receipts(
        collection_path,
        expected_file_sha256=expected_collection_sha256,
        expected_byte_count=expected_collection_byte_count,
    )
    collection_binding = _binding(collection_path)
    receipt = derive_calibration_receipt_v1(
        collection,
        collection_binding=collection_binding,
        analyzer_binding=_binding(Path(__file__)),
        checker_binding=_binding(Path(checker.__file__)),
        joiner_binding=_binding(
            REPO_ROOT / "scripts/join_go2_world_model_counterfactual_pilot_v1.py"
        ),
    )
    output_binding = _write_exclusive(output_path, receipt)
    return receipt, output_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--expected-collection-sha256", required=True)
    parser.add_argument("--expected-collection-byte-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt, binding = analyze_calibration(
        collection_path=args.collection,
        expected_collection_sha256=args.expected_collection_sha256,
        expected_collection_byte_count=args.expected_collection_byte_count,
        output_path=args.output,
    )
    print(json.dumps({
        "decision": receipt["decision"],
        "calibration_receipt": binding,
    }, sort_keys=True))
    return 0 if receipt["decision"] == "FREEZE_PILOT_CONTRACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_RECEIPT_SCHEMA",
    "RESOURCE_MEASUREMENTS_SCHEMA",
    "CalibrationAnalysisError",
    "derive_calibration_receipt_v1",
    "load_bound_calibration_receipt_v1",
    "validate_calibration_receipt_v1",
]
