"""Pure validators and reducer for physical graph-edge handoff qualification.

The functions in this module consume ordinary mappings and row sequences.
They perform no file I/O, checkpoint loading, simulation, rendering, encoding,
inference, or training.  NPZ bytes are inspected by an independent caller and
passed back as a strict inspection projection.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as C


class PhysicalGraphEdgeHandoffMetricsError(ValueError):
    """Raised when raw physical-handoff evidence is incomplete or inconsistent."""


FILE_BINDING_FIELDS = {"path", "bytes", "sha256"}
SLICE_BINDING_FIELDS = {
    "file_path", "file_sha256", "member", "start", "stop", "slice_sha256"
}
NPZ_INSPECTION_FIELDS = {"path", "bytes", "sha256", "members"}
NPZ_MEMBER_INSPECTION_FIELDS = {
    "descr", "digest_dtype", "shape", "c_contiguous", "object_dtype",
    "member_sha256", "row_or_slice_sha256s", "offset_values",
}

PANEL_FIELDS = {
    "schema", "experiment_id", "constructed_set_caveat", "identity_domain",
    "prior_exclusion_evidence", "prospective_pool_selection",
    "physical_runtime_environment", "states", "content_digest",
}
PANEL_STATE_FIELDS = {
    "state_id", "scene_id", "episode_id", "graph_id", "family", "turn_direction",
    "route_direction", "passage_width_id", "port_distance_id", "stratum_index",
    "candidate_spec_id", "role", "procedural_seed", "geometry_sha256", "source_node_id",
    "target_node_id", "directed_edge_id", "goal_reachable", "eligible",
    "eligibility_evidence",
}
ELIGIBILITY_FIELDS = {
    "snapshot_complete", "exact_reset_fixture_passed", "teacher_trace_contact_free",
    "teacher_valid", "directed_port_defined", "current_rgb_valid",
    "graph_edge_physically_executable",
}
EXCLUSION_EVIDENCE_FIELDS = {
    "authority_digest", "checked_before_simulator_creation", "scene_overlap_count",
    "scene_hash_overlap_count", "state_or_episode_overlap_count", "seed_overlap_count",
    "path_or_geometry_overlap_count", "structured_path_overlap_count", "all_zero",
}
POOL_SELECTION_FIELDS = {
    "candidate_spec_count", "candidate_specs_sha256", "selection_rule",
    "teacher_scan_completed_before_role_assignment", "ranker_or_fanout_outcomes_opened",
    "per_family_scanned_counts", "per_family_teacher_eligible_counts",
    "selected_candidate_spec_ids_by_family_and_stratum", "rejection_reason_counts",
    "qualification_rows", "qualification_projection_sha256",
    "selected_state_ids", "complete",
}
POOL_QUALIFICATION_FIELDS = {
    "candidate_spec_id", "state_id", "family", "stratum_index", "variant_index",
    "canonical_spec_sha256", "teacher_trace_id", "teacher_trace_index",
    "teacher_trace_slice_sha256", "initial_decision_state_sha256",
    "goal_reachable", "teacher_trace_contact_free", "teacher_left_source_region",
    "teacher_crossed_directed_port", "teacher_positive_route_progress",
    "teacher_competing_port_entered", "teacher_normal_positive",
    "teacher_within_lateral_bounds", "teacher_dwell_satisfied",
    "directed_port_defined", "current_rgb_valid", "graph_edge_physically_executable",
    "teacher_valid", "qualified", "rejection_reason", "selection_key_sha256",
    "rank_within_stratum", "selected",
}

SPLIT_FIELDS = {
    "schema", "experiment_id", "assignment_algorithm",
    "complete_eligible_population_sha256", "heldout_outcomes_opened_before_assignment",
    "role_counts", "family_role_counts", "assignments", "content_digest",
}
SPLIT_ASSIGNMENT_FIELDS = {"state_id", "role", "assignment_sha256", "rank_within_family"}

GRAPH_FIELDS = {"schema", "experiment_id", "graphs", "content_digest"}
GRAPH_RECORD_FIELDS = {
    "state_id", "graph_id", "source_node_id", "target_node_id", "directed_edge_id",
    "nodes", "edges",
}
GRAPH_NODE_FIELDS = {"node_id", "node_kind", "centre_world", "boundary_polygon_world"}
GRAPH_EDGE_FIELDS = {
    "edge_id", "source_node_id", "target_node_id", "port_label",
    "route_polyline_world", "edge_length_m", "oracle_reachable",
    "physically_executable", "edge_kind", "opening_segment_world",
    "opening_normal_world",
}

SNAPSHOT_INDEX_FIELDS = {
    "schema", "experiment_id", "snapshots_file", "reset_fixture", "records",
    "content_digest",
}
RESET_FIXTURE_FIELDS = {
    "fixture_id", "serialized_restore_used", "clone_equivalence_used", "passed"
}
SNAPSHOT_RECORD_FIELDS = {
    "state_id", "snapshot_id", "snapshot_payload", "array_row_index",
    "source_teacher_trace_id", "teacher_initial_state_sha256",
    "serialized_solver_state_sha256", "serialized_controller_state_sha256",
    "serialized_rng_state_sha256", "policy_last_action_sha256",
    "base_pose_world_sha256", "base_twist_world_sha256",
    "joint_position_sha256", "joint_velocity_sha256",
    "camera_world_transform_sha256",
    "capture_timestamp_s", "controller_observation_sha256",
    "previous_policy_action_sha256",
    "torch_cpu_rng_state_sha256", "torch_device_rng_state_sha256s",
    "torch_device_count",
    "previous_applied_command_sha256", "command_history_sha256",
    "low_level_policy_state_sha256",
    "control_history_sha256", "solver_field_inventory", "controller_field_inventory",
    "rng_field_inventory", "reset_trials", "reset_pair_comparison",
}
RESET_TRIAL_FIELDS = {
    "trial_index", "serialized_restore_used", "restored_snapshot_sha256",
    "post_restore_state_sha256", "current_rgb_sha256", "base_pose_world",
    "joint_position_sha256", "joint_velocity_sha256", "controller_state_sha256",
    "rng_state_sha256", "trace_index", "trace_slice",
    "trace_array_slice_sha256s", "requested_command_sequence_sha256",
    "post_slew_applied_command_sequence_sha256", "contact_sequence_sha256",
    "termination_reason", "stuck",
}
RESET_PAIR_COMPARISON_FIELDS = {
    "state_id", "trial_indices", "physics_sample_count",
    "maximum_base_position_error_m", "maximum_base_quaternion_component_error",
    "maximum_base_twist_error", "maximum_joint_position_error_rad",
    "maximum_joint_velocity_error_rad_s", "exact_member_equal",
    "endpoint_position_error_m", "endpoint_heading_error_rad",
    "termination_reason_equal", "stuck_equal", "passed",
}

TEACHER_INDEX_FIELDS = {
    "schema", "experiment_id", "traces_file", "records", "content_digest"
}
TEACHER_RECORD_FIELDS = {
    "candidate_spec_id", "state_id", "family", "stratum_index", "variant_index",
    "canonical_spec_sha256", "teacher_trace_id", "trace_index", "selected",
    "initial_decision_state_sha256", "current_rgb_sha256", "current_rgb_valid",
    "goal_reachable",
    "directed_port_defined", "graph_edge_physically_executable",
    "source_node_id", "target_node_id",
    "directed_edge_id", "trace_slice", "trace_array_slice_sha256s", "sample_count",
    "contact_free", "left_source_region", "first_source_exit_sample_index",
    "first_crossing_sample_index",
    "crossing_segment_fraction", "crossed_directed_port", "positive_route_progress",
    "route_progress_m",
    "crossing_directed_normal_dot", "crossing_lateral_fraction",
    "crossing_velocity_world_xy", "crossing_velocity_heading_world_rad",
    "beyond_port_consecutive_physics_samples", "target_entered_before_dwell_complete",
    "competing_port_entered", "reached_target_node", "first_target_entry_sample_index",
    "where_reached", "endpoint_lateral_error_m", "endpoint_angular_error_rad",
    "successor_viable", "stuck", "teacher_valid",
}

EDGE_PORT_INDEX_FIELDS = {"schema", "experiment_id", "records", "content_digest"}
EDGE_PORT_RECORD_FIELDS = {
    "state_id", "directed_edge_id", "source_node_id", "target_node_id",
    "teacher_trace_id", "source_boundary_polygon_world", "route_polyline_world",
    "crossing_sample_before", "crossing_sample_after", "crossing_fraction",
    "teacher_crossing_velocity_world_xy", "teacher_crossing_velocity_heading_world_rad",
    "directed_port_world", "route_lookahead_world", "route_lookahead_clipped",
    "remaining_route_length_m", "port_definition",
}

WAYPOINT_CONTRACT_FIELDS = {
    "schema", "experiment_id", "target_ids", "feature_order", "rows",
    "training_contract_binding", "original_ranker_support",
    "contract_difference_inventory", "target_support_counts", "content_digest",
}
WAYPOINT_ROW_FIELDS = {
    "state_id", "target_id", "source_body_pose_world", "target_world_pose",
    "target_body_pose", "dx_m", "dy_m", "distance_m", "relative_heading_rad",
    "relative_heading_sin", "relative_heading_cos", "target_tangent_heading_rad",
    "target_tangent_heading_sin", "target_tangent_heading_cos", "route_intent_dx",
    "route_intent_dy", "roundtrip_world_pose", "position_transform_error_m",
    "heading_transform_error_rad", "transform_valid",
}

PIXEL_INDEX_FIELDS = {
    "schema", "experiment_id", "rgb_file", "hash_domain", "records",
    "unique_pixel_count", "content_digest",
}
PIXEL_RECORD_FIELDS = {
    "state_id", "capture_id", "rgb_row_index", "pixel_sha256",
    "canonical_pixel_index", "row_sha256",
}
LATENT_INDEX_FIELDS = {
    "schema", "experiment_id", "latents_file", "encoder_binding",
    "preprocessing_authority", "external_encoder_source",
    "encoder_runtime_environment", "records", "content_digest",
}
LATENT_RECORD_FIELDS = {
    "canonical_pixel_index", "pixel_sha256", "raw_token_row_index",
    "raw_token_sha256", "spatial_descriptor_row_index", "spatial_descriptor_sha256",
    "preprocessed_tensor_sha256",
}

CANDIDATE_FANOUT_FIELDS = {
    "branch_id", "state_id", "role", "family", "candidate_index", "candidate_id",
    "snapshot_id", "restored_snapshot_sha256", "trace_index", "trace_slice",
    "trace_array_slice_sha256s", "requested_commands", "post_slew_applied_commands",
    "physics_sample_count",
    "port_crossing_sample_before", "port_crossing_sample_after",
    "port_crossing_fraction", "port_crossing_directed_normal_dot",
    "port_crossing_lateral_fraction", "port_crossing_displacement_world_xy",
    "port_crossing_direction_heading_world_rad",
    "beyond_port_consecutive_physics_samples",
    "target_entered_before_dwell_complete", "competing_port_crossing_first",
    "first_wrong_edge_id", "wrong_port_crossing_sample_before",
    "wrong_port_crossing_sample_after", "wrong_port_crossing_fraction",
    "wrong_port_crossing_directed_normal_dot", "wrong_port_crossing_lateral_fraction",
    "wrong_port_crossing_displacement_world_xy",
    "wrong_port_crossing_direction_heading_world_rad",
    "h1_endpoint_body", "h2_endpoint_body", "h3_endpoint_body", "h3_endpoint_world",
    "h3_base_height_m", "h3_roll_rad", "h3_pitch_rad", "h3_solver_finite",
    "h3_disallowed_contact",
    "physics_contact", "stuck", "successor_viable", "entered_correct_edge",
    "entered_wrong_edge", "no_edge", "port_progress_m", "lateral_error_m",
    "angular_error_rad", "oracle_admissible", "positive_port_progress",
    "endpoint_node_id", "endpoint_edge_id",
    "left_source_region", "first_source_exit_sample_index", "reached_target_node",
    "first_target_entry_sample_index",
    "command_tracking_rows", "physical_runtime_core_sha256",
}
COMMAND_TRACKING_ROW_FIELDS = {
    "command_tick_index", "requested_command", "post_slew_command",
    "mean_achieved_body_velocity", "active_vx", "active_yaw",
}

DEVELOPMENT_SELECTION_FIELDS = {
    "schema", "experiment_id", "role", "target_ids", "selection_lexicographic",
    "heldout_outcome_documents_opened", "state_target_rows", "target_summaries",
    "selected_target_id", "selected_target_index", "selection_frozen",
    "ranker_runtime_environment", "content_digest",
}
STATE_TARGET_ROW_FIELDS = {
    "state_id", "target_id", "candidate_ids", "scores", "ranking",
    "eligible_correct_edge_candidate_indices", "selected_candidate_index",
    "correct_edge_top1", "correct_edge_top3", "correct_edge_mrr",
    "selected_correct_edge_execution", "selected_port_progress_m",
    "oracle_best_port_progress_m", "minimum_admissible_port_progress_m",
    "normalized_port_regret", "target_transform_error_m",
    "pairwise_correct_edge_ordering", "selected_wrong_edge", "selected_no_edge",
    "selected_lateral_error_m", "selected_angular_error_rad", "selected_contact",
    "selected_stuck", "selected_successor_viable",
}
TARGET_SUMMARY_FIELDS = {
    "target_id", "state_count", "selected_correct_edge_execution_rate",
    "correct_edge_top3_rate", "correct_edge_top1_rate", "correct_edge_mrr",
    "normalized_port_regret", "mean_selected_port_progress_m",
    "mean_target_transform_error_m", "selection_key",
    "pairwise_correct_edge_ordering", "selected_wrong_edge_rate",
    "selected_no_edge_rate", "mean_selected_lateral_error_m",
    "mean_selected_angular_error_rad", "selected_contact_rate",
    "selected_stuck_rate", "selected_successor_viable_rate",
}

HELDOUT_SCORE_FIELDS = {
    "score_row_id", "state_id", "family", "condition_id", "target_id",
    "candidate_ids", "scores", "ranking", "eligible_correct_edge_candidate_indices",
    "selected_candidate_index", "correct_edge_top1", "correct_edge_top3",
    "correct_edge_mrr", "selected_correct_edge_execution", "selected_port_progress_m",
    "oracle_best_port_progress_m", "minimum_admissible_port_progress_m",
    "normalized_port_regret", "teacher_trace_id", "teacher_correct_execution",
    "ranker_checkpoint_sha256",
    "pairwise_correct_edge_ordering", "selected_wrong_edge", "selected_no_edge",
    "selected_lateral_error_m", "selected_angular_error_rad", "selected_contact",
    "selected_stuck", "selected_successor_viable",
    "ranker_runtime_environment_sha256",
}

REPEATED_EXECUTION_FIELDS = {
    "repeat_id", "state_id", "family", "branch_selector_id", "repeat_index",
    "source_candidate_index", "source_branch_id", "snapshot_id",
    "restored_snapshot_sha256", "trace_index", "trace_slice",
    "trace_array_slice_sha256s", "source_correct_edge_execution",
    "repeat_correct_edge_execution", "source_endpoint_body", "repeat_endpoint_body",
    "endpoint_position_error_m", "endpoint_heading_error_rad", "physics_contact",
    "source_applied_command_sequence_sha256", "repeat_applied_command_sequence_sha256",
    "source_endpoint_edge_id", "repeat_endpoint_edge_id", "source_physics_contact",
    "source_stuck", "stuck", "repeat_success", "physical_runtime_core_sha256",
}

CLASSIFICATION_INPUT_FIELDS = {
    "teacher_correct_execution_count", "coverage_rate", "ranker_correct_edge_top1_rate",
    "ranker_correct_edge_top3_rate", "ranker_selected_correct_edge_execution_rate",
    "ranker_normalized_port_regret", "oracle_selected_correct_edge_execution_rate",
    "oracle_covered_state_correct_execution_rate",
    "repeatability_rate", "command_tracking_pass", "minimum_family_correct_execution_count",
    "selected_target_id", "selected_target_passes_handoff_gate",
    "selected_target_materially_outperforms_node_centre",
}

EVIDENCE_KEYS = {
    "panel_manifest", "split_manifest", "graph_manifest", "state_snapshot_index",
    "teacher_trace_index", "edge_port_index", "waypoint_contracts", "pixel_index",
    "latent_index", "candidate_fanout", "development_target_selection",
    "heldout_ranker_scores", "repeated_execution", "npz_inspections",
}


def _mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} field set drift")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be a sequence")
    return list(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be finite numeric")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be boolean")
    return value


def _sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} must be lowercase SHA-256")
    return text


def _vector(value: Any, length: int, label: str) -> list[float]:
    values = _sequence(value, label)
    if len(values) != length:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} length drift")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(values)]


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def runtime_environment_sha256(value: Mapping[str, Any]) -> str:
    """Hash one observed-runtime mapping in the frozen canonical JSON domain."""

    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffMetricsError(
            "runtime environment must be a mapping"
        )
    try:
        payload = C.canonical_json_bytes(dict(value))[:-1]
    except (TypeError, ValueError) as exc:
        raise PhysicalGraphEdgeHandoffMetricsError(
            "runtime environment is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_physical_runtime_environment(value: Any) -> dict[str, Any]:
    """Validate the official 256+64 physical-runtime equality projection."""

    row = _mapping(
        value, set(C.PHYSICAL_RUNTIME_ENVIRONMENT_FIELDS),
        "physical_runtime_environment",
    )
    authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    if row["stage_id"] != authority["stage_id"]:
        raise PhysicalGraphEdgeHandoffMetricsError("physical runtime stage drift")
    for field in (
        "python_executable", "python_version", "torch_version",
        "torch_hip_version", "genesis_version", "quadrants_version",
        "device", "backend",
    ):
        _string(row[field], f"physical runtime {field}")
    _integer(row["visible_device_count"], "physical visible device count", minimum=0)
    fake = _boolean(row["fake_runtime"], "physical fake_runtime")
    environment = row["deterministic_environment"]
    expected_environment = C.DIRECT_RUNTIME_POLICY[
        "required_environment_before_simulator_creation"
    ]
    if not isinstance(environment, Mapping) or set(environment) != set(expected_environment):
        raise PhysicalGraphEdgeHandoffMetricsError(
            "physical deterministic environment field drift"
        )
    for key, item in environment.items():
        if item is not None and not isinstance(item, str):
            raise PhysicalGraphEdgeHandoffMetricsError(
                f"physical deterministic environment {key} is invalid"
            )
    if not fake:
        expected_real = {
            field: authority[field]
            for field in (
                "real_python_executable", "python_version", "torch_version",
                "torch_hip_version", "genesis_version", "quadrants_version",
                "visible_device_count", "device", "backend",
            )
        }
        observed_real = {
            "real_python_executable": row["python_executable"],
            "python_version": row["python_version"],
            "torch_version": row["torch_version"],
            "torch_hip_version": row["torch_hip_version"],
            "genesis_version": row["genesis_version"],
            "quadrants_version": row["quadrants_version"],
            "visible_device_count": row["visible_device_count"],
            "device": row["device"],
            "backend": row["backend"],
        }
        if observed_real != expected_real or dict(environment) != expected_environment:
            raise PhysicalGraphEdgeHandoffMetricsError(
                "real physical runtime environment drift"
            )
    core = {field: row[field] for field in C.PHYSICAL_RUNTIME_CORE_FIELDS}
    expected_digest = runtime_environment_sha256(core)
    if _sha(row["runtime_core_sha256"], "physical runtime core digest") != expected_digest:
        raise PhysicalGraphEdgeHandoffMetricsError("physical runtime core digest drift")
    for field, count in (
        ("qualification_runtime_sha256s", C.TEACHER_TRACE_COUNT),
        ("selected_snapshot_runtime_sha256s", C.STATE_COUNT),
    ):
        values = _sequence(row[field], field)
        if len(values) != count:
            raise PhysicalGraphEdgeHandoffMetricsError(
                f"{field} cardinality drift"
            )
        for item in values:
            if _sha(item, field) != expected_digest:
                raise PhysicalGraphEdgeHandoffMetricsError(
                    f"{field} contains a divergent shard runtime"
                )
    return row


def validate_visual_runtime_environment(
    value: Any, *, runtime_role: str,
) -> dict[str, Any]:
    """Validate an observed encoder or ranker runtime and frozen model binding."""

    if runtime_role not in {"encoder", "ranker"}:
        raise PhysicalGraphEdgeHandoffMetricsError("unknown visual runtime role")
    row = _mapping(
        value, set(C.VISUAL_RUNTIME_ENVIRONMENT_FIELDS),
        f"{runtime_role}_runtime_environment",
    )
    authority = C.RUNTIME_ENVIRONMENT_AUTHORITY[runtime_role]
    always_exact = (
        "stage_id", "model_role", "checkpoint_sha256", "model_source_path",
        "model_source_sha256", "external_repository_path",
        "external_repository_commit", "external_worktree_clean",
    )
    if any(row[field] != authority[field] for field in always_exact):
        raise PhysicalGraphEdgeHandoffMetricsError(
            f"{runtime_role} model/runtime binding drift"
        )
    for field in (
        "python_executable", "python_version", "torch_version",
        "torch_hip_version", "device", "backend",
    ):
        _string(row[field], f"{runtime_role} runtime {field}")
    _integer(row["visible_device_count"], f"{runtime_role} visible device count", minimum=0)
    fake = _boolean(row["fake_runtime"], f"{runtime_role} fake_runtime")
    capability = row["device_capability"]
    if capability is not None:
        values = _sequence(capability, f"{runtime_role} device capability")
        if len(values) != 2:
            raise PhysicalGraphEdgeHandoffMetricsError("device capability rank drift")
        for item in values:
            _integer(item, "device capability", minimum=0)
    if row["device_name"] is not None:
        _string(row["device_name"], f"{runtime_role} device name")
    if not fake:
        for field in (
            "python_version", "torch_version", "torch_hip_version",
            "visible_device_count", "device", "device_name",
            "device_capability", "backend",
        ):
            if row[field] != authority[field]:
                raise PhysicalGraphEdgeHandoffMetricsError(
                    f"real {runtime_role} runtime {field} drift"
                )
        if row["python_executable"] != authority["real_python_executable"]:
            raise PhysicalGraphEdgeHandoffMetricsError(
                f"real {runtime_role} interpreter drift"
            )
    return row


def point_in_polygon_inclusive(
    point_xy: Sequence[Any], polygon_xy: Sequence[Sequence[Any]], *,
    tolerance_m: float = C.NUMERICAL_TOLERANCES["se2_position_m"],
) -> bool:
    """Boundary-inclusive deterministic 2-D polygon membership."""

    point = _vector(point_xy, 2, "point_xy")
    polygon = [_vector(item, 2, "polygon vertex") for item in _sequence(polygon_xy, "polygon_xy")]
    if len(polygon) < 3:
        raise PhysicalGraphEdgeHandoffMetricsError("polygon_xy is degenerate")
    tolerance = _finite(tolerance_m, "tolerance_m")
    if tolerance < 0.0:
        raise PhysicalGraphEdgeHandoffMetricsError("tolerance_m must be nonnegative")
    x, y = point
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            raise PhysicalGraphEdgeHandoffMetricsError("polygon has a zero-length edge")
        projection = max(
            0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_sq)
        )
        nearest_x = start[0] + projection * dx
        nearest_y = start[1] + projection * dy
        if math.hypot(x - nearest_x, y - nearest_y) <= tolerance:
            return True
        if (start[1] > y) != (end[1] > y):
            crossing_x = start[0] + (y - start[1]) * dx / dy
            if x < crossing_x:
                inside = not inside
    return inside


def transverse_port_crossing(
    position_before_xy: Sequence[Any], position_after_xy: Sequence[Any],
    opening_segment_world: Sequence[Sequence[Any]],
    opening_normal_world: Sequence[Any], *,
    tolerance_m: float = C.NUMERICAL_TOLERANCES["se2_position_m"],
) -> dict[str, Any] | None:
    """Return the frozen signed-plane transverse crossing, or ``None``."""

    before = _vector(position_before_xy, 2, "position_before_xy")
    after = _vector(position_after_xy, 2, "position_after_xy")
    segment_rows = _sequence(opening_segment_world, "opening_segment_world")
    if len(segment_rows) != 2:
        raise PhysicalGraphEdgeHandoffMetricsError("opening segment must have two endpoints")
    segment = [_vector(item, 2, "opening endpoint") for item in segment_rows]
    normal = _vector(opening_normal_world, 2, "opening_normal_world")
    tolerance = _finite(tolerance_m, "tolerance_m")
    if tolerance < 0.0:
        raise PhysicalGraphEdgeHandoffMetricsError("tolerance_m must be nonnegative")
    normal_length = math.hypot(*normal)
    if not _close(normal_length, 1.0, tolerance):
        raise PhysicalGraphEdgeHandoffMetricsError("opening normal must be unit length")
    segment_dx = segment[1][0] - segment[0][0]
    segment_dy = segment[1][1] - segment[0][1]
    segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy
    if segment_length_sq <= 0.0:
        raise PhysicalGraphEdgeHandoffMetricsError("opening segment is degenerate")
    d0 = ((before[0] - segment[0][0]) * normal[0]
          + (before[1] - segment[0][1]) * normal[1])
    d1 = ((after[0] - segment[0][0]) * normal[0]
          + (after[1] - segment[0][1]) * normal[1])
    denominator = d1 - d0
    if not (d0 <= tolerance and d1 > tolerance and denominator > 0.0):
        return None
    raw_alpha = -d0 / denominator
    if raw_alpha < -tolerance or raw_alpha > 1.0 + tolerance:
        return None
    alpha = max(0.0, min(1.0, raw_alpha))
    point = [
        before[axis] + alpha * (after[axis] - before[axis])
        for axis in range(2)
    ]
    normal_displacement = ((after[0] - before[0]) * normal[0]
                           + (after[1] - before[1]) * normal[1])
    if normal_displacement <= 0.0:
        return None
    lateral_numerator = ((point[0] - segment[0][0]) * segment_dx
                         + (point[1] - segment[0][1]) * segment_dy)
    lateral_fraction = lateral_numerator / segment_length_sq
    lateral_tolerance = tolerance / math.sqrt(segment_length_sq)
    if lateral_fraction < -lateral_tolerance or lateral_fraction > 1.0 + lateral_tolerance:
        return None
    return {
        "crossing_fraction": alpha,
        "crossing_point_world_xy": point,
        "directed_normal_displacement_m": normal_displacement,
        "lateral_fraction": max(0.0, min(1.0, lateral_fraction)),
        "signed_distance_before_m": d0,
        "signed_distance_after_m": d1,
    }


def first_registered_port_crossing(
    positions_xy: Sequence[Sequence[Any]],
    selected_edge: Mapping[str, Any],
    competing_edges: Sequence[Mapping[str, Any]],
    *,
    tolerance_m: float = C.NUMERICAL_TOLERANCES["se2_position_m"],
) -> dict[str, Any] | None:
    """Return the first crossing under the frozen all-port ordering rule.

    Every registered edge, including the selected edge, participates with its
    actual ``edge_id``.  Ordering is by sample-after index, then exact crossing
    fraction, then lexical edge ID.  This makes a simultaneous selected versus
    competing crossing deterministic; with the frozen IDs, a competing edge
    wins an otherwise exact tie.
    """

    positions = [
        _vector(item, 2, f"positions_xy[{index}]")
        for index, item in enumerate(_sequence(positions_xy, "positions_xy"))
    ]
    if len(positions) < 2:
        raise PhysicalGraphEdgeHandoffMetricsError(
            "positions_xy must contain at least two samples"
        )
    edge_rows: list[tuple[bool, Mapping[str, Any]]] = [(True, selected_edge)]
    edge_rows.extend((False, item) for item in _sequence(competing_edges, "competing_edges"))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for is_selected, edge in edge_rows:
        if not isinstance(edge, Mapping):
            raise PhysicalGraphEdgeHandoffMetricsError("registered edge must be a mapping")
        edge_id = _string(edge.get("edge_id"), "registered edge_id")
        if edge_id in seen:
            raise PhysicalGraphEdgeHandoffMetricsError("registered edge_id is duplicated")
        seen.add(edge_id)
        if "opening_segment_world" not in edge or "opening_normal_world" not in edge:
            raise PhysicalGraphEdgeHandoffMetricsError("registered edge geometry is incomplete")
        for after_index in range(1, len(positions)):
            crossing = transverse_port_crossing(
                positions[after_index - 1],
                positions[after_index],
                edge["opening_segment_world"],
                edge["opening_normal_world"],
                tolerance_m=tolerance_m,
            )
            if crossing is not None:
                candidates.append({
                    "edge_id": edge_id,
                    "is_selected_edge": is_selected,
                    "crossing_sample_before": after_index - 1,
                    "crossing_sample_after": after_index,
                    **crossing,
                })
                break
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row["crossing_sample_after"],
            row["crossing_fraction"],
            row["edge_id"],
        ),
    )


def _validate_document(
    value: Any, fields: set[str], schema: str, label: str
) -> dict[str, Any]:
    row = _mapping(value, fields, label)
    try:
        C.validate_content_digest(row)
    except C.PhysicalGraphEdgeHandoffContractError as exc:
        raise PhysicalGraphEdgeHandoffMetricsError(str(exc)) from exc
    if row["schema"] != schema or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} identity drift")
    return row


def _validate_file_binding(value: Any, label: str, expected_path: str | None = None) -> dict[str, Any]:
    row = _mapping(value, FILE_BINDING_FIELDS, label)
    if expected_path is not None and row["path"] != expected_path:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} path drift")
    _integer(row["bytes"], f"{label}.bytes", minimum=1)
    _sha(row["sha256"], f"{label}.sha256")
    return row


def _validate_slice(value: Any, label: str, expected_file: str) -> dict[str, Any]:
    row = _mapping(value, SLICE_BINDING_FIELDS, label)
    if row["file_path"] != expected_file:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} file drift")
    _sha(row["file_sha256"], f"{label}.file_sha256")
    _string(row["member"], f"{label}.member")
    start = _integer(row["start"], f"{label}.start")
    stop = _integer(row["stop"], f"{label}.stop")
    if stop <= start:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} slice is empty")
    _sha(row["slice_sha256"], f"{label}.slice_sha256")
    return row


def _rank(scores: Sequence[Any]) -> list[int]:
    values = [_finite(score, f"scores[{index}]") for index, score in enumerate(scores)]
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def _correct_candidate(row: Mapping[str, Any]) -> bool:
    return bool(
        row["oracle_admissible"]
        and row["entered_correct_edge"]
        and row["successor_viable"]
        and row["positive_port_progress"]
        and not row["physics_contact"]
        and not row["stuck"]
    )


def _normalized_regret(best: float, selected: float, minimum: float) -> float:
    span = best - minimum
    if span <= C.NUMERICAL_TOLERANCES["division_floor"]:
        return 0.0
    return min(1.0, max(0.0, (best - selected) / span))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise PhysicalGraphEdgeHandoffMetricsError("mean of empty sequence")
    return math.fsum(values) / len(values)


def panel_manifest_authority() -> dict[str, Any]:
    return {
        "schema": "physical_graph_edge_handoff_qualification_v1.panel_manifest.v1",
        "root_fields": sorted(PANEL_FIELDS),
        "state_fields": sorted(PANEL_STATE_FIELDS),
        "eligibility_fields": sorted(ELIGIBILITY_FIELDS),
        "exclusion_evidence_fields": sorted(EXCLUSION_EVIDENCE_FIELDS),
        "pool_selection_fields": sorted(POOL_SELECTION_FIELDS),
        "pool_qualification_fields": sorted(POOL_QUALIFICATION_FIELDS),
        "state_count": C.STATE_COUNT,
    }


def physical_trace_reduction_authority() -> dict[str, Any]:
    """Return geometry and formulas sufficient for independent NPZ reduction."""

    geometry_rows = []
    for pool_order_index, spec in enumerate(C.build_candidate_specs()):
        geometry = spec["geometry"]
        geometry_rows.append({
            "candidate_spec_id": spec["candidate_spec_id"],
            "state_id": spec["state_id"],
            "pool_order_index": pool_order_index,
            "canonical_spec_sha256": spec["canonical_spec_sha256"],
            "source_boundary_polygon_world": geometry["source_node"]["boundary_polygon_world"],
            "target_boundary_polygon_world": geometry["target_node"]["boundary_polygon_world"],
            "selected_edge": {
                "edge_id": geometry["selected_directed_edge"]["edge_id"],
                "opening_segment_world": geometry["selected_directed_edge"]["opening_segment_world"],
                "opening_normal_world": geometry["selected_directed_edge"]["opening_normal_world"],
            },
            "competing_edges": [
                {
                    "edge_id": row["edge_id"],
                    "opening_segment_world": row["opening_segment_world"],
                    "opening_normal_world": row["opening_normal_world"],
                }
                for row in geometry["competing_directed_edges"]
            ],
            "teacher_route_polyline_world": geometry["teacher_route_polyline_world"],
            "spawn_se2_world": geometry["spawn_se2_world"],
        })
    return C.attach_content_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.physical_trace_reduction_authority.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "physics_dt_s": 0.002,
        "sustained_beyond_samples": C.PORT_DWELL_PHYSICS_SAMPLES,
        "source_exit_tolerance_m": C.NUMERICAL_TOLERANCES["se2_position_m"],
        "port_crossing_tolerance_m": C.NUMERICAL_TOLERANCES["se2_position_m"],
        "persisted_summary_float_tolerance": C.NUMERICAL_TOLERANCES["trace_summary_float"],
        "endpoint_indices": {"H1": 249, "H2": 499, "H3": 749},
        "base_pose_member_order": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        "base_twist_member_order": ["vx_world", "vy_world", "vz_world", "wx_world", "wy_world", "wz_world"],
        "teacher_member_mapping": {
            "timestamp": "timestamp_s", "base_pose": "base_pose_world",
            "base_twist": "base_twist_world", "contact": "physics_contact",
            "requested_command": "requested_command",
            "applied_command": "applied_command",
            "source_membership": "source_region_member", "selected_edge_membership": "edge_region_member",
            "target_membership": "target_region_member",
        },
        "candidate_member_mapping": {
            "timestamp": "timestamp_s", "base_pose": "base_pose_world",
            "base_twist": "base_twist_world", "requested_command": "requested_command",
            "post_slew_command": "post_slew_applied_command", "contact": "physics_contact",
            "source_membership": "source_region_member", "selected_edge_membership": "correct_edge_region_member",
            "competing_edge_membership": "wrong_edge_region_member", "target_membership": "target_region_member",
        },
        "point_in_polygon": "boundary-inclusive winding parity under se2_position_m tolerance",
        "point_in_polygon_helper": (
            "lewm.safety.physical_graph_edge_handoff_qualification_v1_metrics."
            "point_in_polygon_inclusive"
        ),
        "transverse_port_crossing": (
            "for adjacent positions p0,p1 compute signed distances to the port plane; "
            "with tolerance eps=port_crossing_tolerance_m accept the first "
            "d0<=eps and d1>eps, interpolate alpha=-d0/(d1-d0), require "
            "dot(p1-p0,outward_normal)>0 and the interpolated point's segment projection in [0,1]"
        ),
        "transverse_port_crossing_helper": (
            "lewm.safety.physical_graph_edge_handoff_qualification_v1_metrics."
            "transverse_port_crossing"
        ),
        "first_registered_port_crossing_helper": (
            "lewm.safety.physical_graph_edge_handoff_qualification_v1_metrics."
            "first_registered_port_crossing"
        ),
        "source_exit": "first boundary-inclusive source membership true-to-false transition",
        "crossing_velocity": (
            "linearly interpolate base_twist_world vx_world,vy_world at crossing alpha; "
            "heading=atan2(vy_world,vx_world), requiring nonzero planar speed"
        ),
        "selected_vs_competing_precedence": (
            "derive every selected and competing transverse crossing from base poses; "
            "order all crossings by (crossing_sample_after, exact crossing_fraction, "
            "actual registered edge_id); lexical edge_id breaks an exact tie. The "
            "selected edge uses its actual ID selected-edge, never an empty sentinel, so "
            "competing-edge-* wins an otherwise exact selected/competing tie"
        ),
        "correct_edge_event": C.PHYSICAL_OUTCOME_AUTHORITY["entered_correct_edge"],
        "teacher_selected_port_crossing": (
            "the first selected transverse-port crossing returned by transverse_port_crossing; "
            "this crossing predicate is separate from the independently reported dwell condition"
        ),
        "wrong_edge_event": C.PHYSICAL_OUTCOME_AUTHORITY["entered_wrong_edge"],
        "dwell_samples": C.PORT_DWELL_PHYSICS_SAMPLES,
        "beyond_port_consecutive_physics_samples": (
            "starting at crossing_sample_after, count the full consecutive run of samples "
            "whose signed distance along the selected outward normal is greater than "
            "port_crossing_tolerance_m; stop at the first non-beyond sample or trace end. "
            "Target entry does not truncate the count unless execution itself ends there"
        ),
        "target_entered_before_dwell_complete": (
            "when first_target_entry_sample_index is at or after crossing_sample_after, "
            "count inclusively as first_target_entry_sample_index-crossing_sample_after+1; "
            "the flag is true exactly when that count is <100. Entry on the 100th "
            "post-crossing sample is not early and satisfies the ordinary dwell threshold"
        ),
        "teacher_route_progress_m": (
            "initial Euclidean distance from base xy to the selected opening midpoint minus "
            "the minimum such distance over all retained teacher samples; strictly >0 passes"
        ),
        "candidate_port_progress_m": C.PHYSICAL_OUTCOME_AUTHORITY["port_progress_m"],
        "lateral_error_m": C.PHYSICAL_OUTCOME_AUTHORITY["lateral_error_m"],
        "angular_error_rad": C.PHYSICAL_OUTCOME_AUTHORITY["angular_error_rad"],
        "teacher_endpoint_errors": (
            "apply the same selected-port lateral_error_m and angular_error_rad formulas "
            "to the final retained teacher base pose"
        ),
        "teacher_stuck": (
            "apply the same frozen stuck projection to the teacher requested-command and "
            "base-pose trace; report it descriptively without adding it to the frozen "
            "teacher-valid eligibility predicate"
        ),
        "teacher_successor_viable": (
            "apply the same frozen successor_viable projection to the final teacher "
            "base pose/twist, finite-state, and contact evidence; report it descriptively "
            "without adding target-node completion or a new eligibility condition"
        ),
        "stuck": copy.deepcopy(C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]),
        "successor_viable": copy.deepcopy(C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]),
        "contact": copy.deepcopy(C.CONTACT_AUTHORITY),
        "command_tracking_authority": copy.deepcopy(C.COMMAND_TRACKING_AUTHORITY),
        "outcome_thresholds": {
            "dwell_samples": C.PORT_DWELL_PHYSICS_SAMPLES,
            "stuck": copy.deepcopy(C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]),
            "successor_viable": copy.deepcopy(C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]),
        },
        "specs": geometry_rows,
        "selected_spec_rule": "derive selected=true only from validated panel qualification_rows",
    })


def reducer_authority() -> dict[str, Any]:
    """Return the canonical public schema and metric authority."""

    return C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.reducer_authority.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "evidence_keys": sorted(EVIDENCE_KEYS),
            "documents": {
                "panel_manifest": {"root": sorted(PANEL_FIELDS), "row": sorted(PANEL_STATE_FIELDS), "container": "states", "count": 64, "identity_order": ["state_id"], "qualification_row": sorted(POOL_QUALIFICATION_FIELDS), "qualification_container": "prospective_pool_selection.qualification_rows", "qualification_count": C.PROSPECTIVE_POOL_COUNT, "qualification_identity_order": ["family", "stratum_index", "variant_index"]},
                "split_manifest": {"root": sorted(SPLIT_FIELDS), "row": sorted(SPLIT_ASSIGNMENT_FIELDS), "container": "assignments", "count": 64, "identity_order": ["state_id"]},
                "graph_manifest": {"root": sorted(GRAPH_FIELDS), "row": sorted(GRAPH_RECORD_FIELDS), "container": "graphs", "count": 64, "identity_order": ["state_id"]},
                "state_snapshot_index": {"root": sorted(SNAPSHOT_INDEX_FIELDS), "row": sorted(SNAPSHOT_RECORD_FIELDS), "container": "records", "count": 64, "identity_order": ["state_id"]},
                "teacher_trace_index": {"root": sorted(TEACHER_INDEX_FIELDS), "row": sorted(TEACHER_RECORD_FIELDS), "container": "records", "count": C.TEACHER_TRACE_COUNT, "identity_order": ["family", "stratum_index", "variant_index"]},
                "edge_port_index": {"root": sorted(EDGE_PORT_INDEX_FIELDS), "row": sorted(EDGE_PORT_RECORD_FIELDS), "container": "records", "count": 64, "identity_order": ["state_id"]},
                "waypoint_contracts": {"root": sorted(WAYPOINT_CONTRACT_FIELDS), "row": sorted(WAYPOINT_ROW_FIELDS), "container": "rows", "count": 192, "identity_order": ["state_id", "target_id"]},
                "pixel_index": {"root": sorted(PIXEL_INDEX_FIELDS), "row": sorted(PIXEL_RECORD_FIELDS), "container": "records", "count": 64, "identity_order": ["state_id"]},
                "latent_index": {"root": sorted(LATENT_INDEX_FIELDS), "row": sorted(LATENT_RECORD_FIELDS), "container": "records", "count": "unique_pixel_count", "identity_order": ["canonical_pixel_index"]},
                "development_target_selection": {"root": sorted(DEVELOPMENT_SELECTION_FIELDS), "state_target_row": sorted(STATE_TARGET_ROW_FIELDS), "state_target_container": "state_target_rows", "state_target_count": 144, "state_target_identity_order": ["state_id", "target_id"], "summary_row": sorted(TARGET_SUMMARY_FIELDS), "summary_container": "target_summaries", "summary_count": 3, "summary_identity_order": ["target_id"]},
            },
            "ledgers": {
                "candidate_fanout": {"fields": sorted(CANDIDATE_FANOUT_FIELDS), "count": C.BRANCH_COUNT, "identity_order": ["state_id", "candidate_index"]},
                "heldout_ranker_scores": {"fields": sorted(HELDOUT_SCORE_FIELDS), "count": C.HELDOUT_SCORE_ROW_COUNT, "identity_order": ["state_id", "condition_id"]},
                "repeated_execution": {"fields": sorted(REPEATED_EXECUTION_FIELDS), "count": C.REPEATED_EXECUTION_ROW_COUNT, "identity_order": ["state_id", "branch_selector_id", "repeat_index"]},
            },
            "npz_authorities": copy.deepcopy(C.NPZ_PAYLOAD_AUTHORITY),
            "trace_index_ranges": copy.deepcopy(C.TRACE_INDEX_RANGES),
            "reset_fixture_physics_samples": C.RESET_FIXTURE_PHYSICS_SAMPLES,
            "branch_physics_samples": C.PHYSICS_STEPS_PER_BRANCH,
            "reset_pair_comparison_fields": sorted(RESET_PAIR_COMPARISON_FIELDS),
            "reset_trace_pair_comparison_authority": copy.deepcopy(
                C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY
            ),
            "runtime_environment_authority": copy.deepcopy(
                C.RUNTIME_ENVIRONMENT_AUTHORITY
            ),
            "runtime_paths": copy.deepcopy(C.RUNTIME_OUTPUT_PATHS),
            "candidate_ids": list(C.CANDIDATE_IDS),
            "target_ids": list(C.TARGET_IDS),
            "heldout_condition_ids": list(C.HELDOUT_CONDITION_IDS),
            "repeat_branch_ids": list(C.REPEAT_BRANCH_IDS),
            "roles": copy.deepcopy(C.ROLE_COUNTS),
            "families": list(C.FAMILY_IDS),
            "gate": copy.deepcopy(C.HANDOFF_GATE),
            "materiality": copy.deepcopy(C.TARGET_MATERIALITY),
            "classification_precedence": list(C.CLASSIFICATION_PRECEDENCE),
            "classification_input_fields": sorted(CLASSIFICATION_INPUT_FIELDS),
            "next_decisions": copy.deepcopy(C.NEXT_DECISION_BY_CLASSIFICATION),
            "composite_next_by_earliest_component": copy.deepcopy(
                C.COMPOSITE_NEXT_BY_EARLIEST_COMPONENT
            ),
            "external_artifact_roles": [row["role"] for row in C.EXTERNAL_ARTIFACT_BINDINGS],
            "predecessor_context_binding": copy.deepcopy(C.V2_CONTEXT_BINDING),
            "physical_trace_reduction_authority": physical_trace_reduction_authority(),
        }
    )


def external_artifact_bindings(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the seven frozen external runtime artifacts; evidence stays in-root."""

    if contract.get("schema") == "physical_graph_edge_handoff_qualification_v1.runtime_contract.v1":
        runtime = C.validate_runtime_contract(contract)
        return copy.deepcopy(runtime["external_artifact_bindings"])
    C.validate_contract(contract)
    return [copy.deepcopy(row) for row in C.EXTERNAL_ARTIFACT_BINDINGS]


def predecessor_context_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project all eight frozen V2 chronology leaves without granting reuse."""

    if contract.get("schema") == "physical_graph_edge_handoff_qualification_v1.runtime_contract.v1":
        runtime = C.validate_runtime_contract(contract)
        nested = runtime["scientific_contract"]["v2_context_binding"]
    else:
        nested = C.validate_contract(contract)["v2_context_binding"]
    if nested != C.V2_CONTEXT_BINDING:
        raise PhysicalGraphEdgeHandoffMetricsError("V2 context binding drift")
    return copy.deepcopy(nested)


def validate_npz_inspections(value: Any) -> dict[str, dict[str, Any]]:
    rows = _sequence(value, "npz_inspections")
    if len(rows) != len(C.NPZ_PAYLOAD_AUTHORITY):
        raise PhysicalGraphEdgeHandoffMetricsError("NPZ inspection count drift")
    by_path: dict[str, dict[str, Any]] = {}
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, NPZ_INSPECTION_FIELDS, f"npz[{index}]")
        path = _string(row["path"], f"npz[{index}].path")
        if path in by_path or path not in C.NPZ_PAYLOAD_AUTHORITY:
            raise PhysicalGraphEdgeHandoffMetricsError("NPZ path identity drift")
        _integer(row["bytes"], f"npz[{index}].bytes", minimum=1)
        _sha(row["sha256"], f"npz[{index}].sha256")
        members = row["members"]
        expected = C.NPZ_PAYLOAD_AUTHORITY[path]
        if not isinstance(members, Mapping) or set(members) != set(expected):
            raise PhysicalGraphEdgeHandoffMetricsError(f"{path} member set drift")
        file_symbols: dict[str, int] = {}
        observed_shapes: dict[str, list[int]] = {}
        offset_values_by_member: dict[str, list[int]] = {}
        for member_name, expected_spec in expected.items():
            observed = _mapping(
                members[member_name], NPZ_MEMBER_INSPECTION_FIELDS,
                f"{path}.{member_name}",
            )
            if observed["descr"] != expected_spec["descr"] or observed["digest_dtype"] != expected_spec["digest_dtype"]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} dtype drift")
            shape = _sequence(observed["shape"], f"{path}.{member_name}.shape")
            for dimension in shape:
                _integer(dimension, f"{path}.{member_name}.shape", minimum=0)
            expected_shape = expected_spec["shape"]
            if len(shape) != len(expected_shape):
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} rank drift")
            for observed_dim, expected_dim in zip(shape, expected_shape):
                if isinstance(expected_dim, int) and observed_dim != expected_dim:
                    raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} shape drift")
                if isinstance(expected_dim, str):
                    old = file_symbols.setdefault(expected_dim, observed_dim)
                    if old != observed_dim:
                        raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} symbolic shape drift")
            if observed["c_contiguous"] is not True or observed["object_dtype"] is not False:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} unsafe array layout")
            _sha(observed["member_sha256"], f"{path}.{member_name}.member_sha256")
            digests = _sequence(observed["row_or_slice_sha256s"], f"{path}.{member_name}.digests")
            for digest_index, digest in enumerate(digests):
                _sha(digest, f"{path}.{member_name}.digests[{digest_index}]")
            mode = expected_spec["hash_mode"]
            if mode == "whole" and len(digests) != 1:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} whole digest coverage drift")
            if mode == "rows_axis0" and len(digests) != shape[0]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} row digest coverage drift")
            offset_values = observed["offset_values"]
            if member_name.endswith("offsets"):
                offsets = _sequence(offset_values, f"{path}.{member_name}.offset_values")
                offsets = [_integer(item, f"{path}.{member_name}.offset") for item in offsets]
                if len(offsets) != shape[0] or not offsets or offsets[0] != 0 or any(
                    right <= left for left, right in zip(offsets, offsets[1:])
                ):
                    raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} offsets drift")
                offset_values_by_member[member_name] = offsets
            elif offset_values is not None:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} unexpected offsets")
            observed_shapes[member_name] = shape
        for member_name, expected_spec in expected.items():
            if expected_spec["hash_mode"] != "offset_slices":
                continue
            offsets_name = expected_spec["offsets_member"]
            offsets = offset_values_by_member[offsets_name]
            shape = observed_shapes[member_name]
            if offsets[-1] != shape[0]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} final offset drift")
            digests = members[member_name]["row_or_slice_sha256s"]
            if len(digests) != len(offsets) - 1:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{path}.{member_name} slice digest coverage drift")
        if path == "candidate_traces.npz":
            offsets = offset_values_by_member["trace_offsets"]
            if any(
                right - left != C.PHYSICS_STEPS_PER_BRANCH
                for left, right in zip(offsets, offsets[1:])
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("candidate trace slice duration drift")
        by_path[path] = row
    return by_path


def validate_panel_manifest(value: Any) -> dict[str, Any]:
    root = _validate_document(
        value, PANEL_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.panel_manifest.v1",
        "panel_manifest",
    )
    root["physical_runtime_environment"] = validate_physical_runtime_environment(
        root["physical_runtime_environment"]
    )
    if root["constructed_set_caveat"] != C.CONSTRUCTED_SET_CAVEAT or root["identity_domain"] != C.IDENTITY_DOMAIN:
        raise PhysicalGraphEdgeHandoffMetricsError("panel claims/identity drift")
    exclusion = _mapping(root["prior_exclusion_evidence"], EXCLUSION_EVIDENCE_FIELDS, "prior_exclusion")
    expected_exclusion_digest = hashlib.sha256(
        C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
    ).hexdigest()
    if exclusion["authority_digest"] != expected_exclusion_digest:
        raise PhysicalGraphEdgeHandoffMetricsError("prior exclusion authority digest drift")
    for key in (
        "scene_overlap_count", "scene_hash_overlap_count", "state_or_episode_overlap_count",
        "seed_overlap_count", "path_or_geometry_overlap_count", "structured_path_overlap_count",
    ):
        if _integer(exclusion[key], f"prior_exclusion.{key}") != 0:
            raise PhysicalGraphEdgeHandoffMetricsError("prior-panel identity overlap")
    if exclusion["checked_before_simulator_creation"] is not True or exclusion["all_zero"] is not True:
        raise PhysicalGraphEdgeHandoffMetricsError("prior-panel exclusion not established")
    pool = _mapping(root["prospective_pool_selection"], POOL_SELECTION_FIELDS, "prospective_pool_selection")
    expected_pool_digest = hashlib.sha256(C.canonical_json_bytes(C.build_candidate_specs())[:-1]).hexdigest()
    if pool["candidate_spec_count"] != C.PROSPECTIVE_POOL_COUNT or pool["candidate_specs_sha256"] != expected_pool_digest:
        raise PhysicalGraphEdgeHandoffMetricsError("prospective pool binding drift")
    if pool["selection_rule"] != C.GEOMETRY_AUTHORITY["teacher_only_scan"]["selection"]:
        raise PhysicalGraphEdgeHandoffMetricsError("prospective pool selection rule drift")
    if pool["teacher_scan_completed_before_role_assignment"] is not True or pool["ranker_or_fanout_outcomes_opened"] != 0 or pool["complete"] is not True:
        raise PhysicalGraphEdgeHandoffMetricsError("prospective selection boundary drift")
    if pool["per_family_scanned_counts"] != {family: C.PROSPECTIVE_POOL_PER_FAMILY for family in C.FAMILY_IDS}:
        raise PhysicalGraphEdgeHandoffMetricsError("prospective scan count drift")
    eligible_counts = pool["per_family_teacher_eligible_counts"]
    if not isinstance(eligible_counts, Mapping) or set(eligible_counts) != set(C.FAMILY_IDS):
        raise PhysicalGraphEdgeHandoffMetricsError("eligible family projection drift")
    if any(_integer(value, "eligible count") < C.STATES_PER_FAMILY for value in eligible_counts.values()):
        raise PhysicalGraphEdgeHandoffMetricsError("insufficient eligible candidates")
    specifications = C.build_candidate_specs()
    spec_by_id = {row["candidate_spec_id"]: row for row in specifications}
    qualification_rows = _sequence(pool["qualification_rows"], "qualification_rows")
    if len(qualification_rows) != C.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("qualification row count drift")
    qualification_projection: list[dict[str, Any]] = []
    per_family_eligible: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selected_by_stratum: dict[tuple[str, int], str] = {}
    selected_state_ids_from_pool: list[str] = []
    for index, (raw_qualification, spec) in enumerate(zip(qualification_rows, specifications)):
        row = _mapping(raw_qualification, POOL_QUALIFICATION_FIELDS, f"qualification[{index}]")
        for field in (
            "candidate_spec_id", "state_id", "family", "stratum_index",
            "variant_index", "canonical_spec_sha256",
        ):
            if row[field] != spec[field]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"qualification {field} drift")
        if row["teacher_trace_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError("qualification teacher trace order drift")
        _string(row["teacher_trace_id"], "teacher_trace_id")
        _sha(row["teacher_trace_slice_sha256"], "teacher trace slice")
        _sha(row["initial_decision_state_sha256"], "initial decision state")
        component_values = {
            "goal_reachable": _boolean(row["goal_reachable"], "goal_reachable"),
            "teacher_trace_contact_free": _boolean(row["teacher_trace_contact_free"], "teacher_trace_contact_free"),
            "teacher_left_source_region": _boolean(row["teacher_left_source_region"], "teacher_left_source_region"),
            "teacher_crossed_directed_port": _boolean(row["teacher_crossed_directed_port"], "teacher_crossed_directed_port"),
            "teacher_positive_route_progress": _boolean(row["teacher_positive_route_progress"], "teacher_positive_route_progress"),
            "teacher_no_competing_port": not _boolean(row["teacher_competing_port_entered"], "teacher_competing_port_entered"),
            "teacher_normal_positive": _boolean(row["teacher_normal_positive"], "teacher_normal_positive"),
            "teacher_within_lateral_bounds": _boolean(row["teacher_within_lateral_bounds"], "teacher_within_lateral_bounds"),
            "teacher_dwell_satisfied": _boolean(row["teacher_dwell_satisfied"], "teacher_dwell_satisfied"),
            "directed_port_defined": _boolean(row["directed_port_defined"], "directed_port_defined"),
            "current_rgb_valid": _boolean(row["current_rgb_valid"], "current_rgb_valid"),
            "graph_edge_physically_executable": _boolean(row["graph_edge_physically_executable"], "graph_edge_physically_executable"),
        }
        teacher_valid = all(
            component_values[name]
            for name in C.TEACHER_QUALIFICATION_COMPONENT_IDS
            if name not in {"goal_reachable", "current_rgb_valid"}
        )
        qualified = all(component_values.values())
        if row["teacher_valid"] is not teacher_valid or row["qualified"] is not qualified:
            raise PhysicalGraphEdgeHandoffMetricsError("qualification predicate drift")
        if row["selection_key_sha256"] != spec["canonical_spec_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("qualification selection key drift")
        if qualified:
            per_family_eligible[spec["family"]] += 1
        qualification_projection.append(row)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in qualification_projection:
        if row["qualified"]:
            grouped[(row["family"], row["stratum_index"])].append(row)
    for family in C.FAMILY_IDS:
        for stratum_index in range(16):
            group = sorted(
                grouped[(family, stratum_index)],
                key=lambda row: (row["selection_key_sha256"], row["candidate_spec_id"]),
            )
            if not group:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher-qualified stratum empty")
            for rank, row in enumerate(group):
                if row["rank_within_stratum"] != rank or row["selected"] is not (rank == 0):
                    raise PhysicalGraphEdgeHandoffMetricsError("qualification hash rank drift")
                expected_reason = None if rank == 0 else "HASH_ORDER_NOT_SELECTED"
                if row["rejection_reason"] != expected_reason:
                    raise PhysicalGraphEdgeHandoffMetricsError("qualification disposition drift")
            selected = group[0]
            selected_by_stratum[(family, stratum_index)] = selected["candidate_spec_id"]
            selected_state_ids_from_pool.append(selected["state_id"])
    for row in qualification_projection:
        if row["qualified"]:
            continue
        if row["rank_within_stratum"] is not None or row["selected"] is not False:
            raise PhysicalGraphEdgeHandoffMetricsError("unqualified selection evidence drift")
        false_components = sorted(
            name
            for name, value in {
                "goal_reachable": row["goal_reachable"],
                "teacher_trace_contact_free": row["teacher_trace_contact_free"],
                "teacher_left_source_region": row["teacher_left_source_region"],
                "teacher_crossed_directed_port": row["teacher_crossed_directed_port"],
                "teacher_positive_route_progress": row["teacher_positive_route_progress"],
                "teacher_no_competing_port": not row["teacher_competing_port_entered"],
                "teacher_normal_positive": row["teacher_normal_positive"],
                "teacher_within_lateral_bounds": row["teacher_within_lateral_bounds"],
                "teacher_dwell_satisfied": row["teacher_dwell_satisfied"],
                "directed_port_defined": row["directed_port_defined"],
                "current_rgb_valid": row["current_rgb_valid"],
                "graph_edge_physically_executable": row["graph_edge_physically_executable"],
            }.items()
            if not value
        )
        expected_reason = ";".join(false_components)
        if row["rejection_reason"] != expected_reason:
            raise PhysicalGraphEdgeHandoffMetricsError("unqualified rejection reason drift")
        reason_counts[expected_reason] += 1
    projection_digest = hashlib.sha256(C.canonical_json_bytes(qualification_projection)[:-1]).hexdigest()
    if pool["qualification_projection_sha256"] != projection_digest:
        raise PhysicalGraphEdgeHandoffMetricsError("qualification projection digest drift")
    if dict(per_family_eligible) != dict(eligible_counts):
        raise PhysicalGraphEdgeHandoffMetricsError("eligible count regeneration drift")
    if dict(reason_counts) != dict(pool["rejection_reason_counts"]):
        raise PhysicalGraphEdgeHandoffMetricsError("rejection count regeneration drift")
    selected_projection = pool["selected_candidate_spec_ids_by_family_and_stratum"]
    if not isinstance(selected_projection, Mapping) or set(selected_projection) != set(C.FAMILY_IDS):
        raise PhysicalGraphEdgeHandoffMetricsError("selected stratum projection drift")
    selected_specs: set[str] = set()
    for family in C.FAMILY_IDS:
        values = _sequence(selected_projection[family], f"selected[{family}]")
        if len(values) != 16:
            raise PhysicalGraphEdgeHandoffMetricsError("selected stratum count drift")
        for stratum_index, item in enumerate(values):
            item = _string(item, "candidate_spec_id")
            if item != selected_by_stratum[(family, stratum_index)]:
                raise PhysicalGraphEdgeHandoffMetricsError("selected stratum regeneration drift")
            selected_specs.add(item)
    if len(selected_specs) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("selected spec uniqueness drift")
    if pool["selected_state_ids"] != selected_state_ids_from_pool:
        raise PhysicalGraphEdgeHandoffMetricsError("selected state ordering drift")
    states = _sequence(root["states"], "panel.states")
    if len(states) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("panel state count drift")
    ids: set[str] = set()
    scenes: set[str] = set()
    episodes: set[str] = set()
    graphs: set[str] = set()
    seeds: set[int] = set()
    family_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    family_role: Counter[tuple[str, str]] = Counter()
    turn_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    family_route_counts: Counter[tuple[str, str]] = Counter()
    width_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    strata: Counter[tuple[str, int]] = Counter()
    result: list[dict[str, Any]] = []
    for index, value_row in enumerate(states):
        row = _mapping(value_row, PANEL_STATE_FIELDS, f"panel.states[{index}]")
        state_id = _string(row["state_id"], "state_id")
        scene_id = _string(row["scene_id"], "scene_id")
        episode_id = _string(row["episode_id"], "episode_id")
        graph_id = _string(row["graph_id"], "graph_id")
        if not state_id.startswith(C.STATE_ID_PREFIX) or not scene_id.startswith(C.SCENE_ID_PREFIX):
            raise PhysicalGraphEdgeHandoffMetricsError("current identity namespace drift")
        if state_id in ids or scene_id in scenes or episode_id in episodes or graph_id in graphs:
            raise PhysicalGraphEdgeHandoffMetricsError("panel identity not unique")
        ids.add(state_id); scenes.add(scene_id); episodes.add(episode_id); graphs.add(graph_id)
        seed = _integer(row["procedural_seed"], "procedural_seed")
        if seed not in C.PROCEDURAL_SEED_VALUES or seed in seeds:
            raise PhysicalGraphEdgeHandoffMetricsError("procedural seed drift")
        seeds.add(seed)
        family = row["family"]
        role = row["role"]
        if family not in C.FAMILY_IDS or role not in C.ROLE_IDS:
            raise PhysicalGraphEdgeHandoffMetricsError("family/role drift")
        family_counts[family] += 1; role_counts[role] += 1; family_role[(family, role)] += 1
        route = row["route_direction"]
        if route not in {"STRAIGHT", "LEFT", "RIGHT"}:
            raise PhysicalGraphEdgeHandoffMetricsError("route direction drift")
        route_counts[route] += 1; family_route_counts[(family, route)] += 1
        width = row["passage_width_id"]; distance = row["port_distance_id"]
        if width not in {"NARROW", "WIDE"} or distance not in {"NEAR", "FAR"}:
            raise PhysicalGraphEdgeHandoffMetricsError("width/distance stratum drift")
        width_counts[width] += 1; distance_counts[distance] += 1
        stratum = _integer(row["stratum_index"], "stratum_index")
        if stratum >= 16:
            raise PhysicalGraphEdgeHandoffMetricsError("stratum index drift")
        strata[(family, stratum)] += 1
        if row["candidate_spec_id"] not in selected_specs:
            raise PhysicalGraphEdgeHandoffMetricsError("state not selected from prospective pool")
        turn = row["turn_direction"]
        if family == "TURNING_JUNCTION":
            if turn not in {"LEFT", "RIGHT"}:
                raise PhysicalGraphEdgeHandoffMetricsError("turning-junction direction drift")
            turn_counts[turn] += 1
        elif turn is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("non-turn family has turn direction")
        spec = spec_by_id[row["candidate_spec_id"]]
        for field in (
            "state_id", "scene_id", "episode_id", "graph_id", "family",
            "route_direction", "passage_width_id", "port_distance_id",
            "stratum_index", "procedural_seed",
        ):
            if row[field] != spec[field]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"panel/spec {field} drift")
        expected_geometry_sha = hashlib.sha256(
            C.canonical_json_bytes(spec["geometry"])[:-1]
        ).hexdigest()
        if row["geometry_sha256"] != expected_geometry_sha:
            raise PhysicalGraphEdgeHandoffMetricsError("geometry digest drift")
        for field in ("source_node_id", "target_node_id", "directed_edge_id"):
            _string(row[field], field)
        eligibility = _mapping(row["eligibility_evidence"], ELIGIBILITY_FIELDS, "eligibility")
        if row["goal_reachable"] is not True or row["eligible"] is not True or not all(value is True for value in eligibility.values()):
            raise PhysicalGraphEdgeHandoffMetricsError("ineligible state in frozen panel")
        result.append(row)
    if family_counts != Counter({family: 16 for family in C.FAMILY_IDS}):
        raise PhysicalGraphEdgeHandoffMetricsError("family quota drift")
    if role_counts != Counter(C.ROLE_COUNTS):
        raise PhysicalGraphEdgeHandoffMetricsError("role count drift")
    if family_role != Counter({(family, role): count for family, values in C.FAMILY_ROLE_COUNTS.items() for role, count in values.items()}):
        raise PhysicalGraphEdgeHandoffMetricsError("family-role quota drift")
    if turn_counts != Counter(C.TURNING_JUNCTION_SIDE_COUNTS):
        raise PhysicalGraphEdgeHandoffMetricsError("left/right balance drift")
    if route_counts != Counter(C.GEOMETRY_AUTHORITY["direction_counts"]):
        raise PhysicalGraphEdgeHandoffMetricsError("route direction balance drift")
    expected_family_routes = Counter({
        (family, direction): count
        for family, counts in C.GEOMETRY_AUTHORITY["family_direction_counts"].items()
        for direction, count in counts.items()
    })
    if family_route_counts != expected_family_routes:
        raise PhysicalGraphEdgeHandoffMetricsError("family route balance drift")
    if width_counts != Counter({"NARROW": 32, "WIDE": 32}) or distance_counts != Counter({"NEAR": 32, "FAR": 32}):
        raise PhysicalGraphEdgeHandoffMetricsError("width/distance balance drift")
    if strata != Counter({(family, index): 1 for family in C.FAMILY_IDS for index in range(16)}):
        raise PhysicalGraphEdgeHandoffMetricsError("stratum coverage drift")
    if set(pool["selected_state_ids"]) != ids or len(pool["selected_state_ids"]) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("selected state projection drift")
    state_order = [row["state_id"] for row in result]
    if state_order != sorted(state_order) or state_order != pool["selected_state_ids"]:
        raise PhysicalGraphEdgeHandoffMetricsError("panel canonical state order drift")
    root["states"] = result
    return root


def validate_split_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, SPLIT_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.split_manifest.v1",
        "split_manifest",
    )
    expected_algorithm = (
        "after the complete teacher-qualified 16-state family population is frozen, "
        "sort each family by SHA256(state canonical identity); first 12 DEVELOPMENT, last 4 DEVELOPMENT_HELDOUT"
    )
    if root["assignment_algorithm"] != expected_algorithm or root["heldout_outcomes_opened_before_assignment"] != 0:
        raise PhysicalGraphEdgeHandoffMetricsError("split assignment boundary drift")
    if root["role_counts"] != C.ROLE_COUNTS or root["family_role_counts"] != C.FAMILY_ROLE_COUNTS:
        raise PhysicalGraphEdgeHandoffMetricsError("split count projection drift")
    assignments = _sequence(root["assignments"], "split.assignments")
    if len(assignments) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("split assignment count drift")
    panel_by_id = {row["state_id"]: row for row in panel["states"]}
    spec_by_id = {row["candidate_spec_id"]: row for row in C.build_candidate_specs()}
    expected_by_id: dict[str, dict[str, Any]] = {}
    population_projection: list[dict[str, Any]] = []
    for family in C.FAMILY_IDS:
        ranked: list[tuple[str, str, dict[str, Any]]] = []
        for panel_row in panel["states"]:
            if panel_row["family"] != family:
                continue
            spec = spec_by_id[panel_row["candidate_spec_id"]]
            key_projection = {
                "experiment_id": C.EXPERIMENT_ID,
                "state_id": panel_row["state_id"],
                "family": family,
                "candidate_spec_id": panel_row["candidate_spec_id"],
                "canonical_spec_sha256": spec["canonical_spec_sha256"],
            }
            assignment_sha = hashlib.sha256(
                C.canonical_json_bytes(key_projection)[:-1]
            ).hexdigest()
            ranked.append((assignment_sha, panel_row["state_id"], panel_row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if len(ranked) != C.STATES_PER_FAMILY:
            raise PhysicalGraphEdgeHandoffMetricsError("split family population drift")
        for rank, (assignment_sha, state_id, panel_row) in enumerate(ranked):
            role = "DEVELOPMENT" if rank < 12 else "DEVELOPMENT_HELDOUT"
            if panel_row["role"] != role:
                raise PhysicalGraphEdgeHandoffMetricsError("panel role not hash-derived")
            expected_by_id[state_id] = {
                "state_id": state_id,
                "role": role,
                "assignment_sha256": assignment_sha,
                "rank_within_family": rank,
            }
            population_projection.append({
                "state_id": state_id,
                "family": family,
                "candidate_spec_id": panel_row["candidate_spec_id"],
                "assignment_sha256": assignment_sha,
            })
    population_projection.sort(key=lambda row: row["state_id"])
    expected_population_sha = hashlib.sha256(
        C.canonical_json_bytes(population_projection)[:-1]
    ).hexdigest()
    if root["complete_eligible_population_sha256"] != expected_population_sha:
        raise PhysicalGraphEdgeHandoffMetricsError("eligible population digest drift")
    observed_by_id: dict[str, dict[str, Any]] = {}
    for index, value_row in enumerate(assignments):
        row = _mapping(value_row, SPLIT_ASSIGNMENT_FIELDS, f"assignment[{index}]")
        state_id = row["state_id"]
        if state_id in observed_by_id or state_id not in expected_by_id:
            raise PhysicalGraphEdgeHandoffMetricsError("split identity drift")
        if row != expected_by_id[state_id]:
            raise PhysicalGraphEdgeHandoffMetricsError("split assignment regeneration drift")
        observed_by_id[state_id] = row
    if set(observed_by_id) != set(expected_by_id):
        raise PhysicalGraphEdgeHandoffMetricsError("split coverage drift")
    return root


def validate_graph_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, GRAPH_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.graph_manifest.v1",
        "graph_manifest",
    )
    rows = _sequence(root["graphs"], "graph_manifest.graphs")
    if len(rows) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("graph count drift")
    panel_by_id = {row["state_id"]: row for row in panel["states"]}
    seen: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, GRAPH_RECORD_FIELDS, f"graph[{index}]")
        state_id = row["state_id"]
        if state_id != panel["states"][index]["state_id"]:
            raise PhysicalGraphEdgeHandoffMetricsError("graph canonical row order drift")
        if state_id in seen or state_id not in panel_by_id:
            raise PhysicalGraphEdgeHandoffMetricsError("graph state identity drift")
        seen.add(state_id); panel_row = panel_by_id[state_id]
        for field in ("graph_id", "source_node_id", "target_node_id", "directed_edge_id"):
            if row[field] != panel_row[field]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"graph {field} drift")
        nodes = [_mapping(item, GRAPH_NODE_FIELDS, "graph node") for item in _sequence(row["nodes"], "nodes")]
        edges = [_mapping(item, GRAPH_EDGE_FIELDS, "graph edge") for item in _sequence(row["edges"], "edges")]
        node_by_id = {item["node_id"]: item for item in nodes}
        edge_by_id = {item["edge_id"]: item for item in edges}
        if len(node_by_id) != len(nodes) or len(edge_by_id) != len(edges):
            raise PhysicalGraphEdgeHandoffMetricsError("graph node/edge identity duplicate")
        if row["source_node_id"] not in node_by_id or row["target_node_id"] not in node_by_id or row["directed_edge_id"] not in edge_by_id:
            raise PhysicalGraphEdgeHandoffMetricsError("graph authority target absent")
        selected = edge_by_id[row["directed_edge_id"]]
        if selected["source_node_id"] != row["source_node_id"] or selected["target_node_id"] != row["target_node_id"] or selected["oracle_reachable"] is not True or selected["physically_executable"] is not True:
            raise PhysicalGraphEdgeHandoffMetricsError("selected directed-edge authority drift")
        panel_spec = next(
            spec for spec in C.build_candidate_specs()
            if spec["candidate_spec_id"] == panel_row["candidate_spec_id"]
        )
        geometry = panel_spec["geometry"]
        expected_edges = [geometry["selected_directed_edge"], *geometry["competing_directed_edges"]]
        if set(edge_by_id) != {item["edge_id"] for item in expected_edges}:
            raise PhysicalGraphEdgeHandoffMetricsError("graph port edge inventory drift")
        expected_nodes: dict[str, dict[str, Any]] = {
            "source": {
                "node_kind": "SOURCE",
                "centre_world": geometry["source_node"]["centre_world"],
                "boundary_polygon_world": geometry["source_node"]["boundary_polygon_world"],
            },
            "target": {
                "node_kind": "TARGET",
                "centre_world": geometry["target_node"]["centre_world"],
                "boundary_polygon_world": geometry["target_node"]["boundary_polygon_world"],
            },
        }
        expected_graph_edges: dict[str, dict[str, Any]] = {
            geometry["selected_directed_edge"]["edge_id"]: {
                "source_node_id": "source", "target_node_id": "target",
                "port_label": geometry["selected_directed_edge"]["route_direction"],
                "route_polyline_world": geometry["teacher_route_polyline_world"],
                "edge_kind": "SELECTED",
            }
        }
        source_centre = geometry["source_node"]["centre_world"]
        for competitor_index, competitor in enumerate(geometry["competing_directed_edges"]):
            target_id = f"competing-node-{competitor_index}"
            polygon = competitor["edge_region_polygon_world"]
            centre = [
                math.fsum(float(point[axis]) for point in polygon) / len(polygon)
                for axis in range(2)
            ]
            midpoint = [
                (competitor["opening_segment_world"][0][axis]
                 + competitor["opening_segment_world"][1][axis]) / 2.0
                for axis in range(2)
            ]
            expected_nodes[target_id] = {
                "node_kind": "COMPETING", "centre_world": centre,
                "boundary_polygon_world": polygon,
            }
            expected_graph_edges[competitor["edge_id"]] = {
                "source_node_id": "source", "target_node_id": target_id,
                "port_label": competitor["boundary_side"],
                "route_polyline_world": [source_centre, midpoint, centre],
                "edge_kind": "COMPETING",
            }
        if set(node_by_id) != set(expected_nodes):
            raise PhysicalGraphEdgeHandoffMetricsError("graph node inventory drift")
        for node in nodes:
            expected_node = expected_nodes[node["node_id"]]
            if any(
                node[field] != expected_node[field]
                for field in ("node_kind", "centre_world", "boundary_polygon_world")
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("graph node geometry drift")
            _vector(node["centre_world"], 2, "node centre")
            polygon = _sequence(node["boundary_polygon_world"], "node boundary")
            if len(polygon) < 3:
                raise PhysicalGraphEdgeHandoffMetricsError("node boundary degenerate")
            for point in polygon: _vector(point, 2, "boundary point")
        for edge in edges:
            polyline = _sequence(edge["route_polyline_world"], "edge route")
            if len(polyline) < 2: raise PhysicalGraphEdgeHandoffMetricsError("edge route degenerate")
            for point in polyline: _vector(point, 2, "route point")
            if _finite(edge["edge_length_m"], "edge length") <= 0: raise PhysicalGraphEdgeHandoffMetricsError("edge length nonpositive")
            expected_edge = next(item for item in expected_edges if item["edge_id"] == edge["edge_id"])
            expected_graph_edge = expected_graph_edges[edge["edge_id"]]
            expected_length = math.fsum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(
                    expected_graph_edge["route_polyline_world"],
                    expected_graph_edge["route_polyline_world"][1:],
                )
            )
            if (
                any(
                    edge[field] != expected_graph_edge[field]
                    for field in (
                        "source_node_id", "target_node_id", "port_label",
                        "route_polyline_world", "edge_kind",
                    )
                )
                or not _close(
                    _finite(edge["edge_length_m"], "edge length"), expected_length,
                    C.NUMERICAL_TOLERANCES["route_arclength_m"],
                )
                or edge["oracle_reachable"] is not True
                or edge["physically_executable"] is not True
                or edge["opening_segment_world"] != expected_edge["opening_segment_world"]
                or edge["opening_normal_world"] != expected_edge["opening_normal_world"]
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("graph transverse-port geometry drift")
            segment = _sequence(edge["opening_segment_world"], "edge opening segment")
            if len(segment) != 2:
                raise PhysicalGraphEdgeHandoffMetricsError("edge opening segment drift")
            for point in segment: _vector(point, 2, "edge opening point")
            normal = _vector(edge["opening_normal_world"], 2, "edge opening normal")
            if not _close(math.hypot(*normal), 1.0, C.NUMERICAL_TOLERANCES["se2_position_m"]):
                raise PhysicalGraphEdgeHandoffMetricsError("edge opening normal not unit")
    return root


def validate_state_snapshot_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, SNAPSHOT_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.state_snapshot_index.v1",
        "state_snapshot_index",
    )
    binding = _validate_file_binding(root["snapshots_file"], "snapshots_file", "state_snapshots.npz")
    fixture = _mapping(root["reset_fixture"], RESET_FIXTURE_FIELDS, "reset_fixture")
    if fixture != {"fixture_id": "SERIALIZED_RESTORE_TWO_TRIAL_V1", "serialized_restore_used": True, "clone_equivalence_used": False, "passed": True}:
        raise PhysicalGraphEdgeHandoffMetricsError("reset fixture authority drift")
    rows = _sequence(root["records"], "snapshot records")
    if len(rows) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("snapshot count drift")
    panel_ids = {row["state_id"] for row in panel["states"]}
    selected_qualification = {
        row["state_id"]: row
        for row in panel["prospective_pool_selection"]["qualification_rows"]
        if row["selected"]
    }
    seen: set[str] = set(); trace_indices: set[int] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, SNAPSHOT_RECORD_FIELDS, f"snapshot[{index}]")
        if row["state_id"] != panel["states"][index]["state_id"]:
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot canonical row order drift")
        if row["state_id"] not in panel_ids or row["state_id"] in seen:
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot state identity drift")
        seen.add(row["state_id"]); _string(row["snapshot_id"], "snapshot_id")
        qualification = selected_qualification[row["state_id"]]
        if row["source_teacher_trace_id"] != qualification["teacher_trace_id"] or row["teacher_initial_state_sha256"] != qualification["initial_decision_state_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot/teacher initial-state binding drift")
        payload = _validate_slice(row["snapshot_payload"], "snapshot_payload", "state_snapshots.npz")
        if payload["file_sha256"] != binding["sha256"] or payload["member"] != "snapshot_payload_bytes":
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot payload binding drift")
        if payload["slice_sha256"] != row["teacher_initial_state_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("selected teacher snapshot bytes were not retained")
        if _integer(row["array_row_index"], "array_row_index") != index:
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot row order drift")
        for field in (
            "serialized_solver_state_sha256", "serialized_controller_state_sha256",
            "serialized_rng_state_sha256", "policy_last_action_sha256",
            "base_pose_world_sha256", "base_twist_world_sha256",
            "joint_position_sha256", "joint_velocity_sha256",
            "camera_world_transform_sha256",
            "controller_observation_sha256", "previous_policy_action_sha256",
            "torch_cpu_rng_state_sha256", "previous_applied_command_sha256",
            "command_history_sha256", "control_history_sha256",
            "low_level_policy_state_sha256",
        ): _sha(row[field], field)
        device_count = _integer(row["torch_device_count"], "torch_device_count")
        _finite(row["capture_timestamp_s"], "capture_timestamp_s")
        device_digests = _sequence(row["torch_device_rng_state_sha256s"], "torch device RNG")
        if len(device_digests) != device_count: raise PhysicalGraphEdgeHandoffMetricsError("torch device RNG coverage drift")
        for digest in device_digests: _sha(digest, "torch device RNG digest")
        for field in ("solver_field_inventory", "controller_field_inventory", "rng_field_inventory"):
            values = _sequence(row[field], field)
            if not values or len(set(values)) != len(values): raise PhysicalGraphEdgeHandoffMetricsError(f"{field} drift")
            for item in values: _string(item, field)
        trials = _sequence(row["reset_trials"], "reset_trials")
        if len(trials) != 2: raise PhysicalGraphEdgeHandoffMetricsError("reset trial count drift")
        trial_projection = []
        for trial_index, trial_value in enumerate(trials):
            trial = _mapping(trial_value, RESET_TRIAL_FIELDS, "reset trial")
            if trial["trial_index"] != trial_index or trial["serialized_restore_used"] is not True:
                raise PhysicalGraphEdgeHandoffMetricsError("serialized reset trial drift")
            if trial["restored_snapshot_sha256"] != payload["slice_sha256"]:
                raise PhysicalGraphEdgeHandoffMetricsError("reset trial did not restore retained snapshot bytes")
            trace_index = _integer(trial["trace_index"], "fixture trace_index")
            if trace_index not in range(*C.TRACE_INDEX_RANGES["reset_fixture"]) or trace_index in trace_indices:
                raise PhysicalGraphEdgeHandoffMetricsError("fixture trace index drift")
            trace_indices.add(trace_index)
            trace_slice = _validate_slice(trial["trace_slice"], "fixture trace slice", "candidate_traces.npz")
            if trace_slice["member"] != "timestamp_s": raise PhysicalGraphEdgeHandoffMetricsError("fixture trace slice member drift")
            if trace_slice["stop"] - trace_slice["start"] != C.RESET_FIXTURE_PHYSICS_SAMPLES:
                raise PhysicalGraphEdgeHandoffMetricsError("fixture trace duration drift")
            digests = trial["trace_array_slice_sha256s"]
            if not isinstance(digests, Mapping) or set(digests) != set(C.NPZ_PAYLOAD_AUTHORITY["candidate_traces.npz"]) - {"trace_offsets"}:
                raise PhysicalGraphEdgeHandoffMetricsError("fixture trace array coverage drift")
            for digest in digests.values(): _sha(digest, "fixture trace digest")
            for field in (
                "restored_snapshot_sha256", "post_restore_state_sha256", "current_rgb_sha256",
                "joint_position_sha256", "joint_velocity_sha256", "controller_state_sha256",
                "rng_state_sha256", "requested_command_sequence_sha256",
                "post_slew_applied_command_sequence_sha256", "contact_sequence_sha256",
            ): _sha(trial[field], field)
            _vector(trial["base_pose_world"], 7, "reset base pose")
            if _string(trial["termination_reason"], "termination_reason") != C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY["completion_termination_reason"]:
                raise PhysicalGraphEdgeHandoffMetricsError("reset fixture did not complete H3")
            _boolean(trial["stuck"], "stuck")
            trial_projection.append(trial)
        if trial_projection[0]["restored_snapshot_sha256"] != trial_projection[1]["restored_snapshot_sha256"] or any(
            trial_projection[0][field] != trial_projection[1][field]
            for field in (
                "post_restore_state_sha256", "current_rgb_sha256", "controller_state_sha256",
                "rng_state_sha256", "requested_command_sequence_sha256",
                "post_slew_applied_command_sequence_sha256", "contact_sequence_sha256",
                "termination_reason", "stuck",
            )
        ):
            raise PhysicalGraphEdgeHandoffMetricsError("two reset fixture trials differ")
        comparison = _mapping(
            row["reset_pair_comparison"], RESET_PAIR_COMPARISON_FIELDS,
            "reset_pair_comparison",
        )
        if comparison["state_id"] != row["state_id"] or comparison["trial_indices"] != [0, 1] or comparison["physics_sample_count"] != C.RESET_FIXTURE_PHYSICS_SAMPLES:
            raise PhysicalGraphEdgeHandoffMetricsError("reset comparison identity drift")
        tolerance_fields = (
            ("maximum_base_position_error_m", "reset_base_position_m"),
            ("maximum_base_quaternion_component_error", "reset_base_quaternion_component"),
            ("maximum_base_twist_error", "reset_base_twist"),
            ("maximum_joint_position_error_rad", "reset_joint_position_rad"),
            ("maximum_joint_velocity_error_rad_s", "reset_joint_velocity_rad_s"),
            ("endpoint_position_error_m", "repeat_endpoint_position_m"),
            ("endpoint_heading_error_rad", "repeat_endpoint_heading_rad"),
        )
        comparison_pass = True
        for field, tolerance_id in tolerance_fields:
            observed = _finite(comparison[field], field)
            comparison_pass &= observed <= C.NUMERICAL_TOLERANCES[tolerance_id]
        exact_members = comparison["exact_member_equal"]
        expected_exact_members = set(C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY["exact_members"])
        if not isinstance(exact_members, Mapping) or set(exact_members) != expected_exact_members:
            raise PhysicalGraphEdgeHandoffMetricsError("reset exact-member coverage drift")
        for value in exact_members.values():
            comparison_pass &= _boolean(value, "reset exact member")
        comparison_pass &= _boolean(comparison["termination_reason_equal"], "termination reason equal")
        comparison_pass &= _boolean(comparison["stuck_equal"], "stuck equal")
        if comparison["passed"] is not comparison_pass or not comparison_pass:
            raise PhysicalGraphEdgeHandoffMetricsError("independent reset trace comparison failed")
    if trace_indices != set(range(*C.TRACE_INDEX_RANGES["reset_fixture"])):
        raise PhysicalGraphEdgeHandoffMetricsError("fixture trace coverage drift")
    return root


def validate_teacher_trace_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, TEACHER_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.teacher_trace_index.v1",
        "teacher_trace_index",
    )
    binding = _validate_file_binding(root["traces_file"], "teacher traces file", "teacher_traces.npz")
    rows = _sequence(root["records"], "teacher records")
    if len(rows) != C.TEACHER_TRACE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("teacher trace count drift")
    specifications = C.build_candidate_specs()
    panel_by_id = {row["state_id"]: row for row in panel["states"]}
    qualification_by_spec = {
        row["candidate_spec_id"]: row
        for row in panel["prospective_pool_selection"]["qualification_rows"]
    }
    seen: set[str] = set()
    for index, (value_row, spec) in enumerate(zip(rows, specifications)):
        row = _mapping(value_row, TEACHER_RECORD_FIELDS, f"teacher[{index}]")
        state_id = row["state_id"]
        if state_id in seen or state_id != spec["state_id"]:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher state drift")
        seen.add(state_id)
        for field in (
            "candidate_spec_id", "family", "stratum_index", "variant_index",
            "canonical_spec_sha256",
        ):
            if row[field] != spec[field]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"teacher {field} drift")
        for field, expected in (
            ("source_node_id", "source"), ("target_node_id", "target"),
            ("directed_edge_id", "selected-edge"),
        ):
            if row[field] != expected:
                raise PhysicalGraphEdgeHandoffMetricsError(f"teacher {field} drift")
        if row["trace_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher trace order drift")
        _string(row["teacher_trace_id"], "teacher_trace_id")
        _sha(row["initial_decision_state_sha256"], "initial decision state")
        _sha(row["current_rgb_sha256"], "teacher current RGB")
        trace_slice = _validate_slice(row["trace_slice"], "teacher trace slice", "teacher_traces.npz")
        if trace_slice["file_sha256"] != binding["sha256"] or trace_slice["member"] != "timestamp_s":
            raise PhysicalGraphEdgeHandoffMetricsError("teacher trace file binding drift")
        sample_count = _integer(row["sample_count"], "teacher sample_count", minimum=2)
        if trace_slice["stop"] - trace_slice["start"] != sample_count:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher trace sample count drift")
        digests = row["trace_array_slice_sha256s"]
        expected_members = set(C.NPZ_PAYLOAD_AUTHORITY["teacher_traces.npz"]) - {"trace_offsets"}
        if not isinstance(digests, Mapping) or set(digests) != expected_members:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher trace array coverage drift")
        for digest in digests.values(): _sha(digest, "teacher slice digest")
        left_source = _boolean(row["left_source_region"], "left_source_region")
        if left_source:
            source_exit = _integer(row["first_source_exit_sample_index"], "source exit sample")
            if source_exit <= 0 or source_exit >= sample_count:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher source-exit index drift")
        elif row["first_source_exit_sample_index"] is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher without source exit has exit index")
        crossed = _boolean(row["crossed_directed_port"], "crossed_directed_port")
        crossing: int | None = None
        if crossed:
            crossing = _integer(row["first_crossing_sample_index"], "teacher crossing")
            if crossing <= 0 or crossing >= sample_count:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher crossing index drift")
            fraction = _finite(row["crossing_segment_fraction"], "crossing fraction")
            if not 0.0 <= fraction <= 1.0:
                raise PhysicalGraphEdgeHandoffMetricsError("crossing fraction drift")
            normal_dot = _finite(row["crossing_directed_normal_dot"], "crossing normal dot")
            lateral_fraction = _finite(row["crossing_lateral_fraction"], "crossing lateral fraction")
            velocity = _vector(row["crossing_velocity_world_xy"], 2, "teacher crossing velocity")
            velocity_heading = _finite(row["crossing_velocity_heading_world_rad"], "teacher crossing velocity heading")
            if math.hypot(*velocity) <= 0.0 or not _close(
                _angle(math.atan2(velocity[1], velocity[0]) - velocity_heading),
                0.0, C.NUMERICAL_TOLERANCES["se2_heading_rad"],
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("teacher crossing velocity heading drift")
        else:
            if any(row[field] is not None for field in (
                "first_crossing_sample_index", "crossing_segment_fraction",
                "crossing_directed_normal_dot", "crossing_lateral_fraction",
                "crossing_velocity_world_xy", "crossing_velocity_heading_world_rad",
            )):
                raise PhysicalGraphEdgeHandoffMetricsError("uncrossed teacher has crossing evidence")
            normal_dot = float("-inf"); lateral_fraction = float("inf")
        dwell = _integer(row["beyond_port_consecutive_physics_samples"], "teacher port dwell")
        target_early = _boolean(row["target_entered_before_dwell_complete"], "teacher target early")
        if not crossed and (dwell != 0 or target_early):
            raise PhysicalGraphEdgeHandoffMetricsError(
                "uncrossed teacher has selected-port dwell evidence"
            )
        route_progress = _finite(row["route_progress_m"], "teacher route progress")
        if row["positive_route_progress"] is not (route_progress > 0.0):
            raise PhysicalGraphEdgeHandoffMetricsError("teacher route-progress projection drift")
        contact_free = _boolean(row["contact_free"], "teacher contact_free")
        positive_progress = _boolean(
            row["positive_route_progress"], "teacher positive_route_progress"
        )
        competing_entered = _boolean(
            row["competing_port_entered"], "teacher competing_port_entered"
        )
        teacher_valid = bool(
            contact_free and left_source and crossed
            and positive_progress and normal_dot > 0.0
            and 0.0 <= lateral_fraction <= 1.0 and not competing_entered
            and (dwell >= C.PORT_DWELL_PHYSICS_SAMPLES or target_early)
            and row["directed_port_defined"] and row["graph_edge_physically_executable"]
        )
        if row["teacher_valid"] is not teacher_valid:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher qualification drift")
        for field in ("goal_reachable", "directed_port_defined", "graph_edge_physically_executable", "current_rgb_valid"):
            _boolean(row[field], field)
        reached_target = _boolean(row["reached_target_node"], "reached_target_node")
        target_index: int | None = None
        if reached_target:
            target_index = _integer(row["first_target_entry_sample_index"], "target entry sample")
            if target_index < 0 or target_index >= sample_count:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher target-entry index drift")
        elif row["first_target_entry_sample_index"] is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher without target entry has target index")
        expected_target_early = bool(
            crossing is not None and target_index is not None
            and target_index >= crossing
            and target_index - crossing + 1 < C.PORT_DWELL_PHYSICS_SAMPLES
        )
        if target_early is not expected_target_early:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher target-before-dwell projection drift")
        _string(row["where_reached"], "where_reached")
        _finite(row["endpoint_lateral_error_m"], "teacher endpoint lateral error")
        _finite(row["endpoint_angular_error_rad"], "teacher endpoint angular error")
        _boolean(row["successor_viable"], "teacher successor viability")
        _boolean(row["stuck"], "teacher stuck")
        qualification = qualification_by_spec[row["candidate_spec_id"]]
        expected_selected = qualification["selected"]
        if row["selected"] is not expected_selected:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher selected projection drift")
        if expected_selected and state_id not in panel_by_id:
            raise PhysicalGraphEdgeHandoffMetricsError("selected teacher missing final panel state")
        if (not expected_selected) and state_id in panel_by_id:
            raise PhysicalGraphEdgeHandoffMetricsError("unselected teacher appears in final panel")
        projected = {
            "teacher_trace_contact_free": row["contact_free"],
            "teacher_left_source_region": row["left_source_region"],
            "teacher_crossed_directed_port": crossed,
            "teacher_positive_route_progress": row["positive_route_progress"],
            "teacher_competing_port_entered": row["competing_port_entered"],
            "teacher_normal_positive": crossed and normal_dot > 0.0,
            "teacher_within_lateral_bounds": crossed and 0.0 <= lateral_fraction <= 1.0,
            "teacher_dwell_satisfied": (
                dwell >= C.PORT_DWELL_PHYSICS_SAMPLES or target_early
            ),
            "teacher_valid": teacher_valid,
            "goal_reachable": row["goal_reachable"],
            "directed_port_defined": row["directed_port_defined"],
            "graph_edge_physically_executable": row["graph_edge_physically_executable"],
            "current_rgb_valid": row["current_rgb_valid"],
        }
        for field, expected in projected.items():
            if qualification[field] is not expected:
                raise PhysicalGraphEdgeHandoffMetricsError(f"teacher qualification {field} drift")
        if qualification["teacher_trace_id"] != row["teacher_trace_id"] or qualification["teacher_trace_index"] != index or qualification["initial_decision_state_sha256"] != row["initial_decision_state_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher qualification identity binding drift")
        if qualification["teacher_trace_slice_sha256"] != row["trace_array_slice_sha256s"]["base_pose_world"]:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher qualification trace binding drift")
    return root


def _segment_lateral_fraction(point: Sequence[float], segment: Sequence[Sequence[float]]) -> tuple[float, float]:
    a = segment[0]; b = segment[1]
    dx = b[0] - a[0]; dy = b[1] - a[1]; denom = dx * dx + dy * dy
    if denom <= 0: raise PhysicalGraphEdgeHandoffMetricsError("opening segment degenerate")
    fraction = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom
    distance = abs((point[0] - a[0]) * dy - (point[1] - a[1]) * dx) / math.sqrt(denom)
    return fraction, distance


def validate_edge_port_index(
    value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest); graphs = validate_graph_manifest(graph_manifest, panel)
    teachers = validate_teacher_trace_index(teacher_trace_index, panel)
    root = _validate_document(
        value, EDGE_PORT_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.edge_port_index.v1",
        "edge_port_index",
    )
    rows = _sequence(root["records"], "edge ports")
    if len(rows) != C.STATE_COUNT: raise PhysicalGraphEdgeHandoffMetricsError("edge port count drift")
    panel_by_id = {row["state_id"]: row for row in panel["states"]}
    graph_by_id = {row["state_id"]: row for row in graphs["graphs"]}
    teacher_by_id = {row["state_id"]: row for row in teachers["records"]}
    seen: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, EDGE_PORT_RECORD_FIELDS, f"edge_port[{index}]")
        state_id = row["state_id"]
        if state_id != panel["states"][index]["state_id"]:
            raise PhysicalGraphEdgeHandoffMetricsError("edge-port canonical row order drift")
        if state_id in seen or state_id not in panel_by_id: raise PhysicalGraphEdgeHandoffMetricsError("edge port state drift")
        seen.add(state_id); panel_row = panel_by_id[state_id]; teacher = teacher_by_id[state_id]
        for field in ("directed_edge_id", "source_node_id", "target_node_id"):
            if row[field] != panel_row[field]: raise PhysicalGraphEdgeHandoffMetricsError(f"edge port {field} drift")
        if row["teacher_trace_id"] != teacher["teacher_trace_id"]: raise PhysicalGraphEdgeHandoffMetricsError("edge port teacher binding drift")
        before = _integer(row["crossing_sample_before"], "crossing before")
        after = _integer(row["crossing_sample_after"], "crossing after")
        if after != before + 1 or after != teacher["first_crossing_sample_index"]:
            raise PhysicalGraphEdgeHandoffMetricsError("first crossing binding drift")
        fraction = _finite(row["crossing_fraction"], "crossing fraction")
        if not _close(fraction, teacher["crossing_segment_fraction"], 0.0): raise PhysicalGraphEdgeHandoffMetricsError("crossing fraction authority drift")
        crossing_velocity = _vector(
            row["teacher_crossing_velocity_world_xy"], 2,
            "teacher crossing velocity",
        )
        crossing_heading = _finite(
            row["teacher_crossing_velocity_heading_world_rad"],
            "teacher crossing velocity heading",
        )
        if crossing_velocity != teacher["crossing_velocity_world_xy"] or not _close(
            crossing_heading, teacher["crossing_velocity_heading_world_rad"], 0.0
        ):
            raise PhysicalGraphEdgeHandoffMetricsError("teacher crossing-direction binding drift")
        port = _vector(row["directed_port_world"], 3, "directed port")
        lookahead = _vector(row["route_lookahead_world"], 3, "route lookahead")
        polygon = _sequence(row["source_boundary_polygon_world"], "source boundary")
        if len(polygon) < 3: raise PhysicalGraphEdgeHandoffMetricsError("source boundary degenerate")
        for point in polygon: _vector(point, 2, "source boundary point")
        route = _sequence(row["route_polyline_world"], "route polyline")
        if len(route) < 2: raise PhysicalGraphEdgeHandoffMetricsError("route polyline degenerate")
        for point in route: _vector(point, 2, "route point")
        edge = next(item for item in graph_by_id[state_id]["edges"] if item["edge_id"] == row["directed_edge_id"])
        spec = next(item for item in C.build_candidate_specs() if item["candidate_spec_id"] == panel_row["candidate_spec_id"])
        opening = spec["geometry"]["selected_directed_edge"]["opening_segment_world"]
        lateral_fraction, lateral_distance = _segment_lateral_fraction(port[:2], opening)
        if not -C.NUMERICAL_TOLERANCES["se2_position_m"] <= lateral_fraction <= 1.0 + C.NUMERICAL_TOLERANCES["se2_position_m"] or lateral_distance > C.NUMERICAL_TOLERANCES["se2_position_m"]:
            raise PhysicalGraphEdgeHandoffMetricsError("actual teacher crossing is outside selected transverse opening")
        normal = spec["geometry"]["selected_directed_edge"]["opening_normal_world"]
        if math.cos(port[2]) * normal[0] + math.sin(port[2]) * normal[1] <= 0:
            raise PhysicalGraphEdgeHandoffMetricsError("directed port heading reverses opening normal")
        if not _close(_angle(port[2] - math.atan2(normal[1], normal[0])), 0.0, C.NUMERICAL_TOLERANCES["se2_heading_rad"]):
            raise PhysicalGraphEdgeHandoffMetricsError("canonical port heading is not directed normal")
        remaining = _finite(row["remaining_route_length_m"], "remaining route length")
        if remaining < 0: raise PhysicalGraphEdgeHandoffMetricsError("remaining route length negative")
        if row["route_lookahead_clipped"] is not (remaining < C.ROUTE_LOOKAHEAD_DISTANCE_M):
            raise PhysicalGraphEdgeHandoffMetricsError("lookahead clip disposition drift")
        if row["port_definition"] != "first actual teacher source-boundary crossing through selected transverse opening":
            raise PhysicalGraphEdgeHandoffMetricsError("port definition drift")
        del edge, lookahead
    return root


def validate_waypoint_contracts(
    value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any],
    edge_port_index: Mapping[str, Any],
) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest); graphs = validate_graph_manifest(graph_manifest, panel)
    ports = _validate_document(edge_port_index, EDGE_PORT_INDEX_FIELDS, "physical_graph_edge_handoff_qualification_v1.edge_port_index.v1", "edge_port_index")
    root = _validate_document(
        value, WAYPOINT_CONTRACT_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.waypoint_contracts.v1",
        "waypoint_contracts",
    )
    if root["target_ids"] != list(C.TARGET_IDS) or root["feature_order"] != list(C.TARGET_FEATURE_ORDER):
        raise PhysicalGraphEdgeHandoffMetricsError("waypoint target authority drift")
    if root["training_contract_binding"] != C.CURRENT_VISUAL_RANKER_BINDING or root["original_ranker_support"] != C.ORIGINAL_RANKER_TRAINING_SUPPORT or root["contract_difference_inventory"] != list(C.RANKER_CONTRACT_DIFFERENCE_INVENTORY):
        raise PhysicalGraphEdgeHandoffMetricsError("ranker training-contract evidence drift")
    rows = _sequence(root["rows"], "waypoint rows")
    if len(rows) != C.STATE_COUNT * len(C.TARGET_IDS): raise PhysicalGraphEdgeHandoffMetricsError("waypoint row count drift")
    graph_by_id = {row["state_id"]: row for row in graphs["graphs"]}; port_by_id = {row["state_id"]: row for row in ports["records"]}
    seen: set[tuple[str, str]] = set(); support_counts = {target: {"inside": 0, "outside": 0} for target in C.TARGET_IDS}
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, WAYPOINT_ROW_FIELDS, f"waypoint[{index}]")
        expected_identity = (
            panel["states"][index // len(C.TARGET_IDS)]["state_id"],
            C.TARGET_IDS[index % len(C.TARGET_IDS)],
        )
        identity = (row["state_id"], row["target_id"])
        if identity != expected_identity:
            raise PhysicalGraphEdgeHandoffMetricsError("waypoint canonical row order drift")
        if identity in seen or row["state_id"] not in graph_by_id or row["target_id"] not in C.TARGET_IDS:
            raise PhysicalGraphEdgeHandoffMetricsError("waypoint identity drift")
        seen.add(identity); state_id, target_id = identity
        source = _vector(row["source_body_pose_world"], 3, "source body pose")
        world = _vector(row["target_world_pose"], 3, "target world pose")
        body = _vector(row["target_body_pose"], 3, "target body pose")
        if target_id == "TARGET_NODE_CENTRE":
            graph = graph_by_id[state_id]; node = next(item for item in graph["nodes"] if item["node_id"] == graph["target_node_id"])
            expected_world = [*node["centre_world"], world[2]]
        elif target_id == "DIRECTED_EDGE_PORT": expected_world = port_by_id[state_id]["directed_port_world"]
        else: expected_world = port_by_id[state_id]["route_lookahead_world"]
        if any(not _close(a, b, C.NUMERICAL_TOLERANCES["se2_position_m"] if i < 2 else C.NUMERICAL_TOLERANCES["se2_heading_rad"]) for i, (a, b) in enumerate(zip(world, expected_world))):
            raise PhysicalGraphEdgeHandoffMetricsError("waypoint world target drift")
        dxw, dyw = world[0] - source[0], world[1] - source[1]; c, s = math.cos(source[2]), math.sin(source[2])
        expected_body = [c * dxw + s * dyw, -s * dxw + c * dyw, _angle(world[2] - source[2])]
        if any(not _close(a, b, C.NUMERICAL_TOLERANCES["se2_position_m"] if i < 2 else C.NUMERICAL_TOLERANCES["se2_heading_rad"]) for i, (a, b) in enumerate(zip(body, expected_body))):
            raise PhysicalGraphEdgeHandoffMetricsError("world-to-body transform drift")
        bearing = math.atan2(body[1], body[0])
        expected_features = [body[0], body[1], math.hypot(body[0], body[1]), bearing, math.sin(bearing), math.cos(bearing)]
        for field, expected in zip(("dx_m", "dy_m", "distance_m", "relative_heading_rad", "relative_heading_sin", "relative_heading_cos"), expected_features):
            if not _close(_finite(row[field], field), expected, 1e-9): raise PhysicalGraphEdgeHandoffMetricsError(f"{field} drift")
        for field, expected in (
            ("target_tangent_heading_rad", body[2]),
            ("target_tangent_heading_sin", math.sin(body[2])),
            ("target_tangent_heading_cos", math.cos(body[2])),
        ):
            if not _close(_finite(row[field], field), expected, 1e-9):
                raise PhysicalGraphEdgeHandoffMetricsError(f"{field} drift")
        _finite(row["route_intent_dx"], "route_intent_dx"); _finite(row["route_intent_dy"], "route_intent_dy")
        _vector(row["roundtrip_world_pose"], 3, "roundtrip world pose")
        position_error = _finite(row["position_transform_error_m"], "position transform error")
        heading_error = _finite(row["heading_transform_error_rad"], "heading transform error")
        valid = position_error <= C.NUMERICAL_TOLERANCES["inverse_transform_position_m"] and heading_error <= C.NUMERICAL_TOLERANCES["inverse_transform_heading_rad"]
        if row["transform_valid"] is not valid or not valid: raise PhysicalGraphEdgeHandoffMetricsError("waypoint inverse transform drift")
        support = C.ORIGINAL_RANKER_TRAINING_SUPPORT; tol = support["support_tolerance"]
        inside = all(
            interval[0] - tol <= value <= interval[1] + tol
            for value, interval in (
                (row["dx_m"], support["body_dx_m"]), (row["dy_m"], support["body_dy_m"]),
                (row["distance_m"], support["distance_m"]), (row["relative_heading_rad"], support["relative_heading_rad"]),
                (row["relative_heading_sin"], support["relative_heading_sin"]), (row["relative_heading_cos"], support["relative_heading_cos"]),
            )
        )
        support_counts[target_id]["inside" if inside else "outside"] += 1
    if root["target_support_counts"] != support_counts:
        raise PhysicalGraphEdgeHandoffMetricsError("target support count drift")
    return root


def validate_pixel_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, PIXEL_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.pixel_index.v1",
        "pixel_index",
    )
    binding = _validate_file_binding(root["rgb_file"], "rgb_file", "rgb_observations.npz")
    if root["hash_domain"] != C.CANONICAL_ENCODING_AUTHORITY["pixel_identity"]:
        raise PhysicalGraphEdgeHandoffMetricsError("pixel hash domain drift")
    rows = _sequence(root["records"], "pixel records")
    if len(rows) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("pixel record count drift")
    expected_state_ids = [row["state_id"] for row in panel["states"]]
    seen_capture: set[str] = set(); pixel_hashes: list[str] = []
    for index, (raw, state_id) in enumerate(zip(rows, expected_state_ids)):
        row = _mapping(raw, PIXEL_RECORD_FIELDS, f"pixel[{index}]")
        if row["state_id"] != state_id or row["rgb_row_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError("pixel row order drift")
        capture_id = _string(row["capture_id"], "capture_id")
        if capture_id in seen_capture:
            raise PhysicalGraphEdgeHandoffMetricsError("pixel capture identity duplicate")
        seen_capture.add(capture_id)
        pixel_hashes.append(_sha(row["pixel_sha256"], "pixel_sha256"))
        _sha(row["row_sha256"], "RGB row digest")
    unique_hashes = sorted(set(pixel_hashes))
    if root["unique_pixel_count"] != len(unique_hashes):
        raise PhysicalGraphEdgeHandoffMetricsError("unique pixel count drift")
    canonical_by_hash = {digest: index for index, digest in enumerate(unique_hashes)}
    for row in rows:
        if row["canonical_pixel_index"] != canonical_by_hash[row["pixel_sha256"]]:
            raise PhysicalGraphEdgeHandoffMetricsError("canonical pixel index drift")
    del binding
    return root


def validate_latent_index(value: Any, pixel_index: Mapping[str, Any]) -> dict[str, Any]:
    pixels = _validate_document(
        pixel_index, PIXEL_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.pixel_index.v1", "pixel_index",
    )
    root = _validate_document(
        value, LATENT_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.latent_index.v1", "latent_index",
    )
    _validate_file_binding(root["latents_file"], "latents_file", "canonical_latents.npz")
    if root["encoder_binding"] != C.VJEPA_ENCODER_BINDING:
        raise PhysicalGraphEdgeHandoffMetricsError("encoder binding drift")
    root["encoder_runtime_environment"] = validate_visual_runtime_environment(
        root["encoder_runtime_environment"], runtime_role="encoder"
    )
    if root["preprocessing_authority"] != C.CANONICAL_ENCODING_AUTHORITY["preprocessing"]:
        raise PhysicalGraphEdgeHandoffMetricsError("encoder preprocessing authority drift")
    expected_source = C.CANONICAL_ENCODING_AUTHORITY["external_encoder_source"]
    if root["external_encoder_source"] != {
        "repository_path": expected_source["repository_path"],
        "commit": expected_source["commit"],
        "worktree_clean": True,
    }:
        raise PhysicalGraphEdgeHandoffMetricsError("external encoder source custody drift")
    unique_hashes = sorted({row["pixel_sha256"] for row in pixels["records"]})
    rows = _sequence(root["records"], "latent records")
    if len(rows) != len(unique_hashes):
        raise PhysicalGraphEdgeHandoffMetricsError("latent record count drift")
    for index, (raw, pixel_sha) in enumerate(zip(rows, unique_hashes)):
        row = _mapping(raw, LATENT_RECORD_FIELDS, f"latent[{index}]")
        if row["canonical_pixel_index"] != index or row["pixel_sha256"] != pixel_sha:
            raise PhysicalGraphEdgeHandoffMetricsError("latent identity/order drift")
        if row["raw_token_row_index"] != index or row["spatial_descriptor_row_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError("latent row binding drift")
        _sha(row["raw_token_sha256"], "raw token digest")
        _sha(row["spatial_descriptor_sha256"], "descriptor digest")
        _sha(row["preprocessed_tensor_sha256"], "preprocessed tensor digest")
    return root


def _expected_requested_commands(candidate_index: int) -> list[list[float]]:
    _, blocks = C.CANDIDATE_BANK[candidate_index]
    expanded: list[list[float]] = []
    for primitive_id in blocks[: C.EXECUTED_BLOCK_COUNT]:
        expanded.extend(
            [list(C.CANDIDATE_PRIMITIVES[primitive_id])] * C.COMMAND_TICKS_PER_BLOCK
        )
    return expanded


def _validate_command_tracking_rows(value: Any, label: str) -> list[dict[str, Any]]:
    rows = _sequence(value, label)
    if len(rows) != C.HORIZON_TICKS[C.PRIMARY_HORIZON]:
        raise PhysicalGraphEdgeHandoffMetricsError(f"{label} count drift")
    result: list[dict[str, Any]] = []
    threshold = C.COMMAND_TRACKING_AUTHORITY["active_command_threshold"]
    for index, raw in enumerate(rows):
        row = _mapping(raw, COMMAND_TRACKING_ROW_FIELDS, f"{label}[{index}]")
        if row["command_tick_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError(f"{label} order drift")
        _vector(row["requested_command"], 3, "requested command")
        applied = _vector(row["post_slew_command"], 3, "post-slew command")
        _vector(row["mean_achieved_body_velocity"], 3, "achieved velocity")
        if row["active_vx"] is not (abs(applied[0]) > threshold) or row["active_yaw"] is not (abs(applied[2]) > threshold):
            raise PhysicalGraphEdgeHandoffMetricsError("command activity projection drift")
        result.append(row)
    return result


def validate_candidate_fanout_rows(
    value: Any, panel_manifest: Mapping[str, Any],
    state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    snapshots = _validate_document(
        state_snapshot_index, SNAPSHOT_INDEX_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.state_snapshot_index.v1",
        "state_snapshot_index",
    )
    rows = _sequence(value, "candidate_fanout")
    if len(rows) != C.BRANCH_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("candidate fanout count drift")
    snapshot_by_state = {row["state_id"]: row for row in snapshots["records"]}
    spec_by_id = {row["candidate_spec_id"]: row for row in C.build_candidate_specs()}
    physical_runtime_digest = panel["physical_runtime_environment"][
        "runtime_core_sha256"
    ]
    result: list[dict[str, Any]] = []
    for branch_offset, raw in enumerate(rows):
        state_offset, candidate_index = divmod(branch_offset, C.CANDIDATE_COUNT)
        panel_row = panel["states"][state_offset]
        row = _mapping(raw, CANDIDATE_FANOUT_FIELDS, f"fanout[{branch_offset}]")
        if row["physical_runtime_core_sha256"] != physical_runtime_digest:
            raise PhysicalGraphEdgeHandoffMetricsError(
                "fanout physical runtime cross-binding drift"
            )
        if row["state_id"] != panel_row["state_id"] or row["role"] != panel_row["role"] or row["family"] != panel_row["family"]:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout state projection drift")
        if row["candidate_index"] != candidate_index or row["candidate_id"] != C.CANDIDATE_IDS[candidate_index]:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout candidate identity drift")
        if row["branch_id"] != f"{panel_row['state_id']}::{C.CANDIDATE_IDS[candidate_index]}":
            raise PhysicalGraphEdgeHandoffMetricsError("branch identity drift")
        snapshot = snapshot_by_state[row["state_id"]]
        spec = spec_by_id[panel_row["candidate_spec_id"]]
        if row["snapshot_id"] != snapshot["snapshot_id"] or row["restored_snapshot_sha256"] != snapshot["snapshot_payload"]["slice_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout snapshot binding drift")
        trace_index = C.TRACE_INDEX_RANGES["candidate_fanout"][0] + branch_offset
        if row["trace_index"] != trace_index:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout trace index drift")
        trace_slice = _validate_slice(row["trace_slice"], "fanout trace slice", "candidate_traces.npz")
        if trace_slice["member"] != "timestamp_s" or trace_slice["stop"] - trace_slice["start"] != C.PHYSICS_STEPS_PER_BRANCH:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout trace slice drift")
        digests = row["trace_array_slice_sha256s"]
        expected_members = set(C.NPZ_PAYLOAD_AUTHORITY["candidate_traces.npz"]) - {"trace_offsets"}
        if not isinstance(digests, Mapping) or set(digests) != expected_members:
            raise PhysicalGraphEdgeHandoffMetricsError("fanout trace digest coverage drift")
        for digest in digests.values():
            _sha(digest, "fanout trace digest")
        if row["requested_commands"] != _expected_requested_commands(candidate_index):
            raise PhysicalGraphEdgeHandoffMetricsError("requested candidate plan drift")
        applied = _sequence(row["post_slew_applied_commands"], "post-slew plan")
        if len(applied) != C.HORIZON_TICKS[C.PRIMARY_HORIZON]:
            raise PhysicalGraphEdgeHandoffMetricsError("post-slew plan length drift")
        for command in applied: _vector(command, 3, "post-slew command")
        if row["physics_sample_count"] != C.PHYSICS_STEPS_PER_BRANCH:
            raise PhysicalGraphEdgeHandoffMetricsError("physics sample count drift")
        for field in ("h1_endpoint_body", "h2_endpoint_body", "h3_endpoint_body", "h3_endpoint_world"):
            _vector(row[field], 3, field)
        height = _finite(row["h3_base_height_m"], "H3 base height")
        roll = _finite(row["h3_roll_rad"], "H3 roll")
        pitch = _finite(row["h3_pitch_rad"], "H3 pitch")
        solver_finite = _boolean(row["h3_solver_finite"], "H3 solver finite")
        h3_disallowed_contact = _boolean(
            row["h3_disallowed_contact"], "H3 disallowed contact"
        )
        physics_contact = _boolean(row["physics_contact"], "physics contact")
        successor = bool(
            solver_finite and height >= C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["minimum_base_height_m"]
            and abs(roll) <= C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_roll_rad"]
            and abs(pitch) <= C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_pitch_rad"]
            and not h3_disallowed_contact
        )
        if row["successor_viable"] is not successor:
            raise PhysicalGraphEdgeHandoffMetricsError("successor viability drift")
        correct = _boolean(row["entered_correct_edge"], "entered_correct_edge")
        wrong = _boolean(row["entered_wrong_edge"], "entered_wrong_edge")
        no_edge = _boolean(row["no_edge"], "no_edge")
        competing_first = _boolean(
            row["competing_port_crossing_first"], "competing_port_crossing_first"
        )
        if sum((correct, wrong, no_edge)) != 1 or no_edge is not (not correct and not wrong):
            raise PhysicalGraphEdgeHandoffMetricsError("edge outcome partition drift")
        left_source = _boolean(row["left_source_region"], "left_source_region")
        if left_source:
            source_exit = _integer(row["first_source_exit_sample_index"], "source exit sample")
            if source_exit <= 0 or source_exit >= C.PHYSICS_STEPS_PER_BRANCH:
                raise PhysicalGraphEdgeHandoffMetricsError("candidate source-exit index drift")
        elif row["first_source_exit_sample_index"] is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("candidate without source exit has exit index")
        reached_target = _boolean(row["reached_target_node"], "reached_target_node")
        target_index: int | None = None
        if reached_target:
            target_index = _integer(row["first_target_entry_sample_index"], "target entry sample")
            if target_index < 0 or target_index >= C.PHYSICS_STEPS_PER_BRANCH:
                raise PhysicalGraphEdgeHandoffMetricsError("candidate target-entry index drift")
        elif row["first_target_entry_sample_index"] is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("candidate without target entry has target index")
        crossing_fields = (
            "port_crossing_sample_before", "port_crossing_sample_after",
            "port_crossing_fraction", "port_crossing_directed_normal_dot",
            "port_crossing_lateral_fraction",
        )
        if correct:
            if not left_source:
                raise PhysicalGraphEdgeHandoffMetricsError(
                    "correct-edge branch did not leave source region"
                )
            before = _integer(row[crossing_fields[0]], "correct crossing before")
            after = _integer(row[crossing_fields[1]], "correct crossing after")
            if (
                after != before + 1 or before < 0
                or after >= C.PHYSICS_STEPS_PER_BRANCH
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("correct crossing adjacency drift")
            fraction = _finite(row[crossing_fields[2]], "correct crossing fraction")
            normal_dot = _finite(row[crossing_fields[3]], "correct normal dot")
            lateral = _finite(row[crossing_fields[4]], "correct lateral fraction")
            if not 0.0 <= fraction <= 1.0 or normal_dot <= 0.0 or not 0.0 <= lateral <= 1.0:
                raise PhysicalGraphEdgeHandoffMetricsError("correct transverse-port crossing drift")
            displacement = _vector(row["port_crossing_displacement_world_xy"], 2, "correct crossing displacement")
            heading = _finite(row["port_crossing_direction_heading_world_rad"], "correct crossing heading")
            if math.hypot(*displacement) <= 0.0 or not _close(_angle(math.atan2(displacement[1], displacement[0]) - heading), 0.0, C.NUMERICAL_TOLERANCES["trace_summary_float"]):
                raise PhysicalGraphEdgeHandoffMetricsError("correct crossing direction drift")
            normal = spec["geometry"]["selected_directed_edge"]["opening_normal_world"]
            expected_dot = displacement[0] * normal[0] + displacement[1] * normal[1]
            if not _close(normal_dot, expected_dot, C.NUMERICAL_TOLERANCES["trace_summary_float"]):
                raise PhysicalGraphEdgeHandoffMetricsError("correct crossing normal projection drift")
            dwell = _integer(row["beyond_port_consecutive_physics_samples"], "correct port dwell")
            target_early = _boolean(
                row["target_entered_before_dwell_complete"],
                "candidate target before dwell",
            )
            expected_target_early = bool(
                target_index is not None and target_index >= after
                and target_index - after + 1 < C.PORT_DWELL_PHYSICS_SAMPLES
            )
            if target_early is not expected_target_early:
                raise PhysicalGraphEdgeHandoffMetricsError(
                    "candidate target-before-dwell projection drift"
                )
            if (
                dwell < C.PORT_DWELL_PHYSICS_SAMPLES
                and not target_early
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("correct port dwell drift")
            if competing_first:
                raise PhysicalGraphEdgeHandoffMetricsError("correct edge followed prior competing port")
        elif any(row[field] is not None for field in (*crossing_fields, "port_crossing_displacement_world_xy", "port_crossing_direction_heading_world_rad")):
            raise PhysicalGraphEdgeHandoffMetricsError("non-correct branch has selected-port crossing evidence")
        if not correct:
            if _integer(
                row["beyond_port_consecutive_physics_samples"],
                "non-correct selected-port dwell",
            ) != 0 or _boolean(
                row["target_entered_before_dwell_complete"],
                "non-correct target-before-dwell",
            ):
                raise PhysicalGraphEdgeHandoffMetricsError(
                    "non-correct branch has selected-port dwell evidence"
                )
        wrong_fields = (
            "first_wrong_edge_id", "wrong_port_crossing_sample_before",
            "wrong_port_crossing_sample_after", "wrong_port_crossing_fraction",
            "wrong_port_crossing_directed_normal_dot", "wrong_port_crossing_lateral_fraction",
            "wrong_port_crossing_displacement_world_xy",
            "wrong_port_crossing_direction_heading_world_rad",
        )
        if wrong:
            if not left_source:
                raise PhysicalGraphEdgeHandoffMetricsError(
                    "wrong-edge branch did not leave source region"
                )
            wrong_edge_id = _string(row[wrong_fields[0]], "wrong edge id")
            competing_by_id = {
                item["edge_id"]: item
                for item in spec["geometry"]["competing_directed_edges"]
            }
            if wrong_edge_id not in competing_by_id:
                raise PhysicalGraphEdgeHandoffMetricsError("wrong edge not a registered competing port")
            before = _integer(row[wrong_fields[1]], "wrong crossing before")
            after = _integer(row[wrong_fields[2]], "wrong crossing after")
            fraction = _finite(row[wrong_fields[3]], "wrong crossing fraction")
            normal_dot = _finite(row[wrong_fields[4]], "wrong normal dot")
            lateral = _finite(row[wrong_fields[5]], "wrong lateral fraction")
            if (
                after != before + 1 or before < 0
                or after >= C.PHYSICS_STEPS_PER_BRANCH
                or not 0.0 <= fraction <= 1.0 or normal_dot <= 0.0
                or not 0.0 <= lateral <= 1.0 or not competing_first
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("wrong transverse-port crossing drift")
            displacement = _vector(row[wrong_fields[6]], 2, "wrong crossing displacement")
            heading = _finite(row[wrong_fields[7]], "wrong crossing heading")
            if math.hypot(*displacement) <= 0.0 or not _close(_angle(math.atan2(displacement[1], displacement[0]) - heading), 0.0, C.NUMERICAL_TOLERANCES["trace_summary_float"]):
                raise PhysicalGraphEdgeHandoffMetricsError("wrong crossing direction drift")
            normal = competing_by_id[wrong_edge_id]["opening_normal_world"]
            expected_dot = displacement[0] * normal[0] + displacement[1] * normal[1]
            if not _close(normal_dot, expected_dot, C.NUMERICAL_TOLERANCES["trace_summary_float"]):
                raise PhysicalGraphEdgeHandoffMetricsError("wrong crossing normal projection drift")
        elif any(row[field] is not None for field in wrong_fields):
            raise PhysicalGraphEdgeHandoffMetricsError("non-wrong branch has competing-port crossing evidence")
        if no_edge and competing_first:
            raise PhysicalGraphEdgeHandoffMetricsError("no-edge branch claims a competing crossing")
        progress = _finite(row["port_progress_m"], "port progress")
        if row["positive_port_progress"] is not (progress > 0.0):
            raise PhysicalGraphEdgeHandoffMetricsError("positive progress projection drift")
        if _finite(row["lateral_error_m"], "lateral error") < 0.0:
            raise PhysicalGraphEdgeHandoffMetricsError("lateral error is negative")
        if _finite(row["angular_error_rad"], "angular error") < 0.0:
            raise PhysicalGraphEdgeHandoffMetricsError("angular error is negative")
        h3 = row["h3_endpoint_body"]
        activity = max(max(abs(x) for x in command) for command in row["requested_commands"])
        stuck = bool(
            activity > C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"]
            and math.hypot(h3[0], h3[1]) < C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_translation_threshold_m"]
            and abs(_angle(h3[2])) < C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_heading_threshold_rad"]
        )
        if row["stuck"] is not stuck:
            raise PhysicalGraphEdgeHandoffMetricsError("stuck projection drift")
        admissible = bool(not physics_contact and solver_finite and successor and not stuck)
        if row["oracle_admissible"] is not admissible:
            raise PhysicalGraphEdgeHandoffMetricsError("oracle admissibility drift")
        tracking_rows = _validate_command_tracking_rows(
            row["command_tracking_rows"], "command_tracking_rows"
        )
        for tick_index, tracking in enumerate(tracking_rows):
            if (
                tracking["requested_command"] != row["requested_commands"][tick_index]
                or tracking["post_slew_command"]
                != row["post_slew_applied_commands"][tick_index]
            ):
                raise PhysicalGraphEdgeHandoffMetricsError(
                    "command-tracking row/branch command drift"
                )
        result.append(row)
    return result


def _fanout_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(dict(row))
    for state_id, values in grouped.items():
        values.sort(key=lambda item: item["candidate_index"])
        if [row["candidate_index"] for row in values] != list(range(C.CANDIDATE_COUNT)):
            raise PhysicalGraphEdgeHandoffMetricsError(f"candidate coverage drift for {state_id}")
    return grouped


def _selection_projection(
    row: Mapping[str, Any], fanout: Sequence[Mapping[str, Any]], *,
    allow_scores: bool = True,
) -> dict[str, Any]:
    candidate_ids = list(C.CANDIDATE_IDS)
    if row["candidate_ids"] != candidate_ids:
        raise PhysicalGraphEdgeHandoffMetricsError("candidate order drift")
    scores = row["scores"]
    if not allow_scores:
        ranking = list(row["ranking"])
    else:
        scores = _sequence(scores, "scores")
        if len(scores) != C.CANDIDATE_COUNT:
            raise PhysicalGraphEdgeHandoffMetricsError("score count drift")
        ranking = _rank(scores)
        if row["ranking"] != ranking:
            raise PhysicalGraphEdgeHandoffMetricsError("score ranking drift")
    if sorted(ranking) != list(range(C.CANDIDATE_COUNT)):
        raise PhysicalGraphEdgeHandoffMetricsError("ranking is not a permutation")
    selected = row["selected_candidate_index"]
    if selected != ranking[0]:
        raise PhysicalGraphEdgeHandoffMetricsError("selected candidate/ranking drift")
    correct_indices = [item["candidate_index"] for item in fanout if _correct_candidate(item)]
    if row["eligible_correct_edge_candidate_indices"] != correct_indices:
        raise PhysicalGraphEdgeHandoffMetricsError("correct-edge candidate projection drift")
    top1 = selected in correct_indices
    top3 = any(index in correct_indices for index in ranking[:3])
    first_rank = next((rank for rank, index in enumerate(ranking, 1) if index in correct_indices), None)
    mrr = 0.0 if first_rank is None else 1.0 / first_rank
    incorrect_indices = [index for index in range(C.CANDIDATE_COUNT) if index not in correct_indices]
    pair_credits: list[float] = []
    rank_position = {candidate_index: rank for rank, candidate_index in enumerate(ranking)}
    for correct_index in correct_indices:
        for incorrect_index in incorrect_indices:
            if allow_scores:
                left = float(scores[correct_index]); right = float(scores[incorrect_index])
                pair_credits.append(1.0 if left > right else 0.5 if left == right else 0.0)
            else:
                pair_credits.append(float(rank_position[correct_index] < rank_position[incorrect_index]))
    pairwise = _mean(pair_credits) if pair_credits else 0.0
    selected_branch = fanout[selected]
    admissible_progress = [item["port_progress_m"] for item in fanout if item["oracle_admissible"]]
    if not admissible_progress:
        raise PhysicalGraphEdgeHandoffMetricsError("state has no oracle-admissible candidate")
    best, minimum = max(admissible_progress), min(admissible_progress)
    regret = _normalized_regret(best, selected_branch["port_progress_m"], minimum)
    expected = {
        "correct_edge_top1": top1,
        "correct_edge_top3": top3,
        "correct_edge_mrr": mrr,
        "selected_correct_edge_execution": selected_branch["entered_correct_edge"],
        "selected_port_progress_m": selected_branch["port_progress_m"],
        "oracle_best_port_progress_m": best,
        "minimum_admissible_port_progress_m": minimum,
        "normalized_port_regret": regret,
        "pairwise_correct_edge_ordering": pairwise,
        "selected_wrong_edge": selected_branch["entered_wrong_edge"],
        "selected_no_edge": selected_branch["no_edge"],
        "selected_lateral_error_m": selected_branch["lateral_error_m"],
        "selected_angular_error_rad": selected_branch["angular_error_rad"],
        "selected_contact": selected_branch["physics_contact"],
        "selected_stuck": selected_branch["stuck"],
        "selected_successor_viable": selected_branch["successor_viable"],
    }
    for field, expected_value in expected.items():
        observed = row[field]
        if isinstance(expected_value, bool):
            if observed is not expected_value:
                raise PhysicalGraphEdgeHandoffMetricsError(f"{field} drift")
        elif not _close(_finite(observed, field), expected_value, 1.0e-12):
            raise PhysicalGraphEdgeHandoffMetricsError(f"{field} drift")
    return expected


def _target_summary(rows: Sequence[Mapping[str, Any]], target_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target_id"] == target_id]
    if len(selected) != C.ROLE_COUNTS["DEVELOPMENT"]:
        raise PhysicalGraphEdgeHandoffMetricsError("development target row coverage drift")
    result = {
        "target_id": target_id,
        "state_count": len(selected),
        "selected_correct_edge_execution_rate": _mean([float(row["selected_correct_edge_execution"]) for row in selected]),
        "correct_edge_top3_rate": _mean([float(row["correct_edge_top3"]) for row in selected]),
        "correct_edge_top1_rate": _mean([float(row["correct_edge_top1"]) for row in selected]),
        "correct_edge_mrr": _mean([float(row["correct_edge_mrr"]) for row in selected]),
        "normalized_port_regret": _mean([float(row["normalized_port_regret"]) for row in selected]),
        "mean_selected_port_progress_m": _mean([float(row["selected_port_progress_m"]) for row in selected]),
        "mean_target_transform_error_m": _mean([float(row["target_transform_error_m"]) for row in selected]),
        "pairwise_correct_edge_ordering": _mean([float(row["pairwise_correct_edge_ordering"]) for row in selected]),
        "selected_wrong_edge_rate": _mean([float(row["selected_wrong_edge"]) for row in selected]),
        "selected_no_edge_rate": _mean([float(row["selected_no_edge"]) for row in selected]),
        "mean_selected_lateral_error_m": _mean([float(row["selected_lateral_error_m"]) for row in selected]),
        "mean_selected_angular_error_rad": _mean([float(row["selected_angular_error_rad"]) for row in selected]),
        "selected_contact_rate": _mean([float(row["selected_contact"]) for row in selected]),
        "selected_stuck_rate": _mean([float(row["selected_stuck"]) for row in selected]),
        "selected_successor_viable_rate": _mean([float(row["selected_successor_viable"]) for row in selected]),
    }
    result["selection_key"] = [
        -result["selected_correct_edge_execution_rate"],
        -result["correct_edge_top3_rate"],
        result["normalized_port_regret"],
        -result["mean_selected_port_progress_m"],
        result["mean_target_transform_error_m"],
        C.TARGET_IDS.index(target_id),
    ]
    return result


def validate_development_target_selection(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    waypoint_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    root = _validate_document(
        value, DEVELOPMENT_SELECTION_FIELDS,
        "physical_graph_edge_handoff_qualification_v1.development_target_selection.v1",
        "development_target_selection",
    )
    if root["role"] != "DEVELOPMENT" or root["target_ids"] != list(C.TARGET_IDS) or root["selection_lexicographic"] != list(C.DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC):
        raise PhysicalGraphEdgeHandoffMetricsError("development selection authority drift")
    if root["heldout_outcome_documents_opened"] != 0 or root["selection_frozen"] is not True:
        raise PhysicalGraphEdgeHandoffMetricsError("heldout outcome leaked into target selection")
    root["ranker_runtime_environment"] = validate_visual_runtime_environment(
        root["ranker_runtime_environment"], runtime_role="ranker"
    )
    development_ids = [row["state_id"] for row in panel["states"] if row["role"] == "DEVELOPMENT"]
    fanout = _fanout_groups(candidate_fanout)
    waypoint_rows = {
        (row["state_id"], row["target_id"]): row
        for row in waypoint_contracts["rows"]
    }
    rows = _sequence(root["state_target_rows"], "state_target_rows")
    expected_count = C.ROLE_COUNTS["DEVELOPMENT"] * len(C.TARGET_IDS)
    if len(rows) != expected_count:
        raise PhysicalGraphEdgeHandoffMetricsError("development target row count drift")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        state_id = development_ids[index // len(C.TARGET_IDS)]
        target_id = C.TARGET_IDS[index % len(C.TARGET_IDS)]
        row = _mapping(raw, STATE_TARGET_ROW_FIELDS, f"development row[{index}]")
        if row["state_id"] != state_id or row["target_id"] != target_id:
            raise PhysicalGraphEdgeHandoffMetricsError("development target identity/order drift")
        _selection_projection(row, fanout[state_id])
        waypoint = waypoint_rows[(state_id, target_id)]
        expected_error = max(
            waypoint["position_transform_error_m"],
            waypoint["heading_transform_error_rad"],
        )
        if not _close(_finite(row["target_transform_error_m"], "target transform error"), expected_error, 0.0):
            raise PhysicalGraphEdgeHandoffMetricsError("development target transform error drift")
        validated.append(row)
    summaries = _sequence(root["target_summaries"], "target_summaries")
    expected_summaries = [_target_summary(validated, target_id) for target_id in C.TARGET_IDS]
    if len(summaries) != len(C.TARGET_IDS):
        raise PhysicalGraphEdgeHandoffMetricsError("target summary count drift")
    for index, (raw, expected) in enumerate(zip(summaries, expected_summaries)):
        observed = _mapping(raw, TARGET_SUMMARY_FIELDS, f"target_summary[{index}]")
        if observed.keys() != expected.keys():
            raise PhysicalGraphEdgeHandoffMetricsError("target summary schema drift")
        for field, expected_value in expected.items():
            if isinstance(expected_value, (str, int, list)):
                if observed[field] != expected_value:
                    raise PhysicalGraphEdgeHandoffMetricsError(f"target summary {field} drift")
            elif not _close(_finite(observed[field], field), expected_value, 1.0e-12):
                raise PhysicalGraphEdgeHandoffMetricsError(f"target summary {field} drift")
    winner = min(range(len(expected_summaries)), key=lambda index: tuple(expected_summaries[index]["selection_key"]))
    if root["selected_target_index"] != winner or root["selected_target_id"] != C.TARGET_IDS[winner]:
        raise PhysicalGraphEdgeHandoffMetricsError("development target lexicographic winner drift")
    root["state_target_rows"] = validated
    return root


def _kinematic_scores(
    fanout: Sequence[Mapping[str, Any]], target_body: Sequence[float],
) -> list[float]:
    scores: list[float] = []
    start_distance = math.hypot(target_body[0], target_body[1])
    for branch in fanout:
        x = y = yaw = 0.0
        for command in branch["post_slew_applied_commands"]:
            vx, vy, yaw_rate = command
            dt = 0.10
            x += dt * (vx * math.cos(yaw) - vy * math.sin(yaw))
            y += dt * (vx * math.sin(yaw) + vy * math.cos(yaw))
            yaw = _angle(yaw + dt * yaw_rate)
        scores.append(start_distance - math.hypot(target_body[0] - x, target_body[1] - y))
    return scores


def _oracle_ranking(fanout: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(
        range(C.CANDIDATE_COUNT),
        key=lambda index: (
            0 if _correct_candidate(fanout[index]) else 1,
            -float(fanout[index]["port_progress_m"]),
            float(fanout[index]["lateral_error_m"]),
            float(fanout[index]["angular_error_rad"]),
            index,
        ),
    )


def validate_heldout_ranker_score_rows(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    development_target_selection: Mapping[str, Any],
    waypoint_contracts: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    rows = _sequence(value, "heldout_ranker_scores")
    if len(rows) != C.HELDOUT_SCORE_ROW_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("heldout score row count drift")
    heldout_states = [row for row in panel["states"] if row["role"] == "DEVELOPMENT_HELDOUT"]
    fanout = _fanout_groups(candidate_fanout)
    selected_target = development_target_selection["selected_target_id"]
    ranker_runtime_digest = runtime_environment_sha256(
        development_target_selection["ranker_runtime_environment"]
    )
    waypoint_by_state = {
        row["state_id"]: row
        for row in waypoint_contracts["rows"]
        if row["target_id"] == selected_target
    }
    teacher_by_state = {
        row["state_id"]: row
        for row in teacher_trace_index["records"]
        if row["selected"]
    }
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        state = heldout_states[index // len(C.HELDOUT_CONDITION_IDS)]
        condition = C.HELDOUT_CONDITION_IDS[index % len(C.HELDOUT_CONDITION_IDS)]
        row = _mapping(raw, HELDOUT_SCORE_FIELDS, f"heldout[{index}]")
        if row["ranker_runtime_environment_sha256"] != ranker_runtime_digest:
            raise PhysicalGraphEdgeHandoffMetricsError(
                "heldout ranker runtime cross-binding drift"
            )
        if row["state_id"] != state["state_id"] or row["family"] != state["family"] or row["condition_id"] != condition or row["target_id"] != selected_target:
            raise PhysicalGraphEdgeHandoffMetricsError("heldout identity/order drift")
        if row["score_row_id"] != f"{state['state_id']}::{condition}":
            raise PhysicalGraphEdgeHandoffMetricsError("heldout score row identity drift")
        branches = fanout[state["state_id"]]
        if condition == "TEACHER_TRACE":
            teacher = teacher_by_state[state["state_id"]]
            expected_null = (
                "scores", "ranking", "selected_candidate_index", "correct_edge_top1",
                "correct_edge_top3", "correct_edge_mrr",
                "oracle_best_port_progress_m", "minimum_admissible_port_progress_m",
                "normalized_port_regret", "ranker_checkpoint_sha256",
                "pairwise_correct_edge_ordering",
            )
            if row["candidate_ids"] != [] or row["eligible_correct_edge_candidate_indices"] != [] or any(row[field] is not None for field in expected_null):
                raise PhysicalGraphEdgeHandoffMetricsError("teacher condition fabricates candidate ranking")
            if row["selected_correct_edge_execution"] is not None:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher condition fabricates selected candidate execution")
            if row["teacher_trace_id"] != teacher["teacher_trace_id"] or row["teacher_correct_execution"] is not teacher["teacher_valid"]:
                raise PhysicalGraphEdgeHandoffMetricsError("teacher heldout binding drift")
            teacher_physical = {
                "selected_port_progress_m": teacher["route_progress_m"],
                "selected_wrong_edge": teacher["competing_port_entered"],
                "selected_no_edge": not teacher["crossed_directed_port"] and not teacher["competing_port_entered"],
                "selected_lateral_error_m": teacher["endpoint_lateral_error_m"],
                "selected_angular_error_rad": teacher["endpoint_angular_error_rad"],
                "selected_contact": not teacher["contact_free"],
                "selected_stuck": teacher["stuck"],
                "selected_successor_viable": teacher["successor_viable"],
            }
            for field, expected in teacher_physical.items():
                if isinstance(expected, bool):
                    if row[field] is not expected:
                        raise PhysicalGraphEdgeHandoffMetricsError(f"teacher {field} drift")
                elif not _close(_finite(row[field], field), expected, C.NUMERICAL_TOLERANCES["trace_summary_float"]):
                    raise PhysicalGraphEdgeHandoffMetricsError(f"teacher {field} drift")
            result.append(row)
            continue
        if row["teacher_trace_id"] is not None or row["teacher_correct_execution"] is not None:
            raise PhysicalGraphEdgeHandoffMetricsError("candidate condition has teacher outcome")
        if condition == "DETERMINISTIC_KINEMATICS":
            expected_scores = _kinematic_scores(branches, waypoint_by_state[state["state_id"]]["target_body_pose"])
            observed_scores = _sequence(row["scores"], "kinematic scores")
            if len(observed_scores) != C.CANDIDATE_COUNT or any(
                not _close(_finite(observed, "kinematic score"), expected, 1.0e-12)
                for observed, expected in zip(observed_scores, expected_scores)
            ):
                raise PhysicalGraphEdgeHandoffMetricsError("deterministic kinematics score drift")
            if row["ranker_checkpoint_sha256"] is not None:
                raise PhysicalGraphEdgeHandoffMetricsError("kinematic row claims ranker checkpoint")
            _selection_projection(row, branches)
        elif condition == "FROZEN_CURRENT_VISUAL_RANKER":
            if row["ranker_checkpoint_sha256"] != C.CURRENT_VISUAL_RANKER_BINDING["checkpoint_sha256"]:
                raise PhysicalGraphEdgeHandoffMetricsError("ranker checkpoint drift")
            _selection_projection(row, branches)
        else:
            if row["scores"] is not None or row["ranker_checkpoint_sha256"] is not None:
                raise PhysicalGraphEdgeHandoffMetricsError("oracle row has learned/dummy scores")
            ranking = _oracle_ranking(branches)
            if row["ranking"] != ranking:
                raise PhysicalGraphEdgeHandoffMetricsError("oracle lexicographic ranking drift")
            _selection_projection(row, branches, allow_scores=False)
        result.append(row)
    return result


def validate_repeated_execution_rows(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    heldout_ranker_scores: Sequence[Mapping[str, Any]],
    state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    rows = _sequence(value, "repeated_execution")
    if len(rows) != C.REPEATED_EXECUTION_ROW_COUNT:
        raise PhysicalGraphEdgeHandoffMetricsError("repeat row count drift")
    heldout_states = [row for row in panel["states"] if row["role"] == "DEVELOPMENT_HELDOUT"]
    fanout = _fanout_groups(candidate_fanout)
    score_by_identity = {
        (row["state_id"], row["condition_id"]): row
        for row in heldout_ranker_scores
    }
    snapshot_by_state = {
        row["state_id"]: row for row in state_snapshot_index["records"]
    }
    physical_runtime_digest = panel["physical_runtime_environment"][
        "runtime_core_sha256"
    ]
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        state = heldout_states[index // (len(C.REPEAT_BRANCH_IDS) * C.REPEATS_PER_BRANCH)]
        within = index % (len(C.REPEAT_BRANCH_IDS) * C.REPEATS_PER_BRANCH)
        selector = C.REPEAT_BRANCH_IDS[within // C.REPEATS_PER_BRANCH]
        repeat_index = within % C.REPEATS_PER_BRANCH
        row = _mapping(raw, REPEATED_EXECUTION_FIELDS, f"repeat[{index}]")
        if row["physical_runtime_core_sha256"] != physical_runtime_digest:
            raise PhysicalGraphEdgeHandoffMetricsError(
                "repeat physical runtime cross-binding drift"
            )
        if row["state_id"] != state["state_id"] or row["family"] != state["family"] or row["branch_selector_id"] != selector or row["repeat_index"] != repeat_index:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat identity/order drift")
        if row["repeat_id"] != f"{state['state_id']}::{selector}::{repeat_index}":
            raise PhysicalGraphEdgeHandoffMetricsError("repeat_id drift")
        condition = (
            "FROZEN_CURRENT_VISUAL_RANKER"
            if selector == "FROZEN_CURRENT_VISUAL_RANKER_SELECTED"
            else "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
        )
        selected_index = score_by_identity[(state["state_id"], condition)]["selected_candidate_index"]
        source = fanout[state["state_id"]][selected_index]
        if row["source_candidate_index"] != selected_index or row["source_branch_id"] != source["branch_id"]:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat source branch drift")
        snapshot = snapshot_by_state[state["state_id"]]
        if row["snapshot_id"] != snapshot["snapshot_id"] or row["restored_snapshot_sha256"] != snapshot["snapshot_payload"]["slice_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat snapshot binding drift")
        trace_index = C.TRACE_INDEX_RANGES["repeated_execution"][0] + index
        if row["trace_index"] != trace_index:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat trace index drift")
        trace_slice = _validate_slice(row["trace_slice"], "repeat trace slice", "candidate_traces.npz")
        if trace_slice["member"] != "timestamp_s" or trace_slice["stop"] - trace_slice["start"] != C.PHYSICS_STEPS_PER_BRANCH:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat trace duration drift")
        digests = row["trace_array_slice_sha256s"]
        expected_members = set(C.NPZ_PAYLOAD_AUTHORITY["candidate_traces.npz"]) - {"trace_offsets"}
        if not isinstance(digests, Mapping) or set(digests) != expected_members:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat trace digest coverage drift")
        for digest in digests.values(): _sha(digest, "repeat trace digest")
        source_endpoint = _vector(row["source_endpoint_body"], 3, "source endpoint")
        repeat_endpoint = _vector(row["repeat_endpoint_body"], 3, "repeat endpoint")
        position_error = math.hypot(source_endpoint[0] - repeat_endpoint[0], source_endpoint[1] - repeat_endpoint[1])
        heading_error = abs(_angle(source_endpoint[2] - repeat_endpoint[2]))
        if not _close(row["endpoint_position_error_m"], position_error, 1.0e-12) or not _close(row["endpoint_heading_error_rad"], heading_error, 1.0e-12):
            raise PhysicalGraphEdgeHandoffMetricsError("repeat endpoint error drift")
        expected_success = bool(
            row["source_applied_command_sequence_sha256"] == row["repeat_applied_command_sequence_sha256"]
            and row["source_correct_edge_execution"] is row["repeat_correct_edge_execution"]
            and row["source_endpoint_edge_id"] == row["repeat_endpoint_edge_id"]
            and row["source_physics_contact"] is row["physics_contact"]
            and row["source_stuck"] is row["stuck"]
            and position_error <= C.NUMERICAL_TOLERANCES["repeat_endpoint_position_m"]
            and heading_error <= C.NUMERICAL_TOLERANCES["repeat_endpoint_heading_rad"]
        )
        source_applied_sha = source["trace_array_slice_sha256s"][
            "post_slew_applied_command"
        ]
        repeat_applied_sha = row["trace_array_slice_sha256s"][
            "post_slew_applied_command"
        ]
        if (
            row["source_applied_command_sequence_sha256"] != source_applied_sha
            or row["repeat_applied_command_sequence_sha256"] != repeat_applied_sha
        ):
            raise PhysicalGraphEdgeHandoffMetricsError(
                "repeat applied-command trace binding drift"
            )
        if row["repeat_success"] is not expected_success:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat success drift")
        if row["source_correct_edge_execution"] is not source["entered_correct_edge"] or row["source_endpoint_body"] != source["h3_endpoint_body"] or row["source_endpoint_edge_id"] != source["endpoint_edge_id"] or row["source_physics_contact"] is not source["physics_contact"] or row["source_stuck"] is not source["stuck"]:
            raise PhysicalGraphEdgeHandoffMetricsError("repeat source outcome binding drift")
        result.append(row)
    return result


def _materially_outperforms(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any],
) -> bool:
    return bool(
        candidate["selected_correct_edge_execution_rate"]
        - baseline["selected_correct_edge_execution_rate"]
        >= C.TARGET_MATERIALITY["selected_correct_edge_execution_rate_improvement"]
        or candidate["correct_edge_top3_rate"] - baseline["correct_edge_top3_rate"]
        >= C.TARGET_MATERIALITY["correct_edge_top3_rate_improvement"]
        or baseline["normalized_port_regret"] - candidate["normalized_port_regret"]
        >= C.TARGET_MATERIALITY["normalized_port_regret_reduction"]
        or candidate["mean_selected_port_progress_m"]
        - baseline["mean_selected_port_progress_m"]
        >= C.TARGET_MATERIALITY["selected_port_progress_m_improvement"]
    )


def classify_physical_handoff_aggregates(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen primary-classification precedence to exact aggregates."""

    row = _mapping(value, CLASSIFICATION_INPUT_FIELDS, "classification input")
    teacher_count = _integer(row["teacher_correct_execution_count"], "teacher count")
    coverage = _finite(row["coverage_rate"], "coverage")
    top1 = _finite(row["ranker_correct_edge_top1_rate"], "ranker top1")
    top3 = _finite(row["ranker_correct_edge_top3_rate"], "ranker top3")
    execution = _finite(row["ranker_selected_correct_edge_execution_rate"], "ranker execution")
    regret = _finite(row["ranker_normalized_port_regret"], "ranker regret")
    oracle_execution = _finite(row["oracle_selected_correct_edge_execution_rate"], "oracle execution")
    oracle_covered_execution = _finite(
        row["oracle_covered_state_correct_execution_rate"],
        "oracle covered-state execution",
    )
    repeatability = _finite(row["repeatability_rate"], "repeatability")
    family_minimum = _integer(row["minimum_family_correct_execution_count"], "family minimum")
    tracking = _boolean(row["command_tracking_pass"], "command tracking pass")
    selected_target = row["selected_target_id"]
    if selected_target not in C.TARGET_IDS:
        raise PhysicalGraphEdgeHandoffMetricsError("selected target identity drift")
    coverage_failure = coverage < C.HANDOFF_GATE["coverage_rate_minimum"]
    repeat_pass = repeatability >= C.HANDOFF_GATE["repeatability_rate_minimum"]
    oracle_succeeds = oracle_covered_execution >= 1.0
    low_level_failure = bool(
        coverage > 0.0 and teacher_count == C.HANDOFF_GATE["teacher_correct_execution_count"]
        and (not oracle_succeeds or not repeat_pass or not tracking)
    )
    ranker_gate_pass = bool(
        top1 >= C.HANDOFF_GATE["ranker_correct_edge_top1_rate_minimum"]
        and top3 >= C.HANDOFF_GATE["ranker_correct_edge_top3_rate_minimum"]
        and execution >= C.HANDOFF_GATE["ranker_selected_correct_edge_execution_rate_minimum"]
        and regret <= C.HANDOFF_GATE["ranker_normalized_port_regret_maximum"]
        and family_minimum >= C.HANDOFF_GATE["minimum_correct_execution_per_family"]
    )
    ranker_failure = bool(
        not coverage_failure and oracle_succeeds and repeat_pass and tracking
        and not ranker_gate_pass
    )
    full_handoff_gate = bool(
        teacher_count == C.HANDOFF_GATE["teacher_correct_execution_count"]
        and not coverage_failure and oracle_succeeds and ranker_gate_pass
        and repeat_pass and tracking
    )
    if row["selected_target_passes_handoff_gate"] is not full_handoff_gate:
        raise PhysicalGraphEdgeHandoffMetricsError("selected target full-handoff gate projection drift")
    target_failure = bool(
        selected_target != "TARGET_NODE_CENTRE" and full_handoff_gate
        and row["selected_target_materially_outperforms_node_centre"] is True
    )
    components = {
        "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT": target_failure,
        "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO": coverage_failure,
        "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO": ranker_failure,
        "LOW_LEVEL_PREFIX_EXECUTION_NO_GO": low_level_failure,
    }
    active = [name for name in C.COMPONENT_PRECEDENCE if components[name]]
    signal_gate = full_handoff_gate
    if len(active) > 1:
        primary = "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO"
    elif target_failure:
        primary = "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT"
    elif signal_gate:
        primary = "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL"
    elif coverage_failure:
        primary = "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO"
    elif ranker_failure:
        primary = "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO"
    elif low_level_failure:
        primary = "LOW_LEVEL_PREFIX_EXECUTION_NO_GO"
    else:
        raise PhysicalGraphEdgeHandoffMetricsError("classification tree has no valid disposition")
    next_experiment = (
        C.COMPOSITE_NEXT_BY_EARLIEST_COMPONENT[active[0]]
        if primary == "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO"
        else C.NEXT_DECISION_BY_CLASSIFICATION[primary]
    )
    return {
        "handoff_gate_passed": signal_gate,
        "component_failures": components,
        "active_components_in_precedence_order": active,
        "earliest_failing_component": active[0] if active else None,
        "primary_classification": primary,
        "next_experiment": next_experiment,
    }


def _condition_summary(
    rows: Sequence[Mapping[str, Any]], condition_id: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["condition_id"] == condition_id]
    if not selected:
        raise PhysicalGraphEdgeHandoffMetricsError("heldout condition coverage drift")
    if condition_id == "TEACHER_TRACE":
        return {
            "condition_id": condition_id,
            "state_count": len(selected),
            "teacher_correct_execution_count": sum(bool(row["teacher_correct_execution"]) for row in selected),
            "teacher_correct_execution_rate": _mean([float(row["teacher_correct_execution"]) for row in selected]),
            "candidate_ranking_metrics_applicable": False,
            "mean_port_progress_m": _mean([float(row["selected_port_progress_m"]) for row in selected]),
            "wrong_edge_rate": _mean([float(row["selected_wrong_edge"]) for row in selected]),
            "no_edge_rate": _mean([float(row["selected_no_edge"]) for row in selected]),
            "mean_lateral_error_m": _mean([float(row["selected_lateral_error_m"]) for row in selected]),
            "mean_angular_error_rad": _mean([float(row["selected_angular_error_rad"]) for row in selected]),
            "contact_rate": _mean([float(row["selected_contact"]) for row in selected]),
            "stuck_rate": _mean([float(row["selected_stuck"]) for row in selected]),
            "successor_viable_rate": _mean([float(row["selected_successor_viable"]) for row in selected]),
        }
    return {
        "condition_id": condition_id,
        "state_count": len(selected),
        "correct_edge_top1_rate": _mean([float(row["correct_edge_top1"]) for row in selected]),
        "correct_edge_top3_rate": _mean([float(row["correct_edge_top3"]) for row in selected]),
        "correct_edge_mrr": _mean([float(row["correct_edge_mrr"]) for row in selected]),
        "selected_correct_edge_execution_rate": _mean([float(row["selected_correct_edge_execution"]) for row in selected]),
        "mean_selected_port_progress_m": _mean([float(row["selected_port_progress_m"]) for row in selected]),
        "normalized_port_regret": _mean([float(row["normalized_port_regret"]) for row in selected]),
        "pairwise_correct_edge_ordering": _mean([float(row["pairwise_correct_edge_ordering"]) for row in selected]),
        "wrong_edge_rate": _mean([float(row["selected_wrong_edge"]) for row in selected]),
        "no_edge_rate": _mean([float(row["selected_no_edge"]) for row in selected]),
        "mean_lateral_error_m": _mean([float(row["selected_lateral_error_m"]) for row in selected]),
        "mean_angular_error_rad": _mean([float(row["selected_angular_error_rad"]) for row in selected]),
        "contact_rate": _mean([float(row["selected_contact"]) for row in selected]),
        "stuck_rate": _mean([float(row["selected_stuck"]) for row in selected]),
        "successor_viable_rate": _mean([float(row["selected_successor_viable"]) for row in selected]),
        "candidate_ranking_metrics_applicable": True,
    }


def _command_tracking_summary(
    heldout_rows: Sequence[Mapping[str, Any]],
    fanout_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    ticks: list[Mapping[str, Any]] = []
    for row in heldout_rows:
        if row["condition_id"] not in {
            "FROZEN_CURRENT_VISUAL_RANKER", "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
        }:
            continue
        ticks.extend(
            fanout_by_state[row["state_id"]][row["selected_candidate_index"]]["command_tracking_rows"]
        )
    vx_errors: list[float] = []; yaw_errors: list[float] = []
    vy_values: list[float] = []; sign_matches: list[float] = []
    for tick in ticks:
        command = tick["post_slew_command"]
        achieved = tick["mean_achieved_body_velocity"]
        vy_values.append(abs(achieved[1]))
        if tick["active_vx"]:
            vx_errors.append(abs(achieved[0] - command[0]))
            sign_matches.append(float(achieved[0] * command[0] > 0.0))
        if tick["active_yaw"]:
            yaw_errors.append(abs(achieved[2] - command[2]))
            sign_matches.append(float(achieved[2] * command[2] > 0.0))
    vx_mae = _mean(vx_errors) if vx_errors else 0.0
    vy_mean = _mean(vy_values)
    yaw_mae = _mean(yaw_errors) if yaw_errors else 0.0
    sign_rate = _mean(sign_matches) if sign_matches else 1.0
    passed = bool(
        vx_mae <= C.COMMAND_TRACKING_AUTHORITY["vx_mae_maximum_mps"]
        and vy_mean <= C.COMMAND_TRACKING_AUTHORITY["vy_absolute_mean_maximum_mps"]
        and yaw_mae <= C.COMMAND_TRACKING_AUTHORITY["yaw_rate_mae_maximum_rad_s"]
        and sign_rate >= C.COMMAND_TRACKING_AUTHORITY["commanded_sign_agreement_minimum"]
    )
    return {
        "selected_branch_count": len(ticks) // C.HORIZON_TICKS[C.PRIMARY_HORIZON],
        "command_tick_count": len(ticks),
        "active_vx_component_count": len(vx_errors),
        "active_yaw_component_count": len(yaw_errors),
        "vx_mae_mps": vx_mae,
        "vy_absolute_mean_mps": vy_mean,
        "yaw_rate_mae_rad_s": yaw_mae,
        "commanded_sign_agreement_rate": sign_rate,
        "passed": passed,
    }


def _cross_validate_npz_bindings(
    inspections: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any], teachers: Mapping[str, Any],
    pixels: Mapping[str, Any], latents: Mapping[str, Any],
    fanout: Sequence[Mapping[str, Any]], repeats: Sequence[Mapping[str, Any]],
) -> None:
    roots = (
        (snapshots["snapshots_file"], "state_snapshots.npz"),
        (teachers["traces_file"], "teacher_traces.npz"),
        (pixels["rgb_file"], "rgb_observations.npz"),
        (latents["latents_file"], "canonical_latents.npz"),
    )
    for binding, path in roots:
        if binding["sha256"] != inspections[path]["sha256"] or binding["bytes"] != inspections[path]["bytes"]:
            raise PhysicalGraphEdgeHandoffMetricsError(f"{path} root binding drift")
    snapshot_inspection = inspections["state_snapshots.npz"]["members"]
    for index, row in enumerate(snapshots["records"]):
        if row["snapshot_payload"]["slice_sha256"] != snapshot_inspection["snapshot_payload_bytes"]["row_or_slice_sha256s"][index]:
            raise PhysicalGraphEdgeHandoffMetricsError("snapshot payload slice digest drift")
        digest_fields = {
            "base_pose_world": "base_pose_world_sha256",
            "base_twist_world": "base_twist_world_sha256",
            "joint_position": "joint_position_sha256",
            "joint_velocity": "joint_velocity_sha256",
            "camera_world_transform": "camera_world_transform_sha256",
            "controller_observation": "controller_observation_sha256",
            "policy_last_action": "policy_last_action_sha256",
            "previous_policy_action": "previous_policy_action_sha256",
            "previous_applied_command": "previous_applied_command_sha256",
            "command_history": "command_history_sha256",
            "control_history": "control_history_sha256",
            "low_level_policy_state": "low_level_policy_state_sha256",
        }
        for member, field in digest_fields.items():
            if row[field] != snapshot_inspection[member]["row_or_slice_sha256s"][index]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"snapshot {member} row digest drift")
    teacher_inspection = inspections["teacher_traces.npz"]["members"]
    teacher_members = set(C.NPZ_PAYLOAD_AUTHORITY["teacher_traces.npz"]) - {"trace_offsets"}
    for index, row in enumerate(teachers["records"]):
        if row["trace_index"] != index:
            raise PhysicalGraphEdgeHandoffMetricsError("teacher trace index order drift")
        for member in teacher_members:
            if row["trace_array_slice_sha256s"][member] != teacher_inspection[member]["row_or_slice_sha256s"][index]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"teacher {member} slice digest drift")
    rgb_digests = inspections["rgb_observations.npz"]["members"]["rgb"]["row_or_slice_sha256s"]
    selected_teacher_by_state = {
        row["state_id"]: row for row in teachers["records"] if row["selected"]
    }
    snapshot_by_state = {row["state_id"]: row for row in snapshots["records"]}
    for index, row in enumerate(pixels["records"]):
        if row["row_sha256"] != rgb_digests[index]:
            raise PhysicalGraphEdgeHandoffMetricsError("RGB row digest drift")
        if selected_teacher_by_state[row["state_id"]]["current_rgb_sha256"] != row["row_sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("selected teacher/RGB capture drift")
        if any(
            trial["current_rgb_sha256"] != row["row_sha256"]
            for trial in snapshot_by_state[row["state_id"]]["reset_trials"]
        ):
            raise PhysicalGraphEdgeHandoffMetricsError("reset fixture/RGB capture drift")
    latent_members = inspections["canonical_latents.npz"]["members"]
    for index, row in enumerate(latents["records"]):
        if row["raw_token_sha256"] != latent_members["raw_tokens"]["row_or_slice_sha256s"][index] or row["spatial_descriptor_sha256"] != latent_members["spatial_descriptors"]["row_or_slice_sha256s"][index]:
            raise PhysicalGraphEdgeHandoffMetricsError("latent row digest drift")
    candidate_inspection = inspections["candidate_traces.npz"]
    candidate_members = candidate_inspection["members"]
    expected_candidate_members = set(C.NPZ_PAYLOAD_AUTHORITY["candidate_traces.npz"]) - {"trace_offsets"}
    for snapshot in snapshots["records"]:
        for trial in snapshot["reset_trials"]:
            trace_index = trial["trace_index"]
            for member in expected_candidate_members:
                if trial["trace_array_slice_sha256s"][member] != candidate_members[member]["row_or_slice_sha256s"][trace_index]:
                    raise PhysicalGraphEdgeHandoffMetricsError(f"reset {member} trace digest drift")
    for row in [*fanout, *repeats]:
        if row["trace_slice"]["file_sha256"] != candidate_inspection["sha256"]:
            raise PhysicalGraphEdgeHandoffMetricsError("candidate trace file binding drift")
        trace_index = row["trace_index"]
        for member in expected_candidate_members:
            if row["trace_array_slice_sha256s"][member] != candidate_members[member]["row_or_slice_sha256s"][trace_index]:
                raise PhysicalGraphEdgeHandoffMetricsError(f"candidate {member} trace digest drift")


def recompute_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly regenerate all registered metrics and the sole primary class."""

    row = _mapping(evidence, EVIDENCE_KEYS, "evidence")
    inspections = validate_npz_inspections(row["npz_inspections"])
    panel = validate_panel_manifest(row["panel_manifest"])
    split = validate_split_manifest(row["split_manifest"], panel)
    graph = validate_graph_manifest(row["graph_manifest"], panel)
    snapshots = validate_state_snapshot_index(row["state_snapshot_index"], panel)
    teachers = validate_teacher_trace_index(row["teacher_trace_index"], panel)
    ports = validate_edge_port_index(row["edge_port_index"], panel, graph, teachers)
    waypoints = validate_waypoint_contracts(row["waypoint_contracts"], panel, graph, ports)
    pixels = validate_pixel_index(row["pixel_index"], panel)
    latents = validate_latent_index(row["latent_index"], pixels)
    fanout = validate_candidate_fanout_rows(row["candidate_fanout"], panel, snapshots)
    development = validate_development_target_selection(
        row["development_target_selection"], panel, fanout, waypoints,
    )
    heldout = validate_heldout_ranker_score_rows(
        row["heldout_ranker_scores"], panel, fanout, development, waypoints, teachers,
    )
    repeats = validate_repeated_execution_rows(
        row["repeated_execution"], panel, fanout, heldout, snapshots,
    )
    _cross_validate_npz_bindings(
        inspections, snapshots, teachers, pixels, latents, fanout, repeats,
    )

    target_summaries = {item["target_id"]: item for item in development["target_summaries"]}
    selected_target = development["selected_target_id"]
    selected_summary = target_summaries[selected_target]
    node_summary = target_summaries["TARGET_NODE_CENTRE"]
    development_selected_pass = bool(
        selected_summary["correct_edge_top1_rate"] >= C.HANDOFF_GATE["ranker_correct_edge_top1_rate_minimum"]
        and selected_summary["correct_edge_top3_rate"] >= C.HANDOFF_GATE["ranker_correct_edge_top3_rate_minimum"]
        and selected_summary["selected_correct_edge_execution_rate"] >= C.HANDOFF_GATE["ranker_selected_correct_edge_execution_rate_minimum"]
        and selected_summary["normalized_port_regret"] <= C.HANDOFF_GATE["ranker_normalized_port_regret_maximum"]
    )
    selected_material = _materially_outperforms(selected_summary, node_summary)
    condition_summaries = {
        condition: _condition_summary(heldout, condition)
        for condition in C.HELDOUT_CONDITION_IDS
    }
    fanout_by_state = _fanout_groups(fanout)
    heldout_states = [item for item in panel["states"] if item["role"] == "DEVELOPMENT_HELDOUT"]
    correct_candidate_count_by_state = {
        item["state_id"]: sum(_correct_candidate(branch) for branch in fanout_by_state[item["state_id"]])
        for item in heldout_states
    }
    coverage_by_state = {
        state_id: count >= 1 for state_id, count in correct_candidate_count_by_state.items()
    }
    coverage_rate = _mean([float(value) for value in coverage_by_state.values()])
    two_candidate_coverage_rate = _mean([
        float(value >= 2) for value in correct_candidate_count_by_state.values()
    ])
    ranker_rows = [item for item in heldout if item["condition_id"] == "FROZEN_CURRENT_VISUAL_RANKER"]
    oracle_rows = [item for item in heldout if item["condition_id"] == "ORACLE_BEST_ADMISSIBLE_CANDIDATE"]
    teacher_summary = condition_summaries["TEACHER_TRACE"]
    ranker_summary = condition_summaries["FROZEN_CURRENT_VISUAL_RANKER"]
    oracle_summary = condition_summaries["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]
    covered_oracle_rows = [item for item in oracle_rows if coverage_by_state[item["state_id"]]]
    oracle_covered_rate = (
        _mean([float(item["selected_correct_edge_execution"]) for item in covered_oracle_rows])
        if covered_oracle_rows else 0.0
    )
    family_execution = {
        family: sum(
            bool(item["selected_correct_edge_execution"])
            for item in ranker_rows if item["family"] == family
        )
        for family in C.FAMILY_IDS
    }
    repeatability_rate = _mean([float(item["repeat_success"]) for item in repeats])
    command_tracking = _command_tracking_summary(heldout, fanout_by_state)
    full_heldout_handoff_gate = bool(
        teacher_summary["teacher_correct_execution_count"] == C.HANDOFF_GATE["teacher_correct_execution_count"]
        and coverage_rate >= C.HANDOFF_GATE["coverage_rate_minimum"]
        and oracle_covered_rate >= 1.0
        and ranker_summary["correct_edge_top1_rate"] >= C.HANDOFF_GATE["ranker_correct_edge_top1_rate_minimum"]
        and ranker_summary["correct_edge_top3_rate"] >= C.HANDOFF_GATE["ranker_correct_edge_top3_rate_minimum"]
        and ranker_summary["selected_correct_edge_execution_rate"] >= C.HANDOFF_GATE["ranker_selected_correct_edge_execution_rate_minimum"]
        and ranker_summary["normalized_port_regret"] <= C.HANDOFF_GATE["ranker_normalized_port_regret_maximum"]
        and repeatability_rate >= C.HANDOFF_GATE["repeatability_rate_minimum"]
        and command_tracking["passed"]
        and min(family_execution.values()) >= C.HANDOFF_GATE["minimum_correct_execution_per_family"]
    )
    classification_input = {
        "teacher_correct_execution_count": teacher_summary["teacher_correct_execution_count"],
        "coverage_rate": coverage_rate,
        "ranker_correct_edge_top1_rate": ranker_summary["correct_edge_top1_rate"],
        "ranker_correct_edge_top3_rate": ranker_summary["correct_edge_top3_rate"],
        "ranker_selected_correct_edge_execution_rate": ranker_summary["selected_correct_edge_execution_rate"],
        "ranker_normalized_port_regret": ranker_summary["normalized_port_regret"],
        "oracle_selected_correct_edge_execution_rate": oracle_summary["selected_correct_edge_execution_rate"],
        "oracle_covered_state_correct_execution_rate": oracle_covered_rate,
        "repeatability_rate": repeatability_rate,
        "command_tracking_pass": command_tracking["passed"],
        "minimum_family_correct_execution_count": min(family_execution.values()),
        "selected_target_id": selected_target,
        "selected_target_passes_handoff_gate": full_heldout_handoff_gate,
        "selected_target_materially_outperforms_node_centre": selected_material,
    }
    disposition = classify_physical_handoff_aggregates(classification_input)

    def grouped_rate(attribute: str, value: Sequence[str], *, outcome: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for group in value:
            states = [item for item in heldout_states if item[attribute] == group]
            covered = [coverage_by_state[item["state_id"]] for item in states]
            two_or_more = [correct_candidate_count_by_state[item["state_id"]] >= 2 for item in states]
            rows_for_states = {item["state_id"]: item for item in ranker_rows}
            executed = [bool(rows_for_states[item["state_id"]]["selected_correct_edge_execution"]) for item in states]
            result[group] = {
                "state_count": len(states),
                "coverage_rate": _mean([float(item) for item in covered]) if covered else 0.0,
                "two_or_more_candidate_count": sum(two_or_more),
                "two_or_more_candidate_rate": _mean([float(item) for item in two_or_more]) if two_or_more else 0.0,
                "ranker_execution_rate": _mean([float(item) for item in executed]) if executed else 0.0,
            }
        del outcome
        return result

    direction = grouped_rate("route_direction", ("STRAIGHT", "LEFT", "RIGHT"), outcome="direction")
    distance = grouped_rate("port_distance_id", ("NEAR", "FAR"), outcome="distance")
    family_coverage = grouped_rate("family", C.FAMILY_IDS, outcome="family")
    candidate_identity_contributions = {
        candidate_id: {
            "correct_edge_candidate_count": sum(
                _correct_candidate(fanout_by_state[state["state_id"]][candidate_index])
                for state in heldout_states
            ),
            "heldout_state_rate": _mean([
                float(_correct_candidate(fanout_by_state[state["state_id"]][candidate_index]))
                for state in heldout_states
            ]),
        }
        for candidate_index, candidate_id in enumerate(C.CANDIDATE_IDS)
    }
    per_family_conditions = {
        family: {
            condition: _condition_summary(
                [item for item in heldout if item["family"] == family], condition
            )
            for condition in C.HELDOUT_CONDITION_IDS
        }
        for family in C.FAMILY_IDS
    }
    repeat_by_selector = {
        selector: {
            "row_count": sum(item["branch_selector_id"] == selector for item in repeats),
            "agreement_rate": _mean([
                float(item["repeat_success"])
                for item in repeats if item["branch_selector_id"] == selector
            ]),
        }
        for selector in C.REPEAT_BRANCH_IDS
    }
    repeat_by_family = {
        family: {
            "row_count": sum(item["family"] == family for item in repeats),
            "agreement_rate": _mean([
                float(item["repeat_success"])
                for item in repeats if item["family"] == family
            ]),
        }
        for family in C.FAMILY_IDS
    }
    left_right_asymmetry = max(
        abs(direction["LEFT"]["coverage_rate"] - direction["RIGHT"]["coverage_rate"]),
        abs(direction["LEFT"]["ranker_execution_rate"] - direction["RIGHT"]["ranker_execution_rate"]),
    )
    far_limitation = max(
        distance["NEAR"]["coverage_rate"] - distance["FAR"]["coverage_rate"],
        distance["NEAR"]["ranker_execution_rate"] - distance["FAR"]["ranker_execution_rate"],
    )
    support_counts = waypoints["target_support_counts"]
    secondary: list[str] = []
    if support_counts["TARGET_NODE_CENTRE"]["outside"] > 0:
        secondary.append("NODE_CENTRE_TARGET_OUT_OF_DISTRIBUTION")
    if _materially_outperforms(target_summaries["DIRECTED_EDGE_PORT"], node_summary):
        secondary.append("DIRECTED_PORT_TARGET_SIGNAL")
    if _materially_outperforms(target_summaries["ROUTE_LOOKAHEAD"], node_summary):
        secondary.append("ROUTE_LOOKAHEAD_TARGET_SIGNAL")
    if any(value["coverage_rate"] < 0.90 for value in direction.values()):
        secondary.append("CANDIDATE_BANK_DIRECTIONAL_GAP")
    if left_right_asymmetry >= 0.20:
        secondary.append("LEFT_RIGHT_ASYMMETRY")
    if far_limitation >= 0.20:
        secondary.append("PORT_DISTANCE_LIMITATION")
    if support_counts[selected_target]["outside"] > 0 and ranker_summary["selected_correct_edge_execution_rate"] < oracle_summary["selected_correct_edge_execution_rate"]:
        secondary.append("RANKER_TARGET_DISTRIBUTION_SHIFT")
    if repeatability_rate < C.HANDOFF_GATE["repeatability_rate_minimum"] or not command_tracking["passed"]:
        secondary.append("CONTROLLER_TRACKING_LIMITATION")
    secondary = [name for name in C.SECONDARY_CLASSIFICATIONS if name in secondary]

    return C.attach_content_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.metrics.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "evidence_counts": {
            "prospective_teacher_traces": C.TEACHER_TRACE_COUNT,
            "selected_states": C.STATE_COUNT,
            "candidate_branches": len(fanout),
            "development_target_rows": len(development["state_target_rows"]),
            "heldout_rows": len(heldout),
            "repeat_rows": len(repeats),
        },
        "panel": {
            "role_counts": copy.deepcopy(C.ROLE_COUNTS),
            "family_counts": {family: C.STATES_PER_FAMILY for family in C.FAMILY_IDS},
            "coverage_rate": coverage_rate,
            "covered_heldout_state_count": sum(coverage_by_state.values()),
            "two_or_more_correct_candidate_state_count": sum(
                value >= 2 for value in correct_candidate_count_by_state.values()
            ),
            "two_or_more_correct_candidate_rate": two_candidate_coverage_rate,
            "per_family_coverage": family_coverage,
            "candidate_identity_contributions": candidate_identity_contributions,
        },
        "development": {
            "target_summaries": [target_summaries[target] for target in C.TARGET_IDS],
            "selected_target_id": selected_target,
            "selected_target_index": development["selected_target_index"],
            "development_selected_target_passes_ranker_thresholds": development_selected_pass,
            "selected_target_passes_full_heldout_handoff_gate": full_heldout_handoff_gate,
            "selected_target_materially_outperforms_node_centre": selected_material,
        },
        "heldout": {
            "condition_summaries": [condition_summaries[condition] for condition in C.HELDOUT_CONDITION_IDS],
            "oracle_covered_state_correct_execution_rate": oracle_covered_rate,
            "family_ranker_correct_execution_counts": family_execution,
            "per_family_condition_summaries": per_family_conditions,
        },
        "repeatability": {"rate": repeatability_rate, "successful_rows": sum(bool(item["repeat_success"]) for item in repeats), "row_count": len(repeats), "by_selector": repeat_by_selector, "by_family": repeat_by_family},
        "command_tracking": command_tracking,
        "runtime_environments": {
            "physical": copy.deepcopy(panel["physical_runtime_environment"]),
            "encoder": copy.deepcopy(latents["encoder_runtime_environment"]),
            "ranker": copy.deepcopy(development["ranker_runtime_environment"]),
            "any_fake_runtime": bool(
                panel["physical_runtime_environment"]["fake_runtime"]
                or latents["encoder_runtime_environment"]["fake_runtime"]
                or development["ranker_runtime_environment"]["fake_runtime"]
            ),
        },
        "stratified": {
            "route_direction": direction,
            "port_distance": distance,
            "left_right_maximum_gap": left_right_asymmetry,
            "far_port_maximum_deficit": far_limitation,
        },
        "classification_input": classification_input,
        "gate": {"authority": copy.deepcopy(C.HANDOFF_GATE), "passed": disposition["handoff_gate_passed"]},
        "component_failures": disposition["component_failures"],
        "active_components_in_precedence_order": disposition["active_components_in_precedence_order"],
        "earliest_failing_component": disposition["earliest_failing_component"],
        "primary_classification": disposition["primary_classification"],
        "secondary_classifications": secondary,
        "next_experiment": disposition["next_experiment"],
        "predecessor_context": {
            "stage_a": C.V2_CONTEXT_BINDING["result_classifications"]["stage_a"],
            "stage_b": C.V2_CONTEXT_BINDING["result_classifications"]["stage_b"],
            "interpretation": C.V2_CONTEXT_BINDING["scientific_interpretation"],
            "runtime_reuse_authorized": False,
        },
        "claims": copy.deepcopy(C.CLAIMS),
        "safety_workstream": C.SAFETY_WORKSTREAM_STATUS,
        "prohibited_action_counters": copy.deepcopy(C.SCIENTIFIC_PROHIBITIONS),
    })


__all__ = [
    "PhysicalGraphEdgeHandoffMetricsError",
    "classify_physical_handoff_aggregates",
    "external_artifact_bindings",
    "first_registered_port_crossing",
    "panel_manifest_authority",
    "physical_trace_reduction_authority",
    "point_in_polygon_inclusive",
    "predecessor_context_binding",
    "recompute_metrics",
    "reducer_authority",
    "runtime_environment_sha256",
    "transverse_port_crossing",
    "validate_candidate_fanout_rows",
    "validate_development_target_selection",
    "validate_edge_port_index",
    "validate_graph_manifest",
    "validate_heldout_ranker_score_rows",
    "validate_latent_index",
    "validate_npz_inspections",
    "validate_panel_manifest",
    "validate_pixel_index",
    "validate_physical_runtime_environment",
    "validate_repeated_execution_rows",
    "validate_split_manifest",
    "validate_state_snapshot_index",
    "validate_teacher_trace_index",
    "validate_waypoint_contracts",
    "validate_visual_runtime_environment",
]
