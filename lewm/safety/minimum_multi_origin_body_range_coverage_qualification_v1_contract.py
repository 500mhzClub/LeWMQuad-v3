"""Prospective contract for MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1.

This module is payload-free.  Importing it reads neither the repaired transition
corpus nor any scientific result, Git state, simulator, GPU, or output tree.  It
only exposes the frozen no-training experiment contract, output schema,
deterministic phase binding, and fail-closed canonical receipt helpers.

The experiment asks a sensor-observability question.  Its true-future clouds use
actual transition geometry and therefore do not establish planning-time
prediction, deployment safety, or learned navigation capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1"
CONTRACT_SCHEMA_VERSION = (
    "minimum_multi_origin_body_range_coverage_qualification_v1.contract.v1"
)
OUTPUT_SCHEMA_VERSION = (
    "minimum_multi_origin_body_range_coverage_qualification_v1.output.v1"
)
PHASE_NAMESPACE = (
    "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1/L2_PHASE_V1"
)

REGRESSION_CONDITION_IDS = (
    "REALISTIC_PLATFORM_SCAN",
    "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
)
DUAL_CONDITION_IDS = (
    "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "DUAL_DENSE_L2_FOV_UPPER_BOUND",
    "DUAL_REALISTIC_L2_SCAN",
)
THREE_CONDITION_IDS = (
    "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "THREE_DENSE_L2_FOV_UPPER_BOUND",
    "THREE_REALISTIC_L2_SCAN",
)
DIAGNOSTIC_CONDITION_IDS = (
    "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC",
)
CONDITION_IDS = (
    *REGRESSION_CONDITION_IDS,
    *DUAL_CONDITION_IDS,
    *THREE_CONDITION_IDS,
    *DIAGNOSTIC_CONDITION_IDS,
)
EVIDENCE_MODE_IDS = (
    "PLANNING_TIME_CAUSAL_CLOUD",
    "TRUE_FUTURE_OBSERVABILITY_CLOUD",
)

MOUNT_IDS = (
    "HEAD_STOCK",
    "REAR_TOP_TRUNK",
    "LEFT_UPPER_FLANK",
    "RIGHT_UPPER_FLANK",
)
SUPPLEMENTAL_MOUNT_IDS = MOUNT_IDS[1:]
ORIENTATION_IDS = (
    "LEVEL",
    "INVERTED",
    "OUTWARD_DOWNWARD",
    "INWARD_DOWNWARD",
)
PAIR_LAYOUT_IDS = (
    "HEAD_STOCK__REAR_TOP_TRUNK",
    "HEAD_STOCK__LEFT_UPPER_FLANK",
    "HEAD_STOCK__RIGHT_UPPER_FLANK",
)
THREE_LAYOUT_IDS = (
    "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK",
    "HEAD_STOCK__REAR_TOP_TRUNK__RIGHT_UPPER_FLANK",
    "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK",
)
ALL_FOUR_LAYOUT_ID = (
    "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
)

PRIMARY_CLASSIFICATIONS = (
    "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL",
    "THREE_ORIGIN_REALISTIC_RANGE_SIGNAL",
    "DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK",
    "THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK",
    "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
)
SECONDARY_CLASSIFICATIONS = (
    "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED",
    "PLANNING_TIME_MULTI_ORIGIN_OBSERVABILITY_LIMITATION",
    "ROBOT_BODY_SELF_OCCLUSION",
    "VERTICAL_FOV_LIMITATION",
    "SCAN_DENSITY_OR_TIMING_LIMITATION",
    "ASSUMED_SENSOR_CONTRACT",
    "REPLANNING_INTERFACE_UNRESOLVED",
)
COVERAGE_ERROR_CLASSES = (
    "INSUFFICIENT_ORIGIN_COUNT",
    "VERTICAL_FOV_LIMITATION",
    "SCAN_PATTERN_SPARSITY",
    "SCAN_TIMING_LIMITATION",
    "ROBOT_SELF_OCCLUSION",
    "NEAR_BLIND_REGION",
    "MOUNT_POSITION_LIMITATION",
    "POINT_FUSION_ERROR",
    "UNRESOLVED",
)
COMPUTE_CLASSIFICATIONS = (
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL",
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY",
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO",
)
BODY_REGION_IDS = (
    "TRUNK",
    "FRONT_LIMBS",
    "REAR_LIMBS",
    "HIPS_AND_THIGHS",
    "CALVES",
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "minimum_multi_origin_body_range_coverage_qualification_v1"
)
TRACKED_PREREGISTRATION_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "preregistration_2026-08-26.md"
)
TRACKED_CONTRACT_RECEIPT_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "contract_2026-08-26.json"
)
TRACKED_OUTPUT_SCHEMA_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "output_schema_2026-08-26.json"
)
TRACKED_MOUNT_LIBRARY_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "mount_library_2026-08-26.json"
)
TRACKED_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "source_closure_2026-08-26.json"
)
TRACKED_FIXTURE_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "fixture_2026-08-26.json"
)
TRACKED_RESULT_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "result_2026-08-26.json"
)
TRACKED_REPORT_PATH = Path(
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_"
    "result_2026-08-26.md"
)

# Prospectively computed in the preserved Python/NumPy environment from only
# the frozen nominal URDF geometry and the outcome-independent static selector.
MOUNT_LIBRARY_RECEIPT_SHA256 = (
    "47028f5ca82e983995aac6dea989acba1dd049fcffb553bbbee464b788e5d7b8"
)


class ContractError(ValueError):
    """Raised when a contract, schema, or bound receipt fails validation."""


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{location} contains a non-string JSON key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ContractError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON without a trailing newline."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_lower_sha256(value: str, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase 64-hex SHA-256")
    return bytes.fromhex(value)


def derive_multi_origin_scan_phases(
    *,
    contract_digest_sha256: str,
    transition_uid: str,
    mount_id: str,
) -> dict[str, Any]:
    """Derive one deterministic, independent realistic-scan phase per origin.

    The exact UTF-8 transition UID and canonical mount ID are included in the
    hash.  The function accepts no label, contact, clearance, H3, or outcome
    argument.  The first two big-endian uint64 words bind azimuth and elevation
    phases in turns.
    """

    contract_digest = _require_lower_sha256(
        contract_digest_sha256, "contract_digest_sha256"
    )
    if not isinstance(transition_uid, str) or not transition_uid:
        raise ContractError("transition_uid must be a non-empty string")
    if mount_id not in MOUNT_IDS:
        raise ContractError(f"mount_id must be one of {MOUNT_IDS!r}")
    raw_digest = hashlib.sha256(
        PHASE_NAMESPACE.encode("utf-8")
        + b"\x00"
        + contract_digest
        + b"\x00"
        + transition_uid.encode("utf-8")
        + b"\x00"
        + mount_id.encode("ascii")
    ).digest()
    return {
        "phase_namespace": PHASE_NAMESPACE,
        "contract_digest_sha256": contract_digest_sha256,
        "transition_uid": transition_uid,
        "mount_id": mount_id,
        "phase_digest_sha256": raw_digest.hex(),
        "horizontal_phase_cycles": int.from_bytes(raw_digest[0:8], "big") / 2**64,
        "vertical_phase_cycles": int.from_bytes(raw_digest[8:16], "big") / 2**64,
    }


_OUTPUT_SCHEMA_CORE: dict[str, Any] = {
    "schema_version": OUTPUT_SCHEMA_VERSION,
    "experiment_id": EXPERIMENT_ID,
    "serialization": {
        "canonical_receipts": (
            "UTF-8, sorted keys, compact separators, finite numbers, LF terminated"
        ),
        "operational_and_result_json": (
            "UTF-8, sorted keys, two-space indentation, finite numbers, LF terminated"
        ),
        "jsonl": "one canonical finite JSON object followed by LF per row",
        "missing_numeric_value": None,
        "infinity_policy": (
            "encode unsupported or unbounded values as null with an explicit status"
        ),
    },
    "files": {
        "source_closure_receipt": {
            "tracked_path": str(TRACKED_SOURCE_CLOSURE_PATH),
            "cardinality": 1,
            "required_keys": [
                "schema",
                "experiment",
                "files",
                "predecessor_result_commit",
                "excludes",
                "source_freeze_commit",
                "content_digest",
            ],
        },
        "fixture_receipt": {
            "tracked_path": str(TRACKED_FIXTURE_PATH),
            "cardinality": 1,
            "required_keys": [
                "schema",
                "experiment",
                "core",
                "metrics",
                "pass",
                "content_digest",
            ],
            "core_required_keys": [
                "schema",
                "fixtures",
                "raw_fixture_evidence",
                "requirements",
                "pass",
                "content_digest",
            ],
            "raw_fixture_evidence_schema": (
                "minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1"
            ),
            "raw_fixture_evidence_required_keys": [
                "schema",
                "complete_raw_ray_queries",
                "complete_reduced_origin_evidence",
                "reconstructible_noncloud_inputs",
                "raw_fixture_cloud_policy",
                "content_digest",
            ],
            "complete_raw_ray_query_fixture_ids": [
                "near_blind",
                "between_scan_samples",
                "one_origin_occluded_another_observes",
            ],
            "raw_ray_query_required_keys": {
                "near_blind": [
                    "origin_world_xyz_m",
                    "directions_world",
                    "timestamps_s",
                    "near_m",
                    "far_m",
                    "environment_boxes",
                    "robot_primitives",
                    "first_hits",
                ],
                "between_scan_samples": [
                    "origin_world_xyz_m",
                    "directions_world",
                    "timestamps_s",
                    "near_m",
                    "far_m",
                    "environment_boxes",
                    "robot_primitives",
                    "first_hits",
                    "continuum_query",
                ],
                "one_origin_occluded_another_observes": [
                    "target_world_xyz_m",
                    "environment_boxes",
                    "robot_primitives",
                    "blocked_origin_world_xyz_m",
                    "observed_origin_world_xyz_m",
                    "blocked_direction_world",
                    "observed_direction_world",
                    "timestamps_s",
                    "near_m",
                    "far_m",
                    "blocked_result",
                    "observed_result",
                ],
            },
            "complete_reduced_origin_evidence_fixture_ids": [
                "one_origin_occluded_another_observes",
                "complementary_left_right_flank",
                "complementary_head_rear",
                "synchronized_scan_overlap",
            ],
            "reconstructible_noncloud_input_required_keys": [
                "robot_primitives",
                "clear_query_points_world_xyz_m",
                "clear_query_result_m",
                "contact_queries",
                "scan_phase_inputs",
                "safe_successor_inputs",
                "threshold_tie_input",
                "h3_rows",
            ],
            "raw_fixture_persistence": (
                "embedded canonical JSON retains every origin, direction, timestamp, "
                "environment and robot primitive, physical first hit, and range status "
                "for every finite ray in the near-blind, between-scan-samples, and "
                "occluded-versus-observed fixtures; it retains complete reduced "
                "per-origin evidence for every multi-origin union fixture and complete "
                "reconstructible inputs for clear/contact/H3/safe/tie/phase fixtures"
            ),
            "regeneration": "must be byte-identical before every scientific execution",
        },
        "preexecution_receipt": {
            "relative_path": "preexecution_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "head",
                "contract_sha256",
                "output_schema_sha256",
                "source_closure_sha256",
                "mount_library_receipt_sha256",
                "mount_library_sha256",
                "predecessor_bindings",
                "inputs",
                "environment",
                "workspace_filesystem",
                "output_filesystem",
                "predicted_temporary_storage_bytes",
                "predicted_final_storage_bytes",
                "no_active_scientific_process",
                "pass",
                "content_digest",
            ],
        },
        "environment_receipt": {
            "relative_path": "receipts/environment_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "python",
                "packages",
                "package_sources",
                "import_closure",
                "experiment_import_closure",
                "environment_prefix_resolved",
                "cpu_only",
                "torch_loaded",
                "genesis_steps",
                "tinyquadjepa_required",
                "model_checkpoints_opened",
                "content_digest",
            ],
        },
        "mount_library_receipt": {
            "tracked_path": str(TRACKED_MOUNT_LIBRARY_PATH),
            "output_relative_path": "receipts/mount_library.json",
            "cardinality": 1,
            "required_keys": [
                "schema",
                "experiment_id",
                "selection_role",
                "nominal_geometry_source",
                "trunk_envelope",
                "housing_full_extents_xyz_m",
                "housing_occlusion_frame",
                "housing_clearance_validation",
                "mechanical_clearance_m",
                "mount_candidates",
                "orientation_library_ids",
                "orientation_selections",
                "layouts",
                "body_region_ids",
                "witness_count",
                "witness_digest",
                "contact_outcomes_used",
                "outcome_fields_read",
                "pass",
                "content_digest",
            ],
            "selected_orientation_required_keys": [
                "mount_id",
                "selected_orientation_id",
                "selected_pose",
                "candidates",
                "selection_rule",
            ],
            "selected_pose_required_keys": [
                "mount_id",
                "orientation_id",
                "parent_link",
                "pole_body_xyz",
                "translation_body_xyz_m",
                "quaternion_body_wxyz",
                "rpy_body_rad",
            ],
        },
        "layout_selection_receipt": {
            "relative_path": "layout_selection/selected_layouts.json",
            "cardinality": 1,
            "required_keys": [
                "role",
                "training_state_count",
                "pair_candidates",
                "three_candidates",
                "selected_pair_layout_id",
                "selected_three_layout_id",
                "selection_metric_order",
                "candidate_metrics",
                "outcome_fields_used_by_layout_objective",
                "contact_labels_used_for_layout_selection",
                "frozen_outcomes_read_only_for_corpus_custody_validation",
                "calibration_rows_read",
                "heldout_rows_read",
                "frozen_before_calibration",
                "label_free_geometry_reuse",
                "training_layout_evidence",
                "pass",
                "content_digest",
            ],
            "label_free_geometry_reuse_required_keys": [
                "fields",
                "boundary_snapshot_digest_required",
                "numeric_tolerance",
                "action_group_partition_only",
                "outcome_fields_accessed",
                "representatives",
            ],
            "candidate_metric_required_keys": [
                "layout_id",
                "mount_ids",
                "region_support",
                "per_link_counts",
                "per_link_support",
                "minimum_body_region_support",
                "transition_support_p05",
                "rear_limb_support",
                "calf_support",
                "overall_mean_support",
                "self_occluded_fraction",
                "minimum_family_support",
                "per_family_support",
                "transition_count",
            ],
            "per_link_metric_semantics": (
                "candidate_metrics reports supported and total protected-link/physics-step "
                "witness counts and their support fraction for each of the exact 13 frozen "
                "protected links"
            ),
            "training_layout_evidence_binding_required_keys": [
                "schema",
                "path",
                "sha256",
                "bytes",
                "rows",
                "role",
                "layout_ids",
                "body_region_ids",
                "outcome_fields_used_by_layout_objective",
                "content_digest",
            ],
        },
        "training_layout_evidence": {
            "relative_path": "layout_selection/training_layout_evidence.jsonl.gz",
            "cardinality": (
                "one deterministic gzip JSONL row per transition in all 128 training-role "
                "states; one row contains metrics for all six frozen layout candidates"
            ),
            "compression": "deterministic gzip with mtime=0 and no original filename",
            "required_keys": [
                "schema",
                "state_id",
                "family",
                "role",
                "transition_index",
                "transition_key",
                "transition_kind",
                "current_action_index",
                "action_index",
                "layout_metrics",
                "contact_label",
                "safe_action_count",
                "route_outcome",
            ],
            "row_schema": "minimum_multi_origin_training_layout_transition_evidence_v1",
            "layout_metric_required_keys": [
                "mount_ids",
                "supported_witnesses",
                "total_witnesses",
                "support_fraction",
                "unsupported_swept_volume_fraction",
                "nominally_eligible_witnesses",
                "all_origin_self_occluded_witnesses",
                "self_occluded_fraction_of_nominal",
                "body_region_counts",
                "body_region_support",
                "protected_link_counts",
                "protected_link_support",
            ],
            "protected_link_cardinality": 13,
            "layout_ids": [*PAIR_LAYOUT_IDS, *THREE_LAYOUT_IDS],
            "body_region_ids": list(BODY_REGION_IDS),
            "forbidden_outcome_value_rule": {
                "contact_label": None,
                "safe_action_count": None,
                "route_outcome": None,
            },
        },
        "predecessor_regression_receipt": {
            "relative_path": "receipts/predecessor_regression.json",
            "cardinality": 1,
            "required_keys": [
                "predecessor_result_commit",
                "predecessor_result_content_sha256",
                "predecessor_result_file_sha256",
                "condition_ids",
                "evidence_modes",
                "referenced_thresholds",
                "referenced_metrics",
                "row_evidence_reused_by_reference",
                "rematerialized_rows",
                "reinterpretation",
                "pass",
                "content_digest",
            ],
        },
        "execution_plan_receipt": {
            "relative_path": "receipts/execution_plan.json",
            "cardinality": 1,
            "required_keys": [
                "schema",
                "experiment_id",
                "source_freeze_commit",
                "selected_dual_layout_id",
                "selected_three_layout_id",
                "dual_realistic_pass",
                "three_origin_executed",
                "all_four_diagnostic_executed",
                "executed_condition_ids",
                "phase_order",
                "materialization_indices",
                "threshold_freezes",
                "conditional_rule",
                "pass",
                "content_digest",
            ],
            "conditional_semantics": (
                "executed_condition_ids and the two conditional execution Booleans must be "
                "validated against the prospectively frozen stop/continue rule"
            ),
        },
        "materialization_index": {
            "relative_path": "materialization/{dual,three,diagnostic}_index.json",
            "cardinality": (
                "dual exactly once; three iff dual realistic fails; diagnostic iff selected "
                "three dense spherical fails"
            ),
            "required_keys": [
                "source_freeze_commit",
                "contract_sha256",
                "source_closure_sha256",
                "layout_selection_sha256",
                "phase",
                "states",
                "transitions",
                "action_representatives",
                "geometry_representatives",
                "physics_frames",
                "protected_links",
                "conditions",
                "scan_counts",
                "dense_analytic_target_query_counts",
                "support_dominance",
                "state_records",
                "records",
                "raw_audit_manifest_path",
                "raw_audit_manifest_sha256",
                "storage_bytes",
                "runtime_s",
                "status",
                "content_digest",
            ],
            "state_record_required_keys": [
                "state_id",
                "scene_id",
                "family",
                "role",
                "transitions",
                "action_representatives",
                "geometry_representatives",
                "condition_records",
                "state_receipt_path",
                "state_receipt_sha256",
                "state_receipt_bytes",
                "runtime_s",
            ],
            "state_phase_receipt_required_keys": [
                "schema",
                "status",
                "state_id",
                "scene_id",
                "family",
                "role",
                "conditions",
                "transitions",
                "transition_uid_by_index",
                "transition_uid_by_index_sha256",
                "action_representatives",
                "geometry_representatives",
                "condition_records",
                "scan_receipts",
                "support_dominance",
                "raw_audit",
                "contract_sha256",
                "source_closure_sha256",
                "source_freeze_commit",
                "predecessor_state_sha256",
                "runtime_s",
                "simulator_steps",
                "training_steps",
                "jepa_predictor_opens",
                "content_digest",
            ],
            "transition_uid_binding_semantics": (
                "transition_uid_by_index has exact state transition cardinality, contains "
                "unique canonical transition UIDs in index order, and is bound by "
                "transition_uid_by_index_sha256; every finite-scan receipt must match both "
                "its transition index and UID before the deterministic phase is recomputed"
            ),
            "nested_condition_record_required_keys": [
                "condition_id",
                "path",
                "sha256",
                "bytes",
            ],
            "npz_required_arrays": [
                "action_representative_transition",
                "geometry_representative_transition",
                "support",
                "finite_scan_support_inherited",
                "per_origin_support",
                "per_origin_event_time_support",
                "per_origin_nominal_fov",
                "per_origin_direct_visibility",
                "per_origin_self_occluded",
                "per_origin_finite_scan_support_inherited",
                "per_origin_point_support_count",
                "per_origin_support_acquisition_index",
                "per_origin_nearest_ray_index",
                "per_origin_point_age_s",
            ],
            "scan_receipt_required_keys": [
                "transition_index",
                "transition_uid",
                "condition_id",
                "evidence_mode",
                "layout_id",
                "mount_id",
                "phase",
                "ray_count",
                "render_cache_reused",
                "dense_l2_inherited_support_witnesses",
                "dense_spherical_inherited_support_witnesses",
            ],
            "support_dominance_chain_key_format": (
                "<REALISTIC_CONDITION_ID><=<MATCHED_DENSE_L2_FOV_CONDITION_ID>"
                "<=<MATCHED_DENSE_SPHERICAL_CONDITION_ID>"
            ),
            "support_dominance_evidence_modes": list(EVIDENCE_MODE_IDS),
            "support_dominance_required_keys": [
                "condition_chain",
                "evidence_mode",
                "per_origin_queries_checked",
                "per_origin_subset_violations",
                "fused_queries_checked",
                "fused_subset_violations",
                "clearance_monotonicity_pairs_checked",
                "clearance_monotonicity_violations",
                "maximum_clearance_excess_m",
                "inherited_support_witnesses_by_origin",
                "pass",
            ],
            "support_dominance_pass_rule": (
                "per_origin_queries_checked > 0, fused_queries_checked > 0, "
                "clearance_monotonicity_pairs_checked > 0, per_origin_subset_violations == 0, "
                "fused_subset_violations == 0, and clearance_monotonicity_violations == 0 for "
                "the full three-level chain in every executed evidence mode; clearance "
                "tolerance is exactly 1e-9 m"
            ),
            "support_dominance_phase_semantics": (
                "every dual and three state receipt and aggregate index requires exactly one "
                "expected complete three-level condition-chain key with exactly both evidence "
                "modes; the spherical-only all-four diagnostic requires an empty "
                "support_dominance object"
            ),
            "record_required_keys": [
                "state_id",
                "scene_id",
                "family",
                "role",
                "condition_id",
                "transitions",
                "shard_path",
                "shard_sha256",
                "shard_bytes",
            ],
            "state_shard_semantics": (
                "one NPZ shard per newly materialized condition and frozen state; predecessor "
                "regression evidence is bound by reference and is not copied into these shards"
            ),
            "scan_count_semantics": (
                "for each REALISTIC condition/mode, sensor_scans = unique_rendered_scans = "
                "29,470 x selected layout origin count and rays = sensor_scans x 6,400; dense "
                "conditions use exact-geometry representative reuse and do not claim finite scans"
            ),
        },
        "calibration_thresholds": {
            "relative_path": "calibration/{phase}_thresholds_frozen.json",
            "cardinality": (
                "one per materialization phase that was prospectively required and executed"
            ),
            "required_keys": [
                "phase",
                "calibration_role",
                "calibration_state_count",
                "materialization_index_sha256",
                "heldout_rows_used_by_selection",
                "thresholds",
                "pass",
                "content_digest",
            ],
        },
        "transition_evidence": {
            "relative_path": "evidence/transition_evidence.jsonl.gz",
            "cardinality": (
                "one gzip JSONL row per frozen transition x each newly executed multi-origin "
                "condition/evidence mode; predecessor regression rows remain bound by reference"
            ),
            "compression": "deterministic gzip with mtime=0 and no original filename",
            "required_keys": [
                "transition_uid",
                "state_id",
                "transition_index",
                "action_representative_transition_index",
                "geometry_representative_transition_index",
                "role",
                "family",
                "condition_id",
                "evidence_mode",
                "layout_id",
                "origin_mount_ids",
                "oracle_contact",
                "sensor_global_minimum_clearance_m",
                "unsupported_risk",
                "threshold_m",
                "predicted_contact",
                "predicted_safe_next_action_count",
                "oracle_safe_next_action_count",
                "admitted",
                "per_link_row_count",
            ],
        },
        "row_level_evidence_receipt": {
            "relative_path": "evidence/row_level_evidence_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "transition_evidence",
                "per_link_evidence",
                "regression_evidence_authority",
                "pass",
                "content_digest",
            ],
        },
        "per_link_evidence": {
            "relative_path": "evidence/per_link_evidence.jsonl.gz",
            "cardinality": (
                "one gzip JSONL row per frozen transition x each newly executed multi-origin "
                "condition/mode x 13 links"
            ),
            "compression": "deterministic gzip with mtime=0 and no original filename",
            "required_keys": [
                "transition_uid",
                "state_id",
                "transition_index",
                "action_representative_transition_index",
                "geometry_representative_transition_index",
                "role",
                "family",
                "condition_id",
                "evidence_mode",
                "layout_id",
                "protected_link",
                "collision_region",
                "minimum_observed_environment_clearance_m",
                "time_to_minimum_clearance_s",
                "first_threshold_crossing_time_s",
                "obstacle_direction_body_rad",
                "obstacle_sector",
                "observation_support",
                "unsupported_swept_volume_fraction",
                "responsible_environment_object",
                "oracle_contact_link",
                "nominal_fov_inclusion_by_origin",
                "direct_visibility_by_origin",
                "observation_support_by_origin",
                "self_occlusion_by_origin",
                "finite_scan_support_inherited_by_origin",
                "finite_scan_support_inherited_count",
                "point_support_count_by_origin",
                "supporting_origin_ids",
                "responsible_origin_id",
                "responsible_acquisition_time_s",
                "support_acquisition_times_s_by_origin",
                "nearest_ray_or_point_by_origin",
                "point_age_s_by_origin",
            ],
        },
        "coverage_errors": {
            "relative_path": "evidence/coverage_errors.jsonl.gz",
            "cardinality": "one row per complete frozen-threshold coverage error",
            "compression": "deterministic gzip with mtime=0 and no original filename",
            "required_keys": [
                "transition_uid",
                "state_id",
                "condition_id",
                "evidence_mode",
                "layout_id",
                "robot_link_or_body_region",
                "contact_or_minimum_clearance_physics_step",
                "origin_visibility",
                "origin_self_occlusion",
                "origin_scan_point_availability",
                "supporting_origin_ids",
                "nearest_ray_or_point_by_origin",
                "point_age_s_by_origin",
                "exact_clearance_m",
                "sensor_derived_clearance_m",
                "error_class",
            ],
            "error_class_enum": list(COVERAGE_ERROR_CLASSES),
        },
        "coverage_error_receipt": {
            "relative_path": "evidence/coverage_error_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "path",
                "sha256",
                "bytes",
                "rows",
                "counts",
                "compression",
                "pass",
                "content_digest",
            ],
        },
        "raw_audit_manifest": {
            "relative_path": "raw_audit/{dual,three,diagnostic}_manifest.jsonl",
            "cardinality": "one manifest per prospectively executed materialization phase",
            "required_keys": [
                "transition_uid",
                "role",
                "family",
                "transition_kind",
                "condition_id",
                "evidence_mode",
                "layout_id",
                "mount_ids",
                "mount_id",
                "phase_digest_sha256",
                "phase_digest_sha256_by_mount",
                "source_action_representative_transition_uid",
                "source_geometry_representative_transition_uid",
                "boundary_snapshot_digest",
                "selection_sha256",
                "artifact_relative_path",
                "artifact_sha256",
                "point_or_witness_count",
                "bytes",
            ],
        },
        "compute_benchmark": {
            "relative_path": "compute_benchmark.json",
            "cardinality": 1,
            "required_keys": [
                "condition_id",
                "evidence_mode",
                "threshold_m",
                "representative_state_ids",
                "numeric_binding",
                "timed_sample_unit",
                "includes",
                "includes_ray_generation",
                "includes_future_trajectory_acquisition",
                "complete_set",
                "per_state",
                "classification",
                "peak_rss_bytes",
                "peak_vram_bytes",
            ],
        },
        "result": {
            "relative_path": "result.json",
            "tracked_path": str(TRACKED_RESULT_PATH),
            "cardinality": 1,
            "required_keys": [
                "schema_version",
                "experiment_id",
                "source_freeze_commit",
                "predecessor_result_commit",
                "contract_sha256",
                "output_schema_sha256",
                "source_closure_sha256",
                "mount_library_receipt_sha256",
                "mount_library_sha256",
                "environment_receipt",
                "storage",
                "storage_preflight",
                "storage_actual",
                "fixture_results",
                "layout_selection",
                "execution_plan",
                "conditional_execution",
                "executed_conditions",
                "single_origin_regression",
                "materialisation_indices",
                "materialisation_counts",
                "support_dominance",
                "calibration_thresholds",
                "condition_metrics",
                "planning_time_true_future_comparison",
                "causal_vs_true_future",
                "gate_results",
                "coverage_error_counts",
                "coverage_errors",
                "coverage_support",
                "row_level_evidence",
                "compute_benchmark",
                "compute_classification",
                "hardware_accounting",
                "hardware_feasibility",
                "primary_classification",
                "secondary_classifications",
                "classification_receipt",
                "next_decision",
                "preserved_findings",
                "raw_audit_manifest",
                "runtime_s",
                "custody",
                "result_content_digest",
                "result_content_sha256",
            ],
        },
        "persistence_receipt": {
            "relative_path": "persistence_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "experiment_id",
                "contract_sha256",
                "source_closure_sha256",
                "mount_library_receipt_sha256",
                "layout_selection_receipt",
                "execution_plan_receipt",
                "executed_materialization_indices",
                "executed_threshold_freeze_receipts",
                "row_level_evidence",
                "coverage_errors",
                "raw_audit_manifests",
                "result",
                "report",
                "allocated_bytes_final",
                "final_storage_ceiling_bytes",
                "pass",
                "content_digest",
            ],
            "conditional_semantics": (
                "aggregate only prospectively executed phases; the bound execution-plan receipt "
                "and its conditional Booleans identify every phase skipped by the frozen rule"
            ),
        },
        "report": {
            "relative_path": "report.md",
            "tracked_path": str(TRACKED_REPORT_PATH),
            "cardinality": 1,
            "machine_readable_authority": "result.json and row-level JSONL ledgers",
        },
    },
    "condition_metric_groups": [
        "current_contact",
        "successor_contact",
        "combined_contact",
        "safe_action_count",
        "viability",
        "per_link",
        "per_region",
        "per_family",
        "coverage",
    ],
    "coverage_support_path": "result.coverage_support[condition_id]",
    "coverage_support_groups": [
        "per_origin",
        "per_link",
        "per_region",
        "per_family",
    ],
    "coverage_support_semantics": (
        "true-future held-out support/visibility attribution is reported separately from "
        "condition_metrics at the top-level coverage_support path"
    ),
    "contact_metric_keys": [
        "auc",
        "average_precision",
        "recall",
        "fnr",
        "negative_retention",
    ],
    "safe_action_count_metric_keys": [
        "mae",
        "spearman",
        "exact_count_accuracy",
        "zero_vs_nonzero_accuracy",
        "false_zero_rate",
        "false_nonzero_rate",
    ],
    "viability_metric_keys": [
        "states_retaining_admitted_action",
        "selected_immediate_contacts",
        "selected_oracle_nonviable_successors",
        "false_abstentions",
        "correct_abstentions",
        "unsafe_movement_decisions",
        "falsely_viable_candidates",
        "selected_h3_route_progress_m",
        "oracle_progress_fraction",
        "normalized_regret",
        "best_admissible_top1",
        "best_admissible_top3",
    ],
}

OUTPUT_SCHEMA_SHA256 = canonical_json_sha256(_OUTPUT_SCHEMA_CORE)
OUTPUT_SCHEMA: dict[str, Any] = {
    **copy.deepcopy(_OUTPUT_SCHEMA_CORE),
    "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
}


def _condition(
    condition_id: str,
    *,
    stage: str,
    layout: str,
    representation: str,
    evidence_modes: list[str],
    primary_eligible: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "id": condition_id,
        "stage": stage,
        "layout": layout,
        "representation": representation,
        "evidence_modes": evidence_modes,
        "primary_classification_eligible": primary_eligible,
        "source": source,
    }


def _contract_core() -> dict[str, Any]:
    output_root = str(OUTPUT_ROOT)
    full_modes = list(EVIDENCE_MODE_IDS)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_frozen_utc_date": "2026-08-26",
        "execution_class": "NO_TRAINING_DEVELOPMENT_MODE_END_TO_END_EXECUTION",
        "scientific_question": (
            "What is the minimum prospectively selected two- or three-origin body-range layout "
            "that observes enough protected geometry to reproduce the exact per-link two-ply "
            "micro-viability decision?"
        ),
        "active_policies": [
            "EVALUATION_FIRST_SINGLE_SEED",
            "ROW_LEVEL_EVIDENCE_PERSISTENCE",
            "DEVELOPMENT_MODE_END_TO_END_EXECUTION",
        ],
        "seeds": {
            "model_seed": None,
            "model_seed_reason": "no model is trained",
            "sensor_random_seed": None,
            "scan_phase": (
                "deterministic SHA-256 of contract digest, transition UID, and mount ID"
            ),
        },
        "claim_boundary": {
            "target": "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT",
            "positive_claim": (
                "simulated robot-environment contact/separation proxy over the committed "
                "one-tick 100 ms horizon"
            ),
            "true_future_limitation": (
                "actual current/successor transition geometry is an observability upper bound; "
                "it does not establish pre-action prediction"
            ),
            "does_not_establish": [
                "material-impact safety",
                "injury prevention",
                "property-damage prevention",
                "human safety",
                "fragile-infrastructure safety",
                "platform-equivalent emergency stopping",
                "closed-loop learned navigation safety",
            ],
        },
        "predecessor_bindings": {
            "experiment": "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1",
            "source_freeze_commit": "6fb55dec810b8fb8337d4519096f17a294c78425",
            "completed_result_commit": "d9748abe0fad0a25face56801f6b0c5e699db92f",
            "result_content_sha256": (
                "98699ac43046a4d1f425998217d5527637209ca233a872cf667e484123a967ac"
            ),
            "result_file_sha256": (
                "8844c0ee5a8bcdd28f505d64a670b5dee595933af05a00290f327cb8f3702019"
            ),
            "contract_sha256": (
                "f79eedb072dcfb32853b90881699ccbe3a3066ffd395970a655f980410713980"
            ),
            "source_closure_sha256": (
                "14ef33f786bc9f931b6bac214c020d694931c9efaf55bea11f48f131f7862dd5"
            ),
            "materialization_index_sha256": (
                "4e35e813d6f1bc2593c6a889a24e3de952b2797ce48f2c28224cfe16e5af398d"
            ),
            "threshold_freeze_sha256": (
                "8bda91c4127ec2a43e877f64ce58cde39379682bd674dc1354166290ec61b75e"
            ),
            "persistence_receipt_file_sha256": (
                "12bdb2e6e0e7b54a9ad1947126715ac85d367afcbfd403ca1c80ebd593f51a43"
            ),
            "persistence_receipt_content_digest": (
                "24c1e8677d0158838b1943680cfb66b3dfdc2cf52972f0b3af1e169544c1ae23"
            ),
            "transition_evidence_sha256": (
                "c25c5a0c8bcea8cdad3e7f14946126ae53098ed6d14f90a04ace5d053e303991"
            ),
            "per_link_evidence_sha256": (
                "78f5e21416b603fb6481968f2e6d70753cbb138dff49f85f3910ec1c12bae218"
            ),
            "coverage_errors_sha256": (
                "2e2402d33a1044a8b9b025a5d58ae14cbaab47221a451c2bf6d8ae1e4073849d"
            ),
            "primary_classification": "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
            "passing_sensor_conditions": [],
            "per_link_predictor_authorized": False,
            "training_authorized": False,
            "corpus_source_lineage_commit": (
                "10b3a190d506830e6a87e04a0f1c832b92295bd7"
            ),
            "corpus_logical_digest": (
                "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223"
            ),
            "corpus_index_sha256": (
                "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0"
            ),
            "action_contract_sha256": (
                "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06"
            ),
            "repaired_row_ledger_sha256": (
                "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94"
            ),
            "frozen_state_count": 176,
            "frozen_transition_count": 29470,
            "frozen_physics_frame_count": 1473500,
            "protected_link_count": 13,
            "protected_collision_shape_count": 27,
            "role_state_counts": {
                "training": 128,
                "internal_calibration": 24,
                "development_held_out": 24,
            },
            "regression_reuse": {
                "condition_ids": list(REGRESSION_CONDITION_IDS),
                "policy": (
                    "reuse byte-bound predecessor scans, row evidence, thresholds, and metrics "
                    "without rematerialisation unless integrity validation proves reuse impossible"
                ),
                "reinterpretation": "forbidden",
            },
            "mutable_fields": [],
        },
        "frozen_corpus": {
            "states": 176,
            "transitions": 29470,
            "physics_frames": 1473500,
            "protected_links": 13,
            "protected_collision_shapes": 27,
            "state_role_action_transition_contact_and_h3_identities": "immutable",
            "oracle_label": "repaired frozen_contact_label",
            "action_authority": "unique deployable applied-action contract",
            "route_authority": "deterministic H3 route scores",
            "fresh_panel": "forbidden",
        },
        "representative_maps": {
            "decision_action_copy_map": {
                "contract_id": "DECISION_ACTION_COPY_MAP",
                "representatives": 13385,
                "authority": (
                    "unique deployable applied-action identity for calibration, safe-action "
                    "counting, admission, and H3 selection"
                ),
                "sensor_reuse_authority": False,
            },
            "exact_sensor_materialization_map": {
                "contract_id": "EXACT_SENSOR_MATERIALIZATION_MAP",
                "representatives": 13584,
                "authority": (
                    "only predecessor map authorized to reuse exact transition geometry and "
                    "copy deterministic dense-condition per-origin sensor materialisation"
                ),
                "exact_fields": [
                    "qpos",
                    "link_transform",
                    "geom_transform",
                    "native_contact",
                    "exact_contact",
                    "frozen_contact_label",
                    "boundary_snapshot_digest",
                ],
                "numeric_tolerance": 0.0,
                "exact_reused_transition_pairs": 15886,
                "independently_materialized_nonexact_action_copy_pairs": 199,
                "realistic_scan_reuse_authority": False,
                "realistic_scan_exclusion": (
                    "every REALISTIC condition is independently rendered for all 29,470 "
                    "transition UIDs because its phase is transition-identity-bound even when "
                    "two rows have byte-identical transition geometry"
                ),
            },
            "mapping_change": "forbidden",
        },
        "roles": {
            "training": {
                "states": 128,
                "use": (
                    "label-free mount-orientation and pair/three-layout coverage selection only"
                ),
                "forbidden_selection_objective_fields": [
                    "frozen_contact_label",
                    "native_contact",
                    "exact_contact",
                    "safe-action labels",
                    "H3 outcomes",
                ],
            },
            "internal_calibration": {
                "states": 24,
                "use": "threshold calibration only after layouts are frozen",
            },
            "development_held_out": {
                "states": 24,
                "use": "evaluation only after layouts and thresholds are frozen",
            },
            "untouched_g2": "FORBIDDEN_NOT_READ",
        },
        "hardware_binding": {
            "binding_class": "ASSUMED_GO2_HEAD_LIDAR_L2",
            "sensor": "Unitree 4D LiDAR L2",
            "deployment_selection": False,
            "scan_classification": "APPROXIMATED_REALISTIC_PLATFORM_SCAN",
            "effective_points_per_s_per_origin": 64000,
            "raw_ranging_samples_per_s_per_origin": 128000,
            "rays_per_100ms_per_origin": 6400,
            "horizontal_fov_deg": 360.0,
            "elevation_min_deg": -6.0,
            "elevation_max_deg": 90.0,
            "near_blind_region_m": 0.05,
            "qualification_maximum_range_m": 30.0,
            "point_timestamps": "one deterministic relative timestamp per ray/point",
            "range_values": "ideal on rays that exist",
            "random_noise": "none",
            "random_dropout": "none",
            "robot_self_return_identity": "retained for coverage audit",
            "robot_self_return_clearance_use": "excluded from environment clearance",
            "interfaces": ["point cloud", "IMU"],
            "manual": {
                "version": "v1.1 (2024-10)",
                "sha256": (
                    "95a3e52ce5fa1cc9366095d646e6c6ff23de0fd23eddb1fff7bf5a698cbe76b1"
                ),
                "url": (
                    "https://oss-global-cdn.unitree.com/static/"
                    "Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf"
                ),
            },
            "housing": {
                "aabb_full_extents_m": [0.075, 0.075, 0.065],
                "mass_kg": 0.230,
                "model": (
                    "body-axis-aligned box centered on the ray origin independent of optical "
                    "orientation; explicit mixed development approximation"
                ),
                "full_housing_outside_static_trunk_required": True,
            },
            "power": {
                "typical_w_per_sensor": 10,
                "peak_w_per_sensor": 13,
                "source": "official Unitree L2 documentation",
            },
            "hardware_accounting_required": [
                "sensor_count",
                "aggregate_effective_point_rate_hz",
                "aggregate_rays_per_100ms",
                "estimated_data_bandwidth",
                "aggregate_payload_mass_kg",
                "aggregate_power",
                "mounting_envelope",
                "cable_routing",
            ],
            "unresolved_hardware_fields": {
                "data_bandwidth": (
                    "persist documented or explicitly calculated assumption with units"
                ),
                "cables": "persist plausibility and unresolved physical design",
            },
        },
        "scan_contract": {
            "classification": "APPROXIMATED_REALISTIC_PLATFORM_SCAN",
            "duration_s": 0.1,
            "sample_indices": "k=0,...,6399 independently for every origin",
            "timestamp_law": "t_k=(k+0.5)/64000 seconds",
            "azimuth_law": "-pi + 2*pi*frac(azimuth_phase + 5.55*t)",
            "elevation_law": (
                "u=frac(elevation_phase+216*t); triangle=1-4*abs(u-0.5); "
                "elevation_deg=-6+48*(triangle+1)"
            ),
            "phase_namespace": PHASE_NAMESPACE,
            "phase_binding_api": (
                "derive_multi_origin_scan_phases(contract_digest_sha256, transition_uid, mount_id)"
            ),
            "phase_digest": (
                "SHA-256(namespace UTF-8 || 0x00 || raw contract digest || 0x00 || "
                "transition UID UTF-8 || 0x00 || mount ID ASCII)"
            ),
            "origin_phase_independence": (
                "mount identity is hashed, so realistic scans from different origins have "
                "independent deterministic phases"
            ),
            "transition_identity_independence": (
                "every one of the 29,470 transition UIDs is independently rendered in each "
                "executed REALISTIC condition/mode; neither the exact-geometry map nor a "
                "shared planning-boundary digest authorizes cross-UID scan reuse"
            ),
            "realistic_scan_count_semantics": {
                "sensor_scans_per_condition_mode": (
                    "29,470 transitions multiplied by the selected layout origin count"
                ),
                "unique_rendered_scans": "exactly sensor_scans; render_cache_reused is false",
                "rays": "sensor_scans multiplied by 6,400 rays",
                "dense_conditions": (
                    "have analytic witness/acquisition counts, not finite realistic scan counts, "
                    "and may reuse the exact sensor materialization map"
                ),
            },
            "outcome_independence": (
                "phase derivation accepts no contact, clearance, route, action outcome, or label"
            ),
            "motion_compensation": (
                "world-frame fusion using each exact point timestamp and articulated pose"
            ),
        },
        "mounts": {
            "coordinate_convention": (
                "translation in base/body frame metres; orientation is body-from-sensor "
                "right-handed active rotation"
            ),
            "trunk_collision_envelope": {
                "center_m": [0.0, 0.0, 0.0],
                "half_extents_m": [0.1881, 0.04675, 0.057],
                "source": "frozen predecessor trunk collision-envelope bounding box",
            },
            "mechanical_clearance_m": 0.01,
            "static_full_housing_clearance_validation": {
                "receipt": str(TRACKED_MOUNT_LIBRARY_PATH),
                "required_mount_ids": list(MOUNT_IDS),
                "supplemental_protected_primitive_count": 27,
                "supplemental_required_clearance_m": 0.01,
                "housing_occlusion_frame": (
                    "BODY_AXIS_ALIGNED_TRUNK_FRAME_INDEPENDENT_OF_OPTICAL_RAY_FRAME"
                ),
                "all_rows_must_pass": True,
            },
            "HEAD_STOCK": {
                "parent_link": "base",
                "child_frame": "radar",
                "translation_m": [0.28945, 0.0, -0.046825],
                "rotation_rpy_rad": [0.0, 2.8782, 0.0],
                "source": "frozen stock Unitree Go2 base-to-radar fixed joint",
                "orientation_selection": "not searched",
                "emitting_housing_policy": (
                    "reuse the predecessor optimistic ray-only coarse head-housing exemption; "
                    "protected contact geometry remains unchanged"
                ),
            },
            "REAR_TOP_TRUNK": {
                "parent_link": "base",
                "translation_m": [-0.1254, 0.0, 0.0995],
                "selection_geometry": (
                    "rear-third centre of trunk top; housing half-height 0.0325 m plus "
                    "0.010 m clearance above z=0.057 m"
                ),
                "horizontal_outward_body_vector": [-1.0, 0.0, 0.0],
                "housing_separating_axis": "+Z",
                "payload_plausibility": "top payload position with plausible rear cable route",
            },
            "LEFT_UPPER_FLANK": {
                "parent_link": "base",
                "translation_m": [0.0, 0.09425, 0.0285],
                "selection_geometry": (
                    "trunk longitudinal centre and upper-quarter height; housing half-width "
                    "0.0375 m plus 0.010 m clearance outside y=+0.04675 m"
                ),
                "horizontal_outward_body_vector": [0.0, 1.0, 0.0],
                "housing_separating_axis": "+Y",
                "payload_plausibility": "upper flank with plausible body-routed cable",
            },
            "RIGHT_UPPER_FLANK": {
                "parent_link": "base",
                "translation_m": [0.0, -0.09425, 0.0285],
                "selection_geometry": "exact sagittal mirror of LEFT_UPPER_FLANK",
                "horizontal_outward_body_vector": [0.0, -1.0, 0.0],
                "housing_separating_axis": "-Y",
                "payload_plausibility": "upper flank with plausible body-routed cable",
            },
            "supplemental_mount_constraints": {
                "leg_mounts": "forbidden",
                "mount_search": "exactly the three frozen supplemental origins",
                "outcome_tuning": False,
                "housing_intersection": "forbidden on static nominal trunk envelope",
                "manufacturing_and_dynamic_clearance": (
                    "unresolved development assumption; not a deployment design"
                ),
            },
        },
        "orientation_library": {
            "mount_ids": list(SUPPLEMENTAL_MOUNT_IDS),
            "orientation_ids_in_tie_order": list(ORIENTATION_IDS),
            "pole_definitions": {
                "LEVEL": "sensor local +Z = body [0,0,+1]",
                "INVERTED": "sensor local +Z = body [0,0,-1]",
                "OUTWARD_DOWNWARD": (
                    "sensor local +Z = normalize(horizontal_outward_body_vector + [0,0,-1])"
                ),
                "INWARD_DOWNWARD": (
                    "sensor local +Z = normalize(-horizontal_outward_body_vector + [0,0,-1])"
                ),
            },
            "canonical_basis": {
                "sensor_z": "the exact normalized pole vector",
                "sensor_x": (
                    "normalize(body +X projected orthogonal to sensor_z); if degenerate, "
                    "normalize(body +Y projected orthogonal to sensor_z)"
                ),
                "sensor_y": "sensor_z cross sensor_x",
                "matrix_columns": "[sensor_x,sensor_y,sensor_z] in body coordinates",
                "quaternion": (
                    "canonical wxyz quaternion with unit norm and sign chosen by the first "
                    "nonzero w/x/y/z component being positive"
                ),
                "rpy": "canonical intrinsic XYZ roll/pitch/yaw in radians",
            },
            "static_selection": {
                "input": (
                    "static nominal URDF protected surfaces and frozen mount/housing geometry only"
                ),
                "labels_or_transition_outcomes_read": [],
                "maximize_in_order": [
                    "protected-surface direct visibility fraction",
                    "negative robot-self-occluded fraction among nominally FOV/range-eligible witnesses",
                    "calf protected-surface direct visibility fraction",
                    "rear-limb protected-surface direct visibility fraction",
                ],
                "self_occlusion_denominator": (
                    "that origin/orientation's nominally FOV-and-range-eligible static "
                    "protected-surface witnesses"
                ),
                "zero_nominal_eligible_witnesses": "fail closed",
                "final_tie_break": "orientation_ids_in_tie_order",
                "selected_numeric_binding": str(TRACKED_MOUNT_LIBRARY_PATH),
                "selected_numeric_binding_sha256": MOUNT_LIBRARY_RECEIPT_SHA256,
                "selected_numeric_binding_content_digest": (
                    "85683ac020d9fff0ec245ca9d2bf99a3942ae506f2c8c8bce7a49028a24392e6"
                ),
                "static_witness_count": 842,
                "static_witness_digest": (
                    "663de93b5cf840b7fe05788a1460702349094b24a882f7500c481e30d5a3f264"
                ),
                "frozen_selections": {
                    "HEAD_STOCK": {
                        "orientation_id": "STOCK",
                        "quaternion_body_wxyz": [
                            0.13131596830945733,
                            0.0,
                            0.9913405653290648,
                            0.0,
                        ],
                        "rpy_body_rad": [0.0, 2.8782, 0.0],
                    },
                    "REAR_TOP_TRUNK": {
                        "orientation_id": "INVERTED",
                        "quaternion_body_wxyz": [0.0, 1.0, 0.0, 0.0],
                        "rpy_body_rad": [math.pi, -0.0, 0.0],
                    },
                    "LEFT_UPPER_FLANK": {
                        "orientation_id": "INWARD_DOWNWARD",
                        "quaternion_body_wxyz": [
                            0.3826834323650898,
                            0.9238795325112867,
                            0.0,
                            0.0,
                        ],
                        "rpy_body_rad": [2.356194490192345, -0.0, 0.0],
                    },
                    "RIGHT_UPPER_FLANK": {
                        "orientation_id": "INWARD_DOWNWARD",
                        "quaternion_body_wxyz": [
                            0.3826834323650898,
                            -0.9238795325112867,
                            -0.0,
                            -0.0,
                        ],
                        "rpy_body_rad": [-2.356194490192345, -0.0, 0.0],
                    },
                },
                "must_be_frozen_before_transition_sensor_materialization": True,
            },
        },
        "layout_candidates": {
            "common_mount": "HEAD_STOCK",
            "pair_layout_ids_in_tie_order": list(PAIR_LAYOUT_IDS),
            "pair_layout_mounts": {
                "HEAD_STOCK__REAR_TOP_TRUNK": ["HEAD_STOCK", "REAR_TOP_TRUNK"],
                "HEAD_STOCK__LEFT_UPPER_FLANK": ["HEAD_STOCK", "LEFT_UPPER_FLANK"],
                "HEAD_STOCK__RIGHT_UPPER_FLANK": ["HEAD_STOCK", "RIGHT_UPPER_FLANK"],
            },
            "three_layout_ids_in_tie_order": list(THREE_LAYOUT_IDS),
            "three_layout_mounts": {
                "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK": [
                    "HEAD_STOCK",
                    "REAR_TOP_TRUNK",
                    "LEFT_UPPER_FLANK",
                ],
                "HEAD_STOCK__REAR_TOP_TRUNK__RIGHT_UPPER_FLANK": [
                    "HEAD_STOCK",
                    "REAR_TOP_TRUNK",
                    "RIGHT_UPPER_FLANK",
                ],
                "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK": [
                    "HEAD_STOCK",
                    "LEFT_UPPER_FLANK",
                    "RIGHT_UPPER_FLANK",
                ],
            },
            "all_four_diagnostic_layout": {
                "id": ALL_FOUR_LAYOUT_ID,
                "mounts": list(MOUNT_IDS),
                "primary_layout_candidate": False,
            },
            "candidate_counts": {"pair": 3, "three": 3, "all_four_diagnostic": 1},
            "layout_search_outside_list": "forbidden",
        },
        "layout_selection": {
            "role": "training",
            "state_count": 128,
            "geometry_reuse_contract": {
                "id": "LABEL_FREE_LAYOUT_GEOMETRY_REUSE_MAP",
                "partition": (
                    "within each deployable applied-action copy group only"
                ),
                "exact_fields": ["qpos", "link_transform", "geom_transform"],
                "boundary_snapshot_digest_must_match": True,
                "numeric_tolerance": 0.0,
                "outcome_fields_used_by_reuse_or_objective": [],
                "explicitly_forbidden_authority": (
                    "EXACT_SENSOR_MATERIALIZATION_MAP because its validation fields include "
                    "contact outcomes"
                ),
                "coverage": "every transition in all 128 training-role states exactly once",
            },
            "evidence": (
                "dense nominal-L2-FOV true-future raycast support generated without contact "
                "labels, safe-action labels, H3 outcomes, or calibration/held-out access"
            ),
            "selection_population": (
                "all training-role frozen transitions, 50 physics steps, 13 protected links"
            ),
            "support_unit": (
                "one protected-link/physics-step closest-environment witness; supported when at "
                "least one layout origin has valid direct environment support"
            ),
            "metrics": {
                "minimum_body_region_support": (
                    "minimum support fraction across exactly TRUNK, FRONT_LIMBS, REAR_LIMBS, "
                    "HIPS_AND_THIGHS, and CALVES"
                ),
                "p5_transition_support": (
                    "fifth percentile across per-transition support fractions over all 13x50 witnesses"
                ),
                "rear_limb_support": "support fraction over rear_limb witnesses",
                "calf_support": "support fraction over calf witnesses",
                "overall_mean_support": "support fraction over all witnesses",
                "self_occlusion": (
                    "among witnesses with at least one nominally FOV/range-eligible origin, the "
                    "fraction unsupported because every nominally eligible origin is self-blocked"
                ),
            },
            "body_region_ids": list(BODY_REGION_IDS),
            "zero_nominal_support_denominator": "fail closed; never encode as zero occlusion",
            "lexicographic_selection": [
                "highest minimum_body_region_support",
                "highest p5_transition_support",
                "highest rear_limb_support",
                "highest calf_support",
                "highest overall_mean_support",
                "lowest self_occlusion",
                "earlier fixed layout ID order",
            ],
            "pair_and_three_selection": "independent selection within each candidate cardinality",
            "outcome_fields_used_by_layout_objective": [],
            "contact_labels_used_for_layout_selection": False,
            "frozen_outcomes_read_only_for_corpus_custody_validation": True,
            "row_level_evidence": {
                "path": "layout_selection/training_layout_evidence.jsonl.gz",
                "population": (
                    "every transition in all 128 training-role states, one row with all six "
                    "frozen layout candidates"
                ),
                "metrics": (
                    "support/total, unsupported fraction, nominal eligibility, all-origin "
                    "self-occlusion, exact five-region support counts/fractions, and per-link "
                    "support counts/fractions for every one of the 13 protected links"
                ),
                "forbidden_outcomes_persisted_as_null": [
                    "contact_label",
                    "safe_action_count",
                    "route_outcome",
                ],
                "receipt_binding": "path, SHA-256, bytes, rows, role, layouts, regions, digest",
            },
            "candidate_per_link_reporting": (
                "candidate_metrics persists per_link_counts and per_link_support for all 13 "
                "frozen protected links over the complete training selection population"
            ),
            "selection_freeze": (
                "persist self-digesting selected pair/three receipt before calibration or "
                "development-held-out access"
            ),
        },
        "conditions": [
            _condition(
                "REALISTIC_PLATFORM_SCAN",
                stage="SINGLE_ORIGIN_REGRESSION",
                layout="HEAD_STOCK",
                representation="predecessor realistic platform scan",
                evidence_modes=full_modes,
                primary_eligible=False,
                source="byte-bound predecessor result/evidence reuse",
            ),
            _condition(
                "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
                stage="SINGLE_ORIGIN_REGRESSION",
                layout="HEAD_STOCK",
                representation="predecessor dense nominal-L2-FOV continuum",
                evidence_modes=full_modes,
                primary_eligible=False,
                source="byte-bound predecessor result/evidence reuse",
            ),
            _condition(
                "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
                stage="DUAL_REQUIRED",
                layout="selected pair layout",
                representation="dense ideal full-sphere continuum at both origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "DUAL_DENSE_L2_FOV_UPPER_BOUND",
                stage="DUAL_REQUIRED",
                layout="selected pair layout",
                representation="dense ideal nominal L2 FOV at both origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "DUAL_REALISTIC_L2_SCAN",
                stage="DUAL_REQUIRED",
                layout="selected pair layout",
                representation="independently phased realistic L2 scan at both origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
                stage="THREE_CONDITIONAL",
                layout="selected three-origin layout",
                representation="dense ideal full-sphere continuum at all three origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "THREE_DENSE_L2_FOV_UPPER_BOUND",
                stage="THREE_CONDITIONAL",
                layout="selected three-origin layout",
                representation="dense ideal nominal L2 FOV at all three origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "THREE_REALISTIC_L2_SCAN",
                stage="THREE_CONDITIONAL",
                layout="selected three-origin layout",
                representation="independently phased realistic L2 scan at all three origins",
                evidence_modes=full_modes,
                primary_eligible=True,
                source="new fused materialisation",
            ),
            _condition(
                "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC",
                stage="ALL_FOUR_DIAGNOSTIC_CONDITIONAL",
                layout=ALL_FOUR_LAYOUT_ID,
                representation="dense ideal full-sphere continuum at all four frozen origins",
                evidence_modes=["TRUE_FUTURE_OBSERVABILITY_CLOUD"],
                primary_eligible=False,
                source="new diagnostic-only fused materialisation",
            ),
        ],
        "evaluation_matrix": [
            *[
                {"condition_id": condition_id, "evidence_mode": evidence_mode}
                for condition_id in (*REGRESSION_CONDITION_IDS, *DUAL_CONDITION_IDS)
                for evidence_mode in EVIDENCE_MODE_IDS
            ],
            *[
                {
                    "condition_id": condition_id,
                    "evidence_mode": evidence_mode,
                    "conditional_on": "DUAL_REALISTIC_L2_SCAN true-future gate fails",
                }
                for condition_id in THREE_CONDITION_IDS
                for evidence_mode in EVIDENCE_MODE_IDS
            ],
            {
                "condition_id": "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC",
                "evidence_mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
                "conditional_on": (
                    "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND true-future gate fails"
                ),
            },
        ],
        "conditional_execution": {
            "order": [
                "validate/reuse both single-origin regressions",
                "freeze selected pair and three-origin layouts from training-only label-free support",
                "execute and fully evaluate all three selected-dual conditions in both modes",
                "if DUAL_REALISTIC_L2_SCAN true-future passes the complete gate, stop before every three-origin materialisation",
                "otherwise execute and fully evaluate all three selected-three conditions in both modes",
                "if THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND true-future fails, execute ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC true-future only",
            ],
            "dual_early_stop": {
                "condition": "DUAL_REALISTIC_L2_SCAN",
                "requires_complete_gate_pass": True,
                "effect": "three-origin and all-four conditions are NOT_RUN_BY_PROSPECTIVE_RULE",
            },
            "three_stage_required_when": (
                "DUAL_REALISTIC_L2_SCAN does not pass every immutable true-future gate"
            ),
            "all_four_diagnostic_required_when": (
                "three stage ran and THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND did not pass"
            ),
            "all_four_limits": {
                "mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD only",
                "representation": "dense spherical only",
                "realistic_four_origin_scan": "forbidden",
                "primary_classification_authority": False,
                "selected_pair_or_three_replacement": False,
            },
            "unexecuted_status": "NOT_RUN_BY_PROSPECTIVE_CONDITIONAL_RULE",
        },
        "evidence_modes": [
            {
                "id": "PLANNING_TIME_CAUSAL_CLOUD",
                "point_generation": "current planning-boundary acquisition only at every origin",
                "future_geometry_in_point_generation": False,
                "interpretation": "information available before candidate execution",
                "primary_gate": False,
            },
            {
                "id": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
                "point_generation": (
                    "actual timestamped acquisitions during the committed 100 ms transition"
                ),
                "future_geometry_in_point_generation": True,
                "interpretation": (
                    "representation-sufficiency upper bound; not planning-time prediction"
                ),
                "primary_gate": True,
            },
        ],
        "visibility_and_fusion": {
            "self_occlusion": (
                "exact 27-shape articulated robot geometry independently blocks each origin; "
                "rays never pass through robot links"
            ),
            "emitting_housing_ray_policy": {
                "own_emitter": (
                    "each origin's own 75x75x65 mm housing is ray-exempt as an optimistic "
                    "emitting-aperture approximation"
                ),
                "other_installed_supplemental_housings": (
                    "every other installed supplemental housing is a rigid body-axis-aligned "
                    "trunk-frame box self-occluder independent of optical orientation"
                ),
                "optical_mechanical_decoupling": (
                    "the selected optical ray frame rotates inside the frozen body-axis-aligned "
                    "75x75x65 mm mechanical/occlusion envelope approximation"
                ),
                "head_origin_robot_geom_exemption": [1, 2],
                "head_origin_rule": (
                    "HEAD_STOCK retains the predecessor exact emitting-host geom 1/2 exemption "
                    "only for rays emitted by HEAD_STOCK"
                ),
                "supplemental_origin_head_rule": (
                    "robot geom 1/2 remain ordinary self-occluders for every supplemental origin"
                ),
                "protected_geometry_effect": (
                    "sensor housings never alter the 13 protected links, 27 protected collision "
                    "shapes, contact labels, or oracle attribution"
                ),
                "mechanical_feasibility": (
                    "static mount checks always use each complete housing envelope"
                ),
                "limitation": (
                    "optimistic own-aperture plus conservative other-envelope approximation; "
                    "requires exact CAD/aperture confirmation and is not approved hardware"
                ),
                "outcome_tuning": False,
            },
            "self_returns": (
                "retain origin/link/shape identity for audit, exclude from environment clearance"
            ),
            "environment_occlusion": "first valid environment hit per origin and ray",
            "occluded_space": "unsupported, never free",
            "motion_compensation": "exact timestamped world-frame transformation before fusion",
            "fusion": (
                "deterministic set union of valid environment points from layout origins; no "
                "averaging may increase clearance or erase a nearer return"
            ),
            "nearest_supported_clearance": (
                "minimum valid per-origin supported environment clearance for each link/frame"
            ),
            "multi_origin_support": (
                "a protected swept-volume witness is supported when at least one origin validly "
                "observes it; unsupported only when no layout origin supports it"
            ),
            "provenance": (
                "persist every supporting origin and acquisition time plus the deterministic "
                "responsible origin/point selected for minimum clearance"
            ),
            "dense_spherical": {
                "horizontal_fov_deg": 360.0,
                "elevation_min_deg": -90.0,
                "elevation_max_deg": 90.0,
                "range_and_blind_region": "same 0.05-30 m L2 limits",
                "semantics": "analytic target-directed directional continuum, not a finite grid",
            },
            "dense_l2_fov": {
                "horizontal_fov_deg": 360.0,
                "elevation_min_deg": -6.0,
                "elevation_max_deg": 90.0,
                "range_and_blind_region": "same 0.05-30 m L2 limits",
                "semantics": "analytic target-directed directional continuum, not a finite grid",
            },
            "realistic_matched_dense_l2_fov_compatibility": {
                "per_origin_realistic_enrichment": (
                    "each realistic per-origin witness row also carries the label-free matched "
                    "dense-L2-FOV target nominal-FOV, direct-visibility, self-occlusion, "
                    "environment-occlusion, near-blind, and support fields"
                ),
                "same_object_witness_radius_m": 0.10,
                "inheritance_rule": (
                    "a valid finite realistic same-object return within 0.10 m of the protected "
                    "witness is inherited into that origin's matched dense-L2-FOV evidence"
                ),
                "dominance_assertion": (
                    "REALISTIC support must be a subset of DENSE_L2_FOV support separately for "
                    "every origin and after fused multi-origin union, in both evidence modes"
                ),
                "nested_dominance_assertion": (
                    "REALISTIC support subset DENSE_L2_FOV support subset DENSE_SPHERICAL "
                    "support must hold separately for every origin and after fusion"
                ),
                "clearance_monotonicity": (
                    "for each supported lower-representation query, matched upper-bound "
                    "clearance must be no greater than lower clearance plus 1e-9 m, separately "
                    "per origin and after fusion"
                ),
                "clearance_monotonicity_tolerance_m": 1e-9,
                "dense_spherical_compatibility": (
                    "matched dense spherical evidence inherits any supported dense-L2-FOV "
                    "witness needed to enforce the nominal-FOV-to-spherical continuum nesting"
                ),
                "decision_timing": (
                    "complete UID-group inheritance and both dominance assertions before any "
                    "threshold calibration, gate, conditional execution, or classification decision"
                ),
                "exact_geometry_copy_group_rule": (
                    "because realistic phase is transition-UID-bound while dense evidence uses "
                    "EXACT_SENSOR_MATERIALIZATION_MAP, the one dense row for an exact-geometry "
                    "copy group receives the union of finite inherited support from every member "
                    "transition UID before that enriched dense row is copied consistently to all "
                    "members"
                ),
                "provenance": (
                    "persist inherited support flags and counts on matched dense evidence; each "
                    "source realistic row retains its transition UID, mount, ray, time, range, "
                    "and responsible environment object provenance"
                ),
                "scientific_status": (
                    "predecessor witness-estimator compatibility rule; label-free and not "
                    "angular-grid tuning"
                ),
                "outcome_fields_used": [],
            },
        },
        "threshold_calibration": {
            "role": "internal_calibration",
            "scope": "one threshold independently per executed condition and evidence mode",
            "score": "continuous global minimum fused observed environment clearance in metres",
            "contact_rule": "score <= threshold is predicted contact; exact ties are contact-positive",
            "candidate_enumeration": (
                "every distinct finite calibration score plus the two exterior decision sentinels"
            ),
            "eligibility": {
                "combined_current_successor_contact_recall_gte": 0.95,
                "combined_current_successor_false_negative_rate_lte": 0.05,
            },
            "lexicographic_selection": [
                "highest contact-negative transition retention",
                "highest oracle-viable states retaining a derived viability-admissible action",
                "highest correct abstention count on oracle-nonviable states",
                "highest selected H3 route progress",
                "lowest normalized viability-constrained regret",
                "highest best-admissible top-3",
                "more conservative threshold (numerically larger clearance threshold)",
            ],
            "freeze_before_heldout": True,
            "layout_mount_orientation_phase_selection_from_outcomes": "forbidden",
        },
        "two_ply_viability": {
            "admission_rule": (
                "admit only when the current tick is predicted contact-free and its actual "
                "successor retains at least one predicted contact-free unique deployable action"
            ),
            "safe_next_action_margins": [1, 2, 3],
            "actual_successor_use": "observability assay only; not a pre-action prediction claim",
        },
        "development_heldout_metrics": {
            "contact": [
                "AUC",
                "average precision",
                "recall",
                "FNR",
                "negative retention",
                "per-link",
                "per-family",
            ],
            "safe_action_count": [
                "MAE",
                "Spearman",
                "exact count accuracy",
                "zero/nonzero accuracy",
                "false-zero rate",
                "false-nonzero rate",
            ],
            "viability": [
                "oracle-viable states retaining action",
                "selected immediate contacts",
                "selected oracle-nonviable successors",
                "false abstentions",
                "H3 route progress",
                "exact-geometry progress fraction",
                "normalized regret",
                "best-admissible top-1/top-3",
                "oracle-nonviable abstentions",
                "unsafe movement decisions",
                "falsely viable candidates",
            ],
            "coverage": [
                "per-origin support",
                "fused support",
                "nominal FOV inclusion",
                "direct visibility",
                "self-occlusion",
                "unsupported swept-volume fraction",
                "finite realistic support inherited into matched dense evidence",
                "per-origin and fused support-dominance query/violation counts",
                "per-link",
                "per-region",
                "per-family",
            ],
        },
        "immutable_gate": {
            "mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD only",
            "current_contact_auc_gte": 0.90,
            "successor_contact_auc_gte": 0.90,
            "combined_contact_recall_gte": 0.95,
            "combined_contact_false_negative_rate_lte": 0.05,
            "safe_action_zero_nonzero_accuracy_gte": 0.90,
            "safe_action_false_nonzero_rate_lte": 0.05,
            "oracle_viable_denominator": 20,
            "oracle_viable_states_retaining_action_gte": 18,
            "oracle_nonviable_denominator": 4,
            "oracle_nonviable_correct_abstentions_eq": 4,
            "selected_immediate_contacts_eq": 0,
            "selected_nonviable_successors_eq": 0,
            "h3_route_progress_fraction_of_exact_geometry_gte": 0.80,
            "normalized_viability_constrained_regret_lte": 0.20,
            "best_admissible_top_3_gte": 0.75,
            "no_family_collapse": {"required": True},
            "weakening": "forbidden",
        },
        "coverage_attribution": {
            "error_classes": list(COVERAGE_ERROR_CLASSES),
            "mutually_exclusive_hierarchy": [
                {
                    "class": "POINT_FUSION_ERROR",
                    "evidence": (
                        "valid supporting per-origin evidence exists but deterministic union or "
                        "nearest-clearance fusion loses or changes it"
                    ),
                },
                {
                    "class": "NEAR_BLIND_REGION",
                    "evidence": (
                        "relevant closest witness is inside 0.05 m for every otherwise eligible origin"
                    ),
                },
                {
                    "class": "ROBOT_SELF_OCCLUSION",
                    "evidence": (
                        "every nominally FOV/range-eligible selected-layout origin is blocked first "
                        "by robot geometry"
                    ),
                },
                {
                    "class": "VERTICAL_FOV_LIMITATION",
                    "evidence": (
                        "matched-layout dense spherical supports while dense nominal L2 FOV does not"
                    ),
                },
                {
                    "class": "SCAN_TIMING_LIMITATION",
                    "evidence": (
                        "an existing realistic ray provides spatial support but not at or before "
                        "the witness event time"
                    ),
                },
                {
                    "class": "SCAN_PATTERN_SPARSITY",
                    "evidence": (
                        "dense nominal-L2-FOV supports but no realistic ray provides spatial support"
                    ),
                },
                {
                    "class": "INSUFFICIENT_ORIGIN_COUNT",
                    "evidence": (
                        "a larger frozen allowed layout, including the conditional all-four dense "
                        "diagnostic, supports where the selected lower-cardinality layout does not"
                    ),
                },
                {
                    "class": "MOUNT_POSITION_LIMITATION",
                    "evidence": (
                        "geometric nonvisibility remains after the preceding evidence tests"
                    ),
                },
                {
                    "class": "UNRESOLVED",
                    "evidence": "no unique preceding class is supported",
                },
            ],
            "ambiguity_rule": (
                "if evidence is ambiguous at any causal comparison level, classify UNRESOLVED "
                "rather than forcing the earlier priority"
            ),
            "population": (
                "every frozen-threshold contact error and every decision-level coverage failure"
            ),
            "required_provenance": [
                "state and transition/candidate identity",
                "protected link/body region",
                "contact/minimum-clearance physics step",
                "layout and all origin mount IDs",
                "per-origin nominal FOV and self-occlusion",
                "per-origin scan-point availability and nearest ray/point",
                "per-origin point age and acquisition time",
                "supporting origins",
                "exact and fused sensor-derived clearance",
            ],
        },
        "classifications": {
            "primary_exactly_one": list(PRIMARY_CLASSIFICATIONS),
            "precedence": [
                {
                    "when": "DUAL_REALISTIC_L2_SCAN true-future passes",
                    "classification": "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL",
                },
                {
                    "when": (
                        "dual realistic fails and THREE_REALISTIC_L2_SCAN true-future passes"
                    ),
                    "classification": "THREE_ORIGIN_REALISTIC_RANGE_SIGNAL",
                },
                {
                    "when": (
                        "no realistic condition passes and either selected dual dense spherical "
                        "or selected dual dense L2-FOV true-future passes"
                    ),
                    "classification": (
                        "DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
                    ),
                },
                {
                    "when": (
                        "no realistic or dual dense condition passes and either selected three "
                        "dense spherical or selected three dense L2-FOV true-future passes"
                    ),
                    "classification": (
                        "THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
                    ),
                },
                {
                    "when": "no selected dual or three-origin dense upper bound passes",
                    "classification": "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
                },
            ],
            "secondary": list(SECONDARY_CLASSIFICATIONS),
            "all_four_secondary_rule": (
                "add FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED only when the diagnostic ran "
                "because selected three-origin dense spherical failed and the all-four dense "
                "spherical true-future diagnostic passes; it never replaces the primary"
            ),
            "assumed_sensor_secondary_required": "ASSUMED_SENSOR_CONTRACT",
            "replanning_interface": "REPLANNING_INTERFACE_UNRESOLVED",
        },
        "compute_benchmark": {
            "target": "strongest executed selected dual or three-origin condition",
            "device": "CPU",
            "numeric_dtype": "float32",
            "timed_inputs": (
                "already materialized float32 physics-step/protected-link clearance and "
                "observation-support witnesses; raw ray clouds are not timed inputs"
            ),
            "scope": (
                "deterministically reduce physics-step/protected-link witnesses into the "
                "structured per-link state and transition contact decision, then evaluate all "
                "current actions and next-action sets, safe-action counts, two-ply admission, "
                "threshold decisions, and H3 route selection"
            ),
            "ray_generation_included": False,
            "future_trajectory_acquisition_included": False,
            "population": (
                "representative development-held-out states including every frozen family; "
                "each timed sample evaluates each selected state's complete candidate set"
            ),
            "warmups": 30,
            "timed_iterations_gte": 1000,
            "report": [
                "P50",
                "P90",
                "P95",
                "P99",
                "maximum",
                "misses at 50/80/100 ms",
                "peak RSS",
                "peak VRAM",
            ],
            "classification": {
                "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL": (
                    "P99 <= 50 ms and maximum <= 80 ms"
                ),
                "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY": (
                    "not SIGNAL, P99 <= 80 ms and maximum <= 100 ms"
                ),
                "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO": "otherwise",
            },
            "replanning_interface": "REPLANNING_INTERFACE_UNRESOLVED",
        },
        "fixtures": {
            "required": [
                "clear full-body sweep",
                "front trunk contact",
                "side trunk contact",
                "rear trunk contact",
                "front-limb contact",
                "rear-limb contact",
                "calf contact",
                "one origin occluded while another observes",
                "complementary left/right flank coverage",
                "complementary head/rear coverage",
                "contact within near-blind region",
                "contact between scan samples",
                "synchronized scan overlap",
                "independent phased scans",
                "one safe successor",
                "zero safe successors",
                "exact threshold tie",
                "correct abstention",
                "deterministic H3 route ranking",
                "byte-identical regeneration",
            ],
            "requirements": [
                "deterministic mount basis/quaternion/RPY",
                "deterministic ray generation and independent phases",
                "deterministic self/environment occlusion",
                "deterministic point fusion and provenance",
                "deterministic per-link reduction",
                "deterministic layout selection",
                "byte-identical receipt regeneration",
            ],
            "raw_evidence": {
                "location": "fixture.core.raw_fixture_evidence",
                "schema": (
                    "minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1"
                ),
                "serialization": "embedded canonical JSON with a self content digest",
                "complete_raw_ray_queries": [
                    "near_blind",
                    "between_scan_samples",
                    "one_origin_occluded_another_observes",
                ],
                "complete_reduced_origin_evidence": [
                    "one_origin_occluded_another_observes",
                    "complementary_left_right_flank",
                    "complementary_head_rear",
                    "synchronized_scan_overlap",
                ],
                "reconstructible_noncloud_inputs": (
                    "complete primitive/query inputs and outputs for clear/contact/H3/safe/"
                    "threshold-tie/scan-phase fixtures"
                ),
                "raw_ray_completeness": (
                    "every finite ray used by a ray fixture retains its origin, direction, "
                    "timestamp, environment and robot primitives, physical first hit, and "
                    "range-filter status"
                ),
                "regeneration": (
                    "the complete tracked fixture receipt, including embedded raw evidence, "
                    "must regenerate byte-identically at the fixture gate"
                ),
                "separate_raw_artifact": False,
            },
            "failure_policy": "fail closed before scientific materialisation",
        },
        "raw_audit_subset": {
            "scientific_selection": (
                "single lowest SHA-256-ranked transition within each frozen "
                "(role,family,transition_kind) stratum, plus deterministic fixtures"
            ),
            "selection_depends_on_sensor_results": False,
            "retain": (
                "complete raw per-origin clouds, fused cloud, phase/mount/layout provenance"
            ),
            "other_raw_ray_or_point_persistence": "forbidden",
        },
        "storage": {
            "capacity_unit": "decimal GB = 1,000,000,000 bytes",
            "workspace_root": "/home/andrewknowles/Workspace/LeWMQuad-v3",
            "output_root": output_root,
            "cache_root": f"{output_root}/cache",
            "intermediate_root": f"{output_root}/intermediate",
            "evidence_root": f"{output_root}/evidence",
            "raw_audit_root": f"{output_root}/raw_audit",
            "receipt_root": f"{output_root}/receipts",
            "output_filesystem_required_type": "ext4",
            "output_must_be_different_device_from_workspace": True,
            "minimum_output_free_bytes": 100_000_000_000,
            "minimum_workspace_free_bytes": 20_000_000_000,
            "temporary_storage_ceiling_bytes": 50_000_000_000,
            "final_storage_ceiling_bytes": 25_000_000_000,
            "evaluation_mode": "streaming",
            "raw_point_policy": "fixtures and frozen audit subset only",
            "preflight_rule": (
                "stop before materialisation unless device identity, filesystem type, free-space "
                "floors, and predicted temporary/final byte ceilings all pass"
            ),
            "workspace_large_cache_policy": "forbidden",
        },
        "source_and_environment": {
            "expected_entrypoint": (
                "scripts/evaluate_minimum_multi_origin_body_range_coverage_qualification_v1.py"
            ),
            "contract_module": (
                "lewm/safety/minimum_multi_origin_body_range_coverage_qualification_v1_contract.py"
            ),
            "tracked_preregistration": str(TRACKED_PREREGISTRATION_PATH),
            "tracked_contract_receipt": str(TRACKED_CONTRACT_RECEIPT_PATH),
            "tracked_output_schema": str(TRACKED_OUTPUT_SCHEMA_PATH),
            "tracked_mount_library": str(TRACKED_MOUNT_LIBRARY_PATH),
            "tracked_source_closure": str(TRACKED_SOURCE_CLOSURE_PATH),
            "tracked_fixture": str(TRACKED_FIXTURE_PATH),
            "tracked_result": str(TRACKED_RESULT_PATH),
            "tracked_report": str(TRACKED_REPORT_PATH),
            "source_closure_timing": (
                "persist before the first training-transition layout selection or scientific "
                "sensor result is calculated"
            ),
            "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
            "known_runtime": {
                "classification": "EXISTING_ENVIRONMENT_REUSABLE",
                "environment_name": "genesis_render_vulkan",
                "compatibility_path": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan"
                ),
                "resolved_prefix": (
                    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan"
                ),
                "python": "3.12.3",
                "genesis": "0.3.14",
                "numpy": "2.4.6",
                "scipy": "1.17.1",
                "cpu_only": True,
                "tinyquadjepa_required": False,
                "training_packages_required": False,
                "model_checkpoints_required": False,
            },
            "environment_policy": (
                "use predecessor-compatible environment; do not install or upgrade opportunistically"
            ),
            "import_closure_receipt_required": True,
            "experiment_import_closure_required": (
                "runtime receipt must bind the new contract, core, metrics, predecessor "
                "evaluator, new evaluator, scripts namespace, functools and zipfile standard-"
                "library imports, and all first-party source SHA-256 values"
            ),
        },
        "commit_sequence": [
            "implement contract, static mount receipt, source closure, evaluator, and fixtures",
            "commit: Freeze minimum multi-origin body range coverage qualification",
            "execute conditional no-training qualification",
            "persist row evidence, machine-readable result, and Markdown report",
            "commit: Evaluate minimum multi-origin body range coverage qualification",
        ],
        "stop_condition": (
            "stop after conditional materialisation/evaluation, attribution, hardware accounting, "
            "benchmark, result persistence, and result commit"
        ),
        "prohibitions": [
            "model training",
            "fresh panel or corpus collection",
            "reading or opening untouched G2 evaluation",
            "opening or executing the JEPA predictor",
            "opening model checkpoints",
            "changing frozen state identities or role membership",
            "changing transition identities or repaired transitions",
            "changing action identities or the action bank",
            "altering contact labels or attribution authority",
            "changing the 13 protected links or 27 protected collision shapes",
            "using contact, safe-action, H3, calibration, or held-out outcomes for mount orientation or layout selection",
            "searching mounts, orientations, pairs, or triples outside the frozen candidate sets",
            "retraining recurrent memory",
            "training a successor or per-link predictor",
            "learned closed-loop navigation",
            "implementing memory, novelty, routing, or beacon capture",
            "restarting or reinterpreting completed predecessor experiments",
            "executing a realistic four-origin scan condition",
            "allowing the all-four diagnostic to replace the selected pair/three result or primary classification",
            "persisting non-audit raw scientific point clouds",
        ],
    }


_CONTRACT_CORE = _contract_core()
CONTRACT_SHA256 = canonical_json_sha256(_CONTRACT_CORE)
CONTRACT: dict[str, Any] = {
    **copy.deepcopy(_CONTRACT_CORE),
    "contract_sha256": CONTRACT_SHA256,
}


def build_contract() -> dict[str, Any]:
    return {**copy.deepcopy(_CONTRACT_CORE), "contract_sha256": CONTRACT_SHA256}


def contract_receipt() -> dict[str, Any]:
    return build_contract()


def build_output_schema() -> dict[str, Any]:
    return {
        **copy.deepcopy(_OUTPUT_SCHEMA_CORE),
        "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
    }


def contract_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_contract()) + b"\n"


def output_schema_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_output_schema()) + b"\n"


CONTRACT_RECEIPT_SHA256 = hashlib.sha256(contract_receipt_bytes()).hexdigest()
OUTPUT_SCHEMA_RECEIPT_SHA256 = hashlib.sha256(output_schema_receipt_bytes()).hexdigest()


def _validate_exact_receipt(
    value: Mapping[str, Any],
    *,
    digest_key: str,
    expected_core: dict[str, Any],
    expected_digest: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    candidate = copy.deepcopy(dict(value))
    if set(candidate) != set(expected_core) | {digest_key}:
        raise ContractError(f"{label} top-level keys do not match the frozen schema")
    declared = candidate.pop(digest_key)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ContractError(f"{label} {digest_key} must be a 64-character SHA-256")
    actual = canonical_json_sha256(candidate)
    if declared != actual:
        raise ContractError(f"{label} content SHA-256 mismatch")
    if declared != expected_digest or candidate != expected_core:
        raise ContractError(f"{label} does not exactly match prospectively frozen content")
    return {**candidate, digest_key: declared}


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if len(MOUNT_LIBRARY_RECEIPT_SHA256) != 64:
        raise ContractError("mount-library receipt binding has not been prospectively frozen")
    return _validate_exact_receipt(
        value,
        digest_key="contract_sha256",
        expected_core=_CONTRACT_CORE,
        expected_digest=CONTRACT_SHA256,
        label="contract",
    )


def validate_output_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_exact_receipt(
        value,
        digest_key="output_schema_sha256",
        expected_core=_OUTPUT_SCHEMA_CORE,
        expected_digest=OUTPUT_SCHEMA_SHA256,
        label="output schema",
    )


def _write_immutable(path: Path, payload: bytes, label: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite non-identical {label}: {path}")
        return path
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_contract(path: str | Path = TRACKED_CONTRACT_RECEIPT_PATH) -> Path:
    if len(MOUNT_LIBRARY_RECEIPT_SHA256) != 64:
        raise ContractError("refusing to write before mount-library receipt binding is frozen")
    return _write_immutable(Path(path), contract_receipt_bytes(), "contract receipt")


def write_output_schema(path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH) -> Path:
    return _write_immutable(
        Path(path), output_schema_receipt_bytes(), "output schema receipt"
    )


def _load_canonical_receipt(path: Path, expected_bytes: bytes, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if raw != expected_bytes:
        raise ContractError(f"{label} is not the byte-identical frozen canonical receipt")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ContractError(f"{label} JSON root must be an object")
    return decoded


def load_and_validate_contract(
    path: str | Path = TRACKED_CONTRACT_RECEIPT_PATH,
) -> dict[str, Any]:
    decoded = _load_canonical_receipt(
        Path(path), contract_receipt_bytes(), "contract receipt"
    )
    return validate_contract(decoded)


def load_and_validate_output_schema(
    path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH,
) -> dict[str, Any]:
    decoded = _load_canonical_receipt(
        Path(path), output_schema_receipt_bytes(), "output schema receipt"
    )
    return validate_output_schema(decoded)


def load_and_validate_mount_library(
    path: str | Path = TRACKED_MOUNT_LIBRARY_PATH,
) -> dict[str, Any]:
    """Load the separately generated static receipt and enforce its frozen binding."""

    if len(MOUNT_LIBRARY_RECEIPT_SHA256) != 64:
        raise ContractError("mount-library receipt binding has not been prospectively frozen")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MOUNT_LIBRARY_RECEIPT_SHA256:
        raise ContractError("mount-library receipt file SHA-256 mismatch")
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("mount-library receipt is not valid UTF-8 JSON") from exc
    if not isinstance(receipt, dict):
        raise ContractError("mount-library receipt JSON root must be an object")
    if receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("mount-library receipt experiment mismatch")
    if receipt.get("pass") is not True:
        raise ContractError("mount-library receipt did not pass")
    if receipt.get("outcome_fields_read") != [] or receipt.get("contact_outcomes_used") is not False:
        raise ContractError("mount-library selection is not outcome-independent")
    if receipt.get("housing_occlusion_frame") != (
        "BODY_AXIS_ALIGNED_TRUNK_FRAME_INDEPENDENT_OF_OPTICAL_RAY_FRAME"
    ):
        raise ContractError("mount-library housing occlusion frame mismatch")
    clearance_rows = receipt.get("housing_clearance_validation")
    if not isinstance(clearance_rows, list) or tuple(
        row.get("mount_id") for row in clearance_rows if isinstance(row, dict)
    ) != MOUNT_IDS:
        raise ContractError("mount-library housing-clearance mount IDs mismatch")
    if any(not isinstance(row, dict) or row.get("pass") is not True for row in clearance_rows):
        raise ContractError("mount-library full-housing clearance validation failed")
    for row in clearance_rows[1:]:
        if (
            row.get("protected_primitive_count") != 27
            or row.get("required_clearance_m") != 0.01
            or row.get("rule")
            != "BODY_AXIS_ALIGNED_FULL_HOUSING_OUTSIDE_ALL_27_NOMINAL_PROTECTED_PRIMITIVES"
        ):
            raise ContractError("supplemental full-housing validation contract mismatch")
    selected = receipt.get("orientation_selections")
    if not isinstance(selected, list) or tuple(
        item.get("mount_id") for item in selected if isinstance(item, dict)
    ) != MOUNT_IDS:
        raise ContractError("mount-library receipt selected mount IDs mismatch")
    for row in selected:
        if not isinstance(row, dict):
            raise ContractError("mount-library orientation selection must be an object")
        if not {"mount_id", "selected_orientation_id", "selected_pose", "candidates", "selection_rule"}.issubset(row):
            raise ContractError("mount-library orientation selection is incomplete")
        pose = row["selected_pose"]
        if not isinstance(pose, dict) or not {
            "mount_id",
            "orientation_id",
            "parent_link",
            "pole_body_xyz",
            "translation_body_xyz_m",
            "quaternion_body_wxyz",
            "rpy_body_rad",
        }.issubset(pose):
            raise ContractError("mount-library selected pose is incomplete")
        if pose["mount_id"] != row["mount_id"] or pose["orientation_id"] != row[
            "selected_orientation_id"
        ]:
            raise ContractError("mount-library selected pose identity mismatch")
    declared = receipt.get("content_digest")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ContractError("mount-library receipt content_digest is invalid")
    payload = copy.deepcopy(receipt)
    payload.pop("content_digest")
    if canonical_json_sha256(payload) != declared:
        raise ContractError("mount-library receipt content digest mismatch")
    return receipt


__all__ = [
    "ALL_FOUR_LAYOUT_ID",
    "BODY_REGION_IDS",
    "COMPUTE_CLASSIFICATIONS",
    "CONDITION_IDS",
    "CONTRACT",
    "CONTRACT_RECEIPT_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_SHA256",
    "COVERAGE_ERROR_CLASSES",
    "ContractError",
    "DIAGNOSTIC_CONDITION_IDS",
    "DUAL_CONDITION_IDS",
    "EVIDENCE_MODE_IDS",
    "EXPERIMENT_ID",
    "MOUNT_IDS",
    "MOUNT_LIBRARY_RECEIPT_SHA256",
    "ORIENTATION_IDS",
    "OUTPUT_ROOT",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_RECEIPT_SHA256",
    "OUTPUT_SCHEMA_SHA256",
    "OUTPUT_SCHEMA_VERSION",
    "PAIR_LAYOUT_IDS",
    "PHASE_NAMESPACE",
    "PRIMARY_CLASSIFICATIONS",
    "REGRESSION_CONDITION_IDS",
    "SECONDARY_CLASSIFICATIONS",
    "SUPPLEMENTAL_MOUNT_IDS",
    "THREE_CONDITION_IDS",
    "THREE_LAYOUT_IDS",
    "TRACKED_CONTRACT_RECEIPT_PATH",
    "TRACKED_FIXTURE_PATH",
    "TRACKED_MOUNT_LIBRARY_PATH",
    "TRACKED_OUTPUT_SCHEMA_PATH",
    "TRACKED_PREREGISTRATION_PATH",
    "TRACKED_REPORT_PATH",
    "TRACKED_RESULT_PATH",
    "TRACKED_SOURCE_CLOSURE_PATH",
    "build_contract",
    "build_output_schema",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "contract_receipt",
    "contract_receipt_bytes",
    "derive_multi_origin_scan_phases",
    "load_and_validate_contract",
    "load_and_validate_mount_library",
    "load_and_validate_output_schema",
    "output_schema_receipt_bytes",
    "validate_contract",
    "validate_output_schema",
    "write_contract",
    "write_output_schema",
]
