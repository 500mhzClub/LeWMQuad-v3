"""Frozen contract for BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1.

This module is deliberately payload-free.  Importing it does not inspect the
transition corpus, the Git repository, Genesis, a GPU, or an output directory.
It only exposes the prospectively frozen experiment contract, output schema,
canonical serialization helpers, and fail-closed receipt writers/validators.

The experiment is a deterministic, no-training sensor-observability assay.  A
true-future cloud is an observability upper bound and is never evidence that a
planning-time model can predict future geometry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1"
CONTRACT_SCHEMA_VERSION = "body_centric_range_coverage_qualification_v1.contract.v1"
OUTPUT_SCHEMA_VERSION = "body_centric_range_coverage_qualification_v1.output.v1"
L2_PHASE_NAMESPACE = "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1/L2_PHASE_V1"

CONDITION_IDS = (
    "CURRENT_SPARSE_RANGE_BASELINE",
    "REALISTIC_PLATFORM_SCAN",
    "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
    "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
)
EVIDENCE_MODE_IDS = (
    "PLANNING_TIME_CAUSAL_CLOUD",
    "TRUE_FUTURE_OBSERVABILITY_CLOUD",
)

PRIMARY_CLASSIFICATIONS = (
    "PLATFORM_RANGE_COVERAGE_SIGNAL",
    "SCAN_DENSITY_OR_TIMING_BOTTLENECK",
    "BODY_CENTRIC_MOUNT_REQUIRED",
    "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
    "RANGE_SENSOR_CONTRACT_UNRESOLVED",
)
SECONDARY_CLASSIFICATIONS = (
    "FOUR_CHANNEL_RANGE_BASELINE_INSUFFICIENT",
    "FRONT_LIMB_VISIBILITY_FAILURE",
    "REAR_LIMB_VISIBILITY_FAILURE",
    "CALF_VISIBILITY_FAILURE",
    "TRUNK_VISIBILITY_FAILURE",
    "ROBOT_BODY_SELF_OCCLUSION",
    "PLANNING_TIME_RANGE_OBSERVABILITY_LIMITATION",
    "ASSUMED_SENSOR_CONTRACT",
)
COVERAGE_ERROR_CLASSES = (
    "SCAN_PATTERN_SPARSITY",
    "SCAN_TIMING_LIMITATION",
    "VERTICAL_COVERAGE_LIMITATION",
    "HORIZONTAL_COVERAGE_LIMITATION",
    "NEAR_BLIND_REGION",
    "ROBOT_SELF_OCCLUSION",
    "PLATFORM_MOUNT_LIMITATION",
    "SINGLE_ORIGIN_LIMITATION",
    "POINT_ACCUMULATION_ERROR",
    "UNRESOLVED",
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "body_centric_range_coverage_qualification_v1"
)
TRACKED_CONTRACT_RECEIPT_PATH = Path(
    "docs/lewm_go2_body_centric_range_coverage_qualification_v1_contract_2026-08-25.json"
)
TRACKED_OUTPUT_SCHEMA_PATH = Path(
    "docs/lewm_go2_body_centric_range_coverage_qualification_v1_output_schema_2026-08-25.json"
)
TRACKED_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_go2_body_centric_range_coverage_qualification_v1_source_closure_2026-08-25.json"
)


class ContractError(ValueError):
    """Raised when a contract or schema fails closed validation."""


def _validate_json_value(value: Any, location: str = "$") -> None:
    """Reject values whose JSON rendering is ambiguous or non-finite."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
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


def derive_l2_scan_phases(
    *,
    boundary_snapshot_digest: str,
) -> dict[str, Any]:
    """Derive the exact planning-boundary-keyed L2 phases.

    The input is the lowercase SHA-256 hex digest of the exact embodied and
    environment snapshot at the planning boundary.  It deliberately excludes
    candidate action identity and successor endpoint, so every candidate at
    one boundary receives the same causal scan phase.  Returning the phase
    digest with both IEEE-754 phase values makes receipts auditable.
    """

    if (
        not isinstance(boundary_snapshot_digest, str)
        or len(boundary_snapshot_digest) != 64
        or any(character not in "0123456789abcdef" for character in boundary_snapshot_digest)
    ):
        raise ContractError("boundary_snapshot_digest must be a lowercase 64-hex SHA-256")
    boundary_digest_bytes = bytes.fromhex(boundary_snapshot_digest)
    raw_digest = hashlib.sha256(
        L2_PHASE_NAMESPACE.encode("utf-8") + b"\x00" + boundary_digest_bytes
    ).digest()
    return {
        "phase_namespace": L2_PHASE_NAMESPACE,
        "boundary_snapshot_digest": boundary_snapshot_digest,
        "phase_digest_sha256": raw_digest.hex(),
        "horizontal_phase_cycles": int.from_bytes(raw_digest[0:8], "big") / 2**64,
        "vertical_phase_cycles": int.from_bytes(raw_digest[8:16], "big") / 2**64,
    }


_OUTPUT_SCHEMA_CORE: dict[str, Any] = {
    "schema_version": OUTPUT_SCHEMA_VERSION,
    "experiment_id": EXPERIMENT_ID,
    "serialization": {
        "canonical_receipts": "UTF-8, sorted keys, compact separators, finite numbers, LF terminated",
        "operational_and_result_json": "UTF-8, sorted keys, two-space indentation, finite numbers, LF terminated",
        "jsonl": "one canonical finite JSON object followed by LF per row",
        "missing_numeric_value": None,
        "infinity_policy": "encode unsupported/unbounded values as null with an explicit status",
    },
    "files": {
        "preexecution_receipt": {
            "relative_path": "preexecution_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "head", "contract_sha256", "source_closure_sha256", "inputs",
                "environment", "platform_emitting_housing_smoke",
                "workspace_filesystem", "output_filesystem", "pass"
            ],
        },
        "environment_receipt": {
            "relative_path": "receipts/environment_receipt.json",
            "cardinality": 1,
            "required_keys": [
                "python", "packages", "package_sources", "import_closure",
                "environment_prefix_resolved", "scipy_ckdtree_smoke", "torch_loaded",
                "environment_classification", "recovery_environment_receipt",
                "pip_freeze_receipt",
                "content_digest"
            ],
        },
        "materialization_index": {
            "relative_path": "materialization_index.json",
            "cardinality": 1,
            "required_keys": [
                "head", "contract_sha256", "source_closure_sha256", "states",
                "transitions", "representatives", "records", "storage_bytes", "status",
                "scan_materialization_counts", "materialization_workers",
                "process_start_method", "numeric_thread_environment"
            ],
        },
        "state_evidence": {
            "relative_path": "states/<state_id>.npz and states/<state_id>.json",
            "cardinality": "176 NPZ shards and 176 bound state receipts",
            "required_keys": [
                "state_id", "role", "family", "transitions", "representatives",
                "shard_path", "shard_sha256", "contract_sha256", "source_closure_sha256"
            ],
        },
        "result": {
            "relative_path": "result.json",
            "cardinality": 1,
            "required_keys": [
                "schema_version",
                "experiment_id",
                "contract_sha256",
                "source_closure_sha256",
                "environment_receipt",
                "storage",
                "materialisation_counts",
                "fixture_results",
                "calibration_thresholds",
                "condition_metrics",
                "sparse_baseline_regression",
                "planning_time_true_future_comparison",
                "coverage_error_counts",
                "compute_benchmark",
                "primary_classification",
                "secondary_classifications",
                "custody",
                "result_content_sha256",
            ],
        },
        "transition_evidence": {
            "relative_path": "evidence/transition_evidence.jsonl",
            "cardinality": "one row per frozen transition x condition x evidence_mode",
            "required_keys": [
                "transition_uid",
                "state_id",
                "transition_index",
                "transition_level",
                "current_action_index",
                "action_index",
                "successor_state_id",
                "applied_action_id",
                "boundary_snapshot_digest",
                "phase_digest_sha256",
                "role",
                "family",
                "condition_id",
                "evidence_mode",
                "oracle_contact",
                "sensor_global_minimum_clearance_m",
                "unsupported_risk",
                "threshold_m",
                "predicted_contact",
                "successor_set_available",
                "decision_role_scored",
                "predicted_safe_next_action_count",
                "oracle_safe_next_action_count",
                "admitted",
                "per_link_row_count",
            ],
            "current_row_only_nullable_keys": [
                "successor_state_id", "predicted_safe_next_action_count",
                "oracle_safe_next_action_count", "admitted"
            ],
            "decision_scope": (
                "state-decision fields are populated only for development-held-out current rows; "
                "counts are null when no frozen successor set exists"
            ),
        },
        "per_link_evidence": {
            "relative_path": "evidence/per_link_evidence.jsonl",
            "cardinality": (
                "one row per frozen transition x condition x evidence_mode x protected link"
            ),
            "required_keys": [
                "transition_uid",
                "condition_id",
                "evidence_mode",
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
                "nominal_fov_inclusion",
                "direct_visibility_after_self_occlusion",
                "point_support_count",
                "finite_scan_support_inherited_fraction",
                "responsible_ray_or_point",
                "responsible_point_age_s",
                "responsible_point_range_m",
                "responsible_acquisition_index",
                "support_ray_or_point",
                "support_point_age_s",
                "support_point_range_m",
                "support_acquisition_index",
            ],
        },
        "coverage_errors": {
            "relative_path": "evidence/coverage_errors.jsonl",
            "cardinality": "one row per error attribution",
            "required_keys": [
                "transition_uid",
                "state_id",
                "transition_level",
                "current_action_index",
                "action_index",
                "candidate_action_id",
                "condition_id",
                "evidence_mode",
                "robot_link_or_body_region",
                "contact_or_minimum_clearance_physics_step",
                "nominal_fov_inclusion",
                "self_occlusion",
                "self_occluder_geom_index",
                "self_occluder_link",
                "scan_point_availability",
                "nearest_ray_or_point",
                "point_age_s",
                "nearest_point_range_m",
                "acquisition_index",
                "exact_clearance_m",
                "sensor_derived_clearance_m",
                "error_class",
            ],
        },
        "raw_audit_manifest": {
            "relative_path": "raw_audit/manifest.jsonl",
            "cardinality": "only fixtures plus the frozen hash-ranked audit subset",
            "required_keys": [
                "transition_uid",
                "role",
                "family",
                "transition_kind",
                "source_representative_transition_uid",
                "boundary_snapshot_digest",
                "pattern_identity",
                "phase_digest_sha256",
                "condition_id",
                "evidence_mode",
                "selection_sha256",
                "representation_kind",
                "artifact_relative_path",
                "artifact_sha256",
                "point_or_witness_count",
            ],
        },
        "report": {
            "relative_path": "report.md",
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


def _contract_core() -> dict[str, Any]:
    output_root = str(OUTPUT_ROOT)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_frozen_utc_date": "2026-08-25",
        "execution_class": "NO_TRAINING_DEVELOPMENT_MODE_END_TO_END_EXECUTION",
        "scientific_question": (
            "Can a physically plausible body-relevant range-sensor configuration observe "
            "enough geometry to reproduce the exact per-link two-ply micro-viability decision?"
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
            "ray_phase_source": (
                "deterministic exact planning-boundary snapshot SHA-256; shared by all candidates"
            ),
        },
        "claim_boundary": {
            "target": "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT",
            "positive_claim": (
                "simulated robot-environment contact/separation proxy over the committed one-tick horizon"
            ),
            "true_future_limitation": (
                "actual transition geometry is an observability upper bound and does not establish "
                "pre-action prediction"
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
        "recovery_lineage": {
            "prior_terminal_classification": "NOT_RUN_FAIL_CLOSED_NO_FROZEN_EXECUTABLE_CONTRACT",
            "scope": (
                "Recovery completed successfully. The proposed body-centric sensor qualification "
                "did not run because no executable experiment contract existed. This contract "
                "prospectively defines a new scientific experiment after recovery."
            ),
            "final_recovery_receipt": "/home/andrewknowles/recovery/REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1/final_recovery_receipt.json",
            "final_recovery_receipt_sha256": "1046c214b8b67560d0604b69ab438795c051c19b7b248c0d254c18d17769f2ec",
            "scientific_execution_gate_receipt": "/home/andrewknowles/recovery/REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1/scientific_execution_gate_receipt.json",
            "scientific_execution_gate_receipt_sha256": "64f3c64f262fba4f2370aeb89e0c82ca1c49ecab2b0e0834062c0cbf6f840388",
        },
        "predecessor_bindings": {
            "experiment": "EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1",
            "source_lineage_commit": "10b3a190d506830e6a87e04a0f1c832b92295bd7",
            "completed_result_commit": "034c2fb902997ac29e2742fc4ddc2c28ad1706b6",
            "corpus_logical_digest": "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223",
            "corpus_index_sha256": "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0",
            "geometry_index_sha256": "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f",
            "role_split_sha256": "eb2b41ca3ca4d4f7d2d2fc41495944e306e39798ede8865dd0904fa6c3d88021",
            "action_contract_sha256": "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06",
            "repaired_row_ledger_sha256": "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94",
            "frozen_state_count": 176,
            "frozen_current_and_successor_transition_count": 29470,
            "frozen_current_transition_count": 2464,
            "frozen_successor_transition_count": 27006,
            "frozen_physics_frame_count": 1473500,
            "role_state_counts": {"training": 128, "internal_calibration": 24, "development_held_out": 24},
            "oracle_label_array": "frozen_contact_label",
            "contact_authority": (
                "repaired frozen_contact_label bound by the repaired row ledger; this is the "
                "predecessor's authoritative physics-rate H1 target and is never replaced by "
                "native-replay or history-free exact-query reconstruction"
            ),
            "contact_attribution_authority": (
                "use the frozen first-contact physics step; link/object attribution is native-replay "
                "only when both its verdict and first-contact step agree with the frozen label, otherwise "
                "history-free exact-query attribution only when both its verdict and first-contact step "
                "agree, otherwise UNRESOLVED; reconstruction verdicts "
                "never alter the frozen target"
            ),
            "action_authority": "unique deployable applied-action contract",
            "route_authority": "deterministic H3 route scores",
            "development_heldout_exact_unique_action_h3_progress_m": 4.2499809517354254,
            "mutable_fields": [],
        },
        "roles": {
            "training": {
                "source_role_key": "development_training_state_ids",
                "use": "reporting and materialisation only; never threshold or mount selection",
            },
            "internal_calibration": {
                "source_role_key": "internal_calibration_state_ids",
                "use": "the only threshold-calibration role",
            },
            "development_held_out": {
                "source_role_key": "development_heldout_state_ids",
                "use": "evaluation only after all thresholds are frozen",
            },
            "untouched_g2": "FORBIDDEN_NOT_READ",
        },
        "timebase": {
            "committed_tick_s": 0.1,
            "physics_step_s": 0.002,
            "physics_steps_per_tick": 50,
            "interval_convention": "half-open [0.0, 0.1) for sampled rays; terminal state at 0.1",
        },
        "hardware_binding": {
            "binding_class": "ASSUMED_GO2_HEAD_LIDAR_L2",
            "secondary_classification_required": "ASSUMED_SENSOR_CONTRACT",
            "reason": (
                "local BOM/platform documents do not select an exact range sensor; this is an "
                "explicit development assumption, not a final deployment selection"
            ),
            "sensor": "Unitree 4D LiDAR L2",
            "local_bom_audit": (
                "no exact intended range sensor is selected in the local Go2 platform manifest, "
                "hardware/BOM material, or prior RGB-only progression contract. The pinned, "
                "pending-license/build-audit third-party Gazebo xacro includes a generic L1 visual "
                "and synthetic gpu_lidar scaffold, but sensors_required does not select it, it was "
                "not captured, and the README explicitly drops lidar from the v3 corpus"
            ),
            "local_selection_authorities": {
                "platform_manifest": "config/go2_platform_manifest.yaml",
                "platform_manifest_sha256": "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
                "repository_readme": "README.md",
                "repository_readme_sha256": "cf70176d240038f653d64d00a380d4c9c0511f6c701c570258b3bc416923f945",
            },
            "excluded_local_l1_scaffold": {
                "robot_xacro": "third_party/unitree_go2_ros2/unitree_go2_description/urdf/unitree_go2_robot.xacro",
                "robot_xacro_sha256": "be0abf5e860427bfb8762c85b58232c9ff466a3bc8f522915f8ca71e3b8c05e9",
                "lidar_xacro": "third_party/unitree_go2_ros2/unitree_go2_description/urdf/lidar_4D_lidar.xacro",
                "lidar_xacro_sha256": "3e8239745a52a3f8f9240f754a8dbd09133d15aef634a1032496edc877e60f9d",
                "declared_device": "Lidar L1 synthetic Gazebo gpu_lidar",
                "declared_mount_xyz_m": [0.25, -0.038, -0.03],
                "declared_mount_rpy_rad": [2.879, 0.0, 1.5705],
                "declared_scan": "600x30 at 10 Hz, 0.8-30 m, Gaussian noise",
                "exclusion_reason": (
                    "upstream simulation scaffold on a pinned_pending_license_and_build_audit "
                    "backend; not present in platform sensors_required and explicitly dropped from corpus scope"
                ),
            },
            "confidence": {
                "official_sensor_specification": "HIGH_PRIMARY_UNITREE_DOCUMENTATION",
                "wide_profile_elevation_interval": "MEDIUM_DOCUMENTED_GEOMETRY_INFERENCE",
                "go2_physical_sensor_selection": "LOW_EXPLICIT_DEVELOPMENT_ASSUMPTION",
                "nominal_urdf_mount": "HIGH_MATCHED_LOCAL_AND_OFFICIAL_URDF",
                "physical_mount_calibration": "UNRESOLVED",
            },
            "selected_profile": "wide_negative_angle_360_x_96",
            "profile_default_claim": "NOT_CLAIMED",
            "fov": {
                "horizontal_deg": 360.0,
                "vertical_deg": 96.0,
                "elevation_min_deg": -6.0,
                "elevation_max_deg": 90.0,
                "elevation_interval_basis": (
                    "inference from the documented 96 degree wide/negative-angle mode and "
                    "documented negative-angle extent"
                ),
            },
            "rates": {
                "raw_sampling_per_s": 128000,
                "effective_points_per_s": 64000,
                "circumferential_hz": 5.55,
                "vertical_hz": 216.0,
                "imu_sampling_hz": 1000,
                "imu_reporting_hz": 500,
            },
            "range": {
                "near_blind_region_m": 0.05,
                "maximum_m_at_90_percent_reflectivity": 30.0,
                "maximum_m_at_10_percent_reflectivity": 15.0,
                "qualification_ideal_ray_maximum_m": 30.0,
                "accuracy_m_lte": 0.02,
                "distance_resolution_m": 0.0045,
            },
            "scan_mechanism": "non-repetitive brushless rotating-mirror scan",
            "interfaces": {
                "transport": ["100BASE-TX Ethernet UDP", "3.3V TTL UART"],
                "standalone_point_cloud": {
                    "topic": "unilidar/cloud",
                    "message": "sensor_msgs/msg/PointCloud2",
                    "frame": "unilidar_lidar",
                },
                "standalone_imu": {
                    "topic": "unilidar/imu",
                    "message": "sensor_msgs/msg/Imu",
                    "frame": "unilidar_imu",
                },
                "go2_point_cloud": {
                    "topic": "utlidar/cloud",
                    "message": "sensor_msgs/msg/PointCloud2",
                    "frame": "utlidar_lidar",
                },
                "point_fields": ["x", "y", "z", "intensity", "ring", "time"],
            },
            "coordinate_frame": {
                "origin": "centre of the bottom mounting surface",
                "x_axis": "opposite the cable outlet",
                "y_axis": "counter-clockwise 90 degrees from +X viewed from +Z",
                "z_axis": "outward normal of the bottom mounting surface",
                "handedness": "right-handed",
                "imu_axes": "parallel to LiDAR axes",
                "imu_origin_in_lidar_m": [-0.007698, -0.014655, 0.00667],
            },
            "timestamps": {
                "cloud_stamp": "start time of the point cloud",
                "per_point_time_field": "seconds relative to cloud start",
                "system_mode": "system wall time minus one scan period",
                "hardware_mode": "packet hardware timestamp",
                "qualification": "preserve a deterministic timestamp for every ray/point",
            },
            "official_sources": [
                {
                    "kind": "Go2 product page",
                    "url": "https://www.unitree.com/go2/",
                },
                {
                    "kind": "L2 product page",
                    "url": "https://www.unitree.com/L2/",
                },
                {
                    "kind": "L2 user manual v1.1 (2024-10)",
                    "url": "https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf",
                    "sha256": "95a3e52ce5fa1cc9366095d646e6c6ff23de0fd23eddb1fff7bf5a698cbe76b1",
                    "bytes": 995957,
                },
                {
                    "kind": "Unitree unilidar_sdk2",
                    "url": "https://github.com/unitreerobotics/unilidar_sdk2/tree/0e3c51f512e6b8ff60b8c32f160b412cb48445c2",
                    "commit": "0e3c51f512e6b8ff60b8c32f160b412cb48445c2",
                    "release": "v2.0.10",
                    "readme_sha256": "fa05a0ffbcfd7370714a2f15e9501f49de9748e05f4af419990b0b6e1ffe15aa",
                    "utilities_sha256": "3360eb798f95e08cc2153547081bcd86efe9595f9c7cae753e70ffa9f1341e4a",
                    "ros2_header_sha256": "bf7dc99381ed981a58a34b654534c6569d026908b158ef32fc12f2afbc2c5bad",
                },
            ],
        },
        "mounts": {
            "sparse_baseline": {
                "parent_link": "base",
                "translation_m": [0.0, 0.0, 0.25],
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
                "source": "completed synthetic sparse range condition",
                "tuning": "frozen predecessor value; no outcome tuning",
            },
            "platform": {
                "parent_link": "base",
                "child_frame": "radar",
                "translation_m": [0.28945, 0.0, -0.046825],
                "rotation_rpy_rad": [0.0, 2.8782, 0.0],
                "coordinate_convention": "URDF fixed-joint parent-to-child origin, RPY radians",
                "ray_frame_mapping": (
                    "the deterministic scan-pattern FRU axes are interpreted directly as the URDF "
                    "radar child-frame +X forward, +Y left, +Z up axes before applying the fixed-joint rotation"
                ),
                "source": "stock Unitree Go2 base-to-radar fixed joint",
                "local_genesis_urdf_sha256": "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4",
                "official_urdf_commit": "4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48",
                "official_urdf_sha256": "7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e",
                "mounting_uncertainty": "physical calibration and manufacturing tolerance unquantified",
                "tuning": "selected before contact outcomes; no mount search",
                "emitting_housing_ray_policy": {
                    "classification": "OPTIMISTIC_RAY_ONLY_COARSE_HOUSING_EXEMPTION",
                    "exempt_frozen_geom_indices": [1, 2],
                    "exempt_frozen_geom_identities": ["base:01", "base:02"],
                    "frozen_parent_link": "base",
                    "urdf_collision_lineage_names": ["Head_upper", "Head_lower"],
                    "reason": (
                        "the nominal radar optical origin lies inside both coarse head-housing "
                        "collision primitives; those coarse solids do not encode the optical aperture"
                    ),
                    "scope": (
                        "exclude only geometry indices 1 and 2 from B/C self-ray intersection; "
                        "retain both unchanged as protected contact and clearance geometry; every "
                        "other articulated collision primitive remains a self-occluder"
                    ),
                    "limitation": (
                        "whole coarse housing primitives are omitted rather than a CAD-resolved aperture, "
                        "so B/C remain optimistic physical-coverage upper bounds"
                    ),
                    "outcome_tuning": False,
                },
            },
            "body_centric": {
                "parent_link": "base",
                "translation_m": [0.0, 0.0, 0.067],
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
                "coordinate_convention": "level in body/base frame",
                "selection_rule": "top-centre of trunk collision-envelope bounding box plus clearance",
                "trunk_collision_envelope_size_m": [0.3762, 0.0935, 0.114],
                "trunk_top_z_m": 0.057,
                "mechanical_clearance_m": 0.01,
                "mounting_uncertainty": "development assumption; physical installation not calibrated",
                "tuning": "single prospectively selected mount; no contact-label or outcome tuning",
            },
        },
        "realistic_scan_approximation": {
            "classification": "APPROXIMATED_REALISTIC_PLATFORM_SCAN",
            "reason": "the proprietary exact non-repetitive ray sequence is unavailable",
            "effective_ray_rate_hz": 64000,
            "raw_sampling_rate_hz_documented_only": 128000,
            "transition_ray_count": 6400,
            "sample_index": "k = 0,...,6399",
            "timestamp_law": "t_k = (k + 0.5) / 64000 seconds in the 100 ms window",
            "phase_namespace": L2_PHASE_NAMESPACE,
            "boundary_snapshot_binding": (
                "lowercase SHA-256 hex digest of the exact planning-boundary embodied and "
                "environment snapshot already bound by the frozen predecessor geometry"
            ),
            "phase_digest": (
                "SHA-256(namespace UTF-8 || 0x00 || raw 32-byte boundary snapshot SHA-256)"
            ),
            "azimuth_phase": "big-endian uint64 digest[0:8] / 2^64 turns",
            "elevation_phase": "big-endian uint64 digest[8:16] / 2^64 cycles",
            "azimuth_law": "-pi + 2*pi*frac(azimuth_phase + 5.55*t)",
            "elevation_law": (
                "u=frac(elevation_phase+216*t); triangle=1-4*abs(u-0.5); "
                "elevation_deg=-6+48*(triangle+1)"
            ),
            "phase_continuity": (
                "planning window t in [-0.1,0) and true-future window t in [0,0.1) "
                "share the same boundary-keyed phase at t=0"
            ),
            "candidate_invariance": (
                "every candidate action evaluated at the same planning boundary uses the same phase; "
                "candidate identity and successor endpoint are excluded from phase derivation"
            ),
            "successor_boundary_rule": (
                "a successor transition row uses the exact snapshot digest of that successor/current "
                "planning boundary, shared across all next-action candidates from it"
            ),
            "randomness": "none; phase hashing is deterministic and outcome-independent",
            "binding_api": (
                "derive_l2_scan_phases(boundary_snapshot_digest=...)"
            ),
            "range_noise": "none",
            "random_dropout": "none",
            "interpretation": "optimistic physical-coverage upper bound",
        },
        "evidence_modes": [
            {
                "id": "PLANNING_TIME_CAUSAL_CLOUD",
                "information_boundary": "only points available at the current planning boundary",
                "sparse_timing": (
                    "use t in [-0.1,0); because predecessor prehistory is not in the frozen "
                    "corpus, hold exact current articulated geometry constant over that trailing window"
                ),
                "dense_timing": "continuum visibility of exact current geometry at t=0 only",
                "motion_note": (
                    "zero-order current-state hold is an explicit optimistic causal observation approximation. "
                    "Offline scoring still queries that causal cloud against the frozen actual 50-step protected "
                    "sweep and labels, solely to measure whether current evidence covers the later body volume; "
                    "future geometry never enters point generation and is not deployable planning-time information"
                ),
                "gate_status": "reported separately; not the primary representation-sufficiency gate",
            },
            {
                "id": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
                "information_boundary": (
                    "points generated or received during the actual committed 100 ms transition"
                ),
                "sparse_timing": "use t in [0,0.1) at the frozen deterministic ray timestamps",
                "dense_timing": (
                    "accumulated union of target-directed continuum visibility at t=0 and all "
                    "50 exact post-step endpoints t=0.002,...,0.100 seconds; full support is any "
                    "acquisition, while event-time support is separately restricted to acquisitions "
                    "no later than the protected witness time"
                ),
                "motion_note": (
                    "uses actual current-to-successor geometry as a true-future observability upper bound"
                ),
                "gate_status": "primary gate mode",
                "prediction_disclaimer": (
                    "does not establish that future geometry is available or predictable before execution"
                ),
            },
        ],
        "evaluation_matrix": [
            {"condition_id": condition_id, "evidence_mode": evidence_mode}
            for condition_id in CONDITION_IDS
            for evidence_mode in EVIDENCE_MODE_IDS
        ],
        "conditions": [
            {
                "id": "CURRENT_SPARSE_RANGE_BASELINE",
                "ordinal": "A",
                "mount": "sparse_baseline",
                "horizontal_coverage_deg": 360.0,
                "azimuth_bins": 180,
                "azimuth_step_deg": 2.0,
                "vertical_channels_deg": [-15.0, -5.0, 5.0, 15.0],
                "minimum_range_m": 0.05,
                "maximum_range_m": 10.0,
                "scan_timing": (
                    "planning-time uses one instantaneous boundary sweep at t=0; true-future "
                    "uses the deterministic union of instantaneous sweeps at t=0 and the "
                    "actual terminal boundary t=0.1, reproducing the predecessor boundary-union contract"
                ),
                "pose_semantics": (
                    "reproduce the predecessor yaw-only body pose for both baseline mount and rays; "
                    "roll and pitch are deliberately stripped only for condition A"
                ),
                "ray_semantics": "finite ideal first-hit rays",
                "baseline_source": "scripts/materialize_geometry_modality_safety_sufficiency_v1.py",
                "baseline_source_sha256": "a8d6a3e7f7fbfd86273a15f706f27ab69db40c2baebcb630c90feea1d96348e5",
                "purpose": "regression baseline",
            },
            {
                "id": "REALISTIC_PLATFORM_SCAN",
                "ordinal": "B",
                "mount": "platform",
                "sensor_binding": "ASSUMED_GO2_HEAD_LIDAR_L2",
                "scan_pattern": "realistic_scan_approximation",
                "horizontal_coverage_deg": 360.0,
                "elevation_min_deg": -6.0,
                "elevation_max_deg": 90.0,
                "minimum_range_m": 0.05,
                "maximum_range_m": 30.0,
                "accumulation_s": 0.1,
                "ray_semantics": "finite timestamped ideal first-hit rays",
                "purpose": "optimistic physical-coverage upper bound",
            },
            {
                "id": "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
                "ordinal": "C",
                "mount": "platform",
                "horizontal_coverage_deg": 360.0,
                "elevation_min_deg": -6.0,
                "elevation_max_deg": 90.0,
                "minimum_range_m": 0.05,
                "maximum_range_m": 30.0,
                "ray_semantics": "mathematical directional continuum with exact first-hit visibility",
                "removes": ["scan sparsity", "scan timing"],
                "preserves": ["platform mount", "nominal FOV", "range", "blind region", "self occlusion"],
            },
            {
                "id": "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
                "ordinal": "D",
                "mount": "body_centric",
                "horizontal_coverage_deg": 360.0,
                "elevation_min_deg": -90.0,
                "elevation_max_deg": 90.0,
                "minimum_range_m": 0.05,
                "maximum_range_m": 30.0,
                "ray_semantics": "full-sphere mathematical directional continuum with exact first-hit visibility",
                "mount_search_count": 1,
                "purpose": "strongest single physically plausible body-origin observability upper bound",
            },
        ],
        "visibility_and_reduction": {
            "environment_occlusion": "nearest environment or robot collision-geometry intersection wins",
            "robot_self_occlusion": (
                "exact articulated collision geometry blocks rays except the prospectively frozen "
                "platform emitting-housing ray-only exemption for geom indices 1 and 2 in B/C"
            ),
            "equal_distance_tie": "robot self-return wins conservatively",
            "self_return_identity": "retained for coverage auditing",
            "self_return_clearance_use": "excluded from environment clearance",
            "ground_return_handling": (
                "the analytic ground plane participates as a first-hit occluder and its return count/identity "
                "is retained, but ground points are excluded from disallowed-environment clearance and local "
                "target support. This reproduces the predecessor sparse baseline's post-render removal of "
                "ground-plane points at z <= 0.025 m while retaining their occlusion effect"
            ),
            "occluded_sector_semantics": "unsupported, never free",
            "near_blind_semantics": (
                "intersections inside minimum range remain occluders but yield no valid environment range point"
            ),
            "point_validity": (
                "finite first environment return within inclusive [minimum_range, maximum_range], "
                "not robot-self, with a valid timestamp and direction"
            ),
            "point_age_definition": (
                "signed event-relative age = protected-witness physics time minus return timestamp; "
                "negative values explicitly mean the nearest supporting return arrived after that event"
            ),
            "local_target_support_radius_m": 0.1,
            "local_target_support_rule": (
                "sparse local point support requires an environment return from the responsible "
                "object within 0.10 m Euclidean distance of the closest-surface target witness; "
                "selected prospectively without outcome data"
            ),
            "local_target_support_scope": (
                "support and attribution only; continuous clearance remains the exact "
                "point-to-protected-primitive reduction"
            ),
            "swept_volume_support_estimator": (
                "for each protected link, use the 50 equal-weight exact-physics-step closest-environment "
                "surface witnesses along the frozen articulated sweep; unsupported_swept_volume_fraction "
                "is the fraction of those 50 witnesses without condition-valid support. This is a "
                "prospectively frozen witness-weighted swept-volume coverage estimator, not an exact "
                "Lebesgue-volume measurement"
            ),
            "transition_unsupported_rule": (
                "a transition is unsupported risk when any protected-link physics-step closest-surface "
                "witness lacks condition-valid support; this full protected-sweep rule is outcome-blind "
                "and may not be reduced to the oracle contact or minimum-clearance link after outcomes"
            ),
            "unsupported_contact_rule": (
                "an unsupported transition is predicted contact/risk independently of the finite "
                "clearance threshold; unsupported protected volume is never treated as free"
            ),
            "motion_compensation": (
                "transform each valid return using the exact articulated/body pose at its timestamp "
                "into the immutable world frame and reduce against the world-frame articulated sweep; "
                "this is rigid-transform equivalent to moving both cloud and sweep into the current-boundary "
                "body reference. Between adjacent frozen 2 ms physics samples use linear position "
                "interpolation and shortest-arc normalized linear quaternion interpolation; never "
                "extrapolate beyond the frozen trace"
            ),
            "planning_motion_compensation": (
                "current geometry is held fixed over its causal trailing observation window; the resulting cloud "
                "is retrospectively reduced against the actual frozen candidate sweep as an evaluation target"
            ),
            "dense_semantics": (
                "analytic target-directed visibility, not a tunable finite angular grid. The exact, "
                "outcome-independent 50x13 closest-environment surface witnesses determine the frozen "
                "witness-weighted coverage and clearance metric; they are conservative sufficient queries, "
                "not a complete visible-surface cloud and not proof that hidden geometry could not be inferred"
            ),
            "dense_support_provenance": (
                "select the latest supporting acquisition at or before the witness event; when none "
                "exists select the earliest later supporting acquisition; signed point age is event "
                "time minus selected acquisition time; deterministic acquisition index breaks all ties"
            ),
            "realistic_dense_platform_dominance": (
                "condition C support is the union of exact target-directed continuum support and any "
                "valid condition B same-object return within the frozen 0.10 m witness radius. Such a "
                "B ray necessarily exists in C's identical mount/FOV/range continuum. B support must "
                "therefore imply C support for every witness; inherited finite-ray provenance is flagged"
            ),
            "inherited_support_provenance": (
                "finite_scan_support_inherited=true binds the B ray index/time/range; dense acquisition "
                "indices are -1/null because no discrete continuum acquisition produced that inherited ray"
            ),
            "global_minimum_use": (
                "a global minimum may only aggregate deterministic per-link state for contact filtering"
            ),
            "per_link_required_fields": [
                "minimum_observed_environment_clearance_m",
                "time_to_minimum_clearance_s",
                "first_threshold_crossing_time_s",
                "obstacle_direction_body_rad",
                "obstacle_sector",
                "observation_support",
                "unsupported_swept_volume_fraction",
                "responsible_environment_object",
                "oracle_contact_link",
                "nominal_fov_inclusion",
                "direct_visibility_after_self_occlusion",
                "point_support_count",
            ],
            "obstacle_sector_rule": {
                "direction_frame": "current physics-step body/base FRU frame",
                "angle": "atan2(left, forward) in radians, wrapped to [-pi, pi)",
                "sectors": [
                    "FRONT", "FRONT_LEFT", "LEFT", "REAR_LEFT",
                    "REAR", "REAR_RIGHT", "RIGHT", "FRONT_RIGHT"
                ],
                "bin_width_rad": "pi/4",
                "tie_rule": "add pi/8 then floor modulo eight; exact upper ties enter the next CCW sector",
            },
        },
        "fixtures": {
            "required": [
                "clear full-body sweep",
                "front trunk contact",
                "side trunk contact",
                "front-limb contact",
                "rear-limb contact",
                "calf contact",
                "contact inside the near blind region",
                "contact hidden by robot self-occlusion",
                "contact between scan samples",
                "current-state visible geometry",
                "future-only visible geometry",
                "one safe successor action",
                "zero safe successor actions",
                "exact threshold tie",
                "correct abstention",
                "deterministic H3 route selection",
            ],
            "requirements": [
                "deterministic ray generation",
                "deterministic point timestamps",
                "deterministic occlusion",
                "deterministic per-link reduction",
                "byte-identical receipt regeneration",
            ],
            "gate": "all fixtures must pass before the first scientific sensor result is calculated",
        },
        "threshold_calibration": {
            "role": "internal_calibration",
            "scope": "one threshold independently for each condition and evidence mode",
            "score": "continuous global minimum observed environment clearance in metres",
            "candidate_enumeration": (
                "every distinct finite calibration score plus the two exterior decision sentinels"
            ),
            "contact_rule": "score <= threshold is predicted contact; exact ties are contact-positive",
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
            "heldout_isolation": (
                "development-held-out outcomes may not select thresholds, mounts, scan laws, or conditions"
            ),
        },
        "two_ply_viability": {
            "current_action_rule": [
                "derive immediate contact from sensor-derived per-link state",
                "use the frozen actual successor state",
                "derive contact for every unique deployable next action",
                "count predicted contact-free next actions",
            ],
            "admission_rule": (
                "admit only if the current tick is predicted contact-free and the actual successor "
                "has at least one predicted contact-free deployable next action"
            ),
            "safe_action_count_metric_denominator": (
                "every unique current candidate with its frozen successor action set, including "
                "candidates whose current tick has oracle contact"
            ),
            "safe_next_action_margins": [1, 2, 3],
        },
        "development_heldout_metrics": {
            "current_and_successor_contact": [
                "auc",
                "average_precision",
                "recall",
                "false_negative_rate",
                "negative_retention",
                "per_link",
                "per_family",
            ],
            "safe_action_count": [
                "mae",
                "spearman",
                "exact_count_accuracy",
                "zero_nonzero_accuracy",
                "false_zero_rate",
                "false_nonzero_rate",
            ],
            "oracle_viable_states": [
                "states retaining an admitted action",
                "selected immediate contacts",
                "selected oracle-nonviable successors",
                "false abstentions",
                "H3 route progress",
                "exact-geometry progress fraction",
                "normalized viability-constrained regret",
                "best-admissible top-1",
                "best-admissible top-3",
            ],
            "oracle_nonviable_states": [
                "correct abstentions",
                "unsafe movement decisions",
                "falsely viable candidates",
            ],
        },
        "immutable_gate": {
            "mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD only",
            "current_contact_auc_gte": 0.9,
            "successor_contact_auc_gte": 0.9,
            "combined_contact_recall_gte": 0.95,
            "combined_contact_false_negative_rate_lte": 0.05,
            "safe_action_zero_nonzero_accuracy_gte": 0.9,
            "safe_action_false_nonzero_rate_lte": 0.05,
            "oracle_viable_states_retaining_action_gte": 18,
            "oracle_viable_state_denominator": 20,
            "oracle_nonviable_correct_abstentions_eq": 4,
            "oracle_nonviable_state_denominator": 4,
            "selected_immediate_contacts_eq": 0,
            "selected_nonviable_successors_eq": 0,
            "h3_route_progress_fraction_of_exact_geometry_gte": 0.8,
            "normalized_viability_constrained_regret_lte": 0.2,
            "best_admissible_top_3_gte": 0.75,
            "no_family_collapse": {
                "required": True,
                "definition": [
                    "each family with a positive contact has combined current/successor recall >= 0.80",
                    "each family with both contact classes has combined AUC >= 0.75",
                    "each family with oracle-viable states retains an action for at least one such state",
                    "every oracle-nonviable state in every family abstains",
                ],
            },
            "weakening": "forbidden",
        },
        "coverage_attribution": {
            "error_classes": list(COVERAGE_ERROR_CLASSES),
            "error_population": (
                "every development-held-out transition whose frozen-threshold contact prediction "
                "differs from the repaired frozen_contact_label, for every condition and evidence mode; "
                "decision-level unsafe selections and abstentions remain separately complete in viability metrics"
            ),
            "sparsity_timing_counterfactual": (
                "for the realistic condition, spatial support ignores emission time but preserves the exact "
                "boundary-keyed set of 6400 sensor-frame ray directions: at the target physics-step platform "
                "pose, a ray supports the closest-surface witness when its forward ray line passes within "
                "the frozen 0.10 m support radius and condition C proves direct target visibility. Temporal "
                "support additionally requires an actually timestamped accumulated B return within 0.10 m "
                "from the responsible object. Spatial absence is SCAN_PATTERN_SPARSITY; spatial presence "
                "with temporal absence is SCAN_TIMING_LIMITATION"
            ),
            "required_fields": [
                "state_id",
                "candidate_action_id",
                "robot_link_or_body_region",
                "contact_or_minimum_clearance_physics_step",
                "nominal_fov_inclusion",
                "self_occlusion",
                "scan_point_availability",
                "nearest_ray_or_point",
                "point_age_s",
                "exact_clearance_m",
                "sensor_derived_clearance_m",
                "error_class",
            ],
            "classification_rule": "exactly one primary error class per error; UNRESOLVED is fail-closed",
            "error_witness_rule": {
                "false_positive_unsupported": (
                    "select the unsupported step/link with lowest exact clearance, then earliest step/link"
                ),
                "false_positive_finite_threshold": "select the sensor-derived global-minimum step/link",
                "false_negative": (
                    "select the authoritative frozen first-contact step and agreeing attributed link; "
                    "if the link is unresolved, retain UNRESOLVED and use the minimum-clearance link only "
                    "to index diagnostic arrays"
                ),
            },
            "condition_specific_hierarchy": (
                "scan pattern/timing counterfactuals apply only to REALISTIC_PLATFORM_SCAN; "
                "A uses its own blind/FOV/self/direct evidence, C compares platform to D, and D "
                "uses body blind/self/single-origin evidence"
            ),
            "temporal_support_rule": (
                "a realistic finite return supports an event only when its timestamp is no later "
                "than that physics-step event; later full-window returns remain available to the "
                "upper-bound cloud but are not temporal-support evidence"
            ),
            "point_accumulation_error_policy": (
                "report only with an explicit raw-return-versus-accumulated-cloud loss witness; "
                "otherwise do not infer this class from a whole-transition prediction mismatch"
            ),
        },
        "compute_benchmark": {
            "condition": "DENSE_BODY_CENTRIC_SINGLE_ORIGIN true-future, the prospectively strongest condition",
            "decision_unit": (
                "the complete 24-state development-held-out set per timed sample; each sample "
                "includes every current action and every associated next-action set"
            ),
            "warmups": 30,
            "timed_samples": 1000,
            "includes": [
                "deterministic per-link reduction",
                "all current-action decisions",
                "all next-action-set safe counts",
                "threshold decisions",
                "H3 route selection",
            ],
            "excludes": ["ray generation", "future trajectory acquisition"],
            "latency_statistics_ms": ["p50", "p90", "p95", "p99", "maximum"],
            "deadlines_ms": [50, 80, 100],
            "memory_statistics": ["peak_rss_bytes", "peak_vram_bytes"],
        },
        "storage": {
            "workspace_root": "/home/andrewknowles/Workspace/LeWMQuad-v3",
            "output_root": output_root,
            "cache_root": f"{output_root}/cache",
            "intermediate_root": f"{output_root}/intermediate",
            "evidence_root": f"{output_root}/evidence",
            "raw_audit_root": f"{output_root}/raw_audit",
            "receipt_root": f"{output_root}/receipts",
            "output_filesystem_required_type": "ext4",
            "output_must_be_different_device_from_workspace": True,
            "capacity_unit": "decimal GB = 1,000,000,000 bytes",
            "minimum_output_free_bytes": 40_000_000_000,
            "minimum_workspace_free_bytes": 20_000_000_000,
            "temporary_storage_ceiling_bytes": 30_000_000_000,
            "final_storage_ceiling_bytes": 20_000_000_000,
            "evaluation_mode": "streaming",
            "preflight_rule": (
                "stop before materialisation unless device identity, filesystem type, free-space "
                "floors, and predicted temporary/final byte ceilings all pass"
            ),
            "workspace_large_cache_policy": "forbidden",
            "materialization_parallelism": {
                "process_workers": 12,
                "process_start_method": "spawn",
                "numeric_threads_per_process": 1,
                "state_isolation": "one frozen state per task; distinct output paths",
                "index_order": "canonical frozen state order independent of completion order",
            },
        },
        "raw_audit_subset": {
            "fixtures": (
                "retain the canonical complete synthetic fixture inputs, exact first-hit/per-link outputs, "
                "and full ray-pattern parameter/digest receipts in the tracked fixture artifact; fixture "
                "point sets are fixed inline constants and need not be duplicated as binary clouds"
            ),
            "scientific_selection": (
                "within each (role, family, transition_kind) stratum, retain the single lowest "
                "SHA-256 ranked transition identity"
            ),
            "transition_kinds": ["current", "successor"],
            "rank_namespace": "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1/RAW_AUDIT_V1",
            "rank_input": (
                "namespace UTF-8 || 0x00 || canonical UTF-8 JSON of transition_uid, role, family, "
                "and transition_kind"
            ),
            "selection_depends_on_sensor_results": False,
            "selected_transition_retention": (
                "retain finite raw clouds for all four conditions and both evidence modes"
            ),
            "dense_continuum_rule": (
                "a continuum has no finite complete cloud; persist the frozen per-step/per-link "
                "closest-surface target witnesses, acquisition times, and compact support/nominal/self/"
                "environment visibility matrices. These exact target-directed queries determine the "
                "prospectively frozen witness estimator but are not a replacement finite angular scan"
            ),
            "other_raw_ray_or_point_persistence": "forbidden",
            "always_persist": [
                "ray-pattern definitions",
                "phase digests",
                "point timestamps for retained rows",
                "per-transition per-link structured outputs",
            ],
        },
        "classifications": {
            "primary_exactly_one": list(PRIMARY_CLASSIFICATIONS),
            "primary_rules": {
                "PLATFORM_RANGE_COVERAGE_SIGNAL": "REALISTIC_PLATFORM_SCAN true-future passes",
                "SCAN_DENSITY_OR_TIMING_BOTTLENECK": (
                    "realistic platform fails and dense platform-mount FOV passes"
                ),
                "BODY_CENTRIC_MOUNT_REQUIRED": (
                    "dense platform-mount FOV fails and dense body-centric single-origin passes"
                ),
                "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO": (
                    "dense body-centric single-origin true-future fails"
                ),
                "RANGE_SENSOR_CONTRACT_UNRESOLVED": (
                    "realistic platform condition cannot be specified; dense conditions still execute"
                ),
            },
            "secondary": list(SECONDARY_CLASSIFICATIONS),
        },
        "next_architecture_if_any_condition_passes": {
            "id": "PER_LINK_CLEARANCE_PREDICTOR_V1",
            "status": "SPECIFY_ONLY_DO_NOT_TRAIN",
            "outputs": [
                "per-link minimum clearance through the committed tick",
                "time to first clearance violation",
                "obstacle/body sector",
                "observation support",
                "lower confidence bound or uncertainty interval",
            ],
            "inputs": [
                "planning-time range observation",
                "articulated embodied state",
                "one-tick candidate action",
                "control history",
            ],
            "deterministic_derivations": ["contact", "successor viability"],
            "forbidden_only_outputs": [
                "binary contact scalar",
                "binary nonviability scalar",
                "utility score",
            ],
        },
        "source_and_environment": {
            "expected_entrypoint": "scripts/evaluate_body_centric_range_coverage_qualification_v1.py",
            "ray_core": "lewm/safety/body_centric_range_coverage_v1.py",
            "contract_module": "lewm/safety/body_centric_range_coverage_qualification_v1_contract.py",
            "fixture_test": "lewm/tests/test_body_centric_range_coverage_v1.py",
            "evaluator_fixture_test": "lewm/tests/test_evaluate_body_centric_range_coverage_qualification_v1.py",
            "contract_test": "lewm/tests/test_body_centric_range_coverage_qualification_v1_contract.py",
            "tracked_contract_receipt": str(TRACKED_CONTRACT_RECEIPT_PATH),
            "tracked_output_schema": str(TRACKED_OUTPUT_SCHEMA_PATH),
            "tracked_source_closure": str(TRACKED_SOURCE_CLOSURE_PATH),
            "source_closure_timing": "persist before the first scientific sensor result is calculated",
            "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
            "known_runtime": {
                "classification": "EXISTING_ENVIRONMENT_REUSABLE",
                "environment_name": "genesis_render_vulkan",
                "compatibility_path": "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_render_vulkan",
                "resolved_prefix": "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_render_vulkan",
                "python": "3.12.3",
                "genesis": "0.3.14",
                "torch": "2.12.0+cu130",
                "torch_distribution_version": "2.12.0",
                "torch_build_version_source_sha256": "a3f452de1a9dcee621e34adc683ab3b1e20836c6a69e40df94f93de21fced9df",
                "numpy": "2.4.6",
                "scipy": "1.17.1",
                "tinyquadjepa_required": False,
                "recovery_environment_receipt": "/home/andrewknowles/recovery/REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1/environment_receipt.json",
                "recovery_environment_receipt_sha256": "c56018c80c47dcc09350a9dc6b4d4930837ded2d1c5a1693e996b91f3777b8f5",
                "pip_freeze_receipt": "/home/andrewknowles/recovery/REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1/genesis_render_vulkan_pip_freeze.txt",
                "pip_freeze_sha256": "324b3b3b9d90ed85dd11e5d563d767489fbd928cef3602712812493eeccd5a90",
                "execution_imports_genesis": False,
                "execution_imports_torch": False,
            },
            "environment_policy": (
                "use predecessor-compatible environment; do not install or upgrade opportunistically"
            ),
            "import_closure_receipt_required": True,
        },
        "prohibitions": [
            "model training",
            "fresh panel or corpus collection",
            "opening or executing the JEPA predictor",
            "changing frozen state identities or role membership",
            "changing action identities or the action bank",
            "replacing or changing repaired transitions",
            "altering contact labels",
            "reading or opening untouched G2 evaluation",
            "restarting interrupted occupancy v3",
            "restarting or reinterpreting completed occupancy v4",
            "retraining recurrent memory",
            "training a successor predictor",
            "learned closed-loop navigation",
            "implementing memory, novelty, routing, or beacon capture",
            "mount search using contact outcomes",
            "development-held-out threshold selection",
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
    """Return an independent copy of the frozen, self-digesting receipt."""

    return {
        **copy.deepcopy(_CONTRACT_CORE),
        "contract_sha256": CONTRACT_SHA256,
    }


def contract_receipt() -> dict[str, Any]:
    """Compatibility name for evaluator code that materialises the receipt."""

    return build_contract()


def build_output_schema() -> dict[str, Any]:
    """Return an independent copy of the frozen output-schema receipt."""

    return {
        **copy.deepcopy(_OUTPUT_SCHEMA_CORE),
        "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
    }


def contract_receipt_bytes() -> bytes:
    """Canonical on-disk contract bytes, including exactly one final LF."""

    return canonical_json_bytes(build_contract()) + b"\n"


def output_schema_receipt_bytes() -> bytes:
    """Canonical on-disk output-schema bytes, including exactly one final LF."""

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
        raise ContractError(f"{label} does not exactly match the prospectively frozen content")
    return {**candidate, digest_key: declared}


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate digest and exact prospective content; re-signing a mutation fails."""

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
    """Create the contract receipt once; accept an existing byte-identical file."""

    return _write_immutable(Path(path), contract_receipt_bytes(), "contract receipt")


def write_output_schema(path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH) -> Path:
    """Create the output-schema receipt once; never replace divergent content."""

    return _write_immutable(Path(path), output_schema_receipt_bytes(), "output schema receipt")


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


def load_and_validate_contract(path: str | Path = TRACKED_CONTRACT_RECEIPT_PATH) -> dict[str, Any]:
    decoded = _load_canonical_receipt(Path(path), contract_receipt_bytes(), "contract receipt")
    return validate_contract(decoded)


def load_and_validate_output_schema(
    path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH,
) -> dict[str, Any]:
    decoded = _load_canonical_receipt(
        Path(path), output_schema_receipt_bytes(), "output schema receipt"
    )
    return validate_output_schema(decoded)


__all__ = [
    "CONDITION_IDS",
    "CONTRACT",
    "CONTRACT_RECEIPT_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_SHA256",
    "ContractError",
    "COVERAGE_ERROR_CLASSES",
    "EVIDENCE_MODE_IDS",
    "EXPERIMENT_ID",
    "L2_PHASE_NAMESPACE",
    "OUTPUT_ROOT",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_RECEIPT_SHA256",
    "OUTPUT_SCHEMA_SHA256",
    "OUTPUT_SCHEMA_VERSION",
    "PRIMARY_CLASSIFICATIONS",
    "SECONDARY_CLASSIFICATIONS",
    "TRACKED_CONTRACT_RECEIPT_PATH",
    "TRACKED_OUTPUT_SCHEMA_PATH",
    "TRACKED_SOURCE_CLOSURE_PATH",
    "build_contract",
    "build_output_schema",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "contract_receipt",
    "contract_receipt_bytes",
    "derive_l2_scan_phases",
    "load_and_validate_contract",
    "load_and_validate_output_schema",
    "output_schema_receipt_bytes",
    "validate_contract",
    "validate_output_schema",
    "write_contract",
    "write_output_schema",
]
