"""Source-only contract for Patch-Whitened Action-Residual JEPA V2 Action-Gain.

Importing this module reads no generated input, RGB payload, checkpoint,
runtime output, or accelerator state and imports no tensor library.  The
separately reviewed runner owns the bounded execution lifecycle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_jepa_encoder_pretraining_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_runner.py"
)
OBJECTIVE_MODEL_RELATIVE_PATH = (
    "lewm/models/patch_whitened_action_residual_jepa.py"
)
OBJECTIVE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_patch_whitened_action_residual_jepa.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py"
)
SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_source_closure_v3.py"
)
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH
STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py"
)
STATIC_PHYSICAL_CONTRACT_FILE_SHA256 = (
    "3553810c79686f642a30fdfd0d2ff6ae047a97ea65c1366cae4cb3231e44e669"
)
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py"
)
BASE_LAUNCHER_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
MATCHED_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py"
)
MATCHED_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
)
SCHEDULE_ADAPTER_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
PHASE2D_MODEL_RELATIVE_PATH = "lewm/models/phase2d_spatial_lewm.py"
ENCODER_MODEL_RELATIVE_PATH = "lewm/models/encoders.py"
SIGREG_MODEL_RELATIVE_PATH = "lewm/models/sigreg.py"
SPATIAL_MODEL_RELATIVE_PATH = "lewm/models/spatial_lewm.py"
SPATIAL_PREDICTOR_MODEL_RELATIVE_PATH = (
    "lewm/models/spatial_predictor.py"
)
MULTIRES_MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
)
SHARED_V5_MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5.py"
)
EGOMOTION_BEV_JEPA_MODEL_RELATIVE_PATH = (
    "lewm/models/egomotion_bev_jepa.py"
)
CAMERA_EVIDENCE_MODEL_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4.py"
)
CAMERA_EVIDENCE_TRAINING_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_training.py"
)
HIERARCHICAL_FIRST_HIT_RELATIVE_PATH = (
    "lewm/models/"
    "observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py"
)
GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH = (
    "lewm/models/"
    "observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py"
)
TAIL_DEPTH_LOSS_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_"
    "protected_camera_adaptation_v4_tail_depth.py"
)
CAMERA_EVIDENCE_BENCHMARK_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
)
CAMERA_FIT_METRICS_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain_"
    "preregistration_2026-07-25.md"
)
PREREGISTRATION_COMMIT = "85b8bd6ae41652af744a011794060323c47be172"
PREREGISTRATION_FILE_SHA256 = (
    "1897c6841e88b7ab9116649b5f8f8af009a70bbe6d49938ee59d14683efcb095"
)
PREREGISTRATION_BYTE_COUNT = 5_711
PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v1_"
    "terminal_audit_2026-07-25.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = (
    "5c1ebb2b5f07f7be9ee152ea75b409358fb41477"
)
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "a87d1a706b912e8774a8e13b858e568ae91fbc1529ea4744adb189f0569457c7"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "ad6a97738c7143f6649d43a85376507f82c3522d79de667406a9a73ecffb5a8c"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 13_309

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain_"
    "source_manifest_2026-07-25.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain_"
    "source_review_2026-07-25.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain_"
    "execution_authorization_2026-07-25.json"
)
SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    OBJECTIVE_MODEL_RELATIVE_PATH,
    OBJECTIVE_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted((
    STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH,
    BASE_LAUNCHER_RELATIVE_PATH,
    BASE_LAUNCHER_CONTRACT_RELATIVE_PATH,
    MATCHED_V1_CONTRACT_RELATIVE_PATH,
    MATCHED_V1_RUNNER_RELATIVE_PATH,
    SCHEDULE_ADAPTER_RELATIVE_PATH,
    PHASE2D_MODEL_RELATIVE_PATH,
    ENCODER_MODEL_RELATIVE_PATH,
    SIGREG_MODEL_RELATIVE_PATH,
    SPATIAL_MODEL_RELATIVE_PATH,
    SPATIAL_PREDICTOR_MODEL_RELATIVE_PATH,
    MULTIRES_MODEL_RELATIVE_PATH,
    SHARED_V5_MODEL_RELATIVE_PATH,
    EGOMOTION_BEV_JEPA_MODEL_RELATIVE_PATH,
    CAMERA_EVIDENCE_MODEL_RELATIVE_PATH,
    CAMERA_EVIDENCE_TRAINING_RELATIVE_PATH,
    HIERARCHICAL_FIRST_HIT_RELATIVE_PATH,
    GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH,
    TAIL_DEPTH_LOSS_RELATIVE_PATH,
    CAMERA_EVIDENCE_BENCHMARK_RELATIVE_PATH,
    CAMERA_FIT_METRICS_RELATIVE_PATH,
    SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH,
)))
SOURCE_PATHS = tuple(sorted((
    *ADDITIVE_SOURCE_PATHS,
    *REUSED_SOURCE_PATHS,
)))
SOURCE_MANIFEST_ENTRYPOINTS = (
    LAUNCHER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_patch_whitened_action_residual_jepa_probe_v2_action_gain"
)

RAW_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/manifest.json"
RAW_AUDIT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit_v13.json"
)
RAW_PAIRS_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/pairs.jsonl"
RAW_ENDPOINTS_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/endpoints.jsonl"
N320_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1"
)
N320_GATE_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/gate.json"
N320_CHECKPOINT_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/checkpoint.pt"
SCHEDULE_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v4/schedule.json"
)

RUNTIME_FILE_SHA256 = {
    RAW_MANIFEST_RELATIVE_PATH:
        "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360",
    RAW_AUDIT_RELATIVE_PATH:
        "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76",
    N320_GATE_RELATIVE_PATH:
        "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6",
    N320_CHECKPOINT_RELATIVE_PATH:
        "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0",
    SCHEDULE_RELATIVE_PATH:
        "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
}
RUNTIME_CONTENT_SHA256 = {
    RAW_MANIFEST_RELATIVE_PATH:
        "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a",
    RAW_AUDIT_RELATIVE_PATH:
        "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca",
    N320_GATE_RELATIVE_PATH:
        "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b",
    N320_CHECKPOINT_RELATIVE_PATH:
        "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b",
    SCHEDULE_RELATIVE_PATH:
        "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
}
RUNTIME_BYTE_COUNTS = {
    RAW_MANIFEST_RELATIVE_PATH: 311_598,
    RAW_AUDIT_RELATIVE_PATH: 26_975,
    N320_GATE_RELATIVE_PATH: 7_960,
    N320_CHECKPOINT_RELATIVE_PATH: 13_777_100,
    SCHEDULE_RELATIVE_PATH: 607_373,
}

TRAIN_ROLE_COUNTS = {
    "pairs": 4_262,
    "unique_endpoints": 7_777,
    "scenes": 72,
}
SELECTION_ROLE_COUNTS = {
    "pairs": 495,
    "unique_endpoints": 924,
    "scenes": 8,
}
SELECTION_NON_HOLD_PAIR_COUNT = 435
ACTION_VOCABULARY = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
SCENE_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

BASE_INITIALIZATION_SEED = 20260712
SCHEDULE_SEED = 20260713
HOLD_ACTION_INDEX = 6
RESIDUAL_SCALE = 0.1 / math.sqrt(192.0)
ACTION_DISCRIMINATION_WEIGHT = 10.0
WHITENING_EPSILON = 1e-4
WHITENING_VARIANCE_WEIGHT = 0.50
WHITENING_COVARIANCE_WEIGHT = 0.02
ACTION_GATE_BIAS = 0.01
ACTION_GATE_WEIGHT_STD = 0.01 / math.sqrt(192.0)
PHASE_A_MAXIMUM_UPDATE = 1_000
PHASE_B_MAXIMUM_UPDATE = 1_000
MAXIMUM_UPDATE = 1_000
CUMULATIVE_MAXIMUM_UPDATE = (
    PHASE_A_MAXIMUM_UPDATE + PHASE_B_MAXIMUM_UPDATE
)
CHECKPOINT_UPDATES = (100, 400, 1_000)
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
EFFECTIVE_BATCH_SIZE = 16
PHASE_A_MAXIMUM_PRESENTATIONS = 16_000
PHASE_B_MAXIMUM_PRESENTATIONS = 16_000
MAXIMUM_PRESENTATIONS = 16_000
CUMULATIVE_MAXIMUM_PRESENTATIONS = (
    PHASE_A_MAXIMUM_PRESENTATIONS + PHASE_B_MAXIMUM_PRESENTATIONS
)
PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES = 60
CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES = 120

CHECKPOINT_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}
PHASE_A_SCHEDULE_PREFIX_SHA256 = dict(CHECKPOINT_SCHEDULE_PREFIX_SHA256)
PHASE_B_SCHEDULE_PREFIX_SHA256 = dict(CHECKPOINT_SCHEDULE_PREFIX_SHA256)

PHASE_A_PASS_THRESHOLDS = {
    "centered_raw_patch_effective_rank_minimum": 48.0,
    "centered_projected_target_effective_rank_minimum": 48.0,
    "update_zero_health_fraction_minimum": 0.25,
    "shuffled_next_ratio_maximum": 0.90,
    "mean_target_ratio_maximum": 0.90,
    "cyclic_wrong_action_ratio_maximum": 0.95,
    "hold_action_ratio_maximum": 0.95,
    "positive_family_margin_count_minimum": 6,
    "shuffled_current_ratio_maximum": 0.95,
}
PHASE_A_UPDATE_100_THRESHOLDS = {
    "centered_raw_patch_effective_rank_strictly_greater_than":
        27.717458724975586,
    "centered_projected_target_effective_rank_strictly_greater_than":
        17.426651000976562,
    "cyclic_wrong_action_ratio_strictly_less_than": 0.99,
    "hold_action_ratio_strictly_less_than": 0.99,
    "positive_family_margin_count_minimum": 6,
}
PHASE_A_UPDATE_400_THRESHOLDS = {
    "centered_raw_patch_effective_rank_minimum": 37.85872936248779,
    "centered_projected_target_effective_rank_minimum": 32.71332550048828,
    "cyclic_wrong_action_ratio_maximum": 0.975,
    "hold_action_ratio_maximum": 0.975,
    "positive_family_margin_count_minimum": 6,
}
PHASE_B_PASS_THRESHOLDS = {
    "complete_physical_scope_count_minimum": 1,
    "passed_margin_count_minimum": 98,
    "total_shortfall_strictly_less_than": 41.01776266878769,
    "rough_pixel_balanced_accuracy_strictly_greater_than":
        0.8198594673963917,
    "rough_ground_balanced_accuracy_strictly_greater_than":
        0.647134926562893,
    "rough_depth_p95_m_strictly_less_than": 0.9777327477931971,
}
PASS_THRESHOLDS = PHASE_B_PASS_THRESHOLDS
MARGIN_COUNT = 189

PHASE_A_ENCODER_PARAMETER_PREFIXES = ("encoder.",)
PHASE_A_AUXILIARY_PARAMETER_PREFIXES = (
    "online_target_projector.",
    "prediction_projector.",
    "predictor.",
)
PHASE_A_TRAINABLE_PARAMETER_PREFIXES = (
    *PHASE_A_ENCODER_PARAMETER_PREFIXES,
    *PHASE_A_AUXILIARY_PARAMETER_PREFIXES,
)
PHASE_A_FROZEN_PARAMETER_PREFIXES = (
    "appearance_projector.",
    "target_encoder.",
    "target_geometry_module.",
    "target_projector.",
)
PHASE_B_TRAINABLE_PARAMETER_PREFIXES = ("evidence_head.",)
PHASE_B_FROZEN_PARAMETER_PREFIXES = (
    "encoder.",
    "bev_decoder.",
    "predictor.",
    "occupancy_head.",
    "target_encoder.",
    "target_bev_decoder.",
)

PHASE_A_METRIC_FIELDS = frozenset({
    "all_values_finite",
    "ema_target_gradient_free",
    "pair_count",
    "scene_family_count",
    "centered_raw_patch_effective_rank",
    "centered_projected_target_effective_rank",
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
    "true_pair_mse",
    "shuffled_next_mse",
    "mean_target_mse",
    "cyclic_wrong_action_mse",
    "cyclic_wrong_action_pair_count",
    "all_wrong_action_candidate_count",
    "hardest_wrong_action_mse",
    "non_hold_pair_count",
    "non_hold_true_pair_mse",
    "hold_action_mse",
    "hold_action_pair_count",
    "hold_action_rows_match_non_hold_rows",
    "shuffled_current_mse",
    "per_family",
})
PHASE_A_UPDATE0_FIELDS = frozenset({
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
})
PHASE_A_OBSERVATION_INTEGRITY_FIELDS = frozenset({
    "rng_state_preserved",
    "state_mutation_count",
})

CONTROL_CONTINUE = "CONTINUE_INFORMATIONAL"
CONTROL_PHASE_A_PASS = "PASS_PHASE_A_ENTER_FROZEN_PHYSICAL_PROBE"
CONTROL_PHASE_A_FAIL = "FAIL_PHASE_A_TERMINAL_NO_PHASE_B_NO_RETRY"
CONTROL_PHASE_A_UPDATE_100_FAIL = (
    "FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL"
)
CONTROL_PHASE_A_UPDATE_400_FAIL = (
    "FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL"
)
CONTROL_PASS = "PASS_BOUNDED_FALSIFICATION"
CONTROL_FAIL = "FAIL_TERMINAL_NO_RETRY"
CONTROL_INTEGRITY_FAIL = "INTEGRITY_FAILURE_TERMINAL_NO_RETRY"

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
PHASE_A_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_metrics_v1"
PHASE_A_ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_artifact_v1"
PHASE_B_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_b_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

DOWNSTREAM_DENIALS = {
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_resume_second_seed_schedule_extension_or_replacement_authorized":
        False,
}
SOURCE_ONLY_AUTHORITY = {
    "execution_authorized": False,
    "gpu_or_hardware_query_authorized": False,
    "generated_input_open_authorized": False,
    "dataset_or_rgb_open_authorized": False,
    "checkpoint_or_tensor_open_authorized": False,
    "runtime_output_open_authorized": False,
    "output_mutation_authorized": False,
    "phase_b_authorized_without_phase_a_pass": False,
    **DOWNSTREAM_DENIALS,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
SCIENTIFIC_REVIEW_CHECKS = {
    "prior_v1_terminal_audit_bound": True,
    "fresh_v2_is_not_v1_retry_or_resume": True,
    "action_discrimination_weight_exactly_10": True,
    "only_action_weight_and_fresh_identity_changed": True,
    "ema_current_residual_skip_exact": True,
    "patch_whitening_matches_preregistration": True,
    "all_nine_real_actions_and_hold_exact": True,
    "cyclic_acceptance_distinct_from_all_candidate_training": True,
    "zero_vector_absent": True,
    "continuation_gates_exact": True,
    "terminal_phase_a_gate_exact": True,
    "phase_b_conditional_and_unchanged": True,
    "no_threshold_role_seed_schedule_or_cap_drift": True,
}
EXECUTION_AUTHORITY = {
    "one_exact_fresh_attempt_authorized": True,
    "phase_a_authorized": True,
    "phase_b_only_after_exact_phase_a_pass_authorized": True,
    "n320_initialization_only_authorized": True,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent": True,
    **DOWNSTREAM_DENIALS,
}

PROHIBITED_RUNTIME_CATEGORIES = (
    "probability_calibration",
    "prior_attempt_roots",
    "rejected_checkpoints",
    "g2",
    "navigation",
    "heldout",
    "sealed",
    "production",
    "deployment",
)


def _load_static_physical_contract() -> Any:
    source = ROOT / STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != STATIC_PHYSICAL_CONTRACT_FILE_SHA256:
        raise ImportError("frozen static physical contract source changed")
    spec = importlib.util.spec_from_file_location(
        "_lewm_jepa_encoder_v2_action_gain_static_physical_contract",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen static physical contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STATIC_PHYSICAL = _load_static_physical_contract()
SCOPES = tuple(_STATIC_PHYSICAL.SCOPES)
PHYSICAL_LOWER_THRESHOLDS = dict(
    _STATIC_PHYSICAL.PHYSICAL_LOWER_THRESHOLDS
)
PHYSICAL_UPPER_THRESHOLDS = dict(
    _STATIC_PHYSICAL.PHYSICAL_UPPER_THRESHOLDS
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise TypeError("self-hashed core must be a plain dict without its hash")
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"canonical JSON repeats key {key}")
        value[key] = item
    return value


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{name} must be exactly one canonical JSON line")
    try:
        value = json.loads(
            raw[:-1].decode("ascii"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite constant {item}")
            ),
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not finite duplicate-safe ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} self hash changed")
    return value


def safe_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a safe relative path")
    return value


def artifact_binding(
    path: str,
    raw: bytes,
    *,
    content_sha256: str,
) -> dict[str, Any]:
    safe_relative_path(path, name="artifact path")
    if not is_sha256(content_sha256):
        raise ValueError("artifact content hash is malformed")
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
    }


def validate_binding(
    value: object,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "path",
        "file_sha256",
        "content_sha256",
        "byte_count",
    }:
        raise ValueError("artifact binding fields changed")
    safe_relative_path(value["path"], name="artifact path")
    if (
        (path is not None and value["path"] != path)
        or not is_sha256(value["file_sha256"])
        or not is_sha256(value["content_sha256"])
        or type(value["byte_count"]) is not int
        or value["byte_count"] <= 0
    ):
        raise ValueError("artifact binding changed")
    return dict(value)


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def prior_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": PRIOR_TERMINAL_AUDIT_COMMIT,
        "file_sha256": PRIOR_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": PRIOR_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": PRIOR_TERMINAL_AUDIT_BYTE_COUNT,
    }


def _runtime_leaf(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": RUNTIME_FILE_SHA256[path],
        "content_sha256": RUNTIME_CONTENT_SHA256[path],
        "byte_count": RUNTIME_BYTE_COUNTS[path],
    }


def runtime_authorization_template() -> dict[str, Any]:
    return {
        "raw": {
            "root": RAW_ROOT_RELATIVE_PATH,
            "manifest": _runtime_leaf(RAW_MANIFEST_RELATIVE_PATH),
            "audit": _runtime_leaf(RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": dict(TRAIN_ROLE_COUNTS),
                "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            },
            "phase_a_grant": {
                "allowed_inputs": [
                    "bound_pair_index",
                    "bound_endpoint_index",
                    "bound_current_rgb",
                    "bound_next_rgb",
                    "requested_primitive",
                ],
                "camera_supervision_array_open_authorized": False,
                "general_raw_v13_frame_loader_authorized": False,
            },
            "phase_b_grant": {
                "unchanged_multiresolution_physical_probe": True,
            },
        },
        "camera": {
            "root": N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(N320_GATE_RELATIVE_PATH),
            "checkpoint": _runtime_leaf(N320_CHECKPOINT_RELATIVE_PATH),
            "fit_seed": 20260710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(SCHEDULE_RELATIVE_PATH),
    }


def validate_runtime_inputs(value: object) -> dict[str, Any]:
    expected = runtime_authorization_template()
    if value != expected:
        raise PermissionError("runtime input authority changed")
    return {
        "raw": dict(value["raw"]),
        "camera": dict(value["camera"]),
        "schedule": dict(value["schedule"]),
    }


def phase_a_model_config() -> dict[str, Any]:
    """Return exact ``Phase2DSpatialLeWorldModel`` constructor arguments."""

    return {
        "latent_dim": 192,
        "cmd_dim": 9,
        "pred_layers": 2,
        "pred_heads": 6,
        "pred_dim_head": 32,
        "pred_mlp_dim": 384,
        "pred_dropout": 0.0,
        "image_size": 112,
        "patch_size": 7,
        "encoder_depth": 6,
        "encoder_heads": 6,
        "encoder_mlp_ratio": 4,
        "encoder_dropout": 0.0,
        "appearance_sigreg_lambda": 0.0,
        "spatial_variance_lambda": 0.0,
        "action_identifiability_lambda": 0.0,
        "zero_action_lambda": 0.0,
        "action_margin_fraction": 0.10,
        "action_margin_floor": 1e-4,
        "detach_action_control_state": True,
        "consequence_dim": 0,
        "consequence_loss_lambda": 0.0,
        "action_utility_loss_lambda": 0.0,
        "target_geometry": "patch",
        "num_target_slots": 16,
        "sigreg_projections": 64,
        "sigreg_knots": 9,
        "target_ema_momentum": 0.996,
        "prediction_input_mode": "state_action",
    }


def build_schedule_identity(phase: str) -> dict[str, Any]:
    if phase not in {"phase_a", "phase_b"}:
        raise ValueError("schedule phase must be phase_a or phase_b")
    prefixes = (
        PHASE_A_SCHEDULE_PREFIX_SHA256
        if phase == "phase_a"
        else PHASE_B_SCHEDULE_PREFIX_SHA256
    )
    return {
        "phase": phase,
        "source": _runtime_leaf(SCHEDULE_RELATIVE_PATH),
        "seed": SCHEDULE_SEED,
        "updates": 1_000,
        "presentations": 16_000,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoints": list(CHECKPOINT_UPDATES),
        "prefix_sha256": {
            str(update): digest for update, digest in prefixes.items()
        },
        "reuse_same_frozen_prefix_independently": True,
    }


def validate_schedule_identity(
    value: object,
    *,
    phase: str,
) -> dict[str, Any]:
    expected = build_schedule_identity(phase)
    if value != expected:
        raise PermissionError(f"{phase} schedule identity changed")
    return dict(value)


def science_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "scientific_question":
            "tenfold_action_discrimination_gain_preserves_whitened_rank_and_"
            "passes_action_and_scene_disjoint_physical_gates",
        "initialization": {
            "seed": BASE_INITIALIZATION_SEED,
            "n320_online_encoder_copy": True,
            "n320_ema_encoder_copy": True,
            "predictor_and_projectors_from_fixed_seed": True,
            "adalan_gate_generator": {
                "device": "cpu",
                "dtype": "float32",
                "seed": BASE_INITIALIZATION_SEED,
                "seed_count": 1,
                "draw_order":
                    "block_order_attention_gate_then_mlp_gate",
                "gate_row_slices": ["2D:3D", "5D:6D"],
                "weight_std": ACTION_GATE_WEIGHT_STD,
                "bias": ACTION_GATE_BIAS,
                "all_non_gate_modulation_rows_zero": True,
            },
            "rejected_checkpoint_open_count": 0,
        },
        "data": {
            "raw_v13_reused_exactly": True,
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "probability_calibration_open_count": 0,
            "rebuild_refine_rebalance_filter_or_resample": False,
        },
        "phase_a": {
            "model_class": "Phase2DSpatialLeWorldModel",
            "model_config": phase_a_model_config(),
            "inert_constructor_compatibility": {
                "appearance_sigreg_lambda": 0.0,
                "spatial_variance_lambda": 0.0,
                "action_identifiability_lambda": 0.0,
                "zero_action_lambda": 0.0,
                "action_margin_fraction_unused": 0.10,
                "action_margin_floor_unused": 1e-4,
                "sigreg_projections_unused": 64,
                "sigreg_knots_unused": 9,
                "legacy_objective_call_count": 0,
            },
            "current_only_adapter_required": True,
            "model_forward_call_authorized": False,
            "rgb_preprocessing": {
                "decode": "PIL",
                "convert": "RGB",
                "resize": [112, 112],
                "resize_mode": "bilinear",
                "scale_divisor": 255.0,
                "layout": "CHW",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "action_vocabulary": list(ACTION_VOCABULARY),
            "hold_action_index": HOLD_ACTION_INDEX,
            "training_action_candidates":
                "all_nine_real_one_hot_primitives_no_zero_vector",
            "acceptance_wrong_action":
                "cyclic_index_plus_one_mod_9_for_every_row",
            "hold_action_mask": "requested_primitive_is_not_hold",
            "objective": {
                "ema_current_skip_stop_gradient": True,
                "ema_next_target_stop_gradient": True,
                "residual_scale": RESIDUAL_SCALE,
                "prediction":
                    "normalize(z_current_ema+alpha*"
                    "P_prediction(Predictor(raw_online_current,action)))",
                "jepa_loss":
                    "mean_patch_feature_mse(true_prediction,z_next_ema)",
                "action_discrimination_weight":
                    ACTION_DISCRIMINATION_WEIGHT,
                "wrong_action_loss":
                    "mean_rows(mean_eligible_candidates(relu("
                    "stop_gradient(E_true)/0.95-E_wrong)))",
                "hold_action_loss":
                    "mean_non_hold(relu(stop_gradient(E_true)/0.95-E_hold))",
                "empty_hold_microbatch_loss": 0.0,
                "true_path_online_state_detached": False,
                "non_true_control_state_detached": True,
                "appearance_projector_frozen": True,
                "old_cls_sigreg_count": 0,
                "old_marginal_spatial_variance_count": 0,
                "whitening": {
                    "branches": [
                        "online_raw_current_patch",
                        "normalized_online_projected_current_patch",
                    ],
                    "position_centering": "subtract_batch_mean_per_patch",
                    "matrix_shape": ["B*N", 192],
                    "rms_denominator_stop_gradient": True,
                    "epsilon": WHITENING_EPSILON,
                    "covariance_denominator": "B*N-1",
                    "variance":
                        "mean(relu(1-sqrt(diag(C)+epsilon)))",
                    "covariance":
                        "sum(square(off_diagonal(C)))/D",
                    "variance_weight": WHITENING_VARIANCE_WEIGHT,
                    "covariance_weight": WHITENING_COVARIANCE_WEIGHT,
                },
            },
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 1e-4,
                "precision": "float32",
                "autocast": False,
                "global_clip_norm": 1.0,
                "encoder_learning_rate": 1e-4,
                "other_learning_rate": 3e-4,
                "other_prefixes": [
                    "online_target_projector.",
                    "prediction_projector.",
                    "predictor.",
                ],
                "frozen_compatibility_prefixes": [
                    "appearance_projector.",
                ],
                "target_parameters_excluded": True,
            },
            "schedule": build_schedule_identity("phase_a"),
            "gate": {
                "thresholds": dict(PHASE_A_PASS_THRESHOLDS),
                "update_100_continuation_thresholds":
                    dict(PHASE_A_UPDATE_100_THRESHOLDS),
                "update_400_continuation_thresholds":
                    dict(PHASE_A_UPDATE_400_THRESHOLDS),
                "selection_pair_count": 495,
                "selection_non_hold_pair_count":
                    SELECTION_NON_HOLD_PAIR_COUNT,
                "all_wrong_action_candidate_count": 3_960,
                "scene_family_count": 8,
                "position_centering":
                    "subtract_population_mean_independently_per_patch",
                "raw_cross_sample_variance":
                    "mean_patch_feature(var_population(raw_ema_next_tokens))",
                "content_residual_spatial_diversity":
                    "mean_row_feature(var_patch(position_centered_raw_tokens))",
                "effective_rank":
                    "exp(-sum(p_i_log_p_i))_of_covariance_eigenvalues",
                "mse_reduction":
                    "per_row_mean_patch_feature_then_population_mean",
                "cyclic_wrong_action_population":
                    "all_495_rows_same_rows_as_true_pair_population",
                "hold_action_population":
                    "exact_same_non_hold_rows_as_non_hold_true_pair_population",
                "hardest_wrong_action":
                    "informational_minimum_mse_over_eight_wrong_candidates",
                "shuffled_current":
                    "shuffle_online_raw_current_and_matching_ema_current_skip_"
                    "together_keep_action_and_ema_next_fixed",
            },
        },
        "phase_b": {
            "entered_only_after_phase_a_pass": True,
            "copied_state": "phase_a_terminal_online_raw_encoder_only",
            "trainable_prefixes": ["evidence_head."],
            "frozen_prefixes": [
                "encoder.",
                "jepa_predictor_projectors_and_ema.",
                "bev_modules.",
                "occupancy_head.",
            ],
            "hard_sync": {
                "count": 1,
                "copied_prefixes": ["target_encoder."],
                "forbidden_copy_prefixes": ["target_bev_decoder."],
                "target_bev_decoder_initialization_identity_verified_without_"
                "copy": True,
            },
            "jepa_objective_count": 0,
            "ema_update_count": 0,
            "unchanged_multiresolution_evidence_head_and_physical_evaluator":
                True,
            "schedule": build_schedule_identity("phase_b"),
            "pass_thresholds": dict(PHASE_B_PASS_THRESHOLDS),
            "promotable_shared_v5_checkpoint": False,
        },
        "cumulative_caps": {
            "updates": CUMULATIVE_MAXIMUM_UPDATE,
            "presentations": CUMULATIVE_MAXIMUM_PRESENTATIONS,
            "phase_a_gpu_active_minutes": PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            "total_gpu_active_minutes":
                CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES,
            "maximum_attempts": 1,
        },
        "authority": dict(DOWNSTREAM_DENIALS),
    }


def learning_rates(update: int) -> tuple[float, float]:
    """Return the unchanged Phase-B evidence-head schedule."""

    return _STATIC_PHYSICAL.learning_rates(update)


def physical_margins(scope: Mapping[str, Any]) -> list[float]:
    """Return the unchanged 21 physical margins for one scope."""

    return _STATIC_PHYSICAL.physical_margins(scope)


def evaluate_physical_scopes(
    scopes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the unchanged nine-scope, 189-margin physical summary."""

    return _STATIC_PHYSICAL.evaluate_physical_scopes(scopes)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_denominator_ratio(
    numerator: float,
    denominator: float,
    *,
    name: str,
) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return numerator / denominator


def evaluate_phase_a(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact preregistered Phase-A terminal conjunction."""

    if type(metrics) is not dict or set(metrics) != PHASE_A_METRIC_FIELDS:
        raise ValueError("Phase-A metric fields changed")
    if (
        type(update0_metrics) is not dict
        or set(update0_metrics) != PHASE_A_UPDATE0_FIELDS
    ):
        raise ValueError("Phase-A update-zero fields changed")
    if (
        type(observation_integrity) is not dict
        or set(observation_integrity)
        != PHASE_A_OBSERVATION_INTEGRITY_FIELDS
        or type(observation_integrity["rng_state_preserved"]) is not bool
        or type(observation_integrity["state_mutation_count"]) is not int
        or observation_integrity["state_mutation_count"] < 0
    ):
        raise ValueError("Phase-A observation-integrity fields changed")
    if type(metrics["all_values_finite"]) is not bool:
        raise TypeError("all_values_finite must be Boolean")
    if type(metrics["ema_target_gradient_free"]) is not bool:
        raise TypeError("ema_target_gradient_free must be Boolean")
    if metrics["pair_count"] != 495 or type(metrics["pair_count"]) is not int:
        raise ValueError("Phase-A selection pair count changed")
    if (
        metrics["scene_family_count"] != 8
        or type(metrics["scene_family_count"]) is not int
    ):
        raise ValueError("Phase-A scene-family count changed")
    non_hold_count = metrics["non_hold_pair_count"]
    if (
        type(non_hold_count) is not int
        or non_hold_count != SELECTION_NON_HOLD_PAIR_COUNT
    ):
        raise ValueError("Phase-A non-hold population changed")
    if (
        type(metrics["cyclic_wrong_action_pair_count"]) is not int
        or metrics["cyclic_wrong_action_pair_count"]
        != metrics["pair_count"]
        or type(metrics["all_wrong_action_candidate_count"]) is not int
        or metrics["all_wrong_action_candidate_count"]
        != metrics["pair_count"] * 8
        or type(metrics["hold_action_pair_count"]) is not int
        or metrics["hold_action_pair_count"] != non_hold_count
        or type(metrics["hold_action_rows_match_non_hold_rows"]) is not bool
    ):
        raise ValueError("Phase-A control populations changed")

    numeric_names = PHASE_A_METRIC_FIELDS - {
        "all_values_finite",
        "ema_target_gradient_free",
        "pair_count",
        "scene_family_count",
        "cyclic_wrong_action_pair_count",
        "all_wrong_action_candidate_count",
        "non_hold_pair_count",
        "hold_action_pair_count",
        "hold_action_rows_match_non_hold_rows",
        "per_family",
    }
    values = {
        name: _finite_nonnegative(metrics[name], name=name)
        for name in numeric_names
    }
    update0 = {
        name: _finite_nonnegative(update0_metrics[name], name=f"update0 {name}")
        for name in PHASE_A_UPDATE0_FIELDS
    }
    if any(value <= 0.0 for value in update0.values()):
        raise ValueError("Phase-A update-zero health denominators must be positive")

    per_family = metrics["per_family"]
    if type(per_family) is not dict or tuple(per_family) != SCENE_FAMILIES:
        raise ValueError("Phase-A scene-family order changed")
    wrong_positive = 0
    hold_positive = 0
    normalized_families: dict[str, dict[str, float]] = {}
    for family in SCENE_FAMILIES:
        row = per_family[family]
        if type(row) is not dict or set(row) != {
            "cyclic_wrong_action_minus_true_mse",
            "hardest_wrong_action_minus_true_mse",
            "hold_action_minus_non_hold_true_mse",
            "hold_action_rows_match_non_hold_rows",
        }:
            raise ValueError(f"Phase-A family metrics changed: {family}")
        if type(row["hold_action_rows_match_non_hold_rows"]) is not bool:
            raise TypeError(
                f"{family} hold-action population identity must be Boolean"
            )
        wrong = _finite_signed(
            row["cyclic_wrong_action_minus_true_mse"],
            name=f"{family} cyclic-wrong-action margin",
        )
        hardest = _finite_signed(
            row["hardest_wrong_action_minus_true_mse"],
            name=f"{family} hardest-wrong-action margin",
        )
        hold = _finite_signed(
            row["hold_action_minus_non_hold_true_mse"],
            name=f"{family} hold-action margin",
        )
        wrong_positive += int(wrong > 0.0)
        hold_positive += int(
            hold > 0.0 and row["hold_action_rows_match_non_hold_rows"]
        )
        normalized_families[family] = {
            "cyclic_wrong_action_minus_true_mse": wrong,
            "hardest_wrong_action_minus_true_mse": hardest,
            "hold_action_minus_non_hold_true_mse": hold,
            "hold_action_rows_match_non_hold_rows":
                row["hold_action_rows_match_non_hold_rows"],
        }

    ratios = {
        "raw_cross_sample_variance_to_update0":
            values["raw_cross_sample_variance"]
            / update0["raw_cross_sample_variance"],
        "content_residual_spatial_diversity_to_update0":
            values["content_residual_spatial_diversity"]
            / update0["content_residual_spatial_diversity"],
        "true_to_shuffled_next": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_next_mse"],
            name="shuffled-next ratio",
        ),
        "true_to_mean_target": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["mean_target_mse"],
            name="mean-target ratio",
        ),
        "true_to_cyclic_wrong_action": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["cyclic_wrong_action_mse"],
            name="cyclic-wrong-action ratio",
        ),
        "non_hold_true_to_hold_action": _positive_denominator_ratio(
            values["non_hold_true_pair_mse"],
            values["hold_action_mse"],
            name="hold-action ratio",
        ),
        "true_to_hardest_wrong_action_informational":
            _positive_denominator_ratio(
                values["true_pair_mse"],
                values["hardest_wrong_action_mse"],
                name="hardest-wrong-action ratio",
            ),
        "true_to_shuffled_current": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_current_mse"],
            name="shuffled-current ratio",
        ),
    }
    threshold = PHASE_A_PASS_THRESHOLDS
    conjuncts = {
        "diagnostic_rng_and_model_state_preserved":
            observation_integrity["rng_state_preserved"]
            and observation_integrity["state_mutation_count"] == 0,
        "finite_and_ema_gradient_free":
            metrics["all_values_finite"]
            and metrics["ema_target_gradient_free"],
        "control_populations_exact":
            metrics["cyclic_wrong_action_pair_count"]
            == metrics["pair_count"]
            and metrics["all_wrong_action_candidate_count"]
            == metrics["pair_count"] * 8
            and metrics["hold_action_pair_count"] == non_hold_count
            and metrics["hold_action_rows_match_non_hold_rows"],
        "centered_raw_rank_at_least_48":
            values["centered_raw_patch_effective_rank"]
            >= threshold["centered_raw_patch_effective_rank_minimum"],
        "centered_projected_rank_at_least_48":
            values["centered_projected_target_effective_rank"]
            >= threshold["centered_projected_target_effective_rank_minimum"],
        "raw_cross_sample_variance_at_least_quarter_update0":
            ratios["raw_cross_sample_variance_to_update0"]
            >= threshold["update_zero_health_fraction_minimum"],
        "spatial_diversity_at_least_quarter_update0":
            ratios["content_residual_spatial_diversity_to_update0"]
            >= threshold["update_zero_health_fraction_minimum"],
        "true_at_most_point90_shuffled_next":
            ratios["true_to_shuffled_next"]
            <= threshold["shuffled_next_ratio_maximum"],
        "true_at_most_point90_mean_target":
            ratios["true_to_mean_target"]
            <= threshold["mean_target_ratio_maximum"],
        "true_at_most_point95_cyclic_wrong_action":
            ratios["true_to_cyclic_wrong_action"]
            <= threshold["cyclic_wrong_action_ratio_maximum"],
        "non_hold_true_at_most_point95_hold_action":
            ratios["non_hold_true_to_hold_action"]
            <= threshold["hold_action_ratio_maximum"],
        "cyclic_wrong_action_margin_positive_in_at_least_six_families":
            wrong_positive
            >= threshold["positive_family_margin_count_minimum"],
        "hold_action_margin_positive_in_at_least_six_families":
            hold_positive
            >= threshold["positive_family_margin_count_minimum"],
        "true_at_most_point95_shuffled_current":
            ratios["true_to_shuffled_current"]
            <= threshold["shuffled_current_ratio_maximum"],
    }
    passed = all(conjuncts.values())
    return {
        "passed": passed,
        "control": CONTROL_PHASE_A_PASS if passed else CONTROL_PHASE_A_FAIL,
        "conjuncts": conjuncts,
        "ratios": ratios,
        "counts": {
            "pair_count": 495,
            "cyclic_wrong_action_pair_count":
                metrics["cyclic_wrong_action_pair_count"],
            "all_wrong_action_candidate_count":
                metrics["all_wrong_action_candidate_count"],
            "non_hold_pair_count": non_hold_count,
            "hold_action_pair_count": metrics["hold_action_pair_count"],
            "scene_family_count": 8,
            "cyclic_wrong_action_positive_family_count": wrong_positive,
            "hold_action_positive_family_count": hold_positive,
        },
        "per_family": normalized_families,
    }


def evaluate_phase_a_continuation(
    update: int,
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact update-100 or update-400 continuation gate."""

    if update not in {100, 400}:
        raise ValueError("continuation gate update must be 100 or 400")
    terminal = evaluate_phase_a(
        metrics,
        update0_metrics,
        observation_integrity,
    )
    ratios = terminal["ratios"]
    counts = terminal["counts"]
    conjuncts = {
        "diagnostic_rng_and_model_state_preserved":
            terminal["conjuncts"][
                "diagnostic_rng_and_model_state_preserved"
            ],
        "finite_and_ema_gradient_free":
            terminal["conjuncts"]["finite_and_ema_gradient_free"],
        "control_populations_exact":
            terminal["conjuncts"]["control_populations_exact"],
    }
    if update == 100:
        threshold = PHASE_A_UPDATE_100_THRESHOLDS
        conjuncts.update({
            "centered_raw_rank_above_v3_update_zero":
                float(metrics["centered_raw_patch_effective_rank"])
                > threshold[
                    "centered_raw_patch_effective_rank_strictly_greater_than"
                ],
            "centered_projected_rank_above_v3_update_zero":
                float(metrics["centered_projected_target_effective_rank"])
                > threshold[
                    "centered_projected_target_effective_rank_"
                    "strictly_greater_than"
                ],
            "true_strictly_below_point99_cyclic_wrong_action":
                ratios["true_to_cyclic_wrong_action"]
                < threshold[
                    "cyclic_wrong_action_ratio_strictly_less_than"
                ],
            "non_hold_true_strictly_below_point99_hold_action":
                ratios["non_hold_true_to_hold_action"]
                < threshold["hold_action_ratio_strictly_less_than"],
            "cyclic_wrong_action_margin_positive_in_at_least_six_families":
                counts["cyclic_wrong_action_positive_family_count"]
                >= threshold["positive_family_margin_count_minimum"],
            "hold_action_margin_positive_in_at_least_six_families":
                counts["hold_action_positive_family_count"]
                >= threshold["positive_family_margin_count_minimum"],
        })
        failure_control = CONTROL_PHASE_A_UPDATE_100_FAIL
    else:
        threshold = PHASE_A_UPDATE_400_THRESHOLDS
        conjuncts.update({
            "centered_raw_rank_at_least_halfway_to_48":
                float(metrics["centered_raw_patch_effective_rank"])
                >= threshold["centered_raw_patch_effective_rank_minimum"],
            "centered_projected_rank_at_least_halfway_to_48":
                float(metrics["centered_projected_target_effective_rank"])
                >= threshold[
                    "centered_projected_target_effective_rank_minimum"
                ],
            "true_at_most_point975_cyclic_wrong_action":
                ratios["true_to_cyclic_wrong_action"]
                <= threshold["cyclic_wrong_action_ratio_maximum"],
            "non_hold_true_at_most_point975_hold_action":
                ratios["non_hold_true_to_hold_action"]
                <= threshold["hold_action_ratio_maximum"],
            "cyclic_wrong_action_margin_positive_in_at_least_six_families":
                counts["cyclic_wrong_action_positive_family_count"]
                >= threshold["positive_family_margin_count_minimum"],
            "hold_action_margin_positive_in_at_least_six_families":
                counts["hold_action_positive_family_count"]
                >= threshold["positive_family_margin_count_minimum"],
            "raw_cross_sample_variance_at_least_quarter_update0":
                terminal["conjuncts"][
                    "raw_cross_sample_variance_at_least_quarter_update0"
                ],
            "spatial_diversity_at_least_quarter_update0":
                terminal["conjuncts"][
                    "spatial_diversity_at_least_quarter_update0"
                ],
            "true_at_most_point90_shuffled_next":
                terminal["conjuncts"]["true_at_most_point90_shuffled_next"],
            "true_at_most_point90_mean_target":
                terminal["conjuncts"]["true_at_most_point90_mean_target"],
            "true_at_most_point95_shuffled_current":
                terminal["conjuncts"]["true_at_most_point95_shuffled_current"],
        })
        failure_control = CONTROL_PHASE_A_UPDATE_400_FAIL
    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": CONTROL_CONTINUE if passed else failure_control,
        "conjuncts": conjuncts,
        "ratios": ratios,
        "counts": counts,
        "thresholds": dict(threshold),
        "per_family": terminal["per_family"],
    }


def _finite_signed(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_phase_b(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the exact six-part terminal physical conjunction."""

    required = {
        "complete_physical_scope_count",
        "margin_count",
        "passed_margin_count",
        "total_shortfall",
        "rough_motion",
    }
    if type(evaluation) is not dict or set(evaluation) != required:
        raise ValueError("Phase-B evaluation summary changed")
    complete = evaluation["complete_physical_scope_count"]
    passed_margins = evaluation["passed_margin_count"]
    if (
        type(complete) is not int
        or not 0 <= complete <= 9
        or type(passed_margins) is not int
        or not 0 <= passed_margins <= MARGIN_COUNT
        or evaluation["margin_count"] != MARGIN_COUNT
        or type(evaluation["margin_count"]) is not int
    ):
        raise ValueError("Phase-B physical counts changed")
    rough = evaluation["rough_motion"]
    if type(rough) is not dict or set(rough) != {
        "pixel_balanced_accuracy",
        "ground_balanced_accuracy",
        "depth_p95_m",
    }:
        raise ValueError("Phase-B rough-motion summary changed")
    shortfall = _finite_nonnegative(
        evaluation["total_shortfall"],
        name="Phase-B total shortfall",
    )
    pixel = _finite_nonnegative(
        rough["pixel_balanced_accuracy"],
        name="Phase-B rough pixel balanced accuracy",
    )
    ground = _finite_nonnegative(
        rough["ground_balanced_accuracy"],
        name="Phase-B rough ground balanced accuracy",
    )
    depth = _finite_nonnegative(
        rough["depth_p95_m"],
        name="Phase-B rough depth p95",
    )
    threshold = PHASE_B_PASS_THRESHOLDS
    conjuncts = {
        "complete_physical_scope_count_at_least_1":
            complete >= threshold["complete_physical_scope_count_minimum"],
        "passed_margin_count_at_least_98":
            passed_margins >= threshold["passed_margin_count_minimum"],
        "total_shortfall_strictly_below_threshold":
            shortfall < threshold["total_shortfall_strictly_less_than"],
        "rough_pixel_balanced_accuracy_strictly_above_threshold":
            pixel
            > threshold[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ],
        "rough_ground_balanced_accuracy_strictly_above_threshold":
            ground
            > threshold[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ],
        "rough_depth_p95_strictly_below_threshold":
            depth < threshold["rough_depth_p95_m_strictly_less_than"],
    }
    passed = all(conjuncts.values())
    return {
        "passed": passed,
        "control": CONTROL_PASS if passed else CONTROL_FAIL,
        "conjuncts": conjuncts,
    }


def _read_regular_source(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required for source custody")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"source is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PermissionError(f"opened source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (
        fingerprint(before)
        == fingerprint(opened)
        == fingerprint(after_open)
        == fingerprint(after)
    ):
        raise RuntimeError(f"source changed while read: {path}")
    return b"".join(chunks)


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="source manifest")
    fields = {
        "schema",
        "status",
        "entrypoints",
        "forced_dynamic_sources",
        "excluded_runtime_categories",
        "source_paths",
        "source_bindings",
        "source_bindings_sha256",
        "source_count",
        "generated_input_open_count",
        "checkpoint_or_tensor_open_count",
        "sealed_or_heldout_open_count",
        "whole_tree_export_authorized",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("source manifest fields changed")
    paths = value["source_paths"]
    bindings = value["source_bindings"]
    if (
        value["schema"] != SOURCE_MANIFEST_SCHEMA
        or value["status"] != "PASS_SOURCE_CLOSURE"
        or value["entrypoints"] != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value["forced_dynamic_sources"]
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value["excluded_runtime_categories"]
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or type(paths) is not list
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or not set(SOURCE_PATHS).issubset(paths)
        or type(bindings) is not list
        or len(bindings) != len(paths)
        or value["source_count"] != len(paths)
        or type(value["source_count"]) is not int
        or value["source_bindings_sha256"]
        != canonical_json_sha256(bindings)
        or value["generated_input_open_count"] != 0
        or value["checkpoint_or_tensor_open_count"] != 0
        or value["sealed_or_heldout_open_count"] != 0
        or value["whole_tree_export_authorized"] is not False
        or value["authority"] != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("source manifest contract changed")
    normalized_paths: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path",
            "file_sha256",
            "byte_count",
        }:
            raise PermissionError("source binding fields changed")
        path = safe_relative_path(binding["path"], name="source path")
        parts = PurePosixPath(path).parts
        if (
            not path.endswith(".py")
            or path.endswith("sealed_test.json")
            or any(
                part in {".generated", "sealed"} or part.startswith("sealed_")
                for part in parts
            )
            or not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError(f"forbidden source path: {path}")
        normalized_paths.append(path)
    if normalized_paths != paths:
        raise PermissionError("source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        raw = _read_regular_source(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != binding["file_sha256"]
            or len(raw) != binding["byte_count"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
        result[relative] = digest
    preregistration_raw = _read_regular_source(
        root / PREREGISTRATION_RELATIVE_PATH
    )
    if (
        len(preregistration_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(preregistration_raw).hexdigest()
        != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("frozen preregistration changed")
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result[PREREGISTRATION_RELATIVE_PATH] = PREREGISTRATION_FILE_SHA256
    prior_raw = _read_regular_source(
        root / PRIOR_TERMINAL_AUDIT_RELATIVE_PATH
    )
    if (
        len(prior_raw) != PRIOR_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(prior_raw).hexdigest()
        != PRIOR_TERMINAL_AUDIT_FILE_SHA256
        or parse_canonical_json(
            prior_raw, name="prior terminal audit"
        )["content_sha256"] != PRIOR_TERMINAL_AUDIT_CONTENT_SHA256
    ):
        raise PermissionError("prior terminal audit changed")
    result[PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] = (
        PRIOR_TERMINAL_AUDIT_FILE_SHA256
    )
    return result


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "reviewed_sources",
        "source_manifest",
        "preregistration",
        "prior_terminal_audit",
        "science_contract",
        "source_only_checks",
        "scientific_checks",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("source-review fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    try:
        manifest_binding = validate_binding(
            value["source_manifest"],
            path=SOURCE_MANIFEST_RELATIVE_PATH,
        )
    except (TypeError, ValueError) as error:
        raise PermissionError("source manifest review binding changed") from error
    del manifest_binding
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_AND_SCIENCE"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH)
        != value["source_manifest"]["file_sha256"]
        or expected_sources.get(PREREGISTRATION_RELATIVE_PATH)
        != PREREGISTRATION_FILE_SHA256
        or value["preregistration"] != preregistration_binding()
        or value["prior_terminal_audit"] != prior_terminal_audit_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("source review did not pass these exact sources")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "authorizer",
        "independent_source_review",
        "preregistration",
        "runtime_inputs",
        "experiment",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution-authorization fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    try:
        expected_review = validate_binding(
            dict(review_binding),
            path=REVIEW_RELATIVE_PATH,
        )
    except (TypeError, ValueError) as error:
        raise PermissionError("source review authorization binding changed") from error
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "AUTHORIZED_ONE_EXACT_TWO_PHASE_PROBE"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["preregistration"] != preregistration_binding()
        or validate_runtime_inputs(value["runtime_inputs"])
        != value["runtime_inputs"]
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


__all__ = [
    name for name in globals()
    if name.isupper() or name in {
        "artifact_binding",
        "build_schedule_identity",
        "canonical_json_bytes",
        "canonical_json_sha256",
        "current_source_bindings",
        "evaluate_phase_a",
        "evaluate_phase_a_continuation",
        "evaluate_phase_b",
        "evaluate_physical_scopes",
        "is_sha256",
        "learning_rates",
        "parse_canonical_json",
        "phase_a_model_config",
        "physical_margins",
        "preregistration_binding",
        "prior_terminal_audit_binding",
        "runtime_authorization_template",
        "safe_relative_path",
        "science_contract",
        "validate_authorization",
        "validate_binding",
        "validate_review",
        "validate_runtime_inputs",
        "validate_schedule_identity",
        "validate_source_manifest",
        "with_content_sha256",
    }
]
