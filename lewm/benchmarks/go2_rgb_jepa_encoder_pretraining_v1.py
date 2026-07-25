"""Source-only contract for All-Candidate Correspondence JEPA V8.

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
    "lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8"
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
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8_"
    "preregistration_2026-07-25.md"
)
PREREGISTRATION_COMMIT = "2d5e3c01e363d4910f09597119393c57e7e8ca34"
PREREGISTRATION_FILE_SHA256 = (
    "3c532525fbd3109ec005bc32ad145ad1a7349a3602029ebc47177b7d986c81f7"
)
PREREGISTRATION_BYTE_COUNT = 18_744
PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_transport_"
    "jepa_v7_terminal_audit_2026-07-25.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = (
    "cf21f4a3ed2caed103a765584bcadd29284c9282"
)
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "1e284375a5d1c79419aa21c553e48a5d396c1d33b27e3a56c0e58c4dae08e28f"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "6b30ac4bb3784ea58822de7114197d184cd3a0a257ca29a60b858ab97b99c6f3"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 23_123

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8_"
    "source_manifest_2026-07-25.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8_"
    "source_review_2026-07-25.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8_"
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
    "rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_probe_v8"
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
LATENT_DIM = 192
LATENT_GRID_HEIGHT = 16
LATENT_GRID_WIDTH = 16
LATENT_GRID_TOKEN_COUNT = LATENT_GRID_HEIGHT * LATENT_GRID_WIDTH
LOCAL_CORRESPONDENCE_FULL_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
LOCAL_CORRESPONDENCE_CENTER_INDEX = 4
LOCAL_CORRESPONDENCE_NONCENTER_OFFSETS = (
    LOCAL_CORRESPONDENCE_FULL_OFFSETS[:LOCAL_CORRESPONDENCE_CENTER_INDEX]
    + LOCAL_CORRESPONDENCE_FULL_OFFSETS[
        LOCAL_CORRESPONDENCE_CENTER_INDEX + 1:
    ]
)
LOCAL_CORRESPONDENCE_NEIGHBOR_COUNT = len(
    LOCAL_CORRESPONDENCE_FULL_OFFSETS
)
TRANSPORT_OUTPUT_DIM = len(LOCAL_CORRESPONDENCE_NONCENTER_OFFSETS)
TRANSPORT_PROJECTION_SHAPE = (TRANSPORT_OUTPUT_DIM, LATENT_DIM)
TRANSPORT_PROJECTION_PARAMETER_COUNT = (
    TRANSPORT_OUTPUT_DIM * LATENT_DIM
)
LOCAL_CORRESPONDENCE_LAYER_NORM_EPSILON = 1e-5
LOCAL_CORRESPONDENCE_TARGET_LOGIT_SCALE = (
    1.0 / math.sqrt(float(LATENT_DIM))
)
LOCAL_CORRESPONDENCE_LOSS_WEIGHT = 1.0
CORRESPONDENCE_ACTION_IDENTIFICATION_LOSS_WEIGHT = 1.0
CORRESPONDENCE_ACTION_MACRO_BALANCED_ACCURACY_STRICTLY_GREATER_THAN = (
    2.0 / 9.0
)
EXPECTED_OFFSET_COMPONENT_BOUND = 1.0
PHASE_B_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX = (
    "grid_sampler_2d_backward_cuda does not have a deterministic "
    "implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'."
)
NON_HOLD_ACTION_COUNT = len(ACTION_VOCABULARY) - 1
ACTION_INDEXED_ENERGY_NLL_WEIGHT = 1.0
ACTION_ENERGY_SCALE_EPSILON = 1e-8
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
PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES = 60
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
    "hardest_wrong_action_ratio_maximum": 0.95,
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
    "hardest_wrong_action_ratio_strictly_less_than": 0.99,
    "hold_action_ratio_strictly_less_than": 0.99,
    "mean_target_ratio_strictly_less_than": 1.0,
    "positive_family_margin_count_minimum": 6,
}
PHASE_A_UPDATE_400_THRESHOLDS = {
    "centered_raw_patch_effective_rank_minimum": 37.85872936248779,
    "centered_projected_target_effective_rank_minimum": 32.71332550048828,
    "cyclic_wrong_action_ratio_maximum": 0.975,
    "hardest_wrong_action_ratio_maximum": 0.975,
    "hold_action_ratio_maximum": 0.975,
    "positive_family_margin_count_minimum": 6,
}
PHASE_A_LOCAL_CORRESPONDENCE_THRESHOLDS = {
    "correct_to_deranged_cross_entropy_ratio_strictly_less_than": 0.99,
    "deranged_positive_family_margin_count_minimum": 6,
    "executed_to_hardest_wrong_cross_entropy_ratio_strictly_less_than":
        0.99,
    "hardest_wrong_positive_family_margin_count_minimum": 6,
    "non_hold_action_distribution_different_from_hold_count_required":
        NON_HOLD_ACTION_COUNT,
    "maximum_absolute_expected_offset_component_maximum":
        EXPECTED_OFFSET_COMPONENT_BOUND,
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
    "local_correspondence",
})
PHASE_A_UPDATE0_FIELDS = frozenset({
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
    "all_action_predictions_bitwise_equal",
    "all_action_unordered_pair_count",
    "all_action_prediction_row_count",
    "local_correspondence",
})
PHASE_A_UPDATE0_HEALTH_FIELDS = frozenset({
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
})
PHASE_A_OBSERVATION_INTEGRITY_FIELDS = frozenset({
    "rng_state_preserved",
    "state_mutation_count",
})
LOCAL_CORRESPONDENCE_OBSERVATION_FIELDS = frozenset({
    "all_values_finite",
    "target_all_values_finite",
    "target_all_strictly_positive",
    "target_rows_normalized",
    "student_all_strictly_positive",
    "student_rows_normalized",
    "transport_weight_all_values_finite",
    "transport_weight_any_nonzero",
    "maximum_absolute_student_logit",
    "correct_centered_log_cross_entropy",
    "deranged_centered_log_cross_entropy",
    "correct_to_deranged_cross_entropy_ratio",
    "deranged_positive_family_margin_count",
    "per_family_deranged_minus_correct_cross_entropy",
    "per_action_correct_target_centered_log_cross_entropy",
    "hardest_wrong_centered_log_cross_entropy",
    "executed_to_hardest_wrong_cross_entropy_ratio",
    "hardest_wrong_positive_family_margin_count",
    "per_family_hardest_wrong_minus_executed_cross_entropy",
    "mean_target_kl_to_uniform",
    "per_action_probability_rows_positive_and_normalized",
    "non_hold_action_distribution_different_from_hold_count",
    "per_action_distribution_different_from_hold",
    "maximum_absolute_expected_offset_component",
    "hold_probabilities_bitwise_uniform",
    "hold_expected_offset_exactly_zero",
    "hold_transport_identity_exact",
    "all_action_distributions_bitwise_equal_to_hold",
    "all_action_distributions_bitwise_equal_to_uniform",
    "correct_and_deranged_cross_entropy_bitwise_equal",
    "all_action_transports_identity_exact",
    "unscaled_correspondence_action_nll",
    "correspondence_action_probabilities_all_values_finite",
    "correspondence_action_probability_rows_normalized",
    "correspondence_action_top1_accuracy",
    "per_executed_action_correspondence_identification",
    "correspondence_action_macro_balanced_accuracy",
    "all_candidate_correspondence_costs_bitwise_equal",
    "all_candidate_correspondence_scores_bitwise_equal",
    "correspondence_action_posterior_bitwise_equal_to_uniform",
    "correspondence_action_nll_bitwise_equal_to_zero_logit_reference",
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
    "prior_v7_terminal_audit_bound": True,
    "fresh_v8_is_not_v7_retry_resume_or_checkpoint_continuation": True,
    "v7_model_encoder_transport_and_objective_preserved_exact": True,
    "v5_v6_flow_and_inverse_tensors_removed_exact": True,
    "action_independent_zero_adaln_conditioning_exact": True,
    "existing_action_embedder_reused_by_transport_exact": True,
    "bias_free_zero_initialized_transport_projection_exact": True,
    "transport_projection_shape_8_by_192_and_count_1536_exact": True,
    "fixed_3_by_3_clamped_neighbor_table_exact": True,
    "detached_ema_local_correlation_target_exact": True,
    "parameter_free_layer_norm_and_sqrt192_target_scale_exact": True,
    "centered_nine_logit_student_parameterization_exact": True,
    "uniform_centered_residual_transport_and_shared_residual_exact": True,
    "correspondence_loss_uses_centered_log_form_and_detached_energy_scale":
        True,
    "transport_zero_and_nonzero_gradient_topology_fixtures_exact": True,
    "transport_projection_in_auxiliary_optimizer_and_excluded_from_phase_b":
        True,
    "hold_relative_action_embedding_exact": True,
    "hold_uniform_zero_offset_and_identity_transport_exact": True,
    "indexed_neighbor_reads_without_grid_sample_unfold_or_padding_exact":
        True,
    "phase_a_strict_determinism_warn_only_false_zero_warnings_exact": True,
    "all_action_update_zero_bitwise_identity_exact": True,
    "detached_row_mean_energy_scaled_cross_entropy_exact": True,
    "action_indexed_energy_nll_weight_exactly_1": True,
    "correspondence_loss_weight_exactly_1": True,
    "parameter_free_all_candidate_correspondence_identification_exact": True,
    "correspondence_action_identification_loss_weight_exactly_1": True,
    "all_candidate_cost_score_and_detach_topology_exact": True,
    "executed_label_value_invariance_fixture_exact": True,
    "correspondence_action_update_zero_equal_logit_reference_exact": True,
    "correspondence_action_nll_and_macro_balanced_accuracy_gates_exact": True,
    "wrong_action_shared_path_detached_transport_and_action_embedder_live":
        True,
    "no_hinge_fixed_temperature_margin_or_sentinel_specific_training": True,
    "patch_whitening_matches_preregistration": True,
    "all_nine_real_actions_and_hold_exact": True,
    "diagnostic_sentinels_absent_from_training": True,
    "hardest_wrong_action_gate_preserved_exact": True,
    "local_correspondence_update0_observation_exact": True,
    "local_correspondence_update100_update400_and_final_gates_exact": True,
    "target_kl_and_probability_health_gates_exact": True,
    "both_six_of_eight_correspondence_family_gates_exact": True,
    "all_eight_non_hold_distribution_activation_gate_exact": True,
    "expected_offset_bound_and_hold_invariants_exact": True,
    "within_scene_next_endpoint_derangement_exact": True,
    "update100_true_below_mean_target_gate_exact": True,
    "continuation_gates_exact": True,
    "terminal_phase_a_gate_exact": True,
    "phase_b_conditional_and_unchanged": True,
    "no_data_whitening_base_init_role_seed_schedule_or_cap_drift": True,
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
        "_lewm_jepa_encoder_v8_all_candidate_static_physical_contract",
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
            "explicit_parameter_free_all_candidate_correspondence_"
            "identification_binds_the_existing_rgb_only_v7_local_"
            "correspondence_signal_to_the_executed_action_while_all_v7_"
            "forward_jepa_gates_hold",
        "initialization": {
            "seed": BASE_INITIALIZATION_SEED,
            "n320_online_encoder_copy": True,
            "n320_ema_encoder_copy": True,
            "predictor_and_projectors_from_fixed_seed": True,
            "removed_rejected_tensors": [
                "prediction_projector.flow_weight",
                "prediction_projector.inverse_weight",
            ],
            "transport_projection": {
                "path": "prediction_projector.transport_weight",
                "shape": list(TRANSPORT_PROJECTION_SHAPE),
                "bias": False,
                "parameter_count": TRANSPORT_PROJECTION_PARAMETER_COUNT,
                "value": 0.0,
                "rng_draw_count": 0,
                "trainable_parameter_tensor_count": 1,
                "output_offsets":
                    [list(offset) for offset
                     in LOCAL_CORRESPONDENCE_NONCENTER_OFFSETS],
            },
            "existing_action_embedder": {
                "path": "predictor.action_embed",
                "reused_without_reset": True,
                "used_outside_adaln_only": True,
                "trainable": True,
                "hold_relative_self_subtraction": True,
                "used_by_transport": True,
            },
            "transport_gradient_fixtures": {
                "required_before_execution_authorization": True,
                "source_only": True,
                "zero_weight": {
                    "student_logits_exactly_zero": True,
                    "student_probabilities_bitwise_uniform": True,
                    "centered_coefficients_exactly_zero": True,
                    "transport_identity_exact": True,
                    "hold_identity_exact": True,
                    "transport_weight_gradient_finite_and_nonzero": True,
                    "shared_trunk_gradient_exactly_zero": True,
                    "online_encoder_gradient_exactly_zero": True,
                    "action_embedding_gradient_exactly_zero": True,
                    "all_candidate_correspondence_costs_bitwise_equal": True,
                    "all_candidate_correspondence_scores_bitwise_equal": True,
                    "correspondence_action_posterior_uniform_exact": True,
                    "correspondence_action_nll_equal_to_same_device_"
                    "zero_logit_reference": True,
                    "correspondence_action_identification_loss_transport_"
                    "weight_gradient_finite_and_nonzero": True,
                },
                "bitwise_nonzero_weight": {
                    "transport_weight_gradient_finite_and_nonzero": True,
                    "shared_trunk_gradient_finite_and_nonzero": True,
                    "online_state_gradient_finite_and_nonzero": True,
                    "action_embedding_gradient_finite_and_nonzero": True,
                    "ema_gradient_absent": True,
                    "correspondence_action_identification_loss_transport_"
                    "weight_gradient_finite_and_nonzero": True,
                    "wrong_candidate_online_path_gradient_absent": True,
                },
                "executed_label_value_invariance": {
                    "rgb_states_and_model_tensors_fixed": True,
                    "candidate_logits_probabilities_costs_and_scores_"
                    "bitwise_identical": True,
                    "only_cross_entropy_target_and_gradient_may_change":
                        True,
                },
                "runtime_continuation_gate": False,
            },
            "neighbor_table": {
                "shape": [LATENT_GRID_TOKEN_COUNT,
                          LOCAL_CORRESPONDENCE_NEIGHBOR_COUNT],
                "persistent": False,
                "grid_shape":
                    [LATENT_GRID_HEIGHT, LATENT_GRID_WIDTH],
                "grid_layout": "row_major",
                "full_offset_order":
                    [list(offset) for offset
                     in LOCAL_CORRESPONDENCE_FULL_OFFSETS],
                "source_index":
                    "16*clamp(y+dy,0,15)+clamp(x+dx,0,15)",
                "border_rule": "clamp_duplicate_entries",
                "runtime_geometry_input_count": 0,
            },
            "transport_zero_initialization_gradient_topology": {
                "transport_weight_gradient_finite": True,
                "transport_weight_gradient_nonzero": True,
                "shared_trunk_gradient_exactly_zero": True,
                "online_encoder_gradient_exactly_zero": True,
                "action_embedding_gradient_exactly_zero": True,
            },
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
            "global_rng_state_preserved": True,
            "rejected_checkpoint_open_count": 0,
        },
        "data": {
            "raw_v13_reused_exactly": True,
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "existing_current_next_pairs_reused_exactly": True,
            "multiple_tokens_neighbors_actions_or_encoder_calls_are_not_new_"
            "presentations": True,
            "probability_calibration_open_count": 0,
            "rebuild_refine_rebalance_filter_or_resample": False,
            "new_frame_prior_rgb_third_frame_or_render_count": 0,
            "forbidden_model_features": [
                "pose",
                "depth",
                "odometry",
                "optical_flow_label",
                "occupancy",
                "traversability",
                "physical_label",
                "navigation_label",
                "scene_family",
                "heldout",
                "sealed",
            ],
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
            "existing_pair_adapter_required": True,
            "online_current_and_ema_current_next_encoder_calls_required":
                True,
            "one_scheduled_pair_is_one_presentation": True,
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
                "all_nine_real_actions_in_frozen_vocabulary_order",
            "training_action_input":
                "executed_action_index_and_ordered_nine_candidate_energies_"
                "plus_executed_correspondence_and_all_candidate_"
                "correspondence_identification",
            "training_forbidden_diagnostic_inputs": [
                "cyclic_mapping",
                "hold_specific_mask_or_weight",
                "scene_family_identity",
                "hardest_wrong_index",
                "diagnostic_sentinel_identity",
            ],
            "acceptance_wrong_action":
                "cyclic_index_plus_one_mod_9_for_every_row",
            "observation_hold_action_population_mask":
                "requested_primitive_is_not_hold",
            "objective": {
                "ema_current_skip_stop_gradient": True,
                "ema_next_target_stop_gradient": True,
                "residual_scale": RESIDUAL_SCALE,
                "action_independent_shared_trunk": {
                    "formula": "h=H(raw_online_current,zero_condition)",
                    "action_embedder_passed_to_adaln": False,
                    "action_embedder_reused_by_transport": True,
                    "block_conditioning":
                        "same_exact_all_zero_tensor_for_all_rows_and_candidates",
                    "executed_or_candidate_action_passed_to_adaln_count": 0,
                },
                "detached_correspondence_target": {
                    "current": "zc=detach(ema_current_geometry_tokens)",
                    "next": "zn=detach(ema_next_geometry_tokens)",
                    "normalization":
                        "F.layer_norm(v,(192,),weight=None,bias=None,eps=1e-5)",
                    "layer_norm_epsilon":
                        LOCAL_CORRESPONDENCE_LAYER_NORM_EPSILON,
                    "target_logit":
                        "dot(LN(zn_i),LN(zc_J(i,o)))/sqrt(192)",
                    "target_logit_scale":
                        LOCAL_CORRESPONDENCE_TARGET_LOGIT_SCALE,
                    "target_distribution": "Q_i=softmax_o(s_target_i_o)",
                    "target_detached": True,
                    "learned_target_parameter_count": 0,
                    "mean_target_kl_to_uniform_update_zero":
                        "finite_and_strictly_positive",
                },
                "local_correspondence_transport": {
                    "shared_projector_path":
                        "prediction_projector.shared_projector",
                    "transport_projection_path":
                        "prediction_projector.transport_weight",
                    "transport_projection_shape":
                        list(TRANSPORT_PROJECTION_SHAPE),
                    "transport_projection_bias": False,
                    "transport_projection_parameter_count":
                        TRANSPORT_PROJECTION_PARAMETER_COUNT,
                    "action_embedder_path": "predictor.action_embed",
                    "hold_relative_embedding":
                        "e_rel_a=E(a)-E(hold)",
                    "state_action_interaction": "u_i_a=h_i*e_rel_a",
                    "noncenter_logits":
                        "g_noncenter=F.linear(u_i_a,transport_weight,bias=None)",
                    "center_logit":
                        "g_center=-sum_noncenter(g_noncenter)",
                    "center_full_offset_index":
                        LOCAL_CORRESPONDENCE_CENTER_INDEX,
                    "each_nine_logit_row_sum_exactly_zero": True,
                    "student_distribution": "P_i_a=softmax_o(g_i_a_o)",
                    "uniform_reference":
                        "U_i_a=softmax_o(zeros_like(g_i_a_o))",
                    "centered_coefficients": "C_i_a_o=P_i_a_o-U_i_a_o",
                    "transport":
                        "v_i_a=zc_i+sum_o(C_i_a_o*(zc_J(i,o)-zc_i))",
                    "prediction":
                        "normalize(v_i_a+(0.1/sqrt(192))*r_shared_i)",
                    "hold_uniform_exact": True,
                    "hold_expected_offset_exactly_zero": True,
                    "hold_transport_identity_exact": True,
                    "zero_weight_all_action_uniform_and_identity_exact":
                        True,
                    "expected_offset_component_closed_bound":
                        [-EXPECTED_OFFSET_COMPONENT_BOUND,
                         EXPECTED_OFFSET_COMPONENT_BOUND],
                    "per_action_transport_bank_count": 0,
                    "hidden_attention_mlp_offset_temperature_bias_occlusion_"
                    "head_count": 0,
                    "forbidden_operations": [
                        "grid_sample",
                        "unfold",
                        "differentiable_padding",
                        "materialize_B_by_9_by_256_by_9_by_192",
                    ],
                },
                "prediction":
                    "normalize(local_transport_i_a+alpha*r_shared_i)",
                "candidate_energy":
                    "E_i_a=mean_patch_feature_mse("
                    "prediction_i_a,z_next_ema_i)",
                "detached_row_energy_scale":
                    "m_i=stop_gradient(mean_a(E_i_a)).clamp_min(1e-8)",
                "detached_row_energy_scale_epsilon":
                    ACTION_ENERGY_SCALE_EPSILON,
                "action_indexed_energy_nll":
                    "mean_i(m_i*cross_entropy("
                    "-E_i_all/m_i,executed_action_i))",
                "action_indexed_energy_nll_weight":
                    ACTION_INDEXED_ENERGY_NLL_WEIGHT,
                "centered_log_soft_cross_entropy": {
                    "formula":
                        "Hc(Q,logP)=-logP_4-sum_o(Q_o*(logP_o-logP_4))",
                    "center_full_offset_index":
                        LOCAL_CORRESPONDENCE_CENTER_INDEX,
                    "executed_pair_value":
                        "CE_corr_i=mean_256_tokens(Hc(Q_i,logP_i_executed))",
                    "loss":
                        "L_CORR=mean_i(m_i*CE_corr_i)",
                    "loss_weight": LOCAL_CORRESPONDENCE_LOSS_WEIGHT,
                    "row_energy_scale_detached_and_shared_with_forward":
                        True,
                    "wrong_candidate_correspondence_training_count": 0,
                },
                "correspondence_action_identification": {
                    "target": "detach(Q_i) broadcast as Q[:,None,:,:]",
                    "candidate_transport_logits":
                        "existing_g_with_shape_B_by_9_by_256_by_9",
                    "candidate_token_cost_helper":
                        "centered_log_soft_cross_entropy(Q[:,None,:,:],g)",
                    "helper_log_softmax_count": 1,
                    "candidate_cost":
                        "C_i_a=mean_256_tokens(Hc(Q_i,logP_i_a))",
                    "candidate_score": "S_i_a=-C_i_a",
                    "unscaled_nll":
                        "mean_i(cross_entropy(S_i_all,executed_action_i))",
                    "loss":
                        "mean_i(m_i*cross_entropy("
                        "S_i_all,executed_action_i))",
                    "loss_weight":
                        CORRESPONDENCE_ACTION_IDENTIFICATION_LOSS_WEIGHT,
                    "row_energy_scale":
                        "exact_existing_detached_action_losses.row_scale",
                    "score_division_by_row_energy_scale": False,
                    "new_parameter_count": 0,
                    "new_encoder_predictor_or_transport_forward_count": 0,
                    "target_candidate_materialization_count": 0,
                    "candidate_count": len(ACTION_VOCABULARY),
                    "executed_non_hold_h_and_online_encoder_live": True,
                    "wrong_candidate_h_and_online_encoder_detached": True,
                    "hold_relative_embedding_exact_zero": True,
                    "shared_residual_or_online_target_projector_path": False,
                    "temperature_margin_bias_class_weight_label_smoothing_"
                    "or_candidate_subsampling": False,
                },
                "jepa_loss":
                    "mean_patch_feature_mse(true_prediction,z_next_ema)",
                "total_loss":
                    "jepa_loss+action_indexed_energy_nll+"
                    "local_correspondence_loss+"
                    "correspondence_action_identification_loss+"
                    "0.50*(V_raw+V_projected)+"
                    "0.02*(K_raw+K_projected)",
                "wrong_action_hinge_count": 0,
                "hold_action_hinge_count": 0,
                "fixed_temperature_or_temperature_sweep": False,
                "margin_or_sentinel_specific_training_term_count": 0,
                "executed_action_shared_h_and_r_shared_detached": False,
                "wrong_action_shared_h_and_r_shared_detached": True,
                "wrong_action_transport_projection_detached": False,
                "wrong_action_action_embedder_detached": False,
                "correspondence_target_detached": True,
                "correspondence_executed_shared_h_detached": False,
                "correspondence_derangement_observation_no_grad": True,
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
                "other_prefixes":
                    list(PHASE_A_AUXILIARY_PARAMETER_PREFIXES),
                "transport_projection": {
                    "path": "prediction_projector.transport_weight",
                    "parameter_tensor_count": 1,
                    "parameter_count": TRANSPORT_PROJECTION_PARAMETER_COUNT,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "included_in_global_clip_norm": True,
                    "phase_b_optimizer_included": False,
                },
                "frozen_compatibility_prefixes": [
                    "appearance_projector.",
                ],
                "target_parameters_excluded": True,
                "determinism": {
                    "strict_deterministic_algorithms": True,
                    "warn_only": False,
                    "expected_warning_count": 0,
                    "permitted_warning_count": 0,
                    "strict_state_restored": True,
                },
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
                    "required_rowwise_minimum_mse_over_eight_wrong_candidates",
                "update_zero_action_indexed_symmetry": {
                    "all_action_prediction_row_count": 495,
                    "all_action_unordered_pair_count": 36,
                    "all_action_predictions_bitwise_equal": True,
                },
                "local_correspondence_observation": {
                    "fields":
                        sorted(LOCAL_CORRESPONDENCE_OBSERVATION_FIELDS),
                    "action_order": list(ACTION_VOCABULARY),
                    "pair_count": 495,
                    "action_count": len(ACTION_VOCABULARY),
                    "token_count_per_pair": LATENT_GRID_TOKEN_COUNT,
                    "correct_value":
                        "mean_i(mean_token(Hc(Q_correct_i,logP_executed_i)))",
                    "deranged_value":
                        "mean_i(mean_token(Hc(Q_deranged_i,logP_executed_i)))",
                    "hardest_wrong_value":
                        "mean_i(min_eight_wrong_candidate_token_mean_Hc_i)",
                    "correct_to_deranged_ratio":
                        "mean_i(c_i)/mean_i(d_i)",
                    "executed_to_hardest_wrong_ratio":
                        "mean_i(c_i)/mean_i(h_i)",
                    "per_family_deranged_margin":
                        "mean_i_in_family(d_i-c_i)",
                    "per_family_hardest_wrong_margin":
                        "mean_i_in_family(h_i-c_i)",
                    "training_loss_alone_uses_detached_row_energy_scale":
                        True,
                    "correspondence_action_identification": {
                        "unscaled_nll_field":
                            "unscaled_correspondence_action_nll",
                        "posterior": "softmax_a(-C_i_a)",
                        "top1_tie_break":
                            "lowest_index_in_frozen_action_vocabulary",
                        "per_executed_action_fields": [
                            "row_count",
                            "mean_nll",
                            "recall",
                        ],
                        "per_executed_action_order":
                            list(ACTION_VOCABULARY),
                        "macro_balanced_accuracy":
                            "arithmetic_mean_of_nine_per_action_recalls",
                        "row_count_total": SELECTION_ROLE_COUNTS["pairs"],
                    },
                    "within_scene_next_endpoint_derangement": {
                        "group_by": "scene_id",
                        "sort_rows_by": "content_sha256",
                        "walk":
                            "cyclic_forward_from_next_sorted_position",
                        "select":
                            "first_row_with_different_next_endpoint_sha256",
                        "fixed_inputs": [
                            "current_rgb",
                            "executed_action",
                        ],
                        "replaced_input":
                            "detached_ema_next_state_used_to_construct_Q",
                        "no_grad": True,
                        "fail_if_no_distinct_endpoint": True,
                        "fail_if_selected_endpoint_unchanged": True,
                    },
                    "update_zero": {
                        "transport_weight_exact_zero": True,
                        "all_student_logits_exact_zero": True,
                        "all_action_distributions_bitwise_uniform": True,
                        "all_action_transports_identity_exact": True,
                        "correct_and_deranged_cross_entropy_bitwise_equal":
                            True,
                        "mean_target_kl_to_uniform":
                            "finite_and_strictly_positive",
                        "all_candidate_correspondence_costs_bitwise_equal":
                            True,
                        "all_candidate_correspondence_scores_bitwise_equal":
                            True,
                        "correspondence_action_posterior_bitwise_equal_to_"
                        "zeros_like_score_softmax": True,
                        "correspondence_action_nll_bitwise_equal_to_"
                        "zeros_like_score_cross_entropy": True,
                        "correspondence_action_macro_balanced_accuracy":
                            1.0 / float(len(ACTION_VOCABULARY)),
                    },
                    "update_100_update_400_and_final_thresholds":
                        dict(PHASE_A_LOCAL_CORRESPONDENCE_THRESHOLDS),
                    "update_100_update_400_and_final_correspondence_action_"
                    "identification_gates": {
                        "unscaled_nll":
                            "finite_and_strictly_below_frozen_update_zero",
                        "macro_balanced_accuracy_strictly_greater_than":
                            CORRESPONDENCE_ACTION_MACRO_BALANCED_ACCURACY_STRICTLY_GREATER_THAN,
                    },
                },
                "update_100_additional_gates": {
                    "true_pair_mse_over_mean_target_mse_strictly_less_than":
                        1.0,
                    "all_local_correspondence_gates": True,
                },
                "shuffled_current":
                    "shuffle_online_raw_current_and_matching_ema_current_skip_"
                    "together_keep_action_and_ema_next_fixed",
            },
        },
        "phase_b": {
            "entered_only_after_phase_a_pass": True,
            "copied_state": "phase_a_terminal_online_raw_encoder_only",
            "transport_projection_frozen": True,
            "transport_projection_optimizer_included": False,
            "transport_projection_copied_into_phase_b_model": False,
            "local_correspondence_training_or_metric_target_count": 0,
            "correspondence_action_identification_training_or_metric_"
            "target_count": 0,
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
            "phase_b_gpu_active_minutes": PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES,
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


def _finite_unit_interval(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be at most one")
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


def _validate_local_correspondence_observation(
    value: object,
    *,
    name: str,
    require_update_zero: bool = False,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != LOCAL_CORRESPONDENCE_OBSERVATION_FIELDS
    ):
        raise ValueError(f"{name} fields changed")
    boolean_fields = (
        "all_values_finite",
        "target_all_values_finite",
        "target_all_strictly_positive",
        "target_rows_normalized",
        "student_all_strictly_positive",
        "student_rows_normalized",
        "transport_weight_all_values_finite",
        "transport_weight_any_nonzero",
        "hold_probabilities_bitwise_uniform",
        "hold_expected_offset_exactly_zero",
        "hold_transport_identity_exact",
        "all_action_distributions_bitwise_equal_to_hold",
        "all_action_distributions_bitwise_equal_to_uniform",
        "correct_and_deranged_cross_entropy_bitwise_equal",
        "all_action_transports_identity_exact",
        "correspondence_action_probabilities_all_values_finite",
        "correspondence_action_probability_rows_normalized",
        "all_candidate_correspondence_costs_bitwise_equal",
        "all_candidate_correspondence_scores_bitwise_equal",
        "correspondence_action_posterior_bitwise_equal_to_uniform",
        "correspondence_action_nll_bitwise_equal_to_zero_logit_reference",
    )
    for field in boolean_fields:
        if type(value[field]) is not bool:
            raise TypeError(f"{name} {field} must be Boolean")
    maximum_logit = _finite_nonnegative(
        value["maximum_absolute_student_logit"],
        name=f"{name} maximum absolute student logit",
    )
    correct_ce = _finite_nonnegative(
        value["correct_centered_log_cross_entropy"],
        name=f"{name} correct centered-log cross-entropy",
    )
    deranged_ce = _finite_nonnegative(
        value["deranged_centered_log_cross_entropy"],
        name=f"{name} deranged centered-log cross-entropy",
    )
    correct_to_deranged = _positive_denominator_ratio(
        correct_ce,
        deranged_ce,
        name=f"{name} correct-to-deranged cross-entropy ratio",
    )
    reported_correct_to_deranged = _finite_nonnegative(
        value["correct_to_deranged_cross_entropy_ratio"],
        name=f"{name} reported correct-to-deranged cross-entropy ratio",
    )
    hardest_ce = _finite_nonnegative(
        value["hardest_wrong_centered_log_cross_entropy"],
        name=f"{name} hardest-wrong centered-log cross-entropy",
    )
    executed_to_hardest = _positive_denominator_ratio(
        correct_ce,
        hardest_ce,
        name=f"{name} executed-to-hardest-wrong cross-entropy ratio",
    )
    reported_executed_to_hardest = _finite_nonnegative(
        value["executed_to_hardest_wrong_cross_entropy_ratio"],
        name=f"{name} reported executed-to-hardest-wrong ratio",
    )
    if not math.isclose(
        reported_correct_to_deranged,
        correct_to_deranged,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ) or not math.isclose(
        reported_executed_to_hardest,
        executed_to_hardest,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError(f"{name} cross-entropy ratio is inconsistent")
    target_kl = _finite_nonnegative(
        value["mean_target_kl_to_uniform"],
        name=f"{name} mean target KL to uniform",
    )
    maximum_expected_offset = _finite_nonnegative(
        value["maximum_absolute_expected_offset_component"],
        name=f"{name} maximum absolute expected-offset component",
    )
    correspondence_action_nll = _finite_nonnegative(
        value["unscaled_correspondence_action_nll"],
        name=f"{name} unscaled correspondence-action NLL",
    )
    correspondence_action_top1 = _finite_unit_interval(
        value["correspondence_action_top1_accuracy"],
        name=f"{name} correspondence-action top-1 accuracy",
    )
    correspondence_action_macro = _finite_unit_interval(
        value["correspondence_action_macro_balanced_accuracy"],
        name=f"{name} correspondence-action macro balanced accuracy",
    )
    per_executed_action = value[
        "per_executed_action_correspondence_identification"
    ]
    if (
        type(per_executed_action) is not dict
        or tuple(per_executed_action) != ACTION_VOCABULARY
    ):
        raise ValueError(
            f"{name} executed-action identification order changed"
        )
    normalized_identification: dict[str, dict[str, int | float]] = {}
    total_rows = 0
    weighted_nll = 0.0
    weighted_correct = 0.0
    recalls: list[float] = []
    for action in ACTION_VOCABULARY:
        row = per_executed_action[action]
        if type(row) is not dict or set(row) != {
            "row_count",
            "mean_nll",
            "recall",
        }:
            raise ValueError(
                f"{name} {action} identification fields changed"
            )
        row_count = row["row_count"]
        if type(row_count) is not int or row_count <= 0:
            raise ValueError(
                f"{name} {action} row count must be a positive integer"
            )
        mean_nll = _finite_nonnegative(
            row["mean_nll"],
            name=f"{name} {action} mean NLL",
        )
        recall = _finite_unit_interval(
            row["recall"],
            name=f"{name} {action} recall",
        )
        correct_count = round(float(row_count) * recall)
        if not math.isclose(
            float(row_count) * recall,
            float(correct_count),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name} {action} recall is not an integer-count ratio"
            )
        normalized_identification[action] = {
            "row_count": row_count,
            "mean_nll": mean_nll,
            "recall": recall,
        }
        total_rows += row_count
        weighted_nll += float(row_count) * mean_nll
        weighted_correct += float(row_count) * recall
        recalls.append(recall)
    if total_rows != SELECTION_ROLE_COUNTS["pairs"]:
        raise ValueError(f"{name} executed-action row total changed")
    recomputed_nll = weighted_nll / float(total_rows)
    recomputed_top1 = weighted_correct / float(total_rows)
    recomputed_macro = sum(recalls) / float(len(recalls))
    if (
        not math.isclose(
            correspondence_action_nll,
            recomputed_nll,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            correspondence_action_top1,
            recomputed_top1,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            correspondence_action_macro,
            recomputed_macro,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    ):
        raise ValueError(
            f"{name} correspondence-action aggregations are inconsistent"
        )
    if (
        value["all_values_finite"]
        and not value[
            "correspondence_action_probabilities_all_values_finite"
        ]
    ):
        raise ValueError(
            f"{name} all-values-finite omits correspondence-action values"
        )

    per_family_deranged = value[
        "per_family_deranged_minus_correct_cross_entropy"
    ]
    per_family_hardest = value[
        "per_family_hardest_wrong_minus_executed_cross_entropy"
    ]
    if (
        type(per_family_deranged) is not dict
        or tuple(per_family_deranged) != SCENE_FAMILIES
        or type(per_family_hardest) is not dict
        or tuple(per_family_hardest) != SCENE_FAMILIES
    ):
        raise ValueError(f"{name} scene-family correspondence order changed")
    normalized_deranged = {
        family: _finite_signed(
            per_family_deranged[family],
            name=f"{name} {family} deranged-minus-correct cross-entropy",
        )
        for family in SCENE_FAMILIES
    }
    normalized_hardest = {
        family: _finite_signed(
            per_family_hardest[family],
            name=f"{name} {family} hardest-wrong-minus-executed "
                 "cross-entropy",
        )
        for family in SCENE_FAMILIES
    }
    deranged_positive_count = value[
        "deranged_positive_family_margin_count"
    ]
    hardest_positive_count = value[
        "hardest_wrong_positive_family_margin_count"
    ]
    expected_deranged_positive_count = sum(
        int(margin > 0.0) for margin in normalized_deranged.values()
    )
    expected_hardest_positive_count = sum(
        int(margin > 0.0) for margin in normalized_hardest.values()
    )
    if (
        type(deranged_positive_count) is not int
        or deranged_positive_count != expected_deranged_positive_count
        or type(hardest_positive_count) is not int
        or hardest_positive_count != expected_hardest_positive_count
    ):
        raise ValueError(
            f"{name} positive correspondence-family count is inconsistent"
        )

    per_action_ce = value[
        "per_action_correct_target_centered_log_cross_entropy"
    ]
    per_action_health = value[
        "per_action_probability_rows_positive_and_normalized"
    ]
    per_action_different = value[
        "per_action_distribution_different_from_hold"
    ]
    for field, mapping in (
        ("candidate cross-entropy", per_action_ce),
        ("probability health", per_action_health),
        ("hold difference", per_action_different),
    ):
        if type(mapping) is not dict or tuple(mapping) != ACTION_VOCABULARY:
            raise ValueError(f"{name} per-action {field} order changed")
    normalized_action_ce = {
        action: _finite_nonnegative(
            per_action_ce[action],
            name=f"{name} {action} correct-target cross-entropy",
        )
        for action in ACTION_VOCABULARY
    }
    if (
        any(type(item) is not bool for item in per_action_health.values())
        or any(type(item) is not bool for item in per_action_different.values())
        or per_action_different["hold"]
    ):
        raise ValueError(f"{name} per-action correspondence receipt changed")
    active_count = value[
        "non_hold_action_distribution_different_from_hold_count"
    ]
    observed_active_count = sum(
        int(per_action_different[action])
        for action in ACTION_VOCABULARY
        if action != "hold"
    )
    if (
        type(active_count) is not int
        or active_count != observed_active_count
        or not 0 <= active_count <= NON_HOLD_ACTION_COUNT
    ):
        raise ValueError(f"{name} non-hold activation count is inconsistent")

    if correct_ce <= 0.0:
        raise ValueError(f"{name} correct cross-entropy must be positive")
    if require_update_zero and (
        not value["all_values_finite"]
        or not value["target_all_values_finite"]
        or not value["target_all_strictly_positive"]
        or not value["target_rows_normalized"]
        or not value["student_all_strictly_positive"]
        or not value["student_rows_normalized"]
        or not value["transport_weight_all_values_finite"]
        or value["transport_weight_any_nonzero"]
        or maximum_logit != 0.0
        or correct_ce != deranged_ce
        or reported_correct_to_deranged != 1.0
        or hardest_ce != correct_ce
        or reported_executed_to_hardest != 1.0
        or target_kl <= 0.0
        or deranged_positive_count != 0
        or hardest_positive_count != 0
        or any(margin != 0.0 for margin in normalized_deranged.values())
        or any(margin != 0.0 for margin in normalized_hardest.values())
        or any(item != correct_ce for item in normalized_action_ce.values())
        or not all(per_action_health.values())
        or active_count != 0
        or any(per_action_different.values())
        or maximum_expected_offset != 0.0
        or not value["hold_probabilities_bitwise_uniform"]
        or not value["hold_expected_offset_exactly_zero"]
        or not value["hold_transport_identity_exact"]
        or not value["all_action_distributions_bitwise_equal_to_hold"]
        or not value["all_action_distributions_bitwise_equal_to_uniform"]
        or not value["correct_and_deranged_cross_entropy_bitwise_equal"]
        or not value["all_action_transports_identity_exact"]
        or not value[
            "correspondence_action_probabilities_all_values_finite"
        ]
        or not value[
            "correspondence_action_probability_rows_normalized"
        ]
        or not value[
            "all_candidate_correspondence_costs_bitwise_equal"
        ]
        or not value[
            "all_candidate_correspondence_scores_bitwise_equal"
        ]
        or not value[
            "correspondence_action_posterior_bitwise_equal_to_uniform"
        ]
        or not value[
            "correspondence_action_nll_bitwise_equal_to_zero_logit_reference"
        ]
        or correspondence_action_macro
        != 1.0 / float(len(ACTION_VOCABULARY))
        or normalized_identification[ACTION_VOCABULARY[0]]["recall"]
        != 1.0
        or any(
            normalized_identification[action]["recall"] != 0.0
            for action in ACTION_VOCABULARY[1:]
        )
        or any(
            not math.isclose(
                float(row["mean_nll"]),
                correspondence_action_nll,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
            for row in normalized_identification.values()
        )
        or correspondence_action_top1
        != (
            float(
                normalized_identification[ACTION_VOCABULARY[0]][
                    "row_count"
                ]
            )
            / float(SELECTION_ROLE_COUNTS["pairs"])
        )
    ):
        raise ValueError(
            f"{name} update-zero local-correspondence receipt changed"
        )
    return {
        "all_values_finite": value["all_values_finite"],
        "target_all_values_finite": value["target_all_values_finite"],
        "target_all_strictly_positive":
            value["target_all_strictly_positive"],
        "target_rows_normalized": value["target_rows_normalized"],
        "student_all_strictly_positive":
            value["student_all_strictly_positive"],
        "student_rows_normalized": value["student_rows_normalized"],
        "transport_weight_all_values_finite":
            value["transport_weight_all_values_finite"],
        "transport_weight_any_nonzero":
            value["transport_weight_any_nonzero"],
        "maximum_absolute_student_logit": maximum_logit,
        "correct_centered_log_cross_entropy": correct_ce,
        "deranged_centered_log_cross_entropy": deranged_ce,
        "correct_to_deranged_cross_entropy_ratio":
            reported_correct_to_deranged,
        "deranged_positive_family_margin_count":
            deranged_positive_count,
        "per_family_deranged_minus_correct_cross_entropy":
            normalized_deranged,
        "per_action_correct_target_centered_log_cross_entropy":
            normalized_action_ce,
        "hardest_wrong_centered_log_cross_entropy": hardest_ce,
        "executed_to_hardest_wrong_cross_entropy_ratio":
            reported_executed_to_hardest,
        "hardest_wrong_positive_family_margin_count":
            hardest_positive_count,
        "per_family_hardest_wrong_minus_executed_cross_entropy":
            normalized_hardest,
        "mean_target_kl_to_uniform": target_kl,
        "per_action_probability_rows_positive_and_normalized":
            dict(per_action_health),
        "non_hold_action_distribution_different_from_hold_count":
            active_count,
        "per_action_distribution_different_from_hold":
            dict(per_action_different),
        "maximum_absolute_expected_offset_component":
            maximum_expected_offset,
        "hold_probabilities_bitwise_uniform":
            value["hold_probabilities_bitwise_uniform"],
        "hold_expected_offset_exactly_zero":
            value["hold_expected_offset_exactly_zero"],
        "hold_transport_identity_exact":
            value["hold_transport_identity_exact"],
        "all_action_distributions_bitwise_equal_to_hold":
            value["all_action_distributions_bitwise_equal_to_hold"],
        "all_action_distributions_bitwise_equal_to_uniform":
            value["all_action_distributions_bitwise_equal_to_uniform"],
        "correct_and_deranged_cross_entropy_bitwise_equal":
            value["correct_and_deranged_cross_entropy_bitwise_equal"],
        "all_action_transports_identity_exact":
            value["all_action_transports_identity_exact"],
        "unscaled_correspondence_action_nll": correspondence_action_nll,
        "correspondence_action_probabilities_all_values_finite":
            value[
                "correspondence_action_probabilities_all_values_finite"
            ],
        "correspondence_action_probability_rows_normalized":
            value["correspondence_action_probability_rows_normalized"],
        "correspondence_action_top1_accuracy": correspondence_action_top1,
        "per_executed_action_correspondence_identification":
            normalized_identification,
        "correspondence_action_macro_balanced_accuracy":
            correspondence_action_macro,
        "all_candidate_correspondence_costs_bitwise_equal":
            value["all_candidate_correspondence_costs_bitwise_equal"],
        "all_candidate_correspondence_scores_bitwise_equal":
            value["all_candidate_correspondence_scores_bitwise_equal"],
        "correspondence_action_posterior_bitwise_equal_to_uniform":
            value[
                "correspondence_action_posterior_bitwise_equal_to_uniform"
            ],
        "correspondence_action_nll_bitwise_equal_to_zero_logit_reference":
            value[
                "correspondence_action_nll_bitwise_equal_to_"
                "zero_logit_reference"
            ],
    }


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
    if (
        type(update0_metrics["all_action_predictions_bitwise_equal"]) is not bool
        or type(update0_metrics["all_action_unordered_pair_count"]) is not int
        or update0_metrics["all_action_unordered_pair_count"] != 36
        or type(update0_metrics["all_action_prediction_row_count"]) is not int
        or update0_metrics["all_action_prediction_row_count"] != 495
    ):
        raise ValueError("Phase-A update-zero action symmetry receipt changed")
    update0_correspondence = _validate_local_correspondence_observation(
        update0_metrics["local_correspondence"],
        name="Phase-A update-zero local correspondence",
        require_update_zero=True,
    )
    correspondence = _validate_local_correspondence_observation(
        metrics["local_correspondence"],
        name="Phase-A local correspondence",
    )
    for action in ACTION_VOCABULARY:
        if (
            correspondence[
                "per_executed_action_correspondence_identification"
            ][action]["row_count"]
            != update0_correspondence[
                "per_executed_action_correspondence_identification"
            ][action]["row_count"]
        ):
            raise ValueError(
                "Phase-A correspondence-action populations changed"
            )
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
        "local_correspondence",
    }
    values = {
        name: _finite_nonnegative(metrics[name], name=name)
        for name in numeric_names
    }
    update0 = {
        name: _finite_nonnegative(update0_metrics[name], name=f"update0 {name}")
        for name in PHASE_A_UPDATE0_HEALTH_FIELDS
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
        "true_to_hardest_wrong_action": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["hardest_wrong_action_mse"],
            name="hardest-wrong-action ratio",
        ),
        "true_to_shuffled_current": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_current_mse"],
            name="shuffled-current ratio",
        ),
        "local_correspondence_correct_to_deranged_cross_entropy":
            correspondence["correct_to_deranged_cross_entropy_ratio"],
        "local_correspondence_executed_to_hardest_wrong_cross_entropy":
            correspondence[
                "executed_to_hardest_wrong_cross_entropy_ratio"
            ],
        "correspondence_action_nll_to_update_zero":
            _positive_denominator_ratio(
                correspondence["unscaled_correspondence_action_nll"],
                update0_correspondence[
                    "unscaled_correspondence_action_nll"
                ],
                name="correspondence-action update-zero NLL ratio",
            ),
    }
    threshold = PHASE_A_PASS_THRESHOLDS
    correspondence_threshold = PHASE_A_LOCAL_CORRESPONDENCE_THRESHOLDS
    conjuncts = {
        "diagnostic_rng_and_model_state_preserved":
            observation_integrity["rng_state_preserved"]
            and observation_integrity["state_mutation_count"] == 0,
        "update_zero_all_action_predictions_bitwise_equal":
            update0_metrics["all_action_predictions_bitwise_equal"],
        "update_zero_local_correspondence_uniform_identity_and_target_viable":
            not update0_correspondence["transport_weight_any_nonzero"]
            and update0_correspondence[
                "maximum_absolute_student_logit"
            ] == 0.0
            and update0_correspondence[
                "all_action_distributions_bitwise_equal_to_hold"
            ]
            and update0_correspondence[
                "all_action_distributions_bitwise_equal_to_uniform"
            ]
            and update0_correspondence[
                "correct_and_deranged_cross_entropy_bitwise_equal"
            ]
            and update0_correspondence[
                "all_action_transports_identity_exact"
            ]
            and update0_correspondence["mean_target_kl_to_uniform"] > 0.0,
        "update_zero_correspondence_action_identification_uniform_exact":
            update0_correspondence[
                "all_candidate_correspondence_costs_bitwise_equal"
            ]
            and update0_correspondence[
                "all_candidate_correspondence_scores_bitwise_equal"
            ]
            and update0_correspondence[
                "correspondence_action_posterior_bitwise_equal_to_uniform"
            ]
            and update0_correspondence[
                "correspondence_action_nll_bitwise_equal_to_"
                "zero_logit_reference"
            ]
            and update0_correspondence[
                "correspondence_action_macro_balanced_accuracy"
            ]
            == 1.0 / float(len(ACTION_VOCABULARY)),
        "finite_and_ema_gradient_free":
            metrics["all_values_finite"]
            and metrics["ema_target_gradient_free"],
        "local_correspondence_values_finite_positive_and_normalized":
            correspondence["all_values_finite"]
            and correspondence["target_all_values_finite"]
            and correspondence["target_all_strictly_positive"]
            and correspondence["target_rows_normalized"]
            and correspondence["student_all_strictly_positive"]
            and correspondence["student_rows_normalized"]
            and correspondence["transport_weight_all_values_finite"]
            and correspondence[
                "correspondence_action_probabilities_all_values_finite"
            ]
            and correspondence[
                "correspondence_action_probability_rows_normalized"
            ]
            and all(
                correspondence[
                    "per_action_probability_rows_positive_and_normalized"
                ].values()
            ),
        "finite_unscaled_correspondence_action_nll_strictly_below_frozen_"
        "update_zero_log9":
            correspondence["unscaled_correspondence_action_nll"]
            < update0_correspondence[
                "unscaled_correspondence_action_nll"
            ],
        "correspondence_action_identification_macro_balanced_accuracy_"
        "strictly_above_two_ninths":
            correspondence[
                "correspondence_action_macro_balanced_accuracy"
            ]
            > CORRESPONDENCE_ACTION_MACRO_BALANCED_ACCURACY_STRICTLY_GREATER_THAN,
        "transport_weight_finite_and_bitwise_nonzero":
            correspondence["transport_weight_all_values_finite"]
            and correspondence["transport_weight_any_nonzero"],
        "correct_correspondence_cross_entropy_strictly_below_update_zero":
            correspondence["correct_centered_log_cross_entropy"]
            < update0_correspondence[
                "correct_centered_log_cross_entropy"
            ],
        "correct_to_deranged_correspondence_ratio_strictly_below_point99":
            correspondence["correct_to_deranged_cross_entropy_ratio"]
            < correspondence_threshold[
                "correct_to_deranged_cross_entropy_ratio_"
                "strictly_less_than"
            ],
        "deranged_correspondence_margin_positive_in_at_least_six_families":
            correspondence["deranged_positive_family_margin_count"]
            >= correspondence_threshold[
                "deranged_positive_family_margin_count_minimum"
            ],
        "executed_to_hardest_wrong_correspondence_ratio_below_point99":
            correspondence[
                "executed_to_hardest_wrong_cross_entropy_ratio"
            ]
            < correspondence_threshold[
                "executed_to_hardest_wrong_cross_entropy_ratio_"
                "strictly_less_than"
            ],
        "hardest_wrong_correspondence_margin_positive_in_six_families":
            correspondence[
                "hardest_wrong_positive_family_margin_count"
            ]
            >= correspondence_threshold[
                "hardest_wrong_positive_family_margin_count_minimum"
            ],
        "all_eight_non_hold_distributions_differ_from_hold":
            correspondence[
                "non_hold_action_distribution_different_from_hold_count"
            ]
            == correspondence_threshold[
                "non_hold_action_distribution_different_from_hold_count_"
                "required"
            ],
        "hold_uniform_zero_offset_and_identity_transport_exact":
            correspondence["hold_probabilities_bitwise_uniform"]
            and correspondence["hold_expected_offset_exactly_zero"]
            and correspondence["hold_transport_identity_exact"],
        "expected_offset_components_within_closed_unit_bound":
            correspondence[
                "maximum_absolute_expected_offset_component"
            ]
            <= correspondence_threshold[
                "maximum_absolute_expected_offset_component_maximum"
            ],
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
        "true_at_most_point95_hardest_wrong_action":
            ratios["true_to_hardest_wrong_action"]
            <= threshold["hardest_wrong_action_ratio_maximum"],
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
            "non_hold_action_distribution_different_from_hold_count":
                correspondence[
                    "non_hold_action_distribution_different_from_hold_count"
                ],
            "correspondence_deranged_positive_family_margin_count":
                correspondence["deranged_positive_family_margin_count"],
            "correspondence_hardest_wrong_positive_family_margin_count":
                correspondence[
                    "hardest_wrong_positive_family_margin_count"
                ],
        },
        "per_family": normalized_families,
        "local_correspondence": correspondence,
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
        "update_zero_all_action_predictions_bitwise_equal":
            terminal["conjuncts"][
                "update_zero_all_action_predictions_bitwise_equal"
            ],
        "update_zero_local_correspondence_uniform_identity_and_target_viable":
            terminal["conjuncts"][
                "update_zero_local_correspondence_uniform_identity_and_"
                "target_viable"
            ],
        "update_zero_correspondence_action_identification_uniform_exact":
            terminal["conjuncts"][
                "update_zero_correspondence_action_identification_uniform_"
                "exact"
            ],
        "finite_and_ema_gradient_free":
            terminal["conjuncts"]["finite_and_ema_gradient_free"],
        "local_correspondence_values_finite_positive_and_normalized":
            terminal["conjuncts"][
                "local_correspondence_values_finite_positive_and_normalized"
            ],
        "finite_unscaled_correspondence_action_nll_strictly_below_frozen_"
        "update_zero_log9":
            terminal["conjuncts"][
                "finite_unscaled_correspondence_action_nll_strictly_below_"
                "frozen_update_zero_log9"
            ],
        "correspondence_action_identification_macro_balanced_accuracy_"
        "strictly_above_two_ninths":
            terminal["conjuncts"][
                "correspondence_action_identification_macro_balanced_"
                "accuracy_strictly_above_two_ninths"
            ],
        "transport_weight_finite_and_bitwise_nonzero":
            terminal["conjuncts"][
                "transport_weight_finite_and_bitwise_nonzero"
            ],
        "correct_correspondence_cross_entropy_strictly_below_update_zero":
            terminal["conjuncts"][
                "correct_correspondence_cross_entropy_strictly_below_update_"
                "zero"
            ],
        "correct_to_deranged_correspondence_ratio_strictly_below_point99":
            terminal["conjuncts"][
                "correct_to_deranged_correspondence_ratio_strictly_below_"
                "point99"
            ],
        "deranged_correspondence_margin_positive_in_at_least_six_families":
            terminal["conjuncts"][
                "deranged_correspondence_margin_positive_in_at_least_six_"
                "families"
            ],
        "executed_to_hardest_wrong_correspondence_ratio_below_point99":
            terminal["conjuncts"][
                "executed_to_hardest_wrong_correspondence_ratio_below_"
                "point99"
            ],
        "hardest_wrong_correspondence_margin_positive_in_six_families":
            terminal["conjuncts"][
                "hardest_wrong_correspondence_margin_positive_in_six_"
                "families"
            ],
        "all_eight_non_hold_distributions_differ_from_hold":
            terminal["conjuncts"][
                "all_eight_non_hold_distributions_differ_from_hold"
            ],
        "hold_uniform_zero_offset_and_identity_transport_exact":
            terminal["conjuncts"][
                "hold_uniform_zero_offset_and_identity_transport_exact"
            ],
        "expected_offset_components_within_closed_unit_bound":
            terminal["conjuncts"][
                "expected_offset_components_within_closed_unit_bound"
            ],
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
            "true_strictly_below_point99_hardest_wrong_action":
                ratios["true_to_hardest_wrong_action"]
                < threshold[
                    "hardest_wrong_action_ratio_strictly_less_than"
                ],
            "non_hold_true_strictly_below_point99_hold_action":
                ratios["non_hold_true_to_hold_action"]
                < threshold["hold_action_ratio_strictly_less_than"],
            "true_strictly_below_mean_target":
                ratios["true_to_mean_target"]
                < threshold["mean_target_ratio_strictly_less_than"],
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
            "true_at_most_point975_hardest_wrong_action":
                ratios["true_to_hardest_wrong_action"]
                <= threshold["hardest_wrong_action_ratio_maximum"],
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
        "local_correspondence": terminal["local_correspondence"],
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
        or expected_sources.get(PRIOR_TERMINAL_AUDIT_RELATIVE_PATH)
        != PRIOR_TERMINAL_AUDIT_FILE_SHA256
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
