"""Source-only contract for Dense Pairwise Spatial Cost-Volume JEPA V9.

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
    "lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9"
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
    "docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9_"
    "preregistration_2026-07-25.md"
)
PREREGISTRATION_COMMIT = "b775093897669c91d8c1b9e7d148e257881bcedf"
PREREGISTRATION_FILE_SHA256 = (
    "bfb0f1c2bd77ee78f6d4bf34cff8ec8d3d3c4bced8fb7b4269fa0a3f0bb30f2b"
)
PREREGISTRATION_BYTE_COUNT = 22_115
PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
    "all_candidate_identification_jepa_v8_terminal_audit_2026-07-25.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = (
    "9f3e2bc96a6e4ea419574f109c890299d0608659"
)
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "3ea4a8cc4405b0880d2e05217e4b4acefc5b9df5fad9bcdd9a682db42e273173"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "ff8339aa6109933e85d60ad118dc912fd091dddf7dfd80b18d00453ce7c01367"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 20_028

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9_"
    "source_manifest_2026-07-25.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9_"
    "source_review_2026-07-25.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9_"
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
    "rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_probe_v9"
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
FLOW_OUTPUT_DIM = 2
FLOW_PROJECTION_SHAPE = (FLOW_OUTPUT_DIM, LATENT_DIM)
FLOW_PROJECTION_PARAMETER_COUNT = FLOW_OUTPUT_DIM * LATENT_DIM
FLOW_CELL_BOUND = 1.0
FLOW_GRID_SCALE = 2.0 / float(LATENT_GRID_WIDTH - 1)
FLOW_X_COMPONENT_INDEX = 0
FLOW_Y_COMPONENT_INDEX = 1
FLOW_GRID_SAMPLE_MODE = "bilinear"
FLOW_GRID_SAMPLE_PADDING_MODE = "border"
FLOW_GRID_SAMPLE_ALIGN_CORNERS = True
PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX = (
    "grid_sampler_2d_backward_cuda does not have a deterministic "
    "implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'."
)
DENSE_PAIRWISE_HEAD_INITIALIZATION_SEED = 20260725
DENSE_PAIRWISE_LAYER_NORM_EPSILON = 1e-5
DENSE_PAIRWISE_COST_SCALE = 1.0 / math.sqrt(float(LATENT_DIM))
DENSE_PAIRWISE_VOLUME_INPUT_CHANNELS = LATENT_GRID_TOKEN_COUNT
DENSE_PAIRWISE_HEAD_HIDDEN_CHANNELS = 16
DENSE_PAIRWISE_HEAD_OUTPUT_COUNT = len(ACTION_VOCABULARY)
DENSE_PAIRWISE_HEAD_PARAMETER_COUNT = 8_713
DENSE_PAIRWISE_INVERSE_LOSS_WEIGHT = 1.0
DENSE_PAIRWISE_VOLUME_VALUE_BOUND = 1.0
DENSE_PAIRWISE_DISPLACEMENT_COMPONENT_BOUND = 2.0
DENSE_PAIRWISE_VOLUME_CONSERVATION_ATOL = 1e-6
DENSE_PAIRWISE_UPDATE_100_NLL_LOG9_FACTOR = 0.98
DENSE_PAIRWISE_MACRO_BALANCED_ACCURACY_STRICTLY_GREATER_THAN = 2.0 / 9.0
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
    "dense_inverse_nll_log9_factor_strictly_less_than":
        DENSE_PAIRWISE_UPDATE_100_NLL_LOG9_FACTOR,
    "dense_inverse_macro_balanced_accuracy_strictly_greater_than":
        DENSE_PAIRWISE_MACRO_BALANCED_ACCURACY_STRICTLY_GREATER_THAN,
    "correct_to_deranged_nll_ratio_strictly_less_than": 0.99,
    "non_hold_correct_to_current_current_nll_ratio_strictly_less_than":
        0.99,
    "deranged_positive_family_margin_count_minimum": 6,
    "centered_raw_patch_effective_rank_strictly_greater_than":
        27.717458724975586,
    "centered_projected_target_effective_rank_strictly_greater_than":
        17.426651000976562,
    "update_zero_health_fraction_minimum": 0.25,
    "shuffled_next_ratio_maximum": 0.90,
    "shuffled_current_ratio_maximum": 0.95,
    "non_hold_action_nonzero_flow_count_required": NON_HOLD_ACTION_COUNT,
}
PHASE_A_UPDATE_400_THRESHOLDS = {
    "centered_raw_patch_effective_rank_minimum": 37.85872936248779,
    "centered_projected_target_effective_rank_minimum": 32.71332550048828,
    "cyclic_wrong_action_ratio_strictly_less_than": 0.99,
    "hardest_wrong_action_ratio_strictly_less_than": 0.99,
    "hold_action_ratio_strictly_less_than": 0.99,
    "mean_target_ratio_strictly_less_than": 1.0,
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
    "dense_pairwise_inverse_head.",
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
    "latent_flow",
    "dense_pairwise_inverse",
})
PHASE_A_UPDATE0_FIELDS = frozenset({
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
    "all_action_predictions_bitwise_equal",
    "all_action_unordered_pair_count",
    "all_action_prediction_row_count",
    "latent_flow",
    "dense_pairwise_inverse",
})
PHASE_A_UPDATE0_HEALTH_FIELDS = frozenset({
    "raw_cross_sample_variance",
    "content_residual_spatial_diversity",
})
PHASE_A_OBSERVATION_INTEGRITY_FIELDS = frozenset({
    "rng_state_preserved",
    "state_mutation_count",
})
DENSE_PAIRWISE_INVERSE_OBSERVATION_FIELDS = frozenset({
    "all_values_finite",
    "probabilities_all_values_finite",
    "probability_rows_normalized",
    "volume_all_values_finite",
    "volume_values_within_closed_unit_interval",
    "volume_channel_conservation",
    "displacement_all_values_finite",
    "displacement_values_within_closed_two_bound",
    "maximum_absolute_displacement_component",
    "cross_pair_displacement_rms",
    "cross_pair_displacement_value_count",
    "same_tensor_diff_exact_zero",
    "same_tensor_volume_exact_zero",
    "same_tensor_displacement_exact_zero",
    "head_parameters_all_values_finite",
    "head_parameter_count",
    "head_weight_tensors_all_nonzero",
    "zero_logit_reference_nll",
    "unscaled_dense_inverse_nll",
    "dense_inverse_top1_accuracy",
    "per_executed_action_dense_inverse",
    "dense_inverse_macro_balanced_accuracy",
    "correct_pair_nll",
    "correct_pair_count",
    "deranged_next_nll",
    "deranged_next_pair_count",
    "correct_to_deranged_nll_ratio",
    "non_hold_correct_pair_nll",
    "non_hold_correct_pair_count",
    "non_hold_current_current_nll",
    "non_hold_current_current_pair_count",
    "non_hold_correct_to_current_current_nll_ratio",
    "deranged_positive_family_margin_count",
    "per_family_deranged_minus_correct_nll",
})
LATENT_FLOW_OBSERVATION_FIELDS = frozenset({
    "all_values_finite",
    "all_components_within_closed_one_patch_bound",
    "hold_flow_exactly_zero",
    "maximum_absolute_flow_cell",
    "non_hold_action_nonzero_count",
    "per_action_any_nonzero",
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

ATTEMPT_INDEX = 1
MAXIMUM_ATTEMPTS = 1
NORMAL_PHASE_A_RECEIPT_PATHS = (
    "reservation.json",
    "phase_a/metrics.json",
    "phase_a/artifact.json",
    "access.json",
    "result.json",
    "completed.json",
)
PHASE_B_RECEIPT_PATHS = (
    "phase_b/metrics.json",
    "phase_b/artifact.json",
)
OPERATIONAL_FAILURE_RECEIPT_PATHS = (
    "failure.json",
    "completed.json",
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)
RESERVATION_PUBLICATION_FAILURE_STATUS = (
    "TERMINAL_RESERVATION_PUBLICATION_FAILURE"
)
OPERATIONAL_FAILURE_COMPLETION_STATUS = "TERMINAL_FAILURE"
PHASE_A_TERMINAL_CONTROLS = (
    CONTROL_PHASE_A_UPDATE_100_FAIL,
    CONTROL_PHASE_A_UPDATE_400_FAIL,
    CONTROL_PHASE_A_FAIL,
    CONTROL_PHASE_A_PASS,
)
PHASE_A_FAILURE_CONTROLS = PHASE_A_TERMINAL_CONTROLS[:-1]
PHASE_B_METRICS_STATUSES = ("PASS_PHASE_B", "FAIL_PHASE_B_TERMINAL")
PHASE_B_ARTIFACT_STATUSES = (
    "PASS_FROZEN_ENCODER_PHYSICAL_PROBE",
    "FAIL_FROZEN_ENCODER_PHYSICAL_PROBE_TERMINAL",
)
RESULT_TERMINAL_STATUSES = (
    "PASS_BOUNDED_FALSIFICATION_SEPARATE_QUALIFICATION_ONLY",
    "FAIL_PHASE_B_MECHANISM_TERMINATED",
    *PHASE_A_FAILURE_CONTROLS,
)
COMPLETION_TERMINAL_STATUSES = (
    *PHASE_A_FAILURE_CONTROLS,
    "TERMINAL_PASS",
    "TERMINAL_FAIL",
)
ACCESS_ZERO_COUNTER_FIELDS = (
    "probability_calibration_open_count",
    "prior_runtime_output_open_count",
    "rejected_checkpoint_open_count",
    "phase_a_camera_supervision_array_open_count",
    "phase_a_general_raw_loader_call_count",
    "g2_open_count",
    "navigation_open_count",
    "heldout_open_count",
    "sealed_open_count",
    "production_input_open_count",
    "deployment_input_open_count",
    "observer_rerun_count",
)

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
    "prior_v8_terminal_audit_bound": True,
    "fresh_v9_is_not_v8_retry_resume_or_checkpoint_continuation": True,
    "exact_reviewed_v5_forward_base_restored": True,
    "v6_v7_v8_inverse_and_local_correspondence_tensors_removed_exact": True,
    "action_independent_zero_adaln_conditioning_exact": True,
    "v5_state_dependent_bounded_latent_flow_exact": True,
    "v5_shared_bias_free_zero_initialized_flow_projection_exact": True,
    "hold_relative_action_embedding_exact": True,
    "v5_one_patch_tanh_bound_and_grid_sample_exact": True,
    "phase_a_grid_sample_warn_only_scope_and_strict_restore_exact": True,
    "all_action_update_zero_bitwise_identity_exact": True,
    "detached_row_mean_energy_scaled_cross_entropy_exact": True,
    "action_indexed_energy_nll_weight_exactly_1": True,
    "dense_pairwise_live_current_and_next_encoder_paths_exact": True,
    "dense_pairwise_label_blind_all_pairs_cost_volumes_exact": True,
    "dense_pairwise_current_next_minus_current_current_volume_exact": True,
    "full_256_channel_volume_preserved_before_learned_nonlinearity": True,
    "diagnostic_displacement_never_enters_head_or_loss": True,
    "dense_head_architecture_and_parameter_count_8713_exact": True,
    "isolated_seed_20260725_initialization_and_rng_preservation_exact": True,
    "dense_inverse_detached_energy_scaled_cross_entropy_exact": True,
    "dense_inverse_loss_weight_exactly_1": True,
    "both_online_encoder_branches_receive_dense_inverse_gradient": True,
    "ema_target_receives_no_dense_inverse_gradient": True,
    "executed_label_value_invariance_fixture_exact": True,
    "full_volume_layout_and_spatial_sensitivity_fixtures_exact": True,
    "same_tensor_diff_volume_and_displacement_exact_zero": True,
    "dense_pairwise_population_and_ratio_diagnostics_exact": True,
    "dense_inverse_nll_macro_and_derangement_gates_exact": True,
    "no_hinge_fixed_temperature_margin_or_sentinel_specific_training": True,
    "patch_whitening_matches_preregistration": True,
    "all_nine_real_actions_and_hold_exact": True,
    "diagnostic_sentinels_absent_from_training": True,
    "hardest_wrong_action_gate_preserved_exact": True,
    "within_scene_next_endpoint_derangement_exact": True,
    "update100_staged_gate_exact": True,
    "update400_staged_gate_and_prior_comparison_exact": True,
    "terminal_phase_a_gate_exact": True,
    "phase_b_conditional_and_unchanged": True,
    "normal_and_operational_terminal_receipt_rules_exact": True,
    "access_rehash_counters_and_downstream_denials_exact": True,
    "no_data_whitening_base_init_role_seed_schedule_or_cap_drift": True,
}
EXECUTION_AUTHORITY = {
    "one_exact_fresh_attempt_authorized": True,
    "attempt_index": ATTEMPT_INDEX,
    "maximum_attempts": MAXIMUM_ATTEMPTS,
    "phase_a_authorized": True,
    "phase_b_only_after_exact_phase_a_pass_authorized": True,
    "n320_initialization_only_authorized": True,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "bound_authority_and_index_metadata_open_authorized": True,
    "bound_raw_manifest_audit_pairs_and_endpoints_open_authorized": True,
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


def validate_access_zero_counters(value: object) -> dict[str, int]:
    """Fail closed unless every preregistered forbidden-open counter is zero."""

    if type(value) is not dict or tuple(value) != ACCESS_ZERO_COUNTER_FIELDS:
        raise PermissionError("access zero-counter fields changed")
    if any(type(value[field]) is not int or value[field] != 0
           for field in ACCESS_ZERO_COUNTER_FIELDS):
        raise PermissionError("a forbidden access or observer counter is nonzero")
    return dict(value)


def validate_phase_a_failure_status_chain(value: object) -> dict[str, str]:
    """Validate the exact four-receipt status chain for a Phase-A failure."""

    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("Phase-A failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in PHASE_A_FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("Phase-A failure statuses are not the exact control")
    return dict(value)


def _load_static_physical_contract() -> Any:
    source = ROOT / STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != STATIC_PHYSICAL_CONTRACT_FILE_SHA256:
        raise ImportError("frozen static physical contract source changed")
    spec = importlib.util.spec_from_file_location(
        "_lewm_jepa_encoder_v9_dense_pairwise_static_physical_contract",
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
            "role_policy": {
                "metadata_only_roles": ["authority", "index"],
                "model_facing_roles": ["train", "checkpoint_selection"],
                "raw_manifest_audit_pairs_and_endpoints_may_be_opened_only_"
                "for_bound_authority_or_index_validation": True,
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
    """Return the frozen V9 science, lifecycle, and custody contract."""

    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "scientific_question":
            "can_a_label_blind_dense_current_next_minus_current_current_"
            "spatial_cost_volume_train_the_rgb_encoder_to_retain_action_"
            "relevant_local_motion_while_the_exact_v5_jepa_gate_holds",
        "initialization": {
            "base_seed": BASE_INITIALIZATION_SEED,
            "n320_online_encoder_copy": True,
            "n320_ema_encoder_copy": True,
            "predictor_and_projectors_from_fixed_seed": True,
            "v5_latent_flow_projection": {
                "path": "prediction_projector.flow_weight",
                "shape": list(FLOW_PROJECTION_SHAPE),
                "bias": False,
                "parameter_count": FLOW_PROJECTION_PARAMETER_COUNT,
                "value": 0.0,
                "rng_draw_count": 0,
            },
            "dense_pairwise_inverse_head": {
                "path": "dense_pairwise_inverse_head.",
                "device_at_construction": "cpu",
                "dtype_at_construction": "float32",
                "isolated_generator_seed":
                    DENSE_PAIRWISE_HEAD_INITIALIZATION_SEED,
                "draw_order": [
                    "conv1.weight",
                    "conv2.weight",
                    "linear.weight",
                ],
                "architecture": [
                    "Conv2d(256,16,kernel_size=1,stride=1,padding=0,"
                    "bias=False)",
                    "GELU(approximate=none)",
                    "Conv2d(16,16,kernel_size=3,stride=1,padding=1,"
                    "bias=False)",
                    "GELU(approximate=none)",
                    "AvgPool2d(kernel_size=4,stride=4,padding=0)",
                    "Flatten(16*4*4)",
                    "Linear(256,9,bias=True)",
                ],
                "convolution_initialization":
                    "kaiming_normal_(a=0,mode=fan_in,nonlinearity=relu,"
                    "generator=g)",
                "linear_weight_initialization":
                    "normal_(mean=0,std=1/16,generator=g)",
                "linear_bias": 0.0,
                "parameter_count": DENSE_PAIRWISE_HEAD_PARAMETER_COUNT,
                "all_three_weight_tensors_every_scalar_nonzero": True,
                "construction_initialization_and_transfer_preserve_global_"
                "cpu_and_accelerator_rng_states_bitwise": True,
            },
            "global_rng_state_preserved": True,
            "rejected_checkpoint_open_count": 0,
            "prior_runtime_output_open_count": 0,
        },
        "data": {
            "raw_v13_reused_exactly": True,
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "model_facing_roles": ["train", "checkpoint_selection"],
            "existing_current_next_pairs_reused_exactly": True,
            "one_scheduled_pair_is_one_presentation": True,
            "probability_calibration_open_count": 0,
            "rebuild_refine_rebalance_filter_or_resample": False,
            "new_frame_prior_rgb_third_frame_or_render_count": 0,
            "model_inputs": [
                "current_rgb",
                "next_rgb",
                "executed_requested_action",
            ],
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
            "reviewed_forward_base_commit":
                "c93124b15387acf1fd440d281e9c4503a9e8355a",
            "model_class": "Phase2DSpatialLeWorldModel",
            "model_config": phase_a_model_config(),
            "online_state_definition":
                "online_geometry(encoder.forward_tokens(rgb)[:,1:])",
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
            "objective": {
                "preserved_v5_forward": {
                    "current_plus_action_state_dependent_latent_flow": True,
                    "flow_projection_path":
                        "prediction_projector.flow_weight",
                    "flow_projection_shape": list(FLOW_PROJECTION_SHAPE),
                    "hold_relative_action_embedding":
                        "predictor.action_embed(a)-predictor.action_embed(hold)",
                    "flow_cell_bound": FLOW_CELL_BOUND,
                    "grid_scale": FLOW_GRID_SCALE,
                    "grid_sample": {
                        "mode": FLOW_GRID_SAMPLE_MODE,
                        "padding_mode": FLOW_GRID_SAMPLE_PADDING_MODE,
                        "align_corners": FLOW_GRID_SAMPLE_ALIGN_CORNERS,
                    },
                    "residual_scale": RESIDUAL_SCALE,
                    "executed_jepa_energy":
                        "mean_patch_feature_mse(true_prediction,z_next_ema)",
                    "all_action_energy_candidate_count":
                        len(ACTION_VOCABULARY),
                    "detached_row_energy_scale":
                        "m_i=stop_gradient(mean_a(E_i_a)).clamp_min(1e-8)",
                    "action_indexed_energy_nll":
                        "mean_i(m_i*cross_entropy(-E_i_all/m_i,a_i))",
                    "action_indexed_energy_nll_weight":
                        ACTION_INDEXED_ENERGY_NLL_WEIGHT,
                },
                "dense_pairwise_spatial_cost_volume": {
                    "both_current_and_next_online_encoder_branches_live": True,
                    "normalization":
                        "parameter_free_layer_norm_last_dim_eps_1e-5",
                    "layer_norm_epsilon":
                        DENSE_PAIRWISE_LAYER_NORM_EPSILON,
                    "current_next_cost":
                        "C_cn=u_cur@transpose(u_next)/sqrt(192)",
                    "current_current_cost":
                        "C_cc=u_cur@transpose(u_cur)/sqrt(192)",
                    "cost_scale": DENSE_PAIRWISE_COST_SCALE,
                    "row_softmax_dimension": -1,
                    "diff": "P_cn-P_cc",
                    "diff_axes": ["batch", "source_token", "target_token"],
                    "volume":
                        "diff.transpose(1,2).reshape(B,256,16,16).contiguous()",
                    "volume_axes": [
                        "batch",
                        "target_token_channel",
                        "source_y",
                        "source_x",
                    ],
                    "volume_value_closed_bound": [
                        -DENSE_PAIRWISE_VOLUME_VALUE_BOUND,
                        DENSE_PAIRWISE_VOLUME_VALUE_BOUND,
                    ],
                    "volume_channel_conservation": {
                        "formula": "V.sum(dim=1)",
                        "rtol": 0.0,
                        "atol": DENSE_PAIRWISE_VOLUME_CONSERVATION_ATOL,
                    },
                    "head_input": "V_only",
                    "action_label_or_embedding_in_head_input_count": 0,
                    "learned_temperature_radius_mask_topk_or_candidate_"
                    "transport_count": 0,
                },
                "displacement_observation_only": {
                    "axis_cpu":
                        "torch.linspace(-1,1,16,device=cpu,dtype=float32)",
                    "meshgrid": "indexing=ij",
                    "coordinate_columns": ["dy", "dx"],
                    "coordinate_layout": "row_major",
                    "formula": "D=(P_cn-P_cc)@q",
                    "layout":
                        "D.reshape(B,16,16,2).permute(0,3,1,2).contiguous()",
                    "component_closed_bound": [
                        -DENSE_PAIRWISE_DISPLACEMENT_COMPONENT_BOUND,
                        DENSE_PAIRWISE_DISPLACEMENT_COMPONENT_BOUND,
                    ],
                    "head_or_loss_input_count": 0,
                },
                "dense_inverse": {
                    "target": "executed_action_index",
                    "unscaled_row_loss":
                        "CE(dense_pairwise_inverse_head(V),a_i)",
                    "loss": "mean_i(m_i*CE_i)",
                    "loss_weight": DENSE_PAIRWISE_INVERSE_LOSS_WEIGHT,
                    "class_weight_focal_margin_temperature_label_smoothing":
                        False,
                    "only_action_label_use": "cross_entropy_target",
                },
                "whitening": {
                    "branches": [
                        "online_raw_current_patch",
                        "normalized_online_projected_current_patch",
                    ],
                    "position_centering": "subtract_batch_mean_per_patch",
                    "matrix_shape": ["B*N", LATENT_DIM],
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
                "total_loss":
                    "L_JEPA+L_EnergyNLL+L_dense_inverse+"
                    "0.50*(V_raw+V_projected)+"
                    "0.02*(K_raw+K_projected)",
                "removed_v6_v7_v8_inverse_or_local_correspondence_"
                "training_term_count": 0,
            },
            "source_acceptance": {
                "label_changes_leave_costs_probabilities_diff_volume_"
                "displacement_and_logits_bitwise_unchanged": True,
                "current_live_next_detached_encoder_parameter_gradient_"
                "finite_and_nonzero": True,
                "current_detached_next_live_encoder_parameter_gradient_"
                "finite_and_nonzero": True,
                "both_live_all_head_parameter_gradients_finite_and_nonzero":
                    True,
                "ema_gradient_absent": True,
                "index_coded_volume_layout_exact": True,
                "rolled_volume_changes_logits": True,
                "changed_or_permuted_displacement_cannot_change_logits": True,
                "identical_states_diff_volume_displacement_bitwise_zero":
                    True,
                "no_spatial_reduction_before_two_convolution_stack": True,
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
                "encoder_prefixes": list(PHASE_A_ENCODER_PARAMETER_PREFIXES),
                "other_learning_rate": 3e-4,
                "other_prefixes": list(PHASE_A_AUXILIARY_PARAMETER_PREFIXES),
                "target_parameters_excluded": True,
                "determinism": {
                    "strict_deterministic_algorithms": True,
                    "phase_a_grid_sample_warn_only": True,
                    "only_permitted_warning_prefix":
                        PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX,
                    "strict_state_restored": True,
                },
            },
            "schedule": build_schedule_identity("phase_a"),
            "gate": {
                "selection_pair_count": SELECTION_ROLE_COUNTS["pairs"],
                "selection_non_hold_pair_count":
                    SELECTION_NON_HOLD_PAIR_COUNT,
                "scene_family_count": len(SCENE_FAMILIES),
                "same_device_zero_logit_reference":
                    "mean(CE(zeros([495,9],float32),executed_actions))",
                "update_100": dict(PHASE_A_UPDATE_100_THRESHOLDS),
                "update_400": dict(PHASE_A_UPDATE_400_THRESHOLDS),
                "update_100_to_400_progress": {
                    "unscaled_dense_inverse_nll": "strictly_lower",
                    "dense_inverse_macro_balanced_accuracy":
                        "greater_than_or_equal",
                },
                "update_400_to_1000_progress": {
                    "unscaled_dense_inverse_nll": "strictly_lower",
                    "dense_inverse_macro_balanced_accuracy":
                        "greater_than_or_equal",
                },
                "terminal": dict(PHASE_A_PASS_THRESHOLDS),
                "top1_tie_break":
                    "lowest_index_in_frozen_action_vocabulary",
                "correct_and_deranged_population": 495,
                "non_hold_correct_and_current_current_population":
                    SELECTION_NON_HOLD_PAIR_COUNT,
                "dense_diagnostics": {
                    "correct_nll": "mean_unscaled_CE_over_exact_495_rows",
                    "deranged_next_nll":
                        "mean_unscaled_CE_over_same_exact_495_rows",
                    "non_hold_correct_nll":
                        "mean_unscaled_CE_over_exact_435_non_hold_rows",
                    "non_hold_current_current_nll":
                        "mean_unscaled_CE_over_same_exact_435_non_hold_rows",
                    "cross_pair_displacement_rms":
                        "sqrt(mean(D.float()**2))_over_495*2*16*16_values",
                    "per_family_deranged_margin":
                        "mean(deranged_row_CE-correct_row_CE)_strictly_"
                        "positive_count",
                },
                "derangement": {
                    "scope": "scene_local_frozen_mapping",
                    "changes_only": "z_next",
                    "fixed": ["current_state", "label", "row_order"],
                },
            },
        },
        "phase_b": {
            "entered_only_after_phase_a_pass": True,
            "copied_state": "phase_a_terminal_in_memory_online_encoder_only",
            "phase_a_checkpoint_payload_read_count": 0,
            "dense_pairwise_inverse_head_copied": False,
            "latent_flow_predictor_or_projector_copied": False,
            "optimizer_copied": False,
            "trainable_prefixes": list(PHASE_B_TRAINABLE_PARAMETER_PREFIXES),
            "hard_sync": {
                "count": 1,
                "copied_prefixes": ["target_encoder."],
                "forbidden_copy_prefixes": ["target_bev_decoder."],
            },
            "jepa_objective_count": 0,
            "ema_update_count": 0,
            "unchanged_multiresolution_evidence_head_and_physical_evaluator":
                True,
            "schedule": build_schedule_identity("phase_b"),
            "pass_thresholds": dict(PHASE_B_PASS_THRESHOLDS),
            "promotable_shared_v5_checkpoint": False,
        },
        "lifecycle": {
            "attempt_index": ATTEMPT_INDEX,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_must_be_absent_before_reservation": True,
            "reserve_mode": "0700",
            "normal_phase_a_receipts": list(NORMAL_PHASE_A_RECEIPT_PATHS),
            "phase_b_receipts": list(PHASE_B_RECEIPT_PATHS),
            "operational_failure_receipts":
                list(OPERATIONAL_FAILURE_RECEIPT_PATHS),
            "normal_phase_a_terminal_controls":
                list(PHASE_A_TERMINAL_CONTROLS),
            "phase_a_failure_status_chain":
                "metrics_artifact_result_and_completion_each_equal_the_"
                "exact_failure_control",
            "phase_b_metrics_statuses": list(PHASE_B_METRICS_STATUSES),
            "phase_b_artifact_statuses": list(PHASE_B_ARTIFACT_STATUSES),
            "result_terminal_statuses": list(RESULT_TERMINAL_STATUSES),
            "completion_terminal_statuses":
                list(COMPLETION_TERMINAL_STATUSES),
            "operational_failure": {
                "failure_status": OPERATIONAL_FAILURE_STATUS,
                "completion_status": OPERATIONAL_FAILURE_COMPLETION_STATUS,
                "reservation_publication_failure_status":
                    RESERVATION_PUBLICATION_FAILURE_STATUS,
                "retain_every_partial_file": True,
                "publish_only_failure_and_completion_after_partial_chain":
                    True,
                "record_exact_partial_inventory_operation_counts_"
                "determinism_access_state_missing_normal_receipts_and_"
                "available_bindings": True,
                "missing_normal_receipts_named_explicitly": True,
                "missing_receipts_never_synthesized_or_fabricated": True,
                "reservation_may_be_absent_only_when_its_publication_failed":
                    True,
            },
            "runtime_checkpoint_and_trace_payloads_write_only": True,
            "no_retry_resume_second_seed_or_observer_rerun": True,
            "terminal_inventory_sealed_read_only": True,
        },
        "cumulative_caps": {
            "phase_a_updates": PHASE_A_MAXIMUM_UPDATE,
            "phase_a_presentations": PHASE_A_MAXIMUM_PRESENTATIONS,
            "phase_a_gpu_active_minutes":
                PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            "phase_b_updates": PHASE_B_MAXIMUM_UPDATE,
            "phase_b_presentations": PHASE_B_MAXIMUM_PRESENTATIONS,
            "phase_b_gpu_active_minutes":
                PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES,
            "updates": CUMULATIVE_MAXIMUM_UPDATE,
            "presentations": CUMULATIVE_MAXIMUM_PRESENTATIONS,
            "total_gpu_active_minutes":
                CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
        },
        "access_zero_counters": list(ACCESS_ZERO_COUNTER_FIELDS),
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


def _validate_latent_flow_observation(
    value: object,
    *,
    name: str,
    require_update_zero: bool = False,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != LATENT_FLOW_OBSERVATION_FIELDS:
        raise ValueError(f"{name} fields changed")
    for field in (
        "all_values_finite",
        "all_components_within_closed_one_patch_bound",
        "hold_flow_exactly_zero",
    ):
        if type(value[field]) is not bool:
            raise TypeError(f"{name} {field} must be Boolean")
    maximum = _finite_nonnegative(
        value["maximum_absolute_flow_cell"],
        name=f"{name} maximum absolute flow cell",
    )
    count = value["non_hold_action_nonzero_count"]
    if type(count) is not int or not 0 <= count <= NON_HOLD_ACTION_COUNT:
        raise ValueError(f"{name} non-hold action count changed")
    per_action = value["per_action_any_nonzero"]
    if (
        type(per_action) is not dict
        or tuple(per_action) != ACTION_VOCABULARY
        or any(type(item) is not bool for item in per_action.values())
    ):
        raise ValueError(f"{name} per-action flow receipt changed")
    observed_count = sum(
        int(per_action[action])
        for action in ACTION_VOCABULARY
        if action != "hold"
    )
    if (
        count != observed_count
        or value["hold_flow_exactly_zero"] is per_action["hold"]
        or value["all_components_within_closed_one_patch_bound"]
        is not (maximum <= FLOW_CELL_BOUND)
    ):
        raise ValueError(f"{name} flow receipt is internally inconsistent")
    if require_update_zero and (
        not value["all_values_finite"]
        or not value["all_components_within_closed_one_patch_bound"]
        or not value["hold_flow_exactly_zero"]
        or maximum != 0.0
        or count != 0
        or any(per_action.values())
    ):
        raise ValueError(f"{name} update-zero flow is not exact zero")
    return {
        "all_values_finite": value["all_values_finite"],
        "all_components_within_closed_one_patch_bound":
            value["all_components_within_closed_one_patch_bound"],
        "hold_flow_exactly_zero": value["hold_flow_exactly_zero"],
        "maximum_absolute_flow_cell": maximum,
        "non_hold_action_nonzero_count": count,
        "per_action_any_nonzero": dict(per_action),
    }


def _validate_dense_pairwise_inverse_observation(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != DENSE_PAIRWISE_INVERSE_OBSERVATION_FIELDS
    ):
        raise ValueError(f"{name} fields changed")
    boolean_fields = (
        "all_values_finite",
        "probabilities_all_values_finite",
        "probability_rows_normalized",
        "volume_all_values_finite",
        "volume_values_within_closed_unit_interval",
        "volume_channel_conservation",
        "displacement_all_values_finite",
        "displacement_values_within_closed_two_bound",
        "same_tensor_diff_exact_zero",
        "same_tensor_volume_exact_zero",
        "same_tensor_displacement_exact_zero",
        "head_parameters_all_values_finite",
        "head_weight_tensors_all_nonzero",
    )
    for field in boolean_fields:
        if type(value[field]) is not bool:
            raise TypeError(f"{name} {field} must be Boolean")
    parameter_count = value["head_parameter_count"]
    if (
        type(parameter_count) is not int
        or parameter_count != DENSE_PAIRWISE_HEAD_PARAMETER_COUNT
    ):
        raise ValueError(f"{name} head parameter count changed")

    maximum_displacement = _finite_nonnegative(
        value["maximum_absolute_displacement_component"],
        name=f"{name} maximum absolute displacement component",
    )
    if (
        value["displacement_values_within_closed_two_bound"]
        is not (
            maximum_displacement <= DENSE_PAIRWISE_DISPLACEMENT_COMPONENT_BOUND
        )
    ):
        raise ValueError(f"{name} displacement bound receipt is inconsistent")
    displacement_rms = _finite_nonnegative(
        value["cross_pair_displacement_rms"],
        name=f"{name} cross-pair displacement RMS",
    )
    displacement_value_count = value["cross_pair_displacement_value_count"]
    expected_displacement_value_count = (
        SELECTION_ROLE_COUNTS["pairs"]
        * 2
        * LATENT_GRID_HEIGHT
        * LATENT_GRID_WIDTH
    )
    if (
        type(displacement_value_count) is not int
        or displacement_value_count != expected_displacement_value_count
    ):
        raise ValueError(f"{name} displacement RMS population changed")
    zero_logit_reference = _finite_nonnegative(
        value["zero_logit_reference_nll"],
        name=f"{name} zero-logit NLL reference",
    )
    if (
        zero_logit_reference <= 0.0
        or not math.isclose(
            zero_logit_reference,
            math.log(float(len(ACTION_VOCABULARY))),
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
    ):
        raise ValueError(f"{name} zero-logit NLL reference changed")

    unscaled_nll = _finite_nonnegative(
        value["unscaled_dense_inverse_nll"],
        name=f"{name} unscaled dense inverse NLL",
    )
    top1 = _finite_unit_interval(
        value["dense_inverse_top1_accuracy"],
        name=f"{name} dense inverse top-1 accuracy",
    )
    macro = _finite_unit_interval(
        value["dense_inverse_macro_balanced_accuracy"],
        name=f"{name} dense inverse macro balanced accuracy",
    )
    per_action = value["per_executed_action_dense_inverse"]
    if type(per_action) is not dict or tuple(per_action) != ACTION_VOCABULARY:
        raise ValueError(f"{name} executed-action order changed")
    normalized_actions: dict[str, dict[str, int | float]] = {}
    total_rows = 0
    weighted_nll = 0.0
    weighted_correct = 0.0
    recalls: list[float] = []
    for action in ACTION_VOCABULARY:
        row = per_action[action]
        if type(row) is not dict or set(row) != {
            "row_count",
            "mean_nll",
            "recall",
        }:
            raise ValueError(f"{name} {action} action fields changed")
        row_count = row["row_count"]
        if type(row_count) is not int or row_count <= 0:
            raise ValueError(f"{name} {action} population must be nonempty")
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
            raise ValueError(f"{name} {action} recall is not a count ratio")
        normalized_actions[action] = {
            "row_count": row_count,
            "mean_nll": mean_nll,
            "recall": recall,
        }
        total_rows += row_count
        weighted_nll += float(row_count) * mean_nll
        weighted_correct += float(row_count) * recall
        recalls.append(recall)
    if total_rows != SELECTION_ROLE_COUNTS["pairs"]:
        raise ValueError(f"{name} all-action population changed")
    non_hold_action_rows = sum(
        normalized_actions[action]["row_count"]
        for action in ACTION_VOCABULARY
        if action != "hold"
    )
    hold_action_rows = normalized_actions["hold"]["row_count"]
    if (
        non_hold_action_rows != SELECTION_NON_HOLD_PAIR_COUNT
        or hold_action_rows
        != SELECTION_ROLE_COUNTS["pairs"] - SELECTION_NON_HOLD_PAIR_COUNT
    ):
        raise ValueError(f"{name} hold/non-hold action population changed")
    if (
        not math.isclose(
            unscaled_nll,
            weighted_nll / float(total_rows),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            top1,
            weighted_correct / float(total_rows),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            macro,
            sum(recalls) / float(len(recalls)),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    ):
        raise ValueError(f"{name} action aggregations are inconsistent")

    correct_nll = _finite_nonnegative(
        value["correct_pair_nll"],
        name=f"{name} all-495 correct-pair NLL",
    )
    correct_count = value["correct_pair_count"]
    deranged_nll = _finite_nonnegative(
        value["deranged_next_nll"],
        name=f"{name} all-495 deranged-next NLL",
    )
    deranged_count = value["deranged_next_pair_count"]
    reported_deranged_ratio = _finite_nonnegative(
        value["correct_to_deranged_nll_ratio"],
        name=f"{name} correct-to-deranged NLL ratio",
    )
    non_hold_correct = _finite_nonnegative(
        value["non_hold_correct_pair_nll"],
        name=f"{name} exact-435 non-hold correct NLL",
    )
    non_hold_correct_count = value["non_hold_correct_pair_count"]
    non_hold_current_current = _finite_nonnegative(
        value["non_hold_current_current_nll"],
        name=f"{name} exact-435 current-current NLL",
    )
    non_hold_current_current_count = value[
        "non_hold_current_current_pair_count"
    ]
    reported_current_current_ratio = _finite_nonnegative(
        value["non_hold_correct_to_current_current_nll_ratio"],
        name=f"{name} non-hold correct-to-current-current NLL ratio",
    )
    expected_deranged_ratio = _positive_denominator_ratio(
        correct_nll,
        deranged_nll,
        name=f"{name} correct-to-deranged NLL ratio",
    )
    expected_current_current_ratio = _positive_denominator_ratio(
        non_hold_correct,
        non_hold_current_current,
        name=f"{name} non-hold correct-to-current-current NLL ratio",
    )
    if (
        type(correct_count) is not int
        or correct_count != SELECTION_ROLE_COUNTS["pairs"]
        or type(deranged_count) is not int
        or deranged_count != SELECTION_ROLE_COUNTS["pairs"]
        or type(non_hold_correct_count) is not int
        or non_hold_correct_count != SELECTION_NON_HOLD_PAIR_COUNT
        or type(non_hold_current_current_count) is not int
        or non_hold_current_current_count != SELECTION_NON_HOLD_PAIR_COUNT
        or not math.isclose(
            correct_nll,
            unscaled_nll,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            reported_deranged_ratio,
            expected_deranged_ratio,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            reported_current_current_ratio,
            expected_current_current_ratio,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    ):
        raise ValueError(f"{name} pairing populations or ratios changed")

    per_family = value["per_family_deranged_minus_correct_nll"]
    if type(per_family) is not dict or tuple(per_family) != SCENE_FAMILIES:
        raise ValueError(f"{name} derangement family order changed")
    normalized_families = {
        family: _finite_signed(
            per_family[family],
            name=f"{name} {family} deranged-minus-correct NLL",
        )
        for family in SCENE_FAMILIES
    }
    reported_positive = value["deranged_positive_family_margin_count"]
    expected_positive = sum(
        int(margin > 0.0) for margin in normalized_families.values()
    )
    if type(reported_positive) is not int or reported_positive != expected_positive:
        raise ValueError(f"{name} derangement family count is inconsistent")

    component_finiteness = (
        value["probabilities_all_values_finite"]
        and value["volume_all_values_finite"]
        and value["displacement_all_values_finite"]
        and value["head_parameters_all_values_finite"]
    )
    if value["all_values_finite"] is not component_finiteness:
        raise ValueError(f"{name} aggregate finiteness is inconsistent")
    return {
        **{
            field: value[field]
            for field in boolean_fields
        },
        "head_parameter_count": parameter_count,
        "maximum_absolute_displacement_component": maximum_displacement,
        "cross_pair_displacement_rms": displacement_rms,
        "cross_pair_displacement_value_count": displacement_value_count,
        "zero_logit_reference_nll": zero_logit_reference,
        "unscaled_dense_inverse_nll": unscaled_nll,
        "dense_inverse_top1_accuracy": top1,
        "per_executed_action_dense_inverse": normalized_actions,
        "dense_inverse_macro_balanced_accuracy": macro,
        "correct_pair_nll": correct_nll,
        "correct_pair_count": correct_count,
        "deranged_next_nll": deranged_nll,
        "deranged_next_pair_count": deranged_count,
        "correct_to_deranged_nll_ratio": reported_deranged_ratio,
        "non_hold_correct_pair_nll": non_hold_correct,
        "non_hold_correct_pair_count": non_hold_correct_count,
        "non_hold_current_current_nll": non_hold_current_current,
        "non_hold_current_current_pair_count":
            non_hold_current_current_count,
        "non_hold_correct_to_current_current_nll_ratio":
            reported_current_current_ratio,
        "deranged_positive_family_margin_count": reported_positive,
        "per_family_deranged_minus_correct_nll": normalized_families,
    }


def _normalize_phase_a_inputs(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
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
        type(update0_metrics["all_action_predictions_bitwise_equal"])
        is not bool
        or type(update0_metrics["all_action_unordered_pair_count"]) is not int
        or update0_metrics["all_action_unordered_pair_count"] != 36
        or type(update0_metrics["all_action_prediction_row_count"]) is not int
        or update0_metrics["all_action_prediction_row_count"]
        != SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise ValueError("Phase-A update-zero action symmetry receipt changed")

    update0_flow = _validate_latent_flow_observation(
        update0_metrics["latent_flow"],
        name="Phase-A update-zero latent flow",
        require_update_zero=True,
    )
    latent_flow = _validate_latent_flow_observation(
        metrics["latent_flow"],
        name="Phase-A latent flow",
    )
    update0_dense = _validate_dense_pairwise_inverse_observation(
        update0_metrics["dense_pairwise_inverse"],
        name="Phase-A update-zero dense pairwise inverse",
    )
    dense = _validate_dense_pairwise_inverse_observation(
        metrics["dense_pairwise_inverse"],
        name="Phase-A dense pairwise inverse",
    )
    if type(metrics["all_values_finite"]) is not bool:
        raise TypeError("all_values_finite must be Boolean")
    if type(metrics["ema_target_gradient_free"]) is not bool:
        raise TypeError("ema_target_gradient_free must be Boolean")
    if (
        type(metrics["pair_count"]) is not int
        or metrics["pair_count"] != SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise ValueError("Phase-A selection pair count changed")
    if (
        type(metrics["scene_family_count"]) is not int
        or metrics["scene_family_count"] != len(SCENE_FAMILIES)
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
        or metrics["cyclic_wrong_action_pair_count"] != metrics["pair_count"]
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
        "latent_flow",
        "dense_pairwise_inverse",
    }
    values = {
        field: _finite_nonnegative(metrics[field], name=field)
        for field in numeric_names
    }
    update0 = {
        field: _finite_nonnegative(
            update0_metrics[field],
            name=f"update0 {field}",
        )
        for field in PHASE_A_UPDATE0_HEALTH_FIELDS
    }
    if any(item <= 0.0 for item in update0.values()):
        raise ValueError("Phase-A update-zero health denominators must be positive")

    per_family = metrics["per_family"]
    if type(per_family) is not dict or tuple(per_family) != SCENE_FAMILIES:
        raise ValueError("Phase-A scene-family order changed")
    cyclic_positive = 0
    hold_positive = 0
    normalized_families: dict[str, dict[str, float | bool]] = {}
    for family in SCENE_FAMILIES:
        row = per_family[family]
        if type(row) is not dict or set(row) != {
            "cyclic_wrong_action_minus_true_mse",
            "hardest_wrong_action_minus_true_mse",
            "hold_action_minus_non_hold_true_mse",
            "hold_action_rows_match_non_hold_rows",
        }:
            raise ValueError(f"Phase-A family metrics changed: {family}")
        population_matches = row["hold_action_rows_match_non_hold_rows"]
        if type(population_matches) is not bool:
            raise TypeError(f"{family} hold population identity must be Boolean")
        cyclic = _finite_signed(
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
        cyclic_positive += int(cyclic > 0.0)
        hold_positive += int(hold > 0.0 and population_matches)
        normalized_families[family] = {
            "cyclic_wrong_action_minus_true_mse": cyclic,
            "hardest_wrong_action_minus_true_mse": hardest,
            "hold_action_minus_non_hold_true_mse": hold,
            "hold_action_rows_match_non_hold_rows": population_matches,
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
        "true_to_hardest_wrong_action": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["hardest_wrong_action_mse"],
            name="hardest-wrong-action ratio",
        ),
        "non_hold_true_to_hold_action": _positive_denominator_ratio(
            values["non_hold_true_pair_mse"],
            values["hold_action_mse"],
            name="hold-action ratio",
        ),
        "true_to_shuffled_current": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_current_mse"],
            name="shuffled-current ratio",
        ),
    }
    counts = {
        "pair_count": metrics["pair_count"],
        "cyclic_wrong_action_pair_count":
            metrics["cyclic_wrong_action_pair_count"],
        "all_wrong_action_candidate_count":
            metrics["all_wrong_action_candidate_count"],
        "non_hold_pair_count": non_hold_count,
        "hold_action_pair_count": metrics["hold_action_pair_count"],
        "scene_family_count": metrics["scene_family_count"],
        "cyclic_wrong_action_positive_family_count": cyclic_positive,
        "hold_action_positive_family_count": hold_positive,
        "non_hold_action_nonzero_flow_count":
            latent_flow["non_hold_action_nonzero_count"],
        "dense_deranged_positive_family_margin_count":
            dense["deranged_positive_family_margin_count"],
    }
    common_conjuncts = {
        "diagnostic_rng_and_model_state_preserved":
            observation_integrity["rng_state_preserved"]
            and observation_integrity["state_mutation_count"] == 0,
        "update_zero_all_action_predictions_bitwise_equal":
            update0_metrics["all_action_predictions_bitwise_equal"],
        "update_zero_all_action_flows_exactly_zero":
            update0_flow["non_hold_action_nonzero_count"] == 0
            and update0_flow["maximum_absolute_flow_cell"] == 0.0
            and not any(update0_flow["per_action_any_nonzero"].values()),
        "update_zero_dense_volume_and_displacement_health_exact":
            update0_dense["all_values_finite"]
            and update0_dense["probability_rows_normalized"]
            and update0_dense["volume_values_within_closed_unit_interval"]
            and update0_dense["volume_channel_conservation"]
            and update0_dense["displacement_values_within_closed_two_bound"]
            and update0_dense["same_tensor_diff_exact_zero"]
            and update0_dense["same_tensor_volume_exact_zero"]
            and update0_dense["same_tensor_displacement_exact_zero"]
            and update0_dense["head_weight_tensors_all_nonzero"],
        "latent_flow_finite_within_bound_and_hold_exactly_zero":
            latent_flow["all_values_finite"]
            and latent_flow["all_components_within_closed_one_patch_bound"]
            and latent_flow["hold_flow_exactly_zero"],
        "dense_pairwise_values_probabilities_volume_and_displacement_healthy":
            dense["all_values_finite"]
            and dense["probability_rows_normalized"]
            and dense["volume_values_within_closed_unit_interval"]
            and dense["volume_channel_conservation"]
            and dense["displacement_values_within_closed_two_bound"]
            and dense["same_tensor_diff_exact_zero"]
            and dense["same_tensor_volume_exact_zero"]
            and dense["same_tensor_displacement_exact_zero"],
        "finite_and_ema_gradient_free":
            metrics["all_values_finite"]
            and metrics["ema_target_gradient_free"],
        "control_populations_exact":
            metrics["cyclic_wrong_action_pair_count"] == metrics["pair_count"]
            and metrics["all_wrong_action_candidate_count"]
            == metrics["pair_count"] * 8
            and metrics["hold_action_pair_count"] == non_hold_count
            and metrics["hold_action_rows_match_non_hold_rows"],
    }
    return {
        "metrics": metrics,
        "values": values,
        "update0": update0,
        "latent_flow": latent_flow,
        "dense_pairwise_inverse": dense,
        "ratios": ratios,
        "counts": counts,
        "per_family": normalized_families,
        "common_conjuncts": common_conjuncts,
    }


def _update_100_conjuncts(normalized: Mapping[str, Any]) -> dict[str, bool]:
    values = normalized["values"]
    ratios = normalized["ratios"]
    counts = normalized["counts"]
    dense = normalized["dense_pairwise_inverse"]
    threshold = PHASE_A_UPDATE_100_THRESHOLDS
    return {
        **dict(normalized["common_conjuncts"]),
        "dense_inverse_nll_strictly_below_point98_log9":
            dense["unscaled_dense_inverse_nll"]
            < threshold["dense_inverse_nll_log9_factor_strictly_less_than"]
            * dense["zero_logit_reference_nll"],
        "dense_inverse_macro_balanced_accuracy_strictly_above_two_ninths":
            dense["dense_inverse_macro_balanced_accuracy"]
            > threshold[
                "dense_inverse_macro_balanced_accuracy_strictly_greater_than"
            ],
        "correct_to_deranged_nll_ratio_strictly_below_point99":
            dense["correct_to_deranged_nll_ratio"]
            < threshold["correct_to_deranged_nll_ratio_strictly_less_than"],
        "non_hold_correct_to_current_current_nll_ratio_strictly_below_"
        "point99":
            dense["non_hold_correct_to_current_current_nll_ratio"]
            < threshold[
                "non_hold_correct_to_current_current_nll_ratio_"
                "strictly_less_than"
            ],
        "deranged_nll_margin_positive_in_at_least_six_families":
            counts["dense_deranged_positive_family_margin_count"]
            >= threshold["deranged_positive_family_margin_count_minimum"],
        "centered_raw_rank_above_v3_update_zero":
            values["centered_raw_patch_effective_rank"]
            > threshold[
                "centered_raw_patch_effective_rank_strictly_greater_than"
            ],
        "centered_projected_rank_above_v3_update_zero":
            values["centered_projected_target_effective_rank"]
            > threshold[
                "centered_projected_target_effective_rank_"
                "strictly_greater_than"
            ],
        "raw_cross_sample_variance_at_least_quarter_update0":
            ratios["raw_cross_sample_variance_to_update0"]
            >= threshold["update_zero_health_fraction_minimum"],
        "spatial_diversity_at_least_quarter_update0":
            ratios["content_residual_spatial_diversity_to_update0"]
            >= threshold["update_zero_health_fraction_minimum"],
        "true_at_most_point90_shuffled_next":
            ratios["true_to_shuffled_next"]
            <= threshold["shuffled_next_ratio_maximum"],
        "true_at_most_point95_shuffled_current":
            ratios["true_to_shuffled_current"]
            <= threshold["shuffled_current_ratio_maximum"],
        "all_eight_non_hold_actions_have_nonzero_flow":
            counts["non_hold_action_nonzero_flow_count"]
            == threshold["non_hold_action_nonzero_flow_count_required"],
    }


def _update_400_mechanism_conjuncts(
    normalized: Mapping[str, Any],
) -> dict[str, bool]:
    values = normalized["values"]
    ratios = normalized["ratios"]
    counts = normalized["counts"]
    threshold = PHASE_A_UPDATE_400_THRESHOLDS
    return {
        "centered_raw_rank_at_least_halfway_to_48":
            values["centered_raw_patch_effective_rank"]
            >= threshold["centered_raw_patch_effective_rank_minimum"],
        "centered_projected_rank_at_least_halfway_to_48":
            values["centered_projected_target_effective_rank"]
            >= threshold["centered_projected_target_effective_rank_minimum"],
        "true_strictly_below_point99_cyclic_wrong_action":
            ratios["true_to_cyclic_wrong_action"]
            < threshold["cyclic_wrong_action_ratio_strictly_less_than"],
        "true_strictly_below_point99_hardest_wrong_action":
            ratios["true_to_hardest_wrong_action"]
            < threshold["hardest_wrong_action_ratio_strictly_less_than"],
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
    }


def _previous_dense_pairwise_inverse(
    previous_metrics: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if (
        type(previous_metrics) is not dict
        or set(previous_metrics) != PHASE_A_METRIC_FIELDS
    ):
        raise ValueError(f"{name} metric fields changed")
    return _validate_dense_pairwise_inverse_observation(
        previous_metrics["dense_pairwise_inverse"],
        name=f"{name} dense pairwise inverse",
    )


def evaluate_phase_a(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
    previous_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate update 1,000 against the frozen final V9 conjunction."""

    normalized = _normalize_phase_a_inputs(
        metrics,
        update0_metrics,
        observation_integrity,
    )
    previous_dense = _previous_dense_pairwise_inverse(
        previous_metrics,
        name="Phase-A update-400",
    )
    values = normalized["values"]
    ratios = normalized["ratios"]
    counts = normalized["counts"]
    dense = normalized["dense_pairwise_inverse"]
    threshold = PHASE_A_PASS_THRESHOLDS
    conjuncts = {
        **_update_100_conjuncts(normalized),
        **_update_400_mechanism_conjuncts(normalized),
        "dense_inverse_nll_strictly_lower_than_update400":
            dense["unscaled_dense_inverse_nll"]
            < previous_dense["unscaled_dense_inverse_nll"],
        "dense_inverse_macro_balanced_accuracy_not_below_update400":
            dense["dense_inverse_macro_balanced_accuracy"]
            >= previous_dense["dense_inverse_macro_balanced_accuracy"],
        "centered_raw_rank_at_least_48":
            values["centered_raw_patch_effective_rank"]
            >= threshold["centered_raw_patch_effective_rank_minimum"],
        "centered_projected_rank_at_least_48":
            values["centered_projected_target_effective_rank"]
            >= threshold[
                "centered_projected_target_effective_rank_minimum"
            ],
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
    }
    passed = all(conjuncts.values())
    return {
        "update": 1_000,
        "passed": passed,
        "control": CONTROL_PHASE_A_PASS if passed else CONTROL_PHASE_A_FAIL,
        "conjuncts": conjuncts,
        "ratios": dict(ratios),
        "counts": dict(counts),
        "thresholds": {
            "update_100": dict(PHASE_A_UPDATE_100_THRESHOLDS),
            "update_400": dict(PHASE_A_UPDATE_400_THRESHOLDS),
            "terminal": dict(PHASE_A_PASS_THRESHOLDS),
        },
        "per_family": dict(normalized["per_family"]),
        "latent_flow": dict(normalized["latent_flow"]),
        "dense_pairwise_inverse": dict(dense),
    }


def evaluate_phase_a_continuation(
    update: int,
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
    previous_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the preregistered update-100 or update-400 V9 gate."""

    if update not in {100, 400}:
        raise ValueError("continuation gate update must be 100 or 400")
    normalized = _normalize_phase_a_inputs(
        metrics,
        update0_metrics,
        observation_integrity,
    )
    conjuncts = _update_100_conjuncts(normalized)
    if update == 100:
        if previous_metrics is not None:
            raise ValueError("update-100 gate has no previous checkpoint")
        threshold: Mapping[str, Any] = PHASE_A_UPDATE_100_THRESHOLDS
        failure_control = CONTROL_PHASE_A_UPDATE_100_FAIL
    else:
        if previous_metrics is None:
            raise ValueError("update-400 gate requires update-100 metrics")
        previous_dense = _previous_dense_pairwise_inverse(
            previous_metrics,
            name="Phase-A update-100",
        )
        dense = normalized["dense_pairwise_inverse"]
        conjuncts.update(_update_400_mechanism_conjuncts(normalized))
        conjuncts.update({
            "dense_inverse_nll_strictly_lower_than_update100":
                dense["unscaled_dense_inverse_nll"]
                < previous_dense["unscaled_dense_inverse_nll"],
            "dense_inverse_macro_balanced_accuracy_not_below_update100":
                dense["dense_inverse_macro_balanced_accuracy"]
                >= previous_dense["dense_inverse_macro_balanced_accuracy"],
        })
        threshold = PHASE_A_UPDATE_400_THRESHOLDS
        failure_control = CONTROL_PHASE_A_UPDATE_400_FAIL
    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": CONTROL_CONTINUE if passed else failure_control,
        "conjuncts": conjuncts,
        "ratios": dict(normalized["ratios"]),
        "counts": dict(normalized["counts"]),
        "thresholds": dict(threshold),
        "per_family": dict(normalized["per_family"]),
        "latent_flow": dict(normalized["latent_flow"]),
        "dense_pairwise_inverse":
            dict(normalized["dense_pairwise_inverse"]),
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
        "validate_access_zero_counters",
        "validate_binding",
        "validate_phase_a_failure_status_chain",
        "validate_review",
        "validate_runtime_inputs",
        "validate_schedule_identity",
        "validate_source_manifest",
        "with_content_sha256",
    }
]
