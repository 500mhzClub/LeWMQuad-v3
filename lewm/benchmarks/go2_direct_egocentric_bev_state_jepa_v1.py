"""Source-only contract for Direct Egocentric BEV-State JEPA V1.

Importing this module reads no generated input, RGB, raster labels, checkpoint,
runtime output, trace, accelerator state, held-out material, or sealed material.
It imports only the Python standard library.  A future reviewed runner owns the
single bounded execution and may run only after a separate authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/plan_efficiency/direct_bev_prereg_audit"
SCHEMA_PREFIX = "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1"
EXPERIMENT_ID = "go2_rgb_direct_egocentric_bev_state_jepa_v1"


# Frozen documents.
ARCHITECTURE_DECISION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "architecture_decision_2026-07-26.md"
)
ARCHITECTURE_DECISION_COMMIT = (
    "4831f77d9ddae15fa8504ffb1d06f73e0af427a4"
)
ARCHITECTURE_DECISION_FILE_SHA256 = (
    "9d56f98f33ab501b1f1298a1a94f640305aa4c38de1e09440f8249c716a772fc"
)
ARCHITECTURE_DECISION_BYTE_COUNT = 16_096

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "4831f77d9ddae15fa8504ffb1d06f73e0af427a4"
PREREGISTRATION_FILE_SHA256 = (
    "6863041a0a498a297c92c011ede97f1ffebaeb2121eebbf61054243a185bb3c0"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "8d97a9f4769f8ce1ebf69c7665c9fe4a57f693eb5b02acd6a9b3224b796c0943"
)
PREREGISTRATION_BYTE_COUNT = 21_561

PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "preregistration_independent_review_2026-07-26.json"
)
PREREGISTRATION_INDEPENDENT_REVIEW_COMMIT = (
    "f2301d78554d000d99bcc733f0a502127525cda8"
)
PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256 = (
    "0af804ca375c1fd8503f4b2800cb8f40d5c713219e39b0ea7282d4b98dac0f73"
)
PREREGISTRATION_INDEPENDENT_REVIEW_CONTENT_SHA256 = (
    "b7de0581b3d782c264ca12019448201151696efc3a6a87436e8ef5b5989e7bce"
)
PREREGISTRATION_INDEPENDENT_REVIEW_BYTE_COUNT = 10_374
PREREGISTRATION_INDEPENDENT_REVIEW_STATUS = (
    "PASS_INDEPENDENT_PREREGISTRATION_REVIEW_SOURCE_IMPLEMENTABLE_NO_"
    "EXECUTION_AUTHORITY"
)

PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_"
    "learning_curve_continuation_terminal_audit_2026-07-26.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = "e03f6eb2dbfadad188e2cb07d5451096b4179969"
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "1486a102b010d06dc8b8a91130eb6c79a95d9a8ca426dea9a4833fc4aee488d8"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "eea443eca48a6cc85bd054f93fb38c94f3d7e6fd4050cd15f442406ffa09b28e"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 30_116
PRIOR_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_SCIENTIFIC_UPDATE_400_GATE_FAILURE_PERMANENTLY_CLOSES_"
    "MASKED_PAIR_TUBELET_TIMING_SUCCESSORS"
)
PRIOR_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_SCIENTIFIC_GATE_FAILURE_AT_UPDATE_400_CLOSES_EXACT_V13_"
    "LEARNING_CURVE_AND_ALL_MASKED_PAIR_TUBELET_TIMING_SUCCESSORS"
)


# Future source graph.  These declarations grant no runtime authority.
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v1_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v1_runner.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_direct_egocentric_bev_state_jepa_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v1_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v1_"
    "source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "execution_authorization_2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
FROZEN_V11_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
FROZEN_V11_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
FROZEN_V11_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
FROZEN_V11_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_closure.py"
)
REUSED_SOURCE_PATHS = (
    "lewm/__init__.py",
    "lewm/benchmarks/__init__.py",
    "lewm/benchmarks/counterfactual.py",
    "lewm/benchmarks/finalize_shared_observable_camera_ray_jepa_v5_g2.py",
    "lewm/benchmarks/finalize_shared_observable_camera_ray_jepa_v5_g3.py",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py",
    FROZEN_V11_CONTRACT_RELATIVE_PATH,
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
    "lewm/benchmarks/shared_observable_camera_ray_jepa_v5_finalizer_core.py",
    "lewm/benchmarks/shared_observable_camera_ray_jepa_v5_runner_policy.py",
    "lewm/models/__init__.py",
    "lewm/models/egomotion_bev_jepa.py",
    "lewm/models/encoders.py",
    "lewm/models/lewm.py",
    "lewm/models/observable_camera_ray_evidence_v4.py",
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py",
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py",
    "lewm/models/observable_camera_ray_evidence_v4_training.py",
    "lewm/models/patch_whitened_action_residual_jepa.py",
    "lewm/models/phase2d_spatial_lewm.py",
    "lewm/models/predictor.py",
    "lewm/models/primitive_affordance.py",
    "lewm/models/rgb_masked_current_next_pair_tubelet_jepa_v11.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_authority.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v4_loss.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_registry_policy.py",
    "lewm/models/sigreg.py",
    "lewm/models/source_action_utility.py",
    "lewm/models/spatial_lewm.py",
    "lewm/models/spatial_predictor.py",
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_contract.py",
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_runner.py",
    "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py",
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_contract.py",
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_runner.py",
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_source_closure.py",
    "lewm/tests/test_patch_whitened_action_residual_jepa.py",
    "lewm/tests/test_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
    "scripts/check_go2_multires_probe_source_closure_v3.py",
    "scripts/check_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py",
    FROZEN_V11_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
    "scripts/launch_go2_rgb_jepa_encoder_pretraining_v1.py",
    FROZEN_V11_LAUNCHER_RELATIVE_PATH,
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py",
    FROZEN_V11_RUNNER_RELATIVE_PATH,
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
)
SOURCE_PATHS = tuple(sorted((*ADDITIVE_SOURCE_PATHS, *REUSED_SOURCE_PATHS)))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH,
    ARCHITECTURE_DECISION_RELATIVE_PATH,
    PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v1"
)


# Bound runtime identities.  These are declarations only and are never opened
# by this module.
RAW_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/manifest.json"
RAW_AUDIT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit_v13.json"
)
N320_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1"
)
N320_GATE_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/gate.json"
N320_CHECKPOINT_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/checkpoint.pt"
SCHEDULE_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v4/schedule.json"
)

RUNTIME_BINDINGS = {
    RAW_MANIFEST_RELATIVE_PATH: {
        "path": RAW_MANIFEST_RELATIVE_PATH,
        "file_sha256": (
            "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
        ),
        "content_sha256": (
            "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
        ),
        "byte_count": 311_598,
    },
    RAW_AUDIT_RELATIVE_PATH: {
        "path": RAW_AUDIT_RELATIVE_PATH,
        "file_sha256": (
            "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76"
        ),
        "content_sha256": (
            "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca"
        ),
        "byte_count": 26_975,
    },
    N320_GATE_RELATIVE_PATH: {
        "path": N320_GATE_RELATIVE_PATH,
        "file_sha256": (
            "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6"
        ),
        "content_sha256": (
            "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b"
        ),
        "byte_count": 7_960,
    },
    N320_CHECKPOINT_RELATIVE_PATH: {
        "path": N320_CHECKPOINT_RELATIVE_PATH,
        "file_sha256": (
            "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0"
        ),
        "content_sha256": (
            "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b"
        ),
        "byte_count": 13_777_100,
    },
    SCHEDULE_RELATIVE_PATH: {
        "path": SCHEDULE_RELATIVE_PATH,
        "file_sha256": (
            "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270"
        ),
        "content_sha256": (
            "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15"
        ),
        "byte_count": 607_373,
    },
}


# Frozen data and mappings.
TRAIN_ROLE_COUNTS = {"pairs": 4_262, "unique_endpoints": 7_777, "scenes": 72}
SELECTION_ROLE_COUNTS = {"pairs": 495, "unique_endpoints": 924, "scenes": 8}
SELECTION_NON_HOLD_PAIR_COUNT = 435
SELECTION_HOLD_PAIR_COUNT = 60
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
HOLD_ACTION_INDEX = 6
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
SELECTION_FAMILY_BINDINGS = {
    "large_enclosed_maze": {
        "scene_id": "large_enclosed_maze_d78318b1e87b",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
    "local_composite_motifs": {
        "scene_id": "local_composite_motifs_811b818f1914",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 56,
    },
    "loop_alias_stress": {
        "scene_id": "loop_alias_stress_aeb36ab10bc1",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
    "medium_enclosed_maze": {
        "scene_id": "medium_enclosed_maze_f30352cb052e",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
    "open_obstacle_field": {
        "scene_id": "open_obstacle_field_25cc6fe2de4f",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
    "rough_local_dynamics": {
        "scene_id": "rough_local_dynamics_0e631dbfbd46",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
    "small_enclosed_maze": {
        "scene_id": "small_enclosed_maze_16b0fc2c449b",
        "row_count": 47,
        "same_action_row_count": 46,
        "non_hold_row_count": 37,
    },
    "visual_sensor_stress": {
        "scene_id": "visual_sensor_stress_dc440a3fb679",
        "row_count": 64,
        "same_action_row_count": 64,
        "non_hold_row_count": 57,
    },
}
SELECTION_FAMILY_BINDINGS_SHA256 = (
    "c39efe48afd6d4c02a24af77f1f11e7f6cd5a69d571b0a9416924b07bbacbb11"
)
TARGET_MAPPING_BINDINGS = {
    "train": {
        "row_count": 4_262,
        "input_rows_sha256": (
            "bb119abb33b7c56f3c1d96e7cb1b52fbe4d2db27d80df4f95a5b1cd9d0cf729e"
        ),
        "mapping_sha256": (
            "c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b"
        ),
        "same_action_eligible_count": 4_237,
        "fallback_count": 25,
        "non_singleton_primitive_group_count": 578,
        "primitive_group_count": 603,
    },
    "checkpoint_selection": {
        "row_count": 495,
        "input_rows_sha256": (
            "81f85cdf0ad00ec68918f5eeb7637bf20aa3f17f5615bedfb10e6a4859eb91f1"
        ),
        "mapping_sha256": (
            "95d42273a8319316ad68781cb2158146e7672eda529984c3aeddc0937d87a9c1"
        ),
        "same_action_eligible_count": 494,
        "fallback_count": 1,
        "non_singleton_primitive_group_count": 71,
        "primitive_group_count": 72,
    },
}
SELECTION_ACTION_PERMUTATION_BINDING = {
    "row_count": 495,
    "input_rows_sha256": TARGET_MAPPING_BINDINGS[
        "checkpoint_selection"
    ]["input_rows_sha256"],
    "mapping_sha256": (
        "2740be362829c172a06aebae0d077e69ede8af80cbf6f00569eb460dc559bb0f"
    ),
    "changed_action_count": 495,
    "scene_count": 8,
    "scene_size_histogram": {"47": 1, "64": 7},
    "shift_histogram": {"8": 7, "13": 1},
}
FIXED_MAPPED_NEGATIVE_RULE = {
    "sort_key": "lowercase_hex_content_sha256_bytewise_ascending",
    "non_singleton": (
        "cyclic_next_row_within_same_scene_and_primitive_group_wrapping_once"
    ),
    "singleton": (
        "cyclic_next_row_in_complete_same_scene_content_order_wrapping_once"
    ),
    "mapped_value": "next_rgb_endpoint_only",
    "same_role_and_scene_required": True,
    "different_next_endpoint_required": True,
    "random_draw_count": 0,
}


# Exact six-call training/control graph.
CALL_GRAPH = {
    "O_current_rgb": {
        "stack": "one_weight_shared_online_encoder_decoder_state_head",
        "gradient": "G_current_and_causal_prediction_objectives",
        "consumers": ["G_current", "causal_predictor"],
        "causal_or_deployment_path": True,
    },
    "O_next_rgb": {
        "stack": "same_weight_shared_online_encoder_decoder_state_head",
        "gradient": "G_next_only",
        "consumers": ["G_next"],
        "causal_or_deployment_path": False,
    },
    "T_next_rgb": {
        "stack": "detached_ema_encoder_decoder_state_head",
        "gradient": "none",
        "consumers": ["J_true_next_target", "C_true_next_target"],
        "causal_or_deployment_path": False,
    },
    "T_current_rgb": {
        "stack": "detached_ema_encoder_decoder_state_head",
        "gradient": "none",
        "consumers": ["C_non_hold_current_target_negative_only"],
        "causal_or_deployment_path": False,
    },
    "T_fixed_negative_rgb": {
        "stack": "detached_ema_encoder_decoder_state_head",
        "gradient": "none",
        "consumers": ["C_fixed_mapped_negative_target"],
        "causal_or_deployment_path": False,
    },
    "O_fixed_negative_rgb": {
        "stack": "same_weight_shared_online_encoder_decoder_state_head",
        "gradient": "none_observation_only",
        "consumers": ["wrong_rgb_grounding_control_only"],
        "causal_or_deployment_path": False,
    },
}
CAUSAL_RUNTIME_INPUTS = ("current_rgb", "executed_action_identity")
TRAINING_ROW_ARRAY_VALUES = (
    "current_rgb",
    "next_rgb",
    "fixed_negative_rgb",
    "raster_labels.u1",
)
ALLOWED_SUPERVISION_ARRAYS = ("raster_labels.u1",)
FORBIDDEN_CALL_GRAPH_CONSUMERS = (
    "O_next_rgb_to_J",
    "O_next_rgb_to_C",
    "O_next_rgb_to_transition",
    "O_fixed_negative_rgb_to_optimizer_loss",
    "O_fixed_negative_rgb_to_transition",
    "next_rgb_to_causal_predictor",
    "fixed_negative_rgb_to_causal_predictor",
    "raster_labels_to_encoder_or_transition",
    "hidden_64d_decoder_bypass",
)


# Architecture, objective, optimizer, schedule, and caps.
BASE_INITIALIZATION_SEED = 20260712
N320_FIT_SEED = 20260710
SCHEDULE_SEED = 20260713
TARGET_EMA_MOMENTUM = 0.996
MAXIMUM_ATTEMPTS = 1
ATTEMPT_INDEX = 1
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
GPU_ACTIVE_TIME_CAP_MINUTES = 60
EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
CHECKPOINT_UPDATES = (100, 400, 1_000)
OBSERVATION_UPDATES = (0, *CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}

AGGREGATE_RASTER_ENDPOINT_COUNT = 924
AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256 = (
    "dd84fc73e14056c9d6c8f7c066c2dcafe9726827193c42982d51f412ea744fa4"
)
ROUGH_RASTER_FAMILY = "rough_local_dynamics"
ROUGH_RASTER_ENDPOINT_COUNT = 123
RASTER_CLASS_ORDER = ("UNKNOWN", "FREE", "OCCUPIED")
RASTER_CONFUSION_ORIENTATION = "target_rows_predicted_columns"
RASTER_ARGMAX_TIE_BREAK = "lowest_class_index"
RASTER_NLL_PROBABILITY_FLOOR = "float32_machine_epsilon"

MODEL_PARAMETER_INVENTORY = {
    "encoder": {
        "parameter_count": 2_747_520,
        "tensor_count": 78,
        "ordered_parameter_name_sha256": (
            "8b83921e9766a68b59b35d1c5ef15dea6db13aeb4a6c91c2c13ab2e22d8b1c5e"
        ),
    },
    "decoder_state": {
        "parameter_count": 370_051,
        "tensor_count": 21,
        "ordered_parameter_name_sha256": (
            "c03efa2cc957b98959cb2602a21d9737f2b57c600f94e06b7facd7bee112a4d1"
        ),
    },
    "predictor": {
        "parameter_count": 160_134,
        "tensor_count": 10,
        "ordered_parameter_name_sha256": (
            "e9c7d89552bdda55e323b87ac955589ffb2cfe487947919254f79129c35abdff"
        ),
    },
    "detached_target_encoder_decoder_state": {
        "parameter_count": 3_117_571,
        "tensor_count": 99,
        "ordered_parameter_name_sha256": (
            "cfd687efeec2ac6864c59ab02aedbd353ddc0ebb98477bb715262c7dafa8fc6c"
        ),
    },
    "total": {
        "parameter_count": 6_395_276,
        "tensor_count": 208,
    },
}

OBSERVATION_METRIC_CONTRACT = {
    "G_and_J": {
        "population": "all_495_checkpoint_selection_pairs",
        "reduction": "plain_row_mean",
        "gradient": "none",
        "formulas_and_call_isolation": "exact_training_objective",
    },
    "action_retrieval": {
        "population": 495,
        "logits": (
            "nine_negative_action_candidate_energies_over_detached_C_row_scale"
        ),
        "nll": "plain_cross_entropy_executed_action_target",
        "top1_tie_break": "lowest_action_vocabulary_index",
        "macro_balanced_accuracy": "mean_of_nine_per_executed_action_recalls",
        "hardest_wrong": "minimum_energy_of_other_eight_actions_per_row",
        "positive_scene_margin": (
            "strictly_positive_row_mean_hardest_wrong_minus_executed_energy"
        ),
    },
    "same_action_target": {
        "population": 494,
        "fallback_rows_excluded": 1,
        "logits_in_order": [
            "negative_correct_energy_over_detached_C_row_scale",
            "negative_deranged_energy_over_detached_C_row_scale",
        ],
        "nll": "plain_two_logit_cross_entropy_target_zero",
        "strict_win": "correct_energy_strictly_less_than_deranged_energy",
        "positive_scene_margin": (
            "strictly_positive_eligible_row_mean_deranged_minus_correct_energy"
        ),
    },
    "aggregate_raster": {
        "population": AGGREGATE_RASTER_ENDPOINT_COUNT,
        "ordered_endpoint_identity_sha256": (
            AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256
        ),
        "construction": (
            "deduplicate_current_and_next_endpoint_identities_within_family_"
            "then_sort_and_pool_all_families"
        ),
    },
    "rough_raster": {
        "population": ROUGH_RASTER_ENDPOINT_COUNT,
        "family": ROUGH_RASTER_FAMILY,
        "construction": "same_endpoint_protocol_restricted_to_frozen_family",
    },
    "raster_reduction": {
        "evaluate_each_endpoint_once": True,
        "call": "no_gradient_O_endpoint_rgb",
        "label": "endpoints_own_target_raster_labels",
        "class_order": list(RASTER_CLASS_ORDER),
        "prediction": "argmax_direct_softmax",
        "tie_break": RASTER_ARGMAX_TIE_BREAK,
        "confusion_shape": [3, 3],
        "confusion_orientation": RASTER_CONFUSION_ORIENTATION,
        "class_recall": "diagonal_divided_by_target_row_count",
        "balanced_accuracy": "plain_mean_of_present_target_class_recalls",
        "nll": "plain_per_cell_mean_negative_log_target_probability",
        "nll_probability_floor": RASTER_NLL_PROBABILITY_FLOOR,
        "required_receipts": [
            "endpoint_count",
            "family_membership",
            "label_identity",
            "confusion_counts",
            "nll_count",
        ],
    },
    "wrong_rgb": {
        "population": 495,
        "separate_from_endpoint_raster_populations": True,
    },
}


def model_config() -> dict[str, Any]:
    return {
        "encoder": {
            "rgb_resolution": 112,
            "patch_size": 7,
            "dim": 192,
            "depth": 6,
            "heads": 6,
            "mlp_ratio": 4,
        },
        "bev_decoder": {
            "attention": "learned_global_cross_attention",
            "internal_dim": 64,
            "output_shape": [64, 64, 64],
            "forward_cell_centres_metres": [-0.95, 5.35],
            "left_cell_centres_metres": [-3.15, 3.15],
            "cell_metres": 0.1,
        },
        "state_head": {
            "shape": "one_1x1_projection_to_three_logits",
            "classes_in_order": ["UNKNOWN", "FREE", "OCCUPIED"],
            "class_values": {"UNKNOWN": 0, "FREE": 1, "OCCUPIED": 2},
            "sole_state_bottleneck": True,
            "hidden_bypass_authorized": False,
        },
        "transition": {
            "shape": "BevResidualPredictor",
            "bev_dim": 3,
            "action_dim": 9,
            "hidden_dim": 128,
            "inputs": ["current_learned_bev_state", "nine_way_one_hot_action"],
            "legacy_three_vector": "internally_supplied_exact_zero_vector",
            "final_residual_layer_initialization": "exact_zero",
            "warp_calls": 0,
        },
        "target": {
            "inventory": ["encoder", "bev_decoder", "state_head"],
            "hard_sync_count_before_update_zero": 1,
            "ema_decay": TARGET_EMA_MOMENTUM,
            "ema_updates": "once_after_every_optimizer_update",
            "predictor_target_copy": False,
            "gradient": "none",
        },
        "initialization": {
            "n320_encoder_only_migration": True,
            "n320_fit_seed": N320_FIT_SEED,
            "fresh_parameter_seed": BASE_INITIALIZATION_SEED,
            "fresh_parameters": ["bev_decoder", "state_head", "predictor"],
            "prior_v1_through_v13_runtime_reuse": False,
        },
        "call_graph": {key: dict(value) for key, value in CALL_GRAPH.items()},
        "parameter_inventory": {
            group: dict(binding)
            for group, binding in MODEL_PARAMETER_INVENTORY.items()
        },
    }


def objective_contract() -> dict[str, Any]:
    return {
        "G": {
            "formula": "mean(G_current,G_next)",
            "current_call": "O_current_rgb",
            "next_call": "O_next_rgb",
            "loss": "hierarchical_raster_cross_entropy_v4",
            "occupied_vs_rest_bce_weight": 0.5,
            "free_vs_unknown_nonoccupied_bce_weight": 0.5,
            "hard_label_array": "raster_labels.u1",
        },
        "J": {
            "formula": "soft_hierarchical_probability_energy",
            "prediction": "executed_action_predicted_probabilities",
            "target": "detached_T_next_rgb_probabilities",
            "occupied_bce_weight": 0.5,
            "conditional_free_bce_weight": 0.5,
        },
        "C": {
            "formula": "mean(row_NCE/log(candidate_count))",
            "positive": "executed_prediction_vs_T_next_rgb",
            "wrong_action_negatives": 8,
            "mapped_negative": "executed_prediction_vs_T_fixed_negative_rgb",
            "non_hold_only_negative": "executed_prediction_vs_T_current_rgb",
            "hold_candidate_count": 10,
            "non_hold_candidate_count": 11,
            "all_train_rows_included": 4_262,
            "train_fallback_rows_included": 25,
            "candidate_energy": "same_soft_hierarchy_as_J",
            "logit": "negative_energy_over_detached_row_mean_energy",
            "mean_energy_minimum": 1e-6,
        },
        "wrong_rgb_control": {
            "gradient": "none",
            "correct_call": "O_next_rgb",
            "wrong_call": "O_fixed_negative_rgb",
            "label": "same_true_next_raster_label",
            "loss": "hierarchical_raster_cross_entropy_v4",
            "population": 495,
            "fallback_rows_included": 1,
            "scene_win": "mean_correct_loss_strictly_less_than_mean_wrong_loss",
        },
        "same_action_target_metrics": {
            "population": 494,
            "fallback_rows_excluded": 1,
        },
        "total": "1*G/log(2) + 1*J/log(2) + 1*normalized_C",
        "absent": [
            "camera_loss",
            "ray_loss",
            "depth_loss",
            "ground_loss",
            "rasterizer",
            "equivariance_loss",
            "warp_loss",
            "variance_loss",
            "auxiliary_loss",
        ],
    }


def optimizer_contract() -> dict[str, Any]:
    return {
        "name": "AdamW",
        "precision": "float32",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "learning_rates": {
            "encoder": 1e-4,
            "decoder_state_predictor": 3e-4,
        },
        "gradient_clipping": {
            "encoder_decoder_state_joint_norm": 1.0,
            "predictor_separate_norm": 1.0,
        },
        "target_parameters_excluded": True,
    }


def build_schedule_identity() -> dict[str, Any]:
    return {
        "source": dict(RUNTIME_BINDINGS[SCHEDULE_RELATIVE_PATH]),
        "seed": SCHEDULE_SEED,
        "updates": MAXIMUM_UPDATES,
        "presentations": MAXIMUM_PRESENTATIONS,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoints": list(CHECKPOINT_UPDATES),
        "observation_updates": list(OBSERVATION_UPDATES),
        "prefix_sha256": {
            str(update): digest
            for update, digest in SCHEDULE_PREFIX_SHA256.items()
        },
        "one_scheduled_pair_is_one_presentation": True,
        "preserve_rows_roles_order_and_mappings": True,
    }


UPDATE_ZERO_ACTION_TOLERANCE = 8.0 * (2.0 ** -23)
GATE_THRESHOLDS = {
    100: {
        "action_nll_strictly_less_than": math.log(9.0),
        "action_macro_balanced_accuracy_strictly_greater_than": 1.0 / 9.0,
        "correct_rgb_scene_wins_minimum": 6,
    },
    400: {
        "G_update_zero_factor_maximum": 0.90,
        "J_update_zero_factor_maximum": 0.90,
        "action_nll_strictly_less_than": 0.99 * math.log(9.0),
        "action_macro_balanced_accuracy_strictly_greater_than": 0.15,
        "hardest_wrong_positive_scene_count_minimum": 4,
        "same_action_target_nll_strictly_less_than": 0.99 * math.log(2.0),
        "same_action_target_strict_win_rate_minimum": 0.60,
        "correct_rgb_scene_wins_minimum": 8,
    },
    1_000: {
        "aggregate_raster_balanced_accuracy_strictly_greater_than": (
            0.9009460724448773
        ),
        "aggregate_free_recall_strictly_greater_than": 0.91637020862468,
        "aggregate_occupied_recall_strictly_greater_than": (
            0.8059679976935274
        ),
        "aggregate_raster_nll_strictly_less_than": 0.18704089070408247,
        "rough_raster_balanced_accuracy_strictly_greater_than": (
            0.7719525130620232
        ),
        "rough_raster_occupied_recall_strictly_greater_than": (
            0.4319466882067851
        ),
        "correct_rgb_scene_wins_minimum": 8,
        "action_nll_strictly_less_than": 0.95 * math.log(9.0),
        "action_macro_balanced_accuracy_strictly_greater_than": 2.0 / 9.0,
        "hardest_wrong_positive_scene_count_minimum": 6,
        "same_action_target_nll_strictly_less_than": 0.95 * math.log(2.0),
        "same_action_target_strict_win_rate_minimum": 0.65,
        "target_positive_scene_count_minimum": 6,
    },
}

CONTROL_CONTINUE_UPDATE_100 = "CONTINUE_AFTER_UPDATE_100_GATE"
CONTROL_CONTINUE_UPDATE_400 = "CONTINUE_AFTER_UPDATE_400_GATE"
CONTROL_PASS = "PASS_DIRECT_BEV_STATE_JEPA_PERCEPTION_GATE_REQUALIFICATION_ONLY"
CONTROL_UPDATE_ZERO_FAIL = "FAIL_UPDATE_ZERO_INTEGRITY_GATE_TERMINAL_NO_RETRY"
CONTROL_UPDATE_100_FAIL = "FAIL_UPDATE_100_DIRECTIONAL_GATE_TERMINAL_NO_RETRY"
CONTROL_UPDATE_400_FAIL = "FAIL_UPDATE_400_MECHANISM_GATE_TERMINAL_NO_RETRY"
CONTROL_UPDATE_1000_FAIL = "FAIL_UPDATE_1000_PERCEPTION_GATE_TERMINAL_NO_RETRY"
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_ZERO_FAIL, "CONTINUE_AFTER_UPDATE_ZERO_GATE"),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    400: (CONTROL_UPDATE_400_FAIL, CONTROL_CONTINUE_UPDATE_400),
    1_000: (CONTROL_UPDATE_1000_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be one finite number")
    return float(value)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate exactly one preregistered stage without opening any payload."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    _exact_bool(prior_gates_passed, name="prior_gates_passed")
    conjuncts: dict[str, bool] = {"prior_gates_passed": prior_gates_passed}

    if update == 0:
        for field in (
            "three_logit_bottleneck_exact",
            "no_hidden_or_auxiliary_bypass",
            "prediction_is_exact_persistence",
            "all_nine_action_predictions_bitwise_equal",
            "target_parameters_gradient_free",
            "intended_online_path_gradient_nonzero",
            "six_call_graph_isolation_exact",
            "all_registered_values_finite",
        ):
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        action_nll = _finite_number(metrics.get("action_nll"), name="action_nll")
        action_ba = _finite_number(
            metrics.get("action_macro_balanced_accuracy"),
            name="action_macro_balanced_accuracy",
        )
        conjuncts["action_nll_equals_log9"] = (
            abs(action_nll - math.log(9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
        conjuncts["action_macro_balanced_accuracy_equals_one_ninth"] = (
            abs(action_ba - (1.0 / 9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
    elif update == 100:
        if update_zero is None:
            raise ValueError("update_zero baselines are required")
        g = _finite_number(metrics.get("G"), name="G")
        j = _finite_number(metrics.get("J"), name="J")
        g0 = _finite_number(update_zero.get("G"), name="update_zero.G")
        j0 = _finite_number(update_zero.get("J"), name="update_zero.J")
        conjuncts.update({
            "G_strictly_decreased": g < g0,
            "J_strictly_decreased": j < j0,
            "action_nll_below_log9": (
                _finite_number(metrics.get("action_nll"), name="action_nll")
                < GATE_THRESHOLDS[100]["action_nll_strictly_less_than"]
            ),
            "action_macro_balanced_accuracy_above_one_ninth": (
                _finite_number(
                    metrics.get("action_macro_balanced_accuracy"),
                    name="action_macro_balanced_accuracy",
                )
                > GATE_THRESHOLDS[100][
                    "action_macro_balanced_accuracy_strictly_greater_than"
                ]
            ),
            "correct_rgb_wins_at_least_six_scenes": (
                _finite_number(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                )
                >= 6
            ),
            "registered_values_finite": _exact_bool(
                metrics.get("all_registered_values_finite"),
                name="all_registered_values_finite",
            ),
            "state_nonconstant": _exact_bool(
                metrics.get("state_nonconstant"), name="state_nonconstant"
            ),
        })
    elif update == 400:
        if update_zero is None:
            raise ValueError("update_zero baselines are required")
        g0 = _finite_number(update_zero.get("G"), name="update_zero.G")
        j0 = _finite_number(update_zero.get("J"), name="update_zero.J")
        conjuncts.update({
            "G_at_most_point90_update_zero": (
                _finite_number(metrics.get("G"), name="G") <= 0.90 * g0
            ),
            "J_at_most_point90_update_zero": (
                _finite_number(metrics.get("J"), name="J") <= 0.90 * j0
            ),
            "action_nll_below_point99_log9": (
                _finite_number(metrics.get("action_nll"), name="action_nll")
                < GATE_THRESHOLDS[400]["action_nll_strictly_less_than"]
            ),
            "action_macro_balanced_accuracy_above_point15": (
                _finite_number(
                    metrics.get("action_macro_balanced_accuracy"),
                    name="action_macro_balanced_accuracy",
                )
                > 0.15
            ),
            "hardest_wrong_positive_at_least_four_scenes": (
                _finite_number(
                    metrics.get("hardest_wrong_positive_scene_count"),
                    name="hardest_wrong_positive_scene_count",
                )
                >= 4
            ),
            "same_action_target_nll_below_point99_log2": (
                _finite_number(
                    metrics.get("same_action_target_nll"),
                    name="same_action_target_nll",
                )
                < GATE_THRESHOLDS[400][
                    "same_action_target_nll_strictly_less_than"
                ]
            ),
            "same_action_target_strict_win_rate_at_least_point60": (
                _finite_number(
                    metrics.get("same_action_target_strict_win_rate"),
                    name="same_action_target_strict_win_rate",
                )
                >= 0.60
            ),
            "correct_rgb_wins_all_eight_scenes": (
                _finite_number(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                )
                >= 8
            ),
        })
    else:
        thresholds = GATE_THRESHOLDS[1_000]
        conjuncts.update({
            "aggregate_raster_balanced_accuracy": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                )
                > thresholds[
                    "aggregate_raster_balanced_accuracy_strictly_greater_than"
                ]
            ),
            "aggregate_free_recall": (
                _finite_number(
                    metrics.get("aggregate_free_recall"),
                    name="aggregate_free_recall",
                )
                > thresholds["aggregate_free_recall_strictly_greater_than"]
            ),
            "aggregate_occupied_recall": (
                _finite_number(
                    metrics.get("aggregate_occupied_recall"),
                    name="aggregate_occupied_recall",
                )
                > thresholds[
                    "aggregate_occupied_recall_strictly_greater_than"
                ]
            ),
            "aggregate_raster_nll": (
                _finite_number(
                    metrics.get("aggregate_raster_nll"),
                    name="aggregate_raster_nll",
                )
                < thresholds["aggregate_raster_nll_strictly_less_than"]
            ),
            "rough_raster_balanced_accuracy": (
                _finite_number(
                    metrics.get("rough_raster_balanced_accuracy"),
                    name="rough_raster_balanced_accuracy",
                )
                > thresholds[
                    "rough_raster_balanced_accuracy_strictly_greater_than"
                ]
            ),
            "rough_raster_occupied_recall": (
                _finite_number(
                    metrics.get("rough_raster_occupied_recall"),
                    name="rough_raster_occupied_recall",
                )
                > thresholds[
                    "rough_raster_occupied_recall_strictly_greater_than"
                ]
            ),
            "correct_rgb_wins_all_eight_scenes": (
                _finite_number(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                )
                >= 8
            ),
            "action_nll_below_point95_log9": (
                _finite_number(metrics.get("action_nll"), name="action_nll")
                < thresholds["action_nll_strictly_less_than"]
            ),
            "action_macro_balanced_accuracy_above_two_ninths": (
                _finite_number(
                    metrics.get("action_macro_balanced_accuracy"),
                    name="action_macro_balanced_accuracy",
                )
                > thresholds[
                    "action_macro_balanced_accuracy_strictly_greater_than"
                ]
            ),
            "hardest_wrong_positive_at_least_six_scenes": (
                _finite_number(
                    metrics.get("hardest_wrong_positive_scene_count"),
                    name="hardest_wrong_positive_scene_count",
                )
                >= 6
            ),
            "same_action_target_nll_below_point95_log2": (
                _finite_number(
                    metrics.get("same_action_target_nll"),
                    name="same_action_target_nll",
                )
                < thresholds[
                    "same_action_target_nll_strictly_less_than"
                ]
            ),
            "same_action_target_strict_win_rate_at_least_point65": (
                _finite_number(
                    metrics.get("same_action_target_strict_win_rate"),
                    name="same_action_target_strict_win_rate",
                )
                >= 0.65
            ),
            "target_positive_at_least_six_scenes": (
                _finite_number(
                    metrics.get("target_positive_scene_count"),
                    name="target_positive_scene_count",
                )
                >= 6
            ),
        })

    passed = all(conjuncts.values())
    fail_control, pass_control = GATE_CONTROLS[update]
    return {
        "update": update,
        "passed": passed,
        "control": pass_control if passed else fail_control,
        "conjuncts": conjuncts,
        "thresholds": (
            {} if update == 0 else dict(GATE_THRESHOLDS[update])
        ),
    }


# Custody and authority.  Authorized label/RGB/index opens are counted
# honestly; only forbidden categories are required to remain zero.
ALLOWED_ACCESS_COUNTER_FIELDS = (
    "raw_manifest_open_count",
    "raw_audit_open_count",
    "pair_index_open_count",
    "endpoint_index_open_count",
    "current_rgb_row_request_count",
    "next_rgb_row_request_count",
    "fixed_negative_rgb_row_request_count",
    "endpoint_rgb_row_request_count",
    "rgb_cache_hit_count",
    "rgb_cache_miss_count",
    "rgb_physical_file_open_count",
    "raster_label_row_request_count",
    "raster_label_row_cache_hit_count",
    "raster_label_row_cache_miss_count",
    "raster_label_underlying_array_cache_hit_count",
    "raster_label_underlying_array_cache_miss_count",
    "raster_label_physical_array_open_count",
    "n320_gate_open_count",
    "n320_checkpoint_open_count",
    "schedule_open_count",
)
FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS = (
    "camera_supervision_array_open_count",
    "ray_supervision_array_open_count",
    "depth_supervision_array_open_count",
    "ground_supervision_array_open_count",
    "other_supervision_array_open_count",
    "general_raw_frame_loader_call_count",
    "prior_runtime_output_open_count",
    "rejected_checkpoint_open_count",
    "training_trace_read_count",
    "written_checkpoint_read_count",
    "g2_open_count",
    "navigation_open_count",
    "heldout_open_count",
    "sealed_open_count",
    "production_input_open_count",
    "deployment_input_open_count",
    "observer_rerun_count",
)
ACCESS_COUNTER_FIELDS = (
    *ALLOWED_ACCESS_COUNTER_FIELDS,
    *FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS,
)
RUNNER_ACCESS_COUNTER_MAPPING = {
    "current_rgb_row_request_count": "rgb_request_count.current",
    "next_rgb_row_request_count": "rgb_request_count.next",
    "fixed_negative_rgb_row_request_count": "rgb_request_count.fixed_negative",
    "endpoint_rgb_row_request_count": "rgb_request_count.endpoint",
    "rgb_cache_hit_count": "sum(rgb_cache_hit_count.*)",
    "rgb_cache_miss_count": "sum(rgb_cache_miss_count.*)",
    "rgb_physical_file_open_count": "sum(rgb_physical_read_success_count.*)",
    "raster_label_row_request_count": "sum(raster_row_request_count.*)",
    "raster_label_row_cache_hit_count": "sum(raster_row_cache_hit_count.*)",
    "raster_label_row_cache_miss_count": "sum(raster_row_cache_miss_count.*)",
    "raster_label_underlying_array_cache_hit_count": (
        "raster_underlying_array_cache_hit_count"
    ),
    "raster_label_underlying_array_cache_miss_count": (
        "raster_underlying_array_cache_miss_count"
    ),
    "raster_label_physical_array_open_count": (
        "raster_physical_array_open_success_count"
    ),
}
PROHIBITED_RUNTIME_CATEGORIES = (
    "other_supervision_arrays",
    "general_raw_frame_loader",
    "prior_attempt_roots",
    "rejected_checkpoints",
    "checkpoint_or_trace_reads_after_write",
    "g2",
    "navigation",
    "heldout",
    "sealed",
    "production",
    "deployment",
)

DOWNSTREAM_DENIALS = {
    "checkpoint_qualified": False,
    "physical_requalification_authorized": False,
    "matched_no_jepa_arm_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_resume_repair_recovery_replacement_or_second_seed_authorized": False,
    "v2_parameter_loss_or_timing_successor_authorized": False,
}
SOURCE_ONLY_AUTHORITY = {
    "implementation_authorized": True,
    "source_only_tests_authorized": True,
    "source_closure_authorized": True,
    "independent_source_review_authorized": True,
    "execution_authorized": False,
    "generated_input_access_authorized": False,
    "dataset_rgb_or_label_access_authorized": False,
    "checkpoint_tensor_or_runtime_output_access_authorized": False,
    "gpu_training_or_evaluation_authorized": False,
    "heldout_or_sealed_access_authorized": False,
    "navigation_g2_production_or_promotion_authorized": False,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    "one_exact_fresh_attempt_authorized": True,
    "attempt_index": ATTEMPT_INDEX,
    "maximum_attempts": MAXIMUM_ATTEMPTS,
    "maximum_updates": MAXIMUM_UPDATES,
    "maximum_presentations": MAXIMUM_PRESENTATIONS,
    "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "n320_encoder_initialization_only_authorized": True,
    "narrow_row_array_loader_only_authorized": True,
    "allowed_row_array_values": list(TRAINING_ROW_ARRAY_VALUES),
    "allowed_supervision_arrays": list(ALLOWED_SUPERVISION_ARRAYS),
    "other_supervision_array_access_authorized": False,
    "general_raw_frame_loader_authorized": False,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent_before_reservation": True,
    "phase_b_authorized": False,
    **DOWNSTREAM_DENIALS,
}

AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_DIRECT_BEV_STATE_JEPA_V1_PROBE"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
NORMAL_RECEIPT_PATHS = (
    "reservation.json",
    "metrics.json",
    "artifact.json",
    "access.json",
    "result.json",
    "completed.json",
)
OPERATIONAL_FAILURE_RECEIPT_PATHS = ("failure.json", "completed.json")
OPERATIONAL_FAILURE_STATUS = "TERMINAL_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
RESERVATION_PUBLICATION_FAILURE_STATUS = (
    "TERMINAL_RESERVATION_PUBLICATION_FAILURE_NO_RETRY"
)

SCIENTIFIC_REVIEW_CHECKS = {
    "corrected_preregistration_and_architecture_hashes_exact": True,
    "v13_terminal_audit_bound_and_family_closed": True,
    "six_call_online_target_graph_and_isolation_exact": True,
    "three_logit_state_is_sole_jepa_predictor_navigation_state": True,
    "online_next_gradients_only_through_G_next": True,
    "wrong_rgb_online_call_is_observation_only_and_no_gradient": True,
    "ema_targets_detached_and_predictor_has_no_target_copy": True,
    "mapped_negative_rule_hashes_fallbacks_and_populations_exact": True,
    "training_C_4262_wrong_rgb_495_same_action_metrics_494_exact": True,
    "raster_labels_u1_is_only_supervision_array": True,
    "authorized_and_forbidden_access_counters_are_honest": True,
    "model_objective_optimizer_schedule_seeds_and_caps_exact": True,
    "all_four_staged_conjunctive_gates_and_comparators_exact": True,
    "one_fresh_attempt_no_retry_resume_repair_replacement_or_v2": True,
    "no_runtime_heldout_navigation_or_downstream_authority": True,
}


def validate_access_counters(value: object) -> dict[str, int]:
    if type(value) is not dict or set(value) != set(ACCESS_COUNTER_FIELDS):
        raise PermissionError("access counter inventory changed")
    for field in ALLOWED_ACCESS_COUNTER_FIELDS:
        if type(value[field]) is not int or value[field] < 0:
            raise PermissionError(f"authorized access counter invalid: {field}")
    for field in FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS:
        if type(value[field]) is not int or value[field] != 0:
            raise PermissionError(f"forbidden access counter nonzero: {field}")
    rgb_requests = sum(
        value[field]
        for field in (
            "current_rgb_row_request_count",
            "next_rgb_row_request_count",
            "fixed_negative_rgb_row_request_count",
            "endpoint_rgb_row_request_count",
        )
    )
    if (
        rgb_requests
        != value["rgb_cache_hit_count"] + value["rgb_cache_miss_count"]
        or value["rgb_cache_miss_count"]
        != value["rgb_physical_file_open_count"]
    ):
        raise PermissionError("RGB request/cache/physical-open accounting changed")
    if (
        value["raster_label_row_request_count"]
        != value["raster_label_row_cache_hit_count"]
        + value["raster_label_row_cache_miss_count"]
        or value["raster_label_row_cache_miss_count"]
        != value["raster_label_underlying_array_cache_hit_count"]
        + value["raster_label_underlying_array_cache_miss_count"]
        or value["raster_label_underlying_array_cache_miss_count"]
        != value["raster_label_physical_array_open_count"]
    ):
        raise PermissionError(
            "raster-label request/cache/physical-open accounting changed"
        )
    return {field: value[field] for field in ACCESS_COUNTER_FIELDS}


def validate_failure_status_chain(value: object) -> dict[str, str]:
    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("failure receipt statuses are not one exact gate control")
    return dict(value)


def runtime_authorization_template() -> dict[str, Any]:
    return {
        "raw": {
            "root": RAW_ROOT_RELATIVE_PATH,
            "manifest": dict(RUNTIME_BINDINGS[RAW_MANIFEST_RELATIVE_PATH]),
            "audit": dict(RUNTIME_BINDINGS[RAW_AUDIT_RELATIVE_PATH]),
            "roles": {
                "train": dict(TRAIN_ROLE_COUNTS),
                "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            },
            "grant": {
                "narrow_loader": "_row_array_only",
                "allowed_values": list(TRAINING_ROW_ARRAY_VALUES),
                "allowed_supervision_arrays": list(ALLOWED_SUPERVISION_ARRAYS),
                "other_supervision_array_access_authorized": False,
                "general_raw_frame_loader_authorized": False,
                "target_mapping_bindings": {
                    role: dict(binding)
                    for role, binding in TARGET_MAPPING_BINDINGS.items()
                },
                "selection_action_permutation_binding": dict(
                    SELECTION_ACTION_PERMUTATION_BINDING
                ),
                "selection_family_bindings": {
                    family: dict(binding)
                    for family, binding in SELECTION_FAMILY_BINDINGS.items()
                },
                "selection_family_bindings_sha256": (
                    SELECTION_FAMILY_BINDINGS_SHA256
                ),
                "fixed_mapped_negative_rule": dict(FIXED_MAPPED_NEGATIVE_RULE),
            },
        },
        "n320": {
            "root": N320_ROOT_RELATIVE_PATH,
            "gate": dict(RUNTIME_BINDINGS[N320_GATE_RELATIVE_PATH]),
            "checkpoint": dict(RUNTIME_BINDINGS[N320_CHECKPOINT_RELATIVE_PATH]),
            "fit_seed": N320_FIT_SEED,
            "fit_size": 320,
            "fit_updates": 40_000,
            "required_gate_check_count": 26,
            "encoder_only_migration": True,
        },
        "schedule": build_schedule_identity(),
        "access_counter_fields": {
            "authorized_nonnegative": list(ALLOWED_ACCESS_COUNTER_FIELDS),
            "forbidden_exact_zero": list(FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS),
            "equations": {
                "rgb": (
                    "current_plus_next_plus_fixed_negative_plus_endpoint_"
                    "requests_equals_"
                    "cache_hits_plus_cache_misses_and_misses_equal_physical_opens"
                ),
                "raster_labels.u1": (
                    "row_requests_equal_row_cache_hits_plus_row_cache_misses;"
                    "row_cache_misses_equal_underlying_array_hits_plus_"
                    "underlying_array_misses;underlying_array_misses_equal_"
                    "physical_array_opens"
                ),
            },
            "runner_counter_mapping": dict(RUNNER_ACCESS_COUNTER_MAPPING),
        },
    }


def science_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "scientific_question": (
            "can_a_direct_rgb_grounded_three_class_egocentric_bev_state_and_"
            "action_conditioned_jepa_transition_learn_useful_maze_perception"
        ),
        "repository_goal": (
            "fully_learned_perception_only_rgb_jepa_navigation_validated_"
            "later_on_untouched_externally_custodied_heldout_mazes"
        ),
        "governing_documents": {
            "preregistration": preregistration_binding(),
            "preregistration_independent_review": (
                preregistration_independent_review_binding()
            ),
            "architecture_decision": architecture_decision_binding(),
            "prior_terminal_audit": prior_terminal_audit_binding(),
        },
        "model": model_config(),
        "data": {
            "roles": ["train", "checkpoint_selection"],
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "target_mapping_bindings": {
                role: dict(binding)
                for role, binding in TARGET_MAPPING_BINDINGS.items()
            },
            "fixed_mapped_negative_rule": dict(FIXED_MAPPED_NEGATIVE_RULE),
            "selection_action_permutation_binding": dict(
                SELECTION_ACTION_PERMUTATION_BINDING
            ),
            "selection_family_bindings_sha256": (
                SELECTION_FAMILY_BINDINGS_SHA256
            ),
            "new_render_label_row_role_filter_resample_or_mapping": False,
        },
        "loader": {
            "implementation": "narrow__row_array_only",
            "allowed_values": list(TRAINING_ROW_ARRAY_VALUES),
            "allowed_supervision_arrays": list(ALLOWED_SUPERVISION_ARRAYS),
            "other_supervision_array_or_general_loader": False,
        },
        "objective": objective_contract(),
        "optimizer": optimizer_contract(),
        "schedule": build_schedule_identity(),
        "gates": {
            "observations": list(OBSERVATION_UPDATES),
            "metric_definitions": {
                name: dict(definition)
                for name, definition in OBSERVATION_METRIC_CONTRACT.items()
            },
            "thresholds": {
                str(update): dict(thresholds)
                for update, thresholds in GATE_THRESHOLDS.items()
            },
            "all_conditions_conjunctive": True,
            "stop_at_first_failure": True,
            "later_stage_requires_all_prior_stages": True,
            "metric_populations": {
                "training_C": 4_262,
                "training_C_fallbacks": 25,
                "aggregate_C_and_wrong_rgb": 495,
                "aggregate_C_and_wrong_rgb_fallbacks": 1,
                "same_action_target_nll_and_strict_win": 494,
                "same_action_target_fallbacks": 0,
                "aggregate_raster_unique_endpoints": (
                    AGGREGATE_RASTER_ENDPOINT_COUNT
                ),
                "aggregate_raster_ordered_endpoint_identity_sha256": (
                    AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256
                ),
                "rough_raster_unique_endpoints": ROUGH_RASTER_ENDPOINT_COUNT,
                "rough_raster_family": ROUGH_RASTER_FAMILY,
            },
        },
        "lifecycle": {
            "attempt_index": ATTEMPT_INDEX,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_must_be_absent_before_reservation": True,
            "phase_b_authorized": False,
            "retry_resume_repair_recovery_replacement_second_seed_or_v2": False,
            "normal_receipts": list(NORMAL_RECEIPT_PATHS),
            "operational_failure_receipts": list(
                OPERATIONAL_FAILURE_RECEIPT_PATHS
            ),
            "checkpoint_and_training_trace_write_only": True,
        },
        "access_policy": {
            "authorized_nonnegative_counters": list(ALLOWED_ACCESS_COUNTER_FIELDS),
            "forbidden_zero_counters": list(
                FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS
            ),
            "prohibited_runtime_categories": list(
                PROHIBITED_RUNTIME_CATEGORIES
            ),
            "accounting_equations": {
                "rgb": (
                    "sum_current_next_fixed_endpoint_rgb_row_requests="
                    "rgb_cache_hits+rgb_cache_misses;"
                    "rgb_cache_misses=rgb_physical_file_opens"
                ),
                "raster_labels.u1": (
                    "row_requests=row_cache_hits+row_cache_misses;"
                    "row_cache_misses=underlying_array_cache_hits+"
                    "underlying_array_cache_misses;underlying_array_cache_"
                    "misses=physical_array_opens"
                ),
            },
            "runner_counter_mapping": dict(RUNNER_ACCESS_COUNTER_MAPPING),
        },
        "authority": dict(DOWNSTREAM_DENIALS),
    }


# Canonical source-document and review helpers.
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


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical-content-bound artifact without mutating ``core``."""

    value = dict(core)
    if "content_sha256" in value:
        raise ValueError("core already contains content_sha256")
    value["content_sha256"] = canonical_json_sha256(value)
    return value


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    declared = value.get("content_sha256")
    core = dict(value)
    core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content hash changed")
    return dict(value)


def _parse_pretty_content_bound_json(
    raw: bytes,
    *,
    name: str,
    expected_content_sha256: str,
) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise PermissionError(f"{name} top level changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        declared != expected_content_sha256
        or canonical_json_sha256(core) != expected_content_sha256
    ):
        raise PermissionError(f"{name} content hash changed")
    return dict(value)


def artifact_binding(
    path: str,
    raw: bytes,
    *,
    content_sha256: str | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    safe_relative_path(path, name="artifact path")
    if content_sha256 is not None and not is_sha256(content_sha256):
        raise ValueError("artifact content hash is malformed")
    value: dict[str, Any] = {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if content_sha256 is not None:
        value["content_sha256"] = content_sha256
    if commit is not None:
        value["commit"] = commit
    return value


def safe_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a safe relative path")
    return value


def validate_binding(
    value: object,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "path", "file_sha256", "content_sha256", "byte_count"
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
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def architecture_decision_binding() -> dict[str, Any]:
    return {
        "path": ARCHITECTURE_DECISION_RELATIVE_PATH,
        "commit": ARCHITECTURE_DECISION_COMMIT,
        "file_sha256": ARCHITECTURE_DECISION_FILE_SHA256,
        "byte_count": ARCHITECTURE_DECISION_BYTE_COUNT,
    }


def preregistration_independent_review_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH,
        "commit": PREREGISTRATION_INDEPENDENT_REVIEW_COMMIT,
        "file_sha256": PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256,
        "content_sha256": PREREGISTRATION_INDEPENDENT_REVIEW_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_INDEPENDENT_REVIEW_BYTE_COUNT,
    }


def prior_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": PRIOR_TERMINAL_AUDIT_COMMIT,
        "file_sha256": PRIOR_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": PRIOR_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": PRIOR_TERMINAL_AUDIT_BYTE_COUNT,
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


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    architecture = _read_regular_source(root / ARCHITECTURE_DECISION_RELATIVE_PATH)
    preregistration = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    preregistration_review = _read_regular_source(
        root / PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH
    )
    prior = _read_regular_source(root / PRIOR_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(architecture) != ARCHITECTURE_DECISION_BYTE_COUNT
        or hashlib.sha256(architecture).hexdigest()
        != ARCHITECTURE_DECISION_FILE_SHA256
    ):
        raise PermissionError("architecture decision changed")
    if (
        len(preregistration) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(preregistration).hexdigest()
        != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("preregistration raw identity changed")
    prereg_value = _parse_pretty_content_bound_json(
        preregistration,
        name="preregistration",
        expected_content_sha256=PREREGISTRATION_CONTENT_SHA256,
    )
    if prereg_value.get("architecture_decision") != {
        "bytes": ARCHITECTURE_DECISION_BYTE_COUNT,
        "path": ARCHITECTURE_DECISION_RELATIVE_PATH,
        "sha256": ARCHITECTURE_DECISION_FILE_SHA256,
    }:
        raise PermissionError("preregistration architecture binding changed")
    if (
        len(preregistration_review)
        != PREREGISTRATION_INDEPENDENT_REVIEW_BYTE_COUNT
        or hashlib.sha256(preregistration_review).hexdigest()
        != PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256
    ):
        raise PermissionError("preregistration independent review changed")
    prereg_review_value = _parse_pretty_content_bound_json(
        preregistration_review,
        name="preregistration independent review",
        expected_content_sha256=(
            PREREGISTRATION_INDEPENDENT_REVIEW_CONTENT_SHA256
        ),
    )
    if (
        prereg_review_value.get("status")
        != PREREGISTRATION_INDEPENDENT_REVIEW_STATUS
        or prereg_review_value.get("findings") != []
        or prereg_review_value.get("authority_review", {}).get(
            "execution_authorized_by_this_review"
        )
        is not False
    ):
        raise PermissionError("preregistration independent review did not pass")
    if (
        len(prior) != PRIOR_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(prior).hexdigest() != PRIOR_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("prior terminal audit raw identity changed")
    prior_value = _parse_pretty_content_bound_json(
        prior,
        name="prior terminal audit",
        expected_content_sha256=PRIOR_TERMINAL_AUDIT_CONTENT_SHA256,
    )
    if (
        prior_value.get("status") != PRIOR_TERMINAL_AUDIT_STATUS
        or prior_value.get("classification")
        != PRIOR_TERMINAL_AUDIT_CLASSIFICATION
    ):
        raise PermissionError("prior terminal audit conclusion changed")
    return {
        ARCHITECTURE_DECISION_RELATIVE_PATH: ARCHITECTURE_DECISION_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH: (
            PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256
        ),
        PRIOR_TERMINAL_AUDIT_RELATIVE_PATH: PRIOR_TERMINAL_AUDIT_FILE_SHA256,
    }


def safe_relative_source_path(value: object) -> str:
    if type(value) is not str or not value:
        raise PermissionError("source path must be one nonempty string")
    path = PurePosixPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or ".." in parts
        or not value.endswith(".py")
        or value.endswith("sealed_test.json")
        or any(
            part in {".generated", "heldout", "sealed"}
            or part.startswith(("heldout_", "sealed_"))
            for part in parts
        )
    ):
        raise PermissionError(f"forbidden source path: {value}")
    return value


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
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources")
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or type(paths) is not list
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or not set(SOURCE_PATHS).issubset(paths)
        or type(bindings) is not list
        or len(bindings) != len(paths)
        or value.get("source_count") != len(paths)
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("source binding fields changed")
        relative = safe_relative_source_path(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("source binding identity changed")
        normalized.append(relative)
    if normalized != paths:
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
        if digest != binding["file_sha256"] or len(raw) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound source changed: {relative}")
        result[relative] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "reviewed_sources",
        "source_manifest",
        "preregistration",
        "preregistration_independent_review",
        "architecture_decision",
        "prior_terminal_audit",
        "science_contract",
        "source_only_checks",
        "scientific_checks",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("source review fields changed")
    if source_manifest_binding is None:
        manifest_raw = _read_regular_source(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        manifest = validate_source_manifest(manifest_raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            manifest_raw,
            content_sha256=str(manifest["content_sha256"]),
        )
    try:
        expected_manifest_binding = validate_binding(
            dict(source_manifest_binding),
            path=SOURCE_MANIFEST_RELATIVE_PATH,
        )
    except (TypeError, ValueError) as error:
        raise PermissionError("source manifest review binding changed") from error
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_AND_SCIENCE"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"] != expected_manifest_binding
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH)
        != expected_manifest_binding["file_sha256"]
        or expected_sources.get(PREREGISTRATION_RELATIVE_PATH)
        != PREREGISTRATION_FILE_SHA256
        or expected_sources.get(PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH)
        != PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256
        or expected_sources.get(ARCHITECTURE_DECISION_RELATIVE_PATH)
        != ARCHITECTURE_DECISION_FILE_SHA256
        or expected_sources.get(PRIOR_TERMINAL_AUDIT_RELATIVE_PATH)
        != PRIOR_TERMINAL_AUDIT_FILE_SHA256
        or value["preregistration"] != preregistration_binding()
        or value["preregistration_independent_review"]
        != preregistration_independent_review_binding()
        or value["architecture_decision"] != architecture_decision_binding()
        or value["prior_terminal_audit"] != prior_terminal_audit_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("source review did not pass exact frozen science")
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
        "architecture_decision",
        "prior_terminal_audit",
        "runtime_inputs",
        "experiment",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    try:
        expected_review_binding = validate_binding(
            dict(review_binding), path=REVIEW_RELATIVE_PATH
        )
    except (TypeError, ValueError) as error:
        raise PermissionError("source review authorization binding changed") from error
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review_binding
        or value["preregistration"] != preregistration_binding()
        or value["architecture_decision"] != architecture_decision_binding()
        or value["prior_terminal_audit"] != prior_terminal_audit_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
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
        "architecture_decision_binding",
        "artifact_binding",
        "build_schedule_identity",
        "canonical_json_bytes",
        "canonical_json_sha256",
        "current_source_bindings",
        "evaluate_gate",
        "is_sha256",
        "model_config",
        "objective_contract",
        "optimizer_contract",
        "parse_canonical_json",
        "preregistration_binding",
        "preregistration_independent_review_binding",
        "prior_terminal_audit_binding",
        "runtime_authorization_template",
        "safe_relative_path",
        "safe_relative_source_path",
        "science_contract",
        "validate_access_counters",
        "validate_authorization",
        "validate_binding",
        "validate_failure_status_chain",
        "validate_governing_documents",
        "validate_review",
        "validate_source_manifest",
    }
]
