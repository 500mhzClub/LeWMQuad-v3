"""Source-only contract for geometry-anchored deformable BEV joint-JEPA V1.

Importing this module opens no generated input, checkpoint, runtime output,
trace, accelerator, held-out, or sealed material.  It binds one fresh RGB-only
perception warmup followed by genuine joint online-representation/predictor
JEPA training.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]


def _source_only_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_direct = _source_only_module(
    "_lewm_geometry_anchored_joint_jepa_v1_frozen_direct_contract",
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py",
)


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/semantic_v3_terminal_audit_fast",
    "/root/semantic_v3_terminal_audit",
    "/root/projective_jepa_prereg_draft",
)
SCHEMA_PREFIX = "lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1"
EXPERIMENT_ID = "geometry_anchored_deformable_bev_lift_joint_jepa_v1"

# Frozen governing documents.
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "561e946443dd9eec668255708982565739de033b"
PREREGISTRATION_FILE_SHA256 = (
    "4d59e71702a716fb3a669ddffd87fb124da76ee2957c3e4dd16a8ad6dadbf402"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "8729a45c23cbb88f2409256d03b3326d4f1348ee78e4e8852aef69e67fda7356"
)
PREREGISTRATION_BYTE_COUNT = 19_986

V3_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_terminal_audit_2026-07-27.json"
)
V3_TERMINAL_AUDIT_COMMIT = "cfc5253304f94d56c422ad3c09880c78be0513bc"
V3_TERMINAL_AUDIT_FILE_SHA256 = (
    "9acabffb5f297f6a6b23e06c7f7cbdde34b20d5fc99c21f1a80c7a6f98417c3f"
)
V3_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "0a28ab1d9551b68307814ed282e6269a564a5b9c8360c37e716e9a1258f3a140"
)
V3_TERMINAL_AUDIT_BYTE_COUNT = 9_817
V3_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_UPDATE_400_SCIENTIFIC_FAILURE_CLOSES_V3_AND_SIGNED_BOUNDARY_"
    "SEMANTIC_ANCHOR_FAMILY_NO_RETRY"
)
V3_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ONE_SHOT_TERMINAL_SCIENTIFIC_FAILURE_AT_UPDATE_400_AFTER_PASSED_"
    "UPDATE_ZERO_AND_DEFERRED_UPDATE_100_TREND_GATES_SEMANTIC_ANCHOR_DID_NOT_"
    "PREVENT_OCCUPIED_RECALL_COLLAPSE_V3_AND_SEMANTIC_ANCHOR_FAMILY_"
    "PERMANENTLY_CLOSED_NO_RETRY_WEIGHT_TUNE_OR_TIMING_SUCCESSOR"
)

# Source graph names.  These declarations grant no runtime authority.
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_closure.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "contract.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "runner.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_launch_go2_geometry_anchored_deformable_bev_lift_joint_"
    "jepa_v1.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "execution_authorization_2026-07-27.json"
)
ADDITIVE_SOURCE_PATHS = tuple(sorted((
    "lewm/__init__.py",
    "lewm/benchmarks/__init__.py",
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py",
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    LAUNCHER_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_direct.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

# Exact runtime interpreter handoff.
RUNTIME_INTERPRETER_PATH = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64/bin/python"
)
RUNTIME_SYS_PREFIX = (
    "/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64"
)
RUNTIME_INTERPRETER_ARGUMENTS = ("-I", "-B")

# Exact development inputs.  Merely declaring these identities opens nothing.
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
        "file_sha256": "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360",
        "content_sha256": "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a",
        "byte_count": 311_598,
    },
    RAW_AUDIT_RELATIVE_PATH: {
        "path": RAW_AUDIT_RELATIVE_PATH,
        "file_sha256": "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76",
        "content_sha256": "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca",
        "byte_count": 26_975,
    },
    N320_GATE_RELATIVE_PATH: {
        "path": N320_GATE_RELATIVE_PATH,
        "file_sha256": "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6",
        "content_sha256": "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b",
        "byte_count": 7_960,
    },
    N320_CHECKPOINT_RELATIVE_PATH: {
        "path": N320_CHECKPOINT_RELATIVE_PATH,
        "file_sha256": "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0",
        "content_sha256": "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b",
        "byte_count": 13_777_100,
    },
    SCHEDULE_RELATIVE_PATH: {
        "path": SCHEDULE_RELATIVE_PATH,
        "file_sha256": "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
        "content_sha256": "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
        "byte_count": 607_373,
    },
}

# Frozen role, action, family, and derangement identities.
TRAIN_ROLE_COUNTS = {"pairs": 4_262, "unique_endpoints": 7_777, "scenes": 72}
SELECTION_ROLE_COUNTS = {"pairs": 495, "unique_endpoints": 924, "scenes": 8}
ACTION_VOCABULARY = tuple(_direct.ACTION_VOCABULARY)
HOLD_ACTION_INDEX = 6
SCENE_FAMILIES = tuple(_direct.SCENE_FAMILIES)
SELECTION_FAMILY_BINDINGS = copy.deepcopy(_direct.SELECTION_FAMILY_BINDINGS)
SELECTION_FAMILY_BINDINGS_SHA256 = (
    "c39efe48afd6d4c02a24af77f1f11e7f6cd5a69d571b0a9416924b07bbacbb11"
)
TARGET_MAPPING_BINDINGS = copy.deepcopy(_direct.TARGET_MAPPING_BINDINGS)
SELECTION_ACTION_PERMUTATION_BINDING = copy.deepcopy(
    _direct.SELECTION_ACTION_PERMUTATION_BINDING
)
FIXED_MAPPED_NEGATIVE_RULE = copy.deepcopy(_direct.FIXED_MAPPED_NEGATIVE_RULE)
AGGREGATE_RASTER_ENDPOINT_COUNT = 924
AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256 = (
    "dd84fc73e14056c9d6c8f7c066c2dcafe9726827193c42982d51f412ea744fa4"
)
ROUGH_RASTER_FAMILY = "rough_local_dynamics"
ROUGH_RASTER_ENDPOINT_COUNT = 123
RASTER_CLASS_ORDER = ("UNKNOWN", "FREE", "OCCUPIED")
UNKNOWN_CLASS = 0
TRAINING_ROW_ARRAY_VALUES = (
    "current_rgb", "next_rgb", "fixed_negative_rgb", "raster_labels.u1"
)
ALLOWED_SUPERVISION_ARRAYS = ("raster_labels.u1",)

# Schedule, lifecycle, and phase boundary.
INITIALIZATION_SEED = 20260712
BASE_INITIALIZATION_SEED = INITIALIZATION_SEED
N320_FIT_SEED = 20260710
SCHEDULE_SEED = 20260713
TARGET_EMA_MOMENTUM = 0.996
MAXIMUM_ATTEMPTS = 1
ATTEMPT_INDEX = 1
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
GPU_ACTIVE_TIME_CAP_MINUTES = 30
GPU_ACTIVE_MINUTES_MAX = GPU_ACTIVE_TIME_CAP_MINUTES
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
EFFECTIVE_BATCH_SIZE = 16
CHECKPOINT_UPDATES = (100, 400, 1_000)
OBSERVATION_UPDATES = (0, 100, 400, 1_000)
JOINT_PHASE_FIRST_UPDATE = 401
JOINT_PHASE_UPDATE_COUNT = 600
WARMUP_UPDATES = 400
SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1/"
    "attempt_v1"
)

# Model and science constants.
BEV_SHAPE = (64, 64)
BEV_LATENT_WIDTH = 64
BEV_FORWARD_RANGE_METRES = (-0.95, 5.35)
BEV_LEFT_RANGE_METRES = (-3.15, 3.15)
DEFORMABLE_SAMPLES_PER_CELL = 4
DEFORMABLE_OFFSET_RADIUS_TOKEN_CELLS = 2.0
TARGET_NORMALIZATION = "per_cell_channel_LayerNorm"
LATENT_SMOOTH_L1_BETA = 1.0
LOG2 = math.log(2.0)
LOG3 = math.log(3.0)
LOG9 = math.log(9.0)
SHARED_GRADIENT_RATIO_MINIMUM = 1.0 / 32.0
SHARED_GRADIENT_RATIO_MAXIMUM = 32.0

MODEL_CONFIG = {
    "encoder": {
        "input": "single_112x112_RGB",
        "patch_size": 7,
        "width": 192,
        "depth": 6,
        "heads": 6,
        "initialization": "exact_N320_encoder_state_only",
        "trainable": True,
    },
    "bev_lattice": {
        "shape": list(BEV_SHAPE),
        "latent_width": BEV_LATENT_WIDTH,
        "forward_range_m": list(BEV_FORWARD_RANGE_METRES),
        "left_range_m": list(BEV_LEFT_RANGE_METRES),
    },
    "geometry_anchored_deformable_lift": {
        "samples_per_cell": DEFORMABLE_SAMPLES_PER_CELL,
        "fixed_level_camera_projective_anchor": True,
        "offset_formula": "2*tanh(raw_offset)_encoder_token_cells_xy",
        "offset_radius_encoder_token_cells_maximum": (
            DEFORMABLE_OFFSET_RADIUS_TOKEN_CELLS
        ),
        "sample_operator": "bilinear_spatial_RGB_token_map_only",
        "weight_operator": "softmax_over_four_samples",
        "invalid_anchor_policy": "learned_null_no_sampling_semantic_UNKNOWN",
        "local_refinement": "two_width64_residual_Conv2d_k3_s1_p1_blocks",
        "global_attention_pooling_mixing_or_auxiliary_bypass": False,
    },
    "semantic_head": {
        "operator": "Conv2d(64,3,kernel_size=1,bias=True)",
        "class_order": list(RASTER_CLASS_ORDER),
        "labels_are_inference_inputs": False,
    },
    "predictor": {
        "action_count": 9,
        "action_vocabulary": list(ACTION_VOCABULARY),
        "action_embedding_trainable": True,
        "operator": "width64_local_two_block_3x3_residual_predictor",
        "predict_all_actions": True,
        "coordinate_goal_map_pose_label_future_or_global_bypass": False,
    },
    "target": {
        "inventory": ["encoder", "deformable_lift", "local_refinement"],
        "requires_grad": False,
        "hard_sync_before_update_zero": 1,
        "ema_momentum": TARGET_EMA_MOMENTUM,
        "ema_after_every_online_update": True,
    },
    "inference_inputs": ["single_current_RGB", "commanded_action_for_prediction_only"],
    "forbidden_inference_inputs": [
        "depth", "pose", "odometry", "attitude", "runtime_camera_motion",
        "map", "goal", "raster_or_semantic_labels", "privileged_geometry",
        "future_RGB",
    ],
}

OBJECTIVE_CONTRACT = {
    "semantic_A": (
        "0.5*equal_row_equal_present_final_class_macro_NLL_current+"
        "0.5*equal_row_equal_present_final_class_macro_NLL_next"
    ),
    "semantic_S": "A/log(3)",
    "latent_energy": (
        "mean_smooth_L1_beta1(per_cell_channel_LayerNorm(prediction),"
        "stop_gradient_per_cell_channel_LayerNorm(target))"
    ),
    "warmup_updates_1_400": "L_warmup=S",
    "persistence_baseline_B400": (
        "checkpoint_selection_mean_target_current_to_target_next_identity_energy"
    ),
    "joint_updates_401_1000": {
        "P_latent_prediction": "executed_action_energy/B400",
        "R_action": "cross_entropy(-E_action/detached_mean_action_energy,action)/log(9)",
        "C_same_action_contrast": (
            "two_way_cross_entropy(correct_next_energy,deranged_next_energy)/log(2)"
        ),
        "total": "S+P_latent_prediction+R_action+C_same_action_contrast",
        "component_weights": {"S": 1.0, "P": 1.0, "R": 1.0, "C": 1.0},
    },
}

OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "precision": "float32",
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 1e-4,
    "learning_rates": {
        "encoder": 1e-4,
        "deformable_lift_refinement_semantic_head": 3e-4,
        "predictor": 3e-4,
    },
    "ordered_groups": ["encoder", "lift_semantic", "predictor"],
    "constructed_once_at_update_zero": True,
    "rebuilt_at_update_401": False,
    "separate_group_clips": {
        "representation_and_semantic_l2_max": 1.0,
        "predictor_l2_max": 1.0,
        "global_combined_clip": False,
    },
}

CALL_GRAPH_CONTRACT = {
    "online_current_RGB": "trainable_encoder_lift_refinement_to_semantics_and_prediction",
    "online_next_RGB": "same_trainable_representation_to_semantic_S_only",
    "target_next_RGB": "stop_gradient_EMA_representation_to_JEPA_target_only",
    "target_deranged_next_RGB": "stop_gradient_EMA_representation_to_same_action_contrast_only",
    "predictor_input": "online_current_BEV_latent_plus_commanded_action_only",
    "global_attention_or_privileged_bypass": False,
}

ALLOWED_ACCESS_COUNTER_FIELDS = tuple(_direct.ALLOWED_ACCESS_COUNTER_FIELDS)
FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS = tuple(
    _direct.FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS
)
ACCESS_COUNTER_FIELDS = (
    *ALLOWED_ACCESS_COUNTER_FIELDS, *FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS
)
PROHIBITED_RUNTIME_CATEGORIES = tuple(_direct.PROHIBITED_RUNTIME_CATEGORIES)

DOWNSTREAM_DENIALS = {
    "checkpoint_qualified": False,
    "checkpoint_read_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_resume_repair_recovery_extension_second_seed_or_second_attempt_authorized": False,
}
SOURCE_ONLY_AUTHORITY = {
    "implementation_authorized": True,
    "cpu_synthetic_tests_authorized": True,
    "source_closure_authorized": True,
    "independent_source_review_authorized": True,
    "execution_authorized": False,
    "generated_input_dataset_rgb_raster_checkpoint_tensor_gpu_or_training_access_authorized": False,
    "navigation_g2_heldout_sealed_production_promotion_or_deployment_authorized": False,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    "one_exact_fresh_attempt_authorized": True,
    "attempt_index": ATTEMPT_INDEX,
    "maximum_attempts": MAXIMUM_ATTEMPTS,
    "maximum_updates": MAXIMUM_UPDATES,
    "maximum_presentations": MAXIMUM_PRESENTATIONS,
    "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
    "runtime_interpreter_path": RUNTIME_INTERPRETER_PATH,
    "runtime_sys_prefix": RUNTIME_SYS_PREFIX,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "n320_encoder_initialization_only_authorized": True,
    "allowed_row_array_values": list(TRAINING_ROW_ARRAY_VALUES),
    "allowed_supervision_arrays": list(ALLOWED_SUPERVISION_ARRAYS),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent_before_reservation": True,
    **DOWNSTREAM_DENIALS,
}
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1"
)


# Re-export the frozen duplicate-safe canonical JSON primitives.
canonical_json_bytes = _direct.canonical_json_bytes
canonical_json_sha256 = _direct.canonical_json_sha256
with_content_sha256 = _direct.with_content_sha256
is_sha256 = _direct.is_sha256
parse_canonical_json = _direct.parse_canonical_json


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def v3_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V3_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V3_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V3_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V3_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V3_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V3_TERMINAL_AUDIT_STATUS,
        "classification": V3_TERMINAL_AUDIT_CLASSIFICATION,
    }


def build_runtime_inputs() -> dict[str, Any]:
    return {
        "runtime_interpreter": {
            "path": RUNTIME_INTERPRETER_PATH,
            "sys_prefix": RUNTIME_SYS_PREFIX,
            "arguments": list(RUNTIME_INTERPRETER_ARGUMENTS),
        },
        "raw_manifest": copy.deepcopy(RUNTIME_BINDINGS[RAW_MANIFEST_RELATIVE_PATH]),
        "raw_audit": copy.deepcopy(RUNTIME_BINDINGS[RAW_AUDIT_RELATIVE_PATH]),
        "n320_gate": copy.deepcopy(RUNTIME_BINDINGS[N320_GATE_RELATIVE_PATH]),
        "n320_encoder_checkpoint": copy.deepcopy(
            RUNTIME_BINDINGS[N320_CHECKPOINT_RELATIVE_PATH]
        ),
        "schedule": copy.deepcopy(RUNTIME_BINDINGS[SCHEDULE_RELATIVE_PATH]),
    }


def build_schedule_identity() -> dict[str, Any]:
    return {
        "source": copy.deepcopy(RUNTIME_BINDINGS[SCHEDULE_RELATIVE_PATH]),
        "seed": SCHEDULE_SEED,
        "updates": MAXIMUM_UPDATES,
        "presentations": MAXIMUM_PRESENTATIONS,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoints": list(CHECKPOINT_UPDATES),
        "observation_updates": list(OBSERVATION_UPDATES),
        "prefix_sha256": {
            str(update): digest for update, digest in SCHEDULE_PREFIX_SHA256.items()
        },
        "one_scheduled_pair_is_one_presentation": True,
        "preserve_rows_roles_order_labels_actions_endpoints_and_mappings": True,
    }


def model_config() -> dict[str, Any]:
    return copy.deepcopy(MODEL_CONFIG)


def objective_contract() -> dict[str, Any]:
    return copy.deepcopy(OBJECTIVE_CONTRACT)


def optimizer_contract() -> dict[str, Any]:
    return copy.deepcopy(OPTIMIZER_CONTRACT)


def runtime_authorization_template() -> dict[str, Any]:
    """Return the exact frozen Direct-V1 loader/input authorization schema.

    The new experiment scope, interpreter, output root, and 30-minute cap live
    in ``science_contract`` and ``EXECUTION_AUTHORITY``; keeping this value
    byte-structurally compatible lets the reviewed Direct-V1 loader consume the
    same raw/N320/schedule grants without an adapter.
    """

    return copy.deepcopy(_direct.runtime_authorization_template())


COMMON_GATE_BOOLEAN_FIELDS = (
    "source_authority_exact",
    "runtime_input_bindings_exact",
    "schedule_prefix_exact",
    "role_and_mapping_bindings_exact",
    "model_parameter_inventory_exact",
    "optimizer_inventory_exact",
    "rgb_only_causal_call_graph_exact",
    "forbidden_input_and_bypass_counts_zero",
    "fresh_model_target_predictor_optimizer_registry_observations_and_rng",
    "target_requires_grad_false",
    "out_of_frustum_sampling_blocked",
    "out_of_frustum_semantic_unknown",
    "state_nonconstant",
    "rgb_response_nonconstant",
    "all_registered_values_finite",
    "all_forbidden_access_counts_zero",
)

GATE_THRESHOLDS = {
    100: {
        "balanced_accuracy_minimum": 0.60,
        "free_recall_minimum": 0.55,
        "occupied_recall_minimum": 0.30,
        "free_occupied_gap_maximum": 0.50,
        "correct_rgb_scene_wins_minimum": 6,
    },
    400: {
        "raster_nll_maximum": 0.42,
        "balanced_accuracy_minimum_absolute": 0.74,
        "balanced_accuracy_minimum_relative_to_update_100": -0.01,
        "unknown_recall_minimum": 0.75,
        "free_recall_minimum": 0.70,
        "occupied_recall_minimum_absolute": 0.60,
        "occupied_recall_minimum_relative_to_update_100": -0.03,
        "free_occupied_gap_maximum": 0.30,
        "rough_balanced_accuracy_minimum_absolute": 0.72,
        "rough_balanced_accuracy_minimum_relative_to_update_100": -0.01,
        "rough_occupied_recall_minimum_absolute": 0.60,
        "rough_occupied_recall_minimum_relative_to_update_100": -0.03,
        "correct_rgb_scene_wins": 8,
    },
    1_000: {
        "raster_nll_maximum_absolute": 0.38,
        "raster_nll_maximum_relative_to_update_400": 0.01,
        "balanced_accuracy_minimum_absolute": 0.80,
        "balanced_accuracy_minimum_relative_to_update_400": -0.01,
        "unknown_recall_minimum": 0.80,
        "free_recall_minimum": 0.75,
        "occupied_recall_minimum_absolute": 0.70,
        "occupied_recall_minimum_relative_to_update_400": -0.03,
        "free_occupied_gap_maximum": 0.25,
        "rough_balanced_accuracy_minimum_absolute": 0.772,
        "rough_balanced_accuracy_minimum_relative_to_update_400": -0.01,
        "rough_occupied_recall_minimum_absolute": 0.65,
        "rough_occupied_recall_minimum_relative_to_update_400": -0.03,
        "latent_prediction_B400_factor_maximum": 0.90,
        "action_nll_strict_maximum": 0.95 * LOG9,
        "action_macro_balanced_accuracy_strict_minimum": 2.0 / 9.0,
        "family_count_minimum": 6,
        "same_action_nll_strict_maximum": 0.95 * LOG2,
        "same_action_win_rate_minimum": 0.65,
        "target_baseline_retention_factor_minimum": 0.75,
        "shared_gradient_ratio_minimum": SHARED_GRADIENT_RATIO_MINIMUM,
        "shared_gradient_ratio_maximum": SHARED_GRADIENT_RATIO_MAXIMUM,
    },
}

GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_STRUCTURAL_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_STRUCTURAL_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_PERCEPTION_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_PERCEPTION_GATE",
    ),
    400: (
        "FAIL_UPDATE_400_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_ANTI_COLLAPSE_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_ANTI_COLLAPSE_GATE",
    ),
    1_000: (
        "FAIL_UPDATE_1000_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_QUALIFICATION_GATE_TERMINAL_NO_RETRY",
        "PASS_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_MECHANISM_ONLY",
    ),
}
CONTROL_UPDATE_ZERO_FAIL = GATE_CONTROLS[0][0]
CONTROL_UPDATE_100_FAIL = GATE_CONTROLS[100][0]
CONTROL_UPDATE_400_FAIL = GATE_CONTROLS[400][0]
CONTROL_UPDATE_1000_FAIL = GATE_CONTROLS[1_000][0]
CONTROL_PASS = GATE_CONTROLS[1_000][1]
FAILURE_CONTROLS = tuple(value[0] for value in GATE_CONTROLS.values())
CONTROL_FAIL_JOINT_GRADIENT = (
    "FAIL_JOINT_SHARED_GRADIENT_CONTRIBUTION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = (
    "TERMINAL_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_"
    "INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)
PHASE_SWITCH_CONTROLS = (
    "FAIL_UPDATE_401_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_PHASE_SWITCH_TERMINAL_NO_RETRY",
    "CONTINUE_AFTER_UPDATE_401_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_PHASE_SWITCH",
)


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be one finite number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _metric(metrics: Mapping[str, Any], name: str, *aliases: str) -> Any:
    runner_aliases = {
        "online_optimizer_update_count": ("online_optimizer_updates",),
        "predictor_optimizer_update_count": ("predictor_optimizer_updates",),
        "joint_optimizer_update_count": ("joint_optimizer_updates",),
        "shared_gradient_ratio_evaluation_count": (
            "shared_gradient_gate_pass_count",
        ),
        "target_gradient_tensor_count": ("target_gradient_count",),
    }
    keys = (name, *aliases, *runner_aliases.get(name, ()))
    for container in (
        metrics,
        metrics.get("integrity", {}),
        metrics.get("joint_accounting", {}),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container:
                return container[key]
    raise ValueError(f"missing metric: {name}")


def _prior(
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None,
    update: int,
) -> Mapping[str, Any]:
    if prior_metrics is None:
        raise ValueError(f"update {update} prior metrics are required")
    value = prior_metrics.get(update, prior_metrics.get(str(update)))
    if not isinstance(value, Mapping):
        raise ValueError(f"update {update} prior metrics are required")
    return value


def _common_conjuncts(
    metrics: Mapping[str, Any], *, prior_gates_passed: bool
) -> dict[str, bool]:
    conjuncts = {"prior_gates_passed": _boolean(
        prior_gates_passed, "prior_gates_passed"
    )}
    for field in COMMON_GATE_BOOLEAN_FIELDS:
        conjuncts[field] = _boolean(metrics.get(field), field)
    return conjuncts


def _accounting_zero(metrics: Mapping[str, Any], conjuncts: dict[str, bool]) -> None:
    for field in (
        "online_optimizer_update_count",
        "target_ema_update_count",
        "predictor_forward_count",
        "predictor_objective_count",
        "predictor_backward_count",
        "predictor_optimizer_update_count",
        "joint_optimizer_update_count",
        "shared_gradient_ratio_evaluation_count",
    ):
        conjuncts[f"{field}_equals_0"] = _integer(
            _metric(metrics, field), field
        ) == 0


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None = None,
    *,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate one preregistered scientific gate without opening payloads."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    conjuncts = _common_conjuncts(
        metrics, prior_gates_passed=prior_gates_passed
    )
    expected_presentations = {0: 0, 100: 1_600, 400: 6_400, 1_000: 16_000}[update]
    conjuncts[f"presentations_equals_{expected_presentations}"] = (
        _integer(_metric(metrics, "presentations"), "presentations")
        == expected_presentations
    )

    if update == 0:
        _accounting_zero(metrics, conjuncts)
        required = (
            "online_target_representation_bitwise_equal",
            "predictor_parameter_group_present",
            "semantic_objective_formula_exact",
            "latent_prediction_objective_formula_exact",
            "action_objective_formula_exact",
            "same_action_contrast_formula_exact",
            "deformable_lift_synthetic_mechanism_exact",
            "paired_correct_wrong_rgb_latents_finite_nonidentical",
        )
        for field in required:
            conjuncts[field] = _boolean(metrics.get(field), field)
        conjuncts["initial_target_hard_sync_count_equals_1"] = (
            _integer(
                _metric(metrics, "initial_target_hard_sync_count"),
                "initial_target_hard_sync_count",
            ) == 1
        )
    elif update == 100:
        zero = _prior(prior_metrics, 0)
        A = _finite(_metric(metrics, "A"), "A")
        A0 = _finite(_metric(zero, "A"), "update_0.A")
        nll = _finite(
            _metric(metrics, "raster_nll", "aggregate_raster_nll"), "raster_nll"
        )
        nll0 = _finite(
            _metric(zero, "raster_nll", "aggregate_raster_nll"), "update_0.raster_nll"
        )
        free = _finite(
            _metric(metrics, "free_recall", "aggregate_free_recall"), "free_recall"
        )
        occupied = _finite(
            _metric(metrics, "occupied_recall", "aggregate_occupied_recall"),
            "occupied_recall",
        )
        conjuncts.update({
            "A_strictly_lower_than_update_0": A < A0,
            "raster_nll_strictly_lower_than_update_0": nll < nll0,
            "balanced_accuracy_at_least_point60": _finite(
                _metric(metrics, "balanced_accuracy", "aggregate_raster_balanced_accuracy"),
                "balanced_accuracy",
            ) >= 0.60,
            "free_recall_at_least_point55": free >= 0.55,
            "occupied_recall_at_least_point30": occupied >= 0.30,
            "free_occupied_gap_at_most_point50": abs(free - occupied) <= 0.50,
            "rough_balanced_accuracy_strictly_above_update_0": _finite(
                _metric(metrics, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
                "rough_balanced_accuracy",
            ) > _finite(
                _metric(zero, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
                "update_0.rough_balanced_accuracy",
            ),
            "rough_occupied_recall_strictly_above_update_0": _finite(
                _metric(metrics, "rough_occupied_recall", "rough_raster_occupied_recall"),
                "rough_occupied_recall",
            ) > _finite(
                _metric(zero, "rough_occupied_recall", "rough_raster_occupied_recall"),
                "update_0.rough_occupied_recall",
            ),
            "paired_rgb_margin_strictly_above_update_0": _finite(
                _metric(metrics, "paired_rgb_margin", "paired_rgb_aggregate_margin"),
                "paired_rgb_margin",
            ) > _finite(
                _metric(zero, "paired_rgb_margin", "paired_rgb_aggregate_margin"),
                "update_0.paired_rgb_margin",
            ),
            "paired_rgb_scene_wins_at_least_6": _integer(
                _metric(metrics, "paired_rgb_scene_wins", "correct_rgb_scene_win_count"),
                "paired_rgb_scene_wins",
            ) >= 6,
            "online_optimizer_updates_equal_100": _integer(
                _metric(metrics, "online_optimizer_update_count"),
                "online_optimizer_update_count",
            ) == 100,
            "target_ema_updates_equal_100": _integer(
                _metric(metrics, "target_ema_update_count"), "target_ema_update_count"
            ) == 100,
        })
        for field in (
            "predictor_forward_count", "predictor_objective_count",
            "predictor_backward_count", "predictor_optimizer_update_count",
            "joint_optimizer_update_count", "shared_gradient_ratio_evaluation_count",
        ):
            conjuncts[f"{field}_equals_0"] = _integer(
                _metric(metrics, field), field
            ) == 0
    elif update == 400:
        hundred = _prior(prior_metrics, 100)
        A = _finite(_metric(metrics, "A"), "A")
        A100 = _finite(_metric(hundred, "A"), "update_100.A")
        nll = _finite(
            _metric(metrics, "raster_nll", "aggregate_raster_nll"), "raster_nll"
        )
        ba100 = _finite(
            _metric(hundred, "balanced_accuracy", "aggregate_raster_balanced_accuracy"),
            "update_100.balanced_accuracy",
        )
        occupied100 = _finite(
            _metric(hundred, "occupied_recall", "aggregate_occupied_recall"),
            "update_100.occupied_recall",
        )
        rough_ba100 = _finite(
            _metric(hundred, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
            "update_100.rough_balanced_accuracy",
        )
        rough_occ100 = _finite(
            _metric(hundred, "rough_occupied_recall", "rough_raster_occupied_recall"),
            "update_100.rough_occupied_recall",
        )
        free = _finite(
            _metric(metrics, "free_recall", "aggregate_free_recall"), "free_recall"
        )
        occupied = _finite(
            _metric(metrics, "occupied_recall", "aggregate_occupied_recall"),
            "occupied_recall",
        )
        B400 = _finite(_metric(metrics, "B400"), "B400")
        conjuncts.update({
            "A_strictly_lower_than_update_100": A < A100,
            "raster_nll_at_most_point42": nll <= 0.42,
            "balanced_accuracy_at_least_max_point74_or_update_100_minus_point01": _finite(
                _metric(metrics, "balanced_accuracy", "aggregate_raster_balanced_accuracy"),
                "balanced_accuracy",
            ) >= max(0.74, ba100 - 0.01),
            "unknown_recall_at_least_point75": _finite(
                _metric(metrics, "unknown_recall", "aggregate_unknown_recall"),
                "unknown_recall",
            ) >= 0.75,
            "free_recall_at_least_point70": free >= 0.70,
            "occupied_recall_at_least_max_point60_or_update_100_minus_point03": (
                occupied >= max(0.60, occupied100 - 0.03)
            ),
            "free_occupied_gap_at_most_point30": abs(free - occupied) <= 0.30,
            "rough_balanced_accuracy_at_least_max_point72_or_update_100_minus_point01": _finite(
                _metric(metrics, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
                "rough_balanced_accuracy",
            ) >= max(0.72, rough_ba100 - 0.01),
            "rough_occupied_recall_at_least_max_point60_or_update_100_minus_point03": _finite(
                _metric(metrics, "rough_occupied_recall", "rough_raster_occupied_recall"),
                "rough_occupied_recall",
            ) >= max(0.60, rough_occ100 - 0.03),
            "paired_rgb_margin_strictly_above_update_100": _finite(
                _metric(metrics, "paired_rgb_margin", "paired_rgb_aggregate_margin"),
                "paired_rgb_margin",
            ) > _finite(
                _metric(hundred, "paired_rgb_margin", "paired_rgb_aggregate_margin"),
                "update_100.paired_rgb_margin",
            ),
            "paired_rgb_scene_wins_equal_8": _integer(
                _metric(metrics, "paired_rgb_scene_wins", "correct_rgb_scene_win_count"),
                "paired_rgb_scene_wins",
            ) == 8,
            "B400_finite_strictly_positive": B400 > 0.0,
            "B400_content_hash_frozen": is_sha256(
                _metric(metrics, "B400_content_sha256")
            ),
            "B400_frozen_before_joint_phase": _boolean(
                metrics.get("B400_frozen_before_joint_phase"),
                "B400_frozen_before_joint_phase",
            ),
            "target_effective_rank_baseline_finite_strictly_positive": _finite(
                _metric(metrics, "target_effective_rank"), "target_effective_rank"
            ) > 0.0,
            "target_channel_variance_baseline_finite_strictly_positive": _finite(
                _metric(metrics, "target_channel_variance"), "target_channel_variance"
            ) > 0.0,
            "target_spatial_diversity_baseline_finite_strictly_positive": _finite(
                _metric(metrics, "target_spatial_diversity"), "target_spatial_diversity"
            ) > 0.0,
            "target_collapse_baselines_frozen_before_joint_phase": _boolean(
                metrics.get("target_collapse_baselines_frozen_before_joint_phase"),
                "target_collapse_baselines_frozen_before_joint_phase",
            ),
            "online_optimizer_updates_equal_400": _integer(
                _metric(metrics, "online_optimizer_update_count"),
                "online_optimizer_update_count",
            ) == 400,
            "target_ema_updates_equal_400": _integer(
                _metric(metrics, "target_ema_update_count"), "target_ema_update_count"
            ) == 400,
        })
        for field in (
            "predictor_forward_count", "predictor_objective_count",
            "predictor_backward_count", "predictor_optimizer_update_count",
            "joint_optimizer_update_count", "shared_gradient_ratio_evaluation_count",
        ):
            conjuncts[f"{field}_equals_0"] = _integer(
                _metric(metrics, field), field
            ) == 0
    else:
        four_hundred = _prior(prior_metrics, 400)
        A400 = _finite(_metric(four_hundred, "A"), "update_400.A")
        nll400 = _finite(
            _metric(four_hundred, "raster_nll", "aggregate_raster_nll"),
            "update_400.raster_nll",
        )
        ba400 = _finite(
            _metric(four_hundred, "balanced_accuracy", "aggregate_raster_balanced_accuracy"),
            "update_400.balanced_accuracy",
        )
        occupied400 = _finite(
            _metric(four_hundred, "occupied_recall", "aggregate_occupied_recall"),
            "update_400.occupied_recall",
        )
        rough_ba400 = _finite(
            _metric(four_hundred, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
            "update_400.rough_balanced_accuracy",
        )
        rough_occ400 = _finite(
            _metric(four_hundred, "rough_occupied_recall", "rough_raster_occupied_recall"),
            "update_400.rough_occupied_recall",
        )
        B400 = _finite(_metric(four_hundred, "B400"), "update_400.B400")
        free = _finite(
            _metric(metrics, "free_recall", "aggregate_free_recall"), "free_recall"
        )
        occupied = _finite(
            _metric(metrics, "occupied_recall", "aggregate_occupied_recall"),
            "occupied_recall",
        )
        conjuncts.update({
            "A_at_most_update_400": _finite(_metric(metrics, "A"), "A") <= A400,
            "raster_nll_at_most_min_point38_or_update_400_plus_point01": _finite(
                _metric(metrics, "raster_nll", "aggregate_raster_nll"), "raster_nll"
            ) <= min(0.38, nll400 + 0.01),
            "balanced_accuracy_at_least_max_point80_or_update_400_minus_point01": _finite(
                _metric(metrics, "balanced_accuracy", "aggregate_raster_balanced_accuracy"),
                "balanced_accuracy",
            ) >= max(0.80, ba400 - 0.01),
            "unknown_recall_at_least_point80": _finite(
                _metric(metrics, "unknown_recall", "aggregate_unknown_recall"),
                "unknown_recall",
            ) >= 0.80,
            "free_recall_at_least_point75": free >= 0.75,
            "occupied_recall_at_least_max_point70_or_update_400_minus_point03": (
                occupied >= max(0.70, occupied400 - 0.03)
            ),
            "free_occupied_gap_at_most_point25": abs(free - occupied) <= 0.25,
            "rough_balanced_accuracy_at_least_max_point772_or_update_400_minus_point01": _finite(
                _metric(metrics, "rough_balanced_accuracy", "rough_raster_balanced_accuracy"),
                "rough_balanced_accuracy",
            ) >= max(0.772, rough_ba400 - 0.01),
            "rough_occupied_recall_at_least_max_point65_or_update_400_minus_point03": _finite(
                _metric(metrics, "rough_occupied_recall", "rough_raster_occupied_recall"),
                "rough_occupied_recall",
            ) >= max(0.65, rough_occ400 - 0.03),
            "paired_rgb_margin_strictly_positive": _finite(
                _metric(metrics, "paired_rgb_margin", "paired_rgb_aggregate_margin"),
                "paired_rgb_margin",
            ) > 0.0,
            "paired_rgb_scene_wins_equal_8": _integer(
                _metric(metrics, "paired_rgb_scene_wins", "correct_rgb_scene_win_count"),
                "paired_rgb_scene_wins",
            ) == 8,
            "latent_prediction_loss_at_most_point90_B400": _finite(
                _metric(metrics, "latent_prediction_loss"), "latent_prediction_loss"
            ) <= 0.90 * B400,
            "action_nll_strictly_below_point95_log9": _finite(
                _metric(metrics, "action_nll"), "action_nll"
            ) < 0.95 * LOG9,
            "action_macro_balanced_accuracy_strictly_above_two_ninths": _finite(
                _metric(metrics, "action_macro_balanced_accuracy"),
                "action_macro_balanced_accuracy",
            ) > 2.0 / 9.0,
            "executed_action_beats_hardest_wrong_in_at_least_6_families": _integer(
                _metric(
                    metrics,
                    "executed_action_beats_hardest_wrong_family_count",
                    "hardest_wrong_positive_scene_count",
                ),
                "executed_action_beats_hardest_wrong_family_count",
            ) >= 6,
            "mean_wrong_action_energy_strictly_above_executed": _finite(
                _metric(metrics, "mean_wrong_action_energy"), "mean_wrong_action_energy"
            ) > _finite(
                _metric(metrics, "mean_executed_action_energy"),
                "mean_executed_action_energy",
            ),
            "non_hold_mean_hold_energy_strictly_above_executed": _finite(
                _metric(
                    metrics,
                    "non_hold_mean_hold_or_zero_action_energy",
                    "mean_non_hold_hold_action_energy",
                ),
                "non_hold_mean_hold_or_zero_action_energy",
            ) > _finite(
                _metric(
                    metrics,
                    "non_hold_mean_executed_action_energy",
                    "mean_non_hold_executed_action_energy",
                ),
                "non_hold_mean_executed_action_energy",
            ),
            "same_action_correct_next_deranged_nll_strictly_below_point95_log2": _finite(
                _metric(metrics, "same_action_correct_next_deranged_nll", "same_action_target_nll"),
                "same_action_correct_next_deranged_nll",
            ) < 0.95 * LOG2,
            "same_action_correct_next_strict_win_rate_at_least_point65": _finite(
                _metric(metrics, "same_action_correct_next_strict_win_rate", "same_action_target_strict_win_rate"),
                "same_action_correct_next_strict_win_rate",
            ) >= 0.65,
            "same_action_correct_next_positive_in_at_least_6_families": _integer(
                _metric(
                    metrics,
                    "same_action_correct_next_positive_family_count",
                    "same_action_target_positive_scene_count",
                ),
                "same_action_correct_next_positive_family_count",
            ) >= 6,
            "target_effective_rank_retains_point75_update_400": _finite(
                _metric(metrics, "target_effective_rank"), "target_effective_rank"
            ) >= 0.75 * _finite(
                _metric(four_hundred, "target_effective_rank"),
                "update_400.target_effective_rank",
            ),
            "target_channel_variance_retains_point75_update_400": _finite(
                _metric(metrics, "target_channel_variance"), "target_channel_variance"
            ) >= 0.75 * _finite(
                _metric(four_hundred, "target_channel_variance"),
                "update_400.target_channel_variance",
            ),
            "target_spatial_diversity_retains_point75_update_400": _finite(
                _metric(metrics, "target_spatial_diversity"), "target_spatial_diversity"
            ) >= 0.75 * _finite(
                _metric(four_hundred, "target_spatial_diversity"),
                "update_400.target_spatial_diversity",
            ),
            "all_600_joint_updates_passed_shared_gradient_gate": (
                _integer(_metric(metrics, "shared_gradient_ratio_evaluation_count"),
                         "shared_gradient_ratio_evaluation_count") == 600
                and _integer(_metric(metrics, "shared_gradient_ratio_pass_count"),
                             "shared_gradient_ratio_pass_count") == 600
                and _integer(_metric(metrics, "shared_gradient_ratio_failure_count"),
                             "shared_gradient_ratio_failure_count") == 0
            ),
            "semantic_to_dynamics_gradient_ratio_bounds_exact": (
                _finite(_metric(metrics, "minimum_semantic_to_dynamics_gradient_ratio"),
                        "minimum_semantic_to_dynamics_gradient_ratio") >= (1.0 / 32.0)
                and _finite(_metric(metrics, "maximum_semantic_to_dynamics_gradient_ratio"),
                            "maximum_semantic_to_dynamics_gradient_ratio") <= 32.0
            ),
            "dynamics_to_semantic_gradient_ratio_bounds_exact": (
                _finite(_metric(metrics, "minimum_dynamics_to_semantic_gradient_ratio"),
                        "minimum_dynamics_to_semantic_gradient_ratio") >= (1.0 / 32.0)
                and _finite(_metric(metrics, "maximum_dynamics_to_semantic_gradient_ratio"),
                            "maximum_dynamics_to_semantic_gradient_ratio") <= 32.0
            ),
            "representation_gradient_finite_nonzero_all_1000_updates": _integer(
                _metric(metrics, "representation_gradient_finite_nonzero_update_count"),
                "representation_gradient_finite_nonzero_update_count",
            ) == 1_000,
            "predictor_gradient_finite_nonzero_all_600_joint_updates": _integer(
                _metric(metrics, "predictor_gradient_finite_nonzero_update_count"),
                "predictor_gradient_finite_nonzero_update_count",
            ) == 600,
            "semantic_gradient_finite_nonzero_all_600_joint_updates": _integer(
                _metric(metrics, "semantic_gradient_finite_nonzero_joint_update_count"),
                "semantic_gradient_finite_nonzero_joint_update_count",
            ) == 600,
            "dynamics_gradient_finite_nonzero_all_600_joint_updates": _integer(
                _metric(metrics, "dynamics_gradient_finite_nonzero_joint_update_count"),
                "dynamics_gradient_finite_nonzero_joint_update_count",
            ) == 600,
            "online_optimizer_updates_equal_1000": _integer(
                _metric(metrics, "online_optimizer_update_count"),
                "online_optimizer_update_count",
            ) == 1_000,
            "target_ema_updates_equal_1000": _integer(
                _metric(metrics, "target_ema_update_count"), "target_ema_update_count"
            ) == 1_000,
            "predictor_optimizer_updates_equal_600": _integer(
                _metric(metrics, "predictor_optimizer_update_count"),
                "predictor_optimizer_update_count",
            ) == 600,
            "joint_optimizer_updates_equal_600": _integer(
                _metric(metrics, "joint_optimizer_update_count"),
                "joint_optimizer_update_count",
            ) == 600,
            "target_gradient_tensor_count_equals_0": _integer(
                _metric(metrics, "target_gradient_tensor_count"),
                "target_gradient_tensor_count",
            ) == 0,
            "target_optimizer_membership_count_equals_0": _integer(
                _metric(metrics, "target_optimizer_membership_count"),
                "target_optimizer_membership_count",
            ) == 0,
        })

    passed = all(conjuncts.values())
    fail_control, pass_control = GATE_CONTROLS[update]
    return {
        "update": update,
        "kind": {
            0: "structural_only",
            100: "perception_learning_health",
            400: "decisive_perception_anti_collapse",
            1_000: "joint_perception_and_JEPA_qualification",
        }[update],
        "passed": passed,
        "control": pass_control if passed else fail_control,
        "conjuncts": conjuncts,
        "thresholds": {} if update == 0 else copy.deepcopy(GATE_THRESHOLDS[update]),
        "all_conjunctive": True,
        "scientific_gate_evidence": True,
    }


def evaluate_update_401_phase_switch(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "optimizer_identity_unchanged",
        "optimizer_parameter_group_membership_unchanged",
        "joint_objective_formula_exact",
        "online_representation_gradient_finite_nonzero",
        "predictor_gradient_finite_nonzero",
        "target_gradients_absent",
        "shared_gradient_contribution_gate_passed",
    )
    conjuncts = {field: _boolean(metrics.get(field), field) for field in fields}
    conjuncts.update({
        "online_optimizer_update_count_equals_401": _integer(
            _metric(metrics, "online_optimizer_update_count"),
            "online_optimizer_update_count",
        ) == 401,
        "target_ema_update_count_equals_401": _integer(
            _metric(metrics, "target_ema_update_count"), "target_ema_update_count"
        ) == 401,
        "first_predictor_optimizer_update_count_equals_1": _integer(
            _metric(metrics, "predictor_optimizer_update_count"),
            "predictor_optimizer_update_count",
        ) == 1,
        "first_joint_optimizer_update_count_equals_1": _integer(
            _metric(metrics, "joint_optimizer_update_count"),
            "joint_optimizer_update_count",
        ) == 1,
    })
    passed = all(conjuncts.values())
    return {
        "update": 401,
        "kind": "non_scientific_integrity_receipt",
        "passed": passed,
        "control": PHASE_SWITCH_CONTROLS[1 if passed else 0],
        "conjuncts": conjuncts,
        "scientific_gate_evidence": False,
    }


def science_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "repository_goal": (
            "fully_learned_RGB_only_perception_and_JEPA_navigation_stack_"
            "validated_later_on_untouched_externally_custodied_heldout_mazes"
        ),
        "scientific_question": (
            "can_a_geometry_anchored_local_deformable_RGB_to_BEV_lift_preserve_"
            "balanced_obstacle_semantics_and_support_genuine_joint_JEPA"
        ),
        "governing_documents": {
            "preregistration": preregistration_binding(),
            "v3_terminal_audit": v3_terminal_audit_binding(),
        },
        "runtime_inputs": build_runtime_inputs(),
        "data": {
            "roles": ["train", "checkpoint_selection"],
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "target_mapping_bindings": copy.deepcopy(TARGET_MAPPING_BINDINGS),
            "selection_family_bindings": copy.deepcopy(SELECTION_FAMILY_BINDINGS),
            "selection_family_bindings_sha256": SELECTION_FAMILY_BINDINGS_SHA256,
            "fixed_mapped_negative_rule": copy.deepcopy(FIXED_MAPPED_NEGATIVE_RULE),
        },
        "model": model_config(),
        "call_graph": dict(CALL_GRAPH_CONTRACT),
        "objective": objective_contract(),
        "optimizer": optimizer_contract(),
        "schedule": build_schedule_identity(),
        "gates": {
            "observations": list(OBSERVATION_UPDATES),
            "phase_switch_integrity_receipt": 401,
            "thresholds": {
                str(update): copy.deepcopy(value)
                for update, value in GATE_THRESHOLDS.items()
            },
            "all_conditions_conjunctive": True,
            "stop_at_first_applicable_failure": True,
        },
        "lifecycle": {
            "attempt_index": ATTEMPT_INDEX,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_must_be_absent_before_mode_0700_reservation": True,
            "checkpoint_and_training_trace_write_only": True,
            "retry_resume_repair_recovery_extension_or_second_attempt": False,
        },
        "authority": dict(DOWNSTREAM_DENIALS),
    }


SCIENTIFIC_REVIEW_CHECKS = {
    "preregistration_and_v3_terminal_audit_exact": True,
    "runtime_interpreter_inputs_roles_mappings_schedule_and_caps_exact": True,
    "geometry_anchored_local_deformable_lift_has_no_global_bypass": True,
    "warmup_and_genuine_joint_JEPA_objectives_exact": True,
    "EMA_target_detached_and_predictor_jointly_trained_only_after_update_400": True,
    "separate_gradient_clips_and_every_joint_update_ratio_gate_exact": True,
    "all_gate_arithmetic_comparator_strictness_and_accounting_exact": True,
    "one_fresh_attempt_write_only_outputs_and_downstream_denials_exact": True,
    "source_freeze_commit_matches_reviewed_tree": True,
}


def artifact_binding(
    path: str,
    raw: bytes,
    *,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    safe_relative_path(path)
    value: dict[str, Any] = {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if content_sha256 is not None:
        if not is_sha256(content_sha256):
            raise ValueError("content SHA-256 is malformed")
        value["content_sha256"] = content_sha256
    return value


def safe_relative_path(value: object) -> str:
    if type(value) is not str or not value:
        raise PermissionError("path must be one nonempty relative string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise PermissionError("path is not safe and relative")
    return value


def safe_relative_source_path(value: object) -> str:
    path = safe_relative_path(value)
    parts = PurePosixPath(path).parts
    if (
        not path.endswith(".py")
        or path.endswith("sealed_test.json")
        or any(
            part in {".generated", "heldout", "sealed"}
            or part.startswith(("heldout_", "sealed_"))
            for part in parts
        )
    ):
        raise PermissionError(f"forbidden source path: {path}")
    return path


def _read_regular_source(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"source is not regular: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    fingerprint = lambda row: (
        row.st_dev, row.st_ino, row.st_mode, row.st_size,
        row.st_mtime_ns, row.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or fingerprint(before) != fingerprint(opened)
        or fingerprint(opened) != fingerprint(after_open)
        or fingerprint(after_open) != fingerprint(after)
    ):
        raise PermissionError(f"source changed while read: {path}")
    return b"".join(chunks)


def _parse_pretty_content_bound(
    raw: bytes, *, name: str, expected_content_sha256: str
) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_direct._reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise PermissionError(f"{name} top level changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    # The preregistration's content identity was computed with jq's canonical
    # number rendering; exact raw SHA-256 and byte count are checked by the
    # caller, and the declared jq-bound identity is checked here.  Re-encoding
    # JSON numbers with Python would produce a different digest for values such
    # as 1E-8 without any semantic document change.
    if declared != expected_content_sha256:
        raise PermissionError(f"{name} content hash changed")
    return dict(value)


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    audit_raw = _read_regular_source(root / V3_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("preregistration raw identity changed")
    prereg = _parse_pretty_content_bound(
        prereg_raw,
        name="preregistration",
        expected_content_sha256=PREREGISTRATION_CONTENT_SHA256,
    )
    if prereg.get("decision_basis", {}).get("v3_terminal_audit") != (
        v3_terminal_audit_binding()
    ):
        raise PermissionError("preregistration V3 terminal binding changed")
    if (
        len(audit_raw) != V3_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest() != V3_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V3 terminal audit raw identity changed")
    audit = _parse_pretty_content_bound(
        audit_raw,
        name="V3 terminal audit",
        expected_content_sha256=V3_TERMINAL_AUDIT_CONTENT_SHA256,
    )
    if (
        audit.get("content_sha256") != V3_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != V3_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V3_TERMINAL_AUDIT_CLASSIFICATION
    ):
        raise PermissionError("V3 terminal conclusion changed")
    return {
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        V3_TERMINAL_AUDIT_RELATIVE_PATH: V3_TERMINAL_AUDIT_FILE_SHA256,
    }


def validate_source_manifest(
    raw: bytes, root: Path = ROOT
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_PATHS)
        or value.get("excluded_runtime_categories") != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != len(SOURCE_PATHS)
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("source manifest contract changed")
    for relative, binding in zip(SOURCE_PATHS, bindings, strict=True):
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256", "byte_count"}
            or binding.get("path") != relative
            or safe_relative_source_path(relative) != relative
            or not is_sha256(binding.get("file_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("source binding changed")
        source_raw = _read_regular_source(root / relative)
        if (
            len(source_raw) != binding["byte_count"]
            or hashlib.sha256(source_raw).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    result = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(manifest_raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _validate_artifact_binding(
    value: object, *, path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "path", "file_sha256", "content_sha256", "byte_count"
    }:
        raise PermissionError("artifact binding fields changed")
    if (
        value.get("path") != path
        or not is_sha256(value.get("file_sha256"))
        or not is_sha256(value.get("content_sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
    ):
        raise PermissionError("artifact binding changed")
    return dict(value)


def _source_freeze_commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{name} must be one lowercase 40-hex commit")
    return value


def validate_review(
    raw: bytes,
    manifest_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="source review")
    expected_manifest = _validate_artifact_binding(
        dict(manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    if expected_manifest != artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    ):
        raise PermissionError("review manifest binding changed")
    expected_sources = current_source_bindings(root)
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "v3_terminal_audit", "science_contract",
        "source_only_checks", "scientific_checks", "findings", "authority",
        "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status") != "PASS_SOURCE_AND_SCIENCE"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != expected_sources
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("v3_terminal_audit") != v3_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("source_only_checks") != {
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("source review did not pass exact frozen science")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="execution authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=parse_canonical_json(
            review_raw, name="source review"
        )["content_sha256"],
    ):
        raise PermissionError("authorization review binding changed")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    manifest_binding = artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    review = validate_review(review_raw, manifest_binding, root=root)
    fields = {
        "schema", "status", "authorizer", "source_freeze_commit",
        "independent_source_review", "preregistration", "v3_terminal_audit",
        "runtime_inputs", "experiment", "authority", "content_sha256",
    }
    authorizer = value.get("authorizer")
    source_commit = _source_freeze_commit(
        review.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, review["reviewer"]}
        or value.get("source_freeze_commit") != source_commit
        or value.get("independent_source_review") != expected_review
        or value.get("preregistration") != preregistration_binding()
        or value.get("v3_terminal_audit") != v3_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


NORMAL_RECEIPT_PATHS = (
    "reservation.json", "metrics.json", "artifact.json", "access.json",
    "result.json", "completed.json",
)
OPERATIONAL_FAILURE_RECEIPT_PATHS = (
    "metrics.json", "artifact.json", "access.json", "result.json",
    "failure.json", "completed.json",
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V1_"
    "INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)

__all__ = [
    name for name in globals()
    if name.isupper() or name in {
        "artifact_binding", "build_runtime_inputs", "build_schedule_identity",
        "canonical_json_bytes", "canonical_json_sha256", "current_source_bindings",
        "evaluate_gate", "evaluate_update_401_phase_switch", "is_sha256",
        "model_config", "objective_contract", "optimizer_contract",
        "parse_canonical_json", "preregistration_binding",
        "runtime_authorization_template", "safe_relative_path",
        "safe_relative_source_path", "science_contract", "validate_authorization",
        "validate_governing_documents", "validate_review",
        "validate_source_manifest", "v3_terminal_audit_binding",
        "with_content_sha256",
    }
]
