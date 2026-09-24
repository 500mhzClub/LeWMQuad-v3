"""Contract for one RGB-only signed-boundary-distance Direct-BEV probe.

The frozen V12 143-source perception stack is the direct source base.  The
active mechanism replaces the prototype state head and class-NLL training
objective with two learned tanh fields and the preregistered sign-balanced
Huber objective.  Importing this source grants no runtime authority and opens
no generated input, RGB, raster, checkpoint, trace, heldout, or sealed data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V12_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py"
)


def _source_only_module(name: str, relative_path: str) -> Any:
    source = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V12 = _source_only_module(
    "_lewm_direct_bev_signed_boundary_distance_frozen_v12_contract",
    FROZEN_V12_CONTRACT_RELATIVE_PATH,
)
for _name in _V12.__all__:
    globals()[_name] = getattr(_V12, _name)

canonical_json_bytes = _V12.canonical_json_bytes
canonical_json_sha256 = _V12.canonical_json_sha256
is_sha256 = _V12.is_sha256
with_content_sha256 = _V12.with_content_sha256
parse_canonical_json = _V12.parse_canonical_json
artifact_binding = _V12.artifact_binding
validate_binding = _V12.validate_binding

IMPLEMENTATION_AUTHOR = "/root/sdf_prereg_author"
IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/sdf_math_reviewer",
    "/root/sdf_prereg_author",
    "/root/v9_preregistration_author",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/"
    "check_go2_direct_egocentric_bev_signed_boundary_distance_state_v1_"
    "source_closure.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V12_RUNNER_RELATIVE_PATH = _V12.RUNNER_RELATIVE_PATH
FROZEN_V12_LAUNCHER_RELATIVE_PATH = _V12.LAUNCHER_RELATIVE_PATH
FROZEN_V12_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V12.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
FROZEN_V9_RUNNER_RELATIVE_PATH = _V12._V11._V10.FROZEN_V9_RUNNER_RELATIVE_PATH
FROZEN_V9_LAUNCHER_RELATIVE_PATH = (
    _V12._V11._V10.FROZEN_V9_LAUNCHER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/"
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "4d0213683f82ca468764bd97e97d9d6f8be60142"
PREREGISTRATION_FILE_SHA256 = (
    "28b6fb5199c7f2d697e54ddae63c75cb8837f3d90df24c0a46e2b09b2fdb6191"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "794716506c36d3077cd19957e7205c1eada623d6fcffc2c680e4602f80028dc9"
)
PREREGISTRATION_BYTE_COUNT = 28_005
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_RGB_ONLY_SIGNED_BOUNDARY_DISTANCE_STATE_V1_"
    "PERCEPTION_FALSIFICATION_PENDING_SOURCE_FREEZE_INDEPENDENT_REVIEW_"
    "AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH = _V12.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V12_SOURCE_MANIFEST_COMMIT = (
    "3673812bb325b867c1654e191d4487b5d0cfa600"
)
FROZEN_V12_SOURCE_MANIFEST_FILE_SHA256 = (
    "32bc3ef31790144b06e7993e10d631929179ab242cc072abb32f5e1d25855668"
)
FROZEN_V12_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9da41897d37d91f07169c94be5e1b09dd89cbb6434c3384e9b6505e43cd30bdc"
)
FROZEN_V12_SOURCE_MANIFEST_BYTE_COUNT = 51_081
FROZEN_V12_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V12_SOURCE_COUNT = 143

FROZEN_V12_REVIEW_RELATIVE_PATH = _V12.REVIEW_RELATIVE_PATH
FROZEN_V12_REVIEW_COMMIT = "d38b295bbeaa1a9a101921e95642a8b031b09b03"
FROZEN_V12_REVIEW_FILE_SHA256 = (
    "d90a2b107d332235c1ae7cc5736d2167ee93123af0bc985ba1e5a4b1840e434c"
)
FROZEN_V12_REVIEW_CONTENT_SHA256 = (
    "d21dce5a4c6a00a20b8fba34c8063fd0b06f58c95919d4af74a6ffbb4a0eaadc"
)
FROZEN_V12_REVIEW_BYTE_COUNT = 81_121
FROZEN_V12_REVIEW_STATUS = (
    "PASS_SOURCE_UPDATE50_TREND_GATE_TIMING_SCIENCE_AND_CUSTODY"
)

FROZEN_V12_AUTHORIZATION_RELATIVE_PATH = _V12.AUTHORIZATION_RELATIVE_PATH
FROZEN_V12_AUTHORIZATION_COMMIT = (
    "466b0ff8cfb403350eacc0d2d93cc97489bb897b"
)
FROZEN_V12_AUTHORIZATION_FILE_SHA256 = (
    "7079764583a1c659511180903c100cbd7676209a6c5974930acc46787da7d2bb"
)
FROZEN_V12_AUTHORIZATION_CONTENT_SHA256 = (
    "7762c0687bf4880919d7c3243b15f94068ddfde71aad5e1a084f6b21b1a3a9b7"
)
FROZEN_V12_AUTHORIZATION_BYTE_COUNT = 64_937
FROZEN_V12_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V12_UPDATE50_TREND_GATE_TIMING_"
    "PERCEPTION_FALSIFICATION"
)

V12_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_update50_trend_"
    "gate_timing_terminal_audit_2026-07-27.json"
)
V12_TERMINAL_AUDIT_COMMIT = "8c0015822cfaf0c845c345711c27b8907681ae7a"
V12_TERMINAL_AUDIT_FILE_SHA256 = (
    "e5cbe568253a287e6018f5def20b988158ea4192a345e483d8f946836b29a7c5"
)
V12_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "40cd42094b143c69fcb009948316a5ba2141fa792455b38ba5c1f17b4429061c"
)
V12_TERMINAL_AUDIT_BYTE_COUNT = 14_828
V12_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_RECEIPT_CHAIN_UPDATE_100_SCIENTIFIC_FAILURE_V12_"
    "CLOSED_MODEL_LOSS_LINE_RETIRED_NO_V13"
)
V12_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_100_SCIENTIFIC_GATE_FAILURE_AFTER_STRONG_CONTINUED_"
    "LEARNING_OCCUPIED_RECALL_MATERIALLY_FAILED_AND_NLL_NARROWLY_FAILED_"
    "V12_CLOSED_MODEL_LOSS_LINE_RETIRED_NO_V13"
)

MULTIPROTOTYPE_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1_"
    "terminal_audit_2026-07-27.json"
)
MULTIPROTOTYPE_TERMINAL_AUDIT_COMMIT = (
    "c956c4c2ebb76f8e9e489c7024601ae399842eaf"
)
MULTIPROTOTYPE_TERMINAL_AUDIT_FILE_SHA256 = (
    "700db3735b167b0108fca7d33ab6eba08c1ac8b5d721da59a3ad67784b037481"
)
MULTIPROTOTYPE_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "4d5e0f611195cf1eeaeb266b51b805e9dc67230d5dac5ac32e8aba9ed95ee79c"
)
MULTIPROTOTYPE_TERMINAL_AUDIT_BYTE_COUNT = 18_277
MULTIPROTOTYPE_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_RECEIPT_CHAIN_UPDATE_ZERO_GROUNDING_FAILURE_"
    "MULTIPROTOTYPE_V1_CLOSED_NO_RETRY"
)
MULTIPROTOTYPE_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_ZERO_GROUNDING_CONTROL_FAILURE_ONLY_CORRECT_RGB_SCENE_"
    "WINS_7_OF_8_NO_TRAINING_MULTIPROTOTYPE_V1_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/"
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/"
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/"
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_distance_state_v1_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V12.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
RETIRED_MULTIPROTOTYPE_SOURCE_PATHS = (
    "lewm/benchmarks/go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
    "lewm/models/direct_egocentric_bev_multiprototype_state_head_v1.py",
    "lewm/tests/test_go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
    "scripts/check_go2_direct_egocentric_bev_multiprototype_state_head_v1_source_closure.py",
    "scripts/launch_go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
    "scripts/run_go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
)
if len(REUSED_SOURCE_PATHS) != 143 or len(ADDITIVE_SOURCE_PATHS) != 6:
    raise RuntimeError("signed-boundary source-family cardinality changed")
if len(SOURCE_PATHS) != 149:
    raise RuntimeError("signed-boundary recursive source cardinality changed")
if set(RETIRED_MULTIPROTOTYPE_SOURCE_PATHS) & set(SOURCE_PATHS):
    raise RuntimeError("retired multiprototype source entered active graph")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V12_REVIEW_RELATIVE_PATH,
    FROZEN_V12_AUTHORIZATION_RELATIVE_PATH,
    V12_TERMINAL_AUDIT_RELATIVE_PATH,
    MULTIPROTOTYPE_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_"
    "distance_state_v1/rgb_direct_egocentric_bev_signed_boundary_distance_"
    "state_probe_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_DISTANCE_STATE_V1_"
    "PREFLIGHT_JSON"
)

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

REVIEW_STATUS = (
    "PASS_SOURCE_SIGNED_BOUNDARY_DISTANCE_STATE_V1_SCIENCE_AND_CUSTODY"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_SIGNED_BOUNDARY_DISTANCE_STATE_V1_"
    "PERCEPTION_FALSIFICATION"
)
SOURCE_ONLY_AUTHORITY = dict(_V12.SOURCE_ONLY_AUTHORITY)
REVIEW_AUTHORITY = dict(_V12.REVIEW_AUTHORITY)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

MAXIMUM_ATTEMPTS = 1
ATTEMPT_INDEX = 1
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
GPU_ACTIVE_TIME_CAP_MINUTES = 30
EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
CHECKPOINT_UPDATES = (100, 400, 1_000)
SNAPSHOT_UPDATES = CHECKPOINT_UPDATES
OBSERVATION_UPDATES = (0, *CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}

BOUNDARY_DISTANCE_RADIUS_CELLS = 8.0
BOUNDARY_DISTANCE_HALF_CELL_CORRECTION = 0.5
BOUNDARY_HUBER_DELTA = 0.125
HIERARCHICAL_ADAPTER_SCALE = 16.0
STATE_FIELD_CHANNEL_ORDER = (
    "K_known_signed_boundary_distance",
    "O_free_occupied_signed_boundary_distance",
)
SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_COUNT = 130
SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_TENSOR_COUNT = 2

MODEL_BINDINGS_FROZEN = True
SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256 = (
    "a3582ca41e41963592f4bf76ba7de432b51fed783408dcda3e3de9c070c9f40f"
)
V8_INITIAL_DECODER_STATE_SHA256 = _V12.V8_INITIAL_DECODER_STATE_SHA256
V8_INITIAL_PROTOTYPE_HEAD_STATE_SHA256 = (
    SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256
)
V8_INITIAL_PREDICTOR_STATE_SHA256 = _V12.V8_INITIAL_PREDICTOR_STATE_SHA256
V8_FRESH_DRAW_ORDER = (
    "factorized_row_queries",
    "factorized_column_queries",
    "token_projection",
    "shared_projected_token_layer_norm",
    "cross_attention_block_1",
    "ffn_block_1",
    "cross_attention_block_2",
    "ffn_block_2",
    "signed_boundary_distance_conv1x1_weight_and_bias",
)
MODEL_PARAMETER_INVENTORY: dict[str, Any] = {
    "encoder": {
        "parameter_count": 2_747_520,
        "tensor_count": 78,
        "ordered_parameter_name_sha256": (
            "8b83921e9766a68b59b35d1c5ef15dea6db13aeb4a6c91c2c13ab2e22d8b1c5e"
        ),
    },
    "decoder_state": {
        "parameter_count": 87_746,
        "tensor_count": 32,
        "ordered_parameter_name_sha256": (
            "93facb8b8d4059e7270ebb90dbff26572c6f1700bb302d8a2b7177bb5777c147"
        ),
    },
    "predictor": {
        "parameter_count": 317_107,
        "tensor_count": 79,
        "ordered_parameter_name_sha256": (
            "0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd"
        ),
    },
    "detached_target_encoder_decoder_state": {
        "parameter_count": 2_835_266,
        "tensor_count": 110,
        "ordered_parameter_name_sha256": (
            "5c96af932aa8bfd597c619b4481038eff912feb1c2103b82f7bfb44f29d0bd1b"
        ),
    },
    "total": {
        "parameter_count": 5_987_639,
        "tensor_count": 299,
    },
}

CONTROL_PRELIMINARY = (
    "PRELIMINARY_PASS_SIGNED_BOUNDARY_DISTANCE_V1_DISPATCH_NOT_FINAL_"
    "SCIENTIFIC_EVIDENCE"
)
CONTROL_UPDATE_0_FAIL = (
    "FAIL_UPDATE_ZERO_SIGNED_BOUNDARY_DISTANCE_V1_STRUCTURAL_INTEGRITY_"
    "GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_0 = (
    "CONTINUE_AFTER_UPDATE_ZERO_SIGNED_BOUNDARY_DISTANCE_V1_STRUCTURAL_"
    "INTEGRITY_GATE"
)
CONTROL_UPDATE_100_FAIL = (
    "FAIL_UPDATE_100_SIGNED_BOUNDARY_DISTANCE_V1_LEARNING_HEALTH_GATE_"
    "TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_100 = (
    "CONTINUE_AFTER_UPDATE_100_SIGNED_BOUNDARY_DISTANCE_V1_LEARNING_"
    "HEALTH_GATE"
)
CONTROL_UPDATE_400_FAIL = (
    "FAIL_UPDATE_400_SIGNED_BOUNDARY_DISTANCE_V1_BALANCED_SEMANTIC_GATE_"
    "TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_400 = (
    "CONTINUE_AFTER_UPDATE_400_SIGNED_BOUNDARY_DISTANCE_V1_BALANCED_"
    "SEMANTIC_GATE"
)
CONTROL_UPDATE_1000_FAIL = (
    "FAIL_UPDATE_1000_SIGNED_BOUNDARY_DISTANCE_V1_QUALIFICATION_GATE_"
    "TERMINAL_NO_RETRY"
)
CONTROL_PASS = (
    "PASS_RGB_ONLY_SIGNED_BOUNDARY_DISTANCE_STATE_V1_PERCEPTION_MECHANISM_"
    "ONLY"
)
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_0_FAIL, CONTROL_CONTINUE_UPDATE_0),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    400: (CONTROL_UPDATE_400_FAIL, CONTROL_CONTINUE_UPDATE_400),
    1_000: (CONTROL_UPDATE_1000_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())
GATE_THRESHOLDS = {
    0: {
        "structural_integrity_only": True,
        "semantic_rgb_direction_applied": False,
    },
    100: {
        "G_strictly_less_than_update_0": True,
        "raster_nll_maximum_relative_to_update_0": -0.15,
        "balanced_accuracy_minimum_relative_to_update_0": 0.08,
        "minimum_free_and_occupied_recall": 0.10,
        "paired_rgb_aggregate_margin_strictly_positive": True,
        "correct_rgb_scene_wins_minimum": 6,
    },
    400: {
        "G_strictly_less_than_update_100": True,
        "raster_nll_maximum_absolute": 0.55,
        "raster_nll_maximum_relative_to_update_100": -0.03,
        "balanced_accuracy_minimum_absolute": 0.68,
        "balanced_accuracy_minimum_relative_to_update_100": 0.03,
        "free_recall_minimum": 0.50,
        "occupied_recall_minimum": 0.55,
        "free_occupied_gap_maximum": 0.35,
        "rough_balanced_accuracy_minimum_absolute": 0.65,
        "rough_balanced_accuracy_minimum_relative_to_update_100": 0.02,
        "rough_occupied_recall_minimum": (
            "max(0.50,rough_occupied_update_100+0.05)"
        ),
        "paired_rgb_aggregate_margin_strictly_positive": True,
        "correct_rgb_scene_wins_minimum": 7,
    },
    1_000: {
        "G_strictly_less_than_update_400": True,
        "balanced_accuracy_minimum_absolute": 0.80,
        "balanced_accuracy_at_least_update_400": True,
        "raster_nll_maximum_absolute_scale16_calibration": 0.42,
        "raster_nll_at_most_update_400": True,
        "free_recall_minimum": 0.68,
        "occupied_recall_minimum": 0.88,
        "unknown_recall_minimum": 0.80,
        "free_occupied_gap_maximum": 0.25,
        "rough_balanced_accuracy_minimum": 0.772,
        "rough_occupied_recall_minimum": 0.65,
        "paired_rgb_aggregate_margin_strictly_positive": True,
        "correct_rgb_scene_wins_minimum": 7,
    },
}

INTEGRITY_FIELDS = (
    "fresh_signed_boundary_distance_model_and_optimizer_zero_prior_runtime_reuse",
    "frozen_encoder_decoder_predictor_initialization_exact",
    "registered_seed_draw_order_exact",
    "signed_boundary_distance_initial_head_state_sha256_exact",
    "model_parameter_inventory_exact",
    "signed_boundary_distance_decoder_head_parameter_inventory_exact",
    "learned_only_forbidden_geometry_absent",
    "two_residual_cross_attention_ffn_blocks_exact",
    "signed_boundary_distance_head_shape_tanh_and_channel_order_exact",
    "signed_boundary_distance_center_edt_transform_exact",
    "signed_boundary_distance_huber_macro_objective_exact",
    "hierarchical_adapter_scale16_formula_and_normalization_exact",
    "exact_target_adapter_argmax_semantics",
    "paired_rgb_direction_free_nonidentity_all_8",
    "K_and_O_head_gradients_finite_nonzero",
    "online_target_perception_bitwise_equal",
    "target_requires_grad_false",
    "two_channel_raw_state_exact",
    "three_channel_adapter_logits_exact",
    "predictor_target_and_fixed_negative_gradients_absent",
    "no_hidden_auxiliary_bypass",
    "update_zero_semantic_direction_gate_absent",
    "all_forbidden_access_counts_zero",
)
FINAL_MARKERS = (
    "signed_boundary_distance_mechanism_receipt_ready",
    "active_training_scope_signed_boundary_distance_v1",
)

EXECUTION_AUTHORITY = {
    **dict(_V12.EXECUTION_AUTHORITY),
    "maximum_updates": MAXIMUM_UPDATES,
    "maximum_presentations": MAXIMUM_PRESENTATIONS,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v8_learned_bev_query_prototype_perception_attempt_only": False,
    "learned_bev_query_prototype_perception_only": False,
    "one_fresh_v10_final_class_macro_grounding_perception_attempt_only": False,
    "final_class_macro_grounding_perception_only": False,
    "one_fresh_v12_update50_trend_gate_timing_only": False,
    "science_identical_to_frozen_v11": False,
    "model_data_seed_schedule_loss_optimizer_and_ema_identical_to_frozen_v11": (
        False
    ),
    "one_fresh_signed_boundary_distance_state_v1_perception_attempt_only": (
        True
    ),
    "sole_trainable_mechanism_delta_is_two_field_signed_boundary_state": True,
    "exact_full_direct_bev_16000_presentation_schedule_only": True,
    "science_identical_to_frozen_v8": False,
    "science_identical_to_frozen_v10": False,
    "science_identical_to_frozen_v12": False,
    "v12_or_multiprototype_runtime_output_or_state_reuse_authorized": False,
    "multiprototype_v2_or_v13_timing_successor_authorized": False,
    "predictor_training_or_evaluation_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
}


def frozen_v12_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V12_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V12_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V12_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V12_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V12_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V12_SOURCE_COUNT,
    }


def frozen_v12_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V12_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V12_REVIEW_COMMIT,
        "file_sha256": FROZEN_V12_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V12_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V12_REVIEW_BYTE_COUNT,
        "status": FROZEN_V12_REVIEW_STATUS,
    }


def frozen_v12_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V12_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V12_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V12_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V12_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V12_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V12_AUTHORIZATION_STATUS,
    }


def v12_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V12_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V12_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V12_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V12_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V12_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V12_TERMINAL_AUDIT_STATUS,
        "classification": V12_TERMINAL_AUDIT_CLASSIFICATION,
    }


def multiprototype_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": MULTIPROTOTYPE_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": MULTIPROTOTYPE_TERMINAL_AUDIT_COMMIT,
        "file_sha256": MULTIPROTOTYPE_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": MULTIPROTOTYPE_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": MULTIPROTOTYPE_TERMINAL_AUDIT_BYTE_COUNT,
        "status": MULTIPROTOTYPE_TERMINAL_AUDIT_STATUS,
        "classification": MULTIPROTOTYPE_TERMINAL_AUDIT_CLASSIFICATION,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
        "status": PREREGISTRATION_STATUS,
    }


def build_schedule_identity() -> dict[str, Any]:
    value = deepcopy(_V12.build_schedule_identity())
    value.update({
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
        "prefix_rule": (
            "exact_existing_frozen_full_Direct_BEV_16000_presentation_"
            "schedule_without_filter_reorder_resample_replacement_or_extension"
        ),
    })
    return value


def perception_accounting(update: int) -> dict[str, int]:
    if type(update) is not int or not 0 <= update <= MAXIMUM_UPDATES:
        raise ValueError("signed-boundary perception-accounting out of bounds")
    return {
        "target_update_callback_count": update,
        "online_perception_optimizer_update_count": update,
        "target_ema_update_count": update,
        "presentations": update * EFFECTIVE_BATCH_SIZE,
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_update_count": 0,
        "predictor_optimizer_membership_count": 0,
        "predictor_requires_grad_parameter_count": 0,
    }


PERCEPTION_ACCOUNTING = {
    update: perception_accounting(update) for update in OBSERVATION_UPDATES
}


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V12.runtime_authorization_template())
    value["schedule"] = build_schedule_identity()
    value["experiment_scope"].update({
        "one_fresh_attempt": True,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_index": ATTEMPT_INDEX,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "maximum_active_gpu_minutes": GPU_ACTIVE_TIME_CAP_MINUTES,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "fresh_initialization_required": True,
        "prior_runtime_or_checkpoint_reuse": False,
        "v12_runtime_output_reuse": False,
        "v12_retry_resume_repair_or_recovery": False,
        "multiprototype_runtime_output_reuse": False,
        "signed_boundary_distance_state_v1_only": True,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": (
            False
        ),
    })
    return value


def model_config() -> dict[str, Any]:
    value = deepcopy(_V12.model_config())
    value["state_head"] = {
        "operator": "Conv2d(64,2,kernel_size=1,bias=True)_then_tanh",
        "parameter_count": SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_COUNT,
        "parameter_tensor_count": (
            SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_TENSOR_COUNT
        ),
        "channel_order": list(STATE_FIELD_CHANNEL_ORDER),
        "output_shape": ["B", 2, 64, 64],
        "output_range_inclusive": [-1.0, 1.0],
        "fixed_hierarchical_adapter_scale": HIERARCHICAL_ADAPTER_SCALE,
        "hidden_or_auxiliary_bypass_authorized": False,
    }
    return value


def science_contract() -> dict[str, Any]:
    value = deepcopy(_V12.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Can two learned half-cell-corrected signed center-EDT fields fix "
        "occupied-space fidelity while retaining learned RGB grounding?"
    )
    value["governing_documents"]["active_preregistration"] = (
        preregistration_binding()
    )
    value["governing_documents"]["v12_terminal_audit"] = (
        v12_terminal_audit_binding()
    )
    value["governing_documents"]["multiprototype_terminal_audit"] = (
        multiprototype_terminal_audit_binding()
    )

    active_head = {
        "input_shape": ["B", 64, 64, 64],
        "operator": "Conv2d(64,2,kernel_size=1,bias=True)_then_tanh",
        "weight_shape": [2, 64, 1, 1],
        "bias_shape": [2],
        "parameter_count": SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_COUNT,
        "parameter_tensor_count": (
            SIGNED_BOUNDARY_DISTANCE_HEAD_PARAMETER_TENSOR_COUNT
        ),
        "channel_order": list(STATE_FIELD_CHANNEL_ORDER),
        "output_shape": ["B", 2, 64, 64],
        "output_range_inclusive": [-1.0, 1.0],
        "learned_temperature_scale_class_weight_or_auxiliary_head": False,
    }
    value["model"]["runtime_source"] = MODEL_RELATIVE_PATH
    value["model"]["state_head"] = deepcopy(active_head)
    decoder = value["model"]["bev_decoder"]
    decoder.pop("prototype_state_head", None)
    decoder["signed_boundary_distance_state_head"] = deepcopy(active_head)
    decoder["counts_include_state_head"] = True
    decoder["online_decoder_parameter_count"] = (
        MODEL_PARAMETER_INVENTORY["decoder_state"]["parameter_count"]
    )
    decoder["online_decoder_parameter_tensor_count"] = (
        MODEL_PARAMETER_INVENTORY["decoder_state"]["tensor_count"]
    )
    initialization = value["model"]["initialization"]
    initialization["fresh_v8_draw_order"] = list(V8_FRESH_DRAW_ORDER)
    initialization["prior_v12_or_multiprototype_runtime_or_parameter_reuse"] = (
        False
    )
    initial_binding = initialization["initial_state_component_binding"]
    initial_binding.pop("fresh_prototype_head_local_state_sha256", None)
    initial_binding["fresh_signed_boundary_distance_head_local_state_sha256"] = (
        SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256
    )
    initial_binding["model_bindings_frozen"] = MODEL_BINDINGS_FROZEN
    initialization["initial_state_component_binding_sha256"] = (
        canonical_json_sha256(initial_binding)
    )
    value["model"]["parameter_inventory"] = deepcopy(MODEL_PARAMETER_INVENTORY)
    value["model"]["call_graph"]["online_current_and_next_RGB"] = (
        "weight_shared_encoder_learned_query_decoder_two_tanh_fields_for_G"
    )

    value["objective"] = {
        "G": {
            "G_current": "mean_rows(current_row_boundary_loss)",
            "G_next": "mean_rows(next_row_boundary_loss)",
            "formula": "0.5*G_current+0.5*G_next",
            "pointwise_loss": "torch_huber_delta_1_over_8",
            "per_row_K": (
                "equal_macro_over_present_UNKNOWN_and_known_sign_groups"
            ),
            "per_row_O": (
                "equal_macro_over_present_FREE_and_OCCUPIED_sign_groups_"
                "with_UNKNOWN_completely_masked"
            ),
            "field_weighting": "equal_0.5_K_0.5_O_when_O_available_else_K_only",
            "row_weighting": "equal_after_each_row_macro_is_formed",
            "hard_label_array": "raster_labels.u1_training_side_only",
        },
        "total": "G",
        "perception_only_total": "G",
        "target_transform": {
            "kind": "half_cell_corrected_center_EDT_surrogate",
            "exact_cell_square_interface_geometry_claimed": False,
            "radius_cells": BOUNDARY_DISTANCE_RADIUS_CELLS,
            "half_cell_correction": BOUNDARY_DISTANCE_HALF_CELL_CORRECTION,
            "empty_opposite_magnitude": 1.0,
            "K_signs": "known_positive_UNKNOWN_negative",
            "O_signs": "FREE_positive_OCCUPIED_negative_UNKNOWN_zero_masked",
        },
        "fixed_hierarchical_adapter": {
            "scale": HIERARCHICAL_ADAPTER_SCALE,
            "UNKNOWN_logit": "logsigmoid(-16*K)",
            "FREE_logit": "logsigmoid(16*K)+logsigmoid(16*O)",
            "OCCUPIED_logit": "logsigmoid(16*K)+logsigmoid(-16*O)",
            "class_order": ["UNKNOWN", "FREE", "OCCUPIED"],
            "learned_parameters": 0,
            "inherited_observer_softmax_preserved": True,
        },
        "wrong_rgb_control": {
            "gradient": "none",
            "loss": "inherited_V10_equal_present_final_class_macro_NLL",
            "adapter": "fixed_scale16_hierarchical_log_probability_logits",
            "aggregate_margin": "mapped_negative_minus_correct",
            "scene_win": "correct_strictly_less_than_mapped_negative",
            "tie_is_win": False,
        },
        "predictor_forward_objective_backward_or_update_count": 0,
        "class_frequency_focal_margin_label_smoothing_or_auxiliary_loss": (
            False
        ),
    }

    value["schedule"] = build_schedule_identity()
    value["gates"]["updates"] = list(OBSERVATION_UPDATES)
    value["gates"]["thresholds"] = {
        str(update): deepcopy(thresholds)
        for update, thresholds in GATE_THRESHOLDS.items()
    }
    value["gates"]["controls"] = {
        str(update): list(controls)
        for update, controls in GATE_CONTROLS.items()
    }
    value["gates"]["perception_accounting"] = {
        str(update): dict(accounting)
        for update, accounting in PERCEPTION_ACCOUNTING.items()
    }
    value["gates"]["frozen_v8_runtime_readiness_markers_required"] = False
    value["gates"]["signed_boundary_distance_markers_required"] = list(
        FINAL_MARKERS
    )
    value["gates"]["update_zero_semantic_directional_gate"] = False
    value["gates"]["preliminary_dispatch_is_scientific_evidence"] = False

    lifecycle = value["lifecycle"]
    lifecycle.update({
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_index": ATTEMPT_INDEX,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scientific_successor_of": _V12.EXPERIMENT_ID,
        "new_architecture_family_not_multiprototype_v2_or_v13": True,
        "v12_or_multiprototype_checkpoint_tensor_trace_receipt_parameter_"
        "optimizer_registry_snapshot_or_rng_reuse": False,
        "retry_resume_extension_second_seed_or_replacement_attempt": False,
    })
    value["phase_adapter"].update({
        "scope": "signed_boundary_distance_state_v1_perception_only",
        "updates": [1, MAXIMUM_UPDATES],
        "presentations": [1, MAXIMUM_PRESENTATIONS],
        "total": "G",
        "optimizer": "one_exact_frozen_v12_adamw_constructed_once_never_reset",
        "predictor_forward_objective_backward_or_update_count": 0,
        "second_phase_present": False,
    })
    value["authority"].update({
        "signed_boundary_distance_v1_execution_authorized_by_source_contract": (
            False
        ),
        "v12_or_multiprototype_runtime_output_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
        "g2_navigation_heldout_sealed_production_promotion_or_deployment_"
        "authorized": False,
    })
    value["repository_goal"] = (
        "fully_learned_RGB_only_perception_JEPA_navigation_stack_validated_"
        "later_on_untouched_externally_custodied_heldout_mazes"
    )
    value["scientific_checks"] = {
        "sole_trainable_delta_is_two_field_signed_boundary_state": True,
        "half_cell_corrected_center_EDT_surrogate_not_exact_interface": True,
        "sign_balanced_row_macro_Huber_delta_1_over_8_exact": True,
        "fixed_scale16_hierarchical_adapter_exact": True,
        "exact_target_adapter_argmax_semantics_required": True,
        "G_is_exact_optimized_objective_and_total": True,
        "paired_RGB_control_is_inherited_V10_class_macro_NLL": True,
        "update_zero_is_structural_only_direction_free_RGB_nonidentity": True,
        "all_four_gates_match_committed_preregistration": True,
        "exact_full_16000_presentation_schedule_bound": True,
        "fresh_initialization_and_zero_prior_runtime_reuse": True,
        "predictor_frozen_excluded_and_zero_forward_objective_backward_update": (
            True
        ),
        "six_additive_sources_over_frozen_V12_143_source_base": True,
        "multiprototype_sources_absent_from_active_import_graph": True,
        "one_fresh_attempt_caps_and_downstream_denials_exact": True,
        "no_runtime_or_protected_material_opened_by_source_work": True,
    }
    value["scientific_delta"] = {
        "trainable_mechanism_delta_count": 1,
        "mechanism": "two_tanh_signed_boundary_distance_fields",
        "old_state_head": "three_normalized_class_prototypes",
        "new_state_head": "biased_1x1_64_to_2_then_tanh",
        "old_training_objective": "equal_present_class_macro_NLL",
        "new_training_objective": "sign_balanced_row_macro_Huber_delta_1_over_8",
        "fixed_metrics_adapter_scale": 16.0,
        "decision_contract_is_new_family_specific": True,
        "not_multiprototype_v2_or_v13_timing_successor": True,
    }
    return value


def science_identity_receipt() -> dict[str, Any]:
    return {
        "frozen_v12_science_contract_sha256": canonical_json_sha256(
            _V12.science_contract()
        ),
        "signed_boundary_distance_v1_science_contract_sha256": (
            canonical_json_sha256(science_contract())
        ),
        "preregistration_content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "sole_trainable_mechanism_delta": (
            "two_tanh_signed_boundary_distance_fields"
        ),
        "model_bindings_frozen": MODEL_BINDINGS_FROZEN,
        "frozen_v12_or_multiprototype_runtime_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


def _finite(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _scene_wins(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 8:
        raise ValueError("correct_rgb_scene_win_count must be int in [0,8]")
    return value


def _common_conjuncts(
    update: int,
    metrics: Mapping[str, Any],
    prior: bool,
) -> dict[str, bool]:
    expected = perception_accounting(update)
    result = {
        "prior_gates_passed": prior,
        "signed_boundary_distance_mechanism_receipt_ready": _exact_bool(
            metrics.get("signed_boundary_distance_mechanism_receipt_ready"),
            name="signed_boundary_distance_mechanism_receipt_ready",
        ),
        "active_training_scope_is_perception_only": (
            metrics.get("active_training_scope_signed_boundary_distance_v1")
            == "perception_only"
        ),
        "all_registered_values_finite": _exact_bool(
            metrics.get("all_registered_values_finite"),
            name="all_registered_values_finite",
        ),
        "state_nonconstant": _exact_bool(
            metrics.get("state_nonconstant"), name="state_nonconstant"
        ),
        "all_forbidden_access_counts_zero": _exact_bool(
            metrics.get("all_forbidden_access_counts_zero"),
            name="all_forbidden_access_counts_zero",
        ),
    }
    for field, target in expected.items():
        observed = metrics.get(field)
        if type(observed) is not int:
            raise ValueError(f"{field} must be int")
        result[f"{field}_equals_{target}"] = observed == target
    return result


def _preliminary_gate(update: int, prior_gates_passed: bool) -> dict[str, Any]:
    return {
        "update": update,
        "active_training_scope_signed_boundary_distance_v1_present": False,
        "passed": True,
        "control": CONTROL_PRELIMINARY,
        "gate_mode": (
            "PRELIMINARY_SIGNED_BOUNDARY_DISTANCE_V1_DISPATCH_NOT_FINAL_"
            "SCIENTIFIC_EVIDENCE"
        ),
        "signed_boundary_distance_mechanism_receipt_ready": False,
        "scientific_gate_evidence": False,
        "execution_training_checkpoint_terminal_pass_or_downstream_authority": (
            False
        ),
        "must_be_overwritten_by_signed_boundary_distance_outer_final_dispatch": (
            True
        ),
        "final_gate_evaluated": False,
        "thresholds": deepcopy(GATE_THRESHOLDS[update]),
        "thresholds_applied": False,
        "perception_accounting": perception_accounting(update),
        "perception_accounting_applied": False,
        "prior_gates_passed_validated_only": prior_gates_passed,
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    update_100: Mapping[str, Any] | None = None,
    update_400: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Apply the four preregistered final gates or a preliminary pass."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    if type(prior_gates_passed) is not bool:
        raise ValueError("prior_gates_passed must be bool")
    present = tuple(key in metrics for key in FINAL_MARKERS)
    if present == (False, False):
        return _preliminary_gate(update, prior_gates_passed)
    if present != (True, True):
        raise ValueError("signed-boundary final marker presence is partial")

    conjuncts = _common_conjuncts(update, metrics, prior_gates_passed)
    if update == 0:
        for field in INTEGRITY_FIELDS:
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        hard_sync = metrics.get("initial_online_to_target_hard_sync_count")
        if type(hard_sync) is not int:
            raise ValueError("initial_online_to_target_hard_sync_count must be int")
        conjuncts["initial_online_to_target_hard_sync_count_equals_1"] = (
            hard_sync == 1
        )
    elif update == 100:
        if update_zero is None:
            raise ValueError("update_zero is required at update 100")
        g = _finite(metrics.get("G"), name="G")
        nll = _finite(
            metrics.get("aggregate_raster_nll"), name="aggregate_raster_nll"
        )
        ba = _finite(
            metrics.get("aggregate_raster_balanced_accuracy"),
            name="aggregate_raster_balanced_accuracy",
        )
        free = _finite(
            metrics.get("aggregate_free_recall"), name="aggregate_free_recall"
        )
        occupied = _finite(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        margin = _finite(
            metrics.get("paired_rgb_aggregate_margin"),
            name="paired_rgb_aggregate_margin",
        )
        conjuncts.update({
            "G_strictly_lower_than_update_zero": (
                g < _finite(update_zero.get("G"), name="update_zero.G")
            ),
            "raster_nll_at_most_update_zero_minus_point15": (
                nll
                <= _finite(
                    update_zero.get("aggregate_raster_nll"),
                    name="update_zero.aggregate_raster_nll",
                )
                - 0.15
            ),
            "balanced_accuracy_at_least_update_zero_plus_point08": (
                ba
                >= _finite(
                    update_zero.get("aggregate_raster_balanced_accuracy"),
                    name="update_zero.aggregate_raster_balanced_accuracy",
                )
                + 0.08
            ),
            "minimum_free_and_occupied_recall_at_least_point10": (
                min(free, occupied) >= 0.10
            ),
            "paired_rgb_aggregate_margin_strictly_positive": margin > 0.0,
            "correct_rgb_scene_wins_at_least_6": _scene_wins(
                metrics.get("correct_rgb_scene_win_count")
            )
            >= 6,
        })
    elif update == 400:
        if update_100 is None:
            raise ValueError("update_100 is required at update 400")
        g = _finite(metrics.get("G"), name="G")
        nll = _finite(
            metrics.get("aggregate_raster_nll"), name="aggregate_raster_nll"
        )
        ba = _finite(
            metrics.get("aggregate_raster_balanced_accuracy"),
            name="aggregate_raster_balanced_accuracy",
        )
        free = _finite(
            metrics.get("aggregate_free_recall"), name="aggregate_free_recall"
        )
        occupied = _finite(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        rough_ba = _finite(
            metrics.get("rough_raster_balanced_accuracy"),
            name="rough_raster_balanced_accuracy",
        )
        rough_occupied = _finite(
            metrics.get("rough_raster_occupied_recall"),
            name="rough_raster_occupied_recall",
        )
        margin = _finite(
            metrics.get("paired_rgb_aggregate_margin"),
            name="paired_rgb_aggregate_margin",
        )
        conjuncts.update({
            "G_strictly_lower_than_update_100": (
                g < _finite(update_100.get("G"), name="update_100.G")
            ),
            "raster_nll_at_most_min_point55_or_update100_minus_point03": (
                nll
                <= min(
                    0.55,
                    _finite(
                        update_100.get("aggregate_raster_nll"),
                        name="update_100.aggregate_raster_nll",
                    )
                    - 0.03,
                )
            ),
            "balanced_accuracy_at_least_max_point68_or_update100_plus_point03": (
                ba
                >= max(
                    0.68,
                    _finite(
                        update_100.get("aggregate_raster_balanced_accuracy"),
                        name="update_100.aggregate_raster_balanced_accuracy",
                    )
                    + 0.03,
                )
            ),
            "free_recall_at_least_point50": free >= 0.50,
            "occupied_recall_at_least_point55": occupied >= 0.55,
            "free_occupied_gap_at_most_point35": abs(free - occupied) <= 0.35,
            "rough_balanced_at_least_max_point65_or_update100_plus_point02": (
                rough_ba
                >= max(
                    0.65,
                    _finite(
                        update_100.get("rough_raster_balanced_accuracy"),
                        name="update_100.rough_raster_balanced_accuracy",
                    )
                    + 0.02,
                )
            ),
            "rough_occupied_at_least_max_point50_or_update100_plus_point05": (
                rough_occupied
                >= max(
                    0.50,
                    _finite(
                        update_100.get("rough_raster_occupied_recall"),
                        name="update_100.rough_raster_occupied_recall",
                    )
                    + 0.05,
                )
            ),
            "paired_rgb_aggregate_margin_strictly_positive": margin > 0.0,
            "correct_rgb_scene_wins_at_least_7": _scene_wins(
                metrics.get("correct_rgb_scene_win_count")
            )
            >= 7,
        })
    else:
        if update_400 is None:
            raise ValueError("update_400 is required at update 1000")
        g = _finite(metrics.get("G"), name="G")
        nll = _finite(
            metrics.get("aggregate_raster_nll"), name="aggregate_raster_nll"
        )
        ba = _finite(
            metrics.get("aggregate_raster_balanced_accuracy"),
            name="aggregate_raster_balanced_accuracy",
        )
        free = _finite(
            metrics.get("aggregate_free_recall"), name="aggregate_free_recall"
        )
        occupied = _finite(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        unknown = _finite(
            metrics.get("aggregate_unknown_recall"),
            name="aggregate_unknown_recall",
        )
        rough_ba = _finite(
            metrics.get("rough_raster_balanced_accuracy"),
            name="rough_raster_balanced_accuracy",
        )
        rough_occupied = _finite(
            metrics.get("rough_raster_occupied_recall"),
            name="rough_raster_occupied_recall",
        )
        margin = _finite(
            metrics.get("paired_rgb_aggregate_margin"),
            name="paired_rgb_aggregate_margin",
        )
        conjuncts.update({
            "G_strictly_lower_than_update_400": (
                g < _finite(update_400.get("G"), name="update_400.G")
            ),
            "balanced_accuracy_at_least_max_point80_or_update400": (
                ba
                >= max(
                    0.80,
                    _finite(
                        update_400.get("aggregate_raster_balanced_accuracy"),
                        name="update_400.aggregate_raster_balanced_accuracy",
                    ),
                )
            ),
            "raster_nll_at_most_min_point42_or_update400": (
                nll
                <= min(
                    0.42,
                    _finite(
                        update_400.get("aggregate_raster_nll"),
                        name="update_400.aggregate_raster_nll",
                    ),
                )
            ),
            "free_recall_at_least_point68": free >= 0.68,
            "occupied_recall_at_least_point88": occupied >= 0.88,
            "unknown_recall_at_least_point80": unknown >= 0.80,
            "free_occupied_gap_at_most_point25": abs(free - occupied) <= 0.25,
            "rough_balanced_at_least_point772": rough_ba >= 0.772,
            "rough_occupied_at_least_point65": rough_occupied >= 0.65,
            "paired_rgb_aggregate_margin_strictly_positive": margin > 0.0,
            "correct_rgb_scene_wins_at_least_7": _scene_wins(
                metrics.get("correct_rgb_scene_win_count")
            )
            >= 7,
        })

    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": GATE_CONTROLS[update][1 if passed else 0],
        "gate_mode": "FINAL_SIGNED_BOUNDARY_DISTANCE_STATE_V1_RECEIPT",
        "signed_boundary_distance_mechanism_receipt_ready": True,
        "conjuncts": conjuncts,
        "thresholds": deepcopy(GATE_THRESHOLDS[update]),
        "perception_accounting": perception_accounting(update),
    }


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
        raise ValueError(
            "failure receipts are not one exact signed-boundary gate control"
        )
    return dict(value)


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
    **_binding_metadata: Any,
) -> dict[str, Any]:
    read = (
        _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
    raw = read(ROOT / relative_path)
    parsed = json.loads(raw)
    core = dict(parsed)
    declared = core.pop("content_sha256", None)
    nested_classification = None
    if type(parsed.get("scientific_result")) is dict:
        nested_classification = parsed["scientific_result"].get(
            "classification"
        )
    actual_classification = parsed.get("classification", nested_classification)
    if (
        len(raw) != byte_count
        or hashlib.sha256(raw).hexdigest() != file_sha256
        or declared != content_sha256
        or canonical_json_sha256(core) != content_sha256
        or parsed.get("status") != status
        or (
            classification is not None
            and actual_classification != classification
        )
    ):
        raise PermissionError(f"governing document changed: {relative_path}")
    return parsed


def validate_frozen_v12_source_closure(
    root: Path = ROOT,
) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("frozen V12 closure must use repository root")
    current = _V12.current_source_bindings(root)
    if (
        current.get(FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V12_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V12 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v12_source_closure(root)
    bindings = (
        frozen_v12_review_binding(),
        frozen_v12_authorization_binding(),
        v12_terminal_audit_binding(),
        multiprototype_terminal_audit_binding(),
        preregistration_binding(),
    )
    for binding in bindings:
        _read_bound_json(
            binding["path"],
            file_sha256=binding["file_sha256"],
            content_sha256=binding["content_sha256"],
            byte_count=binding["byte_count"],
            status=binding["status"],
            classification=binding.get("classification"),
        )
        current[binding["path"]] = binding["file_sha256"]
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="signed-boundary source manifest")
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
    core = dict(value)
    declared = core.pop("content_sha256", None)
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
        or value.get("source_paths") != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != 149
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("signed-boundary source manifest changed")
    safe = (
        _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        .safe_relative_source_path
    )
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path",
            "file_sha256",
            "byte_count",
        }:
            raise PermissionError("signed-boundary source binding fields changed")
        normalized.append(safe(binding["path"]))
        if (
            not is_sha256(binding.get("file_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("signed-boundary source binding changed")
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("signed-boundary source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("source closure must use repository root")
    read = (
        _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != binding["file_sha256"]
            or len(payload) != binding["byte_count"]
        ):
            raise PermissionError(
                f"manifest-bound source changed: {binding['path']}"
            )
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        read = (
            _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
            ._read_regular_source
        )
        raw = read(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        manifest = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(manifest["content_sha256"]),
        )
    return validate_binding(
        dict(source_manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )


def _source_freeze_commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{name} must be one exact 40-hex commit")
    return value


REVIEW_CHECKS = {
    "source_only_imports_pass": True,
    "focused_cpu_tests_pass": True,
    "full_recursive_cpu_tests_pass": True,
    "exactly_six_additive_sources_over_frozen_v12_143_sources": True,
    "multiprototype_sources_absent_from_active_import_graph": True,
    "model_bindings_frozen_and_exact": True,
    "half_cell_center_EDT_Huber_and_scale16_adapter_exact": True,
    "exact_target_argmax_semantics_pass": True,
    "preliminary_dispatch_is_nonauthoritative_at_all_four_updates": True,
    "update_zero_is_structural_only_with_direction_free_RGB_nonidentity": True,
    "all_four_final_gates_match_preregistration": True,
    "paired_RGB_control_is_inherited_V10_class_macro_NLL": True,
    "predictor_isolation_and_full_schedule_caps_exact": True,
    "source_freeze_commit_matches_reviewed_tree": True,
    "all_implementation_authors_excluded": True,
    "generated_or_protected_runtime_inputs_opened": [],
    "sealed_or_heldout_opened": [],
}


def _review_source_freeze_commit(
    review_binding: Mapping[str, Any],
) -> str:
    binding = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    read = (
        _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
    raw = read(ROOT / REVIEW_RELATIVE_PATH)
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError("signed-boundary source review binding changed")
    review = parse_canonical_json(raw, name="signed-boundary source review")
    core = dict(review)
    declared = core.pop("content_sha256", None)
    if (
        declared != binding["content_sha256"]
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("signed-boundary source review content changed")
    return _source_freeze_commit(
        review.get("source_freeze_commit"),
        name="review.source_freeze_commit",
    )


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "implementation_authors",
        "reviewer",
        "source_freeze_commit",
        "reviewed_sources",
        "source_manifest",
        "frozen_v12_source_manifest",
        "frozen_v12_source_review",
        "frozen_v12_execution_authorization",
        "v12_terminal_audit",
        "multiprototype_terminal_audit",
        "preregistration",
        "science_contract",
        "science_identity",
        "checks",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("signed-boundary source review fields changed")
    if (
        not MODEL_BINDINGS_FROZEN
        or not is_sha256(SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256)
        or MODEL_PARAMETER_INVENTORY.get("binding_status")
        == "UNBOUND_FAIL_CLOSED_PENDING_FINAL_MODEL_INVENTORY"
    ):
        raise PermissionError("final model bindings remain fail-closed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    reviewer = value["reviewer"]
    required = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    source_commit = _source_freeze_commit(
        value.get("source_freeze_commit"),
        name="review.source_freeze_commit",
    )
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != REVIEW_STATUS
        or value["implementation_authors"] != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or not required.issubset(expected_sources)
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"]
        != _manifest_binding_or_read(source_manifest_binding)
        or value["frozen_v12_source_manifest"]
        != frozen_v12_source_manifest_binding()
        or value["frozen_v12_source_review"] != frozen_v12_review_binding()
        or value["frozen_v12_execution_authorization"]
        != frozen_v12_authorization_binding()
        or value["v12_terminal_audit"] != v12_terminal_audit_binding()
        or value["multiprototype_terminal_audit"]
        != multiprototype_terminal_audit_binding()
        or value["preregistration"] != preregistration_binding()
        or value["science_contract"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["checks"] != REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not source_commit
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("signed-boundary source review did not pass")
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
        "source_freeze_commit",
        "independent_source_review",
        "frozen_v12_source_manifest",
        "frozen_v12_source_review",
        "frozen_v12_execution_authorization",
        "v12_terminal_audit",
        "multiprototype_terminal_audit",
        "preregistration",
        "runtime_inputs",
        "experiment",
        "science_identity",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("signed-boundary authorization fields changed")
    if not MODEL_BINDINGS_FROZEN:
        raise PermissionError("final model bindings remain fail-closed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    authorizer = value["authorizer"]
    expected_review = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_source_commit = _review_source_freeze_commit(expected_review)
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, reviewer}
        or value["source_freeze_commit"] != review_source_commit
        or value["independent_source_review"] != expected_review
        or value["frozen_v12_source_manifest"]
        != frozen_v12_source_manifest_binding()
        or value["frozen_v12_source_review"] != frozen_v12_review_binding()
        or value["frozen_v12_execution_authorization"]
        != frozen_v12_authorization_binding()
        or value["v12_terminal_audit"] != v12_terminal_audit_binding()
        or value["multiprototype_terminal_audit"]
        != multiprototype_terminal_audit_binding()
        or value["preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("signed-boundary execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V12.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v12_authorization_binding",
    "frozen_v12_review_binding",
    "frozen_v12_source_manifest_binding",
    "model_config",
    "multiprototype_terminal_audit_binding",
    "perception_accounting",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v12_terminal_audit_binding",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v12_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
