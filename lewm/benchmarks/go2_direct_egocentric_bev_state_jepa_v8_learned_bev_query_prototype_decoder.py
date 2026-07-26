"""Source-only contract for the Direct BEV V8 perception falsification.

V8 preserves the frozen V7 encoder, data, schedule prefix, objective, and
target-EMA science.  Its one registered scientific delta replaces the RGB
token-to-BEV decoder and linear state head with learned factorized BEV queries,
two cross-attention/FFN blocks, and three normalized state prototypes.  The
predictor is constructed as a frozen control but is never called or optimized.

Importing this module reads source and committed governance documents only.  It
grants no generated-input, checkpoint, tensor, GPU, training, navigation,
held-out, sealed, production, promotion, or deployment authority.
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
FROZEN_V7_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
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


_V7 = _source_only_module(
    "_lewm_direct_bev_v8_frozen_v7_contract",
    FROZEN_V7_CONTRACT_RELATIVE_PATH,
)

for _name in _V7.__all__:
    globals()[_name] = getattr(_V7, _name)

canonical_json_bytes = _V7.canonical_json_bytes
canonical_json_sha256 = _V7.canonical_json_sha256
is_sha256 = _V7.is_sha256
with_content_sha256 = _V7.with_content_sha256
parse_canonical_json = _V7.parse_canonical_json
artifact_binding = _V7.artifact_binding
validate_binding = _V7.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v8_contract_implementation"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
MODEL_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V7_RUNNER_RELATIVE_PATH = _V7.RUNNER_RELATIVE_PATH
FROZEN_V7_LAUNCHER_RELATIVE_PATH = _V7.LAUNCHER_RELATIVE_PATH
FROZEN_V7_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V7.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "a9b54a57c7ce7a29c898e941d2649ad6cfbd6e81"
PREREGISTRATION_FILE_SHA256 = (
    "22310aaaf69ee8544c9022fb337883ed98613e936f3f86d2d78bb49bbcd0c34e"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "065e0befce1c55b6545a442f26bd78d14dd035bc8e2e84eaa2af40bb53368d05"
)
PREREGISTRATION_BYTE_COUNT = 22_649
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_RGB_ONLY_LEARNED_BEV_QUERY_PROTOTYPE_"
    "PERCEPTION_FALSIFICATION_PENDING_SOURCE_FREEZE_INDEPENDENT_REVIEW_"
    "AND_MACHINE_AUTHORIZATION"
)

FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH = _V7.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V7_SOURCE_MANIFEST_COMMIT = (
    "ee13ea91cfbdfa8eae339b57b0e12e16ba7d4638"
)
FROZEN_V7_SOURCE_MANIFEST_FILE_SHA256 = (
    "175c66ac47ccdfe318ac27239b0091a6c694b0e0dc61fe174a138bc5d97c2a20"
)
FROZEN_V7_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "a84416e0e59dd8ced92f5475c1a2130e2909a26eb4678a6b3741ad61c7757e14"
)
FROZEN_V7_SOURCE_MANIFEST_BYTE_COUNT = 39_992
FROZEN_V7_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V7_SOURCE_COUNT = 116

FROZEN_V7_REVIEW_RELATIVE_PATH = _V7.REVIEW_RELATIVE_PATH
FROZEN_V7_REVIEW_COMMIT = "61714d05189740fac59d8204419ef6884bd70332"
FROZEN_V7_REVIEW_FILE_SHA256 = (
    "501e59905ab15414f9b9f1d88f1c556644c980eaf13ddeb9450c6a279b955739"
)
FROZEN_V7_REVIEW_CONTENT_SHA256 = (
    "d1b4a35488bc191e47186b46814349506068710b04649830d7d823b9b8b2d8c9"
)
FROZEN_V7_REVIEW_BYTE_COUNT = 58_296
FROZEN_V7_REVIEW_STATUS = (
    "PASS_SOURCE_SCIENCE_IDENTITY_AND_RUNNER_INTEGRITY"
)

FROZEN_V7_AUTHORIZATION_RELATIVE_PATH = _V7.AUTHORIZATION_RELATIVE_PATH
FROZEN_V7_AUTHORIZATION_COMMIT = (
    "2802fc416c09534b937592e3d9ac6712659f3079"
)
FROZEN_V7_AUTHORIZATION_FILE_SHA256 = (
    "3d294396645a5d409a74eacbd6afc40378ae554fd3452b56ff72e2ff7b6ce1b9"
)
FROZEN_V7_AUTHORIZATION_CONTENT_SHA256 = (
    "9ef30e19f771477c483862450cb89f788717cac3982faa63ad9266f8b1c8b0d8"
)
FROZEN_V7_AUTHORIZATION_BYTE_COUNT = 49_174
FROZEN_V7_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V7_RUNNER_INTEGRITY_REPLACEMENT"
)

V7_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_terminal_audit_2026-07-26.json"
)
V7_TERMINAL_AUDIT_COMMIT = "d5b1b1d3b3574d29193e0c675f6a69113faa661e"
V7_TERMINAL_AUDIT_FILE_SHA256 = (
    "7e458098ecacfffb275ed4adbbd799309ba4bd91ee44436d9573e5ade8571b44"
)
V7_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "28532867666b76655cdc7154273384f33a0f0d89f4fb6951b48fad016b194f15"
)
V7_TERMINAL_AUDIT_BYTE_COUNT = 14_413
V7_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_UPDATE_400_PERCEPTION_SCIENTIFIC_FAILURE_CLOSES_V7_NO_RETRY"
)
V7_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_400_PERCEPTION_BOUNDARY_SCIENTIFIC_FAILURE_PHASE_ONE_"
    "PERCEPTION_IMPROVED_BUT_STRONG_QUALIFICATION_NOT_MET_PREDICTOR_"
    "UNTRAINED_V7_PERMANENTLY_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_execution_authorization_2026-07-26.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V7.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 116 or len(SOURCE_PATHS) != 122:
    raise RuntimeError("V8 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V7_REVIEW_RELATIVE_PATH,
    FROZEN_V7_AUTHORIZATION_RELATIVE_PATH,
    V7_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v8/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v8_"
    "learned_bev_query_prototype_decoder_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V8_"
    "LEARNED_BEV_QUERY_PROTOTYPE_DECODER_PREFLIGHT_JSON"
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

REVIEW_STATUS = "PASS_SOURCE_AND_LEARNED_BEV_QUERY_PROTOTYPE_SCIENCE"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V8_LEARNED_BEV_QUERY_PROTOTYPE_"
    "PERCEPTION_FALSIFICATION"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V7.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v7_retry_resume_repair_or_extension_authorized": False,
    "v7_checkpoint_tensor_trace_receipt_or_runtime_output_reuse_authorized": (
        False
    ),
    "one_fresh_v8_learned_bev_query_prototype_perception_attempt_only": True,
    "predictor_training_or_evaluation_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
}


# Exact schedule prefix and hard caps.
MAXIMUM_ATTEMPTS = 1
ATTEMPT_INDEX = 1
MAXIMUM_UPDATES = 250
MAXIMUM_PRESENTATIONS = 4_000
GPU_ACTIVE_TIME_CAP_MINUTES = 30
EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
CHECKPOINT_UPDATES = (50, 100, 250)
SNAPSHOT_UPDATES = CHECKPOINT_UPDATES
OBSERVATION_UPDATES = (0, *CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_SHA256 = {
    50: "f7e06f741d96af1a3c7796096a38f616f40ee713b6258a217ffd5627afda0788",
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    250: "ee3bc0dcf4c36c8cc66daa2ea8cda6653072fb18c8cf6d6fe1fe3bb50ab1218e",
}
FROZEN_V7_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}


# V8 parameter counts and all newly drawn component states are source-verifiable.
# The encoder tensor bytes stay authority-gated until execution; its exact bound
# checkpoint is composed with these component witnesses and is rehashed by the
# runner before the complete runtime state hash is recorded.
FROZEN_V8_FRESH_DECODER_STATE_SHA256 = (
    "103e5f9436ef2293f64ffc224b393452180d34d794b83078d18e3b1cc6d353b6"
)
V8_INITIAL_DECODER_STATE_SHA256 = (
    "1567c7ebe049bb4046e800534e6fe00059414673388814407a605ebb04ba37e5"
)
V8_INITIAL_PROTOTYPE_HEAD_STATE_SHA256 = (
    "a35374e3704c356742ef6d08cdc8389e0f2af5c44f2d71aa2c2af5823e7ea1c3"
)
V8_INITIAL_PREDICTOR_STATE_SHA256 = (
    "ab62170c9b6c950ad0a21524abe3637d61cf51ff01d04e4bb4848818c144c8b3"
)
V8_FRESH_DRAW_ORDER = (
    "row_query",
    "column_query",
    "token_projection",
    "shared_projected_token_layer_norm",
    "cross_attention_block_1",
    "FFN_block_1",
    "cross_attention_block_2",
    "FFN_block_2",
    "UNKNOWN_FREE_OCCUPIED_prototypes",
)
V8_INITIAL_STATE_COMPONENT_BINDING = {
    "n320_encoder_checkpoint": dict(
        _V7.RUNTIME_BINDINGS[_V7.N320_CHECKPOINT_RELATIVE_PATH]
    ),
    "fresh_decoder_local_state_sha256": V8_INITIAL_DECODER_STATE_SHA256,
    "fresh_prototype_head_local_state_sha256": (
        V8_INITIAL_PROTOTYPE_HEAD_STATE_SHA256
    ),
    "fresh_predictor_normalized_state_sha256": (
        V8_INITIAL_PREDICTOR_STATE_SHA256
    ),
    "online_target_decoder_head_hard_sync_required": True,
    "complete_runtime_state_hash_recorded_after_authorized_encoder_load": True,
}
V8_INITIAL_STATE_COMPONENT_BINDING_SHA256 = canonical_json_sha256(
    V8_INITIAL_STATE_COMPONENT_BINDING
)
MODEL_PARAMETER_INVENTORY = {
    "encoder": dict(_V7.MODEL_PARAMETER_INVENTORY["encoder"]),
    "decoder_state": {
        "parameter_count": 87_808,
        "tensor_count": 31,
        "ordered_parameter_name_sha256": (
            "3d59a484e25593e47bc5eb740618814bf71a15f5e1e41e302598b774c49a0dc8"
        ),
    },
    "predictor": dict(_V7.MODEL_PARAMETER_INVENTORY["predictor"]),
    "detached_target_encoder_decoder_state": {
        "parameter_count": 2_835_328,
        "tensor_count": 109,
        "ordered_parameter_name_sha256": (
            "5a35dbd843a3ceda476ef4f885002df3d71a8dee380a324d9ddb95f37126aa22"
        ),
    },
    "total": {
        "parameter_count": 5_987_763,
        "tensor_count": 297,
    },
}
# Any accidentally reached V4/V6 full-state initializer stays fail-closed on
# the old V7 identity.  V8 owns a different initializer which validates the
# exact component binding above after the authorized N320 load.
FROZEN_V3_INITIAL_MODEL_STATE_SHA256 = _V7.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
PREDICTOR_CONFIG = deepcopy(_V7.PREDICTOR_CONFIG)

LEARNED_QUERY_PROTOTYPE_CONFIG = {
    "input": "256_RGB_patch_tokens_of_dimension_192",
    "output_shape": [64, 64, 64],
    "queries": {
        "formula": "row_query[row]+column_query[column]",
        "row_parameter_shape": [64, 64],
        "column_parameter_shape": [64, 64],
        "numeric_coordinate_features_or_sinusoids": False,
        "full_per_cell_query_parameter": False,
    },
    "token_projection": {
        "type": "Linear",
        "input_dimension": 192,
        "output_dimension": 64,
        "bias": True,
    },
    "shared_projected_token_layer_norm": {
        "dimension": 64,
        "affine": True,
    },
    "residual_cross_attention_ffn_blocks": {
        "count": 2,
        "parameters_shared": False,
        "attention_heads": 4,
        "attention_dropout": 0.0,
        "ffn_hidden_dimension": 128,
        "ffn_activation": "GELU",
        "linear_and_attention_biases": True,
        "bev_self_attention": False,
        "spatial_convolution": False,
        "formula": [
            "x=x+MHA(LN(x),shared_token_norm(projected_RGB_tokens),shared_token_norm(projected_RGB_tokens))",
            "x=x+Linear128to64(GELU(Linear64to128(LN(x))))",
        ],
    },
    "prototype_state_head": {
        "class_order": ["UNKNOWN", "FREE", "OCCUPIED"],
        "prototype_parameter_shape": [3, 64],
        "cell_and_prototype_l2_epsilon": 1e-12,
        "output_formula": (
            "negative_squared_distance_between_L2_normalized_cell_feature_"
            "and_L2_normalized_class_prototype"
        ),
        "output_shape": ["batch", 3, 64, 64],
        "output_logit_range_inclusive": [-4.0, 0.0],
        "temperature_scale_bias_linear_or_convolutional_head": False,
    },
    "online_decoder_parameter_count": 87_808,
    "online_decoder_parameter_tensor_count": 31,
}

SCIENTIFIC_DELTA = {
    "scientific_delta_count": 1,
    "scientific_delta_name": (
        "RGB_token_to_BEV_state_decoder_and_state_head_mechanism"
    ),
    "removed": [
        "global_numeric_coordinate_feature_MLP_plus_full_per_cell_query_bias",
        "single_global_cross_attention_block",
        "two_layer_spatial_Conv2d_refinement",
        "unbounded_linear_1x1_three_logit_state_head",
    ],
    "replacement": deepcopy(LEARNED_QUERY_PROTOTYPE_CONFIG),
    "model_data_seed_schedule_G_optimizer_EMA_metrics_and_custody_preserved": (
        True
    ),
    "predictor_constructed_fresh_frozen_not_called_not_optimized": True,
    "prior_checkpoint_tensor_trace_receipt_or_runtime_output_reuse": False,
}

PHASE_ACCOUNTING = {
    update: {
        "target_update_callback_count": update,
        "perception_optimizer_updates": update,
        "predictor_optimizer_updates": 0,
        "ema_arithmetic_updates": update,
        "boundary_hard_sync_count": 0,
        "phase_two_target_noop_count": 0,
        "presentations": update * EFFECTIVE_BATCH_SIZE,
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_membership_count": 0,
        "predictor_requires_grad_parameter_count": 0,
    }
    for update in OBSERVATION_UPDATES
}

GATE_THRESHOLDS = {
    0: {},
    50: {
        "G_strictly_less_than_update_zero": True,
        "aggregate_raster_balanced_accuracy_strictly_greater_than_update_zero": (
            True
        ),
        "aggregate_free_recall_strictly_greater_than": 0.0,
        "aggregate_occupied_recall_strictly_greater_than": 0.0,
        "rough_raster_balanced_accuracy_strictly_greater_than": 0.0,
        "rough_raster_occupied_recall_strictly_greater_than": 0.0,
        "correct_rgb_scene_win_count_minimum_inclusive": 6,
    },
    100: {
        "G_strictly_less_than_update_zero": True,
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.70,
        "aggregate_free_recall_minimum_inclusive": 0.50,
        "aggregate_occupied_recall_minimum_inclusive": 0.80,
        "aggregate_raster_nll_maximum_inclusive": 0.46,
        "rough_raster_balanced_accuracy_strictly_greater_than_update_zero": (
            True
        ),
        "rough_raster_occupied_recall_strictly_greater_than_update_zero": True,
        "correct_rgb_scene_win_count_minimum_inclusive": 8,
    },
    250: {
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.80,
        "aggregate_free_recall_minimum_inclusive": 0.68,
        "aggregate_occupied_recall_minimum_inclusive": 0.88,
        "absolute_free_minus_occupied_recall_gap_maximum_inclusive": 0.25,
        "aggregate_raster_nll_maximum_inclusive": 0.42,
        "aggregate_raster_nll_update100_addend_maximum_inclusive": 0.01,
        "rough_raster_balanced_accuracy_minimum_inclusive": 0.7719525,
        "rough_raster_occupied_recall_minimum_inclusive": 0.4319467,
        "correct_rgb_scene_win_count_minimum_inclusive": 8,
    },
}

CONTROL_UPDATE_ZERO_FAIL = (
    "FAIL_UPDATE_ZERO_V8_MECHANISM_INTEGRITY_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_ZERO = "CONTINUE_AFTER_UPDATE_ZERO_V8_INTEGRITY_GATE"
CONTROL_UPDATE_50_FAIL = (
    "FAIL_UPDATE_50_V8_PERCEPTION_HEALTH_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_50 = "CONTINUE_AFTER_UPDATE_50_V8_HEALTH_GATE"
CONTROL_UPDATE_100_FAIL = (
    "FAIL_UPDATE_100_V8_PERCEPTION_CONTINUATION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_100 = "CONTINUE_AFTER_UPDATE_100_V8_PERCEPTION_GATE"
CONTROL_UPDATE_250_FAIL = (
    "FAIL_UPDATE_250_V8_PERCEPTION_QUALIFICATION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_PASS = (
    "PASS_DIRECT_BEV_V8_LEARNED_BEV_QUERY_PROTOTYPE_PERCEPTION_MECHANISM_ONLY"
)
CONTROL_PRELIMINARY = (
    "PRELIMINARY_PASS_SOURCE_INTEGRATION_NOT_V8_MECHANISM_EVIDENCE"
)
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_ZERO_FAIL, CONTROL_CONTINUE_UPDATE_ZERO),
    50: (CONTROL_UPDATE_50_FAIL, CONTROL_CONTINUE_UPDATE_50),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    250: (CONTROL_UPDATE_250_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v7_manifest_and_all_116_sources_rehashed": True,
    "frozen_v7_review_authorization_and_terminal_audit_exact": True,
    "v7_permanently_closed_and_runtime_reuse_forbidden": True,
    "v8_preregistration_exact": True,
    "sole_scientific_delta_is_decoder_and_prototype_state_head": True,
    "n320_encoder_data_seed_schedule_prefix_G_optimizer_and_EMA_preserved": True,
    "learned_queries_cross_attention_ffns_and_prototypes_exact": True,
    "numeric_geometry_full_cell_bias_spatial_convolution_and_linear_head_absent": (
        True
    ),
    "predictor_frozen_excluded_and_zero_forward_objective_backward_update": True,
    "u0_u50_u100_u250_gates_and_accounting_fail_closed": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v7_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V7_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V7_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V7_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V7_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V7_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V7_SOURCE_COUNT,
    }


def frozen_v7_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V7_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V7_REVIEW_COMMIT,
        "file_sha256": FROZEN_V7_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V7_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V7_REVIEW_BYTE_COUNT,
        "status": FROZEN_V7_REVIEW_STATUS,
    }


def frozen_v7_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V7_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V7_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V7_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V7_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V7_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V7_AUTHORIZATION_STATUS,
    }


def v7_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V7_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V7_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V7_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V7_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V7_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V7_TERMINAL_AUDIT_STATUS,
        "classification": V7_TERMINAL_AUDIT_CLASSIFICATION,
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


def model_config() -> dict[str, Any]:
    value = _V7.model_config()
    value["bev_decoder"] = deepcopy(LEARNED_QUERY_PROTOTYPE_CONFIG)
    value["state_head"] = {
        "integrated_into_decoder": True,
        **deepcopy(LEARNED_QUERY_PROTOTYPE_CONFIG["prototype_state_head"]),
        "sole_three_channel_state_bottleneck": True,
        "hidden_or_auxiliary_bypass_authorized": False,
    }
    value["transition"] = {
        **deepcopy(PREDICTOR_CONFIG),
        "constructed_fresh": True,
        "requires_grad": False,
        "optimizer_membership": False,
        "forward_objective_backward_and_update_count_required": 0,
        "action_metric_or_gate_authorized": False,
    }
    value["target"] = {
        "inventory": ["encoder", "bev_decoder"],
        "fresh_hard_sync_count_before_update_zero": 1,
        "ema_decay": TARGET_EMA_MOMENTUM,
        "ema_updates": "once_after_every_perception_optimizer_update",
        "predictor_target_copy": False,
        "gradient": "none",
    }
    value["initialization"] = {
        "n320_encoder_only_migration": True,
        "n320_fit_seed": N320_FIT_SEED,
        "fresh_parameter_seed": BASE_INITIALIZATION_SEED,
        "fresh_v8_draw_order": [
            "row_query",
            "column_query",
            "token_projection",
            "shared_projected_token_layer_norm",
            "cross_attention_block_1",
            "FFN_block_1",
            "cross_attention_block_2",
            "FFN_block_2",
            "UNKNOWN_FREE_OCCUPIED_prototypes",
        ],
        "predictor_constructed_fresh_then_frozen": True,
        "prior_v1_through_v7_runtime_reuse": False,
        "initial_state_component_binding": deepcopy(
            V8_INITIAL_STATE_COMPONENT_BINDING
        ),
        "initial_state_component_binding_sha256": (
            V8_INITIAL_STATE_COMPONENT_BINDING_SHA256
        ),
    }
    value["call_graph"] = {
        "online_current_and_next_RGB": (
            "weight_shared_encoder_learned_query_prototype_decoder_for_G"
        ),
        "online_fixed_negative_RGB": "grounding_observation_only_no_gradient",
        "target_perception": "fresh_detached_EMA_no_gradient",
        "predictor": "zero_forward_calls",
        "forbidden_inputs": [
            "pose", "odometry", "depth", "ray", "flow", "metric_geometry",
            "analytic_projection", "warp", "raster_labels_to_encoder",
        ],
    }
    value["parameter_inventory"] = deepcopy(MODEL_PARAMETER_INVENTORY)
    value.pop("optimization_phase_adapter", None)
    return value


def objective_contract() -> dict[str, Any]:
    value = _V7.objective_contract()
    value["perception_only_total"] = "G/log(2)"
    value["total"] = "G/log(2)"
    value["G_formula_and_hierarchical_three_state_energy"] = "exact_frozen_V7"
    value["predictor_J_C_or_v5_A_in_total"] = False
    value["auxiliary_loss"] = None
    value["class_weighting"] = None
    value["temperature"] = None
    value.pop("phase_one_total", None)
    value.pop("phase_two_total", None)
    return value


def optimizer_contract() -> dict[str, Any]:
    return {
        "name": "AdamW",
        "precision": "float32",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "learning_rates": {
            "encoder": 1e-4,
            "decoder_state": 3e-4,
        },
        "gradient_clipping": {
            "encoder_decoder_state_joint_norm": 1.0,
        },
        "constructed_once": True,
        "reset_or_rebuild": False,
        "predictor_parameters_excluded": True,
        "target_parameters_excluded": True,
    }


def build_schedule_identity() -> dict[str, Any]:
    value = _V7.build_schedule_identity()
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
            "exact_first_4000_presentation_entries_of_frozen_V7_schedule_"
            "without_filter_reorder_resample_or_replacement"
        ),
    })
    return value


def runtime_authorization_template() -> dict[str, Any]:
    value = _V7.runtime_authorization_template()
    value["schedule"] = build_schedule_identity()
    value["experiment_scope"] = {
        "one_fresh_attempt": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "maximum_active_gpu_minutes": GPU_ACTIVE_TIME_CAP_MINUTES,
        "perception_only": True,
        "predictor_forward_or_training": False,
        "prior_runtime_or_checkpoint_reuse": False,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    return value


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be one finite number")
    return float(value)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one nonnegative int")
    return value


def _accounting_conjuncts(
    update: int, metrics: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        f"{field}_equals_{expected}": (
            _exact_int(metrics.get(field), name=field) == expected
        )
        for field, expected in PHASE_ACCOUNTING[update].items()
    }


def _common_observation_conjuncts(
    metrics: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "all_registered_values_finite": _exact_bool(
            metrics.get("all_registered_values_finite"),
            name="all_registered_values_finite",
        ),
        "state_nonconstant": _exact_bool(
            metrics.get("state_nonconstant"), name="state_nonconstant"
        ),
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    update_100: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate preliminary integration mode or exact fail-closed V8 gates."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 50, 100, or 250")
    _exact_bool(prior_gates_passed, name="prior_gates_passed")
    if "v8_mechanism_receipt_ready" not in metrics:
        return {
            "update": update,
            "passed": True,
            "control": CONTROL_PRELIMINARY,
            "gate_mode": (
                "PRELIMINARY_SOURCE_INTEGRATION_NOT_V8_MECHANISM_EVIDENCE"
            ),
            "v8_mechanism_receipt_ready": False,
            "conjuncts": {"preliminary_integration_only": True},
            "thresholds": dict(GATE_THRESHOLDS[update]),
            "phase_accounting": dict(PHASE_ACCOUNTING[update]),
        }

    ready = _exact_bool(
        metrics.get("v8_mechanism_receipt_ready"),
        name="v8_mechanism_receipt_ready",
    )
    conjuncts: dict[str, bool] = {
        "prior_gates_passed": prior_gates_passed,
        "v8_mechanism_receipt_ready": ready,
        "active_phase_is_perception_only": (
            metrics.get("active_phase_v6") == "phase_one"
        ),
    }
    conjuncts.update(_accounting_conjuncts(update, metrics))
    conjuncts.update(_common_observation_conjuncts(metrics))

    if update == 0:
        for field in (
            "fresh_v8_model_and_optimizer_zero_prior_runtime_reuse",
            "n320_encoder_only_migration_exact",
            "registered_seed_draw_order_exact",
            "initial_model_state_matches_frozen_v8",
            "model_parameter_inventory_exact",
            "v8_decoder_parameter_inventory_exact",
            "learned_only_forbidden_geometry_absent",
            "two_residual_cross_attention_ffn_blocks_exact",
            "negative_squared_prototype_distance_formula_exact",
            "online_target_perception_bitwise_equal",
            "three_channel_state_exact",
            "all_logits_in_closed_interval_minus4_to0",
            "v8_intended_gradient_coverage_exact",
            "predictor_target_and_fixed_negative_gradients_absent",
            "no_hidden_auxiliary_bypass",
            "all_forbidden_access_counts_zero",
        ):
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        conjuncts["initial_online_to_target_hard_sync_count_equals_1"] = (
            _exact_int(
                metrics.get("initial_online_to_target_hard_sync_count"),
                name="initial_online_to_target_hard_sync_count",
            ) == 1
        )
    elif update == 50:
        if update_zero is None:
            raise ValueError("update_zero baseline is required at update 50")
        conjuncts.update({
            "G_strictly_lower_than_update_zero": (
                _finite_number(metrics.get("G"), name="G")
                < _finite_number(update_zero.get("G"), name="update_zero.G")
            ),
            "aggregate_raster_balanced_accuracy_strictly_higher_than_update_zero": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) > _finite_number(
                    update_zero.get("aggregate_raster_balanced_accuracy"),
                    name="update_zero.aggregate_raster_balanced_accuracy",
                )
            ),
            "aggregate_free_recall_above_zero": (
                _finite_number(
                    metrics.get("aggregate_free_recall"),
                    name="aggregate_free_recall",
                ) > 0.0
            ),
            "aggregate_occupied_recall_above_zero": (
                _finite_number(
                    metrics.get("aggregate_occupied_recall"),
                    name="aggregate_occupied_recall",
                ) > 0.0
            ),
            "rough_raster_balanced_accuracy_above_zero": (
                _finite_number(
                    metrics.get("rough_raster_balanced_accuracy"),
                    name="rough_raster_balanced_accuracy",
                ) > 0.0
            ),
            "rough_raster_occupied_recall_above_zero": (
                _finite_number(
                    metrics.get("rough_raster_occupied_recall"),
                    name="rough_raster_occupied_recall",
                ) > 0.0
            ),
            "correct_rgb_wins_at_least_six_scenes": (
                _exact_int(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                ) >= 6
            ),
        })
    elif update == 100:
        if update_zero is None:
            raise ValueError("update_zero baseline is required at update 100")
        conjuncts.update({
            "G_strictly_lower_than_update_zero": (
                _finite_number(metrics.get("G"), name="G")
                < _finite_number(update_zero.get("G"), name="update_zero.G")
            ),
            "aggregate_raster_balanced_accuracy_at_least_point70": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.70
            ),
            "aggregate_free_recall_at_least_point50": (
                _finite_number(
                    metrics.get("aggregate_free_recall"),
                    name="aggregate_free_recall",
                ) >= 0.50
            ),
            "aggregate_occupied_recall_at_least_point80": (
                _finite_number(
                    metrics.get("aggregate_occupied_recall"),
                    name="aggregate_occupied_recall",
                ) >= 0.80
            ),
            "aggregate_raster_nll_at_most_point46": (
                _finite_number(
                    metrics.get("aggregate_raster_nll"),
                    name="aggregate_raster_nll",
                ) <= 0.46
            ),
            "rough_raster_balanced_accuracy_higher_than_update_zero": (
                _finite_number(
                    metrics.get("rough_raster_balanced_accuracy"),
                    name="rough_raster_balanced_accuracy",
                ) > _finite_number(
                    update_zero.get("rough_raster_balanced_accuracy"),
                    name="update_zero.rough_raster_balanced_accuracy",
                )
            ),
            "rough_raster_occupied_recall_higher_than_update_zero": (
                _finite_number(
                    metrics.get("rough_raster_occupied_recall"),
                    name="rough_raster_occupied_recall",
                ) > _finite_number(
                    update_zero.get("rough_raster_occupied_recall"),
                    name="update_zero.rough_raster_occupied_recall",
                )
            ),
            "correct_rgb_wins_all_eight_scenes": (
                _exact_int(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                ) >= 8
            ),
        })
    else:
        if update_100 is None:
            raise ValueError("update_100 baseline is required at update 250")
        free_recall = _finite_number(
            metrics.get("aggregate_free_recall"),
            name="aggregate_free_recall",
        )
        occupied_recall = _finite_number(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        nll = _finite_number(
            metrics.get("aggregate_raster_nll"),
            name="aggregate_raster_nll",
        )
        conjuncts.update({
            "aggregate_raster_balanced_accuracy_at_least_point80": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.80
            ),
            "aggregate_free_recall_at_least_point68": free_recall >= 0.68,
            "aggregate_occupied_recall_at_least_point88": (
                occupied_recall >= 0.88
            ),
            "absolute_free_occupied_recall_gap_at_most_point25": (
                abs(free_recall - occupied_recall) <= 0.25
            ),
            "aggregate_raster_nll_at_most_point42": nll <= 0.42,
            "aggregate_raster_nll_at_most_update100_plus_point01": (
                nll <= _finite_number(
                    update_100.get("aggregate_raster_nll"),
                    name="update_100.aggregate_raster_nll",
                ) + 0.01
            ),
            "rough_raster_balanced_accuracy_at_least_registered_floor": (
                _finite_number(
                    metrics.get("rough_raster_balanced_accuracy"),
                    name="rough_raster_balanced_accuracy",
                ) >= 0.7719525
            ),
            "rough_raster_occupied_recall_at_least_registered_floor": (
                _finite_number(
                    metrics.get("rough_raster_occupied_recall"),
                    name="rough_raster_occupied_recall",
                ) >= 0.4319467
            ),
            "correct_rgb_wins_all_eight_scenes": (
                _exact_int(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                ) >= 8
            ),
        })

    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": GATE_CONTROLS[update][1 if passed else 0],
        "gate_mode": "FINAL_V8_LEARNED_QUERY_PROTOTYPE_MECHANISM_RECEIPT",
        "v8_mechanism_receipt_ready": ready,
        "conjuncts": conjuncts,
        "thresholds": dict(GATE_THRESHOLDS[update]),
        "phase_accounting": dict(PHASE_ACCOUNTING[update]),
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
        raise ValueError("failure receipts are not one exact V8 gate control")
    return dict(value)


def science_contract() -> dict[str, Any]:
    value = _V7.science_contract()
    frozen_v7_integrity_provenance = value.pop("integrity_replacement", None)
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Can fully learned factorized BEV queries and normalized class prototypes "
        "reduce V7's occupied-cell bias while preserving useful RGB-grounded "
        "maze geometry?"
    )
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v7_source_manifest": frozen_v7_source_manifest_binding(),
        "frozen_v7_source_review": frozen_v7_review_binding(),
        "frozen_v7_execution_authorization": frozen_v7_authorization_binding(),
        "v7_terminal_audit": v7_terminal_audit_binding(),
        "v8_preregistration": preregistration_binding(),
    }
    value["model"] = model_config()
    value["objective"] = objective_contract()
    value["optimizer"] = optimizer_contract()
    value["schedule"] = build_schedule_identity()
    value["gates"] = {
        "updates": list(OBSERVATION_UPDATES),
        "thresholds": {
            str(update): dict(GATE_THRESHOLDS[update])
            for update in OBSERVATION_UPDATES
        },
        "controls": {
            str(update): list(GATE_CONTROLS[update])
            for update in OBSERVATION_UPDATES
        },
        "phase_accounting": {
            str(update): dict(PHASE_ACCOUNTING[update])
            for update in OBSERVATION_UPDATES
        },
        "preliminary_mode_authorizes_execution": False,
        "final_mode_requires_v8_mechanism_receipt_ready": True,
        "stop_at_first_failed_gate": True,
    }
    lifecycle = dict(value["lifecycle"])
    value["lifecycle"] = {
        **lifecycle,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scientific_successor_of": _V7.EXPERIMENT_ID,
        "one_fresh_attempt": True,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "maximum_active_gpu_minutes": GPU_ACTIVE_TIME_CAP_MINUTES,
        "predictor_phase_or_update": False,
        "v7_retry_resume_repair_or_extension": False,
        "v7_checkpoint_tensor_trace_receipt_or_runtime_output_reuse": False,
        "retry_resume_extension_second_seed_or_replacement_attempt": False,
    }
    value["frozen_v7_integrity_provenance"] = {
        "scope": "historical_v6_to_v7_runner_integrity_replacement_only",
        "not_a_v8_unchanged_architecture_claim": True,
        "v6_to_v7": frozen_v7_integrity_provenance,
    }
    value["scientific_delta"] = deepcopy(SCIENTIFIC_DELTA)
    value["authority"] = {
        **value["authority"],
        "v8_execution_authorized_by_source_contract": False,
        "predictor_training_or_evaluation_authorized": False,
        "v7_checkpoint_or_runtime_output_reuse_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    return value


def science_identity_receipt() -> dict[str, Any]:
    return {
        "frozen_v7_science_contract_sha256": canonical_json_sha256(
            _V7.science_contract()
        ),
        "v8_science_contract_sha256": canonical_json_sha256(science_contract()),
        "scientific_delta_count": 1,
        "scientific_delta_name": SCIENTIFIC_DELTA["scientific_delta_name"],
        "v7_runtime_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
) -> dict[str, Any]:
    read = _V7._V6._v5._v4._v3._v2._v1._read_regular_source
    raw = read(ROOT / relative_path)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != file_sha256:
        raise PermissionError(f"governing document changed: {relative_path}")
    value = json.loads(raw)
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        declared != content_sha256
        or canonical_json_sha256(core) != content_sha256
        or value.get("status") != status
        or (
            classification is not None
            and value.get("classification") != classification
        )
    ):
        raise PermissionError(f"governing conclusion changed: {relative_path}")
    return dict(value)


def validate_frozen_v7_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V8 frozen V7 closure must use repository root")
    manifest = _read_bound_json(
        FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH,
        file_sha256=FROZEN_V7_SOURCE_MANIFEST_FILE_SHA256,
        content_sha256=FROZEN_V7_SOURCE_MANIFEST_CONTENT_SHA256,
        byte_count=FROZEN_V7_SOURCE_MANIFEST_BYTE_COUNT,
        status=FROZEN_V7_SOURCE_MANIFEST_STATUS,
    )
    if manifest.get("source_count") != FROZEN_V7_SOURCE_COUNT:
        raise PermissionError("frozen V7 source count changed")
    current = _V7.current_source_bindings(root)
    if current.get(FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V7_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V7 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v7_source_closure(root)
    review = _read_bound_json(
        FROZEN_V7_REVIEW_RELATIVE_PATH,
        file_sha256=FROZEN_V7_REVIEW_FILE_SHA256,
        content_sha256=FROZEN_V7_REVIEW_CONTENT_SHA256,
        byte_count=FROZEN_V7_REVIEW_BYTE_COUNT,
        status=FROZEN_V7_REVIEW_STATUS,
    )
    authorization = _read_bound_json(
        FROZEN_V7_AUTHORIZATION_RELATIVE_PATH,
        file_sha256=FROZEN_V7_AUTHORIZATION_FILE_SHA256,
        content_sha256=FROZEN_V7_AUTHORIZATION_CONTENT_SHA256,
        byte_count=FROZEN_V7_AUTHORIZATION_BYTE_COUNT,
        status=FROZEN_V7_AUTHORIZATION_STATUS,
    )
    _V7.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _V7.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    _read_bound_json(
        V7_TERMINAL_AUDIT_RELATIVE_PATH,
        file_sha256=V7_TERMINAL_AUDIT_FILE_SHA256,
        content_sha256=V7_TERMINAL_AUDIT_CONTENT_SHA256,
        byte_count=V7_TERMINAL_AUDIT_BYTE_COUNT,
        status=V7_TERMINAL_AUDIT_STATUS,
        classification=V7_TERMINAL_AUDIT_CLASSIFICATION,
    )
    _read_bound_json(
        PREREGISTRATION_RELATIVE_PATH,
        file_sha256=PREREGISTRATION_FILE_SHA256,
        content_sha256=PREREGISTRATION_CONTENT_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        status=PREREGISTRATION_STATUS,
    )
    current.update({
        FROZEN_V7_SOURCE_MANIFEST_RELATIVE_PATH: (
            FROZEN_V7_SOURCE_MANIFEST_FILE_SHA256
        ),
        FROZEN_V7_REVIEW_RELATIVE_PATH: FROZEN_V7_REVIEW_FILE_SHA256,
        FROZEN_V7_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V7_AUTHORIZATION_FILE_SHA256
        ),
        V7_TERMINAL_AUDIT_RELATIVE_PATH: V7_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V8 source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
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
        or value.get("source_count") != len(SOURCE_PATHS)
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
        raise PermissionError("V8 source manifest contract changed")
    safe = _V7._V6._v5._v4._v3._v2._v1.safe_relative_source_path
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V8 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V8 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V8 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V7._V6._v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V8 source changed: {binding['path']}")
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
        read = _V7._V6._v5._v4._v3._v2._v1._read_regular_source
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


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "source_manifest", "frozen_v7_source_manifest",
        "frozen_v7_source_review", "frozen_v7_execution_authorization",
        "v7_terminal_audit", "v8_preregistration", "experiment",
        "science_identity", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V8 source review fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    reviewer = value["reviewer"]
    required = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != REVIEW_STATUS
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not required.issubset(expected_sources)
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"]
        != _manifest_binding_or_read(source_manifest_binding)
        or value["frozen_v7_source_manifest"]
        != frozen_v7_source_manifest_binding()
        or value["frozen_v7_source_review"] != frozen_v7_review_binding()
        or value["frozen_v7_execution_authorization"]
        != frozen_v7_authorization_binding()
        or value["v7_terminal_audit"] != v7_terminal_audit_binding()
        or value["v8_preregistration"] != preregistration_binding()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "synthetic_cpu_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V8 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v7_source_manifest", "frozen_v7_source_review",
        "frozen_v7_execution_authorization", "v7_terminal_audit",
        "v8_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V8 execution authorization fields changed")
    expected_review = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    core = dict(value)
    declared = core.pop("content_sha256", None)
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["frozen_v7_source_manifest"]
        != frozen_v7_source_manifest_binding()
        or value["frozen_v7_source_review"] != frozen_v7_review_binding()
        or value["frozen_v7_execution_authorization"]
        != frozen_v7_authorization_binding()
        or value["v7_terminal_audit"] != v7_terminal_audit_binding()
        or value["v8_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V8 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V7.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v7_authorization_binding",
    "frozen_v7_review_binding",
    "frozen_v7_source_manifest_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v7_terminal_audit_binding",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v7_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
