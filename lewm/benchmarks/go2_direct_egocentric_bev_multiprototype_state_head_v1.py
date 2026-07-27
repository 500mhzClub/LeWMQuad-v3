"""Contract for one RGB-only Direct-BEV multiprototype-head probe.

The frozen V12 execution/data stack is reused as source.  The sole trainable
mechanism change is a three-class, four-prototype-per-class state head with an
equal-weight log-mean-exp readout.  Importing this module grants no runtime
authority and imports no tensor or image runtime.
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
    "_lewm_direct_bev_multiprototype_frozen_v12_contract",
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

IMPLEMENTATION_AUTHOR = "/root/v9_preregistration_author"
IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/multiprototype_test_author",
    "/root/v9_preregistration_author",
    "/root/v9_wrapper_implementation",
)
SCHEMA_PREFIX = "lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1"
EXPERIMENT_ID = "go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_multiprototype_state_head_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_multiprototype_state_head_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_multiprototype_state_head_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_multiprototype_state_head_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_multiprototype_state_head_v1_"
    "source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_multiprototype_state_head_v1.py"
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
FROZEN_V9_LAUNCHER_RELATIVE_PATH = _V12._V11._V10.FROZEN_V9_LAUNCHER_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "843e3fda5ff6b7e565e526f3d3c7e71c9cb0db7b"
PREREGISTRATION_FILE_SHA256 = (
    "9f68a8484760358d905109ed437af311ce131c571576ede61ea8a39cc0df382f"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "80c36db61a30efdc77319fca21ba72c6f8334410e58f593d8e6ad11cd6501f6a"
)
PREREGISTRATION_BYTE_COUNT = 28_932
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_RGB_ONLY_DIRECT_BEV_MULTIPROTOTYPE_STATE_HEAD_"
    "V1_PERCEPTION_FALSIFICATION_PENDING_SOURCE_FREEZE_INDEPENDENT_REVIEW_"
    "AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH = _V12.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V12_SOURCE_MANIFEST_COMMIT = "3673812bb325b867c1654e191d4487b5d0cfa600"
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
FROZEN_V12_AUTHORIZATION_COMMIT = "466b0ff8cfb403350eacc0d2d93cc97489bb897b"
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

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_multiprototype_state_head_v1_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V12.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 143 or len(SOURCE_PATHS) != 149:
    raise RuntimeError("multiprototype recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V12_REVIEW_RELATIVE_PATH,
    FROZEN_V12_AUTHORIZATION_RELATIVE_PATH,
    V12_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_multiprototype_state_"
    "head_v1/rgb_direct_egocentric_bev_multiprototype_state_head_probe_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_MULTIPROTOTYPE_STATE_HEAD_V1_PREFLIGHT_JSON"
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

REVIEW_STATUS = "PASS_SOURCE_MULTIPROTOTYPE_STATE_HEAD_V1_SCIENCE_AND_CUSTODY"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_MULTIPROTOTYPE_STATE_HEAD_V1_"
    "PERCEPTION_FALSIFICATION"
)
SOURCE_ONLY_AUTHORITY = dict(_V12.SOURCE_ONLY_AUTHORITY)
REVIEW_AUTHORITY = dict(_V12.REVIEW_AUTHORITY)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

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
SCHEDULE_PREFIX_SHA256 = dict(_V12.SCHEDULE_PREFIX_SHA256)

MULTIPROTOTYPE_INITIAL_HEAD_STATE_SHA256 = (
    "6ae170d9836cd1d32e81ff3178d1fa651e434d8408953f1da4096d8f5d06d436"
)
V8_INITIAL_DECODER_STATE_SHA256 = _V12.V8_INITIAL_DECODER_STATE_SHA256
V8_INITIAL_PROTOTYPE_HEAD_STATE_SHA256 = MULTIPROTOTYPE_INITIAL_HEAD_STATE_SHA256
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
    "UNKNOWN_FREE_OCCUPIED_four_prototypes_each",
)

MODEL_PARAMETER_INVENTORY = deepcopy(_V12.MODEL_PARAMETER_INVENTORY)
MODEL_PARAMETER_INVENTORY["decoder_state"] = {
    **MODEL_PARAMETER_INVENTORY["decoder_state"],
    "parameter_count": 88_384,
}
MODEL_PARAMETER_INVENTORY["detached_target_encoder_decoder_state"] = {
    **MODEL_PARAMETER_INVENTORY["detached_target_encoder_decoder_state"],
    "parameter_count": 2_835_904,
}
MODEL_PARAMETER_INVENTORY["total"] = {
    "parameter_count": 5_988_915,
    "tensor_count": 297,
}

CONTROL_UPDATE_0_FAIL = "FAIL_UPDATE_ZERO_MULTIPROTOTYPE_V1_INTEGRITY_GATE_TERMINAL_NO_RETRY"
CONTROL_CONTINUE_UPDATE_0 = "CONTINUE_AFTER_UPDATE_ZERO_MULTIPROTOTYPE_V1_INTEGRITY_GATE"
CONTROL_UPDATE_50_FAIL = "FAIL_UPDATE_50_MULTIPROTOTYPE_V1_DIRECTIONAL_GATE_TERMINAL_NO_RETRY"
CONTROL_CONTINUE_UPDATE_50 = "CONTINUE_AFTER_UPDATE_50_MULTIPROTOTYPE_V1_DIRECTIONAL_GATE"
CONTROL_UPDATE_100_FAIL = "FAIL_UPDATE_100_MULTIPROTOTYPE_V1_DECISIVE_GATE_TERMINAL_NO_RETRY"
CONTROL_CONTINUE_UPDATE_100 = "CONTINUE_AFTER_UPDATE_100_MULTIPROTOTYPE_V1_DECISIVE_GATE"
CONTROL_UPDATE_250_FAIL = "FAIL_UPDATE_250_MULTIPROTOTYPE_V1_QUALIFICATION_GATE_TERMINAL_NO_RETRY"
CONTROL_PASS = "PASS_DIRECT_BEV_MULTIPROTOTYPE_STATE_HEAD_V1_PERCEPTION_MECHANISM_ONLY"
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_0_FAIL, CONTROL_CONTINUE_UPDATE_0),
    50: (CONTROL_UPDATE_50_FAIL, CONTROL_CONTINUE_UPDATE_50),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    250: (CONTROL_UPDATE_250_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())
GATE_THRESHOLDS = {
    0: {"integrity_only": True},
    50: {
        "G_and_nll_strictly_below_update_zero": True,
        "balanced_occupied_rough_balanced_and_rough_occupied_strictly_above_update_zero": True,
        "free_recall_minimum": 0.25,
        "free_occupied_gap_maximum": 0.60,
        "correct_rgb_scene_wins": 8,
    },
    100: {
        "G_strictly_below_update_zero": True,
        "balanced_accuracy_minimum": 0.72,
        "free_recall_minimum": 0.68,
        "occupied_recall_minimum": 0.80,
        "free_occupied_gap_maximum": 0.20,
        "raster_nll_maximum": 0.46,
        "rough_balanced_strictly_above_v12_u100": 0.732972219013282,
        "rough_occupied_strictly_above_v12_u100": 0.5722940226171244,
        "correct_rgb_scene_wins": 8,
    },
    250: {
        "balanced_accuracy_minimum": 0.80,
        "free_recall_minimum": 0.68,
        "occupied_recall_minimum": 0.88,
        "free_occupied_gap_maximum": 0.25,
        "raster_nll_maximum": 0.42,
        "raster_nll_maximum_above_update100": 0.01,
        "rough_balanced_minimum": 0.7719525,
        "rough_occupied_at_least_update100": True,
        "correct_rgb_scene_wins": 8,
    },
}

INTEGRITY_FIELDS = (
    "fresh_multiprototype_model_and_optimizer_zero_prior_runtime_reuse",
    "frozen_encoder_decoder_predictor_initialization_exact",
    "registered_seed_draw_order_exact",
    "multiprototype_initial_head_state_sha256_exact",
    "model_parameter_inventory_exact",
    "multiprototype_decoder_parameter_inventory_exact",
    "learned_only_forbidden_geometry_absent",
    "two_residual_cross_attention_ffn_blocks_exact",
    "multiprototype_shape_formula_axes_and_equal_weight_exact",
    "all_twelve_prototype_rows_finite_nonidentical",
    "all_twelve_prototype_row_gradients_finite_nonzero",
    "online_target_perception_bitwise_equal",
    "target_requires_grad_false",
    "three_channel_state_exact",
    "all_logits_in_closed_interval_minus4_to0",
    "predictor_target_and_fixed_negative_gradients_absent",
    "no_hidden_auxiliary_bypass",
    "all_forbidden_access_counts_zero",
)
FINAL_MARKERS = (
    "multiprototype_mechanism_receipt_ready",
    "active_training_scope_multiprototype_v1",
)

MULTIPROTOTYPE_UTILIZATION = {
    "gating": False,
    "population": {
        "role": "checkpoint_selection",
        "side": "current",
        "row_count": 495,
        "labels": "current_labels",
        "rgb": "current_rgb",
    },
    "reported_per_target_class": [
        "target_class_valid_cell_count",
        "per_component_posterior_responsibility_mean",
        "per_component_winner_share",
        "mean_responsibility_entropy_nats",
        "effective_component_count",
    ],
}

EXECUTION_AUTHORITY = {
    **dict(_V12.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v12_update50_trend_gate_timing_only": False,
    "science_identical_to_frozen_v11": False,
    "model_data_seed_schedule_loss_optimizer_and_ema_identical_to_frozen_v11": False,
    "one_fresh_multiprototype_state_head_v1_perception_attempt_only": True,
    "sole_trainable_mechanism_delta_is_equal_weight_four_prototypes_per_class": True,
    "science_identical_to_frozen_v8": False,
    "science_identical_to_frozen_v10": False,
    "science_identical_to_frozen_v12": False,
    "v12_runtime_output_or_state_reuse_authorized": False,
    "v13_timing_successor_authorized": False,
}


def frozen_v12_source_manifest_binding() -> dict[str, Any]:
    return {"path": FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH, "commit": FROZEN_V12_SOURCE_MANIFEST_COMMIT, "file_sha256": FROZEN_V12_SOURCE_MANIFEST_FILE_SHA256, "content_sha256": FROZEN_V12_SOURCE_MANIFEST_CONTENT_SHA256, "byte_count": FROZEN_V12_SOURCE_MANIFEST_BYTE_COUNT, "status": FROZEN_V12_SOURCE_MANIFEST_STATUS, "source_count": FROZEN_V12_SOURCE_COUNT}


def frozen_v12_review_binding() -> dict[str, Any]:
    return {"path": FROZEN_V12_REVIEW_RELATIVE_PATH, "commit": FROZEN_V12_REVIEW_COMMIT, "file_sha256": FROZEN_V12_REVIEW_FILE_SHA256, "content_sha256": FROZEN_V12_REVIEW_CONTENT_SHA256, "byte_count": FROZEN_V12_REVIEW_BYTE_COUNT, "status": FROZEN_V12_REVIEW_STATUS}


def frozen_v12_authorization_binding() -> dict[str, Any]:
    return {"path": FROZEN_V12_AUTHORIZATION_RELATIVE_PATH, "commit": FROZEN_V12_AUTHORIZATION_COMMIT, "file_sha256": FROZEN_V12_AUTHORIZATION_FILE_SHA256, "content_sha256": FROZEN_V12_AUTHORIZATION_CONTENT_SHA256, "byte_count": FROZEN_V12_AUTHORIZATION_BYTE_COUNT, "status": FROZEN_V12_AUTHORIZATION_STATUS}


def v12_terminal_audit_binding() -> dict[str, Any]:
    return {"path": V12_TERMINAL_AUDIT_RELATIVE_PATH, "commit": V12_TERMINAL_AUDIT_COMMIT, "file_sha256": V12_TERMINAL_AUDIT_FILE_SHA256, "content_sha256": V12_TERMINAL_AUDIT_CONTENT_SHA256, "byte_count": V12_TERMINAL_AUDIT_BYTE_COUNT, "status": V12_TERMINAL_AUDIT_STATUS, "classification": V12_TERMINAL_AUDIT_CLASSIFICATION}


def preregistration_binding() -> dict[str, Any]:
    return {"path": PREREGISTRATION_RELATIVE_PATH, "commit": PREREGISTRATION_COMMIT, "file_sha256": PREREGISTRATION_FILE_SHA256, "content_sha256": PREREGISTRATION_CONTENT_SHA256, "byte_count": PREREGISTRATION_BYTE_COUNT, "status": PREREGISTRATION_STATUS}


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V12.runtime_authorization_template())
    value["experiment_scope"].update({
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "fresh_initialization_required": True,
        "prior_runtime_or_checkpoint_reuse": False,
        "v12_runtime_output_reuse": False,
        "v12_retry_resume_repair_or_recovery": False,
        "multiprototype_state_head_v1_only": True,
    })
    return value


def science_contract() -> dict[str, Any]:
    value = deepcopy(_V12.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Can equal-weight four-prototype class capacity fix occupied-space "
        "fidelity while retaining V12 RGB grounding?"
    )
    value["governing_documents"]["active_preregistration"] = preregistration_binding()
    value["governing_documents"]["v12_terminal_audit"] = v12_terminal_audit_binding()
    value["model"]["runtime_source"] = MODEL_RELATIVE_PATH
    active_head = {
        "prototype_shape": [3, 4, 64],
        "prototype_parameter_count": 768,
        "cell_and_prototype_l2_epsilon": 1e-12,
        "cell_feature_normalization_axis": "feature_dimension_64_only",
        "prototype_normalization_axis": "feature_dimension_64_only_per_row",
        "squared_distance_reduction_axis": "feature_dimension_64_only",
        "logsumexp_reduction_axis": "prototype_component_dimension_4_only",
        "formula": "logsumexp_k(-sum_d((z_hat-p_hat[c,k])**2))-log(4)",
        "fixed_equal_component_weight": 0.25,
        "learned_weight_temperature_bias_routing_or_auxiliary_loss": False,
        "output_classes": ["UNKNOWN", "FREE", "OCCUPIED"],
        "output_logit_range_inclusive": [-4.0, 0.0],
    }
    value["model"]["state_head"] = deepcopy(active_head)
    # V8 stored the then-active head under the decoder description as well as
    # in the parameter inventory.  Replace that historical active leaf so the
    # new contract contains no contradictory single-prototype specification.
    value["model"]["bev_decoder"]["prototype_state_head"] = deepcopy(active_head)
    value["model"]["bev_decoder"]["online_decoder_parameter_count"] = 88_384
    value["model"]["bev_decoder"]["online_decoder_parameter_tensor_count"] = 31
    value["model"]["bev_decoder"]["counts_include_state_head"] = True
    initialization = value["model"]["initialization"]
    initialization["fresh_v8_draw_order"] = list(V8_FRESH_DRAW_ORDER)
    initialization["prior_v12_runtime_or_parameter_reuse"] = False
    initial_binding = initialization["initial_state_component_binding"]
    initial_binding["fresh_prototype_head_local_state_sha256"] = (
        MULTIPROTOTYPE_INITIAL_HEAD_STATE_SHA256
    )
    initial_binding["prototype_parameter_shape"] = [3, 4, 64]
    initialization["initial_state_component_binding_sha256"] = (
        canonical_json_sha256(initial_binding)
    )
    value["model"]["call_graph"]["online_current_and_next_RGB"] = (
        "weight_shared_encoder_learned_query_multiprototype_decoder_for_G"
    )
    value["model"]["parameter_inventory"] = deepcopy(MODEL_PARAMETER_INVENTORY)
    value["gates"]["thresholds"] = {str(k): deepcopy(v) for k, v in GATE_THRESHOLDS.items()}
    value["gates"]["controls"] = {str(k): list(v) for k, v in GATE_CONTROLS.items()}
    value["gates"]["frozen_v8_runtime_readiness_markers_required"] = False
    value["gates"]["multiprototype_runtime_readiness_markers_required"] = list(
        FINAL_MARKERS
    )
    value["lifecycle"]["output_root"] = OUTPUT_ROOT_RELATIVE_PATH
    value["lifecycle"]["scientific_successor_of"] = _V12.EXPERIMENT_ID
    value["lifecycle"]["new_architecture_family_not_v13_timing_successor"] = True
    value["lifecycle"]["v12_checkpoint_tensor_trace_receipt_parameter_optimizer_registry_snapshot_or_rng_reuse"] = False
    value["phase_adapter"]["scope"] = (
        "multiprototype_state_head_v1_perception_only"
    )
    value["phase_adapter"]["optimizer"] = (
        "one_exact_frozen_v12_adamw_constructed_once_never_reset"
    )
    value["authority"].update({
        "multiprototype_v1_execution_authorized_by_source_contract": False,
        "v12_checkpoint_or_runtime_output_reuse_authorized": False,
        "v13_timing_successor_authorized": False,
    })
    value["repository_goal"] = (
        "fully_learned_RGB_only_perception_JEPA_navigation_stack_validated_"
        "later_on_untouched_externally_custodied_heldout_mazes"
    )
    value["scientific_checks"] = {
        "sole_trainable_delta_is_equal_weight_four_prototypes_per_class": True,
        "decision_contract_changes_are_family_specific_and_separately_enumerated": True,
        "encoder_decoder_predictor_data_labels_seed_schedule_loss_optimizer_ema_preserved": True,
        "fresh_initialization_and_zero_v12_runtime_reuse": True,
        "predictor_frozen_excluded_and_zero_forward_objective_backward_update": True,
        "all_four_gates_match_committed_preregistration": True,
        "non_gating_component_utilization_cannot_change_control": True,
        "one_fresh_attempt_caps_and_downstream_denials_exact": True,
        "v12_terminal_failure_bound_and_no_v13_successor": True,
        "no_runtime_or_protected_material_opened_by_source_work": True,
    }
    value["scientific_delta"] = {
        "trainable_mechanism_delta_count": 1,
        "mechanism": "equal_weight_four_prototype_per_class_normalized_mixture_state_head",
        "old_prototype_shape": [3, 64],
        "new_prototype_shape": [3, 4, 64],
        "online_parameter_delta": 576,
        "decision_contract_is_new_family_specific": True,
        "decision_contract_changes": [
            "update_50_directional_health_gate",
            "update_100_decisive_thresholds_and_strict_v12_rough_comparators",
            "update_250_same_run_rough_occupied_nonregression",
            "new_family_controls_root_source_identity_and_no_successor_rules",
        ],
        "not_v13_timing_successor": True,
    }
    value["multiprototype_utilization"] = deepcopy(MULTIPROTOTYPE_UTILIZATION)
    return value


def science_identity_receipt() -> dict[str, Any]:
    return {
        "frozen_v12_science_contract_sha256": canonical_json_sha256(_V12.science_contract()),
        "multiprototype_v1_science_contract_sha256": canonical_json_sha256(science_contract()),
        "sole_trainable_mechanism_delta": "equal_weight_four_prototypes_per_class",
        "frozen_v12_runtime_reuse_authorized": False,
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


def _common_conjuncts(update: int, metrics: Mapping[str, Any], prior: bool) -> dict[str, bool]:
    expected = perception_accounting(update)
    result = {
        "prior_gates_passed": prior,
        "multiprototype_mechanism_receipt_ready": _exact_bool(metrics.get("multiprototype_mechanism_receipt_ready"), name="multiprototype_mechanism_receipt_ready"),
        "active_training_scope_is_perception_only": metrics.get("active_training_scope_multiprototype_v1") == "perception_only",
        "all_registered_values_finite": _exact_bool(metrics.get("all_registered_values_finite"), name="all_registered_values_finite"),
        "state_nonconstant": _exact_bool(metrics.get("state_nonconstant"), name="state_nonconstant"),
        "all_forbidden_access_counts_zero": _exact_bool(metrics.get("all_forbidden_access_counts_zero"), name="all_forbidden_access_counts_zero"),
    }
    for field, target in expected.items():
        observed = metrics.get(field)
        if type(observed) is not int:
            raise ValueError(f"{field} must be int")
        result[f"{field}_equals_{target}"] = observed == target
    return result


def evaluate_gate(update: int, metrics: Mapping[str, Any], *, update_zero: Mapping[str, Any] | None = None, update_100: Mapping[str, Any] | None = None, prior_gates_passed: bool = True) -> dict[str, Any]:
    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 50, 100, or 250")
    if type(prior_gates_passed) is not bool:
        raise ValueError("prior_gates_passed must be bool")
    present = tuple(key in metrics for key in FINAL_MARKERS)
    if present == (False, False):
        # The inner frozen observer sees no active-family markers.  Preserve
        # its exact non-authoritative V12 dispatch receipt; the outer leaf
        # observer must overwrite it with the family-specific final receipt.
        return _V12.evaluate_gate(
            update,
            metrics,
            update_zero=update_zero,
            update_100=update_100,
            prior_gates_passed=prior_gates_passed,
        )
    if present != (True, True):
        raise ValueError("multiprototype final marker presence is partial")
    conjuncts = _common_conjuncts(update, metrics, prior_gates_passed)
    if update == 0:
        for field in INTEGRITY_FIELDS:
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        conjuncts["initial_online_to_target_hard_sync_count_equals_1"] = metrics.get("initial_online_to_target_hard_sync_count") == 1
        conjuncts["correct_rgb_scene_wins_exactly_8"] = metrics.get("correct_rgb_scene_win_count") == 8
    else:
        if update_zero is None:
            raise ValueError("update_zero is required")
        free = _finite(metrics.get("aggregate_free_recall"), name="aggregate_free_recall")
        occupied = _finite(metrics.get("aggregate_occupied_recall"), name="aggregate_occupied_recall")
        g = _finite(metrics.get("G"), name="G")
        nll = _finite(metrics.get("aggregate_raster_nll"), name="aggregate_raster_nll")
        ba = _finite(metrics.get("aggregate_raster_balanced_accuracy"), name="aggregate_raster_balanced_accuracy")
        rough_ba = _finite(metrics.get("rough_raster_balanced_accuracy"), name="rough_raster_balanced_accuracy")
        rough_occ = _finite(metrics.get("rough_raster_occupied_recall"), name="rough_raster_occupied_recall")
        zero_g = _finite(update_zero.get("G"), name="update_zero.G")
        conjuncts["correct_rgb_scene_wins_exactly_8"] = metrics.get("correct_rgb_scene_win_count") == 8
        if update == 50:
            conjuncts.update({
                "G_strictly_lower_than_update_zero": g < zero_g,
                "raster_nll_strictly_lower_than_update_zero": nll < _finite(update_zero.get("aggregate_raster_nll"), name="update_zero.NLL"),
                "balanced_accuracy_strictly_higher_than_update_zero": ba > _finite(update_zero.get("aggregate_raster_balanced_accuracy"), name="update_zero.BA"),
                "occupied_recall_strictly_higher_than_update_zero": occupied > _finite(update_zero.get("aggregate_occupied_recall"), name="update_zero.occupied"),
                "rough_balanced_strictly_higher_than_update_zero": rough_ba > _finite(update_zero.get("rough_raster_balanced_accuracy"), name="update_zero.rough_BA"),
                "rough_occupied_strictly_higher_than_update_zero": rough_occ > _finite(update_zero.get("rough_raster_occupied_recall"), name="update_zero.rough_occupied"),
                "free_recall_at_least_point25": free >= 0.25,
                "free_occupied_gap_at_most_point60": abs(free - occupied) <= 0.60,
            })
        elif update == 100:
            conjuncts.update({
                "G_strictly_lower_than_update_zero": g < zero_g,
                "balanced_accuracy_at_least_point72": ba >= 0.72,
                "free_recall_at_least_point68": free >= 0.68,
                "occupied_recall_at_least_point80": occupied >= 0.80,
                "free_occupied_gap_at_most_point20": abs(free - occupied) <= 0.20,
                "raster_nll_at_most_point46": nll <= 0.46,
                "rough_balanced_strictly_above_v12_u100": rough_ba > 0.732972219013282,
                "rough_occupied_strictly_above_v12_u100": rough_occ > 0.5722940226171244,
            })
        else:
            if update_100 is None:
                raise ValueError("update_100 is required")
            conjuncts.update({
                "balanced_accuracy_at_least_point80": ba >= 0.80,
                "free_recall_at_least_point68": free >= 0.68,
                "occupied_recall_at_least_point88": occupied >= 0.88,
                "free_occupied_gap_at_most_point25": abs(free - occupied) <= 0.25,
                "raster_nll_at_most_point42": nll <= 0.42,
                "raster_nll_at_most_update100_plus_point01": nll <= _finite(update_100.get("aggregate_raster_nll"), name="update_100.NLL") + 0.01,
                "rough_balanced_at_least_registered_floor": rough_ba >= 0.7719525,
                "rough_occupied_at_least_same_run_update100": rough_occ >= _finite(update_100.get("rough_raster_occupied_recall"), name="update_100.rough_occupied"),
            })
    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": GATE_CONTROLS[update][1 if passed else 0],
        "gate_mode": "FINAL_MULTIPROTOTYPE_STATE_HEAD_V1_RECEIPT",
        "multiprototype_mechanism_receipt_ready": True,
        "conjuncts": conjuncts,
        "thresholds": deepcopy(GATE_THRESHOLDS[update]),
        "perception_accounting": perception_accounting(update),
    }


def validate_failure_status_chain(value: object) -> dict[str, str]:
    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("failure status-chain fields changed")
    control = value["metrics"]
    if type(control) is not str or control not in FAILURE_CONTROLS or any(value[field] != control for field in fields):
        raise ValueError("failure receipts are not one exact multiprototype gate control")
    return dict(value)


def _read_bound_json(relative_path: str, *, file_sha256: str, content_sha256: str, byte_count: int, status: str, classification: str | None = None, **_binding_metadata: Any) -> dict[str, Any]:
    read = _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    raw = read(ROOT / relative_path)
    core = json.loads(raw)
    declared = core.pop("content_sha256", None)
    parsed = json.loads(raw)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != file_sha256 or declared != content_sha256 or canonical_json_sha256(core) != content_sha256 or parsed.get("status") != status or (classification is not None and parsed.get("classification") != classification):
        raise PermissionError(f"governing document changed: {relative_path}")
    return parsed


def validate_frozen_v12_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("frozen V12 closure must use repository root")
    current = _V12.current_source_bindings(root)
    if current.get(FROZEN_V12_SOURCE_MANIFEST_RELATIVE_PATH) != FROZEN_V12_SOURCE_MANIFEST_FILE_SHA256:
        raise PermissionError("frozen V12 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v12_source_closure(root)
    for binding in (frozen_v12_review_binding(), frozen_v12_authorization_binding(), v12_terminal_audit_binding(), preregistration_binding()):
        _read_bound_json(binding["path"], file_sha256=binding["file_sha256"], content_sha256=binding["content_sha256"], byte_count=binding["byte_count"], status=binding["status"])
        current[binding["path"]] = binding["file_sha256"]
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="multiprototype source manifest")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    bindings = value.get("source_bindings")
    expected_fields = {"schema", "status", "entrypoints", "forced_dynamic_sources", "excluded_runtime_categories", "source_paths", "source_bindings", "source_bindings_sha256", "source_count", "generated_input_open_count", "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count", "whole_tree_export_authorized", "authority", "content_sha256"}
    if set(value) != expected_fields or value.get("schema") != SOURCE_MANIFEST_SCHEMA or value.get("status") != "PASS_SOURCE_CLOSURE" or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS) or value.get("forced_dynamic_sources") != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES) or value.get("excluded_runtime_categories") != list(PROHIBITED_RUNTIME_CATEGORIES) or value.get("source_paths") != list(SOURCE_PATHS) or type(bindings) is not list or value.get("source_count") != 149 or value.get("source_bindings_sha256") != canonical_json_sha256(bindings) or value.get("generated_input_open_count") != 0 or value.get("checkpoint_or_tensor_open_count") != 0 or value.get("sealed_or_heldout_open_count") != 0 or value.get("whole_tree_export_authorized") is not False or value.get("authority") != SOURCE_ONLY_AUTHORITY or not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("multiprototype source manifest contract changed")
    normalized = []
    safe = _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1.safe_relative_source_path
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {"path", "file_sha256", "byte_count"} or not is_sha256(binding.get("file_sha256")) or type(binding.get("byte_count")) is not int or binding["byte_count"] <= 0:
            raise PermissionError("multiprototype source binding changed")
        normalized.append(safe(binding["path"]))
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("multiprototype source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(raw)
    result = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound source changed: {binding['path']}")
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(source_manifest_binding: Mapping[str, Any] | None) -> dict[str, Any]:
    if source_manifest_binding is None:
        read = _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
        raw = read(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        manifest = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(SOURCE_MANIFEST_RELATIVE_PATH, raw, content_sha256=str(manifest["content_sha256"]))
    return validate_binding(dict(source_manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH)


def _source_freeze_commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{name} must be one exact 40-hex commit")
    return value


def _review_source_freeze_commit(
    review_binding: Mapping[str, Any],
) -> str:
    binding = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    read = (
        _V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
    raw = read(ROOT / REVIEW_RELATIVE_PATH)
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError("multiprototype source review binding changed")
    review = parse_canonical_json(raw, name="multiprototype source review")
    review_core = dict(review)
    declared = review_core.pop("content_sha256", None)
    if (
        declared != binding["content_sha256"]
        or canonical_json_sha256(review_core) != declared
    ):
        raise PermissionError("multiprototype source review content changed")
    return _source_freeze_commit(
        review.get("source_freeze_commit"),
        name="review.source_freeze_commit",
    )


def validate_review(value: object, *, expected_sources: Mapping[str, str], source_manifest_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fields = {"schema", "status", "implementation_authors", "reviewer", "source_freeze_commit", "reviewed_sources", "source_manifest", "frozen_v12_source_manifest", "frozen_v12_source_review", "frozen_v12_execution_authorization", "v12_terminal_audit", "preregistration", "science_contract", "science_identity", "checks", "findings", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("multiprototype source review fields changed")
    core = dict(value); declared = core.pop("content_sha256", None); reviewer = value["reviewer"]
    required = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    checks = {"source_only_imports_pass": True, "focused_cpu_tests_pass": True, "full_recursive_cpu_tests_pass": True, "two_runner_seams_only": True, "old_v8_observer_unreachable": True, "source_freeze_commit_matches_reviewed_tree": True, "all_implementation_authors_excluded": True, "generated_or_protected_runtime_inputs_opened": [], "sealed_or_heldout_opened": []}
    source_commit = _source_freeze_commit(value.get("source_freeze_commit"), name="review.source_freeze_commit")
    if value["schema"] != REVIEW_SCHEMA or value["status"] != REVIEW_STATUS or value["implementation_authors"] != list(IMPLEMENTATION_AUTHORS) or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer in IMPLEMENTATION_AUTHORS or not required.issubset(expected_sources) or value["reviewed_sources"] != dict(expected_sources) or value["source_manifest"] != _manifest_binding_or_read(source_manifest_binding) or value["frozen_v12_source_manifest"] != frozen_v12_source_manifest_binding() or value["frozen_v12_source_review"] != frozen_v12_review_binding() or value["frozen_v12_execution_authorization"] != frozen_v12_authorization_binding() or value["v12_terminal_audit"] != v12_terminal_audit_binding() or value["preregistration"] != preregistration_binding() or value["science_contract"] != science_contract() or value["science_identity"] != science_identity_receipt() or value["checks"] != checks or value["findings"] != [] or value["authority"] != REVIEW_AUTHORITY or not source_commit or not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("multiprototype source review did not pass")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {"schema", "status", "authorizer", "source_freeze_commit", "independent_source_review", "frozen_v12_source_manifest", "frozen_v12_source_review", "frozen_v12_execution_authorization", "v12_terminal_audit", "preregistration", "runtime_inputs", "experiment", "science_identity", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("multiprototype authorization fields changed")
    core = dict(value); declared = core.pop("content_sha256", None); authorizer = value["authorizer"]
    expected_review = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    review_source_commit = _review_source_freeze_commit(expected_review)
    if value["schema"] != AUTHORIZATION_SCHEMA or value["status"] != AUTHORIZATION_STATUS or type(authorizer) is not str or not authorizer.startswith("/root/") or authorizer in {*IMPLEMENTATION_AUTHORS, reviewer} or value["source_freeze_commit"] != review_source_commit or value["independent_source_review"] != expected_review or value["frozen_v12_source_manifest"] != frozen_v12_source_manifest_binding() or value["frozen_v12_source_review"] != frozen_v12_review_binding() or value["frozen_v12_execution_authorization"] != frozen_v12_authorization_binding() or value["v12_terminal_audit"] != v12_terminal_audit_binding() or value["preregistration"] != preregistration_binding() or value["runtime_inputs"] != runtime_authorization_template() or value["experiment"] != science_contract() or value["science_identity"] != science_identity_receipt() or value["authority"] != EXECUTION_AUTHORITY or not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("multiprototype execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V12.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings", "evaluate_gate", "frozen_v12_authorization_binding",
    "frozen_v12_review_binding", "frozen_v12_source_manifest_binding",
    "preregistration_binding", "runtime_authorization_template", "science_contract",
    "science_identity_receipt", "v12_terminal_audit_binding", "validate_authorization",
    "validate_failure_status_chain", "validate_frozen_v12_source_closure",
    "validate_governing_documents", "validate_review", "validate_source_manifest",
})
