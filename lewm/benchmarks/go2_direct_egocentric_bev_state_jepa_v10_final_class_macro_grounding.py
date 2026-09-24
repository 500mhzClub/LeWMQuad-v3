"""Source-only contract for the Direct BEV V10 macro-grounding probe.

V10 preserves the complete frozen V9 RGB-only perception experiment and
changes one scientific leaf: the grounding loss becomes an equal-present-
final-class macro NLL.  Importing this module reads source and committed
governance documents only and grants no runtime or downstream authority.
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
FROZEN_V9_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py"
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


_V9 = _source_only_module(
    "_lewm_direct_bev_v10_frozen_v9_contract",
    FROZEN_V9_CONTRACT_RELATIVE_PATH,
)
_FROZEN_V9_SCIENCE_CONTRACT = _V9.science_contract()

for _name in _V9.__all__:
    globals()[_name] = getattr(_V9, _name)

canonical_json_bytes = _V9.canonical_json_bytes
canonical_json_sha256 = _V9.canonical_json_sha256
is_sha256 = _V9.is_sha256
with_content_sha256 = _V9.with_content_sha256
parse_canonical_json = _V9.parse_canonical_json
artifact_binding = _V9.artifact_binding
validate_binding = _V9.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v9_contract_implementation"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
)
MODEL_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V9_RUNNER_RELATIVE_PATH = _V9.RUNNER_RELATIVE_PATH
FROZEN_V9_LAUNCHER_RELATIVE_PATH = _V9.LAUNCHER_RELATIVE_PATH
FROZEN_V9_MODEL_RELATIVE_PATH = _V9.MODEL_RELATIVE_PATH
FROZEN_V9_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V9.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "7c1886942a298964b083457f335329259e594593"
PREREGISTRATION_FILE_SHA256 = (
    "3e901a8847c21d44d6d1f1a41e7a71deb8da221043dd6ed461926ced9e2fe4a6"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "341239f40885c5a55042cd88d49b4e6401a332fa066f3b7bb0d7062880105b83"
)
PREREGISTRATION_BYTE_COUNT = 19_648
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_RGB_ONLY_V10_FINAL_CLASS_MACRO_GROUNDING_"
    "PERCEPTION_FALSIFICATION_PENDING_SOURCE_FREEZE_INDEPENDENT_REVIEW_"
    "AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH = _V9.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V9_SOURCE_MANIFEST_COMMIT = (
    "2c349f8225e8525e691c77e2fb0fe5573d92cb89"
)
FROZEN_V9_SOURCE_MANIFEST_FILE_SHA256 = (
    "1da8e6fe8babac775ae6a977e2b480ecb852f4cd2edbefca49d9e3105c9ab474"
)
FROZEN_V9_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "2003108a6a13a9532210f31839a632d44d7cd47e14965ad93a870d4633e5073a"
)
FROZEN_V9_SOURCE_MANIFEST_BYTE_COUNT = 44_703
FROZEN_V9_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V9_SOURCE_COUNT = 127

FROZEN_V9_REVIEW_RELATIVE_PATH = _V9.REVIEW_RELATIVE_PATH
FROZEN_V9_REVIEW_COMMIT = "c0c7d4c6d26dc284b521cf696e7718bfabe73062"
FROZEN_V9_REVIEW_FILE_SHA256 = (
    "f7e45b7085696c20f472376b06020bec27623075e95acd7df8b1bb986708219f"
)
FROZEN_V9_REVIEW_CONTENT_SHA256 = (
    "65cb9734019cd7f7d78a7e9d9ae2d442bde59115c0e22b18dba9dda64efe853b"
)
FROZEN_V9_REVIEW_BYTE_COUNT = 70_484
FROZEN_V9_REVIEW_STATUS = (
    "PASS_SOURCE_SCIENCE_IDENTITY_CHECKPOINT_REGISTRY_ADAPTER_AND_COMPLETE_"
    "FAILURE_RECEIPTS"
)

FROZEN_V9_AUTHORIZATION_RELATIVE_PATH = _V9.AUTHORIZATION_RELATIVE_PATH
FROZEN_V9_AUTHORIZATION_COMMIT = (
    "a4d31f688ccd05cea283db318fb73b988fbf087d"
)
FROZEN_V9_AUTHORIZATION_FILE_SHA256 = (
    "e61545300c9902cd9786f636049a143b945977e76f9fe5cc49abe7d84d02412e"
)
FROZEN_V9_AUTHORIZATION_CONTENT_SHA256 = (
    "f9f4c3acd33ff6de559c0815d980519e1f8e1d48f42e899747aca340715de498"
)
FROZEN_V9_AUTHORIZATION_BYTE_COUNT = 58_605
FROZEN_V9_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V9_CHECKPOINT_SEMANTIC_REGISTRY_"
    "INTEGRITY_REPLACEMENT"
)

V9_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_"
    "terminal_audit_2026-07-27.json"
)
V9_TERMINAL_AUDIT_COMMIT = "7984c7749cc44c6444bf7229a809c6e9f01063bf"
V9_TERMINAL_AUDIT_FILE_SHA256 = (
    "af5f82c809aae3f3954e64147b3a71af96286436e1790440fdd161bc86bd4c03"
)
V9_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "f43745ae1329775f36a0b9ad01b34588e662387bca98c34430324cf09a0cd69c"
)
V9_TERMINAL_AUDIT_BYTE_COUNT = 18_951
V9_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_UPDATE_100_SCIENTIFIC_FAILURE_CLOSES_V9_NO_RETRY"
)
V9_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ONE_SHOT_TERMINAL_SCIENTIFIC_FAILURE_AT_UPDATE_100_AFTER_PASSED_"
    "UPDATE_ZERO_AND_UPDATE_50_GATES_ONLY_THREE_CLASS_BALANCE_THRESHOLDS_"
    "FAILED_V9_PERMANENTLY_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V9.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 127 or len(SOURCE_PATHS) != 133:
    raise RuntimeError("V10 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V9_REVIEW_RELATIVE_PATH,
    FROZEN_V9_AUTHORIZATION_RELATIVE_PATH,
    V9_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v10/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v10_"
    "final_class_macro_grounding_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V10_"
    "FINAL_CLASS_MACRO_GROUNDING_PREFLIGHT_JSON"
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

REVIEW_STATUS = "PASS_SOURCE_AND_FINAL_CLASS_MACRO_GROUNDING_SCIENCE"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V10_FINAL_CLASS_MACRO_GROUNDING_"
    "PERCEPTION_FALSIFICATION"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V9.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v9_checkpoint_semantic_registry_integrity_replacement_only": (
        False
    ),
    "v9_retry_resume_repair_recovery_or_extension_authorized": False,
    "v9_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_"
    "authorized": False,
    "one_fresh_v10_final_class_macro_grounding_perception_attempt_only": True,
    "final_class_macro_grounding_perception_only": True,
    "predictor_training_or_evaluation_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
}

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
SCHEDULE_PREFIX_SHA256 = dict(_V9.SCHEDULE_PREFIX_SHA256)
FROZEN_V9_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V9_SCIENCE_CONTRACT
)
if FROZEN_V9_SCIENCE_CONTRACT_SHA256 != (
    "bacb31b0eb2070821bbd37862e6f3b9a39d7ecb0ab14ed8d758894c36f06f728"
):
    raise PermissionError("frozen V9 science contract identity changed")
if (
    _V9.MODEL_RELATIVE_PATH == MODEL_RELATIVE_PATH
    or _V9.MAXIMUM_UPDATES != MAXIMUM_UPDATES
    or _V9.MAXIMUM_PRESENTATIONS != MAXIMUM_PRESENTATIONS
    or _V9.GPU_ACTIVE_TIME_CAP_MINUTES != GPU_ACTIVE_TIME_CAP_MINUTES
    or tuple(_V9.CHECKPOINT_UPDATES) != CHECKPOINT_UPDATES
    or tuple(_V9.OBSERVATION_UPDATES) != OBSERVATION_UPDATES
    or dict(_V9.SCHEDULE_PREFIX_SHA256) != SCHEDULE_PREFIX_SHA256
):
    raise PermissionError("frozen V9 schedule, cap, or source identity changed")

FINAL_CLASS_ORDER = ("UNKNOWN", "FREE", "OCCUPIED")
GROUNDING_PUBLIC_SCALE = math.log(2.0) / math.log(3.0)

PRESERVED_FROZEN_V9_SCIENCE_TOP_LEVEL_FIELDS = (
    "repository_goal",
    "model",
    "data",
    "loader",
    "optimizer",
    "schedule",
    "access_policy",
    "frozen_v7_integrity_provenance",
)
OBJECTIVE_CHANGED_PATHS = (
    "objective.G",
    "objective.wrong_rgb_control",
    "objective.total",
    "objective.perception_only_total",
    "objective.G_formula_and_hierarchical_three_state_energy",
    "objective.final_class_macro_grounding",
    "objective.public_runner_compatibility_fields",
)

SCIENTIFIC_DELTA = {
    "scientific_delta_count": 1,
    "scientific_delta_name": "final_class_macro_grounding_objective_leaf",
    "sole_scientific_leaf_delta": (
        "replace_the_two_branch_hard_hierarchical_grounding_function_for_"
        "training_and_correct_wrong_RGB_control_with_per_raster_present_"
        "final_class_macro_NLL_and_train_with_raw_G_macro_over_log3"
    ),
    "final_class_order": list(FINAL_CLASS_ORDER),
    "absent_class_rule": (
        "omit_absent_final_classes_from_both_sum_and_denominator_per_raster"
    ),
    "public_G_scale": "log(2)/log(3)",
    "public_total_identity": "G/log(2)=raw_G_macro/log(3)",
    "changed_contract_paths": list(OBJECTIVE_CHANGED_PATHS),
    "model_data_label_seed_initialization_schedule_optimizer_EMA_metrics_"
    "caps_and_gradient_routes_preserved": True,
    "predictor_constructed_fresh_frozen_not_called_not_optimized": True,
    "prior_checkpoint_tensor_trace_receipt_or_runtime_output_reuse": False,
}


def perception_accounting(update: int) -> dict[str, int]:
    if type(update) is not int or not 0 <= update <= MAXIMUM_UPDATES:
        raise ValueError("V10 perception-accounting update is out of bounds")
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
    update: perception_accounting(update)
    for update in OBSERVATION_UPDATES
}

GATE_THRESHOLDS = {
    0: {},
    50: {
        "G_macro_strictly_less_than_update_0": True,
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.60,
        "aggregate_free_recall_minimum_inclusive": 0.25,
        "aggregate_occupied_recall_minimum_inclusive": 0.75,
        "absolute_free_minus_occupied_recall_gap_maximum_inclusive": 0.60,
        "aggregate_raster_nll_maximum_inclusive": 0.80,
        "rough_raster_balanced_accuracy_strictly_greater_than": 0.0,
        "rough_raster_occupied_recall_strictly_greater_than": 0.0,
        "correct_rgb_macro_scene_win_count_required": 8,
    },
    100: {
        "G_macro_strictly_less_than_update_0": True,
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.70,
        "aggregate_free_recall_minimum_inclusive": 0.50,
        "aggregate_occupied_recall_minimum_inclusive": 0.80,
        "absolute_free_minus_occupied_recall_gap_maximum_inclusive": 0.35,
        "aggregate_raster_nll_maximum_inclusive": 0.46,
        "rough_raster_balanced_accuracy_strictly_greater_than_update_zero": (
            True
        ),
        "rough_raster_occupied_recall_strictly_greater_than_update_zero": True,
        "correct_rgb_macro_scene_win_count_required": 8,
    },
    250: {
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.80,
        "aggregate_free_recall_minimum_inclusive": 0.68,
        "aggregate_occupied_recall_minimum_inclusive": 0.88,
        "absolute_free_minus_occupied_recall_gap_maximum_inclusive": 0.25,
        "aggregate_raster_nll_maximum_inclusive": 0.42,
        "aggregate_raster_nll_relative_to_update_100": (
            "NLL_update_250<=NLL_update_100+0.01"
        ),
        "rough_raster_balanced_accuracy_minimum_inclusive": 0.7719525,
        "rough_raster_occupied_recall_minimum_inclusive": 0.4319467,
        "correct_rgb_macro_scene_win_count_required": 8,
    },
}

CONTROL_UPDATE_ZERO_FAIL = (
    "FAIL_UPDATE_ZERO_V10_FINAL_CLASS_MACRO_INTEGRITY_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_ZERO = (
    "CONTINUE_AFTER_UPDATE_ZERO_V10_FINAL_CLASS_MACRO_INTEGRITY_GATE"
)
CONTROL_UPDATE_50_FAIL = (
    "FAIL_UPDATE_50_V10_FINAL_CLASS_MACRO_HEALTH_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_50 = (
    "CONTINUE_AFTER_UPDATE_50_V10_FINAL_CLASS_MACRO_HEALTH_GATE"
)
CONTROL_UPDATE_100_FAIL = (
    "FAIL_UPDATE_100_V10_FINAL_CLASS_MACRO_CONTINUATION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_100 = (
    "CONTINUE_AFTER_UPDATE_100_V10_FINAL_CLASS_MACRO_PERCEPTION_GATE"
)
CONTROL_UPDATE_250_FAIL = (
    "FAIL_UPDATE_250_V10_FINAL_CLASS_MACRO_QUALIFICATION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_PASS = (
    "PASS_DIRECT_BEV_V10_FINAL_CLASS_MACRO_GROUNDING_PERCEPTION_MECHANISM_ONLY"
)
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_ZERO_FAIL, CONTROL_CONTINUE_UPDATE_ZERO),
    50: (CONTROL_UPDATE_50_FAIL, CONTROL_CONTINUE_UPDATE_50),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    250: (CONTROL_UPDATE_250_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v9_manifest_and_all_127_sources_rehashed": True,
    "frozen_v9_review_authorization_and_terminal_audit_exact": True,
    "v9_permanently_closed_and_runtime_reuse_forbidden": True,
    "v10_preregistration_exact": True,
    "sole_scientific_delta_is_final_class_macro_grounding_leaf": True,
    "present_classes_equal_weight_absent_classes_omitted_exact": True,
    "public_G_log2_over_log3_and_total_bit_identity_exact": True,
    "training_and_wrong_RGB_control_share_registered_macro_leaf": True,
    "model_parameter_buffer_module_and_nonloss_forward_graph_exact": True,
    "data_labels_seed_initialization_schedule_optimizer_EMA_and_caps_exact": (
        True
    ),
    "predictor_frozen_excluded_and_zero_forward_objective_backward_update": (
        True
    ),
    "u0_u50_u100_u250_gates_accounting_and_stop_semantics_exact": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v9_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V9_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V9_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V9_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V9_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V9_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V9_SOURCE_COUNT,
    }


def frozen_v9_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V9_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V9_REVIEW_COMMIT,
        "file_sha256": FROZEN_V9_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V9_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V9_REVIEW_BYTE_COUNT,
        "status": FROZEN_V9_REVIEW_STATUS,
    }


def frozen_v9_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V9_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V9_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V9_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V9_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V9_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V9_AUTHORIZATION_STATUS,
    }


def v9_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V9_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V9_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V9_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V9_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V9_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V9_TERMINAL_AUDIT_STATUS,
        "classification": V9_TERMINAL_AUDIT_CLASSIFICATION,
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


def frozen_v9_science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V9_SCIENCE_CONTRACT)


def objective_contract() -> dict[str, Any]:
    value = deepcopy(_FROZEN_V9_SCIENCE_CONTRACT["objective"])
    value["G"] = {
        "formula": "raw_G_macro*log(2)/log(3)",
        "raw_G_macro_formula": "0.5*(raw_macro_current+raw_macro_next)",
        "current_call": "O_current_rgb",
        "next_call": "O_next_rgb",
        "loss": "present_final_class_macro_nll_v10",
        "class_order": list(FINAL_CLASS_ORDER),
        "per_raster": (
            "mean_over_present_classes_of_mean_negative_log_softmax_for_"
            "pixels_of_that_class"
        ),
        "absent_class_rule": "omit_from_sum_and_denominator",
        "row_weighting": "equal_after_each_row_macro_is_formed",
        "hard_label_array": "raster_labels.u1",
        "public_scale": "log(2)/log(3)",
    }
    value["wrong_rgb_control"] = {
        **value["wrong_rgb_control"],
        "loss": "raw_present_final_class_macro_nll_v10",
        "scene_win": (
            "mean_correct_raw_macro_nll_strictly_less_than_mean_wrong_raw_"
            "macro_nll"
        ),
    }
    value["total"] = "G/log(2)=raw_G_macro/log(3)"
    value["perception_only_total"] = "G/log(2)=raw_G_macro/log(3)"
    value.pop("G_formula_and_hierarchical_three_state_energy", None)
    value["final_class_macro_grounding"] = {
        "raw_macro_current": "mean_rows(per_raster_present_class_macro_nll)",
        "raw_macro_next": "mean_rows(per_raster_present_class_macro_nll)",
        "raw_G_macro": "0.5*(raw_macro_current+raw_macro_next)",
        "training_total": "raw_G_macro/log(3)",
        "additional_class_frequency_weights": False,
        "focal_label_smoothing_margin_or_auxiliary_loss": False,
    }
    value["public_runner_compatibility_fields"] = {
        "G": "raw_G_macro*log(2)/log(3)",
        "G_current": "raw_macro_current*log(2)/log(3)",
        "G_next": "raw_macro_next*log(2)/log(3)",
        "required_total_identity": "G/log(2)=raw_G_macro/log(3)",
        "reporting_or_compatibility_only": True,
    }
    return value


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V9.runtime_authorization_template())
    value["experiment_scope"] = {
        **value["experiment_scope"],
        "one_fresh_attempt": True,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_index": ATTEMPT_INDEX,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "maximum_active_gpu_minutes": GPU_ACTIVE_TIME_CAP_MINUTES,
        "fresh_initialization_required": True,
        "perception_only": True,
        "predictor_forward_or_training": False,
        "prior_runtime_or_checkpoint_reuse": False,
        "v9_runtime_output_reuse": False,
        "v9_retry_resume_repair_or_recovery": False,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": (
            False
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    return value


def science_contract() -> dict[str, Any]:
    value = frozen_v9_science_contract()
    frozen_v9_delta = deepcopy(value.pop("scientific_delta"))
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Does equal present-final-class influence in RGB-to-BEV grounding "
        "prevent FREE collapse while retaining OCCUPIED geometry and RGB "
        "sensitivity?"
    )
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v9_source_manifest": frozen_v9_source_manifest_binding(),
        "frozen_v9_source_review": frozen_v9_review_binding(),
        "frozen_v9_execution_authorization": frozen_v9_authorization_binding(),
        "v9_terminal_audit": v9_terminal_audit_binding(),
        "v10_preregistration": preregistration_binding(),
    }
    value["objective"] = objective_contract()
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
        "perception_accounting": {
            str(update): dict(PERCEPTION_ACCOUNTING[update])
            for update in OBSERVATION_UPDATES
        },
        "frozen_v8_runtime_readiness_markers_required": True,
        "separate_v10_observation_seam_or_marker_present": False,
        "stop_at_first_failed_gate": True,
    }
    frozen_lifecycle = dict(value["lifecycle"])
    value["lifecycle"] = {
        **frozen_lifecycle,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "output_root_must_be_absent_before_reservation": True,
        "scientific_successor_of": _V9.EXPERIMENT_ID,
        "one_fresh_attempt": True,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_index": ATTEMPT_INDEX,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
        "perception_only": True,
        "predictor_phase_or_update": False,
        "v9_retry_resume_repair_recovery_or_extension": False,
        "v9_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse": (
            False
        ),
        "retry_resume_extension_second_seed_or_replacement_attempt": False,
    }
    value["phase_adapter"] = {
        **value["phase_adapter"],
        "scope": "v10_final_class_macro_grounding_perception_only",
        "total": "G/log(2)=raw_G_macro/log(3)",
        "optimizer": "one_frozen_v9_adamw_constructed_once_never_reset",
    }
    value["frozen_v9_scientific_provenance"] = {
        "science_contract_sha256": FROZEN_V9_SCIENCE_CONTRACT_SHA256,
        "inherited_v8_architecture_delta": frozen_v9_delta,
        "not_the_active_v10_delta": True,
    }
    value["scientific_delta"] = deepcopy(SCIENTIFIC_DELTA)
    value["authority"] = {
        **value["authority"],
        "v10_execution_authorized_by_source_contract": False,
        "v9_checkpoint_or_runtime_output_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    for field in PRESERVED_FROZEN_V9_SCIENCE_TOP_LEVEL_FIELDS:
        if value[field] != _FROZEN_V9_SCIENCE_CONTRACT[field]:
            raise PermissionError(f"V10 changed frozen V9 science leaf: {field}")
    return value


def normalize_v10_scientific_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V10 experiment differs from its exact contract")
    return frozen_v9_science_contract()


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    return {
        "frozen_v9_science_contract_sha256": (
            FROZEN_V9_SCIENCE_CONTRACT_SHA256
        ),
        "v10_science_contract_sha256": canonical_json_sha256(value),
        "scientific_delta_count": 1,
        "scientific_delta_name": SCIENTIFIC_DELTA["scientific_delta_name"],
        "changed_objective_paths": list(OBJECTIVE_CHANGED_PATHS),
        "preserved_frozen_v9_top_level_fields": list(
            PRESERVED_FROZEN_V9_SCIENCE_TOP_LEVEL_FIELDS
        ),
        "model_data_seed_schedule_optimizer_EMA_and_caps_preserved": True,
        "v9_runtime_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


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
        for field, expected in PERCEPTION_ACCOUNTING[update].items()
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    update_100: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate the exact fail-closed V10 perception gates."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 50, 100, or 250")
    _exact_bool(prior_gates_passed, name="prior_gates_passed")
    inherited_ready = _exact_bool(
        metrics.get("v8_mechanism_receipt_ready"),
        name="v8_mechanism_receipt_ready",
    )
    conjuncts: dict[str, bool] = {
        "prior_gates_passed": prior_gates_passed,
        "frozen_v8_runtime_readiness_witness": inherited_ready,
        "active_training_scope_is_perception_only": (
            metrics.get("active_training_scope_v8") == "perception_only"
        ),
    }
    conjuncts.update(_accounting_conjuncts(update, metrics))
    conjuncts.update({
        "all_registered_values_finite": _exact_bool(
            metrics.get("all_registered_values_finite"),
            name="all_registered_values_finite",
        ),
        "state_nonconstant": _exact_bool(
            metrics.get("state_nonconstant"), name="state_nonconstant"
        ),
    })

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
        conjuncts["correct_rgb_macro_wins_all_eight_scenes"] = (
            _exact_int(
                metrics.get("correct_rgb_scene_win_count"),
                name="correct_rgb_scene_win_count",
            ) >= 8
        )
    elif update == 50:
        if update_zero is None:
            raise ValueError("update_zero baseline is required at update 50")
        free_recall = _finite_number(
            metrics.get("aggregate_free_recall"),
            name="aggregate_free_recall",
        )
        occupied_recall = _finite_number(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        conjuncts.update({
            "G_macro_strictly_lower_than_update_zero": (
                _finite_number(metrics.get("G"), name="G")
                < _finite_number(update_zero.get("G"), name="update_zero.G")
            ),
            "aggregate_raster_balanced_accuracy_at_least_point60": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.60
            ),
            "aggregate_free_recall_at_least_point25": free_recall >= 0.25,
            "aggregate_occupied_recall_at_least_point75": (
                occupied_recall >= 0.75
            ),
            "absolute_free_occupied_recall_gap_at_most_point60": (
                abs(free_recall - occupied_recall) <= 0.60
            ),
            "aggregate_raster_nll_at_most_point80": (
                _finite_number(
                    metrics.get("aggregate_raster_nll"),
                    name="aggregate_raster_nll",
                ) <= 0.80
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
            "correct_rgb_macro_wins_all_eight_scenes": (
                _exact_int(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                ) >= 8
            ),
        })
    elif update == 100:
        if update_zero is None:
            raise ValueError("update_zero baseline is required at update 100")
        free_recall = _finite_number(
            metrics.get("aggregate_free_recall"),
            name="aggregate_free_recall",
        )
        occupied_recall = _finite_number(
            metrics.get("aggregate_occupied_recall"),
            name="aggregate_occupied_recall",
        )
        conjuncts.update({
            "G_macro_strictly_lower_than_update_zero": (
                _finite_number(metrics.get("G"), name="G")
                < _finite_number(update_zero.get("G"), name="update_zero.G")
            ),
            "aggregate_raster_balanced_accuracy_at_least_point70": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.70
            ),
            "aggregate_free_recall_at_least_point50": free_recall >= 0.50,
            "aggregate_occupied_recall_at_least_point80": (
                occupied_recall >= 0.80
            ),
            "absolute_free_occupied_recall_gap_at_most_point35": (
                abs(free_recall - occupied_recall) <= 0.35
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
            "correct_rgb_macro_wins_all_eight_scenes": (
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
            "correct_rgb_macro_wins_all_eight_scenes": (
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
        "gate_mode": "FINAL_V10_FINAL_CLASS_MACRO_GROUNDING_RECEIPT",
        "v10_mechanism_receipt_ready": True,
        "frozen_v8_runtime_readiness_witness": inherited_ready,
        "conjuncts": conjuncts,
        "thresholds": dict(GATE_THRESHOLDS[update]),
        "perception_accounting": dict(PERCEPTION_ACCOUNTING[update]),
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
        raise ValueError("failure receipts are not one exact V10 gate control")
    return dict(value)


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
) -> dict[str, Any]:
    read = _V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
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


def validate_frozen_v9_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V10 frozen V9 closure must use repository root")
    manifest = _read_bound_json(
        FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH,
        file_sha256=FROZEN_V9_SOURCE_MANIFEST_FILE_SHA256,
        content_sha256=FROZEN_V9_SOURCE_MANIFEST_CONTENT_SHA256,
        byte_count=FROZEN_V9_SOURCE_MANIFEST_BYTE_COUNT,
        status=FROZEN_V9_SOURCE_MANIFEST_STATUS,
    )
    if manifest.get("source_count") != FROZEN_V9_SOURCE_COUNT:
        raise PermissionError("frozen V9 source count changed")
    current = _V9.current_source_bindings(root)
    if current.get(FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V9_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V9 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v9_source_closure(root)
    review = _read_bound_json(
        FROZEN_V9_REVIEW_RELATIVE_PATH,
        file_sha256=FROZEN_V9_REVIEW_FILE_SHA256,
        content_sha256=FROZEN_V9_REVIEW_CONTENT_SHA256,
        byte_count=FROZEN_V9_REVIEW_BYTE_COUNT,
        status=FROZEN_V9_REVIEW_STATUS,
    )
    authorization = _read_bound_json(
        FROZEN_V9_AUTHORIZATION_RELATIVE_PATH,
        file_sha256=FROZEN_V9_AUTHORIZATION_FILE_SHA256,
        content_sha256=FROZEN_V9_AUTHORIZATION_CONTENT_SHA256,
        byte_count=FROZEN_V9_AUTHORIZATION_BYTE_COUNT,
        status=FROZEN_V9_AUTHORIZATION_STATUS,
    )
    _V9.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _V9.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    _read_bound_json(
        V9_TERMINAL_AUDIT_RELATIVE_PATH,
        file_sha256=V9_TERMINAL_AUDIT_FILE_SHA256,
        content_sha256=V9_TERMINAL_AUDIT_CONTENT_SHA256,
        byte_count=V9_TERMINAL_AUDIT_BYTE_COUNT,
        status=V9_TERMINAL_AUDIT_STATUS,
        classification=V9_TERMINAL_AUDIT_CLASSIFICATION,
    )
    _read_bound_json(
        PREREGISTRATION_RELATIVE_PATH,
        file_sha256=PREREGISTRATION_FILE_SHA256,
        content_sha256=PREREGISTRATION_CONTENT_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        status=PREREGISTRATION_STATUS,
    )
    current.update({
        FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH: (
            FROZEN_V9_SOURCE_MANIFEST_FILE_SHA256
        ),
        FROZEN_V9_REVIEW_RELATIVE_PATH: FROZEN_V9_REVIEW_FILE_SHA256,
        FROZEN_V9_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V9_AUTHORIZATION_FILE_SHA256
        ),
        V9_TERMINAL_AUDIT_RELATIVE_PATH: V9_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V10 source manifest")
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
        raise PermissionError("V10 source manifest contract changed")
    safe = _V9._V8._V7._V6._v5._v4._v3._v2._v1.safe_relative_source_path
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V10 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V10 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V10 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(
                f"manifest-bound V10 source changed: {binding['path']}"
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
        read = _V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
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
        "reviewed_sources", "source_manifest", "frozen_v9_source_manifest",
        "frozen_v9_source_review", "frozen_v9_execution_authorization",
        "v9_terminal_audit", "v10_preregistration", "experiment",
        "science_identity", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V10 source review fields changed")
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
        or value["frozen_v9_source_manifest"]
        != frozen_v9_source_manifest_binding()
        or value["frozen_v9_source_review"] != frozen_v9_review_binding()
        or value["frozen_v9_execution_authorization"]
        != frozen_v9_authorization_binding()
        or value["v9_terminal_audit"] != v9_terminal_audit_binding()
        or value["v10_preregistration"] != preregistration_binding()
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
        raise PermissionError("V10 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v9_source_manifest", "frozen_v9_source_review",
        "frozen_v9_execution_authorization", "v9_terminal_audit",
        "v10_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V10 execution authorization fields changed")
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
        or value["frozen_v9_source_manifest"]
        != frozen_v9_source_manifest_binding()
        or value["frozen_v9_source_review"] != frozen_v9_review_binding()
        or value["frozen_v9_execution_authorization"]
        != frozen_v9_authorization_binding()
        or value["v9_terminal_audit"] != v9_terminal_audit_binding()
        or value["v10_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V10 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V9.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v9_authorization_binding",
    "frozen_v9_review_binding",
    "frozen_v9_science_contract",
    "frozen_v9_source_manifest_binding",
    "normalize_v10_scientific_identity",
    "objective_contract",
    "perception_accounting",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v9_terminal_audit_binding",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v9_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
