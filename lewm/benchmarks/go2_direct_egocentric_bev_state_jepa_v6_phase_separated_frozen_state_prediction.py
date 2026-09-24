"""Source-only contract for Direct BEV V6 phase-separated prediction.

V6 source-loads the frozen V5 contract only for the already reviewed custody,
receipt, and failure-publication stack.  Its model and objective return to the
frozen V3/V4 science: V5's auxiliary ``A`` is absent.  The only scientific
delta is a registered two-phase optimizer policy: learn RGB-to-BEV perception,
hard-sync and freeze it, then train only the original V3 predictor.

Importing this module grants no execution, generated-input, checkpoint, tensor,
GPU, navigation, held-out, or sealed authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V5_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
_V5_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v6_phase_frozen_v5_contract",
    ROOT / FROZEN_V5_CONTRACT_RELATIVE_PATH,
)
if _V5_SPEC is None or _V5_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V5 source-only contract")
_v5 = importlib.util.module_from_spec(_V5_SPEC)
sys.modules[_V5_SPEC.name] = _v5
_V5_SPEC.loader.exec_module(_v5)

for _name in _v5.__all__:
    globals()[_name] = getattr(_v5, _name)
with_content_sha256 = _v5.with_content_sha256


IMPLEMENTATION_AUTHOR = "/root/v6_prereg_science_review"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction"
)

FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH = _v5.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V5_SOURCE_MANIFEST_COMMIT = (
    "ce64d674af4d653694689889316e87d2b9fd8397"
)
FROZEN_V5_SOURCE_MANIFEST_FILE_SHA256 = (
    "ce87211bbc288f637fe3ced7b4e1fc1d46e7c360d5df249e84aa775de0c10d6d"
)
FROZEN_V5_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "2bc5b4d435eb64af6c4044b6ac1114f0030a0b4f290dfbe9a06e14fcc22d5e32"
)
FROZEN_V5_SOURCE_MANIFEST_BYTE_COUNT = 33_870
FROZEN_V5_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V5_SOURCE_COUNT = 101

FROZEN_V5_REVIEW_RELATIVE_PATH = _v5.REVIEW_RELATIVE_PATH
FROZEN_V5_REVIEW_COMMIT = "5fa005220c9b536172ec503cf2e65c2a90994157"
FROZEN_V5_REVIEW_FILE_SHA256 = (
    "159db622fd3a52779f1418d8025f050c73bf40bed3b5054aa57d8a2443d38112"
)
FROZEN_V5_REVIEW_CONTENT_SHA256 = (
    "1b561de18b637502b1b27447ba6f950a77c99ceecdfb76b067c713439a8ab755"
)
FROZEN_V5_REVIEW_BYTE_COUNT = 57_000
FROZEN_V5_REVIEW_STATUS = (
    "PASS_SOURCE_AND_ALL_ACTIONS_STATE_DELTA_CONTRAST_SCIENCE"
)

FROZEN_V5_AUTHORIZATION_RELATIVE_PATH = _v5.AUTHORIZATION_RELATIVE_PATH
FROZEN_V5_AUTHORIZATION_COMMIT = (
    "76751cd9fcac3c297dbea7930c18f6fc026f499f"
)
FROZEN_V5_AUTHORIZATION_FILE_SHA256 = (
    "dd8459918253ab360c182ac75eb0998c9382e688f8160c89f29806d5d3833680"
)
FROZEN_V5_AUTHORIZATION_CONTENT_SHA256 = (
    "29025ab95b724bf57b8a0a9cad0ecf5ff05c4530f1112c74f87753be5868b2ff"
)
FROZEN_V5_AUTHORIZATION_BYTE_COUNT = 47_414
FROZEN_V5_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V5_ALL_ACTIONS_STATE_DELTA_CONTRAST_PROBE"
)

V5_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_terminal_audit_2026-07-26.json"
)
V5_TERMINAL_AUDIT_COMMIT = "458f590605178f1460d043a48ed629c181f593a4"
V5_TERMINAL_AUDIT_FILE_SHA256 = (
    "e4c9a329322e641b9c096ae3bc163876991e4d90c1bb24dc48146a2dd30acd20"
)
V5_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "b89afbcfaff1a703bb924f5cc028613bd927c316db5a0c1066bccae3e567526e"
)
V5_TERMINAL_AUDIT_BYTE_COUNT = 11_272
V5_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_UPDATE_100_SCIENTIFIC_FAILURE_CLOSES_V5_NO_RETRY"
)
V5_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_100_SCIENTIFIC_FAILURE_ALL_ACTIONS_STATE_DELTA_CONTRAST_"
    "FAILED_ACTION_DISCRIMINATION_AND_DEGRADED_BALANCED_PERCEPTION_V5_"
    "PERMANENTLY_CLOSED_NO_RETRY"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_preregistration_2026-07-26.md"
)
PREREGISTRATION_COMMIT = "2ec3c7a2e216544acab6f43b29b113fdc538a74f"
PREREGISTRATION_FILE_SHA256 = (
    "e71dac233d89aa49e97998afdeaadc6c806671945e25101ceb078fcbac0af4e7"
)
PREREGISTRATION_BYTE_COUNT = 14_618

FROZEN_V5_RUNNER_RELATIVE_PATH = _v5.RUNNER_RELATIVE_PATH
FROZEN_V5_LAUNCHER_RELATIVE_PATH = _v5.LAUNCHER_RELATIVE_PATH
FROZEN_V5_MODEL_RELATIVE_PATH = _v5.MODEL_RELATIVE_PATH
FROZEN_V5_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v5.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_source_closure.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_runner.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_launch_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_execution_authorization_2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
REVIEW_STATUS = "PASS_SOURCE_AND_PHASE_SEPARATED_FROZEN_STATE_PREDICTION_SCIENCE"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V6_PHASE_SEPARATED_FROZEN_STATE_"
    "PREDICTION_PROBE"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    LAUNCHER_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v5.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V5_REVIEW_RELATIVE_PATH,
    FROZEN_V5_AUTHORIZATION_RELATIVE_PATH,
    V5_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v6/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v6_"
    "phase_separated_frozen_state_prediction"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V6_"
    "PHASE_SEPARATED_FROZEN_STATE_PREDICTION_PREFLIGHT_JSON"
)

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **_v5.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "all_actions_state_delta_contrast_loss_only": False,
    "v5_retry_authorized": False,
    "v5_checkpoint_tensor_trace_or_runtime_output_reuse_authorized": False,
    "phase_separated_frozen_state_prediction_only": True,
}

PREDICTOR_CONFIG = _v5._v4.PREDICTOR_CONFIG
MODEL_PARAMETER_INVENTORY = _v5._v4.MODEL_PARAMETER_INVENTORY
FROZEN_V3_INITIAL_MODEL_STATE_SHA256 = (
    _v5._v4.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
)

PHASE_ACCOUNTING = {
    0: {
        "target_update_callback_count": 0,
        "perception_optimizer_updates": 0,
        "predictor_optimizer_updates": 0,
        "ema_arithmetic_updates": 0,
        "boundary_hard_sync_count": 0,
        "phase_two_target_noop_count": 0,
    },
    100: {
        "target_update_callback_count": 100,
        "perception_optimizer_updates": 100,
        "predictor_optimizer_updates": 0,
        "ema_arithmetic_updates": 100,
        "boundary_hard_sync_count": 0,
        "phase_two_target_noop_count": 0,
    },
    400: {
        "target_update_callback_count": 400,
        "perception_optimizer_updates": 400,
        "predictor_optimizer_updates": 0,
        "ema_arithmetic_updates": 400,
        "boundary_hard_sync_count": 1,
        "phase_two_target_noop_count": 0,
    },
    1_000: {
        "target_update_callback_count": 1_000,
        "perception_optimizer_updates": 400,
        "predictor_optimizer_updates": 600,
        "ema_arithmetic_updates": 400,
        "boundary_hard_sync_count": 1,
        "phase_two_target_noop_count": 600,
    },
}

PHASE_ADAPTER_CONFIG = {
    "base_model": "fresh_frozen_v3_coordinate_aware_film_unet_predictor",
    "parameter_buffer_or_module_addition": False,
    "public_seams": [
        "active_phase_v6",
        "phase_counters_v6",
        "set_phase_override_for_integrity_probe_v6",
        "apply_phase_policy_v6",
    ],
    "phase_one": {
        "updates": [1, 400],
        "presentations": [1, 6_400],
        "total": "G/log(2)",
        "trainable": "online_encoder_decoder_state",
        "frozen": "predictor_and_detached_target",
        "target_callback": "ema_0point996",
    },
    "boundary": {
        "after_ema_update": 400,
        "online_to_target_hard_sync_count": 1,
        "reset_global_update_counter": False,
    },
    "phase_two": {
        "updates": [401, 1_000],
        "presentations": [6_401, 16_000],
        "total": "J/log(2)+C",
        "trainable": "predictor_only",
        "frozen": "online_and_target_perception",
        "perception_mode": "deterministic_eval",
        "target_callback": "counted_noop",
    },
    "optimizer": "one_frozen_v3_adamw_constructed_once_never_reset",
    "v5_A_present": False,
}

GATE_THRESHOLDS = {
    0: {},
    100: {
        "G_update_zero_factor_strictly_less_than": 0.90,
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.65,
        "aggregate_raster_nll_strictly_less_than_update_zero": True,
        "rough_raster_balanced_accuracy_strictly_greater_than_update_zero": True,
        "correct_rgb_scene_wins_minimum": 6,
    },
    400: {
        "aggregate_raster_balanced_accuracy_strictly_greater_than": (
            0.9009460724448773
        ),
        "aggregate_free_recall_strictly_greater_than": 0.91637020862468,
        "aggregate_occupied_recall_strictly_greater_than": 0.8059679976935274,
        "aggregate_raster_nll_strictly_less_than": 0.18704089070408247,
        "rough_raster_balanced_accuracy_strictly_greater_than": (
            0.7719525130620232
        ),
        "rough_raster_occupied_recall_strictly_greater_than": (
            0.4319466882067851
        ),
        "correct_rgb_scene_wins_minimum": 8,
    },
    1_000: {
        "J_update400_factor_maximum_inclusive": 0.90,
        "C_strictly_less_than_update400": True,
        "action_nll_strictly_less_than": 0.95 * math.log(9.0),
        "action_macro_balanced_accuracy_strictly_greater_than": 2.0 / 9.0,
        "hardest_wrong_positive_scene_count_minimum": 6,
        "same_action_target_nll_strictly_less_than": 0.95 * math.log(2.0),
        "same_action_target_strict_win_rate_minimum": 0.65,
        "target_positive_scene_count_minimum": 6,
        "correct_rgb_scene_wins_minimum": 8,
    },
}

CONTROL_UPDATE_ZERO_FAIL = (
    "FAIL_UPDATE_ZERO_V6_PHASE_INTEGRITY_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_ZERO = "CONTINUE_AFTER_UPDATE_ZERO_V6_PHASE_GATE"
CONTROL_UPDATE_100_FAIL = (
    "FAIL_UPDATE_100_V6_PERCEPTION_DIRECTIONAL_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_100 = "CONTINUE_AFTER_UPDATE_100_V6_PERCEPTION_GATE"
CONTROL_UPDATE_400_FAIL = (
    "FAIL_UPDATE_400_V6_PERCEPTION_BOUNDARY_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_400 = (
    "CONTINUE_AFTER_UPDATE_400_V6_PHASE_BOUNDARY_GATE"
)
CONTROL_UPDATE_1000_FAIL = (
    "FAIL_UPDATE_1000_V6_PREDICTOR_QUALIFICATION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_PASS = (
    "PASS_DIRECT_BEV_V6_PHASE_SEPARATED_FROZEN_STATE_PREDICTION_GATE_"
    "REQUALIFICATION_ONLY"
)
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_ZERO_FAIL, CONTROL_CONTINUE_UPDATE_ZERO),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    400: (CONTROL_UPDATE_400_FAIL, CONTROL_CONTINUE_UPDATE_400),
    1_000: (CONTROL_UPDATE_1000_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

SCIENTIFIC_DELTA = {
    "scope": "phase_separated_frozen_state_prediction",
    "phase_one_total": "G/log(2)",
    "boundary": "one_online_to_target_hard_sync_after_ema_update_400",
    "phase_two_total": "J/log(2)+C",
    "v5_all_actions_state_delta_contrast_A": "absent",
    "model_architecture_parameter_inventory_and_initialization_delta": False,
    "data_seed_schedule_optimizer_hyperparameter_delta": False,
    "shared_gradient_or_moving_phase_two_perception": False,
    "prior_checkpoint_tensor_trace_or_runtime_output_reuse": False,
}

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v5_manifest_and_all_101_sources_rehashed": True,
    "frozen_v5_review_authorization_and_terminal_audit_exact": True,
    "v5_permanently_closed_and_no_runtime_reuse": True,
    "v6_preregistration_exact": True,
    "fresh_v3_model_inventory_and_initial_state_exact": True,
    "v5_A_absent_and_only_v3_G_J_C_present": True,
    "phase_one_G_only_and_phase_two_predictor_only": True,
    "boundary_sync_modes_stationarity_and_accounting_exact": True,
    "preliminary_mode_is_not_v6_phase_evidence": True,
    "final_u0_u100_u400_u1000_gates_fail_closed": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v5_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V5_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V5_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V5_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V5_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V5_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V5_SOURCE_COUNT,
    }


def frozen_v5_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V5_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V5_REVIEW_COMMIT,
        "file_sha256": FROZEN_V5_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V5_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V5_REVIEW_BYTE_COUNT,
        "status": FROZEN_V5_REVIEW_STATUS,
    }


def frozen_v5_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V5_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V5_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V5_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V5_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V5_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V5_AUTHORIZATION_STATUS,
    }


def v5_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V5_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V5_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V5_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V5_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V5_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V5_TERMINAL_AUDIT_STATUS,
        "classification": V5_TERMINAL_AUDIT_CLASSIFICATION,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def model_config() -> dict[str, Any]:
    value = _v5._v4.model_config()
    value["optimization_phase_adapter"] = dict(PHASE_ADAPTER_CONFIG)
    return value


def objective_contract() -> dict[str, Any]:
    value = _v5._v4.objective_contract()
    value.pop("A", None)
    value.pop("C_v3", None)
    value["phase_one_total"] = "G/log(2)"
    value["phase_two_total"] = "J/log(2)+C"
    value["total"] = "phase_selected(phase_one_total,phase_two_total)"
    value["v5_all_actions_state_delta_contrast_A"] = "absent"
    return value


def optimizer_contract() -> dict[str, Any]:
    return _v5._v4.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v5._v4.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v5._v4.runtime_authorization_template()


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


def _phase_accounting_conjuncts(
    update: int, metrics: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        f"{field}_equals_{expected}": (
            _exact_int(metrics.get(field), name=field) == expected
        )
        for field, expected in PHASE_ACCOUNTING[update].items()
    }


def _sha256_conjunct(value: object, *, name: str) -> bool:
    if type(value) is not str:
        raise ValueError(f"{name} must be one SHA-256 string")
    return is_sha256(value)


def _perception_threshold_conjuncts(
    metrics: Mapping[str, Any], *, update: int
) -> dict[str, bool]:
    thresholds = GATE_THRESHOLDS[400]
    return {
        f"u{update}_aggregate_raster_balanced_accuracy_qualified": (
            _finite_number(
                metrics.get("aggregate_raster_balanced_accuracy"),
                name="aggregate_raster_balanced_accuracy",
            )
            > thresholds[
                "aggregate_raster_balanced_accuracy_strictly_greater_than"
            ]
        ),
        f"u{update}_aggregate_free_recall_qualified": (
            _finite_number(
                metrics.get("aggregate_free_recall"),
                name="aggregate_free_recall",
            )
            > thresholds["aggregate_free_recall_strictly_greater_than"]
        ),
        f"u{update}_aggregate_occupied_recall_qualified": (
            _finite_number(
                metrics.get("aggregate_occupied_recall"),
                name="aggregate_occupied_recall",
            )
            > thresholds["aggregate_occupied_recall_strictly_greater_than"]
        ),
        f"u{update}_aggregate_raster_nll_qualified": (
            _finite_number(
                metrics.get("aggregate_raster_nll"),
                name="aggregate_raster_nll",
            )
            < thresholds["aggregate_raster_nll_strictly_less_than"]
        ),
        f"u{update}_rough_raster_balanced_accuracy_qualified": (
            _finite_number(
                metrics.get("rough_raster_balanced_accuracy"),
                name="rough_raster_balanced_accuracy",
            )
            > thresholds[
                "rough_raster_balanced_accuracy_strictly_greater_than"
            ]
        ),
        f"u{update}_rough_raster_occupied_recall_qualified": (
            _finite_number(
                metrics.get("rough_raster_occupied_recall"),
                name="rough_raster_occupied_recall",
            )
            > thresholds[
                "rough_raster_occupied_recall_strictly_greater_than"
            ]
        ),
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate preliminary inherited gates or exact fail-closed V6 gates.

    Absence of ``v6_phase_receipt_ready`` is an explicit source-integration
    compatibility mode.  It is labelled preliminary and is never evidence for
    V6 execution review.  Once the field is present, it must be exactly true and
    every phase-specific receipt conjunct is mandatory.
    """

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    _exact_bool(prior_gates_passed, name="prior_gates_passed")
    if "v6_phase_receipt_ready" not in metrics:
        inherited = _v5._v4.evaluate_gate(
            update,
            metrics,
            update_zero=update_zero,
            prior_gates_passed=prior_gates_passed,
        )
        result = dict(inherited)
        result["gate_mode"] = "PRELIMINARY_INHERITED_V3_NOT_V6_PHASE_EVIDENCE"
        result["v6_phase_receipt_ready"] = False
        result["control"] = GATE_CONTROLS[update][1 if result["passed"] else 0]
        return result

    ready = _exact_bool(
        metrics.get("v6_phase_receipt_ready"), name="v6_phase_receipt_ready"
    )
    conjuncts: dict[str, bool] = {
        "prior_gates_passed": prior_gates_passed,
        "v6_phase_receipt_ready": ready,
    }
    expected_phase = "phase_one" if update in (0, 100) else "phase_two"
    active_phase = metrics.get("active_phase_v6")
    if type(active_phase) is not str:
        raise ValueError("active_phase_v6 must be str")
    conjuncts["active_phase_v6_exact"] = active_phase == expected_phase
    conjuncts.update(_phase_accounting_conjuncts(update, metrics))

    if update == 0:
        for field in (
            "initial_model_state_matches_frozen_v3",
            "model_parameter_inventory_exact",
            "three_logit_bottleneck_exact",
            "no_hidden_or_auxiliary_bypass",
            "prediction_is_exact_persistence",
            "all_nine_action_predictions_bitwise_equal",
            "target_parameters_gradient_free",
            "intended_online_path_gradient_nonzero",
            "six_call_graph_isolation_exact",
            "all_registered_values_finite",
            "state_nonconstant",
            "registered_state_and_target_nonconstant",
            "no_prior_runtime_or_protected_input",
            "phase_one_trainability_exact",
            "phase_one_gradient_isolation_exact",
            "phase_two_gradient_isolation_exact",
            "dual_gradient_probe_nonmutating_exact",
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
        conjuncts.update({
            "G_below_point90_update_zero": (
                _finite_number(metrics.get("G"), name="G")
                < 0.90 * _finite_number(update_zero.get("G"), name="update_zero.G")
            ),
            "aggregate_raster_balanced_accuracy_at_least_point65": (
                _finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.65
            ),
            "aggregate_raster_nll_strictly_lower": (
                _finite_number(
                    metrics.get("aggregate_raster_nll"),
                    name="aggregate_raster_nll",
                )
                < _finite_number(
                    update_zero.get("aggregate_raster_nll"),
                    name="update_zero.aggregate_raster_nll",
                )
            ),
            "rough_raster_balanced_accuracy_strictly_higher": (
                _finite_number(
                    metrics.get("rough_raster_balanced_accuracy"),
                    name="rough_raster_balanced_accuracy",
                )
                > _finite_number(
                    update_zero.get("rough_raster_balanced_accuracy"),
                    name="update_zero.rough_raster_balanced_accuracy",
                )
            ),
            "correct_rgb_wins_at_least_six_scenes": (
                _finite_number(
                    metrics.get("correct_rgb_scene_win_count"),
                    name="correct_rgb_scene_win_count",
                ) >= 6
            ),
        })
        for field in (
            "all_registered_values_finite",
            "state_nonconstant",
            "predictor_matches_initialization",
            "predictor_residual_head_exact_zero",
            "prediction_is_exact_persistence",
            "all_nine_action_predictions_bitwise_equal",
            "phase_one_trainability_exact",
        ):
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        action_nll = _finite_number(metrics.get("action_nll"), name="action_nll")
        action_ba = _finite_number(
            metrics.get("action_macro_balanced_accuracy"),
            name="action_macro_balanced_accuracy",
        )
        conjuncts["action_nll_remains_log9"] = (
            abs(action_nll - math.log(9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
        conjuncts["action_macro_balanced_accuracy_remains_one_ninth"] = (
            abs(action_ba - (1.0 / 9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
        conjuncts["predictor_state_sha256_valid"] = _sha256_conjunct(
            metrics.get("predictor_state_sha256"), name="predictor_state_sha256"
        )
    elif update == 400:
        conjuncts.update(_perception_threshold_conjuncts(metrics, update=400))
        conjuncts["correct_rgb_wins_all_eight_scenes"] = (
            _finite_number(
                metrics.get("correct_rgb_scene_win_count"),
                name="correct_rgb_scene_win_count",
            ) >= 8
        )
        for field in (
            "all_registered_values_finite",
            "state_nonconstant",
            "predictor_matches_initialization",
            "predictor_residual_head_exact_zero",
            "prediction_is_exact_persistence",
            "all_nine_action_predictions_bitwise_equal",
            "online_target_perception_bitwise_equal",
            "phase_two_trainability_exact",
            "online_perception_eval_mode",
            "target_perception_eval_mode",
            "predictor_train_mode",
            "phase_two_module_modes_exact",
            "zero_rgb_online_repeat_bitwise_equal",
            "zero_rgb_target_repeat_bitwise_equal",
            "zero_rgb_witness_exact",
            "boundary_phase_two_gradient_isolation_exact",
        ):
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        action_nll = _finite_number(metrics.get("action_nll"), name="action_nll")
        action_ba = _finite_number(
            metrics.get("action_macro_balanced_accuracy"),
            name="action_macro_balanced_accuracy",
        )
        conjuncts["action_nll_remains_log9"] = (
            abs(action_nll - math.log(9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
        conjuncts["action_macro_balanced_accuracy_remains_one_ninth"] = (
            abs(action_ba - (1.0 / 9.0)) <= UPDATE_ZERO_ACTION_TOLERANCE
        )
        online_sha = metrics.get("online_perception_state_sha256")
        target_sha = metrics.get("target_perception_state_sha256")
        predictor_sha = metrics.get("predictor_state_sha256")
        predictor_boundary_sha = metrics.get("predictor_update400_sha256")
        conjuncts.update({
            "online_perception_state_sha256_valid": _sha256_conjunct(
                online_sha, name="online_perception_state_sha256"
            ),
            "target_perception_state_sha256_valid": _sha256_conjunct(
                target_sha, name="target_perception_state_sha256"
            ),
            "online_target_perception_sha256_equal": online_sha == target_sha,
            "predictor_state_sha256_valid": _sha256_conjunct(
                predictor_sha, name="predictor_state_sha256"
            ),
            "predictor_update400_sha256_valid": _sha256_conjunct(
                predictor_boundary_sha, name="predictor_update400_sha256"
            ),
            "predictor_update400_sha256_matches": (
                predictor_boundary_sha == predictor_sha
            ),
            "perception_metrics_update400_baseline_sha256_valid": (
                _sha256_conjunct(
                    metrics.get("perception_metrics_update400_baseline_sha256"),
                    name="perception_metrics_update400_baseline_sha256",
                )
            ),
            "J_update400_boundary_recorded": (
                _finite_number(
                    metrics.get("J_update400_boundary"),
                    name="J_update400_boundary",
                ) == _finite_number(metrics.get("J"), name="J")
            ),
            "C_update400_boundary_recorded": (
                _finite_number(
                    metrics.get("C_update400_boundary"),
                    name="C_update400_boundary",
                ) == _finite_number(metrics.get("C"), name="C")
            ),
        })
    else:
        conjuncts.update(_perception_threshold_conjuncts(metrics, update=1_000))
        for field in (
            "all_registered_values_finite",
            "state_nonconstant",
            "online_target_perception_bitwise_equal",
            "online_perception_unchanged_from_update400",
            "target_perception_unchanged_from_update400",
            "perception_metrics_unchanged_from_update400",
            "phase_two_trainability_exact",
            "online_perception_eval_mode",
            "target_perception_eval_mode",
            "predictor_train_mode",
            "phase_two_module_modes_exact",
            "zero_rgb_online_repeat_bitwise_equal",
            "zero_rgb_target_repeat_bitwise_equal",
            "zero_rgb_witness_exact",
        ):
            conjuncts[field] = _exact_bool(metrics.get(field), name=field)
        online_sha = metrics.get("online_perception_state_sha256")
        target_sha = metrics.get("target_perception_state_sha256")
        conjuncts.update({
            "online_perception_state_sha256_valid": _sha256_conjunct(
                online_sha, name="online_perception_state_sha256"
            ),
            "target_perception_state_sha256_valid": _sha256_conjunct(
                target_sha, name="target_perception_state_sha256"
            ),
            "online_target_perception_sha256_equal": online_sha == target_sha,
            "predictor_state_sha256_valid": _sha256_conjunct(
                metrics.get("predictor_state_sha256"),
                name="predictor_state_sha256",
            ),
            "predictor_update400_sha256_valid": _sha256_conjunct(
                metrics.get("predictor_update400_sha256"),
                name="predictor_update400_sha256",
            ),
            "perception_metrics_update400_baseline_sha256_valid": (
                _sha256_conjunct(
                    metrics.get("perception_metrics_update400_baseline_sha256"),
                    name="perception_metrics_update400_baseline_sha256",
                )
            ),
        })
        j_boundary = _finite_number(
            metrics.get("J_update400_boundary"), name="J_update400_boundary"
        )
        c_boundary = _finite_number(
            metrics.get("C_update400_boundary"), name="C_update400_boundary"
        )
        conjuncts.update({
            "J_at_most_point90_update400_boundary": (
                _finite_number(metrics.get("J"), name="J") <= 0.90 * j_boundary
            ),
            "C_strictly_lower_than_update400_boundary": (
                _finite_number(metrics.get("C"), name="C") < c_boundary
            ),
            "action_nll_below_point95_log9": (
                _finite_number(metrics.get("action_nll"), name="action_nll")
                < GATE_THRESHOLDS[1_000]["action_nll_strictly_less_than"]
            ),
            "action_macro_balanced_accuracy_above_two_ninths": (
                _finite_number(
                    metrics.get("action_macro_balanced_accuracy"),
                    name="action_macro_balanced_accuracy",
                ) > 2.0 / 9.0
            ),
            "hardest_wrong_positive_at_least_six_scenes": (
                _finite_number(
                    metrics.get("hardest_wrong_positive_scene_count"),
                    name="hardest_wrong_positive_scene_count",
                ) >= 6
            ),
            "same_action_target_nll_below_point95_log2": (
                _finite_number(
                    metrics.get("same_action_target_nll"),
                    name="same_action_target_nll",
                ) < GATE_THRESHOLDS[1_000][
                    "same_action_target_nll_strictly_less_than"
                ]
            ),
            "same_action_target_strict_win_rate_at_least_point65": (
                _finite_number(
                    metrics.get("same_action_target_strict_win_rate"),
                    name="same_action_target_strict_win_rate",
                ) >= 0.65
            ),
            "target_positive_at_least_six_scenes": (
                _finite_number(
                    metrics.get("target_positive_scene_count"),
                    name="target_positive_scene_count",
                ) >= 6
            ),
            "correct_rgb_wins_all_eight_scenes": (
                _finite_number(
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
        "gate_mode": "FINAL_V6_PHASE_RECEIPT",
        "v6_phase_receipt_ready": ready,
        "conjuncts": conjuncts,
        "thresholds": dict(GATE_THRESHOLDS[update]),
        "phase_accounting": dict(PHASE_ACCOUNTING[update]),
    }


def validate_failure_status_chain(value: object) -> dict[str, str]:
    """Validate only exact V6 failure controls, never inherited lexical state."""

    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("failure receipt statuses are not one exact V6 gate control")
    return dict(value)


def science_contract() -> dict[str, Any]:
    value = _v5._v4.science_contract()
    value.pop("integrity_replacement", None)
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Can the original V3 action predictor learn useful dynamics when RGB-to-BEV "
        "perception is first qualified, synchronized, and then held stationary?"
    )
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v5_source_manifest": frozen_v5_source_manifest_binding(),
        "frozen_v5_source_review": frozen_v5_review_binding(),
        "frozen_v5_execution_authorization": frozen_v5_authorization_binding(),
        "v5_terminal_audit": v5_terminal_audit_binding(),
        "v6_preregistration": preregistration_binding(),
    }
    value["model"] = model_config()
    value["objective"] = objective_contract()
    value["optimizer"] = optimizer_contract()
    value["schedule"] = build_schedule_identity()
    value["gates"] = {
        "updates": [0, 100, 400, 1_000],
        "thresholds": {
            str(update): dict(thresholds)
            for update, thresholds in GATE_THRESHOLDS.items()
        },
        "controls": {
            str(update): list(controls)
            for update, controls in GATE_CONTROLS.items()
        },
        "phase_accounting": {
            str(update): dict(accounting)
            for update, accounting in PHASE_ACCOUNTING.items()
        },
        "preliminary_mode_authorizes_execution": False,
        "final_mode_requires_v6_phase_receipt_ready": True,
        "stop_at_first_failed_gate": True,
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "phase_successor_of": _v5.EXPERIMENT_ID,
        "one_fresh_attempt": True,
        "maximum_updates": 1_000,
        "maximum_presentations": 16_000,
        "maximum_active_gpu_minutes": 60,
        "v5_retry": False,
        "v5_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    }
    value["phase_successor"] = dict(SCIENTIFIC_DELTA)
    value["phase_adapter"] = dict(PHASE_ADAPTER_CONFIG)
    value["authority"] = {
        **value["authority"],
        "v6_execution_authorized_by_source_contract": False,
        "v5_checkpoint_or_runtime_output_reuse_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "promotion_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    return value


def validate_frozen_v5_source_closure(root: Path = ROOT) -> dict[str, str]:
    read = _v5._v4._v3._v2._v1._read_regular_source
    raw = read(root / FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH)
    if (
        len(raw) != FROZEN_V5_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest()
        != FROZEN_V5_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V5 source manifest raw identity changed")
    manifest = _v5.validate_source_manifest(raw)
    if (
        manifest.get("content_sha256")
        != FROZEN_V5_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != FROZEN_V5_SOURCE_MANIFEST_STATUS
        or manifest.get("source_count") != FROZEN_V5_SOURCE_COUNT
        or manifest.get("source_paths") != list(REUSED_SOURCE_PATHS)
    ):
        raise PermissionError("frozen V5 source manifest conclusion changed")
    current = _v5.current_source_bindings(root)
    if current.get(FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V5_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("current V5 source manifest changed")
    for binding in manifest["source_bindings"]:
        if current.get(binding["path"]) != binding["file_sha256"]:
            raise PermissionError(f"current V5 source changed: {binding['path']}")
    return current


def _read_and_validate_frozen_v5_review_and_authorization(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    read = _v5._v4._v3._v2._v1._read_regular_source
    review_raw = read(root / FROZEN_V5_REVIEW_RELATIVE_PATH)
    authorization_raw = read(root / FROZEN_V5_AUTHORIZATION_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V5_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V5_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V5 source review raw identity changed")
    if (
        len(authorization_raw) != FROZEN_V5_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V5_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V5 authorization raw identity changed")
    review = _v5.parse_canonical_json(review_raw, name="frozen V5 source review")
    authorization = _v5.parse_canonical_json(
        authorization_raw, name="frozen V5 authorization"
    )
    if (
        review.get("content_sha256") != FROZEN_V5_REVIEW_CONTENT_SHA256
        or review.get("status") != FROZEN_V5_REVIEW_STATUS
        or authorization.get("content_sha256")
        != FROZEN_V5_AUTHORIZATION_CONTENT_SHA256
        or authorization.get("status") != FROZEN_V5_AUTHORIZATION_STATUS
    ):
        raise PermissionError("frozen V5 review or authorization conclusion changed")
    _v5.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _v5.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    return dict(review), dict(authorization)


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = validate_frozen_v5_source_closure(root)
    _read_and_validate_frozen_v5_review_and_authorization(root)
    read = _v5._v4._v3._v2._v1._read_regular_source
    audit_raw = read(root / V5_TERMINAL_AUDIT_RELATIVE_PATH)
    preregistration_raw = read(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(audit_raw) != V5_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest() != V5_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V5 terminal audit raw identity changed")
    if (
        len(preregistration_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(preregistration_raw).hexdigest()
        != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V6 preregistration changed")
    audit = json.loads(audit_raw)
    core = dict(audit)
    declared = core.pop("content_sha256", None)
    accounting = audit.get("execution_accounting", {})
    consequence = audit.get("scientific_consequence", {})
    if (
        declared != V5_TERMINAL_AUDIT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
        or audit.get("status") != V5_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V5_TERMINAL_AUDIT_CLASSIFICATION
        or accounting.get("updates") != 100
        or accounting.get("presentations") != 1_600
        or accounting.get("optimizer_updates") != 100
        or accounting.get("ema_updates") != 100
        or consequence.get("v5_permanently_closed") is not True
        or consequence.get("do_not_extend_relax_or_repeat_v5") is not True
        or consequence.get("v5_checkpoint_qualified") is not False
        or consequence.get(
            "next_successor_must_be_fresh_and_may_not_reuse_v5_checkpoint_"
            "tensor_trace_or_runtime_output"
        ) is not True
    ):
        raise PermissionError("V5 terminal audit conclusion changed")
    result.update({
        FROZEN_V5_REVIEW_RELATIVE_PATH: FROZEN_V5_REVIEW_FILE_SHA256,
        FROZEN_V5_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V5_AUTHORIZATION_FILE_SHA256
        ),
        V5_TERMINAL_AUDIT_RELATIVE_PATH: V5_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = _v5.parse_canonical_json(raw, name="V6 source manifest")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    expected_fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != expected_fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources")
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != 111
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V6 source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V6 source binding fields changed")
        relative = _v5._v4._v3._v2._v1.safe_relative_source_path(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V6 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V6 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V6 source changed: {binding['path']}")
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(manifest_raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        raw = _v5._v4._v3._v2._v1._read_regular_source(
            ROOT / SOURCE_MANIFEST_RELATIVE_PATH
        )
        value = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(value["content_sha256"]),
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
        "reviewed_sources", "source_manifest", "frozen_v5_source_manifest",
        "frozen_v5_source_review", "frozen_v5_execution_authorization",
        "v5_terminal_audit", "v6_preregistration", "science_contract",
        "source_only_checks", "scientific_checks", "findings", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V6 source review fields changed")
    manifest_binding = _manifest_binding_or_read(source_manifest_binding)
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    required_reviewed = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != REVIEW_STATUS
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not required_reviewed.issubset(expected_sources)
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"] != manifest_binding
        or expected_sources.get(FROZEN_V5_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V5_SOURCE_MANIFEST_FILE_SHA256
        or expected_sources.get(FROZEN_V5_REVIEW_RELATIVE_PATH)
        != FROZEN_V5_REVIEW_FILE_SHA256
        or expected_sources.get(FROZEN_V5_AUTHORIZATION_RELATIVE_PATH)
        != FROZEN_V5_AUTHORIZATION_FILE_SHA256
        or expected_sources.get(V5_TERMINAL_AUDIT_RELATIVE_PATH)
        != V5_TERMINAL_AUDIT_FILE_SHA256
        or expected_sources.get(PREREGISTRATION_RELATIVE_PATH)
        != PREREGISTRATION_FILE_SHA256
        or value["frozen_v5_source_manifest"]
        != frozen_v5_source_manifest_binding()
        or value["frozen_v5_source_review"] != frozen_v5_review_binding()
        or value["frozen_v5_execution_authorization"]
        != frozen_v5_authorization_binding()
        or value["v5_terminal_audit"] != v5_terminal_audit_binding()
        or value["v6_preregistration"] != preregistration_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
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
        raise PermissionError("V6 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v5_source_manifest", "frozen_v5_source_review",
        "frozen_v5_execution_authorization", "v5_terminal_audit",
        "v6_preregistration", "runtime_inputs", "experiment", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V6 execution authorization fields changed")
    expected_review = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["frozen_v5_source_manifest"]
        != frozen_v5_source_manifest_binding()
        or value["frozen_v5_source_review"] != frozen_v5_review_binding()
        or value["frozen_v5_execution_authorization"]
        != frozen_v5_authorization_binding()
        or value["v5_terminal_audit"] != v5_terminal_audit_binding()
        or value["v6_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V6 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v5.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v5_authorization_binding",
    "frozen_v5_review_binding",
    "frozen_v5_source_manifest_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "v5_terminal_audit_binding",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v5_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
    "with_content_sha256",
})
