"""Source-only contract for the Direct BEV V5 state-delta contrast probe.

V5 source-loads the permanently closed V4 contract.  The learned model,
initialization, data, optimizer, schedule, observations, and V3 thresholds stay
frozen.  Its sole scientific change is the preregistered all-actions learned
state-delta contrast ``A``, with ``C_v5 = C_v3 + A``.

Importing this module grants no execution, generated-input, checkpoint, tensor,
GPU, navigation, held-out, or sealed authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V4_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
_V4_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v5_delta_contrast_frozen_v4_contract",
    ROOT / FROZEN_V4_CONTRACT_RELATIVE_PATH,
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V4 source-only contract")
_v4 = importlib.util.module_from_spec(_V4_SPEC)
sys.modules[_V4_SPEC.name] = _v4
_V4_SPEC.loader.exec_module(_v4)

for _name in _v4.__all__:
    globals()[_name] = getattr(_v4, _name)
with_content_sha256 = _v4.with_content_sha256


IMPLEMENTATION_AUTHOR = "/root/v4_failure_chain_diagnosis"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast"
)

FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH = _v4.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V4_SOURCE_MANIFEST_COMMIT = (
    "d82e386c1b846442fa4f2f66d6233ca98380fd74"
)
FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256 = (
    "299a54b683d6926cf2cec4d3887991d4b5df53b0540a10d936c06679d2dc6d98"
)
FROZEN_V4_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "f78fbed617f7f1319ededbf5ae76cb5f822d146c9d407c10f4379052e6cb97be"
)
FROZEN_V4_SOURCE_MANIFEST_BYTE_COUNT = 29_853
FROZEN_V4_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V4_SOURCE_COUNT = 91

FROZEN_V4_REVIEW_RELATIVE_PATH = _v4.REVIEW_RELATIVE_PATH
FROZEN_V4_REVIEW_COMMIT = "478ca4845249f2ab1d79e09c31c46e88a25c5c89"
FROZEN_V4_REVIEW_FILE_SHA256 = (
    "41efdcb4d1cfd7ff60252a396f3d97ec86da986ad01ba48c8ac998acd00aaa7f"
)
FROZEN_V4_REVIEW_CONTENT_SHA256 = (
    "0a0de65fdadfcc51aae3ca5e6c154bf168e7f4a56e8417f14c475fd29b614f59"
)
FROZEN_V4_REVIEW_BYTE_COUNT = 49_781
FROZEN_V4_REVIEW_STATUS = "PASS_SOURCE_AND_SCIENCE_IDENTICAL_HOOK_INTEGRITY"

FROZEN_V4_AUTHORIZATION_RELATIVE_PATH = _v4.AUTHORIZATION_RELATIVE_PATH
FROZEN_V4_AUTHORIZATION_COMMIT = (
    "d9cc2fad0c1f953487756b34226e03a9607f8d3e"
)
FROZEN_V4_AUTHORIZATION_FILE_SHA256 = (
    "18bf0f511c816d2d4155262069bc9b262344c9cc5ea7b123023524e267679273"
)
FROZEN_V4_AUTHORIZATION_CONTENT_SHA256 = (
    "f9ae84e1a9abb36daff35bfb11765010565c26c6b295530b524f186f1029659f"
)
FROZEN_V4_AUTHORIZATION_BYTE_COUNT = 42_722
FROZEN_V4_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V4_HOOK_INTEGRITY_PROBE"
)

V4_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_terminal_audit_2026-07-26.json"
)
V4_TERMINAL_AUDIT_COMMIT = "dcd509d9ded153d07c6a4513da328c92398d1b7c"
V4_TERMINAL_AUDIT_FILE_SHA256 = (
    "94d1a2f15e43d8d04f7f1e6941ae5ce5da4499f7452c297a7b5badadc673fcb2"
)
V4_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "c4e04e181c713c16c14a4fcf259ede41d160bf1ba5b56e31705ba2eaff88d5ed"
)
V4_TERMINAL_AUDIT_BYTE_COUNT = 12_147
V4_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_UPDATE_100_SCIENTIFIC_PREDICTOR_GATE_FAILURE_AND_POST_GATE_"
    "RECEIPT_PACKAGING_FAILURE_CLOSES_V4_NO_RETRY"
)
V4_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_100_SCIENTIFIC_FAILURE_STRONG_PERCEPTION_WEAK_ACTION_SIGNAL_"
    "THEN_STALE_LEXICAL_FAILURE_CONTROL_VALIDATOR_OPERATIONAL_ERROR_V4_"
    "PERMANENTLY_CLOSED"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_preregistration_2026-07-26.md"
)
PREREGISTRATION_COMMIT = "5b503a27b1f3ee6f94b0e9ba1cde339b0d007bb8"
PREREGISTRATION_FILE_SHA256 = (
    "215de2dd0978862acf1f527778642a7151abbe75c35ff30d9ee875b196477ad9"
)
PREREGISTRATION_BYTE_COUNT = 7_007

FROZEN_V4_RUNNER_RELATIVE_PATH = _v4.RUNNER_RELATIVE_PATH
FROZEN_V4_LAUNCHER_RELATIVE_PATH = _v4.LAUNCHER_RELATIVE_PATH
FROZEN_V4_MODEL_RELATIVE_PATH = _v4.MODEL_RELATIVE_PATH
FROZEN_V4_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v4.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_source_closure.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_runner.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_launch_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast_execution_authorization_2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
REVIEW_STATUS = "PASS_SOURCE_AND_ALL_ACTIONS_STATE_DELTA_CONTRAST_SCIENCE"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V5_ALL_ACTIONS_STATE_DELTA_CONTRAST_PROBE"
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
REUSED_SOURCE_PATHS = tuple(sorted(set(_v4.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V4_REVIEW_RELATIVE_PATH,
    FROZEN_V4_AUTHORIZATION_RELATIVE_PATH,
    V4_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v5_"
    "all_actions_state_delta_contrast"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V5_"
    "ALL_ACTIONS_STATE_DELTA_CONTRAST_PREFLIGHT_JSON"
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
    **_v4.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "science_identical_hook_integrity_replacement_only": False,
    "v4_retry_authorized": False,
    "v4_checkpoint_tensor_trace_or_runtime_output_reuse_authorized": False,
    "all_actions_state_delta_contrast_loss_only": True,
}

# The model and every non-objective scientific component remain exact V3/V4.
PREDICTOR_CONFIG = _v4.PREDICTOR_CONFIG
MODEL_PARAMETER_INVENTORY = _v4.MODEL_PARAMETER_INVENTORY
UPDATE_ZERO_V5_C_MINIMUM = 1.99
UPDATE_ZERO_V5_C_MAXIMUM = 2.01
GATE_THRESHOLDS = {
    update: dict(thresholds)
    for update, thresholds in _v4.GATE_THRESHOLDS.items()
}
GATE_THRESHOLDS[0] = {
    "v5_C_minimum_inclusive": UPDATE_ZERO_V5_C_MINIMUM,
    "v5_C_maximum_inclusive": UPDATE_ZERO_V5_C_MAXIMUM,
}

CONTROL_UPDATE_ZERO_FAIL = (
    "FAIL_UPDATE_ZERO_V5_DELTA_CONTRAST_INTEGRITY_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_ZERO = (
    "CONTINUE_AFTER_UPDATE_ZERO_V5_DELTA_CONTRAST_GATE"
)
CONTROL_UPDATE_100_FAIL = (
    "FAIL_UPDATE_100_V5_DELTA_CONTRAST_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_100 = (
    "CONTINUE_AFTER_UPDATE_100_V5_DELTA_CONTRAST_GATE"
)
CONTROL_UPDATE_400_FAIL = (
    "FAIL_UPDATE_400_V5_DELTA_CONTRAST_MECHANISM_GATE_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_400 = (
    "CONTINUE_AFTER_UPDATE_400_V5_DELTA_CONTRAST_GATE"
)
CONTROL_UPDATE_1000_FAIL = (
    "FAIL_UPDATE_1000_V5_DELTA_CONTRAST_PERCEPTION_GATE_TERMINAL_NO_RETRY"
)
CONTROL_PASS = (
    "PASS_DIRECT_BEV_V5_ALL_ACTIONS_STATE_DELTA_CONTRAST_PERCEPTION_GATE_"
    "REQUALIFICATION_ONLY"
)
GATE_CONTROLS = {
    0: (CONTROL_UPDATE_ZERO_FAIL, CONTROL_CONTINUE_UPDATE_ZERO),
    100: (CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100),
    400: (CONTROL_UPDATE_400_FAIL, CONTROL_CONTINUE_UPDATE_400),
    1_000: (CONTROL_UPDATE_1000_FAIL, CONTROL_PASS),
}
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

ALL_ACTIONS_STATE_DELTA_CONTRAST = {
    "name": "A",
    "action_count": 9,
    "inputs": {
        "online_current_state": "S",
        "all_action_predictions": "P",
        "detached_ema_current_state": "T_current",
        "detached_ema_next_state": "T_next",
        "executed_action": "already_authorized_executed_action",
    },
    "state_probability": "softmax_over_three_state_channels",
    "predicted_delta": "softmax(P[b,a])-softmax(S[b])",
    "target_delta": (
        "stop_gradient(softmax(T_next[b])-softmax(T_current[b]))"
    ),
    "distance": (
        "mean_over_state_channel_height_width((predicted_delta-target_delta)^2)"
    ),
    "scale": "stop_gradient(mean_over_nine_actions(distance)).clamp_min(1e-4)",
    "delta_logits": "negative_distance_over_scale",
    "formula": "mean(cross_entropy(delta_logits,executed_action))/log(9)",
    "weight": 1.0,
    "online_current_state_detached": False,
    "both_ema_target_terms_detached": True,
    "exact_persistence_expected_A": 1.0,
    "adds_parameter_buffer_module_state_call_target_call_or_output_head": False,
    "uses_raster_pose_depth_odometry_geometry_ray_warp_or_navigation_signal": False,
}

SCIENTIFIC_DELTA = {
    "scope": "one_all_actions_learned_state_delta_contrast_loss",
    "frozen_objective": "C_v3",
    "added_objective": "A",
    "reported_and_traced_C": "C_v5=C_v3+A",
    "total": "G/log(2)+J/log(2)+C_v5",
    "auxiliary_weight": 1.0,
    "update_zero_C_closed_interval": [
        UPDATE_ZERO_V5_C_MINIMUM,
        UPDATE_ZERO_V5_C_MAXIMUM,
    ],
    "model_parameter_buffer_module_or_output_delta": False,
    "data_seed_initialization_schedule_optimizer_or_ema_delta": False,
    "v3_v4_threshold_relaxation": False,
    "v4_checkpoint_tensor_trace_or_runtime_output_reuse": False,
}

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v4_manifest_and_all_91_sources_rehashed": True,
    "frozen_v4_review_and_execution_authorization_exact": True,
    "v4_terminal_audit_exact_and_v4_permanently_closed": True,
    "v5_preregistration_exact": True,
    "model_inventory_initialization_data_seed_schedule_optimizer_ema_exact": True,
    "C_v5_is_exactly_C_v3_plus_unit_weight_A": True,
    "A_uses_only_learned_rgb_state_and_executed_action": True,
    "A_target_current_and_next_are_detached": True,
    "update_zero_C_closed_interval_exact": True,
    "v3_update_100_400_1000_thresholds_exact": True,
    "version_local_gate_controls_and_failure_status_validator": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "v4_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v4_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V4_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V4_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V4_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V4_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V4_SOURCE_COUNT,
    }


def frozen_v4_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V4_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V4_REVIEW_COMMIT,
        "file_sha256": FROZEN_V4_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V4_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V4_REVIEW_BYTE_COUNT,
        "status": FROZEN_V4_REVIEW_STATUS,
    }


def frozen_v4_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V4_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V4_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V4_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V4_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V4_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V4_AUTHORIZATION_STATUS,
    }


def v4_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V4_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V4_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V4_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V4_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V4_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V4_TERMINAL_AUDIT_STATUS,
        "classification": V4_TERMINAL_AUDIT_CLASSIFICATION,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def model_config() -> dict[str, Any]:
    return _v4.model_config()


def objective_contract() -> dict[str, Any]:
    value = _v4.objective_contract()
    frozen_c = dict(value["C"])
    value["C_v3"] = frozen_c
    value["A"] = dict(ALL_ACTIONS_STATE_DELTA_CONTRAST)
    value["C"] = {
        "name": "C_v5",
        "formula": "C_v3+A",
        "C_v3": frozen_c,
        "A": dict(ALL_ACTIONS_STATE_DELTA_CONTRAST),
        "A_weight": 1.0,
        "reported_and_traced_scalar": "C_v5",
    }
    value["total"] = "1*G/log(2) + 1*J/log(2) + 1*C_v5"
    value["absent"] = [
        name for name in value["absent"] if name != "auxiliary_loss"
    ]
    return value


def optimizer_contract() -> dict[str, Any]:
    return _v4.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v4.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v4.runtime_authorization_template()


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen V3 gates plus V5's sole update-zero C range."""

    result = _v4.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    if update == 0:
        c_value = _v4._v3._v2._v1._finite_number(metrics.get("C"), name="C")
        conjuncts = dict(result["conjuncts"])
        conjuncts["v5_C_in_closed_interval_1point99_2point01"] = (
            UPDATE_ZERO_V5_C_MINIMUM
            <= c_value
            <= UPDATE_ZERO_V5_C_MAXIMUM
        )
        result["conjuncts"] = conjuncts
        result["passed"] = all(conjuncts.values())
        result["thresholds"] = dict(GATE_THRESHOLDS[0])
    result["control"] = GATE_CONTROLS[update][1 if result["passed"] else 0]
    return result


def validate_failure_status_chain(value: object) -> dict[str, str]:
    """Validate against V5 controls, never the inherited V1 lexical globals."""

    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("failure receipt statuses are not one exact V5 gate control")
    return dict(value)


def science_contract() -> dict[str, Any]:
    value = _v4.science_contract()
    value.pop("integrity_replacement", None)
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "Does an all-actions learned state-delta contrast overcome static-map "
        "domination and make the frozen V3 predictor action-discriminative?"
    )
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v4_source_manifest": frozen_v4_source_manifest_binding(),
        "frozen_v4_source_review": frozen_v4_review_binding(),
        "frozen_v4_execution_authorization": frozen_v4_authorization_binding(),
        "v4_terminal_audit": v4_terminal_audit_binding(),
        "v5_preregistration": preregistration_binding(),
    }
    value["model"] = model_config()
    value["objective"] = objective_contract()
    thresholds = dict(value["gates"]["thresholds"])
    thresholds["0"] = dict(GATE_THRESHOLDS[0])
    value["gates"] = {**value["gates"], "thresholds": thresholds}
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "loss_successor_of": _v4.EXPERIMENT_ID,
        "v4_retry": False,
        "v4_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    }
    value["loss_successor"] = dict(SCIENTIFIC_DELTA)
    value["authority"] = {
        **value["authority"],
        "v5_execution_authorized_by_source_contract": False,
        "v4_checkpoint_or_runtime_output_reuse_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    return value


def validate_frozen_v4_source_closure(root: Path = ROOT) -> dict[str, str]:
    read = _v4._v3._v2._v1._read_regular_source
    raw = read(root / FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH)
    if (
        len(raw) != FROZEN_V4_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest()
        != FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V4 source manifest raw identity changed")
    manifest = _v4.validate_source_manifest(raw)
    if (
        manifest.get("content_sha256")
        != FROZEN_V4_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != FROZEN_V4_SOURCE_MANIFEST_STATUS
        or manifest.get("source_count") != FROZEN_V4_SOURCE_COUNT
        or manifest.get("source_paths") != list(REUSED_SOURCE_PATHS)
    ):
        raise PermissionError("frozen V4 source manifest conclusion changed")
    current = _v4.current_source_bindings(root)
    if current.get(FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("current V4 source manifest changed")
    for binding in manifest["source_bindings"]:
        if current.get(binding["path"]) != binding["file_sha256"]:
            raise PermissionError(f"current V4 source changed: {binding['path']}")
    return current


def _read_and_validate_frozen_v4_review_and_authorization(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    read = _v4._v3._v2._v1._read_regular_source
    review_raw = read(root / FROZEN_V4_REVIEW_RELATIVE_PATH)
    authorization_raw = read(root / FROZEN_V4_AUTHORIZATION_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V4_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V4_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V4 source review raw identity changed")
    if (
        len(authorization_raw) != FROZEN_V4_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V4_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V4 authorization raw identity changed")
    review = _v4.parse_canonical_json(review_raw, name="frozen V4 source review")
    authorization = _v4.parse_canonical_json(
        authorization_raw, name="frozen V4 authorization"
    )
    if (
        review.get("content_sha256") != FROZEN_V4_REVIEW_CONTENT_SHA256
        or review.get("status") != FROZEN_V4_REVIEW_STATUS
        or authorization.get("content_sha256")
        != FROZEN_V4_AUTHORIZATION_CONTENT_SHA256
        or authorization.get("status") != FROZEN_V4_AUTHORIZATION_STATUS
    ):
        raise PermissionError("frozen V4 review or authorization conclusion changed")
    _v4.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _v4.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    return dict(review), dict(authorization)


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = validate_frozen_v4_source_closure(root)
    _read_and_validate_frozen_v4_review_and_authorization(root)
    read = _v4._v3._v2._v1._read_regular_source
    audit_raw = read(root / V4_TERMINAL_AUDIT_RELATIVE_PATH)
    preregistration_raw = read(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(audit_raw) != V4_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest() != V4_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V4 terminal audit raw identity changed")
    if (
        len(preregistration_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(preregistration_raw).hexdigest()
        != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V5 preregistration changed")
    audit = json.loads(audit_raw)
    core = dict(audit)
    declared = core.pop("content_sha256", None)
    accounting = audit.get("execution_accounting", {})
    gate = audit.get("update_100_scientific_gate", {})
    diagnosis = audit.get("terminal_packaging_diagnosis", {})
    consequence = audit.get("scientific_consequence", {})
    if (
        declared != V4_TERMINAL_AUDIT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
        or audit.get("status") != V4_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V4_TERMINAL_AUDIT_CLASSIFICATION
        or accounting.get("updates") != 100
        or accounting.get("presentations") != 1_600
        or accounting.get("optimizer_updates") != 100
        or accounting.get("ema_updates") != 100
        or gate.get("status")
        != "FAIL_UPDATE_100_V3_PREDICTOR_GATE_TERMINAL_NO_RETRY"
        or gate.get("passed") is not False
        or diagnosis.get("root_cause")
        != (
            "inherited_v1_validate_failure_status_chain_lexically_closes_over_"
            "v1_failure_controls_and_rejects_the_v3_specific_update_100_control"
        )
        or consequence.get("v4_permanently_closed") is not True
        or consequence.get(
            "v4_retry_resume_repair_or_checkpoint_reuse_authorized"
        ) is not False
        or consequence.get("v4_checkpoint_qualified") is not False
        or consequence.get(
            "next_successor_must_be_fresh_and_may_not_reuse_v4_checkpoint_"
            "tensor_trace_or_runtime_output"
        ) is not True
    ):
        raise PermissionError("V4 terminal audit conclusion changed")
    result.update({
        FROZEN_V4_REVIEW_RELATIVE_PATH: FROZEN_V4_REVIEW_FILE_SHA256,
        FROZEN_V4_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V4_AUTHORIZATION_FILE_SHA256
        ),
        V4_TERMINAL_AUDIT_RELATIVE_PATH: V4_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = _v4.parse_canonical_json(raw, name="V5 source manifest")
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
        or value.get("source_count") != 101
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V5 source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V5 source binding fields changed")
        relative = _v4._v3._v2._v1.safe_relative_source_path(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V5 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V5 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V5 source changed: {binding['path']}")
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(manifest_raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        raw = _v4._v3._v2._v1._read_regular_source(
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
        "reviewed_sources", "source_manifest", "frozen_v4_source_manifest",
        "frozen_v4_source_review", "frozen_v4_execution_authorization",
        "v4_terminal_audit", "v5_preregistration", "science_contract",
        "source_only_checks", "scientific_checks", "findings", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V5 source review fields changed")
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
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH)
        != manifest_binding["file_sha256"]
        or expected_sources.get(FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256
        or expected_sources.get(FROZEN_V4_REVIEW_RELATIVE_PATH)
        != FROZEN_V4_REVIEW_FILE_SHA256
        or expected_sources.get(FROZEN_V4_AUTHORIZATION_RELATIVE_PATH)
        != FROZEN_V4_AUTHORIZATION_FILE_SHA256
        or expected_sources.get(V4_TERMINAL_AUDIT_RELATIVE_PATH)
        != V4_TERMINAL_AUDIT_FILE_SHA256
        or expected_sources.get(PREREGISTRATION_RELATIVE_PATH)
        != PREREGISTRATION_FILE_SHA256
        or value["frozen_v4_source_manifest"]
        != frozen_v4_source_manifest_binding()
        or value["frozen_v4_source_review"] != frozen_v4_review_binding()
        or value["frozen_v4_execution_authorization"]
        != frozen_v4_authorization_binding()
        or value["v4_terminal_audit"] != v4_terminal_audit_binding()
        or value["v5_preregistration"] != preregistration_binding()
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
        raise PermissionError("V5 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v4_source_manifest", "frozen_v4_source_review",
        "frozen_v4_execution_authorization", "v4_terminal_audit",
        "v5_preregistration", "runtime_inputs", "experiment", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V5 execution authorization fields changed")
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
        or value["frozen_v4_source_manifest"]
        != frozen_v4_source_manifest_binding()
        or value["frozen_v4_source_review"] != frozen_v4_review_binding()
        or value["frozen_v4_execution_authorization"]
        != frozen_v4_authorization_binding()
        or value["v4_terminal_audit"] != v4_terminal_audit_binding()
        or value["v5_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V5 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v4.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v4_authorization_binding",
    "frozen_v4_review_binding",
    "frozen_v4_source_manifest_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "v4_terminal_audit_binding",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v4_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
    "with_content_sha256",
})
