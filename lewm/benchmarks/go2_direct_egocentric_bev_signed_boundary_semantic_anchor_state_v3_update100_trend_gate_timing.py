"""Source-only contract for the semantic-anchor V3 gate-timing successor.

V3 preserves the complete frozen V2 training mechanism.  Its sole disclosed
evaluation-decision delta records, but does not terminally apply, the exact
update-100 balanced-accuracy conjunct.  Import grants no runtime authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V2_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement.py"
)


def _source_only_module(name: str, relative: str) -> Any:
    source = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V2 = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v3_frozen_v2_contract",
    FROZEN_V2_CONTRACT_RELATIVE_PATH,
)
for _name in _V2.__all__:
    globals()[_name] = getattr(_V2, _name)

canonical_json_bytes = _V2.canonical_json_bytes
canonical_json_sha256 = _V2.canonical_json_sha256
with_content_sha256 = _V2.with_content_sha256
parse_canonical_json = _V2.parse_canonical_json
is_sha256 = _V2.is_sha256
artifact_binding = _V2.artifact_binding
validate_binding = _V2.validate_binding

IMPLEMENTATION_AUTHOR = "/root/semantic_v2_contract_rescue"
IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/semantic_v2_contract_rescue",
    "/root/semantic_v2_prereg_draft",
    "/root/semantic_v2_source_audit",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v3_"
    "update100_trend_gate_timing_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v3_"
    "update100_trend_gate_timing.py"
)
MODEL_RELATIVE_PATH = _V2.MODEL_RELATIVE_PATH
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_source_closure.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = _V2.MODEL_TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V2_RUNNER_RELATIVE_PATH = _V2.RUNNER_RELATIVE_PATH
FROZEN_V2_LAUNCHER_RELATIVE_PATH = _V2.LAUNCHER_RELATIVE_PATH
FROZEN_V2_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V2.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_preregistration_"
    "2026-07-27.json"
)
PREREGISTRATION_COMMIT = "ef830dd92f923ba36276b44fe0fc0eaf656ef8d4"
PREREGISTRATION_FILE_SHA256 = (
    "e63bcb9b127d62a300bd5447c6e1e37284081c50a39d2b35c00acc3190982a31"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "716860d02eb7e8f98cccd2f3f8eef0ecd8e21d230b44787bf388160a519ef319"
)
PREREGISTRATION_BYTE_COUNT = 15_878
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_SCIENCE_IDENTICAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_"
    "STATE_V3_UPDATE100_TREND_GATE_TIMING_SUCCESSOR_PENDING_SOURCE_FREEZE_"
    "INDEPENDENT_REVIEW_AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH = _V2.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V2_SOURCE_MANIFEST_COMMIT = (
    "bdf5ef34d2353da7e9f321c69664f0891e3fb373"
)
FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256 = (
    "fb6ce9b735051766840e7f0aaf52e3161087bc93544ed384854a0d28b8d0db0b"
)
FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "b15cee9ef92d76e50e9a60008975f62b3fd54c47f2053772d39ae7cc28ed5ec6"
)
FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT = 58_065
FROZEN_V2_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V2_SOURCE_COUNT = 160

FROZEN_V2_REVIEW_RELATIVE_PATH = _V2.REVIEW_RELATIVE_PATH
FROZEN_V2_REVIEW_COMMIT = "a6a8b4de2073166e4ae9acb134f2dc79b40ac93e"
FROZEN_V2_REVIEW_FILE_SHA256 = (
    "ed8195c9c8b0bf2dbfa788816597ee569202ccb2d4925b02f3b0e30240cd1c8c"
)
FROZEN_V2_REVIEW_CONTENT_SHA256 = (
    "1c8f5d48f7fce0c56f4cb13d1e66eaaf84f56eb904e242e60bff68e44d56d4d9"
)
FROZEN_V2_REVIEW_BYTE_COUNT = 90_179
FROZEN_V2_REVIEW_STATUS = _V2.REVIEW_STATUS

FROZEN_V2_AUTHORIZATION_RELATIVE_PATH = _V2.AUTHORIZATION_RELATIVE_PATH
FROZEN_V2_AUTHORIZATION_COMMIT = (
    "a40b4e228a2b741c171bf3d714228a772d5a9833"
)
FROZEN_V2_AUTHORIZATION_FILE_SHA256 = (
    "ab8bef03eadb56fd7f3da39d332e08918a65370e6b74d03ecda4c3cda1f31859"
)
FROZEN_V2_AUTHORIZATION_CONTENT_SHA256 = (
    "49aa85e872607f9af9b9703141bd9955b7d8354e3b5dfa003158508c1f5741b0"
)
FROZEN_V2_AUTHORIZATION_BYTE_COUNT = 70_537
FROZEN_V2_AUTHORIZATION_STATUS = _V2.AUTHORIZATION_STATUS

FROZEN_V2_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_terminal_audit_"
    "2026-07-27.json"
)
FROZEN_V2_TERMINAL_AUDIT_COMMIT = (
    "ef8dfcf5ed659b64cf6adf7480c904cbeb61357c"
)
FROZEN_V2_TERMINAL_AUDIT_FILE_SHA256 = (
    "88a0b03fde5f5cda2088576ae0fd12ef5c8d5dc47925a4df7e7defa85c8132b8"
)
FROZEN_V2_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "3e31ce22a5c19b553299c32155291936d033aedb4a2588842c3251bc0d8c7bc3"
)
FROZEN_V2_TERMINAL_AUDIT_BYTE_COUNT = 23_895
FROZEN_V2_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_RECEIPT_CHAIN_UPDATE_100_SCIENTIFIC_FAILURE_SIGNED_"
    "BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_INTEGRITY_"
    "REPLACEMENT_CLOSED_NO_RETRY"
)
FROZEN_V2_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_100_SINGLE_BALANCED_ACCURACY_CONJUNCT_SCIENTIFIC_FAILURE_"
    "AFTER_STRONG_OBJECTIVE_NLL_AND_SEMANTIC_IMPROVEMENT_SIGNED_BOUNDARY_"
    "SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_"
    "CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_source_manifest_"
    "2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_source_review_"
    "2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V2.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 160 or len(ADDITIVE_SOURCE_PATHS) != 5:
    raise RuntimeError("semantic-anchor V3 source delta changed")
if len(SOURCE_PATHS) != 165 or MODEL_RELATIVE_PATH in ADDITIVE_SOURCE_PATHS:
    raise RuntimeError("semantic-anchor V3 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V2_REVIEW_RELATIVE_PATH,
    FROZEN_V2_AUTHORIZATION_RELATIVE_PATH,
    FROZEN_V2_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_"
    "semantic_anchor_state_v3/rgb_direct_egocentric_bev_signed_boundary_"
    "semantic_anchor_state_probe_v3_update100_trend_gate_timing_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_"
    "UPDATE100_TREND_GATE_TIMING_PREFLIGHT_JSON"
)
RUNTIME_INTERPRETER_PATH = _V2.RUNTIME_INTERPRETER_PATH
RUNTIME_SYS_PREFIX = _V2.RUNTIME_SYS_PREFIX

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
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_TREND_GATE_"
    "TIMING_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)
RESERVATION_PUBLICATION_FAILURE_STATUS = (
    "TERMINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_TREND_GATE_"
    "TIMING_RESERVATION_PUBLICATION_FAILURE_NO_RETRY"
)
REVIEW_STATUS = (
    "PASS_SOURCE_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_TREND_"
    "GATE_TIMING_SCIENCE_AND_CUSTODY"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_SCIENCE_IDENTICAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_"
    "STATE_V3_UPDATE100_TREND_GATE_TIMING_SUCCESSOR"
)

SCHEDULE_SCHEMA_ADAPTER_CHANGED = False
TRAINING_SCIENCE_DELTA_COUNT = 0
EVALUATION_DECISION_PROTOCOL_DELTA_COUNT = 1
SCIENCE_DELTA_COUNT = TRAINING_SCIENCE_DELTA_COUNT
_FROZEN_V2_SCIENCE_CONTRACT = deepcopy(_V2.science_contract())
FROZEN_V2_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V2_SCIENCE_CONTRACT
)
if FROZEN_V2_SCIENCE_CONTRACT_SHA256 != (
    "2d42031e0586c205cfcae783991a497a4b3f4a5b1c5b8013aa3e65ac5ca673f1"
):
    raise RuntimeError("frozen semantic-anchor V2 science identity changed")


def science_contract() -> dict[str, Any]:
    """Return frozen V2 training science plus one disclosed decision overlay."""

    value = deepcopy(_FROZEN_V2_SCIENCE_CONTRACT)
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["gates"]["controls"] = {
        str(update): list(GATE_CONTROLS[update])
        for update in OBSERVATION_UPDATES
    }
    value["gates"]["evaluation_decision_protocol"] = deepcopy(
        EVALUATION_DECISION_PROTOCOL
    )
    return value


def normalize_v3_decision_protocol(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove exactly the V3 receipt/timing overlay and recover frozen V2."""

    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V3 evaluation contract differs from its contract")
    normalized = deepcopy(dict(value))
    normalized["schema"] = _FROZEN_V2_SCIENCE_CONTRACT["schema"]
    normalized["gates"]["controls"] = deepcopy(
        _FROZEN_V2_SCIENCE_CONTRACT["gates"]["controls"]
    )
    protocol = normalized["gates"].pop(
        "evaluation_decision_protocol", None
    )
    if (
        protocol != EVALUATION_DECISION_PROTOCOL
        or normalized != _FROZEN_V2_SCIENCE_CONTRACT
    ):
        raise PermissionError("V3 changed frozen V2 outside decision timing")
    return normalized


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    normalized = normalize_v3_decision_protocol(value)
    return {
        "inherited_v2_science_contract_sha256": (
            FROZEN_V2_SCIENCE_CONTRACT_SHA256
        ),
        "v3_full_evaluation_contract_sha256": canonical_json_sha256(value),
        "normalized_v3_decision_protocol_sha256": canonical_json_sha256(
            normalized
        ),
        "normalized_exactly_equals_v2": (
            normalized == _FROZEN_V2_SCIENCE_CONTRACT
        ),
        "training_science_delta_count": TRAINING_SCIENCE_DELTA_COUNT,
        "evaluation_decision_protocol_delta_count": (
            EVALUATION_DECISION_PROTOCOL_DELTA_COUNT
        ),
        "changed_evaluation_paths": [
            "schema",
            "gates.controls",
            "gates.evaluation_decision_protocol",
        ],
        "schedule_schema_adapter_changed": SCHEDULE_SCHEMA_ADAPTER_CHANGED,
        "v2_runtime_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


build_schedule_identity = _V2.build_schedule_identity
model_config = _V2.model_config
perception_accounting = _V2.perception_accounting

CONTROL_PRELIMINARY = (
    "PRELIMINARY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_TREND_"
    "GATE_TIMING_DISPATCH_NOT_FINAL_SCIENTIFIC_EVIDENCE"
)
GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
        "TREND_GATE_TIMING_STRUCTURAL_INTEGRITY_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_"
        "UPDATE100_TREND_GATE_TIMING_STRUCTURAL_INTEGRITY_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
        "TREND_GATE_TIMING_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_"
        "UPDATE100_TREND_GATE_WITH_BALANCED_ACCURACY_RECORDED_AND_DEFERRED",
    ),
    400: (
        "FAIL_UPDATE_400_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
        "TREND_GATE_TIMING_ANTI_COLLAPSE_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_"
        "UPDATE100_TREND_GATE_TIMING_ANTI_COLLAPSE_GATE",
    ),
    1_000: (
        "FAIL_UPDATE_1000_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
        "TREND_GATE_TIMING_QUALIFICATION_GATE_TERMINAL_NO_RETRY",
        "PASS_RGB_ONLY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
        "TREND_GATE_TIMING_PERCEPTION_MECHANISM_ONLY",
    ),
}
CONTROL_UPDATE_0_FAIL, CONTROL_CONTINUE_UPDATE_0 = GATE_CONTROLS[0]
CONTROL_UPDATE_100_FAIL, CONTROL_CONTINUE_UPDATE_100 = GATE_CONTROLS[100]
CONTROL_UPDATE_400_FAIL, CONTROL_CONTINUE_UPDATE_400 = GATE_CONTROLS[400]
CONTROL_UPDATE_1000_FAIL, CONTROL_PASS = GATE_CONTROLS[1_000]
GATE_THRESHOLDS = deepcopy(_V2.GATE_THRESHOLDS)
BALANCED_ACCURACY_CONJUNCT = (
    "balanced_accuracy_at_least_max_point68_or_update_zero_plus_point08"
)
EVALUATION_DECISION_PROTOCOL = {
    "delta_count": 1,
    "scope": "final_update_100_balanced_accuracy_terminal_application_only",
    "deferred_conjunct": BALANCED_ACCURACY_CONJUNCT,
    "threshold_formula": "max(0.68,BA_0+0.08)",
    "threshold_or_comparator_changed": False,
    "original_v2_conjunct_map_recorded_unchanged": True,
    "continue_iff_all_original_non_balanced_accuracy_conjuncts_true": True,
    "update_0_update_400_update_1000_gate_math_changed": False,
    "update_200_present": False,
    "update_400_failure_hard_closes_semantic_anchor_mechanism": True,
    "further_timing_successor_or_weight_tune_authorized": False,
}
FAILURE_CONTROLS = tuple(
    GATE_CONTROLS[update][0] for update in OBSERVATION_UPDATES
)


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    update_100: Mapping[str, Any] | None = None,
    update_400: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Call V2 once; defer only u100 BA terminal application."""

    frozen = _V2.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        update_400=update_400,
        prior_gates_passed=prior_gates_passed,
    )
    result = deepcopy(frozen)
    if result.get("final_gate_evaluated") is True:
        original_passed = frozen.get("passed")
        conjuncts = frozen.get("conjuncts")
        if type(original_passed) is not bool or type(conjuncts) is not dict:
            raise PermissionError("frozen V2 final gate evidence changed")
        passed = original_passed
        if update == 100:
            if update_zero is None or BALANCED_ACCURACY_CONJUNCT not in conjuncts:
                raise PermissionError("frozen V2 u100 BA evidence changed")
            original_conjuncts = deepcopy(conjuncts)
            active_conjuncts = {
                name: value
                for name, value in original_conjuncts.items()
                if name != BALANCED_ACCURACY_CONJUNCT
            }
            if any(type(value) is not bool for value in original_conjuncts.values()):
                raise PermissionError("frozen V2 u100 conjunct type changed")
            ba_0 = float(update_zero["aggregate_raster_balanced_accuracy"])
            ba_100 = float(metrics["aggregate_raster_balanced_accuracy"])
            threshold = max(0.68, ba_0 + 0.08)
            ba_pass = original_conjuncts[BALANCED_ACCURACY_CONJUNCT]
            if ba_pass is not (ba_100 >= threshold):
                raise PermissionError("frozen V2 u100 BA comparator changed")
            passed = all(active_conjuncts.values())
            result.update({
                "original_v2_conjuncts": original_conjuncts,
                "active_conjuncts": active_conjuncts,
                "original_v2_gate_passed": original_passed,
                "original_v2_control": frozen["control"],
                "deferred_conjunct": BALANCED_ACCURACY_CONJUNCT,
                "balanced_accuracy_evidence": {
                    "BA_0": ba_0,
                    "BA_100": ba_100,
                    "BA_threshold": threshold,
                    "BA_threshold_formula": "max(0.68,BA_0+0.08)",
                    "BA_pass": ba_pass,
                    "BA_margin": ba_100 - threshold,
                    "original_V2_gate_would_pass": original_passed,
                    "balanced_accuracy_recorded": True,
                    "balanced_accuracy_applied_as_terminal_conjunct": False,
                },
                "evaluation_decision_protocol_delta_count": 1,
            })
        result["passed"] = passed
        result["control"] = GATE_CONTROLS[update][1 if passed else 0]
        result["gate_mode"] = (
            "FINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_TREND_"
            "GATE_TIMING_RECEIPT"
        )
    else:
        result["control"] = CONTROL_PRELIMINARY
        result["gate_mode"] = (
            "PRELIMINARY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V3_UPDATE100_"
            "TREND_GATE_TIMING_DISPATCH_NOT_FINAL_"
            "SCIENTIFIC_EVIDENCE"
        )
    return result


def validate_failure_status_chain(value: object) -> dict[str, str]:
    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("V3 failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("V3 failure receipts lost one exact gate control")
    return dict(value)


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V2.runtime_authorization_template())
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
        "v2_runtime_output_reuse": False,
        "v2_retry_resume_repair_or_recovery": False,
        "semantic_anchor_state_v3_update100_trend_gate_timing_only": True,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": False,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "runtime_interpreter_path": RUNTIME_INTERPRETER_PATH,
        "runtime_sys_prefix": RUNTIME_SYS_PREFIX,
        "schedule_schema_adapter_changed": False,
    }
    return value


SOURCE_ONLY_AUTHORITY = dict(_V2.SOURCE_ONLY_AUTHORITY)
REVIEW_AUTHORITY = dict(_V2.REVIEW_AUTHORITY)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = deepcopy(_V2.EXECUTION_AUTHORITY)
EXECUTION_AUTHORITY.update({
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_semantic_anchor_v2_runtime_interpreter_integrity_replacement_only": False,
    "one_fresh_semantic_anchor_v3_update100_trend_gate_timing_only": True,
    "training_science_identical_to_frozen_semantic_anchor_v2": True,
    "evaluation_decision_protocol_delta_count": 1,
    "exact_reviewed_runtime_interpreter_handoff_only": True,
    "runtime_interpreter_path": RUNTIME_INTERPRETER_PATH,
    "runtime_sys_prefix": RUNTIME_SYS_PREFIX,
    "v2_runtime_output_or_state_reuse_authorized": False,
    "v2_retry_resume_repair_recovery_or_extension_authorized": False,
    "v2_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_authorized": False,
    "further_semantic_anchor_gate_timing_successor_authorized": False,
    "semantic_anchor_weight_tune_authorized": False,
})


def frozen_v2_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V2_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT,
        "source_count": FROZEN_V2_SOURCE_COUNT,
        "status": FROZEN_V2_SOURCE_MANIFEST_STATUS,
    }


def frozen_v2_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V2_REVIEW_COMMIT,
        "file_sha256": FROZEN_V2_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V2_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V2_REVIEW_BYTE_COUNT,
        "status": FROZEN_V2_REVIEW_STATUS,
    }


def frozen_v2_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V2_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V2_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V2_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V2_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V2_AUTHORIZATION_STATUS,
    }


def frozen_v2_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": FROZEN_V2_TERMINAL_AUDIT_COMMIT,
        "file_sha256": FROZEN_V2_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": FROZEN_V2_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": FROZEN_V2_TERMINAL_AUDIT_BYTE_COUNT,
        "status": FROZEN_V2_TERMINAL_AUDIT_STATUS,
        "classification": FROZEN_V2_TERMINAL_AUDIT_CLASSIFICATION,
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


_READ_REGULAR_SOURCE = _V2._READ_REGULAR_SOURCE
_SAFE_RELATIVE_SOURCE_PATH = _V2._SAFE_RELATIVE_SOURCE_PATH


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
    **_metadata: Any,
) -> dict[str, Any]:
    raw = _READ_REGULAR_SOURCE(ROOT / relative_path)
    parsed = parse_canonical_json(raw, name=relative_path)
    core = dict(parsed)
    declared = core.pop("content_sha256", None)
    scientific = parsed.get("scientific_result")
    nested_classification = (
        scientific.get("classification")
        if type(scientific) is dict
        else None
    )
    actual_classification = parsed.get(
        "classification", nested_classification
    )
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
    return dict(parsed)


def validate_frozen_v2_source_closure(
    root: Path = ROOT,
) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("frozen V1 closure must use repository root")
    current = _V2.current_source_bindings(root)
    if (
        current.get(FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen semantic-anchor V1 closure changed")
    return current


def validate_governing_documents(
    root: Path = ROOT,
) -> dict[str, str]:
    current = validate_frozen_v2_source_closure(root)
    for binding in (
        frozen_v2_review_binding(),
        frozen_v2_authorization_binding(),
        frozen_v2_terminal_audit_binding(),
        preregistration_binding(),
    ):
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
    value = parse_canonical_json(raw, name="semantic-anchor V3 source manifest")
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
        raise PermissionError("semantic-anchor V3 manifest fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    bindings = value["source_bindings"]
    if (
        value["schema"] != SOURCE_MANIFEST_SCHEMA
        or value["status"] != "PASS_SOURCE_CLOSURE"
        or value["entrypoints"] != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value["forced_dynamic_sources"]
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value["excluded_runtime_categories"]
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or value["source_paths"] != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != 165
        or value["source_count"] != 165
        or value["source_bindings_sha256"]
        != canonical_json_sha256(bindings)
        or value["generated_input_open_count"] != 0
        or value["checkpoint_or_tensor_open_count"] != 0
        or value["sealed_or_heldout_open_count"] != 0
        or value["whole_tree_export_authorized"] is not False
        or value["authority"] != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("semantic-anchor V3 source manifest changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path",
            "file_sha256",
            "byte_count",
        }:
            raise PermissionError("semantic-anchor V3 source binding fields changed")
        normalized.append(_SAFE_RELATIVE_SOURCE_PATH(binding["path"]))
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("semantic-anchor V3 source binding changed")
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("semantic-anchor V3 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("source closure must use repository root")
    manifest_raw = _READ_REGULAR_SOURCE(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = _READ_REGULAR_SOURCE(root / binding["path"])
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
        raw = _READ_REGULAR_SOURCE(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        manifest = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=manifest["content_sha256"],
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
    "exactly_five_additive_sources_over_frozen_v2_155_sources": True,
    "frozen_v2_model_architecture_parameters_and_initialization_unchanged": True,
    "science_contract_and_schedule_adapter_exactly_frozen_v2": True,
    "exact_pre_reservation_runtime_interpreter_handoff": True,
    "direct_wrong_interpreter_runner_rejected_before_reservation": True,
    "inherited_gate_schedule_snapshot_and_failure_receipt_seams_exact": True,
    "v1_zero_work_terminal_audit_bound_and_v1_closed": True,
    "one_attempt_caps_output_root_and_downstream_denials_exact": True,
    "source_freeze_commit_matches_reviewed_tree": True,
    "all_implementation_authors_excluded": True,
    "generated_or_protected_runtime_inputs_opened": [],
    "sealed_or_heldout_opened": [],
}


def _review_source_freeze_commit(
    review_binding: Mapping[str, Any],
) -> str:
    binding = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    raw = _READ_REGULAR_SOURCE(ROOT / REVIEW_RELATIVE_PATH)
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError("semantic-anchor V2 review binding changed")
    review = parse_canonical_json(raw, name="semantic-anchor V2 source review")
    core = dict(review)
    declared = core.pop("content_sha256", None)
    if (
        declared != binding["content_sha256"]
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("semantic-anchor V2 source review content changed")
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
        "frozen_v2_source_manifest",
        "frozen_v2_source_review",
        "frozen_v2_execution_authorization",
        "frozen_v2_terminal_audit",
        "preregistration",
        "science_contract",
        "science_identity",
        "checks",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("semantic-anchor V2 source review fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    reviewer = value["reviewer"]
    required = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    source_commit = _source_freeze_commit(
        value["source_freeze_commit"], name="review.source_freeze_commit"
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
        or value["frozen_v2_source_manifest"]
        != frozen_v2_source_manifest_binding()
        or value["frozen_v2_source_review"] != frozen_v2_review_binding()
        or value["frozen_v2_execution_authorization"]
        != frozen_v2_authorization_binding()
        or value["frozen_v2_terminal_audit"]
        != frozen_v2_terminal_audit_binding()
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
        raise PermissionError("semantic-anchor V2 source review did not pass")
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
        "frozen_v2_source_manifest",
        "frozen_v2_source_review",
        "frozen_v2_execution_authorization",
        "frozen_v2_terminal_audit",
        "preregistration",
        "runtime_inputs",
        "experiment",
        "science_identity",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("semantic-anchor V2 authorization fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    authorizer = value["authorizer"]
    expected_review = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    source_commit = _review_source_freeze_commit(expected_review)
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, reviewer}
        or value["source_freeze_commit"] != source_commit
        or value["independent_source_review"] != expected_review
        or value["frozen_v2_source_manifest"]
        != frozen_v2_source_manifest_binding()
        or value["frozen_v2_source_review"] != frozen_v2_review_binding()
        or value["frozen_v2_execution_authorization"]
        != frozen_v2_authorization_binding()
        or value["frozen_v2_terminal_audit"]
        != frozen_v2_terminal_audit_binding()
        or value["preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("semantic-anchor V2 authorization changed")
    return dict(value)


__all__ = sorted({
    *_V2.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v2_authorization_binding",
    "frozen_v2_review_binding",
    "frozen_v2_source_manifest_binding",
    "frozen_v2_terminal_audit_binding",
    "model_config",
    "perception_accounting",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v2_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
