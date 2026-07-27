"""Source-only contract for the Direct BEV V12 update-50 trend gate.

V12 preserves the frozen V11 model, data, initialization, optimization, and
training schedule.  Only the marker-present update-50 continuation decision
changes; importing this module grants no runtime or downstream authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
_FROZEN_DIRECT_BEV_V11_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
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


_V11 = _source_only_module(
    "_lewm_direct_bev_v12_frozen_v11_contract",
    _FROZEN_DIRECT_BEV_V11_CONTRACT_RELATIVE_PATH,
)
_FROZEN_V11_SCIENCE_CONTRACT = _V11.science_contract()

for _name in _V11.__all__:
    globals()[_name] = getattr(_V11, _name)

FROZEN_V11_CONTRACT_RELATIVE_PATH = (
    _FROZEN_DIRECT_BEV_V11_CONTRACT_RELATIVE_PATH
)

canonical_json_bytes = _V11.canonical_json_bytes
canonical_json_sha256 = _V11.canonical_json_sha256
is_sha256 = _V11.is_sha256
with_content_sha256 = _V11.with_content_sha256
parse_canonical_json = _V11.parse_canonical_json
artifact_binding = _V11.artifact_binding
validate_binding = _V11.validate_binding


IMPLEMENTATION_AUTHOR = "/root"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_RELATIVE_PATH = _V11.MODEL_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V11_MODEL_RELATIVE_PATH = _V11.MODEL_RELATIVE_PATH
FROZEN_V11_RUNNER_RELATIVE_PATH = _V11.RUNNER_RELATIVE_PATH
FROZEN_V11_LAUNCHER_RELATIVE_PATH = _V11.LAUNCHER_RELATIVE_PATH
FROZEN_V11_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V11.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "892e10514aa03396e1d3f797f77e86425b141ec8"
PREREGISTRATION_FILE_SHA256 = (
    "f569f7de16cad2a2aaad50115eeab705371952777eb0c38d702b54c7cab1245e"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "1865cca215ee41a9ac47e7e43e0140544db2921556e3f2964439491605366efa"
)
PREREGISTRATION_BYTE_COUNT = 23_787
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_V12_UPDATE50_TREND_GATE_TIMING_PERCEPTION_"
    "FALSIFICATION_PENDING_SOURCE_FREEZE_INDEPENDENT_REVIEW_AND_SEPARATE_"
    "MACHINE_AUTHORIZATION"
)

FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH = _V11.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V11_SOURCE_MANIFEST_COMMIT = (
    "916be16f6d40dbcaf340284003aa0bbfc2545168"
)
FROZEN_V11_SOURCE_MANIFEST_FILE_SHA256 = (
    "21e34877377d3a839652bdf6d63f9f577789ee69c19a31915f14840e8385b692"
)
FROZEN_V11_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "35e2db00c448b5a99c3061732617a07a77bc36999a513e5153643248158712ee"
)
FROZEN_V11_SOURCE_MANIFEST_BYTE_COUNT = 49_255
FROZEN_V11_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V11_SOURCE_COUNT = 138

FROZEN_V11_REVIEW_RELATIVE_PATH = _V11.REVIEW_RELATIVE_PATH
FROZEN_V11_REVIEW_COMMIT = "88bfe3dacfea16e3afb2fe7a062f12e7c9b4bda3"
FROZEN_V11_REVIEW_FILE_SHA256 = (
    "99df5b35d2aa5817bf94a4d8efe17121b770f4aca12a4e7c3a1c9adfe7419095"
)
FROZEN_V11_REVIEW_CONTENT_SHA256 = (
    "2abf3b02daa23ce37deebf4ae412506880b5919c99f70890307e87f263408ab0"
)
FROZEN_V11_REVIEW_BYTE_COUNT = 79_453
FROZEN_V11_REVIEW_STATUS = (
    "PASS_SOURCE_SCIENCE_IDENTITY_AND_NESTED_OBSERVER_GATE_DISPATCH_"
    "INTEGRITY"
)

FROZEN_V11_AUTHORIZATION_RELATIVE_PATH = _V11.AUTHORIZATION_RELATIVE_PATH
FROZEN_V11_AUTHORIZATION_COMMIT = (
    "4db9516a26d09c8da2c52f87c73405699d02cdc5"
)
FROZEN_V11_AUTHORIZATION_FILE_SHA256 = (
    "ff6c7b373d9af484ac6e7f42e2b8c7d90d00dc5b2248af6349aa733691d491c0"
)
FROZEN_V11_AUTHORIZATION_CONTENT_SHA256 = (
    "d7f3ac133c5c09ef7983dcd5e067b512b1dda3cf47c29372fcfbb79aabb02994"
)
FROZEN_V11_AUTHORIZATION_BYTE_COUNT = 64_454
FROZEN_V11_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V11_NESTED_OBSERVER_GATE_DISPATCH_"
    "INTEGRITY_REPLACEMENT"
)

V11_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_terminal_audit_"
    "2026-07-27.json"
)
V11_TERMINAL_AUDIT_COMMIT = "9379e2e9ee80eeac84a4bc626cb24833ee9f228c"
V11_TERMINAL_AUDIT_FILE_SHA256 = (
    "c3475f5cc7481edf4d9dd019d66eacf49186f025915a6cdda95e782ba83e62ca"
)
V11_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "b1ea22821bc58f2cda595f3f2bf603347c4b9d7a014f4ba81e97480b6e3c2aaa"
)
V11_TERMINAL_AUDIT_BYTE_COUNT = 8_771
V11_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_RECEIPT_CHAIN_UPDATE_50_SCIENTIFIC_FAILURE_V11_"
    "CLOSED_NO_RETRY"
)
V11_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_50_SCIENTIFIC_GATE_FAILURE_WITH_CLEAR_EARLY_IMPROVEMENT"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V11.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 138 or len(SOURCE_PATHS) != 143:
    raise RuntimeError("V12 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V11_REVIEW_RELATIVE_PATH,
    FROZEN_V11_AUTHORIZATION_RELATIVE_PATH,
    V11_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v12/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v12_"
    "update50_trend_gate_timing_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V12_"
    "UPDATE50_TREND_GATE_TIMING_PREFLIGHT_JSON"
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
    "PASS_SOURCE_UPDATE50_TREND_GATE_TIMING_SCIENCE_AND_CUSTODY"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V12_UPDATE50_TREND_GATE_TIMING_"
    "PERCEPTION_FALSIFICATION"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V11.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v11_nested_observer_gate_dispatch_integrity_replacement_only": (
        False
    ),
    "v11_retry_resume_repair_recovery_or_extension_authorized": False,
    "v11_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_"
    "authorized": False,
    "one_fresh_v12_update50_trend_gate_timing_only": True,
    "science_identical_to_frozen_v11": False,
    "model_data_seed_schedule_loss_optimizer_and_ema_identical_to_frozen_"
    "v11": True,
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
SCHEDULE_PREFIX_SHA256 = dict(_V11.SCHEDULE_PREFIX_SHA256)
FROZEN_V11_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V11_SCIENCE_CONTRACT
)
if FROZEN_V11_SCIENCE_CONTRACT_SHA256 != (
    "bf839c0897d73f21b789b8e4c0d9277cba6c2c387e4ccbe347aa4cf91eadff43"
):
    raise PermissionError("frozen V11 science contract identity changed")

GATE_THRESHOLDS = deepcopy(_V11.GATE_THRESHOLDS)
GATE_THRESHOLDS[50] = {
    "G_macro_strictly_less_than_update_0": True,
    "aggregate_raster_balanced_accuracy_strictly_greater_than_update_zero": (
        True
    ),
    "aggregate_free_recall_minimum_inclusive": 0.25,
    "aggregate_occupied_recall_strictly_greater_than_update_zero": True,
    "absolute_free_minus_occupied_recall_gap_maximum_inclusive": 0.60,
    "aggregate_raster_nll_maximum_inclusive": 0.80,
    "rough_raster_balanced_accuracy_strictly_greater_than": 0.0,
    "rough_raster_occupied_recall_strictly_greater_than": 0.0,
    "correct_rgb_macro_scene_win_count_required": 8,
}
CONTROL_UPDATE_50_FAIL = (
    "FAIL_UPDATE_50_V12_TREND_GATE_TIMING_TERMINAL_NO_RETRY"
)
CONTROL_CONTINUE_UPDATE_50 = (
    "CONTINUE_AFTER_UPDATE_50_V12_TREND_GATE_TIMING_TO_EXACT_UPDATE_100_GATE"
)
GATE_CONTROLS = deepcopy(_V11.GATE_CONTROLS)
GATE_CONTROLS[50] = (CONTROL_UPDATE_50_FAIL, CONTROL_CONTINUE_UPDATE_50)
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

SCIENTIFIC_DELTA = {
    "scientific_delta_count": 1,
    "scientific_delta_name": (
        "marker_present_final_update_50_fixed_absolute_floors_to_same_run_"
        "update_zero_directional_trend_gate"
    ),
    "scope": "marker_present_final_update_50_only",
    "replaced_conjuncts": {
        "aggregate_raster_balanced_accuracy_at_least_point60": (
            "aggregate_raster_balanced_accuracy_strictly_higher_than_"
            "update_zero"
        ),
        "aggregate_occupied_recall_at_least_point75": (
            "aggregate_occupied_recall_strictly_higher_than_update_zero"
        ),
    },
    "model_data_seed_schedule_loss_optimizer_and_ema_changed": False,
    "update_zero_update_100_or_update_250_gate_changed": False,
    "v11_runtime_output_or_checkpoint_reuse_authorized": False,
    "maximum_attempts": 1,
}

INTEGRITY_REVIEW_CHECKS = {
    "frozen_v11_manifest_and_all_138_sources_rehashed": True,
    "frozen_v11_review_authorization_and_terminal_audit_exact": True,
    "v11_permanently_closed_and_runtime_reuse_forbidden": True,
    "v12_preregistration_exact": True,
    "v12_adds_no_model_data_objective_optimizer_or_schedule_code": True,
    "exact_v10_model_source_reused_without_modification": True,
    "preliminary_dispatch_result_is_exact_frozen_v11": True,
    "marker_present_update_zero_100_and_250_results_are_exact_frozen_v11": (
        True
    ),
    "marker_present_update_50_replaces_exactly_two_conjuncts": True,
    "update_50_all_other_conjuncts_and_accounting_preserved": True,
    "update_100_is_the_unchanged_decisive_maturity_gate": True,
    "no_v13_threshold_retuning_after_update_100_failure": True,
    "frozen_v11_gate_called_exactly_once_per_v12_gate_call": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v11_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V11_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V11_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V11_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V11_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V11_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V11_SOURCE_COUNT,
    }


def frozen_v11_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V11_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V11_REVIEW_COMMIT,
        "file_sha256": FROZEN_V11_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V11_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V11_REVIEW_BYTE_COUNT,
        "status": FROZEN_V11_REVIEW_STATUS,
    }


def frozen_v11_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V11_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V11_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V11_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V11_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V11_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V11_AUTHORIZATION_STATUS,
    }


def v11_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V11_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V11_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V11_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V11_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V11_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V11_TERMINAL_AUDIT_STATUS,
        "classification": V11_TERMINAL_AUDIT_CLASSIFICATION,
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


def frozen_v11_science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V11_SCIENCE_CONTRACT)


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V11.runtime_authorization_template())
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
        "v11_runtime_output_reuse": False,
        "v11_retry_resume_repair_or_recovery": False,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": (
            False
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    return value


def science_contract() -> dict[str, Any]:
    """Return frozen V11 science with only the preregistered gate delta."""

    value = frozen_v11_science_contract()
    value["gates"]["thresholds"]["50"] = deepcopy(GATE_THRESHOLDS[50])
    value["gates"]["controls"]["50"] = list(GATE_CONTROLS[50])
    return value


def normalize_v12_operational_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V12 experiment differs from its exact contract")
    normalized = deepcopy(dict(value))
    normalized["gates"]["thresholds"]["50"] = deepcopy(
        _FROZEN_V11_SCIENCE_CONTRACT["gates"]["thresholds"]["50"]
    )
    normalized["gates"]["controls"]["50"] = deepcopy(
        _FROZEN_V11_SCIENCE_CONTRACT["gates"]["controls"]["50"]
    )
    if normalized != _FROZEN_V11_SCIENCE_CONTRACT:
        raise PermissionError("V12 changed science outside the update-50 gate")
    return normalized


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    normalized = normalize_v12_operational_identity(value)
    return {
        "frozen_v11_science_contract_sha256": (
            FROZEN_V11_SCIENCE_CONTRACT_SHA256
        ),
        "v12_science_contract_sha256": canonical_json_sha256(value),
        "normalized_v12_science_contract_sha256": canonical_json_sha256(
            normalized
        ),
        "normalized_exactly_equals_frozen_v11": (
            normalized == _FROZEN_V11_SCIENCE_CONTRACT
        ),
        "scientific_delta_count": 1,
        "scientific_delta_name": SCIENTIFIC_DELTA[
            "scientific_delta_name"
        ],
        "changed_science_paths": [
            "gates.thresholds.50",
            "gates.controls.50",
        ],
        "model_data_seed_schedule_loss_optimizer_and_ema_preserved": True,
        "v11_runtime_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    update_100: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Apply only the preregistered marker-present update-50 trend gate."""

    both_absent = all(
        key not in metrics for key in NESTED_DISPATCH_REQUIRED_ABSENT_KEYS
    )
    frozen = _V11.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        prior_gates_passed=prior_gates_passed,
    )
    if both_absent or update != 50:
        return frozen
    if update_zero is None:
        raise ValueError("update_zero baseline is required at update 50")

    old_names = {
        "aggregate_raster_balanced_accuracy_at_least_point60",
        "aggregate_occupied_recall_at_least_point75",
    }
    expected_fields = {
        "update", "passed", "control", "gate_mode",
        "v10_mechanism_receipt_ready",
        "frozen_v8_runtime_readiness_witness", "conjuncts", "thresholds",
        "perception_accounting",
    }
    conjuncts = frozen.get("conjuncts")
    if (
        set(frozen) != expected_fields
        or frozen.get("update") != 50
        or frozen.get("gate_mode")
        != "FINAL_V10_FINAL_CLASS_MACRO_GROUNDING_RECEIPT"
        or frozen.get("control") not in _V11.GATE_CONTROLS[50]
        or frozen.get("thresholds") != _V11.GATE_THRESHOLDS[50]
        or frozen.get("perception_accounting") != PERCEPTION_ACCOUNTING[50]
        or type(conjuncts) is not dict
        or not old_names.issubset(conjuncts)
    ):
        raise PermissionError("frozen V11 update-50 gate receipt changed")

    active = deepcopy(frozen)
    active_conjuncts = active["conjuncts"]
    for name in old_names:
        active_conjuncts.pop(name)
    finite = _V11._V10._finite_number
    active_conjuncts[
        "aggregate_raster_balanced_accuracy_strictly_higher_than_update_zero"
    ] = finite(
        metrics.get("aggregate_raster_balanced_accuracy"),
        name="aggregate_raster_balanced_accuracy",
    ) > finite(
        update_zero.get("aggregate_raster_balanced_accuracy"),
        name="update_zero.aggregate_raster_balanced_accuracy",
    )
    active_conjuncts[
        "aggregate_occupied_recall_strictly_higher_than_update_zero"
    ] = finite(
        metrics.get("aggregate_occupied_recall"),
        name="aggregate_occupied_recall",
    ) > finite(
        update_zero.get("aggregate_occupied_recall"),
        name="update_zero.aggregate_occupied_recall",
    )
    active["passed"] = all(active_conjuncts.values())
    active["control"] = GATE_CONTROLS[50][1 if active["passed"] else 0]
    active["gate_mode"] = "FINAL_V12_UPDATE50_TREND_GATE_TIMING_RECEIPT"
    active["thresholds"] = deepcopy(GATE_THRESHOLDS[50])
    return active


def validate_failure_status_chain(value: object) -> dict[str, str]:
    """Require one exact V12 failure control across all terminal receipts."""

    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("failure receipts are not one exact V12 gate control")
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
    read = (
        _V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
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
            and value.get("scientific_result", {}).get("classification")
            != classification
        )
    ):
        raise PermissionError(f"governing conclusion changed: {relative_path}")
    return dict(value)


def validate_frozen_v11_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V12 frozen V11 closure must use repository root")
    manifest = _read_bound_json(
        FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH,
        file_sha256=FROZEN_V11_SOURCE_MANIFEST_FILE_SHA256,
        content_sha256=FROZEN_V11_SOURCE_MANIFEST_CONTENT_SHA256,
        byte_count=FROZEN_V11_SOURCE_MANIFEST_BYTE_COUNT,
        status=FROZEN_V11_SOURCE_MANIFEST_STATUS,
    )
    if manifest.get("source_count") != FROZEN_V11_SOURCE_COUNT:
        raise PermissionError("frozen V11 source count changed")
    current = _V11.current_source_bindings(root)
    if current.get(FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V11_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V11 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v11_source_closure(root)
    review = _read_bound_json(
        FROZEN_V11_REVIEW_RELATIVE_PATH,
        file_sha256=FROZEN_V11_REVIEW_FILE_SHA256,
        content_sha256=FROZEN_V11_REVIEW_CONTENT_SHA256,
        byte_count=FROZEN_V11_REVIEW_BYTE_COUNT,
        status=FROZEN_V11_REVIEW_STATUS,
    )
    authorization = _read_bound_json(
        FROZEN_V11_AUTHORIZATION_RELATIVE_PATH,
        file_sha256=FROZEN_V11_AUTHORIZATION_FILE_SHA256,
        content_sha256=FROZEN_V11_AUTHORIZATION_CONTENT_SHA256,
        byte_count=FROZEN_V11_AUTHORIZATION_BYTE_COUNT,
        status=FROZEN_V11_AUTHORIZATION_STATUS,
    )
    _V11.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _V11.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    _read_bound_json(
        V11_TERMINAL_AUDIT_RELATIVE_PATH,
        file_sha256=V11_TERMINAL_AUDIT_FILE_SHA256,
        content_sha256=V11_TERMINAL_AUDIT_CONTENT_SHA256,
        byte_count=V11_TERMINAL_AUDIT_BYTE_COUNT,
        status=V11_TERMINAL_AUDIT_STATUS,
        classification=V11_TERMINAL_AUDIT_CLASSIFICATION,
    )
    _read_bound_json(
        PREREGISTRATION_RELATIVE_PATH,
        file_sha256=PREREGISTRATION_FILE_SHA256,
        content_sha256=PREREGISTRATION_CONTENT_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        status=PREREGISTRATION_STATUS,
    )
    current.update({
        FROZEN_V11_SOURCE_MANIFEST_RELATIVE_PATH: (
            FROZEN_V11_SOURCE_MANIFEST_FILE_SHA256
        ),
        FROZEN_V11_REVIEW_RELATIVE_PATH: FROZEN_V11_REVIEW_FILE_SHA256,
        FROZEN_V11_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V11_AUTHORIZATION_FILE_SHA256
        ),
        V11_TERMINAL_AUDIT_RELATIVE_PATH: V11_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V12 source manifest")
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
        raise PermissionError("V12 source manifest contract changed")
    safe = (
        _V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        .safe_relative_source_path
    )
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V12 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V12 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V12 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = (
        _V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
        ._read_regular_source
    )
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(
                f"manifest-bound V12 source changed: {binding['path']}"
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
            _V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
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


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "source_manifest", "frozen_v11_source_manifest",
        "frozen_v11_source_review", "frozen_v11_execution_authorization",
        "v11_terminal_audit", "v12_preregistration", "science_contract",
        "science_identity", "source_only_checks", "integrity_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V12 source review fields changed")
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
        or value["frozen_v11_source_manifest"]
        != frozen_v11_source_manifest_binding()
        or value["frozen_v11_source_review"] != frozen_v11_review_binding()
        or value["frozen_v11_execution_authorization"]
        != frozen_v11_authorization_binding()
        or value["v11_terminal_audit"] != v11_terminal_audit_binding()
        or value["v12_preregistration"] != preregistration_binding()
        or value["science_contract"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["integrity_checks"] != INTEGRITY_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V12 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v11_source_manifest", "frozen_v11_source_review",
        "frozen_v11_execution_authorization", "v11_terminal_audit",
        "v12_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V12 execution authorization fields changed")
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
        or value["frozen_v11_source_manifest"]
        != frozen_v11_source_manifest_binding()
        or value["frozen_v11_source_review"] != frozen_v11_review_binding()
        or value["frozen_v11_execution_authorization"]
        != frozen_v11_authorization_binding()
        or value["v11_terminal_audit"] != v11_terminal_audit_binding()
        or value["v12_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V12 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V11.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v11_authorization_binding",
    "frozen_v11_review_binding",
    "frozen_v11_science_contract",
    "frozen_v11_source_manifest_binding",
    "normalize_v12_operational_identity",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v11_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v11_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
