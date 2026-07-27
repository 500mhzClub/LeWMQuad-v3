"""Source-only contract for the science-identical Direct BEV V11 repair.

V11 preserves the complete frozen V10 experiment.  Its sole behavioral
adapter recognizes only the inherited nested observer's exact pre-marker
dispatch; every final or malformed marker state delegates directly to the
frozen V10 gate.  Importing this module grants no runtime or downstream
authority.
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
FROZEN_V10_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
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


_V10 = _source_only_module(
    "_lewm_direct_bev_v11_frozen_v10_contract",
    FROZEN_V10_CONTRACT_RELATIVE_PATH,
)
_FROZEN_V10_SCIENCE_CONTRACT = _V10.science_contract()

for _name in _V10.__all__:
    globals()[_name] = getattr(_V10, _name)

canonical_json_bytes = _V10.canonical_json_bytes
canonical_json_sha256 = _V10.canonical_json_sha256
is_sha256 = _V10.is_sha256
with_content_sha256 = _V10.with_content_sha256
parse_canonical_json = _V10.parse_canonical_json
artifact_binding = _V10.artifact_binding
validate_binding = _V10.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v9_contract_implementation"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_RELATIVE_PATH = _V10.MODEL_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V10_MODEL_RELATIVE_PATH = _V10.MODEL_RELATIVE_PATH
FROZEN_V10_RUNNER_RELATIVE_PATH = _V10.RUNNER_RELATIVE_PATH
FROZEN_V10_LAUNCHER_RELATIVE_PATH = _V10.LAUNCHER_RELATIVE_PATH
FROZEN_V10_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V10.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "4e6d461e22b443c14506d283554e108c83769758"
PREREGISTRATION_FILE_SHA256 = (
    "ee82d1db974252284fe30c4b113e1d1861dd6e8e0605db9b9695295be6fa6953"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "0a664e17c5524a9ad6c371f1f53628b5a880cc1fecf5dd5861b7e9df5d5134be"
)
PREREGISTRATION_BYTE_COUNT = 23_886
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_SCIENCE_IDENTICAL_V11_NESTED_OBSERVER_GATE_"
    "DISPATCH_INTEGRITY_REPLACEMENT_PENDING_SOURCE_FREEZE_INDEPENDENT_"
    "REVIEW_AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH = _V10.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V10_SOURCE_MANIFEST_COMMIT = (
    "317b45246aca17f3c6e65a6159b06361033fe52c"
)
FROZEN_V10_SOURCE_MANIFEST_FILE_SHA256 = (
    "acfe2ea39f30571d57fe129279175773b85405001204715bfb250e229034f302"
)
FROZEN_V10_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "3b436f48259e02220106335e732b31319df1e3ac12e3f71ed16cc2e18c8408cf"
)
FROZEN_V10_SOURCE_MANIFEST_BYTE_COUNT = 46_907
FROZEN_V10_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V10_SOURCE_COUNT = 133

FROZEN_V10_REVIEW_RELATIVE_PATH = _V10.REVIEW_RELATIVE_PATH
FROZEN_V10_REVIEW_COMMIT = "0e1f1ff1c16af2daef73f5538102c64c6485b59a"
FROZEN_V10_REVIEW_FILE_SHA256 = (
    "e074111735bfdff2c24d93c09806fc9a3d7e9bffecf06e3d71f4164a0316c003"
)
FROZEN_V10_REVIEW_CONTENT_SHA256 = (
    "bdcc5cdfbdbfe1b9f7130f76c9981fc6a8b1104bea4fbbb614c25491a6917e06"
)
FROZEN_V10_REVIEW_BYTE_COUNT = 77_670
FROZEN_V10_REVIEW_STATUS = (
    "PASS_SOURCE_AND_FINAL_CLASS_MACRO_GROUNDING_SCIENCE"
)

FROZEN_V10_AUTHORIZATION_RELATIVE_PATH = _V10.AUTHORIZATION_RELATIVE_PATH
FROZEN_V10_AUTHORIZATION_COMMIT = (
    "c6fb866ad25dc37419a7f0cf9779c830ff3da2f4"
)
FROZEN_V10_AUTHORIZATION_FILE_SHA256 = (
    "e362424ed3a3039aab5653c1ebe5c5cba91ebbef8023697bcd60f3c6e22b5596"
)
FROZEN_V10_AUTHORIZATION_CONTENT_SHA256 = (
    "5c6a830328b1189b9b8d7d97d09323f20ac6f9395b6a8a83b14a083e9fb9b180"
)
FROZEN_V10_AUTHORIZATION_BYTE_COUNT = 64_406
FROZEN_V10_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V10_FINAL_CLASS_MACRO_GROUNDING_"
    "PERCEPTION_FALSIFICATION"
)

V10_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding_terminal_audit_2026-07-27.json"
)
V10_TERMINAL_AUDIT_COMMIT = "1a653d86adaa53bab843b46fd2e75e7ef22cf4c6"
V10_TERMINAL_AUDIT_FILE_SHA256 = (
    "a5cfc2317e7736160657cac1532ca0f6881d491298b36bec193537c50c706091"
)
V10_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "3827a5fd6782352155bae14c17ef6b0ef792ad9023fe656dcad78a11f3c86517"
)
V10_TERMINAL_AUDIT_BYTE_COUNT = 9_665
V10_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_NESTED_OBSERVER_UPDATE_ZERO_OPERATIONAL_INTEGRITY_"
    "FAILURE_CLOSES_V10_NO_RETRY"
)
V10_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ONE_SHOT_OPERATIONAL_INTEGRITY_FAILURE_DURING_NESTED_OBSERVATION_"
    "UPDATE_ZERO_BEFORE_ANY_REGISTERED_OBSERVATION_OR_SCIENTIFIC_WORK_V10_"
    "PERMANENTLY_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V10.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 133 or len(SOURCE_PATHS) != 138:
    raise RuntimeError("V11 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V10_REVIEW_RELATIVE_PATH,
    FROZEN_V10_AUTHORIZATION_RELATIVE_PATH,
    V10_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v11/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v11_"
    "nested_observer_gate_dispatch_integrity_replacement_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V11_"
    "NESTED_OBSERVER_GATE_DISPATCH_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
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
    "PASS_SOURCE_SCIENCE_IDENTITY_AND_NESTED_OBSERVER_GATE_DISPATCH_INTEGRITY"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V11_NESTED_OBSERVER_GATE_DISPATCH_"
    "INTEGRITY_REPLACEMENT"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V10.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v10_final_class_macro_grounding_perception_attempt_only": False,
    "v10_retry_resume_repair_recovery_or_extension_authorized": False,
    "v10_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_"
    "authorized": False,
    "one_fresh_v11_nested_observer_gate_dispatch_integrity_replacement_only": (
        True
    ),
    "science_identical_to_frozen_v10": True,
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
SCHEDULE_PREFIX_SHA256 = dict(_V10.SCHEDULE_PREFIX_SHA256)
FROZEN_V10_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V10_SCIENCE_CONTRACT
)
if FROZEN_V10_SCIENCE_CONTRACT_SHA256 != (
    "bf839c0897d73f21b789b8e4c0d9277cba6c2c387e4ccbe347aa4cf91eadff43"
):
    raise PermissionError("frozen V10 science contract identity changed")

CONTROL_PRELIMINARY_NESTED_DISPATCH = (
    "PRELIMINARY_PASS_NESTED_OBSERVER_DISPATCH_NOT_FINAL_SCIENTIFIC_EVIDENCE"
)
NESTED_DISPATCH_REQUIRED_ABSENT_KEYS = (
    "v8_mechanism_receipt_ready",
    "active_training_scope_v8",
)
GATE_DISPATCH_ACCOUNTING = {
    0: {
        "preliminary_call_count": 2,
        "final_delegate_call_count": 1,
        "total_evaluate_gate_call_count": 3,
    },
    50: {
        "preliminary_call_count": 1,
        "final_delegate_call_count": 1,
        "total_evaluate_gate_call_count": 2,
    },
    100: {
        "preliminary_call_count": 1,
        "final_delegate_call_count": 1,
        "total_evaluate_gate_call_count": 2,
    },
    250: {
        "preliminary_call_count": 1,
        "final_delegate_call_count": 1,
        "total_evaluate_gate_call_count": 2,
    },
}
INTEGRITY_REPLACEMENT_DELTA = {
    "science_changed": False,
    "scientific_delta_count": 0,
    "sole_behavioral_delta": "nested_observer_preliminary_gate_dispatch_only",
    "preliminary_dispatch_condition": (
        "both_v8_mechanism_receipt_ready_and_active_training_scope_v8_keys_"
        "absent_by_membership"
    ),
    "preliminary_dispatch_authorizes_scientific_gate_or_training": False,
    "all_other_marker_presence_combinations_delegate_exactly_once_to_v10": True,
    "frozen_v10_final_gate_threshold_metric_control_or_stop_change": False,
    "metrics_mutated_or_copied_before_delegation": False,
    "delegation_exception_caught_or_translated": False,
    "expected_gate_dispatch_accounting": deepcopy(GATE_DISPATCH_ACCOUNTING),
    "v10_runtime_output_open_or_reuse_authorized": False,
    "v10_retry_resume_repair_or_recovery_authorized": False,
    "v10_output_root": _V10.OUTPUT_ROOT_RELATIVE_PATH,
    "v11_output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v11_maximum_attempts": 1,
}

INTEGRITY_REVIEW_CHECKS = {
    "frozen_v10_manifest_and_all_133_sources_rehashed": True,
    "frozen_v10_review_authorization_and_terminal_audit_exact": True,
    "v10_permanently_closed_science_untested_and_runtime_reuse_forbidden": True,
    "v11_preregistration_exact": True,
    "v11_adds_no_model_data_objective_optimizer_schedule_gate_or_threshold_code": (
        True
    ),
    "v11_science_normalizes_exactly_to_frozen_v10": True,
    "preliminary_dispatch_requires_both_marker_keys_absent_by_membership": True,
    "preliminary_dispatch_validates_only_update_and_prior_bool": True,
    "preliminary_dispatch_is_explicitly_nonauthorizing": True,
    "preliminary_thresholds_and_accounting_are_reporting_only_not_applied": True,
    "every_other_presence_combination_delegates_exactly_once_unchanged": True,
    "delegation_does_not_copy_mutate_catch_or_translate": True,
    "nested_then_outer_final_dispatch_call_order_proven": True,
    "update_zero_dispatch_counts_are_two_preliminary_plus_one_final": True,
    "later_observation_dispatch_counts_are_one_preliminary_plus_one_final": (
        True
    ),
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v10_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V10_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V10_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V10_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V10_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V10_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V10_SOURCE_COUNT,
    }


def frozen_v10_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V10_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V10_REVIEW_COMMIT,
        "file_sha256": FROZEN_V10_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V10_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V10_REVIEW_BYTE_COUNT,
        "status": FROZEN_V10_REVIEW_STATUS,
    }


def frozen_v10_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V10_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V10_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V10_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V10_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V10_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V10_AUTHORIZATION_STATUS,
    }


def v10_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V10_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V10_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V10_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V10_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V10_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V10_TERMINAL_AUDIT_STATUS,
        "classification": V10_TERMINAL_AUDIT_CLASSIFICATION,
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


def frozen_v10_science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V10_SCIENCE_CONTRACT)


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V10.runtime_authorization_template())
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
        "v10_runtime_output_reuse": False,
        "v10_retry_resume_repair_or_recovery": False,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": (
            False
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    return value


def science_contract() -> dict[str, Any]:
    """Return the canonical-identical frozen V10 scientific contract."""

    return frozen_v10_science_contract()


def normalize_v11_operational_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V11 experiment differs from its exact contract")
    return frozen_v10_science_contract()


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    normalized = normalize_v11_operational_identity(value)
    return {
        "frozen_v10_science_contract_sha256": (
            FROZEN_V10_SCIENCE_CONTRACT_SHA256
        ),
        "v11_science_contract_sha256": canonical_json_sha256(value),
        "normalized_v11_science_contract_sha256": canonical_json_sha256(
            normalized
        ),
        "normalized_exactly_equals_frozen_v10": (
            normalized == _FROZEN_V10_SCIENCE_CONTRACT
        ),
        "scientific_delta_count": 0,
        "sole_behavioral_delta": INTEGRITY_REPLACEMENT_DELTA[
            "sole_behavioral_delta"
        ],
        "v10_runtime_reuse_authorized": False,
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
    """Dispatch only the exact inherited nested pre-marker observation."""

    both_absent = all(
        key not in metrics for key in NESTED_DISPATCH_REQUIRED_ABSENT_KEYS
    )
    if both_absent:
        if update not in GATE_CONTROLS:
            raise ValueError("update must be one of 0, 50, 100, or 250")
        if type(prior_gates_passed) is not bool:
            raise ValueError("prior_gates_passed must be bool")
        return {
            "update": update,
            "active_training_scope_v8_present": False,
            "passed": True,
            "control": CONTROL_PRELIMINARY_NESTED_DISPATCH,
            "gate_mode": (
                "PRELIMINARY_NESTED_OBSERVER_GATE_DISPATCH_NOT_FINAL_"
                "SCIENTIFIC_EVIDENCE"
            ),
            "v8_mechanism_receipt_ready": False,
            "scientific_gate_evidence": False,
            "execution_training_checkpoint_terminal_pass_or_downstream_"
            "authority": False,
            "must_be_overwritten_by_frozen_outer_v8_final_dispatch": True,
            "final_gate_evaluated": False,
            "thresholds": dict(GATE_THRESHOLDS[update]),
            "thresholds_applied": False,
            "perception_accounting": dict(PERCEPTION_ACCOUNTING[update]),
            "perception_accounting_applied": False,
            "prior_gates_passed_validated_only": prior_gates_passed,
        }
    return _V10.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        prior_gates_passed=prior_gates_passed,
    )


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
) -> dict[str, Any]:
    read = _V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
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


def validate_frozen_v10_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V11 frozen V10 closure must use repository root")
    manifest = _read_bound_json(
        FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH,
        file_sha256=FROZEN_V10_SOURCE_MANIFEST_FILE_SHA256,
        content_sha256=FROZEN_V10_SOURCE_MANIFEST_CONTENT_SHA256,
        byte_count=FROZEN_V10_SOURCE_MANIFEST_BYTE_COUNT,
        status=FROZEN_V10_SOURCE_MANIFEST_STATUS,
    )
    if manifest.get("source_count") != FROZEN_V10_SOURCE_COUNT:
        raise PermissionError("frozen V10 source count changed")
    current = _V10.current_source_bindings(root)
    if current.get(FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V10_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V10 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v10_source_closure(root)
    review = _read_bound_json(
        FROZEN_V10_REVIEW_RELATIVE_PATH,
        file_sha256=FROZEN_V10_REVIEW_FILE_SHA256,
        content_sha256=FROZEN_V10_REVIEW_CONTENT_SHA256,
        byte_count=FROZEN_V10_REVIEW_BYTE_COUNT,
        status=FROZEN_V10_REVIEW_STATUS,
    )
    authorization = _read_bound_json(
        FROZEN_V10_AUTHORIZATION_RELATIVE_PATH,
        file_sha256=FROZEN_V10_AUTHORIZATION_FILE_SHA256,
        content_sha256=FROZEN_V10_AUTHORIZATION_CONTENT_SHA256,
        byte_count=FROZEN_V10_AUTHORIZATION_BYTE_COUNT,
        status=FROZEN_V10_AUTHORIZATION_STATUS,
    )
    _V10.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _V10.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    _read_bound_json(
        V10_TERMINAL_AUDIT_RELATIVE_PATH,
        file_sha256=V10_TERMINAL_AUDIT_FILE_SHA256,
        content_sha256=V10_TERMINAL_AUDIT_CONTENT_SHA256,
        byte_count=V10_TERMINAL_AUDIT_BYTE_COUNT,
        status=V10_TERMINAL_AUDIT_STATUS,
        classification=V10_TERMINAL_AUDIT_CLASSIFICATION,
    )
    _read_bound_json(
        PREREGISTRATION_RELATIVE_PATH,
        file_sha256=PREREGISTRATION_FILE_SHA256,
        content_sha256=PREREGISTRATION_CONTENT_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        status=PREREGISTRATION_STATUS,
    )
    current.update({
        FROZEN_V10_SOURCE_MANIFEST_RELATIVE_PATH: (
            FROZEN_V10_SOURCE_MANIFEST_FILE_SHA256
        ),
        FROZEN_V10_REVIEW_RELATIVE_PATH: FROZEN_V10_REVIEW_FILE_SHA256,
        FROZEN_V10_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V10_AUTHORIZATION_FILE_SHA256
        ),
        V10_TERMINAL_AUDIT_RELATIVE_PATH: V10_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V11 source manifest")
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
        raise PermissionError("V11 source manifest contract changed")
    safe = _V10._V9._V8._V7._V6._v5._v4._v3._v2._v1.safe_relative_source_path
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V11 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V11 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V11 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(
                f"manifest-bound V11 source changed: {binding['path']}"
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
        read = _V10._V9._V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
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
        "reviewed_sources", "source_manifest", "frozen_v10_source_manifest",
        "frozen_v10_source_review", "frozen_v10_execution_authorization",
        "v10_terminal_audit", "v11_preregistration", "science_contract",
        "science_identity", "source_only_checks", "integrity_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V11 source review fields changed")
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
        or value["frozen_v10_source_manifest"]
        != frozen_v10_source_manifest_binding()
        or value["frozen_v10_source_review"] != frozen_v10_review_binding()
        or value["frozen_v10_execution_authorization"]
        != frozen_v10_authorization_binding()
        or value["v10_terminal_audit"] != v10_terminal_audit_binding()
        or value["v11_preregistration"] != preregistration_binding()
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
        raise PermissionError("V11 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v10_source_manifest", "frozen_v10_source_review",
        "frozen_v10_execution_authorization", "v10_terminal_audit",
        "v11_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V11 execution authorization fields changed")
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
        or value["frozen_v10_source_manifest"]
        != frozen_v10_source_manifest_binding()
        or value["frozen_v10_source_review"] != frozen_v10_review_binding()
        or value["frozen_v10_execution_authorization"]
        != frozen_v10_authorization_binding()
        or value["v10_terminal_audit"] != v10_terminal_audit_binding()
        or value["v11_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V11 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V10.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v10_authorization_binding",
    "frozen_v10_review_binding",
    "frozen_v10_science_contract",
    "frozen_v10_source_manifest_binding",
    "normalize_v11_operational_identity",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v10_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v10_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
