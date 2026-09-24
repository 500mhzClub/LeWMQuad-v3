"""Science-identical runtime-interpreter integrity replacement for V1.

This source is deliberately thin.  It preserves the complete frozen
semantic-anchor V1 scientific contract and changes only pre-reservation
runtime identity, governance schemas, and the fresh output root.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
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


_V1 = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v2_frozen_v1_contract",
    FROZEN_V1_CONTRACT_RELATIVE_PATH,
)
for _name in _V1.__all__:
    globals()[_name] = getattr(_V1, _name)

canonical_json_bytes = _V1.canonical_json_bytes
canonical_json_sha256 = _V1.canonical_json_sha256
with_content_sha256 = _V1.with_content_sha256
parse_canonical_json = _V1.parse_canonical_json
is_sha256 = _V1.is_sha256
artifact_binding = _V1.artifact_binding
validate_binding = _V1.validate_binding

IMPLEMENTATION_AUTHOR = "/root"
IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/semantic_v2_prereg_draft",
    "/root/semantic_v2_adapter_diagnosis",
    "/root/semantic_v2_test_closure_author",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement.py"
)
MODEL_RELATIVE_PATH = _V1.MODEL_RELATIVE_PATH
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_source_closure.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = _V1.MODEL_TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

FROZEN_V1_RUNNER_RELATIVE_PATH = _V1.RUNNER_RELATIVE_PATH
FROZEN_V1_LAUNCHER_RELATIVE_PATH = _V1.LAUNCHER_RELATIVE_PATH
FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V1.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_preregistration_"
    "2026-07-27.json"
)
PREREGISTRATION_COMMIT = "3c33a18a36412895fdbd47e773cf1c885f3d49fa"
PREREGISTRATION_FILE_SHA256 = (
    "7ccff034c1cdd2381a5ede94efbbfb30428fb9fef61223c97203a2b78199fd20"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "b1a70a9ddfbb07f106d105c064b6a47f4252a071580a8f780df8bdbd159f8cdf"
)
PREREGISTRATION_BYTE_COUNT = 13_503
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_SCIENCE_IDENTICAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_"
    "STATE_V2_RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_PENDING_SOURCE_"
    "FREEZE_INDEPENDENT_REVIEW_AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH = _V1.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V1_SOURCE_MANIFEST_COMMIT = (
    "f8902d6dbbf2b24df9d4c63fbdaa1957da5c4a80"
)
FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256 = (
    "6eea430558412c48af92ed850cc1cdde31aac7b8c63b9ce1e535657662cec353"
)
FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "a27054cd63a2a86709c548e67a99f3efde8b08e2bac8c5bc7692a4d330d9f965"
)
FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT = 55_423
FROZEN_V1_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V1_SOURCE_COUNT = 155

FROZEN_V1_REVIEW_RELATIVE_PATH = _V1.REVIEW_RELATIVE_PATH
FROZEN_V1_REVIEW_COMMIT = "47e041beb838ab6f23e5fb0ef8b2263667566011"
FROZEN_V1_REVIEW_FILE_SHA256 = (
    "e1de6fd084999feacaf1b2083a3e5b64a5ce9405aa44052c8f0f7308d42c666c"
)
FROZEN_V1_REVIEW_CONTENT_SHA256 = (
    "201380b6f0c5f8d930b28be5e6979b11287f76dbdb20b43cb62be91ab900bb69"
)
FROZEN_V1_REVIEW_BYTE_COUNT = 88_874
FROZEN_V1_REVIEW_STATUS = _V1.REVIEW_STATUS

FROZEN_V1_AUTHORIZATION_RELATIVE_PATH = _V1.AUTHORIZATION_RELATIVE_PATH
FROZEN_V1_AUTHORIZATION_COMMIT = (
    "a9115dce6283b53873c4f52cdf87084cff0ff925"
)
FROZEN_V1_AUTHORIZATION_FILE_SHA256 = (
    "6df5a317c9a5c5dfcf5751ba80a241d9fbf560227198e13a656fcc673e3e931d"
)
FROZEN_V1_AUTHORIZATION_CONTENT_SHA256 = (
    "bd8133963085538cadc281f0a71cb44acdb2fdb9f5e0a48e6b3529c7733e9716"
)
FROZEN_V1_AUTHORIZATION_BYTE_COUNT = 70_131
FROZEN_V1_AUTHORIZATION_STATUS = _V1.AUTHORIZATION_STATUS

FROZEN_V1_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v1_terminal_audit_2026-07-27.json"
)
FROZEN_V1_TERMINAL_AUDIT_COMMIT = (
    "d5f56905202318be1651e8db7b8537909a602432"
)
FROZEN_V1_TERMINAL_AUDIT_FILE_SHA256 = (
    "59456a174c0adf13800fa3998cd19d454d0a57ac007cd3bd43771d655aa9ce5c"
)
FROZEN_V1_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "6a31cef7757816786f62e7b9e770557bbb2f59f9365a41058b481fbcdcf29ab0"
)
FROZEN_V1_TERMINAL_AUDIT_BYTE_COUNT = 9_861
FROZEN_V1_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_RECEIPT_CHAIN_ZERO_WORK_POST_RESERVATION_"
    "INTERPRETER_PREFLIGHT_FAILURE_SEMANTIC_ANCHOR_STATE_V1_CLOSED_NO_RETRY"
)
FROZEN_V1_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_POST_RESERVATION_INTERPRETER_PREFLIGHT_OPERATIONAL_FAILURE_ZERO_"
    "WORK_SCIENTIFICALLY_UNEVALUATED_SEMANTIC_ANCHOR_STATE_V1_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_source_manifest_"
    "2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_source_review_"
    "2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v2_runtime_interpreter_integrity_replacement_execution_"
    "authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V1.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 155 or len(ADDITIVE_SOURCE_PATHS) != 5:
    raise RuntimeError("semantic-anchor V2 source delta changed")
if len(SOURCE_PATHS) != 160 or MODEL_RELATIVE_PATH in ADDITIVE_SOURCE_PATHS:
    raise RuntimeError("semantic-anchor V2 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V1_REVIEW_RELATIVE_PATH,
    FROZEN_V1_AUTHORIZATION_RELATIVE_PATH,
    FROZEN_V1_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_"
    "semantic_anchor_state_v2/rgb_direct_egocentric_bev_signed_boundary_"
    "semantic_anchor_state_probe_v2_runtime_interpreter_integrity_"
    "replacement_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_"
    "RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)
RUNTIME_INTERPRETER_PATH = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64/bin/python"
)
RUNTIME_SYS_PREFIX = (
    "/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64"
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
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_"
    "INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)
RESERVATION_PUBLICATION_FAILURE_STATUS = (
    "TERMINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_"
    "INTEGRITY_RESERVATION_PUBLICATION_FAILURE_NO_RETRY"
)
REVIEW_STATUS = (
    "PASS_SOURCE_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
    "INTERPRETER_INTEGRITY_REPLACEMENT_SCIENCE_AND_CUSTODY"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_SCIENCE_IDENTICAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_"
    "STATE_V2_RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT"
)

SCHEDULE_SCHEMA_ADAPTER_CHANGED = False
SCIENCE_DELTA_COUNT = 0
_FROZEN_V1_SCIENCE_CONTRACT = deepcopy(_V1.science_contract())
FROZEN_V1_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V1_SCIENCE_CONTRACT
)
if FROZEN_V1_SCIENCE_CONTRACT_SHA256 != (
    "2d42031e0586c205cfcae783991a497a4b3f4a5b1c5b8013aa3e65ac5ca673f1"
):
    raise RuntimeError("frozen semantic-anchor V1 science identity changed")


def science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V1_SCIENCE_CONTRACT)


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    return {
        "v1_science_contract_sha256": FROZEN_V1_SCIENCE_CONTRACT_SHA256,
        "v2_science_contract_sha256": canonical_json_sha256(value),
        "normalized_exactly_equals_frozen_v1": (
            value == _FROZEN_V1_SCIENCE_CONTRACT
        ),
        "scientific_delta_count": SCIENCE_DELTA_COUNT,
        "schedule_schema_adapter_changed": SCHEDULE_SCHEMA_ADAPTER_CHANGED,
        "sole_operational_delta": (
            "pre_reservation_exact_ROCm_venv_interpreter_handoff_and_direct_"
            "runner_wrong_interpreter_rejection_plus_fresh_V2_identity"
        ),
        "v1_runtime_output_or_state_reuse_authorized": False,
        "predictor_training_or_evaluation_authorized": False,
    }


build_schedule_identity = _V1.build_schedule_identity
model_config = _V1.model_config
perception_accounting = _V1.perception_accounting

CONTROL_PRELIMINARY = (
    "PRELIMINARY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
    "INTERPRETER_INTEGRITY_REPLACEMENT_DISPATCH_NOT_FINAL_SCIENTIFIC_EVIDENCE"
)
GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_REPLACEMENT_STRUCTURAL_INTEGRITY_GATE_"
        "TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_"
        "RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_STRUCTURAL_INTEGRITY_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_REPLACEMENT_LEARNING_HEALTH_GATE_TERMINAL_NO_"
        "RETRY",
        "CONTINUE_AFTER_UPDATE_100_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_"
        "RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_LEARNING_HEALTH_GATE",
    ),
    400: (
        "FAIL_UPDATE_400_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_REPLACEMENT_ANTI_COLLAPSE_GATE_TERMINAL_NO_"
        "RETRY",
        "CONTINUE_AFTER_UPDATE_400_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_"
        "RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT_ANTI_COLLAPSE_GATE",
    ),
    1_000: (
        "FAIL_UPDATE_1000_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_REPLACEMENT_QUALIFICATION_GATE_TERMINAL_NO_"
        "RETRY",
        "PASS_RGB_ONLY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_REPLACEMENT_PERCEPTION_MECHANISM_ONLY",
    ),
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
    """Preserve V1 gate mathematics while publishing V2 receipt identity."""

    result = _V1.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        update_400=update_400,
        prior_gates_passed=prior_gates_passed,
    )
    result = deepcopy(result)
    if result.get("final_gate_evaluated") is True:
        passed = result.get("passed")
        if type(passed) is not bool:
            raise PermissionError("frozen V1 final gate lost boolean outcome")
        result["control"] = GATE_CONTROLS[update][1 if passed else 0]
        result["gate_mode"] = (
            "FINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
            "INTERPRETER_INTEGRITY_REPLACEMENT_RECEIPT"
        )
    else:
        result["control"] = CONTROL_PRELIMINARY
        result["gate_mode"] = (
            "PRELIMINARY_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
            "INTERPRETER_INTEGRITY_REPLACEMENT_DISPATCH_NOT_FINAL_"
            "SCIENTIFIC_EVIDENCE"
        )
    return result


def validate_failure_status_chain(value: object) -> dict[str, str]:
    fields = ("metrics", "artifact", "result", "completion")
    if type(value) is not dict or tuple(value) != fields:
        raise ValueError("V2 failure status-chain fields changed")
    control = value["metrics"]
    if (
        type(control) is not str
        or control not in FAILURE_CONTROLS
        or any(value[field] != control for field in fields)
    ):
        raise ValueError("V2 failure receipts lost one exact gate control")
    return dict(value)


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V1.runtime_authorization_template())
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
        "v1_runtime_output_reuse": False,
        "v1_retry_resume_repair_or_recovery": False,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": False,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "runtime_interpreter_path": RUNTIME_INTERPRETER_PATH,
        "runtime_sys_prefix": RUNTIME_SYS_PREFIX,
        "schedule_schema_adapter_changed": False,
    }
    return value


SOURCE_ONLY_AUTHORITY = dict(_V1.SOURCE_ONLY_AUTHORITY)
REVIEW_AUTHORITY = dict(_V1.REVIEW_AUTHORITY)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = deepcopy(_V1.EXECUTION_AUTHORITY)
EXECUTION_AUTHORITY.update({
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_signed_boundary_semantic_anchor_state_v1_perception_attempt_only": False,
    "sole_scientific_delta_is_fixed_one_over_64_semantic_anchor": False,
    "architecture_parameters_data_seed_schedule_optimizer_adapter_and_caps_identical_to_frozen_signed_boundary_v1": False,
    "one_fresh_semantic_anchor_v2_runtime_interpreter_integrity_replacement_only": True,
    "science_identical_to_frozen_semantic_anchor_v1": True,
    "exact_reviewed_runtime_interpreter_handoff_only": True,
    "runtime_interpreter_path": RUNTIME_INTERPRETER_PATH,
    "runtime_sys_prefix": RUNTIME_SYS_PREFIX,
    "v1_retry_resume_repair_recovery_or_extension_authorized": False,
    "v1_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_authorized": False,
})


def frozen_v1_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V1_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT,
        "source_count": FROZEN_V1_SOURCE_COUNT,
        "status": FROZEN_V1_SOURCE_MANIFEST_STATUS,
    }


def frozen_v1_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V1_REVIEW_COMMIT,
        "file_sha256": FROZEN_V1_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V1_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V1_REVIEW_BYTE_COUNT,
        "status": FROZEN_V1_REVIEW_STATUS,
    }


def frozen_v1_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V1_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V1_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V1_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V1_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V1_AUTHORIZATION_STATUS,
    }


def frozen_v1_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": FROZEN_V1_TERMINAL_AUDIT_COMMIT,
        "file_sha256": FROZEN_V1_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": FROZEN_V1_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": FROZEN_V1_TERMINAL_AUDIT_BYTE_COUNT,
        "status": FROZEN_V1_TERMINAL_AUDIT_STATUS,
        "classification": FROZEN_V1_TERMINAL_AUDIT_CLASSIFICATION,
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


_READ_REGULAR_SOURCE = (
    _V1._SIGNED._V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
    ._read_regular_source
)
_SAFE_RELATIVE_SOURCE_PATH = (
    _V1._SIGNED._V12._V11._V10._V9._V8._V7._V6._v5._v4._v3._v2._v1
    .safe_relative_source_path
)


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


def validate_frozen_v1_source_closure(
    root: Path = ROOT,
) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("frozen V1 closure must use repository root")
    current = _V1.current_source_bindings(root)
    if (
        current.get(FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen semantic-anchor V1 closure changed")
    return current


def validate_governing_documents(
    root: Path = ROOT,
) -> dict[str, str]:
    current = validate_frozen_v1_source_closure(root)
    for binding in (
        frozen_v1_review_binding(),
        frozen_v1_authorization_binding(),
        frozen_v1_terminal_audit_binding(),
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
    value = parse_canonical_json(raw, name="semantic-anchor V2 source manifest")
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
        raise PermissionError("semantic-anchor V2 manifest fields changed")
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
        or len(bindings) != 160
        or value["source_count"] != 160
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
        raise PermissionError("semantic-anchor V2 source manifest changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path",
            "file_sha256",
            "byte_count",
        }:
            raise PermissionError("semantic-anchor V2 source binding fields changed")
        normalized.append(_SAFE_RELATIVE_SOURCE_PATH(binding["path"]))
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("semantic-anchor V2 source binding changed")
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("semantic-anchor V2 source binding order changed")
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
    "exactly_five_additive_sources_over_frozen_v1_155_sources": True,
    "frozen_v1_model_architecture_parameters_and_initialization_unchanged": True,
    "science_contract_and_schedule_adapter_exactly_frozen_v1": True,
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
        "frozen_v1_source_manifest",
        "frozen_v1_source_review",
        "frozen_v1_execution_authorization",
        "frozen_v1_terminal_audit",
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
        or value["frozen_v1_source_manifest"]
        != frozen_v1_source_manifest_binding()
        or value["frozen_v1_source_review"] != frozen_v1_review_binding()
        or value["frozen_v1_execution_authorization"]
        != frozen_v1_authorization_binding()
        or value["frozen_v1_terminal_audit"]
        != frozen_v1_terminal_audit_binding()
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
        "frozen_v1_source_manifest",
        "frozen_v1_source_review",
        "frozen_v1_execution_authorization",
        "frozen_v1_terminal_audit",
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
        or value["frozen_v1_source_manifest"]
        != frozen_v1_source_manifest_binding()
        or value["frozen_v1_source_review"] != frozen_v1_review_binding()
        or value["frozen_v1_execution_authorization"]
        != frozen_v1_authorization_binding()
        or value["frozen_v1_terminal_audit"]
        != frozen_v1_terminal_audit_binding()
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
    *_V1.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v1_authorization_binding",
    "frozen_v1_review_binding",
    "frozen_v1_source_manifest_binding",
    "frozen_v1_terminal_audit_binding",
    "model_config",
    "perception_accounting",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "validate_authorization",
    "validate_failure_status_chain",
    "validate_frozen_v1_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
