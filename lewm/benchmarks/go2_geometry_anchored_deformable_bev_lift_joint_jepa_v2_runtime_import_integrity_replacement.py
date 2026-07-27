"""Source-only contract for the science-identical joint-JEPA V2 import fix.

V2 reuses the complete frozen V1 science and model.  Its sole implementation
delta is the lifetime of the repository import root while the post-reservation
runtime stack is loaded in the isolated interpreter.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)


def _source_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v1 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v2_import_frozen_v1_contract",
    FROZEN_V1_CONTRACT_RELATIVE_PATH,
)
for _name in _v1.__all__:
    globals()[_name] = getattr(_v1, _name)

# Private source-only helpers used by the inherited launcher and validators.
_read_regular_source = _v1._read_regular_source
_source_freeze_commit = _v1._source_freeze_commit
_validate_artifact_binding = _v1._validate_artifact_binding


IMPLEMENTATION_AUTHORS = ("/root",)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement"
)
EXPERIMENT_ID = (
    "geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement"
)

FROZEN_V1_SOURCE_COMMIT = "638fc22118f19e24e9a580b79873833d10fd51f8"
FROZEN_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
FROZEN_V1_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_closure.py"
)
FROZEN_V1_MODEL_RELATIVE_PATH = _v1.MODEL_RELATIVE_PATH
FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH = _v1.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V1_SOURCE_MANIFEST_COMMIT = FROZEN_V1_SOURCE_COMMIT
FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256 = (
    "5f5a8931ca9563628c3d1356bb202013830251ec64afca9fee2719c5fd3976a7"
)
FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "003e149244dba7fc336457240831929dab3228defcfd79225b7a98a76df59582"
)
FROZEN_V1_SOURCE_BINDINGS_SHA256 = (
    "a1e0787f566f2c06c7d9e45e30d5d5053be79c1ab8e6d24691c2959a4a5e2d54"
)
FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT = 22_933
FROZEN_V1_SOURCE_COUNT = 74

FROZEN_V1_REVIEW_RELATIVE_PATH = _v1.REVIEW_RELATIVE_PATH
FROZEN_V1_REVIEW_COMMIT = "325ecbca05306c060a3ebb686afca2b45643e924"
FROZEN_V1_REVIEW_FILE_SHA256 = (
    "a11ffc2bafa2e59860d414a0bea64464ff4081e351509e1ef3cf679a9b94d783"
)
FROZEN_V1_REVIEW_CONTENT_SHA256 = (
    "a53ad943e0e8d213b0d948bd16caf228d6f7dfa283eaae24216d2b04ce8bd0c3"
)
FROZEN_V1_REVIEW_BYTE_COUNT = 25_961

FROZEN_V1_AUTHORIZATION_RELATIVE_PATH = _v1.AUTHORIZATION_RELATIVE_PATH
FROZEN_V1_AUTHORIZATION_COMMIT = (
    "41a61bb29e0239e54ff76a3cb8384b0062f87783"
)
FROZEN_V1_AUTHORIZATION_FILE_SHA256 = (
    "c7647c65738298ad68e29173f8a7ebe797322c13ffcb1965f5199dcf7ac7eddb"
)
FROZEN_V1_AUTHORIZATION_CONTENT_SHA256 = (
    "31b0e8b6b14719966852fe2995566c6b904d86a6357b20ce9f0bd9159f36ac53"
)
FROZEN_V1_AUTHORIZATION_BYTE_COUNT = 23_692

V1_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "runtime_import_terminal_audit_2026-07-27.json"
)
V1_TERMINAL_AUDIT_COMMIT = "605198aa253b0ec98bccfd81af7cdb68dd48b48e"
V1_TERMINAL_AUDIT_FILE_SHA256 = (
    "59ee565175ab9bb3718ada88a7d195fd85cc7855b8355e1d0151f5a6ec01332d"
)
V1_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "5427c28a4a4624cc7d786d909e87fda68b8716290456f0bbecf287118cf87f5f"
)
V1_TERMINAL_AUDIT_BYTE_COUNT = 7_693
V1_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_ZERO_EXPOSURE_OPERATIONAL_IMPORT_FAILURE_V1_CONSUMED_"
    "CLOSED_NO_RETRY"
)
V1_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ZERO_EXPOSURE_OPERATIONAL_IMPORT_FAILURE_NOT_A_SCIENTIFIC_OR_"
    "MECHANISM_FAILURE"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_preregistration_2026-07-27.md"
)
PREREGISTRATION_COMMIT = "3408d02769e8094d2c518ef06ef7fc3527401c61"
PREREGISTRATION_FILE_SHA256 = (
    "ff6ec959a580089332cab327a57e1e62e3489b90db6db7dbd56babbb2f2eebe7"
)
PREREGISTRATION_BYTE_COUNT = 8_082

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v1.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != FROZEN_V1_SOURCE_COUNT or len(SOURCE_PATHS) != 79:
    raise PermissionError("V2 must be the frozen 74-source V1 closure plus five files")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

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
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_"
    "V2_RUNTIME_IMPORT_INTEGRITY_REPLACEMENT"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V2_RUNTIME_"
    "IMPORT_INTEGRITY_REPLACEMENT_OPERATIONAL_FAILURE_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement/attempt_v1"
)
EXECUTION_AUTHORITY = {
    **_v1.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v1_retry_authorized": False,
    "science_identical_runtime_import_integrity_replacement_only": True,
}
SOURCE_ONLY_AUTHORITY = {
    **_v1.SOURCE_ONLY_AUTHORITY,
    "isolated_cpu_import_preflight_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

FROZEN_V1_SCIENCE_CONTRACT_SHA256 = (
    "f839076bf7f9db9e9f211703323436f4b607cca21e2e60fb228e4d174c699fa3"
)
FROZEN_V1_SCIENCE_COMPONENT_SHA256 = {
    "model": "595d91a6fc9ae985378ff480780bf7ad5a9beeb3c7f35ab012c010bb74162f39",
    "objective": "93c73c1f1a91de70699f634821159d4d544431b45faa469202016fa0b9fd7ba8",
    "optimizer": "2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34",
    "schedule": "bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3",
    "gate_thresholds": (
        "0c485c0bccb88873c0ff76a1061a315420b6c27c4865b259d3b4c6f374862bd0"
    ),
}
INTEGRITY_DELTA = {
    "scope": "post_reservation_runtime_import_root_lifetime_only",
    "v1_failure": "ModuleNotFoundError: No module named 'lewm'",
    "root_present_during_matched_source_load": True,
    "root_present_during_matched_load_runtime": True,
    "root_present_during_schedule_adapter_source_load": True,
    "root_present_during_unchanged_v1_model_source_load": True,
    "original_sys_path_restored_on_success_and_exception": True,
    "pre_and_post_source_rehash_preserved": True,
    "model_data_seed_schedule_losses_thresholds_initialization_or_caps_changed": False,
}
IMPORT_PREFLIGHT_REQUIREMENTS = {
    "exact_runtime_interpreter": RUNTIME_INTERPRETER_PATH,
    "interpreter_arguments": list(RUNTIME_INTERPRETER_ARGUMENTS),
    "post_reservation_stack_imported": True,
    "sys_path_restored_exactly": True,
    "canonical_root_count_during_lazy_import": 1,
    "generated_inputs_opened": [],
    "checkpoints_or_tensors_opened": [],
    "runtime_outputs_or_traces_opened": [],
    "gpu_queries_or_work_performed": [],
    "heldout_or_sealed_opened": [],
}
SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v1_manifest_and_all_74_sources_rehashed": True,
    "v1_review_and_consumed_authorization_revalidated": True,
    "v1_terminal_audit_exact_and_zero_exposure": True,
    "v1_permanently_closed_and_v2_is_not_a_retry": True,
    "runtime_import_root_lifetime_is_the_only_implementation_delta": True,
    "exact_v1_model_reused_without_new_model_file": True,
    "model_objective_optimizer_schedule_and_gate_hashes_equal_v1": True,
    "data_seed_initialization_losses_thresholds_and_caps_equal_v1": True,
    "complete_failure_receipts_and_downstream_denials_preserved": True,
    "distinct_absent_before_reservation_v2_output_root": True,
    "isolated_post_stack_import_preflight_passed_without_runtime_inputs": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def v1_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V1_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V1_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V1_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V1_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V1_TERMINAL_AUDIT_BYTE_COUNT,
    }


def frozen_v1_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V1_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT,
        "source_bindings_sha256": FROZEN_V1_SOURCE_BINDINGS_SHA256,
        "source_count": FROZEN_V1_SOURCE_COUNT,
    }


def frozen_v1_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V1_REVIEW_COMMIT,
        "file_sha256": FROZEN_V1_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V1_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V1_REVIEW_BYTE_COUNT,
    }


def frozen_v1_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V1_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V1_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V1_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V1_AUTHORIZATION_BYTE_COUNT,
    }


def validate_frozen_v1_source_and_authority(
    root: Path = ROOT,
) -> dict[str, str]:
    manifest_raw = _read_regular_source(
        root / FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH
    )
    if (
        len(manifest_raw) != FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_raw).hexdigest()
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V1 source manifest raw identity changed")
    manifest = _v1.validate_source_manifest(manifest_raw, root)
    if (
        manifest.get("status") != "PASS_SOURCE_CLOSURE"
        or manifest.get("content_sha256")
        != FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("source_bindings_sha256")
        != FROZEN_V1_SOURCE_BINDINGS_SHA256
        or manifest.get("source_count") != FROZEN_V1_SOURCE_COUNT
    ):
        raise PermissionError("frozen V1 source manifest conclusion changed")
    manifest_binding = _v1.artifact_binding(
        FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
    )

    review_raw = _read_regular_source(root / FROZEN_V1_REVIEW_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V1_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V1_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V1 review raw identity changed")
    review = _v1.validate_review(review_raw, manifest_binding, root=root)
    if review.get("content_sha256") != FROZEN_V1_REVIEW_CONTENT_SHA256:
        raise PermissionError("frozen V1 review conclusion changed")
    review_binding = _v1.artifact_binding(
        FROZEN_V1_REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=FROZEN_V1_REVIEW_CONTENT_SHA256,
    )

    authorization_raw = _read_regular_source(
        root / FROZEN_V1_AUTHORIZATION_RELATIVE_PATH
    )
    if (
        len(authorization_raw) != FROZEN_V1_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V1_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V1 authorization raw identity changed")
    authorization = _v1.validate_authorization(
        authorization_raw, review_binding, root=root
    )
    if (
        authorization.get("content_sha256")
        != FROZEN_V1_AUTHORIZATION_CONTENT_SHA256
    ):
        raise PermissionError("frozen V1 authorization conclusion changed")
    return {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }


def model_config() -> dict[str, Any]:
    return _v1.model_config()


def objective_contract() -> dict[str, Any]:
    return _v1.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v1.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v1.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v1.runtime_authorization_template()


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v1.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "v2_preregistration": preregistration_binding(),
        "v1_source_manifest": frozen_v1_source_manifest_binding(),
        "v1_source_review": frozen_v1_review_binding(),
        "v1_consumed_authorization": frozen_v1_authorization_binding(),
        "v1_terminal_audit": v1_terminal_audit_binding(),
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_replacement_of": _v1.EXPERIMENT_ID,
        "v1_retry": False,
    }
    value["integrity_replacement"] = {
        **INTEGRITY_DELTA,
        "frozen_v1_science_contract_sha256": FROZEN_V1_SCIENCE_CONTRACT_SHA256,
        "frozen_v1_science_component_sha256": dict(
            FROZEN_V1_SCIENCE_COMPONENT_SHA256
        ),
    }
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = _v1.validate_governing_documents(root)
    validate_frozen_v1_source_and_authority(root)

    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V2 preregistration changed")

    audit_raw = _read_regular_source(root / V1_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(audit_raw) != V1_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest()
        != V1_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V1 terminal audit raw identity changed")
    audit = _v1.parse_canonical_json(audit_raw, name="V1 terminal audit")
    zero = audit.get("zero_exposure_and_zero_work", {})
    if (
        audit.get("content_sha256") != V1_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != V1_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V1_TERMINAL_AUDIT_CLASSIFICATION
        or any(
            zero.get(name) != 0
            for name in (
                "updates", "presentations", "objective_evaluations",
                "backward_calls", "optimizer_updates", "ema_updates",
            )
        )
        or audit.get("scientific_conclusion", {}).get(
            "scientific_evidence_produced"
        ) is not False
        or audit.get("closure", {}).get("v1_closed") is not True
    ):
        raise PermissionError("V1 terminal audit conclusion changed")

    result.update({
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH:
            FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        FROZEN_V1_REVIEW_RELATIVE_PATH: FROZEN_V1_REVIEW_FILE_SHA256,
        FROZEN_V1_AUTHORIZATION_RELATIVE_PATH:
            FROZEN_V1_AUTHORIZATION_FILE_SHA256,
        V1_TERMINAL_AUDIT_RELATIVE_PATH: V1_TERMINAL_AUDIT_FILE_SHA256,
    })
    return result


def validate_source_manifest(
    raw: bytes, root: Path = ROOT
) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_PATHS)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
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
    ):
        raise PermissionError("V2 source manifest contract changed")
    for relative, binding in zip(SOURCE_PATHS, bindings, strict=True):
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256", "byte_count"}
            or binding.get("path") != relative
            or safe_relative_source_path(relative) != relative
            or not is_sha256(binding.get("file_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V2 source binding changed")
        payload = _read_regular_source(root / relative)
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound V2 source changed: {relative}")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    result = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def validate_review(
    raw: bytes,
    manifest_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 source review")
    expected_manifest = _validate_artifact_binding(
        dict(manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    if expected_manifest != artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    ):
        raise PermissionError("V2 review manifest binding changed")
    expected_sources = current_source_bindings(root)
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "v1_source_manifest", "v1_source_review",
        "v1_consumed_authorization", "v1_terminal_audit", "science_contract",
        "isolated_import_preflight", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status") != "PASS_SOURCE_AND_SCIENCE_IDENTICAL_IMPORT_FIX"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != expected_sources
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("v1_source_manifest")
        != frozen_v1_source_manifest_binding()
        or value.get("v1_source_review") != frozen_v1_review_binding()
        or value.get("v1_consumed_authorization")
        != frozen_v1_authorization_binding()
        or value.get("v1_terminal_audit") != v1_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("isolated_import_preflight")
        != IMPORT_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != {
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("V2 source review did not pass exact import scope")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 execution authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=_v1.parse_canonical_json(
            review_raw, name="V2 source review"
        )["content_sha256"],
    ):
        raise PermissionError("V2 authorization review binding changed")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    manifest_binding = artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    review = validate_review(review_raw, manifest_binding, root=root)
    fields = {
        "schema", "status", "authorizer", "source_freeze_commit",
        "independent_source_review", "preregistration", "v1_terminal_audit",
        "runtime_inputs", "experiment", "authority", "content_sha256",
    }
    authorizer = value.get("authorizer")
    source_commit = _source_freeze_commit(
        review.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, review["reviewer"]}
        or value.get("source_freeze_commit") != source_commit
        or value.get("independent_source_review") != expected_review
        or value.get("preregistration") != preregistration_binding()
        or value.get("v1_terminal_audit") != v1_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("V2 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v1.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity", "current_source_bindings",
    "frozen_v1_authorization_binding", "frozen_v1_review_binding",
    "frozen_v1_source_manifest_binding", "model_config",
    "objective_contract", "optimizer_contract", "preregistration_binding",
    "runtime_authorization_template", "science_contract",
    "v1_terminal_audit_binding", "validate_authorization",
    "validate_frozen_v1_source_and_authority", "validate_governing_documents",
    "validate_review", "validate_source_manifest",
})
