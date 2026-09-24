"""Contract for one zero-science-change Camera V5 environment recovery."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1.py"
PREREGISTRATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_preregistration_2026-07-16.md"
PREREGISTRATION_COMMIT = "7ac1c9dc8a4273b380edf61712d60772a2c6e3dc"

V5_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
V5_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
V5_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
V5_SOURCE_COMMIT = "6d001171d3f79fd8703e449272416191aae0c8b5"
V5_SOURCE_SHA256 = {
    V5_CONTRACT_RELATIVE_PATH: "ee732e692823b3bd9e3ac1c36611c976f8961cf6f6cc694cd82d05652351b582",
    V5_RUNNER_RELATIVE_PATH: "3640ca35300ca36485487d6529dd352c76900c47018f7043cb165a1a078d72c4",
    V5_TEST_RELATIVE_PATH: "b835207f046c099f6a2450c51fe55c4a8bcf730d3f486ed1c9866e55e39cb767",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen Camera V5 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen Camera V5 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v5_contract = _load_exact(
    V5_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v5_contract_for_environment_recovery_v1",
    V5_SOURCE_SHA256[V5_CONTRACT_RELATIVE_PATH],
)
_v1 = _v5_contract._v1

# Training/result schemas remain V5 because the science and lifecycle are byte-for-byte V5.
SCHEMA_PREFIX = _v5_contract.SCHEMA_PREFIX
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v5_native_schedule_completion_environment_recovery_v1"
)
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_independent_review_2026-07-16.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_execution_authorization_2026-07-16.json"
REVIEW_SCHEMA = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_independent_review_v1"
AUTHORIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_execution_authorization_v1"

ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_failure_terminal_audit_2026-07-16.json"
FAILED_ROOT_RELATIVE_PATH = _v5_contract.OUTPUT_ROOT_RELATIVE_PATH
FAILED_RESERVATION_RELATIVE_PATH = f"{FAILED_ROOT_RELATIVE_PATH}/reservation.json"
FAILED_FAILURE_RELATIVE_PATH = f"{FAILED_ROOT_RELATIVE_PATH}/failed.json"
FIXED_EVIDENCE_SHA256 = {
    PREREGISTRATION_RELATIVE_PATH: "473aac99c3418048b124c4217f4e289f6cb2ecf25a36886e79c08ed9d51874b7",
    ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH: "3bfd02b66221dd54a4683e6d1836d3a55bf7ceff8f7a02b9e9f3d580b864d7c9",
    FAILED_RESERVATION_RELATIVE_PATH: "0c3e538c79025dadfd65a5b31b8738293c673f3d2c8e499feb87e4f24a814989",
    FAILED_FAILURE_RELATIVE_PATH: "489a2744b2acdd3985e6a8e3d877ffb2b18c9abb15fcb2c494cda8867e56b0f2",
}
FIXED_EVIDENCE_CONTENT_SHA256 = {
    ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH: "f7b3ce34f594547acc054b0e777fc24753d4e4092e7fa725e9eb363d76dbcfa7",
    FAILED_RESERVATION_RELATIVE_PATH: "c5d375d3d54de3d5d00dd0dcd9fe6138444cf86855a7b68ff92b88977207443d",
    FAILED_FAILURE_RELATIVE_PATH: "ee43555fa374e2f2eedeb6f1feee829543bac3cef45fd7bbe736eaa231a89469",
}
FIXED_EVIDENCE_BYTE_COUNT = {
    PREREGISTRATION_RELATIVE_PATH: 4_495,
    ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH: 2_138,
    FAILED_RESERVATION_RELATIVE_PATH: 15_956,
    FAILED_FAILURE_RELATIVE_PATH: 2_188,
}

REVIEW_AUTHORITY = {
    **dict(_v5_contract.REVIEW_AUTHORITY),
    "environment_recovery_authorized": False,
    "visibility_preflight_authorized": False,
}
EXECUTION_AUTHORITY = {
    **dict(_v5_contract.EXECUTION_AUTHORITY),
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "environment_recovery_authorized": True,
    "visibility_preflight_authorized": True,
}
SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *_v5_contract.SOURCE_PATHS,
    *FIXED_EVIDENCE_SHA256,
)))


def canonical_json_bytes(value: object) -> bytes:
    return _v5_contract.canonical_json_bytes(value)


def canonical_json_sha256(value: object) -> str:
    return _v5_contract.canonical_json_sha256(value)


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return _v5_contract.with_content_sha256(core)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return _v5_contract.parse_canonical_json(raw, name=name)


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v5_contract.artifact_binding(path, raw, content_sha256=content_sha256)


def visibility_preflight_contract() -> dict[str, Any]:
    return {
        "status": "PASS_exactly_one_visible_discrete_R9700",
        "python_flags": ["-I", "-B"],
        "selector_environment": {
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "absent",
            "CUDA_VISIBLE_DEVICES": "absent",
            "HSA_OVERRIDE_GFX_VERSION": "absent",
        },
        "native_thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
        "torch_cuda_available": True,
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon AI PRO R9700",
        "normalized_device_name_must_contain": "r9700",
        "tensor_allocation_count": 0,
        "checkpoint_open_count": 0,
        "dataset_open_count": 0,
        "output_root_absent": True,
        "other_generated_mutator_active": False,
        "gpu_management_query_between_probe_and_launch_authorized": False,
    }


def expected_visibility_preflight() -> dict[str, Any]:
    return visibility_preflight_contract()


def evidence_contract() -> dict[str, Any]:
    return {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v5_source_commit": V5_SOURCE_COMMIT,
        "v5_source_sha256": dict(V5_SOURCE_SHA256),
        "fixed_file_sha256": dict(FIXED_EVIDENCE_SHA256),
        "fixed_content_sha256": dict(FIXED_EVIDENCE_CONTENT_SHA256),
        "failed_attempt": {
            "root": FAILED_ROOT_RELATIVE_PATH,
            "exact_paths": ["failed.json", "reservation.json"],
            "exact_directories_including_root": ["."],
            "stage": "update0_reconstruction",
            "reason": "matched training requires exactly one visible GPU",
            "training_update_count": 0,
            "optimizer_step_count": 0,
            "gpu_training_count": 0,
            "checkpoint_snapshot_count": 0,
            "metric_sidecar_count": 0,
            "scientific_or_numeric_evidence_produced": False,
            "qualified_checkpoint_exists": False,
            "root_reuse_or_deletion_authorized": False,
        },
        "visibility_preflight": visibility_preflight_contract(),
    }


def predecessor_contract() -> dict[str, Any]:
    return {
        "v5_predecessor": _v5_contract.predecessor_contract(),
        "v5_science_contract_sha256": canonical_json_sha256(_v5_contract.science_contract()),
        "terminal_environment_failure": evidence_contract()["failed_attempt"],
        "fresh_update0_initialization_not_resume": True,
        "old_root_preserved": True,
        "recovery_root": OUTPUT_ROOT_RELATIVE_PATH,
    }


def science_contract() -> dict[str, Any]:
    return copy.deepcopy(_v5_contract.science_contract())


def science_delta() -> dict[str, Any]:
    return {
        "base_v5_science_contract_sha256": canonical_json_sha256(_v5_contract.science_contract()),
        "training_science_change_count": 0,
        "training_science_changes": [],
        "control_change_count": 0,
        "reporting_change_count": 0,
        "operational_change_count": 2,
        "operational_changes": [
            {"path": "output_root", "before": _v5_contract.OUTPUT_ROOT_RELATIVE_PATH, "after": OUTPUT_ROOT_RELATIVE_PATH},
            {"path": "pre_reservation_visibility_evidence", "before": "not_required", "after": visibility_preflight_contract()},
        ],
        "architecture_loss_data_sampling_schedule_seed_initialization_optimizer_or_threshold_changes": [],
    }


def control_contract() -> dict[str, Any]:
    return copy.deepcopy(_v5_contract.control_contract())


def reporting_contract() -> dict[str, Any]:
    return copy.deepcopy(_v5_contract.reporting_contract())


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v5_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V5_SOURCE_SHA256.items()):
        raise PermissionError("frozen Camera V5 source binding changed")
    result = dict(inherited)
    for path in SOURCE_PATHS:
        if path in result:
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"recovery review input is not one regular file: {path}")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in FIXED_EVIDENCE_SHA256:
            if digest != FIXED_EVIDENCE_SHA256[path] or len(raw) != FIXED_EVIDENCE_BYTE_COUNT[path]:
                raise PermissionError(f"fixed recovery evidence changed: {path}")
            expected_content = FIXED_EVIDENCE_CONTENT_SHA256.get(path)
            if expected_content is not None:
                parsed = parse_canonical_json(raw, name=f"fixed recovery evidence {path}")
                if parsed.get("content_sha256") != expected_content:
                    raise PermissionError(f"fixed recovery evidence content changed: {path}")
        result[path] = digest
    if set(result) != set(SOURCE_PATHS):
        raise PermissionError("Camera V5 environment-recovery source closure changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer", "reviewed_sources",
        "predecessor", "science_contract", "science_delta", "evidence", "visibility_preflight",
        "reporting_contract", "control_contract", "source_only", "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("environment-recovery independent review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA or value["status"] != "PASS"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["predecessor"] != predecessor_contract()
        or value["science_contract"] != science_contract() or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract() or value["visibility_preflight"] != visibility_preflight_contract()
        or value["reporting_contract"] != reporting_contract() or value["control_contract"] != control_contract()
        or value["source_only"] is not True or value["findings"] != [] or value["authority"] != REVIEW_AUTHORITY
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not pass exact environment-recovery sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_review", "predecessor", "raw", "camera",
        "experiment", "science_delta", "evidence", "visibility_preflight", "reporting_contract",
        "control_contract", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("environment-recovery authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_camera_v5_environment_recovery_v1_attempt"
        or type(authorizer) is not str or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_review"] != dict(review_binding)
        or value["predecessor"] != predecessor_contract()
        or value["raw"] != expected_raw_authority() or value["camera"] != expected_camera_authority()
        or value["experiment"] != science_contract() or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract() or value["visibility_preflight"] != visibility_preflight_contract()
        or value["reporting_contract"] != reporting_contract() or value["control_contract"] != control_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("environment-recovery execution authorization changed")
    return dict(value)


def expected_raw_authority() -> dict[str, Any]:
    return _v5_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    return _v5_contract.expected_camera_authority()


def __getattr__(name: str) -> Any:
    """Delegate every unchanged V5 training, control, and lifecycle contract."""
    return getattr(_v5_contract, name)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "control_contract",
    "current_source_bindings", "evidence_contract", "expected_camera_authority",
    "expected_raw_authority", "expected_visibility_preflight", "parse_canonical_json",
    "predecessor_contract", "reporting_contract", "science_contract", "science_delta",
    "validate_authorization", "validate_review", "visibility_preflight_contract", "with_content_sha256",
]
