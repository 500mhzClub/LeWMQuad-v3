"""Contract for one read-only Shared-V5 update-zero diagnostic attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/update0_diag_impl"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
V4_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v4.py"
V4_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v4.py"
V1_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
V4_SOURCE_SHA256 = {
    V4_CONTRACT_RELATIVE_PATH: "ab778c14120ba8bbe33c370658183b93e0a262ed62964f00338cc8369e34cb6e",
    V4_RUNNER_RELATIVE_PATH: "42974a43a765ecb14305dfd4bf630c6bd7c737878fdb89a86c0d40c34e19b0a3",
    "lewm/tests/test_go2_shared_jepa_v5_matched_training_v4.py": "be2d9b425a490be127c433a8ecd31ae7e6e02e0ec46c89dd69296f6fd47a09f8",
    V1_RUNNER_RELATIVE_PATH: "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise PermissionError(f"frozen module changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v4 = _load_exact(V4_CONTRACT_RELATIVE_PATH, "_lewm_update0_diagnostic_exact_v4", V4_SOURCE_SHA256[V4_CONTRACT_RELATIVE_PATH])
_v1 = _v4._v3._v2._v1
V4_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_matched_training_v4_terminal_numeric_failure_audit_2026-07-15.json"
V4_TERMINAL_AUDIT_BINDING = {
    "path": V4_TERMINAL_AUDIT_RELATIVE_PATH,
    "file_sha256": "70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b",
    "content_sha256": "ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4",
    "byte_count": 21_517,
}
V4_ROOT_RELATIVE_PATH = _v4.OUTPUT_ROOT_RELATIVE_PATH
V4_INITIALIZATION_BINDING = {
    "path": "initialization.json",
    "file_sha256": "2b1bdde3edca4a9112948963ec15065d37891c36c07ed49901b71b64a7e0a103",
    "content_sha256": "c42d3ad3295a1077fef523344f06bf03367b7c183be26e073550873269d97751",
    "byte_count": 1_724,
}
V4_SCHEDULE_BINDING = {
    "path": "schedule.json",
    "file_sha256": "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
    "content_sha256": "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
    "byte_count": 607_373,
}
UPDATE0_STATE_SHA256 = "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"
FIRST_PRESENTATION_INDICES = (1550, 2807, 3399, 1468, 1317, 1451, 448, 1842, 3056, 217, 429, 1601, 3965, 2124, 2875, 1382)
OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/update0_transfer_gradient_diagnostic_v1"
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1_execution_authorization_2026-07-15.json"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
MAXIMUM_ATTEMPTS, RETRY_AUTHORIZED = 1, False
SOURCE_PATHS = tuple(dict.fromkeys((CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH, V4_TERMINAL_AUDIT_RELATIVE_PATH, *_v4.SOURCE_PATHS)))
GRADIENT_COMPONENT_PREFIXES = {
    "encoder": "encoder.",
    "evidence_head": "evidence_head.",
    "bev_decoder": "bev_decoder.",
    "predictor": "predictor.",
    "occupancy_head_expected_zero": "occupancy_head.",
}

DOWNSTREAM_DENIALS = {
    "diagnostic_only": True,
    "training_authorized": False,
    "checkpoint_selection_authorized": False,
    "calibration_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "automatic_successor_authorized": False,
    "retry_authorized": False,
}
REVIEW_AUTHORITY = {"diagnostic_execution_authorized": False, "gpu0_diagnostic_authorized": False, "development_payload_read_authorized": False, "camera_update0_migration_authorized": False, "development_rgb_decode_authorized": False, "checkpoint_selection_metric_evaluation_authorized": False, "gradient_backward_diagnostic_authorized": False, "v4_terminal_read_authorized": False, "generated_mutation_authorized": False, **DOWNSTREAM_DENIALS}
EXECUTION_AUTHORITY = {"diagnostic_execution_authorized": True, "gpu0_diagnostic_authorized": True, "development_payload_read_authorized": True, "camera_update0_migration_authorized": True, "development_rgb_decode_authorized": True, "checkpoint_selection_metric_evaluation_authorized": True, "gradient_backward_diagnostic_authorized": True, "v4_terminal_read_authorized": True, "generated_mutation_authorized": True, "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH, **DOWNSTREAM_DENIALS}


def canonical_json_bytes(value: object) -> bytes:
    return _v1.canonical_json_bytes(value)


def canonical_json_sha256(value: object) -> str:
    return _v1.canonical_json_sha256(value)


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return _v1.with_content_sha256(core)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return _v1.parse_canonical_json(raw, name=name)


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v1.artifact_binding(path, raw, content_sha256=content_sha256)


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    return _v1.validate_binding(value, path=path)


def expected_raw_authority() -> dict[str, Any]:
    return {
        "root": _v1.RAW_ROOT_RELATIVE_PATH,
        "manifest": {"path": _v1.RAW_MANIFEST_RELATIVE_PATH, "file_sha256": _v1.RAW_MANIFEST_FILE_SHA256, "content_sha256": _v1.RAW_MANIFEST_CONTENT_SHA256, "byte_count": 311_598},
        "audit": {"path": _v1.RAW_AUDIT_RELATIVE_PATH, "file_sha256": _v1.RAW_AUDIT_FILE_SHA256, "content_sha256": _v1.RAW_AUDIT_CONTENT_SHA256, "byte_count": 26_975},
        "role_counts": copy.deepcopy(_v1.ROLE_COUNTS),
        "grant": {
            "source_raw_grant_remains_false": True,
            "narrow_grant_created_by_this_authorization": True,
            "allowed_roles": ["train", "checkpoint_selection"],
            "allowed_operations": ["development_rgb_decode", "update0_metric_evaluation", "update0_transfer_gradient_diagnostic"],
            "g2_navigation_heldout_or_production_use": False,
        },
    }


def expected_camera_authority() -> dict[str, Any]:
    return {
        "root": _v1.CAMERA_ROOT_RELATIVE_PATH,
        "gate": {"path": _v1.CAMERA_GATE_RELATIVE_PATH, "file_sha256": "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6", "content_sha256": "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b", "byte_count": 7_960},
        "checkpoint": {"path": _v1.CAMERA_CHECKPOINT_RELATIVE_PATH, "file_sha256": "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0", "content_sha256": "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b", "byte_count": 13_777_100},
        "seed": 20260710,
        "fit_size": 320,
        "updates": 40_000,
        "gate_must_pass_all_checks": 26,
    }


def science_contract() -> dict[str, Any]:
    return {
        "purpose": "measure_update0_transfer_physical_state_and_camera_vs_jepa_gradients_without_learning",
        "predecessor": {"terminal_audit": dict(V4_TERMINAL_AUDIT_BINDING), "root": V4_ROOT_RELATIVE_PATH, "initialization": dict(V4_INITIALIZATION_BINDING), "schedule": dict(V4_SCHEDULE_BINDING), "update0_state_sha256": UPDATE0_STATE_SHA256},
        "evaluation": {"role": "checkpoint_selection", "pair_count": 495, "unique_endpoint_count": 924, "scopes": list(_v1.SCOPES), "exact_v1_scope_evaluator": True, "update": 0},
        "gradient_probe": {
            "role": "train",
            "presentation_indices": list(FIRST_PRESENTATION_INDICES),
            "microbatch_size": 4,
            "accumulation_steps": 4,
            "branches": {"camera": "observable_camera_ray_v4.total", "jepa": "established_jepa.total"},
            "components": dict(GRADIENT_COMPONENT_PREFIXES),
            "gradient_clip_norm": 1.0,
            "clip_application": "counterfactual_factor_only_no_gradient_or_state_mutation",
            "camera_jepa_interaction": "per_component_and_global_dot_cosine_sum_norm",
            "optimizer_construction_count": 0,
            "optimizer_step_count": 0,
            "ema_update_count": 0,
            "exact_v4_warning_capture": True,
        },
        "success_inventory": ["access.json", "completed.json", "reservation.json", "result.json"],
        "maximum_attempts": 1,
        "retry_authorized": False,
        "authority": dict(DOWNSTREAM_DENIALS),
    }


def validate_v4_audit(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != V4_TERMINAL_AUDIT_BINDING["file_sha256"] or len(raw) != V4_TERMINAL_AUDIT_BINDING["byte_count"]:
        raise PermissionError("the bound V4 terminal audit changed")
    value = parse_canonical_json(raw, name="V4 terminal numeric-failure audit")
    failure, boundary, authority = value.get("failure", {}), value.get("training_boundary", {}), value.get("authority", {})
    if value.get("content_sha256") != V4_TERMINAL_AUDIT_BINDING["content_sha256"] or value.get("verdict") != "PASS_CONFIRMED_SCIENTIFIC_NUMERIC_GATE_FAILURE" or failure.get("classification") != "scientific_numeric_development_gate_failure_after_complete_promoted_training_and_complete_frozen_checkpoint_selection" or failure.get("retry_authorized") is not False or failure.get("g2_attempted") is not False or failure.get("heldout_open_count") != 0 or boundary.get("complete_update_count") != 8000 or boundary.get("initial_state_sha256") != UPDATE0_STATE_SHA256 or authority.get("automatic_successor_authorized") is not False or authority.get("heldout_authorized") is not False:
        raise PermissionError("the V4 terminal numeric-failure conclusion changed")
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    _v4.current_source_bindings(root)
    audit_raw = (root / V4_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    audit = validate_v4_audit(audit_raw)
    bindings = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}
    if any(bindings.get(path) != digest for path, digest in V4_SOURCE_SHA256.items()) or any(bindings.get(path) != digest for path, digest in audit["source_bindings"].items()):
        raise PermissionError("a frozen V4/V1 source changed")
    return bindings


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {"schema", "status", "implementation_author", "reviewer", "reviewed_sources", "science_contract", "source_only", "findings", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    if value["schema"] != REVIEW_SCHEMA or value["status"] != "PASS" or value["implementation_author"] != IMPLEMENTATION_AUTHOR or type(value["reviewer"]) is not str or not value["reviewer"].startswith("/root/") or value["reviewer"] == IMPLEMENTATION_AUTHOR or value["reviewed_sources"] != dict(expected_sources) or value["science_contract"] != science_contract() or value["source_only"] is not True or value["findings"] != [] or value["authority"] != REVIEW_AUTHORITY or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("independent review did not pass these exact sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "status", "authorizer", "independent_review", "predecessor_audit", "raw", "camera", "experiment", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    if value["schema"] != AUTHORIZATION_SCHEMA or value["status"] != "authorized_one_exact_development_diagnostic_attempt" or type(value["authorizer"]) is not str or not value["authorizer"].startswith("/root/") or value["independent_review"] != dict(review_binding) or value["predecessor_audit"] != V4_TERMINAL_AUDIT_BINDING or value["raw"] != expected_raw_authority() or value["camera"] != expected_camera_authority() or value["experiment"] != science_contract() or value["authority"] != EXECUTION_AUTHORITY or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("execution authorization changed")
    return dict(value)


def evaluate_scopes(scopes: Mapping[str, Any]) -> dict[str, Any]:
    if type(scopes) is not dict or tuple(scopes) != _v1.SCOPES:
        raise ValueError("update-zero scope order changed")
    evaluated = {scope: _v1.evaluate_checkpoint_scope(scopes[scope]) for scope in _v1.SCOPES}
    physical = {scope: all(item >= 0.0 for item in evaluated[scope]["physical_margins"]) for scope in _v1.SCOPES}
    jepa = {scope: evaluated[scope]["jepa_margins"][0] > 0.0 and all(item >= 0.0 for item in evaluated[scope]["jepa_margins"][1:6]) and all(item > 0.0 for item in evaluated[scope]["jepa_margins"][6:]) for scope in _v1.SCOPES}
    return {"scope_evaluations": evaluated, "physical_pass_by_scope": physical, "physical_pass_count": sum(physical.values()), "all_nine_physical_pass": all(physical.values()), "jepa_pass_by_scope": jepa, "jepa_pass_count": sum(jepa.values()), "all_nine_jepa_pass": all(jepa.values()), "full_gate_pass_count": sum(item["eligible"] for item in evaluated.values()), "all_nine_full_gate_pass": all(item["eligible"] for item in evaluated.values())}


CompactDeterminismWarnings = _v4.CompactDeterminismWarnings
