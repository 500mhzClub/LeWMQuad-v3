from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
V3_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("_test_direct_bev_v4_hook_contract", CONTRACT_PATH)
v3 = _load("_test_direct_bev_v4_frozen_v3_contract", V3_CONTRACT_PATH)


def _synthetic_manifest() -> tuple[dict[str, object], bytes]:
    paths = list(contract.SOURCE_PATHS)
    bindings = [
        {"path": path, "file_sha256": "1" * 64, "byte_count": 1}
        for path in paths
    ]
    core = {
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources": list(
            contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        ),
        "excluded_runtime_categories": list(
            contract.PROHIBITED_RUNTIME_CATEGORIES
        ),
        "source_paths": paths,
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": 91,
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": contract.SOURCE_ONLY_AUTHORITY,
    }
    value = contract.with_content_sha256(core)
    return value, contract.canonical_json_bytes(value) + b"\n"


def test_isolated_import_is_stdlib_source_only() -> None:
    program = f"""
import importlib.util, json, pathlib, sys
path = pathlib.Path({str(CONTRACT_PATH)!r})
spec = importlib.util.spec_from_file_location('_v4_contract_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({{
    'experiment': module.EXPERIMENT_ID,
    'sources': len(module.SOURCE_PATHS),
    'torch': 'torch' in sys.modules,
    'numpy': 'numpy' in sys.modules,
    'PIL': 'PIL' in sys.modules,
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "PIL": False,
        "experiment": (
            "go2_rgb_direct_egocentric_bev_state_jepa_v4_"
            "residual_head_hook_integrity"
        ),
        "numpy": False,
        "sources": 91,
        "torch": False,
    }


def test_frozen_v3_closure_review_authorization_audit_and_amendment_are_exact() -> None:
    current = contract.validate_frozen_v3_source_closure(ROOT)
    assert len(v3.SOURCE_PATHS) == 83
    assert all(current[path] for path in v3.SOURCE_PATHS)
    governed = contract.validate_governing_documents(ROOT)
    assert governed[contract.FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH] == (
        "5669974b4a4b5410bc06ad6f0d3c0038b418755b024be18bfa0b8ece4aa01cb3"
    )
    assert governed[contract.FROZEN_V3_REVIEW_RELATIVE_PATH] == (
        "29f7512f992c89bbe1fce1e3dc4df62d1d83da9794d9cbafdae25fe5d0845418"
    )
    assert governed[contract.FROZEN_V3_AUTHORIZATION_RELATIVE_PATH] == (
        "dd09784bbdfebbf577634b8cd014adcce06a891cc3c939486b2b5b4307bd7f5b"
    )
    assert governed[contract.V3_TERMINAL_AUDIT_RELATIVE_PATH] == (
        "c298a56fe3f4c7ab9d7a02447f6dfdd16ad28c0909b6cd67d6c2b0900bd1f324"
    )
    assert governed[contract.INTEGRITY_AMENDMENT_RELATIVE_PATH] == (
        "87c3c43cf44e71e95e4c9ee5315d721799613f95e528278743be7f9043f071b2"
    )


def test_v4_science_is_v3_exact_except_integrity_and_identity_metadata() -> None:
    v4_science = contract.science_contract()
    v3_science = v3.science_contract()
    for field in (
        "scientific_question",
        "repository_goal",
        "data",
        "loader",
        "model",
        "objective",
        "optimizer",
        "schedule",
        "gates",
        "access_policy",
        "predictor_successor",
        "frozen_v2_integrity_provenance",
    ):
        assert v4_science[field] == v3_science[field]
    assert contract.model_config() == v3.model_config()
    assert contract.objective_contract() == v3.objective_contract()
    assert contract.optimizer_contract() == v3.optimizer_contract()
    assert contract.build_schedule_identity() == v3.build_schedule_identity()
    assert v4_science["integrity_replacement"] == contract.INTEGRITY_DELTA
    assert v4_science["lifecycle"]["integrity_successor_of"] == (
        v3.EXPERIMENT_ID
    )
    assert v4_science["lifecycle"][
        "v3_checkpoint_tensor_trace_or_runtime_output_reuse"
    ] is False


def test_exact_hook_delta_initial_state_gates_caps_and_source_surface() -> None:
    delta = contract.INTEGRITY_DELTA
    assert delta["v3_hook_target"] == "real_model.predictor"
    assert delta["v4_hook_target"] == (
        "real_model.predictor.residual_head"
    )
    assert delta["expected_outer_predictor_forward_hook_count"] == 0
    assert delta["expected_residual_head_forward_hook_count"] == 1
    assert delta["model_output_gradient_parameter_buffer_rng_or_state_changed"] is False
    assert contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256 == (
        "84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a"
    )
    assert contract.MODEL_RELATIVE_PATH == v3.MODEL_RELATIVE_PATH
    assert contract.MODEL_RELATIVE_PATH not in contract.ADDITIVE_SOURCE_PATHS
    assert len(contract.REUSED_SOURCE_PATHS) == 83
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 8
    assert len(contract.SOURCE_PATHS) == 91
    assert contract.GATE_THRESHOLDS == v3.GATE_THRESHOLDS
    assert contract.GATE_CONTROLS == v3.GATE_CONTROLS
    assert contract.CONTROL_PASS == v3.CONTROL_PASS
    assert contract.FAILURE_CONTROLS == v3.FAILURE_CONTROLS
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.GPU_ACTIVE_TIME_CAP_MINUTES == 60
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v3.OUTPUT_ROOT_RELATIVE_PATH


def test_present_and_execution_authority_are_fail_closed() -> None:
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert contract.PRESENT_AUTHORITY["generated_input_access_authorized"] is False
    authority = contract.EXECUTION_AUTHORITY
    assert authority["one_exact_fresh_attempt_authorized"] is True
    assert authority["v3_retry_authorized"] is False
    assert authority["v3_checkpoint_or_runtime_output_reuse_authorized"] is False
    assert authority["science_identical_hook_integrity_replacement_only"] is True
    for field in (
        "g2_authorized",
        "navigation_authorized",
        "heldout_authorized",
        "sealed_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
    ):
        assert authority[field] is False


def test_governing_validation_rejects_terminal_audit_tamper(monkeypatch) -> None:
    original = contract._v3._v2._v1._read_regular_source
    target = ROOT / contract.V3_TERMINAL_AUDIT_RELATIVE_PATH

    def tampered(path: Path) -> bytes:
        raw = original(path)
        return raw + b"\n" if Path(path) == target else raw

    monkeypatch.setattr(
        contract._v3._v2._v1,
        "_read_regular_source",
        tampered,
    )
    with pytest.raises(PermissionError, match="terminal audit"):
        contract.validate_governing_documents(ROOT)


def test_synthetic_manifest_is_exact_and_runtime_free() -> None:
    value, raw = _synthetic_manifest()
    assert contract.validate_source_manifest(raw) == value
    changed = dict(value)
    changed.pop("content_sha256")
    changed["checkpoint_or_tensor_open_count"] = 1
    changed = contract.with_content_sha256(changed)
    with pytest.raises(PermissionError):
        contract.validate_source_manifest(
            contract.canonical_json_bytes(changed) + b"\n"
        )


def test_synthetic_review_and_authorization_validate_exactly() -> None:
    manifest, manifest_raw = _synthetic_manifest()
    manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    required = set(contract.SOURCE_PATHS) | set(
        contract.SOURCE_REVIEW_ADDITIONAL_PATHS
    )
    expected_sources = {path: "2" * 64 for path in required}
    expected_sources[contract.SOURCE_MANIFEST_RELATIVE_PATH] = (
        manifest_binding["file_sha256"]
    )
    expected_sources[contract.FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH] = (
        contract.FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
    )
    expected_sources[contract.FROZEN_V3_REVIEW_RELATIVE_PATH] = (
        contract.FROZEN_V3_REVIEW_FILE_SHA256
    )
    expected_sources[contract.FROZEN_V3_AUTHORIZATION_RELATIVE_PATH] = (
        contract.FROZEN_V3_AUTHORIZATION_FILE_SHA256
    )
    expected_sources[contract.V3_TERMINAL_AUDIT_RELATIVE_PATH] = (
        contract.V3_TERMINAL_AUDIT_FILE_SHA256
    )
    expected_sources[contract.INTEGRITY_AMENDMENT_RELATIVE_PATH] = (
        contract.INTEGRITY_AMENDMENT_FILE_SHA256
    )
    review_core = {
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS_SOURCE_AND_SCIENCE_IDENTICAL_HOOK_INTEGRITY",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/v4_synthetic_reviewer",
        "reviewed_sources": dict(expected_sources),
        "source_manifest": manifest_binding,
        "frozen_v3_source_manifest": contract.frozen_v3_source_manifest_binding(),
        "frozen_v3_source_review": contract.frozen_v3_review_binding(),
        "frozen_v3_execution_authorization": contract.frozen_v3_authorization_binding(),
        "v3_terminal_audit": contract.v3_terminal_audit_binding(),
        "v4_integrity_amendment": contract.integrity_amendment_binding(),
        "science_contract": contract.science_contract(),
        "source_only_checks": {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        },
        "scientific_checks": contract.SCIENTIFIC_REVIEW_CHECKS,
        "findings": [],
        "authority": contract.REVIEW_AUTHORITY,
    }
    review = contract.with_content_sha256(review_core)
    assert contract.validate_review(
        review,
        expected_sources=expected_sources,
        source_manifest_binding=manifest_binding,
    ) == review
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization_core = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/v4_synthetic_authorizer",
        "independent_source_review": review_binding,
        "frozen_v3_source_manifest": contract.frozen_v3_source_manifest_binding(),
        "frozen_v3_source_review": contract.frozen_v3_review_binding(),
        "frozen_v3_execution_authorization": contract.frozen_v3_authorization_binding(),
        "v3_terminal_audit": contract.v3_terminal_audit_binding(),
        "v4_integrity_amendment": contract.integrity_amendment_binding(),
        "runtime_inputs": contract.runtime_authorization_template(),
        "experiment": contract.science_contract(),
        "authority": contract.EXECUTION_AUTHORITY,
    }
    authorization = contract.with_content_sha256(authorization_core)
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == authorization
