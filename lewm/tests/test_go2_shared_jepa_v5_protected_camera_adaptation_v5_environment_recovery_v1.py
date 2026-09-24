from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lewm.benchmarks import (
    go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1
    as contract,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OUTPUT_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v5_native_schedule_completion_environment_recovery_v1"
)


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_test_protected_camera_adaptation_v5_environment_recovery_v1_runner",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(sources: dict[str, str]) -> dict:
    return contract.with_content_sha256(
        {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/camera_v5_environment_recovery_roundtrip_reviewer",
            "reviewed_sources": sources,
            "predecessor": contract.predecessor_contract(),
            "science_contract": contract.science_contract(),
            "science_delta": contract.science_delta(),
            "evidence": contract.evidence_contract(),
            "visibility_preflight": contract.visibility_preflight_contract(),
            "reporting_contract": contract.reporting_contract(),
            "control_contract": contract.control_contract(),
            "source_only": True,
            "findings": [],
            "authority": dict(contract.REVIEW_AUTHORITY),
        }
    )


def test_recovery_has_zero_science_control_or_reporting_delta_and_only_preregistered_operations() -> None:
    assert contract.science_contract() == contract._v5_contract.science_contract()
    assert contract.control_contract() == contract._v5_contract.control_contract()
    assert contract.reporting_contract() == contract._v5_contract.reporting_contract()
    assert contract.canonical_json_sha256(contract.science_contract()) == (
        "d5f1ae7da90c505aca4fb6f0bc10c382d7d2a223ba6217b0b89b608a6dd1da76"
    )
    assert contract.canonical_json_sha256(contract.control_contract()) == (
        "3c7b72318aef6cdec2be4fa4e4c627e1a607b7685d3466dcac4a8ed2f41bd6be"
    )
    assert contract.canonical_json_sha256(contract.reporting_contract()) == (
        "cb9eb1d162b97d2005d552d4189234965a8b4b5b7e1bf6a3a82559601f2d2eed"
    )
    delta = contract.science_delta()
    assert delta["training_science_change_count"] == 0
    assert delta["training_science_changes"] == []
    assert delta["control_change_count"] == 0
    assert delta["reporting_change_count"] == 0
    assert delta["operational_change_count"] == 2
    assert [item["path"] for item in delta["operational_changes"]] == [
        "output_root",
        "pre_reservation_visibility_evidence",
    ]
    assert delta[
        "architecture_loss_data_sampling_schedule_seed_initialization_optimizer_or_threshold_changes"
    ] == []
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == EXPECTED_OUTPUT_ROOT
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != contract._v5_contract.OUTPUT_ROOT_RELATIVE_PATH


def test_source_and_fixed_environment_failure_evidence_closure_is_exact() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert set(contract.V5_SOURCE_SHA256.items()) <= set(bindings.items())
    assert set(contract.FIXED_EVIDENCE_SHA256.items()) <= set(bindings.items())

    evidence = contract.evidence_contract()
    assert evidence["v5_source_commit"] == contract.V5_SOURCE_COMMIT
    assert evidence["v5_source_sha256"] == contract.V5_SOURCE_SHA256
    assert evidence["fixed_file_sha256"] == contract.FIXED_EVIDENCE_SHA256
    assert evidence["fixed_content_sha256"] == contract.FIXED_EVIDENCE_CONTENT_SHA256
    assert evidence["fixed_file_sha256"][
        contract.ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH
    ] == (
        "3bfd02b66221dd54a4683e6d1836d3a55bf7ceff8f7a02b9e9f3d580b864d7c9"
    )
    assert evidence["fixed_content_sha256"][
        contract.ENVIRONMENT_FAILURE_AUDIT_RELATIVE_PATH
    ] == (
        "f7b3ce34f594547acc054b0e777fc24753d4e4092e7fa725e9eb363d76dbcfa7"
    )


def test_canonical_review_and_authorization_bind_the_visibility_preflight() -> None:
    sources = contract.current_source_bindings(ROOT)
    review = _review(sources)
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    parsed_review = contract.parse_canonical_json(
        review_raw, name="round-trip Camera V5 environment-recovery review"
    )
    assert contract.validate_review(parsed_review, expected_sources=sources) == parsed_review
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    visibility_preflight = contract.visibility_preflight_contract()
    authorization = contract.with_content_sha256(
        {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": "authorized_one_exact_camera_v5_environment_recovery_v1_attempt",
            "authorizer": "/root/camera_v5_environment_recovery_roundtrip_authorizer",
            "independent_review": review_binding,
            "predecessor": contract.predecessor_contract(),
            "raw": contract.expected_raw_authority(),
            "camera": contract.expected_camera_authority(),
            "experiment": contract.science_contract(),
            "science_delta": contract.science_delta(),
            "evidence": contract.evidence_contract(),
            "reporting_contract": contract.reporting_contract(),
            "control_contract": contract.control_contract(),
            "visibility_preflight": visibility_preflight,
            "authority": dict(contract.EXECUTION_AUTHORITY),
        }
    )
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    parsed_authorization = contract.parse_canonical_json(
        authorization_raw,
        name="round-trip Camera V5 environment-recovery authorization",
    )
    assert contract.validate_authorization(
        parsed_authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == parsed_authorization
    changed = dict(authorization)
    changed["visibility_preflight"] = {
        **visibility_preflight,
        "visible_device_count": 2,
    }
    changed.pop("content_sha256")
    with pytest.raises(PermissionError):
        contract.validate_authorization(
            contract.with_content_sha256(changed),
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_thin_runner_restores_the_exact_v5_contract_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    original_contract = runner._v5.contract

    def fail_parent(**kwargs):
        assert kwargs == {
            "review_file_sha256": "a" * 64,
            "authorization_file_sha256": "b" * 64,
        }
        assert runner._v5.contract is runner.contract
        raise RuntimeError("synthetic environment-recovery parent failure")

    monkeypatch.setattr(runner, "_BASE_V5_RUN_PARENT", fail_parent)
    with pytest.raises(RuntimeError, match="synthetic environment-recovery parent failure"):
        runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
        )
    assert runner._v5.contract is original_contract


def test_isolated_runner_import_is_accelerator_free_and_does_not_create_the_new_root() -> None:
    output = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    before = output.exists()
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"""
import importlib.util,json,sys
p={str(path)!r}; s=importlib.util.spec_from_file_location('_isolated_camera_v5_recovery',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(json.dumps(sorted(set(sys.modules)&{{'torch','numpy','PIL','cv2'}})))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == []
    assert output.exists() is before
