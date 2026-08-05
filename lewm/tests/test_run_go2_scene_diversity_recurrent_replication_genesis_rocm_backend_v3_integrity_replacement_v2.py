from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2
    as runner,
)


@pytest.fixture(autouse=True)
def exact_scientific_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = runner.plan_builder.rocm_execution_environment("scientific")
    keys = (
        set(runner.collector.kernel._SANITIZED_SELECTOR_KEYS)  # noqa: SLF001
        | set(runner.collector.ROCM_ADDITIONAL_SANITIZED_KEYS)
        | set(expected)
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in expected.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(scope="module")
def scientific_plan() -> dict:
    runtime = runner.plan_builder.build_rocm_runtime_bindings()
    frozen = copy.deepcopy(
        runner.plan_builder.predecessor.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return runner.plan_builder.build_scientific_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )


def test_source_closure_freezes_replacement_v1_and_adds_review_only_custody() -> None:
    names = set(runner.SOURCE_PATHS)
    assert {
        "v2_rocm_plan_builder_source",
        "predecessor_v2_qualification_terminal_review",
        "predecessor_v3_qualification_terminal_review",
        "rocm_backend_v3_integrity_replacement_v1_runner",
    } <= names
    assert {
        "rocm_backend_v3_integrity_replacement_v1_runner",
        "rocm_backend_v3_integrity_replacement_v2_runner",
        "predecessor_replacement_v1_qualification_terminal_review",
    } <= names
    assert len(runner.FROZEN_REPLACEMENT_V1_SOURCE_PATHS) == 236
    assert len(runner.SOURCE_PATHS) == 252
    assert runner.SOURCE_PATHS["v2_rocm_preregistration_source"] == (
        runner.V2_PREREGISTRATION_SOURCE
    )
    assert runner.SOURCE_PATHS[
        "predecessor_replacement_v1_qualification_terminal_review"
    ] == runner.plan_builder.REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW
    serialized = json.dumps(
        {name: str(path) for name, path in runner.SOURCE_PATHS.items()}
    )
    assert "backend_v3_qualification_command_receipt" not in serialized
    assert "backend_v3_qualification_authority_2026" not in serialized
    assert "integrity_replacement_v1_qualification_authority_2026" not in serialized
    assert ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3/" not in serialized


def test_failure_evidence_is_flat_and_adds_only_replacement_v1_review() -> None:
    evidence = runner.predecessor_failure_bindings_rocm()
    assert len(evidence) == 12
    assert evidence[
        "predecessor_replacement_v1_qualification_terminal_review"
    ] == (
        runner.collector._standard_replacement_v1_review_binding()  # noqa: SLF001
    )
    assert evidence["v2_rocm_preregistration_source"] == (
        runner.v2_preregistration_source_binding()
    )
    assert not any(
        "backend_v3_qualification/attempt_v1" in value["path"]
        for value in evidence.values()
    )
    assert not any(
        "genesis_rocm_backend_v3_integrity_replacement_v1/attempt_v1"
        in value["path"]
        for value in evidence.values()
    )


def _valid_preflight_result(*, home: str = "/home/andrewknowles") -> dict:
    preflight = {name: "placeholder" for name in runner.ROCM_EGL_PREFLIGHT_FIELDS}
    driver = str(runner.plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT)
    target = str(
        runner.plan_builder.ROCM_RUNTIME_PATHS[
            "rocm_lld_executable"
        ].resolve(strict=True)
    )
    preflight.update(
        {
            "path_ld_lld": target,
            "rocm_path_ld_lld": target,
            "path_ld_lld_driver": driver,
            "rocm_path_ld_lld_driver": driver,
            "lld_driver_entrypoint": driver,
            "lld_driver_link_text": "lld",
            "lld_resolved_target": target,
            "lld_invocation_argv": [driver, "--version"],
            "lld_version_prefix_passed": True,
            "expectation": runner.plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION,
            "identity": {"home": home},
        }
    )
    return {"rocm_egl_preflight": preflight}


def test_replay_semantically_requires_exact_home_receipt() -> None:
    runner._validate_v3_preflight_evidence(_valid_preflight_result())  # noqa: SLF001
    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="driver evidence changed",
    ):
        runner._validate_v3_preflight_evidence(  # noqa: SLF001
            _valid_preflight_result(home="/tmp/wrong-home")
        )


def test_overlay_propagates_expanded_identity_fields_and_restores() -> None:
    lower = runner.predecessor.predecessor
    original = lower.ROCM_IDENTITY_FIELDS
    with runner._configured_predecessor_runner_rocm():  # noqa: SLF001
        assert lower.ROCM_IDENTITY_FIELDS == runner.ROCM_IDENTITY_FIELDS
        assert "home" in lower.ROCM_IDENTITY_FIELDS
    assert lower.ROCM_IDENTITY_FIELDS is original


def test_execute_uses_frozen_science_and_v3_home_guard(
    scientific_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "qualification_result_binding": {
            "path": "/q",
            "sha256": "a" * 64,
            "byte_count": 1,
        },
        "predecessor_v3_qualification_terminal_review_binding": (
            dict(runner.collector._EXACT_V3_REVIEW_BINDING)  # noqa: SLF001
        ),
        "predecessor_replacement_v1_qualification_terminal_review_binding": (
            runner.collector._standard_replacement_v1_review_binding()  # noqa: SLF001
        ),
    }
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(runner, "predecessor_failure_bindings_rocm", lambda: {})
    monkeypatch.setattr(
        runner, "validate_qualification_result_binding", lambda _value: ({}, {})
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "execute_v3",
        lambda actual, *, authority_binding, plan: observed.append((actual, plan))
        or {"status": "PASS"},
    )
    result = runner.execute_rocm(
        authority,
        authority_binding={"path": "/a", "sha256": "b" * 64, "byte_count": 1},
        plan=scientific_plan,
    )
    assert result == {"status": "PASS"}
    assert observed == [(authority, scientific_plan)]


def test_owned_main_calls_v3_validate_and_execute_never_v2_main(
    scientific_plan: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt_root = tmp_path / "owned-v3-main"
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(attempt_root),
    }
    binding = {"path": "/authority", "sha256": "c" * 64, "byte_count": 1}
    executed: list[bool] = []
    inherited: list[bool] = []
    monkeypatch.setattr(
        runner,
        "_validate_authority_rocm",
        lambda *_args, **_kwargs: (authority, binding, scientific_plan),
    )
    monkeypatch.setattr(
        runner,
        "execute_rocm",
        lambda *_args, **_kwargs: executed.append(True)
        or {"status": "PASS_V3_MAIN"},
    )
    monkeypatch.setattr(
        runner.predecessor,
        "main",
        lambda *_args, **_kwargs: inherited.append(True),
    )
    status = runner.main(
        [
            "--authority",
            "/tmp/authority",
            "--expected-authority-sha256",
            "d" * 64,
            "--expected-authority-byte-count",
            "1",
        ]
    )
    assert status == 0
    assert executed == [True]
    assert inherited == []
    assert json.loads(capsys.readouterr().out)["status"] == "PASS_V3_MAIN"
