from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2
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
        runner.plan_builder.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return runner.plan_builder.build_scientific_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )


def test_source_closure_is_v2_and_contains_no_v1_runtime_payload() -> None:
    assert {
        "rocm_backend_v2_plan_builder",
        "rocm_backend_v2_collector",
        "rocm_backend_v2_qualifier",
        "rocm_backend_v2_runner",
        "predecessor_v1_qualification_terminal_review",
    } <= set(runner.SOURCE_PATHS)
    forbidden = (
        "genesis_rocm_backend_v1_exact_plan",
        "genesis_rocm_backend_v1_qualification_exact_plan",
        ".generated/dev/go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v1",
    )
    paths = [str(path) for path in runner.SOURCE_PATHS.values()]
    assert all(not any(token in path for token in forbidden) for path in paths)
    assert all("sealed" not in Path(path).name.lower() for path in paths)


def test_failure_evidence_adds_only_exact_v1_review_document() -> None:
    evidence = runner.predecessor_failure_bindings_rocm()
    assert evidence["predecessor_v1_qualification_terminal_review"] == (
        runner._standard_binding(  # noqa: SLF001
            runner.plan_builder.V1_QUALIFICATION_TERMINAL_REVIEW_BINDING
        )
    )
    assert not any(
        "genesis_rocm_backend_v1_qualification/attempt_v1" in value["path"]
        for value in evidence.values()
    )


def test_runner_overlay_routes_plan_and_runtime_to_v2_and_restores(
    scientific_plan: dict,
) -> None:
    overrides = runner._configuration_overrides_v2()  # noqa: SLF001
    originals = {name: getattr(runner.predecessor, name) for name in overrides}
    with runner._configured_predecessor_runner_rocm():  # noqa: SLF001
        assert runner.predecessor.plan_builder is runner.plan_builder
        assert runner.predecessor.collector is runner.collector
        assert runner.predecessor.qualifier is runner._QUALIFIER_REPLAY_FACADE  # noqa: SLF001
        assert runner.predecessor.DEFAULT_ATTEMPT_ROOT == (
            runner.plan_builder.DEFAULT_ATTEMPT_ROOT
        )
        runner._validate_plan_rocm(  # noqa: SLF001
            scientific_plan,
            {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID},
        )
    assert all(
        getattr(runner.predecessor, name) is value
        for name, value in originals.items()
    )


def test_replay_facade_exposes_v2_api_and_scene_predecessor() -> None:
    facade = runner._QUALIFIER_REPLAY_FACADE  # noqa: SLF001
    assert facade._configured_qualification_collector is (  # noqa: SLF001
        runner.qualifier._configured_qualification_collector  # noqa: SLF001
    )
    assert facade.predecessor is runner.qualifier.scene_predecessor
    for name in (
        "_scene_slices_v2",
        "_load_scene_result_v2",
        "_scene_expected_counts_v2",
        "_barrier_with_identity_v2",
        "_validate_release_barrier_shape_v2",
        "_rehash_relative_binding_v2",
        "STORED_FRAMES_PER_SCENE",
    ):
        assert hasattr(facade.predecessor, name)


def _valid_preflight_result() -> dict:
    fields = runner.ROCM_EGL_PREFLIGHT_FIELDS
    preflight = {name: "placeholder" for name in fields}
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
        }
    )
    return {"rocm_egl_preflight": preflight}


def test_runner_semantically_validates_new_driver_evidence() -> None:
    result = _valid_preflight_result()
    runner._validate_v2_preflight_evidence(result)  # noqa: SLF001
    changed = copy.deepcopy(result)
    changed["rocm_egl_preflight"]["lld_invocation_argv"][0] = changed[
        "rocm_egl_preflight"
    ]["lld_resolved_target"]
    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="driver evidence changed",
    ):
        runner._validate_v2_preflight_evidence(changed)  # noqa: SLF001


def test_execute_delegates_to_frozen_science_not_v1_public_runner(
    scientific_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "qualification_result_binding": {"path": "/q", "sha256": "a" * 64, "byte_count": 1},
        "predecessor_v1_qualification_terminal_review_binding": (
            runner.collector._standard_v1_review_binding()  # noqa: SLF001
        ),
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "predecessor_failure_bindings_rocm", lambda: {})
    monkeypatch.setattr(
        runner,
        "validate_qualification_result_binding",
        lambda _value: ({}, {}),
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "execute_v3",
        lambda *args, **kwargs: observed.update(
            {"args": args, "kwargs": kwargs}
        )
        or {"status": "PASS"},
    )
    result = runner.execute_rocm(
        authority,
        authority_binding={"path": "/a", "sha256": "b" * 64, "byte_count": 1},
        plan=scientific_plan,
    )
    assert result == {"status": "PASS"}
    assert observed["args"] == (authority,)
    assert observed["kwargs"]["plan"] == scientific_plan


def test_wrong_python_is_rejected_before_scientific_delegate_or_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "scientific_attempt"
    delegated: list[bool] = []
    monkeypatch.setattr(
        runner.plan_builder, "DEFAULT_ATTEMPT_ROOT", attempt_root
    )
    monkeypatch.setattr(
        runner.collector.sys,
        "executable",
        "/tmp/wrong-scientific-venv/bin/python",
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "execute_v3",
        lambda *_args, **_kwargs: delegated.append(True),
    )

    with pytest.raises(
        runner.collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        runner.execute_rocm({}, authority_binding={}, plan={})

    assert delegated == []
    assert not attempt_root.exists()


def test_wrong_python_cli_never_delegates_or_reserves_science_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "scientific_cli_attempt"
    delegated: list[bool] = []
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(
        runner.plan_builder, "DEFAULT_ATTEMPT_ROOT", attempt_root
    )
    monkeypatch.setattr(
        runner.collector.sys,
        "executable",
        "/tmp/wrong-scientific-cli-venv/bin/python",
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "main",
        lambda *_args, **_kwargs: delegated.append(True),
    )

    with pytest.raises(
        runner.collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        runner.main([])

    assert delegated == []
    assert not attempt_root.exists()


def _main_argv() -> list[str]:
    return [
        "--authority",
        "/tmp/v2-scientific-authority.json",
        "--expected-authority-sha256",
        "a" * 64,
        "--expected-authority-byte-count",
        "1",
    ]


@pytest.mark.parametrize(
    "qualification_binding",
    [
        None,
        {"path": "/wrong", "sha256": "b" * 64, "byte_count": 1},
    ],
)
def test_main_rejects_missing_or_mutated_qualification_binding_before_execute(
    qualification_binding: object,
    scientific_plan: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt_root = tmp_path / "invalid-qualification-attempt"
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(attempt_root),
        "qualification_result_binding": qualification_binding,
        "predecessor_v1_qualification_terminal_review_binding": (
            runner.collector._standard_v1_review_binding()  # noqa: SLF001
        ),
    }
    executed: list[bool] = []
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(runner, "predecessor_failure_bindings_rocm", lambda: {})
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_VALIDATE_AUTHORITY_ROCM",
        lambda *_args, **_kwargs: (authority, {}, scientific_plan),
    )
    monkeypatch.setattr(
        runner,
        "execute_rocm",
        lambda *_args, **_kwargs: executed.append(True),
    )

    assert runner.main(_main_argv()) == 1
    assert executed == []
    assert not attempt_root.exists()
    assert "ROCm qualification result binding malformed" in capsys.readouterr().err


def test_main_reaches_exact_v1_review_semantic_check_before_execute(
    scientific_plan: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt_root = tmp_path / "invalid-review-attempt"
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(attempt_root),
        "qualification_result_binding": None,
        "predecessor_v1_qualification_terminal_review_binding": {
            **runner.collector._standard_v1_review_binding(),  # noqa: SLF001
            "sha256": "0" * 64,
        },
    }
    executed: list[bool] = []
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(runner, "predecessor_failure_bindings_rocm", lambda: {})
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_VALIDATE_AUTHORITY_ROCM",
        lambda *_args, **_kwargs: (authority, {}, scientific_plan),
    )
    monkeypatch.setattr(
        runner,
        "execute_rocm",
        lambda *_args, **_kwargs: executed.append(True),
    )

    assert runner.main(_main_argv()) == 1
    assert executed == []
    assert not attempt_root.exists()
    assert "V2 authority V1 terminal-review binding changed" in (
        capsys.readouterr().err
    )


def test_main_success_calls_v2_execute_and_never_inherited_main(
    scientific_plan: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt_root = tmp_path / "successful-owned-main"
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(attempt_root),
    }
    binding = {"path": "/authority", "sha256": "c" * 64, "byte_count": 1}
    executed: list[tuple[object, object, object]] = []
    inherited: list[bool] = []
    monkeypatch.setattr(
        runner,
        "_validate_authority_rocm",
        lambda *_args, **_kwargs: (authority, binding, scientific_plan),
    )
    monkeypatch.setattr(
        runner,
        "execute_rocm",
        lambda actual, *, authority_binding, plan: executed.append(
            (actual, authority_binding, plan)
        )
        or {"status": "PASS_OWNED_V2_MAIN"},
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "main",
        lambda *_args, **_kwargs: inherited.append(True),
    )

    assert runner.main(_main_argv()) == 0
    assert inherited == []
    assert executed == [(authority, binding, scientific_plan)]
    assert json.loads(capsys.readouterr().out) == {
        "status": "PASS_OWNED_V2_MAIN",
        "attempt_root": str(attempt_root),
    }


def test_main_failure_after_reservation_writes_only_v2_terminal(
    scientific_plan: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root = tmp_path / "consumed-v2-attempt"
    authority = {
        "attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(attempt_root),
    }
    binding = {"path": "/authority", "sha256": "d" * 64, "byte_count": 1}
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(
        runner,
        "_validate_authority_rocm",
        lambda *_args, **_kwargs: (authority, binding, scientific_plan),
    )

    def fail_after_reservation(*_args, **_kwargs):
        attempt_root.mkdir()
        raise runner.SceneDiversityRunnerError("forced owned-main failure")

    monkeypatch.setattr(runner, "execute_rocm", fail_after_reservation)

    assert runner.main(_main_argv()) == 1
    terminal = json.loads((attempt_root / "terminal.json").read_text())
    assert terminal == {
        "schema": runner.TERMINAL_SCHEMA,
        "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
        "authorizes_retry_or_resume": False,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": False,
        "result_binding": None,
        "failure": {
            "type": "SceneDiversityRunnerError",
            "message": "forced owned-main failure",
        },
    }
