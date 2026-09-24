from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts import (
    qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2
    as qualifier,
)


def _set_exact_qualification_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = qualifier.plan_builder.rocm_execution_environment(
        "qualification"
    )
    keys = (
        set(qualifier.collector.kernel._SANITIZED_SELECTOR_KEYS)  # noqa: SLF001
        | set(qualifier.collector.ROCM_ADDITIONAL_SANITIZED_KEYS)
        | set(expected)
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in expected.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def exact_qualification_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_qualification_environment(monkeypatch)


@pytest.fixture(scope="module")
def qualification_plan() -> dict:
    runtime = qualifier.plan_builder.build_rocm_runtime_bindings()
    frozen = copy.deepcopy(
        qualifier.plan_builder.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return qualifier.plan_builder.build_qualification_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )


def test_qualification_contract_is_fresh_v2_and_nonreusable() -> None:
    contract = qualifier.QUALIFICATION_CONTRACT
    assert contract["probe_scene_indices_in_order"] == [12, 0]
    assert contract["worker_process_group_watchdog_seconds"] == 300.0
    assert contract["ld_lld_driver_entrypoint"] == str(
        qualifier.plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT
    )
    assert contract["ld_lld_driver_link_text"] == "lld"
    assert contract["direct_bound_lld_target_invocation_forbidden"] is True
    assert contract["v1_runtime_payload_reuse_authorized"] is False
    assert contract["probe_output_scientific_reuse_authorized"] is False


def test_v2_identity_overlay_patches_and_restores_full_qualifier() -> None:
    overrides = qualifier._configuration_overrides_v2()  # noqa: SLF001
    originals = {
        name: getattr(qualifier.predecessor, name) for name in overrides
    }
    with qualifier._configured_predecessor_qualifier_v2():  # noqa: SLF001
        assert qualifier.predecessor.plan_builder is qualifier.plan_builder
        assert qualifier.predecessor.collector is qualifier.collector
        assert qualifier.predecessor.QUALIFICATION_CONTRACT == (
            qualifier.QUALIFICATION_CONTRACT
        )
        assert qualifier.predecessor._run_rocm_egl_preflight is (  # noqa: SLF001
            qualifier._run_rocm_egl_preflight  # noqa: SLF001
        )
        assert qualifier.predecessor._worker_argv_qualification is (  # noqa: SLF001
            qualifier._worker_argv_qualification  # noqa: SLF001
        )
    assert all(
        getattr(qualifier.predecessor, name) is value
        for name, value in originals.items()
    )


def test_fresh_root_is_reserved_before_any_preflight() -> None:
    source = Path(qualifier.predecessor.__file__).read_text()
    execute = source[source.index("def execute_qualification(") :]
    execute = execute[: execute.index("\ndef build_parser", 1)]
    assert execute.index("_reserve_qualification(") < execute.index(
        "_run_rocm_egl_preflight("
    )
    assert qualifier.plan_builder.QUALIFICATION_ATTEMPT_ROOT != (
        qualifier.plan_builder.predecessor.QUALIFICATION_ATTEMPT_ROOT
    )


def _mock_run(plan: dict, observed: list[list[str]]):
    identity = {
        "arch_name": "gfx1201:sramecc-:xnack-",
        "device_name": "AMD Radeon AI PRO R9700",
        "genesis_backend_symbol": "gs.amdgpu",
        "genesis_file": plan["runtime_bindings"]["genesis_init_source"]["path"],
        "genesis_version": "0.4.6",
        "hsa_override_present": False,
        "numpy_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "numpy/__init__.py"
            ).resolve()
        ),
        "pillow_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "PIL/__init__.py"
            ).resolve()
        ),
        "torch_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "torch/__init__.py"
            ).resolve()
        ),
        "torch_hip_version": "7.2.0",
        "torch_version": "2.12.0+rocm7.2",
        "visible_device_count": 1,
    }

    def run(argv, **_kwargs):
        args = [str(value) for value in argv]
        observed.append(args)
        target = plan["runtime_bindings"]["rocm_lld_executable"]["path"]
        if args[0] == target:
            raise AssertionError("production preflight invoked generic target")
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                args, 0, "AMD LLD 20.0.0 exact test banner\n", ""
            )
        if Path(args[0]).name == "rocminfo":
            return subprocess.CompletedProcess(args, 0, "Name: gfx1201\n", "")
        if args[0] == plan["execution_contract"]["python_invocation_path"]:
            return subprocess.CompletedProcess(
                args, 0, "Genesis banner\n" + json.dumps(identity) + "\n", ""
            )
        egl = (
            "Device #0:\nOpenGL renderer: llvmpipe\n"
            "Device #1:\nOpenGL renderer: AMD Radeon AI PRO R9700\n"
        )
        return subprocess.CompletedProcess(args, 2, egl, "expected stderr")

    return run


def test_preflight_invokes_exact_driver_never_bound_target(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    child["QUALIFIER_TEST_SECRET"] = "must-not-persist"
    observed: list[list[str]] = []
    monkeypatch.setattr(
        qualifier.subprocess,
        "run",
        _mock_run(qualification_plan, observed),
    )
    result = qualifier._run_rocm_egl_preflight(  # noqa: SLF001
        qualification_plan, child_env=child
    )
    driver = str(qualifier.plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT)
    target = qualification_plan["runtime_bindings"]["rocm_lld_executable"][
        "path"
    ]
    assert observed[0] == [driver, "--version"]
    assert all(argv[0] != target for argv in observed)
    assert result["path_ld_lld"] == target
    assert result["rocm_path_ld_lld"] == target
    assert result["path_ld_lld_driver"] == driver
    assert result["rocm_path_ld_lld_driver"] == driver
    assert result["lld_driver_entrypoint"] == driver
    assert result["lld_driver_link_text"] == "lld"
    assert result["lld_resolved_target"] == target
    assert result["lld_invocation_argv"] == [driver, "--version"]
    assert result["lld_version_prefix_passed"] is True
    assert result["environment"] == (
        qualification_plan["execution_contract"]["environment"]
    )
    assert "QUALIFIER_TEST_SECRET" not in result["environment"]


def test_worker_argv_is_v2_qualifier_module() -> None:
    kwargs = {
        "scene_index": 12,
        "plan_path": Path("/tmp/plan"),
        "expected_plan_byte_count": 1,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority"),
        "expected_authority_byte_count": 2,
        "expected_authority_sha256": "b" * 64,
        "reservation_binding": {"byte_count": 3, "file_sha256": "c" * 64},
        "orchestrator_nonce": "d" * 64,
    }
    argv = qualifier._worker_argv_qualification(**kwargs)  # noqa: SLF001
    assert argv[0] == str(qualifier.plan_builder.ROCM_PYTHON.absolute())
    assert argv[1] == str(Path(qualifier.__file__).resolve())
    assert Path(argv[1]).resolve() == Path(qualifier.__file__).resolve()
    assert Path(argv[1]).resolve() != Path(
        qualifier.predecessor.__file__
    ).resolve()


def test_wrong_python_is_rejected_before_qualification_delegate_or_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "qualification_attempt"
    delegated: list[bool] = []
    monkeypatch.setattr(
        qualifier.plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt_root
    )
    monkeypatch.setattr(
        qualifier.collector.sys,
        "executable",
        "/tmp/wrong-qualification-venv/bin/python",
    )
    monkeypatch.setattr(
        qualifier.predecessor,
        "execute_qualification",
        lambda *_args, **_kwargs: delegated.append(True),
    )

    with pytest.raises(
        qualifier.collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        qualifier.execute_qualification()

    assert delegated == []
    assert not attempt_root.exists()


def test_wrong_python_cli_stops_before_qualification_parse_or_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "qualification_cli_attempt"
    parser_built: list[bool] = []
    monkeypatch.setattr(
        qualifier.plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt_root
    )
    monkeypatch.setattr(
        qualifier.collector.sys,
        "executable",
        "/tmp/wrong-qualification-cli-venv/bin/python",
    )
    monkeypatch.setattr(
        qualifier,
        "build_parser",
        lambda: parser_built.append(True),
    )

    with pytest.raises(
        qualifier.collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        qualifier.main([])

    assert parser_built == []
    assert not attempt_root.exists()


def test_failure_terminal_is_v2_and_permanently_nonretryable(
    tmp_path: Path,
    qualification_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "v2_attempt"
    attempt.mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(qualification_plan))
    plan_binding = qualifier.pilot.file_binding(plan_path)
    standard_plan = {
        "path": plan_binding["path"],
        "sha256": plan_binding["file_sha256"],
        "byte_count": plan_binding["byte_count"],
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}\n")
    authority_binding = qualifier.pilot.file_binding(authority_path)
    standard_authority = {
        "path": authority_binding["path"],
        "sha256": authority_binding["file_sha256"],
        "byte_count": authority_binding["byte_count"],
    }
    monkeypatch.setattr(
        qualifier.plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt
    )
    _set_exact_qualification_environment(monkeypatch)
    monkeypatch.setattr(
        qualifier,
        "validate_qualification_authority",
        lambda *_args, **_kwargs: (
            {},
            standard_authority,
            qualification_plan,
            standard_plan,
        ),
    )
    monkeypatch.setattr(
        qualifier,
        "execute_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            qualifier.GenesisRocmBackendV2QualificationError("forced failure")
        ),
    )
    status = qualifier.main(
        [
            "--plan",
            str(plan_path),
            "--expected-plan-byte-count",
            str(standard_plan["byte_count"]),
            "--expected-plan-sha256",
            str(standard_plan["sha256"]),
            "--authority",
            str(authority_path),
            "--expected-authority-byte-count",
            str(standard_authority["byte_count"]),
            "--expected-authority-sha256",
            str(standard_authority["sha256"]),
        ]
    )
    assert status == 1
    terminal = json.loads((attempt / "terminal.json").read_text())
    assert terminal["status"] == (
        "FAIL_GENESIS_ROCM_BACKEND_V2_QUALIFICATION_HARD_STOP"
    )
    assert terminal["authorizes_scientific_authority"] is False
    assert terminal["authorizes_retry_or_resume"] is False


def test_qualification_authority_rejects_mutated_v1_review_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = {
        "predecessor_v1_qualification_terminal_review_binding": {
            **qualifier.collector._standard_v1_review_binding(),  # noqa: SLF001
            "sha256": "0" * 64,
        }
    }
    monkeypatch.setattr(
        qualifier.predecessor,
        "validate_qualification_authority",
        lambda *_args, **_kwargs: (changed, {}, {}, {}),
    )
    with pytest.raises(
        qualifier.collector.SceneProcessCollectionError,
        match="terminal-review binding changed",
    ):
        qualifier.validate_qualification_authority(Path("/unused"))
