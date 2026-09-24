from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = (
    ROOT
    / "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)


def _load(name: str = "_geometry_anchored_joint_jepa_launcher_test"):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_and_runtime_command_are_source_only_and_exact() -> None:
    heavy_before = {
        name for name in sys.modules
        if name == "torch" or name.startswith("torch.")
        or name == "numpy" or name.startswith("numpy.")
    }
    launcher = _load()
    heavy_after = {
        name for name in sys.modules
        if name == "torch" or name.startswith("torch.")
        or name == "numpy" or name.startswith("numpy.")
    }
    assert heavy_after == heavy_before
    args = launcher.parse_args([
        "--review-sha256", "0" * 64,
        "--authorization-sha256", "1" * 64,
    ])
    assert launcher._runtime_argv(args) == [
        "/home/andrewknowles/.local/share/"
        "lewmquad-v12-runtime-torch291-rocm64/bin/python",
        "-I",
        "-B",
        str(ROOT / launcher.contract.RUNNER_RELATIVE_PATH),
        "--review-sha256",
        "0" * 64,
        "--authorization-sha256",
        "1" * 64,
    ]


def test_launch_environment_isolates_python_accelerator_and_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load("_geometry_anchored_joint_jepa_environment_test")
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        *launcher.CONFLICTING_ACCELERATOR_ENVIRONMENT,
    ):
        monkeypatch.setenv(key, "forbidden")
    environment = launcher._launch_environment()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        *launcher.CONFLICTING_ACCELERATOR_ENVIRONMENT,
    ):
        assert key not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert all(environment[key] == "1" for key in launcher.THREAD_ENVIRONMENT)


def test_bad_hash_is_rejected_before_authority_or_exec() -> None:
    launcher = _load("_geometry_anchored_joint_jepa_hash_test")
    with pytest.raises(SystemExit):
        launcher.parse_args([
            "--review-sha256", "A" * 64,
            "--authorization-sha256", "1" * 64,
        ])


def test_main_checks_authority_and_absent_root_before_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load("_geometry_anchored_joint_jepa_main_test")
    events: list[str] = []

    def authority(**_: str) -> dict[str, object]:
        events.append("authority")
        return {}

    class ExecReached(RuntimeError):
        pass

    def execute(_: argparse.Namespace) -> None:
        events.append("exec")
        raise ExecReached

    monkeypatch.setattr(launcher, "_validate_authority", authority)
    monkeypatch.setattr(launcher, "_exec_runtime", execute)
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", tmp_path / "absent")
    with pytest.raises(ExecReached):
        launcher.main([
            "--review-sha256", "0" * 64,
            "--authorization-sha256", "1" * 64,
        ])
    assert events == ["authority", "exec"]

    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", existing)
    with pytest.raises(FileExistsError):
        launcher.main([
            "--review-sha256", "0" * 64,
            "--authorization-sha256", "1" * 64,
        ])
    assert events == ["authority", "exec", "authority"]


def test_execve_receives_exact_interpreter_argv_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load("_geometry_anchored_joint_jepa_execve_test")
    args = argparse.Namespace(
        review_sha256="0" * 64,
        authorization_sha256="1" * 64,
    )
    observed: dict[str, object] = {}

    class ExecveReached(RuntimeError):
        pass

    def fake_execve(path: str, argv: list[str], environment: dict[str, str]) -> None:
        observed.update(path=path, argv=argv, environment=environment)
        raise ExecveReached

    monkeypatch.setattr(os, "execve", fake_execve)
    with pytest.raises(ExecveReached):
        launcher._exec_runtime(args)
    assert observed["path"] == launcher.contract.RUNTIME_INTERPRETER_PATH
    assert observed["argv"] == launcher._runtime_argv(args)
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert all(environment[key] == "1" for key in launcher.THREAD_ENVIRONMENT)
