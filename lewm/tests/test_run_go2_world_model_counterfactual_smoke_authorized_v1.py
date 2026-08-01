from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts" / "run_go2_world_model_counterfactual_smoke_authorized_v1.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("counterfactual_smoke_supervisor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    supervisor = _load_module()
    with pytest.raises(supervisor.SmokeSupervisionError, match="duplicate JSON key"):
        supervisor.strict_json_bytes(b'{"x": 1, "x": 2}', label="fixture")
    with pytest.raises(supervisor.SmokeSupervisionError, match="non-finite"):
        supervisor.strict_json_bytes(b'{"x": NaN}', label="fixture")


def test_file_binding_rejects_symlinks(tmp_path: Path) -> None:
    supervisor = _load_module()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(supervisor.SmokeSupervisionError, match="symlink"):
        supervisor.file_binding(link)


def test_file_binding_rejects_protected_paths_before_open(tmp_path: Path) -> None:
    supervisor = _load_module()
    protected = tmp_path / "sealed_test.json"
    with pytest.raises(supervisor.SmokeSupervisionError, match="protected"):
        supervisor.file_binding(protected)


def test_fresh_attempt_root_must_be_absent_and_inside_development_root(
    tmp_path: Path,
) -> None:
    supervisor = _load_module()
    fresh = supervisor.DEVELOPMENT_ROOT / "supervisor_unit_fixture" / "attempt_v1"
    assert supervisor._require_fresh_development_root(str(fresh)) == fresh
    with pytest.raises(supervisor.SmokeSupervisionError, match="escapes"):
        supervisor._require_fresh_development_root(str(tmp_path / "attempt_v1"))
    existing = supervisor.DEVELOPMENT_ROOT
    with pytest.raises(supervisor.SmokeSupervisionError, match="cannot equal"):
        supervisor._require_fresh_development_root(str(existing))


def test_run_once_records_success_and_rejects_nonzero_exit() -> None:
    supervisor = _load_module()
    receipt = supervisor._run_once(
        [sys.executable, "-c", "raise SystemExit(0)"], timeout=5.0
    )
    assert receipt["exit_code"] == 0
    with pytest.raises(supervisor.SmokeSupervisionError, match="status 7"):
        supervisor._run_once(
            [sys.executable, "-c", "raise SystemExit(7)"], timeout=5.0
        )


def test_run_once_enforces_hard_timeout() -> None:
    supervisor = _load_module()
    with pytest.raises(supervisor.SmokeSupervisionError, match="wall ceiling"):
        supervisor._run_once(
            [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05
        )


def test_help_is_source_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected-authority-sha256" in completed.stdout


def test_terminal_json_is_strict_and_non_authorizing(tmp_path: Path) -> None:
    supervisor = _load_module()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    payload = {
        "schema": supervisor.TERMINAL_SCHEMA,
        "status": "CONSUMED_TERMINAL_FAILURE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
    }
    binding = supervisor._write_terminal(attempt, payload)
    assert binding is not None
    assert json.loads((attempt / "terminal_supervision.json").read_text()) == payload
    with pytest.raises(FileExistsError):
        supervisor._write_terminal(attempt, payload)


def test_owned_reservation_requires_exact_keys_and_nonce(tmp_path: Path) -> None:
    supervisor = _load_module()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    nonce = "a" * 64
    reservation = {
        "schema": "lewm_go2_world_model_counterfactual_smoke_reservation_v1",
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": {"id": "fixture"},
        "plan_binding": {},
        "authority_binding": {},
        "supervisor_nonce": nonce,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    (attempt / "reservation.json").write_text(
        json.dumps(reservation) + "\n", encoding="utf-8"
    )
    assert supervisor._owned_reservation(attempt, supervisor_nonce=nonce) is not None
    assert supervisor._owned_reservation(
        attempt, supervisor_nonce="b" * 64
    ) is None
    reservation["unexpected"] = True
    (attempt / "reservation.json").write_text(
        json.dumps(reservation) + "\n", encoding="utf-8"
    )
    assert supervisor._owned_reservation(attempt, supervisor_nonce=nonce) is None


def test_optional_physics_result_loader_binds_failed_receipt(tmp_path: Path) -> None:
    supervisor = _load_module()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    payload = {"schema": supervisor.PHYSICS_RESULT_SCHEMA, "status": "FAILED"}
    path = attempt / "physics_result.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    loaded, binding = supervisor._load_physics_result_if_present(attempt)
    assert loaded == payload
    assert binding == supervisor.file_binding(path)

    path.unlink()
    assert supervisor._load_physics_result_if_present(attempt) == (None, None)


def test_failed_collector_receipt_does_not_mask_an_owned_reservation(
    tmp_path: Path,
) -> None:
    supervisor = _load_module()
    reservation_path = tmp_path / "reservation.json"
    reservation_path.write_text("{}\n", encoding="utf-8")
    binding = supervisor.file_binding(reservation_path)
    attempt = {"id": "fixture"}
    plan_binding = {"path": "/plan", "file_sha256": "1" * 64, "byte_count": 1}
    authority_binding = {
        "path": "/authority",
        "file_sha256": "2" * 64,
        "byte_count": 2,
    }
    document = {
        "schema": "lewm_go2_world_model_counterfactual_smoke_reservation_v1",
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": attempt,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    owned = {"document": document, "binding": binding}
    authority = {"attempt": attempt}
    assert supervisor._owned_reservation_contract_changed(
        owned,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        physics_result=None,
    ) is False

    relative_binding = dict(binding)
    relative_binding["path"] = "reservation.json"
    assert supervisor._owned_reservation_contract_changed(
        owned,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        physics_result={"reservation_binding": relative_binding},
    ) is False


def test_supervise_preserves_child_failure_and_binds_failed_physics_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_module()
    attempt_root = tmp_path / "attempt"
    plan_binding = {
        "path": str(tmp_path / "plan.json"),
        "file_sha256": "1" * 64,
        "byte_count": 1,
    }
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "file_sha256": "2" * 64,
        "byte_count": 2,
    }
    attempt_contract = {
        "id": "fixture-attempt",
        "root": str(attempt_root),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    authority = {
        "plan_binding": plan_binding,
        "attempt": attempt_contract,
        "caps": {"wall_seconds": 30.0},
        "source_commit": "3" * 40,
        "external_supervisor": {"terminal_reviewer": "/root/reviewer"},
    }
    plan = {
        "attempt_id": "fixture-attempt",
        "output_root": str(attempt_root),
    }
    monkeypatch.setattr(
        supervisor,
        "load_and_validate_authority",
        lambda *_args, **_kwargs: (
            authority,
            authority_binding,
            plan,
            plan_binding,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_python_invocation_before_launch",
        lambda _plan: Path(sys.executable),
    )
    monkeypatch.setattr(supervisor, "_child_environment", lambda _plan: {})
    monkeypatch.setattr(
        supervisor,
        "_run_graphics_preflight",
        lambda _plan, *, child_env: {
            "phase": "graphics_preflight",
            "status": "PASS",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_require_fresh_development_root",
        lambda _path: attempt_root,
    )
    monkeypatch.setattr(supervisor, "_git_head", lambda: "4" * 40)
    launched: list[list[str]] = []

    def fail_after_receipts(argv, *, timeout, env):
        del timeout, env
        launched.append(list(argv))
        nonce = argv[argv.index("--supervisor-nonce") + 1]
        attempt_root.mkdir()
        reservation = {
            "schema": "lewm_go2_world_model_counterfactual_smoke_reservation_v1",
            "status": "RESERVED_ATTEMPT_CONSUMED",
            "attempt": attempt_contract,
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "supervisor_nonce": nonce,
            "retry_authorized": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
            "refill_authorized": False,
        }
        reservation_path = attempt_root / "reservation.json"
        reservation_path.write_text(
            json.dumps(reservation) + "\n", encoding="utf-8"
        )
        reservation_binding = supervisor.file_binding(reservation_path)
        reservation_binding["path"] = "reservation.json"
        (attempt_root / "physics_result.json").write_text(
            json.dumps(
                {
                    "schema": supervisor.PHYSICS_RESULT_SCHEMA,
                    "status": "FAILED",
                    "reservation_binding": reservation_binding,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise supervisor.SmokeSupervisionError(
            "supervised command exited with status 2"
        )

    monkeypatch.setattr(supervisor, "_run_once", fail_after_receipts)
    terminal, terminal_binding = supervisor.supervise(
        tmp_path / "authority.json",
        expected_authority_byte_count=2,
        expected_authority_sha256="2" * 64,
    )
    assert len(launched) == 1
    assert terminal_binding is not None
    assert terminal["status"] == "CONSUMED_TERMINAL_FAILURE"
    assert terminal["physics_result_binding"] == supervisor.file_binding(
        attempt_root / "physics_result.json"
    )
    assert "supervised command exited with status 2" in terminal["failure"]
    assert "owned reservation contract changed" not in terminal["failure"]
    assert (attempt_root / "terminal_supervision.json").is_file()


def test_unbound_python_invocation_is_rejected_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_module()
    invocation = tmp_path / "venv" / "bin" / "python"
    target = tmp_path / "other-python"
    environment = tmp_path / "venv" / "pyvenv.cfg"
    invocation.parent.mkdir(parents=True)
    invocation.write_bytes(b"invocation\n")
    target.write_bytes(b"target\n")
    environment.write_bytes(b"venv\n")
    plan = {
        "output_root": str(
            supervisor.DEVELOPMENT_ROOT / "unbound_invocation_fixture" / "attempt_v1"
        ),
        "execution_contract": {
            "python_invocation_path": str(invocation.resolve()),
            "environment": dict(supervisor.pilot.EXECUTION_ENVIRONMENT),
        },
        "runtime_bindings": {
            "python_executable_target": supervisor.pilot.file_binding(target),
            "python_environment_config": supervisor.pilot.file_binding(environment),
        },
    }
    authority = {"caps": {"wall_seconds": 5.0}}
    monkeypatch.setattr(
        supervisor,
        "load_and_validate_authority",
        lambda *_args, **_kwargs: (authority, {}, plan, {}),
    )
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Popen path reached")

    monkeypatch.setattr(supervisor, "_run_once", forbidden_run)
    with pytest.raises(
        supervisor.SmokeSupervisionError,
        match="does not resolve inside the bound environment",
    ):
        supervisor.supervise(
            tmp_path / "authority.json",
            expected_authority_byte_count=1,
            expected_authority_sha256="0" * 64,
        )
    assert called is False


def test_child_environment_removes_ambient_device_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_module()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "wrong")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/wrong/icd.json")
    monkeypatch.setenv("GS_PARA_LEVEL", "2")
    plan = {
        "execution_contract": {
            "environment": dict(supervisor.pilot.EXECUTION_ENVIRONMENT)
        }
    }
    child = supervisor._child_environment(plan)
    assert child["EGL_DEVICE_ID"] == "1"
    assert child["MESA_VK_DEVICE_SELECT"] == "1002:7551!"
    assert child["GS_PARA_LEVEL"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in child
    assert "VK_ICD_FILENAMES" not in child


def test_graphics_preflight_requires_exact_selected_r9700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_module()
    eglinfo = tmp_path / "eglinfo"
    vulkaninfo = tmp_path / "vulkaninfo"
    eglinfo.write_bytes(b"egl\n")
    vulkaninfo.write_bytes(b"vulkan\n")
    plan = {
        "runtime_bindings": {
            "eglinfo_executable": supervisor.pilot.file_binding(eglinfo),
            "vulkaninfo_executable": supervisor.pilot.file_binding(vulkaninfo),
        },
        "execution_contract": {
            "environment": dict(supervisor.pilot.EXECUTION_ENVIRONMENT),
            "graphics_preflight": dict(
                supervisor.pilot.GRAPHICS_PREFLIGHT_EXPECTATION
            ),
        },
    }
    outputs = iter(
        (
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "GPU0:\n\tvendorID = 0x1002\n\tdeviceID = 0x7551\n"
                    "\tdeviceName = AMD Radeon AI PRO R9700\n"
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=2,
                stdout=(
                    "Device #0:\nrenderer: integrated\n\nDevice #1:\n"
                    "OpenGL core profile renderer: AMD Radeon AI PRO R9700\n"
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_a, **_k: next(outputs))
    receipt = supervisor._run_graphics_preflight(plan, child_env={})
    assert receipt["status"] == "PASS"
    assert receipt["expectation"]["vulkan_device_name"] == "AMD Radeon AI PRO R9700"
