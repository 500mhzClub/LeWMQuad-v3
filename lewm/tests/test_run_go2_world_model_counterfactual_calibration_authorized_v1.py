from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "run_go2_world_model_counterfactual_calibration_authorized_v1.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_calibration_supervisor", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_smoke_authority_binds_calibration_supervisor() -> None:
    supervisor = _load_module()
    assert supervisor.collector.NON_SMOKE_SOURCE_PATHS["external_supervisor"] == (
        "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    )


def test_pre_reservation_validation_rejects_different_supervisor_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_module()
    plan_binding = {
        "path": "/fixture/plan.json",
        "file_sha256": "a" * 64,
        "byte_count": 1,
    }
    authority_binding = {
        "path": "/fixture/authority.json",
        "file_sha256": "b" * 64,
        "byte_count": 1,
    }
    review_binding = {
        "path": "/fixture/review.json",
        "file_sha256": "c" * 64,
        "byte_count": 1,
    }
    bound_supervisor = {
        "path": str(SCRIPT),
        "file_sha256": "d" * 64,
        "byte_count": 1,
    }
    raw_authority = {"plan_binding": plan_binding}
    normalized_authority = {
        "schema": supervisor.AUTHORITY_SCHEMA,
        "status": supervisor.AUTHORITY_STATUS,
        "review_binding": review_binding,
        "source_bindings": [
            {"name": "external_supervisor", "binding": bound_supervisor}
        ],
    }
    plan = {"purpose": supervisor.PURPOSE}
    reads = iter(
        (
            (raw_authority, authority_binding),
            (plan, plan_binding),
            ({"review": "fixture"}, review_binding),
        )
    )
    monkeypatch.setattr(
        supervisor.pilot, "read_bound_json", lambda *_args, **_kwargs: next(reads)
    )
    monkeypatch.setattr(
        supervisor.pilot, "_validate_binding_shape", lambda *_args, **_kwargs: plan_binding
    )
    monkeypatch.setattr(supervisor.pilot, "validate_plan", lambda _value: plan)
    monkeypatch.setattr(
        supervisor.collector,
        "_validate_authority_for_plan",
        lambda *_args, **_kwargs: normalized_authority,
    )
    monkeypatch.setattr(supervisor.pilot, "validate_source_review", lambda *_a, **_k: None)
    monkeypatch.setattr(
        supervisor.collector, "_validate_git_authority_boundary", lambda **_kwargs: None
    )
    monkeypatch.setattr(supervisor.pilot, "require_plan_bindings", lambda _plan: None)
    monkeypatch.setattr(
        supervisor.pilot,
        "file_binding",
        lambda _path: {**bound_supervisor, "file_sha256": "e" * 64},
    )
    with pytest.raises(
        supervisor.CalibrationSupervisionError, match="different calibration supervisor"
    ):
        supervisor.load_and_validate_authority(
            Path("/fixture/authority.json"),
            expected_authority_byte_count=1,
            expected_authority_sha256="b" * 64,
        )


def test_child_environment_removes_ambient_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "ambient")
    monkeypatch.setenv("LD_PRELOAD", "ambient")
    monkeypatch.setenv("UNRELATED_FIXTURE_VALUE", "preserved")
    plan = {
        "execution_contract": {
            "environment": dict(supervisor.pilot.EXECUTION_ENVIRONMENT)
        }
    }
    child = supervisor._child_environment(plan)
    assert "CUDA_VISIBLE_DEVICES" not in child
    assert "LD_PRELOAD" not in child
    assert child["UNRELATED_FIXTURE_VALUE"] == "preserved"
    assert {
        key: child[key] for key in supervisor.pilot.EXECUTION_ENVIRONMENT
    } == supervisor.pilot.EXECUTION_ENVIRONMENT


def test_graphics_preflight_is_bound_and_receipted(
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
    outputs = iter((
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
    ))
    monkeypatch.setattr(
        supervisor.subprocess, "run", lambda *_args, **_kwargs: next(outputs)
    )
    receipt = supervisor._run_graphics_preflight(
        plan,
        child_env={},
        wall_started=time.monotonic(),
        wall_ceiling=10.0,
    )
    assert receipt["status"] == "PASS"
    assert receipt["phase"] == "graphics_preflight"
    assert receipt["expectation"]["vulkan_device_name"] == (
        "AMD Radeon AI PRO R9700"
    )


def test_selected_device_global_vram_sampler_reports_baseline_peak_and_delta(
    tmp_path: Path,
) -> None:
    supervisor = _load_module()
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x7551\n")
    used = device / "mem_info_vram_used"
    total = device / "mem_info_vram_total"
    used.write_text("100\n")
    total.write_text("1000\n")
    plan = {
        "execution_contract": {
            "graphics_preflight": dict(
                supervisor.pilot.GRAPHICS_PREFLIGHT_EXPECTATION
            )
        }
    }
    used_path, total_path, vendor, device_id = supervisor._selected_gpu_memory_files(
        plan, drm_root=tmp_path
    )
    sampler = supervisor._GlobalVramSampler(
        used_path,
        total_path,
        vendor_id=vendor,
        device_id=device_id,
        interval_seconds=0.005,
    )
    sampler.start()
    used.write_text("260\n")
    time.sleep(0.02)
    measurement = sampler.stop()
    assert measurement["baseline_used_bytes"] == 100
    assert measurement["peak_used_bytes"] == 260
    assert measurement["peak_delta_above_baseline_bytes"] == 160
    assert measurement["device_total_bytes"] == 1000
    assert measurement["scope"] == (
        "selected_device_global_vram_not_process_attributed"
    )


def test_fresh_attempt_root_is_below_development_root(tmp_path: Path) -> None:
    supervisor = _load_module()
    fresh = supervisor.DEVELOPMENT_ROOT / "calibration_supervisor_fixture" / "v1"
    assert supervisor._require_fresh_attempt_root(str(fresh)) == fresh
    with pytest.raises(supervisor.CalibrationSupervisionError, match="escapes"):
        supervisor._require_fresh_attempt_root(str(tmp_path / "outside"))
    with pytest.raises(supervisor.CalibrationSupervisionError, match="cannot equal"):
        supervisor._require_fresh_attempt_root(str(supervisor.DEVELOPMENT_ROOT))


def test_run_once_accepts_terminal_stop_code_and_rejects_other_codes() -> None:
    supervisor = _load_module()
    receipt = supervisor._run_once(
        [sys.executable, "-c", "raise SystemExit(2)"],
        timeout=5.0,
        env=os.environ,
        accepted_returncodes=frozenset({0, 2}),
    )
    assert receipt["exit_code"] == 2
    with pytest.raises(supervisor.CalibrationSupervisionError, match="status 3"):
        supervisor._run_once(
            [sys.executable, "-c", "raise SystemExit(3)"],
            timeout=5.0,
            env=os.environ,
            accepted_returncodes=frozenset({0, 2}),
        )


def test_owned_reservation_is_exact_and_nonce_bound(tmp_path: Path) -> None:
    supervisor = _load_module()
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    nonce = "a" * 64
    attempt = {"id": "fixture"}
    plan_binding = {"path": "/fixture/plan", "file_sha256": "b" * 64, "byte_count": 1}
    authority_binding = {
        "path": "/fixture/authority",
        "file_sha256": "c" * 64,
        "byte_count": 1,
    }
    reservation = {
        "schema": "lewm_go2_world_model_counterfactual_attempt_reservation_v1",
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": attempt,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "supervisor_nonce": nonce,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    (attempt_root / "reservation.json").write_text(
        json.dumps(reservation), encoding="utf-8"
    )
    assert supervisor._owned_reservation(
        attempt_root,
        supervisor_nonce=nonce,
        authority={"attempt": attempt},
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
    assert not supervisor._owned_reservation(
        attempt_root,
        supervisor_nonce="d" * 64,
        authority={"attempt": attempt},
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )


def test_failed_physics_result_is_bound_for_terminal_supervision(
    tmp_path: Path,
) -> None:
    supervisor = _load_module()
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    plan_binding = {
        "path": "/fixture/plan.json",
        "file_sha256": "a" * 64,
        "byte_count": 1,
    }
    authority_binding = {
        "path": "/fixture/authority.json",
        "file_sha256": "b" * 64,
        "byte_count": 1,
    }
    review_binding = {
        "path": "/fixture/review.json",
        "file_sha256": "c" * 64,
        "byte_count": 1,
    }
    source_bindings = [{"name": "fixture", "binding": plan_binding}]
    caps = {"wall_seconds": 10.0}
    plan = {
        "attempt_id": "calibration-v2-fixture",
        "purpose": supervisor.PURPOSE,
    }
    authority = {
        "review_binding": review_binding,
        "source_bindings": source_bindings,
        "caps": caps,
    }
    result = {
        "schema": supervisor.pilot.PHYSICS_RESULT_SCHEMA,
        "status": "FAILED",
        "attempt_id": plan["attempt_id"],
        "purpose": supervisor.PURPOSE,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "review_binding": review_binding,
        "source_bindings": source_bindings,
        "caps": caps,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "failure": {
            "type": "PilotContractError",
            "message": "fixture terminal failure",
        },
    }
    result_path = attempt_root / "physics_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    loaded, binding = supervisor._load_physics_result_if_present(
        attempt_root,
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
    )
    assert loaded is not None and loaded["status"] == "FAILED"
    assert binding == supervisor.pilot.file_binding(result_path)
    result["citable_as_scientific_evidence"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(
        supervisor.CalibrationSupervisionError, match="terminal receipt"
    ):
        supervisor._load_physics_result_if_present(
            attempt_root,
            plan=plan,
            plan_binding=plan_binding,
            authority=authority,
            authority_binding=authority_binding,
        )


def test_help_does_not_start_a_calibration() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected-authority-sha256" in completed.stdout
