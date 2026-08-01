from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
)


def _load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "existing_pool_three_arm_supervisor", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inert(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(path.encode()).hexdigest(),
        "byte_count": 1,
    }


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    supervisor = _load_supervisor()
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="duplicate JSON key"):
        supervisor.strict_json_bytes(b'{"x": 1, "x": 2}', label="fixture")
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="non-finite"):
        supervisor.strict_json_bytes(b'{"x": Infinity}', label="fixture")


def test_file_binding_rejects_symlink_and_protected_path(tmp_path: Path) -> None:
    supervisor = _load_supervisor()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="symlink"):
        supervisor.file_binding(symlink)
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="protected"):
        supervisor.file_binding(tmp_path / "sealed_test.json")


def test_attempt_contract_is_exact_max_one_and_non_retriable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_supervisor()
    attempt_root = tmp_path / "campaign" / "attempt_v1"
    monkeypatch.setattr(supervisor, "ATTEMPT_ROOT", attempt_root)
    attempt = {
        "id": "attempt-v1",
        "root": str(attempt_root.resolve()),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    assert supervisor._validate_attempt(
        attempt, output_root=str(attempt_root.resolve())
    ) == attempt
    changed = dict(attempt)
    changed["resume"] = True
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
        supervisor._validate_attempt(
            changed, output_root=str(attempt_root.resolve())
        )


def test_caps_cannot_exceed_preregistered_wall_or_gpu_ceiling() -> None:
    supervisor = _load_supervisor()
    valid = {
        "maximum_wall_seconds": 43_200.0,
        "maximum_gpu_seconds": 36_000.0,
        "maximum_training_updates": 700,
    }
    assert supervisor._validate_caps(valid) == valid
    for key, value in (
        ("maximum_wall_seconds", 43_200.1),
        ("maximum_gpu_seconds", 36_000.1),
        ("maximum_training_updates", 701),
    ):
        changed = dict(valid)
        changed[key] = value
        with pytest.raises(supervisor.ThreeArmSupervisionError, match="caps"):
            supervisor._validate_caps(changed)


def test_required_source_closure_names_exact_runtime_dependencies() -> None:
    supervisor = _load_supervisor()
    assert len(supervisor.WORKER_OUTPUT_PATHS) == 57
    assert "pack/train_frames.u8" in supervisor.WORKER_OUTPUT_PATHS
    assert len(supervisor.REQUIRED_SOURCE_PATHS) == 32
    assert supervisor.REQUIRED_SOURCE_PATHS["lewm_package"] == "lewm/__init__.py"
    assert supervisor.REQUIRED_SOURCE_PATHS["benchmarks_package"] == (
        "lewm/benchmarks/__init__.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["models_package"] == (
        "lewm/models/__init__.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["base_world_model"] == (
        "lewm/models/lewm.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["worker"].endswith(
        "execute_go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["experiment_metrics"].endswith(
        "go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["place_data"] == (
        "lewm/datasets/go2_memory_role_place_triplets_v1.py"
    )
    assert supervisor.REQUIRED_SOURCE_PATHS["scaled_runtime"] == (
        "scripts/dev_train_temporal_jepa_scaled.py"
    )


def test_reservation_precedes_worker_and_is_exclusive(tmp_path: Path) -> None:
    supervisor = _load_supervisor()
    attempt_root = tmp_path / "campaign" / "attempt_v1"
    attempt = {
        "id": "attempt-v1",
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
        "output_root": str(attempt_root),
        "execution": {"worker_path": "/worker.py", "checker_path": "/checker.py"},
        "review_binding": _inert("/review.json"),
        "source_commit": "a" * 40,
        "source_bindings": [],
        "runtime": {},
        "input_bindings": {},
        "attempt": attempt,
        "caps": {
            "maximum_wall_seconds": 10.0,
            "maximum_gpu_seconds": 8.0,
            "maximum_training_updates": 700,
        },
    }
    authority_binding = _inert("/authority.json")
    plan_binding = _inert("/plan.json")
    worker_binding = _inert("/worker.py")
    reservation, binding = supervisor._reserve_attempt(
        attempt_root,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        worker_binding=worker_binding,
        checker_binding=_inert("/checker.py"),
        worker_command=["python", "worker.py"],
        checker_command_template=["python", "checker.py", "<RESULT>"],
        supervisor_nonce="b" * 64,
    )
    assert reservation["status"] == "RESERVED_ATTEMPT_CONSUMED"
    assert reservation["maximum_attempts"] == 1
    assert reservation["retry_authorized"] is False
    assert binding == supervisor.file_binding(attempt_root / "reservation.json")
    with pytest.raises(FileExistsError):
        supervisor._reserve_attempt(
            attempt_root,
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            worker_binding=worker_binding,
            checker_binding=_inert("/checker.py"),
            worker_command=["python", "worker.py"],
            checker_command_template=["python", "checker.py", "<RESULT>"],
            supervisor_nonce="c" * 64,
        )


def test_fresh_attempt_requires_conservative_free_space_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_supervisor()
    development = tmp_path / "dev"
    development.mkdir()
    attempt_root = development / "campaign" / "attempt_v1"
    monkeypatch.setattr(supervisor, "DEVELOPMENT_ROOT", development)
    monkeypatch.setattr(supervisor, "ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(
        supervisor.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=100 * 1024**3,
            used=90 * 1024**3,
            free=10 * 1024**3,
        ),
    )
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="16 GiB"):
        supervisor._require_fresh_attempt_root(str(attempt_root.resolve()))


def test_run_once_enforces_nonzero_and_timeout() -> None:
    supervisor = _load_supervisor()
    receipt = supervisor._run_once(
        [sys.executable, "-c", "raise SystemExit(0)"],
        timeout=5.0,
        env={},
    )
    assert receipt["exit_code"] == 0
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="status 9"):
        supervisor._run_once(
            [sys.executable, "-c", "raise SystemExit(9)"],
            timeout=5.0,
            env={},
        )
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="wall ceiling"):
        supervisor._run_once(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.05,
            env={},
        )


def test_child_environment_removes_ambient_python_and_device_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor()
    monkeypatch.setenv("PYTHONPATH", "/wrong")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "wrong")
    monkeypatch.setenv("UNBOUND_ARBITRARY_VALUE", "wrong")
    runtime = {
        "environment": dict(supervisor.EXACT_CHILD_ENVIRONMENT)
    }
    child = supervisor._child_environment(runtime)
    assert child == supervisor.EXACT_CHILD_ENVIRONMENT
    assert child["HIP_VISIBLE_DEVICES"] == "0"
    assert child["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in child
    assert "UNBOUND_ARBITRARY_VALUE" not in child


def test_git_identity_checks_ignore_ambient_git_control_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor()
    expected = supervisor._git_head()
    monkeypatch.setenv("GIT_DIR", "/definitely/wrong")
    monkeypatch.setenv("GIT_WORK_TREE", "/definitely/wrong")
    monkeypatch.setenv("GIT_INDEX_FILE", "/definitely/wrong")
    assert supervisor._git_head() == expected


def test_worker_result_requires_exact_consumed_reservation_link() -> None:
    supervisor = _load_supervisor()
    authority_binding = _inert("/authority.json")
    plan_binding = _inert("/plan.json")
    review_binding = _inert("/review.json")
    reservation_binding = _inert("/reservation.json")
    attempt = {
        "id": "attempt-v1",
        "root": "/attempt",
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    authority = {
        "review_binding": review_binding,
        "source_commit": "a" * 40,
        "attempt": attempt,
        "caps": {
            "maximum_wall_seconds": 10.0,
            "maximum_gpu_seconds": 8.0,
            "maximum_training_updates": 700,
        },
        "runtime": {"identity": "runtime"},
        "input_bindings": {"input": _inert("/input.json")},
    }
    nonce = "f" * 64
    result = {
        "schema": supervisor.RESULT_SCHEMA,
        "status": supervisor.RESULT_STATUS,
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
        "review_binding": review_binding,
        "source_commit": "a" * 40,
        "attempt": supervisor._expected_result_attempt(
            attempt,
            reservation_binding=reservation_binding,
            supervisor_nonce=nonce,
        ),
        "caps": authority["caps"],
        "runtime": {
            "authorized": authority["runtime"],
            "observed": {
                "device_name": "AMD Radeon AI PRO R9700",
                "device_arch": "gfx1201",
                "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
                "torch_hip": "7.2.53211-e1a6bc5663",
                "numpy_version": "1.26.4",
                "pillow_version": "11.3.0",
                "gpu_phase_elapsed_seconds": 1.0,
                "wall_elapsed_seconds": 2.0,
                "output_inventory": sorted(supervisor.WORKER_OUTPUT_PATHS),
            },
        },
        "input_bindings": authority["input_bindings"],
    }
    supervisor._validate_worker_result(
        result,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        reservation_binding=reservation_binding,
        supervisor_nonce=nonce,
    )
    result["attempt"]["reservation"]["retry"] = True
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="exact linked"):
        supervisor._validate_worker_result(
            result,
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            supervisor_nonce=nonce,
        )


def test_supervise_launches_exact_worker_then_checker_after_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_supervisor()
    attempt_root = tmp_path / "campaign" / "attempt_v1"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}\n", encoding="utf-8")
    authority_binding = supervisor.file_binding(authority_path)
    plan_binding = _inert("/plan.json")
    review_binding = _inert("/review.json")
    worker_binding = _inert("/worker.py")
    attempt = {
        "id": "attempt-v1",
        "root": str(attempt_root.resolve()),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    authority = {
        "output_root": str(attempt_root.resolve()),
        "plan_binding": plan_binding,
        "review_binding": review_binding,
        "source_commit": "a" * 40,
        "source_bindings": [{"name": "worker", "binding": worker_binding}],
        "runtime": {
            "python_invocation_path": sys.executable,
            "environment": dict(supervisor.EXACT_CHILD_ENVIRONMENT),
            "bindings": {},
        },
        "input_bindings": {"input": _inert("/input.json")},
        "attempt": attempt,
        "caps": {
            "maximum_wall_seconds": 30.0,
            "maximum_gpu_seconds": 20.0,
            "maximum_training_updates": 700,
        },
        "external_supervisor": {"terminal_reviewer": "/root/reviewer"},
        "execution": {
            "worker_path": "/synthetic/worker.py",
            "checker_path": "/synthetic/checker.py",
        },
    }
    monkeypatch.setattr(
        supervisor,
        "load_and_validate_authority",
        lambda *_args, **_kwargs: (
            authority,
            authority_binding,
            {},
            plan_binding,
            {"worker": worker_binding, "checker": _inert("/checker.py")},
        ),
    )
    monkeypatch.setattr(
        supervisor, "_require_fresh_attempt_root", lambda _path: attempt_root
    )
    monkeypatch.setattr(supervisor, "_reverify_contract", lambda _authority: None)
    monkeypatch.setattr(supervisor, "_git_head", lambda: "b" * 40)
    launched: list[list[str]] = []

    def fake_run(argv, *, timeout, env):
        del timeout, env
        launched.append(list(argv))
        reservation_binding = supervisor.file_binding(
            attempt_root / "reservation.json"
        )
        reservation = json.loads(
            (attempt_root / "reservation.json").read_text(encoding="utf-8")
        )
        if len(launched) == 1:
            result = {
                "schema": supervisor.RESULT_SCHEMA,
                "status": supervisor.RESULT_STATUS,
                "authority_binding": authority_binding,
                "plan_binding": plan_binding,
                "review_binding": review_binding,
                "source_commit": authority["source_commit"],
                "attempt": supervisor._expected_result_attempt(
                    attempt,
                    reservation_binding=reservation_binding,
                    supervisor_nonce=reservation["supervisor_nonce"],
                ),
                "caps": authority["caps"],
                "runtime": {
                    "authorized": authority["runtime"],
                    "observed": {
                        "device_name": "AMD Radeon AI PRO R9700",
                        "device_arch": "gfx1201",
                        "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
                        "torch_hip": "7.2.53211-e1a6bc5663",
                        "numpy_version": "1.26.4",
                        "pillow_version": "11.3.0",
                        "gpu_phase_elapsed_seconds": 0.01,
                        "wall_elapsed_seconds": 0.02,
                        "output_inventory": sorted(supervisor.WORKER_OUTPUT_PATHS),
                    },
                },
                "input_bindings": authority["input_bindings"],
            }
            (attempt_root / "result.json").write_text(
                json.dumps(result) + "\n", encoding="utf-8"
            )
        else:
            result_binding = supervisor.file_binding(attempt_root / "result.json")
            check = {
                "schema": supervisor.CHECK_SCHEMA,
                "status": "PASS",
                "manifest_binding": result_binding,
                "pack_payloads_opened": False,
                "input_data_opened": False,
                "runtime_payloads_opened": False,
                "rgb_bytes_opened": False,
                "checkpoints_opened": False,
                "sealed_material_opened": False,
            }
            (attempt_root / "receipt_check.json").write_text(
                json.dumps(check) + "\n", encoding="utf-8"
            )
        return {"argv": list(argv), "elapsed_seconds": 0.01, "exit_code": 0}

    monkeypatch.setattr(supervisor, "_run_once", fake_run)
    terminal, terminal_binding = supervisor.supervise(
        authority_path,
        expected_authority_byte_count=authority_binding["byte_count"],
        expected_authority_sha256=authority_binding["file_sha256"],
    )
    assert terminal_binding is not None
    assert terminal["status"] == supervisor.RESULT_STATUS
    assert len(launched) == 2
    assert Path(launched[0][1]).name == (
        "execute_go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert "--expected-authority-sha256" in launched[0]
    assert Path(launched[1][1]).name == (
        "check_go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert (attempt_root / "reservation.json").is_file()
    assert (attempt_root / "terminal_supervision.json").is_file()


def test_help_is_source_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected-authority-sha256" in completed.stdout
