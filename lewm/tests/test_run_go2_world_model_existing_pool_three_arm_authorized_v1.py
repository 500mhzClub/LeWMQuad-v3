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
REPLACEMENT_SCHEMA_PREFIX = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v2_"
)
REPLACEMENT_ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v2/attempt_v1"
)
REPLACEMENT_ATTEMPT_ROOT = (
    ROOT
    / ".generated/dev"
    / "world_model_existing_pool_three_arm_v1_integrity_replacement_v2"
    / "attempt_v1"
)
CONSUMED_ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v1/attempt_v1"
)
CONSUMED_ATTEMPT_ROOT = (
    ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v1/attempt_v1"
)
ORIGINAL_CONSUMED_ATTEMPT_ID = "world_model_existing_pool_three_arm_v1/attempt_v1"
ORIGINAL_CONSUMED_ATTEMPT_ROOT = (
    ROOT / ".generated/dev/world_model_existing_pool_three_arm_v1/attempt_v1"
)
PREDECESSOR_FAILURE_AUDIT = (
    ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v1_terminal_pretraining_source_failure_result_2026-08-01.json"
)
REPLACEMENT_PLAN = (
    ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v2_plan_2026-08-01.json"
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
    assert supervisor.ATTEMPT_ID == REPLACEMENT_ATTEMPT_ID
    assert supervisor.ATTEMPT_ROOT == REPLACEMENT_ATTEMPT_ROOT
    assert supervisor.AUTHORITY_SCHEMA == (
        REPLACEMENT_SCHEMA_PREFIX + "execution_authority_v1"
    )
    assert supervisor.PLAN_SCHEMA == REPLACEMENT_SCHEMA_PREFIX + "plan_v1"
    assert supervisor.RESERVATION_SCHEMA == (
        REPLACEMENT_SCHEMA_PREFIX + "reservation_v1"
    )
    assert supervisor.RESULT_SCHEMA == REPLACEMENT_SCHEMA_PREFIX + "result_v1"
    assert supervisor.CHECK_SCHEMA == (
        REPLACEMENT_SCHEMA_PREFIX + "receipt_check_v1"
    )
    assert supervisor.TERMINAL_SCHEMA == (
        REPLACEMENT_SCHEMA_PREFIX + "supervision_terminal_v1"
    )
    attempt_root = tmp_path / "campaign" / "attempt_v1"
    monkeypatch.setattr(supervisor, "ATTEMPT_ROOT", attempt_root)
    attempt = {
        "id": supervisor.ATTEMPT_ID,
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
    changed = dict(attempt)
    changed["id"] = CONSUMED_ATTEMPT_ID
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
        supervisor._validate_attempt(
            changed, output_root=str(attempt_root.resolve())
        )
    changed = dict(attempt)
    changed["root"] = str(CONSUMED_ATTEMPT_ROOT.resolve())
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
        supervisor._validate_attempt(
            changed, output_root=str(attempt_root.resolve())
        )
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
        supervisor._validate_attempt(
            attempt, output_root=str(CONSUMED_ATTEMPT_ROOT.resolve())
        )
    for consumed_id, consumed_root in (
        (CONSUMED_ATTEMPT_ID, CONSUMED_ATTEMPT_ROOT),
        (ORIGINAL_CONSUMED_ATTEMPT_ID, ORIGINAL_CONSUMED_ATTEMPT_ROOT),
    ):
        changed = dict(attempt)
        changed["id"] = consumed_id
        with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
            supervisor._validate_attempt(
                changed, output_root=str(attempt_root.resolve())
            )
        changed = dict(attempt)
        changed["root"] = str(consumed_root.resolve())
        with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
            supervisor._validate_attempt(
                changed, output_root=str(attempt_root.resolve())
            )
        with pytest.raises(supervisor.ThreeArmSupervisionError, match="one-shot"):
            supervisor._validate_attempt(
                attempt, output_root=str(consumed_root.resolve())
            )


def test_predecessor_failure_audit_must_close_consumed_attempt() -> None:
    supervisor = _load_supervisor()
    audit = json.loads(PREDECESSOR_FAILURE_AUDIT.read_text(encoding="utf-8"))
    assert supervisor.file_binding(PREDECESSOR_FAILURE_AUDIT) == (
        supervisor.PREDECESSOR_FAILURE_BINDING
    )
    supervisor._validate_predecessor_failure(audit)
    supervisor._reverify_predecessor_failure_evidence(audit)
    mutations = (
        ("attempt", "retry_authorized", True),
        ("execution_accounting", "training_updates_completed", 1),
        ("execution_accounting", "supervisor_wall_elapsed_seconds_at_terminal", 0.0),
        ("terminal_evidence", "phase_receipts_empty", False),
        ("failure", "classification", "wrong"),
        ("failure", "location", "after scientific evaluation"),
        ("root_cause", "registered_arm_parameter_tensor_count", 35),
        ("root_cause", "causal_chain", ["contradictory alternative cause"]),
        ("narrow_integrity_correction", "parameter_values_changed", True),
        ("scientific_conclusion", "data_learnability_tested", True),
        ("successor_boundary", "this_document_authorizes_v2", True),
        ("custody", "network_access_used", True),
    )
    for section, key, value in mutations:
        changed = json.loads(json.dumps(audit))
        changed[section][key] = value
        with pytest.raises(
            supervisor.ThreeArmSupervisionError,
            match="replacement-safe",
        ):
            supervisor._validate_predecessor_failure(changed)
    changed = json.loads(json.dumps(audit))
    changed["terminal_artifacts"]["failure"]["file_sha256"] = "0" * 64
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="replacement-safe",
    ):
        supervisor._validate_predecessor_failure(changed)
    for section, key in (
        ("successor_boundary", "authorizes_v2"),
        ("scientific_conclusion", "scientific_verdict_available"),
        ("custody", "protected_payload_opened"),
    ):
        changed = json.loads(json.dumps(audit))
        changed[section][key] = True
        with pytest.raises(
            supervisor.ThreeArmSupervisionError,
            match="keys changed",
        ):
            supervisor._validate_predecessor_failure(changed)
    changed = json.loads(json.dumps(audit))
    changed["authorizes_v2"] = True
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="keys changed",
    ):
        supervisor._validate_predecessor_failure(changed)


def test_plan_registered_fields_are_exact_and_fail_closed() -> None:
    supervisor = _load_supervisor()
    plan = json.loads(REPLACEMENT_PLAN.read_text(encoding="utf-8"))
    supervisor._validate_plan_registered_fields(plan)
    assert supervisor._validate_exact_runtime(
        plan["runtime"], verify_files=True
    ) == supervisor.EXPECTED_RUNTIME
    assert supervisor._validate_exact_inputs(
        plan["input_bindings"], verify_files=False
    ) == supervisor.EXPECTED_INPUT_BINDINGS
    mutations = (
        ("development_only", False),
        ("claim_scope", "architecture_learnability"),
        ("network_access", True),
        ("minimum_free_output_bytes_before_reservation", 1),
        ("result_chain", []),
        ("prior_attempt_runtime_payloads_authorized_as_inputs", True),
        ("pack_rebuilt_fresh", False),
    )
    for key, value in mutations:
        changed = json.loads(json.dumps(plan))
        changed[key] = value
        with pytest.raises(
            supervisor.ThreeArmSupervisionError,
            match="exact authorized experiment",
        ):
            supervisor._validate_plan_registered_fields(changed)
    changed = json.loads(json.dumps(plan))
    changed["input_binding_interpretation"]["permitted_temporal_positions"] = [0, 1]
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="exact authorized experiment",
    ):
        supervisor._validate_plan_registered_fields(changed)
    changed = json.loads(json.dumps(plan))
    changed["unreviewed_extension"] = True
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="keys changed",
    ):
        supervisor._validate_plan_registered_fields(changed)
    changed_runtime = json.loads(json.dumps(plan["runtime"]))
    changed_runtime["bindings"]["python_environment_config"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="exact preregistered runtime",
    ):
        supervisor._validate_exact_runtime(changed_runtime, verify_files=False)
    changed_inputs = json.loads(json.dumps(plan["input_bindings"]))
    changed_inputs["train_index"]["file_sha256"] = "0" * 64
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="exact preregistered inputs",
    ):
        supervisor._validate_exact_inputs(changed_inputs, verify_files=False)


def test_review_binds_preregistration_and_commit_order_is_strict() -> None:
    supervisor = _load_supervisor()
    assert supervisor._validate_authorizer(
        {"identity": "workspace owner"}, issued_at="2026-08-01T00:00:00+01:00"
    ) == {"identity": "workspace owner"}
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="keys changed",
    ):
        supervisor._validate_authorizer(
            {"identity": "workspace owner", "authority_granted": False},
            issued_at="2026-08-01T00:00:00+01:00",
        )
    source_commit = "a" * 40
    source_bindings = [{"name": "worker", "binding": _inert("/worker.py")}]
    plan_binding = _inert("/plan.json")
    preregistration_binding = _inert("/preregistration.md")
    predecessor_failure_binding = _inert("/predecessor_failure.json")
    review = {
        "schema": supervisor.REVIEW_SCHEMA,
        "status": supervisor.REVIEW_STATUS,
        "authority_granted_by_this_document": False,
        "reviewer": {"identity": "/root/reviewer", "materialization": "fixture"},
        "reviewed_source_commit": source_commit,
        "reviewed_source_bindings": source_bindings,
        "reviewed_plan_binding": plan_binding,
        "reviewed_predecessor_terminal_failure_binding": (
            predecessor_failure_binding
        ),
        "reviewed_preregistration_binding": preregistration_binding,
        "review_scope": dict(supervisor.EXPECTED_REVIEW_SCOPE),
        "verification": dict(supervisor.EXPECTED_REVIEW_VERIFICATION),
        "custody": dict(supervisor.EXPECTED_REVIEW_CUSTODY),
        "resolved_findings": [],
        "remaining_findings": [],
    }
    supervisor._validate_review(
        review,
        source_commit=source_commit,
        source_bindings=source_bindings,
        plan_binding=plan_binding,
        preregistration_binding=preregistration_binding,
        predecessor_failure_binding=predecessor_failure_binding,
    )
    changed = json.loads(json.dumps(review))
    changed["reviewed_preregistration_binding"] = _inert("/wrong.md")
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="non-authorizing PASS",
    ):
        supervisor._validate_review(
            changed,
            source_commit=source_commit,
            source_bindings=source_bindings,
            plan_binding=plan_binding,
            preregistration_binding=preregistration_binding,
            predecessor_failure_binding=predecessor_failure_binding,
        )
    for section, key, value in (
        ("review_scope", "normalized_scientific_difference_count", 1),
        ("verification", "focused_tests_passed", 0),
        ("custody", "heldout_or_sealed_opened", True),
    ):
        changed = json.loads(json.dumps(review))
        changed[section][key] = value
        with pytest.raises(
            supervisor.ThreeArmSupervisionError,
            match="non-authorizing PASS",
        ):
            supervisor._validate_review(
                changed,
                source_commit=source_commit,
                source_bindings=source_bindings,
                plan_binding=plan_binding,
                preregistration_binding=preregistration_binding,
                predecessor_failure_binding=predecessor_failure_binding,
            )
    parent = supervisor._git_output("rev-parse", "HEAD^")
    head = supervisor._git_head()
    supervisor._require_strict_commit_ancestor(
        parent, head, label="fixture-order"
    )
    with pytest.raises(
        supervisor.ThreeArmSupervisionError,
        match="distinct",
    ):
        supervisor._require_strict_commit_ancestor(
            head, head, label="fixture-order"
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
        ("maximum_wall_seconds", 1.0),
        ("maximum_wall_seconds", 43_200.1),
        ("maximum_gpu_seconds", 1.0),
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
        "id": REPLACEMENT_ATTEMPT_ID,
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
        "review_commit": "b" * 40,
        "preregistration_binding": _inert("/preregistration.md"),
        "source_bindings": [],
        "runtime": {},
        "input_bindings": {},
        "predecessor_terminal_failure_binding": _inert(
            "/predecessor_terminal_failure.json"
        ),
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
    assert reservation["schema"] == REPLACEMENT_SCHEMA_PREFIX + "reservation_v1"
    assert reservation["maximum_attempts"] == 1
    assert reservation["authorized_device_idle_preflight_passed"] is True
    assert reservation["retry_authorized"] is False
    assert reservation["predecessor_terminal_failure_binding"] == authority[
        "predecessor_terminal_failure_binding"
    ]
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

    monkeypatch.setattr(
        supervisor,
        "verify_binding",
        lambda binding, *, label: dict(binding),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "gpu": 0,
                        "process_list": [
                            {"process_info": "No running processes detected"}
                        ],
                    }
                ]
            ).encode("utf-8"),
            stderr=b"",
        ),
    )
    supervisor._require_idle_authorized_device()
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [{"gpu": 0, "process_list": [{"pid": 1234}]}]
            ).encode("utf-8"),
            stderr=b"",
        ),
    )
    with pytest.raises(supervisor.ThreeArmSupervisionError, match="not idle"):
        supervisor._require_idle_authorized_device()


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
        "id": REPLACEMENT_ATTEMPT_ID,
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
        "predecessor_terminal_failure_binding": _inert(
            "/predecessor_terminal_failure.json"
        ),
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
        "predecessor_terminal_failure_binding": authority[
            "predecessor_terminal_failure_binding"
        ],
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
        "id": REPLACEMENT_ATTEMPT_ID,
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
        "review_commit": "b" * 40,
        "preregistration_binding": _inert("/preregistration.md"),
        "source_bindings": [{"name": "worker", "binding": worker_binding}],
        "runtime": {
            "python_invocation_path": sys.executable,
            "environment": dict(supervisor.EXACT_CHILD_ENVIRONMENT),
            "bindings": {},
        },
        "input_bindings": {"input": _inert("/input.json")},
        "predecessor_terminal_failure_binding": _inert(
            "/predecessor_terminal_failure.json"
        ),
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
    monkeypatch.setattr(supervisor, "_require_idle_authorized_device", lambda: None)
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
                "predecessor_terminal_failure_binding": authority[
                    "predecessor_terminal_failure_binding"
                ],
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
                    "predecessor_terminal_failure_binding": authority[
                        "predecessor_terminal_failure_binding"
                    ],
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
    assert terminal["schema"] == (
        REPLACEMENT_SCHEMA_PREFIX + "supervision_terminal_v1"
    )
    assert terminal["predecessor_terminal_failure_binding"] == authority[
        "predecessor_terminal_failure_binding"
    ]
    assert len(launched) == 2
    assert Path(launched[0][1]).name == (
        "execute_go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert "--expected-authority-sha256" in launched[0]
    assert Path(launched[1][1]).name == (
        "check_go2_world_model_existing_pool_three_arm_v1.py"
    )
    assert (attempt_root / "reservation.json").is_file()
    reservation = json.loads(
        (attempt_root / "reservation.json").read_text(encoding="utf-8")
    )
    assert reservation["schema"] == (
        REPLACEMENT_SCHEMA_PREFIX + "reservation_v1"
    )
    assert reservation["predecessor_terminal_failure_binding"] == authority[
        "predecessor_terminal_failure_binding"
    ]
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
