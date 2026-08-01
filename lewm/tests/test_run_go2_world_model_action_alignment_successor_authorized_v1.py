from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from scripts import execute_go2_world_model_action_alignment_successor_v1 as worker
from scripts import run_go2_world_model_action_alignment_successor_authorized_v1 as supervisor


def test_environment_guard_accepts_only_exact_allowlist():
    with mock.patch.dict(supervisor.os.environ, worker.EXACT_CHILD_ENVIRONMENT, clear=True):
        supervisor._require_exact_environment_before_worker_import()
    changed = dict(worker.EXACT_CHILD_ENVIRONMENT)
    changed["EXTRA"] = "1"
    with mock.patch.dict(supervisor.os.environ, changed, clear=True), pytest.raises(SystemExit):
        supervisor._require_exact_environment_before_worker_import()


def test_reservation_templates_bind_one_worker_and_checker():
    authority = {
        "source_commit": "1" * 40,
        "review_commit": "2" * 40,
        "execution_head": "3" * 40,
        "plan_binding": {"x": 1},
        "review_binding": {"x": 2},
        "source_bindings": {"x": 3},
        "test_bindings": {"x": 4},
        "input_bindings": {"x": 5},
        "evidence_bindings": {"x": 6},
        "runtime": worker.EXPECTED_RUNTIME,
        "caps": {
            "maximum_wall_seconds": worker.MAXIMUM_WALL_SECONDS,
            "maximum_gpu_seconds": worker.MAXIMUM_GPU_SECONDS,
            "maximum_training_updates": worker.TRAINING_UPDATES,
        },
    }
    binding = {"path": str(worker.AUTHORITY_PATH), "file_sha256": "a" * 64, "byte_count": 1}
    reservation = worker.expected_reservation(authority, binding, supervisor_nonce="b" * 64)
    assert reservation["maximum_attempts"] == 1
    assert reservation["retry"] is False
    assert reservation["resume"] is False
    assert reservation["worker_command_template"].count("<SUPERVISOR_BOUND_RESERVATION_SHA256>") == 1
    assert reservation["checker_command_template"].count("<WORKER_RESULT_SHA256>") == 1
    assert "integrity_replacement_v1" in reservation["attempt_id"]
    assert worker.ATTEMPT_ROOT.parent.name.endswith("integrity_replacement_v1")


def test_command_instantiation_replaces_only_exact_placeholders():
    assert supervisor._instantiate(
        ["a", "<X>", "b"], {"<X>": "value"}
    ) == ["a", "value", "b"]


def test_replacement_is_final_and_binds_closed_attempt_receipts():
    required = {
        "original_successor_authority",
        "original_successor_reservation",
        "original_successor_failure",
        "original_successor_terminal",
        "original_successor_failure_audit",
    }
    assert required <= set(worker.EXPECTED_EVIDENCE_BINDINGS)
    assert all(
        worker._binding_is_exact(
            Path(worker.EXPECTED_EVIDENCE_BINDINGS[name]["path"]),
            worker.EXPECTED_EVIDENCE_BINDINGS[name],
        )
        for name in required
    )
    assert {path.name for path in worker.ORIGINAL_ATTEMPT_ROOT.iterdir()} == {
        "reservation.json", "failure.json", "terminal_supervision.json",
    }
