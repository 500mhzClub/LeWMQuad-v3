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
            "maximum_additional_training_updates": worker.ADDITIONAL_TRAINING_UPDATES,
            "maximum_global_training_update": worker.TRAINING_UPDATES,
        },
    }
    binding = {"path": str(worker.AUTHORITY_PATH), "file_sha256": "a" * 64, "byte_count": 1}
    reservation = worker.expected_reservation(authority, binding, supervisor_nonce="b" * 64)
    assert reservation["maximum_attempts"] == 1
    assert reservation["retry"] is False
    assert reservation["resume"] is False
    assert reservation["worker_command_template"].count("<SUPERVISOR_BOUND_RESERVATION_SHA256>") == 1
    assert reservation["checker_command_template"].count("<WORKER_RESULT_SHA256>") == 1
    assert "fixed_same_mechanism_continuation_v1" in reservation["attempt_id"]
    assert worker.ATTEMPT_ROOT.parent.name.endswith("fixed_same_mechanism_continuation_v1")


def test_command_instantiation_replaces_only_exact_placeholders():
    assert supervisor._instantiate(
        ["a", "<X>", "b"], {"<X>": "value"}
    ) == ["a", "value", "b"]


def test_continuation_is_bounded_and_binds_completed_predecessor_receipts():
    required = {
        "completed_successor_authority",
        "completed_successor_reservation",
        "completed_successor_result",
        "completed_successor_receipt_check",
        "completed_successor_terminal",
        "completed_successor_terminal_review",
        "preauthority_identity_read_disclosure",
        "continuation_governance_correction",
    }
    assert required <= set(worker.EXPECTED_EVIDENCE_BINDINGS)
    assert all(
        worker._binding_is_exact(
            Path(worker.EXPECTED_EVIDENCE_BINDINGS[name]["path"]),
            worker.EXPECTED_EVIDENCE_BINDINGS[name],
        )
        for name in required
    )
    assert "completed_successor_metric_bundle" not in worker.EXPECTED_EVIDENCE_BINDINGS
    assert {
        "baseline_u700_snapshot", "alignment_u700_snapshot"
    } <= set(worker.EXPECTED_INPUT_BINDINGS)
