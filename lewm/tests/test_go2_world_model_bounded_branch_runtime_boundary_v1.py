from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as collector
from scripts import run_go2_world_model_bounded_branch_experiment_authorized_v1 as supervisor


def _binding(path: str, digit: str) -> dict[str, object]:
    return {
        "path": path,
        "file_sha256": digit * 64,
        "byte_count": 1,
    }


def _authority(attempt_root: Path) -> dict[str, object]:
    return {
        "attempt": {
            "id": "bounded-attempt",
            "root": str(attempt_root),
            "maximum_attempts": 1,
            "must_be_absent": True,
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
        "review_binding": _binding("/tmp/review.json", "3"),
        "source_bindings": [],
        "caps": {"selected_device_vram_byte_ceiling": 100},
    }


def _render_metrics() -> dict[str, object]:
    return {
        "native_render_calls": 96,
        "rgb_render_calls": 96,
        "auxiliary_depth_render_calls": 96,
        "stored_rgb_frames": 96,
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": pilot.TEXTURED_V03_VISUAL_MODE,
        "derived_mesh_bindings": [_binding("/tmp/mesh.obj", "6")],
    }


def _render_plan() -> dict[str, object]:
    return {
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "visual_domain_parity_result_binding": _binding(
            "/tmp/visual-parity-result.json", "7"
        ),
        "visual_domain_parity_terminal_binding": _binding(
            "/tmp/visual-parity-terminal.json", "8"
        ),
        "visual_domain_parity_review_binding": _binding(
            "/tmp/visual-parity-review.json", "9"
        ),
    }


def test_bounded_render_receipt_identity_is_exact_textured_v03() -> None:
    plan = _render_plan()
    identity = collector._validated_render_receipt_identity_v1(
        plan=plan,
        metrics=_render_metrics(),
    )
    assert identity == {
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "native_render_calls": 96,
        "rgb_render_calls": 96,
        "auxiliary_depth_render_calls": 96,
        "stored_rgb_frames": 96,
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": pilot.TEXTURED_V03_VISUAL_MODE,
        "visual_domain_fidelity_claimed": True,
        "visual_domain_parity_result_binding": _binding(
            "/tmp/visual-parity-result.json", "7"
        ),
        "visual_domain_parity_terminal_binding": _binding(
            "/tmp/visual-parity-terminal.json", "8"
        ),
        "visual_domain_parity_review_binding": _binding(
            "/tmp/visual-parity-review.json", "9"
        ),
        "derived_mesh_bindings": [_binding("/tmp/mesh.obj", "6")],
    }
    source = Path(collector.__file__).read_text(encoding="utf-8")
    assert '"schema": pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA' in source


@pytest.mark.parametrize(
    ("plan", "metrics", "message"),
    [
        (
            {"render_contract": dict(pilot.RENDER_CONTRACT)},
            _render_metrics(),
            "versioned textured_v03",
        ),
        (
            _render_plan(),
            {**_render_metrics(), "visual_mode": "solid_materials"},
            "metrics disagree",
        ),
        (
            _render_plan(),
            {**_render_metrics(), "depth_rendered": False},
            "metrics disagree",
        ),
        (
            _render_plan(),
            {**_render_metrics(), "stored_rgb_frames": 95},
            "accounting changed",
        ),
        (
            _render_plan(),
            {**_render_metrics(), "rgb_render_calls": 95},
            "accounting changed",
        ),
        (
            _render_plan(),
            {**_render_metrics(), "auxiliary_depth_render_calls": 95},
            "accounting changed",
        ),
    ],
)
def test_bounded_render_receipt_identity_rejects_contract_or_metric_drift(
    plan: dict[str, object],
    metrics: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(pilot.PilotContractError, match=message):
        collector._validated_render_receipt_identity_v1(
            plan=plan,
            metrics=metrics,
        )


def test_supervisor_exclusively_reserves_then_direct_child_can_enter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_root = tmp_path / ".generated/dev"
    development_root.mkdir(parents=True)
    attempt_root = development_root / "bounded-attempt"
    authority = _authority(attempt_root)
    authority_binding = _binding("/tmp/authority.json", "1")
    plan_binding = _binding("/tmp/plan.json", "2")
    nonce = "4" * 64
    supervisor_pid = os.getpid()
    monkeypatch.setattr(supervisor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)

    reservation_binding = supervisor._reserve_attempt_v1(
        attempt_root,
        nonce=nonce,
        supervisor_pid=supervisor_pid,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
    assert reservation_binding["path"] == "reservation.json"
    assert {path.name for path in attempt_root.iterdir()} == {"reservation.json"}
    reservation_before = pilot.file_binding(attempt_root / "reservation.json")

    monkeypatch.setattr(collector.os, "getppid", lambda: supervisor_pid)
    output_root, observed_binding = collector._load_supervisor_owned_reservation_v1(
        output_root_text=str(attempt_root),
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        supervisor_nonce=nonce,
        supervisor_pid=supervisor_pid,
    )
    assert output_root == attempt_root
    assert observed_binding == reservation_binding

    with pytest.raises(
        pilot.PilotContractError, match="reservation identity changed"
    ):
        collector._load_supervisor_owned_reservation_v1(
            output_root_text=str(attempt_root),
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            supervisor_nonce="5" * 64,
            supervisor_pid=supervisor_pid,
        )
    with pytest.raises(
        pilot.PilotContractError, match="not a direct child"
    ):
        collector._load_supervisor_owned_reservation_v1(
            output_root_text=str(attempt_root),
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            supervisor_nonce=nonce,
            supervisor_pid=supervisor_pid + 1,
        )
    with pytest.raises(
        supervisor.BoundedBranchSupervisionError, match="exclusive supervisor reservation"
    ):
        supervisor._reserve_attempt_v1(
            attempt_root,
            nonce=nonce,
            supervisor_pid=supervisor_pid,
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
        )
    assert pilot.file_binding(attempt_root / "reservation.json") == reservation_before


def test_collector_rejects_any_state_preceding_its_owned_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_root = tmp_path / ".generated/dev"
    development_root.mkdir(parents=True)
    attempt_root = development_root / "bounded-attempt"
    authority = _authority(attempt_root)
    authority_binding = _binding("/tmp/authority.json", "1")
    plan_binding = _binding("/tmp/plan.json", "2")
    nonce = "4" * 64
    supervisor_pid = os.getpid()
    monkeypatch.setattr(supervisor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(collector.os, "getppid", lambda: supervisor_pid)
    supervisor._reserve_attempt_v1(
        attempt_root,
        nonce=nonce,
        supervisor_pid=supervisor_pid,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
    (attempt_root / "foreign-state").write_text("not this attempt", encoding="utf-8")
    with pytest.raises(pilot.PilotContractError, match="pre-existing attempt state"):
        collector._load_supervisor_owned_reservation_v1(
            output_root_text=str(attempt_root),
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            supervisor_nonce=nonce,
            supervisor_pid=supervisor_pid,
        )


def test_root_creation_without_reservation_still_consumes_bounded_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development_root = tmp_path / ".generated/dev"
    development_root.mkdir(parents=True)
    attempt_root = development_root / "crashed-before-reservation"
    attempt_root.mkdir()
    authority = _authority(attempt_root)
    monkeypatch.setattr(supervisor, "REPO_ROOT", tmp_path)

    with pytest.raises(
        supervisor.BoundedBranchSupervisionError,
        match="exclusive supervisor reservation",
    ):
        supervisor._reserve_attempt_v1(
            attempt_root,
            nonce="4" * 64,
            supervisor_pid=os.getpid(),
            authority=authority,
            authority_binding=_binding("/tmp/authority.json", "1"),
            plan_binding=_binding("/tmp/plan.json", "2"),
        )
    assert not (attempt_root / "reservation.json").exists()


def test_active_vram_ceiling_terminates_collector_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 9876

        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return -15 if self.terminated else None

    process = FakeProcess()
    popen_calls: list[tuple[object, object]] = []

    def fake_popen(argv, **kwargs):
        popen_calls.append((argv, kwargs))
        return process

    def fake_terminate(selected) -> None:
        assert selected is process
        process.terminated = True

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        supervisor.calibration_supervisor,
        "_terminate_process_group",
        fake_terminate,
    )
    enforcement = {
        "peak_observed_during_collector_bytes": 90,
        "collector_started": False,
        "collector_pid": None,
        "collector_terminated": False,
        "termination_reason": None,
    }
    sampler = SimpleNamespace(peak_used_bytes=101, read_errors=0)
    with pytest.raises(
        supervisor.BoundedBranchSupervisionError,
        match="active selected-device VRAM ceiling exceeded",
    ):
        supervisor._run_collector_once_with_vram_ceiling(
            ["python", "collector.py"],
            timeout=10.0,
            env={},
            sampler=sampler,
            ceiling_bytes=100,
            enforcement=enforcement,
        )
    assert len(popen_calls) == 1
    assert process.terminated is True
    assert enforcement["collector_terminated"] is True
    assert enforcement["termination_reason"] == "vram_ceiling_exceeded"
    assert enforcement["peak_observed_during_collector_bytes"] == 101


def test_failed_physics_result_remains_bound_to_owned_reservation(
    tmp_path: Path,
) -> None:
    plan_binding = _binding("/tmp/plan.json", "2")
    authority_binding = _binding("/tmp/authority.json", "1")
    reservation_binding = _binding("reservation.json", "4")
    authority = _authority(tmp_path)
    plan = {"attempt_id": "bounded-attempt"}
    result = {
        "schema": pilot.PHYSICS_RESULT_SCHEMA,
        "attempt_id": plan["attempt_id"],
        "purpose": "bounded_wm_a_pilot",
        "status": "FAILED",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "reservation_binding": reservation_binding,
        "review_binding": authority["review_binding"],
        "source_bindings": authority["source_bindings"],
        "caps": authority["caps"],
        "failure": {"type": "PilotContractError", "message": "task-valid failure"},
    }
    (tmp_path / "physics_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    observed, binding = supervisor._load_physics_result_if_present(
        tmp_path,
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        reservation_binding=reservation_binding,
    )
    assert observed is not None and observed["status"] == "FAILED"
    assert binding == pilot.file_binding(tmp_path / "physics_result.json")

    with pytest.raises(
        supervisor.BoundedBranchSupervisionError, match="exact terminal receipt"
    ):
        supervisor._load_physics_result_if_present(
            tmp_path,
            plan=plan,
            plan_binding=plan_binding,
            authority=authority,
            authority_binding=authority_binding,
            reservation_binding=_binding("reservation.json", "5"),
        )
