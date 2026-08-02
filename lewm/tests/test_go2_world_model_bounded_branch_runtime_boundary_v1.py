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


def test_bounded_plan_accepts_complete_parity_prerequisite_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _render_plan()
    expected = {
        "result_binding": plan["visual_domain_parity_result_binding"],
        "terminal_binding": plan["visual_domain_parity_terminal_binding"],
        "review_binding": plan["visual_domain_parity_review_binding"],
    }
    monkeypatch.setattr(
        collector.kernel,
        "_validate_visual_domain_parity_result",
        lambda _plan: expected,
    )
    assert collector._validate_plan_parity_prerequisites_v1(plan) == expected


def test_bounded_plan_rejects_parity_prerequisite_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _render_plan()
    monkeypatch.setattr(
        collector.kernel,
        "_validate_visual_domain_parity_result",
        lambda _plan: {
            "result_binding": plan["visual_domain_parity_result_binding"],
            "terminal_binding": plan["visual_domain_parity_terminal_binding"],
            "review_binding": _binding("/tmp/wrong-review.json", "a"),
        },
    )
    with pytest.raises(pilot.PilotContractError, match="prerequisite bindings"):
        collector._validate_plan_parity_prerequisites_v1(plan)


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


def test_authorized_collector_runs_two_fixed_batches_and_one_scene_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    urdf_path = tmp_path / "go2.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    scene_manifest = _binding(str(tmp_path / "scene-manifest.json"), "a")
    scene_genesis = _binding(str(tmp_path / "scene-genesis.json"), "b")
    states = [
        {
            "state_id": f"train-state-{index}",
            "role": "train",
            "family": "open_obstacle_field",
            "scene_id": "open-obstacle-train",
            "group_index": index,
            "state_index_in_scene": index,
            "scene_generation": None,
            "scene_manifest_binding": scene_manifest,
            "scene_genesis_binding": scene_genesis,
        }
        for index in range(8)
    ]
    plan = {
        **_render_plan(),
        "attempt_id": "bounded-attempt",
        "purpose": "bounded_wm_a_pilot",
        "output_root": str(output_root),
        "states": states,
        "runtime_bindings": {
            "platform_manifest": {"path": str(tmp_path / "platform.json")},
            "primitive_registry": {"path": str(tmp_path / "registry.yaml")},
            "go2_urdf": pilot.file_binding(urdf_path),
        },
        "execution_contract": {},
        "action_catalog": [{"action_id": index} for index in range(9)],
        "expected_counts": {
            "scenes": 1,
            "states": 8,
            "roles": {"train": 8},
            "actions": 9,
            "candidate_branches": 72,
            "sentinel_branches": 0,
            "total_branches": 72,
            "context_frames": 24,
            "target_frames": 72,
        },
    }
    authority = {
        "attempt": {"id": "bounded-attempt"},
        "review_binding": _binding("/tmp/review.json", "3"),
        "source_bindings": [],
        "caps": {
            "wall_seconds": 100.0,
            "stored_rgb_byte_ceiling": 1_000,
            "native_render_calls": 96,
            "rgb_render_calls": 96,
            "auxiliary_depth_render_calls": 96,
            "stored_rgb_frames": 96,
        },
    }
    plan_binding = _binding("/tmp/plan.json", "2")
    authority_binding = _binding("/tmp/authority.json", "1")
    monkeypatch.setattr(
        collector,
        "load_and_validate_v1",
        lambda **_kwargs: (authority, authority_binding, plan, plan_binding),
    )
    monkeypatch.setattr(
        collector,
        "_load_supervisor_owned_reservation_v1",
        lambda **_kwargs: (output_root, _binding("reservation.json", "4")),
    )
    monkeypatch.setattr(
        collector.kernel,
        "_copy_exact_plan_receipt",
        lambda *_args, **_kwargs: _binding("authorized_plan.json", "5"),
    )
    monkeypatch.setattr(collector.pilot, "require_plan_bindings", lambda _plan: None)
    monkeypatch.setattr(collector.kernel, "_validate_python_runtime", lambda _plan: None)
    monkeypatch.setattr(
        collector.kernel, "_validate_execution_environment", lambda _plan: None
    )
    monkeypatch.setattr(
        collector.kernel,
        "_capture_runtime_versions",
        lambda: {"python": "test"},
    )

    class Registry:
        @classmethod
        def from_yaml(cls, _path: str) -> object:
            return object()

    runtime = {
        "load_platform_manifest": lambda _path: {},
        "resolve_go2_urdf": lambda _platform, _root: urdf_path,
        "PrimitiveRegistry": Registry,
        "expand_primitive_to_block": lambda *_args: [],
    }
    monkeypatch.setattr(collector.kernel, "_runtime_imports", lambda: runtime)
    monkeypatch.setattr(
        collector.kernel,
        "_load_action_blocks",
        lambda **_kwargs: [],
    )
    observed_batches: list[list[str]] = []

    def collect_batch(*, states, **_kwargs):
        state_ids = [str(state["state_id"]) for state in states]
        observed_batches.append(state_ids)
        receipts = [
            {
                "state": {
                    "state_id": state["state_id"],
                    "role": state["role"],
                    "scene_id": state["scene_id"],
                },
                "branches": [{"kind": "candidate"} for _ in range(9)],
                "context": {"frame_identities": ["h0", "h1", "h2"]},
            }
            for state in states
        ]
        frames = [
            {
                "frame_identity": f"{state['state_id']}-{frame_index}",
                "byte_count": 1,
            }
            for state in states
            for frame_index in range(12)
        ]
        metrics = {
            "scene_id": "open-obstacle-train",
            "family": "open_obstacle_field",
            "role": "train",
            "states": 4,
            "envs": 36,
            "physics_build_wall_seconds": 0.0,
            "physics_simulation_wall_seconds": 0.0,
            "common_prefix_step_wall_seconds": 0.0,
            "branch_step_wall_seconds": 0.0,
            "render_scene_build_wall_seconds": 0.0,
            "native_render_wall_seconds": 0.0,
            "camera_quality_resize_wall_seconds": 0.0,
            "png_encode_write_hash_wall_seconds": 0.0,
            "lockstep_execution_wall_seconds": 0.0,
            "post_lockstep_receipt_wall_seconds": 0.0,
            "scene_pipeline_wall_seconds": 0.0,
            "scene_total_wall_seconds": 0.0,
            "native_render_calls": 48,
            "rgb_render_calls": 48,
            "auxiliary_depth_render_calls": 48,
            "stored_rgb_frames": 48,
            "depth_rendered": True,
            "depth_persisted": False,
            "visual_mode": pilot.TEXTURED_V03_VISUAL_MODE,
            "derived_mesh_bindings": [_binding("/tmp/mesh.obj", "6")],
        }
        return receipts, frames, [], [], metrics

    monkeypatch.setattr(collector.kernel, "_collect_scene", collect_batch)
    result, result_path = collector.collect_v1(
        plan_path=Path("/tmp/plan.json"),
        expected_plan_byte_count=1,
        expected_plan_sha256="2" * 64,
        authority_path=Path("/tmp/authority.json"),
        expected_authority_byte_count=1,
        expected_authority_sha256="1" * 64,
        supervisor_nonce="4" * 64,
        supervisor_pid=123,
    )
    assert result["status"] == "PHYSICS_COMPLETE", result["failure"]
    assert result_path == output_root / "physics_result.json"
    assert observed_batches == [
        [f"train-state-{index}" for index in range(4)],
        [f"train-state-{index}" for index in range(4, 8)],
    ]
    assert len(result["render_receipt_bindings"]) == 1
    assert len(result["scene_metrics"]) == 1
    assert result["scene_metrics"][0]["states"] == 8
    assert result["scene_metrics"][0]["envs"] == 72
    assert result["scene_metrics"][0]["stored_rgb_frames"] == 96
    render_receipt = json.loads(
        (output_root / "scenes/train/open-obstacle-train/live_render_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(render_receipt["frame_receipts"]) == 96
    assert render_receipt["frame_receipts"][0]["frame_identity"].startswith(
        "train-state-0-"
    )
    assert render_receipt["frame_receipts"][48]["frame_identity"].startswith(
        "train-state-4-"
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
