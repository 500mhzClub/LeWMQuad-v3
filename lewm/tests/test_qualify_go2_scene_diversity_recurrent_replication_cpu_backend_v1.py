from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import (
    qualify_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as qualifier,
)


def test_qualification_contract_is_exact_bounded_and_nonreusable() -> None:
    contract = qualifier.QUALIFICATION_CONTRACT
    assert contract["probe_scene_indices_in_order"] == [12, 0]
    assert contract["fresh_worker_processes"] == 2
    assert contract["states_per_worker"] == 4
    assert contract["candidate_actions_per_state"] == 9
    assert contract["stored_rgb_frames_per_worker"] == 48
    assert contract["auxiliary_depth_validation_renders_per_worker"] == 48
    assert contract["worker_process_group_watchdog_seconds"] == 180.0
    assert contract["timing_gate_formula"] == "64*max(worker_elapsed_seconds)+900<=7200"
    assert contract["probe_output_scientific_reuse_authorized"] is False
    assert qualifier.plan_builder.QUALIFICATION_ATTEMPT_ROOT != (
        qualifier.plan_builder.DEFAULT_ATTEMPT_ROOT
    )


def test_qualification_overlay_uses_separate_authority_and_result_identity() -> None:
    originals = {
        name: getattr(qualifier.collector, name)
        for name in (
            "AUTHORITY_FIELDS",
            "AUTHORITY_SCHEMA",
            "AUTHORITY_STATUS",
            "ATTEMPT_ID",
            "RESERVATION_SCHEMA",
            "SCENE_RESULT_SCHEMA",
            "_read_collection_reservation_cpu",
            "_worker_argv_cpu",
        )
    }
    with qualifier._configured_qualification_collector():  # noqa: SLF001
        assert qualifier.collector.ATTEMPT_ID == (
            qualifier.plan_builder.QUALIFICATION_ATTEMPT_ID
        )
        assert qualifier.collector.AUTHORITY_FIELDS == (
            qualifier.QUALIFICATION_AUTHORITY_FIELDS
        )
        assert qualifier.collector._worker_argv_cpu is (  # noqa: SLF001
            qualifier._worker_argv_qualification  # noqa: SLF001
        )
    assert all(
        getattr(qualifier.collector, name) is value
        for name, value in originals.items()
    )


def test_nested_qualification_runtime_validates_against_immutable_frozen_plan() -> None:
    plan = json.loads(qualifier.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text())
    with (
        qualifier._configured_qualification_collector(),  # noqa: SLF001
        qualifier.collector._configured_predecessor_collector_cpu(),  # noqa: SLF001
    ):
        assert qualifier.collector.pilot.validate_plan(plan) == plan


def test_nested_scientific_runtime_validates_against_immutable_frozen_plan() -> None:
    plan = json.loads(qualifier.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    with qualifier.collector._configured_predecessor_collector_cpu():  # noqa: SLF001
        assert qualifier.collector.pilot.validate_plan(plan) == plan


def test_execute_qualification_runs_fixed_order_and_timing_gate(
    tmp_path: Path, monkeypatch
) -> None:
    plan = json.loads(qualifier.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text())
    collection = tmp_path / "collection"
    collection.mkdir()
    reservation_path = collection / "reservation.json"
    reservation_path.write_text("{}\n")
    reservation = {
        "path": str(reservation_path.resolve()),
        "file_sha256": "a" * 64,
        "byte_count": 3,
    }
    monkeypatch.setattr(
        qualifier,
        "_reserve_qualification",
        lambda **_kwargs: (collection, reservation),
    )
    monkeypatch.setattr(
        qualifier.calibration,
        "_child_environment",
        lambda _plan: dict(qualifier.plan_builder.CPU_EXECUTION_ENVIRONMENT),
    )
    monkeypatch.setattr(
        qualifier.calibration,
        "_run_graphics_preflight",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    counter = tmp_path / "vram"
    counter.write_text("0")
    monkeypatch.setattr(
        qualifier.calibration,
        "_selected_gpu_memory_files",
        lambda _plan: (counter, counter, "0x1002", "0x7551"),
    )
    observed_order: list[int] = []

    def worker(_argv, *, scene, **_kwargs):
        index = int(scene["scene_index"])
        observed_order.append(index)
        return {
            "scene_index": index,
            "pid": 1000 + index,
            "prelaunch_baseline_used_bytes": 0,
            "exit_code": 0,
            "watchdog_timeout": False,
            "selected_device_vram_cap_breached": False,
            "fresh_process_group": True,
            "elapsed_seconds": 80.0 if index == 12 else 70.0,
        }

    monkeypatch.setattr(qualifier, "_run_worker_with_watchdog", worker)
    monkeypatch.setattr(
        qualifier.predecessor,
        "_wait_for_vram_release_v2",
        lambda *_args, **_kwargs: {"status": "PASSED"},
    )

    def load(*, scene, **_kwargs):
        role = str(scene["role"])
        counts = qualifier.predecessor._scene_expected_counts_v2(role)  # noqa: SLF001
        return (
            {"observed_counts": counts, "scene_metric": {"native_render_calls": 48}},
            {"path": f"scene_results/{int(scene['scene_index']):03d}.json", "file_sha256": "b" * 64, "byte_count": 1},
        )

    monkeypatch.setattr(qualifier.predecessor, "_load_scene_result_v2", load)
    monkeypatch.setattr(
        qualifier,
        "_kernel_events_since",
        lambda _epoch: {
            "query_succeeded": True,
            "new_amdgpu_ring_timeout_or_reset_count": 0,
            "matching_lines_sha256": "c" * 64,
        },
    )
    output = tmp_path / "qualification_result.json"
    monkeypatch.setattr(qualifier, "QUALIFICATION_RESULT_PATH", output)
    monkeypatch.setattr(
        qualifier.pilot,
        "write_json_exclusive",
        lambda path, value: path.write_text(json.dumps(value)) or {},
    )

    result = qualifier.execute_qualification(
        {
            "collection_root": str(collection),
            "caps": qualifier.collector.EXPECTED_CAPS,
            "source_bindings": {},
        },
        authority_binding={"path": "/authority", "sha256": "d" * 64, "byte_count": 1},
        plan=plan,
        plan_binding={
            "path": str(qualifier.plan_builder.QUALIFICATION_PLAN_OUTPUT),
            "sha256": "e" * 64,
            "byte_count": 1,
        },
    )

    assert observed_order == [12, 0]
    assert result["status"] == qualifier.QUALIFICATION_RESULT_STATUS
    assert result["timing_gate"]["projected_scientific_wall_seconds"] == 6020.0
    assert result["probe_output_scientific_reuse_authorized"] is False
    assert not qualifier.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()


def test_preflight_failure_consumes_and_terminalizes_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = qualifier.plan_builder.QUALIFICATION_PLAN_OUTPUT
    plan = json.loads(plan_path.read_text())
    plan_binding = qualifier._binding(plan_path)  # noqa: SLF001
    authority_path = tmp_path / "qualification_authority.json"
    authority_path.write_text("{}\n")
    authority_binding = qualifier._binding(authority_path)  # noqa: SLF001
    attempt_root = tmp_path / "attempt_v1"
    collection_root = attempt_root / "collection"
    terminal_path = attempt_root / "terminal.json"
    order: list[str] = []

    monkeypatch.setattr(
        qualifier.plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt_root
    )
    monkeypatch.setattr(
        qualifier.plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection_root
    )

    def reserve(**_kwargs):
        order.append("reserve")
        (collection_root / "scene_results").mkdir(parents=True)
        reservation_path = collection_root / "reservation.json"
        reservation_path.write_text("{}\n")
        return collection_root, qualifier.pilot.file_binding(reservation_path)

    def fail_child_environment(_plan):
        order.append("child_environment")
        raise qualifier.CpuBackendQualificationError("preflight failed")

    monkeypatch.setattr(qualifier, "_reserve_qualification", reserve)
    monkeypatch.setattr(
        qualifier.calibration, "_child_environment", fail_child_environment
    )
    monkeypatch.setattr(
        qualifier,
        "validate_qualification_authority",
        lambda *_args, **_kwargs: (
            {}, authority_binding, plan, plan_binding
        ),
    )
    status = qualifier.main(
        [
            "--plan", str(plan_path),
            "--expected-plan-byte-count", str(plan_binding["byte_count"]),
            "--expected-plan-sha256", str(plan_binding["sha256"]),
            "--authority", str(authority_path),
            "--expected-authority-byte-count", str(authority_binding["byte_count"]),
            "--expected-authority-sha256", str(authority_binding["sha256"]),
        ]
    )
    assert status == 1
    assert order == ["reserve", "child_environment"]
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "FAIL_CPU_BACKEND_QUALIFICATION_HARD_STOP"
    assert terminal["authorizes_cpu_scientific_authority"] is False
    assert terminal["authorizes_retry_or_resume"] is False
