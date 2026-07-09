from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_go2_fully_learned_demo.py"


def _base_result() -> dict[str, Any]:
    return {
        "result": {
            "scene": "medium_enclosed_maze_01732aabc542",
            "success": True,
            "ticks_used": 12,
            "claimed_colors": ["red", "green", "blue", "yellow"],
            "wall_metrics": {
                "fully_learned_runtime_contract": True,
                "fully_learned_runtime_contract_report": {
                    "passed": True,
                    "runtime_path": "learned_local_policy",
                },
                "source": "learned_action_outcome",
                "explore_goal_policy": "learned_policy",
                "learned_local_policy_checkpoint": "models/checkpoints/policy.pt",
                "learned_local_policy_ticks": 12,
                "learned_local_policy_disabled_ticks": 0,
                "learned_local_policy_feature_mismatch_ticks": 0,
                "learned_local_policy_privileged_explorer_skipped_ticks": 12,
                "contact_like_stalls": 0,
                "hard_contact_like_stalls": 0,
                "body_clearance_contact_events": 0,
                "body_clearance_violation_events": 0,
                "fall_events": 0,
                "tip_events": 0,
                "unstable_base_events": 0,
            },
        },
        "log": [],
        "provenance": {"argv": ["benchmark_go2_memory_closed_loop.py", "--mode", "physical"]},
    }


def _run_checker(tmp_path: Path, payload: dict[str, Any], *args: str) -> tuple[int, dict[str, Any]]:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--result", str(result_path), "--max-ticks", "50", *args],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.stderr == ""
    return proc.returncode, json.loads(proc.stdout)


def test_rejects_slice_result_by_default(tmp_path: Path) -> None:
    payload = _base_result()
    wall = payload["result"]["wall_metrics"]
    wall["slice_benchmark"] = True
    wall["slice_start"] = {"start_tick": 352, "preclaimed_colors": ["red", "green", "blue"]}

    code, report = _run_checker(tmp_path, payload)

    assert code == 1
    assert report["gates"]["not_slice_benchmark"] is False
    assert report["result"]["slice_benchmark"] is True


def test_can_allow_slice_result_for_slice_benchmarks(tmp_path: Path) -> None:
    payload = _base_result()
    wall = payload["result"]["wall_metrics"]
    wall["slice_benchmark"] = True
    wall["slice_start"] = {"start_tick": 352, "preclaimed_colors": ["red", "green", "blue"]}

    code, report = _run_checker(tmp_path, payload, "--allow-slice-result")

    assert code == 0
    assert report["gates"]["not_slice_benchmark"] is True


def test_locomotion_render_must_start_from_scene_spawn(tmp_path: Path) -> None:
    render_report = {
        "replay_mode": "physical",
        "locomotion_policy_replayed": True,
        "replay_start_source": "slice_start",
        "capture_rate": "policy",
        "frame_count": 12,
        "expected_frame_count": 12,
    }
    render_path = tmp_path / "render.json"
    render_path.write_text(json.dumps(render_report), encoding="utf-8")

    code, report = _run_checker(
        tmp_path,
        _base_result(),
        "--render-report",
        str(render_path),
        "--require-locomotion-policy-render",
    )
    assert code == 1
    assert report["gates"]["locomotion_policy_render"] is False
    assert report["result"]["render_replay_start_source"] == "slice_start"

    render_report["replay_start_source"] = "scene_spawn"
    render_path.write_text(json.dumps(render_report), encoding="utf-8")
    code, report = _run_checker(
        tmp_path,
        _base_result(),
        "--render-report",
        str(render_path),
        "--require-locomotion-policy-render",
    )
    assert code == 0
    assert report["gates"]["locomotion_policy_render"] is True


def test_can_require_exact_scene_and_physical_mode(tmp_path: Path) -> None:
    code, report = _run_checker(
        tmp_path,
        _base_result(),
        "--require-scene-id",
        "medium_enclosed_maze_01732aabc542",
        "--require-physical-mode",
    )
    assert code == 0
    assert report["gates"]["expected_scene_id"] is True
    assert report["gates"]["physical_benchmark_mode"] is True

    payload = _base_result()
    payload["result"]["scene"] = "medium_enclosed_maze_b7"
    payload["provenance"]["argv"] = ["benchmark_go2_memory_closed_loop.py", "--mode", "kinematic"]
    code, report = _run_checker(
        tmp_path,
        payload,
        "--require-scene-id",
        "medium_enclosed_maze_01732aabc542",
        "--require-physical-mode",
    )
    assert code == 1
    assert report["gates"]["expected_scene_id"] is False
    assert report["gates"]["physical_benchmark_mode"] is False


def test_can_require_learned_local_policy_runtime(tmp_path: Path) -> None:
    code, report = _run_checker(
        tmp_path,
        _base_result(),
        "--require-learned-local-policy-runtime",
    )
    assert code == 0
    assert report["gates"]["learned_local_policy_runtime"] is True

    payload = _base_result()
    wall = payload["result"]["wall_metrics"]
    wall["fully_learned_runtime_contract_report"]["runtime_path"] = "learned_topology_route_memory"
    wall["explore_goal_policy"] = "route"
    wall["learned_local_policy_ticks"] = 0
    wall["learned_topology_route_table"] = "route_table.json"
    wall["learned_topology_route_ticks"] = 12
    wall["learned_topology_route_privileged_explorer_skipped_ticks"] = 12

    code, report = _run_checker(
        tmp_path,
        payload,
        "--require-learned-local-policy-runtime",
    )
    assert code == 1
    assert report["gates"]["learned_local_policy_runtime"] is False


def test_locomotion_render_rejects_recorded_replay(tmp_path: Path) -> None:
    render_report = {
        "replay_mode": "recorded",
        "locomotion_policy_replayed": False,
        "replay_start_source": "scene_spawn",
        "capture_rate": "policy",
        "frame_count": 12,
        "expected_frame_count": 12,
    }
    render_path = tmp_path / "render.json"
    render_path.write_text(json.dumps(render_report), encoding="utf-8")

    code, report = _run_checker(
        tmp_path,
        _base_result(),
        "--render-report",
        str(render_path),
        "--require-locomotion-policy-render",
    )
    assert code == 1
    assert report["gates"]["locomotion_policy_render"] is False
    assert report["result"]["render_replay_mode"] == "recorded"


def test_body_contacts_are_rejected_and_backfilled_from_old_logs(tmp_path: Path) -> None:
    payload = _base_result()
    del payload["result"]["wall_metrics"]["body_clearance_contact_events"]
    payload["result"]["wall_metrics"]["body_clearance_contact_threshold_m"] = 0.001
    payload["log"] = [
        {"tick": 1, "primitive": "forward_fast", "post_body_clearance_m": 0.0},
        {"tick": 2, "primitive": "yaw_left", "post_body_clearance_m": 0.2},
    ]

    code, report = _run_checker(tmp_path, payload)

    assert code == 1
    assert report["gates"]["body_clearance_contacts"] is False
    assert report["result"]["body_clearance_contact_events"] == 1
