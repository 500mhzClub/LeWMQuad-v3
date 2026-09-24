from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import pytest

from lewm.benchmarks.go2_physical_claim_canonical import (
    canonical_content_sha256,
)
from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    object_id_reference,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKERS = {
    "generalized": ROOT / "scripts/check_go2_generalized_suite.py",
    "clean": ROOT / "scripts/check_go2_clean_demo_candidate.py",
    "fully_learned": ROOT / "scripts/check_go2_fully_learned_demo.py",
    "teacher": ROOT / "scripts/check_go2_teacher_dataset.py",
    "wallaware": ROOT / "scripts/check_go2_wallaware_closed_loop_gate.py",
}
MUTATIONS = (
    "fabricated_self_consistent_summary",
    "geometry",
    "factor",
    "event_hash",
    "summary_hash",
    "trace_hash",
    "wrong_manifest",
    "wrong_scene",
    "bool_ledger",
    "int_ledger",
)


def _manifest(*, task_count: int) -> SceneManifest:
    specs = (
        ("red", 0.8, 0.0),
        ("green", 0.0, 0.8),
        ("blue", -0.8, 0.0),
        ("yellow", 0.0, -0.8),
    )[:task_count]
    landmarks = tuple(
        BoxObject(
            object_id=f"beacon_{color}",
            kind="landmark",
            center_xyz_m=(x, y, 0.5),
            size_xyz_m=(0.2, 0.2, 1.0),
            yaw_rad=0.0,
            material_id=f"landmark_{color}",
        )
        for color, x, y in specs
    )
    return SceneManifest(
        scene_id="synthetic_checker_scene",
        family="test",
        difficulty_tier="test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec((0.0, 0.0, 0.375), (1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.1),
    )


def _canonical_result(manifest: SceneManifest) -> dict[str, Any]:
    attempts = []
    for index, landmark in enumerate(manifest.landmarks):
        reference = object_id_reference(landmark.object_id)
        attempts.append(
            build_claim_attempt(
                manifest=manifest,
                trace_id="trace",
                episode_id="episode",
                event_id=f"event-{index}",
                tick=index,
                event_index=index,
                requested_target=reference,
                claimed_target=reference,
                robot_pose_world_xy_yaw=(
                    0.0,
                    0.0,
                    math.atan2(
                        landmark.center_xyz_m[1], landmark.center_xyz_m[0]
                    ),
                ),
                pose_provenance="runtime_full_precision",
            )
        )
    raw, task_ids, task_hash = build_claim_trace(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        controller_claim_attempts=attempts,
    )
    evaluated = evaluate_physical_claim_trace(raw, manifest, task_ids, task_hash)
    colors = sorted(
        landmark.material_id.removeprefix("landmark_")
        for landmark in manifest.landmarks
    )
    return {
        "scene": manifest.scene_id,
        "claimed": True,
        "success": True,
        "claimed_colors": colors,
        "ticks_used": 12,
        "final_dist_to_target_m": 0.8,
        "first_seen_tick": 1,
        "canonical_physical_claim_trace": evaluated,
        "runtime_evaluator_access_ledger": {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
        "wall_metrics": {
            "fully_learned_runtime_contract": True,
            "generalized_runtime_contract": True,
            "fully_learned_runtime_contract_report": {
                "passed": True,
                "generalized": True,
                "runtime_path": "learned_local_policy",
            },
            "source": "learned_action_outcome",
            "explore_goal_policy": "learned_policy",
            "learned_local_policy_checkpoint": "synthetic-policy.pt",
            "learned_local_policy_ticks": 12,
            "learned_local_policy_explore_state_ticks": 12,
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
    }


def _rehash_event(event: dict[str, Any]) -> None:
    event["content_sha256"] = canonical_content_sha256(
        event, hash_field="content_sha256"
    )


def _rehash_summary(summary: dict[str, Any]) -> None:
    summary["content_sha256"] = canonical_content_sha256(
        summary, hash_field="content_sha256"
    )


def _rehash_trace(trace: dict[str, Any]) -> None:
    trace["trace_content_sha256"] = canonical_content_sha256(
        trace, hash_field="trace_content_sha256"
    )


def _mutate(
    result: dict[str, Any],
    manifest: SceneManifest,
    mutation: str,
) -> tuple[dict[str, Any], SceneManifest]:
    result = deepcopy(result)
    supplied_manifest = manifest
    trace = result["canonical_physical_claim_trace"]
    event = trace["physical_claim_evaluations"][0]
    summary = trace["physical_claim_summary"]
    if mutation == "fabricated_self_consistent_summary":
        attempt = trace["controller_claim_attempts"][0]
        pose = [10.0, 10.0, 0.0]
        attempt["robot_pose_world_xy_yaw"] = pose
        attempt["pose_hex"] = [value.hex() for value in pose]
        attempt["pose_binary64_le_sha256"] = hashlib.sha256(
            struct.pack("<3d", *pose)
        ).hexdigest()
        _rehash_trace(trace)
    elif mutation == "geometry":
        event["distance_m"] = 0.125
        event["distance_hex"] = float(0.125).hex()
        _rehash_event(event)
        _rehash_trace(trace)
    elif mutation == "factor":
        event["factors"]["identity_passes"] = False
        _rehash_event(event)
        _rehash_trace(trace)
    elif mutation == "event_hash":
        event["content_sha256"] = "0" * 64
        _rehash_trace(trace)
    elif mutation == "summary_hash":
        summary["content_sha256"] = "0" * 64
        _rehash_trace(trace)
    elif mutation == "trace_hash":
        trace["trace_content_sha256"] = "0" * 64
    elif mutation == "wrong_manifest":
        first = manifest.landmarks[0]
        moved = replace(
            first,
            center_xyz_m=(
                first.center_xyz_m[0] + 0.25,
                first.center_xyz_m[1],
                first.center_xyz_m[2],
            ),
        )
        supplied_manifest = replace(
            manifest, landmarks=(moved, *manifest.landmarks[1:])
        )
    elif mutation == "wrong_scene":
        result["scene"] = "wrong_scene"
    elif mutation == "bool_ledger":
        result["runtime_evaluator_access_ledger"][
            "evaluator_output_reads_by_controller"
        ] = False
    elif mutation == "int_ledger":
        result["runtime_evaluator_access_ledger"][
            "evaluator_output_reads_by_controller"
        ] = 1
    else:  # pragma: no cover - protects the parameter table
        raise AssertionError(mutation)
    return result, supplied_manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _wall_report(result: dict[str, Any], arm: str) -> dict[str, Any]:
    result = deepcopy(result)
    if arm == "baseline":
        result["wall_metrics"] = {
            "source": "diagnostic_only",
            "enabled": False,
            "blocked_forward_executions": 4,
            "forward_executions": 8,
            "contact_like_stalls": 1,
        }
        log = [
            {
                "state": "EXPLORE",
                "wall_guard": {"enabled": False, "vetoed": False},
            }
        ]
    elif arm == "wall":
        result["wall_metrics"] = {
            "source": "synthetic",
            "commands_total": 1,
            "blocked_forward_requests": 1,
            "blocked_forward_executions": 1,
            "forward_executions": 8,
            "wall_vetoes": 1,
            "contact_like_stalls": 1,
        }
        log = [
            {
                "state": "EXPLORE",
                "stalled": True,
                "wall_guard": {
                    "enabled": True,
                    "vetoed": True,
                    "requested_blocked": True,
                    "selected_blocked": False,
                    "candidates": [],
                },
            },
            {"state": "CLAIM", "bearing": 0.0, "area": 2.0},
        ]
    else:
        result["wall_metrics"] = {"source": "synthetic", "wall_vetoes": 0}
        log = []
    return {"result": result, "log": log}


def _run_checker(
    tmp_path: Path,
    checker: str,
    result: dict[str, Any],
    manifest: SceneManifest,
) -> tuple[int, dict[str, Any]]:
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    command = [sys.executable, str(CHECKERS[checker])]
    if checker == "wallaware":
        baseline_path = tmp_path / "baseline.json"
        wall_path = tmp_path / "wall.json"
        recall_path = tmp_path / "recall.json"
        _write_json(baseline_path, _wall_report(result, "baseline"))
        _write_json(wall_path, _wall_report(result, "wall"))
        _write_json(recall_path, _wall_report(result, "recall"))
        command.extend(
            [
                "--baseline-explore",
                str(baseline_path),
                "--wallaware-explore",
                str(wall_path),
                "--wallaware-recall",
                str(recall_path),
                "--scene-manifests",
                str(manifest_path),
            ]
        )
    else:
        result_path = tmp_path / "result.json"
        payload = {
            "result": result,
            "log": [
                {
                    "tick": tick,
                    "primitive": "forward_slow",
                    "executed_displacement_m": 0.1,
                }
                for tick in range(12)
            ],
            "provenance": {"argv": []},
        }
        _write_json(result_path, payload)
        if checker == "generalized":
            command.extend(
                [
                    "--results",
                    str(result_path),
                    "--scene-manifests",
                    str(manifest_path),
                ]
            )
        else:
            command.extend(
                [
                    "--result",
                    str(result_path),
                    "--scene-manifest",
                    str(manifest_path),
                ]
            )
    proc = subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.stderr == ""
    return proc.returncode, json.loads(proc.stdout)


@pytest.mark.parametrize("checker", tuple(CHECKERS))
def test_checker_accepts_genuine_manifest_recomputed_trace(
    tmp_path: Path,
    checker: str,
) -> None:
    task_count = 1 if checker == "wallaware" else 4
    manifest = _manifest(task_count=task_count)
    code, report = _run_checker(
        tmp_path, checker, _canonical_result(manifest), manifest
    )
    assert code == 0, report
    assert report["passed"] is True


@pytest.mark.parametrize("checker", tuple(CHECKERS))
def test_checker_requires_external_scene_manifest_argument(
    tmp_path: Path,
    checker: str,
) -> None:
    placeholder = tmp_path / "result.json"
    _write_json(placeholder, {})
    if checker == "wallaware":
        command = [
            sys.executable,
            str(CHECKERS[checker]),
            "--baseline-explore",
            str(placeholder),
            "--wallaware-explore",
            str(placeholder),
            "--wallaware-recall",
            str(placeholder),
        ]
        expected_flag = "--scene-manifests"
    elif checker == "generalized":
        command = [
            sys.executable,
            str(CHECKERS[checker]),
            "--results",
            str(placeholder),
        ]
        expected_flag = "--scene-manifests"
    else:
        command = [
            sys.executable,
            str(CHECKERS[checker]),
            "--result",
            str(placeholder),
        ]
        expected_flag = "--scene-manifest"
    proc = subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    assert expected_flag in proc.stderr


@pytest.mark.parametrize("checker", ("generalized", "wallaware"))
def test_multi_result_checker_rejects_extra_manifest_mapping(
    tmp_path: Path,
    checker: str,
) -> None:
    manifest = _manifest(task_count=1 if checker == "wallaware" else 4)
    result = _canonical_result(manifest)
    manifest_path = tmp_path / "manifest.json"
    extra_path = tmp_path / "extra_manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    _write_json(
        extra_path,
        replace(manifest, scene_id="unused_extra_scene").to_dict(),
    )
    if checker == "generalized":
        result_path = tmp_path / "result.json"
        _write_json(result_path, {"result": result})
        command = [
            sys.executable,
            str(CHECKERS[checker]),
            "--results",
            str(result_path),
            "--scene-manifests",
            str(manifest_path),
            str(extra_path),
        ]
    else:
        baseline_path = tmp_path / "baseline.json"
        wall_path = tmp_path / "wall.json"
        recall_path = tmp_path / "recall.json"
        _write_json(baseline_path, _wall_report(result, "baseline"))
        _write_json(wall_path, _wall_report(result, "wall"))
        _write_json(recall_path, _wall_report(result, "recall"))
        command = [
            sys.executable,
            str(CHECKERS[checker]),
            "--baseline-explore",
            str(baseline_path),
            "--wallaware-explore",
            str(wall_path),
            "--wallaware-recall",
            str(recall_path),
            "--scene-manifests",
            str(manifest_path),
            str(extra_path),
        ]
    proc = subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.stderr == ""
    report = json.loads(proc.stdout)
    assert proc.returncode == 1
    assert report["passed"] is False


@pytest.mark.parametrize("checker", tuple(CHECKERS))
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_checker_rejects_manifest_trace_and_ledger_tampering(
    tmp_path: Path,
    checker: str,
    mutation: str,
) -> None:
    task_count = 1 if checker == "wallaware" else 4
    manifest = _manifest(task_count=task_count)
    result, supplied_manifest = _mutate(
        _canonical_result(manifest), manifest, mutation
    )
    code, report = _run_checker(tmp_path, checker, result, supplied_manifest)
    assert code == 1, (checker, mutation, report)
    assert report["passed"] is False
