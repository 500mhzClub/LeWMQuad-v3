from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

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
SPEC = importlib.util.spec_from_file_location(
    "check_go2_generalized_suite",
    ROOT / "scripts/check_go2_generalized_suite.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> SceneManifest:
    landmarks = tuple(
        BoxObject(
            object_id=f"beacon_{color}",
            kind="landmark",
            center_xyz_m=(x, y, 0.5),
            size_xyz_m=(0.2, 0.2, 1.0),
            yaw_rad=0.0,
            material_id=f"landmark_{color}",
        )
        for color, x, y in (
            ("red", 0.8, 0.0),
            ("green", 0.0, 0.8),
            ("blue", -0.8, 0.0),
            ("yellow", 0.0, -0.8),
        )
    )
    return SceneManifest(
        scene_id="checker_scene",
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


def _result(manifest: SceneManifest) -> dict:
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
    return {
        "scene": manifest.scene_id,
        "claimed": True,
        "success": True,
        "claimed_colors": ["blue", "green", "red", "yellow"],
        "canonical_physical_claim_trace": evaluated,
        "runtime_evaluator_access_ledger": {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    }


def _write(tmp_path: Path, result: dict) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"result": result}), encoding="utf-8")
    return path


def test_suite_row_recomputes_canonical_physical_trace(tmp_path: Path) -> None:
    manifest = _manifest()
    row = MODULE._result_row(_write(tmp_path, _result(manifest)), manifest)
    assert row["canonical_physical_claims"] is True
    assert row["scene_manifest_match"] is True
    assert row["all_beacons_claimed"] is True
    assert row["success"] is True


def test_suite_row_rejects_proxy_only_success(tmp_path: Path) -> None:
    manifest = _manifest()
    result = {
        "scene": manifest.scene_id,
        "success": True,
        "claimed_colors": ["red", "green", "blue", "yellow"],
    }
    row = MODULE._result_row(_write(tmp_path, result), manifest)
    assert row["canonical_physical_claims"] is False
    assert row["all_beacons_claimed"] is False
    assert row["success"] is False


def test_suite_row_rejects_missing_or_wrong_scene_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    path = _write(tmp_path, _result(manifest))
    row = MODULE._result_row(path, None)
    assert row["canonical_physical_claims"] is False
    assert row["scene_manifest_match"] is False
