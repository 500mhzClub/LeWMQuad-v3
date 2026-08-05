from __future__ import annotations

import ast
import inspect
from pathlib import Path

from lewm.benchmarks.go2_physical_claim_evaluator import evaluate_physical_claim_trace
from lewm.benchmarks.go2_physical_claim_observer import (
    empty_evaluator_access_ledger,
    evaluate_runtime_claim_trace,
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


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="observer_toy",
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
        landmarks=(
            BoxObject(
                "beacon_red",
                "landmark",
                (1.0, 0.0, 0.5),
                (0.2, 0.2, 1.0),
                0.0,
                "landmark_red",
            ),
        ),
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.1),
    )


def test_runtime_observer_is_bit_identical_and_access_ledger_is_zero() -> None:
    manifest = _manifest()
    reference = object_id_reference("beacon_red")
    attempt = build_claim_attempt(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        event_id="event",
        tick=1,
        event_index=0,
        requested_target=reference,
        claimed_target=reference,
        robot_pose_world_xy_yaw=(0.0, 0.0, 0.0),
        pose_provenance="runtime_full_precision",
    )
    trace, ids, commitment = build_claim_trace(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        controller_claim_attempts=[attempt],
    )

    assert evaluate_runtime_claim_trace(trace, manifest, ids, commitment) == (
        evaluate_physical_claim_trace(trace, manifest, ids, commitment)
    )
    assert empty_evaluator_access_ledger() == {
        "evaluator_output_reads_by_controller": 0,
        "evaluator_callbacks_into_controller": 0,
        "evaluator_derived_termination_signals": 0,
    }


def test_runtime_trace_builder_does_not_import_the_evaluator() -> None:
    from lewm.benchmarks import go2_physical_claim_trace

    tree = ast.parse(inspect.getsource(go2_physical_claim_trace))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "lewm.benchmarks.go2_physical_claim_evaluator" not in imports


def test_controller_attempt_does_not_create_physical_credit_when_factor_fails() -> None:
    manifest = _manifest()
    reference = object_id_reference("beacon_red")
    attempt = build_claim_attempt(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        event_id="event",
        tick=1,
        event_index=0,
        requested_target=reference,
        claimed_target=reference,
        robot_pose_world_xy_yaw=(4.0, 0.0, 0.0),
        pose_provenance="runtime_full_precision",
    )
    trace, ids, commitment = build_claim_trace(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        controller_claim_attempts=[attempt],
    )

    evaluated = evaluate_runtime_claim_trace(trace, manifest, ids, commitment)
    assert evaluated["physical_claim_evaluations"][0]["decision"] == "rejected"
    assert evaluated["physical_claim_summary"]["credited_count"] == 0
    assert evaluated["physical_claim_summary"]["all_targets_claimed"] is False


def test_closed_loop_runtime_calls_observer_only_after_controller_loop() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark_go2_memory_closed_loop.py"
    ).read_text(encoding="utf-8")
    assert "lewm.benchmarks.go2_physical_claim_evaluator" not in source
    observer_call = source.index("canonical_physical_claim_trace = evaluate_runtime_claim_trace(")
    controller_end = source.index("final_pos, _ = _current_pose(build)")
    result_write = source.index("result = {", controller_end)
    assert controller_end < observer_call < result_write
