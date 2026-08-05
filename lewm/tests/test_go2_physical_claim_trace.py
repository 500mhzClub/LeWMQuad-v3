from __future__ import annotations

import hashlib
import struct

import pytest

from lewm.benchmarks.go2_physical_claim_evaluator import evaluate_physical_claim_trace
from lewm.benchmarks.generalization_protocol import (
    evaluate_physical_claim_trace_protocol_adapter,
)
from lewm.benchmarks.go2_oracle_positive_control import (
    evaluate_physical_claim_trace_oracle_adapter,
)
from lewm.benchmarks.go2_physical_eligibility import (
    evaluate_physical_claim_trace_eligibility_adapter,
)
from lewm.benchmarks.go2_physical_claim_observer import (
    evaluate_runtime_claim_trace,
)
from lewm.benchmarks.strict_result_scorer import (
    recompute_canonical_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    canonical_task_object_ids,
    object_id_reference,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


def _manifest() -> SceneManifest:
    target = BoxObject(
        object_id="beacon_red",
        kind="landmark",
        center_xyz_m=(1.0, 0.0, 0.5),
        size_xyz_m=(0.2, 0.2, 1.0),
        yaw_rad=0.0,
        material_id="landmark_red",
    )
    return SceneManifest(
        scene_id="trace_toy",
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
        landmarks=(target,),
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.1),
    )


def test_builder_emits_exact_pose_commitment_and_evaluator_input() -> None:
    manifest = _manifest()
    pose = (-0.0, 0.0, 0.0)
    reference = object_id_reference("beacon_red")
    event = build_claim_attempt(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        event_id="event-0",
        tick=7,
        event_index=0,
        requested_target=reference,
        claimed_target=reference,
        robot_pose_world_xy_yaw=pose,
        pose_provenance="runtime_full_precision",
    )
    trace, task_ids, task_hash = build_claim_trace(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        controller_claim_attempts=[event],
    )

    assert event["pose_hex"] == [value.hex() for value in pose]
    assert event["pose_binary64_le_sha256"] == hashlib.sha256(
        struct.pack("<3d", *pose)
    ).hexdigest()
    evaluated = evaluate_physical_claim_trace(trace, manifest, task_ids, task_hash)
    assert evaluated["physical_claim_summary"]["all_targets_claimed"] is True


def test_builder_rejects_nonmanifest_duplicate_and_nonfinite_inputs() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="exact manifest"):
        canonical_task_object_ids(manifest, ["missing"])
    with pytest.raises(ValueError, match="unique"):
        canonical_task_object_ids(manifest, ["beacon_red", "beacon_red"])
    with pytest.raises(ValueError, match="UTF-8"):
        canonical_task_object_ids(manifest, ["\ud800"])
    with pytest.raises(ValueError, match="finite"):
        build_claim_attempt(
            manifest=manifest,
            trace_id="trace",
            episode_id="episode",
            event_id="event",
            tick=0,
            event_index=0,
            requested_target=object_id_reference("beacon_red"),
            claimed_target=object_id_reference("beacon_red"),
            robot_pose_world_xy_yaw=(float("nan"), 0.0, 0.0),
            pose_provenance="runtime_full_precision",
        )
    with pytest.raises(ValueError, match="exact JSON numbers"):
        build_claim_attempt(
            manifest=manifest,
            trace_id="trace",
            episode_id="episode",
            event_id="event",
            tick=0,
            event_index=0,
            requested_target=object_id_reference("beacon_red"),
            claimed_target=object_id_reference("beacon_red"),
            robot_pose_world_xy_yaw=(True, 0.0, 0.0),
            pose_provenance="runtime_full_precision",
        )


def test_all_production_adapters_are_bit_identical_for_one_synthetic_trace() -> None:
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
    direct = evaluate_physical_claim_trace(trace, manifest, ids, commitment)
    adapters = (
        evaluate_physical_claim_trace_protocol_adapter(
            trace, manifest, ids, commitment
        ),
        evaluate_physical_claim_trace_oracle_adapter(
            trace, manifest, ids, commitment
        ),
        evaluate_physical_claim_trace_eligibility_adapter(
            trace, manifest, ids, commitment
        ),
        evaluate_runtime_claim_trace(trace, manifest, ids, commitment),
        recompute_canonical_physical_claim_trace(
            direct,
            scene_manifest=manifest,
            expected_task_object_ids=ids,
            expected_task_object_set_sha256=commitment,
        ),
    )
    assert all(item == direct for item in adapters)
