from __future__ import annotations

from copy import deepcopy
import math

from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_result import (
    canonical_physical_claim_status,
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
        scene_id="physical_status_toy",
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


def _result() -> tuple[SceneManifest, dict]:
    manifest = _manifest()
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
                        float(landmark.center_xyz_m[1]),
                        float(landmark.center_xyz_m[0]),
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
    return manifest, {
        "claimed": True,
        "claimed_colors": ["blue", "green", "red", "yellow"],
        "success": True,
        "canonical_physical_claim_trace": evaluated,
        "runtime_evaluator_access_ledger": {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    }


def test_result_status_accepts_only_manifest_recomputed_physical_summary() -> None:
    manifest, result = _result()
    status = canonical_physical_claim_status(
        result,
        scene_manifest=manifest,
        required_task_count=4,
    )
    assert status.valid
    assert status.all_targets_claimed


def test_result_status_rejects_proxy_geometry_hash_and_numeric_type_mutations() -> None:
    manifest, base = _result()
    proxy = {"success": True, "claimed_colors": ["red", "green", "blue", "yellow"]}
    assert not canonical_physical_claim_status(
        proxy, scene_manifest=manifest, required_task_count=4
    ).valid

    mutations = []
    reordered = deepcopy(base)
    reordered["canonical_physical_claim_trace"]["physical_claim_evaluations"].reverse()
    mutations.append(reordered)
    ledger = deepcopy(base)
    ledger["runtime_evaluator_access_ledger"][
        "evaluator_callbacks_into_controller"
    ] = 1
    mutations.append(ledger)
    bool_ledger = deepcopy(base)
    bool_ledger["runtime_evaluator_access_ledger"][
        "evaluator_callbacks_into_controller"
    ] = False
    mutations.append(bool_ledger)
    float_count = deepcopy(base)
    float_count["canonical_physical_claim_trace"]["physical_claim_summary"][
        "credited_count"
    ] = 4.0
    mutations.append(float_count)
    bool_factor = deepcopy(base)
    bool_factor["canonical_physical_claim_trace"]["physical_claim_evaluations"][0][
        "factors"
    ]["identity_passes"] = 1
    mutations.append(bool_factor)
    wrong_hash = deepcopy(base)
    wrong_hash["canonical_physical_claim_trace"]["trace_content_sha256"] = "0" * 64
    mutations.append(wrong_hash)
    wrong_manifest = deepcopy(base)
    wrong_manifest["canonical_physical_claim_trace"]["physical_manifest_sha256"] = (
        "0" * 64
    )
    mutations.append(wrong_manifest)
    for result in mutations:
        assert not canonical_physical_claim_status(
            result,
            scene_manifest=manifest,
            required_task_count=4,
        ).valid
