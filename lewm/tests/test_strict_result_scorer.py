"""Tests for standalone strict closed-loop result scoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest

from lewm.benchmarks.strict_result_scorer import (
    SealedEvaluationAuthorizationError,
    score_result_payload,
)
from lewm.benchmarks.go2_physical_claim_evaluator import evaluate_physical_claim_trace
from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    object_id_reference,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    LightingSpec,
    SceneManifest,
    SpawnSpec,
    VisualRandomization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def geometry_contract():
    return load_geometry_contract(
        repository_root=REPOSITORY_ROOT,
        verify_sources=False,
    )


def _box(
    object_id: str,
    *,
    kind: str,
    xy: tuple[float, float],
    size_xy: tuple[float, float],
    material_id: str,
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind=kind,
        center_xyz_m=(xy[0], xy[1], 0.3),
        size_xyz_m=(size_xy[0], size_xy[1], 0.6),
        yaw_rad=0.0,
        material_id=material_id,
    )


def _manifest(
    *,
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    red_xy: tuple[float, float] = (0.8, 0.0),
    walls: tuple[BoxObject, ...] = (),
    split: str | None = "candidate",
) -> SceneManifest:
    landmarks = (
        _box(
            "landmark_red",
            kind="landmark",
            xy=red_xy,
            size_xy=(0.2, 0.2),
            material_id="landmark_red",
        ),
        _box(
            "landmark_green",
            kind="landmark",
            xy=(0.0, 0.8),
            size_xy=(0.2, 0.2),
            material_id="landmark_green",
        ),
        _box(
            "landmark_blue",
            kind="landmark",
            xy=(-0.8, 0.0),
            size_xy=(0.2, 0.2),
            material_id="landmark_blue",
        ),
        _box(
            "landmark_yellow",
            kind="landmark",
            xy=(0.0, -0.8),
            size_xy=(0.2, 0.2),
            material_id="landmark_yellow",
        ),
    )
    return SceneManifest(
        scene_id="strict_score_toy",
        family="score_test",
        difficulty_tier="test",
        topology_seed=101,
        visual_seed=102,
        physics_seed=103,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(spawn_xy[0], spawn_xy[1], 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(GraphNode(0, spawn_xy, 1.0),),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.10),
        split=split,
        walls=walls,
    )


def _claim_row(tick: int, color: str, xy: tuple[float, float]) -> dict:
    return {
        "tick": tick,
        "state": "CLAIM",
        "target_color": color,
        "claim_pose_xy": list(xy),
        "pre_xy": list(xy),
        "post_xy": list(xy),
        "dist_to_target_m": 0.5,
        "stalled": False,
        "hard_stalled": False,
        "body_clearance_contact": False,
    }


def _canonical_trace(manifest: SceneManifest) -> dict:
    trace_id = "strict-trace"
    episode_id = "strict-episode"
    attempts = []
    for index, landmark in enumerate(
        sorted(manifest.landmarks, key=lambda item: item.object_id)
    ):
        target_x = float(landmark.center_xyz_m[0])
        target_y = float(landmark.center_xyz_m[1])
        reference = object_id_reference(landmark.object_id)
        attempts.append(
            build_claim_attempt(
                manifest=manifest,
                trace_id=trace_id,
                episode_id=episode_id,
                event_id=f"event-{index}",
                tick=index + 2,
                event_index=index,
                requested_target=reference,
                claimed_target=reference,
                robot_pose_world_xy_yaw=(
                    0.0,
                    0.0,
                    math.atan2(target_y, target_x),
                ),
                pose_provenance="runtime_full_precision",
            )
        )
    trace, ids, commitment = build_claim_trace(
        manifest=manifest,
        trace_id=trace_id,
        episode_id=episode_id,
        controller_claim_attempts=attempts,
    )
    return evaluate_physical_claim_trace(trace, manifest, ids, commitment)


def _with_canonical_trace(result: dict, manifest: SceneManifest) -> dict:
    result["canonical_physical_claim_trace"] = _canonical_trace(manifest)
    result["runtime_evaluator_access_ledger"] = {
        "evaluator_output_reads_by_controller": 0,
        "evaluator_callbacks_into_controller": 0,
        "evaluator_derived_termination_signals": 0,
    }
    return result


def test_strict_score_reconstructs_completion_and_swept_coverage(
    geometry_contract,
) -> None:
    manifest = _manifest()
    log = [
        {
            "tick": 0,
            "state": "EXPLORE",
            "pre_xy": [0.0, 0.0],
            "post_xy": [0.3, 0.0],
            "stalled": False,
            "hard_stalled": False,
            "body_clearance_contact": False,
        },
        {
            "tick": 1,
            "state": "EXPLORE",
            "pre_xy": [0.3, 0.0],
            "post_xy": [0.3, 0.3],
            "stalled": False,
            "hard_stalled": False,
            "body_clearance_contact": False,
        },
        *(
            _claim_row(tick, color, (0.3, 0.3))
            for tick, color in enumerate(
                ("red", "green", "blue", "yellow"),
                start=2,
            )
        ),
    ]
    claims = [row for row in log if row["state"] == "CLAIM"]
    payload = {
        "schema": "test_result_v1",
        "result": _with_canonical_trace({
            "scene": manifest.scene_id,
            "ticks_used": len(log),
            "target_color": "all",
            "target_colors": ["red", "green", "blue", "yellow"],
            "claimed": True,
            "claimed_colors": ["red", "green", "blue", "yellow"],
            "beacon_claims": claims,
            "final_xy": [0.3, 0.3],
            "success": True,
            "wall_metrics": {
                "body_clearance_contact_events": 0,
                "contact_like_stalls": 0,
                "hard_contact_like_stalls": 0,
            },
        }, manifest),
        "log": log,
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    assert score.trajectory_complete
    assert score.strict_claim_evaluation_complete
    assert score.strict_accepted_claim_event_count == 4
    assert score.strict_four_of_four_complete is True
    assert score.strict_completion_tick == 5
    assert score.coverage_final_fraction is not None
    assert 0.0 < score.coverage_normalized_auc <= score.coverage_final_fraction
    assert (
        score.coverage_unique_swept_cell_count
        > score.coverage_unique_pose_cell_count
    )
    assert score.canonical_geometry_collision_ticks == ()
    assert score.logged_stall_ticks == ()
    assert score.score_complete


def test_proxy_claim_outside_radius_is_rejected(geometry_contract) -> None:
    manifest = _manifest(red_xy=(1.7, 0.0))
    claim = _claim_row(0, "red", (0.0, 0.0))
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "ticks_used": 1,
            "target_color": "red",
            "claimed": True,
            "claimed_colors": ["red"],
            "beacon_claims": [{**claim, "dist_to_target_m": 0.9}],
            "final_xy": [0.0, 0.0],
            "success": True,
            "wall_metrics": {
                "body_clearance_contact_events": 0,
                "contact_like_stalls": 0,
                "hard_contact_like_stalls": 0,
            },
        },
        "log": [claim],
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    verification = score.claim_verifications[0]
    assert verification.true_distance_m == pytest.approx(1.7)
    assert verification.within_claim_radius is False
    assert verification.strict_accepted is None
    assert "outside_claim_radius" in verification.rejection_reasons
    codes = {item.code for item in score.discrepancies}
    assert "proxy_claim_distance_mismatch" in codes
    assert "proxy_claim_unverifiable_at_strict_boundary" in codes
    assert "proxy_claimed_true_without_strict_completion" in codes
    assert "proxy_success_true_without_strict_completion" in codes


def test_manifest_distractor_blocks_strict_claim_los(geometry_contract) -> None:
    base = _manifest()
    distractor = _box(
        "distractor",
        kind="distractor",
        xy=(0.4, 0.0),
        size_xy=(0.1, 0.3),
        material_id="distractor",
    )
    manifest = replace(
        base,
        visual_randomization=VisualRandomization(
            material_overrides=(),
            lighting=LightingSpec(
                direction=(0.0, 0.0, -1.0),
                diffuse_rgb=(0.8, 0.8, 0.8),
                specular_rgb=(0.2, 0.2, 0.2),
                ambient_rgb=(0.2, 0.2, 0.2),
            ),
            distractor_objects=(distractor,),
        ),
    )
    claim = _claim_row(0, "red", (0.0, 0.0))
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "ticks_used": 1,
            "target_color": "red",
            "claimed": True,
            "claimed_colors": ["red"],
            "beacon_claims": [claim],
            "final_xy": [0.0, 0.0],
            "success": True,
        },
        "log": [claim],
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    verification = score.claim_verifications[0]
    assert verification.within_claim_radius is True
    assert verification.line_of_sight is False
    assert verification.strict_accepted is None


def test_current_legacy_claim_pose_comes_from_prior_post(geometry_contract) -> None:
    manifest = _manifest()
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "target_color": "red",
            "ticks_used": 2,
            "claimed": True,
            "success": True,
            "final_xy": [0.2, 0.0],
            "wall_metrics": {"contact_like_stalls": 0},
        },
        "log": [
            {
                "tick": 0,
                "state": "SEEK",
                "post_xy": [0.2, 0.0],
                "stalled": False,
            },
            {"tick": 1, "state": "CLAIM"},
        ],
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    verification = score.claim_verifications[0]
    assert verification.target_object_id == "landmark_red"
    assert verification.pose_xy_m == (0.2, 0.0)
    assert verification.true_distance_m == pytest.approx(0.6)
    assert verification.strict_accepted is None
    assert "legacy_provenance_noncanonical" in verification.rejection_reasons
    assert any(
        "legacy_post_xy_precision" in limitation
        for limitation in score.limitations
    )
    assert "proxy_claimed_target_list_missing" in score.limitations


def test_legacy_claim_at_radius_boundary_is_unverifiable(
    geometry_contract,
) -> None:
    manifest = _manifest(red_xy=(1.4, 0.0))
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "target_color": "red",
            "ticks_used": 2,
            "claimed": True,
            "final_xy": [0.2, 0.0],
            "success": True,
        },
        "log": [
            {
                "tick": 0,
                "state": "SEEK",
                "post_xy": [0.2, 0.0],
                "stalled": False,
            },
            {"tick": 1, "state": "CLAIM"},
        ],
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    assert score.claim_verifications[0].true_distance_m == pytest.approx(1.2)
    assert score.claim_verifications[0].strict_accepted is None
    assert not score.strict_claim_evaluation_complete
    assert any(
        item.code == "proxy_claim_unverifiable_at_strict_boundary"
        for item in score.discrepancies
    )


def test_canonical_collisions_and_logged_stalls_are_separate(
    geometry_contract,
) -> None:
    wall = _box(
        "partition",
        kind="wall",
        xy=(0.0, 0.0),
        size_xy=(0.1, 4.0),
        material_id="wall",
    )
    manifest = _manifest(spawn_xy=(-1.0, 0.0), walls=(wall,))
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "ticks_used": 1,
            "target_color": "all",
            "claimed": False,
            "claimed_colors": [],
            "final_xy": [1.0, 0.0],
            "success": False,
            "wall_metrics": {
                "body_clearance_contact_events": 1,
                "contact_like_stalls": 1,
                "hard_contact_like_stalls": 1,
            },
        },
        "log": [
            {
                "tick": 0,
                "state": "EXPLORE",
                "pre_xy": [-1.0, 0.0],
                "post_xy": [1.0, 0.0],
                "body_clearance_contact": True,
                "stalled": True,
                "hard_stalled": True,
            }
        ],
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    assert score.canonical_geometry_collision_ticks == (0,)
    assert score.canonical_minimum_clearance_m is not None
    assert score.canonical_minimum_clearance_m < 0.0
    assert score.logged_collision_ticks == (0,)
    assert score.logged_stall_ticks == (0,)
    assert score.logged_hard_stall_ticks == (0,)


def test_missing_log_returns_unavailable_metrics_not_proxy_truth(
    geometry_contract,
) -> None:
    manifest = _manifest()
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "ticks_used": 40,
            "target_color": "red",
            "claimed": True,
            "claimed_colors": ["red"],
            "beacon_claims": [
                {
                    "tick": 39,
                    "state": "CLAIM",
                    "target_color": "red",
                    "dist_to_target_m": 0.5,
                }
            ],
            "success": True,
        }
    }

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    assert not score.trajectory_complete
    assert score.coverage_final_fraction is None
    assert score.canonical_geometry_collision_ticks is None
    assert score.claim_verifications[0].strict_accepted is None
    assert not score.score_complete
    assert "per_tick_log_missing" in score.limitations
    assert any(item.code == "proxy_claim_unverifiable" for item in score.discrepancies)


def test_raw_log_only_payload_is_supported(geometry_contract) -> None:
    manifest = _manifest()
    claim = _claim_row(0, "red", (0.0, 0.0))

    score = score_result_payload(
        [claim],
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
    )

    assert score.source_schema == "legacy_log_only"
    assert score.claim_verifications[0].strict_accepted is None
    assert "result_summary_missing_log_only_payload" in score.limitations


def test_sealed_scene_requires_per_call_authorization(geometry_contract) -> None:
    manifest = _manifest(split="sealed_test")
    payload = {
        "result": {
            "scene": manifest.scene_id,
            "ticks_used": 1,
            "target_color": "all",
            "claimed": False,
            "claimed_colors": [],
            "final_xy": [0.0, 0.0],
            "success": False,
        },
        "log": [
            {
                "tick": 0,
                "state": "EXPLORE",
                "pre_xy": [0.0, 0.0],
                "post_xy": [0.0, 0.0],
                "stalled": False,
                "hard_stalled": False,
                "body_clearance_contact": False,
            }
        ],
    }

    with pytest.raises(SealedEvaluationAuthorizationError):
        score_result_payload(
            payload,
            scene_manifest=manifest,
            geometry_contract=geometry_contract,
        )

    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=geometry_contract,
        authorize_sealed_final_evaluation=True,
    )
    assert score.sealed_final_evaluation_authorized


def test_sealed_benchmark_payload_also_requires_authorization(
    geometry_contract,
) -> None:
    manifest = _manifest(split="candidate")
    sealed = {
        "schema": "lewm_navigation_sealed_test_manifest_v0",
        "scenes": [{"scene_id": manifest.scene_id}],
    }

    with pytest.raises(SealedEvaluationAuthorizationError):
        score_result_payload(
            [],
            scene_manifest=manifest,
            geometry_contract=geometry_contract,
            benchmark_manifest=sealed,
        )


def test_result_scene_must_match_exact_manifest(geometry_contract) -> None:
    with pytest.raises(ValueError, match="result/manifest scene mismatch"):
        score_result_payload(
            {"result": {"scene": "wrong"}, "log": []},
            scene_manifest=_manifest(),
            geometry_contract=geometry_contract,
        )


def test_canonical_trace_recomputation_rejects_every_stored_join_mutation(
    geometry_contract,
) -> None:
    manifest = _manifest()
    base_result = _with_canonical_trace(
        {
            "scene": manifest.scene_id,
            "ticks_used": 1,
            "target_color": "all",
            "target_colors": ["red", "green", "blue", "yellow"],
            "claimed": True,
            "claimed_colors": ["red", "green", "blue", "yellow"],
            "success": True,
        },
        manifest,
    )
    mutations = []

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["physical_claim_evaluations"].pop()
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["physical_claim_evaluations"].append(
        deepcopy(value["canonical_physical_claim_trace"]["physical_claim_evaluations"][0])
    )
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["physical_claim_evaluations"].reverse()
    mutations.append(value)

    for path in (
        ("physical_claim_evaluations", 0, "decision"),
        ("physical_claim_evaluations", 0, "content_sha256"),
        ("physical_claim_summary", "content_sha256"),
        ("trace_content_sha256",),
    ):
        value = deepcopy(base_result)
        cursor = value["canonical_physical_claim_trace"]
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = "0" * 64
        mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["controller_claim_attempts"].reverse()
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["evaluator_feedback_to_controller"] = [
        "forbidden"
    ]
    mutations.append(value)

    value = deepcopy(base_result)
    value["runtime_evaluator_access_ledger"][
        "evaluator_output_reads_by_controller"
    ] = 1
    mutations.append(value)

    value = deepcopy(base_result)
    value["runtime_evaluator_access_ledger"][
        "evaluator_output_reads_by_controller"
    ] = False
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["physical_claim_evaluations"][0][
        "factors"
    ]["identity_passes"] = 1
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["physical_claim_summary"][
        "credited_count"
    ] = 4.0
    mutations.append(value)

    value = deepcopy(base_result)
    value["canonical_physical_claim_trace"]["controller_claim_attempts"][0][
        "tick"
    ] = 1.0
    mutations.append(value)

    for result in mutations:
        score = score_result_payload(
            {"result": result, "log": []},
            scene_manifest=manifest,
            geometry_contract=geometry_contract,
        )
        assert score.canonical_physical_claim_trace_present
        assert not score.canonical_physical_claim_trace_verified
        assert score.strict_accepted_claim_event_count == 0
        assert score.strict_all_targets_complete is False
        assert not score.score_complete
