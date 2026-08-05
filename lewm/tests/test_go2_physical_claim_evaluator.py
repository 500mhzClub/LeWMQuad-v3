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

import pytest

from lewm.benchmarks import go2_physical_claim_evaluator as claims
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    LightingSpec,
    SceneManifest,
    SpawnSpec,
    VisualRandomization,
    manifest_sha256,
)


# Gates 15-20 are exercised by the dedicated trace, observer, strict-scorer,
# eligibility, oracle, and result-checker suites. This file remains focused on
# the pure evaluator boundary.
_PENDING_INTEGRATION_GATES: tuple[int, ...] = ()


class _SerializedItemsDict(dict):
    """Expose one JSON view while retaining a different inherited mapping."""

    def __init__(self, semantic: dict, serialized: dict) -> None:
        super().__init__(semantic)
        self._serialized = serialized

    def items(self):
        return self._serialized.items()


class _FirstThenList(list):
    """Serialize one sequence, then expose another on later iteration."""

    def __init__(self, first: list, later: list) -> None:
        super().__init__(later)
        self._first = first
        self._later = later
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        values = self._first if self.iteration_count == 1 else self._later
        return iter(values)


class _AlwaysEqualString(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _ManifestAliasString(_AlwaysEqualString):
    def __hash__(self) -> int:
        return hash("beacon_red")


class _CoercingFloat(float):
    def __float__(self) -> float:
        return 1.0


class _CoercingMaterial(str):
    def casefold(self) -> str:
        return "landmark_red"


class _SplitSerializationManifest(SceneManifest):
    def to_dict(self) -> dict:
        semantic = super().to_dict()
        serialized = deepcopy(semantic)
        serialized["landmarks"][0]["center_xyz_m"] = (100.0, 0.0, 0.5)
        return _SerializedItemsDict(semantic, serialized)


def _assert_exact_plain_json_builtins(value: object) -> None:
    if isinstance(value, dict):
        assert type(value) is dict
        for key, item in value.items():
            assert type(key) is str
            _assert_exact_plain_json_builtins(item)
    elif isinstance(value, list):
        assert type(value) is list
        for item in value:
            _assert_exact_plain_json_builtins(item)
    else:
        assert type(value) in {str, int, float, bool, type(None)}


def _box(
    object_id: str,
    x: float,
    y: float,
    *,
    size_x: float = 0.2,
    size_y: float = 0.2,
    yaw: float = 0.0,
    material_id: str = "wall",
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind="box",
        center_xyz_m=(x, y, 0.5),
        size_xyz_m=(size_x, size_y, 1.0),
        yaw_rad=yaw,
        material_id=material_id,
    )


def _manifest(
    *,
    landmarks: tuple[BoxObject, ...] | None = None,
    walls: tuple[BoxObject, ...] = (),
    obstacles: tuple[BoxObject, ...] = (),
    distractors: tuple[BoxObject, ...] = (),
    scene_id: str = "scene",
) -> SceneManifest:
    if landmarks is None:
        landmarks = (
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
        )
    visual = None
    if distractors:
        visual = VisualRandomization(
            material_overrides=(),
            lighting=LightingSpec(
                direction=(0.0, 0.0, -1.0),
                diffuse_rgb=(0.8, 0.8, 0.8),
                specular_rgb=(0.2, 0.2, 0.2),
                ambient_rgb=(0.2, 0.2, 0.2),
            ),
            distractor_objects=distractors,
        )
    return SceneManifest(
        scene_id=scene_id,
        family="unit_test",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=obstacles,
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=100.0,
            min_camera_clearance_m=0.1,
        ),
        split="train",
        walls=walls,
        visual_randomization=visual,
    )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_hash(manifest: SceneManifest, task_ids: list[str]) -> str:
    return _canonical_hash(
        {
            "schema": "lewm_go2_claim_task_set_v1",
            "scene_id": manifest.scene_id,
            "physical_manifest_sha256": manifest_sha256(manifest),
            "task_object_ids": task_ids,
        }
    )


def _reference(object_id: str) -> dict[str, str]:
    return {"namespace": "object_id", "value": object_id}


def _event(
    manifest: SceneManifest,
    *,
    event_id: str = "event-0",
    event_index: int = 0,
    tick: int = 10,
    requested: dict[str, str] | None = None,
    claimed: dict[str, str] | None = None,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    provenance: str = "runtime_full_precision",
) -> dict:
    if requested is None:
        requested = _reference("beacon_red")
    if claimed is None:
        claimed = _reference("beacon_red")
    pose_values = list(pose)
    return {
        "trace_id": "trace",
        "episode_id": "episode",
        "scene_id": manifest.scene_id,
        "event_id": event_id,
        "tick": tick,
        "event_index": event_index,
        "requested_target": requested,
        "claimed_target": claimed,
        "robot_pose_world_xy_yaw": pose_values,
        "pose_binary64_le_sha256": hashlib.sha256(
            struct.pack("<3d", *pose_values)
        ).hexdigest(),
        "pose_hex": [value.hex() for value in pose_values],
        "pose_provenance": provenance,
        "physical_manifest_sha256": manifest_sha256(manifest),
    }


def _trace(
    manifest: SceneManifest,
    events: list[object],
    *,
    task_ids: list[str] | None = None,
) -> tuple[dict, str]:
    if task_ids is None:
        task_ids = sorted(
            [landmark.object_id for landmark in manifest.landmarks],
            key=lambda value: value.encode("utf-8"),
        )
    task_hash = _task_hash(manifest, task_ids)
    return (
        {
            "schema": "lewm_go2_claim_trace_v1",
            "trace_id": "trace",
            "episode_id": "episode",
            "scene_id": manifest.scene_id,
            "physical_manifest_sha256": manifest_sha256(manifest),
            "task_object_ids": task_ids,
            "task_object_set_sha256": task_hash,
            "controller_claim_attempts": events,
            "evaluator_feedback_to_controller": [],
        },
        task_hash,
    )


def _evaluate(
    manifest: SceneManifest,
    events: list[object],
    *,
    task_ids: list[str] | None = None,
) -> dict:
    trace, task_hash = _trace(manifest, events, task_ids=task_ids)
    return claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        trace["task_object_ids"],
        task_hash,
    )


def _only(result: dict) -> dict:
    return result["physical_claim_evaluations"][0]


def test_accepts_exact_identity_distance_los_and_bearing_with_canonical_hashes() -> None:
    manifest = _manifest()
    result = _evaluate(manifest, [_event(manifest)])
    evaluation = _only(result)

    assert evaluation["decision"] == "accepted"
    assert evaluation["factors"] == {
        "identity_passes": True,
        "distance_passes": True,
        "line_of_sight_passes": True,
        "bearing_passes": True,
    }
    assert evaluation["credited"] is True
    assert evaluation["physically_verified"] is True
    assert evaluation["physical_manifest_sha256"] == manifest_sha256(manifest)
    assert evaluation["physical_manifest_sha256"] == _canonical_hash(
        manifest.to_dict()
    )
    assert result["physical_claim_summary"]["all_targets_claimed"] is True
    assert evaluation["content_sha256"] == _canonical_hash(
        {key: value for key, value in evaluation.items() if key != "content_sha256"}
    )
    summary = result["physical_claim_summary"]
    assert summary["content_sha256"] == _canonical_hash(
        {key: value for key, value in summary.items() if key != "content_sha256"}
    )
    assert result["trace_content_sha256"] == _canonical_hash(
        {
            key: value
            for key, value in result.items()
            if key != "trace_content_sha256"
        }
    )


def test_exact_output_schemas_contract_hash_and_physical_literals() -> None:
    manifest = _manifest()
    result = _evaluate(manifest, [_event(manifest)])
    evaluation = _only(result)
    summary = result["physical_claim_summary"]

    assert set(result) == {
        "schema",
        "trace_id",
        "episode_id",
        "scene_id",
        "physical_manifest_sha256",
        "task_object_ids",
        "task_object_set_sha256",
        "controller_claim_attempts",
        "evaluator_feedback_to_controller",
        "physical_claim_evaluations",
        "physical_claim_summary",
        "trace_content_sha256",
    }
    assert set(summary) == {
        "schema",
        "evaluator_contract_sha256",
        "trace_id",
        "episode_id",
        "scene_id",
        "physical_manifest_sha256",
        "task_object_ids",
        "task_object_set_sha256",
        "attempted_count",
        "accepted_count",
        "rejected_count",
        "unverifiable_count",
        "credited_count",
        "duplicate_physical_claim_not_credited_count",
        "unverifiable_reason_counts",
        "rejection_reason_counts",
        "aggregate_reason_counts",
        "trace_unverifiable_reasons",
        "credited_object_ids",
        "first_credited_by_object",
        "event_content_sha256s",
        "all_targets_claimed",
        "content_sha256",
    }
    assert set(evaluation) == {
        "schema",
        "evaluator_contract_sha256",
        "trace_id",
        "episode_id",
        "scene_id",
        "physical_manifest_sha256",
        "task_object_ids",
        "task_object_set_sha256",
        "event_id",
        "tick",
        "event_index",
        "pose_provenance",
        "requested_target",
        "claimed_target",
        "requested_resolution",
        "claimed_resolution",
        "requested_in_task_set",
        "claimed_in_task_set",
        "robot_pose_world_xy_yaw",
        "pose_hex",
        "pose_binary64_le_sha256",
        "claimed_target_object_id",
        "claimed_target_center_xyz_m",
        "claimed_target_center_hex",
        "physical_contract",
        "distance_m",
        "distance_hex",
        "target_world_bearing_rad",
        "target_world_bearing_hex",
        "signed_bearing_error_rad",
        "signed_bearing_error_hex",
        "absolute_bearing_error_rad",
        "absolute_bearing_error_hex",
        "physical_blockers",
        "factors",
        "decision",
        "accepted",
        "physically_verified",
        "unverifiable_reasons",
        "rejection_reasons",
        "credited",
        "duplicate_physical_claim_not_credited",
        "content_sha256",
    }
    assert result["schema"] == "lewm_go2_evaluated_claim_trace_v1"
    assert summary["schema"] == "lewm_go2_physical_claim_summary_v1"
    assert evaluation["schema"] == "lewm_go2_physical_claim_evaluation_v1"
    assert summary["evaluator_contract_sha256"] == claims.EVALUATOR_CONTRACT_SHA256
    assert evaluation["evaluator_contract_sha256"] == claims.EVALUATOR_CONTRACT_SHA256
    assert evaluation["physical_contract"] == {
        "claim_distance_m": 1.20,
        "claim_absolute_bearing_rad": 0.25,
        "line_of_sight_inflation_m": 0.0,
        "line_of_sight_geometry": (
            "closed_segment_oriented_rectangles_scalar_binary64_x_then_y"
        ),
    }


def test_reason_code_and_precedence_order_is_exact_binding() -> None:
    assert claims.UNVERIFIABLE_REASONS == (
        "trace_schema_or_key_set_invalid",
        "trace_id_missing_or_invalid",
        "episode_id_missing_or_invalid",
        "scene_manifest_identity_mismatch",
        "physical_manifest_commitment_mismatch",
        "task_object_ids_not_exact_sorted_unique",
        "task_object_set_mismatch",
        "task_object_commitment_mismatch",
        "evaluator_feedback_to_controller_nonempty",
        "trace_event_order_invalid",
        "manifest_duplicate_object_id",
        "manifest_invalid_physical_geometry",
        "event_key_set_or_type_invalid",
        "event_trace_identity_mismatch",
        "event_id_missing_or_duplicate",
        "claim_tick_or_index_invalid",
        "requested_reference_malformed",
        "requested_namespace_forbidden_for_provenance",
        "requested_identity_unresolved",
        "requested_identity_ambiguous",
        "claimed_reference_malformed",
        "claimed_namespace_forbidden_for_provenance",
        "claimed_identity_unresolved",
        "claimed_identity_ambiguous",
        "pose_provenance_invalid",
        "claim_pose_missing_or_nonfinite",
        "claim_pose_precision_commitment_mismatch",
        "physical_computation_nonfinite",
        "legacy_provenance_noncanonical",
        "legacy_pose_missing_yaw",
        "legacy_pose_rounded_or_inferred",
    )
    assert claims.REJECTION_REASONS == (
        "requested_identity_not_in_task_set",
        "claimed_identity_not_in_task_set",
        "requested_claimed_identity_mismatch",
        "outside_inclusive_claim_distance",
        "zero_inflation_physical_los_blocked",
        "outside_inclusive_claim_bearing",
    )


def test_distance_boundary_is_inclusive_and_one_ulp_outside_rejects() -> None:
    target = _box("beacon_red", 0.0, 0.0, material_id="landmark_red")
    manifest = _manifest(landmarks=(target,))
    boundary = _only(
        _evaluate(manifest, [_event(manifest, pose=(-1.2, 0.0, 0.0))])
    )
    outside_distance = math.nextafter(1.2, math.inf)
    outside = _only(
        _evaluate(
            manifest,
            [_event(manifest, pose=(-outside_distance, 0.0, 0.0))],
        )
    )

    assert boundary["distance_m"] == 1.2
    assert boundary["factors"]["distance_passes"] is True
    assert outside["distance_m"] == outside_distance
    assert outside["factors"]["distance_passes"] is False
    assert outside["rejection_reasons"] == [
        "outside_inclusive_claim_distance"
    ]


@pytest.mark.parametrize("error", [0.25, -0.25])
def test_bearing_boundaries_are_inclusive(error: float) -> None:
    manifest = _manifest()
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(0.0, 0.0, -error))])
    )
    assert evaluation["absolute_bearing_error_rad"] == 0.25
    assert evaluation["factors"]["bearing_passes"] is True


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_ulp_outside_bearing_rejects(sign: float) -> None:
    manifest = _manifest()
    error = math.nextafter(0.25, math.inf) * sign
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(0.0, 0.0, -error))])
    )
    assert evaluation["factors"]["bearing_passes"] is False
    assert evaluation["rejection_reasons"] == [
        "outside_inclusive_claim_bearing"
    ]


def test_bearing_wraps_across_minus_pi_plus_pi() -> None:
    target = _box("beacon_red", -1.0, -0.01, material_id="landmark_red")
    manifest = _manifest(landmarks=(target,))
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(0.0, 0.0, math.pi - 0.01))])
    )
    assert abs(evaluation["signed_bearing_error_rad"]) < 0.03
    assert evaluation["factors"]["bearing_passes"] is True


def test_arbitrary_finite_yaw_uses_the_frozen_wrapped_formula() -> None:
    yaw = 123.456789
    target = _box(
        "beacon_red",
        math.cos(0.37),
        math.sin(0.37),
        material_id="landmark_red",
    )
    manifest = _manifest(landmarks=(target,))
    evaluation = _only(_evaluate(manifest, [_event(manifest, pose=(0.0, 0.0, yaw))]))
    expected = math.atan2(math.sin(0.37 - yaw), math.cos(0.37 - yaw))
    assert evaluation["signed_bearing_error_rad"] == expected
    assert evaluation["absolute_bearing_error_rad"] == abs(expected)


@pytest.mark.parametrize(
    ("collection", "kwargs", "expected_collection"),
    (
        ("wall", {"walls": (_box("wall", 0.5, 0.0),)}, "walls"),
        (
            "obstacle",
            {"obstacles": (_box("obstacle", 0.5, 0.0),)},
            "obstacles",
        ),
        (
            "rotated",
            {
                "obstacles": (
                    _box("rotated", 0.5, 0.0, size_x=0.5, yaw=0.7),
                )
            },
            "obstacles",
        ),
        (
            "distractor",
            {"distractors": (_box("distractor", 0.5, 0.0),)},
            "visual_randomization.distractor_objects",
        ),
    ),
)
def test_every_physical_occluder_collection_blocks(
    collection: str, kwargs: dict, expected_collection: str
) -> None:
    del collection
    manifest = _manifest(**kwargs)
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["factors"]["line_of_sight_passes"] is False
    assert evaluation["physical_blockers"][0][0] == expected_collection
    assert evaluation["rejection_reasons"] == [
        "zero_inflation_physical_los_blocked"
    ]


def test_other_beacon_blocks_but_exact_claimed_target_is_excluded() -> None:
    red = _box("beacon_red", 1.0, 0.0, material_id="landmark_red")
    green = _box("beacon_green", 0.5, 0.0, material_id="landmark_green")
    blocked_manifest = _manifest(landmarks=(red, green))
    blocked = _only(
        _evaluate(blocked_manifest, [_event(blocked_manifest)])
    )
    clear_manifest = _manifest(landmarks=(red,))
    clear = _only(_evaluate(clear_manifest, [_event(clear_manifest)]))

    assert blocked["physical_blockers"] == [["landmarks", "beacon_green"]]
    assert blocked["factors"]["line_of_sight_passes"] is False
    assert clear["physical_blockers"] == []
    assert clear["factors"]["line_of_sight_passes"] is True


def test_closed_segment_tangency_and_endpoint_inside_block() -> None:
    tangent = _box("tangent", 0.5, 0.1, size_x=0.2, size_y=0.2)
    endpoint = _box("endpoint", 1.0, 0.0, size_x=0.1, size_y=0.1)
    manifest = _manifest(obstacles=(tangent, endpoint))
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["physical_blockers"] == [
        ["obstacles", "endpoint"],
        ["obstacles", "tangent"],
    ]


def test_source_inside_blocker_is_closed_segment_blocking() -> None:
    source = _box("source", 0.0, 0.0, size_x=0.1, size_y=0.1)
    manifest = _manifest(obstacles=(source,))
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["physical_blockers"] == [["obstacles", "source"]]
    assert evaluation["factors"]["line_of_sight_passes"] is False


def test_narrow_uninflated_gap_remains_clear() -> None:
    upper = _box("upper", 0.5, 0.11, size_x=0.4, size_y=0.2)
    lower = _box("lower", 0.5, -0.11, size_x=0.4, size_y=0.2)
    manifest = _manifest(obstacles=(upper, lower))
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["physical_blockers"] == []
    assert evaluation["factors"]["line_of_sight_passes"] is True


def test_zero_length_segment_is_still_tested_against_blockers() -> None:
    target = _box("beacon_red", 0.0, 0.0, material_id="landmark_red")
    blocker = _box("blocker", 0.0, 0.0)
    manifest = _manifest(landmarks=(target,), obstacles=(blocker,))
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["distance_m"] == 0.0
    assert evaluation["physical_blockers"] == [["obstacles", "blocker"]]


def test_zero_distance_still_uses_exact_atan2_bearing_contract() -> None:
    target = _box("beacon_red", 0.0, 0.0, material_id="landmark_red")
    manifest = _manifest(landmarks=(target,))
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(0.0, 0.0, math.pi))])
    )
    assert evaluation["distance_m"] == 0.0
    assert evaluation["target_world_bearing_rad"] == 0.0
    assert evaluation["absolute_bearing_error_rad"] == math.pi
    assert evaluation["factors"] == {
        "identity_passes": True,
        "distance_passes": True,
        "line_of_sight_passes": True,
        "bearing_passes": False,
    }
    assert evaluation["rejection_reasons"] == [
        "outside_inclusive_claim_bearing"
    ]


def test_scalar_inverse_yaw_and_x_then_y_slab_goldens() -> None:
    base = 1e15
    large = _box("large", base, base, size_x=0.125, size_y=0.125, yaw=0.1)
    assert claims._segment_intersects_box(
        (base - 1.0, base - 1.0),
        (base + 0.25, base + 0.125),
        large,
    ) == (True, True)

    positive_yaw = _box("positive", 0.1, -0.2, size_x=0.8, size_y=0.2, yaw=0.7)
    negative_yaw = replace(positive_yaw, object_id="negative", yaw_rad=-0.7)
    segment = ((-1.5, -1.5), (-0.2, -0.4))
    assert claims._segment_intersects_box(*segment, positive_yaw) == (True, True)
    assert claims._segment_intersects_box(*segment, negative_yaw) == (False, True)

    subnormal = math.ulp(0.0)
    axis_aligned = _box("axis", 0.0, 0.0, size_x=1.0, size_y=2.0)
    assert claims._segment_intersects_box(
        (2.0, 0.0), (3.0, subnormal), axis_aligned
    ) == (False, True)
    assert claims._segment_intersects_box(
        (-1.0, 0.0), (1.0, subnormal), axis_aligned
    ) == (None, False)


def test_slab_intermediate_overflow_is_publicly_unverifiable() -> None:
    subnormal = math.ulp(0.0)
    target = _box(
        "beacon_red",
        1.0,
        subnormal,
        material_id="landmark_red",
    )
    blocker = _box("blocker", 0.0, 0.0, size_x=1.0, size_y=2.0)
    manifest = _manifest(landmarks=(target,), obstacles=(blocker,))
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(-1.0, 0.0, 0.0))])
    )
    assert evaluation["unverifiable_reasons"] == ["physical_computation_nonfinite"]
    assert evaluation["factors"]["line_of_sight_passes"] is None
    assert evaluation["decision"] == "unverifiable"
    assert evaluation["rejection_reasons"] == []


def test_duplicate_event_id_marks_every_occurrence_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    events = [
        _event(manifest, event_id="duplicate", event_index=0, tick=1),
        _event(manifest, event_id="duplicate", event_index=1, tick=1),
    ]
    hashed_events: list[dict] = []
    content_sha256 = claims._content_sha256

    def recording_content_sha256(value: dict) -> str:
        if value.get("schema") == claims.EVENT_SCHEMA:
            hashed_events.append(deepcopy(value))
        return content_sha256(value)

    monkeypatch.setattr(claims, "_content_sha256", recording_content_sha256)
    result = _evaluate(manifest, events)
    for evaluation in result["physical_claim_evaluations"]:
        assert evaluation["decision"] == "unverifiable"
        assert "event_id_missing_or_duplicate" in evaluation["unverifiable_reasons"]
        assert evaluation["accepted"] is False
        assert evaluation["credited"] is False
        assert evaluation["content_sha256"] == _canonical_hash(
            {
                key: value
                for key, value in evaluation.items()
                if key != "content_sha256"
            }
        )
    assert len(hashed_events) == 2
    assert all(event["decision"] == "unverifiable" for event in hashed_events)
    assert all(event["accepted"] is False for event in hashed_events)
    assert all(event["credited"] is False for event in hashed_events)
    assert all(
        "event_id_missing_or_duplicate" in event["unverifiable_reasons"]
        for event in hashed_events
    )
    assert result["physical_claim_summary"]["credited_count"] == 0


def test_only_first_accepted_event_for_object_receives_credit() -> None:
    manifest = _manifest()
    events = [
        _event(manifest, event_id="first", event_index=0, tick=1),
        _event(manifest, event_id="second", event_index=1, tick=2),
    ]
    evaluations = _evaluate(manifest, events)["physical_claim_evaluations"]
    assert [event["decision"] for event in evaluations] == ["accepted", "accepted"]
    assert [event["credited"] for event in evaluations] == [True, False]
    assert [event["duplicate_physical_claim_not_credited"] for event in evaluations] == [
        False,
        True,
    ]


def test_same_tick_credit_is_unique_per_object_and_all_target_exact() -> None:
    red = _box("beacon_red", 1.0, 0.0, material_id="landmark_red")
    green = _box("beacon_green", 0.0, 1.0, material_id="landmark_green")
    manifest = _manifest(landmarks=(red, green))
    events = [
        _event(manifest, event_id="red-first", event_index=0, tick=7),
        _event(manifest, event_id="red-again", event_index=1, tick=7),
        _event(
            manifest,
            event_id="green-first",
            event_index=2,
            tick=7,
            requested=_reference("beacon_green"),
            claimed=_reference("beacon_green"),
            pose=(0.0, 0.0, math.pi / 2.0),
        ),
    ]
    result = _evaluate(manifest, events)
    evaluations = result["physical_claim_evaluations"]
    summary = result["physical_claim_summary"]
    assert [event["accepted"] for event in evaluations] == [True, True, True]
    assert [event["credited"] for event in evaluations] == [True, False, True]
    assert [event["duplicate_physical_claim_not_credited"] for event in evaluations] == [
        False,
        True,
        False,
    ]
    assert summary["credited_object_ids"] == ["beacon_green", "beacon_red"]
    assert summary["first_credited_by_object"] == [
        {"object_id": "beacon_green", "tick": 7, "event_id": "green-first"},
        {"object_id": "beacon_red", "tick": 7, "event_id": "red-first"},
    ]
    assert summary["duplicate_physical_claim_not_credited_count"] == 1
    assert summary["all_targets_claimed"] is True


def test_noncanonical_event_order_is_not_silently_sorted() -> None:
    manifest = _manifest()
    events = [
        _event(manifest, event_id="later", event_index=1, tick=2),
        _event(manifest, event_id="earlier", event_index=0, tick=1),
    ]
    result = _evaluate(manifest, events)
    assert [event["event_id"] for event in result["physical_claim_evaluations"]] == [
        "later",
        "earlier",
    ]
    assert all(
        "trace_event_order_invalid" in event["unverifiable_reasons"]
        for event in result["physical_claim_evaluations"]
    )


@pytest.mark.parametrize("invalid_event_id", ("", "\ud800"))
def test_invalid_event_id_does_not_contaminate_tick_or_index_reason(
    invalid_event_id: str,
) -> None:
    manifest = _manifest()
    event = _event(manifest, event_id=invalid_event_id, event_index=0, tick=1)
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [
        "trace_event_order_invalid",
        "event_id_missing_or_duplicate",
    ]
    assert "claim_tick_or_index_invalid" not in evaluation["unverifiable_reasons"]


@pytest.mark.parametrize("field", ("tick", "event_index"))
@pytest.mark.parametrize("invalid_value", (True, -1, 2**63))
def test_tick_and_index_ranges_are_independently_validated(
    field: str, invalid_value: object
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[field] = invalid_value
    evaluation = _only(_evaluate(manifest, [event]))
    assert "trace_event_order_invalid" in evaluation["unverifiable_reasons"]
    assert "claim_tick_or_index_invalid" in evaluation["unverifiable_reasons"]
    if isinstance(invalid_value, bool):
        assert "event_key_set_or_type_invalid" in evaluation["unverifiable_reasons"]


def test_same_tick_nonincreasing_index_marks_only_later_index_invalid() -> None:
    manifest = _manifest()
    events = [
        _event(manifest, event_id="first", event_index=0, tick=1),
        _event(manifest, event_id="second", event_index=0, tick=1),
    ]
    evaluations = _evaluate(manifest, events)["physical_claim_evaluations"]
    assert "trace_event_order_invalid" in evaluations[0]["unverifiable_reasons"]
    assert "claim_tick_or_index_invalid" not in evaluations[0]["unverifiable_reasons"]
    assert evaluations[1]["unverifiable_reasons"] == [
        "trace_event_order_invalid",
        "claim_tick_or_index_invalid",
    ]


def test_unverifiable_precedence_keeps_all_reasons_and_hides_rejections() -> None:
    manifest = _manifest()
    event = _event(manifest, pose=(-2.0, 0.0, 0.0))
    event["extra"] = "not allowed"
    event["trace_id"] = "wrong"
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [
        "event_key_set_or_type_invalid",
        "event_trace_identity_mismatch",
    ]
    assert evaluation["factors"]["distance_passes"] is False
    assert evaluation["rejection_reasons"] == []
    assert evaluation["decision"] == "unverifiable"


def test_every_unverifiable_reason_has_a_witness_in_frozen_order() -> None:
    observed: set[str] = set()

    duplicate_invalid_manifest = _manifest(
        landmarks=(
            _box(
                "beacon_red",
                1.0,
                0.0,
                size_x=0.0,
                material_id="landmark_red",
            ),
            _box("beacon_red", 0.0, 1.0, material_id="landmark_green"),
        )
    )
    event = _event(duplicate_invalid_manifest)
    event.update(
        {
            "trace_id": "event-trace",
            "episode_id": "event-episode",
            "scene_id": "event-scene",
            "event_id": "",
            "tick": True,
            "requested_target": None,
            "claimed_target": None,
            "robot_pose_world_xy_yaw": [],
            "pose_hex": [],
            "pose_provenance": [],
            "extra": True,
        }
    )
    trace, expected_hash = _trace(
        duplicate_invalid_manifest,
        [event],
        task_ids=["beacon_red"],
    )
    trace.update(
        {
            "schema": "wrong-schema",
            "trace_id": "",
            "episode_id": "",
            "scene_id": "wrong-scene",
            "physical_manifest_sha256": "0" * 64,
            "task_object_ids": ["beacon_red", "beacon_red"],
            "task_object_set_sha256": "0" * 64,
            "evaluator_feedback_to_controller": ["forbidden"],
        }
    )
    broad = _only(
        claims.evaluate_physical_claim_trace(
            trace,
            duplicate_invalid_manifest,
            ["beacon_red"],
            expected_hash,
        )
    )
    broad_expected = {
        "trace_schema_or_key_set_invalid",
        "trace_id_missing_or_invalid",
        "episode_id_missing_or_invalid",
        "scene_manifest_identity_mismatch",
        "physical_manifest_commitment_mismatch",
        "task_object_ids_not_exact_sorted_unique",
        "task_object_set_mismatch",
        "task_object_commitment_mismatch",
        "evaluator_feedback_to_controller_nonempty",
        "trace_event_order_invalid",
        "manifest_duplicate_object_id",
        "manifest_invalid_physical_geometry",
        "event_key_set_or_type_invalid",
        "event_trace_identity_mismatch",
        "event_id_missing_or_duplicate",
        "claim_tick_or_index_invalid",
        "requested_reference_malformed",
        "claimed_reference_malformed",
        "pose_provenance_invalid",
        "claim_pose_missing_or_nonfinite",
    }
    assert broad["unverifiable_reasons"] == [
        reason for reason in claims.UNVERIFIABLE_REASONS if reason in broad_expected
    ]
    observed.update(broad["unverifiable_reasons"])

    manifest = _manifest()
    cases: list[dict] = []
    unresolved = _event(
        manifest,
        requested=_reference("missing"),
        claimed=_reference("missing"),
    )
    cases.append(_only(_evaluate(manifest, [unresolved])))

    repeated = _manifest(
        landmarks=(
            _box("red-a", 1.0, 0.0, material_id="landmark_red"),
            _box("red-b", 0.0, 1.0, material_id="landmark_red"),
        )
    )
    color = {"namespace": "task_color", "value": "red"}
    cases.append(
        _only(_evaluate(repeated, [_event(repeated, requested=color, claimed=color)]))
    )

    alias = {"namespace": "legacy_alias", "value": "red"}
    cases.append(
        _only(_evaluate(manifest, [_event(manifest, requested=alias, claimed=alias)]))
    )

    precision = _event(manifest)
    precision["pose_binary64_le_sha256"] = "0" * 64
    cases.append(_only(_evaluate(manifest, [precision])))

    far_target = _box("beacon_red", 1e308, 0.0, material_id="landmark_red")
    far_manifest = _manifest(landmarks=(far_target,))
    cases.append(
        _only(
            _evaluate(
                far_manifest,
                [_event(far_manifest, pose=(-1e308, 0.0, 0.0))],
            )
        )
    )

    legacy = _event(manifest, provenance="legacy_inferred")
    legacy["robot_pose_world_xy_yaw"] = [0.0, 0.0]
    legacy["pose_hex"] = ["0x0.0p+0", "0x0.0p+0"]
    cases.append(_only(_evaluate(manifest, [legacy])))

    for evaluation in cases:
        assert evaluation["unverifiable_reasons"] == [
            reason
            for reason in claims.UNVERIFIABLE_REASONS
            if reason in set(evaluation["unverifiable_reasons"])
        ]
        observed.update(evaluation["unverifiable_reasons"])
    assert observed == set(claims.UNVERIFIABLE_REASONS)


def test_all_six_rejection_reasons_are_gathered_in_frozen_order() -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_red", 0.0, -2.0, material_id="landmark_red"),
            _box("beacon_green", 0.0, 2.0, material_id="landmark_green"),
            _box("beacon_blue", 1.0, 0.0, material_id="landmark_blue"),
        ),
        obstacles=(_box("blocker", 0.0, 0.0),),
    )
    event = _event(
        manifest,
        requested=_reference("beacon_green"),
        claimed=_reference("beacon_blue"),
        pose=(-2.0, 0.0, math.pi),
    )
    evaluation = _only(_evaluate(manifest, [event], task_ids=["beacon_red"]))
    assert evaluation["unverifiable_reasons"] == []
    assert evaluation["rejection_reasons"] == list(claims.REJECTION_REASONS)
    assert evaluation["decision"] == "rejected"
    assert evaluation["accepted"] is False
    assert evaluation["credited"] is False


def test_task_color_resolution_is_unique_or_ambiguous() -> None:
    unique_manifest = _manifest()
    color = {"namespace": "task_color", "value": "red"}
    unique = _only(
        _evaluate(
            unique_manifest,
            [_event(unique_manifest, requested=color, claimed=color)],
        )
    )
    assert unique["requested_resolution"] == {
        "status": "resolved",
        "resolved_object_id": "beacon_red",
    }

    repeated_manifest = _manifest(
        landmarks=(
            _box("red_a", 1.0, 0.0, material_id="landmark_red"),
            _box("red_b", 1.0, 0.2, material_id="LANDMARK_RED"),
        )
    )
    ambiguous = _only(
        _evaluate(
            repeated_manifest,
            [
                _event(
                    repeated_manifest,
                    requested=color,
                    claimed=color,
                )
            ],
        )
    )
    assert "requested_identity_ambiguous" in ambiguous["unverifiable_reasons"]
    assert "claimed_identity_ambiguous" in ambiguous["unverifiable_reasons"]


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("requested_target", "requested_identity_ambiguous"),
        ("claimed_target", "claimed_identity_ambiguous"),
    ),
)
def test_requested_and_claimed_color_ambiguity_are_independent(
    field: str, reason: str
) -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
            _box("red_again", 0.0, 1.0, material_id="LANDMARK_RED"),
        )
    )
    event = _event(manifest)
    event[field] = {"namespace": "task_color", "value": "red"}
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [reason]


@pytest.mark.parametrize(
    ("reference", "reason"),
    (
        (None, "requested_reference_malformed"),
        ({}, "requested_reference_malformed"),
        (
            {"namespace": "object_id", "value": "beacon_red", "extra": True},
            "requested_reference_malformed",
        ),
        (
            {"namespace": 1, "value": "beacon_red"},
            "requested_reference_malformed",
        ),
        (
            {"namespace": "object_id", "value": ""},
            "requested_reference_malformed",
        ),
        (
            {"namespace": "unknown", "value": "beacon_red"},
            "requested_reference_malformed",
        ),
        (
            {"namespace": "object_id", "value": "missing"},
            "requested_identity_unresolved",
        ),
        (
            {"namespace": "task_color", "value": "RED"},
            "requested_identity_unresolved",
        ),
    ),
)
def test_typed_reference_mutations_fail_with_exact_reason(
    reference: object, reason: str
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event["requested_target"] = reference
    evaluation = _only(_evaluate(manifest, [event]))
    assert reason in evaluation["unverifiable_reasons"]
    assert evaluation["decision"] == "unverifiable"


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("requested_target", "requested_reference_malformed"),
        ("claimed_target", "claimed_reference_malformed"),
    ),
)
@pytest.mark.parametrize(
    "reference",
    (
        None,
        [],
        {},
        {"namespace": "object_id"},
        {"value": "beacon_red"},
        {"namespace": "object_id", "value": "beacon_red", "extra": 1},
        {"namespace": 1, "value": "beacon_red"},
        {"namespace": "object_id", "value": 1},
        {"namespace": "object_id", "value": ""},
        {"namespace": "unknown", "value": "beacon_red"},
        {"namespace": "object_id", "value": "\ud800"},
    ),
)
def test_requested_and_claimed_reference_schema_mutations_fail_closed(
    field: str, reason: str, reference: object
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[field] = reference
    evaluation = _only(_evaluate(manifest, [event]))
    expected = [reason]
    if not isinstance(reference, dict):
        expected.insert(0, "event_key_set_or_type_invalid")
    assert evaluation["unverifiable_reasons"] == expected
    resolution_key = (
        "requested_resolution"
        if field == "requested_target"
        else "claimed_resolution"
    )
    assert evaluation[resolution_key] == {
        "status": "malformed",
        "resolved_object_id": None,
    }


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("requested_target", "requested_identity_unresolved"),
        ("claimed_target", "claimed_identity_unresolved"),
    ),
)
def test_requested_and_claimed_unresolved_are_independent(
    field: str, reason: str
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[field] = _reference("missing")
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [reason]


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("requested_target", "requested_namespace_forbidden_for_provenance"),
        ("claimed_target", "claimed_namespace_forbidden_for_provenance"),
    ),
)
def test_requested_and_claimed_legacy_alias_are_independently_forbidden(
    field: str, reason: str
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[field] = {"namespace": "legacy_alias", "value": "red"}
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [reason]


def test_wrong_identity_and_non_task_identity_are_rejected_in_reason_order() -> None:
    red = _box("beacon_red", 1.0, 0.0, material_id="landmark_red")
    green = _box("beacon_green", 1.0, 0.3, material_id="landmark_green")
    manifest = _manifest(landmarks=(red, green))
    wrong_identity = _only(
        _evaluate(
            manifest,
            [
                _event(
                    manifest,
                    requested=_reference("beacon_red"),
                    claimed=_reference("beacon_green"),
                    pose=(0.0, 0.3, 0.0),
                )
            ],
        )
    )
    assert wrong_identity["rejection_reasons"] == [
        "requested_claimed_identity_mismatch"
    ]

    non_task = _only(
        _evaluate(
            manifest,
            [
                _event(
                    manifest,
                    requested=_reference("beacon_green"),
                    claimed=_reference("beacon_green"),
                    pose=(0.0, 0.3, 0.0),
                )
            ],
            task_ids=["beacon_red"],
        )
    )
    assert non_task["rejection_reasons"] == [
        "requested_identity_not_in_task_set",
        "claimed_identity_not_in_task_set",
    ]
    assert non_task["credited"] is False


@pytest.mark.parametrize(
    ("requested_id", "claimed_id", "pose", "expected_reasons"),
    (
        (
            "beacon_green",
            "beacon_red",
            (0.0, 0.0, 0.0),
            [
                "requested_identity_not_in_task_set",
                "requested_claimed_identity_mismatch",
            ],
        ),
        (
            "beacon_red",
            "beacon_green",
            (0.0, 0.0, math.pi / 2.0),
            [
                "claimed_identity_not_in_task_set",
                "requested_claimed_identity_mismatch",
            ],
        ),
    ),
)
def test_each_non_task_identity_is_rejected_and_never_credited(
    requested_id: str,
    claimed_id: str,
    pose: tuple[float, float, float],
    expected_reasons: list[str],
) -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
            _box("beacon_green", 0.0, 1.0, material_id="landmark_green"),
        )
    )
    event = _event(
        manifest,
        requested=_reference(requested_id),
        claimed=_reference(claimed_id),
        pose=pose,
    )
    evaluation = _only(_evaluate(manifest, [event], task_ids=["beacon_red"]))
    assert evaluation["decision"] == "rejected"
    assert evaluation["accepted"] is False
    assert evaluation["rejection_reasons"] == expected_reasons
    assert evaluation["factors"]["distance_passes"] is True
    assert evaluation["factors"]["line_of_sight_passes"] is True
    assert evaluation["factors"]["bearing_passes"] is True
    assert evaluation["credited"] is False


def test_legacy_alias_is_forbidden_modern_and_unconditional_unverifiable_legacy() -> None:
    manifest = _manifest()
    alias = {"namespace": "legacy_alias", "value": "  RED  "}
    modern = _only(
        _evaluate(manifest, [_event(manifest, requested=alias, claimed=alias)])
    )
    assert modern["unverifiable_reasons"] == [
        "requested_namespace_forbidden_for_provenance",
        "claimed_namespace_forbidden_for_provenance",
    ]

    legacy = _only(
        _evaluate(
            manifest,
            [
                _event(
                    manifest,
                    requested=alias,
                    claimed=alias,
                    provenance="legacy_exact",
                )
            ],
        )
    )
    assert legacy["requested_resolution"]["resolved_object_id"] == "beacon_red"
    assert legacy["unverifiable_reasons"] == [
        "legacy_provenance_noncanonical"
    ]


def test_legacy_missing_yaw_and_rounded_pose_reasons_are_independent() -> None:
    manifest = _manifest()
    rounded = _event(manifest, provenance="legacy_rounded")
    rounded_eval = _only(_evaluate(manifest, [rounded]))
    assert rounded_eval["unverifiable_reasons"] == [
        "legacy_provenance_noncanonical",
        "legacy_pose_rounded_or_inferred",
    ]

    missing = _event(manifest, provenance="legacy_inferred")
    missing["robot_pose_world_xy_yaw"] = [0.0, 0.0]
    missing["pose_hex"] = ["0x0.0p+0", "0x0.0p+0"]
    missing_eval = _only(_evaluate(manifest, [missing]))
    assert "legacy_pose_missing_yaw" in missing_eval["unverifiable_reasons"]
    assert "legacy_pose_rounded_or_inferred" in missing_eval[
        "unverifiable_reasons"
    ]


@pytest.mark.parametrize("missing_representation", ("decimal", "hex"))
def test_legacy_missing_yaw_is_gathered_for_either_required_representation(
    missing_representation: str,
) -> None:
    manifest = _manifest()
    event = _event(manifest, provenance="legacy_inferred")
    if missing_representation == "decimal":
        event["robot_pose_world_xy_yaw"] = [0.0, 0.0]
    else:
        event["pose_hex"] = ["0x0.0p+0", "0x0.0p+0"]
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [
        "claim_pose_missing_or_nonfinite",
        "legacy_provenance_noncanonical",
        "legacy_pose_missing_yaw",
        "legacy_pose_rounded_or_inferred",
    ]


def test_legacy_unverifiable_precedence_hides_available_physical_rejections() -> None:
    manifest = _manifest()
    evaluation = _only(
        _evaluate(
            manifest,
            [
                _event(
                    manifest,
                    provenance="legacy_exact",
                    pose=(-2.0, 0.0, math.pi),
                )
            ],
        )
    )
    assert evaluation["unverifiable_reasons"] == [
        "legacy_provenance_noncanonical"
    ]
    assert evaluation["factors"]["distance_passes"] is False
    assert evaluation["factors"]["bearing_passes"] is False
    assert evaluation["decision"] == "unverifiable"
    assert evaluation["rejection_reasons"] == []


def test_pose_decimal_hex_and_packed_commitment_must_match() -> None:
    manifest = _manifest()
    event = _event(manifest)
    event["pose_binary64_le_sha256"] = "0" * 64
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [
        "claim_pose_precision_commitment_mismatch"
    ]
    assert evaluation["factors"] == {
        "identity_passes": True,
        "distance_passes": True,
        "line_of_sight_passes": True,
        "bearing_passes": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_reasons"),
    (
        (
            "robot_pose_world_xy_yaw",
            [math.nextafter(0.0, math.inf), 0.0, 0.0],
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "robot_pose_world_xy_yaw",
            [-0.0, 0.0, 0.0],
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "pose_hex",
            ["0x0p+0", "0x0.0p+0", "0x0.0p+0"],
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "pose_hex",
            [math.nextafter(0.0, math.inf).hex(), "0x0.0p+0", "0x0.0p+0"],
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "pose_binary64_le_sha256",
            "A" * 64,
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "pose_binary64_le_sha256",
            "0" * 63,
            ["claim_pose_precision_commitment_mismatch"],
        ),
        (
            "pose_binary64_le_sha256",
            1,
            [
                "event_key_set_or_type_invalid",
                "claim_pose_precision_commitment_mismatch",
            ],
        ),
        (
            "robot_pose_world_xy_yaw",
            [0.0, 0.0],
            ["claim_pose_missing_or_nonfinite"],
        ),
        (
            "robot_pose_world_xy_yaw",
            [False, 0.0, 0.0],
            ["claim_pose_missing_or_nonfinite"],
        ),
        (
            "pose_hex",
            ["0x0.0p+0", "0x0.0p+0"],
            ["claim_pose_missing_or_nonfinite"],
        ),
        (
            "pose_hex",
            ["invalid", "0x0.0p+0", "0x0.0p+0"],
            ["claim_pose_missing_or_nonfinite"],
        ),
        (
            "pose_hex",
            ["inf", "0x0.0p+0", "0x0.0p+0"],
            ["claim_pose_missing_or_nonfinite"],
        ),
        (
            "robot_pose_world_xy_yaw",
            None,
            ["event_key_set_or_type_invalid", "claim_pose_missing_or_nonfinite"],
        ),
        (
            "pose_hex",
            None,
            ["event_key_set_or_type_invalid", "claim_pose_missing_or_nonfinite"],
        ),
    ),
)
def test_pose_commitment_mutation_dependencies_are_exact(
    field: str, value: object, expected_reasons: list[str]
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[field] = value
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == expected_reasons
    assert not {
        "claim_pose_missing_or_nonfinite",
        "claim_pose_precision_commitment_mismatch",
    }.issubset(evaluation["unverifiable_reasons"])


def test_pose_mismatch_result_uses_one_authoritative_bit_consistent_triple() -> None:
    manifest = _manifest()
    event = _event(manifest)
    event["robot_pose_world_xy_yaw"] = [-0.0, 0.0, 0.0]
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["unverifiable_reasons"] == [
        "claim_pose_precision_commitment_mismatch"
    ]
    assert evaluation["robot_pose_world_xy_yaw"] == [0.0, 0.0, 0.0]
    assert math.copysign(1.0, evaluation["robot_pose_world_xy_yaw"][0]) == 1.0
    assert evaluation["pose_hex"] == ["0x0.0p+0"] * 3
    assert evaluation["pose_binary64_le_sha256"] == hashlib.sha256(
        struct.pack("<3d", *evaluation["robot_pose_world_xy_yaw"])
    ).hexdigest()


def test_signed_zero_is_preserved_in_pose_and_hash() -> None:
    manifest = _manifest()
    event = _event(manifest, pose=(-0.0, 0.0, -0.0))
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["pose_hex"] == ["-0x0.0p+0", "0x0.0p+0", "-0x0.0p+0"]
    assert evaluation["robot_pose_world_xy_yaw"][0] == 0.0
    assert math.copysign(1.0, evaluation["robot_pose_world_xy_yaw"][0]) == -1.0
    assert evaluation["pose_binary64_le_sha256"] == hashlib.sha256(
        struct.pack("<3d", -0.0, 0.0, -0.0)
    ).hexdigest()


def test_duplicate_manifest_object_id_and_invalid_geometry_are_unverifiable() -> None:
    duplicate = _manifest(
        landmarks=(
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
        ),
        obstacles=(_box("beacon_red", 0.5, 0.0),),
    )
    duplicate_eval = _only(_evaluate(duplicate, [_event(duplicate)]))
    assert "manifest_duplicate_object_id" in duplicate_eval["unverifiable_reasons"]

    invalid = _manifest(
        landmarks=(
            _box(
                "beacon_red",
                1.0,
                0.0,
                size_x=0.0,
                material_id="landmark_red",
            ),
        )
    )
    invalid_eval = _only(_evaluate(invalid, [_event(invalid)]))
    assert "manifest_invalid_physical_geometry" in invalid_eval[
        "unverifiable_reasons"
    ]


def test_duplicate_landmark_ids_resolve_as_ambiguous_without_credit() -> None:
    duplicate = _manifest(
        landmarks=(
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
            _box("beacon_red", 0.0, 1.0, material_id="landmark_green"),
        )
    )
    evaluation = _only(
        _evaluate(duplicate, [_event(duplicate)], task_ids=["beacon_red"])
    )
    assert evaluation["requested_resolution"] == {
        "status": "ambiguous",
        "resolved_object_id": None,
    }
    assert evaluation["claimed_resolution"] == {
        "status": "ambiguous",
        "resolved_object_id": None,
    }
    assert evaluation["unverifiable_reasons"] == [
        "manifest_duplicate_object_id",
        "requested_identity_ambiguous",
        "claimed_identity_ambiguous",
    ]
    assert evaluation["credited"] is False


@pytest.mark.parametrize("invalid_id", ([], {}, "", "\ud800", 7))
def test_malformed_manifest_object_ids_fail_closed(invalid_id: object) -> None:
    manifest = _manifest()
    malformed = replace(
        manifest,
        obstacles=(replace(_box("obstacle", 0.5, 0.5), object_id=invalid_id),),
    )
    evaluation = _only(_evaluate(malformed, [_event(malformed)]))
    assert evaluation["unverifiable_reasons"] == [
        "manifest_invalid_physical_geometry"
    ]
    assert evaluation["decision"] == "unverifiable"


@pytest.mark.parametrize("duplicate_invalid_id", ("", "\ud800"))
def test_duplicate_invalid_string_object_ids_gather_both_global_reasons(
    duplicate_invalid_id: str,
) -> None:
    manifest = _manifest()
    malformed = replace(
        manifest,
        obstacles=(
            replace(_box("first", 0.5, 0.5), object_id=duplicate_invalid_id),
            replace(_box("second", 0.5, -0.5), object_id=duplicate_invalid_id),
        ),
    )
    evaluation = _only(_evaluate(malformed, [_event(malformed)]))
    assert evaluation["unverifiable_reasons"] == [
        "manifest_duplicate_object_id",
        "manifest_invalid_physical_geometry",
    ]
    assert evaluation["decision"] == "unverifiable"
    assert evaluation["credited"] is False


def test_finite_inputs_with_overflowing_physical_intermediate_are_unverifiable() -> None:
    target = _box("beacon_red", 1e308, 0.0, material_id="landmark_red")
    manifest = _manifest(landmarks=(target,))
    evaluation = _only(
        _evaluate(manifest, [_event(manifest, pose=(-1e308, 0.0, 0.0))])
    )
    assert "physical_computation_nonfinite" in evaluation["unverifiable_reasons"]
    assert evaluation["factors"]["distance_passes"] is None


@pytest.mark.parametrize("nonfinite", (math.inf, -math.inf, math.nan))
def test_nonfinite_manifest_geometry_is_invalid_without_physical_factors(
    nonfinite: float,
) -> None:
    manifest = _manifest()
    invalid_target = replace(
        manifest.landmarks[0],
        center_xyz_m=(nonfinite, 0.0, 0.5),
    )
    invalid = replace(manifest, landmarks=(invalid_target,))
    event = _event(invalid)
    trace, task_hash = _trace(invalid, [event])
    evaluation = _only(
        claims.evaluate_physical_claim_trace(
            trace,
            invalid,
            trace["task_object_ids"],
            task_hash,
        )
    )
    assert evaluation["unverifiable_reasons"] == [
        "physical_manifest_commitment_mismatch",
        "task_object_commitment_mismatch",
        "manifest_invalid_physical_geometry",
    ]
    assert evaluation["factors"]["distance_passes"] is None


def test_trace_feedback_extra_key_and_commitment_mutations_fail_closed() -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    trace["evaluator_feedback_to_controller"] = ["forbidden"]
    trace["extra"] = True
    trace["task_object_set_sha256"] = "0" * 64
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )
    evaluation = _only(result)
    assert evaluation["unverifiable_reasons"][:3] == [
        "trace_schema_or_key_set_invalid",
        "task_object_commitment_mismatch",
        "evaluator_feedback_to_controller_nonempty",
    ]
    assert evaluation["rejection_reasons"] == []


@pytest.mark.parametrize(
    ("key", "wrong_value", "expected_global_reason"),
    (
        ("schema", 1, "trace_schema_or_key_set_invalid"),
        ("trace_id", 1, "trace_id_missing_or_invalid"),
        ("episode_id", 1, "episode_id_missing_or_invalid"),
        ("scene_id", 1, "scene_manifest_identity_mismatch"),
        (
            "physical_manifest_sha256",
            1,
            "physical_manifest_commitment_mismatch",
        ),
        ("task_object_ids", {}, "task_object_ids_not_exact_sorted_unique"),
        ("task_object_set_sha256", 1, "task_object_commitment_mismatch"),
        ("controller_claim_attempts", {}, "trace_event_order_invalid"),
        (
            "evaluator_feedback_to_controller",
            {},
            "evaluator_feedback_to_controller_nonempty",
        ),
    ),
)
def test_every_trace_envelope_wrong_type_fails_closed(
    key: str, wrong_value: object, expected_global_reason: str
) -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    trace[key] = wrong_value
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )
    summary = result["physical_claim_summary"]
    assert expected_global_reason in summary["trace_unverifiable_reasons"]
    assert summary["credited_count"] == 0
    assert summary["all_targets_claimed"] is False


@pytest.mark.parametrize("wrong_schema", ("", "lewm_go2_claim_trace_v0"))
def test_wrong_trace_schema_literal_fails_closed(wrong_schema: str) -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    trace["schema"] = wrong_schema
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, ["beacon_red"], task_hash
    )
    assert result["physical_claim_summary"]["trace_unverifiable_reasons"] == [
        "trace_schema_or_key_set_invalid"
    ]


@pytest.mark.parametrize("key", ("trace_id", "episode_id"))
@pytest.mark.parametrize("invalid_id", ("", "\ud800"))
def test_trace_and_episode_identifiers_must_be_nonempty_utf8(
    key: str, invalid_id: str
) -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    trace[key] = invalid_id
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, ["beacon_red"], task_hash
    )
    reason = (
        "trace_id_missing_or_invalid"
        if key == "trace_id"
        else "episode_id_missing_or_invalid"
    )
    assert reason in result["physical_claim_summary"]["trace_unverifiable_reasons"]
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_scene_and_manifest_identity_mismatches_are_independent() -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    trace["scene_id"] = "other-scene"
    trace["physical_manifest_sha256"] = "0" * 64
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, ["beacon_red"], task_hash
    )
    assert result["physical_claim_summary"]["trace_unverifiable_reasons"] == [
        "scene_manifest_identity_mismatch",
        "physical_manifest_commitment_mismatch",
    ]


@pytest.mark.parametrize("invalid_scene_id", ("", "\ud800"))
def test_scene_identifier_must_be_nonempty_utf8(invalid_scene_id: str) -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [])
    trace["scene_id"] = invalid_scene_id
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, ["beacon_red"], task_hash
    )
    assert result["physical_claim_summary"]["trace_unverifiable_reasons"] == [
        "scene_manifest_identity_mismatch"
    ]


@pytest.mark.parametrize(
    "trace_task_ids",
    (
        ["beacon_red", "beacon_green", "beacon_red"],
        ["beacon_red", "beacon_green"],
        [],
        ["unknown"],
    ),
)
def test_trace_task_set_mutations_never_receive_credit_or_completion(
    trace_task_ids: list[str],
) -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_green", 0.0, 1.0, material_id="landmark_green"),
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
        )
    )
    expected_ids = ["beacon_green", "beacon_red"]
    trace, expected_hash = _trace(manifest, [_event(manifest)], task_ids=expected_ids)
    trace["task_object_ids"] = trace_task_ids
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        expected_ids,
        expected_hash,
    )
    reasons = result["physical_claim_summary"]["trace_unverifiable_reasons"]
    assert "task_object_set_mismatch" in reasons
    if len(trace_task_ids) != len(set(trace_task_ids)) or trace_task_ids != sorted(
        trace_task_ids, key=lambda value: value.encode("utf-8")
    ):
        assert "task_object_ids_not_exact_sorted_unique" in reasons
    assert result["physical_claim_summary"]["credited_count"] == 0
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_trace_task_surrogate_identifier_fails_closed_without_utf8_exception() -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [], task_ids=["beacon_red"])
    trace["task_object_ids"] = ["\ud800"]
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, ["beacon_red"], task_hash
    )
    assert result["physical_claim_summary"]["trace_unverifiable_reasons"] == [
        "task_object_ids_not_exact_sorted_unique",
        "task_object_set_mismatch",
    ]
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_invalid_expected_task_binding_never_falls_back_to_untrusted_trace() -> None:
    manifest = _manifest()
    trace, empty_trace_hash = _trace(manifest, [], task_ids=[])
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        [1],  # type: ignore[list-item]
        empty_trace_hash,
    )
    summary = result["physical_claim_summary"]
    assert summary["task_object_ids"] == []
    assert summary["task_object_set_sha256"] is None
    assert summary["trace_unverifiable_reasons"] == [
        "task_object_set_mismatch",
        "task_object_commitment_mismatch",
    ]
    assert summary["credited_count"] == 0
    assert summary["all_targets_claimed"] is False


def test_independent_expected_task_set_is_authoritative_not_trace_fallback() -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_green", 0.0, 1.0, material_id="landmark_green"),
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
        )
    )
    trace, _red_hash = _trace(manifest, [], task_ids=["beacon_red"])
    green_ids = ["beacon_green"]
    green_hash = _task_hash(manifest, green_ids)
    result = claims.evaluate_physical_claim_trace(
        trace, manifest, green_ids, green_hash
    )
    summary = result["physical_claim_summary"]
    assert summary["task_object_ids"] == green_ids
    assert summary["task_object_set_sha256"] == green_hash
    assert summary["trace_unverifiable_reasons"] == [
        "task_object_set_mismatch",
        "task_object_commitment_mismatch",
    ]
    assert summary["all_targets_claimed"] is False


@pytest.mark.parametrize(
    "expected_ids",
    (
        ["beacon_red", "beacon_green"],
        ["beacon_green", "beacon_green"],
        [],
        ["unknown"],
        ["\ud800"],
    ),
)
def test_independent_expected_task_set_mutations_fail_closed(
    expected_ids: list[str],
) -> None:
    manifest = _manifest(
        landmarks=(
            _box("beacon_green", 0.0, 1.0, material_id="landmark_green"),
            _box("beacon_red", 1.0, 0.0, material_id="landmark_red"),
        )
    )
    canonical_ids = ["beacon_green", "beacon_red"]
    trace, canonical_hash = _trace(manifest, [], task_ids=canonical_ids)
    expected_hash = _task_hash(manifest, expected_ids)
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        expected_ids,
        expected_hash,
    )
    summary = result["physical_claim_summary"]
    assert "task_object_set_mismatch" in summary["trace_unverifiable_reasons"]
    assert "task_object_commitment_mismatch" in summary[
        "trace_unverifiable_reasons"
    ]
    assert summary["credited_count"] == 0
    assert summary["all_targets_claimed"] is False
    if expected_ids == []:
        assert summary["task_object_ids"] == []
        assert summary["task_object_set_sha256"] == expected_hash
    else:
        assert summary["task_object_set_sha256"] is None
    assert trace["task_object_set_sha256"] == canonical_hash


@pytest.mark.parametrize("invalid_commitment", ("0" * 64, "A" * 64, "", 1))
def test_independent_expected_task_commitment_mutations_fail_closed(
    invalid_commitment: object,
) -> None:
    manifest = _manifest()
    trace, _task_commitment = _trace(manifest, [], task_ids=["beacon_red"])
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        invalid_commitment,  # type: ignore[arg-type]
    )
    summary = result["physical_claim_summary"]
    assert summary["trace_unverifiable_reasons"] == [
        "task_object_commitment_mismatch"
    ]
    assert summary["all_targets_claimed"] is False


@pytest.mark.parametrize(
    "key",
    (
        "schema",
        "trace_id",
        "episode_id",
        "scene_id",
        "physical_manifest_sha256",
        "task_object_ids",
        "task_object_set_sha256",
        "controller_claim_attempts",
        "evaluator_feedback_to_controller",
    ),
)
def test_every_missing_trace_envelope_key_fails_closed(key: str) -> None:
    manifest = _manifest()
    trace, task_hash = _trace(manifest, [_event(manifest)])
    del trace[key]
    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )
    summary = result["physical_claim_summary"]
    assert "trace_schema_or_key_set_invalid" in summary[
        "trace_unverifiable_reasons"
    ]
    assert summary["credited_count"] == 0


@pytest.mark.parametrize("key", tuple(sorted(claims._EVENT_KEYS)))
def test_every_missing_event_key_is_unverifiable(key: str) -> None:
    manifest = _manifest()
    event = _event(manifest)
    del event[key]
    evaluation = _only(_evaluate(manifest, [event]))
    assert "event_key_set_or_type_invalid" in evaluation["unverifiable_reasons"]
    assert evaluation["decision"] == "unverifiable"


@pytest.mark.parametrize(
    ("key", "wrong_value"),
    (
        ("trace_id", 1),
        ("episode_id", 1),
        ("scene_id", 1),
        ("event_id", 1),
        ("tick", "10"),
        ("event_index", "0"),
        ("requested_target", []),
        ("claimed_target", []),
        ("robot_pose_world_xy_yaw", {}),
        ("pose_binary64_le_sha256", 1),
        ("pose_hex", {}),
        ("pose_provenance", []),
        ("physical_manifest_sha256", 1),
    ),
)
def test_every_event_top_level_wrong_type_is_unverifiable_not_exceptional(
    key: str, wrong_value: object
) -> None:
    manifest = _manifest()
    event = _event(manifest)
    event[key] = wrong_value
    evaluation = _only(_evaluate(manifest, [event]))
    assert "event_key_set_or_type_invalid" in evaluation["unverifiable_reasons"]
    assert evaluation["decision"] == "unverifiable"
    assert evaluation["accepted"] is False
    assert evaluation["credited"] is False


def test_geometry_order_does_not_change_factors_but_changes_manifest_bound_hashes() -> None:
    left = _box("left", 0.5, 0.5)
    right = _box("right", 0.5, -0.5)
    first = _manifest(obstacles=(left, right))
    second = _manifest(obstacles=(right, left))
    first_result = _evaluate(first, [_event(first)])
    second_result = _evaluate(second, [_event(second)])
    first_eval = _only(first_result)
    second_eval = _only(second_result)

    assert first_eval["factors"] == second_eval["factors"]
    assert first_eval["physical_blockers"] == second_eval["physical_blockers"]
    assert manifest_sha256(first) != manifest_sha256(second)
    assert first_eval["content_sha256"] != second_eval["content_sha256"]


def test_utf8_task_identity_sort_is_exact() -> None:
    landmarks = (
        _box("z", 1.0, 0.0, material_id="landmark_red"),
        _box("\u03b2", 1.0, 0.2, material_id="landmark_green"),
        _box("\u00e9", 1.0, -0.2, material_id="landmark_blue"),
    )
    manifest = _manifest(landmarks=landmarks)
    task_ids = sorted(
        [item.object_id for item in landmarks],
        key=lambda item: item.encode("utf-8"),
    )
    event = _event(
        manifest,
        requested=_reference("z"),
        claimed=_reference("z"),
    )
    result = _evaluate(manifest, [event], task_ids=task_ids)
    assert result["physical_claim_summary"]["task_object_ids"] == task_ids
    assert "\\u00e9" in json.dumps(result, ensure_ascii=True)


def test_utf8_blockers_are_sorted_by_exact_encoded_object_id_bytes() -> None:
    blockers = (
        _box("\u03b2", 0.5, 0.0),
        _box("\u00e9", 0.5, 0.0),
        _box("z", 0.5, 0.0),
    )
    manifest = _manifest(obstacles=blockers)
    evaluation = _only(_evaluate(manifest, [_event(manifest)]))
    assert evaluation["physical_blockers"] == [
        ["obstacles", "z"],
        ["obstacles", "\u00e9"],
        ["obstacles", "\u03b2"],
    ]


def test_composed_and_decomposed_unicode_identities_remain_distinct() -> None:
    composed = "\u00e9"
    decomposed = "e\u0301"
    manifest = _manifest(
        landmarks=(
            _box(composed, 1.0, 0.0, material_id="landmark_red"),
            _box(decomposed, 0.0, 1.0, material_id="landmark_green"),
        )
    )
    event = _event(
        manifest,
        requested=_reference(composed),
        claimed=_reference(decomposed),
        pose=(0.0, 0.0, math.pi / 2.0),
    )
    evaluation = _only(_evaluate(manifest, [event]))
    assert evaluation["requested_resolution"]["resolved_object_id"] == composed
    assert evaluation["claimed_resolution"]["resolved_object_id"] == decomposed
    assert evaluation["rejection_reasons"] == [
        "requested_claimed_identity_mismatch"
    ]


def test_canonical_unicode_signed_zero_event_hash_rejects_json_mutations() -> None:
    object_id = "\u00e9"
    manifest = _manifest(
        landmarks=(_box(object_id, 1.0, 0.0, material_id="landmark_red"),)
    )
    event = _event(
        manifest,
        requested=_reference(object_id),
        claimed=_reference(object_id),
        pose=(-0.0, 0.0, -0.0),
    )
    evaluation = _only(_evaluate(manifest, [event]))
    core = {
        key: value for key, value in evaluation.items() if key != "content_sha256"
    }
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert b"\\u00e9" in canonical
    assert evaluation["content_sha256"] == hashlib.sha256(canonical).hexdigest()
    mutations = (
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8"),
        json.dumps(
            core,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
        json.dumps(
            core,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
    )
    assert all(
        hashlib.sha256(payload).hexdigest() != evaluation["content_sha256"]
        for payload in mutations
    )


def test_public_surface_exposes_only_trace_evaluator() -> None:
    assert claims.__all__ == ["evaluate_physical_claim_trace"]
    assert not hasattr(claims, "evaluate_physical_claim_event")


def test_contract_hash_is_the_exact_frozen_binding_bytes() -> None:
    binding = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "lewm_go2_canonical_physical_claim_evaluator_binding_2026-07-11.md"
    )
    assert hashlib.sha256(binding.read_bytes()).hexdigest() == (
        claims.EVALUATOR_CONTRACT_SHA256
    )


def test_import_source_and_module_globals_are_pure_and_stdlib_only() -> None:
    command = """
import ast
import inspect
import sys
from types import ModuleType
from lewm.benchmarks import go2_physical_claim_evaluator as evaluator
assert evaluator.__all__ == ["evaluate_physical_claim_trace"]
tree = ast.parse(inspect.getsource(evaluator))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name.split(".", 1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add((node.module or "").split(".", 1)[0])
assert imports == {
    "__future__", "collections", "copy", "dataclasses", "hashlib", "json",
    "math", "struct", "typing",
}
module_globals = {
    value.__name__
    for value in vars(evaluator).values()
    if isinstance(value, ModuleType)
}
assert module_globals == {"hashlib", "json", "math", "struct"}
assert not any(name.startswith("lewm_worlds") for name in sys.modules)
assert not any(name.startswith("lewm.datasets") for name in sys.modules)
assert not any(name.startswith("lewm.models") for name in sys.modules)
assert not any("renderer" in name or "exporter" in name for name in sys.modules)
assert "torch" not in sys.modules
assert "np" not in vars(evaluator)
assert not any(
    isinstance(value, ModuleType) and value.__name__.startswith("numpy")
    for value in vars(evaluator).values()
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=claims.__file__.rsplit("/lewm/benchmarks/", 1)[0],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_binding_integration_gates_have_no_evaluator_local_pending_marker() -> None:
    assert _PENDING_INTEGRATION_GATES == ()


def test_nonfinite_raw_trace_is_rejected_before_unhashable_output() -> None:
    manifest = _manifest()
    event = _event(manifest)
    event["robot_pose_world_xy_yaw"][0] = math.nan
    trace, task_hash = _trace(manifest, [event])
    with pytest.raises(ValueError, match="finite canonical-JSON"):
        claims.evaluate_physical_claim_trace(
            trace,
            manifest,
            trace["task_object_ids"],
            task_hash,
        )


def test_trace_dict_subclass_evaluates_exact_canonical_byte_roundtrip() -> None:
    manifest = _manifest()
    serialized_trace, task_hash = _trace(manifest, [_event(manifest)])
    semantic_trace = deepcopy(serialized_trace)
    semantic_trace["schema"] = "semantic-only-wrong-schema"
    attacked = _SerializedItemsDict(semantic_trace, serialized_trace)
    twin = _SerializedItemsDict(deepcopy(semantic_trace), deepcopy(serialized_trace))
    plain = json.loads(claims._canonical_bytes(twin).decode("utf-8"))

    attacked_result = claims.evaluate_physical_claim_trace(
        attacked, manifest, ["beacon_red"], task_hash
    )
    plain_result = claims.evaluate_physical_claim_trace(
        plain, manifest, ["beacon_red"], task_hash
    )
    assert attacked_result == plain_result
    assert attacked_result["physical_claim_summary"]["credited_count"] == 1
    _assert_exact_plain_json_builtins(attacked_result["controller_claim_attempts"])


def test_event_dict_subclass_evaluates_exact_canonical_byte_roundtrip() -> None:
    manifest = _manifest()

    def attacked_trace() -> tuple[dict, str]:
        serialized_event = _event(manifest)
        semantic_event = deepcopy(serialized_event)
        semantic_event["pose_provenance"] = []
        attacked_event = _SerializedItemsDict(semantic_event, serialized_event)
        return _trace(manifest, [attacked_event])

    attacked, task_hash = attacked_trace()
    twin, _twin_hash = attacked_trace()
    plain = json.loads(claims._canonical_bytes(twin).decode("utf-8"))
    attacked_result = claims.evaluate_physical_claim_trace(
        attacked, manifest, ["beacon_red"], task_hash
    )
    plain_result = claims.evaluate_physical_claim_trace(
        plain, manifest, ["beacon_red"], task_hash
    )
    assert attacked_result == plain_result
    assert _only(attacked_result)["decision"] == "accepted"
    _assert_exact_plain_json_builtins(attacked_result["controller_claim_attempts"])


def test_reference_dict_subclass_evaluates_exact_canonical_byte_roundtrip() -> None:
    manifest = _manifest()

    def attacked_trace() -> tuple[dict, str]:
        serialized_reference = _reference("beacon_red")
        semantic_reference = _reference("missing")
        event = _event(
            manifest,
            requested=_SerializedItemsDict(
                semantic_reference,
                serialized_reference,
            ),
            claimed=_SerializedItemsDict(
                deepcopy(semantic_reference),
                deepcopy(serialized_reference),
            ),
        )
        return _trace(manifest, [event])

    attacked, task_hash = attacked_trace()
    twin, _twin_hash = attacked_trace()
    plain = json.loads(claims._canonical_bytes(twin).decode("utf-8"))
    attacked_result = claims.evaluate_physical_claim_trace(
        attacked, manifest, ["beacon_red"], task_hash
    )
    plain_result = claims.evaluate_physical_claim_trace(
        plain, manifest, ["beacon_red"], task_hash
    )
    assert attacked_result == plain_result
    assert _only(attacked_result)["requested_resolution"] == {
        "status": "resolved",
        "resolved_object_id": "beacon_red",
    }
    _assert_exact_plain_json_builtins(attacked_result["controller_claim_attempts"])


def test_stateful_list_subclasses_are_frozen_once_at_both_api_boundaries() -> None:
    manifest = _manifest()

    def attacked_inputs() -> tuple[dict, _FirstThenList, _FirstThenList, str]:
        valid_event = _event(manifest)
        invalid_event = deepcopy(valid_event)
        invalid_event["pose_provenance"] = []
        attempts = _FirstThenList([valid_event], [invalid_event])
        trace, task_hash = _trace(manifest, attempts)
        trace_tasks = _FirstThenList(["beacon_red"], ["unknown"])
        trace["task_object_ids"] = trace_tasks
        expected_tasks = _FirstThenList(["beacon_red"], ["unknown"])
        return trace, trace_tasks, expected_tasks, task_hash

    attacked, trace_tasks, expected_tasks, task_hash = attacked_inputs()
    twin, _twin_trace_tasks, twin_expected_tasks, _twin_hash = attacked_inputs()
    plain_trace = json.loads(claims._canonical_bytes(twin).decode("utf-8"))
    plain_expected = json.loads(
        claims._canonical_bytes(twin_expected_tasks).decode("utf-8")
    )

    attacked_result = claims.evaluate_physical_claim_trace(
        attacked,
        manifest,
        expected_tasks,
        task_hash,
    )
    plain_result = claims.evaluate_physical_claim_trace(
        plain_trace,
        manifest,
        plain_expected,
        task_hash,
    )
    assert attacked_result == plain_result
    assert attacked["controller_claim_attempts"].iteration_count == 1
    assert trace_tasks.iteration_count == 1
    assert expected_tasks.iteration_count == 1
    assert _only(attacked_result)["decision"] == "accepted"
    _assert_exact_plain_json_builtins(attacked_result["controller_claim_attempts"])
    _assert_exact_plain_json_builtins(attacked_result["task_object_ids"])


def test_expected_task_hash_subclass_uses_its_canonical_string_bytes() -> None:
    manifest = _manifest()
    trace, _task_hash = _trace(manifest, [_event(manifest)])
    attacked_hash = _AlwaysEqualString("0" * 64)
    attacked = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        attacked_hash,
    )
    plain = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        "0" * 64,
    )
    assert attacked == plain
    assert _only(attacked)["decision"] == "unverifiable"
    assert "task_object_commitment_mismatch" in _only(attacked)[
        "unverifiable_reasons"
    ]


def test_manifest_string_subclass_cannot_alias_expected_task_identity() -> None:
    manifest = _manifest(
        landmarks=(
            _box(
                _ManifestAliasString("evil_target"),
                1.0,
                0.0,
                material_id="landmark_red",
            ),
        )
    )
    event = _event(
        manifest,
        requested=_reference("beacon_red"),
        claimed=_reference("beacon_red"),
    )
    trace, task_hash = _trace(manifest, [event], task_ids=["beacon_red"])

    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )

    assert _only(result)["decision"] == "unverifiable"
    assert "manifest_invalid_physical_geometry" in _only(result)[
        "unverifiable_reasons"
    ]
    assert result["physical_claim_summary"]["credited_count"] == 0
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_manifest_float_subclass_cannot_change_committed_target_geometry() -> None:
    manifest = _manifest(
        landmarks=(
            _box(
                "beacon_red",
                _CoercingFloat(100.0),
                0.0,
                material_id="landmark_red",
            ),
        )
    )
    trace, task_hash = _trace(manifest, [_event(manifest)])

    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )

    assert _only(result)["decision"] == "unverifiable"
    assert "manifest_invalid_physical_geometry" in _only(result)[
        "unverifiable_reasons"
    ]
    assert _only(result)["distance_m"] is None
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_manifest_string_subclass_cannot_change_task_color_resolution() -> None:
    manifest = _manifest(
        landmarks=(
            _box(
                "beacon_red",
                1.0,
                0.0,
                material_id=_CoercingMaterial("not_a_landmark"),
            ),
        )
    )
    task_color = {"namespace": "task_color", "value": "red"}
    trace, task_hash = _trace(
        manifest,
        [_event(manifest, requested=task_color, claimed=task_color)],
    )

    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )

    assert _only(result)["decision"] == "unverifiable"
    assert "manifest_invalid_physical_geometry" in _only(result)[
        "unverifiable_reasons"
    ]
    assert result["physical_claim_summary"]["credited_count"] == 0
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_manifest_to_dict_cannot_commit_different_geometry_than_fields() -> None:
    base = _manifest()
    manifest = _SplitSerializationManifest(**base.__dict__)
    trace, task_hash = _trace(manifest, [_event(manifest)])

    result = claims.evaluate_physical_claim_trace(
        trace,
        manifest,
        ["beacon_red"],
        task_hash,
    )

    assert _only(result)["decision"] == "unverifiable"
    assert "physical_manifest_commitment_mismatch" in _only(result)[
        "unverifiable_reasons"
    ]
    assert "manifest_invalid_physical_geometry" in _only(result)[
        "unverifiable_reasons"
    ]
    assert result["physical_claim_summary"]["credited_count"] == 0
    assert result["physical_claim_summary"]["all_targets_claimed"] is False


def test_raw_event_mutation_does_not_mutate_already_returned_trace() -> None:
    manifest = _manifest()
    event = _event(manifest)
    result = _evaluate(manifest, [event])
    before = deepcopy(result)
    event["requested_target"]["value"] = "changed"
    assert result == before
