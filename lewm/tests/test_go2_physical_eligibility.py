from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from lewm.benchmarks.generalization_protocol import (
    SceneDisjointManifests,
    build_hashed_scene_role_commitment,
    scene_id_sha256,
    scene_role_token,
)
from lewm.benchmarks.go2_physical_eligibility import (
    DirectionalSE2Lattice,
    PhysicalEligibilityConfig,
    audit_physical_scene_eligibility,
    load_observed_max_directional_policy,
    policy_from_geometry_contract,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm.planning.oriented_footprint import (
    DirectionalSupportFootprint,
    ManifestDirectionalFootprintFeasibility,
    Pose2D,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "config/go2_generalization_geometry_v2.json"


def _box(
    object_id: str,
    *,
    kind: str,
    xy: tuple[float, float],
    size_xy: tuple[float, float] = (0.15, 0.15),
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind=kind,
        center_xyz_m=(xy[0], xy[1], 0.5),
        size_xyz_m=(size_xy[0], size_xy[1], 1.0),
        yaw_rad=0.0,
        material_id=object_id,
    )


def _open_manifest(*, yaw_rad: float = 0.0, walls: tuple[BoxObject, ...] = ()) -> SceneManifest:
    return SceneManifest(
        scene_id="physical_eligibility_toy",
        family="go2_deployment_medium_maze",
        difficulty_tier="test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-2.5, -2.5), (2.5, 2.5)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.375),
            quat_wxyz=(math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)),
        ),
        graph_nodes=(
            GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=5.0),
            GraphNode(node_id=1, center_xy_m=(-1.6, -1.6), width_m=1.0),
            GraphNode(node_id=2, center_xy_m=(1.6, -1.6), width_m=1.0),
            GraphNode(node_id=3, center_xy_m=(-1.6, 1.6), width_m=1.0),
            GraphNode(node_id=4, center_xy_m=(1.6, 1.6), width_m=1.0),
        ),
        graph_edges=(),
        obstacles=(),
        landmarks=tuple(
            _box(f"landmark_{color}", kind="landmark", xy=xy)
            for color, xy in (
                ("red", (-1.6, -1.6)),
                ("blue", (1.6, -1.6)),
                ("green", (-1.6, 1.6)),
                ("yellow", (1.6, 1.6)),
            )
        ),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=20.0,
            min_camera_clearance_m=0.1,
        ),
        split="candidate",
        walls=walls,
    )


@pytest.fixture(scope="module")
def geometry_and_policy():
    geometry = load_geometry_contract(GEOMETRY, repository_root=ROOT)
    policy = policy_from_geometry_contract(geometry, repository_root=ROOT)
    return geometry, policy


def test_geometry_bound_policy_is_observed_max_and_content_addressed(
    geometry_and_policy,
) -> None:
    geometry, policy = geometry_and_policy

    assert policy.profile_name == "observed_max_plus_margin"
    assert policy.content_sha256 == (
        geometry.swept_footprint.directional_policy_content_sha256
    )
    assert policy.content_sha256 in policy.source_path.name
    assert policy.footprint.maximum_vertex_radius_m == pytest.approx(
        geometry.swept_footprint.maximum_vertex_radius_m
    )


def test_policy_loader_rejects_tampered_content(
    geometry_and_policy,
    tmp_path: Path,
) -> None:
    geometry, policy = geometry_and_policy
    payload = json.loads(policy.source_path.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    tampered = tmp_path / policy.source_path.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_observed_max_directional_policy(
            tampered,
            expected_content_sha256=policy.content_sha256,
            expected_policy_id=str(geometry.swept_footprint.directional_policy_id),
        )


def test_all_four_claim_anchors_have_staged_se2_witnesses(
    geometry_and_policy,
) -> None:
    _geometry, policy = geometry_and_policy
    config = PhysicalEligibilityConfig(
        cell_size_m=0.10,
        yaw_bins=8,
        rotation_subsamples=2,
        claim_radius_m=1.0,
        preferred_standoff_m=0.85,
        mask_validation_samples=12,
    )

    report = audit_physical_scene_eligibility(
        _open_manifest(),
        policy=policy,
        config=config,
    )

    assert report.eligible
    assert report.spawn_clear_at_actual_yaw
    assert report.spawn_snaps_to_lattice
    assert len(report.claim_anchors) == 4
    assert all(anchor.reachable for anchor in report.claim_anchors)
    assert all(anchor.anchor_has_line_of_sight for anchor in report.claim_anchors)
    assert all(anchor.shortest_staged_action_count is not None for anchor in report.claim_anchors)
    assert all(
        sum(anchor.shortest_staged_action_counts.values())
        == anchor.shortest_staged_action_count
        for anchor in report.claim_anchors
        if anchor.shortest_staged_action_counts is not None
        and anchor.shortest_staged_action_count is not None
    )
    assert len(report.sha256) == 64


def test_spawn_clearance_uses_manifest_actual_yaw(
    geometry_and_policy,
) -> None:
    _geometry, policy = geometry_and_policy
    near_rear = _box(
        "near_rear",
        kind="wall",
        xy=(0.405, 0.0),
        size_xy=(0.02, 0.10),
    )
    config = PhysicalEligibilityConfig(
        cell_size_m=0.20,
        yaw_bins=8,
        rotation_subsamples=2,
        claim_radius_m=1.0,
        preferred_standoff_m=0.85,
        mask_validation_samples=0,
    )

    forward = audit_physical_scene_eligibility(
        _open_manifest(yaw_rad=0.0, walls=(near_rear,)),
        policy=policy,
        config=config,
    )
    reversed_spawn = audit_physical_scene_eligibility(
        _open_manifest(yaw_rad=math.pi, walls=(near_rear,)),
        policy=policy,
        config=config,
    )

    assert forward.spawn_clear_at_actual_yaw
    assert not reversed_spawn.spawn_clear_at_actual_yaw
    assert "actual_yaw_spawn_not_polygon_clear" in reversed_spawn.failure_reason


def test_scene_role_commitment_contains_only_reproducible_hash_tokens() -> None:
    def entry(scene_id: str) -> dict[str, object]:
        return {"scene_id": scene_id}

    manifests = SceneDisjointManifests(
        development={
            "benchmark_id": "v4-test",
            "geometry_contract_sha256": "a" * 64,
            "train_scenes": [entry("train-secret")],
            "validation_scenes": [entry("development-secret")],
            "excluded_scenes": [],
        },
        sealed_test={
            "benchmark_id": "v4-test",
            "commitment_sha256": "b" * 64,
            "scenes": [entry("sealed-secret")],
        },
    )

    commitment = build_hashed_scene_role_commitment(manifests)
    serialized = json.dumps(commitment, sort_keys=True)

    assert "train-secret" not in serialized
    assert "development-secret" not in serialized
    assert "sealed-secret" not in serialized
    assert commitment["roles"]["development"] == [
        scene_role_token(
            "development-secret",
            role="development",
            benchmark_id="v4-test",
        )
    ]
    assert commitment["roles"]["sealed_test"] == [
        scene_role_token(
            "sealed-secret",
            role="sealed_test",
            benchmark_id="v4-test",
        )
    ]
    assert commitment["scene_id_sha256_by_role"]["development"] == [
        scene_id_sha256("development-secret")
    ]
    assert commitment["scene_id_sha256_by_role"]["sealed_test"] == [
        scene_id_sha256("sealed-secret")
    ]
    assert len(
        commitment["set_sha256_by_role"]["sealed_test"]["scene_id_sha256"]
    ) == 64
    assert len(commitment["content_sha256"]) == 64


def test_lattice_rejects_unaligned_translation_and_mid_edge_collision() -> None:
    tiny = DirectionalSupportFootprint.from_directional_support(
        {0.0: 0.02, 90.0: 0.02, 180.0: 0.02, 270.0: 0.02}
    )
    thin_wall = _box(
        "thin_mid_edge_wall",
        kind="wall",
        xy=(0.10, 0.0),
        size_xy=(0.01, 0.10),
    )
    manifest = replace(
        _open_manifest(walls=(thin_wall,)),
        landmarks=(),
    )
    checker = ManifestDirectionalFootprintFeasibility(manifest, tiny)
    lattice = DirectionalSE2Lattice(
        checker,
        cell_size_m=0.20,
        yaw_bins=16,
        rotation_subsamples=2,
        maximum_translation_substep_m=0.025,
    )
    cardinal = lattice.snap_pose(Pose2D(0.0, 0.0, 0.0))
    assert cardinal is not None
    cardinal_neighbors = tuple(lattice._neighbors(cardinal))
    assert all(action != "forward" for _state, action in cardinal_neighbors)

    diagonal_half_heading = (1, cardinal[1], cardinal[2])
    assert lattice._state_free(*diagonal_half_heading)
    unaligned_neighbors = tuple(lattice._neighbors(diagonal_half_heading))
    assert {action for _state, action in unaligned_neighbors} <= {
        "turn_left",
        "turn_right",
    }
