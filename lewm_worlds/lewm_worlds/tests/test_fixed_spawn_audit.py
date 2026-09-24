"""Tests for the exact fixed-spawn benchmark geometry audit."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lewm_worlds.fixed_spawn_audit import (
    FixedSpawnAuditConfig,
    audit_fixed_spawn,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)


def _box(
    object_id: str,
    *,
    kind: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind=kind,
        center_xyz_m=(center_xy[0], center_xy[1], 0.3),
        size_xyz_m=(size_xy[0], size_xy[1], 0.6),
        yaw_rad=0.0,
        material_id="test",
    )


def _manifest(
    *,
    scene_id: str = "audit_toy",
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    walls: tuple[BoxObject, ...] = (),
    landmarks: tuple[BoxObject, ...] = (),
) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="audit_test",
        difficulty_tier="test",
        topology_seed=7,
        visual_seed=8,
        physics_seed=9,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(spawn_xy[0], spawn_xy[1], 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(
            GraphNode(node_id=0, center_xy_m=spawn_xy, width_m=1.0),
        ),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=20.0,
            min_camera_clearance_m=0.10,
        ),
        split="candidate",
        walls=walls,
    )


@pytest.fixture
def audit_config() -> FixedSpawnAuditConfig:
    return FixedSpawnAuditConfig(
        cell_size_m=0.05,
        coverage_cell_size_m=0.10,
        body_radius_m=0.20,
        claim_radius_m=0.60,
        standoff_m=0.50,
        standoff_candidates=16,
    )


def test_open_scene_beacon_is_reachable_from_exact_spawn(
    audit_config: FixedSpawnAuditConfig,
) -> None:
    beacon = _box(
        "beacon",
        kind="landmark",
        center_xy=(0.9, 0.0),
        size_xy=(0.2, 0.2),
    )

    report = audit_fixed_spawn(
        _manifest(landmarks=(beacon,)),
        config=audit_config,
    )

    assert report.fully_reachable
    assert report.failure_reason == ""
    assert report.spawn_xy_m == (0.0, 0.0)
    assert report.spawn_is_body_clear
    assert report.beacons[0].reachable
    assert report.beacons[0].claim_reachable
    assert report.beacons[0].preferred_standoff_reachable
    assert report.beacons[0].reachable_claim_cell_count > 0
    assert report.beacons[0].reachable_navigable_standoff_count > 0
    assert report.all_beacons_have_preferred_standoff
    assert report.reachable_cell_count > report.coverage_reachable_cell_count > 0
    assert "reachable_cells" not in report.to_dict()
    assert "coverage_reachable_cells" not in report.to_dict()


def test_full_partition_makes_beacon_unreachable(
    audit_config: FixedSpawnAuditConfig,
) -> None:
    partition = _box(
        "partition",
        kind="wall",
        center_xy=(0.0, 0.0),
        size_xy=(0.1, 4.0),
    )
    beacon = _box(
        "beacon",
        kind="landmark",
        center_xy=(1.0, 0.0),
        size_xy=(0.2, 0.2),
    )

    report = audit_fixed_spawn(
        _manifest(
            spawn_xy=(-1.0, 0.0),
            walls=(partition,),
            landmarks=(beacon,),
        ),
        config=audit_config,
    )

    assert not report.fully_reachable
    assert not report.beacons[0].reachable
    assert report.failure_reason == "beacons_unreachable_from_fixed_spawn:beacon"


def test_preferred_standoff_is_diagnostic_not_reachability_gate(
    audit_config: FixedSpawnAuditConfig,
) -> None:
    beacon = _box(
        "beacon",
        kind="landmark",
        center_xy=(0.9, 0.0),
        size_xy=(0.2, 0.2),
    )
    diagnostic_only_failure = replace(
        audit_config,
        minimum_navigable_corridor_width_m=100.0,
    )

    report = audit_fixed_spawn(
        _manifest(landmarks=(beacon,)),
        config=diagnostic_only_failure,
    )

    assert report.fully_reachable
    assert report.failure_reason == ""
    assert report.beacons[0].claim_reachable
    assert not report.beacons[0].preferred_standoff_reachable
    assert not report.all_beacons_have_preferred_standoff


def test_unsafe_exact_spawn_is_rejected_without_snapping(
    audit_config: FixedSpawnAuditConfig,
) -> None:
    near_spawn = _box(
        "near_spawn",
        kind="wall",
        center_xy=(0.2, 0.0),
        size_xy=(0.1, 1.0),
    )
    beacon = _box(
        "beacon",
        kind="landmark",
        center_xy=(1.0, 1.0),
        size_xy=(0.2, 0.2),
    )

    report = audit_fixed_spawn(
        _manifest(walls=(near_spawn,), landmarks=(beacon,)),
        config=audit_config,
    )

    assert not report.spawn_is_body_clear
    assert report.reachable_cell_count == 0
    assert report.coverage_reachable_cell_count == 0
    assert report.failure_reason == "fixed_spawn_lacks_body_clearance"


def test_default_geometry_matches_generalization_contract_values() -> None:
    config = FixedSpawnAuditConfig()

    assert config.cell_size_m == 0.05
    assert config.coverage_cell_size_m == 0.10
    assert config.body_radius_m == 0.20
    assert config.claim_radius_m == 1.20
    assert config.standoff_m == 1.05
    assert config.standoff_candidates == 32
    assert config.minimum_navigable_corridor_width_m == 0.50
    assert config.connectivity == 8
    assert not config.allow_diagonal_corner_cutting
    assert config.require_line_of_sight
