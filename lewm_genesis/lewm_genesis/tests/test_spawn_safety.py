"""Unit tests for the rollout spawn-safety gate.

Covers :func:`lewm_genesis.rollout._select_spawn_pose`. The helper is pure
(no Genesis dependency), so we drive it with a small hand-built
:class:`~lewm_worlds.scene_graph.SceneGraph` and check that the floor is
enforced for both randomized and fixed-spawn paths.
"""

from __future__ import annotations

import random

import pytest

import math

from lewm_genesis.rollout import _select_spawn_pose
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid
from lewm_worlds.scene_graph import SceneGraph


def _corridor_manifest(spawn_xy: tuple[float, float]) -> SceneManifest:
    """Three-cell straight corridor; spawn xy is parameterised so tests can
    pin it inside the safe centerline or shove it up against a wall."""

    cell = 1.0
    wall_t = 0.2
    wall_h = 0.8
    nodes = tuple(
        GraphNode(
            node_id=i,
            center_xy_m=(i * cell, 0.0),
            width_m=cell - wall_t,
            tags=("spawn",) if i == 0 else (),
        )
        for i in range(3)
    )
    edges = tuple(
        GraphEdge(source=i, target=i + 1, width_m=cell - wall_t, traversable=True)
        for i in range(2)
    )
    walls = (
        BoxObject(
            object_id="north_wall",
            kind="wall",
            center_xyz_m=(cell, cell * 0.5, wall_h * 0.5),
            size_xyz_m=(3 * cell, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="south_wall",
            kind="wall",
            center_xyz_m=(cell, -cell * 0.5, wall_h * 0.5),
            size_xyz_m=(3 * cell, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    return SceneManifest(
        scene_id="spawn_safety_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.5, -1.5), (3.5, 1.5)),
        spawn=SpawnSpec(
            xyz_m=(float(spawn_xy[0]), float(spawn_xy[1]), 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        walls=walls,
    )


def test_fixed_spawn_clear_manifest_uses_manifest_pose():
    graph = SceneGraph(_corridor_manifest(spawn_xy=(0.0, 0.0)))
    rng = random.Random(0)
    xyz, quat, _cell, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=0.30,
        rng=rng,
    )
    assert source == "manifest"
    assert xyz == (0.0, 0.0, 0.375)
    assert quat == (1.0, 0.0, 0.0, 0.0)


def test_fixed_spawn_wedged_manifest_falls_back_to_safe_cell():
    # 0.40 m corridor half-width minus 0.05 m offset puts the manifest spawn
    # 0.05 m from the south wall — well below any sensible floor.
    graph = SceneGraph(_corridor_manifest(spawn_xy=(1.0, -0.35)))
    rng = random.Random(0)
    floor = 0.30
    xyz, _quat, _cell, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(1.0, -0.35, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=floor,
        rng=rng,
    )
    assert source == "safety_fallback"
    assert graph.clearance_to_walls((xyz[0], xyz[1])) >= floor - 1e-6


def test_randomized_spawn_honors_clearance_floor():
    graph = SceneGraph(_corridor_manifest(spawn_xy=(0.0, 0.0)))
    rng = random.Random(0)
    floor = 0.30
    xyz, _quat, cell_id, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=True,
        clearance_floor_m=floor,
        rng=rng,
    )
    assert source == "randomized"
    assert 0 <= cell_id < graph.n_nodes
    assert graph.clearance_to_walls((xyz[0], xyz[1])) >= floor - 1e-6


def test_no_scene_graph_passes_manifest_through():
    rng = random.Random(0)
    xyz, quat, cell_id, source = _select_spawn_pose(
        scene_graph=None,
        manifest_xyz=(1.5, -0.2, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=True,
        clearance_floor_m=0.45,
        rng=rng,
    )
    assert source == "manifest"
    assert cell_id == -1
    assert xyz == (1.5, -0.2, 0.375)
    assert quat == (1.0, 0.0, 0.0, 0.0)


def test_floor_below_cell_clearance_keeps_manifest_when_clear():
    # Cell-center clearance in this corridor is 0.40 m. A 0.20 m floor is
    # trivially satisfied at (1.0, 0.0), so the helper must not perturb it.
    graph = SceneGraph(_corridor_manifest(spawn_xy=(1.0, 0.0)))
    rng = random.Random(0)
    xyz, _quat, _cell, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(1.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=0.20,
        rng=rng,
    )
    assert source == "manifest"
    assert xyz == (1.0, 0.0, 0.375)


def test_no_safe_cell_anywhere_falls_back_to_sampler_fallback():
    # Force a floor higher than any cell-center clearance (0.40 m). The
    # sampler then returns the manifest cell as its own fallback. The gate
    # still labels the call ``safety_fallback`` because the manifest pose
    # failed the floor.
    graph = SceneGraph(_corridor_manifest(spawn_xy=(1.0, -0.35)))
    rng = random.Random(0)
    xyz, _quat, cell_id, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(1.0, -0.35, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=5.0,
        rng=rng,
    )
    assert source == "safety_fallback"
    # The sampler's fallback locates whichever cell the manifest xy snaps to.
    assert 0 <= cell_id < graph.n_nodes
    assert xyz[2] == pytest.approx(0.375)


def test_spawn_pose_honors_restriction_list():
    graph = SceneGraph(_corridor_manifest(spawn_xy=(0.0, 0.0)))
    rng = random.Random(0)
    # Cells are 0, 1, 2. Restrict to only cell 2.
    restrict = (2,)
    xyz, _quat, cell_id, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=True,
        clearance_floor_m=0.20,
        rng=rng,
        restrict_to_cells=restrict,
    )
    assert source == "randomized"
    assert cell_id == 2
    # Check that the chosen xyz is indeed in cell 2.
    assert graph.locate((xyz[0], xyz[1])).cell_id == 2


def _y_corridor_manifest(spawn_xy: tuple[float, float]) -> SceneManifest:
    """Two parallel walls leaving a long free corridor along the y axis.

    The manifest spawn quat is identity (facing +x), so without yaw
    alignment the robot would face into a wall.
    """

    walls = (
        BoxObject(
            object_id="east_wall",
            kind="wall",
            center_xyz_m=(0.4, 0.0, 0.4),
            size_xyz_m=(0.1, 4.0, 0.8),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="west_wall",
            kind="wall",
            center_xyz_m=(-0.4, 0.0, 0.4),
            size_xyz_m=(0.1, 4.0, 0.8),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    return SceneManifest(
        scene_id="y_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.0, -2.5), (1.0, 2.5)),
        spawn=SpawnSpec(
            xyz_m=(float(spawn_xy[0]), float(spawn_xy[1]), 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=0.6),),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        walls=walls,
    )


def _yaw_from_quat(quat: tuple[float, float, float, float]) -> float:
    return 2.0 * math.atan2(float(quat[3]), float(quat[0]))


def test_fixed_spawn_yaw_aligned_when_facing_wall():
    # Manifest spawn faces +x (yaw=0) but the corridor runs ±y, so yaw
    # alignment should rotate the spawn to face ±pi/2.
    manifest = _y_corridor_manifest(spawn_xy=(0.0, 0.0))
    graph = SceneGraph(manifest)
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.15)
    rng = random.Random(0)
    _xyz, quat, _cell, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=0.20,
        rng=rng,
        planning_grid=grid,
        yaw_align_min_free_ray_m=0.6,
    )
    assert source == "manifest"
    yaw = _yaw_from_quat(quat)
    assert min(abs(yaw - math.pi / 2.0), abs(yaw + math.pi / 2.0)) < 0.25


def test_fixed_spawn_yaw_kept_when_already_clear():
    # Manifest spawn already faces +y (yaw=pi/2), down the corridor.
    manifest = _y_corridor_manifest(spawn_xy=(0.0, 0.0))
    graph = SceneGraph(manifest)
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.15)
    spawn_quat = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    rng = random.Random(0)
    _xyz, quat, _cell, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=spawn_quat,
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=0.20,
        rng=rng,
        planning_grid=grid,
        yaw_align_min_free_ray_m=0.6,
    )
    assert source == "manifest"
    assert quat == spawn_quat


def test_randomized_spawn_yaw_aligned_to_corridor():
    manifest = _y_corridor_manifest(spawn_xy=(0.0, 0.0))
    graph = SceneGraph(manifest)
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.15)
    # Make many draws; yaw should always end up pointing down the corridor
    # (±pi/2) because every other yaw has < 0.6 m of free space ahead.
    for seed in range(8):
        rng = random.Random(seed)
        _xyz, quat, _cell, source = _select_spawn_pose(
            scene_graph=graph,
            manifest_xyz=(0.0, 0.0, 0.375),
            manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            spawn_z_m=0.375,
            randomize=True,
            clearance_floor_m=0.20,
            rng=rng,
            planning_grid=grid,
            yaw_align_min_free_ray_m=0.6,
        )
        assert source == "randomized"
        yaw = _yaw_from_quat(quat)
        assert min(abs(yaw - math.pi / 2.0), abs(yaw + math.pi / 2.0)) < 0.3


def test_fixed_spawn_outside_restriction_list_falls_back():
    graph = SceneGraph(_corridor_manifest(spawn_xy=(0.0, 0.0)))
    rng = random.Random(0)
    # Manifest is in cell 0. Restrict to only cell 2.
    restrict = (2,)
    xyz, _quat, cell_id, source = _select_spawn_pose(
        scene_graph=graph,
        manifest_xyz=(0.0, 0.0, 0.375),
        manifest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        spawn_z_m=0.375,
        randomize=False,
        clearance_floor_m=0.20,
        rng=rng,
        restrict_to_cells=restrict,
    )
    # Should fall back because cell 0 is not in restrict.
    assert source == "safety_fallback"
    assert cell_id == 2
