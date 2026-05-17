"""Tests for the privileged :class:`SceneGraph` accessor."""

from __future__ import annotations

import math
import random

import pytest

from lewm_worlds.families import build_family_manifest
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.scene_graph import SceneGraph, bearing_from_to, wrap_angle_pi


def _toy_corridor_manifest() -> SceneManifest:
    """Three-cell straight corridor with walls on both sides.

    Layout (x increases right):

        +---+---+---+
        | 0 - 1 - 2 |
        +---+---+---+
    """

    cell = 1.0
    wall_t = 0.2
    wall_h = 0.8
    nodes = tuple(
        GraphNode(node_id=i, center_xy_m=(i * cell, 0.0), width_m=cell - wall_t, tags=("spawn",) if i == 0 else ())
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
        scene_id="toy_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.5, -1.5), (3.5, 1.5)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10
        ),
        walls=walls,
    )


def test_bfs_distance_walks_traversable_edges():
    graph = SceneGraph(_toy_corridor_manifest())
    assert graph.bfs_distance(0, 0) == 0
    assert graph.bfs_distance(0, 1) == 1
    assert graph.bfs_distance(0, 2) == 2


def test_bfs_distance_returns_none_for_unreachable():
    manifest = _toy_corridor_manifest()
    # Mark the only edge between 0 and 1 as non-traversable.
    edges = list(manifest.graph_edges)
    edges[0] = GraphEdge(source=0, target=1, width_m=0.8, traversable=False)
    blocked = SceneManifest(
        scene_id=manifest.scene_id,
        family=manifest.family,
        difficulty_tier=manifest.difficulty_tier,
        topology_seed=manifest.topology_seed,
        visual_seed=manifest.visual_seed,
        physics_seed=manifest.physics_seed,
        world_bounds_xy_m=manifest.world_bounds_xy_m,
        spawn=manifest.spawn,
        graph_nodes=manifest.graph_nodes,
        graph_edges=tuple(edges),
        obstacles=manifest.obstacles,
        landmarks=manifest.landmarks,
        camera_constraints=manifest.camera_constraints,
        walls=manifest.walls,
    )
    graph = SceneGraph(blocked)
    assert graph.bfs_distance(0, 1) is None
    assert graph.bfs_distance(0, 2) is None


def test_next_waypoint_returns_first_hop():
    graph = SceneGraph(_toy_corridor_manifest())
    assert graph.next_waypoint(0, 2) == 1
    assert graph.next_waypoint(2, 0) == 1
    assert graph.next_waypoint(0, 0) is None


def test_locate_picks_nearest_cell():
    graph = SceneGraph(_toy_corridor_manifest())
    hit = graph.locate((0.05, 0.0))
    assert hit.cell_id == 0
    assert hit.distance_m == pytest.approx(0.05)
    hit = graph.locate((1.95, 0.05))
    assert hit.cell_id == 2


def test_clearance_to_walls_handles_inside_outside():
    graph = SceneGraph(_toy_corridor_manifest())
    # Standing on the centerline midway through the corridor: nearest wall
    # is the north or south wall, ~0.5 - wall_thickness/2 = 0.4 m away.
    assert graph.clearance_to_walls((1.0, 0.0)) == pytest.approx(0.4, abs=1e-3)
    # Standing on the wall surface gives ~0 clearance.
    assert graph.clearance_to_walls((1.0, 0.4)) == pytest.approx(0.0, abs=1e-3)


def test_sample_spawn_pose_returns_reachable_cell():
    graph = SceneGraph(_toy_corridor_manifest())
    rng = random.Random(0)
    xyz, quat, cell_id = graph.sample_spawn_pose(rng, clearance_floor_m=0.1)
    assert 0 <= cell_id < graph.n_nodes
    # quaternion is wxyz around z-axis (yaw-only)
    assert quat[0] ** 2 + quat[3] ** 2 == pytest.approx(1.0, abs=1e-3)
    assert quat[1] == 0.0 and quat[2] == 0.0


def test_sample_spawn_pose_falls_back_when_no_cell_passes_clearance():
    graph = SceneGraph(_toy_corridor_manifest())
    rng = random.Random(0)
    # Set the clearance floor higher than any cell's clearance (0.4 m).
    xyz, quat, cell_id = graph.sample_spawn_pose(rng, clearance_floor_m=5.0, max_attempts=10)
    # Fallback returns the manifest spawn cell.
    assert cell_id == 0


def test_dead_ends_for_real_maze_scene():
    manifest = build_family_manifest(
        scene_seed=42, family="small_enclosed_maze", split=None, difficulty_tier=None
    )
    graph = SceneGraph(manifest)
    dead_ends = graph.dead_end_cells()
    # Spawning corner usually has degree 1 in small DFS mazes; just make sure
    # we get at least one dead-end and it's a valid cell id.
    assert dead_ends, "small mazes should produce at least one dead-end"
    for cell in dead_ends:
        assert 0 <= cell < graph.n_nodes


def test_wrap_angle_pi():
    assert wrap_angle_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle_pi(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    assert wrap_angle_pi(0.5) == pytest.approx(0.5)


def test_bearing_from_to():
    assert bearing_from_to((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert bearing_from_to((0.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi * 0.5)
    assert bearing_from_to((1.0, 1.0), (0.0, 0.0)) == pytest.approx(-math.pi * 0.75)
