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


def test_bfs_distance_respects_transit_blocked():
    graph = SceneGraph(_toy_corridor_manifest())
    # Normal path 0 -> 2 is 0 -> 1 -> 2 (2 hops)
    assert graph.bfs_distance(0, 2) == 2
    # If 1 is blocked for transit, 0 -> 2 becomes unreachable in this straight corridor
    assert graph.bfs_distance(0, 2, transit_blocked=frozenset([1])) is None
    # But we can still reach 1 if it is the goal
    assert graph.bfs_distance(0, 1, transit_blocked=frozenset([1])) == 1
    # And we can still leave 0 if it is blocked
    assert graph.bfs_distance(0, 1, transit_blocked=frozenset([0])) == 1


def test_shortest_path_respects_transit_blocked():
    graph = SceneGraph(_toy_corridor_manifest())
    assert graph.shortest_path(0, 2) == (1, 2)
    assert graph.shortest_path(0, 2, transit_blocked=frozenset([1])) == ()
    assert graph.shortest_path(0, 1, transit_blocked=frozenset([1])) == (1,)


def test_beacon_cells_set_property():
    graph = SceneGraph(_landmark_corridor_manifest())
    # In _landmark_corridor_manifest, cell 2 has the beacon
    assert 2 in graph.beacon_cells_set
    assert 0 not in graph.beacon_cells_set
    assert 1 not in graph.beacon_cells_set


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


def test_nav_blocked_cells_includes_low_clearance():
    # In _toy_corridor_manifest, cells are at (0,0), (1,0), (2,0).
    # Walls are at (0, 0.5), (1, 0.5), etc. half-height is 0.5.
    # So center (0,0) has 0.5m clearance to wall at y=0.5.
    graph = SceneGraph(_toy_corridor_manifest())
    # 0.5m clearance > 0.20m threshold, so no cells should be blocked by clearance.
    assert len(graph.nav_blocked_cells) == 0

    # Now create a manifest with a wall very close to cell 1.
    m_base = _toy_corridor_manifest()
    # Add an obstacle at (1, 0.15) with size (0.1, 0.1).
    # Distance from (1,0) to box at (1, 0.15) with half-y 0.05 is 0.15 - 0.05 = 0.10m.
    import dataclasses
    m = dataclasses.replace(m_base, obstacles=(
        BoxObject(
            object_id="narrow_wall",
            kind="wall",
            center_xyz_m=(1.0, 0.15, 0.0),
            size_xyz_m=(0.1, 0.1, 1.0),
            yaw_rad=0.0,
            material_id="wall",
        ),
    ))
    graph2 = SceneGraph(m)
    # Cell 1 (at (1,0)) now has 0.10m clearance < 0.20m threshold.
    assert 1 in graph2.nav_blocked_cells
    # Path 0 -> 2 should now be blocked
    assert graph2.bfs_distance(0, 2, transit_blocked=graph2.nav_blocked_cells) is None


def _landmark_corridor_manifest() -> SceneManifest:
    """Corridor with a beacon sitting at the far cell, used for LOS tests."""

    base = _toy_corridor_manifest()
    landmark = BoxObject(
        object_id="goal_beacon",
        kind="landmark",
        center_xyz_m=(2.0, 0.0, 0.5),
        size_xyz_m=(0.30, 0.30, 1.0),
        yaw_rad=0.0,
        material_id="landmark_red",
    )
    return SceneManifest(
        scene_id=base.scene_id,
        family=base.family,
        difficulty_tier=base.difficulty_tier,
        topology_seed=base.topology_seed,
        visual_seed=base.visual_seed,
        physics_seed=base.physics_seed,
        world_bounds_xy_m=base.world_bounds_xy_m,
        spawn=base.spawn,
        graph_nodes=base.graph_nodes,
        graph_edges=base.graph_edges,
        obstacles=base.obstacles,
        landmarks=(landmark,),
        camera_constraints=base.camera_constraints,
        walls=base.walls,
    )


def test_has_line_of_sight_treats_landmarks_as_occluders():
    graph = SceneGraph(_landmark_corridor_manifest())
    # Beacon sits between (1.0, 0.0) and (3.0, 0.0); a ray that pierces the
    # beacon's footprint must be reported as blocked.
    assert graph.has_line_of_sight((1.0, 0.0), (3.0, 0.0)) is False


def test_has_line_of_sight_exclude_landmark_xy_skips_self_occlusion():
    graph = SceneGraph(_landmark_corridor_manifest())
    # Without exclusion, the segment from (1.0, 0.0) → the beacon center
    # is blocked by the beacon itself; with the beacon excluded, it clears.
    assert graph.has_line_of_sight((1.0, 0.0), (2.0, 0.0)) is False
    assert (
        graph.has_line_of_sight(
            (1.0, 0.0), (2.0, 0.0), exclude_landmark_xy=(2.0, 0.0)
        )
        is True
    )


def test_maze_walls_block_los_between_non_traversable_cells():
    """Regression: every non-traversable grid pair must have a real wall.

    _maze_walls used to swap the size axes between horizontal and vertical
    neighbours, leaving thin strips through the cell centres while the
    actual gap between adjacent cells stayed open. The BFS reported "not
    connected" but the rendered scene was passable, so collectors steered
    straight through invisible walls.
    """

    for seed in (0, 7, 42):
        manifest = build_family_manifest(
            scene_seed=seed,
            family="small_enclosed_maze",
            split=None,
            difficulty_tier=None,
        )
        graph = SceneGraph(manifest)
        node_by_id = {node.node_id: node for node in manifest.graph_nodes}
        for edge in manifest.graph_edges:
            if edge.traversable:
                continue
            src = node_by_id[edge.source].center_xy_m
            dst = node_by_id[edge.target].center_xy_m
            assert graph.has_line_of_sight(src, dst) is False, (
                f"seed={seed} non-traversable edge {edge.source}-{edge.target} "
                f"has open LOS between {src} and {dst} — wall is missing or "
                f"placed with the wrong orientation"
            )


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


def test_local_composite_motifs_do_not_place_landmark_on_spawn():
    """Regression: s_bend and slalom used to put a landmark at the spawn
    cell, which dropped the robot inside the 0.30m landmark footprint at
    reset and wedged it for the whole episode.
    """

    for seed in range(50):
        manifest = build_family_manifest(
            scene_seed=seed,
            family="local_composite_motifs",
            split=None,
            difficulty_tier=None,
        )
        spawn_xy = (manifest.spawn.xyz_m[0], manifest.spawn.xyz_m[1])
        for landmark in manifest.landmarks:
            lx, ly, _lz = landmark.center_xyz_m
            assert (lx, ly) != spawn_xy, (
                f"seed={seed} landmark {landmark.object_id} placed at spawn "
                f"xy={spawn_xy} — robot would respawn inside the beacon footprint"
            )


def test_wrap_angle_pi():
    assert wrap_angle_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle_pi(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    assert wrap_angle_pi(0.5) == pytest.approx(0.5)


def test_bearing_from_to():
    assert bearing_from_to((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert bearing_from_to((0.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi * 0.5)
    assert bearing_from_to((1.0, 1.0), (0.0, 0.0)) == pytest.approx(-math.pi * 0.75)
