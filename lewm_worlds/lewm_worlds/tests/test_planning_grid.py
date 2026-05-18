"""Tests for :mod:`lewm_worlds.planning_grid`."""

from __future__ import annotations

import math

import pytest

from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import (
    InflatedOccupancyGrid,
    best_free_yaw,
    safe_standoff_xys,
)


def _wall(object_id: str, cx: float, cy: float, sx: float, sy: float, yaw: float = 0.0) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind="wall",
        center_xyz_m=(cx, cy, 0.4),
        size_xyz_m=(sx, sy, 0.8),
        yaw_rad=yaw,
        material_id="wall_interior",
    )


def _landmark(object_id: str, cx: float, cy: float, sx: float = 0.3, sy: float = 0.3) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind="landmark",
        center_xyz_m=(cx, cy, 0.15),
        size_xyz_m=(sx, sy, 0.3),
        yaw_rad=0.0,
        material_id="landmark_red",
    )


def _manifest(
    *,
    walls: tuple[BoxObject, ...] = (),
    landmarks: tuple[BoxObject, ...] = (),
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-3.0, -3.0), (3.0, 3.0)),
) -> SceneManifest:
    return SceneManifest(
        scene_id="grid_toy",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=bounds,
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=1.0),),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10
        ),
        walls=walls,
    )


def test_open_world_is_all_free():
    grid = InflatedOccupancyGrid(_manifest(), cell_size_m=0.1, inflation_m=0.2)
    assert grid.free_mask.all()
    assert grid.is_free((0.0, 0.0))
    assert grid.is_free((-2.5, -2.5))


def test_inflation_blocks_cells_within_radius_of_wall():
    walls = (_wall("w", cx=0.0, cy=1.0, sx=2.0, sy=0.1),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls), cell_size_m=0.05, inflation_m=0.20
    )
    # Wall's south face sits at y=0.95. Inflation 0.20 → free starts at y<=0.75.
    assert not grid.is_free((0.0, 0.90))
    assert not grid.is_free((0.0, 0.80))
    assert grid.is_free((0.0, 0.70))


def test_landmark_treated_as_obstacle_by_default():
    landmarks = (_landmark("lm", cx=0.0, cy=0.0),)
    grid = InflatedOccupancyGrid(
        _manifest(landmarks=landmarks), cell_size_m=0.05, inflation_m=0.20
    )
    assert not grid.is_free((0.0, 0.0))
    # 0.15 m beacon half-width + 0.20 m inflation = 0.35 m clear zone.
    assert not grid.is_free((0.30, 0.0))
    assert grid.is_free((0.40, 0.0))


def test_landmark_passable_when_disabled():
    landmarks = (_landmark("lm", cx=0.0, cy=0.0),)
    grid = InflatedOccupancyGrid(
        _manifest(landmarks=landmarks),
        cell_size_m=0.05,
        inflation_m=0.20,
        treat_landmarks_as_obstacles=False,
    )
    assert grid.is_free((0.0, 0.0))


def test_astar_straight_line_open_world():
    grid = InflatedOccupancyGrid(_manifest(), cell_size_m=0.1, inflation_m=0.1)
    path = grid.astar((-1.0, 0.0), (1.0, 0.0))
    assert path is not None
    assert path.waypoints_xy[-1] == pytest.approx((1.05, 0.05), abs=0.1)
    # Path stays roughly straight along y=0.
    for x, y in path.waypoints_xy:
        assert abs(y) < 0.1


def test_astar_routes_around_wall():
    # Vertical wall from y=-0.5 to y=0.5 at x=0, leaving a gap above/below.
    walls = (_wall("w", cx=0.0, cy=0.0, sx=0.1, sy=1.0),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, bounds=((-2.0, -2.0), (2.0, 2.0))),
        cell_size_m=0.05,
        inflation_m=0.20,
    )
    path = grid.astar((-1.5, 0.0), (1.5, 0.0))
    assert path is not None
    # The robot can't go through x=0 along y=0, so the path must detour to
    # |y| >= wall_half + inflation = 0.5 + 0.20 = 0.70.
    max_abs_y = max(abs(p[1]) for p in path.waypoints_xy)
    assert max_abs_y >= 0.70 - 1e-3


def test_astar_returns_none_when_goal_walled_off():
    # Box enclosure around the goal — no path in.
    walls = (
        _wall("n", cx=1.0, cy=1.3, sx=0.8, sy=0.1),
        _wall("s", cx=1.0, cy=0.7, sx=0.8, sy=0.1),
        _wall("e", cx=1.4, cy=1.0, sx=0.1, sy=0.7),
        _wall("w", cx=0.6, cy=1.0, sx=0.1, sy=0.7),
    )
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, bounds=((-2.0, -2.0), (3.0, 3.0))),
        cell_size_m=0.05,
        inflation_m=0.15,
    )
    path = grid.astar((-1.0, -1.0), (1.0, 1.0))
    assert path is None


def test_astar_snaps_start_into_free_space():
    # Start drifts into the inflated zone of a wall but is recoverable.
    walls = (_wall("w", cx=0.0, cy=1.0, sx=2.0, sy=0.1),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls), cell_size_m=0.05, inflation_m=0.20
    )
    # 0.88 is just inside the inflated zone but ~0.2 m from the body radius.
    path = grid.astar((0.0, 0.85), (0.0, 0.0))
    assert path is not None
    assert path.waypoints_xy[-1][1] == pytest.approx(0.025, abs=0.1)


def test_path_clear_of_inflated_obstacles():
    # Every cell on the returned path must itself be free.
    walls = (
        _wall("h", cx=0.0, cy=0.0, sx=2.0, sy=0.1),
        _wall("v", cx=0.0, cy=0.5, sx=0.1, sy=1.0),
    )
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, bounds=((-2.0, -2.0), (2.0, 2.0))),
        cell_size_m=0.05,
        inflation_m=0.20,
    )
    path = grid.astar((-1.5, -1.5), (1.5, 1.5))
    assert path is not None
    for x, y in path.waypoints_xy:
        assert grid.is_free((x, y)), f"waypoint {(x, y)} is not free"


def test_has_free_line():
    walls = (_wall("w", cx=0.0, cy=0.0, sx=0.1, sy=1.0),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls), cell_size_m=0.05, inflation_m=0.20
    )
    assert not grid.has_free_line((-1.0, 0.0), (1.0, 0.0))
    assert grid.has_free_line((-1.0, -1.5), (1.0, -1.5))


def test_nearest_free_returns_self_when_already_free():
    grid = InflatedOccupancyGrid(_manifest(), cell_size_m=0.1, inflation_m=0.1)
    snapped = grid.nearest_free((0.0, 0.0))
    assert snapped is not None
    assert math.hypot(snapped[0], snapped[1]) < 0.15


def test_nearest_free_escapes_inflated_zone():
    walls = (_wall("w", cx=0.0, cy=1.0, sx=2.0, sy=0.1),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls), cell_size_m=0.05, inflation_m=0.20
    )
    snapped = grid.nearest_free((0.0, 0.85), max_radius_m=0.5)
    assert snapped is not None
    assert grid.is_free(snapped)
    assert abs(snapped[0]) < 0.2 and snapped[1] < 0.80


def test_best_free_yaw_aligns_with_open_corridor():
    # Corridor along the y axis: north and south walls leave a strip of free
    # space at x=0. The longest free ray from origin should point ±y.
    walls = (
        _wall("east", cx=0.4, cy=0.0, sx=0.1, sy=4.0),
        _wall("west", cx=-0.4, cy=0.0, sx=0.1, sy=4.0),
    )
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, bounds=((-2.0, -2.5), (2.0, 2.5))),
        cell_size_m=0.05,
        inflation_m=0.15,
    )
    yaw, dist = best_free_yaw(grid, (0.0, 0.0), n_rays=16, max_m=2.0)
    # Ray must lie along ±y, i.e. yaw close to ±pi/2.
    assert min(abs(yaw - math.pi / 2.0), abs(yaw + math.pi / 2.0)) < 1e-3
    assert dist >= 1.5


def test_best_free_yaw_returns_zero_when_xy_not_free():
    walls = (_wall("w", cx=0.0, cy=0.0, sx=2.0, sy=2.0),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls), cell_size_m=0.05, inflation_m=0.20
    )
    yaw, dist = best_free_yaw(grid, (0.0, 0.0), n_rays=8, max_m=1.0)
    assert yaw == 0.0
    assert dist == 0.0


def test_best_free_yaw_picks_longer_arm_in_t_junction():
    # T with a long stem south of origin and short east/west arms across.
    # Longest free ray should point south.
    walls = (
        # north wall just above origin (caps the T)
        _wall("north", cx=0.0, cy=0.4, sx=2.5, sy=0.1),
        # corridor walls on the long stem (south of origin)
        _wall("stem_east", cx=0.3, cy=-1.2, sx=0.1, sy=2.5),
        _wall("stem_west", cx=-0.3, cy=-1.2, sx=0.1, sy=2.5),
        # short arm caps on either side
        _wall("east_cap", cx=1.2, cy=0.0, sx=0.1, sy=0.7),
        _wall("west_cap", cx=-1.2, cy=0.0, sx=0.1, sy=0.7),
        # arm walls (north sides of the arms)
        _wall("east_arm_n", cx=0.7, cy=0.4, sx=1.0, sy=0.1),
        _wall("west_arm_n", cx=-0.7, cy=0.4, sx=1.0, sy=0.1),
        # arm walls (south sides of the arms)
        _wall("east_arm_s", cx=0.7, cy=-0.4, sx=0.6, sy=0.1),
        _wall("west_arm_s", cx=-0.7, cy=-0.4, sx=0.6, sy=0.1),
    )
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, bounds=((-2.0, -3.0), (2.0, 1.5))),
        cell_size_m=0.05,
        inflation_m=0.15,
    )
    yaw, dist = best_free_yaw(grid, (0.0, 0.0), n_rays=32, max_m=2.5)
    # Longest free ray runs along -y (south), so yaw ≈ -pi/2.
    assert abs(yaw + math.pi / 2.0) < 0.25
    assert dist >= 1.5


def test_safe_standoff_xys_keeps_only_free_candidates():
    # Beacon at origin, wall to the east (blocks one cardinal standoff).
    landmarks = (_landmark("lm", cx=0.0, cy=0.0),)
    walls = (_wall("east", cx=1.0, cy=0.0, sx=0.1, sy=2.0),)
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, landmarks=landmarks),
        cell_size_m=0.05,
        inflation_m=0.20,
    )
    candidates = safe_standoff_xys(grid, (0.0, 0.0), standoff_m=0.85, n_candidates=4)
    # 4 candidates at angles 0, 90, 180, 270. The east one (0°, x=+0.85)
    # is wedged between beacon and wall; the other three are clear.
    assert len(candidates) == 3
    for x, y in candidates:
        assert grid.is_free((x, y))
        assert math.hypot(x, y) == pytest.approx(0.85, abs=1e-6)


def test_safe_standoff_xys_returns_empty_when_beacon_walled_in():
    landmarks = (_landmark("lm", cx=0.0, cy=0.0),)
    walls = (
        _wall("n", cx=0.0, cy=0.8, sx=2.0, sy=0.1),
        _wall("s", cx=0.0, cy=-0.8, sx=2.0, sy=0.1),
        _wall("e", cx=0.8, cy=0.0, sx=0.1, sy=2.0),
        _wall("w", cx=-0.8, cy=0.0, sx=0.1, sy=2.0),
    )
    grid = InflatedOccupancyGrid(
        _manifest(walls=walls, landmarks=landmarks),
        cell_size_m=0.05,
        inflation_m=0.20,
    )
    candidates = safe_standoff_xys(grid, (0.0, 0.0), standoff_m=0.85, n_candidates=8)
    assert candidates == ()
