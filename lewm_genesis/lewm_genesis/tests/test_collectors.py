"""Tests for collector policies and the episode scheduler.

These tests don't require Genesis at runtime — they exercise the pure-Python
parts of the §13 collection mix against a tiny scene built by hand and the
real primitive registry from ``config/go2_primitive_registry.yaml``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lewm_genesis.collectors import (
    DEFAULT_COLLECTION_MIX,
    EpisodeScheduler,
    FrontierTeacher,
    OUExploration,
    PrimitiveCurriculum,
    RecoveryCurriculum,
    build_default_policies,
)
from lewm_genesis.collectors import RouteTeacher as _RouteTeacherImpl


def RouteTeacher(*args, **kwargs):  # noqa: N802 — shim mirrors class name
    """Test-only shim: defaults to the legacy body-frame visibility gate.

    The production default for ``landmark_camera_forward_offset_m`` is the
    Go2 camera mount (0.326 m forward of body center), which puts the
    beacon ~15° wider in the camera frame than in the body frame at the
    claim envelope. Most of these tests were written against the older
    body-frame check and use body-on-beacon arrival poses that the
    camera-aware gate (correctly) rejects. The shim lets us keep those
    tests as-is while production picks up the camera-aware behavior;
    individual tests that exercise the new gate pass an explicit
    ``landmark_camera_forward_offset_m`` to override.
    """

    kwargs.setdefault("landmark_camera_forward_offset_m", 0.0)
    return _RouteTeacherImpl(*args, **kwargs)
from lewm_genesis.collectors.base import (
    BlockChoice,
    EnvObservation,
    primitive_toward_bearing,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.scene_graph import SceneGraph


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "config" / "go2_primitive_registry.yaml"


@pytest.fixture(scope="module")
def registry() -> PrimitiveRegistry:
    if not REGISTRY_PATH.is_file():
        pytest.skip(f"primitive registry not found at {REGISTRY_PATH}")
    return PrimitiveRegistry.from_yaml(REGISTRY_PATH)


def _grid_corridor_manifest() -> SceneManifest:
    """4-cell straight corridor (nodes 0..3 along +x) with side walls."""

    cell = 1.0
    wall_t = 0.2
    wall_h = 0.8
    # Increase corridor width to 1.2m (from 0.8m) to ensure the 0.3m landmark
    # plus 0.2m inflation on each side (0.7m total) doesn't block it.
    width = 1.2
    nodes = tuple(
        GraphNode(
            node_id=i,
            center_xy_m=(i * cell, 0.0),
            width_m=width,
            tags=("spawn",) if i == 0 else (),
        )
        for i in range(4)
    )
    edges = tuple(
        GraphEdge(source=i, target=i + 1, width_m=width, traversable=True)
        for i in range(3)
    )
    walls = (
        BoxObject(
            object_id="north_wall",
            kind="wall",
            center_xyz_m=(1.5 * cell, (width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(4 * cell, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="south_wall",
            kind="wall",
            center_xyz_m=(1.5 * cell, -(width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(4 * cell, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    landmarks = (
        BoxObject(
            object_id="goal_landmark",
            kind="landmark",
            center_xyz_m=(3.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
        BoxObject(
            object_id="secondary_landmark",
            kind="landmark",
            center_xyz_m=(0.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
    )
    return SceneManifest(
        scene_id="grid_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.0, -1.025), (4.0, 0.975)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10
        ),
        walls=walls,
    )


def _narrow_endpoint_landmark_manifest() -> SceneManifest:
    """A narrow motif where spawn-beacon yaw drift can snap outside the route."""

    width = 0.504
    nodes = tuple(
        GraphNode(
            node_id=i,
            center_xy_m=xy,
            width_m=width,
            tags=("spawn",) if i == 0 else (),
        )
        for i, xy in enumerate(
            (
                (0.0, 0.0),
                (0.0, 0.8),
                (0.0, 1.6),
                (0.8, 1.6),
                (1.6, 1.6),
                (1.6, 2.4),
                (1.6, 3.2),
                (1.6, 4.0),
                (2.4, 4.0),
                (3.2, 4.0),
            )
        )
    )
    edges = tuple(
        GraphEdge(source=i, target=i + 1, width_m=width, traversable=True)
        for i in range(9)
    )
    wall_specs = (
        ((0.4, 0.0), (0.296, 0.8400000000000001)),
        ((-0.4, 0.0), (0.296, 0.8400000000000001)),
        ((0.0, -0.4), (0.8400000000000001, 0.296)),
        ((0.4, 0.8), (0.296, 0.8400000000000001)),
        ((-0.4, 0.8), (0.296, 0.8400000000000001)),
        ((0.0, 2.0), (0.8400000000000001, 0.296)),
        ((-0.4, 1.6), (0.296, 0.8400000000000001)),
        ((0.8, 2.0), (0.8400000000000001, 0.296)),
        ((0.8, 1.2), (0.8400000000000001, 0.296)),
        ((2.0, 1.6), (0.296, 0.8400000000000001)),
        ((1.6, 1.2), (0.8400000000000001, 0.296)),
        ((2.0, 2.4), (0.296, 0.8400000000000001)),
        ((1.2, 2.4), (0.296, 0.8400000000000001)),
        ((2.0, 3.2), (0.296, 0.8400000000000001)),
        ((1.2, 3.2), (0.296, 0.8400000000000001)),
        ((1.6, 4.4), (0.8400000000000001, 0.296)),
        ((1.2, 4.0), (0.296, 0.8400000000000001)),
        ((2.4, 4.4), (0.8400000000000001, 0.296)),
        ((2.4, 3.6), (0.8400000000000001, 0.296)),
        ((3.6, 4.0), (0.296, 0.8400000000000001)),
        ((3.2, 4.4), (0.8400000000000001, 0.296)),
        ((3.2, 3.6), (0.8400000000000001, 0.296)),
    )
    walls = tuple(
        BoxObject(
            object_id=f"wall_motif_{idx:04d}",
            kind="wall",
            center_xyz_m=(xy[0], xy[1], 0.4),
            size_xyz_m=(size[0], size[1], 0.8),
            yaw_rad=0.0,
            material_id="wall_perimeter",
        )
        for idx, (xy, size) in enumerate(wall_specs)
    )
    landmarks = (
        BoxObject(
            object_id="landmark_00_landmark_red",
            kind="landmark",
            center_xyz_m=(3.2, 4.0, 0.44),
            size_xyz_m=(0.3, 0.3, 0.88),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
        BoxObject(
            object_id="landmark_01_landmark_blue",
            kind="landmark",
            center_xyz_m=(0.0, 0.0, 0.44),
            size_xyz_m=(0.3, 0.3, 0.88),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
    )
    return SceneManifest(
        scene_id="narrow_endpoint_landmark",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-6.096, -6.096), (6.096, 6.096)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.12,
            near_m=0.08,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        walls=walls,
    )


def _observation(
    *,
    env_idx: int = 0,
    base_xy: tuple[float, float] = (0.0, 0.0),
    yaw: float = 0.0,
    cell_id: int = 0,
    clearance: float = 1.0,
    last_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0),
    episode_id: int = 1,
    episode_step: int = 0,
    block_idx: int = 0,
) -> EnvObservation:
    return EnvObservation(
        env_idx=env_idx,
        base_xy_world=base_xy,
        base_yaw_world=yaw,
        current_cell_id=cell_id,
        nearest_cell_distance_m=0.0,
        clearance_to_walls_m=clearance,
        last_executed_cmd=last_cmd,
        episode_id=episode_id,
        episode_step=episode_step,
        block_idx_in_episode=block_idx,
    )


# ---------------------------------------------------------------------------
# primitive_toward_bearing
# ---------------------------------------------------------------------------


def test_primitive_toward_bearing_quadrants():
    # Bands: |err| <= π/6 → forward; <= π/2 → arc; else yaw. Translation
    # is preferred — any turn up to 90° should keep the body moving via
    # an arc primitive instead of stopping to yaw.
    assert primitive_toward_bearing(heading_error_rad=0.0) == "forward_medium"
    assert primitive_toward_bearing(heading_error_rad=math.pi / 6) == "forward_medium"
    assert primitive_toward_bearing(heading_error_rad=math.pi / 4) == "arc_left"
    assert primitive_toward_bearing(heading_error_rad=-math.pi / 4) == "arc_right"
    assert primitive_toward_bearing(heading_error_rad=math.pi - 0.01) == "yaw_left"
    assert primitive_toward_bearing(heading_error_rad=-math.pi + 0.01) == "yaw_right"


# ---------------------------------------------------------------------------
# PrimitiveCurriculum
# ---------------------------------------------------------------------------


def test_primitive_curriculum_only_emits_trainable_primitives(registry):
    collector = PrimitiveCurriculum(registry, n_envs=1)
    rng = np.random.default_rng(0)
    seen: set[str] = set()
    for _ in range(200):
        choice = collector.on_block(observation=_observation(), scene=None, rng=rng)
        assert choice.command_source == "primitive_curriculum"
        assert isinstance(choice.requested_block, np.ndarray)
        assert choice.requested_block.shape == (registry.block_size, 3)
        seen.add(choice.primitive_name)
    trainable = set(registry.trainable_velocity_names())
    assert seen.issubset(trainable)
    # Should at least cover several primitives within 200 draws.
    assert len(seen) >= 3


# ---------------------------------------------------------------------------
# RouteTeacher
# ---------------------------------------------------------------------------


def test_route_teacher_picks_landmark_goal_and_steers_toward_waypoint(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    # At cell 0 facing +x, the secondary landmark sits at the spawn cell so
    # it is claimed immediately; the remaining goal landmark is at cell 3.
    choice = collector.on_block(
        observation=_observation(base_xy=(0.0, 0.0), yaw=0.0, cell_id=0),
        scene=scene,
        rng=rng,
    )
    assert choice.command_source == "route_teacher"
    assert choice.route_target_id == 3  # landmark cell
    # Robot starts inside the 'secondary_landmark' (inflated), so it must
    # dodge it to reach cell 3.
    assert choice.primitive_name == "arc_right"


def test_route_teacher_yaws_when_facing_wrong_way(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # Facing -x while goal sits at +x → heading error ~π → yaw primitive.
    choice = collector.on_block(
        observation=_observation(base_xy=(0.0, 0.0), yaw=math.pi, cell_id=0),
        scene=scene,
        rng=rng,
    )
    assert choice.primitive_name.startswith("yaw_")


def test_route_teacher_finishes_after_visiting_each_landmark_once(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # First block: the spawn-cell beacon is auto-claimed and the teacher
    # targets the remaining landmark at cell 3.
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3
    # Arrive at cell 3 facing the beacon — both landmarks now claimed, no
    # further beacons to chase.
    arrival = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(3.0, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert arrival.primitive_name == "hold"
    assert arrival.route_target_id == -1


def test_route_teacher_reuses_sticky_path_after_bad_snap_replan(registry):
    scene = SceneGraph(_narrow_endpoint_landmark_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(
            base_xy=(0.0, 0.0),
            cell_id=scene.locate((0.0, 0.0)).cell_id,
            clearance=scene.clearance_to_walls((0.0, 0.0)),
        ),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 9

    # If the base drifts slightly south at the spawn landmark, the nearest
    # free grid cell is in the exterior component while the previous route
    # path remains close and viable.
    drift_xy = (-0.021, -0.175)
    collector._blocks_since_plan[0] = collector._replan_every_blocks
    recovered = collector.on_block(
        observation=_observation(
            base_xy=drift_xy,
            yaw=0.2,
            cell_id=scene.locate(drift_xy).cell_id,
            clearance=scene.clearance_to_walls(drift_xy),
            last_cmd=(0.0, 0.0, 0.45),
            block_idx=1,
        ),
        scene=scene,
        rng=rng,
    )

    assert recovered.route_target_id == 9
    assert recovered.primitive_name != "hold"
    assert 9 not in collector._unreachable_beacons[0]


def test_route_teacher_targets_next_unvisited_landmark(registry):
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=0.8, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(1.0, 0.0), width_m=0.8),
        GraphNode(node_id=2, center_xy_m=(2.0, 0.0), width_m=0.8),
        GraphNode(node_id=3, center_xy_m=(1.0, 1.0), width_m=0.8),
    )
    edges = (
        GraphEdge(source=0, target=1, width_m=0.8, traversable=True),
        GraphEdge(source=1, target=2, width_m=0.8, traversable=True),
        GraphEdge(source=1, target=3, width_m=0.8, traversable=True),
    )
    landmarks = (
        BoxObject(
            object_id="east_landmark",
            kind="landmark",
            center_xyz_m=(2.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
        BoxObject(
            object_id="north_landmark",
            kind="landmark",
            center_xyz_m=(1.0, 1.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
    )
    scene = SceneGraph(
        SceneManifest(
            scene_id="branched_landmarks",
            family="test",
            difficulty_tier="test",
            topology_seed=0,
            visual_seed=0,
            physics_seed=0,
            world_bounds_xy_m=((-1.5, -1.525), (3.5, 1.475)),
            spawn=SpawnSpec(
                xyz_m=(0.0, 0.0, 0.375),

                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            graph_nodes=nodes,
            graph_edges=edges,
            obstacles=(),
            landmarks=landmarks,
            camera_constraints=CameraValidityConstraints(
                min_wall_thickness_m=0.08,
                near_m=0.05,
                far_m=200.0,
                min_camera_clearance_m=0.10,
            ),
            walls=(),
        )
    )
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    # A* on the inflated grid picks the cheaper beacon first; either order
    # is fine. What matters is that after claiming one, the teacher pivots
    # to the other instead of repeatedly chasing the same one.
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id in {2, 3}
    first_target = first.route_target_id

    # Simulate arrival at the first beacon facing it.
    arrival_xy = (2.0, 0.0) if first_target == 2 else (1.0, 1.0)
    arrival_yaw = 0.0 if first_target == 2 else math.pi / 2
    _ = collector.on_block(
        observation=_observation(
            cell_id=first_target, base_xy=arrival_xy, yaw=arrival_yaw
        ),
        scene=scene,
        rng=rng,
    )

    # Next block from the same pose: the teacher must pivot to the other
    # beacon (or report no further reachable beacon).
    after_goal = collector.on_block(
        observation=_observation(
            cell_id=first_target, base_xy=arrival_xy, yaw=arrival_yaw
        ),
        scene=scene,
        rng=rng,
    )
    other = 3 if first_target == 2 else 2
    assert after_goal.route_target_id in {other, -1}


def test_route_teacher_does_not_resume_after_unreachable_beacons(registry):
    # Two beacons separated by a solid wall — the remote one is in a
    # disconnected free region. After claiming the local beacon, the
    # teacher must report -1 instead of looping on the unreachable one.
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=0.8, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(1.0, 0.0), width_m=0.8),
        GraphNode(node_id=2, center_xy_m=(10.0, 0.0), width_m=0.8),
        GraphNode(node_id=3, center_xy_m=(11.0, 0.0), width_m=0.8),
    )
    edges = (
        GraphEdge(source=0, target=1, width_m=0.8, traversable=True),
        GraphEdge(source=2, target=3, width_m=0.8, traversable=True),
    )
    landmarks = (
        BoxObject(
            object_id="local_landmark",
            kind="landmark",
            center_xyz_m=(1.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
        BoxObject(
            object_id="remote_landmark",
            kind="landmark",
            center_xyz_m=(11.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
    )
    # Wall spans the full world height so no free path exists around it.
    walls = (
        BoxObject(
            object_id="divider",
            kind="wall",
            center_xyz_m=(5.5, 0.0, 0.5),
            size_xyz_m=(0.2, 4.4, 1.0),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    scene = SceneGraph(
        SceneManifest(
            scene_id="disconnected_landmarks",
            family="test",
            difficulty_tier="test",
            topology_seed=0,
            visual_seed=0,
            physics_seed=0,
            world_bounds_xy_m=((-1.5, -1.525), (3.5, 1.475)),
            spawn=SpawnSpec(
                xyz_m=(0.0, 0.0, 0.375),

                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            graph_nodes=nodes,
            graph_edges=edges,
            obstacles=(),
            landmarks=landmarks,
            camera_constraints=CameraValidityConstraints(
                min_wall_thickness_m=0.08,
                near_m=0.05,
                far_m=200.0,
                min_camera_clearance_m=0.10,
            ),
            walls=walls,
        )
    )
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 1

    # Arrive at the local beacon facing it.
    _ = collector.on_block(
        observation=_observation(cell_id=1, base_xy=(1.0, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    done = collector.on_block(
        observation=_observation(cell_id=1, base_xy=(1.0, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert done.route_target_id == -1


def test_route_teacher_does_not_arrive_outside_claim_envelope(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    # Standoff just past the inflated beacon perimeter; claim envelope =
    # standoff + claim_radius = 0.50 m around the beacon centre.
    collector = RouteTeacher(
        registry, n_envs=1, landmark_standoff_m=0.40, landmark_claim_radius_m=0.10
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3

    # 2.20 m → 0.80 m from beacon at (3.0, 0.0); outside the 0.50 m
    # envelope → still chasing.
    not_arrived = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(2.20, 0.0)), scene=scene, rng=rng
    )
    assert not_arrived.route_target_id == 3


def test_route_teacher_requires_landmark_in_fov_to_claim_arrival(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1, landmark_fov_deg=60.0)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # First block: lock onto the landmark goal at cell 3.
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 3

    # Robot stands inside the claim envelope (0.05 m from beacon) but is
    # facing -x while the beacon sits at +x. Beacon is outside the 60°
    # FOV → arrival must not fire and the goal must remain pinned.
    facing_wall = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(2.95, 0.0), yaw=math.pi),
        scene=scene,
        rng=rng,
    )
    assert facing_wall.route_target_id == 3


def test_route_teacher_claims_landmark_when_facing_with_clear_los(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1, landmark_fov_deg=60.0)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 3
    # At the goal cell, facing +x toward the beacon, no occluder between →
    # arrival should fire and the teacher must not re-target a visited beacon.
    arrived = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(3.0, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert arrived.primitive_name == "hold"
    assert arrived.route_target_id == -1


def test_route_teacher_does_not_claim_landmark_behind_wall(registry):
    # Two-cell scene with a thin pillar wall along x = 1.0 that fully blocks
    # the segment from (0,0) to (2,0). The landmark sits at cell 1's centre.
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=0.8, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(2.0, 0.0), width_m=0.8),
    )
    edges = (GraphEdge(source=0, target=1, width_m=0.8, traversable=True),)
    walls = (
        BoxObject(
            object_id="pillar",
            kind="wall",
            center_xyz_m=(1.0, 0.0, 0.5),
            size_xyz_m=(0.2, 4.0, 1.0),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    landmarks = (
        BoxObject(
            object_id="goal_landmark",
            kind="landmark",
            center_xyz_m=(2.0, 0.0, 0.5),
            size_xyz_m=(0.3, 0.3, 1.0),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
    )
    manifest = SceneManifest(
        scene_id="pillar_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.0, -1.0), (3.0, 1.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10
        ),
        walls=walls,
    )
    scene = SceneGraph(manifest)
    # Sanity: the segment from cell 0 to the landmark should be blocked.
    assert scene.has_line_of_sight((0.0, 0.0), (2.0, 0.0)) is False
    assert scene.landmark_xy_for_cell(1) == (2.0, 0.0)

    collector = RouteTeacher(
        registry,
        n_envs=1,
        landmark_fov_deg=60.0,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # Pillar wall blocks the only graph edge → A* finds no path to a free
    # standoff around the beacon, so it is reported unreachable.
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == -1
    assert first.primitive_name == "hold"


def test_route_teacher_arcs_for_moderate_heading_error_when_seeking_beacon(registry):
    # The teacher no longer suppresses arc primitives on the beacon approach.
    # A moderate (45°) heading error should fall into the arc band so the
    # robot keeps moving while reorienting — the old "yaw-then-forward"
    # cycle produced visible hesitation throughout long routes.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    choice = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0), yaw=math.pi / 4),
        scene=scene,
        rng=rng,
    )
    assert choice.route_target_id == 3
    assert choice.primitive_name in {"arc_left", "arc_right"}


def test_route_teacher_yaws_to_face_open_beacon(registry):
    # Near the standoff of an *open* beacon (wide corridor / room), an arc
    # just circles the beacon at standoff radius and never locks the claim
    # pose (the orbiting-failure cause). The teacher should yaw in place to
    # face the beacon. The grid_corridor fixture is 1.2 m wide, so the
    # standoff clearance is well above the narrow-corridor threshold.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3
    standoff_xy = collector._goal_standoff_xy[0]
    assert standoff_xy is not None

    obs = _observation(cell_id=2, base_xy=standoff_xy, yaw=0.75 * math.pi)
    # Sanity: the beacon neighborhood is open (above the narrow threshold),
    # so the open-beacon branch applies.
    assert obs.clearance_to_walls_m >= 0.45
    choice = collector.on_block(observation=obs, scene=scene, rng=rng)
    assert choice.route_target_id == 3
    assert choice.primitive_name in ("yaw_left", "yaw_right")


def test_route_teacher_rejects_stale_standoff_and_replans(registry):
    # A geometrically valid standoff can still be a poor approach for the
    # locomotion policy. If path progress stalls, the teacher should blacklist
    # that standoff for the episode and choose another one instead of looping.
    manifest = _grid_corridor_manifest()
    scene = SceneGraph(
        replace(
            manifest,
            walls=(),
            landmarks=(manifest.landmarks[0],),
            world_bounds_xy_m=((-1.5, -2.0), (4.5, 2.0)),
        )
    )
    collector = RouteTeacher(
        registry,
        n_envs=1,
        approach_retry_blocks=2,
        approach_retry_progress_m=0.01,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0), block_idx=0),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 3
    first_standoff = collector._goal_standoff_xy[0]
    assert first_standoff is not None
    first_key = collector._standoff_key(first_standoff)

    _ = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0), block_idx=1),
        scene=scene,
        rng=rng,
    )
    retried = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0), block_idx=2),
        scene=scene,
        rng=rng,
    )

    assert first_key in collector._rejected_standoffs[0][3]
    assert retried.route_target_id == 3
    assert collector._goal_standoff_xy[0] is not None
    assert collector._standoff_key(collector._goal_standoff_xy[0]) != first_key


def test_route_teacher_penalizes_standoff_when_locomotion_sticks(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0), block_idx=0),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 3
    standoff_xy = collector._goal_standoff_xy[0]
    assert standoff_xy is not None
    standoff_key = collector._standoff_key(standoff_xy)

    moving_cmd = (0.2, 0.0, 0.0)
    for block_idx in (1, 2):
        choice = collector.on_block(
            observation=_observation(
                cell_id=0,
                base_xy=(0.0, 0.0),
                last_cmd=moving_cmd,
                block_idx=block_idx,
            ),
            scene=scene,
            rng=rng,
        )
        assert choice.primitive_name != "backward"

    stuck = collector.on_block(
        observation=_observation(
            cell_id=0,
            base_xy=(0.0, 0.0),
            last_cmd=moving_cmd,
            block_idx=3,
        ),
        scene=scene,
        rng=rng,
    )

    assert stuck.primitive_name == "backward"
    assert standoff_key in collector._rejected_standoffs[0][3]


def test_route_teacher_uses_landmark_standoff_for_arrival(registry):
    # Arrival to a beacon goal must fire on the *standoff* radius, not on the
    # cell-width-derived threshold. The body should stop short of the mesh.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1, landmark_standoff_m=0.55)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3
    # At (2.5, 0.0) the robot is 0.50 m from the beacon at (3.0, 0.0) — well
    # inside the 0.55 m standoff and facing the beacon → arrival fires.
    arrived = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(2.5, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert arrived.primitive_name == "hold"
    assert arrived.route_target_id == -1


def test_route_teacher_requires_tight_landmark_claim_radius(registry):
    # Claim envelope is (standoff + claim_radius) around the beacon. A
    # tighter claim radius shrinks that envelope so an early-arriving
    # robot must keep chasing until it sits properly close to the beacon.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(
        registry,
        n_envs=1,
        landmark_standoff_m=0.40,
        landmark_claim_radius_m=0.05,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3

    # Envelope around beacon (3.0, 0) is 0.40 + 0.05 = 0.45 m. At (2.40, 0)
    # the robot is 0.60 m away → outside envelope, still chasing.
    early = collector.on_block(
        observation=_observation(cell_id=2, base_xy=(2.40, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert early.route_target_id == 3

    # At (2.60, 0) the robot is 0.40 m from the beacon — within envelope.
    claimed = collector.on_block(
        observation=_observation(cell_id=2, base_xy=(2.60, 0.0), yaw=0.0),
        scene=scene,
        rng=rng,
    )
    assert claimed.primitive_name == "hold"
    assert claimed.route_target_id == -1


def test_route_teacher_arrives_at_landmark_from_neighbor_cell(registry):
    # Landmarks with standoff > cell_half_width must be claimable from a
    # neighbor cell — the body should not be forced into the beacon cell.
    scene = SceneGraph(_grid_corridor_manifest())
    assert scene.locate((2.2, 0.0)).cell_id == 2
    assert scene.locate((2.6, 0.0)).cell_id == 3

    collector = RouteTeacher(registry, n_envs=1, landmark_standoff_m=0.85)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    obs = _observation(cell_id=0, base_xy=(0.0, 0.0))
    first = collector.on_block(observation=obs, scene=scene, rng=rng)
    assert first.route_target_id == 3

    obs_near = _observation(cell_id=2, base_xy=(2.2, 0.0), yaw=0.0)
    assert collector._landmark_visible(obs_near, 3, scene) is True

    # 0.80 m from beacon < (0.85 standoff + 0.20 claim) envelope → arrives.
    arrived = collector.on_block(observation=obs_near, scene=scene, rng=rng)
    assert arrived.primitive_name == "hold"
    assert arrived.route_target_id == -1


def test_route_teacher_emergency_backout_on_collision(registry):
    # The only reactive safety check that survived the rewrite: if the
    # body has driven into a wall (clearance ≤ 0.05 m), reverse out
    # before re-planning.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    choice = collector.on_block(
        observation=_observation(
            cell_id=0, base_xy=(0.0, 0.0), yaw=0.0, clearance=0.03
        ),
        scene=scene,
        rng=rng,
    )
    assert choice.primitive_name == "backward"
    assert choice.command_source == "route_teacher"


def test_route_teacher_camera_aware_gate_rejects_off_axis_beacon(registry):
    """The camera-aware gate must reject claims where the beacon is in the
    body FOV but outside the camera FOV — that's the visual-data gap that
    motivates the gate. Empirical setup: body-bearing 40° to beacon (passes
    a 90° body-frame gate) but camera-bearing ~58° (fails a 78° camera FOV)
    because the camera sits 0.326 m forward of the body.
    """

    scene = SceneGraph(_grid_corridor_manifest())
    # Production defaults: camera-aware gate at 78.3° FOV.
    collector = _RouteTeacherImpl(
        registry,
        n_envs=1,
        landmark_fov_deg=78.323,
        landmark_camera_forward_offset_m=0.326,
    )
    collector.on_episode_reset(0)
    # Body at (2.2, -0.65), yaw=0. Beacon at (3.0, 0.0).
    # Body→beacon: dx=0.8, dy=0.65, dist=1.03 m, bearing ≈ 39° → inside
    # the body 45° gate, would also be inside a 90° gate.
    # Camera at (2.526, -0.65): dx=0.474, dy=0.65, bearing ≈ 54° — outside
    # the camera's 39° half-FOV, so the claim must be rejected.
    obs = _observation(cell_id=2, base_xy=(2.20, -0.65), yaw=0.0)
    assert collector._landmark_visible(obs, 3, scene) is False


def test_route_teacher_camera_aware_gate_accepts_on_axis_beacon(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = _RouteTeacherImpl(
        registry,
        n_envs=1,
        landmark_fov_deg=78.323,
        landmark_camera_forward_offset_m=0.326,
    )
    collector.on_episode_reset(0)
    # Body at (2.2, 0), yaw=0, beacon at (3.0, 0). Beacon directly ahead;
    # camera-frame bearing is 0° — well inside FOV.
    obs = _observation(cell_id=2, base_xy=(2.20, 0.0), yaw=0.0)
    assert collector._landmark_visible(obs, 3, scene) is True


def test_route_teacher_dense_standoffs_only_for_loop_alias(registry):
    collector = _RouteTeacherImpl(registry, n_envs=1)
    base_scene = SceneGraph(_grid_corridor_manifest())
    loop_scene = SceneGraph(
        replace(_grid_corridor_manifest(), family="loop_alias_stress")
    )

    assert collector._standoff_candidate_count(base_scene) == 12
    assert collector._standoff_candidate_count(loop_scene) == 24


def test_route_teacher_relaxes_arrival_only_for_wedge_families(registry):
    collector = _RouteTeacherImpl(
        registry,
        n_envs=1,
        landmark_fov_deg=78.323,
        landmark_camera_forward_offset_m=0.326,
    )
    base_scene = SceneGraph(_grid_corridor_manifest())
    wedge_scene = SceneGraph(
        replace(_grid_corridor_manifest(), family="small_enclosed_maze")
    )

    obs = _observation(cell_id=2, base_xy=(2.02, -0.56), yaw=0.0)
    # Body range is just outside the standard 0.85 + 0.20 m envelope, and
    # camera-frame bearing is just outside the strict 78.323 degree FOV.
    assert collector._claim_check(0, obs, 3, base_scene) is False
    assert collector._claim_check(0, obs, 3, wedge_scene) is True
    assert collector._claim_radius_m(base_scene) == pytest.approx(0.20)
    assert collector._claim_radius_m(wedge_scene) == pytest.approx(0.35)


def test_route_teacher_relaxed_claim_telemetry_fires_only_on_wedge(registry):
    """Relaxed-claim counter must record wedge-family relaxations and
    only those: a base-family claim under the same pose stays at zero,
    while the wedge family logs a single event carrying the family name
    and the scene_id for downstream auditing.
    """

    collector = _RouteTeacherImpl(
        registry,
        n_envs=2,
        landmark_fov_deg=78.323,
        landmark_camera_forward_offset_m=0.326,
    )
    base_scene = SceneGraph(_grid_corridor_manifest())
    wedge_scene = SceneGraph(
        replace(_grid_corridor_manifest(), family="small_enclosed_maze")
    )
    # Pose where the strict envelope rejects but the relaxed envelope
    # accepts — same one used by the gating test above.
    obs = _observation(cell_id=2, base_xy=(2.02, -0.56), yaw=0.0)

    # env 0 — base family, claim must not fire and counter stays zero.
    collector.on_episode_reset(0)
    collector._goal_cell[0] = 3
    collector._mark_arrival_if_present(0, obs, base_scene)
    assert 3 not in collector._claimed[0]
    assert collector.relaxed_claim_count(0) == 0
    assert collector.relaxed_claim_events(0) == ()

    # env 1 — wedge family, claim fires and gets recorded as relaxed.
    collector.on_episode_reset(1)
    collector._goal_cell[1] = 3
    collector._mark_arrival_if_present(1, obs, wedge_scene)
    assert 3 in collector._claimed[1]
    assert collector.relaxed_claim_count(1) == 1
    events = collector.relaxed_claim_events(1)
    assert len(events) == 1
    assert events[0]["scene_family"] == "small_enclosed_maze"
    assert events[0]["beacon_cell"] == 3
    assert events[0]["env_idx"] == 1


def test_route_teacher_strict_claim_does_not_count_as_relaxed(registry):
    """A claim pose that already passes the strict gate must claim the
    beacon without bumping the relaxed-claim counter, even on a wedge
    family — otherwise the audit can't trust the counter as a relaxation
    metric.
    """

    collector = _RouteTeacherImpl(
        registry,
        n_envs=1,
        landmark_fov_deg=78.323,
        landmark_camera_forward_offset_m=0.326,
    )
    wedge_scene = SceneGraph(
        replace(_grid_corridor_manifest(), family="small_enclosed_maze")
    )

    # On-axis arrival: body in line with beacon at 0.85 m → inside the
    # strict envelope (FOV ~0, dist ≤ 1.05 m).
    obs = _observation(cell_id=2, base_xy=(2.20, 0.0), yaw=0.0)
    collector.on_episode_reset(0)
    collector._goal_cell[0] = 3
    collector._mark_arrival_if_present(0, obs, wedge_scene)
    assert 3 in collector._claimed[0]
    assert collector.relaxed_claim_count(0) == 0
    assert collector.relaxed_claim_events(0) == ()


def test_route_teacher_corridor_penalty_zero_on_wide_standoffs(registry):
    """Wide corridors (>= safe width) must not be penalized so families
    like open_obstacle_field and rough_local_dynamics see no scoring
    change from the corridor-aware standoff selector."""

    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    collector = _RouteTeacherImpl(registry, n_envs=1)
    grid = InflatedOccupancyGrid(
        _grid_corridor_manifest(), cell_size_m=0.05, inflation_m=0.20
    )
    # The grid_corridor manifest has 1.2 m corridor width — well above the
    # 0.50 m safe threshold. Probe the centerline at three cells.
    for xy in ((1.0, 0.0), (1.5, 0.0), (2.5, 0.0)):
        assert collector._standoff_corridor_penalty(grid, xy) == 0.0


def test_route_teacher_corridor_penalty_family_scoped(registry):
    """The penalty fires only for families listed in
    ``_STANDOFF_CORRIDOR_PENALTY_FAMILIES``; other families see zero
    regardless of corridor width. Earlier sweeps showed unscoped
    application regressing large/medium/visual yields, so the gate must
    stay enforced.
    """

    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    collector = _RouteTeacherImpl(registry, n_envs=1)
    # Use a manifest whose corridor is narrow enough to trigger the
    # penalty when scope check is bypassed — same construction as the
    # narrow-corridor test below.
    from lewm_worlds.manifest import (
        BoxObject,
        CameraValidityConstraints,
        GraphEdge,
        GraphNode,
        SceneManifest,
        SpawnSpec,
    )

    wall_t = 0.2
    wall_h = 0.8
    width = 0.50
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=width, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(1.0, 0.0), width_m=width, tags=()),
    )
    edges = (GraphEdge(source=0, target=1, width_m=width, traversable=True),)
    walls = (
        BoxObject(
            object_id="n",
            kind="wall",
            center_xyz_m=(0.5, (width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(2.0, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="s",
            kind="wall",
            center_xyz_m=(0.5, -(width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(2.0, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    base = SceneManifest(
        scene_id="narrow_corridor",
        family="medium_enclosed_maze",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-0.5, -0.5), (1.5, 0.5)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
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
    grid = InflatedOccupancyGrid(base, cell_size_m=0.05, inflation_m=0.20)
    out_of_scope_scene = SceneGraph(base)
    in_scope_scene = SceneGraph(replace(base, family="small_enclosed_maze"))

    # Out-of-scope family: penalty must be zero even for a narrow corridor.
    assert collector._standoff_corridor_penalty(grid, (0.5, 0.0), out_of_scope_scene) == 0.0
    # In-scope family: penalty fires.
    assert collector._standoff_corridor_penalty(grid, (0.5, 0.0), in_scope_scene) > 0.0
    # Backward-compat: omitting scene preserves the prior unscoped behavior
    # for direct callers (unit tests, etc.).
    assert collector._standoff_corridor_penalty(grid, (0.5, 0.0)) > 0.0


def test_route_teacher_corridor_penalty_grows_in_narrow_corridor(registry):
    """Narrow corridors trigger a penalty roughly proportional to the
    width deficit. The penalty must stay well below the rejected-standoff
    cost (1e6) so it influences ranking without acting as a hard reject.
    """

    from lewm_worlds.manifest import (
        BoxObject,
        CameraValidityConstraints,
        GraphEdge,
        GraphNode,
        SceneManifest,
        SpawnSpec,
    )
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    # 0.30 m corridor between two parallel walls along the x axis. After
    # 0.20 m inflation, the free band is 0.30 - 0.40 = ... negative,
    # which means *no* free cells. Use 0.50 m corridor so a 1-cell-wide
    # free band remains.
    wall_t = 0.2
    wall_h = 0.8
    width = 0.50
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=width, tags=("spawn",)),
        GraphNode(node_id=1, center_xy_m=(1.0, 0.0), width_m=width, tags=()),
    )
    edges = (GraphEdge(source=0, target=1, width_m=width, traversable=True),)
    walls = (
        BoxObject(
            object_id="n",
            kind="wall",
            center_xyz_m=(0.5, (width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(2.0, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="s",
            kind="wall",
            center_xyz_m=(0.5, -(width + wall_t) * 0.5, wall_h * 0.5),
            size_xyz_m=(2.0, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
    )
    manifest = SceneManifest(
        scene_id="narrow_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-0.5, -0.5), (1.5, 0.5)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
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
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.20)
    collector = _RouteTeacherImpl(registry, n_envs=1)

    # Sample at the corridor centerline — narrow axis is N-S (~0.10 m
    # free band after inflation).
    penalty = collector._standoff_corridor_penalty(grid, (0.5, 0.0))
    assert penalty > 0.0
    # Penalty should be well below the rejected-standoff penalty so it
    # influences ranking without rejecting outright.
    assert penalty < 1.0e6


def test_route_teacher_available_standoffs_falls_back_when_all_rejected(registry):
    """When every standoff for a beacon has been rejected via stuck-FSM,
    ``_available_standoffs`` falls back to the full list so the planner
    can still attempt the beacon — the ``_rejected_standoff_penalty``
    (1e6) keeps it at the bottom of the ranking unless no alternative
    exists, which lets controller-drift retries recover a wedge-prone
    beacon without permanently writing it off (an earlier attempt to
    blacklist the beacon outright regressed large/medium/visual yields).
    """

    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    scene = SceneGraph(_grid_corridor_manifest())
    grid = InflatedOccupancyGrid(
        _grid_corridor_manifest(), cell_size_m=0.05, inflation_m=0.20
    )
    collector = _RouteTeacherImpl(registry, n_envs=1)
    collector.on_episode_reset(0)

    beacon_cell = 3
    beacon_xy = (3.0, 0.0)
    standoffs = collector._standoffs_with_los(grid, beacon_xy, scene)
    assert standoffs, "test fixture must have at least one LOS-valid standoff"

    # Reject every standoff key for this beacon, mimicking repeated
    # stuck-FSM triggers that exhaust every approach option.
    for standoff in standoffs:
        collector._rejected_standoffs[0].setdefault(beacon_cell, set()).add(
            collector._standoff_key(standoff)
        )

    fallback = collector._available_standoffs(
        0, beacon_cell, grid, beacon_xy, scene
    )
    assert set(fallback) == set(standoffs)
    # Every returned standoff incurs the rejection penalty so the
    # planner prefers an unrejected alternative if any other beacon
    # exposes one.
    for s in fallback:
        assert collector._rejected_standoff_penalty(0, beacon_cell, s) == pytest.approx(
            1.0e6
        )


def test_route_teacher_legacy_arrival_when_visibility_disabled(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(
        registry,
        n_envs=1,
        require_landmark_visible=False,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
        scene=scene,
        rng=rng,
    )
    assert first.route_target_id == 3
    # Robot in goal cell facing -x — with visibility disabled the legacy
    # metric-only arrival should still fire, but should not revisit a beacon
    # already seen in the episode.
    arrival = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(3.0, 0.0), yaw=math.pi),
        scene=scene,
        rng=rng,
    )
    assert arrival.primitive_name == "hold"
    assert arrival.route_target_id == -1


def test_route_teacher_hold_when_no_reachable_goal(registry):
    # Scene with only one cell — no reachable goal exists.
    nodes = (
        GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=0.8, tags=("spawn",)),
    )
    manifest = SceneManifest(
        scene_id="single_cell",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.0, -1.0), (1.0, 1.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10
        ),
    )
    scene = SceneGraph(manifest)
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    choice = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert choice.primitive_name == "hold"
    assert choice.route_target_id == -1


# ---------------------------------------------------------------------------
# FrontierTeacher
# ---------------------------------------------------------------------------


def test_frontier_teacher_prefers_unvisited_cells(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = FrontierTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # Visit cell 0 a lot, then visit cell 1 once; the frontier should pick a
    # goal that is further down the corridor (cell 2 or 3), not cell 1.
    for _ in range(5):
        collector.on_block(
            observation=_observation(cell_id=0, base_xy=(0.0, 0.0)),
            scene=scene,
            rng=rng,
        )
    targets: list[int] = []
    for _ in range(30):
        choice = collector.on_block(
            observation=_observation(cell_id=1, base_xy=(1.0, 0.0)),
            scene=scene,
            rng=rng,
        )
        targets.append(choice.route_target_id)
    # Most picks should be the *least-visited* cells (2 or 3), not 0.
    assert sum(1 for t in targets if t in {2, 3}) >= sum(1 for t in targets if t == 0)


# ---------------------------------------------------------------------------
# RecoveryCurriculum
# ---------------------------------------------------------------------------


def test_recovery_curriculum_runs_approach_backout_pivot_cycle(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RecoveryCurriculum(
        registry,
        n_envs=1,
        approach_clearance_m=0.20,
        backout_clearance_m=0.50,
        backout_blocks=2,
        max_backout_blocks=4,
        pivot_blocks=2,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    # Approach phase while clear of walls: should emit a forward / arc / yaw
    # primitive (no backout yet).
    choice = collector.on_block(
        observation=_observation(clearance=1.0), scene=scene, rng=rng
    )
    assert choice.primitive_name != "backward"
    assert choice.command_source == "recovery"

    # Clearance falls below threshold → enter backout phase.
    backout_a = collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    )
    backout_b = collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    )
    assert backout_a.primitive_name == "backward"
    assert backout_b.primitive_name == "backward"

    # It keeps backing up after the minimum window while the wall is still too
    # close to provide useful egocentric feedback.
    still_backing = collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    )
    assert still_backing.primitive_name == "backward"

    # Once clear, pivot phase.
    pivot = collector.on_block(
        observation=_observation(clearance=0.60), scene=scene, rng=rng
    )
    assert pivot.primitive_name.startswith("yaw_")


def test_recovery_curriculum_default_clearance_is_body_aware(registry):
    # Default approach_clearance_m must sit below typical maze corridor
    # half-width (0.25–0.46 m) so the FSM can actually run end-to-end in
    # narrow corridors: at half-corridor clearance the FSM stays in
    # `approach` and the robot keeps walking forward, only flipping to
    # `backout` once the body is genuinely grazing a wall.
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RecoveryCurriculum(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    walking = collector.on_block(
        observation=_observation(clearance=0.50), scene=scene, rng=rng
    )
    assert walking.command_source == "recovery"
    assert walking.primitive_name != "backward"

    backout = collector.on_block(
        observation=_observation(clearance=0.10), scene=scene, rng=rng
    )
    assert backout.primitive_name == "backward"


def test_recovery_curriculum_backout_has_timeout(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RecoveryCurriculum(
        registry,
        n_envs=1,
        approach_clearance_m=0.20,
        backout_clearance_m=0.90,
        backout_blocks=2,
        max_backout_blocks=3,
    )
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)

    assert collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    ).primitive_name == "backward"
    assert collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    ).primitive_name == "backward"
    assert collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    ).primitive_name == "backward"
    pivot = collector.on_block(
        observation=_observation(clearance=0.05), scene=scene, rng=rng
    )
    assert pivot.primitive_name.startswith("yaw_")


# ---------------------------------------------------------------------------
# OU exploration
# ---------------------------------------------------------------------------


def test_ou_exploration_snaps_to_trainable_primitive(registry):
    collector = OUExploration(registry, n_envs=1, sigma=0.5)
    rng = np.random.default_rng(0)
    primitives = Counter()
    for _ in range(200):
        choice = collector.on_block(observation=_observation(), scene=None, rng=rng)
        primitives[choice.primitive_name] += 1
        assert choice.command_source == "ou_noise"
    trainable = set(registry.trainable_velocity_names())
    assert set(primitives).issubset(trainable)
    # Should cover several different primitives over 200 OU draws.
    assert len(primitives) >= 3


# ---------------------------------------------------------------------------
# EpisodeScheduler
# ---------------------------------------------------------------------------


def test_episode_scheduler_realises_share_table(registry):
    policies = build_default_policies(registry, n_envs=1)
    rng = np.random.default_rng(0)
    scheduler = EpisodeScheduler(
        policies=policies,
        shares=DEFAULT_COLLECTION_MIX,
        rng=rng,
        n_envs=1,
    )
    # Reassign 5000 times and verify the empirical distribution sits within
    # 3% of the configured shares for each policy.
    counts: Counter[str] = Counter()
    for _ in range(5000):
        name = scheduler.on_episode_reset(0)
        counts[name] += 1
    total = float(sum(counts.values()))
    for name, expected in DEFAULT_COLLECTION_MIX.items():
        observed = counts[name] / total
        assert abs(observed - expected) < 0.03, (
            f"{name}: expected ~{expected:.2f}, observed {observed:.3f}"
        )


def test_episode_scheduler_redistributes_when_policy_missing(registry):
    policies = {
        "route_teacher": build_default_policies(registry, n_envs=1)["route_teacher"],
        "primitive_curriculum": build_default_policies(registry, n_envs=1)["primitive_curriculum"],
    }
    rng = np.random.default_rng(0)
    # Full §13 mix references frontier/recovery/etc., but only two policies
    # are registered. The scheduler should renormalise over the available
    # subset rather than crash or emit unregistered names.
    scheduler = EpisodeScheduler(
        policies=policies,
        shares=DEFAULT_COLLECTION_MIX,
        rng=rng,
        n_envs=1,
    )
    seen: set[str] = set()
    for _ in range(500):
        seen.add(scheduler.on_episode_reset(0))
    assert seen == {"route_teacher", "primitive_curriculum"}


def test_episode_scheduler_falls_back_to_uniform_when_no_share_overlaps(registry):
    policies = {
        "route_teacher": build_default_policies(registry, n_envs=1)["route_teacher"],
    }
    rng = np.random.default_rng(0)
    # Shares table has no entry for route_teacher.
    scheduler = EpisodeScheduler(
        policies=policies,
        shares={"frontier": 1.0},
        rng=rng,
        n_envs=1,
    )
    # Should still produce a draw — uniform fallback.
    name = scheduler.on_episode_reset(0)
    assert name == "route_teacher"


def test_rollout_interlock_recognizes_route_like_policies(registry):
    from lewm_genesis.rollout import _is_route_like_policy

    assert _is_route_like_policy(RouteTeacher(registry, n_envs=1))
    assert _is_route_like_policy(
        RouteTeacher(
            registry,
            n_envs=1,
            revisit_after_arrival=True,
            name="loop_revisit",
        )
    )
    assert not _is_route_like_policy(FrontierTeacher(registry, n_envs=1))


def test_route_teacher_provides_spawn_restriction(registry):
    manifest = _grid_corridor_manifest()
    collector = RouteTeacher(registry, n_envs=1)

    class MockScene:
        def __init__(self, manifest):
            self.manifest = manifest
            self.scene_graph = SceneGraph(manifest)

    mock_scene = MockScene(manifest)
    cells = collector.spawn_restriction_cells(mock_scene)

    # Cells 0 and 3 each have a 0.3 m landmark sitting on the cell center,
    # so the inflated grid rejects them as spawn sites — the robot would
    # otherwise respawn inside the beacon footprint. Cells 1 and 2 are
    # clear corridor and remain canonical.
    assert cells is not None
    assert set(cells) == {1, 2}


def test_route_teacher_restricts_spawn_on_fragmented_scene(registry):
    # Two disconnected 2-cell corridors. Landmark in cell 3 (second corridor).
    m = _grid_corridor_manifest()
    # Remove edge between 1 and 2.
    new_edges = tuple(e for e in m.graph_edges if not (e.source == 1 and e.target == 2))
    # Add a wall between them.
    new_walls = list(m.walls)
    new_walls.append(
        BoxObject(
            object_id="fragment_wall",
            kind="wall",
            center_xyz_m=(1.5, 0.0, 0.4),
            size_xyz_m=(0.2, 5.0, 0.8),
            yaw_rad=0.0,
            material_id="wall_interior",
        )
    )
    manifest = replace(m, graph_edges=new_edges, walls=tuple(new_walls), landmarks=(m.landmarks[0],))

    collector = RouteTeacher(registry, n_envs=1)

    class MockScene:
        def __init__(self, manifest):
            self.manifest = manifest
            self.scene_graph = SceneGraph(manifest)

    mock_scene = MockScene(manifest)
    cells = collector.spawn_restriction_cells(mock_scene)

    # The fragment wall splits the corridor; only the right segment
    # (cells 2 and 3) can reach the surviving landmark at x=3.0. Cell 3
    # itself has the landmark sitting on its center, so the inflated
    # grid excludes it as a spawn site — only cell 2 remains canonical.
    assert cells is not None
    assert 0 not in cells
    assert 1 not in cells
    assert 2 in cells
    assert 3 not in cells
