"""Tests for collector policies and the episode scheduler.

These tests don't require Genesis at runtime — they exercise the pure-Python
parts of the §13 collection mix against a tiny scene built by hand and the
real primitive registry from ``config/go2_primitive_registry.yaml``.
"""

from __future__ import annotations

import math
from collections import Counter
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
    RouteTeacher,
    build_default_policies,
)
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
    nodes = tuple(
        GraphNode(
            node_id=i,
            center_xy_m=(i * cell, 0.0),
            width_m=cell - wall_t,
            tags=("spawn",) if i == 0 else (),
        )
        for i in range(4)
    )
    edges = tuple(
        GraphEdge(source=i, target=i + 1, width_m=cell - wall_t, traversable=True)
        for i in range(3)
    )
    walls = (
        BoxObject(
            object_id="north_wall",
            kind="wall",
            center_xyz_m=(1.5 * cell, cell * 0.5, wall_h * 0.5),
            size_xyz_m=(4 * cell, wall_t, wall_h),
            yaw_rad=0.0,
            material_id="wall_interior",
        ),
        BoxObject(
            object_id="south_wall",
            kind="wall",
            center_xyz_m=(1.5 * cell, -cell * 0.5, wall_h * 0.5),
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
    )
    return SceneManifest(
        scene_id="grid_corridor",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-1.0, -1.0), (4.0, 1.0)),
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
    assert primitive_toward_bearing(heading_error_rad=0.0) == "forward_medium"
    assert primitive_toward_bearing(heading_error_rad=math.pi / 6) == "arc_left"
    assert primitive_toward_bearing(heading_error_rad=-math.pi / 6) == "arc_right"
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

    # At cell 0 facing +x, the goal landmark sits at cell 3. The next waypoint
    # should be cell 1 and heading error ~0 → forward primitive.
    choice = collector.on_block(
        observation=_observation(base_xy=(0.0, 0.0), yaw=0.0, cell_id=0),
        scene=scene,
        rng=rng,
    )
    assert choice.command_source == "route_teacher"
    assert choice.route_target_id == 3  # landmark cell
    assert choice.next_waypoint_id == 1
    assert choice.primitive_name in {"forward_medium", "arc_left", "arc_right"}


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


def test_route_teacher_repicks_goal_after_arrival(registry):
    scene = SceneGraph(_grid_corridor_manifest())
    collector = RouteTeacher(registry, n_envs=1)
    collector.on_episode_reset(0)
    rng = np.random.default_rng(0)
    # First block: at cell 0, goal=3.
    first = collector.on_block(
        observation=_observation(cell_id=0, base_xy=(0.0, 0.0)), scene=scene, rng=rng
    )
    assert first.route_target_id == 3
    # Pretend we arrived at cell 3 — the teacher should re-target on the
    # same block so we don't waste compute holding at the goal. The new
    # goal must differ from the one we just arrived at.
    arrival = collector.on_block(
        observation=_observation(cell_id=3, base_xy=(3.0, 0.0)), scene=scene, rng=rng
    )
    assert arrival.route_target_id != 3
    assert arrival.route_target_id != -1


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
        backout_blocks=2,
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

    # After backout: pivot phase.
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
