"""Privileged-BFS route teacher (data spec §13 — 30% share).

Picks a goal cell per episode (landmark cell, or a random distant cell),
then each block:

1. Locates the env's current cell from the scene graph.
2. Computes the next waypoint via BFS on the traversable subgraph.
3. Steers a velocity primitive toward the waypoint center (arc/yaw/forward
   chosen by heading-error tolerance).
4. Re-targets when the goal is reached. If ``revisit_after_arrival`` is set,
   re-targets to a previously visited goal cell to drive loop-closure
   examples; otherwise picks a fresh goal.

Reaching the goal counts as arriving within ``arrival_cell_radius`` cells
on the graph (default: 0, exact match). Once arrived, the teacher emits one
``hold`` block to mark the success then re-targets.
"""

from __future__ import annotations

import math

import numpy as np

from lewm_genesis.collectors.base import (
    BlockChoice,
    EnvObservation,
    primitive_toward_bearing,
    wrap_angle_pi,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry, expand_primitive_to_block


class RouteTeacher:
    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        name: str = "route_teacher",
        arrival_cell_radius: int = 0,
        replan_every_blocks: int | None = 6,
        revisit_after_arrival: bool = False,
        min_goal_distance_hops: int = 3,
    ) -> None:
        self.name = name
        self._registry = registry
        self._arrival = int(arrival_cell_radius)
        self._replan = replan_every_blocks
        self._revisit = bool(revisit_after_arrival)
        self._min_goal_hops = int(min_goal_distance_hops)
        # Per-env state. All arrays sized to n_envs and reset on reset.
        self._goal: list[int] = [-1] * int(n_envs)
        self._goal_history: list[list[int]] = [[] for _ in range(int(n_envs))]
        self._blocks_since_plan: list[int] = [0] * int(n_envs)

    def on_episode_reset(self, env_idx: int) -> None:
        self._goal[env_idx] = -1
        self._blocks_since_plan[env_idx] = 0
        # Keep goal history across resets so revisit mode has something to
        # pick from on the very first episode of a new env after the reset.

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,
        rng: np.random.Generator,
    ) -> BlockChoice:
        env_idx = observation.env_idx
        # Pick or refresh a goal.
        goal = self._goal[env_idx]
        need_new_goal = (
            goal < 0
            or goal >= scene.n_nodes
            or scene.bfs_distance(observation.current_cell_id, goal) is None
            or self._is_arrived(observation.current_cell_id, goal, scene)
        )
        if need_new_goal:
            goal = self._pick_goal(observation.current_cell_id, scene, rng)
            self._goal[env_idx] = goal
            self._blocks_since_plan[env_idx] = 0
            if goal >= 0:
                self._goal_history[env_idx].append(goal)
        else:
            self._blocks_since_plan[env_idx] += 1
            if self._replan is not None and self._blocks_since_plan[env_idx] >= self._replan:
                # Periodic replan keeps the teacher honest if the robot has
                # been carried off-route by clipping / collisions.
                self._blocks_since_plan[env_idx] = 0

        if goal < 0:
            # No reachable goal — fall back to a hold block so the scheduler
            # can still tag the source.
            return self._hold_block(route_target_id=-1, next_waypoint_id=-1)

        waypoint = scene.next_waypoint(observation.current_cell_id, goal)
        if waypoint is None:
            # Already at the goal (or no path) — emit hold and let the next
            # block re-pick a goal.
            self._goal[env_idx] = -1
            return self._hold_block(route_target_id=goal, next_waypoint_id=-1)

        wp_xy = scene.cell_center(waypoint)
        bearing = math.atan2(
            wp_xy[1] - observation.base_xy_world[1],
            wp_xy[0] - observation.base_xy_world[0],
        )
        heading_err = wrap_angle_pi(bearing - observation.base_yaw_world)
        primitive_name = primitive_toward_bearing(heading_error_rad=heading_err)
        block = expand_primitive_to_block(self._registry, primitive_name)
        return BlockChoice(
            requested_block=block,
            primitive_name=primitive_name,
            command_source=self.name,
            route_target_id=int(goal),
            next_waypoint_id=int(waypoint),
        )

    # ------------------------------------------------------------------

    def _is_arrived(self, current_cell: int, goal_cell: int, scene) -> bool:
        if current_cell == goal_cell:
            return True
        if self._arrival <= 0:
            return False
        dist = scene.bfs_distance(current_cell, goal_cell)
        return dist is not None and dist <= self._arrival

    def _pick_goal(self, current_cell: int, scene, rng: np.random.Generator) -> int:
        env_idx_history = None  # not used; goal history is per env elsewhere
        # Prefer revisiting a prior goal in loop-revisit mode.
        if self._revisit:
            # Search across *all* per-env histories so a freshly-spawned env
            # can still revisit a cell another env has been to. This makes
            # the loop-revisit collector behave well even at n_envs=1 by
            # falling back to the route-teacher behaviour when no history
            # exists yet.
            pool = [
                cell
                for history in self._goal_history
                for cell in history
                if cell != current_cell
            ]
            if pool:
                return int(rng.choice(pool))

        # Default: pick a landmark cell that's far enough away; otherwise pick
        # a random reachable cell.
        reachable = scene.reachable_cells(current_cell)
        landmark_pool = [
            cell
            for _name, cell in scene.landmark_cells
            if cell in reachable and cell != current_cell
        ]
        if landmark_pool:
            # Bias toward landmarks more than ``min_goal_hops`` away.
            distant = [
                cell
                for cell in landmark_pool
                if (
                    scene.bfs_distance(current_cell, cell) or 0
                )
                >= self._min_goal_hops
            ]
            choice = rng.choice(distant) if distant else rng.choice(landmark_pool)
            return int(choice)

        # No landmarks — pick a reachable cell at least min_goal_hops away.
        candidates: list[int] = []
        for cell in reachable:
            if cell == current_cell:
                continue
            dist = scene.bfs_distance(current_cell, cell)
            if dist is not None and dist >= self._min_goal_hops:
                candidates.append(int(cell))
        if not candidates:
            candidates = [int(c) for c in reachable if c != current_cell]
        if not candidates:
            return -1
        return int(rng.choice(candidates))

    def _hold_block(self, *, route_target_id: int, next_waypoint_id: int) -> BlockChoice:
        block = expand_primitive_to_block(self._registry, "hold")
        return BlockChoice(
            requested_block=block,
            primitive_name="hold",
            command_source=self.name,
            route_target_id=int(route_target_id),
            next_waypoint_id=int(next_waypoint_id),
        )
