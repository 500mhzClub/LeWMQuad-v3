"""Recovery / contact curriculum (data spec §13 — 10% share).

Drives the robot deliberately toward walls, then once it's within
``approach_clearance_m`` of an obstacle / wall, sequences a backout + yaw +
resume phase to generate the "wall contact, stuck, backout" trajectories
called out in spec §10. Per-env state machine: ``approach`` → ``backout``
→ ``pivot`` → ``approach`` (next wall).

Approach behaviour: pick a random reachable cell at least ``approach_hops``
away as a heading target, then drive forward toward it. The recovery curriculum
deliberately ignores clearance during ``approach`` so the robot actually makes
contact; once clearance drops below ``approach_clearance_m``, the FSM advances.
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


_PHASE_APPROACH = "approach"
_PHASE_BACKOUT = "backout"
_PHASE_PIVOT = "pivot"


class RecoveryCurriculum:
    name = "recovery"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        approach_clearance_m: float = 0.18,
        backout_blocks: int = 2,
        pivot_blocks: int = 2,
        approach_hops: int = 2,
    ) -> None:
        self._registry = registry
        self._clearance = float(approach_clearance_m)
        self._backout_blocks = int(backout_blocks)
        self._pivot_blocks = int(pivot_blocks)
        self._approach_hops = int(approach_hops)

        self._phase: list[str] = [_PHASE_APPROACH] * int(n_envs)
        self._phase_blocks: list[int] = [0] * int(n_envs)
        self._approach_target: list[int] = [-1] * int(n_envs)
        self._pivot_sign: list[int] = [1] * int(n_envs)

    def on_episode_reset(self, env_idx: int) -> None:
        self._phase[env_idx] = _PHASE_APPROACH
        self._phase_blocks[env_idx] = 0
        self._approach_target[env_idx] = -1

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,
        rng: np.random.Generator,
    ) -> BlockChoice:
        env_idx = observation.env_idx
        phase = self._phase[env_idx]

        # Phase transitions.
        if (
            phase == _PHASE_APPROACH
            and observation.clearance_to_walls_m <= self._clearance
        ):
            phase = _PHASE_BACKOUT
            self._phase_blocks[env_idx] = 0
            # Pick a yaw direction for the upcoming pivot.
            self._pivot_sign[env_idx] = int(rng.choice([-1, 1]))
        elif phase == _PHASE_BACKOUT and self._phase_blocks[env_idx] >= self._backout_blocks:
            phase = _PHASE_PIVOT
            self._phase_blocks[env_idx] = 0
        elif phase == _PHASE_PIVOT and self._phase_blocks[env_idx] >= self._pivot_blocks:
            phase = _PHASE_APPROACH
            self._phase_blocks[env_idx] = 0
            self._approach_target[env_idx] = -1  # repick

        self._phase[env_idx] = phase
        self._phase_blocks[env_idx] += 1

        if phase == _PHASE_BACKOUT:
            return BlockChoice(
                requested_block=expand_primitive_to_block(self._registry, "backward"),
                primitive_name="backward",
                command_source=self.name,
            )
        if phase == _PHASE_PIVOT:
            primitive_name = "yaw_left" if self._pivot_sign[env_idx] > 0 else "yaw_right"
            return BlockChoice(
                requested_block=expand_primitive_to_block(self._registry, primitive_name),
                primitive_name=primitive_name,
                command_source=self.name,
            )

        # Approach: head toward a wall by walking toward a distant cell.
        target = self._approach_target[env_idx]
        if (
            target < 0
            or target >= scene.n_nodes
            or scene.bfs_distance(observation.current_cell_id, target) is None
        ):
            target = self._pick_approach_target(observation.current_cell_id, scene, rng)
            self._approach_target[env_idx] = target

        if target < 0:
            # Empty graph fallback — just walk forward.
            return BlockChoice(
                requested_block=expand_primitive_to_block(self._registry, "forward_medium"),
                primitive_name="forward_medium",
                command_source=self.name,
            )

        waypoint = scene.next_waypoint(observation.current_cell_id, target) or target
        wp_xy = scene.cell_center(int(waypoint))
        bearing = math.atan2(
            wp_xy[1] - observation.base_xy_world[1],
            wp_xy[0] - observation.base_xy_world[0],
        )
        heading_err = wrap_angle_pi(bearing - observation.base_yaw_world)
        primitive_name = primitive_toward_bearing(heading_error_rad=heading_err)
        return BlockChoice(
            requested_block=expand_primitive_to_block(self._registry, primitive_name),
            primitive_name=primitive_name,
            command_source=self.name,
            route_target_id=int(target),
            next_waypoint_id=int(waypoint),
        )

    # ------------------------------------------------------------------

    def _pick_approach_target(
        self, current_cell: int, scene, rng: np.random.Generator
    ) -> int:
        reachable = scene.reachable_cells(current_cell)
        candidates = [
            int(c)
            for c in reachable
            if c != current_cell
            and (scene.bfs_distance(current_cell, c) or 0) >= self._approach_hops
        ]
        if not candidates:
            candidates = [int(c) for c in reachable if c != current_cell]
        if not candidates:
            return -1
        return int(rng.choice(candidates))
