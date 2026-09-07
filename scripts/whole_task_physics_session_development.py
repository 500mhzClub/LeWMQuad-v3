"""Geometry-free native trace, preserving gait, contact and sensor wrappers.

The two methods differ from the inherited physical logger only by removal of
legacy region annotation and aggregation over actual recorded trace fields.
"""
from __future__ import annotations
from typing import Any, Sequence

from scripts.fast_gyro_scan_session_development import FastGyroSession
from scripts.run_go2_contact_attributed_execution_development_v1 import BASE
from scripts.run_physical_graph_edge_handoff_qualification_v1 import ExperimentError, TRACE_DT_S


class GeometryFreePhysicalSample(BASE._GenesisPhysicalSession):
    def _sample(self, requested: Sequence[float], applied: Sequence[float], timestamp_s: float) -> dict[str, Any]:
        import numpy as np

        runner = self.ctx.runner
        robot = self.ctx.build.robot
        pos = np.asarray(runner._as_np(robot.get_pos()), dtype=np.float64).reshape(-1, 3)[0]
        quat = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64).reshape(-1, 4)[0]
        vel = np.asarray(runner._as_np(robot.get_vel()), dtype=np.float64).reshape(-1, 3)[0]
        ang = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64).reshape(-1, 3)[0]
        joints = np.asarray(
            runner._as_np(robot.get_dofs_position(runner._leg_dof_idx.tolist())),
            dtype=np.float64,
        ).reshape(-1, 12)[0]
        joint_vel = np.asarray(
            runner._as_np(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())),
            dtype=np.float64,
        ).reshape(-1, 12)[0]
        return {
            "timestamp_s": float(timestamp_s),
            "base_pose_world": np.asarray([*pos, quat[1], quat[2], quat[3], quat[0]], dtype=np.float64),
            "base_twist_world": np.asarray([*vel, *ang], dtype=np.float64),
            "joint_position": joints,
            "joint_velocity": joint_vel,
            "requested_command": np.asarray(requested, dtype=np.float64),
            "post_slew_applied_command": np.asarray(applied, dtype=np.float64),
            "applied_command": np.asarray(applied, dtype=np.float64),
            "physics_contact": np.uint8(self._disallowed_contact()),
        }

    def execute_requested_ticks(
        self, requested_ticks: Sequence[Sequence[float]], *, record: bool = True
    ) -> dict[str, Any] | None:
        import numpy as np

        if len(requested_ticks) % 5 != 0:
            raise ExperimentError("physical command tape is not block aligned")
        runner = self.ctx.runner
        rows: list[dict[str, Any]] = []
        for block_start in range(0, len(requested_ticks), 5):
            requested = np.asarray(requested_ticks[block_start:block_start + 5], dtype=np.float32)[None, ...]
            block = runner._clip_block(requested)
            for tick_offset in range(5):
                command = np.asarray(block.executed[0, tick_offset], dtype=np.float32)
                for _policy_step in range(runner._policy_steps_per_command_tick):
                    observation = runner._build_observation(command[None, :])
                    cached_action = self.ctx.policy._last_actions
                    if cached_action is None:
                        # Exact first-act initialization performed by
                        # GenesisGo2PPOPolicy.act before it builds the 45-D
                        # policy observation.
                        self.ctx.policy._last_actions = np.zeros(
                            (1, 12), dtype=np.float32
                        )
                    previous_policy_action = (
                        np.zeros((12,), dtype=np.float64)
                        if cached_action is None
                        else np.asarray(cached_action, dtype=np.float64)
                        .reshape(1, 12)[0].copy()
                    )
                    controller_observation = np.asarray(
                        self.ctx.policy._build_policy_observation(observation),
                        dtype=np.float64,
                    ).reshape(1, 45)[0].copy()
                    targets = self.ctx.policy.act(observation)
                    self._last_controller_observation = controller_observation
                    self._previous_policy_action = previous_policy_action
                    self._low_level_policy_state = (
                        previous_policy_action.copy()
                        if self.ctx.policy.simulate_action_latency
                        else np.asarray(self.ctx.policy._last_actions, dtype=np.float64)
                        .reshape(1, 12)[0].copy()
                    )
                    runner._apply_joint_targets(targets)
                    base_time = float(runner._sim_time_ns) / 1.0e9
                    for physics_step in range(runner._physics_steps_per_policy):
                        self.ctx.build.scene.step()
                        if record:
                            rows.append(self._sample(
                                requested[0, tick_offset], command,
                                base_time + (physics_step + 1) * TRACE_DT_S,
                            ))
                    runner._sim_time_ns += runner._policy_dt_ns
                for state in runner.episode_states:
                    state.step()
                self._command_history[:-1] = self._command_history[1:]
                self._command_history[-1] = np.asarray(command, dtype=np.float64)
                self._control_history[:-1] = self._control_history[1:]
                self._control_history[-1] = np.asarray(
                    [command[0], command[2]], dtype=np.float64
                )
            runner._last_executed = np.asarray(block.executed[:, -1, :], dtype=np.float32).copy()
            runner._blocks_in_episode += 1
            self.ctx.ticks_executed += 5
            self.ctx.episode_ticks += 5
            self.ctx.policy_steps += 25
            self.ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if not record:
            return None
        members = tuple(rows[0])
        return {
            member: np.ascontiguousarray(np.stack([row[member] for row in rows], axis=0))
            if member not in {"timestamp_s", "physics_contact", "source_region_member", "correct_edge_region_member", "wrong_edge_region_member", "target_region_member"}
            else np.ascontiguousarray(np.asarray([row[member] for row in rows]))
            for member in members
        }


class WholeTaskPhysicsSession(FastGyroSession, GeometryFreePhysicalSample):
    """C3 ordering retains contact-stop, 50-Hz and 500-Hz acquisition wrappers."""
