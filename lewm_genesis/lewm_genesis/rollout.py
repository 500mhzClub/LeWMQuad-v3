"""Per-scene Genesis bulk rollout loop.

Drives one ``SceneBuild`` with a locomotion policy for ``n_blocks`` command
blocks, captures every per-tick observation and label, and streams the
result through an ``MCAPSceneWriter``.

Nesting (taken from the platform manifest):

- physics step ``physics_dt_s`` (0.002 s)
- policy step ``policy_dt_s`` (0.02 s) → 10 physics steps per policy step
- command tick ``command_dt_s`` (0.10 s) → 5 policy steps per command tick
- command block ``action_block_size`` ticks (5) → one block per 0.5 s

Data is emitted at command-tick cadence (10 Hz). RGB renders once per command
tick. The command request is emitted at block-start; the executed-block record
is emitted at block-end with the full requested/executed arrays.

Locomotion policy is pluggable via :class:`PolicyInterface`. The
:class:`StancePolicy` shipped here is a placeholder that holds the robot in
the default Go2 stance — fine for end-to-end smoke runs but not a
substitute for the Tier A trained policy.

Genesis is required at runtime. Import-time is safe (lazy import in
``scene_builder``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np

from lewm_genesis.lewm_contract import (
    BaseStateRecord,
    CommandBlockRecord,
    EpisodeInfoRecord,
    EpisodeState,
    ExecutedCommandBlockRecord,
    FootContactsRecord,
    PrimitiveRegistry,
    ResetEventRecord,
    SafetyLimits,
    apply_safety_limits_batch,
    base_state_from_genesis,
    foot_contacts_record,
    make_episode_states,
    quat_to_rpy,
    reconstruct_executed_block,
    rotate_body_to_world,
    sample_command_tape,
)
from lewm_genesis.scene_builder import SceneBuild
from lewm_genesis.scene_loader import ScenePack


# Default Go2 stance qpos for the 12 leg DOFs, ordered to match Genesis's
# joint-type grouping: [hip_FL, hip_FR, hip_RL, hip_RR,
#                       thigh_FL, thigh_FR, thigh_RL, thigh_RR,
#                       calf_FL, calf_FR, calf_RL, calf_RR].
DEFAULT_GO2_STANCE_RAD = np.array(
    [
        0.0, 0.0, 0.0, 0.0,
        0.9, 0.9, 0.9, 0.9,
        -1.8, -1.8, -1.8, -1.8,
    ],
    dtype=np.float32,
)

# Genesis dof indices for the 12 leg joints (skipping the 6-DOF free root joint).
DEFAULT_GO2_LEG_DOF_INDICES: tuple[int, ...] = tuple(range(6, 18))


@dataclass(frozen=True)
class RolloutConfig:
    """Per-rollout knobs that aren't fixed by the platform manifest."""

    n_blocks: int = 200
    fall_z_threshold_m: float = 0.15
    out_of_bounds_pad_m: float = 0.5
    rgb_capture_per_block: bool = True
    seed: int = 0
    leg_dof_indices: tuple[int, ...] = DEFAULT_GO2_LEG_DOF_INDICES
    stance_qpos_rad: tuple[float, ...] = tuple(DEFAULT_GO2_STANCE_RAD.tolist())
    backend_id: str = "genesis_tier_a"
    log_progress_every_blocks: int = 50


class PolicyInterface(Protocol):
    """Pluggable locomotion policy.

    Implementations are called once per policy step (50 Hz at platform
    default). ``observation`` is a dict keyed by name with ``(n_envs, ...)``
    numpy arrays; ``act`` returns per-env leg-DOF position targets.
    """

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        ...


@dataclass
class StancePolicy:
    """Placeholder policy that commands a constant Go2 stance qpos.

    Replace with the Tier A Genesis-trained PPO policy when its checkpoint
    is available.
    """

    stance_rad: np.ndarray = field(default_factory=lambda: DEFAULT_GO2_STANCE_RAD.copy())

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        n = observation["base_pos_world"].shape[0]
        return np.tile(self.stance_rad, (n, 1))


@dataclass
class _BlockTrajectory:
    """Per-block requested-vs-executed (n_envs, T, 3) arrays."""

    requested: np.ndarray
    executed: np.ndarray
    clipped: np.ndarray  # (n_envs,) bool


class RolloutRunner:
    """Drives one scene's full rollout."""

    def __init__(
        self,
        build: SceneBuild,
        policy: PolicyInterface,
        registry: PrimitiveRegistry,
        safety_limits: SafetyLimits,
        *,
        config: RolloutConfig = RolloutConfig(),
    ) -> None:
        self.build = build
        self.policy = policy
        self.registry = registry
        self.safety = safety_limits
        self.config = config
        self.pack: ScenePack = build.pack
        self.timing = build.pack.timing
        self.n_envs = int(build.n_envs)

        self._physics_steps_per_policy = self.timing.policy_decimation
        self._policy_steps_per_command_tick = self.timing.command_ticks_per_block
        self._command_dt_ns = int(round(self.timing.command_dt_s * 1e9))
        self._policy_dt_ns = int(round(self.timing.policy_dt_s * 1e9))
        self._block_size = int(registry.block_size)
        self._leg_dof_idx = np.array(config.leg_dof_indices, dtype=np.int64)
        self._stance = np.array(config.stance_qpos_rad, dtype=np.float32)

        scene_id_int = int(abs(hash(self.pack.scene_id)) & 0x7FFF_FFFF)
        self.episode_states: list[EpisodeState] = make_episode_states(
            self.n_envs,
            scene_id=scene_id_int,
            scene_family=self.pack.family,
            split=self.pack.split,
            manifest_sha256=self.pack.manifest_sha256,
        )
        self._last_executed = np.zeros((self.n_envs, 3), dtype=np.float32)
        self._sim_time_ns = 0
        self._sequence_id_counter = 0
        self._rng = np.random.default_rng(int(config.seed))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, writer: Any) -> dict[str, Any]:
        """Run the configured number of command blocks, streaming to ``writer``.

        Returns a stats dict for the caller's summary aggregation.
        """

        from lewm_genesis import ros_msg_adapter as adapter

        self._reset_robot_to_spawn(envs_idx=None)
        wall_start = time.time()
        tape, names = sample_command_tape(
            self.registry,
            self.n_envs,
            self.config.n_blocks,
            self._rng,
        )

        for block_idx in range(self.config.n_blocks):
            block = self._clip_block(tape[:, block_idx])
            self._emit_command_block_request(block_idx, tape[:, block_idx], names, writer, adapter)

            for tick_idx in range(self._block_size):
                target_cmd = block.executed[:, tick_idx]
                self._step_command_tick(target_cmd)
                self._emit_per_tick_records(writer, adapter, tick_idx == self._block_size - 1)
                writer.write_clock(self._sim_time_ns)

            self._emit_executed_command_block(block_idx, block, names, writer, adapter)
            self._last_executed = block.executed[:, -1, :]
            self._check_and_reset_fallen_envs(writer, adapter)

            if (
                self.config.log_progress_every_blocks
                and (block_idx + 1) % self.config.log_progress_every_blocks == 0
            ):
                elapsed = time.time() - wall_start
                print(
                    f"[rollout {self.pack.scene_id}] block {block_idx + 1}/{self.config.n_blocks} "
                    f"sim_time={self._sim_time_ns / 1e9:.2f}s wall={elapsed:.1f}s"
                )

        wall_total = time.time() - wall_start
        return {
            "scene_id": self.pack.scene_id,
            "n_envs": self.n_envs,
            "n_blocks": self.config.n_blocks,
            "command_ticks": self.config.n_blocks * self._block_size,
            "wall_seconds": wall_total,
            "final_sim_time_s": self._sim_time_ns / 1e9,
        }

    # ------------------------------------------------------------------
    # Inner loop steps
    # ------------------------------------------------------------------

    def _clip_block(self, requested_block: np.ndarray) -> _BlockTrajectory:
        executed, clipped = apply_safety_limits_batch(
            requested_block, self._last_executed, self.safety
        )
        return _BlockTrajectory(requested=requested_block, executed=executed, clipped=clipped)

    def _step_command_tick(self, target_cmd: np.ndarray) -> None:
        """Run ``policy_steps_per_command_tick`` policy steps for one 100 ms tick."""

        for _ in range(self._policy_steps_per_command_tick):
            obs = self._build_observation(target_cmd)
            joint_targets = self.policy.act(obs)
            self._apply_joint_targets(joint_targets)
            for _step in range(self._physics_steps_per_policy):
                self.build.scene.step()
            self._sim_time_ns += self._policy_dt_ns

    def _build_observation(self, target_cmd: np.ndarray) -> dict[str, np.ndarray]:
        robot = self.build.robot
        pos = self._as_np(robot.get_pos())
        quat_wxyz = self._as_np(robot.get_quat())
        lin_world = self._as_np(robot.get_vel())
        ang_world = self._as_np(robot.get_ang())
        joint_pos = self._as_np(robot.get_dofs_position(self._leg_dof_idx.tolist()))
        joint_vel = self._as_np(robot.get_dofs_velocity(self._leg_dof_idx.tolist()))
        quat_xyzw = np.stack(
            [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
            axis=-1,
        )
        return {
            "base_pos_world": pos.astype(np.float32),
            "base_quat_xyzw": quat_xyzw.astype(np.float32),
            "base_lin_vel_world": lin_world.astype(np.float32),
            "base_ang_vel_world": ang_world.astype(np.float32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "command": np.asarray(target_cmd, dtype=np.float32),
        }

    def _apply_joint_targets(self, joint_targets: np.ndarray) -> None:
        robot = self.build.robot
        targets = np.asarray(joint_targets, dtype=np.float32)
        if targets.shape[-1] != len(self._leg_dof_idx):
            raise ValueError(
                f"policy returned {targets.shape[-1]} joint targets; "
                f"expected {len(self._leg_dof_idx)}"
            )
        robot.control_dofs_position(targets, self._leg_dof_idx.tolist())

    def _emit_per_tick_records(self, writer: Any, adapter: Any, is_last_tick: bool) -> None:
        """Write BaseState, FootContacts, EpisodeInfo per env at this command tick.

        RGB rendering is gated on ``rgb_capture_per_block``; when True we
        render exactly once per command block on the final tick (10 Hz at
        platform default).
        """

        robot = self.build.robot
        pos_arr = self._as_np(robot.get_pos())
        quat_wxyz = self._as_np(robot.get_quat())
        lin_world = self._as_np(robot.get_vel())
        ang_world = self._as_np(robot.get_ang())
        quat_xyzw = np.stack(
            [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
            axis=-1,
        )
        contacts = self._extract_foot_contacts()

        for env_idx in range(self.n_envs):
            self.episode_states[env_idx].step()
            base = base_state_from_genesis(
                pos_world=pos_arr[env_idx],
                quat_xyzw=quat_xyzw[env_idx],
                linear_vel_body=self._world_to_body(lin_world[env_idx], quat_xyzw[env_idx]),
                angular_vel_body=self._world_to_body(ang_world[env_idx], quat_xyzw[env_idx]),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(env_idx, "base_state", adapter.base_state_record_to_msg(base), self._sim_time_ns)
            writer.write_env(
                env_idx,
                "odom",
                adapter.odometry_from_base_state(base),
                self._sim_time_ns,
            )

            foot_rec = foot_contacts_record(
                tuple(contacts[env_idx].tolist()),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "foot_contacts", adapter.foot_contacts_record_to_msg(foot_rec), self._sim_time_ns
            )

            info_rec = self.episode_states[env_idx].episode_info(stamp_ns=self._sim_time_ns)
            writer.write_env(
                env_idx, "episode_info", adapter.episode_info_record_to_msg(info_rec), self._sim_time_ns
            )

        if is_last_tick and self.config.rgb_capture_per_block:
            self._render_and_emit_rgb(writer, adapter)

    def _emit_command_block_request(
        self,
        block_idx: int,
        requested_block: np.ndarray,
        names: list[list[str]],
        writer: Any,
        adapter: Any,
    ) -> None:
        for env_idx in range(self.n_envs):
            sequence_id = self._next_sequence_id()
            req = requested_block[env_idx]  # (T, 3)
            record = CommandBlockRecord(
                sequence_id=sequence_id,
                block_size=int(req.shape[0]),
                command_dt_s=float(self.timing.command_dt_s),
                primitive_name=str(names[env_idx][block_idx]),
                vx_body_mps=[float(v) for v in req[:, 0]],
                vy_body_mps=[float(v) for v in req[:, 1]],
                yaw_rate_radps=[float(v) for v in req[:, 2]],
                event_name="",
                event_allowed_in_training=False,
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "command_block", adapter.command_block_record_to_msg(record), self._sim_time_ns
            )

    def _emit_executed_command_block(
        self,
        block_idx: int,
        block: _BlockTrajectory,
        names: list[list[str]],
        writer: Any,
        adapter: Any,
    ) -> None:
        for env_idx in range(self.n_envs):
            requested = [tuple(map(float, block.requested[env_idx, t])) for t in range(self._block_size)]
            executed = [tuple(map(float, block.executed[env_idx, t])) for t in range(self._block_size)]
            sequence_id = self._sequence_id_counter - self.n_envs + env_idx
            record = reconstruct_executed_block(
                requested,
                executed,
                sequence_id=sequence_id,
                primitive_name=str(names[env_idx][block_idx]),
                command_dt_s=float(self.timing.command_dt_s),
                clipped=bool(block.clipped[env_idx]),
                safety_overridden=False,
                controller_mode="cmd_vel",
                backend_id=str(self.config.backend_id),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "executed_command_block",
                adapter.executed_command_block_record_to_msg(record),
                self._sim_time_ns,
            )

    # ------------------------------------------------------------------
    # Reset handling
    # ------------------------------------------------------------------

    def _check_and_reset_fallen_envs(self, writer: Any, adapter: Any) -> None:
        pos = self._as_np(self.build.robot.get_pos())
        fell = pos[:, 2] < float(self.config.fall_z_threshold_m)
        (xmin, ymin), (xmax, ymax) = self.pack.world_bounds_xy_m
        pad = float(self.config.out_of_bounds_pad_m)
        oob = (
            (pos[:, 0] < xmin - pad)
            | (pos[:, 0] > xmax + pad)
            | (pos[:, 1] < ymin - pad)
            | (pos[:, 1] > ymax + pad)
        )
        to_reset = np.where(fell | oob)[0]
        if to_reset.size == 0:
            return
        reasons = ["fall" if fell[i] else "out_of_bounds" for i in to_reset]
        self._reset_robot_to_spawn(envs_idx=to_reset.tolist())
        for env_idx, reason in zip(to_reset.tolist(), reasons):
            event = self.episode_states[env_idx].reset(
                reason=reason,
                success=True,
                spawn_pose_xyz=self.pack.robot.spawn_xyz_m,
                spawn_pose_quat_xyzw=self._wxyz_to_xyzw(self.pack.robot.spawn_quat_wxyz),
                stamp_ns=self._sim_time_ns,
            )
            self._last_executed[env_idx] = 0.0
            writer.write_env(
                env_idx, "reset_event", adapter.reset_event_record_to_msg(event), self._sim_time_ns
            )

    def _reset_robot_to_spawn(self, envs_idx: list[int] | None) -> None:
        robot = self.build.robot
        spawn_pos = np.array(self.pack.robot.spawn_xyz_m, dtype=np.float32)
        spawn_quat = np.array(self.pack.robot.spawn_quat_wxyz, dtype=np.float32)
        if envs_idx is None:
            envs = list(range(self.n_envs))
        else:
            envs = list(envs_idx)
        if not envs:
            return
        pos_batch = np.tile(spawn_pos, (len(envs), 1))
        quat_batch = np.tile(spawn_quat, (len(envs), 1))
        stance_batch = np.tile(self._stance, (len(envs), 1))
        robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
        robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)
        robot.set_dofs_position(stance_batch, self._leg_dof_idx.tolist(), envs_idx=envs)
        robot.set_dofs_velocity(
            np.zeros_like(stance_batch), self._leg_dof_idx.tolist(), envs_idx=envs
        )

    # ------------------------------------------------------------------
    # Camera + contacts
    # ------------------------------------------------------------------

    def _render_and_emit_rgb(self, writer: Any, adapter: Any) -> None:
        result = self.build.camera.render()
        rgb = self._extract_rgb(result)
        if rgb is None:
            return
        # rgb shape (n_envs, H, W, 3) uint8 — single-env Genesis returns (H, W, 3).
        if rgb.ndim == 3:
            rgb = rgb[None, ...]
        info_msg = adapter.camera_info_to_msg(
            self.pack.camera, stamp_ns=self._sim_time_ns, frame_id="camera_link"
        )
        for env_idx in range(min(self.n_envs, rgb.shape[0])):
            frame = np.ascontiguousarray(rgb[env_idx]).astype(np.uint8, copy=False)
            img_msg = adapter.rgb_image_to_msg(
                frame, stamp_ns=self._sim_time_ns, frame_id="camera_link"
            )
            writer.write_env(env_idx, "rgb_image", img_msg, self._sim_time_ns)
            writer.write_env(env_idx, "camera_info", info_msg, self._sim_time_ns)

    @staticmethod
    def _extract_rgb(render_result: Any) -> np.ndarray | None:
        """Genesis camera.render() returns either an ndarray or a tuple. Unpack defensively."""

        if render_result is None:
            return None
        if isinstance(render_result, np.ndarray):
            return render_result
        if isinstance(render_result, tuple) and render_result:
            first = render_result[0]
            if isinstance(first, np.ndarray):
                return first
            try:
                import torch  # type: ignore[import-not-found]
            except ImportError:
                return None
            if isinstance(first, torch.Tensor):
                return first.detach().cpu().numpy()
        return None

    def _extract_foot_contacts(self) -> np.ndarray:
        """Return per-env ``(n_envs, 4)`` bool array in LeWM fl/fr/rl/rr order.

        Uses ``get_links_net_contact_force`` against the calf links as a
        contact proxy. The Genesis-shipped Go2 URDF has no ``_foot`` links;
        the foot is the terminal feature of the calf. If a future URDF
        exposes explicit foot links, switch the link-name lookup.
        """

        robot = self.build.robot
        target_names = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
        link_indices = []
        for name in target_names:
            for link in robot.links:
                if link.name == name:
                    link_indices.append(link.idx)
                    break
        if len(link_indices) != 4:
            return np.zeros((self.n_envs, 4), dtype=bool)

        try:
            forces = self._as_np(robot.get_links_net_contact_force(link_indices))
        except Exception:
            return np.zeros((self.n_envs, 4), dtype=bool)
        # forces shape: (n_envs, 4, 3) or (4, 3). Threshold magnitude.
        if forces.ndim == 2:
            forces = forces[None, ...]
        magnitudes = np.linalg.norm(forces, axis=-1)  # (n_envs, 4)
        return magnitudes > 1e-3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_np(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return np.asarray(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: Iterable[float]) -> tuple[float, float, float, float]:
        w, x, y, z = (float(c) for c in q_wxyz)
        return (x, y, z, w)

    @staticmethod
    def _world_to_body(v_world: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        # Inverse of rotate_body_to_world: use conjugate quaternion.
        q_inv = np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float64)
        return rotate_body_to_world(v_world, q_inv).astype(np.float32)

    def _next_sequence_id(self) -> int:
        sid = self._sequence_id_counter
        self._sequence_id_counter += 1
        return sid
