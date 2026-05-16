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

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
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

# Go2 entity-local DOF indices for the 12 leg joints (skipping the 6-DOF free
# root joint). Built scenes may shift these globally when other entities are
# inserted first; RolloutRunner resolves concrete indices from joint names.
DEFAULT_GO2_LEG_DOF_INDICES: tuple[int, ...] = tuple(range(6, 18))

DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER: tuple[str, ...] = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

# Upstream Genesis Go2 policy joint order maps to Genesis DOF starts in this
# order for the bundled Go2 URDF. Rollout application remains in sorted DOF
# order via DEFAULT_GO2_LEG_DOF_INDICES.
GENESIS_GO2_POLICY_DOF_INDICES: tuple[int, ...] = (
    7, 11, 15,  # FR hip/thigh/calf
    6, 10, 14,  # FL hip/thigh/calf
    9, 13, 17,  # RR hip/thigh/calf
    8, 12, 16,  # RL hip/thigh/calf
)


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
    foot_contact_source: str = "genesis"


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
class GenesisGo2PPOPolicy:
    """RSL-RL Go2 PPO policy adapter for the Genesis bulk rollout loop.

    The trained actor consumes the upstream Genesis Go2 observation layout:
    body angular velocity, projected gravity, scaled command, joint position
    error, joint velocity, and previous action. It emits 12 position-offset
    actions in the upstream policy joint-name order. This adapter returns
    absolute joint targets in the rollout DOF order expected by
    :meth:`RolloutRunner._apply_joint_targets`.
    """

    checkpoint_path: str | Path
    cfg_path: str | Path | None = None
    rollout_leg_dof_indices: tuple[int, ...] = DEFAULT_GO2_LEG_DOF_INDICES
    policy_dof_indices: tuple[int, ...] = GENESIS_GO2_POLICY_DOF_INDICES
    device: str | None = None
    simulate_action_latency: bool = True

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path)
        self.cfg_path = Path(self.cfg_path) if self.cfg_path is not None else self.checkpoint_path.parent / "cfgs.pkl"
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"missing PPO checkpoint: {self.checkpoint_path}")
        if not self.cfg_path.is_file():
            raise FileNotFoundError(f"missing PPO cfgs.pkl: {self.cfg_path}")

        with self.cfg_path.open("rb") as f:
            env_cfg, obs_cfg, _reward_cfg, _command_cfg, train_cfg = pickle.load(f)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.train_cfg = train_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.command_scale = np.array(
            [
                float(self.obs_scales["lin_vel"]),
                float(self.obs_scales["lin_vel"]),
                float(self.obs_scales["ang_vel"]),
            ],
            dtype=np.float32,
        )
        self.action_scale = float(env_cfg["action_scale"])
        self.default_dof_pos_policy = np.array(
            [float(env_cfg["default_joint_angles"][name]) for name in env_cfg["joint_names"]],
            dtype=np.float32,
        )

        # For the Genesis-bundled Go2 URDF, these policy joint names map to
        # DOF starts 6..17. Keep the mapping explicit so a future URDF change
        # fails loudly instead of silently permuting actions.
        self.policy_joint_names = tuple(str(name) for name in env_cfg["joint_names"])
        if len(self.policy_dof_indices) != len(self.policy_joint_names):
            raise ValueError(
                f"policy_dof_indices has {len(self.policy_dof_indices)} entries; "
                f"expected {len(self.policy_joint_names)}"
            )
        self._rollout_from_policy = np.array(
            [self.policy_dof_indices.index(int(dof_idx)) for dof_idx in self.rollout_leg_dof_indices],
            dtype=np.int64,
        )
        self._policy_from_rollout = np.array(
            [self.rollout_leg_dof_indices.index(int(dof_idx)) for dof_idx in self.policy_dof_indices],
            dtype=np.int64,
        )
        self.reset_stance_rad = self.default_dof_pos_policy[self._rollout_from_policy].astype(
            np.float32,
            copy=True,
        )

        self._device = self.device or self._default_torch_device()
        self._policy = self._load_policy()
        self._last_actions: np.ndarray | None = None

    @classmethod
    def from_platform_manifest(
        cls,
        platform_manifest: dict[str, Any],
        workspace_root: str | Path,
        **kwargs: Any,
    ) -> "GenesisGo2PPOPolicy":
        """Create the adapter from ``locomotion.policy_artifact`` metadata."""

        artifact = platform_manifest.get("locomotion", {}).get("policy_artifact", {})
        checkpoint = artifact.get("path")
        if not checkpoint:
            raise ValueError("platform manifest missing locomotion.policy_artifact.path")
        root = Path(workspace_root)
        cfg_path = artifact.get("cfg_path")
        return cls(
            checkpoint_path=_resolve_manifest_path(root, checkpoint),
            cfg_path=_resolve_manifest_path(root, cfg_path) if cfg_path else None,
            **kwargs,
        )

    @staticmethod
    def _default_torch_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _load_policy(self):
        try:
            import torch
            from rsl_rl.runners import OnPolicyRunner
            from tensordict import TensorDict
        except ImportError as exc:  # pragma: no cover - depends on training venv
            raise RuntimeError(
                "GenesisGo2PPOPolicy requires torch, tensordict, and rsl-rl-lib. "
                "Use the Genesis ROCm training venv."
            ) from exc

        class _DummyEnv:
            def __init__(self, env_cfg: dict[str, Any], device: str) -> None:
                self.num_envs = 1
                self.num_actions = int(env_cfg["num_actions"])
                self.max_episode_length = 1000
                self.episode_length_buf = torch.zeros((1,), dtype=torch.int32, device=device)
                self.device = device
                self.cfg = env_cfg

            def get_observations(self):
                return TensorDict(
                    {"policy": torch.zeros((1, 45), dtype=torch.float32, device=self.device)},
                    batch_size=[1],
                )

            def step(self, actions):
                raise NotImplementedError

        runner = OnPolicyRunner(_DummyEnv(self.env_cfg, self._device), self.train_cfg, log_dir=None, device=self._device)
        runner.load(str(self.checkpoint_path), map_location=self._device)
        policy = runner.get_inference_policy(device=self._device)
        policy.eval()
        return policy

    def reset(self, envs_idx: list[int] | None = None) -> None:
        """Clear action-history latency state after rollout resets."""

        if self._last_actions is None:
            return
        if envs_idx is None:
            self._last_actions.fill(0.0)
            return
        if envs_idx:
            self._last_actions[np.asarray(envs_idx, dtype=np.int64)] = 0.0

    def validate_rollout_robot(self, robot: Any) -> None:
        """Verify the built Genesis robot matches the policy joint contract."""

        joints = getattr(robot, "joints", None)
        if joints is None:
            return
        joint_by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
        missing = [name for name in self.policy_joint_names if name not in joint_by_name]
        if missing:
            raise ValueError(
                "Genesis Go2 PPO policy requires joints missing from the rollout robot: "
                + ", ".join(missing)
            )
        actual_policy_dofs = tuple(
            _entity_local_dof_index(robot, joint_by_name[name]) for name in self.policy_joint_names
        )
        expected_policy_dofs = tuple(int(idx) for idx in self.policy_dof_indices)
        if actual_policy_dofs != expected_policy_dofs:
            pairs = ", ".join(
                f"{name}: expected {expected}, actual {actual}"
                for name, expected, actual in zip(
                    self.policy_joint_names,
                    expected_policy_dofs,
                    actual_policy_dofs,
                )
                if expected != actual
            )
            raise ValueError(
                "Genesis Go2 PPO policy joint mapping does not match rollout robot "
                f"({pairs})"
            )

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        try:
            import torch
            from tensordict import TensorDict
        except ImportError as exc:  # pragma: no cover - depends on training venv
            raise RuntimeError(
                "GenesisGo2PPOPolicy requires torch and tensordict at inference time."
            ) from exc

        n_envs = int(observation["command"].shape[0])
        if self._last_actions is None or self._last_actions.shape[0] != n_envs:
            self._last_actions = np.zeros((n_envs, len(self.policy_joint_names)), dtype=np.float32)

        obs_np = self._build_policy_observation(observation)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        obs_dict = TensorDict({"policy": obs_tensor}, batch_size=[n_envs])
        with torch.no_grad():
            actions = self._policy(obs_dict).detach().cpu().numpy().astype(np.float32, copy=False)
        if actions.shape != self._last_actions.shape:
            raise ValueError(f"PPO policy returned {actions.shape}; expected {self._last_actions.shape}")

        exec_actions = self._last_actions if self.simulate_action_latency else actions
        target_policy_order = exec_actions * self.action_scale + self.default_dof_pos_policy
        self._last_actions = actions.copy()
        return target_policy_order[:, self._rollout_from_policy].astype(np.float32, copy=False)

    def _build_policy_observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        quat_xyzw = np.asarray(observation["base_quat_xyzw"], dtype=np.float32)
        ang_body = _rotate_world_to_body(
            np.asarray(observation["base_ang_vel_world"], dtype=np.float32),
            quat_xyzw,
        )
        gravity_world = np.zeros_like(ang_body, dtype=np.float32)
        gravity_world[:, 2] = -1.0
        projected_gravity = _rotate_world_to_body(gravity_world, quat_xyzw)
        joint_pos = np.asarray(observation["joint_pos"], dtype=np.float32)[:, self._policy_from_rollout]
        joint_vel = np.asarray(observation["joint_vel"], dtype=np.float32)[:, self._policy_from_rollout]
        command = np.asarray(observation["command"], dtype=np.float32) * self.command_scale
        return np.concatenate(
            (
                ang_body * float(self.obs_scales["ang_vel"]),
                projected_gravity,
                command,
                (joint_pos - self.default_dof_pos_policy) * float(self.obs_scales["dof_pos"]),
                joint_vel * float(self.obs_scales["dof_vel"]),
                self._last_actions,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)


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
        self._leg_dof_idx = _resolve_rollout_leg_dof_indices(build.robot, config.leg_dof_indices)
        self._stance = np.array(config.stance_qpos_rad, dtype=np.float32)
        if config.foot_contact_source not in {"genesis", "zero"}:
            raise ValueError(
                "foot_contact_source must be 'genesis' or 'zero'; "
                f"got {config.foot_contact_source!r}"
            )
        validate_robot = getattr(self.policy, "validate_rollout_robot", None)
        if callable(validate_robot):
            validate_robot(build.robot)

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
        self._emit_initial_reset_events(writer, adapter)
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
        joint_pos = self._as_np(robot.get_dofs_position(self._leg_dof_idx.tolist()))
        joint_vel = self._as_np(robot.get_dofs_velocity(self._leg_dof_idx.tolist()))
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
            writer.write_env(
                env_idx,
                "joint_states",
                adapter.joint_state_to_msg(
                    joint_names=DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER,
                    positions=joint_pos[env_idx],
                    velocities=joint_vel[env_idx],
                    efforts=None,
                    stamp_ns=self._sim_time_ns,
                ),
                self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "imu",
                adapter.imu_to_msg(
                    quat_xyzw=tuple(float(q) for q in quat_xyzw[env_idx]),
                    angular_vel_body=base.twist_body_angular,
                    linear_accel_body=(0.0, 0.0, 0.0),
                    stamp_ns=self._sim_time_ns,
                    frame_id="imu_link",
                ),
                self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "mode",
                adapter.mode_state_to_msg(
                    stamp_ns=self._sim_time_ns,
                    backend_id=str(self.config.backend_id),
                    controller_mode="cmd_vel",
                    gait_name="genesis_contract_ppo",
                    moving=True,
                    status_text="genesis bulk PPO rollout",
                ),
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

    def _emit_initial_reset_events(self, writer: Any, adapter: Any) -> None:
        for env_idx in range(self.n_envs):
            event = self.episode_states[env_idx].reset(
                reason="initial_spawn",
                success=True,
                spawn_pose_xyz=self.pack.robot.spawn_xyz_m,
                spawn_pose_quat_xyzw=self._wxyz_to_xyzw(self.pack.robot.spawn_quat_wxyz),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "reset_event", adapter.reset_event_record_to_msg(event), self._sim_time_ns
            )

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
        reset_stance = getattr(self.policy, "reset_stance_rad", self._stance)
        stance_batch = np.tile(np.asarray(reset_stance, dtype=np.float32), (len(envs), 1))
        robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
        robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)
        robot.set_dofs_position(stance_batch, self._leg_dof_idx.tolist(), envs_idx=envs)
        robot.set_dofs_velocity(
            np.zeros_like(stance_batch), self._leg_dof_idx.tolist(), envs_idx=envs
        )
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(envs_idx)

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

        if self.config.foot_contact_source == "zero":
            return np.zeros((self.n_envs, 4), dtype=bool)

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
        return _rotate_world_to_body(v_world[None, :], q_xyzw[None, :])[0]

    def _next_sequence_id(self) -> int:
        sid = self._sequence_id_counter
        self._sequence_id_counter += 1
        return sid


def _rotate_world_to_body(v_world: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the body frame using xyzw quaternions."""

    q_inv = np.asarray(q_xyzw, dtype=np.float64).copy()
    q_inv[..., :3] *= -1.0
    return rotate_body_to_world(np.asarray(v_world, dtype=np.float64), q_inv).astype(np.float32)


def _resolve_manifest_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_rollout_leg_dof_indices(robot: Any, configured_indices: Iterable[int]) -> np.ndarray:
    configured = tuple(int(idx) for idx in configured_indices)
    if configured != DEFAULT_GO2_LEG_DOF_INDICES:
        return np.array(configured, dtype=np.int64)
    joints = getattr(robot, "joints", None)
    if joints is None:
        return np.array(configured, dtype=np.int64)
    joint_by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
    if not all(name in joint_by_name for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER):
        return np.array(configured, dtype=np.int64)
    return np.array(
        [
            _single_dof_index(joint_by_name[name])
            for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER
        ],
        dtype=np.int64,
    )


def _entity_local_dof_index(robot: Any, joint: Any) -> int:
    return _single_dof_index(joint) - int(getattr(robot, "dof_start", 0))


def _single_dof_index(joint: Any) -> int:
    dofs_idx = getattr(joint, "dofs_idx", None)
    if dofs_idx is None:
        dofs_idx = getattr(joint, "dof_idx", None)
    if isinstance(dofs_idx, (list, tuple)):
        if len(dofs_idx) != 1:
            raise ValueError(
                f"joint {getattr(joint, 'name', '<unnamed>')} has {len(dofs_idx)} DOFs; expected 1"
            )
        return int(dofs_idx[0])
    if dofs_idx is None:
        raise ValueError(f"joint {getattr(joint, 'name', '<unnamed>')} has no DOF index")
    return int(dofs_idx)
