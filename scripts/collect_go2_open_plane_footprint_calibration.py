#!/usr/bin/env python3
"""Collect deterministic open-plane PPO gait states for footprint calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm_genesis.lewm_contract import PrimitiveRegistry  # noqa: E402
from lewm_genesis.rollout import (  # noqa: E402
    DEFAULT_GO2_LEG_DOF_INDICES,
    DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER,
    GenesisGo2PPOPolicy,
    _pitch_from_quat_wxyz,
    _resolve_rollout_leg_dof_indices,
    _roll_from_quat_wxyz,
)
from lewm_genesis.scene_builder import initialize_genesis  # noqa: E402
from lewm_genesis.scene_loader import load_platform_manifest  # noqa: E402


DEFAULT_PRIMITIVES = (
    "hold",
    "forward_slow",
    "forward_fast",
    "backward",
    "yaw_right",
)
DEFAULT_OUTPUT = ROOT / ".generated/go2_footprint_calibration/open_plane_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        return np.asarray(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_bundled_urdf() -> Path:
    import genesis

    path = (
        Path(genesis.__file__).resolve().parent
        / "assets/urdf/go2/urdf/go2.urdf"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Genesis-bundled Go2 URDF not found: {path}")
    return path


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _record(
    topic: str,
    env_index: int,
    timestamp_ns: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "canonical_topic": topic,
        "env_index": int(env_index),
        "timestamp_ns": int(timestamp_ns),
        "payload": payload,
    }


def _command_vector(
    registry: PrimitiveRegistry,
    primitive_name: str,
) -> np.ndarray:
    spec = registry.get(primitive_name)
    if spec.get("type") != "velocity_block":
        raise ValueError(f"{primitive_name!r} is not a velocity-block primitive")
    command = spec.get("command")
    if not isinstance(command, dict):
        raise ValueError(f"{primitive_name!r} has no command mapping")
    return np.asarray(
        [float(command.get(name, 0.0)) for name in registry.command_order],
        dtype=np.float32,
    )


def _build_observation(
    robot: Any,
    leg_dof_indices: np.ndarray,
    command: np.ndarray,
) -> dict[str, np.ndarray]:
    quat_wxyz = _as_numpy(robot.get_quat())
    quat_xyzw = np.stack(
        (
            quat_wxyz[..., 1],
            quat_wxyz[..., 2],
            quat_wxyz[..., 3],
            quat_wxyz[..., 0],
        ),
        axis=-1,
    )
    return {
        "base_pos_world": _as_numpy(robot.get_pos()).astype(np.float32),
        "base_quat_xyzw": quat_xyzw.astype(np.float32),
        "base_lin_vel_world": _as_numpy(robot.get_vel()).astype(np.float32),
        "base_ang_vel_world": _as_numpy(robot.get_ang()).astype(np.float32),
        "joint_pos": _as_numpy(
            robot.get_dofs_position(leg_dof_indices.tolist())
        ).astype(np.float32),
        "joint_vel": _as_numpy(
            robot.get_dofs_velocity(leg_dof_indices.tolist())
        ).astype(np.float32),
        "command": np.asarray(command, dtype=np.float32),
    }


def _write_state_records(
    stream: Any,
    *,
    robot: Any,
    leg_dof_indices: np.ndarray,
    primitives: tuple[str, ...],
    timestamp_ns: int,
) -> None:
    positions = _as_numpy(robot.get_pos())
    quaternions = _as_numpy(robot.get_quat())
    joint_positions = _as_numpy(
        robot.get_dofs_position(leg_dof_indices.tolist())
    )
    for env_index, primitive_name in enumerate(primitives):
        qw, qx, qy, qz = (float(value) for value in quaternions[env_index])
        records = (
            _record(
                "/joint_states",
                env_index,
                timestamp_ns,
                {
                    "name": list(DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER),
                    "position": [float(value) for value in joint_positions[env_index]],
                },
            ),
            _record(
                "/lewm/go2/base_state",
                env_index,
                timestamp_ns,
                {
                    "roll_rad": _roll_from_quat_wxyz(qw, qx, qy, qz),
                    "pitch_rad": _pitch_from_quat_wxyz(qw, qx, qy, qz),
                    "pose_world": {
                        "position": {
                            "x": float(positions[env_index, 0]),
                            "y": float(positions[env_index, 1]),
                            "z": float(positions[env_index, 2]),
                        }
                    },
                },
            ),
            _record(
                "/lewm/episode_info",
                env_index,
                timestamp_ns,
                {
                    "split": "train",
                    "scene_family": "open_obstacle_field",
                    "scene_id": "go2_open_plane_footprint_calibration",
                    "primitive_name": primitive_name,
                },
            ),
        )
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect(args: argparse.Namespace) -> Path:
    platform_path = args.platform_manifest.resolve()
    registry_path = args.primitive_registry.resolve()
    platform = load_platform_manifest(platform_path)
    registry = PrimitiveRegistry.from_yaml(registry_path)
    primitives = tuple(args.primitive)
    if len(set(primitives)) != len(primitives):
        raise ValueError("primitive names must be unique")
    commands = np.stack(
        [_command_vector(registry, name) for name in primitives],
        axis=0,
    )

    checkpoint = (ROOT / platform["locomotion"]["policy_artifact"]["path"]).resolve()
    policy_cfg = (ROOT / platform["locomotion"]["policy_artifact"]["cfg_path"]).resolve()
    urdf = args.urdf.resolve() if args.urdf is not None else _resolve_bundled_urdf()

    initialize_genesis(backend=args.backend, seed=args.seed)
    import genesis as gs

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=float(platform["timing"]["physics_dt_s"]),
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf),
            pos=(0.0, 0.0, float(args.spawn_z_m)),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=False,
            visualization=False,
            collision=True,
        ),
        name="go2",
    )
    scene.build(n_envs=len(primitives))

    policy = GenesisGo2PPOPolicy(
        checkpoint_path=checkpoint,
        cfg_path=policy_cfg,
        device=args.policy_device,
    )
    policy.validate_rollout_robot(robot)
    leg_dof_indices = _resolve_rollout_leg_dof_indices(
        robot,
        DEFAULT_GO2_LEG_DOF_INDICES,
    )
    env_indices = list(range(len(primitives)))
    spawn_positions = np.tile(
        np.asarray((0.0, 0.0, args.spawn_z_m), dtype=np.float32),
        (len(primitives), 1),
    )
    spawn_quaternions = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        (len(primitives), 1),
    )
    stance = np.tile(policy.reset_stance_rad, (len(primitives), 1))
    robot.set_pos(spawn_positions, envs_idx=env_indices, zero_velocity=True)
    robot.set_quat(spawn_quaternions, envs_idx=env_indices, zero_velocity=False)
    robot.set_dofs_position(
        stance,
        leg_dof_indices.tolist(),
        envs_idx=env_indices,
    )
    robot.set_dofs_velocity(
        np.zeros_like(stance),
        leg_dof_indices.tolist(),
        envs_idx=env_indices,
    )
    policy.reset()

    physics_dt_s = float(platform["timing"]["physics_dt_s"])
    policy_dt_s = float(platform["timing"]["policy_dt_s"])
    policy_steps_per_block = int(
        round(registry.block_size * registry.command_dt_s / policy_dt_s)
    )
    physics_steps_per_policy = int(round(policy_dt_s / physics_dt_s))
    policy_dt_ns = int(round(policy_dt_s * 1e9))
    sim_time_ns = 0

    def step_policy() -> None:
        nonlocal sim_time_ns
        observation = _build_observation(robot, leg_dof_indices, commands)
        targets = policy.act(observation)
        robot.control_dofs_position(targets, leg_dof_indices.tolist())
        for _ in range(physics_steps_per_policy):
            scene.step()
        sim_time_ns += policy_dt_ns

    for _ in range(args.warmup_blocks * policy_steps_per_block):
        step_policy()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    raw_dir = output / "raw"
    source_dir = output / "source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    messages_path = raw_dir / "messages.jsonl"
    start_time_ns = sim_time_ns
    with messages_path.open("w", encoding="utf-8") as stream:
        for env_index, primitive_name in enumerate(primitives):
            stream.write(
                json.dumps(
                    _record(
                        "/lewm/go2/reset_event",
                        env_index,
                        sim_time_ns,
                        {
                            "reason": "post_settle_calibration_start",
                            "success": True,
                            "primitive_name": primitive_name,
                        },
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        sequence_id = 0
        for _block_index in range(args.record_blocks):
            block_start_ns = sim_time_ns
            for env_index, primitive_name in enumerate(primitives):
                payload = {
                    "sequence_id": sequence_id,
                    "block_size": int(registry.block_size),
                    "command_dt_s": float(registry.command_dt_s),
                    "primitive_name": primitive_name,
                    "command_source": "open_plane_footprint_calibration",
                    "command": [float(value) for value in commands[env_index]],
                }
                stream.write(
                    json.dumps(
                        _record(
                            "/lewm/go2/command_block",
                            env_index,
                            block_start_ns,
                            payload,
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                sequence_id += 1
            for _ in range(policy_steps_per_block):
                step_policy()
                _write_state_records(
                    stream,
                    robot=robot,
                    leg_dof_indices=leg_dof_indices,
                    primitives=primitives,
                    timestamp_ns=sim_time_ns,
                )

    policy_artifact = dict(platform["locomotion"]["policy_artifact"])
    source_summary = {
        "schema": "lewm_go2_open_plane_footprint_rollout_v0",
        "scene_id": "go2_open_plane_footprint_calibration",
        "family": "open_obstacle_field",
        "split": "train",
        "n_envs": len(primitives),
        "geometry": "infinite_plane_only",
        "backend": args.backend,
        "seed": int(args.seed),
        "warmup_blocks": int(args.warmup_blocks),
        "record_blocks": int(args.record_blocks),
        "primitives": list(primitives),
        "sim_duration_s": sim_time_ns / 1e9,
        "extra": {"policy_artifact": policy_artifact},
    }
    source_summary_path = source_dir / "summary.json"
    _write_json(source_summary_path, source_summary)
    raw_summary = {
        "schema": "lewm_go2_open_plane_footprint_raw_v0",
        "source_bag": _relative_or_absolute(source_dir),
        "messages_sha256": _sha256(messages_path),
        "source_summary_sha256": _sha256(source_summary_path),
        "record_start_time_ns": start_time_ns,
        "record_end_time_ns": sim_time_ns,
        "state_rate_hz": 1.0 / policy_dt_s,
        "source_artifacts": {
            "collector": {
                "path": _relative_or_absolute(Path(__file__)),
                "sha256": _sha256(Path(__file__)),
            },
            "platform_manifest": {
                "path": _relative_or_absolute(platform_path),
                "sha256": _sha256(platform_path),
            },
            "primitive_registry": {
                "path": _relative_or_absolute(registry_path),
                "sha256": _sha256(registry_path),
            },
            "go2_urdf": {
                "path": _relative_or_absolute(urdf),
                "sha256": _sha256(urdf),
            },
            "policy_checkpoint": {
                "path": _relative_or_absolute(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "policy_config": {
                "path": _relative_or_absolute(policy_cfg),
                "sha256": _sha256(policy_cfg),
            },
        },
    }
    _write_json(raw_dir / "summary.json", raw_summary)
    return messages_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-manifest",
        type=Path,
        default=ROOT / "config/go2_platform_manifest.yaml",
    )
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--spawn-z-m", type=float, default=0.375)
    parser.add_argument("--warmup-blocks", type=int, default=2)
    parser.add_argument("--record-blocks", type=int, default=11)
    parser.add_argument(
        "--primitive",
        action="append",
        default=None,
        help="Primitive to assign to one environment; repeat as needed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.primitive = tuple(args.primitive or DEFAULT_PRIMITIVES)
    if args.warmup_blocks < 0 or args.record_blocks < 2:
        raise ValueError("warmup blocks must be non-negative and record blocks >= 2")
    messages = collect(args)
    print(f"wrote {messages}")
    print(
        f"primitives={','.join(args.primitive)} "
        f"warmup_blocks={args.warmup_blocks} record_blocks={args.record_blocks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
