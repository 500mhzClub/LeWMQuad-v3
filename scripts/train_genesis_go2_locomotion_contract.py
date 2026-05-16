#!/usr/bin/env python3
"""Train Genesis Go2 PPO against the LeWM velocity-primitive contract."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import torch
import yaml


def _load_command_bank(registry_path: Path) -> tuple[list[str], list[list[float]]]:
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    default_train = bool(data.get("defaults", {}).get("train", True))
    names: list[str] = []
    commands: list[list[float]] = []
    for name, spec in data.get("primitives", {}).items():
        if spec.get("type") != "velocity_block":
            continue
        if not bool(spec.get("train", default_train)):
            continue
        command = spec.get("command", {})
        names.append(str(name))
        commands.append(
            [
                float(command.get("vx_body_mps", 0.0)),
                float(command.get("vy_body_mps", 0.0)),
                float(command.get("yaw_rate_radps", 0.0)),
            ]
        )
    if not commands:
        raise ValueError(f"no trainable velocity primitives in {registry_path}")
    return names, commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--examples-dir", type=Path, default=None)
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("-B", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=501)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--command-jitter-std", type=float, default=0.0)
    parser.add_argument("--tracking-ang-vel-scale", type=float, default=0.8)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    examples_dir = (
        args.examples_dir.resolve()
        if args.examples_dir is not None
        else repo_root / ".generated" / "upstream_genesis" / "locomotion"
    )
    registry_path = repo_root / "config" / "go2_primitive_registry.yaml"
    if not (examples_dir / "go2_env.py").is_file() or not (examples_dir / "go2_train.py").is_file():
        raise FileNotFoundError(f"missing upstream Genesis Go2 examples in {examples_dir}")

    sys.path.insert(0, str(examples_dir))
    os.chdir(examples_dir)

    import genesis as gs
    from go2_env import Go2Env
    from go2_train import get_cfgs, get_train_cfg
    from rsl_rl.runners import OnPolicyRunner

    command_names, command_bank = _load_command_bank(registry_path)
    command_tensor_cpu = torch.tensor(command_bank, dtype=torch.float32)
    command_low_cpu = command_tensor_cpu.min(dim=0).values
    command_high_cpu = command_tensor_cpu.max(dim=0).values

    class LeWMCommandGo2Env(Go2Env):
        def __init__(self, *env_args, **env_kwargs):
            self.lewm_command_bank = command_tensor_cpu.to(device=gs.device, dtype=gs.tc_float)
            self.lewm_command_low = command_low_cpu.to(device=gs.device, dtype=gs.tc_float)
            self.lewm_command_high = command_high_cpu.to(device=gs.device, dtype=gs.tc_float)
            self.lewm_command_jitter_std = float(args.command_jitter_std)
            super().__init__(*env_args, **env_kwargs)

        def _resample_commands(self, envs_idx):
            indices = torch.randint(
                0,
                self.lewm_command_bank.shape[0],
                (self.num_envs,),
                device=gs.device,
            )
            commands = self.lewm_command_bank[indices].clone()
            if self.lewm_command_jitter_std > 0.0:
                commands += torch.randn_like(commands) * self.lewm_command_jitter_std
                commands = torch.max(torch.min(commands, self.lewm_command_high), self.lewm_command_low)
            if envs_idx is None:
                self.commands.copy_(commands)
            else:
                torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)
    train_cfg["save_interval"] = int(args.save_interval)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = dict(reward_cfg["reward_scales"])
    reward_cfg["reward_scales"]["tracking_ang_vel"] = float(args.tracking_ang_vel_scale)

    command_cfg = dict(command_cfg)
    command_cfg["lin_vel_x_range"] = [float(command_low_cpu[0]), float(command_high_cpu[0])]
    command_cfg["lin_vel_y_range"] = [float(command_low_cpu[1]), float(command_high_cpu[1])]
    command_cfg["ang_vel_range"] = [float(command_low_cpu[2]), float(command_high_cpu[2])]
    command_cfg["lewm_command_names"] = command_names
    command_cfg["lewm_command_bank"] = command_bank
    command_cfg["sampler"] = "lewm_trainable_velocity_primitives"
    command_cfg["command_jitter_std"] = float(args.command_jitter_std)

    log_dir = Path("logs") / args.exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "cfgs.pkl").open("wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    with (log_dir / "lewm_command_bank.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "source": str(registry_path),
                "names": command_names,
                "commands": command_bank,
                "command_jitter_std": float(args.command_jitter_std),
            },
            f,
            sort_keys=False,
        )

    print("Genesis Go2 LeWM-contract PPO training")
    print(f"  exp_name:       {args.exp_name}")
    print(f"  num_envs:       {args.num_envs}")
    print(f"  max_iterations: {args.max_iterations}")
    print(f"  seed:           {args.seed}")
    print(f"  command_bank:   {', '.join(command_names)}")
    print(f"  command_low:    {[float(x) for x in command_low_cpu]}")
    print(f"  command_high:   {[float(x) for x in command_high_cpu]}")
    print(f"  log dir:        {log_dir}")
    print()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = LeWMCommandGo2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
