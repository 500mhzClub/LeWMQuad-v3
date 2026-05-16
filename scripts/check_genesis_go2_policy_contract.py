#!/usr/bin/env python3
"""Probe a Genesis Go2 PPO checkpoint against the LeWM command contract.

The upstream Genesis Go2 env exposes a 3-D command vector in the policy
observation. This check forces each trainable LeWM velocity primitive into that
command vector, runs a short rollout, and verifies that measured body-frame
motion has the requested direction.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _latest_exp_name(logs_dir: Path) -> str:
    runs = [p for p in logs_dir.iterdir() if p.is_dir() and (p / "cfgs.pkl").is_file()]
    if not runs:
        raise FileNotFoundError(f"no Genesis Go2 runs found under {logs_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime).name


def _load_trainable_velocity_primitives(registry_path: Path) -> list[tuple[str, np.ndarray]]:
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    default_train = bool(data.get("defaults", {}).get("train", True))
    primitives: list[tuple[str, np.ndarray]] = []
    for name, spec in data.get("primitives", {}).items():
        if spec.get("type") != "velocity_block":
            continue
        if not bool(spec.get("train", default_train)):
            continue
        command = spec.get("command", {})
        primitives.append(
            (
                str(name),
                np.asarray(
                    [
                        float(command.get("vx_body_mps", 0.0)),
                        float(command.get("vy_body_mps", 0.0)),
                        float(command.get("yaw_rate_radps", 0.0)),
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if not primitives:
        raise ValueError(f"no trainable velocity primitives in {registry_path}")
    return primitives


def _check_primitive(
    name: str,
    requested: np.ndarray,
    measured: np.ndarray,
    resets: int,
    *,
    min_vx_abs: float,
    min_yaw_abs: float,
    min_yaw_fraction: float,
    hold_vx_abs: float,
    hold_yaw_abs: float,
    pure_yaw_max_vx_abs: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if resets:
        failures.append(f"{resets} resets")

    if name == "hold":
        if abs(float(measured[0])) > hold_vx_abs:
            failures.append(f"hold vx {measured[0]:.3f} exceeds {hold_vx_abs:.3f}")
        if abs(float(measured[2])) > hold_yaw_abs:
            failures.append(f"hold yaw {measured[2]:.3f} exceeds {hold_yaw_abs:.3f}")
        return not failures, failures

    if abs(float(requested[0])) >= 0.05:
        if np.sign(measured[0]) != np.sign(requested[0]) or abs(float(measured[0])) < min_vx_abs:
            failures.append(
                f"vx direction/magnitude mismatch requested {requested[0]:.3f}, measured {measured[0]:.3f}"
            )

    if abs(float(requested[2])) >= 0.10:
        yaw_threshold = max(min_yaw_abs, abs(float(requested[2])) * min_yaw_fraction)
        if np.sign(measured[2]) != np.sign(requested[2]) or abs(float(measured[2])) < yaw_threshold:
            failures.append(
                f"yaw direction/magnitude mismatch requested {requested[2]:.3f}, measured {measured[2]:.3f}"
            )
        if abs(float(requested[0])) < 0.05 and abs(float(measured[0])) > pure_yaw_max_vx_abs:
            failures.append(
                f"pure-yaw vx drift {measured[0]:.3f} exceeds {pure_yaw_max_vx_abs:.3f}"
            )

    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--examples-dir", type=Path, default=None)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--horizon-s", type=float, default=5.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--min-vx-abs", type=float, default=0.05)
    parser.add_argument("--min-yaw-abs", type=float, default=0.05)
    parser.add_argument("--min-yaw-fraction", type=float, default=0.25)
    parser.add_argument("--hold-vx-abs", type=float, default=0.05)
    parser.add_argument("--hold-yaw-abs", type=float, default=0.05)
    parser.add_argument("--pure-yaw-max-vx-abs", type=float, default=0.15)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    examples_dir = (
        args.examples_dir.resolve()
        if args.examples_dir is not None
        else repo_root / ".generated" / "upstream_genesis" / "locomotion"
    )
    logs_dir = examples_dir / "logs"
    exp_name = args.exp_name or _latest_exp_name(logs_dir)
    log_dir = logs_dir / exp_name
    registry_path = repo_root / "config" / "go2_primitive_registry.yaml"

    if not (log_dir / "cfgs.pkl").is_file():
        raise FileNotFoundError(f"missing cfgs.pkl for {exp_name}: {log_dir}")
    if not (log_dir / f"model_{args.ckpt}.pt").is_file():
        raise FileNotFoundError(f"missing checkpoint model_{args.ckpt}.pt in {log_dir}")

    sys.path.insert(0, str(examples_dir))
    os.chdir(examples_dir)

    import genesis as gs
    from go2_env import Go2Env
    from rsl_rl.runners import OnPolicyRunner

    primitives = _load_trainable_velocity_primitives(registry_path)
    names: list[str] = []
    commands: list[np.ndarray] = []
    for name, command in primitives:
        for _ in range(args.repeats):
            names.append(name)
            commands.append(command)
    command_tensor_cpu = torch.tensor(np.stack(commands), dtype=torch.float32)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=20260516, performance_mode=True)

    with (log_dir / "cfgs.pkl").open("rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env_cfg = dict(env_cfg)
    env_cfg["resampling_time_s"] = max(float(args.horizon_s) + 1.0, 10_000.0)

    env = Go2Env(
        num_envs=len(commands),
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    command_tensor = command_tensor_cpu.to(device=gs.device, dtype=gs.tc_float)
    horizon_steps = max(1, int(round(args.horizon_s / env.dt)))
    warmup_steps = min(horizon_steps - 1, max(0, int(round(args.warmup_s / env.dt))))

    velocity_samples: list[np.ndarray] = []
    done_samples: list[np.ndarray] = []
    with torch.no_grad():
        env.reset()
        for step in range(horizon_steps):
            env.commands.copy_(command_tensor)
            env._update_observation()
            actions = policy(env.get_observations())
            _, _, dones, _ = env.step(actions)
            if step >= warmup_steps:
                sample = torch.cat(
                    [env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]],
                    dim=1,
                )
                velocity_samples.append(sample.detach().cpu().numpy())
            done_samples.append(dones.detach().cpu().numpy())

    velocities = np.stack(velocity_samples, axis=0)
    resets = np.stack(done_samples, axis=0).sum(axis=0)

    print(
        f"policy_contract_probe exp={exp_name} ckpt={args.ckpt} "
        f"envs={len(commands)} horizon_s={args.horizon_s:.2f} warmup_s={args.warmup_s:.2f}"
    )
    print(
        "name requested_vx requested_vy requested_yaw "
        "mean_vx mean_vy mean_yaw resets result"
    )

    failures: list[str] = []
    for primitive_name, requested in primitives:
        idx = [i for i, name in enumerate(names) if name == primitive_name]
        measured = velocities[:, idx, :].mean(axis=(0, 1))
        reset_count = int(resets[idx].sum())
        passed, reasons = _check_primitive(
            primitive_name,
            requested,
            measured,
            reset_count,
            min_vx_abs=args.min_vx_abs,
            min_yaw_abs=args.min_yaw_abs,
            min_yaw_fraction=args.min_yaw_fraction,
            hold_vx_abs=args.hold_vx_abs,
            hold_yaw_abs=args.hold_yaw_abs,
            pure_yaw_max_vx_abs=args.pure_yaw_max_vx_abs,
        )
        result = "PASS" if passed else "FAIL"
        print(
            f"{primitive_name} "
            f"{requested[0]:.3f} {requested[1]:.3f} {requested[2]:.3f} "
            f"{measured[0]:.3f} {measured[1]:.3f} {measured[2]:.3f} "
            f"{reset_count} {result}"
        )
        if not passed:
            failures.append(f"{primitive_name}: {', '.join(reasons)}")

    if failures:
        print("policy_contract_probe_fail")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("policy_contract_probe_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
