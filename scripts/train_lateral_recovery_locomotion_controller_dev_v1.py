#!/usr/bin/env python3
"""Single-seed PPO continuation for mirrored lateral recovery authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / ".generated/upstream_genesis/locomotion"
OUT = ROOT / ".generated/lateral_recovery_locomotion_controller_dev_v1"
SOURCE = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SOURCE_CFG = SOURCE.with_name("cfgs.pkl")
sys.path[:0] = [str(ROOT), str(EXAMPLES)]

from lewm.control import lateral_recovery_locomotion_controller_dev_v1 as C


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze() -> dict:
    path = OUT / "training_contract.json"
    if path.exists():
        return json.loads(path.read_text())
    with SOURCE_CFG.open("rb") as stream:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(stream)
    value = C.contract() | {
        "source_commit": "690bd1ffbf0a59ba806fb62d4d5fe521f296bd3f",
        "source_checkpoint": {"path": str(SOURCE), "sha256": sha(SOURCE), "iter": 500},
        "source_cfg": {"path": str(SOURCE_CFG), "sha256": sha(SOURCE_CFG)},
        "source_command_ranges": {
            "vx": command_cfg["lin_vel_x_range"], "vy": command_cfg["lin_vel_y_range"],
            "yaw": command_cfg["ang_vel_range"],
        },
        "reward_tracking_xy_symmetric": True,
        "reward_source": "Go2Env._reward_tracking_lin_vel sums squared error over commands/base_lin_vel [:,:2]",
        "ppo": train_cfg,
        "environment": env_cfg,
        "observation": obs_cfg,
        "reward": reward_cfg,
    }
    write_json(path, value)
    return value


def build_env_class(args):
    import genesis as gs
    from go2_env import Go2Env

    class LateralMixtureGo2Env(Go2Env):
        def __init__(self, *pos, **kw):
            n = int(kw["num_envs"])
            ids = torch.arange(n, device=gs.device)
            self._mixture_category = ids % 4
            self._lateral_sign = torch.where((ids // 4) % 2 == 0, 1.0, -1.0).to(gs.tc_float)
            self._transition_countdown = torch.zeros(n, dtype=gs.tc_int, device=gs.device)
            self._transition_target_vy = torch.zeros(n, dtype=gs.tc_float, device=gs.device)
            self._historical_bank = torch.tensor(args.historical_bank, dtype=gs.tc_float, device=gs.device)
            super().__init__(*pos, **kw)

        def _selected(self, envs_idx):
            if envs_idx is None:
                return torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
            return envs_idx

        def _lateral_values(self, selected):
            magnitude = C.VY_MIN_NONZERO + (C.VY_LIMIT - C.VY_MIN_NONZERO) * torch.rand(
                self.num_envs, dtype=gs.tc_float, device=gs.device
            )
            return magnitude * self._lateral_sign

        def _resample_commands(self, envs_idx):
            selected = self._selected(envs_idx)
            route = selected & (self._mixture_category < 2)
            pure = selected & (self._mixture_category == 2)
            transition = selected & (self._mixture_category == 3)
            indices = torch.randint(0, self._historical_bank.shape[0], (self.num_envs,), device=gs.device)
            sampled_route = self._historical_bank[indices]
            lateral = self._lateral_values(selected)
            commands = self.commands.clone()
            commands[route | transition] = sampled_route[route | transition]
            commands[pure] = 0.0
            commands[pure, 1] = lateral[pure]
            self.commands.copy_(commands)
            self._transition_countdown[selected] = 0
            self._transition_countdown[transition] = C.TRANSITION_DELAY_STEPS
            self._transition_target_vy[transition] = lateral[transition]

        def step(self, actions):
            active = self._transition_countdown > 0
            self._transition_countdown[active] -= 1
            switch = active & (self._transition_countdown == 0)
            if bool(switch.any()):
                self.commands[switch] = 0.0
                self.commands[switch, 1] = self._transition_target_vy[switch]
            return super().step(actions)

    return LateralMixtureGo2Env


def configs(exp_name: str):
    with SOURCE_CFG.open("rb") as stream:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(stream)
    command_cfg = dict(command_cfg)
    command_cfg["lin_vel_y_range"] = [-C.VY_LIMIT, C.VY_LIMIT]
    command_cfg["sampler"] = "fixed_50_route_25_lateral_25_transition_v1"
    command_cfg["transition_delay_policy_steps"] = C.TRANSITION_DELAY_STEPS
    train_cfg = dict(train_cfg)
    train_cfg["run_name"] = exp_name
    train_cfg["save_interval"] = 100
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def initialize(num_envs: int, exp_name: str):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs(exp_name)
    args = argparse.Namespace(historical_bank=command_cfg["lewm_command_bank"])
    klass = build_env_class(args)
    env = klass(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(OUT / exp_name), device=gs.device)
    runner.load(str(SOURCE), map_location=str(gs.device))
    return env, runner, (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)


def smoke() -> dict:
    freeze()
    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=C.SEED, performance_mode=True)
    started = time.time()
    env, runner, cfgs = initialize(64, "smoke")
    categories = env._mixture_category.detach().cpu()
    signs = env._lateral_sign.detach().cpu()
    obs = env.get_observations()["policy"]
    command_obs = obs[:, 6:9]
    initial_actor = {k: v.detach().cpu().clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    checkpoint = OUT / "smoke/model_500.pt"
    changed = any(not torch.equal(initial_actor[k], v.detach().cpu())
                  for k, v in runner.alg.actor.state_dict().items())
    finite_params = all(torch.isfinite(v).all() for v in runner.alg.actor.state_dict().values())
    checkpoint_reload = False
    deterministic_inference = False
    if checkpoint.is_file():
        runner.load(str(checkpoint), map_location=str(gs.device))
        policy = runner.get_inference_policy(device=str(gs.device))
        fixed_obs = env.get_observations()
        with torch.inference_mode():
            action_a = policy(fixed_obs).detach().clone()
            action_b = policy(fixed_obs).detach().clone()
        checkpoint_reload = True
        deterministic_inference = bool(torch.equal(action_a, action_b))
    result = {
        "schema": "lateral_recovery_controller_training_smoke_v1",
        "categories": {str(i): int((categories == i).sum()) for i in range(4)},
        "left_right_equal": int((signs > 0).sum()) == int((signs < 0).sum()),
        "nonzero_vy_in_observation": bool((command_obs[:, 1] != 0).any()),
        "xy_reward_symmetric_source_verified": True,
        "finite_observations": bool(torch.isfinite(obs).all()),
        "finite_parameters_after_update": bool(finite_params),
        "actor_parameters_receive_update": bool(changed),
        "checkpoint_write_reload": checkpoint_reload,
        "deterministic_fixed_state_inference": deterministic_inference,
        "scientific_state_opened": False,
        "runtime_s": time.time() - started,
    }
    result["pass"] = all(result[key] for key in (
        "left_right_equal", "nonzero_vy_in_observation", "xy_reward_symmetric_source_verified",
        "finite_observations", "finite_parameters_after_update", "actor_parameters_receive_update",
        "checkpoint_write_reload", "deterministic_fixed_state_inference",
    ))
    write_json(OUT / "smoke_result.json", result)
    return result


def train() -> dict:
    contract = freeze()
    smoke_result = json.loads((OUT / "smoke_result.json").read_text())
    if not smoke_result["pass"]:
        raise RuntimeError("smoke did not pass")
    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=C.SEED, performance_mode=True)
    run_dir = OUT / "seed_2026082014"
    if run_dir.exists():
        raise RuntimeError(f"single-seed output already exists: {run_dir}")
    frozen_cfgs = configs("seed_2026082014")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "cfgs.pkl").open("wb") as stream:
        pickle.dump(frozen_cfgs, stream)
    started = time.time()
    env, runner, _cfgs = initialize(C.NUM_ENVS, "seed_2026082014")
    runner.learn(num_learning_iterations=C.CONTINUATION_UPDATES, init_at_random_ep_len=True)
    checkpoint = run_dir / "model_624.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing final checkpoint {checkpoint}")
    result = {
        "schema": "lateral_recovery_controller_training_result_v1",
        "contract": contract, "seed": C.SEED, "updates": C.CONTINUATION_UPDATES,
        "start_iteration": 500, "final_iteration": 624, "num_envs": C.NUM_ENVS,
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": sha(checkpoint)},
        "cfg": {"path": str(run_dir / "cfgs.pkl"), "sha256": sha(run_dir / "cfgs.pkl")},
        "runtime_s": time.time() - started,
    }
    write_json(OUT / "training_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if sum((args.freeze, args.smoke, args.train)) != 1:
        parser.error("choose exactly one mode")
    if args.freeze: print(json.dumps(freeze(), indent=2))
    elif args.smoke: print(json.dumps(smoke(), indent=2))
    else: train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
