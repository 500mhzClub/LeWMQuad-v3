#!/usr/bin/env python3
"""Run the upstream Genesis Go2 locomotion env with batched ROCm/GPU envs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=repo_root / ".generated" / "upstream_genesis" / "locomotion",
        help="Directory containing upstream go2_train.py and go2_env.py.",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("GS_BACKEND", "amdgpu"),
        help="Genesis backend symbol to require, e.g. amdgpu, cuda, gpu, cpu.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel env count. Must be >= 2.",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of env.step() calls.")
    parser.add_argument("--seed", type=int, default=3, help="Genesis seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    n_envs = int(args.n_envs)
    if n_envs < 2:
        raise SystemExit("locomotion env smoke requires --n-envs >= 2")

    examples_dir = args.examples_dir.resolve()
    missing = [name for name in ("go2_train.py", "go2_env.py") if not (examples_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing upstream Genesis example files under {examples_dir}: {', '.join(missing)}. "
            "Run scripts/fetch_genesis_go2_locomotion_examples.sh first."
        )
    sys.path.insert(0, str(examples_dir))

    try:
        import torch  # noqa: PLC0415
        import genesis as gs  # noqa: PLC0415
        from go2_env import Go2Env  # noqa: PLC0415
        from go2_train import get_cfgs  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "Failed to import upstream Go2 locomotion dependencies. "
            "Install rsl-rl-lib>=5.0.0, tensordict, and tensorboard into the active venv."
        ) from exc

    backend_name = args.backend.lower().strip()
    backend = getattr(gs, backend_name, None)
    if backend is None:
        version = getattr(gs, "__version__", "unknown")
        raise SystemExit(f"Genesis backend '{backend_name}' is unavailable in genesis-world {version}.")

    try:
        gs.init(
            backend=backend,
            precision="32",
            logging_level="warning",
            seed=int(args.seed),
            performance_mode=True,
        )
    except TypeError:
        gs.init(backend=backend, precision="32", logging_level="warning", seed=int(args.seed))

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Go2Env(
        num_envs=n_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    obs = env.reset()
    actions = torch.zeros((n_envs, env.num_actions), dtype=gs.tc_float, device=gs.device)
    rew = None
    dones = None
    for _ in range(int(args.steps)):
        obs, rew, dones, _infos = env.step(actions)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    policy_obs = obs["policy"]
    if tuple(policy_obs.shape[:1]) != (n_envs,):
        raise SystemExit(f"expected policy obs batch {n_envs}, got shape {tuple(policy_obs.shape)}")
    if not torch.isfinite(policy_obs).all():
        raise SystemExit("upstream Go2 env produced non-finite observations")
    if rew is None or dones is None:
        raise SystemExit("upstream Go2 env did not run any steps")
    if not torch.isfinite(rew).all():
        raise SystemExit("upstream Go2 env produced non-finite rewards")

    print(
        "genesis_upstream_go2_env_smoke_pass "
        f"backend={backend_name} n_envs={n_envs} steps={int(args.steps)} "
        f"obs_shape={tuple(policy_obs.shape)} done_count={int(dones.sum().detach().cpu())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
