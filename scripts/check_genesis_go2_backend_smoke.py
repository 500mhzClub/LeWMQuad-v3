#!/usr/bin/env python3
"""Run a Genesis Go2 backend smoke test with batched envs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_GO2_STANCE_RAD = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.9,
        0.9,
        0.9,
        0.9,
        -1.8,
        -1.8,
        -1.8,
        -1.8,
    ],
    dtype=np.float32,
)
DEFAULT_GO2_LEG_DOF_INDICES = tuple(range(6, 18))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.getenv("GS_BACKEND", "amdgpu"),
        help="Genesis backend symbol to require, e.g. amdgpu, cuda, gpu, cpu.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel env count for scene.build(). Must be >= 2.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of control/sim steps.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Optional Go2 URDF path. Defaults to the Genesis-bundled Go2 URDF.",
    )
    return parser.parse_args()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resolve_go2_urdf(gs, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return Path(gs.__file__).parent / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"


def main() -> int:
    args = parse_args()
    n_envs = int(args.n_envs)
    if n_envs < 2:
        raise SystemExit("Go2 backend smoke requires --n-envs >= 2 to validate parallel envs")

    import genesis as gs  # noqa: PLC0415

    backend_name = args.backend.lower().strip()
    backend = getattr(gs, backend_name, None)
    if backend is None:
        version = getattr(gs, "__version__", "unknown")
        symbols = [
            name
            for name in ("cpu", "gpu", "cuda", "amdgpu", "vulkan", "metal")
            if getattr(gs, name, None) is not None
        ]
        raise SystemExit(
            f"Genesis backend '{backend_name}' is unavailable in genesis-world {version}. "
            f"Available backend symbols: {', '.join(symbols) or 'none'}."
        )

    urdf = _resolve_go2_urdf(gs, args.urdf)
    if not urdf.is_file():
        raise SystemExit(f"Go2 URDF not found: {urdf}")

    init_kwargs = {
        "backend": backend,
        "precision": "32",
        "logging_level": "info",
        "seed": 1,
    }
    try:
        gs.init(**init_kwargs, performance_mode=True)
    except TypeError:
        gs.init(**init_kwargs)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=float(args.dt), gravity=(0.0, 0.0, -9.81)),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf),
            pos=(0.0, 0.0, 0.42),
            fixed=False,
            requires_jac_and_IK=False,
        ),
        name="go2",
    )
    scene.build(n_envs=n_envs)

    if int(robot.n_dofs) < 18:
        raise SystemExit(f"Go2 smoke expected at least 18 DOFs, got {robot.n_dofs}")

    leg_dofs = list(DEFAULT_GO2_LEG_DOF_INDICES)
    stance_batch = np.tile(DEFAULT_GO2_STANCE_RAD, (n_envs, 1)).astype(np.float32)
    robot.set_dofs_kp(np.full(len(leg_dofs), 40.0, dtype=np.float32), leg_dofs)
    robot.set_dofs_kv(np.full(len(leg_dofs), 1.0, dtype=np.float32), leg_dofs)
    robot.set_dofs_position(stance_batch, leg_dofs)
    robot.set_dofs_velocity(np.zeros_like(stance_batch), leg_dofs)

    for _ in range(int(args.steps)):
        robot.control_dofs_position(stance_batch, leg_dofs)
        scene.step()

    pos = _as_numpy(robot.get_pos())
    qpos = _as_numpy(robot.get_dofs_position(leg_dofs))
    if pos.shape != (n_envs, 3):
        raise SystemExit(f"expected base position shape {(n_envs, 3)}, got {pos.shape}")
    if qpos.shape != (n_envs, len(leg_dofs)):
        raise SystemExit(f"expected leg qpos shape {(n_envs, len(leg_dofs))}, got {qpos.shape}")
    if not np.isfinite(pos).all() or not np.isfinite(qpos).all():
        raise SystemExit("Go2 smoke produced non-finite state")

    print(
        "genesis_go2_backend_smoke_pass "
        f"backend={backend_name} n_envs={n_envs} steps={int(args.steps)} "
        f"n_dofs={int(robot.n_dofs)} urdf={urdf}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
