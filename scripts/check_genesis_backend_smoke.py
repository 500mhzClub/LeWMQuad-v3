#!/usr/bin/env python3
"""Run a minimal Genesis backend smoke test."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.getenv("GS_BACKEND", "vulkan"),
        help="Genesis backend symbol to require, e.g. vulkan, amdgpu, cuda, gpu, cpu.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel env count for scene.build().",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps to run.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.n_envs) < 2:
        raise SystemExit("backend smoke requires --n-envs >= 2 to validate parallel envs")

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
        sim_options=gs.options.SimOptions(dt=float(args.dt)),
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.1))
    scene.build(n_envs=int(args.n_envs))
    for _ in range(int(args.steps)):
        scene.step()

    print(
        "genesis_backend_smoke_pass "
        f"backend={backend_name} n_envs={int(args.n_envs)} steps={int(args.steps)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
