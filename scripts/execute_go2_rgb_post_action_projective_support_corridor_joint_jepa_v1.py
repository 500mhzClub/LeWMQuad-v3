#!/usr/bin/env python3
"""Execute the one bound projective-support corridor joint-JEPA attempt.

This file is deliberately a thin reviewed entrypoint.  The core runner validates
the execution binding and reserves the sole attempt before importing the tensor
runtime or opening development inputs.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORE_RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)


def _core_runner() -> Any:
    name = "_lewm_post_action_projective_support_attempt_runner_v1"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, CORE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the reviewed projective-support runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_from_binding_v1(
    binding_path: Path,
    *,
    repository_root: Path = ROOT,
) -> Any:
    runner = _core_runner()
    entrypoint = getattr(runner, "run_from_execution_binding_v1", None)
    if not callable(entrypoint):
        raise RuntimeError("reviewed runner has no bound execution entrypoint")
    return entrypoint(
        Path(binding_path).absolute(),
        repository_root=Path(repository_root).absolute(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-binding", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    return int(
        execute_from_binding_v1(
            args.execution_binding,
            repository_root=args.repository_root,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
