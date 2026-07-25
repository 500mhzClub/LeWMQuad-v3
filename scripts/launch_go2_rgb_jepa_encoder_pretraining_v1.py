#!/usr/bin/env python3
"""Authority-first launcher for Action-Residual JEPA V3 Live-Reference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_PATCH_WHITENED_ACTION_RESIDUAL_JEPA_"
    "V3_LIVE_REFERENCE_HINGE_PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_go2_rgb_jepa_encoder_pretraining_v3_live_reference_launcher_contract",
    ROOT / "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py",
)
_BASE = _source_only_module(
    "_lewm_go2_rgb_jepa_encoder_pretraining_v3_live_reference_base_launcher",
    ROOT / "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
)

# Reuse the reviewed authority-before-hardware and isolated no-tensor R9700
# preflight. Only the successor identities and receipt namespace change.
_BASE.contract = contract
_BASE.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
_BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
_BASE.__file__ = str(Path(__file__).resolve())

NO_TENSOR_PREFLIGHT_PROGRAM = _BASE.NO_TENSOR_PREFLIGHT_PROGRAM
parse_args = _BASE.parse_args


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
