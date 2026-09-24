#!/usr/bin/env python3
"""Isolated no-tensor preflight for overlap-tokenization V2."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_OVERLAPPING_TOKENIZATION_V2_INTEGRITY_REPLACEMENT_"
    "PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_go2_rgb_overlapping_tokenization_v2_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_overlapping_tokenization_v2_integrity_replacement.py",
)
_V1 = _source_only_module(
    "_lewm_go2_rgb_overlapping_tokenization_v2_frozen_v1_launcher",
    ROOT / contract.V1_LAUNCHER_RELATIVE_PATH,
)
_MOTION = _V1._MOTION
_BASE = _V1._BASE

_V1.contract = contract
_V1.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
_MOTION.contract = contract
_MOTION.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
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
