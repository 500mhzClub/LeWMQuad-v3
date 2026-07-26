#!/usr/bin/env python3
"""Run the science-identical V10R retrieval-JEPA integrity replacement.

This source-only operational adapter reuses the frozen V10 runner.  Runtime,
RGB, checkpoint, tensor, and accelerator access remains behind V10's exact
authority and fresh-root reservation boundary.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_ACTION_CONDITIONED_NEXT_TARGET_RETRIEVAL_JEPA_"
    "V10R_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
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
    "_lewm_go2_rgb_retrieval_jepa_v10r_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py",
)
_V10 = _source_only_module(
    "_lewm_go2_rgb_retrieval_jepa_v10r_frozen_v10_runner",
    ROOT / contract.V10_RUNNER_RELATIVE_PATH,
)

_V10.contract = contract
_V10.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY

run_parent = _V10.run_parent
parse_args = _V10.parse_args


def main(argv: Sequence[str] | None = None) -> int:
    return _V10.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
