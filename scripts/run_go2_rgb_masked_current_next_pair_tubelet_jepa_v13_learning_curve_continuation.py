#!/usr/bin/env python3
"""Run the final V13 learning-curve continuation through frozen V12 code."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_"
    "V13_LEARNING_CURVE_CONTINUATION_PREFLIGHT_JSON"
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
    "_lewm_go2_rgb_masked_pair_tubelet_v13_learning_curve_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation.py",
)
_V12 = _source_only_module(
    "_lewm_go2_rgb_masked_pair_tubelet_v13_frozen_v12_runner",
    ROOT
    / "scripts/"
    "run_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py",
)


def _rebind_inherited_contracts() -> None:
    """Make the V12, V11, and V10 runner layers resolve V13."""
    _V12.contract = contract
    _V12.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V12._V11.contract = contract
    _V12._V11.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V12._V11._V10.contract = contract
    _V12._V11._V10.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY


_rebind_inherited_contracts()

ScientificGateFailure = _V12.ScientificGateFailure
RGBOnlyLoader = _V12.RGBOnlyLoader
_phase_a_train = _V12._phase_a_train
_execute_after_reservation = _V12._execute_after_reservation


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_contracts()
    return _V12.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_contracts()
    return _V12.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
