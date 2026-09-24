#!/usr/bin/env python3
"""Run the one-shot V12 gate-timing successor through frozen V11 code.

Importing this wrapper is source-only.  It rebinds both levels of inherited
runner state to the V12 contract before exposing any execution entrypoint;
Torch and every generated runtime input remain deferred until after the V12
authority check and fresh-root reservation performed by the frozen runner.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_"
    "V12_GATE_TIMING_PREFLIGHT_JSON"
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
    "_lewm_go2_rgb_masked_pair_tubelet_v12_gate_timing_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py",
)
_V11 = _source_only_module(
    "_lewm_go2_rgb_masked_pair_tubelet_v12_frozen_v11_runner",
    ROOT
    / "scripts/"
    "run_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
)


def _rebind_inherited_contracts() -> None:
    """Make every inherited helper resolve authority through V12."""
    _V11.contract = contract
    _V11.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V11._V10.contract = contract
    _V11._V10.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY


_rebind_inherited_contracts()

# These aliases remain functions owned by the frozen V11 module.  Their global
# contract lookups resolve dynamically through the bindings above; in
# particular, update zero calls V12's two-conjunct timing gate while updates
# 100, 400, and 1000 retain the inherited V11 evaluators unchanged.
ScientificGateFailure = _V11.ScientificGateFailure
RGBOnlyLoader = _V11.RGBOnlyLoader
_load_post_reservation_stack = _V11._load_post_reservation_stack
_phase_a_parameter_partition = _V11._phase_a_parameter_partition
_phase_a_model = _V11._phase_a_model
_phase_a_gate_references = _V11._phase_a_gate_references
_phase_a_loss = _V11._phase_a_loss
_phase_a_diagnostics = _V11._phase_a_diagnostics
_phase_a_train = _V11._phase_a_train
_execute_after_reservation = _V11._execute_after_reservation


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_contracts()
    return _V11.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_contracts()
    return _V11.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
