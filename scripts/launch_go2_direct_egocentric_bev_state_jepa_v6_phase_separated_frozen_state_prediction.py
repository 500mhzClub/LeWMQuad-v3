#!/usr/bin/env python3
"""Authority-first launcher for the phase-separated Direct BEV V6 probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V6_"
    "PHASE_SEPARATED_FROZEN_STATE_PREDICTION_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v6_phase_separated_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V6 launcher identity changed")

_V5 = _source_only_module(
    "_lewm_direct_bev_v6_phase_separated_frozen_v5_launcher",
    ROOT / contract.FROZEN_V5_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the complete frozen authority stack to only V6 identities."""

    wrapper_path = Path(__file__).resolve()
    _V5.contract = contract
    _V5.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V5.__file__ = str(wrapper_path)
    _V5._rebind_inherited_launcher()

    owners = (
        _V5,
        _V5._V4,
        _V5._V4._V3,
        _V5._V4._V3._V2,
        _V5._V4._V3._V2._V1,
        _V5._V4._V3._V2._V1._V11,
        _V5._V4._V3._V2._V1._V11._BASE,
    )
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V6 contract did not reach the authority stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V6 preflight identity did not reach authority stack")
    if any(Path(owner.__file__).resolve() != wrapper_path for owner in owners):
        raise RuntimeError("V6 launcher path did not reach the authority stack")
    if _V5._V4._V3._V2._V1._V11._BASE.RUNNER_PATH != (
        ROOT / contract.RUNNER_RELATIVE_PATH
    ):
        raise RuntimeError("V6 runner path did not reach the preflight base")


_rebind_inherited_launcher()

NO_TENSOR_PREFLIGHT_PROGRAM = _V5.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V5.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V5.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
