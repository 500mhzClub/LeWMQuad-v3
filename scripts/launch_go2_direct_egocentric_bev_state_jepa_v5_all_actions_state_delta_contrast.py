#!/usr/bin/env python3
"""Authority-first launcher for the Direct BEV V5 delta contrast probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V5_"
    "ALL_ACTIONS_STATE_DELTA_CONTRAST_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v5_state_delta_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V5 launcher identity changed")

_V4 = _source_only_module(
    "_lewm_direct_bev_v5_state_delta_frozen_v4_launcher",
    ROOT / contract.FROZEN_V4_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the complete frozen authority stack to only V5 identities."""

    wrapper_path = Path(__file__).resolve()
    _V4.contract = contract
    _V4.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V4.__file__ = str(wrapper_path)
    _V4._rebind_inherited_launcher()

    owners = (
        _V4,
        _V4._V3,
        _V4._V3._V2,
        _V4._V3._V2._V1,
        _V4._V3._V2._V1._V11,
        _V4._V3._V2._V1._V11._BASE,
    )
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V5 contract did not reach the authority stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V5 preflight identity did not reach the authority stack")
    if any(Path(owner.__file__).resolve() != wrapper_path for owner in owners):
        raise RuntimeError("V5 launcher path did not reach the authority stack")
    if _V4._V3._V2._V1._V11._BASE.RUNNER_PATH != (
        ROOT / contract.RUNNER_RELATIVE_PATH
    ):
        raise RuntimeError("V5 runner path did not reach the preflight base")


_rebind_inherited_launcher()

NO_TENSOR_PREFLIGHT_PROGRAM = _V4.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V4.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V4.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
