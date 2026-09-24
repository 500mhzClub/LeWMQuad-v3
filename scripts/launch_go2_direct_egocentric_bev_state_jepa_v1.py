#!/usr/bin/env python3
"""Authority-first launcher for Direct Egocentric BEV-State JEPA V1."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V1_PREFLIGHT_JSON"
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
    "_lewm_go2_direct_egocentric_bev_state_jepa_v1_launcher_contract",
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py",
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve():
    raise PermissionError("Direct-BEV launcher path changed")
_V11 = _source_only_module(
    "_lewm_go2_direct_bev_v1_frozen_v11_launcher",
    ROOT / contract.FROZEN_V11_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_contracts() -> None:
    """Bind V11 and its authority base to the direct-BEV experiment."""

    wrapper_path = Path(__file__).resolve()
    _V11.contract = contract
    _V11.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V11.__file__ = str(wrapper_path)
    _V11._BASE.contract = contract
    _V11._BASE.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
    _V11._BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V11._BASE.__file__ = str(wrapper_path)


_rebind_inherited_contracts()

NO_TENSOR_PREFLIGHT_PROGRAM = _V11.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_contracts()
    return _V11.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_contracts()
    return _V11.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
