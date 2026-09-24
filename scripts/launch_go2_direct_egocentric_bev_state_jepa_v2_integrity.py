#!/usr/bin/env python3
"""Authority-first launcher for the Direct BEV V2 integrity probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V2_INTEGRITY_PREFLIGHT_JSON"
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
    "_lewm_direct_egocentric_bev_state_jepa_v2_integrity_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v2_integrity.py",
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve():
    raise PermissionError("Direct-BEV V2 integrity launcher path changed")
_V1 = _source_only_module(
    "_lewm_direct_egocentric_bev_state_jepa_v2_integrity_frozen_v1_launcher",
    ROOT / contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the frozen V1 launcher and its inherited authority stack to V2."""

    wrapper_path = Path(__file__).resolve()
    _V1.contract = contract
    _V1.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V1.__file__ = str(wrapper_path)
    _V1._rebind_inherited_contracts()


_rebind_inherited_launcher()

NO_TENSOR_PREFLIGHT_PROGRAM = _V1.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V1.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
