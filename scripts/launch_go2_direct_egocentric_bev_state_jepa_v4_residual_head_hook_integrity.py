#!/usr/bin/env python3
"""Authority-first launcher for the Direct BEV V4 integrity successor."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V4_"
    "RESIDUAL_HEAD_HOOK_INTEGRITY_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v4_residual_head_hook_integrity_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V4 launcher path changed")
_V3 = _source_only_module(
    "_lewm_direct_bev_v4_residual_head_hook_integrity_frozen_v3_launcher",
    ROOT / contract.FROZEN_V3_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the frozen V3 authority stack to only the V4 identities."""

    wrapper_path = Path(__file__).resolve()
    _V3.contract = contract
    _V3.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V3.__file__ = str(wrapper_path)
    _V3._rebind_inherited_launcher()


_rebind_inherited_launcher()

NO_TENSOR_PREFLIGHT_PROGRAM = _V3.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V3.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V3.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
