#!/usr/bin/env python3
"""Authority-first launcher for the two-mode event-delta joint-JEPA V1."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
FROZEN_RIGID_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_two_mode_event_delta_launcher_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH:
    raise PermissionError("two-mode event-delta launcher path changed")
_RIGID = _source_module(
    "_lewm_two_mode_event_delta_frozen_rigid_launcher",
    ROOT / FROZEN_RIGID_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Rebind the reviewed authority-first launcher to the new identity."""

    _RIGID.contract = contract
    _RIGID.LAUNCHER_PATH = LAUNCHER_PATH
    _RIGID.__file__ = str(LAUNCHER_PATH)
    _RIGID._rebind_inherited_launcher()
    v3 = _RIGID._V3
    v2 = v3._V2
    base = v2._V1
    v3.contract = contract
    v3.LAUNCHER_PATH = LAUNCHER_PATH
    v2.contract = contract
    v2.LAUNCHER_PATH = LAUNCHER_PATH
    base.contract = contract
    base.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    base.LAUNCHER_PATH = LAUNCHER_PATH
    base.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
    base.OUTPUT_ROOT = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    base.__file__ = str(LAUNCHER_PATH)


_rebind_inherited_launcher()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _RIGID.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _RIGID.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
