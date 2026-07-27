#!/usr/bin/env python3
"""Authority-first launcher for global rigid-BEV transport joint JEPA V1."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
FROZEN_V3_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
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
    "_lewm_global_rigid_bev_transport_launcher_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH:
    raise PermissionError("global rigid-BEV transport launcher path changed")
_V3 = _source_module(
    "_lewm_global_rigid_bev_transport_frozen_v3_launcher",
    ROOT / FROZEN_V3_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    _V3.contract = contract
    _V3.LAUNCHER_PATH = LAUNCHER_PATH
    _V3.__file__ = str(LAUNCHER_PATH)
    _V3._rebind_inherited_launcher()
    _V2 = _V3._V2
    _BASE = _V2._V1
    _V2.contract = contract
    _V2.LAUNCHER_PATH = LAUNCHER_PATH
    _BASE.contract = contract
    _BASE.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _BASE.LAUNCHER_PATH = LAUNCHER_PATH
    _BASE.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
    _BASE.OUTPUT_ROOT = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    _BASE.__file__ = str(LAUNCHER_PATH)


_rebind_inherited_launcher()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V3.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V3.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
