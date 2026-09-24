#!/usr/bin/env python3
"""Authority-first launcher for the joint-JEPA V3 scalar-hash replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
FROZEN_V2_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
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
    "_lewm_geometry_anchored_joint_jepa_v3_scalar_hash_launcher_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH:
    raise PermissionError("joint-JEPA V3 scalar-hash launcher path changed")
_V2 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v3_scalar_hash_frozen_v2_launcher",
    ROOT / FROZEN_V2_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the frozen V2 launcher to the V3 contract, runner, and root."""

    _V2.contract = contract
    _V2.LAUNCHER_PATH = LAUNCHER_PATH
    _V2.__file__ = str(LAUNCHER_PATH)
    _V2._rebind_inherited_launcher()


_rebind_inherited_launcher()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V2.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
