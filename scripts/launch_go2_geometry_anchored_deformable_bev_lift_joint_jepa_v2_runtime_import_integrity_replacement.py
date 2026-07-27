#!/usr/bin/env python3
"""Authority-first launcher for the joint-JEPA V2 import replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
FROZEN_V1_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
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
    "_lewm_geometry_anchored_joint_jepa_v2_import_launcher_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH:
    raise PermissionError("joint-JEPA V2 runtime-import launcher path changed")
_V1 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v2_import_frozen_v1_launcher",
    ROOT / FROZEN_V1_LAUNCHER_RELATIVE_PATH,
)


def _rebind_inherited_launcher() -> None:
    """Bind the frozen V1 launcher to the V2 contract, runner, and root."""

    _V1.contract = contract
    _V1.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _V1.LAUNCHER_PATH = LAUNCHER_PATH
    _V1.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
    _V1.OUTPUT_ROOT = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    _V1.__file__ = str(LAUNCHER_PATH)


_rebind_inherited_launcher()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V1.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
