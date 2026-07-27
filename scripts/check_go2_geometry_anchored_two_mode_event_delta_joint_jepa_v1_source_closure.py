#!/usr/bin/env python3
"""Build or verify the two-mode event-delta joint-JEPA V1 source closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
FROZEN_RIGID_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_source_closure.py"
)


def _source_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_two_mode_event_delta_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_RIGID_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    != FROZEN_RIGID_CHECKER_RELATIVE_PATH
):
    raise PermissionError("two-mode event-delta source-closure identity changed")
_RIGID = _source_module(
    "_lewm_two_mode_event_delta_frozen_rigid_closure",
    FROZEN_RIGID_CHECKER_RELATIVE_PATH,
)

EXPECTED_SOURCE_PATHS = frozenset({
    *_RIGID.contract.SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
})
if (
    len(_RIGID.contract.SOURCE_PATHS) != contract.FROZEN_RIGID_SOURCE_COUNT
    or len(contract.ADDITIVE_SOURCE_PATHS) != 7
    or len(EXPECTED_SOURCE_PATHS) != 98
    or set(contract.SOURCE_PATHS) != EXPECTED_SOURCE_PATHS
    or set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    != EXPECTED_SOURCE_PATHS
):
    raise PermissionError("event-delta closure is not rigid V1 plus seven files")

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_RIGID.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"event-delta dynamic source roots missing: {missing}")

# Rebind the reviewed recursive walker instead of introducing a second walker.
_V3 = _RIGID._V3
_V2 = _V3._V2
_V1 = _V2._V1
_DIRECT = _V1._DIRECT
_V11 = _DIRECT._V11
_V10 = _V11._V10
for _module in (_RIGID, _V3, _V2, _V1, _DIRECT, _V11, _V10):
    _module.contract = contract
    _module.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_BASE = _V10._BASE
_BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _RIGID.discover_source_closure
build_manifest = _RIGID.build_manifest
verify_manifest = _RIGID.verify_manifest
_write_manifest_exclusive = _RIGID._write_manifest_exclusive
_read_regular_source = _RIGID._read_regular_source
_safe_source_path = _RIGID._safe_source_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args(argv)
    if args.emit:
        print(json.dumps(
            build_manifest(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ))
        return 0
    if args.write:
        _write_manifest_exclusive(build_manifest())
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print("Go2 two-mode event-delta joint-JEPA V1 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
