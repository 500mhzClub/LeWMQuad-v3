#!/usr/bin/env python3
"""Build or verify the 103-file event-delta V2 source closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
FROZEN_V1_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "source_closure.py"
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
    "_lewm_two_mode_event_delta_v2_delegation_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    != FROZEN_V1_CHECKER_RELATIVE_PATH
):
    raise PermissionError("event-delta V2 source-closure identity changed")
_V1 = _source_module(
    "_lewm_two_mode_event_delta_v2_delegation_frozen_v1_closure",
    FROZEN_V1_CHECKER_RELATIVE_PATH,
)

EXPECTED_SOURCE_PATHS = frozenset({
    *_V1.contract.SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
})
if (
    len(_V1.contract.SOURCE_PATHS) != contract.FROZEN_V1_SOURCE_COUNT
    or len(contract.ADDITIVE_SOURCE_PATHS) != 5
    or len(EXPECTED_SOURCE_PATHS) != 103
    or set(contract.REUSED_SOURCE_PATHS) != set(_V1.contract.SOURCE_PATHS)
    or set(contract.SOURCE_PATHS) != EXPECTED_SOURCE_PATHS
    or set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    != EXPECTED_SOURCE_PATHS
):
    raise PermissionError("event-delta V2 closure is not frozen V1 plus five files")

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V1.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"event-delta V2 dynamic source roots missing: {missing}")

# Rebind the reviewed V1 -> rigid -> V3 -> V2 -> V1 -> direct -> V11 -> V10
# recursive walker instead of adding a second source-discovery implementation.
for _module in (
    _V1,
    _V1._RIGID,
    _V1._V3,
    _V1._V2,
    _V1._V1,
    _V1._DIRECT,
    _V1._V11,
    _V1._V10,
):
    _module.contract = contract
    _module.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_BASE = _V1._V10._BASE
_BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V1.discover_source_closure
build_manifest = _V1.build_manifest
verify_manifest = _V1.verify_manifest
_write_manifest_exclusive = _V1._write_manifest_exclusive
_read_regular_source = _V1._read_regular_source
_safe_source_path = _V1._safe_source_path


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
    print("Go2 two-mode event-delta joint-JEPA V2 delegation closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
