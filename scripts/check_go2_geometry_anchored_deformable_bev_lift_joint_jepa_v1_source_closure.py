#!/usr/bin/env python3
"""Build or verify the recursive source-only closure for joint-JEPA V1."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
FROZEN_DIRECT_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v1_source_closure.py"
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
    "_lewm_geometry_anchored_joint_jepa_v1_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_DIRECT = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v1_frozen_direct_closure",
    FROZEN_DIRECT_CHECKER_RELATIVE_PATH,
)
_FROZEN_DIRECT_CONTRACT = _DIRECT.contract
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or _FROZEN_DIRECT_CONTRACT.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    != FROZEN_DIRECT_CHECKER_RELATIVE_PATH
):
    raise PermissionError("geometry-anchored closure identity changed")

ADDITIVE_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.MODEL_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.MODEL_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.LAUNCHER_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
})
EXPECTED_SOURCE_PATHS = frozenset({
    *_FROZEN_DIRECT_CONTRACT.SOURCE_PATHS,
    *ADDITIVE_SOURCE_PATHS,
})
if (
    set(contract.SOURCE_PATHS) != EXPECTED_SOURCE_PATHS
    or len(contract.SOURCE_PATHS) != len(EXPECTED_SOURCE_PATHS)
    or set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    != EXPECTED_SOURCE_PATHS
    or set(contract.SOURCE_MANIFEST_ENTRYPOINTS)
    != {contract.LAUNCHER_RELATIVE_PATH, contract.RUNNER_RELATIVE_PATH}
    or len(contract.SOURCE_MANIFEST_ENTRYPOINTS) != 2
):
    raise PermissionError(
        "joint-JEPA closure must be the exact frozen Direct-BEV recursive "
        "closure plus ten additive sources"
    )

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_DIRECT.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *ADDITIVE_SOURCE_PATHS,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"joint-JEPA dynamic source roots missing: {missing}")

# Reuse the reviewed AST walker and file-safety implementation while rebinding
# every inherited layer to this candidate's exact schema and source surface.
_DIRECT.contract = contract
_DIRECT.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_DIRECT._V11.contract = contract
_DIRECT._V11.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_DIRECT._V11._V10.contract = contract
_DIRECT._V11._V10.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_DIRECT._V11._V10._BASE.ENTRYPOINTS = tuple(
    contract.SOURCE_MANIFEST_ENTRYPOINTS
)
_DIRECT._V11._V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _DIRECT.discover_source_closure
build_manifest = _DIRECT.build_manifest
verify_manifest = _DIRECT.verify_manifest
_write_manifest_exclusive = _DIRECT._write_manifest_exclusive
_read_regular_source = _DIRECT._read_regular_source
_safe_source_path = _DIRECT._safe_source_path


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
    print("Go2 geometry-anchored deformable-BEV joint-JEPA V1 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
