#!/usr/bin/env python3
"""Build or verify the Direct-BEV semantic-anchor V1 source closure."""
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
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)


def _source_only_module(name: str, relative: str) -> Any:
    source = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v1_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_SIGNED_BOUNDARY = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v1_frozen_signed_boundary_closure",
    contract.FROZEN_SIGNED_BOUNDARY_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
)
_FROZEN_SIGNED_BOUNDARY_CONTRACT = _SIGNED_BOUNDARY.contract
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or _SIGNED_BOUNDARY.CONTRACT_RELATIVE_PATH
    != contract.FROZEN_SIGNED_BOUNDARY_CONTRACT_RELATIVE_PATH
    or contract.FROZEN_SIGNED_BOUNDARY_CONTRACT_RELATIVE_PATH
    != _FROZEN_SIGNED_BOUNDARY_CONTRACT.CONTRACT_RELATIVE_PATH
    or contract.FROZEN_SIGNED_BOUNDARY_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    != _FROZEN_SIGNED_BOUNDARY_CONTRACT.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    or tuple(contract.REUSED_SOURCE_PATHS)
    != tuple(_FROZEN_SIGNED_BOUNDARY_CONTRACT.SOURCE_PATHS)
):
    raise PermissionError("Direct-BEV semantic-anchor closure identity changed")

SEMANTIC_ANCHOR_ADDITIVE_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.MODEL_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.TEST_RELATIVE_PATH,
})
if (
    set(contract.ADDITIVE_SOURCE_PATHS)
    != SEMANTIC_ANCHOR_ADDITIVE_SOURCE_PATHS
    or len(contract.ADDITIVE_SOURCE_PATHS) != 6
    or len(contract.REUSED_SOURCE_PATHS) != 149
    or len(contract.SOURCE_PATHS) != 155
    or set(contract.REUSED_SOURCE_PATHS)
    & SEMANTIC_ANCHOR_ADDITIVE_SOURCE_PATHS
    or set(contract.SOURCE_PATHS)
    != set(contract.REUSED_SOURCE_PATHS) | SEMANTIC_ANCHOR_ADDITIVE_SOURCE_PATHS
):
    raise PermissionError("semantic-anchor six-file source surface changed")
if (
    set(contract.SOURCE_MANIFEST_ENTRYPOINTS)
    != {contract.RUNNER_RELATIVE_PATH, contract.LAUNCHER_RELATIVE_PATH}
    or len(contract.SOURCE_MANIFEST_ENTRYPOINTS) != 2
):
    raise PermissionError("semantic-anchor source entrypoints changed")

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_SIGNED_BOUNDARY.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *SEMANTIC_ANCHOR_ADDITIVE_SOURCE_PATHS,
    _FROZEN_SIGNED_BOUNDARY_CONTRACT.CONTRACT_RELATIVE_PATH,
    contract.FROZEN_SIGNED_BOUNDARY_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(
        f"semantic-anchor dynamic source roots are incomplete: {missing}"
    )

# Rebind every inherited closure-discovery layer to this exact 155-path family.
layers = [_SIGNED_BOUNDARY, *_SIGNED_BOUNDARY.layers]
for layer in layers:
    layer.contract = contract
    layer.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_BASE = _SIGNED_BOUNDARY._BASE
_BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _SIGNED_BOUNDARY.discover_source_closure
build_manifest = _SIGNED_BOUNDARY.build_manifest
verify_manifest = _SIGNED_BOUNDARY.verify_manifest
_write_manifest_exclusive = _SIGNED_BOUNDARY._write_manifest_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build a candidate without requiring a frozen V1 manifest",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="exclusively create the candidate manifest; never replace one",
    )
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args()
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
    print("Go2 Direct BEV signed-boundary semantic-anchor state V1 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
