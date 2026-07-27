#!/usr/bin/env python3
"""Build or verify the Direct-BEV V11 recursive source closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py"
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
    "_lewm_direct_bev_v11_nested_observer_gate_dispatch_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_V10 = _source_only_module(
    "_lewm_direct_bev_v11_nested_observer_gate_dispatch_frozen_v10_closure",
    contract.FROZEN_V10_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or _V10.CONTRACT_RELATIVE_PATH
    != contract.FROZEN_V10_CONTRACT_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V11 source-closure identity changed")

V11_ADDITIVE_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.TEST_RELATIVE_PATH,
})
if (
    set(contract.ADDITIVE_SOURCE_PATHS) != V11_ADDITIVE_SOURCE_PATHS
    or len(contract.ADDITIVE_SOURCE_PATHS) != 5
    or len(contract.REUSED_SOURCE_PATHS) != 133
    or len(contract.SOURCE_PATHS) != 138
):
    raise PermissionError("Direct-BEV V11 five-file source surface changed")
if (
    set(contract.SOURCE_MANIFEST_ENTRYPOINTS)
    != {contract.RUNNER_RELATIVE_PATH, contract.LAUNCHER_RELATIVE_PATH}
    or len(contract.SOURCE_MANIFEST_ENTRYPOINTS) != 2
):
    raise PermissionError("Direct-BEV V11 source entrypoints changed")

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V10.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *V11_ADDITIVE_SOURCE_PATHS,
    contract.FROZEN_V10_CONTRACT_RELATIVE_PATH,
    contract.FROZEN_V10_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
})
if (
    contract.MODEL_RELATIVE_PATH
    not in contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    or not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
        contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    )
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(
        f"Direct-BEV V11 dynamic source roots are incomplete: {missing}"
    )

# Rebind every inherited checker layer to the V11 manifest identities.
layers = [_V10, *_V10.layers]
for layer in layers:
    layer.contract = contract
    layer.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_BASE = _V10._BASE
_BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V10.discover_source_closure
build_manifest = _V10.build_manifest
verify_manifest = _V10.verify_manifest
_write_manifest_exclusive = _V10._write_manifest_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build a candidate without requiring a frozen V11 manifest",
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
    print(
        "Go2 Direct BEV-State JEPA V11 nested-observer gate-dispatch "
        "integrity-replacement source closure: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
