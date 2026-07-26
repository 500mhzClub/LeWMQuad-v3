#!/usr/bin/env python3
"""Build or verify the source-only V12 gate-timing closure."""
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
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
V11_CHECKER_RELATIVE_PATH = (
    "scripts/"
    "check_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_closure.py"
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
    "_lewm_pair_tubelet_v12_gate_timing_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_V11 = _source_only_module(
    "_lewm_pair_tubelet_v12_frozen_v11_closure",
    V11_CHECKER_RELATIVE_PATH,
)

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V11.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.V11_MODEL_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
    "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"V12 dynamic source roots are incomplete: {missing}")

# Reuse the frozen safe AST walker and canonical manifest implementation while
# rebinding every inherited layer to the V12 contract and recursive roots.
_V11.contract = contract
_V11.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V11._V10.contract = contract
_V11._V10.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V11._V10._BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_V11._V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V11.discover_source_closure
build_manifest = _V11.build_manifest
verify_manifest = _V11.verify_manifest
_write_manifest_exclusive = _V11._write_manifest_exclusive
_read_regular_source = _V11._read_regular_source
_safe_source_path = _V11._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--write", action="store_true")
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
    print("Go2 Masked Pair-Tubelet JEPA V12 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
