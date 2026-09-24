#!/usr/bin/env python3
"""Build or verify the source-only V11 masked pair-tubelet closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
V10_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py"
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
    "_lewm_masked_pair_tubelet_v11_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_V10 = _source_only_module(
    "_lewm_masked_pair_tubelet_v11_frozen_v10_source_closure",
    V10_CHECKER_RELATIVE_PATH,
)

REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V10.REQUIRED_DYNAMIC_SOURCE_PATHS,
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.V11_MODEL_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.V11_MODEL_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    contract.FROZEN_V10_RUNNER_RELATIVE_PATH,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"V11 dynamic source roots are incomplete: {missing}")

# Reuse the reviewed safe AST walker and canonical manifest implementation,
# replacing only the experiment contract and recursive roots.
_V10.contract = contract
_V10.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V10._BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V10.discover_source_closure
build_manifest = _V10.build_manifest
verify_manifest = _V10.verify_manifest
_write_manifest_exclusive = _V10._write_manifest_exclusive
_read_regular_source = _V10._read_regular_source
_safe_source_path = _V10._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args()
    if args.emit:
        print(
            json.dumps(
                build_manifest(),
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 0
    if args.write:
        _write_manifest_exclusive(build_manifest())
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print("Go2 Masked Pair-Tubelet JEPA V11 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
