#!/usr/bin/env python3
"""Build or verify the source-only Direct-BEV V1 recursive closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)
V11_CHECKER_RELATIVE_PATH = (
    "scripts/"
    "check_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_source_closure.py"
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
    "_lewm_go2_direct_bev_v1_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_V11_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    != V11_CHECKER_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV source-closure identity changed")
_V11 = _source_only_module(
    "_lewm_go2_direct_bev_v1_frozen_v11_source_closure",
    V11_CHECKER_RELATIVE_PATH,
)

# Dynamic importlib edges and source-only tests must be explicit roots.  The
# inherited V11 roots remain required because the direct runner and launcher
# deliberately reuse their reviewed custody machinery.
DIRECT_REQUIRED_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.MODEL_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.MODEL_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    contract.FROZEN_V11_CONTRACT_RELATIVE_PATH,
    contract.FROZEN_V11_RUNNER_RELATIVE_PATH,
    contract.FROZEN_V11_LAUNCHER_RELATIVE_PATH,
    contract.FROZEN_V11_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
})
REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V11.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *DIRECT_REQUIRED_SOURCE_PATHS,
    *contract.ADDITIVE_SOURCE_PATHS,
})
if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(
        f"Direct-BEV V1 dynamic source roots are incomplete: {missing}"
    )

# Rebind every global consulted by the inherited V11 -> V10 -> AST-walker
# function objects.  This keeps manifest construction source-only while using
# the direct experiment's schemas, paths, authority, and recursive roots.
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
    print("Go2 Direct Egocentric BEV-State JEPA V1 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
