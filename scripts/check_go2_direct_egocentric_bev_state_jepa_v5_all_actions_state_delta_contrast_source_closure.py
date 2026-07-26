#!/usr/bin/env python3
"""Build or verify the source-only Direct-BEV V5 delta-contrast closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
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
    "_lewm_go2_direct_bev_v5_delta_contrast_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
V4_CHECKER_RELATIVE_PATH = contract.FROZEN_V4_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
_V4 = _source_only_module(
    "_lewm_go2_direct_bev_v5_delta_contrast_frozen_v4_source_closure",
    V4_CHECKER_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_V4_CONTRACT_RELATIVE_PATH
    != _V4.CONTRACT_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V5 source-closure identity changed")


V5_ADDITIVE_SOURCE_PATHS = frozenset({
    contract.MODEL_RELATIVE_PATH,
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.MODEL_TEST_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.LAUNCHER_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
})
if (
    set(contract.ADDITIVE_SOURCE_PATHS) != V5_ADDITIVE_SOURCE_PATHS
    or len(contract.ADDITIVE_SOURCE_PATHS) != 10
    or len(contract.SOURCE_PATHS) != 101
):
    raise PermissionError("Direct-BEV V5 ten-file source surface changed")

V5_REQUIRED_SOURCE_PATHS = frozenset({
    *V5_ADDITIVE_SOURCE_PATHS,
    contract.FROZEN_V4_CONTRACT_RELATIVE_PATH,
    V4_CHECKER_RELATIVE_PATH,
})
REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V4.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *V5_REQUIRED_SOURCE_PATHS,
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
        f"Direct-BEV V5 dynamic source roots are incomplete: {missing}"
    )


# Rebind the reviewed V4→V3→V2→V1→V11→V10 closure stack to V5.
# The stack remains source-only and excludes generated/runtime/protected data.
_V4.contract = contract
_V4.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V4._V3.contract = contract
_V4._V3.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V4._V3._V2.contract = contract
_V4._V3._V2.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V4._V3._V2._V1.contract = contract
_V4._V3._V2._V1.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V4._V3._V2._V1._V11.contract = contract
_V4._V3._V2._V1._V11.REQUIRED_DYNAMIC_SOURCE_PATHS = (
    REQUIRED_DYNAMIC_SOURCE_PATHS
)
_V4._V3._V2._V1._V11._V10.contract = contract
_V4._V3._V2._V1._V11._V10.REQUIRED_DYNAMIC_SOURCE_PATHS = (
    REQUIRED_DYNAMIC_SOURCE_PATHS
)
_V4._V3._V2._V1._V11._V10._BASE.ENTRYPOINTS = tuple(
    contract.SOURCE_MANIFEST_ENTRYPOINTS
)
_V4._V3._V2._V1._V11._V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V4.discover_source_closure
build_manifest = _V4.build_manifest
verify_manifest = _V4.verify_manifest
_write_manifest_exclusive = _V4._write_manifest_exclusive
_read_regular_source = _V4._read_regular_source
_safe_source_path = _V4._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build a candidate without requiring a frozen V5 manifest",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="exclusively create the candidate manifest; never replace one",
    )
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
    print("Go2 Direct BEV-State JEPA V5 delta-contrast source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
