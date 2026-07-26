#!/usr/bin/env python3
"""Build or verify the source-only Direct-BEV V4 integrity closure.

``--emit`` and the exclusive ``--write`` path prepare a candidate without an
existing V4 manifest.  With neither flag, verification is strict against the
already-frozen manifest.  Source discovery and reads remain mediated by the
reviewed V3 custody checker.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
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
    "_lewm_go2_direct_bev_v4_residual_head_hook_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
V3_CHECKER_RELATIVE_PATH = (
    contract.FROZEN_V3_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
_V3 = _source_only_module(
    "_lewm_go2_direct_bev_v4_residual_head_hook_frozen_v3_source_closure",
    V3_CHECKER_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_V3_CONTRACT_RELATIVE_PATH
    != _V3.CONTRACT_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V4 source-closure identity changed")


V4_ADDITIVE_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.LAUNCHER_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
})
if (
    set(contract.ADDITIVE_SOURCE_PATHS) != V4_ADDITIVE_SOURCE_PATHS
    or len(contract.ADDITIVE_SOURCE_PATHS) != 8
    or len(contract.SOURCE_PATHS) != 91
):
    raise PermissionError("Direct-BEV V4 eight-file source surface changed")

V4_REQUIRED_SOURCE_PATHS = frozenset({
    *V4_ADDITIVE_SOURCE_PATHS,
    contract.FROZEN_V3_CONTRACT_RELATIVE_PATH,
    V3_CHECKER_RELATIVE_PATH,
})
REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V3.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *V4_REQUIRED_SOURCE_PATHS,
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
        f"Direct-BEV V4 dynamic source roots are incomplete: {missing}"
    )


# Rebind V3 -> V2 -> V1 -> V11 -> V10 -> AST-walker globals to V4.  This
# opens source only: never generated inputs, tensors, runtime outputs, held-out,
# or sealed material.
_V3.contract = contract
_V3.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V3._V2.contract = contract
_V3._V2.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V3._V2._V1.contract = contract
_V3._V2._V1.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V3._V2._V1._V11.contract = contract
_V3._V2._V1._V11.REQUIRED_DYNAMIC_SOURCE_PATHS = (
    REQUIRED_DYNAMIC_SOURCE_PATHS
)
_V3._V2._V1._V11._V10.contract = contract
_V3._V2._V1._V11._V10.REQUIRED_DYNAMIC_SOURCE_PATHS = (
    REQUIRED_DYNAMIC_SOURCE_PATHS
)
_V3._V2._V1._V11._V10._BASE.ENTRYPOINTS = tuple(
    contract.SOURCE_MANIFEST_ENTRYPOINTS
)
_V3._V2._V1._V11._V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V3.discover_source_closure
build_manifest = _V3.build_manifest
verify_manifest = _V3.verify_manifest
_write_manifest_exclusive = _V3._write_manifest_exclusive
_read_regular_source = _V3._read_regular_source
_safe_source_path = _V3._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build a candidate without requiring a frozen V4 manifest",
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
    print("Go2 Direct BEV-State JEPA V4 integrity source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
