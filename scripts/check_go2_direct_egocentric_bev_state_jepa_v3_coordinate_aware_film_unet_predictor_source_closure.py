#!/usr/bin/env python3
"""Build or verify the source-only Direct-BEV V3 predictor closure.

``--emit`` and the exclusive ``--write`` path are preparation operations and
do not require an existing V3 manifest.  With neither flag, verification is
strictly against the already-frozen manifest.  Every path is still mediated by
the reviewed source-only walker inherited from V2.
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
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
V2_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v2_integrity_"
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
    "_lewm_go2_direct_bev_v3_film_unet_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
if (
    contract.CONTRACT_RELATIVE_PATH != CONTRACT_RELATIVE_PATH
    or contract.FROZEN_V2_CONTRACT_RELATIVE_PATH
    != (
        "lewm/benchmarks/"
        "go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
    )
    or contract.PREREGISTRATION_RELATIVE_PATH
    != (
        "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
        "coordinate_aware_film_unet_predictor_preregistration_"
        "2026-07-26.md"
    )
    or contract.PREREGISTRATION_FILE_SHA256
    != (
        "be75f268816f422f1a40b7ee56dbf4bf544cd6893f9d3b296540ff4a98176c02"
    )
    or contract.PREREGISTRATION_BYTE_COUNT != 7_951
):
    raise PermissionError("Direct-BEV V3 source-closure identity changed")

_V2 = _source_only_module(
    "_lewm_go2_direct_bev_v3_film_unet_frozen_v2_source_closure",
    V2_CHECKER_RELATIVE_PATH,
)
if _V2.CONTRACT_RELATIVE_PATH != contract.FROZEN_V2_CONTRACT_RELATIVE_PATH:
    raise PermissionError("Direct-BEV V3 frozen V2 checker identity changed")


V3_REQUIRED_SOURCE_PATHS = frozenset({
    contract.CONTRACT_RELATIVE_PATH,
    contract.RUNNER_RELATIVE_PATH,
    contract.LAUNCHER_RELATIVE_PATH,
    contract.MODEL_RELATIVE_PATH,
    contract.CONTRACT_TEST_RELATIVE_PATH,
    contract.RUNNER_TEST_RELATIVE_PATH,
    contract.LAUNCHER_TEST_RELATIVE_PATH,
    contract.MODEL_TEST_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    contract.FROZEN_V2_CONTRACT_RELATIVE_PATH,
    V2_CHECKER_RELATIVE_PATH,
})
REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    *_V2.REQUIRED_DYNAMIC_SOURCE_PATHS,
    *V3_REQUIRED_SOURCE_PATHS,
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
        f"Direct-BEV V3 dynamic source roots are incomplete: {missing}"
    )


# Rebind the inherited V2 -> V1 -> V11 -> V10 -> AST-walker globals to V3.
# This reads source code only: never runtime data, tensors, outputs, held-out,
# or sealed material.
_V2.contract = contract
_V2.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V2._V1.contract = contract
_V2._V1.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V2._V1._V11.contract = contract
_V2._V1._V11.REQUIRED_DYNAMIC_SOURCE_PATHS = REQUIRED_DYNAMIC_SOURCE_PATHS
_V2._V1._V11._V10.contract = contract
_V2._V1._V11._V10.REQUIRED_DYNAMIC_SOURCE_PATHS = (
    REQUIRED_DYNAMIC_SOURCE_PATHS
)
_V2._V1._V11._V10._BASE.ENTRYPOINTS = tuple(
    contract.SOURCE_MANIFEST_ENTRYPOINTS
)
_V2._V1._V11._V10._BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _V2.discover_source_closure
build_manifest = _V2.build_manifest
verify_manifest = _V2.verify_manifest
_write_manifest_exclusive = _V2._write_manifest_exclusive
_read_regular_source = _V2._read_regular_source
_safe_source_path = _V2._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build a candidate manifest without requiring a frozen manifest",
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
    print("Go2 Direct BEV-State JEPA V3 FiLM U-Net source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
