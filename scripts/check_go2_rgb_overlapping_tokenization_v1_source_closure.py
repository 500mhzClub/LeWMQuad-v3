#!/usr/bin/env python3
"""Verify the recursive Python source closure of overlap tokenization V1.

The corrected motion source-closure adapter is reused with the overlap roots.
This module imports no project runtime, Torch, generated input, checkpoint,
dataset payload, or sealed material.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py"
)
MOTION_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_causal_motion_alignment_v1_source_closure.py"
)


def _load(name: str, relative: str):
    source = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "_overlapping_tokenization_v1_source_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_MOTION_CHECKER = _load(
    "_overlapping_tokenization_v1_motion_source_closure_checker",
    MOTION_CHECKER_RELATIVE_PATH,
)
_WALKER = _MOTION_CHECKER._WALKER

MANIFEST_PATH = ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH
SCHEMA = contract.SOURCE_MANIFEST_SCHEMA
ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

_WALKER.MANIFEST_PATH = MANIFEST_PATH
_WALKER.SCHEMA = SCHEMA
_WALKER.ENTRYPOINTS = ENTRYPOINTS
_WALKER.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES

discover_source_closure = _WALKER.discover_source_closure
build_manifest = _WALKER.build_manifest
verify_manifest = _WALKER.verify_manifest
_read_regular_source = _WALKER._read_regular_source
_safe_source_path = _WALKER._safe_source_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit",
        action="store_true",
        help="print the canonical manifest candidate instead of validating",
    )
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="also require every closure source to exist in the Git index",
    )
    args = parser.parse_args()
    if args.emit:
        print(json.dumps(build_manifest(), sort_keys=True, indent=2))
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print("Go2 RGB overlapping tokenization source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
