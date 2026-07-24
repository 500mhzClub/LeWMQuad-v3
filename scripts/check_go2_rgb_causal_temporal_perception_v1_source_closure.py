#!/usr/bin/env python3
"""Verify the recursive Python source closure of the temporal perception probe.

This reuses the already reviewed, source-only AST closure walker from the V3
multiresolution probe with new roots.  It never imports project runtime modules
and never opens generated inputs, checkpoints, datasets, or sealed material.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_source_closure_v3.py"
)
MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "source_manifest_2026-07-24.json"
)
SCHEMA = "lewm_go2_rgb_causal_temporal_perception_v1_source_manifest"

ENTRYPOINTS = (
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
    "scripts/run_go2_rgb_causal_temporal_perception_v1.py",
)

# importlib loads these exact reviewed files, so their edges are not all
# discoverable from normal import statements.
FORCED_DYNAMIC_SOURCES = (
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py",
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_contract.py",
    "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py",
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_runner.py",
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_evaluator.py",
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_receipt_boundary.py",
    "scripts/check_go2_rgb_causal_temporal_perception_v1_source_closure.py",
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_source_closure.py",
    BASE_CHECKER_RELATIVE_PATH,
)


def _load_base_checker():
    source = ROOT / BASE_CHECKER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_temporal_v1_source_closure_base",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the reviewed source-closure walker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_checker()
_BASE.MANIFEST_PATH = MANIFEST_PATH
_BASE.SCHEMA = SCHEMA
_BASE.ENTRYPOINTS = ENTRYPOINTS
_BASE.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES

discover_source_closure = _BASE.discover_source_closure
build_manifest = _BASE.build_manifest
verify_manifest = _BASE.verify_manifest
_read_regular_source = _BASE._read_regular_source
_safe_source_path = _BASE._safe_source_path


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
    print("Go2 RGB causal temporal perception source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
