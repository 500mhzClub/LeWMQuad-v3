#!/usr/bin/env python3
"""Build or verify the RGB JEPA encoder-pretraining V1 source closure.

This is a source-only AST walk.  It imports no tensor library and opens no
generated input, checkpoint, RGB payload, attempt output, or protected role.
Dynamic ``importlib`` edges are explicit roots in the reviewed contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
BASE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_source_closure_v3.py"
)

# These importlib edges are already visible before the experiment runner is
# complete.  Runner-specific dynamic roots are additionally frozen by the
# contract and checked by the focused source-closure test.
REQUIRED_DYNAMIC_SOURCE_PATHS = frozenset({
    CONTRACT_RELATIVE_PATH,
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py",
    "lewm/models/phase2d_spatial_lewm.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
    (
        "lewm/models/shared_observable_camera_ray_jepa_v5_"
        "protected_camera_adaptation_v4_tail_depth.py"
    ),
    BASE_CHECKER_RELATIVE_PATH,
})


def _source_module(name: str, relative: str) -> Any:
    source = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_rgb_jepa_encoder_pretraining_v1_closure_contract",
    CONTRACT_RELATIVE_PATH,
)
_BASE = _source_module(
    "_lewm_rgb_jepa_encoder_pretraining_v1_closure_base",
    BASE_CHECKER_RELATIVE_PATH,
)

if not REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
):
    missing = sorted(
        REQUIRED_DYNAMIC_SOURCE_PATHS
        - set(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
    )
    raise PermissionError(f"dynamic source roots are incomplete: {missing}")

_BASE.ENTRYPOINTS = tuple(contract.SOURCE_MANIFEST_ENTRYPOINTS)
_BASE.FORCED_DYNAMIC_SOURCES = tuple(
    contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)

discover_source_closure = _BASE.discover_source_closure
_read_regular_source = _BASE._read_regular_source
_safe_source_path = _BASE._safe_source_path


def build_manifest() -> dict[str, object]:
    sources = discover_source_closure()
    bindings = []
    for relative in sources:
        raw = _read_regular_source(ROOT / relative)
        bindings.append({
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        })
    core: dict[str, object] = {
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources":
            list(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES),
        "excluded_runtime_categories":
            list(contract.PROHIBITED_RUNTIME_CATEGORIES),
        "source_paths": list(sources),
        "source_bindings": bindings,
        "source_bindings_sha256":
            contract.canonical_json_sha256(bindings),
        "source_count": len(sources),
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": dict(contract.SOURCE_ONLY_AUTHORITY),
    }
    return contract.with_content_sha256(core)


def verify_manifest(*, require_tracked: bool = False) -> dict[str, object]:
    raw = _read_regular_source(ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH)
    expected = contract.validate_source_manifest(raw)
    actual = build_manifest()
    if expected != actual:
        expected_paths = set(expected.get("source_paths", []))
        actual_paths = set(actual["source_paths"])
        raise RuntimeError(
            "RGB JEPA source manifest changed: "
            f"missing={sorted(actual_paths - expected_paths)}, "
            f"stale={sorted(expected_paths - actual_paths)}"
        )
    if require_tracked:
        _BASE._verify_tracked(actual["source_paths"])
    return actual


def _write_manifest_exclusive(value: object) -> None:
    path = ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH
    raw = contract.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


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
    print("Go2 RGB JEPA encoder-pretraining V1 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
