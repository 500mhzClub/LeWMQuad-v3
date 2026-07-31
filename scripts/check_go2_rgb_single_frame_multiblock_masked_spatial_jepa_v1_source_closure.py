#!/usr/bin/env python3
"""Build or verify the masked-spatial JEPA V1 source-only closure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER = "scripts/check_go2_multires_probe_source_closure_v3.py"
MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
    "source_manifest_2026-07-31.json"
)
ENTRYPOINTS = (
    "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "lewm/models/rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "scripts/evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "scripts/execute_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "scripts/launch_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "scripts/run_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
)
EXACT_DATASET_SOURCES = (
    "lewm/datasets/__init__.py",
    "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py",
    "lewm/datasets/go2_memory_role_place_triplets_v1.py",
    "lewm/datasets/go2_recurrent_h4_rgb_sequences.py",
    "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py",
)
FORCED_DYNAMIC_SOURCES = (
    BASE_CHECKER,
    *EXACT_DATASET_SOURCES,
    "scripts/check_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_source_closure.py",
    "lewm/tests/test_execute_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "lewm/tests/test_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_metrics.py",
    "lewm/tests/test_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_source_closure.py",
    "lewm/tests/test_launch_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "lewm/tests/test_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
    "lewm/tests/test_run_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
)
EXCLUDED_RUNTIME_CATEGORIES = (
    "generated_input",
    "dataset_payload",
    "rgb_payload",
    "checkpoint_or_tensor",
    "runtime_artifact",
    "probability_calibration",
    "g2",
    "navigation",
    "heldout",
    "sealed",
)


def _load_base() -> Any:
    path = ROOT / BASE_CHECKER
    spec = importlib.util.spec_from_file_location(
        "_lewm_masked_spatial_v1_source_closure_base", path
    )
    if spec is None or spec.loader is None:
        raise ImportError("source-closure base is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()
_BASE.ENTRYPOINTS = ENTRYPOINTS
_BASE.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
_INHERITED_SAFE_SOURCE_PATH = _BASE._safe_source_path
_INHERITED_MODULE_INDEX = _BASE._module_index


def _safe_source_path(relative: str) -> None:
    if relative in EXACT_DATASET_SOURCES:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise PermissionError(f"unsafe exact dataset adapter: {relative}")
        return
    _INHERITED_SAFE_SOURCE_PATH(relative)


def _module_index() -> tuple[dict[str, Path], dict[Path, str]]:
    by_module, by_path = _INHERITED_MODULE_INDEX()
    for relative in EXACT_DATASET_SOURCES:
        path = (ROOT / relative).resolve(strict=True)
        parts = list(Path(relative).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        if module in by_module or path in by_path:
            raise RuntimeError(f"exact dataset module repeats: {relative}")
        by_module[module] = path
        by_path[path] = module
    return by_module, by_path


_BASE._safe_source_path = _safe_source_path
_BASE._module_index = _module_index


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def discover_source_closure() -> tuple[str, ...]:
    return tuple(_BASE.discover_source_closure())


def build_manifest() -> dict[str, Any]:
    sources = discover_source_closure()
    bindings: list[dict[str, Any]] = []
    for relative in sources:
        raw = _BASE._read_regular_source(ROOT / relative)
        bindings.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    core: dict[str, Any] = {
        "schema": (
            "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
            "source_manifest_v1"
        ),
        "date": "2026-07-31",
        "status": "PASS_SOURCE_ONLY_RECURSIVE_CLOSURE",
        "entrypoints": list(ENTRYPOINTS),
        "forced_dynamic_sources": list(FORCED_DYNAMIC_SOURCES),
        "source_count": len(sources),
        "source_paths": list(sources),
        "source_bindings": bindings,
        "source_bindings_sha256": hashlib.sha256(
            _canonical_bytes(bindings)
        ).hexdigest(),
        "excluded_runtime_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
        "generated_input_open_count": 0,
        "dataset_or_rgb_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "heldout_or_sealed_open_count": 0,
        "execution_authorized": False,
        "whole_tree_export_authorized": False,
    }
    return {
        **core,
        "content_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _read_manifest() -> dict[str, Any]:
    path = ROOT / MANIFEST_RELATIVE_PATH
    raw = _BASE._read_regular_source(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("source manifest is not strict JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value) + b"\n":
        raise PermissionError("source manifest is not canonical JSON")
    return value


def _verify_tracked(paths: Iterable[str]) -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PermissionError("source closure contains an untracked path")


def verify_manifest(*, require_tracked: bool = False) -> dict[str, Any]:
    expected = _read_manifest()
    actual = build_manifest()
    if expected != actual:
        raise RuntimeError("masked-spatial JEPA source closure changed")
    if require_tracked:
        _verify_tracked(actual["source_paths"])
    return actual


def _write_manifest_exclusive(value: Any) -> None:
    path = ROOT / MANIFEST_RELATIVE_PATH
    raw = _canonical_bytes(value) + b"\n"
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
        print(json.dumps(build_manifest(), sort_keys=True, indent=2))
        return 0
    if args.write:
        _write_manifest_exclusive(build_manifest())
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print("Go2 masked-spatial JEPA V1 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
