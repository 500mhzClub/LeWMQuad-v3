#!/usr/bin/env python3
"""Build or verify the denied-by-default V25 recursive source closure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_PATH = (
    "scripts/check_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_closure.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "2b6178a4d876dc17c45fb340a4ab03ee302649b0"
)
BASE_CHECKER_FILE_SHA256 = (
    "465af32f3388b10a3658a32496c69ee663995f98057dbfc3fd642c49b682b5ac"
)
BASE_CHECKER_BYTE_COUNT = 7_965

MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_manifest_2026-07-30.json"
)
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_clean_export_certification_2026-07-30.json"
)
EXECUTION_AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_execution_authorization_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25_"
    "source_manifest"
)
PASS_STATUS_TEXT = "Go2 RGB V25 per-row temporal source closure: PASS"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "f00e20df3b429f9242516ac38f67fea587e04b22"
PREREGISTRATION_FILE_SHA256 = (
    "b9ce16b251415c50cb643daad919699c32965e23ddcd77d22bb3b69334f8b299"
)
PREREGISTRATION_BYTE_COUNT = 18_965
V24_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_scientific_result_2026-07-30.json"
)
V24_SCIENTIFIC_RESULT_COMMIT = "2824c80c54fc7502b1413b3371fc87c9206f82a2"
V24_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "f901d49eb9db0c39a068e67496b0b1cdaec954c9238edb40648140b924894e48"
)
V24_SCIENTIFIC_RESULT_BYTE_COUNT = 22_361
V24_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "0349f41da529b0c8658bf14ae51d85892a6f21fb461a281a9e157c7e7ff571dc"
)

RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)
IMPLEMENTATION_PATHS = (
    RUNNER_RELATIVE_PATH,
    EXECUTOR_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
)
LIFECYCLE_PATHS = {
    "preregistration": PREREGISTRATION_RELATIVE_PATH,
    "predecessor_scientific_result": V24_SCIENTIFIC_RESULT_RELATIVE_PATH,
    "source_manifest": MANIFEST_RELATIVE_PATH,
    "source_review": SOURCE_REVIEW_RELATIVE_PATH,
    "clean_export_certification": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "execution_authority": EXECUTION_AUTHORITY_RELATIVE_PATH,
}
EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "V25 execution remains denied until recursive closure, independent source "
    "review, narrow clean-export certification, and separate one-shot authority "
    "are complete and exact-bound"
)


def _source_only_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(f"source-only checker is absent: {relative}") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(
            f"source-only checker escaped or is not regular: {relative}"
        )
    source = path.read_bytes()
    if (
        len(source) != BASE_CHECKER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_CHECKER_FILE_SHA256
    ):
        raise PermissionError("frozen V24 source-closure checker binding changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only checker {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


_V24 = _source_only_module(
    "_lewm_v25_per_row_temporal_source_closure_base", BASE_CHECKER_PATH
)
if Path(_V24.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V25 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v18_object_space_height_volume.py"
    ),
    *IMPLEMENTATION_PATHS,
)
V24_PARENT_ENTRYPOINTS = tuple(_V24.ENTRYPOINTS[1:])
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys((*_V24.FORCED_DYNAMIC_SOURCES, *V24_PARENT_ENTRYPOINTS))
)
EXCLUDED_RUNTIME_CATEGORIES = _V24.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V24.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V24.FORBIDDEN_RUNNER_PREFIX


def _configure_private_base() -> None:
    modules = (
        _V24,
        _V24._V23,
        _V24._V23._V22,
        _V24._V23._V22._V21,
        _V24._V23._V22._V21._V20,
        _V24._V23._V22._V21._V20._V18,
        _V24._V23._V22._V21._V20._V18._V13,
        _V24._V23._V22._V21._V20._V18._V13._BASE,
    )
    for module in modules:
        module.MANIFEST_PATH = MANIFEST_PATH
        module.SCHEMA = SCHEMA
        module.ENTRYPOINTS = ENTRYPOINTS
        module.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
        module.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V24._V23._V22._V21._V20._V18._V13._BASE.build_manifest = (
        _V24._V23._V22._V21._V20._V18._V13.build_manifest
    )


_configure_private_base()
build_manifest = _V24.build_manifest
discover_source_closure = _V24.discover_source_closure
verify_manifest = _V24.verify_manifest
_safe_source_path = _V24._safe_source_path
_read_regular_source = _V24._read_regular_source
_verify_tracked = _V24._verify_tracked
_write_manifest_exclusive = _V24._write_manifest_exclusive


def execution_denial_receipt_v25() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_execution_denial_v1",
        "status": "DENIED_INCOMPLETE_SOURCE_LIFECYCLE",
        "source_manifest_path": MANIFEST_RELATIVE_PATH,
        "source_review_path": SOURCE_REVIEW_RELATIVE_PATH,
        "clean_export_certification_path": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
        "execution_authority_path": EXECUTION_AUTHORITY_RELATIVE_PATH,
        "recovery_state_opened": False,
        "checkpoint_opened": False,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args(argv)
    if args.emit or args.write:
        manifest = build_manifest()
        if args.require_tracked:
            _verify_tracked(manifest["source_paths"])
        if args.emit:
            print(json.dumps(manifest, sort_keys=True, indent=2))
        else:
            _write_manifest_exclusive(manifest)
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print(PASS_STATUS_TEXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
