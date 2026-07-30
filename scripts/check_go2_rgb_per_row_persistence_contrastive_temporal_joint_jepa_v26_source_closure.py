#!/usr/bin/env python3
"""Build or verify the denied-by-default V26 recursive source closure."""
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
    "scripts/check_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_closure.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "43231c689547b66de83f3cafbfac270455a7a234"
)
BASE_CHECKER_FILE_SHA256 = (
    "338118e8a37a07d5edf068b34c34cf16ede0cf8ec5c5d355dd395e9768873229"
)
BASE_CHECKER_BYTE_COUNT = 8_016

MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_manifest_2026-07-30.json"
)
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_clean_export_certification_2026-07-30.json"
)
EXECUTION_AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_execution_authorization_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26_"
    "source_manifest"
)
PASS_STATUS_TEXT = "Go2 RGB V26 schema-compatibility source closure: PASS"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "0c277fd7350931a7993d5affc2d1d4633ffed916"
PREREGISTRATION_FILE_SHA256 = (
    "97061601af2922622673d7e4f8b4c1a6625edcdf899abd647373c28daa192a18"
)
PREREGISTRATION_BYTE_COUNT = 7_999
V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_terminal_failure_result_2026-07-30.json"
)
V25_TERMINAL_FAILURE_RESULT_COMMIT = (
    "26c8fd902319c06d4dbf25cab36a63ec2df44081"
)
V25_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "5c8d6d80ce24c60900c49f6cf49979c3001024666a2156d945e526b396dd1596"
)
V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 10_380
V25_TERMINAL_FAILURE_RESULT_CONTENT_SHA256 = (
    "59423f03ca153ca481d71ea4e88aaa625128ece4a15eb8b6253ae4f009272929"
)

RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)
IMPLEMENTATION_PATHS = (
    RUNNER_RELATIVE_PATH,
    EXECUTOR_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
)
LIFECYCLE_PATHS = {
    "preregistration": PREREGISTRATION_RELATIVE_PATH,
    "predecessor_terminal_failure": V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
    "source_manifest": MANIFEST_RELATIVE_PATH,
    "source_review": SOURCE_REVIEW_RELATIVE_PATH,
    "clean_export_certification": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "execution_authority": EXECUTION_AUTHORITY_RELATIVE_PATH,
}
EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "V26 execution remains denied until recursive closure, independent source "
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
        raise PermissionError("frozen V25 source-closure checker binding changed")
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


_V25 = _source_only_module(
    "_lewm_v26_schema_compat_source_closure_base", BASE_CHECKER_PATH
)
if Path(_V25.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V26 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v18_object_space_height_volume.py"
    ),
    *IMPLEMENTATION_PATHS,
)
V25_PARENT_ENTRYPOINTS = tuple(_V25.ENTRYPOINTS[1:])
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys((*_V25.FORCED_DYNAMIC_SOURCES, *V25_PARENT_ENTRYPOINTS))
)
EXCLUDED_RUNTIME_CATEGORIES = _V25.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V25.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V25.FORBIDDEN_RUNNER_PREFIX
EXPECTED_SOURCE_COUNT = 107


def _configure_private_base() -> None:
    _V25.MANIFEST_PATH = MANIFEST_PATH
    _V25.SCHEMA = SCHEMA
    _V25.ENTRYPOINTS = ENTRYPOINTS
    _V25.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
    _V25.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V25._configure_private_base()


_configure_private_base()
build_manifest = _V25.build_manifest
discover_source_closure = _V25.discover_source_closure
verify_manifest = _V25.verify_manifest
_safe_source_path = _V25._safe_source_path
_read_regular_source = _V25._read_regular_source
_verify_tracked = _V25._verify_tracked
_write_manifest_exclusive = _V25._write_manifest_exclusive


def execution_denial_receipt_v26() -> dict[str, Any]:
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
        if len(manifest["source_paths"]) != EXPECTED_SOURCE_COUNT:
            raise RuntimeError("V26 recursive source count changed")
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
