#!/usr/bin/env python3
"""Build or verify the denied-by-default V23 source closure."""
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
    "scripts/check_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_source_closure.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "e0697a6f2b8498ec64484b216f7366a8d7f199a5"
)
BASE_CHECKER_FILE_SHA256 = (
    "66fc73c9490cc6c7bc40dd869bf8136275b7967bc541564f4037cb6142f5444a"
)
BASE_CHECKER_BYTE_COUNT = 4_948

MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_source_manifest_2026-07-30.json"
)
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_clean_export_certification_2026-07-30.json"
)
EXECUTION_AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_execution_authorization_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_source_manifest"
)
PASS_STATUS_TEXT = "Go2 RGB V23 survival-output source closure: PASS"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "a7cf9692dd93212a82cb598d3175ff1c3598941b"
PREREGISTRATION_FILE_SHA256 = (
    "d5702759866138db1467778553ef8494d05f4593fcca14822050b1e0991180ae"
)
PREREGISTRATION_BYTE_COUNT = 14_294
V22_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_"
    "scientific_result_2026-07-30.json"
)
V22_SCIENTIFIC_RESULT_COMMIT = "f184a41ac99b1c66ea4db1e0b0a0845f23b48bbd"
V22_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0"
)
V22_SCIENTIFIC_RESULT_BYTE_COUNT = 18_445
V22_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "d9c0376f381bb65c4246c9ff12611f4b563698a0539f81c63b95e8b083de18a2"
)

RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
IMPLEMENTATION_PATHS = (
    RUNNER_RELATIVE_PATH,
    EXECUTOR_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
)
LIFECYCLE_PATHS = {
    "preregistration": PREREGISTRATION_RELATIVE_PATH,
    "predecessor_scientific_result": V22_SCIENTIFIC_RESULT_RELATIVE_PATH,
    "source_manifest": MANIFEST_RELATIVE_PATH,
    "source_review": SOURCE_REVIEW_RELATIVE_PATH,
    "clean_export_certification": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "execution_authority": EXECUTION_AUTHORITY_RELATIVE_PATH,
}
EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "V23 execution remains denied until recursive closure, independent source "
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
        raise PermissionError("frozen V22 source-closure checker binding changed")
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


_V22 = _source_only_module(
    "_lewm_v23_survival_output_source_closure_base",
    BASE_CHECKER_PATH,
)
if Path(_V22.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V23 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v18_object_space_height_volume.py"
    ),
    *IMPLEMENTATION_PATHS,
)
V22_PARENT_ENTRYPOINTS = tuple(_V22.ENTRYPOINTS[1:])
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys((*_V22.FORCED_DYNAMIC_SOURCES, *V22_PARENT_ENTRYPOINTS))
)
EXCLUDED_RUNTIME_CATEGORIES = _V22.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V22.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V22.FORBIDDEN_RUNNER_PREFIX


def _configure_private_base() -> None:
    modules = (
        _V22,
        _V22._V21,
        _V22._V21._V20,
        _V22._V21._V20._V18,
        _V22._V21._V20._V18._V13,
        _V22._V21._V20._V18._V13._BASE,
    )
    for module in modules:
        module.MANIFEST_PATH = MANIFEST_PATH
        module.SCHEMA = SCHEMA
        module.ENTRYPOINTS = ENTRYPOINTS
        module.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
        module.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V22._V21._V20._V18._V13._BASE.build_manifest = (
        _V22._V21._V20._V18._V13.build_manifest
    )


_configure_private_base()
build_manifest = _V22.build_manifest
discover_source_closure = _V22.discover_source_closure
verify_manifest = _V22.verify_manifest
_safe_source_path = _V22._safe_source_path
_read_regular_source = _V22._read_regular_source
_verify_tracked = _V22._verify_tracked
_write_manifest_exclusive = _V22._write_manifest_exclusive


def execution_denial_receipt_v23() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_execution_denial_v1",
        "status": "DENIED_INCOMPLETE_SOURCE_LIFECYCLE",
        "source_manifest_path": MANIFEST_RELATIVE_PATH,
        "source_review_path": SOURCE_REVIEW_RELATIVE_PATH,
        "clean_export_certification_path": (
            CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
        ),
        "execution_authority_path": EXECUTION_AUTHORITY_RELATIVE_PATH,
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
