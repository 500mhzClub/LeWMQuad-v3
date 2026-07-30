#!/usr/bin/env python3
"""Build or verify the denied-by-default V24 source closure."""
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
    "scripts/check_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23_source_closure.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "44938145362e5accdf8e12b906bfbaa970d62f25"
)
BASE_CHECKER_FILE_SHA256 = (
    "255ef1a1f62b814fe64c9703bf79dc40fc2ea326214ec630928ca71aaf837076"
)
BASE_CHECKER_BYTE_COUNT = 7_994

MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_manifest_2026-07-30.json"
)
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_clean_export_certification_2026-07-30.json"
)
EXECUTION_AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_execution_authorization_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24_"
    "source_manifest"
)
PASS_STATUS_TEXT = "Go2 RGB V24 core-protected source closure: PASS"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "475f1867149f5c5b764973bb5a371de83c29c3eb"
PREREGISTRATION_FILE_SHA256 = (
    "ad0514668b20fd3bb58a2c70e71bb153428f3a9b121c1f8b64ca6e08965c6933"
)
PREREGISTRATION_BYTE_COUNT = 12_137
V23_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_scientific_result_2026-07-30.json"
)
V23_SCIENTIFIC_RESULT_COMMIT = "04b0fa48c6c4e10868c2f302bc51100394e3907e"
V23_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "753c91babd4f7116444654167d2507ffb52d22f970fc926c05d287683954c994"
)
V23_SCIENTIFIC_RESULT_BYTE_COUNT = 20_640
V23_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "a5a6b8aa7312706d2ae3a5b53e39370462e9de6eda6b7a2ca4e2e0226a518ed8"
)

RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24.py"
)
IMPLEMENTATION_PATHS = (
    RUNNER_RELATIVE_PATH,
    EXECUTOR_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
)
LIFECYCLE_PATHS = {
    "preregistration": PREREGISTRATION_RELATIVE_PATH,
    "predecessor_scientific_result": V23_SCIENTIFIC_RESULT_RELATIVE_PATH,
    "source_manifest": MANIFEST_RELATIVE_PATH,
    "source_review": SOURCE_REVIEW_RELATIVE_PATH,
    "clean_export_certification": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "execution_authority": EXECUTION_AUTHORITY_RELATIVE_PATH,
}
EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "V24 execution remains denied until recursive closure, independent source "
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
        raise PermissionError("frozen V23 source-closure checker binding changed")
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


_V23 = _source_only_module(
    "_lewm_v24_core_protected_source_closure_base",
    BASE_CHECKER_PATH,
)
if Path(_V23.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V24 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v18_object_space_height_volume.py"
    ),
    *IMPLEMENTATION_PATHS,
)
V23_PARENT_ENTRYPOINTS = tuple(_V23.ENTRYPOINTS[1:])
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys((*_V23.FORCED_DYNAMIC_SOURCES, *V23_PARENT_ENTRYPOINTS))
)
EXCLUDED_RUNTIME_CATEGORIES = _V23.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V23.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V23.FORBIDDEN_RUNNER_PREFIX


def _configure_private_base() -> None:
    modules = (
        _V23,
        _V23._V22,
        _V23._V22._V21,
        _V23._V22._V21._V20,
        _V23._V22._V21._V20._V18,
        _V23._V22._V21._V20._V18._V13,
        _V23._V22._V21._V20._V18._V13._BASE,
    )
    for module in modules:
        module.MANIFEST_PATH = MANIFEST_PATH
        module.SCHEMA = SCHEMA
        module.ENTRYPOINTS = ENTRYPOINTS
        module.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
        module.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V23._V22._V21._V20._V18._V13._BASE.build_manifest = (
        _V23._V22._V21._V20._V18._V13.build_manifest
    )


_configure_private_base()
build_manifest = _V23.build_manifest
discover_source_closure = _V23.discover_source_closure
verify_manifest = _V23.verify_manifest
_safe_source_path = _V23._safe_source_path
_read_regular_source = _V23._read_regular_source
_verify_tracked = _V23._verify_tracked
_write_manifest_exclusive = _V23._write_manifest_exclusive


def execution_denial_receipt_v24() -> dict[str, Any]:
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
