#!/usr/bin/env python3
"""Build or verify the V22 scene-action innovation source closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_PATH = (
    "scripts/check_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_source_closure.py"
)
MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_source_manifest_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_"
    "source_manifest"
)
PASS_STATUS_TEXT = "Go2 RGB V22 scene-action innovation source closure: PASS"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_"
    "preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "43053ae49c28082c616f45ed857eedb727380952"
PREREGISTRATION_FILE_SHA256 = (
    "7ee36433d739663654de593cf018500cc5547e249173f08201ad4ac5c6b1959e"
)
PREREGISTRATION_BYTE_COUNT = 11_986


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


_V21 = _source_only_module(
    "_lewm_v22_scene_action_innovation_source_closure_base",
    BASE_CHECKER_PATH,
)
if Path(_V21.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V22 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v18_object_space_height_volume.py"
    ),
    (
        "scripts/run_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
        "v22.py"
    ),
    (
        "scripts/execute_go2_rgb_scene_action_contrastive_innovation_joint_"
        "jepa_v22.py"
    ),
    (
        "scripts/launch_go2_rgb_scene_action_contrastive_innovation_joint_"
        "jepa_v22.py"
    ),
)
V21_PARENT_ENTRYPOINTS = tuple(_V21.ENTRYPOINTS[1:])
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys((*_V21.FORCED_DYNAMIC_SOURCES, *V21_PARENT_ENTRYPOINTS))
)
EXCLUDED_RUNTIME_CATEGORIES = _V21.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V21.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V21.FORBIDDEN_RUNNER_PREFIX


def _configure_private_base() -> None:
    modules = (
        _V21,
        _V21._V20,
        _V21._V20._V18,
        _V21._V20._V18._V13,
        _V21._V20._V18._V13._BASE,
    )
    for module in modules:
        module.MANIFEST_PATH = MANIFEST_PATH
        module.SCHEMA = SCHEMA
        module.ENTRYPOINTS = ENTRYPOINTS
        module.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
        module.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V21._V20._V18._V13._BASE.build_manifest = (
        _V21._V20._V18._V13.build_manifest
    )


_configure_private_base()
build_manifest = _V21.build_manifest
discover_source_closure = _V21.discover_source_closure
verify_manifest = _V21.verify_manifest
_safe_source_path = _V21._safe_source_path
_read_regular_source = _V21._read_regular_source
_verify_tracked = _V21._verify_tracked
_write_manifest_exclusive = _V21._write_manifest_exclusive


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
