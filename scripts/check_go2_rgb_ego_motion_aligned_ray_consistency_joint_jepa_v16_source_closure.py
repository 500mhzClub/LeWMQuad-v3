#!/usr/bin/env python3
"""Build or verify the V16 ray-consistency recursive source closure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_PATH = (
    "scripts/check_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "source_closure.py"
)
MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "integrity_replacement_v1_source_manifest_2026-07-29.json"
)
SCHEMA = (
    "lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "integrity_replacement_v1_source_manifest"
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


_V13 = _source_only_module(
    "_lewm_v16_ray_consistency_source_closure_base",
    BASE_CHECKER_PATH,
)
if Path(_V13.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("V16 source-closure base escaped the repository")

ENTRYPOINTS = (
    "lewm/models/ego_motion_aligned_ray_consistency_v16.py",
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v14_unified_ray_survival.py"
    ),
    "scripts/run_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16.py",
    (
        "scripts/execute_go2_rgb_ego_motion_aligned_ray_consistency_joint_"
        "jepa_v16.py"
    ),
    (
        "scripts/launch_go2_rgb_ego_motion_aligned_ray_consistency_joint_"
        "jepa_v16.py"
    ),
)
FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys(
        (
            *_V13.FORCED_DYNAMIC_SOURCES,
            (
                "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
                "camera_evidence_bottleneck.py"
            ),
            (
                "scripts/go2_rgb_ego_motion_aligned_ray_consistency_joint_"
                "jepa_v16_lifecycle.py"
            ),
            (
                "scripts/go2_rgb_ego_motion_aligned_ray_consistency_joint_"
                "jepa_v16_recovery.py"
            ),
            "scripts/execute_go2_rgb_unified_ray_survival_joint_jepa_v14.py",
            (
                "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_"
                "v13_camera_evidence_bottleneck.py"
            ),
            (
                "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_"
                "v13_camera_evidence_bottleneck.py"
            ),
        )
    )
)
EXCLUDED_RUNTIME_CATEGORIES = _V13.EXCLUDED_RUNTIME_CATEGORIES
FORBIDDEN_PATH_PARTS = _V13.FORBIDDEN_PATH_PARTS
FORBIDDEN_RUNNER_PREFIX = _V13.FORBIDDEN_RUNNER_PREFIX


def _configure_private_base() -> None:
    for module in (_V13, _V13._BASE):
        module.MANIFEST_PATH = MANIFEST_PATH
        module.SCHEMA = SCHEMA
        module.ENTRYPOINTS = ENTRYPOINTS
        module.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
        module.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
    _V13._BASE.build_manifest = _V13.build_manifest


_configure_private_base()
build_manifest = _V13.build_manifest
discover_source_closure = _V13.discover_source_closure
verify_manifest = _V13.verify_manifest
_safe_source_path = _V13._safe_source_path
_read_regular_source = _V13._read_regular_source
_verify_tracked = _V13._verify_tracked
_write_manifest_exclusive = _V13._write_manifest_exclusive


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
    print("Go2 RGB V16 ego-motion-aligned ray-consistency source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
