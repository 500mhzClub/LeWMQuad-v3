#!/usr/bin/env python3
"""Build or verify the V13 recursive Python source closure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_PATH = "scripts/check_go2_multires_probe_source_closure_v3.py"
MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "source_manifest_2026-07-29.json"
)
SCHEMA = "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_source_manifest"


def _source_only_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only checker {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _source_only_module(
    "_lewm_v13_camera_evidence_source_closure_base",
    BASE_CHECKER_PATH,
)
if _BASE.ROOT.resolve() != ROOT.resolve():
    raise PermissionError("V13 source-closure base escaped the repository")

ENTRYPOINTS = (
    (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
        "v13_camera_evidence_bottleneck.py"
    ),
    (
        "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck.py"
    ),
    (
        "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck.py"
    ),
    (
        "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck.py"
    ),
)

# Exact local modules reached through importlib by the deferred V13 composer,
# V13 tensor core, and the reused V1/Shared-V5 loading and scoring surfaces.
FORCED_DYNAMIC_SOURCES = (
    "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v1.py",
    "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v1.py",
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py",
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py",
    "lewm/benchmarks/go2_swept_progress_survival_labels_v1.py",
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py",
    "lewm/benchmarks/go2_swept_progress_survival_joint_jepa_v1.py",
    "lewm/benchmarks/go2_post_action_projective_support_metrics_v1.py",
    "lewm/benchmarks/go2_post_action_projective_support_joint_jepa_v1.py",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py",
    "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v1.py",
    "lewm/models/observable_camera_ray_evidence_v4.py",
    "lewm/models/observable_camera_ray_evidence_v4_training.py",
    (
        "lewm/models/observable_camera_ray_evidence_v4_"
        "hierarchical_first_hit_v9.py"
    ),
    "lewm/models/shared_observable_camera_ray_jepa_v5.py",
)

EXCLUDED_RUNTIME_CATEGORIES = (
    ".generated artifacts, attempt registries, and runtime outputs",
    "tensor checkpoints, schedules, receipts, and metric sidecars",
    "raw RGB, scene shards, datasets, and role payloads",
    "configuration and custody roots",
    "sealed or held-out benchmark material",
    "review, authorization, result, and completion records",
)
FORBIDDEN_PATH_PARTS = frozenset(
    set(_BASE.FORBIDDEN_PATH_PARTS)
    | {
        "heldout",
        "held_out",
        "outputs",
        "runtime_artifacts",
        "runtime_inputs",
    }
)
FORBIDDEN_RUNNER_PREFIX = _BASE.FORBIDDEN_RUNNER_PREFIX

_base_safe_source_path = _BASE._safe_source_path


def _safe_source_path(relative: str) -> None:
    """Add an explicit held-out/runtime rejection to the reviewed walker."""

    _base_safe_source_path(relative)
    path = PurePosixPath(relative)
    parts = tuple(part.casefold() for part in path.parts)
    if (
        any(part in FORBIDDEN_PATH_PARTS for part in parts)
        or any(part.startswith("heldout_") for part in parts)
        or any(part.startswith("held_out_") for part in parts)
    ):
        raise PermissionError(f"forbidden V13 source-closure path: {relative}")


# Rebind the reviewed AST walker once; no project, Torch, data, or GPU module is
# imported by discovery.  A protected filename discovered by `rg --files`
# fails closed before its contents can be opened.
_BASE.MANIFEST_PATH = MANIFEST_PATH
_BASE.SCHEMA = SCHEMA
_BASE.ENTRYPOINTS = ENTRYPOINTS
_BASE.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
_BASE.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
_BASE.FORBIDDEN_PATH_PARTS = set(FORBIDDEN_PATH_PARTS)
_BASE._safe_source_path = _safe_source_path

_base_build_manifest = _BASE.build_manifest


def build_manifest() -> dict[str, object]:
    value = _base_build_manifest()
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    core.pop("consumed_adaptation_runner_source_count", None)
    core.pop("generated_input_open_count", None)
    core.update(
        {
            "date": "2026-07-29",
            "authority": (
                "source_closure_only_no_generated_or_runtime_input_checkpoint_"
                "training_gpu_qualification_g2_navigation_heldout_production_"
                "or_promotion_authority"
            ),
            "generated_or_runtime_artifact_open_count": 0,
            "dataset_or_rgb_open_count": 0,
        }
    )
    return {
        **core,
        "content_sha256": hashlib.sha256(_BASE._canonical_bytes(core)).hexdigest(),
    }


_BASE.build_manifest = build_manifest
discover_source_closure = _BASE.discover_source_closure
verify_manifest = _BASE.verify_manifest
_read_regular_source = _BASE._read_regular_source
_verify_tracked = _BASE._verify_tracked


def _write_manifest_exclusive(value: Mapping[str, object]) -> None:
    raw = json.dumps(
        dict(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        MANIFEST_PATH,
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
    print("Go2 RGB Camera-evidence-bottleneck V13 source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
