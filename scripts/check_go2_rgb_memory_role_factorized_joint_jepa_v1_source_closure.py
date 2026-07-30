#!/usr/bin/env python3
"""Build or verify the memory-role factorized joint-JEPA V1 source closure.

The checker extends the reviewed V13 AST walker with only the exact dataset
adapter modules required by this candidate.  Discovery reads Python source
only; it does not import project, dataset, Torch, checkpoint, or runtime
modules and it never opens generated inputs.
"""
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
BASE_CHECKER_PATH = (
    "scripts/check_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "source_closure.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "972dd727f0d84f90cdd90e1c43b1faa46d763fd6"
)
BASE_CHECKER_FILE_SHA256 = (
    "7b3e89b690ca2d41c0bcd5d0a27ab0721a704948b6fd79cd1f3c0448214d6f4e"
)
BASE_CHECKER_BYTE_COUNT = 10_554

MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "source_manifest_2026-07-30.json"
)
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "clean_export_certification_2026-07-30.json"
)
EXECUTION_AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "execution_authorization_2026-07-30.json"
)
SCHEMA = (
    "lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_source_manifest"
)
PASS_STATUS_TEXT = (
    "Go2 RGB memory-role factorized V2 source closure: PASS"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "429cb57bd89348502cd5b695a25ae864d33fdfa7"
PREREGISTRATION_FILE_SHA256 = (
    "1fc6201b6137d57b5c97cf2b042b1f987476facdee28ce430b1aa0da3d0c2ba3"
)
PREREGISTRATION_BYTE_COUNT = 7_194
ORIGINAL_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "preregistration_2026-07-30.md"
)
ORIGINAL_PREREGISTRATION_COMMIT = (
    "01d78284a22a52816a41f31a78411491714b4f9c"
)
ORIGINAL_PREREGISTRATION_FILE_SHA256 = (
    "a9deae0b3335540b26791302566cdcb6a7d8397e96618b691dba1fa8db0c85c7"
)
ORIGINAL_PREREGISTRATION_BYTE_COUNT = 11_170
INTEGRITY_REPLACEMENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "integrity_replacement_v1_preregistration_2026-07-30.md"
)
INTEGRITY_REPLACEMENT_PREREGISTRATION_COMMIT = (
    "ba6e37d63f099cd51184642dea39808ae1f2f99e"
)
INTEGRITY_REPLACEMENT_PREREGISTRATION_FILE_SHA256 = (
    "a7c757f4a58b9a7d068ceb2e6676573843d58e72606b55713868ddfe86b97820"
)
INTEGRITY_REPLACEMENT_PREREGISTRATION_BYTE_COUNT = 7_211
TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "terminal_infrastructure_failure_result_2026-07-30.json"
)
TERMINAL_FAILURE_RESULT_COMMIT = (
    "291a7bcfaf95f24d5c84bd3d590afd54556d5b3d"
)
TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "80eaeb508a988b54e655df5b530fa3adab6a89bb13b6f5c45902ac851bc464f4"
)
TERMINAL_FAILURE_RESULT_BYTE_COUNT = 6_060
INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "integrity_replacement_v1_terminal_infrastructure_failure_result_"
    "2026-07-30.json"
)
INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_COMMIT = (
    "79c83b21e6447881cb43961eea404b28ec6ad87a"
)
INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "bedfafa247ee0c39697b16327eff96ed420204000f25f0255f5de26128f1c548"
)
INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 9_867
SPLIT_INTEGRITY_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "split_integrity_amendment_2026-07-30.md"
)
SPLIT_INTEGRITY_AMENDMENT_COMMIT = (
    "5a1535567bf00b8e47d67d8966ef42a52726bd5b"
)
SPLIT_INTEGRITY_AMENDMENT_FILE_SHA256 = (
    "8350289c0288f9f98d18b17f401318247bd4ecf8ae0597f14a6641606aa77c1f"
)
SPLIT_INTEGRITY_AMENDMENT_BYTE_COUNT = 3_136

MODEL_RELATIVE_PATH = "lewm/models/memory_role_factorized_joint_jepa_v1.py"
PLACE_DATASET_RELATIVE_PATH = (
    "lewm/datasets/go2_memory_role_place_triplets_v1.py"
)
PLACE_INDEX_BUILDER_RELATIVE_PATH = (
    "scripts/build_go2_memory_role_place_triplet_index_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_memory_role_factorized_joint_jepa_v1.py"
)
EVALUATOR_RELATIVE_PATH = (
    "scripts/evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_rgb_memory_role_factorized_joint_jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_memory_role_factorized_joint_jepa_v1.py"
)
IMPLEMENTATION_PATHS = (
    MODEL_RELATIVE_PATH,
    PLACE_DATASET_RELATIVE_PATH,
    PLACE_INDEX_BUILDER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    EVALUATOR_RELATIVE_PATH,
    EXECUTOR_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
)
ENTRYPOINTS = IMPLEMENTATION_PATHS

# These exact predecessor modules are loaded through read_bytes/exec or
# importlib.  They therefore cannot be discovered from ordinary import ASTs.
# Keep the complete frozen chains explicit so a source-only export can import
# the runner, executor, and denied-by-default launcher without the worktree.
RUNNER_PREDECESSOR_SOURCES = (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25.py",
    "scripts/run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py",
    "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23.py",
    "scripts/run_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21.py",
    "scripts/run_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v19.py",
    "scripts/run_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
    "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck.py",
)
EXECUTOR_PREDECESSOR_SOURCES = (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26.py",
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25.py",
    "scripts/execute_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py",
    "scripts/execute_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23.py",
    "scripts/execute_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21.py",
    "scripts/execute_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v19.py",
    "scripts/execute_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
    "scripts/execute_go2_rgb_unified_ray_survival_joint_jepa_v14.py",
    "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck.py",
)
LAUNCHER_PREDECESSOR_SOURCES = (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25.py",
    "scripts/launch_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py",
    "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23.py",
    "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21.py",
    "scripts/launch_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v19.py",
    "scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
    "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck.py",
)
H6_DATASET_DYNAMIC_SOURCE = (
    "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py"
)

# Dataset directories remain globally forbidden.  Only this finite source
# set can enter the source-only module index, including exact transitive
# package/import dependencies of the two reviewed adapters.
ALLOWED_DATASET_SOURCES = frozenset(
    {
        "lewm/datasets/__init__.py",
        PLACE_DATASET_RELATIVE_PATH,
        H6_DATASET_DYNAMIC_SOURCE,
        "lewm/datasets/go2_recurrent_h4_rgb_sequences.py",
        "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py",
    }
)

LIFECYCLE_PATHS = {
    "preregistration": PREREGISTRATION_RELATIVE_PATH,
    "original_preregistration": ORIGINAL_PREREGISTRATION_RELATIVE_PATH,
    "integrity_replacement_preregistration": (
        INTEGRITY_REPLACEMENT_PREREGISTRATION_RELATIVE_PATH
    ),
    "terminal_failure_result": TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
    "integrity_replacement_terminal_failure_result": (
        INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH
    ),
    "split_integrity_amendment": SPLIT_INTEGRITY_AMENDMENT_RELATIVE_PATH,
    "source_manifest": MANIFEST_RELATIVE_PATH,
    "source_review": SOURCE_REVIEW_RELATIVE_PATH,
    "clean_export_certification": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "execution_authority": EXECUTION_AUTHORITY_RELATIVE_PATH,
}
EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "memory-role factorized V2 execution remains denied until recursive "
    "closure, independent source review, narrow clean-export certification, "
    "and separate one-shot authority are complete and exact-bound"
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
        raise PermissionError("frozen V13 source-closure checker binding changed")
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
    "_lewm_memory_role_factorized_v1_source_closure_base", BASE_CHECKER_PATH
)
if Path(_V13.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
    raise PermissionError("memory-role source-closure base escaped the repository")

FORCED_DYNAMIC_SOURCES = tuple(
    dict.fromkeys(
        (
            *_V13.FORCED_DYNAMIC_SOURCES,
            *RUNNER_PREDECESSOR_SOURCES,
            *EXECUTOR_PREDECESSOR_SOURCES,
            *LAUNCHER_PREDECESSOR_SOURCES,
            H6_DATASET_DYNAMIC_SOURCE,
        )
    )
)
EXCLUDED_RUNTIME_CATEGORIES = tuple(
    dict.fromkeys(
        (
            *_V13.EXCLUDED_RUNTIME_CATEGORIES,
            "probability-calibration, G2, and protected-role material",
        )
    )
)
FORBIDDEN_PATH_PARTS = frozenset(
    set(_V13.FORBIDDEN_PATH_PARTS)
    | {
        "probability_calibration",
        "g2",
        "runtime",
        "runtime_artifacts",
        "runtime_inputs",
    }
)
FORBIDDEN_RUNNER_PREFIX = _V13.FORBIDDEN_RUNNER_PREFIX
_v13_safe_source_path = _V13._safe_source_path
_v13_module_index = _V13._module_index


def _safe_source_path(relative: str) -> None:
    """Reject protected/runtime paths while admitting exact adapter source."""

    path = PurePosixPath(relative)
    if relative in ALLOWED_DATASET_SOURCES:
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or path.suffix != ".py"
            or tuple(path.parts[:2]) != ("lewm", "datasets")
        ):
            raise PermissionError(f"unsafe allowed dataset source: {relative}")
        return

    _v13_safe_source_path(relative)
    folded_parts = tuple(part.casefold() for part in path.parts)
    file_name = path.name.casefold()
    if (
        any(part in FORBIDDEN_PATH_PARTS for part in folded_parts)
        or any(
            part.startswith(("sealed_", "heldout_", "held_out_"))
            for part in folded_parts
        )
        or "probability_calibration" in file_name
        or "calibration" in file_name
        or file_name.startswith("g2_")
        or "_g2_" in file_name
    ):
        raise PermissionError(
            f"forbidden memory-role source-closure path: {relative}"
        )


def _module_index() -> tuple[dict[str, Path], dict[Path, str]]:
    """Extend the ignore-honoring V13 index by five exact source files."""

    by_module, by_path = _v13_module_index()
    package_root = ROOT / "lewm"
    for relative_text in sorted(ALLOWED_DATASET_SOURCES):
        _safe_source_path(relative_text)
        path = ROOT / relative_text
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f"required dataset adapter source is absent: {relative_text}"
            ) from error
        if path.is_symlink() or not path.is_file() or resolved != path.absolute():
            raise PermissionError(
                f"dataset adapter source is not exact regular source: {relative_text}"
            )
        relative = path.relative_to(package_root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join(("lewm", *parts)) if parts else "lewm"
        existing = by_module.get(module_name)
        if existing is not None and existing != resolved:
            raise RuntimeError(
                f"duplicate local memory-role module {module_name}: "
                f"{existing}, {resolved}"
            )
        by_module[module_name] = resolved
        by_path[resolved] = module_name
    return by_module, by_path


_BASE = _V13._BASE
_BASE.MANIFEST_PATH = MANIFEST_PATH
_BASE.SCHEMA = SCHEMA
_BASE.ENTRYPOINTS = ENTRYPOINTS
_BASE.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
_BASE.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES
_BASE.FORBIDDEN_PATH_PARTS = set(FORBIDDEN_PATH_PARTS)
_BASE._safe_source_path = _safe_source_path
_BASE._module_index = _module_index
_V13.MANIFEST_PATH = MANIFEST_PATH
_V13.SCHEMA = SCHEMA
_V13.ENTRYPOINTS = ENTRYPOINTS
_V13.FORCED_DYNAMIC_SOURCES = FORCED_DYNAMIC_SOURCES
_V13.EXCLUDED_RUNTIME_CATEGORIES = EXCLUDED_RUNTIME_CATEGORIES

_v13_build_manifest = _V13.build_manifest


def build_manifest() -> dict[str, object]:
    value = _v13_build_manifest()
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    core.update(
        {
            "date": "2026-07-30",
            "authority": (
                "source_closure_only_no_generated_or_runtime_input_checkpoint_"
                "dataset_rgb_training_gpu_probability_calibration_g2_"
                "navigation_heldout_production_or_promotion_authority"
            ),
            "preregistration": {
                "path": PREREGISTRATION_RELATIVE_PATH,
                "commit": PREREGISTRATION_COMMIT,
                "file_sha256": PREREGISTRATION_FILE_SHA256,
                "byte_count": PREREGISTRATION_BYTE_COUNT,
            },
            "original_preregistration": {
                "path": ORIGINAL_PREREGISTRATION_RELATIVE_PATH,
                "commit": ORIGINAL_PREREGISTRATION_COMMIT,
                "file_sha256": ORIGINAL_PREREGISTRATION_FILE_SHA256,
                "byte_count": ORIGINAL_PREREGISTRATION_BYTE_COUNT,
            },
            "integrity_replacement_preregistration": {
                "path": INTEGRITY_REPLACEMENT_PREREGISTRATION_RELATIVE_PATH,
                "commit": INTEGRITY_REPLACEMENT_PREREGISTRATION_COMMIT,
                "file_sha256": (
                    INTEGRITY_REPLACEMENT_PREREGISTRATION_FILE_SHA256
                ),
                "byte_count": (
                    INTEGRITY_REPLACEMENT_PREREGISTRATION_BYTE_COUNT
                ),
            },
            "terminal_failure_result": {
                "path": TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
                "commit": TERMINAL_FAILURE_RESULT_COMMIT,
                "file_sha256": TERMINAL_FAILURE_RESULT_FILE_SHA256,
                "byte_count": TERMINAL_FAILURE_RESULT_BYTE_COUNT,
            },
            "integrity_replacement_terminal_failure_result": {
                "path": (
                    INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH
                ),
                "commit": (
                    INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_COMMIT
                ),
                "file_sha256": (
                    INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_FILE_SHA256
                ),
                "byte_count": (
                    INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_BYTE_COUNT
                ),
            },
            "split_integrity_amendment": {
                "path": SPLIT_INTEGRITY_AMENDMENT_RELATIVE_PATH,
                "commit": SPLIT_INTEGRITY_AMENDMENT_COMMIT,
                "file_sha256": SPLIT_INTEGRITY_AMENDMENT_FILE_SHA256,
                "byte_count": SPLIT_INTEGRITY_AMENDMENT_BYTE_COUNT,
            },
            "execution_authorized": False,
            "generated_or_runtime_artifact_open_count": 0,
            "dataset_payload_or_rgb_open_count": 0,
            "probability_calibration_open_count": 0,
            "g2_or_heldout_open_count": 0,
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


def execution_denial_receipt_v1() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_execution_denial_v1",
        "status": "DENIED_INCOMPLETE_SOURCE_LIFECYCLE",
        "source_manifest_path": MANIFEST_RELATIVE_PATH,
        "source_review_path": SOURCE_REVIEW_RELATIVE_PATH,
        "clean_export_certification_path": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
        "execution_authority_path": EXECUTION_AUTHORITY_RELATIVE_PATH,
        "generated_or_runtime_artifact_opened": False,
        "dataset_payload_or_rgb_opened": False,
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
