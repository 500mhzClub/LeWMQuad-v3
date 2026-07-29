#!/usr/bin/env python3
"""Denied-by-default one-shot launcher for V16 ray consistency.

The reviewed V13 launcher retains custody, data loading, evaluation, and
write-once publication.  This adapter changes the experiment selectors and
adds the already-bound realized relative SE(2) row to each training batch.
Import and the no-argument CLI remain source-only and open no scientific data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "execution_authorization_2026-07-29.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "source_manifest_2026-07-29.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "source_review_2026-07-29.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "clean_export_certification_2026-07-29.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v14_"
    "unified_ray_survival"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16"
)
EXPERIMENT_ARM_NAME = "ego_motion_aligned_ray_consistency_v16"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"


_PRISTINE_BASE_DEFAULTS = {
    "AUTHORITY_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_execution_authorization_2026-07-29.json"
    ),
    "SOURCE_MANIFEST_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_source_manifest_2026-07-29.json"
    ),
    "SOURCE_REVIEW_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_source_review_2026-07-29.json"
    ),
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_clean_export_certification_2026-07-29.json"
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": (
        "scripts/check_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "source_closure.py"
    ),
    "EXECUTOR_MODULE_NAME": (
        "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    ),
    "MODEL_MODULE_NAME": (
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    ),
    "TRAINING_MODULE_NAME": (
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    ),
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13"
    ),
    "EXPERIMENT_ARM_NAME": "camera_evidence_bottleneck_v13",
    "LAUNCHER_SCHEMA": (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_launcher_v1"
    ),
}
_V16_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "SOURCE_MANIFEST_RELATIVE_PATH": SOURCE_MANIFEST_RELATIVE_PATH,
    "SOURCE_REVIEW_RELATIVE_PATH": SOURCE_REVIEW_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SOURCE_EVIDENCE_SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}


def _load_private_base_launcher_v16() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("reviewed V13 base launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("reviewed V13 base launcher escaped or is not regular")
    module_name = "_lewm_v16_ray_consistency_private_base_launcher"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed V13 base launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("reviewed V13 base launcher resolved another root")
        observed = {
            name: getattr(module, name, None) for name in _PRISTINE_BASE_DEFAULTS
        }
        if observed != _PRISTINE_BASE_DEFAULTS:
            raise PermissionError("reviewed V13 base launcher defaults changed")
        for name, value in _V16_BASE_OVERRIDES.items():
            setattr(module, name, value)
        return module
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


_BASE_LAUNCHER = _load_private_base_launcher_v16()


def _build_one_microbatch_v16(
    *,
    runtime: Any,
    indices: Sequence[int],
    stage: str,
) -> Mapping[str, Any]:
    """Compose the frozen V13 batch plus one already-bound SE(2) tensor."""

    base = runtime.v1_training.build_microbatch_v1(
        runtime.loader,
        runtime.pairs["train"],
        runtime.labels["train"],
        indices,
        runtime.device,
        stage=stage,
    )
    selected = [runtime.pairs["train"][int(index)] for index in indices]
    current = _BASE_LAUNCHER._stack_camera_rows_v13(
        runtime.raw_inputs,
        [str(pair["current_endpoint_sha256"]) for pair in selected],
        role="train",
        arm=EXPERIMENT_ARM_NAME,
        stage=stage,
        torch=runtime.torch,
    )
    next_ = _BASE_LAUNCHER._stack_camera_rows_v13(
        runtime.raw_inputs,
        [str(pair["next_endpoint_sha256"]) for pair in selected],
        role="train",
        arm=EXPERIMENT_ARM_NAME,
        stage=stage,
        torch=runtime.torch,
    )
    motion_rows = [pair.get("relative_se2_current_frame") for pair in selected]
    if any(
        not isinstance(row, list)
        or len(row) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row)
        for row in motion_rows
    ):
        raise PermissionError("V16 pair realized SE(2) schema changed")
    realized = runtime.torch.tensor(
        motion_rows,
        dtype=runtime.torch.float32,
        device=runtime.device,
    )
    if tuple(realized.shape) != (4, 3) or not bool(
        runtime.torch.isfinite(realized).all().item()
    ):
        raise PermissionError("V16 realized SE(2) batch is malformed or nonfinite")

    additions = {
        runtime.training_module.CURRENT_CAMERA_ORIGIN_KEY: current["camera_origin"],
        runtime.training_module.NEXT_CAMERA_ORIGIN_KEY: next_["camera_origin"],
        runtime.training_module.CURRENT_CAMERA_BASIS_KEY: current["camera_basis"],
        runtime.training_module.NEXT_CAMERA_BASIS_KEY: next_["camera_basis"],
        runtime.training_module.CURRENT_GROUND_PLANE_Z_KEY: current["ground"],
        runtime.training_module.NEXT_GROUND_PLANE_Z_KEY: next_["ground"],
        runtime.training_module.CURRENT_PIXEL_HIT_KEY: current["pixel_hit"],
        runtime.training_module.NEXT_PIXEL_HIT_KEY: next_["pixel_hit"],
        runtime.training_module.CURRENT_PIXEL_DISTANCE_KEY: current["pixel_distance"],
        runtime.training_module.NEXT_PIXEL_DISTANCE_KEY: next_["pixel_distance"],
        runtime.training_module.CURRENT_GROUND_IN_FRUSTUM_KEY: current[
            "ground_in_frustum"
        ],
        runtime.training_module.NEXT_GROUND_IN_FRUSTUM_KEY: next_[
            "ground_in_frustum"
        ],
        runtime.training_module.CURRENT_GROUND_CLEAR_KEY: current["ground_clear"],
        runtime.training_module.NEXT_GROUND_CLEAR_KEY: next_["ground_clear"],
    }
    additions = {
        name: value.to(device=runtime.device) for name, value in additions.items()
    }
    additions[runtime.training_module.REALIZED_RELATIVE_SE2_KEY] = realized
    if set(base) & set(additions):
        raise RuntimeError("V16 additions overlap the V1 base batch")
    result = {**base, **additions}
    if tuple(result) != tuple(runtime.training_module.REQUIRED_BATCH_KEYS):
        raise RuntimeError("V16 composed microbatch key order or membership changed")
    return result


_BASE_LAUNCHER._build_one_microbatch_v13 = _build_one_microbatch_v16


def _assert_configured_base_v16() -> None:
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V16_BASE_OVERRIDES
    }
    if observed != _V16_BASE_OVERRIDES:
        raise PermissionError("private V16 launcher adaptation changed after import")
    if _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v16:
        raise PermissionError("private V16 batch hook changed after import")


def _load_authority_file_v16(path: Path) -> dict[str, Any]:
    _assert_configured_base_v16()
    return _BASE_LAUNCHER._load_authority_file_v13(path)


def execute_future_authorized_v16(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v16()
    return _BASE_LAUNCHER.execute_future_authorized_v13(
        repository_root=repository_root,
        authority=authority,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-authority", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            json.dumps(
                {
                    "schema": LAUNCHER_SCHEMA,
                    "status": "DENIED_NO_FUTURE_AUTHORITY",
                    "scientific_payload_opened": False,
                    "reservation_created": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v16(parsed.future_authority)
    result = execute_future_authorized_v16(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V16 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
