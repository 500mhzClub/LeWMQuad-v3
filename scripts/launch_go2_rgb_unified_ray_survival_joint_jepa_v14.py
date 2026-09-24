#!/usr/bin/env python3
"""One-shot launcher adapter for RGB unified-ray-survival joint JEPA V14.

The reviewed V13 launcher supplies the unchanged custody, data-loading,
evaluation, and write-once runtime.  This module loads that source into a
private module namespace and changes only explicit experiment selectors.  Its
import and no-argument CLI remain source-only and denied by default.
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
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "execution_authorization_2026-07-29.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "source_manifest_2026-07-29.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "source_review_2026-07-29.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "clean_export_certification_2026-07-29.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_unified_ray_survival_joint_jepa_v14_source_closure.py"
)
EXECUTOR_MODULE_NAME = "scripts.execute_go2_rgb_unified_ray_survival_joint_jepa_v14"
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v14_"
    "unified_ray_survival"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14"
)
EXPERIMENT_ARM_NAME = "unified_ray_survival_v14"
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
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13"
    ),
    "EXPERIMENT_ARM_NAME": "camera_evidence_bottleneck_v13",
    "LAUNCHER_SCHEMA": (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_launcher_v1"
    ),
}
_V14_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "SOURCE_MANIFEST_RELATIVE_PATH": SOURCE_MANIFEST_RELATIVE_PATH,
    "SOURCE_REVIEW_RELATIVE_PATH": SOURCE_REVIEW_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": (
        SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    ),
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SOURCE_EVIDENCE_SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}


def _load_private_base_launcher_v14() -> Any:
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
    module_name = "_lewm_v14_unified_ray_survival_private_base_launcher"
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
        for name, value in _V14_BASE_OVERRIDES.items():
            setattr(module, name, value)
        return module
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


_BASE_LAUNCHER = _load_private_base_launcher_v14()


def _assert_configured_base_v14() -> None:
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V14_BASE_OVERRIDES
    }
    if observed != _V14_BASE_OVERRIDES:
        raise PermissionError("private V14 launcher adaptation changed after import")


def _load_authority_file_v14(path: Path) -> dict[str, Any]:
    _assert_configured_base_v14()
    return _BASE_LAUNCHER._load_authority_file_v13(path)


def execute_future_authorized_v14(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute through the privately configured custody runtime exactly once."""

    _assert_configured_base_v14()
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
    authority = _load_authority_file_v14(parsed.future_authority)
    result = execute_future_authorized_v14(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V14 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
