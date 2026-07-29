#!/usr/bin/env python3
"""One-shot launcher adapter for V15 extended-horizon joint JEPA.

The reviewed V13 launcher supplies the unchanged custody, data loading,
evaluation, runtime, and write-once publication mechanisms.  It is loaded in
a private namespace and adapted for V15's frozen source selectors, doubled
in-memory schedule, longer accounting caps, and fresh update-1400 controls.
Import and the no-argument CLI remain source-only and denied by default.
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
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "execution_authorization_2026-07-29.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "source_manifest_2026-07-29.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "source_review_2026-07-29.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "clean_export_certification_2026-07-29.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_"
    "horizon_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v14_"
    "unified_ray_survival"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon"
)
EXPERIMENT_ARM_NAME = "unified_ray_survival_v15_extended_horizon"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

BASE_MAXIMUM_UPDATES = 1_000
BASE_MAXIMUM_PRESENTATIONS = 16_000
MAXIMUM_UPDATES = 2_000
MAXIMUM_PRESENTATIONS = 32_000
PRESENTATIONS_PER_UPDATE = 16
OBSERVATION_UPDATES = (0, 100, 400, 1_000, 1_400, 2_000)
CONTROL_OBSERVATION_UPDATES = (400, 1_400)
TERMINAL_UPDATES = (400, 1_400, 2_000)
SUCCESS_UPDATE = 2_000


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
    "MAXIMUM_UPDATES": BASE_MAXIMUM_UPDATES,
    "MAXIMUM_PRESENTATIONS": BASE_MAXIMUM_PRESENTATIONS,
    "OBSERVATION_UPDATES": (0, 100, 400, 1_000),
}
_V15_BASE_OVERRIDES = {
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
    "MAXIMUM_UPDATES": MAXIMUM_UPDATES,
    "MAXIMUM_PRESENTATIONS": MAXIMUM_PRESENTATIONS,
    "OBSERVATION_UPDATES": OBSERVATION_UPDATES,
}


def _load_private_base_launcher_v15() -> Any:
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
    module_name = "_lewm_v15_extended_horizon_private_base_launcher"
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
        for name, value in _V15_BASE_OVERRIDES.items():
            setattr(module, name, value)
        return module
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


_BASE_LAUNCHER = _load_private_base_launcher_v15()
_ORIGINAL_VALIDATE_SCHEDULE_V13 = _BASE_LAUNCHER._validate_schedule_v13
_ORIGINAL_V12_OBSERVATION_V13 = _BASE_LAUNCHER._v12_observation_v13


def _validate_schedule_v15(
    schedule: Sequence[int],
    *,
    executor_api: Any,
    labels_api: Any,
) -> tuple[int, ...]:
    """Validate the frozen 16k schedule, then repeat it once in memory."""

    if isinstance(schedule, (str, bytes)) or len(schedule) != BASE_MAXIMUM_PRESENTATIONS:
        raise PermissionError("V15 base schedule must contain exactly 16,000 presentations")
    base = tuple(schedule)
    if any(type(value) is not int or value < 0 for value in base):
        raise PermissionError("V15 schedule indices must be nonnegative exact integers")
    expected = dict(executor_api.CHECKPOINT_SCHEDULE_PREFIX_SHA256)
    observed = {
        100: labels_api.v4.canonical_json_sha256(list(base[:1_600])),
        400: labels_api.v4.canonical_json_sha256(list(base[:6_400])),
        1_000: labels_api.v4.canonical_json_sha256(list(base)),
    }
    if observed != expected:
        raise PermissionError("V15 frozen base schedule-prefix identity changed")
    extended = base + base
    if (
        len(extended) != MAXIMUM_PRESENTATIONS
        or extended[:BASE_MAXIMUM_PRESENTATIONS]
        != extended[BASE_MAXIMUM_PRESENTATIONS:]
    ):
        raise RuntimeError("V15 in-memory repeated schedule identity changed")
    return extended


def _v12_observation_v15(
    runtime: Any,
    model: Any,
    *,
    update: int,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, bool]] | None]:
    """Recompute controls from the current model at updates 400 and 1,400."""

    if update not in OBSERVATION_UPDATES:
        raise PermissionError("V15 inherited V12 observation update is not registered")
    control_selector = 400 if update in CONTROL_OBSERVATION_UPDATES else update
    return _ORIGINAL_V12_OBSERVATION_V13(
        runtime,
        model,
        update=control_selector,
    )


# The frozen composition and observation methods resolve these two hooks from
# their private module globals.  Physical provenance continues to receive the
# actual caller update; only the legacy Boolean-materialization selector is
# mapped from 1,400 to its existing control-enabled branch.
_BASE_LAUNCHER._validate_schedule_v13 = _validate_schedule_v15
_BASE_LAUNCHER._v12_observation_v13 = _v12_observation_v15


def _assert_configured_base_v15() -> None:
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V15_BASE_OVERRIDES
    }
    if observed != _V15_BASE_OVERRIDES:
        raise PermissionError("private V15 launcher adaptation changed after import")
    if (
        _BASE_LAUNCHER._validate_schedule_v13 is not _validate_schedule_v15
        or _BASE_LAUNCHER._v12_observation_v13 is not _v12_observation_v15
    ):
        raise PermissionError("private V15 launcher hooks changed after import")


def _load_authority_file_v15(path: Path) -> dict[str, Any]:
    _assert_configured_base_v15()
    return _BASE_LAUNCHER._load_authority_file_v13(path)


def execute_future_authorized_v15(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute through the privately configured custody runtime exactly once."""

    _assert_configured_base_v15()
    return _BASE_LAUNCHER.execute_future_authorized_v13(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v15() -> dict[str, Any]:
    return {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_private_launcher_adapter_v1",
        "base_launcher": BASE_LAUNCHER_RELATIVE_PATH,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "observation_updates": list(OBSERVATION_UPDATES),
        "control_observation_updates": list(CONTROL_OBSERVATION_UPDATES),
        "terminal_updates": list(TERMINAL_UPDATES),
        "schedule_base_presentations": BASE_MAXIMUM_PRESENTATIONS,
        "schedule_repeat_count": 2,
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "execution_authorized": False,
    }


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
    authority = _load_authority_file_v15(parsed.future_authority)
    result = execute_future_authorized_v15(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE2000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V15 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
