#!/usr/bin/env python3
"""Denied-by-default launcher for V19 executed-successor grounding.

The frozen V18/V3 launcher is loaded in a private namespace and retains its
custody, data loading, evaluation, and write-once execution surface.  This
source-only adapter selects the V19 executor and training core while retaining
the exact V18 model.  Its sole runtime hook keeps the numeric paired-control
comparisons that the inherited evaluator already computes; it performs no
additional scoring or input access.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py"
)
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "691ed5d39f0b8d1b40071045dc181b9a4b215573"
PREREGISTRATION_FILE_SHA256 = (
    "9a1910e6c12ce27bf7951fe4bddbcfc80d19e1d0fc33d03359cc27d12dd1b79b"
)
PREREGISTRATION_BYTE_COUNT = 8_107
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
    "grounding_joint_jepa_v19_integrity_replacement_v1"
)
EXPERIMENT_ARM_NAME = (
    "object_space_height_volume_executed_successor_semantic_grounding_v19_"
    "integrity_replacement_v1"
)
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

CONTROL_NAMES = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "wrong_rgb",
    "train_action_mean_prior",
)
REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
COMPARISON_FIELDS = (
    "scene_count",
    "bootstrap_replicates",
    "bootstrap_seed",
    "equal_scene_mean_delta",
    "bootstrap_lower_95",
    "per_scene_delta",
    "family_deltas",
    "positive_family_count",
)


_PRISTINE_V18_DEFAULTS = {
    "AUTHORITY_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_execution_authorization_2026-07-30.json"
    ),
    "SOURCE_MANIFEST_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_source_manifest_2026-07-30.json"
    ),
    "SOURCE_REVIEW_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_source_review_2026-07-30.json"
    ),
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_clean_export_certification_2026-07-30.json"
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": (
        "scripts/check_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "source_closure.py"
    ),
    "EXECUTOR_MODULE_NAME": (
        "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
    ),
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": (
        "scripts.run_go2_rgb_object_space_height_volume_joint_jepa_v18"
    ),
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": (
        "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3"
    ),
    "EXPERIMENT_ARM_NAME": (
        "object_space_height_volume_v18_integrity_replacement_v3"
    ),
    "LAUNCHER_SCHEMA": (
        "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_launcher_v1"
    ),
}
_V19_BASE_OVERRIDES = {
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


def _load_private_v18_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("reviewed V18 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("reviewed V18 launcher escaped or is not regular")
    module_name = "_lewm_v19_semantic_grounding_private_v18_launcher"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed V18 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("reviewed V18 launcher resolved another root")
        observed = {
            name: getattr(module, name, None) for name in _PRISTINE_V18_DEFAULTS
        }
        if observed != _PRISTINE_V18_DEFAULTS:
            raise PermissionError("reviewed V18 launcher defaults changed")
        module._assert_configured_base_v18()
        for name, value in _V19_BASE_OVERRIDES.items():
            setattr(module, name, value)
            module._V18_BASE_OVERRIDES[name] = value
            setattr(module._BASE_LAUNCHER, name, value)
        return module
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


_V18_LAUNCHER = _load_private_v18_launcher()
_BASE_LAUNCHER = _V18_LAUNCHER._BASE_LAUNCHER
_ORIGINAL_V12_OBSERVATION_V13 = _BASE_LAUNCHER._v12_observation_v13


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sanitize_comparison_v19(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != COMPARISON_FIELDS:
        raise RuntimeError("V19 inherited paired-comparison schema changed")
    if (
        type(value["scene_count"]) is not int
        or value["scene_count"] != 8
        or type(value["bootstrap_replicates"]) is not int
        or value["bootstrap_replicates"] != 10_000
        or type(value["bootstrap_seed"]) is not int
        or value["bootstrap_seed"] != 20_260_728
        or type(value["positive_family_count"]) is not int
        or not 0 <= value["positive_family_count"] <= len(REGISTERED_FAMILIES)
    ):
        raise RuntimeError("V19 inherited paired-comparison counts changed")
    scene_deltas = value["per_scene_delta"]
    family_deltas = value["family_deltas"]
    if (
        not isinstance(scene_deltas, Mapping)
        or len(scene_deltas) != 8
        or any(not isinstance(name, str) or not name for name in scene_deltas)
        or not isinstance(family_deltas, Mapping)
        or tuple(family_deltas) != REGISTERED_FAMILIES
    ):
        raise RuntimeError("V19 inherited paired-comparison grouping changed")
    retained_family_deltas = {
        name: _finite_float(delta, name=f"family delta {name}")
        for name, delta in family_deltas.items()
    }
    if value["positive_family_count"] != sum(
        delta > 0.0 for delta in retained_family_deltas.values()
    ):
        raise RuntimeError("V19 inherited positive-family count changed")
    return {
        "scene_count": value["scene_count"],
        "bootstrap_replicates": value["bootstrap_replicates"],
        "bootstrap_seed": value["bootstrap_seed"],
        "equal_scene_mean_delta": _finite_float(
            value["equal_scene_mean_delta"], name="equal-scene mean delta"
        ),
        "bootstrap_lower_95": _finite_float(
            value["bootstrap_lower_95"], name="bootstrap lower 95"
        ),
        "per_scene_delta": {
            name: _finite_float(delta, name=f"scene delta {name}")
            for name, delta in scene_deltas.items()
        },
        "family_deltas": retained_family_deltas,
        "positive_family_count": value["positive_family_count"],
    }


def _v12_observation_v19(
    runtime: Any,
    model: Any,
    *,
    update: int,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, bool]] | None]:
    """Retain comparisons produced by the one unchanged inherited scoring pass."""

    if update not in _BASE_LAUNCHER.OBSERVATION_UPDATES:
        raise PermissionError("V19 inherited observation update is not registered")
    executor = runtime.v1_executor
    if tuple(executor.CONTROL_NAMES) != CONTROL_NAMES:
        raise RuntimeError("V19 inherited control order changed")
    if (
        tuple(executor.REGISTERED_FAMILIES) != REGISTERED_FAMILIES
        or tuple(runtime.executor_api.REGISTERED_FAMILIES)
        != REGISTERED_FAMILIES
    ):
        raise RuntimeError("V19 inherited family registry changed")
    stored = getattr(runtime, "causal_comparisons_v19", None)
    if stored is None:
        stored = {}
        runtime.causal_comparisons_v19 = stored
    if type(stored) is not dict or update in stored:
        raise RuntimeError("V19 comparison observation is not one-shot")

    original_comparison = executor.paired_control_comparison_v1
    captured: list[dict[str, Any]] = []

    def capture_once(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        result = original_comparison(*args, **kwargs)
        captured.append(_sanitize_comparison_v19(result))
        return result

    executor.paired_control_comparison_v1 = capture_once
    try:
        gate, controls = _ORIGINAL_V12_OBSERVATION_V13(
            runtime,
            model,
            update=update,
        )
    finally:
        executor.paired_control_comparison_v1 = original_comparison
    if len(captured) != len(CONTROL_NAMES):
        raise RuntimeError("V19 inherited observation did not compute four comparisons")
    stored[update] = {
        name: row for name, row in zip(CONTROL_NAMES, captured, strict=True)
    }
    return gate, controls


# V13's frozen runtime method resolves this helper in its private module
# globals.  The hook captures values while the sole inherited checkpoint-role
# scoring pass is already in progress and leaves its gate/Boolean return
# unchanged.
_BASE_LAUNCHER._v12_observation_v13 = _v12_observation_v19


def _assert_configured_base_v19() -> None:
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None)
        for name in _V19_BASE_OVERRIDES
    }
    outer = {
        name: getattr(_V18_LAUNCHER, name, None)
        for name in _V19_BASE_OVERRIDES
    }
    if observed != _V19_BASE_OVERRIDES or outer != _V19_BASE_OVERRIDES:
        raise PermissionError("private V19 launcher adaptation changed after import")
    if _V18_LAUNCHER._V18_BASE_OVERRIDES != _V19_BASE_OVERRIDES:
        raise PermissionError("private V19 launcher selector binding changed")
    if _BASE_LAUNCHER._v12_observation_v13 is not _v12_observation_v19:
        raise PermissionError("private V19 observation hook changed after import")


def _load_authority_file_v19(path: Path) -> dict[str, Any]:
    _assert_configured_base_v19()
    return _V18_LAUNCHER._load_authority_file_v18(path)


def execute_future_authorized_v19(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v19()
    return _V18_LAUNCHER.execute_future_authorized_v18(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v19() -> dict[str, Any]:
    return {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_private_launcher_adapter_v1",
        "base_launcher": BASE_LAUNCHER_RELATIVE_PATH,
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "file_sha256": PREREGISTRATION_FILE_SHA256,
            "byte_count": PREREGISTRATION_BYTE_COUNT,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "maximum_updates": _BASE_LAUNCHER.MAXIMUM_UPDATES,
        "maximum_presentations": _BASE_LAUNCHER.MAXIMUM_PRESENTATIONS,
        "numeric_comparisons_retained_without_rescoring": True,
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
    authority = _load_authority_file_v19(parsed.future_authority)
    result = execute_future_authorized_v19(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V19 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
