#!/usr/bin/env python3
"""Denied-by-default V22 launcher over the frozen V21 launcher.

V22 inherits V21's full-schedule different-scene metadata preflight and exact
one-field microbatch adapter.  The action axis is derived inside training from
the existing all-action prediction tensor, so this launcher adds no data field
or tensor construction.  This source shell grants no execution authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21.py"
)
BASE_LAUNCHER_COMMIT = "7071a006dda3851280fbdf030e156862c4f19ab3"
BASE_LAUNCHER_FILE_SHA256 = (
    "4cb6fb3302919d6090e3ef456068ba209890237d65c87c8515a723628ee5b486"
)
BASE_LAUNCHER_BYTE_COUNT = 22_650
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v22_scene_action_private_v21_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "43053ae49c28082c616f45ed857eedb727380952"
PREREGISTRATION_FILE_SHA256 = (
    "7ee36433d739663654de593cf018500cc5547e249173f08201ad4ac5c6b1959e"
)
PREREGISTRATION_BYTE_COUNT = 11_986
V21_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_scientific_result_2026-07-30.json"
)
V21_SCIENTIFIC_RESULT_COMMIT = "e5b5e56b30cee0c1eb818d52c4d886909f570f4d"
V21_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "c9544055b11d162b5b5fc9b02d0a04f3961a61b4547411964812a9ae4c5da1e7"
)
V21_SCIENTIFIC_RESULT_BYTE_COUNT = 15_724
V21_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "2195025bf24e3de621e76a5a5e3ea272ced05bd9f6e4fb91302035137ab7b9ec"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/LeWMQuad-v3-v22-scene-action-innovation-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22/attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
EXPERIMENT_ARM_NAME = "scene_action_contrastive_innovation_v22"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

_V22_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "SOURCE_MANIFEST_RELATIVE_PATH": SOURCE_MANIFEST_RELATIVE_PATH,
    "SOURCE_REVIEW_RELATIVE_PATH": SOURCE_REVIEW_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SOURCE_EVIDENCE_SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}


def _load_private_v21_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("frozen V21 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V21 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V21 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V21 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V21 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    # V21 deliberately retains its privately loaded V20 launcher.  A public
    # V21 source-only test may therefore have occupied that fixed private name
    # earlier in the same interpreter.  The old V21 module keeps a direct
    # object reference, so release only the registry name before executing an
    # independently bound V21 instance for V22.
    inherited_private_name = "_lewm_v21_scene_innovation_private_v20_launcher"
    previous_inherited = sys.modules.pop(inherited_private_name, None)
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V21 launcher resolved another root")
        module._assert_configured_base_v21()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        if previous_inherited is not None:
            sys.modules[inherited_private_name] = previous_inherited
        raise


_V21_LAUNCHER = _load_private_v21_launcher()
_V20_LAUNCHER = _V21_LAUNCHER._V20_LAUNCHER
_V18_LAUNCHER = _V21_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V21_LAUNCHER._BASE_LAUNCHER

# Retarget only lifecycle selectors.  V21's reviewed metadata preflight and
# microbatch hook remain the exact installed callables.
for _name, _value in _V22_BASE_OVERRIDES.items():
    setattr(_V21_LAUNCHER, _name, _value)
    _V21_LAUNCHER._V21_BASE_OVERRIDES[_name] = _value
    setattr(_V20_LAUNCHER, _name, _value)
    _V20_LAUNCHER._V19_BASE_OVERRIDES[_name] = _value
    setattr(_V18_LAUNCHER, _name, _value)
    _V18_LAUNCHER._V18_BASE_OVERRIDES[_name] = _value
    setattr(_BASE_LAUNCHER, _name, _value)

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    _V21_LAUNCHER.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21 = (
    _V21_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21
)
first_cyclic_different_scene_rows_v21 = (
    _V21_LAUNCHER.first_cyclic_different_scene_rows_v21
)
negative_rows_from_train_metadata_v21 = (
    _V21_LAUNCHER.negative_rows_from_train_metadata_v21
)
preflight_schedule_negative_rows_v21 = (
    _V21_LAUNCHER.preflight_schedule_negative_rows_v21
)
_build_one_microbatch_v21 = _V21_LAUNCHER._build_one_microbatch_v21
CONTROL_NAMES = _V21_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V21_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V21_LAUNCHER.COMPARISON_FIELDS


def _assert_configured_base_v22() -> None:
    _V21_LAUNCHER._assert_configured_base_v21()
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V22_BASE_OVERRIDES
    }
    outer = {
        name: getattr(_V18_LAUNCHER, name, None) for name in _V22_BASE_OVERRIDES
    }
    if observed != _V22_BASE_OVERRIDES or outer != _V22_BASE_OVERRIDES:
        raise PermissionError("private V22 launcher adaptation changed after import")
    if _V21_LAUNCHER._V21_BASE_OVERRIDES != _V22_BASE_OVERRIDES:
        raise PermissionError("private V22 V21-selector binding changed")
    if _V20_LAUNCHER._V19_BASE_OVERRIDES != _V22_BASE_OVERRIDES:
        raise PermissionError("private V22 V20-selector binding changed")
    if _V18_LAUNCHER._V18_BASE_OVERRIDES != _V22_BASE_OVERRIDES:
        raise PermissionError("private V22 V18-selector binding changed")
    if _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v21:
        raise PermissionError("private V22 inherited microbatch hook changed")


def _load_authority_file_v22(path: Path) -> dict[str, Any]:
    _assert_configured_base_v22()
    return _V21_LAUNCHER._load_authority_file_v21(path)


def execute_future_authorized_v22(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v22()
    return _V21_LAUNCHER.execute_future_authorized_v21(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v22() -> dict[str, Any]:
    return {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_private_launcher_adapter_v1",
        "base_launcher": {
            "path": BASE_LAUNCHER_RELATIVE_PATH,
            "commit": BASE_LAUNCHER_COMMIT,
            "file_sha256": BASE_LAUNCHER_FILE_SHA256,
            "byte_count": BASE_LAUNCHER_BYTE_COUNT,
        },
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "file_sha256": PREREGISTRATION_FILE_SHA256,
            "byte_count": PREREGISTRATION_BYTE_COUNT,
        },
        "predecessor_scientific_result": {
            "path": V21_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "commit": V21_SCIENTIFIC_RESULT_COMMIT,
            "file_sha256": V21_SCIENTIFIC_RESULT_FILE_SHA256,
            "byte_count": V21_SCIENTIFIC_RESULT_BYTE_COUNT,
            "content_sha256": V21_SCIENTIFIC_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scene_negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "scene_negative_preflight_inherited_exactly_from_v21": True,
        "new_batch_fields_over_v21": 0,
        "action_negatives_derived_from_existing_all_action_prediction": True,
        "numeric_comparisons_retained_without_rescoring": True,
        "update100_new_terminal_branch": False,
        "update400_and_update1000_gates_inherited": True,
        "maximum_updates": _BASE_LAUNCHER.MAXIMUM_UPDATES,
        "maximum_presentations": _BASE_LAUNCHER.MAXIMUM_PRESENTATIONS,
        "one_shot_attempt_count": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "public_base_was_loaded_before_adapter": (
            _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_base_loaded_by_adapter": BASE_PUBLIC_MODULE_NAME in sys.modules,
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
    authority = _load_authority_file_v22(parsed.future_authority)
    result = execute_future_authorized_v22(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V22 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
