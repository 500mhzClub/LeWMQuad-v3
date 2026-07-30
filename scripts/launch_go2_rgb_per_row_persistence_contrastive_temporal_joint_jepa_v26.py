#!/usr/bin/env python3
"""Denied-by-default V26 launcher over the frozen V25 lifecycle.

V26 changes only the training module's schema-compatibility aliases.  This
launcher republishes the exact V25 preflight, batch builder, evaluator, gates,
and one-shot controller under fresh V26 custody identities.  It grants no
execution authority.
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
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25.py"
)
BASE_LAUNCHER_COMMIT = "43231c689547b66de83f3cafbfac270455a7a234"
BASE_LAUNCHER_FILE_SHA256 = (
    "dac097313656e7dd77b93fcf9433ece72c180b1edfbe6b9339ac4b4b8a1ceb0a"
)
BASE_LAUNCHER_BYTE_COUNT = 14_818
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v26_schema_compat_private_v25_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "0c277fd7350931a7993d5affc2d1d4633ffed916"
PREREGISTRATION_FILE_SHA256 = (
    "97061601af2922622673d7e4f8b4c1a6625edcdf899abd647373c28daa192a18"
)
PREREGISTRATION_BYTE_COUNT = 7_999
V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_terminal_failure_result_2026-07-30.json"
)
V25_TERMINAL_FAILURE_RESULT_COMMIT = (
    "26c8fd902319c06d4dbf25cab36a63ec2df44081"
)
V25_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "5c8d6d80ce24c60900c49f6cf49979c3001024666a2156d945e526b396dd1596"
)
V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 10_380
V25_TERMINAL_FAILURE_RESULT_CONTENT_SHA256 = (
    "59423f03ca153ca481d71ea4e88aaa625128ece4a15eb8b6253ae4f009272929"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-v26-per-row-persistence-contrastive-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26/"
    "attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v26"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26"
)
EXPERIMENT_ARM_NAME = "per_row_persistence_contrastive_temporal_v26"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

_V26_BASE_OVERRIDES = {
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


def _load_private_v25_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("frozen V25 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V25 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V25 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V25 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V25 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    inherited_names = (
        "_lewm_v25_per_row_temporal_private_v24_launcher",
        "_lewm_v24_core_protected_private_v23_launcher",
        "_lewm_v23_scene_action_private_v21_launcher",
        "_lewm_v21_scene_innovation_private_v20_launcher",
    )
    previous = {name: sys.modules.pop(name, None) for name in inherited_names}
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V25 launcher resolved another root")
        module._assert_configured_base_v25()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        for name, value in previous.items():
            if value is not None:
                sys.modules[name] = value
        raise


_V25_LAUNCHER = _load_private_v25_launcher()
_V24_LAUNCHER = _V25_LAUNCHER._V24_LAUNCHER
_V23_LAUNCHER = _V25_LAUNCHER._V23_LAUNCHER
_V21_LAUNCHER = _V25_LAUNCHER._V21_LAUNCHER
_V20_LAUNCHER = _V25_LAUNCHER._V20_LAUNCHER
_V18_LAUNCHER = _V25_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V25_LAUNCHER._BASE_LAUNCHER

for _name, _value in _V26_BASE_OVERRIDES.items():
    setattr(_V25_LAUNCHER, _name, _value)
    _V25_LAUNCHER._V25_BASE_OVERRIDES[_name] = _value
    setattr(_V24_LAUNCHER, _name, _value)
    _V24_LAUNCHER._V24_BASE_OVERRIDES[_name] = _value
    setattr(_V23_LAUNCHER, _name, _value)
    _V23_LAUNCHER._V23_BASE_OVERRIDES[_name] = _value
    setattr(_V21_LAUNCHER, _name, _value)
    _V21_LAUNCHER._V21_BASE_OVERRIDES[_name] = _value
    setattr(_V20_LAUNCHER, _name, _value)
    _V20_LAUNCHER._V19_BASE_OVERRIDES[_name] = _value
    setattr(_V18_LAUNCHER, _name, _value)
    _V18_LAUNCHER._V18_BASE_OVERRIDES[_name] = _value
    setattr(_BASE_LAUNCHER, _name, _value)

_V23_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23 = (
    "_v26_per_row_persistence_contrastive_schedule_preflight_receipt"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23 = (
    "_v26_per_row_persistence_contrastive_schedule_preflight_token"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_V23 = object()

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    _V25_LAUNCHER.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
ACTION_PRIOR_M_KEY_V23 = _V25_LAUNCHER.ACTION_PRIOR_M_KEY_V23
CONTROL_NAMES = _V25_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V25_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V25_LAUNCHER.COMPARISON_FIELDS
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V26 = (
    _V23_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23
)
_INHERITED_PREFLIGHT_V25 = (
    _V25_LAUNCHER.preflight_schedule_per_row_persistence_contrastive_v25
)
_INHERITED_BUILD_ONE_MICROBATCH_V25 = _V25_LAUNCHER._build_one_microbatch_v25


def preflight_schedule_per_row_persistence_contrastive_v26(
    runtime: Any,
) -> Mapping[str, Any]:
    receipt = _INHERITED_PREFLIGHT_V25(runtime)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema")
        != f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1"
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("V26 inherited schedule preflight identity changed")
    return receipt


def _build_one_microbatch_v26(
    *, runtime: Any, indices: Sequence[int], stage: str
) -> Mapping[str, Any]:
    result = _INHERITED_BUILD_ONE_MICROBATCH_V25(
        runtime=runtime, indices=indices, stage=stage
    )
    training = getattr(runtime, "training_module", None)
    required_v25 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V25", ()))
    required_v26 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V26", ()))
    if type(result) is not dict or tuple(result) != required_v25 or required_v26 != required_v25:
        raise RuntimeError("V26 inherited microbatch or zero-field extension changed")
    return result


def _assert_configured_base_v26() -> None:
    _V25_LAUNCHER._assert_configured_base_v25()
    for module in (
        _V25_LAUNCHER,
        _V24_LAUNCHER,
        _V23_LAUNCHER,
        _V21_LAUNCHER,
        _V20_LAUNCHER,
        _V18_LAUNCHER,
        _BASE_LAUNCHER,
    ):
        if {name: getattr(module, name, None) for name in _V26_BASE_OVERRIDES} != (
            _V26_BASE_OVERRIDES
        ):
            raise PermissionError("private V26 lifecycle selector changed after import")
    if (
        _V25_LAUNCHER._V25_BASE_OVERRIDES != _V26_BASE_OVERRIDES
        or _V24_LAUNCHER._V24_BASE_OVERRIDES != _V26_BASE_OVERRIDES
        or _V23_LAUNCHER._V23_BASE_OVERRIDES != _V26_BASE_OVERRIDES
        or _V21_LAUNCHER._V21_BASE_OVERRIDES != _V26_BASE_OVERRIDES
        or _V20_LAUNCHER._V19_BASE_OVERRIDES != _V26_BASE_OVERRIDES
        or _V18_LAUNCHER._V18_BASE_OVERRIDES != _V26_BASE_OVERRIDES
    ):
        raise PermissionError("private V26 launcher adaptation changed after import")


def _load_authority_file_v26(path: Path) -> dict[str, Any]:
    _assert_configured_base_v26()
    return _V25_LAUNCHER._load_authority_file_v25(path)


def execute_future_authorized_v26(
    *, repository_root: Path, authority: Mapping[str, Any]
) -> Mapping[str, Any]:
    _assert_configured_base_v26()
    return _V25_LAUNCHER.execute_future_authorized_v25(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v26() -> dict[str, Any]:
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
        "predecessor_terminal_failure": {
            "path": V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
            "commit": V25_TERMINAL_FAILURE_RESULT_COMMIT,
            "file_sha256": V25_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            "byte_count": V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
            "content_sha256": V25_TERMINAL_FAILURE_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "schedule_preflight_delegated_exactly_to_v25": True,
        "microbatch_builder_delegated_exactly_to_v25": True,
        "new_batch_fields_over_v25": 0,
        "extra_tensor_payload_reads": 0,
        "extra_predictor_forwards": 0,
        "update400_and_update1000_gates_inherited": True,
        "update400_recovery_write_owned_by_executor": True,
        "recovery_read_or_resume_implemented": False,
        "maximum_updates": _BASE_LAUNCHER.MAXIMUM_UPDATES,
        "maximum_presentations": _BASE_LAUNCHER.MAXIMUM_PRESENTATIONS,
        "one_shot_attempt_count": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
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
                    "recovery_state_opened": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v26(parsed.future_authority)
    result = execute_future_authorized_v26(repository_root=ROOT, authority=authority)
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V26 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
