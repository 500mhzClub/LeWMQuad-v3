#!/usr/bin/env python3
"""Denied-by-default V25 launcher adapter over frozen V24.

V25 keeps V24's exact schedule preflight, batch construction, evaluator,
thresholds, and one-shot launch custody.  Only lifecycle identities change;
the V25 executor owns the preregistered write-only update-400 recovery step.
This source shell grants no execution authority.
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
    "scripts/launch_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24.py"
)
BASE_LAUNCHER_COMMIT = "2b6178a4d876dc17c45fb340a4ab03ee302649b0"
BASE_LAUNCHER_FILE_SHA256 = (
    "40c43ecdbaf9dc41a7b09ca2f42730538e2645033d5c0a24809d4bec82a80c5b"
)
BASE_LAUNCHER_BYTE_COUNT = 14_684
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v25_per_row_temporal_private_v24_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "f00e20df3b429f9242516ac38f67fea587e04b22"
PREREGISTRATION_FILE_SHA256 = (
    "b9ce16b251415c50cb643daad919699c32965e23ddcd77d22bb3b69334f8b299"
)
PREREGISTRATION_BYTE_COUNT = 18_965
V24_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_scientific_result_2026-07-30.json"
)
V24_SCIENTIFIC_RESULT_COMMIT = "2824c80c54fc7502b1413b3371fc87c9206f82a2"
V24_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "f901d49eb9db0c39a068e67496b0b1cdaec954c9238edb40648140b924894e48"
)
V24_SCIENTIFIC_RESULT_BYTE_COUNT = 22_361
V24_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "0349f41da529b0c8658bf14ae51d85892a6f21fb461a281a9e157c7e7ff571dc"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-v25-per-row-persistence-contrastive-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25/"
    "attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25"
)
EXPERIMENT_ARM_NAME = "per_row_persistence_contrastive_temporal_v25"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

_V25_BASE_OVERRIDES = {
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


def _load_private_v24_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("frozen V24 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V24 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V24 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V24 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V24 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    inherited_names = (
        "_lewm_v24_core_protected_private_v23_launcher",
        "_lewm_v23_scene_action_private_v21_launcher",
        "_lewm_v21_scene_innovation_private_v20_launcher",
    )
    previous = {name: sys.modules.pop(name, None) for name in inherited_names}
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V24 launcher resolved another root")
        module._assert_configured_base_v24()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        for name, value in previous.items():
            if value is not None:
                sys.modules[name] = value
        raise


_V24_LAUNCHER = _load_private_v24_launcher()
_V23_LAUNCHER = _V24_LAUNCHER._V23_LAUNCHER
_V21_LAUNCHER = _V24_LAUNCHER._V21_LAUNCHER
_V20_LAUNCHER = _V24_LAUNCHER._V20_LAUNCHER
_V18_LAUNCHER = _V24_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V24_LAUNCHER._BASE_LAUNCHER

for _name, _value in _V25_BASE_OVERRIDES.items():
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
    "_v25_per_row_persistence_contrastive_schedule_preflight_receipt"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23 = (
    "_v25_per_row_persistence_contrastive_schedule_preflight_token"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_V23 = object()

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    _V24_LAUNCHER.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
ACTION_PRIOR_M_KEY_V23 = _V24_LAUNCHER.ACTION_PRIOR_M_KEY_V23
CONTROL_NAMES = _V24_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V24_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V24_LAUNCHER.COMPARISON_FIELDS
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V25 = (
    _V23_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23
)
_INHERITED_PREFLIGHT_V24 = (
    _V24_LAUNCHER.preflight_schedule_predictor_core_protected_survival_v24
)
_INHERITED_BUILD_ONE_MICROBATCH_V24 = _V24_LAUNCHER._build_one_microbatch_v24


def preflight_schedule_per_row_persistence_contrastive_v25(
    runtime: Any,
) -> Mapping[str, Any]:
    receipt = _INHERITED_PREFLIGHT_V24(runtime)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema")
        != f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1"
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("V25 inherited schedule preflight identity changed")
    return receipt


def _build_one_microbatch_v25(
    *, runtime: Any, indices: Sequence[int], stage: str
) -> Mapping[str, Any]:
    preflight_schedule_per_row_persistence_contrastive_v25(runtime)
    result = _INHERITED_BUILD_ONE_MICROBATCH_V24(
        runtime=runtime, indices=indices, stage=stage
    )
    training = getattr(runtime, "training_module", None)
    required_v24 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V24", ()))
    required_v25 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V25", ()))
    if (
        type(result) is not dict
        or tuple(result) != required_v24
        or required_v25 != required_v24
        or getattr(training, "ACTION_PRIOR_M_KEY_V23", None)
        != ACTION_PRIOR_M_KEY_V23
    ):
        raise RuntimeError("V25 inherited microbatch or zero-field extension changed")
    return result


_V24_LAUNCHER._build_one_microbatch_v24 = _build_one_microbatch_v25
_V23_LAUNCHER._build_one_microbatch_v23 = _build_one_microbatch_v25
_V21_LAUNCHER._build_one_microbatch_v21 = _build_one_microbatch_v25
_BASE_LAUNCHER._build_one_microbatch_v13 = _build_one_microbatch_v25


def _assert_configured_base_v25() -> None:
    _V24_LAUNCHER._assert_configured_base_v24()
    for module in (
        _V24_LAUNCHER,
        _V23_LAUNCHER,
        _V21_LAUNCHER,
        _V20_LAUNCHER,
        _V18_LAUNCHER,
        _BASE_LAUNCHER,
    ):
        if {name: getattr(module, name, None) for name in _V25_BASE_OVERRIDES} != (
            _V25_BASE_OVERRIDES
        ):
            raise PermissionError("private V25 lifecycle selector changed after import")
    if (
        _V24_LAUNCHER._V24_BASE_OVERRIDES != _V25_BASE_OVERRIDES
        or _V23_LAUNCHER._V23_BASE_OVERRIDES != _V25_BASE_OVERRIDES
        or _V21_LAUNCHER._V21_BASE_OVERRIDES != _V25_BASE_OVERRIDES
        or _V20_LAUNCHER._V19_BASE_OVERRIDES != _V25_BASE_OVERRIDES
        or _V18_LAUNCHER._V18_BASE_OVERRIDES != _V25_BASE_OVERRIDES
        or _V24_LAUNCHER._build_one_microbatch_v24 is not _build_one_microbatch_v25
        or _V23_LAUNCHER._build_one_microbatch_v23 is not _build_one_microbatch_v25
        or _V21_LAUNCHER._build_one_microbatch_v21 is not _build_one_microbatch_v25
        or _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v25
    ):
        raise PermissionError("private V25 launcher adaptation changed after import")


def _load_authority_file_v25(path: Path) -> dict[str, Any]:
    _assert_configured_base_v25()
    return _V24_LAUNCHER._load_authority_file_v24(path)


def execute_future_authorized_v25(
    *, repository_root: Path, authority: Mapping[str, Any]
) -> Mapping[str, Any]:
    _assert_configured_base_v25()
    return _V24_LAUNCHER.execute_future_authorized_v24(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v25() -> dict[str, Any]:
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
            "path": V24_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "commit": V24_SCIENTIFIC_RESULT_COMMIT,
            "file_sha256": V24_SCIENTIFIC_RESULT_FILE_SHA256,
            "byte_count": V24_SCIENTIFIC_RESULT_BYTE_COUNT,
            "content_sha256": V24_SCIENTIFIC_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "schedule_preflight_inherited_exactly_from_v24": True,
        "microbatch_builder_inherited_exactly_from_v24": True,
        "new_batch_fields_over_v24": 0,
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
                    "recovery_state_opened": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v25(parsed.future_authority)
    result = execute_future_authorized_v25(repository_root=ROOT, authority=authority)
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V25 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
