#!/usr/bin/env python3
"""Denied-by-default V24 launcher adapter over frozen V23.

V24 retains V23's exact schedule preflight, train-action prior, microbatch
construction, evaluator, gates, and one-shot lifecycle.  This adapter changes
only lifecycle identities and verifies that the batch schema gained no field.
It grants no execution authority.
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
    "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23.py"
)
BASE_LAUNCHER_COMMIT = "44938145362e5accdf8e12b906bfbaa970d62f25"
BASE_LAUNCHER_FILE_SHA256 = (
    "d189433cc2e10352f15deec5c81109e9cf4b52047ba90867a1e30aa33dd10fcb"
)
BASE_LAUNCHER_BYTE_COUNT = 20_923
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v24_core_protected_private_v23_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "475f1867149f5c5b764973bb5a371de83c29c3eb"
PREREGISTRATION_FILE_SHA256 = (
    "ad0514668b20fd3bb58a2c70e71bb153428f3a9b121c1f8b64ca6e08965c6933"
)
PREREGISTRATION_BYTE_COUNT = 12_137
V23_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_scientific_result_2026-07-30.json"
)
V23_SCIENTIFIC_RESULT_COMMIT = "04b0fa48c6c4e10868c2f302bc51100394e3907e"
V23_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "753c91babd4f7116444654167d2507ffb52d22f970fc926c05d287683954c994"
)
V23_SCIENTIFIC_RESULT_BYTE_COUNT = 20_640
V23_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "a5a6b8aa7312706d2ae3a5b53e39370462e9de6eda6b7a2ca4e2e0226a518ed8"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/LeWMQuad-v3-v24-core-protected-survival-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24/"
    "attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24"
)
EXPERIMENT_ARM_NAME = "predictor_core_protected_survival_output_v24"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

_V24_BASE_OVERRIDES = {
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


def _load_private_v23_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("frozen V23 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V23 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V23 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V23 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V23 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    inherited_private_name = "_lewm_v23_scene_action_private_v21_launcher"
    previous_inherited = sys.modules.pop(inherited_private_name, None)
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V23 launcher resolved another root")
        module._assert_configured_base_v23()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        if previous_inherited is not None:
            sys.modules[inherited_private_name] = previous_inherited
        raise


_V23_LAUNCHER = _load_private_v23_launcher()
_V21_LAUNCHER = _V23_LAUNCHER._V21_LAUNCHER
_V20_LAUNCHER = _V23_LAUNCHER._V20_LAUNCHER
_V18_LAUNCHER = _V23_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V23_LAUNCHER._BASE_LAUNCHER

# Retarget only lifecycle selectors.  V23's reviewed schedule preflight and
# microbatch builder remain the actual computation.
for _name, _value in _V24_BASE_OVERRIDES.items():
    setattr(_V23_LAUNCHER, _name, _value)
    _V23_LAUNCHER._V23_BASE_OVERRIDES[_name] = _value
    setattr(_V21_LAUNCHER, _name, _value)
    _V21_LAUNCHER._V21_BASE_OVERRIDES[_name] = _value
    setattr(_V20_LAUNCHER, _name, _value)
    _V20_LAUNCHER._V19_BASE_OVERRIDES[_name] = _value
    setattr(_V18_LAUNCHER, _name, _value)
    _V18_LAUNCHER._V18_BASE_OVERRIDES[_name] = _value
    setattr(_BASE_LAUNCHER, _name, _value)

# Rename the cached preflight identity while leaving its computation exact.
_V23_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23 = (
    "_v24_predictor_core_protected_survival_schedule_preflight_receipt"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23 = (
    "_v24_predictor_core_protected_survival_schedule_preflight_token"
)
_V23_LAUNCHER._SCHEDULE_PREFLIGHT_TOKEN_V23 = object()

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    _V23_LAUNCHER.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
ACTION_PRIOR_M_KEY_V23 = _V23_LAUNCHER.ACTION_PRIOR_M_KEY_V23
NON_HOLD_ACTION_INDICES_V23 = _V23_LAUNCHER.NON_HOLD_ACTION_INDICES_V23
CONTROL_NAMES = _V23_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V23_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V23_LAUNCHER.COMPARISON_FIELDS
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V24 = (
    _V23_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23
)
_INHERITED_PREFLIGHT_V23 = (
    _V23_LAUNCHER.preflight_schedule_state_residual_survival_v23
)
_INHERITED_BUILD_ONE_MICROBATCH_V23 = _V23_LAUNCHER._build_one_microbatch_v23


def preflight_schedule_predictor_core_protected_survival_v24(
    runtime: Any,
) -> Mapping[str, Any]:
    receipt = _INHERITED_PREFLIGHT_V23(runtime)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema")
        != f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1"
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("V24 inherited schedule preflight identity changed")
    return receipt


def _build_one_microbatch_v24(
    *, runtime: Any, indices: Sequence[int], stage: str
) -> Mapping[str, Any]:
    preflight_schedule_predictor_core_protected_survival_v24(runtime)
    result = _INHERITED_BUILD_ONE_MICROBATCH_V23(
        runtime=runtime, indices=indices, stage=stage
    )
    training = getattr(runtime, "training_module", None)
    required_v23 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V23", ()))
    required_v24 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V24", ()))
    if (
        type(result) is not dict
        or tuple(result) != required_v23
        or required_v24 != required_v23
        or getattr(training, "ACTION_PRIOR_M_KEY_V23", None)
        != ACTION_PRIOR_M_KEY_V23
    ):
        raise RuntimeError("V24 inherited microbatch or zero-field extension changed")
    return result


_V23_LAUNCHER._build_one_microbatch_v23 = _build_one_microbatch_v24
_V21_LAUNCHER._build_one_microbatch_v21 = _build_one_microbatch_v24
_BASE_LAUNCHER._build_one_microbatch_v13 = _build_one_microbatch_v24


def _assert_configured_base_v24() -> None:
    _V23_LAUNCHER._assert_configured_base_v23()
    for module in (
        _V23_LAUNCHER,
        _V21_LAUNCHER,
        _V20_LAUNCHER,
        _V18_LAUNCHER,
        _BASE_LAUNCHER,
    ):
        observed = {name: getattr(module, name, None) for name in _V24_BASE_OVERRIDES}
        if observed != _V24_BASE_OVERRIDES:
            raise PermissionError("private V24 lifecycle selector changed after import")
    if (
        _V23_LAUNCHER._V23_BASE_OVERRIDES != _V24_BASE_OVERRIDES
        or _V21_LAUNCHER._V21_BASE_OVERRIDES != _V24_BASE_OVERRIDES
        or _V20_LAUNCHER._V19_BASE_OVERRIDES != _V24_BASE_OVERRIDES
        or _V18_LAUNCHER._V18_BASE_OVERRIDES != _V24_BASE_OVERRIDES
        or _V23_LAUNCHER._build_one_microbatch_v23 is not _build_one_microbatch_v24
        or _V21_LAUNCHER._build_one_microbatch_v21 is not _build_one_microbatch_v24
        or _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v24
    ):
        raise PermissionError("private V24 launcher adaptation changed after import")


def _load_authority_file_v24(path: Path) -> dict[str, Any]:
    _assert_configured_base_v24()
    return _V23_LAUNCHER._load_authority_file_v23(path)


def execute_future_authorized_v24(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v24()
    return _V23_LAUNCHER.execute_future_authorized_v23(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v24() -> dict[str, Any]:
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
            "path": V23_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "commit": V23_SCIENTIFIC_RESULT_COMMIT,
            "file_sha256": V23_SCIENTIFIC_RESULT_FILE_SHA256,
            "byte_count": V23_SCIENTIFIC_RESULT_BYTE_COUNT,
            "content_sha256": V23_SCIENTIFIC_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scene_negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "action_prior_batch_key": ACTION_PRIOR_M_KEY_V23,
        "schedule_preflight_inherited_exactly_from_v23": True,
        "microbatch_builder_inherited_exactly_from_v23": True,
        "new_batch_fields_over_v23": 0,
        "extra_tensor_payload_reads": 0,
        "extra_predictor_forwards": 0,
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
    authority = _load_authority_file_v24(parsed.future_authority)
    result = execute_future_authorized_v24(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V24 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
