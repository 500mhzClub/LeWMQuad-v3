#!/usr/bin/env python3
"""Denied-by-default executor adapter for V18 object-space height volume."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V14_EXECUTOR_PATH = (
    ROOT / "scripts/execute_go2_rgb_unified_ray_survival_joint_jepa_v14.py"
)
V14_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_unified_ray_survival_joint_jepa_v14"
)
PRIVATE_V14_MODULE_NAME = f"{__name__}.__private_v14_executor"
_PUBLIC_V14_WAS_LOADED_BEFORE_ADAPTER = V14_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
    "integrity_replacement_v3"
)
PREREGISTRATION_COMMIT = "81d1557cce55a448a84e00b5e822923b590e6f7d"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v3_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "4bc9b7baf42e0af3041ff2c766f6fc26698706e16964b8571cc40b28af70242e"
)
PREREGISTRATION_BYTE_COUNT = 6_527
REPLACEMENT_V2_PREREGISTRATION_COMMIT = (
    "baad8efaf524bb3f88f2d4516db7ef368f15684e"
)
REPLACEMENT_V2_PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v2_preregistration_2026-07-30.md"
)
REPLACEMENT_V2_PREREGISTRATION_FILE_SHA256 = (
    "71d3530df714719381df45d08fc8818d7def712c7c8647ac9db80690c0bf5167"
)
REPLACEMENT_V2_PREREGISTRATION_BYTE_COUNT = 6_507
REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_COMMIT = (
    "432b356f545a539f2cfcbb2b3d50895a03af1c73"
)
REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v2_terminal_failure_result_2026-07-30.json"
)
REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "350e3e3c73e0c5ed5d9b7c8d9045661aba1d471c58439c576f15a63ee9b6e114"
)
REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 7_072
REPLACEMENT_V1_PREREGISTRATION_COMMIT = (
    "402f61522d59943e0def9df0b90ebf785867d366"
)
REPLACEMENT_V1_PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v1_preregistration_2026-07-30.md"
)
REPLACEMENT_V1_PREREGISTRATION_FILE_SHA256 = (
    "0b50421844b09544c0de259ae8cf9386baf49f02ec50166ded72d9f8f5497daf"
)
REPLACEMENT_V1_PREREGISTRATION_BYTE_COUNT = 5_800
REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_COMMIT = (
    "8acbb240a59c22d65ab5082a81596dcc24de86ee"
)
REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v1_terminal_failure_result_2026-07-30.json"
)
REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "4b6246a94bd5ce8265807d255620deb46fb32406dfc1c92c396e87aa42bc8dc8"
)
REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 7_828
ORIGINAL_V18_PREREGISTRATION_COMMIT = (
    "5522b226e845907b091ff98ebac3b6f6315a4ca7"
)
ORIGINAL_V18_PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
    "preregistration_2026-07-30.md"
)
ORIGINAL_V18_PREREGISTRATION_FILE_SHA256 = (
    "c9997c6f335d6a7788e9cf8badb971ad2929d67094903d6e3676ec873bb8cae5"
)
ORIGINAL_V18_PREREGISTRATION_BYTE_COUNT = 9_718
V18_TERMINAL_FAILURE_RESULT_COMMIT = (
    "0c0b0804730028bdd5dadf4e5180685b4bc33e0e"
)
V18_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
    "terminal_failure_result_2026-07-30.json"
)
V18_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "a04b90e31298d5aa0a0764478ce4794d21f5120b981c6f16edac5fda103ee66f"
)
V18_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 5_828
V14_RESULT_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "scientific_result_2026-07-29.json"
)
V14_RESULT_FILE_SHA256 = (
    "290cde5ef5dd2bf4fc93fd15b5fc1fd107fd857291abf29d4d57351d843f5263"
)
V14_RESULT_BYTE_COUNT = 9_806
V15_RESULT_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "integrity_replacement_v1_scientific_result_2026-07-29.json"
)
V15_RESULT_FILE_SHA256 = (
    "f2597d6d73d39c66352eda301a661650cbd52e7936143512aa76bac9f5a58a01"
)
V15_RESULT_BYTE_COUNT = 10_935
V17_RESULT_PATH = (
    "docs/lewm_go2_rgb_delayed_onset_ego_motion_aligned_ray_consistency_"
    "joint_jepa_v17_scientific_result_2026-07-30.json"
)
V17_RESULT_FILE_SHA256 = (
    "063c4159423576afec5ef7926eb0ee9dc0cde8d57e7542f0ed79bdc7912b1126"
)
V17_RESULT_BYTE_COUNT = 12_464
V10_RESULT_PATH = (
    "docs/lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_result_2026-07-29.md"
)
V10_RESULT_FILE_SHA256 = (
    "1cb9e817377d356ebbb0923f8ec9a43276551c31a001d791d332a5dfc5fdda80"
)
V10_RESULT_BYTE_COUNT = 3_389
V10_CALIBRATION_RESULT_PATH = (
    "docs/lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_physical_evidence_calibration_"
    "result_2026-07-29.md"
)
V10_CALIBRATION_RESULT_FILE_SHA256 = (
    "c7d063ebd58ea60fce491afe6c1aa4a90b0b5fcfc7297140d0aa2e4e9f6b9feb"
)
V10_CALIBRATION_RESULT_BYTE_COUNT = 3_688

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v3/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
MODEL_REQUIRED_METHODS = (
    "encode_online",
    "encode_target",
    "encode_online_with_evidence",
    "encode_online_with_auxiliary_evidence",
    "encode_online_training",
    "semantic_logits_from_latent",
    "online_state",
    "predict_all_actions_with_survival",
    "update_target_ema_after_optimizer_step",
    "trainable_parameter_groups_v13",
    "trainable_parameter_groups_v18",
)
MODEL_REQUIRED_CONSTANTS = {
    "SHARED_ROUTE_PARAMETER_COUNT_V13": 3_102_824,
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13": 77_506,
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13": 259_073,
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13": 3_439_403,
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13": 3_106_344,
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13": 3_520,
}
MODEL_REQUIRED_NATIVE_CONSTANTS = {
    "OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18": 3_520,
    "OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18": 73_986,
    "SHARED_ROUTE_PARAMETER_COUNT_V18": 3_102_824,
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V18": 77_506,
    "PREDICTOR_GROUP_PARAMETER_COUNT_V18": 259_073,
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V18": 3_439_403,
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V18": 3_106_344,
    "VOLUME_INITIALIZATION_SEED_V18": 20_260_729,
    "HEIGHT_CENTRES_M_V18": (
        -0.333,
        -0.183,
        -0.033,
        0.117,
        0.267,
        0.417,
        0.567,
        0.717,
    ),
}
EXPECTED_PARAMETER_PREFIXES = {
    "SHARED_PARAMETER_PREFIXES_V13": (
        "encoder.",
        "bev_lift.evidence_head.",
    ),
    "REPRESENTATION_PARAMETER_PREFIXES_V13": (
        "bev_lift.point_projection.",
        "bev_lift.volume_block.",
        "semantic_head.",
    ),
    "PREDICTOR_PARAMETER_PREFIXES_V13": ("predictor.",),
    "TARGET_PARAMETER_PREFIXES_V13": (
        "target_encoder.",
        "target_bev_lift.evidence_head.",
        "target_bev_lift.point_projection.",
        "target_bev_lift.volume_block.",
    ),
}
MATCHED_UPDATE400_THRESHOLDS = {
    "passed_margin_count_strictly_greater_than": 72,
    "total_shortfall_strictly_less_than": 68.96954700805838,
    "rough_depth_p95_m_strictly_less_than": 1.8582415819168085,
}

_PRISTINE_V14_DEFAULTS = {
    "SCHEMA_PREFIX": "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14",
    "PREREGISTRATION_COMMIT": "456d864b9e03a46f3f79ef413a1bd29ae88b6ace",
    "PREREGISTRATION_PATH": (
        "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
        "preregistration_2026-07-29.md"
    ),
    "PREREGISTRATION_FILE_SHA256": (
        "7fb608208e2e76dfefa3e039b0ab1128230423642b783fe9f79376cae107e16f"
    ),
    "PREREGISTRATION_BYTE_COUNT": 8_214,
    "OUTPUT_ROOT_RELATIVE_PATH": (
        ".generated/go2_rgb_unified_ray_survival_joint_jepa_v14/attempt_v1"
    ),
    "MODEL_CLASS_NAME": "GeometryAnchoredSweptProgressSurvivalJointJepaV14",
    "MAXIMUM_UPDATES": 1_000,
    "MAXIMUM_PRESENTATIONS": 16_000,
    "OBSERVATION_UPDATES": (0, 100, 400, 1_000),
    "TERMINAL_UPDATES": (400, 1_000),
}


def _load_private_v14_executor() -> ModuleType:
    if V14_EXECUTOR_PATH.is_symlink() or not V14_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("V14 executor source is absent or not regular")
    source = V14_EXECUTOR_PATH.read_bytes()
    if not source:
        raise RuntimeError("V14 executor source is empty")
    if PRIVATE_V14_MODULE_NAME in sys.modules:
        raise RuntimeError("private V14 executor module name is already occupied")
    module = ModuleType(PRIVATE_V14_MODULE_NAME)
    module.__file__ = str(V14_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V14_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V14_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V14_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V14_MODULE_NAME)
    return module


def _assert_pristine_v14_defaults(module: ModuleType) -> None:
    observed = {
        name: getattr(module, name, object()) for name in _PRISTINE_V14_DEFAULTS
    }
    if observed != _PRISTINE_V14_DEFAULTS:
        changed = sorted(
            name
            for name, expected in _PRISTINE_V14_DEFAULTS.items()
            if observed[name] != expected
        )
        raise RuntimeError(f"V14 executor adapter defaults changed: {changed}")
    if getattr(module, "CURRENT_EXECUTION_AUTHORIZED", None) is not False:
        raise PermissionError("V14 source shell unexpectedly grants execution")


_base = _load_private_v14_executor()
_assert_pristine_v14_defaults(_base)
_engine = _base._engine
_original_validate_bound_sources = _engine.validate_bound_sources_v13
_original_validate_update_integrity = _engine._validate_update_integrity_v13
_original_evaluate_update400_gate = _engine.evaluate_update400_gate_v13

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
for _path, _binding in {
    PREREGISTRATION_PATH: (
        PREREGISTRATION_FILE_SHA256,
        PREREGISTRATION_BYTE_COUNT,
    ),
    REPLACEMENT_V2_PREREGISTRATION_PATH: (
        REPLACEMENT_V2_PREREGISTRATION_FILE_SHA256,
        REPLACEMENT_V2_PREREGISTRATION_BYTE_COUNT,
    ),
    REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_PATH: (
        REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_FILE_SHA256,
        REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
    ),
    REPLACEMENT_V1_PREREGISTRATION_PATH: (
        REPLACEMENT_V1_PREREGISTRATION_FILE_SHA256,
        REPLACEMENT_V1_PREREGISTRATION_BYTE_COUNT,
    ),
    REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_PATH: (
        REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_FILE_SHA256,
        REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
    ),
    ORIGINAL_V18_PREREGISTRATION_PATH: (
        ORIGINAL_V18_PREREGISTRATION_FILE_SHA256,
        ORIGINAL_V18_PREREGISTRATION_BYTE_COUNT,
    ),
    V18_TERMINAL_FAILURE_RESULT_PATH: (
        V18_TERMINAL_FAILURE_RESULT_FILE_SHA256,
        V18_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
    ),
    V14_RESULT_PATH: (V14_RESULT_FILE_SHA256, V14_RESULT_BYTE_COUNT),
    V15_RESULT_PATH: (V15_RESULT_FILE_SHA256, V15_RESULT_BYTE_COUNT),
    V17_RESULT_PATH: (V17_RESULT_FILE_SHA256, V17_RESULT_BYTE_COUNT),
    V10_RESULT_PATH: (V10_RESULT_FILE_SHA256, V10_RESULT_BYTE_COUNT),
    V10_CALIBRATION_RESULT_PATH: (
        V10_CALIBRATION_RESULT_FILE_SHA256,
        V10_CALIBRATION_RESULT_BYTE_COUNT,
    ),
}.items():
    _bound_parent_sources[_path] = _binding

_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.MODEL_CLASS_NAME = MODEL_CLASS_NAME
_engine.MODEL_REQUIRED_METHODS = MODEL_REQUIRED_METHODS
_engine.MODEL_REQUIRED_CONSTANTS = dict(MODEL_REQUIRED_CONSTANTS)
_engine.MATCHED_UPDATE400_THRESHOLDS = dict(MATCHED_UPDATE400_THRESHOLDS)
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V18 integrity-replacement scientific execution is denied until "
    "recursive source closure, independent exact-binding review, narrow "
    "clean-export certification, and one-shot authority are committed and "
    "validated"
)


def validate_model_api_v18(module: Any) -> dict[str, Any]:
    model_class = getattr(module, MODEL_CLASS_NAME, None)
    if not isinstance(model_class, type):
        raise RuntimeError("V18 model class is absent")
    missing = [
        name for name in MODEL_REQUIRED_METHODS if not callable(getattr(model_class, name, None))
    ]
    if missing:
        raise RuntimeError(f"V18 model API is incomplete: {missing}")
    for name, expected in MODEL_REQUIRED_CONSTANTS.items():
        if getattr(module, name, None) != expected:
            raise RuntimeError(f"V18 model constant changed: {name}")
    for name, expected in MODEL_REQUIRED_NATIVE_CONSTANTS.items():
        if getattr(module, name, None) != expected:
            raise RuntimeError(f"V18 native model constant changed: {name}")
    for name, expected in EXPECTED_PARAMETER_PREFIXES.items():
        if tuple(getattr(module, name, ())) != expected:
            raise RuntimeError(f"V18 model parameter prefixes changed: {name}")
    return {
        "model_class": MODEL_CLASS_NAME,
        "method_count": len(MODEL_REQUIRED_METHODS),
        "online_trainable_parameter_count": MODEL_REQUIRED_CONSTANTS[
            "ONLINE_TRAINABLE_PARAMETER_COUNT_V13"
        ],
    }


def validate_bound_sources_v18(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_update_integrity_v18(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _original_validate_update_integrity(
        runtime,
        model,
        result,
        update=update,
        access_receipt=access_receipt,
    )
    representation = result.gradient_routes["representation"]
    representation_active = float(representation.preclip_l2) > 0.0
    if update == 100 and not representation_active:
        raise RuntimeError("V18 update-100 representation gradient route is zero")
    receipt["v18_height_volume"] = {
        "representation_route_active": representation_active,
        "height_count": 8,
        "volume_channel_count": 8,
        "flattened_channel_count": 64,
    }
    return receipt


def evaluate_update400_gate_v18(
    update100: Mapping[str, Any],
    update400: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    matched_update400_thresholds: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    selected = (
        MATCHED_UPDATE400_THRESHOLDS
        if matched_update400_thresholds is None
        else matched_update400_thresholds
    )
    inherited = _original_evaluate_update400_gate(
        update100,
        update400,
        controls,
        integrity_pass=integrity_pass,
        matched_update400_thresholds=selected,
    )
    retained_names = (
        "structural_integrity_pass",
        "all_twelve_causal_control_checks_true",
        "passed_physical_margin_count_strictly_above_matched_update400",
        "total_physical_shortfall_strictly_below_matched_update400",
        "rough_depth_p95_strictly_below_matched_update400",
    )
    checks = {name: inherited["checks"][name] for name in retained_names}
    passed = all(checks.values())
    return {
        "schema": f"{SCHEMA_PREFIX}_update400_falsification_gate_v1",
        "checks": checks,
        "diagnostic_direction_checks": {
            name: value
            for name, value in inherited["checks"].items()
            if name not in retained_names
        },
        "rough_direction_checks": inherited["rough_direction_checks"],
        "causal_control_checks": inherited["causal_control_checks"],
        "matched_update400_thresholds": dict(selected),
        "passed": passed,
        "action": (
            "CONTINUE_TO_UPDATE_1000"
            if passed
            else "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
        ),
        "next_update": 1_000 if passed else None,
        "retry_authorized": False,
        "resume_authorized": False,
    }


_engine.validate_bound_sources_v13 = validate_bound_sources_v18
_engine.validate_model_api_v13 = validate_model_api_v18
_engine._validate_update_integrity_v13 = validate_update_integrity_v18
_engine.evaluate_update400_gate_v13 = evaluate_update400_gate_v18

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = _engine.TRAINING_REQUIRED_BATCH_KEYS
RUNTIME_INPUT_BINDING_NAMES = _engine.RUNTIME_INPUT_BINDING_NAMES
CONSTRUCTOR_INITIALIZATION_SEED = _engine.CONSTRUCTOR_INITIALIZATION_SEED
SCHEDULE_SEED = _engine.SCHEDULE_SEED
EXPERIMENT_SEED = _engine.EXPERIMENT_SEED
BOOTSTRAP_SEED = _engine.BOOTSTRAP_SEED
PROJECTION_INITIALIZATION_SEED = _engine.PROJECTION_INITIALIZATION_SEED
MICROBATCH_SIZE = _engine.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _engine.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _engine.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _engine.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _engine.MAXIMUM_PRESENTATIONS
OBSERVATION_UPDATES = _engine.OBSERVATION_UPDATES
TERMINAL_UPDATES = _engine.TERMINAL_UPDATES
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = _engine.CHECKPOINT_SCHEDULE_PREFIX_SHA256
CONTROL_NAMES = _engine.CONTROL_NAMES
CONTROL_CHECK_NAMES = _engine.CONTROL_CHECK_NAMES
V12_GATE_CHECK_NAMES = _engine.V12_GATE_CHECK_NAMES
SCOPES = _engine.SCOPES
REGISTERED_FAMILIES = _engine.REGISTERED_FAMILIES
FINAL_PHYSICAL_THRESHOLDS = _engine.FINAL_PHYSICAL_THRESHOLDS
DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = _engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
)
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = (
    _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
)
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH

validate_content_bound_v18 = _engine.validate_content_bound_v13
validate_training_api_v18 = _engine.validate_training_api_v13
validate_future_execution_prerequisites_v18 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v18 = _engine.execution_denial_receipt_v13
reserve_attempt_v18 = _engine.reserve_attempt_v13
terminalize_failure_v18 = _engine.terminalize_failure_v13
evaluate_final_gate_v18 = _engine.evaluate_final_gate_v13
validate_schedule_v18 = _engine.validate_schedule_v13
validate_attempt_reservation_v18 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v18 = _engine.run_future_authorized_engine_v13
validate_terminal_accounting_v18 = _engine.validate_terminal_accounting_v13
execute_v18 = _engine.execute_v13
flatten_physical_metrics_v18 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v18 = _engine.registered_wrong_rgb_mapping_v13

# Compatibility names consumed by the private V13 custody launcher/runtime.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v18
validate_bound_sources_v13 = validate_bound_sources_v18
validate_model_api_v13 = validate_model_api_v18
validate_training_api_v13 = validate_training_api_v18
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v18
)
execution_denial_receipt_v13 = execution_denial_receipt_v18
reserve_attempt_v13 = reserve_attempt_v18
terminalize_failure_v13 = terminalize_failure_v18
flatten_physical_metrics_v13 = flatten_physical_metrics_v18
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v18
evaluate_update400_gate_v13 = evaluate_update400_gate_v18
evaluate_final_gate_v13 = evaluate_final_gate_v18
validate_schedule_v13 = validate_schedule_v18
validate_attempt_reservation_v13 = validate_attempt_reservation_v18
run_future_authorized_engine_v13 = run_future_authorized_engine_v18
validate_terminal_accounting_v13 = validate_terminal_accounting_v18
execute_v13 = execute_v18


def private_adapter_receipt_v18() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v14_executor_adapter_v1",
        "base_executor": str(V14_EXECUTOR_PATH.relative_to(ROOT)),
        "public_v14_was_loaded_before_adapter": (
            _PUBLIC_V14_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_v14_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V14_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "replacement_v2_preregistration_commit": (
            REPLACEMENT_V2_PREREGISTRATION_COMMIT
        ),
        "replacement_v2_terminal_failure_result_commit": (
            REPLACEMENT_V2_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "replacement_v1_preregistration_commit": (
            REPLACEMENT_V1_PREREGISTRATION_COMMIT
        ),
        "replacement_v1_terminal_failure_result_commit": (
            REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "original_v18_preregistration_commit": (
            ORIGINAL_V18_PREREGISTRATION_COMMIT
        ),
        "v18_terminal_failure_result_commit": (
            V18_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "model_class": MODEL_CLASS_NAME,
        "matched_update400_thresholds": dict(MATCHED_UPDATE400_THRESHOLDS),
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V18 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v18(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
