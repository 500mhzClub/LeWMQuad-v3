#!/usr/bin/env python3
"""Denied-by-default executor adapter for unified ray-survival V14.

The reviewed V13 lifecycle is loaded into a private module object and adapted
only through its registered successor hooks.  The process-global V13 module
is never imported or mutated.  V14 remains source-only until a later launcher
supplies independently reviewed clean-export and one-shot authority.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V13_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
V13_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck"
)
PRIVATE_V13_MODULE_NAME = f"{__name__}.__private_v13_executor"
_PUBLIC_V13_WAS_LOADED_BEFORE_ADAPTER = V13_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14"
PREREGISTRATION_COMMIT = "456d864b9e03a46f3f79ef413a1bd29ae88b6ace"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "preregistration_2026-07-29.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "7fb608208e2e76dfefa3e039b0ab1128230423642b783fe9f79376cae107e16f"
)
PREREGISTRATION_BYTE_COUNT = 8_214
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_unified_ray_survival_joint_jepa_v14/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV14"

MATCHED_UPDATE400_THRESHOLDS = {
    "passed_margin_count_strictly_greater_than": 71,
    "total_shortfall_strictly_less_than": 71.67935936391197,
    "rough_depth_p95_m_strictly_less_than": 1.936374711990354,
}
MODEL_REQUIRED_CONSTANTS = {
    # Compatibility keys are retained because the frozen V13 initialization
    # validator indexes them directly; their values are the V14 counts.
    "SHARED_ROUTE_PARAMETER_COUNT_V13": 3_102_824,
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13": 22_020,
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13": 259_073,
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13": 3_383_917,
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13": 3_106_216,
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13": 3_392,
}

_PRISTINE_V13_DEFAULTS = {
    "SCHEMA_PREFIX": "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13",
    "PREREGISTRATION_COMMIT": "a285129651a0c418467d95c7c1e3d7a1767453d2",
    "PREREGISTRATION_PATH": (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_preregistration_2026-07-29.md"
    ),
    "OUTPUT_ROOT_RELATIVE_PATH": (
        ".generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3/attempt_v1"
    ),
    "MODEL_CLASS_NAME": "GeometryAnchoredSweptProgressSurvivalJointJepaV13",
    "MODEL_REQUIRED_METHODS": (
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
    ),
    "MODEL_REQUIRED_CONSTANTS": {
        "SHARED_ROUTE_PARAMETER_COUNT_V13": 3_105_513,
        "REPRESENTATION_GROUP_PARAMETER_COUNT_V13": 22_020,
        "PREDICTOR_GROUP_PARAMETER_COUNT_V13": 259_073,
        "ONLINE_TRAINABLE_PARAMETER_COUNT_V13": 3_386_606,
        "TARGET_BOTTLENECK_PARAMETER_COUNT_V13": 3_108_905,
        "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13": 3_392,
    },
    "MATCHED_UPDATE400_THRESHOLDS": None,
}


def _load_private_v13_executor() -> ModuleType:
    """Execute the exact V13 source without registering a shared singleton."""

    if V13_EXECUTOR_PATH.is_symlink() or not V13_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("V13 executor source is absent or not regular")
    source = V13_EXECUTOR_PATH.read_bytes()
    if not source:
        raise RuntimeError("V13 executor source is empty")
    if PRIVATE_V13_MODULE_NAME in sys.modules:
        raise RuntimeError("private V13 executor module name is already occupied")
    module = ModuleType(PRIVATE_V13_MODULE_NAME)
    module.__file__ = str(V13_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V13_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V13_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V13_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V13_MODULE_NAME)
    return module


def _assert_pristine_v13_defaults(module: ModuleType) -> None:
    observed = {
        name: getattr(module, name, object()) for name in _PRISTINE_V13_DEFAULTS
    }
    if observed != _PRISTINE_V13_DEFAULTS:
        changed = sorted(
            name
            for name, expected in _PRISTINE_V13_DEFAULTS.items()
            if observed[name] != expected
        )
        raise RuntimeError(f"V13 executor adapter defaults changed: {changed}")
    if getattr(module, "CURRENT_EXECUTION_AUTHORIZED", None) is not False:
        raise PermissionError("V13 source shell unexpectedly grants execution")
    old_binding = module.BOUND_PARENT_SOURCES.get(module.PREREGISTRATION_PATH)
    if old_binding != (
        "cd800bb409054947bbfa1159ef362a0d465799238ed1862b0af7acc9ab883a08",
        4_005,
    ):
        raise RuntimeError("V13 preregistration source binding changed")


_engine = _load_private_v13_executor()
_assert_pristine_v13_defaults(_engine)
_original_validate_model_api = _engine.validate_model_api_v13

_old_preregistration_path = _engine.PREREGISTRATION_PATH
_bound_parent_sources = {
    (PREREGISTRATION_PATH if path == _old_preregistration_path else path): (
        (PREREGISTRATION_FILE_SHA256, PREREGISTRATION_BYTE_COUNT)
        if path == _old_preregistration_path
        else binding
    )
    for path, binding in _engine.BOUND_PARENT_SOURCES.items()
}

_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.MODEL_CLASS_NAME = MODEL_CLASS_NAME
_engine.MODEL_REQUIRED_METHODS = (
    *_engine.MODEL_REQUIRED_METHODS,
    "trainable_parameter_groups_v14",
)
_engine.MODEL_REQUIRED_CONSTANTS = dict(MODEL_REQUIRED_CONSTANTS)
_engine.MATCHED_UPDATE400_THRESHOLDS = dict(MATCHED_UPDATE400_THRESHOLDS)
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V14 scientific execution is denied until its recursive source closure, "
    "independent exact-binding review, custody clean-export exception and "
    "certification, frozen-export validation, and one-shot execution binding "
    "are committed and validated by the custodian-owned launcher"
)


def validate_model_api_v14(module: Any) -> dict[str, Any]:
    return _original_validate_model_api(module)


# The frozen initialization path resolves this hook from its private globals.
_engine.validate_model_api_v13 = validate_model_api_v14

# Re-export the reviewed lifecycle under V14 names.  Function globals continue
# to point only at the privately adapted module object.
EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
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
TRACE_RELATIVE_PATH = _engine.TRACE_RELATIVE_PATH
METRIC_RELATIVE_PATHS = _engine.METRIC_RELATIVE_PATHS
DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = _engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
)
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = (
    _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
)
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH
SemanticProbabilityRasterV14 = _engine.SemanticProbabilityRasterV13

validate_content_bound_v14 = _engine.validate_content_bound_v13


def validate_bound_sources_v14(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _engine.validate_bound_sources_v13(repository_root, selected)


validate_training_api_v14 = _engine.validate_training_api_v13
validate_future_execution_prerequisites_v14 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v14 = _engine.execution_denial_receipt_v13
reserve_attempt_v14 = _engine.reserve_attempt_v13
terminalize_failure_v14 = _engine.terminalize_failure_v13
adapt_nominal_logits_with_target_metadata_v14 = (
    _engine.adapt_nominal_logits_with_target_metadata_v13
)
update_physical_accumulator_from_rgb_v14 = (
    _engine.update_physical_accumulator_from_rgb_v13
)
flatten_physical_metrics_v14 = _engine.flatten_physical_metrics_v13
physical_margins_v14 = _engine.physical_margins_v13
evaluate_physical_scopes_v14 = _engine.evaluate_physical_scopes_v13
registered_wrong_rgb_mapping_v14 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v14 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v14 = _engine.evaluate_final_gate_v13
validate_schedule_v14 = _engine.validate_schedule_v13
validate_attempt_reservation_v14 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v14 = _engine.run_future_authorized_engine_v13
validate_terminal_accounting_v14 = _engine.validate_terminal_accounting_v13
execute_v14 = _engine.execute_v13

# Compatibility surface consumed by the privately cloned V13 launcher and
# runtime.  Each object below belongs to this module's private adapted engine;
# none is taken from or installed onto the process-global V13 module.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v14
validate_bound_sources_v13 = validate_bound_sources_v14
validate_model_api_v13 = validate_model_api_v14
validate_training_api_v13 = validate_training_api_v14
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v14
)
execution_denial_receipt_v13 = execution_denial_receipt_v14
reserve_attempt_v13 = reserve_attempt_v14
terminalize_failure_v13 = terminalize_failure_v14
adapt_nominal_logits_with_target_metadata_v13 = (
    adapt_nominal_logits_with_target_metadata_v14
)
update_physical_accumulator_from_rgb_v13 = (
    update_physical_accumulator_from_rgb_v14
)
flatten_physical_metrics_v13 = flatten_physical_metrics_v14
physical_margins_v13 = physical_margins_v14
evaluate_physical_scopes_v13 = evaluate_physical_scopes_v14
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v14
evaluate_update400_gate_v13 = evaluate_update400_gate_v14
evaluate_final_gate_v13 = evaluate_final_gate_v14
validate_schedule_v13 = validate_schedule_v14
validate_attempt_reservation_v13 = validate_attempt_reservation_v14
run_future_authorized_engine_v13 = run_future_authorized_engine_v14
validate_terminal_accounting_v13 = validate_terminal_accounting_v14
execute_v13 = execute_v14


def private_adapter_receipt_v14() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v13_executor_adapter_v1",
        "base_executor": str(V13_EXECUTOR_PATH.relative_to(ROOT)),
        "public_v13_was_loaded_before_adapter": (
            _PUBLIC_V13_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_v13_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V13_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "model_class": MODEL_CLASS_NAME,
        "matched_update400_thresholds": dict(MATCHED_UPDATE400_THRESHOLDS),
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V14 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v14(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
