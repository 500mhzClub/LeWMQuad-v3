#!/usr/bin/env python3
"""Denied-by-default executor adapter for V15 extended-horizon training.

V15 keeps the exact V14 model and scientific training mechanism.  It loads
the frozen V14 adapter into a private module object, changes only the
preregistered lifecycle constants, and delegates the longer one-shot
controller to a V15-only helper.  Importing this module grants no execution
authority and opens no scientific payload.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

from scripts import (
    go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_lifecycle
    as _lifecycle,
)


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
    "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon"
)
PREREGISTRATION_COMMIT = "af0f786841b1404d1f42542b507ad198ee574250"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_"
    "extended_horizon_preregistration_2026-07-29.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "bbecbb533abee54fff408ccbcdb648b922a883c6c81a9c5ee8795367e2f6a187"
)
PREREGISTRATION_BYTE_COUNT = 8_806
V14_RESULT_COMMIT = "d54dfea445dc9bc80cee6421c1b0aea2639463f1"
V14_RESULT_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_"
    "scientific_result_2026-07-29.json"
)
V14_RESULT_FILE_SHA256 = (
    "290cde5ef5dd2bf4fc93fd15b5fc1fd107fd857291abf29d4d57351d843f5263"
)
V14_RESULT_BYTE_COUNT = 9_806
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_unified_ray_survival_joint_jepa_v15_"
    "extended_horizon/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV14"

MAXIMUM_UPDATES = 2_000
MAXIMUM_PRESENTATIONS = 32_000
OBSERVATION_UPDATES = (0, 100, 400, 1_000, 1_400, 2_000)
TERMINAL_UPDATES = (400, 1_400, 2_000)
FINAL_UPDATE = 2_000
BASE_SCHEDULE_PRESENTATIONS = 16_000
SCHEDULE_REPETITIONS = 2

UPDATE1400_THRESHOLDS = dict(_lifecycle.UPDATE1400_THRESHOLDS_V15)

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

_original_validate_schedule = _engine.validate_schedule_v13
_original_evaluate_update400_gate = _engine.evaluate_update400_gate_v13
_original_evaluate_final_gate = _engine.evaluate_final_gate_v13

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources[PREREGISTRATION_PATH] = (
    PREREGISTRATION_FILE_SHA256,
    PREREGISTRATION_BYTE_COUNT,
)
_bound_parent_sources[V14_RESULT_PATH] = (
    V14_RESULT_FILE_SHA256,
    V14_RESULT_BYTE_COUNT,
)

_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.MAXIMUM_UPDATES = MAXIMUM_UPDATES
_engine.MAXIMUM_PRESENTATIONS = MAXIMUM_PRESENTATIONS
_engine.OBSERVATION_UPDATES = OBSERVATION_UPDATES
_engine.TERMINAL_UPDATES = TERMINAL_UPDATES
_engine.METRIC_RELATIVE_PATHS = {
    update: f"metrics/update_{update}.json" for update in OBSERVATION_UPDATES
}
_engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = "checkpoints/update_2000.pt"
_engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    "checkpoints/update_2000.binding.json"
)
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V15 scientific execution is denied until its recursive source closure, "
    "independent exact-binding review, custody clean-export exception and "
    "certification, frozen-export validation, and one-shot execution binding "
    "are committed and validated by the custodian-owned launcher"
)


def validate_schedule_v15(
    schedule: Sequence[int],
    *,
    train_pair_count: int = 4_262,
) -> dict[str, Any]:
    """Validate the frozen base schedule repeated exactly twice in memory."""

    receipt = _original_validate_schedule(
        schedule,
        train_pair_count=train_pair_count,
    )
    normalized = tuple(schedule)
    first = normalized[:BASE_SCHEDULE_PRESENTATIONS]
    second = normalized[BASE_SCHEDULE_PRESENTATIONS:]
    if len(first) != BASE_SCHEDULE_PRESENTATIONS or second != first:
        raise PermissionError("V15 schedule halves are not elementwise identical")
    return {
        **receipt,
        "base_presentation_count": BASE_SCHEDULE_PRESENTATIONS,
        "schedule_repetition_count": SCHEDULE_REPETITIONS,
        "repeated_halves_elementwise_identical": True,
    }


def evaluate_update400_gate_v15(
    update100: Mapping[str, Any],
    update400: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    matched_update400_thresholds: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    decision = _original_evaluate_update400_gate(
        update100,
        update400,
        controls,
        integrity_pass=integrity_pass,
        matched_update400_thresholds=matched_update400_thresholds,
    )
    if decision["passed"]:
        decision["action"] = "CONTINUE_TO_UPDATE_1400"
        decision["next_update"] = 1_400
    return decision


def evaluate_update1400_gate_v15(
    v12_gate: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
) -> dict[str, Any]:
    return _lifecycle.evaluate_update1400_gate_v15(
        v12_gate,
        physical_summary,
        controls,
        integrity_pass=integrity_pass,
        engine=_engine,
    )


def evaluate_final_gate_v15(
    v12_gate: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    *,
    integrity_pass: bool,
) -> dict[str, Any]:
    decision = _original_evaluate_final_gate(
        v12_gate,
        physical_summary,
        integrity_pass=integrity_pass,
    )
    decision["update"] = FINAL_UPDATE
    return decision


def run_future_authorized_engine_v15(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    return _lifecycle.run_future_authorized_engine_v15(
        authority=authority,
        reservation=reservation,
        runtime=runtime,
        publisher=publisher,
        engine=_engine,
    )


_engine.validate_schedule_v13 = validate_schedule_v15
_engine.evaluate_update400_gate_v13 = evaluate_update400_gate_v15
_engine.evaluate_final_gate_v13 = evaluate_final_gate_v15
_engine.run_future_authorized_engine_v13 = run_future_authorized_engine_v15

# Public V15 contract.
EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
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
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = _engine.CHECKPOINT_SCHEDULE_PREFIX_SHA256
CONTROL_NAMES = _engine.CONTROL_NAMES
CONTROL_CHECK_NAMES = _engine.CONTROL_CHECK_NAMES
V12_GATE_CHECK_NAMES = _engine.V12_GATE_CHECK_NAMES
SCOPES = _engine.SCOPES
REGISTERED_FAMILIES = _engine.REGISTERED_FAMILIES
FINAL_PHYSICAL_THRESHOLDS = _engine.FINAL_PHYSICAL_THRESHOLDS
MATCHED_UPDATE400_THRESHOLDS = _engine.MATCHED_UPDATE400_THRESHOLDS
ACCOUNTING_MULTIPLIERS = _engine.ACCOUNTING_MULTIPLIERS
TRACE_RELATIVE_PATH = _engine.TRACE_RELATIVE_PATH
METRIC_RELATIVE_PATHS = _engine.METRIC_RELATIVE_PATHS
DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = (
    _engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
)
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
)
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = (
    _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
)
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH
SemanticProbabilityRasterV15 = _engine.SemanticProbabilityRasterV13

validate_content_bound_v15 = _engine.validate_content_bound_v13


def validate_bound_sources_v15(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _engine.validate_bound_sources_v13(repository_root, selected)


validate_model_api_v15 = _engine.validate_model_api_v13
validate_training_api_v15 = _engine.validate_training_api_v13
validate_future_execution_prerequisites_v15 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v15 = _engine.execution_denial_receipt_v13
reserve_attempt_v15 = _engine.reserve_attempt_v13
terminalize_failure_v15 = _engine.terminalize_failure_v13
adapt_nominal_logits_with_target_metadata_v15 = (
    _engine.adapt_nominal_logits_with_target_metadata_v13
)
update_physical_accumulator_from_rgb_v15 = (
    _engine.update_physical_accumulator_from_rgb_v13
)
flatten_physical_metrics_v15 = _engine.flatten_physical_metrics_v13
physical_margins_v15 = _engine.physical_margins_v13
evaluate_physical_scopes_v15 = _engine.evaluate_physical_scopes_v13
registered_wrong_rgb_mapping_v15 = _engine.registered_wrong_rgb_mapping_v13
validate_attempt_reservation_v15 = _engine.validate_attempt_reservation_v13
validate_terminal_accounting_v15 = _engine.validate_terminal_accounting_v13
execute_v15 = _engine.execute_v13

# Compatibility surface consumed by the privately cloned V13 launcher/runtime.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v15
validate_bound_sources_v13 = validate_bound_sources_v15
validate_model_api_v13 = validate_model_api_v15
validate_training_api_v13 = validate_training_api_v15
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v15
)
execution_denial_receipt_v13 = execution_denial_receipt_v15
reserve_attempt_v13 = reserve_attempt_v15
terminalize_failure_v13 = terminalize_failure_v15
adapt_nominal_logits_with_target_metadata_v13 = (
    adapt_nominal_logits_with_target_metadata_v15
)
update_physical_accumulator_from_rgb_v13 = (
    update_physical_accumulator_from_rgb_v15
)
flatten_physical_metrics_v13 = flatten_physical_metrics_v15
physical_margins_v13 = physical_margins_v15
evaluate_physical_scopes_v13 = evaluate_physical_scopes_v15
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v15
evaluate_update400_gate_v13 = evaluate_update400_gate_v15
evaluate_final_gate_v13 = evaluate_final_gate_v15
validate_schedule_v13 = validate_schedule_v15
validate_attempt_reservation_v13 = validate_attempt_reservation_v15
run_future_authorized_engine_v13 = run_future_authorized_engine_v15
validate_terminal_accounting_v13 = validate_terminal_accounting_v15
execute_v13 = execute_v15


def private_adapter_receipt_v15() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v14_executor_adapter_v1",
        "base_executor": str(V14_EXECUTOR_PATH.relative_to(ROOT)),
        "public_v14_was_loaded_before_adapter": (
            _PUBLIC_V14_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_v14_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V14_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v14_result_commit": V14_RESULT_COMMIT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "model_class": MODEL_CLASS_NAME,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V15 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v15(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
