#!/usr/bin/env python3
"""Denied-by-default executor adapter for V16 ray consistency."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence

from scripts import (
    go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_lifecycle
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

SCHEMA_PREFIX = "lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16"
PREREGISTRATION_COMMIT = "2792343e14d3376add9d6adbda7f29346a3e9e29"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "preregistration_2026-07-29.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "f984a34baa8c5541fd1d35ea744c8ced051b92882fd8e153d3be8b2f9f62747d"
)
PREREGISTRATION_BYTE_COUNT = 9_844
V15_RESULT_COMMIT = "7a0dbc1f850bc8917bc45566425116fdef87ef42"
V15_RESULT_PATH = (
    "docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
    "integrity_replacement_v1_scientific_result_2026-07-29.json"
)
V15_RESULT_FILE_SHA256 = (
    "f2597d6d73d39c66352eda301a661650cbd52e7936143512aa76bac9f5a58a01"
)
V15_RESULT_BYTE_COUNT = 10_935
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16/"
    "attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV14"
REALIZED_RELATIVE_SE2_KEY = "realized_relative_se2_current_frame"

MATCHED_UPDATE400_THRESHOLDS = {
    "passed_margin_count_strictly_greater_than": 71,
    "total_shortfall_strictly_less_than": 68.96964862816927,
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
_original_validate_update_integrity = _engine._validate_update_integrity_v13

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources[PREREGISTRATION_PATH] = (
    PREREGISTRATION_FILE_SHA256,
    PREREGISTRATION_BYTE_COUNT,
)
_bound_parent_sources[V15_RESULT_PATH] = (
    V15_RESULT_FILE_SHA256,
    V15_RESULT_BYTE_COUNT,
)

_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.MATCHED_UPDATE400_THRESHOLDS = dict(MATCHED_UPDATE400_THRESHOLDS)
_engine.TRAINING_REQUIRED_BATCH_KEYS = (
    *_engine.TRAINING_REQUIRED_BATCH_KEYS,
    REALIZED_RELATIVE_SE2_KEY,
)
_engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = (
    "checkpoints/update_1000.pt"
)
_engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    "checkpoints/update_1000.binding.json"
)
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V16 scientific execution is denied until recursive source closure, "
    "independent review, narrow clean-export certification, and one-shot "
    "execution authority are committed and validated"
)


def validate_update_integrity_v16(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain every V13 integrity check and validate the two V16 loss fields."""

    losses = dict(result.mean_losses)
    old_names = {"S", "P", "U", "R", "O", "N", "C", "L"}
    if set(losses) != old_names | {"C_base", "M"}:
        raise RuntimeError("V16 per-update mean-loss receipt set changed")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in losses.values()
    ):
        raise FloatingPointError("V16 mean-loss receipt is nonfinite")
    if losses["M"] < -1e-7 or not math.isclose(
        losses["C"],
        losses["C_base"] + 0.1 * losses["M"],
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V16 Camera loss equation changed")

    proxy_values = dict(vars(result))
    proxy_values["mean_losses"] = {
        name: losses[name]
        for name in ("S", "P", "U", "R", "O", "N", "C", "L")
    }
    receipt = _original_validate_update_integrity(
        runtime,
        model,
        SimpleNamespace(**proxy_values),
        update=update,
        access_receipt=access_receipt,
    )
    valid_count = result.ray_consistency_shared_valid_cell_count
    weighted_count = result.ray_consistency_positive_weight_cell_count
    weight_sum = result.ray_consistency_weight_sum
    if (
        type(valid_count) is not int
        or type(weighted_count) is not int
        or valid_count <= 0
        or weighted_count <= 0
        or weighted_count > valid_count
        or isinstance(weight_sum, bool)
        or not isinstance(weight_sum, (int, float))
        or not math.isfinite(float(weight_sum))
        or float(weight_sum) <= 0.0
    ):
        raise RuntimeError("V16 ray-consistency support receipt is invalid")
    receipt["mean_losses"] = losses
    receipt["ray_consistency"] = {
        "shared_valid_cell_count": valid_count,
        "positive_weight_cell_count": weighted_count,
        "weight_sum": float(weight_sum),
        "loss_weight": 0.1,
    }
    return receipt


def run_future_authorized_engine_v16(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    return _lifecycle.run_future_authorized_engine_v16(
        authority=authority,
        reservation=reservation,
        runtime=runtime,
        publisher=publisher,
        engine=_engine,
    )


_engine._validate_update_integrity_v13 = validate_update_integrity_v16
_engine.run_future_authorized_engine_v13 = run_future_authorized_engine_v16

# Public V16 contract.
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
SemanticProbabilityRasterV16 = _engine.SemanticProbabilityRasterV13

validate_content_bound_v16 = _engine.validate_content_bound_v13


def validate_bound_sources_v16(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _engine.validate_bound_sources_v13(repository_root, selected)


validate_model_api_v16 = _engine.validate_model_api_v13
validate_training_api_v16 = _engine.validate_training_api_v13
validate_future_execution_prerequisites_v16 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v16 = _engine.execution_denial_receipt_v13
reserve_attempt_v16 = _engine.reserve_attempt_v13
terminalize_failure_v16 = _engine.terminalize_failure_v13
adapt_nominal_logits_with_target_metadata_v16 = (
    _engine.adapt_nominal_logits_with_target_metadata_v13
)
update_physical_accumulator_from_rgb_v16 = (
    _engine.update_physical_accumulator_from_rgb_v13
)
flatten_physical_metrics_v16 = _engine.flatten_physical_metrics_v13
physical_margins_v16 = _engine.physical_margins_v13
evaluate_physical_scopes_v16 = _engine.evaluate_physical_scopes_v13
registered_wrong_rgb_mapping_v16 = _engine.registered_wrong_rgb_mapping_v13
def evaluate_update400_gate_v16(
    update100: Mapping[str, Any],
    update400: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    matched_update400_thresholds: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    """Compatibility surface for the exact preregistered V16 gate."""

    _engine._validate_physical_summary(update100)
    if matched_update400_thresholds not in (None, MATCHED_UPDATE400_THRESHOLDS):
        raise ValueError("V16 update-400 thresholds changed")
    return _lifecycle.evaluate_update400_gate_v16(
        update400,
        controls,
        integrity_pass=integrity_pass,
        engine=_engine,
    )


_engine.evaluate_update400_gate_v13 = evaluate_update400_gate_v16
evaluate_final_gate_v16 = _engine.evaluate_final_gate_v13
validate_schedule_v16 = _engine.validate_schedule_v13
validate_attempt_reservation_v16 = _engine.validate_attempt_reservation_v13
validate_terminal_accounting_v16 = _engine.validate_terminal_accounting_v13
execute_v16 = _engine.execute_v13

# Compatibility surface consumed by the privately cloned V13 launcher/runtime.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v16
validate_bound_sources_v13 = validate_bound_sources_v16
validate_model_api_v13 = validate_model_api_v16
validate_training_api_v13 = validate_training_api_v16
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v16
)
execution_denial_receipt_v13 = execution_denial_receipt_v16
reserve_attempt_v13 = reserve_attempt_v16
terminalize_failure_v13 = terminalize_failure_v16
adapt_nominal_logits_with_target_metadata_v13 = (
    adapt_nominal_logits_with_target_metadata_v16
)
update_physical_accumulator_from_rgb_v13 = (
    update_physical_accumulator_from_rgb_v16
)
flatten_physical_metrics_v13 = flatten_physical_metrics_v16
physical_margins_v13 = physical_margins_v16
evaluate_physical_scopes_v13 = evaluate_physical_scopes_v16
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v16
evaluate_update400_gate_v13 = evaluate_update400_gate_v16
evaluate_final_gate_v13 = evaluate_final_gate_v16
validate_schedule_v13 = validate_schedule_v16
validate_attempt_reservation_v13 = validate_attempt_reservation_v16
run_future_authorized_engine_v13 = run_future_authorized_engine_v16
validate_terminal_accounting_v13 = validate_terminal_accounting_v16
execute_v13 = execute_v16


def private_adapter_receipt_v16() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v14_executor_adapter_v1",
        "base_executor": str(V14_EXECUTOR_PATH.relative_to(ROOT)),
        "public_v14_was_loaded_before_adapter": (
            _PUBLIC_V14_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_v14_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V14_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v15_result_commit": V15_RESULT_COMMIT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "model_class": MODEL_CLASS_NAME,
        "ray_consistency_weight": 0.1,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V16 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v16(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
