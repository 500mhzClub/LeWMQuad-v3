#!/usr/bin/env python3
"""Denied-by-default V22 executor adapter over the frozen V21 lifecycle.

Only the auxiliary accounting, loss, route, and diagnostic receipt surfaces
change.  V21's evaluator, causal controls, gates, custody, and one-shot engine
remain authoritative.  This source shell grants no execution authority.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V21_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21.py"
)
V21_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "7071a006dda3851280fbdf030e156862c4f19ab3"
)
V21_EXECUTOR_FILE_SHA256 = (
    "f8f919fc1e05d60d12e84409643b2dffc97a953635a33b0e3931b6d8ae8741d7"
)
V21_EXECUTOR_BYTE_COUNT = 29_263
V21_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
PRIVATE_V21_MODULE_NAME = f"{__name__}.__private_v21_executor"
_PUBLIC_V21_WAS_LOADED_BEFORE_ADAPTER = V21_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
PREREGISTRATION_COMMIT = "43053ae49c28082c616f45ed857eedb727380952"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "7ee36433d739663654de593cf018500cc5547e249173f08201ad4ac5c6b1959e"
)
PREREGISTRATION_BYTE_COUNT = 11_986
V21_SCIENTIFIC_RESULT_COMMIT = "e5b5e56b30cee0c1eb818d52c4d886909f570f4d"
V21_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_scientific_result_2026-07-30.json"
)
V21_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "c9544055b11d162b5b5fc9b02d0a04f3961a61b4547411964812a9ae4c5da1e7"
)
V21_SCIENTIFIC_RESULT_BYTE_COUNT = 15_724
V21_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "2195025bf24e3de621e76a5a5e3ea272ced05bd9f6e4fb91302035137ab7b9ec"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
    "v22/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
SCENE_NEGATIVE_ROW_KEY = "scene_innovation_negative_row"
TWO_AXIS_INNOVATION_ROUTE_NAME = "two_axis_innovation_predictor"
TWO_AXIS_PARAMETER_TENSOR_COUNT = 13
TWO_AXIS_PARAMETER_COUNT = 259_008
TWO_AXIS_GRADIENT_NORM_CAP = 1.0
NONREQUESTED_ACTION_COUNT = 8
SALIENCE_CELL_COUNT = 128
MEAN_LOSS_NAMES = (
    "S", "P", "U", "R", "O", "I_fit", "I_scene_rank",
    "I_action_rank", "I_two_axis", "N", "C", "L",
)
TWO_AXIS_DIAGNOSTIC_NAMES = (
    "positive_energy_mean",
    "scene_negative_energy_mean",
    "scene_advantage_sum",
    "scene_advantage_count",
    "scene_advantage_mean",
    "action_negative_energy_mean",
    "action_advantage_sum",
    "action_advantage_count",
    "action_advantage_mean",
    "nonrequested_action_count_per_row",
    "action_candidate_energy_count",
    "matching_predictor_gradient_cosine",
    "valid_cell_count",
    "high_salience_cell_count",
    "low_salience_cell_count",
)
ACCOUNTING_MULTIPLIERS_V22 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 12,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "two_axis_innovation_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 8,
    "two_axis_innovation_objectives": 4,
}


def _load_private_v21_executor() -> ModuleType:
    if V21_EXECUTOR_PATH.is_symlink() or not V21_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("frozen V21 executor source is absent or not regular")
    source = V21_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V21_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V21_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V21 executor source binding changed")
    if PRIVATE_V21_MODULE_NAME in sys.modules:
        raise RuntimeError("private V21 executor module name is already occupied")
    module = ModuleType(PRIVATE_V21_MODULE_NAME)
    module.__file__ = str(V21_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V21_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V21_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V21_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V21_MODULE_NAME)
    return module


_base = _load_private_v21_executor()
if (
    not _base.SCHEMA_PREFIX.endswith("joint_jepa_v21")
    or _base.MODEL_CLASS_NAME != MODEL_CLASS_NAME
    or _base.MAXIMUM_UPDATES != 1_000
    or _base.MAXIMUM_PRESENTATIONS != 16_000
    or _base.OBSERVATION_UPDATES != (0, 100, 400, 1_000)
    or _base.TERMINAL_UPDATES != (400, 1_000)
    or _base.CURRENT_EXECUTION_AUTHORIZED is not False
):
    raise RuntimeError("frozen V21 executor defaults changed")

_engine = _base._engine
_original_validate_bound_sources = _engine.validate_bound_sources_v13
_original_validate_model_api = _engine.validate_model_api_v13
_original_validate_training_api = _engine.validate_training_api_v13
_original_validate_microbatches = _engine._validate_microbatches_for_engine_v13
_original_validate_update_integrity = _engine._validate_update_integrity_v13
_original_observation = _engine._observation_v13
_original_validate_terminal_accounting = _engine.validate_terminal_accounting_v13
TRAINING_REQUIRED_BATCH_KEYS_V22 = tuple(_base.TRAINING_REQUIRED_BATCH_KEYS_V21)

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V21_SCIENTIFIC_RESULT_PATH: (
            V21_SCIENTIFIC_RESULT_FILE_SHA256,
            V21_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
)
for _module in (_base, _engine):
    _module.SCHEMA_PREFIX = SCHEMA_PREFIX
    _module.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
    _module.PREREGISTRATION_PATH = PREREGISTRATION_PATH
    _module.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
    _module.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V22 scene-action innovation execution is denied until recursive source "
    "closure, independent review, narrow certification, and one-shot authority"
)


def _receipt_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"{name} must be a dataclass or mapping")
    if not all(isinstance(key, str) for key in result):
        raise TypeError(f"{name} keys must be strings")
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def _expected_accounting_v22(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V22.items()
    }


def _project_accounting_to_v21(accounting: Mapping[str, int]) -> dict[str, int]:
    projected = dict(accounting)
    projected["scene_innovation_grad_calls"] = projected.pop(
        "two_axis_innovation_grad_calls"
    )
    projected["scene_innovation_objectives"] = projected.pop(
        "two_axis_innovation_objectives"
    )
    if set(projected) != set(_base.ACCOUNTING_MULTIPLIERS_V21):
        raise RuntimeError("V22-to-V21 accounting projection changed")
    return projected


def validate_bound_sources_v22(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_model_api_v22(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V22 did not retain the exact V18 model class")
    return receipt


def validate_training_api_v22(module: Any) -> dict[str, Any]:
    receipt = dict(_original_validate_training_api(module))
    for name in ("JointTrainingAccountingV22", "JointUpdateResultV22"):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V22 training type is absent: {name}")
    for name in ("joint_training_update_v22", "validate_accounting_v22"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V22 training callable is absent: {name}")
    if (
        tuple(getattr(module, "REQUIRED_BATCH_KEYS_V22", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V22
        or getattr(module, "SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21", None)
        != SCENE_NEGATIVE_ROW_KEY
    ):
        raise RuntimeError("V22 frozen V21 batch schema changed")
    receipt.pop("scene_innovation_route", None)
    receipt.pop("scene_innovation_objectives_per_update", None)
    return {
        **receipt,
        "required_batch_key_count_v22": len(TRAINING_REQUIRED_BATCH_KEYS_V22),
        "two_axis_innovation_route": TWO_AXIS_INNOVATION_ROUTE_NAME,
        "backward_calls_per_update": 12,
        "predictor_objectives_per_update": 8,
        "two_axis_innovation_objectives_per_update": 4,
        "new_batch_fields_over_v21": 0,
    }


def validate_microbatches_for_engine_v22(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != _engine.MICROBATCHES_PER_UPDATE or any(
        type(batch) is not dict or tuple(batch) != TRAINING_REQUIRED_BATCH_KEYS_V22
        for batch in microbatches
    ):
        raise PermissionError("V22 engine microbatch schema changed")
    _original_validate_microbatches(runtime, model, microbatches)


def _validate_two_axis_parameter_subset(model: Any) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("V22 model named-parameter API is absent")
    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters()
        if name.startswith("predictor.")
        and not name.startswith("predictor.swept_progress_head.")
    )
    if (
        len(selected) != TWO_AXIS_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in selected)
        != TWO_AXIS_PARAMETER_COUNT
    ):
        raise RuntimeError("V22 two-axis predictor subset changed")


def _validate_two_axis_diagnostics(value: Any) -> dict[str, float | int]:
    diagnostics = _receipt_mapping(value, name="V22 two-axis diagnostics")
    if set(diagnostics) != set(TWO_AXIS_DIAGNOSTIC_NAMES):
        raise RuntimeError("V22 two-axis diagnostic fields changed")
    integer_names = {
        "scene_advantage_count",
        "action_advantage_count",
        "nonrequested_action_count_per_row",
        "action_candidate_energy_count",
        "valid_cell_count",
        "high_salience_cell_count",
        "low_salience_cell_count",
    }
    retained: dict[str, float | int] = {}
    for name, item in diagnostics.items():
        if name in integer_names:
            if type(item) is not int:
                raise TypeError(f"V22 diagnostic {name} must be an exact integer")
            retained[name] = item
        else:
            retained[name] = _finite(item, name=f"V22 diagnostic {name}")
    positive = float(retained["positive_energy_mean"])
    scene_negative = float(retained["scene_negative_energy_mean"])
    action_negative = float(retained["action_negative_energy_mean"])
    scene_sum = float(retained["scene_advantage_sum"])
    scene_mean = float(retained["scene_advantage_mean"])
    action_sum = float(retained["action_advantage_sum"])
    action_mean = float(retained["action_advantage_mean"])
    cosine = float(retained["matching_predictor_gradient_cosine"])
    if (
        positive < 0.0
        or scene_negative < 0.0
        or action_negative < 0.0
        or retained["scene_advantage_count"] != 16
        or retained["action_advantage_count"] != 16
        or retained["nonrequested_action_count_per_row"] != NONREQUESTED_ACTION_COUNT
        or retained["action_candidate_energy_count"]
        != 16 * NONREQUESTED_ACTION_COUNT
        or retained["valid_cell_count"] < 2 * SALIENCE_CELL_COUNT
        or retained["valid_cell_count"] > 64 * 64
        or retained["high_salience_cell_count"] != SALIENCE_CELL_COUNT
        or retained["low_salience_cell_count"] != SALIENCE_CELL_COUNT
        or not -1.0 <= cosine <= 1.0
        or not math.isclose(scene_mean, scene_sum / 16.0, rel_tol=2e-6, abs_tol=2e-6)
        or not math.isclose(action_mean, action_sum / 16.0, rel_tol=2e-6, abs_tol=2e-6)
        or not math.isclose(scene_mean, scene_negative - positive, rel_tol=2e-6, abs_tol=2e-6)
        or not math.isclose(action_mean, action_negative - positive, rel_tol=2e-6, abs_tol=2e-6)
    ):
        raise RuntimeError("V22 two-axis diagnostics are inconsistent")
    return retained


def validate_update_integrity_v22(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if type(update) is not int or not 1 <= update <= _engine.MAXIMUM_UPDATES:
        raise ValueError("V22 update integrity index escaped the cap")
    accounting = _receipt_mapping(result.accounting, name="V22 accounting")
    expected_accounting = _expected_accounting_v22(update)
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise RuntimeError("V22 per-update accounting changed")
    inherited_route_names = (
        "camera_shared", "joint_shared", "representation", "predictor"
    )
    routes = result.gradient_routes
    if not isinstance(routes, Mapping) or set(routes) != {
        *inherited_route_names,
        TWO_AXIS_INNOVATION_ROUTE_NAME,
    }:
        raise RuntimeError("V22 gradient-route receipt set changed")
    losses = dict(result.mean_losses)
    if set(losses) != set(MEAN_LOSS_NAMES):
        raise RuntimeError("V22 per-update mean-loss receipt set changed")
    finite_losses = {
        name: _finite(value, name=f"V22 mean loss {name}")
        for name, value in losses.items()
    }
    if (
        any(
            finite_losses[name] < 0.0
            for name in ("I_fit", "I_scene_rank", "I_action_rank")
        )
        or not math.isclose(
            finite_losses["N"],
            sum(finite_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["I_two_axis"],
            finite_losses["I_fit"]
            + 0.5 * (
                finite_losses["I_scene_rank"] + finite_losses["I_action_rank"]
            ),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["L"],
            finite_losses["N"] + finite_losses["C"] + finite_losses["I_two_axis"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V22 N/L/I loss equations changed")
    diagnostics = _validate_two_axis_diagnostics(
        result.two_axis_innovation_diagnostics
    )
    if not math.isclose(
        finite_losses["I_fit"],
        float(diagnostics["positive_energy_mean"]),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V22 I_fit disagrees with positive energy")
    innovation_route = _receipt_mapping(
        routes[TWO_AXIS_INNOVATION_ROUTE_NAME],
        name="V22 two-axis gradient receipt",
    )
    expected_route_fields = {
        "preclip_l2",
        "applied_scale",
        "parameter_tensor_count",
        "absent_tensor_gradient_count",
    }
    if set(innovation_route) != expected_route_fields:
        raise RuntimeError("V22 two-axis route fields changed")
    norm = _finite(innovation_route["preclip_l2"], name="V22 route preclip norm")
    scale = _finite(innovation_route["applied_scale"], name="V22 route scale")
    torch = runtime.torch
    expected_scale = float(
        torch.minimum(
            torch.tensor(1.0, dtype=torch.float32),
            torch.reciprocal(
                torch.maximum(
                    torch.tensor(norm, dtype=torch.float32),
                    torch.tensor(torch.finfo(torch.float32).tiny, dtype=torch.float32),
                )
            ),
        ).item()
    )
    if (
        norm <= 0.0
        or scale != expected_scale
        or innovation_route["parameter_tensor_count"] != TWO_AXIS_PARAMETER_TENSOR_COUNT
        or innovation_route["absent_tensor_gradient_count"] != 0
    ):
        raise RuntimeError("V22 two-axis gradient route failed integrity")
    _validate_two_axis_parameter_subset(model)

    projected_diagnostics = {
        "positive_energy_mean": diagnostics["positive_energy_mean"],
        "negative_energy_mean": diagnostics["scene_negative_energy_mean"],
        "advantage_sum": diagnostics["scene_advantage_sum"],
        "advantage_count": diagnostics["scene_advantage_count"],
        "advantage_mean": diagnostics["scene_advantage_mean"],
        "matching_predictor_gradient_cosine": diagnostics[
            "matching_predictor_gradient_cosine"
        ],
        "valid_cell_count": diagnostics["valid_cell_count"],
        "high_salience_cell_count": diagnostics["high_salience_cell_count"],
        "low_salience_cell_count": diagnostics["low_salience_cell_count"],
    }
    projected_losses = {
        name: finite_losses[name] for name in ("S", "P", "U", "R", "O", "N", "C", "L")
    }
    projected_losses.update(
        {
            "I_fit": finite_losses["I_fit"],
            "I_rank": 0.5 * (
                finite_losses["I_scene_rank"] + finite_losses["I_action_rank"]
            ),
            "I_scene": finite_losses["I_two_axis"],
        }
    )
    projected_result = SimpleNamespace(
        accounting=_project_accounting_to_v21(accounting),
        gradient_routes={
            **{name: routes[name] for name in inherited_route_names},
            _base.SCENE_INNOVATION_ROUTE_NAME: innovation_route,
        },
        mean_losses=projected_losses,
        scene_innovation_diagnostics=projected_diagnostics,
        ranking_active_microbatches=result.ranking_active_microbatches,
        ranking_eligible_pairs=result.ranking_eligible_pairs,
        survival_supervised_decisions=result.survival_supervised_decisions,
        target_gradient_tensor_count=result.target_gradient_tensor_count,
        optimizer_steps_this_update=result.optimizer_steps_this_update,
        ema_steps_this_update=result.ema_steps_this_update,
    )
    receipt = _original_validate_update_integrity(
        runtime,
        model,
        projected_result,
        update=update,
        access_receipt=access_receipt,
    )
    receipt["accounting"] = accounting
    receipt["gradient_routes"] = {
        **{name: receipt["gradient_routes"][name] for name in inherited_route_names},
        TWO_AXIS_INNOVATION_ROUTE_NAME: innovation_route,
    }
    receipt["mean_losses"] = finite_losses
    receipt.pop("scene_innovation_diagnostics", None)
    receipt.pop("v21_same_action_cross_scene_contrastive_innovation", None)
    receipt["two_axis_innovation_diagnostics"] = dict(diagnostics)
    receipt["v22_scene_action_contrastive_innovation"] = {
        "parameter_tensor_count": TWO_AXIS_PARAMETER_TENSOR_COUNT,
        "parameter_count": TWO_AXIS_PARAMETER_COUNT,
        "scene_negative_row_batch_key": SCENE_NEGATIVE_ROW_KEY,
        "nonrequested_action_count_per_row": NONREQUESTED_ACTION_COUNT,
        "encoder_gradient_from_i_two_axis": False,
        "representation_gradient_from_i_two_axis": False,
        "semantic_head_gradient_from_i_two_axis": False,
        "survival_head_gradient_from_i_two_axis": False,
        "target_gradient_from_i_two_axis": False,
        "passed": True,
    }
    cached = getattr(runtime, "two_axis_innovation_diagnostics_v22", None)
    if cached is None:
        cached = {}
        runtime.two_axis_innovation_diagnostics_v22 = cached
    if type(cached) is not dict or update in cached:
        raise RuntimeError("V22 two-axis diagnostic cache is not one-shot")
    cached[update] = dict(diagnostics)
    return receipt


def observation_v22(
    runtime: Any,
    model: Any,
    *,
    update: int,
    integrity_pass: bool,
) -> dict[str, Any]:
    observed = _original_observation(
        runtime,
        model,
        update=update,
        integrity_pass=integrity_pass,
    )
    observed.pop("scene_innovation_diagnostics", None)
    if update == 0:
        observed["two_axis_innovation_diagnostics"] = None
        return observed
    cached = getattr(runtime, "two_axis_innovation_diagnostics_v22", None)
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V22 observation lacks current-update diagnostics")
    observed["two_axis_innovation_diagnostics"] = dict(cached[update])
    return observed


def validate_terminal_accounting_v22(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    if terminal_update not in _engine.TERMINAL_UPDATES:
        raise ValueError("V22 terminal update must be exactly 400 or 1000")
    value = _receipt_mapping(accounting, name="V22 terminal accounting")
    expected = _expected_accounting_v22(terminal_update)
    if (
        set(value) != set(expected)
        or any(type(item) is not int for item in value.values())
        or value != expected
    ):
        raise RuntimeError("V22 terminal accounting is inconsistent with the cap")
    _original_validate_terminal_accounting(
        _project_accounting_to_v21(value),
        terminal_update=terminal_update,
    )
    return expected


_engine.validate_bound_sources_v13 = validate_bound_sources_v22
_engine.validate_model_api_v13 = validate_model_api_v22
_engine.validate_training_api_v13 = validate_training_api_v22
_engine._validate_microbatches_for_engine_v13 = validate_microbatches_for_engine_v22
_engine._validate_update_integrity_v13 = validate_update_integrity_v22
_engine._observation_v13 = observation_v22
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v22

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = _base.TRAINING_REQUIRED_BATCH_KEYS
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
MATCHED_UPDATE400_THRESHOLDS = _engine.MATCHED_UPDATE400_THRESHOLDS
TRACE_RELATIVE_PATH = _engine.TRACE_RELATIVE_PATH
METRIC_RELATIVE_PATHS = _engine.METRIC_RELATIVE_PATHS
DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = _engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH

validate_content_bound_v22 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v22 = _engine.validate_future_execution_prerequisites_v13
execution_denial_receipt_v22 = _engine.execution_denial_receipt_v13
reserve_attempt_v22 = _engine.reserve_attempt_v13
terminalize_failure_v22 = _engine.terminalize_failure_v13
flatten_physical_metrics_v22 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v22 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v22 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v22 = _engine.evaluate_final_gate_v13
validate_schedule_v22 = _engine.validate_schedule_v13
validate_attempt_reservation_v22 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v22 = _engine.run_future_authorized_engine_v13
execute_v22 = _engine.execute_v13

# Compatibility names consumed inside the unchanged inherited lifecycle.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v22
validate_bound_sources_v13 = validate_bound_sources_v22
validate_model_api_v13 = validate_model_api_v22
validate_training_api_v13 = validate_training_api_v22
validate_future_execution_prerequisites_v13 = validate_future_execution_prerequisites_v22
execution_denial_receipt_v13 = execution_denial_receipt_v22
reserve_attempt_v13 = reserve_attempt_v22
terminalize_failure_v13 = terminalize_failure_v22
flatten_physical_metrics_v13 = flatten_physical_metrics_v22
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v22
evaluate_update400_gate_v13 = evaluate_update400_gate_v22
evaluate_final_gate_v13 = evaluate_final_gate_v22
validate_schedule_v13 = validate_schedule_v22
validate_attempt_reservation_v13 = validate_attempt_reservation_v22
run_future_authorized_engine_v13 = run_future_authorized_engine_v22
validate_terminal_accounting_v13 = validate_terminal_accounting_v22
execute_v13 = execute_v22


def private_adapter_receipt_v22() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v21_executor_adapter_v1",
        "base_executor": str(V21_EXECUTOR_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": V21_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_executor_file_sha256": V21_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V21_EXECUTOR_BYTE_COUNT,
        "public_v21_was_loaded_before_adapter": _PUBLIC_V21_WAS_LOADED_BEFORE_ADAPTER,
        "public_v21_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V21_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v21_scientific_result_commit": V21_SCIENTIFIC_RESULT_COMMIT,
        "v21_scientific_result_content_sha256": V21_SCIENTIFIC_RESULT_CONTENT_SHA256,
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "accounting_multipliers": dict(ACCOUNTING_MULTIPLIERS_V22),
        "two_axis_parameter_tensor_count": TWO_AXIS_PARAMETER_TENSOR_COUNT,
        "two_axis_parameter_count": TWO_AXIS_PARAMETER_COUNT,
        "scene_negative_row_batch_key": SCENE_NEGATIVE_ROW_KEY,
        "nonrequested_action_count": NONREQUESTED_ACTION_COUNT,
        "new_batch_fields_over_v21": 0,
        "update100_informational": True,
        "v21_update400_and_update1000_gates_unchanged": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V22 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v22(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
