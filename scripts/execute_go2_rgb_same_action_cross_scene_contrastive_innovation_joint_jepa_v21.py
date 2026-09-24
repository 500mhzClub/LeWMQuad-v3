#!/usr/bin/env python3
"""Denied-by-default executor adapter for V21 scene innovation.

The frozen V20 executor is loaded into a private namespace so its accounting
isolation, evaluator, gates, custody lifecycle, and one-shot publisher remain
unchanged.  This adapter replaces only V20's factual-successor receipt surface
with the preregistered V21 scene-innovation surface and admits the one reviewed
train-metadata batch field.  This source shell grants no execution authority.
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
V20_EXECUTOR_PATH = (
    ROOT
    / "scripts/execute_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19.py"
)
V20_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "1692c6029d9e772ad2a7d65447ad70fc634a7afc"
)
V20_EXECUTOR_FILE_SHA256 = (
    "dc4f2504424b55f0c8fadee213c5a88016dea41e1c5b14adb9bb147c4b877b2e"
)
V20_EXECUTOR_BYTE_COUNT = 33_138
V20_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
PRIVATE_V20_MODULE_NAME = f"{__name__}.__private_v20_executor"
_PUBLIC_V20_WAS_LOADED_BEFORE_ADAPTER = V20_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21"
)
PREREGISTRATION_COMMIT = "c2bbce067175dd980c9ed2511dc14db5a222afe4"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "f4ff1453e5cb63677dad66253d568c9204bd5504b3b3871e2b0c341402b1850e"
)
PREREGISTRATION_BYTE_COUNT = 11_594
V20_SCIENTIFIC_RESULT_COMMIT = "8321d76004aa1f3c87dfa04c3b18d701267a89ec"
V20_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json"
)
V20_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956"
)
V20_SCIENTIFIC_RESULT_BYTE_COUNT = 17_166
V20_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "37f683c1b2a5086c92d9cb081e9ba55b4fef4ed61f8cefea99fb0e5760e5cab2"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = "scene_innovation_negative_row"
SCENE_INNOVATION_ROUTE_NAME = "scene_innovation_predictor"
SCENE_INNOVATION_PARAMETER_TENSOR_COUNT = 13
SCENE_INNOVATION_PARAMETER_COUNT = 259_008
SCENE_INNOVATION_GRADIENT_NORM_CAP = 1.0
SALIENCE_CELL_COUNT = 128
MEAN_LOSS_NAMES = (
    "S",
    "P",
    "U",
    "R",
    "O",
    "I_fit",
    "I_rank",
    "I_scene",
    "N",
    "C",
    "L",
)
SCENE_INNOVATION_DIAGNOSTIC_NAMES = (
    "positive_energy_mean",
    "negative_energy_mean",
    "advantage_sum",
    "advantage_count",
    "advantage_mean",
    "matching_predictor_gradient_cosine",
    "valid_cell_count",
    "high_salience_cell_count",
    "low_salience_cell_count",
)
ACCOUNTING_MULTIPLIERS_V21 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 12,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "scene_innovation_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 8,
    "scene_innovation_objectives": 4,
}


def _load_private_v20_executor() -> ModuleType:
    if V20_EXECUTOR_PATH.is_symlink() or not V20_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("frozen V20 executor source is absent or not regular")
    source = V20_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V20_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V20_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V20 executor source binding changed")
    if PRIVATE_V20_MODULE_NAME in sys.modules:
        raise RuntimeError("private V20 executor module name is already occupied")
    module = ModuleType(PRIVATE_V20_MODULE_NAME)
    module.__file__ = str(V20_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V20_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V20_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V20_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V20_MODULE_NAME)
    return module


_base = _load_private_v20_executor()
if (
    _base.SCHEMA_PREFIX
    != "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
    "grounding_joint_jepa_v20"
    or _base.MODEL_CLASS_NAME != MODEL_CLASS_NAME
    or _base.MAXIMUM_UPDATES != 1_000
    or _base.MAXIMUM_PRESENTATIONS != 16_000
    or _base.OBSERVATION_UPDATES != (0, 100, 400, 1_000)
    or _base.TERMINAL_UPDATES != (400, 1_000)
    or _base.CURRENT_EXECUTION_AUTHORIZED is not False
):
    raise RuntimeError("frozen V20 executor defaults changed")

_engine = _base._engine
_original_validate_bound_sources = _engine.validate_bound_sources_v13
_original_validate_model_api = _engine.validate_model_api_v13
_original_validate_training_api = _engine.validate_training_api_v13
_original_validate_update_integrity = _engine._validate_update_integrity_v13
_original_observation = _engine._observation_v13
_original_validate_microbatches_for_engine = (
    _engine._validate_microbatches_for_engine_v13
)
_INHERITED_TRAINING_REQUIRED_BATCH_KEYS = tuple(
    _engine.TRAINING_REQUIRED_BATCH_KEYS
)
TRAINING_REQUIRED_BATCH_KEYS_V21 = (
    *_INHERITED_TRAINING_REQUIRED_BATCH_KEYS,
    SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
)

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V20_SCIENTIFIC_RESULT_PATH: (
            V20_SCIENTIFIC_RESULT_FILE_SHA256,
            V20_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
)

# V21 lifecycle identities change only the private V20 adapter and its private
# engine.  The inherited batch-key registry intentionally remains untouched;
# the one-field extension is validated by the localized V21 hook below.
_base.SCHEMA_PREFIX = SCHEMA_PREFIX
_base.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_base.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_base.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_base.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V21 scene-innovation scientific execution is denied until recursive "
    "source closure, independent exact-binding review, narrow clean-export "
    "certification, and one-shot authority are committed and validated"
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


def _expected_accounting_v21(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V21.items()
    }


def _project_accounting_to_v20(accounting: Mapping[str, int]) -> dict[str, int]:
    projected = dict(accounting)
    projected["factual_successor_grad_calls"] = projected.pop(
        "scene_innovation_grad_calls"
    )
    projected["factual_successor_objectives"] = projected.pop(
        "scene_innovation_objectives"
    )
    if set(projected) != set(_base.ACCOUNTING_MULTIPLIERS_V19):
        raise RuntimeError("V21-to-V20 accounting projection changed")
    return projected


def validate_bound_sources_v21(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_model_api_v21(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V21 did not retain the exact V18 model class")
    return receipt


def validate_training_api_v21(module: Any) -> dict[str, Any]:
    receipt = dict(_original_validate_training_api(module))
    for name in ("JointTrainingAccountingV21", "JointUpdateResultV21"):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V21 training type is absent: {name}")
    for name in ("joint_training_update_v21", "validate_accounting_v21"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V21 training callable is absent: {name}")
    if (
        getattr(module, "SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21", None)
        != SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
        or tuple(getattr(module, "REQUIRED_BATCH_KEYS_V21", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V21
    ):
        raise RuntimeError("V21 one-field training batch extension changed")
    receipt.pop("factual_successor_route", None)
    receipt.pop("factual_successor_objectives_per_update", None)
    return {
        **receipt,
        "required_batch_key_count_v21": len(TRAINING_REQUIRED_BATCH_KEYS_V21),
        "scene_innovation_route": SCENE_INNOVATION_ROUTE_NAME,
        "backward_calls_per_update": 12,
        "predictor_objectives_per_update": 8,
        "scene_innovation_objectives_per_update": 4,
    }


def validate_microbatches_for_engine_v21(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != _engine.MICROBATCHES_PER_UPDATE:
        raise PermissionError("V21 engine did not receive exactly four microbatches")
    for batch in microbatches:
        if type(batch) is not dict or tuple(batch) != TRAINING_REQUIRED_BATCH_KEYS_V21:
            raise PermissionError("V21 engine microbatch schema changed")
        _engine._validate_batch_query_identity_v13(model, batch)
    validator = getattr(runtime.training_module, "_validate_microbatches_v21", None)
    if not callable(validator):
        raise RuntimeError("V21 training microbatch validator is absent")
    validator(runtime.torch, microbatches)


def _validate_scene_innovation_parameter_subset(model: Any) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("V21 model named-parameter API is absent")
    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters()
        if name.startswith("predictor.")
        and not name.startswith("predictor.swept_progress_head.")
    )
    if (
        len(selected) != SCENE_INNOVATION_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in selected)
        != SCENE_INNOVATION_PARAMETER_COUNT
    ):
        raise RuntimeError("V21 scene-innovation predictor subset changed")


def _validate_scene_diagnostics(value: Any) -> dict[str, float | int]:
    diagnostics = _receipt_mapping(value, name="V21 scene-innovation diagnostics")
    if set(diagnostics) != set(SCENE_INNOVATION_DIAGNOSTIC_NAMES):
        raise RuntimeError("V21 scene-innovation diagnostic fields changed")
    integer_names = {
        "advantage_count",
        "valid_cell_count",
        "high_salience_cell_count",
        "low_salience_cell_count",
    }
    retained: dict[str, float | int] = {}
    for name, value_item in diagnostics.items():
        if name in integer_names:
            if type(value_item) is not int:
                raise TypeError(f"V21 diagnostic {name} must be an exact integer")
            retained[name] = value_item
        else:
            retained[name] = _finite(value_item, name=f"V21 diagnostic {name}")
    positive = float(retained["positive_energy_mean"])
    negative = float(retained["negative_energy_mean"])
    advantage_sum = float(retained["advantage_sum"])
    advantage_mean = float(retained["advantage_mean"])
    cosine = float(retained["matching_predictor_gradient_cosine"])
    if (
        positive < 0.0
        or negative < 0.0
        or retained["advantage_count"] != 16
        or retained["valid_cell_count"] < 2 * SALIENCE_CELL_COUNT
        or retained["valid_cell_count"] > 64 * 64
        or retained["high_salience_cell_count"] != SALIENCE_CELL_COUNT
        or retained["low_salience_cell_count"] != SALIENCE_CELL_COUNT
        or not -1.0 <= cosine <= 1.0
        or not math.isclose(
            advantage_mean,
            advantage_sum / 16.0,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            advantage_mean,
            negative - positive,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V21 scene-innovation diagnostics are inconsistent")
    return retained


def validate_update_integrity_v21(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if type(update) is not int or not 1 <= update <= _engine.MAXIMUM_UPDATES:
        raise ValueError("V21 update integrity index escaped the cap")
    accounting = _receipt_mapping(result.accounting, name="V21 accounting")
    expected_accounting = _expected_accounting_v21(update)
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise RuntimeError("V21 per-update accounting changed")

    inherited_route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
    )
    routes = result.gradient_routes
    if not isinstance(routes, Mapping) or set(routes) != {
        *inherited_route_names,
        SCENE_INNOVATION_ROUTE_NAME,
    }:
        raise RuntimeError("V21 gradient-route receipt set changed")

    losses = dict(result.mean_losses)
    if set(losses) != set(MEAN_LOSS_NAMES):
        raise RuntimeError("V21 per-update mean-loss receipt set changed")
    finite_losses = {
        name: _finite(value, name=f"V21 mean loss {name}")
        for name, value in losses.items()
    }
    if (
        finite_losses["I_fit"] < 0.0
        or finite_losses["I_rank"] < 0.0
        or not math.isclose(
            finite_losses["N"],
            sum(finite_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["I_scene"],
            finite_losses["I_fit"] + finite_losses["I_rank"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["L"],
            finite_losses["N"] + finite_losses["C"] + finite_losses["I_scene"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V21 N/L/I loss equations changed")

    diagnostics = _validate_scene_diagnostics(result.scene_innovation_diagnostics)
    if not math.isclose(
        finite_losses["I_fit"],
        float(diagnostics["positive_energy_mean"]),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V21 I_fit disagrees with positive energy")

    innovation_route = _receipt_mapping(
        routes[SCENE_INNOVATION_ROUTE_NAME],
        name="V21 scene-innovation gradient receipt",
    )
    expected_route_fields = {
        "preclip_l2",
        "applied_scale",
        "parameter_tensor_count",
        "absent_tensor_gradient_count",
    }
    if set(innovation_route) != expected_route_fields:
        raise RuntimeError("V21 scene-innovation route fields changed")
    norm = _finite(innovation_route["preclip_l2"], name="V21 route preclip norm")
    scale = _finite(innovation_route["applied_scale"], name="V21 route scale")
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
        or innovation_route["parameter_tensor_count"]
        != SCENE_INNOVATION_PARAMETER_TENSOR_COUNT
        or innovation_route["absent_tensor_gradient_count"] != 0
    ):
        raise RuntimeError("V21 scene-innovation gradient route failed integrity")
    _validate_scene_innovation_parameter_subset(model)

    # The projection crosses the already-reviewed V20 accounting-isolation and
    # inherited V18/V13 integrity boundary.  Projected semantic diagnostics are
    # structural placeholders only and are never published as V21 evidence.
    projected_losses = {
        name: finite_losses[name] for name in ("S", "P", "U", "R", "O", "N", "C")
    }
    projected_losses["Q"] = finite_losses["I_scene"]
    projected_losses["L"] = finite_losses["L"]
    projected_result = SimpleNamespace(
        accounting=_project_accounting_to_v20(accounting),
        gradient_routes={
            **{name: routes[name] for name in inherited_route_names},
            _base.FACTUAL_SUCCESSOR_ROUTE_NAME: innovation_route,
        },
        mean_losses=projected_losses,
        factual_successor_diagnostics={
            "successor_semantic_nll_normalized": finite_losses["I_scene"],
            "persistence_semantic_nll_normalized": finite_losses["I_scene"],
            "successor_minus_persistence_nll_normalized": 0.0,
            "changed_cell_fraction": 0.0,
            "non_hold_row_count": 0,
            "matching_predictor_gradient_cosine": diagnostics[
                "matching_predictor_gradient_cosine"
            ],
        },
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
        SCENE_INNOVATION_ROUTE_NAME: innovation_route,
    }
    receipt["mean_losses"] = finite_losses
    receipt.pop("factual_successor_diagnostics", None)
    receipt.pop("v19_executed_successor_semantic_grounding", None)
    receipt["scene_innovation_diagnostics"] = dict(diagnostics)
    receipt["v21_same_action_cross_scene_contrastive_innovation"] = {
        "parameter_tensor_count": SCENE_INNOVATION_PARAMETER_TENSOR_COUNT,
        "parameter_count": SCENE_INNOVATION_PARAMETER_COUNT,
        "negative_row_batch_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "representation_gradient_from_i_scene": False,
        "semantic_head_gradient_from_i_scene": False,
        "target_gradient_from_i_scene": False,
        "passed": True,
    }

    cached = getattr(runtime, "scene_innovation_diagnostics_v21", None)
    if cached is None:
        cached = {}
        runtime.scene_innovation_diagnostics_v21 = cached
    if type(cached) is not dict or update in cached:
        raise RuntimeError("V21 scene-innovation diagnostic cache is not one-shot")
    cached[update] = dict(diagnostics)
    return receipt


def observation_v21(
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
    if update == 0:
        return observed
    cached = getattr(runtime, "scene_innovation_diagnostics_v21", None)
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V21 observation lacks current-update scene diagnostics")
    if "scene_innovation_diagnostics" in observed:
        raise RuntimeError("V21 inherited observation supplied scene diagnostics")
    return {
        **observed,
        "scene_innovation_diagnostics": dict(cached[update]),
    }


def validate_terminal_accounting_v21(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    if terminal_update not in _engine.TERMINAL_UPDATES:
        raise ValueError("V21 terminal update must be exactly 400 or 1000")
    value = _receipt_mapping(accounting, name="V21 terminal accounting")
    expected = _expected_accounting_v21(terminal_update)
    if (
        set(value) != set(expected)
        or any(type(item) is not int for item in value.values())
        or value != expected
    ):
        raise RuntimeError("V21 terminal accounting is inconsistent with the frozen cap")
    if (
        value["updates"] > _engine.MAXIMUM_UPDATES
        or value["presentations"] > _engine.MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V21 scientific cap was exceeded")
    return expected


_engine.validate_bound_sources_v13 = validate_bound_sources_v21
_engine.validate_model_api_v13 = validate_model_api_v21
_engine.validate_training_api_v13 = validate_training_api_v21
_engine._validate_microbatches_for_engine_v13 = validate_microbatches_for_engine_v21
_engine._validate_update_integrity_v13 = validate_update_integrity_v21
_engine._observation_v13 = observation_v21
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v21

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = _INHERITED_TRAINING_REQUIRED_BATCH_KEYS
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
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
)
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = (
    _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
)
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH

validate_content_bound_v21 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v21 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v21 = _engine.execution_denial_receipt_v13
reserve_attempt_v21 = _engine.reserve_attempt_v13
terminalize_failure_v21 = _engine.terminalize_failure_v13
flatten_physical_metrics_v21 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v21 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v21 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v21 = _engine.evaluate_final_gate_v13
validate_schedule_v21 = _engine.validate_schedule_v13
validate_attempt_reservation_v21 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v21 = _engine.run_future_authorized_engine_v13
execute_v21 = _engine.execute_v13

# Compatibility names resolved inside the unchanged private lifecycle.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v21
validate_bound_sources_v13 = validate_bound_sources_v21
validate_model_api_v13 = validate_model_api_v21
validate_training_api_v13 = validate_training_api_v21
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v21
)
execution_denial_receipt_v13 = execution_denial_receipt_v21
reserve_attempt_v13 = reserve_attempt_v21
terminalize_failure_v13 = terminalize_failure_v21
flatten_physical_metrics_v13 = flatten_physical_metrics_v21
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v21
evaluate_update400_gate_v13 = evaluate_update400_gate_v21
evaluate_final_gate_v13 = evaluate_final_gate_v21
validate_schedule_v13 = validate_schedule_v21
validate_attempt_reservation_v13 = validate_attempt_reservation_v21
run_future_authorized_engine_v13 = run_future_authorized_engine_v21
validate_terminal_accounting_v13 = validate_terminal_accounting_v21
execute_v13 = execute_v21


def private_adapter_receipt_v21() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v20_executor_adapter_v1",
        "base_executor": str(V20_EXECUTOR_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": (
            V20_FROZEN_SOURCE_AND_REVIEW_COMMIT
        ),
        "base_executor_file_sha256": V20_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V20_EXECUTOR_BYTE_COUNT,
        "public_v20_was_loaded_before_adapter": (
            _PUBLIC_V20_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_v20_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V20_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v20_scientific_result_commit": V20_SCIENTIFIC_RESULT_COMMIT,
        "v20_scientific_result_content_sha256": (
            V20_SCIENTIFIC_RESULT_CONTENT_SHA256
        ),
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "accounting_multipliers": dict(ACCOUNTING_MULTIPLIERS_V21),
        "scene_innovation_parameter_tensor_count": (
            SCENE_INNOVATION_PARAMETER_TENSOR_COUNT
        ),
        "scene_innovation_parameter_count": SCENE_INNOVATION_PARAMETER_COUNT,
        "negative_row_batch_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "inherited_batch_registry_unchanged": (
            tuple(_engine.TRAINING_REQUIRED_BATCH_KEYS)
            == _INHERITED_TRAINING_REQUIRED_BATCH_KEYS
        ),
        "update100_informational": True,
        "v20_update400_and_update1000_gates_unchanged": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V21 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v21(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
