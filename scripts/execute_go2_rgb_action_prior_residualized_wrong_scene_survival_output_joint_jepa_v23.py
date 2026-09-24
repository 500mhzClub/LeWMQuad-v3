#!/usr/bin/env python3
"""Denied-by-default V23 executor adapter over the frozen V21 lifecycle.

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
    "lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23"
)
PREREGISTRATION_COMMIT = "a7cf9692dd93212a82cb598d3175ff1c3598941b"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "d5702759866138db1467778553ef8494d05f4593fcca14822050b1e0991180ae"
)
PREREGISTRATION_BYTE_COUNT = 14_294
V22_SCIENTIFIC_RESULT_COMMIT = "f184a41ac99b1c66ea4db1e0b0a0845f23b48bbd"
V22_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_"
    "scientific_result_2026-07-30.json"
)
V22_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0"
)
V22_SCIENTIFIC_RESULT_BYTE_COUNT = 18_445
V22_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "d9c0376f381bb65c4246c9ff12611f4b563698a0539f81c63b95e8b083de18a2"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
SCENE_NEGATIVE_ROW_KEY = "scene_innovation_negative_row"
ACTION_PRIOR_M_KEY = "train_action_prior_m"
STATE_RESIDUAL_SURVIVAL_ROUTE_NAME = "state_residual_survival_output"
STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT = 109
STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT = 3_365_417
NON_HOLD_ACTION_COUNT = 8
MEAN_LOSS_NAMES = (
    "S", "P", "U", "R", "O", "F", "J_rank", "J23", "N", "C", "L",
)
STATE_RESIDUAL_SURVIVAL_DIAGNOSTIC_NAMES = (
    "positive_energy_sum",
    "positive_energy_count",
    "positive_energy_mean",
    "scene_negative_energy_sum",
    "scene_eligible_count",
    "scene_negative_energy_mean",
    "scene_advantage_sum",
    "scene_advantage_mean",
    "scene_rank_sum",
    "prior_negative_energy_sum",
    "prior_eligible_count",
    "prior_negative_energy_mean",
    "prior_advantage_sum",
    "prior_advantage_mean",
    "prior_rank_sum",
    "non_hold_action_count_per_row",
)
ACCOUNTING_MULTIPLIERS_V23 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 12,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "state_residual_survival_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 8,
    "state_residual_survival_objectives": 4,
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
TRAINING_REQUIRED_BATCH_KEYS_V21 = tuple(_base.TRAINING_REQUIRED_BATCH_KEYS_V21)
TRAINING_REQUIRED_BATCH_KEYS_V23 = (
    *TRAINING_REQUIRED_BATCH_KEYS_V21,
    ACTION_PRIOR_M_KEY,
)

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V22_SCIENTIFIC_RESULT_PATH: (
            V22_SCIENTIFIC_RESULT_FILE_SHA256,
            V22_SCIENTIFIC_RESULT_BYTE_COUNT,
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
    "V23 action-prior-residualized wrong-scene survival-output innovation execution is denied until recursive source "
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


def _expected_accounting_v23(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V23.items()
    }


def _project_accounting_to_v21(accounting: Mapping[str, int]) -> dict[str, int]:
    projected = dict(accounting)
    projected["scene_innovation_grad_calls"] = projected.pop(
        "state_residual_survival_grad_calls"
    )
    projected["scene_innovation_objectives"] = projected.pop(
        "state_residual_survival_objectives"
    )
    if set(projected) != set(_base.ACCOUNTING_MULTIPLIERS_V21):
        raise RuntimeError("V23-to-V21 accounting projection changed")
    return projected


def validate_bound_sources_v23(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_model_api_v23(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V23 did not retain the exact V18 model class")
    return receipt


def validate_training_api_v23(module: Any) -> dict[str, Any]:
    receipt = dict(_original_validate_training_api(module))
    for name in ("JointTrainingAccountingV23", "JointUpdateResultV23"):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V23 training type is absent: {name}")
    for name in ("joint_training_update_v23", "validate_accounting_v23"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V23 training callable is absent: {name}")
    if (
        tuple(getattr(module, "REQUIRED_BATCH_KEYS_V23", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V23
        or tuple(getattr(module, "REQUIRED_BATCH_KEYS_V21", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V21
        or getattr(module, "SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21", None)
        != SCENE_NEGATIVE_ROW_KEY
        or getattr(module, "ACTION_PRIOR_M_KEY_V23", None) != ACTION_PRIOR_M_KEY
    ):
        raise RuntimeError("V23 one-field batch extension changed")
    receipt.pop("scene_innovation_route", None)
    receipt.pop("scene_innovation_objectives_per_update", None)
    return {
        **receipt,
        "required_batch_key_count_v23": len(TRAINING_REQUIRED_BATCH_KEYS_V23),
        "state_residual_survival_route": STATE_RESIDUAL_SURVIVAL_ROUTE_NAME,
        "backward_calls_per_update": 12,
        "predictor_objectives_per_update": 8,
        "state_residual_survival_objectives_per_update": 4,
        "new_batch_fields_over_v21": 1,
    }


def validate_microbatches_for_engine_v23(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != _engine.MICROBATCHES_PER_UPDATE or any(
        type(batch) is not dict or tuple(batch) != TRAINING_REQUIRED_BATCH_KEYS_V23
        for batch in microbatches
    ):
        raise PermissionError("V23 engine microbatch schema changed")
    projected = tuple(
        {name: batch[name] for name in TRAINING_REQUIRED_BATCH_KEYS_V21}
        for batch in microbatches
    )
    _original_validate_microbatches(runtime, model, projected)
    torch = runtime.torch
    for batch in microbatches:
        prior = batch[ACTION_PRIOR_M_KEY]
        if (
            not isinstance(prior, torch.Tensor)
            or tuple(prior.shape) != (9,)
            or prior.dtype != torch.float32
            or prior.requires_grad
            or prior.device != batch[TRAINING_REQUIRED_BATCH_KEYS_V21[0]].device
            or not bool(torch.isfinite(prior).all().item())
        ):
            raise PermissionError("V23 action-prior batch tensor changed")


def _validate_state_residual_survival_parameter_subset(model: Any) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("V23 model named-parameter API is absent")
    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters()
        if name.startswith(
            (
                "encoder.",
                "bev_lift.evidence_head.",
                "bev_lift.point_projection.",
                "bev_lift.volume_block.",
                "predictor.",
            )
        )
    )
    if (
        len(selected) != STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in selected)
        != STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT
        or any(name.startswith("semantic_head.") for name, _ in selected)
        or "predictor.swept_progress_head.output.weight"
        not in {name for name, _ in selected}
        or "predictor.swept_progress_head.output.bias"
        not in {name for name, _ in selected}
    ):
        raise RuntimeError("V23 full survival-output parameter subset changed")


def _validate_state_residual_survival_diagnostics(
    value: Any,
) -> dict[str, float | int]:
    diagnostics = _receipt_mapping(value, name="V23 output diagnostics")
    if set(diagnostics) != set(STATE_RESIDUAL_SURVIVAL_DIAGNOSTIC_NAMES):
        raise RuntimeError("V23 output diagnostic fields changed")
    integer_names = {
        "positive_energy_count",
        "scene_eligible_count",
        "prior_eligible_count",
        "non_hold_action_count_per_row",
    }
    retained: dict[str, float | int] = {}
    for name, item in diagnostics.items():
        if name in integer_names:
            if type(item) is not int:
                raise TypeError(f"V23 diagnostic {name} must be an exact integer")
            retained[name] = item
        else:
            retained[name] = _finite(item, name=f"V23 diagnostic {name}")
    positive_count = int(retained["positive_energy_count"])
    scene_count = int(retained["scene_eligible_count"])
    prior_count = int(retained["prior_eligible_count"])
    if (
        positive_count != 16 * NON_HOLD_ACTION_COUNT
        or not 4 <= scene_count <= positive_count
        or not 4 <= prior_count <= positive_count
        or retained["non_hold_action_count_per_row"] != NON_HOLD_ACTION_COUNT
        or any(
            float(retained[name]) < 0.0
            for name in (
                "positive_energy_sum",
                "positive_energy_mean",
                "scene_negative_energy_sum",
                "scene_negative_energy_mean",
                "scene_rank_sum",
                "prior_negative_energy_sum",
                "prior_negative_energy_mean",
                "prior_rank_sum",
            )
        )
        or not math.isclose(
            float(retained["positive_energy_mean"]),
            float(retained["positive_energy_sum"]) / positive_count,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            float(retained["scene_negative_energy_mean"]),
            float(retained["scene_negative_energy_sum"]) / scene_count,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            float(retained["scene_advantage_mean"]),
            float(retained["scene_advantage_sum"]) / scene_count,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            float(retained["prior_negative_energy_mean"]),
            float(retained["prior_negative_energy_sum"]) / prior_count,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            float(retained["prior_advantage_mean"]),
            float(retained["prior_advantage_sum"]) / prior_count,
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V23 output diagnostics are inconsistent")
    return retained



def validate_update_integrity_v23(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate V23 exactly, then project only inherited fields to V21."""

    if type(update) is not int or not 1 <= update <= _engine.MAXIMUM_UPDATES:
        raise ValueError("V23 update integrity index escaped the cap")
    accounting = _receipt_mapping(result.accounting, name="V23 accounting")
    expected_accounting = _expected_accounting_v23(update)
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise RuntimeError("V23 per-update accounting changed")
    inherited_route_names = (
        "camera_shared", "joint_shared", "representation", "predictor"
    )
    routes = result.gradient_routes
    if not isinstance(routes, Mapping) or set(routes) != {
        *inherited_route_names,
        STATE_RESIDUAL_SURVIVAL_ROUTE_NAME,
    }:
        raise RuntimeError("V23 gradient-route receipt set changed")
    losses = dict(result.mean_losses)
    if set(losses) != set(MEAN_LOSS_NAMES):
        raise RuntimeError("V23 per-update mean-loss receipt set changed")
    finite_losses = {
        name: _finite(value, name=f"V23 mean loss {name}")
        for name, value in losses.items()
    }
    if (
        any(finite_losses[name] < 0.0 for name in ("F", "J_rank", "J23"))
        or not math.isclose(
            finite_losses["N"],
            sum(finite_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["J23"],
            finite_losses["F"] + finite_losses["J_rank"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["L"],
            finite_losses["N"] + finite_losses["C"] + finite_losses["J23"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V23 N/L/J23 loss equations changed")
    diagnostics = _validate_state_residual_survival_diagnostics(
        result.state_residual_survival_diagnostics
    )
    if not math.isclose(
        finite_losses["F"],
        float(diagnostics["positive_energy_mean"]),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V23 F disagrees with positive energy")

    auxiliary_route = _receipt_mapping(
        routes[STATE_RESIDUAL_SURVIVAL_ROUTE_NAME],
        name="V23 state-residual survival gradient receipt",
    )
    expected_route_fields = {
        "preclip_l2",
        "applied_scale",
        "parameter_tensor_count",
        "absent_tensor_gradient_count",
    }
    if set(auxiliary_route) != expected_route_fields:
        raise RuntimeError("V23 output route fields changed")
    norm = _finite(auxiliary_route["preclip_l2"], name="V23 route preclip norm")
    scale = _finite(auxiliary_route["applied_scale"], name="V23 route scale")
    torch = runtime.torch
    expected_scale = float(
        torch.minimum(
            torch.tensor(1.0, dtype=torch.float32),
            torch.reciprocal(
                torch.maximum(
                    torch.tensor(norm, dtype=torch.float32),
                    torch.tensor(
                        torch.finfo(torch.float32).tiny, dtype=torch.float32
                    ),
                )
            ),
        ).item()
    )
    if (
        norm <= 0.0
        or scale != expected_scale
        or auxiliary_route["parameter_tensor_count"]
        != STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT
        or auxiliary_route["absent_tensor_gradient_count"] != 0
    ):
        raise RuntimeError("V23 output gradient route failed integrity")
    _validate_state_residual_survival_parameter_subset(model)

    # The inherited V21 validator owns the unchanged optimizer/EMA/access
    # checks.  Give it a compatibility view of the extra route; the exact V23
    # route and diagnostics were already validated above and replace this view
    # in the returned receipt.
    projected_route = {
        **auxiliary_route,
        "parameter_tensor_count": _base.SCENE_INNOVATION_PARAMETER_TENSOR_COUNT,
    }
    # V21's scene diagnostics are a private structural compatibility surface,
    # not V23 evidence.  Use an internally exact neutral placeholder; the
    # actual eligible-scene sums, counts, means, and ranks were validated above
    # and replace this projection in the returned receipt.
    projected_diagnostics = {
        "positive_energy_mean": finite_losses["F"],
        "negative_energy_mean": finite_losses["F"],
        "advantage_sum": 0.0,
        "advantage_count": 16,
        "advantage_mean": 0.0,
        "matching_predictor_gradient_cosine": 0.0,
        "valid_cell_count": 256,
        "high_salience_cell_count": 128,
        "low_salience_cell_count": 128,
    }
    projected_losses = {
        name: finite_losses[name] for name in ("S", "P", "U", "R", "O", "N", "C", "L")
    }
    projected_losses.update(
        {
            "I_fit": finite_losses["F"],
            "I_rank": finite_losses["J_rank"],
            "I_scene": finite_losses["J23"],
        }
    )
    projected_result = SimpleNamespace(
        accounting=_project_accounting_to_v21(accounting),
        gradient_routes={
            **{name: routes[name] for name in inherited_route_names},
            _base.SCENE_INNOVATION_ROUTE_NAME: projected_route,
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
        STATE_RESIDUAL_SURVIVAL_ROUTE_NAME: auxiliary_route,
    }
    receipt["mean_losses"] = finite_losses
    receipt.pop("scene_innovation_diagnostics", None)
    receipt.pop("v21_same_action_cross_scene_contrastive_innovation", None)
    receipt["state_residual_survival_diagnostics"] = dict(diagnostics)
    receipt["v23_action_prior_residualized_wrong_scene_survival_output"] = {
        "parameter_tensor_count": STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT,
        "parameter_count": STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT,
        "scene_negative_row_batch_key": SCENE_NEGATIVE_ROW_KEY,
        "action_prior_batch_key": ACTION_PRIOR_M_KEY,
        "non_hold_action_count_per_row": NON_HOLD_ACTION_COUNT,
        "encoder_gradient_from_j23": True,
        "representation_gradient_from_j23": True,
        "semantic_head_gradient_from_j23": False,
        "survival_head_gradient_from_j23": True,
        "target_gradient_from_j23": False,
        "passed": True,
    }
    cached = getattr(runtime, "state_residual_survival_diagnostics_v23", None)
    if cached is None:
        cached = {}
        runtime.state_residual_survival_diagnostics_v23 = cached
    if type(cached) is not dict or update in cached:
        raise RuntimeError("V23 output diagnostic cache is not one-shot")
    cached[update] = dict(diagnostics)
    return receipt


def observation_v23(
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
        observed["state_residual_survival_diagnostics"] = None
        return observed
    cached = getattr(runtime, "state_residual_survival_diagnostics_v23", None)
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V23 observation lacks current-update diagnostics")
    observed["state_residual_survival_diagnostics"] = dict(cached[update])
    return observed


def validate_terminal_accounting_v23(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    if terminal_update not in _engine.TERMINAL_UPDATES:
        raise ValueError("V23 terminal update must be exactly 400 or 1000")
    value = _receipt_mapping(accounting, name="V23 terminal accounting")
    expected = _expected_accounting_v23(terminal_update)
    if (
        set(value) != set(expected)
        or any(type(item) is not int for item in value.values())
        or value != expected
    ):
        raise RuntimeError("V23 terminal accounting is inconsistent with the cap")
    _original_validate_terminal_accounting(
        _project_accounting_to_v21(value),
        terminal_update=terminal_update,
    )
    return expected


_engine.validate_bound_sources_v13 = validate_bound_sources_v23
_engine.validate_model_api_v13 = validate_model_api_v23
_engine.validate_training_api_v13 = validate_training_api_v23
_engine._validate_microbatches_for_engine_v13 = validate_microbatches_for_engine_v23
_engine._validate_update_integrity_v13 = validate_update_integrity_v23
_engine._observation_v13 = observation_v23
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v23

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = TRAINING_REQUIRED_BATCH_KEYS_V23
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

validate_content_bound_v23 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v23 = _engine.validate_future_execution_prerequisites_v13
execution_denial_receipt_v23 = _engine.execution_denial_receipt_v13
reserve_attempt_v23 = _engine.reserve_attempt_v13
terminalize_failure_v23 = _engine.terminalize_failure_v13
flatten_physical_metrics_v23 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v23 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v23 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v23 = _engine.evaluate_final_gate_v13
validate_schedule_v23 = _engine.validate_schedule_v13
validate_attempt_reservation_v23 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v23 = _engine.run_future_authorized_engine_v13
execute_v23 = _engine.execute_v13

# Compatibility names consumed inside the unchanged inherited lifecycle.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v23
validate_bound_sources_v13 = validate_bound_sources_v23
validate_model_api_v13 = validate_model_api_v23
validate_training_api_v13 = validate_training_api_v23
validate_future_execution_prerequisites_v13 = validate_future_execution_prerequisites_v23
execution_denial_receipt_v13 = execution_denial_receipt_v23
reserve_attempt_v13 = reserve_attempt_v23
terminalize_failure_v13 = terminalize_failure_v23
flatten_physical_metrics_v13 = flatten_physical_metrics_v23
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v23
evaluate_update400_gate_v13 = evaluate_update400_gate_v23
evaluate_final_gate_v13 = evaluate_final_gate_v23
validate_schedule_v13 = validate_schedule_v23
validate_attempt_reservation_v13 = validate_attempt_reservation_v23
run_future_authorized_engine_v13 = run_future_authorized_engine_v23
validate_terminal_accounting_v13 = validate_terminal_accounting_v23
execute_v13 = execute_v23


def private_adapter_receipt_v23() -> dict[str, Any]:
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
        "v22_scientific_result_commit": V22_SCIENTIFIC_RESULT_COMMIT,
        "v22_scientific_result_content_sha256": V22_SCIENTIFIC_RESULT_CONTENT_SHA256,
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "accounting_multipliers": dict(ACCOUNTING_MULTIPLIERS_V23),
        "state_residual_survival_parameter_tensor_count": (
            STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT
        ),
        "state_residual_survival_parameter_count": (
            STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT
        ),
        "scene_negative_row_batch_key": SCENE_NEGATIVE_ROW_KEY,
        "action_prior_batch_key": ACTION_PRIOR_M_KEY,
        "non_hold_action_count": NON_HOLD_ACTION_COUNT,
        "new_batch_fields_over_v21": 1,
        "update100_informational": True,
        "v21_update400_and_update1000_gates_unchanged": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V23 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v23(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
