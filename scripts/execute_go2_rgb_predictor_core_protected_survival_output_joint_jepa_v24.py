#!/usr/bin/env python3
"""Denied-by-default V24 executor adapter over frozen V23.

The V23 evaluator, causal controls, gates, custody, and one-shot engine remain
authoritative.  V24 changes only the auxiliary parameter-route validation and
publishes only V24 accounting, loss, route, and diagnostic identities.  This
source shell grants no execution authority.
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
V23_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23.py"
)
V23_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "44938145362e5accdf8e12b906bfbaa970d62f25"
)
V23_EXECUTOR_FILE_SHA256 = (
    "9f816eff5353984cd8335de49bb914f23fe4affeefcdeb0cab40b210e6db1884"
)
V23_EXECUTOR_BYTE_COUNT = 31_407
V23_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23"
)
PRIVATE_V23_MODULE_NAME = f"{__name__}.__private_v23_executor"
_PUBLIC_V23_WAS_LOADED_BEFORE_ADAPTER = V23_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24"
)
PREREGISTRATION_COMMIT = "475f1867149f5c5b764973bb5a371de83c29c3eb"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "ad0514668b20fd3bb58a2c70e71bb153428f3a9b121c1f8b64ca6e08965c6933"
)
PREREGISTRATION_BYTE_COUNT = 12_137
V23_SCIENTIFIC_RESULT_COMMIT = "04b0fa48c6c4e10868c2f302bc51100394e3907e"
V23_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23_scientific_result_2026-07-30.json"
)
V23_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "753c91babd4f7116444654167d2507ffb52d22f970fc926c05d287683954c994"
)
V23_SCIENTIFIC_RESULT_BYTE_COUNT = 20_640
V23_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "a5a6b8aa7312706d2ae3a5b53e39370462e9de6eda6b7a2ca4e2e0226a518ed8"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24/"
    "attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
ACTION_PRIOR_M_KEY = "train_action_prior_m"
SCENE_NEGATIVE_ROW_KEY = "scene_innovation_negative_row"
PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME = (
    "predictor_core_protected_survival_output"
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT = 96
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT = 3_106_409
PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT = 13
PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT = 259_008
SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES = (
    "predictor.swept_progress_head.output.weight",
    "predictor.swept_progress_head.output.bias",
)
PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES = (
    "predictor.action_embedding.weight",
    "predictor.input_projection.weight",
    "predictor.input_projection.bias",
    "predictor.residual_blocks.0.conv1.weight",
    "predictor.residual_blocks.0.conv1.bias",
    "predictor.residual_blocks.0.conv2.weight",
    "predictor.residual_blocks.0.conv2.bias",
    "predictor.residual_blocks.1.conv1.weight",
    "predictor.residual_blocks.1.conv1.bias",
    "predictor.residual_blocks.1.conv2.weight",
    "predictor.residual_blocks.1.conv2.bias",
    "predictor.residual_head.weight",
    "predictor.residual_head.bias",
)
PROTECTED_PREDICTOR_CORE_PARAMETER_SIZES = (
    576,
    73_728,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
)
NON_HOLD_ACTION_COUNT = 8
MEAN_LOSS_NAMES = (
    "S", "P", "U", "R", "O", "F", "J_rank", "J24", "N", "C", "L",
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_DIAGNOSTIC_NAMES = (
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
ACCOUNTING_MULTIPLIERS_V24 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 12,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "predictor_core_protected_survival_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 8,
    "predictor_core_protected_survival_objectives": 4,
}


def _load_private_v23_executor() -> ModuleType:
    if V23_EXECUTOR_PATH.is_symlink() or not V23_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("frozen V23 executor source is absent or not regular")
    source = V23_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V23_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V23_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V23 executor source binding changed")
    if PRIVATE_V23_MODULE_NAME in sys.modules:
        raise RuntimeError("private V23 executor module name is already occupied")
    module = ModuleType(PRIVATE_V23_MODULE_NAME)
    module.__file__ = str(V23_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V23_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V23_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V23_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V23_MODULE_NAME)
    return module


_v23 = _load_private_v23_executor()
if (
    not _v23.SCHEMA_PREFIX.endswith("joint_jepa_v23")
    or _v23.STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT != 109
    or _v23.STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT != 3_365_417
    or _v23.MAXIMUM_UPDATES != 1_000
    or _v23.MAXIMUM_PRESENTATIONS != 16_000
    or _v23.CURRENT_EXECUTION_AUTHORIZED is not False
):
    raise RuntimeError("frozen V23 executor defaults changed")

_engine = _v23._engine
_original_validate_bound_sources = _v23.validate_bound_sources_v23
_original_validate_model_api = _v23.validate_model_api_v23
_original_validate_training_api = _v23.validate_training_api_v23
_original_validate_microbatches = _v23.validate_microbatches_for_engine_v23
_original_validate_update_integrity = _v23.validate_update_integrity_v23
# Bypass only V23's diagnostic relabeler.  The inherited V21 observation still
# owns every unchanged evaluator/control receipt and needs no V23-private cache.
_original_observation = _v23._base.observation_v21
_original_validate_terminal_accounting = _v23.validate_terminal_accounting_v23
TRAINING_REQUIRED_BATCH_KEYS_V23 = tuple(_v23.TRAINING_REQUIRED_BATCH_KEYS_V23)
TRAINING_REQUIRED_BATCH_KEYS_V24 = TRAINING_REQUIRED_BATCH_KEYS_V23

_bound_parent_sources = dict(_v23.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V23_SCIENTIFIC_RESULT_PATH: (
            V23_SCIENTIFIC_RESULT_FILE_SHA256,
            V23_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
)
for _module in (_v23._base, _v23, _engine):
    _module.SCHEMA_PREFIX = SCHEMA_PREFIX
    _module.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
    _module.PREREGISTRATION_PATH = PREREGISTRATION_PATH
    _module.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
    _module.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V24 predictor-core-protected survival-output execution is denied until "
    "recursive source closure, independent review, narrow certification, and "
    "one-shot authority"
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


def _expected_accounting_v24(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V24.items()
    }


def _project_accounting_to_v23(accounting: Mapping[str, int]) -> dict[str, int]:
    projected = dict(accounting)
    projected["state_residual_survival_grad_calls"] = projected.pop(
        "predictor_core_protected_survival_grad_calls"
    )
    projected["state_residual_survival_objectives"] = projected.pop(
        "predictor_core_protected_survival_objectives"
    )
    if set(projected) != set(_v23.ACCOUNTING_MULTIPLIERS_V23):
        raise RuntimeError("V24-to-V23 accounting projection changed")
    return projected


def validate_bound_sources_v24(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_model_api_v24(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V24 did not retain the exact V18 model class")
    return receipt


def validate_training_api_v24(module: Any) -> dict[str, Any]:
    inherited = dict(_original_validate_training_api(module))
    for name in ("JointTrainingAccountingV24", "JointUpdateResultV24"):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V24 training type is absent: {name}")
    for name in ("joint_training_update_v24", "validate_accounting_v24"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V24 training callable is absent: {name}")
    if (
        tuple(getattr(module, "REQUIRED_BATCH_KEYS_V24", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V24
        or tuple(getattr(module, "REQUIRED_BATCH_KEYS_V23", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V23
        or getattr(module, "ACTION_PRIOR_M_KEY_V23", None) != ACTION_PRIOR_M_KEY
        or getattr(module, "PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24", None)
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
        or getattr(
            module,
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24",
            None,
        )
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        or getattr(
            module,
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24",
            None,
        )
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT
        or getattr(
            module,
            "PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24",
            None,
        )
        != PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
        or getattr(
            module, "PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24", None
        )
        != PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
        or tuple(
            getattr(module, "PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES_V24", ())
        )
        != PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES
    ):
        raise RuntimeError("V24 training API or frozen batch schema changed")
    for name in (
        "required_batch_key_count_v23",
        "state_residual_survival_route",
        "state_residual_survival_objectives_per_update",
    ):
        inherited.pop(name, None)
    return {
        **inherited,
        "required_batch_key_count_v24": len(TRAINING_REQUIRED_BATCH_KEYS_V24),
        "predictor_core_protected_survival_route": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
        ),
        "backward_calls_per_update": 12,
        "predictor_objectives_per_update": 8,
        "predictor_core_protected_survival_objectives_per_update": 4,
        "new_batch_fields_over_predecessor": 0,
        "j24_parameter_tensor_count": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        ),
        "j24_parameter_count": PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT,
        "protected_predictor_core_parameter_tensor_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
        ),
        "protected_predictor_core_parameter_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
        ),
    }


def validate_microbatches_for_engine_v24(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != _engine.MICROBATCHES_PER_UPDATE or any(
        type(batch) is not dict or tuple(batch) != TRAINING_REQUIRED_BATCH_KEYS_V24
        for batch in microbatches
    ):
        raise PermissionError("V24 engine microbatch schema changed")
    _original_validate_microbatches(runtime, model, microbatches)


def _parameter_name_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(tuple(names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_predictor_core_protected_parameter_subset(
    model: Any,
) -> dict[str, Any]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("V24 model named-parameter API is absent")
    named = tuple(named_parameters())
    if (
        any(not isinstance(name, str) for name, _ in named)
        or len({name for name, _ in named}) != len(named)
        or len({id(parameter) for _, parameter in named}) != len(named)
    ):
        raise RuntimeError("V24 model named-parameter inventory is not unique")
    full_v23 = tuple(
        (name, parameter)
        for name, parameter in named
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
    included = tuple(
        (name, parameter)
        for name, parameter in full_v23
        if not name.startswith("predictor.")
        or name in SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES
    )
    protected = tuple(
        (name, parameter)
        for name, parameter in full_v23
        if name.startswith("predictor.")
        and name not in SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES
    )
    included_names = tuple(name for name, _ in included)
    protected_names = tuple(name for name, _ in protected)
    protected_sizes = tuple(int(parameter.numel()) for _, parameter in protected)
    if (
        len(full_v23) != 109
        or sum(int(parameter.numel()) for _, parameter in full_v23) != 3_365_417
        or len(included)
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in included)
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT
        or len(protected) != PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in protected)
        != PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
        or protected_names != PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES
        or protected_sizes != PROTECTED_PREDICTOR_CORE_PARAMETER_SIZES
        or tuple(name for name, _ in included[-2:])
        != SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES
        or set(included_names) & set(protected_names)
        or set(included_names) | set(protected_names)
        != {name for name, _ in full_v23}
        or any(name.startswith("semantic_head.") for name in included_names)
        or any(name.startswith(("target_encoder.", "target_bev_lift.")) for name in included_names)
    ):
        raise RuntimeError("V24 predictor-core-protected parameter subset changed")
    return {
        "included_parameter_tensor_count": len(included),
        "included_parameter_count": sum(
            int(parameter.numel()) for _, parameter in included
        ),
        "included_parameter_name_sha256": _parameter_name_sha256(included_names),
        "protected_parameter_tensor_count": len(protected),
        "protected_parameter_count": sum(
            int(parameter.numel()) for _, parameter in protected
        ),
        "protected_parameter_names": protected_names,
        "protected_parameter_name_sha256": _parameter_name_sha256(protected_names),
    }


def _validate_predictor_core_protected_survival_diagnostics(
    value: Any,
) -> dict[str, float | int]:
    retained = _v23._validate_state_residual_survival_diagnostics(value)
    if set(retained) != set(PREDICTOR_CORE_PROTECTED_SURVIVAL_DIAGNOSTIC_NAMES):
        raise RuntimeError("V24 output diagnostic fields changed")
    return retained


def _pop_private_compatibility_cache(runtime: Any, name: str, update: int) -> None:
    cached = getattr(runtime, name, None)
    if isinstance(cached, dict):
        cached.pop(update, None)
        if not cached:
            try:
                delattr(runtime, name)
            except AttributeError:
                pass


def validate_update_integrity_v24(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact V24 evidence around a private V23 compatibility view."""

    if type(update) is not int or not 1 <= update <= _engine.MAXIMUM_UPDATES:
        raise ValueError("V24 update integrity index escaped the cap")
    accounting = _receipt_mapping(result.accounting, name="V24 accounting")
    expected_accounting = _expected_accounting_v24(update)
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise RuntimeError("V24 per-update accounting changed")
    inherited_route_names = (
        "camera_shared", "joint_shared", "representation", "predictor"
    )
    routes = result.gradient_routes
    if not isinstance(routes, Mapping) or set(routes) != {
        *inherited_route_names,
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME,
    }:
        raise RuntimeError("V24 gradient-route receipt set changed")
    losses = dict(result.mean_losses)
    if set(losses) != set(MEAN_LOSS_NAMES):
        raise RuntimeError("V24 per-update mean-loss receipt set changed")
    finite_losses = {
        name: _finite(value, name=f"V24 mean loss {name}")
        for name, value in losses.items()
    }
    if (
        any(finite_losses[name] < 0.0 for name in ("F", "J_rank", "J24"))
        or not math.isclose(
            finite_losses["N"],
            sum(finite_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["J24"],
            finite_losses["F"] + finite_losses["J_rank"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["L"],
            finite_losses["N"] + finite_losses["C"] + finite_losses["J24"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V24 N/L/J24 loss equations changed")
    diagnostics = _validate_predictor_core_protected_survival_diagnostics(
        result.predictor_core_protected_survival_diagnostics
    )
    if not math.isclose(
        finite_losses["F"],
        float(diagnostics["positive_energy_mean"]),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V24 F disagrees with positive energy")

    auxiliary_route = _receipt_mapping(
        routes[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME],
        name="V24 predictor-core-protected survival gradient receipt",
    )
    expected_route_fields = {
        "preclip_l2",
        "applied_scale",
        "parameter_tensor_count",
        "absent_tensor_gradient_count",
    }
    if set(auxiliary_route) != expected_route_fields:
        raise RuntimeError("V24 output route fields changed")
    norm = _finite(auxiliary_route["preclip_l2"], name="V24 route preclip norm")
    scale = _finite(auxiliary_route["applied_scale"], name="V24 route scale")
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
        or auxiliary_route["parameter_tensor_count"]
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        or auxiliary_route["absent_tensor_gradient_count"] != 0
    ):
        raise RuntimeError("V24 output gradient route failed integrity")
    route_inventory = _validate_predictor_core_protected_parameter_subset(model)

    projected_result = SimpleNamespace(
        accounting=_project_accounting_to_v23(accounting),
        gradient_routes={
            **{name: routes[name] for name in inherited_route_names},
            _v23.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME: {
                **auxiliary_route,
                "parameter_tensor_count": (
                    _v23.STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT
                ),
            },
        },
        mean_losses={
            **{
                name: finite_losses[name]
                for name in ("S", "P", "U", "R", "O", "F", "J_rank", "N", "C", "L")
            },
            "J23": finite_losses["J24"],
        },
        state_residual_survival_diagnostics=dict(diagnostics),
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
    # The V23 validator creates this cache solely for its own observation
    # relabeler.  V24 never publishes or retains that compatibility evidence.
    _pop_private_compatibility_cache(
        runtime, "state_residual_survival_diagnostics_v23", update
    )
    inherited_predictor_route = _receipt_mapping(
        receipt["gradient_routes"]["predictor"],
        name="V24 inherited predictor gradient receipt",
    )
    if (
        inherited_predictor_route.get("parameter_tensor_count") != 15
        or inherited_predictor_route.get("absent_tensor_gradient_count") != 0
        or _finite(
            inherited_predictor_route.get("preclip_l2"),
            name="V24 inherited predictor preclip norm",
        )
        <= 0.0
    ):
        raise RuntimeError("V24 inherited joint route did not retain all 15 predictors")
    receipt["accounting"] = accounting
    receipt["gradient_routes"] = {
        **{name: receipt["gradient_routes"][name] for name in inherited_route_names},
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME: auxiliary_route,
    }
    receipt["mean_losses"] = finite_losses
    receipt.pop("state_residual_survival_diagnostics", None)
    receipt.pop("v23_action_prior_residualized_wrong_scene_survival_output", None)
    receipt["predictor_core_protected_survival_diagnostics"] = dict(diagnostics)
    receipt["v24_predictor_core_protected_survival_output"] = {
        "j24_parameter_tensor_count": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        ),
        "j24_parameter_count": PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT,
        "protected_predictor_core_parameter_tensor_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
        ),
        "protected_predictor_core_parameter_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
        ),
        "inherited_joint_predictor_parameter_tensor_count": 15,
        "included_parameter_name_sha256": route_inventory[
            "included_parameter_name_sha256"
        ],
        "protected_predictor_core_parameter_names": list(
            route_inventory["protected_parameter_names"]
        ),
        "protected_predictor_core_parameter_name_sha256": route_inventory[
            "protected_parameter_name_sha256"
        ],
        "j24_computational_graph_through_predictor_core": True,
        "predictor_core_gradient_from_j24": False,
        "predictor_core_gradient_from_inherited_joint": True,
        "swept_progress_output_gradient_from_j24": True,
        "encoder_gradient_from_j24": True,
        "representation_gradient_from_j24": True,
        "semantic_head_gradient_from_j24": False,
        "target_gradient_from_j24": False,
        "objective_bit_identical_to_v23": True,
        "passed": True,
    }
    cached = getattr(
        runtime, "predictor_core_protected_survival_diagnostics_v24", None
    )
    if cached is None:
        cached = {}
        runtime.predictor_core_protected_survival_diagnostics_v24 = cached
    if type(cached) is not dict or update in cached:
        raise RuntimeError("V24 output diagnostic cache is not one-shot")
    cached[update] = dict(diagnostics)
    forbidden_v23_fields = {
        "state_residual_survival_diagnostics",
        "v23_action_prior_residualized_wrong_scene_survival_output",
    }
    if forbidden_v23_fields & set(receipt):
        raise RuntimeError("V24 receipt leaked private V23 compatibility evidence")
    if "J23" in receipt["mean_losses"] or _v23.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME in receipt[
        "gradient_routes"
    ]:
        raise RuntimeError("V24 receipt leaked a private V23 loss or route")
    return receipt


def observation_v24(
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
        observed["predictor_core_protected_survival_diagnostics"] = None
        return observed
    cached = getattr(
        runtime, "predictor_core_protected_survival_diagnostics_v24", None
    )
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V24 observation lacks current-update diagnostics")
    observed["predictor_core_protected_survival_diagnostics"] = dict(cached[update])
    return observed


def validate_terminal_accounting_v24(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    if terminal_update not in _engine.TERMINAL_UPDATES:
        raise ValueError("V24 terminal update must be exactly 400 or 1000")
    value = _receipt_mapping(accounting, name="V24 terminal accounting")
    expected = _expected_accounting_v24(terminal_update)
    if (
        set(value) != set(expected)
        or any(type(item) is not int for item in value.values())
        or value != expected
    ):
        raise RuntimeError("V24 terminal accounting is inconsistent with the cap")
    _original_validate_terminal_accounting(
        _project_accounting_to_v23(value),
        terminal_update=terminal_update,
    )
    return expected


_engine.validate_bound_sources_v13 = validate_bound_sources_v24
_engine.validate_model_api_v13 = validate_model_api_v24
_engine.validate_training_api_v13 = validate_training_api_v24
_engine._validate_microbatches_for_engine_v13 = validate_microbatches_for_engine_v24
_engine._validate_update_integrity_v13 = validate_update_integrity_v24
_engine._observation_v13 = observation_v24
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v24

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = TRAINING_REQUIRED_BATCH_KEYS_V24
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

validate_content_bound_v24 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v24 = _engine.validate_future_execution_prerequisites_v13
execution_denial_receipt_v24 = _engine.execution_denial_receipt_v13
reserve_attempt_v24 = _engine.reserve_attempt_v13
terminalize_failure_v24 = _engine.terminalize_failure_v13
flatten_physical_metrics_v24 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v24 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v24 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v24 = _engine.evaluate_final_gate_v13
validate_schedule_v24 = _engine.validate_schedule_v13
validate_attempt_reservation_v24 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v24 = _engine.run_future_authorized_engine_v13
execute_v24 = _engine.execute_v13

# Compatibility names consumed inside the unchanged inherited lifecycle.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v24
validate_bound_sources_v13 = validate_bound_sources_v24
validate_model_api_v13 = validate_model_api_v24
validate_training_api_v13 = validate_training_api_v24
validate_future_execution_prerequisites_v13 = validate_future_execution_prerequisites_v24
execution_denial_receipt_v13 = execution_denial_receipt_v24
reserve_attempt_v13 = reserve_attempt_v24
terminalize_failure_v13 = terminalize_failure_v24
flatten_physical_metrics_v13 = flatten_physical_metrics_v24
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v24
evaluate_update400_gate_v13 = evaluate_update400_gate_v24
evaluate_final_gate_v13 = evaluate_final_gate_v24
validate_schedule_v13 = validate_schedule_v24
validate_attempt_reservation_v13 = validate_attempt_reservation_v24
run_future_authorized_engine_v13 = run_future_authorized_engine_v24
validate_terminal_accounting_v13 = validate_terminal_accounting_v24
execute_v13 = execute_v24


def private_adapter_receipt_v24() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v23_executor_adapter_v1",
        "base_executor": str(V23_EXECUTOR_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": V23_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_executor_file_sha256": V23_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V23_EXECUTOR_BYTE_COUNT,
        "public_v23_was_loaded_before_adapter": _PUBLIC_V23_WAS_LOADED_BEFORE_ADAPTER,
        "public_v23_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V23_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v23_scientific_result_commit": V23_SCIENTIFIC_RESULT_COMMIT,
        "v23_scientific_result_content_sha256": (
            V23_SCIENTIFIC_RESULT_CONTENT_SHA256
        ),
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "accounting_multipliers": dict(ACCOUNTING_MULTIPLIERS_V24),
        "j24_parameter_tensor_count": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
        ),
        "j24_parameter_count": PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT,
        "protected_predictor_core_parameter_tensor_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
        ),
        "protected_predictor_core_parameter_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
        ),
        "inherited_joint_predictor_parameter_tensor_count": 15,
        "objective_bit_identical_to_v23": True,
        "new_batch_fields_over_v23": 0,
        "update100_informational": True,
        "v23_update400_and_update1000_gates_unchanged": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V24 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v24(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
