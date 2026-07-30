#!/usr/bin/env python3
"""Denied-by-default executor adapter for V20 accounting isolation.

The exact frozen V18 executor is loaded into a private module object.  V20
preserves V19's factual-successor route while isolating its extended accounting
from the inherited V13 receipt validator.  This source shell grants no
execution authority.
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
V18_EXECUTOR_PATH = (
    ROOT / "scripts/execute_go2_rgb_object_space_height_volume_joint_jepa_v18.py"
)
V18_EXECUTOR_COMMIT = "5567c9aa152b8aedcc085cfff46a7975668f7bfa"
V18_EXECUTOR_FILE_SHA256 = (
    "5ce4259126c21d0f474c0548f0ee6757f78225daa8ed778540f83764496d0e92"
)
V18_EXECUTOR_BYTE_COUNT = 22_830
V18_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
)
PRIVATE_V18_MODULE_NAME = f"{__name__}.__private_v18_executor"
_PUBLIC_V18_WAS_LOADED_BEFORE_ADAPTER = V18_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
    "grounding_joint_jepa_v20"
)
PREREGISTRATION_COMMIT = "c99837b91aeb959e07da94e898e3ba11ccbb4c04"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v20_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "3f450b8949022514f82448d122de637d4cefd91829a72d0ac3f8b14a789a42bd"
)
PREREGISTRATION_BYTE_COUNT = 9_732
V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_COMMIT = (
    "691ed5d39f0b8d1b40071045dc181b9a4b215573"
)
V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "preregistration_2026-07-30.md"
)
V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_FILE_SHA256 = (
    "9a1910e6c12ce27bf7951fe4bddbcfc80d19e1d0fc33d03359cc27d12dd1b79b"
)
V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_BYTE_COUNT = 8_107
V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_COMMIT = (
    "7105e2d9ed6e724f364c837e84177b6b4c4cd163"
)
V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
    "terminal_failure_result_2026-07-30.json"
)
V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "1b155248194ffd6d7943f84d88c25e29843fb9c977fc5b9bd8053e381c49b886"
)
V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 9_497
ORIGINAL_V19_PREREGISTRATION_COMMIT = (
    "6255a9a2cccffde4e777169eacf95105a828cf7e"
)
ORIGINAL_V19_PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_preregistration_2026-07-30.md"
)
ORIGINAL_V19_PREREGISTRATION_FILE_SHA256 = (
    "350885460f1efbd0bcb5640d4657cdd34ec0244d71d2174103e53ce37daf4a4f"
)
ORIGINAL_V19_PREREGISTRATION_BYTE_COUNT = 13_376
V19_TERMINAL_FAILURE_RESULT_COMMIT = (
    "37a87ac49ebcdebe57263476c20b1476877e36c2"
)
V19_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19_terminal_failure_result_2026-07-30.json"
)
V19_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "1f1708d615cbf375c99fa49efd11699882c302ad92ae436e925af152e18da36d"
)
V19_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 9_292
V18_SCIENTIFIC_RESULT_COMMIT = "f2e290ce42f7b0cd142131f3272d1119b7b5d3d1"
V18_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
    "replacement_v3_scientific_result_2026-07-30.json"
)
V18_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "48f1168b33b6bf8cc7c437940ed0dabef9b5a29802813e3a1351e8bba1e2875a"
)
V18_SCIENTIFIC_RESULT_BYTE_COUNT = 11_380

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v20/attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"

FACTUAL_SUCCESSOR_ROUTE_NAME = "factual_successor_predictor"
FACTUAL_SUCCESSOR_PARAMETER_TENSOR_COUNT = 13
FACTUAL_SUCCESSOR_PARAMETER_COUNT = 259_008
FACTUAL_SUCCESSOR_DIAGNOSTIC_NAMES = (
    "successor_semantic_nll_normalized",
    "persistence_semantic_nll_normalized",
    "successor_minus_persistence_nll_normalized",
    "changed_cell_fraction",
    "non_hold_row_count",
    "matching_predictor_gradient_cosine",
)
MEAN_LOSS_NAMES = ("S", "P", "U", "R", "O", "Q", "N", "C", "L")
ACCOUNTING_MULTIPLIERS_V19 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 12,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "factual_successor_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 8,
    "factual_successor_objectives": 4,
}
_INHERITED_ACCOUNTING_MULTIPLIERS = {
    name: multiplier
    for name, multiplier in ACCOUNTING_MULTIPLIERS_V19.items()
    if name not in {"factual_successor_grad_calls", "factual_successor_objectives"}
}
_INHERITED_ACCOUNTING_MULTIPLIERS["backward_calls"] = 8
_INHERITED_ACCOUNTING_MULTIPLIERS["predictor_objectives"] = 4
INHERITED_ACCOUNTING_MULTIPLIERS_V13 = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 8,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 4,
}
if _INHERITED_ACCOUNTING_MULTIPLIERS != INHERITED_ACCOUNTING_MULTIPLIERS_V13:
    raise RuntimeError("V20 inherited accounting projection changed")

_PRISTINE_V18_DEFAULTS = {
    "SCHEMA_PREFIX": (
        "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3"
    ),
    "PREREGISTRATION_COMMIT": "81d1557cce55a448a84e00b5e822923b590e6f7d",
    "PREREGISTRATION_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3_preregistration_2026-07-30.md"
    ),
    "OUTPUT_ROOT_RELATIVE_PATH": (
        ".generated/go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v3/attempt_v1"
    ),
    "MODEL_CLASS_NAME": MODEL_CLASS_NAME,
    "MAXIMUM_UPDATES": 1_000,
    "MAXIMUM_PRESENTATIONS": 16_000,
    "OBSERVATION_UPDATES": (0, 100, 400, 1_000),
    "TERMINAL_UPDATES": (400, 1_000),
    "MATCHED_UPDATE400_THRESHOLDS": {
        "passed_margin_count_strictly_greater_than": 72,
        "total_shortfall_strictly_less_than": 68.96954700805838,
        "rough_depth_p95_m_strictly_less_than": 1.8582415819168085,
    },
}


def _load_private_v18_executor() -> ModuleType:
    if V18_EXECUTOR_PATH.is_symlink() or not V18_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("V18 executor source is absent or not regular")
    source = V18_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V18_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V18_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V18 executor source binding changed")
    if PRIVATE_V18_MODULE_NAME in sys.modules:
        raise RuntimeError("private V18 executor module name is already occupied")
    module = ModuleType(PRIVATE_V18_MODULE_NAME)
    module.__file__ = str(V18_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V18_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V18_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V18_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V18_MODULE_NAME)
    return module


def _assert_pristine_v18_defaults(module: ModuleType) -> None:
    observed = {
        name: getattr(module, name, object()) for name in _PRISTINE_V18_DEFAULTS
    }
    if observed != _PRISTINE_V18_DEFAULTS:
        changed = sorted(
            name
            for name, expected in _PRISTINE_V18_DEFAULTS.items()
            if observed[name] != expected
        )
        raise RuntimeError(f"V18 executor adapter defaults changed: {changed}")
    if getattr(module, "CURRENT_EXECUTION_AUTHORIZED", None) is not False:
        raise PermissionError("V18 source shell unexpectedly grants execution")


_base = _load_private_v18_executor()
_assert_pristine_v18_defaults(_base)
_engine = _base._engine
_original_validate_bound_sources = _engine.validate_bound_sources_v13
_original_validate_model_api = _engine.validate_model_api_v13
_original_validate_training_api = _engine.validate_training_api_v13
_original_validate_update_integrity = _engine._validate_update_integrity_v13
_original_observation = _engine._observation_v13
_original_evaluate_update400_gate = _engine.evaluate_update400_gate_v13


def _assert_inherited_accounting_registry_v20() -> None:
    observed = getattr(_engine, "ACCOUNTING_MULTIPLIERS", None)
    if observed != INHERITED_ACCOUNTING_MULTIPLIERS_V13:
        raise RuntimeError("V20 inherited accounting registry changed")


_assert_inherited_accounting_registry_v20()

_bound_parent_sources = dict(_engine.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_PATH: (
            V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_FILE_SHA256,
            V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_BYTE_COUNT,
        ),
        V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_PATH: (
            V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        ORIGINAL_V19_PREREGISTRATION_PATH: (
            ORIGINAL_V19_PREREGISTRATION_FILE_SHA256,
            ORIGINAL_V19_PREREGISTRATION_BYTE_COUNT,
        ),
        V19_TERMINAL_FAILURE_RESULT_PATH: (
            V19_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            V19_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        V18_SCIENTIFIC_RESULT_PATH: (
            V18_SCIENTIFIC_RESULT_FILE_SHA256,
            V18_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
)

_engine.SCHEMA_PREFIX = SCHEMA_PREFIX
_engine.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
_engine.PREREGISTRATION_PATH = PREREGISTRATION_PATH
_engine.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
_engine.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V20 accounting-isolation scientific execution is denied until "
    "recursive source closure, "
    "independent exact-binding review, narrow clean-export certification, "
    "and one-shot authority are committed and validated"
)


def _receipt_mapping_v19(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"{name} must be a dataclass or mapping")
    if not all(isinstance(key, str) for key in result):
        raise TypeError(f"{name} keys must be strings")
    return result


def _finite_v19(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def validate_model_api_v19(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V19 did not retain the exact V18 model class")
    return receipt


def validate_bound_sources_v19(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_training_api_v19(module: Any) -> dict[str, Any]:
    receipt = _original_validate_training_api(module)
    for name in ("JointTrainingAccountingV19", "JointUpdateResultV19"):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V19 training type is absent: {name}")
    for name in ("joint_training_update_v19", "validate_accounting_v19"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V19 training callable is absent: {name}")
    return {
        **receipt,
        "factual_successor_route": FACTUAL_SUCCESSOR_ROUTE_NAME,
        "backward_calls_per_update": 12,
        "predictor_objectives_per_update": 8,
        "factual_successor_objectives_per_update": 4,
    }


def _expected_accounting_v19(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V19.items()
    }


def _validate_factual_successor_parameter_subset_v19(model: Any) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("V19 model named-parameter API is absent")
    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters()
        if name.startswith("predictor.")
        and not name.startswith("predictor.swept_progress_head.")
    )
    if (
        len(selected) != FACTUAL_SUCCESSOR_PARAMETER_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in selected)
        != FACTUAL_SUCCESSOR_PARAMETER_COUNT
    ):
        raise RuntimeError("V19 factual-successor predictor subset changed")


def validate_update_integrity_v19(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_inherited_accounting_registry_v20()
    if type(update) is not int or not 1 <= update <= _engine.MAXIMUM_UPDATES:
        raise ValueError("V19 update integrity index escaped the cap")

    accounting = _receipt_mapping_v19(result.accounting, name="V19 accounting")
    expected_accounting = _expected_accounting_v19(update)
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise RuntimeError("V19 per-update accounting changed")

    routes = result.gradient_routes
    inherited_route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
    )
    if not isinstance(routes, Mapping) or set(routes) != {
        *inherited_route_names,
        FACTUAL_SUCCESSOR_ROUTE_NAME,
    }:
        raise RuntimeError("V19 gradient-route receipt set changed")

    losses = dict(result.mean_losses)
    if set(losses) != set(MEAN_LOSS_NAMES):
        raise RuntimeError("V19 per-update mean-loss receipt set changed")
    finite_losses = {
        name: _finite_v19(value, name=f"mean loss {name}")
        for name, value in losses.items()
    }
    if (
        not math.isclose(
            finite_losses["N"],
            sum(finite_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["L"],
            finite_losses["N"] + finite_losses["C"] + finite_losses["Q"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V19 N/L/Q loss equations changed")

    diagnostics = _receipt_mapping_v19(
        result.factual_successor_diagnostics,
        name="V19 factual-successor diagnostics",
    )
    if set(diagnostics) != set(FACTUAL_SUCCESSOR_DIAGNOSTIC_NAMES):
        raise RuntimeError("V19 factual-successor diagnostic fields changed")
    finite_diagnostics = {
        name: _finite_v19(value, name=f"factual-successor diagnostic {name}")
        for name, value in diagnostics.items()
        if name != "non_hold_row_count"
    }
    non_hold_count = diagnostics["non_hold_row_count"]
    if (
        type(non_hold_count) is not int
        or not 0 <= non_hold_count <= _engine.PRESENTATIONS_PER_UPDATE
    ):
        raise RuntimeError("V19 non-HOLD row accounting changed")
    if (
        finite_diagnostics["successor_semantic_nll_normalized"] < 0.0
        or finite_diagnostics["persistence_semantic_nll_normalized"] < 0.0
        or not 0.0 <= finite_diagnostics["changed_cell_fraction"] <= 1.0
        or not -1.0
        <= finite_diagnostics["matching_predictor_gradient_cosine"]
        <= 1.0
        or not math.isclose(
            finite_diagnostics["successor_minus_persistence_nll_normalized"],
            finite_diagnostics["successor_semantic_nll_normalized"]
            - finite_diagnostics["persistence_semantic_nll_normalized"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
        or not math.isclose(
            finite_losses["Q"],
            finite_diagnostics["successor_semantic_nll_normalized"],
            rel_tol=2e-6,
            abs_tol=2e-6,
        )
    ):
        raise RuntimeError("V19 factual-successor diagnostics are inconsistent")

    q_route = _receipt_mapping_v19(
        routes[FACTUAL_SUCCESSOR_ROUTE_NAME],
        name="V19 factual-successor gradient receipt",
    )
    expected_route_fields = {
        "preclip_l2",
        "applied_scale",
        "parameter_tensor_count",
        "absent_tensor_gradient_count",
    }
    if set(q_route) != expected_route_fields:
        raise RuntimeError("V19 factual-successor route fields changed")
    norm = _finite_v19(q_route["preclip_l2"], name="V19 Q-route preclip norm")
    scale = _finite_v19(q_route["applied_scale"], name="V19 Q-route scale")
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
        or q_route["parameter_tensor_count"]
        != FACTUAL_SUCCESSOR_PARAMETER_TENSOR_COUNT
        or q_route["absent_tensor_gradient_count"] != 0
    ):
        raise RuntimeError("V19 factual-successor gradient route failed integrity")
    _validate_factual_successor_parameter_subset_v19(model)

    inherited_accounting = {
        name: update * multiplier
        for name, multiplier in _INHERITED_ACCOUNTING_MULTIPLIERS.items()
    }
    inherited_losses = {
        name: finite_losses[name] for name in ("S", "P", "U", "R", "O", "N", "C")
    }
    inherited_losses["L"] = inherited_losses["N"] + inherited_losses["C"]
    inherited_result = SimpleNamespace(
        accounting=inherited_accounting,
        gradient_routes={name: routes[name] for name in inherited_route_names},
        mean_losses=inherited_losses,
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
        inherited_result,
        update=update,
        access_receipt=access_receipt,
    )
    receipt["accounting"] = accounting
    receipt["gradient_routes"] = {
        **receipt["gradient_routes"],
        FACTUAL_SUCCESSOR_ROUTE_NAME: q_route,
    }
    receipt["mean_losses"] = finite_losses
    receipt["factual_successor_diagnostics"] = {
        **finite_diagnostics,
        "non_hold_row_count": non_hold_count,
    }
    receipt["v19_executed_successor_semantic_grounding"] = {
        "parameter_tensor_count": FACTUAL_SUCCESSOR_PARAMETER_TENSOR_COUNT,
        "parameter_count": FACTUAL_SUCCESSOR_PARAMETER_COUNT,
        "representation_gradient_from_q": False,
        "semantic_head_gradient_from_q": False,
        "target_gradient_from_q": False,
        "passed": True,
    }
    return receipt


def _validate_causal_comparisons_v19(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_engine.CONTROL_NAMES):
        raise RuntimeError("V19 numeric causal-comparison control set changed")
    retained: dict[str, Any] = {}
    expected_row_fields = {
        "scene_count",
        "bootstrap_replicates",
        "bootstrap_seed",
        "equal_scene_mean_delta",
        "bootstrap_lower_95",
        "per_scene_delta",
        "positive_family_count",
        "family_deltas",
    }
    for control in _engine.CONTROL_NAMES:
        row = value[control]
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise RuntimeError(f"V19 numeric causal-comparison fields changed: {control}")
        if (
            row["scene_count"] != 8
            or type(row["scene_count"]) is not int
            or row["bootstrap_replicates"] != 10_000
            or type(row["bootstrap_replicates"]) is not int
            or row["bootstrap_seed"] != _engine.BOOTSTRAP_SEED
            or type(row["bootstrap_seed"]) is not int
        ):
            raise RuntimeError(f"V19 causal comparison constants changed: {control}")
        per_scene_delta = row["per_scene_delta"]
        if (
            not isinstance(per_scene_delta, Mapping)
            or len(per_scene_delta) != 8
            or not all(isinstance(scene, str) and scene for scene in per_scene_delta)
        ):
            raise RuntimeError(f"V19 causal scene set changed: {control}")
        retained_per_scene_delta = {
            scene: _finite_v19(
                delta,
                name=f"{control} scene delta {scene}",
            )
            for scene, delta in per_scene_delta.items()
        }
        family_deltas = row["family_deltas"]
        if (
            not isinstance(family_deltas, Mapping)
            or set(family_deltas) != set(_engine.REGISTERED_FAMILIES)
        ):
            raise RuntimeError(f"V19 causal family set changed: {control}")
        retained_family_deltas = {
            family: _finite_v19(
                family_deltas[family],
                name=f"{control} family delta {family}",
            )
            for family in _engine.REGISTERED_FAMILIES
        }
        positive_family_count = row["positive_family_count"]
        if (
            type(positive_family_count) is not int
            or positive_family_count
            != sum(delta > 0.0 for delta in retained_family_deltas.values())
        ):
            raise RuntimeError(f"V19 positive-family count changed: {control}")
        retained[control] = {
            "scene_count": 8,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": _engine.BOOTSTRAP_SEED,
            "equal_scene_mean_delta": _finite_v19(
                row["equal_scene_mean_delta"],
                name=f"{control} equal-scene mean delta",
            ),
            "bootstrap_lower_95": _finite_v19(
                row["bootstrap_lower_95"],
                name=f"{control} bootstrap lower 95",
            ),
            "per_scene_delta": retained_per_scene_delta,
            "positive_family_count": positive_family_count,
            "family_deltas": retained_family_deltas,
        }
    return retained


def observation_v19(
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
    cached = getattr(runtime, "causal_comparisons_v19", None)
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V19 scorer did not retain numeric causal comparisons")
    comparisons = _validate_causal_comparisons_v19(cached[update])
    if "causal_comparisons" in observed:
        raise RuntimeError("V19 inherited observation unexpectedly supplied comparisons")
    return {**observed, "causal_comparisons": comparisons}


def evaluate_update400_gate_v19(
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
    return {
        **decision,
        "schema": f"{SCHEMA_PREFIX}_update400_falsification_gate_v1",
    }


def validate_terminal_accounting_v19(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    _assert_inherited_accounting_registry_v20()
    if terminal_update not in _engine.TERMINAL_UPDATES:
        raise ValueError("V19 terminal update must be exactly 400 or 1000")
    value = _receipt_mapping_v19(accounting, name="V19 terminal accounting")
    expected = _expected_accounting_v19(terminal_update)
    if (
        set(value) != set(expected)
        or any(type(item) is not int for item in value.values())
        or value != expected
    ):
        raise RuntimeError("V19 terminal accounting is inconsistent with the frozen cap")
    if (
        value["updates"] > _engine.MAXIMUM_UPDATES
        or value["presentations"] > _engine.MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V19 scientific cap was exceeded")
    return expected


_engine.validate_bound_sources_v13 = validate_bound_sources_v19
_engine.validate_model_api_v13 = validate_model_api_v19
_engine.validate_training_api_v13 = validate_training_api_v19
_engine._validate_update_integrity_v13 = validate_update_integrity_v19
_engine._observation_v13 = observation_v19
_engine.evaluate_update400_gate_v13 = evaluate_update400_gate_v19
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v19

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

validate_content_bound_v19 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v19 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v19 = _engine.execution_denial_receipt_v13
reserve_attempt_v19 = _engine.reserve_attempt_v13
terminalize_failure_v19 = _engine.terminalize_failure_v13
evaluate_final_gate_v19 = _engine.evaluate_final_gate_v13
validate_schedule_v19 = _engine.validate_schedule_v13
validate_attempt_reservation_v19 = _engine.validate_attempt_reservation_v13
run_future_authorized_engine_v19 = _engine.run_future_authorized_engine_v13
execute_v19 = _engine.execute_v13
flatten_physical_metrics_v19 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v19 = _engine.registered_wrong_rgb_mapping_v13

# Compatibility names consumed by the private V13 custody lifecycle/runtime.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v19
validate_bound_sources_v13 = validate_bound_sources_v19
validate_model_api_v13 = validate_model_api_v19
validate_training_api_v13 = validate_training_api_v19
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v19
)
execution_denial_receipt_v13 = execution_denial_receipt_v19
reserve_attempt_v13 = reserve_attempt_v19
terminalize_failure_v13 = terminalize_failure_v19
flatten_physical_metrics_v13 = flatten_physical_metrics_v19
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v19
evaluate_update400_gate_v13 = evaluate_update400_gate_v19
evaluate_final_gate_v13 = evaluate_final_gate_v19
validate_schedule_v13 = validate_schedule_v19
validate_attempt_reservation_v13 = validate_attempt_reservation_v19
run_future_authorized_engine_v13 = run_future_authorized_engine_v19
validate_terminal_accounting_v13 = validate_terminal_accounting_v19
execute_v13 = execute_v19


def private_adapter_receipt_v19() -> dict[str, Any]:
    _assert_inherited_accounting_registry_v20()
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v18_executor_adapter_v1",
        "base_executor": str(V18_EXECUTOR_PATH.relative_to(ROOT)),
        "base_executor_commit": V18_EXECUTOR_COMMIT,
        "base_executor_file_sha256": V18_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V18_EXECUTOR_BYTE_COUNT,
        "public_v18_was_loaded_before_adapter": _PUBLIC_V18_WAS_LOADED_BEFORE_ADAPTER,
        "public_v18_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V18_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v19_integrity_replacement_v1_preregistration_commit": (
            V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_COMMIT
        ),
        "v19_integrity_replacement_v1_terminal_failure_result_commit": (
            V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "original_v19_preregistration_commit": (
            ORIGINAL_V19_PREREGISTRATION_COMMIT
        ),
        "v19_terminal_failure_result_commit": (
            V19_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "v18_scientific_result_commit": V18_SCIENTIFIC_RESULT_COMMIT,
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "accounting_multipliers": dict(ACCOUNTING_MULTIPLIERS_V19),
        "factual_successor_parameter_tensor_count": (
            FACTUAL_SUCCESSOR_PARAMETER_TENSOR_COUNT
        ),
        "factual_successor_parameter_count": FACTUAL_SUCCESSOR_PARAMETER_COUNT,
        "inherited_accounting_registry": dict(
            INHERITED_ACCOUNTING_MULTIPLIERS_V13
        ),
        "extended_accounting_is_local": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V20 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v19(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
