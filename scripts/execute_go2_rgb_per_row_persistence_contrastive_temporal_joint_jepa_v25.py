#!/usr/bin/env python3
"""Denied-by-default V25 executor adapter over frozen V24.

V24 remains authoritative for model/evaluator/gate/custody behavior.  V25
validates the replacement P25 receipts and owns one science-neutral,
write-only recovery publication after a passed update-400 gate.  No recovery
reader or resume path exists here, and this source shell grants no execution
authority.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V24_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24.py"
)
V24_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "2b6178a4d876dc17c45fb340a4ab03ee302649b0"
)
V24_EXECUTOR_FILE_SHA256 = (
    "87ee973802e87cc5e3d98aeb14a8c08f02c92f50649d923000a36b51f0e8f03e"
)
V24_EXECUTOR_BYTE_COUNT = 35_114
V24_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24"
)
PRIVATE_V24_MODULE_NAME = f"{__name__}.__private_v24_executor"
_PUBLIC_V24_WAS_LOADED_BEFORE_ADAPTER = V24_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25"
)
PREREGISTRATION_COMMIT = "f00e20df3b429f9242516ac38f67fea587e04b22"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "b9ce16b251415c50cb643daad919699c32965e23ddcd77d22bb3b69334f8b299"
)
PREREGISTRATION_BYTE_COUNT = 18_965
V24_SCIENTIFIC_RESULT_COMMIT = "2824c80c54fc7502b1413b3371fc87c9206f82a2"
V24_SCIENTIFIC_RESULT_PATH = (
    "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24_scientific_result_2026-07-30.json"
)
V24_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "f901d49eb9db0c39a068e67496b0b1cdaec954c9238edb40648140b924894e48"
)
V24_SCIENTIFIC_RESULT_BYTE_COUNT = 22_361
V24_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "0349f41da529b0c8658bf14ae51d85892a6f21fb461a281a9e157c7e7ff571dc"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25/"
    "attempt_v1"
)
MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
RECOVERY_STATE_RELATIVE_PATH = "recovery/update_400_training_state.pt"
RECOVERY_BINDING_RELATIVE_PATH = (
    "recovery/update_400_training_state.binding.json"
)
RECOVERY_UPDATE = 400
RECOVERY_NEXT_UPDATE = 401
RECOVERY_NEXT_SCHEDULE_POSITION = 6_400
P25_VECTOR_NAMES = (
    "prediction_energy_per_row",
    "persistence_energy_per_row",
    "gap_per_row",
    "row_loss_per_row",
)
P25_STATISTIC_STEMS = (
    "prediction_energy",
    "persistence_energy",
    "gap",
    "row_loss",
)
P25_DIAGNOSTIC_NAMES = (
    *P25_VECTOR_NAMES,
    "mechanism",
    *(f"{stem}_{suffix}" for stem in P25_STATISTIC_STEMS for suffix in (
        "count", "sum", "mean", "minimum", "maximum"
    )),
    "negative_gap_count",
    "negative_gap_fraction",
    "legacy_global_ratio_per_microbatch",
    "legacy_global_ratio_count",
    "legacy_global_ratio_sum",
    "legacy_global_ratio_mean",
    "legacy_global_ratio_minimum",
    "legacy_global_ratio_maximum",
    "log2_normalizer",
    "softplus_beta",
    "softplus_threshold",
    "denominator_used",
)


def _load_private_v24_executor() -> ModuleType:
    if V24_EXECUTOR_PATH.is_symlink() or not V24_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("frozen V24 executor source is absent or not regular")
    source = V24_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V24_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V24_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V24 executor source binding changed")
    if PRIVATE_V24_MODULE_NAME in sys.modules:
        raise RuntimeError("private V24 executor module name is already occupied")
    module = ModuleType(PRIVATE_V24_MODULE_NAME)
    module.__file__ = str(V24_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V24_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V24_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V24_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V24_MODULE_NAME)
    return module


_v24 = _load_private_v24_executor()
if (
    not _v24.SCHEMA_PREFIX.endswith("joint_jepa_v24")
    or _v24.MAXIMUM_UPDATES != 1_000
    or _v24.MAXIMUM_PRESENTATIONS != 16_000
    or _v24.CURRENT_EXECUTION_AUTHORIZED is not False
):
    raise RuntimeError("frozen V24 executor defaults changed")

_engine = _v24._engine
_original_validate_bound_sources = _v24.validate_bound_sources_v24
_original_validate_model_api = _v24.validate_model_api_v24
_original_validate_training_api = _v24.validate_training_api_v24
_original_validate_microbatches = _v24.validate_microbatches_for_engine_v24
_original_validate_update_integrity = _v24.validate_update_integrity_v24
_original_observation = _v24.observation_v24
_original_validate_terminal_accounting = _v24.validate_terminal_accounting_v24
TRAINING_REQUIRED_BATCH_KEYS_V24 = tuple(_v24.TRAINING_REQUIRED_BATCH_KEYS_V24)
TRAINING_REQUIRED_BATCH_KEYS_V25 = TRAINING_REQUIRED_BATCH_KEYS_V24
ACCOUNTING_MULTIPLIERS_V25 = dict(_v24.ACCOUNTING_MULTIPLIERS_V24)

PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME = (
    _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT = (
    _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT = (
    _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT
)
PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT = (
    _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT
)
PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT = (
    _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT
)
PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES = (
    _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES
)
PROTECTED_PREDICTOR_CORE_PARAMETER_SIZES = (
    _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_SIZES
)
SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES = _v24.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES
MEAN_LOSS_NAMES = tuple(_v24.MEAN_LOSS_NAMES)

_bound_parent_sources = dict(_v24.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V24_SCIENTIFIC_RESULT_PATH: (
            V24_SCIENTIFIC_RESULT_FILE_SHA256,
            V24_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
)
for _module in (_v24, _v24._v23._base, _v24._v23, _engine):
    _module.SCHEMA_PREFIX = SCHEMA_PREFIX
    _module.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
    _module.PREREGISTRATION_PATH = PREREGISTRATION_PATH
    _module.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
    _module.BOUND_PARENT_SOURCES = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V25 per-row persistence-contrastive execution is denied until recursive "
    "source closure, independent review, narrow certification, and one-shot authority"
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


def _expected_accounting_v25(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS_V25.items()
    }


def validate_bound_sources_v25(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _original_validate_bound_sources(repository_root, selected)


def validate_model_api_v25(module: Any) -> dict[str, Any]:
    receipt = _original_validate_model_api(module)
    if receipt.get("model_class") != MODEL_CLASS_NAME:
        raise RuntimeError("V25 did not retain the exact V18 model class")
    return receipt


def validate_training_api_v25(module: Any) -> dict[str, Any]:
    inherited = dict(_original_validate_training_api(module))
    for name in (
        "JointTrainingAccountingV25",
        "JointUpdateResultV25",
        "PerRowPersistenceContrastiveTemporalTermsV25",
    ):
        if not isinstance(getattr(module, name, None), type):
            raise RuntimeError(f"V25 training type is absent: {name}")
    for name in (
        "joint_training_update_v25",
        "validate_accounting_v25",
        "per_row_persistence_contrastive_temporal_loss_v25",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"V25 training callable is absent: {name}")
    if (
        tuple(getattr(module, "REQUIRED_BATCH_KEYS_V25", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V25
        or tuple(getattr(module, "REQUIRED_BATCH_KEYS_V24", ()))
        != TRAINING_REQUIRED_BATCH_KEYS_V24
        or getattr(module, "PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24", None)
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
    ):
        raise RuntimeError("V25 training API or frozen batch schema changed")
    inherited.pop("required_batch_key_count_v24", None)
    return {
        **inherited,
        "required_batch_key_count_v25": len(TRAINING_REQUIRED_BATCH_KEYS_V25),
        "temporal_objective": "P25_per_row_softplus_energy_gap_over_log2",
        "new_batch_fields_over_v24": 0,
        "extra_predictor_forwards": 0,
        "legacy_global_ratio_is_diagnostic_only": True,
        "j24_bit_identical_to_v24": True,
    }


def validate_microbatches_for_engine_v25(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != _engine.MICROBATCHES_PER_UPDATE or any(
        type(batch) is not dict or tuple(batch) != TRAINING_REQUIRED_BATCH_KEYS_V25
        for batch in microbatches
    ):
        raise PermissionError("V25 engine microbatch schema changed")
    _original_validate_microbatches(runtime, model, microbatches)


def _stable_softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _validate_p25_diagnostics(value: Any) -> dict[str, Any]:
    diagnostics = _receipt_mapping(value, name="V25 P25 diagnostics")
    if set(diagnostics) != set(P25_DIAGNOSTIC_NAMES):
        raise RuntimeError("V25 P25 diagnostic fields changed")
    vectors: dict[str, tuple[float, ...]] = {}
    for name in P25_VECTOR_NAMES:
        raw = diagnostics[name]
        if not isinstance(raw, (list, tuple)) or len(raw) != 16:
            raise RuntimeError(f"V25 {name} must contain 16 ordered rows")
        vectors[name] = tuple(
            _finite(item, name=f"V25 {name}") for item in raw
        )
    legacy_raw = diagnostics["legacy_global_ratio_per_microbatch"]
    if not isinstance(legacy_raw, (list, tuple)) or len(legacy_raw) != 4:
        raise RuntimeError("V25 legacy ratio must contain four microbatches")
    legacy = tuple(_finite(item, name="V25 legacy ratio") for item in legacy_raw)
    if (
        diagnostics["mechanism"] != "per_row_persistence_contrastive_temporal_v1"
        or any(diagnostics[f"{stem}_count"] != 16 for stem in P25_STATISTIC_STEMS)
        or diagnostics["legacy_global_ratio_count"] != 4
        or diagnostics["negative_gap_count"]
        != sum(value < 0.0 for value in vectors["gap_per_row"])
        or diagnostics["denominator_used"] is not False
        or not math.isclose(
            _finite(diagnostics["log2_normalizer"], name="V25 log2 normalizer"),
            math.log(2.0),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or _finite(diagnostics["softplus_beta"], name="V25 softplus beta") != 1.0
        or _finite(
            diagnostics["softplus_threshold"], name="V25 softplus threshold"
        )
        != 20.0
    ):
        raise RuntimeError("V25 P25 constants or counts changed")
    prediction = vectors["prediction_energy_per_row"]
    persistence = vectors["persistence_energy_per_row"]
    gaps = vectors["gap_per_row"]
    row_losses = vectors["row_loss_per_row"]
    if any(value < 0.0 for value in (*prediction, *persistence, *row_losses)):
        raise RuntimeError("V25 energy or row loss is negative")
    for index, (predicted, baseline, gap, row_loss) in enumerate(
        zip(prediction, persistence, gaps, row_losses, strict=True)
    ):
        if not math.isclose(gap, predicted - baseline, rel_tol=2e-6, abs_tol=2e-6):
            raise RuntimeError(f"V25 row {index} gap equation changed")
        expected_loss = _stable_softplus(gap) / math.log(2.0)
        if not math.isclose(row_loss, expected_loss, rel_tol=2e-6, abs_tol=2e-6):
            raise RuntimeError(f"V25 row {index} softplus equation changed")
    vector_by_stem = dict(zip(P25_STATISTIC_STEMS, vectors.values(), strict=True))
    for stem, items in vector_by_stem.items():
        expected = {
            "sum": sum(items),
            "mean": sum(items) / len(items),
            "minimum": min(items),
            "maximum": max(items),
        }
        for suffix, expected_value in expected.items():
            observed = _finite(
                diagnostics[f"{stem}_{suffix}"],
                name=f"V25 {stem} {suffix}",
            )
            if not math.isclose(observed, expected_value, rel_tol=2e-6, abs_tol=2e-6):
                raise RuntimeError(f"V25 {stem} {suffix} changed")
    expected_negative_fraction = diagnostics["negative_gap_count"] / 16.0
    if not math.isclose(
        _finite(
            diagnostics["negative_gap_fraction"],
            name="V25 negative gap fraction",
        ),
        expected_negative_fraction,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("V25 negative gap fraction changed")
    expected_legacy = []
    for start in range(0, 16, 4):
        predicted_mean = sum(prediction[start : start + 4]) / 4.0
        persistence_mean = sum(persistence[start : start + 4]) / 4.0
        expected_legacy.append(predicted_mean / max(persistence_mean, 1e-6))
    if any(
        not math.isclose(observed, expected, rel_tol=2e-6, abs_tol=2e-6)
        for observed, expected in zip(legacy, expected_legacy, strict=True)
    ):
        raise RuntimeError("V25 detached legacy ratio changed")
    for suffix, expected in (
        ("sum", sum(legacy)),
        ("mean", sum(legacy) / 4.0),
        ("minimum", min(legacy)),
        ("maximum", max(legacy)),
    ):
        if not math.isclose(
            _finite(
                diagnostics[f"legacy_global_ratio_{suffix}"],
                name=f"V25 legacy ratio {suffix}",
            ),
            expected,
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise RuntimeError("V25 legacy ratio summary changed")
    return {**diagnostics, **vectors, "legacy_global_ratio_per_microbatch": legacy}


def validate_update_integrity_v25(
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
    diagnostics = _validate_p25_diagnostics(
        result.per_row_persistence_contrastive_diagnostics
    )
    p25 = _finite(result.mean_losses["P"], name="V25 mean P25")
    if not math.isclose(
        p25,
        float(diagnostics["row_loss_mean"]),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V25 P slot disagrees with per-row losses")
    receipt.pop("v24_predictor_core_protected_survival_output", None)
    receipt["per_row_persistence_contrastive_diagnostics"] = diagnostics
    receipt["v25_per_row_persistence_contrastive_temporal"] = {
        "temporal_loss_slot": "P",
        "mechanism": "mean_softplus_row_energy_gap_over_log2",
        "row_count": 16,
        "microbatch_count": 4,
        "legacy_ratio_diagnostic_only": True,
        "cross_row_normalizer_used": False,
        "j24_bit_identical_to_v24": True,
        "predictor_core_gradient_from_inherited_joint": True,
        "predictor_core_gradient_from_j24": False,
        "target_gradient_from_p25": False,
        "passed": True,
    }
    cached = getattr(runtime, "per_row_persistence_contrastive_diagnostics_v25", None)
    if cached is None:
        cached = {}
        runtime.per_row_persistence_contrastive_diagnostics_v25 = cached
    if type(cached) is not dict or update in cached:
        raise RuntimeError("V25 P25 diagnostic cache is not one-shot")
    cached[update] = diagnostics
    if "v24_predictor_core_protected_survival_output" in receipt:
        raise RuntimeError("V25 receipt leaked private V24 mechanism identity")
    return receipt


def observation_v25(
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
        observed["per_row_persistence_contrastive_diagnostics"] = None
        return observed
    cached = getattr(runtime, "per_row_persistence_contrastive_diagnostics_v25", None)
    if not isinstance(cached, Mapping) or update not in cached:
        raise RuntimeError("V25 observation lacks current-update P25 diagnostics")
    observed["per_row_persistence_contrastive_diagnostics"] = dict(cached[update])
    return observed


def validate_terminal_accounting_v25(
    accounting: Any,
    *,
    terminal_update: int,
) -> dict[str, int]:
    value = _receipt_mapping(accounting, name="V25 terminal accounting")
    expected = _expected_accounting_v25(terminal_update)
    if value != expected or any(type(item) is not int for item in value.values()):
        raise RuntimeError("V25 terminal accounting is inconsistent with the cap")
    _original_validate_terminal_accounting(
        value,
        terminal_update=terminal_update,
    )
    return expected


def _clone_recovery_value_v25(torch: Any, np: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            key: _clone_recovery_value_v25(torch, np, item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_recovery_value_v25(torch, np, item) for item in value)
    if isinstance(value, list):
        return [_clone_recovery_value_v25(torch, np, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"unsupported V25 recovery value: {type(value).__name__}")


def _tree_sha256_v25(torch: Any, np: Any, value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            for key in keys:
                visit(key)
                visit(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                visit(child)
        elif isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                visit(child)
        elif item is None:
            digest.update(b"none\0")
        elif isinstance(item, bool):
            digest.update(b"bool\0" + (b"1" if item else b"0"))
        elif isinstance(item, int):
            digest.update(b"int\0" + str(item).encode("ascii") + b"\0")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise FloatingPointError("V25 recovery state contains a nonfinite float")
            digest.update(b"float\0" + item.hex().encode("ascii") + b"\0")
        elif isinstance(item, str):
            digest.update(b"str\0" + item.encode("utf-8") + b"\0")
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + item + b"\0")
        else:
            raise TypeError(f"unsupported V25 identity value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _capture_training_state_v25(
    runtime: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
) -> dict[str, Any]:
    torch = runtime.torch
    np = runtime.np
    accounting_value = _receipt_mapping(accounting, name="V25 recovery accounting")
    if accounting_value != _expected_accounting_v25(RECOVERY_UPDATE):
        raise RuntimeError("V25 recovery accounting is not at update 400")
    ema_count = int(model.ema_update_count.item())
    if ema_count != RECOVERY_UPDATE:
        raise RuntimeError("V25 recovery EMA counter is not at update 400")
    named_parameters = tuple(model.named_parameters())
    state = {
        "model_state_dict": _clone_recovery_value_v25(
            torch, np, model.state_dict()
        ),
        "optimizer_state_dict": _clone_recovery_value_v25(
            torch, np, optimizer.state_dict()
        ),
        "parameter_gradients": {
            name: (
                None
                if parameter.grad is None
                else _clone_recovery_value_v25(torch, np, parameter.grad)
            )
            for name, parameter in named_parameters
        },
        "module_training_modes": {
            name: bool(module.training) for name, module in model.named_modules()
        },
        "ema_update_count": ema_count,
        "accounting": accounting_value,
        "completed_update": RECOVERY_UPDATE,
        "next_update": RECOVERY_NEXT_UPDATE,
        "next_schedule_position": RECOVERY_NEXT_SCHEDULE_POSITION,
        "rng_states": {
            "python": random.getstate(),
            "numpy": _clone_recovery_value_v25(torch, np, np.random.get_state()),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "visible_rocm_devices": tuple(
                value.clone() for value in torch.cuda.get_rng_state_all()
            ),
        },
    }
    state["training_state_sha256"] = _tree_sha256_v25(torch, np, state)
    return state


def _recovery_identity_view_v25(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "training_state_sha256"}


def _trace_prefix_identity_v25(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = b"".join(
        _engine._canonical_json_bytes(_engine._content_bound(row)) + b"\n"
        for row in trace
    )
    return {
        "row_count": len(trace),
        "byte_count": len(raw),
        "file_prefix_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _publish_update400_recovery_v25(
    *,
    authority: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    gate_decision: Mapping[str, Any],
    metric_bindings: Sequence[Mapping[str, Any]],
    trace_prefix_identity: Mapping[str, Any],
    publication_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        type(gate_decision) is not dict
        or gate_decision.get("passed") is not True
        or gate_decision.get("action") != "CONTINUE_TO_UPDATE_1000"
    ):
        raise PermissionError("V25 recovery writer requires a passed update-400 gate")
    metric_identities = tuple(dict(value) for value in metric_bindings)
    if len(metric_identities) != 3 or publication_state:
        raise RuntimeError("V25 recovery metric or publication state changed")
    before = _capture_training_state_v25(runtime, model, optimizer, accounting)
    before_sha = before["training_state_sha256"]
    authority_sha = hashlib.sha256(
        _engine._canonical_json_bytes(authority)
    ).hexdigest()
    gate_sha = _engine._canonical_value_sha256(gate_decision)
    payload = {
        "schema": f"{SCHEMA_PREFIX}_update400_training_state_v1",
        "recovery_only_not_development_checkpoint": True,
        "completed_update": RECOVERY_UPDATE,
        "next_update": RECOVERY_NEXT_UPDATE,
        "next_schedule_position": RECOVERY_NEXT_SCHEDULE_POSITION,
        "model_state_dict": before["model_state_dict"],
        "optimizer_state_dict": before["optimizer_state_dict"],
        "parameter_gradients": before["parameter_gradients"],
        "module_training_modes": before["module_training_modes"],
        "ema_update_count": before["ema_update_count"],
        "accounting": before["accounting"],
        "rng_states": before["rng_states"],
        "training_state_sha256": before_sha,
        "scientific_identity": {
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "frozen_source_and_review_commit": authority[
                "frozen_source_and_review_commit"
            ],
            "recursive_source_closure_manifest_sha256": authority[
                "recursive_source_closure_manifest_sha256"
            ],
            "independent_source_review_sha256": authority[
                "independent_source_review_sha256"
            ],
            "clean_export_certification_sha256": authority[
                "clean_export_certification_sha256"
            ],
            "execution_binding_commit": authority["execution_binding_commit"],
            "authority_sha256": authority_sha,
            "runtime_input_bindings": dict(authority["runtime_inputs"]),
            "metric_bindings": metric_identities,
            "update400_gate_decision_sha256": gate_sha,
            "trace_prefix": dict(trace_prefix_identity),
        },
        "dataset_or_rgb_payload_included": False,
        "recovery_state_opened": False,
        "resume_authorized": False,
    }
    stream = io.BytesIO()
    runtime.torch.save(payload, stream)
    raw = stream.getvalue()
    if not raw:
        raise RuntimeError("V25 recovery serialization is empty")
    snapshot_binding = _engine._publisher_bytes_v13(
        publisher,
        RECOVERY_STATE_RELATIVE_PATH,
        raw,
    )
    publication_state["snapshot"] = snapshot_binding
    after = _capture_training_state_v25(runtime, model, optimizer, accounting)
    after_sha = after["training_state_sha256"]
    if before_sha != after_sha or _tree_sha256_v25(
        runtime.torch,
        runtime.np,
        _recovery_identity_view_v25(before),
    ) != _tree_sha256_v25(
        runtime.torch,
        runtime.np,
        _recovery_identity_view_v25(after),
    ):
        raise RuntimeError("V25 recovery write mutated live training or RNG state")
    binding_value, binding_artifact = _engine._publisher_json_v13(
        publisher,
        RECOVERY_BINDING_RELATIVE_PATH,
        {
            "schema": f"{SCHEMA_PREFIX}_update400_training_state_binding_v1",
            "completed_update": RECOVERY_UPDATE,
            "next_update": RECOVERY_NEXT_UPDATE,
            "next_schedule_position": RECOVERY_NEXT_SCHEDULE_POSITION,
            "update400_gate_decision_sha256": gate_sha,
            "metric_bindings": metric_identities,
            "trace_prefix": dict(trace_prefix_identity),
            "snapshot": snapshot_binding,
            "training_state_sha256_before_write": before_sha,
            "training_state_sha256_after_write": after_sha,
            "training_state_and_rng_unchanged": True,
            "snapshot_file_read_count": 0,
            "development_checkpoint": False,
            "resume_authorized": False,
        },
    )
    completed_binding = {
        **binding_artifact,
        "content_sha256": binding_value["content_sha256"],
    }
    publication_state["binding"] = completed_binding
    return snapshot_binding, completed_binding


def run_future_authorized_engine_v25(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    """Run V24's lifecycle with the preregistered write-only V25 recovery seam."""

    stage = "validate_post_reservation_authority"
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    trace_binding: dict[str, Any] | None = None
    terminal_access_binding: dict[str, Any] | None = None
    terminal_access_content_sha256: str | None = None
    recovery_snapshot_binding: dict[str, Any] | None = None
    recovery_binding_artifact: dict[str, Any] | None = None
    recovery_publication_state: dict[str, Any] = {}
    terminal_published = False
    validated_authority: dict[str, Any] | None = None

    def publish_trace() -> dict[str, Any]:
        nonlocal trace_binding
        if trace_binding is None:
            raw = b"".join(
                _engine._canonical_json_bytes(_engine._content_bound(row)) + b"\n"
                for row in trace
            )
            trace_binding = _engine._publisher_bytes_v13(
                publisher, _engine.TRACE_RELATIVE_PATH, raw
            )
        return trace_binding

    def publish_terminal_access(receipt: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal terminal_access_binding, terminal_access_content_sha256
        if terminal_access_binding is None:
            value, terminal_access_binding = _engine._publisher_json_v13(
                publisher,
                _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH,
                {
                    "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
                    "receipt": dict(receipt),
                },
            )
            terminal_access_content_sha256 = value["content_sha256"]
        return terminal_access_binding

    try:
        validated_authority = _engine.validate_future_execution_prerequisites_v13(
            authority
        )
        validated_reservation = _engine.validate_attempt_reservation_v13(reservation)
        if validated_reservation["authority_sha256"] != hashlib.sha256(
            _engine._canonical_json_bytes(validated_authority)
        ).hexdigest():
            raise PermissionError("V25 reservation does not bind supplied authority")

        stage = "validate_deferred_runtime_and_schedule"
        schedule_receipt = _engine.validate_schedule_v13(
            runtime.schedule,
            train_pair_count=int(runtime.train_pair_count),
        )
        stage = "initialize_n320_v25_model_optimizer"
        model, optimizer, initialization = runtime.initialize_model_v13()
        initialization_receipt = _engine._validate_initialization_v13(
            runtime, model, initialization
        )
        initial_structural = _engine._derive_initial_structural_integrity_v13(
            runtime, model
        )
        access = _engine._validate_access_receipt_v13(runtime.access_receipt_v13())
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "initialized",
                "update": 0,
                "initialization": initialization_receipt,
                "structural_integrity": initial_structural,
                "schedule": schedule_receipt,
                "access_receipt_sha256": _engine._canonical_value_sha256(access),
            }
        )

        stage = "observe_update_0"
        observations: dict[int, dict[str, Any]] = {}
        observations[0] = _engine._observation_v13(
            runtime,
            model,
            update=0,
            integrity_pass=bool(initial_structural["passed"]),
        )
        _, binding = _engine._publisher_json_v13(
            publisher, _engine.METRIC_RELATIVE_PATHS[0], observations[0]
        )
        metric_bindings.append(binding)

        accounting: Any = None
        structural_pass = bool(observations[0]["integrity_pass"])
        terminal_update: int | None = None
        scientific_decision: dict[str, Any] | None = None
        for update in range(1, _engine.MAXIMUM_UPDATES + 1):
            stage = f"train_update_{update}"
            start = (update - 1) * _engine.PRESENTATIONS_PER_UPDATE
            indices = list(
                runtime.schedule[start : start + _engine.PRESENTATIONS_PER_UPDATE]
            )
            if len(indices) != _engine.PRESENTATIONS_PER_UPDATE:
                raise PermissionError("V25 frozen schedule ended early")
            microbatches = runtime.build_microbatches_v13(indices, update=update)
            _engine._validate_microbatches_for_engine_v13(runtime, model, microbatches)
            result = runtime.training_module.joint_training_update_v13(
                model,
                optimizer,
                microbatches,
                accounting=accounting,
            )
            accounting = result.accounting
            integrity = _engine._validate_update_integrity_v13(
                runtime,
                model,
                result,
                update=update,
                access_receipt=runtime.access_receipt_v13(),
            )
            structural_pass = structural_pass and bool(integrity["passed"])
            trace.append(
                {
                    "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                    "event": "optimizer_ema_update",
                    **integrity,
                }
            )

            if update not in (100, 400, 1_000):
                continue
            stage = f"observe_update_{update}"
            observations[update] = _engine._observation_v13(
                runtime,
                model,
                update=update,
                integrity_pass=structural_pass,
            )
            structural_pass = bool(observations[update]["integrity_pass"])
            _, binding = _engine._publisher_json_v13(
                publisher,
                _engine.METRIC_RELATIVE_PATHS[update],
                observations[update],
            )
            metric_bindings.append(binding)
            if update == RECOVERY_UPDATE:
                controls = observations[RECOVERY_UPDATE]["controls"]
                scientific_decision = _engine.evaluate_update400_gate_v13(
                    observations[100]["physical"],
                    observations[RECOVERY_UPDATE]["physical"],
                    controls,
                    integrity_pass=structural_pass,
                    matched_update400_thresholds=_engine.MATCHED_UPDATE400_THRESHOLDS,
                )
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update400_control",
                        "update": RECOVERY_UPDATE,
                        "decision": scientific_decision,
                    }
                )
                if not scientific_decision["passed"]:
                    terminal_update = RECOVERY_UPDATE
                    break
                stage = "publish_update400_recovery_snapshot"
                trace_prefix = _trace_prefix_identity_v25(trace)
                recovery_snapshot_binding, recovery_binding_artifact = (
                    _publish_update400_recovery_v25(
                        authority=validated_authority,
                        runtime=runtime,
                        publisher=publisher,
                        model=model,
                        optimizer=optimizer,
                        accounting=accounting,
                        gate_decision=scientific_decision,
                        metric_bindings=metric_bindings,
                        trace_prefix_identity=trace_prefix,
                        publication_state=recovery_publication_state,
                    )
                )
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update400_recovery_written",
                        "update": RECOVERY_UPDATE,
                        "snapshot": recovery_snapshot_binding,
                        "binding": recovery_binding_artifact,
                        "trace_prefix": trace_prefix,
                        "recovery_state_opened": False,
                        "resume_authorized": False,
                    }
                )
            elif update == 1_000:
                scientific_decision = _engine.evaluate_final_gate_v13(
                    observations[1_000]["v12_gate"],
                    observations[1_000]["physical"],
                    integrity_pass=structural_pass,
                )
                terminal_update = 1_000
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update1000_final_gate",
                        "update": 1_000,
                        "decision": scientific_decision,
                    }
                )

        if terminal_update not in _engine.TERMINAL_UPDATES or scientific_decision is None:
            raise RuntimeError("V25 engine did not reach one frozen terminal update")
        terminal_accounting = _engine.validate_terminal_accounting_v13(
            accounting, terminal_update=terminal_update
        )
        terminal_access_reader = getattr(runtime, "terminal_access_receipt_v13", None)
        final_access = _engine._validate_access_receipt_v13(
            (
                terminal_access_reader()
                if callable(terminal_access_reader)
                else runtime.access_receipt_v13()
            ),
            terminal=True,
        )
        if final_access["runtime_data_root"] != validated_authority["runtime_data_root"]:
            raise PermissionError("V25 terminal rehash changed runtime data root")
        if final_access["source_root"] != validated_authority["certified_source_root"]:
            raise PermissionError("V25 terminal rehash changed certified source root")
        if final_access["runtime_fingerprint"] != validated_authority["runtime"]:
            raise PermissionError("V25 terminal rehash changed runtime stack")
        final_access_artifact = publish_terminal_access(final_access)
        trace_record = publish_trace()

        recovery_receipt = {
            "recovery_snapshot_published": recovery_snapshot_binding is not None,
            "recovery_snapshot": recovery_snapshot_binding,
            "recovery_binding": recovery_binding_artifact,
            "recovery_state_opened": False,
            "resume_authorized": False,
        }
        if not scientific_decision["passed"]:
            stage = "publish_terminal_scientific_failure"
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            failure_core = {
                "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                "status": (
                    "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"
                    if terminal_update == RECOVERY_UPDATE
                    else "FAIL_SCIENTIFIC_UPDATE1000_GATE_TERMINAL"
                ),
                "terminal_update": terminal_update,
                "decision": scientific_decision,
                "accounting": terminal_accounting,
                "metrics": metric_bindings,
                "trace": trace_record,
                **recovery_receipt,
                "access_receipt_sha256": _engine._canonical_value_sha256(final_access),
                "terminal_access_receipt": final_access_artifact,
                "terminal_access_receipt_content_sha256": terminal_access_content_sha256,
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
            }
            value, _ = _engine._publisher_json_v13(
                publisher, _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            terminal_published = True
            return value

        stage = "publish_pass1000_development_checkpoint"
        checkpoint_raw, checkpoint_core = _engine._serialize_development_checkpoint_v13(
            runtime, model, validated_authority
        )
        checkpoint_binding = _engine._publisher_bytes_v13(
            publisher,
            _engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH,
            checkpoint_raw,
        )
        checkpoint_core["checkpoint"] = checkpoint_binding
        checkpoint_value, checkpoint_metadata_binding = _engine._publisher_json_v13(
            publisher,
            _engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH,
            checkpoint_core,
        )
        if callable(getattr(runtime, "close_v13", None)):
            runtime.close_v13()
        stage = "publish_terminal_success"
        success_core = {
            "schema": f"{SCHEMA_PREFIX}_success_v1",
            "status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL",
            "terminal_update": 1_000,
            "decision": scientific_decision,
            "accounting": terminal_accounting,
            "metrics": metric_bindings,
            "trace": trace_record,
            **recovery_receipt,
            "checkpoint": checkpoint_binding,
            "checkpoint_metadata": checkpoint_metadata_binding,
            "checkpoint_metadata_content_sha256": checkpoint_value["content_sha256"],
            "access_receipt_sha256": _engine._canonical_value_sha256(final_access),
            "terminal_access_receipt": final_access_artifact,
            "terminal_access_receipt_content_sha256": terminal_access_content_sha256,
            "physical_adapter_preregistration_eligible": True,
            "probability_calibration_authorized": False,
            "probability_calibration_opened": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "held_out_authorized": False,
            "attempt_consumed": True,
            "retry_authorized": False,
        }
        value, _ = _engine._publisher_json_v13(
            publisher, _engine.SUCCESS_RELATIVE_PATH, success_core
        )
        terminal_published = True
        return value
    except BaseException as error:
        if terminal_published:
            raise
        try:
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            terminal_reader = getattr(runtime, "terminal_access_receipt_v13", None)
            exception_access = _engine._validate_access_receipt_v13(
                (
                    terminal_reader()
                    if callable(terminal_reader)
                    else runtime.access_receipt_v13()
                ),
                terminal=True,
            )
            if validated_authority is not None and (
                exception_access["runtime_data_root"]
                != validated_authority["runtime_data_root"]
                or exception_access["source_root"]
                != validated_authority["certified_source_root"]
                or exception_access["runtime_fingerprint"]
                != validated_authority["runtime"]
            ):
                raise PermissionError(
                    "V25 exception access receipt used an unbound source or data root"
                )
            exception_access_artifact = publish_terminal_access(exception_access)
            trace_record = publish_trace()
            failure_core = {
                "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
                "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "trace": trace_record,
                "recovery_snapshot_published": (
                    recovery_publication_state.get("snapshot") is not None
                ),
                "recovery_snapshot": recovery_publication_state.get("snapshot"),
                "recovery_binding": recovery_publication_state.get("binding"),
                "recovery_state_opened": False,
                "access_receipt_sha256": _engine._canonical_value_sha256(
                    exception_access
                ),
                "terminal_access_receipt": exception_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
                "resume_authorized": False,
            }
            value, _ = _engine._publisher_json_v13(
                publisher, _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            return value
        except BaseException:
            raise error


_engine.validate_bound_sources_v13 = validate_bound_sources_v25
_engine.validate_model_api_v13 = validate_model_api_v25
_engine.validate_training_api_v13 = validate_training_api_v25
_engine._validate_microbatches_for_engine_v13 = validate_microbatches_for_engine_v25
_engine._validate_update_integrity_v13 = validate_update_integrity_v25
_engine._observation_v13 = observation_v25
_engine.validate_terminal_accounting_v13 = validate_terminal_accounting_v25
_engine.run_future_authorized_engine_v13 = run_future_authorized_engine_v25

EXPECTED_RUNTIME_FINGERPRINT = _engine.EXPECTED_RUNTIME_FINGERPRINT
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
MODEL_REQUIRED_METHODS = _engine.MODEL_REQUIRED_METHODS
MODEL_REQUIRED_CONSTANTS = _engine.MODEL_REQUIRED_CONSTANTS
TRAINING_REQUIRED_FUNCTIONS = _engine.TRAINING_REQUIRED_FUNCTIONS
TRAINING_REQUIRED_BATCH_KEYS = TRAINING_REQUIRED_BATCH_KEYS_V25
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
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = _engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
SUCCESS_RELATIVE_PATH = _engine.SUCCESS_RELATIVE_PATH
SCIENTIFIC_FAILURE_RELATIVE_PATH = _engine.SCIENTIFIC_FAILURE_RELATIVE_PATH

validate_content_bound_v25 = _engine.validate_content_bound_v13
validate_future_execution_prerequisites_v25 = (
    _engine.validate_future_execution_prerequisites_v13
)
execution_denial_receipt_v25 = _engine.execution_denial_receipt_v13
reserve_attempt_v25 = _engine.reserve_attempt_v13
terminalize_failure_v25 = _engine.terminalize_failure_v13
flatten_physical_metrics_v25 = _engine.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v25 = _engine.registered_wrong_rgb_mapping_v13
evaluate_update400_gate_v25 = _engine.evaluate_update400_gate_v13
evaluate_final_gate_v25 = _engine.evaluate_final_gate_v13
validate_schedule_v25 = _engine.validate_schedule_v13
validate_attempt_reservation_v25 = _engine.validate_attempt_reservation_v13
execute_v25 = _engine.execute_v13

# Compatibility names consumed by the unchanged launcher/runtime.
_canonical_json_bytes = _engine._canonical_json_bytes
_write_immutable_json_v13 = _engine._write_immutable_json_v13
validate_content_bound_v13 = validate_content_bound_v25
validate_bound_sources_v13 = validate_bound_sources_v25
validate_model_api_v13 = validate_model_api_v25
validate_training_api_v13 = validate_training_api_v25
validate_future_execution_prerequisites_v13 = validate_future_execution_prerequisites_v25
execution_denial_receipt_v13 = execution_denial_receipt_v25
reserve_attempt_v13 = reserve_attempt_v25
terminalize_failure_v13 = terminalize_failure_v25
flatten_physical_metrics_v13 = flatten_physical_metrics_v25
registered_wrong_rgb_mapping_v13 = registered_wrong_rgb_mapping_v25
evaluate_update400_gate_v13 = evaluate_update400_gate_v25
evaluate_final_gate_v13 = evaluate_final_gate_v25
validate_schedule_v13 = validate_schedule_v25
validate_attempt_reservation_v13 = validate_attempt_reservation_v25
run_future_authorized_engine_v13 = run_future_authorized_engine_v25
validate_terminal_accounting_v13 = validate_terminal_accounting_v25
execute_v13 = execute_v25


def private_adapter_receipt_v25() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v24_executor_adapter_v1",
        "base_executor": str(V24_EXECUTOR_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": V24_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_executor_file_sha256": V24_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V24_EXECUTOR_BYTE_COUNT,
        "public_v24_was_loaded_before_adapter": _PUBLIC_V24_WAS_LOADED_BEFORE_ADAPTER,
        "public_v24_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V24_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v24_scientific_result_commit": V24_SCIENTIFIC_RESULT_COMMIT,
        "v24_scientific_result_content_sha256": V24_SCIENTIFIC_RESULT_CONTENT_SHA256,
        "model_class": MODEL_CLASS_NAME,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "temporal_objective": "P25_per_row_softplus_energy_gap_over_log2",
        "j24_bit_identical_to_v24": True,
        "new_batch_fields_over_v24": 0,
        "recovery_state_path": RECOVERY_STATE_RELATIVE_PATH,
        "recovery_binding_path": RECOVERY_BINDING_RELATIVE_PATH,
        "recovery_write_after_passed_update400_gate": True,
        "recovery_reader_implemented": False,
        "resume_implemented": False,
        "update400_and_update1000_gates_unchanged": True,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V25 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v25(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
