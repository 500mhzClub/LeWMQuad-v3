#!/usr/bin/env python3
"""Run the one-shot geometry-anchored two-mode event-delta JEPA V1 probe."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, NamedTuple, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
FROZEN_RIGID_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_two_mode_event_delta_runner_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH:
    raise PermissionError("two-mode event-delta runner path changed")
_RIGID = _source_module(
    "_lewm_two_mode_event_delta_frozen_rigid_runner",
    ROOT / FROZEN_RIGID_RUNNER_RELATIVE_PATH,
)
_V3 = _RIGID._V3
_V2 = _V3._V2
_BASE = _V2._V1

_tensor_state_sha256 = _RIGID._tensor_state_sha256
DeterministicWarningFailure = _RIGID.DeterministicWarningFailure
ROCM_GRID_SAMPLE_DETERMINISM_WARNING = (
    _RIGID.ROCM_GRID_SAMPLE_DETERMINISM_WARNING
)
canonicalize_rocm_determinism_warning = (
    _RIGID.canonicalize_rocm_determinism_warning
)
_is_allowed_rocm_determinism_warning = (
    _RIGID._is_allowed_rocm_determinism_warning
)
_finalize_deterministic_warnings = _RIGID._finalize_deterministic_warnings
_run_deterministic = _RIGID._run_deterministic

_BASE_PARAMETER_RECEIPT = _RIGID._BASE_PARAMETER_RECEIPT
_BASE_EXECUTE = _RIGID._BASE_EXECUTE
_BASE_BUILD_OPTIMIZER = _BASE._build_optimizer
_BASE_EVALUATE_OBSERVATION = _BASE._evaluate_observation
_BASE_TRAIN_PROBE = _BASE._train_probe
_BASE_LOAD_DEVELOPMENT_INPUTS = _BASE._load_development_inputs
_BASE_PUBLISH_JSON = _BASE._publish_json
_BASE_SEAL = _BASE._seal
_CONTRACT_PHASE_SWITCH = contract.evaluate_update_401_phase_switch


class _BalancedView(NamedTuple):
    changed: Any
    static: Any
    balanced: Any


_FROZEN_REFERENCES: dict[str, Any] = {}
_ACTIVE_OPTIMIZER: Any | None = None
_ACTIVE_SOURCE_BINDINGS: dict[str, str] = {}
_EVENT_ACCOUNTING: dict[str, Any] = {}
_PREDICTOR_PARAMETER_ROWS: list[tuple[str, Any]] = []
_OBSERVATION_LIVE: dict[str, Any] = {}


def _reset_event_runtime_state() -> None:
    global _ACTIVE_OPTIMIZER
    _FROZEN_REFERENCES.clear()
    _PREDICTOR_PARAMETER_ROWS.clear()
    _EVENT_ACCOUNTING.clear()
    _EVENT_ACCOUNTING.update({
        "training_microbatch_count": 0,
        "joint_combined_objective_evaluation_count": 0,
        "scheduled_pair_presentations_loaded": 0,
        "semantic_term_evaluation_count": 0,
        "online_encoder_lift_training_forward_count": 0,
        "semantic_head_training_forward_count": 0,
        "target_encoder_lift_training_forward_count": 0,
        "all_action_predictor_training_forward_count": 0,
        "context_swap_predictor_training_forward_count": 0,
        "event_persistence_term_evaluation_count": 0,
        "action_term_evaluation_count": 0,
        "target_term_evaluation_count": 0,
        "context_term_evaluation_count": 0,
        "action_embedding_dynamics_gradient_update_count": 0,
        "predictor_trunk_dynamics_gradient_update_count": 0,
        "event_mean_head_dynamics_gradient_update_count": 0,
        "event_logit_head_dynamics_gradient_update_count": 0,
    })
    _OBSERVATION_LIVE.clear()
    _ACTIVE_OPTIMIZER = None


_reset_event_runtime_state()


def _state_value_sha256(torch: Any, value: Any) -> str:
    """Canonical scalar-safe hash used only for read-only observer integrity."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(str(len(item)).encode("ascii") + b"\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")
        else:
            raise TypeError(f"unsupported observer-state value: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def _ordered_float64_sum(value: Any, *, initial: float = 0.0) -> float:
    """Accumulate float32 values as float64 in exact flattened row order."""

    total = float(initial)
    for item in value.detach().to(device="cpu").reshape(-1).tolist():
        total += float(item)
    if not math.isfinite(total):
        raise FloatingPointError("ordered float64 accumulation became nonfinite")
    return total


def _build_optimizer(runtime: Any, groups: Mapping[str, Sequence[Any]]) -> Any:
    global _ACTIVE_OPTIMIZER
    optimizer = _BASE_BUILD_OPTIMIZER(runtime, groups)
    original_step = optimizer.step

    def audited_step(*args: Any, **kwargs: Any) -> Any:
        rows = [
            (name, parameter, parameter.grad)
            for name, parameter in _PREDICTOR_PARAMETER_ROWS
        ]
        action_ok = False
        trunk_ok = False
        mean_ok = False
        logit_ok = False
        if any(gradient is not None for _name, _parameter, gradient in rows):
            if any(gradient is None for _name, _parameter, gradient in rows):
                raise FloatingPointError("predictor update has an absent gradient")

            def route(prefixes: tuple[str, ...]) -> bool:
                selected = [
                    gradient for name, _parameter, gradient in rows
                    if name.startswith(prefixes)
                ]
                return bool(
                    selected
                    and all(gradient is not None and gradient.isfinite().all() for gradient in selected)
                    and any(bool((gradient != 0).any()) for gradient in selected)
                )

            action_ok = route(("predictor.action_embedding.",))
            trunk_routes = (
                route(("predictor.input_projection.",)),
                route(("predictor.residual_blocks.0.conv1.",)),
                route(("predictor.residual_blocks.0.conv2.",)),
                route(("predictor.residual_blocks.1.conv1.",)),
                route(("predictor.residual_blocks.1.conv2.",)),
            )
            trunk_ok = all(trunk_routes)
            mean_ok = route(("predictor.event_mean_head.",))
            logit_ok = route(("predictor.event_logit_head.",))
        result = original_step(*args, **kwargs)
        if action_ok:
            _EVENT_ACCOUNTING["action_embedding_dynamics_gradient_update_count"] += 1
        if trunk_ok:
            _EVENT_ACCOUNTING["predictor_trunk_dynamics_gradient_update_count"] += 1
        if mean_ok:
            _EVENT_ACCOUNTING["event_mean_head_dynamics_gradient_update_count"] += 1
        if logit_ok:
            _EVENT_ACCOUNTING["event_logit_head_dynamics_gradient_update_count"] += 1
        return result

    optimizer.step = audited_step
    _ACTIVE_OPTIMIZER = optimizer
    return optimizer


def _semantic_terms(
    model_api: Any, model: Any, batch: Mapping[str, Any]
) -> dict[str, Any]:
    training_call = bool(model.training)
    if training_call:
        _EVENT_ACCOUNTING["scheduled_pair_presentations_loaded"] += int(
            batch["action_indices"].shape[0]
        )
    current_latent = model.encode_online(batch["current_rgb"])
    if training_call:
        _EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] += 1
    next_latent = model.encode_online(batch["next_rgb"])
    if training_call:
        _EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] += 1
    current_logits = model.semantic_logits_from_latent(current_latent)
    if training_call:
        _EVENT_ACCOUNTING["semantic_head_training_forward_count"] += 1
    next_logits = model.semantic_logits_from_latent(next_latent)
    if training_call:
        _EVENT_ACCOUNTING["semantic_head_training_forward_count"] += 1
    current_rows = model_api.final_class_macro_nll_per_row(
        current_logits, batch["current_labels"]
    )
    next_rows = model_api.final_class_macro_nll_per_row(
        next_logits, batch["next_labels"]
    )
    A = 0.5 * current_rows.mean() + 0.5 * next_rows.mean()
    if training_call:
        _EVENT_ACCOUNTING["semantic_term_evaluation_count"] += 1
    return {
        "current_latent": current_latent,
        "next_latent": next_latent,
        "current_logits": current_logits,
        "next_logits": next_logits,
        "A": A,
        "S": A / math.log(3.0),
    }


def _parameter_receipt(
    model: Any,
    contract_api: Any,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    groups, receipt = _BASE_PARAMETER_RECEIPT(model, contract_api)
    names = [
        name for name, _parameter in model.named_parameters()
        if name.startswith("predictor.")
    ]
    _PREDICTOR_PARAMETER_ROWS[:] = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith("predictor.")
    ]
    predictor = receipt["predictor"]
    if (
        names != list(contract.PREDICTOR_ORDERED_PARAMETER_NAMES)
        or predictor["parameter_count"] != contract.PREDICTOR_PARAMETER_COUNT
        or predictor["tensor_count"] != contract.PREDICTOR_PARAMETER_TENSOR_COUNT
    ):
        raise PermissionError("two-mode event-delta predictor inventory changed")
    return groups, receipt


def _normalize(model_api: Any, latent: Any) -> Any:
    value = model_api.normalize_latent_per_cell_v1(latent)
    if value.shape != latent.shape or value.dtype != latent.dtype:
        raise RuntimeError("per-cell latent normalization changed shape or dtype")
    return value


def _prediction_parts(value: Any) -> Any:
    try:
        _unused_mu = value.mu_event
        _unused_logit = value.event_logit
    except AttributeError as error:
        raise TypeError("event predictor did not return the registered pair") from error
    return value


def _select_prediction_with_torch(torch: Any, value: Any, actions: Any) -> Any:
    prediction = _prediction_parts(value)
    rows = torch.arange(actions.shape[0], device=actions.device)
    return type(prediction)(
        prediction.mu_event[rows, actions],
        prediction.event_logit[rows, actions],
    )


def _action_one_hot(torch: Any, actions: Any, *, dtype: Any) -> Any:
    return torch.nn.functional.one_hot(actions, num_classes=9).to(dtype=dtype)


def _event_cells(model_api: Any, target_delta: Any, prediction: Any) -> tuple[Any, Any, Any]:
    prediction = _prediction_parts(prediction)
    e0, e1 = model_api.event_delta_cell_energies_v1(target_delta, prediction)
    logit = prediction.event_logit
    if logit.ndim == 4:
        if logit.shape[1] != 1:
            raise RuntimeError("one-action event logit singleton channel changed")
        expected = target_delta.shape[0], target_delta.shape[-2], target_delta.shape[-1]
    elif logit.ndim == 5:
        if logit.shape[2] != 1:
            raise RuntimeError("all-action event logit singleton channel changed")
        expected = (
            target_delta.shape[0],
            logit.shape[1],
            target_delta.shape[-2],
            target_delta.shape[-1],
        )
    else:
        raise RuntimeError("event logit rank changed")
    mixed = model_api.two_mode_event_energy_v1(
        e0, e1, logit, float(_FROZEN_REFERENCES["T400"])
    )
    if tuple(mixed.shape) != tuple(expected):
        raise RuntimeError("event mixture cell shape changed")
    return e0, e1, mixed


def _balanced(cell_energy: Any, weight: Any) -> _BalancedView:
    if weight.ndim != 3 or cell_energy.ndim not in (3, 4):
        raise RuntimeError("balanced event energy rank changed")
    if cell_energy.shape[0] != weight.shape[0] or cell_energy.shape[-2:] != weight.shape[-2:]:
        raise RuntimeError("balanced event energy axes changed")
    changed_weight = weight if cell_energy.ndim == 3 else weight[:, None]
    static_weight = 1.0 - changed_weight
    changed_denominator = changed_weight.sum(dim=(-2, -1))
    static_denominator = static_weight.sum(dim=(-2, -1))
    if not bool(
        (changed_denominator > 1e-6).all()
        and (static_denominator > 1e-6).all()
    ):
        raise FloatingPointError("changed/static event denominator failed")
    changed = (changed_weight * cell_energy).sum(dim=(-2, -1)) / changed_denominator
    static = (static_weight * cell_energy).sum(dim=(-2, -1)) / static_denominator
    balanced = 0.5 * changed + 0.5 * static
    if not bool(
        cell_energy.isfinite().all()
        and changed.isfinite().all()
        and static.isfinite().all()
        and balanced.isfinite().all()
    ):
        raise FloatingPointError("event energy reduction became nonfinite")
    return _BalancedView(changed, static, balanced)


def _joint_terms(
    runtime: Any,
    model_api: Any,
    model: Any,
    batch: Mapping[str, Any],
    current_latent: Any,
    *,
    persistence_baseline: float,
) -> dict[str, Any]:
    """Registered joint loss, fitted into the inherited combined-backward loop."""

    torch = runtime.torch
    if not _FROZEN_REFERENCES:
        raise RuntimeError("joint event objective entered before T400/B400 freeze")
    if float(persistence_baseline) != float(_FROZEN_REFERENCES["B400"]):
        raise RuntimeError("joint event objective received a different B400")
    training_call = bool(model.training)

    x = _normalize(model_api, current_latent)
    with torch.no_grad():
        t0 = _normalize(model_api, model.encode_target(batch["current_rgb"]))
        if training_call:
            _EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] += 1
        t1 = _normalize(model_api, model.encode_target(batch["next_rgb"]))
        if training_call:
            _EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] += 1
        tn = _normalize(model_api, model.encode_target(batch["fixed_negative_rgb"]))
        if training_call:
            _EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] += 1
        x_fixed = _normalize(
            model_api, model.encode_online(batch["fixed_negative_rgb"])
        ).detach()
        if training_call:
            _EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] += 1
    target_delta = (t1 - t0).detach()
    negative_delta = (tn - t0).detach()
    persist = torch.nn.functional.smooth_l1_loss(
        target_delta,
        torch.zeros_like(target_delta),
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    weight = (persist / (persist + float(_FROZEN_REFERENCES["T400"]))).detach()

    prediction_all = _prediction_parts(model.predict_all_action_event_deltas(x))
    if training_call:
        _EVENT_ACCOUNTING["all_action_predictor_training_forward_count"] += 1
    prediction_exec = _select_prediction_with_torch(
        torch, prediction_all, batch["action_indices"]
    )
    action_one_hot = _action_one_hot(
        torch, batch["action_indices"], dtype=x_fixed.dtype
    )
    prediction_context = _prediction_parts(
        model.predict_event_delta(x_fixed, action_one_hot)
    )
    if training_call:
        _EVENT_ACCOUNTING["context_swap_predictor_training_forward_count"] += 1

    _e0_all, _e1_all, mixed_all = _event_cells(
        model_api, target_delta, prediction_all
    )
    energies = _balanced(mixed_all, weight).balanced
    rows = torch.arange(energies.shape[0], device=energies.device)
    executed = energies[rows, batch["action_indices"]]
    _e0_negative, _e1_negative, mixed_negative = _event_cells(
        model_api, negative_delta, prediction_exec
    )
    negative = _balanced(mixed_negative, weight).balanced
    _e0_context, _e1_context, mixed_context = _event_cells(
        model_api, target_delta, prediction_context
    )
    context = _balanced(mixed_context, weight).balanced

    scale = energies.mean(dim=1, keepdim=True).detach().clamp_min(1e-6)
    action_logits = -energies / scale
    raw_action_ce = torch.nn.functional.cross_entropy(
        action_logits, batch["action_indices"]
    )
    target_logits = torch.stack(
        (-executed / scale.squeeze(1), -negative / scale.squeeze(1)), dim=1
    )
    context_logits = torch.stack(
        (-executed / scale.squeeze(1), -context / scale.squeeze(1)), dim=1
    )
    zero_labels = torch.zeros(
        energies.shape[0], dtype=torch.long, device=energies.device
    )
    raw_target_ce = torch.nn.functional.cross_entropy(target_logits, zero_labels)
    raw_context_ce = torch.nn.functional.cross_entropy(context_logits, zero_labels)
    P_event = executed.mean() / float(_FROZEN_REFERENCES["B400"])
    if training_call:
        _EVENT_ACCOUNTING["event_persistence_term_evaluation_count"] += 1
    R_action = raw_action_ce / math.log(9.0)
    if training_call:
        _EVENT_ACCOUNTING["action_term_evaluation_count"] += 1
    C_target = raw_target_ce / math.log(2.0)
    if training_call:
        _EVENT_ACCOUNTING["target_term_evaluation_count"] += 1
    C_context = raw_context_ce / math.log(2.0)
    if training_call:
        _EVENT_ACCOUNTING["context_term_evaluation_count"] += 1
    dynamics = P_event + R_action + C_target + C_context
    if not bool(dynamics.isfinite()):
        raise FloatingPointError("event-delta joint objective became nonfinite")

    if training_call:
        _EVENT_ACCOUNTING["training_microbatch_count"] += 1
        _EVENT_ACCOUNTING["joint_combined_objective_evaluation_count"] += 1

    return {
        "target_next": t1,
        "target_negative": tn,
        "target_current": t0,
        "target_delta": target_delta,
        "negative_delta": negative_delta,
        "predictions": prediction_all.mu_event,
        "prediction_all": prediction_all,
        "prediction_exec": prediction_exec,
        "prediction_context": prediction_context,
        "energies": energies,
        "executed_energy": executed,
        "negative_energy": negative,
        "context_energy": context,
        "energy_scale": scale.squeeze(1),
        "action_scale": scale,
        "action_logits": action_logits,
        "raw_action_ce": raw_action_ce,
        "raw_target_ce": raw_target_ce,
        "raw_context_ce": raw_context_ce,
        "P": P_event,
        "R": R_action,
        "C": C_target + C_context,
        "P_event": P_event,
        "R_action": R_action,
        "C_target": C_target,
        "C_context": C_context,
        "D": dynamics,
    }


def _rng_snapshot(torch: Any, device: Any) -> tuple[Any, list[Any]]:
    cpu = torch.get_rng_state().clone()
    accelerator = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
        else []
    )
    return cpu, accelerator


def _restore_rng(torch: Any, snapshot: tuple[Any, list[Any]]) -> None:
    cpu, accelerator = snapshot
    torch.set_rng_state(cpu)
    if accelerator:
        torch.cuda.set_rng_state_all(accelerator)


def _rng_equal(torch: Any, left: tuple[Any, list[Any]], right: tuple[Any, list[Any]]) -> bool:
    return bool(
        torch.equal(left[0], right[0])
        and len(left[1]) == len(right[1])
        and all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True))
    )


def _persistence_baseline(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    device: Any,
) -> float:
    """Freeze raw T400 and balanced B400 without mutating runtime state."""

    torch = runtime.torch
    if _FROZEN_REFERENCES:
        raise RuntimeError("T400/B400 attempted to freeze more than once")
    model_before = contract.canonical_json_sha256({
        "encoder": _BASE._module_state_sha256(torch, model.encoder),
        "lift": _BASE._module_state_sha256(torch, model.bev_lift),
        "semantic": _BASE._module_state_sha256(torch, model.semantic_head),
        "predictor": _BASE._module_state_sha256(torch, model.predictor),
    })
    target_before = contract.canonical_json_sha256({
        "encoder": _BASE._module_state_sha256(torch, model.target_encoder),
        "lift": _BASE._module_state_sha256(torch, model.target_bev_lift),
    })
    optimizer_before = (
        None if _ACTIVE_OPTIMIZER is None
        else _state_value_sha256(torch, _ACTIVE_OPTIMIZER.state_dict())
    )
    rng_before = _rng_snapshot(torch, device)
    cpu_rng_before_sha256 = _state_value_sha256(torch, rng_before[0])
    accelerator_rng_before_sha256 = _state_value_sha256(torch, rng_before[1])
    action_actual: list[int] = []
    action_predicted: list[int] = []
    symmetry = True
    priors_half = True
    action_ce_sum = 0.0
    try:
        raw_sum = 0.0
        raw_count = 0
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(range(start, min(start + contract.MICROBATCH_SIZE, len(pairs))))
                batch = loader.batch(
                    pairs, indices, device,
                    role="checkpoint_selection",
                    stage="event_T400_update_400",
                    mapped_negative_indices=mapping["negative_indices"],
                    scope="observation",
                )
                t0 = _normalize(model_api, model.encode_target(batch["current_rgb"]))
                t1 = _normalize(model_api, model.encode_target(batch["next_rgb"]))
                delta = t1 - t0
                persist = torch.nn.functional.smooth_l1_loss(
                    delta, torch.zeros_like(delta), beta=1.0, reduction="none"
                ).mean(dim=1)
                raw_sum = _ordered_float64_sum(persist, initial=raw_sum)
                raw_count += int(persist.numel())
        T400 = raw_sum / raw_count
        if not math.isfinite(T400) or T400 <= 0.0:
            raise FloatingPointError("T400 is absent, nonfinite, or nonpositive")
        _FROZEN_REFERENCES["T400"] = T400

        balanced_sum = 0.0
        row_count = 0
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(range(start, min(start + contract.MICROBATCH_SIZE, len(pairs))))
                batch = loader.batch(
                    pairs, indices, device,
                    role="checkpoint_selection",
                    stage="event_B400_and_symmetry_update_400",
                    mapped_negative_indices=mapping["negative_indices"],
                    scope="observation",
                )
                t0 = _normalize(model_api, model.encode_target(batch["current_rgb"]))
                t1 = _normalize(model_api, model.encode_target(batch["next_rgb"]))
                delta = t1 - t0
                persist = torch.nn.functional.smooth_l1_loss(
                    delta, torch.zeros_like(delta), beta=1.0, reduction="none"
                ).mean(dim=1)
                weight = persist / (persist + T400)
                baseline_rows = _balanced(persist, weight).balanced
                balanced_sum = _ordered_float64_sum(
                    baseline_rows, initial=balanced_sum
                )
                row_count += len(indices)

                x = _normalize(model_api, model.encode_online(batch["current_rgb"]))
                prediction = _prediction_parts(model.predict_all_action_event_deltas(x))
                _e0, _e1, mixed = _event_cells(model_api, delta, prediction)
                energies = _balanced(mixed, weight).balanced
                symmetry = bool(
                    symmetry
                    and all(torch.equal(energies[:, 0], energies[:, action]) for action in range(1, 9))
                )
                prior = torch.sigmoid(prediction.event_logit)
                priors_half = bool(
                    priors_half
                    and torch.equal(prior, torch.full_like(prior, 0.5))
                )
                scale = energies.mean(dim=1, keepdim=True).clamp_min(1e-6)
                logits = -energies / scale
                ce = torch.nn.functional.cross_entropy(
                    logits, batch["action_indices"], reduction="none"
                )
                action_ce_sum = _ordered_float64_sum(
                    ce, initial=action_ce_sum
                )
                action_actual.extend(map(int, batch["action_indices"].cpu().tolist()))
                action_predicted.extend(map(int, logits.argmax(dim=1).cpu().tolist()))
        B400 = balanced_sum / row_count
        if (
            row_count != contract.SELECTION_ROLE_COUNTS["pairs"]
            or raw_count != row_count * 64 * 64
            or not math.isfinite(B400)
            or B400 <= 0.0
        ):
            raise FloatingPointError("B400 is absent, nonfinite, or nonpositive")
    finally:
        _restore_rng(torch, rng_before)

    rng_after = _rng_snapshot(torch, device)
    model_after = contract.canonical_json_sha256({
        "encoder": _BASE._module_state_sha256(torch, model.encoder),
        "lift": _BASE._module_state_sha256(torch, model.bev_lift),
        "semantic": _BASE._module_state_sha256(torch, model.semantic_head),
        "predictor": _BASE._module_state_sha256(torch, model.predictor),
    })
    target_after = contract.canonical_json_sha256({
        "encoder": _BASE._module_state_sha256(torch, model.target_encoder),
        "lift": _BASE._module_state_sha256(torch, model.target_bev_lift),
    })
    optimizer_after = (
        None if _ACTIVE_OPTIMIZER is None
        else _state_value_sha256(torch, _ACTIVE_OPTIMIZER.state_dict())
    )
    cpu_rng_after_sha256 = _state_value_sha256(torch, rng_after[0])
    accelerator_rng_after_sha256 = _state_value_sha256(torch, rng_after[1])
    action_ba, _recalls = _BASE._action_balanced_accuracy(
        action_actual, action_predicted
    )
    joint_hash = contract.canonical_json_sha256({
        "definition": "raw_cell_temperature_and_balanced_persistence_baseline",
        "update": 400,
        "T400": T400,
        "B400": B400,
    })
    _FROZEN_REFERENCES.update({
        "B400": B400,
        "T400_B400_content_sha256": joint_hash,
        "actual_population_action_energy_bitwise_symmetric": symmetry,
        "actual_population_action_nll": action_ce_sum / row_count,
        "actual_population_action_macro_balanced_accuracy": action_ba,
        "event_prior_bitwise_half": priors_half,
        "calibration_online_model_state_before_sha256": model_before,
        "calibration_online_model_state_after_sha256": model_after,
        "calibration_target_state_before_sha256": target_before,
        "calibration_target_state_after_sha256": target_after,
        "calibration_optimizer_state_before_sha256": optimizer_before,
        "calibration_optimizer_state_after_sha256": optimizer_after,
        "calibration_cpu_rng_before_sha256": cpu_rng_before_sha256,
        "calibration_cpu_rng_after_sha256": cpu_rng_after_sha256,
        "calibration_accelerator_rng_before_sha256": accelerator_rng_before_sha256,
        "calibration_accelerator_rng_after_sha256": accelerator_rng_after_sha256,
        "calibration_model_state_preserved": model_before == model_after,
        "calibration_target_state_preserved": target_before == target_after,
        "calibration_optimizer_state_preserved": optimizer_before == optimizer_after,
        "calibration_cpu_rng_preserved": torch.equal(rng_before[0], rng_after[0]),
        "calibration_accelerator_rng_preserved": _rng_equal(torch, rng_before, rng_after),
    })
    if not all((
        symmetry,
        priors_half,
        model_before == model_after,
        target_before == target_after,
        optimizer_before == optimizer_after,
        _rng_equal(torch, rng_before, rng_after),
    )):
        raise RuntimeError("update-400 event calibration mutated frozen state")
    return B400


def _ratio(numerator: float, denominator: float, name: str) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError(f"{name} denominator is not strictly positive")
    value = numerator / denominator
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} ratio is nonfinite")
    return value


def _event_observation_metrics(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Compute registered U1000 event/ablation diagnostics without training."""

    torch = runtime.torch
    T400 = float(_FROZEN_REFERENCES["T400"])
    eligible = list(mapping["same_action_eligible"])
    mapped_negative_indices = mapping["negative_indices"]
    if len(pairs) != contract.SELECTION_ROLE_COUNTS["pairs"] or sum(map(bool, eligible)) != 494:
        raise PermissionError("event observation population changed")

    model_before = _BASE._module_state_sha256(torch, model)
    optimizer_before = (
        None if _ACTIVE_OPTIMIZER is None
        else _state_value_sha256(torch, _ACTIVE_OPTIMIZER.state_dict())
    )
    rng_before = _rng_snapshot(torch, device)
    was_training = bool(model.training)
    model.eval()
    template_sums = torch.zeros(9, 64, 64, 64, dtype=torch.float64)
    template_counts = torch.zeros(9, dtype=torch.long)
    totals: dict[str, float] = defaultdict(float)
    families: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    actual_actions: list[int] = []
    predicted_actions: list[int] = []
    row_count = 0
    target_count = 0
    non_hold_count = 0
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(range(start, min(start + contract.MICROBATCH_SIZE, len(pairs))))
                batch = loader.batch(
                    pairs, indices, device,
                    role="checkpoint_selection",
                    stage="event_state_template_accumulation_update_1000",
                    mapped_negative_indices=mapped_negative_indices,
                    scope="observation",
                )
                x = _normalize(model_api, model.encode_online(batch["current_rgb"]))
                x_cpu = x.detach().cpu().double()
                for offset, action in enumerate(batch["action_indices"].cpu().tolist()):
                    template_sums[int(action)].add_(x_cpu[offset])
                    template_counts[int(action)] += 1
            if not bool((template_counts > 0).all()):
                raise RuntimeError("state-template action group is empty")
            templates = (
                template_sums / template_counts[:, None, None, None]
            ).to(dtype=torch.float32)

            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(range(start, min(start + contract.MICROBATCH_SIZE, len(pairs))))
                batch = loader.batch(
                    pairs, indices, device,
                    role="checkpoint_selection",
                    stage="event_diagnostics_update_1000",
                    mapped_negative_indices=mapped_negative_indices,
                    scope="observation",
                )
                actions = batch["action_indices"]
                size = len(indices)
                t0 = _normalize(model_api, model.encode_target(batch["current_rgb"]))
                t1 = _normalize(model_api, model.encode_target(batch["next_rgb"]))
                tn = _normalize(model_api, model.encode_target(batch["fixed_negative_rgb"]))
                x = _normalize(model_api, model.encode_online(batch["current_rgb"]))
                x_fixed = _normalize(model_api, model.encode_online(batch["fixed_negative_rgb"]))
                delta = t1 - t0
                negative_delta = tn - t0
                persist = torch.nn.functional.smooth_l1_loss(
                    delta, torch.zeros_like(delta), beta=1.0, reduction="none"
                ).mean(dim=1)
                weight = persist / (persist + T400)
                one_hot = _action_one_hot(torch, actions, dtype=x.dtype)
                prediction_all = _prediction_parts(model.predict_all_action_event_deltas(x))
                prediction_exec = _select_prediction_with_torch(torch, prediction_all, actions)
                prediction_context = _prediction_parts(model.predict_event_delta(x_fixed, one_hot))
                template_batch = templates.index_select(0, actions.cpu()).to(device=device)
                prediction_template = _prediction_parts(model.predict_event_delta(template_batch, one_hot))

                e0_all, e1_all, mix_all = _event_cells(model_api, delta, prediction_all)
                all_parts = _balanced(mix_all, weight)
                energies = all_parts.balanced
                rows = torch.arange(size, device=device)
                executed = energies[rows, actions]
                e0_exec = e0_all[rows, actions]
                e1_exec = e1_all[rows, actions]
                mix_exec = mix_all[rows, actions]
                zero_parts = _balanced(e0_exec, weight)
                event_parts = _balanced(e1_exec, weight)

                _ne0, _ne1, negative_mix = _event_cells(model_api, negative_delta, prediction_exec)
                negative_energy = _balanced(negative_mix, weight).balanced
                _ce0, _ce1, context_mix = _event_cells(model_api, delta, prediction_context)
                context_energy = _balanced(context_mix, weight).balanced
                _te0, _te1, template_mix = _event_cells(model_api, delta, prediction_template)
                template_energy = _balanced(template_mix, weight).balanced

                matched_cell = model_api.matched_single_mean_cell_energy_v1(
                    delta, prediction_exec
                )
                matched_energy = _balanced(matched_cell, weight).balanced
                state_only_cell = -T400 * torch.logsumexp(
                    -mix_all / T400 - math.log(9.0), dim=1
                )
                state_only_energy = _balanced(state_only_cell, weight).balanced
                prior = model_api.event_prior_probability_v1(prediction_exec)
                prior_context = model_api.event_prior_probability_v1(prediction_context)
                prior_parts = _balanced(prior, weight)
                posterior = model_api.event_posterior_responsibility_v1(
                    e0_exec, e1_exec, prediction_exec.event_logit, T400
                )
                posterior_parts = _balanced(posterior, weight)
                mu_abs_cell = prediction_exec.mu_event.abs().mean(dim=1)
                mu_changed = _balanced(mu_abs_cell, weight).changed
                prior_mean = prior.mean(dim=(-2, -1))
                prior_variance = (
                    (prior - prior_mean[:, None, None]).square().mean(dim=(-2, -1))
                )
                prior_context_difference = (
                    prior - prior_context
                ).abs().mean(dim=(-2, -1))

                scale = energies.mean(dim=1, keepdim=True).clamp_min(1e-6)
                action_logits = -energies / scale
                action_nll = torch.nn.functional.cross_entropy(
                    action_logits, actions, reduction="none"
                )
                target_logits = torch.stack((
                    -executed / scale.squeeze(1),
                    -negative_energy / scale.squeeze(1),
                ), dim=1)
                context_logits = torch.stack((
                    -executed / scale.squeeze(1),
                    -context_energy / scale.squeeze(1),
                ), dim=1)
                labels = torch.zeros(size, dtype=torch.long, device=device)
                target_nll = torch.nn.functional.cross_entropy(
                    target_logits, labels, reduction="none"
                )
                context_nll = torch.nn.functional.cross_entropy(
                    context_logits, labels, reduction="none"
                )
                wrong_mask = torch.ones_like(energies, dtype=torch.bool)
                wrong_mask[rows, actions] = False
                hardest = energies.masked_fill(~wrong_mask, torch.inf).min(dim=1).values
                mean_wrong = energies.masked_fill(~wrong_mask, 0.0).sum(dim=1) / 8.0

                row_values = {
                    "event_balanced_energy": executed,
                    "action_nll": action_nll,
                    "mean_executed_action_energy": executed,
                    "mean_wrong_action_energy": mean_wrong,
                    "context_true_energy": executed,
                    "context_swap_energy": context_energy,
                    "state_true_energy": executed,
                    "state_template_energy": template_energy,
                    "executed_action_energy": executed,
                    "state_only_energy": state_only_energy,
                    "two_mode_energy": executed,
                    "matched_single_mean_energy": matched_energy,
                    "event_changed_energy": event_parts.changed,
                    "zero_changed_energy": zero_parts.changed,
                    "zero_static_energy": zero_parts.static,
                    "event_static_energy": event_parts.static,
                    "mixture_overall_energy": executed,
                    "zero_overall_energy": zero_parts.balanced,
                    "event_overall_energy": event_parts.balanced,
                    "mu_event_changed_abs": mu_changed,
                    "prior_changed_mean": prior_parts.changed,
                    "prior_static_mean": prior_parts.static,
                    "prior_mean": prior_mean,
                    "prior_spatial_variance": prior_variance,
                    "prior_context_difference": prior_context_difference,
                    "posterior_changed_mean": posterior_parts.changed,
                    "posterior_static_mean": posterior_parts.static,
                    "posterior_mean": posterior.mean(dim=(-2, -1)),
                    "context_nll": context_nll,
                }
                for name, values in row_values.items():
                    totals[name] = _ordered_float64_sum(
                        values, initial=totals[name]
                    )
                actual_actions.extend(map(int, actions.cpu().tolist()))
                predicted_actions.extend(map(int, action_logits.argmax(dim=1).cpu().tolist()))
                row_count += size

                non_hold = batch["non_hold_mask"]
                hold_energy = energies[:, contract.HOLD_ACTION_INDEX]
                totals["mean_non_hold_executed_action_energy"] = _ordered_float64_sum(
                    executed[non_hold],
                    initial=totals["mean_non_hold_executed_action_energy"],
                )
                totals["mean_non_hold_hold_action_energy"] = _ordered_float64_sum(
                    hold_energy[non_hold],
                    initial=totals["mean_non_hold_hold_action_energy"],
                )
                non_hold_count += int(non_hold.sum())

                for offset, source_index in enumerate(indices):
                    family = str(pairs[source_index]["family"])
                    family_row = families[family]
                    family_row["rows"] += 1.0
                    margins = {
                        "hardest": hardest[offset] - executed[offset],
                        "context": context_energy[offset] - executed[offset],
                        "state": template_energy[offset] - executed[offset],
                        "state_only": state_only_energy[offset] - executed[offset],
                        "matched": matched_energy[offset] - executed[offset],
                        "event_changed": zero_parts.changed[offset] - event_parts.changed[offset],
                        "zero_static": event_parts.static[offset] - zero_parts.static[offset],
                        "mixture_zero": zero_parts.balanced[offset] - executed[offset],
                        "mixture_event": event_parts.balanced[offset] - executed[offset],
                    }
                    for name, value in margins.items():
                        family_row[name] += float(value.detach().cpu())
                    family_row["prior_context_difference"] += float(
                        prior_context_difference[offset].detach().cpu()
                    )
                    family_values = {
                        "event_changed_energy": event_parts.changed[offset],
                        "zero_changed_energy": zero_parts.changed[offset],
                        "zero_static_energy": zero_parts.static[offset],
                        "event_static_energy": event_parts.static[offset],
                        "mixture_overall_energy": executed[offset],
                        "zero_overall_energy": zero_parts.balanced[offset],
                        "event_overall_energy": event_parts.balanced[offset],
                        "mu_event_changed_abs": mu_changed[offset],
                        "prior_changed_mean": prior_parts.changed[offset],
                        "prior_static_mean": prior_parts.static[offset],
                        "prior_mean": prior_mean[offset],
                        "prior_spatial_variance": prior_variance[offset],
                        "posterior_changed_mean": posterior_parts.changed[offset],
                        "posterior_static_mean": posterior_parts.static[offset],
                        "posterior_mean": posterior[offset].mean(),
                    }
                    for name, value in family_values.items():
                        family_row[name] += float(value.detach().cpu())
                    totals["context_true_wins"] += float(
                        context_energy[offset] > executed[offset]
                    )
                    if bool(eligible[source_index]):
                        target_margin = negative_energy[offset] - executed[offset]
                        family_row["target"] += float(target_margin.detach().cpu())
                        family_row["target_rows"] += 1.0
                        totals["target_nll"] += float(target_nll[offset].detach().cpu())
                        totals["target_wins"] += float(target_margin > 0.0)
                        target_count += 1
    finally:
        model.train(was_training)
        _restore_rng(torch, rng_before)

    if row_count != 495 or target_count != 494 or non_hold_count != 435:
        raise RuntimeError("event diagnostic accounting changed")
    if _BASE._module_state_sha256(torch, model) != model_before:
        raise RuntimeError("event observer mutated model state")
    optimizer_after = (
        None if _ACTIVE_OPTIMIZER is None
        else _state_value_sha256(torch, _ACTIVE_OPTIMIZER.state_dict())
    )
    if optimizer_after != optimizer_before or not _rng_equal(torch, rng_before, _rng_snapshot(torch, device)):
        raise RuntimeError("event observer mutated optimizer or RNG state")

    family_counts: dict[str, int] = defaultdict(int)
    posterior_two_sided = 0
    scene_updates: dict[str, Any] = {}
    for family in contract.SCENE_FAMILIES:
        row = families[family]
        expected = int(contract.SELECTION_FAMILY_BINDINGS[family]["row_count"])
        if int(row["rows"]) != expected or row["target_rows"] <= 0:
            raise RuntimeError("event diagnostic family population changed")
        for name in (
            "hardest", "context", "state", "state_only", "matched",
            "event_changed", "zero_static", "mixture_zero", "mixture_event",
            "prior_context_difference", "target",
        ):
            denominator = row["target_rows"] if name == "target" else row["rows"]
            margin = row[name] / denominator
            family_counts[name] += int(margin > 0.0)
        posterior_mean = row["posterior_mean"] / row["rows"]
        posterior_two_sided += int(
            posterior_mean >= 0.05 and (1.0 - posterior_mean) >= 0.05
        )
        scene_updates[family] = {
            "event_hardest_wrong_minus_executed_energy": row["hardest"] / row["rows"],
            "event_context_swap_minus_true_energy": row["context"] / row["rows"],
            "event_state_template_minus_true_energy": row["state"] / row["rows"],
            "event_state_only_minus_executed_energy": row["state_only"] / row["rows"],
            "event_matched_single_minus_two_mode_energy": row["matched"] / row["rows"],
            "event_target_margin": row["target"] / row["target_rows"],
            "event_changed_energy": row["event_changed_energy"] / row["rows"],
            "zero_changed_energy": row["zero_changed_energy"] / row["rows"],
            "zero_static_energy": row["zero_static_energy"] / row["rows"],
            "event_static_energy": row["event_static_energy"] / row["rows"],
            "mixture_overall_energy": row["mixture_overall_energy"] / row["rows"],
            "zero_overall_energy": row["zero_overall_energy"] / row["rows"],
            "event_overall_energy": row["event_overall_energy"] / row["rows"],
            "mu_event_changed_abs": row["mu_event_changed_abs"] / row["rows"],
            "prior_changed_mean": row["prior_changed_mean"] / row["rows"],
            "prior_static_mean": row["prior_static_mean"] / row["rows"],
            "prior_mean": row["prior_mean"] / row["rows"],
            "prior_spatial_variance": row["prior_spatial_variance"] / row["rows"],
            "prior_context_difference": row["prior_context_difference"] / row["rows"],
            "posterior_changed_mean": row["posterior_changed_mean"] / row["rows"],
            "posterior_static_mean": row["posterior_static_mean"] / row["rows"],
            "posterior_mean": posterior_mean,
            "posterior_zero_mean": 1.0 - posterior_mean,
            "posterior_event_and_zero_materially_used": bool(
                posterior_mean >= 0.05 and (1.0 - posterior_mean) >= 0.05
            ),
            "event_over_zero_changed_margin": row["event_changed"] / row["rows"],
            "zero_over_event_static_margin": row["zero_static"] / row["rows"],
            "mixture_over_zero_margin": row["mixture_zero"] / row["rows"],
            "mixture_over_event_margin": row["mixture_event"] / row["rows"],
        }

    action_ba, action_recalls = _BASE._action_balanced_accuracy(
        actual_actions, predicted_actions
    )
    mean = lambda name: totals[name] / row_count
    result: dict[str, Any] = {
        name: mean(name)
        for name in (
            "event_balanced_energy", "action_nll", "mean_executed_action_energy",
            "mean_wrong_action_energy", "context_true_energy", "context_swap_energy",
            "state_true_energy", "state_template_energy", "executed_action_energy",
            "state_only_energy", "two_mode_energy", "matched_single_mean_energy",
            "event_changed_energy", "zero_changed_energy", "zero_static_energy",
            "event_static_energy", "mixture_overall_energy", "zero_overall_energy",
            "event_overall_energy", "mu_event_changed_abs", "prior_changed_mean",
            "prior_static_mean", "prior_mean", "prior_spatial_variance",
            "prior_context_difference", "posterior_changed_mean",
            "posterior_static_mean", "posterior_mean", "context_nll",
        )
    }
    result.update({
        "action_macro_balanced_accuracy": action_ba,
        "action_per_class_recall": action_recalls,
        "hardest_wrong_positive_family_count": family_counts["hardest"],
        "mean_non_hold_executed_action_energy": (
            totals["mean_non_hold_executed_action_energy"] / non_hold_count
        ),
        "mean_non_hold_hold_action_energy": (
            totals["mean_non_hold_hold_action_energy"] / non_hold_count
        ),
        "target_nll": totals["target_nll"] / target_count,
        "target_strict_win_rate": totals["target_wins"] / target_count,
        "target_positive_family_count": family_counts["target"],
        "context_true_to_swap_energy_ratio": _ratio(
            mean("context_true_energy"), mean("context_swap_energy"), "context"
        ),
        "context_true_strict_win_rate": totals["context_true_wins"] / row_count,
        "context_positive_family_count": family_counts["context"],
        "state_true_to_template_energy_ratio": _ratio(
            mean("state_true_energy"), mean("state_template_energy"), "state template"
        ),
        "state_positive_family_count": family_counts["state"],
        "executed_to_state_only_energy_ratio": _ratio(
            mean("executed_action_energy"), mean("state_only_energy"), "state only"
        ),
        "state_only_positive_family_count": family_counts["state_only"],
        "two_mode_to_matched_single_ratio": _ratio(
            mean("two_mode_energy"), mean("matched_single_mean_energy"), "matched mean"
        ),
        "matched_single_positive_family_count": family_counts["matched"],
        "event_over_zero_changed_positive_family_count": family_counts["event_changed"],
        "zero_over_event_static_positive_family_count": family_counts["zero_static"],
        "mixture_beats_zero_family_count": family_counts["mixture_zero"],
        "mixture_beats_event_family_count": family_counts["mixture_event"],
        "prior_context_difference_positive_family_count": family_counts["prior_context_difference"],
        "posterior_event_and_zero_family_count": posterior_two_sided,
        "event_observer_model_optimizer_rng_preserved": True,
        "event_scene_metrics": scene_updates,
    })
    return result


def _reviewed_cpu_source_witness(
    *, source_authority_exact: bool
) -> dict[str, Any]:
    """Bind, without executing it, the reviewed CPU-only synthetic witness.

    The forced source-closure model test covers exact initialization, zero
    embedding and half prior, the online encoder/lift route, input projection,
    each of four residual convolutions, both heads, absent EMA gradients, and
    the registered constant-state/action-template and action-ignored
    anti-degeneracy fixtures. Runtime U0 performs no synthetic accelerator
    work.
    """

    source_test = str(contract.MODEL_TEST_RELATIVE_PATH)
    source_test_sha256 = _ACTIVE_SOURCE_BINDINGS.get(source_test)
    reviewed_full_route_witness_bound = bool(
        source_authority_exact
        and source_test in contract.SOURCE_PATHS
        and source_test in contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        and isinstance(source_test_sha256, str)
        and len(source_test_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_test_sha256)
    )
    return {
        "event_tensor_shapes_exact": reviewed_full_route_witness_bound,
        "fixed_mode_identities_exact": reviewed_full_route_witness_bound,
        "output_parameter_action_symmetry_exact": reviewed_full_route_witness_bound,
        "synthetic_positive_temperature_stable_energy_exact": reviewed_full_route_witness_bound,
        "zero_mean_persistence_identity_exact": reviewed_full_route_witness_bound,
        "synthetic_action_nll_log9_exact": reviewed_full_route_witness_bound,
        "synthetic_action_macro_balanced_accuracy_one_ninth_exact": reviewed_full_route_witness_bound,
        "event_initialization_and_gradient_witness_exact": reviewed_full_route_witness_bound,
        "action_embedding_exact_zero_at_update_zero": reviewed_full_route_witness_bound,
        "event_prior_bitwise_float32_half_at_update_zero": reviewed_full_route_witness_bound,
        "mean_and_logit_head_initialization_and_rng_order_exact": reviewed_full_route_witness_bound,
        "online_encoder_lift_and_each_predictor_submodule_gradient_witness_exact": reviewed_full_route_witness_bound,
        "constant_state_action_template_fixture_fails_context_and_state_gates_exact": reviewed_full_route_witness_bound,
        "action_ignored_fixture_fails_action_and_state_only_gates_exact": reviewed_full_route_witness_bound,
        "runtime_update_zero_synthetic_accelerator_call_count": 0,
        "reviewed_model_source_synthetic_encoder_lift_and_each_residual_conv_witness_bound": reviewed_full_route_witness_bound,
        "reviewed_model_source_synthetic_witness_path": source_test,
        "reviewed_model_source_synthetic_witness_sha256": source_test_sha256,
        "no_scale_inverse_pair_posterior_transport_or_future_bypass": reviewed_full_route_witness_bound,
    }


def _accounting_metrics(
    update: int,
    *,
    objective_evaluations: int | None = None,
    backward_calls: int | None = None,
) -> dict[str, Any]:
    semantic = int(_EVENT_ACCOUNTING["semantic_term_evaluation_count"])
    combined = int(
        min(
            semantic,
            (contract.JOINT_PHASE_FIRST_UPDATE - 1)
            * contract.MICROBATCHES_PER_UPDATE,
        )
        + _EVENT_ACCOUNTING["joint_combined_objective_evaluation_count"]
    )
    if objective_evaluations is not None and int(objective_evaluations) > combined:
        raise RuntimeError("inherited objective count exceeds measured boundaries")
    backward = int(combined if backward_calls is None else backward_calls)
    event_terms = int(_EVENT_ACCOUNTING["event_persistence_term_evaluation_count"])
    action_terms = int(_EVENT_ACCOUNTING["action_term_evaluation_count"])
    target_terms = int(_EVENT_ACCOUNTING["target_term_evaluation_count"])
    context_terms = int(_EVENT_ACCOUNTING["context_term_evaluation_count"])
    warning_policy_bound = bool(
        contract.WARNING_POLICY["allowed_category"] == "UserWarning"
        and contract.WARNING_POLICY["allowed_base_message"]
        == ROCM_GRID_SAMPLE_DETERMINISM_WARNING
        and contract.WARNING_POLICY["reject_every_other_warning"] is True
        and canonicalize_rocm_determinism_warning(
            ROCM_GRID_SAMPLE_DETERMINISM_WARNING
        ) == ROCM_GRID_SAMPLE_DETERMINISM_WARNING
    )
    state_hash_bound = bool(
        _BASE._tensor_state_sha256 is _tensor_state_sha256
    )
    receipt_schema_bound = bool(
        tuple(contract.OPERATIONAL_FAILURE_RECEIPT_PATHS)
        == ("failure.json", "completed.json")
        and all(
            isinstance(getattr(contract, name), str)
            for name in (
                "METRICS_SCHEMA", "ARTIFACT_SCHEMA", "ACCESS_SCHEMA",
                "RESULT_SCHEMA", "FAILURE_SCHEMA", "COMPLETION_SCHEMA",
            )
        )
    )
    return {
        "combined_objective_evaluation_count": combined,
        "objective_evaluations": combined,
        "backward_call_count": backward,
        "backward_calls": backward,
        "pair_presentations_loaded": int(_EVENT_ACCOUNTING["scheduled_pair_presentations_loaded"]),
        "semantic_term_evaluation_count": semantic,
        "event_persistence_term_evaluation_count": event_terms,
        "action_term_evaluation_count": action_terms,
        "target_term_evaluation_count": target_terms,
        "context_term_evaluation_count": context_terms,
        "registered_scalar_term_evaluation_count": semantic + event_terms + action_terms + target_terms + context_terms,
        "all_action_predictor_training_forward_count": int(_EVENT_ACCOUNTING["all_action_predictor_training_forward_count"]),
        "context_swap_predictor_training_forward_count": int(_EVENT_ACCOUNTING["context_swap_predictor_training_forward_count"]),
        "online_encoder_lift_training_forward_count": int(_EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"]),
        "semantic_head_training_forward_count": int(_EVENT_ACCOUNTING["semantic_head_training_forward_count"]),
        "target_encoder_lift_training_forward_count": int(_EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"]),
        "action_embedding_dynamics_gradient_update_count": int(_EVENT_ACCOUNTING["action_embedding_dynamics_gradient_update_count"]),
        "predictor_trunk_dynamics_gradient_update_count": int(_EVENT_ACCOUNTING["predictor_trunk_dynamics_gradient_update_count"]),
        "event_mean_head_dynamics_gradient_update_count": int(_EVENT_ACCOUNTING["event_mean_head_dynamics_gradient_update_count"]),
        "event_logit_head_dynamics_gradient_update_count": int(_EVENT_ACCOUNTING["event_logit_head_dynamics_gradient_update_count"]),
        "warning_policy_exact": warning_policy_bound,
        "warning_policy_configuration_bound_exact": warning_policy_bound,
        "state_hash_accounting_exact": state_hash_bound,
        "state_hash_adapter_configuration_bound_exact": state_hash_bound,
        "receipt_schema_accounting_exact": receipt_schema_bound,
        "receipt_schema_configuration_bound_exact": receipt_schema_bound,
    }


def _begin_observation_accounting(update: int) -> None:
    """Start an unscheduled, call-boundary observation-work receipt."""

    _OBSERVATION_LIVE.clear()
    _OBSERVATION_LIVE.update({
        "observation_update": int(update),
        "observation_status": "in_progress",
        "observation_pair_successful_microbatch_count": 0,
        "observation_endpoint_successful_microbatch_count": 0,
        "observation_pair_rows_loaded": 0,
        "observation_endpoint_rows_loaded": 0,
        "observation_pair_stage_successful_microbatch_counts": {},
        "observation_pair_stage_rows_loaded": {},
        "observation_endpoint_stage_successful_microbatch_counts": {},
        "observation_endpoint_stage_rows_loaded": {},
        "observation_online_encoder_lift_forward_count": 0,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 0,
        "observation_all_action_predictor_forward_count": 0,
        "observation_one_action_predictor_forward_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
        "last_started_call": None,
        "last_successful_call": None,
    })


def _observation_live_receipt() -> dict[str, Any] | None:
    if not _OBSERVATION_LIVE:
        return None
    value = dict(_OBSERVATION_LIVE)
    for name in (
        "observation_pair_stage_successful_microbatch_counts",
        "observation_pair_stage_rows_loaded",
        "observation_endpoint_stage_successful_microbatch_counts",
        "observation_endpoint_stage_rows_loaded",
    ):
        value[name] = {
            str(stage): int(count)
            for stage, count in sorted(value[name].items())
        }
    pair_rows = value["observation_pair_stage_rows_loaded"]
    endpoint_rows = value["observation_endpoint_stage_rows_loaded"]
    value["observation_pair_completed_pass_count"] = sum(
        int(rows == contract.SELECTION_ROLE_COUNTS["pairs"])
        for rows in pair_rows.values()
    )
    value["observation_endpoint_completed_pass_count"] = sum(
        int(rows == contract.SELECTION_ROLE_COUNTS["unique_endpoints"])
        for rows in endpoint_rows.values()
    )
    value["observation_microbatch_count"] = int(
        value["observation_pair_successful_microbatch_count"]
        + value["observation_endpoint_successful_microbatch_count"]
    )
    value["observation_predictor_forward_count"] = int(
        value["observation_all_action_predictor_forward_count"]
        + value["observation_one_action_predictor_forward_count"]
    )
    return value


def _mark_observation_failure(error: BaseException) -> None:
    if not _OBSERVATION_LIVE:
        return
    _OBSERVATION_LIVE["observation_status"] = "failed"
    _OBSERVATION_LIVE["failure_type"] = type(error).__name__
    _OBSERVATION_LIVE["failure_message_sha256"] = hashlib.sha256(
        str(error).encode("utf-8")
    ).hexdigest()


@contextmanager
def _instrument_observation_calls(model: Any, loader: Any) -> Any:
    """Count only calls that return successfully during an observation."""

    restorations: list[tuple[type[Any], str, bool, Any]] = []

    def install_model(name: str, counter: str) -> None:
        owner = type(model)
        existed = name in owner.__dict__
        prior = owner.__dict__.get(name)
        original = getattr(owner, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            _OBSERVATION_LIVE["last_started_call"] = name
            result = original(instance, *args, **kwargs)
            _OBSERVATION_LIVE[counter] += 1
            _OBSERVATION_LIVE["last_successful_call"] = name
            return result

        setattr(owner, name, wrapped)
        restorations.append((owner, name, existed, prior))

    def install_loader(name: str, *, endpoint: bool) -> None:
        owner = type(loader)
        existed = name in owner.__dict__
        prior = owner.__dict__.get(name)
        original = getattr(owner, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            stage = str(kwargs.get("stage", "unknown"))
            call_name = f"loader.{name}:{stage}"
            _OBSERVATION_LIVE["last_started_call"] = call_name
            result = original(instance, *args, **kwargs)
            if endpoint:
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError(
                        "observation endpoint batch return shape changed"
                    )
                rows = int(result[0].shape[0])
                count_name = "observation_endpoint_successful_microbatch_count"
                rows_name = "observation_endpoint_rows_loaded"
                stages_name = (
                    "observation_endpoint_stage_successful_microbatch_counts"
                )
                stage_rows_name = "observation_endpoint_stage_rows_loaded"
            else:
                if not isinstance(result, Mapping) or "action_indices" not in result:
                    raise RuntimeError(
                        "observation pair batch return shape changed"
                    )
                rows = int(result["action_indices"].shape[0])
                count_name = "observation_pair_successful_microbatch_count"
                rows_name = "observation_pair_rows_loaded"
                stages_name = "observation_pair_stage_successful_microbatch_counts"
                stage_rows_name = "observation_pair_stage_rows_loaded"
            _OBSERVATION_LIVE[count_name] += 1
            _OBSERVATION_LIVE[rows_name] += int(rows)
            stage_counts = _OBSERVATION_LIVE[stages_name]
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
            stage_rows = _OBSERVATION_LIVE[stage_rows_name]
            stage_rows[stage] = int(stage_rows.get(stage, 0)) + int(rows)
            _OBSERVATION_LIVE["last_successful_call"] = call_name
            return result

        setattr(owner, name, wrapped)
        restorations.append((owner, name, existed, prior))

    install_model(
        "encode_online", "observation_online_encoder_lift_forward_count"
    )
    install_model(
        "encode_target", "observation_target_encoder_lift_forward_count"
    )
    install_model(
        "semantic_logits_from_latent",
        "observation_semantic_head_forward_count",
    )
    install_model(
        "predict_all_action_event_deltas",
        "observation_all_action_predictor_forward_count",
    )
    install_model(
        "predict_event_delta", "observation_one_action_predictor_forward_count"
    )
    install_loader("batch", endpoint=False)
    install_loader("endpoint_batch", endpoint=True)
    try:
        yield
    finally:
        for owner, name, existed, prior in reversed(restorations):
            if existed:
                setattr(owner, name, prior)
            else:
                delattr(owner, name)


def _validate_completed_observation_accounting(update: int) -> dict[str, Any]:
    receipt = _observation_live_receipt()
    if receipt is None:
        raise RuntimeError("observation live receipt is absent")
    expected = _observation_accounting(update)
    comparisons = {
        "observation_pair_pass_count": "observation_pair_completed_pass_count",
        "observation_endpoint_pass_count": (
            "observation_endpoint_completed_pass_count"
        ),
        "observation_pair_microbatch_count": (
            "observation_pair_successful_microbatch_count"
        ),
        "observation_endpoint_microbatch_count": (
            "observation_endpoint_successful_microbatch_count"
        ),
        "observation_microbatch_count": "observation_microbatch_count",
        "observation_online_encoder_lift_forward_count": (
            "observation_online_encoder_lift_forward_count"
        ),
        "observation_target_encoder_lift_forward_count": (
            "observation_target_encoder_lift_forward_count"
        ),
        "observation_semantic_head_forward_count": (
            "observation_semantic_head_forward_count"
        ),
        "observation_all_action_predictor_forward_count": (
            "observation_all_action_predictor_forward_count"
        ),
        "observation_one_action_predictor_forward_count": (
            "observation_one_action_predictor_forward_count"
        ),
        "observation_predictor_forward_count": (
            "observation_predictor_forward_count"
        ),
        "observation_presentations_count": "observation_presentations_count",
        "observation_schedule_advance_count": (
            "observation_schedule_advance_count"
        ),
    }
    mismatches = {
        expected_name: {
            "expected": expected[expected_name],
            "observed": receipt[receipt_name],
        }
        for expected_name, receipt_name in comparisons.items()
        if receipt[receipt_name] != expected[expected_name]
    }
    if mismatches:
        raise RuntimeError(
            "live observation accounting changed: "
            + contract.canonical_json_sha256(mismatches)
        )
    _OBSERVATION_LIVE["observation_status"] = "complete"
    completed = _observation_live_receipt()
    if completed is None:
        raise RuntimeError("completed observation receipt is absent")
    return completed


def _observation_accounting(update: int) -> dict[str, Any]:
    pair_batches = math.ceil(
        contract.SELECTION_ROLE_COUNTS["pairs"] / contract.MICROBATCH_SIZE
    )
    endpoint_batches = math.ceil(
        contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        / contract.MICROBATCH_SIZE
    )
    if update in (0, 100):
        pair_passes, endpoint_passes = 1, 1
        online, target, predictor_all, predictor_one = 603, 0, 0, 0
    elif update == 400:
        pair_passes, endpoint_passes = 3, 2
        online, target, predictor_all, predictor_one = 727, 727, 124, 0
    elif update == 1_000:
        pair_passes, endpoint_passes = 3, 2
        online, target, predictor_all, predictor_one = 1_099, 975, 248, 372
    else:
        raise ValueError("observation accounting update changed")
    return {
        "observation_pair_population_count": int(
            contract.SELECTION_ROLE_COUNTS["pairs"]
        ),
        "observation_endpoint_population_count": int(
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        ),
        "observation_pair_pass_count": pair_passes,
        "observation_endpoint_pass_count": endpoint_passes,
        "observation_pair_microbatch_count": pair_passes * pair_batches,
        "observation_endpoint_microbatch_count": endpoint_passes * endpoint_batches,
        "observation_microbatch_count": (
            pair_passes * pair_batches + endpoint_passes * endpoint_batches
        ),
        "observation_online_encoder_lift_forward_count": online,
        "observation_target_encoder_lift_forward_count": target,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": predictor_all,
        "observation_one_action_predictor_forward_count": predictor_one,
        "observation_predictor_forward_count": predictor_all + predictor_one,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    }


def _evaluate_observation(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    prior_metrics: Mapping[int, Mapping[str, Any]],
    integrity: Mapping[str, Any],
    joint_accounting: Mapping[str, Any],
) -> tuple[dict[str, Any], float | None]:
    _begin_observation_accounting(update)
    try:
        with _instrument_observation_calls(model, loader):
            metrics, baseline = _BASE_EVALUATE_OBSERVATION(
                runtime, model_api, model, loader, selection_pairs,
                selection_mapping, device, update=update,
                prior_metrics=prior_metrics, integrity=integrity,
                joint_accounting=joint_accounting,
            )
            metrics.update(_accounting_metrics(update))
            metrics.update(_observation_accounting(update))
            if update == 0:
                metrics.update(_reviewed_cpu_source_witness(
                    source_authority_exact=bool(
                        integrity["source_authority_exact"]
                    ),
                ))
                metrics["training_predictor_work_count"] = 0
            if update == 400:
                if baseline is None or not _FROZEN_REFERENCES:
                    raise RuntimeError(
                        "U400 observation omitted event references"
                    )
                metrics.update({
                    key: value for key, value in _FROZEN_REFERENCES.items()
                })
                metrics.update({
                    "T400_finite_strictly_positive": (
                        float(_FROZEN_REFERENCES["T400"]) > 0.0
                    ),
                    "B400_finite_strictly_positive": (
                        float(_FROZEN_REFERENCES["B400"]) > 0.0
                    ),
                    "T400_B400_frozen_before_joint_phase": True,
                    "B400_content_sha256": _FROZEN_REFERENCES[
                        "T400_B400_content_sha256"
                    ],
                    "training_predictor_work_count": 0,
                })
            if update >= 1_000:
                metrics.update(_event_observation_metrics(
                    runtime, model_api, model, loader, selection_pairs,
                    selection_mapping, device,
                ))
                metrics.update(_accounting_metrics(update))
                # Compatibility aliases keep inherited receipt readers useful.
                metrics.update({
                    "latent_prediction_loss": metrics[
                        "event_balanced_energy"
                    ],
                    "executed_action_beats_hardest_wrong_family_count": (
                        metrics["hardest_wrong_positive_family_count"]
                    ),
                    "same_action_target_nll": metrics["target_nll"],
                    "same_action_target_strict_win_rate": metrics[
                        "target_strict_win_rate"
                    ],
                    "same_action_target_positive_scene_count": metrics[
                        "target_positive_family_count"
                    ],
                    "same_action_correct_next_positive_family_count": metrics[
                        "target_positive_family_count"
                    ],
                    "non_hold_mean_executed_action_energy": metrics[
                        "mean_non_hold_executed_action_energy"
                    ],
                    "non_hold_mean_hold_or_zero_action_energy": metrics[
                        "mean_non_hold_hold_action_energy"
                    ],
                })
                for family, values in metrics.pop(
                    "event_scene_metrics"
                ).items():
                    metrics["scene_metrics"][family].update(values)
        metrics["observation_live_work"] = (
            _validate_completed_observation_accounting(update)
        )
        return metrics, baseline
    except BaseException as error:
        _mark_observation_failure(error)
        raise


def _train_probe(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    _reset_event_runtime_state()
    progress = kwargs.get("progress")
    if isinstance(progress, dict) and "gpu_started" in kwargs:
        progress.setdefault(
            "_gpu_active_started_monotonic", float(kwargs["gpu_started"])
        )
    try:
        model, probe = _BASE_TRAIN_PROBE(*args, **kwargs)
    except BaseException:
        if isinstance(progress, dict):
            observation_work = _observation_live_receipt()
            if (
                isinstance(observation_work, dict)
                and str(progress.get("stage"))
                == (
                    "observation_update_"
                    f"{observation_work.get('observation_update')}"
                )
            ):
                progress["_terminal_observation_work"] = observation_work
                if observation_work.get("observation_status") == "failed":
                    progress["_partial_observation_work"] = observation_work
            state = progress.get("_probe_failure_state")
            if isinstance(state, dict):
                update = int(state.get("updates", 0))
                accounting = _accounting_metrics(
                    update,
                    objective_evaluations=int(
                        state.get("objective_evaluations", 0)
                    ),
                    backward_calls=int(state.get("backward_calls", 0)),
                )
                state.update(accounting)
                state["presentations"] = int(
                    accounting["pair_presentations_loaded"]
                )
                integrity = state.get("integrity")
                if isinstance(integrity, dict):
                    integrity.update(accounting)
                if "_partial_observation_work" in progress:
                    state["partial_observation_work"] = progress[
                        "_partial_observation_work"
                    ]
                if "_terminal_observation_work" in progress:
                    state["terminal_observation_work"] = progress[
                        "_terminal_observation_work"
                    ]
        raise
    update = int(probe.get("updates", 0))
    for name in (
        "predictor_forward_count", "predictor_objective_count",
        "predictor_backward_count", "predictor_optimizer_updates",
        "joint_optimizer_updates", "shared_gradient_gate_pass_count",
    ):
        probe.setdefault(name, 0)
    probe.setdefault("phase_switch_receipt", None)
    accounting = _accounting_metrics(
        update,
        objective_evaluations=int(probe.get("objective_evaluations", 0)),
        backward_calls=int(probe.get("backward_calls", 0)),
    )
    probe.update(accounting)
    if isinstance(probe.get("integrity"), dict):
        probe["integrity"].update(accounting)
    return model, probe


def _evaluate_update_401_phase_switch(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Enrich the inherited update-401 receipt with event-specific routes."""

    value = dict(metrics)
    shared_route = bool(
        value.get("online_representation_gradient_finite_nonzero")
        and value.get("shared_gradient_contribution_gate_passed")
    )
    value.update({
        "event_joint_objective_formula_exact": bool(
            value.get("joint_objective_formula_exact")
        ),
        "all_action_logits_derived_mechanically_from_forward_event_energies": True,
        "semantic_gradient_finite_nonzero": shared_route,
        "dynamics_gradient_finite_nonzero": shared_route,
        "action_embedding_gradient_finite_nonzero": (
            _EVENT_ACCOUNTING["action_embedding_dynamics_gradient_update_count"] == 1
        ),
        "predictor_trunk_gradient_finite_nonzero": (
            _EVENT_ACCOUNTING["predictor_trunk_dynamics_gradient_update_count"] == 1
        ),
        "event_mean_head_gradient_finite_nonzero": (
            _EVENT_ACCOUNTING["event_mean_head_dynamics_gradient_update_count"] == 1
        ),
        "event_logit_head_gradient_finite_nonzero": (
            _EVENT_ACCOUNTING["event_logit_head_dynamics_gradient_update_count"] == 1
        ),
        "target_gradient_and_optimizer_membership_zero": bool(
            value.get("target_gradients_absent")
        ),
        "unit_weighted_joint_objective_exact": bool(
            value.get("joint_objective_formula_exact")
        ),
    })
    return _CONTRACT_PHASE_SWITCH(value)


def _canonical_root_entry(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == ROOT
    except (OSError, RuntimeError):
        return False


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, ...]:
    """Load inherited runtime plus the new model under one canonical root."""

    _ACTIVE_SOURCE_BINDINGS.clear()
    _ACTIVE_SOURCE_BINDINGS.update({
        str(relative): str(expected) for relative, expected in sources.items()
    })
    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if not _canonical_root_entry(entry)]
        sys.path.insert(0, str(ROOT))
        if (
            sys.path[0] != str(ROOT)
            or sum(_canonical_root_entry(entry) for entry in sys.path) != 1
        ):
            raise PermissionError("canonical repository import root is not exact")
        matched = _BASE._source_module(
            "_lewm_two_mode_event_delta_matched_runtime",
            _BASE.MATCHED_RUNNER_PATH,
        )
        runtime = matched._load_runtime()
        schedule_adapter = _BASE._source_module(
            "_lewm_two_mode_event_delta_schedule_adapter",
            _BASE.SCHEDULE_ADAPTER_PATH,
        )
        model_api = _BASE._source_module(
            "lewm.models.geometry_anchored_two_mode_event_delta_joint_jepa_v1",
            ROOT / contract.MODEL_RELATIVE_PATH,
        )
    finally:
        sys.path[:] = original_path
    if sys.path != original_path:
        raise PermissionError("post-stack import did not restore sys.path")
    model_class = getattr(model_api, contract.MODEL_CLASS_NAME, None)
    if (
        model_class is None
        or getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_class
    ):
        raise PermissionError("new model compatibility class binding changed")
    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)
    return matched, runtime, schedule_adapter, model_api


def _retain_returned_science(progress: dict[str, Any], scientific_result: Any) -> None:
    _RIGID._retain_returned_science(progress, scientific_result)


def _runtime_input_authority_receipt(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_inputs = authorization.get("runtime_inputs")
    if not isinstance(runtime_inputs, Mapping):
        return {
            "availability": "unknown_or_absent",
            "binding": None,
            "content_sha256": None,
            "matches_frozen_authorization_template_exact": False,
        }
    binding = dict(runtime_inputs)
    return {
        "availability": "available",
        "binding": binding,
        "content_sha256": contract.canonical_json_sha256(binding),
        "matches_frozen_authorization_template_exact": bool(
            binding == contract.runtime_authorization_template()
        ),
    }


def _execute(
    *,
    sources: Mapping[str, str],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    """Persist warning and event accounting evidence on every terminal path."""

    _ACTIVE_SOURCE_BINDINGS.clear()
    _ACTIVE_SOURCE_BINDINGS.update({
        str(relative): str(digest) for relative, digest in sources.items()
    })
    progress["_authorized_runtime_inputs"] = (
        _runtime_input_authority_receipt(authorization)
    )
    staged: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    normal_names = {
        "metrics.json", "artifact.json", "access.json", "result.json",
        "completed.json",
    }

    def stage_publication(
        path: Path, core: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        if path.parent == output_root and path.name in normal_names:
            if path.name in staged or path.exists():
                raise FileExistsError(f"duplicate staged receipt {path.name}")
            staged_core = dict(core)
            if path.name == "metrics.json":
                operation = dict(staged_core.get("operation", {}))
                probe = progress.get("_probe")
                if isinstance(probe, Mapping):
                    for name, value in probe.items():
                        if name.endswith("_count") or name in (
                            "objective_evaluations", "backward_calls",
                            "pair_presentations_loaded",
                        ):
                            operation[name] = value
                staged_core["operation"] = operation
            value = contract.with_content_sha256(staged_core)
            raw = contract.canonical_json_bytes(value) + b"\n"
            staged[path.name] = (path, value, raw)
            return value, raw
        return _BASE_PUBLISH_JSON(path, core)

    def defer_seal(path: Path) -> dict[str, Any]:
        if path == output_root:
            return {"deferred_until_all_receipts_validated": True}
        return _BASE_SEAL(path)

    def timed_load_development_inputs(*args: Any, **kwargs: Any) -> Any:
        loaded = _BASE_LOAD_DEVELOPMENT_INPUTS(*args, **kwargs)
        progress["_development_inputs_loaded"] = True
        # This boundary is before inherited GPU validation and loader setup.
        # It therefore closes the earlier gap where failures after input
        # validation incorrectly reported zero active-GPU elapsed time.
        progress.setdefault("_gpu_active_started_monotonic", time.monotonic())
        return loaded

    _BASE._publish_json = stage_publication
    _BASE._seal = defer_seal
    _BASE._load_development_inputs = timed_load_development_inputs
    try:
        result = _BASE_EXECUTE(
            sources=sources,
            authorization=authorization,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except DeterministicWarningFailure as error:
        started = progress.get("_gpu_active_started_monotonic")
        progress["_gpu_active_elapsed_seconds"] = (
            0.0 if started is None else max(0.0, time.monotonic() - float(started))
        )
        progress["_determinism"] = dict(error.warning_receipt)
        _retain_returned_science(progress, error.scientific_result)
        raise
    except BaseException as error:
        started = progress.get("_gpu_active_started_monotonic")
        progress["_gpu_active_elapsed_seconds"] = (
            0.0 if started is None else max(0.0, time.monotonic() - float(started))
        )
        receipt = getattr(error, "determinism_warning_receipt", None)
        if isinstance(receipt, Mapping):
            progress["_determinism"] = dict(receipt)
        raise
    else:
        expected = {
            "metrics.json", "artifact.json", "access.json", "result.json",
            "completed.json",
        }
        if set(staged) != expected:
            raise RuntimeError("normal terminal receipt staging is incomplete")
        for name in (
            "metrics.json", "artifact.json", "access.json", "result.json",
            "completed.json",
        ):
            path, _value, raw = staged[name]
            _BASE._write_exclusive(path, raw)
        expected_inventory = [
            "access.json", "artifact.json", "completed.json", "metrics.json",
            "reservation.json", "result.json",
        ]
        if _receipt_inventory(output_root) != expected_inventory:
            raise RuntimeError("normal terminal receipt inventory changed")
        _BASE_SEAL(output_root)
        return result
    finally:
        _BASE._publish_json = _BASE_PUBLISH_JSON
        _BASE._seal = _BASE_SEAL
        _BASE._load_development_inputs = _BASE_LOAD_DEVELOPMENT_INPUTS


_TOP_LEVEL_RECEIPT_NAMES = frozenset({
    "reservation.json", "metrics.json", "artifact.json", "access.json",
    "result.json", "failure.json", "completed.json",
})


def _receipt_inventory(output_root: Path) -> list[str]:
    return sorted(
        path.name for path in output_root.iterdir()
        if path.is_file() and path.name in _TOP_LEVEL_RECEIPT_NAMES
    )


def _terminal_trace_binding(
    output_root: Path, progress: Mapping[str, Any]
) -> Any:
    binding = progress.get("_training_trace_binding")
    if binding is None and isinstance(progress.get("_trace_rows"), list):
        trace_path = output_root / "training_trace.json"
        if not trace_path.exists():
            binding = _BASE._write_training_trace(
                output_root, progress["_trace_rows"]
            )
            if isinstance(progress, dict):
                progress["_training_trace_binding"] = binding
    return binding


def _terminal_access(progress: Mapping[str, Any]) -> dict[str, Any]:
    loader = progress.get("_loader")
    inputs = progress.get("_inputs")
    protected_zero_counts = {
        "rejected_checkpoint_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "written_checkpoint_read_count": 0,
        "training_trace_read_count": 0,
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "sealed_open_count": 0,
        "production_open_count": 0,
    }

    consumed = None if inputs is None else getattr(inputs, "consumed", None)
    if isinstance(consumed, Mapping):
        consumption_exact = True
        roles: list[str] | None = sorted({
            str(record.get("role"))
            for record in consumed.values()
            if isinstance(record, Mapping) and record.get("role") is not None
        })
        consumed_count: int | None = len(consumed)
    else:
        consumption_exact = False
        roles = None
        consumed_count = None

    if loader is None or inputs is None:
        raw_loader_receipt: Any = None
        model_facing_counts: Any = None
        if loader is not None:
            receipt_callable = getattr(loader, "receipt", None)
            if callable(receipt_callable):
                try:
                    raw_loader_receipt = receipt_callable()
                except BaseException as raw_error:
                    raw_loader_receipt = {
                        "raw_receipt_error_type": type(raw_error).__name__,
                        "raw_receipt_error_message_sha256": hashlib.sha256(
                            str(raw_error).encode("utf-8")
                        ).hexdigest(),
                    }
            counts_callable = getattr(
                loader, "model_facing_access_counts", None
            )
            if callable(counts_callable):
                try:
                    model_facing_counts = counts_callable()
                except BaseException as counts_error:
                    model_facing_counts = {
                        "availability": "unknown",
                        "error_type": type(counts_error).__name__,
                        "error_message_sha256": hashlib.sha256(
                            str(counts_error).encode("utf-8")
                        ).hexdigest(),
                    }
        reported_protected_counts = dict(protected_zero_counts)
        if isinstance(raw_loader_receipt, Mapping):
            for name in reported_protected_counts:
                if name in raw_loader_receipt:
                    reported_protected_counts[name] = raw_loader_receipt[name]
        return {
            "loader_receipt_available": raw_loader_receipt is not None,
            "access_phase": "before_loader_construction",
            "access_progress_stage": str(progress.get("stage", "unknown")),
            "roles_opened": roles,
            "consumed_record_count": consumed_count,
            "development_input_consumption_known_exact": consumption_exact,
            "model_facing_counts": model_facing_counts,
            "raw_loader_receipt": raw_loader_receipt,
            "raw_constructor_reads": progress.get("_raw_constructor_reads"),
            **reported_protected_counts,
        }
    try:
        return {
            **_BASE._access_receipt(loader, inputs),
            "loader_receipt_available": True,
            "access_phase": str(progress.get("stage", "unknown")),
            "development_input_consumption_known_exact": True,
        }
    except BaseException as access_error:
        raw_receipt: Any = None
        receipt_callable = getattr(loader, "receipt", None)
        if callable(receipt_callable):
            try:
                raw_receipt = receipt_callable()
            except BaseException as raw_error:
                raw_receipt = {
                    "raw_receipt_error_type": type(raw_error).__name__,
                    "raw_receipt_error_message_sha256": hashlib.sha256(
                        str(raw_error).encode("utf-8")
                    ).hexdigest(),
                }
        reported_protected_counts = dict(protected_zero_counts)
        if isinstance(raw_receipt, Mapping):
            for name in reported_protected_counts:
                if name in raw_receipt:
                    reported_protected_counts[name] = raw_receipt[name]
        return {
            "loader_receipt_available": False,
            "access_receipt_error_type": type(access_error).__name__,
            "access_receipt_error_message_sha256": hashlib.sha256(
                str(access_error).encode("utf-8")
            ).hexdigest(),
            "raw_loader_receipt": raw_receipt,
            "validated_forbidden_access_counts": None,
            "roles_opened": roles,
            "consumed_record_count": consumed_count,
            "development_input_consumption_known_exact": consumption_exact,
            "model_facing_counts": None,
            "access_phase": str(progress.get("stage", "unknown")),
            **reported_protected_counts,
        }


def _terminal_probe_state(progress: Mapping[str, Any]) -> Mapping[str, Any]:
    probe = progress.get("_probe")
    if isinstance(probe, Mapping):
        return probe
    partial = progress.get("_probe_failure_state")
    return partial if isinstance(partial, Mapping) else {}


def _terminal_operation(
    progress: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    names = (
        "updates", "presentations", "pair_presentations_loaded",
        "objective_evaluations", "backward_calls", "predictor_forward_count",
        "predictor_objective_count", "predictor_backward_count",
        "predictor_optimizer_updates", "joint_optimizer_updates",
        "shared_gradient_gate_pass_count", "target_ema_update_count",
        "semantic_term_evaluation_count",
        "event_persistence_term_evaluation_count", "action_term_evaluation_count",
        "target_term_evaluation_count", "context_term_evaluation_count",
        "registered_scalar_term_evaluation_count",
        "all_action_predictor_training_forward_count",
        "context_swap_predictor_training_forward_count",
        "online_encoder_lift_training_forward_count",
        "semantic_head_training_forward_count",
        "target_encoder_lift_training_forward_count",
    )
    return {
        name: state.get(name, progress.get(name, 0)) for name in names
    }


def _publish_partial_scientific_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: Mapping[str, Any],
    error: BaseException,
) -> None:
    """Publish a registered partial scientific/numerical gate normally."""

    state = _terminal_probe_state(progress)
    status = str(getattr(error, "control", contract.CONTROL_FAIL_JOINT_GRADIENT))
    observations = progress.get("_observations")
    observations = observations if isinstance(observations, list) else []
    checkpoints = progress.get("_checkpoint_bindings")
    checkpoints = checkpoints if isinstance(checkpoints, list) else []
    trace_binding = _terminal_trace_binding(output_root, progress)
    access_core = _terminal_access(progress)
    terminal_gate = {
        "kind": "partial_joint_scientific_numerical_gate",
        "passed": False,
        "control": status,
        "first_failure_stage": progress.get("stage"),
        "scientific_gate_evidence": True,
        "all_conjunctive": True,
        "conjuncts": {
            "joint_gradient_routes_and_ratio_remained_qualified": False,
        },
    }
    metrics, metrics_raw = _BASE._publish_json(
        output_root / "metrics.json",
        {
            "schema": contract.METRICS_SCHEMA,
            "status": status,
            "classification": "REGISTERED_PARTIAL_SCIENTIFIC_NUMERICAL_GATE_FAILURE",
            "observations": observations,
            "terminal_gate": terminal_gate,
            "phase_switch_receipt": state.get("phase_switch_receipt"),
            "operation": _terminal_operation(progress, state),
            "integrity": state.get("integrity", {}),
        },
    )
    artifact, artifact_raw = _BASE._publish_json(
        output_root / "artifact.json",
        {
            "schema": contract.ARTIFACT_SCHEMA,
            "status": status,
            "classification": "REGISTERED_PARTIAL_SCIENTIFIC_NUMERICAL_GATE_FAILURE",
            "checkpoints": checkpoints,
            "training_trace": trace_binding,
            "all_checkpoints_write_only_and_unqualified": True,
            "checkpoint_read_count_after_write": 0,
            "training_trace_read_count_after_write": 0,
        },
    )
    access, access_raw = _BASE._publish_json(
        output_root / "access.json",
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": status,
            "classification": "REGISTERED_PARTIAL_SCIENTIFIC_NUMERICAL_GATE_FAILURE",
            **access_core,
        },
    )
    result, result_raw = _BASE._publish_json(
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": status,
            "classification": "REGISTERED_PARTIAL_SCIENTIFIC_NUMERICAL_GATE_FAILURE",
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "metrics": _BASE._binding("metrics.json", metrics, metrics_raw),
            "artifact": _BASE._binding("artifact.json", artifact, artifact_raw),
            "access": _BASE._binding("access.json", access, access_raw),
            "hardware": progress.get("_hardware"),
            "determinism": progress.get("_determinism"),
            "schedule": progress.get("_schedule_receipt"),
            "n320_checkpoint": progress.get("_n320_checkpoint_binding"),
            "gpu_active_elapsed_seconds": progress.get("_gpu_active_elapsed_seconds"),
            "mechanism_passed": False,
            "checkpoint_qualified": False,
            "downstream_authority": "none",
            "retry_authorized": False,
        },
    )
    _BASE._publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": status,
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "metrics": _BASE._binding("metrics.json", metrics, metrics_raw),
            "artifact": _BASE._binding("artifact.json", artifact, artifact_raw),
            "access": _BASE._binding("access.json", access, access_raw),
            "result": _BASE._binding("result.json", result, result_raw),
            "receipt_inventory": [
                "reservation.json", "metrics.json", "artifact.json",
                "access.json", "result.json", "completed.json",
            ],
            "checkpoint_qualified": False,
            "retry_authorized": False,
        },
    )
    expected = [
        "access.json", "artifact.json", "completed.json", "metrics.json",
        "reservation.json", "result.json",
    ]
    if _receipt_inventory(output_root) != expected:
        raise RuntimeError("partial scientific receipt inventory changed")
    _BASE._seal(output_root)


def _publish_compact_operational_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: Mapping[str, Any],
    error: BaseException,
) -> None:
    """Publish the self-contained two-receipt operational terminal path."""

    state = _terminal_probe_state(progress)
    trace_binding = _terminal_trace_binding(output_root, progress)
    checkpoints = progress.get("_checkpoint_bindings")
    checkpoints = checkpoints if isinstance(checkpoints, list) else []
    observations = progress.get("_observations")
    observations = observations if isinstance(observations, list) else []
    access_core = _terminal_access(progress)
    public_progress = {
        name: value for name, value in progress.items()
        if not name.startswith("_")
    }
    failure, failure_raw = _BASE._publish_json(
        output_root / "failure.json",
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": contract.CONTROL_FAIL_OPERATIONAL,
            "classification": "OPERATIONAL_OR_INTEGRITY_FAILURE",
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "progress": public_progress,
            "first_failure_stage": progress.get("stage"),
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    "".join(traceback.format_exception(error)).encode("utf-8")
                ).hexdigest(),
            },
            "observations": observations,
            "partial_observation_work": progress.get(
                "_partial_observation_work"
            ),
            "terminal_observation_work": progress.get(
                "_terminal_observation_work"
            ),
            "terminal_gate": state.get("terminal_gate"),
            "phase_switch_receipt": state.get("phase_switch_receipt"),
            "operation": _terminal_operation(progress, state),
            "integrity": state.get("integrity", {}),
            "checkpoints": checkpoints,
            "training_trace": trace_binding,
            "access": access_core,
            "hardware": progress.get("_hardware"),
            "determinism": progress.get("_determinism"),
            "schedule": progress.get("_schedule_receipt"),
            "source_bindings": dict(_ACTIVE_SOURCE_BINDINGS),
            "authorized_runtime_inputs": progress.get(
                "_authorized_runtime_inputs",
                {
                    "availability": "unknown_or_absent",
                    "binding": None,
                    "content_sha256": None,
                    "matches_frozen_authorization_template_exact": False,
                },
            ),
            "n320_gate": progress.get("_n320_gate"),
            "n320_checkpoint": progress.get("_n320_checkpoint_binding"),
            "gpu_active_elapsed_seconds": progress.get("_gpu_active_elapsed_seconds"),
            "all_checkpoints_write_only_and_unqualified": True,
            "checkpoint_read_count_after_write": 0,
            "training_trace_read_count_after_write": 0,
            "mechanism_passed": False,
            "checkpoint_qualified": False,
            "downstream_authority": "none",
            "retry_resume_repair_or_replacement_authorized": False,
            "complete_failure_receipt": True,
        },
    )
    _BASE._publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": contract.CONTROL_FAIL_OPERATIONAL,
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "failure": _BASE._binding("failure.json", failure, failure_raw),
            "receipt_inventory": [
                "reservation.json", "failure.json", "completed.json"
            ],
            "checkpoint_qualified": False,
            "retry_authorized": False,
            "complete_failure_receipt": True,
        },
    )
    expected = ["completed.json", "failure.json", "reservation.json"]
    if _receipt_inventory(output_root) != expected:
        raise RuntimeError("operational receipt inventory changed")
    _BASE._seal(output_root)


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: Mapping[str, Any],
    error: BaseException,
) -> None:
    if (output_root / "completed.json").exists():
        _BASE._seal(output_root)
        return
    scientific_partial = bool(
        isinstance(error, _BASE.ScientificGateFailure)
        and getattr(error, "control", None) == contract.CONTROL_FAIL_JOINT_GRADIENT
    )
    if scientific_partial:
        _publish_partial_scientific_failure(
            output_root, reservation, reservation_raw, progress, error
        )
    else:
        _publish_compact_operational_failure(
            output_root, reservation, reservation_raw, progress, error
        )


def _rebind_inherited_runner() -> None:
    """Bind reviewed one-shot execution to the event-delta scientific hooks."""

    _RIGID.contract = contract
    _RIGID.RUNNER_PATH = RUNNER_PATH
    _RIGID.__file__ = str(RUNNER_PATH)
    _RIGID._rebind_inherited_runner()
    _V3.contract = contract
    _V3.RUNNER_PATH = RUNNER_PATH
    _V3.__file__ = str(RUNNER_PATH)
    _V2.contract = contract
    _V2.RUNNER_PATH = RUNNER_PATH
    _BASE.contract = contract
    _BASE.RUNNER_PATH = RUNNER_PATH
    _BASE.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _BASE.__file__ = str(RUNNER_PATH)
    _BASE._load_post_reservation_stack = _load_post_reservation_stack
    _BASE._tensor_state_sha256 = _tensor_state_sha256
    _BASE._parameter_receipt = _parameter_receipt
    _BASE._build_optimizer = _build_optimizer
    _BASE._semantic_terms = _semantic_terms
    _BASE._joint_terms = _joint_terms
    _BASE._persistence_baseline = _persistence_baseline
    _BASE._evaluate_observation = _evaluate_observation
    _BASE._train_probe = _train_probe
    _BASE._execute = _execute
    _BASE._terminal_failure = _terminal_failure
    _BASE._run_deterministic = _run_deterministic
    _BASE._is_allowed_rocm_determinism_warning = _is_allowed_rocm_determinism_warning
    contract.evaluate_update_401_phase_switch = _evaluate_update_401_phase_switch


_rebind_inherited_runner()


def run_isolated_import_preflight() -> dict[str, Any]:
    _rebind_inherited_runner()
    before_path = list(sys.path)
    sources = contract.current_source_bindings(ROOT)
    loaded = _load_post_reservation_stack(sources)
    if len(loaded) != 4 or sys.path != before_path:
        raise PermissionError("isolated event-delta import preflight failed")
    return {
        "post_reservation_stack_imported": True,
        "new_model_class_bound": True,
        "scalar_safe_state_hash_bound": True,
        "event_objective_hooks_bound": True,
        "canonical_root_count_during_lazy_import": 1,
        "sys_path_restored_exactly": True,
        "runtime_or_generated_inputs_opened": [],
        "checkpoints_tensors_traces_or_predecessor_outputs_opened": [],
        "accelerators_queried_or_used": [],
        "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _RIGID.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _RIGID.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
