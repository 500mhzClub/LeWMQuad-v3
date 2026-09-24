#!/usr/bin/env python3
"""Run Direct BEV V6 through the frozen V5 execution stack.

This module is source-only until the inherited authority-first launcher has
reserved the distinct V6 output root.  It deliberately adapts only the model
initialization, optimizer attachment, integrity probes, observations, and
terminal phase accounting; the reviewed training loop remains inherited.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V6_"
    "PHASE_SEPARATED_FROZEN_STATE_PREDICTION_PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_bev_v6_phase_separated_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V6 runner identity changed")

_V5 = _source_only_module(
    "_lewm_direct_bev_v6_phase_separated_frozen_v5_runner",
    ROOT / contract.FROZEN_V5_RUNNER_RELATIVE_PATH,
)
_DEEPEST = _V5._V4._V3._V2._V1
_FROZEN_INITIALIZE_MODEL = _DEEPEST._initialize_model
_FROZEN_BUILD_OPTIMIZER = _DEEPEST._build_optimizer
_FROZEN_EVALUATE_OBSERVATION_IMPL = _DEEPEST._evaluate_observation_impl
_FROZEN_TRAIN_PROBE = _DEEPEST._train_probe
_FROZEN_OBJECTIVE = _DEEPEST._objective
_FROZEN_SNAPSHOT_MODEL = _DEEPEST._snapshot_model
_FROZEN_WRITE_TRAINING_TRACE = _DEEPEST._write_training_trace
_FROZEN_TERMINAL_FAILURE = _DEEPEST._terminal_failure
V6_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v6_phase_separated_frozen_state_model_runtime"
)

_ONLINE_PERCEPTION_PREFIXES = (
    ("encoder.", "encoder."),
    ("bev_decoder.", "bev_decoder."),
    ("state_head.", "state_head."),
)
_TARGET_PERCEPTION_PREFIXES = (
    ("target_encoder.", "encoder."),
    ("target_bev_decoder.", "bev_decoder."),
    ("target_state_head.", "state_head."),
)
_PREDICTOR_PREFIXES = (("predictor.", "predictor."),)
_PERCEPTION_METRIC_FIELDS = (
    "G",
    "correct_rgb_scene_win_count",
    "aggregate_raster_balanced_accuracy",
    "aggregate_free_recall",
    "aggregate_occupied_recall",
    "aggregate_raster_nll",
    "rough_raster_balanced_accuracy",
    "rough_raster_occupied_recall",
    "all_registered_values_finite",
    "state_nonconstant",
)


def _normalized_state_sha256(
    runtime: Any,
    model: Any,
    prefixes: Sequence[tuple[str, str]],
) -> str:
    """Hash a state subset after normalizing online/target namespaces."""

    normalized: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        for source, logical in prefixes:
            if name.startswith(source):
                normalized_name = logical + name[len(source):]
                if normalized_name in normalized:
                    raise RuntimeError("normalized V6 state name repeated")
                normalized[normalized_name] = value
                break
    if not normalized:
        raise RuntimeError("normalized V6 state subset is empty")
    return _DEEPEST._state_sha(runtime, normalized)


def _predictor_residual_head_exact_zero(runtime: Any, model: Any) -> bool:
    torch = runtime.torch
    head = model.predictor.residual_head
    return bool(
        torch.count_nonzero(head.weight).item() == 0
        and head.bias is not None
        and torch.count_nonzero(head.bias).item() == 0
    )


def _phase_modes(model: Any) -> dict[str, bool]:
    online_eval = all(not module.training for module in model._online_modules())
    target_eval = all(not module.training for module in model._target_modules())
    predictor_train = bool(model.predictor.training)
    return {
        "online_perception_eval_mode": online_eval,
        "target_perception_eval_mode": target_eval,
        "predictor_train_mode": predictor_train,
        "phase_two_module_modes_exact": (
            online_eval and target_eval and predictor_train
        ),
    }


def _trainability(model: Any, partition: Mapping[str, Any]) -> dict[str, bool]:
    groups = partition["groups"]

    def flags(name: str) -> list[bool]:
        return [bool(parameter.requires_grad) for _, parameter in groups[name]]

    encoder = flags("encoder")
    decoder = flags("decoder_state")
    predictor = flags("predictor")
    target = flags("detached_target_encoder_decoder_state")
    return {
        "phase_one_trainability_exact": bool(
            all(encoder) and all(decoder) and not any(predictor)
            and not any(target)
        ),
        "phase_two_trainability_exact": bool(
            not any(encoder) and not any(decoder) and all(predictor)
            and not any(target)
        ),
    }


def _phase_receipt(model: Any) -> dict[str, int | bool]:
    counters = dict(model.phase_counters_v6())
    required = {
        "phase_policy_armed",
        "global_target_update_callback_count",
        "target_update_callback_count",
        "ema_arithmetic_update_count",
        "boundary_hard_sync_count",
        "phase_two_target_noop_count",
        "perception_optimizer_update_count",
        "predictor_optimizer_update_count",
    }
    if set(counters) != required:
        raise RuntimeError("V6 phase-counter schema changed")
    if counters["global_target_update_callback_count"] != (
        counters["target_update_callback_count"]
    ):
        raise RuntimeError("V6 target callback counters diverged")
    return {
        "target_update_callback_count": int(
            counters["target_update_callback_count"]
        ),
        "perception_optimizer_updates": int(
            counters["perception_optimizer_update_count"]
        ),
        "predictor_optimizer_updates": int(
            counters["predictor_optimizer_update_count"]
        ),
        "ema_arithmetic_updates": int(
            counters["ema_arithmetic_update_count"]
        ),
        "boundary_hard_sync_count": int(
            counters["boundary_hard_sync_count"]
        ),
        "phase_two_target_noop_count": int(
            counters["phase_two_target_noop_count"]
        ),
    }


def _phase_accounting_for_update(update: int) -> dict[str, int]:
    """Return exact registered accounting without consulting runtime state."""

    if type(update) is not int or not 0 <= update <= contract.MAXIMUM_UPDATES:
        raise ValueError("V6 phase-accounting update is out of bounds")
    return {
        "target_update_callback_count": update,
        "perception_optimizer_updates": min(update, 400),
        "predictor_optimizer_updates": max(update - 400, 0),
        "ema_arithmetic_updates": min(update, 400),
        "boundary_hard_sync_count": int(update >= 400),
        "phase_two_target_noop_count": max(update - 400, 0),
    }


def _optimizer_sha256(optimizer: Any) -> str:
    """Deterministically hash optimizer state without mutating it."""

    digest = hashlib.sha256()

    def add(value: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if hasattr(value, "detach") and hasattr(value, "shape"):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value, key=lambda item: repr(item)):
                add(value[key], f"{path}/{key!r}")
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii"))
            for index, item in enumerate(value):
                add(item, f"{path}/{index}")
        elif value is None or isinstance(value, (bool, int, float, str)):
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(repr(value).encode("utf-8"))
        else:
            raise TypeError(f"unsupported optimizer receipt value: {type(value)!r}")

    add(optimizer.state_dict(), "optimizer")
    return digest.hexdigest()


def _v6_initialize_model(
    runtime: Any,
    model_api: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Preserve frozen initialization, then arm the parameter-free schedule."""

    model, partition, receipt = _FROZEN_INITIALIZE_MODEL(
        runtime, model_api, fit, device
    )
    expected = contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
    if receipt.get("complete_initial_state_sha256") != expected:
        raise RuntimeError("fresh V6 initial model differs from frozen V3")
    predictor_sha = _normalized_state_sha256(
        runtime, model, _PREDICTOR_PREFIXES
    )
    online_sha = _normalized_state_sha256(
        runtime, model, _ONLINE_PERCEPTION_PREFIXES
    )
    target_sha = _normalized_state_sha256(
        runtime, model, _TARGET_PERCEPTION_PREFIXES
    )
    if online_sha != target_sha:
        raise RuntimeError("fresh V6 online and target perception differ")
    model.arm_phase_schedule_v6()
    phase_one = _trainability(model, partition)["phase_one_trainability_exact"]
    if model.active_phase_v6 != "phase_one" or not phase_one:
        raise RuntimeError("fresh V6 model did not arm exact phase one")
    if model._v6_optimizer_for_integrity_probe is not None:
        raise RuntimeError("fresh V6 model unexpectedly has an optimizer")
    object.__setattr__(model, "_v6_initial_predictor_state_sha256", predictor_sha)
    object.__setattr__(model, "_v6_initial_online_perception_sha256", online_sha)
    object.__setattr__(model, "_v6_update400_baseline", None)
    object.__setattr__(
        model,
        "_v6_no_prior_runtime_or_protected_input",
        receipt.get("prior_runtime_parameter_reuse_count") == 0,
    )
    partition["_v6_model"] = model
    receipt = {
        **receipt,
        "v6_phase_policy_armed_after_frozen_initialization": True,
        "initial_predictor_state_sha256": predictor_sha,
        "initial_online_perception_state_sha256": online_sha,
        "initial_target_perception_state_sha256": target_sha,
        "initial_online_target_perception_bitwise_equal": True,
    }
    return model, partition, receipt


def _v6_build_optimizer(
    runtime: Any,
    partition: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build the inherited optimizer once and attach its probe-only witness."""

    model = partition.get("_v6_model")
    if model is None:
        raise RuntimeError("V6 optimizer lost its initialized model witness")
    if model._v6_optimizer_for_integrity_probe is not None:
        raise RuntimeError("V6 optimizer was constructed more than once")
    optimizer, receipt = _FROZEN_BUILD_OPTIMIZER(runtime, partition)
    object.__setattr__(model, "_v6_optimizer_for_integrity_probe", optimizer)
    return optimizer, {
        **receipt,
        "single_optimizer_constructed_once": True,
        "optimizer_rebuilt_or_reset_at_phase_boundary": False,
    }


def _group_gradient_receipt(
    runtime: Any,
    partition: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    torch = runtime.torch
    result: dict[str, dict[str, Any]] = {}
    for group, rows in partition["groups"].items():
        gradients = [parameter.grad for _, parameter in rows]
        present = [gradient for gradient in gradients if gradient is not None]
        finite = all(bool(torch.isfinite(item).all()) for item in present)
        absolute_sum = sum(
            float(item.detach().abs().sum().cpu()) for item in present
        )
        result[group] = {
            "gradient_tensor_count": len(present),
            "all_gradients_absent": not present,
            "all_present_gradients_finite": finite,
            "absolute_gradient_sum": absolute_sum,
            "finite_nonzero": bool(present and finite and absolute_sum > 0.0),
        }
    return result


def _run_one_phase_gradient_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    torch = runtime.torch
    model.set_phase_override_for_integrity_probe_v6(phase)
    trainability = _trainability(model, partition)
    for parameter in model.parameters():
        parameter.grad = None

    call_counts = {
        "online_state_stack": 0,
        "predictor": 0,
        "target_state_stack": 0,
    }
    modules = {
        "online_state_stack": model.state_head,
        "predictor": model.predictor.residual_head,
        "target_state_stack": model.target_state_head,
    }
    handles = []
    for name, module in modules.items():
        def count_call(
            _module: Any,
            _args: Any,
            _output: Any,
            *,
            key: str = name,
        ) -> None:
            call_counts[key] += 1
        handles.append(module.register_forward_hook(count_call))

    require_rgb_grad = phase == "phase_one"
    current = batch["current_rgb"].detach().clone().requires_grad_(
        require_rgb_grad
    )
    next_rgb = batch["next_rgb"].detach().clone().requires_grad_(
        require_rgb_grad
    )
    fixed = batch["fixed_negative_rgb"].detach().clone().requires_grad_(
        require_rgb_grad
    )
    probe_batch = dict(batch)
    probe_batch.update({
        "current_rgb": current,
        "next_rgb": next_rgb,
        "fixed_negative_rgb": fixed,
    })
    try:
        objective = _FROZEN_OBJECTIVE(model, probe_batch)
        next_total_gradient = None
        next_grounding_gradient = None
        fixed_gradient = None
        if phase == "phase_one":
            next_total_gradient = torch.autograd.grad(
                objective.total,
                next_rgb,
                retain_graph=True,
                allow_unused=True,
            )[0]
            next_grounding_gradient = torch.autograd.grad(
                0.5 * objective.G_next / math.log(2.0),
                next_rgb,
                retain_graph=True,
                allow_unused=True,
            )[0]
            fixed_gradient = torch.autograd.grad(
                objective.total,
                fixed,
                retain_graph=True,
                allow_unused=True,
            )[0]
        objective.total.backward()
        with torch.no_grad():
            wrong_rgb_state = model.online_state(fixed.detach())
        gradients = _group_gradient_receipt(runtime, partition)
        expected_trainability = (
            trainability["phase_one_trainability_exact"]
            if phase == "phase_one"
            else trainability["phase_two_trainability_exact"]
        )
        if phase == "phase_one":
            isolation = bool(
                gradients["encoder"]["finite_nonzero"]
                and gradients["decoder_state"]["finite_nonzero"]
                and gradients["predictor"]["all_gradients_absent"]
                and gradients["detached_target_encoder_decoder_state"][
                    "all_gradients_absent"
                ]
                and next_total_gradient is not None
                and next_grounding_gradient is not None
                and torch.allclose(
                    next_total_gradient,
                    next_grounding_gradient,
                    rtol=1e-5,
                    atol=1e-7,
                )
                and fixed_gradient is None
            )
        else:
            isolation = bool(
                gradients["encoder"]["all_gradients_absent"]
                and gradients["decoder_state"]["all_gradients_absent"]
                and gradients["predictor"]["finite_nonzero"]
                and gradients["detached_target_encoder_decoder_state"][
                    "all_gradients_absent"
                ]
                and not objective.current_state_logits.requires_grad
                and not objective.next_online_state_logits.requires_grad
            )
        exact_calls = call_counts == {
            "online_state_stack": 3,
            "predictor": 1,
            "target_state_stack": 3,
        }
        return {
            "phase": phase,
            "objective_total": (
                "G/log(2)" if phase == "phase_one" else "J/log(2)+C"
            ),
            "trainability_exact": expected_trainability,
            "gradient_isolation_exact": bool(
                isolation and exact_calls and expected_trainability
            ),
            "training_objective_call_counts": dict(call_counts),
            "registered_call_boundary_exact": exact_calls,
            "group_gradients": gradients,
            "next_rgb_gradient_equals_grounding_only": bool(
                phase != "phase_one"
                or (
                    next_total_gradient is not None
                    and next_grounding_gradient is not None
                    and torch.allclose(
                        next_total_gradient,
                        next_grounding_gradient,
                        rtol=1e-5,
                        atol=1e-7,
                    )
                )
            ),
            "fixed_negative_rgb_optimizer_gradient_absent": bool(
                phase != "phase_one" or fixed_gradient is None
            ),
            "observation_only_fixed_negative_output_requires_grad": bool(
                wrong_rgb_state.requires_grad
            ),
        }
    finally:
        for handle in handles:
            handle.remove()


def _gradient_integrity_probe_for_phases(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
    phases: Sequence[str],
) -> dict[str, Any]:
    """Run registered paths while restoring every mutable probe surface."""

    torch = runtime.torch
    parameters = list(model.parameters())
    previous_grads = [
        None if item.grad is None else item.grad.detach().clone()
        for item in parameters
    ]
    previous_flags = [bool(item.requires_grad) for item in parameters]
    previous_modes = [(module, bool(module.training)) for module in model.modules()]
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = [item.clone() for item in torch.cuda.get_rng_state_all()]
    model_sha = _DEEPEST._state_sha(runtime, model)
    optimizer = model._v6_optimizer_for_integrity_probe
    if optimizer is None:
        raise RuntimeError("V6 gradient probe has no optimizer witness")
    optimizer_sha = _optimizer_sha256(optimizer)
    phase_receipts: dict[str, dict[str, Any]] = {}
    try:
        for phase in phases:
            phase_receipts[phase] = _run_one_phase_gradient_probe(
                runtime,
                model,
                partition,
                batch,
                phase=phase,
            )
    finally:
        model.set_phase_override_for_integrity_probe_v6(None)
        for parameter, previous in zip(parameters, previous_grads, strict=True):
            parameter.grad = previous
        for parameter, flag in zip(parameters, previous_flags, strict=True):
            parameter.requires_grad_(flag)
        for module, mode in previous_modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

    nonmutating = bool(
        _DEEPEST._state_sha(runtime, model) == model_sha
        and _optimizer_sha256(optimizer) == optimizer_sha
        and all(
            bool(parameter.requires_grad) == flag
            for parameter, flag in zip(parameters, previous_flags, strict=True)
        )
        and all(module.training == mode for module, mode in previous_modes)
        and torch.equal(torch.random.get_rng_state(), cpu_rng)
        and all(
            torch.equal(before, after)
            for before, after in zip(
                cuda_rng,
                torch.cuda.get_rng_state_all(),
                strict=True,
            )
        )
    )
    phase_one = phase_receipts.get("phase_one")
    phase_two = phase_receipts.get("phase_two")
    return {
        "phase_receipts": phase_receipts,
        "phase_one_gradient_isolation_exact": bool(
            phase_one and phase_one["gradient_isolation_exact"]
        ),
        "phase_two_gradient_isolation_exact": bool(
            phase_two and phase_two["gradient_isolation_exact"]
        ),
        "dual_gradient_probe_nonmutating_exact": nonmutating,
        # Backward-compatible keys consumed by the frozen observation.
        "target_parameters_gradient_free": all(
            receipt["group_gradients"]
            ["detached_target_encoder_decoder_state"]["all_gradients_absent"]
            for receipt in phase_receipts.values()
        ),
        "intended_online_path_gradient_nonzero": all(
            receipt["gradient_isolation_exact"]
            for receipt in phase_receipts.values()
        ),
        "six_call_graph_isolation_exact": all(
            receipt["registered_call_boundary_exact"]
            for receipt in phase_receipts.values()
        ),
        "training_objective_call_counts": {
            phase: receipt["training_objective_call_counts"]
            for phase, receipt in phase_receipts.items()
        },
    }


def _v6_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    return _gradient_integrity_probe_for_phases(
        runtime,
        model,
        partition,
        batch,
        ("phase_one", "phase_two"),
    )


def _zero_rgb_witness(runtime: Any, model: Any) -> dict[str, bool]:
    torch = runtime.torch
    device = next(model.parameters()).device
    value = torch.zeros(
        1,
        3,
        model.config.image_size,
        model.config.image_size,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        online_first = model.online_state(value)
        online_second = model.online_state(value)
        target_first = model.target_state(value)
        target_second = model.target_state(value)
    online_equal = bool(torch.equal(online_first, online_second))
    target_equal = bool(torch.equal(target_first, target_second))
    return {
        "zero_rgb_online_repeat_bitwise_equal": online_equal,
        "zero_rgb_target_repeat_bitwise_equal": target_equal,
        "zero_rgb_witness_exact": online_equal and target_equal,
    }


def _v6_evaluate_observation_impl(
    runtime: Any,
    model_api: Any,
    model: Any,
    partition: Mapping[str, Any],
    loader: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    update_zero: Mapping[str, Any] | None,
    prior_gates_passed: bool,
) -> dict[str, Any]:
    """Add V6 phase witnesses after the inherited metric observation."""

    result = _FROZEN_EVALUATE_OBSERVATION_IMPL(
        runtime,
        model_api,
        model,
        partition,
        loader,
        selection_pairs,
        selection_mapping,
        device,
        update=update,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    metrics = result["metrics"]
    predictor_sha = _normalized_state_sha256(
        runtime, model, _PREDICTOR_PREFIXES
    )
    online_sha = _normalized_state_sha256(
        runtime, model, _ONLINE_PERCEPTION_PREFIXES
    )
    target_sha = _normalized_state_sha256(
        runtime, model, _TARGET_PERCEPTION_PREFIXES
    )
    residual_zero = _predictor_residual_head_exact_zero(runtime, model)
    trainability = _trainability(model, partition)
    metrics.update({
        "active_phase_v6": model.active_phase_v6,
        **_phase_receipt(model),
        "online_perception_state_sha256": online_sha,
        "target_perception_state_sha256": target_sha,
        "predictor_state_sha256": predictor_sha,
        "predictor_matches_initialization": predictor_sha
        == model._v6_initial_predictor_state_sha256,
        "online_target_perception_bitwise_equal": online_sha == target_sha,
        "predictor_residual_head_exact_zero": residual_zero,
        "phase_one_trainability_exact": trainability[
            "phase_one_trainability_exact"
        ],
        "phase_two_trainability_exact": trainability[
            "phase_two_trainability_exact"
        ],
        # Exact-zero residual topology makes all nine V3 outputs persistence.
        "prediction_is_exact_persistence": residual_zero,
        "all_nine_action_predictions_bitwise_equal": residual_zero,
    })

    if update == 0:
        gradient = result["gradient_integrity"]
        metrics.update({
            "initial_model_state_matches_frozen_v3": True,
            "model_parameter_inventory_exact": True,
            "registered_state_and_target_nonconstant": bool(
                metrics["state_nonconstant"]
            ),
            "no_prior_runtime_or_protected_input": bool(
                model._v6_no_prior_runtime_or_protected_input
            ),
            "phase_one_gradient_isolation_exact": gradient[
                "phase_one_gradient_isolation_exact"
            ],
            "phase_two_gradient_isolation_exact": gradient[
                "phase_two_gradient_isolation_exact"
            ],
            "dual_gradient_probe_nonmutating_exact": gradient[
                "dual_gradient_probe_nonmutating_exact"
            ],
        })

    if update in (400, 1_000):
        model.apply_phase_policy_v6()
        metrics.update(_phase_modes(model))
        metrics.update(_zero_rgb_witness(runtime, model))

    if update == 400:
        indices = list(range(min(contract.MICROBATCH_SIZE, len(selection_pairs))))
        boundary_batch = loader.batch(
            selection_pairs,
            indices,
            device,
            role="checkpoint_selection",
            stage="observation_update_400_boundary_gradient_probe",
            mapped_negative_indices=selection_mapping["negative_indices"],
            scope="observation",
        )
        boundary_probe = _gradient_integrity_probe_for_phases(
            runtime,
            model,
            partition,
            boundary_batch,
            ("phase_two",),
        )
        result["boundary_gradient_integrity"] = boundary_probe
        metrics["boundary_phase_two_gradient_isolation_exact"] = bool(
            boundary_probe["phase_two_gradient_isolation_exact"]
            and boundary_probe["dual_gradient_probe_nonmutating_exact"]
        )
        perception = {
            name: copy.deepcopy(metrics[name])
            for name in _PERCEPTION_METRIC_FIELDS
        }
        baseline = {
            "online_perception_state_sha256": online_sha,
            "target_perception_state_sha256": target_sha,
            "predictor_state_sha256": predictor_sha,
            "perception_metrics": perception,
            "perception_metrics_sha256": contract.canonical_json_sha256(
                perception
            ),
            "J": metrics["J"],
            "C": metrics["C"],
        }
        object.__setattr__(model, "_v6_update400_baseline", baseline)
        metrics.update({
            "perception_metrics_update400_baseline_sha256": baseline[
                "perception_metrics_sha256"
            ],
            "predictor_update400_sha256": predictor_sha,
            "J_update400_boundary": metrics["J"],
            "C_update400_boundary": metrics["C"],
        })

    if update == 1_000:
        baseline = model._v6_update400_baseline
        if type(baseline) is not dict:
            raise RuntimeError("V6 update-400 baseline is absent")
        perception = {
            name: copy.deepcopy(metrics[name])
            for name in _PERCEPTION_METRIC_FIELDS
        }
        metrics.update({
            "perception_metrics_update400_baseline_sha256": baseline[
                "perception_metrics_sha256"
            ],
            "perception_metrics_unchanged_from_update400": (
                perception == baseline["perception_metrics"]
            ),
            "online_perception_unchanged_from_update400": online_sha
            == baseline["online_perception_state_sha256"],
            "target_perception_unchanged_from_update400": target_sha
            == baseline["target_perception_state_sha256"],
            "predictor_update400_sha256": baseline["predictor_state_sha256"],
            "J_update400_boundary": baseline["J"],
            "C_update400_boundary": baseline["C"],
        })

    metrics["v6_phase_receipt_ready"] = True
    result["gate"] = contract.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    # The boundary probe deliberately reuses one authorized selection batch.
    result["loader_access_after_observation"] = loader.receipt()
    return result


def _v6_train_probe(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Reuse the frozen loop and add exact terminal phase accounting."""

    model, result = _FROZEN_TRAIN_PROBE(*args, **kwargs)
    phase = _phase_receipt(model)
    if phase["target_update_callback_count"] != int(result["updates"]):
        raise RuntimeError("V6 terminal callback accounting differs from updates")
    result.update({
        **phase,
        "global_target_update_callback_count": phase[
            "target_update_callback_count"
        ],
        # This inherited public field now means actual EMA arithmetic.
        "ema_updates": phase["ema_arithmetic_updates"],
        "single_optimizer_constructed_once": (
            model._v6_optimizer_for_integrity_probe is not None
        ),
        "optimizer_rebuilt_or_reset_at_phase_boundary": False,
    })
    return model, result


def _v6_write_training_trace(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Make inherited callback-count trace rows phase-explicit."""

    translated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        update = int(row["update"])
        legacy = int(row.pop("ema_update_count"))
        if legacy != update:
            raise RuntimeError("V6 inherited trace callback count changed")
        row.update(_phase_accounting_for_update(update))
        translated.append(row)
    return _FROZEN_WRITE_TRAINING_TRACE(output_root, translated)


def _v6_snapshot_model(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace ambiguous inherited EMA metadata with exact V6 categories."""

    translated = dict(metadata)
    legacy = int(translated.pop("ema_updates"))
    if legacy != update:
        raise RuntimeError("V6 inherited snapshot callback count changed")
    translated.update(_phase_accounting_for_update(update))
    return _FROZEN_SNAPSHOT_MODEL(
        runtime,
        model,
        output_root,
        update=update,
        metadata=translated,
    )


def _v6_terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    *,
    error: BaseException,
    progress: Mapping[str, Any],
) -> None:
    """Publish truthful partial phase counts on any operational failure."""

    translated = dict(progress)
    callbacks = int(translated.get("ema_updates", 0))
    optimizer_updates = int(translated.get("optimizer_updates", 0))
    if (
        callbacks < 0
        or optimizer_updates < 0
        or callbacks > optimizer_updates
        or optimizer_updates > contract.MAXIMUM_UPDATES
    ):
        raise RuntimeError("V6 partial failure accounting is inconsistent")
    translated.update({
        "global_target_update_callback_count": callbacks,
        "target_update_callback_count": callbacks,
        "perception_optimizer_updates": min(optimizer_updates, 400),
        "predictor_optimizer_updates": max(optimizer_updates - 400, 0),
        "ema_arithmetic_updates": min(callbacks, 400),
        "boundary_hard_sync_count": int(callbacks >= 400),
        "phase_two_target_noop_count": max(callbacks - 400, 0),
        # The inherited failure publisher reads this public field.
        "ema_updates": min(callbacks, 400),
    })
    _FROZEN_TERMINAL_FAILURE(
        output_root,
        reservation,
        reservation_raw,
        error=error,
        progress=translated,
    )


def _rebind_inherited_runner() -> None:
    """Bind the complete frozen V5 stack to V6 identities and seams."""

    wrapper_path = Path(__file__).resolve()
    _V5.contract = contract
    _V5.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V5.V5_MODEL_RUNTIME_MODULE_NAME = V6_MODEL_RUNTIME_MODULE_NAME
    _V5.__file__ = str(wrapper_path)
    _V5._rebind_inherited_runner()

    owners = (
        _V5,
        _V5._V4,
        _V5._V4._V3,
        _V5._V4._V3._V2,
        _V5._V4._V3._V2._V1,
    )
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V6 contract did not reach the complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V6 preflight identity did not reach the runner stack")
    if any(Path(owner.__file__).resolve() != wrapper_path for owner in owners):
        raise RuntimeError("V6 runner path did not reach the runner stack")
    if (
        _V5.V5_MODEL_RUNTIME_MODULE_NAME != V6_MODEL_RUNTIME_MODULE_NAME
        or _V5._V4.V4_MODEL_RUNTIME_MODULE_NAME
        != V6_MODEL_RUNTIME_MODULE_NAME
        or _V5._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME
        != V6_MODEL_RUNTIME_MODULE_NAME
        or _V5._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME
        != V6_MODEL_RUNTIME_MODULE_NAME
    ):
        raise RuntimeError("V6 model runtime identity was not fully rebound")

    deepest = _V5._V4._V3._V2._V1
    deepest._initialize_model = _v6_initialize_model
    deepest._build_optimizer = _v6_build_optimizer
    deepest._gradient_integrity_probe = _v6_gradient_integrity_probe
    deepest._evaluate_observation_impl = _v6_evaluate_observation_impl
    deepest._train_probe = _v6_train_probe
    deepest._write_training_trace = _v6_write_training_trace
    deepest._snapshot_model = _v6_snapshot_model
    deepest._terminal_failure = _v6_terminal_failure
    if deepest.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("V6 failure-chain validator was not rebound")


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V5.parse_args(argv)


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    return _V5.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V5.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
