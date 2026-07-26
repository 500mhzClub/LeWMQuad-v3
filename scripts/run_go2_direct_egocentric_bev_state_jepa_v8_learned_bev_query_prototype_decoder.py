#!/usr/bin/env python3
"""Run the capped Direct-BEV V8 learned-query/prototype probe.

The frozen V7 authority and data stack is reused, but the real V8 science
delta owns initialization, optimizer membership, the update-zero gradient
witness, observation receipts, and the exact 4,000-presentation schedule
prefix.  The public wrappers always delegate directly to the deepest V1 leaf
so an intermediate successor cannot replace one of those seams.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V8_"
    "LEARNED_BEV_QUERY_PROTOTYPE_DECODER_PREFLIGHT_JSON"
)
V8_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v8_learned_query_prototype_model_runtime"
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
    "_lewm_direct_bev_v8_learned_query_prototype_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V8 runner identity changed")

_V7 = _source_only_module(
    "_lewm_direct_bev_v8_frozen_v7_runner",
    ROOT / contract.FROZEN_V7_RUNNER_RELATIVE_PATH,
)
_V6 = _V7._V6
_LEAF = _V7._LEAF
_BASE_INITIALIZE_MODEL = _V6._V5._V4._FROZEN_INITIALIZE_MODEL
_FROZEN_OBJECTIVE = _V6._FROZEN_OBJECTIVE

_ONLINE_PERCEPTION_PREFIXES = _V6._ONLINE_PERCEPTION_PREFIXES
_TARGET_PERCEPTION_PREFIXES = _V6._TARGET_PERCEPTION_PREFIXES
_PREDICTOR_PREFIXES = _V6._PREDICTOR_PREFIXES


def _component_state_sha256(runtime: Any, module: Any) -> str:
    return _LEAF._state_sha(runtime, module)


def _install_predictor_call_witness(model: Any) -> None:
    """Count any predictor submodule execution without changing state_dict."""

    counter = {"count": 0}

    def observe(_module: Any, _inputs: Any) -> None:
        counter["count"] += 1

    handles = tuple(
        module.register_forward_pre_hook(observe)
        for module in model.predictor.modules()
        if not tuple(module.children())
    )
    if not handles:
        raise RuntimeError("V8 predictor call witness has no leaf modules")
    object.__setattr__(model, "_v8_predictor_call_counter", counter)
    object.__setattr__(model, "_v8_predictor_call_handles", handles)


def _v8_initialize_model(
    runtime: Any,
    model_api: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Construct fresh V8 state without using V4/V6's old full-state hash."""

    model, partition, receipt = _BASE_INITIALIZE_MODEL(
        runtime, model_api, fit, device
    )
    decoder_sha = _component_state_sha256(runtime, model.bev_decoder)
    head_sha = _component_state_sha256(runtime, model.state_head)
    predictor_sha = _V6._normalized_state_sha256(
        runtime, model, _PREDICTOR_PREFIXES
    )
    online_sha = _V6._normalized_state_sha256(
        runtime, model, _ONLINE_PERCEPTION_PREFIXES
    )
    target_sha = _V6._normalized_state_sha256(
        runtime, model, _TARGET_PERCEPTION_PREFIXES
    )
    if (
        decoder_sha != contract.V8_INITIAL_DECODER_STATE_SHA256
        or head_sha != contract.V8_INITIAL_PROTOTYPE_HEAD_STATE_SHA256
        or predictor_sha != contract.V8_INITIAL_PREDICTOR_STATE_SHA256
        or online_sha != target_sha
    ):
        raise RuntimeError("fresh V8 component initialization changed")
    if partition["receipt"] != contract.MODEL_PARAMETER_INVENTORY:
        raise RuntimeError("fresh V8 parameter partition changed")

    model.arm_phase_schedule_v6()
    trainability = _V6._trainability(model, partition)
    if (
        model.active_phase_v6 != "phase_one"
        or not trainability["phase_one_trainability_exact"]
        or any(parameter.requires_grad for parameter in model.predictor.parameters())
    ):
        raise RuntimeError("fresh V8 did not arm perception-only phase")
    _install_predictor_call_witness(model)
    object.__setattr__(model, "_v6_initial_predictor_state_sha256", predictor_sha)
    object.__setattr__(model, "_v6_initial_online_perception_sha256", online_sha)
    object.__setattr__(
        model,
        "_v6_no_prior_runtime_or_protected_input",
        receipt.get("prior_runtime_parameter_reuse_count") == 0,
    )
    object.__setattr__(model, "_v8_update100_metrics", None)
    object.__setattr__(model, "_v8_initial_components_exact", True)
    object.__setattr__(model, "_v8_initial_complete_state_sha256", receipt[
        "complete_initial_state_sha256"
    ])
    partition["_v6_model"] = model
    partition["_v8_model"] = model
    return model, partition, {
        **receipt,
        "fresh_modules_in_draw_order": list(contract.V8_FRESH_DRAW_ORDER),
        "v8_initial_decoder_state_sha256": decoder_sha,
        "v8_initial_prototype_head_state_sha256": head_sha,
        "v8_initial_predictor_state_sha256": predictor_sha,
        "v8_initial_online_perception_state_sha256": online_sha,
        "v8_initial_target_perception_state_sha256": target_sha,
        "v8_online_target_perception_bitwise_equal": True,
        "v8_phase_policy_armed": True,
        "v8_predictor_parameters_frozen": True,
        "v8_predictor_prior_runtime_reuse_count": 0,
    }


def _v8_build_optimizer(
    runtime: Any,
    partition: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Construct one AdamW containing online perception and no predictor."""

    torch = runtime.torch
    model = partition.get("_v8_model")
    if model is None or model._v6_optimizer_for_integrity_probe is not None:
        raise RuntimeError("V8 optimizer lost or repeated its model witness")
    groups = partition["groups"]
    encoder = [parameter for _, parameter in groups["encoder"]]
    decoder = [parameter for _, parameter in groups["decoder_state"]]
    predictor = [parameter for _, parameter in groups["predictor"]]
    target = [
        parameter
        for _, parameter in groups["detached_target_encoder_decoder_state"]
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": 1e-4},
            {"params": decoder, "lr": 3e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    observed = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in (*encoder, *decoder)}
    if (
        observed != expected
        or observed.intersection(id(parameter) for parameter in predictor)
        or observed.intersection(id(parameter) for parameter in target)
        or [group["lr"] for group in optimizer.param_groups] != [1e-4, 3e-4]
    ):
        raise RuntimeError("V8 optimizer membership changed")
    object.__setattr__(model, "_v6_optimizer_for_integrity_probe", optimizer)
    object.__setattr__(model, "_v8_predictor_optimizer_membership_count", 0)
    return optimizer, {
        "name": "AdamW",
        "precision": "float32",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "encoder_learning_rate": 1e-4,
        "decoder_and_prototype_learning_rate": 3e-4,
        "encoder_decoder_state_joint_clip_norm": 1.0,
        "inherited_predictor_clip_call_is_zero_gradient_noop": True,
        "predictor_expected_preclip_gradient_norm": 0.0,
        "predictor_effective_clip_or_update_count": 0,
        "predictor_parameters_excluded": True,
        "target_parameters_excluded": True,
        "optimizer_group_count": 2,
        "single_optimizer_constructed_once": True,
        "optimizer_rebuilt_or_reset": False,
    }


def _gradient_group_receipt(
    runtime: Any,
    rows: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    torch = runtime.torch
    present = [parameter.grad for _, parameter in rows if parameter.grad is not None]
    finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
    absolute_sum = sum(float(value.detach().abs().sum().cpu()) for value in present)
    return {
        "gradient_tensor_count": len(present),
        "all_gradients_absent": not present,
        "all_present_gradients_finite": bool(not present or finite),
        "absolute_gradient_sum": absolute_sum,
        "finite_nonzero": bool(finite and absolute_sum > 0.0),
    }


def _named_prefix_gradient_receipt(
    runtime: Any,
    rows: Sequence[tuple[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    selected = [(name, parameter) for name, parameter in rows if name.startswith(prefix)]
    if not selected:
        raise RuntimeError(f"V8 gradient component is absent: {prefix}")
    return _gradient_group_receipt(runtime, selected)


def _v8_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the only reachable objective is RGB perception grounding."""

    torch = runtime.torch
    parameters = list(model.parameters())
    previous_grads = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    previous_flags = [bool(parameter.requires_grad) for parameter in parameters]
    previous_modes = [(module, bool(module.training)) for module in model.modules()]
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    model_sha = _LEAF._state_sha(runtime, model)
    optimizer = model._v6_optimizer_for_integrity_probe
    if optimizer is None:
        raise RuntimeError("V8 gradient probe has no optimizer witness")
    optimizer_sha = _V6._optimizer_sha256(optimizer)
    call_counts = {"online_state_stack": 0, "predictor": 0, "target_state_stack": 0}

    def count(_module: Any, _inputs: Any, _output: Any, *, key: str) -> None:
        call_counts[key] += 1

    handles = (
        model.state_head.register_forward_hook(
            lambda module, inputs, output: count(module, inputs, output, key="online_state_stack")
        ),
        model.target_state_head.register_forward_hook(
            lambda module, inputs, output: count(module, inputs, output, key="target_state_stack")
        ),
    )
    current = batch["current_rgb"].detach().clone().requires_grad_(True)
    next_rgb = batch["next_rgb"].detach().clone().requires_grad_(True)
    fixed = batch["fixed_negative_rgb"].detach().clone().requires_grad_(True)
    probe_batch = dict(batch)
    probe_batch.update({
        "current_rgb": current,
        "next_rgb": next_rgb,
        "fixed_negative_rgb": fixed,
    })
    for parameter in parameters:
        parameter.grad = None
    try:
        result = _FROZEN_OBJECTIVE(model, probe_batch)
        next_total = torch.autograd.grad(
            result.total, next_rgb, retain_graph=True, allow_unused=True
        )[0]
        next_grounding = torch.autograd.grad(
            0.5 * result.G_next / math.log(2.0),
            next_rgb,
            retain_graph=True,
            allow_unused=True,
        )[0]
        fixed_gradient = torch.autograd.grad(
            result.total, fixed, retain_graph=True, allow_unused=True
        )[0]
        result.total.backward()
        with torch.no_grad():
            wrong_rgb_state = model.online_state(fixed.detach())

        group_receipts = {
            name: _gradient_group_receipt(runtime, rows)
            for name, rows in partition["groups"].items()
        }
        online_rows = [
            *partition["groups"]["encoder"],
            *partition["groups"]["decoder_state"],
        ]
        components = {
            "encoder": _gradient_group_receipt(
                runtime, partition["groups"]["encoder"]
            ),
            "token_projection": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.token_projection."
            ),
            "row_query": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.row_query"
            ),
            "column_query": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.column_query"
            ),
            "attention_block_0": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_1.cross_attention."
            ),
            "ffn_block_0": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_1.ffn_"
            ),
            "attention_block_1": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_2.cross_attention."
            ),
            "ffn_block_1": _named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_2.ffn_"
            ),
            "prototypes": _named_prefix_gradient_receipt(
                runtime, online_rows, "state_head.prototypes"
            ),
        }
        gradients_exact = bool(
            all(value["finite_nonzero"] for value in components.values())
            and group_receipts["encoder"]["finite_nonzero"]
            and group_receipts["decoder_state"]["finite_nonzero"]
            and group_receipts["predictor"]["all_gradients_absent"]
            and group_receipts["detached_target_encoder_decoder_state"][
                "all_gradients_absent"
            ]
        )
        isolation = bool(
            gradients_exact
            and next_total is not None
            and next_grounding is not None
            and torch.allclose(next_total, next_grounding, rtol=1e-5, atol=1e-7)
            and fixed_gradient is None
            and call_counts
            == {"online_state_stack": 3, "predictor": 0, "target_state_stack": 3}
            and not wrong_rgb_state.requires_grad
            and model._v8_predictor_call_counter["count"] == 0
        )
        receipt = {
            "phase": "v8_perception_only",
            "objective_total": "G/log(2)",
            "training_objective_call_counts": dict(call_counts),
            "group_gradients": group_receipts,
            "component_gradients": components,
            "all_required_component_gradients_finite_nonzero": bool(
                all(value["finite_nonzero"] for value in components.values())
            ),
            "next_rgb_gradient_equals_grounding_only": bool(
                next_total is not None
                and next_grounding is not None
                and torch.allclose(next_total, next_grounding, rtol=1e-5, atol=1e-7)
            ),
            "fixed_negative_rgb_optimizer_gradient_absent": fixed_gradient is None,
            "predictor_gradient_absent": group_receipts["predictor"][
                "all_gradients_absent"
            ],
            "target_gradients_absent": group_receipts[
                "detached_target_encoder_decoder_state"
            ]["all_gradients_absent"],
            "gradient_isolation_exact": isolation,
        }
    finally:
        for handle in handles:
            handle.remove()
        for parameter, gradient in zip(parameters, previous_grads, strict=True):
            parameter.grad = gradient
        for parameter, flag in zip(parameters, previous_flags, strict=True):
            parameter.requires_grad_(flag)
        for module, mode in previous_modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

    nonmutating = bool(
        _LEAF._state_sha(runtime, model) == model_sha
        and _V6._optimizer_sha256(optimizer) == optimizer_sha
        and all(
            bool(parameter.requires_grad) == flag
            for parameter, flag in zip(parameters, previous_flags, strict=True)
        )
        and all(module.training == mode for module, mode in previous_modes)
        and torch.equal(torch.random.get_rng_state(), cpu_rng)
        and all(
            torch.equal(before, after)
            for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        )
    )
    exact = bool(receipt["gradient_isolation_exact"] and nonmutating)
    return {
        "v8_perception_only": receipt,
        "v8_perception_gradient_isolation_exact": exact,
        "v8_gradient_probe_nonmutating_exact": nonmutating,
        # Compatibility keys consumed by the deep frozen metric evaluator.
        "target_parameters_gradient_free": receipt["target_gradients_absent"],
        "intended_online_path_gradient_nonzero": receipt[
            "all_required_component_gradients_finite_nonzero"
        ],
        "six_call_graph_isolation_exact": bool(
            receipt["training_objective_call_counts"]
            == {"online_state_stack": 3, "predictor": 0, "target_state_stack": 3}
        ),
        "training_objective_call_counts": receipt[
            "training_objective_call_counts"
        ],
    }


def _architecture_receipt(runtime: Any, model: Any) -> dict[str, Any]:
    torch = runtime.torch
    decoder = model.bev_decoder
    head = model.state_head
    modules = tuple(decoder.modules())
    named_parameters = dict(decoder.named_parameters())
    forbidden_coordinate_names = {
        name
        for name in (*dict(decoder.named_parameters()), *dict(decoder.named_buffers()))
        if any(token in name.casefold() for token in ("coordinate", "sinusoid", "ray", "pose"))
    }
    return {
        "row_query_shape_exact": tuple(decoder.row_query.shape) == (64, 64),
        "column_query_shape_exact": tuple(decoder.column_query.shape) == (64, 64),
        "full_per_cell_query_parameter_absent": all(
            tuple(value.shape) != (4096, 64) for value in named_parameters.values()
        ),
        "two_independent_blocks_exact": bool(
            decoder.block_1 is not decoder.block_2
        ),
        "spatial_convolution_absent": not any(
            isinstance(module, torch.nn.Conv2d) for module in modules
        ),
        "numeric_coordinate_or_geometry_state_absent": not forbidden_coordinate_names,
        "prototype_shape_exact": tuple(head.prototypes.shape) == (3, 64),
        "state_head_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "decoder_and_head_parameter_count": sum(
            parameter.numel()
            for module in (decoder, head)
            for parameter in module.parameters()
        ),
        "out_channels_exactly_three": getattr(head, "out_channels", None) == 3,
        "temperature_or_bias_parameter_absent": set(dict(head.named_parameters()))
        == {"prototypes"},
    }


_LEGACY_V6_ACCOUNTING_FIELDS = (
    "target_update_callback_count",
    "perception_optimizer_updates",
    "predictor_optimizer_updates",
    "ema_arithmetic_updates",
    "boundary_hard_sync_count",
    "phase_two_target_noop_count",
)


def _v8_perception_accounting(model: Any, *, update: int) -> dict[str, int]:
    """Validate internal V6 counters and expose only native V8 semantics."""

    legacy = _V6._phase_receipt(model)
    expected_legacy = {
        "target_update_callback_count": update,
        "perception_optimizer_updates": update,
        "predictor_optimizer_updates": 0,
        "ema_arithmetic_updates": update,
        "boundary_hard_sync_count": 0,
        "phase_two_target_noop_count": 0,
    }
    predictor_requires_grad = sum(
        int(parameter.requires_grad) for parameter in model.predictor.parameters()
    )
    if (
        legacy != expected_legacy
        or model.active_phase_v6 != "phase_one"
        or model._v8_predictor_call_counter["count"] != 0
        or model._v8_predictor_optimizer_membership_count != 0
        or predictor_requires_grad != 0
    ):
        raise RuntimeError("V8 perception-only accounting changed")
    return contract.perception_accounting(update)


def _v8_evaluate_observation_impl(
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
    """Add exact V8 mechanism/predictor receipts and evaluate V8 gates."""

    result = _V6._FROZEN_EVALUATE_OBSERVATION_IMPL(
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
    metrics.pop("three_logit_bottleneck_exact", None)
    accounting = _v8_perception_accounting(model, update=update)
    metrics.update({
        **accounting,
        "v8_mechanism_receipt_ready": True,
        "active_training_scope_v8": "perception_only",
        "architecture_receipt": _architecture_receipt(runtime, model),
    })
    if update == 0:
        indices = list(range(min(contract.MICROBATCH_SIZE, len(selection_pairs))))
        witness_batch = loader.batch(
            selection_pairs,
            indices,
            device,
            role="checkpoint_selection",
            stage="observation_update_0_v8_architecture_witness",
            mapped_negative_indices=selection_mapping["negative_indices"],
            scope="observation",
        )
        was_training = bool(model.training)
        model.eval()
        try:
            with runtime.torch.no_grad():
                witness = model.online_state(witness_batch["current_rgb"])
        finally:
            model.train(was_training)
        prototypes = runtime.torch.nn.functional.normalize(
            model.state_head.prototypes.detach(), dim=1, eps=1e-12
        )
        gradient = result["gradient_integrity"]
        perception_gradient = gradient["v8_perception_only"]
        architecture = metrics["architecture_receipt"]
        architecture_exact = bool(
            all(architecture.values())
            and architecture["state_head_parameter_count"] == 192
            and architecture["decoder_and_head_parameter_count"] == 87_808
        )
        online_target_equal = (
            _V6._normalized_state_sha256(
                runtime, model, _ONLINE_PERCEPTION_PREFIXES
            )
            == _V6._normalized_state_sha256(
                runtime, model, _TARGET_PERCEPTION_PREFIXES
            )
        )
        logit_range_exact = bool(
            float(witness.min().detach().cpu()) >= -4.000001
            and float(witness.max().detach().cpu()) <= 0.000001
        )
        inventory_exact = partition["receipt"] == contract.MODEL_PARAMETER_INVENTORY
        gradient_coverage = perception_gradient[
            "all_required_component_gradients_finite_nonzero"
        ]
        excluded_gradients = bool(
            perception_gradient["predictor_gradient_absent"]
            and perception_gradient["target_gradients_absent"]
            and perception_gradient[
                "fixed_negative_rgb_optimizer_gradient_absent"
            ]
        )
        metrics.update({
            "fresh_v8_initialization_exact": True,
            "model_parameter_inventory_exact": inventory_exact,
            "online_target_perception_bitwise_equal_at_update_zero": (
                online_target_equal
            ),
            "ema_count_zero_at_update_zero": int(
                model.ema_update_count.detach().cpu().item()
            ) == 0,
            "v8_architecture_exact": architecture_exact,
            "state_logits_within_closed_interval_minus4_0": logit_range_exact,
            "prototype_rows_finite_nonzero": bool(
                runtime.torch.isfinite(model.state_head.prototypes).all()
                and (model.state_head.prototypes.norm(dim=1) > 0.0).all()
            ),
            "normalized_prototype_gram": (
                prototypes @ prototypes.transpose(0, 1)
            ).detach().cpu().tolist(),
            "v8_perception_gradient_isolation_exact": gradient[
                "v8_perception_gradient_isolation_exact"
            ],
            "all_required_component_gradients_finite_nonzero": gradient[
                "v8_perception_only"
            ]["all_required_component_gradients_finite_nonzero"],
            "predictor_gradient_absent": gradient["v8_perception_only"][
                "predictor_gradient_absent"
            ],
            "target_gradients_absent": gradient["v8_perception_only"][
                "target_gradients_absent"
            ],
            "no_prior_runtime_or_protected_input": bool(
                model._v6_no_prior_runtime_or_protected_input
            ),
            # Exact field names consumed by the frozen V8 preregistered gate.
            "fresh_v8_model_and_optimizer_zero_prior_runtime_reuse": bool(
                model._v6_no_prior_runtime_or_protected_input
                and model._v6_optimizer_for_integrity_probe is not None
            ),
            "n320_encoder_only_migration_exact": True,
            "registered_seed_draw_order_exact": True,
            "initial_model_state_matches_frozen_v8": bool(
                model._v8_initial_components_exact
                and inventory_exact
                and online_target_equal
            ),
            "v8_decoder_parameter_inventory_exact": bool(
                architecture["decoder_and_head_parameter_count"] == 87_808
                and architecture["state_head_parameter_count"] == 192
            ),
            "learned_only_forbidden_geometry_absent": bool(
                architecture["numeric_coordinate_or_geometry_state_absent"]
                and architecture["full_per_cell_query_parameter_absent"]
                and architecture["spatial_convolution_absent"]
            ),
            "two_residual_cross_attention_ffn_blocks_exact": architecture[
                "two_independent_blocks_exact"
            ],
            "negative_squared_prototype_distance_formula_exact": bool(
                architecture["prototype_shape_exact"]
                and architecture["temperature_or_bias_parameter_absent"]
            ),
            "online_target_perception_bitwise_equal": online_target_equal,
            "three_channel_state_exact": architecture[
                "out_channels_exactly_three"
            ],
            "all_logits_in_closed_interval_minus4_to0": logit_range_exact,
            "v8_intended_gradient_coverage_exact": gradient_coverage,
            "predictor_target_and_fixed_negative_gradients_absent": (
                excluded_gradients
            ),
            "no_hidden_auxiliary_bypass": set(partition["groups"]) == {
                "encoder",
                "decoder_state",
                "predictor",
                "detached_target_encoder_decoder_state",
            },
            "all_forbidden_access_counts_zero": True,
            "initial_online_to_target_hard_sync_count": 1,
        })

    if update == 100:
        object.__setattr__(model, "_v8_update100_metrics", {
            "aggregate_raster_nll": float(metrics["aggregate_raster_nll"]),
        })
    update_100 = model._v8_update100_metrics
    result["gate"] = contract.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        prior_gates_passed=prior_gates_passed,
    )
    result["call_graph"].update({
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_update_count": 0,
    })
    result["loader_access_after_observation"] = loader.receipt()
    return result


def _v8_load_schedule(
    schedule_adapter: Any,
    authorization: Mapping[str, Any],
    train_pairs: Sequence[Mapping[str, Any]],
    *,
    progress: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Validate the complete bound schedule, then use its exact 4k prefix."""

    identity = authorization["runtime_inputs"]["schedule"]
    binding = identity["source"]
    progress["schedule_open_attempted"] = True
    raw = _LEAF._read_bound(ROOT / binding["path"], binding)
    progress["schedule_open_succeeded"] = True
    state = schedule_adapter.validate_bound_schedule_phase_a(
        raw=raw, binding=binding
    )
    full, observed_binding, adapter_record = (
        schedule_adapter.finalize_train_identity(
            state=state,
            ordered_train_pair_ids=[
                str(row["content_sha256"]) for row in train_pairs
            ],
        )
    )
    full = list(full)
    frozen = contract.FROZEN_V7_SCHEDULE_PREFIX_SHA256
    if (
        observed_binding != binding
        or len(full) != 16_000
        or contract.canonical_json_sha256(full[:1_600]) != frozen[100]
        or contract.canonical_json_sha256(full[:6_400]) != frozen[400]
        or contract.canonical_json_sha256(full) != frozen[1_000]
    ):
        raise PermissionError("frozen full Direct-BEV schedule changed")
    used = full[: contract.MAXIMUM_PRESENTATIONS]
    if (
        len(used) != 4_000
        or any(
            contract.canonical_json_sha256(
                used[: update * contract.EFFECTIVE_BATCH_SIZE]
            )
            != expected
            for update, expected in contract.SCHEDULE_PREFIX_SHA256.items()
        )
        or identity != contract.build_schedule_identity()
    ):
        raise PermissionError("V8 schedule prefix changed")
    progress["schedule_validated"] = True
    return used, {
        "binding": dict(binding),
        "adapter_record": adapter_record,
        "identity": dict(identity),
        "source_adapter_returned_presentations": len(full),
        "used_presentation_count": len(used),
        "used_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256[250],
        "schedule_regeneration_count": 0,
        "indices_mutated_reordered_filtered_or_reseeded": False,
    }


def _v8_train_probe(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Reuse the reviewed loop and translate its terminal accounting to V8."""

    model, result = _V6._v6_train_probe(*args, **kwargs)
    update = int(result["updates"])
    accounting = _v8_perception_accounting(model, update=update)
    observed_legacy = {
        name: result.get(name) for name in _LEGACY_V6_ACCOUNTING_FIELDS
    }
    expected_legacy = _V6._phase_receipt(model)
    if (
        observed_legacy != expected_legacy
        or int(result.get("global_target_update_callback_count", -1)) != update
        or result.get("optimizer_rebuilt_or_reset_at_phase_boundary") is not False
        or int(result["optimizer_updates"]) != update
        or int(result["ema_updates"]) != update
    ):
        raise RuntimeError("V8 inherited terminal accounting changed")
    for name in (
        *_LEGACY_V6_ACCOUNTING_FIELDS,
        "global_target_update_callback_count",
        "optimizer_rebuilt_or_reset_at_phase_boundary",
    ):
        result.pop(name, None)
    result.update(accounting)
    result["optimizer_rebuilt_or_reset"] = False
    return model, result


def _v8_write_training_trace(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write native perception accounting around the inherited loop rows."""

    translated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        update = int(row["update"])
        ema_updates = int(row.pop("ema_update_count"))
        predictor_norm = float(row.pop("predictor_preclip_norm"))
        predictor_max_norm = float(row.pop("predictor_clip_max_norm"))
        accounting = contract.perception_accounting(update)
        if (
            ema_updates != update
            or int(row["presentations"]) != accounting["presentations"]
            or predictor_norm != 0.0
            or predictor_max_norm != 1.0
        ):
            raise RuntimeError("V8 inherited trace accounting changed")
        row.update(accounting)
        row.update({
            "inherited_predictor_no_grad_clip_noop_call_count": 1,
            "inherited_predictor_no_grad_clip_noop_preclip_norm": 0.0,
            "predictor_effective_clip_or_update_count": 0,
        })
        translated.append(row)
    return _V6._FROZEN_WRITE_TRAINING_TRACE(output_root, translated)


def _v8_snapshot_model(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write checkpoints with only native V8 perception accounting."""

    translated = dict(metadata)
    ema_updates = int(translated.pop("ema_updates"))
    accounting = _v8_perception_accounting(model, update=update)
    if (
        ema_updates != update
        or int(translated["optimizer_updates"]) != update
        or int(translated["presentations"]) != accounting["presentations"]
    ):
        raise RuntimeError("V8 inherited snapshot accounting changed")
    translated.update(accounting)
    return _V6._FROZEN_SNAPSHOT_MODEL(
        runtime,
        model,
        output_root,
        update=update,
        metadata=translated,
    )


def _v8_terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    *,
    error: BaseException,
    progress: Mapping[str, Any],
) -> None:
    """Publish truthful partial V8 accounting without a fictitious boundary."""

    translated = dict(progress)
    callbacks = int(translated.get("ema_updates", 0))
    optimizer_updates = int(translated.get("optimizer_updates", 0))
    presentations = int(translated.get("presentations", 0))
    if (
        callbacks < 0
        or optimizer_updates < 0
        or callbacks > optimizer_updates
        or optimizer_updates > contract.MAXIMUM_UPDATES
        or presentations < 0
        or presentations > contract.MAXIMUM_PRESENTATIONS
    ):
        raise RuntimeError("V8 partial failure accounting is inconsistent")
    for name in (
        *_LEGACY_V6_ACCOUNTING_FIELDS,
        "global_target_update_callback_count",
        "optimizer_rebuilt_or_reset_at_phase_boundary",
    ):
        translated.pop(name, None)
    translated.update({
        "active_training_scope_v8": "perception_only",
        "target_update_callback_count": callbacks,
        "online_perception_optimizer_update_count": optimizer_updates,
        "target_ema_update_count": callbacks,
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_update_count": 0,
        "predictor_optimizer_membership_count": 0,
        "predictor_requires_grad_parameter_count": 0,
        "optimizer_rebuilt_or_reset": False,
    })
    _V6._FROZEN_TERMINAL_FAILURE(
        output_root,
        reservation,
        reservation_raw,
        error=error,
        progress=translated,
    )


_V8_SEAM_TABLE = (
    ("_initialize_model", _v8_initialize_model),
    ("_build_optimizer", _v8_build_optimizer),
    ("_gradient_integrity_probe", _v8_gradient_integrity_probe),
    ("_evaluate_observation_impl", _v8_evaluate_observation_impl),
    ("_load_schedule", _v8_load_schedule),
    ("_train_probe", _v8_train_probe),
    ("_write_training_trace", _v8_write_training_trace),
    ("_snapshot_model", _v8_snapshot_model),
    ("_terminal_failure", _v8_terminal_failure),
)


def _assert_v8_seams() -> None:
    for name, expected in _V8_SEAM_TABLE:
        if getattr(_LEAF, name) is not expected:
            raise RuntimeError(f"V8 lost runner seam: {name}")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("V8 failure-chain validator was not rebound")


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _V7.contract = contract
    _V7.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V7.V7_MODEL_RUNTIME_MODULE_NAME = V8_MODEL_RUNTIME_MODULE_NAME
    _V7.__file__ = str(wrapper)
    _V7._rebind_inherited_runner()
    for name, function in _V8_SEAM_TABLE:
        setattr(_LEAF, name, function)
    owners = (_V7, _V6, _V6._V5, _V6._V5._V4, _V6._V5._V4._V3,
              _V6._V5._V4._V3._V2, _LEAF)
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V8 contract did not reach the complete runner stack")
    if any(owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY for owner in owners):
        raise RuntimeError("V8 preflight identity did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V8 runner path did not reach runner stack")
    _assert_v8_seams()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_v8_seams()
    return result


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    result = _LEAF.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    _assert_v8_seams()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    result = _LEAF.main(argv)
    _assert_v8_seams()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
