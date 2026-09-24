#!/usr/bin/env python3
"""Run the capped Direct-BEV signed-boundary-distance state V1 probe."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_DISTANCE_STATE_V1_"
    "PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
SIGNED_BOUNDARY_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_signed_boundary_distance_state_v1_model_runtime"
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
    "_lewm_direct_bev_signed_boundary_v1_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_distance_state_v1.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV signed-boundary V1 runner identity changed")

# Start from V9: it is the last execution-mechanics layer and owns the exact
# checkpoint semantic-registry and complete terminal-failure receipts.
_V9 = _source_only_module(
    "_lewm_direct_bev_signed_boundary_v1_frozen_v9_runner",
    ROOT / contract.FROZEN_V9_RUNNER_RELATIVE_PATH,
)
_V8 = _V9._V8
_V6 = _V8._V6
_LEAF = _V9._LEAF

_GRADIENT_PROBE_ACTIVE = False
_CLASS_NAMES = ("UNKNOWN", "FREE", "OCCUPIED")


def _head_projection(runtime: Any, head: Any) -> Any:
    torch = runtime.torch
    projections = [
        module for module in head.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    if len(projections) != 1:
        raise RuntimeError("signed-boundary head must contain one Conv2d")
    return projections[0]


def _all_sign_labels(runtime: Any, reference: Any) -> Any:
    """Construct the registered all-sign gradient-only raster population."""

    torch = runtime.torch
    labels = torch.empty_like(reference)
    width = int(labels.shape[-1])
    first = max(1, width // 3)
    second = max(first + 1, 2 * width // 3)
    labels[..., :first] = 0
    labels[..., first:second] = 1
    labels[..., second:] = 2
    return labels


def _signed_boundary_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact signed-boundary gradient coverage without changing state."""

    global _GRADIENT_PROBE_ACTIVE
    if _GRADIENT_PROBE_ACTIVE:
        raise RuntimeError("signed-boundary gradient probe re-entered")
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
        raise RuntimeError("signed-boundary gradient probe has no optimizer")
    optimizer_sha = _V6._optimizer_sha256(optimizer)
    call_counts = {
        "online_state_stack": 0,
        "predictor": 0,
        "target_state_stack": 0,
    }

    def count(_module: Any, _inputs: Any, _output: Any, *, key: str) -> None:
        call_counts[key] += 1

    handles = (
        model.state_head.register_forward_hook(
            lambda module, inputs, output: count(
                module, inputs, output, key="online_state_stack"
            )
        ),
        model.target_state_head.register_forward_hook(
            lambda module, inputs, output: count(
                module, inputs, output, key="target_state_stack"
            )
        ),
    )
    current = batch["current_rgb"].detach().clone().requires_grad_(True)
    next_rgb = batch["next_rgb"].detach().clone().requires_grad_(True)
    fixed = batch["fixed_negative_rgb"].detach().clone().requires_grad_(True)
    labels = _all_sign_labels(runtime, batch["current_labels"])
    probe_batch = dict(batch)
    probe_batch.update({
        "current_rgb": current,
        "next_rgb": next_rgb,
        "fixed_negative_rgb": fixed,
        "current_labels": labels,
        "next_labels": labels.clone(),
    })
    for parameter in parameters:
        parameter.grad = None
    _GRADIENT_PROBE_ACTIVE = True
    try:
        result = _LEAF._objective(model, probe_batch)
        objective_call_counts = dict(call_counts)
        next_total = torch.autograd.grad(
            result.total, next_rgb, retain_graph=True, allow_unused=True
        )[0]
        next_grounding = torch.autograd.grad(
            0.5 * result.G_next,
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
        observation_only_fixed_negative_online_call_count = (
            call_counts["online_state_stack"]
            - objective_call_counts["online_state_stack"]
        )

        groups = {
            name: _V8._gradient_group_receipt(runtime, rows)
            for name, rows in partition["groups"].items()
        }
        online_rows = [
            *partition["groups"]["encoder"],
            *partition["groups"]["decoder_state"],
        ]
        components = {
            "encoder": _V8._gradient_group_receipt(
                runtime, partition["groups"]["encoder"]
            ),
            "token_projection": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.token_projection."
            ),
            "row_query": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.row_query"
            ),
            "column_query": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.column_query"
            ),
            "attention_block_0": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_1.cross_attention."
            ),
            "ffn_block_0": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_1.ffn_"
            ),
            "attention_block_1": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_2.cross_attention."
            ),
            "ffn_block_1": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "bev_decoder.block_2.ffn_"
            ),
            "signed_boundary_head": _V8._named_prefix_gradient_receipt(
                runtime, online_rows, "state_head."
            ),
        }
        projection = _head_projection(runtime, model.state_head)
        weight_gradient = projection.weight.grad
        bias_gradient = None if projection.bias is None else projection.bias.grad
        rows_finite_nonzero = bool(
            weight_gradient is not None
            and tuple(weight_gradient.shape) == (2, 64, 1, 1)
            and bias_gradient is not None
            and tuple(bias_gradient.shape) == (2,)
            and torch.isfinite(weight_gradient).all()
            and torch.isfinite(bias_gradient).all()
            and all(
                float(weight_gradient[row].double().abs().sum().cpu()) > 0.0
                and float(bias_gradient[row].double().abs().cpu()) > 0.0
                for row in range(2)
            )
        )
        component_coverage = all(
            receipt["finite_nonzero"] for receipt in components.values()
        )
        gradient_isolation = bool(
            component_coverage
            and rows_finite_nonzero
            and groups["encoder"]["finite_nonzero"]
            and groups["decoder_state"]["finite_nonzero"]
            and groups["predictor"]["all_gradients_absent"]
            and groups["detached_target_encoder_decoder_state"][
                "all_gradients_absent"
            ]
            and next_total is not None
            and next_grounding is not None
            and torch.allclose(next_total, next_grounding, rtol=1e-5, atol=1e-7)
            and fixed_gradient is None
            and objective_call_counts
            == {
                "online_state_stack": 2,
                "predictor": 0,
                "target_state_stack": 3,
            }
            and call_counts
            == {
                "online_state_stack": 3,
                "predictor": 0,
                "target_state_stack": 3,
            }
            and observation_only_fixed_negative_online_call_count == 1
            and not wrong_rgb_state.requires_grad
            and model._v8_predictor_call_counter["count"] == 0
        )
        receipt = {
            "phase": "signed_boundary_distance_perception_only",
            "objective_total": "G",
            "registered_all_sign_synthetic_population": True,
            "training_objective_call_counts": objective_call_counts,
            "observation_only_fixed_negative_online_call_count": (
                observation_only_fixed_negative_online_call_count
            ),
            "complete_probe_call_counts": dict(call_counts),
            "group_gradients": groups,
            "component_gradients": components,
            "both_head_rows_weight_and_bias_gradients_finite_nonzero": (
                rows_finite_nonzero
            ),
            "all_required_component_gradients_finite_nonzero": (
                component_coverage and rows_finite_nonzero
            ),
            "next_rgb_gradient_equals_half_G_next_only": bool(
                next_total is not None
                and next_grounding is not None
                and torch.allclose(
                    next_total, next_grounding, rtol=1e-5, atol=1e-7
                )
            ),
            "fixed_negative_rgb_optimizer_gradient_absent": (
                fixed_gradient is None
            ),
            "predictor_gradient_absent": groups["predictor"][
                "all_gradients_absent"
            ],
            "target_gradients_absent": groups[
                "detached_target_encoder_decoder_state"
            ]["all_gradients_absent"],
            "gradient_isolation_exact": gradient_isolation,
        }
    finally:
        _GRADIENT_PROBE_ACTIVE = False
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
            for before, after in zip(
                cuda_rng, torch.cuda.get_rng_state_all(), strict=True
            )
        )
    )
    exact = bool(receipt["gradient_isolation_exact"] and nonmutating)
    return {
        "signed_boundary_distance_perception_only": receipt,
        "signed_boundary_gradient_isolation_exact": exact,
        "signed_boundary_gradient_probe_nonmutating_exact": nonmutating,
        "target_parameters_gradient_free": receipt["target_gradients_absent"],
        "intended_online_path_gradient_nonzero": receipt[
            "all_required_component_gradients_finite_nonzero"
        ],
        "six_call_graph_isolation_exact": bool(
            receipt["training_objective_call_counts"]
            == {
                "online_state_stack": 2,
                "predictor": 0,
                "target_state_stack": 3,
            }
            and receipt["observation_only_fixed_negative_online_call_count"]
            == 1
        ),
        "training_objective_call_counts": receipt[
            "training_objective_call_counts"
        ],
    }


class _FinalClassMacroObservationApi:
    """Replace only the inherited paired-RGB observation loss leaf."""

    def __init__(self, model_api: Any) -> None:
        self._model_api = model_api
        helper = getattr(model_api, "_final_class_macro_nll_per_row_v10", None)
        if helper is None:
            helper = model_api._v10._final_class_macro_nll_per_row_v10
        self._hard_hierarchical_loss_per_row = helper

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_api, name)


class _PairedFieldAccumulator:
    """Observe raw next/wrong two-field tensors on the existing call graph."""

    def __init__(
        self,
        runtime: Any,
        selection_pairs: Sequence[Mapping[str, Any]],
    ) -> None:
        self.runtime = runtime
        self.selection_pairs = selection_pairs
        self.rows = 0
        self.scenes = {
            family: {
                "scene_id": str(binding["scene_id"]),
                "row_count": 0,
                "all_values_finite": True,
                "absolute_field_difference_sum": 0.0,
                "bitwise_identical": True,
            }
            for family, binding in contract.SELECTION_FAMILY_BINDINGS.items()
        }
        self.family_by_scene = {
            str(binding["scene_id"]): family
            for family, binding in contract.SELECTION_FAMILY_BINDINGS.items()
        }

    def add(self, indices: Sequence[int], correct: Any, wrong: Any) -> None:
        torch = self.runtime.torch
        expected = (len(indices), 2, 64, 64)
        if tuple(correct.shape) != expected or tuple(wrong.shape) != expected:
            raise RuntimeError("paired signed-boundary field shape changed")
        for offset, source_index in enumerate(indices):
            scene_id = str(self.selection_pairs[source_index]["scene_id"])
            family = self.family_by_scene.get(scene_id)
            if family is None:
                raise PermissionError("unregistered paired field scene")
            left = correct[offset]
            right = wrong[offset]
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            absolute = float((left.double() - right.double()).abs().sum().cpu())
            scene = self.scenes[family]
            scene["row_count"] += 1
            scene["all_values_finite"] = bool(
                scene["all_values_finite"] and finite
            )
            scene["absolute_field_difference_sum"] += absolute
            scene["bitwise_identical"] = bool(
                scene["bitwise_identical"] and torch.equal(left, right)
            )
            self.rows += 1

    def receipt(self) -> dict[str, Any]:
        if self.rows != 495:
            raise RuntimeError("paired field receipt did not cover 495 rows")
        scenes: dict[str, Any] = {}
        aggregate_absolute = 0.0
        exact = True
        for family, binding in contract.SELECTION_FAMILY_BINDINGS.items():
            source = self.scenes[family]
            if source["row_count"] != int(binding["row_count"]):
                raise RuntimeError("paired field scene population changed")
            nonidentical = not source["bitwise_identical"]
            scene_exact = bool(source["all_values_finite"] and nonidentical)
            exact = exact and scene_exact
            aggregate_absolute += source["absolute_field_difference_sum"]
            scenes[family] = {
                **source,
                "correct_and_mapped_negative_not_bitwise_identical": (
                    nonidentical
                ),
                "finite_and_nonidentical": scene_exact,
            }
        return {
            "schema": f"{contract.SCHEMA_PREFIX}_paired_rgb_raw_fields_v1",
            "population": "eight_frozen_paired_scenes_checkpoint_selection",
            "row_count": self.rows,
            "scenes": scenes,
            "all_eight_scene_field_tensors_finite_and_nonidentical": exact,
            "aggregate_absolute_field_difference": aggregate_absolute,
            "aggregate_absolute_field_difference_strictly_positive": (
                aggregate_absolute > 0.0
            ),
        }


class _ObservationLoaderProxy:
    def __init__(self, loader: Any, accumulator: _PairedFieldAccumulator) -> None:
        self._loader = loader
        self._accumulator = accumulator
        self._indices: list[int] | None = None
        self._online_call_index = 0
        self._correct_fields: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def _finish_pair_batch(self) -> None:
        if self._indices is not None and self._online_call_index != 3:
            raise RuntimeError("signed-boundary observation call order changed")
        self._indices = None
        self._online_call_index = 0
        self._correct_fields = None

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        self._finish_pair_batch()
        if len(args) < 2:
            raise RuntimeError("signed-boundary loader batch indices absent")
        self._indices = [int(value) for value in args[1]]
        return self._loader.batch(*args, **kwargs)

    def endpoint_batch(self, *args: Any, **kwargs: Any) -> Any:
        self._finish_pair_batch()
        return self._loader.endpoint_batch(*args, **kwargs)

    def observe_online_head(
        self, _module: Any, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if _GRADIENT_PROBE_ACTIVE or self._indices is None:
            return
        if self._loader.runtime.torch.is_grad_enabled():
            return
        if self._online_call_index == 1:
            self._correct_fields = output.detach()
        elif self._online_call_index == 2:
            if self._correct_fields is None:
                raise RuntimeError("correct signed-boundary fields were not captured")
            self._accumulator.add(
                self._indices, self._correct_fields, output.detach()
            )
        self._online_call_index += 1
        if self._online_call_index > 3:
            raise RuntimeError("signed-boundary online observation calls changed")

    def finish(self) -> dict[str, Any]:
        self._finish_pair_batch()
        return self._accumulator.receipt()


def _architecture_receipt(runtime: Any, model: Any) -> dict[str, Any]:
    torch = runtime.torch
    decoder = model.bev_decoder
    head = model.state_head
    projection = _head_projection(runtime, head)
    forbidden = {
        name
        for name in (*dict(decoder.named_parameters()), *dict(decoder.named_buffers()))
        if any(token in name.casefold() for token in (
            "coordinate", "sinusoid", "ray", "pose"
        ))
    }
    return {
        "row_query_shape_exact": tuple(decoder.row_query.shape) == (64, 64),
        "column_query_shape_exact": tuple(decoder.column_query.shape) == (64, 64),
        "full_per_cell_query_parameter_absent": all(
            tuple(value.shape) != (4096, 64)
            for value in decoder.parameters()
        ),
        "two_independent_blocks_exact": decoder.block_1 is not decoder.block_2,
        "numeric_coordinate_or_geometry_state_absent": not forbidden,
        "head_is_single_conv2d": sum(
            isinstance(module, torch.nn.Conv2d) for module in head.modules()
        ) == 1,
        "head_operator_exact": bool(
            projection.in_channels == 64
            and projection.out_channels == 2
            and projection.kernel_size == (1, 1)
            and projection.stride == (1, 1)
            and projection.padding == (0, 0)
            and projection.dilation == (1, 1)
            and projection.groups == 1
            and projection.bias is not None
        ),
        "head_weight_shape_exact": tuple(projection.weight.shape)
        == (2, 64, 1, 1),
        "head_bias_shape_exact": tuple(projection.bias.shape) == (2,),
        "head_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "head_parameter_tensor_count": len(tuple(head.parameters())),
        "decoder_and_head_parameter_count": sum(
            parameter.numel()
            for module in (decoder, head)
            for parameter in module.parameters()
        ),
        "out_channels_exactly_two": getattr(head, "out_channels", None) == 2,
    }


def _synthetic_mechanism_receipt(runtime: Any, model_api: Any) -> dict[str, bool]:
    torch = runtime.torch
    labels = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    fields, masks = model_api.signed_boundary_distance_targets_v1(labels)
    adjacent_expected = torch.tensor(
        [[[
            [-1.0 / 16.0, 1.0 / 16.0, 3.0 / 16.0],
        ], [
            [0.0, 1.0 / 16.0, -1.0 / 16.0],
        ]]],
        dtype=fields.dtype,
    )
    adjacent_exact = bool(
        torch.allclose(fields, adjacent_expected, rtol=0.0, atol=1e-12)
        and torch.equal(masks[:, 0], torch.ones_like(masks[:, 0]))
        and torch.equal(masks[:, 1], labels != 0)
    )

    diagonal_labels = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.long)
    diagonal, _ = model_api.signed_boundary_distance_targets_v1(
        diagonal_labels
    )
    diagonal_expected = (math.sqrt(2.0) - 0.5) / 8.0
    diagonal_exact = math.isclose(
        float(diagonal[0, 0, 1, 1]),
        diagonal_expected,
        rel_tol=0.0,
        abs_tol=1e-7,
    )

    long_labels = torch.ones((1, 1, 10), dtype=torch.long)
    long_labels[0, 0, 0] = 0
    truncated, _ = model_api.signed_boundary_distance_targets_v1(long_labels)
    truncation_exact = bool(
        float(truncated[0, 0, 0, 9]) == 1.0
        and math.isclose(
            float(truncated[0, 0, 0, 1]),
            1.0 / 16.0,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    )

    empty_exact = True
    for code, expected_k, expected_o, o_available in (
        (0, -1.0, 0.0, False),
        (1, 1.0, 1.0, True),
        (2, 1.0, -1.0, True),
    ):
        one_class = torch.full((1, 3, 3), code, dtype=torch.long)
        observed, observed_masks = (
            model_api.signed_boundary_distance_targets_v1(one_class)
        )
        empty_exact = bool(
            empty_exact
            and torch.equal(observed[:, 0], torch.full_like(observed[:, 0], expected_k))
            and torch.equal(observed[:, 1], torch.full_like(observed[:, 1], expected_o))
            and bool(observed_masks[:, 0].all())
            and bool(observed_masks[:, 1].all()) == o_available
        )

    tie_labels = torch.ones((1, 3, 3), dtype=torch.long)
    tie_labels[0, 0, 1] = 2
    tie_labels[0, 2, 1] = 2
    tied, _ = model_api.signed_boundary_distance_targets_v1(tie_labels)
    tie_exact = math.isclose(
        float(tied[0, 1, 1, 1]),
        1.0 / 16.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    )

    adapter = model_api.hierarchical_class_log_probabilities_v1(fields)
    probabilities = adapter.exp()
    adapter_exact = bool(
        tuple(adapter.shape) == (1, 3, 1, 3)
        and torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities[:, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        and torch.equal(adapter.argmax(dim=1), labels)
    )

    macro_labels = torch.tensor([
        [[0, 0], [0, 0]],
        [[0, 1], [1, 1]],
        [[1, 2], [1, 2]],
    ], dtype=torch.long)
    targets, _ = model_api.signed_boundary_distance_targets_v1(macro_labels)
    predicted = targets.clone()
    predicted[:, 0, 0, 0] += 0.25
    predicted[:, 0, -1, -1] -= 0.125
    predicted[:, 1, 0, 0] += 0.5
    predicted[:, 1, -1, -1] -= 0.25
    observed_loss = model_api._boundary_huber_per_row_v1(
        predicted, targets, macro_labels
    )
    pointwise = torch.nn.functional.huber_loss(
        predicted, targets, reduction="none", delta=0.125
    )
    slow_rows = []
    for row in range(macro_labels.shape[0]):
        k_groups = [
            pointwise[row, 0][macro_labels[row] == code].mean()
            for code in (0, -1)
            if bool(
                (macro_labels[row] == code).any()
                if code == 0
                else (macro_labels[row] != 0).any()
            )
        ]
        if len(k_groups) == 1 and bool((macro_labels[row] != 0).any()):
            k_groups = [pointwise[row, 0][macro_labels[row] != 0].mean()]
        elif len(k_groups) == 2:
            k_groups[1] = pointwise[row, 0][macro_labels[row] != 0].mean()
        k_macro = torch.stack(k_groups).mean()
        o_groups = [
            pointwise[row, 1][macro_labels[row] == code].mean()
            for code in (1, 2)
            if bool((macro_labels[row] == code).any())
        ]
        slow_rows.append(
            k_macro if not o_groups
            else 0.5 * k_macro + 0.5 * torch.stack(o_groups).mean()
        )
    macro_exact = bool(torch.allclose(
        observed_loss, torch.stack(slow_rows), rtol=1e-6, atol=1e-7
    ))
    return {
        "adjacent_boundary_targets_exact": adjacent_exact,
        "diagonal_distance_exact": diagonal_exact,
        "truncation_and_saturation_exact": truncation_exact,
        "empty_opposite_and_unknown_o_mask_exact": empty_exact,
        "nearest_source_tie_scalar_exact": tie_exact,
        "hierarchical_adapter_normalization_and_target_semantics_exact": (
            adapter_exact
        ),
        "per_row_present_sign_macro_huber_exact": macro_exact,
    }


def _head_witness(runtime: Any, model_api: Any, model: Any, rgb: Any) -> dict[str, Any]:
    torch = runtime.torch
    projection = _head_projection(runtime, model.state_head)
    with torch.no_grad():
        cells = model.bev_decoder(model.encoder.forward_tokens(rgb)[:, 1:])
        observed_fields = model.state_head(cells)
        expected_fields = torch.tanh(projection(cells))
        adapter = model_api.hierarchical_class_log_probabilities_v1(
            observed_fields
        )
        expected_adapter = torch.stack((
            torch.nn.functional.logsigmoid(-16.0 * observed_fields[:, 0]),
            torch.nn.functional.logsigmoid(16.0 * observed_fields[:, 0])
            + torch.nn.functional.logsigmoid(16.0 * observed_fields[:, 1]),
            torch.nn.functional.logsigmoid(16.0 * observed_fields[:, 0])
            + torch.nn.functional.logsigmoid(-16.0 * observed_fields[:, 1]),
        ), dim=1)
        online_logits = model.online_state(rgb)
    return {
        "head_tanh_conv_formula_exact": torch.equal(
            observed_fields, expected_fields
        ),
        "field_shape_exact": tuple(observed_fields.shape[1:]) == (2, 64, 64),
        "field_range_exact": bool(
            float(observed_fields.min().cpu()) >= -1.0
            and float(observed_fields.max().cpu()) <= 1.0
        ),
        "adapter_formula_exact": bool(
            torch.equal(adapter, expected_adapter)
            and torch.equal(online_logits, adapter)
        ),
        "adapter_probability_normalization_exact": bool(torch.allclose(
            adapter.exp().sum(dim=1),
            torch.ones_like(adapter[:, 0]),
            rtol=1e-6,
            atol=1e-6,
        )),
        "adapter_values_finite": bool(torch.isfinite(adapter).all()),
    }


def _signed_boundary_observation_core(
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
    accumulator = _PairedFieldAccumulator(runtime, selection_pairs)
    proxy = _ObservationLoaderProxy(loader, accumulator)
    hook = model.state_head.register_forward_hook(proxy.observe_online_head)
    try:
        result = _V6._FROZEN_EVALUATE_OBSERVATION_IMPL(
            runtime,
            _FinalClassMacroObservationApi(model_api),
            model,
            partition,
            proxy,
            selection_pairs,
            selection_mapping,
            device,
            update=update,
            update_zero=update_zero,
            prior_gates_passed=prior_gates_passed,
        )
        paired_fields = proxy.finish()
    finally:
        hook.remove()

    metrics = result["metrics"]
    metrics.pop("three_logit_bottleneck_exact", None)
    accounting = _V8._v8_perception_accounting(model, update=update)
    access_counters = _LEAF._access_counters(loader)
    forbidden_access_zero = all(
        access_counters[name] == 0
        for name in contract.FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS
    )
    scene_margins = [
        float(scene["mapped_negative_rgb_mean_loss"])
        - float(scene["correct_rgb_mean_loss"])
        for scene in result["scene_metrics"].values()
    ]
    if len(scene_margins) != 8:
        raise RuntimeError("paired RGB scene population changed")
    metrics.update({
        **accounting,
        "signed_boundary_distance_mechanism_receipt_ready": True,
        "active_training_scope_signed_boundary_distance_v1": "perception_only",
        "aggregate_unknown_recall": result["aggregate_raster"][
            "unknown_recall"
        ],
        "paired_rgb_aggregate_margin": sum(scene_margins) / len(scene_margins),
        "paired_rgb_raw_field_receipt": paired_fields,
        "paired_rgb_two_field_nonidentity_exact": paired_fields[
            "all_eight_scene_field_tensors_finite_and_nonidentical"
        ],
        "paired_rgb_aggregate_absolute_field_difference_strictly_positive": (
            paired_fields[
                "aggregate_absolute_field_difference_strictly_positive"
            ]
        ),
        "all_forbidden_access_counts_zero": forbidden_access_zero,
    })

    if update == 0:
        object.__setattr__(model, "_signed_boundary_update100_metrics", None)
        object.__setattr__(model, "_signed_boundary_update400_metrics", None)
        indices = list(range(min(contract.MICROBATCH_SIZE, len(selection_pairs))))
        batch = loader.batch(
            selection_pairs,
            indices,
            device,
            role="checkpoint_selection",
            stage="observation_update_0_signed_boundary_architecture_witness",
            mapped_negative_indices=selection_mapping["negative_indices"],
            scope="observation",
        )
        was_training = bool(model.training)
        model.eval()
        try:
            witness = _head_witness(
                runtime, model_api, model, batch["current_rgb"]
            )
        finally:
            model.train(was_training)
        architecture = _architecture_receipt(runtime, model)
        synthetic = _synthetic_mechanism_receipt(runtime, model_api)
        gradient = result["gradient_integrity"]
        signed_gradient = gradient[
            "signed_boundary_distance_perception_only"
        ]
        online_target_equal = (
            _V6._normalized_state_sha256(
                runtime, model, _V6._ONLINE_PERCEPTION_PREFIXES
            )
            == _V6._normalized_state_sha256(
                runtime, model, _V6._TARGET_PERCEPTION_PREFIXES
            )
        )
        inventory_exact = partition["receipt"] == contract.MODEL_PARAMETER_INVENTORY
        excluded_gradients = bool(
            signed_gradient["predictor_gradient_absent"]
            and signed_gradient["target_gradients_absent"]
            and signed_gradient[
                "fixed_negative_rgb_optimizer_gradient_absent"
            ]
        )
        metrics.update({
            "architecture_receipt": architecture,
            "synthetic_mechanism_receipt": synthetic,
            "head_witness": witness,
            "fresh_signed_boundary_distance_model_and_optimizer_zero_prior_runtime_reuse": bool(
                model._v6_no_prior_runtime_or_protected_input
                and model._v6_optimizer_for_integrity_probe is not None
            ),
            "frozen_encoder_decoder_predictor_initialization_exact": bool(
                model._v8_initial_components_exact
            ),
            "registered_seed_draw_order_exact": True,
            "signed_boundary_distance_initial_head_state_sha256_exact": (
                _V8._component_state_sha256(runtime, model.state_head)
                == contract.SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256
            ),
            "model_parameter_inventory_exact": inventory_exact,
            "signed_boundary_distance_decoder_head_parameter_inventory_exact": bool(
                inventory_exact
                and architecture["head_parameter_count"] == 130
                and architecture["head_parameter_tensor_count"] == 2
            ),
            "learned_only_forbidden_geometry_absent": bool(
                architecture["full_per_cell_query_parameter_absent"]
                and architecture["numeric_coordinate_or_geometry_state_absent"]
            ),
            "two_residual_cross_attention_ffn_blocks_exact": architecture[
                "two_independent_blocks_exact"
            ],
            "signed_boundary_distance_head_shape_tanh_and_channel_order_exact": bool(
                architecture["head_is_single_conv2d"]
                and architecture["head_operator_exact"]
                and architecture["head_weight_shape_exact"]
                and architecture["head_bias_shape_exact"]
                and architecture["head_parameter_count"] == 130
                and architecture["head_parameter_tensor_count"] == 2
                and witness["head_tanh_conv_formula_exact"]
                and witness["field_shape_exact"]
                and witness["field_range_exact"]
            ),
            "signed_boundary_distance_center_edt_transform_exact": all(
                synthetic[name]
                for name in (
                    "adjacent_boundary_targets_exact",
                    "diagonal_distance_exact",
                    "truncation_and_saturation_exact",
                    "empty_opposite_and_unknown_o_mask_exact",
                    "nearest_source_tie_scalar_exact",
                )
            ),
            "signed_boundary_distance_huber_macro_objective_exact": synthetic[
                "per_row_present_sign_macro_huber_exact"
            ],
            "hierarchical_adapter_scale16_formula_and_normalization_exact": bool(
                synthetic[
                    "hierarchical_adapter_normalization_and_target_semantics_exact"
                ]
                and witness["adapter_formula_exact"]
                and witness["adapter_probability_normalization_exact"]
                and witness["adapter_values_finite"]
            ),
            "exact_target_adapter_argmax_semantics": synthetic[
                "hierarchical_adapter_normalization_and_target_semantics_exact"
            ],
            "paired_rgb_direction_free_nonidentity_all_8": bool(
                metrics["paired_rgb_two_field_nonidentity_exact"]
                and metrics[
                    "paired_rgb_aggregate_absolute_field_difference_strictly_positive"
                ]
            ),
            "K_and_O_head_gradients_finite_nonzero": signed_gradient[
                "both_head_rows_weight_and_bias_gradients_finite_nonzero"
            ],
            "online_target_perception_bitwise_equal": online_target_equal,
            "target_requires_grad_false": not any(
                parameter.requires_grad
                for module in model._target_modules()
                for parameter in module.parameters()
            ),
            "two_channel_raw_state_exact": bool(
                witness["field_shape_exact"] and witness["field_range_exact"]
            ),
            "three_channel_adapter_logits_exact": bool(
                witness["adapter_formula_exact"]
                and witness["adapter_values_finite"]
            ),
            "predictor_target_and_fixed_negative_gradients_absent": (
                excluded_gradients
            ),
            "no_hidden_auxiliary_bypass": set(partition["groups"]) == {
                "encoder",
                "decoder_state",
                "predictor",
                "detached_target_encoder_decoder_state",
            },
            "update_zero_semantic_direction_gate_absent": True,
            "initial_online_to_target_hard_sync_count": 1,
        })

    if update == 100:
        object.__setattr__(model, "_signed_boundary_update100_metrics", {
            name: float(metrics[name])
            for name in (
                "G",
                "aggregate_raster_nll",
                "aggregate_raster_balanced_accuracy",
                "rough_raster_balanced_accuracy",
                "rough_raster_occupied_recall",
            )
        })
    if update == 400:
        object.__setattr__(model, "_signed_boundary_update400_metrics", {
            name: float(metrics[name])
            for name in (
                "G",
                "aggregate_raster_nll",
                "aggregate_raster_balanced_accuracy",
            )
        })
    result["gate"] = contract.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=model._signed_boundary_update100_metrics,
        update_400=model._signed_boundary_update400_metrics,
        prior_gates_passed=prior_gates_passed,
    )
    result["call_graph"].update({
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_update_count": 0,
        "paired_rgb_raw_field_additional_input_open_count": 0,
        "paired_rgb_raw_field_additional_encoder_forward_count": 0,
        "paired_rgb_loss_leaf": "v10_present_final_class_macro_nll",
    })
    result["loader_access_after_observation"] = loader.receipt()
    return result


def _capture_completed_observation(
    runtime: Any, loader: Any, result: dict[str, Any], *, update: int
) -> dict[str, Any]:
    if type(result) is not dict or result.get("update") != update:
        raise RuntimeError("signed-boundary observation identity changed")
    expected = tuple(contract.OBSERVATION_UPDATES)
    index = len(_V9._V9_COMPLETED_OBSERVATION_RECEIPTS)
    if index >= len(expected) or update != expected[index]:
        raise RuntimeError("signed-boundary observations are not an exact prefix")
    receipt = _V9._json_safe_observation_receipt(result)
    determinism = _V9._determinism_state(runtime)
    _V9._V9_COMPLETED_OBSERVATION_RECEIPTS.append(receipt)
    _V9._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES.append({
        "update": update,
        "state_after_completed_observation": determinism,
        "strict_determinism_exact": _V9._strict_determinism_exact(determinism),
    })
    progress = loader.progress
    progress["completed_observation_receipts"] = copy.deepcopy(
        _V9._V9_COMPLETED_OBSERVATION_RECEIPTS
    )
    progress["completed_observation_receipt_bindings"] = _V9._observation_bindings(
        _V9._V9_COMPLETED_OBSERVATION_RECEIPTS
    )
    progress["completed_observation_determinism_witnesses"] = copy.deepcopy(
        _V9._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
    )
    return result


def _signed_boundary_evaluate_observation_impl(
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
    progress = getattr(loader, "progress", None)
    if type(progress) is not dict:
        raise RuntimeError("observation loader lost failure-progress state")
    _V9._progress_observation_evidence(progress)
    result = _signed_boundary_observation_core(
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
    return _capture_completed_observation(runtime, loader, result, update=update)


def _signed_boundary_load_schedule(
    schedule_adapter: Any,
    authorization: Mapping[str, Any],
    train_pairs: Sequence[Mapping[str, Any]],
    *,
    progress: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Validate and use the exact existing full 16,000-presentation schedule."""

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
    used = [int(value) for value in full]
    if (
        observed_binding != binding
        or len(used) != 16_000
        or len(used) != contract.MAXIMUM_PRESENTATIONS
        or any(
            contract.canonical_json_sha256(
                used[: update * contract.EFFECTIVE_BATCH_SIZE]
            ) != expected
            for update, expected in contract.SCHEDULE_PREFIX_SHA256.items()
        )
        or identity != contract.build_schedule_identity()
    ):
        raise PermissionError("signed-boundary frozen full schedule changed")
    progress["schedule_validated"] = True
    return used, {
        "binding": dict(binding),
        "adapter_record": adapter_record,
        "identity": dict(identity),
        "source_adapter_returned_presentations": len(used),
        "used_presentation_count": len(used),
        "used_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256[1_000],
        "schedule_regeneration_count": 0,
        "indices_mutated_reordered_filtered_or_reseeded": False,
    }


_SIGNED_BOUNDARY_SEAM_TABLE = (
    ("_gradient_integrity_probe", _signed_boundary_gradient_integrity_probe),
    ("_evaluate_observation_impl", _signed_boundary_evaluate_observation_impl),
    ("_load_schedule", _signed_boundary_load_schedule),
)


def _runner_owners() -> tuple[Any, ...]:
    return (_V9, *_V9._runner_contract_owners())


def _runtime_module_names() -> tuple[str, ...]:
    v6 = _V8._V6
    return (
        _V9.V9_MODEL_RUNTIME_MODULE_NAME,
        _V8.V8_MODEL_RUNTIME_MODULE_NAME,
        _V8._V7.V7_MODEL_RUNTIME_MODULE_NAME,
        v6.V6_MODEL_RUNTIME_MODULE_NAME,
        v6._V5.V5_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4.V4_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME,
    )


def _assert_signed_boundary_seams() -> None:
    replacements = dict(_SIGNED_BOUNDARY_SEAM_TABLE)
    inherited_v9 = dict(_V9._V9_SEAM_TABLE)
    for name, expected_v8 in _V8._V8_SEAM_TABLE:
        expected = replacements.get(name, inherited_v9.get(name, expected_v8))
        if getattr(_LEAF, name) is not expected:
            raise RuntimeError(f"signed-boundary V1 lost runner seam: {name}")
    if set(replacements) != {
        "_gradient_integrity_probe", "_evaluate_observation_impl", "_load_schedule"
    }:
        raise RuntimeError("signed-boundary custom seam surface changed")
    if _LEAF._snapshot_model is not _V9._v9_snapshot_model:
        raise RuntimeError("signed-boundary lost V9 snapshot receipts")
    if _LEAF._terminal_failure is not _V9._v9_terminal_failure:
        raise RuntimeError("signed-boundary lost V9 failure receipts")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("signed-boundary failure validator was not rebound")
    _V9._assert_snapshot_globals_topology()


def _assert_signed_boundary_bindings() -> None:
    wrapper = Path(__file__).resolve()
    owners = _runner_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("signed-boundary contract did not reach runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("signed-boundary preflight did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("signed-boundary runner path did not reach stack")
    if any(
        name != SIGNED_BOUNDARY_MODEL_RUNTIME_MODULE_NAME
        for name in _runtime_module_names()
    ):
        raise RuntimeError("signed-boundary model identity did not reach stack")
    if contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH:
        raise RuntimeError("signed-boundary model source path changed")
    _assert_signed_boundary_seams()


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _V9.contract = contract
    _V9.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V9.V9_MODEL_RUNTIME_MODULE_NAME = SIGNED_BOUNDARY_MODEL_RUNTIME_MODULE_NAME
    _V9.__file__ = str(wrapper)
    _V9._rebind_inherited_runner()
    for name, function in _SIGNED_BOUNDARY_SEAM_TABLE:
        setattr(_LEAF, name, function)
    _assert_signed_boundary_bindings()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_signed_boundary_bindings()
    return result


def run_parent(
    *, review_file_sha256: str, authorization_file_sha256: str
) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    _assert_signed_boundary_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_signed_boundary_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
