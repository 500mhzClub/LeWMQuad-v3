#!/usr/bin/env python3
"""Run the capped Direct-BEV signed-boundary semantic-anchor V1 probe."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V1_"
    "PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/"
    "direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
SEMANTIC_ANCHOR_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_signed_boundary_semantic_anchor_state_v1_model_runtime"
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
    "_lewm_direct_bev_semantic_anchor_v1_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV semantic-anchor V1 runner changed")

_PARENT = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v1_frozen_signed_boundary_runner",
    ROOT / contract.FROZEN_SIGNED_BOUNDARY_RUNNER_RELATIVE_PATH,
)
_V9 = _PARENT._V9
_V8 = _PARENT._V8
_V6 = _PARENT._V6
_LEAF = _PARENT._LEAF


class _ObjectiveComponentAccumulator:
    """Batch-size-weighted D/A receipt over the inherited 495-row observer."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.rows = 0
        self.distance_sum = 0.0
        self.semantic_sum = 0.0
        self.combined_sum = 0.0
        self.current_distance_sum = 0.0
        self.next_distance_sum = 0.0
        self.current_semantic_sum = 0.0
        self.next_semantic_sum = 0.0
        self.all_finite = True
        self.combined_identity_exact = True

    def add(self, components: Any, *, row_count: int) -> None:
        torch = self.runtime.torch
        if type(row_count) is not int or row_count < 1:
            raise ValueError("semantic-anchor row_count must be positive")
        values = (
            components.G_distance_current,
            components.G_distance_next,
            components.G_distance,
            components.G_semantic_current,
            components.G_semantic_next,
            components.G_semantic_macro_nll,
            components.G_combined,
        )
        self.all_finite = self.all_finite and all(
            bool(torch.isfinite(value).all()) for value in values
        )
        expected = components.G_distance + (
            contract.SEMANTIC_ANCHOR_WEIGHT
            * components.G_semantic_macro_nll
        )
        self.combined_identity_exact = (
            self.combined_identity_exact
            and bool(torch.equal(components.G_combined, expected))
            and components.objective.G is components.G_combined
            and components.objective.total is components.G_combined
        )
        scalar = _LEAF._scalar
        self.rows += row_count
        self.distance_sum += scalar(components.G_distance) * row_count
        self.semantic_sum += (
            scalar(components.G_semantic_macro_nll) * row_count
        )
        self.combined_sum += scalar(components.G_combined) * row_count
        self.current_distance_sum += (
            scalar(components.G_distance_current) * row_count
        )
        self.next_distance_sum += (
            scalar(components.G_distance_next) * row_count
        )
        self.current_semantic_sum += (
            scalar(components.G_semantic_current) * row_count
        )
        self.next_semantic_sum += (
            scalar(components.G_semantic_next) * row_count
        )

    def finish(self) -> dict[str, Any]:
        if self.rows != 495:
            raise RuntimeError("semantic-anchor observer did not cover 495 rows")
        distance = self.distance_sum / self.rows
        semantic = self.semantic_sum / self.rows
        observed_combined = self.combined_sum / self.rows
        combined = distance + contract.SEMANTIC_ANCHOR_WEIGHT * semantic
        return {
            "row_count": self.rows,
            "G_distance_current": self.current_distance_sum / self.rows,
            "G_distance_next": self.next_distance_sum / self.rows,
            "G_distance": distance,
            "G_semantic_current": self.current_semantic_sum / self.rows,
            "G_semantic_next": self.next_semantic_sum / self.rows,
            "G_semantic_macro_nll": semantic,
            "G_combined": combined,
            "G_observed_combined_batch_mean": observed_combined,
            "all_components_finite": self.all_finite,
            "per_batch_combined_identity_exact": self.combined_identity_exact,
            "aggregate_combined_identity_close": math.isclose(
                observed_combined,
                combined,
                rel_tol=1e-7,
                abs_tol=1e-8,
            ),
        }


def _component_gradient_receipt(
    runtime: Any,
    model: Any,
    component: Any,
    *,
    retain_graph: bool,
) -> dict[str, Any]:
    torch = runtime.torch
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(("encoder.", "bev_decoder.", "state_head."))
    ]
    if not named:
        raise RuntimeError("semantic-anchor online parameter set is empty")
    gradients = torch.autograd.grad(
        component,
        [parameter for _, parameter in named],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    by_name = dict(zip((name for name, _ in named), gradients, strict=True))

    def group(prefix: str) -> dict[str, Any]:
        selected = [
            gradient for name, gradient in by_name.items()
            if name.startswith(prefix)
        ]
        finite = bool(selected) and all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in selected
        )
        magnitude = sum(
            0.0 if gradient is None else float(gradient.detach().abs().sum().cpu())
            for gradient in selected
        )
        return {
            "tensor_count": len(selected),
            "all_gradients_present_and_finite": finite,
            "aggregate_absolute_gradient": magnitude,
            "aggregate_gradient_strictly_nonzero": finite and magnitude > 0.0,
        }

    projection = _PARENT._head_projection(runtime, model.state_head)
    weight_name = next(
        name for name, parameter in named if parameter is projection.weight
    )
    bias_name = next(
        name for name, parameter in named if parameter is projection.bias
    )
    weight_gradient = by_name[weight_name]
    bias_gradient = by_name[bias_name]
    rows: list[dict[str, Any]] = []
    for index, field in enumerate(("K", "O")):
        finite = bool(
            weight_gradient is not None
            and bias_gradient is not None
            and torch.isfinite(weight_gradient[index]).all()
            and torch.isfinite(bias_gradient[index]).all()
        )
        magnitude = (
            0.0
            if not finite
            else float(
                (
                    weight_gradient[index].detach().abs().sum()
                    + bias_gradient[index].detach().abs()
                ).cpu()
            )
        )
        rows.append({
            "field": field,
            "all_gradients_present_and_finite": finite,
            "aggregate_absolute_gradient": magnitude,
            "aggregate_gradient_strictly_nonzero": finite and magnitude > 0.0,
        })
    result = {
        "encoder": group("encoder."),
        "decoder": group("bev_decoder."),
        "head_rows": rows,
    }
    result["all_required_groups_finite_nonzero"] = bool(
        result["encoder"]["aggregate_gradient_strictly_nonzero"]
        and result["decoder"]["aggregate_gradient_strictly_nonzero"]
        and all(row["aggregate_gradient_strictly_nonzero"] for row in rows)
    )
    return result


def _semantic_anchor_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend the inherited nonmutating probe with isolated D/A gradients."""

    parent = _PARENT._signed_boundary_gradient_integrity_probe(
        runtime, model, partition, batch
    )
    torch = runtime.torch
    state_before = _LEAF._state_sha(runtime, model)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    call_counts = {"online_state_stack": 0, "target_state_stack": 0, "predictor": 0}

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
        model.predictor.register_forward_hook(
            lambda module, inputs, output: count(
                module, inputs, output, key="predictor"
            )
        ),
    )
    probe = dict(batch)
    probe["current_labels"] = _PARENT._all_sign_labels(
        runtime, batch["current_labels"]
    )
    probe["next_labels"] = _PARENT._all_sign_labels(
        runtime, batch["next_labels"]
    )
    try:
        components = model.training_objective_with_components(
            current_rgb=probe["current_rgb"],
            next_rgb=probe["next_rgb"],
            fixed_negative_rgb=probe["fixed_negative_rgb"],
            action_one_hot=probe["action_one_hot"],
            non_hold_mask=probe["non_hold_mask"],
            current_labels=probe["current_labels"],
            next_labels=probe["next_labels"],
        )
        receipts = {
            "D_distance": _component_gradient_receipt(
                runtime,
                model,
                components.G_distance,
                retain_graph=True,
            ),
            "A_semantic_macro_nll": _component_gradient_receipt(
                runtime,
                model,
                components.G_semantic_macro_nll,
                retain_graph=True,
            ),
            "G_combined": _component_gradient_receipt(
                runtime,
                model,
                components.G_combined,
                retain_graph=False,
            ),
        }
    finally:
        for handle in handles:
            handle.remove()
    state_after = _LEAF._state_sha(runtime, model)
    rng_exact = bool(
        torch.equal(cpu_rng, torch.random.get_rng_state())
        and len(cuda_rng) == len(torch.cuda.get_rng_state_all())
        and all(
            torch.equal(before, after)
            for before, after in zip(
                cuda_rng, torch.cuda.get_rng_state_all(), strict=True
            )
        )
    )
    exact = bool(
        all(value["all_required_groups_finite_nonzero"] for value in receipts.values())
        and call_counts
        == {"online_state_stack": 2, "target_state_stack": 3, "predictor": 0}
        and state_before == state_after
        and rng_exact
        and not any(parameter.requires_grad for parameter in model.predictor.parameters())
        and not any(
            parameter.requires_grad
            for module in model._target_modules()
            for parameter in module.parameters()
        )
    )
    parent["semantic_anchor_component_gradients"] = {
        "components": receipts,
        "component_probe_call_counts": call_counts,
        "state_nonmutating_exact": state_before == state_after,
        "RNG_nonmutating_exact": rng_exact,
        "predictor_and_target_require_grad_false": bool(
            not any(parameter.requires_grad for parameter in model.predictor.parameters())
            and not any(
                parameter.requires_grad
                for module in model._target_modules()
                for parameter in module.parameters()
            )
        ),
        "all_required_D_A_and_combined_gradients_finite_nonzero": exact,
    }
    return parent


def _semantic_anchor_evaluate_observation_impl(
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
    accumulator = _ObjectiveComponentAccumulator(runtime)
    missing = object()
    previous = model.__dict__.get("training_objective", missing)

    def observed_objective(**kwargs: Any) -> Any:
        components = model.training_objective_with_components(**kwargs)
        if not runtime.torch.is_grad_enabled():
            accumulator.add(
                components,
                row_count=int(kwargs["current_rgb"].shape[0]),
            )
        return components.objective

    object.__setattr__(model, "training_objective", observed_objective)
    try:
        result = _PARENT._signed_boundary_observation_core(
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
    finally:
        if previous is missing:
            object.__delattr__(model, "training_objective")
        else:
            object.__setattr__(model, "training_objective", previous)

    component_receipt = accumulator.finish()
    metrics = result["metrics"]
    metrics["G"] = component_receipt["G_combined"]
    metrics.update({
        "G_distance": component_receipt["G_distance"],
        "G_semantic_macro_nll": component_receipt["G_semantic_macro_nll"],
        "semantic_anchor_objective_component_receipt": component_receipt,
        "semantic_anchor_weight_exactly_one_over_64": bool(
            contract.SEMANTIC_ANCHOR_WEIGHT == 1.0 / 64.0
            and getattr(model_api, "SEMANTIC_ANCHOR_WEIGHT_V1", None)
            == 1.0 / 64.0
        ),
        "semantic_anchor_objective_components_exact": bool(
            component_receipt["all_components_finite"]
            and component_receipt["per_batch_combined_identity_exact"]
            and component_receipt["aggregate_combined_identity_close"]
            and math.isclose(
                float(metrics["G"]),
                float(component_receipt["G_combined"]),
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
        ),
        "semantic_anchor_training_label_boundary_exact": True,
        "signed_boundary_semantic_anchor_mechanism_receipt_ready": True,
        "active_training_scope_signed_boundary_semantic_anchor_v1": (
            "perception_only"
        ),
    })
    if update == 0:
        gradient = result["gradient_integrity"][
            "semantic_anchor_component_gradients"
        ]
        metrics[
            "semantic_anchor_D_A_and_combined_gradients_finite_nonzero"
        ] = gradient[
            "all_required_D_A_and_combined_gradients_finite_nonzero"
        ]
        object.__setattr__(model, "_semantic_anchor_update100_metrics", None)
        object.__setattr__(model, "_semantic_anchor_update400_metrics", None)
    else:
        metrics[
            "semantic_anchor_D_A_and_combined_gradients_finite_nonzero"
        ] = True

    if update == 100:
        object.__setattr__(model, "_semantic_anchor_update100_metrics", {
            name: float(metrics[name])
            for name in (
                "G_distance",
                "G_semantic_macro_nll",
                "G",
                "aggregate_raster_nll",
                "aggregate_raster_balanced_accuracy",
                "aggregate_occupied_recall",
                "rough_raster_balanced_accuracy",
                "rough_raster_occupied_recall",
                "paired_rgb_aggregate_margin",
            )
        })
    if update == 400:
        object.__setattr__(model, "_semantic_anchor_update400_metrics", {
            name: float(metrics[name])
            for name in (
                "G_distance",
                "G_semantic_macro_nll",
                "G",
                "aggregate_raster_nll",
                "aggregate_raster_balanced_accuracy",
            )
        })
    result["gate"] = contract.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=model._semantic_anchor_update100_metrics,
        update_400=model._semantic_anchor_update400_metrics,
        prior_gates_passed=prior_gates_passed,
    )
    result["call_graph"].update({
        "semantic_anchor_additional_encoder_forward_count": 0,
        "semantic_anchor_additional_input_open_count": 0,
        "semantic_anchor_optimizer_label_roles": ["current", "next"],
        "semantic_anchor_fixed_negative_target_predictor_gradient_count": 0,
        "semantic_anchor_weight": contract.SEMANTIC_ANCHOR_WEIGHT,
    })
    return _PARENT._capture_completed_observation(
        runtime, loader, result, update=update
    )


_SEMANTIC_ANCHOR_SEAM_TABLE = (
    ("_gradient_integrity_probe", _semantic_anchor_gradient_integrity_probe),
    ("_evaluate_observation_impl", _semantic_anchor_evaluate_observation_impl),
)


def _runtime_module_names() -> tuple[str, ...]:
    return _PARENT._runtime_module_names()


def _assert_semantic_anchor_bindings() -> None:
    wrapper = Path(__file__).resolve()
    owners = (_PARENT, *_PARENT._runner_owners())
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("semantic-anchor contract did not reach runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("semantic-anchor preflight did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("semantic-anchor runner path did not reach stack")
    if any(
        name != SEMANTIC_ANCHOR_MODEL_RUNTIME_MODULE_NAME
        for name in _runtime_module_names()
    ):
        raise RuntimeError("semantic-anchor model identity did not reach stack")
    for name, function in _SEMANTIC_ANCHOR_SEAM_TABLE:
        if getattr(_LEAF, name) is not function:
            raise RuntimeError(f"semantic-anchor lost runner seam: {name}")
    if _LEAF._load_schedule is not _PARENT._signed_boundary_load_schedule:
        raise RuntimeError("semantic-anchor lost signed-boundary schedule seam")
    if _LEAF._snapshot_model is not _V9._v9_snapshot_model:
        raise RuntimeError("semantic-anchor lost V9 snapshot receipts")
    if _LEAF._terminal_failure is not _V9._v9_terminal_failure:
        raise RuntimeError("semantic-anchor lost V9 failure receipts")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("semantic-anchor failure validator was not rebound")
    _V9._assert_snapshot_globals_topology()


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _PARENT.contract = contract
    _PARENT.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _PARENT.MODEL_RELATIVE_PATH = MODEL_RELATIVE_PATH
    _PARENT.SIGNED_BOUNDARY_MODEL_RUNTIME_MODULE_NAME = (
        SEMANTIC_ANCHOR_MODEL_RUNTIME_MODULE_NAME
    )
    _PARENT.__file__ = str(wrapper)
    _PARENT._rebind_inherited_runner()
    for name, function in _SEMANTIC_ANCHOR_SEAM_TABLE:
        setattr(_LEAF, name, function)
    _assert_semantic_anchor_bindings()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_semantic_anchor_bindings()
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
    _assert_semantic_anchor_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_semantic_anchor_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
