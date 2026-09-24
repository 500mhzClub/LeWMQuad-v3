#!/usr/bin/env python3
"""Run the capped Direct-BEV multiprototype state-head V1 probe."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_MULTIPROTOTYPE_STATE_HEAD_V1_PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_multiprototype_state_head_v1.py"
)
MULTIPROTOTYPE_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_multiprototype_state_head_v1_model_runtime"
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
    "_lewm_direct_bev_multiprototype_v1_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV multiprototype V1 runner identity changed")

# V10 through V12 add identity/gate wrappers but no runner mechanics after V9.
# Start at V9 so its snapshot and complete-failure receipts remain authoritative.
_V9 = _source_only_module(
    "_lewm_direct_bev_multiprototype_v1_frozen_v9_runner",
    ROOT / contract.FROZEN_V9_RUNNER_RELATIVE_PATH,
)
_V8 = _V9._V8
_V6 = _V8._V6
_LEAF = _V9._LEAF
_FROZEN_V8_GRADIENT_INTEGRITY_PROBE = _V8._v8_gradient_integrity_probe

_GRADIENT_PROBE_ACTIVE = False
_CLASS_NAMES = ("UNKNOWN", "FREE", "OCCUPIED")


def _prototype_row_gradient_receipt(runtime: Any, gradient: Any) -> dict[str, Any]:
    torch = runtime.torch
    if tuple(gradient.shape) != (3, 4, 64):
        raise RuntimeError("multiprototype gradient shape changed")
    classes: dict[str, Any] = {}
    all_exact = True
    for class_index, class_name in enumerate(_CLASS_NAMES):
        rows = []
        for component in range(4):
            row = gradient[class_index, component]
            finite = bool(torch.isfinite(row).all())
            absolute_sum = float(row.double().abs().sum().cpu())
            exact = finite and absolute_sum > 0.0
            all_exact = all_exact and exact
            rows.append({
                "component": component,
                "all_finite": finite,
                "absolute_gradient_sum": absolute_sum,
                "finite_nonzero": exact,
            })
        classes[class_name] = rows
    return {
        "shape": [3, 4, 64],
        "capture_count": 1,
        "classes": classes,
        "all_twelve_rows_finite_nonzero": all_exact,
    }


def _multiprototype_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture all twelve row gradients before frozen V8 restores them."""

    global _GRADIENT_PROBE_ACTIVE
    if _GRADIENT_PROBE_ACTIVE:
        raise RuntimeError("multiprototype gradient probe re-entered")
    prototypes = model.state_head.prototypes
    if tuple(prototypes.shape) != (3, 4, 64):
        raise RuntimeError("multiprototype parameter shape changed")
    captured: list[Any] = []

    def capture(gradient: Any) -> None:
        captured.append(gradient.detach().clone())

    handle = prototypes.register_hook(capture)
    _GRADIENT_PROBE_ACTIVE = True
    try:
        result = _FROZEN_V8_GRADIENT_INTEGRITY_PROBE(
            runtime, model, partition, batch
        )
    finally:
        _GRADIENT_PROBE_ACTIVE = False
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("multiprototype row gradient capture count changed")
    receipt = _prototype_row_gradient_receipt(runtime, captured[0])
    result["multiprototype_prototype_row_gradients"] = receipt
    result["all_twelve_prototype_row_gradients_finite_nonzero"] = receipt[
        "all_twelve_rows_finite_nonzero"
    ]
    return result


class _MultiprototypeUtilizationAccumulator:
    """Aggregate descriptive current-side responsibilities without new calls."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.row_count = 0
        self.cell_counts = [0, 0, 0]
        self.responsibility_sums = [[0.0] * 4 for _ in range(3)]
        self.winner_counts = [[0] * 4 for _ in range(3)]
        self.entropy_sums = [0.0, 0.0, 0.0]

    def add(self, component_logits: Any, labels: Any) -> None:
        torch = self.runtime.torch
        if (
            tuple(component_logits.shape[1:]) != (3, 4, 64, 64)
            or tuple(labels.shape) != (
                component_logits.shape[0], 64, 64
            )
            or not bool(torch.isfinite(component_logits).all())
        ):
            raise RuntimeError("multiprototype utilization population changed")
        responsibilities = torch.softmax(component_logits, dim=2)
        self.row_count += int(labels.shape[0])
        for state_class in range(3):
            mask = labels == state_class
            count = int(mask.sum().detach().cpu())
            if count == 0:
                continue
            selected = responsibilities[:, state_class].permute(
                0, 2, 3, 1
            )[mask]
            sums = selected.double().sum(dim=0).cpu().tolist()
            winners = torch.bincount(
                selected.argmax(dim=1), minlength=4
            ).cpu().tolist()
            entropy = -(selected.double() * selected.double().log()).sum(dim=1)
            self.cell_counts[state_class] += count
            self.entropy_sums[state_class] += float(entropy.sum().cpu())
            for component in range(4):
                self.responsibility_sums[state_class][component] += float(
                    sums[component]
                )
                self.winner_counts[state_class][component] += int(
                    winners[component]
                )

    def receipt(self) -> dict[str, Any]:
        if self.row_count != 495:
            raise RuntimeError("multiprototype utilization did not cover 495 rows")
        classes: dict[str, Any] = {}
        for state_class, class_name in enumerate(_CLASS_NAMES):
            count = self.cell_counts[state_class]
            if count == 0:
                classes[class_name] = {
                    "target_class_valid_cell_count": 0,
                    "per_component_posterior_responsibility_mean": None,
                    "per_component_winner_share": None,
                    "mean_responsibility_entropy_nats": None,
                    "effective_component_count": None,
                }
                continue
            mean_entropy = self.entropy_sums[state_class] / count
            classes[class_name] = {
                "target_class_valid_cell_count": count,
                "per_component_posterior_responsibility_mean": [
                    value / count
                    for value in self.responsibility_sums[state_class]
                ],
                "per_component_winner_share": [
                    value / count for value in self.winner_counts[state_class]
                ],
                "mean_responsibility_entropy_nats": mean_entropy,
                "effective_component_count": math.exp(mean_entropy),
            }
        return {
            "schema": f"{contract.SCHEMA_PREFIX}_multiprototype_utilization_v1",
            "descriptive_only": True,
            "population": {
                "role": "checkpoint_selection",
                "side": "current",
                "row_count": self.row_count,
            },
            "classes": classes,
        }


class _MultiprototypeObservationLoaderProxy:
    """Associate existing no-grad online calls with their current labels."""

    def __init__(self, loader: Any, accumulator: Any) -> None:
        self._loader = loader
        self._accumulator = accumulator
        self._current_labels: Any | None = None
        self._online_call_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def _finish_pair_batch(self) -> None:
        if self._current_labels is not None and self._online_call_index != 3:
            raise RuntimeError("frozen observation online-call order changed")
        self._current_labels = None
        self._online_call_index = 0

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        self._finish_pair_batch()
        batch = self._loader.batch(*args, **kwargs)
        self._current_labels = batch["current_labels"]
        return batch

    def endpoint_batch(self, *args: Any, **kwargs: Any) -> Any:
        self._finish_pair_batch()
        return self._loader.endpoint_batch(*args, **kwargs)

    def observe_online_head(
        self, module: Any, inputs: tuple[Any, ...], _output: Any
    ) -> None:
        if _GRADIENT_PROBE_ACTIVE or self._current_labels is None:
            return
        if self._loader.runtime.torch.is_grad_enabled():
            return
        if self._online_call_index == 0:
            if len(inputs) != 1:
                raise RuntimeError("multiprototype state-head inputs changed")
            self._accumulator.add(
                module.component_logits(inputs[0]), self._current_labels
            )
        self._online_call_index += 1
        if self._online_call_index > 3:
            raise RuntimeError("frozen observation online-call count changed")

    def finish(self) -> dict[str, Any]:
        self._finish_pair_batch()
        return self._accumulator.receipt()


def _architecture_receipt(runtime: Any, model: Any) -> dict[str, Any]:
    torch = runtime.torch
    decoder = model.bev_decoder
    head = model.state_head
    modules = tuple(decoder.modules())
    named_parameters = dict(decoder.named_parameters())
    forbidden = {
        name
        for name in (*dict(decoder.named_parameters()), *dict(decoder.named_buffers()))
        if any(token in name.casefold() for token in (
            "coordinate", "sinusoid", "ray", "pose"
        ))
    }
    prototypes = head.prototypes.detach()
    rows = prototypes.reshape(12, 64)
    pairwise_distinct = all(
        not torch.equal(rows[left], rows[right])
        for left in range(12)
        for right in range(left + 1, 12)
    )
    return {
        "row_query_shape_exact": tuple(decoder.row_query.shape) == (64, 64),
        "column_query_shape_exact": tuple(decoder.column_query.shape) == (64, 64),
        "full_per_cell_query_parameter_absent": all(
            tuple(value.shape) != (4096, 64)
            for value in named_parameters.values()
        ),
        "two_independent_blocks_exact": decoder.block_1 is not decoder.block_2,
        "spatial_convolution_absent": not any(
            isinstance(module, torch.nn.Conv2d) for module in modules
        ),
        "numeric_coordinate_or_geometry_state_absent": not forbidden,
        "prototype_shape_exact": tuple(prototypes.shape) == (3, 4, 64),
        "prototype_parameter_count": int(prototypes.numel()),
        "prototype_rows_finite_nonzero": bool(
            torch.isfinite(rows).all() and (rows.norm(dim=1) > 0.0).all()
        ),
        "prototype_rows_pairwise_bitwise_distinct": pairwise_distinct,
        "state_head_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "decoder_and_head_parameter_count": sum(
            parameter.numel()
            for module in (decoder, head)
            for parameter in module.parameters()
        ),
        "out_channels_exactly_three": getattr(head, "out_channels", None) == 3,
        "prototype_is_only_head_parameter": set(dict(head.named_parameters()))
        == {"prototypes"},
        "head_buffer_count_zero": not dict(head.named_buffers()),
    }


def _head_witness(runtime: Any, model: Any, rgb: Any) -> dict[str, Any]:
    torch = runtime.torch
    with torch.no_grad():
        tokens = model.encoder.forward_tokens(rgb)[:, 1:]
        cells = model.bev_decoder(tokens)
        observed_components = model.state_head.component_logits(cells)
        observed_logits = model.state_head(cells)
        features = torch.nn.functional.normalize(cells, dim=1, eps=1e-12)
        prototypes = torch.nn.functional.normalize(
            model.state_head.prototypes, dim=2, eps=1e-12
        ).to(dtype=cells.dtype)
        exact_components = -(
            features[:, None, None]
            - prototypes[None, :, :, :, None, None]
        ).square().sum(dim=3)
        exact_logits = torch.logsumexp(exact_components, dim=2) - math.log(4.0)
    return {
        "components_exact": torch.equal(observed_components, exact_components),
        "aggregation_exact": torch.equal(observed_logits, exact_logits),
        "logits": observed_logits,
        "normalized_prototype_gram": torch.einsum(
            "ckd,cld->ckl", prototypes, prototypes
        ).detach().cpu().tolist(),
    }


def _multiprototype_observation_core(
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
    accumulator = _MultiprototypeUtilizationAccumulator(runtime)
    proxy = _MultiprototypeObservationLoaderProxy(loader, accumulator)
    hook = model.state_head.register_forward_hook(proxy.observe_online_head)
    try:
        result = _V6._FROZEN_EVALUATE_OBSERVATION_IMPL(
            runtime,
            model_api,
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
        utilization = proxy.finish()
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
    metrics.update({
        **accounting,
        "multiprototype_mechanism_receipt_ready": True,
        "active_training_scope_multiprototype_v1": "perception_only",
        "multiprototype_utilization": utilization,
        "all_forbidden_access_counts_zero": forbidden_access_zero,
    })

    if update == 0:
        indices = list(range(min(contract.MICROBATCH_SIZE, len(selection_pairs))))
        batch = loader.batch(
            selection_pairs,
            indices,
            device,
            role="checkpoint_selection",
            stage="observation_update_0_multiprototype_architecture_witness",
            mapped_negative_indices=selection_mapping["negative_indices"],
            scope="observation",
        )
        was_training = bool(model.training)
        model.eval()
        try:
            witness = _head_witness(runtime, model, batch["current_rgb"])
        finally:
            model.train(was_training)
        architecture = _architecture_receipt(runtime, model)
        gradient = result["gradient_integrity"]
        perception_gradient = gradient["v8_perception_only"]
        online_target_equal = (
            _V6._normalized_state_sha256(
                runtime, model, _V6._ONLINE_PERCEPTION_PREFIXES
            )
            == _V6._normalized_state_sha256(
                runtime, model, _V6._TARGET_PERCEPTION_PREFIXES
            )
        )
        inventory_exact = partition["receipt"] == contract.MODEL_PARAMETER_INVENTORY
        logits = witness["logits"]
        rows_exact = bool(
            architecture["prototype_rows_finite_nonzero"]
            and architecture["prototype_rows_pairwise_bitwise_distinct"]
        )
        formula_exact = bool(
            witness["components_exact"]
            and witness["aggregation_exact"]
            and architecture["prototype_shape_exact"]
            and architecture["prototype_is_only_head_parameter"]
            and architecture["head_buffer_count_zero"]
        )
        excluded_gradients = bool(
            perception_gradient["predictor_gradient_absent"]
            and perception_gradient["target_gradients_absent"]
            and perception_gradient[
                "fixed_negative_rgb_optimizer_gradient_absent"
            ]
        )
        metrics.update({
            "architecture_receipt": architecture,
            "normalized_within_class_prototype_gram": witness[
                "normalized_prototype_gram"
            ],
            "fresh_multiprototype_model_and_optimizer_zero_prior_runtime_reuse": bool(
                model._v6_no_prior_runtime_or_protected_input
                and model._v6_optimizer_for_integrity_probe is not None
            ),
            "frozen_encoder_decoder_predictor_initialization_exact": bool(
                model._v8_initial_components_exact
            ),
            "registered_seed_draw_order_exact": True,
            "multiprototype_initial_head_state_sha256_exact": (
                _V8._component_state_sha256(runtime, model.state_head)
                == contract.MULTIPROTOTYPE_INITIAL_HEAD_STATE_SHA256
            ),
            "model_parameter_inventory_exact": inventory_exact,
            "multiprototype_decoder_parameter_inventory_exact": bool(
                architecture["decoder_and_head_parameter_count"] == 88_384
                and architecture["state_head_parameter_count"] == 768
                and architecture["prototype_parameter_count"] == 768
            ),
            "learned_only_forbidden_geometry_absent": bool(
                architecture["numeric_coordinate_or_geometry_state_absent"]
                and architecture["full_per_cell_query_parameter_absent"]
                and architecture["spatial_convolution_absent"]
            ),
            "two_residual_cross_attention_ffn_blocks_exact": architecture[
                "two_independent_blocks_exact"
            ],
            "multiprototype_shape_formula_axes_and_equal_weight_exact": formula_exact,
            "all_twelve_prototype_rows_finite_nonidentical": rows_exact,
            "all_twelve_prototype_row_gradients_finite_nonzero": gradient[
                "all_twelve_prototype_row_gradients_finite_nonzero"
            ],
            "online_target_perception_bitwise_equal": online_target_equal,
            "target_requires_grad_false": not any(
                parameter.requires_grad
                for module in model._target_modules()
                for parameter in module.parameters()
            ),
            "three_channel_state_exact": architecture[
                "out_channels_exactly_three"
            ],
            "all_logits_in_closed_interval_minus4_to0": bool(
                float(logits.min().detach().cpu()) >= -4.000001
                and float(logits.max().detach().cpu()) <= 0.000001
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
            "initial_online_to_target_hard_sync_count": 1,
        })

    if update == 100:
        object.__setattr__(model, "_v8_update100_metrics", {
            "aggregate_raster_nll": float(metrics["aggregate_raster_nll"]),
            "rough_raster_occupied_recall": float(
                metrics["rough_raster_occupied_recall"]
            ),
        })
    result["gate"] = contract.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        update_100=model._v8_update100_metrics,
        prior_gates_passed=prior_gates_passed,
    )
    result["call_graph"].update({
        "predictor_forward_call_count": 0,
        "predictor_objective_evaluation_count": 0,
        "predictor_backward_call_count": 0,
        "predictor_optimizer_update_count": 0,
        "multiprototype_utilization_additional_input_open_count": 0,
        "multiprototype_utilization_additional_encoder_forward_count": 0,
    })
    result["loader_access_after_observation"] = loader.receipt()
    return result


def _capture_completed_observation(
    runtime: Any, loader: Any, result: dict[str, Any], *, update: int
) -> dict[str, Any]:
    if type(result) is not dict or result.get("update") != update:
        raise RuntimeError("multiprototype completed observation identity changed")
    expected = tuple(contract.OBSERVATION_UPDATES)
    index = len(_V9._V9_COMPLETED_OBSERVATION_RECEIPTS)
    if index >= len(expected) or update != expected[index]:
        raise RuntimeError("completed observations are not an exact prefix")
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


def _multiprototype_evaluate_observation_impl(
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
    result = _multiprototype_observation_core(
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


_MULTIPROTOTYPE_SEAM_TABLE = (
    ("_gradient_integrity_probe", _multiprototype_gradient_integrity_probe),
    ("_evaluate_observation_impl", _multiprototype_evaluate_observation_impl),
)


def _runner_owners() -> tuple[Any, ...]:
    return (_V9, *_V9._runner_contract_owners())


def _runtime_module_names() -> tuple[str, ...]:
    v8 = _V9._V8
    v6 = v8._V6
    return (
        _V9.V9_MODEL_RUNTIME_MODULE_NAME,
        v8.V8_MODEL_RUNTIME_MODULE_NAME,
        v8._V7.V7_MODEL_RUNTIME_MODULE_NAME,
        v6.V6_MODEL_RUNTIME_MODULE_NAME,
        v6._V5.V5_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4.V4_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME,
    )


def _assert_multiprototype_seams() -> None:
    replacements = dict(_MULTIPROTOTYPE_SEAM_TABLE)
    inherited_v9 = dict(_V9._V9_SEAM_TABLE)
    for name, expected_v8 in _V8._V8_SEAM_TABLE:
        expected = replacements.get(name, inherited_v9.get(name, expected_v8))
        if getattr(_LEAF, name) is not expected:
            raise RuntimeError(f"multiprototype V1 lost runner seam: {name}")
    if set(replacements) != {
        "_gradient_integrity_probe", "_evaluate_observation_impl"
    }:
        raise RuntimeError("multiprototype V1 custom seam surface changed")
    if _LEAF._snapshot_model is not _V9._v9_snapshot_model:
        raise RuntimeError("multiprototype V1 lost V9 snapshot receipts")
    if _LEAF._terminal_failure is not _V9._v9_terminal_failure:
        raise RuntimeError("multiprototype V1 lost V9 failure receipts")
    if _V9._FROZEN_V8_EVALUATE_OBSERVATION_IMPL is not (
        _V8._v8_evaluate_observation_impl
    ):
        raise RuntimeError("captured frozen V8 observer was mutated")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("failure-chain validator was not rebound")
    _V9._assert_snapshot_globals_topology()


def _assert_multiprototype_bindings() -> None:
    wrapper = Path(__file__).resolve()
    owners = _runner_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("multiprototype V1 contract did not reach runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("multiprototype preflight did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("multiprototype runner path did not reach stack")
    if any(
        name != MULTIPROTOTYPE_MODEL_RUNTIME_MODULE_NAME
        for name in _runtime_module_names()
    ):
        raise RuntimeError("multiprototype model identity did not reach stack")
    if contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH:
        raise RuntimeError("multiprototype model source path changed")
    _assert_multiprototype_seams()


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _V9.contract = contract
    _V9.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V9.V9_MODEL_RUNTIME_MODULE_NAME = MULTIPROTOTYPE_MODEL_RUNTIME_MODULE_NAME
    _V9.__file__ = str(wrapper)
    _V9._rebind_inherited_runner()
    for name, function in _MULTIPROTOTYPE_SEAM_TABLE:
        setattr(_LEAF, name, function)
    _assert_multiprototype_bindings()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_multiprototype_bindings()
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
    _assert_multiprototype_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_multiprototype_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
