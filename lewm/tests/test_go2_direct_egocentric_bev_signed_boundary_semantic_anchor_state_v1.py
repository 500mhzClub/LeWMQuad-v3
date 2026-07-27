from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
MODEL = ROOT / "lewm/models" / f"direct_egocentric_bev_{STEM.removeprefix('go2_direct_egocentric_bev_')}.py"
TEST = ROOT / "lewm/tests" / f"test_{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
DISTANCE_MODEL = (
    ROOT
    / "lewm/models/direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _encoder_state(distance_api: Any, *, retain_values: bool) -> Mapping[str, Any]:
    encoder = (
        distance_api._v10._v8._v6._v3._v1
        ._construct_n320_encoder_without_rng_draw()
    )
    if not retain_values:
        for value in encoder.state_dict().values():
            if value.is_floating_point():
                value.zero_()
    return encoder.state_dict()


def _slow_present_class_macro_nll(
    torch: Any,
    logits: Any,
    labels: Any,
) -> Any:
    log_probabilities = torch.log_softmax(logits, dim=1)
    rows = []
    for row in range(labels.shape[0]):
        present = []
        for state_class in (0, 1, 2):
            mask = labels[row] == state_class
            if bool(mask.any()):
                present.append(
                    (-log_probabilities[row, state_class][mask]).mean()
                )
        rows.append(torch.stack(present).mean())
    return torch.stack(rows)


def _objective_values(distance: float, semantic: float) -> dict[str, float]:
    combined = distance + semantic / 64.0
    return {
        "G_distance": distance,
        "G_semantic_macro_nll": semantic,
        "G": combined,
    }


def _common_gate_metrics(
    contract: Any,
    update: int,
    *,
    distance: float,
    semantic: float,
) -> dict[str, Any]:
    return {
        **contract.perception_accounting(update),
        **_objective_values(distance, semantic),
        "signed_boundary_semantic_anchor_mechanism_receipt_ready": True,
        "active_training_scope_signed_boundary_semantic_anchor_v1": (
            "perception_only"
        ),
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "all_forbidden_access_counts_zero": True,
    }


def _update_zero_metrics(contract: Any) -> dict[str, Any]:
    return {
        **_common_gate_metrics(
            contract,
            0,
            distance=1.0,
            semantic=2.0,
        ),
        **{field: True for field in contract.INTEGRITY_FIELDS},
        "initial_online_to_target_hard_sync_count": 1,
        # These are references for the update-100 relative gates, not
        # update-zero scientific direction checks.
        "aggregate_raster_nll": 1.0,
        "aggregate_raster_balanced_accuracy": 0.60,
        "paired_rgb_aggregate_margin": 0.20,
    }


def _update_100_metrics(contract: Any) -> dict[str, Any]:
    zero = _update_zero_metrics(contract)
    return {
        **_common_gate_metrics(
            contract,
            100,
            distance=0.80,
            semantic=1.50,
        ),
        "aggregate_raster_nll": zero["aggregate_raster_nll"] - 0.15,
        "aggregate_raster_balanced_accuracy": max(
            0.68,
            zero["aggregate_raster_balanced_accuracy"] + 0.08,
        ),
        "aggregate_free_recall": 0.60,
        "aggregate_occupied_recall": 0.30,
        "rough_raster_balanced_accuracy": 0.63,
        "rough_raster_occupied_recall": 0.45,
        "paired_rgb_aggregate_margin": math.nextafter(
            zero["paired_rgb_aggregate_margin"], math.inf
        ),
        "correct_rgb_scene_win_count": 6,
    }


def _update_400_metrics(contract: Any) -> dict[str, Any]:
    hundred = _update_100_metrics(contract)
    return {
        **_common_gate_metrics(
            contract,
            400,
            distance=0.70,
            semantic=1.40,
        ),
        "aggregate_raster_nll": min(
            0.55,
            hundred["aggregate_raster_nll"],
        ),
        "aggregate_raster_balanced_accuracy": max(
            0.72,
            hundred["aggregate_raster_balanced_accuracy"] - 0.01,
        ),
        "aggregate_free_recall": 0.65,
        "aggregate_occupied_recall": max(
            0.55,
            hundred["aggregate_occupied_recall"],
        ),
        "rough_raster_balanced_accuracy": max(
            0.65,
            hundred["rough_raster_balanced_accuracy"] + 0.02,
        ),
        "rough_raster_occupied_recall": max(
            0.50,
            hundred["rough_raster_occupied_recall"] + 0.05,
        ),
        "paired_rgb_aggregate_margin": math.nextafter(0.0, math.inf),
        "correct_rgb_scene_win_count": 7,
    }


def _update_1000_metrics(contract: Any) -> dict[str, Any]:
    four_hundred = _update_400_metrics(contract)
    return {
        **_common_gate_metrics(
            contract,
            1_000,
            distance=four_hundred["G_distance"],
            semantic=four_hundred["G_semantic_macro_nll"],
        ),
        "aggregate_raster_nll": min(
            0.42,
            four_hundred["aggregate_raster_nll"],
        ),
        "aggregate_raster_balanced_accuracy": max(
            0.80,
            four_hundred["aggregate_raster_balanced_accuracy"],
        ),
        "aggregate_unknown_recall": 0.80,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "rough_raster_balanced_accuracy": 0.772,
        "rough_raster_occupied_recall": 0.65,
        "paired_rgb_aggregate_margin": math.nextafter(0.0, math.inf),
        "correct_rgb_scene_win_count": 7,
    }


def _evaluate(
    contract: Any,
    update: int,
    metrics: Mapping[str, Any],
    *,
    zero: Mapping[str, Any],
    hundred: Mapping[str, Any],
    four_hundred: Mapping[str, Any],
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"prior_gates_passed": prior_gates_passed}
    if update == 100:
        kwargs["update_zero"] = zero
    elif update == 400:
        kwargs["update_100"] = hundred
    elif update == 1_000:
        kwargs["update_400"] = four_hundred
    return contract.evaluate_gate(update, metrics, **kwargs)


def _replace_objective(
    metrics: Mapping[str, Any],
    *,
    distance: float | None = None,
    semantic: float | None = None,
    combined: float | None = None,
) -> dict[str, Any]:
    result = dict(metrics)
    if distance is not None:
        result["G_distance"] = distance
    if semantic is not None:
        result["G_semantic_macro_nll"] = semantic
    if combined is None:
        result["G"] = (
            result["G_distance"] + result["G_semantic_macro_nll"] / 64.0
        )
    else:
        result["G"] = combined
    return result


def test_integration_sentinel_all_six_additive_sources_exist() -> None:
    """INTEGRATION SENTINEL: concurrent source authors must fill every role."""

    missing = [
        str(path.relative_to(ROOT))
        for path in (CONTRACT, MODEL, TEST, CHECKER, RUNNER, LAUNCHER)
        if not path.is_file()
    ]
    assert not missing, f"semantic-anchor integration source(s) absent: {missing}"


def test_fixed_weight_and_fresh_model_state_are_identical_to_predecessor() -> None:
    distance_api = _load(DISTANCE_MODEL, "_semantic_anchor_frozen_distance_model")
    api = _load(MODEL, "_semantic_anchor_model_identity")
    torch = api.torch
    encoder_state = _encoder_state(distance_api, retain_values=False)
    caller_rng = torch.random.get_rng_state().clone()
    predecessor = distance_api.DirectEgocentricBevStateJepaV1(encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    successor = api.DirectEgocentricBevStateJepaV1(encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    assert api.SEMANTIC_ANCHOR_WEIGHT_V1 == 1.0 / 64.0
    assert tuple(successor.state_dict()) == tuple(predecessor.state_dict())
    for name, value in successor.state_dict().items():
        assert torch.equal(value, predecessor.state_dict()[name]), name
    assert [name for name, _ in successor.named_parameters()] == [
        name for name, _ in predecessor.named_parameters()
    ]
    assert [name for name, _ in successor.named_buffers()] == [
        name for name, _ in predecessor.named_buffers()
    ]
    assert [name for name, _ in successor.named_modules()] == [
        name for name, _ in predecessor.named_modules()
    ]
    assert (
        sum(parameter.numel() for parameter in successor.parameters()),
        len(tuple(successor.parameters())),
    ) == (5_987_639, 299)
    assert not any(
        "anchor" in name.lower()
        for name, _ in (
            *tuple(successor.named_parameters()),
            *tuple(successor.named_buffers()),
        )
    )


def test_exact_anchor_macro_absent_classes_side_averaging_and_combined_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distance_api = _load(DISTANCE_MODEL, "_semantic_anchor_distance_math_base")
    api = _load(MODEL, "_semantic_anchor_exact_math")
    torch = api.torch
    model = api.DirectEgocentricBevStateJepaV1(
        _encoder_state(distance_api, retain_values=False)
    )
    model.arm_phase_schedule_v6()

    batch = 3
    current_labels = torch.empty(batch, 64, 64, dtype=torch.long)
    current_labels[0].zero_()  # UNKNOWN only: FREE and OCCUPIED are absent.
    current_labels[1].copy_(
        torch.arange(64 * 64).reshape(64, 64).remainder(2).add(1)
    )  # FREE/OCCUPIED only: UNKNOWN is absent.
    current_labels[2].copy_(
        torch.arange(64 * 64).reshape(64, 64).remainder(3)
    )
    next_labels = current_labels.roll(1, dims=2)

    current_fields = torch.linspace(
        -0.45,
        0.45,
        batch * 2 * 64 * 64,
        dtype=torch.float32,
    ).reshape(batch, 2, 64, 64).requires_grad_()
    next_fields = torch.linspace(
        0.40,
        -0.40,
        batch * 2 * 64 * 64,
        dtype=torch.float32,
    ).reshape(batch, 2, 64, 64).requires_grad_()
    calls = {"online": 0, "target": 0, "predictor": 0}

    def online_fields(self: Any, rgb: Any) -> Any:
        calls["online"] += 1
        return current_fields if calls["online"] == 1 else next_fields

    def target_fields(self: Any, rgb: Any) -> Any:
        calls["target"] += 1
        return torch.zeros(
            rgb.shape[0],
            2,
            64,
            64,
            dtype=rgb.dtype,
            device=rgb.device,
        )

    monkeypatch.setattr(
        model,
        "online_state_fields",
        MethodType(online_fields, model),
    )
    monkeypatch.setattr(
        model,
        "target_state_fields",
        MethodType(target_fields, model),
    )
    handle = model.predictor.register_forward_pre_hook(
        lambda *_: calls.__setitem__("predictor", calls["predictor"] + 1)
    )

    current_rgb = torch.zeros(batch, 3, 112, 112)
    next_rgb = torch.full_like(current_rgb, 0.10)
    fixed_negative_rgb = torch.full_like(current_rgb, -0.10).requires_grad_()
    executed = torch.tensor([0, 1, 2], dtype=torch.long)
    action = torch.nn.functional.one_hot(
        executed,
        num_classes=len(model.action_vocabulary),
    ).to(dtype=torch.float32)
    components = model.training_objective_with_components(
        current_rgb=current_rgb,
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed_negative_rgb,
        action_one_hot=action,
        non_hold_mask=executed != api.HOLD_ACTION_INDEX_V1,
        current_labels=current_labels,
        next_labels=next_labels,
    )
    handle.remove()
    assert calls == {"online": 2, "target": 3, "predictor": 0}
    assert isinstance(
        components,
        api.SignedBoundarySemanticAnchorObjectiveComponentsV1,
    )

    current_targets, _ = api.signed_boundary_distance_targets_v1(
        current_labels
    )
    next_targets, _ = api.signed_boundary_distance_targets_v1(next_labels)
    expected_distance_current = api._boundary_huber_per_row_v1(
        current_fields,
        current_targets,
        current_labels,
    ).mean()
    expected_distance_next = api._boundary_huber_per_row_v1(
        next_fields,
        next_targets,
        next_labels,
    ).mean()
    expected_distance = (
        0.5 * expected_distance_current + 0.5 * expected_distance_next
    )

    current_logits = api.hierarchical_class_log_probabilities_v1(
        current_fields
    )
    next_logits = api.hierarchical_class_log_probabilities_v1(next_fields)
    expected_semantic_rows_current = _slow_present_class_macro_nll(
        torch,
        current_logits,
        current_labels,
    )
    expected_semantic_rows_next = _slow_present_class_macro_nll(
        torch,
        next_logits,
        next_labels,
    )
    inherited_current = api._final_class_macro_nll_per_row_v10(
        current_logits,
        current_labels,
    )
    inherited_next = api._final_class_macro_nll_per_row_v10(
        next_logits,
        next_labels,
    )
    assert torch.equal(inherited_current, expected_semantic_rows_current)
    assert torch.equal(inherited_next, expected_semantic_rows_next)
    # A one-class raster remains a one-class mean; absent classes contribute
    # neither a zero nor a denominator slot.
    assert torch.equal(
        inherited_current[0],
        (-torch.log_softmax(current_logits[0], dim=0)[0]).mean(),
    )
    expected_semantic_current = expected_semantic_rows_current.mean()
    expected_semantic_next = expected_semantic_rows_next.mean()
    expected_semantic = (
        0.5 * expected_semantic_current + 0.5 * expected_semantic_next
    )
    expected_combined = expected_distance + expected_semantic / 64.0

    assert torch.equal(
        components.G_distance_current,
        expected_distance_current,
    )
    assert torch.equal(components.G_distance_next, expected_distance_next)
    assert torch.equal(components.G_distance, expected_distance)
    assert torch.equal(
        components.G_semantic_current,
        expected_semantic_current,
    )
    assert torch.equal(components.G_semantic_next, expected_semantic_next)
    assert torch.equal(components.G_semantic_macro_nll, expected_semantic)
    assert torch.equal(components.G_combined, expected_combined)
    assert components.objective.G is components.G_combined
    assert components.objective.total is components.G_combined
    assert torch.equal(
        components.objective.G_current,
        expected_distance_current + expected_semantic_current / 64.0,
    )
    assert torch.equal(
        components.objective.G_next,
        expected_distance_next + expected_semantic_next / 64.0,
    )
    assert torch.autograd.grad(
        components.G_combined,
        fixed_negative_rgb,
        allow_unused=True,
    )[0] is None


def test_D_A_and_combined_each_reach_encoder_decoder_and_both_head_rows() -> None:
    distance_api = _load(DISTANCE_MODEL, "_semantic_anchor_distance_grad_base")
    api = _load(MODEL, "_semantic_anchor_registered_gradients")
    torch = api.torch
    model = api.DirectEgocentricBevStateJepaV1(
        _encoder_state(distance_api, retain_values=True)
    )
    model.eval()
    model.arm_phase_schedule_v6()

    calls = {
        "online_encoder": 0,
        "online_decoder": 0,
        "online_head": 0,
        "target_encoder": 0,
        "target_decoder": 0,
        "target_head": 0,
        "predictor": 0,
    }
    handles = []
    online_forward_tokens = model.encoder.forward_tokens
    target_forward_tokens = model.target_encoder.forward_tokens

    def counted_online_forward_tokens(self: Any, rgb: Any) -> Any:
        calls["online_encoder"] += 1
        return online_forward_tokens(rgb)

    def counted_target_forward_tokens(self: Any, rgb: Any) -> Any:
        calls["target_encoder"] += 1
        return target_forward_tokens(rgb)

    model.encoder.forward_tokens = MethodType(
        counted_online_forward_tokens,
        model.encoder,
    )
    model.target_encoder.forward_tokens = MethodType(
        counted_target_forward_tokens,
        model.target_encoder,
    )
    for field, module in (
        ("online_decoder", model.bev_decoder),
        ("online_head", model.state_head),
        ("target_decoder", model.target_bev_decoder),
        ("target_head", model.target_state_head),
        ("predictor", model.predictor),
    ):
        handles.append(
            module.register_forward_pre_hook(
                lambda *_, field=field: calls.__setitem__(
                    field,
                    calls[field] + 1,
                )
            )
        )

    generator = torch.Generator().manual_seed(20260727)
    current_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    next_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    fixed_negative_rgb = torch.rand(
        1,
        3,
        112,
        112,
        generator=generator,
    ).requires_grad_()
    labels = torch.arange(64 * 64).reshape(1, 64, 64).remainder(3)
    next_labels = labels.roll(1, dims=2)
    executed = torch.tensor([0], dtype=torch.long)
    components = model.training_objective_with_components(
        current_rgb=current_rgb,
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed_negative_rgb,
        action_one_hot=torch.nn.functional.one_hot(
            executed,
            num_classes=len(model.action_vocabulary),
        ).to(dtype=torch.float32),
        non_hold_mask=executed != api.HOLD_ACTION_INDEX_V1,
        current_labels=labels,
        next_labels=next_labels,
    )
    for handle in handles:
        handle.remove()
    assert calls == {
        "online_encoder": 2,
        "online_decoder": 2,
        "online_head": 2,
        "target_encoder": 3,
        "target_decoder": 3,
        "target_head": 3,
        "predictor": 0,
    }

    named_parameters = [
        (f"encoder.{name}", parameter)
        for name, parameter in model.encoder.named_parameters()
    ]
    named_parameters += [
        (f"decoder.{name}", parameter)
        for name, parameter in model.bev_decoder.named_parameters()
    ]
    named_parameters += [
        (f"head.{name}", parameter)
        for name, parameter in model.state_head.named_parameters()
    ]
    parameters = [parameter for _, parameter in named_parameters]

    for name, scalar in (
        ("D", components.G_distance),
        ("A", components.G_semantic_macro_nll),
        ("combined", components.G_combined),
    ):
        gradients = torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        by_name = dict(zip((key for key, _ in named_parameters), gradients))
        for prefix in ("encoder.", "decoder.", "head."):
            group = [
                gradient
                for key, gradient in by_name.items()
                if key.startswith(prefix) and gradient is not None
            ]
            assert group, f"{name} has no {prefix[:-1]} gradients"
            assert all(bool(torch.isfinite(value).all()) for value in group)
            assert sum(float(value.abs().sum()) for value in group) > 0.0
        weight = by_name["head.projection.weight"]
        bias = by_name["head.projection.bias"]
        assert weight is not None and bias is not None
        for row in (0, 1):
            row_mass = weight[row].abs().sum() + bias[row].abs()
            assert bool(torch.isfinite(row_mass))
            assert float(row_mass) > 0.0, f"{name} missed K/O row {row}"

    assert torch.autograd.grad(
        components.G_combined,
        fixed_negative_rgb,
        retain_graph=True,
        allow_unused=True,
    )[0] is None
    assert all(
        not parameter.requires_grad
        for module in model._target_modules()
        for parameter in module.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.predictor.parameters())


def test_preliminary_dispatch_is_nonauthoritative_and_final_markers_are_atomic() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_dispatch_contract")
    for update in contract.OBSERVATION_UPDATES:
        preliminary = contract.evaluate_gate(update, {})
        assert preliminary["passed"] is True
        assert preliminary["scientific_gate_evidence"] is False
        assert preliminary["control"] == contract.CONTROL_PRELIMINARY
        assert preliminary["final_gate_evaluated"] is False

        for marker in contract.FINAL_MARKERS:
            with pytest.raises(ValueError, match="partial"):
                contract.evaluate_gate(update, {marker: True})

    zero = _update_zero_metrics(contract)
    structural_only = {
        key: value
        for key, value in zero.items()
        if key
        not in {
            "aggregate_raster_nll",
            "aggregate_raster_balanced_accuracy",
            "paired_rgb_aggregate_margin",
        }
    }
    assert "correct_rgb_scene_win_count" not in structural_only
    final = contract.evaluate_gate(0, structural_only)
    assert final["passed"] is True
    assert final["control"] == contract.CONTROL_CONTINUE_UPDATE_0
    assert final["scientific_gate_evidence"] is True


def test_common_objective_accounting_integrity_and_stop_edges_at_all_gates() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_common_gate_edges")
    zero = _update_zero_metrics(contract)
    hundred = _update_100_metrics(contract)
    four_hundred = _update_400_metrics(contract)
    thousand = _update_1000_metrics(contract)
    metrics_by_update = {
        0: zero,
        100: hundred,
        400: four_hundred,
        1_000: thousand,
    }
    expected_pass_controls = {
        0: contract.CONTROL_CONTINUE_UPDATE_0,
        100: contract.CONTROL_CONTINUE_UPDATE_100,
        400: contract.CONTROL_CONTINUE_UPDATE_400,
        1_000: contract.CONTROL_PASS,
    }
    expected_fail_controls = {
        0: contract.CONTROL_UPDATE_0_FAIL,
        100: contract.CONTROL_UPDATE_100_FAIL,
        400: contract.CONTROL_UPDATE_400_FAIL,
        1_000: contract.CONTROL_UPDATE_1000_FAIL,
    }

    for update, metrics in metrics_by_update.items():
        observed = _evaluate(
            contract,
            update,
            metrics,
            zero=zero,
            hundred=hundred,
            four_hundred=four_hundred,
        )
        assert observed["passed"] is True, observed
        assert observed["control"] == expected_pass_controls[update]

        for field in contract.perception_accounting(update):
            failed = _evaluate(
                contract,
                update,
                {**metrics, field: metrics[field] + 1},
                zero=zero,
                hundred=hundred,
                four_hundred=four_hundred,
            )
            assert failed["passed"] is False, (update, field)
            assert failed["control"] == expected_fail_controls[update]

        for field in (
            "all_registered_values_finite",
            "state_nonconstant",
            "all_forbidden_access_counts_zero",
        ):
            failed = _evaluate(
                contract,
                update,
                {**metrics, field: False},
                zero=zero,
                hundred=hundred,
                four_hundred=four_hundred,
            )
            assert failed["passed"] is False, (update, field)

        failed_prior = _evaluate(
            contract,
            update,
            metrics,
            zero=zero,
            hundred=hundred,
            four_hundred=four_hundred,
            prior_gates_passed=False,
        )
        assert failed_prior["passed"] is False

        broken_identity = {
            **metrics,
            "G": math.nextafter(metrics["G"], math.inf),
        }
        failed_identity = _evaluate(
            contract,
            update,
            broken_identity,
            zero=zero,
            hundred=hundred,
            four_hundred=four_hundred,
        )
        assert failed_identity["passed"] is False

    for field in contract.INTEGRITY_FIELDS:
        failed = contract.evaluate_gate(0, {**zero, field: False})
        assert failed["passed"] is False, field
        assert failed["control"] == contract.CONTROL_UPDATE_0_FAIL
    for field, value in (
        ("initial_online_to_target_hard_sync_count", 2),
        ("signed_boundary_semantic_anchor_mechanism_receipt_ready", False),
        ("active_training_scope_signed_boundary_semantic_anchor_v1", "wrong"),
    ):
        failed = contract.evaluate_gate(0, {**zero, field: value})
        assert failed["passed"] is False, field


def test_update_100_all_strict_and_inclusive_one_step_edges() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_update100_edges")
    zero = _update_zero_metrics(contract)
    hundred = _update_100_metrics(contract)
    four_hundred = _update_400_metrics(contract)
    passed = contract.evaluate_gate(100, hundred, update_zero=zero)
    assert passed["passed"] is True
    assert passed["control"] == contract.CONTROL_CONTINUE_UPDATE_100

    strict_component_failures = (
        _replace_objective(hundred, distance=zero["G_distance"]),
        _replace_objective(
            hundred,
            semantic=zero["G_semantic_macro_nll"],
        ),
    )
    for mutation in strict_component_failures:
        failed = contract.evaluate_gate(100, mutation, update_zero=zero)
        assert failed["passed"] is False
        assert failed["control"] == contract.CONTROL_UPDATE_100_FAIL

    # Make both components strictly better while holding the prior combined
    # reference at exact equality, isolating the strict combined comparator.
    combined_reference = {
        **zero,
        "G_distance": hundred["G_distance"] + 1.0,
        "G_semantic_macro_nll": hundred["G_semantic_macro_nll"] + 1.0,
        "G": hundred["G"],
    }
    assert not contract.evaluate_gate(
        100,
        hundred,
        update_zero=combined_reference,
    )["passed"]

    equality_and_failure = (
        (
            "aggregate_raster_nll",
            zero["aggregate_raster_nll"] - 0.15,
            math.nextafter(zero["aggregate_raster_nll"] - 0.15, math.inf),
        ),
        (
            "aggregate_raster_balanced_accuracy",
            max(0.68, zero["aggregate_raster_balanced_accuracy"] + 0.08),
            math.nextafter(
                max(0.68, zero["aggregate_raster_balanced_accuracy"] + 0.08),
                -math.inf,
            ),
        ),
        ("aggregate_free_recall", 0.60, math.nextafter(0.60, -math.inf)),
        (
            "aggregate_occupied_recall",
            0.30,
            math.nextafter(0.30, -math.inf),
        ),
        ("correct_rgb_scene_win_count", 6, 5),
    )
    for field, equality, beyond in equality_and_failure:
        assert contract.evaluate_gate(
            100,
            {**hundred, field: equality},
            update_zero=zero,
        )["passed"], field
        failed = contract.evaluate_gate(
            100,
            {**hundred, field: beyond},
            update_zero=zero,
        )
        assert failed["passed"] is False, field

    gap_edge = {
        **hundred,
        "aggregate_occupied_recall": 0.30,
        "aggregate_free_recall": 0.30 + 0.50,
    }
    assert contract.evaluate_gate(100, gap_edge, update_zero=zero)["passed"]
    assert not contract.evaluate_gate(
        100,
        {
            **gap_edge,
            "aggregate_free_recall": math.nextafter(
                gap_edge["aggregate_free_recall"],
                math.inf,
            ),
        },
        update_zero=zero,
    )["passed"]

    margin_edge = {
        **hundred,
        "paired_rgb_aggregate_margin": math.nextafter(
            zero["paired_rgb_aggregate_margin"],
            math.inf,
        ),
    }
    assert contract.evaluate_gate(100, margin_edge, update_zero=zero)["passed"]
    assert not contract.evaluate_gate(
        100,
        {
            **margin_edge,
            "paired_rgb_aggregate_margin": zero["paired_rgb_aggregate_margin"],
        },
        update_zero=zero,
    )["passed"]


def test_update_400_all_absolute_relative_and_one_step_edges() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_update400_edges")
    zero = _update_zero_metrics(contract)
    hundred = _update_100_metrics(contract)
    four_hundred = _update_400_metrics(contract)
    passed = contract.evaluate_gate(
        400,
        four_hundred,
        update_100=hundred,
    )
    assert passed["passed"] is True, passed
    assert passed["control"] == contract.CONTROL_CONTINUE_UPDATE_400

    for mutation in (
        _replace_objective(
            four_hundred,
            distance=hundred["G_distance"],
        ),
        _replace_objective(
            four_hundred,
            semantic=hundred["G_semantic_macro_nll"],
        ),
    ):
        failed = contract.evaluate_gate(400, mutation, update_100=hundred)
        assert failed["passed"] is False
        assert failed["control"] == contract.CONTROL_UPDATE_400_FAIL
    combined_reference = {
        **hundred,
        "G_distance": four_hundred["G_distance"] + 1.0,
        "G_semantic_macro_nll": (
            four_hundred["G_semantic_macro_nll"] + 1.0
        ),
        "G": four_hundred["G"],
    }
    assert not contract.evaluate_gate(
        400,
        four_hundred,
        update_100=combined_reference,
    )["passed"]

    absolute_edges = (
        ("aggregate_raster_nll", 0.55, math.nextafter(0.55, math.inf)),
        (
            "aggregate_raster_balanced_accuracy",
            0.72,
            math.nextafter(0.72, -math.inf),
        ),
        ("aggregate_free_recall", 0.65, math.nextafter(0.65, -math.inf)),
        (
            "aggregate_occupied_recall",
            0.55,
            math.nextafter(0.55, -math.inf),
        ),
        (
            "rough_raster_balanced_accuracy",
            0.65,
            math.nextafter(0.65, -math.inf),
        ),
        (
            "rough_raster_occupied_recall",
            0.50,
            math.nextafter(0.50, -math.inf),
        ),
        ("correct_rgb_scene_win_count", 7, 6),
    )
    absolute_reference = {
        **hundred,
        "aggregate_raster_nll": 0.80,
        "aggregate_raster_balanced_accuracy": 0.60,
        "aggregate_occupied_recall": 0.40,
        "rough_raster_balanced_accuracy": 0.50,
        "rough_raster_occupied_recall": 0.40,
    }
    for field, equality, beyond in absolute_edges:
        candidate = {**four_hundred, field: equality}
        # Keep free/occupied balance valid while exercising each recall floor.
        if field == "aggregate_occupied_recall":
            candidate["aggregate_free_recall"] = 0.65
        assert contract.evaluate_gate(
            400,
            candidate,
            update_100=absolute_reference,
        )["passed"], field
        failed = contract.evaluate_gate(
            400,
            {**candidate, field: beyond},
            update_100=absolute_reference,
        )
        assert failed["passed"] is False, field

    relative_reference = {
        **hundred,
        "aggregate_raster_nll": 0.40,
        "aggregate_raster_balanced_accuracy": 0.90,
        "aggregate_occupied_recall": 0.70,
        "rough_raster_balanced_accuracy": 0.70,
        "rough_raster_occupied_recall": 0.60,
    }
    relative_candidate = {
        **four_hundred,
        "aggregate_raster_nll": 0.40,
        "aggregate_raster_balanced_accuracy": 0.89,
        "aggregate_free_recall": 0.90,
        "aggregate_occupied_recall": 0.70,
        "rough_raster_balanced_accuracy": 0.72,
        "rough_raster_occupied_recall": 0.65,
    }
    assert contract.evaluate_gate(
        400,
        relative_candidate,
        update_100=relative_reference,
    )["passed"]
    for field, direction in (
        ("aggregate_raster_nll", math.inf),
        ("aggregate_raster_balanced_accuracy", -math.inf),
        ("aggregate_occupied_recall", -math.inf),
        ("rough_raster_balanced_accuracy", -math.inf),
        ("rough_raster_occupied_recall", -math.inf),
    ):
        failed = contract.evaluate_gate(
            400,
            {
                **relative_candidate,
                field: math.nextafter(relative_candidate[field], direction),
            },
            update_100=relative_reference,
        )
        assert failed["passed"] is False, field

    gap_edge = {
        **four_hundred,
        "aggregate_occupied_recall": 0.55,
        "aggregate_free_recall": 0.55 + 0.35,
    }
    assert contract.evaluate_gate(400, gap_edge, update_100=hundred)["passed"]
    assert not contract.evaluate_gate(
        400,
        {
            **gap_edge,
            "aggregate_free_recall": math.nextafter(
                gap_edge["aggregate_free_recall"],
                math.inf,
            ),
        },
        update_100=hundred,
    )["passed"]
    assert contract.evaluate_gate(
        400,
        {
            **four_hundred,
            "paired_rgb_aggregate_margin": math.nextafter(0.0, math.inf),
        },
        update_100=hundred,
    )["passed"]
    assert not contract.evaluate_gate(
        400,
        {**four_hundred, "paired_rgb_aggregate_margin": 0.0},
        update_100=hundred,
    )["passed"]


def test_update_1000_all_nonincreasing_absolute_relative_and_one_step_edges() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_update1000_edges")
    zero = _update_zero_metrics(contract)
    hundred = _update_100_metrics(contract)
    four_hundred = _update_400_metrics(contract)
    thousand = _update_1000_metrics(contract)
    passed = contract.evaluate_gate(
        1_000,
        thousand,
        update_400=four_hundred,
    )
    assert passed["passed"] is True, passed
    assert passed["control"] == contract.CONTROL_PASS

    assert thousand["G_distance"] == four_hundred["G_distance"]
    assert (
        thousand["G_semantic_macro_nll"]
        == four_hundred["G_semantic_macro_nll"]
    )
    assert thousand["G"] == four_hundred["G"]
    for mutation in (
        _replace_objective(
            thousand,
            distance=math.nextafter(
                four_hundred["G_distance"],
                math.inf,
            ),
        ),
        _replace_objective(
            thousand,
            semantic=math.nextafter(
                four_hundred["G_semantic_macro_nll"],
                math.inf,
            ),
        ),
        _replace_objective(
            thousand,
            combined=math.nextafter(four_hundred["G"], math.inf),
        ),
    ):
        failed = contract.evaluate_gate(
            1_000,
            mutation,
            update_400=four_hundred,
        )
        assert failed["passed"] is False
        assert failed["control"] == contract.CONTROL_UPDATE_1000_FAIL

    absolute_edges = (
        ("aggregate_raster_nll", 0.42, math.nextafter(0.42, math.inf)),
        (
            "aggregate_raster_balanced_accuracy",
            0.80,
            math.nextafter(0.80, -math.inf),
        ),
        ("aggregate_unknown_recall", 0.80, math.nextafter(0.80, -math.inf)),
        ("aggregate_free_recall", 0.68, math.nextafter(0.68, -math.inf)),
        (
            "aggregate_occupied_recall",
            0.88,
            math.nextafter(0.88, -math.inf),
        ),
        (
            "rough_raster_balanced_accuracy",
            0.772,
            math.nextafter(0.772, -math.inf),
        ),
        (
            "rough_raster_occupied_recall",
            0.65,
            math.nextafter(0.65, -math.inf),
        ),
        ("correct_rgb_scene_win_count", 7, 6),
    )
    absolute_reference = {
        **four_hundred,
        "aggregate_raster_nll": 0.80,
        "aggregate_raster_balanced_accuracy": 0.70,
    }
    for field, equality, beyond in absolute_edges:
        candidate = {**thousand, field: equality}
        assert contract.evaluate_gate(
            1_000,
            candidate,
            update_400=absolute_reference,
        )["passed"], field
        assert not contract.evaluate_gate(
            1_000,
            {**candidate, field: beyond},
            update_400=absolute_reference,
        )["passed"], field

    relative_reference = {
        **four_hundred,
        "aggregate_raster_nll": 0.30,
        "aggregate_raster_balanced_accuracy": 0.85,
    }
    relative_candidate = {
        **thousand,
        "aggregate_raster_nll": 0.30,
        "aggregate_raster_balanced_accuracy": 0.85,
    }
    assert contract.evaluate_gate(
        1_000,
        relative_candidate,
        update_400=relative_reference,
    )["passed"]
    for field, direction in (
        ("aggregate_raster_nll", math.inf),
        ("aggregate_raster_balanced_accuracy", -math.inf),
    ):
        assert not contract.evaluate_gate(
            1_000,
            {
                **relative_candidate,
                field: math.nextafter(relative_candidate[field], direction),
            },
            update_400=relative_reference,
        )["passed"], field

    gap_edge = {
        **thousand,
        "aggregate_occupied_recall": 1.0,
        "aggregate_free_recall": 1.0 - 0.25,
    }
    assert contract.evaluate_gate(
        1_000,
        gap_edge,
        update_400=four_hundred,
    )["passed"]
    assert not contract.evaluate_gate(
        1_000,
        {
            **gap_edge,
            "aggregate_free_recall": math.nextafter(
                gap_edge["aggregate_free_recall"],
                -math.inf,
            ),
        },
        update_400=four_hundred,
    )["passed"]
    assert contract.evaluate_gate(
        1_000,
        {
            **thousand,
            "paired_rgb_aggregate_margin": math.nextafter(0.0, math.inf),
        },
        update_400=four_hundred,
    )["passed"]
    assert not contract.evaluate_gate(
        1_000,
        {**thousand, "paired_rgb_aggregate_margin": 0.0},
        update_400=four_hundred,
    )["passed"]
