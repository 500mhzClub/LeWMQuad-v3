from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)
V21_FIXTURE_PATH = ROOT / (
    "lewm/tests/test_run_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_v25_training_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_v21_fixture():
    spec = importlib.util.spec_from_file_location(
        "_v21_training_fixture_for_v25", V21_FIXTURE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ControlledEnergyApi:
    @staticmethod
    def latent_energy_per_row(
        predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return predicted[:, 0, 0, 0] + 0.25 * target[:, 0, 0, 0]


class _DirectScores:
    @staticmethod
    def survival_scores_v1(logits: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(expected_progress_m=logits[..., 0])


def _temporal_inputs(
    prediction_energy: tuple[float, float, float, float],
    persistence_energy: tuple[float, float, float, float],
) -> tuple[torch.Tensor, ...]:
    executed = torch.tensor((0, 1, 2, 3), dtype=torch.int64)
    predicted = torch.zeros((4, 9, 64, 1, 1), dtype=torch.float32)
    current = torch.zeros((4, 64, 1, 1), dtype=torch.float32)
    next_ = torch.zeros((4, 64, 1, 1), dtype=torch.float32)
    for row, action in enumerate(executed.tolist()):
        predicted[row, action, 0, 0, 0] = prediction_energy[row]
        current[row, 0, 0, 0] = persistence_energy[row]
    return (
        predicted.requires_grad_(),
        executed,
        current.requires_grad_(),
        next_.requires_grad_(),
    )


def _objective_inputs() -> tuple[torch.Tensor, ...]:
    logits = torch.zeros((4, 9, 16), dtype=torch.float32, requires_grad=True)
    prefix = torch.tensor(
        (
            (0, 2, 4, 6, 8, 10, 7, 12, 14),
            (14, 12, 10, 8, 6, 4, 7, 2, 0),
            (2, 4, 6, 8, 10, 12, 7, 14, 0),
            (12, 10, 8, 6, 4, 2, 7, 0, 14),
        ),
        dtype=torch.int64,
    )
    negatives = torch.tensor((1, 0, 3, 2), dtype=torch.int64)
    prior = torch.full((9,), 0.55, dtype=torch.float32)
    return logits, prefix, negatives, prior


def test_private_adapter_binds_frozen_v24_without_public_import() -> None:
    module = _load_module()
    receipt = module.private_training_adapter_receipt_v25()
    assert receipt["base_training_file_sha256"] == (
        "0a149aadfc8f4f0860c4bdfd9fe330e96ab95cbbe556e5b2c16ef1e390e819c6"
    )
    assert receipt["base_training_byte_count"] == 34_726
    assert receipt["preregistration_commit"] == (
        "f00e20df3b429f9242516ac38f67fea587e04b22"
    )
    assert receipt["preregistration_file_sha256"] == (
        "b9ce16b251415c50cb643daad919699c32965e23ddcd77d22bb3b69334f8b299"
    )
    assert receipt["temporal_mechanism"] == (
        module.PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25
    )
    assert receipt["denominator_used"] is False
    assert receipt["legacy_global_ratio_diagnostic_only"] is True
    assert receipt["j24_delegated_bit_identical_to_v24"] is True
    assert receipt["j24_parameter_tensor_count"] == 96
    assert receipt["protected_predictor_core_parameter_tensor_count"] == 13
    assert receipt["public_base_loaded_by_adapter"] is False
    assert receipt["private_module_registered"] is False


def test_exact_per_row_equation_diagnostics_and_equal_energy_identity() -> None:
    module = _load_module()
    predicted, executed, current, next_ = _temporal_inputs(
        (0.1, 0.2, 0.4, 0.7), (0.3, 0.2, 0.1, 0.9)
    )
    terms = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, predicted, executed, current, next_
    )
    expected_prediction = torch.tensor((0.1, 0.2, 0.4, 0.7))
    expected_persistence = torch.tensor((0.3, 0.2, 0.1, 0.9))
    expected_gap = expected_prediction - expected_persistence
    expected_rows = F.softplus(
        expected_gap,
        beta=1.0,
        threshold=20.0,
    ) / math.log(2.0)
    assert torch.equal(terms.prediction_energy_per_row, expected_prediction)
    assert torch.equal(terms.persistence_energy_per_row, expected_persistence)
    assert torch.equal(terms.gap_per_row, expected_gap)
    assert torch.equal(terms.row_loss_per_row, expected_rows)
    assert torch.equal(terms.loss, expected_rows.mean())
    assert terms.legacy_global_ratio.item() == pytest.approx(
        expected_prediction.mean().item() / expected_persistence.mean().item()
    )
    for value in (
        terms.prediction_energy_per_row,
        terms.persistence_energy_per_row,
        terms.gap_per_row,
        terms.row_loss_per_row,
        terms.legacy_global_ratio,
    ):
        assert value.requires_grad is False

    equal_inputs = _temporal_inputs((0.4,) * 4, (0.4,) * 4)
    equal = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *equal_inputs
    )
    assert torch.equal(equal.row_loss_per_row, torch.ones(4))
    assert equal.loss.item() == 1.0


def test_piecewise_derivative_is_bounded_and_targets_are_detached() -> None:
    module = _load_module()
    gaps = torch.tensor((-2.0, 0.0, 2.0, 21.0), dtype=torch.float32)
    predicted, executed, current, next_ = _temporal_inputs(
        tuple(float(value) for value in gaps), (0.0, 0.0, 0.0, 0.0)
    )
    terms = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, predicted, executed, current, next_
    )
    gradient = torch.autograd.grad(
        terms.loss, (predicted, current, next_), allow_unused=True
    )
    selected = gradient[0][torch.arange(4), executed, 0, 0, 0]
    expected = torch.where(
        gaps <= 20.0,
        torch.sigmoid(gaps),
        torch.ones_like(gaps),
    ) / (4.0 * math.log(2.0))
    assert torch.allclose(selected, expected, rtol=1.0e-6, atol=1.0e-7)
    assert bool((selected > 0.0).all())
    assert float(selected.max()) <= 1.0 / (4.0 * math.log(2.0))
    assert gradient[1] is None
    assert gradient[2] is None
    mask = torch.ones_like(gradient[0], dtype=torch.bool)
    mask[torch.arange(4), executed, 0, 0, 0] = False
    assert torch.count_nonzero(gradient[0][mask]) == 0


def test_row_locality_and_permutation_invariance() -> None:
    module = _load_module()
    prediction_values = (0.2, 0.4, 0.6, 0.8)
    baseline_a = (0.1, 0.3, 0.5, 0.7)
    baseline_b = (9.0, 0.3, 0.5, 0.7)

    inputs_a = _temporal_inputs(prediction_values, baseline_a)
    terms_a = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *inputs_a
    )
    gradient_a = torch.autograd.grad(terms_a.loss, inputs_a[0])[0]
    inputs_b = _temporal_inputs(prediction_values, baseline_b)
    terms_b = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *inputs_b
    )
    gradient_b = torch.autograd.grad(terms_b.loss, inputs_b[0])[0]
    for row in (1, 2, 3):
        action = row
        assert torch.equal(
            gradient_a[row, action, 0, 0, 0],
            gradient_b[row, action, 0, 0, 0],
        )

    order = (2, 0, 3, 1)
    permuted_inputs = _temporal_inputs(
        tuple(prediction_values[index] for index in order),
        tuple(baseline_a[index] for index in order),
    )
    permuted = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *permuted_inputs
    )
    assert torch.equal(permuted.loss, terms_a.loss)
    assert torch.equal(
        permuted.row_loss_per_row,
        terms_a.row_loss_per_row[torch.tensor(order)],
    )


def test_legacy_ratio_is_detached_and_cannot_change_loss_or_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    first_inputs = _temporal_inputs((0.1, 0.2, 0.3, 0.4), (0.0,) * 4)
    first = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *first_inputs
    )
    first_gradient = torch.autograd.grad(first.loss, first_inputs[0])[0]
    assert first.legacy_global_ratio.item() > 100_000.0

    monkeypatch.setattr(module, "LEGACY_PERSISTENCE_BASELINE_MIN_V25", 100.0)
    second_inputs = _temporal_inputs((0.1, 0.2, 0.3, 0.4), (0.0,) * 4)
    second = module.per_row_persistence_contrastive_temporal_loss_v25(
        torch, _ControlledEnergyApi, *second_inputs
    )
    second_gradient = torch.autograd.grad(second.loss, second_inputs[0])[0]
    assert torch.equal(first.loss, second.loss)
    assert torch.equal(first_gradient, second_gradient)
    assert first.legacy_global_ratio.item() != second.legacy_global_ratio.item()
    assert first.legacy_global_ratio.requires_grad is False
    assert second.legacy_global_ratio.requires_grad is False


def test_j24_objective_and_subset_are_delegated_bit_identically() -> None:
    module = _load_module()
    assert module.predictor_core_protected_survival_objective_v25 is (
        module._v24.predictor_core_protected_survival_objective_v24
    )
    assert module.predictor_core_protected_survival_parameter_subset_v25 is (
        module._v24.predictor_core_protected_survival_parameter_subset_v24
    )
    logits, prefix, negatives, prior = _objective_inputs()
    expected = module._v24.predictor_core_protected_survival_objective_v24(
        torch, _DirectScores, logits, prefix, negatives, prior
    )
    actual = module.predictor_core_protected_survival_objective_v25(
        torch, _DirectScores, logits, prefix, negatives, prior
    )
    for name in actual.__dataclass_fields__:
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(expected_value, actual_value)
        else:
            assert expected_value == actual_value


def test_accounting_preserves_exact_v24_work() -> None:
    module = _load_module()
    value = module.JointTrainingAccountingV25(
        updates=2,
        presentations=32,
        microbatch_graphs=8,
        backward_calls=24,
        camera_route_grad_calls=8,
        joint_route_grad_calls=8,
        predictor_core_protected_survival_grad_calls=8,
        camera_frame_objectives=64,
        optimizer_steps=2,
        ema_steps=2,
        predictor_forwards=8,
        predictor_objectives=16,
        predictor_core_protected_survival_objectives=8,
    )
    module.validate_accounting_v25(value)


def test_one_cpu_update_uses_p25_once_and_preserves_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _load_v21_fixture()
    model = fixture._TinyModel()
    partition = fixture._partition(model)

    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current_logits, current_labels, next_logits, next_labels):
            current_loss = F.cross_entropy(
                current_logits, current_labels, reduction="none"
            ).mean(dim=(1, 2))
            next_loss = F.cross_entropy(
                next_logits, next_labels, reduction="none"
            ).mean(dim=(1, 2))
            return SimpleNamespace(
                loss=0.5 * (current_loss.mean() + next_loss.mean()) / math.log(3.0)
            )

        @staticmethod
        def latent_energy_per_row(predicted, target):
            return (predicted - target.detach()).square().mean(dim=(1, 2, 3))

        @staticmethod
        def microbatch_persistence_loss_v1(*_args):
            raise AssertionError("V25 called the rejected global-ratio objective")

    class _SurvivalApi:
        @staticmethod
        def survival_scores_v1(logits):
            probabilities = torch.sigmoid(logits)
            survival = probabilities[..., :1] * probabilities[..., 1:].cumprod(
                dim=-1
            )
            return SimpleNamespace(expected_progress_m=0.1 * survival.sum(dim=-1))

        @staticmethod
        def joint_survival_loss_v1(**values):
            semantic = values["semantic_loss"]
            temporal = values["executed_action_ema_latent_loss"]
            survival = 0.05 * values["survival_logits"].square().mean()
            ranking = 0.02 * values["survival_logits"].square().mean()
            return SimpleNamespace(
                loss=semantic + temporal + survival + ranking,
                semantic=semantic,
                executed_action_ema_latent=temporal,
                survival=survival,
                progress_ranking=ranking,
                ranking_terms=SimpleNamespace(eligible_pair_count=torch.tensor(3)),
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9)
                ),
            )

    monkeypatch.setattr(
        module._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(module._v24, "_validate_microbatches_v24", lambda *_: None)
    monkeypatch.setattr(module._v24, "partition_parameters_v24", lambda _: partition)
    monkeypatch.setattr(module._v24, "validate_optimizer_v24", lambda *_: None)
    monkeypatch.setattr(
        module._tensor_core._v3,
        "occupied_safety_aux_loss_v3",
        lambda current_value, _a, next_value, _b: SimpleNamespace(
            loss=0.01 * (current_value.square().mean() + next_value.square().mean())
        ),
    )
    monkeypatch.setattr(
        module._tensor_core._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (prediction.predicted_latents, prediction.survival_logits),
    )
    monkeypatch.setattr(
        module._base,
        "camera_evidence_pair_loss_v13",
        lambda current_value, next_value, *_: SimpleNamespace(
            total=current_value.square() + next_value.square()
        ),
    )

    auxiliary_parameters = (model.shared, *tuple(model.survival))
    subset = module._v24.PredictorCoreProtectedSurvivalParameterSubsetV24(
        parameters=auxiliary_parameters,
        names=("encoder.synthetic", *fixture.SURVIVAL_NAMES),
        parameter_count=sum(value.numel() for value in auxiliary_parameters),
        protected_predictor_core_parameters=tuple(model.transition),
        protected_predictor_core_names=fixture.TRANSITION_NAMES,
        protected_predictor_core_parameter_count=sum(
            value.numel() for value in model.transition
        ),
    )
    monkeypatch.setattr(
        module._v24,
        "predictor_core_protected_survival_parameter_subset_v24",
        lambda _: subset,
    )

    prefix = _objective_inputs()[1]
    microbatches = []
    for batch in fixture._microbatches():
        value = dict(batch)
        value[module.PREFIX_LENGTHS_KEY] = prefix.clone()
        value[module.ACTION_PRIOR_M_KEY_V23] = torch.full(
            (9,), 0.55, dtype=torch.float32
        )
        microbatches.append(value)

    captured_applied: dict[int, torch.Tensor] = {}

    class _CapturingSgd(fixture._CountingSgd):
        def step(self, closure: Any = None) -> Any:
            for parameter in partition.online:
                if parameter.grad is not None:
                    captured_applied[id(parameter)] = parameter.grad.detach().clone()
            return super().step(closure)

    optimizer = _CapturingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    calls: list[tuple[tuple[int, ...], tuple[torch.Tensor | None, ...]]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        result = original_grad(*args, **kwargs)
        calls.append(
            (
                tuple(id(value) for value in args[1]),
                tuple(
                    None if value is None else value.detach().clone()
                    for value in result
                ),
            )
        )
        return result

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    result = module.joint_training_update_v25(
        model, optimizer, tuple(microbatches)
    )

    assert len(calls) == 12
    assert [len(names) for names, _ in calls] == [1, 17, 3] * 4
    assert all(
        names == tuple(id(value) for value in auxiliary_parameters)
        for names, _ in calls[2::3]
    )
    predictor_scale = result.gradient_routes["predictor"].applied_scale
    auxiliary_scale = result.gradient_routes[
        module.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
    ].applied_scale
    predictor_ids = tuple(id(value) for value in partition.predictor)
    inherited_sums = {
        parameter_id: torch.zeros_like(parameter)
        for parameter_id, parameter in zip(
            predictor_ids, partition.predictor, strict=True
        )
    }
    for names, gradients in calls[1::3]:
        for parameter_id, gradient in zip(names, gradients, strict=True):
            if parameter_id in inherited_sums:
                assert gradient is not None
                inherited_sums[parameter_id].add_(gradient)
    auxiliary_sums = {
        id(parameter): torch.zeros_like(parameter) for parameter in auxiliary_parameters
    }
    for names, gradients in calls[2::3]:
        for parameter_id, gradient in zip(names, gradients, strict=True):
            assert gradient is not None
            auxiliary_sums[parameter_id].add_(gradient)

    for parameter in model.transition:
        expected = predictor_scale * inherited_sums[id(parameter)]
        assert torch.equal(captured_applied[id(parameter)], expected)
        assert id(parameter) not in auxiliary_sums
    for parameter in model.survival:
        expected = (
            predictor_scale * inherited_sums[id(parameter)]
            + auxiliary_scale * auxiliary_sums[id(parameter)]
        )
        assert torch.equal(captured_applied[id(parameter)], expected)

    diagnostics = result.per_row_persistence_contrastive_diagnostics
    assert diagnostics["mechanism"] == (
        module.PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25
    )
    assert diagnostics["denominator_used"] is False
    assert diagnostics["prediction_energy_count"] == 16
    assert diagnostics["persistence_energy_count"] == 16
    assert diagnostics["gap_count"] == 16
    assert diagnostics["row_loss_count"] == 16
    assert diagnostics["legacy_global_ratio_count"] == 4
    assert len(diagnostics["prediction_energy_per_row"]) == 16
    assert len(diagnostics["legacy_global_ratio_per_microbatch"]) == 4
    assert result.mean_losses["P"] == pytest.approx(
        diagnostics["row_loss_mean"], rel=1.0e-6
    )
    assert result.mean_losses["L"] == pytest.approx(
        result.mean_losses["N"]
        + result.mean_losses["C"]
        + result.mean_losses["J24"],
        rel=2.0e-6,
    )
    assert result.mean_losses["J24"] == pytest.approx(
        result.mean_losses["F"] + result.mean_losses["J_rank"], rel=2.0e-6
    )
    assert model.predictor_forward_count == 4
    assert optimizer.step_calls == 1
    assert int(model.ema_update_count.item()) == 1
    assert result.target_gradient_tensor_count == 0
    assert model.target.grad is None
    assert result.accounting == module.JointTrainingAccountingV25(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=12,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        predictor_core_protected_survival_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=8,
        predictor_core_protected_survival_objectives=4,
    )
