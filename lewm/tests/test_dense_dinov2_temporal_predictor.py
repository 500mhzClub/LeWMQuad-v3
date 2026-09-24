from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from lewm.models.dense_dinov2_temporal_predictor import (
    DenseDINOv2TemporalPredictorV1,
)


_FEATURE_DIM = 12
_ACTION_DIM = 5
_TOKEN_COUNT = 4


def _model() -> DenseDINOv2TemporalPredictorV1:
    torch.manual_seed(31)
    return DenseDINOv2TemporalPredictorV1(
        feature_dim=_FEATURE_DIM,
        action_dim=_ACTION_DIM,
        token_count=_TOKEN_COUNT,
        n_layers=2,
        n_heads=3,
        dim_head=4,
        mlp_dim=24,
    )


def _inputs(
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(32)
    context = F.normalize(
        torch.randn(batch, 3, _TOKEN_COUNT, _FEATURE_DIM), dim=-1
    )
    history = torch.randn(batch, 2, _ACTION_DIM)
    candidate = torch.randn(batch, _ACTION_DIM)
    return context, history, candidate


def _open_action_path(model: DenseDINOv2TemporalPredictorV1) -> None:
    """Open AdaLN-zero/output gates to test the trainable action path."""

    with torch.no_grad():
        torch.nn.init.eye_(model.output_projection.weight)
        model.output_projection.weight.mul_(0.05)
        model.output_projection.bias.zero_()
        for block in model.blocks:
            modulation = block.adaLN_modulation[-1]
            modulation.weight.normal_(mean=0.0, std=0.02)
            modulation.bias.zero_()
            width = model.feature_dim
            modulation.bias[2 * width : 3 * width].fill_(0.1)
            modulation.bias[5 * width : 6 * width].fill_(0.1)


def test_forward_shape_unit_norm_and_exact_initial_persistence() -> None:
    model = _model()
    context, history, candidate = _inputs()

    prediction = model(context, history, candidate)

    assert prediction.shape == (2, _TOKEN_COUNT, _FEATURE_DIM)
    assert torch.equal(prediction, F.normalize(context[:, -1], dim=-1))
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction, dim=-1),
        torch.ones(2, _TOKEN_COUNT),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_rollout_is_deterministic_and_matches_explicit_context_shift() -> None:
    model = _model()
    _open_action_path(model)
    model.eval()
    context, history, _ = _inputs()
    torch.manual_seed(33)
    actions = torch.randn(2, 3, _ACTION_DIM)

    rollout = model.rollout(context, history, actions)
    repeated = model.rollout(context, history, actions)

    manual_predictions = []
    rolling_context = context
    rolling_history = history
    for step in range(actions.shape[1]):
        prediction = model(rolling_context, rolling_history, actions[:, step])
        manual_predictions.append(prediction)
        rolling_context = torch.cat(
            (rolling_context[:, 1:], prediction.unsqueeze(1)), dim=1
        )
        rolling_history = torch.cat(
            (rolling_history[:, 1:], actions[:, step : step + 1]), dim=1
        )
    manual = torch.stack(manual_predictions, dim=1)

    assert rollout.shape == (2, 3, _TOKEN_COUNT, _FEATURE_DIM)
    assert torch.equal(rollout, repeated)
    assert torch.equal(rollout, manual)


def test_opened_model_is_action_sensitive_with_finite_action_gradients() -> None:
    model = _model()
    _open_action_path(model)
    context, history, candidate = _inputs()
    changed_candidate = candidate.clone()
    changed_candidate[:, 0].add_(1.0)

    prediction = model(context, history, candidate)
    changed = model(context, history, changed_candidate)
    assert not torch.allclose(prediction, changed)

    target = F.normalize(torch.randn_like(prediction), dim=-1)
    F.mse_loss(prediction, target).backward()
    action_gradients = [
        parameter.grad
        for parameter in model.action_embedder.parameters()
        if parameter.grad is not None
    ]
    assert action_gradients
    assert all(torch.isfinite(gradient).all() for gradient in action_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in action_gradients) > 0.0


def _nonfinite_candidate(
    context: torch.Tensor,
    history: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    changed = candidate.clone()
    changed[0, 0] = float("nan")
    return context, history, changed


@pytest.mark.parametrize(
    ("mutator", "error"),
    (
        (lambda c, h, a: (c[:, :, :-1], h, a), ValueError),
        (lambda c, h, a: (c.to(torch.float64), h, a), TypeError),
        (lambda c, h, a: (c, h[:, :-1], a), ValueError),
        (lambda c, h, a: (c, h.to(torch.float64), a), TypeError),
        (lambda c, h, a: (c, h, a[:, :-1]), ValueError),
        (_nonfinite_candidate, FloatingPointError),
    ),
)
def test_invalid_forward_inputs_are_rejected(
    mutator: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    error: type[Exception],
) -> None:
    model = _model()
    context, history, candidate = _inputs()
    bad_context, bad_history, bad_candidate = mutator(context, history, candidate)
    with pytest.raises(error):
        model(bad_context, bad_history, bad_candidate)


@pytest.mark.parametrize(
    "actions",
    (
        torch.empty(2, 0, _ACTION_DIM),
        torch.empty(2, 2, _ACTION_DIM - 1),
        torch.empty(2, 2, _ACTION_DIM, dtype=torch.float64),
    ),
)
def test_invalid_rollout_inputs_are_rejected(actions: torch.Tensor) -> None:
    model = _model()
    context, history, _ = _inputs()
    with pytest.raises((TypeError, ValueError)):
        model.rollout(context, history, actions)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"feature_dim": 0}, ValueError),
        ({"context_steps": 2}, ValueError),
        ({"n_layers": 0}, ValueError),
        ({"dropout": True}, TypeError),
        ({"dropout": 1.0}, ValueError),
    ),
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        DenseDINOv2TemporalPredictorV1(**kwargs)
