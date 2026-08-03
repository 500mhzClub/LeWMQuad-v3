from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_matched_branch_successor_screen_v1 import (
    CompactRSSMPredictorV1,
    DenseActionConditionedPredictorV1,
    DeterministicStateSpacePredictorV1,
    diagonal_gaussian_kl_v1,
)


_FEATURE_DIM = 12
_HIDDEN_DIM = 16


def _inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.randn(batch, 3, 256, _FEATURE_DIM, dtype=torch.float32)
    history = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)[:batch]
    candidate = torch.tensor([4, 5], dtype=torch.long)[:batch]
    return context, history, candidate


def _with_nonfinite(
    context: torch.Tensor,
    history: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    changed = context.clone()
    changed[0, 0, 0, 0] = float("nan")
    return changed, history, candidate


@pytest.mark.parametrize(
    "model_class",
    (
        DenseActionConditionedPredictorV1,
        DeterministicStateSpacePredictorV1,
    ),
)
def test_deterministic_predictors_have_common_shape_and_candidate_sensitivity(
    model_class: type[torch.nn.Module],
) -> None:
    torch.manual_seed(11)
    model = model_class(feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM)
    context, history, candidate = _inputs()

    prediction = model(context, history, candidate)
    changed = model(context, history, (candidate + 1) % 9)

    assert prediction.shape == (2, 256, _FEATURE_DIM)
    assert torch.isfinite(prediction).all()
    assert not torch.allclose(prediction, changed)


def test_rssm_prior_inference_is_deterministic_and_action_sensitive() -> None:
    torch.manual_seed(12)
    model = CompactRSSMPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        stochastic_dim=7,
    )
    context, history, candidate = _inputs()

    first = model(context, history, candidate)
    repeated = model(context, history, candidate)
    changed = model(context, history, (candidate + 1) % 9)

    assert first.shape == (2, 256, _FEATURE_DIM)
    assert torch.equal(first, repeated)
    assert not torch.allclose(first, changed)


def test_rssm_training_posterior_and_kl_are_finite() -> None:
    torch.manual_seed(13)
    model = CompactRSSMPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        stochastic_dim=7,
    )
    context, history, candidate = _inputs()
    target = torch.randn(2, 256, _FEATURE_DIM)

    result = model.training_output(
        context,
        history,
        candidate,
        target,
        sample_posterior=False,
    )
    kl = model.kl_divergence(result, reduction="none")

    assert result.posterior_mean is not None
    assert result.posterior_log_std is not None
    assert result.posterior_mean.shape == (2, 7)
    assert torch.equal(result.latent_state, result.posterior_mean)
    assert kl.shape == (2,)
    assert torch.isfinite(kl).all()
    assert bool((kl >= -1.0e-6).all())
    assert torch.allclose(
        diagonal_gaussian_kl_v1(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            reduction="none",
        ),
        torch.zeros(2),
    )


@pytest.mark.parametrize(
    "model",
    (
        DenseActionConditionedPredictorV1(
            feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM
        ),
        DeterministicStateSpacePredictorV1(
            feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM
        ),
    ),
)
def test_deterministic_predictors_have_finite_backward_gradients(
    model: torch.nn.Module,
) -> None:
    context, history, candidate = _inputs()
    target = torch.randn(2, 256, _FEATURE_DIM)

    F.mse_loss(model(context, history, candidate), target).backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_rssm_reconstruction_and_kl_have_finite_backward_gradients() -> None:
    torch.manual_seed(14)
    model = CompactRSSMPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        stochastic_dim=7,
    )
    context, history, candidate = _inputs()
    target = torch.randn(2, 256, _FEATURE_DIM)
    result = model.training_output(context, history, candidate, target)

    loss = F.mse_loss(result.prediction, target) + 0.01 * model.kl_divergence(result)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_zero_output_updates_recover_exact_last_frame_persistence() -> None:
    context, history, candidate = _inputs()
    dense = DenseActionConditionedPredictorV1(
        feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM
    )
    deterministic = DeterministicStateSpacePredictorV1(
        feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM
    )
    rssm = CompactRSSMPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        stochastic_dim=7,
    )
    projections = (
        dense.output_projection,
        deterministic.decoder.output_projection,
        rssm.decoder.output_projection,
    )
    for projection in projections:
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.zeros_(projection.bias)

    assert torch.equal(dense(context, history, candidate), context[:, -1])
    assert torch.equal(deterministic(context, history, candidate), context[:, -1])
    assert torch.equal(rssm(context, history, candidate), context[:, -1])
    target = torch.randn(2, 256, _FEATURE_DIM)
    posterior = rssm.training_output(
        context,
        history,
        candidate,
        target,
        sample_posterior=False,
    )
    assert torch.equal(posterior.prediction, context[:, -1])


@pytest.mark.parametrize(
    ("mutator", "error"),
    (
        (lambda c, h, a: (c[:, :, :-1], h, a), ValueError),
        (lambda c, h, a: (c.to(torch.float64), h, a), TypeError),
        (lambda c, h, a: (c, h.to(torch.int32), a), TypeError),
        (lambda c, h, a: (c, h, torch.full_like(a, 9)), ValueError),
        (_with_nonfinite, FloatingPointError),
    ),
)
def test_invalid_common_inputs_are_rejected(
    mutator: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    error: type[Exception],
) -> None:
    model = DenseActionConditionedPredictorV1(
        feature_dim=_FEATURE_DIM, hidden_dim=_HIDDEN_DIM
    )
    context, history, candidate = _inputs()
    bad_context, bad_history, bad_candidate = mutator(context, history, candidate)
    with pytest.raises(error):
        model(bad_context, bad_history, bad_candidate)


def test_rssm_rejects_invalid_target() -> None:
    model = CompactRSSMPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        stochastic_dim=7,
    )
    context, history, candidate = _inputs()
    with pytest.raises(ValueError, match="target"):
        model.training_output(
            context,
            history,
            candidate,
            torch.randn(2, 255, _FEATURE_DIM),
        )
