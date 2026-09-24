from __future__ import annotations

import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.go2_rgb_full_whitened_predictive_state_h4_jepa_v1 import (
    FullWhitenedPredictiveStateConfig,
    FullWhitenedPredictiveStateH4JEPA,
    _covariance_identity_loss,
    _cross_covariance_identity_loss,
)
from lewm.models.go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 import (
    _mean_loss,
)


def _config() -> FullWhitenedPredictiveStateConfig:
    return FullWhitenedPredictiveStateConfig(
        image_size=8,
        patch_size=4,
        feature_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        recurrent_spatial_heads=3,
        cross_attention_heads=3,
    )


def _model() -> FullWhitenedPredictiveStateH4JEPA:
    torch.manual_seed(81)
    config = _config()
    encoder = VisionEncoder(
        image_size=8,
        patch_size=4,
        hidden_dim=12,
        depth=1,
        n_heads=3,
        mlp_ratio=4,
        dropout=0.0,
    )
    return FullWhitenedPredictiveStateH4JEPA(
        encoder.state_dict(), config=config
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(82)
    history = torch.randn(16, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(16, 4, 3, 8, 8, generator=generator)
    past = torch.randint(0, 9, (16, 2), generator=generator)
    future = torch.randint(0, 9, (16, 4), generator=generator)
    return history, past, future, future_rgb


def test_covariance_identity_loss_is_exact_and_rejects_duplicate_axes() -> None:
    scale = ((16 - 1) / 2.0) ** 0.5
    identity_samples = torch.zeros(16, 8)
    for index in range(8):
        identity_samples[2 * index, index] = scale
        identity_samples[2 * index + 1, index] = -scale
    state = identity_samples[:, None].expand(-1, 4, -1).contiguous()
    torch.testing.assert_close(_covariance_identity_loss(state), torch.tensor(0.0))

    duplicated = state[:, :, :1].expand(-1, -1, 8).contiguous()
    torch.testing.assert_close(_covariance_identity_loss(duplicated), torch.tensor(7.0))
    torch.testing.assert_close(
        _cross_covariance_identity_loss(state, state),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        _cross_covariance_identity_loss(torch.zeros_like(state), state),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        _cross_covariance_identity_loss(duplicated, duplicated),
        torch.tensor(7.0),
    )


def test_cross_identity_rejects_a_whitened_but_misaligned_target() -> None:
    scale = ((16 - 1) / 2.0) ** 0.5
    samples = torch.zeros(16, 8)
    for index in range(8):
        samples[2 * index, index] = scale
        samples[2 * index + 1, index] = -scale
    predicted = samples[:, None].expand(-1, 4, -1).contiguous()
    target = predicted.roll(shifts=1, dims=-1)
    torch.testing.assert_close(_covariance_identity_loss(predicted), torch.tensor(0.0))
    torch.testing.assert_close(_covariance_identity_loss(target), torch.tensor(0.0))
    assert float(_cross_covariance_identity_loss(predicted, target)) > 0.0


def test_covariance_identity_is_computed_per_horizon_not_flattened() -> None:
    horizon_code = torch.arange(4, dtype=torch.float32)[None, :, None].expand(16, 4, 8)
    torch.testing.assert_close(
        _covariance_identity_loss(horizon_code),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        _cross_covariance_identity_loss(horizon_code, horizon_code),
        torch.tensor(1.0),
    )


def test_config_removes_hinge_variance_and_balances_full_whitening() -> None:
    config = _config()
    assert config.similarity_weight == 25.0
    assert config.variance_regularization_weight == 0.0
    assert config.mean_regularization_weight == 25.0
    assert config.covariance_regularization_weight == 25.0
    assert config.history_teacher_alignment_weight == 1.0


def test_forward_keeps_exact_persistence_and_exact_loss_coefficients() -> None:
    model = _model().train()
    output = model(*_batch())
    assert output.target_state is not None
    torch.testing.assert_close(
        output.predicted_state,
        torch.zeros_like(output.predicted_state),
        rtol=0.0,
        atol=0.0,
    )
    assert output.predicted_variance_loss is not None
    assert output.target_variance_loss is not None
    torch.testing.assert_close(output.predicted_variance_loss, torch.tensor(0.0))
    torch.testing.assert_close(output.target_variance_loss, torch.tensor(0.0))
    assert output.state_prediction_loss is not None
    assert output.predicted_covariance_loss is not None
    torch.testing.assert_close(output.state_prediction_loss, torch.tensor(1.0))
    torch.testing.assert_close(output.predicted_covariance_loss, torch.tensor(1.0))
    components = (
        output.state_prediction_loss,
        output.predicted_covariance_loss,
        output.target_covariance_loss,
        output.predicted_mean_loss,
        output.target_mean_loss,
        output.history_teacher_alignment_loss,
        output.total_loss,
    )
    assert all(value is not None for value in components)
    similarity, predicted_w, target_w, predicted_m, target_m, alignment, total = components
    torch.testing.assert_close(
        similarity,
        _cross_covariance_identity_loss(output.predicted_state, output.target_state),
    )
    torch.testing.assert_close(
        predicted_w,
        _covariance_identity_loss(output.predicted_state),
    )
    torch.testing.assert_close(target_w, _covariance_identity_loss(output.target_state))
    torch.testing.assert_close(predicted_m, _mean_loss(output.predicted_state))
    torch.testing.assert_close(target_m, _mean_loss(output.target_state))
    expected = (
        25.0 * similarity
        + 12.5 * (predicted_w + target_w)
        + 12.5 * (predicted_m + target_m)
        + alignment
    )
    torch.testing.assert_close(total, expected)


def test_joint_objective_opens_predictor_then_reaches_every_learned_branch() -> None:
    model = _model().train()
    batch = _batch()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    first = model(*batch)
    assert first.total_loss is not None
    first.total_loss.backward()
    assert model.state_projector[-1].weight.grad is not None
    assert float(model.state_projector[-1].weight.grad.norm()) > 0.0
    opening_gradient = model.state_projector[-1].weight.grad
    assert bool((opening_gradient.norm(dim=1) > 0).all())
    assert int(torch.linalg.matrix_rank(opening_gradient).item()) == 8
    assert model.target_state_compressor.value_weight.grad is not None
    assert float(model.target_state_compressor.value_weight.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = model(*batch)
    assert second.total_loss is not None
    second.total_loss.backward()
    gradients = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief.spatial_embedding.weight.grad,
        model.action_embedding.weight.grad,
        model.future_cell.horizon_embedding.weight.grad,
        model.target_state_compressor.value_weight.grad,
        model.state_projector[-1].weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
        for value in gradients
        if value is not None
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
