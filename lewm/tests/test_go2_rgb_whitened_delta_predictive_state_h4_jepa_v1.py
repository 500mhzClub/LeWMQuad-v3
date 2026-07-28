from __future__ import annotations

import copy

import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 import (
    WhitenedDeltaPredictiveStateConfig,
    WhitenedDeltaPredictiveStateH4JEPA,
    _covariance_loss,
    _mean_loss,
    _variance_loss,
)
from scripts.run_go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 import (
    _parameter_groups,
)


def _config() -> WhitenedDeltaPredictiveStateConfig:
    return WhitenedDeltaPredictiveStateConfig(
        image_size=8,
        patch_size=4,
        feature_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        recurrent_spatial_heads=3,
        cross_attention_heads=3,
    )


def _model() -> WhitenedDeltaPredictiveStateH4JEPA:
    torch.manual_seed(71)
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
    return WhitenedDeltaPredictiveStateH4JEPA(
        encoder.state_dict(), config=config
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(72)
    history = torch.randn(16, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(16, 4, 3, 8, 8, generator=generator)
    past = torch.randint(0, 9, (16, 2), generator=generator)
    future = torch.randint(0, 9, (16, 4), generator=generator)
    return history, past, future, future_rgb


def test_compact_target_is_exactly_zero_preserving_and_predictor_starts_zero() -> None:
    model = _model().eval()
    zero_delta = torch.zeros(2, 4, model.spatial_token_count, 12)
    torch.testing.assert_close(
        model.target_state_compressor(zero_delta),
        torch.zeros(2, 4, 8),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(model.target_state_compressor.query) > 0
    assert torch.count_nonzero(model.target_state_compressor.position_logits) > 0
    assert torch.count_nonzero(model.target_state_compressor.value_weight) > 0
    value_rows = model.target_state_compressor.value_weight
    torch.testing.assert_close(
        value_rows @ value_rows.T,
        torch.eye(8),
        rtol=1e-5,
        atol=1e-6,
    )
    assert model.target_state_compressor.output_scale == 2.0

    history, past, future, future_rgb = _batch()
    output = model(history, past, future, future_rgb)
    assert output.predicted_state.shape == (16, 4, 8)
    assert output.target_state is not None
    assert output.target_state.shape == (16, 4, 8)
    torch.testing.assert_close(
        output.predicted_state,
        torch.zeros_like(output.predicted_state),
        rtol=0.0,
        atol=0.0,
    )


def test_target_state_uses_fixed_teacher_not_online_encoder_or_actions() -> None:
    model = _model().eval()
    history, past, future, future_rgb = _batch()
    first = model(history, past, future, future_rgb)
    changed_actions = (future + 3) % 9
    changed_history = history.clone()
    changed_history[:, :2].mul_(-1)
    second = model(changed_history, past.flip(1), changed_actions, future_rgb)
    assert first.target_state is not None and second.target_state is not None
    torch.testing.assert_close(first.target_state, second.target_state, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.5)
    third = model(history, past, future, future_rgb)
    assert third.target_state is not None
    torch.testing.assert_close(first.target_state, third.target_state, rtol=0.0, atol=0.0)
    assert not torch.equal(first.history_latents, third.history_latents)


def test_target_only_objective_cannot_reach_online_or_fixed_encoder() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future, future_rgb)
    assert output.target_state is not None
    target_only = (
        _variance_loss(output.target_state, target_std=1.0, epsilon=1e-4)
        + _covariance_loss(output.target_state)
    )
    target_only.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert model.target_state_compressor.value_weight.grad is not None
    assert float(model.target_state_compressor.value_weight.grad.norm()) > 0.0


def test_joint_objective_opens_head_then_reaches_history_action_and_encoder() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    first = model(history, past, future, future_rgb)
    assert first.total_loss is not None
    first.total_loss.backward()
    final = model.state_projector[-1]
    assert final.weight.grad is not None and float(final.weight.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = model(history, past, future, future_rgb)
    assert not torch.equal(second.predicted_state, torch.zeros_like(second.predicted_state))
    assert second.total_loss is not None and torch.isfinite(second.total_loss)
    second.total_loss.backward()
    gradients = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief.spatial_embedding.weight.grad,
        model.initial_belief.encoder[0].self_attn.in_proj_weight.grad,
        model.action_embedding.weight.grad,
        model.future_cell.horizon_embedding.weight.grad,
        model.future_cell.future_action_path[0].weight.grad,
        model.future_cell.decoder[0].multihead_attn.in_proj_weight.grad,
        model.target_state_compressor.query.grad,
        model.target_state_compressor.position_logits.grad,
        model.target_state_compressor.value_weight.grad,
        final.weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
        for value in gradients
        if value is not None
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_variance_mean_and_covariance_are_computed_per_horizon() -> None:
    # Every horizon is constant across the batch but different from the others.
    # Flattening B*H would look diverse; the registered per-horizon loss must not.
    horizon_code = torch.arange(4, dtype=torch.float32)[None, :, None].expand(16, 4, 8)
    variance = _variance_loss(horizon_code, target_std=1.0, epsilon=1e-4)
    torch.testing.assert_close(variance, torch.tensor(0.9801), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(_mean_loss(horizon_code), torch.tensor(3.5))
    torch.testing.assert_close(_covariance_loss(horizon_code), torch.tensor(0.0))


def test_total_loss_has_the_exact_registered_coefficients() -> None:
    model = _model().train()
    output = model(*_batch())
    components = (
        output.state_prediction_loss,
        output.predicted_variance_loss,
        output.target_variance_loss,
        output.predicted_mean_loss,
        output.target_mean_loss,
        output.predicted_covariance_loss,
        output.target_covariance_loss,
        output.history_teacher_alignment_loss,
        output.total_loss,
    )
    assert all(value is not None for value in components)
    similarity, predicted_v, target_v, predicted_m, target_m, predicted_c, target_c, alignment, total = components
    expected = (
        25.0 * similarity
        + 12.5 * (predicted_v + target_v)
        + 12.5 * (predicted_m + target_m)
        + 0.5 * (predicted_c + target_c)
        + alignment
    )
    torch.testing.assert_close(total, expected)


def test_fixed_teacher_identity_and_optimizer_inventory_are_exact() -> None:
    model = _model().train()
    fixed = copy.deepcopy(model.target_encoder.state_dict())
    model.hard_sync_target()
    model.update_target(0.0)
    for name, expected in fixed.items():
        torch.testing.assert_close(
            model.target_encoder.state_dict()[name], expected, rtol=0.0, atol=0.0
        )
    assert int(model.ema_update_count.item()) == 0
    assert not model.target_encoder.training

    groups = _parameter_groups(model)
    assert set(groups) == {"encoder", "history", "predictor", "target_state"}
    ids = [{id(parameter) for parameter in values} for values in groups.values()]
    assert all(not (left & right) for i, left in enumerate(ids) for right in ids[i + 1 :])
    grouped = set().union(*ids)
    assert grouped == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    assert not grouped & target_ids
