from __future__ import annotations

import pytest
import torch

from lewm.models.predictor import TransformerPredictor
from lewm.models.spatial_predictor import SpatialTokenPredictor, trainable_parameter_count


def _model() -> SpatialTokenPredictor:
    return SpatialTokenPredictor(
        latent_dim=12,
        cmd_dim=6,
        num_spatial_tokens=4,
        n_layers=2,
        n_heads=3,
        dim_head=4,
        mlp_dim=24,
        dropout=0.0,
    )


def test_spatial_predictor_rollout_preserves_grid_shape_and_gradients() -> None:
    torch.manual_seed(7)
    model = _model()
    start = torch.randn(2, 4, 12)
    actions = torch.randn(2, 3, 6)

    predicted = model.rollout(start, actions)
    predicted.square().mean().backward()

    assert predicted.shape == (2, 3, 4, 12)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_spatial_predictor_teacher_forcing_uses_aligned_future_tokens() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    start = torch.randn(2, 4, 12)
    actions = torch.randn(2, 2, 6)
    teachers = torch.randn(2, 2, 4, 12)

    predicted = model.rollout(
        start,
        actions,
        teacher_tokens=teachers,
        teacher_prob=1.0,
    )
    expected_second = model.predict_step(teachers[:, 0], actions[:, 1])

    assert torch.allclose(predicted[:, 1], expected_second)


def test_spatial_predictor_rejects_wrong_token_count() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Expected 4 spatial tokens"):
        model.predict_step(torch.randn(2, 5, 12), torch.randn(2, 6))


def test_spatial_predictor_uses_bidirectional_patch_attention(monkeypatch) -> None:
    model = _model().eval()
    causal_values = []
    original = model.blocks[0].attn.forward

    def record_causal(x, causal=True):
        causal_values.append(causal)
        return original(x, causal=causal)

    monkeypatch.setattr(model.blocks[0].attn, "forward", record_causal)
    tokens = torch.randn(1, 4, 12)
    action = torch.randn(1, 6)

    model.predict_step(tokens, action)

    assert causal_values == [False]


def test_default_spatial_predictor_is_capacity_matched_to_pooled_predictor() -> None:
    pooled = TransformerPredictor()
    spatial = SpatialTokenPredictor()
    relative_difference = abs(
        trainable_parameter_count(spatial) - trainable_parameter_count(pooled)
    ) / trainable_parameter_count(pooled)

    assert relative_difference < 0.01
