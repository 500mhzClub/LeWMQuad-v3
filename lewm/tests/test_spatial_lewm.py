from __future__ import annotations

import pytest
import torch

from lewm.models.lewm import LeWorldModel
from lewm.models.spatial_lewm import SpatialLeWorldModel, spatial_variance_floor_loss
from lewm.models.spatial_predictor import trainable_parameter_count


def _model() -> SpatialLeWorldModel:
    return SpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        sigreg_projections=8,
        sigreg_knots=5,
    )


def test_spatial_variance_floor_penalizes_collapsed_tokens() -> None:
    collapsed = torch.zeros(3, 2, 4, 8)
    varied = torch.randn(64, 2, 4, 8) * 2.0

    assert spatial_variance_floor_loss(collapsed) > 0.9
    assert spatial_variance_floor_loss(varied) < 0.1


def test_spatial_lewm_trains_encoder_and_predictor_end_to_end() -> None:
    torch.manual_seed(17)
    model = _model()
    vision = torch.randn(2, 3, 3, 28, 28)
    actions = torch.randn(2, 2, 6)

    output = model(vision, actions, return_latents=True)
    output["loss"].backward()

    assert output["spatial_raw"].shape == (2, 3, 4, 12)
    assert output["predicted_spatial_proj"].shape == (2, 2, 4, 12)
    rollout = model.rollout_spatial(output["spatial_raw"][:, 0], actions)
    assert rollout.shape == (2, 2, 4, 12)
    assert model.encoder.patch_embed.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_spatial_lewm_rejects_misaligned_actions() -> None:
    model = _model()
    with pytest.raises(ValueError, match="cmd_seq"):
        model(torch.randn(2, 3, 3, 28, 28), torch.randn(2, 1, 6))


def test_default_spatial_lewm_is_capacity_matched_to_pooled_lewm() -> None:
    pooled = LeWorldModel()
    spatial = SpatialLeWorldModel()
    relative_difference = abs(
        trainable_parameter_count(spatial) - trainable_parameter_count(pooled)
    ) / trainable_parameter_count(pooled)

    assert relative_difference < 0.01
