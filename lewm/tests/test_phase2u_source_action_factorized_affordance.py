from __future__ import annotations

import torch

from lewm.models.primitive_affordance import factorized_affordance_losses
from lewm.models.source_action_utility import SourceActionFactorizedAffordanceModel


def test_source_action_factorized_affordance_model_outputs_factor_logits() -> None:
    torch.manual_seed(202)
    model = SourceActionFactorizedAffordanceModel(
        cmd_dim=15,
        horizon=2,
        factor_count=6,
        latent_dim=24,
        action_hidden_dim=32,
        image_size=32,
        patch_size=8,
        encoder_depth=1,
        encoder_heads=2,
        encoder_mlp_ratio=2,
        fusion_mode="film_interaction",
    )
    logits = model(torch.rand(5, 3, 32, 32), torch.randn(5, 2, 15))
    losses = factorized_affordance_losses(
        factor_logits=logits[:, None, :],
        factor_targets=torch.rand(5, 1, 6),
        factor_mask=torch.ones(5, 1, 6, dtype=torch.bool),
    )

    losses["factorized_affordance_loss"].backward()

    assert logits.shape == (5, 6)
    assert model.head[-1].weight.grad is not None


def test_source_action_factorized_affordance_action_only_ignores_source() -> None:
    torch.manual_seed(303)
    model = SourceActionFactorizedAffordanceModel(
        cmd_dim=3,
        horizon=1,
        factor_count=6,
        latent_dim=16,
        action_hidden_dim=16,
        image_size=32,
        patch_size=8,
        encoder_depth=1,
        encoder_heads=2,
        encoder_mlp_ratio=2,
        input_mode="action_only",
    )
    model.eval()
    actions = torch.randn(2, 1, 3)

    first = model(torch.zeros(2, 3, 32, 32), actions)
    second = model(torch.ones(2, 3, 32, 32), actions)

    assert torch.allclose(first, second, atol=1e-6)
