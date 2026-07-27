from __future__ import annotations

import torch

from lewm.models.go2_recurrent_h4_joint_jepa import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
)


def _tiny_model() -> JointRecurrentH4JEPA:
    return JointRecurrentH4JEPA(
        config=JointRecurrentH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            recurrent_spatial_mlp_ratio=1,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past_actions = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future_actions = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past_actions, future_actions, future


def test_joint_forward_backward_and_target_is_frozen() -> None:
    model = _tiny_model()
    output = model(*_batch())
    assert output.predicted_latents.shape == (2, 4, 4, 12)
    assert output.target_latents is not None
    assert output.per_horizon_loss is not None
    assert output.total_loss is not None and torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.history_cell.weight_hh.grad is not None
    assert model.future_cell.weight_hh.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_future_rgb_cannot_change_online_predictions() -> None:
    model = _tiny_model().eval()
    history, past, actions, future = _batch()
    first = model(history, past, actions, future).predicted_latents
    second = model(history, past, actions, future.flip(1)).predicted_latents
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_fixed_belief_responds_to_counterfactual_actions() -> None:
    model = _tiny_model().eval()
    history, past, actions, _future = _batch()
    belief = model.encode_history(history, past).belief_latents
    actual = model.predict_from_belief(belief, actions)
    wrong = model.predict_from_belief(belief, (actions + 1) % 9)
    assert not torch.equal(actual, wrong)
