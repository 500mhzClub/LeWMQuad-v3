from __future__ import annotations

import torch
import torch.nn.functional as F

from lewm.models.go2_recurrent_h4_persistence_residual_joint_jepa_v2 import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
)


def _model() -> JointRecurrentH4JEPA:
    torch.manual_seed(11)
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
    generator = torch.Generator().manual_seed(12)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def test_zero_gates_make_every_horizon_exact_online_persistence() -> None:
    model = _model().eval()
    history, past, future, _ = _batch()
    output = model(history, past, future)
    current = F.normalize(output.history_latents[:, 2], dim=-1)
    expected = current[:, None].expand_as(output.predicted_latents)
    torch.testing.assert_close(output.predicted_latents, expected, rtol=0.0, atol=0.0)


def test_auxiliary_hinges_train_both_zero_gates_without_target_gradients() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    target = model.encode_target(future_rgb)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=output,
    )
    assert set(losses) == {"persistence_ranking", "history_ranking"}
    prediction = model._distance(output.predicted_latents, target).mean()
    total = prediction + output.variance_loss + sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.initial_belief.gate.grad is not None
    assert model.prediction_projector.gate.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_open_future_gate_restores_action_counterfactuals() -> None:
    model = _model().eval()
    model.prediction_projector.gate.data.fill_(0.1)
    history, past, future, _ = _batch()
    belief = model.encode_history(history, past).belief_latents
    actual = model.predict_from_belief(belief, future)
    wrong = model.predict_from_belief(belief, (future + 1) % 9)
    assert not torch.equal(actual, wrong)
