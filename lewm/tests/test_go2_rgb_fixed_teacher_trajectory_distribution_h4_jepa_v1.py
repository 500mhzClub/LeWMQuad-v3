from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> JointRecurrentH4JEPA:
    torch.manual_seed(61)
    return JointRecurrentH4JEPA(
        config=JointRecurrentH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            cross_attention_heads=3,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(62)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def test_energy_score_uses_exact_all_pair_uniform_distribution_coefficient() -> None:
    target = torch.zeros(1, 1, 1, 1)
    atoms = torch.zeros(1, 4, 1, 1, 1)
    atoms[:, 0] = 8.0
    horizon, joint, combined = trajectory_energy_score(atoms, target)

    # Fit is R/4.  The all-K^2 pair mean is 6R/16 and receives coefficient 1/2.
    expected = torch.tensor([0.5])
    torch.testing.assert_close(horizon, expected[:, None])
    torch.testing.assert_close(joint, expected)
    torch.testing.assert_close(combined, expected)
    zero_horizon, zero_joint, zero_combined = trajectory_energy_score(
        torch.zeros_like(atoms), target
    )
    torch.testing.assert_close(zero_horizon, torch.zeros_like(zero_horizon))
    torch.testing.assert_close(zero_joint, torch.zeros_like(zero_joint))
    torch.testing.assert_close(zero_combined, torch.zeros_like(zero_combined))


def test_four_coherent_atoms_start_at_exact_persistence() -> None:
    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    current = F.normalize(output.history_latents[:, 2], dim=-1)
    expected = current[:, None, None].expand_as(output.trajectory_latents)

    assert output.trajectory_latents.shape == (2, 4, 4, 4, 12)
    assert output.trajectory_deltas.shape == (2, 4, 4, 4, 12)
    torch.testing.assert_close(output.trajectory_latents, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output.trajectory_deltas,
        torch.zeros_like(output.trajectory_deltas),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.predicted_latents,
        current[:, None].expand_as(output.predicted_latents),
        rtol=1e-6,
        atol=1e-7,
    )
    assert model.future_cell.mode_embedding.num_embeddings == 4
    assert model.future_cell.horizon_embedding.num_embeddings == 4


def test_future_rgb_cannot_change_online_trajectory_atoms() -> None:
    model = _model().eval()
    history, past, future, future_rgb = _batch()
    first = model(history, past, future, future_rgb).trajectory_latents
    second = model(history, past, future, future_rgb.flip(1)).trajectory_latents
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_objective_is_only_proper_energy_score_and_history_alignment() -> None:
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
    assert set(losses) == {
        "history_teacher_alignment",
        "future_teacher_trajectory_energy_score",
    }
    _horizon, _joint, expected_energy = trajectory_energy_score(
        output.trajectory_latents, target
    )
    teacher_history = model._encode_fixed_teacher_history(history)
    online_history = F.normalize(output.history_latents, dim=-1)
    expected_alignment = (
        (online_history - teacher_history).square().sum(dim=-1).mean()
    )
    torch.testing.assert_close(
        losses["future_teacher_trajectory_energy_score"],
        expected_energy.mean(),
    )
    torch.testing.assert_close(
        losses["history_teacher_alignment"], expected_alignment
    )
    assert output.variance_loss.item() == 0.0


def test_zero_head_opens_then_modes_diverge_and_gradients_reach_joint_stack() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )

    first = model(history, past, future)
    target = model.encode_target(future_rgb)
    first_losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=first,
    )
    sum(first_losses.values()).backward()
    final = model.prediction_projector[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.norm()) > 0.0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = model(history, past, future)
    assert float(second.trajectory_latents.var(dim=1).max().detach()) > 0.0
    second_losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=model.encode_target(future_rgb),
        output=second,
    )
    sum(second_losses.values()).backward()
    gradients = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief.spatial_embedding.weight.grad,
        model.action_embedding.weight.grad,
        model.future_cell.mode_embedding.weight.grad,
        model.future_cell.horizon_embedding.weight.grad,
        model.future_cell.future_action_path[0].weight.grad,
        model.future_cell.decoder[0].multihead_attn.in_proj_weight.grad,
        final.weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
        for value in gradients
        if value is not None
    )


def test_fixed_teacher_and_shared_runner_parameter_inventory_are_exact() -> None:
    model = _model().train()
    fixed_teacher = copy.deepcopy(model.target_encoder.state_dict())
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.25)
    model.hard_sync_target()
    model.update_target(0.0)
    for name, expected in fixed_teacher.items():
        torch.testing.assert_close(
            model.target_encoder.state_dict()[name], expected, rtol=0.0, atol=0.0
        )
    assert int(model.ema_update_count.item()) == 0
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    groups = _parameter_groups(model)
    assert set(groups) == {"encoder", "history", "predictor"}
    predictor_ids = {id(parameter) for parameter in groups["predictor"]}
    assert id(model.future_cell.mode_embedding.weight) in predictor_ids
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert trainable_ids == {
        id(parameter) for values in groups.values() for parameter in values
    }
