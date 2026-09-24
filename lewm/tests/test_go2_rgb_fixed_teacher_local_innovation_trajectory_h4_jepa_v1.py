from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    LocalInnovationTrajectoryH4JEPA,
    LocalInnovationTrajectoryH4JEPAConfig,
    LocalInnovationTrajectoryH4JEPAOutput,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> LocalInnovationTrajectoryH4JEPA:
    torch.manual_seed(91)
    return LocalInnovationTrajectoryH4JEPA(
        config=LocalInnovationTrajectoryH4JEPAConfig(
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
    generator = torch.Generator().manual_seed(92)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def test_exact_transition_registration_and_telescoping() -> None:
    current = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    increments = torch.stack(
        tuple(torch.full_like(current, float(step)) for step in (1, 2, 3, 4)),
        dim=1,
    )
    future = current[:, None] + increments.cumsum(dim=1)
    registered = fixed_teacher_local_innovations(current, future)
    torch.testing.assert_close(registered, increments, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        current[:, None] + registered.cumsum(dim=1),
        future,
        rtol=0.0,
        atol=0.0,
    )

    atoms = future[:, None].expand(-1, 4, -1, -1, -1).contiguous()
    realized = realized_trajectory_innovations(current, atoms)
    assert realized.shape == (1, 4, 4, 2, 6)
    torch.testing.assert_close(
        realized,
        increments[:, None].expand_as(realized),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        current[:, None, None] + realized.cumsum(dim=2),
        atoms,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_head_is_exact_persistence_with_zero_realized_innovations() -> None:
    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    assert isinstance(output, LocalInnovationTrajectoryH4JEPAOutput)
    current = F.normalize(output.history_latents[:, 2], dim=-1)
    expected = current[:, None, None].expand_as(output.trajectory_latents)

    assert output.trajectory_latents.shape == (2, 4, 4, 4, 12)
    assert output.trajectory_deltas.shape == (2, 4, 4, 4, 12)
    assert output.trajectory_innovations.shape == (2, 4, 4, 4, 12)
    torch.testing.assert_close(
        output.trajectory_deltas,
        torch.zeros_like(output.trajectory_deltas),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.trajectory_latents,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.trajectory_innovations,
        torch.zeros_like(output.trajectory_innovations),
        rtol=0.0,
        atol=0.0,
    )


def test_auxiliary_objective_is_local_proper_score_alignment_and_rankings() -> None:
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
        "future_teacher_local_innovation_energy_score",
        "cyclic_wrong_action_score_ranking",
        "history_counterfactual_score_ranking",
    }

    teacher_history = model._encode_fixed_teacher_history(history)
    target_innovations = fixed_teacher_local_innovations(
        teacher_history[:, 2],
        target,
    )
    _horizon, _joint, combined = trajectory_energy_score(
        output.trajectory_innovations,
        target_innovations,
    )
    expected_alignment = (
        (
            F.normalize(output.history_latents, dim=-1)
            - teacher_history
        )
        .square()
        .sum(dim=-1)
        .mean()
    )
    torch.testing.assert_close(
        losses["future_teacher_local_innovation_energy_score"],
        combined.mean(),
    )
    torch.testing.assert_close(
        losses["history_teacher_alignment"],
        expected_alignment,
    )
    torch.testing.assert_close(
        losses["cyclic_wrong_action_score_ranking"],
        torch.tensor(0.05),
    )
    torch.testing.assert_close(
        losses["history_counterfactual_score_ranking"],
        torch.tensor(0.03),
    )


def test_zero_head_has_nonzero_opening_gradient_and_fixed_target_has_none() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=model.encode_target(future_rgb),
        output=output,
    )
    final = model.prediction_projector[-1]
    energy_gradient = torch.autograd.grad(
        losses["future_teacher_local_innovation_energy_score"],
        final.weight,
        retain_graph=True,
    )[0]
    action_gradient = torch.autograd.grad(
        losses["cyclic_wrong_action_score_ranking"],
        final.weight,
        retain_graph=True,
    )[0]
    assert float(energy_gradient.norm()) > 0.0
    assert float(action_gradient.norm()) > 0.0
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert final.weight.grad is not None
    assert bool(torch.isfinite(final.weight.grad).all())
    assert float(final.weight.grad.norm()) > 0.0
    assert all(
        parameter.grad is None
        for parameter in model.target_encoder.parameters()
    )


def test_perturbed_head_registers_action_prefixes_and_control_differences() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.prediction_projector[-1].weight.normal_(std=0.02)
        model.prediction_projector[-1].bias.zero_()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    changed = future.clone()
    changed[:, 3] = (changed[:, 3] + 1) % len(GO2_H4_PRIMITIVE_VOCABULARY)
    changed_atoms = model.predict_trajectory_atoms_from_belief(
        output.belief_latents,
        changed,
    )
    torch.testing.assert_close(
        output.trajectory_latents[:, :, :3],
        changed_atoms[:, :, :3],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(output.trajectory_latents[:, :, 3], changed_atoms[:, :, 3])

    wrong = (future + 1) % len(GO2_H4_PRIMITIVE_VOCABULARY)
    wrong_atoms = model.predict_trajectory_atoms_from_belief(
        output.belief_latents,
        wrong,
    )
    assert not torch.equal(output.trajectory_latents, wrong_atoms)

    reversed_belief = model._belief_from_encoded_history(
        output.history_latents[:, [1, 0, 2]],
        past.flip(dims=(1,)),
    )
    reversed_atoms = model.predict_trajectory_atoms_from_belief(
        reversed_belief,
        future,
    )
    assert not torch.equal(output.trajectory_latents, reversed_atoms)

    hold_index = GO2_H4_PRIMITIVE_VOCABULARY.index("hold")
    reset_belief = model._belief_from_encoded_history(
        output.history_latents[:, 2:3].expand(-1, 3, -1, -1),
        torch.full_like(past, hold_index),
    )
    reset_atoms = model.predict_trajectory_atoms_from_belief(
        reset_belief,
        future,
    )
    assert not torch.equal(output.trajectory_latents, reset_atoms)


def test_history_ranking_reaches_dense_history_after_head_is_open() -> None:
    model = _model().train()
    with torch.no_grad():
        model.prediction_projector[-1].weight.normal_(std=1e-4)
        model.prediction_projector[-1].bias.zero_()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=model.encode_target(future_rgb),
        output=output,
    )
    ranking = losses["history_counterfactual_score_ranking"]
    assert float(ranking.detach()) > 0.0
    ranking.backward()
    gradients = (
        model.initial_belief.spatial_embedding.weight.grad,
        model.initial_belief.time_embedding.weight.grad,
        model.initial_belief.transition_step_embedding.weight.grad,
        model.initial_belief.encoder[0].self_attn.in_proj_weight.grad,
        model.future_cell.decoder[0].multihead_attn.in_proj_weight.grad,
        model.prediction_projector[-1].weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
        for value in gradients
        if value is not None
    )


def test_fixed_target_config_and_trainable_inventory_are_exact() -> None:
    config = LocalInnovationTrajectoryH4JEPAConfig()
    assert config.trajectory_atom_count == 4
    assert config.action_vocabulary == GO2_H4_PRIMITIVE_VOCABULARY
    assert config.teacher_alignment_weight == 1.0
    assert config.teacher_delta_weight == 1.0
    assert config.cyclic_wrong_action_ranking_weight == 1.0
    assert config.history_ranking_weight == 1.0
    assert config.cyclic_wrong_action_margin == 0.05
    assert config.history_margin == 0.03
    with pytest.raises(ValueError):
        LocalInnovationTrajectoryH4JEPAConfig(history_margin=0.031)

    model = _model().train()
    fixed_target = copy.deepcopy(model.target_encoder.state_dict())
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.25)
    model.hard_sync_target()
    model.update_target(0.0)
    for name, expected in fixed_target.items():
        torch.testing.assert_close(
            model.target_encoder.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert int(model.ema_update_count.item()) == 0
    assert not model.target_encoder.training
    assert all(
        not parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )

    groups = _parameter_groups(model)
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    covered_ids = {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
    assert set(groups) == {"encoder", "history", "predictor"}
    assert trainable_ids == covered_ids
    assert len(trainable_ids) == sum(len(parameters) for parameters in groups.values())
