from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1 import (
    FactualSharedTransitionTrajectoryH4JEPA,
    FactualSharedTransitionTrajectoryH4JEPAConfig,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> FactualSharedTransitionTrajectoryH4JEPA:
    torch.manual_seed(131)
    return FactualSharedTransitionTrajectoryH4JEPA(
        config=FactualSharedTransitionTrajectoryH4JEPAConfig(
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
    generator = torch.Generator().manual_seed(132)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _open_head(model: FactualSharedTransitionTrajectoryH4JEPA) -> None:
    generator = torch.Generator().manual_seed(133)
    with torch.no_grad():
        weight = torch.randn(
            model.prediction_projector[-1].weight.shape,
            generator=generator,
        )
        model.prediction_projector[-1].weight.copy_(0.02 * weight)
        model.prediction_projector[-1].bias.zero_()


def _observed_prior_score(
    model: FactualSharedTransitionTrajectoryH4JEPA,
    history: torch.Tensor,
    output: object,
) -> torch.Tensor:
    teacher_history = model._encode_fixed_teacher_history(history)
    target = teacher_history[:, 1:] - teacher_history[:, :-1]
    innovations = output.all_six_trajectory_innovations[:, :, :2]
    _horizon, _joint, combined = trajectory_energy_score(innovations, target)
    return combined.mean()


def test_inventory_uses_one_transition_object_on_all_six_edges() -> None:
    with pytest.raises(ValueError):
        FactualSharedTransitionTrajectoryH4JEPAConfig(trajectory_atom_count=3)
    with pytest.raises(ValueError):
        FactualSharedTransitionTrajectoryH4JEPAConfig(
            local_innovation_score_weight=0.6
        )

    model = _model().eval()
    forbidden = (
        nn.RNN,
        nn.GRU,
        nn.LSTM,
        nn.RNNCell,
        nn.GRUCell,
        nn.LSTMCell,
    )
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert model.future_cell.layer_count == 1
    assert not any(
        "horizon" in name or "action_prefix" in name
        for name, _module in model.named_modules()
    )

    calls = 0

    def count_call(_module: nn.Module, _args: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.future_cell.register_forward_hook(count_call)
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    handle.remove()
    assert calls == 6
    assert output.observed_prior_latents.shape == (2, 4, 2, 4, 12)
    assert output.trajectory_latents.shape == (2, 4, 4, 4, 12)
    assert output.all_six_trajectory_innovations.shape == (2, 4, 6, 4, 12)

    groups = _parameter_groups(model)
    assert set(groups) == {"encoder", "history", "predictor"}
    grouped = {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert grouped == trainable


def test_zero_head_is_exact_factual_and_future_persistence() -> None:
    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    online = F.normalize(output.history_latents, dim=-1)
    expected_observed = torch.stack((online[:, 0], online[:, 1]), dim=1)
    expected_observed = expected_observed[:, None].expand_as(
        output.observed_prior_latents
    )
    expected_future = online[:, 2, None, None].expand_as(
        output.trajectory_latents
    )

    torch.testing.assert_close(
        output.observed_prior_latents,
        expected_observed,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.trajectory_latents,
        expected_future,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.observed_prior_deltas,
        torch.zeros_like(output.observed_prior_deltas),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.all_six_trajectory_innovations,
        torch.zeros_like(output.all_six_trajectory_innovations),
        rtol=0.0,
        atol=0.0,
    )


def test_preobservation_priors_are_causal_and_future_rgb_never_leaks() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, future_rgb = _batch()
    base = model(history, past, future, future_rgb)

    after_e0 = history.clone()
    after_e0[:, 1:] = torch.flip(after_e0[:, 1:], dims=(1, 3, 4))
    changed_after_e0 = model(after_e0, past, future, future_rgb.flip(1))
    torch.testing.assert_close(
        base.observed_prior_latents[:, :, 0],
        changed_after_e0.observed_prior_latents[:, :, 0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        base.observed_prior_latents[:, :, 1],
        changed_after_e0.observed_prior_latents[:, :, 1],
    )

    after_e1 = history.clone()
    after_e1[:, 2] = torch.flip(after_e1[:, 2], dims=(2, 3))
    changed_after_e1 = model(after_e1, past, future, future_rgb.flip(1))
    torch.testing.assert_close(
        base.observed_prior_latents,
        changed_after_e1.observed_prior_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(base.trajectory_latents, changed_after_e1.trajectory_latents)

    other_targets = model(history, past, future, future_rgb.flip(1))
    torch.testing.assert_close(
        base.trajectory_latents,
        other_targets.trajectory_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert base.target_latents is not None
    assert not base.target_latents.requires_grad


def test_objective_is_exact_factual_dual_score_and_fixed_alignment() -> None:
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
        "half_all_six_factual_local_innovation_energy_score",
        "half_open_loop_future_cumulative_trajectory_energy_score",
    }

    teacher_history = model._encode_fixed_teacher_history(history)
    teacher_path = torch.cat((teacher_history, target), dim=1)
    target_innovations = teacher_path[:, 1:] - teacher_path[:, :-1]
    _lh, _lj, local = trajectory_energy_score(
        output.all_six_trajectory_innovations,
        target_innovations,
    )
    _ch, _cj, cumulative = trajectory_energy_score(
        output.trajectory_latents,
        target,
    )
    alignment = (
        (F.normalize(output.history_latents, dim=-1) - teacher_history)
        .square()
        .sum(dim=-1)
        .mean()
    )
    torch.testing.assert_close(losses["history_teacher_alignment"], alignment)
    torch.testing.assert_close(
        losses["half_all_six_factual_local_innovation_energy_score"],
        0.5 * local.mean(),
    )
    torch.testing.assert_close(
        losses["half_open_loop_future_cumulative_trajectory_energy_score"],
        0.5 * cumulative.mean(),
    )
    sum(losses.values()).backward()
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )
    frozen = copy.deepcopy(model.target_encoder.state_dict())
    model.hard_sync_target()
    model.update_target(0.0)
    for name, expected in frozen.items():
        torch.testing.assert_close(
            model.target_encoder.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_zero_head_stages_upstream_gradients_until_the_head_opens() -> None:
    model = _model().train()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    score = _observed_prior_score(model, history, output)
    head = model.prediction_projector[-1]
    upstream = (
        model.encoder.patch_embed.weight,
        model.initial_belief.mode_embedding.weight,
        model.initial_belief.spatial_embedding.weight,
        model.action_embedding.weight,
        model.future_cell.spatial_block.attn.in_proj_weight,
    )
    gradients = torch.autograd.grad(
        score,
        (head.weight, *upstream),
        allow_unused=True,
    )
    assert bool(torch.isfinite(gradients[0]).all())
    assert float(gradients[0].norm()) > 0.0
    for gradient in gradients[1:]:
        if gradient is not None:
            torch.testing.assert_close(
                gradient,
                torch.zeros_like(gradient),
                rtol=0.0,
                atol=1e-7,
            )

    _open_head(model)
    opened = model(history, past, future)
    opened_score = _observed_prior_score(model, history, opened)
    opened_gradients = torch.autograd.grad(
        opened_score,
        upstream,
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in opened_gradients)
    assert all(
        bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0.0
        for gradient in opened_gradients
        if gradient is not None
    )


def test_future_is_carrier_recursive_and_modes_only_permute_particles() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, future_rgb = _batch()
    _history_latents, belief = model.encode_history(history, past)
    visual, hidden = model._unpack_belief(belief)
    first, hidden, _delta = model._transition_step(
        visual,
        hidden,
        future[:, 0],
    )
    intervenable = first.detach().requires_grad_(True)
    visual = intervenable
    for step in range(1, 4):
        visual, hidden, _delta = model._transition_step(
            visual,
            hidden,
            future[:, step],
        )
    probe = torch.randn_like(visual)
    carrier_gradient = torch.autograd.grad((visual * probe).sum(), intervenable)[0]
    assert bool(torch.isfinite(carrier_gradient).all())
    assert float(carrier_gradient.norm()) > 0.0

    original = model(history, past, future).trajectory_latents
    permuted_model = copy.deepcopy(model)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        source = model.initial_belief.mode_embedding.weight
        permuted_model.initial_belief.mode_embedding.weight.copy_(
            source[permutation]
        )
    permuted = permuted_model(history, past, future).trajectory_latents
    torch.testing.assert_close(
        permuted,
        original[:, permutation],
        rtol=1e-5,
        atol=1e-6,
    )
    target = model.encode_target(future_rgb)
    _oh, _oj, original_score = trajectory_energy_score(original, target)
    _ph, _pj, permuted_score = trajectory_energy_score(permuted, target)
    torch.testing.assert_close(original_score, permuted_score)


def test_earlier_history_changes_belief_and_forecast_with_e2_fixed() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, _future_rgb = _batch()
    changed = history.clone()
    changed[:, :2] = torch.flip(changed[:, :2], dims=(1, 3, 4))
    first = model(history, past, future)
    second = model(changed, past, future)
    torch.testing.assert_close(
        F.normalize(first.history_latents[:, 2], dim=-1),
        F.normalize(second.history_latents[:, 2], dim=-1),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(first.final_hidden_particles, second.final_hidden_particles)
    assert not torch.equal(first.trajectory_latents, second.trajectory_latents)
