from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1 import (
    FactorizedConditionalIncrementTrajectoryH4JEPA,
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> FactorizedConditionalIncrementTrajectoryH4JEPA:
    torch.manual_seed(231)
    return FactorizedConditionalIncrementTrajectoryH4JEPA(
        config=FactorizedConditionalIncrementTrajectoryH4JEPAConfig(
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
    generator = torch.Generator().manual_seed(232)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _open_w0(
    model: FactorizedConditionalIncrementTrajectoryH4JEPA,
    scale: float = 0.05,
) -> None:
    with torch.no_grad():
        feature_dim = model.config.feature_dim
        model.prediction_projector[-1].weight.copy_(
            scale * torch.eye(feature_dim)
        )


def _observed_prior_score(
    model: FactorizedConditionalIncrementTrajectoryH4JEPA,
    history: torch.Tensor,
    output: object,
) -> torch.Tensor:
    teacher_history = model._encode_fixed_teacher_history(history)
    target = teacher_history[:, 1:] - teacher_history[:, :-1]
    innovations = output.all_six_trajectory_innovations[:, :, :2]
    _horizon, _joint, combined = trajectory_energy_score(innovations, target)
    return combined.mean()


def test_inventory_and_centered_factorization_are_exact() -> None:
    model = _model().eval()
    assert model.future_cell.layer_count == 1
    assert model.history_cell.norm.elementwise_affine is False
    assert model.history_cell.projection.bias is None
    assert model.prediction_projector[-1].bias is None
    torch.testing.assert_close(
        model.prediction_projector[-1].weight,
        torch.zeros_like(model.prediction_projector[-1].weight),
        rtol=0.0,
        atol=0.0,
    )
    assert all(
        module.bias is None
        for module in model.future_spatial_refiner.modules()
        if isinstance(module, nn.Linear)
    )
    assert not hasattr(model.future_cell, "increment_path")

    codes = model._centered_action_codes()
    assert codes.shape == (9, 12)
    torch.testing.assert_close(
        codes.mean(dim=0),
        torch.zeros_like(codes[0]),
        rtol=0.0,
        atol=5e-8,
    )
    zero_increment = torch.zeros(2, 4, 4, 12)
    torch.testing.assert_close(
        model.history_cell(zero_increment),
        torch.zeros_like(zero_increment),
        rtol=0.0,
        atol=0.0,
    )

    calls = 0

    def count_b_call(_module: nn.Module, args: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1
        # B sees z, h, and D(d), never the current categorical action.
        assert len(args) == 3
        assert all(getattr(value, "ndim", None) == 4 for value in args)

    handle = model.future_cell.register_forward_pre_hook(count_b_call)
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    handle.remove()
    assert calls == 6
    assert output.belief_latents.shape == (2, 6, 4, 12)
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


def test_no_generic_successor_bypass_and_action_does_not_enter_b() -> None:
    model = _model().eval()
    _open_w0(model)
    generator = torch.Generator().manual_seed(233)
    visual = F.normalize(
        torch.randn(2, 4, 4, 12, generator=generator),
        dim=-1,
    )
    hidden = torch.randn(2, 4, 4, 12, generator=generator)
    zero_d = torch.zeros_like(visual)
    action_a = torch.tensor([0, 1], dtype=torch.long)
    action_b = torch.tensor([7, 8], dtype=torch.long)

    next_a, belief_a, increment_a = model._transition_step(
        visual,
        hidden,
        zero_d,
        action_a,
    )
    next_b, belief_b, increment_b = model._transition_step(
        visual,
        hidden,
        zero_d,
        action_b,
    )
    torch.testing.assert_close(belief_a, belief_b, rtol=0.0, atol=0.0)
    assert not torch.equal(increment_a, increment_b)
    assert not torch.equal(next_a, next_b)

    # The identity is asserted before renormalization: averaging the nine
    # categorical branches leaves exactly the projected incoming-d route.
    all_actions = torch.arange(9, dtype=torch.long)
    repeated_visual = visual[:1].expand(9, -1, -1, -1).clone()
    repeated_hidden = hidden[:1].expand(9, -1, -1, -1).clone()
    repeated_zero = torch.zeros_like(repeated_visual)
    _all_next, _all_b, all_projected = model._transition_step(
        repeated_visual,
        repeated_hidden,
        repeated_zero,
        all_actions,
    )
    torch.testing.assert_close(
        all_projected.mean(dim=0),
        torch.zeros_like(all_projected[0]),
        rtol=0.0,
        atol=5e-8,
    )
    one_incoming = 0.1 * torch.randn(1, 4, 4, 12, generator=generator)
    repeated_incoming = one_incoming.expand(9, -1, -1, -1).clone()
    _all_next, _all_b, all_projected = model._transition_step(
        repeated_visual,
        repeated_hidden,
        repeated_incoming,
        all_actions,
    )
    expected_history_route = model.prediction_projector(one_incoming)[0]
    torch.testing.assert_close(
        all_projected.mean(dim=0),
        expected_history_route,
        rtol=0.0,
        atol=5e-8,
    )

    collapsed = copy.deepcopy(model)
    with torch.no_grad():
        one_code = collapsed.action_embedding.weight[0].clone()
        collapsed.action_embedding.weight.copy_(
            one_code.expand_as(collapsed.action_embedding.weight)
        )
    collapsed_codes = collapsed._centered_action_codes()
    torch.testing.assert_close(
        collapsed_codes,
        torch.zeros_like(collapsed_codes),
        rtol=0.0,
        atol=5e-8,
    )
    collapsed_next, _collapsed_b, collapsed_increment = (
        collapsed._transition_step(
            visual,
            hidden,
            zero_d,
            action_a,
        )
    )
    torch.testing.assert_close(
        collapsed_increment,
        torch.zeros_like(collapsed_increment),
        rtol=0.0,
        atol=5e-8,
    )
    torch.testing.assert_close(
        collapsed_next,
        visual,
        rtol=1e-6,
        atol=5e-8,
    )

    incoming = 0.1 * torch.randn(2, 4, 4, 12, generator=generator)
    history_next_a, history_b_a, history_increment_a = (
        collapsed._transition_step(
            visual,
            hidden,
            incoming,
            action_a,
        )
    )
    history_next_b, history_b_b, history_increment_b = (
        collapsed._transition_step(
            visual,
            hidden,
            incoming,
            action_b,
        )
    )
    torch.testing.assert_close(history_b_a, history_b_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        history_increment_a,
        history_increment_b,
        rtol=0.0,
        atol=5e-8,
    )
    torch.testing.assert_close(
        history_next_a,
        history_next_b,
        rtol=0.0,
        atol=5e-8,
    )
    assert float(history_increment_a.detach().norm()) > 0.0


def test_update_zero_is_exact_factual_and_future_persistence() -> None:
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


def test_observed_increments_are_factual_then_future_increments_are_recursive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    _open_w0(model, scale=0.02)
    history, past, future, _future_rgb = _batch()
    seen: list[torch.Tensor] = []
    original = model._transition_step

    def record(
        visual: torch.Tensor,
        hidden: torch.Tensor,
        incoming: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seen.append(incoming.detach().clone())
        return original(visual, hidden, incoming, action)

    monkeypatch.setattr(model, "_transition_step", record)
    output = model(history, past, future)
    assert len(seen) == 6
    online = F.normalize(output.history_latents, dim=-1)
    atom_count = model.config.trajectory_atom_count

    torch.testing.assert_close(seen[0], torch.zeros_like(seen[0]))
    factual_d1 = (online[:, 1] - online[:, 0])[:, None].expand(
        -1, atom_count, -1, -1
    )
    factual_d2 = (online[:, 2] - online[:, 1])[:, None].expand(
        -1, atom_count, -1, -1
    )
    torch.testing.assert_close(seen[1], factual_d1)
    torch.testing.assert_close(seen[2], factual_d2)
    recursive_d3 = output.trajectory_latents[:, :, 0] - online[:, 2, None]
    recursive_d4 = (
        output.trajectory_latents[:, :, 1]
        - output.trajectory_latents[:, :, 0]
    )
    recursive_d5 = (
        output.trajectory_latents[:, :, 2]
        - output.trajectory_latents[:, :, 1]
    )
    torch.testing.assert_close(seen[3], recursive_d3)
    torch.testing.assert_close(seen[4], recursive_d4)
    torch.testing.assert_close(seen[5], recursive_d5)


def test_priors_are_causal_and_future_targets_never_leak() -> None:
    model = _model().eval()
    _open_w0(model, scale=0.02)
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


def test_objective_remains_the_exact_joint_factual_jepa_loss() -> None:
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


def test_zero_w0_stages_factorized_gradients_until_the_head_opens() -> None:
    model = _model().train()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    score = _observed_prior_score(model, history, output)
    w0 = model.prediction_projector[-1].weight
    upstream = (
        model.initial_belief.mode_embedding.weight,
        model.action_embedding.weight,
        model.history_cell.projection.weight,
        model.future_cell.spatial_block.attn.in_proj_weight,
        model.future_spatial_refiner.tower[-1].weight,
    )
    gradients = torch.autograd.grad(
        score,
        (w0, *upstream),
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

    _open_w0(model, scale=0.02)
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


def test_future_is_recursive_and_modes_only_permute_particles() -> None:
    model = _model().eval()
    _open_w0(model, scale=0.02)
    history, past, future, future_rgb = _batch()
    _history_latents, belief = model.encode_history(history, past)
    visual, incoming, hidden = model._unpack_belief(belief)
    first, hidden, _projected = model._transition_step(
        visual,
        hidden,
        incoming,
        future[:, 0],
    )
    intervenable = first.detach().requires_grad_(True)
    incoming = intervenable - visual
    visual = intervenable
    for step in range(1, 4):
        next_visual, hidden, _projected = model._transition_step(
            visual,
            hidden,
            incoming,
            future[:, step],
        )
        incoming = next_visual - visual
        visual = next_visual
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


def test_earlier_history_changes_packed_increment_and_forecast_with_e2_fixed() -> None:
    model = _model().eval()
    _open_w0(model, scale=0.02)
    history, past, future, _future_rgb = _batch()
    changed = history.clone()
    changed[:, :2] = torch.flip(changed[:, :2], dims=(1, 3, 4))
    # Preserve e2 exactly while changing both its incoming factual increment
    # and the earlier causal hidden context.
    first = model(history, past, future)
    second = model(changed, past, future)
    torch.testing.assert_close(
        F.normalize(first.history_latents[:, 2], dim=-1),
        F.normalize(second.history_latents[:, 2], dim=-1),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(first.belief_latents[:, 1:], second.belief_latents[:, 1:])
    assert not torch.equal(first.trajectory_latents, second.trajectory_latents)
