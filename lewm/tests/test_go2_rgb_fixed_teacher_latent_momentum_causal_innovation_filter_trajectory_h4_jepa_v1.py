from __future__ import annotations

from copy import deepcopy
import inspect

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_trajectory_h4_jepa_v1 import (
    LatentMomentumCausalInnovationFilterTrajectoryH4JEPA,
    LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig,
    _tangent_projection,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> LatentMomentumCausalInnovationFilterTrajectoryH4JEPA:
    torch.manual_seed(241)
    return LatentMomentumCausalInnovationFilterTrajectoryH4JEPA(
        config=LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig(
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
    generator = torch.Generator().manual_seed(242)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _open_heads(
    model: LatentMomentumCausalInnovationFilterTrajectoryH4JEPA,
    *,
    scale: float = 0.02,
) -> None:
    with torch.no_grad():
        identity = torch.eye(model.config.feature_dim)
        model.prediction_projector[-1].weight.copy_(scale * identity)
        model.history_cell.content_gain_head.weight.copy_(scale * identity)
        model.history_cell.momentum_correction_head.weight.copy_(scale * identity)


def _isolated_prediction_score(
    model: LatentMomentumCausalInnovationFilterTrajectoryH4JEPA,
    history: torch.Tensor,
    past: torch.Tensor,
    future: torch.Tensor,
    future_rgb: torch.Tensor,
) -> torch.Tensor:
    output = model(history, past, future)
    teacher_history = model._encode_fixed_teacher_history(history)
    target = model.encode_target(future_rgb)
    target_path = torch.cat((teacher_history, target), dim=1)
    target_innovations = target_path[:, 1:] - target_path[:, :-1]
    _local_horizon, _local_joint, local = trajectory_energy_score(
        output.all_six_trajectory_innovations,
        target_innovations,
    )
    _future_horizon, _future_joint, cumulative = trajectory_energy_score(
        output.trajectory_latents,
        target,
    )
    return 0.5 * local.mean() + 0.5 * cumulative.mean()


def test_state_geometry_centering_and_update_zero_contract() -> None:
    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    online = F.normalize(
        output.history_latents,
        dim=-1,
        eps=model.config.normalization_epsilon,
    )
    atom_count = model.config.trajectory_atom_count
    token_count = model.spatial_token_count
    dim = model.config.feature_dim

    assert output.belief_latents.shape == (2, 2 * atom_count, token_count, dim)
    content, momentum = model._unpack_belief(output.belief_latents)
    assert content.shape == momentum.shape == (2, atom_count, token_count, dim)
    torch.testing.assert_close(
        (content * momentum).sum(dim=-1),
        torch.zeros_like(content[..., 0]),
        rtol=0.0,
        atol=1e-6,
    )

    mode_rows = model.initial_belief.mode_embedding.weight
    torch.testing.assert_close(
        (mode_rows - mode_rows.mean(dim=0, keepdim=True)).mean(dim=0),
        torch.zeros_like(mode_rows[0]),
        rtol=0.0,
        atol=5e-8,
    )
    action_codes = model._centered_action_codes()
    torch.testing.assert_close(
        action_codes.mean(dim=0),
        torch.zeros_like(action_codes[0]),
        rtol=0.0,
        atol=5e-8,
    )
    torch.testing.assert_close(
        model.prediction_projector[-1].weight,
        torch.zeros_like(model.prediction_projector[-1].weight),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.history_cell.content_gain_head.weight,
        torch.zeros_like(model.history_cell.content_gain_head.weight),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.history_cell.momentum_correction_head.weight,
        torch.zeros_like(model.history_cell.momentum_correction_head.weight),
        rtol=0.0,
        atol=0.0,
    )

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
        atol=1e-5,
    )
    torch.testing.assert_close(
        output.trajectory_latents,
        expected_future,
        rtol=0.0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        output.all_six_trajectory_innovations,
        torch.zeros_like(output.all_six_trajectory_innovations),
        rtol=0.0,
        atol=1e-5,
    )

    generator = torch.Generator().manual_seed(243)
    probe_content = F.normalize(
        torch.randn(2, atom_count, token_count, dim, generator=generator),
        dim=-1,
    )
    probe_momentum = torch.randn(
        2,
        atom_count,
        token_count,
        dim,
        generator=generator,
    )
    tangent = _tangent_projection(
        probe_content,
        probe_momentum,
        epsilon=model.config.normalization_epsilon,
    )
    torch.testing.assert_close(
        (probe_content * tangent).sum(dim=-1),
        torch.zeros_like(tangent[..., 0]),
        rtol=0.0,
        atol=1e-6,
    )


def test_predict_before_observe_order_and_future_rgb_cannot_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    history, past, future, future_rgb = _batch()
    events: list[str] = []
    original_transition = model._transition_step
    original_observe = model._observe

    def transition(
        content: torch.Tensor,
        momentum: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert content.ndim == 4 and momentum.shape == content.shape
        assert action.ndim == 1 and action.dtype == torch.long
        events.append("T")
        return original_transition(content, momentum, action)

    def observe(
        prior_content: torch.Tensor,
        prior_momentum: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert prior_content.ndim == 4
        assert prior_momentum.shape == prior_content.shape
        assert observation.ndim == 3
        events.append("U")
        return original_observe(prior_content, prior_momentum, observation)

    initialization_handle = model.initial_belief.register_forward_pre_hook(
        lambda _module, _args: events.append("I")
    )
    monkeypatch.setattr(model, "_transition_step", transition)
    monkeypatch.setattr(model, "_observe", observe)
    base = model(history, past, future, future_rgb)
    initialization_handle.remove()
    assert events == ["I", "T", "U", "T", "U", "T", "T", "T", "T"]

    changed_after_e0 = history.clone()
    changed_after_e0[:, 1:] = torch.flip(
        changed_after_e0[:, 1:],
        dims=(1, 3, 4),
    )
    after_e0 = model(changed_after_e0, past, future, future_rgb)
    torch.testing.assert_close(
        base.observed_prior_latents[:, :, 0],
        after_e0.observed_prior_latents[:, :, 0],
        rtol=0.0,
        atol=0.0,
    )

    changed_e2 = history.clone()
    changed_e2[:, 2] = torch.flip(changed_e2[:, 2], dims=(2, 3))
    after_e1 = model(changed_e2, past, future, future_rgb)
    torch.testing.assert_close(
        base.observed_prior_latents,
        after_e1.observed_prior_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(base.trajectory_latents, after_e1.trajectory_latents)

    other_targets = model(history, past, future, torch.flip(future_rgb, dims=(1, 3)))
    torch.testing.assert_close(
        base.trajectory_latents,
        other_targets.trajectory_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert base.target_latents is not None and other_targets.target_latents is not None
    assert not torch.equal(base.target_latents, other_targets.target_latents)


def test_belief_is_only_qv_and_opened_state_uses_history_and_action() -> None:
    model = _model().eval()
    _open_heads(model)
    history, past, future, _future_rgb = _batch()
    first = model(history, past, future)
    content, momentum = model._unpack_belief(first.belief_latents)
    repacked = model._pack_belief(content, momentum)
    torch.testing.assert_close(repacked, first.belief_latents, rtol=0.0, atol=0.0)
    assert set(inspect.signature(model._transition_step).parameters) == {
        "content",
        "momentum",
        "action_indices",
    }
    assert set(inspect.signature(model._rollout_future).parameters) == {
        "belief_latents",
        "future_actions",
    }

    changed_history = history.clone()
    changed_history[:, :2] = torch.flip(
        changed_history[:, :2],
        dims=(1, 3, 4),
    )
    changed_past = torch.flip(past, dims=(1,))
    second = model(changed_history, changed_past, future)
    torch.testing.assert_close(
        F.normalize(first.history_latents[:, 2], dim=-1),
        F.normalize(second.history_latents[:, 2], dim=-1),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(first.belief_latents, second.belief_latents)
    assert not torch.equal(first.trajectory_latents, second.trajectory_latents)

    wrong_future = (future + 1) % len(model.action_vocabulary)
    wrong_atoms = model.predict_trajectory_atoms_from_belief(
        first.belief_latents,
        wrong_future,
    )
    hold_atoms = model.predict_trajectory_atoms_from_belief(
        first.belief_latents,
        torch.full_like(future, model.action_vocabulary.index("hold")),
    )
    assert wrong_atoms.shape == hold_atoms.shape == first.trajectory_latents.shape
    assert not torch.equal(first.trajectory_latents, wrong_atoms)
    assert bool(torch.isfinite(hold_atoms).all())


def test_mode_rows_only_permute_equal_mass_trajectory_atoms() -> None:
    model = _model().eval()
    _open_heads(model)
    permuted = deepcopy(model)
    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    with torch.no_grad():
        permuted.initial_belief.mode_embedding.weight.copy_(
            model.initial_belief.mode_embedding.weight[permutation]
        )
    history, past, future, _future_rgb = _batch()
    original = model(history, past, future)
    reordered = permuted(history, past, future)
    torch.testing.assert_close(
        reordered.trajectory_latents,
        original.trajectory_latents[:, permutation],
    )
    torch.testing.assert_close(
        reordered.predicted_latents,
        original.predicted_latents,
    )


def test_realized_innovations_loss_arithmetic_target_and_optimizer_contract() -> None:
    model = _model().train()
    _open_heads(model)
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    online = F.normalize(
        output.history_latents,
        dim=-1,
        eps=model.config.normalization_epsilon,
    )
    expected_observed = torch.stack(
        (
            output.observed_prior_latents[:, :, 0] - online[:, None, 0],
            output.observed_prior_latents[:, :, 1] - online[:, None, 1],
        ),
        dim=2,
    )
    torch.testing.assert_close(
        output.all_six_trajectory_innovations[:, :, :2],
        expected_observed,
    )
    anchor, _momentum = model._unpack_belief(output.belief_latents)
    expected_future = torch.cat(
        (
            output.trajectory_latents[:, :, :1] - anchor[:, :, None],
            output.trajectory_latents[:, :, 1:]
            - output.trajectory_latents[:, :, :-1],
        ),
        dim=2,
    )
    torch.testing.assert_close(output.trajectory_innovations, expected_future)
    torch.testing.assert_close(
        output.all_six_trajectory_innovations[:, :, 2:],
        expected_future,
    )

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
    _fh, _fj, cumulative = trajectory_energy_score(
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

    groups = _parameter_groups(model)
    flattened = [parameter for values in groups.values() for parameter in values]
    assert set(groups) == {"encoder", "history", "predictor"}
    assert len({id(parameter) for parameter in flattened}) == len(flattened)
    assert {id(parameter) for parameter in flattened} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    sum(losses.values()).backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_zero_heads_stage_factors_then_opened_prediction_reaches_every_path() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    score = _isolated_prediction_score(
        model,
        history,
        past,
        future,
        future_rgb,
    )
    heads = (
        model.prediction_projector[-1].weight,
        model.history_cell.content_gain_head.weight,
        model.history_cell.momentum_correction_head.weight,
    )
    staged = (
        model.future_cell.spatial_block.attn.in_proj_weight,
        model.history_cell.spatial_block.attn.in_proj_weight,
        model.action_embedding.weight,
        model.future_spatial_refiner.tower[-1].weight,
        model.initial_belief.mode_embedding.weight,
    )
    gradients = torch.autograd.grad(
        score,
        (*heads, *staged),
        allow_unused=True,
    )
    for gradient in gradients[: len(heads)]:
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.norm()) > 0.0
    for gradient in gradients[len(heads) :]:
        if gradient is not None:
            torch.testing.assert_close(
                gradient,
                torch.zeros_like(gradient),
                rtol=0.0,
                atol=1e-7,
            )

    _open_heads(model)
    opened_score = _isolated_prediction_score(
        model,
        history,
        past,
        future,
        future_rgb,
    )
    encoder_parameters = tuple(model.encoder.parameters())
    opened_parameters = (*encoder_parameters, *heads, *staged)
    opened_gradients = torch.autograd.grad(
        opened_score,
        opened_parameters,
        allow_unused=True,
    )
    encoder_gradients = opened_gradients[: len(encoder_parameters)]
    active_encoder_gradients = [
        gradient for gradient in encoder_gradients if gradient is not None
    ]
    assert active_encoder_gradients
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in active_encoder_gradients
    )
    assert sum(float(gradient.norm()) for gradient in active_encoder_gradients) > 0.0
    for gradient in opened_gradients[len(encoder_parameters) :]:
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.norm()) > 0.0
