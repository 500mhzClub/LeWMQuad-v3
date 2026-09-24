from __future__ import annotations

from copy import deepcopy
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_trajectory_h4_jepa_v1 import (
    CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA,
    CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig,
    GO2_H4_PRIMITIVE_VOCABULARY,
    trajectory_energy_score,
    weighted_pairwise_spread,
    weighted_spherical_centroid,
    weighted_trajectory_energy_score,
)
from lewm.models.go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    fixed_teacher_local_innovations,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA:
    torch.manual_seed(271)
    return CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA(
        config=CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig(
            image_size=8,
            patch_size=2,
            feature_dim=20,
            encoder_depth=1,
            encoder_heads=4,
            recurrent_spatial_heads=4,
            cross_attention_heads=4,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(272)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _open_head(
    model: CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA,
    *,
    scale: float = 0.03,
) -> None:
    with torch.no_grad():
        model.prediction_projector[-1].weight.copy_(
            scale * torch.eye(model.config.feature_dim)
        )


def _assert_finite_nonzero(value: torch.Tensor | None) -> None:
    assert value is not None
    assert bool(torch.isfinite(value).all())
    assert float(value.norm()) > 0.0


HOLD_ACTION = GO2_H4_PRIMITIVE_VOCABULARY.index("hold")


def _isolated_score(
    model: CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA,
    history: torch.Tensor,
    past: torch.Tensor,
    future: torch.Tensor,
    future_rgb: torch.Tensor,
) -> torch.Tensor:
    output = model(history, past, future)
    teacher_history = model._encode_fixed_teacher_history(history)
    target = model.encode_target(future_rgb)
    target_innovations = torch.cat(
        (
            teacher_history[:, 1:] - teacher_history[:, :-1],
            fixed_teacher_local_innovations(teacher_history[:, 2], target),
        ),
        dim=1,
    )
    _lh, _lj, local = trajectory_energy_score(
        output.all_six_trajectory_innovations,
        target_innovations,
    )
    _fh, _fj, future_score = weighted_trajectory_energy_score(
        output.trajectory_latents,
        target,
        output.posterior_probabilities,
    )
    return 0.5 * local.mean() + 0.5 * future_score.mean()


def test_q_probability_state_pack_centering_and_update_zero_contract() -> None:
    with pytest.raises(ValueError, match="exactly 1e-6"):
        CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig(
            normalization_epsilon=1e-5
        )

    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    online = F.normalize(
        output.history_latents,
        dim=-1,
        eps=model.config.normalization_epsilon,
    )
    atom_count = model.config.trajectory_atom_count
    uniform = torch.full((2, atom_count), 0.25)
    content, probabilities = model._unpack_belief(output.belief_latents)

    assert content.shape == (2, 4, 16, 20)
    assert probabilities.shape == (2, 4)
    assert output.belief_latents.shape == (2, 5, 16, 20)
    torch.testing.assert_close(
        content,
        online[:, 2, None].expand_as(content),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(probabilities, uniform, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output.posterior_history,
        uniform[:, None].expand_as(output.posterior_history),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.posterior_probabilities,
        model.posterior_probabilities_from_belief(output.belief_latents),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.final_hidden_particles,
        output.posterior_probabilities,
        rtol=0.0,
        atol=0.0,
    )

    carrier = output.belief_latents[:, 4].reshape(2, -1)
    torch.testing.assert_close(carrier[:, :4], uniform, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        carrier[:, 4:],
        torch.zeros_like(carrier[:, 4:]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model._pack_belief(content, probabilities),
        output.belief_latents,
        rtol=0.0,
        atol=0.0,
    )
    bad_padding = output.belief_latents.clone()
    bad_padding[:, 4].reshape(2, -1)[:, 4] = 1.0
    with pytest.raises(ValueError, match="padding"):
        model._unpack_belief(bad_padding)
    with pytest.raises(ValueError, match="strictly positive"):
        model._pack_belief(content, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1))

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
        output.all_six_trajectory_innovations,
        torch.zeros_like(output.all_six_trajectory_innovations),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.prediction_projector[-1].weight,
        torch.zeros_like(model.prediction_projector[-1].weight),
        rtol=0.0,
        atol=0.0,
    )
    centered_modes = (
        model.initial_belief.mode_embedding.weight
        - model.initial_belief.mode_embedding.weight.mean(dim=0, keepdim=True)
    )
    torch.testing.assert_close(
        centered_modes.mean(dim=0),
        torch.zeros_like(centered_modes[0]),
        rtol=0.0,
        atol=5e-8,
    )
    torch.testing.assert_close(
        model._centered_action_codes().mean(dim=0),
        torch.zeros(model.config.feature_dim),
        rtol=0.0,
        atol=5e-8,
    )
    assert isinstance(model.history_observation_norm, torch.nn.Identity)
    assert isinstance(model.history_cell, torch.nn.Identity)
    assert isinstance(model.history_spatial_refiner, torch.nn.Identity)
    forbidden = ("memory", "momentum", "incoming_increment")
    assert not any(
        token in name
        for name in model.state_dict()
        for token in forbidden
    )


def test_exact_likelihood_two_updates_and_six_prior_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    errors = torch.tensor([0.25, 1.0, 2.25, 4.0])
    prior = torch.zeros(1, 4, 16, 20)
    prior[0, :, :, 0] = errors.sqrt()[:, None]
    observation = torch.zeros(1, 16, 20)
    updated, observed_errors, likelihoods = model._evidence_update(
        probabilities,
        prior,
        observation,
    )
    expected_likelihoods = torch.exp(
        -errors / (errors.mean() + 1e-6)
    )[None]
    expected = probabilities * expected_likelihoods
    expected = expected / expected.sum(dim=1, keepdim=True)
    torch.testing.assert_close(observed_errors, errors[None], rtol=0.0, atol=0.0)
    torch.testing.assert_close(likelihoods, expected_likelihoods)
    torch.testing.assert_close(updated, expected)
    assert float(updated[0, 0] / updated[0, 3]) > float(
        probabilities[0, 0] / probabilities[0, 3]
    )

    equal_prior = prior[:, :1].expand_as(prior)
    same, equal_error, equal_likelihood = model._evidence_update(
        probabilities,
        equal_prior,
        observation,
    )
    torch.testing.assert_close(same, probabilities, rtol=0.0, atol=2e-8)
    assert torch.equal(equal_error, equal_error[:, :1].expand_as(equal_error))
    assert torch.equal(
        equal_likelihood,
        equal_likelihood[:, :1].expand_as(equal_likelihood),
    )

    events: list[str] = []
    transition_actions: list[torch.Tensor] = []
    evidence_priors: list[torch.Tensor] = []
    evidence_observations: list[torch.Tensor] = []
    original_initializer = model.initial_belief.forward
    original_transition = model._transition_step
    original_update = model._evidence_update
    original_assimilate = model._assimilate_observation

    def initialize(*args: object, **kwargs: object) -> object:
        events.append("I")
        return original_initializer(*args, **kwargs)

    def transition(content: torch.Tensor, actions: torch.Tensor) -> object:
        events.append("T")
        transition_actions.append(actions.detach().clone())
        return original_transition(content, actions)

    def update(
        weights: torch.Tensor,
        prior_content: torch.Tensor,
        destination: torch.Tensor,
    ) -> object:
        events.append("U")
        evidence_priors.append(prior_content.detach().clone())
        evidence_observations.append(destination.detach().clone())
        return original_update(weights, prior_content, destination)

    def assimilate(prior_content: torch.Tensor, destination: torch.Tensor) -> object:
        events.append("A")
        return original_assimilate(prior_content, destination)

    monkeypatch.setattr(model.initial_belief, "forward", initialize)
    monkeypatch.setattr(model, "_transition_step", transition)
    monkeypatch.setattr(model, "_evidence_update", update)
    monkeypatch.setattr(model, "_assimilate_observation", assimilate)
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    assert events == ["I", "T", "U", "A", "T", "U", "A", "T", "T", "T", "T"]
    assert len(evidence_priors) == 2
    assert len(evidence_observations) == 2
    assert len(transition_actions) == 6
    for step in range(2):
        torch.testing.assert_close(
            evidence_priors[step],
            output.observed_prior_latents[:, :, step],
            rtol=0.0,
            atol=0.0,
        )
        factual = F.normalize(
            output.history_latents[:, step + 1],
            dim=-1,
            eps=model.config.normalization_epsilon,
        )
        torch.testing.assert_close(
            evidence_observations[step], factual, rtol=0.0, atol=0.0
        )
    expected_actions = [past[:, 0], past[:, 1], *(future[:, step] for step in range(4))]
    assert all(torch.equal(actual, expected) for actual, expected in zip(transition_actions, expected_actions, strict=True))


def test_weighted_score_centroid_spread_and_uniform_reduction() -> None:
    generator = torch.Generator().manual_seed(273)
    atoms = torch.randn(2, 4, 3, 2, 5, generator=generator)
    target = torch.randn(2, 3, 2, 5, generator=generator)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3]])
    horizon, joint, combined = weighted_trajectory_energy_score(
        atoms,
        target,
        weights,
    )
    distance = torch.linalg.vector_norm(
        atoms - target[:, None], dim=(-2, -1)
    ) / math.sqrt(2.0)
    pair_distance = torch.linalg.vector_norm(
        atoms[:, :, None] - atoms[:, None, :], dim=(-2, -1)
    ) / math.sqrt(2.0)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    expected_horizon = (
        (distance * weights[:, :, None]).sum(dim=1)
        - 0.5
        * (pair_distance * pair_weights[:, :, :, None]).sum(dim=(1, 2))
    )
    flat_atoms = atoms.reshape(2, 4, 6, 5)
    flat_target = target.reshape(2, 6, 5)
    expected_joint = (
        (
            torch.linalg.vector_norm(
                flat_atoms - flat_target[:, None], dim=(-2, -1)
            )
            / math.sqrt(6.0)
            * weights
        ).sum(dim=1)
        - 0.5
        * (
            torch.linalg.vector_norm(
                flat_atoms[:, :, None] - flat_atoms[:, None, :],
                dim=(-2, -1),
            )
            / math.sqrt(6.0)
            * pair_weights
        ).sum(dim=(1, 2))
    )
    torch.testing.assert_close(horizon, expected_horizon)
    torch.testing.assert_close(joint, expected_joint)
    torch.testing.assert_close(
        combined,
        0.5 * expected_joint + 0.5 * expected_horizon.mean(dim=1),
    )

    uniform = torch.full((2, 4), 0.25)
    weighted_uniform = weighted_trajectory_energy_score(atoms, target, uniform)
    inherited_uniform = trajectory_energy_score(atoms, target)
    for actual, expected in zip(weighted_uniform, inherited_uniform, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    one_hot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(2, -1)
    one_horizon, one_joint, _one_combined = weighted_trajectory_energy_score(
        atoms,
        target,
        one_hot,
    )
    torch.testing.assert_close(one_horizon, distance[:, 2])
    selected_flat_distance = torch.linalg.vector_norm(
        flat_atoms[:, 2] - flat_target,
        dim=(-2, -1),
    ) / math.sqrt(6.0)
    torch.testing.assert_close(one_joint, selected_flat_distance)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = weighted_trajectory_energy_score(
        atoms[:, permutation],
        target,
        weights[:, permutation],
    )
    for actual, expected in zip(permuted, (horizon, joint, combined), strict=True):
        torch.testing.assert_close(actual, expected)
    weights_only = weighted_trajectory_energy_score(
        atoms,
        target,
        weights[:, permutation],
    )[2]
    assert not torch.allclose(weights_only, combined)

    centroid = weighted_spherical_centroid(atoms, weights)
    expected_centroid = F.normalize(
        (atoms * weights[:, :, None, None, None]).sum(dim=1),
        dim=-1,
        eps=1e-6,
    )
    torch.testing.assert_close(centroid, expected_centroid)
    spread = weighted_pairwise_spread(atoms, weights)
    torch.testing.assert_close(
        spread,
        (pair_distance * pair_weights[:, :, :, None]).sum(dim=(1, 2)),
    )

    differentiable_atoms = atoms.detach().clone().requires_grad_(True)
    logits = torch.randn(2, 4, generator=generator, requires_grad=True)
    differentiable_weights = logits.softmax(dim=1)
    differentiable_score = weighted_trajectory_energy_score(
        differentiable_atoms,
        target,
        differentiable_weights,
    )[2].mean()
    differentiable_score.backward()
    _assert_finite_nonzero(differentiable_atoms.grad)
    _assert_finite_nonzero(logits.grad)


def test_probabilities_only_weight_readout_and_freeze_through_future() -> None:
    model = _model().eval()
    _open_head(model)
    generator = torch.Generator().manual_seed(274)
    content = F.normalize(
        torch.randn(2, 4, 16, 20, generator=generator),
        dim=-1,
    )
    first_weights = torch.tensor(
        [[0.55, 0.20, 0.15, 0.10], [0.10, 0.15, 0.20, 0.55]]
    )
    second_weights = first_weights.flip(1).contiguous()
    future = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    first_atoms, first_returned = (
        model.predict_trajectory_atoms_and_probabilities_from_belief(
            model._pack_belief(content, first_weights),
            future,
        )
    )
    second_atoms, second_returned = (
        model.predict_trajectory_atoms_and_probabilities_from_belief(
            model._pack_belief(content, second_weights),
            future,
        )
    )
    torch.testing.assert_close(first_atoms, second_atoms, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_returned, first_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_returned, second_weights, rtol=0.0, atol=0.0)
    assert not torch.allclose(
        weighted_spherical_centroid(first_atoms, first_weights),
        weighted_spherical_centroid(first_atoms, second_weights),
    )

    changed_future = future.clone()
    changed_future[:, 0] = (changed_future[:, 0] + 1) % 9
    changed_atoms, changed_weights = (
        model.predict_trajectory_atoms_and_probabilities_from_belief(
            model._pack_belief(content, first_weights),
            changed_future,
        )
    )
    assert not torch.allclose(first_atoms, changed_atoms)
    torch.testing.assert_close(changed_weights, first_weights, rtol=0.0, atol=0.0)
    hold = torch.full_like(future, HOLD_ACTION)
    hold_atoms, hold_weights = (
        model.predict_trajectory_atoms_and_probabilities_from_belief(
            model._pack_belief(content, first_weights),
            hold,
        )
    )
    assert bool(torch.isfinite(hold_atoms).all())
    torch.testing.assert_close(hold_weights, first_weights, rtol=0.0, atol=0.0)

    signature = tuple(inspect.signature(model._transition_step).parameters)
    assert signature == ("content", "action_indices")
    repeated = content[:1].expand(9, -1, -1, -1).contiguous()
    _next, projected, _realized = model._transition_step(
        repeated,
        torch.arange(9, dtype=torch.long),
    )
    torch.testing.assert_close(
        projected.mean(dim=0),
        torch.zeros_like(projected[0]),
        rtol=0.0,
        atol=2e-7,
    )


def test_history_is_causal_changes_only_posterior_and_future_rgb_never_leaks() -> None:
    model = _model().eval()
    _open_head(model, scale=0.08)
    history, past, future, future_rgb = _batch()
    original = model(history, past, future)

    later_changed = history.clone()
    later_changed[:, 1:] = later_changed[:, 1:].flip(0) + 0.7
    later = model(later_changed, past, future)
    torch.testing.assert_close(
        original.observed_prior_latents[:, :, 0],
        later.observed_prior_latents[:, :, 0],
        rtol=0.0,
        atol=0.0,
    )
    e2_changed = history.clone()
    e2_changed[:, 2] = e2_changed[:, 2].flip(0) - 0.4
    changed_e2 = model(e2_changed, past, future)
    torch.testing.assert_close(
        original.observed_prior_latents[:, :, 1],
        changed_e2.observed_prior_latents[:, :, 1],
        rtol=0.0,
        atol=0.0,
    )

    earlier_changed = history.clone()
    earlier_changed[:, :2] = earlier_changed[:, :2].flip(0) + 0.35
    changed = model(earlier_changed, past.flip(1), future)
    original_q, _original_w = model._unpack_belief(original.belief_latents)
    changed_q, _changed_w = model._unpack_belief(changed.belief_latents)
    torch.testing.assert_close(original_q, changed_q, rtol=0.0, atol=0.0)
    assert not torch.allclose(
        original.posterior_probabilities,
        changed.posterior_probabilities,
    )
    torch.testing.assert_close(
        original.trajectory_latents,
        changed.trajectory_latents,
        rtol=0.0,
        atol=0.0,
    )
    original_score = weighted_trajectory_energy_score(
        original.trajectory_latents,
        model.encode_target(future_rgb),
        original.posterior_probabilities,
    )[2]
    changed_score = weighted_trajectory_energy_score(
        changed.trajectory_latents,
        model.encode_target(future_rgb),
        changed.posterior_probabilities,
    )[2]
    assert bool((original_score != changed_score).any())

    reversed_output = model(history[:, [1, 0, 2]], past[:, [1, 0]], future)
    reset_output = model(
        history[:, 2:3].expand(-1, 3, -1, -1, -1).contiguous(),
        torch.full_like(past, HOLD_ACTION),
        future,
    )
    assert reversed_output.posterior_probabilities.data_ptr() != (
        original.posterior_probabilities.data_ptr()
    )
    assert reset_output.posterior_probabilities.data_ptr() != (
        original.posterior_probabilities.data_ptr()
    )
    repeat = model(history, past, future)
    torch.testing.assert_close(
        repeat.posterior_probabilities,
        original.posterior_probabilities,
        rtol=0.0,
        atol=0.0,
    )

    first_target = model(history, past, future, future_rgb)
    second_target = model(history, past, future, future_rgb.flip(1))
    torch.testing.assert_close(
        first_target.trajectory_latents,
        second_target.trajectory_latents,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first_target.posterior_probabilities,
        second_target.posterior_probabilities,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.allclose(first_target.target_latents, second_target.target_latents)


def test_exact_losses_groups_one_step_and_gradient_routes() -> None:
    model = _model().train()
    _open_head(model)
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    output.posterior_probabilities.retain_grad()
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
    online_history = F.normalize(
        output.history_latents,
        dim=-1,
        eps=model.config.normalization_epsilon,
    )
    expected_alignment = (
        (online_history - teacher_history).square().sum(dim=-1).mean()
    )
    target_innovations = torch.cat(
        (
            teacher_history[:, 1:] - teacher_history[:, :-1],
            fixed_teacher_local_innovations(teacher_history[:, 2], target),
        ),
        dim=1,
    )
    _lh, _lj, expected_local = trajectory_energy_score(
        output.all_six_trajectory_innovations,
        target_innovations,
    )
    _fh, _fj, expected_future = weighted_trajectory_energy_score(
        output.trajectory_latents,
        target,
        output.posterior_probabilities,
    )
    torch.testing.assert_close(losses["history_teacher_alignment"], expected_alignment)
    torch.testing.assert_close(
        losses["half_all_six_factual_local_innovation_energy_score"],
        0.5 * expected_local.mean(),
    )
    torch.testing.assert_close(
        losses["half_open_loop_future_cumulative_trajectory_energy_score"],
        0.5 * expected_future.mean(),
    )

    groups = _parameter_groups(model)
    assert set(groups) == {"encoder", "history", "predictor"}
    group_ids = [{id(parameter) for parameter in values} for values in groups.values()]
    assert all(group_ids[left].isdisjoint(group_ids[right]) for left in range(3) for right in range(left + 1, 3))
    assert set().union(*group_ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["encoder"], "lr": 1e-4},
            {"params": groups["history"], "lr": 3e-4},
            {"params": groups["predictor"], "lr": 3e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    before = model.prediction_projector[-1].weight.detach().clone()
    sum(losses.values()).backward()
    _assert_finite_nonzero(output.posterior_probabilities.grad)
    routes = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief.mode_embedding.weight.grad,
        model.initial_belief.spatial_embedding.weight.grad,
        model.action_embedding.weight.grad,
        model.future_cell.visual_path[1].weight.grad,
        model.future_cell.spatial_block.attn.in_proj_weight.grad,
        model.future_spatial_refiner.tower[-1].weight.grad,
        model.prediction_projector[-1].weight.grad,
    )
    for gradient in routes:
        _assert_finite_nonzero(gradient)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer.step()
    assert not torch.equal(before, model.prediction_projector[-1].weight)


def test_zero_head_stages_then_opened_score_reaches_posterior_experts() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    head = model.prediction_projector[-1].weight
    upstream = (
        model.initial_belief.mode_embedding.weight,
        model.initial_belief.spatial_embedding.weight,
        model.action_embedding.weight,
        model.future_cell.visual_path[1].weight,
        model.future_cell.spatial_block.attn.in_proj_weight,
        model.future_spatial_refiner.tower[-1].weight,
    )
    staged = torch.autograd.grad(
        _isolated_score(model, history, past, future, future_rgb),
        (head, *upstream),
        allow_unused=True,
    )
    _assert_finite_nonzero(staged[0])
    for gradient in staged[1:]:
        if gradient is not None:
            torch.testing.assert_close(
                gradient,
                torch.zeros_like(gradient),
                rtol=0.0,
                atol=1e-7,
            )

    _open_head(model)
    output = model(history, past, future)
    output.posterior_probabilities.retain_grad()
    target = model.encode_target(future_rgb)
    score = weighted_trajectory_energy_score(
        output.trajectory_latents,
        target,
        output.posterior_probabilities,
    )[2].mean()
    encoder_parameters = tuple(model.encoder.parameters())
    opened = torch.autograd.grad(
        score,
        (*encoder_parameters, head, *upstream, output.posterior_probabilities),
        allow_unused=True,
    )
    encoder_gradients = [
        gradient
        for gradient in opened[: len(encoder_parameters)]
        if gradient is not None
    ]
    assert encoder_gradients
    assert all(bool(torch.isfinite(value).all()) for value in encoder_gradients)
    assert sum(float(value.norm()) for value in encoder_gradients) > 0.0
    for gradient in opened[len(encoder_parameters) :]:
        _assert_finite_nonzero(gradient)
    mode_gradient = opened[len(encoder_parameters) + 1]
    assert mode_gradient is not None
    assert bool((mode_gradient.norm(dim=1) > 0.0).all())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_joint_expert_mode_and_probability_permutation_is_distribution_invariant() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    target = model.encode_target(future_rgb)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_model = deepcopy(model)
    with torch.no_grad():
        permuted_model.initial_belief.mode_embedding.weight.copy_(
            model.initial_belief.mode_embedding.weight[permutation]
        )
    permuted = permuted_model(history, past, future)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        permuted.trajectory_latents[:, inverse],
        output.trajectory_latents,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        permuted.posterior_probabilities[:, inverse],
        output.posterior_probabilities,
        rtol=1e-6,
        atol=1e-7,
    )
    original_score = weighted_trajectory_energy_score(
        output.trajectory_latents,
        target,
        output.posterior_probabilities,
    )[2]
    permuted_score = weighted_trajectory_energy_score(
        permuted.trajectory_latents,
        target,
        permuted.posterior_probabilities,
    )[2]
    torch.testing.assert_close(permuted_score, original_score, rtol=1e-6, atol=1e-7)
