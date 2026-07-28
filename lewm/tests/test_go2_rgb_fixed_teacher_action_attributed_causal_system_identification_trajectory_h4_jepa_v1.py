from __future__ import annotations

from copy import deepcopy
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_action_attributed_causal_system_identification_trajectory_h4_jepa_v1 import (
    ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA,
    ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA:
    torch.manual_seed(251)
    return ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA(
        config=ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig(
            image_size=8,
            patch_size=2,
            feature_dim=20,
            encoder_depth=1,
            encoder_heads=4,
            recurrent_spatial_heads=4,
            cross_attention_heads=4,
            system_identification_dim=16,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(252)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _open_head(
    model: ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA,
    *,
    scale: float = 0.02,
) -> None:
    with torch.no_grad():
        model.prediction_projector[-1].weight.copy_(
            scale * torch.eye(model.config.feature_dim)
        )


def _isolated_prediction_score(
    model: ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA,
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


def _assert_finite_nonzero(gradient: torch.Tensor | None) -> None:
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 0.0


def test_matrix_pack_centering_rank_and_update_zero_contract() -> None:
    with pytest.raises(ValueError, match="exactly 16"):
        ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig(
            system_identification_dim=8
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
    token_count = model.spatial_token_count
    dim = model.config.feature_dim
    memory_dim = model.config.system_identification_dim
    assert model.history_observation_norm.elementwise_affine is False
    assert model.history_observation_norm.eps == 1e-5
    assert model.history_cell.bias is None
    assert model.history_cell.weight.shape == (memory_dim, dim)
    assert model.history_spatial_refiner.key_projection.bias is None
    assert model.history_spatial_refiner.key_projection.weight.shape == (
        memory_dim,
        dim,
    )
    assert model.history_spatial_refiner.memory_projection.bias is None
    assert model.history_spatial_refiner.memory_projection.weight.shape == (
        dim,
        memory_dim * memory_dim,
    )
    assert model.prediction_projector[-1].bias is None

    initial_content, initial_memory = model.initial_belief(online[:, 0])
    torch.testing.assert_close(
        initial_content,
        online[:, 0, None].expand_as(initial_content),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        initial_memory,
        torch.zeros_like(initial_memory),
        rtol=0.0,
        atol=0.0,
    )
    content, memory = model._unpack_belief(output.belief_latents)
    assert content.shape == (2, atom_count, token_count, dim)
    assert memory.shape == (2, atom_count, memory_dim, memory_dim)
    repacked = model._pack_belief(content, memory)
    torch.testing.assert_close(repacked, output.belief_latents, rtol=0.0, atol=0.0)
    assert repacked.shape == (2, 2 * atom_count, token_count, dim)
    carriers = repacked[:, atom_count:].reshape(2, atom_count, -1)
    stored = memory.reshape(2, atom_count, -1)
    torch.testing.assert_close(
        carriers[..., : memory_dim * memory_dim], stored, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        carriers[..., memory_dim * memory_dim :],
        torch.zeros_like(carriers[..., memory_dim * memory_dim :]),
        rtol=0.0,
        atol=0.0,
    )
    assert bool(
        (torch.linalg.matrix_rank(memory.float(), atol=1e-5, rtol=1e-5) <= 2).all()
    )

    mode_rows = model.initial_belief.mode_embedding.weight
    torch.testing.assert_close(
        (mode_rows - mode_rows.mean(dim=0, keepdim=True)).mean(dim=0),
        torch.zeros_like(mode_rows[0]),
        rtol=0.0,
        atol=5e-8,
    )
    for table in (model._centered_action_codes(), model._centered_action_keys()):
        torch.testing.assert_close(
            table.mean(dim=0),
            torch.zeros_like(table[0]),
            rtol=0.0,
            atol=5e-8,
        )
    torch.testing.assert_close(
        model._memory_response(torch.zeros_like(content)),
        torch.zeros(2, atom_count, memory_dim),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.prediction_projector[-1].weight,
        torch.zeros_like(model.prediction_projector[-1].weight),
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
        output.observed_prior_latents, expected_observed, rtol=0.0, atol=1e-5
    )
    torch.testing.assert_close(
        output.trajectory_latents, expected_future, rtol=0.0, atol=1e-5
    )
    torch.testing.assert_close(
        output.all_six_trajectory_innovations,
        torch.zeros_like(output.all_six_trajectory_innovations),
        rtol=0.0,
        atol=1e-5,
    )


def test_fixed_outer_product_write_and_exact_action_attribution() -> None:
    model = _model().eval()
    atom_count = model.config.trajectory_atom_count
    memory_dim = model.config.system_identification_dim
    generator = torch.Generator().manual_seed(253)
    innovation = torch.randn(
        2,
        atom_count,
        model.spatial_token_count,
        model.config.feature_dim,
        generator=generator,
    )
    memory = torch.zeros(2, atom_count, memory_dim, memory_dim)
    actions = torch.tensor((0, 3), dtype=torch.long)
    response = model._memory_response(innovation)
    selected_keys = model._centered_action_keys().index_select(0, actions)
    expected = memory + (
        response[..., :, None] * selected_keys[:, None, None, :]
    ) / math.sqrt(float(memory_dim))
    actual = model._write_memory(memory, innovation, actions)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        model._write_memory(memory, torch.zeros_like(innovation), actions),
        memory,
        rtol=0.0,
        atol=0.0,
    )
    swapped = model._write_memory(memory, innovation, actions.flip(0))
    assert not torch.equal(actual, swapped)
    second_innovation = torch.randn(
        innovation.shape,
        generator=generator,
        dtype=innovation.dtype,
        device=innovation.device,
    )
    second_actions = torch.tensor((1, 4), dtype=torch.long)
    second_response = model._memory_response(second_innovation)
    second_keys = model._centered_action_keys().index_select(0, second_actions)
    assert float((response - second_response).detach().abs().max()) > 1e-3
    assert float((selected_keys - second_keys).detach().abs().max()) > 1e-3
    paired = model._write_memory(
        actual,
        second_innovation,
        second_actions,
    )
    keys_swapped = model._write_memory(
        model._write_memory(memory, innovation, second_actions),
        second_innovation,
        actions,
    )
    assert float((paired - keys_swapped).detach().abs().max()) > 1e-6
    assert set(inspect.signature(model._write_memory).parameters) == {
        "memory",
        "innovation",
        "action_indices",
    }


def test_predict_write_assimilate_order_and_prior_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    history, past, future, future_rgb = _batch()
    events: list[tuple[str, torch.Tensor | None]] = []
    original_transition = model._transition_step
    original_observe = model._observe
    original_write = model._write_memory

    def transition(
        content: torch.Tensor,
        memory: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        events.append(("T", action_indices.detach().clone()))
        return original_transition(content, memory, action_indices)

    def write(
        memory: torch.Tensor,
        innovation: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        events.append(("W", action_indices.detach().clone()))
        return original_write(memory, innovation, action_indices)

    def observe(
        prior_content: torch.Tensor,
        memory: torch.Tensor,
        observation: torch.Tensor,
        past_action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = original_observe(
            prior_content,
            memory,
            observation,
            past_action_indices,
        )
        torch.testing.assert_close(
            result[2],
            observation[:, None].expand_as(prior_content) - prior_content,
        )
        events.append(("A", past_action_indices.detach().clone()))
        return result

    def prove_action_free_context(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        assert len(args) == 3
        torch.testing.assert_close(
            args[2], torch.zeros_like(args[2]), rtol=0.0, atol=0.0
        )

    handle = model.initial_belief.register_forward_pre_hook(
        lambda _module, _args: events.append(("I", None))
    )
    context_handle = model.future_cell.register_forward_pre_hook(
        prove_action_free_context
    )
    monkeypatch.setattr(model, "_transition_step", transition)
    monkeypatch.setattr(model, "_write_memory", write)
    monkeypatch.setattr(model, "_observe", observe)
    base = model(history, past, future, future_rgb)
    handle.remove()
    context_handle.remove()
    assert [name for name, _value in events] == [
        "I",
        "T",
        "W",
        "A",
        "T",
        "W",
        "A",
        "T",
        "T",
        "T",
        "T",
    ]
    torch.testing.assert_close(events[1][1], past[:, 0])
    torch.testing.assert_close(events[2][1], past[:, 0])
    torch.testing.assert_close(events[3][1], past[:, 0])
    torch.testing.assert_close(events[4][1], past[:, 1])
    torch.testing.assert_close(events[5][1], past[:, 1])
    torch.testing.assert_close(events[6][1], past[:, 1])

    changed_after_e0 = history.clone()
    changed_after_e0[:, 1:] = torch.flip(changed_after_e0[:, 1:], dims=(1, 3, 4))
    after_e0 = model(changed_after_e0, past, future)
    torch.testing.assert_close(
        base.observed_prior_latents[:, :, 0],
        after_e0.observed_prior_latents[:, :, 0],
        rtol=0.0,
        atol=0.0,
    )
    changed_e2 = history.clone()
    changed_e2[:, 2] = torch.flip(changed_e2[:, 2], dims=(2, 3))
    after_e1 = model(changed_e2, past, future)
    torch.testing.assert_close(
        base.observed_prior_latents,
        after_e1.observed_prior_latents,
        rtol=0.0,
        atol=0.0,
    )


def test_memory_is_modulation_only_and_all_action_mean_delta_is_zero() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    content, memory = model._unpack_belief(output.belief_latents)
    action_count = len(model.action_vocabulary)
    repeated_content = content[:1].expand(action_count, -1, -1, -1)
    repeated_memory = memory[:1].expand(action_count, -1, -1, -1)
    actions = torch.arange(action_count, dtype=torch.long)
    _next, _same_memory, deltas, _realized = model._transition_step(
        repeated_content,
        repeated_memory,
        actions,
    )
    torch.testing.assert_close(
        deltas.mean(dim=0),
        torch.zeros_like(deltas[0]),
        rtol=0.0,
        atol=1e-7,
    )
    zero_memory = torch.zeros_like(memory)
    modulation = model._memory_modulation(memory)
    assert bool(((modulation > 0.0) & (modulation < 2.0)).all())
    torch.testing.assert_close(
        model._memory_modulation(zero_memory),
        torch.ones(
            memory.shape[0],
            memory.shape[1],
            model.config.feature_dim,
        ),
        rtol=0.0,
        atol=0.0,
    )
    _q_memory, _m_memory, with_memory, _r_memory = model._transition_step(
        content, memory, future[:, 0]
    )
    _q_base, _m_base, without_memory, _r_base = model._transition_step(
        content, zero_memory, future[:, 0]
    )
    assert not torch.equal(with_memory, without_memory)

    no_action = deepcopy(model)
    with torch.no_grad():
        no_action.future_spatial_refiner.tower[-1].weight.zero_()
    _q, _m, no_action_delta, _realized = no_action._transition_step(
        content, memory, future[:, 0]
    )
    torch.testing.assert_close(
        no_action_delta,
        torch.zeros_like(no_action_delta),
        rtol=0.0,
        atol=0.0,
    )


def test_opened_history_action_memory_freeze_and_future_target_isolation() -> None:
    model = _model().eval()
    _open_head(model)
    history, past, future, future_rgb = _batch()
    first = model(history, past, future, future_rgb)
    content, memory = model._unpack_belief(first.belief_latents)
    assert set(inspect.signature(model._transition_step).parameters) == {
        "content",
        "memory",
        "action_indices",
    }
    assert set(inspect.signature(model._rollout_future).parameters) == {
        "belief_latents",
        "future_actions",
    }

    changed_history = history.clone()
    changed_history[:, :2] = torch.flip(changed_history[:, :2], dims=(1, 3, 4))
    second = model(changed_history, past.flip(1), future)
    second_content, second_memory = model._unpack_belief(second.belief_latents)
    torch.testing.assert_close(content, second_content, rtol=0.0, atol=0.0)
    assert not torch.equal(memory, second_memory)
    assert not torch.equal(first.trajectory_latents, second.trajectory_latents)

    wrong_future = (future + 1) % len(model.action_vocabulary)
    wrong_atoms = model.predict_trajectory_atoms_from_belief(
        first.belief_latents, wrong_future
    )
    hold_atoms = model.predict_trajectory_atoms_from_belief(
        first.belief_latents,
        torch.full_like(future, model.action_vocabulary.index("hold")),
    )
    assert not torch.equal(first.trajectory_latents, wrong_atoms)
    assert bool(torch.isfinite(hold_atoms).all())

    _states, _deltas, _innovations, final_memory = model._rollout_future(
        first.belief_latents,
        future,
    )
    torch.testing.assert_close(final_memory, memory, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.final_hidden_particles, memory, rtol=0.0, atol=0.0
    )
    other_target = model(
        history,
        past,
        future,
        torch.flip(future_rgb, dims=(1, 3)),
    )
    torch.testing.assert_close(
        first.trajectory_latents,
        other_target.trajectory_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert first.target_latents is not None and other_target.target_latents is not None
    assert not torch.equal(first.target_latents, other_target.target_latents)


def test_mode_rows_only_permute_equal_mass_trajectory_atoms() -> None:
    model = _model().eval()
    _open_head(model)
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


def test_realized_innovations_loss_groups_and_one_joint_step() -> None:
    model = _model().train()
    _open_head(model)
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
        output.all_six_trajectory_innovations[:, :, :2], expected_observed
    )
    content, _memory = model._unpack_belief(output.belief_latents)
    expected_future = torch.cat(
        (
            output.trajectory_latents[:, :, :1] - content[:, :, None],
            output.trajectory_latents[:, :, 1:]
            - output.trajectory_latents[:, :, :-1],
        ),
        dim=2,
    )
    torch.testing.assert_close(output.trajectory_innovations, expected_future)
    torch.testing.assert_close(
        output.all_six_trajectory_innovations[:, :, 2:], expected_future
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
    target_path = torch.cat((teacher_history, target), dim=1)
    target_innovations = target_path[:, 1:] - target_path[:, :-1]
    _lh, _lj, local = trajectory_energy_score(
        output.all_six_trajectory_innovations, target_innovations
    )
    _fh, _fj, cumulative = trajectory_energy_score(
        output.trajectory_latents, target
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
    optimizer = torch.optim.AdamW(
        [{"params": values} for values in groups.values()], lr=3e-4
    )
    before = model.prediction_projector[-1].weight.detach().clone()
    sum(losses.values()).backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer.step()
    assert not torch.equal(before, model.prediction_projector[-1].weight)


def test_zero_head_stages_then_opened_score_reaches_every_learned_route() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    head = model.prediction_projector[-1].weight
    staged = (
        model.future_cell.spatial_block.attn.in_proj_weight,
        model.history_cell.weight,
        model.history_spatial_refiner.key_projection.weight,
        model.history_spatial_refiner.memory_projection.weight,
        model.action_embedding.weight,
        model.future_spatial_refiner.tower[-1].weight,
        model.initial_belief.mode_embedding.weight,
        model.initial_belief.spatial_embedding.weight,
    )
    score = _isolated_prediction_score(
        model, history, past, future, future_rgb
    )
    gradients = torch.autograd.grad(score, (head, *staged), allow_unused=True)
    _assert_finite_nonzero(gradients[0])
    for gradient in gradients[1:]:
        if gradient is not None:
            torch.testing.assert_close(
                gradient,
                torch.zeros_like(gradient),
                rtol=0.0,
                atol=1e-7,
            )

    _open_head(model)
    opened_score = _isolated_prediction_score(
        model, history, past, future, future_rgb
    )
    encoder_parameters = tuple(model.encoder.parameters())
    opened_gradients = torch.autograd.grad(
        opened_score,
        (*encoder_parameters, head, *staged),
        allow_unused=True,
    )
    encoder_gradients = [
        gradient
        for gradient in opened_gradients[: len(encoder_parameters)]
        if gradient is not None
    ]
    assert encoder_gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in encoder_gradients)
    assert sum(float(gradient.norm()) for gradient in encoder_gradients) > 0.0
    for gradient in opened_gradients[len(encoder_parameters) :]:
        _assert_finite_nonzero(gradient)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
