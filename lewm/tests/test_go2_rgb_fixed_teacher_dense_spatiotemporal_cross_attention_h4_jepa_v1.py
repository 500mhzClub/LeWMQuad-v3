from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1 import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> JointRecurrentH4JEPA:
    torch.manual_seed(51)
    return JointRecurrentH4JEPA(
        config=JointRecurrentH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            cross_attention_heads=3,
            cross_attention_mlp_ratio=4,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(52)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def _assert_state_identical(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        torch.testing.assert_close(value, actual[name], rtol=0.0, atol=0.0)


def test_dense_model_has_no_recurrence_and_starts_at_exact_persistence() -> None:
    default_config = JointRecurrentH4JEPAConfig()
    assert default_config.spatial_token_count == 256
    assert 3 * default_config.spatial_token_count + 2 == 770
    model = _model().eval()
    history, past, future, _ = _batch()
    output = model(history, past, future)
    current = F.normalize(output.history_latents[:, 2], dim=-1)
    expected = current[:, None].expand_as(output.predicted_latents)

    torch.testing.assert_close(output.predicted_latents, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output.predicted_deltas,
        torch.zeros_like(output.predicted_deltas),
        rtol=0.0,
        atol=0.0,
    )
    assert output.belief_latents.shape == (2, 18, 12)
    assert sum(
        isinstance(module, nn.TransformerEncoderLayer)
        for module in model.initial_belief.modules()
    ) == 2
    assert sum(
        isinstance(module, nn.TransformerDecoderLayer)
        for module in model.future_cell.modules()
    ) == 2
    assert all(layer.norm_first for layer in model.initial_belief.encoder)
    assert all(layer.norm_first for layer in model.future_cell.decoder)
    assert all(
        layer.self_attn.num_heads == 3
        and layer.linear1.out_features == 4 * model.config.feature_dim
        and layer.dropout.p == 0.0
        for layer in model.initial_belief.encoder
    )
    assert all(
        layer.self_attn.num_heads == 3
        and layer.multihead_attn.num_heads == 3
        and layer.linear1.out_features == 4 * model.config.feature_dim
        and layer.dropout.p == 0.0
        for layer in model.future_cell.decoder
    )
    assert model.future_cell.future_action_path[0].in_features == 4 * 12
    assert model.future_cell.future_action_path[0].out_features == 12
    assert model.future_cell.future_action_path[2].in_features == 12
    assert model.future_cell.future_action_path[2].out_features == 12
    assert not torch.equal(
        model.initial_belief.encoder[0].linear1.weight,
        model.initial_belief.encoder[1].linear1.weight,
    )
    assert not torch.equal(
        model.future_cell.decoder[0].linear1.weight,
        model.future_cell.decoder[1].linear1.weight,
    )
    forbidden = (nn.RNN, nn.RNNCell, nn.GRU, nn.GRUCell, nn.LSTM, nn.LSTMCell)
    assert not any(isinstance(module, forbidden) for module in model.modules())


def test_context_is_normalized_and_actions_are_explicitly_interleaved() -> None:
    model = _model().eval()
    history, past, future, _ = _batch()
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.initial_belief.encoder[0].register_forward_pre_hook(capture)
    output = model(history, past, future)
    handle.remove()
    assert len(captured) == 1

    tokens = model.spatial_token_count
    memory_input = captured[0]
    assert memory_input.shape == (2, 3 * tokens + 2, 12)
    normalized = F.normalize(
        output.history_latents,
        dim=-1,
        eps=model.config.normalization_epsilon,
    )
    spatial = model.initial_belief.spatial_embedding.weight
    times = model.initial_belief.time_embedding.weight
    expected_frames = normalized + spatial[None, None] + times[None, :, None]
    torch.testing.assert_close(memory_input[:, :tokens], expected_frames[:, 0])
    torch.testing.assert_close(
        memory_input[:, tokens + 1 : 2 * tokens + 1],
        expected_frames[:, 1],
    )
    torch.testing.assert_close(memory_input[:, 2 * tokens + 2 :], expected_frames[:, 2])
    expected_actions = (
        model.action_embedding(past)
        + model.initial_belief.transition_step_embedding.weight[None]
    )
    torch.testing.assert_close(memory_input[:, tokens], expected_actions[:, 0])
    torch.testing.assert_close(
        memory_input[:, 2 * tokens + 1],
        expected_actions[:, 1],
    )


def test_future_action_prefix_uses_ordered_fixed_slots_with_zero_suffix() -> None:
    model = _model().eval()
    history, past, future, _ = _batch()
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.future_cell.future_action_path.register_forward_pre_hook(capture)
    model(history, past, future)
    handle.remove()
    assert len(captured) == 1
    slots = captured[0].reshape(2, 4, 4, 12)
    embedded = model.action_embedding(future)
    for horizon in range(4):
        torch.testing.assert_close(
            slots[:, horizon, : horizon + 1],
            embedded[:, : horizon + 1],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            slots[:, horizon, horizon + 1 :],
            torch.zeros_like(slots[:, horizon, horizon + 1 :]),
            rtol=0.0,
            atol=0.0,
        )


def test_future_rgb_cannot_change_dense_online_predictions() -> None:
    model = _model().eval()
    history, past, future, future_rgb = _batch()
    first = model(history, past, future, future_rgb).predicted_latents
    second = model(history, past, future, future_rgb.flip(1)).predicted_latents
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_fixed_teacher_never_syncs_or_ema_updates() -> None:
    model = _model().train()
    fixed_teacher = copy.deepcopy(model.target_encoder.state_dict())
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.25)
    assert not torch.equal(
        model.encoder.patch_embed.weight,
        model.target_encoder.patch_embed.weight,
    )

    model.hard_sync_target()
    model.update_target(0.0)
    _assert_state_identical(fixed_teacher, model.target_encoder.state_dict())
    assert int(model.ema_update_count.item()) == 0
    assert not model.target_encoder.training
    assert all(
        not parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )


def test_fixed_teacher_losses_are_exactly_alignment_and_raw_delta() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    target = model.encode_target(future_rgb)
    auxiliary = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=output,
    )
    assert set(auxiliary) == {
        "history_teacher_alignment",
        "future_teacher_delta",
    }

    teacher_history = model._encode_fixed_teacher_history(history)
    online_history = F.normalize(output.history_latents, dim=-1)
    expected_alignment = (
        (online_history - teacher_history).square().sum(dim=-1).mean()
    )
    expected_delta = (
        (output.predicted_deltas - (target - teacher_history[:, 2:3]))
        .square()
        .sum(dim=-1)
        .mean()
    )
    torch.testing.assert_close(
        auxiliary["history_teacher_alignment"],
        expected_alignment,
    )
    torch.testing.assert_close(auxiliary["future_teacher_delta"], expected_delta)
    assert output.variance_loss.item() == 0.0
    assert output.total_loss is None


def test_open_delta_head_reaches_dense_memory_queries_actions_and_encoder() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )

    first = model(history, past, future)
    first_target = model.encode_target(future_rgb)
    first_auxiliary = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=first_target,
        output=first,
    )
    first_loss = sum(first_auxiliary.values())
    first_loss.backward()
    final = model.prediction_projector[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.norm()) > 0.0
    optimizer.step()
    assert torch.count_nonzero(final.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second = model(history, past, future)
    second_target = model.encode_target(future_rgb)
    second_auxiliary = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=second_target,
        output=second,
    )
    loss = sum(second_auxiliary.values())
    assert torch.isfinite(loss)
    loss.backward()

    gradients = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief.spatial_embedding.weight.grad,
        model.initial_belief.time_embedding.weight.grad,
        model.initial_belief.transition_step_embedding.weight.grad,
        model.initial_belief.encoder[0].self_attn.in_proj_weight.grad,
        model.action_embedding.weight.grad,
        model.future_cell.horizon_embedding.weight.grad,
        model.future_cell.future_action_path[0].weight.grad,
        model.future_cell.decoder[0].multihead_attn.in_proj_weight.grad,
        final.weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all())
        for value in gradients
        if value is not None
    )
    assert all(float(value.norm()) > 0.0 for value in gradients if value is not None)
    assert all(
        parameter.grad is None
        for parameter in model.target_encoder.parameters()
    )

    actual = model.predict_from_belief(second.belief_latents, future)
    wrong = model.predict_from_belief(second.belief_latents, (future + 1) % 9)
    assert not torch.equal(actual, wrong)


def test_later_action_cannot_change_earlier_direct_horizon_queries() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.prediction_projector[-1].weight.normal_(std=0.02)
        model.prediction_projector[-1].bias.zero_()
    history, past, future, _ = _batch()
    belief = model.encode_history(history, past)[1]
    changed = future.clone()
    changed[:, 3] = (changed[:, 3] + 1) % 9
    actual = model.predict_from_belief(belief, future)
    counterfactual = model.predict_from_belief(belief, changed)
    torch.testing.assert_close(actual[:, :3], counterfactual[:, :3], rtol=0.0, atol=0.0)
    assert not torch.equal(actual[:, 3], counterfactual[:, 3])


def test_shared_runner_optimizer_inventory_covers_dense_model_exactly() -> None:
    model = _model()
    groups = _parameter_groups(model)
    covered = {id(parameter) for values in groups.values() for parameter in values}
    trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert covered == trainable
    assert sum(len(values) for values in groups.values()) == len(covered)
