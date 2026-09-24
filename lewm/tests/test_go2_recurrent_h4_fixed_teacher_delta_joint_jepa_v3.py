from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from lewm.models.go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3 import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> JointRecurrentH4JEPA:
    torch.manual_seed(31)
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
    generator = torch.Generator().manual_seed(32)
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


def test_zero_delta_head_makes_every_horizon_exact_online_persistence() -> None:
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
    assert output.belief_latents.shape == (2, 4, 24)
    final = model.prediction_projector[-1]
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0
    assert not any(name.endswith("gate") for name, _ in model.named_parameters())


def test_target_is_fixed_across_hard_sync_and_update_calls() -> None:
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


def test_shared_runner_optimizer_inventory_covers_v3_exactly() -> None:
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


def test_alignment_and_delta_losses_match_fixed_teacher_targets() -> None:
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
    teacher_delta = target - teacher_history[:, 2:3]
    expected_delta = (
        (output.predicted_deltas - teacher_delta).square().sum(dim=-1).mean()
    )
    torch.testing.assert_close(
        auxiliary["history_teacher_alignment"],
        expected_alignment,
    )
    torch.testing.assert_close(auxiliary["future_teacher_delta"], expected_delta)
    assert output.variance_loss.item() == 0.0


def test_open_delta_head_reaches_history_action_and_online_encoder() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )

    # The first delta regression step opens the sole zero-initialized head.
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
    assert torch.isfinite(first_loss)
    first_loss.backward()
    final = model.prediction_projector[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.norm()) > 0.0
    optimizer.step()
    assert torch.count_nonzero(final.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second = model(history, past, future, future_rgb)
    assert second.prediction_loss is not None
    assert torch.isfinite(second.prediction_loss)
    assert second.total_loss is None
    assert second.target_latents is not None
    second_auxiliary = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=second.target_latents,
        output=second,
    )
    loss = sum(second_auxiliary.values())
    assert torch.isfinite(loss)
    loss.backward()

    gradients = (
        model.encoder.patch_embed.weight.grad,
        model.initial_belief[1].weight.grad,
        model.history_cell.weight_hh.grad,
        model.future_cell.weight_hh.grad,
        model.action_embedding.weight.grad,
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
