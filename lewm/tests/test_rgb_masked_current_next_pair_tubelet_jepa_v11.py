from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from lewm.models.encoders import ViTBlock, VisionEncoder
from lewm.models.rgb_masked_current_next_pair_tubelet_jepa_v11 import (
    ACTION_VOCABULARY_V11,
    AllActionPredictionsV11,
    FixedCurrentTargetsV11,
    MaskedCurrentNextPairTubeletJepaV11,
    action_retrieval_loss_v11,
    masked_pair_tubelet_objective_v11,
    normalized_token_energy_v11,
    projected_future_whitening_v11,
    target_retrieval_loss_v11,
)


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(1701)
    encoder = VisionEncoder(
        image_size=112,
        patch_size=7,
        hidden_dim=192,
        depth=6,
        n_heads=6,
        mlp_ratio=4,
        dropout=0.0,
    )
    return {
        name: value.detach().clone()
        for name, value in encoder.state_dict().items()
    }


def _model(
    n320_encoder_state: dict[str, torch.Tensor],
) -> MaskedCurrentNextPairTubeletJepaV11:
    torch.manual_seed(20260712)
    return MaskedCurrentNextPairTubeletJepaV11(n320_encoder_state).eval()


def _rgb(batch: int, *, offset: float = 0.0) -> torch.Tensor:
    values = torch.linspace(
        -1.0 + offset,
        1.0 + offset,
        batch * 3 * 112 * 112,
        dtype=torch.float32,
    )
    return values.reshape(batch, 3, 112, 112)


def _normalized(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return nn.functional.normalize(
        torch.randn(*shape, generator=generator), dim=-1, eps=1e-8
    )


def _assert_nonzero_finite_gradient(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0


def test_v11_initialization_draw_order_architecture_and_target_inventory(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(20260712)
    expected_mask = torch.empty(1, 1, 192)
    nn.init.trunc_normal_(expected_mask, std=0.02)
    expected_time = torch.empty(1, 1, 192)
    nn.init.trunc_normal_(expected_time, std=0.02)
    expected_projector = torch.empty(192, 192)
    nn.init.xavier_uniform_(expected_projector)
    expected_rng = torch.random.get_rng_state()

    model = _model(n320_encoder_state)

    assert torch.equal(model.online_future_mask_token, expected_mask)
    assert torch.equal(model.online_future_temporal_embedding, expected_time)
    assert torch.equal(model.online_future_projector.weight, expected_projector)
    assert torch.equal(
        model.online_action_embedding.weight,
        torch.zeros(9, 192),
    )
    assert torch.equal(model.online_future_projector.bias, torch.zeros(192))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert torch.equal(model.current_temporal_embedding, torch.zeros(1, 1, 192))

    assert len(model.encoder.blocks) == 6
    assert len(model.target_encoder.blocks) == 6
    assert sum(isinstance(module, ViTBlock) for module in model.modules()) == 12
    assert ACTION_VOCABULARY_V11[6] == "hold"
    assert model.action_vocabulary == ACTION_VOCABULARY_V11
    assert not model.encoder.cls_token.requires_grad
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert not model.target_encoder.training
    assert not model.target_future_projector.training
    assert int(model.ema_update_count) == 0

    inventory = dict(model.ema_inventory_exact())
    assert inventory["encoder.patch_embed.weight"] == (
        "target_encoder.patch_embed.weight"
    )
    assert inventory["encoder.cls_token"] == "target_encoder.cls_token"
    assert inventory["encoder.pos_embed"] == "target_encoder.pos_embed"
    assert inventory["encoder.blocks.0.norm1.weight"] == (
        "target_encoder.blocks.0.norm1.weight"
    )
    assert inventory["encoder.blocks.5.norm2.bias"] == (
        "target_encoder.blocks.5.norm2.bias"
    )
    assert inventory["encoder.norm.weight"] == "target_encoder.norm.weight"
    assert inventory["online_future_temporal_embedding"] == (
        "target_future_temporal_embedding"
    )
    assert inventory["online_future_projector.weight"] == (
        "target_future_projector.weight"
    )
    assert "online_future_mask_token" not in inventory
    assert "online_action_embedding.weight" not in inventory
    parameters = dict(model.named_parameters())
    for online_name, target_name in inventory.items():
        assert torch.equal(parameters[online_name], parameters[target_name])


def test_v11_positions_action_order_shapes_and_shared_current(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    with torch.no_grad():
        values = torch.arange(9, dtype=torch.float32)[:, None] / 100.0
        model.online_action_embedding.weight.copy_(values.expand(-1, 192))
    current = _rgb(1)

    result = model.predict_all_actions(current, capture_intermediates=True)

    assert result.normalized_projected_future.shape == (1, 9, 256, 192)
    assert result.action_indices.dtype == torch.long
    assert result.action_indices.tolist() == list(range(9))
    assert result.shared_current_patch_tokens.shape == (1, 256, 192)
    assert result.tubelet_input is not None
    assert result.tubelet_input.shape == (1, 9, 512, 192)
    assert len(result.block_outputs) == 6
    assert all(value.shape == (1, 9, 512, 192) for value in result.block_outputs)
    assert torch.allclose(
        result.normalized_projected_future.norm(dim=-1),
        torch.ones(1, 9, 256),
        atol=1e-6,
    )

    expected_current = (
        model.encoder.patch_embed(current).flatten(2).transpose(1, 2)
        + model.encoder.pos_embed[:, 1:]
        + model.current_temporal_embedding
    )
    assert torch.equal(result.shared_current_patch_tokens, expected_current)
    for action in range(9):
        assert torch.equal(
            result.tubelet_input[:, action, :256], expected_current
        )
        expected_future = model.online_future_mask_token.expand(1, 256, -1)
        expected_future = expected_future + model.encoder.pos_embed[:, 1:]
        expected_future = expected_future + model.online_future_temporal_embedding
        expected_future = (
            expected_future
            + model.online_action_embedding.weight[action][None, None]
        )
        assert torch.equal(
            result.tubelet_input[:, action, 256:], expected_future
        )
    assert not torch.equal(
        result.block_outputs[0][:, 0, 256:],
        result.block_outputs[0][:, 8, 256:],
    )
    assert not torch.equal(
        result.block_outputs[5][:, 0, 256:],
        result.block_outputs[5][:, 8, 256:],
    )


def _online_observation(
    model: MaskedCurrentNextPairTubeletJepaV11,
    current: torch.Tensor,
    ignored_correct_next: torch.Tensor,
    ignored_deranged_next: torch.Tensor,
) -> tuple[object, torch.Tensor, dict[str, torch.Tensor]]:
    # Exercise the real detached target path first.  Its outputs deliberately
    # have no route into the online API or its full six-block path.
    assert ignored_correct_next.data_ptr() != ignored_deranged_next.data_ptr()
    before_rng = torch.random.get_rng_state().clone()
    before_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("target_")
    }
    model.build_fixed_current_targets(
        current,
        ignored_correct_next,
        ignored_deranged_next,
    )
    result = model.predict_all_actions(current, capture_intermediates=True)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    after_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("target_")
    }
    return result, before_rng, {
        name: after_state[name] for name in before_state
        if torch.equal(before_state[name], after_state[name])
    }


def _assert_online_observations_bitwise_equal(left: object, right: object) -> None:
    assert torch.equal(left.action_indices, right.action_indices)
    assert torch.equal(
        left.shared_current_patch_tokens,
        right.shared_current_patch_tokens,
    )
    assert torch.equal(left.tubelet_input, right.tubelet_input)
    assert len(left.block_outputs) == len(right.block_outputs) == 6
    for left_block, right_block in zip(left.block_outputs, right.block_outputs):
        assert torch.equal(left_block, right_block)
    assert torch.equal(
        left.normalized_projected_future,
        right.normalized_projected_future,
    )


def test_v11_future_replacement_is_bitwise_inert_at_update_zero_and_after_step(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    current = _rgb(1)
    next_a, deranged_a = _rgb(1, offset=0.1), _rgb(1, offset=0.2)
    next_b, deranged_b = _rgb(1, offset=-0.3), _rgb(1, offset=0.4)

    zero_a, rng_a, unchanged_a = _online_observation(
        model, current, next_a, deranged_a
    )
    zero_b, rng_b, unchanged_b = _online_observation(
        model, current, next_b, deranged_b
    )
    _assert_online_observations_bitwise_equal(zero_a, zero_b)
    assert torch.equal(rng_a, rng_b)
    assert unchanged_a.keys() == unchanged_b.keys() == {
        name for name in model.state_dict() if not name.startswith("target_")
    }

    model.train()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    prediction = model.forward_online_context(current, torch.tensor([3]))
    target = model.encode_target_future(current, next_a)
    loss = (prediction.normalized_projected_future - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.update_target_ema()
    model.eval()
    assert int(model.ema_update_count) == 1

    step_a, rng_a, _ = _online_observation(model, current, next_a, deranged_a)
    step_b, rng_b, _ = _online_observation(model, current, next_b, deranged_b)
    _assert_online_observations_bitwise_equal(step_a, step_b)
    assert torch.equal(rng_a, rng_b)


def test_v11_target_is_nonvacuous_no_grad_action_free_and_fixed_current(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    model.train()
    assert not model.target_encoder.training
    assert not model.target_future_projector.training
    assert "future" not in inspect.signature(model.forward_online_context).parameters
    assert "next" not in inspect.signature(model.forward_online_context).parameters
    assert "action" not in inspect.signature(model.encode_target_future).parameters
    assert "action" not in inspect.signature(model.build_fixed_current_targets).parameters

    current = _rgb(1).requires_grad_()
    next_a = _rgb(1, offset=0.25).requires_grad_()
    next_b = _rgb(1, offset=-0.35).requires_grad_()
    deranged = _rgb(1, offset=0.6).requires_grad_()
    target_a = model.encode_target_future(current, next_a)
    target_b = model.encode_target_future(current, next_b)
    assert not target_a.requires_grad and target_a.grad_fn is None
    assert not target_b.requires_grad and target_b.grad_fn is None
    assert torch.count_nonzero(target_a != target_b) > 0

    patch_calls = 0

    def count_patch_call(_module: nn.Module, _inputs: object, _output: object) -> None:
        nonlocal patch_calls
        patch_calls += 1

    handle = model.target_encoder.patch_embed.register_forward_hook(count_patch_call)
    try:
        targets = model.build_fixed_current_targets(
            current,
            next_a,
            deranged,
            capture_intermediates=True,
        )
    finally:
        handle.remove()
    assert patch_calls == 3  # current once, plus correct and deranged future
    assert targets.tubelet_inputs is not None
    assert targets.tubelet_inputs.shape == (1, 3, 512, 192)
    for candidate in range(3):
        assert torch.equal(
            targets.tubelet_inputs[:, candidate, :256],
            targets.shared_current_patch_tokens,
        )
    assert targets.correct_next.shape == (1, 256, 192)
    assert targets.deranged_next.shape == (1, 256, 192)
    assert targets.no_change_current.shape == (1, 256, 192)
    assert all(
        not tensor.requires_grad
        for tensor in (
            targets.correct_next,
            targets.deranged_next,
            targets.no_change_current,
        )
    )

    online = model.forward_online_context(current, torch.tensor([4]))
    online.normalized_projected_future.square().mean().backward()
    assert current.grad is not None
    assert next_a.grad is None
    assert next_b.grad is None
    assert deranged.grad is None
    assert not any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert not any(
        parameter.grad is not None
        for parameter in model.target_future_projector.parameters()
    )


def test_v11_retrieval_whitening_and_total_match_registered_math() -> None:
    predictions = _normalized((2, 9, 256, 192), seed=31).requires_grad_()
    correct = _normalized((2, 256, 192), seed=32)
    deranged = _normalized((2, 256, 192), seed=33)
    current = _normalized((2, 256, 192), seed=34)
    actions = torch.tensor([6, 2], dtype=torch.long)
    all_actions = AllActionPredictionsV11(
        predictions,
        torch.arange(9),
        torch.empty(2, 256, 192),
        None,
        (),
    )
    targets = FixedCurrentTargetsV11(
        correct, deranged, current, torch.empty(2, 256, 192), None
    )

    manual_action_energy = (
        predictions - correct[:, None]
    ).square().sum(dim=-1).mean(dim=-1)
    action_terms = action_retrieval_loss_v11(all_actions, correct, actions)
    assert torch.equal(action_terms.energies, manual_action_energy)
    assert torch.equal(action_terms.logits, -action_terms.energies)

    executed = predictions[torch.arange(2), actions]
    candidates = torch.stack((correct, deranged, current), dim=1)
    manual_target_energy = (
        executed[:, None] - candidates
    ).square().sum(dim=-1).mean(dim=-1)
    target_terms = target_retrieval_loss_v11(executed, targets, actions)
    assert torch.equal(target_terms.energies, manual_target_energy)
    assert torch.equal(target_terms.logits, -target_terms.energies)
    assert target_terms.candidate_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]

    objective = masked_pair_tubelet_objective_v11(
        all_actions, targets, actions
    )
    whitening = projected_future_whitening_v11(executed)
    expected_masked = (executed - correct).square().mean()
    expected_total = (
        expected_masked
        + action_terms.loss
        + target_terms.loss
        + 0.50 * whitening.variance
        + 0.02 * whitening.covariance
    )
    assert torch.equal(objective.masked_future_jepa, expected_masked)
    assert torch.equal(objective.action_logits, -objective.action_energies)
    assert torch.equal(objective.target_logits, -objective.target_energies)
    assert torch.equal(objective.total, expected_total)
    assert normalized_token_energy_v11(correct, correct).eq(0).all()
    assert (normalized_token_energy_v11(correct, -correct) <= 4.0).all()


@pytest.mark.parametrize(
    "component",
    ["masked", "action", "target", "whitening"],
)
def test_v11_each_loss_reaches_the_full_online_path(
    n320_encoder_state: dict[str, torch.Tensor],
    component: str,
) -> None:
    model = _model(n320_encoder_state).train()
    with torch.no_grad():
        # Break only the update-zero action symmetry for this gradient fixture.
        model.online_action_embedding.weight.copy_(
            torch.randn(9, 192) * 0.01
        )
    batch = 2 if component == "whitening" else 1
    current = _rgb(batch)
    actions = torch.tensor([1, 8][:batch], dtype=torch.long)

    if component == "action":
        all_actions = model.predict_all_actions(current)
        correct = _normalized((batch, 256, 192), seed=91)
        loss = action_retrieval_loss_v11(
            all_actions, correct, actions
        ).loss
    else:
        path = model.forward_online_context(current, actions)
        prediction = path.normalized_projected_future
        if component == "masked":
            target = _normalized((batch, 256, 192), seed=92)
            loss = (prediction - target).square().mean()
        elif component == "target":
            targets = FixedCurrentTargetsV11(
                _normalized((batch, 256, 192), seed=93),
                _normalized((batch, 256, 192), seed=94),
                _normalized((batch, 256, 192), seed=95),
                torch.empty(batch, 256, 192),
                None,
            )
            loss = target_retrieval_loss_v11(
                prediction, targets, actions
            ).loss
        else:
            whitening = projected_future_whitening_v11(prediction)
            loss = 0.50 * whitening.variance + 0.02 * whitening.covariance

    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    for parameter in (
        model.encoder.patch_embed.weight,
        model.encoder.blocks[0].attn.in_proj_weight,
        model.encoder.blocks[5].attn.in_proj_weight,
        model.online_future_mask_token,
        model.online_future_temporal_embedding,
        model.online_future_projector.weight,
    ):
        _assert_nonzero_finite_gradient(parameter)
    _assert_nonzero_finite_gradient(model.online_action_embedding.weight)
    assert not any(parameter.grad is not None for parameter in model.target_encoder.parameters())


def test_v11_ema_is_exact_complete_and_excludes_online_only_parameters(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    parameters = dict(model.named_parameters())
    inventory = model.ema_inventory_exact()
    before_target = {
        target_name: parameters[target_name].detach().clone()
        for _, target_name in inventory
    }
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.25)
        model.online_future_temporal_embedding.add_(0.5)
        model.online_future_projector.bias.add_(0.75)
        model.online_future_mask_token.add_(1.0)
        model.online_action_embedding.weight.add_(2.0)
    online_only_before = (
        model.online_future_mask_token.detach().clone(),
        model.online_action_embedding.weight.detach().clone(),
    )

    expected = {
        target_name: (
            before_target[target_name] * 0.996
            + parameters[online_name].detach() * 0.004
        )
        for online_name, target_name in inventory
    }
    model.update_target_ema()

    assert int(model.ema_update_count) == 1
    for _, target_name in inventory:
        assert torch.allclose(parameters[target_name], expected[target_name])
        assert not parameters[target_name].requires_grad
    assert torch.equal(model.online_future_mask_token, online_only_before[0])
    assert torch.equal(model.online_action_embedding.weight, online_only_before[1])
    assert not model.target_encoder.training
    assert not model.target_future_projector.training

    model.hard_sync_target_from_online()
    assert int(model.ema_update_count) == 0
    for online_name, target_name in inventory:
        assert torch.equal(parameters[online_name], parameters[target_name])
