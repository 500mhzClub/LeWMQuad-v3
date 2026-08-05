from __future__ import annotations

from copy import deepcopy
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_grounded_dense_dino_joint_jepa_v1 import (
    dense_patch_cosine_loss_v1,
    within_state_action_infonce_loss_v1,
)
from lewm.models.go2_grounded_dense_dino_joint_jepa_v1 import (
    ACTION_DIM,
    DEFAULT_EMA_MOMENTUM,
    FEATURE_DIM,
    FULL_TOKEN_COUNT,
    PATCH_TOKEN_COUNT,
    PHYSICAL_INPUT_DIM,
    DINOv2TrainableTailV1,
    GroundedDenseDINOJointJEPAV1,
)


class _TailBlock(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.register_buffer("marker", torch.tensor(scale, dtype=torch.float32))
        self.calls = 0
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(FEATURE_DIM) * scale)
            self.projection.bias.zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tokens + self.projection(tokens)


def _tail_parts() -> tuple[list[nn.Module], nn.Module]:
    return [_TailBlock(0.01), _TailBlock(0.02)], nn.LayerNorm(FEATURE_DIM)


def _model() -> GroundedDenseDINOJointJEPAV1:
    blocks, norm = _tail_parts()
    return GroundedDenseDINOJointJEPAV1(blocks, norm)


def _inputs(
    *, batch: int = 1, actions: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(2_026_080_408)
    context = torch.randn(
        batch,
        3,
        FULL_TOKEN_COUNT,
        FEATURE_DIM,
        generator=generator,
    )
    history = torch.randn(batch, 2, ACTION_DIM, generator=generator)
    candidates = torch.randn(batch, actions, ACTION_DIM, generator=generator)
    physical = torch.randn(
        batch, actions, PHYSICAL_INPUT_DIM, generator=generator
    )
    return context, history, candidates, physical


def _open_predictor_action_path(model: GroundedDenseDINOJointJEPAV1) -> None:
    generator = torch.Generator(device="cpu").manual_seed(2_026_080_409)
    with torch.no_grad():
        model.predictor.output_projection.weight.copy_(
            0.02 * torch.eye(FEATURE_DIM)
        )
        model.predictor.output_projection.bias.zero_()
        for block in model.predictor.blocks:
            modulation = block.adaLN_modulation[-1]
            modulation.weight.copy_(
                torch.randn(
                    modulation.weight.shape,
                    generator=generator,
                    dtype=modulation.weight.dtype,
                )
                * 0.01
            )
            modulation.bias.zero_()
            modulation.bias[2 * FEATURE_DIM : 3 * FEATURE_DIM].fill_(0.1)
            modulation.bias[5 * FEATURE_DIM : 6 * FEATURE_DIM].fill_(0.1)


def test_tail_requires_exactly_two_blocks_and_preserves_full_token_contract() -> None:
    blocks, norm = _tail_parts()
    tail = DINOv2TrainableTailV1(blocks, norm)
    tokens = torch.randn(2, FULL_TOKEN_COUNT, FEATURE_DIM)

    full = tail(tokens)
    patches = tail.patch_tokens(tokens)

    assert full.shape == (2, FULL_TOKEN_COUNT, FEATURE_DIM)
    assert patches.shape == (2, PATCH_TOKEN_COUNT, FEATURE_DIM)
    torch.testing.assert_close(
        torch.linalg.vector_norm(patches, dim=-1),
        torch.ones(2, PATCH_TOKEN_COUNT),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    with pytest.raises(ValueError, match="exactly two"):
        DINOv2TrainableTailV1(blocks[:1], deepcopy(norm))
    with pytest.raises(ValueError, match="exactly two"):
        DINOv2TrainableTailV1([*blocks, deepcopy(blocks[0])], deepcopy(norm))


def test_batch_prediction_shapes_attention_and_zero_physical_residual() -> None:
    model = _model()
    context, history, candidates, physical = _inputs(batch=2, actions=3)

    result = model(context, history, candidates, physical)

    assert result.successor_tokens.shape == (2, 3, PATCH_TOKEN_COUNT, FEATURE_DIM)
    assert result.standardized_physical_residuals.shape == (2, 3, 4)
    assert result.physical_attention.shape == (2, 3, PATCH_TOKEN_COUNT)
    assert torch.count_nonzero(result.standardized_physical_residuals) == 0
    torch.testing.assert_close(
        result.physical_attention.sum(dim=-1),
        torch.ones(2, 3),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.successor_tokens, dim=-1),
        torch.ones(2, 3, PATCH_TOKEN_COUNT),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # The evaluator's registered decode therefore yields exact action means.
    action_means = torch.randn(3, 4)
    residual_scales = torch.rand(4) + 0.1
    decoded = (
        result.standardized_physical_residuals * residual_scales
        + action_means.unsqueeze(0)
    )
    assert torch.equal(decoded, action_means.unsqueeze(0).expand(2, -1, -1))


def test_target_encoding_is_detached_nontrainable_and_stays_in_eval_mode() -> None:
    model = _model().train()
    target = torch.randn(1, 2, FULL_TOKEN_COUNT, FEATURE_DIM, requires_grad=True)

    encoded = model.encode_target(target)

    assert encoded.shape == (1, 2, PATCH_TOKEN_COUNT, FEATURE_DIM)
    assert encoded.requires_grad is False
    assert encoded.grad_fn is None
    assert target.grad is None
    assert model.target_tail.training is False
    assert all(not parameter.requires_grad for parameter in model.target_tail.parameters())
    assert all(parameter.grad is None for parameter in model.target_tail.parameters())


def test_ema_starts_exact_and_applies_registered_parameter_update() -> None:
    model = _model()
    online = dict(model.online_tail.named_parameters())
    target = dict(model.target_tail.named_parameters())
    assert online.keys() == target.keys()
    for name in online:
        assert torch.equal(online[name], target[name])

    name = next(iter(online))
    before = target[name].detach().clone()
    with torch.no_grad():
        online[name].add_(2.0)
        next(iter(model.online_tail.named_buffers()))[1].fill_(7.0)
    model.update_target_ema(momentum=0.75)

    expected = before * 0.75 + online[name].detach() * 0.25
    assert torch.equal(target[name], expected)
    online_buffers = dict(model.online_tail.named_buffers())
    target_buffers = dict(model.target_tail.named_buffers())
    assert torch.equal(target_buffers["blocks.0.marker"], online_buffers["blocks.0.marker"])
    assert model.ema_momentum == DEFAULT_EMA_MOMENTUM
    assert all(not parameter.requires_grad for parameter in target.values())
    with pytest.raises(ValueError, match="EMA momentum"):
        model.update_target_ema(momentum=True)


def test_candidate_permutation_preserves_state_and_branch_alignment() -> None:
    model = _model().eval()
    _open_predictor_action_path(model)
    context, history, candidates, physical = _inputs(batch=2, actions=3)
    first = model(context, history, candidates, physical)
    permutations = (torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]))
    changed_candidates = torch.stack(
        [candidates[row, order] for row, order in enumerate(permutations)]
    )
    changed_physical = torch.stack(
        [physical[row, order] for row, order in enumerate(permutations)]
    )

    changed = model(context, history, changed_candidates, changed_physical)

    assert not torch.allclose(
        first.successor_tokens[:, 0], first.successor_tokens[:, 1]
    )
    for row, order in enumerate(permutations):
        torch.testing.assert_close(
            changed.successor_tokens[row], first.successor_tokens[row, order]
        )
        torch.testing.assert_close(
            changed.physical_attention[row], first.physical_attention[row, order]
        )
        torch.testing.assert_close(
            changed.standardized_physical_residuals[row],
            first.standardized_physical_residuals[row, order],
        )


def test_two_arm_construction_has_identical_initial_state_and_inventory() -> None:
    torch.manual_seed(41)
    blocks, norm = _tail_parts()
    physical_only = GroundedDenseDINOJointJEPAV1(
        deepcopy(blocks), deepcopy(norm)
    )
    joint = GroundedDenseDINOJointJEPAV1(deepcopy(blocks), deepcopy(norm))

    left = physical_only.state_dict()
    right = joint.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)
    assert [name for name, value in physical_only.named_parameters() if value.requires_grad] == [
        name for name, value in joint.named_parameters() if value.requires_grad
    ]
    assert sum(
        parameter.numel()
        for parameter in physical_only.parameters()
        if parameter.requires_grad
    ) == sum(
        parameter.numel()
        for parameter in joint.parameters()
        if parameter.requires_grad
    )


def test_inference_signature_and_execution_have_no_true_future_input() -> None:
    model = _model()
    context, history, candidates, physical = _inputs()
    parameter_names = tuple(inspect.signature(model.forward).parameters)
    assert parameter_names == (
        "context_trunk_tokens",
        "history_commands",
        "candidate_commands",
        "physical_inputs",
    )

    model(context, history, candidates, physical)
    assert all(block.calls > 0 for block in model.online_tail.blocks)
    assert all(block.calls == 0 for block in model.target_tail.blocks)
    with pytest.raises(TypeError):
        model(
            context,
            history,
            candidates,
            physical,
            torch.randn(1, 3, FULL_TOKEN_COUNT, FEATURE_DIM),
        )


def test_invalid_shapes_types_and_nonfinite_inputs_fail_closed() -> None:
    model = _model()
    context, history, candidates, physical = _inputs()
    with pytest.raises(ValueError, match="context_trunk_tokens"):
        model(context[:, :, :-1], history, candidates, physical)
    with pytest.raises(TypeError, match="history_commands"):
        model(context, history.to(torch.float64), candidates, physical)
    with pytest.raises(ValueError, match="candidate_commands"):
        model(context, history, candidates[:, :0], physical[:, :0])
    with pytest.raises(ValueError, match="physical_inputs"):
        model(context, history, candidates, physical[..., :-1])
    nonfinite = candidates.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="candidate_commands"):
        model(context, history, nonfinite, physical)
    with pytest.raises(ValueError, match="target_trunk_tokens"):
        model.encode_target(torch.randn(1, 1, FULL_TOKEN_COUNT - 1, FEATURE_DIM))


def test_physical_loss_backpropagates_without_target_tail_gradients() -> None:
    model = _model()
    context, history, candidates, physical = _inputs(batch=1, actions=2)
    result = model(context, history, candidates, physical)
    target = torch.randn_like(result.standardized_physical_residuals)

    F.mse_loss(result.standardized_physical_residuals, target).backward()

    final_grad = model.physical_head.output_projection.weight.grad
    assert final_grad is not None
    assert torch.isfinite(final_grad).all()
    assert float(final_grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.target_tail.parameters())


def test_joint_losses_route_gradients_to_online_tail_and_predictor_only() -> None:
    model = _model()
    context, history, candidates, physical = _inputs(batch=1, actions=9)
    target_trunks = torch.randn(1, 9, FULL_TOKEN_COUNT, FEATURE_DIM)

    prediction = model(context, history, candidates, physical)
    targets = model.encode_target(target_trunks)
    loss = dense_patch_cosine_loss_v1(prediction.successor_tokens, targets)
    loss = loss + 0.1 * within_state_action_infonce_loss_v1(
        prediction.successor_tokens,
        targets,
    )
    loss.backward()

    online_gradients = [
        parameter.grad for parameter in model.online_tail.parameters()
    ]
    predictor_gradients = [
        parameter.grad for parameter in model.predictor.parameters()
    ]
    assert any(
        gradient is not None and float(gradient.abs().sum()) > 0.0
        for gradient in online_gradients
    )
    assert any(
        gradient is not None and float(gradient.abs().sum()) > 0.0
        for gradient in predictor_gradients
    )
    assert all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in (*online_gradients, *predictor_gradients)
    )
    assert all(parameter.grad is None for parameter in model.target_tail.parameters())
