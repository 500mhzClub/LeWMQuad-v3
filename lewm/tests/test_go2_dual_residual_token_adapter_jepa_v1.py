from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_dual_residual_token_adapter_jepa_v1 import (
    JointResidualTokenAdapterJEPAV1,
    ResidualSpatialTokenAdapterBlockV1,
    ResidualSpatialTokenAdapterV1,
)
from lewm.models.go2_matched_branch_successor_screen_v1 import (
    DenseActionConditionedPredictorV1,
)


_FEATURE_DIM = 12


def _tokens(*leading: int, feature_dim: int = _FEATURE_DIM) -> torch.Tensor:
    values = torch.randn(*leading, 256, feature_dim, dtype=torch.float32)
    return F.normalize(values, p=2.0, dim=-1)


def _joint_inputs(
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = _tokens(batch, 3)
    history = torch.tensor(((0, 1), (2, 3)), dtype=torch.long)[:batch]
    candidate = torch.tensor((4, 5), dtype=torch.long)[:batch]
    return context, history, candidate


def test_zero_initialized_adapter_is_near_identity_token_normalization() -> None:
    torch.manual_seed(101)
    adapter = ResidualSpatialTokenAdapterV1(feature_dim=_FEATURE_DIM)
    tokens = torch.randn(2, 3, 256, _FEATURE_DIM, dtype=torch.float32) * 2.5

    adapted = adapter(tokens)
    expected = F.normalize(tokens, p=2.0, dim=-1, eps=1.0e-12)

    assert adapted.shape == tokens.shape
    assert torch.isfinite(adapted).all()
    torch.testing.assert_close(adapted, expected, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        torch.linalg.vector_norm(adapted, dim=-1),
        torch.ones_like(adapted[..., 0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    for block in adapter.blocks:
        assert torch.count_nonzero(block.output_projection.weight) == 0
        assert torch.count_nonzero(block.output_projection.bias) == 0


def test_adapter_exact_block_inventory_and_bounded_update() -> None:
    torch.manual_seed(102)
    block = ResidualSpatialTokenAdapterBlockV1(feature_dim=_FEATURE_DIM)
    assert block.norm.eps == 1.0e-5
    assert block.input_projection.in_features == _FEATURE_DIM
    assert block.input_projection.out_features == 64
    assert block.depthwise.kernel_size == (3, 3)
    assert block.depthwise.stride == (1, 1)
    assert block.depthwise.padding == (1, 1)
    assert block.depthwise.dilation == (1, 1)
    assert block.depthwise.groups == 64
    assert block.channel_mixing.kernel_size == (1, 1)
    assert block.channel_mixing.groups == 1
    assert block.output_projection.in_features == 64
    assert block.output_projection.out_features == _FEATURE_DIM

    with torch.no_grad():
        block.output_projection.weight.zero_()
        block.output_projection.bias.fill_(1.0e6)
    tokens = _tokens(2)
    update = block.bounded_update(tokens)
    update_norm = torch.linalg.vector_norm(update, ord=2, dim=-1)

    assert update.shape == tokens.shape
    assert torch.isfinite(update).all()
    assert bool((update_norm < 0.125).all())
    assert bool((update_norm > 0.124).all())
    output = block(tokens)
    torch.testing.assert_close(
        torch.linalg.vector_norm(output, dim=-1),
        torch.ones_like(output[..., 0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_joint_constructs_predecessor_predictor_before_adapter_rng_draws() -> None:
    torch.manual_seed(103)
    predecessor = DenseActionConditionedPredictorV1(
        feature_dim=_FEATURE_DIM,
        hidden_dim=128,
        action_count=9,
    )
    expected = {
        name: value.detach().clone() for name, value in predecessor.state_dict().items()
    }

    torch.manual_seed(103)
    joint = JointResidualTokenAdapterJEPAV1(feature_dim=_FEATURE_DIM)

    assert joint.predictor.state_dict().keys() == expected.keys()
    for name, value in joint.predictor.state_dict().items():
        assert torch.equal(value, expected[name])


@pytest.mark.parametrize(
    ("feature_dim", "trainable_count", "target_count"),
    (
        (384, 428_160, 110_336),
        (768, 627_456, 210_944),
    ),
)
def test_exact_preregistered_parameter_counts(
    feature_dim: int,
    trainable_count: int,
    target_count: int,
) -> None:
    torch.manual_seed(104)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=feature_dim)

    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == trainable_count
    assert sum(parameter.numel() for parameter in model.target_adapter.parameters()) == (
        target_count
    )
    assert sum(parameter.numel() for parameter in model.online_adapter.parameters()) == (
        target_count
    )


def test_online_and_target_routes_preserve_shapes_and_target_is_detached() -> None:
    torch.manual_seed(105)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=_FEATURE_DIM)
    tokens = _tokens(2, 12).requires_grad_()

    online = model.adapt_online(tokens)
    target = model.adapt_target(tokens)

    assert online.shape == (2, 12, 256, _FEATURE_DIM)
    assert target.shape == online.shape
    assert online.requires_grad
    assert not target.requires_grad
    assert torch.isfinite(online).all()
    assert torch.isfinite(target).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(online, dim=-1),
        torch.ones_like(online[..., 0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(online, target, rtol=0.0, atol=0.0)


def test_target_stays_frozen_and_eval_while_ema_moves_it() -> None:
    torch.manual_seed(106)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=_FEATURE_DIM)
    model.train()

    assert model.predictor.training
    assert model.online_adapter.training
    assert not model.target_adapter.training
    assert all(not parameter.requires_grad for parameter in model.target_adapter.parameters())
    for online, target in zip(
        model.online_adapter.parameters(),
        model.target_adapter.parameters(),
        strict=True,
    ):
        assert torch.equal(online, target)

    with torch.no_grad():
        model.online_adapter.blocks[0].output_projection.bias.fill_(1.0)
    target_bias = model.target_adapter.blocks[0].output_projection.bias
    assert torch.count_nonzero(target_bias) == 0

    model.update_target_ema_(0.996)

    torch.testing.assert_close(
        target_bias,
        torch.full_like(target_bias, 0.004),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert int(model.ema_update_count) == 1
    assert not model.target_adapter.training
    assert all(not parameter.requires_grad for parameter in model.target_adapter.parameters())
    with pytest.raises(ValueError, match="0.996"):
        model.update_target_ema_(0.995)


def test_joint_prediction_is_finite_and_candidate_action_sensitive() -> None:
    torch.manual_seed(107)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=_FEATURE_DIM)
    context, history, candidate = _joint_inputs()
    adapted = model.adapt_online(context)

    prediction = model.predict_from_adapted_context(adapted, history, candidate)
    through_forward = model(context, history, candidate)
    changed = model(context, history, (candidate + 1) % 9)

    assert prediction.shape == (2, 256, _FEATURE_DIM)
    assert torch.isfinite(prediction).all()
    assert torch.equal(prediction, through_forward)
    assert not torch.allclose(prediction, changed)


def test_joint_backward_has_only_finite_online_gradients() -> None:
    torch.manual_seed(108)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=_FEATURE_DIM)
    context, history, candidate = _joint_inputs()
    target = _tokens(2)

    loss = F.mse_loss(model(context, history, candidate), target)
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert sum(
        float(gradient.abs().sum())
        for gradient in gradients
        if gradient is not None
    ) > 0.0
    for block in model.online_adapter.blocks:
        gradient = block.output_projection.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.target_adapter.parameters())
