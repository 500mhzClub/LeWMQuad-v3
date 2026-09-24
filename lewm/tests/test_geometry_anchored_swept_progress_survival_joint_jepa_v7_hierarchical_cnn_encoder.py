from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder import (
    HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7,
    HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7,
    GeometryAnchoredSweptProgressSurvivalJointJepaV7,
    HierarchicalCnnEncoderV7,
    HierarchicalCnnResidualBlockV7,
)
from scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1 import (
    build_frozen_optimizer_v1,
    partition_parameters_v1,
    validate_optimizer_v1,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(9917)
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
    finally:
        torch.random.set_rng_state(caller_rng)


def _assert_conv(
    module: nn.Conv2d,
    *,
    channels: tuple[int, int],
    kernel: int,
    stride: int,
    padding: int,
) -> None:
    assert isinstance(module, nn.Conv2d)
    assert (module.in_channels, module.out_channels) == channels
    assert module.kernel_size == (kernel, kernel)
    assert module.stride == (stride, stride)
    assert module.padding == (padding, padding)
    assert module.bias is not None


def test_exact_architecture_parameter_count_and_token_interface() -> None:
    torch.random.default_generator.manual_seed(4321)
    caller_rng = torch.random.get_rng_state().clone()
    encoder = HierarchicalCnnEncoderV7()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7 == 20260715
    assert HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7 == 1_994_880
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 1_994_880

    _assert_conv(
        encoder.stem_conv,
        channels=(3, 48),
        kernel=5,
        stride=2,
        padding=2,
    )
    _assert_conv(
        encoder.down96_conv,
        channels=(48, 96),
        kernel=3,
        stride=2,
        padding=1,
    )
    _assert_conv(
        encoder.down192_conv,
        channels=(96, 192),
        kernel=3,
        stride=2,
        padding=1,
    )
    _assert_conv(
        encoder.output_projection,
        channels=(192, 192),
        kernel=1,
        stride=1,
        padding=0,
    )
    for norm, groups, channels in (
        (encoder.stem_norm, 6, 48),
        (encoder.down96_norm, 8, 96),
        (encoder.down192_norm, 12, 192),
    ):
        assert isinstance(norm, nn.GroupNorm)
        assert (norm.num_groups, norm.num_channels, norm.affine) == (
            groups,
            channels,
            True,
        )
    for stage, width, groups in (
        (encoder.stage48, 48, 6),
        (encoder.stage96, 96, 8),
        (encoder.stage192, 192, 12),
    ):
        assert len(stage) == 2
        for block in stage:
            assert isinstance(block, HierarchicalCnnResidualBlockV7)
            _assert_conv(
                block.conv1,
                channels=(width, width),
                kernel=3,
                stride=1,
                padding=1,
            )
            _assert_conv(
                block.conv2,
                channels=(width, width),
                kernel=3,
                stride=1,
                padding=1,
            )
            assert (block.norm1.num_groups, block.norm2.num_groups) == (
                groups,
                groups,
            )
            assert block.norm1.affine and block.norm2.affine

    projected: list[torch.Tensor] = []
    handle = encoder.output_projection.register_forward_hook(
        lambda _module, _inputs, output: projected.append(output.detach().clone())
    )
    rgb = torch.rand((1, 3, 112, 112), generator=torch.Generator().manual_seed(17))
    tokens = encoder.forward_tokens(rgb)
    handle.remove()
    assert tokens.shape == (1, 257, 192)
    assert tokens.dtype == torch.float32
    assert len(projected) == 1 and projected[0].shape == (1, 192, 16, 16)
    assert torch.equal(
        tokens[:, 1:], projected[0].flatten(start_dim=2).transpose(1, 2)
    )
    assert torch.equal(tokens[:, :1], tokens[:, 1:].mean(dim=1, keepdim=True))
    assert torch.equal(encoder(rgb), tokens[:, 0])


def test_encoder_input_validation() -> None:
    encoder = HierarchicalCnnEncoderV7()
    with pytest.raises(TypeError, match="tensor"):
        encoder.forward_tokens(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        encoder.forward_tokens(torch.zeros((3, 112, 112)))
    with pytest.raises(ValueError, match="shape"):
        encoder.forward_tokens(torch.zeros((1, 2, 112, 112)))
    with pytest.raises(ValueError, match="shape"):
        encoder.forward_tokens(torch.zeros((1, 3, 111, 112)))
    with pytest.raises(ValueError, match="at least one"):
        encoder.forward_tokens(torch.zeros((0, 3, 112, 112)))
    with pytest.raises(TypeError, match="float32"):
        encoder.forward_tokens(torch.zeros((1, 3, 112, 112), dtype=torch.float64))
    nonfinite = torch.zeros((1, 3, 112, 112))
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        encoder.forward_tokens(nonfinite)


def test_fixed_seed_is_deterministic_and_ignores_n320_values(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    different_n320 = {
        name: value.detach().clone()
        for name, value in n320_encoder_state.items()
    }
    changed_name = next(iter(different_n320))
    different_n320[changed_name].add_(0.125)
    assert not torch.equal(
        n320_encoder_state[changed_name], different_n320[changed_name]
    )

    torch.random.default_generator.manual_seed(8103)
    first_caller_rng = torch.random.get_rng_state().clone()
    first = GeometryAnchoredSweptProgressSurvivalJointJepaV7(
        n320_encoder_state, _sweep_masks()
    )
    assert torch.equal(torch.random.get_rng_state(), first_caller_rng)

    torch.random.default_generator.manual_seed(1907)
    second_caller_rng = torch.random.get_rng_state().clone()
    second = GeometryAnchoredSweptProgressSurvivalJointJepaV7(
        different_n320, _sweep_masks()
    )
    assert torch.equal(torch.random.get_rng_state(), second_caller_rng)
    assert all(
        torch.equal(value, second.encoder.state_dict()[name])
        for name, value in first.encoder.state_dict().items()
    )


def test_target_partition_gradients_and_ema(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV7(
        n320_encoder_state, _sweep_masks()
    ).train()
    assert isinstance(model.encoder, HierarchicalCnnEncoderV7)
    assert isinstance(model.target_encoder, HierarchicalCnnEncoderV7)
    assert all(
        torch.equal(value, model.target_encoder.state_dict()[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    )
    assert not model.target_encoder.training

    partition = partition_parameters_v1(model)
    online_names = {
        name for name, _parameter in model.named_parameters() if name.startswith("encoder.")
    }
    target_names = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("target_encoder.")
    }
    assert online_names == set(partition.names["encoder"])
    assert target_names <= set(partition.names["target"])
    assert online_names.isdisjoint(partition.names["lift_semantic"])
    assert online_names.isdisjoint(partition.names["predictor"])
    optimizer = build_frozen_optimizer_v1(partition)
    validate_optimizer_v1(optimizer, partition)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert {id(parameter) for parameter in model.encoder.parameters()} <= optimizer_ids
    assert {
        id(parameter) for parameter in model.target_encoder.parameters()
    }.isdisjoint(optimizer_ids)

    rgb = torch.rand((1, 3, 112, 112), generator=torch.Generator().manual_seed(41))
    online_latent = model.encode_online(rgb)
    target_latent = model.encode_target(rgb)
    assert online_latent.shape == target_latent.shape == (1, 64, 64, 64)
    (online_latent.square().mean() + target_latent.square().mean()).backward()
    for name, parameter in model.encoder.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert not any(
        parameter.grad is not None for parameter in model.target_encoder.parameters()
    )

    target_before = model.target_encoder.output_projection.weight.detach().clone()
    optimizer.step()
    online_after = model.encoder.output_projection.weight.detach().clone()
    model.update_target_ema_after_optimizer_step()
    expected = target_before.mul(model.config.target_ema_momentum).add(
        online_after, alpha=1.0 - model.config.target_ema_momentum
    )
    torch.testing.assert_close(
        model.target_encoder.output_projection.weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert not model.target_encoder.training
    assert not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    )
