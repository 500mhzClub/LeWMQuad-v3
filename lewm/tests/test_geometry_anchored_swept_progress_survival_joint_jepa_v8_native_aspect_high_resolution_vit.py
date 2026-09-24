from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit import (
    ENCODER_DIM_V8,
    NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8,
    NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8,
    NATIVE_IMAGE_HEIGHT_V8,
    NATIVE_IMAGE_WIDTH_V8,
    NATIVE_SPATIAL_TOKEN_COUNT_V8,
    NATIVE_TOKEN_CELL_RADII_XY_V8,
    NATIVE_TOKEN_HEIGHT_V8,
    NATIVE_TOKEN_WIDTH_V8,
    SPATIAL_TOKEN_COUNT_V8,
    TOKEN_COUNT_WITH_CLS_V8,
    TOKEN_HEIGHT_V8,
    TOKEN_WIDTH_V8,
    GeometryAnchoredSweptProgressSurvivalJointJepaV8,
    NativeAspectGeometryAnchoredDeformableBevLiftV8,
    NativeAspectHighResolutionBevLiftV8,
    NativeAspectHighResolutionVisionEncoderV8,
    NativeAspectVisionEncoderV8,
    resize_v4_positional_embedding_v8,
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


@pytest.fixture(scope="module")
def clean_v4_and_v8(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    GeometryAnchoredSweptProgressSurvivalJointJepaV8,
]:
    clean = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
        n320_encoder_state, _sweep_masks()
    )
    native = GeometryAnchoredSweptProgressSurvivalJointJepaV8(
        n320_encoder_state, _sweep_masks()
    )
    return clean, native


def test_exact_v4_migration_parameter_count_state_and_target(
    n320_encoder_state: dict[str, torch.Tensor],
    clean_v4_and_v8: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV8,
    ],
) -> None:
    clean, native = clean_v4_and_v8
    assert isinstance(native.encoder, NativeAspectHighResolutionVisionEncoderV8)
    assert isinstance(native.bev_lift, NativeAspectHighResolutionBevLiftV8)
    assert NativeAspectVisionEncoderV8 is NativeAspectHighResolutionVisionEncoderV8
    assert (
        NativeAspectGeometryAnchoredDeformableBevLiftV8
        is NativeAspectHighResolutionBevLiftV8
    )
    assert isinstance(
        native.target_encoder, NativeAspectHighResolutionVisionEncoderV8
    )
    assert isinstance(native.target_bev_lift, NativeAspectHighResolutionBevLiftV8)
    assert (
        NATIVE_IMAGE_HEIGHT_V8,
        NATIVE_IMAGE_WIDTH_V8,
        TOKEN_HEIGHT_V8,
        TOKEN_WIDTH_V8,
        SPATIAL_TOKEN_COUNT_V8,
        TOKEN_COUNT_WITH_CLS_V8,
    ) == (168, 224, 24, 32, 768, 769)
    assert (
        NATIVE_TOKEN_HEIGHT_V8,
        NATIVE_TOKEN_WIDTH_V8,
        NATIVE_SPATIAL_TOKEN_COUNT_V8,
        NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8,
    ) == (24, 32, 768, 2_845_824)
    assert (
        sum(parameter.numel() for parameter in native.encoder.parameters())
        == NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8
        == 2_845_824
    )
    assert (
        sum(parameter.numel() for parameter in native.encoder.parameters())
        - sum(parameter.numel() for parameter in clean.encoder.parameters())
        == 98_304
    )

    expected_position = resize_v4_positional_embedding_v8(
        n320_encoder_state["pos_embed"]
    )
    assert torch.equal(native.encoder.pos_embed, expected_position)
    assert torch.equal(
        native.encoder.pos_embed[:, :1], n320_encoder_state["pos_embed"][:, :1]
    )

    clean_state = clean.state_dict()
    native_state = native.state_dict()
    assert clean_state.keys() == native_state.keys()
    changed = {"encoder.pos_embed", "target_encoder.pos_embed"}
    for name, clean_value in clean_state.items():
        if name in changed:
            assert tuple(clean_value.shape) == (1, 257, 192)
            assert tuple(native_state[name].shape) == (1, 769, 192)
        else:
            assert torch.equal(clean_value, native_state[name]), name

    assert int(native.target_hard_sync_count.item()) == 1
    assert int(native.ema_update_count.item()) == 0
    assert all(
        torch.equal(value, native.target_encoder.state_dict()[name])
        for name, value in native.encoder.state_dict().items()
    )
    assert all(
        torch.equal(value, native.target_bev_lift.state_dict()[name])
        for name, value in native.bev_lift.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in native.encoder.parameters())
    assert all(parameter.requires_grad for parameter in native.bev_lift.parameters())
    assert not any(
        parameter.requires_grad for parameter in native.target_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in native.target_bev_lift.parameters()
    )
    assert not native.target_encoder.training
    assert not native.target_bev_lift.training

    torch.random.default_generator.manual_seed(7821)
    caller_rng = torch.random.get_rng_state().clone()
    GeometryAnchoredSweptProgressSurvivalJointJepaV8(
        n320_encoder_state, _sweep_masks()
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)


def test_float32_bicubic_position_resize_is_exact_and_deterministic(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    source = n320_encoder_state["pos_embed"].detach().clone()
    source_before = source.clone()
    spatial = (
        source[:, 1:]
        .reshape(1, 16, 16, ENCODER_DIM_V8)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    independently_resized = F.interpolate(
        spatial,
        size=(24, 32),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).flatten(start_dim=2).transpose(1, 2).contiguous()
    expected = torch.cat((source[:, :1], independently_resized), dim=1)

    first = resize_v4_positional_embedding_v8(source)
    second = resize_v4_positional_embedding_v8(source)
    assert torch.equal(source, source_before)
    assert torch.equal(first, second)
    assert torch.equal(first, expected)
    assert first.shape == (1, 769, 192)
    assert first.dtype == torch.float32 and first.device.type == "cpu"

    with pytest.raises(ValueError, match="shape"):
        resize_v4_positional_embedding_v8(torch.zeros((1, 256, 192)))
    with pytest.raises(TypeError, match="float32"):
        resize_v4_positional_embedding_v8(
            torch.zeros((1, 257, 192), dtype=torch.float64)
        )


def test_native_tokens_are_24_by_32_row_major_and_inputs_are_strict(
    clean_v4_and_v8: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV8,
    ],
) -> None:
    _clean, native = clean_v4_and_v8
    encoder = native.encoder
    captured: list[torch.Tensor] = []
    handle = encoder.pos_drop.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    rgb = torch.rand(
        (1, 3, 168, 224), generator=torch.Generator().manual_seed(17)
    )
    tokens = encoder.forward_tokens(rgb)
    handle.remove()

    patch_map = encoder.patch_embed(rgb)
    assert patch_map.shape == (1, 192, 24, 32)
    row_major = patch_map.flatten(start_dim=2).transpose(1, 2)
    expected_input = torch.cat(
        (encoder.cls_token.expand(1, -1, -1), row_major), dim=1
    ) + encoder.pos_embed
    assert len(captured) == 1
    assert torch.equal(captured[0], expected_input)
    assert tokens.shape == (1, 769, 192)
    assert tokens.dtype == torch.float32
    assert torch.equal(encoder(rgb), tokens[:, 0])

    with pytest.raises(TypeError, match="tensor"):
        encoder.forward_tokens(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        encoder.forward_tokens(torch.zeros((1, 3, 224, 168)))
    with pytest.raises(ValueError, match="at least one"):
        encoder.forward_tokens(torch.zeros((0, 3, 168, 224)))
    with pytest.raises(TypeError, match="float32"):
        encoder.forward_tokens(torch.zeros((1, 3, 168, 224), dtype=torch.float64))
    nonfinite = torch.zeros((1, 3, 168, 224))
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        encoder.forward_tokens(nonfinite)


def test_rectangular_lift_preserves_v4_grid_masks_weights_and_state(
    clean_v4_and_v8: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV8,
    ],
) -> None:
    clean, native = clean_v4_and_v8
    assert clean.bev_lift.state_dict().keys() == native.bev_lift.state_dict().keys()
    for name, value in clean.bev_lift.state_dict().items():
        assert torch.equal(value, native.bev_lift.state_dict()[name]), name

    legacy = clean.bev_lift.forward_with_sampling(
        torch.zeros((1, 256, 192), dtype=torch.float32)
    )
    rectangular = native.bev_lift.forward_with_sampling(
        torch.zeros((1, 768, 192), dtype=torch.float32)
    )
    assert torch.equal(rectangular.anchor_in_frustum, legacy.anchor_in_frustum)
    assert torch.equal(rectangular.sample_valid_mask, legacy.sample_valid_mask)
    assert torch.equal(rectangular.cell_valid_mask, legacy.cell_valid_mask)
    assert torch.equal(rectangular.sample_grid_xy, legacy.sample_grid_xy)
    assert torch.equal(rectangular.sample_weights, legacy.sample_weights)

    raw = native.bev_lift.raw_offsets[None]
    expected_native_offsets = torch.tanh(raw) * raw.new_tensor((4.0, 3.0))
    assert NATIVE_TOKEN_CELL_RADII_XY_V8 == (4.0, 3.0)
    assert torch.equal(rectangular.offsets_token_cells, expected_native_offsets)
    expected_legacy_offsets = 2.0 * torch.tanh(raw)
    assert torch.equal(legacy.offsets_token_cells, expected_legacy_offsets)

    with pytest.raises(ValueError, match="shape"):
        native.bev_lift(torch.zeros((1, 256, 192)))
    with pytest.raises(TypeError, match="float32"):
        native.bev_lift(torch.zeros((1, 768, 192), dtype=torch.float64))


def test_optimizer_partitions_online_route_gradients_and_exact_ema(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV8(
        n320_encoder_state, _sweep_masks()
    ).train()
    partition = partition_parameters_v1(model)
    online_encoder_names = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("encoder.")
    }
    target_encoder_names = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("target_encoder.")
    }
    assert online_encoder_names == set(partition.names["encoder"])
    assert target_encoder_names <= set(partition.names["target"])
    assert online_encoder_names.isdisjoint(partition.names["lift_semantic"])
    assert online_encoder_names.isdisjoint(partition.names["predictor"])

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

    rgb = torch.rand(
        (1, 3, 168, 224), generator=torch.Generator().manual_seed(41)
    )
    online_latent = model.encode_online(rgb)
    target_latent = model.encode_target(rgb)
    assert online_latent.shape == target_latent.shape == (1, 64, 64, 64)
    weights = torch.rand(
        online_latent.shape, generator=torch.Generator().manual_seed(43)
    ).mul_(2.0).sub_(1.0)
    (online_latent * weights).mean().backward()
    for name, parameter in model.encoder.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert not any(
        parameter.grad is not None for parameter in model.target_encoder.parameters()
    )
    assert not any(
        parameter.grad is not None for parameter in model.target_bev_lift.parameters()
    )

    target_position_before = model.target_encoder.pos_embed.detach().clone()
    target_offset_before = model.target_bev_lift.raw_offsets.detach().clone()
    optimizer.step()
    online_position_after = model.encoder.pos_embed.detach().clone()
    online_offset_after = model.bev_lift.raw_offsets.detach().clone()
    model.update_target_ema_after_optimizer_step()

    expected_position = target_position_before.clone()
    expected_position.mul_(model.config.target_ema_momentum).add_(
        online_position_after, alpha=1.0 - model.config.target_ema_momentum
    )
    expected_offset = target_offset_before.clone()
    expected_offset.mul_(model.config.target_ema_momentum).add_(
        online_offset_after, alpha=1.0 - model.config.target_ema_momentum
    )
    torch.testing.assert_close(
        model.target_encoder.pos_embed, expected_position, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.target_bev_lift.raw_offsets, expected_offset, rtol=0.0, atol=0.0
    )
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 1
    assert not model.target_encoder.training
    assert not model.target_bev_lift.training
    assert not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.target_bev_lift.parameters()
    )
