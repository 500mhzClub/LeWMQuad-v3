from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from lewm.models.encoders import ViTBlock, VisionEncoder
from lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (
    SingleFrameMultiblockMaskedSpatialJepaV1,
    SingleFrameMultiblockMaskedSpatialJepaV1Config,
    _encode_selected_spatial_tokens,
    normalized_half_squared_jepa_loss_v1,
    normalized_half_squared_token_energy_v1,
)


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state()
    try:
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
    finally:
        torch.random.set_rng_state(caller_rng)


def _model(
    state: dict[str, torch.Tensor],
) -> SingleFrameMultiblockMaskedSpatialJepaV1:
    return SingleFrameMultiblockMaskedSpatialJepaV1(state).eval()


def _rgb(batch: int, *, offset: float = 0.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9321)
    return torch.randn(
        batch,
        3,
        112,
        112,
        generator=generator,
        dtype=torch.float32,
    ) + offset


def _four_block_target_indices(batch: int) -> torch.Tensor:
    # One valid 4x4 block in each 8x8 quadrant.
    specifications = (
        (0, 0, 0, 0),
        (0, 8, 4, 4),
        (8, 0, 2, 1),
        (8, 8, 1, 3),
    )
    indices: list[int] = []
    for base_row, base_col, row_offset, col_offset in specifications:
        for row in range(base_row + row_offset, base_row + row_offset + 4):
            for col in range(base_col + col_offset, base_col + col_offset + 4):
                indices.append(row * 16 + col)
    result = torch.tensor(sorted(indices), dtype=torch.long)
    assert result.numel() == 64
    assert torch.unique(result).numel() == 64
    return result.unsqueeze(0).expand(batch, -1).clone()


def _assert_nonzero_finite_gradient(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0


def test_v1_frozen_architecture_initialization_and_inventory(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(401)
    caller_rng = torch.random.get_rng_state().clone()
    model = _model(n320_encoder_state)

    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert torch.equal(
        model.predictor_position,
        n320_encoder_state["pos_embed"][:, 1:].squeeze(0),
    )
    assert model.predictor_position.shape == (256, 192)
    assert model.predictor_mask_token.shape == (1, 1, 192)
    assert len(model.encoder.blocks) == 6
    assert len(model.target_encoder.blocks) == 6
    assert len(model.predictor_blocks) == 2
    assert sum(isinstance(module, ViTBlock) for module in model.modules()) == 14
    assert model.predictor_blocks[0].mlp[0].out_features == 384
    assert model.predictor_blocks[0].attn.num_heads == 6

    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.predictor_position.requires_grad
    assert model.predictor_mask_token.requires_grad
    assert all(
        parameter.requires_grad
        for parameter in model.predictor_blocks.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )
    assert not model.target_encoder.training
    assert int(model.ema_update_count) == 0

    inventory = model.ema_inventory_exact()
    parameters = dict(model.named_parameters())
    assert inventory
    assert len(inventory) == len(tuple(model.encoder.named_parameters()))
    assert all(
        online.startswith("encoder.") and target.startswith("target_encoder.")
        for online, target in inventory
    )
    assert not any(
        "predictor" in online or "predictor" in target
        for online, target in inventory
    )
    for online, target in inventory:
        assert torch.equal(parameters[online], parameters[target])

    forbidden = (
        "action",
        "temporal",
        "memory",
        "proprio",
        "pose",
        "depth",
        "geometry",
        "object",
        "place",
    )
    assert not any(
        fragment in name
        for name, _ in model.named_parameters()
        for fragment in forbidden
    )
    assert tuple(inspect.signature(model.forward).parameters) == (
        "rgb",
        "target_indices",
        "capture_intermediates",
    )

    with pytest.raises(ValueError, match="constants cannot change"):
        SingleFrameMultiblockMaskedSpatialJepaV1Config(predictor_depth=3)


def test_v1_visible_online_path_and_original_order_predictor_scatter(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    rgb = _rgb(2)
    target_indices = _four_block_target_indices(2)

    output = model(
        rgb,
        target_indices,
        capture_intermediates=True,
    )
    prediction = output.prediction
    target = output.target

    assert prediction.raw_predicted_target_tokens.shape == (2, 64, 192)
    assert prediction.normalized_predicted_target_tokens.shape == (2, 64, 192)
    assert prediction.encoded_visible_tokens.shape == (2, 192, 192)
    assert prediction.visible_indices.shape == (2, 192)
    assert prediction.online_input is not None
    assert prediction.online_input.shape == (2, 193, 192)
    assert len(prediction.online_block_outputs) == 6
    assert all(
        value.shape == (2, 193, 192)
        for value in prediction.online_block_outputs
    )
    assert prediction.predictor_input is not None
    assert prediction.predictor_input.shape == (2, 256, 192)
    assert len(prediction.predictor_block_outputs) == 2
    assert all(
        value.shape == (2, 256, 192)
        for value in prediction.predictor_block_outputs
    )
    assert target.raw_target_tokens.shape == (2, 64, 192)
    assert target.normalized_target_tokens.shape == (2, 64, 192)
    assert not target.raw_target_tokens.requires_grad
    assert torch.allclose(
        prediction.normalized_predicted_target_tokens.norm(dim=-1),
        torch.ones(2, 64),
        atol=1e-5,
    )
    assert torch.allclose(
        target.normalized_target_tokens.norm(dim=-1),
        torch.ones(2, 64),
        atol=1e-5,
    )
    assert torch.isfinite(output.loss)

    complete = torch.cat(
        (prediction.target_indices, prediction.visible_indices),
        dim=1,
    ).sort(dim=1).values
    assert torch.equal(
        complete,
        torch.arange(256).expand(2, -1),
    )

    predictor_input = prediction.predictor_input
    expected_targets = (
        model.predictor_mask_token.expand(2, 64, -1)
        + model.predictor_position[target_indices]
    )
    assert torch.equal(
        predictor_input.gather(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, 192),
        ),
        expected_targets,
    )
    expected_visible = (
        prediction.encoded_visible_tokens
        + model.predictor_position[prediction.visible_indices]
    )
    assert torch.equal(
        predictor_input.gather(
            1,
            prediction.visible_indices.unsqueeze(-1).expand(-1, -1, 192),
        ),
        expected_visible,
    )


def test_v1_masked_pixels_cannot_reach_online_prediction(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    target_indices = _four_block_target_indices(1)
    rgb = _rgb(1).requires_grad_(True)
    changed = rgb.detach().clone()
    for flat_index in target_indices[0].tolist():
        row, col = divmod(flat_index, 16)
        changed[
            :,
            :,
            row * 7 : (row + 1) * 7,
            col * 7 : (col + 1) * 7,
        ] += 17.0

    original_prediction = model.forward_online(
        rgb,
        target_indices,
    ).raw_predicted_target_tokens
    changed_prediction = model.forward_online(
        changed,
        target_indices,
    ).raw_predicted_target_tokens
    assert torch.equal(original_prediction, changed_prediction)

    original_target = model.encode_target(rgb.detach(), target_indices)
    changed_target = model.encode_target(changed, target_indices)
    assert not torch.equal(
        original_target.raw_target_tokens,
        changed_target.raw_target_tokens,
    )

    original_prediction.square().mean().backward()
    assert rgb.grad is not None
    masked_gradient = []
    visible_gradient = []
    target_set = set(target_indices[0].tolist())
    for flat_index in range(256):
        row, col = divmod(flat_index, 16)
        patch_gradient = rgb.grad[
            :,
            :,
            row * 7 : (row + 1) * 7,
            col * 7 : (col + 1) * 7,
        ]
        if flat_index in target_set:
            masked_gradient.append(patch_gradient.reshape(-1))
        else:
            visible_gradient.append(patch_gradient.reshape(-1))
    assert torch.count_nonzero(torch.cat(masked_gradient)) == 0
    assert torch.count_nonzero(torch.cat(visible_gradient)) > 0


def test_v1_selected_encoder_full_and_empty_boundaries_match_full_helper(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    rgb = _rgb(1)
    full_indices = torch.arange(256, dtype=torch.long).unsqueeze(0)

    public_full = model.encode_online_full_frame(rgb)
    native_full = model.encoder.forward_tokens(rgb)[:, 1:]
    selected_full, selected_input, selected_blocks = (
        _encode_selected_spatial_tokens(
            model.encoder,
            rgb,
            full_indices,
            capture_intermediates=True,
        )
    )
    assert torch.equal(public_full, native_full)
    assert torch.equal(selected_full, native_full)
    assert selected_input is not None
    assert selected_input.shape == (1, 257, 192)
    assert all(value.shape == (1, 257, 192) for value in selected_blocks)

    empty_indices = torch.empty(1, 0, dtype=torch.long)
    selected_empty, empty_input, empty_blocks = (
        _encode_selected_spatial_tokens(
            model.encoder,
            rgb,
            empty_indices,
            capture_intermediates=True,
        )
    )
    assert selected_empty.shape == (1, 0, 192)
    assert empty_input is not None and empty_input.shape == (1, 1, 192)
    assert len(empty_blocks) == 6
    assert all(value.shape == (1, 1, 192) for value in empty_blocks)

    target_full = model.encode_target_full_frame(rgb)
    expected_target = model.target_encoder.forward_tokens(rgb)[:, 1:]
    assert torch.equal(target_full, expected_target)
    assert not target_full.requires_grad


def test_v1_joint_loss_trains_encoder_and_predictor_but_never_target(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    rgb = _rgb(1)
    target_indices = _four_block_target_indices(1)

    output = model(rgb, target_indices)
    expected = 0.5 * (
        F.normalize(
            output.prediction.raw_predicted_target_tokens,
            dim=-1,
            eps=1e-8,
        )
        - F.normalize(
            output.target.raw_target_tokens,
            dim=-1,
            eps=1e-8,
        )
    ).square().sum(dim=-1).mean()
    assert torch.equal(output.loss, expected)
    assert torch.equal(
        normalized_half_squared_token_energy_v1(
            output.prediction.raw_predicted_target_tokens,
            output.target.raw_target_tokens,
        ).mean(),
        output.loss,
    )

    output.loss.backward()
    for parameter in (
        model.encoder.patch_embed.weight,
        model.encoder.cls_token,
        model.encoder.blocks[0].attn.in_proj_weight,
        model.encoder.blocks[5].attn.in_proj_weight,
        model.predictor_position,
        model.predictor_mask_token,
        model.predictor_blocks[0].attn.in_proj_weight,
        model.predictor_blocks[1].attn.in_proj_weight,
        model.predictor_output.weight,
    ):
        _assert_nonzero_finite_gradient(parameter)
    assert not any(
        parameter.grad is not None
        for parameter in model.target_encoder.parameters()
    )

    detached_target_test = torch.randn(
        1,
        64,
        192,
        requires_grad=True,
    )
    prediction_test = torch.randn(
        1,
        64,
        192,
        requires_grad=True,
    )
    normalized_half_squared_jepa_loss_v1(
        prediction_test,
        detached_target_test,
    ).backward()
    assert prediction_test.grad is not None
    assert detached_target_test.grad is None


def test_v1_ema_is_exact_complete_and_excludes_predictor(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = _model(n320_encoder_state)
    parameters = dict(model.named_parameters())
    inventory = model.ema_inventory_exact()
    target_before = {
        target: parameters[target].detach().clone()
        for _, target in inventory
    }
    predictor_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith("predictor_")
    }
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.25)
        model.encoder.cls_token.add_(0.50)
        model.encoder.blocks[5].norm2.bias.add_(0.75)

    expected = {}
    for online, target in inventory:
        value = target_before[target].clone()
        value.mul_(0.996).add_(parameters[online], alpha=0.004)
        expected[target] = value

    model.train()
    model.update_target_ema()

    assert int(model.ema_update_count) == 1
    assert not model.target_encoder.training
    for _, target in inventory:
        assert torch.equal(parameters[target], expected[target])
        assert not parameters[target].requires_grad
    for name, before in predictor_before.items():
        assert torch.equal(parameters[name], before)

    model.hard_sync_target_from_online()
    assert int(model.ema_update_count) == 0
    for online, target in inventory:
        assert torch.equal(parameters[online], parameters[target])
    assert not model.target_encoder.training


def test_v1_rejects_invalid_state_rgb_masks_and_loss_inputs(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    missing = dict(n320_encoder_state)
    missing.pop("cls_token")
    with pytest.raises(ValueError, match="state keys changed"):
        _model(missing)

    wrong_dtype = {
        name: (
            value.to(torch.float64)
            if name == "cls_token"
            else value
        )
        for name, value in n320_encoder_state.items()
    }
    with pytest.raises(TypeError, match="exact float32"):
        _model(wrong_dtype)

    model = _model(n320_encoder_state)
    target_indices = _four_block_target_indices(1)
    with pytest.raises(TypeError, match="exact float32"):
        model(_rgb(1).to(torch.float64), target_indices)
    with pytest.raises(TypeError, match="long with shape"):
        model(_rgb(1), target_indices.to(torch.int32))
    with pytest.raises(TypeError, match="long with shape"):
        model(_rgb(1), target_indices[:, :-1])
    duplicated = target_indices.clone()
    duplicated[:, 1] = duplicated[:, 0]
    with pytest.raises(ValueError, match="strictly increasing"):
        model(_rgb(1), duplicated)
    reversed_indices = target_indices.flip(1)
    with pytest.raises(ValueError, match="strictly increasing"):
        model(_rgb(1), reversed_indices)

    prediction = torch.randn(1, 64, 192)
    with pytest.raises(ValueError, match="shape"):
        normalized_half_squared_jepa_loss_v1(
            prediction[:, :-1],
            prediction[:, :-1],
        )
    with pytest.raises(TypeError, match="exact float32"):
        normalized_half_squared_jepa_loss_v1(
            prediction.to(torch.float64),
            prediction.to(torch.float64),
        )
