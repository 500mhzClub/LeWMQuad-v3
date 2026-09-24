from __future__ import annotations

import copy
import hashlib
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift import (
    CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
    CELL_VOLUME_ATTENTION_HEADS_V10,
    CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
    CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
    CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
    CELL_VOLUME_HEIGHTS_M_V10,
    CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10,
    CELL_VOLUME_SUPPORT_COUNT_V10,
    CELL_VOLUME_VALID_CELL_COUNT_V10,
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10,
    GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ProjectiveCellVolumeTokenBevLiftV10,
    ProjectiveCellVolumeTokenLiftSamplingV10,
    ProjectiveCellVolumeTokenLiftV10,
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
def clean_v4_v9_v10(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
    GeometryAnchoredSweptProgressSurvivalJointJepaV10,
]:
    return (
        GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_encoder_state, _sweep_masks()
        ).eval(),
        GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            n320_encoder_state, _sweep_masks()
        ).eval(),
        GeometryAnchoredSweptProgressSurvivalJointJepaV10(
            n320_encoder_state, _sweep_masks()
        ).eval(),
    )


def _attention_named_parameters(lift: nn.Module) -> dict[str, nn.Parameter]:
    prefixes = (
        "query_projection.",
        "key_projection.",
        "value_projection.",
        "output_projection.",
    )
    return {
        name: parameter
        for name, parameter in lift.named_parameters()
        if name.startswith(prefixes)
    }


def test_v10_preserves_every_v9_parameter_and_attention_initialization(
    clean_v4_v9_v10: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ],
) -> None:
    clean, v9, model = clean_v4_v9_v10
    assert isinstance(model.bev_lift, ProjectiveCellVolumeTokenLiftV10)
    assert ProjectiveCellVolumeTokenBevLiftV10 is ProjectiveCellVolumeTokenLiftV10
    assert isinstance(model.target_bev_lift, ProjectiveCellVolumeTokenLiftV10)
    assert not hasattr(model.bev_lift, "raw_offsets")
    assert not hasattr(model.bev_lift, "weight_logits")
    assert not hasattr(model.bev_lift, "support_offsets_token_cells")

    v9_parameters = dict(v9.named_parameters())
    v10_parameters = dict(model.named_parameters())
    assert v10_parameters.keys() == v9_parameters.keys()
    for name, value in v9_parameters.items():
        assert torch.equal(value, v10_parameters[name]), name

    attention_names = {
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    }
    online_attention = _attention_named_parameters(model.bev_lift)
    target_attention = _attention_named_parameters(model.target_bev_lift)
    assert set(online_attention) == attention_names == set(target_attention)
    assert len(online_attention) == CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10
    assert sum(value.numel() for value in online_attention.values()) == (
        CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10
    )
    assert (
        CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
        CELL_VOLUME_ATTENTION_HEADS_V10,
        CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
    ) == (20260729, 4, 16)
    assert all(
        torch.equal(value, target_attention[name])
        for name, value in online_attention.items()
    )
    assert all(value.requires_grad for value in online_attention.values())
    assert not any(value.requires_grad for value in target_attention.values())
    assert not model.target_bev_lift.training
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 0

    torch.random.default_generator.manual_seed(8173)
    caller_rng = torch.random.get_rng_state().clone()
    replacement = ProjectiveCellVolumeTokenLiftV10(clean.bev_lift)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    v9_attention = _attention_named_parameters(v9.bev_lift)
    replacement_attention = _attention_named_parameters(replacement)
    assert all(
        torch.equal(value, replacement_attention[name])
        for name, value in v9_attention.items()
    )


def test_static_cell_volume_geometry_order_projection_counts_and_hash(
    clean_v4_v9_v10: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ],
) -> None:
    _clean, _v9, model = clean_v4_v9_v10
    lift = model.bev_lift
    assert torch.equal(
        lift.support_offsets_xy_m,
        torch.tensor(CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10),
    )
    assert torch.equal(
        lift.support_heights_m, torch.tensor(CELL_VOLUME_HEIGHTS_M_V10)
    )
    assert lift.support_xyz_m.shape == (64, 64, 25, 3)
    points = lift.support_xyz_m.reshape(64, 64, 5, 5, 3)
    assert torch.equal(points[:, :, 0, 0], lift.bev_ground_xyz_m)
    assert torch.equal(
        points[..., 2],
        lift.support_heights_m[None, None, None, :].expand(64, 64, 5, 5),
    )
    assert torch.equal(
        points[..., :2], points[..., :1, :2].expand(64, 64, 5, 5, 2)
    )

    forward = torch.linspace(-0.95, 5.35, 64, dtype=torch.float64)
    left = torch.linspace(-3.15, 3.15, 64, dtype=torch.float64)
    forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
    centres = torch.stack((forward_grid, left_grid), dim=-1)
    horizontal = centres[..., None, :] + torch.tensor(
        CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10, dtype=torch.float64
    )
    assert torch.equal(points[:, :, :, 0, :2], horizontal.to(torch.float32))

    relative = lift.support_xyz_m.to(torch.float64) - torch.tensor(
        (0.326, 0.0, 0.043), dtype=torch.float64
    )
    depth = relative[..., 0]
    expected_x = -relative[..., 1] / (
        depth * math.tan(math.radians(78.323) / 2.0)
    )
    expected_y = -relative[..., 2] / (
        depth * math.tan(math.radians(62.8370386364) / 2.0)
    )
    expected_valid = (
        (depth >= 0.05)
        & (expected_x >= -1.0)
        & (expected_x <= 1.0)
        & (expected_y >= -1.0)
        & (expected_y <= 1.0)
    )
    assert torch.equal(lift.support_valid_mask, expected_valid)
    assert torch.equal(lift.cell_valid_mask, expected_valid.any(dim=-1))
    assert int(lift.cell_valid_mask.sum()) == CELL_VOLUME_VALID_CELL_COUNT_V10
    payload = bytes(
        lift.cell_valid_mask.to(torch.uint8).reshape(-1).tolist()
    )
    assert hashlib.sha256(payload).hexdigest() == (
        CELL_VOLUME_VALID_MASK_SHA256_V10
    )
    invalid_grid = lift.support_grid_xy[~lift.support_valid_mask]
    assert torch.equal(invalid_grid, torch.full_like(invalid_grid, 2.0))
    expected_grid = torch.stack((expected_x, expected_y), dim=-1).to(torch.float32)
    torch.testing.assert_close(
        lift.support_grid_xy[lift.support_valid_mask],
        expected_grid[lift.support_valid_mask],
        rtol=1e-6,
        atol=1e-6,
    )

    within_two_metres = torch.linalg.vector_norm(centres, dim=-1) <= 2.0
    assert int(within_two_metres.sum()) == (
        CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10
    )
    assert int((within_two_metres & lift.cell_valid_mask).sum()) == (
        CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10
    )
    assert bool((lift.cell_valid_mask & ~lift.anchor_in_frustum).any())


def test_masked_mean_is_query_and_base_with_exact_attention_and_null_masks(
    clean_v4_v9_v10: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ],
) -> None:
    _clean, _v9, model = clean_v4_v9_v10
    lift = copy.deepcopy(model.bev_lift).eval()
    tokens = torch.randn(
        (1, 256, 192), generator=torch.Generator().manual_seed(31)
    )
    query_inputs: list[torch.Tensor] = []

    def _capture_query_input(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        query_inputs.append(inputs[0].detach().clone())

    handle = lift.query_projection.register_forward_pre_hook(_capture_query_input)
    try:
        receipt = lift.forward_with_sampling(tokens)
    finally:
        handle.remove()
    assert isinstance(receipt, ProjectiveCellVolumeTokenLiftSamplingV10)
    assert ProjectiveCellVolumeTokenLiftSamplingV10._fields == (
        "latent",
        "anchor_in_frustum",
        "support_valid_mask",
        "cell_valid_mask",
        "support_grid_xy",
        "support_xyz_m",
        "support_offsets_xy_m",
        "support_heights_m",
        "masked_mean",
        "attention_weights",
    )
    assert receipt.support_valid_mask.shape == (1, 64, 64, 25)
    assert receipt.support_grid_xy.shape == (1, 64, 64, 25, 2)
    assert receipt.masked_mean.shape == (1, 64, 64, 64)
    assert receipt.attention_weights.shape == (1, 64, 64, 4, 25)
    assert torch.equal(receipt.cell_valid_mask, receipt.support_valid_mask.any(-1))

    token_map = tokens.transpose(1, 2).reshape(1, 192, 16, 16)
    projected = lift.token_projection(token_map)
    sampled = F.grid_sample(
        projected,
        receipt.support_grid_xy.reshape(1, 64, 64 * 25, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(1, 64, 64, 64, 25)
    samples = sampled.permute(0, 2, 3, 4, 1).contiguous()
    samples = torch.where(
        receipt.support_valid_mask[..., None], samples, torch.zeros_like(samples)
    )
    counts = receipt.support_valid_mask.sum(-1, keepdim=True).to(torch.float32)
    expected_mean = samples.sum(-2) / counts.clamp_min(1.0)
    assert torch.equal(receipt.masked_mean, expected_mean)
    assert torch.count_nonzero(
        receipt.masked_mean[~receipt.cell_valid_mask]
    ) == 0
    assert len(query_inputs) == 1
    expected_queries = receipt.masked_mean.reshape(-1, 64)[
        receipt.cell_valid_mask.reshape(-1)
    ]
    assert torch.equal(query_inputs[0], expected_queries)

    weights = receipt.attention_weights
    assert bool(torch.isfinite(weights).all())
    assert torch.count_nonzero(
        weights.masked_select(~receipt.support_valid_mask[..., None, :])
    ) == 0
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(
        sums[receipt.cell_valid_mask],
        torch.ones_like(sums[receipt.cell_valid_mask]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.count_nonzero(sums[~receipt.cell_valid_mask]) == 0

    latent_channel_last = receipt.latent.permute(0, 2, 3, 1)
    expected_null = lift.null_evidence[None].expand(
        int((~receipt.cell_valid_mask).sum()), -1
    )
    assert torch.equal(
        latent_channel_last[~receipt.cell_valid_mask], expected_null
    )


def test_cell_volume_validity_controls_exact_semantic_unknown_mask(
    clean_v4_v9_v10: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ],
) -> None:
    _clean, _v9, model = clean_v4_v9_v10
    latent = torch.randn(
        (2, 64, 64, 64), generator=torch.Generator().manual_seed(53)
    )
    with torch.no_grad():
        raw = model.semantic_head(latent)
        masked = model.semantic_logits_from_latent(latent)
    valid = model.bev_lift.cell_valid_mask
    invalid = ~valid
    assert torch.equal(masked[:, :, valid], raw[:, :, valid])
    expected_unknown = torch.tensor((0.0, -20.0, -20.0))[
        None, :, None
    ].expand(2, -1, int(invalid.sum()))
    assert torch.equal(masked[:, :, invalid], expected_unknown)

    volume_only = valid & ~model.bev_lift.anchor_in_frustum
    assert bool(volume_only.any())
    assert torch.equal(masked[:, :, volume_only], raw[:, :, volume_only])


def test_all_attention_gradients_and_mirrored_ema_are_finite_and_nonzero(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV10(
        n320_encoder_state, _sweep_masks()
    ).train()
    lift = model.bev_lift
    tokens = torch.randn(
        (1, 256, 192), generator=torch.Generator().manual_seed(61)
    ).requires_grad_(True)
    latent = lift(tokens)
    coefficients = torch.randn(
        latent.shape, generator=torch.Generator().manual_seed(67)
    )
    (latent * coefficients).mean().backward()
    assert tokens.grad is not None
    assert bool(torch.isfinite(tokens.grad).all())
    assert torch.count_nonzero(tokens.grad) > 0
    online = _attention_named_parameters(lift)
    target = _attention_named_parameters(model.target_bev_lift)
    for name, parameter in online.items():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert all(value.grad is None for value in target.values())
    assert not any(value.requires_grad for value in target.values())

    target_before = {
        name: value.detach().clone() for name, value in target.items()
    }
    with torch.no_grad():
        for index, value in enumerate(online.values(), start=1):
            value.add_(index * 0.01)
    model.update_target_ema_after_optimizer_step()
    momentum = model.config.target_ema_momentum
    for name, target_value in target.items():
        expected = target_before[name].clone()
        expected.mul_(momentum).add_(online[name], alpha=1.0 - momentum)
        torch.testing.assert_close(target_value, expected, rtol=0.0, atol=0.0)
        assert not target_value.requires_grad
        assert target_value.grad is None
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 1
    assert not model.target_bev_lift.training


def test_token_contract_fails_closed(
    clean_v4_v9_v10: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ],
) -> None:
    _clean, _v9, model = clean_v4_v9_v10
    lift = model.bev_lift
    with pytest.raises(TypeError, match="tensor"):
        lift(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        lift(torch.zeros((1, 255, 192)))
    with pytest.raises(ValueError, match="at least one"):
        lift(torch.zeros((0, 256, 192)))
    with pytest.raises(TypeError, match="float32"):
        lift(torch.zeros((1, 256, 192), dtype=torch.float64))
    nonfinite = torch.zeros((1, 256, 192))
    nonfinite[0, 0, 0] = math.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        lift(nonfinite)
