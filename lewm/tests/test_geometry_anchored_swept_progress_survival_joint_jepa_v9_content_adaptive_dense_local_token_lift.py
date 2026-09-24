from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift import (
    DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
    DENSE_LOCAL_ATTENTION_HEADS_V9,
    DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
    DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
    DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
    DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9,
    DENSE_LOCAL_SUPPORT_COUNT_V9,
    DENSE_LOCAL_SUPPORT_SIDE_V9,
    ContentAdaptiveDenseLocalTokenBevLiftV9,
    ContentAdaptiveDenseLocalTokenLiftSamplingV9,
    ContentAdaptiveDenseLocalTokenLiftV9,
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
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
def clean_v4_and_v9(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
]:
    return (
        GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_encoder_state, _sweep_masks()
        ),
        GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            n320_encoder_state, _sweep_masks()
        ),
    )


def _attention_named_parameters(
    lift: ContentAdaptiveDenseLocalTokenLiftV9,
) -> dict[str, nn.Parameter]:
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


def test_v9_migrates_every_inherited_v4_tensor_exactly_and_isolates_rng(
    clean_v4_and_v9: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
    ],
) -> None:
    clean, model = clean_v4_and_v9
    assert isinstance(model.bev_lift, ContentAdaptiveDenseLocalTokenLiftV9)
    assert ContentAdaptiveDenseLocalTokenBevLiftV9 is (
        ContentAdaptiveDenseLocalTokenLiftV9
    )
    assert isinstance(model.target_bev_lift, ContentAdaptiveDenseLocalTokenLiftV9)
    assert not hasattr(model.bev_lift, "raw_offsets")
    assert not hasattr(model.bev_lift, "weight_logits")

    removed = {
        "bev_lift.raw_offsets",
        "bev_lift.weight_logits",
        "target_bev_lift.raw_offsets",
        "target_bev_lift.weight_logits",
    }
    clean_state = clean.state_dict()
    model_state = model.state_dict()
    for name, value in clean_state.items():
        if name not in removed:
            assert name in model_state, name
            assert torch.equal(value, model_state[name]), name

    new_names = set(model_state) - (set(clean_state) - removed)
    expected_new_names = {
        "bev_lift.support_offsets_token_cells",
        "target_bev_lift.support_offsets_token_cells",
    }
    attention_names = {
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    }
    expected_new_names.update(f"bev_lift.{name}" for name in attention_names)
    expected_new_names.update(
        f"target_bev_lift.{name}" for name in attention_names
    )
    assert new_names == expected_new_names

    online_attention = _attention_named_parameters(model.bev_lift)
    target_attention = _attention_named_parameters(model.target_bev_lift)
    assert set(online_attention) == attention_names == set(target_attention)
    assert len(online_attention) == DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9
    assert sum(value.numel() for value in online_attention.values()) == (
        DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9
    )
    assert model.bev_lift.key_projection.bias is None
    assert all(
        torch.equal(value, target_attention[name])
        for name, value in online_attention.items()
    )
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 0
    assert all(parameter.requires_grad for parameter in online_attention.values())
    assert not any(parameter.requires_grad for parameter in target_attention.values())
    assert not model.target_bev_lift.training

    torch.random.default_generator.manual_seed(7821)
    caller_rng = torch.random.get_rng_state().clone()
    replacement = ContentAdaptiveDenseLocalTokenLiftV9(clean.bev_lift)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9)
    for projection in (
        replacement.query_projection,
        replacement.key_projection,
        replacement.value_projection,
        replacement.output_projection,
    ):
        expected = torch.empty_like(projection.weight)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(projection.weight, expected)
        if projection.bias is not None:
            assert torch.count_nonzero(projection.bias) == 0


def test_dense_support_grid_order_masks_attention_sums_and_finite_nulls(
    clean_v4_and_v9: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
    ],
) -> None:
    _clean, model = clean_v4_and_v9
    lift = model.bev_lift.eval()
    tokens = torch.randn((1, 256, 192), generator=torch.Generator().manual_seed(31))
    receipt = lift.forward_with_sampling(tokens)
    assert isinstance(receipt, ContentAdaptiveDenseLocalTokenLiftSamplingV9)
    assert ContentAdaptiveDenseLocalTokenLiftSamplingV9._fields == (
        "latent",
        "anchor_in_frustum",
        "support_valid_mask",
        "cell_valid_mask",
        "support_grid_xy",
        "support_offsets_token_cells",
        "attention_weights",
    )
    assert (
        DENSE_LOCAL_SUPPORT_SIDE_V9,
        DENSE_LOCAL_SUPPORT_COUNT_V9,
        DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9,
        DENSE_LOCAL_ATTENTION_HEADS_V9,
        DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
    ) == (5, 25, 12, 4, 16)

    expected_offsets = torch.tensor(
        [(x, y) for y in range(-2, 3) for x in range(-2, 3)],
        dtype=torch.float32,
    )
    assert torch.equal(receipt.support_offsets_token_cells, expected_offsets)
    assert torch.equal(expected_offsets[12], torch.tensor((0.0, 0.0)))
    anchor = lift.anchor_grid_xy[None]
    proposed = anchor[..., None, :] + expected_offsets * (2.0 / 16.0)
    expected_valid = lift.anchor_in_frustum[None, ..., None] & (
        (proposed[..., 0] >= -1.0)
        & (proposed[..., 0] <= 1.0)
        & (proposed[..., 1] >= -1.0)
        & (proposed[..., 1] <= 1.0)
    )
    expected_safe = torch.where(
        expected_valid[..., None], proposed, torch.full_like(proposed, 2.0)
    )
    assert receipt.support_valid_mask.shape == (1, 64, 64, 25)
    assert receipt.support_grid_xy.shape == (1, 64, 64, 25, 2)
    assert torch.equal(receipt.support_valid_mask, expected_valid)
    assert torch.equal(receipt.support_grid_xy, expected_safe)
    assert torch.equal(
        receipt.cell_valid_mask, expected_valid.any(dim=-1)
    )
    assert torch.equal(receipt.cell_valid_mask, receipt.anchor_in_frustum)
    assert torch.equal(
        receipt.support_grid_xy[..., 12, :][receipt.cell_valid_mask],
        anchor.expand_as(receipt.support_grid_xy[..., 12, :])[
            receipt.cell_valid_mask
        ],
    )

    weights = receipt.attention_weights
    assert weights.shape == (1, 64, 64, 4, 25)
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

    assert receipt.latent.shape == (1, 64, 64, 64)
    assert bool(torch.isfinite(receipt.latent).all())
    invalid = ~receipt.cell_valid_mask
    expected_null = lift.null_evidence[None, :, None].expand(
        1, 64, int(invalid.sum())
    )
    assert torch.equal(receipt.latent[:, :, invalid[0]], expected_null)


def test_every_online_attention_tensor_gets_finite_nonzero_gradient(
    clean_v4_and_v9: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
    ],
) -> None:
    _clean, model = clean_v4_and_v9
    lift = copy.deepcopy(model.bev_lift).train()
    tokens = torch.randn(
        (1, 256, 192), generator=torch.Generator().manual_seed(43)
    ).requires_grad_(True)
    latent = lift(tokens)
    coefficients = torch.randn(
        latent.shape, generator=torch.Generator().manual_seed(47)
    )
    (latent * coefficients).mean().backward()

    assert tokens.grad is not None
    assert bool(torch.isfinite(tokens.grad).all())
    assert torch.count_nonzero(tokens.grad) > 0
    attention = _attention_named_parameters(lift)
    assert len(attention) == 7
    for name, parameter in attention.items():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_target_is_exact_frozen_gradient_free_and_ema_updated(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV9(
        n320_encoder_state, _sweep_masks()
    ).train()
    online = _attention_named_parameters(model.bev_lift)
    target = _attention_named_parameters(model.target_bev_lift)
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


def test_constructor_and_token_contract_fail_closed_under_mutation(
    clean_v4_and_v9: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
        GeometryAnchoredSweptProgressSurvivalJointJepaV9,
    ],
) -> None:
    clean, model = clean_v4_and_v9
    missing_parameter = copy.deepcopy(clean.bev_lift)
    del missing_parameter.raw_offsets
    with pytest.raises(RuntimeError, match="parameter inventory"):
        ContentAdaptiveDenseLocalTokenLiftV9(missing_parameter)

    changed_projection = copy.deepcopy(clean.bev_lift)
    changed_projection.token_projection.bias = None
    with pytest.raises(RuntimeError, match="token projection"):
        ContentAdaptiveDenseLocalTokenLiftV9(changed_projection)

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
