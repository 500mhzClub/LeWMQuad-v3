from __future__ import annotations

import pytest
import torch

from lewm.models.categorical_radial_perception import (
    CategoricalRadialPerception,
    ENCODER_DEPTH,
    ENCODER_DIM,
    ENCODER_HEADS,
    IMAGE_SIZE,
    PATCH_SIZE,
    POLAR_SHAPE,
    TOKEN_SIDE,
    VERTICAL_ANCHOR_Z_BODY_M,
)


@pytest.fixture(scope="module")
def model() -> CategoricalRadialPerception:
    torch.manual_seed(17)
    return CategoricalRadialPerception().eval()


@pytest.fixture(scope="module")
def zero_outputs(
    model: CategoricalRadialPerception,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        return model.polar_logits(image), model(image)


def test_registered_encoder_and_output_shapes_are_finite(
    model: CategoricalRadialPerception,
    zero_outputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    polar, cartesian = zero_outputs

    assert model.encoder.image_size == IMAGE_SIZE
    assert model.encoder.patch_size == PATCH_SIZE
    assert model.encoder.hidden_dim == ENCODER_DIM
    assert len(model.encoder.blocks) == ENCODER_DEPTH
    assert model.encoder.blocks[0].attn.num_heads == ENCODER_HEADS
    assert model.encoder.num_patches == TOKEN_SIDE * TOKEN_SIDE
    assert model.radial_context.net[0].kernel_size == (5, 1)
    assert model.angular_context.net[0].kernel_size == (1, 5)
    assert polar.shape == (1, 3, *POLAR_SHAPE)
    assert cartesian.shape == (1, 3, 64, 64)
    assert torch.isfinite(polar).all()
    assert torch.isfinite(cartesian).all()


def test_anchor_projection_is_separate_and_height_distinct(
    model: CategoricalRadialPerception,
) -> None:
    grid = model.projective_sample_grid
    validity = model.projective_anchor_validity

    assert grid.shape == (len(VERTICAL_ANCHOR_Z_BODY_M), *POLAR_SHAPE, 2)
    assert validity.shape == (len(VERTICAL_ANCHOR_Z_BODY_M), *POLAR_SHAPE)
    for first in range(len(VERTICAL_ANCHOR_Z_BODY_M)):
        assert validity[first].any()
        for second in range(first + 1, len(VERTICAL_ANCHOR_Z_BODY_M)):
            jointly_valid = validity[first] & validity[second]
            assert jointly_valid.any()
            assert not torch.equal(
                grid[first, jointly_valid],
                grid[second, jointly_valid],
            )

    projected = torch.zeros(1, model.token_feature_dim, TOKEN_SIDE, TOKEN_SIDE)
    vertical_ramp = torch.linspace(-1.0, 1.0, TOKEN_SIDE).view(1, 1, -1, 1)
    projected[:] = vertical_ramp
    sampled = model.sample_projective_anchors(projected)
    assert sampled.shape == (
        1,
        len(VERTICAL_ANCHOR_Z_BODY_M),
        model.token_feature_dim,
        *POLAR_SHAPE,
    )
    anchor_means = []
    for anchor in range(len(VERTICAL_ANCHOR_Z_BODY_M)):
        valid = validity[anchor]
        anchor_means.append(sampled[0, anchor, 0][valid].mean())
    assert torch.unique(torch.stack(anchor_means)).numel() == len(anchor_means)


def test_outside_factorization_support_is_deterministically_unknown(
    model: CategoricalRadialPerception,
    zero_outputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _polar, cartesian = zero_outputs
    outside = ~model.cartesian_support_mask
    minimum = torch.finfo(cartesian.dtype).min

    assert outside.any()
    assert torch.equal(
        cartesian[0, 0][outside],
        torch.zeros_like(cartesian[0, 0][outside]),
    )
    assert torch.equal(
        cartesian[0, 1][outside],
        torch.full_like(cartesian[0, 1][outside], minimum),
    )
    assert torch.equal(
        cartesian[0, 2][outside],
        torch.full_like(cartesian[0, 2][outside], minimum),
    )
    assert torch.equal(
        cartesian.argmax(dim=1)[0][outside],
        torch.zeros_like(outside[outside], dtype=torch.long),
    )


def test_cartesian_output_preserves_image_gradients(
    model: CategoricalRadialPerception,
) -> None:
    model.zero_grad(set_to_none=True)
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)
    cartesian = model(image)
    support = model.cartesian_support_mask
    loss = cartesian[0, :, support].square().mean()
    loss.backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert model.token_projection.weight.grad is not None
    assert model.token_projection.weight.grad.abs().sum() > 0
    assert model.polar_head.weight.grad is not None
    assert model.polar_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "bad_image",
    (
        torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 3, IMAGE_SIZE - 1, IMAGE_SIZE),
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
    ),
)
def test_wrong_image_contract_is_rejected(
    model: CategoricalRadialPerception,
    bad_image: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="image must"):
        model(bad_image)


def test_wrong_projected_token_shape_is_rejected(
    model: CategoricalRadialPerception,
) -> None:
    with pytest.raises(ValueError, match="projected_tokens must"):
        model.sample_projective_anchors(
            torch.zeros(1, model.token_feature_dim, TOKEN_SIDE)
        )
    with pytest.raises(ValueError, match="projected_tokens must"):
        model.sample_projective_anchors(
            torch.zeros(1, model.token_feature_dim + 1, TOKEN_SIDE, TOKEN_SIDE)
        )


def test_evaluation_is_deterministic(
    model: CategoricalRadialPerception,
) -> None:
    generator = torch.Generator().manual_seed(29)
    image = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=generator,
    )
    with torch.no_grad():
        first = model(image)
        second = model(image)
    assert torch.equal(first, second)
