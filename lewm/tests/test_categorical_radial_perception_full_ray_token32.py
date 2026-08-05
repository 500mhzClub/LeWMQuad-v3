from __future__ import annotations

import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_factorization import (
    gather_polar_logits_to_cartesian,
)
from lewm.models.categorical_radial_perception import (
    IMAGE_SIZE,
    POLAR_SHAPE,
    TOKEN_SIDE,
    VERTICAL_ANCHOR_Z_BODY_M,
)
from lewm.models.categorical_radial_perception_full_ray import (
    RADIAL_DILATIONS,
    direct_radial_reachability,
)
from lewm.models.categorical_radial_perception_full_ray_token32 import (
    CategoricalRadialPerceptionFullRayToken32,
    REGISTERED_CONTEXT_DIM,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_SHAPE_CHANGED_STATE_KEYS,
    REGISTERED_SHAPE_CHANGES,
    REGISTERED_TOKEN_FEATURE_DIM,
    build_comparable_width24_and_token32_models,
)


def test_registered_architecture_has_only_the_token32_width_change() -> None:
    torch.manual_seed(17)
    model = CategoricalRadialPerceptionFullRayToken32()

    assert model.token_feature_dim == REGISTERED_TOKEN_FEATURE_DIM == 32
    assert model.context_dim == REGISTERED_CONTEXT_DIM == 64
    assert tuple(model.token_projection.weight.shape) == (32, 192, 1, 1)
    assert tuple(model.token_projection.bias.shape) == (32,)
    assert tuple(model.context_stem[0].weight.shape) == (64, 194, 1, 1)
    assert tuple(block.dilation for block in model.radial_context) == RADIAL_DILATIONS
    _adjacencies, reachability = direct_radial_reachability()
    assert reachability.shape == (64, 64)
    assert bool(reachability.all())
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        REGISTERED_PARAMETER_COUNT
    )


def test_projected_sample_polar_and_cartesian_shapes_are_unchanged() -> None:
    torch.manual_seed(31)
    model = CategoricalRadialPerceptionFullRayToken32().eval()
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        projected = model._projected_token_map(image)
        sampled = model.sample_projective_anchors(projected)
        polar = model.polar_logits(image)
        cartesian = model(image)
        gathered = gather_polar_logits_to_cartesian(
            polar,
            factorization=model.factorization,
        )

    assert projected.shape == (1, 32, TOKEN_SIDE, TOKEN_SIDE)
    assert sampled.shape == (
        1,
        len(VERTICAL_ANCHOR_Z_BODY_M),
        32,
        *POLAR_SHAPE,
    )
    assert polar.shape == (1, 3, *POLAR_SHAPE)
    assert cartesian.shape == (1, 3, 64, 64)
    assert torch.isfinite(polar).all()
    assert torch.isfinite(cartesian).all()
    assert torch.equal(cartesian, gathered)
    outside = ~model.cartesian_support_mask
    assert outside.any()
    assert torch.equal(
        cartesian.argmax(dim=1)[0][outside],
        torch.zeros_like(outside[outside], dtype=torch.long),
    )


def test_comparable_builder_changes_exactly_three_shapes_and_copies_the_rest() -> None:
    torch.manual_seed(20260710)
    initial_rng_state = torch.get_rng_state().clone()

    torch.set_rng_state(initial_rng_state)
    raw_token32 = CategoricalRadialPerceptionFullRayToken32()
    raw_token32_state = {
        name: value.detach().clone()
        for name, value in raw_token32.state_dict().items()
    }
    expected_final_rng_state = torch.get_rng_state().clone()

    width24, token32 = build_comparable_width24_and_token32_models(
        initial_rng_state
    )
    width24_state = width24.state_dict()
    token32_state = token32.state_dict()
    changed = {
        name: (tuple(width24_state[name].shape), tuple(token32_state[name].shape))
        for name in width24_state
        if width24_state[name].shape != token32_state[name].shape
    }

    assert changed == REGISTERED_SHAPE_CHANGES
    assert tuple(changed) == REGISTERED_SHAPE_CHANGED_STATE_KEYS
    assert tuple(width24_state) == tuple(token32_state)
    assert torch.equal(torch.get_rng_state(), expected_final_rng_state)
    same_shape_names = [name for name in width24_state if name not in changed]
    assert len(same_shape_names) == 130
    for name in width24_state:
        if name in changed:
            assert torch.equal(token32_state[name], raw_token32_state[name])
        else:
            assert width24_state[name].dtype == token32_state[name].dtype
            assert torch.equal(width24_state[name], token32_state[name])


def test_token32_backward_reaches_encoder_projection_and_full_ray_context() -> None:
    torch.manual_seed(43)
    model = CategoricalRadialPerceptionFullRayToken32().train()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)

    loss = model.polar_logits(image).square().mean()
    loss.backward()

    registered_gradients = {
        "encoder": model.encoder.patch_embed.weight.grad,
        "token_projection": model.token_projection.weight.grad,
        "context_stem": model.context_stem[0].weight.grad,
        "full_ray_first": model.radial_context[0].radial_conv.weight.grad,
        "full_ray_last": model.radial_context[-1].pointwise.weight.grad,
    }
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    for name, gradient in registered_gradients.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


@pytest.mark.parametrize(
    "keyword",
    ({"token_feature_dim": 24}, {"context_dim": 32}),
)
def test_registered_widths_cannot_be_changed(keyword: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        CategoricalRadialPerceptionFullRayToken32(**keyword)


@pytest.mark.parametrize(
    "bad_state",
    (
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(2, 2, dtype=torch.uint8),
    ),
)
def test_comparable_builder_rejects_invalid_rng_state(
    bad_state: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="CPU uint8"):
        build_comparable_width24_and_token32_models(bad_state)


def test_comparable_builder_rejects_non_tensor_rng_state() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        build_comparable_width24_and_token32_models(b"not a tensor")  # type: ignore[arg-type]
