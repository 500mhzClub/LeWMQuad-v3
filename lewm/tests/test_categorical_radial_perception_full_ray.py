from __future__ import annotations

import torch
import torch.nn as nn

from lewm.benchmarks.go2_categorical_radial_factorization import (
    gather_polar_logits_to_cartesian,
)
from lewm.models.categorical_radial_perception import (
    CategoricalRadialPerception,
    IMAGE_SIZE,
    POLAR_SHAPE,
)
from lewm.models.categorical_radial_perception_full_ray import (
    CategoricalRadialPerceptionFullRay,
    FullRayRadialContext,
    RADIAL_DILATIONS,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_RADIAL_BIN_COUNT,
    direct_radial_reachability,
)


def test_registered_dilations_blocks_and_exact_parameter_count() -> None:
    torch.manual_seed(17)
    model = CategoricalRadialPerceptionFullRay()
    blocks = tuple(model.radial_context)

    assert isinstance(model, CategoricalRadialPerception)
    assert tuple(block.dilation for block in blocks) == RADIAL_DILATIONS
    assert len(blocks) == 6
    assert all(block.radial_conv.kernel_size == (3, 1) for block in blocks)
    assert all(
        block.radial_conv.dilation == (dilation, 1)
        for block, dilation in zip(blocks, RADIAL_DILATIONS)
    )
    assert all(
        block.radial_conv.padding == (dilation, 0)
        for block, dilation in zip(blocks, RADIAL_DILATIONS)
    )
    assert all(block.radial_conv.padding_mode == "zeros" for block in blocks)
    assert all(block.norm.num_groups == 8 for block in blocks)
    assert all(isinstance(block.activation, nn.GELU) for block in blocks)
    assert all(block.pointwise.kernel_size == (1, 1) for block in blocks)
    block_parameter_counts = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in blocks
    ]
    assert block_parameter_counts == [16_640] * 6
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        REGISTERED_PARAMETER_COUNT
    )


def test_binary_direct_reachability_is_complete_and_each_layer_is_clipped() -> None:
    adjacencies, reachability = direct_radial_reachability()
    indices = torch.arange(REGISTERED_RADIAL_BIN_COUNT)
    output_index = indices[:, None]
    input_index = indices[None, :]

    assert len(adjacencies) == len(RADIAL_DILATIONS)
    for adjacency, dilation in zip(adjacencies, RADIAL_DILATIONS):
        expected = ((output_index - input_index).abs() == dilation) | (
            output_index == input_index
        )
        assert adjacency.shape == (64, 64)
        assert torch.equal(adjacency, expected)
        assert adjacency[0, 0]
        assert not adjacency[0, -1]
        assert not adjacency[-1, 0]
    assert reachability.shape == (64, 64)
    assert bool(reachability.all())
    assert reachability[0, 63]
    assert reachability[63, 0]


def _linearized_context() -> FullRayRadialContext:
    context = FullRayRadialContext(8)
    with torch.no_grad():
        for block in context:
            block.norm = nn.Identity()
            block.activation = nn.Identity()
            block.radial_conv.weight.zero_()
            block.radial_conv.bias.zero_()
            block.pointwise.weight.zero_()
            block.pointwise.bias.zero_()
            for channel in range(8):
                block.radial_conv.weight[channel, channel, 0, 0] = 1.0
                block.radial_conv.weight[channel, channel, 2, 0] = 1.0
                block.pointwise.weight[channel, channel, 0, 0] = 1.0
    return context


def test_explicit_linearized_convolution_path_reaches_both_opposite_edges() -> None:
    context = _linearized_context()
    from_near = torch.zeros(1, 8, 64, 1)
    from_near[0, 0, 0, 0] = 1.0
    from_far = torch.zeros_like(from_near)
    from_far[0, 0, 63, 0] = 1.0

    near_output = context(from_near)
    far_output = context(from_far)

    assert near_output[0, 0, 63, 0] > 0
    assert far_output[0, 0, 0, 0] > 0


def test_individual_registered_convolutions_do_not_wrap() -> None:
    context = FullRayRadialContext(8)
    for block in context:
        with torch.no_grad():
            block.radial_conv.weight.zero_()
            block.radial_conv.bias.zero_()
            block.radial_conv.weight[0, 0, :, 0] = 1.0
        near = torch.zeros(1, 8, 64, 1)
        near[0, 0, 0, 0] = 1.0
        far = torch.zeros_like(near)
        far[0, 0, 63, 0] = 1.0
        near_output = block.radial_conv(near)
        far_output = block.radial_conv(far)

        assert near_output[0, 0, 0, 0] == 1.0
        assert far_output[0, 0, 63, 0] == 1.0
        assert near_output[0, 0, 63, 0] == 0.0
        assert far_output[0, 0, 0, 0] == 0.0


def test_common_v2_initialization_is_bitwise_identical() -> None:
    torch.manual_seed(20260710)
    v2_model = CategoricalRadialPerception()
    torch.manual_seed(20260710)
    v3_model = CategoricalRadialPerceptionFullRay()
    v2_state = v2_model.state_dict()
    v3_state = v3_model.state_dict()
    common_names = {
        name for name in v2_state if not name.startswith("radial_context.")
    }

    assert common_names == {
        name for name in v3_state if not name.startswith("radial_context.")
    }
    assert common_names
    assert all(torch.equal(v2_state[name], v3_state[name]) for name in common_names)


def test_polar_cartesian_shapes_gather_support_and_finiteness() -> None:
    torch.manual_seed(31)
    model = CategoricalRadialPerceptionFullRay().eval()
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        polar = model.polar_logits(image)
        cartesian = model(image)
        gathered = gather_polar_logits_to_cartesian(
            polar,
            factorization=model.factorization,
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


def test_registered_widths_cannot_be_changed() -> None:
    for keyword in ({"token_feature_dim": 16}, {"context_dim": 32}):
        try:
            CategoricalRadialPerceptionFullRay(**keyword)
        except ValueError as error:
            assert "frozen" in str(error)
        else:
            raise AssertionError("registered full-ray width change was accepted")
