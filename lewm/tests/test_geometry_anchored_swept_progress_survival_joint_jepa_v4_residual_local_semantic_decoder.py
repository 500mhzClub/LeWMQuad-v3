from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v1 import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV1,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4,
    RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4,
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    ResidualLocalSemanticDecoderV4,
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
def matched_models(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV1,
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
]:
    v1 = GeometryAnchoredSweptProgressSurvivalJointJepaV1(
        n320_encoder_state, _sweep_masks()
    ).eval()
    v4 = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
        n320_encoder_state, _sweep_masks()
    ).eval()
    return v1, v4


def test_architecture_base_identity_and_exact_added_parameter_count(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV1,
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    ],
) -> None:
    v1, v4 = matched_models
    head = v4.semantic_head
    assert isinstance(head, ResidualLocalSemanticDecoderV4)
    assert isinstance(head.base, nn.Conv2d)
    assert (head.base.in_channels, head.base.out_channels) == (64, 3)
    assert head.base.kernel_size == (1, 1) and head.base.bias is not None
    assert isinstance(head.local, nn.Conv2d)
    assert (head.local.in_channels, head.local.out_channels) == (64, 64)
    assert head.local.kernel_size == (3, 3)
    assert head.local.padding == (1, 1) and head.local.bias is not None
    assert isinstance(head.activation, nn.GELU)
    assert head.activation.approximate == "none"
    assert isinstance(head.residual_output, nn.Conv2d)
    assert (head.residual_output.in_channels, head.residual_output.out_channels) == (
        64,
        3,
    )
    assert head.residual_output.kernel_size == (1, 1)
    assert head.residual_output.bias is not None
    assert torch.count_nonzero(head.residual_output.weight) == 0
    assert torch.count_nonzero(head.residual_output.bias) == 0

    for name, value in v1.semantic_head.state_dict().items():
        assert torch.equal(value, head.base.state_dict()[name])
    assert (
        sum(parameter.numel() for parameter in v4.parameters())
        - sum(parameter.numel() for parameter in v1.parameters())
        == RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4
        == 37_123
    )
    assert RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4 == 1


def test_constructor_preserves_caller_rng_and_is_deterministic_across_callers(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    original = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(12345)
        caller_a = torch.random.get_rng_state().clone()
        model_a = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_encoder_state, _sweep_masks()
        )
        assert torch.equal(torch.random.get_rng_state(), caller_a)

        torch.random.default_generator.manual_seed(67890)
        caller_b = torch.random.get_rng_state().clone()
        model_b = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_encoder_state, _sweep_masks()
        )
        assert torch.equal(torch.random.get_rng_state(), caller_b)

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        assert state_a.keys() == state_b.keys()
        assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    finally:
        torch.random.set_rng_state(original)


def test_initial_semantic_logits_are_bitwise_v1_including_visibility_mask(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV1,
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    ],
) -> None:
    v1, v4 = matched_models
    generator = torch.Generator().manual_seed(7721)
    latent = torch.randn((2, 64, 64, 64), generator=generator)
    with torch.no_grad():
        v1_logits = v1.semantic_logits_from_latent(latent)
        v4_logits = v4.semantic_logits_from_latent(latent)
    assert torch.equal(
        v1_logits.contiguous().view(torch.int32),
        v4_logits.contiguous().view(torch.int32),
    )

    invisible = ~v4.bev_lift.anchor_in_frustum
    assert bool(invisible.any())
    expected = torch.tensor((0.0, -20.0, -20.0))[:, None].expand(
        -1, int(invisible.sum())
    )
    assert torch.equal(v4_logits[0, :, invisible], expected)


def test_residual_branch_has_exact_radius_one_local_receptive_field() -> None:
    base = nn.Conv2d(64, 3, kernel_size=1, bias=True)
    decoder = ResidualLocalSemanticDecoderV4(base, initialization_seed=20260713)
    with torch.no_grad():
        decoder.base.weight.zero_()
        decoder.base.bias.zero_()
        decoder.local.weight.zero_()
        decoder.local.bias.zero_()
        decoder.local.weight[0, 0].fill_(1.0)
        decoder.residual_output.weight.zero_()
        decoder.residual_output.bias.zero_()
        decoder.residual_output.weight[0, 0, 0, 0] = 1.0

    baseline = torch.zeros((1, 64, 64, 64))
    impulse = baseline.clone()
    impulse[0, 0, 32, 32] = 1.0
    delta = decoder(impulse) - decoder(baseline)
    expected_support = torch.zeros((64, 64), dtype=torch.bool)
    expected_support[31:34, 31:34] = True
    assert torch.equal(delta[0, 0] != 0.0, expected_support)
    assert torch.count_nonzero(delta[0, 1:]) == 0


def test_partition_and_optimizer_include_residual_only_in_lift_semantic(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV1,
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    ],
) -> None:
    _, model = matched_models
    partition = partition_parameters_v1(model)
    added_names = {
        "semantic_head.local.weight",
        "semantic_head.local.bias",
        "semantic_head.residual_output.weight",
        "semantic_head.residual_output.bias",
    }
    assert added_names <= set(partition.names["lift_semantic"])
    assert added_names.isdisjoint(partition.names["predictor"])
    assert added_names.isdisjoint(partition.names["target"])
    assert "semantic_head.base.weight" in partition.names["lift_semantic"]
    assert "semantic_head.base.bias" in partition.names["lift_semantic"]
    assert not any("semantic_head" in name for name in partition.names["target"])

    named = dict(model.named_parameters())
    added_ids = {id(named[name]) for name in added_names}
    assert added_ids <= set(map(id, partition.lift_semantic))
    assert added_ids.isdisjoint(map(id, partition.predictor))
    assert added_ids.isdisjoint(map(id, partition.target))
    optimizer = build_frozen_optimizer_v1(partition)
    validate_optimizer_v1(optimizer, partition)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert added_ids <= {id(value) for value in groups["lift_semantic"]["params"]}
    assert added_ids.isdisjoint(id(value) for value in groups["predictor"]["params"])


def test_zero_gate_first_backward_then_context_gradient_unlocks() -> None:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(319)
        base = nn.Conv2d(64, 3, kernel_size=1, bias=True)
        decoder = ResidualLocalSemanticDecoderV4(
            base, initialization_seed=20260713
        )
        inputs = torch.randn((2, 64, 5, 5))
        labels = torch.arange(50).reshape(2, 5, 5).remainder(3)
    finally:
        torch.random.set_rng_state(caller_rng)

    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.05)
    first = F.cross_entropy(decoder(inputs), labels)
    first.backward()
    assert decoder.residual_output.weight.grad is not None
    assert torch.count_nonzero(decoder.residual_output.weight.grad) > 0
    assert decoder.residual_output.bias.grad is not None
    assert torch.count_nonzero(decoder.residual_output.bias.grad) > 0
    assert decoder.local.weight.grad is not None
    assert decoder.local.bias.grad is not None
    assert torch.count_nonzero(decoder.local.weight.grad) == 0
    assert torch.count_nonzero(decoder.local.bias.grad) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = F.cross_entropy(decoder(inputs), labels)
    second.backward()
    assert decoder.local.weight.grad is not None
    assert decoder.local.bias.grad is not None
    assert torch.count_nonzero(decoder.local.weight.grad) > 0
    assert torch.count_nonzero(decoder.local.bias.grad) > 0
