from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion import (
    FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6,
    FINE_RGB_BRANCH_INITIALIZATION_SEED_V6,
    FineRgbBevResidualV6,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredSweptProgressSurvivalJointJepaV6,
    _fuse_fine_rgb_v6,
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


def test_architecture_partition_target_and_rng(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.random.default_generator.manual_seed(8103)
    caller_rng = torch.random.get_rng_state().clone()
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV6(
        n320_encoder_state, _sweep_masks()
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    online = model.bev_lift.fine_rgb_branch
    target = model.target_bev_lift.fine_rgb_branch
    assert isinstance(online, FineRgbBevResidualV6)
    assert isinstance(online.conv1, nn.Conv2d)
    assert (online.conv1.in_channels, online.conv1.out_channels) == (3, 32)
    assert online.conv1.kernel_size == (3, 3) and online.conv1.padding == (1, 1)
    assert (online.conv2.in_channels, online.conv2.out_channels) == (32, 32)
    assert online.conv2.kernel_size == (3, 3) and online.conv2.padding == (1, 1)
    assert (online.output.in_channels, online.output.out_channels) == (32, 64)
    assert online.output.kernel_size == (1, 1)
    assert online.activation1.approximate == online.activation2.approximate == "none"
    assert sum(parameter.numel() for parameter in online.parameters()) == 12_256
    assert FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6 == 12_256
    assert FINE_RGB_BRANCH_INITIALIZATION_SEED_V6 == 20260714
    assert torch.count_nonzero(online.output.weight) == 0
    assert torch.count_nonzero(online.output.bias) == 0
    assert all(
        torch.equal(value, target.state_dict()[name])
        for name, value in online.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in online.parameters())
    assert not any(parameter.requires_grad for parameter in target.parameters())

    partition = partition_parameters_v1(model)
    online_names = {
        f"bev_lift.fine_rgb_branch.{layer}.{part}"
        for layer in ("conv1", "conv2", "output")
        for part in ("weight", "bias")
    }
    target_names = {
        name.replace("bev_lift.", "target_bev_lift.", 1) for name in online_names
    }
    assert online_names <= set(partition.names["lift_semantic"])
    assert target_names <= set(partition.names["target"])
    assert online_names.isdisjoint(partition.names["predictor"])
    optimizer = build_frozen_optimizer_v1(partition)
    validate_optimizer_v1(optimizer, partition)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert {id(dict(model.named_parameters())[name]) for name in online_names} <= (
        optimizer_ids
    )


def test_initial_v4_parity_and_exact_sampling_receipts(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    v4 = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
        n320_encoder_state, _sweep_masks()
    ).eval()
    v6 = GeometryAnchoredSweptProgressSurvivalJointJepaV6(
        n320_encoder_state, _sweep_masks()
    ).eval()
    rgb = torch.rand((1, 3, 112, 112), generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        old = v4.encode_online_with_sampling(rgb)
        new = v6.encode_online_with_sampling(rgb)
        old_logits = v4.semantic_logits_from_latent(old.latent)
        new_logits = v6.semantic_logits_from_latent(new.latent)
        actions = torch.eye(9)[3:4]
        old_prediction = v4.predict(old.latent, actions)
        new_prediction = v6.predict(new.latent, actions)
    for field in (
        "latent",
        "anchor_in_frustum",
        "sample_valid_mask",
        "cell_valid_mask",
        "sample_grid_xy",
        "offsets_token_cells",
        "sample_weights",
    ):
        assert torch.equal(getattr(old, field), getattr(new, field))
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_prediction, new_prediction)


def test_sampling_uses_inherited_grid_weights_and_masks_invalid_cells(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV6(
        n320_encoder_state, _sweep_masks()
    ).eval()
    with torch.no_grad():
        model.bev_lift.fine_rgb_branch.output.weight.fill_(0.02)
        model.bev_lift.fine_rgb_branch.output.bias.fill_(0.1)
    rgb = torch.rand((1, 3, 112, 112), generator=torch.Generator().manual_seed(29))
    with torch.no_grad():
        tokens = model.encoder.forward_tokens(rgb)[:, 1:]
        inherited = model.bev_lift.forward_with_sampling(tokens)
        fused = model.encode_online_with_sampling(rgb)
        fine = model.bev_lift.fine_rgb_branch(rgb)
        grid = inherited.sample_grid_xy.reshape(1, 64, 64 * 4, 2)
        sampled = torch.nn.functional.grid_sample(
            fine,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(1, 64, 64, 64, 4)
        residual = (sampled * inherited.sample_weights.unsqueeze(1)).sum(-1)
        residual = torch.where(
            inherited.cell_valid_mask[:, None], residual, torch.zeros_like(residual)
        )
    assert torch.equal(fused.latent, inherited.latent + residual)
    invalid = ~inherited.cell_valid_mask
    assert torch.equal(fused.latent[:, :, invalid[0]], inherited.latent[:, :, invalid[0]])
    for field in (
        "sample_grid_xy",
        "sample_weights",
        "sample_valid_mask",
        "cell_valid_mask",
    ):
        assert torch.equal(getattr(fused, field), getattr(inherited, field))


def test_fine_route_does_not_detach_inherited_grids_or_weights() -> None:
    branch = FineRgbBevResidualV6()
    with torch.no_grad():
        branch.output.weight.fill_(0.02)
        branch.output.bias.fill_(0.1)
    rgb = torch.linspace(0.0, 1.0, 3 * 112 * 112).reshape(1, 3, 112, 112)
    grid = torch.tensor(
        (
            (-0.8, -0.7),
            (-0.3, -0.2),
            (0.2, 0.3),
            (0.7, 0.8),
        )
        * 4,
        dtype=torch.float32,
    ).reshape(1, 2, 2, 4, 2).requires_grad_()
    weights = torch.full((1, 2, 2, 4), 0.25, requires_grad=True)
    inherited = GeometryAnchoredBevSamplingV1(
        latent=torch.zeros((1, 64, 2, 2)),
        anchor_in_frustum=torch.ones((1, 2, 2), dtype=torch.bool),
        sample_valid_mask=torch.ones((1, 2, 2, 4), dtype=torch.bool),
        cell_valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        sample_grid_xy=grid,
        offsets_token_cells=torch.zeros((1, 2, 2, 4, 2)),
        sample_weights=weights,
    )
    _fuse_fine_rgb_v6(rgb, inherited, branch).latent.square().mean().backward()
    for value in (grid.grad, weights.grad):
        assert value is not None
        assert bool(torch.isfinite(value).all())
        assert torch.count_nonzero(value) > 0


def test_zero_gate_gradient_unlock_and_inherited_ema(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV6(
        n320_encoder_state, _sweep_masks()
    ).train()
    branch = model.bev_lift.fine_rgb_branch
    rgb = torch.rand((1, 3, 112, 112), generator=torch.Generator().manual_seed(41))
    first = model.encode_online(rgb).square().mean()
    first.backward()
    assert branch.output.weight.grad is not None
    assert torch.count_nonzero(branch.output.weight.grad) > 0
    assert branch.conv1.weight.grad is not None
    assert branch.conv2.weight.grad is not None
    assert torch.count_nonzero(branch.conv1.weight.grad) == 0
    assert torch.count_nonzero(branch.conv2.weight.grad) == 0

    with torch.no_grad():
        branch.output.weight.add_(-0.01 * branch.output.weight.grad)
    model.zero_grad(set_to_none=True)
    model.encode_online(rgb).square().mean().backward()
    assert torch.count_nonzero(branch.conv1.weight.grad) > 0
    assert torch.count_nonzero(branch.conv2.weight.grad) > 0
    assert not any(
        parameter.grad is not None
        for parameter in model.target_bev_lift.fine_rgb_branch.parameters()
    )

    online_before = branch.output.weight.detach().clone()
    target_before = model.target_bev_lift.fine_rgb_branch.output.weight.detach().clone()
    model.update_target_ema_after_optimizer_step()
    expected = target_before.mul(model.config.target_ema_momentum).add(
        online_before, alpha=1.0 - model.config.target_ema_momentum
    )
    torch.testing.assert_close(
        model.target_bev_lift.fine_rgb_branch.output.weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    model.hard_sync_target_from_online()
    assert all(
        torch.equal(value, model.target_bev_lift.fine_rgb_branch.state_dict()[name])
        for name, value in branch.state_dict().items()
    )
