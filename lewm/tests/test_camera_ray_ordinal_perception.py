from __future__ import annotations

import math

import pytest
import torch

from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    build_dynamic_cell_square_support_mask,
)
from lewm.models.camera_ray_ordinal_perception import (
    CARTESIAN_SHAPE,
    CameraRayOrdinalPerception,
    DYNAMIC_CARTESIAN_OCCUPANCY_PARAMETER_COUNT,
    REGISTERED_DECODER_PARAMETER_COUNT,
    REGISTERED_PARAMETER_COUNT,
    apply_full_ray_context_per_ray,
    ordered_first_surface_log_probabilities,
    yaw_aligned_camera_basis,
)
from lewm.models.categorical_radial_perception_full_ray import (
    FullRayRadialContext,
)


def _identity_attitude(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(batch, -1).clone()
    yaw = torch.zeros(batch)
    return quaternion, yaw


def test_ordered_first_surface_probabilities_normalize_and_respect_order() -> None:
    torch.manual_seed(7)
    hazard = torch.randn(2, 9, 3, 4, dtype=torch.float64)
    surface_type = torch.randn_like(hazard)
    result = ordered_first_surface_log_probabilities(hazard, surface_type)

    total = result.no_hit.exp()
    total = total + result.obstacle.exp().sum(dim=1)
    total = total + result.floor.exp().sum(dim=1)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-12, rtol=1e-12)

    early_hazard = torch.full((1, 5, 1, 1), -20.0)
    early_hazard[:, 0] = 20.0
    obstacle_type = torch.full_like(early_hazard, 20.0)
    early = ordered_first_surface_log_probabilities(
        early_hazard,
        obstacle_type,
    )
    assert float(early.obstacle[:, 0].exp().item()) > 0.999999
    assert float(early.obstacle[:, 1:].exp().sum().item()) < 1e-7
    assert float(early.floor.exp().sum().item()) < 1e-7


def test_chunked_full_ray_context_matches_unchunked_forward_and_gradient() -> None:
    torch.manual_seed(11)
    context = FullRayRadialContext(8)
    full_input = torch.randn(2, 8, 16, 23, requires_grad=True)
    chunked_input = full_input.detach().clone().requires_grad_(True)

    full = apply_full_ray_context_per_ray(
        context,
        full_input,
        ray_chunk_size=None,
    )
    chunked = apply_full_ray_context_per_ray(
        context,
        chunked_input,
        ray_chunk_size=7,
    )
    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)

    full_gradient = torch.autograd.grad(full.square().mean(), full_input)[0]
    chunked_gradient = torch.autograd.grad(
        chunked.square().mean(),
        chunked_input,
    )[0]
    torch.testing.assert_close(
        chunked_gradient,
        full_gradient,
        atol=2e-6,
        rtol=2e-6,
    )


def test_yaw_aligned_camera_basis_fails_closed() -> None:
    quaternion, yaw = _identity_attitude()
    basis = yaw_aligned_camera_basis(quaternion, yaw)
    torch.testing.assert_close(
        basis.origin,
        torch.tensor([[0.326, 0.0, 0.043]]),
    )
    torch.testing.assert_close(basis.forward, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(basis.left, torch.tensor([[0.0, 1.0, 0.0]]))
    torch.testing.assert_close(basis.up, torch.tensor([[0.0, 0.0, 1.0]]))

    with pytest.raises(ValueError, match="shape"):
        yaw_aligned_camera_basis(torch.zeros(1, 3), yaw)
    with pytest.raises(ValueError, match="finite"):
        invalid = quaternion.clone()
        invalid[0, 0] = float("nan")
        yaw_aligned_camera_basis(invalid, yaw)
    with pytest.raises(ValueError, match="norm"):
        invalid = quaternion.clone()
        invalid[0, 3] = 2.0
        yaw_aligned_camera_basis(invalid, yaw)
    with pytest.raises(ValueError, match="disagrees"):
        yaw_aligned_camera_basis(quaternion, torch.tensor([0.1]))


def test_registered_level_support_matches_stdlib_dynamic_geometry() -> None:
    model = CameraRayOrdinalPerception(encoder_depth=1)
    quaternion, yaw = _identity_attitude()
    observed = model.registered_support_visibility(quaternion, yaw)[0]
    expected = torch.tensor(
        build_dynamic_cell_square_support_mask(
            (0.0, 0.0, 0.0, 1.0),
            0.0,
        ),
        dtype=torch.bool,
    )
    assert tuple(observed.shape) == CARTESIAN_SHAPE
    assert int(observed.sum().item()) == 2062
    assert torch.equal(observed, expected)


def test_roll_changes_full_attitude_projection_and_bad_yaw_gather_rejects() -> None:
    model = CameraRayOrdinalPerception(
        encoder_depth=1,
        ray_height=4,
        ray_width=5,
        depth_bin_count=8,
        depth_bin_size_m=0.8,
    )
    identity, yaw = _identity_attitude()
    roll = 0.2
    tilted = torch.tensor(
        [[math.sin(roll * 0.5), 0.0, 0.0, math.cos(roll * 0.5)]]
    )
    identity_grid, _identity_visible = model._project_registered_support(
        identity,
        yaw,
    )
    tilted_grid, _tilted_visible = model._project_registered_support(
        tilted,
        yaw,
    )
    assert not torch.equal(identity_grid, tilted_grid)

    hazard = torch.zeros(1, 8, 4, 5)
    surface = torch.zeros_like(hazard)
    ray_probabilities = ordered_first_surface_log_probabilities(hazard, surface)
    with pytest.raises(ValueError, match="disagrees"):
        model.cartesian_log_probabilities(
            ray_probabilities,
            identity,
            torch.tensor([0.1]),
        )


def test_reduced_model_forward_backward_is_finite() -> None:
    torch.manual_seed(19)
    model = CameraRayOrdinalPerception(
        encoder_depth=1,
        ray_height=4,
        ray_width=5,
        depth_bin_count=8,
        depth_bin_size_m=0.8,
        ray_chunk_size=7,
    )
    image = torch.randn(1, 3, 112, 112)
    quaternion, yaw = _identity_attitude()
    logits = model(image, quaternion, yaw)

    assert tuple(logits.shape) == (1, 3, *CARTESIAN_SHAPE)
    assert bool(torch.isfinite(logits).all().item())
    torch.testing.assert_close(
        torch.logsumexp(logits, dim=1),
        torch.zeros(1, *CARTESIAN_SHAPE),
        atol=2e-6,
        rtol=2e-6,
    )
    visible = model.registered_support_visibility(quaternion, yaw)
    loss = -logits[:, 2][visible].mean()
    loss.backward()

    gradient = model.encoder.patch_embed.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all().item())
    assert float(gradient.abs().sum().item()) > 0.0


def test_registered_parameter_count_is_capacity_matched() -> None:
    model = CameraRayOrdinalPerception()
    total = sum(parameter.numel() for parameter in model.parameters())
    decoder = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    )
    assert total == REGISTERED_PARAMETER_COUNT
    assert decoder == REGISTERED_DECODER_PARAMETER_COUNT
    relative_difference = (
        abs(total - DYNAMIC_CARTESIAN_OCCUPANCY_PARAMETER_COUNT)
        / DYNAMIC_CARTESIAN_OCCUPANCY_PARAMETER_COUNT
    )
    assert relative_difference < 0.001
