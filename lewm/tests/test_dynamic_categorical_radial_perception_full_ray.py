from __future__ import annotations

import math

import pytest
import torch

from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    VERTICAL_ANCHOR_Z_M,
    _camera_coordinates,
    camera_coordinates_in_frustum,
    compose_yaw_aligned_camera,
)
from lewm.models.categorical_radial_perception_full_ray import (
    CategoricalRadialPerceptionFullRay,
    REGISTERED_PARAMETER_COUNT,
)
from lewm.models.dynamic_categorical_radial_perception_full_ray import (
    DynamicCategoricalRadialPerceptionFullRay,
)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=torch.float32,
    )


def test_dynamic_model_keeps_static_state_and_parameter_contract() -> None:
    static = CategoricalRadialPerceptionFullRay()
    dynamic = DynamicCategoricalRadialPerceptionFullRay()
    assert static.state_dict().keys() == dynamic.state_dict().keys()
    assert sum(parameter.numel() for parameter in dynamic.parameters()) == (
        REGISTERED_PARAMETER_COUNT
    )
    dynamic.load_state_dict(static.state_dict(), strict=True)


def test_level_attitude_matches_registered_static_sampling() -> None:
    torch.manual_seed(7)
    static = CategoricalRadialPerceptionFullRay().eval()
    dynamic = DynamicCategoricalRadialPerceptionFullRay().eval()
    dynamic.load_state_dict(static.state_dict(), strict=True)
    projected = torch.randn(1, static.token_feature_dim, 16, 16)
    expected = static.sample_projective_anchors(projected)
    observed, validity = dynamic.sample_dynamic_projective_anchors(
        projected,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([0.0]),
    )
    assert torch.equal(validity[0], static.projective_anchor_validity)
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)


def test_tilted_validity_matches_stdlib_camera_geometry() -> None:
    model = DynamicCategoricalRadialPerceptionFullRay()
    quaternion = _quat_from_rpy(0.17, -0.23, 0.41)
    grid, validity = model.dynamic_anchor_geometry(
        quaternion[None], torch.tensor([0.41])
    )
    assert torch.isfinite(grid).all()
    camera = compose_yaw_aligned_camera(quaternion.tolist(), 0.41)
    for anchor_index, anchor in enumerate(VERTICAL_ANCHOR_Z_M):
        for radial_index, angular_index in ((0, 128), (15, 100), (31, 140), (63, 128)):
            point = (
                float(model.dynamic_polar_forward_m[radial_index, angular_index]),
                float(model.dynamic_polar_left_m[radial_index, angular_index]),
                anchor,
            )
            expected = camera_coordinates_in_frustum(
                *_camera_coordinates(point, camera)
            )
            assert bool(validity[0, anchor_index, radial_index, angular_index]) is expected


def test_attitude_is_required_validated_and_changes_output() -> None:
    torch.manual_seed(11)
    model = DynamicCategoricalRadialPerceptionFullRay().eval()
    image = torch.randn(1, 3, 112, 112)
    level = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    tilted = _quat_from_rpy(0.12, -0.18, 0.0)[None]
    with torch.no_grad():
        level_logits = model(image, level, torch.tensor([0.0]))
        tilted_logits = model(image, tilted, torch.tensor([0.0]))
    assert level_logits.shape == (1, 3, 64, 64)
    assert torch.isfinite(level_logits).all() and torch.isfinite(tilted_logits).all()
    assert not torch.equal(level_logits, tilted_logits)
    with pytest.raises(ValueError, match="norm"):
        model(image, level * 2.0, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="disagrees"):
        model(image, level, torch.tensor([0.2]))


def test_dynamic_head_backpropagates_to_the_shared_encoder() -> None:
    torch.manual_seed(13)
    model = DynamicCategoricalRadialPerceptionFullRay().train()
    image = torch.randn(1, 3, 112, 112)
    logits = model(
        image,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([0.0]),
    )
    logits.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
