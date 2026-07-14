from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    GROUND_SUPPORT_COUNT,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
    project_canonical_ground_support_v4,
)
from lewm.models.encoders import VisionEncoder
from lewm.models.observable_camera_ray_evidence_v4 import (
    DENSE_FEATURE_DIM,
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    IMAGE_SIZE,
    PATCH_SIZE,
    REGISTERED_PARAMETER_COUNT,
    ObservableCameraRayEvidenceV4Model,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)


def _calibration(
    batch: int = 1,
    *,
    pitch_rad: float = 0.0,
    origin_forward_m: float = 0.326,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = math.cos(pitch_rad)
    sine = math.sin(pitch_rad)
    origin = torch.tensor(
        (origin_forward_m, 0.02, 0.043),
        dtype=dtype,
    )[None].expand(batch, -1).clone()
    # Rows are forward, right, up. Body +Y is left, hence camera right is -Y.
    basis = torch.tensor(
        (
            (cosine, 0.0, sine),
            (0.0, -1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=dtype,
    )[None].expand(batch, -1, -1).clone()
    ground_z = torch.full((batch,), -0.35, dtype=dtype)
    return origin, basis, ground_z


def test_reduced_shape_forward_uses_one_encoder_and_returns_raw_outputs() -> None:
    torch.manual_seed(3)
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(3, 4),
        pixel_ray_shape=(5, 7),
        query_chunk_size=7,
    ).eval()
    image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    origin, basis, ground_z = _calibration(batch=2)

    with mock.patch.object(
        model.encoder,
        "forward_tokens",
        wraps=model.encoder.forward_tokens,
    ) as encoder_call:
        output = model(image, origin, basis, ground_z)

    assert isinstance(output, ObservableCameraRayEvidenceV4RawOutput)
    assert encoder_call.call_count == 1
    assert output.pixel_first_hit_hazard_logits.shape == (2, 64, 5, 7)
    assert output.pixel_within_bin_offset_m.shape == (2, 64, 5, 7)
    assert output.ground_clear_to_target_logits.shape == (
        2,
        3,
        4,
        GROUND_SUPPORT_COUNT,
    )
    assert output.ground_query_in_frustum.shape == (
        2,
        3,
        4,
        GROUND_SUPPORT_COUNT,
    )
    assert output.ground_query_uv_px.shape == (
        2,
        3,
        4,
        GROUND_SUPPORT_COUNT,
        2,
    )
    assert output.ground_target_distance_m.shape == (
        2,
        3,
        4,
        GROUND_SUPPORT_COUNT,
    )
    assert torch.isfinite(output.ground_clear_to_target_logits).all()
    assert float(output.pixel_within_bin_offset_m.detach().abs().max()) <= (
        0.5 * DEPTH_BIN_SIZE_M
    )


def test_default_model_returns_full_v4_contract_shapes() -> None:
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        query_chunk_size=8192,
    ).eval()
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    origin, basis, ground_z = _calibration()

    with torch.no_grad():
        output = model(image, origin, basis, ground_z)

    assert output.pixel_first_hit_hazard_logits.shape == (
        1,
        DEPTH_BIN_COUNT,
        *PIXEL_RAY_SHAPE,
    )
    assert output.pixel_within_bin_offset_m.shape == (
        1,
        DEPTH_BIN_COUNT,
        *PIXEL_RAY_SHAPE,
    )
    assert output.ground_clear_to_target_logits.shape == (
        1,
        *SOURCE_SHAPE,
        GROUND_SUPPORT_COUNT,
    )
    assert output.ground_query_uv_px.shape == (
        1,
        *SOURCE_SHAPE,
        GROUND_SUPPORT_COUNT,
        2,
    )


def test_ordered_obstacle_hazards_normalize_and_respect_depth_order() -> None:
    torch.manual_seed(5)
    hazard = torch.randn(2, 11, 3, 4, dtype=torch.float64)
    probabilities = ordered_obstacle_first_hit_log_probabilities_v4(hazard)

    total = probabilities.hit.exp().sum(dim=1) + probabilities.no_hit.exp()
    torch.testing.assert_close(
        total,
        torch.ones_like(total),
        atol=1e-12,
        rtol=1e-12,
    )

    early_hazard = torch.full((1, 6, 1, 1), -20.0, dtype=torch.float64)
    early_hazard[:, 0] = 20.0
    early = ordered_obstacle_first_hit_log_probabilities_v4(early_hazard)
    assert float(early.hit[:, 0].exp().item()) > 0.999999
    assert float(early.hit[:, 1:].exp().sum().item()) < 1e-7


def test_ground_query_calibration_matches_frozen_numpy_contract_in_order() -> None:
    model = ObservableCameraRayEvidenceV4Model(encoder_depth=0)
    origin, basis, ground_z = _calibration(
        pitch_rad=0.13,
        origin_forward_m=0.311,
        dtype=torch.float64,
    )

    geometry = model.ground_query_geometry(origin, basis, ground_z)
    expected = project_canonical_ground_support_v4(
        camera_origin_body_m=origin[0].numpy(),
        camera_basis_body_fru=basis[0].numpy(),
        ground_plane_z_body_m=float(ground_z[0]),
    )

    np.testing.assert_array_equal(
        geometry.in_frustum[0].numpy(),
        expected.in_frustum,
    )
    np.testing.assert_allclose(
        geometry.uv_px[0].numpy(),
        expected.uv_px,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        geometry.target_distance_m[0].numpy(),
        expected.target_distance_m,
        rtol=0.0,
        atol=2e-12,
    )


def test_chunked_ground_branch_matches_unchunked_output_and_gradient() -> None:
    torch.manual_seed(7)
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(4, 6),
        pixel_ray_shape=(3, 5),
        query_chunk_size=None,
    )
    origin, basis, ground_z = _calibration(batch=2)
    geometry = model.ground_query_geometry(origin, basis, ground_z)
    full_dense = torch.randn(
        2,
        DENSE_FEATURE_DIM,
        IMAGE_SIZE,
        IMAGE_SIZE,
        requires_grad=True,
    )
    chunked_dense = full_dense.detach().clone().requires_grad_(True)

    full = model.ground_branch(
        full_dense,
        geometry,
        query_chunk_size=None,
    )
    chunked = model.ground_branch(
        chunked_dense,
        geometry,
        query_chunk_size=11,
    )
    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)

    full_gradient = torch.autograd.grad(full.square().mean(), full_dense)[0]
    chunked_gradient = torch.autograd.grad(
        chunked.square().mean(),
        chunked_dense,
    )[0]
    torch.testing.assert_close(
        chunked_gradient,
        full_gradient,
        atol=2e-6,
        rtol=2e-6,
    )


def test_reduced_model_has_finite_end_to_end_gradients() -> None:
    torch.manual_seed(11)
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=1,
        source_shape=(3, 4),
        pixel_ray_shape=(4, 5),
        query_chunk_size=7,
    )
    image = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        requires_grad=True,
    )
    origin, basis, ground_z = _calibration()

    output = model(image, origin, basis, ground_z)
    loss = output.pixel_first_hit_hazard_logits.square().mean()
    loss = loss + output.pixel_within_bin_offset_m.square().mean()
    loss = loss + output.ground_clear_to_target_logits.square().mean()
    loss.backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_ground_outputs_are_sensitive_to_wrong_but_valid_calibration() -> None:
    torch.manual_seed(13)
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(5, 7),
        pixel_ray_shape=(4, 6),
        query_chunk_size=13,
    ).eval()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    correct = _calibration(pitch_rad=0.02, origin_forward_m=0.326)
    wrong = _calibration(pitch_rad=0.17, origin_forward_m=0.21)

    with torch.no_grad():
        expected = model(image, *correct)
        miscalibrated = model(image, *wrong)

    torch.testing.assert_close(
        miscalibrated.pixel_first_hit_hazard_logits,
        expected.pixel_first_hit_hazard_logits,
    )
    torch.testing.assert_close(
        miscalibrated.pixel_within_bin_offset_m,
        expected.pixel_within_bin_offset_m,
    )
    assert not torch.allclose(
        miscalibrated.ground_query_uv_px,
        expected.ground_query_uv_px,
    )
    assert not torch.allclose(
        miscalibrated.ground_target_distance_m,
        expected.ground_target_distance_m,
    )
    assert not torch.allclose(
        miscalibrated.ground_clear_to_target_logits,
        expected.ground_clear_to_target_logits,
    )


def test_invalid_camera_basis_fails_closed() -> None:
    model = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(2, 2),
        pixel_ray_shape=(2, 2),
    )
    origin, basis, ground_z = _calibration()
    basis[:, 1] = basis[:, 0]

    with pytest.raises(ValueError, match="orthonormal"):
        model.ground_query_geometry(origin, basis, ground_z)


def test_registered_model_has_one_patch7_encoder_and_frozen_parameter_count() -> None:
    model = ObservableCameraRayEvidenceV4Model()

    encoders = [module for module in model.modules() if isinstance(module, VisionEncoder)]
    assert len(encoders) == 1
    assert encoders[0].image_size == IMAGE_SIZE
    assert encoders[0].patch_size == PATCH_SIZE == 7
    assert len(encoders[0].blocks) == 6
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        REGISTERED_PARAMETER_COUNT
    )
