from __future__ import annotations

import math

import pytest
import torch

from lewm.models.egomotion_bev_jepa import warp_bev_current_to_next
from lewm.models.ego_motion_aligned_ray_consistency_v16 import (
    EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16,
    FORWARD_CENTER_RANGE_M_V16,
    LEFT_CENTER_RANGE_M_V16,
    EgoMotionAlignedRayConsistencyReceiptV16,
    ego_motion_aligned_ray_consistency_v16,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    ObservableCameraRayEvidenceV4RawOutput,
)


def _evidence(
    *,
    hit_bin: int = 29,
    offset_m: float = 0.0,
    valid: bool = True,
    batch: int = 1,
    requires_grad: bool = False,
) -> tuple[ObservableCameraRayEvidenceV4RawOutput, torch.Tensor, torch.Tensor]:
    hazards = torch.full((batch, DEPTH_BIN_COUNT, 1, 1), -10.0)
    hazards[:, hit_bin] = 2.0
    offsets = torch.zeros_like(hazards)
    offsets[:, hit_bin] = offset_m
    hazards.requires_grad_(requires_grad)
    offsets.requires_grad_(requires_grad)
    ground_shape = (batch, 128, 128, 5)
    evidence = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazards,
        pixel_within_bin_offset_m=offsets,
        ground_clear_to_target_logits=torch.full(ground_shape, 10.0),
        ground_query_in_frustum=torch.full(ground_shape, valid, dtype=torch.bool),
        ground_query_uv_px=torch.zeros((*ground_shape, 2)),
        ground_target_distance_m=torch.ones(ground_shape),
    )
    return evidence, hazards, offsets


def _calibration(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    origin = torch.zeros((batch, 3), dtype=torch.float32)
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )[None].expand(batch, -1, -1).clone()
    return origin, basis


def _loss(
    current: ObservableCameraRayEvidenceV4RawOutput,
    next_: ObservableCameraRayEvidenceV4RawOutput,
    delta: torch.Tensor,
) -> EgoMotionAlignedRayConsistencyReceiptV16:
    current_origin, current_basis = _calibration(current.pixel_first_hit_hazard_logits.shape[0])
    next_origin, next_basis = _calibration(next_.pixel_first_hit_hazard_logits.shape[0])
    return ego_motion_aligned_ray_consistency_v16(
        current,
        next_,
        current_camera_origin_body_m=current_origin,
        current_camera_basis_body_fru=current_basis,
        next_camera_origin_body_m=next_origin,
        next_camera_basis_body_fru=next_basis,
        relative_se2_current_frame=delta,
    )


def test_contract_weight_and_identical_zero_motion_are_exact() -> None:
    current, _, _ = _evidence(offset_m=0.017)
    next_, _, _ = _evidence(offset_m=0.017)
    receipt = _loss(current, next_, torch.zeros((1, 3), dtype=torch.float32))

    assert EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16 == 0.1
    assert receipt.loss.item() == pytest.approx(0.0, abs=2.0e-7)
    assert receipt.shared_valid_cell_count == 64 * 64
    assert receipt.positive_weight_cell_count == 64 * 64
    assert receipt.weight_sum > 0.0


def test_translated_feature_uses_next_pose_expressed_in_current_convention() -> None:
    current = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    current[0, 0, 20, 31] = 1.0
    warped, overlap = warp_bev_current_to_next(
        current,
        torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
        forward_range_m=FORWARD_CENTER_RANGE_M_V16,
        left_range_m=LEFT_CENTER_RANGE_M_V16,
    )

    assert overlap[0, 0, 19, 31]
    assert warped[0, 0, 19, 31].item() == pytest.approx(1.0, abs=2.0e-5)
    assert warped[0, 0, 20, 31].item() == pytest.approx(0.0, abs=2.0e-5)


def test_metric_depth_translation_aligns_better_than_zero_motion() -> None:
    current, _, _ = _evidence(hit_bin=29)
    next_, _, _ = _evidence(hit_bin=28)
    aligned = _loss(
        current,
        next_,
        torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
    ).loss
    unaligned = _loss(
        current,
        next_,
        torch.zeros((1, 3), dtype=torch.float32),
    ).loss

    assert torch.isfinite(aligned)
    assert aligned.item() < unaligned.item()


def test_both_frames_receive_hazard_and_offset_gradients() -> None:
    current, current_hazards, current_offsets = _evidence(
        offset_m=0.025, requires_grad=True
    )
    next_, next_hazards, next_offsets = _evidence(
        offset_m=-0.025, requires_grad=True
    )
    receipt = _loss(current, next_, torch.zeros((1, 3), dtype=torch.float32))
    receipt.loss.backward()

    for gradient in (
        current_hazards.grad,
        current_offsets.grad,
        next_hazards.grad,
        next_offsets.grad,
    ):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert gradient.abs().sum().item() > 0.0


def test_fully_masked_batch_is_exact_differentiable_zero() -> None:
    current, current_hazards, current_offsets = _evidence(
        valid=False, requires_grad=True
    )
    next_, next_hazards, next_offsets = _evidence(requires_grad=True)
    receipt = _loss(current, next_, torch.zeros((1, 3), dtype=torch.float32))

    assert receipt.loss.item() == 0.0
    assert receipt.shared_valid_cell_count == 0
    assert receipt.positive_weight_cell_count == 0
    assert receipt.weight_sum == 0.0
    receipt.loss.backward()
    for gradient in (
        current_hazards.grad,
        current_offsets.grad,
        next_hazards.grad,
        next_offsets.grad,
    ):
        assert gradient is not None
        assert torch.count_nonzero(gradient).item() == 0


@pytest.mark.parametrize(
    "delta",
    (
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([[math.nan, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, math.inf, 0.0]], dtype=torch.float32),
    ),
)
def test_malformed_or_nonfinite_se2_fails_closed(delta: torch.Tensor) -> None:
    current, _, _ = _evidence()
    next_, _, _ = _evidence()
    with pytest.raises((ValueError, FloatingPointError)):
        _loss(current, next_, delta)


def test_changed_pair_batch_shape_fails_closed() -> None:
    current, _, _ = _evidence(batch=1)
    next_, _, _ = _evidence(batch=2)
    with pytest.raises(ValueError, match="batches must match"):
        ego_motion_aligned_ray_consistency_v16(
            current,
            next_,
            current_camera_origin_body_m=_calibration(1)[0],
            current_camera_basis_body_fru=_calibration(1)[1],
            next_camera_origin_body_m=_calibration(2)[0],
            next_camera_basis_body_fru=_calibration(2)[1],
            relative_se2_current_frame=torch.zeros((1, 3), dtype=torch.float32),
        )
