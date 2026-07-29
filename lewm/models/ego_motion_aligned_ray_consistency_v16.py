"""Training-only ego-motion alignment for unified camera-ray evidence.

This module adds no inference input or model parameter.  It rasterizes the
existing current/next ray evidence, aligns the current raster with measured
relative SE(2), and compares occupied probability conditional on the cell
being known.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .egomotion_bev_jepa import warp_bev_current_to_next
from .observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    GROUND_SUPPORT_COUNT,
    ObservableCameraRayEvidenceV4RawOutput,
)
from .observable_camera_ray_evidence_v4_training import (
    soft_rasterize_observable_camera_ray_evidence_v4,
)


RASTER_SHAPE_V16 = (64, 64)
FORWARD_CENTER_RANGE_M_V16 = (-0.95, 5.35)
LEFT_CENTER_RANGE_M_V16 = (-3.15, 3.15)
EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16 = 0.1
VALID_WARP_THRESHOLD_V16 = 0.999


@dataclass(frozen=True)
class EgoMotionAlignedRayConsistencyReceiptV16:
    """Loss and fixed diagnostics for one paired frame batch."""

    loss: torch.Tensor
    shared_valid_cell_count: int
    positive_weight_cell_count: int
    weight_sum: float


def _validate_evidence_v16(
    evidence: ObservableCameraRayEvidenceV4RawOutput,
    *,
    name: str,
) -> tuple[int, torch.dtype, torch.device]:
    if not isinstance(evidence, ObservableCameraRayEvidenceV4RawOutput):
        raise TypeError(f"{name} must be ObservableCameraRayEvidenceV4RawOutput")
    hazard = evidence.pixel_first_hit_hazard_logits
    offset = evidence.pixel_within_bin_offset_m
    ground = evidence.ground_clear_to_target_logits
    if (
        hazard.ndim != 4
        or hazard.shape[1] != DEPTH_BIN_COUNT
        or tuple(offset.shape) != tuple(hazard.shape)
    ):
        raise ValueError(f"{name} ray evidence must have shape (B,64,Hray,Wray)")
    if tuple(ground.shape[1:]) != (
        2 * RASTER_SHAPE_V16[0],
        2 * RASTER_SHAPE_V16[1],
        GROUND_SUPPORT_COUNT,
    ):
        raise ValueError(f"{name} ground evidence must have shape (B,128,128,5)")
    floating = (
        hazard,
        offset,
        ground,
        evidence.ground_query_uv_px,
        evidence.ground_target_distance_m,
    )
    if any(value.dtype != torch.float32 for value in floating):
        raise ValueError(f"{name} floating evidence must be float32")
    if any(value.device != hazard.device for value in floating) or (
        evidence.ground_query_in_frustum.device != hazard.device
    ):
        raise ValueError(f"{name} evidence fields must share one device")
    return int(hazard.shape[0]), hazard.dtype, hazard.device


def _validate_calibration_v16(
    origin: torch.Tensor,
    basis: torch.Tensor,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if not isinstance(origin, torch.Tensor) or not isinstance(basis, torch.Tensor):
        raise TypeError(f"{name} camera calibration must be tensors")
    if tuple(origin.shape) != (batch, 3) or tuple(basis.shape) != (batch, 3, 3):
        raise ValueError(f"{name} camera calibration has the wrong batch shape")
    if origin.dtype != dtype or basis.dtype != dtype:
        raise ValueError(f"{name} camera calibration must match evidence dtype")
    if origin.device != device or basis.device != device:
        raise ValueError(f"{name} camera calibration must match evidence device")


def _coarse_ground_validity_v16(
    evidence: ObservableCameraRayEvidenceV4RawOutput,
) -> torch.Tensor:
    """Require all five supports in all four source cells."""

    source_valid = evidence.ground_query_in_frustum.all(dim=-1)
    return source_valid.reshape(
        source_valid.shape[0],
        RASTER_SHAPE_V16[0],
        2,
        RASTER_SHAPE_V16[1],
        2,
    ).all(dim=4).all(dim=2)


def _bernoulli_kl_v16(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def ego_motion_aligned_ray_consistency_v16(
    current_evidence: ObservableCameraRayEvidenceV4RawOutput,
    next_evidence: ObservableCameraRayEvidenceV4RawOutput,
    *,
    current_camera_origin_body_m: torch.Tensor,
    current_camera_basis_body_fru: torch.Tensor,
    next_camera_origin_body_m: torch.Tensor,
    next_camera_basis_body_fru: torch.Tensor,
    relative_se2_current_frame: torch.Tensor,
) -> EgoMotionAlignedRayConsistencyReceiptV16:
    """Compute preregistered V16 consistency in the next-frame lattice."""

    current_batch, dtype, device = _validate_evidence_v16(
        current_evidence, name="current_evidence"
    )
    next_batch, next_dtype, next_device = _validate_evidence_v16(
        next_evidence, name="next_evidence"
    )
    if (next_batch, next_dtype, next_device) != (current_batch, dtype, device):
        raise ValueError("current and next evidence batches must match")
    _validate_calibration_v16(
        current_camera_origin_body_m,
        current_camera_basis_body_fru,
        batch=current_batch,
        dtype=dtype,
        device=device,
        name="current",
    )
    _validate_calibration_v16(
        next_camera_origin_body_m,
        next_camera_basis_body_fru,
        batch=current_batch,
        dtype=dtype,
        device=device,
        name="next",
    )
    if not isinstance(relative_se2_current_frame, torch.Tensor):
        raise TypeError("relative_se2_current_frame must be a tensor")
    if tuple(relative_se2_current_frame.shape) != (current_batch, 3):
        raise ValueError("relative_se2_current_frame must have shape (B,3)")
    if (
        relative_se2_current_frame.dtype != torch.float32
        or relative_se2_current_frame.device != device
    ):
        raise ValueError("relative_se2_current_frame must be float32 on evidence device")
    if not bool(torch.isfinite(relative_se2_current_frame).all().item()):
        raise FloatingPointError("relative_se2_current_frame became nonfinite")

    current_raster = soft_rasterize_observable_camera_ray_evidence_v4(
        current_evidence,
        camera_origin_body_m=current_camera_origin_body_m,
        camera_basis_body_fru=current_camera_basis_body_fru,
        output_shape=RASTER_SHAPE_V16,
    )
    next_raster = soft_rasterize_observable_camera_ray_evidence_v4(
        next_evidence,
        camera_origin_body_m=next_camera_origin_body_m,
        camera_basis_body_fru=next_camera_basis_body_fru,
        output_shape=RASTER_SHAPE_V16,
    )
    warped_current, overlap = warp_bev_current_to_next(
        current_raster.class_probabilities,
        relative_se2_current_frame,
        forward_range_m=FORWARD_CENTER_RANGE_M_V16,
        left_range_m=LEFT_CENTER_RANGE_M_V16,
    )
    warped_current_validity, _ = warp_bev_current_to_next(
        _coarse_ground_validity_v16(current_evidence)[:, None].to(dtype=dtype),
        relative_se2_current_frame,
        forward_range_m=FORWARD_CENTER_RANGE_M_V16,
        left_range_m=LEFT_CENTER_RANGE_M_V16,
    )
    next_validity = _coarse_ground_validity_v16(next_evidence)
    shared_valid = (
        overlap[:, 0]
        & (warped_current_validity[:, 0] > VALID_WARP_THRESHOLD_V16)
        & next_validity
    )

    epsilon = torch.finfo(torch.float32).eps
    warped_current = warped_current / warped_current.sum(
        dim=1, keepdim=True
    ).clamp_min(epsilon)
    current_known = warped_current[:, 1] + warped_current[:, 2]
    next_probabilities = next_raster.class_probabilities
    next_known = next_probabilities[:, 1] + next_probabilities[:, 2]
    current_q = (warped_current[:, 2] / current_known.clamp_min(epsilon)).clamp(
        min=epsilon, max=1.0 - epsilon
    )
    next_q = (next_probabilities[:, 2] / next_known.clamp_min(epsilon)).clamp(
        min=epsilon, max=1.0 - epsilon
    )
    weight = torch.minimum(current_known, next_known).detach()
    weight = torch.where(shared_valid, weight, torch.zeros_like(weight))
    weight_sum_tensor = weight.sum().detach()
    positive_weight = shared_valid & (weight > 0.0)

    forward_kl = _bernoulli_kl_v16(current_q.detach(), next_q)
    reverse_kl = _bernoulli_kl_v16(next_q.detach(), current_q)
    if bool((weight_sum_tensor > 0.0).item()):
        loss = 0.5 * (weight * (forward_kl + reverse_kl)).sum() / weight_sum_tensor
    else:
        loss = (current_q.sum() + next_q.sum()) * 0.0
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("V16 consistency loss became nonfinite")
    return EgoMotionAlignedRayConsistencyReceiptV16(
        loss=loss,
        shared_valid_cell_count=int(shared_valid.sum().item()),
        positive_weight_cell_count=int(positive_weight.sum().item()),
        weight_sum=float(weight_sum_tensor.item()),
    )


__all__ = [
    "EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16",
    "FORWARD_CENTER_RANGE_M_V16",
    "LEFT_CENTER_RANGE_M_V16",
    "RASTER_SHAPE_V16",
    "EgoMotionAlignedRayConsistencyReceiptV16",
    "ego_motion_aligned_ray_consistency_v16",
]
