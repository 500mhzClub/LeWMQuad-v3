"""Sole Camera V4 delta: replace target-bin offset loss with tail depth."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_NEAR_EDGE_M,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from .observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12 import (
    compose_gate_aligned_objective_v12,
)
from .observable_camera_ray_evidence_v4_training import (
    HierarchicalRasterCrossEntropyV4,
    ObservableCameraRayEvidenceV4Targets,
    derive_observable_camera_ray_evidence_v4_targets,
)
from .shared_observable_camera_ray_jepa_v5 import (
    ObservableCameraRayV4FrameSupervisionV5,
    SharedTrainingPairV5,
)
from . import shared_observable_camera_ray_jepa_v5_full_training_v4_loss as _base


TAIL_FRACTION_V4 = 0.05
DEPTH_P95_CEILING_M_V4 = 0.25
_BASE_OBSERVABLE_CAMERA_RAY_V4_LOSS = _base.observable_camera_ray_v4_loss_v4


@dataclass(frozen=True)
class TailDepthFrameLossV4:
    hierarchical_first_hit_nll: torch.Tensor
    tail_depth_p95_cvar: torch.Tensor
    ground_clear_distance_state_balanced_bce: torch.Tensor
    derived_raster_hierarchical_bce: HierarchicalRasterCrossEntropyV4
    derived_raster_cell_nll: torch.Tensor
    four_term_base_total: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class TailDepthCameraLossV4:
    current: TailDepthFrameLossV4
    next: TailDepthFrameLossV4
    total: torch.Tensor


def tail_depth_p95_cvar_v4(
    hazard_logits: torch.Tensor,
    predicted_offset_m: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> torch.Tensor:
    """Mean worst five-percent conditional finite-hit metric-depth errors."""

    if not isinstance(hazard_logits, torch.Tensor) or not isinstance(
        predicted_offset_m, torch.Tensor
    ):
        raise TypeError("tail-depth predictions must be tensors")
    if not isinstance(targets, ObservableCameraRayEvidenceV4Targets):
        raise TypeError("tail-depth targets changed type")
    if (
        hazard_logits.ndim != 4
        or tuple(hazard_logits.shape) != tuple(predicted_offset_m.shape)
        or hazard_logits.shape[1] != DEPTH_BIN_COUNT
        or not hazard_logits.is_floating_point()
        or not predicted_offset_m.is_floating_point()
        or hazard_logits.dtype != predicted_offset_m.dtype
    ):
        raise ValueError("tail-depth predictions must share shape (B,64,H,W)")
    expected = (hazard_logits.shape[0], hazard_logits.shape[2], hazard_logits.shape[3])
    if (
        tuple(targets.pixel_in_range_hit_mask.shape) != expected
        or tuple(targets.pixel_hit_bin_index.shape) != expected
        or tuple(targets.pixel_within_bin_offset_m.shape) != expected
    ):
        raise ValueError("tail-depth targets do not match predictions")
    devices = {hazard_logits.device, predicted_offset_m.device,
               targets.pixel_in_range_hit_mask.device, targets.pixel_hit_bin_index.device,
               targets.pixel_within_bin_offset_m.device}
    if len(devices) != 1:
        raise ValueError("tail-depth predictions and targets must share a device")
    if (not bool(torch.isfinite(hazard_logits).all().item())
            or not bool(torch.isfinite(predicted_offset_m).all().item())
            or not bool(torch.isfinite(targets.pixel_within_bin_offset_m).all().item())):
        raise FloatingPointError("tail-depth input became nonfinite")
    mask = targets.pixel_in_range_hit_mask
    count = int(mask.sum().item())
    if count == 0:
        return (hazard_logits.sum() + predicted_offset_m.sum()) * 0.0

    log_hit = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits).hit
    conditional_hit = torch.softmax(log_hit, dim=1)
    centers = DEPTH_NEAR_EDGE_M + (
        torch.arange(DEPTH_BIN_COUNT, device=hazard_logits.device, dtype=predicted_offset_m.dtype)
        + 0.5
    ) * DEPTH_BIN_SIZE_M
    predicted_depth = centers[None, :, None, None] + predicted_offset_m
    target_center = centers[targets.pixel_hit_bin_index]
    target_depth = target_center + targets.pixel_within_bin_offset_m.to(
        dtype=predicted_offset_m.dtype
    )
    normalized_error = (conditional_hit * (predicted_depth - target_depth[:, None]).abs()).sum(
        dim=1
    ) / DEPTH_P95_CEILING_M_V4
    worst_count = math.ceil(TAIL_FRACTION_V4 * count)
    result = torch.topk(normalized_error[mask], k=worst_count, sorted=False).values.mean()
    if not bool(torch.isfinite(result).item()):
        raise FloatingPointError("tail-depth objective became nonfinite")
    return result


def _targets(supervision: ObservableCameraRayV4FrameSupervisionV5) -> ObservableCameraRayEvidenceV4Targets:
    return derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=supervision.pixel_hit_mask,
        pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
        ground_support_in_frustum=supervision.ground_support_in_frustum,
        ground_support_clear_to_target=supervision.ground_support_clear_to_target,
    )


def _replace_frame(base: object, tail: torch.Tensor) -> TailDepthFrameLossV4:
    objective = compose_gate_aligned_objective_v12(
        {
            "hierarchical_first_hit_nll": base.hierarchical_first_hit_nll,
            "target_bin_offset_smooth_l1": tail,
            "ground_clear_distance_state_balanced_bce": base.ground_clear_distance_state_balanced_bce,
            "derived_raster_hierarchical_bce": base.derived_raster_hierarchical_bce.total,
        },
        base.derived_raster_cell_nll,
    )
    return TailDepthFrameLossV4(
        hierarchical_first_hit_nll=base.hierarchical_first_hit_nll,
        tail_depth_p95_cvar=tail,
        ground_clear_distance_state_balanced_bce=base.ground_clear_distance_state_balanced_bce,
        derived_raster_hierarchical_bce=base.derived_raster_hierarchical_bce,
        derived_raster_cell_nll=base.derived_raster_cell_nll,
        four_term_base_total=objective.v11_base_total,
        total=objective.total,
    )


def observable_camera_ray_v4_tail_depth_loss_v4(
    model: object,
    pair: SharedTrainingPairV5,
    current_supervision: ObservableCameraRayV4FrameSupervisionV5,
    next_supervision: ObservableCameraRayV4FrameSupervisionV5,
    *,
    require_b4: bool = True,
) -> TailDepthCameraLossV4:
    """Reuse the exact baseline once, then substitute only its offset slot."""

    baseline = _BASE_OBSERVABLE_CAMERA_RAY_V4_LOSS(
        model, pair, current_supervision, next_supervision, require_b4=require_b4
    )
    current = _replace_frame(
        baseline.current,
        tail_depth_p95_cvar_v4(
            pair.current.evidence.pixel_first_hit_hazard_logits,
            pair.current.evidence.pixel_within_bin_offset_m,
            _targets(current_supervision),
        ),
    )
    next_ = _replace_frame(
        baseline.next,
        tail_depth_p95_cvar_v4(
            pair.next.evidence.pixel_first_hit_hazard_logits,
            pair.next.evidence.pixel_within_bin_offset_m,
            _targets(next_supervision),
        ),
    )
    total = 0.5 * current.total + 0.5 * next_.total
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("tail-depth current/next objective became nonfinite")
    return TailDepthCameraLossV4(current=current, next=next_, total=total)
