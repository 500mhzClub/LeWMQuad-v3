"""Full Training V4 Camera objective and exact B4 reduction adapter.

This additive module leaves the reviewed Shared V5 model untouched. It
replaces the obsolete ordered first-hit term only at the full-training loss
boundary, retains the Camera V11 hierarchical objective, and adds Camera
V12/V13 gate-aligned all-cell raster NLL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12 import (
    compose_gate_aligned_objective_v12,
    derived_raster_cell_nll_v12,
)
from .observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
    HierarchicalFirstHitNLLBreakdownV9,
    hierarchical_first_hit_nll_breakdown_v9,
)
from .observable_camera_ray_evidence_v4_training import (
    HierarchicalRasterCrossEntropyV4,
    balanced_ground_clear_bce_v4,
    derive_observable_camera_ray_evidence_v4_targets,
    hierarchical_raster_cross_entropy_v4,
    soft_rasterize_observable_camera_ray_evidence_v4,
)
from .shared_observable_camera_ray_jepa_v5 import (
    EstablishedJepaPackageV5,
    ObservableCameraRayV4FrameSupervisionV5,
    SharedTrainingPairV5,
    _skew_balanced_pixel_offset_loss_v5,
)


EXPECTED_MICROBATCH_SIZE_V4 = 4
CAMERA_MODEL_CONFIG_WEIGHT_V4 = 1.0


@dataclass(frozen=True)
class SharedObservableCameraRayV4FrameLossV4:
    """One separately computed real-B4 Full Training V4 Camera objective."""

    hierarchical_first_hit_nll: torch.Tensor
    hierarchical_first_hit_breakdown: HierarchicalFirstHitNLLBreakdownV9
    target_bin_offset_smooth_l1: torch.Tensor
    ground_clear_distance_state_balanced_bce: torch.Tensor
    derived_raster_hierarchical_bce: HierarchicalRasterCrossEntropyV4
    derived_raster_cell_nll: torch.Tensor
    retained_v11_base_total: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class SharedObservableCameraRayV4LossV4:
    """Exact 0.5/0.5 average of separately computed current and next B4 losses."""

    current: SharedObservableCameraRayV4FrameLossV4
    next: SharedObservableCameraRayV4FrameLossV4
    total: torch.Tensor


@dataclass(frozen=True)
class SharedJointLossV4:
    """Promoted JEPA plus the corrected five-term Camera supervision."""

    total: torch.Tensor
    established_jepa: EstablishedJepaPackageV5
    observable_camera_ray_v4: SharedObservableCameraRayV4LossV4
    observable_camera_ray_v4_weight: float


def _require_real_b4_frame(
    pair: SharedTrainingPairV5,
    supervision: ObservableCameraRayV4FrameSupervisionV5,
    *,
    frame_name: str,
    require_b4: bool,
) -> None:
    if not isinstance(pair, SharedTrainingPairV5):
        raise TypeError("V4 pair must be SharedTrainingPairV5")
    if not isinstance(supervision, ObservableCameraRayV4FrameSupervisionV5):
        raise TypeError("V4 supervision type changed")
    frame = pair.current if frame_name == "current" else pair.next
    evidence = frame.evidence
    observed = int(evidence.pixel_first_hit_hazard_logits.shape[0])
    target_batch = int(supervision.target_raster_labels.shape[0])
    if target_batch != observed:
        raise ValueError("V4 Camera supervision batch does not match its frame")
    if require_b4 and observed != EXPECTED_MICROBATCH_SIZE_V4:
        raise ValueError(
            "V4 nonlinear Camera loss requires one real B=4 frame microbatch; "
            "synthetic-B16 pooling is forbidden"
        )


def _frame_loss_v4(
    model: object,
    pair: SharedTrainingPairV5,
    supervision: ObservableCameraRayV4FrameSupervisionV5,
    *,
    frame_name: str,
    require_b4: bool,
) -> SharedObservableCameraRayV4FrameLossV4:
    _require_real_b4_frame(
        pair,
        supervision,
        frame_name=frame_name,
        require_b4=require_b4,
    )
    frame = pair.current if frame_name == "current" else pair.next
    targets = derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=supervision.pixel_hit_mask,
        pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
        ground_support_in_frustum=supervision.ground_support_in_frustum,
        ground_support_clear_to_target=(
            supervision.ground_support_clear_to_target
        ),
    )
    evidence = frame.evidence
    if not torch.equal(
        evidence.ground_query_in_frustum,
        targets.ground_in_frustum,
    ):
        raise ValueError("model calibration does not reproduce V4 ground visibility")

    first_hit = hierarchical_first_hit_nll_breakdown_v9(
        evidence.pixel_first_hit_hazard_logits,
        targets,
    )
    offset = _skew_balanced_pixel_offset_loss_v5(
        evidence.pixel_within_bin_offset_m,
        targets,
    )
    ground = balanced_ground_clear_bce_v4(
        evidence.ground_clear_to_target_logits,
        targets,
        evidence.ground_target_distance_m,
    )
    raster = soft_rasterize_observable_camera_ray_evidence_v4(
        evidence,
        camera_origin_body_m=frame.camera_origin_body_m,
        camera_basis_body_fru=frame.camera_basis_body_fru,
        pixel_ray_chunk_size=int(model.model_config.v4_pixel_ray_chunk_size),
    )
    derived = hierarchical_raster_cross_entropy_v4(
        raster,
        supervision.target_raster_labels,
    )
    cell_nll = derived_raster_cell_nll_v12(
        raster.class_probabilities,
        supervision.target_raster_labels,
    )
    objective = compose_gate_aligned_objective_v12(
        {
            "hierarchical_first_hit_nll": first_hit.total,
            "target_bin_offset_smooth_l1": offset,
            "ground_clear_distance_state_balanced_bce": ground,
            "derived_raster_hierarchical_bce": derived.total,
        },
        cell_nll,
    )
    if not bool(torch.isfinite(objective.total).item()):
        raise FloatingPointError("V4 frame Camera objective became nonfinite")
    return SharedObservableCameraRayV4FrameLossV4(
        hierarchical_first_hit_nll=first_hit.total,
        hierarchical_first_hit_breakdown=first_hit,
        target_bin_offset_smooth_l1=offset,
        ground_clear_distance_state_balanced_bce=ground,
        derived_raster_hierarchical_bce=derived,
        derived_raster_cell_nll=cell_nll,
        retained_v11_base_total=objective.v11_base_total,
        total=objective.total,
    )


def observable_camera_ray_v4_loss_v4(
    model: object,
    pair: SharedTrainingPairV5,
    current_supervision: ObservableCameraRayV4FrameSupervisionV5,
    next_supervision: ObservableCameraRayV4FrameSupervisionV5,
    *,
    require_b4: bool = True,
) -> SharedObservableCameraRayV4LossV4:
    """Compute current and next independently at B4, then average two scalars."""

    model_config = getattr(model, "model_config", None)
    if getattr(model_config, "observable_camera_ray_v4_weight", None) != (
        CAMERA_MODEL_CONFIG_WEIGHT_V4
    ):
        raise ValueError("V4 Camera model-config weight must remain exactly 1.0")
    current = _frame_loss_v4(
        model,
        pair,
        current_supervision,
        frame_name="current",
        require_b4=require_b4,
    )
    next_ = _frame_loss_v4(
        model,
        pair,
        next_supervision,
        frame_name="next",
        require_b4=require_b4,
    )
    total = 0.5 * current.total + 0.5 * next_.total
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("V4 current/next Camera objective became nonfinite")
    return SharedObservableCameraRayV4LossV4(
        current=current,
        next=next_,
        total=total,
    )


def combine_joint_losses_v4(
    model: object,
    pair: SharedTrainingPairV5,
    current_supervision: ObservableCameraRayV4FrameSupervisionV5,
    next_supervision: ObservableCameraRayV4FrameSupervisionV5,
) -> SharedJointLossV4:
    """Combine established JEPA and the corrected V4 Camera pair scalar."""

    camera = observable_camera_ray_v4_loss_v4(
        model,
        pair,
        current_supervision,
        next_supervision,
    )
    total = pair.jepa.total + camera.total
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("V4 promoted joint objective became nonfinite")
    return SharedJointLossV4(
        total=total,
        established_jepa=pair.jepa,
        observable_camera_ray_v4=camera,
        observable_camera_ray_v4_weight=CAMERA_MODEL_CONFIG_WEIGHT_V4,
    )


def average_four_microbatch_tensor_scalars_v4(
    values: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return the exact equal mean of four already-complete B4 scalars."""

    if isinstance(values, (str, bytes)) or len(values) != 4:
        raise ValueError("V4 update requires exactly four complete B4 scalars")
    if any(
        not isinstance(value, torch.Tensor) or value.ndim != 0
        for value in values
    ):
        raise TypeError("V4 microbatch losses must be scalar tensors")
    device = values[0].device
    if any(value.device != device for value in values):
        raise ValueError("V4 microbatch losses must share one device")
    result = sum(0.25 * value for value in values)
    if not bool(torch.isfinite(result).item()):
        raise FloatingPointError("V4 update scalar became nonfinite")
    return result


__all__ = [
    "CAMERA_MODEL_CONFIG_WEIGHT_V4",
    "EXPECTED_MICROBATCH_SIZE_V4",
    "SharedJointLossV4",
    "SharedObservableCameraRayV4FrameLossV4",
    "SharedObservableCameraRayV4LossV4",
    "average_four_microbatch_tensor_scalars_v4",
    "combine_joint_losses_v4",
    "observable_camera_ray_v4_loss_v4",
]
