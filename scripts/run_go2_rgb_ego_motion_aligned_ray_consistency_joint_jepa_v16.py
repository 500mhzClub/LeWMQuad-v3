#!/usr/bin/env python3
"""Source-only V16 joint-training adapter.

V16 preserves the reviewed V13 joint-JEPA update and adds the preregistered
ego-motion-aligned ray-consistency term only to the Camera/shared-encoder
route.  This module performs no data discovery, checkpoint access,
accelerator selection, or experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_inserted_root = str(ROOT) not in sys.path
if _inserted_root:
    sys.path.insert(0, str(ROOT))
try:
    from scripts import (
        run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
        as _base,
    )
    from lewm.models.ego_motion_aligned_ray_consistency_v16 import (
        EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16,
        EgoMotionAlignedRayConsistencyReceiptV16,
        ego_motion_aligned_ray_consistency_v16,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


if EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16 != 0.1:
    raise RuntimeError("V16 ray-consistency weight changed")

# Re-export the reviewed V13 tensor-core surface.  The compatibility names
# overridden below let the unchanged executor select this adapter.
for _name in _base.__all__:
    globals()[_name] = getattr(_base, _name)

REALIZED_RELATIVE_SE2_KEY = "realized_relative_se2_current_frame"
CAMERA_BATCH_KEYS = (*_base.CAMERA_BATCH_KEYS, REALIZED_RELATIVE_SE2_KEY)
REQUIRED_BATCH_KEYS = (*_base.REQUIRED_BATCH_KEYS, REALIZED_RELATIVE_SE2_KEY)


@dataclass(frozen=True)
class JointUpdateResultV16(_base.JointUpdateResultV13):
    """V13 update receipt plus aggregate V16 consistency support."""

    ray_consistency_shared_valid_cell_count: int
    ray_consistency_positive_weight_cell_count: int
    ray_consistency_weight_sum: float


def _validate_microbatches_v16(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    """Require the exact V13 batch plus finite realized SE(2) rows."""

    _base._validate_microbatches_v13(torch, microbatches)
    for index, batch in enumerate(microbatches):
        if tuple(batch) != REQUIRED_BATCH_KEYS:
            raise ValueError(
                f"V16 microbatch {index} key order or membership changed"
            )
        realized = batch[REALIZED_RELATIVE_SE2_KEY]
        if (
            not isinstance(realized, torch.Tensor)
            or tuple(realized.shape) != (MICROBATCH_SIZE, 3)
            or realized.dtype != torch.float32
        ):
            raise ValueError(
                "V16 realized relative SE(2) must be float32 with shape (4,3)"
            )
        if realized.device != batch[CURRENT_RGB_KEY].device:
            raise ValueError("V16 realized relative SE(2) must share the RGB device")
        if not bool(torch.isfinite(realized).all().item()):
            raise FloatingPointError("V16 realized relative SE(2) is nonfinite")


def _validate_consistency_receipt_v16(
    receipt: EgoMotionAlignedRayConsistencyReceiptV16,
) -> None:
    if not isinstance(receipt, EgoMotionAlignedRayConsistencyReceiptV16):
        raise TypeError("V16 consistency helper returned the wrong receipt type")
    for name, value in (
        ("shared valid cell count", receipt.shared_valid_cell_count),
        ("positive weight cell count", receipt.positive_weight_cell_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V16 {name} must be a nonnegative integer")
    if (
        not math.isfinite(receipt.weight_sum)
        or receipt.weight_sum < 0.0
    ):
        raise ValueError("V16 consistency weight sum must be finite and nonnegative")


def joint_training_update_v16(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV13 | None = None,
) -> JointUpdateResultV16:
    """Run one four-microbatch V16 update with C=C_base+0.1*M."""

    torch, semantic_api, survival_api, *_ = _base._runtime_apis()
    state = JointTrainingAccountingV13() if accounting is None else accounting
    _base._validate_update_capacity_v13(state)
    _validate_microbatches_v16(torch, microbatches)
    partition = partition_parameters_v13(model)
    validate_optimizer_v13(optimizer, partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V16 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V16 EMA target already has a gradient")

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _base._zero_accumulators(partition.shared)
    joint_shared = _base._zero_accumulators(partition.shared)
    joint_representation = _base._zero_accumulators(partition.representation)
    joint_predictor = _base._zero_accumulators(partition.predictor)
    absent = {
        name: 0
        for name in ("camera_shared", "joint_shared", "representation", "predictor")
    }
    sums = {
        name: 0.0
        for name in ("S", "P", "U", "R", "O", "N", "C_base", "M", "C", "L")
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    shared_valid_cells = positive_weight_cells = 0
    ray_weight_sum = 0.0

    for batch in microbatches:
        current_encoding = model.encode_online_training(
            batch[CURRENT_RGB_KEY],
            camera_origin_body_m=batch[CURRENT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[CURRENT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[CURRENT_GROUND_PLANE_Z_KEY],
        )
        next_encoding = model.encode_online_training(
            batch[NEXT_RGB_KEY],
            camera_origin_body_m=batch[NEXT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[NEXT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[NEXT_GROUND_PLANE_Z_KEY],
        )
        current_latent = current_encoding.latent
        next_latent = next_encoding.latent
        current_logits = model.semantic_logits_from_latent(current_latent)
        next_logits = model.semantic_logits_from_latent(next_latent)
        semantic = semantic_api.semantic_loss_v1(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        occupied = _base._v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(current_latent)
        predicted, survival_logits = _base._v3._v2._v1._prediction_parts(prediction)
        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
        persistence = semantic_api.microbatch_persistence_loss_v1(
            predicted,
            batch[EXECUTED_ACTION_KEY],
            ema_current,
            ema_next,
        )
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=persistence.loss,
            survival_logits=survival_logits,
            immediate_feasible=batch[IMMEDIATE_FEASIBLE_KEY],
            prefix_lengths=batch[PREFIX_LENGTHS_KEY],
        )
        navigation = joint.loss + occupied.loss
        camera = camera_evidence_pair_loss_v13(
            current_encoding.auxiliary_evidence,
            next_encoding.auxiliary_evidence,
            CameraEvidenceFrameSupervisionV13(
                batch[CURRENT_PIXEL_HIT_KEY],
                batch[CURRENT_PIXEL_DISTANCE_KEY],
                batch[CURRENT_GROUND_IN_FRUSTUM_KEY],
                batch[CURRENT_GROUND_CLEAR_KEY],
            ),
            CameraEvidenceFrameSupervisionV13(
                batch[NEXT_PIXEL_HIT_KEY],
                batch[NEXT_PIXEL_DISTANCE_KEY],
                batch[NEXT_GROUND_IN_FRUSTUM_KEY],
                batch[NEXT_GROUND_CLEAR_KEY],
            ),
        )
        consistency = ego_motion_aligned_ray_consistency_v16(
            current_encoding.auxiliary_evidence,
            next_encoding.auxiliary_evidence,
            current_camera_origin_body_m=batch[CURRENT_CAMERA_ORIGIN_KEY],
            current_camera_basis_body_fru=batch[CURRENT_CAMERA_BASIS_KEY],
            next_camera_origin_body_m=batch[NEXT_CAMERA_ORIGIN_KEY],
            next_camera_basis_body_fru=batch[NEXT_CAMERA_BASIS_KEY],
            relative_se2_current_frame=batch[REALIZED_RELATIVE_SE2_KEY],
        )
        _validate_consistency_receipt_v16(consistency)
        camera_v16 = camera.total + (
            EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16 * consistency.loss
        )
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("joint N", navigation),
            ("Camera C_base", camera.total),
            ("ray consistency M", consistency.loss),
            ("Camera C", camera_v16),
        ):
            _base._finite_tensor(torch, value, name)
        if not camera_v16.requires_grad or not navigation.requires_grad:
            raise RuntimeError("V16 C and N must both retain gradient graphs")

        # Exactly one Camera/shared and one Navigation/joint grad call per
        # microbatch, preserving the reviewed optimizer routing.
        c_gradients = torch.autograd.grad(
            camera_v16 / MICROBATCHES_PER_UPDATE,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        n_parameters = partition.shared + partition.representation + partition.predictor
        n_gradients = torch.autograd.grad(
            navigation / MICROBATCHES_PER_UPDATE,
            n_parameters,
            allow_unused=True,
        )
        shared_end = len(partition.shared)
        representation_end = shared_end + len(partition.representation)
        absent["camera_shared"] += _base._accumulate_gradients(
            camera_shared, c_gradients
        )
        absent["joint_shared"] += _base._accumulate_gradients(
            joint_shared, n_gradients[:shared_end]
        )
        absent["representation"] += _base._accumulate_gradients(
            joint_representation, n_gradients[shared_end:representation_end]
        )
        absent["predictor"] += _base._accumulate_gradients(
            joint_predictor, n_gradients[representation_end:]
        )

        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("N", navigation),
            ("C_base", camera.total),
            ("M", consistency.loss),
            ("C", camera_v16),
            ("L", navigation + camera_v16),
        ):
            sums[name] += _base._scalar(value)
        shared_valid_cells += consistency.shared_valid_cell_count
        positive_weight_cells += consistency.positive_weight_cell_count
        ray_weight_sum += consistency.weight_sum
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (partition.representation, joint_representation),
        "predictor": (partition.predictor, joint_predictor),
    }
    route_values = {
        name: _base._route_norm_and_scale_v13(torch, gradients)
        for name, (_, gradients) in route_tensors.items()
    }
    for name in ("camera_shared", "joint_shared", "predictor"):
        if not (_base._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V16 gradient route {name!r} is zero")

    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    for parameter, c_gradient, n_gradient in zip(
        partition.shared, camera_shared, joint_shared, strict=True
    ):
        parameter.grad = c_scale * c_gradient + n_scale * n_gradient
    representation_scale = route_values["representation"][1]
    for parameter, gradient in zip(
        partition.representation, joint_representation, strict=True
    ):
        parameter.grad = representation_scale * gradient
    predictor_scale = route_values["predictor"][1]
    for parameter, gradient in zip(
        partition.predictor, joint_predictor, strict=True
    ):
        parameter.grad = predictor_scale * gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V16 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _base._finite_tensor(torch, parameter, "V16 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V16 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V16 EMA target received a gradient")

    advanced = _base._advance_accounting_v13(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V16 EMA count disagrees with accounting")
    receipts = {
        name: _base._receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    return JointUpdateResultV16(
        accounting=advanced,
        mean_losses={
            name: value / MICROBATCHES_PER_UPDATE for name, value in sums.items()
        },
        gradient_routes=receipts,
        gradient_l2={name: receipt.preclip_l2 for name, receipt in receipts.items()},
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
        ray_consistency_shared_valid_cell_count=shared_valid_cells,
        ray_consistency_positive_weight_cell_count=positive_weight_cells,
        ray_consistency_weight_sum=ray_weight_sum,
    )


# Compatibility hooks used by the reviewed V13 executor.
_validate_microbatches_v13 = _validate_microbatches_v16
joint_training_update_v13 = joint_training_update_v16

__all__ = tuple(
    dict.fromkeys(
        (
            *_base.__all__,
            "EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16",
            "EgoMotionAlignedRayConsistencyReceiptV16",
            "JointUpdateResultV16",
            "REALIZED_RELATIVE_SE2_KEY",
            "ego_motion_aligned_ray_consistency_v16",
            "joint_training_update_v16",
        )
    )
)
