#!/usr/bin/env python3
"""Source-only V13 joint-training tensor core.

Callers supply four already-reviewed microbatches.  This module performs no
data discovery, checkpoint access, accelerator selection, or experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
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
        run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
        as _v3,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


ACTION_ORDER = _v3.ACTION_ORDER
MICROBATCHES_PER_UPDATE = _v3.MICROBATCHES_PER_UPDATE
MICROBATCH_SIZE = _v3.MICROBATCH_SIZE
PRESENTATIONS_PER_UPDATE = _v3.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v3.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v3.MAXIMUM_PRESENTATIONS

CURRENT_RGB_KEY = _v3.CURRENT_RGB_KEY
NEXT_RGB_KEY = _v3.NEXT_RGB_KEY
CURRENT_LABELS_KEY = _v3.CURRENT_LABELS_KEY
NEXT_LABELS_KEY = _v3.NEXT_LABELS_KEY
EXECUTED_ACTION_KEY = _v3.EXECUTED_ACTION_KEY
IMMEDIATE_FEASIBLE_KEY = _v3.IMMEDIATE_FEASIBLE_KEY
PREFIX_LENGTHS_KEY = _v3.PREFIX_LENGTHS_KEY

CURRENT_CAMERA_ORIGIN_KEY = "current_camera_origin_body_m"
NEXT_CAMERA_ORIGIN_KEY = "next_camera_origin_body_m"
CURRENT_CAMERA_BASIS_KEY = "current_camera_basis_body_fru"
NEXT_CAMERA_BASIS_KEY = "next_camera_basis_body_fru"
CURRENT_GROUND_PLANE_Z_KEY = "current_ground_plane_z_body_m"
NEXT_GROUND_PLANE_Z_KEY = "next_ground_plane_z_body_m"
CURRENT_PIXEL_HIT_KEY = "current_pixel_hit_mask"
NEXT_PIXEL_HIT_KEY = "next_pixel_hit_mask"
CURRENT_PIXEL_DISTANCE_KEY = "current_pixel_first_hit_distance_m"
NEXT_PIXEL_DISTANCE_KEY = "next_pixel_first_hit_distance_m"
CURRENT_GROUND_IN_FRUSTUM_KEY = "current_ground_support_in_frustum"
NEXT_GROUND_IN_FRUSTUM_KEY = "next_ground_support_in_frustum"
CURRENT_GROUND_CLEAR_KEY = "current_ground_support_clear_to_target"
NEXT_GROUND_CLEAR_KEY = "next_ground_support_clear_to_target"

CAMERA_BATCH_KEYS = (
    CURRENT_CAMERA_ORIGIN_KEY,
    NEXT_CAMERA_ORIGIN_KEY,
    CURRENT_CAMERA_BASIS_KEY,
    NEXT_CAMERA_BASIS_KEY,
    CURRENT_GROUND_PLANE_Z_KEY,
    NEXT_GROUND_PLANE_Z_KEY,
    CURRENT_PIXEL_HIT_KEY,
    NEXT_PIXEL_HIT_KEY,
    CURRENT_PIXEL_DISTANCE_KEY,
    NEXT_PIXEL_DISTANCE_KEY,
    CURRENT_GROUND_IN_FRUSTUM_KEY,
    NEXT_GROUND_IN_FRUSTUM_KEY,
    CURRENT_GROUND_CLEAR_KEY,
    NEXT_GROUND_CLEAR_KEY,
)
REQUIRED_BATCH_KEYS = (*_v3.REQUIRED_BATCH_KEYS, *CAMERA_BATCH_KEYS)

ENCODER_LEARNING_RATE = 1e-4
OTHER_ONLINE_LEARNING_RATE = 3e-4
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPSILON = 1e-8
ADAMW_WEIGHT_DECAY = 1e-4
CAMERA_FRAME_TERM_COUNT = 3
CAMERA_FRAMES_PER_MICROBATCH = 2 * MICROBATCH_SIZE


@dataclass(frozen=True)
class CameraEvidenceFrameSupervisionV13:
    pixel_hit_mask: Any
    pixel_first_hit_distance_m: Any
    ground_support_in_frustum: Any
    ground_support_clear_to_target: Any


@dataclass(frozen=True)
class CameraEvidenceFrameLossV13:
    hierarchical_first_hit_nll: Any
    skew_balanced_pixel_offset: Any
    balanced_ground_clear_bce: Any
    total: Any


@dataclass(frozen=True)
class CameraEvidencePairLossV13:
    current_frames: tuple[CameraEvidenceFrameLossV13, ...]
    next_frames: tuple[CameraEvidenceFrameLossV13, ...]
    current_mean: Any
    next_mean: Any
    total: Any


@dataclass(frozen=True)
class ParameterPartitionV13:
    encoder: tuple[Any, ...]
    evidence_head: tuple[Any, ...]
    representation: tuple[Any, ...]
    predictor: tuple[Any, ...]
    target: tuple[Any, ...]
    names: Mapping[str, tuple[str, ...]]

    @property
    def shared(self) -> tuple[Any, ...]:
        return self.encoder + self.evidence_head

    @property
    def online(self) -> tuple[Any, ...]:
        return self.shared + self.representation + self.predictor

    @property
    def lift_semantic(self) -> tuple[Any, ...]:
        """The single non-encoder/non-predictor AdamW group."""

        return self.evidence_head + self.representation


@dataclass(frozen=True)
class JointTrainingAccountingV13:
    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0


@dataclass(frozen=True)
class GradientRouteReceiptV13:
    preclip_l2: float
    applied_scale: float
    parameter_tensor_count: int
    absent_tensor_gradient_count: int


@dataclass(frozen=True)
class JointUpdateResultV13:
    accounting: JointTrainingAccountingV13
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, GradientRouteReceiptV13]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


def _runtime_apis() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    roots = (str(ROOT), str(ROOT / "lewm_worlds"))
    inserted = [value for value in roots if value not in sys.path]
    try:
        for value in reversed(inserted):
            sys.path.insert(0, value)
        return (
            importlib.import_module("torch"),
            importlib.import_module(
                "lewm.benchmarks.go2_post_action_projective_support_joint_jepa_v1"
            ),
            importlib.import_module(
                "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
            ),
            importlib.import_module(
                "lewm.models.observable_camera_ray_evidence_v4_training"
            ),
            importlib.import_module(
                "lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9"
            ),
            importlib.import_module(
                "lewm.models.shared_observable_camera_ray_jepa_v5"
            ),
            importlib.import_module(
                "lewm.models.observable_camera_ray_evidence_v4"
            ),
        )
    finally:
        for value in inserted:
            sys.path.remove(value)


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite scalar")
    return result


def _finite_tensor(torch: Any, value: Any, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is absent or nonfinite")


def partition_parameters_v13(model: Any) -> ParameterPartitionV13:
    """Resolve the exact three online gradient routes and frozen target."""

    groups: dict[str, list[Any]] = {
        "encoder": [],
        "evidence_head": [],
        "representation": [],
        "predictor": [],
        "target": [],
    }
    names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith("bev_lift.evidence_head."):
            group = "evidence_head"
        elif name.startswith(
            (
                "bev_lift.free_projection.",
                "bev_lift.occupied_projection.",
                "semantic_head.",
            )
        ):
            group = "representation"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith(("target_encoder.", "target_bev_lift.")):
            group = "target"
        else:
            raise RuntimeError(f"unregistered V13 model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)
    if any(not values for values in groups.values()):
        raise RuntimeError("V13 parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("V13 parameter partition overlaps")
    if set(identities) != {id(value) for value in model.parameters()}:
        raise RuntimeError("V13 parameter partition does not cover the model")
    if any(value.requires_grad for value in groups["target"]):
        raise RuntimeError("V13 EMA target parameter is trainable")
    if any(
        not value.requires_grad or str(value.dtype) != "torch.float32"
        for group in ("encoder", "evidence_head", "representation", "predictor")
        for value in groups[group]
    ):
        raise RuntimeError("every V13 online parameter must be trainable float32")
    return ParameterPartitionV13(
        **{name: tuple(values) for name, values in groups.items()},
        names={name: tuple(values) for name, values in names.items()},
    )


def build_frozen_optimizer_v13(model_or_partition: Any) -> Any:
    """Build the sole three-group float32 AdamW optimizer."""

    torch, *_ = _runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV13)
        else partition_parameters_v13(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "encoder",
                "params": list(partition.encoder),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "name": "evidence_projection_semantic",
                "params": list(partition.lift_semantic),
                "lr": OTHER_ONLINE_LEARNING_RATE,
            },
            {
                "name": "predictor",
                "params": list(partition.predictor),
                "lr": OTHER_ONLINE_LEARNING_RATE,
            },
        ],
        betas=ADAMW_BETAS,
        eps=ADAMW_EPSILON,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )
    validate_optimizer_v13(optimizer, partition)
    return optimizer


def validate_optimizer_v13(
    optimizer: Any, partition: ParameterPartitionV13
) -> None:
    expected = (
        ("encoder", partition.encoder, ENCODER_LEARNING_RATE),
        (
            "evidence_projection_semantic",
            partition.lift_semantic,
            OTHER_ONLINE_LEARNING_RATE,
        ),
        ("predictor", partition.predictor, OTHER_ONLINE_LEARNING_RATE),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("V13 optimizer must be the sole three-group AdamW")
    observed_ids: list[int] = []
    for observed, (name, parameters, learning_rate) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        observed_parameters = tuple(observed["params"])
        observed_ids.extend(id(value) for value in observed_parameters)
        if (
            observed.get("name") != name
            or tuple(map(id, observed_parameters)) != tuple(map(id, parameters))
            or float(observed["lr"]) != learning_rate
            or tuple(observed["betas"]) != ADAMW_BETAS
            or float(observed["eps"]) != ADAMW_EPSILON
            or float(observed["weight_decay"]) != ADAMW_WEIGHT_DECAY
        ):
            raise RuntimeError(f"V13 optimizer group {name!r} changed")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != {
        id(value) for value in partition.online
    }:
        raise RuntimeError("V13 optimizer membership is incomplete or overlapping")


def validate_accounting_v13(accounting: JointTrainingAccountingV13) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V13 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV13(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 2 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=updates
        * MICROBATCHES_PER_UPDATE
        * CAMERA_FRAMES_PER_MICROBATCH,
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * MICROBATCHES_PER_UPDATE,
    )
    if accounting != expected:
        raise RuntimeError("V13 joint-training accounting is inconsistent")


def _advance_accounting_v13(
    accounting: JointTrainingAccountingV13,
) -> JointTrainingAccountingV13:
    result = JointTrainingAccountingV13(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=accounting.microbatch_graphs + MICROBATCHES_PER_UPDATE,
        backward_calls=accounting.backward_calls + 2 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=accounting.camera_route_grad_calls
        + MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=accounting.joint_route_grad_calls
        + MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=accounting.camera_frame_objectives
        + MICROBATCHES_PER_UPDATE * CAMERA_FRAMES_PER_MICROBATCH,
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
        predictor_forwards=accounting.predictor_forwards
        + MICROBATCHES_PER_UPDATE,
        predictor_objectives=accounting.predictor_objectives
        + MICROBATCHES_PER_UPDATE,
    )
    validate_accounting_v13(result)
    return result


def _validate_update_capacity_v13(
    accounting: JointTrainingAccountingV13,
) -> None:
    """Fail before graph/model/optimizer work if one more update exceeds the cap."""

    validate_accounting_v13(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE
        > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V13 training cap leaves no complete update available")


def _slice_raw_output_v13(raw: Any, row: int, raw_type: Any) -> Any:
    return raw_type(
        pixel_first_hit_hazard_logits=raw.pixel_first_hit_hazard_logits[row : row + 1],
        pixel_within_bin_offset_m=raw.pixel_within_bin_offset_m[row : row + 1],
        ground_clear_to_target_logits=raw.ground_clear_to_target_logits[row : row + 1],
        ground_query_in_frustum=raw.ground_query_in_frustum[row : row + 1],
        ground_query_uv_px=raw.ground_query_uv_px[row : row + 1],
        ground_target_distance_m=raw.ground_target_distance_m[row : row + 1],
    )


def camera_evidence_frame_loss_v13(
    raw_output: Any,
    supervision: CameraEvidenceFrameSupervisionV13,
) -> CameraEvidenceFrameLossV13:
    """Compute exactly the three equal fine-evidence terms for one B=1 frame."""

    torch, _, _, training, hierarchical, shared_v5, _ = _runtime_apis()
    hazard = raw_output.pixel_first_hit_hazard_logits
    if tuple(hazard.shape[:2]) != (1, 64):
        raise ValueError("one V13 Camera frame must contain 64 ordered depth bins")
    targets = training.derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=supervision.pixel_hit_mask,
        pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
        ground_support_in_frustum=supervision.ground_support_in_frustum,
        ground_support_clear_to_target=supervision.ground_support_clear_to_target,
    )
    if not torch.equal(
        raw_output.ground_query_in_frustum,
        targets.ground_in_frustum,
    ):
        raise ValueError(
            "V13 auxiliary calibration validity differs from ground targets"
        )
    first_hit = hierarchical.hierarchical_first_hit_nll_breakdown_v9(
        hazard, targets
    ).total
    offset = shared_v5._skew_balanced_pixel_offset_loss_v5(
        raw_output.pixel_within_bin_offset_m, targets
    )
    ground = training.balanced_ground_clear_bce_v4(
        raw_output.ground_clear_to_target_logits,
        targets,
        raw_output.ground_target_distance_m,
    )
    total = (first_hit + offset + ground) / CAMERA_FRAME_TERM_COUNT
    for name, value in (
        ("hierarchical first-hit NLL", first_hit),
        ("skew-balanced pixel offset", offset),
        ("balanced ground-clear BCE", ground),
        ("Camera frame loss", total),
    ):
        _finite_tensor(torch, value, name)
    return CameraEvidenceFrameLossV13(first_hit, offset, ground, total)


def camera_evidence_pair_loss_v13(
    current_raw: Any,
    next_raw: Any,
    current_supervision: CameraEvidenceFrameSupervisionV13,
    next_supervision: CameraEvidenceFrameSupervisionV13,
) -> CameraEvidencePairLossV13:
    """Average four B=1 frame losses, then current/next with equal weight."""

    torch, *_, raw_api = _runtime_apis()
    raw_type = raw_api.ObservableCameraRayEvidenceV4RawOutput
    if current_raw.pixel_first_hit_hazard_logits.shape[0] != MICROBATCH_SIZE or (
        next_raw.pixel_first_hit_hazard_logits.shape[0] != MICROBATCH_SIZE
    ):
        raise ValueError("V13 Camera pair loss requires current/next B=4")

    def frame(supervision: CameraEvidenceFrameSupervisionV13, row: int) -> CameraEvidenceFrameSupervisionV13:
        return CameraEvidenceFrameSupervisionV13(
            pixel_hit_mask=supervision.pixel_hit_mask[row : row + 1],
            pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m[
                row : row + 1
            ],
            ground_support_in_frustum=supervision.ground_support_in_frustum[
                row : row + 1
            ],
            ground_support_clear_to_target=supervision.ground_support_clear_to_target[
                row : row + 1
            ],
        )

    current_frames = tuple(
        camera_evidence_frame_loss_v13(
            _slice_raw_output_v13(current_raw, row, raw_type), frame(current_supervision, row)
        )
        for row in range(MICROBATCH_SIZE)
    )
    next_frames = tuple(
        camera_evidence_frame_loss_v13(
            _slice_raw_output_v13(next_raw, row, raw_type), frame(next_supervision, row)
        )
        for row in range(MICROBATCH_SIZE)
    )
    current_mean = torch.stack([value.total for value in current_frames]).mean()
    next_mean = torch.stack([value.total for value in next_frames]).mean()
    return CameraEvidencePairLossV13(
        current_frames=current_frames,
        next_frames=next_frames,
        current_mean=current_mean,
        next_mean=next_mean,
        total=0.5 * current_mean + 0.5 * next_mean,
    )


def _validate_microbatches_v13(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("one V13 update requires exactly four microbatches")
    for index, batch in enumerate(microbatches):
        missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f"V13 microbatch {index} is missing {missing}")
        if any(
            not isinstance(batch[key], torch.Tensor)
            or batch[key].shape[0] != MICROBATCH_SIZE
            for key in REQUIRED_BATCH_KEYS
        ):
            raise ValueError(f"V13 microbatch {index} tensors must contain four rows")
        expected_shapes = {
            CURRENT_CAMERA_ORIGIN_KEY: (4, 3),
            NEXT_CAMERA_ORIGIN_KEY: (4, 3),
            CURRENT_CAMERA_BASIS_KEY: (4, 3, 3),
            NEXT_CAMERA_BASIS_KEY: (4, 3, 3),
            CURRENT_GROUND_PLANE_Z_KEY: (4,),
            NEXT_GROUND_PLANE_Z_KEY: (4,),
            IMMEDIATE_FEASIBLE_KEY: (4, 9),
            PREFIX_LENGTHS_KEY: (4, 9),
        }
        for key, shape in expected_shapes.items():
            if tuple(batch[key].shape) != shape:
                raise ValueError(f"V13 {key} must have shape {shape}")


def _zero_accumulators(parameters: Sequence[Any]) -> list[Any]:
    return [parameter.detach().new_zeros(parameter.shape) for parameter in parameters]


def _accumulate_gradients(
    accumulators: list[Any], gradients: Sequence[Any | None]
) -> int:
    absent = 0
    for accumulator, gradient in zip(accumulators, gradients, strict=True):
        if gradient is None:
            absent += 1
        else:
            accumulator.add_(gradient.detach())
    return absent


def _route_norm_and_scale_v13(torch: Any, gradients: Sequence[Any]) -> tuple[Any, Any]:
    if not gradients:
        raise RuntimeError("V13 gradient route is empty")
    total = gradients[0].new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        total = total + (gradient.float() * gradient.float()).sum(dtype=torch.float32)
    norm = torch.sqrt(total)
    tiny = norm.new_tensor(torch.finfo(torch.float32).tiny)
    scale = torch.minimum(
        norm.new_tensor(1.0),
        torch.reciprocal(torch.maximum(norm, tiny)),
    )
    if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(scale)):
        raise FloatingPointError("V13 gradient norm or scale is nonfinite")
    return norm, scale


def _receipt_v13(
    norm: Any, scale: Any, parameters: Sequence[Any], absent: int
) -> GradientRouteReceiptV13:
    return GradientRouteReceiptV13(
        preclip_l2=_scalar(norm),
        applied_scale=_scalar(scale),
        parameter_tensor_count=len(parameters),
        absent_tensor_gradient_count=absent,
    )


def joint_training_update_v13(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV13 | None = None,
) -> JointUpdateResultV13:
    """Run one exact four-microbatch C/N-balanced optimizer/EMA update."""

    torch, semantic_api, survival_api, *_ = _runtime_apis()
    state = JointTrainingAccountingV13() if accounting is None else accounting
    _validate_update_capacity_v13(state)
    _validate_microbatches_v13(torch, microbatches)
    partition = partition_parameters_v13(model)
    validate_optimizer_v13(optimizer, partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V13 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V13 EMA target already has a gradient")

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _zero_accumulators(partition.shared)
    joint_shared = _zero_accumulators(partition.shared)
    joint_representation = _zero_accumulators(partition.representation)
    joint_predictor = _zero_accumulators(partition.predictor)
    absent = {name: 0 for name in ("camera_shared", "joint_shared", "representation", "predictor")}
    sums = {name: 0.0 for name in ("S", "P", "U", "R", "O", "N", "C", "L")}
    active_ranking = eligible_pairs = supervised_decisions = 0

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
        occupied = _v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(current_latent)
        predicted, survival_logits = _v3._v2._v1._prediction_parts(prediction)
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
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _finite_tensor(torch, value, name)
        if not camera.total.requires_grad or not navigation.requires_grad:
            raise RuntimeError("V13 C and N must both retain gradient graphs")

        c_gradients = torch.autograd.grad(
            camera.total / MICROBATCHES_PER_UPDATE,
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
        absent["camera_shared"] += _accumulate_gradients(camera_shared, c_gradients)
        absent["joint_shared"] += _accumulate_gradients(
            joint_shared, n_gradients[:shared_end]
        )
        absent["representation"] += _accumulate_gradients(
            joint_representation, n_gradients[shared_end:representation_end]
        )
        absent["predictor"] += _accumulate_gradients(
            joint_predictor, n_gradients[representation_end:]
        )

        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("N", navigation),
            ("C", camera.total),
            ("L", navigation + camera.total),
        ):
            sums[name] += _scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(joint.survival_terms.supervised_decision_count.item())

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (partition.representation, joint_representation),
        "predictor": (partition.predictor, joint_predictor),
    }
    route_values = {
        name: _route_norm_and_scale_v13(torch, gradients)
        for name, (_, gradients) in route_tensors.items()
    }
    for name in ("camera_shared", "joint_shared", "predictor"):
        if not (_scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V13 gradient route {name!r} is zero")

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
    for parameter, gradient in zip(partition.predictor, joint_predictor, strict=True):
        parameter.grad = predictor_scale * gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V13 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _finite_tensor(torch, parameter, "V13 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V13 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V13 EMA target received a gradient")

    advanced = _advance_accounting_v13(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V13 EMA count disagrees with accounting")
    receipts = {
        name: _receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    return JointUpdateResultV13(
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
    )


__all__ = [
    "ACTION_ORDER",
    "ADAMW_BETAS",
    "ADAMW_EPSILON",
    "ADAMW_WEIGHT_DECAY",
    "CAMERA_BATCH_KEYS",
    "CURRENT_CAMERA_BASIS_KEY",
    "CURRENT_CAMERA_ORIGIN_KEY",
    "CURRENT_GROUND_CLEAR_KEY",
    "CURRENT_GROUND_IN_FRUSTUM_KEY",
    "CURRENT_GROUND_PLANE_Z_KEY",
    "CURRENT_LABELS_KEY",
    "CURRENT_PIXEL_DISTANCE_KEY",
    "CURRENT_PIXEL_HIT_KEY",
    "CURRENT_RGB_KEY",
    "CameraEvidenceFrameLossV13",
    "CameraEvidenceFrameSupervisionV13",
    "CameraEvidencePairLossV13",
    "ENCODER_LEARNING_RATE",
    "EXECUTED_ACTION_KEY",
    "GradientRouteReceiptV13",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV13",
    "JointUpdateResultV13",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_CAMERA_BASIS_KEY",
    "NEXT_CAMERA_ORIGIN_KEY",
    "NEXT_GROUND_CLEAR_KEY",
    "NEXT_GROUND_IN_FRUSTUM_KEY",
    "NEXT_GROUND_PLANE_Z_KEY",
    "NEXT_LABELS_KEY",
    "NEXT_PIXEL_DISTANCE_KEY",
    "NEXT_PIXEL_HIT_KEY",
    "NEXT_RGB_KEY",
    "OTHER_ONLINE_LEARNING_RATE",
    "PRESENTATIONS_PER_UPDATE",
    "PREFIX_LENGTHS_KEY",
    "ParameterPartitionV13",
    "REQUIRED_BATCH_KEYS",
    "build_frozen_optimizer_v13",
    "camera_evidence_frame_loss_v13",
    "camera_evidence_pair_loss_v13",
    "joint_training_update_v13",
    "partition_parameters_v13",
    "validate_accounting_v13",
    "validate_optimizer_v13",
]
