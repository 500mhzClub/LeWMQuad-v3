#!/usr/bin/env python3
"""Lean tensor core for the memory-role factorized joint-JEPA V1 probe.

The inherited physical route stays intact.  Two small RGB-only routes jointly
train the same online trunk: an action-sensitive immediate-control state and a
repeated-observation place key intended for a later learned memory.  This module
does no dataset or experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from scripts import (
    run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25 as v25,
)


MICROBATCH_SIZE_V1 = 4
PHYSICAL_MICROBATCHES_PER_UPDATE_V1 = 4
LOCAL_MICROBATCHES_PER_UPDATE_V1 = 2
PLACE_MICROBATCHES_PER_UPDATE_V1 = 2
PLACE_GRAPH_BATCH_SIZE_V3 = 8
PLACE_GRAPHS_PER_UPDATE_V3 = 1
PHYSICAL_PRESENTATIONS_PER_UPDATE_V1 = 16
LOCAL_PRESENTATIONS_PER_UPDATE_V1 = 8
PLACE_PRESENTATIONS_PER_UPDATE_V1 = 8
PRESENTATIONS_PER_UPDATE_V1 = 32
MAXIMUM_UPDATES_V1 = 400
MAXIMUM_PRESENTATIONS_V1 = 12_800

PHYSICAL_RGB_DECODES_PER_UPDATE_V1 = 32
LOCAL_RGB_DECODES_PER_UPDATE_V1 = 16
PLACE_RGB_DECODES_PER_UPDATE_V1 = 24
RGB_DECODES_PER_UPDATE_V1 = 72
ONLINE_RGB_ENCODINGS_PER_UPDATE_V1 = 48
EMA_TARGET_RGB_ENCODINGS_PER_UPDATE_V1 = 24

ACTION_COUNT_V1 = 9
LOCAL_WRONG_ACTION_MARGIN_V1 = 0.05
PLACE_CONTRAST_TEMPERATURE_V3 = 0.10
PLACE_VARIANCE_FLOOR_V3 = 0.05
PLACE_VARIANCE_EPSILON_V3 = 1.0e-4
PLACE_COVARIANCE_WEIGHT_V3 = 0.10
PLACE_KEY_DIMENSION_V3 = 64
LOCAL_ROUTE_NAME_V1 = "immediate_action_local_control"
PLACE_ROUTE_NAME_V1 = "same_place_retrieval_key"

LOCAL_CURRENT_RGB_KEY_V1 = "current_rgb"
LOCAL_NEXT_RGB_KEY_V1 = "next_rgb"
LOCAL_ACTION_KEY_V1 = "action"
REQUIRED_LOCAL_BATCH_KEYS_V1 = (
    LOCAL_CURRENT_RGB_KEY_V1,
    LOCAL_NEXT_RGB_KEY_V1,
    LOCAL_ACTION_KEY_V1,
)
PLACE_ANCHOR_RGB_KEY_V1 = "anchor_rgb"
PLACE_POSITIVE_RGB_KEY_V1 = "positive_rgb"
PLACE_NEGATIVE_RGB_KEY_V1 = "negative_rgb"
REQUIRED_PLACE_BATCH_KEYS_V1 = (
    PLACE_ANCHOR_RGB_KEY_V1,
    PLACE_POSITIVE_RGB_KEY_V1,
    PLACE_NEGATIVE_RGB_KEY_V1,
)

# Exact compatibility facade consumed by the reviewed physical batch builder.
CURRENT_RGB_KEY = v25.CURRENT_RGB_KEY
REQUIRED_BATCH_KEYS = v25.REQUIRED_BATCH_KEYS
CURRENT_CAMERA_ORIGIN_KEY = v25.CURRENT_CAMERA_ORIGIN_KEY
NEXT_CAMERA_ORIGIN_KEY = v25.NEXT_CAMERA_ORIGIN_KEY
CURRENT_CAMERA_BASIS_KEY = v25.CURRENT_CAMERA_BASIS_KEY
NEXT_CAMERA_BASIS_KEY = v25.NEXT_CAMERA_BASIS_KEY
CURRENT_GROUND_PLANE_Z_KEY = v25.CURRENT_GROUND_PLANE_Z_KEY
NEXT_GROUND_PLANE_Z_KEY = v25.NEXT_GROUND_PLANE_Z_KEY
CURRENT_PIXEL_HIT_KEY = v25.CURRENT_PIXEL_HIT_KEY
NEXT_PIXEL_HIT_KEY = v25.NEXT_PIXEL_HIT_KEY
CURRENT_PIXEL_DISTANCE_KEY = v25.CURRENT_PIXEL_DISTANCE_KEY
NEXT_PIXEL_DISTANCE_KEY = v25.NEXT_PIXEL_DISTANCE_KEY
CURRENT_GROUND_IN_FRUSTUM_KEY = v25.CURRENT_GROUND_IN_FRUSTUM_KEY
NEXT_GROUND_IN_FRUSTUM_KEY = v25.NEXT_GROUND_IN_FRUSTUM_KEY
CURRENT_GROUND_CLEAR_KEY = v25.CURRENT_GROUND_CLEAR_KEY
NEXT_GROUND_CLEAR_KEY = v25.NEXT_GROUND_CLEAR_KEY
SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    v25.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
REQUIRED_BATCH_KEYS_V21 = v25.REQUIRED_BATCH_KEYS_V21
ACTION_PRIOR_M_KEY_V23 = v25.ACTION_PRIOR_M_KEY_V23
REQUIRED_BATCH_KEYS_V23 = v25.REQUIRED_BATCH_KEYS_V23
REQUIRED_BATCH_KEYS_V24 = v25.REQUIRED_BATCH_KEYS_V24
REQUIRED_BATCH_KEYS_V25 = v25.REQUIRED_BATCH_KEYS_V25
REQUIRED_BATCH_KEYS_V1 = REQUIRED_BATCH_KEYS_V25


@dataclass(frozen=True)
class ParameterPartitionV1:
    encoder: tuple[Any, ...]
    evidence_head: tuple[Any, ...]
    representation: tuple[Any, ...]
    predictor: tuple[Any, ...]
    role_factorizer: tuple[Any, ...]
    place_predictor: tuple[Any, ...]
    local_predictor: tuple[Any, ...]
    target: tuple[Any, ...]
    names: Mapping[str, tuple[str, ...]]

    @property
    def shared(self) -> tuple[Any, ...]:
        return self.encoder + self.evidence_head

    @property
    def lift_semantic_roles(self) -> tuple[Any, ...]:
        return self.evidence_head + self.representation + self.role_factorizer

    @property
    def online(self) -> tuple[Any, ...]:
        return (
            self.shared
            + self.representation
            + self.predictor
            + self.role_factorizer
            + self.place_predictor
            + self.local_predictor
        )

    @property
    def physical_view(self) -> Any:
        return v25._base.ParameterPartitionV13(
            encoder=self.encoder,
            evidence_head=self.evidence_head,
            representation=self.representation,
            predictor=self.predictor,
            target=self.target,
            names={
                name: self.names[name]
                for name in (
                    "encoder",
                    "evidence_head",
                    "representation",
                    "predictor",
                    "target",
                )
            },
        )

    @property
    def spatial_trunk(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for name, parameter in zip(
                self.names["representation"], self.representation, strict=True
            )
            if name.startswith(
                ("bev_lift.point_projection.", "bev_lift.volume_block.")
            )
        )

    @property
    def place_factorizer(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for name, parameter in zip(
                self.names["role_factorizer"], self.role_factorizer, strict=True
            )
            if name.startswith(
                (
                    "role_factorizer.place_projection.",
                    "role_factorizer.place_output.",
                )
            )
        )

    @property
    def local_factorizer(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for name, parameter in zip(
                self.names["role_factorizer"], self.role_factorizer, strict=True
            )
            if name.startswith("role_factorizer.local_projection.")
        )

    @property
    def place_recipients(self) -> tuple[Any, ...]:
        return (
            self.shared
            + self.spatial_trunk
            + self.place_factorizer
            + self.place_predictor
        )

    @property
    def local_recipients(self) -> tuple[Any, ...]:
        return (
            self.shared
            + self.spatial_trunk
            + self.local_factorizer
            + self.local_predictor
        )


@dataclass(frozen=True)
class JointTrainingAccountingV1:
    updates: int = 0
    presentations: int = 0
    physical_presentations: int = 0
    local_presentations: int = 0
    place_presentations: int = 0
    rgb_decodes: int = 0
    physical_rgb_decodes: int = 0
    local_rgb_decodes: int = 0
    place_rgb_decodes: int = 0
    online_rgb_encodings: int = 0
    ema_target_rgb_encodings: int = 0
    physical_microbatch_graphs: int = 0
    local_microbatch_graphs: int = 0
    place_microbatch_graphs: int = 0
    autograd_grad_calls: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0


@dataclass(frozen=True)
class JointUpdateResultV1:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    local_diagnostics: Mapping[str, Any]
    place_diagnostics: Mapping[str, Any]
    predictor_core_protected_survival_diagnostics: Mapping[str, float | int]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


@dataclass(frozen=True)
class PlaceObjectiveTermsV3:
    loss: Any
    alignment: Any
    contrast: Any
    variance: Any
    covariance: Any
    positive_energy: Any
    negative_energy: Any
    logits: Any


def partition_parameters_v1(model: Any) -> ParameterPartitionV1:
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "evidence_head": [],
        "representation": [],
        "predictor": [],
        "role_factorizer": [],
        "place_predictor": [],
        "local_predictor": [],
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
                "bev_lift.point_projection.",
                "bev_lift.volume_block.",
                "semantic_head.",
            )
        ):
            group = "representation"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith("role_factorizer."):
            group = "role_factorizer"
        elif name.startswith("place_predictor."):
            group = "place_predictor"
        elif name.startswith("local_predictor."):
            group = "local_predictor"
        elif name.startswith(
            (
                "target_encoder.",
                "target_bev_lift.evidence_head.",
                "target_bev_lift.point_projection.",
                "target_bev_lift.volume_block.",
                "target_role_factorizer.",
            )
        ):
            group = "target"
        else:
            raise RuntimeError(f"unregistered memory-role model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)
    if any(not values for values in groups.values()):
        raise RuntimeError("memory-role parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if len(identities) != len(set(identities)) or set(identities) != {
        id(value) for value in model.parameters()
    }:
        raise RuntimeError("memory-role parameter partition is incomplete or overlapping")
    if any(parameter.requires_grad for parameter in groups["target"]):
        raise RuntimeError("memory-role EMA target parameter is trainable")
    if any(
        not parameter.requires_grad or str(parameter.dtype) != "torch.float32"
        for group in (
            "encoder",
            "evidence_head",
            "representation",
            "predictor",
            "role_factorizer",
            "place_predictor",
            "local_predictor",
        )
        for parameter in groups[group]
    ):
        raise RuntimeError("every memory-role online parameter must be trainable float32")
    partition = ParameterPartitionV1(
        **{name: tuple(values) for name, values in groups.items()},
        names={name: tuple(values) for name, values in names.items()},
    )
    if (
        not partition.spatial_trunk
        or not partition.place_factorizer
        or not partition.local_factorizer
        or set(map(id, partition.place_factorizer))
        & set(map(id, partition.local_factorizer))
    ):
        raise RuntimeError("memory-role route-specific parameter binding changed")
    v25._v24.predictor_core_protected_survival_parameter_subset_v24(
        partition.physical_view
    )
    return partition


def build_optimizer_v1(model_or_partition: Any) -> Any:
    torch, *_ = v25._tensor_core._runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV1)
        else partition_parameters_v1(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {"name": "encoder", "params": list(partition.encoder), "lr": 1.0e-4},
            {
                "name": "evidence_projection_semantic_roles",
                "params": list(partition.lift_semantic_roles),
                "lr": 3.0e-4,
            },
            {
                "name": "predictors",
                "params": list(
                    partition.predictor
                    + partition.place_predictor
                    + partition.local_predictor
                ),
                "lr": 3.0e-4,
            },
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-4,
    )
    validate_optimizer_v1(optimizer, partition)
    return optimizer


def validate_optimizer_v1(optimizer: Any, partition: ParameterPartitionV1) -> None:
    expected = (
        ("encoder", partition.encoder, 1.0e-4),
        (
            "evidence_projection_semantic_roles",
            partition.lift_semantic_roles,
            3.0e-4,
        ),
        (
            "predictors",
            partition.predictor
            + partition.place_predictor
            + partition.local_predictor,
            3.0e-4,
        ),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("memory-role optimizer must be one three-group AdamW")
    observed_ids: list[int] = []
    for observed, (name, parameters, learning_rate) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        values = tuple(observed["params"])
        observed_ids.extend(id(value) for value in values)
        if (
            observed.get("name") != name
            or tuple(map(id, values)) != tuple(map(id, parameters))
            or float(observed["lr"]) != learning_rate
            or tuple(observed["betas"]) != (0.9, 0.999)
            or float(observed["eps"]) != 1.0e-8
            or float(observed["weight_decay"]) != 1.0e-4
        ):
            raise RuntimeError(f"memory-role optimizer group {name!r} changed")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != {
        id(value) for value in partition.online
    }:
        raise RuntimeError("memory-role optimizer membership is incomplete or overlapping")


def validate_accounting_v1(accounting: JointTrainingAccountingV1) -> None:
    if not isinstance(accounting, JointTrainingAccountingV1):
        raise TypeError("memory-role accounting has the wrong type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("memory-role accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV1(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE_V1,
        physical_presentations=updates * PHYSICAL_PRESENTATIONS_PER_UPDATE_V1,
        local_presentations=updates * LOCAL_PRESENTATIONS_PER_UPDATE_V1,
        place_presentations=updates * PLACE_PRESENTATIONS_PER_UPDATE_V1,
        rgb_decodes=updates * RGB_DECODES_PER_UPDATE_V1,
        physical_rgb_decodes=updates * PHYSICAL_RGB_DECODES_PER_UPDATE_V1,
        local_rgb_decodes=updates * LOCAL_RGB_DECODES_PER_UPDATE_V1,
        place_rgb_decodes=updates * PLACE_RGB_DECODES_PER_UPDATE_V1,
        online_rgb_encodings=updates * ONLINE_RGB_ENCODINGS_PER_UPDATE_V1,
        ema_target_rgb_encodings=(
            updates * EMA_TARGET_RGB_ENCODINGS_PER_UPDATE_V1
        ),
        physical_microbatch_graphs=(
            updates * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        ),
        local_microbatch_graphs=updates * LOCAL_MICROBATCHES_PER_UPDATE_V1,
        place_microbatch_graphs=updates * PLACE_GRAPHS_PER_UPDATE_V3,
        autograd_grad_calls=updates
        * (
            3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
            + LOCAL_MICROBATCHES_PER_UPDATE_V1
            + PLACE_GRAPHS_PER_UPDATE_V3
        ),
        optimizer_steps=updates,
        ema_steps=updates,
    )
    if accounting != expected:
        raise RuntimeError("memory-role accounting is inconsistent")


def _advance_accounting_v1(
    accounting: JointTrainingAccountingV1,
) -> JointTrainingAccountingV1:
    result = JointTrainingAccountingV1(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE_V1,
        physical_presentations=(
            accounting.physical_presentations
            + PHYSICAL_PRESENTATIONS_PER_UPDATE_V1
        ),
        local_presentations=(
            accounting.local_presentations + LOCAL_PRESENTATIONS_PER_UPDATE_V1
        ),
        place_presentations=(
            accounting.place_presentations + PLACE_PRESENTATIONS_PER_UPDATE_V1
        ),
        rgb_decodes=accounting.rgb_decodes + RGB_DECODES_PER_UPDATE_V1,
        physical_rgb_decodes=(
            accounting.physical_rgb_decodes + PHYSICAL_RGB_DECODES_PER_UPDATE_V1
        ),
        local_rgb_decodes=(
            accounting.local_rgb_decodes + LOCAL_RGB_DECODES_PER_UPDATE_V1
        ),
        place_rgb_decodes=(
            accounting.place_rgb_decodes + PLACE_RGB_DECODES_PER_UPDATE_V1
        ),
        online_rgb_encodings=(
            accounting.online_rgb_encodings + ONLINE_RGB_ENCODINGS_PER_UPDATE_V1
        ),
        ema_target_rgb_encodings=(
            accounting.ema_target_rgb_encodings
            + EMA_TARGET_RGB_ENCODINGS_PER_UPDATE_V1
        ),
        physical_microbatch_graphs=(
            accounting.physical_microbatch_graphs
            + PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        ),
        local_microbatch_graphs=(
            accounting.local_microbatch_graphs
            + LOCAL_MICROBATCHES_PER_UPDATE_V1
        ),
        place_microbatch_graphs=(
            accounting.place_microbatch_graphs
            + PLACE_GRAPHS_PER_UPDATE_V3
        ),
        autograd_grad_calls=(
            accounting.autograd_grad_calls
            + 3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
            + LOCAL_MICROBATCHES_PER_UPDATE_V1
            + PLACE_GRAPHS_PER_UPDATE_V3
        ),
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
    )
    validate_accounting_v1(result)
    return result


def _validate_capacity_v1(accounting: JointTrainingAccountingV1) -> None:
    validate_accounting_v1(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES_V1
        or accounting.presentations + PRESENTATIONS_PER_UPDATE_V1
        > MAXIMUM_PRESENTATIONS_V1
    ):
        raise PermissionError("memory-role training cap leaves no complete update")


def _validate_rgb(torch: Any, value: Any, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (MICROBATCH_SIZE_V1, 3, 112, 112)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be finite float32 with shape (4,3,112,112)")


def _validate_role_microbatches_v1(
    torch: Any,
    local_batches: Sequence[Mapping[str, Any]],
    place_batches: Sequence[Mapping[str, Any]],
) -> None:
    if len(local_batches) != LOCAL_MICROBATCHES_PER_UPDATE_V1:
        raise ValueError("memory-role update requires exactly two local microbatches")
    if len(place_batches) != PLACE_MICROBATCHES_PER_UPDATE_V1:
        raise ValueError("memory-role update requires exactly two place microbatches")
    for batch in local_batches:
        if tuple(batch) != REQUIRED_LOCAL_BATCH_KEYS_V1:
            raise ValueError("memory-role local batch keys changed")
        current = batch[LOCAL_CURRENT_RGB_KEY_V1]
        next_rgb = batch[LOCAL_NEXT_RGB_KEY_V1]
        action = batch[LOCAL_ACTION_KEY_V1]
        _validate_rgb(torch, current, LOCAL_CURRENT_RGB_KEY_V1)
        _validate_rgb(torch, next_rgb, LOCAL_NEXT_RGB_KEY_V1)
        if current.device != next_rgb.device:
            raise ValueError("memory-role local RGB tensors use different devices")
        if (
            not isinstance(action, torch.Tensor)
            or tuple(action.shape) != (MICROBATCH_SIZE_V1,)
            or action.is_floating_point()
            or action.dtype == torch.bool
            or action.device != current.device
            or bool(((action < 0) | (action >= ACTION_COUNT_V1)).any())
        ):
            raise ValueError("memory-role local action must be integer shape (4,) in [0,9)")
    for batch in place_batches:
        if tuple(batch) != REQUIRED_PLACE_BATCH_KEYS_V1:
            raise ValueError("memory-role place batch keys changed")
        values = tuple(batch[key] for key in REQUIRED_PLACE_BATCH_KEYS_V1)
        for key, value in zip(REQUIRED_PLACE_BATCH_KEYS_V1, values, strict=True):
            _validate_rgb(torch, value, key)
        if len({value.device for value in values}) != 1:
            raise ValueError("memory-role place RGB tensors use different devices")


def _energy_per_row_v1(torch: Any, predicted: Any, target: Any) -> Any:
    if (
        not isinstance(predicted, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or predicted.shape != target.shape
        or predicted.ndim < 2
        or predicted.shape[0] != MICROBATCH_SIZE_V1
        or predicted.dtype != torch.float32
        or target.dtype != torch.float32
        or predicted.device != target.device
        or target.requires_grad
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise RuntimeError("memory-role energy operands are invalid")
    return (predicted - target).square().mean(
        dim=tuple(range(1, predicted.ndim))
    )


def _place_energy_per_row_v1(torch: Any, predicted: Any, target: Any) -> Any:
    if (
        not isinstance(predicted, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or tuple(predicted.shape) != (
            PLACE_GRAPH_BATCH_SIZE_V3,
            PLACE_KEY_DIMENSION_V3,
        )
        or predicted.shape != target.shape
        or predicted.dtype != torch.float32
        or target.dtype != torch.float32
        or predicted.device != target.device
        or target.requires_grad
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise RuntimeError("memory-role place-energy operands are invalid")
    return 1.0 - torch.nn.functional.cosine_similarity(
        predicted, target, dim=1, eps=1.0e-6
    )


def place_objective_v3(
    torch: Any,
    online_anchor_keys: Any,
    predictions: Any,
    positive_targets: Any,
    negative_targets: Any,
) -> PlaceObjectiveTermsV3:
    """Compute the exact preregistered V3 eight-row place objective."""

    expected_shape = (PLACE_GRAPH_BATCH_SIZE_V3, PLACE_KEY_DIMENSION_V3)
    values = (
        online_anchor_keys,
        predictions,
        positive_targets,
        negative_targets,
    )
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != expected_shape
        or value.dtype != torch.float32
        or value.device != online_anchor_keys.device
        or not bool(torch.isfinite(value).all())
        for value in values
    ):
        raise RuntimeError("V3 place-objective operands are invalid")
    if (
        not online_anchor_keys.requires_grad
        or not predictions.requires_grad
        or positive_targets.requires_grad
        or negative_targets.requires_grad
    ):
        raise RuntimeError("V3 place-objective gradient topology changed")

    positive_energy = _place_energy_per_row_v1(
        torch, predictions, positive_targets
    )
    negative_energy = _place_energy_per_row_v1(
        torch, predictions, negative_targets
    )
    alignment = positive_energy.mean()

    candidates = torch.cat((positive_targets, negative_targets), dim=0)
    logits = (
        predictions @ candidates.transpose(0, 1)
    ) / PLACE_CONTRAST_TEMPERATURE_V3
    labels = torch.arange(
        PLACE_GRAPH_BATCH_SIZE_V3,
        dtype=torch.long,
        device=predictions.device,
    )
    contrast = torch.nn.functional.cross_entropy(logits, labels)

    centered = online_anchor_keys - online_anchor_keys.mean(
        dim=0, keepdim=True
    )
    covariance_matrix = (
        centered.transpose(0, 1) @ centered
    ) / float(PLACE_GRAPH_BATCH_SIZE_V3 - 1)
    standard_deviation = torch.sqrt(
        covariance_matrix.diagonal() + PLACE_VARIANCE_EPSILON_V3
    )
    variance = torch.relu(
        PLACE_VARIANCE_FLOOR_V3 - standard_deviation
    ).mean()
    off_diagonal_mask = ~torch.eye(
        PLACE_KEY_DIMENSION_V3,
        dtype=torch.bool,
        device=online_anchor_keys.device,
    )
    covariance = covariance_matrix.square().masked_select(
        off_diagonal_mask
    ).sum() / float(PLACE_KEY_DIMENSION_V3)
    loss = (
        alignment
        + contrast
        + variance
        + PLACE_COVARIANCE_WEIGHT_V3 * covariance
    )
    for name, value in (
        ("alignment", alignment),
        ("contrast", contrast),
        ("variance", variance),
        ("covariance", covariance),
        ("loss", loss),
    ):
        v25._tensor_core._finite_tensor(torch, value, f"V3 place {name}")
    if not loss.requires_grad:
        raise RuntimeError("V3 place objective must retain a gradient graph")
    return PlaceObjectiveTermsV3(
        loss=loss,
        alignment=alignment,
        contrast=contrast,
        variance=variance,
        covariance=covariance,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
        logits=logits,
    )


def _target_roles_v1(model: Any, rgb: Any) -> Any:
    torch, *_ = v25._tensor_core._runtime_apis()
    with torch.no_grad():
        target = model.encode_target_roles(rgb)
    if (
        not hasattr(target, "place_key")
        or not hasattr(target, "local_control")
        or tuple(target.place_key.shape) != (rgb.shape[0], 64)
        or tuple(target.local_control.shape) != (rgb.shape[0], 32, 16, 16)
        or target.place_key.requires_grad
        or target.local_control.requires_grad
        or target.place_key.dtype != torch.float32
        or target.local_control.dtype != torch.float32
        or not bool(torch.isfinite(target.place_key).all())
        or not bool(torch.isfinite(target.local_control).all())
    ):
        raise RuntimeError("memory-role EMA target encoding is invalid")
    return target


def _row_values(value: Any) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())
    if not result or any(not math.isfinite(item) for item in result):
        raise FloatingPointError("memory-role diagnostic rows are empty or nonfinite")
    return result


def joint_training_update_v1(
    model: Any,
    optimizer: Any,
    physical_microbatches: Sequence[Mapping[str, Any]],
    local_microbatches: Sequence[Mapping[str, Any]],
    place_microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV1:
    """Accumulate all three routes, then perform one optimizer and EMA step."""

    torch, semantic_api, survival_api, *_ = v25._tensor_core._runtime_apis()
    state = JointTrainingAccountingV1() if accounting is None else accounting
    _validate_capacity_v1(state)
    v25._validate_microbatches_v25(torch, physical_microbatches)
    _validate_role_microbatches_v1(torch, local_microbatches, place_microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    physical_partition = partition.physical_view
    auxiliary_subset = (
        v25._v24.predictor_core_protected_survival_parameter_subset_v24(
            physical_partition
        )
    )
    auxiliary_ids = {id(value) for value in auxiliary_subset.parameters}
    protected_ids = {
        id(value)
        for value in auxiliary_subset.protected_predictor_core_parameters
    }
    role_ids = {
        id(value)
        for value in (
            partition.role_factorizer
            + partition.place_predictor
            + partition.local_predictor
        )
    }
    if auxiliary_ids & protected_ids or (auxiliary_ids | protected_ids) & role_ids:
        raise RuntimeError("memory-role parameters entered the inherited J24 route")
    if set(map(id, partition.place_recipients)) & set(
        map(id, partition.local_recipients)
    ) != set(map(id, partition.shared + partition.spatial_trunk)):
        raise RuntimeError("memory-role route overlap changed")

    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with memory-role accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("memory-role EMA target already has a gradient")

    optimizer.zero_grad(set_to_none=True)
    camera_shared = v25._tensor_core._zero_accumulators(partition.shared)
    joint_shared = v25._tensor_core._zero_accumulators(partition.shared)
    joint_representation = v25._tensor_core._zero_accumulators(
        partition.representation
    )
    joint_predictor = v25._tensor_core._zero_accumulators(partition.predictor)
    auxiliary_gradients = v25._tensor_core._zero_accumulators(
        auxiliary_subset.parameters
    )
    local_gradients = v25._tensor_core._zero_accumulators(
        partition.local_recipients
    )
    place_gradients = v25._tensor_core._zero_accumulators(
        partition.place_recipients
    )
    absent = {
        "camera_shared": 0,
        "joint_shared": 0,
        "representation": 0,
        "predictor": 0,
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: 0,
        LOCAL_ROUTE_NAME_V1: 0,
        PLACE_ROUTE_NAME_V1: 0,
    }
    sums = {name: 0.0 for name in ("S", "U", "R", "O", "N", "C", "J24", "L")}
    auxiliary_sums = {
        name: 0.0
        for name in (
            "positive_energy_sum",
            "scene_negative_energy_sum",
            "prior_negative_energy_sum",
            "scene_advantage_sum",
            "prior_advantage_sum",
            "scene_rank_sum",
            "prior_rank_sum",
        )
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    scene_count = prior_count = positive_count = 0

    for batch in physical_microbatches:
        current_encoding = model.encode_online_training(
            batch[v25.CURRENT_RGB_KEY],
            camera_origin_body_m=batch[v25.CURRENT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[v25.CURRENT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[v25.CURRENT_GROUND_PLANE_Z_KEY],
        )
        next_encoding = model.encode_online_training(
            batch[v25.NEXT_RGB_KEY],
            camera_origin_body_m=batch[v25.NEXT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[v25.NEXT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[v25.NEXT_GROUND_PLANE_Z_KEY],
        )
        current_logits = model.semantic_logits_from_latent(current_encoding.latent)
        next_logits = model.semantic_logits_from_latent(next_encoding.latent)
        semantic = semantic_api.semantic_loss_v1(
            current_logits,
            batch[v25.CURRENT_LABELS_KEY],
            next_logits,
            batch[v25.NEXT_LABELS_KEY],
        )
        occupied = v25._tensor_core._v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[v25.CURRENT_LABELS_KEY],
            next_logits,
            batch[v25.NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(current_encoding.latent)
        _, survival_logits = v25._tensor_core._v3._v2._v1._prediction_parts(
            prediction
        )
        auxiliary = v25._v24.predictor_core_protected_survival_objective_v24(
            torch,
            survival_api,
            survival_logits,
            batch[v25.PREFIX_LENGTHS_KEY],
            batch[v25.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
            batch[v25.ACTION_PRIOR_M_KEY_V23],
        )
        zero_temporal = 0.0 * semantic.loss
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=zero_temporal,
            survival_logits=survival_logits,
            immediate_feasible=batch[v25.IMMEDIATE_FEASIBLE_KEY],
            prefix_lengths=batch[v25.PREFIX_LENGTHS_KEY],
        )
        navigation = joint.loss + occupied.loss
        camera = v25._base.camera_evidence_pair_loss_v13(
            current_encoding.auxiliary_evidence,
            next_encoding.auxiliary_evidence,
            v25.CameraEvidenceFrameSupervisionV13(
                batch[v25.CURRENT_PIXEL_HIT_KEY],
                batch[v25.CURRENT_PIXEL_DISTANCE_KEY],
                batch[v25.CURRENT_GROUND_IN_FRUSTUM_KEY],
                batch[v25.CURRENT_GROUND_CLEAR_KEY],
            ),
            v25.CameraEvidenceFrameSupervisionV13(
                batch[v25.NEXT_PIXEL_HIT_KEY],
                batch[v25.NEXT_PIXEL_DISTANCE_KEY],
                batch[v25.NEXT_GROUND_IN_FRUSTUM_KEY],
                batch[v25.NEXT_GROUND_CLEAR_KEY],
            ),
        )
        for name, value in (
            ("physical navigation", navigation),
            ("camera evidence", camera.total),
            ("predictor-core-protected survival", auxiliary.loss),
        ):
            v25._tensor_core._finite_tensor(torch, value, name)
            if not value.requires_grad:
                raise RuntimeError(f"{name} must retain a gradient graph")

        c_gradients = torch.autograd.grad(
            camera.total / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        n_parameters = (
            partition.shared + partition.representation + partition.predictor
        )
        n_gradients = torch.autograd.grad(
            navigation / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            n_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        j_gradients = torch.autograd.grad(
            auxiliary.loss / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            auxiliary_subset.parameters,
            allow_unused=True,
        )
        shared_end = len(partition.shared)
        representation_end = shared_end + len(partition.representation)
        absent["camera_shared"] += v25._tensor_core._accumulate_gradients(
            camera_shared, c_gradients
        )
        absent["joint_shared"] += v25._tensor_core._accumulate_gradients(
            joint_shared, n_gradients[:shared_end]
        )
        absent["representation"] += v25._tensor_core._accumulate_gradients(
            joint_representation, n_gradients[shared_end:representation_end]
        )
        absent["predictor"] += v25._tensor_core._accumulate_gradients(
            joint_predictor, n_gradients[representation_end:]
        )
        route_name = v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
        absent[route_name] += v25._tensor_core._accumulate_gradients(
            auxiliary_gradients, j_gradients
        )
        values = {
            "S": joint.semantic,
            "U": joint.survival,
            "R": joint.progress_ranking,
            "O": occupied.loss,
            "N": navigation,
            "C": camera.total,
            "J24": auxiliary.loss,
            "L": navigation + camera.total + auxiliary.loss,
        }
        for name, value in values.items():
            sums[name] += v25._tensor_core._scalar(value)
        for name in auxiliary_sums:
            if name == "positive_energy_sum":
                value = auxiliary.positive_energy.sum()
            elif name == "scene_negative_energy_sum":
                value = auxiliary.scene_negative_energy[
                    auxiliary.scene_eligible
                ].sum()
            elif name == "prior_negative_energy_sum":
                value = auxiliary.prior_negative_energy[
                    auxiliary.prior_eligible
                ].sum()
            else:
                value = getattr(auxiliary, name)
            auxiliary_sums[name] += v25._tensor_core._scalar(value)
        positive_count += MICROBATCH_SIZE_V1 * v25.NON_HOLD_ACTION_COUNT_V23
        scene_count += auxiliary.scene_eligible_count
        prior_count += auxiliary.prior_eligible_count
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    local_loss_sum = 0.0
    local_correct_rows: list[float] = []
    local_wrong_rows: list[float] = []
    local_margin_rows: list[float] = []
    for batch in local_microbatches:
        current = model.encode_online_roles(batch[LOCAL_CURRENT_RGB_KEY_V1])
        action = batch[LOCAL_ACTION_KEY_V1].to(dtype=torch.long)
        wrong_action = (action + 1) % ACTION_COUNT_V1
        action_one_hot = torch.nn.functional.one_hot(
            action, num_classes=ACTION_COUNT_V1
        ).to(dtype=torch.float32)
        wrong_one_hot = torch.nn.functional.one_hot(
            wrong_action, num_classes=ACTION_COUNT_V1
        ).to(dtype=torch.float32)
        correct_prediction = model.local_predictor(
            current.local_control, action_one_hot
        )
        wrong_prediction = model.local_predictor(
            current.local_control, wrong_one_hot
        )
        target = _target_roles_v1(
            model, batch[LOCAL_NEXT_RGB_KEY_V1]
        ).local_control
        correct_energy = _energy_per_row_v1(torch, correct_prediction, target)
        wrong_energy = _energy_per_row_v1(torch, wrong_prediction, target)
        margin = torch.relu(
            LOCAL_WRONG_ACTION_MARGIN_V1 + correct_energy - wrong_energy
        )
        local_loss = correct_energy.mean() + margin.mean()
        v25._tensor_core._finite_tensor(torch, local_loss, "local role loss")
        if not local_loss.requires_grad:
            raise RuntimeError("local role loss must retain a gradient graph")
        gradients = torch.autograd.grad(
            local_loss / LOCAL_MICROBATCHES_PER_UPDATE_V1,
            partition.local_recipients,
            allow_unused=True,
        )
        absent[LOCAL_ROUTE_NAME_V1] += v25._tensor_core._accumulate_gradients(
            local_gradients, gradients
        )
        local_loss_sum += v25._tensor_core._scalar(local_loss)
        local_correct_rows.extend(_row_values(correct_energy))
        local_wrong_rows.extend(_row_values(wrong_energy))
        local_margin_rows.extend(_row_values(margin))

    anchor_rgb = torch.cat(
        tuple(batch[PLACE_ANCHOR_RGB_KEY_V1] for batch in place_microbatches),
        dim=0,
    )
    positive_rgb = torch.cat(
        tuple(batch[PLACE_POSITIVE_RGB_KEY_V1] for batch in place_microbatches),
        dim=0,
    )
    negative_rgb = torch.cat(
        tuple(batch[PLACE_NEGATIVE_RGB_KEY_V1] for batch in place_microbatches),
        dim=0,
    )
    anchor = model.encode_online_roles(anchor_rgb)
    predicted = model.place_predictor(anchor.place_key)
    targets = _target_roles_v1(
        model, torch.cat((positive_rgb, negative_rgb), dim=0)
    ).place_key
    positive_target, negative_target = targets.split(PLACE_GRAPH_BATCH_SIZE_V3)
    place_terms = place_objective_v3(
        torch,
        anchor.place_key,
        predicted,
        positive_target,
        negative_target,
    )
    gradients = torch.autograd.grad(
        place_terms.loss,
        partition.place_recipients,
        allow_unused=True,
    )
    absent[PLACE_ROUTE_NAME_V1] += v25._tensor_core._accumulate_gradients(
        place_gradients, gradients
    )
    place_positive_rows = list(_row_values(place_terms.positive_energy))
    place_negative_rows = list(_row_values(place_terms.negative_energy))

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (partition.representation, joint_representation),
        "predictor": (partition.predictor, joint_predictor),
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: (
            auxiliary_subset.parameters,
            auxiliary_gradients,
        ),
        LOCAL_ROUTE_NAME_V1: (partition.local_recipients, local_gradients),
        PLACE_ROUTE_NAME_V1: (partition.place_recipients, place_gradients),
    }
    route_values = {
        name: v25._tensor_core._route_norm_and_scale_v13(torch, gradients)
        for name, (_, gradients) in route_tensors.items()
    }
    for name in (
        "camera_shared",
        "joint_shared",
        "predictor",
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25,
        LOCAL_ROUTE_NAME_V1,
        PLACE_ROUTE_NAME_V1,
    ):
        if not (v25._tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required memory-role route {name!r} is zero")
    if absent[v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25] != 0:
        raise RuntimeError("inherited J24 route has an absent gradient")
    if absent[LOCAL_ROUTE_NAME_V1] != 0:
        raise RuntimeError("local role route has an absent gradient")
    if absent[PLACE_ROUTE_NAME_V1] != 0:
        raise RuntimeError("place role route has an absent gradient")

    scales = {name: values[1] for name, values in route_values.items()}
    auxiliary_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            auxiliary_subset.parameters, auxiliary_gradients, strict=True
        )
    }
    local_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            partition.local_recipients, local_gradients, strict=True
        )
    }
    place_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            partition.place_recipients, place_gradients, strict=True
        )
    }

    def add_role_gradients(parameter: Any, gradient: Any) -> Any:
        if id(parameter) in local_by_id:
            gradient = gradient + scales[LOCAL_ROUTE_NAME_V1] * local_by_id[
                id(parameter)
            ]
        if id(parameter) in place_by_id:
            gradient = gradient + scales[PLACE_ROUTE_NAME_V1] * place_by_id[
                id(parameter)
            ]
        return gradient

    for parameter, c_gradient, n_gradient in zip(
        partition.shared, camera_shared, joint_shared, strict=True
    ):
        gradient = (
            scales["camera_shared"] * c_gradient
            + scales["joint_shared"] * n_gradient
        )
        if id(parameter) in auxiliary_by_id:
            gradient = gradient + scales[
                v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
            ] * auxiliary_by_id[id(parameter)]
        parameter.grad = add_role_gradients(parameter, gradient)
    for parameter, inherited_gradient in zip(
        partition.representation, joint_representation, strict=True
    ):
        gradient = scales["representation"] * inherited_gradient
        if id(parameter) in auxiliary_by_id:
            gradient = gradient + scales[
                v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
            ] * auxiliary_by_id[id(parameter)]
        parameter.grad = add_role_gradients(parameter, gradient)
    for name, parameter, inherited_gradient in zip(
        partition.names["predictor"],
        partition.predictor,
        joint_predictor,
        strict=True,
    ):
        gradient = scales["predictor"] * inherited_gradient
        if name in v25.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24:
            if id(parameter) not in auxiliary_by_id:
                raise RuntimeError("swept-progress output left inherited J24")
            gradient = gradient + scales[
                v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
            ] * auxiliary_by_id[id(parameter)]
        elif id(parameter) in auxiliary_by_id:
            raise RuntimeError("protected old predictor core entered inherited J24")
        parameter.grad = gradient
    for parameter in partition.role_factorizer:
        if id(parameter) in local_by_id and id(parameter) in place_by_id:
            raise RuntimeError("a role-specific factorizer parameter entered both routes")
        if id(parameter) in local_by_id:
            parameter.grad = scales[LOCAL_ROUTE_NAME_V1] * local_by_id[id(parameter)]
        elif id(parameter) in place_by_id:
            parameter.grad = scales[PLACE_ROUTE_NAME_V1] * place_by_id[id(parameter)]
        else:
            raise RuntimeError("role factorizer parameter left both role routes")
    for parameter in partition.local_predictor:
        if id(parameter) not in local_by_id or id(parameter) in place_by_id:
            raise RuntimeError("local predictor route binding changed")
        parameter.grad = scales[LOCAL_ROUTE_NAME_V1] * local_by_id[id(parameter)]
    for parameter in partition.place_predictor:
        if id(parameter) not in place_by_id or id(parameter) in local_by_id:
            raise RuntimeError("place predictor route binding changed")
        parameter.grad = scales[PLACE_ROUTE_NAME_V1] * place_by_id[id(parameter)]

    if any(parameter.grad is None for parameter in partition.online):
        raise RuntimeError("an online memory-role parameter has no final gradient")
    for parameter in partition.online:
        v25._tensor_core._finite_tensor(
            torch, parameter.grad, "memory-role final gradient"
        )
    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("memory-role EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        v25._tensor_core._finite_tensor(torch, parameter, "memory-role online parameter")
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("memory-role EMA did not advance exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("memory-role EMA target received a gradient after EMA")
    advanced = _advance_accounting_v1(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("memory-role post-update EMA accounting differs")

    receipts = {
        name: v25._tensor_core._receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    if (
        len(local_correct_rows) != LOCAL_PRESENTATIONS_PER_UPDATE_V1
        or len(local_wrong_rows) != LOCAL_PRESENTATIONS_PER_UPDATE_V1
        or len(place_positive_rows) != PLACE_PRESENTATIONS_PER_UPDATE_V1
        or len(place_negative_rows) != PLACE_PRESENTATIONS_PER_UPDATE_V1
    ):
        raise RuntimeError("memory-role diagnostic row accounting changed")
    mean_losses = {
        name: value / PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        for name, value in sums.items()
    }
    mean_losses["local"] = local_loss_sum / LOCAL_MICROBATCHES_PER_UPDATE_V1
    mean_losses["place"] = v25._tensor_core._scalar(place_terms.loss)
    mean_losses["total"] = (
        mean_losses["L"] + mean_losses["local"] + mean_losses["place"]
    )

    def summary(values: Sequence[float]) -> Mapping[str, float | int]:
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    local_advantage = tuple(
        wrong - correct
        for correct, wrong in zip(local_correct_rows, local_wrong_rows, strict=True)
    )
    place_advantage = tuple(
        negative - positive
        for positive, negative in zip(
            place_positive_rows, place_negative_rows, strict=True
        )
    )
    return JointUpdateResultV1(
        accounting=advanced,
        mean_losses=mean_losses,
        gradient_routes=receipts,
        local_diagnostics={
            "mechanism": LOCAL_ROUTE_NAME_V1,
            "correct_energy_per_row": tuple(local_correct_rows),
            "wrong_energy_per_row": tuple(local_wrong_rows),
            "wrong_minus_correct_per_row": local_advantage,
            "margin_loss_per_row": tuple(local_margin_rows),
            "correct_energy": summary(local_correct_rows),
            "wrong_energy": summary(local_wrong_rows),
            "wrong_minus_correct": summary(local_advantage),
            "cyclic_wrong_action_margin": LOCAL_WRONG_ACTION_MARGIN_V1,
        },
        place_diagnostics={
            "mechanism": PLACE_ROUTE_NAME_V1,
            "objective_version": 3,
            "positive_energy_per_row": tuple(place_positive_rows),
            "negative_energy_per_row": tuple(place_negative_rows),
            "negative_minus_positive_per_row": place_advantage,
            "positive_energy": summary(place_positive_rows),
            "negative_energy": summary(place_negative_rows),
            "negative_minus_positive": summary(place_advantage),
            "alignment": v25._tensor_core._scalar(place_terms.alignment),
            "contrast": v25._tensor_core._scalar(place_terms.contrast),
            "variance": v25._tensor_core._scalar(place_terms.variance),
            "covariance": v25._tensor_core._scalar(place_terms.covariance),
            "candidate_count": 2 * PLACE_GRAPH_BATCH_SIZE_V3,
            "positive_candidate_count": PLACE_GRAPH_BATCH_SIZE_V3,
            "negative_candidate_count": PLACE_GRAPH_BATCH_SIZE_V3,
            "contrast_temperature": PLACE_CONTRAST_TEMPERATURE_V3,
            "variance_floor": PLACE_VARIANCE_FLOOR_V3,
            "variance_epsilon": PLACE_VARIANCE_EPSILON_V3,
            "covariance_weight": PLACE_COVARIANCE_WEIGHT_V3,
        },
        predictor_core_protected_survival_diagnostics={
            **auxiliary_sums,
            "positive_energy_count": positive_count,
            "positive_energy_mean": auxiliary_sums["positive_energy_sum"]
            / positive_count,
            "scene_eligible_count": scene_count,
            "scene_negative_energy_mean": auxiliary_sums[
                "scene_negative_energy_sum"
            ]
            / scene_count,
            "scene_advantage_mean": auxiliary_sums["scene_advantage_sum"]
            / scene_count,
            "prior_eligible_count": prior_count,
            "prior_negative_energy_mean": auxiliary_sums[
                "prior_negative_energy_sum"
            ]
            / prior_count,
            "prior_advantage_mean": auxiliary_sums["prior_advantage_sum"]
            / prior_count,
            "non_hold_action_count_per_row": v25.NON_HOLD_ACTION_COUNT_V23,
        },
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


# Runtime initializer aliases: the controller supplies all three routes itself.
partition_parameters_v13 = partition_parameters_v1
build_frozen_optimizer_v13 = build_optimizer_v1


__all__ = [
    "ACTION_COUNT_V1",
    "EMA_TARGET_RGB_ENCODINGS_PER_UPDATE_V1",
    "JointTrainingAccountingV1",
    "JointUpdateResultV1",
    "LOCAL_ACTION_KEY_V1",
    "LOCAL_CURRENT_RGB_KEY_V1",
    "LOCAL_NEXT_RGB_KEY_V1",
    "LOCAL_ROUTE_NAME_V1",
    "MAXIMUM_PRESENTATIONS_V1",
    "MAXIMUM_UPDATES_V1",
    "MICROBATCH_SIZE_V1",
    "ONLINE_RGB_ENCODINGS_PER_UPDATE_V1",
    "PLACE_ANCHOR_RGB_KEY_V1",
    "PLACE_CONTRAST_TEMPERATURE_V3",
    "PLACE_COVARIANCE_WEIGHT_V3",
    "PLACE_GRAPH_BATCH_SIZE_V3",
    "PLACE_GRAPHS_PER_UPDATE_V3",
    "PLACE_KEY_DIMENSION_V3",
    "PLACE_NEGATIVE_RGB_KEY_V1",
    "PLACE_POSITIVE_RGB_KEY_V1",
    "PLACE_ROUTE_NAME_V1",
    "PLACE_VARIANCE_EPSILON_V3",
    "PLACE_VARIANCE_FLOOR_V3",
    "PRESENTATIONS_PER_UPDATE_V1",
    "ParameterPartitionV1",
    "PlaceObjectiveTermsV3",
    "REQUIRED_LOCAL_BATCH_KEYS_V1",
    "REQUIRED_PLACE_BATCH_KEYS_V1",
    "RGB_DECODES_PER_UPDATE_V1",
    "build_frozen_optimizer_v13",
    "build_optimizer_v1",
    "joint_training_update_v1",
    "place_objective_v3",
    "partition_parameters_v1",
    "partition_parameters_v13",
    "validate_accounting_v1",
    "validate_optimizer_v1",
]
