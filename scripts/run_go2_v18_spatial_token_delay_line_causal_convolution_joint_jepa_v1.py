#!/usr/bin/env python3
"""Lean tensor-training core for the V18 spatial delay-line joint-JEPA.

This module owns no data or execution lifecycle.  It combines the frozen
physical route with eight B2 causal-memory graphs, then performs exactly one
optimizer step and one EMA update.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from scripts import (
    run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25 as v25,
)


PHYSICAL_MICROBATCH_SIZE_V1 = 4
PHYSICAL_MICROBATCHES_PER_UPDATE_V1 = 2
PHYSICAL_PRESENTATIONS_PER_UPDATE_V1 = 8
MEMORY_MICROBATCH_SIZE_V1 = 2
MEMORY_MICROBATCHES_PER_UPDATE_V1 = 8
MEMORY_PRESENTATIONS_PER_UPDATE_V1 = 16
PRESENTATIONS_PER_UPDATE_V1 = 24
MAXIMUM_UPDATES_V1 = 1_000
MAXIMUM_PRESENTATIONS_V1 = 24_000
FULL_MEMORY_LOSS_WEIGHT_V1 = 1.0
MASKED_MEMORY_LOSS_WEIGHT_V1 = 0.5
MEMORY_ROUTE_NAME_V1 = "spatial_token_delay_line_causal_convolution"

MEMORY_HISTORY_RGB_KEY_V1 = "history_rgb"
MEMORY_HISTORY_ACTIONS_KEY_V1 = "history_actions"
MEMORY_FUTURE_RGB_KEY_V1 = "future_rgb"
MEMORY_FUTURE_ACTIONS_KEY_V1 = "future_actions"
REQUIRED_MEMORY_BATCH_KEYS_V1 = (
    MEMORY_HISTORY_RGB_KEY_V1,
    MEMORY_HISTORY_ACTIONS_KEY_V1,
    MEMORY_FUTURE_RGB_KEY_V1,
    MEMORY_FUTURE_ACTIONS_KEY_V1,
)

# Compatibility names consumed by the inherited physical batch builder and
# runtime initializer.  The memory schema above remains a separate route.
CURRENT_RGB_KEY = v25.CURRENT_RGB_KEY
NEXT_RGB_KEY = v25.NEXT_RGB_KEY
CURRENT_LABELS_KEY = v25.CURRENT_LABELS_KEY
NEXT_LABELS_KEY = v25.NEXT_LABELS_KEY
IMMEDIATE_FEASIBLE_KEY = v25.IMMEDIATE_FEASIBLE_KEY
PREFIX_LENGTHS_KEY = v25.PREFIX_LENGTHS_KEY
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
ACTION_PRIOR_M_KEY_V23 = v25.ACTION_PRIOR_M_KEY_V23
REQUIRED_BATCH_KEYS = v25.REQUIRED_BATCH_KEYS
REQUIRED_BATCH_KEYS_V21 = v25.REQUIRED_BATCH_KEYS_V21
REQUIRED_BATCH_KEYS_V23 = v25.REQUIRED_BATCH_KEYS_V23
REQUIRED_BATCH_KEYS_V24 = v25.REQUIRED_BATCH_KEYS_V24
REQUIRED_BATCH_KEYS_V25 = v25.REQUIRED_BATCH_KEYS_V25
PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25 = (
    v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
)


@dataclass(frozen=True)
class ParameterPartitionV1:
    encoder: tuple[Any, ...]
    evidence_head: tuple[Any, ...]
    representation: tuple[Any, ...]
    predictor: tuple[Any, ...]
    memory_predictor: tuple[Any, ...]
    target: tuple[Any, ...]
    frozen_diagnostics: tuple[Any, ...]
    names: Mapping[str, tuple[str, ...]]

    @property
    def shared(self) -> tuple[Any, ...]:
        return self.encoder + self.evidence_head

    @property
    def lift_semantic(self) -> tuple[Any, ...]:
        return self.evidence_head + self.representation

    @property
    def online(self) -> tuple[Any, ...]:
        return (
            self.shared
            + self.representation
            + self.predictor
            + self.memory_predictor
        )

    @property
    def memory_recipients(self) -> tuple[Any, ...]:
        spatial = tuple(
            parameter
            for name, parameter in zip(
                self.names["representation"],
                self.representation,
                strict=True,
            )
            if name.startswith(
                ("bev_lift.point_projection.", "bev_lift.volume_block.")
            )
        )
        return self.shared + spatial + self.memory_predictor

    @property
    def physical_view(self) -> Any:
        """Exact inherited V18 parameter view used by frozen J24 helpers."""

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


@dataclass(frozen=True)
class JointTrainingAccountingV1:
    updates: int = 0
    presentations: int = 0
    physical_presentations: int = 0
    memory_presentations: int = 0
    physical_microbatch_graphs: int = 0
    memory_microbatch_graphs: int = 0
    autograd_grad_calls: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0


@dataclass(frozen=True)
class JointUpdateResultV1:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    memory_diagnostics: Mapping[str, Any]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


def partition_parameters_v1(model: Any) -> ParameterPartitionV1:
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "evidence_head": [],
        "representation": [],
        "predictor": [],
        "memory_predictor": [],
        "target": [],
        "frozen_diagnostics": [],
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
        elif name.startswith("memory_predictor."):
            group = "memory_predictor"
        elif name.startswith("target_"):
            group = "target"
        elif name.startswith(
            ("role_factorizer.", "place_predictor.", "local_predictor.")
        ):
            group = "frozen_diagnostics"
        else:
            raise RuntimeError(f"unregistered delay-line model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)

    required = (
        "encoder",
        "evidence_head",
        "representation",
        "predictor",
        "memory_predictor",
        "target",
    )
    if any(not groups[name] for name in required):
        raise RuntimeError("delay-line parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if (
        len(identities) != len(set(identities))
        or set(identities) != {id(value) for value in model.parameters()}
    ):
        raise RuntimeError(
            "delay-line parameter partition is incomplete or overlapping"
        )
    if any(
        parameter.requires_grad
        for group in ("target", "frozen_diagnostics")
        for parameter in groups[group]
    ):
        raise RuntimeError("target and diagnostic-only parameters must be frozen")
    if any(
        not parameter.requires_grad or str(parameter.dtype) != "torch.float32"
        for group in (
            "encoder",
            "evidence_head",
            "representation",
            "predictor",
            "memory_predictor",
        )
        for parameter in groups[group]
    ):
        raise RuntimeError("every delay-line online parameter must be trainable float32")
    return ParameterPartitionV1(
        **{name: tuple(values) for name, values in groups.items()},
        names={name: tuple(values) for name, values in names.items()},
    )


def build_optimizer_v1(model_or_partition: Any) -> Any:
    torch, *_ = v25._tensor_core._runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV1)
        else partition_parameters_v1(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "encoder",
                "params": list(partition.encoder),
                "lr": 1.0e-4,
            },
            {
                "name": "evidence_projection_semantic",
                "params": list(partition.lift_semantic),
                "lr": 3.0e-4,
            },
            {
                "name": "physical_and_memory_predictors",
                "params": list(
                    partition.predictor + partition.memory_predictor
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


def validate_optimizer_v1(
    optimizer: Any,
    partition: ParameterPartitionV1,
) -> None:
    expected = (
        ("encoder", partition.encoder, 1.0e-4),
        (
            "evidence_projection_semantic",
            partition.lift_semantic,
            3.0e-4,
        ),
        (
            "physical_and_memory_predictors",
            partition.predictor + partition.memory_predictor,
            3.0e-4,
        ),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("delay-line optimizer must be one three-group AdamW")
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
            raise RuntimeError(f"delay-line optimizer group {name!r} changed")
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != {id(value) for value in partition.online}
    ):
        raise RuntimeError(
            "delay-line optimizer membership is incomplete or overlapping"
        )


def validate_accounting_v1(accounting: JointTrainingAccountingV1) -> None:
    if not isinstance(accounting, JointTrainingAccountingV1):
        raise TypeError("delay-line accounting has the wrong type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("delay-line accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV1(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE_V1,
        physical_presentations=(
            updates * PHYSICAL_PRESENTATIONS_PER_UPDATE_V1
        ),
        memory_presentations=updates * MEMORY_PRESENTATIONS_PER_UPDATE_V1,
        physical_microbatch_graphs=(
            updates * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        ),
        memory_microbatch_graphs=(
            updates * MEMORY_MICROBATCHES_PER_UPDATE_V1
        ),
        autograd_grad_calls=updates
        * (
            3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
            + MEMORY_MICROBATCHES_PER_UPDATE_V1
        ),
        optimizer_steps=updates,
        ema_steps=updates,
    )
    if accounting != expected:
        raise RuntimeError("delay-line accounting is inconsistent")


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
        memory_presentations=(
            accounting.memory_presentations
            + MEMORY_PRESENTATIONS_PER_UPDATE_V1
        ),
        physical_microbatch_graphs=(
            accounting.physical_microbatch_graphs
            + PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        ),
        memory_microbatch_graphs=(
            accounting.memory_microbatch_graphs
            + MEMORY_MICROBATCHES_PER_UPDATE_V1
        ),
        autograd_grad_calls=(
            accounting.autograd_grad_calls
            + 3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V1
            + MEMORY_MICROBATCHES_PER_UPDATE_V1
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
        raise PermissionError("delay-line training cap leaves no complete update")


def _validate_physical_microbatches_v1(
    torch: Any,
    batches: Sequence[Mapping[str, Any]],
) -> None:
    if len(batches) != PHYSICAL_MICROBATCHES_PER_UPDATE_V1:
        raise ValueError("delay-line update requires exactly two physical microbatches")
    # The frozen V25 validator is four-batch-accounting aware.  Repeating this
    # read-only validation view preserves every per-batch contract while the
    # actual training loop below consumes each supplied graph exactly once.
    values = tuple(batches)
    v25._validate_microbatches_v25(torch, values + values)


def _validate_memory_microbatches_v1(
    torch: Any,
    batches: Sequence[Mapping[str, Any]],
) -> None:
    if len(batches) != MEMORY_MICROBATCHES_PER_UPDATE_V1:
        raise ValueError("delay-line update requires exactly eight memory microbatches")
    for batch in batches:
        if tuple(batch) != REQUIRED_MEMORY_BATCH_KEYS_V1:
            raise ValueError("delay-line memory batch keys changed")
        history = batch[MEMORY_HISTORY_RGB_KEY_V1]
        history_actions = batch[MEMORY_HISTORY_ACTIONS_KEY_V1]
        future = batch[MEMORY_FUTURE_RGB_KEY_V1]
        future_actions = batch[MEMORY_FUTURE_ACTIONS_KEY_V1]
        if (
            not isinstance(history, torch.Tensor)
            or tuple(history.shape) != (2, 3, 3, 112, 112)
            or history.dtype != torch.float32
            or not isinstance(future, torch.Tensor)
            or tuple(future.shape) != (2, 4, 3, 112, 112)
            or future.dtype != torch.float32
            or history.device != future.device
            or not bool(torch.isfinite(history).all())
            or not bool(torch.isfinite(future).all())
        ):
            raise ValueError("delay-line memory RGB tensors changed")
        for name, actions, shape in (
            ("history_actions", history_actions, (2, 2)),
            ("future_actions", future_actions, (2, 4)),
        ):
            if (
                not isinstance(actions, torch.Tensor)
                or tuple(actions.shape) != shape
                or actions.is_floating_point()
                or actions.dtype == torch.bool
                or actions.device != history.device
                or bool(((actions < 0) | (actions >= 9)).any())
            ):
                raise ValueError(
                    f"delay-line {name} must be an in-range integer tensor"
                )


def _validate_memory_output_v1(torch: Any, output: Any) -> None:
    for name in (
        "target_future_tokens",
        "full_predictions",
        "masked_current_predictions",
    ):
        value = getattr(output, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (2, 4, 64, 16, 16)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise RuntimeError(f"delay-line {name} contract changed")
    if output.target_future_tokens.requires_grad:
        raise RuntimeError("delay-line future target is not stop-gradient")
    keep = getattr(output, "newest_keep_mask", None)
    if (
        not isinstance(keep, torch.Tensor)
        or tuple(keep.shape) != (2, 1, 16, 16)
        or keep.dtype != torch.bool
        or not torch.equal(
            keep.sum(dim=(1, 2, 3)),
            torch.full((2,), 128, dtype=torch.long, device=keep.device),
        )
    ):
        raise RuntimeError("delay-line newest-token mask is not exact half")
    for name in ("full_loss", "masked_current_loss", "loss"):
        value = getattr(output, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value))
        ):
            raise RuntimeError(f"delay-line {name} must be one finite float32 scalar")
    expected = (
        FULL_MEMORY_LOSS_WEIGHT_V1 * output.full_loss
        + MASKED_MEMORY_LOSS_WEIGHT_V1 * output.masked_current_loss
    )
    if not bool(torch.allclose(output.loss, expected, rtol=0.0, atol=1.0e-7)):
        raise RuntimeError("delay-line full/masked objective weighting changed")


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise FloatingPointError("delay-line diagnostic scalar is nonfinite")
    return result


def joint_training_update_v1(
    model: Any,
    optimizer: Any,
    physical_microbatches: Sequence[Mapping[str, Any]],
    memory_microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV1:
    """Accumulate the two physical and eight memory graphs into one update."""

    torch, semantic_api, survival_api, *_ = v25._tensor_core._runtime_apis()
    state = JointTrainingAccountingV1() if accounting is None else accounting
    _validate_capacity_v1(state)
    _validate_physical_microbatches_v1(torch, physical_microbatches)
    _validate_memory_microbatches_v1(torch, memory_microbatches)
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
    memory_ids = {id(value) for value in partition.memory_predictor}
    if (
        auxiliary_ids & protected_ids
        or memory_ids & (auxiliary_ids | protected_ids)
    ):
        raise RuntimeError("delay-line memory parameters entered a physical route")

    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("delay-line model EMA disagrees with accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("delay-line target already has a gradient")

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
    memory_gradients = v25._tensor_core._zero_accumulators(
        partition.memory_recipients
    )
    absent = {
        "camera_shared": 0,
        "joint_shared": 0,
        "representation": 0,
        "predictor": 0,
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: 0,
        MEMORY_ROUTE_NAME_V1: 0,
    }
    physical_sums = {
        name: 0.0 for name in ("S", "U", "R", "O", "N", "C", "J24", "L")
    }
    ranking_active = ranking_pairs = supervised_decisions = 0

    for batch in physical_microbatches:
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
        current_logits = model.semantic_logits_from_latent(current_encoding.latent)
        next_logits = model.semantic_logits_from_latent(next_encoding.latent)
        semantic = semantic_api.semantic_loss_v1(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        occupied = v25._tensor_core._v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(
            current_encoding.latent
        )
        _, survival_logits = v25._tensor_core._v3._v2._v1._prediction_parts(
            prediction
        )
        auxiliary = (
            v25._v24.predictor_core_protected_survival_objective_v24(
                torch,
                survival_api,
                survival_logits,
                batch[PREFIX_LENGTHS_KEY],
                batch[SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
                batch[ACTION_PRIOR_M_KEY_V23],
            )
        )
        zero_temporal = 0.0 * semantic.loss
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=zero_temporal,
            survival_logits=survival_logits,
            immediate_feasible=batch[IMMEDIATE_FEASIBLE_KEY],
            prefix_lengths=batch[PREFIX_LENGTHS_KEY],
        )
        navigation = joint.loss + occupied.loss
        camera = v25._base.camera_evidence_pair_loss_v13(
            current_encoding.auxiliary_evidence,
            next_encoding.auxiliary_evidence,
            v25.CameraEvidenceFrameSupervisionV13(
                batch[CURRENT_PIXEL_HIT_KEY],
                batch[CURRENT_PIXEL_DISTANCE_KEY],
                batch[CURRENT_GROUND_IN_FRUSTUM_KEY],
                batch[CURRENT_GROUND_CLEAR_KEY],
            ),
            v25.CameraEvidenceFrameSupervisionV13(
                batch[NEXT_PIXEL_HIT_KEY],
                batch[NEXT_PIXEL_DISTANCE_KEY],
                batch[NEXT_GROUND_IN_FRUSTUM_KEY],
                batch[NEXT_GROUND_CLEAR_KEY],
            ),
        )
        for name, value in (
            ("physical navigation", navigation),
            ("camera evidence", camera.total),
            ("J24", auxiliary.loss),
        ):
            v25._tensor_core._finite_tensor(torch, value, name)

        camera_values = torch.autograd.grad(
            camera.total / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        navigation_parameters = (
            partition.shared + partition.representation + partition.predictor
        )
        navigation_values = torch.autograd.grad(
            navigation / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            navigation_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_values = torch.autograd.grad(
            auxiliary.loss / PHYSICAL_MICROBATCHES_PER_UPDATE_V1,
            auxiliary_subset.parameters,
            allow_unused=True,
        )
        shared_end = len(partition.shared)
        representation_end = shared_end + len(partition.representation)
        absent["camera_shared"] += v25._tensor_core._accumulate_gradients(
            camera_shared, camera_values
        )
        absent["joint_shared"] += v25._tensor_core._accumulate_gradients(
            joint_shared, navigation_values[:shared_end]
        )
        absent["representation"] += v25._tensor_core._accumulate_gradients(
            joint_representation,
            navigation_values[shared_end:representation_end],
        )
        absent["predictor"] += v25._tensor_core._accumulate_gradients(
            joint_predictor,
            navigation_values[representation_end:],
        )
        route_name = PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
        absent[route_name] += v25._tensor_core._accumulate_gradients(
            auxiliary_gradients, auxiliary_values
        )
        losses = {
            "S": joint.semantic,
            "U": joint.survival,
            "R": joint.progress_ranking,
            "O": occupied.loss,
            "N": navigation,
            "C": camera.total,
            "J24": auxiliary.loss,
            "L": navigation + camera.total + auxiliary.loss,
        }
        for name, value in losses.items():
            physical_sums[name] += _scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        ranking_active += int(pairs > 0)
        ranking_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    full_losses: list[float] = []
    masked_losses: list[float] = []
    combined_losses: list[float] = []
    keep_fractions: list[float] = []
    for batch in memory_microbatches:
        action_indices = torch.cat(
            (
                batch[MEMORY_HISTORY_ACTIONS_KEY_V1],
                batch[MEMORY_FUTURE_ACTIONS_KEY_V1],
            ),
            dim=1,
        )
        action_sequence = torch.nn.functional.one_hot(
            action_indices.to(dtype=torch.long),
            num_classes=9,
        ).to(dtype=batch[MEMORY_HISTORY_RGB_KEY_V1].dtype)
        output = model.forward_memory(
            batch[MEMORY_HISTORY_RGB_KEY_V1],
            action_sequence,
            batch[MEMORY_FUTURE_RGB_KEY_V1],
        )
        _validate_memory_output_v1(torch, output)
        values = torch.autograd.grad(
            output.loss / MEMORY_MICROBATCHES_PER_UPDATE_V1,
            partition.memory_recipients,
            allow_unused=True,
        )
        absent[MEMORY_ROUTE_NAME_V1] += (
            v25._tensor_core._accumulate_gradients(
                memory_gradients,
                values,
            )
        )
        full_losses.append(_scalar(output.full_loss))
        masked_losses.append(_scalar(output.masked_current_loss))
        combined_losses.append(_scalar(output.loss))
        keep_fractions.append(
            _scalar(output.newest_keep_mask.float().mean())
        )

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (
            partition.representation,
            joint_representation,
        ),
        "predictor": (partition.predictor, joint_predictor),
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: (
            auxiliary_subset.parameters,
            auxiliary_gradients,
        ),
        MEMORY_ROUTE_NAME_V1: (
            partition.memory_recipients,
            memory_gradients,
        ),
    }
    route_values = {
        name: v25._tensor_core._route_norm_and_scale_v13(torch, gradients)
        for name, (_, gradients) in route_tensors.items()
    }
    for name in (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25,
        MEMORY_ROUTE_NAME_V1,
    ):
        if not (_scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required delay-line route {name!r} is zero")
    if absent[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25] != 0:
        raise RuntimeError("delay-line J24 has an absent gradient")
    if absent[MEMORY_ROUTE_NAME_V1] != 0:
        raise RuntimeError("delay-line memory route has an absent gradient")

    route_gradient_maps = {
        name: {
            id(parameter): gradient
            for parameter, gradient in zip(
                parameters,
                gradients,
                strict=True,
            )
        }
        for name, (parameters, gradients) in route_tensors.items()
    }
    for parameter in partition.online:
        gradient = parameter.detach().new_zeros(parameter.shape)
        participated = False
        for name, (_, scale) in route_values.items():
            value = route_gradient_maps[name].get(id(parameter))
            if value is not None:
                gradient = gradient + scale * value
                participated = True
        if not participated:
            raise RuntimeError("an online parameter has no registered gradient route")
        parameter.grad = gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("delay-line target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        v25._tensor_core._finite_tensor(
            torch,
            parameter,
            "delay-line online parameter",
        )
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("delay-line EMA did not advance exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("delay-line target received a gradient after EMA")
    advanced = _advance_accounting_v1(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("delay-line EMA accounting differs after update")

    receipts = {
        name: v25._tensor_core._receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    physical_means = {
        name: value / PHYSICAL_MICROBATCHES_PER_UPDATE_V1
        for name, value in physical_sums.items()
    }
    memory_full = sum(full_losses) / len(full_losses)
    memory_masked = sum(masked_losses) / len(masked_losses)
    memory_combined = sum(combined_losses) / len(combined_losses)
    mean_losses = {
        **physical_means,
        "memory_full": memory_full,
        "memory_masked": memory_masked,
        "memory": memory_combined,
        "total": physical_means["L"] + memory_combined,
    }
    return JointUpdateResultV1(
        accounting=advanced,
        mean_losses=mean_losses,
        gradient_routes=receipts,
        memory_diagnostics={
            "mechanism": MEMORY_ROUTE_NAME_V1,
            "full_weight": FULL_MEMORY_LOSS_WEIGHT_V1,
            "masked_weight": MASKED_MEMORY_LOSS_WEIGHT_V1,
            "full_loss_per_microbatch": tuple(full_losses),
            "masked_loss_per_microbatch": tuple(masked_losses),
            "combined_loss_per_microbatch": tuple(combined_losses),
            "newest_keep_fraction": sum(keep_fractions) / len(keep_fractions),
            "future_online_access_count": 0,
        },
        ranking_active_microbatches=ranking_active,
        ranking_eligible_pairs=ranking_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


# Compatibility aliases consumed by the inherited V13 runtime initializer.
partition_parameters_v13 = partition_parameters_v1
build_frozen_optimizer_v13 = build_optimizer_v1


__all__ = [
    "FULL_MEMORY_LOSS_WEIGHT_V1",
    "JointTrainingAccountingV1",
    "JointUpdateResultV1",
    "MASKED_MEMORY_LOSS_WEIGHT_V1",
    "MAXIMUM_PRESENTATIONS_V1",
    "MAXIMUM_UPDATES_V1",
    "MEMORY_FUTURE_RGB_KEY_V1",
    "MEMORY_FUTURE_ACTIONS_KEY_V1",
    "MEMORY_HISTORY_ACTIONS_KEY_V1",
    "MEMORY_HISTORY_RGB_KEY_V1",
    "MEMORY_MICROBATCHES_PER_UPDATE_V1",
    "MEMORY_MICROBATCH_SIZE_V1",
    "MEMORY_PRESENTATIONS_PER_UPDATE_V1",
    "MEMORY_ROUTE_NAME_V1",
    "PHYSICAL_MICROBATCHES_PER_UPDATE_V1",
    "PHYSICAL_MICROBATCH_SIZE_V1",
    "PHYSICAL_PRESENTATIONS_PER_UPDATE_V1",
    "PRESENTATIONS_PER_UPDATE_V1",
    "ParameterPartitionV1",
    "REQUIRED_MEMORY_BATCH_KEYS_V1",
    "build_frozen_optimizer_v13",
    "build_optimizer_v1",
    "joint_training_update_v1",
    "partition_parameters_v1",
    "partition_parameters_v13",
    "validate_accounting_v1",
    "validate_optimizer_v1",
]
