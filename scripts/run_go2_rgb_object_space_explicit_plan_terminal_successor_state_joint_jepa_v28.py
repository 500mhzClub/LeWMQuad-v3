#!/usr/bin/env python3
"""Tensor core for the V28 explicit-plan terminal successor-state JEPA."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from scripts import (
    run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25 as v25,
)


MICROBATCH_SIZE_V28 = 4
PHYSICAL_MICROBATCHES_PER_UPDATE_V28 = 4
PLAN_MICROBATCHES_PER_UPDATE_V28 = 4
PHYSICAL_PRESENTATIONS_PER_UPDATE_V28 = 16
PLAN_PRESENTATIONS_PER_UPDATE_V28 = 16
PRESENTATIONS_PER_UPDATE_V28 = 32
MAXIMUM_UPDATES_V28 = 400
MAXIMUM_PRESENTATIONS_V28 = 12_800
PLAN_ROUTE_NAME_V28 = "explicit_plan_terminal_successor_state"
H6_CURRENT_RGB_KEY_V28 = "current_rgb"
H6_TERMINAL_RGB_KEY_V28 = "terminal_rgb"
H6_FUTURE_ACTIONS_KEY_V28 = "future_actions"
REQUIRED_H6_BATCH_KEYS_V28 = (
    H6_CURRENT_RGB_KEY_V28,
    H6_TERMINAL_RGB_KEY_V28,
    H6_FUTURE_ACTIONS_KEY_V28,
)

# Compatibility names used only by the inherited, already-reviewed physical
# batch builder and runtime initializer.  The new H6 route remains separate.
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
REQUIRED_BATCH_KEYS_V28 = REQUIRED_BATCH_KEYS_V25


@dataclass(frozen=True)
class ParameterPartitionV28:
    encoder: tuple[Any, ...]
    evidence_head: tuple[Any, ...]
    representation: tuple[Any, ...]
    predictor: tuple[Any, ...]
    plan: tuple[Any, ...]
    target: tuple[Any, ...]
    names: Mapping[str, tuple[str, ...]]

    @property
    def shared(self) -> tuple[Any, ...]:
        return self.encoder + self.evidence_head

    @property
    def lift_semantic(self) -> tuple[Any, ...]:
        return self.evidence_head + self.representation

    @property
    def online(self) -> tuple[Any, ...]:
        return self.shared + self.representation + self.predictor + self.plan

    @property
    def plan_recipients(self) -> tuple[Any, ...]:
        spatial = tuple(
            parameter
            for name, parameter in zip(
                self.names["representation"], self.representation, strict=True
            )
            if name.startswith(
                ("bev_lift.point_projection.", "bev_lift.volume_block.")
            )
        )
        return self.shared + spatial + self.plan

    @property
    def physical_view(self) -> Any:
        """Return the exact old-predictor view consumed by unchanged J24."""

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
class JointTrainingAccountingV28:
    updates: int = 0
    presentations: int = 0
    physical_presentations: int = 0
    plan_presentations: int = 0
    physical_microbatch_graphs: int = 0
    plan_microbatch_graphs: int = 0
    autograd_grad_calls: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0


@dataclass(frozen=True)
class JointUpdateResultV28:
    accounting: JointTrainingAccountingV28
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    plan_diagnostics: Mapping[str, float | int | tuple[float, ...]]
    predictor_core_protected_survival_diagnostics: Mapping[str, float | int]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


def partition_parameters_v28(model: Any) -> ParameterPartitionV28:
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "evidence_head": [],
        "representation": [],
        "predictor": [],
        "plan": [],
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
        elif name.startswith("plan_predictor."):
            group = "plan"
        elif name.startswith(
            (
                "target_encoder.",
                "target_bev_lift.evidence_head.",
                "target_bev_lift.point_projection.",
                "target_bev_lift.volume_block.",
            )
        ):
            group = "target"
        else:
            raise RuntimeError(f"unregistered V28 model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)
    if any(not values for values in groups.values()):
        raise RuntimeError("V28 parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if len(identities) != len(set(identities)) or set(identities) != {
        id(value) for value in model.parameters()
    }:
        raise RuntimeError("V28 parameter partition is incomplete or overlapping")
    if any(parameter.requires_grad for parameter in groups["target"]):
        raise RuntimeError("V28 EMA target parameter is trainable")
    if any(
        not parameter.requires_grad or str(parameter.dtype) != "torch.float32"
        for group in ("encoder", "evidence_head", "representation", "predictor", "plan")
        for parameter in groups[group]
    ):
        raise RuntimeError("every V28 online parameter must be trainable float32")
    partition = ParameterPartitionV28(
        **{name: tuple(values) for name, values in groups.items()},
        names={name: tuple(values) for name, values in names.items()},
    )
    # This call is intentionally strict: adding the plan role may not change
    # the inherited 109-tensor J24 physical view.
    v25._v24.predictor_core_protected_survival_parameter_subset_v24(
        partition.physical_view
    )
    return partition


def build_optimizer_v28(model_or_partition: Any) -> Any:
    torch, *_ = v25._tensor_core._runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV28)
        else partition_parameters_v28(model_or_partition)
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
                "name": "predictor",
                "params": list(partition.predictor + partition.plan),
                "lr": 3.0e-4,
            },
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-4,
    )
    validate_optimizer_v28(optimizer, partition)
    return optimizer


def validate_optimizer_v28(optimizer: Any, partition: ParameterPartitionV28) -> None:
    expected = (
        ("encoder", partition.encoder, 1.0e-4),
        ("evidence_projection_semantic", partition.lift_semantic, 3.0e-4),
        ("predictor", partition.predictor + partition.plan, 3.0e-4),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("V28 optimizer must be one three-group AdamW")
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
            raise RuntimeError(f"V28 optimizer group {name!r} changed")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != {
        id(value) for value in partition.online
    }:
        raise RuntimeError("V28 optimizer membership is incomplete or overlapping")


def validate_accounting_v28(accounting: JointTrainingAccountingV28) -> None:
    if not isinstance(accounting, JointTrainingAccountingV28):
        raise TypeError("V28 accounting has the wrong type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V28 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV28(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE_V28,
        physical_presentations=updates * PHYSICAL_PRESENTATIONS_PER_UPDATE_V28,
        plan_presentations=updates * PLAN_PRESENTATIONS_PER_UPDATE_V28,
        physical_microbatch_graphs=(
            updates * PHYSICAL_MICROBATCHES_PER_UPDATE_V28
        ),
        plan_microbatch_graphs=updates * PLAN_MICROBATCHES_PER_UPDATE_V28,
        autograd_grad_calls=updates
        * (
            3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V28
            + PLAN_MICROBATCHES_PER_UPDATE_V28
        ),
        optimizer_steps=updates,
        ema_steps=updates,
    )
    if accounting != expected:
        raise RuntimeError("V28 accounting is inconsistent")


def _advance_accounting_v28(
    accounting: JointTrainingAccountingV28,
) -> JointTrainingAccountingV28:
    result = JointTrainingAccountingV28(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE_V28,
        physical_presentations=(
            accounting.physical_presentations
            + PHYSICAL_PRESENTATIONS_PER_UPDATE_V28
        ),
        plan_presentations=(
            accounting.plan_presentations + PLAN_PRESENTATIONS_PER_UPDATE_V28
        ),
        physical_microbatch_graphs=(
            accounting.physical_microbatch_graphs
            + PHYSICAL_MICROBATCHES_PER_UPDATE_V28
        ),
        plan_microbatch_graphs=(
            accounting.plan_microbatch_graphs + PLAN_MICROBATCHES_PER_UPDATE_V28
        ),
        autograd_grad_calls=(
            accounting.autograd_grad_calls
            + 3 * PHYSICAL_MICROBATCHES_PER_UPDATE_V28
            + PLAN_MICROBATCHES_PER_UPDATE_V28
        ),
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
    )
    validate_accounting_v28(result)
    return result


def _validate_capacity_v28(accounting: JointTrainingAccountingV28) -> None:
    validate_accounting_v28(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES_V28
        or accounting.presentations + PRESENTATIONS_PER_UPDATE_V28
        > MAXIMUM_PRESENTATIONS_V28
    ):
        raise PermissionError("V28 training cap leaves no complete update")


def _validate_h6_microbatches_v28(torch: Any, batches: Sequence[Mapping[str, Any]]) -> None:
    if len(batches) != PLAN_MICROBATCHES_PER_UPDATE_V28:
        raise ValueError("V28 requires exactly four H6 microbatches")
    for batch in batches:
        if tuple(batch) != REQUIRED_H6_BATCH_KEYS_V28:
            raise ValueError("V28 H6 batch keys changed")
        current = batch[H6_CURRENT_RGB_KEY_V28]
        terminal = batch[H6_TERMINAL_RGB_KEY_V28]
        actions = batch[H6_FUTURE_ACTIONS_KEY_V28]
        if (
            not isinstance(current, torch.Tensor)
            or tuple(current.shape) != (4, 3, 112, 112)
            or current.dtype != torch.float32
            or not isinstance(terminal, torch.Tensor)
            or tuple(terminal.shape) != (4, 3, 112, 112)
            or terminal.dtype != torch.float32
            or current.device != terminal.device
            or not bool(torch.isfinite(current).all())
            or not bool(torch.isfinite(terminal).all())
        ):
            raise ValueError("V28 H6 RGB tensors changed")
        if (
            not isinstance(actions, torch.Tensor)
            or tuple(actions.shape) != (4, 4)
            or actions.is_floating_point()
            or actions.dtype == torch.bool
            or actions.device != current.device
            or bool(((actions < 0) | (actions >= 9)).any())
        ):
            raise ValueError("V28 H6 action tensor changed")


def terminal_successor_target_v28(model: Any, terminal_rgb: Any) -> Any:
    torch, *_ = v25._tensor_core._runtime_apis()
    if terminal_rgb.ndim != 4 or tuple(terminal_rgb.shape[1:]) != (3, 112, 112):
        raise ValueError("V28 terminal RGB must have shape (B,3,112,112)")
    with torch.no_grad():
        target = model.encode_target(terminal_rgb)
    if (
        tuple(target.shape) != (terminal_rgb.shape[0], 64, 64, 64)
        or target.requires_grad
        or not bool(torch.isfinite(target).all())
    ):
        raise RuntimeError("V28 terminal EMA target is invalid")
    return target.detach()


def _row_values(value: Any) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())
    if not result or any(not math.isfinite(item) for item in result):
        raise FloatingPointError("V28 diagnostic rows are empty or nonfinite")
    return result


def joint_training_update_v28(
    model: Any,
    optimizer: Any,
    physical_microbatches: Sequence[Mapping[str, Any]],
    plan_microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV28 | None = None,
) -> JointUpdateResultV28:
    """Combine the physical and plan JEPA routes before one step and EMA."""

    torch, semantic_api, survival_api, *_ = v25._tensor_core._runtime_apis()
    state = JointTrainingAccountingV28() if accounting is None else accounting
    _validate_capacity_v28(state)
    v25._validate_microbatches_v25(torch, physical_microbatches)
    _validate_h6_microbatches_v28(torch, plan_microbatches)
    partition = partition_parameters_v28(model)
    validate_optimizer_v28(optimizer, partition)
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
    if auxiliary_ids & protected_ids or any(
        id(value) in auxiliary_ids | protected_ids for value in partition.plan
    ):
        raise RuntimeError("V28 plan parameters entered J24")
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("V28 model EMA disagrees with accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V28 target already has a gradient")

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
    plan_gradients = v25._tensor_core._zero_accumulators(
        partition.plan_recipients
    )
    absent = {
        "camera_shared": 0,
        "joint_shared": 0,
        "representation": 0,
        "predictor": 0,
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: 0,
        PLAN_ROUTE_NAME_V28: 0,
    }
    sums = {name: 0.0 for name in ("S", "U", "R", "O", "N28", "C", "J24", "L")}
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
        zero_p = 0.0 * semantic.loss
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=zero_p,
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
            ("N28", navigation),
            ("Camera C", camera.total),
            ("J24", auxiliary.loss),
        ):
            v25._tensor_core._finite_tensor(torch, value, name)
        c_gradients = torch.autograd.grad(
            camera.total / PHYSICAL_MICROBATCHES_PER_UPDATE_V28,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        n_parameters = partition.shared + partition.representation + partition.predictor
        n_gradients = torch.autograd.grad(
            navigation / PHYSICAL_MICROBATCHES_PER_UPDATE_V28,
            n_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        j_gradients = torch.autograd.grad(
            auxiliary.loss / PHYSICAL_MICROBATCHES_PER_UPDATE_V28,
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
            "N28": navigation,
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
                value = auxiliary.scene_negative_energy[auxiliary.scene_eligible].sum()
            elif name == "prior_negative_energy_sum":
                value = auxiliary.prior_negative_energy[auxiliary.prior_eligible].sum()
            else:
                value = getattr(auxiliary, name)
            auxiliary_sums[name] += v25._tensor_core._scalar(value)
        positive_count += MICROBATCH_SIZE_V28 * v25.NON_HOLD_ACTION_COUNT_V23
        scene_count += auxiliary.scene_eligible_count
        prior_count += auxiliary.prior_eligible_count
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    plan_loss_sum = 0.0
    plan_energy_rows: list[float] = []
    for batch in plan_microbatches:
        current = model.encode_online(batch[H6_CURRENT_RGB_KEY_V28])
        predicted = model.predict_plan_successor(
            current, batch[H6_FUTURE_ACTIONS_KEY_V28]
        )
        target = terminal_successor_target_v28(
            model, batch[H6_TERMINAL_RGB_KEY_V28]
        )
        energy = semantic_api.latent_energy_per_row(predicted, target)
        plan_loss = energy.mean()
        v25._tensor_core._finite_tensor(torch, plan_loss, "V28 plan loss")
        gradients = torch.autograd.grad(
            plan_loss / PLAN_MICROBATCHES_PER_UPDATE_V28,
            partition.plan_recipients,
            allow_unused=True,
        )
        absent[PLAN_ROUTE_NAME_V28] += v25._tensor_core._accumulate_gradients(
            plan_gradients, gradients
        )
        plan_loss_sum += v25._tensor_core._scalar(plan_loss)
        plan_energy_rows.extend(_row_values(energy))

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (partition.representation, joint_representation),
        "predictor": (partition.predictor, joint_predictor),
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: (
            auxiliary_subset.parameters,
            auxiliary_gradients,
        ),
        PLAN_ROUTE_NAME_V28: (partition.plan_recipients, plan_gradients),
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
        PLAN_ROUTE_NAME_V28,
    ):
        if not (v25._tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V28 route {name!r} is zero")
    if absent[v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25] != 0:
        raise RuntimeError("V28 J24 has an absent gradient")
    if absent[PLAN_ROUTE_NAME_V28] != 0:
        raise RuntimeError("V28 plan route has an absent gradient")

    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    representation_scale = route_values["representation"][1]
    predictor_scale = route_values["predictor"][1]
    auxiliary_scale = route_values[
        v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
    ][1]
    plan_scale = route_values[PLAN_ROUTE_NAME_V28][1]
    auxiliary_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            auxiliary_subset.parameters, auxiliary_gradients, strict=True
        )
    }
    plan_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            partition.plan_recipients, plan_gradients, strict=True
        )
    }
    for parameter, c_gradient, n_gradient in zip(
        partition.shared, camera_shared, joint_shared, strict=True
    ):
        gradient = c_scale * c_gradient + n_scale * n_gradient
        if id(parameter) in auxiliary_by_id:
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        if id(parameter) in plan_by_id:
            gradient = gradient + plan_scale * plan_by_id[id(parameter)]
        parameter.grad = gradient
    for parameter, inherited_gradient in zip(
        partition.representation, joint_representation, strict=True
    ):
        gradient = representation_scale * inherited_gradient
        if id(parameter) in auxiliary_by_id:
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        if id(parameter) in plan_by_id:
            gradient = gradient + plan_scale * plan_by_id[id(parameter)]
        parameter.grad = gradient
    for name, parameter, inherited_gradient in zip(
        partition.names["predictor"],
        partition.predictor,
        joint_predictor,
        strict=True,
    ):
        gradient = predictor_scale * inherited_gradient
        if name in v25.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24:
            if id(parameter) not in auxiliary_by_id:
                raise RuntimeError("V28 swept-progress output left J24")
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        elif id(parameter) in auxiliary_by_id:
            raise RuntimeError("V28 protected old predictor core entered J24")
        parameter.grad = gradient
    for parameter in partition.plan:
        if id(parameter) not in plan_by_id:
            raise RuntimeError("V28 plan parameter left its sole route")
        parameter.grad = plan_scale * plan_by_id[id(parameter)]

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V28 target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        v25._tensor_core._finite_tensor(torch, parameter, "V28 online parameter")
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("V28 EMA did not advance exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V28 target received a gradient after EMA")
    advanced = _advance_accounting_v28(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("V28 post-update EMA accounting differs")

    receipts = {
        name: v25._tensor_core._receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    if len(plan_energy_rows) != PLAN_PRESENTATIONS_PER_UPDATE_V28:
        raise RuntimeError("V28 plan energy accounting changed")
    mean_losses = {
        name: value / PHYSICAL_MICROBATCHES_PER_UPDATE_V28
        for name, value in sums.items()
    }
    mean_losses["P28"] = plan_loss_sum / PLAN_MICROBATCHES_PER_UPDATE_V28
    mean_losses["L28"] = mean_losses["L"] + mean_losses["P28"]
    return JointUpdateResultV28(
        accounting=advanced,
        mean_losses=mean_losses,
        gradient_routes=receipts,
        plan_diagnostics={
            "mechanism": PLAN_ROUTE_NAME_V28,
            "energy_per_row": tuple(plan_energy_rows),
            "energy_mean": sum(plan_energy_rows) / len(plan_energy_rows),
            "p25_evaluation_count": 0,
        },
        predictor_core_protected_survival_diagnostics={
            **auxiliary_sums,
            "positive_energy_count": positive_count,
            "positive_energy_mean": (
                auxiliary_sums["positive_energy_sum"] / positive_count
            ),
            "scene_eligible_count": scene_count,
            "scene_negative_energy_mean": (
                auxiliary_sums["scene_negative_energy_sum"] / scene_count
            ),
            "scene_advantage_mean": (
                auxiliary_sums["scene_advantage_sum"] / scene_count
            ),
            "prior_eligible_count": prior_count,
            "prior_negative_energy_mean": (
                auxiliary_sums["prior_negative_energy_sum"] / prior_count
            ),
            "prior_advantage_mean": (
                auxiliary_sums["prior_advantage_sum"] / prior_count
            ),
            "non_hold_action_count_per_row": v25.NON_HOLD_ACTION_COUNT_V23,
        },
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


# Narrow runtime-initialization aliases.  The controller calls the V28 update
# explicitly because it supplies a second, H6-only microbatch sequence.
partition_parameters_v13 = partition_parameters_v28
build_frozen_optimizer_v13 = build_optimizer_v28


__all__ = [
    "H6_CURRENT_RGB_KEY_V28",
    "H6_FUTURE_ACTIONS_KEY_V28",
    "H6_TERMINAL_RGB_KEY_V28",
    "JointTrainingAccountingV28",
    "JointUpdateResultV28",
    "MAXIMUM_PRESENTATIONS_V28",
    "MAXIMUM_UPDATES_V28",
    "MICROBATCH_SIZE_V28",
    "PLAN_ROUTE_NAME_V28",
    "PRESENTATIONS_PER_UPDATE_V28",
    "ParameterPartitionV28",
    "REQUIRED_H6_BATCH_KEYS_V28",
    "build_optimizer_v28",
    "build_frozen_optimizer_v13",
    "joint_training_update_v28",
    "partition_parameters_v28",
    "partition_parameters_v13",
    "validate_accounting_v28",
    "validate_optimizer_v28",
    "terminal_successor_target_v28",
]
