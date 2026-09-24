#!/usr/bin/env python3
"""Lean V2 training core adding one occupied-vs-rest semantic auxiliary.

V2 deliberately reuses the V1 model, data adapter, optimizer, controls,
schedule, masks, and accounting.  Its only scientific delta is the
coefficient-one term ``O`` computed from the existing three-class semantic
logits and labels jointly from the first update.

This module performs no discovery and opens no data, checkpoints, or runtime
artifacts.  Callers supply the same reviewed inputs as V1.
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
        run_go2_rgb_swept_progress_survival_joint_jepa_v1 as _v1,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


# Exact V1 identities retained by the V2 core.
ACTION_ORDER = _v1.ACTION_ORDER
MICROBATCHES_PER_UPDATE = _v1.MICROBATCHES_PER_UPDATE
MICROBATCH_SIZE = _v1.MICROBATCH_SIZE
PRESENTATIONS_PER_UPDATE = _v1.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v1.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v1.MAXIMUM_PRESENTATIONS

CURRENT_RGB_KEY = _v1.CURRENT_RGB_KEY
NEXT_RGB_KEY = _v1.NEXT_RGB_KEY
CURRENT_LABELS_KEY = _v1.CURRENT_LABELS_KEY
NEXT_LABELS_KEY = _v1.NEXT_LABELS_KEY
EXECUTED_ACTION_KEY = _v1.EXECUTED_ACTION_KEY
IMMEDIATE_FEASIBLE_KEY = _v1.IMMEDIATE_FEASIBLE_KEY
PREFIX_LENGTHS_KEY = _v1.PREFIX_LENGTHS_KEY
REQUIRED_BATCH_KEYS = _v1.REQUIRED_BATCH_KEYS

FrozenSurvivalRoleLabelsV1 = _v1.FrozenSurvivalRoleLabelsV1
ProgressControlScoresV1 = _v1.ProgressControlScoresV1
ParameterPartitionV1 = _v1.ParameterPartitionV1
JointTrainingAccountingV1 = _v1.JointTrainingAccountingV1
freeze_role_labels_v1 = _v1.freeze_role_labels_v1
validate_pairs_against_labels_v1 = _v1.validate_pairs_against_labels_v1
build_microbatch_v1 = _v1.build_microbatch_v1
partition_parameters_v1 = _v1.partition_parameters_v1
build_frozen_optimizer_v1 = _v1.build_frozen_optimizer_v1
validate_optimizer_v1 = _v1.validate_optimizer_v1
validate_accounting_v1 = _v1.validate_accounting_v1
score_full_control_v1 = _v1.score_full_control_v1
score_shuffled_action_control_v1 = _v1.score_shuffled_action_control_v1
score_persistence_control_v1 = _v1.score_persistence_control_v1
score_wrong_rgb_control_v1 = _v1.score_wrong_rgb_control_v1

OCCUPIED_CLASS_INDEX = 2
OCCUPIED_SAFETY_AUX_COEFFICIENT = 1.0
OCCUPIED_SAFETY_AUX_NORMALIZATION = math.log(2.0)


@dataclass(frozen=True)
class OccupiedSafetyAuxLossTermsV2:
    """Normalized occupied-vs-rest term and its raw per-raster row values."""

    loss: Any
    current_per_row: Any
    next_per_row: Any


@dataclass(frozen=True)
class JointUpdateResultV2:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_l2: Mapping[str, float]
    representation_clip_pre_l2: float
    predictor_clip_pre_l2: float
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int


def _torch_api() -> Any:
    inserted = str(ROOT) not in sys.path
    if inserted:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("torch")
    finally:
        if inserted:
            sys.path.remove(str(ROOT))


def _occupied_vs_rest_per_row_v2(logits: Any, labels: Any) -> Any:
    """Return equal-present-binary-class BCE for every raster row.

    The positive logit is exactly ``occupied - logsumexp(unknown, free)``.
    Within each row, the mean loss for REST and OCCUPIED is computed when that
    binary class is present, and the present class means receive equal weight.
    """

    torch = _torch_api()
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError("semantic logits must have shape (B,3,H,W)")
    if logits.shape[0] < 1 or logits.shape[1] != 3 or min(logits.shape[2:]) < 1:
        raise ValueError("semantic logits must have shape (B,3,H,W)")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("semantic logits must be finite floating tensors")
    if not isinstance(labels, torch.Tensor) or labels.shape != (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
    ):
        raise ValueError("semantic labels must have shape (B,H,W)")
    if labels.device != logits.device:
        raise ValueError("semantic logits and labels must share a device")
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError("semantic labels must use an integer dtype")
    if not bool(((labels >= 0) & (labels <= OCCUPIED_CLASS_INDEX)).all()):
        raise ValueError("semantic labels must contain only classes 0, 1, and 2")

    occupied_logit = logits[:, OCCUPIED_CLASS_INDEX] - torch.logsumexp(
        logits[:, :OCCUPIED_CLASS_INDEX], dim=1
    )
    occupied_target = labels == OCCUPIED_CLASS_INDEX
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        occupied_logit,
        occupied_target.to(dtype=logits.dtype),
        reduction="none",
    )
    flat_loss = element_loss.flatten(start_dim=1)
    flat_occupied = occupied_target.flatten(start_dim=1)
    occupied_count = flat_occupied.sum(dim=1)
    rest_count = (~flat_occupied).sum(dim=1)
    occupied_mean = (
        (flat_loss * flat_occupied.to(dtype=logits.dtype)).sum(dim=1)
        / occupied_count.clamp_min(1).to(dtype=logits.dtype)
    )
    rest_mean = (
        (flat_loss * (~flat_occupied).to(dtype=logits.dtype)).sum(dim=1)
        / rest_count.clamp_min(1).to(dtype=logits.dtype)
    )
    occupied_present = occupied_count > 0
    rest_present = rest_count > 0
    present_count = occupied_present.to(dtype=logits.dtype) + rest_present.to(
        dtype=logits.dtype
    )
    return (
        occupied_mean * occupied_present.to(dtype=logits.dtype)
        + rest_mean * rest_present.to(dtype=logits.dtype)
    ) / present_count


def occupied_safety_aux_loss_v2(
    current_logits: Any,
    current_labels: Any,
    next_logits: Any,
    next_labels: Any,
) -> OccupiedSafetyAuxLossTermsV2:
    """Compute coefficient-one normalized term ``O`` for current and next."""

    return _occupied_safety_aux_loss_with_coefficient_v2(
        current_logits,
        current_labels,
        next_logits,
        next_labels,
        coefficient=OCCUPIED_SAFETY_AUX_COEFFICIENT,
    )


def _occupied_safety_aux_loss_with_coefficient_v2(
    current_logits: Any,
    current_labels: Any,
    next_logits: Any,
    next_labels: Any,
    *,
    coefficient: float,
) -> OccupiedSafetyAuxLossTermsV2:
    """Shared immutable implementation used by coefficient-frozen variants."""

    if current_logits.shape != next_logits.shape:
        raise ValueError("current and next semantic logits must have matching shapes")
    if current_labels.shape != next_labels.shape:
        raise ValueError("current and next semantic labels must have matching shapes")
    current_rows = _occupied_vs_rest_per_row_v2(current_logits, current_labels)
    next_rows = _occupied_vs_rest_per_row_v2(next_logits, next_labels)
    loss = coefficient * (
        0.5 * current_rows.mean() + 0.5 * next_rows.mean()
    ) / OCCUPIED_SAFETY_AUX_NORMALIZATION
    return OccupiedSafetyAuxLossTermsV2(
        loss=loss,
        current_per_row=current_rows,
        next_per_row=next_rows,
    )


def _joint_training_update_with_occupied_coefficient_v2(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    occupied_coefficient: float,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV2:
    """Shared joint update for variants with one frozen ``O`` coefficient."""

    torch, semantic_api, survival_api = _v1._runtime_apis()
    _v1._validate_microbatches(torch, microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    state = JointTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with training accounting")

    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in ("S", "P", "U", "R", "O", "L")}
    active_ranking = 0
    eligible_pairs = 0
    supervised_decisions = 0
    for batch in microbatches:
        current_latent = model.encode_online(batch[CURRENT_RGB_KEY])
        next_latent = model.encode_online(batch[NEXT_RGB_KEY])
        current_logits = model.semantic_logits_from_latent(current_latent)
        next_logits = model.semantic_logits_from_latent(next_latent)
        semantic = semantic_api.semantic_loss_v1(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        occupied = _occupied_safety_aux_loss_with_coefficient_v2(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
            coefficient=occupied_coefficient,
        )
        prediction = model.predict_all_actions_with_survival(current_latent)
        predicted, survival_logits = _v1._prediction_parts(prediction)
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
        total = joint.loss + occupied.loss
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("occupied auxiliary", occupied.loss),
            ("joint loss", total),
        ):
            _v1._base._finite_tensor(torch, value, name)
        (total / MICROBATCHES_PER_UPDATE).backward()
        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("L", total),
        ):
            sums[name] += _v1._base._scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    gradient_l2 = {
        "encoder": _v1._base._gradient_l2(torch, partition.encoder, "encoder"),
        "lift_semantic": _v1._base._gradient_l2(
            torch, partition.lift_semantic, "lift/semantic"
        ),
        "predictor": _v1._base._gradient_l2(torch, partition.predictor, "predictor"),
    }
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")
    representation_pre = torch.nn.utils.clip_grad_norm_(
        partition.representation, max_norm=1.0, error_if_nonfinite=True
    )
    predictor_pre = torch.nn.utils.clip_grad_norm_(
        partition.predictor, max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    for parameter in partition.online:
        _v1._base._finite_tensor(torch, parameter, "online parameter")
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")

    advanced = _v1._base._advanced_accounting(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("post-update EMA count disagrees with accounting")
    return JointUpdateResultV2(
        accounting=advanced,
        mean_losses={
            name: value / MICROBATCHES_PER_UPDATE for name, value in sums.items()
        },
        gradient_l2=gradient_l2,
        representation_clip_pre_l2=_v1._base._scalar(representation_pre),
        predictor_clip_pre_l2=_v1._base._scalar(predictor_pre),
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
    )


def joint_training_update_v2(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV2:
    """Accumulate four V1 graphs plus coefficient-one ``O``, then step once."""

    return _joint_training_update_with_occupied_coefficient_v2(
        model,
        optimizer,
        microbatches,
        occupied_coefficient=OCCUPIED_SAFETY_AUX_COEFFICIENT,
        accounting=accounting,
    )


def _run_fixed_training_core_v2(
    model: Any,
    optimizer: Any,
    loader: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    train_labels: FrozenSurvivalRoleLabelsV1,
    schedule: Sequence[int],
    device: Any,
    *,
    action_order: Sequence[str] = ACTION_ORDER,
    maximum_updates: int = MAXIMUM_UPDATES,
    microbatch_builder: Any,
    joint_update: Any,
) -> tuple[JointTrainingAccountingV1, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Shared fixed driver with immutable builder and update dependencies."""

    if maximum_updates != MAXIMUM_UPDATES or len(schedule) != MAXIMUM_PRESENTATIONS:
        raise PermissionError("training cap or schedule length changed")
    if tuple(action_order) != ACTION_ORDER:
        raise PermissionError("action order changed")
    validate_pairs_against_labels_v1(train_pairs, train_labels)
    accounting = JointTrainingAccountingV1()
    trace: list[dict[str, Any]] = []
    active_ranking = eligible_pairs = supervised_decisions = 0
    min_gradients = {
        name: math.inf for name in ("encoder", "lift_semantic", "predictor")
    }
    max_gradients = {name: 0.0 for name in min_gradients}
    for update in range(1, MAXIMUM_UPDATES + 1):
        start = (update - 1) * PRESENTATIONS_PER_UPDATE
        update_indices = schedule[start : start + PRESENTATIONS_PER_UPDATE]
        if len(update_indices) != PRESENTATIONS_PER_UPDATE:
            raise RuntimeError("training schedule exhausted before update 1000")
        microbatches = [
            microbatch_builder(
                loader,
                train_pairs,
                train_labels,
                update_indices[offset : offset + MICROBATCH_SIZE],
                device,
                stage=f"train_update_{update}",
                action_order=action_order,
            )
            for offset in range(0, PRESENTATIONS_PER_UPDATE, MICROBATCH_SIZE)
        ]
        result = joint_update(
            model, optimizer, microbatches, accounting=accounting
        )
        accounting = result.accounting
        for name, value in result.gradient_l2.items():
            min_gradients[name] = min(min_gradients[name], value)
            max_gradients[name] = max(max_gradients[name], value)
        active_ranking += result.ranking_active_microbatches
        eligible_pairs += result.ranking_eligible_pairs
        supervised_decisions += result.survival_supervised_decisions
        trace.append(
            {
                "update": update,
                "presentations": accounting.presentations,
                "losses": dict(result.mean_losses),
                "gradient_l2": dict(result.gradient_l2),
            }
        )
    validate_accounting_v1(accounting)
    if accounting != JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    ):
        raise RuntimeError("terminal training accounting changed")
    return accounting, tuple(trace), {
        "ranking_active_microbatch_count": active_ranking,
        "ranking_eligible_pair_count": eligible_pairs,
        "survival_supervised_decision_count": supervised_decisions,
        "minimum_gradient_l2": min_gradients,
        "maximum_gradient_l2": max_gradients,
    }


def run_fixed_training_v2(
    model: Any,
    optimizer: Any,
    loader: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    train_labels: FrozenSurvivalRoleLabelsV1,
    schedule: Sequence[int],
    device: Any,
    *,
    action_order: Sequence[str] = ACTION_ORDER,
    maximum_updates: int = MAXIMUM_UPDATES,
) -> tuple[JointTrainingAccountingV1, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Consume the exact V1 schedule with coefficient-one ``O`` from update one."""

    return _run_fixed_training_core_v2(
        model,
        optimizer,
        loader,
        train_pairs,
        train_labels,
        schedule,
        device,
        action_order=action_order,
        maximum_updates=maximum_updates,
        microbatch_builder=build_microbatch_v1,
        joint_update=joint_training_update_v2,
    )


__all__ = [
    "ACTION_ORDER",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "FrozenSurvivalRoleLabelsV1",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV2",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "OccupiedSafetyAuxLossTermsV2",
    "PREFIX_LENGTHS_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ProgressControlScoresV1",
    "REQUIRED_BATCH_KEYS",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v2",
    "occupied_safety_aux_loss_v2",
    "partition_parameters_v1",
    "run_fixed_training_v2",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_action_control_v1",
    "score_wrong_rgb_control_v1",
    "validate_accounting_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]
