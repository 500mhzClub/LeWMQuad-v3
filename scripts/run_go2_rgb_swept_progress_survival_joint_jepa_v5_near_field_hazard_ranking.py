#!/usr/bin/env python3
"""Lean V5 joint-training core with one near-field hazard-ranking loss.

V5 retains the V3/V4 model and every inherited training term.  Its only
scientific change is the parameter-free loss ``H`` preregistered over the
complete Cartesian ordering of near OCCUPIED and near FREE raster cells.
This source opens no data, checkpoint, or runtime artifact.
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
        run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
        as _v3,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


# Exact inherited identities.
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
REQUIRED_BATCH_KEYS = _v3.REQUIRED_BATCH_KEYS

FrozenSurvivalRoleLabelsV1 = _v3.FrozenSurvivalRoleLabelsV1
ProgressControlScoresV1 = _v3.ProgressControlScoresV1
ParameterPartitionV1 = _v3.ParameterPartitionV1
JointTrainingAccountingV1 = _v3.JointTrainingAccountingV1
freeze_role_labels_v1 = _v3.freeze_role_labels_v1
validate_pairs_against_labels_v1 = _v3.validate_pairs_against_labels_v1
build_microbatch_v1 = _v3.build_microbatch_v1
partition_parameters_v1 = _v3.partition_parameters_v1
build_frozen_optimizer_v1 = _v3.build_frozen_optimizer_v1
validate_optimizer_v1 = _v3.validate_optimizer_v1
validate_accounting_v1 = _v3.validate_accounting_v1
score_full_control_v1 = _v3.score_full_control_v1
score_shuffled_action_control_v1 = _v3.score_shuffled_action_control_v1
score_persistence_control_v1 = _v3.score_persistence_control_v1
score_wrong_rgb_control_v1 = _v3.score_wrong_rgb_control_v1
OCCUPIED_SAFETY_AUX_COEFFICIENT = _v3.OCCUPIED_SAFETY_AUX_COEFFICIENT
OCCUPIED_SAFETY_AUX_NORMALIZATION = _v3.OCCUPIED_SAFETY_AUX_NORMALIZATION

UNKNOWN_CLASS_INDEX = 0
FREE_CLASS_INDEX = 1
OCCUPIED_CLASS_INDEX = 2
RASTER_SIZE = 64
RASTER_SIDE = RASTER_SIZE
FORWARD_MIN_M = -0.95
FORWARD_MAX_M = 5.35
LEFT_MIN_M = -3.15
LEFT_MAX_M = 3.15
FORWARD_RANGE_M = (FORWARD_MIN_M, FORWARD_MAX_M)
LEFT_RANGE_M = (LEFT_MIN_M, LEFT_MAX_M)
NEAR_FIELD_RANGE_M = 2.0
NEAR_FIELD_CELL_COUNT = 1_016
HAZARD_RANKING_COEFFICIENT = 1.0
HAZARD_RANKING_NORMALIZATION = math.log(2.0)


@dataclass(frozen=True)
class NearFieldHazardRankingLossTermsV5:
    """Exact loss and eligibility accounting for both raster views."""

    loss: Any
    current_per_eligible_row: Any
    next_per_eligible_row: Any
    current_eligible_row_count: int
    next_eligible_row_count: int
    current_ranked_pair_count: int
    next_ranked_pair_count: int
    active: bool


@dataclass(frozen=True)
class JointUpdateResultV5:
    """Inherited update receipt plus exact per-microbatch ``H`` evidence."""

    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_l2: Mapping[str, float]
    representation_clip_pre_l2: float
    predictor_clip_pre_l2: float
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    hazard_ranking_microbatches: tuple[Mapping[str, Any], ...]


def _validate_semantic_view_v5(torch: Any, logits: Any, labels: Any, name: str) -> None:
    if (
        not isinstance(logits, torch.Tensor)
        or tuple(logits.shape[1:]) != (3, RASTER_SIDE, RASTER_SIDE)
        or logits.shape[0] < 1
    ):
        raise ValueError(f"{name} logits must have shape (B,3,64,64)")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} logits must be finite floating tensors")
    if (
        not isinstance(labels, torch.Tensor)
        or tuple(labels.shape) != (logits.shape[0], RASTER_SIDE, RASTER_SIDE)
    ):
        raise ValueError(f"{name} labels must have shape (B,64,64)")
    if labels.device != logits.device:
        raise ValueError(f"{name} logits and labels must share a device")
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"{name} labels must use an integer dtype")
    if not bool(((labels >= UNKNOWN_CLASS_INDEX) & (labels <= OCCUPIED_CLASS_INDEX)).all()):
        raise ValueError(f"{name} labels contain an unsupported class")


def _near_field_mask_v5(torch: Any, *, device: Any) -> Any:
    forward = torch.linspace(
        FORWARD_RANGE_M[0],
        FORWARD_RANGE_M[1],
        RASTER_SIDE,
        dtype=torch.float64,
        device="cpu",
    )
    left = torch.linspace(
        LEFT_RANGE_M[0],
        LEFT_RANGE_M[1],
        RASTER_SIDE,
        dtype=torch.float64,
        device="cpu",
    )
    mask = torch.sqrt(forward[:, None].square() + left[None, :].square()) <= (
        NEAR_FIELD_RANGE_M
    )
    if tuple(mask.shape) != (RASTER_SIZE, RASTER_SIZE) or int(mask.sum()) != (
        NEAR_FIELD_CELL_COUNT
    ):
        raise RuntimeError("fixed V5 near-field mask changed")
    return mask.to(device=device)


def _rank_view_v5(torch: Any, logits: Any, labels: Any, near: Any) -> tuple[Any, int]:
    hazard = logits[:, OCCUPIED_CLASS_INDEX] - torch.logsumexp(
        logits[:, :OCCUPIED_CLASS_INDEX], dim=1
    )
    losses = []
    ranked_pairs = 0
    for row in range(logits.shape[0]):
        occupied = hazard[row][near & (labels[row] == OCCUPIED_CLASS_INDEX)]
        free = hazard[row][near & (labels[row] == FREE_CLASS_INDEX)]
        if occupied.numel() == 0 or free.numel() == 0:
            continue
        ranked_pairs += int(occupied.numel() * free.numel())
        losses.append(
            torch.nn.functional.softplus(free[:, None] - occupied[None, :]).mean()
            / HAZARD_RANKING_NORMALIZATION
        )
    if losses:
        return torch.stack(losses), ranked_pairs
    return logits.new_empty((0,)), ranked_pairs


def near_field_hazard_ranking_loss_v5(
    current_logits: Any,
    current_labels: Any,
    next_logits: Any,
    next_labels: Any,
) -> NearFieldHazardRankingLossTermsV5:
    """Compute the exact complete-pair, 2 m hazard-ordering loss ``H``."""

    torch = _v3._v2._torch_api()
    _validate_semantic_view_v5(torch, current_logits, current_labels, "current")
    _validate_semantic_view_v5(torch, next_logits, next_labels, "next")
    if current_logits.shape != next_logits.shape or current_labels.shape != next_labels.shape:
        raise ValueError("current and next semantic views must have matching shapes")
    if current_logits.device != next_logits.device or current_logits.dtype != next_logits.dtype:
        raise ValueError("current and next logits must share device and dtype")

    near = _near_field_mask_v5(torch, device=current_logits.device)
    current_rows, current_pairs = _rank_view_v5(
        torch, current_logits, current_labels, near
    )
    next_rows, next_pairs = _rank_view_v5(torch, next_logits, next_labels, near)
    present_view_means = [
        rows.mean() for rows in (current_rows, next_rows) if rows.numel() > 0
    ]
    if present_view_means:
        loss = HAZARD_RANKING_COEFFICIENT * torch.stack(present_view_means).mean()
    else:
        loss = (
            current_logits.mul(0.0).sum() + next_logits.mul(0.0).sum()
        )
    return NearFieldHazardRankingLossTermsV5(
        loss=loss,
        current_per_eligible_row=current_rows,
        next_per_eligible_row=next_rows,
        current_eligible_row_count=int(current_rows.numel()),
        next_eligible_row_count=int(next_rows.numel()),
        current_ranked_pair_count=current_pairs,
        next_ranked_pair_count=next_pairs,
        active=bool(present_view_means),
    )


def joint_training_update_v5(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV5:
    """Accumulate four exact V4+H graphs, then clip, step, and EMA once."""

    torch, semantic_api, survival_api = _v3._v2._v1._runtime_apis()
    _v3._v2._v1._validate_microbatches(torch, microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    state = JointTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with training accounting")

    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in ("S", "P", "U", "R", "O", "H", "L")}
    active_ranking = eligible_pairs = supervised_decisions = 0
    hazard_receipts: list[Mapping[str, Any]] = []
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
        occupied = _v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        hazard = near_field_hazard_ranking_loss_v5(
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
        total = joint.loss + occupied.loss + hazard.loss
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("occupied auxiliary", occupied.loss),
            ("hazard ranking", hazard.loss),
            ("joint loss", total),
        ):
            _v3._v2._v1._base._finite_tensor(torch, value, name)
        (total / MICROBATCHES_PER_UPDATE).backward()
        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("H", hazard.loss),
            ("L", total),
        ):
            sums[name] += _v3._v2._v1._base._scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )
        hazard_receipts.append(
            {
                "H": _v3._v2._v1._base._scalar(hazard.loss),
                "hazard_active": hazard.active,
                "hazard_current_eligible_row_count": (
                    hazard.current_eligible_row_count
                ),
                "hazard_next_eligible_row_count": hazard.next_eligible_row_count,
                "hazard_current_ranked_pair_count": (
                    hazard.current_ranked_pair_count
                ),
                "hazard_next_ranked_pair_count": hazard.next_ranked_pair_count,
            }
        )

    gradient_l2 = {
        "encoder": _v3._v2._v1._base._gradient_l2(
            torch, partition.encoder, "encoder"
        ),
        "lift_semantic": _v3._v2._v1._base._gradient_l2(
            torch, partition.lift_semantic, "lift/semantic"
        ),
        "predictor": _v3._v2._v1._base._gradient_l2(
            torch, partition.predictor, "predictor"
        ),
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
        _v3._v2._v1._base._finite_tensor(torch, parameter, "online parameter")
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")

    advanced = _v3._v2._v1._base._advanced_accounting(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("post-update EMA count disagrees with accounting")
    return JointUpdateResultV5(
        accounting=advanced,
        mean_losses={
            name: value / MICROBATCHES_PER_UPDATE for name, value in sums.items()
        },
        gradient_l2=gradient_l2,
        representation_clip_pre_l2=_v3._v2._v1._base._scalar(representation_pre),
        predictor_clip_pre_l2=_v3._v2._v1._base._scalar(predictor_pre),
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        hazard_ranking_microbatches=tuple(hazard_receipts),
    )


def _hazard_totals_v5(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_pairs = sum(
        int(row["hazard_current_ranked_pair_count"]) for row in receipts
    )
    next_pairs = sum(
        int(row["hazard_next_ranked_pair_count"]) for row in receipts
    )
    return {
        "hazard_active_microbatch_count": sum(
            bool(row["hazard_active"]) for row in receipts
        ),
        "hazard_current_eligible_row_count": sum(
            int(row["hazard_current_eligible_row_count"]) for row in receipts
        ),
        "hazard_next_eligible_row_count": sum(
            int(row["hazard_next_eligible_row_count"]) for row in receipts
        ),
        "hazard_current_ranked_pair_count": current_pairs,
        "hazard_next_ranked_pair_count": next_pairs,
        "hazard_ranked_pair_count": current_pairs + next_pairs,
    }


def run_fixed_training_v5(
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
    """Consume the exact inherited cap with ``H`` active from update one."""

    update_receipts: list[tuple[Mapping[str, Any], ...]] = []

    def tracked_update(
        candidate: Any,
        candidate_optimizer: Any,
        microbatches: Sequence[Mapping[str, Any]],
        *,
        accounting: JointTrainingAccountingV1,
    ) -> JointUpdateResultV5:
        result = joint_training_update_v5(
            candidate,
            candidate_optimizer,
            microbatches,
            accounting=accounting,
        )
        update_receipts.append(result.hazard_ranking_microbatches)
        return result

    accounting, inherited_trace, inherited_diagnostics = (
        _v3._v2._run_fixed_training_core_v2(
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
            joint_update=tracked_update,
        )
    )
    if len(update_receipts) != MAXIMUM_UPDATES or any(
        len(rows) != MICROBATCHES_PER_UPDATE for rows in update_receipts
    ):
        raise RuntimeError("V5 hazard-ranking microbatch accounting changed")

    trace: list[dict[str, Any]] = []
    for inherited, rows in zip(inherited_trace, update_receipts, strict=True):
        trace.append(
            {
                **inherited,
                "hazard_ranking_activity": {
                    "microbatches": [dict(row) for row in rows],
                    **_hazard_totals_v5(rows),
                },
            }
        )
    windows = []
    for start in range(0, MAXIMUM_UPDATES, 100):
        rows = [
            row
            for update_rows in update_receipts[start : start + 100]
            for row in update_rows
        ]
        windows.append(
            {
                "first_update": start + 1,
                "last_update": start + 100,
                "hazard_mean_microbatch_H": (
                    sum(float(row["H"]) for row in rows) / len(rows)
                ),
                **_hazard_totals_v5(rows),
            }
        )
    all_receipts = [row for rows in update_receipts for row in rows]
    diagnostics = {
        **inherited_diagnostics,
        "hazard_ranking_activity": {
            **_hazard_totals_v5(all_receipts),
            "hazard_microbatch_count": len(all_receipts),
            "hazard_windows_100_updates": windows,
        },
    }
    return accounting, tuple(trace), diagnostics


__all__ = [
    "ACTION_ORDER",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "FORWARD_RANGE_M",
    "FORWARD_MAX_M",
    "FORWARD_MIN_M",
    "FREE_CLASS_INDEX",
    "FrozenSurvivalRoleLabelsV1",
    "HAZARD_RANKING_COEFFICIENT",
    "HAZARD_RANKING_NORMALIZATION",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV5",
    "LEFT_RANGE_M",
    "LEFT_MAX_M",
    "LEFT_MIN_M",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEAR_FIELD_RANGE_M",
    "NEAR_FIELD_CELL_COUNT",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "NearFieldHazardRankingLossTermsV5",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "PREFIX_LENGTHS_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ProgressControlScoresV1",
    "RASTER_SIDE",
    "RASTER_SIZE",
    "REQUIRED_BATCH_KEYS",
    "UNKNOWN_CLASS_INDEX",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v5",
    "near_field_hazard_ranking_loss_v5",
    "partition_parameters_v1",
    "run_fixed_training_v5",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_action_control_v1",
    "score_wrong_rgb_control_v1",
    "validate_accounting_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]
