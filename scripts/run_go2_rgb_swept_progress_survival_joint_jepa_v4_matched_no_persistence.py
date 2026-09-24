#!/usr/bin/env python3
"""Matched V4 training core with persistence excluded from backward membership.

This control preserves the V4/V3 model, inputs, forwards, optimizer, schedule,
clipping, target EMA, occupied auxiliary, and accounting.  It still computes
the executed-action persistence term ``P`` as a diagnostic; the direct
backward scalar is exactly ``S + U + R + O``.

The module performs no discovery and opens no data, checkpoint, or runtime
artifact.  Callers supply the reviewed V4 model and the inherited V3 inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import MappingProxyType
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

OCCUPIED_CLASS_INDEX = _v3.OCCUPIED_CLASS_INDEX
OCCUPIED_SAFETY_AUX_COEFFICIENT = _v3.OCCUPIED_SAFETY_AUX_COEFFICIENT
OCCUPIED_SAFETY_AUX_NORMALIZATION = _v3.OCCUPIED_SAFETY_AUX_NORMALIZATION
occupied_safety_aux_loss_v4_matched_no_persistence = (
    _v3.occupied_safety_aux_loss_v3
)

COMPONENT_KEYS = ("S", "P_diagnostic", "U", "R", "O")
TRACE_LOSS_KEYS = COMPONENT_KEYS + ("L_full_diagnostic", "L_backward")
GRADIENT_GROUPS = ("encoder", "lift_semantic", "predictor")
FIRST_UPDATE_COMPONENT_MEANS = MappingProxyType(
    {
        "S": 1.313827022910118,
        "P_diagnostic": 1.0,
        "U": 0.9792981296777725,
        "R": 1.0,
        "O": 1.026371382176876,
    }
)


class FirstUpdateComponentWitnessMismatchV1(RuntimeError):
    """Structured pre-step failure evidence for the one-shot executor."""

    def __init__(
        self, expected: Mapping[str, float], observed: Mapping[str, float]
    ) -> None:
        expected_copy = {name: expected[name] for name in COMPONENT_KEYS}
        observed_copy = {name: observed[name] for name in COMPONENT_KEYS}
        mismatch = {
            name: MappingProxyType(
                {
                    "expected": expected_copy[name],
                    "observed": observed_copy[name],
                }
            )
            for name in COMPONENT_KEYS
            if observed_copy[name] != expected_copy[name]
        }
        self.expected = MappingProxyType(expected_copy)
        self.observed = MappingProxyType(observed_copy)
        self.mismatch = MappingProxyType(mismatch)
        self.pre_step_operation_counts = MappingProxyType(
            {
                "presentations_consumed": PRESENTATIONS_PER_UPDATE,
                "microbatch_graphs_completed": MICROBATCHES_PER_UPDATE,
                "backward_calls_completed": MICROBATCHES_PER_UPDATE,
                "optimizer_steps_completed": 0,
                "ema_steps_completed": 0,
                "predictor_forwards_completed": MICROBATCHES_PER_UPDATE,
                "predictor_objectives_evaluated": MICROBATCHES_PER_UPDATE,
            }
        )
        super().__init__(f"first-update component witness mismatch: {mismatch}")


@dataclass(frozen=True)
class JointUpdateResultV4MatchedNoPersistence:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_l2: Mapping[str, float]
    representation_clip_pre_l2: float
    predictor_clip_pre_l2: float
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    first_update_component_witness_checked: bool = False


def _check_loss_identities(mean_losses: Mapping[str, float]) -> None:
    if tuple(mean_losses) != TRACE_LOSS_KEYS:
        raise RuntimeError("matched-control trace loss keys changed")
    if not all(math.isfinite(value) for value in mean_losses.values()):
        raise FloatingPointError("matched-control trace contains a nonfinite loss")
    full_sum = sum(mean_losses[name] for name in COMPONENT_KEYS)
    backward_sum = sum(mean_losses[name] for name in ("S", "U", "R", "O"))
    if not math.isclose(
        mean_losses["L_full_diagnostic"], full_sum, rel_tol=2e-6, abs_tol=2e-6
    ):
        raise RuntimeError("L_full_diagnostic identity changed")
    if not math.isclose(
        mean_losses["L_backward"], backward_sum, rel_tol=2e-6, abs_tol=2e-6
    ):
        raise RuntimeError("L_backward identity changed")


def _check_first_update_component_witness(
    observed: Mapping[str, float], expected: Mapping[str, float]
) -> None:
    if tuple(expected) != COMPONENT_KEYS:
        raise ValueError("first-update witness keys changed")
    if any(observed[name] != expected[name] for name in COMPONENT_KEYS):
        raise FirstUpdateComponentWitnessMismatchV1(expected, observed)


def joint_training_update_v4_matched_no_persistence(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
    expected_component_means: Mapping[str, float] | None = None,
) -> JointUpdateResultV4MatchedNoPersistence:
    """Accumulate four no-``P`` graphs, then clip, step, and EMA exactly once."""

    torch, semantic_api, survival_api = _v3._v2._v1._runtime_apis()
    _v3._v2._v1._validate_microbatches(torch, microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    state = JointTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    if expected_component_means is not None and state.updates != 0:
        raise ValueError("the component witness belongs only to update 1")
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with training accounting")

    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in TRACE_LOSS_KEYS}
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
        full_diagnostic = joint.loss + occupied.loss
        backward_loss = (
            joint.semantic
            + joint.survival
            + joint.progress_ranking
            + occupied.loss
        )
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("persistence diagnostic", joint.executed_action_ema_latent),
            ("occupied auxiliary", occupied.loss),
            ("full diagnostic", full_diagnostic),
            ("backward loss", backward_loss),
        ):
            _v3._v2._v1._base._finite_tensor(torch, value, name)

        # This is deliberately a direct four-term scalar.  P is not a member
        # of the backward graph and is neither cancelled nor zero-weighted.
        (backward_loss / MICROBATCHES_PER_UPDATE).backward()
        for name, value in (
            ("S", joint.semantic),
            ("P_diagnostic", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("L_full_diagnostic", full_diagnostic),
            ("L_backward", backward_loss),
        ):
            sums[name] += _v3._v2._v1._base._scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    mean_losses = {
        name: sums[name] / MICROBATCHES_PER_UPDATE for name in TRACE_LOSS_KEYS
    }
    _check_loss_identities(mean_losses)
    witness_checked = expected_component_means is not None
    if expected_component_means is not None:
        # By construction this is after all four backward calls and before any
        # clipping, optimizer step, or target EMA update.
        _check_first_update_component_witness(
            {name: mean_losses[name] for name in COMPONENT_KEYS},
            expected_component_means,
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
    return JointUpdateResultV4MatchedNoPersistence(
        accounting=advanced,
        mean_losses=mean_losses,
        gradient_l2=gradient_l2,
        representation_clip_pre_l2=_v3._v2._v1._base._scalar(representation_pre),
        predictor_clip_pre_l2=_v3._v2._v1._base._scalar(predictor_pre),
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        first_update_component_witness_checked=witness_checked,
    )


def _run_fixed_training_core_v4_matched_no_persistence(
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
    """Shared fixed driver with locked cap and update-1 functional witness."""

    if maximum_updates != MAXIMUM_UPDATES or len(schedule) != MAXIMUM_PRESENTATIONS:
        raise PermissionError("training cap or schedule length changed")
    if tuple(action_order) != ACTION_ORDER:
        raise PermissionError("action order changed")
    validate_pairs_against_labels_v1(train_pairs, train_labels)
    accounting = JointTrainingAccountingV1()
    trace: list[dict[str, Any]] = []
    active_ranking = eligible_pairs = supervised_decisions = 0
    min_gradients = {name: math.inf for name in GRADIENT_GROUPS}
    max_gradients = {name: 0.0 for name in GRADIENT_GROUPS}
    first_observed: dict[str, float] | None = None
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
            model,
            optimizer,
            microbatches,
            accounting=accounting,
            expected_component_means=(
                FIRST_UPDATE_COMPONENT_MEANS if update == 1 else None
            ),
        )
        if update == 1:
            if not result.first_update_component_witness_checked:
                raise RuntimeError("update-1 component witness was not checked")
            first_observed = {
                name: result.mean_losses[name] for name in COMPONENT_KEYS
            }
        accounting = result.accounting
        if tuple(result.gradient_l2) != GRADIENT_GROUPS:
            raise RuntimeError("gradient diagnostic groups changed")
        for name, value in result.gradient_l2.items():
            min_gradients[name] = min(min_gradients[name], value)
            max_gradients[name] = max(max_gradients[name], value)
        active_ranking += result.ranking_active_microbatches
        eligible_pairs += result.ranking_eligible_pairs
        supervised_decisions += result.survival_supervised_decisions
        _check_loss_identities(result.mean_losses)
        trace.append(
            {
                "update": update,
                "presentations": accounting.presentations,
                "losses": dict(result.mean_losses),
                "gradient_l2": dict(result.gradient_l2),
            }
        )

    validate_accounting_v1(accounting)
    terminal = JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    )
    if accounting != terminal:
        raise RuntimeError("terminal training accounting changed")
    return accounting, tuple(trace), {
        "ranking_active_microbatch_count": active_ranking,
        "ranking_eligible_pair_count": eligible_pairs,
        "survival_supervised_decision_count": supervised_decisions,
        "gradient_groups": list(GRADIENT_GROUPS),
        "minimum_gradient_l2": min_gradients,
        "maximum_gradient_l2": max_gradients,
        "first_update_component_witness": {
            "expected": dict(FIRST_UPDATE_COMPONENT_MEANS),
            "observed": first_observed,
            "exact_match": first_observed == dict(FIRST_UPDATE_COMPONENT_MEANS),
            "checked_after_backward_calls": MICROBATCHES_PER_UPDATE,
            "checked_before_optimizer_step": True,
        },
    }


def run_fixed_training_v4_matched_no_persistence(
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
    """Consume the exact V4 cap with ``P`` diagnostic-only from update one."""

    return _run_fixed_training_core_v4_matched_no_persistence(
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
        joint_update=joint_training_update_v4_matched_no_persistence,
    )


__all__ = [
    "ACTION_ORDER",
    "COMPONENT_KEYS",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "FIRST_UPDATE_COMPONENT_MEANS",
    "FirstUpdateComponentWitnessMismatchV1",
    "FrozenSurvivalRoleLabelsV1",
    "GRADIENT_GROUPS",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV4MatchedNoPersistence",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "PREFIX_LENGTHS_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ProgressControlScoresV1",
    "REQUIRED_BATCH_KEYS",
    "TRACE_LOSS_KEYS",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v4_matched_no_persistence",
    "occupied_safety_aux_loss_v4_matched_no_persistence",
    "partition_parameters_v1",
    "run_fixed_training_v4_matched_no_persistence",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_action_control_v1",
    "score_wrong_rgb_control_v1",
    "validate_accounting_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]
