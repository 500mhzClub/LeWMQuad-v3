#!/usr/bin/env python3
"""Lean V3 training core with the occupied-safety auxiliary weighted by 0.5.

V3 is an immutable coefficient variant of V2.  It reuses the exact V2 loss
math and joint-training loop while preserving the V1 model, data adapter,
optimizer, schedule, masks, controls, and accounting.  It adds no parameter or
head and opens no data, checkpoint, GPU, or runtime artifact.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_inserted_root = str(ROOT) not in sys.path
if _inserted_root:
    sys.path.insert(0, str(ROOT))
try:
    from scripts import (
        run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux
        as _v2,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


# Exact inherited identities.
ACTION_ORDER = _v2.ACTION_ORDER
MICROBATCHES_PER_UPDATE = _v2.MICROBATCHES_PER_UPDATE
MICROBATCH_SIZE = _v2.MICROBATCH_SIZE
PRESENTATIONS_PER_UPDATE = _v2.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v2.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v2.MAXIMUM_PRESENTATIONS

CURRENT_RGB_KEY = _v2.CURRENT_RGB_KEY
NEXT_RGB_KEY = _v2.NEXT_RGB_KEY
CURRENT_LABELS_KEY = _v2.CURRENT_LABELS_KEY
NEXT_LABELS_KEY = _v2.NEXT_LABELS_KEY
EXECUTED_ACTION_KEY = _v2.EXECUTED_ACTION_KEY
IMMEDIATE_FEASIBLE_KEY = _v2.IMMEDIATE_FEASIBLE_KEY
PREFIX_LENGTHS_KEY = _v2.PREFIX_LENGTHS_KEY
REQUIRED_BATCH_KEYS = _v2.REQUIRED_BATCH_KEYS

FrozenSurvivalRoleLabelsV1 = _v2.FrozenSurvivalRoleLabelsV1
ProgressControlScoresV1 = _v2.ProgressControlScoresV1
ParameterPartitionV1 = _v2.ParameterPartitionV1
JointTrainingAccountingV1 = _v2.JointTrainingAccountingV1
OccupiedSafetyAuxLossTermsV3 = _v2.OccupiedSafetyAuxLossTermsV2
JointUpdateResultV3 = _v2.JointUpdateResultV2
freeze_role_labels_v1 = _v2.freeze_role_labels_v1
validate_pairs_against_labels_v1 = _v2.validate_pairs_against_labels_v1
build_microbatch_v1 = _v2.build_microbatch_v1
partition_parameters_v1 = _v2.partition_parameters_v1
build_frozen_optimizer_v1 = _v2.build_frozen_optimizer_v1
validate_optimizer_v1 = _v2.validate_optimizer_v1
validate_accounting_v1 = _v2.validate_accounting_v1
score_full_control_v1 = _v2.score_full_control_v1
score_shuffled_action_control_v1 = _v2.score_shuffled_action_control_v1
score_persistence_control_v1 = _v2.score_persistence_control_v1
score_wrong_rgb_control_v1 = _v2.score_wrong_rgb_control_v1

OCCUPIED_CLASS_INDEX = _v2.OCCUPIED_CLASS_INDEX
OCCUPIED_SAFETY_AUX_COEFFICIENT = 0.5
OCCUPIED_SAFETY_AUX_NORMALIZATION = _v2.OCCUPIED_SAFETY_AUX_NORMALIZATION


def occupied_safety_aux_loss_v3(
    current_logits: Any,
    current_labels: Any,
    next_logits: Any,
    next_labels: Any,
) -> OccupiedSafetyAuxLossTermsV3:
    """Compute the exact V2 auxiliary math with coefficient 0.5."""

    return _v2._occupied_safety_aux_loss_with_coefficient_v2(
        current_logits,
        current_labels,
        next_logits,
        next_labels,
        coefficient=OCCUPIED_SAFETY_AUX_COEFFICIENT,
    )


def joint_training_update_v3(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV3:
    """Accumulate four inherited graphs plus half-weight ``O``, then step once."""

    return _v2._joint_training_update_with_occupied_coefficient_v2(
        model,
        optimizer,
        microbatches,
        occupied_coefficient=OCCUPIED_SAFETY_AUX_COEFFICIENT,
        accounting=accounting,
    )


def run_fixed_training_v3(
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
    """Consume the exact inherited cap with half-weight ``O`` from update one."""

    return _v2._run_fixed_training_core_v2(
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
        joint_update=joint_training_update_v3,
    )


__all__ = [
    "ACTION_ORDER",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "FrozenSurvivalRoleLabelsV1",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV3",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "OccupiedSafetyAuxLossTermsV3",
    "PREFIX_LENGTHS_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ProgressControlScoresV1",
    "REQUIRED_BATCH_KEYS",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v3",
    "occupied_safety_aux_loss_v3",
    "partition_parameters_v1",
    "run_fixed_training_v3",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_action_control_v1",
    "score_wrong_rgb_control_v1",
    "validate_accounting_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]
