#!/usr/bin/env python3
"""Lean tensor/training core for swept-progress survival joint JEPA V1.

This module deliberately does not discover data, reserve attempts, open
artifacts, or implement receipt/authority machinery.  Callers supply reviewed
labels, pairs, a narrow loader, a frozen schedule, and an initialized model.
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
        run_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1
        as _base,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


ACTION_ORDER = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
MICROBATCHES_PER_UPDATE = _base.MICROBATCHES_PER_UPDATE
MICROBATCH_SIZE = _base.MICROBATCH_SIZE
PRESENTATIONS_PER_UPDATE = _base.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000

CURRENT_RGB_KEY = _base.CURRENT_RGB_KEY
NEXT_RGB_KEY = _base.NEXT_RGB_KEY
CURRENT_LABELS_KEY = _base.CURRENT_LABELS_KEY
NEXT_LABELS_KEY = _base.NEXT_LABELS_KEY
EXECUTED_ACTION_KEY = _base.EXECUTED_ACTION_KEY
IMMEDIATE_FEASIBLE_KEY = "immediate_feasible"
PREFIX_LENGTHS_KEY = "swept_progress_prefix_lengths"
REQUIRED_BATCH_KEYS = (
    CURRENT_RGB_KEY,
    NEXT_RGB_KEY,
    CURRENT_LABELS_KEY,
    NEXT_LABELS_KEY,
    EXECUTED_ACTION_KEY,
    IMMEDIATE_FEASIBLE_KEY,
    PREFIX_LENGTHS_KEY,
)

# These are the already-reviewed optimizer/partition/accounting primitives.
ParameterPartitionV1 = _base.ParameterPartitionV1
JointTrainingAccountingV1 = _base.JointTrainingAccountingV1
partition_parameters_v1 = _base.partition_parameters_v1
build_frozen_optimizer_v1 = _base.build_frozen_optimizer_v1
validate_optimizer_v1 = _base.validate_optimizer_v1
validate_accounting_v1 = _base.validate_accounting_v1


@dataclass(frozen=True)
class FrozenSurvivalRoleLabelsV1:
    role: str
    rows: tuple[Mapping[str, Any], ...]
    state_groups: tuple[tuple[Mapping[str, Any], ...], ...]
    immediate_feasible: Any
    prefix_lengths: Any
    scene_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProgressControlScoresV1:
    predicted_latents: Any | None
    survival_logits: Any
    expected_progress_m: Any


@dataclass(frozen=True)
class JointUpdateResultV1:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_l2: Mapping[str, float]
    representation_clip_pre_l2: float
    predictor_clip_pre_l2: float
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int


def _runtime_apis() -> tuple[Any, Any, Any]:
    roots = (str(ROOT), str(ROOT / "lewm_worlds"))
    inserted = [value for value in roots if value not in sys.path]
    try:
        for value in reversed(inserted):
            sys.path.insert(0, value)
        torch = importlib.import_module("torch")
        semantic = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_joint_jepa_v1"
        )
        survival = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
        )
    finally:
        for value in inserted:
            sys.path.remove(value)
    return torch, semantic, survival


def _immediate_value(row: Mapping[str, Any]) -> bool:
    values: list[Any] = []
    if "immediate_feasible" in row:
        values.append(row["immediate_feasible"])
    if "immediate_primitive_feasible" in row:
        values.append(row["immediate_primitive_feasible"])
    primitive = row.get("immediate_primitive")
    if isinstance(primitive, Mapping) and "feasible" in primitive:
        values.append(primitive["feasible"])
    if not values or any(type(value) is not bool for value in values):
        raise PermissionError("immediate feasibility must be an exact bool")
    if any(value is not values[0] for value in values[1:]):
        raise PermissionError("immediate feasibility schema variants disagree")
    return bool(values[0])


def freeze_role_labels_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    np: Any,
) -> FrozenSurvivalRoleLabelsV1:
    """Freeze already-validated rows into state-major 9-action arrays."""

    normalized = tuple(rows)
    if not normalized or len(normalized) % len(ACTION_ORDER):
        raise PermissionError(f"{role} labels are not nine-action state groups")
    groups = tuple(
        tuple(normalized[offset : offset + len(ACTION_ORDER)])
        for offset in range(0, len(normalized), len(ACTION_ORDER))
    )
    immediate_rows: list[list[bool]] = []
    prefix_rows: list[list[int]] = []
    for state_index, group in enumerate(groups):
        first = group[0]
        if (
            [row.get("action_index") for row in group] != list(range(9))
            or [row.get("action") for row in group] != list(ACTION_ORDER)
            or any(row.get("dataset_role") != role for row in group)
            or any(row.get("role_state_index") != state_index for row in group)
            or any(
                row.get(name) != first.get(name)
                for row in group
                for name in (
                    "pair_content_sha256",
                    "current_endpoint_sha256",
                    "scene_id",
                    "family",
                )
            )
        ):
            raise PermissionError(f"{role} label state grouping changed")
        state_immediate: list[bool] = []
        state_prefixes: list[int] = []
        for row in group:
            feasible = _immediate_value(row)
            prefix = row.get("swept_progress_prefix_length")
            if type(prefix) is not int or prefix < 0 or prefix > 15:
                raise PermissionError("swept-progress prefix must be an integer 0 through 15")
            if not feasible and prefix != 0:
                raise PermissionError("an infeasible primitive cannot have safe progress")
            state_immediate.append(feasible)
            state_prefixes.append(prefix)
        immediate_rows.append(state_immediate)
        prefix_rows.append(state_prefixes)
    immediate = np.asarray(immediate_rows, dtype=np.bool_)
    prefixes = np.asarray(prefix_rows, dtype=np.int64)
    if immediate.shape != (len(groups), 9) or prefixes.shape != (len(groups), 9):
        raise PermissionError(f"{role} survival label shapes changed")
    return FrozenSurvivalRoleLabelsV1(
        role=role,
        rows=normalized,
        state_groups=groups,
        immediate_feasible=immediate,
        prefix_lengths=prefixes,
        scene_ids=tuple(str(group[0]["scene_id"]) for group in groups),
        family_ids=tuple(str(group[0]["family"]) for group in groups),
        endpoint_ids=tuple(
            str(group[0]["current_endpoint_sha256"]) for group in groups
        ),
    )


def validate_pairs_against_labels_v1(
    pairs: Sequence[Mapping[str, Any]],
    labels: FrozenSurvivalRoleLabelsV1,
) -> None:
    if len(pairs) != len(labels.state_groups):
        raise PermissionError(f"{labels.role} pair/label population changed")
    for index, (pair, group) in enumerate(zip(pairs, labels.state_groups, strict=True)):
        first = group[0]
        provenance = first.get("provenance")
        executed = (
            provenance.get("executed_pair_primitive")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            pair.get("dataset_role") != labels.role
            or pair.get("content_sha256") != first.get("pair_content_sha256")
            or pair.get("current_endpoint_sha256")
            != first.get("current_endpoint_sha256")
            or pair.get("scene_id") != first.get("scene_id")
            or pair.get("family") != first.get("family")
            or pair.get("primitive") != executed
        ):
            raise PermissionError(
                f"{labels.role} pair escaped labels at state {index}"
            )


def build_microbatch_v1(
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    labels: FrozenSurvivalRoleLabelsV1,
    indices: Sequence[int],
    device: Any,
    *,
    stage: str,
    action_order: Sequence[str] = ACTION_ORDER,
) -> dict[str, Any]:
    """Load only paired current/next RGB and their semantic rasters."""

    if len(indices) != MICROBATCH_SIZE:
        raise ValueError("one training graph requires exactly four rows")
    if tuple(action_order) != ACTION_ORDER:
        raise PermissionError("action order changed")
    selected_indices = tuple(int(index) for index in indices)
    selected_pairs = [pairs[index] for index in selected_indices]
    if any(pair.get("dataset_role") != labels.role for pair in selected_pairs):
        raise PermissionError("microbatch crossed its development role")
    torch = loader.runtime.torch
    current_rgb = torch.stack(
        [
            loader.image(
                str(pair["current_endpoint_sha256"]),
                role=labels.role,
                stage=stage,
                kind="current",
            )
            for pair in selected_pairs
        ]
    ).to(device)
    next_rgb = torch.stack(
        [
            loader.image(
                str(pair["next_endpoint_sha256"]),
                role=labels.role,
                stage=stage,
                kind="next",
            )
            for pair in selected_pairs
        ]
    ).to(device)
    current_labels = torch.stack(
        [
            loader.raster_label(
                str(pair["current_endpoint_sha256"]),
                role=labels.role,
                stage=stage,
                scope="training",
            )
            for pair in selected_pairs
        ]
    ).to(device=device, dtype=torch.long)
    next_labels = torch.stack(
        [
            loader.raster_label(
                str(pair["next_endpoint_sha256"]),
                role=labels.role,
                stage=stage,
                scope="training",
            )
            for pair in selected_pairs
        ]
    ).to(device=device, dtype=torch.long)
    executed = torch.tensor(
        [ACTION_ORDER.index(str(pair["primitive"])) for pair in selected_pairs],
        dtype=torch.long,
        device=device,
    )
    immediate = torch.tensor(
        labels.immediate_feasible[list(selected_indices)],
        dtype=torch.bool,
        device=device,
    )
    prefixes = torch.tensor(
        labels.prefix_lengths[list(selected_indices)],
        dtype=torch.long,
        device=device,
    )
    return {
        CURRENT_RGB_KEY: current_rgb,
        NEXT_RGB_KEY: next_rgb,
        CURRENT_LABELS_KEY: current_labels,
        NEXT_LABELS_KEY: next_labels,
        EXECUTED_ACTION_KEY: executed,
        IMMEDIATE_FEASIBLE_KEY: immediate,
        PREFIX_LENGTHS_KEY: prefixes,
    }


def _prediction_parts(prediction: Any) -> tuple[Any, Any]:
    try:
        return prediction.predicted_latents, prediction.survival_logits
    except AttributeError as error:
        raise TypeError("model survival prediction API changed") from error


def _control_scores(predicted: Any | None, logits: Any) -> ProgressControlScoresV1:
    torch, _, survival = _runtime_apis()
    scores = survival.survival_scores_v1(logits)
    result = ProgressControlScoresV1(
        predicted_latents=predicted,
        survival_logits=logits,
        expected_progress_m=scores.expected_progress_m,
    )
    values = (result.survival_logits, result.expected_progress_m)
    if predicted is not None:
        values = (predicted, *values)
    if not all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("control scores are absent or nonfinite")
    return result


def score_full_control_v1(model: Any, current_latent: Any) -> ProgressControlScoresV1:
    prediction = model.predict_all_actions_with_survival(current_latent)
    predicted, logits = _prediction_parts(prediction)
    return _control_scores(predicted, logits)


def score_shuffled_action_control_v1(
    model: Any,
    predicted_latents: Any,
) -> ProgressControlScoresV1:
    """Shift predicted slot ``a+1`` into candidate slot ``a`` before pooling."""

    torch, _, _ = _runtime_apis()
    if predicted_latents.ndim != 5 or predicted_latents.shape[1] != 9:
        raise ValueError("predicted latents must have shape (B,9,64,H,W)")
    indices = (torch.arange(9, device=predicted_latents.device) + 1) % 9
    shuffled = predicted_latents.index_select(1, indices)
    logits = model.predictor.swept_progress_head(shuffled)
    return _control_scores(shuffled, logits)


def score_persistence_control_v1(
    model: Any,
    current_latent: Any,
) -> ProgressControlScoresV1:
    """Apply candidate masks/head to the unchanged current latent."""

    if current_latent.ndim != 4 or current_latent.shape[1] != 64:
        raise ValueError("current latent must have shape (B,64,H,W)")
    repeated = current_latent[:, None].expand(-1, 9, -1, -1, -1)
    logits = model.predictor.swept_progress_head(repeated)
    return _control_scores(None, logits)


def score_wrong_rgb_control_v1(model: Any, wrong_rgb: Any) -> ProgressControlScoresV1:
    """Encode a caller-bound wrong RGB and otherwise run the full arm."""

    return score_full_control_v1(model, model.encode_online(wrong_rgb))


def _validate_microbatches(torch: Any, microbatches: Sequence[Mapping[str, Any]]) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("one update requires exactly four microbatches")
    for index, batch in enumerate(microbatches):
        missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f"microbatch {index} is missing {missing}")
        if any(
            not isinstance(batch[key], torch.Tensor)
            or batch[key].shape[0] != MICROBATCH_SIZE
            for key in REQUIRED_BATCH_KEYS
        ):
            raise ValueError(f"microbatch {index} tensors must contain four rows")
        if batch[IMMEDIATE_FEASIBLE_KEY].shape != (4, 9):
            raise ValueError("immediate-feasible batch must have shape (4,9)")
        if batch[PREFIX_LENGTHS_KEY].shape != (4, 9):
            raise ValueError("prefix-length batch must have shape (4,9)")


def joint_training_update_v1(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV1:
    """Accumulate four joint graphs, then clip/step/EMA exactly once."""

    torch, semantic_api, survival_api = _runtime_apis()
    _validate_microbatches(torch, microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    state = JointTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with training accounting")

    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in ("S", "P", "U", "R", "L")}
    active_ranking = 0
    eligible_pairs = 0
    supervised_decisions = 0
    for batch in microbatches:
        current_latent = model.encode_online(batch[CURRENT_RGB_KEY])
        next_latent = model.encode_online(batch[NEXT_RGB_KEY])
        semantic = semantic_api.semantic_loss_v1(
            model.semantic_logits_from_latent(current_latent),
            batch[CURRENT_LABELS_KEY],
            model.semantic_logits_from_latent(next_latent),
            batch[NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(current_latent)
        predicted, survival_logits = _prediction_parts(prediction)
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
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("joint loss", joint.loss),
        ):
            _base._finite_tensor(torch, value, name)
        (joint.loss / MICROBATCHES_PER_UPDATE).backward()
        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("L", joint.loss),
        ):
            sums[name] += _base._scalar(value)
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    gradient_l2 = {
        "encoder": _base._gradient_l2(torch, partition.encoder, "encoder"),
        "lift_semantic": _base._gradient_l2(
            torch, partition.lift_semantic, "lift/semantic"
        ),
        "predictor": _base._gradient_l2(torch, partition.predictor, "predictor"),
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
        _base._finite_tensor(torch, parameter, "online parameter")
    model.update_target_ema_after_optimizer_step()
    if int(model.ema_update_count.item()) != ema_before + 1:
        raise RuntimeError("EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")

    advanced = _base._advanced_accounting(state)
    if advanced.ema_steps != int(model.ema_update_count.item()):
        raise RuntimeError("post-update EMA count disagrees with accounting")
    return JointUpdateResultV1(
        accounting=advanced,
        mean_losses={
            name: value / MICROBATCHES_PER_UPDATE for name, value in sums.items()
        },
        gradient_l2=gradient_l2,
        representation_clip_pre_l2=_base._scalar(representation_pre),
        predictor_clip_pre_l2=_base._scalar(predictor_pre),
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
    )


def run_fixed_training_v1(
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
    """Consume one exact 1,000-update/16,000-presentation schedule."""

    if maximum_updates != MAXIMUM_UPDATES or len(schedule) != MAXIMUM_PRESENTATIONS:
        raise PermissionError("training cap or schedule length changed")
    if tuple(action_order) != ACTION_ORDER:
        raise PermissionError("action order changed")
    validate_pairs_against_labels_v1(train_pairs, train_labels)
    accounting = JointTrainingAccountingV1()
    trace: list[dict[str, Any]] = []
    active_ranking = eligible_pairs = supervised_decisions = 0
    min_gradients = {name: math.inf for name in ("encoder", "lift_semantic", "predictor")}
    max_gradients = {name: 0.0 for name in min_gradients}
    for update in range(1, MAXIMUM_UPDATES + 1):
        start = (update - 1) * PRESENTATIONS_PER_UPDATE
        update_indices = schedule[start : start + PRESENTATIONS_PER_UPDATE]
        if len(update_indices) != PRESENTATIONS_PER_UPDATE:
            raise RuntimeError("training schedule exhausted before update 1000")
        microbatches = [
            build_microbatch_v1(
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
        result = joint_training_update_v1(
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


__all__ = [
    "ACTION_ORDER",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "FrozenSurvivalRoleLabelsV1",
    "IMMEDIATE_FEASIBLE_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV1",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "PREFIX_LENGTHS_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ProgressControlScoresV1",
    "REQUIRED_BATCH_KEYS",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v1",
    "partition_parameters_v1",
    "run_fixed_training_v1",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_action_control_v1",
    "score_wrong_rgb_control_v1",
    "validate_accounting_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]
