#!/usr/bin/env python3
"""Lean source-only training core for projective-support corridor joint JEPA V1.

Importing this script is intentionally Torch/NumPy-free.  Runtime functions
accept already-materialized tensors and lazily import the frozen model/scoring
APIs; they never discover or open data, checkpoints, generated output, or
authority receipts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import dataclasses
import hashlib
import importlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_post_action_projective_support_corridor_contract_v1.py"
)
AUTHORITY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_post_action_projective_support_source_authority_v1.py"
)
EXECUTION_BINDING_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "execution_binding_v2_2026-07-28.json"
)
LABELS_RELATIVE_PATH = (
    "lewm/benchmarks/go2_post_action_projective_support_labels_v1.py"
)
METRICS_RELATIVE_PATH = (
    "lewm/benchmarks/go2_post_action_projective_support_metrics_v1.py"
)
DIRECT_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"
)
MATCHED_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
)
EXPERIMENT_ARM = "rgb_post_action_projective_support_corridor_joint_jepa_v1"
MICROBATCHES_PER_UPDATE = 4
MICROBATCH_SIZE = 4
PRESENTATIONS_PER_UPDATE = 16
CURRENT_RGB_KEY = "current_rgb"
NEXT_RGB_KEY = "next_rgb"
CURRENT_LABELS_KEY = "current_labels"
NEXT_LABELS_KEY = "next_labels"
EXECUTED_ACTION_KEY = "executed_action_indices"
STATION_SAFE_KEY = "corridor_station_safe"
REQUIRED_BATCH_KEYS = (
    CURRENT_RGB_KEY,
    NEXT_RGB_KEY,
    CURRENT_LABELS_KEY,
    NEXT_LABELS_KEY,
    EXECUTED_ACTION_KEY,
    STATION_SAFE_KEY,
)


@dataclass(frozen=True)
class ParameterPartitionV1:
    encoder: tuple[Any, ...]
    lift_semantic: tuple[Any, ...]
    predictor: tuple[Any, ...]
    target: tuple[Any, ...]
    names: Mapping[str, tuple[str, ...]]

    @property
    def representation(self) -> tuple[Any, ...]:
        return self.encoder + self.lift_semantic

    @property
    def online(self) -> tuple[Any, ...]:
        return self.representation + self.predictor


@dataclass(frozen=True)
class JointTrainingAccountingV1:
    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0


@dataclass(frozen=True)
class BatchControlScoresV1:
    predicted_latents: Any | None
    semantic_logits: Any
    free_log_odds: Any
    station_logits: Any
    station_probabilities: Any
    prefix_utility: Any


@dataclass(frozen=True)
class JointUpdateResultV1:
    accounting: JointTrainingAccountingV1
    mean_losses: Mapping[str, float]
    gradient_l2: Mapping[str, float]
    representation_clip_pre_l2: float
    predictor_clip_pre_l2: float
    ranking_active_microbatches: int
    ranking_eligible_rows: int
    ranking_eligible_pairs: int


@dataclass(frozen=True)
class FrozenRoleLabelsV1:
    role: str
    rows: tuple[Mapping[str, Any], ...]
    state_groups: tuple[tuple[Mapping[str, Any], ...], ...]
    station_safe: Any
    immediate_feasible: Any
    blind_bridge_feasible: Any
    scene_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class RoleScorePopulationV1:
    probabilities: Mapping[str, Any]
    semantic_confusion: Any
    rough_semantic_confusion: Any
    current_latents_nonconstant: bool
    paired_latents_nonconstant: bool
    current_and_paired_latents_nonidentical: bool
    all_values_finite: bool


class ExperimentArmRawInputsProxyV1:
    """Relabel the reviewed narrow loader's ledger arm without widening access."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def read_rgb(
        self,
        relative: str,
        expected_sha256: str,
        *,
        role: str,
        arm: str,
        stage: str,
    ) -> bytes:
        del arm
        return self._target.read_rgb(
            relative,
            expected_sha256,
            role=role,
            arm=EXPERIMENT_ARM,
            stage=stage,
        )

    def _shard(self, endpoint: Mapping[str, Any], *, arm: str, stage: str) -> Any:
        del arm
        return self._target._shard(
            endpoint,
            arm=EXPERIMENT_ARM,
            stage=stage,
        )

    def _row_array(
        self,
        endpoint: Mapping[str, Any],
        shard: Mapping[str, Any],
        filename: str,
        *,
        arm: str,
        stage: str,
    ) -> Any:
        del arm
        return self._target._row_array(
            endpoint,
            shard,
            filename,
            arm=EXPERIMENT_ARM,
            stage=stage,
        )


def _runtime_apis() -> tuple[Any, Any, Any]:
    required_roots = (str(ROOT), str(ROOT / "lewm_worlds"))
    inserted = [value for value in required_roots if value not in sys.path]
    try:
        for value in reversed(inserted):
            sys.path.insert(0, value)
        torch = importlib.import_module("torch")
        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1"
        )
        scoring_api = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_joint_jepa_v1"
        )
    finally:
        for value in inserted:
            sys.path.remove(value)
    return torch, model_api, scoring_api


def partition_parameters_v1(model: Any) -> ParameterPartitionV1:
    """Partition every model parameter into the four frozen, disjoint roles."""

    groups: dict[str, list[Any]] = {
        "encoder": [],
        "lift_semantic": [],
        "predictor": [],
        "target": [],
    }
    names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith(("bev_lift.", "semantic_head.")):
            group = "lift_semantic"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith(("target_encoder.", "target_bev_lift.")):
            group = "target"
        else:
            raise RuntimeError(f"unregistered model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)
    if any(not values for values in groups.values()):
        raise RuntimeError("parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("parameter partition overlaps")
    if any(value.requires_grad for value in groups["target"]):
        raise RuntimeError("EMA target parameter is trainable")
    if any(
        not value.requires_grad
        for group in ("encoder", "lift_semantic", "predictor")
        for value in groups[group]
    ):
        raise RuntimeError("online parameter is frozen")
    return ParameterPartitionV1(
        encoder=tuple(groups["encoder"]),
        lift_semantic=tuple(groups["lift_semantic"]),
        predictor=tuple(groups["predictor"]),
        target=tuple(groups["target"]),
        names={name: tuple(values) for name, values in names.items()},
    )


def build_frozen_optimizer_v1(
    model_or_partition: Any,
) -> Any:
    """Construct the sole three-group AdamW optimizer with frozen hyperparameters."""

    torch, _, _ = _runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV1)
        else partition_parameters_v1(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {"name": "encoder", "params": list(partition.encoder), "lr": 1e-4},
            {
                "name": "lift_semantic",
                "params": list(partition.lift_semantic),
                "lr": 3e-4,
            },
            {"name": "predictor", "params": list(partition.predictor), "lr": 3e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    validate_optimizer_v1(optimizer, partition)
    return optimizer


def validate_optimizer_v1(
    optimizer: Any,
    partition: ParameterPartitionV1,
) -> None:
    """Fail closed if optimizer type, order, membership, or settings changed."""

    expected = (
        ("encoder", partition.encoder, 1e-4),
        ("lift_semantic", partition.lift_semantic, 3e-4),
        ("predictor", partition.predictor, 3e-4),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("optimizer must be the sole three-group AdamW")
    all_ids: list[int] = []
    for observed, (name, parameters, learning_rate) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        observed_parameters = tuple(observed["params"])
        all_ids.extend(id(value) for value in observed_parameters)
        if (
            observed.get("name") != name
            or tuple(map(id, observed_parameters)) != tuple(map(id, parameters))
            or float(observed["lr"]) != learning_rate
            or tuple(observed["betas"]) != (0.9, 0.999)
            or float(observed["eps"]) != 1e-8
            or float(observed["weight_decay"]) != 1e-4
        ):
            raise RuntimeError(f"optimizer group {name!r} changed")
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(map(id, partition.online)):
        raise RuntimeError("optimizer membership is incomplete or overlapping")


def validate_accounting_v1(accounting: JointTrainingAccountingV1) -> None:
    values = tuple(accounting.__dict__.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("training accounting values must be nonnegative integers")
    updates = accounting.updates
    if (
        accounting.presentations != updates * PRESENTATIONS_PER_UPDATE
        or accounting.microbatch_graphs != updates * MICROBATCHES_PER_UPDATE
        or accounting.backward_calls != updates * MICROBATCHES_PER_UPDATE
        or accounting.optimizer_steps != updates
        or accounting.ema_steps != updates
        or accounting.predictor_forwards != updates * MICROBATCHES_PER_UPDATE
        or accounting.predictor_objectives != updates * MICROBATCHES_PER_UPDATE
    ):
        raise RuntimeError("joint training accounting is inconsistent")


def _advanced_accounting(
    accounting: JointTrainingAccountingV1,
) -> JointTrainingAccountingV1:
    result = JointTrainingAccountingV1(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=accounting.microbatch_graphs + MICROBATCHES_PER_UPDATE,
        backward_calls=accounting.backward_calls + MICROBATCHES_PER_UPDATE,
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
        predictor_forwards=accounting.predictor_forwards + MICROBATCHES_PER_UPDATE,
        predictor_objectives=accounting.predictor_objectives + MICROBATCHES_PER_UPDATE,
    )
    validate_accounting_v1(result)
    return result


def _finite_tensor(torch: Any, value: Any, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is absent or nonfinite")


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite scalar")
    return result


def _gradient_l2(torch: Any, parameters: Sequence[Any], name: str) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"{name} gradient is absent or nonfinite")
        total = total + gradient.detach().double().square().sum()
    norm = _scalar(total.sqrt())
    if norm <= 0.0:
        raise FloatingPointError(f"{name} gradient is zero")
    return norm


def _validate_microbatches(torch: Any, microbatches: Sequence[Mapping[str, Any]]) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("one update requires exactly four microbatches")
    for index, batch in enumerate(microbatches):
        missing = [name for name in REQUIRED_BATCH_KEYS if name not in batch]
        if missing:
            raise KeyError(f"microbatch {index} is missing {missing}")
        for name in REQUIRED_BATCH_KEYS:
            value = batch[name]
            if not isinstance(value, torch.Tensor) or value.shape[0] != MICROBATCH_SIZE:
                raise ValueError(f"microbatch {index} {name} must contain four rows")


def _control_result(predicted: Any | None, logits: Any, scores: Any) -> BatchControlScoresV1:
    return BatchControlScoresV1(
        predicted_latents=predicted,
        semantic_logits=logits,
        free_log_odds=scores.free_log_odds,
        station_logits=scores.station_logits,
        station_probabilities=scores.station_probabilities,
        prefix_utility=scores.prefix_utility,
    )


def _control_outputs_finite_v1(
    torch: Any,
    scores: BatchControlScoresV1,
    *,
    predicted_latents_expected: bool,
) -> bool:
    if (scores.predicted_latents is not None) != predicted_latents_expected:
        return False
    values = (
        scores.semantic_logits,
        scores.free_log_odds,
        scores.station_logits,
        scores.station_probabilities,
        scores.prefix_utility,
    )
    if scores.predicted_latents is not None:
        values = (scores.predicted_latents, *values)
    return all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in values
    )


def score_full_control_v1(
    model: Any,
    current_latent: Any,
    *,
    full_masks: Any | None = None,
) -> BatchControlScoresV1:
    """Score action-conditioned predicted latents with predicted-next masks."""

    _, _, scoring = _runtime_apis()
    masks = scoring.build_full_corridor_masks_v1() if full_masks is None else full_masks
    predicted, logits = scoring.predict_and_decode_all_actions_v1(model, current_latent)
    scores = scoring.corridor_scores_from_semantic_logits_v1(logits, masks)
    return _control_result(predicted, logits, scores)


def score_persistence_control_v1(
    model: Any,
    current_latent: Any,
    *,
    persistence_masks: Any | None = None,
) -> BatchControlScoresV1:
    """Score the current semantic field with coordinate-matched masks."""

    _, _, scoring = _runtime_apis()
    masks = (
        scoring.build_persistence_corridor_masks_v1()
        if persistence_masks is None
        else persistence_masks
    )
    logits = model.semantic_logits_from_latent(current_latent)
    scores = scoring.corridor_scores_from_semantic_logits_v1(logits, masks)
    return _control_result(None, logits, scores)


def score_shuffled_control_v1(
    model: Any,
    predicted_latents: Any,
    *,
    full_masks: Any | None = None,
) -> BatchControlScoresV1:
    """Score slot ``(a+1) mod 9`` while retaining candidate ``a`` masks/labels."""

    torch, _, scoring = _runtime_apis()
    if predicted_latents.ndim != 5 or predicted_latents.shape[1] != 9:
        raise ValueError("predicted latents must have shape (B,9,64,64,64)")
    indices = (torch.arange(9, device=predicted_latents.device) + 1) % 9
    shuffled = predicted_latents.index_select(1, indices)
    logits = scoring.decode_all_action_semantic_logits_v1(model, shuffled)
    masks = scoring.build_full_corridor_masks_v1() if full_masks is None else full_masks
    scores = scoring.corridor_scores_from_semantic_logits_v1(logits, masks)
    return _control_result(shuffled, logits, scores)


def joint_training_update_v1(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    full_masks: Any | None = None,
    accounting: JointTrainingAccountingV1 | None = None,
) -> JointUpdateResultV1:
    """Run four graphs/backwards, two clips, one AdamW step, and one EMA step."""

    torch, _, scoring = _runtime_apis()
    _validate_microbatches(torch, microbatches)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    state = JointTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    if not hasattr(model, "ema_update_count"):
        raise RuntimeError("model has no EMA update counter")
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with training accounting")

    masks = scoring.build_full_corridor_masks_v1() if full_masks is None else full_masks
    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in ("S", "P", "Q", "R", "L")}
    eligible_rows = 0
    eligible_pairs = 0
    active_r_microbatches = 0
    for batch in microbatches:
        current_latent = model.encode_online(batch[CURRENT_RGB_KEY])
        next_latent = model.encode_online(batch[NEXT_RGB_KEY])
        current_logits = model.semantic_logits_from_latent(current_latent)
        next_logits = model.semantic_logits_from_latent(next_latent)
        semantic = scoring.semantic_loss_v1(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        predicted, predicted_logits = scoring.predict_and_decode_all_actions_v1(
            model, current_latent
        )
        corridor = scoring.corridor_scores_from_semantic_logits_v1(
            predicted_logits, masks
        )
        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
        joint = scoring.joint_microbatch_loss_v1(
            semantic_loss=semantic.loss,
            predicted_latents=predicted,
            executed_action_indices=batch[EXECUTED_ACTION_KEY],
            ema_current_latent=ema_current,
            ema_next_latent=ema_next,
            station_logits=corridor.station_logits,
            station_safe=batch[STATION_SAFE_KEY],
        )
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("predicted latent", predicted),
            ("semantic logits", predicted_logits),
            ("station logits", corridor.station_logits),
            ("joint loss", joint.loss),
        ):
            _finite_tensor(torch, value, name)
        (joint.loss / MICROBATCHES_PER_UPDATE).backward()
        for name, value in (
            ("S", joint.semantic),
            ("P", joint.persistence),
            ("Q", joint.corridor_binary),
            ("R", joint.prefix_ranking),
            ("L", joint.loss),
        ):
            sums[name] += _scalar(value)
        eligible_rows += int(joint.ranking_terms.eligible_row_count.item())
        eligible_pairs += int(joint.ranking_terms.eligible_pair_count.item())
        active_r_microbatches += int(
            int(joint.ranking_terms.eligible_row_count.item()) > 0
        )

    gradient_l2 = {
        "encoder": _gradient_l2(torch, partition.encoder, "encoder"),
        "lift_semantic": _gradient_l2(
            torch, partition.lift_semantic, "lift/semantic"
        ),
        "predictor": _gradient_l2(torch, partition.predictor, "predictor"),
    }
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")
    representation_pre = torch.nn.utils.clip_grad_norm_(
        partition.representation,
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    predictor_pre = torch.nn.utils.clip_grad_norm_(
        partition.predictor,
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    for parameter in partition.online:
        _finite_tensor(torch, parameter, "online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target received a gradient")

    advanced = _advanced_accounting(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update EMA count disagrees with accounting")
    return JointUpdateResultV1(
        accounting=advanced,
        mean_losses={
            name: value / MICROBATCHES_PER_UPDATE for name, value in sums.items()
        },
        gradient_l2=gradient_l2,
        representation_clip_pre_l2=_scalar(representation_pre),
        predictor_clip_pre_l2=_scalar(predictor_pre),
        ranking_active_microbatches=active_r_microbatches,
        ranking_eligible_rows=eligible_rows,
        ranking_eligible_pairs=eligible_pairs,
    )


def freeze_role_labels_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    np: Any,
) -> FrozenRoleLabelsV1:
    """Convert one already-validated role file into state-major frozen arrays."""

    normalized = tuple(rows)
    if not normalized or len(normalized) % 9:
        raise PermissionError(f"{role} labels are not nine-action state groups")
    groups = tuple(
        tuple(normalized[offset : offset + 9])
        for offset in range(0, len(normalized), 9)
    )
    for state_index, group in enumerate(groups):
        first = group[0]
        if (
            [row.get("action_index") for row in group] != list(range(9))
            or any(row.get("dataset_role") != role for row in group)
            or any(row.get("role_state_index") != state_index for row in group)
            or any(
                row.get("pair_content_sha256") != first.get("pair_content_sha256")
                or row.get("current_endpoint_sha256")
                != first.get("current_endpoint_sha256")
                or row.get("scene_id") != first.get("scene_id")
                or row.get("family") != first.get("family")
                for row in group
            )
        ):
            raise PermissionError(f"{role} label state grouping changed")
    station_safe = np.asarray(
        [[row["station_safe"] for row in group] for group in groups],
        dtype=np.bool_,
    )
    immediate = np.asarray(
        [
            [bool(row["immediate_primitive"]["feasible"]) for row in group]
            for group in groups
        ],
        dtype=np.bool_,
    )
    blind = np.asarray(
        [
            [bool(row["blind_bridge"]["feasible"]) for row in group]
            for group in groups
        ],
        dtype=np.bool_,
    )
    if station_safe.shape != (len(groups), 9, 11):
        raise PermissionError(f"{role} station labels changed shape")
    return FrozenRoleLabelsV1(
        role=role,
        rows=normalized,
        state_groups=groups,
        station_safe=station_safe,
        immediate_feasible=immediate,
        blind_bridge_feasible=blind,
        scene_ids=tuple(str(group[0]["scene_id"]) for group in groups),
        family_ids=tuple(str(group[0]["family"]) for group in groups),
        endpoint_ids=tuple(
            str(group[0]["current_endpoint_sha256"]) for group in groups
        ),
    )


def validate_pairs_against_labels_v1(
    pairs: Sequence[Mapping[str, Any]],
    labels: FrozenRoleLabelsV1,
) -> None:
    if len(pairs) != len(labels.state_groups):
        raise PermissionError(f"{labels.role} pair/label population changed")
    for index, (pair, group) in enumerate(zip(pairs, labels.state_groups, strict=True)):
        first = group[0]
        if (
            pair.get("dataset_role") != labels.role
            or pair.get("content_sha256") != first.get("pair_content_sha256")
            or pair.get("current_endpoint_sha256")
            != first.get("current_endpoint_sha256")
            or pair.get("scene_id") != first.get("scene_id")
            or pair.get("family") != first.get("family")
            or pair.get("primitive")
            != first.get("provenance", {}).get("executed_pair_primitive")
        ):
            raise PermissionError(f"{labels.role} pair escaped labels at state {index}")


def build_microbatch_v1(
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    labels: FrozenRoleLabelsV1,
    indices: Sequence[int],
    device: Any,
    *,
    stage: str,
    action_order: Sequence[str],
) -> dict[str, Any]:
    """Build current/next RGB+raster batches without fixed-negative RGB access."""

    if len(indices) != MICROBATCH_SIZE:
        raise ValueError("one training graph requires exactly four rows")
    selected_pairs = [pairs[int(index)] for index in indices]
    selected_groups = [labels.state_groups[int(index)] for index in indices]
    if any(pair.get("dataset_role") != labels.role for pair in selected_pairs):
        raise PermissionError("microbatch crossed its development role")
    torch = loader.runtime.torch
    current_rgb = torch.stack([
        loader.image(
            str(pair["current_endpoint_sha256"]),
            role=labels.role,
            stage=stage,
            kind="current",
        )
        for pair in selected_pairs
    ]).to(device)
    next_rgb = torch.stack([
        loader.image(
            str(pair["next_endpoint_sha256"]),
            role=labels.role,
            stage=stage,
            kind="next",
        )
        for pair in selected_pairs
    ]).to(device)
    current_labels = torch.stack([
        loader.raster_label(
            str(pair["current_endpoint_sha256"]),
            role=labels.role,
            stage=stage,
            scope="training",
        )
        for pair in selected_pairs
    ]).to(device=device, dtype=torch.long)
    next_labels = torch.stack([
        loader.raster_label(
            str(pair["next_endpoint_sha256"]),
            role=labels.role,
            stage=stage,
            scope="training",
        )
        for pair in selected_pairs
    ]).to(device=device, dtype=torch.long)
    executed = torch.tensor(
        [action_order.index(str(pair["primitive"])) for pair in selected_pairs],
        dtype=torch.long,
        device=device,
    )
    station_safe = torch.tensor(
        [[row["station_safe"] for row in group] for group in selected_groups],
        dtype=torch.float32,
        device=device,
    )
    return {
        CURRENT_RGB_KEY: current_rgb,
        NEXT_RGB_KEY: next_rgb,
        CURRENT_LABELS_KEY: current_labels,
        NEXT_LABELS_KEY: next_labels,
        EXECUTED_ACTION_KEY: executed,
        STATION_SAFE_KEY: station_safe,
    }


def score_role_population_v1(
    model: Any,
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    labels: FrozenRoleLabelsV1,
    wrong_rgb_by_endpoint: Mapping[tuple[str, str, str], str],
    action_prior: Any,
    device: Any,
    *,
    stage: str,
    np: Any,
    full_masks: Any | None = None,
    persistence_masks: Any | None = None,
) -> RoleScorePopulationV1:
    """Score all five frozen arms and semantic retention for one role."""

    validate_pairs_against_labels_v1(pairs, labels)
    if tuple(action_prior.shape) != (9, 11):
        raise ValueError("action prior must have shape (9,11)")
    torch = loader.runtime.torch
    probabilities: dict[str, list[Any]] = {
        name: []
        for name in (
            "full",
            "coordinate_matched_persistence",
            "shuffled_action",
            "wrong_rgb",
        )
    }
    semantic_confusion = torch.zeros((3, 3), dtype=torch.long)
    rough_confusion = torch.zeros((3, 3), dtype=torch.long)
    current_reference: Any | None = None
    paired_reference: Any | None = None
    current_nonconstant = False
    paired_nonconstant = False
    nonidentical = False
    finite = True
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), MICROBATCH_SIZE):
                selected_pairs = pairs[start : start + MICROBATCH_SIZE]
                current_rgb = torch.stack([
                    loader.image(
                        str(pair["current_endpoint_sha256"]),
                        role=labels.role,
                        stage=stage,
                        kind="current",
                    )
                    for pair in selected_pairs
                ]).to(device)
                next_rgb = torch.stack([
                    loader.image(
                        str(pair["next_endpoint_sha256"]),
                        role=labels.role,
                        stage=stage,
                        kind="next",
                    )
                    for pair in selected_pairs
                ]).to(device)
                current_labels = torch.stack([
                    loader.raster_label(
                        str(pair["current_endpoint_sha256"]),
                        role=labels.role,
                        stage=stage,
                        scope="observation",
                    )
                    for pair in selected_pairs
                ]).to(device=device, dtype=torch.long)
                next_labels = torch.stack([
                    loader.raster_label(
                        str(pair["next_endpoint_sha256"]),
                        role=labels.role,
                        stage=stage,
                        scope="observation",
                    )
                    for pair in selected_pairs
                ]).to(device=device, dtype=torch.long)
                wrong_rgb = torch.stack([
                    loader.image(
                        wrong_rgb_by_endpoint[
                            (
                                labels.role,
                                str(pair["scene_id"]),
                                str(pair["current_endpoint_sha256"]),
                            )
                        ],
                        role=labels.role,
                        stage=f"{stage}_wrong_rgb",
                        kind="endpoint",
                    )
                    for pair in selected_pairs
                ]).to(device)

                current_latent = model.encode_online(current_rgb)
                paired_latent = model.encode_online(next_rgb)
                current_logits = model.semantic_logits_from_latent(current_latent)
                paired_logits = model.semantic_logits_from_latent(paired_latent)
                full = score_full_control_v1(
                    model, current_latent, full_masks=full_masks
                )
                probabilities["full"].append(
                    full.station_probabilities.detach().cpu().numpy()
                )
                persistence = score_persistence_control_v1(
                    model,
                    current_latent,
                    persistence_masks=persistence_masks,
                )
                probabilities["coordinate_matched_persistence"].append(
                    persistence.station_probabilities.detach().cpu().numpy()
                )
                shuffled = score_shuffled_control_v1(
                    model, full.predicted_latents, full_masks=full_masks
                )
                probabilities["shuffled_action"].append(
                    shuffled.station_probabilities.detach().cpu().numpy()
                )
                intermediate_controls_finite = all(
                    (
                        _control_outputs_finite_v1(
                            torch, full, predicted_latents_expected=True
                        ),
                        _control_outputs_finite_v1(
                            torch, persistence, predicted_latents_expected=False
                        ),
                        _control_outputs_finite_v1(
                            torch, shuffled, predicted_latents_expected=True
                        ),
                    )
                )
                del shuffled, persistence, full
                wrong_latent = model.encode_online(wrong_rgb)
                wrong = score_full_control_v1(
                    model, wrong_latent, full_masks=full_masks
                )
                probabilities["wrong_rgb"].append(
                    wrong.station_probabilities.detach().cpu().numpy()
                )

                for logits, target in (
                    (current_logits, current_labels),
                    (paired_logits, next_labels),
                ):
                    predicted = logits.argmax(dim=1)
                    codes = (target * 3 + predicted).reshape(-1)
                    semantic_confusion += torch.bincount(
                        codes, minlength=9
                    ).reshape(3, 3).cpu()
                    rough_rows = [
                        offset
                        for offset, pair in enumerate(selected_pairs)
                        if pair.get("family") == "rough_local_dynamics"
                    ]
                    if rough_rows:
                        index = torch.tensor(
                            rough_rows, dtype=torch.long, device=device
                        )
                        rough_codes = (
                            target.index_select(0, index) * 3
                            + predicted.index_select(0, index)
                        ).reshape(-1)
                        rough_confusion += torch.bincount(
                            rough_codes, minlength=9
                        ).reshape(3, 3).cpu()

                for row in current_latent:
                    if current_reference is None:
                        current_reference = row.detach().clone()
                    else:
                        current_nonconstant = current_nonconstant or not torch.equal(
                            current_reference, row
                        )
                for row in paired_latent:
                    if paired_reference is None:
                        paired_reference = row.detach().clone()
                    else:
                        paired_nonconstant = paired_nonconstant or not torch.equal(
                            paired_reference, row
                        )
                nonidentical = nonidentical or not torch.equal(
                    current_latent, paired_latent
                )
                finite = finite and intermediate_controls_finite and all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        current_latent,
                        paired_latent,
                        current_logits,
                        paired_logits,
                        wrong_latent,
                    )
                ) and _control_outputs_finite_v1(
                    torch, wrong, predicted_latents_expected=True
                )
                del wrong, wrong_latent
    finally:
        model.train(was_training)
    assembled = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in probabilities.items()
    }
    assembled["action_prior"] = np.broadcast_to(
        np.asarray(action_prior, dtype=np.float64),
        (len(pairs), 9, 11),
    ).copy()
    if any(value.shape != (len(pairs), 9, 11) for value in assembled.values()):
        raise RuntimeError(f"{labels.role} control score population changed")
    finite = finite and all(
        bool(np.isfinite(value).all()) for value in assembled.values()
    )
    return RoleScorePopulationV1(
        probabilities=assembled,
        semantic_confusion=semantic_confusion.numpy(),
        rough_semantic_confusion=rough_confusion.numpy(),
        current_latents_nonconstant=current_nonconstant,
        paired_latents_nonconstant=paired_nonconstant,
        current_and_paired_latents_nonidentical=nonidentical,
        all_values_finite=finite,
    )


def _source_module_v1(name: str, path: Path, *, repository_root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    roots = (str(repository_root), str(repository_root / "lewm_worlds"))
    inserted = [value for value in roots if value not in sys.path]
    try:
        for value in reversed(inserted):
            sys.path.insert(0, value)
        spec.loader.exec_module(module)
    finally:
        for value in inserted:
            sys.path.remove(value)
    return module


def _read_regular_v1(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> bytes:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise PermissionError(f"not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"file changed while read: {path}")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError(f"SHA-256 changed: {path}")
    if expected_byte_count is not None and len(raw) != expected_byte_count:
        raise PermissionError(f"byte count changed: {path}")
    return raw


def _read_bound_v1(
    repository_root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str,
) -> bytes:
    if (
        type(binding) is not dict
        or binding.get("path") != expected_path
        or type(binding.get("file_sha256")) is not str
        or type(binding.get("byte_count")) is not int
    ):
        raise PermissionError(f"artifact binding changed: {expected_path}")
    return _read_regular_v1(
        repository_root / expected_path,
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
    )


def _read_bound_tracked_v1(
    progress: dict[str, Any],
    repository_root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str,
) -> bytes:
    progress.setdefault("authorized_input_open_attempts", []).append(expected_path)
    raw = _read_bound_v1(
        repository_root, binding, expected_path=expected_path
    )
    progress.setdefault("authorized_input_open_successes", []).append(expected_path)
    return raw


def _write_exclusive_v1(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _publish_json_v1(
    contract: Any,
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive_v1(path, raw)
    return value, raw


def _load_published_json_v1(
    contract: Any,
    path: Path,
) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"published receipt is not a regular file: {path}")
    raw = path.read_bytes()
    return contract.parse_canonical_json(raw, name=path.name), raw


def _publish_or_reuse_json_v1(
    contract: Any,
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Publish once, or reuse only an existing byte-identical terminal receipt."""

    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    if path.exists() or path.is_symlink():
        existing_value, existing_raw = _load_published_json_v1(contract, path)
        if existing_raw != raw or existing_value != value:
            raise FileExistsError(f"conflicting write-once artifact exists: {path}")
        return existing_value, existing_raw
    _write_exclusive_v1(path, raw)
    return value, raw


def _artifact_binding_v1(
    relative: str,
    value: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": str(value["content_sha256"]),
        "byte_count": len(raw),
    }


def validate_execution_envelope_v1(
    binding: Mapping[str, Any],
    *,
    contract: Any,
    authority: Any,
) -> None:
    """Validate everything needed to reserve without opening generated labels."""

    expected_caps = {
        "attempts": 1,
        "updates": 1_000,
        "presentations": 16_000,
        "microbatch_size": 4,
        "microbatches_per_update": 4,
        "effective_batch_size": 16,
        "target_ema_momentum": 0.996,
    }
    attempt = binding.get("attempt")
    authority_record = binding.get("authority")
    if (
        binding.get("schema") != contract.EXECUTION_BINDING_SCHEMA
        or binding.get("status") != authority.AUTHORIZATION_STATUS
        or binding.get("output_root") != contract.OUTPUT_ROOT_RELATIVE_PATH
        or binding.get("caps") != expected_caps
        or binding.get("seeds")
        != {
            "initialization": 20260712,
            "schedule": 20260713,
            "experiment": 20260728,
            "bootstrap": 20260728,
        }
        or attempt
        != {
            "index": 1,
            "maximum_attempts": 1,
            "fresh": True,
            "retry": False,
            "resume": False,
        }
        or type(authority_record) is not dict
        or authority_record.get("one_exact_fresh_attempt_authorized") is not True
        or authority_record.get("retry_or_resume_authorized") is not False
        or binding.get("downstream_denials") != contract.DOWNSTREAM_DENIALS
        or binding.get("runtime")
        != {
            "interpreter_path": contract.RUNTIME_INTERPRETER_PATH,
            "sys_prefix": contract.RUNTIME_SYS_PREFIX,
        }
        or type(binding.get("wrong_rgb_mapping")) is not dict
        or binding["wrong_rgb_mapping"].get("algorithm")
        != authority.WRONG_RGB_MAPPING_ALGORITHM
    ):
        raise PermissionError("execution binding envelope changed")


def _load_authority_envelope_v1(
    binding_path: Path,
    *,
    repository_root: Path,
) -> tuple[Any, Any, dict[str, Any], bytes, bytes, bytes]:
    contract = _source_module_v1(
        "_lewm_projective_support_runner_contract",
        repository_root / CONTRACT_RELATIVE_PATH,
        repository_root=repository_root,
    )
    authority = _source_module_v1(
        "_lewm_projective_support_runner_authority",
        repository_root / AUTHORITY_RELATIVE_PATH,
        repository_root=repository_root,
    )
    expected_binding_path = repository_root / authority.EXECUTION_BINDING_RELATIVE_PATH
    if binding_path.absolute() != expected_binding_path.absolute():
        raise PermissionError("runner accepts only the canonical execution binding path")
    binding_raw = _read_regular_v1(binding_path)
    binding = contract.parse_canonical_json(binding_raw, name="execution binding")
    validate_execution_envelope_v1(binding, contract=contract, authority=authority)
    source_manifest_raw = _read_bound_v1(
        repository_root,
        binding["source_manifest"],
        expected_path=authority.SOURCE_MANIFEST_RELATIVE_PATH,
    )
    source_review_raw = _read_bound_v1(
        repository_root,
        binding["independent_source_review"],
        expected_path=authority.SOURCE_REVIEW_RELATIVE_PATH,
    )
    authority.validate_source_review_receipt(
        source_review_raw,
        source_manifest_raw,
        root=repository_root,
    )
    return (
        contract,
        authority,
        binding,
        binding_raw,
        source_manifest_raw,
        source_review_raw,
    )


def reserve_attempt_root_v1(
    *,
    repository_root: Path,
    contract: Any,
    binding: Mapping[str, Any],
    binding_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes]:
    """Atomically consume the sole attempt before Torch, data, RGB, or tensors."""

    output_root = repository_root / contract.OUTPUT_ROOT_RELATIVE_PATH
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("the one-shot attempt root already exists")
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise PermissionError("Torch/NumPy was imported before attempt reservation")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o700)
    if stat.S_IMODE(output_root.stat(follow_symlinks=False).st_mode) != 0o700:
        raise PermissionError("attempt root mode is not 0700")
    reservation, raw = _publish_json_v1(
        contract,
        output_root / "reservation.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_reservation_v1",
            "status": "RESERVED_BEFORE_TORCH_DATA_RGB_LABEL_OR_CHECKPOINT",
            "attempt_index": 1,
            "maximum_attempts": 1,
            "execution_binding": {
                "path": EXECUTION_BINDING_RELATIVE_PATH,
                "file_sha256": hashlib.sha256(binding_raw).hexdigest(),
                "content_sha256": binding["content_sha256"],
                "byte_count": len(binding_raw),
            },
            "environment": {
                "interpreter": sys.executable,
                "sys_prefix": sys.prefix,
                "isolated": bool(sys.flags.isolated),
                "bytecode_disabled": bool(sys.dont_write_bytecode),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            },
            "torch_imported_before_reservation": False,
            "numpy_imported_before_reservation": False,
            "development_inputs_opened_before_reservation": False,
            "output_root_absent_before_reservation": True,
            "retry_or_resume_authorized": False,
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    return output_root, reservation, raw


def run_fixed_training_v1(
    model: Any,
    optimizer: Any,
    loader: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    train_labels: FrozenRoleLabelsV1,
    schedule: Sequence[int],
    device: Any,
    *,
    action_order: Sequence[str],
    maximum_updates: int = 1_000,
    progress: dict[str, Any] | None = None,
    full_masks: Any | None = None,
) -> tuple[JointTrainingAccountingV1, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run the exact 1,000-update/16,000-presentation joint schedule."""

    if maximum_updates != 1_000 or len(schedule) != 16_000:
        raise PermissionError("training cap or schedule length changed")
    validate_pairs_against_labels_v1(train_pairs, train_labels)
    accounting = JointTrainingAccountingV1()
    trace: list[dict[str, Any]] = []
    active_r_microbatches = 0
    eligible_rows = 0
    eligible_pairs = 0
    minimum_gradients = {
        "encoder": math.inf,
        "lift_semantic": math.inf,
        "predictor": math.inf,
    }
    maximum_gradients = {name: 0.0 for name in minimum_gradients}
    for update in range(1, maximum_updates + 1):
        update_indices = schedule[
            (update - 1) * PRESENTATIONS_PER_UPDATE
            : update * PRESENTATIONS_PER_UPDATE
        ]
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
            model,
            optimizer,
            microbatches,
            accounting=accounting,
            full_masks=full_masks,
        )
        accounting = result.accounting
        for name, value in result.gradient_l2.items():
            minimum_gradients[name] = min(minimum_gradients[name], value)
            maximum_gradients[name] = max(maximum_gradients[name], value)
        active_r_microbatches += result.ranking_active_microbatches
        eligible_rows += result.ranking_eligible_rows
        eligible_pairs += result.ranking_eligible_pairs
        trace.append({
            "update": update,
            "presentations": accounting.presentations,
            "losses": dict(result.mean_losses),
            "gradient_l2": dict(result.gradient_l2),
            "representation_clip_pre_l2": result.representation_clip_pre_l2,
            "predictor_clip_pre_l2": result.predictor_clip_pre_l2,
            "ranking_active_microbatches": result.ranking_active_microbatches,
            "ranking_eligible_rows": result.ranking_eligible_rows,
            "ranking_eligible_pairs": result.ranking_eligible_pairs,
        })
        if progress is not None:
            progress["accounting"] = dict(accounting.__dict__)
            progress["trace"] = trace
    validate_accounting_v1(accounting)
    if (
        accounting.updates != 1_000
        or accounting.presentations != 16_000
        or accounting.microbatch_graphs != 4_000
        or len(trace) != 1_000
    ):
        raise RuntimeError("terminal training accounting changed")
    return accounting, tuple(trace), {
        "active_r_microbatch_count": active_r_microbatches,
        "active_r_microbatch_fraction": active_r_microbatches / 4_000,
        "ranking_eligible_row_count": eligible_rows,
        "ranking_eligible_pair_count": eligible_pairs,
        "minimum_gradient_l2": minimum_gradients,
        "maximum_gradient_l2": maximum_gradients,
    }


def _checkpoint_bytes_v1(
    torch: Any,
    model: Any,
    accounting: JointTrainingAccountingV1,
) -> bytes:
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    payload = {
        "schema": (
            "lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
            "checkpoint_v1"
        ),
        "update": 1_000,
        "presentations": 16_000,
        "accounting": dict(accounting.__dict__),
        "model_state_dict": state,
        "write_only": True,
        "read_count_after_write": 0,
        "checkpoint_qualified": False,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_frozen_checkpoint_v1(
    path: Path,
    *,
    torch: Any,
    model: Any,
    accounting: JointTrainingAccountingV1,
) -> dict[str, Any]:
    if accounting.updates != 1_000 or accounting.presentations != 16_000:
        raise RuntimeError("checkpoint cannot precede the exact terminal update")
    raw = _checkpoint_bytes_v1(torch, model, accounting)
    _write_exclusive_v1(path, raw)
    return {
        "path": path.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "update": 1_000,
        "presentations": 16_000,
        "write_count": 1,
        "read_count_after_write": 0,
        "write_only": True,
        "checkpoint_qualified": False,
    }


def _jsonable_v1(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable_v1(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name
            not in {
                "primary_mask",
                "predicted_prefix_lengths",
                "target_prefix_lengths",
                "selected_action_indices",
                "oracle_action_indices",
                "selected_target_prefix",
                "oracle_target_prefix",
                "primary_utility",
                "scene_ids",
                "family_ids",
            }
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_v1(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable_v1(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize receipt value {type(value).__name__}")


def build_wrong_rgb_mapping_v1(
    labels: FrozenRoleLabelsV1,
    *,
    metrics: Any,
) -> Any:
    return metrics.wrong_rgb_endpoint_mapping(
        tuple(
            (labels.role, scene, endpoint)
            for scene, endpoint in zip(
                labels.scene_ids, labels.endpoint_ids, strict=True
            )
        )
    )


def validate_wrong_rgb_role_binding_v1(
    execution_binding: Mapping[str, Any],
    mapping: Any,
    pairs: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> None:
    frozen = execution_binding.get("wrong_rgb_mapping")
    per_role = frozen.get("per_role") if type(frozen) is dict else None
    expected = per_role.get(role) if type(per_role) is dict else None
    if (
        frozen.get("algorithm")
        != "role_scene_local_lexicographic_cyclic_derangement_v1"
        or frozen.get("roles")
        != ["train", "probability_calibration", "checkpoint_selection"]
        or type(expected) is not dict
        or expected.get("row_count") != len(mapping.rows)
        or expected.get("mapping_sha256") != mapping.mapping_sha256
        or frozen.get("paired_next_collision_count") != 0
        or frozen.get("paired_next_collision_rows_sha256")
        != hashlib.sha256(b"[]").hexdigest()
        or frozen.get("mapped_endpoint_is_never_paired_next") is not True
    ):
        raise PermissionError(f"{role} wrong-RGB mapping escaped execution binding")
    if len(pairs) != len(mapping.rows):
        raise PermissionError(f"{role} wrong-RGB pair population changed")
    for state_index, pair in enumerate(pairs):
        key = (
            pair.get("dataset_role"),
            pair.get("scene_id"),
            pair.get("current_endpoint_sha256"),
        )
        mapped_endpoint = mapping.by_endpoint.get(key)
        if mapped_endpoint is None:
            raise PermissionError(
                f"{role} wrong-RGB pair identity changed at state {state_index}"
            )
        if mapped_endpoint == pair.get("next_endpoint_sha256"):
            raise PermissionError(
                f"{role} wrong-RGB mapping selected a paired future endpoint"
            )


def validate_wrong_rgb_complete_binding_v1(
    execution_binding: Mapping[str, Any],
    mappings: Mapping[str, Any],
    *,
    metrics: Any,
) -> None:
    if set(mappings) != {
        "train",
        "probability_calibration",
        "checkpoint_selection",
    }:
        raise PermissionError("wrong-RGB role mapping population is incomplete")
    endpoints = tuple(
        row[:3]
        for role in (
            "train",
            "probability_calibration",
            "checkpoint_selection",
        )
        for row in mappings[role].rows
    )
    combined = metrics.wrong_rgb_endpoint_mapping(endpoints)
    frozen = execution_binding.get("wrong_rgb_mapping")
    if (
        type(frozen) is not dict
        or frozen.get("row_count") != len(combined.rows)
        or frozen.get("mapping_sha256") != combined.mapping_sha256
        or frozen.get("paired_next_collision_count") != 0
        or frozen.get("paired_next_collision_rows_sha256")
        != hashlib.sha256(b"[]").hexdigest()
        or frozen.get("mapped_endpoint_is_never_paired_next") is not True
    ):
        raise PermissionError("complete wrong-RGB mapping escaped execution binding")


def _load_role_labels_bound_v1(
    labels_api: Any,
    execution_binding: Mapping[str, Any],
    repository_root: Path,
    *,
    role: str,
    progress: dict[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    relative = (
        ".generated/go2_post_action_projective_support_labels_v2/"
        f"{role}.jsonl"
    )
    files = execution_binding["label_bundle"]["files"]
    record = files.get(relative)
    if type(record) is not dict or record.get("path") != relative:
        raise PermissionError(f"{role} label binding is absent")
    progress.setdefault("authorized_input_open_attempts", []).append(relative)
    rows = labels_api.load_role_labels_v1(
        repository_root / relative,
        role=role,
        expected_file_sha256=str(record["file_sha256"]),
    )
    progress.setdefault("authorized_input_open_successes", []).append(relative)
    return rows


def _access_receipt_v1(
    progress: Mapping[str, Any],
    *,
    contract: Any,
) -> dict[str, Any]:
    loader = progress.get("loader")
    inputs = progress.get("inputs")
    detailed = loader.receipt() if loader is not None else None
    semantic_forbidden = (
        0
        if detailed is None
        else sum(
            int(value)
            for value in detailed["forbidden_semantic_counters"].values()
        )
    )
    fixed_negative = (
        0
        if detailed is None
        else int(detailed["rgb_request_count"]["fixed_negative"])
    )
    consumed = getattr(inputs, "consumed", {}) if inputs is not None else {}
    return {
        "schema": f"{contract.SCHEMA_PREFIX}_access_v1",
        "status": "DEVELOPMENT_RGB_RASTER_AND_FROZEN_LABELS_ONLY",
        "role_transitions": list(progress.get("role_transitions", [])),
        "roles_opened": list(progress.get("roles_opened", [])),
        "raw_consumed_record_count": len(consumed),
        "raw_constructor_reads": progress.get("_raw_constructor_reads", {}),
        "n320_gate_open_attempted": bool(
            progress.get("n320_gate_open_attempted", False)
        ),
        "n320_gate_open_succeeded": bool(
            progress.get("n320_gate_open_succeeded", False)
        ),
        "n320_checkpoint_open_attempted": bool(
            progress.get("n320_checkpoint_open_attempted", False)
        ),
        "n320_checkpoint_open_succeeded": bool(
            progress.get("n320_checkpoint_open_succeeded", False)
        ),
        "authorized_input_open_attempts": list(
            progress.get("authorized_input_open_attempts", [])
        ),
        "authorized_input_open_successes": list(
            progress.get("authorized_input_open_successes", [])
        ),
        "raw_consumed_roles": sorted({
            str(role)
            for record in consumed.values()
            if isinstance(record, Mapping)
            for role in record.get("roles", [])
        }),
        "narrow_loader": detailed,
        "fixed_negative_rgb_request_count": fixed_negative,
        "written_checkpoint_read_count": 0,
        "training_trace_read_count": 0,
        "rejected_checkpoint_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "forbidden_input_count": semantic_forbidden + fixed_negative,
        "bypass_count": 0,
        "forbidden_open_count": 0,
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "sealed_open_count": 0,
        "production_open_count": 0,
        "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
    }


def _load_post_reservation_inputs_v1(
    *,
    repository_root: Path,
    contract: Any,
    authority: Any,
    execution_binding: Mapping[str, Any],
    source_manifest_raw: bytes,
    source_review_raw: bytes,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """Validate bound labels and construct the reviewed narrow runtime stack."""

    label_manifest_binding = execution_binding["label_bundle"]["manifest"]
    label_manifest_raw = _read_bound_tracked_v1(
        progress,
        repository_root,
        label_manifest_binding,
        expected_path=contract.LABEL_MANIFEST_RELATIVE_PATH,
    )
    label_files = execution_binding["label_bundle"]["files"]
    label_preflight_raw = _read_bound_tracked_v1(
        progress,
        repository_root,
        execution_binding["label_preflight_receipt"],
        expected_path=authority.LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH,
    )
    label_preflight = contract.parse_canonical_json(
        label_preflight_raw, name="label-preflight receipt"
    )
    label_builder_raw = _read_bound_tracked_v1(
        progress,
        repository_root,
        label_preflight["label_builder_execution_binding"],
        expected_path=authority.LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH,
    )
    authority.validate_execution_binding(
        contract.canonical_json_bytes(execution_binding) + b"\n",
        source_manifest_raw,
        source_review_raw,
        label_manifest_raw,
        label_files,
        label_builder_execution_binding_raw=label_builder_raw,
        label_preflight_receipt_raw=label_preflight_raw,
        root=repository_root,
    )
    labels_api = _source_module_v1(
        "_lewm_projective_support_runner_labels",
        repository_root / LABELS_RELATIVE_PATH,
        repository_root=repository_root,
    )
    metrics = _source_module_v1(
        "_lewm_projective_support_runner_metrics",
        repository_root / METRICS_RELATIVE_PATH,
        repository_root=repository_root,
    )
    direct = _source_module_v1(
        "_lewm_projective_support_runner_direct_loader",
        repository_root / DIRECT_RUNNER_RELATIVE_PATH,
        repository_root=repository_root,
    )
    matched = _source_module_v1(
        "_lewm_projective_support_runner_matched_runtime",
        repository_root / MATCHED_RUNNER_RELATIVE_PATH,
        repository_root=repository_root,
    )
    runtime = matched._load_runtime()
    torch = runtime.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible ROCm device is required")
    device = torch.device("cuda:0")
    progress["hardware"] = {
        "name": torch.cuda.get_device_name(0),
        "visible_device_count": int(torch.cuda.device_count()),
        "total_memory_bytes": int(
            torch.cuda.get_device_properties(0).total_memory
        ),
    }

    runtime_inputs = execution_binding["runtime_inputs"]
    adapted = {
        "raw": {
            "manifest": runtime_inputs["raw_manifest"],
            "audit": runtime_inputs["raw_audit"],
        },
        "camera": {
            "gate": runtime_inputs["n320_gate"],
            "checkpoint": runtime_inputs["n320_encoder_checkpoint"],
        },
    }
    raw_index_paths = (
        contract.RAW_MANIFEST_RELATIVE_PATH,
        contract.RAW_PAIRS_RELATIVE_PATH,
        contract.RAW_ENDPOINTS_RELATIVE_PATH,
    )
    progress["authorized_input_open_attempts"].extend(raw_index_paths)
    raw_indexes = labels_api.load_and_validate_raw_indexes(
        repository_root / contract.RAW_MANIFEST_RELATIVE_PATH,
        repository_root / contract.RAW_PAIRS_RELATIVE_PATH,
        repository_root / contract.RAW_ENDPOINTS_RELATIVE_PATH,
    )
    progress["authorized_input_open_successes"].extend(raw_index_paths)
    progress["authorized_input_open_attempts"].append(
        contract.RAW_AUDIT_RELATIVE_PATH
    )
    labels_api.validate_raw_audit_v1(
        repository_root / contract.RAW_AUDIT_RELATIVE_PATH
    )
    progress["authorized_input_open_successes"].append(
        contract.RAW_AUDIT_RELATIVE_PATH
    )
    progress["authorized_input_open_attempts"].append(
        contract.SCHEDULE_RELATIVE_PATH
    )
    schedule = labels_api.load_schedule_indices_v1(
        repository_root / contract.SCHEDULE_RELATIVE_PATH,
        raw_indexes=raw_indexes,
    )
    progress["authorized_input_open_successes"].append(
        contract.SCHEDULE_RELATIVE_PATH
    )
    if (
        len(schedule) != 16_000
        or contract.canonical_json_sha256(list(schedule))
        != contract.SCHEDULE_PREFIX_SHA256
    ):
        raise PermissionError("16,000-presentation schedule prefix changed")

    inputs = direct._construct_raw_inputs_with_progress(
        matched, runtime, adapted, progress
    )
    direct._normalize_endpoint_paths(inputs)
    progress["inputs"] = inputs
    proxy = ExperimentArmRawInputsProxyV1(inputs)
    loader = direct.DirectBevNarrowLoader(runtime, proxy, progress=progress)
    progress["loader"] = loader
    fit, n320_gate, n320_checkpoint = direct._load_n320_with_progress(
        matched, runtime, adapted, progress
    )
    progress["n320_gate"] = n320_gate
    progress["n320_checkpoint"] = n320_checkpoint

    for name in (
        "predicted_next_corridor_masks.u1",
        "persistence_corridor_masks.u1",
        "projective_support_mask.u1",
    ):
        relative = f"{contract.LABEL_ROOT_RELATIVE_PATH}/{name}"
        _read_bound_tracked_v1(
            progress,
            repository_root,
            label_files[relative],
            expected_path=relative,
        )
    return {
        "labels_api": labels_api,
        "metrics": metrics,
        "direct": direct,
        "runtime": runtime,
        "torch": torch,
        "device": device,
        "inputs": inputs,
        "loader": loader,
        "fit": fit,
        "schedule": schedule,
        "label_preflight": label_preflight,
    }


def _publish_training_trace_v1(
    contract: Any,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
) -> tuple[dict[str, Any], bytes]:
    return _publish_json_v1(
        contract,
        output_root / "training_trace.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_training_trace_v1",
            "status": "COMPLETE" if complete else "TERMINAL_PARTIAL",
            "row_count": len(rows),
            "rows": [dict(row) for row in rows],
            "write_only": True,
            "read_count_after_write": 0,
            "resume_authorized": False,
        },
    )


def _execute_reserved_science_v1(
    *,
    repository_root: Path,
    contract: Any,
    authority: Any,
    execution_binding: Mapping[str, Any],
    source_manifest_raw: bytes,
    source_review_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress["stage"] = "post_reservation_authority_and_input_validation"
    context = _load_post_reservation_inputs_v1(
        repository_root=repository_root,
        contract=contract,
        authority=authority,
        execution_binding=execution_binding,
        source_manifest_raw=source_manifest_raw,
        source_review_raw=source_review_raw,
        progress=progress,
    )
    labels_api = context["labels_api"]
    metrics = context["metrics"]
    runtime = context["runtime"]
    torch = context["torch"]
    device = context["device"]
    inputs = context["inputs"]
    loader = context["loader"]
    schedule = context["schedule"]
    fit = context["fit"]
    np = runtime.np

    torch.manual_seed(contract.EXPERIMENT_SEED)
    torch.cuda.manual_seed_all(contract.EXPERIMENT_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    progress["determinism"] = {
        "experiment_seed": contract.EXPERIMENT_SEED,
        "deterministic_algorithms": True,
        "warn_only_for_reviewed_rocm_grid_sample_backward": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }

    progress["stage"] = "train"
    progress["role_transitions"].append("train")
    progress["roles_opened"].append("train")
    train_rows = _load_role_labels_bound_v1(
        labels_api,
        execution_binding,
        repository_root,
        role="train",
        progress=progress,
    )
    train_labels = freeze_role_labels_v1(train_rows, role="train", np=np)
    train_pairs = inputs.role_pairs("train")
    validate_pairs_against_labels_v1(train_pairs, train_labels)
    train_wrong_mapping = build_wrong_rgb_mapping_v1(train_labels, metrics=metrics)
    validate_wrong_rgb_role_binding_v1(
        execution_binding, train_wrong_mapping, train_pairs, role="train"
    )
    wrong_mappings: dict[str, Any] = {"train": train_wrong_mapping}
    action_prior = metrics.action_prior_probabilities(train_labels.station_safe)
    frozen_action_prior = context["label_preflight"].get("action_prior")
    action_prior_rows = action_prior.tolist()
    if (
        type(frozen_action_prior) is not dict
        or frozen_action_prior.get("source_role") != "train"
        or frozen_action_prior.get("source_roles") != ["train"]
        or frozen_action_prior.get("source_state_count") != len(train_pairs)
        or frozen_action_prior.get("action_order")
        != list(contract.ACTION_VOCABULARY)
        or frozen_action_prior.get("station_count") != 11
        or frozen_action_prior.get("shape") != [9, 11]
        or frozen_action_prior.get("probabilities") != action_prior_rows
        or frozen_action_prior.get("probabilities_sha256")
        != contract.canonical_json_sha256(action_prior_rows)
    ):
        raise PermissionError("train-only action prior escaped label preflight")

    torch_api, model_api, scoring = _runtime_apis()
    if torch_api is not torch:
        raise RuntimeError("runtime imported two Torch identities")
    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    model = model_api.GeometryAnchoredDeformableBevLiftJointJepaV1(
        n320_encoder_state_dict=n320_encoder
    ).to(device)
    model.train()
    partition = partition_parameters_v1(model)
    optimizer = build_frozen_optimizer_v1(partition)
    masks = scoring.build_validated_corridor_masks_v1()
    scoring.validate_corridor_masks_v1(
        masks.full, masks.persistence, masks.projective_support
    )
    immediate_support_regression = (
        scoring.build_immediate_footprint_support_regression_v1()
    )
    full_masks = masks.full.to(device=device)
    persistence_masks = masks.persistence.to(device=device)
    progress["model_initialized"] = True
    progress["model_initial_integrity"] = {
        "fresh_components": True,
        "target_hard_sync_count": int(model.target_hard_sync_count.item()),
        "target_ema_update_count": int(model.ema_update_count.item()),
        "target_optimizer_membership_count": 0,
        "target_requires_grad_count": sum(
            int(parameter.requires_grad) for parameter in partition.target
        ),
        "initialization_seed": int(model.config.initialization_seed),
        "target_ema_momentum": float(model.config.target_ema_momentum),
    }
    if progress["model_initial_integrity"] != {
        "fresh_components": True,
        "target_hard_sync_count": 1,
        "target_ema_update_count": 0,
        "target_optimizer_membership_count": 0,
        "target_requires_grad_count": 0,
        "initialization_seed": 20260712,
        "target_ema_momentum": 0.996,
    }:
        raise RuntimeError("fresh model integrity changed")

    accounting, trace, training_diagnostics = run_fixed_training_v1(
        model,
        optimizer,
        loader,
        train_pairs,
        train_labels,
        schedule,
        device,
        action_order=contract.ACTION_VOCABULARY,
        progress=progress,
        full_masks=full_masks,
    )
    progress["accounting"] = dict(accounting.__dict__)
    progress["trace"] = trace
    progress["training_diagnostics"] = training_diagnostics
    model.eval()
    progress["stage"] = "frozen_update_1000"
    progress["role_transitions"].append("frozen_update_1000")
    checkpoint = write_frozen_checkpoint_v1(
        output_root / "checkpoint_update_1000.pt",
        torch=torch,
        model=model,
        accounting=accounting,
    )
    progress["checkpoint"] = checkpoint
    trace_value, trace_raw = _publish_training_trace_v1(
        contract, output_root, trace, complete=True
    )
    progress["training_trace"] = _artifact_binding_v1(
        "training_trace.json", trace_value, trace_raw
    )

    progress["stage"] = "probability_calibration"
    progress["role_transitions"].append("probability_calibration")
    progress["roles_opened"].append("probability_calibration")
    calibration_rows = _load_role_labels_bound_v1(
        labels_api,
        execution_binding,
        repository_root,
        role="probability_calibration",
        progress=progress,
    )
    calibration_labels = freeze_role_labels_v1(
        calibration_rows, role="probability_calibration", np=np
    )
    calibration_pairs = inputs.role_pairs("probability_calibration")
    validate_pairs_against_labels_v1(calibration_pairs, calibration_labels)
    calibration_wrong_mapping = build_wrong_rgb_mapping_v1(
        calibration_labels, metrics=metrics
    )
    validate_wrong_rgb_role_binding_v1(
        execution_binding,
        calibration_wrong_mapping,
        calibration_pairs,
        role="probability_calibration",
    )
    wrong_mappings["probability_calibration"] = calibration_wrong_mapping
    calibration_scores = score_role_population_v1(
        model,
        loader,
        calibration_pairs,
        calibration_labels,
        calibration_wrong_mapping.by_endpoint,
        action_prior,
        device,
        stage="probability_calibration",
        np=np,
        full_masks=full_masks,
        persistence_masks=persistence_masks,
    )
    calibrations = metrics.calibrate_arms(
        calibration_scores.probabilities,
        calibration_labels.station_safe,
        family_ids=calibration_labels.family_ids,
    )
    calibration_value, calibration_raw = _publish_json_v1(
        contract,
        output_root / "calibration.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_calibration_v1",
            "status": "COMPARABLE" if calibrations.comparable else "NON_COMPARABLE",
            "role": "probability_calibration",
            "state_count": len(calibration_pairs),
            "arms": _jsonable_v1(calibrations.arms),
            "failed_arms": list(calibrations.failed_arms),
            "selection_role_open_count": 0,
        },
    )
    progress["calibration"] = _artifact_binding_v1(
        "calibration.json", calibration_value, calibration_raw
    )
    progress["calibrations"] = calibrations
    if not calibrations.comparable:
        gate = metrics.GateDecision(
            status="TERMINAL_NON_COMPARABLE_CONTROL_CALIBRATION",
            passed=False,
            checks={"all_arm_calibrations_eligible": False},
            failed_checks=("all_arm_calibrations_eligible",),
            comparisons={},
            failed_calibration_arms=calibrations.failed_arms,
        )
        return {
            "gate": gate,
            "semantic": None,
            "integrity": None,
            "evaluations": {},
            "calibrations": calibrations,
        }, context

    progress["stage"] = "checkpoint_selection"
    progress["role_transitions"].append("checkpoint_selection")
    progress["roles_opened"].append("checkpoint_selection")
    selection_rows = _load_role_labels_bound_v1(
        labels_api,
        execution_binding,
        repository_root,
        role="checkpoint_selection",
        progress=progress,
    )
    selection_labels = freeze_role_labels_v1(
        selection_rows, role="checkpoint_selection", np=np
    )
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    validate_pairs_against_labels_v1(selection_pairs, selection_labels)
    selection_wrong_mapping = build_wrong_rgb_mapping_v1(
        selection_labels, metrics=metrics
    )
    validate_wrong_rgb_role_binding_v1(
        execution_binding,
        selection_wrong_mapping,
        selection_pairs,
        role="checkpoint_selection",
    )
    wrong_mappings["checkpoint_selection"] = selection_wrong_mapping
    validate_wrong_rgb_complete_binding_v1(
        execution_binding, wrong_mappings, metrics=metrics
    )
    selection_scores = score_role_population_v1(
        model,
        loader,
        selection_pairs,
        selection_labels,
        selection_wrong_mapping.by_endpoint,
        action_prior,
        device,
        stage="checkpoint_selection",
        np=np,
        full_masks=full_masks,
        persistence_masks=persistence_masks,
    )
    evaluations = {
        name: metrics.evaluate_arm(
            selection_scores.probabilities[name],
            selection_labels.station_safe,
            float(calibrations.arms[name].threshold),
            selection_labels.scene_ids,
            selection_labels.family_ids,
            selection_labels.immediate_feasible,
            selection_labels.blind_bridge_feasible,
            arm_name=name,
        )
        for name in metrics.ARM_NAMES
    }
    semantic = metrics.semantic_retention_from_confusions(
        selection_scores.semantic_confusion,
        selection_scores.rough_semantic_confusion,
    )
    access = _access_receipt_v1(progress, contract=contract)
    full_nonempty = bool((masks.full.sum(dim=(-2, -1)) > 0).all())
    persistence_nonempty = bool(
        (masks.persistence.sum(dim=(-2, -1)) > 0).all()
    )
    inside_support = bool(
        not (masks.full.bool() & ~masks.projective_support).any()
        and not (
            masks.persistence.bool() & ~masks.projective_support[None, None]
        ).any()
    )
    integrity = metrics.IntegrityMetrics(
        exact_accounting=(
            accounting.updates == 1_000
            and accounting.presentations == 16_000
            and accounting.microbatch_graphs == 4_000
            and accounting.backward_calls == 4_000
            and accounting.optimizer_steps == 1_000
            and accounting.ema_steps == 1_000
            and int(model.ema_update_count.item()) == 1_000
            and progress["role_transitions"]
            == [
                "train",
                "frozen_update_1000",
                "probability_calibration",
                "checkpoint_selection",
            ]
            and progress["roles_opened"]
            == ["train", "probability_calibration", "checkpoint_selection"]
            and int(access["fixed_negative_rgb_request_count"]) == 0
            and int(access["written_checkpoint_read_count"]) == 0
            and progress["checkpoint"]["write_count"] == 1
            and progress["checkpoint"]["read_count_after_write"] == 0
            and bool(access["n320_gate_open_succeeded"])
            and bool(access["n320_checkpoint_open_succeeded"])
            and access["authorized_input_open_attempts"]
            == access["authorized_input_open_successes"]
        ),
        outputs_and_gradients_finite=(
            calibration_scores.all_values_finite
            and selection_scores.all_values_finite
        ),
        target_gradients_zero=all(
            parameter.grad is None for parameter in partition.target
        ),
        target_optimizer_membership_zero=not bool(
            set(map(id, partition.target))
            & {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
        ),
        online_gradients_nonzero_every_update=all(
            value > 0.0
            for value in training_diagnostics["minimum_gradient_l2"].values()
        ),
        predictor_forward_count=accounting.predictor_forwards,
        predictor_objective_count=accounting.predictor_objectives,
        backward_count=accounting.backward_calls,
        predictor_optimizer_update_count=accounting.optimizer_steps,
        forbidden_input_count=int(access["forbidden_input_count"]),
        bypass_count=int(access["bypass_count"]),
        forbidden_open_count=int(access["forbidden_open_count"]),
        current_latents_nonconstant=selection_scores.current_latents_nonconstant,
        paired_latents_nonconstant=selection_scores.paired_latents_nonconstant,
        current_and_paired_latents_nonidentical=(
            selection_scores.current_and_paired_latents_nonidentical
        ),
        one_step_zero_support_witnessed=immediate_support_regression.passed,
        all_corridor_masks_nonempty=full_nonempty and persistence_nonempty,
        corridor_masks_inside_support=inside_support,
    )
    gate = metrics.evaluate_conjunctive_gate(
        calibrations, evaluations, integrity, semantic
    )
    return {
        "gate": gate,
        "semantic": semantic,
        "integrity": integrity,
        "evaluations": evaluations,
        "calibrations": calibrations,
        "wrong_rgb_mapping": execution_binding["wrong_rgb_mapping"],
        "immediate_support_regression": immediate_support_regression,
    }, context


def _publish_terminal_result_v1(
    *,
    contract: Any,
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    science: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> int:
    gate = science["gate"]
    passed = bool(gate.passed)
    metrics_value, metrics_raw = _publish_or_reuse_json_v1(
        contract,
        output_root / "metrics.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_metrics_v1",
            "status": str(gate.status),
            "gate": _jsonable_v1(gate),
            "semantic_retention": _jsonable_v1(science.get("semantic")),
            "integrity": _jsonable_v1(science.get("integrity")),
            "calibrations": _jsonable_v1(science["calibrations"]),
            "evaluations": _jsonable_v1(science["evaluations"]),
            "training_accounting": progress.get("accounting"),
            "training_diagnostics": progress.get("training_diagnostics"),
            "wrong_rgb_mapping": science.get("wrong_rgb_mapping"),
            "immediate_support_regression": _jsonable_v1(
                science.get("immediate_support_regression")
            ),
            "hardware": progress.get("hardware"),
            "determinism": progress.get("determinism"),
            "n320_gate_content_sha256": (
                progress.get("n320_gate", {}).get("content_sha256")
                if isinstance(progress.get("n320_gate"), Mapping)
                else None
            ),
            "n320_checkpoint": progress.get("n320_checkpoint"),
            "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
        },
    )
    access_core = _access_receipt_v1(progress, contract=contract)
    access_value, access_raw = _publish_or_reuse_json_v1(
        contract, output_root / "access.json", access_core
    )
    artifact_value, artifact_raw = _publish_or_reuse_json_v1(
        contract,
        output_root / "artifact.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_artifact_v1",
            "status": str(gate.status),
            "checkpoint": progress.get("checkpoint"),
            "training_trace": progress.get("training_trace"),
            "calibration": progress.get("calibration"),
            "all_checkpoint_and_trace_outputs_write_only": True,
            "checkpoint_read_count_after_write": 0,
            "checkpoint_qualified": False,
        },
    )
    failure_binding: dict[str, Any] | None = None
    if not passed:
        failure_value, failure_raw = _publish_or_reuse_json_v1(
            contract,
            output_root / "failure.json",
            {
                "schema": contract.FAILURE_SCHEMA,
                "status": str(gate.status),
                "failure_class": "scientific_gate",
                "failed_checks": list(gate.failed_checks),
                "failed_calibration_arms": list(gate.failed_calibration_arms),
                "updates": int(progress.get("accounting", {}).get("updates", 0)),
                "presentations": int(
                    progress.get("accounting", {}).get("presentations", 0)
                ),
                "selection_opened": (
                    "checkpoint_selection" in progress.get("roles_opened", [])
                ),
                "retry_or_resume_authorized": False,
                "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
                "complete_failure_receipt": True,
            },
        )
        failure_binding = _artifact_binding_v1(
            "failure.json", failure_value, failure_raw
        )
    result_value, result_raw = _publish_or_reuse_json_v1(
        contract,
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": str(gate.status),
            "reservation": _artifact_binding_v1(
                "reservation.json", reservation, reservation_raw
            ),
            "metrics": _artifact_binding_v1(
                "metrics.json", metrics_value, metrics_raw
            ),
            "access": _artifact_binding_v1(
                "access.json", access_value, access_raw
            ),
            "artifact": _artifact_binding_v1(
                "artifact.json", artifact_value, artifact_raw
            ),
            "failure": failure_binding,
            "hardware": progress.get("hardware"),
            "determinism": progress.get("determinism"),
            "n320_checkpoint": progress.get("n320_checkpoint"),
            "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
            "mechanism_passed": passed,
            "checkpoint_qualified": False,
            "checkpoint_read_authorized": False,
            "retry_or_resume_authorized": False,
            "next_authority": (
                "separate_preregistered_matched_no_jepa_development_arm_only"
                if passed
                else "none"
            ),
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    _publish_or_reuse_json_v1(
        contract,
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": str(gate.status),
            "result": _artifact_binding_v1("result.json", result_value, result_raw),
            "mechanism_passed": passed,
            "checkpoint_qualified": False,
            "retry_or_resume_authorized": False,
            "complete_failure_receipt": not passed,
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    return 0 if passed else 2


def _publish_terminal_exception_v1(
    *,
    contract: Any,
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: dict[str, Any],
    error: BaseException,
) -> int:
    result_path = output_root / "result.json"
    if result_path.exists() or result_path.is_symlink():
        result_value, result_raw = _load_published_json_v1(contract, result_path)
        passed = bool(result_value.get("mechanism_passed"))
        _publish_or_reuse_json_v1(
            contract,
            output_root / "completed.json",
            {
                "schema": contract.COMPLETION_SCHEMA,
                "status": str(result_value.get("status")),
                "result": _artifact_binding_v1(
                    "result.json", result_value, result_raw
                ),
                "mechanism_passed": passed,
                "checkpoint_qualified": False,
                "retry_or_resume_authorized": False,
                "complete_failure_receipt": not passed,
                "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
            },
        )
        return 0 if passed else 2

    trace = tuple(progress.get("trace", ()))
    if not (output_root / "training_trace.json").exists():
        trace_value, trace_raw = _publish_training_trace_v1(
            contract, output_root, trace, complete=False
        )
        progress["training_trace"] = _artifact_binding_v1(
            "training_trace.json", trace_value, trace_raw
        )
    elif progress.get("training_trace") is None:
        trace_value, trace_raw = _load_published_json_v1(
            contract, output_root / "training_trace.json"
        )
        progress["training_trace"] = _artifact_binding_v1(
            "training_trace.json", trace_value, trace_raw
        )
    access_core = _access_receipt_v1(progress, contract=contract)
    access_value, access_raw = _publish_or_reuse_json_v1(
        contract, output_root / "access.json", access_core
    )

    partial_bindings: dict[str, Any] = {}
    for name in ("metrics.json", "artifact.json", "calibration.json"):
        path = output_root / name
        if path.exists() or path.is_symlink():
            value, raw = _load_published_json_v1(contract, path)
            partial_bindings[name] = _artifact_binding_v1(name, value, raw)
    if progress.get("training_trace") is not None:
        partial_bindings["training_trace.json"] = progress["training_trace"]

    failure_path = output_root / "failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        failure_value, failure_raw = _load_published_json_v1(
            contract, failure_path
        )
    else:
        failure_value, failure_raw = _publish_or_reuse_json_v1(
            contract,
            failure_path,
            {
                "schema": contract.FAILURE_SCHEMA,
                "status": "TERMINAL_EXECUTION_FAILURE",
                "failure_class": "integrity_or_numerical_exception",
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2_000],
                "traceback": "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                )[-12_000:],
                "stage": progress.get("stage"),
                "role_transitions": list(progress.get("role_transitions", [])),
                "roles_opened": list(progress.get("roles_opened", [])),
                "training_accounting": progress.get("accounting"),
                "training_trace": progress.get("training_trace"),
                "checkpoint": progress.get("checkpoint"),
                "partial_terminal_artifacts": partial_bindings,
                "hardware": progress.get("hardware"),
                "determinism": progress.get("determinism"),
                "n320_checkpoint": progress.get("n320_checkpoint"),
                "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
                "retry_or_resume_authorized": False,
                "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
                "complete_failure_receipt": True,
            },
        )
    terminal_status = str(
        failure_value.get("status", "TERMINAL_EXECUTION_FAILURE")
    )
    result_value, result_raw = _publish_or_reuse_json_v1(
        contract,
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": terminal_status,
            "reservation": _artifact_binding_v1(
                "reservation.json", reservation, reservation_raw
            ),
            "access": _artifact_binding_v1(
                "access.json", access_value, access_raw
            ),
            "failure": _artifact_binding_v1(
                "failure.json", failure_value, failure_raw
            ),
            "partial_terminal_artifacts": partial_bindings,
            "terminal_publication_exception": {
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2_000],
                "stage": progress.get("stage"),
            },
            "mechanism_passed": False,
            "checkpoint_qualified": False,
            "checkpoint_read_authorized": False,
            "retry_or_resume_authorized": False,
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    _publish_or_reuse_json_v1(
        contract,
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": terminal_status,
            "result": _artifact_binding_v1("result.json", result_value, result_raw),
            "mechanism_passed": False,
            "checkpoint_qualified": False,
            "retry_or_resume_authorized": False,
            "complete_failure_receipt": True,
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    return 2


def run_from_execution_binding_v1(
    binding_path: Path | str,
    repository_root: Path = ROOT,
) -> int:
    """Validate, reserve, and consume the sole reviewed execution binding."""

    repository_root = Path(repository_root)
    if repository_root.resolve() != ROOT.resolve():
        raise PermissionError("operational execution requires the reviewed repository root")
    (
        contract,
        authority,
        execution_binding,
        binding_raw,
        source_manifest_raw,
        source_review_raw,
    ) = _load_authority_envelope_v1(
        Path(binding_path), repository_root=repository_root
    )
    if (
        sys.executable != contract.RUNTIME_INTERPRETER_PATH
        or sys.prefix != contract.RUNTIME_SYS_PREFIX
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise PermissionError("runner requires the exact isolated reviewed ROCm runtime")
    output_root, reservation, reservation_raw = reserve_attempt_root_v1(
        repository_root=repository_root,
        contract=contract,
        binding=execution_binding,
        binding_raw=binding_raw,
    )
    progress: dict[str, Any] = {
        "stage": "reserved",
        "role_transitions": [],
        "roles_opened": [],
        "accounting": dict(JointTrainingAccountingV1().__dict__),
        "trace": (),
        "g2_navigation_heldout_sealed_open_count": 0,
    }
    try:
        progress["stage"] = "bounded_science"
        science, _context = _execute_reserved_science_v1(
            repository_root=repository_root,
            contract=contract,
            authority=authority,
            execution_binding=execution_binding,
            source_manifest_raw=source_manifest_raw,
            source_review_raw=source_review_raw,
            output_root=output_root,
            progress=progress,
        )
        progress["stage"] = "terminal_receipts"
        return _publish_terminal_result_v1(
            contract=contract,
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            science=science,
            progress=progress,
        )
    except BaseException as error:
        progress["stage"] = f"terminal_failure:{progress.get('stage')}"
        terminal_code = _publish_terminal_exception_v1(
            contract=contract,
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            progress=progress,
            error=error,
        )
        if terminal_code != 0:
            print(
                f"terminal failure: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return terminal_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-binding",
        type=Path,
        default=ROOT / EXECUTION_BINDING_RELATIVE_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_from_execution_binding_v1(args.execution_binding)


__all__ = [
    "BatchControlScoresV1",
    "CURRENT_LABELS_KEY",
    "CURRENT_RGB_KEY",
    "EXECUTED_ACTION_KEY",
    "JointTrainingAccountingV1",
    "JointUpdateResultV1",
    "FrozenRoleLabelsV1",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NEXT_LABELS_KEY",
    "NEXT_RGB_KEY",
    "PRESENTATIONS_PER_UPDATE",
    "ParameterPartitionV1",
    "REQUIRED_BATCH_KEYS",
    "STATION_SAFE_KEY",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "build_wrong_rgb_mapping_v1",
    "freeze_role_labels_v1",
    "joint_training_update_v1",
    "main",
    "parse_args",
    "partition_parameters_v1",
    "score_full_control_v1",
    "score_persistence_control_v1",
    "score_shuffled_control_v1",
    "run_fixed_training_v1",
    "run_from_execution_binding_v1",
    "validate_accounting_v1",
    "validate_execution_envelope_v1",
    "validate_optimizer_v1",
    "validate_pairs_against_labels_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
