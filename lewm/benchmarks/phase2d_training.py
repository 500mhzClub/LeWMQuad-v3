"""Training-batch and cell contracts for the preregistered Phase 2D experiment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .phase2_data import (
    HardNegativeIndex,
    action_name,
    action_vector,
    is_zero_action,
    materialized_frame_paths,
    source_key,
    transition_validity,
)


@dataclass(frozen=True)
class Phase2DCell:
    """Fixed model-side differences for one registered cell or diagnostic control."""

    name: str
    target_ema_momentum: float | None
    action_identifiability_lambda: float
    zero_action_lambda: float
    prediction_input_mode: str
    participates_in_checkpoint_selection: bool


REGISTERED_CELLS = {
    "C0": Phase2DCell("C0", None, 0.0, 0.0, "state_action", True),
    "C1": Phase2DCell("C1", 0.99, 0.0, 0.0, "state_action", True),
    "C2": Phase2DCell("C2", 0.99, 1.0, 1.0, "state_action", True),
    "state_only": Phase2DCell(
        "state_only",
        0.99,
        0.0,
        0.0,
        "state_only",
        False,
    ),
    "action_only": Phase2DCell(
        "action_only",
        0.99,
        0.0,
        0.0,
        "action_only",
        False,
    ),
}

CONSEQUENCE_TARGET_NAMES = (
    "target_progress_norm",
    "clearance_gain_norm",
    "minimum_clearance_norm",
    "p05_clearance_norm",
    "unsafe_sample_fraction",
    "enters_grid_unsafe",
    "ends_grid_unsafe",
    "target_recoverable",
    "heading_alignment",
)
CONSEQUENCE_TARGET_DIM = len(CONSEQUENCE_TARGET_NAMES)
ACTION_UTILITY_TARGET_VERSION = "phase2g_oracle_order_utility_v0"


def registered_cell(name: str) -> Phase2DCell:
    """Return one immutable registered model cell configuration."""

    try:
        return REGISTERED_CELLS[name]
    except KeyError as error:
        raise ValueError(f"unknown Phase 2D cell: {name}") from error


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamped(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def consequence_target_vector(row: dict) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return normalized sequence-level consequence labels and field mask.

    These labels are privileged training/evaluation signals from the
    counterfactual generator. They are not runtime inputs. Nullable fields remain
    masked rather than being coerced into synthetic targets.
    """

    labels = row.get("consequence_labels")
    if not isinstance(labels, dict):
        return (
            tuple(0.0 for _ in CONSEQUENCE_TARGET_NAMES),
            tuple(False for _ in CONSEQUENCE_TARGET_NAMES),
        )

    values: list[float] = []
    mask: list[bool] = []

    progress = _finite_float(labels.get("target_progress_m"))
    values.append(0.0 if progress is None else _clamped(progress / 0.3, -1.0, 1.0))
    mask.append(progress is not None)

    clearance_gain = _finite_float(labels.get("clearance_gain_m"))
    values.append(
        0.0
        if clearance_gain is None
        else _clamped(clearance_gain / 0.3, -1.0, 1.0)
    )
    mask.append(clearance_gain is not None)

    minimum_clearance = _finite_float(
        labels.get("minimum_swept_configuration_clearance_m")
    )
    values.append(
        0.0
        if minimum_clearance is None
        else _clamped((minimum_clearance + 0.2) / 1.2, 0.0, 1.0)
    )
    mask.append(minimum_clearance is not None)

    p05_clearance = _finite_float(labels.get("p05_swept_configuration_clearance_m"))
    values.append(
        0.0
        if p05_clearance is None
        else _clamped((p05_clearance + 0.2) / 1.2, 0.0, 1.0)
    )
    mask.append(p05_clearance is not None)

    unsafe_fraction = _finite_float(labels.get("unsafe_sample_fraction"))
    values.append(
        0.0
        if unsafe_fraction is None
        else _clamped(unsafe_fraction, 0.0, 1.0)
    )
    mask.append(unsafe_fraction is not None)

    for key in ("enters_grid_unsafe", "ends_grid_unsafe", "target_recoverable"):
        if key in labels:
            values.append(1.0 if bool(labels[key]) else 0.0)
            mask.append(True)
        else:
            values.append(0.0)
            mask.append(False)

    heading_error = _finite_float(labels.get("target_heading_error_rad"))
    values.append(
        0.0
        if heading_error is None
        else 1.0 - _clamped(abs(heading_error), 0.0, math.pi) / math.pi
    )
    mask.append(heading_error is not None)

    return tuple(values), tuple(mask)


def action_utility_target(row: dict) -> tuple[float, bool]:
    """Return a scalar safety-first utility target for within-source ranking.

    The score is a differentiable proxy for the benchmark oracle ordering:
    unsafe entry, unsafe end state, and unrecoverability dominate; task progress
    or clearance gain, swept clearance, and heading alignment break ties. It is
    intended only for ranking candidates from the same source state.
    """

    labels = row.get("consequence_labels")
    if not isinstance(labels, dict):
        return 0.0, False

    clearance_gain = _finite_float(labels.get("clearance_gain_m"))
    progress = _finite_float(labels.get("target_progress_m"))
    task_gain = progress if progress is not None else clearance_gain
    p05_clearance = _finite_float(labels.get("p05_swept_configuration_clearance_m"))
    unsafe_fraction = _finite_float(labels.get("unsafe_sample_fraction"))
    heading_error = _finite_float(labels.get("target_heading_error_rad"))
    required_booleans = (
        "enters_grid_unsafe",
        "ends_grid_unsafe",
        "target_recoverable",
    )
    if (
        task_gain is None
        or p05_clearance is None
        or unsafe_fraction is None
        or any(key not in labels for key in required_booleans)
    ):
        return 0.0, False

    enters_unsafe = bool(labels["enters_grid_unsafe"])
    ends_unsafe = bool(labels["ends_grid_unsafe"])
    unrecoverable = labels["target_recoverable"] is False
    task_gain_norm = _clamped(task_gain / 0.3, -1.0, 1.0)
    p05_norm = _clamped((p05_clearance + 0.2) / 1.2, 0.0, 1.0)
    heading_penalty = (
        0.0
        if heading_error is None
        else _clamped(abs(heading_error), 0.0, math.pi) / math.pi
    )
    raw_utility = (
        2.0 * task_gain_norm
        + p05_norm
        - 8.0 * float(enters_unsafe)
        - 6.0 * float(ends_unsafe)
        - 4.0 * float(unrecoverable)
        - 2.0 * _clamped(unsafe_fraction, 0.0, 1.0)
        - 0.25 * heading_penalty
    )
    return raw_utility / 10.0, True


def image_tensor(path: Path, *, image_size: int = 224) -> torch.Tensor:
    """Load one RGB observation using the fixed Phase 2 image contract."""

    with Image.open(path) as image:
        resized = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array.copy())


@dataclass
class Phase2DBatch:
    """Materialized source-grouped batch with explicit masks and hard negatives."""

    row_indices: tuple[int, ...]
    rows: tuple[dict, ...]
    vision: torch.Tensor
    actions: torch.Tensor
    transition_mask: torch.Tensor
    wrong_actions: torch.Tensor
    wrong_mask: torch.Tensor
    non_hold_mask: torch.Tensor
    consequence_targets: torch.Tensor
    consequence_mask: torch.Tensor
    action_utility_targets: torch.Tensor
    action_utility_mask: torch.Tensor
    action_utility_group_ids: torch.Tensor

    def to(self, device: torch.device) -> Phase2DBatch:
        """Move tensors to one device without changing row provenance."""

        return Phase2DBatch(
            row_indices=self.row_indices,
            rows=self.rows,
            vision=self.vision.to(device),
            actions=self.actions.to(device),
            transition_mask=self.transition_mask.to(device),
            wrong_actions=self.wrong_actions.to(device),
            wrong_mask=self.wrong_mask.to(device),
            non_hold_mask=self.non_hold_mask.to(device),
            consequence_targets=self.consequence_targets.to(device),
            consequence_mask=self.consequence_mask.to(device),
            action_utility_targets=self.action_utility_targets.to(device),
            action_utility_mask=self.action_utility_mask.to(device),
            action_utility_group_ids=self.action_utility_group_ids.to(device),
        )


def _validate_horizon(rows: Sequence[dict], indices: Sequence[int]) -> tuple[int, int]:
    horizons = {len(rows[index]["active_blocks"]) for index in indices}
    if len(horizons) != 1:
        raise ValueError(f"batch rows must have one common horizon, got {sorted(horizons)}")
    horizon = next(iter(horizons))
    if horizon < 1:
        raise ValueError("Phase 2D rows must contain at least one action block")
    command_dims = {
        len(action_vector(rows[index], step))
        for index in indices
        for step in range(horizon)
    }
    if len(command_dims) != 1:
        raise ValueError("batch action vectors must have one common dimension")
    return horizon, next(iter(command_dims))


def materialize_phase2d_batch(
    rows: Sequence[dict],
    indices: Sequence[int],
    *,
    hard_negatives: Sequence[HardNegativeIndex],
    image_size: int = 224,
) -> Phase2DBatch:
    """Build one source-grouped batch with exhaustive in-batch hard negatives."""

    row_indices = tuple(int(index) for index in indices)
    if not row_indices:
        raise ValueError("cannot materialize an empty Phase 2D batch")
    horizon, command_dim = _validate_horizon(rows, row_indices)
    if len(hard_negatives) != horizon:
        raise ValueError("one hard-negative index is required for every horizon step")
    if any(index.step != step for step, index in enumerate(hard_negatives)):
        raise ValueError("hard-negative indexes must be ordered by step")
    batch_indices = set(row_indices)
    selected = tuple(rows[index] for index in row_indices)
    paths_and_masks = [materialized_frame_paths(row) for row in selected]
    image_cache: dict[Path, torch.Tensor] = {}

    def cached_image(path: Path) -> torch.Tensor:
        cached = image_cache.get(path)
        if cached is None:
            cached = image_tensor(path, image_size=image_size)
            image_cache[path] = cached
        return cached

    vision = torch.stack(
        [
            torch.stack([cached_image(path) for path in paths])
            for paths, _mask in paths_and_masks
        ]
    )
    actions = torch.tensor(
        [[action_vector(row, step) for step in range(horizon)] for row in selected],
        dtype=torch.float32,
    )
    transition_mask = torch.tensor(
        [mask for _paths, mask in paths_and_masks],
        dtype=torch.bool,
    )
    non_hold_mask = torch.tensor(
        [
            [not is_zero_action(row, step) for step in range(horizon)]
            for row in selected
        ],
        dtype=torch.bool,
    )
    consequence_values_and_masks = [consequence_target_vector(row) for row in selected]
    consequence_targets = torch.tensor(
        [values for values, _mask in consequence_values_and_masks],
        dtype=torch.float32,
    )
    consequence_mask = torch.tensor(
        [mask for _values, mask in consequence_values_and_masks],
        dtype=torch.bool,
    )
    utility_values_and_masks = [action_utility_target(row) for row in selected]
    action_utility_targets = torch.tensor(
        [value for value, _mask in utility_values_and_masks],
        dtype=torch.float32,
    )
    action_utility_mask = torch.tensor(
        [mask for _value, mask in utility_values_and_masks],
        dtype=torch.bool,
    )
    source_group_index: dict[tuple[str, int], int] = {}
    source_group_ids = []
    for row in selected:
        key = source_key(row)
        if key not in source_group_index:
            source_group_index[key] = len(source_group_index)
        source_group_ids.append(source_group_index[key])
    action_utility_group_ids = torch.tensor(source_group_ids, dtype=torch.long)

    maximum_negatives = max(
        (
            len(hard_negatives[step].candidates.get(index, ()))
            for index in row_indices
            for step in range(horizon)
        ),
        default=0,
    )
    negative_slots = max(1, maximum_negatives)
    wrong_actions = torch.zeros(
        len(selected),
        horizon,
        negative_slots,
        command_dim,
        dtype=torch.float32,
    )
    wrong_mask = torch.zeros(
        len(selected),
        horizon,
        negative_slots,
        dtype=torch.bool,
    )
    for local_index, global_index in enumerate(row_indices):
        for step in range(horizon):
            candidates = hard_negatives[step].candidates.get(global_index, ())
            missing = [index for index in candidates if index not in batch_indices]
            if missing:
                raise ValueError(
                    "source-grouped batch omitted registered hard negatives for "
                    f"row {global_index}, step {step + 1}: {missing[:4]}"
                )
            for slot, negative_index in enumerate(candidates):
                wrong_actions[local_index, step, slot] = torch.tensor(
                    action_vector(rows[negative_index], step),
                    dtype=torch.float32,
                )
                wrong_mask[local_index, step, slot] = True

    return Phase2DBatch(
        row_indices=row_indices,
        rows=selected,
        vision=vision,
        actions=actions,
        transition_mask=transition_mask,
        wrong_actions=wrong_actions,
        wrong_mask=wrong_mask,
        non_hold_mask=non_hold_mask,
        consequence_targets=consequence_targets,
        consequence_mask=consequence_mask,
        action_utility_targets=action_utility_targets,
        action_utility_mask=action_utility_mask,
        action_utility_group_ids=action_utility_group_ids,
    )


def batch_contract_audit(batch: Phase2DBatch) -> dict:
    """Return compact evidence that one materialized batch obeys the mask contract."""

    return {
        "rows": len(batch.rows),
        "horizon": int(batch.actions.shape[1]),
        "command_dim": int(batch.actions.shape[2]),
        "valid_transitions": int(batch.transition_mask.sum()),
        "non_hold_valid_transitions": int(
            (batch.non_hold_mask & batch.transition_mask).sum()
        ),
        "eligible_wrong_pairs": int(batch.wrong_mask.sum()),
        "eligible_wrong_transitions": int(batch.wrong_mask.any(dim=2).sum()),
        "invalid_transitions": int((~batch.transition_mask).sum()),
        "consequence_label_fields": int(batch.consequence_targets.shape[1]),
        "consequence_label_values": int(batch.consequence_mask.sum()),
        "action_utility_targets": int(batch.action_utility_mask.sum()),
        "action_utility_source_groups": int(
            torch.unique(batch.action_utility_group_ids).numel()
        ),
        "action_utility_target_version": ACTION_UTILITY_TARGET_VERSION,
        "all_materialized_frames_finite": bool(torch.isfinite(batch.vision).all()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _normalized_advantage(advantage: float, target_change: float) -> float | None:
    return advantage / target_change if target_change > 0.0 else None


def _mean_defined(records: Sequence[dict], key: str) -> float | None:
    values = [record[key] for record in records if record[key] is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def prediction_control_records(
    batch: Phase2DBatch,
    output: dict[str, torch.Tensor],
    *,
    seed: int,
) -> list[dict]:
    """Emit candidate-step prediction/control rows for registered estimands."""

    required = (
        "real_mse",
        "mean_wrong_mse",
        "zero_mse",
        "target_change_mse",
        "eligible_wrong_mask",
        "eligible_zero_mask",
        "transition_mask",
    )
    missing = [key for key in required if key not in output]
    if missing:
        raise ValueError(f"model output missing prediction-control tensors: {missing}")
    rows = []
    real_mse = output["real_mse"].detach().cpu()
    wrong_mse = output["mean_wrong_mse"].detach().cpu()
    zero_mse = output["zero_mse"].detach().cpu()
    target_change_mse = output["target_change_mse"].detach().cpu()
    valid = output["transition_mask"].detach().cpu().bool()
    eligible_wrong = output["eligible_wrong_mask"].detach().cpu().bool()
    eligible_zero = output["eligible_zero_mask"].detach().cpu().bool()
    for local_index, row in enumerate(batch.rows):
        scene_id, source_index = source_key(row)
        for step in range(batch.actions.shape[1]):
            if not bool(valid[local_index, step]):
                continue
            real = float(real_mse[local_index, step])
            persistence = float(target_change_mse[local_index, step])
            wrong = float(wrong_mse[local_index, step])
            zero = float(zero_mse[local_index, step])
            wrong_advantage = wrong - real
            zero_advantage = zero - real
            rows.append(
                {
                    "schema": "jepa_phase2d_candidate_step_prediction_control_v0",
                    "seed": int(seed),
                    "scene_id": scene_id,
                    "source_index": int(source_index),
                    "candidate_index": int(row.get("candidate_index", local_index)),
                    "row_index": int(batch.row_indices[local_index]),
                    "step": step + 1,
                    "primitive_name": action_name(row, step),
                    "action_is_zero": is_zero_action(row, step),
                    "valid_transition": True,
                    "eligible_wrong": bool(eligible_wrong[local_index, step]),
                    "eligible_zero": bool(eligible_zero[local_index, step]),
                    "real_mse": real,
                    "persistence_mse": persistence,
                    "mean_wrong_mse": wrong,
                    "zero_action_mse": zero,
                    "target_change_mse": persistence,
                    "real_vs_persistence_ratio": _safe_ratio(real, persistence),
                    "hard_negative_action_advantage": wrong_advantage,
                    "hard_negative_action_advantage_over_target_change": (
                        _normalized_advantage(wrong_advantage, persistence)
                    ),
                    "zero_action_advantage": zero_advantage,
                    "zero_action_advantage_over_target_change": (
                        _normalized_advantage(zero_advantage, persistence)
                    ),
                    "real_action_beats_persistence": real < persistence,
                    "real_action_beats_hard_negative": wrong_advantage > 0.0,
                    "real_action_beats_zero": zero_advantage > 0.0,
                }
            )
    return rows


def action_utility_selection_records(
    batch: Phase2DBatch,
    output: dict[str, torch.Tensor],
    *,
    seed: int,
) -> list[dict]:
    """Return source-local utility-selection diagnostics for one batch."""

    prediction = output.get("action_utility_prediction")
    if prediction is None:
        return []
    scores = prediction.detach().cpu().float()
    targets = batch.action_utility_targets.detach().cpu().float()
    valid = batch.action_utility_mask.detach().cpu().bool()
    group_ids = batch.action_utility_group_ids.detach().cpu()
    records = []
    for group_id in sorted(set(int(value) for value in group_ids.tolist())):
        indices = [
            index
            for index, value in enumerate(group_ids.tolist())
            if int(value) == group_id and bool(valid[index])
        ]
        if len(indices) < 2:
            continue
        local_scores = scores[indices]
        local_targets = targets[indices]
        selected_local = int(torch.argmax(local_scores))
        oracle_local = int(torch.argmax(local_targets))
        selected_index = indices[selected_local]
        oracle_index = indices[oracle_local]
        selected_row = batch.rows[selected_index]
        oracle_row = batch.rows[oracle_index]
        scene_id, source_index = source_key(selected_row)
        oracle_utility = float(local_targets[oracle_local])
        selected_utility = float(local_targets[selected_local])
        records.append(
            {
                "schema": "jepa_phase2g_source_action_utility_selection_v0",
                "seed": int(seed),
                "scene_id": scene_id,
                "source_index": int(source_index),
                "candidate_rows": len(indices),
                "selected_row_index": int(batch.row_indices[selected_index]),
                "oracle_row_index": int(batch.row_indices[oracle_index]),
                "selected_candidate_index": int(
                    selected_row.get("candidate_index", selected_index)
                ),
                "oracle_candidate_index": int(
                    oracle_row.get("candidate_index", oracle_index)
                ),
                "selected_first_primitive": action_name(selected_row, 0),
                "oracle_first_primitive": action_name(oracle_row, 0),
                "selected_sequence": list(selected_row.get("primitive_sequence", ())),
                "oracle_sequence": list(oracle_row.get("primitive_sequence", ())),
                "selected_predicted_utility": float(local_scores[selected_local]),
                "oracle_predicted_utility": float(local_scores[oracle_local]),
                "selected_target_utility": selected_utility,
                "oracle_target_utility": oracle_utility,
                "target_utility_regret": oracle_utility - selected_utility,
                "top1_match": selected_index == oracle_index,
                "first_primitive_match": (
                    action_name(selected_row, 0) == action_name(oracle_row, 0)
                ),
            }
        )
    return records


def action_utility_selection_summary(records: Sequence[dict]) -> dict | None:
    """Summarize utility selection over source states."""

    if not records:
        return None
    return {
        "schema": "jepa_phase2g_action_utility_selection_summary_v0",
        "source_state_count": len(records),
        "mean_candidate_rows": float(
            sum(int(record["candidate_rows"]) for record in records) / len(records)
        ),
        "top1_match_rate": float(
            sum(bool(record["top1_match"]) for record in records) / len(records)
        ),
        "first_primitive_match_rate": float(
            sum(bool(record["first_primitive_match"]) for record in records)
            / len(records)
        ),
        "mean_target_utility_regret": float(
            sum(float(record["target_utility_regret"]) for record in records)
            / len(records)
        ),
        "mean_selected_target_utility": float(
            sum(float(record["selected_target_utility"]) for record in records)
            / len(records)
        ),
        "mean_oracle_target_utility": float(
            sum(float(record["oracle_target_utility"]) for record in records)
            / len(records)
        ),
    }


def primary_source_state_prediction_table(
    candidate_step_records: Sequence[dict],
    *,
    primary_step: int = 1,
) -> list[dict]:
    """Aggregate valid candidate-step rows to one primary row per source state."""

    grouped: dict[tuple[int, str, int], list[dict]] = {}
    for record in candidate_step_records:
        if int(record["step"]) != primary_step:
            continue
        key = (
            int(record["seed"]),
            str(record["scene_id"]),
            int(record["source_index"]),
        )
        grouped.setdefault(key, []).append(record)
    result = []
    for (seed, scene_id, source_index), records in sorted(grouped.items()):
        means: dict[str, float | None] = {
            key: sum(float(record[key]) for record in records) / len(records)
            for key in (
                "real_mse",
                "persistence_mse",
                "mean_wrong_mse",
                "zero_action_mse",
                "hard_negative_action_advantage",
                "zero_action_advantage",
            )
        }
        means["hard_negative_action_advantage_over_target_change"] = _mean_defined(
            records,
            "hard_negative_action_advantage_over_target_change",
        )
        means["zero_action_advantage_over_target_change"] = _mean_defined(
            records,
            "zero_action_advantage_over_target_change",
        )
        result.append(
            {
                "schema": "jepa_phase2d_primary_source_state_prediction_control_v0",
                "seed": seed,
                "scene_id": scene_id,
                "source_index": source_index,
                "step": primary_step,
                "candidate_rows": len(records),
                "eligible_wrong_candidate_rows": sum(
                    bool(record["eligible_wrong"]) for record in records
                ),
                "eligible_zero_candidate_rows": sum(
                    bool(record["eligible_zero"]) for record in records
                ),
                **means,
                "one_step_rollout_persistence_ratio": _safe_ratio(
                    means["real_mse"],
                    means["persistence_mse"],
                ),
            }
        )
    return result


REGISTERED_ACTION_ADVANTAGE_THRESHOLD = 0.10
REGISTERED_PERSISTENCE_RATIO_THRESHOLD = 1.0


def checkpoint_rule_record(
    source_state_records: Sequence[dict],
    *,
    epoch: int,
    stability: dict,
) -> dict:
    """Summarize one validation checkpoint in the registered rule format."""

    if not source_state_records:
        raise ValueError("checkpoint rule requires source-state records")
    hard_negative_values = [
        float(record["hard_negative_action_advantage_over_target_change"])
        for record in source_state_records
        if record["hard_negative_action_advantage_over_target_change"] is not None
    ]
    zero_action_values = [
        float(record["zero_action_advantage_over_target_change"])
        for record in source_state_records
        if record["zero_action_advantage_over_target_change"] is not None
    ]
    ratio_values = [
        float(record["one_step_rollout_persistence_ratio"])
        for record in source_state_records
        if record["one_step_rollout_persistence_ratio"] is not None
    ]
    if not hard_negative_values or not zero_action_values or not ratio_values:
        raise ValueError("checkpoint rule requires defined primary estimands")
    hard_negative_advantage = sum(hard_negative_values) / len(hard_negative_values)
    zero_action_advantage = sum(zero_action_values) / len(zero_action_values)
    ratio = sum(ratio_values) / len(ratio_values)
    stability_pass = not any(
        bool(stability.get(key, False))
        for key in (
            "collapse_warning",
            "effective_rank_warning",
            "near_static_target_warning",
        )
    )
    hard_negative_action_advantage_pass = (
        hard_negative_advantage >= REGISTERED_ACTION_ADVANTAGE_THRESHOLD
    )
    zero_action_advantage_pass = (
        zero_action_advantage >= REGISTERED_ACTION_ADVANTAGE_THRESHOLD
    )
    persistence_pass = ratio < REGISTERED_PERSISTENCE_RATIO_THRESHOLD
    gate_pass = (
        stability_pass
        and hard_negative_action_advantage_pass
        and zero_action_advantage_pass
        and persistence_pass
    )
    return {
        "epoch": int(epoch),
        "stability_pass": stability_pass,
        "hard_negative_action_advantage": hard_negative_advantage,
        "zero_action_advantage": zero_action_advantage,
        "one_step_rollout_persistence_ratio": ratio,
        "hard_negative_action_advantage_pass": hard_negative_action_advantage_pass,
        "zero_action_advantage_pass": zero_action_advantage_pass,
        "persistence_pass": persistence_pass,
        "gate_pass": gate_pass,
        "registered_action_advantage_threshold": (
            REGISTERED_ACTION_ADVANTAGE_THRESHOLD
        ),
        "registered_persistence_ratio_threshold": (
            REGISTERED_PERSISTENCE_RATIO_THRESHOLD
        ),
        "source_state_count": len(source_state_records),
        "stability": {
            key: stability.get(key)
            for key in (
                "collapse_warning",
                "effective_rank_warning",
                "near_static_target_warning",
                "mean_feature_std",
                "effective_rank",
                "effective_rank_fraction",
                "target_change_over_feature_variance",
            )
            if key in stability
        },
    }
