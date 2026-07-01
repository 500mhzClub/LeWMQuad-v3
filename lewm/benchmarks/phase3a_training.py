"""Training helpers for the Phase 3A positive-control JEPA task."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch

from .phase3a_marker_memory import (
    egocentric_marker_memory_score,
    marker_memory_start_cell_target,
    marker_memory_start_target,
    marker_memory_target,
)
from .phase3a_positive_control import DEFAULT_RENDER_PALETTE

CONSEQUENCE_TARGET_NAMES = (
    "step_collision",
    "cumulative_collision_fraction",
    "step_progress_fraction",
    "cumulative_progress_fraction",
    "reached_goal",
    "safe_recoverable",
    "target_utility_scaled",
)


@dataclass(frozen=True)
class Phase3ABatch:
    """Materialized Phase 3A batch."""

    vision: torch.Tensor
    history_vision: torch.Tensor
    history_actions: torch.Tensor
    actions: torch.Tensor
    utility_targets: torch.Tensor
    consequence_targets: torch.Tensor
    utility_group_ids: torch.Tensor
    utility_mask: torch.Tensor
    wrong_actions: torch.Tensor
    wrong_mask: torch.Tensor
    non_hold_mask: torch.Tensor
    marker_memory_valid_mask: torch.Tensor
    marker_memory_delta_targets: torch.Tensor
    marker_memory_claim_targets: torch.Tensor
    marker_memory_score_targets: torch.Tensor
    marker_memory_start_valid_mask: torch.Tensor
    marker_memory_start_delta_targets: torch.Tensor
    marker_memory_start_cell_valid_mask: torch.Tensor
    marker_memory_start_cell_targets: torch.Tensor
    spatial_frontier_history_observation_targets: torch.Tensor
    spatial_frontier_vision_observation_targets: torch.Tensor
    source_keys: tuple[tuple[str, int], ...]
    first_primitives: tuple[str, ...]

    def to(self, device: torch.device) -> Phase3ABatch:
        return Phase3ABatch(
            vision=self.vision.to(device),
            history_vision=self.history_vision.to(device),
            history_actions=self.history_actions.to(device),
            actions=self.actions.to(device),
            utility_targets=self.utility_targets.to(device),
            consequence_targets=self.consequence_targets.to(device),
            utility_group_ids=self.utility_group_ids.to(device),
            utility_mask=self.utility_mask.to(device),
            wrong_actions=self.wrong_actions.to(device),
            wrong_mask=self.wrong_mask.to(device),
            non_hold_mask=self.non_hold_mask.to(device),
            marker_memory_valid_mask=self.marker_memory_valid_mask.to(device),
            marker_memory_delta_targets=self.marker_memory_delta_targets.to(device),
            marker_memory_claim_targets=self.marker_memory_claim_targets.to(device),
            marker_memory_score_targets=self.marker_memory_score_targets.to(device),
            marker_memory_start_valid_mask=self.marker_memory_start_valid_mask.to(device),
            marker_memory_start_delta_targets=(
                self.marker_memory_start_delta_targets.to(device)
            ),
            marker_memory_start_cell_valid_mask=(
                self.marker_memory_start_cell_valid_mask.to(device)
            ),
            marker_memory_start_cell_targets=(
                self.marker_memory_start_cell_targets.to(device)
            ),
            spatial_frontier_history_observation_targets=(
                self.spatial_frontier_history_observation_targets.to(device)
            ),
            spatial_frontier_vision_observation_targets=(
                self.spatial_frontier_vision_observation_targets.to(device)
            ),
            source_keys=self.source_keys,
            first_primitives=self.first_primitives,
        )


@dataclass(frozen=True)
class _Phase3AMaterializedRow:
    vision: torch.Tensor
    history_vision: torch.Tensor
    history_actions: torch.Tensor
    actions: torch.Tensor
    utility_target: torch.Tensor
    consequence_targets: torch.Tensor
    wrong_actions: torch.Tensor
    wrong_mask: torch.Tensor
    non_hold_mask: torch.Tensor
    marker_memory_valid_mask: torch.Tensor
    marker_memory_delta_targets: torch.Tensor
    marker_memory_claim_targets: torch.Tensor
    marker_memory_score_targets: torch.Tensor
    marker_memory_start_valid_mask: torch.Tensor
    marker_memory_start_delta_targets: torch.Tensor
    marker_memory_start_cell_valid_mask: torch.Tensor
    marker_memory_start_cell_targets: torch.Tensor
    spatial_frontier_history_observation_targets: torch.Tensor
    spatial_frontier_vision_observation_targets: torch.Tensor
    source_key: tuple[str, int]
    first_primitive: str


def source_key(row: dict) -> tuple[str, int]:
    return str(row["scene_id"]), int(row["source_index"])


def source_grouped_batches(
    rows: Sequence[dict],
    *,
    source_states_per_batch: int,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Return row-index batches that keep same-source candidates together."""

    import random

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    keys = sorted(grouped)
    if shuffle:
        random.Random(seed).shuffle(keys)
    batches = []
    for offset in range(0, len(keys), source_states_per_batch):
        selected = keys[offset : offset + source_states_per_batch]
        batches.append(tuple(index for key in selected for index in grouped[key]))
    return tuple(batches)


def _vision_sequence(row: dict) -> torch.Tensor:
    frames = [row["start_observation_rgb"]]
    frames.extend(
        observation["observation_rgb"]
        for observation in row["future_observations"]
    )
    return torch.tensor(frames, dtype=torch.float32)


def _history_vision_sequence(row: dict) -> torch.Tensor:
    frames = row.get("history_observations_rgb", [])
    if frames:
        return torch.tensor(frames, dtype=torch.float32)
    start = torch.tensor(row["start_observation_rgb"], dtype=torch.float32)
    return start.new_empty((0, *start.shape))


def _history_action_sequence(row: dict, *, action_dim: int) -> torch.Tensor:
    actions = row.get("history_actions", [])
    if actions:
        return torch.tensor(actions, dtype=torch.float32)
    return torch.empty((0, action_dim), dtype=torch.float32)


def _row_render_palette(row: dict) -> dict[str, tuple[float, float, float]]:
    palette = row.get("render_palette") or DEFAULT_RENDER_PALETTE
    return {
        str(key): tuple(float(channel) for channel in value)
        for key, value in palette.items()
    }


def _color_mask(
    frame: torch.Tensor,
    color: tuple[float, float, float],
    *,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    target = frame.new_tensor(color).view(3, 1, 1)
    return ((frame - target).square().sum(dim=0) <= tolerance).to(dtype=frame.dtype)


def _spatial_frontier_observation_targets(
    row: dict,
    frames: Sequence,
) -> torch.Tensor:
    """Return marker/observed/free/blocked maps for RGB observation frames."""

    reference = frames[0] if frames else row["start_observation_rgb"]
    height = len(reference[0])
    width = len(reference[0][0])
    if not frames:
        return torch.empty((0, 4, height, width), dtype=torch.float32)
    palette = _row_render_palette(row)
    targets = []
    for frame_value in frames:
        frame = torch.tensor(frame_value, dtype=torch.float32)
        marker = _color_mask(frame, palette["goal"])
        blocked = torch.maximum(
            _color_mask(frame, palette["outside"]),
            _color_mask(frame, palette["wall"]),
        )
        observed = torch.ones_like(marker)
        free = (1.0 - blocked).clamp(0.0, 1.0)
        targets.append(torch.stack([marker, observed, free, blocked], dim=0))
    return torch.stack(targets, dim=0)


def _consequence_sequence(row: dict) -> torch.Tensor:
    labels = row["consequence_labels"]
    observations = row["future_observations"]
    horizon = len(observations)
    if horizon < 1:
        raise ValueError("future_observations must be non-empty")
    progress = float(labels["target_progress_cells"]) / float(horizon)
    utility = float(labels["target_utility"]) / float(horizon + 5)
    reached_goal = float(bool(labels["reached_goal"]))
    safe_recoverable = float(bool(labels["safe_recoverable"]))
    cumulative_collisions = 0
    cumulative_progress = 0.0
    targets = []
    for observation in observations:
        step_collision = float(bool(observation["collision"]))
        cumulative_collisions += int(step_collision)
        step_progress = float(observation.get("step_progress_cells", progress))
        cumulative_progress += step_progress
        targets.append(
            [
                step_collision,
                float(cumulative_collisions) / float(horizon),
                step_progress / float(horizon),
                cumulative_progress / float(horizon),
                reached_goal,
                safe_recoverable,
                utility,
            ]
        )
    return torch.tensor(targets, dtype=torch.float32)


def _same_source_negative(
    rows: Sequence[dict],
    index: int,
    *,
    step: int,
    grouped: dict[tuple[str, int], list[int]],
) -> tuple[float, ...] | None:
    row = rows[index]
    action = tuple(float(value) for value in row["active_blocks"][step])
    for candidate in grouped[source_key(row)]:
        if candidate == index:
            continue
        other = tuple(float(value) for value in rows[candidate]["active_blocks"][step])
        if other != action:
            return other
    return None


class Phase3AMaterializedDataset:
    """Reusable tensor cache for Phase 3A rows.

    The public JSON row format is convenient for data generation and audits, but
    repeatedly converting nested row fields to tensors dominates small-batch
    GPU training. This cache preserves the existing batch contract while doing
    the per-row tensorization and same-source hard-negative lookup once.
    """

    def __init__(self, rows: Sequence[dict]) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        self.rows = rows
        self.grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.grouped[source_key(row)].append(index)
        self.items = tuple(self._materialize_row(index) for index in range(len(rows)))

    def _materialize_row(self, index: int) -> _Phase3AMaterializedRow:
        row = self.rows[index]
        horizon = len(row["active_blocks"])
        action_dim = len(row["active_blocks"][0])
        actions = torch.tensor(
            [block for block in row["active_blocks"]],
            dtype=torch.float32,
        )
        valid, delta, claimed = marker_memory_target(row)
        start_valid, start_delta = marker_memory_start_target(row)
        start_cell_valid, start_cell = marker_memory_start_cell_target(row)
        vision_frames = [row["start_observation_rgb"]]
        vision_frames.extend(
            observation["observation_rgb"]
            for observation in row["future_observations"]
        )
        wrong_actions = torch.zeros((horizon, 1, action_dim), dtype=torch.float32)
        wrong_mask = torch.zeros((horizon, 1), dtype=torch.bool)
        non_hold_mask = torch.zeros((horizon,), dtype=torch.bool)
        for step, name in enumerate(row["primitive_sequence"]):
            non_hold_mask[step] = str(name) != "hold"
            negative = _same_source_negative(
                self.rows,
                index,
                step=step,
                grouped=self.grouped,
            )
            if negative is not None:
                wrong_actions[step, 0] = torch.tensor(negative)
                wrong_mask[step, 0] = True
        return _Phase3AMaterializedRow(
            vision=_vision_sequence(row),
            history_vision=_history_vision_sequence(row),
            history_actions=_history_action_sequence(row, action_dim=action_dim),
            actions=actions,
            utility_target=torch.tensor(
                float(row["consequence_labels"]["target_utility"]),
                dtype=torch.float32,
            ),
            consequence_targets=_consequence_sequence(row),
            wrong_actions=wrong_actions,
            wrong_mask=wrong_mask,
            non_hold_mask=non_hold_mask,
            marker_memory_valid_mask=torch.tensor(bool(valid), dtype=torch.bool),
            marker_memory_delta_targets=torch.tensor(delta, dtype=torch.float32),
            marker_memory_claim_targets=torch.tensor(float(claimed), dtype=torch.float32),
            marker_memory_score_targets=torch.tensor(
                float(egocentric_marker_memory_score(row)),
                dtype=torch.float32,
            ),
            marker_memory_start_valid_mask=torch.tensor(
                bool(start_valid),
                dtype=torch.bool,
            ),
            marker_memory_start_delta_targets=torch.tensor(
                start_delta,
                dtype=torch.float32,
            ),
            marker_memory_start_cell_valid_mask=torch.tensor(
                bool(start_cell_valid),
                dtype=torch.bool,
            ),
            marker_memory_start_cell_targets=torch.tensor(
                int(start_cell),
                dtype=torch.long,
            ),
            spatial_frontier_history_observation_targets=(
                _spatial_frontier_observation_targets(
                    row,
                    row.get("history_observations_rgb", []),
                )
            ),
            spatial_frontier_vision_observation_targets=(
                _spatial_frontier_observation_targets(row, vision_frames)
            ),
            source_key=source_key(row),
            first_primitive=str(row["primitive_sequence"][0]),
        )

    def materialize_batch(self, indices: Sequence[int]) -> Phase3ABatch:
        if not indices:
            raise ValueError("indices must not be empty")
        selected = [self.items[index] for index in indices]
        history_lengths = {item.history_vision.shape[0] for item in selected}
        history_action_lengths = {item.history_actions.shape[0] for item in selected}
        if len(history_lengths) != 1 or history_lengths != history_action_lengths:
            raise ValueError("selected Phase 3A rows must share one history length")
        group_index = {
            key: index
            for index, key in enumerate(sorted({item.source_key for item in selected}))
        }
        return Phase3ABatch(
            vision=torch.stack([item.vision for item in selected]),
            history_vision=torch.stack([item.history_vision for item in selected]),
            history_actions=torch.stack([item.history_actions for item in selected]),
            actions=torch.stack([item.actions for item in selected]),
            utility_targets=torch.stack([item.utility_target for item in selected]),
            consequence_targets=torch.stack(
                [item.consequence_targets for item in selected]
            ),
            utility_group_ids=torch.tensor(
                [group_index[item.source_key] for item in selected],
                dtype=torch.long,
            ),
            utility_mask=torch.ones((len(indices),), dtype=torch.bool),
            wrong_actions=torch.stack([item.wrong_actions for item in selected]),
            wrong_mask=torch.stack([item.wrong_mask for item in selected]),
            non_hold_mask=torch.stack([item.non_hold_mask for item in selected]),
            marker_memory_valid_mask=torch.stack(
                [item.marker_memory_valid_mask for item in selected]
            ),
            marker_memory_delta_targets=torch.stack(
                [item.marker_memory_delta_targets for item in selected]
            ),
            marker_memory_claim_targets=torch.stack(
                [item.marker_memory_claim_targets for item in selected]
            ),
            marker_memory_score_targets=torch.stack(
                [item.marker_memory_score_targets for item in selected]
            ),
            marker_memory_start_valid_mask=torch.stack(
                [item.marker_memory_start_valid_mask for item in selected]
            ),
            marker_memory_start_delta_targets=torch.stack(
                [item.marker_memory_start_delta_targets for item in selected]
            ),
            marker_memory_start_cell_valid_mask=torch.stack(
                [item.marker_memory_start_cell_valid_mask for item in selected]
            ),
            marker_memory_start_cell_targets=torch.stack(
                [item.marker_memory_start_cell_targets for item in selected]
            ),
            spatial_frontier_history_observation_targets=torch.stack(
                [
                    item.spatial_frontier_history_observation_targets
                    for item in selected
                ]
            ),
            spatial_frontier_vision_observation_targets=torch.stack(
                [
                    item.spatial_frontier_vision_observation_targets
                    for item in selected
                ]
            ),
            source_keys=tuple(item.source_key for item in selected),
            first_primitives=tuple(item.first_primitive for item in selected),
        )


def materialize_phase3a_batch(rows: Sequence[dict], indices: Sequence[int]) -> Phase3ABatch:
    """Materialize rows into tensors plus deterministic same-source hard negatives."""

    return Phase3AMaterializedDataset(rows).materialize_batch(indices)


def materialize_phase3a_batch_uncached(
    rows: Sequence[dict],
    indices: Sequence[int],
) -> Phase3ABatch:
    """Legacy uncached implementation kept for equivalence tests."""

    if not indices:
        raise ValueError("indices must not be empty")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    selected = [rows[index] for index in indices]
    horizon = len(selected[0]["active_blocks"])
    action_dim = len(selected[0]["active_blocks"][0])
    vision = torch.stack([_vision_sequence(row) for row in selected])
    history_vision_items = [_history_vision_sequence(row) for row in selected]
    history_action_items = [
        _history_action_sequence(row, action_dim=action_dim) for row in selected
    ]
    history_lengths = {item.shape[0] for item in history_vision_items}
    history_action_lengths = {item.shape[0] for item in history_action_items}
    if len(history_lengths) != 1 or history_lengths != history_action_lengths:
        raise ValueError("selected Phase 3A rows must share one history length")
    history_vision = torch.stack(history_vision_items)
    history_actions = torch.stack(history_action_items)
    actions = torch.tensor(
        [[block for block in row["active_blocks"]] for row in selected],
        dtype=torch.float32,
    )
    utility_targets = torch.tensor(
        [float(row["consequence_labels"]["target_utility"]) for row in selected],
        dtype=torch.float32,
    )
    consequence_targets = torch.stack([_consequence_sequence(row) for row in selected])
    group_index = {
        key: index
        for index, key in enumerate(sorted({source_key(row) for row in selected}))
    }
    utility_group_ids = torch.tensor(
        [group_index[source_key(row)] for row in selected],
        dtype=torch.long,
    )
    utility_mask = torch.ones((len(indices),), dtype=torch.bool)
    wrong_actions = torch.zeros(
        (len(indices), horizon, 1, action_dim),
        dtype=torch.float32,
    )
    wrong_mask = torch.zeros((len(indices), horizon, 1), dtype=torch.bool)
    non_hold_mask = torch.zeros((len(indices), horizon), dtype=torch.bool)
    marker_memory_valid_mask = torch.zeros((len(indices),), dtype=torch.bool)
    marker_memory_delta_targets = torch.zeros((len(indices), 2), dtype=torch.float32)
    marker_memory_claim_targets = torch.zeros((len(indices),), dtype=torch.float32)
    marker_memory_score_targets = torch.zeros((len(indices),), dtype=torch.float32)
    marker_memory_start_valid_mask = torch.zeros((len(indices),), dtype=torch.bool)
    marker_memory_start_delta_targets = torch.zeros(
        (len(indices), 2),
        dtype=torch.float32,
    )
    marker_memory_start_cell_valid_mask = torch.zeros((len(indices),), dtype=torch.bool)
    marker_memory_start_cell_targets = torch.zeros((len(indices),), dtype=torch.long)
    spatial_frontier_history_observation_targets = torch.stack(
        [
            _spatial_frontier_observation_targets(
                row,
                row.get("history_observations_rgb", []),
            )
            for row in selected
        ]
    )
    spatial_frontier_vision_observation_targets = torch.stack(
        [
            _spatial_frontier_observation_targets(
                row,
                [row["start_observation_rgb"]]
                + [
                    observation["observation_rgb"]
                    for observation in row["future_observations"]
                ],
            )
            for row in selected
        ]
    )
    for batch_index, row_index in enumerate(indices):
        row = rows[row_index]
        valid, delta, claimed = marker_memory_target(row)
        start_valid, start_delta = marker_memory_start_target(row)
        start_cell_valid, start_cell = marker_memory_start_cell_target(row)
        marker_memory_valid_mask[batch_index] = bool(valid)
        marker_memory_delta_targets[batch_index] = torch.tensor(delta)
        marker_memory_claim_targets[batch_index] = float(claimed)
        marker_memory_score_targets[batch_index] = float(
            egocentric_marker_memory_score(row)
        )
        marker_memory_start_valid_mask[batch_index] = bool(start_valid)
        marker_memory_start_delta_targets[batch_index] = torch.tensor(start_delta)
        marker_memory_start_cell_valid_mask[batch_index] = bool(start_cell_valid)
        marker_memory_start_cell_targets[batch_index] = int(start_cell)
        for step, name in enumerate(row["primitive_sequence"]):
            non_hold_mask[batch_index, step] = str(name) != "hold"
            negative = _same_source_negative(
                rows,
                row_index,
                step=step,
                grouped=grouped,
            )
            if negative is not None:
                wrong_actions[batch_index, step, 0] = torch.tensor(negative)
                wrong_mask[batch_index, step, 0] = True
    return Phase3ABatch(
        vision=vision,
        history_vision=history_vision,
        history_actions=history_actions,
        actions=actions,
        utility_targets=utility_targets,
        consequence_targets=consequence_targets,
        utility_group_ids=utility_group_ids,
        utility_mask=utility_mask,
        wrong_actions=wrong_actions,
        wrong_mask=wrong_mask,
        non_hold_mask=non_hold_mask,
        marker_memory_valid_mask=marker_memory_valid_mask,
        marker_memory_delta_targets=marker_memory_delta_targets,
        marker_memory_claim_targets=marker_memory_claim_targets,
        marker_memory_score_targets=marker_memory_score_targets,
        marker_memory_start_valid_mask=marker_memory_start_valid_mask,
        marker_memory_start_delta_targets=marker_memory_start_delta_targets,
        marker_memory_start_cell_valid_mask=marker_memory_start_cell_valid_mask,
        marker_memory_start_cell_targets=marker_memory_start_cell_targets,
        spatial_frontier_history_observation_targets=(
            spatial_frontier_history_observation_targets
        ),
        spatial_frontier_vision_observation_targets=(
            spatial_frontier_vision_observation_targets
        ),
        source_keys=tuple(source_key(row) for row in selected),
        first_primitives=tuple(str(row["primitive_sequence"][0]) for row in selected),
    )


def primitive_selection_summary(
    rows: Sequence[dict],
    utility_predictions: Sequence[float],
) -> dict:
    """Score predicted sequence utilities as first-primitive decisions."""

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    selected_counts: dict[str, int] = defaultdict(int)
    oracle_counts: dict[str, int] = defaultdict(int)
    matches = 0
    regret = 0.0
    sequence_regret = 0.0
    for indices in grouped.values():
        predicted_index = max(indices, key=lambda item: float(utility_predictions[item]))
        best_by_first: dict[str, float] = {}
        oracle_index = indices[0]
        for index in indices:
            primitive = str(rows[index]["primitive_sequence"][0])
            utility = float(rows[index]["consequence_labels"]["target_utility"])
            best_by_first[primitive] = max(utility, best_by_first.get(primitive, utility))
            if utility > float(rows[oracle_index]["consequence_labels"]["target_utility"]):
                oracle_index = index
        selected = str(rows[predicted_index]["primitive_sequence"][0])
        oracle = str(rows[oracle_index]["primitive_sequence"][0])
        selected_counts[selected] += 1
        oracle_counts[oracle] += 1
        matches += int(selected == oracle)
        oracle_utility = float(rows[oracle_index]["consequence_labels"]["target_utility"])
        regret += oracle_utility - float(best_by_first[selected])
        sequence_regret += oracle_utility - float(
            rows[predicted_index]["consequence_labels"]["target_utility"]
        )
    count = max(len(grouped), 1)
    selected_max_fraction = max(selected_counts.values(), default=0) / count
    oracle_max_fraction = max(oracle_counts.values(), default=0) / count
    return {
        "source_states": len(grouped),
        "primitive_match_rate": matches / count,
        "mean_target_utility_regret": regret / count,
        "mean_selected_sequence_target_utility_regret": sequence_regret / count,
        "selected_primitive_counts": dict(sorted(selected_counts.items())),
        "oracle_primitive_counts": dict(sorted(oracle_counts.items())),
        "selected_max_primitive_fraction": selected_max_fraction,
        "oracle_max_primitive_fraction": oracle_max_fraction,
    }
