"""Phase 2I source-conditioned action-utility training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .phase2_data import action_vector, source_key
from .phase2d_training import action_utility_target, image_tensor


@dataclass
class Phase2IUtilityBatch:
    """Materialized source-grouped utility batch using only the start image."""

    row_indices: tuple[int, ...]
    rows: tuple[dict, ...]
    start_vision: torch.Tensor
    actions: torch.Tensor
    action_utility_targets: torch.Tensor
    action_utility_mask: torch.Tensor
    action_utility_group_ids: torch.Tensor

    def to(self, device: torch.device) -> Phase2IUtilityBatch:
        return Phase2IUtilityBatch(
            row_indices=self.row_indices,
            rows=self.rows,
            start_vision=self.start_vision.to(device),
            actions=self.actions.to(device),
            action_utility_targets=self.action_utility_targets.to(device),
            action_utility_mask=self.action_utility_mask.to(device),
            action_utility_group_ids=self.action_utility_group_ids.to(device),
        )


def _validate_action_shape(
    rows: Sequence[dict],
    indices: Sequence[int],
) -> tuple[int, int]:
    horizons = {len(rows[index]["active_blocks"]) for index in indices}
    if len(horizons) != 1:
        raise ValueError(f"batch rows must have one horizon, got {sorted(horizons)}")
    horizon = next(iter(horizons))
    if horizon < 1:
        raise ValueError("utility rows must contain at least one action block")
    command_dims = {
        len(action_vector(rows[index], step))
        for index in indices
        for step in range(horizon)
    }
    if len(command_dims) != 1:
        raise ValueError("batch action vectors must have one common dimension")
    return horizon, next(iter(command_dims))


def materialize_phase2i_utility_batch(
    rows: Sequence[dict],
    indices: Sequence[int],
    *,
    image_size: int = 224,
) -> Phase2IUtilityBatch:
    """Build one source-grouped start-observation/action utility batch."""

    row_indices = tuple(int(index) for index in indices)
    if not row_indices:
        raise ValueError("cannot materialize an empty Phase 2I utility batch")
    horizon, _command_dim = _validate_action_shape(rows, row_indices)
    selected = tuple(rows[index] for index in row_indices)
    image_cache: dict[Path, torch.Tensor] = {}

    def cached_image(path: Path) -> torch.Tensor:
        cached = image_cache.get(path)
        if cached is None:
            cached = image_tensor(path, image_size=image_size)
            image_cache[path] = cached
        return cached

    start_vision = torch.stack(
        [cached_image(Path(str(row["start_frame"]))) for row in selected]
    )
    actions = torch.tensor(
        [[action_vector(row, step) for step in range(horizon)] for row in selected],
        dtype=torch.float32,
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
    return Phase2IUtilityBatch(
        row_indices=row_indices,
        rows=selected,
        start_vision=start_vision,
        actions=actions,
        action_utility_targets=action_utility_targets,
        action_utility_mask=action_utility_mask,
        action_utility_group_ids=action_utility_group_ids,
    )


def phase2i_batch_contract_audit(batch: Phase2IUtilityBatch) -> dict:
    """Return compact evidence for one Phase 2I materialized batch."""

    return {
        "schema": "jepa_phase2i_utility_batch_contract_v0",
        "rows": len(batch.rows),
        "horizon": int(batch.actions.shape[1]),
        "command_dim": int(batch.actions.shape[2]),
        "action_utility_targets": int(batch.action_utility_mask.sum()),
        "action_utility_source_groups": int(
            torch.unique(batch.action_utility_group_ids).numel()
        ),
        "all_start_frames_finite": bool(torch.isfinite(batch.start_vision).all()),
    }
