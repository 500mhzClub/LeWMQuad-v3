"""Reachability targets for Phase 3B egocentric memory experiments.

Phase 3A's best controllers show that the map is often present, while the
remaining failures are mostly readout/action-selection failures.  These helpers
make the next target explicit: the latent memory should encode which cells are
reachable and how far useful target/frontier cells are through free space.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

Cell = tuple[int, int]


@dataclass(frozen=True)
class Phase3BReachabilityTarget:
    """Dense egocentric reachability supervision for one memory state."""

    reachable_mask: torch.Tensor
    current_distance: torch.Tensor
    target_distance: torch.Tensor
    target_value: torch.Tensor
    target_mask: torch.Tensor
    frontier_mask: torch.Tensor
    target_cells: tuple[Cell, ...]


@dataclass(frozen=True)
class Phase3BPairwiseReachabilityTarget:
    """Pairwise shortest-path supervision for sampled egocentric cells."""

    cells: tuple[Cell, ...]
    distances: torch.Tensor
    reachable_mask: torch.Tensor


@dataclass(frozen=True)
class Phase3BReachabilityTargetBatch:
    """Stacked dense reachability targets for training."""

    reachable_mask: torch.Tensor
    current_distance: torch.Tensor
    target_distance: torch.Tensor
    target_value: torch.Tensor
    target_mask: torch.Tensor
    frontier_mask: torch.Tensor

    def to(self, device: torch.device) -> Phase3BReachabilityTargetBatch:
        return Phase3BReachabilityTargetBatch(
            reachable_mask=self.reachable_mask.to(device),
            current_distance=self.current_distance.to(device),
            target_distance=self.target_distance.to(device),
            target_value=self.target_value.to(device),
            target_mask=self.target_mask.to(device),
            frontier_mask=self.frontier_mask.to(device),
        )


def egocentric_neighbors(cell: Cell) -> tuple[Cell, ...]:
    """Return four-neighbor cells in Phase 3A egocentric coordinates."""

    ahead, lateral = cell
    return (
        (ahead + 1, lateral),
        (ahead, lateral + 1),
        (ahead, lateral - 1),
        (ahead - 1, lateral),
    )


def cell_to_row_col(cell: Cell, *, memory_size: int) -> tuple[int, int] | None:
    """Map an egocentric cell to a dense ``(row, col)`` tensor index."""

    if memory_size < 3 or memory_size % 2 == 0:
        raise ValueError("memory_size must be an odd integer >= 3")
    radius = memory_size // 2
    row = radius - int(cell[0])
    col = radius + int(cell[1])
    if 0 <= row < memory_size and 0 <= col < memory_size:
        return row, col
    return None


def egocentric_frontier_cells(memory: Mapping) -> tuple[Cell, ...]:
    """Return known-free cells adjacent to at least one unknown cell."""

    free, blocked, _marker = _memory_sets(memory)
    frontiers = [
        cell
        for cell in sorted(free)
        if any(
            neighbor not in free and neighbor not in blocked
            for neighbor in egocentric_neighbors(cell)
        )
    ]
    return tuple(frontiers)


def select_reachability_target_cells(
    memory: Mapping,
    *,
    mode: str = "marker_or_frontier",
) -> tuple[Cell, ...]:
    """Select target cells for discover-then-return reachability supervision."""

    if mode not in {"marker", "frontier", "marker_or_frontier"}:
        raise ValueError("mode must be 'marker', 'frontier', or 'marker_or_frontier'")
    free, blocked, marker = _memory_sets(memory)
    if mode in {"marker", "marker_or_frontier"} and marker is not None:
        if marker not in blocked:
            return (marker,)
        if mode == "marker":
            return ()
    if mode in {"frontier", "marker_or_frontier"}:
        frontiers = tuple(
            cell
            for cell in egocentric_frontier_cells(memory)
            if cell not in blocked
        )
        if frontiers:
            return frontiers
    return ((0, 0),) if (0, 0) in free or mode == "marker_or_frontier" else ()


def build_reachability_target(
    memory: Mapping,
    *,
    memory_size: int,
    target_cells: Sequence[Cell] | None = None,
    target_mode: str = "marker_or_frontier",
    gamma: float = 0.94,
    unreachable_distance: float | None = None,
) -> Phase3BReachabilityTarget:
    """Build dense reachability, distance, and target-value maps.

    ``target_value`` is a discounted shortest-path value to the selected target
    cells, masked to cells reachable from the current egocentric origin.  This
    is the Phase 3B bridge from hand-coded value propagation toward a learned
    reachability/quasimetric latent.
    """

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if unreachable_distance is None:
        unreachable_distance = float(memory_size * memory_size)
    free, blocked, _marker = _memory_sets(memory)
    selected_targets = (
        tuple((int(ahead), int(lateral)) for ahead, lateral in target_cells)
        if target_cells is not None
        else select_reachability_target_cells(memory, mode=target_mode)
    )
    selected_targets = tuple(
        cell
        for cell in selected_targets
        if cell_to_row_col(cell, memory_size=memory_size) is not None
        and cell not in blocked
    )
    passable = set(free)
    passable.add((0, 0))
    passable.update(selected_targets)
    passable.difference_update(blocked)
    passable = {
        cell
        for cell in passable
        if cell_to_row_col(cell, memory_size=memory_size) is not None
    }
    current_distances = _bfs_distances(passable, [(0, 0)])
    target_distances = _bfs_distances(passable, selected_targets)
    reachable = _empty_bool_map(memory_size)
    current_distance = torch.full(
        (1, memory_size, memory_size),
        float(unreachable_distance),
        dtype=torch.float32,
    )
    target_distance = torch.full_like(current_distance, float(unreachable_distance))
    target_value = torch.zeros_like(current_distance)
    target_mask = _empty_bool_map(memory_size)
    frontier_mask = _empty_bool_map(memory_size)
    for cell in selected_targets:
        row_col = cell_to_row_col(cell, memory_size=memory_size)
        if row_col is not None:
            target_mask[0, row_col[0], row_col[1]] = True
    for cell in egocentric_frontier_cells(memory):
        row_col = cell_to_row_col(cell, memory_size=memory_size)
        if row_col is not None:
            frontier_mask[0, row_col[0], row_col[1]] = True
    for cell in passable:
        row_col = cell_to_row_col(cell, memory_size=memory_size)
        if row_col is None:
            continue
        row, col = row_col
        current_distance_value = current_distances.get(cell)
        if current_distance_value is not None:
            reachable[0, row, col] = True
            current_distance[0, row, col] = float(current_distance_value)
        target_distance_value = target_distances.get(cell)
        if target_distance_value is not None:
            target_distance[0, row, col] = float(target_distance_value)
            if current_distance_value is not None:
                target_value[0, row, col] = float(gamma ** target_distance_value)
    return Phase3BReachabilityTarget(
        reachable_mask=reachable,
        current_distance=current_distance,
        target_distance=target_distance,
        target_value=target_value,
        target_mask=target_mask,
        frontier_mask=frontier_mask,
        target_cells=tuple(sorted(selected_targets)),
    )


def build_pairwise_reachability_target(
    memory: Mapping,
    *,
    memory_size: int,
    cells: Sequence[Cell] | None = None,
    unreachable_distance: float | None = None,
) -> Phase3BPairwiseReachabilityTarget:
    """Build pairwise shortest-path distances between known passable cells."""

    if unreachable_distance is None:
        unreachable_distance = float(memory_size * memory_size)
    free, blocked, marker = _memory_sets(memory)
    passable = set(free)
    passable.add((0, 0))
    if marker is not None and marker not in blocked:
        passable.add(marker)
    passable.difference_update(blocked)
    passable = {
        cell
        for cell in passable
        if cell_to_row_col(cell, memory_size=memory_size) is not None
    }
    selected = (
        tuple((int(ahead), int(lateral)) for ahead, lateral in cells)
        if cells is not None
        else tuple(sorted(passable))
    )
    selected = tuple(
        cell
        for cell in selected
        if cell in passable
        and cell_to_row_col(cell, memory_size=memory_size) is not None
    )
    distances = torch.full(
        (len(selected), len(selected)),
        float(unreachable_distance),
        dtype=torch.float32,
    )
    reachable = torch.zeros((len(selected), len(selected)), dtype=torch.bool)
    for source_index, source in enumerate(selected):
        source_distances = _bfs_distances(passable, [source])
        for target_index, target in enumerate(selected):
            distance = source_distances.get(target)
            if distance is None:
                continue
            distances[source_index, target_index] = float(distance)
            reachable[source_index, target_index] = True
    return Phase3BPairwiseReachabilityTarget(
        cells=selected,
        distances=distances,
        reachable_mask=reachable,
    )


def stack_reachability_targets(
    targets: Sequence[Phase3BReachabilityTarget],
) -> Phase3BReachabilityTargetBatch:
    """Stack same-sized reachability targets into a batch."""

    if not targets:
        raise ValueError("targets must not be empty")
    return Phase3BReachabilityTargetBatch(
        reachable_mask=torch.stack([target.reachable_mask for target in targets]),
        current_distance=torch.stack([target.current_distance for target in targets]),
        target_distance=torch.stack([target.target_distance for target in targets]),
        target_value=torch.stack([target.target_value for target in targets]),
        target_mask=torch.stack([target.target_mask for target in targets]),
        frontier_mask=torch.stack([target.frontier_mask for target in targets]),
    )


def reachability_prediction_losses(
    predictions: Mapping[str, torch.Tensor],
    targets: Phase3BReachabilityTargetBatch,
    *,
    reachable_weight: float = 1.0,
    current_distance_weight: float = 0.25,
    target_distance_weight: float = 0.25,
    target_value_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return differentiable losses for a Phase 3B reachability head."""

    reachable_logits = predictions["reachable_logits"]
    current_distance_prediction = F.softplus(predictions["current_distance"])
    target_distance_prediction = F.softplus(predictions["target_distance"])
    target_value_logits = predictions["target_value_logits"]
    target_reachable_mask = targets.target_value > 0.0
    reachable_loss = F.binary_cross_entropy_with_logits(
        reachable_logits,
        targets.reachable_mask.to(dtype=reachable_logits.dtype),
    )
    current_distance_loss = _masked_smooth_l1(
        current_distance_prediction,
        targets.current_distance,
        targets.reachable_mask,
    )
    target_distance_loss = _masked_smooth_l1(
        target_distance_prediction,
        targets.target_distance,
        target_reachable_mask,
    )
    target_value_loss = F.binary_cross_entropy_with_logits(
        target_value_logits,
        targets.target_value.to(dtype=target_value_logits.dtype),
    )
    total = (
        float(reachable_weight) * reachable_loss
        + float(current_distance_weight) * current_distance_loss
        + float(target_distance_weight) * target_distance_loss
        + float(target_value_weight) * target_value_loss
    )
    return {
        "loss": total,
        "reachable_loss": reachable_loss,
        "current_distance_loss": current_distance_loss,
        "target_distance_loss": target_distance_loss,
        "target_value_loss": target_value_loss,
    }


def _bfs_distances(passable: set[Cell], sources: Sequence[Cell]) -> dict[Cell, int]:
    queue: deque[Cell] = deque()
    distances: dict[Cell, int] = {}
    for source in sources:
        if source not in passable or source in distances:
            continue
        distances[source] = 0
        queue.append(source)
    while queue:
        cell = queue.popleft()
        base = distances[cell]
        for neighbor in egocentric_neighbors(cell):
            if neighbor in distances or neighbor not in passable:
                continue
            distances[neighbor] = base + 1
            queue.append(neighbor)
    return distances


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    if mask.shape != target.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    selected = mask.to(dtype=torch.bool)
    if int(selected.sum().item()) == 0:
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[selected], target[selected])


def _empty_bool_map(memory_size: int) -> torch.Tensor:
    return torch.zeros((1, memory_size, memory_size), dtype=torch.bool)


def _memory_sets(memory: Mapping) -> tuple[set[Cell], set[Cell], Cell | None]:
    free = {
        (int(cell[0]), int(cell[1]))
        for cell in memory.get("free", ())
    }
    blocked = {
        (int(cell[0]), int(cell[1]))
        for cell in memory.get("blocked", ())
    }
    marker_value = memory.get("marker")
    marker = None
    if marker_value is not None:
        marker = (int(marker_value[0]), int(marker_value[1]))
    return free, blocked, marker
