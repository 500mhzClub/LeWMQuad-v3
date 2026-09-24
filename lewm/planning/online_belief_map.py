"""Deployment-oriented sparse belief map for online navigation.

The map deliberately separates perception evidence, physical contact evidence,
and traversal evidence. Planning queries are conservative: only connected cells
classified as confirmed free are traversable; unknown and conflicted cells are
never admitted as optimistic shortcuts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

Cell = tuple[int, int]
XY = tuple[float, float]
PoseMean = tuple[float, float, float]
Covariance2 = tuple[tuple[float, float], tuple[float, float]]
Covariance3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

__all__ = [
    "FEATURE_CHANNELS",
    "BeliefMapConfig",
    "Cell",
    "CellBelief",
    "CellState",
    "MapFeatureGrid",
    "OnlineBeliefMap",
    "PoseBelief",
    "TargetBelief",
    "four_neighbors",
]


class CellState(str, Enum):
    """Conservative discrete interpretation of a cell belief."""

    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"
    CONFIRMED_FREE = "confirmed_free"
    CONFIRMED_OCCUPIED = "confirmed_occupied"
    CONFLICTED = "conflicted"


@dataclass(frozen=True)
class BeliefMapConfig:
    """Thresholds and scales controlling sparse belief-map fusion."""

    cell_size_m: float = 0.25
    origin_xy_m: XY = (0.0, 0.0)
    planning_connectivity: int = 4
    allow_diagonal_corner_cutting: bool = False
    free_log_odds_threshold: float = -1.0
    occupied_log_odds_threshold: float = 1.0
    log_odds_cap: float = 8.0
    evidence_cap: float = 16.0
    conflict_evidence_threshold: float = 0.75
    conflict_log_odds_margin: float = 0.75
    physical_block_threshold: float = 0.75
    traversal_free_log_odds: float = -4.0
    visit_age_horizon_ticks: int = 2400
    pose_uncertainty_scale_m: float = 1.0
    target_covariance_floor: float = 1e-4

    def __post_init__(self) -> None:
        _validate_vector(self.origin_xy_m, 2, "map origin")
        object.__setattr__(
            self,
            "origin_xy_m",
            (float(self.origin_xy_m[0]), float(self.origin_xy_m[1])),
        )
        if (
            isinstance(self.planning_connectivity, bool)
            or int(self.planning_connectivity) not in (4, 8)
        ):
            raise ValueError("planning_connectivity must be 4 or 8")
        object.__setattr__(
            self,
            "planning_connectivity",
            int(self.planning_connectivity),
        )
        if not isinstance(self.allow_diagonal_corner_cutting, bool):
            raise ValueError("allow_diagonal_corner_cutting must be boolean")
        finite_values = {
            "cell_size_m": self.cell_size_m,
            "free_log_odds_threshold": self.free_log_odds_threshold,
            "occupied_log_odds_threshold": self.occupied_log_odds_threshold,
            "log_odds_cap": self.log_odds_cap,
            "evidence_cap": self.evidence_cap,
            "conflict_evidence_threshold": self.conflict_evidence_threshold,
            "conflict_log_odds_margin": self.conflict_log_odds_margin,
            "physical_block_threshold": self.physical_block_threshold,
            "traversal_free_log_odds": self.traversal_free_log_odds,
            "pose_uncertainty_scale_m": self.pose_uncertainty_scale_m,
            "target_covariance_floor": self.target_covariance_floor,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive and finite")
        if self.free_log_odds_threshold >= 0.0:
            raise ValueError("free_log_odds_threshold must be negative")
        if self.occupied_log_odds_threshold <= 0.0:
            raise ValueError("occupied_log_odds_threshold must be positive")
        if self.log_odds_cap < max(
            abs(self.free_log_odds_threshold),
            abs(self.occupied_log_odds_threshold),
        ):
            raise ValueError("log_odds_cap must cover both occupancy thresholds")
        if self.evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive")
        if self.conflict_evidence_threshold <= 0.0:
            raise ValueError("conflict_evidence_threshold must be positive")
        if self.conflict_log_odds_margin < 0.0:
            raise ValueError("conflict_log_odds_margin must be non-negative")
        if self.physical_block_threshold <= 0.0:
            raise ValueError("physical_block_threshold must be positive")
        if self.traversal_free_log_odds > self.free_log_odds_threshold:
            raise ValueError(
                "traversal_free_log_odds must meet the confirmed-free threshold"
            )
        if (
            isinstance(self.visit_age_horizon_ticks, bool)
            or int(self.visit_age_horizon_ticks) != self.visit_age_horizon_ticks
            or self.visit_age_horizon_ticks <= 0
        ):
            raise ValueError("visit_age_horizon_ticks must be positive")
        if self.pose_uncertainty_scale_m <= 0.0:
            raise ValueError("pose_uncertainty_scale_m must be positive")
        if self.target_covariance_floor <= 0.0:
            raise ValueError("target_covariance_floor must be positive")


@dataclass
class CellBelief:
    """Accumulated evidence and visit history for one map cell."""

    occupancy_log_odds: float = 0.0
    free_evidence: float = 0.0
    occupied_evidence: float = 0.0
    physical_block_evidence: float = 0.0
    last_observed_tick: int | None = None
    last_visited_tick: int | None = None
    last_physical_block_tick: int | None = None
    visit_count: int = 0

    @property
    def observed(self) -> bool:
        return bool(
            self.free_evidence > 0.0
            or self.occupied_evidence > 0.0
            or self.physical_block_evidence > 0.0
            or self.visit_count > 0
        )

    @property
    def conflicting_evidence(self) -> float:
        return min(self.free_evidence, self.occupied_evidence)


@dataclass(frozen=True)
class PoseBelief:
    """Pose mean and covariance in the map's odometry frame."""

    mean: PoseMean
    covariance: Covariance3
    tick: int
    frame: str = "odometry"

    def __post_init__(self) -> None:
        _validate_vector(self.mean, 3, "pose mean")
        _validate_covariance(self.covariance, 3, "pose covariance")
        if self.tick < 0:
            raise ValueError("pose tick must be non-negative")
        if not self.frame:
            raise ValueError("pose frame must be non-empty")


@dataclass(frozen=True)
class TargetBelief:
    """Gaussian position belief and task state for one target."""

    target_id: str
    mean_xy: XY
    covariance: Covariance2
    confidence: float
    last_seen_tick: int
    observation_count: int = 1
    claimed: bool = False

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be non-empty")
        _validate_vector(self.mean_xy, 2, "target mean")
        _validate_covariance(self.covariance, 2, "target covariance")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("target confidence must be in [0, 1]")
        if self.last_seen_tick < 0:
            raise ValueError("target last_seen_tick must be non-negative")
        if self.observation_count <= 0:
            raise ValueError("target observation_count must be positive")


@dataclass(frozen=True)
class MapFeatureGrid:
    """Dense map crop with a stable channel contract.

    Values use ``[channel, row(y), column(x)]`` ordering. ``min_cell`` is the
    world-map cell represented by ``values[:, 0, 0]``.
    """

    values: np.ndarray
    channel_names: tuple[str, ...]
    center_cell: Cell
    min_cell: Cell
    cell_size_m: float

    def channel(self, name: str) -> np.ndarray:
        try:
            index = self.channel_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.values[index]

    def row_col(self, cell: Cell) -> tuple[int, int] | None:
        col = int(cell[0]) - self.min_cell[0]
        row = int(cell[1]) - self.min_cell[1]
        if 0 <= row < self.values.shape[1] and 0 <= col < self.values.shape[2]:
            return row, col
        return None


FEATURE_CHANNELS = (
    "unknown",
    "uncertain",
    "confirmed_free",
    "confirmed_occupied",
    "conflicted",
    "occupancy_probability",
    "visited",
    "visit_age",
    "physical_blocked",
    "frontier",
    "target_belief",
    "pose_position_uncertainty",
)


class OnlineBeliefMap:
    """Sparse, conflict-aware map with conservative planning queries."""

    STATE_SCHEMA = "lewm_online_belief_map"
    STATE_VERSION = 1

    def __init__(self, config: BeliefMapConfig | None = None) -> None:
        self._config = config or BeliefMapConfig()
        self._current_tick = 0
        self._cells: dict[Cell, CellBelief] = {}
        self._targets: dict[str, TargetBelief] = {}
        self._pose: PoseBelief | None = None

    @property
    def config(self) -> BeliefMapConfig:
        return self._config

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def pose(self) -> PoseBelief | None:
        return self._pose

    @property
    def targets(self) -> Mapping[str, TargetBelief]:
        return dict(self._targets)

    @property
    def known_cells(self) -> frozenset[Cell]:
        return frozenset(self._cells)

    def reset(self) -> None:
        self._current_tick = 0
        self._cells.clear()
        self._targets.clear()
        self._pose = None

    def advance_tick(self, steps: int = 1) -> int:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self._current_tick += int(steps)
        return self._current_tick

    def world_to_cell(self, xy: Sequence[float]) -> Cell:
        if len(xy) != 2:
            raise ValueError("xy must contain exactly two values")
        scale = self._config.cell_size_m
        origin = self._config.origin_xy_m
        x = _finite_float(xy[0], "xy")
        y = _finite_float(xy[1], "xy")
        return (
            int(math.floor((x - origin[0]) / scale)),
            int(math.floor((y - origin[1]) / scale)),
        )

    def cell_center(self, cell: Cell) -> XY:
        cell = _normalize_cell(cell)
        scale = self._config.cell_size_m
        origin = self._config.origin_xy_m
        return (
            origin[0] + (cell[0] + 0.5) * scale,
            origin[1] + (cell[1] + 0.5) * scale,
        )

    def cell_belief(self, cell: Cell) -> CellBelief:
        belief = self._cells.get(_normalize_cell(cell))
        return replace(belief) if belief is not None else CellBelief()

    def cell_state(self, cell: Cell) -> CellState:
        belief = self._cells.get(_normalize_cell(cell))
        if belief is None or not belief.observed:
            return CellState.UNKNOWN
        if belief.physical_block_evidence >= self._config.physical_block_threshold:
            return CellState.CONFIRMED_OCCUPIED
        if (
            belief.conflicting_evidence >= self._config.conflict_evidence_threshold
            and abs(belief.occupancy_log_odds)
            <= self._config.conflict_log_odds_margin
        ):
            return CellState.CONFLICTED
        if belief.occupancy_log_odds <= self._config.free_log_odds_threshold:
            return CellState.CONFIRMED_FREE
        if belief.occupancy_log_odds >= self._config.occupied_log_odds_threshold:
            return CellState.CONFIRMED_OCCUPIED
        return CellState.UNCERTAIN

    def is_confirmed_free(self, cell: Cell) -> bool:
        return self.cell_state(cell) is CellState.CONFIRMED_FREE

    def is_confirmed_occupied(self, cell: Cell) -> bool:
        return self.cell_state(cell) is CellState.CONFIRMED_OCCUPIED

    def occupancy_probability(self, cell: Cell) -> float:
        belief = self._cells.get(_normalize_cell(cell))
        if belief is None:
            return 0.5
        value = max(-60.0, min(60.0, belief.occupancy_log_odds))
        return 1.0 / (1.0 + math.exp(-value))

    def visit_age_ticks(
        self,
        cell: Cell,
        *,
        at_tick: int | None = None,
    ) -> int | None:
        """Return ticks since last traversal, or ``None`` if never visited."""

        belief = self._cells.get(_normalize_cell(cell))
        if belief is None or belief.last_visited_tick is None:
            return None
        reference_tick = (
            self._current_tick
            if at_tick is None
            else _nonnegative_int(at_tick, "at_tick")
        )
        if reference_tick < self._current_tick:
            raise ValueError("at_tick cannot precede the map's current tick")
        return reference_tick - belief.last_visited_tick

    def fuse_cell(
        self,
        cell: Cell,
        *,
        free_evidence: float = 0.0,
        occupied_evidence: float = 0.0,
        physical_block_evidence: float = 0.0,
        tick: int | None = None,
    ) -> None:
        """Fuse non-negative evidence into one cell.

        Free and occupied evidence update a bounded signed log-odds value, while
        their separate totals preserve whether an uncertain value was caused by
        no evidence or by contradictory evidence.
        """

        resolved_tick = self._resolve_tick(tick)
        self._fuse_cell_at_tick(
            _normalize_cell(cell),
            free_evidence=_nonnegative_finite(free_evidence, "free_evidence"),
            occupied_evidence=_nonnegative_finite(
                occupied_evidence, "occupied_evidence"
            ),
            physical_block_evidence=_nonnegative_finite(
                physical_block_evidence, "physical_block_evidence"
            ),
            tick=resolved_tick,
        )

    def fuse_free(
        self,
        cells: Iterable[Cell],
        *,
        confidence: float = 1.0,
        tick: int | None = None,
    ) -> None:
        confidence = _nonnegative_finite(confidence, "confidence")
        resolved_tick = self._resolve_tick(tick)
        for cell in dict.fromkeys(_normalize_cell(cell) for cell in cells):
            self._fuse_cell_at_tick(
                cell,
                free_evidence=confidence,
                occupied_evidence=0.0,
                physical_block_evidence=0.0,
                tick=resolved_tick,
            )

    def fuse_occupied(
        self,
        cells: Iterable[Cell],
        *,
        confidence: float = 1.0,
        tick: int | None = None,
    ) -> None:
        confidence = _nonnegative_finite(confidence, "confidence")
        resolved_tick = self._resolve_tick(tick)
        for cell in dict.fromkeys(_normalize_cell(cell) for cell in cells):
            self._fuse_cell_at_tick(
                cell,
                free_evidence=0.0,
                occupied_evidence=confidence,
                physical_block_evidence=0.0,
                tick=resolved_tick,
            )

    def record_physical_blocks(
        self,
        cells: Iterable[Cell],
        *,
        confidence: float = 1.0,
        tick: int | None = None,
    ) -> None:
        confidence = _nonnegative_finite(confidence, "confidence")
        resolved_tick = self._resolve_tick(tick)
        for cell in dict.fromkeys(_normalize_cell(cell) for cell in cells):
            self._fuse_cell_at_tick(
                cell,
                free_evidence=0.0,
                occupied_evidence=0.0,
                physical_block_evidence=confidence,
                tick=resolved_tick,
            )

    def record_traversal(
        self,
        cells: Iterable[Cell],
        *,
        tick: int | None = None,
    ) -> None:
        """Mark traversed cells free and clear stale obstacle evidence."""

        resolved_tick = self._resolve_tick(tick)
        for cell in dict.fromkeys(_normalize_cell(cell) for cell in cells):
            belief = self._cells.setdefault(cell, CellBelief())
            belief.occupancy_log_odds = min(
                belief.occupancy_log_odds,
                self._config.traversal_free_log_odds,
            )
            belief.free_evidence = min(
                self._config.evidence_cap,
                max(
                    belief.free_evidence,
                    abs(self._config.traversal_free_log_odds),
                ),
            )
            belief.occupied_evidence = 0.0
            belief.physical_block_evidence = 0.0
            belief.last_observed_tick = resolved_tick
            belief.last_visited_tick = resolved_tick
            belief.visit_count += 1

    def cells_along_segment(self, start_xy: XY, end_xy: XY) -> tuple[Cell, ...]:
        """Rasterize a segment as a deterministic four-connected cell chain."""

        start = self.world_to_cell(start_xy)
        end = self.world_to_cell(end_xy)
        return _four_connected_grid_line(start, end)

    def fuse_ray(
        self,
        start_xy: XY,
        end_xy: XY,
        *,
        endpoint_occupied: bool,
        free_confidence: float = 1.0,
        occupied_confidence: float = 1.0,
        tick: int | None = None,
    ) -> tuple[Cell, ...]:
        """Fuse a ray without admitting cells beyond its measured endpoint."""

        ray_cells = self.cells_along_segment(start_xy, end_xy)
        resolved_tick = self._resolve_tick(tick)
        free_cells = ray_cells[:-1] if endpoint_occupied else ray_cells
        self.fuse_free(free_cells, confidence=free_confidence, tick=resolved_tick)
        if endpoint_occupied:
            self.fuse_occupied(
                (ray_cells[-1],),
                confidence=occupied_confidence,
                tick=resolved_tick,
            )
        return ray_cells

    def set_pose(
        self,
        mean: Sequence[float],
        covariance: Sequence[Sequence[float]],
        *,
        tick: int | None = None,
        frame: str = "odometry",
    ) -> PoseBelief:
        resolved_tick = self._resolve_tick(tick)
        pose = PoseBelief(
            mean=_vector_tuple(mean, 3, "pose mean"),
            covariance=_covariance_tuple(covariance, 3, "pose covariance"),
            tick=resolved_tick,
            frame=str(frame),
        )
        self._pose = pose
        return pose

    def fuse_target_observation(
        self,
        target_id: str,
        mean_xy: Sequence[float],
        covariance: Sequence[Sequence[float]],
        *,
        confidence: float = 1.0,
        tick: int | None = None,
    ) -> TargetBelief:
        """Fuse an independent Gaussian target-position observation."""

        if not target_id:
            raise ValueError("target_id must be non-empty")
        confidence = _unit_interval(confidence, "confidence")
        resolved_tick = self._resolve_tick(tick)
        measurement_mean = np.asarray(
            _vector_tuple(mean_xy, 2, "target mean"), dtype=np.float64
        )
        measurement_cov = _regularized_covariance(
            covariance,
            size=2,
            floor=self._config.target_covariance_floor,
            name="target covariance",
        )
        prior = self._targets.get(target_id)
        if prior is None:
            belief = TargetBelief(
                target_id=target_id,
                mean_xy=(float(measurement_mean[0]), float(measurement_mean[1])),
                covariance=_matrix_tuple_2(measurement_cov),
                confidence=confidence,
                last_seen_tick=resolved_tick,
            )
        else:
            prior_cov = _regularized_covariance(
                prior.covariance,
                size=2,
                floor=self._config.target_covariance_floor,
                name="prior target covariance",
            )
            prior_mean = np.asarray(prior.mean_xy, dtype=np.float64)
            prior_weight = max(prior.confidence, 1e-6)
            measurement_weight = max(confidence, 1e-6)
            prior_precision = prior_weight * np.linalg.inv(prior_cov)
            measurement_precision = measurement_weight * np.linalg.inv(measurement_cov)
            fused_cov = np.linalg.inv(prior_precision + measurement_precision)
            fused_mean = fused_cov @ (
                prior_precision @ prior_mean
                + measurement_precision @ measurement_mean
            )
            belief = TargetBelief(
                target_id=target_id,
                mean_xy=(float(fused_mean[0]), float(fused_mean[1])),
                covariance=_matrix_tuple_2(fused_cov),
                confidence=float(
                    1.0 - (1.0 - prior.confidence) * (1.0 - confidence)
                ),
                last_seen_tick=resolved_tick,
                observation_count=prior.observation_count + 1,
                claimed=prior.claimed,
            )
        self._targets[target_id] = belief
        return belief

    def mark_target_claimed(self, target_id: str, *, claimed: bool = True) -> None:
        try:
            belief = self._targets[target_id]
        except KeyError as exc:
            raise KeyError(f"unknown target {target_id!r}") from exc
        self._targets[target_id] = replace(belief, claimed=bool(claimed))

    def confirmed_free_cells(self) -> frozenset[Cell]:
        return frozenset(
            cell
            for cell in self._cells
            if self.cell_state(cell) is CellState.CONFIRMED_FREE
        )

    def connected_confirmed_free(self, start_cell: Cell) -> frozenset[Cell]:
        """Return only the confirmed-free component containing ``start_cell``."""

        start = _normalize_cell(start_cell)
        if not self.is_confirmed_free(start):
            return frozenset()
        free = self.confirmed_free_cells()
        reached = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in self._planning_neighbors(cell):
                if neighbor in free and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return frozenset(reached)

    def frontier_cells(self, start_cell: Cell | None = None) -> tuple[Cell, ...]:
        """Return confirmed-free cells bordering unresolved space.

        When ``start_cell`` is supplied, disconnected free islands are excluded.
        Confirmed occupied neighbors do not create frontiers; unknown, uncertain,
        and conflicted neighbors do.
        """

        candidates = (
            self.confirmed_free_cells()
            if start_cell is None
            else self.connected_confirmed_free(start_cell)
        )
        unresolved = {
            CellState.UNKNOWN,
            CellState.UNCERTAIN,
            CellState.CONFLICTED,
        }
        return tuple(
            sorted(
                cell
                for cell in candidates
                if any(
                    self.cell_state(neighbor) in unresolved
                    for neighbor in four_neighbors(cell)
                )
            )
        )

    def shortest_path(
        self,
        start_cell: Cell,
        goal_cell: Cell,
    ) -> tuple[Cell, ...] | None:
        """Find a shortest four-neighbor path through confirmed-free cells only."""

        start = _normalize_cell(start_cell)
        goal = _normalize_cell(goal_cell)
        if not self.is_confirmed_free(start) or not self.is_confirmed_free(goal):
            return None
        parents: dict[Cell, Cell | None] = {start: None}
        queue = deque([start])
        while queue and goal not in parents:
            cell = queue.popleft()
            for neighbor in self._planning_neighbors(cell):
                if neighbor in parents or not self.is_confirmed_free(neighbor):
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)
        if goal not in parents:
            return None
        path: list[Cell] = []
        cursor: Cell | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parents[cursor]
        return tuple(reversed(path))

    def export_features(
        self,
        center_cell: Cell,
        *,
        size: int,
        at_tick: int | None = None,
        target_id: str | None = None,
    ) -> MapFeatureGrid:
        """Export an odd-sized dense crop with values bounded to ``[0, 1]``."""

        if size < 3 or size % 2 == 0:
            raise ValueError("size must be an odd integer >= 3")
        center = _normalize_cell(center_cell)
        reference_tick = self._current_tick if at_tick is None else int(at_tick)
        if reference_tick < self._current_tick:
            raise ValueError("at_tick cannot precede the map's current tick")
        radius = size // 2
        min_cell = (center[0] - radius, center[1] - radius)
        values = np.zeros((len(FEATURE_CHANNELS), size, size), dtype=np.float32)
        channel = {name: index for index, name in enumerate(FEATURE_CHANNELS)}
        frontiers = set(self.frontier_cells(center))
        target_beliefs = self._selected_targets(target_id)
        pose_uncertainty = self._pose_position_uncertainty_feature()
        values[channel["pose_position_uncertainty"], :, :] = pose_uncertainty

        for row in range(size):
            for col in range(size):
                cell = (min_cell[0] + col, min_cell[1] + row)
                state = self.cell_state(cell)
                belief = self._cells.get(cell)
                if state is CellState.UNKNOWN:
                    values[channel["unknown"], row, col] = 1.0
                elif state is CellState.UNCERTAIN:
                    values[channel["uncertain"], row, col] = 1.0
                elif state is CellState.CONFIRMED_FREE:
                    values[channel["confirmed_free"], row, col] = 1.0
                elif state is CellState.CONFIRMED_OCCUPIED:
                    values[channel["confirmed_occupied"], row, col] = 1.0
                else:
                    values[channel["conflicted"], row, col] = 1.0
                values[channel["occupancy_probability"], row, col] = (
                    self.occupancy_probability(cell)
                )
                if belief is not None and belief.last_visited_tick is not None:
                    values[channel["visited"], row, col] = 1.0
                    age = max(0, reference_tick - belief.last_visited_tick)
                    values[channel["visit_age"], row, col] = min(
                        1.0,
                        age / self._config.visit_age_horizon_ticks,
                    )
                if belief is not None:
                    values[channel["physical_blocked"], row, col] = min(
                        1.0,
                        belief.physical_block_evidence
                        / self._config.physical_block_threshold,
                    )
                if cell in frontiers:
                    values[channel["frontier"], row, col] = 1.0
                values[channel["target_belief"], row, col] = (
                    self._target_feature_at_cell(cell, target_beliefs)
                )

        np.clip(values, 0.0, 1.0, out=values)
        return MapFeatureGrid(
            values=values,
            channel_names=FEATURE_CHANNELS,
            center_cell=center,
            min_cell=min_cell,
            cell_size_m=self._config.cell_size_m,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, versioned snapshot."""

        pose_state = None
        if self._pose is not None:
            pose_state = {
                "mean": list(self._pose.mean),
                "covariance": [list(row) for row in self._pose.covariance],
                "tick": self._pose.tick,
                "frame": self._pose.frame,
            }
        cell_states = []
        for cell in sorted(self._cells):
            belief = self._cells[cell]
            cell_states.append({"cell": list(cell), **asdict(belief)})
        target_states = []
        for target_id in sorted(self._targets):
            belief = self._targets[target_id]
            target_states.append(
                {
                    "target_id": belief.target_id,
                    "mean_xy": list(belief.mean_xy),
                    "covariance": [list(row) for row in belief.covariance],
                    "confidence": belief.confidence,
                    "last_seen_tick": belief.last_seen_tick,
                    "observation_count": belief.observation_count,
                    "claimed": belief.claimed,
                }
            )
        return {
            "schema": self.STATE_SCHEMA,
            "version": self.STATE_VERSION,
            "config": asdict(self._config),
            "current_tick": self._current_tick,
            "pose": pose_state,
            "cells": cell_states,
            "targets": target_states,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> OnlineBeliefMap:
        if state.get("schema") != cls.STATE_SCHEMA:
            raise ValueError(f"unsupported belief-map schema: {state.get('schema')!r}")
        if state.get("version") != cls.STATE_VERSION:
            raise ValueError(
                f"unsupported belief-map version: {state.get('version')!r}"
            )
        config_state = state.get("config")
        if not isinstance(config_state, Mapping):
            raise ValueError("belief-map state is missing config")
        result = cls(BeliefMapConfig(**dict(config_state)))
        result._current_tick = _nonnegative_int(
            state.get("current_tick"), "current_tick"
        )

        pose_state = state.get("pose")
        if pose_state is not None:
            if not isinstance(pose_state, Mapping):
                raise ValueError("pose state must be a mapping or null")
            pose = PoseBelief(
                mean=_vector_tuple(pose_state.get("mean"), 3, "pose mean"),
                covariance=_covariance_tuple(
                    pose_state.get("covariance"), 3, "pose covariance"
                ),
                tick=_nonnegative_int(pose_state.get("tick"), "pose tick"),
                frame=str(pose_state.get("frame", "")),
            )
            if pose.tick > result._current_tick:
                raise ValueError("pose tick cannot exceed current_tick")
            result._pose = pose

        cells_state = state.get("cells")
        if not isinstance(cells_state, list):
            raise ValueError("cells state must be a list")
        for item in cells_state:
            if not isinstance(item, Mapping):
                raise ValueError("each cell state must be a mapping")
            cell_value = item.get("cell")
            if not isinstance(cell_value, Sequence) or len(cell_value) != 2:
                raise ValueError("cell coordinates must contain two integers")
            cell = _normalize_cell((cell_value[0], cell_value[1]))
            if cell in result._cells:
                raise ValueError(f"duplicate cell state for {cell}")
            belief = CellBelief(
                occupancy_log_odds=_finite_float(
                    item.get("occupancy_log_odds"), "occupancy_log_odds"
                ),
                free_evidence=_nonnegative_finite(
                    item.get("free_evidence"), "free_evidence"
                ),
                occupied_evidence=_nonnegative_finite(
                    item.get("occupied_evidence"), "occupied_evidence"
                ),
                physical_block_evidence=_nonnegative_finite(
                    item.get("physical_block_evidence"), "physical_block_evidence"
                ),
                last_observed_tick=_optional_nonnegative_int(
                    item.get("last_observed_tick"), "last_observed_tick"
                ),
                last_visited_tick=_optional_nonnegative_int(
                    item.get("last_visited_tick"), "last_visited_tick"
                ),
                last_physical_block_tick=_optional_nonnegative_int(
                    item.get("last_physical_block_tick"),
                    "last_physical_block_tick",
                ),
                visit_count=_nonnegative_int(item.get("visit_count"), "visit_count"),
            )
            if belief.occupancy_log_odds < -result.config.log_odds_cap or (
                belief.occupancy_log_odds > result.config.log_odds_cap
            ):
                raise ValueError("serialized occupancy_log_odds exceeds configured cap")
            if any(
                evidence > result.config.evidence_cap
                for evidence in (
                    belief.free_evidence,
                    belief.occupied_evidence,
                    belief.physical_block_evidence,
                )
            ):
                raise ValueError("serialized evidence exceeds configured cap")
            for tick_name in (
                "last_observed_tick",
                "last_visited_tick",
                "last_physical_block_tick",
            ):
                value = getattr(belief, tick_name)
                if value is not None and value > result._current_tick:
                    raise ValueError(f"{tick_name} cannot exceed current_tick")
            result._cells[cell] = belief

        targets_state = state.get("targets")
        if not isinstance(targets_state, list):
            raise ValueError("targets state must be a list")
        for item in targets_state:
            if not isinstance(item, Mapping):
                raise ValueError("each target state must be a mapping")
            target = TargetBelief(
                target_id=str(item.get("target_id", "")),
                mean_xy=_vector_tuple(item.get("mean_xy"), 2, "target mean"),
                covariance=_covariance_tuple(
                    item.get("covariance"), 2, "target covariance"
                ),
                confidence=_unit_interval(item.get("confidence"), "confidence"),
                last_seen_tick=_nonnegative_int(
                    item.get("last_seen_tick"), "last_seen_tick"
                ),
                observation_count=_nonnegative_int(
                    item.get("observation_count"), "observation_count"
                ),
                claimed=bool(item.get("claimed")),
            )
            if target.target_id in result._targets:
                raise ValueError(f"duplicate target state for {target.target_id!r}")
            if target.last_seen_tick > result._current_tick:
                raise ValueError("target last_seen_tick cannot exceed current_tick")
            result._targets[target.target_id] = target
        return result

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = type(self).from_state_dict(state)
        self._config = restored._config
        self._current_tick = restored._current_tick
        self._cells = restored._cells
        self._targets = restored._targets
        self._pose = restored._pose

    def _resolve_tick(self, tick: int | None) -> int:
        resolved = (
            self._current_tick
            if tick is None
            else _nonnegative_int(tick, "tick")
        )
        if resolved < self._current_tick:
            raise ValueError("updates cannot move backward in time")
        self._current_tick = resolved
        return resolved

    def _fuse_cell_at_tick(
        self,
        cell: Cell,
        *,
        free_evidence: float,
        occupied_evidence: float,
        physical_block_evidence: float,
        tick: int,
    ) -> None:
        if free_evidence == occupied_evidence == physical_block_evidence == 0.0:
            return
        belief = self._cells.setdefault(cell, CellBelief())
        delta = occupied_evidence + physical_block_evidence - free_evidence
        belief.occupancy_log_odds = max(
            -self._config.log_odds_cap,
            min(self._config.log_odds_cap, belief.occupancy_log_odds + delta),
        )
        belief.free_evidence = min(
            self._config.evidence_cap,
            belief.free_evidence + free_evidence,
        )
        belief.occupied_evidence = min(
            self._config.evidence_cap,
            belief.occupied_evidence + occupied_evidence,
        )
        belief.physical_block_evidence = min(
            self._config.evidence_cap,
            belief.physical_block_evidence + physical_block_evidence,
        )
        belief.last_observed_tick = tick
        if physical_block_evidence > 0.0:
            belief.last_physical_block_tick = tick

    def _selected_targets(self, target_id: str | None) -> tuple[TargetBelief, ...]:
        if target_id is not None:
            target = self._targets.get(target_id)
            return () if target is None else (target,)
        return tuple(target for target in self._targets.values() if not target.claimed)

    def _planning_neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        cardinal = four_neighbors(cell)
        if self._config.planning_connectivity == 4:
            return cardinal
        x, y = _normalize_cell(cell)
        neighbors: list[Cell] = list(cardinal)
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            diagonal = (x + dx, y + dy)
            if not self.is_confirmed_free(diagonal):
                continue
            if not self._config.allow_diagonal_corner_cutting and (
                not self.is_confirmed_free((x + dx, y))
                or not self.is_confirmed_free((x, y + dy))
            ):
                continue
            neighbors.append(diagonal)
        return tuple(neighbors)

    def _target_feature_at_cell(
        self,
        cell: Cell,
        targets: Sequence[TargetBelief],
    ) -> float:
        if not targets:
            return 0.0
        xy = np.asarray(self.cell_center(cell), dtype=np.float64)
        value = 0.0
        for target in targets:
            covariance = _regularized_covariance(
                target.covariance,
                size=2,
                floor=self._config.target_covariance_floor,
                name="target covariance",
            )
            delta = xy - np.asarray(target.mean_xy, dtype=np.float64)
            exponent = -0.5 * float(delta @ np.linalg.solve(covariance, delta))
            value = max(value, target.confidence * math.exp(max(-60.0, exponent)))
        return min(1.0, value)

    def _pose_position_uncertainty_feature(self) -> float:
        if self._pose is None:
            return 0.0
        covariance = np.asarray(self._pose.covariance, dtype=np.float64)
        position_std = math.sqrt(max(0.0, float(covariance[0, 0] + covariance[1, 1])))
        return min(1.0, position_std / self._config.pose_uncertainty_scale_m)


def four_neighbors(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    x, y = _normalize_cell(cell)
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _four_connected_grid_line(start: Cell, end: Cell) -> tuple[Cell, ...]:
    """Grid traversal that changes exactly one coordinate per step."""

    x, y = start
    end_x, end_y = end
    nx = abs(end_x - x)
    ny = abs(end_y - y)
    sign_x = 0 if end_x == x else (1 if end_x > x else -1)
    sign_y = 0 if end_y == y else (1 if end_y > y else -1)
    ix = 0
    iy = 0
    result = [(x, y)]
    while ix < nx or iy < ny:
        if ix == nx:
            y += sign_y
            iy += 1
        elif iy == ny:
            x += sign_x
            ix += 1
        else:
            x_crossing = (1 + 2 * ix) * ny
            y_crossing = (1 + 2 * iy) * nx
            if x_crossing <= y_crossing:
                x += sign_x
                ix += 1
            else:
                y += sign_y
                iy += 1
        result.append((x, y))
    return tuple(result)


def _normalize_cell(cell: Sequence[int]) -> Cell:
    if len(cell) != 2:
        raise ValueError("cell must contain exactly two integers")
    values: list[int] = []
    for value in cell:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError("cell coordinates must be integers")
        values.append(int(value))
    return values[0], values[1]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative_int(value, name)


def _validate_vector(values: Sequence[float], size: int, name: str) -> None:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector of length {size}") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")


def _vector_tuple(values: Any, size: int, name: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    _validate_vector(values, size, name)
    array = np.asarray(values, dtype=np.float64)
    return tuple(float(value) for value in array)


def _validate_covariance(
    covariance: Sequence[Sequence[float]],
    size: int,
    name: str,
) -> None:
    try:
        array = np.asarray(covariance, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {size}x{size} matrix") from exc
    if array.shape != (size, size) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    if not np.allclose(array, array.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(array).min()) < -1e-8:
        raise ValueError(f"{name} must be positive semidefinite")


def _covariance_tuple(
    covariance: Any,
    size: int,
    name: str,
) -> tuple:
    _validate_covariance(covariance, size, name)
    array = np.asarray(covariance, dtype=np.float64)
    return tuple(tuple(float(value) for value in row) for row in array)


def _regularized_covariance(
    covariance: Any,
    *,
    size: int,
    floor: float,
    name: str,
) -> np.ndarray:
    _validate_covariance(covariance, size, name)
    array = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    eigenvalues = np.maximum(eigenvalues, float(floor))
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _matrix_tuple_2(array: np.ndarray) -> Covariance2:
    return (
        (float(array[0, 0]), float(array[0, 1])),
        (float(array[1, 0]), float(array[1, 1])),
    )
