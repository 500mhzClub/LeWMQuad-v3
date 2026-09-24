"""Privileged exact-occupancy backend for shared-map positive controls.

This module is development-only infrastructure. It rasterizes a scene manifest
under the canonical geometry contract, loads that evidence into
``OnlineBeliefMap``, and audits the public conservative planning APIs against an
independent grid reference. It is not an inference adapter and must never be
used by a learned or sealed evaluation path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np

from lewm.planning.geometry_contract import GeometryContract
from lewm.planning.online_belief_map import (
    BeliefMapConfig,
    Cell,
    OnlineBeliefMap,
    four_neighbors,
)
from lewm_worlds.manifest import BoxObject, SceneManifest
from lewm_worlds.planning_grid import InflatedOccupancyGrid


@dataclass(frozen=True)
class ExactOccupancyAgreement:
    """Agreement between shared-map APIs and independent occupancy grids."""

    scene_id: str
    map_cell_size_m: float
    oracle_cell_size_m: float
    map_origin_xy_m: tuple[float, float]
    map_shape: tuple[int, int]
    confirmed_free_cells: int
    confirmed_occupied_cells: int
    map_component_cells: int
    online_reference_component_cells: int
    component_symmetric_difference_cells: int
    map_frontier_cells: int
    online_reference_frontier_cells: int
    frontier_symmetric_difference_cells: int
    projected_oracle_component_cells: int
    map_only_resolution_cells: int
    projected_oracle_only_cells: int
    resolution_symmetric_difference_cells: int
    resolution_jaccard: float

    @property
    def online_topology_agrees(self) -> bool:
        return bool(
            self.component_symmetric_difference_cells == 0
            and self.frontier_symmetric_difference_cells == 0
        )

    @property
    def oracle_projection_agrees(self) -> bool:
        return self.resolution_symmetric_difference_cells == 0

    @property
    def resolution_is_conservative(self) -> bool:
        return self.map_only_resolution_cells == 0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "online_topology_agrees": self.online_topology_agrees,
            "oracle_projection_agrees": self.oracle_projection_agrees,
            "resolution_is_conservative": self.resolution_is_conservative,
        }


@dataclass(frozen=True)
class ConnectedClaimRoute:
    """One true claim endpoint routed through confirmed-free map cells."""

    object_id: str
    endpoint_cell: Cell
    endpoint_xy_m: tuple[float, float]
    target_xy_m: tuple[float, float]
    target_distance_m: float
    path_cells: int
    oracle_endpoint_connected: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["endpoint_cell"] = list(self.endpoint_cell)
        payload["endpoint_xy_m"] = list(self.endpoint_xy_m)
        payload["target_xy_m"] = list(self.target_xy_m)
        return payload


class ExactOccupancyBeliefAdapter:
    """Load canonical exact occupancy into a scene-aligned shared belief map."""

    def __init__(
        self,
        manifest: SceneManifest,
        geometry_contract: GeometryContract,
    ) -> None:
        self.manifest = manifest
        self.geometry_contract = geometry_contract
        config_space = geometry_contract.configuration_space
        if config_space.connectivity not in (4, 8):
            raise ValueError("geometry contract connectivity must be 4 or 8")
        self.oracle_grid = InflatedOccupancyGrid(
            manifest,
            cell_size_m=float(config_space.oracle_cell_size_m),
            inflation_m=float(config_space.body_inflation_radius_m),
            treat_landmarks_as_obstacles=bool(
                config_space.landmarks_are_obstacles
            ),
            treat_distractors_as_obstacles=bool(
                config_space.distractors_are_obstacles
            ),
        )
        self.online_grid = InflatedOccupancyGrid(
            manifest,
            cell_size_m=float(config_space.online_cell_size_m),
            inflation_m=float(config_space.body_inflation_radius_m),
            treat_landmarks_as_obstacles=bool(
                config_space.landmarks_are_obstacles
            ),
            treat_distractors_as_obstacles=bool(
                config_space.distractors_are_obstacles
            ),
        )
        self.visibility_grid = InflatedOccupancyGrid(
            manifest,
            cell_size_m=float(config_space.oracle_cell_size_m),
            inflation_m=0.0,
            treat_landmarks_as_obstacles=False,
            treat_distractors_as_obstacles=bool(
                config_space.distractors_are_obstacles
            ),
        )
        self.online_traversable = _in_world_traversable(
            self.online_grid,
            manifest,
        )
        self.oracle_traversable = _in_world_traversable(
            self.oracle_grid,
            manifest,
        )
        self.belief_map = OnlineBeliefMap(
            BeliefMapConfig(
                cell_size_m=float(config_space.online_cell_size_m),
                origin_xy_m=(
                    float(self.online_grid.origin_xy[0]),
                    float(self.online_grid.origin_xy[1]),
                ),
                planning_connectivity=int(config_space.connectivity),
                allow_diagonal_corner_cutting=bool(
                    config_space.allow_diagonal_corner_cutting
                ),
            )
        )
        self._observed_cells: frozenset[Cell] = frozenset()
        self._oracle_component = self._component(
            self.oracle_traversable,
            self.oracle_grid.to_grid(self.spawn_xy_m),
        )

    @property
    def spawn_xy_m(self) -> tuple[float, float]:
        return (
            float(self.manifest.spawn.xyz_m[0]),
            float(self.manifest.spawn.xyz_m[1]),
        )

    @property
    def spawn_cell(self) -> Cell:
        return self.belief_map.world_to_cell(self.spawn_xy_m)

    @property
    def all_online_cells(self) -> frozenset[Cell]:
        nx, ny = self.online_grid.shape
        return frozenset((x, y) for x in range(nx) for y in range(ny))

    def load(
        self,
        observed_cells: Iterable[Cell] | None = None,
        *,
        tick: int = 0,
    ) -> ExactOccupancyAgreement:
        """Load exact evidence, optionally leaving other cells unresolved."""

        self.belief_map.reset()
        selected = (
            self.all_online_cells
            if observed_cells is None
            else frozenset((int(cell[0]), int(cell[1])) for cell in observed_cells)
        )
        invalid = selected - self.all_online_cells
        if invalid:
            raise ValueError(f"observed cells outside the online grid: {min(invalid)}")
        self._observed_cells = selected
        free = tuple(sorted(cell for cell in selected if self.online_traversable[cell]))
        occupied = tuple(sorted(selected - set(free)))
        evidence = max(
            2.0,
            abs(float(self.belief_map.config.free_log_odds_threshold)),
            float(self.belief_map.config.occupied_log_odds_threshold),
        )
        self.belief_map.fuse_free(free, confidence=evidence, tick=tick)
        self.belief_map.fuse_occupied(occupied, confidence=evidence, tick=tick)
        return self.agreement()

    def agreement(self) -> ExactOccupancyAgreement:
        """Compare public shared-map results with independent references."""

        reference_mask = np.zeros(self.online_grid.shape, dtype=np.bool_)
        for cell in self._observed_cells:
            if self.online_traversable[cell]:
                reference_mask[cell] = True
        reference_component = self._component(reference_mask, self.spawn_cell)
        map_component = set(
            self.belief_map.connected_confirmed_free(self.spawn_cell)
        )
        reference_frontiers = _reference_frontiers(
            reference_component,
            self._observed_cells,
        )
        map_frontiers = set(self.belief_map.frontier_cells(self.spawn_cell))
        projected_oracle = {
            self.belief_map.world_to_cell(self.oracle_grid.to_world(cell))
            for cell in self._oracle_component
        }
        projected_oracle &= self.all_online_cells
        union = map_component | projected_oracle
        intersection = map_component & projected_oracle
        return ExactOccupancyAgreement(
            scene_id=str(self.manifest.scene_id),
            map_cell_size_m=float(self.belief_map.config.cell_size_m),
            oracle_cell_size_m=float(self.oracle_grid.cell_size_m),
            map_origin_xy_m=tuple(self.belief_map.config.origin_xy_m),
            map_shape=tuple(int(value) for value in self.online_grid.shape),
            confirmed_free_cells=len(self.belief_map.confirmed_free_cells()),
            confirmed_occupied_cells=(
                len(self.belief_map.known_cells)
                - len(self.belief_map.confirmed_free_cells())
            ),
            map_component_cells=len(map_component),
            online_reference_component_cells=len(reference_component),
            component_symmetric_difference_cells=len(
                map_component ^ reference_component
            ),
            map_frontier_cells=len(map_frontiers),
            online_reference_frontier_cells=len(reference_frontiers),
            frontier_symmetric_difference_cells=len(
                map_frontiers ^ reference_frontiers
            ),
            projected_oracle_component_cells=len(projected_oracle),
            map_only_resolution_cells=len(map_component - projected_oracle),
            projected_oracle_only_cells=len(projected_oracle - map_component),
            resolution_symmetric_difference_cells=len(
                map_component ^ projected_oracle
            ),
            resolution_jaccard=(
                1.0 if not union else float(len(intersection) / len(union))
            ),
        )

    def connected_claim_route(
        self,
        landmark: BoxObject,
    ) -> ConnectedClaimRoute | None:
        """Route to a true claim endpoint using only ``shortest_path``."""

        visibility = self.geometry_contract.visibility_and_claim
        target_xy = (
            float(landmark.center_xyz_m[0]),
            float(landmark.center_xyz_m[1]),
        )
        candidates: list[tuple[float, Cell, tuple[float, float], float]] = []
        for cell in self.belief_map.connected_confirmed_free(self.spawn_cell):
            xy = self.belief_map.cell_center(cell)
            distance = float(math.dist(xy, target_xy))
            if distance > float(visibility.claim_radius_m):
                continue
            if (
                visibility.require_line_of_sight_for_scene_validity
                and not self.visibility_grid.has_free_line(
                    xy,
                    target_xy,
                )
            ):
                continue
            oracle_connected = (
                self.oracle_grid.to_grid(xy) in self._oracle_component
            )
            if not oracle_connected:
                continue
            candidates.append(
                (
                    abs(distance - float(visibility.standoff_m)),
                    cell,
                    xy,
                    distance,
                )
            )
        for _standoff_error, cell, xy, distance in sorted(candidates):
            path = self.belief_map.shortest_path(self.spawn_cell, cell)
            if path is None or path[0] != self.spawn_cell or path[-1] != cell:
                continue
            return ConnectedClaimRoute(
                object_id=str(landmark.object_id),
                endpoint_cell=cell,
                endpoint_xy_m=(float(xy[0]), float(xy[1])),
                target_xy_m=target_xy,
                target_distance_m=distance,
                path_cells=len(path),
                oracle_endpoint_connected=True,
            )
        return None

    def _component(self, mask: np.ndarray, start: Cell) -> set[Cell]:
        return _reachable_component(
            mask,
            start,
            connectivity=int(
                self.geometry_contract.configuration_space.connectivity
            ),
            allow_diagonal_corner_cutting=bool(
                self.geometry_contract.configuration_space.allow_diagonal_corner_cutting
            ),
        )


def _in_world_traversable(
    grid: InflatedOccupancyGrid,
    manifest: SceneManifest,
) -> np.ndarray:
    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    nx, ny = grid.shape
    xs = grid.origin_xy[0] + (
        np.arange(nx, dtype=np.float64) + 0.5
    ) * grid.cell_size_m
    ys = grid.origin_xy[1] + (
        np.arange(ny, dtype=np.float64) + 0.5
    ) * grid.cell_size_m
    in_world = (
        (xs[:, None] >= float(x_lo))
        & (xs[:, None] <= float(x_hi))
        & (ys[None, :] >= float(y_lo))
        & (ys[None, :] <= float(y_hi))
    )
    return np.asarray(grid.free_mask & in_world, dtype=np.bool_)


def _reachable_component(
    free: np.ndarray,
    start: Cell,
    *,
    connectivity: int,
    allow_diagonal_corner_cutting: bool,
) -> set[Cell]:
    nx, ny = int(free.shape[0]), int(free.shape[1])
    if not (0 <= start[0] < nx and 0 <= start[1] < ny and free[start]):
        return set()
    reached = {start}
    queue: deque[Cell] = deque((start,))
    while queue:
        x, y = queue.popleft()
        for neighbor in four_neighbors((x, y)):
            if (
                0 <= neighbor[0] < nx
                and 0 <= neighbor[1] < ny
                and free[neighbor]
                and neighbor not in reached
            ):
                reached.add(neighbor)
                queue.append(neighbor)
        if connectivity != 8:
            continue
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (x + dx, y + dy)
            if not (
                0 <= neighbor[0] < nx
                and 0 <= neighbor[1] < ny
                and free[neighbor]
                and neighbor not in reached
            ):
                continue
            if not allow_diagonal_corner_cutting and (
                not free[x + dx, y] or not free[x, y + dy]
            ):
                continue
            reached.add(neighbor)
            queue.append(neighbor)
    return reached


def _reference_frontiers(
    component: set[Cell],
    observed_cells: frozenset[Cell],
) -> set[Cell]:
    return {
        cell
        for cell in component
        if any(neighbor not in observed_cells for neighbor in four_neighbors(cell))
    }


__all__ = [
    "ConnectedClaimRoute",
    "ExactOccupancyAgreement",
    "ExactOccupancyBeliefAdapter",
]
