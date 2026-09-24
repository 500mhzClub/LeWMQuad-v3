"""Fixed-spawn geometry audit for held-out navigation benchmarks.

This audit is intentionally stricter than the corpus-generation reachability
gate.  It evaluates the exact pose used by a benchmark, never substitutes a
different spawn candidate, and plans in body-inflated configuration space.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from lewm_worlds.manifest import SceneManifest
from lewm_worlds.planning_grid import (
    InflatedOccupancyGrid,
    corridor_width_m,
    safe_standoff_xys,
)
from lewm_worlds.scene_graph import SceneGraph


@dataclass(frozen=True)
class FixedSpawnAuditConfig:
    """Immutable geometry contract for one benchmark scene audit.

    The defaults mirror ``go2_generalization_geometry_v1`` without importing
    the higher-level :mod:`lewm` package into :mod:`lewm_worlds`.
    """

    cell_size_m: float = 0.05
    coverage_cell_size_m: float = 0.10
    body_radius_m: float = 0.20
    claim_radius_m: float = 1.20
    standoff_m: float = 1.05
    standoff_candidates: int = 32
    minimum_navigable_corridor_width_m: float = 0.50
    minimum_navigable_standoffs_per_beacon: int = 1
    require_line_of_sight: bool = True
    connectivity: int = 8
    allow_diagonal_corner_cutting: bool = False
    treat_landmarks_as_obstacles: bool = True
    treat_distractors_as_obstacles: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        if (
            not math.isfinite(self.coverage_cell_size_m)
            or self.coverage_cell_size_m <= 0.0
        ):
            raise ValueError("coverage_cell_size_m must be positive")
        if not math.isfinite(self.body_radius_m) or self.body_radius_m < 0.0:
            raise ValueError("body_radius_m must be non-negative")
        if not math.isfinite(self.claim_radius_m) or self.claim_radius_m < 0.0:
            raise ValueError("claim_radius_m must be non-negative")
        if not math.isfinite(self.standoff_m) or self.standoff_m <= 0.0:
            raise ValueError("standoff_m must be positive")
        if self.standoff_m > self.claim_radius_m:
            raise ValueError("standoff_m must not exceed claim_radius_m")
        if self.standoff_candidates < 4:
            raise ValueError("standoff_candidates must be at least 4")
        if (
            not math.isfinite(self.minimum_navigable_corridor_width_m)
            or self.minimum_navigable_corridor_width_m <= 0.0
        ):
            raise ValueError("minimum_navigable_corridor_width_m must be positive")
        if self.minimum_navigable_standoffs_per_beacon < 1:
            raise ValueError(
                "minimum_navigable_standoffs_per_beacon must be at least 1"
            )
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass(frozen=True)
class BeaconFixedSpawnReachability:
    """Reachability of one beacon from the benchmark's fixed spawn."""

    object_id: str
    target_xy_m: tuple[float, float]
    reachable: bool
    claim_reachable: bool
    preferred_standoff_reachable: bool
    reachable_claim_cell_count: int
    reachable_standoff_count: int
    reachable_navigable_standoff_count: int
    closest_reachable_distance_m: float | None


@dataclass(frozen=True)
class FixedSpawnAuditReport:
    """Serializable audit result plus cells needed by coverage metrics."""

    scene_id: str
    family: str
    source_split: str | None
    config: FixedSpawnAuditConfig
    spawn_xy_m: tuple[float, float]
    spawn_grid_cell: tuple[int, int]
    spawn_obstacle_clearance_m: float
    spawn_is_body_clear: bool
    reachable_cell_count: int
    reachable_area_m2: float
    beacons: tuple[BeaconFixedSpawnReachability, ...]
    fully_reachable: bool
    all_beacons_have_preferred_standoff: bool
    failure_reason: str
    grid_origin_xy_m: tuple[float, float]
    grid_shape: tuple[int, int]
    reachable_cells: frozenset[tuple[int, int]] = field(
        repr=False,
        compare=False,
    )
    coverage_grid_origin_xy_m: tuple[float, float]
    coverage_grid_shape: tuple[int, int]
    coverage_reachable_cell_count: int
    coverage_reachable_area_m2: float
    coverage_reachable_cells: frozenset[tuple[int, int]] = field(
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, compact representation used in manifests."""

        return {
            "scene_id": self.scene_id,
            "family": self.family,
            "source_split": self.source_split,
            "config": asdict(self.config),
            "spawn_xy_m": list(self.spawn_xy_m),
            "spawn_grid_cell": list(self.spawn_grid_cell),
            "spawn_obstacle_clearance_m": self.spawn_obstacle_clearance_m,
            "spawn_is_body_clear": self.spawn_is_body_clear,
            "reachable_cell_count": self.reachable_cell_count,
            "reachable_area_m2": self.reachable_area_m2,
            "beacons": [asdict(beacon) for beacon in self.beacons],
            "fully_reachable": self.fully_reachable,
            "all_beacons_have_preferred_standoff": (
                self.all_beacons_have_preferred_standoff
            ),
            "failure_reason": self.failure_reason,
            "grid_origin_xy_m": list(self.grid_origin_xy_m),
            "grid_shape": list(self.grid_shape),
            "coverage_grid_origin_xy_m": list(self.coverage_grid_origin_xy_m),
            "coverage_grid_shape": list(self.coverage_grid_shape),
            "coverage_reachable_cell_count": self.coverage_reachable_cell_count,
            "coverage_reachable_area_m2": self.coverage_reachable_area_m2,
        }

    def world_to_grid(self, xy_m: tuple[float, float]) -> tuple[int, int]:
        """Map a world-frame point to this audit's oracle cell."""

        return (
            int(
                math.floor(
                    (float(xy_m[0]) - self.grid_origin_xy_m[0])
                    / self.config.cell_size_m
                )
            ),
            int(
                math.floor(
                    (float(xy_m[1]) - self.grid_origin_xy_m[1])
                    / self.config.cell_size_m
                )
            ),
        )

    def world_to_coverage_grid(self, xy_m: tuple[float, float]) -> tuple[int, int]:
        """Map a world-frame point to the normalized-coverage cell."""

        return (
            int(
                math.floor(
                    (float(xy_m[0]) - self.coverage_grid_origin_xy_m[0])
                    / self.config.coverage_cell_size_m
                )
            ),
            int(
                math.floor(
                    (float(xy_m[1]) - self.coverage_grid_origin_xy_m[1])
                    / self.config.coverage_cell_size_m
                )
            ),
        )


def audit_fixed_spawn(
    manifest: SceneManifest,
    *,
    config: FixedSpawnAuditConfig = FixedSpawnAuditConfig(),
) -> FixedSpawnAuditReport:
    """Audit every beacon from ``manifest.spawn`` under ``config``.

    Connectivity follows the occupancy planner's eight-neighbour convention;
    diagonal steps may not pass between two occupied cardinal neighbours.
    The exact spawn coordinate must itself have sufficient continuous
    clearance and its containing raster cell must be free.  No snapping is
    permitted because that would change the evaluated task.
    """

    config.validate()
    grid = InflatedOccupancyGrid(
        manifest,
        cell_size_m=float(config.cell_size_m),
        inflation_m=float(config.body_radius_m),
        treat_landmarks_as_obstacles=bool(config.treat_landmarks_as_obstacles),
        treat_distractors_as_obstacles=bool(config.treat_distractors_as_obstacles),
    )
    scene = SceneGraph(manifest)
    spawn_xy = (float(manifest.spawn.xyz_m[0]), float(manifest.spawn.xyz_m[1]))
    spawn_cell = grid.to_grid(spawn_xy)
    spawn_clearance = float(grid.obstacle_clearance_m(spawn_xy))

    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    spawn_in_bounds = (
        float(x_lo) <= spawn_xy[0] <= float(x_hi)
        and float(y_lo) <= spawn_xy[1] <= float(y_hi)
    )
    spawn_body_clear = bool(
        spawn_in_bounds
        and spawn_clearance >= float(config.body_radius_m)
    )

    nx, ny = grid.shape
    xs = grid.origin_xy[0] + (np.arange(nx, dtype=np.float64) + 0.5) * grid.cell_size_m
    ys = grid.origin_xy[1] + (np.arange(ny, dtype=np.float64) + 0.5) * grid.cell_size_m
    in_world = (
        (xs[:, None] >= float(x_lo))
        & (xs[:, None] <= float(x_hi))
        & (ys[None, :] >= float(y_lo))
        & (ys[None, :] <= float(y_hi))
    )
    traversable = np.asarray(grid.free_mask & in_world, dtype=bool)
    spawn_cell_free = bool(
        0 <= spawn_cell[0] < nx
        and 0 <= spawn_cell[1] < ny
        and traversable[spawn_cell]
    )
    reachable = (
        _reachable_component(
            traversable,
            spawn_cell,
            connectivity=int(config.connectivity),
            allow_diagonal_corner_cutting=bool(
                config.allow_diagonal_corner_cutting
            ),
        )
        if spawn_body_clear and spawn_cell_free
        else frozenset()
    )

    reachable_world = tuple(grid.to_world(cell) for cell in sorted(reachable))
    coverage_origin = (float(x_lo), float(y_lo))
    coverage_shape = (
        int(
            math.ceil(
                (float(x_hi) - float(x_lo)) / config.coverage_cell_size_m
            )
        ),
        int(
            math.ceil(
                (float(y_hi) - float(y_lo)) / config.coverage_cell_size_m
            )
        ),
    )

    def coverage_cell(xy_m: tuple[float, float]) -> tuple[int, int]:
        return (
            int(
                math.floor(
                    (xy_m[0] - coverage_origin[0])
                    / config.coverage_cell_size_m
                )
            ),
            int(
                math.floor(
                    (xy_m[1] - coverage_origin[1])
                    / config.coverage_cell_size_m
                )
            ),
        )

    coverage_reachable = frozenset(coverage_cell(xy_m) for xy_m in reachable_world)
    beacon_reports: list[BeaconFixedSpawnReachability] = []
    for landmark in manifest.landmarks:
        target_xy = (
            float(landmark.center_xyz_m[0]),
            float(landmark.center_xyz_m[1]),
        )
        closest = min(
            (math.dist(cell_xy, target_xy) for cell_xy in reachable_world),
            default=None,
        )
        claim_cells = 0
        for cell_xy in reachable_world:
            if math.dist(cell_xy, target_xy) > float(config.claim_radius_m):
                continue
            if config.require_line_of_sight and not scene.has_line_of_sight(
                cell_xy,
                target_xy,
                exclude_landmark_xy=target_xy,
            ):
                continue
            claim_cells += 1
        reachable_standoffs = 0
        navigable_standoffs = 0
        for standoff_xy in safe_standoff_xys(
            grid,
            target_xy,
            standoff_m=float(config.standoff_m),
            n_candidates=int(config.standoff_candidates),
        ):
            if grid.to_grid(standoff_xy) not in reachable:
                continue
            if config.require_line_of_sight and not scene.has_line_of_sight(
                standoff_xy,
                target_xy,
                exclude_landmark_xy=target_xy,
            ):
                continue
            reachable_standoffs += 1
            if corridor_width_m(grid, standoff_xy) >= float(
                config.minimum_navigable_corridor_width_m
            ):
                navigable_standoffs += 1
        claim_reachable = claim_cells > 0
        preferred_standoff_reachable = (
            navigable_standoffs
            >= int(config.minimum_navigable_standoffs_per_beacon)
        )
        beacon_reports.append(
            BeaconFixedSpawnReachability(
                object_id=str(landmark.object_id),
                target_xy_m=target_xy,
                reachable=claim_reachable,
                claim_reachable=claim_reachable,
                preferred_standoff_reachable=preferred_standoff_reachable,
                reachable_claim_cell_count=int(claim_cells),
                reachable_standoff_count=int(reachable_standoffs),
                reachable_navigable_standoff_count=int(navigable_standoffs),
                closest_reachable_distance_m=(
                    None if closest is None else float(closest)
                ),
            )
        )

    fully_reachable = bool(beacon_reports) and all(
        beacon.claim_reachable for beacon in beacon_reports
    )
    all_have_preferred_standoff = bool(beacon_reports) and all(
        beacon.preferred_standoff_reachable for beacon in beacon_reports
    )
    if not spawn_in_bounds:
        failure = "fixed_spawn_outside_world_bounds"
    elif not spawn_body_clear:
        failure = "fixed_spawn_lacks_body_clearance"
    elif not spawn_cell_free:
        failure = "fixed_spawn_raster_cell_blocked"
    elif not beacon_reports:
        failure = "scene_has_no_beacons"
    elif not fully_reachable:
        missing = ",".join(
            beacon.object_id for beacon in beacon_reports if not beacon.reachable
        )
        failure = f"beacons_unreachable_from_fixed_spawn:{missing}"
    else:
        failure = ""

    return FixedSpawnAuditReport(
        scene_id=str(manifest.scene_id),
        family=str(manifest.family),
        source_split=manifest.split,
        config=config,
        spawn_xy_m=spawn_xy,
        spawn_grid_cell=spawn_cell,
        spawn_obstacle_clearance_m=spawn_clearance,
        spawn_is_body_clear=spawn_body_clear,
        reachable_cell_count=len(reachable),
        reachable_area_m2=float(len(reachable) * config.cell_size_m**2),
        beacons=tuple(beacon_reports),
        fully_reachable=fully_reachable,
        all_beacons_have_preferred_standoff=all_have_preferred_standoff,
        failure_reason=failure,
        grid_origin_xy_m=(float(grid.origin_xy[0]), float(grid.origin_xy[1])),
        grid_shape=(int(nx), int(ny)),
        reachable_cells=reachable,
        coverage_grid_origin_xy_m=coverage_origin,
        coverage_grid_shape=coverage_shape,
        coverage_reachable_cell_count=len(coverage_reachable),
        coverage_reachable_area_m2=float(
            len(coverage_reachable) * config.coverage_cell_size_m**2
        ),
        coverage_reachable_cells=coverage_reachable,
    )


def _reachable_component(
    free: np.ndarray,
    start: tuple[int, int],
    *,
    connectivity: int,
    allow_diagonal_corner_cutting: bool,
) -> frozenset[tuple[int, int]]:
    nx, ny = int(free.shape[0]), int(free.shape[1])
    if not (0 <= start[0] < nx and 0 <= start[1] < ny and free[start]):
        return frozenset()

    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    seen: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in cardinal:
            cell = (x + dx, y + dy)
            if (
                0 <= cell[0] < nx
                and 0 <= cell[1] < ny
                and free[cell]
                and cell not in seen
            ):
                seen.add(cell)
                queue.append(cell)
        if connectivity != 8:
            continue
        for dx, dy in diagonal:
            cell = (x + dx, y + dy)
            if not (
                0 <= cell[0] < nx
                and 0 <= cell[1] < ny
                and free[cell]
                and cell not in seen
            ):
                continue
            if (
                not allow_diagonal_corner_cutting
                and (not free[x + dx, y] or not free[x, y + dy])
            ):
                continue
            seen.add(cell)
            queue.append(cell)
    return frozenset(seen)


__all__ = [
    "BeaconFixedSpawnReachability",
    "FixedSpawnAuditConfig",
    "FixedSpawnAuditReport",
    "audit_fixed_spawn",
]
