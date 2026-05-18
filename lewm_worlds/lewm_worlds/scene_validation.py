"""Reachability validation for generated scenes.

The route teacher plans on the configuration-space inflated grid (see
:mod:`lewm_worlds.planning_grid`). A scene is *valid for mass data
generation* iff every random spawn pose can navigate to every beacon —
otherwise the rollout produces episodes whose ground-truth solution
depends on which disconnected free-space component the spawn happened
to land in.

This module exposes a single audit function used both by the corpus
builder (to retry seeds that produce fragmented scenes) and by tests
(to assert that planning-grid-based collectors see solvable scenes).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from lewm_worlds.manifest import SceneManifest
from lewm_worlds.planning_grid import (
    InflatedOccupancyGrid,
    safe_standoff_xys,
)
from lewm_worlds.scene_graph import SceneGraph


@dataclass(frozen=True)
class SceneReachabilityReport:
    """Summary of a scene's grid-level reachability properties."""

    is_valid: bool
    beacon_count: int
    component_count: int
    largest_component_size: int
    # The free component that contains every beacon. The spawn sampler
    # should restrict to grid cells in this component so the robot
    # always starts somewhere it can reach all goals. ``-1`` when no
    # such component exists (scene is fragmented across beacon rooms).
    canonical_component_id: int
    unreachable_beacons: tuple[int, ...]
    spawn_candidate_count: int
    spawn_candidates_in_canonical: int
    failure_reason: str = ""


def audit_scene_reachability(
    manifest: SceneManifest,
    *,
    cell_size_m: float = 0.05,
    inflation_m: float = 0.20,
    standoff_m: float = 0.85,
    n_standoff_candidates: int = 12,
    spawn_clearance_floor_m: float = 0.20,
    min_spawn_cells_in_canonical: int = 1,
) -> SceneReachabilityReport:
    """Return a reachability audit of ``manifest``.

    A scene is *valid* when:

    1. There exists a single grid-free connected component that contains
       at least one LOS-valid standoff for every beacon (the
       *canonical* component).
    2. That component contains at least ``min_spawn_cells_in_canonical``
       spawn-candidate cells, so the spawn sampler can actually drop the
       robot somewhere it can reach every goal.

    The spawn sampler is then expected to restrict its picks to the
    canonical component — cells in smaller isolated free regions (e.g.
    closets behind walls) are not used as spawn poses.
    """

    grid = InflatedOccupancyGrid(
        manifest, cell_size_m=cell_size_m, inflation_m=inflation_m
    )
    components = _component_index(grid.free_mask)
    n_components = int(components.max())  # labels are 1..n_components
    sizes = np.bincount(components.ravel())
    largest = int(sizes[1:].max()) if n_components > 0 else 0

    scene = SceneGraph(manifest)
    beacon_components: list[tuple[int, set[int]]] = []
    for _name, beacon_cell in scene.landmark_cells:
        beacon_xy = scene.landmark_xy_for_cell(beacon_cell)
        if beacon_xy is None:
            continue
        candidates = safe_standoff_xys(
            grid,
            beacon_xy,
            standoff_m=standoff_m,
            n_candidates=n_standoff_candidates,
        )
        comp_set: set[int] = set()
        for standoff in candidates:
            if not scene.has_line_of_sight(
                standoff, beacon_xy, exclude_landmark_xy=beacon_xy
            ):
                continue
            ix, iy = grid.to_grid(standoff)
            nx, ny = grid.shape
            if 0 <= ix < nx and 0 <= iy < ny and components[ix, iy] > 0:
                comp_set.add(int(components[ix, iy]))
        beacon_components.append((int(beacon_cell), comp_set))

    # Canonical component = a single component shared by all beacons.
    # If beacons disagree on which component they sit in, no canonical
    # component exists.
    if beacon_components and all(comps for _, comps in beacon_components):
        candidates = set.intersection(*(comps for _, comps in beacon_components))
    else:
        candidates = set()
    # Among shared components, pick the largest (most spawn options).
    canonical_id = -1
    if candidates:
        canonical_id = max(candidates, key=lambda c: int(sizes[c]))

    unreachable = tuple(
        int(b) for b, comps in beacon_components if canonical_id < 0 or canonical_id not in comps
    )

    spawn_cells = _spawn_candidate_grid_cells(
        manifest, grid, spawn_clearance_floor_m
    )
    spawn_in_canonical = (
        sum(1 for i, j in spawn_cells if int(components[i, j]) == canonical_id)
        if canonical_id > 0
        else 0
    )

    failure = ""
    is_valid = True
    if canonical_id < 0:
        is_valid = False
        failure = "no_component_contains_all_beacons"
    elif spawn_in_canonical < int(min_spawn_cells_in_canonical):
        is_valid = False
        failure = "canonical_component_has_no_spawn_candidates"

    return SceneReachabilityReport(
        is_valid=is_valid,
        beacon_count=len(manifest.landmarks),
        component_count=int(n_components),
        largest_component_size=largest,
        canonical_component_id=int(canonical_id),
        unreachable_beacons=unreachable,
        spawn_candidate_count=len(spawn_cells),
        spawn_candidates_in_canonical=int(spawn_in_canonical),
        failure_reason=failure,
    )




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _component_index(free_mask: np.ndarray) -> np.ndarray:
    """Label connected free-space components, 0 = obstacle."""

    nx, ny = free_mask.shape
    labels = np.zeros((nx, ny), dtype=np.int32)
    label = 0
    for i0 in range(nx):
        for j0 in range(ny):
            if not free_mask[i0, j0] or labels[i0, j0] != 0:
                continue
            label += 1
            stack: deque[tuple[int, int]] = deque([(i0, j0)])
            while stack:
                i, j = stack.pop()
                if i < 0 or i >= nx or j < 0 or j >= ny:
                    continue
                if labels[i, j] != 0 or not free_mask[i, j]:
                    continue
                labels[i, j] = label
                stack.append((i + 1, j))
                stack.append((i - 1, j))
                stack.append((i, j + 1))
                stack.append((i, j - 1))
    return labels


def _spawn_candidate_grid_cells(
    manifest: SceneManifest,
    grid: InflatedOccupancyGrid,
    clearance_floor_m: float,
) -> list[tuple[int, int]]:
    """Return grid cells corresponding to room-cell spawn candidates."""

    scene = SceneGraph(manifest)
    out: list[tuple[int, int]] = []
    nx, ny = grid.shape
    for node in manifest.graph_nodes:
        xy = node.center_xy_m
        if scene.clearance_to_walls(xy) < clearance_floor_m:
            continue
        snapped = grid.nearest_free(xy, max_radius_m=4.0 * grid.inflation_m)
        if snapped is None:
            continue
        i, j = grid.to_grid(snapped)
        if 0 <= i < nx and 0 <= j < ny and grid.free_mask[i, j]:
            out.append((int(i), int(j)))
    return out
