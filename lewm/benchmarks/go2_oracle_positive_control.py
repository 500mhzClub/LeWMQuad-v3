"""Ground-truth positive control for Go2 maze coverage and beacon claiming.

This module deliberately contains no learned model, renderer, benchmark-monolith
import, or online-belief-map dependency.  It answers a narrower question: given
the exact static occupancy, fixed scene spawn, and beacon geometry, can a
deterministic planner and a primitive-block follower solve the development
mazes under the repository's kinematic command contract?

The legacy geometry-v1 path uses the versioned inflated configuration-space
grid directly. Geometry v2 instead loads exact 0.47 m disc occupancy into the
shared ``OnlineBeliefMap``, routes only through its confirmed-free API, and
strictly scores every executed actual-yaw microstep with the geometry-bound
observed-maximum directional polygon. The planner emits one unconditional
attempt at each reached terminal pose; the shared physical evaluator scores the
completed scene trace once after execution. Coverage is the
fraction of the spawn-connected inflated free grid swept by a configurable
visit radius; its AUC is normalized by the full tick budget so early coverage
scores better than late coverage.

The CLI is development-only.  It consumes an explicit allow-list and refuses
paths labelled as sealed/final evaluation artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DEVELOPMENT_CORPUS = (
    REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z"
)
GENERALIZATION_DEVELOPMENT_CORPUS = (
    REPO_ROOT / ".generated/scene_corpus/go2_generalization_v3"
)
DEFAULT_DEVELOPMENT_MANIFEST = (
    REPO_ROOT / "config/go2_generalization_v3/development.json"
)
for _source_root in (REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from lewm_genesis.lewm_contract import (  # noqa: E402
    PrimitiveRegistry,
    expand_primitive_to_block,
)
from lewm_worlds.manifest import (  # noqa: E402
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from lewm.planning.geometry_contract import (  # noqa: E402
    DEFAULT_GEOMETRY_CONTRACT,
    GeometryContract,
    load_geometry_contract,
)
from lewm.planning.exact_occupancy_belief_adapter import (  # noqa: E402
    ExactOccupancyAgreement,
    ExactOccupancyBeliefAdapter,
)
from lewm.planning.online_belief_map import OnlineBeliefMap  # noqa: E402
from lewm.planning.oriented_footprint import (  # noqa: E402
    ManifestDirectionalFootprintFeasibility,
    Pose2D as FootprintPose2D,
)
from lewm.benchmarks.go2_physical_eligibility import (  # noqa: E402
    LoadedDirectionalPolicy,
    policy_from_geometry_contract,
    validate_loaded_directional_policy_content,
)
from lewm.benchmarks.experiment_manifest import (  # noqa: E402
    build_experiment_manifest,
)
from lewm.benchmarks.go2_physical_claim_evaluator import (  # noqa: E402
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (  # noqa: E402
    build_claim_attempt,
    build_claim_trace,
    canonical_task_object_ids,
    object_id_reference,
    task_object_set_sha256,
)


GridCell = tuple[int, int]

_CPU_THREAD_CAP_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

CANONICAL_PHYSICAL_CLAIM_REGRESSION_OUTPUT = (
    REPO_ROOT
    / ".generated/oracle_positive_control/go2_generalization_v4_development"
    / "canonical_physical_claim_v1_report.json"
)


@dataclass(frozen=True)
class OracleConfig:
    """Positive-control geometry and follower contract."""

    grid_cell_m: float = 0.05
    planning_inflation_m: float = 0.20
    body_forward_m: float = 0.40
    body_rear_m: float = 0.30
    body_half_width_m: float = 0.24
    body_probe_margin_m: float = 0.03
    claim_distance_m: float = 1.20
    claim_bearing_rad: float = 0.25
    preferred_standoff_m: float = 1.05
    standoff_candidates: int = 32
    claim_anchor_limit: int = 64
    route_clearance_weight: float = 0.35
    route_clearance_target_m: float = 0.10
    route_lookahead_m: float = 0.45
    route_goal_tolerance_m: float = 0.11
    route_replan_ticks: int = 16
    max_goal_ticks: int = 500
    no_progress_limit: int = 40
    coverage_resolution_m: float = 0.40
    coverage_visit_radius_m: float = 0.28
    coverage_cell_m: float = 0.10
    coverage_completion_fraction: float = 0.90
    maximum_translation_substep_m: float = 0.025
    minimum_progress_m: float = 0.001
    planning_connectivity: int = 8
    allow_diagonal_corner_cutting: bool = False
    max_ticks: int = 2400

    @classmethod
    def from_geometry_contract(
        cls,
        geometry: GeometryContract,
        **overrides: Any,
    ) -> "OracleConfig":
        values: dict[str, Any] = {
            "grid_cell_m": geometry.configuration_space.oracle_cell_size_m,
            "planning_inflation_m": geometry.configuration_space.body_inflation_radius_m,
            "body_forward_m": geometry.swept_footprint.forward_m,
            "body_rear_m": geometry.swept_footprint.rear_m,
            "body_half_width_m": geometry.swept_footprint.half_width_m,
            "body_probe_margin_m": geometry.swept_footprint.probe_margin_m,
            "claim_distance_m": geometry.visibility_and_claim.claim_radius_m,
            "preferred_standoff_m": geometry.visibility_and_claim.standoff_m,
            "standoff_candidates": geometry.visibility_and_claim.standoff_candidates,
            "coverage_cell_m": geometry.coverage.cell_size_m,
            "maximum_translation_substep_m": (
                geometry.kinematic_execution.maximum_translation_substep_m
            ),
            "minimum_progress_m": geometry.kinematic_execution.minimum_progress_m,
            "planning_connectivity": geometry.configuration_space.connectivity,
            "allow_diagonal_corner_cutting": (
                geometry.configuration_space.allow_diagonal_corner_cutting
            ),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))


@dataclass(frozen=True)
class Beacon:
    color: str
    xy: tuple[float, float]
    object_id: str

    @property
    def claim_id(self) -> str:
        return str(self.object_id)


@dataclass(frozen=True)
class PrimitiveSimulation:
    primitive: str
    poses: tuple[Pose2D, ...]
    completed: bool
    blocked_reason: str | None
    minimum_swept_probe_clearance_m: float

    @property
    def end_pose(self) -> Pose2D:
        return self.poses[-1]


@dataclass(frozen=True)
class ClaimAnchor:
    beacon: Beacon
    cell: GridCell
    xy: tuple[float, float]
    yaw: float
    path_cost_cells: float


@dataclass(frozen=True)
class OracleClaimTaskBinding:
    """Task identity committed before the oracle executes its first motion."""

    trace_id: str
    episode_id: str
    task_object_ids: tuple[str, ...]
    task_object_set_sha256: str


@dataclass(frozen=True)
class CoveragePlan:
    representatives: dict[tuple[int, int], GridCell]
    walk: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OracleRoute:
    """One route returned by either the legacy grid or shared belief map."""

    waypoints_xy: tuple[tuple[float, float], ...]
    cost_cells: float


class RoutePlanner:
    """Minimal route interface used by claim and coverage execution."""

    source: str

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> OracleRoute | None:
        raise NotImplementedError

    def telemetry(self) -> dict[str, Any]:
        raise NotImplementedError


class InflatedGridRoutePlanner(RoutePlanner):
    """Legacy v1 A* route source."""

    source = "InflatedOccupancyGrid.astar"

    def __init__(self, grid: InflatedOccupancyGrid, config: OracleConfig) -> None:
        self.grid = grid
        self.config = config
        self.queries = 0
        self.failures = 0

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> OracleRoute | None:
        self.queries += 1
        path = self.grid.astar(
            start_xy,
            goal_xy,
            clearance_weight=float(self.config.route_clearance_weight),
            clearance_target_m=float(self.config.route_clearance_target_m),
        )
        if path is None:
            self.failures += 1
            return None
        return OracleRoute(
            waypoints_xy=tuple(
                (float(xy[0]), float(xy[1])) for xy in path.waypoints_xy
            ),
            cost_cells=float(path.cost_cells),
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "queries": int(self.queries),
            "failures": int(self.failures),
        }


class SharedBeliefMapRoutePlanner(RoutePlanner):
    """Geometry-v2 route source restricted to confirmed-free shared-map cells."""

    source = "OnlineBeliefMap.shortest_path"

    def __init__(self, belief_map: OnlineBeliefMap) -> None:
        self.belief_map = belief_map
        self.queries = 0
        self.failures = 0

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> OracleRoute | None:
        self.queries += 1
        start = self.belief_map.world_to_cell(start_xy)
        goal = self.belief_map.world_to_cell(goal_xy)
        cells = self.belief_map.shortest_path(start, goal)
        if cells is None:
            self.failures += 1
            return None
        return OracleRoute(
            waypoints_xy=tuple(self.belief_map.cell_center(cell) for cell in cells),
            cost_cells=float(max(0, len(cells) - 1)),
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "queries": int(self.queries),
            "failures": int(self.failures),
            "confirmed_free_cells": len(self.belief_map.confirmed_free_cells()),
        }


def wrap_angle_pi(angle: float) -> float:
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def _yaw_from_quat_wxyz(quat: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quat)
    return wrap_angle_pi(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _bearing_to(pose: Pose2D, xy: tuple[float, float]) -> float:
    return wrap_angle_pi(math.atan2(float(xy[1]) - pose.y, float(xy[0]) - pose.x) - pose.yaw)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _body_probe_clearance(
    grid: InflatedOccupancyGrid,
    pose: Pose2D,
    config: OracleConfig,
) -> float:
    forward = (math.cos(pose.yaw), math.sin(pose.yaw))
    left = (-forward[1], forward[0])
    probes = (
        (0.0, 0.0),
        (config.body_forward_m, 0.0),
        (config.body_forward_m, config.body_half_width_m),
        (config.body_forward_m, -config.body_half_width_m),
        (-config.body_rear_m, 0.0),
        (-config.body_rear_m, config.body_half_width_m),
        (-config.body_rear_m, -config.body_half_width_m),
        (0.0, config.body_half_width_m),
        (0.0, -config.body_half_width_m),
    )
    clearance = math.inf
    for ahead, lateral in probes:
        xy = (
            pose.x + ahead * forward[0] + lateral * left[0],
            pose.y + ahead * forward[1] + lateral * left[1],
        )
        clearance = min(
            clearance,
            float(grid.obstacle_clearance_m(xy)) - float(config.body_probe_margin_m),
        )
    return float(clearance)


def _pose_feasible(
    grid: InflatedOccupancyGrid,
    pose: Pose2D,
    config: OracleConfig,
) -> tuple[bool, str | None]:
    if not grid.is_free(pose.xy):
        return False, "inflated_center_grid"
    return True, None


def simulate_primitive(
    pose: Pose2D,
    primitive: str,
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid,
    config: OracleConfig,
) -> PrimitiveSimulation:
    """Integrate one canonical command block without mutating simulator state.

    Translation is evaluated at the current yaw, then yaw is advanced, matching
    ``benchmark_lewm_closed_loop_mpc._execute_kinematic_primitive``.  Unlike
    that legacy helper, this positive control also rejects canonical footprint
    violations rather than allowing a center-only move.
    """

    current = pose
    accepted: list[Pose2D] = [pose]
    minimum_probe_clearance = _body_probe_clearance(grid, pose, config)
    for vx_body, vy_body, yaw_rate in expand_primitive_to_block(registry, primitive):
        command_translation = math.hypot(float(vx_body), float(vy_body)) * float(
            registry.command_dt_s
        )
        microsteps = max(
            1,
            int(
                math.ceil(
                    command_translation
                    / max(1e-6, float(config.maximum_translation_substep_m))
                )
            ),
        )
        micro_dt = float(registry.command_dt_s) / float(microsteps)
        for _ in range(microsteps):
            cos_yaw = math.cos(current.yaw)
            sin_yaw = math.sin(current.yaw)
            next_pose = Pose2D(
                x=current.x
                + (float(vx_body) * cos_yaw - float(vy_body) * sin_yaw) * micro_dt,
                y=current.y
                + (float(vx_body) * sin_yaw + float(vy_body) * cos_yaw) * micro_dt,
                yaw=wrap_angle_pi(current.yaw + float(yaw_rate) * micro_dt),
            )
            feasible, reason = _pose_feasible(grid, next_pose, config)
            if not feasible:
                return PrimitiveSimulation(
                    primitive=str(primitive),
                    poses=tuple(accepted),
                    completed=False,
                    blocked_reason=reason,
                    minimum_swept_probe_clearance_m=float(minimum_probe_clearance),
                )
            accepted.append(next_pose)
            current = next_pose
            minimum_probe_clearance = min(
                minimum_probe_clearance,
                _body_probe_clearance(grid, next_pose, config),
            )
    return PrimitiveSimulation(
        primitive=str(primitive),
        poses=tuple(accepted),
        completed=True,
        blocked_reason=None,
        minimum_swept_probe_clearance_m=float(minimum_probe_clearance),
    )


def _legal_neighbors(
    grid: InflatedOccupancyGrid,
    cell: GridCell,
    *,
    connectivity: int = 8,
    allow_diagonal_corner_cutting: bool = False,
) -> tuple[GridCell, ...]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    free = grid.free_mask
    nx, ny = grid.shape
    x, y = cell
    out: list[GridCell] = []
    for dx, dy in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        if connectivity == 4 and dx != 0 and dy != 0:
            continue
        neighbor = (x + dx, y + dy)
        if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
            continue
        if not bool(free[neighbor]):
            continue
        if (
            dx != 0
            and dy != 0
            and not allow_diagonal_corner_cutting
            and (
            not bool(free[x + dx, y]) or not bool(free[x, y + dy])
            )
        ):
            continue
        out.append(neighbor)
    return tuple(out)


def reachable_component(
    grid: InflatedOccupancyGrid,
    start_xy: tuple[float, float],
    *,
    connectivity: int = 8,
    allow_diagonal_corner_cutting: bool = False,
) -> tuple[GridCell, set[GridCell], float]:
    """Return snapped start, exact A*-compatible component, and snap distance."""

    snapped = start_xy if grid.is_free(start_xy) else grid.nearest_free(
        start_xy,
        max_radius_m=max(0.5, 4.0 * float(grid.inflation_m)),
    )
    if snapped is None:
        raise ValueError("spawn has no free cell within the canonical snap radius")
    start = grid.to_grid(snapped)
    component: set[GridCell] = {start}
    queue: deque[GridCell] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in _legal_neighbors(
            grid,
            cell,
            connectivity=connectivity,
            allow_diagonal_corner_cutting=allow_diagonal_corner_cutting,
        ):
            if neighbor in component:
                continue
            component.add(neighbor)
            queue.append(neighbor)
    return start, component, _distance_xy(start_xy, grid.to_world(start))


def _beacon_color(material_id: str, object_id: str) -> str:
    text = f"{material_id} {object_id}".lower()
    for color in ("green", "yellow", "blue", "red"):
        if color in text:
            return color
    return str(object_id)


def beacons_from_manifest(manifest: SceneManifest) -> tuple[Beacon, ...]:
    beacons = [
        Beacon(
            color=_beacon_color(item.material_id, item.object_id),
            xy=(float(item.center_xyz_m[0]), float(item.center_xyz_m[1])),
            object_id=str(item.object_id),
        )
        for item in manifest.landmarks
    ]
    return tuple(sorted(beacons, key=lambda item: (item.color, item.object_id)))


def bind_oracle_claim_task(manifest: SceneManifest) -> OracleClaimTaskBinding:
    """Freeze the complete manifest-landmark task before controller motion."""

    trace_id = f"oracle:{manifest.scene_id}"
    episode_id = f"oracle:{manifest.scene_id}:episode"
    task_ids = canonical_task_object_ids(manifest)
    return OracleClaimTaskBinding(
        trace_id=trace_id,
        episode_id=episode_id,
        task_object_ids=task_ids,
        task_object_set_sha256=task_object_set_sha256(manifest, task_ids),
    )


def _oracle_claim_attempt_id(
    *,
    trace_id: str,
    episode_id: str,
    scene_id: str,
    task_object_id: str,
) -> str:
    payload = {
        "domain": "lewm-go2-oracle-claim-attempt-v1",
        "episode_id": episode_id,
        "scene_id": scene_id,
        "task_object_id": task_object_id,
        "trace_id": trace_id,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def evaluate_physical_claim_trace_oracle_adapter(
    trace: Mapping[str, Any],
    physical_manifest: SceneManifest,
    expected_task_object_ids: Sequence[str],
    expected_task_object_set_sha256: str,
) -> dict[str, Any]:
    """Evaluate the completed oracle trace exactly once after execution."""

    return evaluate_physical_claim_trace(
        trace,
        physical_manifest,
        expected_task_object_ids,
        expected_task_object_set_sha256,
    )


def _has_claim_line(
    visibility_grid: InflatedOccupancyGrid,
    pose_xy: tuple[float, float],
    beacon_xy: tuple[float, float],
) -> bool:
    return bool(visibility_grid.has_free_line(pose_xy, beacon_xy))


def _planned_claim_standoff(config: OracleConfig) -> float:
    # Leave deterministic execution margin between the planned terminal pose
    # and the immutable physical threshold. This is planning geometry only;
    # claim acceptance still comes exclusively from the shared evaluator.
    return max(
        0.0,
        min(
            float(config.preferred_standoff_m),
            float(config.claim_distance_m) - 0.25,
        ),
    )


def claim_anchor_cells(
    beacon: Beacon,
    component: set[GridCell],
    planning_grid: InflatedOccupancyGrid,
    visibility_grid: InflatedOccupancyGrid,
    config: OracleConfig,
) -> tuple[tuple[GridCell, float, float], ...]:
    """All connected true-claim cells, ranked near the preferred standoff.

    The contract's angular standoff count is a route-sampling prior, not the
    scene-validity definition. Exact fixed-spawn reachability must not fail
    merely because a sampled ring missed a narrow but valid claim region.
    """

    candidates: list[tuple[GridCell, float, float]] = []
    for cell in component:
        xy = planning_grid.to_world(cell)
        distance = _distance_xy(xy, beacon.xy)
        if distance > float(config.claim_distance_m):
            continue
        if not _has_claim_line(visibility_grid, xy, beacon.xy):
            continue
        yaw = math.atan2(beacon.xy[1] - xy[1], beacon.xy[0] - xy[0])
        feasible, _ = _pose_feasible(
            planning_grid,
            Pose2D(float(xy[0]), float(xy[1]), float(yaw)),
            config,
        )
        if feasible:
            candidates.append((cell, distance, float(yaw)))
    candidates = sorted(
        candidates,
        key=lambda item: (
            abs(float(item[1]) - _planned_claim_standoff(config)),
            item[0][0],
            item[0][1],
        ),
    )
    return tuple(candidates)


def _best_claim_anchor(
    pose: Pose2D,
    beacon: Beacon,
    candidates: Sequence[tuple[GridCell, float, float]],
    grid: InflatedOccupancyGrid,
    config: OracleConfig,
    route_planner: RoutePlanner,
    *,
    excluded: set[GridCell] | None = None,
) -> ClaimAnchor | None:
    excluded = excluded or set()
    ordered = sorted(
        (item for item in candidates if item[0] not in excluded),
        key=lambda item: (
            abs(float(item[1]) - _planned_claim_standoff(config)),
            _distance_xy(pose.xy, grid.to_world(item[0])),
            item[0],
        ),
    )[: max(1, int(config.claim_anchor_limit))]
    best: ClaimAnchor | None = None
    for cell, _distance, yaw in ordered:
        xy = grid.to_world(cell)
        path = route_planner.plan(pose.xy, xy)
        if path is None:
            continue
        anchor = ClaimAnchor(
            beacon=beacon,
            cell=cell,
            xy=xy,
            yaw=float(yaw),
            path_cost_cells=float(path.cost_cells),
        )
        if best is None or (
            anchor.path_cost_cells,
            anchor.beacon.claim_id,
            anchor.cell,
        ) < (
            best.path_cost_cells,
            best.beacon.claim_id,
            best.cell,
        ):
            best = anchor
    return best


def _snap_to_component(
    grid: InflatedOccupancyGrid,
    component: set[GridCell],
    xy: tuple[float, float],
    *,
    max_radius_m: float,
) -> tuple[float, float] | None:
    """Snap a world point to a specific connected planning component.

    Coverage and planning use distinct canonical raster resolutions. A free
    coverage-cell center may therefore land in a blocked planning cell even
    though the represented area is reachable. Restricting the snap to the
    known spawn component preserves reachability and deterministic tie breaks.
    """

    center = grid.to_grid(xy)
    radius_cells = int(math.ceil(float(max_radius_m) / grid.cell_size_m))
    candidates: list[tuple[float, GridCell]] = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            cell = (center[0] + dx, center[1] + dy)
            if cell not in component:
                continue
            world = grid.to_world(cell)
            distance = _distance_xy(xy, world)
            if distance <= float(max_radius_m) + 1e-9:
                candidates.append((distance, cell))
    if not candidates:
        return None
    _, cell = min(candidates, key=lambda item: (item[0], item[1]))
    return grid.to_world(cell)


def build_coverage_plan(
    grid: InflatedOccupancyGrid,
    component: set[GridCell],
    start: GridCell,
    config: OracleConfig,
) -> CoveragePlan:
    """Build a deterministic DFS walk over a connected coarse free-cell graph."""

    stride = max(1, int(round(float(config.coverage_resolution_m) / grid.cell_size_m)))
    by_bin: dict[tuple[int, int], list[GridCell]] = defaultdict(list)
    for cell in component:
        by_bin[(cell[0] // stride, cell[1] // stride)].append(cell)
    clearance = grid.obstacle_clearance_grid_m
    representatives: dict[tuple[int, int], GridCell] = {}
    for coarse, cells in by_bin.items():
        representatives[coarse] = min(
            cells,
            key=lambda cell: (-float(clearance[cell]), cell[0], cell[1]),
        )

    graph: dict[tuple[int, int], set[tuple[int, int]]] = {
        coarse: set() for coarse in representatives
    }
    for cell in component:
        source = (cell[0] // stride, cell[1] // stride)
        for neighbor in _legal_neighbors(
            grid,
            cell,
            connectivity=int(config.planning_connectivity),
            allow_diagonal_corner_cutting=bool(
                config.allow_diagonal_corner_cutting
            ),
        ):
            if neighbor not in component:
                continue
            target = (neighbor[0] // stride, neighbor[1] // stride)
            if target != source:
                graph[source].add(target)
                graph[target].add(source)

    root = (start[0] // stride, start[1] // stride)
    seen: set[tuple[int, int]] = {root}
    walk: list[tuple[int, int]] = [root]

    def visit(node: tuple[int, int]) -> None:
        for neighbor in sorted(graph[node]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            walk.append(neighbor)
            visit(neighbor)
            walk.append(node)

    visit(root)
    return CoveragePlan(
        representatives=representatives,
        walk=tuple(walk),
    )


class OracleSimulator:
    def __init__(
        self,
        *,
        pose: Pose2D,
        registry: PrimitiveRegistry,
        planning_grid: InflatedOccupancyGrid,
        visibility_grid: InflatedOccupancyGrid,
        coverage_grid: InflatedOccupancyGrid,
        coverage_component: set[GridCell],
        beacons: Sequence[Beacon],
        config: OracleConfig,
        route_planner: RoutePlanner,
        directional_checker: ManifestDirectionalFootprintFeasibility | None = None,
    ) -> None:
        self.pose = pose
        self.registry = registry
        self.grid = planning_grid
        self.visibility_grid = visibility_grid
        self.coverage_grid = coverage_grid
        self.coverage_component = coverage_component
        self.beacons = tuple(beacons)
        self.config = config
        self.route_planner = route_planner
        self.directional_checker = directional_checker
        self.tick = 0
        self.collision_attempts = 0
        self.stalls = 0
        self.blocked_reasons: dict[str, int] = defaultdict(int)
        self.blocked_candidate_evaluations = 0
        self.swept_probe_risk_blocks = 0
        self.minimum_swept_probe_clearance_m = math.inf
        self.covered: set[GridCell] = set()
        self.coverage_fractions: list[float] = []
        self.primitive_counts: dict[str, int] = defaultdict(int)
        self.trajectory: list[dict[str, int | float | str]] = [
            {
                "primitive_tick": 0,
                "microstep": 0,
                "primitive": "initial_pose",
                "x_m": round(float(pose.x), 8),
                "y_m": round(float(pose.y), 8),
                "yaw_rad": round(float(pose.yaw), 8),
            }
        ]
        self.directional_collision_segments = 0
        self.directional_sweep_samples_evaluated = 0
        self.directional_collision_object_ids: set[str] = set()
        self.directional_initial_pose_feasible: bool | None = None
        if self.directional_checker is not None:
            initial = self.directional_checker.pose_feasibility(
                FootprintPose2D(pose.x, pose.y, pose.yaw)
            )
            self.directional_initial_pose_feasible = bool(initial.feasible)
            if not initial.feasible:
                self.directional_collision_segments += 1
                self.directional_collision_object_ids.update(
                    initial.colliding_object_ids
                )
        self._coverage_offsets = self._make_coverage_offsets()
        self._mark_coverage(self.pose)
        self.coverage_fractions.append(self.coverage_fraction)

    @property
    def coverage_fraction(self) -> float:
        return float(len(self.covered) / max(1, len(self.coverage_component)))

    def _make_coverage_offsets(self) -> tuple[GridCell, ...]:
        radius_cells = int(
            math.ceil(
                float(self.config.coverage_visit_radius_m)
                / self.coverage_grid.cell_size_m
            )
        )
        offsets = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx, dy) * self.coverage_grid.cell_size_m <= float(
                    self.config.coverage_visit_radius_m
                ) + 1e-9:
                    offsets.append((dx, dy))
        return tuple(offsets)

    def _mark_coverage(self, pose: Pose2D) -> None:
        center = self.coverage_grid.to_grid(pose.xy)
        for dx, dy in self._coverage_offsets:
            cell = (center[0] + dx, center[1] + dy)
            if cell in self.coverage_component:
                self.covered.add(cell)

    def execute(self, simulation: PrimitiveSimulation) -> None:
        if self.tick >= int(self.config.max_ticks):
            return
        start = self.pose
        for microstep, pose in enumerate(simulation.poses[1:], start=1):
            if self.directional_checker is not None:
                sweep = self.directional_checker.swept_pose_feasibility(
                    FootprintPose2D(self.pose.x, self.pose.y, self.pose.yaw),
                    FootprintPose2D(pose.x, pose.y, pose.yaw),
                    maximum_corner_step_m=float(
                        self.config.maximum_translation_substep_m
                    ),
                )
                self.directional_sweep_samples_evaluated += int(
                    sweep.samples_evaluated
                )
                if not sweep.feasible:
                    self.directional_collision_segments += 1
                    first = sweep.first_infeasible_pose
                    if first is not None:
                        self.directional_collision_object_ids.update(
                            first.colliding_object_ids
                        )
            self.pose = pose
            self.trajectory.append(
                {
                    "primitive_tick": int(self.tick),
                    "microstep": int(microstep),
                    "primitive": str(simulation.primitive),
                    "x_m": round(float(pose.x), 8),
                    "y_m": round(float(pose.y), 8),
                    "yaw_rad": round(float(pose.yaw), 8),
                }
            )
            self._mark_coverage(pose)
        if not simulation.completed:
            self.collision_attempts += 1
            reason = str(simulation.blocked_reason or "unknown")
            self.blocked_reasons[reason] += 1
        self.minimum_swept_probe_clearance_m = min(
            self.minimum_swept_probe_clearance_m,
            float(simulation.minimum_swept_probe_clearance_m),
        )
        if float(simulation.minimum_swept_probe_clearance_m) < 0.0:
            self.swept_probe_risk_blocks += 1
        displacement = _distance_xy(start.xy, self.pose.xy)
        yaw_delta = abs(wrap_angle_pi(self.pose.yaw - start.yaw))
        if displacement < float(self.config.minimum_progress_m) and yaw_delta < 1e-4:
            self.stalls += 1
        self.tick += 1
        self.primitive_counts[simulation.primitive] += 1
        self.coverage_fractions.append(self.coverage_fraction)

    def hold(self) -> None:
        self.execute(
            simulate_primitive(
                self.pose,
                "hold",
                self.registry,
                self.grid,
                self.config,
            )
        )

    def normalized_coverage_auc(self) -> float:
        horizon = max(1, int(self.config.max_ticks))
        values = list(self.coverage_fractions[: horizon + 1])
        if not values:
            return 0.0
        if len(values) < horizon + 1:
            values.extend([values[-1]] * (horizon + 1 - len(values)))
        area = sum(0.5 * (values[index] + values[index + 1]) for index in range(horizon))
        return float(area / horizon)

    def trajectory_sha256(self) -> str:
        encoded = json.dumps(
            self.trajectory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _path_target(
    pose: Pose2D,
    waypoints: Sequence[tuple[float, float]],
    lookahead_m: float,
    grid: InflatedOccupancyGrid,
) -> tuple[float, float]:
    target = waypoints[0]
    path_distance = 0.0
    previous = pose.xy
    for candidate in waypoints:
        path_distance += _distance_xy(previous, candidate)
        if path_distance <= float(lookahead_m) and grid.has_free_line(
            pose.xy,
            candidate,
        ):
            target = candidate
        elif path_distance > float(lookahead_m):
            break
        previous = candidate
    return target


def _prune_route(
    pose: Pose2D,
    waypoints: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Drop A* waypoints already passed by the primitive follower."""

    if not waypoints:
        return ()
    closest = min(
        range(len(waypoints)),
        key=lambda index: (_distance_xy(pose.xy, waypoints[index]), index),
    )
    route = tuple(waypoints[closest:])
    while len(route) > 1 and _distance_xy(pose.xy, route[0]) < 0.08:
        route = route[1:]
    return route


def _candidate_cost(
    outcome: PrimitiveSimulation,
    target_xy: tuple[float, float],
    final_xy: tuple[float, float],
    final_yaw: float | None,
) -> float:
    end = outcome.end_pose
    target_distance = _distance_xy(end.xy, target_xy)
    target_bearing = abs(_bearing_to(end, target_xy))
    final_distance = _distance_xy(end.xy, final_xy)
    final_heading = 0.0
    if final_yaw is not None and final_distance < 0.65:
        final_heading = abs(wrap_angle_pi(float(final_yaw) - end.yaw))
    primitive_penalty = {
        "forward_medium": 0.0,
        "forward_slow": 0.015,
        "arc_left": 0.02,
        "arc_right": 0.02,
        "yaw_left": 0.035,
        "yaw_right": 0.035,
        "backward": 0.18,
    }.get(outcome.primitive, 0.25)
    swept_risk_penalty = 0.15 * max(
        0.0,
        -float(outcome.minimum_swept_probe_clearance_m),
    )
    return float(
        target_distance
        + 0.22 * target_bearing
        + 0.08 * final_distance
        + 0.30 * final_heading
        + primitive_penalty
        + swept_risk_penalty
    )


def _choose_primitive(
    simulator: OracleSimulator,
    target_xy: tuple[float, float],
    final_xy: tuple[float, float],
    final_yaw: float | None,
) -> PrimitiveSimulation | None:
    target_bearing = _bearing_to(simulator.pose, target_xy)
    final_distance = _distance_xy(simulator.pose.xy, final_xy)
    if (
        final_yaw is not None
        and final_distance <= float(simulator.config.route_goal_tolerance_m)
        and abs(wrap_angle_pi(float(final_yaw) - simulator.pose.yaw))
        > float(simulator.config.claim_bearing_rad)
    ):
        desired = (
            "yaw_left"
            if wrap_angle_pi(float(final_yaw) - simulator.pose.yaw) > 0.0
            else "yaw_right"
        )
    elif abs(target_bearing) > 0.30:
        desired = "yaw_left" if target_bearing > 0.0 else "yaw_right"
    elif abs(target_bearing) > 0.08:
        desired = "arc_left" if target_bearing > 0.0 else "arc_right"
    elif final_distance < 0.30:
        desired = "forward_slow"
    else:
        desired = "forward_medium"
    desired_outcome = simulate_primitive(
        simulator.pose,
        desired,
        simulator.registry,
        simulator.grid,
        simulator.config,
    )
    if desired_outcome.completed:
        return desired_outcome
    if not desired_outcome.completed:
        simulator.blocked_candidate_evaluations += 1

    candidates = (
        "forward_medium",
        "forward_slow",
        "arc_left",
        "arc_right",
        "yaw_left",
        "yaw_right",
        "backward",
    )
    scored: list[tuple[float, int, PrimitiveSimulation]] = []

    def yaw_escape_penalty(outcome: PrimitiveSimulation) -> float:
        if outcome.primitive not in {"yaw_left", "yaw_right"}:
            return 0.0
        probe_pose = outcome.end_pose
        current_distance = _distance_xy(probe_pose.xy, target_xy)
        for turn_steps in range(15):
            for translating_primitive in (
                "forward_medium",
                "forward_slow",
                "arc_left",
                "arc_right",
                "backward",
            ):
                translated = simulate_primitive(
                    probe_pose,
                    translating_primitive,
                    simulator.registry,
                    simulator.grid,
                    simulator.config,
                )
                if translated.completed and _distance_xy(
                    translated.end_pose.xy,
                    target_xy,
                ) < current_distance - float(simulator.config.minimum_progress_m):
                    return 0.02 * float(turn_steps)
            turned = simulate_primitive(
                probe_pose,
                outcome.primitive,
                simulator.registry,
                simulator.grid,
                simulator.config,
            )
            if not turned.completed:
                break
            probe_pose = turned.end_pose
        return 0.50

    for order, primitive in enumerate(candidates):
        outcome = simulate_primitive(
            simulator.pose,
            primitive,
            simulator.registry,
            simulator.grid,
            simulator.config,
        )
        if not outcome.completed:
            simulator.blocked_candidate_evaluations += 1
            continue
        scored.append(
            (
                _candidate_cost(outcome, target_xy, final_xy, final_yaw)
                + yaw_escape_penalty(outcome),
                order,
                outcome,
            )
        )
    if not scored:
        return None
    if abs(target_bearing) <= 0.35:
        current_distance = _distance_xy(simulator.pose.xy, target_xy)
        progressing = [
            item
            for item in scored
            if item[2].primitive
            in {"forward_medium", "forward_slow", "arc_left", "arc_right"}
            and _distance_xy(item[2].end_pose.xy, target_xy)
            < current_distance - float(simulator.config.minimum_progress_m)
        ]
        if progressing:
            progressing.sort(key=lambda item: (item[0], item[1]))
            return progressing[0][2]
    # When the nominal action is blocked, permit an in-place realignment to
    # beat a backward/forward oscillation. ``drive_to_goal`` independently
    # bounds consecutive non-translating blocks, so a yaw loop cannot run to
    # the full route timeout.
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def drive_to_goal(
    simulator: OracleSimulator,
    goal_xy: tuple[float, float],
    *,
    final_yaw: float | None = None,
    tick_limit: int | None = None,
    goal_tolerance_m: float | None = None,
    final_yaw_tolerance_rad: float | None = None,
) -> tuple[bool, str | None]:
    """Follow a ground-truth A* route with deterministic primitive lookahead."""

    start_tick = simulator.tick
    max_ticks = int(tick_limit or simulator.config.max_goal_ticks)
    route: tuple[tuple[float, float], ...] = ()
    route_age = int(simulator.config.route_replan_ticks)
    no_translation = 0
    while (
        simulator.tick < int(simulator.config.max_ticks)
        and simulator.tick - start_tick < max_ticks
    ):
        distance = _distance_xy(simulator.pose.xy, goal_xy)
        heading_error = (
            0.0
            if final_yaw is None
            else abs(wrap_angle_pi(float(final_yaw) - simulator.pose.yaw))
        )
        effective_goal_tolerance = (
            float(simulator.config.route_goal_tolerance_m)
            if goal_tolerance_m is None
            else float(goal_tolerance_m)
        )
        effective_yaw_tolerance = (
            float(simulator.config.claim_bearing_rad)
            if final_yaw_tolerance_rad is None
            else float(final_yaw_tolerance_rad)
        )
        if (
            final_yaw is not None
            and distance <= effective_goal_tolerance
            and heading_error > effective_yaw_tolerance
        ):
            signed_heading_error = wrap_angle_pi(
                float(final_yaw) - simulator.pose.yaw
            )
            primitive = "yaw_left" if signed_heading_error > 0.0 else "yaw_right"
            outcome = simulate_primitive(
                simulator.pose,
                primitive,
                simulator.registry,
                simulator.grid,
                simulator.config,
            )
            if not outcome.completed:
                return False, "terminal_orientation_blocked"
            simulator.execute(outcome)
            route_age += 1
            no_translation = 0
            continue
        if distance <= effective_goal_tolerance and (
            final_yaw is None or heading_error <= effective_yaw_tolerance
        ):
            return True, None

        if not route or route_age >= int(simulator.config.route_replan_ticks):
            planned = simulator.route_planner.plan(
                simulator.pose.xy,
                goal_xy,
            )
            if planned is None:
                return False, "planner_no_path"
            route = tuple(planned.waypoints_xy) or (goal_xy,)
            route_age = 0

        route = _prune_route(simulator.pose, route)
        target_xy = _path_target(
            simulator.pose,
            route,
            float(simulator.config.route_lookahead_m),
            simulator.grid,
        )
        outcome = _choose_primitive(
            simulator,
            target_xy,
            goal_xy,
            final_yaw,
        )
        if outcome is None:
            simulator.hold()
            no_translation += 1
        else:
            previous = simulator.pose
            simulator.execute(outcome)
            moved = _distance_xy(previous.xy, simulator.pose.xy)
            no_translation = (
                no_translation + 1
                if moved < float(simulator.config.minimum_progress_m)
                else 0
            )
        route_age += 1

        if no_translation >= int(simulator.config.no_progress_limit):
            return False, "follower_no_translation"
    if simulator.tick >= int(simulator.config.max_ticks):
        return False, "tick_budget"
    return False, "follower_goal_timeout"


def _coverage_checkpoints(values: Sequence[float], every: int = 100) -> list[dict[str, float | int]]:
    if not values:
        return []
    ticks = list(range(0, len(values), max(1, int(every))))
    if ticks[-1] != len(values) - 1:
        ticks.append(len(values) - 1)
    return [
        {"tick": int(tick), "fraction": round(float(values[tick]), 6)}
        for tick in ticks
    ]


def run_scene(
    manifest: SceneManifest,
    registry: PrimitiveRegistry,
    config: OracleConfig | None = None,
    geometry_contract: GeometryContract | None = None,
    directional_policy: LoadedDirectionalPolicy | None = None,
) -> dict[str, Any]:
    """Run one positive-control scene and return a JSON-serializable report."""

    geometry_contract = geometry_contract or load_geometry_contract(
        DEFAULT_GEOMETRY_CONTRACT,
        repository_root=REPO_ROOT,
    )
    config = config or OracleConfig.from_geometry_contract(geometry_contract)
    geometry_v2 = geometry_contract.schema == "lewm_go2_generalization_geometry_v2"
    spawn_xy = (float(manifest.spawn.xyz_m[0]), float(manifest.spawn.xyz_m[1]))
    spawn_yaw = _yaw_from_quat_wxyz(manifest.spawn.quat_wxyz)
    beacons = beacons_from_manifest(manifest)
    claim_task_binding = bind_oracle_claim_task(manifest)
    geometry_failures: list[str] = []
    planner_failures: list[str] = []
    follower_failures: list[str] = []
    shared_map_agreement: ExactOccupancyAgreement | None = None
    directional_checker: ManifestDirectionalFootprintFeasibility | None = None

    try:
        if geometry_v2:
            exact_adapter = ExactOccupancyBeliefAdapter(manifest, geometry_contract)
            shared_map_agreement = exact_adapter.load()
            if not shared_map_agreement.online_topology_agrees:
                raise ValueError("shared-map topology differs from its exact online grid")
            if not shared_map_agreement.resolution_is_conservative:
                raise ValueError("shared map admits cells excluded by the 0.05 m reference")
            planning_grid = exact_adapter.online_grid
            visibility_grid = exact_adapter.visibility_grid
            coverage_grid = exact_adapter.online_grid
            start_cell = exact_adapter.spawn_cell
            component = set(
                exact_adapter.belief_map.connected_confirmed_free(start_cell)
            )
            if not component:
                raise ValueError("fixed spawn is not confirmed free in the shared map")
            coverage_start = start_cell
            coverage_component = set(component)
            spawn_snap_m = 0.0
            start_pose = Pose2D(spawn_xy[0], spawn_xy[1], float(spawn_yaw))
            route_planner: RoutePlanner = SharedBeliefMapRoutePlanner(
                exact_adapter.belief_map
            )
            directional_policy = directional_policy or policy_from_geometry_contract(
                geometry_contract,
                repository_root=REPO_ROOT,
            )
            directional_checker = ManifestDirectionalFootprintFeasibility(
                manifest,
                directional_policy.footprint,
            )
        else:
            planning_grid = InflatedOccupancyGrid(
                manifest,
                cell_size_m=float(config.grid_cell_m),
                inflation_m=float(config.planning_inflation_m),
                treat_landmarks_as_obstacles=(
                    geometry_contract.configuration_space.landmarks_are_obstacles
                ),
                treat_distractors_as_obstacles=(
                    geometry_contract.configuration_space.distractors_are_obstacles
                ),
            )
            visibility_grid = InflatedOccupancyGrid(
                manifest,
                cell_size_m=float(config.grid_cell_m),
                inflation_m=0.0,
                treat_landmarks_as_obstacles=False,
                treat_distractors_as_obstacles=(
                    geometry_contract.configuration_space.distractors_are_obstacles
                ),
            )
            coverage_grid = InflatedOccupancyGrid(
                manifest,
                cell_size_m=float(config.coverage_cell_m),
                inflation_m=float(config.planning_inflation_m),
                treat_landmarks_as_obstacles=(
                    geometry_contract.configuration_space.landmarks_are_obstacles
                ),
                treat_distractors_as_obstacles=(
                    geometry_contract.configuration_space.distractors_are_obstacles
                ),
            )
            start_cell, component, spawn_snap_m = reachable_component(
                planning_grid,
                spawn_xy,
                connectivity=int(config.planning_connectivity),
                allow_diagonal_corner_cutting=bool(
                    config.allow_diagonal_corner_cutting
                ),
            )
            coverage_start, coverage_component, _ = reachable_component(
                coverage_grid,
                spawn_xy,
                connectivity=int(config.planning_connectivity),
                allow_diagonal_corner_cutting=bool(
                    config.allow_diagonal_corner_cutting
                ),
            )
            start_xy = planning_grid.to_world(start_cell)
            start_pose = Pose2D(
                float(start_xy[0]),
                float(start_xy[1]),
                float(spawn_yaw),
            )
            route_planner = InflatedGridRoutePlanner(planning_grid, config)
    except (FileNotFoundError, ValueError) as error:
        return {
            "scene_id": manifest.scene_id,
            "geometry_contract_sha256": geometry_contract.sha256,
            "success": False,
            "all_beacons_claimed": False,
            "claimed_count": 0,
            "beacon_count": len(beacons),
            "failure_class": "scene_geometry",
            "geometry_failures": [str(error)],
            "planner_failures": [],
            "follower_failures": [],
            "ticks": 0,
            "claim_completion_tick": None,
            "coverage_completion_tick": None,
            "completion_tick": None,
            "normalized_coverage_auc": 0.0,
            "final_coverage_fraction": 0.0,
            "collisions": 0,
            "stalls": 0,
            "directional_polygon_collision_segments": 0,
        }

    start_feasible, start_reason = _pose_feasible(planning_grid, start_pose, config)
    if not start_feasible:
        geometry_failures.append(f"spawn footprint infeasible: {start_reason}")
    if directional_checker is not None and not directional_checker.is_pose_feasible(
        FootprintPose2D(start_pose.x, start_pose.y, start_pose.yaw)
    ):
        geometry_failures.append("fixed spawn is not observed-max polygon feasible")

    anchors_by_id = {
        beacon.claim_id: claim_anchor_cells(
            beacon,
            component,
            planning_grid,
            visibility_grid,
            config,
        )
        for beacon in beacons
    }
    for beacon in beacons:
        if not anchors_by_id[beacon.claim_id]:
            geometry_failures.append(
                f"{beacon.claim_id}: no connected true-claim anchor"
            )

    simulator = OracleSimulator(
        pose=start_pose,
        registry=registry,
        planning_grid=planning_grid,
        visibility_grid=visibility_grid,
        coverage_grid=coverage_grid,
        coverage_component=coverage_component,
        beacons=beacons,
        config=config,
        route_planner=route_planner,
        directional_checker=directional_checker,
    )
    claim_attempt_diagnostics: dict[str, list[str]] = defaultdict(list)
    controller_claim_attempts: list[dict[str, Any]] = []
    remaining_claim_ids = {beacon.claim_id for beacon in beacons}

    if not geometry_failures:
        while remaining_claim_ids and simulator.tick < int(config.max_ticks):
            choices: list[ClaimAnchor] = []
            for beacon in beacons:
                if beacon.claim_id not in remaining_claim_ids:
                    continue
                anchor = _best_claim_anchor(
                    simulator.pose,
                    beacon,
                    anchors_by_id[beacon.claim_id],
                    planning_grid,
                    config,
                    route_planner,
                )
                if anchor is None:
                    planner_failures.append(
                        f"{beacon.claim_id}: no A* path to claim anchor"
                    )
                    remaining_claim_ids.remove(beacon.claim_id)
                else:
                    choices.append(anchor)
            if not choices:
                break
            anchor = min(
                choices,
                key=lambda item: (
                    item.path_cost_cells,
                    item.beacon.claim_id,
                    item.cell,
                ),
            )
            reached, reason = drive_to_goal(
                simulator,
                anchor.xy,
                final_yaw=anchor.yaw,
                tick_limit=int(config.max_goal_ticks),
                goal_tolerance_m=min(
                    0.12,
                    float(config.route_goal_tolerance_m),
                ),
                final_yaw_tolerance_rad=0.20,
            )
            remaining_claim_ids.remove(anchor.beacon.claim_id)
            if reached:
                reference = object_id_reference(anchor.beacon.claim_id)
                controller_claim_attempts.append(
                    build_claim_attempt(
                        manifest=manifest,
                        trace_id=claim_task_binding.trace_id,
                        episode_id=claim_task_binding.episode_id,
                        event_id=_oracle_claim_attempt_id(
                            trace_id=claim_task_binding.trace_id,
                            episode_id=claim_task_binding.episode_id,
                            scene_id=manifest.scene_id,
                            task_object_id=anchor.beacon.claim_id,
                        ),
                        tick=int(simulator.tick),
                        event_index=len(controller_claim_attempts),
                        requested_target=reference,
                        claimed_target=reference,
                        robot_pose_world_xy_yaw=(
                            float(simulator.pose.x),
                            float(simulator.pose.y),
                            float(simulator.pose.yaw),
                        ),
                        pose_provenance="oracle_full_precision",
                    )
                )
            else:
                terminal_distance = _distance_xy(simulator.pose.xy, anchor.xy)
                terminal_yaw_error = abs(
                    wrap_angle_pi(anchor.yaw - simulator.pose.yaw)
                )
                failure_detail = (
                    f"{reason or 'planned terminal pose not reached'}:"
                    f"distance={terminal_distance:.6f}:"
                    f"yaw_error={terminal_yaw_error:.6f}"
                )
                claim_attempt_diagnostics[anchor.beacon.claim_id].append(
                    failure_detail
                )
                follower_failures.append(
                    f"{anchor.beacon.claim_id}: {failure_detail}"
                )

    for claim_id in sorted(remaining_claim_ids):
        follower_failures.append(f"{claim_id}: claim route exceeded scene tick budget")

    claim_completion_tick: int | None = None

    coverage_plan = build_coverage_plan(
        coverage_grid,
        coverage_component,
        coverage_start,
        config,
    )
    coverage_completion_tick: int | None = None
    coverage_targets_snapped = 0
    coverage_repair_goals = 0
    if not geometry_failures and not planner_failures and not follower_failures:
        for coarse in coverage_plan.walk:
            if simulator.tick >= int(config.max_ticks):
                break
            target_cell = coverage_plan.representatives[coarse]
            if target_cell in simulator.covered:
                if (
                    coverage_completion_tick is None
                    and simulator.coverage_fraction >= float(config.coverage_completion_fraction)
                ):
                    coverage_completion_tick = int(simulator.tick)
                continue
            coarse_target_xy = coverage_grid.to_world(target_cell)
            target_xy = _snap_to_component(
                planning_grid,
                component,
                coarse_target_xy,
                max_radius_m=max(0.15, 2.0 * float(config.coverage_cell_m)),
            )
            if target_xy is None:
                planner_failures.append(
                    f"coverage {coarse}: no nearby planning-component cell"
                )
                break
            if _distance_xy(target_xy, coarse_target_xy) > 1e-9:
                coverage_targets_snapped += 1
            reached, reason = drive_to_goal(
                simulator,
                target_xy,
                tick_limit=min(int(config.max_goal_ticks), int(config.max_ticks) - simulator.tick),
            )
            if not reached:
                if reason == "planner_no_path":
                    planner_failures.append(f"coverage {coarse}: {reason}")
                elif reason != "tick_budget":
                    follower_failures.append(f"coverage {coarse}: {reason}")
                break
            if (
                coverage_completion_tick is None
                and simulator.coverage_fraction >= float(config.coverage_completion_fraction)
            ):
                coverage_completion_tick = int(simulator.tick)
                break
        while (
            not planner_failures
            and not follower_failures
            and simulator.tick < int(config.max_ticks)
            and simulator.coverage_fraction < float(config.coverage_completion_fraction)
        ):
            uncovered = coverage_component - simulator.covered
            if not uncovered:
                break
            repair_cell = min(
                uncovered,
                key=lambda cell: (
                    _distance_xy(simulator.pose.xy, coverage_grid.to_world(cell)),
                    cell,
                ),
            )
            coarse_target_xy = coverage_grid.to_world(repair_cell)
            target_xy = _snap_to_component(
                planning_grid,
                component,
                coarse_target_xy,
                max_radius_m=max(0.15, 2.0 * float(config.coverage_cell_m)),
            )
            if target_xy is None:
                planner_failures.append(
                    f"coverage repair {repair_cell}: no nearby planning-component cell"
                )
                break
            coverage_repair_goals += 1
            reached, reason = drive_to_goal(
                simulator,
                target_xy,
                tick_limit=min(
                    int(config.max_goal_ticks),
                    int(config.max_ticks) - simulator.tick,
                ),
            )
            if not reached:
                if reason == "planner_no_path":
                    planner_failures.append(f"coverage repair {repair_cell}: {reason}")
                elif reason != "tick_budget":
                    follower_failures.append(f"coverage repair {repair_cell}: {reason}")
                break
        if simulator.coverage_fraction >= float(config.coverage_completion_fraction):
            coverage_completion_tick = int(simulator.tick)

    raw_claim_trace, oracle_task_ids, oracle_task_hash = build_claim_trace(
        manifest=manifest,
        trace_id=claim_task_binding.trace_id,
        episode_id=claim_task_binding.episode_id,
        controller_claim_attempts=controller_claim_attempts,
        task_object_ids=claim_task_binding.task_object_ids,
    )
    if (
        oracle_task_ids != claim_task_binding.task_object_ids
        or oracle_task_hash != claim_task_binding.task_object_set_sha256
    ):
        raise AssertionError("completed oracle trace changed the pre-motion task binding")
    canonical_physical_claim_trace = evaluate_physical_claim_trace_oracle_adapter(
        raw_claim_trace,
        manifest,
        oracle_task_ids,
        oracle_task_hash,
    )
    physical_claim_summary = canonical_physical_claim_trace[
        "physical_claim_summary"
    ]
    claimed_beacon_ids = tuple(physical_claim_summary["credited_object_ids"])
    claim_ticks = {
        str(item["object_id"]): int(item["tick"])
        for item in physical_claim_summary["first_credited_by_object"]
    }
    claim_poses = {
        str(item["claimed_target_object_id"]): list(
            item["robot_pose_world_xy_yaw"]
        )
        for item in canonical_physical_claim_trace["physical_claim_evaluations"]
        if item.get("credited") is True
    }
    all_claimed = bool(physical_claim_summary["all_targets_claimed"])
    claim_completion_tick = max(claim_ticks.values()) if all_claimed else None
    coverage_complete = bool(
        simulator.coverage_fraction >= float(config.coverage_completion_fraction)
    )
    strict_directional_safe = bool(
        not geometry_v2
        or (
            simulator.directional_initial_pose_feasible is True
            and simulator.directional_collision_segments == 0
            and simulator.collision_attempts == 0
            and simulator.stalls == 0
        )
    )
    success = bool(
        all_claimed
        and strict_directional_safe
        and (geometry_v2 or coverage_complete)
    )
    if geometry_failures:
        failure_class = "scene_geometry"
    elif geometry_v2 and not strict_directional_safe:
        failure_class = "strict_collision_or_stall"
    elif not all_claimed and planner_failures:
        failure_class = "planner"
    elif not all_claimed and follower_failures:
        failure_class = "follower"
    elif not success:
        failure_class = "budget"
    else:
        failure_class = "success"
    completion_tick = (
        max(int(claim_completion_tick or 0), int(coverage_completion_tick or 0))
        if all_claimed and coverage_complete
        else None
    )
    total_free = int(np.count_nonzero(coverage_grid.free_mask))
    return {
        "scene_id": manifest.scene_id,
        "geometry_contract_sha256": geometry_contract.sha256,
        "success": success,
        "all_beacons_claimed": all_claimed,
        "claimed_count": len(claimed_beacon_ids),
        "beacon_count": len(beacons),
        "claimed_beacon_ids": list(claimed_beacon_ids),
        "claimed_colors": sorted(
            beacon.color
            for beacon in beacons
            if beacon.claim_id in claimed_beacon_ids
        ),
        "claim_ticks": dict(sorted(claim_ticks.items())),
        "claim_poses": dict(sorted(claim_poses.items())),
        "canonical_physical_claim_trace": canonical_physical_claim_trace,
        "failure_class": failure_class,
        "geometry_failures": geometry_failures,
        "planner_failures": planner_failures,
        "follower_failures": follower_failures,
        "ticks": int(simulator.tick),
        "claim_completion_tick": claim_completion_tick,
        "coverage_completion_tick": coverage_completion_tick,
        "completion_tick": completion_tick,
        "normalized_coverage_auc": round(simulator.normalized_coverage_auc(), 6),
        "final_coverage_fraction": round(simulator.coverage_fraction, 6),
        "coverage_completion_fraction": float(config.coverage_completion_fraction),
        "coverage_checkpoints": _coverage_checkpoints(simulator.coverage_fractions),
        "reachable_free_cells": len(coverage_component),
        "oracle_planning_reachable_free_cells": len(component),
        "total_free_cells": total_free,
        "spawn_component_fraction": round(
            len(coverage_component) / max(1, total_free), 6
        ),
        "spawn_snap_m": round(float(spawn_snap_m), 6),
        "coverage_coarse_cells": len(coverage_plan.representatives),
        "coverage_walk_goals": len(coverage_plan.walk),
        "claim_anchor_counts": {
            claim_id: len(candidates)
            for claim_id, candidates in sorted(anchors_by_id.items())
        },
        "claim_attempt_diagnostics": {
            color: values
            for color, values in sorted(claim_attempt_diagnostics.items())
            if values
        },
        "coverage_targets_snapped": int(coverage_targets_snapped),
        "coverage_repair_goals": int(coverage_repair_goals),
        "collisions": int(simulator.collision_attempts),
        "stalls": int(simulator.stalls),
        "strict_directional_safe": strict_directional_safe,
        "directional_polygon_collision_segments": int(
            simulator.directional_collision_segments
        ),
        "directional_polygon_initial_pose_feasible": (
            simulator.directional_initial_pose_feasible
        ),
        "directional_polygon_sweep_samples_evaluated": int(
            simulator.directional_sweep_samples_evaluated
        ),
        "directional_polygon_collision_object_ids": sorted(
            simulator.directional_collision_object_ids
        ),
        "directional_policy": (
            None
            if directional_policy is None
            else directional_policy.provenance_dict(repository_root=REPO_ROOT)
        ),
        "route_planner": route_planner.telemetry(),
        "shared_map_agreement": (
            None
            if shared_map_agreement is None
            else shared_map_agreement.to_dict()
        ),
        "trajectory_pose_count": len(simulator.trajectory),
        "trajectory_sha256": simulator.trajectory_sha256(),
        "actual_yaw_microstep_trajectory": simulator.trajectory,
        "blocked_reasons": dict(sorted(simulator.blocked_reasons.items())),
        "blocked_candidate_evaluations": int(simulator.blocked_candidate_evaluations),
        "swept_probe_risk_blocks": int(simulator.swept_probe_risk_blocks),
        "minimum_swept_probe_clearance_m": (
            None
            if not math.isfinite(simulator.minimum_swept_probe_clearance_m)
            else round(float(simulator.minimum_swept_probe_clearance_m), 6)
        ),
        "primitive_counts": dict(sorted(simulator.primitive_counts.items())),
    }


def _load_manifest(scene_dir: Path) -> SceneManifest:
    payload = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    return parse_scene_manifest_dict(payload)


def _run_indexed_development_scene(
    job: tuple[
        int,
        SceneManifest,
        PrimitiveRegistry,
        OracleConfig,
        GeometryContract,
        LoadedDirectionalPolicy | None,
    ],
) -> tuple[int, dict[str, Any]]:
    """Process-worker boundary; all inputs are already verified in memory."""

    index, manifest, registry, config, geometry_contract, directional_policy = job
    return index, run_scene(
        manifest,
        registry,
        config,
        geometry_contract,
        directional_policy,
    )


def _initialize_single_thread_scene_worker() -> None:
    """Prevent nested BLAS/OpenMP pools inside the bounded scene pool."""

    for name in _CPU_THREAD_CAP_ENV:
        os.environ[name] = "1"


def merge_indexed_scene_reports(
    indexed_reports: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    expected_scene_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge schedule-independent worker results into exact manifest order."""

    merged: dict[int, dict[str, Any]] = {}
    for index, report in indexed_reports:
        if type(index) is not int or not 0 <= index < len(expected_scene_ids):
            raise ValueError("parallel oracle result index is outside the scene panel")
        if index in merged:
            raise ValueError("parallel oracle result index is duplicated")
        if not isinstance(report, Mapping):
            raise ValueError("parallel oracle result is not a mapping")
        if report.get("scene_id") != expected_scene_ids[index]:
            raise ValueError("parallel oracle result scene identity changed")
        merged[index] = dict(report)
    if set(merged) != set(range(len(expected_scene_ids))):
        raise ValueError("parallel oracle result panel is incomplete")
    return [merged[index] for index in range(len(expected_scene_ids))]


def _development_path_guard(path: Path, *, label: str) -> None:
    lowered = "/".join(part.lower() for part in path.parts)
    forbidden = ("sealed", "final_eval", "final-test", "final_test")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{label} must be development-only, got {path}")


def _generic_output_path_guard(path: Path) -> None:
    """Keep the flexible diagnostic CLI away from the authoritative path."""

    resolved = path.resolve(strict=False)
    _development_path_guard(resolved, label="output")
    if resolved == CANONICAL_PHYSICAL_CLAIM_REGRESSION_OUTPUT.resolve(strict=False):
        raise ValueError(
            "canonical physical-claim regression output is reserved for its "
            "fixed authoritative runner"
        )


def _directional_policy_for_suite(
    geometry_contract: GeometryContract,
    preloaded_policy: LoadedDirectionalPolicy | None,
) -> LoadedDirectionalPolicy | None:
    """Use one verified in-memory policy when supplied by an authoritative caller."""

    if geometry_contract.schema != "lewm_go2_generalization_geometry_v2":
        if preloaded_policy is not None:
            raise ValueError(
                "preloaded directional policy is only valid for geometry contract v2"
            )
        return None
    if preloaded_policy is None:
        return policy_from_geometry_contract(
            geometry_contract,
            repository_root=REPO_ROOT,
        )
    if type(preloaded_policy) is not LoadedDirectionalPolicy:
        raise ValueError("preloaded directional policy has the wrong exact type")
    validate_loaded_directional_policy_content(preloaded_policy)

    swept = geometry_contract.swept_footprint
    source = geometry_contract.source_artifacts.get("directional_footprint_policy")
    if not isinstance(source, Mapping):
        raise ValueError("geometry contract is missing directional policy source")
    expected_path = (REPO_ROOT / str(source.get("path", ""))).resolve()
    expected_radius = swept.maximum_vertex_radius_m
    if (
        preloaded_policy.source_path.resolve() != expected_path
        or preloaded_policy.file_sha256 != source.get("sha256")
        or preloaded_policy.content_sha256
        != swept.directional_policy_content_sha256
        or preloaded_policy.policy_id != swept.directional_policy_id
        or preloaded_policy.profile_name != swept.directional_profile
        or expected_radius is None
        or not math.isclose(
            preloaded_policy.footprint.maximum_vertex_radius_m,
            expected_radius,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("preloaded directional policy differs from geometry contract")
    return preloaded_policy


def run_development_suite(
    *,
    scene_corpus: Path,
    split: str,
    family: str | None,
    scene_ids: Sequence[str],
    scene_families: Mapping[str, str] | None = None,
    expected_manifest_sha256: Mapping[str, str] | None = None,
    expected_beacon_counts: Mapping[str, int] | None = None,
    development_manifest: Path | None = None,
    registry: PrimitiveRegistry,
    config: OracleConfig,
    geometry_contract: GeometryContract,
    progress: Callable[[dict[str, Any]], None] | None = None,
    workers: int = 1,
    preloaded_scene_manifests: Mapping[str, SceneManifest] | None = None,
    preloaded_directional_policy: LoadedDirectionalPolicy | None = None,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an exact integer in [1, 8]")
    reports: list[dict[str, Any]] = []
    directional_policy = _directional_policy_for_suite(
        geometry_contract,
        preloaded_directional_policy,
    )
    jobs = []
    if preloaded_scene_manifests is not None:
        if set(preloaded_scene_manifests) != set(scene_ids):
            raise ValueError("preloaded scene-manifest set differs from the requested panel")
        if any(
            type(manifest) is not SceneManifest or manifest.scene_id != scene_id
            for scene_id, manifest in preloaded_scene_manifests.items()
        ):
            raise ValueError("preloaded scene manifests are not exact bound SceneManifest values")
    for index, scene_id in enumerate(scene_ids):
        scene_family = (
            str(scene_families[scene_id])
            if scene_families is not None
            else str(family)
        )
        if preloaded_scene_manifests is None:
            scene_dir = scene_corpus / split / scene_family / scene_id
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"development scene missing: {scene_dir}")
            manifest = _load_manifest(scene_dir)
        else:
            manifest = preloaded_scene_manifests[scene_id]
        if expected_manifest_sha256 is not None:
            actual_sha256 = manifest_sha256(manifest)
            expected_sha256 = str(expected_manifest_sha256[scene_id])
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"development manifest SHA mismatch for {scene_id}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        if (
            expected_beacon_counts is not None
            and len(manifest.landmarks) != int(expected_beacon_counts[scene_id])
        ):
            raise ValueError(
                f"development beacon count mismatch for {scene_id}: expected "
                f"{expected_beacon_counts[scene_id]}, got {len(manifest.landmarks)}"
            )
        jobs.append(
            (
                index,
                manifest,
                registry,
                config,
                geometry_contract,
                directional_policy,
            )
        )
    if workers == 1:
        indexed_reports = [_run_indexed_development_scene(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_single_thread_scene_worker,
        ) as executor:
            futures = [executor.submit(_run_indexed_development_scene, job) for job in jobs]
            indexed_reports = [future.result() for future in as_completed(futures)]
    reports = merge_indexed_scene_reports(
        indexed_reports,
        expected_scene_ids=scene_ids,
    )
    if progress is not None:
        for report in reports:
            progress(report)
    failure_counts: dict[str, int] = defaultdict(int)
    for report in reports:
        failure_counts[str(report["failure_class"])] += 1
    return {
        "schema": "go2_oracle_coverage_positive_control_v1",
        "development_only": True,
        "scene_corpus": str(scene_corpus),
        "split": str(split),
        "family": str(family) if family is not None else "manifest_defined",
        "scene_families": (
            {scene_id: str(scene_families[scene_id]) for scene_id in scene_ids}
            if scene_families is not None
            else {scene_id: str(family) for scene_id in scene_ids}
        ),
        "development_manifest": (
            None if development_manifest is None else str(development_manifest)
        ),
        "scene_ids": list(scene_ids),
        "scene_execution": {
            "kind": "serial" if workers == 1 else "spawn_process",
            "worker_count": workers,
            "threads_per_worker": 1,
            "merge_order": "development_manifest_index",
            "worker_runtime_input_file_access": False,
        },
        "geometry_contract": {
            "schema": geometry_contract.schema,
            "status": geometry_contract.status,
            "sha256": geometry_contract.sha256,
            "source_path": str(geometry_contract.source_path),
            "physical_promotion_ready": geometry_contract.physical_promotion_ready,
        },
        "config": asdict(config),
        "assumptions": {
            "occupancy": "exact static manifest geometry",
            "spawn": (
                "exact fixed manifest pose"
                if directional_policy is not None
                else "fixed manifest spawn, snapped only if inflated cell is occupied"
            ),
            "planning": (
                "OnlineBeliefMap.shortest_path over exact 0.47 m disc-inflated "
                "confirmed-free cells; no unknown traversal"
                if directional_policy is not None
                else "InflatedOccupancyGrid.astar over contract configuration space"
            ),
            "collision_space": (
                "observed-max directional polygon at every actual-yaw microstep"
                if directional_policy is not None
                else "geometry-contract inflated configuration space"
            ),
            "independent_reference": (
                "0.05 m disc grid is resolution/topology audit evidence only"
                if directional_policy is not None
                else "same grid used by the legacy planner"
            ),
            "swept_probe": (
                "forward/rear/half-width action-clearance diagnostic, not a static hull"
            ),
            "motion": "five 0.1 s registry substeps per primitive; translation before yaw update",
            "claim": "metric distance + point-geometry LOS + bearing",
            "coverage": "spawn-connected inflated free cells swept by visit radius",
            "scope": "privileged development positive control; not a deployable result",
            "success_gate": (
                "all task beacons claimed within max_ticks, zero center-grid "
                "collision attempts, zero stalls, zero observed-max polygon collisions"
                if directional_policy is not None
                else "all task beacons claimed and configured coverage threshold reached"
            ),
        },
        "aggregate": {
            "scene_count": len(reports),
            "all_beacons_claimed_scenes": sum(
                1 for report in reports if report["all_beacons_claimed"]
            ),
            "full_4_of_4_claim_scenes": sum(
                1
                for report in reports
                if report["beacon_count"] == 4 and report["all_beacons_claimed"]
            ),
            "positive_control_success_scenes": sum(
                1 for report in reports if report["success"]
            ),
            "claimed_beacons": sum(int(report["claimed_count"]) for report in reports),
            "expected_beacons": sum(int(report["beacon_count"]) for report in reports),
            "mean_normalized_coverage_auc": round(
                float(np.mean([report["normalized_coverage_auc"] for report in reports]))
                if reports
                else 0.0,
                6,
            ),
            "mean_final_coverage_fraction": round(
                float(np.mean([report["final_coverage_fraction"] for report in reports]))
                if reports
                else 0.0,
                6,
            ),
            "median_normalized_coverage_auc": round(
                float(np.median([report["normalized_coverage_auc"] for report in reports]))
                if reports
                else 0.0,
                6,
            ),
            "median_final_coverage_fraction": round(
                float(np.median([report["final_coverage_fraction"] for report in reports]))
                if reports
                else 0.0,
                6,
            ),
            "minimum_final_coverage_fraction": round(
                min(
                    (float(report["final_coverage_fraction"]) for report in reports),
                    default=0.0,
                ),
                6,
            ),
            "collisions": sum(int(report["collisions"]) for report in reports),
            "stalls": sum(int(report["stalls"]) for report in reports),
            "directional_polygon_collision_segments": sum(
                int(report["directional_polygon_collision_segments"])
                for report in reports
            ),
            "strict_directional_safe_scenes": sum(
                bool(report["strict_directional_safe"]) for report in reports
            ),
            "shared_map_routed_scenes": sum(
                report["route_planner"]["source"]
                == "OnlineBeliefMap.shortest_path"
                for report in reports
            ),
            "all_claims_zero_collision_zero_stall_gate_passed": bool(
                reports
                and all(
                    report["all_beacons_claimed"]
                    and int(report["claimed_count"]) == int(report["beacon_count"])
                    and int(report["ticks"]) <= int(config.max_ticks)
                    and int(report["collisions"]) == 0
                    and int(report["stalls"]) == 0
                    and int(report["directional_polygon_collision_segments"]) == 0
                    for report in reports
                )
            ),
            "development_24x4_strict_gate_passed": bool(
                len(reports) == 24
                and all(
                    int(report["beacon_count"]) == 4
                    and int(report["claimed_count"]) == 4
                    and report["all_beacons_claimed"]
                    and int(report["ticks"]) <= int(config.max_ticks)
                    and int(report["collisions"]) == 0
                    and int(report["stalls"]) == 0
                    and int(report["directional_polygon_collision_segments"]) == 0
                    for report in reports
                )
            ),
            "failure_classes": dict(sorted(failure_counts.items())),
        },
        "scenes": reports,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=None,
    )
    parser.add_argument("--development-split", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=None,
        help=(
            "Development protocol JSON; only validation_scenes are consumed. "
            "Never pass the sealed-test manifest."
        ),
    )
    parser.add_argument(
        "--development-scene-list",
        type=Path,
        default=REPO_ROOT / ".generated/go2_corpus_heads/eval_scenes_full.txt",
    )
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=REPO_ROOT / DEFAULT_GEOMETRY_CONTRACT,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=OracleConfig.max_ticks)
    parser.add_argument(
        "--coverage-completion-fraction",
        type=float,
        default=OracleConfig.coverage_completion_fraction,
    )
    parser.add_argument(
        "--coverage-resolution-m",
        type=float,
        default=OracleConfig.coverage_resolution_m,
    )
    parser.add_argument(
        "--route-lookahead-m",
        type=float,
        default=OracleConfig.route_lookahead_m,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is not None:
        _generic_output_path_guard(args.output)
    geometry_contract = load_geometry_contract(
        args.geometry_contract,
        repository_root=REPO_ROOT,
    )
    scene_families: dict[str, str] | None = None
    expected_manifest_sha256: dict[str, str] | None = None
    expected_beacon_counts: dict[str, int] | None = None
    development_manifest: Path | None = None
    if args.development_manifest is not None:
        development_manifest = args.development_manifest.resolve()
        _development_path_guard(development_manifest, label="development manifest")
        protocol = json.loads(development_manifest.read_text(encoding="utf-8"))
        if protocol.get("schema") != "lewm_navigation_development_manifest_v0":
            raise ValueError(
                f"unsupported development manifest schema: {protocol.get('schema')!r}"
            )
        if str(protocol.get("geometry_contract_sha256")) != geometry_contract.sha256:
            raise ValueError("development manifest geometry-contract SHA mismatch")
        validation_records = list(protocol.get("validation_scenes", ()))
        if args.family is not None:
            validation_records = [
                record
                for record in validation_records
                if str(record.get("family")) == str(args.family)
            ]
        if not validation_records:
            raise ValueError("development manifest has no selected validation_scenes")
        if any(
            not bool(record.get("fully_reachable"))
            or bool(str(record.get("failure_reason", "")))
            for record in validation_records
        ):
            raise ValueError("development validation_scenes include an invalid scene")
        allowed = [str(record["scene_id"]) for record in validation_records]
        if len(set(allowed)) != len(allowed):
            raise ValueError("development validation_scenes contain duplicate scene ids")
        scene_families = {
            str(record["scene_id"]): str(record["family"])
            for record in validation_records
        }
        expected_manifest_sha256 = {
            str(record["scene_id"]): str(record["manifest_sha256"])
            for record in validation_records
        }
        expected_beacon_counts = {
            str(record["scene_id"]): int(record["beacon_count"])
            for record in validation_records
        }
        scene_corpus = (
            args.scene_corpus.resolve()
            if args.scene_corpus is not None
            else REPO_ROOT
            / ".generated/scene_corpus"
            / development_manifest.parent.name
        )
        split = str(args.development_split or "development")
        family = None
    else:
        scene_list = args.development_scene_list.resolve()
        _development_path_guard(scene_list, label="scene list")
        allowed = [
            line.strip()
            for line in scene_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        scene_corpus = (
            args.scene_corpus.resolve()
            if args.scene_corpus is not None
            else LEGACY_DEVELOPMENT_CORPUS
        )
        split = str(args.development_split or "test_id")
        family = str(args.family or "medium_enclosed_maze")
    _development_path_guard(scene_corpus, label="scene corpus")
    requested = list(args.scene_id) if args.scene_id else allowed
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise SystemExit(
            "scene ids are not in the development allow-list: " + ", ".join(unknown)
        )
    config = OracleConfig.from_geometry_contract(
        geometry_contract,
        max_ticks=int(args.max_ticks),
        coverage_completion_fraction=float(args.coverage_completion_fraction),
        coverage_resolution_m=float(args.coverage_resolution_m),
        route_lookahead_m=float(args.route_lookahead_m),
    )
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())

    def print_progress(report: dict[str, Any]) -> None:
        print(
            f"{report['scene_id']}: claims={report['claimed_count']}/{report['beacon_count']} "
            f"coverage={report['final_coverage_fraction']:.3f} "
            f"auc={report['normalized_coverage_auc']:.3f} ticks={report['ticks']} "
            f"collisions={report['collisions']} stalls={report['stalls']} "
            f"polygon_collisions={report['directional_polygon_collision_segments']} "
            f"class={report['failure_class']}",
            file=sys.stderr,
            flush=True,
        )

    report = run_development_suite(
        scene_corpus=scene_corpus,
        split=split,
        family=family,
        scene_ids=requested,
        scene_families=scene_families,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_beacon_counts=expected_beacon_counts,
        development_manifest=development_manifest,
        registry=registry,
        config=config,
        geometry_contract=geometry_contract,
        progress=print_progress,
    )
    provenance_inputs = {
        "oracle_source": Path(__file__).resolve(),
        "primitive_registry": args.primitive_registry.resolve(),
        "online_belief_map_source": (
            REPO_ROOT / "lewm/planning/online_belief_map.py"
        ),
        "exact_occupancy_adapter_source": (
            REPO_ROOT / "lewm/planning/exact_occupancy_belief_adapter.py"
        ),
        "directional_footprint_source": (
            REPO_ROOT / "lewm/planning/oriented_footprint.py"
        ),
        "physical_eligibility_source": (
            REPO_ROOT / "lewm/benchmarks/go2_physical_eligibility.py"
        ),
    }
    if development_manifest is not None:
        provenance_inputs["development_manifest"] = development_manifest
    else:
        provenance_inputs["development_scene_list"] = (
            args.development_scene_list.resolve()
        )
    materialization_report = scene_corpus / "materialization_both.json"
    if materialization_report.is_file():
        provenance_inputs["scene_materialization_report"] = materialization_report
    invocation = [
        sys.executable,
        "-m",
        "lewm.benchmarks.go2_oracle_positive_control",
        *(list(sys.argv[1:]) if argv is None else list(argv)),
    ]
    report["experiment_manifest"] = build_experiment_manifest(
        experiment_id=(
            f"go2_oracle_positive_control_{split}_"
            f"{geometry_contract.sha256[:16]}"
        ),
        repository_root=REPO_ROOT,
        inputs=provenance_inputs,
        config=asdict(config),
        run_command=shlex.join(invocation),
        scene_splits={"development": requested},
        geometry_contract=geometry_contract.source_path,
        runtime_contract={
            "scope": "privileged_development_positive_control",
            "planner": "OnlineBeliefMap.shortest_path",
            "runtime_inputs": [
                "exact_static_manifest_geometry",
                "exact_fixed_spawn_pose",
                "exact_beacon_geometry",
            ],
            "unknown_traversal": False,
            "sealed_artifacts_opened": False,
        },
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote development positive control: {args.output}", file=sys.stderr)
    if (
        geometry_contract.schema == "lewm_go2_generalization_geometry_v2"
        and not args.scene_id
    ):
        return (
            0
            if report["aggregate"]["development_24x4_strict_gate_passed"]
            else 1
        )
    return (
        0
        if report["aggregate"]["positive_control_success_scenes"]
        == len(requested)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
