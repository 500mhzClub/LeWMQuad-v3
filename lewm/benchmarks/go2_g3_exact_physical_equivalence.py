"""Development-only G3 exact-physical morphology equivalence audit.

The audit keeps three questions separate:

* does ``RevisionedPhysicalMemory`` reproduce an independent implementation of
  the frozen 89/69 discrete morphology;
* is every admitted configuration-FREE cell safe against analytic rotated-box
  geometry; and
* does the conservative component retain a valid physical claim endpoint for
  every development beacon.

It intentionally reports the historical strict binary-grid equality as a
separate condition. A conservative/usability result cannot silently amend that
pre-registered condition.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from lewm.planning.geometry_contract import GeometryContract
from lewm.planning.revisioned_physical_configuration_memory import (
    ConfigurationMorphology,
    ConfigurationPlanner,
    ConfigurationSnapshot,
    EvidenceAuthority,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
)
from lewm.planning.zero_inflation_exact_physical_adapter_v1 import (
    ZeroInflationExactPhysicalAdapterV1,
    exact_physical_cells_content_sha256,
)
from lewm_worlds.manifest import BoxObject, SceneManifest


Cell = tuple[int, int]
_EPS = 1e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _collision_boxes(
    manifest: SceneManifest,
    *,
    landmarks_are_obstacles: bool,
    distractors_are_obstacles: bool,
) -> tuple[BoxObject, ...]:
    boxes: list[BoxObject] = [*manifest.walls, *manifest.obstacles]
    if landmarks_are_obstacles:
        boxes.extend(manifest.landmarks)
    if distractors_are_obstacles and manifest.visual_randomization is not None:
        boxes.extend(manifest.visual_randomization.distractor_objects)
    return tuple(boxes)


def registered_lattice(
    manifest: SceneManifest,
    geometry: GeometryContract,
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Return the canonical online-grid origin and shape without rasterizing."""

    config = geometry.configuration_space
    cell = float(config.online_cell_size_m)
    radius = float(config.body_inflation_radius_m)
    if not math.isclose(cell, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("G3 exact equivalence requires the registered 0.10 m lattice")
    if not math.isclose(radius, 0.47, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("G3 exact equivalence requires the registered 0.47 m radius")
    if int(config.connectivity) != 4 or bool(config.allow_diagonal_corner_cutting):
        raise ValueError("G3 exact equivalence requires frozen four-connected planning")
    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    pad = radius + cell
    origin = (float(x_lo) - pad, float(y_lo) - pad)
    shape = (
        int(math.ceil((float(x_hi) - float(x_lo) + 2.0 * pad) / cell)),
        int(math.ceil((float(y_hi) - float(y_lo) + 2.0 * pad) / cell)),
    )
    return origin, shape


def lattice_centers(
    origin_xy_m: Sequence[float],
    shape: Sequence[int],
    *,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(origin_xy_m) != 2 or len(shape) != 2:
        raise ValueError("lattice origin and shape must be two-dimensional")
    nx, ny = int(shape[0]), int(shape[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("lattice shape must be positive")
    cell = float(cell_size_m)
    if not math.isfinite(cell) or cell <= 0.0:
        raise ValueError("cell_size_m must be positive and finite")
    xs = float(origin_xy_m[0]) + (np.arange(nx, dtype=np.float64) + 0.5) * cell
    ys = float(origin_xy_m[1]) + (np.arange(ny, dtype=np.float64) + 0.5) * cell
    return np.meshgrid(xs, ys, indexing="ij")


def _closed_square_intersects_box_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    half_cell_m: float,
    box: BoxObject,
) -> np.ndarray:
    """Vectorized closed axis-aligned square versus oriented-rectangle SAT."""

    center_x, center_y = float(box.center_xyz_m[0]), float(box.center_xyz_m[1])
    half_x = 0.5 * float(box.size_xyz_m[0])
    half_y = 0.5 * float(box.size_xyz_m[1])
    yaw = float(box.yaw_rad)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    abs_cosine, abs_sine = abs(cosine), abs(sine)
    dx = grid_x - center_x
    dy = grid_y - center_y
    # World x/y axes.
    overlap = (
        (np.abs(dx) <= half_cell_m + half_x * abs_cosine + half_y * abs_sine + _EPS)
        & (np.abs(dy) <= half_cell_m + half_x * abs_sine + half_y * abs_cosine + _EPS)
    )
    # The oriented box's local axes.
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    square_radius = half_cell_m * (abs_cosine + abs_sine)
    overlap &= np.abs(local_x) <= half_x + square_radius + _EPS
    overlap &= np.abs(local_y) <= half_y + square_radius + _EPS
    return overlap


def exact_closed_square_physical_labels(
    manifest: SceneManifest,
    geometry: GeometryContract,
    *,
    origin_xy_m: Sequence[float],
    shape: Sequence[int],
) -> np.ndarray:
    """Rasterize full physical cell squares; centre sampling is forbidden."""

    cell = float(geometry.configuration_space.online_cell_size_m)
    grid_x, grid_y = lattice_centers(origin_xy_m, shape, cell_size_m=cell)
    half = 0.5 * cell
    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    occupied = (
        (grid_x - half < float(x_lo) - _EPS)
        | (grid_x + half > float(x_hi) + _EPS)
        | (grid_y - half < float(y_lo) - _EPS)
        | (grid_y + half > float(y_hi) + _EPS)
    )
    for box in _collision_boxes(
        manifest,
        landmarks_are_obstacles=geometry.configuration_space.landmarks_are_obstacles,
        distractors_are_obstacles=geometry.configuration_space.distractors_are_obstacles,
    ):
        occupied |= _closed_square_intersects_box_mask(
            grid_x,
            grid_y,
            half_cell_m=half,
            box=box,
        )
    labels = np.full(grid_x.shape, int(PhysicalLabel.FREE), dtype=np.uint8)
    labels[occupied] = int(PhysicalLabel.OCCUPIED)
    return labels


def analytic_disc_free_mask(
    manifest: SceneManifest,
    geometry: GeometryContract,
    *,
    origin_xy_m: Sequence[float],
    shape: Sequence[int],
) -> np.ndarray:
    """Independent point-centred disc clearance against exact oriented boxes."""

    cell = float(geometry.configuration_space.online_cell_size_m)
    radius = float(geometry.configuration_space.body_inflation_radius_m)
    grid_x, grid_y = lattice_centers(origin_xy_m, shape, cell_size_m=cell)
    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    free = (
        (grid_x >= float(x_lo) + radius - _EPS)
        & (grid_x <= float(x_hi) - radius + _EPS)
        & (grid_y >= float(y_lo) + radius - _EPS)
        & (grid_y <= float(y_hi) - radius + _EPS)
    )
    for box in _collision_boxes(
        manifest,
        landmarks_are_obstacles=geometry.configuration_space.landmarks_are_obstacles,
        distractors_are_obstacles=geometry.configuration_space.distractors_are_obstacles,
    ):
        center_x, center_y = float(box.center_xyz_m[0]), float(box.center_xyz_m[1])
        half_x = 0.5 * float(box.size_xyz_m[0])
        half_y = 0.5 * float(box.size_xyz_m[1])
        cosine, sine = math.cos(float(box.yaw_rad)), math.sin(float(box.yaw_rad))
        dx, dy = grid_x - center_x, grid_y - center_y
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        outside_x = np.maximum(np.abs(local_x) - half_x, 0.0)
        outside_y = np.maximum(np.abs(local_y) - half_y, 0.0)
        distance = np.hypot(outside_x, outside_y)
        free &= distance >= radius - _EPS
    return free


def _support_view(
    labels: np.ndarray,
    offset: Cell,
    *,
    fill: int,
) -> np.ndarray:
    """At output (x,y), return labels[x+dx,y+dy] without wraparound."""

    dx, dy = int(offset[0]), int(offset[1])
    result = np.full(labels.shape, int(fill), dtype=labels.dtype)
    nx, ny = labels.shape
    dst_x0, dst_x1 = max(0, -dx), min(nx, nx - dx)
    dst_y0, dst_y1 = max(0, -dy), min(ny, ny - dy)
    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        result[dst_x0:dst_x1, dst_y0:dst_y1] = labels[
            dst_x0 + dx : dst_x1 + dx,
            dst_y0 + dy : dst_y1 + dy,
        ]
    return result


def independently_derived_morphology_supports(
    *,
    physical_cell_size_m: float,
    footprint_radius_m: float,
) -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Derive the registered kernels without production morphology offsets."""

    cell = float(physical_cell_size_m)
    radius = float(footprint_radius_m)
    if not math.isfinite(cell) or not math.isfinite(radius) or cell <= 0 or radius <= 0:
        raise ValueError("independent morphology dimensions must be positive and finite")
    span = int(math.ceil(radius / cell)) + 2
    free_support: list[Cell] = []
    occupied_support: list[Cell] = []
    half_cell = 0.5 * cell
    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):
            nearest_x = max(abs(float(dx) * cell) - half_cell, 0.0)
            nearest_y = max(abs(float(dy) * cell) - half_cell, 0.0)
            if math.hypot(nearest_x, nearest_y) <= radius + _EPS:
                free_support.append((dx, dy))
            if math.hypot(float(dx) * cell, float(dy) * cell) <= radius + _EPS:
                occupied_support.append((dx, dy))
    free = tuple(sorted(free_support))
    occupied = tuple(sorted(occupied_support))
    if (
        math.isclose(cell, 0.10, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(radius, 0.47, rel_tol=0.0, abs_tol=1e-12)
        and (len(free) != 89 or len(occupied) != 69)
    ):
        raise AssertionError("independent registered morphology is not 89/69")
    return free, occupied


def independent_configuration_labels(
    physical_labels: np.ndarray,
    morphology: ConfigurationMorphology,
) -> np.ndarray:
    """Independent array implementation of occupied-first 89/69 morphology."""

    labels = np.asarray(physical_labels)
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError("physical_labels must be a non-empty 2D array")
    if not np.isin(labels, tuple(int(value) for value in PhysicalLabel)).all():
        raise ValueError("physical_labels contains an unsupported class")
    free_support, occupied_support = independently_derived_morphology_supports(
        physical_cell_size_m=morphology.physical_cell_size_m,
        footprint_radius_m=morphology.footprint_radius_m,
    )
    if free_support != tuple(morphology.free_support_offsets):
        raise AssertionError("production free-support kernel differs from independent derivation")
    if occupied_support != tuple(morphology.occupied_support_offsets):
        raise AssertionError(
            "production occupied-support kernel differs from independent derivation"
        )
    all_free = np.ones(labels.shape, dtype=np.bool_)
    any_occupied = np.zeros(labels.shape, dtype=np.bool_)
    for offset in free_support:
        support = _support_view(
            labels,
            offset,
            fill=int(PhysicalLabel.OCCUPIED),
        )
        all_free &= support == int(PhysicalLabel.FREE)
    for offset in occupied_support:
        support = _support_view(
            labels,
            offset,
            fill=int(PhysicalLabel.OCCUPIED),
        )
        any_occupied |= support == int(PhysicalLabel.OCCUPIED)
    result = np.full(labels.shape, int(PhysicalLabel.UNKNOWN), dtype=np.uint8)
    result[all_free] = int(PhysicalLabel.FREE)
    result[any_occupied] = int(PhysicalLabel.OCCUPIED)
    return result


def _array_cells(mask: np.ndarray) -> frozenset[Cell]:
    return frozenset((int(x), int(y)) for x, y in np.argwhere(mask))


def _component(mask: np.ndarray, start: Cell) -> frozenset[Cell]:
    nx, ny = mask.shape
    if not (0 <= start[0] < nx and 0 <= start[1] < ny and bool(mask[start])):
        return frozenset()
    queue: deque[Cell] = deque([start])
    seen: set[Cell] = {start}
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)):
            if (
                0 <= neighbor[0] < nx
                and 0 <= neighbor[1] < ny
                and bool(mask[neighbor])
                and neighbor not in seen
            ):
                seen.add(neighbor)
                queue.append(neighbor)
    return frozenset(seen)


def _component_distances(mask: np.ndarray, start: Cell) -> dict[Cell, int]:
    component = _component(mask, start)
    if not component:
        return {}
    queue: deque[Cell] = deque([start])
    distances: dict[Cell, int] = {start: 0}
    while queue:
        x, y = queue.popleft()
        next_distance = distances[(x, y)] + 1
        for neighbor in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)):
            if neighbor in component and neighbor not in distances:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def _astar_distance_checks(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    independent_free: np.ndarray,
    start: Cell,
    *,
    maximum_probes: int = 8,
) -> tuple[int, int]:
    distances = _component_distances(independent_free, start)
    if len(distances) <= 1:
        return (0, 1)
    ordered = sorted(distances, key=lambda cell: (distances[cell], cell))
    indices = {
        int(round((len(ordered) - 1) * rank / maximum_probes))
        for rank in range(1, maximum_probes + 1)
    }
    endpoints = tuple(ordered[index] for index in sorted(indices) if index > 0)
    mismatches = 0
    for endpoint in endpoints:
        try:
            path = planner.astar(snapshot, start, endpoint)
            if path is None:
                mismatches += 1
                continue
            planner.validate_path(snapshot, path)
            if (
                path.cells[0] != start
                or path.cells[-1] != endpoint
                or not math.isclose(
                    float(path.cost),
                    float(distances[endpoint]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                mismatches += 1
        except Exception:
            mismatches += 1
    return (len(endpoints), mismatches)


def _physical_mapping(labels: np.ndarray) -> dict[Cell, PhysicalLabel]:
    return {
        (int(x), int(y)): PhysicalLabel(int(labels[x, y]))
        for x, y in np.ndindex(labels.shape)
    }


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _build_exact_snapshot(
    manifest: SceneManifest,
    geometry: GeometryContract,
    physical_labels: np.ndarray,
    *,
    origin_xy_m: tuple[float, float],
) -> tuple[RevisionedPhysicalMemory, object, ConfigurationPlanner]:
    frame = MapFrameIdentity(
        session_id=str(manifest.scene_id),
        origin_xy_m=origin_xy_m,
        cell_size_m=float(geometry.configuration_space.online_cell_size_m),
    )
    camera_sha256 = _sha_text("g3-exact-equivalence-camera")
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=frame,
            planning_connectivity=int(geometry.configuration_space.connectivity),
            allow_diagonal_corner_cutting=bool(
                geometry.configuration_space.allow_diagonal_corner_cutting
            ),
            expected_camera_transform_sha256=camera_sha256,
        )
    )
    mapping = _physical_mapping(physical_labels)
    observation = ObservationIdentity(
        observation_id=f"exact:{manifest.scene_id}",
        payload_sha256=exact_physical_cells_content_sha256(mapping),
        producer_sha256=_sha_text("g3-exact-equivalence-adapter"),
        authority=EvidenceAuthority.EXACT_PHYSICAL,
    )
    pose = PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id=frame.frame_id,
        mean_xy_yaw=(0.0, 0.0, 0.0),
        covariance_xy_yaw=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        timestamp_ns=0,
        synchronization_id=f"exact:{manifest.scene_id}",
        camera_transform_sha256=camera_sha256,
    )
    ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
        mapping,
        observation=observation,
        pose=pose,
        label_inflation_radius_m=0.0,
    )
    morphology = ConfigurationMorphology()
    domain = frozenset(mapping)
    snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells=domain,
    )
    return memory, snapshot, ConfigurationPlanner(memory, morphology)


def _closed_segment_intersects_rotated_box(
    start_xy_m: Sequence[float],
    end_xy_m: Sequence[float],
    box: BoxObject,
) -> bool:
    """Exact closed 2D segment versus closed oriented rectangle."""

    if len(start_xy_m) != 2 or len(end_xy_m) != 2:
        raise ValueError("line-of-sight endpoints must be two-dimensional")
    center_x, center_y = float(box.center_xyz_m[0]), float(box.center_xyz_m[1])
    half_x = 0.5 * float(box.size_xyz_m[0])
    half_y = 0.5 * float(box.size_xyz_m[1])
    cosine = math.cos(float(box.yaw_rad))
    sine = math.sin(float(box.yaw_rad))

    def local(point: Sequence[float]) -> tuple[float, float]:
        dx = float(point[0]) - center_x
        dy = float(point[1]) - center_y
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    start_x, start_y = local(start_xy_m)
    end_x, end_y = local(end_xy_m)
    direction_x, direction_y = end_x - start_x, end_y - start_y
    lower, upper = 0.0, 1.0
    for origin, direction, half_extent in (
        (start_x, direction_x, half_x),
        (start_y, direction_y, half_y),
    ):
        if abs(direction) <= _EPS:
            if origin < -half_extent - _EPS or origin > half_extent + _EPS:
                return False
            continue
        first = (-half_extent - origin) / direction
        second = (half_extent - origin) / direction
        entry, exit_ = min(first, second), max(first, second)
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper + _EPS:
            return False
    return lower <= 1.0 + _EPS and upper >= -_EPS


def _exact_line_of_sight_clear(
    manifest: SceneManifest,
    geometry: GeometryContract,
    start_xy_m: Sequence[float],
    end_xy_m: Sequence[float],
) -> bool:
    blockers = _collision_boxes(
        manifest,
        landmarks_are_obstacles=False,
        distractors_are_obstacles=bool(
            geometry.configuration_space.distractors_are_obstacles
        ),
    )
    return not any(
        _closed_segment_intersects_rotated_box(start_xy_m, end_xy_m, box)
        for box in blockers
    )


def _claim_endpoint_count(
    manifest: SceneManifest,
    geometry: GeometryContract,
    component: Iterable[Cell],
    *,
    origin_xy_m: tuple[float, float],
) -> int:
    cell_size = float(geometry.configuration_space.online_cell_size_m)
    points = tuple(
        (
            origin_xy_m[0] + (cell[0] + 0.5) * cell_size,
            origin_xy_m[1] + (cell[1] + 0.5) * cell_size,
        )
        for cell in component
    )
    count = 0
    for landmark in manifest.landmarks:
        target = (
            float(landmark.center_xyz_m[0]),
            float(landmark.center_xyz_m[1]),
        )
        if any(
            math.dist(point, target)
            <= float(geometry.visibility_and_claim.claim_radius_m) + _EPS
            and (
                not geometry.visibility_and_claim.require_line_of_sight_for_scene_validity
                or _exact_line_of_sight_clear(manifest, geometry, point, target)
            )
            for point in points
        ):
            count += 1
    return count


@dataclass(frozen=True)
class G3ExactSceneResult:
    scene_id: str
    family: str
    lattice_origin_xy_m: tuple[float, float]
    lattice_shape: tuple[int, int]
    physical_free_cells: int
    physical_occupied_cells: int
    snapshot_free_cells: int
    snapshot_occupied_cells: int
    snapshot_unknown_cells: int
    independent_label_mismatch_cells: int
    analytic_free_cells: int
    unsafe_free_cells: int
    conservative_false_reject_cells: int
    strict_binary_label_mismatch_cells: int
    snapshot_component_cells: int
    independent_component_cells: int
    component_mismatch_cells: int
    astar_probe_count: int
    astar_mismatch_count: int
    canonical_component_cells: int
    canonical_component_false_reject_cells: int
    claim_endpoints_retained: int
    beacon_count: int
    exact_sim_tainted: bool

    @property
    def discrete_equivalence_pass(self) -> bool:
        return bool(
            self.independent_label_mismatch_cells == 0
            and self.component_mismatch_cells == 0
            and self.astar_probe_count > 0
            and self.astar_mismatch_count == 0
            and self.exact_sim_tainted
        )

    @property
    def conservative_safety_pass(self) -> bool:
        return self.unsafe_free_cells == 0

    @property
    def claim_endpoint_pass(self) -> bool:
        return self.claim_endpoints_retained == self.beacon_count

    @property
    def legacy_strict_binary_equivalence_pass(self) -> bool:
        return self.strict_binary_label_mismatch_cells == 0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "discrete_equivalence_pass": self.discrete_equivalence_pass,
            "conservative_safety_pass": self.conservative_safety_pass,
            "claim_endpoint_pass": self.claim_endpoint_pass,
            "legacy_strict_binary_equivalence_pass": (
                self.legacy_strict_binary_equivalence_pass
            ),
        }


def evaluate_exact_scene(
    manifest: SceneManifest,
    geometry: GeometryContract,
) -> G3ExactSceneResult:
    origin, shape = registered_lattice(manifest, geometry)
    physical = exact_closed_square_physical_labels(
        manifest,
        geometry,
        origin_xy_m=origin,
        shape=shape,
    )
    independent = independent_configuration_labels(
        physical,
        ConfigurationMorphology(),
    )
    analytic_free = analytic_disc_free_mask(
        manifest,
        geometry,
        origin_xy_m=origin,
        shape=shape,
    )
    memory, snapshot, planner = _build_exact_snapshot(
        manifest,
        geometry,
        physical,
        origin_xy_m=origin,
    )
    snapshot_labels = np.full(shape, int(PhysicalLabel.UNKNOWN), dtype=np.uint8)
    for cell in snapshot.free_cells:
        snapshot_labels[cell] = int(PhysicalLabel.FREE)
    for cell in snapshot.occupied_cells:
        snapshot_labels[cell] = int(PhysicalLabel.OCCUPIED)
    spawn = memory.map_frame.world_to_cell(
        (float(manifest.spawn.xyz_m[0]), float(manifest.spawn.xyz_m[1]))
    )
    snapshot_component = frozenset(
        planner.connected_component(snapshot, spawn).cells
    )
    independent_component = _component(
        independent == int(PhysicalLabel.FREE),
        spawn,
    )
    canonical_component = _component(analytic_free, spawn)
    astar_probe_count, astar_mismatch_count = _astar_distance_checks(
        planner,
        snapshot,
        independent == int(PhysicalLabel.FREE),
        spawn,
    )
    snapshot_free = snapshot_labels == int(PhysicalLabel.FREE)
    strict_binary = np.where(
        analytic_free,
        int(PhysicalLabel.FREE),
        int(PhysicalLabel.OCCUPIED),
    ).astype(np.uint8)
    return G3ExactSceneResult(
        scene_id=str(manifest.scene_id),
        family=str(manifest.family),
        lattice_origin_xy_m=origin,
        lattice_shape=shape,
        physical_free_cells=int(np.count_nonzero(physical == int(PhysicalLabel.FREE))),
        physical_occupied_cells=int(
            np.count_nonzero(physical == int(PhysicalLabel.OCCUPIED))
        ),
        snapshot_free_cells=len(snapshot.free_cells),
        snapshot_occupied_cells=len(snapshot.occupied_cells),
        snapshot_unknown_cells=len(snapshot.unknown_cells),
        independent_label_mismatch_cells=int(np.count_nonzero(snapshot_labels != independent)),
        analytic_free_cells=int(np.count_nonzero(analytic_free)),
        unsafe_free_cells=int(np.count_nonzero(snapshot_free & ~analytic_free)),
        conservative_false_reject_cells=int(np.count_nonzero(analytic_free & ~snapshot_free)),
        strict_binary_label_mismatch_cells=int(
            np.count_nonzero(snapshot_labels != strict_binary)
        ),
        snapshot_component_cells=len(snapshot_component),
        independent_component_cells=len(independent_component),
        component_mismatch_cells=len(snapshot_component ^ independent_component),
        astar_probe_count=astar_probe_count,
        astar_mismatch_count=astar_mismatch_count,
        canonical_component_cells=len(canonical_component),
        canonical_component_false_reject_cells=len(canonical_component - snapshot_component),
        claim_endpoints_retained=_claim_endpoint_count(
            manifest,
            geometry,
            snapshot_component,
            origin_xy_m=origin,
        ),
        beacon_count=len(manifest.landmarks),
        exact_sim_tainted=bool(snapshot.exact_sim_tainted),
    )


def summarize_exact_scenes(
    scene_results: Sequence[G3ExactSceneResult],
    *,
    source_bindings: Mapping[str, str],
) -> dict[str, object]:
    rows = tuple(scene_results)
    if not rows:
        raise ValueError("exact-equivalence summary requires at least one scene")
    if len({row.scene_id for row in rows}) != len(rows):
        raise ValueError("exact-equivalence scene identities must be unique")
    ordered = tuple(sorted(rows, key=lambda row: row.scene_id))
    scene_payloads = [row.to_dict() for row in ordered]
    result: dict[str, object] = {
        "schema": "lewm_go2_g3_exact_physical_equivalence_candidate_v1",
        "status": "candidate_requires_independent_review_and_contract_decision",
        "source_bindings": dict(sorted(source_bindings.items())),
        "scene_count": len(ordered),
        "beacon_count": sum(row.beacon_count for row in ordered),
        "claim_endpoints_retained": sum(
            row.claim_endpoints_retained for row in ordered
        ),
        "discrete_equivalence_scene_count": sum(
            row.discrete_equivalence_pass for row in ordered
        ),
        "conservative_safety_scene_count": sum(
            row.conservative_safety_pass for row in ordered
        ),
        "claim_endpoint_scene_count": sum(row.claim_endpoint_pass for row in ordered),
        "legacy_strict_binary_equivalence_scene_count": sum(
            row.legacy_strict_binary_equivalence_pass for row in ordered
        ),
        "unsafe_free_cells": sum(row.unsafe_free_cells for row in ordered),
        "conservative_false_reject_cells": sum(
            row.conservative_false_reject_cells for row in ordered
        ),
        "strict_binary_label_mismatch_cells": sum(
            row.strict_binary_label_mismatch_cells for row in ordered
        ),
        "scenes": scene_payloads,
    }
    result["candidate_conservative_equivalence_pass"] = bool(
        len(ordered) == 24
        and result["beacon_count"] == 96
        and result["claim_endpoints_retained"] == 96
        and result["discrete_equivalence_scene_count"] == len(ordered)
        and result["conservative_safety_scene_count"] == len(ordered)
        and result["claim_endpoint_scene_count"] == len(ordered)
    )
    result["legacy_strict_binary_equivalence_pass"] = bool(
        len(ordered) == 24
        and result["beacon_count"] == 96
        and result["legacy_strict_binary_equivalence_scene_count"] == len(ordered)
    )
    result["content_sha256"] = _sha256(result)
    return result


__all__ = [
    "G3ExactSceneResult",
    "analytic_disc_free_mask",
    "evaluate_exact_scene",
    "exact_closed_square_physical_labels",
    "independent_configuration_labels",
    "independently_derived_morphology_supports",
    "lattice_centers",
    "registered_lattice",
    "summarize_exact_scenes",
]
