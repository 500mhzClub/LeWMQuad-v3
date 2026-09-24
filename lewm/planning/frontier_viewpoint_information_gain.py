"""Revision-bound G4 viewpoint candidates and deterministic information gain.

The module deliberately separates four things which are easy to conflate:

* physical occupancy evidence, derived only from ``RevisionedPhysicalMemory``;
* visual sweep history, owned by ``PhysicalViewStateIssuer``;
* safe configuration-space routes, issued by ``ConfigurationPlanner``; and
* a frozen, conservative camera-ground observation model used for ranking.

Unknown or out-of-domain cells terminate visibility rays and never become route
cells.  Every public scoring/execution entry point revalidates content hashes,
the current physical revision, and the canonical generated candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    Cell,
    ConfigurationPath,
    ConfigurationPlanner,
    ConfigurationSnapshot,
    PhysicalLabel,
    REGISTERED_PHYSICAL_CELL_SIZE_M,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
)


_EPS = 1e-12
G4_YAW_BIN_COUNT = 16
G4_CAMERA_HORIZONTAL_FOV_DEG = 78.323
G4_CAMERA_VERTICAL_FOV_DEG = 62.8370386364
G4_CAMERA_FORWARD_OFFSET_M = 0.326
G4_CAMERA_LEFT_OFFSET_M = 0.0
G4_CAMERA_UP_OFFSET_M = 0.043
G4_GROUND_PLANE_Z_BODY_M = -0.333
G4_CAMERA_PITCH_RAD = 0.0
G4_CAMERA_NEAR_M = 0.05
G4_VIEW_RANGE_M = 4.0
G4_RAY_COUNT = 31
G4_CANDIDATE_CAP = 512

# Backwards-compatible public name for the horizontal field of view.
G4_CAMERA_FOV_DEG = G4_CAMERA_HORIZONTAL_FOV_DEG


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


def _validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: float, name: str) -> float:
    parsed = float(value)
    if isinstance(value, bool) or not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _cell(value: Sequence[int], name: str = "cell") -> Cell:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or int(item) != item:
            raise ValueError(f"{name} coordinates must be integers")
        result.append(int(item))
    return (result[0], result[1])


def _cells(value: Iterable[Sequence[int]], name: str) -> frozenset[Cell]:
    return frozenset(_cell(cell, name) for cell in value)


def _cells_json(value: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(value)]


def _wrapped_abs_delta(first: float, second: float) -> float:
    return abs((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_for_index(yaw_index: int, *, yaw_bin_count: int = G4_YAW_BIN_COUNT) -> float:
    if isinstance(yaw_index, bool) or int(yaw_index) != yaw_index:
        raise ValueError("yaw_index must be an integer")
    if isinstance(yaw_bin_count, bool) or int(yaw_bin_count) != yaw_bin_count:
        raise ValueError("yaw_bin_count must be an integer")
    count = int(yaw_bin_count)
    index = int(yaw_index)
    if count != G4_YAW_BIN_COUNT or not 0 <= index < count:
        raise ValueError("yaw_index is outside the frozen 16-heading lattice")
    return -math.pi + (2.0 * math.pi * index / count)


@dataclass(frozen=True)
class FrontierViewpointConfig:
    """Frozen G4 baseline and registered camera-ground observation contract."""

    yaw_bin_count: int = G4_YAW_BIN_COUNT
    camera_fov_deg: float = G4_CAMERA_HORIZONTAL_FOV_DEG
    camera_vertical_fov_deg: float = G4_CAMERA_VERTICAL_FOV_DEG
    camera_forward_offset_m: float = G4_CAMERA_FORWARD_OFFSET_M
    camera_left_offset_m: float = G4_CAMERA_LEFT_OFFSET_M
    camera_up_offset_m: float = G4_CAMERA_UP_OFFSET_M
    ground_plane_z_body_m: float = G4_GROUND_PLANE_Z_BODY_M
    camera_pitch_rad: float = G4_CAMERA_PITCH_RAD
    camera_near_m: float = G4_CAMERA_NEAR_M
    view_range_m: float = G4_VIEW_RANGE_M
    ray_count: int = G4_RAY_COUNT
    candidate_cap: int = G4_CANDIDATE_CAP
    physical_cell_size_m: float = REGISTERED_PHYSICAL_CELL_SIZE_M
    coverage_weight: float = 1.0
    entropy_weight: float = 0.35
    discovery_weight: float = 0.50
    path_cost_weight: float = 0.08
    turn_cost_weight: float = 0.05
    pose_uncertainty_weight: float = 0.50
    staleness_weight: float = 0.10
    staleness_horizon_steps: int = 64
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        frozen = {
            "yaw_bin_count": (self.yaw_bin_count, G4_YAW_BIN_COUNT),
            "ray_count": (self.ray_count, G4_RAY_COUNT),
            "candidate_cap": (self.candidate_cap, G4_CANDIDATE_CAP),
            "staleness_horizon_steps": (self.staleness_horizon_steps, 64),
        }
        for name, (actual, expected) in frozen.items():
            if isinstance(actual, bool) or int(actual) != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        frozen_float = {
            "camera_fov_deg": (self.camera_fov_deg, G4_CAMERA_HORIZONTAL_FOV_DEG),
            "camera_vertical_fov_deg": (
                self.camera_vertical_fov_deg,
                G4_CAMERA_VERTICAL_FOV_DEG,
            ),
            "camera_forward_offset_m": (
                self.camera_forward_offset_m,
                G4_CAMERA_FORWARD_OFFSET_M,
            ),
            "camera_left_offset_m": (
                self.camera_left_offset_m,
                G4_CAMERA_LEFT_OFFSET_M,
            ),
            "camera_up_offset_m": (
                self.camera_up_offset_m,
                G4_CAMERA_UP_OFFSET_M,
            ),
            "ground_plane_z_body_m": (
                self.ground_plane_z_body_m,
                G4_GROUND_PLANE_Z_BODY_M,
            ),
            "camera_pitch_rad": (self.camera_pitch_rad, G4_CAMERA_PITCH_RAD),
            "camera_near_m": (self.camera_near_m, G4_CAMERA_NEAR_M),
            "view_range_m": (self.view_range_m, G4_VIEW_RANGE_M),
            "physical_cell_size_m": (
                self.physical_cell_size_m,
                REGISTERED_PHYSICAL_CELL_SIZE_M,
            ),
            "coverage_weight": (self.coverage_weight, 1.0),
            "entropy_weight": (self.entropy_weight, 0.35),
            "discovery_weight": (self.discovery_weight, 0.50),
            "path_cost_weight": (self.path_cost_weight, 0.08),
            "turn_cost_weight": (self.turn_cost_weight, 0.05),
            "pose_uncertainty_weight": (self.pose_uncertainty_weight, 0.50),
            "staleness_weight": (self.staleness_weight, 0.10),
        }
        for name, (actual, expected) in frozen_float.items():
            parsed = _finite(actual, name)
            if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name} is frozen at {expected}")
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_frontier_viewpoint_config_v2",
            "camera_ground_observation_schema": (
                "lewm_g4_registered_camera_ground_closed_supercover_v1"
            ),
            **{
                name: getattr(self, name)
                for name in (
                    "yaw_bin_count",
                    "camera_fov_deg",
                    "camera_vertical_fov_deg",
                    "camera_forward_offset_m",
                    "camera_left_offset_m",
                    "camera_up_offset_m",
                    "ground_plane_z_body_m",
                    "camera_pitch_rad",
                    "camera_near_m",
                    "view_range_m",
                    "ray_count",
                    "candidate_cap",
                    "physical_cell_size_m",
                    "coverage_weight",
                    "entropy_weight",
                    "discovery_weight",
                    "path_cost_weight",
                    "turn_cost_weight",
                    "pose_uncertainty_weight",
                    "staleness_weight",
                    "staleness_horizon_steps",
                )
            },
            "unknown_occlusion": "count_first_unknown_then_stop",
            "missing_domain_occlusion": "stop_before_missing_cell",
            "occupied_occlusion": "stop_before_known_occupied",
        }

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError(
                "G4 config content hash changed after construction"
            )

    @property
    def ground_visible_min_range_m(self) -> float:
        camera_height = self.camera_up_offset_m - self.ground_plane_z_body_m
        half_vertical = math.radians(self.camera_vertical_fov_deg) / 2.0
        return max(self.camera_near_m, camera_height / math.tan(half_vertical))


@dataclass(frozen=True)
class ViewHistoryEntry:
    cell: Cell
    yaw_index: int
    last_observed_step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell))
        yaw_for_index(self.yaw_index)
        object.__setattr__(
            self,
            "last_observed_step",
            _nonnegative_int(self.last_observed_step, "last_observed_step"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": list(self.cell),
            "yaw_index": int(self.yaw_index),
            "last_observed_step": self.last_observed_step,
        }


@dataclass(frozen=True)
class PhysicalViewState:
    """Content-addressed state issued from current physical and visual memory."""

    issuer_sha256: str
    view_memory_sha256: str
    snapshot_sha256: str
    map_frame_sha256: str
    physical_content_sha256: str
    physical_revision: int
    view_revision: int
    view_step: int
    domain_sha256: str
    domain_cells: frozenset[Cell]
    free_cells: frozenset[Cell]
    occupied_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    visually_swept_cells: frozenset[Cell]
    physical_entropy_cells: frozenset[Cell]
    uniform_discovery_cells: frozenset[Cell]
    view_history: tuple[ViewHistoryEntry, ...] = ()
    pose_xy_variance_m2: float = 0.0
    pose_yaw_variance_rad2: float = 0.0
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "issuer_sha256",
            "view_memory_sha256",
            "snapshot_sha256",
            "map_frame_sha256",
            "physical_content_sha256",
            "domain_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in ("physical_revision", "view_revision", "view_step"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        domain = _cells(self.domain_cells, "domain cell")
        free = _cells(self.free_cells, "free cell")
        occupied = _cells(self.occupied_cells, "occupied cell")
        unknown = _cells(self.unknown_cells, "unknown cell")
        swept = _cells(self.visually_swept_cells, "visually swept cell")
        entropy = _cells(self.physical_entropy_cells, "physical entropy cell")
        discovery = _cells(self.uniform_discovery_cells, "uniform discovery cell")
        if free & occupied or free & unknown or occupied & unknown:
            raise ValueError("physical view-state classes must be disjoint")
        if free | occupied | unknown != domain:
            raise ValueError(
                "physical classes must partition the complete registered domain"
            )
        if not swept <= domain:
            raise ValueError(
                "visual sweep map must remain inside the registered domain"
            )
        if entropy != unknown:
            raise ValueError(
                "maximum-entropy cells must exactly equal physical UNKNOWN"
            )
        if discovery != (domain - occupied - swept):
            raise ValueError(
                "uniform discovery map must be unswept non-occupied domain"
            )
        if self.domain_sha256 != _sha256(
            {"schema": "lewm_g4_view_domain_v1", "cells": _cells_json(domain)}
        ):
            raise ValueError("domain_sha256 does not match domain_cells")
        history = tuple(self.view_history)
        if any(not isinstance(row, ViewHistoryEntry) for row in history):
            raise TypeError("view_history entries must be ViewHistoryEntry")
        history = tuple(sorted(history, key=lambda row: (row.cell, row.yaw_index)))
        keys = [(row.cell, row.yaw_index) for row in history]
        if len(set(keys)) != len(keys):
            raise ValueError("view_history may contain one entry per cell/yaw")
        if any(
            row.cell not in domain or row.last_observed_step > self.view_step
            for row in history
        ):
            raise ValueError(
                "view history lies outside the issued domain or future step"
            )
        xy_variance = _finite(self.pose_xy_variance_m2, "pose_xy_variance_m2")
        yaw_variance = _finite(self.pose_yaw_variance_rad2, "pose_yaw_variance_rad2")
        if xy_variance < 0.0 or yaw_variance < 0.0:
            raise ValueError("pose variances must be non-negative")
        for name, value in (
            ("domain_cells", domain),
            ("free_cells", free),
            ("occupied_cells", occupied),
            ("unknown_cells", unknown),
            ("visually_swept_cells", swept),
            ("physical_entropy_cells", entropy),
            ("uniform_discovery_cells", discovery),
            ("view_history", history),
            ("pose_xy_variance_m2", xy_variance),
            ("pose_yaw_variance_rad2", yaw_variance),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_issued_physical_view_state_v2",
            "issuer_sha256": self.issuer_sha256,
            "view_memory_sha256": self.view_memory_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "map_frame_sha256": self.map_frame_sha256,
            "physical_content_sha256": self.physical_content_sha256,
            "physical_revision": self.physical_revision,
            "view_revision": self.view_revision,
            "view_step": self.view_step,
            "domain_sha256": self.domain_sha256,
            "domain_cells": _cells_json(self.domain_cells),
            "free_cells": _cells_json(self.free_cells),
            "occupied_cells": _cells_json(self.occupied_cells),
            "unknown_cells": _cells_json(self.unknown_cells),
            "visually_swept_cells": _cells_json(self.visually_swept_cells),
            "physical_entropy_cells": _cells_json(self.physical_entropy_cells),
            "uniform_discovery_cells": _cells_json(self.uniform_discovery_cells),
            "view_history": [row.to_dict() for row in self.view_history],
            "pose_xy_variance_m2": self.pose_xy_variance_m2,
            "pose_yaw_variance_rad2": self.pose_yaw_variance_rad2,
        }

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("physical view state was mutated after issuance")


class PhysicalViewStateIssuer:
    """Own persistent visual coverage and issue current physical/view projections."""

    def __init__(self, memory: RevisionedPhysicalMemory) -> None:
        if not isinstance(memory, RevisionedPhysicalMemory):
            raise TypeError("memory must be a RevisionedPhysicalMemory")
        self._memory = memory
        self._swept_cells: set[Cell] = set()
        self._history: dict[tuple[Cell, int], int] = {}
        self._records: dict[str, dict[str, object]] = {}
        self._view_revision = 0
        self._view_step = 0
        self._issued_state_sha256: set[str] = set()
        self._issuer_sha256 = _sha256(
            {
                "schema": "lewm_g4_physical_view_state_issuer_v1",
                "map_frame_sha256": memory.map_frame.content_sha256,
                "physical_cell_size_m": memory.map_frame.cell_size_m,
            }
        )

    @property
    def memory(self) -> RevisionedPhysicalMemory:
        return self._memory

    @property
    def issuer_sha256(self) -> str:
        return self._issuer_sha256

    @property
    def view_content_sha256(self) -> str:
        return _sha256(
            {
                "schema": "lewm_g4_visual_coverage_memory_v1",
                "issuer_sha256": self.issuer_sha256,
                "view_revision": self._view_revision,
                "view_step": self._view_step,
                "swept_cells": _cells_json(self._swept_cells),
                "view_history": [
                    {
                        "cell": list(cell),
                        "yaw_index": yaw_index,
                        "last_observed_step": step,
                    }
                    for (cell, yaw_index), step in sorted(self._history.items())
                ],
                "records": [self._records[key] for key in sorted(self._records)],
            }
        )

    def record_view(
        self,
        snapshot: ConfigurationSnapshot,
        *,
        observation_id: str,
        observation_sha256: str,
        observed_cells: Iterable[Sequence[int]],
        viewpoint_cell: Sequence[int],
        yaw_index: int,
        observation_step: int | None = None,
    ) -> None:
        if self._memory.config.promoted_runtime:
            raise PermissionError(
                "promoted visual history requires a qualified camera-view adapter"
            )
        self._memory.assert_current_snapshot(snapshot)
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be non-empty")
        if observation_id in self._records:
            raise ValueError("duplicate visual observation identity")
        _validate_sha256(observation_sha256, "observation_sha256")
        cells = _cells(observed_cells, "observed cell")
        domain = snapshot.evaluated_cells
        if not cells <= domain:
            raise SnapshotBindingError(
                "visual observation lies outside snapshot domain"
            )
        cell = _cell(viewpoint_cell, "viewpoint_cell")
        if cell not in snapshot.free_cells:
            raise SnapshotBindingError("viewpoint must be confirmed configuration-FREE")
        yaw_for_index(yaw_index)
        step = self._view_step + 1 if observation_step is None else _nonnegative_int(
            observation_step, "observation_step"
        )
        if step <= self._view_step:
            raise ValueError("observation_step must increase monotonically")
        self._swept_cells.update(cells)
        self._history[(cell, int(yaw_index))] = step
        self._view_revision += 1
        self._view_step = step
        self._records[observation_id] = {
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "snapshot_sha256": snapshot.content_sha256,
            "physical_revision": snapshot.physical_revision,
            "view_revision": self._view_revision,
            "observation_step": step,
            "observed_cells_sha256": _sha256(
                {"schema": "lewm_g4_observed_cells_v1", "cells": _cells_json(cells)}
            ),
            "viewpoint_cell": list(cell),
            "yaw_index": int(yaw_index),
        }
        self._issued_state_sha256.clear()

    def _build_state(
        self,
        snapshot: ConfigurationSnapshot,
        *,
        pose_xy_variance_m2: float,
        pose_yaw_variance_rad2: float,
    ) -> PhysicalViewState:
        domain = snapshot.evaluated_cells
        free: set[Cell] = set()
        occupied: set[Cell] = set()
        unknown: set[Cell] = set()
        for cell in sorted(domain):
            label = self._memory.physical_state(cell)
            if label is PhysicalLabel.FREE:
                free.add(cell)
            elif label is PhysicalLabel.OCCUPIED:
                occupied.add(cell)
            else:
                unknown.add(cell)
        swept = frozenset(self._swept_cells & set(domain))
        history = tuple(
            ViewHistoryEntry(cell=cell, yaw_index=yaw, last_observed_step=step)
            for (cell, yaw), step in sorted(self._history.items())
            if cell in domain
        )
        domain_sha256 = _sha256(
            {"schema": "lewm_g4_view_domain_v1", "cells": _cells_json(domain)}
        )
        return PhysicalViewState(
            issuer_sha256=self.issuer_sha256,
            view_memory_sha256=self.view_content_sha256,
            snapshot_sha256=snapshot.content_sha256,
            map_frame_sha256=snapshot.map_frame_sha256,
            physical_content_sha256=snapshot.physical_content_sha256,
            physical_revision=snapshot.physical_revision,
            view_revision=self._view_revision,
            view_step=self._view_step,
            domain_sha256=domain_sha256,
            domain_cells=frozenset(domain),
            free_cells=frozenset(free),
            occupied_cells=frozenset(occupied),
            unknown_cells=frozenset(unknown),
            visually_swept_cells=swept,
            physical_entropy_cells=frozenset(unknown),
            uniform_discovery_cells=frozenset(domain - frozenset(occupied) - swept),
            view_history=history,
            pose_xy_variance_m2=pose_xy_variance_m2,
            pose_yaw_variance_rad2=pose_yaw_variance_rad2,
        )

    def issue(
        self,
        snapshot: ConfigurationSnapshot,
        *,
        pose_xy_variance_m2: float = 0.0,
        pose_yaw_variance_rad2: float = 0.0,
    ) -> PhysicalViewState:
        self._memory.assert_current_snapshot(snapshot)
        state = self._build_state(
            snapshot,
            pose_xy_variance_m2=pose_xy_variance_m2,
            pose_yaw_variance_rad2=pose_yaw_variance_rad2,
        )
        self._issued_state_sha256.add(state.content_sha256)
        return state

    def validate_state(
        self,
        snapshot: ConfigurationSnapshot,
        state: PhysicalViewState,
    ) -> None:
        if not isinstance(state, PhysicalViewState):
            raise TypeError("state must be a PhysicalViewState")
        state.assert_integrity()
        self._memory.assert_current_snapshot(snapshot)
        if state.content_sha256 not in self._issued_state_sha256:
            raise SnapshotBindingError(
                "physical view state was not issued by this view memory"
            )
        if (
            state.issuer_sha256 != self.issuer_sha256
            or state.view_memory_sha256 != self.view_content_sha256
            or state.snapshot_sha256 != snapshot.content_sha256
            or state.map_frame_sha256 != self._memory.map_frame.content_sha256
            or state.physical_content_sha256 != self._memory.physical_content_sha256
            or state.physical_revision != self._memory.revision
            or state.view_revision != self._view_revision
            or state.view_step != self._view_step
        ):
            raise SnapshotBindingError("physical view state is stale or foreign")
        expected = self._build_state(
            snapshot,
            pose_xy_variance_m2=state.pose_xy_variance_m2,
            pose_yaw_variance_rad2=state.pose_yaw_variance_rad2,
        )
        if expected.content_sha256 != state.content_sha256:
            raise SnapshotBindingError(
                "physical view state differs from current memory"
            )


@dataclass(frozen=True)
class FrontierViewpointCandidate:
    snapshot_sha256: str
    physical_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    config_sha256: str
    physical_view_state_sha256: str
    start_cell: Cell
    reachable_cell: Cell
    yaw_index: int
    safe_path: tuple[Cell, ...]
    path_cost_m: float
    turn_cost_rad: float
    kind: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "config_sha256",
            "physical_view_state_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        start = _cell(self.start_cell, "start_cell")
        reachable = _cell(self.reachable_cell, "reachable_cell")
        path = tuple(_cell(row, "safe_path cell") for row in self.safe_path)
        if not path or path[0] != start or path[-1] != reachable:
            raise ValueError(
                "safe_path must begin at start_cell and end at reachable_cell"
            )
        yaw_for_index(self.yaw_index)
        path_cost = _finite(self.path_cost_m, "path_cost_m")
        turn_cost = _finite(self.turn_cost_rad, "turn_cost_rad")
        if path_cost < 0.0 or turn_cost < 0.0:
            raise ValueError("path and turn costs must be non-negative")
        if self.kind not in {"frontier", "coverage_frontier", "scan"}:
            raise ValueError("candidate kind is outside the frozen G4 vocabulary")
        object.__setattr__(self, "start_cell", start)
        object.__setattr__(self, "reachable_cell", reachable)
        object.__setattr__(self, "safe_path", path)
        object.__setattr__(self, "path_cost_m", path_cost)
        object.__setattr__(self, "turn_cost_rad", turn_cost)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.to_dict(include_hash=False)),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g4_frontier_viewpoint_candidate_v2",
            "snapshot_sha256": self.snapshot_sha256,
            "physical_revision": self.physical_revision,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "config_sha256": self.config_sha256,
            "physical_view_state_sha256": self.physical_view_state_sha256,
            "start_cell": list(self.start_cell),
            "reachable_cell": list(self.reachable_cell),
            "yaw_index": int(self.yaw_index),
            # Path order is execution authority and must not be set-normalized.
            "safe_path": [list(cell) for cell in self.safe_path],
            "path_cost_m": self.path_cost_m,
            "turn_cost_rad": self.turn_cost_rad,
            "kind": self.kind,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(include_hash=False)):
            raise SnapshotBindingError("candidate was mutated after generation")


@dataclass(frozen=True)
class FrontierViewpointCandidateSet:
    snapshot_sha256: str
    physical_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    config_sha256: str
    physical_view_state_sha256: str
    start_cell: Cell
    current_yaw_rad: float
    candidates: tuple[FrontierViewpointCandidate, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "config_sha256",
            "physical_view_state_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        start = _cell(self.start_cell, "start_cell")
        yaw = _finite(self.current_yaw_rad, "current_yaw_rad")
        candidates = tuple(self.candidates)
        if len(candidates) > G4_CANDIDATE_CAP:
            raise ValueError("candidate set exceeds the frozen cap")
        if any(not isinstance(row, FrontierViewpointCandidate) for row in candidates):
            raise TypeError("candidates must be FrontierViewpointCandidate values")
        for row in candidates:
            row.assert_integrity()
            if (
                row.snapshot_sha256 != self.snapshot_sha256
                or row.physical_revision != self.physical_revision
                or row.free_support_sha256 != self.free_support_sha256
                or row.occupied_support_sha256 != self.occupied_support_sha256
                or row.config_sha256 != self.config_sha256
                or row.physical_view_state_sha256 != self.physical_view_state_sha256
                or row.start_cell != start
            ):
                raise SnapshotBindingError("candidate set contains a foreign binding")
        identities = [row.content_sha256 for row in candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("candidate identities must be unique")
        semantic_order = tuple(
            sorted(candidates, key=lambda row: (row.reachable_cell, row.yaw_index))
        )
        if candidates != semantic_order:
            raise ValueError("candidate set must use canonical cell/yaw order")
        object.__setattr__(self, "start_cell", start)
        object.__setattr__(self, "current_yaw_rad", yaw)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_frontier_viewpoint_candidate_set_v2",
            "snapshot_sha256": self.snapshot_sha256,
            "physical_revision": self.physical_revision,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "config_sha256": self.config_sha256,
            "physical_view_state_sha256": self.physical_view_state_sha256,
            "start_cell": list(self.start_cell),
            "current_yaw_rad": self.current_yaw_rad,
            "candidate_sha256": [row.content_sha256 for row in self.candidates],
        }

    def assert_integrity(self) -> None:
        for row in self.candidates:
            row.assert_integrity()
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("candidate set was mutated after generation")


def _path_turn_cost(
    path: ConfigurationPath,
    current_yaw_rad: float,
    terminal_yaw: float,
) -> float:
    if len(path.cells) == 1:
        return _wrapped_abs_delta(current_yaw_rad, terminal_yaw)
    headings = [
        math.atan2(second[1] - first[1], second[0] - first[0])
        for first, second in zip(path.cells, path.cells[1:])
    ]
    cost = _wrapped_abs_delta(current_yaw_rad, headings[0])
    cost += sum(
        _wrapped_abs_delta(first, second)
        for first, second in zip(headings, headings[1:])
    )
    cost += _wrapped_abs_delta(headings[-1], terminal_yaw)
    return cost


def _coverage_frontier_cell(cell: Cell, state: PhysicalViewState) -> bool:
    if cell not in state.visually_swept_cells:
        return True
    return any(
        (cell[0] + dx, cell[1] + dy) in state.uniform_discovery_cells
        for dx, dy in ((-1, 0), (0, -1), (0, 1), (1, 0))
    )


def _select_spatially_diverse_goals(
    component_cells: frozenset[Cell],
    *,
    start: Cell,
    physical_frontiers: frozenset[Cell],
    state: PhysicalViewState,
    limit: int,
) -> tuple[Cell, ...]:
    """Cap cells after considering the entire connected component.

    The deterministic farthest-point pass avoids consuming all 512 options on
    many headings at the nearest 32 cells. Physical and visual frontiers receive
    priority, while exact known maps still draw from every connected FREE cell.
    """

    if not component_cells or limit <= 0:
        return ()
    selected: list[Cell] = [start]
    remaining = set(component_cells)
    remaining.discard(start)
    while remaining and len(selected) < limit:
        def rank(cell: Cell) -> tuple[int, int, int, int, int]:
            minimum_distance_sq = min(
                (cell[0] - prior[0]) ** 2 + (cell[1] - prior[1]) ** 2
                for prior in selected
            )
            return (
                -int(cell in physical_frontiers),
                -int(_coverage_frontier_cell(cell, state)),
                -minimum_distance_sq,
                cell[0],
                cell[1],
            )

        chosen = min(remaining, key=rank)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def generate_frontier_viewpoint_candidates(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    state: PhysicalViewState,
    *,
    issuer: PhysicalViewStateIssuer,
    start_cell: Sequence[int],
    current_yaw_rad: float,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> FrontierViewpointCandidateSet:
    """Generate safe viewing-pose options from the complete connected component."""

    if not isinstance(planner, ConfigurationPlanner):
        raise TypeError("planner must be a ConfigurationPlanner")
    if not isinstance(snapshot, ConfigurationSnapshot):
        raise TypeError("snapshot must be a ConfigurationSnapshot")
    if not isinstance(issuer, PhysicalViewStateIssuer):
        raise TypeError("issuer must be a PhysicalViewStateIssuer")
    config.assert_integrity()
    issuer.validate_state(snapshot, state)
    start = _cell(start_cell, "start_cell")
    yaw = _finite(current_yaw_rad, "current_yaw_rad")
    component = planner.connected_component(snapshot, start)
    if not component.cells:
        return FrontierViewpointCandidateSet(
            snapshot_sha256=snapshot.content_sha256,
            physical_revision=snapshot.physical_revision,
            free_support_sha256=snapshot.free_support_sha256,
            occupied_support_sha256=snapshot.occupied_support_sha256,
            config_sha256=config.content_sha256,
            physical_view_state_sha256=state.content_sha256,
            start_cell=start,
            current_yaw_rad=yaw,
            candidates=(),
        )
    physical_frontiers = frozenset(
        planner.frontier_cells(snapshot, component).cells
    )
    cell_limit = max(1, config.candidate_cap // config.yaw_bin_count)
    goal_cells = _select_spatially_diverse_goals(
        component.cells,
        start=start,
        physical_frontiers=physical_frontiers,
        state=state,
        limit=cell_limit,
    )
    rows: list[FrontierViewpointCandidate] = []
    for goal in goal_cells:
        path = planner.astar(snapshot, start, goal)
        if path is None:
            raise SnapshotBindingError("connected-component goal has no safe path")
        planner.validate_path(snapshot, path)
        if any(cell not in snapshot.free_cells for cell in path.cells):
            raise SnapshotBindingError("planner returned a path outside confirmed FREE")
        if goal in physical_frontiers:
            kind = "frontier"
        elif goal == start:
            kind = "scan"
        else:
            kind = "coverage_frontier"
        for yaw_index in range(config.yaw_bin_count):
            terminal_yaw = yaw_for_index(yaw_index)
            rows.append(
                FrontierViewpointCandidate(
                    snapshot_sha256=snapshot.content_sha256,
                    physical_revision=snapshot.physical_revision,
                    free_support_sha256=snapshot.free_support_sha256,
                    occupied_support_sha256=snapshot.occupied_support_sha256,
                    config_sha256=config.content_sha256,
                    physical_view_state_sha256=state.content_sha256,
                    start_cell=start,
                    reachable_cell=goal,
                    yaw_index=yaw_index,
                    safe_path=path.cells,
                    path_cost_m=path.cost * config.physical_cell_size_m,
                    turn_cost_rad=_path_turn_cost(path, yaw, terminal_yaw),
                    kind=kind,
                )
            )
    rows.sort(key=lambda row: (row.reachable_cell, row.yaw_index))
    return FrontierViewpointCandidateSet(
        snapshot_sha256=snapshot.content_sha256,
        physical_revision=snapshot.physical_revision,
        free_support_sha256=snapshot.free_support_sha256,
        occupied_support_sha256=snapshot.occupied_support_sha256,
        config_sha256=config.content_sha256,
        physical_view_state_sha256=state.content_sha256,
        start_cell=start,
        current_yaw_rad=yaw,
        candidates=tuple(rows),
    )


def ordered_closed_cell_supercover_groups(
    start_xy_cells: Sequence[float],
    end_xy_cells: Sequence[float],
) -> tuple[tuple[Cell, ...], ...]:
    """Return ordered groups of every closed grid cell touched by a segment.

    At an exact corner crossing both side cells and the diagonal cell share one
    group. A blocker in either side cell therefore occludes the diagonal ray.
    """

    if len(start_xy_cells) != 2 or len(end_xy_cells) != 2:
        raise ValueError("supercover endpoints must be 2D")
    x0 = _finite(start_xy_cells[0], "supercover start x")
    y0 = _finite(start_xy_cells[1], "supercover start y")
    x1 = _finite(end_xy_cells[0], "supercover end x")
    y1 = _finite(end_xy_cells[1], "supercover end y")
    x, y = int(math.floor(x0)), int(math.floor(y0))
    end_x, end_y = int(math.floor(x1)), int(math.floor(y1))
    dx, dy = x1 - x0, y1 - y0
    vertical_boundary_line = abs(dx) <= _EPS and math.isclose(
        x0,
        round(x0),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    horizontal_boundary_line = abs(dy) <= _EPS and math.isclose(
        y0,
        round(y0),
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    def closed_boundary_group(cells: Iterable[Cell]) -> tuple[Cell, ...]:
        expanded: set[Cell] = set(cells)
        for cell_x, cell_y in tuple(expanded):
            if vertical_boundary_line:
                expanded.add((cell_x - 1, cell_y))
            if horizontal_boundary_line:
                expanded.add((cell_x, cell_y - 1))
            if vertical_boundary_line and horizontal_boundary_line:
                expanded.add((cell_x - 1, cell_y - 1))
        return tuple(sorted(expanded))

    initial: set[Cell] = {(x, y)}
    start_on_x_boundary = math.isclose(
        x0, round(x0), rel_tol=0.0, abs_tol=1e-12
    )
    start_on_y_boundary = math.isclose(
        y0, round(y0), rel_tol=0.0, abs_tol=1e-12
    )
    if start_on_x_boundary and not vertical_boundary_line:
        initial.add((x - 1, y))
    if start_on_y_boundary and not horizontal_boundary_line:
        initial.add((x, y - 1))
    if (
        start_on_x_boundary
        and start_on_y_boundary
        and not vertical_boundary_line
        and not horizontal_boundary_line
    ):
        initial.add((x - 1, y - 1))
    groups: list[tuple[Cell, ...]] = [closed_boundary_group(initial)]
    if (x, y) == (end_x, end_y):
        return tuple(groups)
    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    t_delta_x = math.inf if step_x == 0 else abs(1.0 / dx)
    t_delta_y = math.inf if step_y == 0 else abs(1.0 / dy)
    next_boundary_x = float(x + 1 if step_x > 0 else x)
    next_boundary_y = float(y + 1 if step_y > 0 else y)
    t_max_x = math.inf if step_x == 0 else (next_boundary_x - x0) / dx
    t_max_y = math.inf if step_y == 0 else (next_boundary_y - y0) / dy
    while (x, y) != (end_x, end_y):
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            side_x = (x + step_x, y)
            side_y = (x, y + step_y)
            x += step_x
            y += step_y
            groups.append(
                closed_boundary_group({side_x, side_y, (x, y)})
            )
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            x += step_x
            groups.append(closed_boundary_group({(x, y)}))
            t_max_x += t_delta_x
        else:
            y += step_y
            groups.append(closed_boundary_group({(x, y)}))
            t_max_y += t_delta_y
    return tuple(groups)


def conservative_visible_cells_for_ray(
    groups: Sequence[Sequence[Cell]],
    state: PhysicalViewState,
    *,
    eligible_cells: frozenset[Cell] | None = None,
) -> tuple[Cell, ...]:
    """Apply registered missing/occupied/unknown occlusion to ordered groups."""

    state.assert_integrity()
    return _conservative_visible_cells_for_ray_unchecked(
        groups,
        state,
        eligible_cells=eligible_cells,
    )


def _conservative_visible_cells_for_ray_unchecked(
    groups: Sequence[Sequence[Cell]],
    state: PhysicalViewState,
    *,
    eligible_cells: frozenset[Cell] | None = None,
) -> tuple[Cell, ...]:
    """Inner-loop ray trace after the public boundary validated ``state``."""

    eligible = state.domain_cells if eligible_cells is None else eligible_cells
    visible: list[Cell] = []
    seen: set[Cell] = set()
    for raw_group in groups:
        group = tuple(sorted({_cell(cell, "ray cell") for cell in raw_group}))
        if not group or any(cell not in state.domain_cells for cell in group):
            break
        if any(cell in state.occupied_cells for cell in group):
            break
        for cell in group:
            if cell in eligible and cell not in seen:
                visible.append(cell)
                seen.add(cell)
        # With a maximum-entropy occupancy prior, only the first unknown surface
        # is conservatively observable. Cells behind it are not counted.
        if any(cell in state.unknown_cells for cell in group):
            break
    return tuple(visible)


@dataclass(frozen=True)
class PredictedViewObservation:
    candidate_sha256: str
    physical_view_state_sha256: str
    visible_cells: frozenset[Cell]
    newly_swept_cells: frozenset[Cell]
    entropy_reduction_cells: frozenset[Cell]
    discovery_opportunity_cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_sha256(self.candidate_sha256, "candidate_sha256")
        _validate_sha256(
            self.physical_view_state_sha256,
            "physical_view_state_sha256",
        )
        visible = _cells(self.visible_cells, "visible cell")
        swept = _cells(self.newly_swept_cells, "newly swept cell")
        entropy = _cells(self.entropy_reduction_cells, "entropy reduction cell")
        discovery = _cells(
            self.discovery_opportunity_cells,
            "discovery opportunity cell",
        )
        if not swept <= visible or not entropy <= visible or not discovery <= visible:
            raise ValueError("predicted gain maps must be subsets of visible cells")
        for name, value in (
            ("visible_cells", visible),
            ("newly_swept_cells", swept),
            ("entropy_reduction_cells", entropy),
            ("discovery_opportunity_cells", discovery),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_predicted_view_observation_v1",
            "candidate_sha256": self.candidate_sha256,
            "physical_view_state_sha256": self.physical_view_state_sha256,
            "visible_cells": _cells_json(self.visible_cells),
            "newly_swept_cells": _cells_json(self.newly_swept_cells),
            "entropy_reduction_cells": _cells_json(self.entropy_reduction_cells),
            "discovery_opportunity_cells": _cells_json(
                self.discovery_opportunity_cells
            ),
        }


def _camera_origin_grid_cells(
    candidate: FrontierViewpointCandidate,
    config: FrontierViewpointConfig,
) -> tuple[float, float]:
    yaw = yaw_for_index(candidate.yaw_index)
    forward_cells = config.camera_forward_offset_m / config.physical_cell_size_m
    left_cells = config.camera_left_offset_m / config.physical_cell_size_m
    center_x = candidate.reachable_cell[0] + 0.5
    center_y = candidate.reachable_cell[1] + 0.5
    return (
        center_x + math.cos(yaw) * forward_cells - math.sin(yaw) * left_cells,
        center_y + math.sin(yaw) * forward_cells + math.cos(yaw) * left_cells,
    )


def predict_candidate_observation(
    candidate: FrontierViewpointCandidate,
    state: PhysicalViewState,
    *,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> PredictedViewObservation:
    """Predict conservative camera-ground evidence without changing traversability."""

    if not isinstance(candidate, FrontierViewpointCandidate):
        raise TypeError("candidate must be a FrontierViewpointCandidate")
    candidate.assert_integrity()
    state.assert_integrity()
    config.assert_integrity()
    if (
        candidate.snapshot_sha256 != state.snapshot_sha256
        or candidate.physical_revision != state.physical_revision
        or candidate.physical_view_state_sha256 != state.content_sha256
        or candidate.config_sha256 != config.content_sha256
    ):
        raise SnapshotBindingError("candidate, state, and config bindings differ")
    return _predict_candidate_observation_unchecked(candidate, state, config)


def _predict_candidate_observation_unchecked(
    candidate: FrontierViewpointCandidate,
    state: PhysicalViewState,
    config: FrontierViewpointConfig,
) -> PredictedViewObservation:
    """Inner-loop prediction after candidate/state/config validation."""

    terminal_yaw = yaw_for_index(candidate.yaw_index)
    camera_x, camera_y = _camera_origin_grid_cells(candidate, config)
    half_fov = math.radians(config.camera_fov_deg) / 2.0
    ray_offsets = tuple(
        -half_fov + 2.0 * half_fov * index / (config.ray_count - 1)
        for index in range(config.ray_count)
    )
    near_cells = config.camera_near_m / config.physical_cell_size_m
    ground_near_cells = config.ground_visible_min_range_m / config.physical_cell_size_m
    far_cells = config.view_range_m / config.physical_cell_size_m
    visible: set[Cell] = set()
    for offset in ray_offsets:
        angle = terminal_yaw + offset
        cosine, sine = math.cos(angle), math.sin(angle)
        start = (
            camera_x + cosine * near_cells,
            camera_y + sine * near_cells,
        )
        ground_start = (
            camera_x + cosine * ground_near_cells,
            camera_y + sine * ground_near_cells,
        )
        end = (
            camera_x + cosine * far_cells,
            camera_y + sine * far_cells,
        )
        groups = ordered_closed_cell_supercover_groups(start, end)
        ground_groups = ordered_closed_cell_supercover_groups(ground_start, end)
        eligible = frozenset(cell for group in ground_groups for cell in group)
        visible.update(
            _conservative_visible_cells_for_ray_unchecked(
                groups,
                state,
                eligible_cells=eligible,
            )
        )
    visible_frozen = frozenset(visible)
    return PredictedViewObservation(
        candidate_sha256=candidate.content_sha256,
        physical_view_state_sha256=state.content_sha256,
        visible_cells=visible_frozen,
        newly_swept_cells=visible_frozen - state.visually_swept_cells,
        entropy_reduction_cells=visible_frozen & state.physical_entropy_cells,
        discovery_opportunity_cells=(
            visible_frozen & state.uniform_discovery_cells
        ),
    )


def predicted_observable_unknown_cells(
    candidate: FrontierViewpointCandidate,
    state: PhysicalViewState,
    *,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> frozenset[Cell]:
    """Compatibility view of the separate physical-entropy gain map."""

    return predict_candidate_observation(
        candidate,
        state,
        config=config,
    ).entropy_reduction_cells


@dataclass(frozen=True)
class InformationGainScore:
    candidate_sha256: str
    candidate_set_sha256: str
    physical_view_state_sha256: str
    newly_observable_cells: int
    newly_swept_cells: int
    entropy_reduction_cells: int
    discovery_opportunity_cells: int
    normalized_coverage_gain: float
    normalized_entropy_reduction: float
    uniform_discovery_opportunity: float
    normalized_path_cost: float
    normalized_turn_cost: float
    normalized_pose_uncertainty: float
    normalized_view_diversity: float
    normalized_staleness: float
    utility: float
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_sha256",
            "candidate_set_sha256",
            "physical_view_state_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "newly_observable_cells",
            "newly_swept_cells",
            "entropy_reduction_cells",
            "discovery_opportunity_cells",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        for name in (
            "normalized_coverage_gain",
            "normalized_entropy_reduction",
            "uniform_discovery_opportunity",
            "normalized_path_cost",
            "normalized_turn_cost",
            "normalized_pose_uncertainty",
            "normalized_view_diversity",
            "normalized_staleness",
        ):
            value = _finite(getattr(self, name), name)
            if not -_EPS <= value <= 1.0 + _EPS:
                raise ValueError(f"{name} must be in [0, 1]")
        _finite(self.utility, "utility")
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_information_gain_score_v2",
            **{
                name: getattr(self, name)
                for name in (
                    "candidate_sha256",
                    "candidate_set_sha256",
                    "physical_view_state_sha256",
                    "newly_observable_cells",
                    "newly_swept_cells",
                    "entropy_reduction_cells",
                    "discovery_opportunity_cells",
                    "normalized_coverage_gain",
                    "normalized_entropy_reduction",
                    "uniform_discovery_opportunity",
                    "normalized_path_cost",
                    "normalized_turn_cost",
                    "normalized_pose_uncertainty",
                    "normalized_view_diversity",
                    "normalized_staleness",
                    "utility",
                )
            },
        }


def _candidate_relative_history_terms(
    candidate: FrontierViewpointCandidate,
    state: PhysicalViewState,
    config: FrontierViewpointConfig,
) -> tuple[float, float]:
    if not state.view_history:
        return (1.0, 1.0)
    terminal_yaw = yaw_for_index(candidate.yaw_index)
    best_diversity = 1.0
    most_recent_weight = 0.0
    for row in state.view_history:
        spatial = math.hypot(
            candidate.reachable_cell[0] - row.cell[0],
            candidate.reachable_cell[1] - row.cell[1],
        )
        angular = _wrapped_abs_delta(terminal_yaw, yaw_for_index(row.yaw_index))
        diversity = min(1.0, 0.5 * min(1.0, spatial / 4.0) + 0.5 * angular / math.pi)
        best_diversity = min(best_diversity, diversity)
        age = max(0, state.view_step - row.last_observed_step)
        freshness = 1.0 - min(1.0, age / config.staleness_horizon_steps)
        overlap = 1.0 - diversity
        most_recent_weight = max(most_recent_weight, overlap * freshness)
    staleness = 1.0 - most_recent_weight
    return (best_diversity, staleness)


def _assert_candidate_set_current(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    candidate_set: FrontierViewpointCandidateSet,
    state: PhysicalViewState,
    issuer: PhysicalViewStateIssuer,
    config: FrontierViewpointConfig,
) -> None:
    candidate_set.assert_integrity()
    config.assert_integrity()
    issuer.validate_state(snapshot, state)
    if (
        candidate_set.snapshot_sha256 != snapshot.content_sha256
        or candidate_set.physical_revision != snapshot.physical_revision
        or candidate_set.free_support_sha256 != snapshot.free_support_sha256
        or candidate_set.occupied_support_sha256 != snapshot.occupied_support_sha256
        or candidate_set.config_sha256 != config.content_sha256
        or candidate_set.physical_view_state_sha256 != state.content_sha256
    ):
        raise SnapshotBindingError("candidate set has stale or foreign bindings")
    expected = generate_frontier_viewpoint_candidates(
        planner,
        snapshot,
        state,
        issuer=issuer,
        start_cell=candidate_set.start_cell,
        current_yaw_rad=candidate_set.current_yaw_rad,
        config=config,
    )
    if expected.content_sha256 != candidate_set.content_sha256:
        raise SnapshotBindingError("candidate set is not the canonical generated set")


def score_information_gain_candidates(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    candidate_set: FrontierViewpointCandidateSet,
    state: PhysicalViewState,
    *,
    issuer: PhysicalViewStateIssuer,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> tuple[InformationGainScore, ...]:
    """Score the exact issued candidate set with separate observable gain maps."""

    _assert_candidate_set_current(
        planner,
        snapshot,
        candidate_set,
        state,
        issuer,
        config,
    )
    maximum_ray_cells = max(
        1,
        int(math.ceil(config.view_range_m / config.physical_cell_size_m))
        * config.ray_count,
    )
    entropy_denominator = max(1, len(state.physical_entropy_cells))
    discovery_denominator = max(1, len(state.uniform_discovery_cells))
    path_denominator = max(config.view_range_m, config.physical_cell_size_m)
    base_pose_sigma = math.sqrt(state.pose_xy_variance_m2) / config.physical_cell_size_m
    base_pose_sigma += math.sqrt(state.pose_yaw_variance_rad2) / (
        2.0 * math.pi / config.yaw_bin_count
    )
    scores: list[InformationGainScore] = []
    for candidate in candidate_set.candidates:
        observation = _predict_candidate_observation_unchecked(
            candidate,
            state,
            config,
        )
        coverage = min(1.0, len(observation.newly_swept_cells) / maximum_ray_cells)
        entropy = min(
            1.0,
            len(observation.entropy_reduction_cells) / entropy_denominator,
        )
        discovery = min(
            1.0,
            len(observation.discovery_opportunity_cells) / discovery_denominator,
        )
        path = min(1.0, candidate.path_cost_m / path_denominator)
        turn = min(1.0, candidate.turn_cost_rad / (2.0 * math.pi))
        motion_growth = 0.65 * path + 0.35 * turn
        pose = min(
            1.0,
            base_pose_sigma / 4.0 * (1.0 + motion_growth)
            + 0.25 * motion_growth,
        )
        diversity, staleness = _candidate_relative_history_terms(
            candidate,
            state,
            config,
        )
        utility = (
            config.coverage_weight * coverage
            + config.entropy_weight * entropy
            + config.discovery_weight * discovery
            - config.path_cost_weight * path
            - config.turn_cost_weight * turn
            - config.pose_uncertainty_weight * pose
            + config.staleness_weight * (0.5 * diversity + 0.5 * staleness)
        )
        scores.append(
            InformationGainScore(
                candidate_sha256=candidate.content_sha256,
                candidate_set_sha256=candidate_set.content_sha256,
                physical_view_state_sha256=state.content_sha256,
                newly_observable_cells=len(observation.visible_cells),
                newly_swept_cells=len(observation.newly_swept_cells),
                entropy_reduction_cells=len(observation.entropy_reduction_cells),
                discovery_opportunity_cells=len(
                    observation.discovery_opportunity_cells
                ),
                normalized_coverage_gain=coverage,
                normalized_entropy_reduction=entropy,
                uniform_discovery_opportunity=discovery,
                normalized_path_cost=path,
                normalized_turn_cost=turn,
                normalized_pose_uncertainty=pose,
                normalized_view_diversity=diversity,
                normalized_staleness=staleness,
                utility=utility,
            )
        )
    candidate_by_hash = {
        candidate.content_sha256: candidate for candidate in candidate_set.candidates
    }
    scores.sort(
        key=lambda row: (
            -row.utility,
            candidate_by_hash[row.candidate_sha256].reachable_cell,
            candidate_by_hash[row.candidate_sha256].yaw_index,
        )
    )
    return tuple(scores)


def select_information_gain_candidate(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    candidate_set: FrontierViewpointCandidateSet,
    state: PhysicalViewState,
    *,
    issuer: PhysicalViewStateIssuer,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> FrontierViewpointCandidate | None:
    scores = score_information_gain_candidates(
        planner,
        snapshot,
        candidate_set,
        state,
        issuer=issuer,
        config=config,
    )
    if not scores:
        return None
    by_hash = {row.content_sha256: row for row in candidate_set.candidates}
    return by_hash[scores[0].candidate_sha256]


def assert_candidate_executable(
    planner: ConfigurationPlanner,
    snapshot: ConfigurationSnapshot,
    candidate_set: FrontierViewpointCandidateSet,
    candidate: FrontierViewpointCandidate,
    state: PhysicalViewState,
    *,
    issuer: PhysicalViewStateIssuer,
    config: FrontierViewpointConfig = FrontierViewpointConfig(),
) -> None:
    """Fail closed unless this is the exact current generated safe option."""

    _assert_candidate_set_current(
        planner,
        snapshot,
        candidate_set,
        state,
        issuer,
        config,
    )
    candidate.assert_integrity()
    by_hash = {row.content_sha256: row for row in candidate_set.candidates}
    registered = by_hash.get(candidate.content_sha256)
    if registered is None or registered != candidate:
        raise SnapshotBindingError(
            "candidate is not a member of the issued candidate set"
        )
    canonical_path = planner.astar(
        snapshot,
        candidate_set.start_cell,
        candidate.reachable_cell,
    )
    if canonical_path is None:
        raise SnapshotBindingError("candidate endpoint is no longer reachable")
    planner.validate_path(snapshot, canonical_path)
    expected_path_cost = canonical_path.cost * config.physical_cell_size_m
    expected_turn_cost = _path_turn_cost(
        canonical_path,
        candidate_set.current_yaw_rad,
        yaw_for_index(candidate.yaw_index),
    )
    if (
        candidate.start_cell != candidate_set.start_cell
        or candidate.safe_path != canonical_path.cells
        or not math.isclose(
            candidate.path_cost_m,
            expected_path_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            candidate.turn_cost_rad,
            expected_turn_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or any(cell not in snapshot.free_cells for cell in candidate.safe_path)
    ):
        raise SnapshotBindingError(
            "candidate path/cost/turn differs from canonical option"
        )


__all__ = [
    "FrontierViewpointCandidate",
    "FrontierViewpointCandidateSet",
    "FrontierViewpointConfig",
    "G4_CAMERA_FOV_DEG",
    "G4_CAMERA_FORWARD_OFFSET_M",
    "G4_CAMERA_HORIZONTAL_FOV_DEG",
    "G4_CAMERA_LEFT_OFFSET_M",
    "G4_CAMERA_NEAR_M",
    "G4_CAMERA_PITCH_RAD",
    "G4_CAMERA_UP_OFFSET_M",
    "G4_CAMERA_VERTICAL_FOV_DEG",
    "G4_CANDIDATE_CAP",
    "G4_GROUND_PLANE_Z_BODY_M",
    "G4_RAY_COUNT",
    "G4_VIEW_RANGE_M",
    "G4_YAW_BIN_COUNT",
    "InformationGainScore",
    "PhysicalViewState",
    "PhysicalViewStateIssuer",
    "PredictedViewObservation",
    "ViewHistoryEntry",
    "assert_candidate_executable",
    "conservative_visible_cells_for_ray",
    "generate_frontier_viewpoint_candidates",
    "ordered_closed_cell_supercover_groups",
    "predict_candidate_observation",
    "predicted_observable_unknown_cells",
    "score_information_gain_candidates",
    "select_information_gain_candidate",
    "yaw_for_index",
]
