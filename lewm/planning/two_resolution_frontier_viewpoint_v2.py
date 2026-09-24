"""Two-resolution, revision-bound G4 frontier/viewpoint planning.

Configuration-space routes and view history remain on the 0.10 m G3 V2
lattice. Camera visibility, sweep coverage, entropy, and discovery opportunity
remain on the 0.05 m physical lattice. The two are converted only through the
distinct map frames' shared world origin.

This module is additive and non-promotable. Development view recording remains
fail-closed for promoted physical memory until a qualified camera-view receipt
is reviewed separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    PhysicalLabel,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    PHYSICAL_CELL_SIZE_M,
    PROFILE_SHA256,
    ConfigurationComponentV2,
    ConfigurationFrontiersV2,
    ConfigurationPathV2,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_SHA256,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


Cell = tuple[int, int]
Shape = tuple[int, int]
_EPS = 1e-12
G4_V2_YAW_BIN_COUNT = 16
G4_V2_CAMERA_HORIZONTAL_FOV_DEG = 78.323
G4_V2_CAMERA_VERTICAL_FOV_DEG = 62.8370386364
G4_V2_CAMERA_FORWARD_OFFSET_M = 0.326
G4_V2_CAMERA_LEFT_OFFSET_M = 0.0
G4_V2_CAMERA_UP_OFFSET_M = 0.043
G4_V2_GROUND_PLANE_Z_BODY_M = -0.333
G4_V2_CAMERA_PITCH_RAD = 0.0
G4_V2_CAMERA_NEAR_M = 0.05
G4_V2_VIEW_RANGE_M = 4.0
G4_V2_RAY_COUNT = 31
G4_V2_CANDIDATE_CAP = 512


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _cell(value: object, name: str = "cell") -> Cell:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} coordinates must be integers")
    return int(value[0]), int(value[1])


def _shape(value: object, name: str) -> Shape:
    result = _cell(value, name)
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cells(value: Iterable[object], name: str) -> frozenset[Cell]:
    return frozenset(_cell(cell, name) for cell in value)


def _cells_json(value: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(value)]


def _wrapped_abs_delta(first: float, second: float) -> float:
    return abs((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_for_index_v2(
    yaw_index: int,
    *,
    yaw_bin_count: int = G4_V2_YAW_BIN_COUNT,
) -> float:
    if isinstance(yaw_index, bool) or not isinstance(yaw_index, int):
        raise TypeError("yaw_index must be an integer")
    if yaw_bin_count != G4_V2_YAW_BIN_COUNT:
        raise ValueError("yaw_bin_count is frozen at 16")
    if not 0 <= yaw_index < yaw_bin_count:
        raise ValueError("yaw_index is outside the frozen heading lattice")
    return -math.pi + 2.0 * math.pi * yaw_index / yaw_bin_count


def _assert_snapshot_lattices(
    snapshot: TwoResolutionConfigurationSnapshotV2,
) -> None:
    if type(snapshot) is not TwoResolutionConfigurationSnapshotV2:
        raise TypeError("snapshot must be TwoResolutionConfigurationSnapshotV2")
    if (
        snapshot.physical_map_frame.cell_size_m != PHYSICAL_CELL_SIZE_M
        or snapshot.configuration_map_frame.cell_size_m
        != CONFIGURATION_CELL_SIZE_M
        or snapshot.physical_map_frame.origin_xy_m
        != snapshot.configuration_map_frame.origin_xy_m
        or snapshot.physical_shape
        != (
            2 * snapshot.configuration_shape[0],
            2 * snapshot.configuration_shape[1],
        )
        or snapshot.profile_sha256 != PROFILE_SHA256
        or snapshot.free_support_sha256 != FREE_SUPPORT_SHA256
        or snapshot.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
    ):
        raise SnapshotBindingError("G4 V2 snapshot lattice/profile binding changed")


def configuration_cell_center_world_v2(
    snapshot: TwoResolutionConfigurationSnapshotV2,
    cell: Sequence[int],
) -> tuple[float, float]:
    """Return a configuration centre through the bound configuration frame."""

    _assert_snapshot_lattices(snapshot)
    normalized = _cell(cell, "configuration cell")
    if not (
        0 <= normalized[0] < snapshot.configuration_shape[0]
        and 0 <= normalized[1] < snapshot.configuration_shape[1]
    ):
        raise ValueError("configuration cell lies outside the snapshot")
    origin = snapshot.configuration_map_frame.origin_xy_m
    return (
        origin[0] + (normalized[0] + 0.5) * CONFIGURATION_CELL_SIZE_M,
        origin[1] + (normalized[1] + 0.5) * CONFIGURATION_CELL_SIZE_M,
    )


def physical_cell_center_world_v2(
    snapshot: TwoResolutionConfigurationSnapshotV2,
    cell: Sequence[int],
) -> tuple[float, float]:
    """Return a physical centre through the bound physical frame."""

    _assert_snapshot_lattices(snapshot)
    normalized = _cell(cell, "physical cell")
    if not (
        0 <= normalized[0] < snapshot.physical_shape[0]
        and 0 <= normalized[1] < snapshot.physical_shape[1]
    ):
        raise ValueError("physical cell lies outside the snapshot")
    origin = snapshot.physical_map_frame.origin_xy_m
    return (
        origin[0] + (normalized[0] + 0.5) * PHYSICAL_CELL_SIZE_M,
        origin[1] + (normalized[1] + 0.5) * PHYSICAL_CELL_SIZE_M,
    )


def configuration_center_in_physical_grid_v2(
    snapshot: TwoResolutionConfigurationSnapshotV2,
    cell: Sequence[int],
) -> tuple[float, float]:
    """Convert a configuration centre through world metres to the physical grid."""

    world = configuration_cell_center_world_v2(snapshot, cell)
    origin = snapshot.physical_map_frame.origin_xy_m
    result = (
        (world[0] - origin[0]) / PHYSICAL_CELL_SIZE_M,
        (world[1] - origin[1]) / PHYSICAL_CELL_SIZE_M,
    )
    normalized = _cell(cell, "configuration cell")
    expected = (2.0 * normalized[0] + 1.0, 2.0 * normalized[1] + 1.0)
    if not all(
        math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-10)
        for actual, target in zip(result, expected, strict=True)
    ):
        raise SnapshotBindingError("shared-origin world conversion changed")
    return result


@dataclass(frozen=True)
class TwoResolutionFrontierViewpointConfigV2:
    yaw_bin_count: int = G4_V2_YAW_BIN_COUNT
    camera_horizontal_fov_deg: float = G4_V2_CAMERA_HORIZONTAL_FOV_DEG
    camera_vertical_fov_deg: float = G4_V2_CAMERA_VERTICAL_FOV_DEG
    camera_forward_offset_m: float = G4_V2_CAMERA_FORWARD_OFFSET_M
    camera_left_offset_m: float = G4_V2_CAMERA_LEFT_OFFSET_M
    camera_up_offset_m: float = G4_V2_CAMERA_UP_OFFSET_M
    ground_plane_z_body_m: float = G4_V2_GROUND_PLANE_Z_BODY_M
    camera_pitch_rad: float = G4_V2_CAMERA_PITCH_RAD
    camera_near_m: float = G4_V2_CAMERA_NEAR_M
    view_range_m: float = G4_V2_VIEW_RANGE_M
    ray_count: int = G4_V2_RAY_COUNT
    candidate_cap: int = G4_V2_CANDIDATE_CAP
    physical_cell_size_m: float = PHYSICAL_CELL_SIZE_M
    configuration_cell_size_m: float = CONFIGURATION_CELL_SIZE_M
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
        integer_values = {
            "yaw_bin_count": (self.yaw_bin_count, G4_V2_YAW_BIN_COUNT),
            "ray_count": (self.ray_count, G4_V2_RAY_COUNT),
            "candidate_cap": (self.candidate_cap, G4_V2_CANDIDATE_CAP),
            "staleness_horizon_steps": (self.staleness_horizon_steps, 64),
        }
        for name, (actual, expected) in integer_values.items():
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
                or actual != expected
            ):
                raise ValueError(f"{name} is frozen at {expected}")
        float_values = {
            "camera_horizontal_fov_deg": (
                self.camera_horizontal_fov_deg,
                G4_V2_CAMERA_HORIZONTAL_FOV_DEG,
            ),
            "camera_vertical_fov_deg": (
                self.camera_vertical_fov_deg,
                G4_V2_CAMERA_VERTICAL_FOV_DEG,
            ),
            "camera_forward_offset_m": (
                self.camera_forward_offset_m,
                G4_V2_CAMERA_FORWARD_OFFSET_M,
            ),
            "camera_left_offset_m": (
                self.camera_left_offset_m,
                G4_V2_CAMERA_LEFT_OFFSET_M,
            ),
            "camera_up_offset_m": (
                self.camera_up_offset_m,
                G4_V2_CAMERA_UP_OFFSET_M,
            ),
            "ground_plane_z_body_m": (
                self.ground_plane_z_body_m,
                G4_V2_GROUND_PLANE_Z_BODY_M,
            ),
            "camera_pitch_rad": (self.camera_pitch_rad, G4_V2_CAMERA_PITCH_RAD),
            "camera_near_m": (self.camera_near_m, G4_V2_CAMERA_NEAR_M),
            "view_range_m": (self.view_range_m, G4_V2_VIEW_RANGE_M),
            "physical_cell_size_m": (
                self.physical_cell_size_m,
                PHYSICAL_CELL_SIZE_M,
            ),
            "configuration_cell_size_m": (
                self.configuration_cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
            ),
            "coverage_weight": (self.coverage_weight, 1.0),
            "entropy_weight": (self.entropy_weight, 0.35),
            "discovery_weight": (self.discovery_weight, 0.50),
            "path_cost_weight": (self.path_cost_weight, 0.08),
            "turn_cost_weight": (self.turn_cost_weight, 0.05),
            "pose_uncertainty_weight": (self.pose_uncertainty_weight, 0.50),
            "staleness_weight": (self.staleness_weight, 0.10),
        }
        for name, (actual, expected) in float_values.items():
            if not math.isclose(
                _finite(actual, name), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{name} is frozen at {expected}")
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_frontier_viewpoint_config_v1",
            **{
                name: getattr(self, name)
                for name in (
                    "yaw_bin_count",
                    "camera_horizontal_fov_deg",
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
                    "configuration_cell_size_m",
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
            "route_lattice": "configuration_0.10m",
            "visibility_lattice": "physical_0.05m",
            "lattice_conversion": "configuration_world_physical_shared_origin",
            "unknown_occlusion": "count_first_unknown_then_stop",
            "missing_domain_occlusion": "stop_before_missing_cell",
            "occupied_occlusion": "stop_before_known_occupied",
        }

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("G4 V2 config was mutated")

    @property
    def ground_visible_min_range_m(self) -> float:
        camera_height = self.camera_up_offset_m - self.ground_plane_z_body_m
        half_vertical = math.radians(self.camera_vertical_fov_deg) / 2.0
        return max(self.camera_near_m, camera_height / math.tan(half_vertical))


DEFAULT_G4_V2_CONFIG = TwoResolutionFrontierViewpointConfigV2()


@dataclass(frozen=True)
class ConfigurationViewHistoryEntryV2:
    configuration_cell: Cell
    yaw_index: int
    last_observed_step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_cell",
            _cell(self.configuration_cell, "configuration history cell"),
        )
        yaw_for_index_v2(self.yaw_index)
        object.__setattr__(
            self,
            "last_observed_step",
            _nonnegative_int(self.last_observed_step, "last_observed_step"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_cell": list(self.configuration_cell),
            "yaw_index": self.yaw_index,
            "last_observed_step": self.last_observed_step,
        }


@dataclass(frozen=True)
class TwoResolutionPhysicalViewStateV2:
    issuer_sha256: str
    view_memory_sha256: str
    snapshot_sha256: str
    physical_map_frame: MapFrameIdentity
    configuration_map_frame: MapFrameIdentity
    memory_config_sha256: str
    physical_content_sha256: str
    projection_source_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    physical_revision: int
    configuration_revision: int
    view_revision: int
    view_step: int
    physical_shape: Shape
    configuration_shape: Shape
    physical_free_cells: frozenset[Cell]
    physical_occupied_cells: frozenset[Cell]
    physical_unknown_cells: frozenset[Cell]
    visually_swept_physical_cells: frozenset[Cell]
    physical_entropy_cells: frozenset[Cell]
    physical_discovery_opportunity_cells: frozenset[Cell]
    configuration_view_history: tuple[ConfigurationViewHistoryEntryV2, ...] = ()
    pose_xy_variance_m2: float = 0.0
    pose_yaw_variance_rad2: float = 0.0
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "issuer_sha256",
            "view_memory_sha256",
            "snapshot_sha256",
            "memory_config_sha256",
            "physical_content_sha256",
            "projection_source_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if type(self.physical_map_frame) is not MapFrameIdentity or type(
            self.configuration_map_frame
        ) is not MapFrameIdentity:
            raise TypeError("view-state frames must be canonical MapFrameIdentity")
        if (
            self.physical_map_frame.cell_size_m != PHYSICAL_CELL_SIZE_M
            or self.configuration_map_frame.cell_size_m
            != CONFIGURATION_CELL_SIZE_M
            or self.physical_map_frame.origin_xy_m
            != self.configuration_map_frame.origin_xy_m
            or self.physical_map_frame.content_sha256
            == self.configuration_map_frame.content_sha256
            or self.profile_sha256 != PROFILE_SHA256
            or self.free_support_sha256 != FREE_SUPPORT_SHA256
            or self.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
        ):
            raise ValueError("view-state two-resolution frame/profile binding changed")
        for name in (
            "physical_revision",
            "configuration_revision",
            "view_revision",
            "view_step",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape, "configuration_shape"
        )
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("view-state shapes must retain the exact 2:1 ratio")
        domain = frozenset(
            (x, y)
            for x in range(physical_shape[0])
            for y in range(physical_shape[1])
        )
        free = _cells(self.physical_free_cells, "physical FREE cell")
        occupied = _cells(self.physical_occupied_cells, "physical OCCUPIED cell")
        unknown = _cells(self.physical_unknown_cells, "physical UNKNOWN cell")
        swept = _cells(
            self.visually_swept_physical_cells, "visually swept physical cell"
        )
        entropy = _cells(self.physical_entropy_cells, "physical entropy cell")
        discovery = _cells(
            self.physical_discovery_opportunity_cells,
            "physical discovery-opportunity cell",
        )
        if (
            free & occupied
            or free & unknown
            or occupied & unknown
            or free | occupied | unknown != domain
        ):
            raise ValueError("physical view classes must partition the physical raster")
        if not swept <= domain or entropy != unknown:
            raise ValueError("physical sweep/entropy state changed lattice ownership")
        if discovery != domain - occupied - swept:
            raise ValueError("physical discovery opportunity changed definition")
        history = tuple(self.configuration_view_history)
        if any(type(row) is not ConfigurationViewHistoryEntryV2 for row in history):
            raise TypeError("configuration view history has a foreign entry type")
        history = tuple(
            sorted(history, key=lambda row: (row.configuration_cell, row.yaw_index))
        )
        keys = [(row.configuration_cell, row.yaw_index) for row in history]
        if len(keys) != len(set(keys)):
            raise ValueError("configuration view history contains duplicate keys")
        if any(
            row.last_observed_step > self.view_step
            or not (
                0 <= row.configuration_cell[0] < configuration_shape[0]
                and 0 <= row.configuration_cell[1] < configuration_shape[1]
            )
            for row in history
        ):
            raise ValueError("configuration view history lies outside its lattice")
        xy_variance = _finite(self.pose_xy_variance_m2, "pose_xy_variance_m2")
        yaw_variance = _finite(
            self.pose_yaw_variance_rad2, "pose_yaw_variance_rad2"
        )
        if xy_variance < 0.0 or yaw_variance < 0.0:
            raise ValueError("pose variances must be non-negative")
        for name, value in (
            ("physical_shape", physical_shape),
            ("configuration_shape", configuration_shape),
            ("physical_free_cells", free),
            ("physical_occupied_cells", occupied),
            ("physical_unknown_cells", unknown),
            ("visually_swept_physical_cells", swept),
            ("physical_entropy_cells", entropy),
            ("physical_discovery_opportunity_cells", discovery),
            ("configuration_view_history", history),
            ("pose_xy_variance_m2", xy_variance),
            ("pose_yaw_variance_rad2", yaw_variance),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    @property
    def physical_map_frame_sha256(self) -> str:
        return self.physical_map_frame.content_sha256

    @property
    def configuration_map_frame_sha256(self) -> str:
        return self.configuration_map_frame.content_sha256

    @property
    def physical_domain_cells(self) -> frozenset[Cell]:
        return (
            self.physical_free_cells
            | self.physical_occupied_cells
            | self.physical_unknown_cells
        )

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_physical_view_state_v1",
            "issuer_sha256": self.issuer_sha256,
            "view_memory_sha256": self.view_memory_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "physical_map_frame": self.physical_map_frame.to_dict(),
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame": self.configuration_map_frame.to_dict(),
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "memory_config_sha256": self.memory_config_sha256,
            "physical_content_sha256": self.physical_content_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "view_revision": self.view_revision,
            "view_step": self.view_step,
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_free_cells": _cells_json(self.physical_free_cells),
            "physical_occupied_cells": _cells_json(self.physical_occupied_cells),
            "physical_unknown_cells": _cells_json(self.physical_unknown_cells),
            "visually_swept_physical_cells": _cells_json(
                self.visually_swept_physical_cells
            ),
            "physical_entropy_cells": _cells_json(self.physical_entropy_cells),
            "physical_discovery_opportunity_cells": _cells_json(
                self.physical_discovery_opportunity_cells
            ),
            "configuration_view_history": [
                row.to_dict() for row in self.configuration_view_history
            ],
            "pose_xy_variance_m2": self.pose_xy_variance_m2,
            "pose_yaw_variance_rad2": self.pose_yaw_variance_rad2,
            "production_promotion_authorized": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("G4 V2 physical view state was mutated")


class TwoResolutionPhysicalViewStateIssuerV2:
    """Own physical sweep memory and configuration viewpoint history."""

    def __init__(
        self,
        memory: RevisionedPhysicalMemory,
        projection: TwoResolutionConfigurationProjectionV2,
    ) -> None:
        if type(memory) is not RevisionedPhysicalMemory:
            raise TypeError("memory must be RevisionedPhysicalMemory")
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("projection must be TwoResolutionConfigurationProjectionV2")
        if (
            memory.map_frame.cell_size_m != PHYSICAL_CELL_SIZE_M
            or projection.configuration_map_frame.cell_size_m
            != CONFIGURATION_CELL_SIZE_M
            or memory.map_frame.origin_xy_m
            != projection.configuration_map_frame.origin_xy_m
        ):
            raise ValueError("issuer physical/configuration frames are not aligned")
        self._memory = memory
        self._projection = projection
        self._swept_physical_cells: set[Cell] = set()
        self._configuration_history: dict[tuple[Cell, int], int] = {}
        self._records: dict[str, dict[str, object]] = {}
        self._view_revision = 0
        self._view_step = 0
        self._issued_states: dict[int, TwoResolutionPhysicalViewStateV2] = {}
        self._issuer_sha256 = _sha256(
            {
                "schema": "lewm_g4_v2_two_resolution_view_state_issuer_v1",
                "physical_map_frame_sha256": memory.map_frame.content_sha256,
                "configuration_map_frame_sha256": (
                    projection.configuration_map_frame.content_sha256
                ),
                "projection_source_sha256": projection.projection_source_sha256,
                "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
                "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            }
        )

    @property
    def memory(self) -> RevisionedPhysicalMemory:
        return self._memory

    @property
    def projection(self) -> TwoResolutionConfigurationProjectionV2:
        return self._projection

    @property
    def issuer_sha256(self) -> str:
        return self._issuer_sha256

    @property
    def view_content_sha256(self) -> str:
        return _sha256(
            {
                "schema": "lewm_g4_v2_two_resolution_view_memory_v1",
                "issuer_sha256": self.issuer_sha256,
                "view_revision": self._view_revision,
                "view_step": self._view_step,
                "swept_physical_cells": _cells_json(self._swept_physical_cells),
                "configuration_view_history": [
                    {
                        "configuration_cell": list(cell),
                        "yaw_index": yaw,
                        "last_observed_step": step,
                    }
                    for (cell, yaw), step in sorted(
                        self._configuration_history.items()
                    )
                ],
                "records": [self._records[key] for key in sorted(self._records)],
            }
        )

    def _assert_snapshot(
        self, snapshot: TwoResolutionConfigurationSnapshotV2
    ) -> None:
        self._projection.assert_current_snapshot(snapshot)
        _assert_snapshot_lattices(snapshot)
        if (
            snapshot.physical_map_frame_sha256
            != self._memory.map_frame.content_sha256
            or snapshot.configuration_map_frame_sha256
            != self._projection.configuration_map_frame.content_sha256
            or snapshot.memory_config_sha256 != self._memory.config.content_sha256
            or snapshot.physical_content_sha256
            != self._memory.physical_content_sha256
            or snapshot.projection_source_sha256
            != self._projection.projection_source_sha256
            or snapshot.physical_revision != self._memory.revision
        ):
            raise SnapshotBindingError("view issuer snapshot is stale or foreign")

    def record_development_view(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        *,
        observation_id: str,
        observation_sha256: str,
        observed_physical_cells: Iterable[Sequence[int]],
        viewpoint_configuration_cell: Sequence[int],
        yaw_index: int,
        observation_step: int | None = None,
    ) -> None:
        """Record a development receipt; promoted runtime remains fail-closed."""

        if self._memory.config.promoted_runtime:
            raise PermissionError(
                "promoted visual history requires a qualified camera-view receipt"
            )
        self._assert_snapshot(snapshot)
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be non-empty")
        if observation_id in self._records:
            raise ValueError("duplicate visual observation identity")
        _validate_sha256(observation_sha256, "observation_sha256")
        physical_cells = _cells(
            observed_physical_cells, "observed physical cell"
        )
        if any(
            not (
                0 <= cell[0] < snapshot.physical_shape[0]
                and 0 <= cell[1] < snapshot.physical_shape[1]
            )
            for cell in physical_cells
        ):
            raise SnapshotBindingError("view receipt leaves the physical raster")
        configuration_cell = _cell(
            viewpoint_configuration_cell, "viewpoint configuration cell"
        )
        if configuration_cell not in snapshot.free_cells:
            raise SnapshotBindingError(
                "viewpoint must be current confirmed configuration-FREE"
            )
        yaw_for_index_v2(yaw_index)
        step = (
            self._view_step + 1
            if observation_step is None
            else _nonnegative_int(observation_step, "observation_step")
        )
        if step <= self._view_step:
            raise ValueError("observation_step must increase monotonically")
        self._swept_physical_cells.update(physical_cells)
        self._configuration_history[(configuration_cell, yaw_index)] = step
        self._view_revision += 1
        self._view_step = step
        self._records[observation_id] = {
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "snapshot_sha256": snapshot.content_sha256,
            "physical_map_frame_sha256": snapshot.physical_map_frame_sha256,
            "configuration_map_frame_sha256": (
                snapshot.configuration_map_frame_sha256
            ),
            "physical_revision": snapshot.physical_revision,
            "configuration_revision": snapshot.configuration_revision,
            "view_revision": self._view_revision,
            "observation_step": step,
            "observed_physical_cells_sha256": _sha256(
                {
                    "schema": "lewm_g4_v2_observed_physical_cells_v1",
                    "cells": _cells_json(physical_cells),
                }
            ),
            "viewpoint_configuration_cell": list(configuration_cell),
            "yaw_index": yaw_index,
        }
        self._issued_states.clear()

    def _build_state(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        *,
        pose_xy_variance_m2: float,
        pose_yaw_variance_rad2: float,
    ) -> TwoResolutionPhysicalViewStateV2:
        free: set[Cell] = set()
        occupied: set[Cell] = set()
        unknown: set[Cell] = set()
        for x in range(snapshot.physical_shape[0]):
            for y in range(snapshot.physical_shape[1]):
                cell = (x, y)
                label = self._memory.physical_state(cell)
                if label is PhysicalLabel.FREE:
                    free.add(cell)
                elif label is PhysicalLabel.OCCUPIED:
                    occupied.add(cell)
                else:
                    unknown.add(cell)
        domain = frozenset(free | occupied | unknown)
        swept = frozenset(self._swept_physical_cells & set(domain))
        history = tuple(
            ConfigurationViewHistoryEntryV2(
                configuration_cell=cell,
                yaw_index=yaw,
                last_observed_step=step,
            )
            for (cell, yaw), step in sorted(self._configuration_history.items())
            if (
                0 <= cell[0] < snapshot.configuration_shape[0]
                and 0 <= cell[1] < snapshot.configuration_shape[1]
            )
        )
        return TwoResolutionPhysicalViewStateV2(
            issuer_sha256=self.issuer_sha256,
            view_memory_sha256=self.view_content_sha256,
            snapshot_sha256=snapshot.content_sha256,
            physical_map_frame=snapshot.physical_map_frame,
            configuration_map_frame=snapshot.configuration_map_frame,
            memory_config_sha256=snapshot.memory_config_sha256,
            physical_content_sha256=snapshot.physical_content_sha256,
            projection_source_sha256=snapshot.projection_source_sha256,
            profile_sha256=snapshot.profile_sha256,
            free_support_sha256=snapshot.free_support_sha256,
            occupied_support_sha256=snapshot.occupied_support_sha256,
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            view_revision=self._view_revision,
            view_step=self._view_step,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            physical_free_cells=frozenset(free),
            physical_occupied_cells=frozenset(occupied),
            physical_unknown_cells=frozenset(unknown),
            visually_swept_physical_cells=swept,
            physical_entropy_cells=frozenset(unknown),
            physical_discovery_opportunity_cells=domain
            - frozenset(occupied)
            - swept,
            configuration_view_history=history,
            pose_xy_variance_m2=pose_xy_variance_m2,
            pose_yaw_variance_rad2=pose_yaw_variance_rad2,
        )

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        *,
        pose_xy_variance_m2: float = 0.0,
        pose_yaw_variance_rad2: float = 0.0,
    ) -> TwoResolutionPhysicalViewStateV2:
        self._assert_snapshot(snapshot)
        state = self._build_state(
            snapshot,
            pose_xy_variance_m2=pose_xy_variance_m2,
            pose_yaw_variance_rad2=pose_yaw_variance_rad2,
        )
        self._issued_states[id(state)] = state
        return state

    def validate_state(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
    ) -> None:
        self._assert_snapshot(snapshot)
        if type(state) is not TwoResolutionPhysicalViewStateV2:
            raise TypeError("state must be TwoResolutionPhysicalViewStateV2")
        if self._issued_states.get(id(state)) is not state:
            raise SnapshotBindingError(
                "view state is not the exact live object issued by this issuer"
            )
        state.assert_integrity()
        if (
            state.issuer_sha256 != self.issuer_sha256
            or state.view_memory_sha256 != self.view_content_sha256
            or state.snapshot_sha256 != snapshot.content_sha256
            or state.physical_map_frame_sha256
            != snapshot.physical_map_frame_sha256
            or state.configuration_map_frame_sha256
            != snapshot.configuration_map_frame_sha256
            or state.memory_config_sha256 != snapshot.memory_config_sha256
            or state.physical_content_sha256 != snapshot.physical_content_sha256
            or state.projection_source_sha256
            != snapshot.projection_source_sha256
            or state.profile_sha256 != snapshot.profile_sha256
            or state.free_support_sha256 != snapshot.free_support_sha256
            or state.occupied_support_sha256
            != snapshot.occupied_support_sha256
            or state.physical_revision != snapshot.physical_revision
            or state.configuration_revision != snapshot.configuration_revision
            or state.view_revision != self._view_revision
            or state.view_step != self._view_step
            or state.physical_shape != snapshot.physical_shape
            or state.configuration_shape != snapshot.configuration_shape
        ):
            raise SnapshotBindingError("view state is stale or foreign")
        expected = self._build_state(
            snapshot,
            pose_xy_variance_m2=state.pose_xy_variance_m2,
            pose_yaw_variance_rad2=state.pose_yaw_variance_rad2,
        )
        if expected.content_sha256 != state.content_sha256:
            raise SnapshotBindingError("view state differs from current memory")


def _binding_dict(
    snapshot: TwoResolutionConfigurationSnapshotV2,
    state: TwoResolutionPhysicalViewStateV2,
) -> dict[str, object]:
    return {
        "snapshot_sha256": snapshot.content_sha256,
        "physical_map_frame_sha256": snapshot.physical_map_frame_sha256,
        "configuration_map_frame_sha256": snapshot.configuration_map_frame_sha256,
        "memory_config_sha256": snapshot.memory_config_sha256,
        "physical_content_sha256": snapshot.physical_content_sha256,
        "projection_source_sha256": snapshot.projection_source_sha256,
        "profile_sha256": snapshot.profile_sha256,
        "free_support_sha256": snapshot.free_support_sha256,
        "occupied_support_sha256": snapshot.occupied_support_sha256,
        "physical_revision": snapshot.physical_revision,
        "configuration_revision": snapshot.configuration_revision,
        "physical_shape": snapshot.physical_shape,
        "configuration_shape": snapshot.configuration_shape,
        "physical_view_state_sha256": state.content_sha256,
    }


@dataclass(frozen=True)
class TwoResolutionFrontierViewpointCandidateV2:
    snapshot_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    memory_config_sha256: str
    physical_content_sha256: str
    projection_source_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_shape: Shape
    configuration_shape: Shape
    physical_view_state_sha256: str
    component_sha256: str
    frontiers_sha256: str
    config_sha256: str
    path_receipt_sha256: str
    start_configuration_cell: Cell
    reachable_configuration_cell: Cell
    yaw_index: int
    safe_configuration_path: tuple[Cell, ...]
    path_cost_m: float
    turn_cost_rad: float
    kind: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "memory_config_sha256",
            "physical_content_sha256",
            "projection_source_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "physical_view_state_sha256",
            "component_sha256",
            "frontiers_sha256",
            "config_sha256",
            "path_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in ("physical_revision", "configuration_revision"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape, "configuration_shape"
        )
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("candidate shapes changed the exact 2:1 ratio")
        start = _cell(
            self.start_configuration_cell, "start configuration cell"
        )
        reachable = _cell(
            self.reachable_configuration_cell, "reachable configuration cell"
        )
        path = tuple(
            _cell(cell, "safe configuration path cell")
            for cell in self.safe_configuration_path
        )
        if not path or path[0] != start or path[-1] != reachable:
            raise ValueError("safe configuration path has inconsistent endpoints")
        yaw_for_index_v2(self.yaw_index)
        path_cost = _finite(self.path_cost_m, "path_cost_m")
        turn_cost = _finite(self.turn_cost_rad, "turn_cost_rad")
        if path_cost < 0.0 or turn_cost < 0.0:
            raise ValueError("candidate costs must be non-negative")
        if self.kind not in {"frontier", "coverage_frontier", "scan"}:
            raise ValueError("candidate kind is outside the frozen vocabulary")
        for name, value in (
            ("physical_shape", physical_shape),
            ("configuration_shape", configuration_shape),
            ("start_configuration_cell", start),
            ("reachable_configuration_cell", reachable),
            ("safe_configuration_path", path),
            ("path_cost_m", path_cost),
            ("turn_cost_rad", turn_cost),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_frontier_viewpoint_candidate_v1",
            **{
                name: getattr(self, name)
                for name in (
                    "snapshot_sha256",
                    "physical_map_frame_sha256",
                    "configuration_map_frame_sha256",
                    "memory_config_sha256",
                    "physical_content_sha256",
                    "projection_source_sha256",
                    "profile_sha256",
                    "free_support_sha256",
                    "occupied_support_sha256",
                    "physical_revision",
                    "configuration_revision",
                    "physical_view_state_sha256",
                    "component_sha256",
                    "frontiers_sha256",
                    "config_sha256",
                    "path_receipt_sha256",
                    "yaw_index",
                    "path_cost_m",
                    "turn_cost_rad",
                    "kind",
                )
            },
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "start_configuration_cell": list(self.start_configuration_cell),
            "reachable_configuration_cell": list(
                self.reachable_configuration_cell
            ),
            "safe_configuration_path": [
                list(cell) for cell in self.safe_configuration_path
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("G4 V2 candidate was mutated")


@dataclass(frozen=True)
class TwoResolutionFrontierViewpointCandidateSetV2:
    snapshot_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    memory_config_sha256: str
    physical_content_sha256: str
    projection_source_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_shape: Shape
    configuration_shape: Shape
    physical_view_state_sha256: str
    component_sha256: str
    frontiers_sha256: str
    config_sha256: str
    start_configuration_cell: Cell
    current_yaw_rad: float
    candidates: tuple[TwoResolutionFrontierViewpointCandidateV2, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "memory_config_sha256",
            "physical_content_sha256",
            "projection_source_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "physical_view_state_sha256",
            "component_sha256",
            "frontiers_sha256",
            "config_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in ("physical_revision", "configuration_revision"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape, "configuration_shape"
        )
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("candidate-set shapes changed the exact ratio")
        start = _cell(
            self.start_configuration_cell, "start configuration cell"
        )
        yaw = _finite(self.current_yaw_rad, "current_yaw_rad")
        candidates = tuple(self.candidates)
        if len(candidates) > G4_V2_CANDIDATE_CAP:
            raise ValueError("candidate set exceeds the frozen cap")
        if any(
            type(candidate) is not TwoResolutionFrontierViewpointCandidateV2
            for candidate in candidates
        ):
            raise TypeError("candidate set contains a foreign candidate type")
        if len({candidate.content_sha256 for candidate in candidates}) != len(
            candidates
        ):
            raise ValueError("candidate identities must be unique")
        for name, value in (
            ("physical_shape", physical_shape),
            ("configuration_shape", configuration_shape),
            ("start_configuration_cell", start),
            ("current_yaw_rad", yaw),
            ("candidates", candidates),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_candidate_set_v1",
            **{
                name: getattr(self, name)
                for name in (
                    "snapshot_sha256",
                    "physical_map_frame_sha256",
                    "configuration_map_frame_sha256",
                    "memory_config_sha256",
                    "physical_content_sha256",
                    "projection_source_sha256",
                    "profile_sha256",
                    "free_support_sha256",
                    "occupied_support_sha256",
                    "physical_revision",
                    "configuration_revision",
                    "physical_view_state_sha256",
                    "component_sha256",
                    "frontiers_sha256",
                    "config_sha256",
                    "current_yaw_rad",
                )
            },
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "start_configuration_cell": list(self.start_configuration_cell),
            "candidate_sha256s": [
                candidate.content_sha256 for candidate in self.candidates
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    def assert_integrity(self) -> None:
        for candidate in self.candidates:
            candidate.assert_integrity()
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("G4 V2 candidate set was mutated")


def ordered_closed_physical_supercover_groups_v2(
    start_xy_physical_cells: Sequence[float],
    end_xy_physical_cells: Sequence[float],
) -> tuple[tuple[Cell, ...], ...]:
    """Return ordered closed physical-cell groups touched by a ray segment."""

    if len(start_xy_physical_cells) != 2 or len(end_xy_physical_cells) != 2:
        raise ValueError("supercover endpoints must be 2D")
    x0 = _finite(start_xy_physical_cells[0], "supercover start x")
    y0 = _finite(start_xy_physical_cells[1], "supercover start y")
    x1 = _finite(end_xy_physical_cells[0], "supercover end x")
    y1 = _finite(end_xy_physical_cells[1], "supercover end y")
    x, y = int(math.floor(x0)), int(math.floor(y0))
    end_x, end_y = int(math.floor(x1)), int(math.floor(y1))
    dx, dy = x1 - x0, y1 - y0
    vertical_line = abs(dx) <= _EPS and math.isclose(
        x0, round(x0), rel_tol=0.0, abs_tol=1e-12
    )
    horizontal_line = abs(dy) <= _EPS and math.isclose(
        y0, round(y0), rel_tol=0.0, abs_tol=1e-12
    )

    def closed_group(cells: Iterable[Cell]) -> tuple[Cell, ...]:
        expanded = set(cells)
        for cell_x, cell_y in tuple(expanded):
            if vertical_line:
                expanded.add((cell_x - 1, cell_y))
            if horizontal_line:
                expanded.add((cell_x, cell_y - 1))
            if vertical_line and horizontal_line:
                expanded.add((cell_x - 1, cell_y - 1))
        return tuple(sorted(expanded))

    initial: set[Cell] = {(x, y)}
    start_on_x = math.isclose(x0, round(x0), rel_tol=0.0, abs_tol=1e-12)
    start_on_y = math.isclose(y0, round(y0), rel_tol=0.0, abs_tol=1e-12)
    if start_on_x and not vertical_line:
        initial.add((x - 1, y))
    if start_on_y and not horizontal_line:
        initial.add((x, y - 1))
    if start_on_x and start_on_y and not vertical_line and not horizontal_line:
        initial.add((x - 1, y - 1))
    groups: list[tuple[Cell, ...]] = [closed_group(initial)]
    if (x, y) == (end_x, end_y):
        return tuple(groups)
    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    delta_x = math.inf if step_x == 0 else abs(1.0 / dx)
    delta_y = math.inf if step_y == 0 else abs(1.0 / dy)
    boundary_x = float(x + 1 if step_x > 0 else x)
    boundary_y = float(y + 1 if step_y > 0 else y)
    maximum_x = math.inf if step_x == 0 else (boundary_x - x0) / dx
    maximum_y = math.inf if step_y == 0 else (boundary_y - y0) / dy
    while (x, y) != (end_x, end_y):
        if math.isclose(maximum_x, maximum_y, rel_tol=0.0, abs_tol=1e-12):
            side_x = (x + step_x, y)
            side_y = (x, y + step_y)
            x += step_x
            y += step_y
            groups.append(closed_group({side_x, side_y, (x, y)}))
            maximum_x += delta_x
            maximum_y += delta_y
        elif maximum_x < maximum_y:
            x += step_x
            groups.append(closed_group({(x, y)}))
            maximum_x += delta_x
        else:
            y += step_y
            groups.append(closed_group({(x, y)}))
            maximum_y += delta_y
    return tuple(groups)


def _visible_physical_cells(
    groups: Sequence[Sequence[Cell]],
    state: TwoResolutionPhysicalViewStateV2,
    *,
    eligible_cells: frozenset[Cell],
) -> tuple[Cell, ...]:
    visible: list[Cell] = []
    seen: set[Cell] = set()
    for raw_group in groups:
        group = tuple(sorted({_cell(cell, "physical ray cell") for cell in raw_group}))
        if not group or any(
            not (
                0 <= cell[0] < state.physical_shape[0]
                and 0 <= cell[1] < state.physical_shape[1]
            )
            for cell in group
        ):
            break
        if any(cell in state.physical_occupied_cells for cell in group):
            break
        for cell in group:
            if cell in eligible_cells and cell not in seen:
                visible.append(cell)
                seen.add(cell)
        if any(cell in state.physical_unknown_cells for cell in group):
            break
    return tuple(visible)


@dataclass(frozen=True)
class TwoResolutionPredictedViewObservationV2:
    candidate_sha256: str
    physical_view_state_sha256: str
    visible_physical_cells: frozenset[Cell]
    newly_swept_physical_cells: frozenset[Cell]
    entropy_reduction_physical_cells: frozenset[Cell]
    discovery_opportunity_physical_cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_sha256(self.candidate_sha256, "candidate_sha256")
        _validate_sha256(
            self.physical_view_state_sha256, "physical_view_state_sha256"
        )
        visible = _cells(self.visible_physical_cells, "visible physical cell")
        swept = _cells(
            self.newly_swept_physical_cells, "newly swept physical cell"
        )
        entropy = _cells(
            self.entropy_reduction_physical_cells,
            "entropy-reduction physical cell",
        )
        discovery = _cells(
            self.discovery_opportunity_physical_cells,
            "discovery-opportunity physical cell",
        )
        if not swept <= visible or not entropy <= visible or not discovery <= visible:
            raise ValueError("predicted physical gain maps must be visible subsets")
        for name, value in (
            ("visible_physical_cells", visible),
            ("newly_swept_physical_cells", swept),
            ("entropy_reduction_physical_cells", entropy),
            ("discovery_opportunity_physical_cells", discovery),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_predicted_observation_v1",
            "candidate_sha256": self.candidate_sha256,
            "physical_view_state_sha256": self.physical_view_state_sha256,
            "visible_physical_cells": _cells_json(self.visible_physical_cells),
            "newly_swept_physical_cells": _cells_json(
                self.newly_swept_physical_cells
            ),
            "entropy_reduction_physical_cells": _cells_json(
                self.entropy_reduction_physical_cells
            ),
            "discovery_opportunity_physical_cells": _cells_json(
                self.discovery_opportunity_physical_cells
            ),
        }


@dataclass(frozen=True)
class TwoResolutionInformationGainScoreV2:
    candidate_sha256: str
    candidate_set_sha256: str
    physical_view_state_sha256: str
    visible_physical_cell_count: int
    newly_swept_physical_cell_count: int
    entropy_reduction_physical_cell_count: int
    discovery_opportunity_physical_cell_count: int
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
            "visible_physical_cell_count",
            "newly_swept_physical_cell_count",
            "entropy_reduction_physical_cell_count",
            "discovery_opportunity_physical_cell_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        for name in (
            "normalized_coverage_gain",
            "normalized_entropy_reduction",
            "uniform_discovery_opportunity",
            "normalized_path_cost",
            "normalized_turn_cost",
            "normalized_pose_uncertainty",
            "normalized_view_diversity",
            "normalized_staleness",
            "utility",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g4_v2_two_resolution_information_gain_score_v1",
            **{
                name: getattr(self, name)
                for name in (
                    "candidate_sha256",
                    "candidate_set_sha256",
                    "physical_view_state_sha256",
                    "visible_physical_cell_count",
                    "newly_swept_physical_cell_count",
                    "entropy_reduction_physical_cell_count",
                    "discovery_opportunity_physical_cell_count",
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

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise SnapshotBindingError("G4 V2 information-gain score was mutated")


def _path_receipt(path: ConfigurationPathV2) -> str:
    return _sha256(
        {
            "schema": "lewm_g4_v2_retained_configuration_path_receipt_v1",
            "snapshot_sha256": path.snapshot_sha256,
            "physical_map_frame_sha256": path.physical_map_frame_sha256,
            "configuration_map_frame_sha256": path.configuration_map_frame_sha256,
            "physical_revision": path.physical_revision,
            "configuration_revision": path.configuration_revision,
            "free_support_sha256": path.free_support_sha256,
            "occupied_support_sha256": path.occupied_support_sha256,
            "cells": _cells_json(path.cells),
            "cost": path.cost,
        }
    )


def _path_turn_cost(
    path: ConfigurationPathV2,
    current_yaw_rad: float,
    terminal_yaw_rad: float,
) -> float:
    if len(path.cells) == 1:
        return _wrapped_abs_delta(current_yaw_rad, terminal_yaw_rad)
    headings = [
        math.atan2(second[1] - first[1], second[0] - first[0])
        for first, second in zip(path.cells, path.cells[1:])
    ]
    cost = _wrapped_abs_delta(current_yaw_rad, headings[0])
    cost += sum(
        _wrapped_abs_delta(first, second)
        for first, second in zip(headings, headings[1:])
    )
    cost += _wrapped_abs_delta(headings[-1], terminal_yaw_rad)
    return cost


def _configuration_cell_has_new_view(
    cell: Cell,
    state: TwoResolutionPhysicalViewStateV2,
) -> bool:
    if all(
        row.configuration_cell != cell
        for row in state.configuration_view_history
    ):
        return True
    physical_children = {
        (2 * cell[0] + dx, 2 * cell[1] + dy)
        for dx in (0, 1)
        for dy in (0, 1)
    }
    return bool(physical_children & state.physical_discovery_opportunity_cells)


def _select_spatially_diverse_goals(
    component_cells: frozenset[Cell],
    *,
    start: Cell,
    frontier_cells: frozenset[Cell],
    state: TwoResolutionPhysicalViewStateV2,
    limit: int,
) -> tuple[Cell, ...]:
    if not component_cells or limit <= 0:
        return ()
    selected = [start]
    remaining = set(component_cells)
    remaining.discard(start)
    while remaining and len(selected) < limit:

        def rank(cell: Cell) -> tuple[int, int, int, int, int]:
            minimum_distance_sq = min(
                (cell[0] - prior[0]) ** 2 + (cell[1] - prior[1]) ** 2
                for prior in selected
            )
            return (
                -int(cell in frontier_cells),
                -int(_configuration_cell_has_new_view(cell, state)),
                -minimum_distance_sq,
                cell[0],
                cell[1],
            )

        chosen = min(remaining, key=rank)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


@dataclass(frozen=True)
class _CandidateSetIssuance:
    candidate_set: TwoResolutionFrontierViewpointCandidateSetV2
    state: TwoResolutionPhysicalViewStateV2
    component: ConfigurationComponentV2
    frontiers: ConfigurationFrontiersV2
    paths_by_candidate_id: dict[int, ConfigurationPathV2]


class TwoResolutionFrontierViewpointPlannerV2:
    """Issue, rank, select, and revalidate safe two-resolution view options."""

    def __init__(
        self,
        planner: TwoResolutionConfigurationPlannerV2,
        issuer: TwoResolutionPhysicalViewStateIssuerV2,
        *,
        config: TwoResolutionFrontierViewpointConfigV2 = DEFAULT_G4_V2_CONFIG,
    ) -> None:
        if type(planner) is not TwoResolutionConfigurationPlannerV2:
            raise TypeError("planner must be TwoResolutionConfigurationPlannerV2")
        if type(issuer) is not TwoResolutionPhysicalViewStateIssuerV2:
            raise TypeError("issuer must be TwoResolutionPhysicalViewStateIssuerV2")
        if type(config) is not TwoResolutionFrontierViewpointConfigV2:
            raise TypeError("config must be TwoResolutionFrontierViewpointConfigV2")
        config.assert_integrity()
        self._planner = planner
        self._issuer = issuer
        self._config = config
        self._issued_sets: dict[int, _CandidateSetIssuance] = {}
        self._issued_candidates: dict[
            int, TwoResolutionFrontierViewpointCandidateV2
        ] = {}
        self._score_cache: dict[
            int, tuple[TwoResolutionInformationGainScoreV2, ...]
        ] = {}

    @property
    def config(self) -> TwoResolutionFrontierViewpointConfigV2:
        return self._config

    def generate(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        *,
        start_configuration_cell: Sequence[int],
        current_yaw_rad: float,
    ) -> TwoResolutionFrontierViewpointCandidateSetV2:
        self._config.assert_integrity()
        self._issuer.validate_state(snapshot, state)
        start = _cell(start_configuration_cell, "start configuration cell")
        yaw = _finite(current_yaw_rad, "current_yaw_rad")
        component = self._planner.connected_component(snapshot, start)
        frontiers = self._planner.frontier_cells(snapshot, component)
        frontier_cells = frozenset(frontiers.cells)
        cell_limit = max(
            1, self._config.candidate_cap // self._config.yaw_bin_count
        )
        goals = _select_spatially_diverse_goals(
            component.cells,
            start=start,
            frontier_cells=frontier_cells,
            state=state,
            limit=cell_limit,
        )
        candidates: list[TwoResolutionFrontierViewpointCandidateV2] = []
        paths: dict[int, ConfigurationPathV2] = {}
        common = {
            **_binding_dict(snapshot, state),
            "component_sha256": component.content_sha256,
            "frontiers_sha256": frontiers.content_sha256,
            "config_sha256": self._config.content_sha256,
        }
        for goal in goals:
            path = self._planner.astar(snapshot, start, goal)
            if path is None:
                raise SnapshotBindingError(
                    "connected-component viewpoint has no current safe path"
                )
            self._planner.validate_path(snapshot, path)
            if any(cell not in snapshot.free_cells for cell in path.cells):
                raise SnapshotBindingError("G4 V2 path leaves configuration FREE")
            if goal in frontier_cells:
                kind = "frontier"
            elif goal == start:
                kind = "scan"
            else:
                kind = "coverage_frontier"
            for yaw_index in range(self._config.yaw_bin_count):
                candidate = TwoResolutionFrontierViewpointCandidateV2(
                    **common,
                    path_receipt_sha256=_path_receipt(path),
                    start_configuration_cell=start,
                    reachable_configuration_cell=goal,
                    yaw_index=yaw_index,
                    safe_configuration_path=path.cells,
                    path_cost_m=(
                        path.cost * self._config.configuration_cell_size_m
                    ),
                    turn_cost_rad=_path_turn_cost(
                        path,
                        yaw,
                        yaw_for_index_v2(yaw_index),
                    ),
                    kind=kind,
                )
                candidates.append(candidate)
                paths[id(candidate)] = path
                self._issued_candidates[id(candidate)] = candidate
        candidates.sort(
            key=lambda candidate: (
                candidate.reachable_configuration_cell,
                candidate.yaw_index,
            )
        )
        candidate_set = TwoResolutionFrontierViewpointCandidateSetV2(
            **common,
            start_configuration_cell=start,
            current_yaw_rad=yaw,
            candidates=tuple(candidates),
        )
        self._issued_sets[id(candidate_set)] = _CandidateSetIssuance(
            candidate_set=candidate_set,
            state=state,
            component=component,
            frontiers=frontiers,
            paths_by_candidate_id=paths,
        )
        return candidate_set

    def _validate_set(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
    ) -> _CandidateSetIssuance:
        self._config.assert_integrity()
        self._issuer.validate_state(snapshot, state)
        if type(candidate_set) is not TwoResolutionFrontierViewpointCandidateSetV2:
            raise TypeError(
                "candidate_set must be TwoResolutionFrontierViewpointCandidateSetV2"
            )
        issuance = self._issued_sets.get(id(candidate_set))
        if issuance is None or issuance.candidate_set is not candidate_set:
            raise SnapshotBindingError(
                "candidate set is not the exact live object issued by this planner"
            )
        if issuance.state is not state:
            raise SnapshotBindingError(
                "candidate set is bound to a different view state"
            )
        candidate_set.assert_integrity()
        expected = {
            **_binding_dict(snapshot, state),
            "component_sha256": issuance.component.content_sha256,
            "frontiers_sha256": issuance.frontiers.content_sha256,
            "config_sha256": self._config.content_sha256,
        }
        if any(
            getattr(candidate_set, name) != value
            for name, value in expected.items()
        ):
            raise SnapshotBindingError("candidate set has stale or foreign bindings")
        self._planner.validate_component(snapshot, issuance.component)
        self._planner.validate_frontiers(
            snapshot, issuance.component, issuance.frontiers
        )
        return issuance

    def validate_candidate(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
        candidate: TwoResolutionFrontierViewpointCandidateV2,
    ) -> None:
        issuance = self._validate_set(snapshot, state, candidate_set)
        self._validate_candidate_with_issuance(
            snapshot,
            state,
            candidate_set,
            candidate,
            issuance,
        )

    def _validate_candidate_with_issuance(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
        candidate: TwoResolutionFrontierViewpointCandidateV2,
        issuance: _CandidateSetIssuance,
        *,
        validate_retained_path: bool = True,
    ) -> None:
        if type(candidate) is not TwoResolutionFrontierViewpointCandidateV2:
            raise TypeError(
                "candidate must be TwoResolutionFrontierViewpointCandidateV2"
            )
        if self._issued_candidates.get(id(candidate)) is not candidate or not any(
            row is candidate for row in candidate_set.candidates
        ):
            raise SnapshotBindingError(
                "candidate is not the exact live member issued in this set"
            )
        candidate.assert_integrity()
        path = issuance.paths_by_candidate_id.get(id(candidate))
        if path is None:
            raise SnapshotBindingError("candidate has no retained G3 V2 path")
        if validate_retained_path:
            self._planner.validate_path(snapshot, path)
        expected = {
            **_binding_dict(snapshot, state),
            "component_sha256": issuance.component.content_sha256,
            "frontiers_sha256": issuance.frontiers.content_sha256,
            "config_sha256": self._config.content_sha256,
        }
        if any(getattr(candidate, name) != value for name, value in expected.items()):
            raise SnapshotBindingError("candidate has stale or foreign bindings")
        expected_path_cost = path.cost * self._config.configuration_cell_size_m
        expected_turn_cost = _path_turn_cost(
            path,
            candidate_set.current_yaw_rad,
            yaw_for_index_v2(candidate.yaw_index),
        )
        expected_kind = (
            "frontier"
            if candidate.reachable_configuration_cell in issuance.frontiers.cells
            else "scan"
            if candidate.reachable_configuration_cell
            == candidate_set.start_configuration_cell
            else "coverage_frontier"
        )
        if (
            candidate.path_receipt_sha256 != _path_receipt(path)
            or candidate.start_configuration_cell
            != candidate_set.start_configuration_cell
            or candidate.safe_configuration_path != path.cells
            or candidate.reachable_configuration_cell != path.cells[-1]
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
            or candidate.kind != expected_kind
            or any(
                cell not in snapshot.free_cells
                for cell in candidate.safe_configuration_path
            )
        ):
            raise SnapshotBindingError(
                "candidate differs from its retained safe option"
            )

    def _predict_unchecked(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate: TwoResolutionFrontierViewpointCandidateV2,
    ) -> TwoResolutionPredictedViewObservationV2:
        terminal_yaw = yaw_for_index_v2(candidate.yaw_index)
        body_world = configuration_cell_center_world_v2(
            snapshot, candidate.reachable_configuration_cell
        )
        camera_world = (
            body_world[0]
            + math.cos(terminal_yaw) * self._config.camera_forward_offset_m
            - math.sin(terminal_yaw) * self._config.camera_left_offset_m,
            body_world[1]
            + math.sin(terminal_yaw) * self._config.camera_forward_offset_m
            + math.cos(terminal_yaw) * self._config.camera_left_offset_m,
        )
        physical_origin = snapshot.physical_map_frame.origin_xy_m
        camera_physical = (
            (camera_world[0] - physical_origin[0]) / PHYSICAL_CELL_SIZE_M,
            (camera_world[1] - physical_origin[1]) / PHYSICAL_CELL_SIZE_M,
        )
        half_fov = math.radians(self._config.camera_horizontal_fov_deg) / 2.0
        offsets = tuple(
            -half_fov
            + 2.0 * half_fov * index / (self._config.ray_count - 1)
            for index in range(self._config.ray_count)
        )
        near_cells = self._config.camera_near_m / PHYSICAL_CELL_SIZE_M
        ground_near_cells = (
            self._config.ground_visible_min_range_m / PHYSICAL_CELL_SIZE_M
        )
        far_cells = self._config.view_range_m / PHYSICAL_CELL_SIZE_M
        visible: set[Cell] = set()
        for offset in offsets:
            angle = terminal_yaw + offset
            cosine, sine = math.cos(angle), math.sin(angle)
            start = (
                camera_physical[0] + cosine * near_cells,
                camera_physical[1] + sine * near_cells,
            )
            ground_start = (
                camera_physical[0] + cosine * ground_near_cells,
                camera_physical[1] + sine * ground_near_cells,
            )
            end = (
                camera_physical[0] + cosine * far_cells,
                camera_physical[1] + sine * far_cells,
            )
            groups = ordered_closed_physical_supercover_groups_v2(start, end)
            ground_groups = ordered_closed_physical_supercover_groups_v2(
                ground_start, end
            )
            eligible = frozenset(cell for group in ground_groups for cell in group)
            visible.update(
                _visible_physical_cells(groups, state, eligible_cells=eligible)
            )
        visible_cells = frozenset(visible)
        return TwoResolutionPredictedViewObservationV2(
            candidate_sha256=candidate.content_sha256,
            physical_view_state_sha256=state.content_sha256,
            visible_physical_cells=visible_cells,
            newly_swept_physical_cells=(
                visible_cells - state.visually_swept_physical_cells
            ),
            entropy_reduction_physical_cells=(
                visible_cells & state.physical_entropy_cells
            ),
            discovery_opportunity_physical_cells=(
                visible_cells & state.physical_discovery_opportunity_cells
            ),
        )

    def predict_observation(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
        candidate: TwoResolutionFrontierViewpointCandidateV2,
    ) -> TwoResolutionPredictedViewObservationV2:
        self.validate_candidate(snapshot, state, candidate_set, candidate)
        return self._predict_unchecked(snapshot, state, candidate)

    def _history_terms(
        self,
        candidate: TwoResolutionFrontierViewpointCandidateV2,
        state: TwoResolutionPhysicalViewStateV2,
    ) -> tuple[float, float]:
        if not state.configuration_view_history:
            return 1.0, 1.0
        terminal_yaw = yaw_for_index_v2(candidate.yaw_index)
        best_diversity = 1.0
        most_recent_weight = 0.0
        for row in state.configuration_view_history:
            spatial = math.hypot(
                candidate.reachable_configuration_cell[0]
                - row.configuration_cell[0],
                candidate.reachable_configuration_cell[1]
                - row.configuration_cell[1],
            )
            angular = _wrapped_abs_delta(
                terminal_yaw, yaw_for_index_v2(row.yaw_index)
            )
            diversity = min(
                1.0,
                0.5 * min(1.0, spatial / 4.0) + 0.5 * angular / math.pi,
            )
            best_diversity = min(best_diversity, diversity)
            age = max(0, state.view_step - row.last_observed_step)
            freshness = 1.0 - min(
                1.0, age / self._config.staleness_horizon_steps
            )
            most_recent_weight = max(
                most_recent_weight, (1.0 - diversity) * freshness
            )
        return best_diversity, 1.0 - most_recent_weight

    def score(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
    ) -> tuple[TwoResolutionInformationGainScoreV2, ...]:
        issuance = self._validate_set(snapshot, state, candidate_set)
        cached = self._score_cache.get(id(candidate_set))
        if cached is not None:
            for score in cached:
                score.assert_integrity()
            return cached
        maximum_ray_cells = max(
            1,
            int(math.ceil(self._config.view_range_m / PHYSICAL_CELL_SIZE_M))
            * self._config.ray_count,
        )
        entropy_denominator = max(1, len(state.physical_entropy_cells))
        discovery_denominator = max(
            1, len(state.physical_discovery_opportunity_cells)
        )
        path_denominator = max(
            self._config.view_range_m, CONFIGURATION_CELL_SIZE_M
        )
        base_pose_sigma = math.sqrt(state.pose_xy_variance_m2) / PHYSICAL_CELL_SIZE_M
        base_pose_sigma += math.sqrt(state.pose_yaw_variance_rad2) / (
            2.0 * math.pi / self._config.yaw_bin_count
        )
        scores: list[TwoResolutionInformationGainScoreV2] = []
        unique_paths = {
            id(path): path for path in issuance.paths_by_candidate_id.values()
        }
        for path in unique_paths.values():
            self._planner.validate_path(snapshot, path)
        for candidate in candidate_set.candidates:
            self._validate_candidate_with_issuance(
                snapshot,
                state,
                candidate_set,
                candidate,
                issuance,
                validate_retained_path=False,
            )
            observation = self._predict_unchecked(snapshot, state, candidate)
            coverage = min(
                1.0,
                len(observation.newly_swept_physical_cells)
                / maximum_ray_cells,
            )
            entropy = min(
                1.0,
                len(observation.entropy_reduction_physical_cells)
                / entropy_denominator,
            )
            discovery = min(
                1.0,
                len(observation.discovery_opportunity_physical_cells)
                / discovery_denominator,
            )
            path = min(1.0, candidate.path_cost_m / path_denominator)
            turn = min(1.0, candidate.turn_cost_rad / (2.0 * math.pi))
            motion_growth = 0.65 * path + 0.35 * turn
            pose = min(
                1.0,
                base_pose_sigma / 4.0 * (1.0 + motion_growth)
                + 0.25 * motion_growth,
            )
            diversity, staleness = self._history_terms(candidate, state)
            utility = (
                self._config.coverage_weight * coverage
                + self._config.entropy_weight * entropy
                + self._config.discovery_weight * discovery
                - self._config.path_cost_weight * path
                - self._config.turn_cost_weight * turn
                - self._config.pose_uncertainty_weight * pose
                + self._config.staleness_weight
                * (0.5 * diversity + 0.5 * staleness)
            )
            scores.append(
                TwoResolutionInformationGainScoreV2(
                    candidate_sha256=candidate.content_sha256,
                    candidate_set_sha256=candidate_set.content_sha256,
                    physical_view_state_sha256=state.content_sha256,
                    visible_physical_cell_count=len(
                        observation.visible_physical_cells
                    ),
                    newly_swept_physical_cell_count=len(
                        observation.newly_swept_physical_cells
                    ),
                    entropy_reduction_physical_cell_count=len(
                        observation.entropy_reduction_physical_cells
                    ),
                    discovery_opportunity_physical_cell_count=len(
                        observation.discovery_opportunity_physical_cells
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
        candidates = {
            candidate.content_sha256: candidate
            for candidate in candidate_set.candidates
        }
        scores.sort(
            key=lambda score: (
                -score.utility,
                candidates[score.candidate_sha256].reachable_configuration_cell,
                candidates[score.candidate_sha256].yaw_index,
            )
        )
        result = tuple(scores)
        self._score_cache[id(candidate_set)] = result
        return result

    def select(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        state: TwoResolutionPhysicalViewStateV2,
        candidate_set: TwoResolutionFrontierViewpointCandidateSetV2,
    ) -> TwoResolutionFrontierViewpointCandidateV2 | None:
        scores = self.score(snapshot, state, candidate_set)
        if not scores:
            return None
        by_hash = {
            candidate.content_sha256: candidate
            for candidate in candidate_set.candidates
        }
        selected = by_hash[scores[0].candidate_sha256]
        self.validate_candidate(snapshot, state, candidate_set, selected)
        return selected


__all__ = [
    "CONFIGURATION_CELL_SIZE_M",
    "DEFAULT_G4_V2_CONFIG",
    "G4_V2_CAMERA_FORWARD_OFFSET_M",
    "G4_V2_CAMERA_HORIZONTAL_FOV_DEG",
    "G4_V2_CAMERA_LEFT_OFFSET_M",
    "G4_V2_CAMERA_NEAR_M",
    "G4_V2_CAMERA_PITCH_RAD",
    "G4_V2_CAMERA_UP_OFFSET_M",
    "G4_V2_CAMERA_VERTICAL_FOV_DEG",
    "G4_V2_CANDIDATE_CAP",
    "G4_V2_GROUND_PLANE_Z_BODY_M",
    "G4_V2_RAY_COUNT",
    "G4_V2_VIEW_RANGE_M",
    "G4_V2_YAW_BIN_COUNT",
    "PHYSICAL_CELL_SIZE_M",
    "ConfigurationViewHistoryEntryV2",
    "TwoResolutionFrontierViewpointCandidateSetV2",
    "TwoResolutionFrontierViewpointCandidateV2",
    "TwoResolutionFrontierViewpointConfigV2",
    "TwoResolutionFrontierViewpointPlannerV2",
    "TwoResolutionInformationGainScoreV2",
    "TwoResolutionPhysicalViewStateIssuerV2",
    "TwoResolutionPhysicalViewStateV2",
    "TwoResolutionPredictedViewObservationV2",
    "configuration_cell_center_world_v2",
    "configuration_center_in_physical_grid_v2",
    "ordered_closed_physical_supercover_groups_v2",
    "physical_cell_center_world_v2",
    "yaw_for_index_v2",
]
