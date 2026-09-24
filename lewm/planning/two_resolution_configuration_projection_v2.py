"""Revision-bound 0.05 m evidence to 0.10 m configuration projection.

The canonical support and projection records in this module are the exact
records preregistered for G3 V2. Extra V4 and planning metadata lives only in
the profile envelope and therefore cannot replace those frozen identities.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from typing import Iterable, Mapping, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    InvalidConfigurationPathError,
    MapFrameIdentity,
    PhysicalLabel,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    StalePathError,
    StaleSnapshotError,
)


Cell = tuple[int, int]
Shape = tuple[int, int]
PHYSICAL_CELL_SIZE_M = 0.05
CONFIGURATION_CELL_SIZE_M = 0.10
FOOTPRINT_RADIUS_M = 0.47
PLANNING_CONNECTIVITY = 4
V4_SOURCE_SHAPE = (128, 128)
V4_SOURCE_CELL_SIZE_M = 0.05
FREE_SUPPORT_COUNT = 316
OCCUPIED_SUPPORT_COUNT = 276
FREE_SUPPORT_SHA256 = (
    "6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e"
)
OCCUPIED_SUPPORT_SHA256 = (
    "a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c"
)
PROFILE_SHA256 = (
    "2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314"
)
_EPS = 1e-12


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


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


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _shape(value: object, name: str) -> Shape:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer sequence")
    result = (
        _nonnegative_int(value[0], f"{name}[0]"),
        _nonnegative_int(value[1], f"{name}[1]"),
    )
    if result[0] == 0 or result[1] == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cell(value: object, name: str = "cell") -> Cell:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} coordinates must be integers")
    return int(value[0]), int(value[1])


def _cells(value: Iterable[object], name: str) -> frozenset[Cell]:
    try:
        return frozenset(_cell(cell, name) for cell in value)
    except TypeError:
        raise


def _cells_json(cells: Iterable[Cell]) -> list[list[int]]:
    return [[int(cell[0]), int(cell[1])] for cell in sorted(cells)]


def derive_cross_grid_supports() -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Derive the preregistered supports from metric floating-point geometry."""

    limit = 12
    free: list[Cell] = []
    occupied: list[Cell] = []
    for offset_x in range(-limit, limit + 1):
        for offset_y in range(-limit, limit + 1):
            center_x = (float(offset_x) - 0.5) * PHYSICAL_CELL_SIZE_M
            center_y = (float(offset_y) - 0.5) * PHYSICAL_CELL_SIZE_M
            nearest_x = max(abs(center_x) - 0.5 * PHYSICAL_CELL_SIZE_M, 0.0)
            nearest_y = max(abs(center_y) - 0.5 * PHYSICAL_CELL_SIZE_M, 0.0)
            if nearest_x**2 + nearest_y**2 <= FOOTPRINT_RADIUS_M**2 + _EPS:
                free.append((offset_x, offset_y))
            if center_x**2 + center_y**2 <= FOOTPRINT_RADIUS_M**2 + _EPS:
                occupied.append((offset_x, offset_y))
    return tuple(sorted(free)), tuple(sorted(occupied))


def _kernel_common() -> dict[str, object]:
    return {
        "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
        "footprint_radius_m": FOOTPRINT_RADIUS_M,
        "inclusive_boundary": True,
        "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
        "physical_index_rule": "(2*cx+dx,2*cy+dy)",
        "shared_origin_cell_boundaries": True,
    }


def _free_support_core(offsets: Sequence[Cell]) -> dict[str, object]:
    return {
        **_kernel_common(),
        "offsets": _cells_json(offsets),
        "schema": "lewm_g3_v2_cross_grid_free_closed_square_intersection_kernel_v1",
    }


def _occupied_support_core(offsets: Sequence[Cell]) -> dict[str, object]:
    return {
        **_kernel_common(),
        "offsets": _cells_json(offsets),
        "schema": "lewm_g3_v2_cross_grid_occupied_center_inside_disc_kernel_v1",
    }


def _projection_contract_core(
    *,
    free_support_sha256: str,
    occupied_support_sha256: str,
) -> dict[str, object]:
    return {
        "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
        "footprint_radius_m": FOOTPRINT_RADIUS_M,
        "free_support_count": FREE_SUPPORT_COUNT,
        "free_support_sha256": free_support_sha256,
        "occupied_precedes_free": True,
        "occupied_support_count": OCCUPIED_SUPPORT_COUNT,
        "occupied_support_sha256": occupied_support_sha256,
        "otherwise": "unknown",
        "out_of_domain_support": "occupied",
        "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
        "physical_shape_per_configuration_cell": [2, 2],
        "schema": "lewm_g3_v2_two_resolution_configuration_projection_v1",
        "shared_origin_cell_boundaries": True,
    }


@dataclass(frozen=True)
class TwoResolutionProfileV2:
    free_support_offsets: tuple[Cell, ...] = field(init=False)
    occupied_support_offsets: tuple[Cell, ...] = field(init=False)
    free_support_sha256: str = field(init=False)
    occupied_support_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        free, occupied = derive_cross_grid_supports()
        free_hash = _sha256(_free_support_core(free))
        occupied_hash = _sha256(_occupied_support_core(occupied))
        profile_hash = _sha256(
            _projection_contract_core(
                free_support_sha256=free_hash,
                occupied_support_sha256=occupied_hash,
            )
        )
        if (
            len(free) != FREE_SUPPORT_COUNT
            or len(occupied) != OCCUPIED_SUPPORT_COUNT
            or free_hash != FREE_SUPPORT_SHA256
            or occupied_hash != OCCUPIED_SUPPORT_SHA256
            or profile_hash != PROFILE_SHA256
        ):
            raise AssertionError("preregistered G3 V2 cross-grid morphology changed")
        object.__setattr__(self, "free_support_offsets", free)
        object.__setattr__(self, "occupied_support_offsets", occupied)
        object.__setattr__(self, "free_support_sha256", free_hash)
        object.__setattr__(self, "occupied_support_sha256", occupied_hash)
        object.__setattr__(self, "content_sha256", profile_hash)

    def assert_integrity(self) -> None:
        free, occupied = derive_cross_grid_supports()
        expected = (
            free,
            occupied,
            _sha256(_free_support_core(free)),
            _sha256(_occupied_support_core(occupied)),
        )
        actual = (
            self.free_support_offsets,
            self.occupied_support_offsets,
            self.free_support_sha256,
            self.occupied_support_sha256,
        )
        if actual != expected or self.content_sha256 != PROFILE_SHA256:
            raise SnapshotBindingError("G3 V2 profile/support state was mutated")
        if (
            expected[2] != FREE_SUPPORT_SHA256
            or expected[3] != OCCUPIED_SUPPORT_SHA256
            or _sha256(
                _projection_contract_core(
                    free_support_sha256=expected[2],
                    occupied_support_sha256=expected[3],
                )
            )
            != PROFILE_SHA256
        ):
            raise SnapshotBindingError("G3 V2 canonical profile identity changed")

    def projection_contract_core(self) -> dict[str, object]:
        self.assert_integrity()
        return _projection_contract_core(
            free_support_sha256=self.free_support_sha256,
            occupied_support_sha256=self.occupied_support_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        self.assert_integrity()
        return {
            "schema": "lewm_g3_v2_two_resolution_profile_envelope_v1",
            "projection_contract_core": self.projection_contract_core(),
            "projection_contract_sha256": self.content_sha256,
            "free_support_kernel": _free_support_core(self.free_support_offsets),
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_kernel": _occupied_support_core(
                self.occupied_support_offsets
            ),
            "occupied_support_sha256": self.occupied_support_sha256,
            "planning_connectivity": PLANNING_CONNECTIVITY,
            "allow_diagonal_corner_cutting": False,
            "v4_source_shape": list(V4_SOURCE_SHAPE),
            "v4_source_cell_size_m": V4_SOURCE_CELL_SIZE_M,
            "full_square_physical_labels": True,
            "production_promotion_authorized": False,
        }


FIXED_PROFILE_V2 = TwoResolutionProfileV2()


def assert_fixed_profile_integrity(
    profile: TwoResolutionProfileV2 = FIXED_PROFILE_V2,
) -> None:
    if type(profile) is not TwoResolutionProfileV2:
        raise TypeError("G3 V2 profile must have the exact canonical type")
    profile.assert_integrity()


def physical_index_for_configuration_offset(
    configuration_cell: Sequence[int],
    offset: Sequence[int],
) -> Cell:
    cell = _cell(configuration_cell, "configuration_cell")
    delta = _cell(offset, "support offset")
    return 2 * cell[0] + delta[0], 2 * cell[1] + delta[1]


def _frame_from_mapping(value: object, name: str) -> MapFrameIdentity:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "session_id",
        "origin_xy_m",
        "cell_size_m",
        "frame_id",
    }:
        raise ValueError(f"{name} is not a canonical map-frame record")
    if value["schema"] != "lewm_g3_map_frame_identity_v1":
        raise ValueError(f"{name} schema changed")
    return MapFrameIdentity(
        session_id=value["session_id"],
        origin_xy_m=value["origin_xy_m"],
        cell_size_m=value["cell_size_m"],
        frame_id=value["frame_id"],
    )


def _execution_block_receipt_sha256(
    *,
    physical_revision: int,
    physical_cells: Iterable[Cell],
    configuration_cells: Iterable[Cell],
) -> str:
    return _sha256(
        {
            "schema": "lewm_g3_v2_execution_block_projection_receipt_v1",
            "physical_revision": physical_revision,
            "physical_execution_block_cells": _cells_json(physical_cells),
            "configuration_execution_block_cells": _cells_json(
                configuration_cells
            ),
            "index_rule": "configuration=(physical_x//2,physical_y//2)",
            "double_dilation": False,
            "configuration_center_precedence": "occupied",
        }
    )


@dataclass(frozen=True)
class TwoResolutionConfigurationSnapshotV2:
    physical_map_frame: MapFrameIdentity
    configuration_map_frame: MapFrameIdentity
    memory_config_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_content_sha256: str
    projection_source_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    physical_shape: Shape
    configuration_shape: Shape
    planning_connectivity: int
    allow_diagonal_corner_cutting: bool
    physical_execution_block_cells: frozenset[Cell]
    configuration_execution_block_cells: frozenset[Cell]
    execution_block_receipt_sha256: str
    free_cells: frozenset[Cell]
    occupied_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    exact_sim_tainted: bool
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        assert_fixed_profile_integrity()
        if type(self.physical_map_frame) is not MapFrameIdentity or type(
            self.configuration_map_frame
        ) is not MapFrameIdentity:
            raise TypeError("snapshot map frames must have the canonical type")
        for name in (
            "memory_config_sha256",
            "physical_content_sha256",
            "projection_source_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "execution_block_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        physical_revision = _nonnegative_int(
            self.physical_revision, "physical_revision"
        )
        configuration_revision = _nonnegative_int(
            self.configuration_revision, "configuration_revision"
        )
        if configuration_revision == 0:
            raise ValueError("configuration_revision must identify an issued projection")
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape, "configuration_shape"
        )
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("snapshot physical shape must be exactly 2x per axis")
        physical_frame = self.physical_map_frame
        configuration_frame = self.configuration_map_frame
        if (
            not math.isclose(
                physical_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                configuration_frame.cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or physical_frame.origin_xy_m != configuration_frame.origin_xy_m
            or physical_frame.content_sha256 == configuration_frame.content_sha256
            or physical_frame.frame_id == configuration_frame.frame_id
        ):
            raise ValueError("snapshot lattice frame alignment changed")
        if (
            self.profile_sha256 != PROFILE_SHA256
            or self.free_support_sha256 != FREE_SUPPORT_SHA256
            or self.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
            or self.planning_connectivity != PLANNING_CONNECTIVITY
            or type(self.allow_diagonal_corner_cutting) is not bool
            or self.allow_diagonal_corner_cutting
        ):
            raise ValueError("snapshot projection/planning profile changed")
        if type(self.exact_sim_tainted) is not bool:
            raise TypeError("exact_sim_tainted must be boolean")
        free = _cells(self.free_cells, "free cell")
        occupied = _cells(self.occupied_cells, "occupied cell")
        unknown = _cells(self.unknown_cells, "unknown cell")
        physical_blocks = _cells(
            self.physical_execution_block_cells,
            "physical execution-block cell",
        )
        configuration_blocks = _cells(
            self.configuration_execution_block_cells,
            "configuration execution-block cell",
        )
        expected_configuration_blocks = frozenset(
            (cell[0] // 2, cell[1] // 2)
            for cell in physical_blocks
            if 0 <= cell[0] // 2 < configuration_shape[0]
            and 0 <= cell[1] // 2 < configuration_shape[1]
        )
        if configuration_blocks != expected_configuration_blocks:
            raise ValueError("execution blocks changed exact-centre projection")
        if self.execution_block_receipt_sha256 != _execution_block_receipt_sha256(
            physical_revision=physical_revision,
            physical_cells=physical_blocks,
            configuration_cells=configuration_blocks,
        ):
            raise SnapshotBindingError("execution-block projection receipt mismatch")
        domain = frozenset(
            (x, y)
            for x in range(configuration_shape[0])
            for y in range(configuration_shape[1])
        )
        if (
            free & occupied
            or free & unknown
            or occupied & unknown
            or free | occupied | unknown != domain
            or not configuration_blocks <= occupied
        ):
            raise ValueError("snapshot must partition the complete configuration raster")
        object.__setattr__(self, "physical_revision", physical_revision)
        object.__setattr__(self, "configuration_revision", configuration_revision)
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        object.__setattr__(self, "free_cells", free)
        object.__setattr__(self, "occupied_cells", occupied)
        object.__setattr__(self, "unknown_cells", unknown)
        object.__setattr__(
            self, "physical_execution_block_cells", physical_blocks
        )
        object.__setattr__(
            self,
            "configuration_execution_block_cells",
            configuration_blocks,
        )
        object.__setattr__(self, "content_sha256", _sha256(self._core_dict()))

    @property
    def physical_map_frame_sha256(self) -> str:
        return self.physical_map_frame.content_sha256

    @property
    def configuration_map_frame_sha256(self) -> str:
        return self.configuration_map_frame.content_sha256

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def evaluated_cells(self) -> frozenset[Cell]:
        return self.free_cells | self.occupied_cells | self.unknown_cells

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_v2_two_resolution_configuration_snapshot_v2",
            "physical_map_frame": self.physical_map_frame.to_dict(),
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame": self.configuration_map_frame.to_dict(),
            "configuration_map_frame_sha256": (
                self.configuration_map_frame_sha256
            ),
            "memory_config_sha256": self.memory_config_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "physical_content_sha256": self.physical_content_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_cells_per_configuration_axis": [2, 2],
            "planning_connectivity": self.planning_connectivity,
            "allow_diagonal_corner_cutting": self.allow_diagonal_corner_cutting,
            "physical_execution_block_cells": _cells_json(
                self.physical_execution_block_cells
            ),
            "configuration_execution_block_cells": _cells_json(
                self.configuration_execution_block_cells
            ),
            "execution_block_receipt_sha256": (
                self.execution_block_receipt_sha256
            ),
            "free_cells": _cells_json(self.free_cells),
            "occupied_cells": _cells_json(self.occupied_cells),
            "unknown_cells": _cells_json(self.unknown_cells),
            "exact_sim_tainted": self.exact_sim_tainted,
            "production_promotion_authorized": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "snapshot_content_sha256": self.content_sha256}

    def serialize(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "TwoResolutionConfigurationSnapshotV2":
        if not isinstance(value, dict):
            raise TypeError("snapshot mapping must be a dict")
        expected_keys = {
            "schema",
            "physical_map_frame",
            "physical_map_frame_sha256",
            "configuration_map_frame",
            "configuration_map_frame_sha256",
            "memory_config_sha256",
            "physical_revision",
            "configuration_revision",
            "physical_content_sha256",
            "projection_source_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "physical_shape",
            "configuration_shape",
            "physical_cells_per_configuration_axis",
            "planning_connectivity",
            "allow_diagonal_corner_cutting",
            "physical_execution_block_cells",
            "configuration_execution_block_cells",
            "execution_block_receipt_sha256",
            "free_cells",
            "occupied_cells",
            "unknown_cells",
            "exact_sim_tainted",
            "production_promotion_authorized",
            "snapshot_content_sha256",
        }
        if set(value) != expected_keys:
            raise ValueError("snapshot mapping keys changed")
        if (
            value["schema"]
            != "lewm_g3_v2_two_resolution_configuration_snapshot_v2"
            or value["physical_cells_per_configuration_axis"] != [2, 2]
            or value["production_promotion_authorized"] is not False
        ):
            raise ValueError("snapshot serialized contract changed")
        snapshot = cls(
            physical_map_frame=_frame_from_mapping(
                value["physical_map_frame"], "physical_map_frame"
            ),
            configuration_map_frame=_frame_from_mapping(
                value["configuration_map_frame"], "configuration_map_frame"
            ),
            memory_config_sha256=value["memory_config_sha256"],
            physical_revision=value["physical_revision"],
            configuration_revision=value["configuration_revision"],
            physical_content_sha256=value["physical_content_sha256"],
            projection_source_sha256=value["projection_source_sha256"],
            profile_sha256=value["profile_sha256"],
            free_support_sha256=value["free_support_sha256"],
            occupied_support_sha256=value["occupied_support_sha256"],
            physical_shape=value["physical_shape"],
            configuration_shape=value["configuration_shape"],
            planning_connectivity=value["planning_connectivity"],
            allow_diagonal_corner_cutting=value["allow_diagonal_corner_cutting"],
            physical_execution_block_cells=frozenset(
                tuple(cell) for cell in value["physical_execution_block_cells"]
            ),
            configuration_execution_block_cells=frozenset(
                tuple(cell)
                for cell in value["configuration_execution_block_cells"]
            ),
            execution_block_receipt_sha256=value[
                "execution_block_receipt_sha256"
            ],
            free_cells=frozenset(tuple(cell) for cell in value["free_cells"]),
            occupied_cells=frozenset(tuple(cell) for cell in value["occupied_cells"]),
            unknown_cells=frozenset(tuple(cell) for cell in value["unknown_cells"]),
            exact_sim_tainted=value["exact_sim_tainted"],
        )
        if (
            value["physical_map_frame_sha256"]
            != snapshot.physical_map_frame_sha256
            or value["configuration_map_frame_sha256"]
            != snapshot.configuration_map_frame_sha256
            or value["snapshot_content_sha256"] != snapshot.content_sha256
        ):
            raise SnapshotBindingError("snapshot serialized hashes do not match")
        return snapshot

    @classmethod
    def deserialize(cls, encoded: bytes) -> "TwoResolutionConfigurationSnapshotV2":
        if not isinstance(encoded, bytes):
            raise TypeError("serialized snapshot must be bytes")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("serialized snapshot is not UTF-8 JSON") from exc
        if encoded != _canonical_bytes(value) + b"\n":
            raise ValueError("serialized snapshot is not canonical JSON")
        snapshot = cls.from_mapping(value)
        if snapshot.serialize() != encoded:
            raise ValueError("serialized snapshot changed after typed replay")
        return snapshot

    def assert_integrity(self) -> None:
        expected = TwoResolutionConfigurationSnapshotV2.from_mapping(self.to_dict())
        if expected.content_sha256 != self.content_sha256:
            raise SnapshotBindingError("G3 V2 snapshot was mutated")

    def state(self, cell: Sequence[int]) -> PhysicalLabel:
        normalized = _cell(cell)
        if normalized in self.free_cells:
            return PhysicalLabel.FREE
        if normalized in self.occupied_cells:
            return PhysicalLabel.OCCUPIED
        return PhysicalLabel.UNKNOWN


class TwoResolutionConfigurationProjectionV2:
    def __init__(
        self,
        memory: RevisionedPhysicalMemory,
        *,
        configuration_map_frame: MapFrameIdentity,
        physical_shape: Sequence[int],
        configuration_shape: Sequence[int],
    ) -> None:
        assert_fixed_profile_integrity()
        if type(memory) is not RevisionedPhysicalMemory:
            raise TypeError("memory must be RevisionedPhysicalMemory")
        if type(configuration_map_frame) is not MapFrameIdentity:
            raise TypeError("configuration_map_frame must be MapFrameIdentity")
        self._memory = memory
        self._profile = FIXED_PROFILE_V2
        self._configuration_map_frame = configuration_map_frame
        self._physical_shape = _shape(physical_shape, "physical_shape")
        self._configuration_shape = _shape(
            configuration_shape, "configuration_shape"
        )
        self._configuration_revision = 0
        self._issued_snapshot: TwoResolutionConfigurationSnapshotV2 | None = None
        self._projection_source_sha256 = self._derive_projection_source_sha256()
        self._assert_setup_integrity()

    @property
    def configuration_revision(self) -> int:
        return self._configuration_revision

    @property
    def projection_source_sha256(self) -> str:
        return self._projection_source_sha256

    @property
    def configuration_map_frame(self) -> MapFrameIdentity:
        return self._configuration_map_frame

    def _derive_projection_source_sha256(self) -> str:
        return _sha256(
            {
                "schema": "lewm_g3_v2_projection_source_identity_v1",
                "physical_map_frame": self._memory.map_frame.to_dict(),
                "physical_map_frame_sha256": self._memory.map_frame.content_sha256,
                "configuration_map_frame": self._configuration_map_frame.to_dict(),
                "configuration_map_frame_sha256": (
                    self._configuration_map_frame.content_sha256
                ),
                "memory_config_sha256": self._memory.config.content_sha256,
                "physical_shape": list(self._physical_shape),
                "configuration_shape": list(self._configuration_shape),
                "projection_contract_sha256": PROFILE_SHA256,
                "free_support_sha256": FREE_SUPPORT_SHA256,
                "occupied_support_sha256": OCCUPIED_SUPPORT_SHA256,
            }
        )

    def _assert_setup_integrity(self) -> None:
        assert_fixed_profile_integrity(self._profile)
        memory = self._memory
        physical_frame = memory.map_frame
        configuration_frame = self._configuration_map_frame
        if memory.config.promoted_runtime:
            raise PermissionError("G3 V2 projection is not promotion-authorized")
        if (
            memory.config.planning_connectivity != PLANNING_CONNECTIVITY
            or memory.config.allow_diagonal_corner_cutting
            or memory.config.physical_projection_contract_sha256 != PROFILE_SHA256
        ):
            raise ValueError("G3 V2 physical memory profile changed")
        if (
            not math.isclose(
                physical_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                configuration_frame.cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or physical_frame.origin_xy_m != configuration_frame.origin_xy_m
            or physical_frame.content_sha256 == configuration_frame.content_sha256
            or physical_frame.frame_id == configuration_frame.frame_id
        ):
            raise ValueError("G3 V2 physical/configuration frame alignment changed")
        if self._physical_shape != (
            2 * self._configuration_shape[0],
            2 * self._configuration_shape[1],
        ):
            raise ValueError("G3 V2 physical shape must be exactly 2x per axis")
        if self._derive_projection_source_sha256() != self._projection_source_sha256:
            raise SnapshotBindingError("G3 V2 projection source identity changed")

    def project(self) -> TwoResolutionConfigurationSnapshotV2:
        self._assert_setup_integrity()

        physical_execution_blocks = frozenset(self._memory.execution_block_cells)
        configuration_execution_blocks = frozenset(
            (cell[0] // 2, cell[1] // 2)
            for cell in physical_execution_blocks
            if 0 <= cell[0] // 2 < self._configuration_shape[0]
            and 0 <= cell[1] // 2 < self._configuration_shape[1]
        )

        def state(cell: Cell) -> PhysicalLabel:
            if not (
                0 <= cell[0] < self._physical_shape[0]
                and 0 <= cell[1] < self._physical_shape[1]
            ):
                return PhysicalLabel.OCCUPIED
            return self._memory.physical_state(cell)

        free: set[Cell] = set()
        occupied: set[Cell] = set()
        unknown: set[Cell] = set()
        for cx in range(self._configuration_shape[0]):
            for cy in range(self._configuration_shape[1]):
                cell = (cx, cy)
                if cell in configuration_execution_blocks or any(
                    state(physical_index_for_configuration_offset(cell, offset))
                    is PhysicalLabel.OCCUPIED
                    for offset in self._profile.occupied_support_offsets
                ):
                    occupied.add(cell)
                elif all(
                    state(physical_index_for_configuration_offset(cell, offset))
                    is PhysicalLabel.FREE
                    for offset in self._profile.free_support_offsets
                ):
                    free.add(cell)
                else:
                    unknown.add(cell)
        self._configuration_revision += 1
        snapshot = TwoResolutionConfigurationSnapshotV2(
            physical_map_frame=self._memory.map_frame,
            configuration_map_frame=self._configuration_map_frame,
            memory_config_sha256=self._memory.config.content_sha256,
            physical_revision=self._memory.revision,
            configuration_revision=self._configuration_revision,
            physical_content_sha256=self._memory.physical_content_sha256,
            projection_source_sha256=self._projection_source_sha256,
            profile_sha256=self._profile.content_sha256,
            free_support_sha256=self._profile.free_support_sha256,
            occupied_support_sha256=self._profile.occupied_support_sha256,
            physical_shape=self._physical_shape,
            configuration_shape=self._configuration_shape,
            planning_connectivity=PLANNING_CONNECTIVITY,
            allow_diagonal_corner_cutting=False,
            physical_execution_block_cells=physical_execution_blocks,
            configuration_execution_block_cells=configuration_execution_blocks,
            execution_block_receipt_sha256=_execution_block_receipt_sha256(
                physical_revision=self._memory.revision,
                physical_cells=physical_execution_blocks,
                configuration_cells=configuration_execution_blocks,
            ),
            free_cells=frozenset(free),
            occupied_cells=frozenset(occupied),
            unknown_cells=frozenset(unknown),
            exact_sim_tainted=self._memory.exact_sim_tainted,
        )
        self._issued_snapshot = snapshot
        return snapshot

    def assert_current_snapshot(
        self, snapshot: TwoResolutionConfigurationSnapshotV2
    ) -> None:
        self._assert_setup_integrity()
        if type(snapshot) is not TwoResolutionConfigurationSnapshotV2:
            raise TypeError("snapshot must be TwoResolutionConfigurationSnapshotV2")
        if snapshot is not self._issued_snapshot:
            raise SnapshotBindingError(
                "snapshot is not the exact live object issued by this projection"
            )
        snapshot.assert_integrity()
        if snapshot.physical_revision != self._memory.revision:
            raise StaleSnapshotError(
                f"snapshot physical revision {snapshot.physical_revision} is stale; "
                f"memory is at {self._memory.revision}"
            )
        if snapshot.configuration_revision != self._configuration_revision:
            raise StaleSnapshotError(
                "snapshot configuration revision is not the current projection"
            )
        if (
            snapshot.physical_map_frame_sha256
            != self._memory.map_frame.content_sha256
            or snapshot.configuration_map_frame_sha256
            != self._configuration_map_frame.content_sha256
            or snapshot.memory_config_sha256 != self._memory.config.content_sha256
            or snapshot.physical_content_sha256
            != self._memory.physical_content_sha256
            or snapshot.projection_source_sha256
            != self._projection_source_sha256
            or snapshot.profile_sha256 != PROFILE_SHA256
            or snapshot.free_support_sha256 != FREE_SUPPORT_SHA256
            or snapshot.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
            or snapshot.physical_shape != self._physical_shape
            or snapshot.configuration_shape != self._configuration_shape
            or snapshot.exact_sim_tainted is not self._memory.exact_sim_tainted
        ):
            raise SnapshotBindingError("snapshot does not bind the live V2 projection")


@dataclass(frozen=True)
class ConfigurationComponentV2:
    snapshot_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    start_cell: Cell
    cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        physical_revision = _nonnegative_int(
            self.physical_revision, "physical_revision"
        )
        configuration_revision = _nonnegative_int(
            self.configuration_revision, "configuration_revision"
        )
        start = _cell(self.start_cell, "start_cell")
        cells = _cells(self.cells, "component cell")
        object.__setattr__(self, "physical_revision", physical_revision)
        object.__setattr__(self, "configuration_revision", configuration_revision)
        object.__setattr__(self, "start_cell", start)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(
                {
                    "schema": "lewm_g3_v2_configuration_component_v1",
                    "snapshot_sha256": self.snapshot_sha256,
                    "physical_map_frame_sha256": self.physical_map_frame_sha256,
                    "configuration_map_frame_sha256": (
                        self.configuration_map_frame_sha256
                    ),
                    "physical_revision": physical_revision,
                    "configuration_revision": configuration_revision,
                    "free_support_sha256": self.free_support_sha256,
                    "occupied_support_sha256": self.occupied_support_sha256,
                    "start_cell": list(start),
                    "cells": _cells_json(cells),
                }
            ),
        )

    def assert_integrity(self) -> None:
        expected = ConfigurationComponentV2(
            snapshot_sha256=self.snapshot_sha256,
            physical_map_frame_sha256=self.physical_map_frame_sha256,
            configuration_map_frame_sha256=self.configuration_map_frame_sha256,
            physical_revision=self.physical_revision,
            configuration_revision=self.configuration_revision,
            free_support_sha256=self.free_support_sha256,
            occupied_support_sha256=self.occupied_support_sha256,
            start_cell=self.start_cell,
            cells=self.cells,
        )
        if expected.content_sha256 != self.content_sha256:
            raise SnapshotBindingError("V2 component was mutated")


@dataclass(frozen=True)
class ConfigurationFrontiersV2:
    snapshot_sha256: str
    component_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    cells: tuple[Cell, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "component_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        physical_revision = _nonnegative_int(
            self.physical_revision, "physical_revision"
        )
        configuration_revision = _nonnegative_int(
            self.configuration_revision, "configuration_revision"
        )
        cells = tuple(_cell(cell, "frontier cell") for cell in self.cells)
        if cells != tuple(sorted(set(cells))):
            raise ValueError("frontier cells must be unique and deterministic")
        object.__setattr__(self, "physical_revision", physical_revision)
        object.__setattr__(self, "configuration_revision", configuration_revision)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(
                {
                    "schema": "lewm_g3_v2_configuration_frontiers_v1",
                    "snapshot_sha256": self.snapshot_sha256,
                    "component_sha256": self.component_sha256,
                    "physical_map_frame_sha256": self.physical_map_frame_sha256,
                    "configuration_map_frame_sha256": (
                        self.configuration_map_frame_sha256
                    ),
                    "physical_revision": physical_revision,
                    "configuration_revision": configuration_revision,
                    "free_support_sha256": self.free_support_sha256,
                    "occupied_support_sha256": self.occupied_support_sha256,
                    "cells": _cells_json(cells),
                }
            ),
        )

    def assert_integrity(self) -> None:
        expected = ConfigurationFrontiersV2(
            snapshot_sha256=self.snapshot_sha256,
            component_sha256=self.component_sha256,
            physical_map_frame_sha256=self.physical_map_frame_sha256,
            configuration_map_frame_sha256=self.configuration_map_frame_sha256,
            physical_revision=self.physical_revision,
            configuration_revision=self.configuration_revision,
            free_support_sha256=self.free_support_sha256,
            occupied_support_sha256=self.occupied_support_sha256,
            cells=self.cells,
        )
        if expected.content_sha256 != self.content_sha256:
            raise SnapshotBindingError("V2 frontier artifact was mutated")


@dataclass(frozen=True)
class ConfigurationPathV2:
    snapshot_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    cells: tuple[Cell, ...]
    cost: float

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        object.__setattr__(
            self,
            "configuration_revision",
            _nonnegative_int(self.configuration_revision, "configuration_revision"),
        )
        cells = tuple(_cell(cell, "path cell") for cell in self.cells)
        if not cells:
            raise ValueError("configuration path cannot be empty")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise TypeError("path cost must be numeric")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("path cost must be finite and non-negative")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "cost", cost)


class TwoResolutionConfigurationPlannerV2:
    def __init__(self, projection: TwoResolutionConfigurationProjectionV2) -> None:
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("planner requires the exact V2 projection type")
        self._projection = projection
        self._issued_components: dict[int, ConfigurationComponentV2] = {}
        self._issued_frontiers: dict[int, ConfigurationFrontiersV2] = {}
        self._issued_paths: dict[int, ConfigurationPathV2] = {}

    def _validate(self, snapshot: TwoResolutionConfigurationSnapshotV2) -> None:
        assert_fixed_profile_integrity()
        self._projection.assert_current_snapshot(snapshot)

    @staticmethod
    def _neighbors(cell: Cell) -> tuple[Cell, ...]:
        x, y = cell
        return ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y))

    @staticmethod
    def _result_binding(
        snapshot: TwoResolutionConfigurationSnapshotV2,
    ) -> dict[str, object]:
        return {
            "snapshot_sha256": snapshot.content_sha256,
            "physical_map_frame_sha256": snapshot.physical_map_frame_sha256,
            "configuration_map_frame_sha256": (
                snapshot.configuration_map_frame_sha256
            ),
            "physical_revision": snapshot.physical_revision,
            "configuration_revision": snapshot.configuration_revision,
            "free_support_sha256": snapshot.free_support_sha256,
            "occupied_support_sha256": snapshot.occupied_support_sha256,
        }

    def connected_component(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        start: Cell,
    ) -> ConfigurationComponentV2:
        self._validate(snapshot)
        normalized = _cell(start, "start")
        queue: deque[Cell] = deque()
        seen: set[Cell] = set()
        if normalized in snapshot.free_cells:
            queue.append(normalized)
            seen.add(normalized)
        while queue:
            current = queue.popleft()
            for neighbor in self._neighbors(current):
                if neighbor in snapshot.free_cells and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        component = ConfigurationComponentV2(
            **self._result_binding(snapshot),
            start_cell=normalized,
            cells=frozenset(seen),
        )
        self._issued_components[id(component)] = component
        return component

    def validate_component(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
    ) -> None:
        self._validate(snapshot)
        if type(component) is not ConfigurationComponentV2:
            raise TypeError("component must be ConfigurationComponentV2")
        if self._issued_components.get(id(component)) is not component:
            raise SnapshotBindingError(
                "component is not the exact live object issued by this planner"
            )
        component.assert_integrity()
        if (
            component.snapshot_sha256 != snapshot.content_sha256
            or component.physical_map_frame_sha256
            != snapshot.physical_map_frame_sha256
            or component.configuration_map_frame_sha256
            != snapshot.configuration_map_frame_sha256
            or component.physical_revision != snapshot.physical_revision
            or component.configuration_revision != snapshot.configuration_revision
            or component.free_support_sha256 != snapshot.free_support_sha256
            or component.occupied_support_sha256
            != snapshot.occupied_support_sha256
            or not component.cells <= snapshot.free_cells
        ):
            raise SnapshotBindingError("component is not bound to the snapshot")
        expected_cells: set[Cell] = set()
        if component.start_cell in snapshot.free_cells:
            queue: deque[Cell] = deque([component.start_cell])
            expected_cells.add(component.start_cell)
            while queue:
                current = queue.popleft()
                for neighbor in self._neighbors(current):
                    if (
                        neighbor in snapshot.free_cells
                        and neighbor not in expected_cells
                    ):
                        expected_cells.add(neighbor)
                        queue.append(neighbor)
        if component.cells != frozenset(expected_cells):
            raise SnapshotBindingError(
                "component is not the complete connected snapshot component"
            )

    def frontier_cells(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
    ) -> ConfigurationFrontiersV2:
        self.validate_component(snapshot, component)
        cells = tuple(
            sorted(
                cell
                for cell in component.cells
                if any(
                    neighbor in snapshot.unknown_cells
                    for neighbor in self._neighbors(cell)
                )
            )
        )
        frontiers = ConfigurationFrontiersV2(
            **self._result_binding(snapshot),
            component_sha256=component.content_sha256,
            cells=cells,
        )
        self._issued_frontiers[id(frontiers)] = frontiers
        return frontiers

    def validate_frontiers(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        frontiers: ConfigurationFrontiersV2,
    ) -> None:
        self.validate_component(snapshot, component)
        if type(frontiers) is not ConfigurationFrontiersV2:
            raise TypeError("frontiers must be ConfigurationFrontiersV2")
        if self._issued_frontiers.get(id(frontiers)) is not frontiers:
            raise SnapshotBindingError(
                "frontiers are not the exact live object issued by this planner"
            )
        frontiers.assert_integrity()
        expected = tuple(
            sorted(
                cell
                for cell in component.cells
                if any(
                    neighbor in snapshot.unknown_cells
                    for neighbor in self._neighbors(cell)
                )
            )
        )
        if (
            frontiers.snapshot_sha256 != snapshot.content_sha256
            or frontiers.component_sha256 != component.content_sha256
            or frontiers.physical_map_frame_sha256
            != snapshot.physical_map_frame_sha256
            or frontiers.configuration_map_frame_sha256
            != snapshot.configuration_map_frame_sha256
            or frontiers.physical_revision != snapshot.physical_revision
            or frontiers.configuration_revision != snapshot.configuration_revision
            or frontiers.free_support_sha256 != snapshot.free_support_sha256
            or frontiers.occupied_support_sha256
            != snapshot.occupied_support_sha256
            or frontiers.cells != expected
        ):
            raise SnapshotBindingError(
                "frontiers are not bound to the current snapshot/component"
            )

    def astar(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        start: Cell,
        goal: Cell,
    ) -> ConfigurationPathV2 | None:
        self._validate(snapshot)
        normalized_start = _cell(start, "start")
        normalized_goal = _cell(goal, "goal")
        if (
            normalized_start not in snapshot.free_cells
            or normalized_goal not in snapshot.free_cells
        ):
            return None
        frontier: list[tuple[int, int, Cell]] = [(0, 0, normalized_start)]
        cost = {normalized_start: 0}
        parent: dict[Cell, Cell] = {}
        while frontier:
            _score, distance, current = heapq.heappop(frontier)
            if distance != cost.get(current):
                continue
            if current == normalized_goal:
                cells = [current]
                while current != normalized_start:
                    current = parent[current]
                    cells.append(current)
                path = ConfigurationPathV2(
                    **self._result_binding(snapshot),
                    cells=tuple(reversed(cells)),
                    cost=float(distance),
                )
                self._issued_paths[id(path)] = path
                try:
                    self.validate_path(snapshot, path)
                except BaseException:
                    self._issued_paths.pop(id(path), None)
                    raise
                return path
            for neighbor in self._neighbors(current):
                if neighbor not in snapshot.free_cells:
                    continue
                next_distance = distance + 1
                if next_distance >= cost.get(neighbor, 1 << 60):
                    continue
                cost[neighbor] = next_distance
                parent[neighbor] = current
                heuristic = abs(neighbor[0] - normalized_goal[0]) + abs(
                    neighbor[1] - normalized_goal[1]
                )
                heapq.heappush(
                    frontier,
                    (next_distance + heuristic, next_distance, neighbor),
                )
        return None

    def validate_path(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        path: ConfigurationPathV2,
    ) -> None:
        self._validate(snapshot)
        if type(path) is not ConfigurationPathV2:
            raise TypeError("path must be ConfigurationPathV2")
        if self._issued_paths.get(id(path)) is not path:
            raise StalePathError(
                "path is not the exact live object issued by this planner"
            )
        if (
            path.snapshot_sha256 != snapshot.content_sha256
            or path.physical_map_frame_sha256
            != snapshot.physical_map_frame_sha256
            or path.configuration_map_frame_sha256
            != snapshot.configuration_map_frame_sha256
            or path.physical_revision != snapshot.physical_revision
            or path.configuration_revision != snapshot.configuration_revision
            or path.free_support_sha256 != snapshot.free_support_sha256
            or path.occupied_support_sha256 != snapshot.occupied_support_sha256
        ):
            raise StalePathError("path is not bound to the current V2 snapshot")
        if any(cell not in snapshot.free_cells for cell in path.cells):
            raise InvalidConfigurationPathError("path leaves configuration FREE")
        if any(
            abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
            for first, second in zip(path.cells, path.cells[1:])
        ):
            raise InvalidConfigurationPathError("path is not four-connected")
        if not math.isclose(
            path.cost,
            float(len(path.cells) - 1),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise InvalidConfigurationPathError("path cost changed")


__all__ = [
    "CONFIGURATION_CELL_SIZE_M",
    "ConfigurationComponentV2",
    "ConfigurationFrontiersV2",
    "ConfigurationPathV2",
    "FIXED_PROFILE_V2",
    "FOOTPRINT_RADIUS_M",
    "FREE_SUPPORT_COUNT",
    "FREE_SUPPORT_SHA256",
    "OCCUPIED_SUPPORT_COUNT",
    "OCCUPIED_SUPPORT_SHA256",
    "PHYSICAL_CELL_SIZE_M",
    "PLANNING_CONNECTIVITY",
    "PROFILE_SHA256",
    "TwoResolutionConfigurationPlannerV2",
    "TwoResolutionConfigurationProjectionV2",
    "TwoResolutionConfigurationSnapshotV2",
    "TwoResolutionProfileV2",
    "assert_fixed_profile_integrity",
    "derive_cross_grid_supports",
    "physical_index_for_configuration_offset",
]
