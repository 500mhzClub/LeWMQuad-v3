"""Revision-bound conversion of G3 V2 paths into world-frame waypoints.

The adapter is intentionally small.  It does not plan, smooth, or execute a
route.  It revalidates an exact live path issued by the frozen G3 V2 planner
and records the deterministic 0.10 m cell-centre conversion needed by a later
development controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    SnapshotBindingError,
    StalePathError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    ConfigurationPathV2,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


Cell = tuple[int, int]
XY = tuple[float, float]

WORLD_WAYPOINT_ADAPTER_SCHEMA = (
    "lewm_g3_v2_configuration_path_world_waypoint_adapter_v1"
)


class WorldWaypointBindingError(SnapshotBindingError):
    """Raised when a waypoint receipt is stale, copied, or misbound."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _cell(value: Sequence[int], name: str) -> Cell:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must contain two integer indices")
    return int(value[0]), int(value[1])


def _finite_xy(value: Sequence[float], name: str) -> XY:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{name} must contain two coordinates")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _path_receipt_sha256(path: ConfigurationPathV2) -> str:
    return _sha256(
        {
            "schema": "lewm_g3_v2_retained_configuration_path_receipt_v1",
            "snapshot_sha256": path.snapshot_sha256,
            "physical_map_frame_sha256": path.physical_map_frame_sha256,
            "configuration_map_frame_sha256": (
                path.configuration_map_frame_sha256
            ),
            "physical_revision": path.physical_revision,
            "configuration_revision": path.configuration_revision,
            "free_support_sha256": path.free_support_sha256,
            "occupied_support_sha256": path.occupied_support_sha256,
            "cells": [list(cell) for cell in path.cells],
            "cost_configuration_steps": path.cost,
        }
    )


@dataclass(frozen=True)
class WorldWaypointV1:
    ordinal: int
    configuration_cell: Cell
    world_xy_m: XY

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("waypoint ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("waypoint ordinal must be non-negative")
        object.__setattr__(
            self,
            "configuration_cell",
            _cell(self.configuration_cell, "configuration_cell"),
        )
        object.__setattr__(
            self,
            "world_xy_m",
            _finite_xy(self.world_xy_m, "world_xy_m"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "configuration_cell": list(self.configuration_cell),
            "world_xy_m": list(self.world_xy_m),
        }


@dataclass(frozen=True)
class ConfigurationPathWorldWaypointReceiptV1:
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
    physical_shape: tuple[int, int]
    configuration_shape: tuple[int, int]
    configuration_origin_xy_m: XY
    retained_path_receipt_sha256: str
    waypoints: tuple[WorldWaypointV1, ...]
    path_cost_configuration_steps: float
    path_cost_m: float
    exact_sim_tainted: bool
    _issuance_capability: object = field(repr=False, compare=False)
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
            "retained_path_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self._issuance_capability is None:
            raise TypeError("waypoint receipt requires an issuance capability")
        for name in ("physical_revision", "configuration_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        physical_shape = _cell(self.physical_shape, "physical_shape")
        configuration_shape = _cell(
            self.configuration_shape,
            "configuration_shape",
        )
        if any(value <= 0 for value in (*physical_shape, *configuration_shape)):
            raise ValueError("waypoint receipt shapes must be positive")
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("waypoint receipt changed the exact 2:1 lattice ratio")
        origin = _finite_xy(
            self.configuration_origin_xy_m,
            "configuration_origin_xy_m",
        )
        waypoints = tuple(self.waypoints)
        if not waypoints or any(type(row) is not WorldWaypointV1 for row in waypoints):
            raise ValueError("waypoint receipt requires typed waypoints")
        if tuple(row.ordinal for row in waypoints) != tuple(range(len(waypoints))):
            raise ValueError("waypoint ordinals must be contiguous and ordered")
        for name in ("path_cost_configuration_steps", "path_cost_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        expected_steps = float(len(waypoints) - 1)
        if not math.isclose(
            float(self.path_cost_configuration_steps),
            expected_steps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(self.path_cost_m),
            expected_steps * CONFIGURATION_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("waypoint path cost changed its 0.10 m step contract")
        if type(self.exact_sim_tainted) is not bool:
            raise TypeError("exact_sim_tainted must be boolean")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        object.__setattr__(self, "configuration_origin_xy_m", origin)
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(
            self,
            "path_cost_configuration_steps",
            float(self.path_cost_configuration_steps),
        )
        object.__setattr__(self, "path_cost_m", float(self.path_cost_m))
        object.__setattr__(self, "content_sha256", _sha256(self._core()))

    @property
    def start_configuration_cell(self) -> Cell:
        return self.waypoints[0].configuration_cell

    @property
    def goal_configuration_cell(self) -> Cell:
        return self.waypoints[-1].configuration_cell

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    def _core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_v2_configuration_path_world_waypoint_receipt_v1",
            "adapter_schema": WORLD_WAYPOINT_ADAPTER_SCHEMA,
            "snapshot_sha256": self.snapshot_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": (
                self.configuration_map_frame_sha256
            ),
            "memory_config_sha256": self.memory_config_sha256,
            "physical_content_sha256": self.physical_content_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_cells_per_configuration_axis": [2, 2],
            "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "configuration_origin_xy_m": list(self.configuration_origin_xy_m),
            "retained_path_receipt_sha256": self.retained_path_receipt_sha256,
            "waypoints": [row.to_dict() for row in self.waypoints],
            "path_cost_configuration_steps": self.path_cost_configuration_steps,
            "path_cost_m": self.path_cost_m,
            "exact_sim_tainted": self.exact_sim_tainted,
            "development_execution_eligible": True,
            "hardware_execution_authorized": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self._core()):
            raise WorldWaypointBindingError("world-waypoint receipt was mutated")

    def __copy__(self) -> "ConfigurationPathWorldWaypointReceiptV1":
        raise TypeError("world-waypoint receipts are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "ConfigurationPathWorldWaypointReceiptV1":
        del memo
        raise TypeError("world-waypoint receipts are non-copyable")


class ConfigurationPathWorldWaypointIssuerV1:
    """Issue single-use world-centre receipts for exact live G3 V2 paths."""

    def __init__(
        self,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
    ) -> None:
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("waypoint issuer requires the exact G3 V2 projection")
        if type(planner) is not TwoResolutionConfigurationPlannerV2:
            raise TypeError("waypoint issuer requires the exact G3 V2 planner")
        if getattr(planner, "_projection", None) is not projection:
            raise SnapshotBindingError("planner and projection instances differ")
        self._projection = projection
        self._planner = planner
        self._capability = object()
        self._issued: dict[int, ConfigurationPathWorldWaypointReceiptV1] = {}
        self._consumed: set[int] = set()

    def __copy__(self) -> "ConfigurationPathWorldWaypointIssuerV1":
        raise TypeError("world-waypoint issuers are non-copyable")

    def __deepcopy__(self, memo: object) -> "ConfigurationPathWorldWaypointIssuerV1":
        del memo
        raise TypeError("world-waypoint issuers are non-copyable")

    def _build(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        path: ConfigurationPathV2,
    ) -> ConfigurationPathWorldWaypointReceiptV1:
        frame = snapshot.configuration_map_frame
        waypoints = tuple(
            WorldWaypointV1(
                ordinal=index,
                configuration_cell=cell,
                world_xy_m=frame.cell_center(cell),
            )
            for index, cell in enumerate(path.cells)
        )
        return ConfigurationPathWorldWaypointReceiptV1(
            snapshot_sha256=snapshot.content_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                snapshot.configuration_map_frame_sha256
            ),
            memory_config_sha256=snapshot.memory_config_sha256,
            physical_content_sha256=snapshot.physical_content_sha256,
            projection_source_sha256=snapshot.projection_source_sha256,
            profile_sha256=snapshot.profile_sha256,
            free_support_sha256=snapshot.free_support_sha256,
            occupied_support_sha256=snapshot.occupied_support_sha256,
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            configuration_origin_xy_m=frame.origin_xy_m,
            retained_path_receipt_sha256=_path_receipt_sha256(path),
            waypoints=waypoints,
            path_cost_configuration_steps=path.cost,
            path_cost_m=path.cost * CONFIGURATION_CELL_SIZE_M,
            exact_sim_tainted=snapshot.exact_sim_tainted,
            _issuance_capability=self._capability,
        )

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        path: ConfigurationPathV2,
    ) -> ConfigurationPathWorldWaypointReceiptV1:
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_path(snapshot, path)
        receipt = self._build(snapshot, path)
        self._issued[id(receipt)] = receipt
        return receipt

    def validate(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        path: ConfigurationPathV2,
        receipt: ConfigurationPathWorldWaypointReceiptV1,
        *,
        consume: bool = False,
    ) -> None:
        if type(receipt) is not ConfigurationPathWorldWaypointReceiptV1:
            raise TypeError("receipt must be ConfigurationPathWorldWaypointReceiptV1")
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_path(snapshot, path)
        receipt.assert_integrity()
        identity = id(receipt)
        if (
            receipt._issuance_capability is not self._capability
            or self._issued.get(identity) is not receipt
        ):
            raise WorldWaypointBindingError(
                "waypoint receipt is not the exact live object issued here"
            )
        if identity in self._consumed:
            raise WorldWaypointBindingError("waypoint receipt was already consumed")
        expected = self._build(snapshot, path)
        if expected.content_sha256 != receipt.content_sha256:
            raise WorldWaypointBindingError(
                "waypoint receipt differs from the current retained path"
            )
        if consume:
            self._consumed.add(identity)


__all__ = [
    "ConfigurationPathWorldWaypointIssuerV1",
    "ConfigurationPathWorldWaypointReceiptV1",
    "WORLD_WAYPOINT_ADAPTER_SCHEMA",
    "WorldWaypointBindingError",
    "WorldWaypointV1",
]
