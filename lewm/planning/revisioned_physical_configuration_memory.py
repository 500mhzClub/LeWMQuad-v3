"""Revisioned physical evidence and immutable configuration-space snapshots.

This module is additive G3 infrastructure. The mutable store contains physical
evidence only. Planning is available solely through ``ConfigurationPlanner``
over an immutable snapshot derived with the registered asymmetric morphology.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import heapq
import json
import math
from typing import Iterable, Mapping, Sequence


Cell = tuple[int, int]
XY = tuple[float, float]
PoseMean = tuple[float, float, float]
Covariance3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

REGISTERED_PHYSICAL_CELL_SIZE_M = 0.10
REGISTERED_FOOTPRINT_RADIUS_M = 0.47
REGISTERED_FREE_SUPPORT_COUNT = 89
REGISTERED_OCCUPIED_SUPPORT_COUNT = 69
REGISTERED_PHYSICAL_PROJECTION_CONTRACT_SHA256 = hashlib.sha256(
    b"lewm_g3_conservative_physical_projection_v1"
).hexdigest()
REGISTERED_POSE_COVARIANCE_DIAGONAL_LIMITS: PoseMean = (0.04, 0.04, 0.01)
ZERO_INFLATION_EXACT_PHYSICAL_SEMANTICS = "zero_inflation_physical_occupancy"
_EPS = 1e-12


class PhysicalLabel(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


class EvidenceAuthority(str, Enum):
    LEARNED_PHYSICAL = "learned_physical"
    EXACT_PHYSICAL = "exact_physical"
    EXECUTOR_OUTCOME = "executor_outcome"
    RESET_CLEARANCE = "reset_clearance"


class FusionMode(str, Enum):
    PERSISTENT = "persistent"
    CURRENT_FRAME_ONLY = "current_frame_only"


class PoseSource(str, Enum):
    DEPLOYMENT_ODOMETRY = "deployment_odometry"
    RESET_CERTIFICATE = "reset_certificate"
    EXACT_SIM_ODOMETRY_ABLATION = "exact_sim_odometry_ablation"


class ExecutionBlockKind(str, Enum):
    CONTACT = "contact"
    STALL = "stall"
    EXECUTION_VETO = "execution_veto"


class ExecutionEvidenceKind(str, Enum):
    RESET_CLEARANCE = "reset_clearance"
    TRAVERSAL_SUCCESS = "traversal_success"
    CONTACT = "contact"
    STALL = "stall"
    EXECUTION_VETO = "execution_veto"


class TransactionRejectedError(ValueError):
    """Raised before mutation when an evidence transaction is inadmissible."""


class SnapshotBindingError(ValueError):
    """Raised when a snapshot does not match the planner's frozen contract."""


class StaleSnapshotError(SnapshotBindingError):
    """Raised when memory advanced after a snapshot was derived."""


class StalePathError(SnapshotBindingError):
    """Raised when a path is not bound to the supplied current snapshot."""


class InvalidConfigurationPathError(ValueError):
    """Raised when a bound path is geometrically invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_clone(value: object) -> object:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


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
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
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


def _xy(value: Sequence[float], name: str) -> XY:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    return (_finite(value[0], name), _finite(value[1], name))


def _pose_mean(value: Sequence[float]) -> PoseMean:
    if len(value) != 3:
        raise ValueError("pose mean must contain three values")
    return tuple(_finite(item, "pose mean") for item in value)  # type: ignore[return-value]


def _covariance3(value: Sequence[Sequence[float]]) -> Covariance3:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError("pose covariance must be 3x3")
    matrix = tuple(
        tuple(_finite(item, "pose covariance") for item in row) for row in value
    )
    for index in range(3):
        if matrix[index][index] < 0.0:
            raise ValueError("pose covariance diagonal must be non-negative")
        for other in range(3):
            if not math.isclose(
                matrix[index][other],
                matrix[other][index],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("pose covariance must be symmetric")
    principal_minors = (
        matrix[0][0],
        matrix[1][1],
        matrix[2][2],
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
        matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
        (
            matrix[0][0]
            * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1]
            * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2]
            * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        ),
    )
    if any(minor < -1e-12 for minor in principal_minors):
        raise ValueError("pose covariance must be positive semidefinite")
    return matrix  # type: ignore[return-value]


def _cells_json(cells: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(cells)]


def _exact_physical_cells_sha256(
    evidence: Iterable[PhysicalCellEvidence],
    unknown_cells: Iterable[Cell],
) -> str:
    labels = [row.to_dict() for row in evidence]
    labels.extend(
        {"cell": list(cell), "label": int(PhysicalLabel.UNKNOWN)}
        for cell in unknown_cells
    )
    labels.sort(key=lambda row: tuple(row["cell"]))  # type: ignore[arg-type]
    return _canonical_sha256(
        {
            "schema": "lewm_g3_zero_inflation_exact_physical_cells_v1",
            "labels": labels,
        }
    )


@dataclass(frozen=True)
class MapFrameIdentity:
    """Content-addressed reset-local physical-map lattice."""

    session_id: str
    origin_xy_m: XY
    cell_size_m: float = REGISTERED_PHYSICAL_CELL_SIZE_M
    frame_id: str = "reset_local_odometry"

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        object.__setattr__(self, "origin_xy_m", _xy(self.origin_xy_m, "map origin"))
        cell_size = _finite(self.cell_size_m, "cell_size_m")
        if cell_size <= 0.0:
            raise ValueError("cell_size_m must be positive")
        object.__setattr__(self, "cell_size_m", cell_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_map_frame_identity_v1",
            "session_id": self.session_id,
            "origin_xy_m": list(self.origin_xy_m),
            "cell_size_m": self.cell_size_m,
            "frame_id": self.frame_id,
        }

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def world_to_cell(self, xy_m: Sequence[float]) -> Cell:
        x, y = _xy(xy_m, "world xy")
        return (
            int(math.floor((x - self.origin_xy_m[0]) / self.cell_size_m)),
            int(math.floor((y - self.origin_xy_m[1]) / self.cell_size_m)),
        )

    def cell_center(self, cell: Sequence[int]) -> XY:
        x, y = _cell(cell)
        return (
            self.origin_xy_m[0] + (x + 0.5) * self.cell_size_m,
            self.origin_xy_m[1] + (y + 0.5) * self.cell_size_m,
        )


@dataclass(frozen=True)
class ObservationIdentity:
    observation_id: str
    payload_sha256: str
    producer_sha256: str
    authority: EvidenceAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        _validate_sha256(self.payload_sha256, "payload_sha256")
        _validate_sha256(self.producer_sha256, "producer_sha256")
        if not isinstance(self.authority, EvidenceAuthority):
            raise TypeError("authority must be an EvidenceAuthority")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "payload_sha256": self.payload_sha256,
            "producer_sha256": self.producer_sha256,
            "authority": self.authority.value,
        }


@dataclass(frozen=True)
class PoseProvenance:
    """Pose and synchronization facts used to register one observation."""

    source: PoseSource
    frame_id: str
    mean_xy_yaw: PoseMean
    covariance_xy_yaw: Covariance3
    timestamp_ns: int
    synchronization_id: str
    camera_transform_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, PoseSource):
            raise TypeError("source must be a PoseSource")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if not isinstance(self.synchronization_id, str) or not self.synchronization_id:
            raise ValueError("synchronization_id must be non-empty")
        object.__setattr__(self, "mean_xy_yaw", _pose_mean(self.mean_xy_yaw))
        object.__setattr__(
            self,
            "covariance_xy_yaw",
            _covariance3(self.covariance_xy_yaw),
        )
        object.__setattr__(
            self, "timestamp_ns", _nonnegative_int(self.timestamp_ns, "timestamp_ns")
        )
        _validate_sha256(self.camera_transform_sha256, "camera_transform_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "frame_id": self.frame_id,
            "mean_xy_yaw": list(self.mean_xy_yaw),
            "covariance_xy_yaw": [list(row) for row in self.covariance_xy_yaw],
            "timestamp_ns": self.timestamp_ns,
            "synchronization_id": self.synchronization_id,
            "camera_transform_sha256": self.camera_transform_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PhysicalCellEvidence:
    """One already-admitted confirmed physical label."""

    cell: Cell
    label: PhysicalLabel

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell))
        if not isinstance(self.label, PhysicalLabel):
            raise TypeError("label must be a PhysicalLabel")
        if self.label is PhysicalLabel.UNKNOWN:
            raise ValueError("UNKNOWN is absence of evidence, not an evidence update")

    def to_dict(self) -> dict[str, object]:
        return {"cell": list(self.cell), "label": int(self.label)}


@dataclass(frozen=True)
class ExactPhysicalAdmission:
    """Serialized exact-evidence contract record for development-only state."""

    payload_sha256: str
    observation_sha256: str
    pose_sha256: str
    projection_contract_sha256: str
    calibration_sha256: str
    pose_uncertainty_contract_sha256: str
    source_semantics: str
    label_inflation_radius_m: float
    exact_sim_tainted: bool

    def __post_init__(self) -> None:
        for name in (
            "payload_sha256",
            "observation_sha256",
            "pose_sha256",
            "projection_contract_sha256",
            "calibration_sha256",
            "pose_uncertainty_contract_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.source_semantics != ZERO_INFLATION_EXACT_PHYSICAL_SEMANTICS:
            raise ValueError("exact physical admission semantics changed")
        inflation = _finite(
            self.label_inflation_radius_m, "label_inflation_radius_m"
        )
        if inflation != 0.0:
            raise ValueError("exact physical admission requires zero inflation")
        object.__setattr__(self, "label_inflation_radius_m", inflation)
        if not isinstance(self.exact_sim_tainted, bool):
            raise TypeError("exact_sim_tainted must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_exact_physical_admission_v1",
            "payload_sha256": self.payload_sha256,
            "observation_sha256": self.observation_sha256,
            "pose_sha256": self.pose_sha256,
            "projection_contract_sha256": self.projection_contract_sha256,
            "calibration_sha256": self.calibration_sha256,
            "pose_uncertainty_contract_sha256": (
                self.pose_uncertainty_contract_sha256
            ),
            "source_semantics": self.source_semantics,
            "label_inflation_radius_m": self.label_inflation_radius_m,
            "exact_sim_tainted": self.exact_sim_tainted,
        }

    def semantic_dict(self) -> dict[str, object]:
        result = self.to_dict()
        del result["payload_sha256"]
        del result["observation_sha256"]
        del result["pose_sha256"]
        return result


@dataclass(frozen=True)
class ExecutionEvidenceAdmission:
    """Historical record from the withdrawn executor/reset candidate.

    Promoted live admission is unconditionally unavailable.  This type remains
    only so old candidate bytes can be parsed and explicitly rejected rather
    than acquiring authority through schema ambiguity.
    """

    admission_id_sha256: str
    adapter_instance_sha256: str
    source_memory_instance_sha256: str
    receipt_content_sha256: str
    adapter_contract_sha256: str
    body_support_contract_sha256: str
    map_frame_sha256: str
    observation_sha256: str
    pose_sha256: str
    evidence_content_sha256: str
    memory_revision_before: int
    evidence_kind: ExecutionEvidenceKind
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "admission_id_sha256",
            "adapter_instance_sha256",
            "source_memory_instance_sha256",
            "receipt_content_sha256",
            "adapter_contract_sha256",
            "body_support_contract_sha256",
            "map_frame_sha256",
            "observation_sha256",
            "pose_sha256",
            "evidence_content_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "memory_revision_before",
            _nonnegative_int(self.memory_revision_before, "memory_revision_before"),
        )
        if not isinstance(self.evidence_kind, ExecutionEvidenceKind):
            raise TypeError("evidence_kind must be an ExecutionEvidenceKind")
        object.__setattr__(self, "content_sha256", _canonical_sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_execution_evidence_admission_v1",
            "admission_id_sha256": self.admission_id_sha256,
            "adapter_instance_sha256": self.adapter_instance_sha256,
            "source_memory_instance_sha256": self.source_memory_instance_sha256,
            "receipt_content_sha256": self.receipt_content_sha256,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "body_support_contract_sha256": self.body_support_contract_sha256,
            "map_frame_sha256": self.map_frame_sha256,
            "observation_sha256": self.observation_sha256,
            "pose_sha256": self.pose_sha256,
            "evidence_content_sha256": self.evidence_content_sha256,
            "memory_revision_before": self.memory_revision_before,
            "evidence_kind": self.evidence_kind.value,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_execution_evidence_semantics_v1",
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "body_support_contract_sha256": self.body_support_contract_sha256,
            "evidence_content_sha256": self.evidence_content_sha256,
            "evidence_kind": self.evidence_kind.value,
        }

    def assert_integrity(self) -> None:
        if self.content_sha256 != _canonical_sha256(self.to_dict(False)):
            raise TransactionRejectedError("execution evidence admission was mutated")


def _orientation(a: XY, b: XY, c: XY) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: XY, start: XY, end: XY) -> bool:
    if abs(_orientation(start, end, point)) > _EPS:
        return False
    return bool(
        min(start[0], end[0]) - _EPS <= point[0] <= max(start[0], end[0]) + _EPS
        and min(start[1], end[1]) - _EPS
        <= point[1]
        <= max(start[1], end[1]) + _EPS
    )


def _segments_intersect_closed(a: XY, b: XY, c: XY, d: XY) -> bool:
    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] * orientations[1] < -_EPS and orientations[2] * orientations[3] < -_EPS:
        return True
    return bool(
        (abs(orientations[0]) <= _EPS and _point_on_segment(c, a, b))
        or (abs(orientations[1]) <= _EPS and _point_on_segment(d, a, b))
        or (abs(orientations[2]) <= _EPS and _point_on_segment(a, c, d))
        or (abs(orientations[3]) <= _EPS and _point_on_segment(b, c, d))
    )


def _segments_cross_properly(a: XY, b: XY, c: XY, d: XY) -> bool:
    return bool(
        _orientation(a, b, c) * _orientation(a, b, d) < -_EPS
        and _orientation(c, d, a) * _orientation(c, d, b) < -_EPS
    )


def _validate_simple_polygon(vertices: tuple[XY, ...]) -> None:
    if len(vertices) < 3:
        raise ValueError("traversal polygon needs at least three vertices")
    if len(set(vertices)) != len(vertices):
        raise ValueError("traversal polygon vertices must be unique")
    area_twice = sum(
        vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
        - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
        for index in range(len(vertices))
    )
    if abs(area_twice) <= _EPS:
        raise ValueError("traversal polygon must have non-zero area")
    count = len(vertices)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if _segments_intersect_closed(
                vertices[first],
                vertices[first_next],
                vertices[second],
                vertices[second_next],
            ):
                raise ValueError("traversal polygon must be simple")


def _point_in_polygon_closed(point: XY, vertices: tuple[XY, ...]) -> bool:
    inside = False
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if _point_on_segment(point, start, end):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            x_crossing = start[0] + (
                (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            )
            if x_crossing > point[0]:
                inside = not inside
    return inside


def _closed_cell_square_covered_by_polygon(
    frame: MapFrameIdentity,
    cell: Cell,
    vertices: tuple[XY, ...],
) -> bool:
    center_x, center_y = frame.cell_center(cell)
    half = 0.5 * frame.cell_size_m
    corners: tuple[XY, ...] = (
        (center_x - half, center_y - half),
        (center_x - half, center_y + half),
        (center_x + half, center_y + half),
        (center_x + half, center_y - half),
    )
    if not all(_point_in_polygon_closed(corner, vertices) for corner in corners):
        return False
    for vertex in vertices:
        if (
            center_x - half + _EPS < vertex[0] < center_x + half - _EPS
            and center_y - half + _EPS < vertex[1] < center_y + half - _EPS
        ):
            return False
    for poly_index, poly_start in enumerate(vertices):
        poly_end = vertices[(poly_index + 1) % len(vertices)]
        for square_index, square_start in enumerate(corners):
            square_end = corners[(square_index + 1) % len(corners)]
            if _segments_cross_properly(poly_start, poly_end, square_start, square_end):
                return False
    return True


@dataclass(frozen=True)
class VerifiedTraversalPolygon:
    traversal_id: str
    vertices_xy_m: tuple[XY, ...]
    outcome_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.traversal_id, str) or not self.traversal_id:
            raise ValueError("traversal_id must be non-empty")
        vertices = tuple(_xy(vertex, "traversal vertex") for vertex in self.vertices_xy_m)
        _validate_simple_polygon(vertices)
        object.__setattr__(self, "vertices_xy_m", vertices)
        _validate_sha256(self.outcome_sha256, "outcome_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "traversal_id": self.traversal_id,
            "vertices_xy_m": [list(vertex) for vertex in self.vertices_xy_m],
            "outcome_sha256": self.outcome_sha256,
        }


@dataclass(frozen=True)
class ExecutionBlock:
    block_id: str
    body_center_xy_m: XY
    kind: ExecutionBlockKind
    outcome_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("block_id must be non-empty")
        object.__setattr__(
            self,
            "body_center_xy_m",
            _xy(self.body_center_xy_m, "block body center"),
        )
        if not isinstance(self.kind, ExecutionBlockKind):
            raise TypeError("kind must be an ExecutionBlockKind")
        _validate_sha256(self.outcome_sha256, "outcome_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "body_center_xy_m": list(self.body_center_xy_m),
            "kind": self.kind.value,
            "outcome_sha256": self.outcome_sha256,
        }


def _execution_evidence_content_sha256(
    traversals: Iterable[VerifiedTraversalPolygon],
    blocks: Iterable[ExecutionBlock],
) -> str:
    return _canonical_sha256(
        {
            "schema": "lewm_g3_admitted_execution_evidence_v1",
            "verified_traversals": [
                row.to_dict()
                for row in sorted(traversals, key=lambda item: item.traversal_id)
            ],
            "execution_blocks": [
                row.to_dict()
                for row in sorted(blocks, key=lambda item: item.block_id)
            ],
        }
    )


@dataclass(frozen=True)
class PhysicalEvidenceTransaction:
    """One atomic update keyed by observation, map frame, and pose provenance."""

    observation: ObservationIdentity
    map_frame: MapFrameIdentity
    pose: PoseProvenance
    physical_evidence: tuple[PhysicalCellEvidence, ...] = ()
    verified_traversals: tuple[VerifiedTraversalPolygon, ...] = ()
    execution_blocks: tuple[ExecutionBlock, ...] = ()
    retract_learned_observation_ids: tuple[str, ...] = ()
    observed_unknown_cells: tuple[Cell, ...] = ()
    exact_admission: ExactPhysicalAdmission | None = None
    execution_admission: ExecutionEvidenceAdmission | None = None
    projection_contract_sha256: str = (
        REGISTERED_PHYSICAL_PROJECTION_CONTRACT_SHA256
    )

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationIdentity):
            raise TypeError("observation must be an ObservationIdentity")
        if not isinstance(self.map_frame, MapFrameIdentity):
            raise TypeError("map_frame must be a MapFrameIdentity")
        if not isinstance(self.pose, PoseProvenance):
            raise TypeError("pose must be a PoseProvenance")
        _validate_sha256(
            self.projection_contract_sha256, "projection_contract_sha256"
        )
        evidence_input = tuple(self.physical_evidence)
        traversal_input = tuple(self.verified_traversals)
        block_input = tuple(self.execution_blocks)
        unknown_input = tuple(self.observed_unknown_cells)
        if any(not isinstance(row, PhysicalCellEvidence) for row in evidence_input):
            raise TypeError("physical_evidence entries must be PhysicalCellEvidence")
        if any(not isinstance(row, VerifiedTraversalPolygon) for row in traversal_input):
            raise TypeError("verified_traversals entries must be VerifiedTraversalPolygon")
        if any(not isinstance(row, ExecutionBlock) for row in block_input):
            raise TypeError("execution_blocks entries must be ExecutionBlock")
        evidence = tuple(sorted(evidence_input, key=lambda row: row.cell))
        traversals = tuple(sorted(traversal_input, key=lambda row: row.traversal_id))
        blocks = tuple(sorted(block_input, key=lambda row: row.block_id))
        unknown_cells = tuple(sorted(_cell(cell, "observed unknown cell") for cell in unknown_input))
        retractions = tuple(sorted(tuple(self.retract_learned_observation_ids)))
        if any(not isinstance(value, str) or not value for value in retractions):
            raise ValueError("retracted observation ids must be non-empty strings")
        if len({row.cell for row in evidence}) != len(evidence):
            raise ValueError("one transaction cannot label a physical cell twice")
        if len({row.traversal_id for row in traversals}) != len(traversals):
            raise ValueError("duplicate traversal_id inside transaction")
        if len({row.block_id for row in blocks}) != len(blocks):
            raise ValueError("duplicate block_id inside transaction")
        if len(set(retractions)) != len(retractions):
            raise ValueError("duplicate learned-evidence retraction")
        if len(set(unknown_cells)) != len(unknown_cells):
            raise ValueError("duplicate observed unknown cell")
        if set(unknown_cells) & {row.cell for row in evidence}:
            raise ValueError("a cell cannot be both labelled and observed UNKNOWN")
        if (
            self.observation.authority is EvidenceAuthority.EXACT_PHYSICAL
            and retractions
        ):
            raise ValueError("exact-physical transactions cannot retract learned evidence")
        authority = self.observation.authority
        if authority is EvidenceAuthority.EXACT_PHYSICAL:
            if not isinstance(self.exact_admission, ExactPhysicalAdmission):
                raise TypeError("EXACT_PHYSICAL transactions require adapter admission")
            if (
                self.exact_admission.payload_sha256
                != self.observation.payload_sha256
                or self.exact_admission.observation_sha256
                != _canonical_sha256(self.observation.to_dict())
                or self.exact_admission.pose_sha256 != self.pose.content_sha256
            ):
                raise ValueError("exact physical admission identity mismatch")
            if traversals or blocks:
                raise ValueError(
                    "exact physical adapter admission cannot authorize traversal or blocks"
                )
            if (
                _exact_physical_cells_sha256(evidence, unknown_cells)
                != self.observation.payload_sha256
            ):
                raise ValueError(
                    "exact physical payload does not match admitted labels"
                )
            if self.execution_admission is not None:
                raise ValueError("exact transactions cannot carry execution admission")
        else:
            if self.exact_admission is not None:
                raise ValueError("non-exact transactions cannot carry exact admission")
            if authority in {
                EvidenceAuthority.EXECUTOR_OUTCOME,
                EvidenceAuthority.RESET_CLEARANCE,
            }:
                admission = self.execution_admission
                if not isinstance(admission, ExecutionEvidenceAdmission):
                    raise TypeError(
                        "executor/reset transactions require adapter admission"
                    )
                admission.assert_integrity()
                if evidence or unknown_cells or retractions:
                    raise ValueError(
                        "executor/reset admission cannot authorize learned labels or retractions"
                    )
                if (
                    admission.receipt_content_sha256
                    != self.observation.payload_sha256
                    or admission.observation_sha256
                    != _canonical_sha256(self.observation.to_dict())
                    or admission.pose_sha256 != self.pose.content_sha256
                    or admission.map_frame_sha256 != self.map_frame.content_sha256
                    or admission.evidence_content_sha256
                    != _execution_evidence_content_sha256(traversals, blocks)
                    or any(
                        row.outcome_sha256 != admission.receipt_content_sha256
                        for row in (*traversals, *blocks)
                    )
                ):
                    raise ValueError("execution evidence admission identity mismatch")
                kind = admission.evidence_kind
                if authority is EvidenceAuthority.RESET_CLEARANCE:
                    if (
                        kind is not ExecutionEvidenceKind.RESET_CLEARANCE
                        or not traversals
                        or blocks
                    ):
                        raise ValueError(
                            "reset clearance admission must contain traversal support only"
                        )
                elif kind is ExecutionEvidenceKind.TRAVERSAL_SUCCESS:
                    if not traversals or blocks:
                        raise ValueError(
                            "successful execution admission must contain traversal support only"
                        )
                else:
                    expected_block_kind = ExecutionBlockKind(kind.value)
                    if traversals or len(blocks) != 1 or blocks[0].kind is not expected_block_kind:
                        raise ValueError(
                            "failed execution admission must contain one matching body block"
                        )
            elif self.execution_admission is not None:
                raise ValueError(
                    "caller transactions cannot carry execution admission"
                )
        if not (evidence or traversals or blocks or retractions or unknown_cells):
            raise ValueError("physical evidence transaction cannot be empty")
        object.__setattr__(self, "physical_evidence", evidence)
        object.__setattr__(self, "verified_traversals", traversals)
        object.__setattr__(self, "execution_blocks", blocks)
        object.__setattr__(self, "retract_learned_observation_ids", retractions)
        object.__setattr__(self, "observed_unknown_cells", unknown_cells)

    @property
    def transaction_key_sha256(self) -> str:
        return _canonical_sha256(
            {
                "observation": self.observation.to_dict(),
                "map_frame": self.map_frame.to_dict(),
                "pose": self.pose.to_dict(),
            }
        )

    @property
    def semantic_transaction_sha256(self) -> str:
        pose = self.pose.to_dict()
        pose.pop("timestamp_ns")
        pose.pop("synchronization_id")
        return _canonical_sha256(
            {
                "schema": "lewm_g3_semantic_physical_transaction_v2",
                "observation": {
                    "producer_sha256": self.observation.producer_sha256,
                    "authority": self.observation.authority.value,
                },
                "map_frame": self.map_frame.to_dict(),
                "projection_contract_sha256": self.projection_contract_sha256,
                "pose": pose,
                "physical_evidence": [
                    row.to_dict() for row in self.physical_evidence
                ],
                "observed_unknown_cells": _cells_json(
                    self.observed_unknown_cells
                ),
                "verified_traversals": [
                    row.to_dict() for row in self.verified_traversals
                ],
                "execution_blocks": [
                    row.to_dict() for row in self.execution_blocks
                ],
                "retract_learned_observation_ids": list(
                    self.retract_learned_observation_ids
                ),
                "exact_admission": (
                    None
                    if self.exact_admission is None
                    else self.exact_admission.semantic_dict()
                ),
                "execution_admission": (
                    None
                    if self.execution_admission is None
                    else self.execution_admission.semantic_dict()
                ),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_physical_evidence_transaction_v2",
            "transaction_key_sha256": self.transaction_key_sha256,
            "semantic_transaction_sha256": self.semantic_transaction_sha256,
            "observation": self.observation.to_dict(),
            "map_frame": self.map_frame.to_dict(),
            "projection_contract_sha256": self.projection_contract_sha256,
            "pose": self.pose.to_dict(),
            "physical_evidence": [row.to_dict() for row in self.physical_evidence],
            "observed_unknown_cells": _cells_json(self.observed_unknown_cells),
            "verified_traversals": [row.to_dict() for row in self.verified_traversals],
            "execution_blocks": [row.to_dict() for row in self.execution_blocks],
            "retract_learned_observation_ids": list(
                self.retract_learned_observation_ids
            ),
            "exact_admission": (
                None if self.exact_admission is None else self.exact_admission.to_dict()
            ),
            "execution_admission": (
                None
                if self.execution_admission is None
                else self.execution_admission.to_dict()
            ),
        }

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PhysicalObservationRecord:
    observation: ObservationIdentity
    pose: PoseProvenance
    transaction_sha256: str
    evidence: tuple[PhysicalCellEvidence, ...]
    observed_unknown_cells: tuple[Cell, ...]
    revision_added: int

    def to_dict(self) -> dict[str, object]:
        return {
            "observation": self.observation.to_dict(),
            "pose": self.pose.to_dict(),
            "transaction_sha256": self.transaction_sha256,
            "evidence": [row.to_dict() for row in self.evidence],
            "observed_unknown_cells": _cells_json(self.observed_unknown_cells),
            "revision_added": self.revision_added,
        }


@dataclass(frozen=True)
class TransactionReceipt:
    observation_id: str
    transaction_key_sha256: str
    transaction_sha256: str
    revision_before: int
    revision_after: int
    physical_evidence_cells: int
    verified_traversal_cells_added: int
    execution_blocks_added: int
    learned_observations_retracted: int


@dataclass(frozen=True)
class PhysicalMemoryConfig:
    map_frame: MapFrameIdentity
    fusion_mode: FusionMode = FusionMode.PERSISTENT
    planning_connectivity: int = 4
    allow_diagonal_corner_cutting: bool = False
    allow_exact_sim_odometry_ablation: bool = False
    require_registered_lattice: bool = True
    physical_projection_contract_sha256: str = (
        REGISTERED_PHYSICAL_PROJECTION_CONTRACT_SHA256
    )
    expected_camera_transform_sha256: str | None = None
    pose_covariance_diagonal_limits: PoseMean = (
        REGISTERED_POSE_COVARIANCE_DIAGONAL_LIMITS
    )
    promoted_runtime: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.map_frame, MapFrameIdentity):
            raise TypeError("map_frame must be a MapFrameIdentity")
        if not isinstance(self.fusion_mode, FusionMode):
            raise TypeError("fusion_mode must be a FusionMode")
        if self.planning_connectivity not in (4, 8):
            raise ValueError("planning_connectivity must be 4 or 8")
        if not isinstance(self.allow_diagonal_corner_cutting, bool):
            raise TypeError("allow_diagonal_corner_cutting must be boolean")
        if not isinstance(self.allow_exact_sim_odometry_ablation, bool):
            raise TypeError("allow_exact_sim_odometry_ablation must be boolean")
        if not isinstance(self.require_registered_lattice, bool):
            raise TypeError("require_registered_lattice must be boolean")
        if not isinstance(self.promoted_runtime, bool):
            raise TypeError("promoted_runtime must be boolean")
        if self.allow_diagonal_corner_cutting:
            raise ValueError("G3 snapshots fail closed: diagonal corner cutting is forbidden")
        if self.allow_exact_sim_odometry_ablation and self.promoted_runtime:
            raise ValueError(
                "exact simulator odometry ablation cannot be a promoted runtime"
            )
        _validate_sha256(
            self.physical_projection_contract_sha256,
            "physical_projection_contract_sha256",
        )
        if self.expected_camera_transform_sha256 is not None:
            _validate_sha256(
                self.expected_camera_transform_sha256,
                "expected_camera_transform_sha256",
            )
        limits = _pose_mean(self.pose_covariance_diagonal_limits)
        if any(value < 0.0 for value in limits):
            raise ValueError("pose covariance diagonal limits must be non-negative")
        object.__setattr__(self, "pose_covariance_diagonal_limits", limits)
        if self.promoted_runtime and (
            self.expected_camera_transform_sha256 is None
            or self.physical_projection_contract_sha256
            != REGISTERED_PHYSICAL_PROJECTION_CONTRACT_SHA256
            or limits != REGISTERED_POSE_COVARIANCE_DIAGONAL_LIMITS
        ):
            raise ValueError(
                "promoted runtime requires frozen projection, calibration, and uncertainty"
            )
        if self.require_registered_lattice and not math.isclose(
            self.map_frame.cell_size_m,
            REGISTERED_PHYSICAL_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("registered G3 physical memory requires a 0.10 m lattice")

    @property
    def pose_uncertainty_contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "lewm_g3_pose_uncertainty_limits_v1",
                "covariance_xy_yaw_diagonal_max": list(
                    self.pose_covariance_diagonal_limits
                ),
                "reject_above_limit": True,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_physical_memory_config_v1",
            "map_frame": self.map_frame.to_dict(),
            "fusion_mode": self.fusion_mode.value,
            "planning_connectivity": self.planning_connectivity,
            "allow_diagonal_corner_cutting": self.allow_diagonal_corner_cutting,
            "allow_exact_sim_odometry_ablation": (
                self.allow_exact_sim_odometry_ablation
            ),
            "require_registered_lattice": self.require_registered_lattice,
            "physical_projection_contract_sha256": (
                self.physical_projection_contract_sha256
            ),
            "expected_camera_transform_sha256": (
                self.expected_camera_transform_sha256
            ),
            "pose_covariance_diagonal_limits": list(
                self.pose_covariance_diagonal_limits
            ),
            "pose_uncertainty_contract_sha256": (
                self.pose_uncertainty_contract_sha256
            ),
            "promoted_runtime": self.promoted_runtime,
        }

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def _rasterize_verified_polygon(
    frame: MapFrameIdentity,
    polygon: VerifiedTraversalPolygon,
) -> frozenset[Cell]:
    xs = [vertex[0] for vertex in polygon.vertices_xy_m]
    ys = [vertex[1] for vertex in polygon.vertices_xy_m]
    minimum = frame.world_to_cell((min(xs), min(ys)))
    maximum = frame.world_to_cell((max(xs), max(ys)))
    covered: set[Cell] = set()
    for x in range(minimum[0] - 1, maximum[0] + 2):
        for y in range(minimum[1] - 1, maximum[1] + 2):
            cell = (x, y)
            if _closed_cell_square_covered_by_polygon(
                frame, cell, polygon.vertices_xy_m
            ):
                covered.add(cell)
    if not covered:
        raise TransactionRejectedError(
            f"verified traversal {polygon.traversal_id!r} covers no complete physical cell"
        )
    return frozenset(covered)


@dataclass(frozen=True)
class ConfigurationMorphology:
    physical_cell_size_m: float = REGISTERED_PHYSICAL_CELL_SIZE_M
    footprint_radius_m: float = REGISTERED_FOOTPRINT_RADIUS_M
    free_support_offsets: tuple[Cell, ...] = field(init=False)
    occupied_support_offsets: tuple[Cell, ...] = field(init=False)
    free_support_sha256: str = field(init=False)
    occupied_support_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cell_size = _finite(self.physical_cell_size_m, "physical_cell_size_m")
        radius = _finite(self.footprint_radius_m, "footprint_radius_m")
        if cell_size <= 0.0 or radius <= 0.0:
            raise ValueError("morphology cell size and radius must be positive")
        limit = int(math.ceil((radius + 0.5 * cell_size) / cell_size)) + 1
        free: list[Cell] = []
        occupied: list[Cell] = []
        for dx in range(-limit, limit + 1):
            for dy in range(-limit, limit + 1):
                square_dx = max(abs(dx) * cell_size - 0.5 * cell_size, 0.0)
                square_dy = max(abs(dy) * cell_size - 0.5 * cell_size, 0.0)
                if square_dx * square_dx + square_dy * square_dy <= radius * radius + _EPS:
                    free.append((dx, dy))
                if (
                    (dx * cell_size) * (dx * cell_size)
                    + (dy * cell_size) * (dy * cell_size)
                    <= radius * radius + _EPS
                ):
                    occupied.append((dx, dy))
        free_offsets = tuple(sorted(free))
        occupied_offsets = tuple(sorted(occupied))
        free_core = {
            "schema": "lewm_g3_free_closed_square_intersection_kernel_v1",
            "physical_cell_size_m": cell_size,
            "footprint_radius_m": radius,
            "inclusive_boundary": True,
            "offsets": _cells_json(free_offsets),
        }
        occupied_core = {
            "schema": "lewm_g3_occupied_center_inside_disc_kernel_v1",
            "physical_cell_size_m": cell_size,
            "footprint_radius_m": radius,
            "inclusive_boundary": True,
            "offsets": _cells_json(occupied_offsets),
        }
        free_hash = _canonical_sha256(free_core)
        occupied_hash = _canonical_sha256(occupied_core)
        contract_core = {
            "schema": "lewm_g3_asymmetric_configuration_morphology_v1",
            "physical_cell_size_m": cell_size,
            "footprint_radius_m": radius,
            "free_support_sha256": free_hash,
            "occupied_support_sha256": occupied_hash,
            "occupied_precedes_free": True,
            "otherwise": "unknown",
        }
        object.__setattr__(self, "physical_cell_size_m", cell_size)
        object.__setattr__(self, "footprint_radius_m", radius)
        object.__setattr__(self, "free_support_offsets", free_offsets)
        object.__setattr__(self, "occupied_support_offsets", occupied_offsets)
        object.__setattr__(self, "free_support_sha256", free_hash)
        object.__setattr__(self, "occupied_support_sha256", occupied_hash)
        object.__setattr__(self, "content_sha256", _canonical_sha256(contract_core))
        if (
            math.isclose(cell_size, REGISTERED_PHYSICAL_CELL_SIZE_M, abs_tol=1e-12)
            and math.isclose(radius, REGISTERED_FOOTPRINT_RADIUS_M, abs_tol=1e-12)
            and (
                len(free_offsets) != REGISTERED_FREE_SUPPORT_COUNT
                or len(occupied_offsets) != REGISTERED_OCCUPIED_SUPPORT_COUNT
            )
        ):
            raise AssertionError("registered morphology did not produce 89/69 supports")

    def assert_integrity(self) -> None:
        canonical = ConfigurationMorphology(
            physical_cell_size_m=self.physical_cell_size_m,
            footprint_radius_m=self.footprint_radius_m,
        )
        for name in (
            "free_support_offsets",
            "occupied_support_offsets",
            "free_support_sha256",
            "occupied_support_sha256",
            "content_sha256",
        ):
            if getattr(self, name) != getattr(canonical, name):
                raise SnapshotBindingError(
                    f"configuration morphology {name} was mutated"
                )


@dataclass(frozen=True)
class ConfigurationSnapshot:
    map_frame_sha256: str
    memory_config_sha256: str
    physical_revision: int
    physical_content_sha256: str
    exact_sim_tainted: bool
    morphology_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    planning_connectivity: int
    allow_diagonal_corner_cutting: bool
    free_cells: frozenset[Cell]
    occupied_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "map_frame_sha256",
            "memory_config_sha256",
            "physical_content_sha256",
            "morphology_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        if self.planning_connectivity not in (4, 8):
            raise ValueError("planning_connectivity must be 4 or 8")
        if self.allow_diagonal_corner_cutting:
            raise ValueError("configuration snapshots may not allow corner cutting")
        if not isinstance(self.exact_sim_tainted, bool):
            raise TypeError("exact_sim_tainted must be boolean")
        free = frozenset(_cell(cell) for cell in self.free_cells)
        occupied = frozenset(_cell(cell) for cell in self.occupied_cells)
        unknown = frozenset(_cell(cell) for cell in self.unknown_cells)
        if free & occupied or free & unknown or occupied & unknown:
            raise ValueError("configuration snapshot classes must be disjoint")
        object.__setattr__(self, "free_cells", free)
        object.__setattr__(self, "occupied_cells", occupied)
        object.__setattr__(self, "unknown_cells", unknown)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                {
                    "schema": "lewm_g3_configuration_snapshot_v1",
                    "map_frame_sha256": self.map_frame_sha256,
                    "memory_config_sha256": self.memory_config_sha256,
                    "physical_revision": self.physical_revision,
                    "physical_content_sha256": self.physical_content_sha256,
                    "exact_sim_tainted": self.exact_sim_tainted,
                    "morphology_sha256": self.morphology_sha256,
                    "free_support_sha256": self.free_support_sha256,
                    "occupied_support_sha256": self.occupied_support_sha256,
                    "planning_connectivity": self.planning_connectivity,
                    "allow_diagonal_corner_cutting": (
                        self.allow_diagonal_corner_cutting
                    ),
                    "free_cells": _cells_json(free),
                    "occupied_cells": _cells_json(occupied),
                    "unknown_cells": _cells_json(unknown),
                }
            ),
        )

    def assert_integrity(self) -> None:
        expected = ConfigurationSnapshot(
            map_frame_sha256=self.map_frame_sha256,
            memory_config_sha256=self.memory_config_sha256,
            physical_revision=self.physical_revision,
            physical_content_sha256=self.physical_content_sha256,
            exact_sim_tainted=self.exact_sim_tainted,
            morphology_sha256=self.morphology_sha256,
            free_support_sha256=self.free_support_sha256,
            occupied_support_sha256=self.occupied_support_sha256,
            planning_connectivity=self.planning_connectivity,
            allow_diagonal_corner_cutting=self.allow_diagonal_corner_cutting,
            free_cells=self.free_cells,
            occupied_cells=self.occupied_cells,
            unknown_cells=self.unknown_cells,
        )
        if self.content_sha256 != expected.content_sha256:
            raise SnapshotBindingError("configuration snapshot was mutated")

    @property
    def evaluated_cells(self) -> frozenset[Cell]:
        return self.free_cells | self.occupied_cells | self.unknown_cells

    def state(self, cell: Sequence[int]) -> PhysicalLabel:
        normalized = _cell(cell)
        if normalized in self.free_cells:
            return PhysicalLabel.FREE
        if normalized in self.occupied_cells:
            return PhysicalLabel.OCCUPIED
        return PhysicalLabel.UNKNOWN


class RevisionedPhysicalMemory:
    """Atomic physical evidence store with no frontier or route API."""

    __slots__ = (
        "_bound_camera_transform_sha256",
        "_config",
        "_config_content_sha256",
        "_evidence_index",
        "_exact_sim_tainted",
        "_execution_block_cells",
        "_execution_blocks",
        "_issued_snapshot_sha256",
        "_observation_records",
        "_physical_state_core_override",
        "_revision",
        "_seen_observation_ids",
        "_seen_semantic_transaction_keys",
        "_seen_transaction_keys",
        "_transaction_log",
        "_traversal_cells",
        "_traversals",
        "_verified_traversal_union",
    )

    def __init__(self, config: PhysicalMemoryConfig) -> None:
        if not isinstance(config, PhysicalMemoryConfig):
            raise TypeError("config must be a PhysicalMemoryConfig")
        self._config = config
        self._config_content_sha256 = config.content_sha256
        self._revision = 0
        self._observation_records: dict[str, PhysicalObservationRecord] = {}
        self._seen_observation_ids: set[str] = set()
        self._seen_transaction_keys: set[str] = set()
        self._seen_semantic_transaction_keys: set[str] = set()
        self._transaction_log: list[dict[str, object]] = []
        self._traversals: dict[str, VerifiedTraversalPolygon] = {}
        self._traversal_cells: dict[str, frozenset[Cell]] = {}
        self._verified_traversal_union: frozenset[Cell] = frozenset()
        self._execution_blocks: dict[str, ExecutionBlock] = {}
        self._execution_block_cells: dict[str, Cell] = {}
        self._evidence_index: dict[
            Cell, tuple[tuple[EvidenceAuthority, PhysicalLabel, str], ...]
        ] = {}
        self._issued_snapshot_sha256: set[str] = set()
        self._bound_camera_transform_sha256 = (
            config.expected_camera_transform_sha256
        )
        self._exact_sim_tainted = False
        self._physical_state_core_override = None

    def __getattribute__(self, name: str) -> object:
        if name == "_physical_state_core":
            try:
                override = object.__getattribute__(
                    self,
                    "_physical_state_core_override",
                )
            except AttributeError:
                override = None
            if override is not None:
                return override
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        # G5 hot-path tests instrument this one read-only diagnostic method.
        # Keep that narrow seam without restoring a general ``__dict__`` clone
        # surface or any mutable evidence-authority table.
        if name == "_physical_state_core":
            object.__setattr__(self, "_physical_state_core_override", value)
            return
        object.__setattr__(self, name, value)

    def __copy__(self) -> "RevisionedPhysicalMemory":
        raise TypeError("revisioned physical memory is non-copyable")

    def __deepcopy__(self, memo: object) -> "RevisionedPhysicalMemory":
        del memo
        raise TypeError("revisioned physical memory is non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError(
            "revisioned physical memory must use canonical serialize/deserialize"
        )

    @property
    def config(self) -> PhysicalMemoryConfig:
        return self._config

    @property
    def map_frame(self) -> MapFrameIdentity:
        return self._config.map_frame

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def exact_sim_tainted(self) -> bool:
        return self._exact_sim_tainted

    @property
    def bound_camera_transform_sha256(self) -> str | None:
        return self._bound_camera_transform_sha256

    @property
    def learned_observation_ids(self) -> frozenset[str]:
        return frozenset(
            observation_id
            for observation_id, record in self._observation_records.items()
            if record.observation.authority is EvidenceAuthority.LEARNED_PHYSICAL
        )

    @property
    def seen_observation_ids(self) -> frozenset[str]:
        """Immutable append-only identities accepted by this memory."""

        return frozenset(self._seen_observation_ids)

    @property
    def exact_observation_ids(self) -> frozenset[str]:
        return frozenset(
            observation_id
            for observation_id, record in self._observation_records.items()
            if record.observation.authority is EvidenceAuthority.EXACT_PHYSICAL
        )

    @property
    def traversal_ids(self) -> frozenset[str]:
        return frozenset(self._traversals)

    @property
    def execution_block_ids(self) -> frozenset[str]:
        return frozenset(self._execution_blocks)

    @property
    def execution_block_cells(self) -> frozenset[Cell]:
        return frozenset(self._execution_block_cells.values())

    @property
    def verified_traversal_cells(self) -> frozenset[Cell]:
        return self._verified_traversal_union

    @property
    def known_physical_cells(self) -> frozenset[Cell]:
        return frozenset(self._evidence_index) | self.verified_traversal_cells

    @property
    def physical_content_sha256(self) -> str:
        self._assert_config_integrity()
        return _canonical_sha256(self._physical_state_core())

    def physical_state(self, cell: Sequence[int]) -> PhysicalLabel:
        self._assert_config_integrity()
        normalized = _cell(cell)
        entries = self._evidence_index.get(normalized, ())
        traversed = normalized in self.verified_traversal_cells
        if traversed:
            entries = tuple(
                entry
                for entry in entries
                if entry[0] is EvidenceAuthority.EXACT_PHYSICAL
            )
            if not entries:
                return PhysicalLabel.FREE
        labels = {entry[1] for entry in entries}
        if len(labels) == 1:
            return next(iter(labels))
        return PhysicalLabel.UNKNOWN

    def _assert_config_integrity(self, *, transaction_error: bool = False) -> None:
        if self._config.content_sha256 != self._config_content_sha256:
            error = TransactionRejectedError if transaction_error else RuntimeError
            raise error("physical memory config was mutated after construction")

    def _physical_state_core(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_revisioned_physical_memory_state_v3",
            "config": self._config.to_dict(),
            "config_sha256": self._config_content_sha256,
            "revision": self._revision,
            "bound_camera_transform_sha256": (
                self._bound_camera_transform_sha256
            ),
            "exact_sim_tainted": self._exact_sim_tainted,
            "seen_observation_ids": sorted(self._seen_observation_ids),
            "seen_transaction_keys": sorted(self._seen_transaction_keys),
            "seen_semantic_transaction_keys": sorted(
                self._seen_semantic_transaction_keys
            ),
            "transactions": list(self._transaction_log),
            "active_observations": [
                self._observation_records[key].to_dict()
                for key in sorted(self._observation_records)
            ],
            "traversals": [
                {
                    **self._traversals[key].to_dict(),
                    "covered_cells": _cells_json(self._traversal_cells[key]),
                }
                for key in sorted(self._traversals)
            ],
            "execution_blocks": [
                {
                    **self._execution_blocks[key].to_dict(),
                    "body_center_cell": list(self._execution_block_cells[key]),
                }
                for key in sorted(self._execution_blocks)
            ],
        }

    def to_dict(self) -> dict[str, object]:
        self._assert_config_integrity()
        core = self._physical_state_core()
        payload = {**core, "physical_content_sha256": _canonical_sha256(core)}
        clone = _canonical_clone(payload)
        if not isinstance(clone, dict):
            raise AssertionError("canonical memory-state clone changed type")
        return clone

    def serialize(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RevisionedPhysicalMemory":
        return _revisioned_physical_memory_from_mapping(value)

    @classmethod
    def deserialize(cls, encoded: bytes) -> "RevisionedPhysicalMemory":
        if not isinstance(encoded, bytes):
            raise TypeError("serialized physical memory must be bytes")
        try:
            parsed = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("serialized physical memory is not UTF-8 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("serialized physical memory root must be an object")
        if encoded != _canonical_json_bytes(parsed) + b"\n":
            raise ValueError("serialized physical memory is not canonical JSON")
        memory = cls.from_mapping(parsed)
        if memory.serialize() != encoded:
            raise ValueError("serialized physical memory changed after typed replay")
        return memory

    def _build_exact_physical_transaction(
        self,
        *,
        observation: ObservationIdentity,
        pose: PoseProvenance,
        physical_evidence: tuple[PhysicalCellEvidence, ...],
        observed_unknown_cells: tuple[Cell, ...],
        source_semantics: str,
        label_inflation_radius_m: float,
    ) -> PhysicalEvidenceTransaction:
        """Mint the opaque exact admission used only by the exact adapter."""

        self._assert_config_integrity(transaction_error=True)
        if self._config.promoted_runtime:
            raise TransactionRejectedError(
                "exact physical evidence is forbidden in promoted runtime"
            )
        if observation.authority is not EvidenceAuthority.EXACT_PHYSICAL:
            raise ValueError("exact adapter requires EXACT_PHYSICAL authority")
        admission = ExactPhysicalAdmission(
            payload_sha256=observation.payload_sha256,
            observation_sha256=_canonical_sha256(observation.to_dict()),
            pose_sha256=pose.content_sha256,
            projection_contract_sha256=(
                self._config.physical_projection_contract_sha256
            ),
            calibration_sha256=pose.camera_transform_sha256,
            pose_uncertainty_contract_sha256=(
                self._config.pose_uncertainty_contract_sha256
            ),
            source_semantics=source_semantics,
            label_inflation_radius_m=label_inflation_radius_m,
            exact_sim_tainted=True,
        )
        return PhysicalEvidenceTransaction(
            observation=observation,
            map_frame=self.map_frame,
            pose=pose,
            physical_evidence=physical_evidence,
            observed_unknown_cells=observed_unknown_cells,
            exact_admission=admission,
            projection_contract_sha256=(
                self._config.physical_projection_contract_sha256
            ),
        )

    def apply_transaction(
        self, transaction: PhysicalEvidenceTransaction
    ) -> TransactionReceipt:
        """Validate completely, then commit exactly one monotonic revision."""

        return self._apply_transaction(transaction)

    def _apply_transaction(
        self,
        transaction: PhysicalEvidenceTransaction,
    ) -> TransactionReceipt:
        if not isinstance(transaction, PhysicalEvidenceTransaction):
            raise TypeError("transaction must be a PhysicalEvidenceTransaction")
        transaction_payload = _canonical_clone(transaction.to_dict())
        if not isinstance(transaction_payload, dict):
            raise AssertionError("canonical transaction clone changed type")
        transaction = _transaction_from_mapping(transaction_payload)
        self._assert_config_integrity(transaction_error=True)
        if transaction.map_frame != self.map_frame:
            raise TransactionRejectedError("transaction map frame/origin mismatch")
        if transaction.pose.frame_id != self.map_frame.frame_id:
            raise TransactionRejectedError("pose frame does not match map frame")
        if (
            transaction.projection_contract_sha256
            != self._config.physical_projection_contract_sha256
        ):
            raise TransactionRejectedError("physical projection contract mismatch")
        if self._config.promoted_runtime:
            authority = transaction.observation.authority
            if authority is EvidenceAuthority.EXACT_PHYSICAL:
                raise TransactionRejectedError(
                    "exact physical evidence is forbidden in promoted runtime"
                )
            if authority is EvidenceAuthority.LEARNED_PHYSICAL and (
                transaction.physical_evidence
                or transaction.observed_unknown_cells
            ):
                raise TransactionRejectedError(
                    "promoted learned evidence requires the qualified projection adapter"
                )
            if authority in {
                EvidenceAuthority.EXECUTOR_OUTCOME,
                EvidenceAuthority.RESET_CLEARANCE,
            }:
                raise TransactionRejectedError(
                    "promoted executor/reset admission is structurally unavailable"
                )
            if transaction.verified_traversals or transaction.execution_blocks:
                raise TransactionRejectedError(
                    "caller-built promoted traversal/block evidence is forbidden; "
                    "no issued outcome adapter is available"
                )
        if (
            transaction.pose.source is PoseSource.EXACT_SIM_ODOMETRY_ABLATION
            and not self._config.allow_exact_sim_odometry_ablation
        ):
            raise TransactionRejectedError(
                "exact simulator odometry is forbidden outside the explicit ablation"
            )
        pose_diagonal = tuple(
            transaction.pose.covariance_xy_yaw[index][index] for index in range(3)
        )
        if any(
            value > limit + _EPS
            for value, limit in zip(
                pose_diagonal, self._config.pose_covariance_diagonal_limits
            )
        ):
            raise TransactionRejectedError("pose covariance exceeds frozen limits")
        next_camera_transform_sha256 = (
            self._bound_camera_transform_sha256
            or transaction.pose.camera_transform_sha256
        )
        if transaction.pose.camera_transform_sha256 != next_camera_transform_sha256:
            raise TransactionRejectedError("camera calibration binding mismatch")
        if (
            self._config.expected_camera_transform_sha256 is not None
            and next_camera_transform_sha256
            != self._config.expected_camera_transform_sha256
        ):
            raise TransactionRejectedError("camera calibration differs from config")
        observation_id = transaction.observation.observation_id
        transaction_key = transaction.transaction_key_sha256
        semantic_key = transaction.semantic_transaction_sha256
        if observation_id in self._seen_observation_ids:
            raise TransactionRejectedError("duplicate observation identity")
        if transaction_key in self._seen_transaction_keys:
            raise TransactionRejectedError("duplicate observation/map/pose key")
        if semantic_key in self._seen_semantic_transaction_keys:
            raise TransactionRejectedError("semantic duplicate physical transaction")
        if transaction.observation.authority is EvidenceAuthority.EXACT_PHYSICAL:
            admission = transaction.exact_admission
            if not isinstance(admission, ExactPhysicalAdmission):
                raise TransactionRejectedError("exact transaction bypassed adapter")
            exact_sim_tainted = True
            if (
                admission.projection_contract_sha256
                != self._config.physical_projection_contract_sha256
                or admission.calibration_sha256
                != transaction.pose.camera_transform_sha256
                or admission.pose_uncertainty_contract_sha256
                != self._config.pose_uncertainty_contract_sha256
                or admission.exact_sim_tainted is not exact_sim_tainted
            ):
                raise TransactionRejectedError(
                    "exact adapter admission contract mismatch"
                )
            for evidence in transaction.physical_evidence:
                existing_exact_labels = {
                    label
                    for authority, label, _observation_id in self._evidence_index.get(
                        evidence.cell, ()
                    )
                    if authority is EvidenceAuthority.EXACT_PHYSICAL
                }
                if existing_exact_labels and existing_exact_labels != {
                    evidence.label
                }:
                    raise TransactionRejectedError(
                        "conflicting exact physical evidence is forbidden"
                    )
        for target_id in transaction.retract_learned_observation_ids:
            record = self._observation_records.get(target_id)
            if record is None:
                raise TransactionRejectedError(
                    f"unknown learned observation retraction {target_id!r}"
                )
            if record.observation.authority is not EvidenceAuthority.LEARNED_PHYSICAL:
                raise TransactionRejectedError("exact physical evidence is not retractable")
        for traversal in transaction.verified_traversals:
            if traversal.traversal_id in self._traversals:
                raise TransactionRejectedError("traversal identity is indelible")
        for block in transaction.execution_blocks:
            if block.block_id in self._execution_blocks:
                raise TransactionRejectedError("execution block identity is indelible")

        rasterized_traversals = {
            traversal.traversal_id: _rasterize_verified_polygon(self.map_frame, traversal)
            for traversal in transaction.verified_traversals
        }
        records = dict(self._observation_records)
        retracted: set[str] = set(transaction.retract_learned_observation_ids)
        if (
            self._config.fusion_mode is FusionMode.CURRENT_FRAME_ONLY
            and transaction.observation.authority
            is EvidenceAuthority.LEARNED_PHYSICAL
        ):
            retracted.update(
                key
                for key, record in records.items()
                if record.observation.authority
                is EvidenceAuthority.LEARNED_PHYSICAL
            )
        for target_id in retracted:
            records.pop(target_id, None)

        revision_before = self._revision
        revision_after = revision_before + 1
        if transaction.physical_evidence:
            records[observation_id] = PhysicalObservationRecord(
                observation=transaction.observation,
                pose=transaction.pose,
                transaction_sha256=transaction.content_sha256,
                evidence=transaction.physical_evidence,
                observed_unknown_cells=transaction.observed_unknown_cells,
                revision_added=revision_after,
            )
        traversals = dict(self._traversals)
        traversal_cells = dict(self._traversal_cells)
        for traversal in transaction.verified_traversals:
            traversals[traversal.traversal_id] = traversal
            traversal_cells[traversal.traversal_id] = rasterized_traversals[
                traversal.traversal_id
            ]
        traversal_union: set[Cell] = set()
        for covered_cells in traversal_cells.values():
            traversal_union.update(covered_cells)
        blocks = dict(self._execution_blocks)
        block_cells = dict(self._execution_block_cells)
        for block in transaction.execution_blocks:
            blocks[block.block_id] = block
            block_cells[block.block_id] = self.map_frame.world_to_cell(
                block.body_center_xy_m
            )
        evidence_index = self._build_evidence_index(records)

        self._observation_records = records
        self._traversals = traversals
        self._traversal_cells = traversal_cells
        self._verified_traversal_union = frozenset(traversal_union)
        self._execution_blocks = blocks
        self._execution_block_cells = block_cells
        self._evidence_index = evidence_index
        self._seen_observation_ids.add(observation_id)
        self._seen_transaction_keys.add(transaction_key)
        self._seen_semantic_transaction_keys.add(semantic_key)
        self._transaction_log.append(transaction_payload)
        self._bound_camera_transform_sha256 = next_camera_transform_sha256
        if (
            transaction.observation.authority is EvidenceAuthority.EXACT_PHYSICAL
            or transaction.pose.source is PoseSource.EXACT_SIM_ODOMETRY_ABLATION
        ):
            self._exact_sim_tainted = True
        self._revision = revision_after
        self._issued_snapshot_sha256.clear()
        return TransactionReceipt(
            observation_id=observation_id,
            transaction_key_sha256=transaction_key,
            transaction_sha256=transaction.content_sha256,
            revision_before=revision_before,
            revision_after=revision_after,
            physical_evidence_cells=len(transaction.physical_evidence),
            verified_traversal_cells_added=sum(
                len(cells) for cells in rasterized_traversals.values()
            ),
            execution_blocks_added=len(transaction.execution_blocks),
            learned_observations_retracted=len(retracted),
        )

    @staticmethod
    def _build_evidence_index(
        records: Mapping[str, PhysicalObservationRecord],
    ) -> dict[Cell, tuple[tuple[EvidenceAuthority, PhysicalLabel, str], ...]]:
        mutable: dict[
            Cell, list[tuple[EvidenceAuthority, PhysicalLabel, str]]
        ] = {}
        for observation_id in sorted(records):
            record = records[observation_id]
            for evidence in record.evidence:
                mutable.setdefault(evidence.cell, []).append(
                    (
                        record.observation.authority,
                        evidence.label,
                        observation_id,
                    )
                )
        return {
            cell: tuple(sorted(entries, key=lambda entry: (entry[2], entry[1])))
            for cell, entries in mutable.items()
        }

    def create_configuration_snapshot(
        self,
        morphology: ConfigurationMorphology,
        *,
        candidate_cells: Iterable[Cell] | None = None,
    ) -> ConfigurationSnapshot:
        self._assert_config_integrity()
        if not isinstance(morphology, ConfigurationMorphology):
            raise TypeError("morphology must be a ConfigurationMorphology")
        morphology.assert_integrity()
        if not math.isclose(
            morphology.physical_cell_size_m,
            self.map_frame.cell_size_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("morphology cell size does not match physical memory")
        if self._config.require_registered_lattice and (
            not math.isclose(
                morphology.footprint_radius_m,
                REGISTERED_FOOTPRINT_RADIUS_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or len(morphology.free_support_offsets)
            != REGISTERED_FREE_SUPPORT_COUNT
            or len(morphology.occupied_support_offsets)
            != REGISTERED_OCCUPIED_SUPPORT_COUNT
            or morphology != ConfigurationMorphology()
        ):
            raise ValueError(
                "registered G3 snapshot requires exact 0.47 m 89/69 morphology"
            )
        candidates = (
            self._implicit_configuration_candidates(morphology)
            if candidate_cells is None
            else {_cell(cell, "candidate cell") for cell in candidate_cells}
        )
        free: set[Cell] = set()
        occupied: set[Cell] = set()
        unknown: set[Cell] = set()
        blocked_centers = self.execution_block_cells
        for candidate in sorted(candidates):
            if candidate in blocked_centers or any(
                self.physical_state(
                    (candidate[0] + offset[0], candidate[1] + offset[1])
                )
                is PhysicalLabel.OCCUPIED
                for offset in morphology.occupied_support_offsets
            ):
                occupied.add(candidate)
            elif all(
                self.physical_state(
                    (candidate[0] + offset[0], candidate[1] + offset[1])
                )
                is PhysicalLabel.FREE
                for offset in morphology.free_support_offsets
            ):
                free.add(candidate)
            else:
                unknown.add(candidate)
        snapshot = ConfigurationSnapshot(
            map_frame_sha256=self.map_frame.content_sha256,
            memory_config_sha256=self._config_content_sha256,
            physical_revision=self._revision,
            physical_content_sha256=self.physical_content_sha256,
            exact_sim_tainted=self._exact_sim_tainted,
            morphology_sha256=morphology.content_sha256,
            free_support_sha256=morphology.free_support_sha256,
            occupied_support_sha256=morphology.occupied_support_sha256,
            planning_connectivity=self._config.planning_connectivity,
            allow_diagonal_corner_cutting=(
                self._config.allow_diagonal_corner_cutting
            ),
            free_cells=frozenset(free),
            occupied_cells=frozenset(occupied),
            unknown_cells=frozenset(unknown),
        )
        self._issued_snapshot_sha256.add(snapshot.content_sha256)
        return snapshot

    def _implicit_configuration_candidates(
        self, morphology: ConfigurationMorphology
    ) -> set[Cell]:
        candidates = set(self.execution_block_cells)
        offsets = set(morphology.free_support_offsets) | set(
            morphology.occupied_support_offsets
        )
        for physical_cell in self.known_physical_cells:
            for offset in offsets:
                candidates.add(
                    (
                        physical_cell[0] - offset[0],
                        physical_cell[1] - offset[1],
                    )
                )
        return candidates

    def assert_current_snapshot(self, snapshot: ConfigurationSnapshot) -> None:
        self._assert_config_integrity()
        if not isinstance(snapshot, ConfigurationSnapshot):
            raise TypeError("snapshot must be a ConfigurationSnapshot")
        snapshot.assert_integrity()
        if snapshot.map_frame_sha256 != self.map_frame.content_sha256:
            raise SnapshotBindingError("snapshot map-frame hash mismatch")
        if snapshot.physical_revision != self._revision:
            raise StaleSnapshotError(
                f"snapshot revision {snapshot.physical_revision} is stale; "
                f"memory is at {self._revision}"
            )
        if snapshot.physical_content_sha256 != self.physical_content_sha256:
            raise SnapshotBindingError("snapshot physical-content hash mismatch")
        if (
            snapshot.memory_config_sha256 != self._config_content_sha256
            or snapshot.exact_sim_tainted is not self._exact_sim_tainted
        ):
            raise SnapshotBindingError("snapshot memory-config/taint mismatch")
        if snapshot.content_sha256 not in self._issued_snapshot_sha256:
            raise SnapshotBindingError("snapshot was not derived by this memory")


@dataclass(frozen=True)
class ConfigurationComponent:
    snapshot_sha256: str
    physical_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    start_cell: Cell
    cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        start = _cell(self.start_cell, "start_cell")
        cells = frozenset(_cell(cell) for cell in self.cells)
        object.__setattr__(self, "start_cell", start)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                {
                    "schema": "lewm_g3_configuration_component_v1",
                    "snapshot_sha256": self.snapshot_sha256,
                    "physical_revision": self.physical_revision,
                    "free_support_sha256": self.free_support_sha256,
                    "occupied_support_sha256": self.occupied_support_sha256,
                    "start_cell": list(start),
                    "cells": _cells_json(cells),
                }
            ),
        )

    def assert_integrity(self) -> None:
        expected = ConfigurationComponent(
            snapshot_sha256=self.snapshot_sha256,
            physical_revision=self.physical_revision,
            free_support_sha256=self.free_support_sha256,
            occupied_support_sha256=self.occupied_support_sha256,
            start_cell=self.start_cell,
            cells=self.cells,
        )
        if self.content_sha256 != expected.content_sha256:
            raise SnapshotBindingError("configuration component was mutated")


@dataclass(frozen=True)
class ConfigurationFrontiers:
    snapshot_sha256: str
    physical_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class ConfigurationPath:
    snapshot_sha256: str
    physical_revision: int
    free_support_sha256: str
    occupied_support_sha256: str
    cells: tuple[Cell, ...]
    cost: float

    def __post_init__(self) -> None:
        _validate_sha256(self.snapshot_sha256, "snapshot_sha256")
        _validate_sha256(self.free_support_sha256, "free_support_sha256")
        _validate_sha256(self.occupied_support_sha256, "occupied_support_sha256")
        object.__setattr__(
            self,
            "physical_revision",
            _nonnegative_int(self.physical_revision, "physical_revision"),
        )
        cells = tuple(_cell(cell) for cell in self.cells)
        if not cells:
            raise ValueError("configuration path cannot be empty")
        object.__setattr__(self, "cells", cells)
        cost = _finite(self.cost, "path cost")
        if cost < 0.0:
            raise ValueError("path cost must be non-negative")
        object.__setattr__(self, "cost", cost)


class ConfigurationPlanner:
    """Revision-bound deterministic planning over configuration snapshots."""

    def __init__(
        self,
        memory: RevisionedPhysicalMemory,
        morphology: ConfigurationMorphology,
    ) -> None:
        if not isinstance(memory, RevisionedPhysicalMemory):
            raise TypeError("memory must be a RevisionedPhysicalMemory")
        if not isinstance(morphology, ConfigurationMorphology):
            raise TypeError("morphology must be a ConfigurationMorphology")
        self._memory = memory
        self._morphology = morphology

    def _assert_current(self, snapshot: ConfigurationSnapshot) -> None:
        self._morphology.assert_integrity()
        if (
            snapshot.morphology_sha256 != self._morphology.content_sha256
            or snapshot.free_support_sha256
            != self._morphology.free_support_sha256
            or snapshot.occupied_support_sha256
            != self._morphology.occupied_support_sha256
        ):
            raise SnapshotBindingError("snapshot morphology binding mismatch")
        self._memory.assert_current_snapshot(snapshot)

    @staticmethod
    def _result_binding(snapshot: ConfigurationSnapshot) -> dict[str, object]:
        return {
            "snapshot_sha256": snapshot.content_sha256,
            "physical_revision": snapshot.physical_revision,
            "free_support_sha256": snapshot.free_support_sha256,
            "occupied_support_sha256": snapshot.occupied_support_sha256,
        }

    def _neighbors(
        self, snapshot: ConfigurationSnapshot, cell: Cell
    ) -> tuple[tuple[Cell, float], ...]:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if snapshot.planning_connectivity == 8:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        neighbors: list[tuple[Cell, float]] = []
        for dx, dy in sorted(offsets):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if neighbor not in snapshot.free_cells:
                continue
            if dx != 0 and dy != 0:
                if (
                    (cell[0] + dx, cell[1]) not in snapshot.free_cells
                    or (cell[0], cell[1] + dy) not in snapshot.free_cells
                ):
                    continue
                cost = math.sqrt(2.0)
            else:
                cost = 1.0
            neighbors.append((neighbor, cost))
        return tuple(neighbors)

    def connected_component(
        self,
        snapshot: ConfigurationSnapshot,
        start_cell: Sequence[int],
    ) -> ConfigurationComponent:
        self._assert_current(snapshot)
        start = _cell(start_cell, "start_cell")
        reached: set[Cell] = set()
        if start in snapshot.free_cells:
            reached.add(start)
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor, _cost in self._neighbors(snapshot, current):
                    if neighbor not in reached:
                        reached.add(neighbor)
                        queue.append(neighbor)
        return ConfigurationComponent(
            **self._result_binding(snapshot),
            start_cell=start,
            cells=frozenset(reached),
        )

    def frontier_cells(
        self,
        snapshot: ConfigurationSnapshot,
        component: ConfigurationComponent,
    ) -> ConfigurationFrontiers:
        self._assert_current(snapshot)
        self._assert_component_binding(snapshot, component)
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
        frontiers = tuple(
            sorted(
                cell
                for cell in component.cells
                if any(
                    (cell[0] + dx, cell[1] + dy) in snapshot.unknown_cells
                    for dx, dy in offsets
                )
            )
        )
        return ConfigurationFrontiers(
            **self._result_binding(snapshot),
            cells=frontiers,
        )

    def astar(
        self,
        snapshot: ConfigurationSnapshot,
        start_cell: Sequence[int],
        goal_cell: Sequence[int],
    ) -> ConfigurationPath | None:
        self._assert_current(snapshot)
        start = _cell(start_cell, "start_cell")
        goal = _cell(goal_cell, "goal_cell")
        if start not in snapshot.free_cells or goal not in snapshot.free_cells:
            return None
        best_cost: dict[Cell, float] = {start: 0.0}
        parent: dict[Cell, Cell | None] = {start: None}
        start_h = self._heuristic(snapshot, start, goal)
        open_heap: list[tuple[float, float, float, int, int]] = [
            (start_h, start_h, 0.0, start[0], start[1])
        ]
        closed: set[Cell] = set()
        while open_heap:
            _f, _h, g_cost, x, y = heapq.heappop(open_heap)
            current = (x, y)
            if current in closed or g_cost > best_cost.get(current, math.inf) + _EPS:
                continue
            if current == goal:
                cells: list[Cell] = []
                cursor: Cell | None = goal
                while cursor is not None:
                    cells.append(cursor)
                    cursor = parent[cursor]
                path = ConfigurationPath(
                    **self._result_binding(snapshot),
                    cells=tuple(reversed(cells)),
                    cost=g_cost,
                )
                self.validate_path(snapshot, path)
                return path
            closed.add(current)
            for neighbor, step_cost in self._neighbors(snapshot, current):
                candidate_cost = g_cost + step_cost
                previous_cost = best_cost.get(neighbor, math.inf)
                previous_parent = parent.get(neighbor)
                improves = candidate_cost < previous_cost - _EPS
                tie_improves = (
                    abs(candidate_cost - previous_cost) <= _EPS
                    and (previous_parent is None or current < previous_parent)
                )
                if not (improves or tie_improves):
                    continue
                best_cost[neighbor] = candidate_cost
                parent[neighbor] = current
                heuristic = self._heuristic(snapshot, neighbor, goal)
                heapq.heappush(
                    open_heap,
                    (
                        candidate_cost + heuristic,
                        heuristic,
                        candidate_cost,
                        neighbor[0],
                        neighbor[1],
                    ),
                )
        return None

    def validate_path(
        self,
        snapshot: ConfigurationSnapshot,
        path: ConfigurationPath,
    ) -> None:
        self._assert_current(snapshot)
        if not isinstance(path, ConfigurationPath):
            raise TypeError("path must be a ConfigurationPath")
        if (
            path.snapshot_sha256 != snapshot.content_sha256
            or path.physical_revision != snapshot.physical_revision
            or path.free_support_sha256 != snapshot.free_support_sha256
            or path.occupied_support_sha256 != snapshot.occupied_support_sha256
        ):
            raise StalePathError("path is not bound to the supplied snapshot")
        if any(cell not in snapshot.free_cells for cell in path.cells):
            raise InvalidConfigurationPathError("path contains a non-free cell")
        expected_cost = 0.0
        for start, end in zip(path.cells, path.cells[1:]):
            options = dict(self._neighbors(snapshot, start))
            if end not in options:
                raise InvalidConfigurationPathError(
                    "path contains a non-adjacent or corner-cutting step"
                )
            expected_cost += options[end]
        if not math.isclose(expected_cost, path.cost, rel_tol=0.0, abs_tol=1e-12):
            raise InvalidConfigurationPathError("path cost does not match its cells")

    @staticmethod
    def _heuristic(
        snapshot: ConfigurationSnapshot, start: Cell, goal: Cell
    ) -> float:
        dx = abs(goal[0] - start[0])
        dy = abs(goal[1] - start[1])
        if snapshot.planning_connectivity == 4:
            return float(dx + dy)
        diagonal = min(dx, dy)
        cardinal = max(dx, dy) - diagonal
        return float(cardinal + math.sqrt(2.0) * diagonal)

    def _assert_component_binding(
        self,
        snapshot: ConfigurationSnapshot,
        component: ConfigurationComponent,
    ) -> None:
        if not isinstance(component, ConfigurationComponent):
            raise TypeError("component must be a ConfigurationComponent")
        component.assert_integrity()
        if (
            component.snapshot_sha256 != snapshot.content_sha256
            or component.physical_revision != snapshot.physical_revision
            or component.free_support_sha256 != snapshot.free_support_sha256
            or component.occupied_support_sha256
            != snapshot.occupied_support_sha256
        ):
            raise SnapshotBindingError("component is not bound to snapshot")
        if not component.cells <= snapshot.free_cells:
            raise SnapshotBindingError("component contains non-free cells")
        expected: set[Cell] = set()
        if component.start_cell in snapshot.free_cells:
            expected.add(component.start_cell)
            queue = deque([component.start_cell])
            while queue:
                current = queue.popleft()
                for neighbor, _cost in self._neighbors(snapshot, current):
                    if neighbor not in expected:
                        expected.add(neighbor)
                        queue.append(neighbor)
        if component.cells != frozenset(expected):
            raise SnapshotBindingError(
                "component is not the complete connected snapshot component"
            )


def _strict_mapping(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields changed")
    return value


def _strict_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _map_frame_from_mapping(value: object) -> MapFrameIdentity:
    raw = _strict_mapping(
        value,
        {"schema", "session_id", "origin_xy_m", "cell_size_m", "frame_id"},
        name="map frame",
    )
    if raw["schema"] != "lewm_g3_map_frame_identity_v1":
        raise ValueError("map-frame schema changed")
    frame = MapFrameIdentity(
        session_id=raw["session_id"],  # type: ignore[arg-type]
        origin_xy_m=_strict_list(raw["origin_xy_m"], name="map origin"),
        cell_size_m=raw["cell_size_m"],  # type: ignore[arg-type]
        frame_id=raw["frame_id"],  # type: ignore[arg-type]
    )
    if frame.to_dict() != dict(raw):
        raise ValueError("map frame is not canonical")
    return frame


def _memory_config_from_mapping(value: object) -> PhysicalMemoryConfig:
    raw = _strict_mapping(
        value,
        {
            "schema",
            "map_frame",
            "fusion_mode",
            "planning_connectivity",
            "allow_diagonal_corner_cutting",
            "allow_exact_sim_odometry_ablation",
            "require_registered_lattice",
            "physical_projection_contract_sha256",
            "expected_camera_transform_sha256",
            "pose_covariance_diagonal_limits",
            "pose_uncertainty_contract_sha256",
            "promoted_runtime",
        },
        name="physical memory config",
    )
    if raw["schema"] != "lewm_g3_physical_memory_config_v1":
        raise ValueError("physical-memory config schema changed")
    try:
        fusion_mode = FusionMode(raw["fusion_mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("physical-memory fusion mode changed") from exc
    config = PhysicalMemoryConfig(
        map_frame=_map_frame_from_mapping(raw["map_frame"]),
        fusion_mode=fusion_mode,
        planning_connectivity=raw["planning_connectivity"],  # type: ignore[arg-type]
        allow_diagonal_corner_cutting=raw["allow_diagonal_corner_cutting"],  # type: ignore[arg-type]
        allow_exact_sim_odometry_ablation=raw[
            "allow_exact_sim_odometry_ablation"
        ],  # type: ignore[arg-type]
        require_registered_lattice=raw["require_registered_lattice"],  # type: ignore[arg-type]
        physical_projection_contract_sha256=raw[
            "physical_projection_contract_sha256"
        ],  # type: ignore[arg-type]
        expected_camera_transform_sha256=raw[
            "expected_camera_transform_sha256"
        ],  # type: ignore[arg-type]
        pose_covariance_diagonal_limits=_strict_list(
            raw["pose_covariance_diagonal_limits"],
            name="pose covariance diagonal limits",
        ),
        promoted_runtime=raw["promoted_runtime"],  # type: ignore[arg-type]
    )
    if config.to_dict() != dict(raw):
        raise ValueError("physical memory config is not canonical")
    return config


def _observation_from_mapping(value: object) -> ObservationIdentity:
    raw = _strict_mapping(
        value,
        {
            "observation_id",
            "payload_sha256",
            "producer_sha256",
            "authority",
        },
        name="observation identity",
    )
    try:
        authority = EvidenceAuthority(raw["authority"])
    except (TypeError, ValueError) as exc:
        raise ValueError("observation authority changed") from exc
    observation = ObservationIdentity(
        observation_id=raw["observation_id"],  # type: ignore[arg-type]
        payload_sha256=raw["payload_sha256"],  # type: ignore[arg-type]
        producer_sha256=raw["producer_sha256"],  # type: ignore[arg-type]
        authority=authority,
    )
    if observation.to_dict() != dict(raw):
        raise ValueError("observation identity is not canonical")
    return observation


def _pose_from_mapping(value: object) -> PoseProvenance:
    raw = _strict_mapping(
        value,
        {
            "source",
            "frame_id",
            "mean_xy_yaw",
            "covariance_xy_yaw",
            "timestamp_ns",
            "synchronization_id",
            "camera_transform_sha256",
        },
        name="pose provenance",
    )
    try:
        source = PoseSource(raw["source"])
    except (TypeError, ValueError) as exc:
        raise ValueError("pose source changed") from exc
    covariance_rows = _strict_list(
        raw["covariance_xy_yaw"], name="pose covariance"
    )
    pose = PoseProvenance(
        source=source,
        frame_id=raw["frame_id"],  # type: ignore[arg-type]
        mean_xy_yaw=_strict_list(raw["mean_xy_yaw"], name="pose mean"),
        covariance_xy_yaw=tuple(
            _strict_list(row, name="pose covariance row")
            for row in covariance_rows
        ),
        timestamp_ns=raw["timestamp_ns"],  # type: ignore[arg-type]
        synchronization_id=raw["synchronization_id"],  # type: ignore[arg-type]
        camera_transform_sha256=raw["camera_transform_sha256"],  # type: ignore[arg-type]
    )
    if pose.to_dict() != dict(raw):
        raise ValueError("pose provenance is not canonical")
    return pose


def _physical_evidence_from_mapping(value: object) -> PhysicalCellEvidence:
    raw = _strict_mapping(value, {"cell", "label"}, name="physical evidence")
    try:
        label = PhysicalLabel(raw["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError("physical evidence label changed") from exc
    evidence = PhysicalCellEvidence(
        cell=_strict_list(raw["cell"], name="physical evidence cell"),
        label=label,
    )
    if evidence.to_dict() != dict(raw):
        raise ValueError("physical evidence is not canonical")
    return evidence


def _traversal_from_mapping(value: object) -> VerifiedTraversalPolygon:
    raw = _strict_mapping(
        value,
        {"traversal_id", "vertices_xy_m", "outcome_sha256"},
        name="verified traversal",
    )
    vertices = _strict_list(raw["vertices_xy_m"], name="traversal vertices")
    traversal = VerifiedTraversalPolygon(
        traversal_id=raw["traversal_id"],  # type: ignore[arg-type]
        vertices_xy_m=tuple(
            _strict_list(vertex, name="traversal vertex") for vertex in vertices
        ),
        outcome_sha256=raw["outcome_sha256"],  # type: ignore[arg-type]
    )
    if traversal.to_dict() != dict(raw):
        raise ValueError("verified traversal is not canonical")
    return traversal


def _execution_block_from_mapping(value: object) -> ExecutionBlock:
    raw = _strict_mapping(
        value,
        {"block_id", "body_center_xy_m", "kind", "outcome_sha256"},
        name="execution block",
    )
    try:
        kind = ExecutionBlockKind(raw["kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("execution-block kind changed") from exc
    block = ExecutionBlock(
        block_id=raw["block_id"],  # type: ignore[arg-type]
        body_center_xy_m=_strict_list(
            raw["body_center_xy_m"], name="block body center"
        ),
        kind=kind,
        outcome_sha256=raw["outcome_sha256"],  # type: ignore[arg-type]
    )
    if block.to_dict() != dict(raw):
        raise ValueError("execution block is not canonical")
    return block


def _exact_admission_from_mapping(value: object) -> ExactPhysicalAdmission:
    raw = _strict_mapping(
        value,
        {
            "schema",
            "payload_sha256",
            "observation_sha256",
            "pose_sha256",
            "projection_contract_sha256",
            "calibration_sha256",
            "pose_uncertainty_contract_sha256",
            "source_semantics",
            "label_inflation_radius_m",
            "exact_sim_tainted",
        },
        name="exact physical admission",
    )
    if raw["schema"] != "lewm_g3_exact_physical_admission_v1":
        raise ValueError("exact physical admission schema changed")
    admission = ExactPhysicalAdmission(
        payload_sha256=raw["payload_sha256"],  # type: ignore[arg-type]
        observation_sha256=raw["observation_sha256"],  # type: ignore[arg-type]
        pose_sha256=raw["pose_sha256"],  # type: ignore[arg-type]
        projection_contract_sha256=raw[
            "projection_contract_sha256"
        ],  # type: ignore[arg-type]
        calibration_sha256=raw["calibration_sha256"],  # type: ignore[arg-type]
        pose_uncertainty_contract_sha256=raw[
            "pose_uncertainty_contract_sha256"
        ],  # type: ignore[arg-type]
        source_semantics=raw["source_semantics"],  # type: ignore[arg-type]
        label_inflation_radius_m=raw["label_inflation_radius_m"],  # type: ignore[arg-type]
        exact_sim_tainted=raw["exact_sim_tainted"],  # type: ignore[arg-type]
    )
    if admission.to_dict() != dict(raw):
        raise ValueError("exact physical admission is not canonical")
    return admission


def _execution_admission_from_mapping(
    value: object,
) -> ExecutionEvidenceAdmission:
    raw = _strict_mapping(
        value,
        {
            "schema",
            "admission_id_sha256",
            "adapter_instance_sha256",
            "source_memory_instance_sha256",
            "receipt_content_sha256",
            "adapter_contract_sha256",
            "body_support_contract_sha256",
            "map_frame_sha256",
            "observation_sha256",
            "pose_sha256",
            "evidence_content_sha256",
            "memory_revision_before",
            "evidence_kind",
            "content_sha256",
        },
        name="execution evidence admission",
    )
    if raw["schema"] != "lewm_g3_execution_evidence_admission_v1":
        raise ValueError("execution evidence admission schema changed")
    try:
        evidence_kind = ExecutionEvidenceKind(raw["evidence_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("execution evidence kind changed") from exc
    admission = ExecutionEvidenceAdmission(
        admission_id_sha256=raw["admission_id_sha256"],  # type: ignore[arg-type]
        adapter_instance_sha256=raw["adapter_instance_sha256"],  # type: ignore[arg-type]
        source_memory_instance_sha256=raw["source_memory_instance_sha256"],  # type: ignore[arg-type]
        receipt_content_sha256=raw["receipt_content_sha256"],  # type: ignore[arg-type]
        adapter_contract_sha256=raw["adapter_contract_sha256"],  # type: ignore[arg-type]
        body_support_contract_sha256=raw["body_support_contract_sha256"],  # type: ignore[arg-type]
        map_frame_sha256=raw["map_frame_sha256"],  # type: ignore[arg-type]
        observation_sha256=raw["observation_sha256"],  # type: ignore[arg-type]
        pose_sha256=raw["pose_sha256"],  # type: ignore[arg-type]
        evidence_content_sha256=raw["evidence_content_sha256"],  # type: ignore[arg-type]
        memory_revision_before=raw["memory_revision_before"],  # type: ignore[arg-type]
        evidence_kind=evidence_kind,
    )
    if admission.to_dict() != dict(raw):
        raise ValueError("execution evidence admission is not canonical")
    return admission


def _transaction_from_mapping(value: object) -> PhysicalEvidenceTransaction:
    raw = _strict_mapping(
        value,
        {
            "schema",
            "transaction_key_sha256",
            "semantic_transaction_sha256",
            "observation",
            "map_frame",
            "projection_contract_sha256",
            "pose",
            "physical_evidence",
            "observed_unknown_cells",
            "verified_traversals",
            "execution_blocks",
            "retract_learned_observation_ids",
            "exact_admission",
            "execution_admission",
        },
        name="physical transaction",
    )
    if raw["schema"] != "lewm_g3_physical_evidence_transaction_v2":
        raise ValueError("physical transaction schema changed")
    evidence_rows = _strict_list(raw["physical_evidence"], name="physical evidence")
    unknown_rows = _strict_list(
        raw["observed_unknown_cells"], name="observed unknown cells"
    )
    traversal_rows = _strict_list(
        raw["verified_traversals"], name="verified traversals"
    )
    block_rows = _strict_list(raw["execution_blocks"], name="execution blocks")
    retractions = _strict_list(
        raw["retract_learned_observation_ids"], name="learned retractions"
    )
    admission = (
        None
        if raw["exact_admission"] is None
        else _exact_admission_from_mapping(raw["exact_admission"])
    )
    execution_admission = (
        None
        if raw["execution_admission"] is None
        else _execution_admission_from_mapping(raw["execution_admission"])
    )
    transaction = PhysicalEvidenceTransaction(
        observation=_observation_from_mapping(raw["observation"]),
        map_frame=_map_frame_from_mapping(raw["map_frame"]),
        projection_contract_sha256=raw[
            "projection_contract_sha256"
        ],  # type: ignore[arg-type]
        pose=_pose_from_mapping(raw["pose"]),
        physical_evidence=tuple(
            _physical_evidence_from_mapping(row) for row in evidence_rows
        ),
        observed_unknown_cells=tuple(
            _strict_list(row, name="observed unknown cell") for row in unknown_rows
        ),
        verified_traversals=tuple(
            _traversal_from_mapping(row) for row in traversal_rows
        ),
        execution_blocks=tuple(
            _execution_block_from_mapping(row) for row in block_rows
        ),
        retract_learned_observation_ids=tuple(retractions),  # type: ignore[arg-type]
        exact_admission=admission,
        execution_admission=execution_admission,
    )
    if transaction.to_dict() != dict(raw):
        raise ValueError("physical transaction is not canonical")
    return transaction


def _revisioned_physical_memory_from_mapping(
    value: Mapping[str, object],
) -> RevisionedPhysicalMemory:
    expected_fields = {
        "schema",
        "config",
        "config_sha256",
        "revision",
        "bound_camera_transform_sha256",
        "exact_sim_tainted",
        "seen_observation_ids",
        "seen_transaction_keys",
        "seen_semantic_transaction_keys",
        "transactions",
        "active_observations",
        "traversals",
        "execution_blocks",
        "physical_content_sha256",
    }
    raw = _strict_mapping(value, expected_fields, name="physical memory state")
    if raw["schema"] != "lewm_g3_revisioned_physical_memory_state_v3":
        raise ValueError("physical memory state schema changed")
    _validate_sha256(raw["physical_content_sha256"], "physical_content_sha256")  # type: ignore[arg-type]
    core = dict(raw)
    claimed_hash = core.pop("physical_content_sha256")
    if claimed_hash != _canonical_sha256(core):
        raise ValueError("physical memory content hash changed")
    config = _memory_config_from_mapping(raw["config"])
    if raw["config_sha256"] != config.content_sha256:
        raise ValueError("physical memory config hash changed")
    memory = RevisionedPhysicalMemory(config)
    transactions = _strict_list(raw["transactions"], name="transaction log")
    for transaction_row in transactions:
        # Pure typed replay uses the same public fail-closed admission path.
        # It has no token or hook that can be reused to admit new evidence.
        memory.apply_transaction(_transaction_from_mapping(transaction_row))
    if _canonical_json_bytes(memory.to_dict()) != _canonical_json_bytes(dict(raw)):
        raise ValueError("serialized physical memory does not replay exactly")
    return memory


__all__ = [
    "Cell",
    "ConfigurationComponent",
    "ConfigurationFrontiers",
    "ConfigurationMorphology",
    "ConfigurationPath",
    "ConfigurationPlanner",
    "ConfigurationSnapshot",
    "EvidenceAuthority",
    "ExecutionBlock",
    "ExecutionBlockKind",
    "ExecutionEvidenceAdmission",
    "ExecutionEvidenceKind",
    "FusionMode",
    "InvalidConfigurationPathError",
    "MapFrameIdentity",
    "ObservationIdentity",
    "PhysicalCellEvidence",
    "PhysicalEvidenceTransaction",
    "PhysicalLabel",
    "PhysicalMemoryConfig",
    "PoseProvenance",
    "PoseSource",
    "REGISTERED_FOOTPRINT_RADIUS_M",
    "REGISTERED_FREE_SUPPORT_COUNT",
    "REGISTERED_OCCUPIED_SUPPORT_COUNT",
    "REGISTERED_PHYSICAL_CELL_SIZE_M",
    "REGISTERED_PHYSICAL_PROJECTION_CONTRACT_SHA256",
    "REGISTERED_POSE_COVARIANCE_DIAGONAL_LIMITS",
    "RevisionedPhysicalMemory",
    "SnapshotBindingError",
    "StalePathError",
    "StaleSnapshotError",
    "TransactionReceipt",
    "TransactionRejectedError",
    "VerifiedTraversalPolygon",
    "ZERO_INFLATION_EXACT_PHYSICAL_SEMANTICS",
]
