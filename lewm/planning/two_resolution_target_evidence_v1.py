"""Synthetic-only G3 V2 to G5 two-resolution target-evidence boundary.

This module is intentionally additive.  It does not replace the passed G5
posterior implementation and it has no production authority.  Its narrow job
is to bind an exact live G3 V2 snapshot/component to runner-owned 0.05 m V5
evidence and issue immutable 0.10 m target-evidence records.

Negative evidence is conservative across resolutions: a configuration cell is
visible only when all four of its physical children are visible, certified
free, and carry a detection probability.  The configuration-cell probability
is the minimum of those four probabilities.
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
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_SHA256,
    PHYSICAL_CELL_SIZE_M,
    PROFILE_SHA256,
    ConfigurationComponentV2,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


Cell = tuple[int, int]
Shape = tuple[int, int]

# No production object or identity has been authorized.  These constants are
# deliberately explicit so a caller cannot mistake the synthetic fixture path
# below for a deployment seam.
PRODUCTION_G3_V2_SNAPSHOT_SOURCE = None
PRODUCTION_G3_V2_COMPONENT_SOURCE = None
PRODUCTION_V5_RUNNER_EXECUTION_IDENTITY = None
PRODUCTION_V5_CHECKPOINT_FILE_SHA256 = None
PRODUCTION_V5_RAW_OUTCOME_SOURCE = None
PRODUCTION_V5_CAMERA_CALIBRATION_SHA256 = None
PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER = None


class TwoResolutionTargetEvidenceError(ValueError):
    """Base error for the two-resolution G5 evidence boundary."""


class TwoResolutionTargetEvidenceBindingError(TwoResolutionTargetEvidenceError):
    """An artifact does not bind the exact live source that purportedly issued it."""


class TwoResolutionTargetEvidenceRejectedError(TwoResolutionTargetEvidenceError):
    """Runner evidence is semantically inadmissible."""


class TwoResolutionTargetEvidenceReplayError(TwoResolutionTargetEvidenceError):
    """A single-use outcome, context, writer, or evidence record was replayed."""


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


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cell(value: object, name: str = "cell") -> Cell:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} coordinates must be integers")
    return int(value[0]), int(value[1])


def _cells(values: Iterable[object], name: str) -> frozenset[Cell]:
    return frozenset(_cell(value, name) for value in values)


def _cells_json(values: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(values)]


def _shape(value: object, name: str) -> Shape:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer sequence")
    return (
        _positive_int(value[0], f"{name}[0]"),
        _positive_int(value[1], f"{name}[1]"),
    )


def _unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _in_shape(cell: Cell, shape: Shape) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _assert_two_resolution_frames(
    physical_frame: MapFrameIdentity,
    configuration_frame: MapFrameIdentity,
    physical_shape: Shape,
    configuration_shape: Shape,
) -> None:
    if type(physical_frame) is not MapFrameIdentity or type(
        configuration_frame
    ) is not MapFrameIdentity:
        raise TypeError("two-resolution frames must have the canonical type")
    if physical_shape != (
        2 * configuration_shape[0],
        2 * configuration_shape[1],
    ):
        raise TwoResolutionTargetEvidenceBindingError(
            "physical shape must be exactly 2x the configuration shape per axis"
        )
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
        raise TwoResolutionTargetEvidenceBindingError(
            "physical/configuration frame identity or exact shared origin changed"
        )


def physical_cell_to_configuration_cell(
    *,
    physical_frame: MapFrameIdentity,
    configuration_frame: MapFrameIdentity,
    physical_shape: Sequence[int],
    configuration_shape: Sequence[int],
    physical_cell: Sequence[int],
) -> Cell:
    """Convert through metric frame APIs after validating the exact lattices."""

    physical = _shape(physical_shape, "physical_shape")
    configuration = _shape(configuration_shape, "configuration_shape")
    _assert_two_resolution_frames(
        physical_frame,
        configuration_frame,
        physical,
        configuration,
    )
    source = _cell(physical_cell, "physical_cell")
    if not _in_shape(source, physical):
        raise TwoResolutionTargetEvidenceRejectedError(
            "physical evidence cell is outside the live physical raster"
        )
    result = configuration_frame.world_to_cell(physical_frame.cell_center(source))
    if not _in_shape(result, configuration):
        raise TwoResolutionTargetEvidenceBindingError(
            "shared-origin conversion escaped the configuration raster"
        )
    expected = (source[0] // 2, source[1] // 2)
    if result != expected:
        raise TwoResolutionTargetEvidenceBindingError(
            "metric conversion disagrees with the exact 2:1 index mapping"
        )
    return result


def _physical_children(
    *,
    physical_frame: MapFrameIdentity,
    configuration_frame: MapFrameIdentity,
    physical_shape: Shape,
    configuration_shape: Shape,
    configuration_cell: Cell,
) -> frozenset[Cell]:
    if not _in_shape(configuration_cell, configuration_shape):
        raise TwoResolutionTargetEvidenceRejectedError(
            "configuration evidence cell is outside the live raster"
        )
    children = frozenset(
        (2 * configuration_cell[0] + dx, 2 * configuration_cell[1] + dy)
        for dx in (0, 1)
        for dy in (0, 1)
    )
    for child in children:
        if physical_cell_to_configuration_cell(
            physical_frame=physical_frame,
            configuration_frame=configuration_frame,
            physical_shape=physical_shape,
            configuration_shape=configuration_shape,
            physical_cell=child,
        ) != configuration_cell:
            raise TwoResolutionTargetEvidenceBindingError(
                "configuration cell does not own exactly four physical children"
            )
    return children


@dataclass(frozen=True)
class V5RunnerExecutionIdentityV1:
    entrypoint_wrapper_file_sha256: str
    captured_launcher_file_sha256: str
    captured_core_file_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "entrypoint_wrapper_file_sha256",
            "captured_launcher_file_sha256",
            "captured_core_file_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if len(
            {
                self.entrypoint_wrapper_file_sha256,
                self.captured_launcher_file_sha256,
                self.captured_core_file_sha256,
            }
        ) != 3:
            raise ValueError("V5 runner execution source identities must be distinct")
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_v5_runner_execution_identity_v1",
            "entrypoint_wrapper_file_sha256": self.entrypoint_wrapper_file_sha256,
            "captured_launcher_file_sha256": self.captured_launcher_file_sha256,
            "captured_core_file_sha256": self.captured_core_file_sha256,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetEvidenceBindingError(
                "V5 runner execution identity was mutated"
            )


@dataclass(frozen=True)
class PhysicalCellProbabilityV1:
    cell: Cell
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell, "physical evidence cell"))
        value = _unit(self.value, "physical evidence probability")
        if value <= 0.0:
            raise ValueError("physical evidence probability must be positive")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"cell": list(self.cell), "value": self.value}


@dataclass(frozen=True)
class ConfigurationCellProbabilityV1:
    cell: Cell
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell, "configuration evidence cell"))
        value = _unit(self.value, "configuration evidence probability")
        if value <= 0.0:
            raise ValueError("configuration evidence probability must be positive")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"cell": list(self.cell), "value": self.value}


def _probability_rows(
    values: Iterable[PhysicalCellProbabilityV1],
) -> tuple[PhysicalCellProbabilityV1, ...]:
    rows = tuple(values)
    if not rows or any(type(row) is not PhysicalCellProbabilityV1 for row in rows):
        raise TypeError("physical evidence must contain typed probability rows")
    rows = tuple(sorted(rows, key=lambda row: row.cell))
    if len({row.cell for row in rows}) != len(rows):
        raise ValueError("physical evidence contains duplicate cells")
    return rows


@dataclass(frozen=True)
class SyntheticV5TargetOutcomeV1:
    """A registered, production-ineligible facsimile of runner-owned output."""

    outcome_sequence: int
    outcome_kind: str
    target_id: str
    pose_timestamp_ns: int
    runner_execution_identity: V5RunnerExecutionIdentityV1
    checkpoint_file_sha256: str
    raw_outcome_file_sha256: str
    camera_calibration_sha256: str
    pose_provenance_sha256: str
    physical_map_frame_sha256: str
    physical_revision: int
    physical_content_sha256: str
    configuration_snapshot_sha256: str
    configuration_revision: int
    physical_shape: Shape
    free_physical_cells: frozenset[Cell]
    unknown_physical_cells: frozenset[Cell]
    target_physical_cells: frozenset[Cell]
    visible_physical_cells: frozenset[Cell]
    physical_probability: tuple[PhysicalCellProbabilityV1, ...]
    unlocalized_probability: float
    confidence: float
    _issuance_capability: object = field(repr=False, compare=False)
    raw_outcome_content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.outcome_sequence, "outcome_sequence")
        if self.outcome_kind not in {"positive", "negative"}:
            raise ValueError("outcome_kind must be positive or negative")
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        _nonnegative_int(self.pose_timestamp_ns, "pose_timestamp_ns")
        if type(self.runner_execution_identity) is not V5RunnerExecutionIdentityV1:
            raise TypeError("runner_execution_identity has the wrong type")
        self.runner_execution_identity.assert_integrity()
        for name in (
            "checkpoint_file_sha256",
            "raw_outcome_file_sha256",
            "camera_calibration_sha256",
            "pose_provenance_sha256",
            "physical_map_frame_sha256",
            "physical_content_sha256",
            "configuration_snapshot_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        physical_shape = _shape(self.physical_shape, "physical_shape")
        free = _cells(self.free_physical_cells, "free physical cell")
        unknown = _cells(self.unknown_physical_cells, "unknown physical cell")
        target = _cells(self.target_physical_cells, "target physical cell")
        visible = _cells(self.visible_physical_cells, "visible physical cell")
        rows = _probability_rows(self.physical_probability)
        for cell in free | unknown | target | visible | {row.cell for row in rows}:
            if not _in_shape(cell, physical_shape):
                raise ValueError("V5 physical cell is outside physical_shape")
        unlocalized = _unit(self.unlocalized_probability, "unlocalized_probability")
        confidence = _unit(self.confidence, "confidence")
        if confidence <= 0.0:
            raise ValueError("confidence must be positive")
        if self.outcome_kind == "positive":
            total = math.fsum(row.value for row in rows) + unlocalized
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("positive physical distribution must sum to one")
        elif unlocalized != 0.0:
            raise ValueError("negative outcome cannot carry unlocalized mass")
        if self._issuance_capability is None:
            raise TypeError("synthetic V5 outcome requires an issuance capability")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "free_physical_cells", free)
        object.__setattr__(self, "unknown_physical_cells", unknown)
        object.__setattr__(self, "target_physical_cells", target)
        object.__setattr__(self, "visible_physical_cells", visible)
        object.__setattr__(self, "physical_probability", rows)
        object.__setattr__(self, "unlocalized_probability", unlocalized)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "raw_outcome_content_sha256",
            _sha256(self.to_dict(False)),
        )

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_synthetic_v5_target_outcome_v1",
            "outcome_sequence": self.outcome_sequence,
            "outcome_kind": self.outcome_kind,
            "target_id": self.target_id,
            "pose_timestamp_ns": self.pose_timestamp_ns,
            "runner_execution_identity": self.runner_execution_identity.to_dict(),
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "pose_provenance_sha256": self.pose_provenance_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "physical_revision": self.physical_revision,
            "physical_content_sha256": self.physical_content_sha256,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "configuration_revision": self.configuration_revision,
            "physical_shape": list(self.physical_shape),
            "free_physical_cells": _cells_json(self.free_physical_cells),
            "unknown_physical_cells": _cells_json(self.unknown_physical_cells),
            "target_physical_cells": _cells_json(self.target_physical_cells),
            "visible_physical_cells": _cells_json(self.visible_physical_cells),
            "physical_probability": [row.to_dict() for row in self.physical_probability],
            "unlocalized_probability": self.unlocalized_probability,
            "confidence": self.confidence,
            "production_eligible": False,
        }
        if include_hash:
            result["raw_outcome_content_sha256"] = self.raw_outcome_content_sha256
        return result

    def assert_integrity(self) -> None:
        self.runner_execution_identity.assert_integrity()
        if self.raw_outcome_content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetEvidenceBindingError(
                "synthetic V5 outcome was mutated"
            )


class SyntheticV5TargetOutcomeIssuerV1:
    """Test-only exact-object issuer; it cannot authorize production output."""

    __slots__ = (
        "_capability",
        "_issued",
        "_consumed",
        "_sequence",
        "_synthetic_test_fixture",
    )

    def __init__(self, *, _synthetic_test_fixture: bool = False) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError("synthetic V5 outcome issuer is test-only")
        self._synthetic_test_fixture = True
        self._capability = object()
        self._issued: dict[int, SyntheticV5TargetOutcomeV1] = {}
        self._consumed: set[int] = set()
        self._sequence = 0

    def __copy__(self) -> "SyntheticV5TargetOutcomeIssuerV1":
        raise TypeError("synthetic V5 outcome issuer is non-copyable")

    def __deepcopy__(self, memo: object) -> "SyntheticV5TargetOutcomeIssuerV1":
        del memo
        raise TypeError("synthetic V5 outcome issuer is non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("synthetic V5 outcome issuer is non-serializable")

    def issue(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome_kind: str,
        target_id: str,
        pose_timestamp_ns: int,
        runner_execution_identity: V5RunnerExecutionIdentityV1,
        checkpoint_file_sha256: str,
        raw_outcome_file_sha256: str,
        camera_calibration_sha256: str,
        pose_provenance_sha256: str,
        free_physical_cells: Iterable[Cell],
        unknown_physical_cells: Iterable[Cell],
        target_physical_cells: Iterable[Cell],
        visible_physical_cells: Iterable[Cell],
        physical_probability: Iterable[PhysicalCellProbabilityV1],
        unlocalized_probability: float = 0.0,
        confidence: float = 1.0,
    ) -> SyntheticV5TargetOutcomeV1:
        if type(snapshot) is not TwoResolutionConfigurationSnapshotV2:
            raise TypeError("snapshot must be TwoResolutionConfigurationSnapshotV2")
        self._sequence += 1
        outcome = SyntheticV5TargetOutcomeV1(
            outcome_sequence=self._sequence,
            outcome_kind=outcome_kind,
            target_id=target_id,
            pose_timestamp_ns=pose_timestamp_ns,
            runner_execution_identity=runner_execution_identity,
            checkpoint_file_sha256=checkpoint_file_sha256,
            raw_outcome_file_sha256=raw_outcome_file_sha256,
            camera_calibration_sha256=camera_calibration_sha256,
            pose_provenance_sha256=pose_provenance_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            physical_revision=snapshot.physical_revision,
            physical_content_sha256=snapshot.physical_content_sha256,
            configuration_snapshot_sha256=snapshot.content_sha256,
            configuration_revision=snapshot.configuration_revision,
            physical_shape=snapshot.physical_shape,
            free_physical_cells=frozenset(free_physical_cells),
            unknown_physical_cells=frozenset(unknown_physical_cells),
            target_physical_cells=frozenset(target_physical_cells),
            visible_physical_cells=frozenset(visible_physical_cells),
            physical_probability=tuple(physical_probability),
            unlocalized_probability=unlocalized_probability,
            confidence=confidence,
            _issuance_capability=self._capability,
        )
        self._issued[id(outcome)] = outcome
        return outcome

    def assert_issued(
        self,
        outcome: SyntheticV5TargetOutcomeV1,
        *,
        consume: bool = False,
    ) -> None:
        if type(outcome) is not SyntheticV5TargetOutcomeV1:
            raise TypeError("outcome must be SyntheticV5TargetOutcomeV1")
        if self._issued.get(id(outcome)) is not outcome:
            raise TwoResolutionTargetEvidenceBindingError(
                "V5 outcome is not the exact live object issued by this source"
            )
        outcome.assert_integrity()
        if id(outcome) in self._consumed:
            raise TwoResolutionTargetEvidenceReplayError("V5 outcome was already consumed")
        if consume:
            self._consumed.add(id(outcome))


@dataclass(frozen=True)
class TwoResolutionTargetContextV1:
    context_sequence: int
    issuance_id_sha256: str
    pose_timestamp_ns: int
    physical_map_frame: MapFrameIdentity
    configuration_map_frame: MapFrameIdentity
    physical_shape: Shape
    configuration_shape: Shape
    physical_revision: int
    configuration_revision: int
    physical_content_sha256: str
    projection_source_sha256: str
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    runner_execution_identity_sha256: str
    checkpoint_file_sha256: str
    raw_outcome_file_sha256: str
    raw_outcome_content_sha256: str
    camera_calibration_sha256: str
    pose_provenance_sha256: str
    target_id: str
    outcome_kind: str
    candidate_domain: frozenset[Cell]
    excluded_target_configuration_cells: frozenset[Cell]
    source_visible_physical_cells: frozenset[Cell]
    source_physical_evidence_cells: frozenset[Cell]
    exact_sim_tainted: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.context_sequence, "context_sequence")
        _require_sha256(self.issuance_id_sha256, "issuance_id_sha256")
        _nonnegative_int(self.pose_timestamp_ns, "pose_timestamp_ns")
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(self.configuration_shape, "configuration_shape")
        _assert_two_resolution_frames(
            self.physical_map_frame,
            self.configuration_map_frame,
            physical_shape,
            configuration_shape,
        )
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        for name in (
            "physical_content_sha256",
            "projection_source_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "runner_execution_identity_sha256",
            "checkpoint_file_sha256",
            "raw_outcome_file_sha256",
            "raw_outcome_content_sha256",
            "camera_calibration_sha256",
            "pose_provenance_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.profile_sha256 != PROFILE_SHA256
            or self.free_support_sha256 != FREE_SUPPORT_SHA256
            or self.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
        ):
            raise TwoResolutionTargetEvidenceBindingError(
                "G3 V2 profile/support identity changed"
            )
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        if self.outcome_kind not in {"positive", "negative"}:
            raise ValueError("outcome_kind must be positive or negative")
        candidate = _cells(self.candidate_domain, "candidate cell")
        excluded = _cells(
            self.excluded_target_configuration_cells,
            "excluded target configuration cell",
        )
        visible = _cells(self.source_visible_physical_cells, "visible physical cell")
        evidence = _cells(self.source_physical_evidence_cells, "evidence physical cell")
        if not candidate or candidate & excluded:
            raise TwoResolutionTargetEvidenceRejectedError(
                "candidate domain must be nonempty and exclude target cells"
            )
        if any(not _in_shape(cell, configuration_shape) for cell in candidate | excluded):
            raise ValueError("configuration context cell is outside configuration_shape")
        if any(not _in_shape(cell, physical_shape) for cell in visible | evidence):
            raise ValueError("physical context cell is outside physical_shape")
        if not evidence or not evidence <= visible:
            raise TwoResolutionTargetEvidenceRejectedError(
                "physical evidence must be nonempty and runner-visible"
            )
        if type(self.exact_sim_tainted) is not bool:
            raise TypeError("exact_sim_tainted must be boolean")
        if self._issuance_capability is None:
            raise TypeError("two-resolution context requires an issuance capability")
        expected_issuance = _sha256(
            {
                "schema": "lewm_g5_two_resolution_context_issuance_identity_v1",
                "sequence": self.context_sequence,
                "snapshot_sha256": self.configuration_snapshot_sha256,
                "component_sha256": self.configuration_component_sha256,
                "physical_revision": self.physical_revision,
                "configuration_revision": self.configuration_revision,
                "raw_outcome_content_sha256": self.raw_outcome_content_sha256,
                "pose_timestamp_ns": self.pose_timestamp_ns,
            }
        )
        if self.issuance_id_sha256 != expected_issuance:
            raise TwoResolutionTargetEvidenceBindingError(
                "two-resolution context issuance identity changed"
            )
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        object.__setattr__(self, "candidate_domain", candidate)
        object.__setattr__(self, "excluded_target_configuration_cells", excluded)
        object.__setattr__(self, "source_visible_physical_cells", visible)
        object.__setattr__(self, "source_physical_evidence_cells", evidence)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def physical_cell_size_m(self) -> float:
        return PHYSICAL_CELL_SIZE_M

    @property
    def posterior_cell_size_m(self) -> float:
        return CONFIGURATION_CELL_SIZE_M

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_context_v1",
            "context_sequence": self.context_sequence,
            "issuance_id_sha256": self.issuance_id_sha256,
            "pose_timestamp_ns": self.pose_timestamp_ns,
            "physical_map_frame": self.physical_map_frame.to_dict(),
            "physical_map_frame_sha256": self.physical_map_frame.content_sha256,
            "configuration_map_frame": self.configuration_map_frame.to_dict(),
            "configuration_map_frame_sha256": self.configuration_map_frame.content_sha256,
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "physical_content_sha256": self.physical_content_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "configuration_component_sha256": self.configuration_component_sha256,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "runner_execution_identity_sha256": self.runner_execution_identity_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "raw_outcome_content_sha256": self.raw_outcome_content_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "pose_provenance_sha256": self.pose_provenance_sha256,
            "target_id": self.target_id,
            "outcome_kind": self.outcome_kind,
            "candidate_domain": _cells_json(self.candidate_domain),
            "excluded_target_configuration_cells": _cells_json(
                self.excluded_target_configuration_cells
            ),
            "source_visible_physical_cells": _cells_json(
                self.source_visible_physical_cells
            ),
            "source_physical_evidence_cells": _cells_json(
                self.source_physical_evidence_cells
            ),
            "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
            "posterior_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "exact_sim_tainted": self.exact_sim_tainted,
            "production_eligible": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetEvidenceBindingError(
                "two-resolution target context was mutated"
            )


@dataclass(frozen=True)
class TwoResolutionPositiveTargetEvidenceV1:
    context_sha256: str
    issuance_id_sha256: str
    target_id: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    runner_execution_identity_sha256: str
    checkpoint_file_sha256: str
    raw_outcome_file_sha256: str
    raw_outcome_content_sha256: str
    camera_calibration_sha256: str
    source_physical_distribution: tuple[PhysicalCellProbabilityV1, ...]
    localized_distribution: tuple[ConfigurationCellProbabilityV1, ...]
    unlocalized_probability: float
    confidence: float
    conversion_receipt_sha256: str
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "context_sha256",
            "issuance_id_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "runner_execution_identity_sha256",
            "checkpoint_file_sha256",
            "raw_outcome_file_sha256",
            "raw_outcome_content_sha256",
            "camera_calibration_sha256",
            "conversion_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        source = _probability_rows(self.source_physical_distribution)
        localized = tuple(self.localized_distribution)
        if not localized or any(
            type(row) is not ConfigurationCellProbabilityV1 for row in localized
        ):
            raise TypeError("localized_distribution must contain typed rows")
        localized = tuple(sorted(localized, key=lambda row: row.cell))
        if len({row.cell for row in localized}) != len(localized):
            raise ValueError("localized_distribution contains duplicate cells")
        unlocalized = _unit(self.unlocalized_probability, "unlocalized_probability")
        confidence = _unit(self.confidence, "confidence")
        if confidence <= 0.0 or not math.isclose(
            math.fsum(row.value for row in localized) + unlocalized,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("positive configuration distribution is invalid")
        if self._issuance_capability is None:
            raise TypeError("positive evidence requires an issuance capability")
        object.__setattr__(self, "source_physical_distribution", source)
        object.__setattr__(self, "localized_distribution", localized)
        object.__setattr__(self, "unlocalized_probability", unlocalized)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def posterior_cell_size_m(self) -> float:
        return CONFIGURATION_CELL_SIZE_M

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_positive_target_evidence_v1",
            "context_sha256": self.context_sha256,
            "issuance_id_sha256": self.issuance_id_sha256,
            "target_id": self.target_id,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "configuration_component_sha256": self.configuration_component_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "runner_execution_identity_sha256": self.runner_execution_identity_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "raw_outcome_content_sha256": self.raw_outcome_content_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "source_physical_distribution": [
                row.to_dict() for row in self.source_physical_distribution
            ],
            "localized_distribution": [row.to_dict() for row in self.localized_distribution],
            "unlocalized_probability": self.unlocalized_probability,
            "confidence": self.confidence,
            "conversion_receipt_sha256": self.conversion_receipt_sha256,
            "posterior_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "production_eligible": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetEvidenceBindingError(
                "positive evidence was mutated"
            )


@dataclass(frozen=True)
class TwoResolutionNegativeTargetEvidenceV1:
    context_sha256: str
    issuance_id_sha256: str
    target_id: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    runner_execution_identity_sha256: str
    checkpoint_file_sha256: str
    raw_outcome_file_sha256: str
    raw_outcome_content_sha256: str
    camera_calibration_sha256: str
    source_visible_physical_cells: frozenset[Cell]
    source_physical_detection_probability: tuple[PhysicalCellProbabilityV1, ...]
    visible_detection_probability: tuple[ConfigurationCellProbabilityV1, ...]
    confidence: float
    conversion_receipt_sha256: str
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "context_sha256",
            "issuance_id_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "runner_execution_identity_sha256",
            "checkpoint_file_sha256",
            "raw_outcome_file_sha256",
            "raw_outcome_content_sha256",
            "camera_calibration_sha256",
            "conversion_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        visible = _cells(self.source_visible_physical_cells, "visible physical cell")
        source = _probability_rows(self.source_physical_detection_probability)
        converted = tuple(self.visible_detection_probability)
        if not converted or any(
            type(row) is not ConfigurationCellProbabilityV1 for row in converted
        ):
            raise TypeError("visible_detection_probability must contain typed rows")
        converted = tuple(sorted(converted, key=lambda row: row.cell))
        if len({row.cell for row in converted}) != len(converted):
            raise ValueError("visible_detection_probability contains duplicate cells")
        confidence = _unit(self.confidence, "confidence")
        if confidence <= 0.0:
            raise ValueError("confidence must be positive")
        if self._issuance_capability is None:
            raise TypeError("negative evidence requires an issuance capability")
        object.__setattr__(self, "source_visible_physical_cells", visible)
        object.__setattr__(self, "source_physical_detection_probability", source)
        object.__setattr__(self, "visible_detection_probability", converted)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def posterior_cell_size_m(self) -> float:
        return CONFIGURATION_CELL_SIZE_M

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_negative_target_evidence_v1",
            "context_sha256": self.context_sha256,
            "issuance_id_sha256": self.issuance_id_sha256,
            "target_id": self.target_id,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "configuration_component_sha256": self.configuration_component_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "runner_execution_identity_sha256": self.runner_execution_identity_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "raw_outcome_content_sha256": self.raw_outcome_content_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "source_visible_physical_cells": _cells_json(
                self.source_visible_physical_cells
            ),
            "source_physical_detection_probability": [
                row.to_dict() for row in self.source_physical_detection_probability
            ],
            "visible_detection_probability": [
                row.to_dict() for row in self.visible_detection_probability
            ],
            "confidence": self.confidence,
            "conversion_receipt_sha256": self.conversion_receipt_sha256,
            "posterior_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "production_eligible": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetEvidenceBindingError(
                "negative evidence was mutated"
            )


@dataclass
class _ContextState:
    snapshot: TwoResolutionConfigurationSnapshotV2
    component: ConfigurationComponentV2
    outcome: SyntheticV5TargetOutcomeV1
    writer: "TwoResolutionTargetEvidenceWriterV1 | None" = None
    evidence: object | None = None


class TwoResolutionTargetEvidenceWriterV1:
    """The sole single-use writer leased for one exact context."""

    __slots__ = ("_issuer", "_context", "_used")

    def __init__(
        self,
        issuer: "TwoResolutionTargetEvidenceIssuerV1",
        context: TwoResolutionTargetContextV1,
    ) -> None:
        self._issuer = issuer
        self._context = context
        self._used = False

    def __copy__(self) -> "TwoResolutionTargetEvidenceWriterV1":
        raise TypeError("two-resolution evidence writer is non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionTargetEvidenceWriterV1":
        del memo
        raise TypeError("two-resolution evidence writer is non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("two-resolution evidence writer is non-serializable")

    @property
    def context(self) -> TwoResolutionTargetContextV1:
        return self._context

    def issue_positive(self) -> TwoResolutionPositiveTargetEvidenceV1:
        return self._issuer._issue_positive(self)

    def issue_negative(self) -> TwoResolutionNegativeTargetEvidenceV1:
        return self._issuer._issue_negative(self)


class TwoResolutionTargetEvidenceIssuerV1:
    """Exact-object G3/V5 binder and conservative cross-grid converter."""

    __slots__ = (
        "_projection",
        "_planner",
        "_outcome_source",
        "_runner_identity",
        "_checkpoint_file_sha256",
        "_camera_calibration_sha256",
        "_capability",
        "_sequence",
        "_contexts",
        "_evidence",
        "_consumed_evidence",
        "_synthetic_test_fixture",
    )

    def __init__(
        self,
        *,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
        outcome_source: SyntheticV5TargetOutcomeIssuerV1,
        runner_execution_identity: V5RunnerExecutionIdentityV1,
        checkpoint_file_sha256: str,
        camera_calibration_sha256: str,
        _synthetic_test_fixture: bool = False,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError(
                "no production two-resolution G5 evidence issuer is configured"
            )
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("projection must be the exact G3 V2 projection type")
        if type(planner) is not TwoResolutionConfigurationPlannerV2:
            raise TypeError("planner must be the exact G3 V2 planner type")
        if getattr(planner, "_projection", None) is not projection:
            raise TwoResolutionTargetEvidenceBindingError(
                "planner is not owned by the supplied live G3 V2 projection"
            )
        if type(outcome_source) is not SyntheticV5TargetOutcomeIssuerV1:
            raise TypeError("outcome_source must be the synthetic V5 issuer")
        if type(runner_execution_identity) is not V5RunnerExecutionIdentityV1:
            raise TypeError("runner_execution_identity has the wrong type")
        runner_execution_identity.assert_integrity()
        self._checkpoint_file_sha256 = _require_sha256(
            checkpoint_file_sha256,
            "checkpoint_file_sha256",
        )
        self._camera_calibration_sha256 = _require_sha256(
            camera_calibration_sha256,
            "camera_calibration_sha256",
        )
        self._projection = projection
        self._planner = planner
        self._outcome_source = outcome_source
        self._runner_identity = runner_execution_identity
        self._capability = object()
        self._sequence = 0
        self._contexts: dict[int, tuple[TwoResolutionTargetContextV1, _ContextState]] = {}
        self._evidence: dict[int, object] = {}
        self._consumed_evidence: set[int] = set()
        self._synthetic_test_fixture = True

    def __copy__(self) -> "TwoResolutionTargetEvidenceIssuerV1":
        raise TypeError("two-resolution evidence issuer is non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionTargetEvidenceIssuerV1":
        del memo
        raise TypeError("two-resolution evidence issuer is non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("two-resolution evidence issuer is non-serializable")

    @property
    def production_eligible(self) -> bool:
        return False

    def _map_physical(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        physical_cell: Cell,
    ) -> Cell:
        return physical_cell_to_configuration_cell(
            physical_frame=snapshot.physical_map_frame,
            configuration_frame=snapshot.configuration_map_frame,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            physical_cell=physical_cell,
        )

    def _assert_outcome_bindings(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticV5TargetOutcomeV1,
    ) -> None:
        if (
            outcome.runner_execution_identity.content_sha256
            != self._runner_identity.content_sha256
            or outcome.checkpoint_file_sha256 != self._checkpoint_file_sha256
            or outcome.camera_calibration_sha256 != self._camera_calibration_sha256
        ):
            raise TwoResolutionTargetEvidenceBindingError(
                "V5 runner execution/checkpoint/calibration identity changed"
            )
        if (
            outcome.physical_map_frame_sha256 != snapshot.physical_map_frame_sha256
            or outcome.physical_revision != snapshot.physical_revision
            or outcome.physical_content_sha256 != snapshot.physical_content_sha256
            or outcome.configuration_snapshot_sha256 != snapshot.content_sha256
            or outcome.configuration_revision != snapshot.configuration_revision
            or outcome.physical_shape != snapshot.physical_shape
        ):
            raise TwoResolutionTargetEvidenceBindingError(
                "V5 outcome does not bind the exact G3 V2 snapshot revision/frame"
            )

    def issue_context(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        outcome: SyntheticV5TargetOutcomeV1,
    ) -> TwoResolutionTargetContextV1:
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_component(snapshot, component)
        self._outcome_source.assert_issued(outcome)
        self._runner_identity.assert_integrity()
        _assert_two_resolution_frames(
            snapshot.physical_map_frame,
            snapshot.configuration_map_frame,
            snapshot.physical_shape,
            snapshot.configuration_shape,
        )
        if (
            snapshot.profile_sha256 != PROFILE_SHA256
            or snapshot.free_support_sha256 != FREE_SUPPORT_SHA256
            or snapshot.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
            or component.snapshot_sha256 != snapshot.content_sha256
            or component.physical_revision != snapshot.physical_revision
            or component.configuration_revision != snapshot.configuration_revision
            or component.physical_map_frame_sha256
            != snapshot.physical_map_frame_sha256
            or component.configuration_map_frame_sha256
            != snapshot.configuration_map_frame_sha256
            or component.free_support_sha256 != snapshot.free_support_sha256
            or component.occupied_support_sha256 != snapshot.occupied_support_sha256
        ):
            raise TwoResolutionTargetEvidenceBindingError(
                "G3 V2 snapshot/component revision or support binding changed"
            )
        self._assert_outcome_bindings(snapshot, outcome)

        evidence_physical = frozenset(row.cell for row in outcome.physical_probability)
        if not evidence_physical <= outcome.visible_physical_cells:
            raise TwoResolutionTargetEvidenceRejectedError(
                "V5 evidence includes a cell outside runner-owned visibility"
            )
        if (
            not outcome.visible_physical_cells <= outcome.free_physical_cells
            or outcome.visible_physical_cells & outcome.unknown_physical_cells
            or outcome.visible_physical_cells & outcome.target_physical_cells
            or evidence_physical & outcome.unknown_physical_cells
            or evidence_physical & outcome.target_physical_cells
        ):
            raise TwoResolutionTargetEvidenceRejectedError(
                "UNKNOWN or target physical cells cannot issue target evidence"
            )

        excluded_target = frozenset(
            self._map_physical(snapshot, cell)
            for cell in outcome.target_physical_cells
        )
        candidate = frozenset(component.cells - excluded_target)
        if not candidate:
            raise TwoResolutionTargetEvidenceRejectedError(
                "target exclusion emptied the configuration candidate domain"
            )
        for physical_cell in outcome.visible_physical_cells | evidence_physical:
            configuration_cell = self._map_physical(snapshot, physical_cell)
            if (
                configuration_cell not in candidate
                or snapshot.state(configuration_cell) is not PhysicalLabel.FREE
            ):
                raise TwoResolutionTargetEvidenceRejectedError(
                    "physical evidence maps to UNKNOWN, occupied, disconnected, or target cell"
                )

        self._sequence += 1
        issuance_id = _sha256(
            {
                "schema": "lewm_g5_two_resolution_context_issuance_identity_v1",
                "sequence": self._sequence,
                "snapshot_sha256": snapshot.content_sha256,
                "component_sha256": component.content_sha256,
                "physical_revision": snapshot.physical_revision,
                "configuration_revision": snapshot.configuration_revision,
                "raw_outcome_content_sha256": outcome.raw_outcome_content_sha256,
                "pose_timestamp_ns": outcome.pose_timestamp_ns,
            }
        )
        context = TwoResolutionTargetContextV1(
            context_sequence=self._sequence,
            issuance_id_sha256=issuance_id,
            pose_timestamp_ns=outcome.pose_timestamp_ns,
            physical_map_frame=snapshot.physical_map_frame,
            configuration_map_frame=snapshot.configuration_map_frame,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_content_sha256=snapshot.physical_content_sha256,
            projection_source_sha256=snapshot.projection_source_sha256,
            configuration_snapshot_sha256=snapshot.content_sha256,
            configuration_component_sha256=component.content_sha256,
            profile_sha256=snapshot.profile_sha256,
            free_support_sha256=snapshot.free_support_sha256,
            occupied_support_sha256=snapshot.occupied_support_sha256,
            runner_execution_identity_sha256=(
                outcome.runner_execution_identity.content_sha256
            ),
            checkpoint_file_sha256=outcome.checkpoint_file_sha256,
            raw_outcome_file_sha256=outcome.raw_outcome_file_sha256,
            raw_outcome_content_sha256=outcome.raw_outcome_content_sha256,
            camera_calibration_sha256=outcome.camera_calibration_sha256,
            pose_provenance_sha256=outcome.pose_provenance_sha256,
            target_id=outcome.target_id,
            outcome_kind=outcome.outcome_kind,
            candidate_domain=candidate,
            excluded_target_configuration_cells=excluded_target,
            source_visible_physical_cells=outcome.visible_physical_cells,
            source_physical_evidence_cells=evidence_physical,
            exact_sim_tainted=snapshot.exact_sim_tainted,
            _issuance_capability=self._capability,
        )
        self._contexts[id(context)] = (
            context,
            _ContextState(snapshot=snapshot, component=component, outcome=outcome),
        )
        self._outcome_source.assert_issued(outcome, consume=True)
        return context

    def _state_for_context(
        self,
        context: TwoResolutionTargetContextV1,
    ) -> _ContextState:
        if type(context) is not TwoResolutionTargetContextV1:
            raise TypeError("context must be TwoResolutionTargetContextV1")
        row = self._contexts.get(id(context))
        if row is None or row[0] is not context:
            raise TwoResolutionTargetEvidenceBindingError(
                "context is not the exact live object issued by this issuer"
            )
        context.assert_integrity()
        state = row[1]
        self._projection.assert_current_snapshot(state.snapshot)
        self._planner.validate_component(state.snapshot, state.component)
        self._assert_outcome_bindings(state.snapshot, state.outcome)
        if (
            context.configuration_snapshot_sha256 != state.snapshot.content_sha256
            or context.configuration_component_sha256 != state.component.content_sha256
            or context.raw_outcome_content_sha256
            != state.outcome.raw_outcome_content_sha256
        ):
            raise TwoResolutionTargetEvidenceBindingError(
                "context source identity changed after issuance"
            )
        return state

    def open_writer(
        self,
        context: TwoResolutionTargetContextV1,
    ) -> TwoResolutionTargetEvidenceWriterV1:
        state = self._state_for_context(context)
        if state.writer is not None:
            raise TwoResolutionTargetEvidenceReplayError(
                "the context already has its single writer"
            )
        writer = TwoResolutionTargetEvidenceWriterV1(self, context)
        state.writer = writer
        return writer

    def _state_for_writer(
        self,
        writer: TwoResolutionTargetEvidenceWriterV1,
    ) -> tuple[TwoResolutionTargetContextV1, _ContextState]:
        if type(writer) is not TwoResolutionTargetEvidenceWriterV1:
            raise TypeError("writer must be TwoResolutionTargetEvidenceWriterV1")
        if writer._issuer is not self:
            raise TwoResolutionTargetEvidenceBindingError(
                "writer belongs to a different evidence issuer"
            )
        context = writer._context
        state = self._state_for_context(context)
        if state.writer is not writer:
            raise TwoResolutionTargetEvidenceBindingError(
                "writer is not the exact lease held by the context"
            )
        if writer._used or state.evidence is not None:
            raise TwoResolutionTargetEvidenceReplayError(
                "the context writer is single-use"
            )
        return context, state

    @staticmethod
    def _common_evidence_fields(
        context: TwoResolutionTargetContextV1,
        issuance_id_sha256: str,
    ) -> dict[str, object]:
        return {
            "context_sha256": context.content_sha256,
            "issuance_id_sha256": issuance_id_sha256,
            "target_id": context.target_id,
            "physical_map_frame_sha256": context.physical_map_frame.content_sha256,
            "configuration_map_frame_sha256": (
                context.configuration_map_frame.content_sha256
            ),
            "physical_revision": context.physical_revision,
            "configuration_revision": context.configuration_revision,
            "configuration_snapshot_sha256": context.configuration_snapshot_sha256,
            "configuration_component_sha256": context.configuration_component_sha256,
            "free_support_sha256": context.free_support_sha256,
            "occupied_support_sha256": context.occupied_support_sha256,
            "runner_execution_identity_sha256": (
                context.runner_execution_identity_sha256
            ),
            "checkpoint_file_sha256": context.checkpoint_file_sha256,
            "raw_outcome_file_sha256": context.raw_outcome_file_sha256,
            "raw_outcome_content_sha256": context.raw_outcome_content_sha256,
            "camera_calibration_sha256": context.camera_calibration_sha256,
        }

    def _register_evidence(
        self,
        writer: TwoResolutionTargetEvidenceWriterV1,
        state: _ContextState,
        evidence: object,
    ) -> None:
        writer._used = True
        state.evidence = evidence
        self._evidence[id(evidence)] = evidence

    def _issue_positive(
        self,
        writer: TwoResolutionTargetEvidenceWriterV1,
    ) -> TwoResolutionPositiveTargetEvidenceV1:
        context, state = self._state_for_writer(writer)
        outcome = state.outcome
        if outcome.outcome_kind != "positive":
            raise TwoResolutionTargetEvidenceRejectedError(
                "negative V5 outcome cannot issue positive evidence"
            )
        aggregate: dict[Cell, float] = {}
        mapping: list[dict[str, object]] = []
        for row in outcome.physical_probability:
            configuration_cell = self._map_physical(state.snapshot, row.cell)
            if configuration_cell not in context.candidate_domain:
                raise TwoResolutionTargetEvidenceRejectedError(
                    "positive evidence maps outside the configuration candidate domain"
                )
            aggregate[configuration_cell] = aggregate.get(configuration_cell, 0.0) + row.value
            mapping.append(
                {
                    "physical_cell": list(row.cell),
                    "configuration_cell": list(configuration_cell),
                    "value": row.value,
                }
            )
        localized = tuple(
            ConfigurationCellProbabilityV1(cell=cell, value=value)
            for cell, value in sorted(aggregate.items())
        )
        receipt = _sha256(
            {
                "schema": "lewm_g5_two_resolution_positive_conversion_receipt_v1",
                "physical_map_frame_sha256": context.physical_map_frame.content_sha256,
                "configuration_map_frame_sha256": (
                    context.configuration_map_frame.content_sha256
                ),
                "physical_shape": list(context.physical_shape),
                "configuration_shape": list(context.configuration_shape),
                "shared_origin_xy_m": list(context.physical_map_frame.origin_xy_m),
                "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
                "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
                "mapping": mapping,
                "aggregation": "sum",
            }
        )
        issuance = _sha256(
            {
                "schema": "lewm_g5_two_resolution_positive_issuance_identity_v1",
                "context_sha256": context.content_sha256,
                "raw_outcome_content_sha256": outcome.raw_outcome_content_sha256,
                "conversion_receipt_sha256": receipt,
            }
        )
        evidence = TwoResolutionPositiveTargetEvidenceV1(
            **self._common_evidence_fields(context, issuance),
            source_physical_distribution=outcome.physical_probability,
            localized_distribution=localized,
            unlocalized_probability=outcome.unlocalized_probability,
            confidence=outcome.confidence,
            conversion_receipt_sha256=receipt,
            _issuance_capability=self._capability,
        )
        self._register_evidence(writer, state, evidence)
        return evidence

    def _issue_negative(
        self,
        writer: TwoResolutionTargetEvidenceWriterV1,
    ) -> TwoResolutionNegativeTargetEvidenceV1:
        context, state = self._state_for_writer(writer)
        outcome = state.outcome
        if outcome.outcome_kind != "negative":
            raise TwoResolutionTargetEvidenceRejectedError(
                "positive V5 outcome cannot issue negative evidence"
            )
        by_cell = {row.cell: row.value for row in outcome.physical_probability}
        parent_cells = frozenset(
            self._map_physical(state.snapshot, cell) for cell in by_cell
        )
        converted: list[ConfigurationCellProbabilityV1] = []
        mapping: list[dict[str, object]] = []
        for configuration_cell in sorted(parent_cells):
            if configuration_cell not in context.candidate_domain:
                raise TwoResolutionTargetEvidenceRejectedError(
                    "negative evidence maps outside the configuration candidate domain"
                )
            children = _physical_children(
                physical_frame=context.physical_map_frame,
                configuration_frame=context.configuration_map_frame,
                physical_shape=context.physical_shape,
                configuration_shape=context.configuration_shape,
                configuration_cell=configuration_cell,
            )
            if (
                not children <= outcome.visible_physical_cells
                or not children <= outcome.free_physical_cells
                or not children <= set(by_cell)
                or children & outcome.unknown_physical_cells
                or children & outcome.target_physical_cells
            ):
                raise TwoResolutionTargetEvidenceRejectedError(
                    "negative evidence requires all four visible, FREE physical children"
                )
            value = min(by_cell[child] for child in children)
            converted.append(
                ConfigurationCellProbabilityV1(
                    cell=configuration_cell,
                    value=value,
                )
            )
            mapping.append(
                {
                    "configuration_cell": list(configuration_cell),
                    "physical_children": _cells_json(children),
                    "physical_values": [by_cell[child] for child in sorted(children)],
                    "configuration_value": value,
                }
            )
        receipt = _sha256(
            {
                "schema": "lewm_g5_two_resolution_negative_conversion_receipt_v1",
                "physical_map_frame_sha256": context.physical_map_frame.content_sha256,
                "configuration_map_frame_sha256": (
                    context.configuration_map_frame.content_sha256
                ),
                "physical_shape": list(context.physical_shape),
                "configuration_shape": list(context.configuration_shape),
                "shared_origin_xy_m": list(context.physical_map_frame.origin_xy_m),
                "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
                "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
                "mapping": mapping,
                "aggregation": "minimum_across_exact_four_children",
            }
        )
        issuance = _sha256(
            {
                "schema": "lewm_g5_two_resolution_negative_issuance_identity_v1",
                "context_sha256": context.content_sha256,
                "raw_outcome_content_sha256": outcome.raw_outcome_content_sha256,
                "conversion_receipt_sha256": receipt,
            }
        )
        evidence = TwoResolutionNegativeTargetEvidenceV1(
            **self._common_evidence_fields(context, issuance),
            source_visible_physical_cells=outcome.visible_physical_cells,
            source_physical_detection_probability=outcome.physical_probability,
            visible_detection_probability=tuple(converted),
            confidence=outcome.confidence,
            conversion_receipt_sha256=receipt,
            _issuance_capability=self._capability,
        )
        self._register_evidence(writer, state, evidence)
        return evidence

    def consume_evidence(
        self,
        evidence: TwoResolutionPositiveTargetEvidenceV1
        | TwoResolutionNegativeTargetEvidenceV1,
    ) -> None:
        if type(evidence) not in {
            TwoResolutionPositiveTargetEvidenceV1,
            TwoResolutionNegativeTargetEvidenceV1,
        }:
            raise TypeError("evidence has the wrong two-resolution type")
        if self._evidence.get(id(evidence)) is not evidence:
            raise TwoResolutionTargetEvidenceBindingError(
                "evidence is not the exact live object issued by this issuer"
            )
        evidence.assert_integrity()
        if id(evidence) in self._consumed_evidence:
            raise TwoResolutionTargetEvidenceReplayError(
                "two-resolution evidence was already consumed"
            )
        self._consumed_evidence.add(id(evidence))


def require_production_two_resolution_target_evidence_issuer() -> object:
    """Fail closed until independently reviewed production identities exist."""

    if (
        PRODUCTION_G3_V2_SNAPSHOT_SOURCE is None
        or PRODUCTION_G3_V2_COMPONENT_SOURCE is None
        or PRODUCTION_V5_RUNNER_EXECUTION_IDENTITY is None
        or PRODUCTION_V5_CHECKPOINT_FILE_SHA256 is None
        or PRODUCTION_V5_RAW_OUTCOME_SOURCE is None
        or PRODUCTION_V5_CAMERA_CALIBRATION_SHA256 is None
        or PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER is None
    ):
        raise PermissionError(
            "production G3/V5/G5 two-resolution evidence identities are unset"
        )
    return PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER


__all__ = [
    "ConfigurationCellProbabilityV1",
    "PhysicalCellProbabilityV1",
    "SyntheticV5TargetOutcomeIssuerV1",
    "SyntheticV5TargetOutcomeV1",
    "TwoResolutionNegativeTargetEvidenceV1",
    "TwoResolutionPositiveTargetEvidenceV1",
    "TwoResolutionTargetContextV1",
    "TwoResolutionTargetEvidenceBindingError",
    "TwoResolutionTargetEvidenceIssuerV1",
    "TwoResolutionTargetEvidenceRejectedError",
    "TwoResolutionTargetEvidenceReplayError",
    "TwoResolutionTargetEvidenceWriterV1",
    "V5RunnerExecutionIdentityV1",
    "physical_cell_to_configuration_cell",
    "require_production_two_resolution_target_evidence_issuer",
]
