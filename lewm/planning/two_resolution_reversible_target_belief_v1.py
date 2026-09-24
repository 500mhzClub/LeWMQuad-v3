"""Reversible target posterior on the G3 V2 configuration lattice.

This additive development boundary consumes only exact, single-use evidence
issued by :mod:`two_resolution_target_evidence_v1`.  Target mass lives on the
0.10 m configuration lattice; the source evidence remains bound to the 0.05 m
physical lattice through its context and conversion receipt.

The module deliberately grants no production or hardware authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    SnapshotBindingError,
    StaleSnapshotError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_SHA256,
    PROFILE_SHA256,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    ConfigurationCellProbabilityV1,
    TwoResolutionNegativeTargetEvidenceV1,
    TwoResolutionPositiveTargetEvidenceV1,
    TwoResolutionTargetContextV1,
    TwoResolutionTargetEvidenceBindingError,
    TwoResolutionTargetEvidenceIssuerV1,
    TwoResolutionTargetEvidenceReplayError,
)


Cell = tuple[int, int]

PRODUCTION_TWO_RESOLUTION_TARGET_MEMORY = None


class TwoResolutionTargetMemoryError(ValueError):
    """Base error for the two-resolution posterior."""


class TwoResolutionTargetMemoryBindingError(
    TwoResolutionTargetMemoryError, SnapshotBindingError
):
    """Evidence or posterior state changed its exact source binding."""


class TwoResolutionTargetMemoryReplayError(TwoResolutionTargetMemoryError):
    """Evidence, context, or snapshot was reused."""


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
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _cell(value: Sequence[int], name: str) -> Cell:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must contain two integer indices")
    return int(value[0]), int(value[1])


def _cells_json(values: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(values)]


@dataclass(frozen=True)
class TwoResolutionTargetMemoryConfigV1:
    target_ids: tuple[str, ...] = ("blue", "green", "red", "yellow")
    maximum_positive_transfer: float = 0.50
    negative_mass_floor_multiplier: float = 1e-4
    posterior_mass_floor: float = 1e-15
    component_mass_floor: float = 1e-12
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.target_ids) is not tuple
            or not self.target_ids
            or tuple(sorted(self.target_ids)) != self.target_ids
            or len(set(self.target_ids)) != len(self.target_ids)
            or any(type(value) is not str or not value for value in self.target_ids)
        ):
            raise ValueError("target_ids must be nonempty, sorted, and unique")
        transfer = _unit(self.maximum_positive_transfer, "maximum_positive_transfer")
        negative_floor = _unit(
            self.negative_mass_floor_multiplier,
            "negative_mass_floor_multiplier",
        )
        posterior_floor = _unit(self.posterior_mass_floor, "posterior_mass_floor")
        component_floor = _unit(self.component_mass_floor, "component_mass_floor")
        if not 0.0 < transfer < 1.0:
            raise ValueError("maximum_positive_transfer must lie in (0, 1)")
        if not 0.0 < negative_floor < 1.0:
            raise ValueError("negative_mass_floor_multiplier must lie in (0, 1)")
        if not 0.0 < posterior_floor < component_floor < 1.0:
            raise ValueError("posterior/component floors are inconsistent")
        object.__setattr__(self, "maximum_positive_transfer", transfer)
        object.__setattr__(self, "negative_mass_floor_multiplier", negative_floor)
        object.__setattr__(self, "posterior_mass_floor", posterior_floor)
        object.__setattr__(self, "component_mass_floor", component_floor)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_memory_config_v1",
            "target_ids": list(self.target_ids),
            "physical_cell_size_m": 0.05,
            "posterior_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "maximum_positive_transfer": self.maximum_positive_transfer,
            "negative_mass_floor_multiplier": self.negative_mass_floor_multiplier,
            "posterior_mass_floor": self.posterior_mass_floor,
            "component_mass_floor": self.component_mass_floor,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetMemoryBindingError("memory config was mutated")


@dataclass(frozen=True)
class TwoResolutionTargetCellMassV1:
    cell: Cell
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell, "posterior cell"))
        value = _unit(self.value, "posterior cell mass")
        if value <= 0.0:
            raise ValueError("posterior cell mass must be positive")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"cell": list(self.cell), "value": self.value}


@dataclass(frozen=True)
class TwoResolutionTargetPosteriorSnapshotV1:
    target_id: str
    target_memory_instance_sha256: str
    target_memory_config_sha256: str
    target_memory_revision: int
    context_sequence: int
    context_sha256: str
    physical_map_frame: MapFrameIdentity
    configuration_map_frame: MapFrameIdentity
    physical_shape: tuple[int, int]
    configuration_shape: tuple[int, int]
    physical_revision: int
    configuration_revision: int
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    projection_source_sha256: str
    runner_execution_identity_sha256: str
    checkpoint_file_sha256: str
    camera_calibration_sha256: str
    candidate_domain: frozenset[Cell]
    excluded_target_configuration_cells: frozenset[Cell]
    cell_mass: tuple[TwoResolutionTargetCellMassV1, ...]
    unlocalized_mass: float
    positive_evidence_count: int
    negative_evidence_count: int
    evidence_chain_sha256: str
    exact_sim_tainted: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        for name in (
            "target_memory_instance_sha256",
            "target_memory_config_sha256",
            "context_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "projection_source_sha256",
            "runner_execution_identity_sha256",
            "checkpoint_file_sha256",
            "camera_calibration_sha256",
            "evidence_chain_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.physical_map_frame) is not MapFrameIdentity or type(
            self.configuration_map_frame
        ) is not MapFrameIdentity:
            raise TypeError("posterior snapshot requires canonical map frames")
        if (
            not math.isclose(
                self.physical_map_frame.cell_size_m, 0.05, rel_tol=0.0, abs_tol=1e-12
            )
            or not math.isclose(
                self.configuration_map_frame.cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.physical_map_frame.origin_xy_m
            != self.configuration_map_frame.origin_xy_m
        ):
            raise TwoResolutionTargetMemoryBindingError("posterior lattice changed")
        for name in (
            "target_memory_revision",
            "context_sequence",
            "physical_revision",
            "configuration_revision",
            "positive_evidence_count",
            "negative_evidence_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.context_sequence == 0 or self.configuration_revision == 0:
            raise ValueError("posterior snapshot requires a committed context")
        physical_shape = _cell(self.physical_shape, "physical_shape")
        configuration_shape = _cell(self.configuration_shape, "configuration_shape")
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise TwoResolutionTargetMemoryBindingError("posterior shape ratio changed")
        candidate = frozenset(
            _cell(cell, "candidate cell") for cell in self.candidate_domain
        )
        excluded = frozenset(
            _cell(cell, "excluded target cell")
            for cell in self.excluded_target_configuration_cells
        )
        if not candidate or candidate & excluded:
            raise TwoResolutionTargetMemoryBindingError("posterior domain is invalid")
        rows = tuple(self.cell_mass)
        if any(type(row) is not TwoResolutionTargetCellMassV1 for row in rows):
            raise TypeError("cell_mass must contain typed rows")
        rows = tuple(sorted(rows, key=lambda row: row.cell))
        if len({row.cell for row in rows}) != len(rows):
            raise ValueError("posterior contains duplicate cells")
        if any(row.cell not in candidate for row in rows):
            raise TwoResolutionTargetMemoryBindingError(
                "posterior mass escaped the current candidate domain"
            )
        unlocalized = _unit(self.unlocalized_mass, "unlocalized_mass")
        if not math.isclose(
            math.fsum(row.value for row in rows) + unlocalized,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("posterior mass must sum to one")
        if self.profile_sha256 != PROFILE_SHA256 or (
            self.free_support_sha256 != FREE_SUPPORT_SHA256
            or self.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
        ):
            raise TwoResolutionTargetMemoryBindingError("G3 V2 support changed")
        if type(self.exact_sim_tainted) is not bool:
            raise TypeError("exact_sim_tainted must be boolean")
        if self._issuance_capability is None:
            raise TypeError("posterior snapshot requires an issuance capability")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        object.__setattr__(self, "candidate_domain", candidate)
        object.__setattr__(self, "excluded_target_configuration_cells", excluded)
        object.__setattr__(self, "cell_mass", rows)
        object.__setattr__(self, "unlocalized_mass", unlocalized)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_posterior_snapshot_v1",
            "target_id": self.target_id,
            "target_memory_instance_sha256": self.target_memory_instance_sha256,
            "target_memory_config_sha256": self.target_memory_config_sha256,
            "target_memory_revision": self.target_memory_revision,
            "context_sequence": self.context_sequence,
            "context_sha256": self.context_sha256,
            "physical_map_frame": self.physical_map_frame.to_dict(),
            "configuration_map_frame": self.configuration_map_frame.to_dict(),
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "configuration_component_sha256": self.configuration_component_sha256,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "runner_execution_identity_sha256": self.runner_execution_identity_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "candidate_domain": _cells_json(self.candidate_domain),
            "excluded_target_configuration_cells": _cells_json(
                self.excluded_target_configuration_cells
            ),
            "cell_mass": [row.to_dict() for row in self.cell_mass],
            "unlocalized_mass": self.unlocalized_mass,
            "positive_evidence_count": self.positive_evidence_count,
            "negative_evidence_count": self.negative_evidence_count,
            "evidence_chain_sha256": self.evidence_chain_sha256,
            "exact_sim_tainted": self.exact_sim_tainted,
            "posterior_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetMemoryBindingError("posterior snapshot was mutated")

    def __copy__(self) -> "TwoResolutionTargetPosteriorSnapshotV1":
        raise TypeError("posterior snapshots are non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionTargetPosteriorSnapshotV1":
        del memo
        raise TypeError("posterior snapshots are non-copyable")


@dataclass(frozen=True)
class TwoResolutionTargetHypothesisV1:
    target_id: str
    cells: frozenset[Cell]
    mass: float
    peak_cell: Cell
    peak_mass: float
    mean_world_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        cells = frozenset(_cell(cell, "hypothesis cell") for cell in self.cells)
        if not cells or self.peak_cell not in cells:
            raise ValueError("hypothesis cells/peak are inconsistent")
        mass = _unit(self.mass, "hypothesis mass")
        peak_mass = _unit(self.peak_mass, "hypothesis peak mass")
        if mass <= 0.0 or peak_mass <= 0.0 or peak_mass > mass:
            raise ValueError("hypothesis mass is invalid")
        mean = tuple(float(value) for value in self.mean_world_xy_m)
        if len(mean) != 2 or any(not math.isfinite(value) for value in mean):
            raise ValueError("hypothesis mean must be finite XY")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "peak_cell", _cell(self.peak_cell, "peak_cell"))
        object.__setattr__(self, "peak_mass", peak_mass)
        object.__setattr__(self, "mean_world_xy_m", mean)


class TwoResolutionReversibleTargetBeliefMemoryV1:
    """One-writer, exact-evidence posterior for all task target identities."""

    def __init__(
        self,
        *,
        evidence_issuer: TwoResolutionTargetEvidenceIssuerV1,
        config: TwoResolutionTargetMemoryConfigV1 | None = None,
        _synthetic_development_fixture: bool = False,
    ) -> None:
        if _synthetic_development_fixture is not True:
            raise PermissionError(
                "no production two-resolution target-memory authority is configured"
            )
        if type(evidence_issuer) is not TwoResolutionTargetEvidenceIssuerV1:
            raise TypeError("evidence_issuer has the wrong exact type")
        supplied = config or TwoResolutionTargetMemoryConfigV1()
        if type(supplied) is not TwoResolutionTargetMemoryConfigV1:
            raise TypeError("config has the wrong exact type")
        supplied.assert_integrity()
        self._issuer = evidence_issuer
        self._config = supplied
        self._capability = object()
        self._instance_sha256 = _sha256(
            {
                "schema": "lewm_g5_two_resolution_target_memory_instance_v1",
                "config_sha256": supplied.content_sha256,
                "issuer_object_identity": id(evidence_issuer),
                "memory_object_identity": id(self),
            }
        )
        self._revision = 0
        self._last_context_sequence = 0
        self._last_pose_timestamp_ns = -1
        self._immutable_binding: dict[str, object] | None = None
        self._mass: dict[str, dict[Cell, float]] = {
            target: {} for target in supplied.target_ids
        }
        self._unlocalized: dict[str, float] = {
            target: 1.0 for target in supplied.target_ids
        }
        self._contexts: dict[str, TwoResolutionTargetContextV1] = {}
        self._positive_count = {target: 0 for target in supplied.target_ids}
        self._negative_count = {target: 0 for target in supplied.target_ids}
        self._chains = {
            target: _sha256(
                {
                    "schema": "lewm_g5_two_resolution_target_chain_seed_v1",
                    "target_id": target,
                    "memory_instance_sha256": self._instance_sha256,
                    "config_sha256": supplied.content_sha256,
                }
            )
            for target in supplied.target_ids
        }
        self._seen_context_ids: set[str] = set()
        self._seen_evidence_hashes: set[str] = set()
        self._seen_raw_outcomes: set[str] = set()
        self._issued_snapshots: dict[int, TwoResolutionTargetPosteriorSnapshotV1] = {}

    def __copy__(self) -> "TwoResolutionReversibleTargetBeliefMemoryV1":
        raise TypeError("two-resolution target memory is non-copyable")

    def __deepcopy__(
        self, memo: object
    ) -> "TwoResolutionReversibleTargetBeliefMemoryV1":
        del memo
        raise TypeError("two-resolution target memory is non-copyable")

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @staticmethod
    def _immutable_context_binding(context: TwoResolutionTargetContextV1) -> dict[str, object]:
        return {
            "physical_map_frame": context.physical_map_frame,
            "configuration_map_frame": context.configuration_map_frame,
            "physical_shape": context.physical_shape,
            "configuration_shape": context.configuration_shape,
            "projection_source_sha256": context.projection_source_sha256,
            "profile_sha256": context.profile_sha256,
            "free_support_sha256": context.free_support_sha256,
            "occupied_support_sha256": context.occupied_support_sha256,
            "runner_execution_identity_sha256": context.runner_execution_identity_sha256,
            "checkpoint_file_sha256": context.checkpoint_file_sha256,
            "camera_calibration_sha256": context.camera_calibration_sha256,
            "exact_sim_tainted": context.exact_sim_tainted,
        }

    @staticmethod
    def _evidence_common(
        context: TwoResolutionTargetContextV1,
        evidence: TwoResolutionPositiveTargetEvidenceV1
        | TwoResolutionNegativeTargetEvidenceV1,
    ) -> None:
        expected = {
            "context_sha256": context.content_sha256,
            "target_id": context.target_id,
            "physical_map_frame_sha256": context.physical_map_frame.content_sha256,
            "configuration_map_frame_sha256": context.configuration_map_frame.content_sha256,
            "physical_revision": context.physical_revision,
            "configuration_revision": context.configuration_revision,
            "configuration_snapshot_sha256": context.configuration_snapshot_sha256,
            "configuration_component_sha256": context.configuration_component_sha256,
            "free_support_sha256": context.free_support_sha256,
            "occupied_support_sha256": context.occupied_support_sha256,
            "runner_execution_identity_sha256": context.runner_execution_identity_sha256,
            "checkpoint_file_sha256": context.checkpoint_file_sha256,
            "raw_outcome_file_sha256": context.raw_outcome_file_sha256,
            "raw_outcome_content_sha256": context.raw_outcome_content_sha256,
            "camera_calibration_sha256": context.camera_calibration_sha256,
        }
        if any(getattr(evidence, name) != value for name, value in expected.items()):
            raise TwoResolutionTargetMemoryBindingError(
                "evidence differs from its exact two-resolution context"
            )

    def _normalize(
        self,
        mass: Mapping[Cell, float],
        unlocalized: float,
    ) -> tuple[dict[Cell, float], float]:
        if any(not math.isfinite(value) or value <= 0.0 for value in mass.values()):
            raise TwoResolutionTargetMemoryError("posterior mass became nonpositive")
        if not math.isfinite(unlocalized) or unlocalized < 0.0:
            raise TwoResolutionTargetMemoryError("unlocalized mass became invalid")
        total = math.fsum(mass.values()) + unlocalized
        if not math.isfinite(total) or total <= 0.0:
            raise TwoResolutionTargetMemoryError("posterior normalization failed")
        result = {cell: value / total for cell, value in mass.items()}
        result_unlocalized = unlocalized / total
        floor = self._config.posterior_mass_floor
        deficits = {cell: floor - value for cell, value in result.items() if value < floor}
        deficit = math.fsum(deficits.values())
        for cell in deficits:
            result[cell] = floor
        if deficit:
            take = min(deficit, result_unlocalized)
            result_unlocalized -= take
            deficit -= take
            donors = {cell: value - floor for cell, value in result.items() if value > floor}
            available = math.fsum(donors.values())
            if deficit > available + 1e-18:
                raise TwoResolutionTargetMemoryError("posterior floor is infeasible")
            if deficit:
                for cell, room in donors.items():
                    result[cell] -= room * deficit / available
        correction = 1.0 - (math.fsum(result.values()) + result_unlocalized)
        if result_unlocalized + correction >= 0.0:
            result_unlocalized += correction
        elif result:
            donor = max(result, key=result.get)  # type: ignore[arg-type]
            result[donor] += correction
        else:
            raise TwoResolutionTargetMemoryError("posterior correction failed")
        if any(value < floor or not math.isfinite(value) for value in result.values()):
            raise TwoResolutionTargetMemoryError("posterior floor enforcement failed")
        return result, result_unlocalized

    def _positive(
        self,
        prior: Mapping[Cell, float],
        unlocalized: float,
        rows: tuple[ConfigurationCellProbabilityV1, ...],
        confidence: float,
    ) -> tuple[dict[Cell, float], float]:
        localized = math.fsum(row.value for row in rows)
        transfer = min(
            unlocalized,
            self._config.maximum_positive_transfer * confidence * localized,
        )
        result = dict(prior)
        applied = 0.0
        if localized > 0.0 and transfer > 0.0:
            for row in rows:
                addition = transfer * row.value / localized
                if row.cell not in result and addition < self._config.posterior_mass_floor:
                    continue
                result[row.cell] = result.get(row.cell, 0.0) + addition
                applied += addition
        return self._normalize(result, unlocalized - applied)

    def _negative(
        self,
        prior: Mapping[Cell, float],
        unlocalized: float,
        rows: tuple[ConfigurationCellProbabilityV1, ...],
        confidence: float,
    ) -> tuple[dict[Cell, float], float]:
        result = dict(prior)
        moved = 0.0
        floor = self._config.posterior_mass_floor
        for row in rows:
            if row.cell not in result:
                continue
            multiplier = max(
                self._config.negative_mass_floor_multiplier,
                1.0 - confidence * row.value,
            )
            before = result[row.cell]
            result[row.cell] = max(before * multiplier, floor)
            moved += before - result[row.cell]
        return self._normalize(result, unlocalized + moved)

    def apply(
        self,
        context: TwoResolutionTargetContextV1,
        evidence: TwoResolutionPositiveTargetEvidenceV1
        | TwoResolutionNegativeTargetEvidenceV1,
    ) -> TwoResolutionTargetPosteriorSnapshotV1:
        if type(context) is not TwoResolutionTargetContextV1:
            raise TypeError("context has the wrong exact type")
        if type(evidence) not in {
            TwoResolutionPositiveTargetEvidenceV1,
            TwoResolutionNegativeTargetEvidenceV1,
        }:
            raise TypeError("evidence has the wrong exact type")
        context.assert_integrity()
        evidence.assert_integrity()
        self._config.assert_integrity()
        self._evidence_common(context, evidence)
        if context.target_id not in self._mass:
            raise TwoResolutionTargetMemoryError("context target is not registered")
        if context.content_sha256 in self._seen_context_ids:
            raise TwoResolutionTargetMemoryReplayError("context was already applied")
        if evidence.content_sha256 in self._seen_evidence_hashes:
            raise TwoResolutionTargetMemoryReplayError("evidence was already applied")
        if context.raw_outcome_content_sha256 in self._seen_raw_outcomes:
            raise TwoResolutionTargetMemoryReplayError("raw V5 outcome was already applied")
        if context.context_sequence <= self._last_context_sequence:
            raise TwoResolutionTargetMemoryReplayError("context sequence did not advance")
        if context.pose_timestamp_ns < self._last_pose_timestamp_ns:
            raise TwoResolutionTargetMemoryBindingError("pose timestamp rolled back")
        immutable = self._immutable_context_binding(context)
        if self._immutable_binding is not None and immutable != self._immutable_binding:
            raise TwoResolutionTargetMemoryBindingError(
                "two-resolution memory immutable execution binding changed"
            )
        for prior_context in self._contexts.values():
            if (
                context.physical_revision < prior_context.physical_revision
                or context.configuration_revision < prior_context.configuration_revision
            ):
                raise TwoResolutionTargetMemoryBindingError("map revision rolled back")

        target = context.target_id
        candidate = context.candidate_domain
        retained = {cell: value for cell, value in self._mass[target].items() if cell in candidate}
        removed = math.fsum(
            value for cell, value in self._mass[target].items() if cell not in candidate
        )
        prior, prior_unlocalized = self._normalize(
            retained,
            self._unlocalized[target] + removed,
        )
        if type(evidence) is TwoResolutionPositiveTargetEvidenceV1:
            if any(row.cell not in candidate for row in evidence.localized_distribution):
                raise TwoResolutionTargetMemoryBindingError(
                    "positive evidence escaped the candidate domain"
                )
            next_mass, next_unlocalized = self._positive(
                prior,
                prior_unlocalized,
                evidence.localized_distribution,
                evidence.confidence,
            )
            kind = "positive"
        else:
            if any(
                row.cell not in candidate
                for row in evidence.visible_detection_probability
            ):
                raise TwoResolutionTargetMemoryBindingError(
                    "negative evidence escaped the candidate domain"
                )
            next_mass, next_unlocalized = self._negative(
                prior,
                prior_unlocalized,
                evidence.visible_detection_probability,
                evidence.confidence,
            )
            kind = "negative"

        try:
            self._issuer.consume_evidence(evidence)
        except (
            TwoResolutionTargetEvidenceBindingError,
            TwoResolutionTargetEvidenceReplayError,
        ) as exc:
            raise TwoResolutionTargetMemoryBindingError(str(exc)) from exc
        self._immutable_binding = immutable
        self._mass[target] = next_mass
        self._unlocalized[target] = next_unlocalized
        self._contexts[target] = context
        self._last_context_sequence = context.context_sequence
        self._last_pose_timestamp_ns = context.pose_timestamp_ns
        self._revision += 1
        self._seen_context_ids.add(context.content_sha256)
        self._seen_evidence_hashes.add(evidence.content_sha256)
        self._seen_raw_outcomes.add(context.raw_outcome_content_sha256)
        if kind == "positive":
            self._positive_count[target] += 1
        else:
            self._negative_count[target] += 1
        self._chains[target] = _sha256(
            {
                "schema": "lewm_g5_two_resolution_target_chain_transition_v1",
                "previous_sha256": self._chains[target],
                "kind": kind,
                "context_sha256": context.content_sha256,
                "evidence_sha256": evidence.content_sha256,
                "memory_revision": self._revision,
            }
        )
        return self.snapshot(target)

    def _build_snapshot(self, target_id: str) -> TwoResolutionTargetPosteriorSnapshotV1:
        context = self._contexts[target_id]
        return TwoResolutionTargetPosteriorSnapshotV1(
            target_id=target_id,
            target_memory_instance_sha256=self._instance_sha256,
            target_memory_config_sha256=self._config.content_sha256,
            target_memory_revision=self._revision,
            context_sequence=context.context_sequence,
            context_sha256=context.content_sha256,
            physical_map_frame=context.physical_map_frame,
            configuration_map_frame=context.configuration_map_frame,
            physical_shape=context.physical_shape,
            configuration_shape=context.configuration_shape,
            physical_revision=context.physical_revision,
            configuration_revision=context.configuration_revision,
            configuration_snapshot_sha256=context.configuration_snapshot_sha256,
            configuration_component_sha256=context.configuration_component_sha256,
            profile_sha256=context.profile_sha256,
            free_support_sha256=context.free_support_sha256,
            occupied_support_sha256=context.occupied_support_sha256,
            projection_source_sha256=context.projection_source_sha256,
            runner_execution_identity_sha256=context.runner_execution_identity_sha256,
            checkpoint_file_sha256=context.checkpoint_file_sha256,
            camera_calibration_sha256=context.camera_calibration_sha256,
            candidate_domain=context.candidate_domain,
            excluded_target_configuration_cells=(
                context.excluded_target_configuration_cells
            ),
            cell_mass=tuple(
                TwoResolutionTargetCellMassV1(cell=cell, value=value)
                for cell, value in sorted(self._mass[target_id].items())
            ),
            unlocalized_mass=self._unlocalized[target_id],
            positive_evidence_count=self._positive_count[target_id],
            negative_evidence_count=self._negative_count[target_id],
            evidence_chain_sha256=self._chains[target_id],
            exact_sim_tainted=context.exact_sim_tainted,
            _issuance_capability=self._capability,
        )

    def snapshot(self, target_id: str) -> TwoResolutionTargetPosteriorSnapshotV1:
        if target_id not in self._mass:
            raise KeyError(target_id)
        if target_id not in self._contexts:
            raise TwoResolutionTargetMemoryError("target has no committed context")
        result = self._build_snapshot(target_id)
        self._issued_snapshots[id(result)] = result
        return result

    def assert_current_snapshot(
        self, snapshot: TwoResolutionTargetPosteriorSnapshotV1
    ) -> None:
        if type(snapshot) is not TwoResolutionTargetPosteriorSnapshotV1:
            raise TypeError("snapshot has the wrong exact type")
        snapshot.assert_integrity()
        if self._issued_snapshots.get(id(snapshot)) is not snapshot:
            raise TwoResolutionTargetMemoryBindingError(
                "posterior snapshot is not the exact live object issued here"
            )
        if snapshot.target_memory_revision != self._revision:
            raise StaleSnapshotError("target posterior snapshot is stale")
        expected = self._build_snapshot(snapshot.target_id)
        if expected.content_sha256 != snapshot.content_sha256:
            raise TwoResolutionTargetMemoryBindingError(
                "posterior snapshot differs from current memory state"
            )

    def hypotheses(
        self, snapshot: TwoResolutionTargetPosteriorSnapshotV1
    ) -> tuple[TwoResolutionTargetHypothesisV1, ...]:
        self.assert_current_snapshot(snapshot)
        mass = {
            row.cell: row.value
            for row in snapshot.cell_mass
            if row.value >= self._config.component_mass_floor
        }
        remaining = set(mass)
        hypotheses: list[TwoResolutionTargetHypothesisV1] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component = {seed}
            remaining.remove(seed)
            while stack:
                x, y = stack.pop()
                for neighbor in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            total = math.fsum(mass[cell] for cell in component)
            peak = min(component, key=lambda cell: (-mass[cell], cell))
            mean_x = math.fsum(
                snapshot.configuration_map_frame.cell_center(cell)[0] * mass[cell]
                for cell in component
            ) / total
            mean_y = math.fsum(
                snapshot.configuration_map_frame.cell_center(cell)[1] * mass[cell]
                for cell in component
            ) / total
            hypotheses.append(
                TwoResolutionTargetHypothesisV1(
                    target_id=snapshot.target_id,
                    cells=frozenset(component),
                    mass=total,
                    peak_cell=peak,
                    peak_mass=mass[peak],
                    mean_world_xy_m=(mean_x, mean_y),
                )
            )
        return tuple(
            sorted(
                hypotheses,
                key=lambda item: (-item.mass, -item.peak_mass, item.peak_cell),
            )
        )


def require_production_two_resolution_target_memory() -> object:
    if PRODUCTION_TWO_RESOLUTION_TARGET_MEMORY is None:
        raise PermissionError("production two-resolution target memory is unset")
    return PRODUCTION_TWO_RESOLUTION_TARGET_MEMORY


__all__ = [
    "PRODUCTION_TWO_RESOLUTION_TARGET_MEMORY",
    "TwoResolutionReversibleTargetBeliefMemoryV1",
    "TwoResolutionTargetCellMassV1",
    "TwoResolutionTargetHypothesisV1",
    "TwoResolutionTargetMemoryBindingError",
    "TwoResolutionTargetMemoryConfigV1",
    "TwoResolutionTargetMemoryError",
    "TwoResolutionTargetMemoryReplayError",
    "TwoResolutionTargetPosteriorSnapshotV1",
    "require_production_two_resolution_target_memory",
]
