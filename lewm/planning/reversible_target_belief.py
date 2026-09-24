"""Synthetic reversible sparse target-belief foundation for the G5 stack.

The controller owns sparse probability mass over physical world cells and an
explicit unlocalized pool.  Physical context, negative visibility, posterior
snapshots, task identity, and claim credit remain separately validated here.
Production observation admission is intentionally absent; the no-argument
one-shot runner boundary stays fail-closed until reviewed identities exist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import secrets
from typing import Iterable, Mapping, Sequence


Cell = tuple[int, int]
_MAX_ABS_CELL_INDEX = 10_000_000


class TargetEvidenceRejectedError(ValueError):
    """Raised before mutation when target evidence is stale or malformed."""


class TargetSnapshotBindingError(ValueError):
    """Raised when a context or posterior snapshot lacks live authority."""


class TargetClaimVerificationError(ValueError):
    """Raised when canonical physical claim credit cannot be established."""


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


def _require_sha256(value: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _unit(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _cell(value: Sequence[int], name: str = "cell") -> Cell:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain two integers")
    if any(abs(item) > _MAX_ABS_CELL_INDEX for item in value):
        raise ValueError(f"{name} exceeds the supported metric lattice extent")
    return (value[0], value[1])


def _cells_json(cells: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(cells)]


def _strict_keys(
    value: object,
    *,
    required: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _canonical_json_text(value: object, name: str) -> str:
    try:
        return _canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{name} is not canonical JSON data") from exc


@dataclass(frozen=True)
class TargetMemoryConfig:
    target_ids: tuple[str, ...] = ("blue", "green", "red", "yellow")
    cell_size_m: float = 0.10
    origin_xy_m: tuple[float, float] = (0.0, 0.0)
    maximum_positive_transfer: float = 0.50
    negative_mass_floor_multiplier: float = 1e-4
    posterior_mass_floor: float = 1e-15
    component_mass_floor: float = 1e-12
    covariance_floor_m2: float = 1e-6
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_ids, tuple)
            or not self.target_ids
            or tuple(sorted(self.target_ids)) != self.target_ids
            or len(set(self.target_ids)) != len(self.target_ids)
            or any(type(value) is not str or not value for value in self.target_ids)
        ):
            raise ValueError("target_ids must be nonempty, sorted, and unique")
        cell_size = _finite(self.cell_size_m, "cell_size_m")
        if not 0.01 <= cell_size <= 10.0:
            raise ValueError("cell_size_m must lie in [0.01,10]")
        if not isinstance(self.origin_xy_m, tuple) or len(self.origin_xy_m) != 2:
            raise ValueError("origin_xy_m must be a two-value tuple")
        origin = tuple(_finite(value, "origin_xy_m") for value in self.origin_xy_m)
        if any(abs(value) > 1_000_000.0 for value in origin):
            raise ValueError("origin_xy_m is outside the supported range")
        transfer = _unit(self.maximum_positive_transfer, "maximum_positive_transfer")
        if not 0.0 < transfer < 1.0:
            raise ValueError("maximum_positive_transfer must lie strictly in (0,1)")
        negative_floor = _finite(
            self.negative_mass_floor_multiplier,
            "negative_mass_floor_multiplier",
        )
        posterior_floor = _finite(self.posterior_mass_floor, "posterior_mass_floor")
        component_floor = _finite(self.component_mass_floor, "component_mass_floor")
        covariance_floor = _finite(self.covariance_floor_m2, "covariance_floor_m2")
        if not 0.0 < negative_floor < 1.0:
            raise ValueError("negative multiplier floor must lie strictly in (0,1)")
        if not 0.0 < posterior_floor < component_floor < 1.0:
            raise ValueError(
                "posterior/component floors must satisfy 0 < posterior < component < 1"
            )
        if not 0.0 < covariance_floor <= cell_size * cell_size:
            raise ValueError("covariance floor must lie in (0, cell_size_m^2]")
        object.__setattr__(self, "cell_size_m", cell_size)
        object.__setattr__(self, "origin_xy_m", origin)
        object.__setattr__(self, "maximum_positive_transfer", transfer)
        object.__setattr__(self, "negative_mass_floor_multiplier", negative_floor)
        object.__setattr__(self, "posterior_mass_floor", posterior_floor)
        object.__setattr__(self, "component_mass_floor", component_floor)
        object.__setattr__(self, "covariance_floor_m2", covariance_floor)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_target_memory_config_v2",
            "target_ids": list(self.target_ids),
            "cell_size_m": self.cell_size_m,
            "origin_xy_m": list(self.origin_xy_m),
            "maximum_positive_transfer": self.maximum_positive_transfer,
            "negative_mass_floor_multiplier": self.negative_mass_floor_multiplier,
            "posterior_mass_floor": self.posterior_mass_floor,
            "component_mass_floor": self.component_mass_floor,
            "covariance_floor_m2": self.covariance_floor_m2,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetSnapshotBindingError("target-memory config was mutated")


@dataclass(frozen=True)
class TargetEpisodeAuthority:
    """Manifest and independently supplied task set fixed before an episode."""

    scene_id: str
    episode_id: str
    physical_manifest_sha256: str
    context_issuer_contract_sha256: str
    expected_task_object_ids: tuple[str, ...]
    task_object_by_target_id: tuple[tuple[str, str], ...]
    task_object_set_sha256: str
    evaluator_observer_mode: str = "end_of_episode_observer_only"
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("scene_id", "episode_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        _require_sha256(self.physical_manifest_sha256, "physical_manifest_sha256")
        _require_sha256(
            self.context_issuer_contract_sha256,
            "context_issuer_contract_sha256",
        )
        task_ids = tuple(self.expected_task_object_ids)
        if (
            not task_ids
            or task_ids != tuple(sorted(task_ids))
            or len(set(task_ids)) != len(task_ids)
            or any(type(value) is not str or not value for value in task_ids)
        ):
            raise ValueError("expected task IDs must be nonempty, sorted, and unique")
        try:
            mapping = tuple(sorted((target, object_id) for target, object_id in self.task_object_by_target_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("task-object mapping is malformed") from exc
        if (
            not mapping
            or len({target for target, _ in mapping}) != len(mapping)
            or len({object_id for _, object_id in mapping}) != len(mapping)
            or any(
                type(target) is not str
                or not target
                or type(object_id) is not str
                or not object_id
                for target, object_id in mapping
            )
            or tuple(sorted(object_id for _, object_id in mapping)) != task_ids
        ):
            raise ValueError("task mapping must be one-to-one and cover expected task IDs")
        task_hash = _sha256(
            {
                "schema": "lewm_go2_claim_task_set_v1",
                "scene_id": self.scene_id,
                "physical_manifest_sha256": self.physical_manifest_sha256,
                "task_object_ids": list(task_ids),
            }
        )
        if self.task_object_set_sha256 != task_hash:
            raise ValueError("task-object set commitment changed")
        if self.evaluator_observer_mode != "end_of_episode_observer_only":
            raise ValueError("evaluator must remain an end-of-episode observer")
        object.__setattr__(self, "expected_task_object_ids", task_ids)
        object.__setattr__(self, "task_object_by_target_id", mapping)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_target_episode_authority_v2",
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "context_issuer_contract_sha256": self.context_issuer_contract_sha256,
            "expected_task_object_ids": list(self.expected_task_object_ids),
            "task_object_by_target_id": [
                {"target_id": target, "object_id": object_id}
                for target, object_id in self.task_object_by_target_id
            ],
            "task_object_set_sha256": self.task_object_set_sha256,
            "evaluator_observer_mode": self.evaluator_observer_mode,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetSnapshotBindingError("episode authority was mutated")


@dataclass(frozen=True)
class TargetMemoryContext:
    """G3-issued current physical/configuration projection for G5."""

    issuer_sha256: str
    issuance_id_sha256: str
    context_sequence: int
    pose_timestamp_ns: int
    map_frame_sha256: str
    physical_content_sha256: str
    physical_revision: int
    configuration_snapshot_sha256: str
    morphology_sha256: str
    pose_provenance_sha256: str
    camera_calibration_sha256: str
    frustum_contract_sha256: str
    physical_los_contract_sha256: str
    positive_evidence_producer_sha256: str
    negative_visibility_producer_sha256: str
    observation_model_checkpoint_sha256: str
    candidate_domain: frozenset[Cell]
    exact_sim_tainted: bool
    ablation_mode: str
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "issuer_sha256",
            "issuance_id_sha256",
            "map_frame_sha256",
            "physical_content_sha256",
            "configuration_snapshot_sha256",
            "morphology_sha256",
            "pose_provenance_sha256",
            "camera_calibration_sha256",
            "frustum_contract_sha256",
            "physical_los_contract_sha256",
            "positive_evidence_producer_sha256",
            "negative_visibility_producer_sha256",
            "observation_model_checkpoint_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.context_sequence, "context_sequence")
        _nonnegative_int(self.pose_timestamp_ns, "pose_timestamp_ns")
        _nonnegative_int(self.physical_revision, "physical_revision")
        if self.context_sequence == 0:
            raise ValueError("context_sequence must be positive")
        expected_issuance_id = _sha256(
            {
                "schema": "lewm_g5_context_issuance_identity_v1",
                "issuer_sha256": self.issuer_sha256,
                "sequence": self.context_sequence,
                "physical_revision": self.physical_revision,
                "pose_timestamp_ns": self.pose_timestamp_ns,
                "configuration_snapshot_sha256": (
                    self.configuration_snapshot_sha256
                ),
                "pose_provenance_sha256": self.pose_provenance_sha256,
            }
        )
        if self.issuance_id_sha256 != expected_issuance_id:
            raise ValueError("target context issuance identity is inconsistent")
        if self._issuance_capability is None:
            raise TypeError("target context requires G3 issuance capability")
        if not isinstance(self.exact_sim_tainted, bool):
            raise TypeError("exact_sim_tainted must be boolean")
        if self.ablation_mode not in {"none", "exact_sim_odometry_ablation"}:
            raise ValueError("unsupported target-memory ablation mode")
        if self.exact_sim_tainted != (
            self.ablation_mode == "exact_sim_odometry_ablation"
        ):
            raise ValueError("target context taint/ablation mismatch")
        domain = frozenset(_cell(cell, "candidate_domain cell") for cell in self.candidate_domain)
        if not domain:
            raise ValueError("candidate_domain must be nonempty")
        object.__setattr__(self, "candidate_domain", domain)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_target_memory_context_v2",
            "issuer_sha256": self.issuer_sha256,
            "issuance_id_sha256": self.issuance_id_sha256,
            "context_sequence": self.context_sequence,
            "pose_timestamp_ns": self.pose_timestamp_ns,
            "map_frame_sha256": self.map_frame_sha256,
            "physical_content_sha256": self.physical_content_sha256,
            "physical_revision": self.physical_revision,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "morphology_sha256": self.morphology_sha256,
            "pose_provenance_sha256": self.pose_provenance_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "frustum_contract_sha256": self.frustum_contract_sha256,
            "physical_los_contract_sha256": self.physical_los_contract_sha256,
            "positive_evidence_producer_sha256": (
                self.positive_evidence_producer_sha256
            ),
            "negative_visibility_producer_sha256": self.negative_visibility_producer_sha256,
            "observation_model_checkpoint_sha256": self.observation_model_checkpoint_sha256,
            "candidate_domain": _cells_json(self.candidate_domain),
            "exact_sim_tainted": self.exact_sim_tainted,
            "ablation_mode": self.ablation_mode,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetSnapshotBindingError("target-memory context was mutated")


def _context_transition_error(
    current: TargetMemoryContext,
    candidate: TargetMemoryContext,
) -> str | None:
    for name in (
        "issuer_sha256",
        "map_frame_sha256",
        "morphology_sha256",
        "camera_calibration_sha256",
        "frustum_contract_sha256",
        "physical_los_contract_sha256",
        "positive_evidence_producer_sha256",
        "negative_visibility_producer_sha256",
        "observation_model_checkpoint_sha256",
        "exact_sim_tainted",
        "ablation_mode",
    ):
        if getattr(candidate, name) != getattr(current, name):
            return f"target memory cannot change {name}"
    if candidate.context_sequence <= current.context_sequence:
        return "context sequence cannot roll back"
    if candidate.physical_revision < current.physical_revision:
        return "physical context revision cannot roll back"
    if candidate.pose_timestamp_ns < current.pose_timestamp_ns:
        return "pose provenance timestamp cannot roll back"
    if candidate.physical_revision == current.physical_revision:
        for name in (
            "physical_content_sha256",
            "configuration_snapshot_sha256",
            "candidate_domain",
        ):
            if getattr(candidate, name) != getattr(current, name):
                return f"pose-only context cannot change {name}"
        if candidate.pose_timestamp_ns <= current.pose_timestamp_ns:
            return "pose-only context must advance pose timestamp"
        if candidate.pose_provenance_sha256 == current.pose_provenance_sha256:
            return "pose-only context must advance pose provenance"
    return None


@dataclass(frozen=True)
class NegativeVisibilityCertificate:
    """One-use G3 proof for one exact negative-likelihood transaction."""

    issuer_sha256: str
    certificate_id_sha256: str
    context_sha256: str
    physical_content_sha256: str
    configuration_snapshot_sha256: str
    pose_provenance_sha256: str
    frustum_contract_sha256: str
    physical_los_contract_sha256: str
    producer_sha256: str
    evidence_identity_sha256: str
    target_id: str
    confidence: float
    certified_detection_probability: tuple[tuple[Cell, float], ...]
    _certificate_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "issuer_sha256",
            "certificate_id_sha256",
            "context_sha256",
            "physical_content_sha256",
            "configuration_snapshot_sha256",
            "pose_provenance_sha256",
            "frustum_contract_sha256",
            "physical_los_contract_sha256",
            "producer_sha256",
            "evidence_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("certificate target_id must be nonempty")
        confidence = _unit(self.confidence, "certificate confidence")
        if confidence <= 0.0:
            raise ValueError("certificate confidence must be positive")
        if self._certificate_capability is None:
            raise TypeError("negative visibility requires G3 certificate capability")
        rows: list[tuple[Cell, float]] = []
        for raw_cell, raw_value in self.certified_detection_probability:
            cell = _cell(raw_cell, "certified visible cell")
            probability = _unit(raw_value, "certified detection probability")
            if probability <= 0.0:
                raise ValueError("certified detection probability must be positive")
            rows.append((cell, probability))
        rows.sort(key=lambda row: row[0])
        if not rows:
            raise ValueError("negative visibility certificate cannot be empty")
        if len({cell for cell, _ in rows}) != len(rows):
            raise ValueError("negative visibility certificate contains duplicate cells")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "certified_detection_probability", tuple(rows))
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def certified_visible_cells(self) -> frozenset[Cell]:
        return frozenset(cell for cell, _ in self.certified_detection_probability)

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_negative_visibility_certificate_v2",
            "issuer_sha256": self.issuer_sha256,
            "certificate_id_sha256": self.certificate_id_sha256,
            "context_sha256": self.context_sha256,
            "physical_content_sha256": self.physical_content_sha256,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "pose_provenance_sha256": self.pose_provenance_sha256,
            "frustum_contract_sha256": self.frustum_contract_sha256,
            "physical_los_contract_sha256": self.physical_los_contract_sha256,
            "producer_sha256": self.producer_sha256,
            "evidence_identity_sha256": self.evidence_identity_sha256,
            "target_id": self.target_id,
            "confidence": self.confidence,
            "certified_detection_probability": [
                {"cell": list(cell), "value": value}
                for cell, value in self.certified_detection_probability
            ],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetEvidenceRejectedError("negative visibility certificate was mutated")


class SyntheticTargetMemoryContextIssuer:
    """Synthetic-only issuer retaining posterior and serialization tests."""

    # Keep the allocation identity outside ``__dict__``. An object shell made
    # with object.__new__ followed by __dict__.update therefore cannot inherit
    # issuer authority, even though all visible values and capabilities match.
    __slots__ = ("_exact_instance_owner", "__dict__")

    def __init__(
        self,
        *,
        issuer_sha256: str,
        frustum_contract_sha256: str,
        physical_los_contract_sha256: str,
        positive_evidence_producer_sha256: str,
        negative_visibility_producer_sha256: str,
        camera_calibration_sha256: str,
        observation_model_checkpoint_sha256: str,
        _physical_memory: object,
        _synthetic_test_fixture: bool,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise TypeError(
                "SyntheticTargetMemoryContextIssuer is available only to synthetic tests"
            )
        from lewm.planning.revisioned_physical_configuration_memory import (
            RevisionedPhysicalMemory,
        )

        if not isinstance(_physical_memory, RevisionedPhysicalMemory):
            raise TypeError("physical memory must be RevisionedPhysicalMemory")
        object.__setattr__(self, "_exact_instance_owner", self)
        for name, value in (
            ("issuer_sha256", issuer_sha256),
            ("frustum_contract_sha256", frustum_contract_sha256),
            ("physical_los_contract_sha256", physical_los_contract_sha256),
            ("positive_evidence_producer_sha256", positive_evidence_producer_sha256),
            ("negative_visibility_producer_sha256", negative_visibility_producer_sha256),
            ("camera_calibration_sha256", camera_calibration_sha256),
            (
                "observation_model_checkpoint_sha256",
                observation_model_checkpoint_sha256,
            ),
        ):
            _require_sha256(value, name)
        self.issuer_sha256 = issuer_sha256
        self.instance_binding_sha256 = hashlib.sha256(
            secrets.token_bytes(32)
        ).hexdigest()
        self.frustum_contract_sha256 = frustum_contract_sha256
        self.physical_los_contract_sha256 = physical_los_contract_sha256
        self.positive_evidence_producer_sha256 = positive_evidence_producer_sha256
        self.negative_visibility_producer_sha256 = negative_visibility_producer_sha256
        self.camera_calibration_sha256 = camera_calibration_sha256
        self.observation_model_checkpoint_sha256 = (
            observation_model_checkpoint_sha256
        )
        self._physical_memory = _physical_memory
        self._g3_memory_config_sha256 = _physical_memory.config.content_sha256
        self._g3_map_frame_sha256 = _physical_memory.map_frame.content_sha256
        self._context_capability = object()
        self._certificate_capability = object()
        self._positive_capability = object()
        self._producer_capability = object()
        self._issued_contexts: dict[int, TargetMemoryContext] = {}
        self._known_context_sha256s: set[str] = set()
        self._physical_content_by_revision: dict[int, str] = {}
        self._consumed_context_ids: set[int] = set()
        self._issued_certificates: dict[int, NegativeVisibilityCertificate] = {}
        self._known_certificate_sha256s: set[str] = set()
        self._consumed_certificate_ids: set[int] = set()
        self._issued_positive_observations: dict[int, PositiveTargetObservation] = {}
        self._known_positive_observation_sha256s: set[str] = set()
        self._consumed_positive_observation_ids: set[int] = set()
        self._known_evaluation_sha256s: set[str] = set()
        self._known_credit_sha256s: set[str] = set()
        self._evaluation_ledger_by_episode: dict[str, list[str]] = {}
        self._credit_set_by_episode: dict[str, set[str]] = {}
        self._latest_memory_revision_by_episode: dict[str, int] = {}
        self._serialized_memory_state_by_episode: dict[str, tuple[int, str]] = {}
        self._active_writer_by_episode: dict[str, object] = {}
        self._sequence = 0
        self._certificate_sequence = 0

    @property
    def synthetic_only(self) -> bool:
        return True

    @property
    def production_authority_eligible(self) -> bool:
        return False

    def __copy__(self) -> SyntheticTargetMemoryContextIssuer:
        raise TypeError("synthetic target-memory context issuer cannot be copied")

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> SyntheticTargetMemoryContextIssuer:
        del memo
        raise TypeError("synthetic target-memory context issuer cannot be copied")

    def _assert_exact_instance(self) -> None:
        try:
            owner = object.__getattribute__(self, "_exact_instance_owner")
        except AttributeError as exc:
            raise TargetSnapshotBindingError(
                "target context issuer object identity is not registered"
            ) from exc
        if owner is not self:
            raise TargetSnapshotBindingError(
                "target context issuer object identity is not registered"
            )

    @classmethod
    def _bind_g3_for_tests(
        cls,
        *,
        physical_memory: object,
        issuer_sha256: str,
        frustum_contract_sha256: str,
        physical_los_contract_sha256: str,
        positive_evidence_producer_sha256: str,
        negative_visibility_producer_sha256: str,
        camera_calibration_sha256: str,
        observation_model_checkpoint_sha256: str,
    ) -> SyntheticG3TargetMemoryContextProducer:
        """Build an irreversibly synthetic G3/G5 fixture for CPU tests."""

        issuer = cls(
            issuer_sha256=issuer_sha256,
            frustum_contract_sha256=frustum_contract_sha256,
            physical_los_contract_sha256=physical_los_contract_sha256,
            positive_evidence_producer_sha256=positive_evidence_producer_sha256,
            negative_visibility_producer_sha256=(
                negative_visibility_producer_sha256
            ),
            camera_calibration_sha256=camera_calibration_sha256,
            observation_model_checkpoint_sha256=(
                observation_model_checkpoint_sha256
            ),
            _physical_memory=physical_memory,
            _synthetic_test_fixture=True,
        )
        return SyntheticG3TargetMemoryContextProducer(
            issuer,
            _producer_capability=issuer._producer_capability,
        )

    def contract_dict(self) -> dict[str, object]:
        self._assert_exact_instance()
        return {
            "schema": "lewm_g5_context_issuer_contract_v2",
            "issuer_sha256": self.issuer_sha256,
            "instance_binding_sha256": self.instance_binding_sha256,
            "g3_memory_config_sha256": self._g3_memory_config_sha256,
            "g3_map_frame_sha256": self._g3_map_frame_sha256,
            "frustum_contract_sha256": self.frustum_contract_sha256,
            "physical_los_contract_sha256": self.physical_los_contract_sha256,
            "positive_evidence_producer_sha256": (
                self.positive_evidence_producer_sha256
            ),
            "negative_visibility_producer_sha256": self.negative_visibility_producer_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "observation_model_checkpoint_sha256": (
                self.observation_model_checkpoint_sha256
            ),
            "synthetic_only": True,
            "production_authority_eligible": False,
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256(self.contract_dict())

    def _issue_context(
        self,
        *,
        _producer_capability: object,
        pose_provenance_sha256: str,
        pose_timestamp_ns: int,
        candidate_domain: frozenset[Cell],
    ) -> TargetMemoryContext:
        self._assert_exact_instance()
        from lewm.planning.revisioned_physical_configuration_memory import (
            ConfigurationMorphology,
        )

        if _producer_capability is not self._producer_capability:
            raise TargetSnapshotBindingError(
                "context issuance requires the registered G3 producer capability"
            )
        if self._physical_memory.config.content_sha256 != self._g3_memory_config_sha256:
            raise TargetSnapshotBindingError("bound G3 memory config changed")
        if self._physical_memory.map_frame.content_sha256 != self._g3_map_frame_sha256:
            raise TargetSnapshotBindingError("bound G3 map frame changed")
        snapshot = self._physical_memory.create_configuration_snapshot(
            ConfigurationMorphology(),
            candidate_cells=candidate_domain,
        )
        self._physical_memory.assert_current_snapshot(snapshot)
        physical_revision = snapshot.physical_revision
        configuration_snapshot_sha256 = snapshot.content_sha256
        known_physical_content = self._physical_content_by_revision.get(
            physical_revision
        )
        if (
            known_physical_content is not None
            and known_physical_content != snapshot.physical_content_sha256
        ):
            raise TargetSnapshotBindingError(
                "one G3 revision produced conflicting physical content"
            )
        next_sequence = self._sequence + 1
        issuance_id = _sha256(
            {
                "schema": "lewm_g5_context_issuance_identity_v1",
                "issuer_sha256": self.issuer_sha256,
                "sequence": next_sequence,
                "physical_revision": physical_revision,
                "pose_timestamp_ns": pose_timestamp_ns,
                "configuration_snapshot_sha256": configuration_snapshot_sha256,
                "pose_provenance_sha256": pose_provenance_sha256,
            }
        )
        context = TargetMemoryContext(
            issuer_sha256=self.issuer_sha256,
            issuance_id_sha256=issuance_id,
            context_sequence=next_sequence,
            pose_timestamp_ns=pose_timestamp_ns,
            map_frame_sha256=snapshot.map_frame_sha256,
            physical_content_sha256=snapshot.physical_content_sha256,
            physical_revision=physical_revision,
            configuration_snapshot_sha256=configuration_snapshot_sha256,
            morphology_sha256=snapshot.morphology_sha256,
            pose_provenance_sha256=pose_provenance_sha256,
            camera_calibration_sha256=self.camera_calibration_sha256,
            frustum_contract_sha256=self.frustum_contract_sha256,
            physical_los_contract_sha256=self.physical_los_contract_sha256,
            positive_evidence_producer_sha256=(
                self.positive_evidence_producer_sha256
            ),
            negative_visibility_producer_sha256=self.negative_visibility_producer_sha256,
            observation_model_checkpoint_sha256=(
                self.observation_model_checkpoint_sha256
            ),
            candidate_domain=snapshot.evaluated_cells,
            exact_sim_tainted=snapshot.exact_sim_tainted,
            ablation_mode=(
                "exact_sim_odometry_ablation"
                if snapshot.exact_sim_tainted
                else "none"
            ),
            _issuance_capability=self._context_capability,
        )
        self._physical_content_by_revision[physical_revision] = (
            snapshot.physical_content_sha256
        )
        self._sequence = next_sequence
        self._issued_contexts[id(context)] = context
        self._known_context_sha256s.add(context.content_sha256)
        return context

    def assert_issued_context(self, context: TargetMemoryContext, *, consume: bool = False) -> None:
        self._assert_exact_instance()
        if not isinstance(context, TargetMemoryContext):
            raise TypeError("context must be TargetMemoryContext")
        context.assert_integrity()
        identifier = id(context)
        if (
            context.issuer_sha256 != self.issuer_sha256
            or context._issuance_capability is not self._context_capability
            or self._issued_contexts.get(identifier) is not context
            or identifier in self._consumed_context_ids
        ):
            raise TargetSnapshotBindingError(
                "target context was not issued by this G3 issuer instance"
            )
        if consume:
            self._consumed_context_ids.add(identifier)

    def _rehydrate_known_context(self, context: TargetMemoryContext) -> TargetMemoryContext:
        self._assert_exact_instance()
        self.assert_known_context_content(context)
        rebound = replace(context, _issuance_capability=self._context_capability)
        self._issued_contexts[id(rebound)] = rebound
        self._sequence = max(self._sequence, rebound.context_sequence)
        return rebound

    def assert_known_context_content(self, context: TargetMemoryContext) -> None:
        """Validate serialized history against content this issuer actually issued."""

        self._assert_exact_instance()
        context.assert_integrity()
        if (
            context.content_sha256 not in self._known_context_sha256s
            or context.issuer_sha256 != self.issuer_sha256
            or context.frustum_contract_sha256 != self.frustum_contract_sha256
            or context.physical_los_contract_sha256 != self.physical_los_contract_sha256
            or context.positive_evidence_producer_sha256
            != self.positive_evidence_producer_sha256
            or context.negative_visibility_producer_sha256
            != self.negative_visibility_producer_sha256
        ):
            raise TargetSnapshotBindingError("serialized context is unknown to this issuer")

    def assert_context_matches_live_g3(self, context: TargetMemoryContext) -> None:
        self._assert_exact_instance()
        context.assert_integrity()
        issued_physical_content = self._physical_content_by_revision.get(
            context.physical_revision
        )
        if (
            self._physical_memory.config.content_sha256
            != self._g3_memory_config_sha256
            or self._physical_memory.map_frame.content_sha256
            != self._g3_map_frame_sha256
        ):
            raise TargetSnapshotBindingError("bound G3 authority changed")
        if (
            context.map_frame_sha256 != self._g3_map_frame_sha256
            or context.physical_revision != self._physical_memory.revision
            or issued_physical_content is None
            or context.physical_content_sha256 != issued_physical_content
            or context.exact_sim_tainted is not self._physical_memory.exact_sim_tainted
        ):
            raise TargetSnapshotBindingError(
                "target context is stale relative to live G3 state"
            )

    def _issue_positive_observation(
        self,
        context: TargetMemoryContext,
        *,
        _producer_capability: object,
        identity: TargetEvidenceIdentity,
        target_id: str,
        localized_distribution: Mapping[Cell, float],
        unlocalized_probability: float,
        confidence: float,
    ) -> PositiveTargetObservation:
        self._assert_exact_instance()
        if _producer_capability is not self._producer_capability:
            raise TargetSnapshotBindingError(
                "positive evidence issuance requires registered producer capability"
            )
        # The registered producer is privileged, while controller-facing memory
        # context access is deliberately defensive-copy based. Bind positive
        # evidence to issuer-known context content rather than object identity.
        self.assert_known_context_content(context)
        if not isinstance(identity, TargetEvidenceIdentity):
            raise TypeError("identity must be TargetEvidenceIdentity")
        if identity.producer_sha256 != self.positive_evidence_producer_sha256:
            raise TargetEvidenceRejectedError(
                "positive evidence producer is not registered"
            )
        if not isinstance(localized_distribution, Mapping):
            raise TypeError("localized_distribution must be a mapping")
        observation = PositiveTargetObservation(
            identity=identity,
            context_sha256=context.content_sha256,
            target_id=target_id,
            localized_distribution=tuple(
                TargetCellValue(cell=cell, value=value)
                for cell, value in localized_distribution.items()
            ),
            unlocalized_probability=unlocalized_probability,
            confidence=confidence,
            _issuance_capability=self._positive_capability,
        )
        self._issued_positive_observations[id(observation)] = observation
        self._known_positive_observation_sha256s.add(observation.content_sha256)
        return observation

    def assert_issued_positive_observation(
        self,
        observation: PositiveTargetObservation,
        *,
        consume: bool = False,
    ) -> None:
        self._assert_exact_instance()
        observation.assert_integrity()
        identifier = id(observation)
        if (
            observation._issuance_capability is not self._positive_capability
            or self._issued_positive_observations.get(identifier) is not observation
            or identifier in self._consumed_positive_observation_ids
        ):
            raise TargetEvidenceRejectedError(
                "positive observation was not issued by the registered producer"
            )
        if consume:
            self._consumed_positive_observation_ids.add(identifier)

    def assert_known_positive_observation_content(
        self,
        observation: PositiveTargetObservation,
    ) -> None:
        self._assert_exact_instance()
        observation.assert_integrity()
        if (
            observation.content_sha256
            not in self._known_positive_observation_sha256s
            or observation.identity.producer_sha256
            != self.positive_evidence_producer_sha256
        ):
            raise TargetSnapshotBindingError(
                "serialized positive observation is unknown to registered producer"
            )

    def _issue_negative_visibility_certificate(
        self,
        context: TargetMemoryContext,
        *,
        _producer_capability: object,
        identity: TargetEvidenceIdentity,
        target_id: str,
        visible_detection_probability: Mapping[Cell, float],
        confidence: float,
    ) -> NegativeVisibilityCertificate:
        self._assert_exact_instance()
        if _producer_capability is not self._producer_capability:
            raise TargetSnapshotBindingError(
                "negative certification requires the registered producer capability"
            )
        context.assert_integrity()
        if (
            context.issuer_sha256 != self.issuer_sha256
            or context._issuance_capability is not self._context_capability
            or self._issued_contexts.get(id(context)) is not context
        ):
            raise TargetSnapshotBindingError("certificate context is not G3-issued")
        if not isinstance(identity, TargetEvidenceIdentity):
            raise TypeError("identity must be TargetEvidenceIdentity")
        if identity.producer_sha256 != self.negative_visibility_producer_sha256:
            raise TargetEvidenceRejectedError(
                "negative evidence producer is not registered with G3"
            )
        if type(target_id) is not str or not target_id:
            raise ValueError("target_id must be nonempty")
        if not isinstance(visible_detection_probability, Mapping):
            raise TypeError("visible_detection_probability must be a mapping")
        rows = tuple(
            sorted(
                (
                    _cell(cell),
                    _unit(probability, "visible detection probability"),
                )
                for cell, probability in visible_detection_probability.items()
            )
        )
        if not rows or any(probability <= 0.0 for _, probability in rows):
            raise ValueError("visible detection probabilities must be nonempty and positive")
        confidence_value = _unit(confidence, "negative confidence")
        if confidence_value <= 0.0:
            raise ValueError("negative confidence must be positive")
        next_sequence = self._certificate_sequence + 1
        certificate_id = _sha256(
            {
                "schema": "lewm_g5_negative_certificate_identity_v1",
                "issuer_sha256": self.issuer_sha256,
                "sequence": next_sequence,
                "context_sha256": context.content_sha256,
                "evidence_identity_sha256": _sha256(identity.to_dict()),
                "target_id": target_id,
                "confidence": confidence_value,
                "certified_detection_probability": [
                    {"cell": list(cell), "value": probability}
                    for cell, probability in rows
                ],
            }
        )
        certificate = NegativeVisibilityCertificate(
            issuer_sha256=self.issuer_sha256,
            certificate_id_sha256=certificate_id,
            context_sha256=context.content_sha256,
            physical_content_sha256=context.physical_content_sha256,
            configuration_snapshot_sha256=context.configuration_snapshot_sha256,
            pose_provenance_sha256=context.pose_provenance_sha256,
            frustum_contract_sha256=context.frustum_contract_sha256,
            physical_los_contract_sha256=context.physical_los_contract_sha256,
            producer_sha256=self.negative_visibility_producer_sha256,
            evidence_identity_sha256=_sha256(identity.to_dict()),
            target_id=target_id,
            confidence=confidence_value,
            certified_detection_probability=rows,
            _certificate_capability=self._certificate_capability,
        )
        self._certificate_sequence = next_sequence
        self._issued_certificates[id(certificate)] = certificate
        self._known_certificate_sha256s.add(certificate.content_sha256)
        return certificate

    def assert_issued_certificate(
        self,
        certificate: NegativeVisibilityCertificate,
        *,
        consume: bool = False,
    ) -> None:
        self._assert_exact_instance()
        if not isinstance(certificate, NegativeVisibilityCertificate):
            raise TypeError("certificate must be NegativeVisibilityCertificate")
        certificate.assert_integrity()
        identifier = id(certificate)
        if (
            certificate.issuer_sha256 != self.issuer_sha256
            or certificate._certificate_capability is not self._certificate_capability
            or self._issued_certificates.get(identifier) is not certificate
            or identifier in self._consumed_certificate_ids
        ):
            raise TargetEvidenceRejectedError(
                "negative visibility certificate was not issued by this G3 issuer"
            )
        if consume:
            self._consumed_certificate_ids.add(identifier)

    def assert_known_certificate_content(
        self,
        certificate: NegativeVisibilityCertificate,
    ) -> None:
        self._assert_exact_instance()
        certificate.assert_integrity()
        if (
            certificate.content_sha256 not in self._known_certificate_sha256s
            or certificate.issuer_sha256 != self.issuer_sha256
            or certificate.frustum_contract_sha256 != self.frustum_contract_sha256
            or certificate.physical_los_contract_sha256
            != self.physical_los_contract_sha256
            or certificate.producer_sha256
            != self.negative_visibility_producer_sha256
        ):
            raise TargetSnapshotBindingError(
                "serialized visibility certificate is unknown to this issuer"
            )

    def _remember_canonical_evaluation(
        self,
        record: PhysicalClaimEvaluationRecord,
        *,
        episode_authority_sha256: str,
        credit: VerifiedClaimCredit | None = None,
    ) -> None:
        self._assert_exact_instance()
        record.assert_integrity()
        _require_sha256(
            episode_authority_sha256,
            "episode_authority_sha256",
        )
        if credit is not None:
            credit.assert_integrity()
        ledger = self._evaluation_ledger_by_episode.setdefault(
            episode_authority_sha256,
            [],
        )
        if ledger and ledger[-1] == record.content_sha256:
            pass
        elif record.content_sha256 in ledger:
            raise TargetClaimVerificationError(
                "canonical evaluation receipt is not the ledger terminus"
            )
        else:
            if record.evaluation_sequence != len(ledger) + 1:
                raise TargetClaimVerificationError(
                    "canonical evaluation sequence differs from runtime ledger"
                )
            ledger.append(record.content_sha256)
        self._known_evaluation_sha256s.add(record.content_sha256)
        if credit is not None:
            self._known_credit_sha256s.add(credit.content_sha256)
            self._credit_set_by_episode.setdefault(
                episode_authority_sha256,
                set(),
            ).add(credit.content_sha256)

    def assert_known_evaluation_content(
        self,
        record: PhysicalClaimEvaluationRecord,
    ) -> None:
        self._assert_exact_instance()
        record.assert_integrity()
        if record.content_sha256 not in self._known_evaluation_sha256s:
            raise TargetSnapshotBindingError(
                "serialized evaluation is unknown to canonical runtime authority"
            )

    def assert_known_credit_content(self, credit: VerifiedClaimCredit) -> None:
        self._assert_exact_instance()
        credit.assert_integrity()
        if credit.content_sha256 not in self._known_credit_sha256s:
            raise TargetSnapshotBindingError(
                "serialized credit is unknown to canonical runtime authority"
            )

    def assert_complete_evaluation_ledger(
        self,
        *,
        episode_authority_sha256: str,
        evaluations: Sequence[PhysicalClaimEvaluationRecord],
        credits: Sequence[VerifiedClaimCredit],
    ) -> None:
        self._assert_exact_instance()
        _require_sha256(
            episode_authority_sha256,
            "episode_authority_sha256",
        )
        expected_evaluations = self._evaluation_ledger_by_episode.get(
            episode_authority_sha256,
            [],
        )
        expected_credits = self._credit_set_by_episode.get(
            episode_authority_sha256,
            set(),
        )
        if [row.content_sha256 for row in evaluations] != expected_evaluations:
            raise TargetSnapshotBindingError(
                "serialized evaluation ledger is incomplete or reordered"
            )
        if {row.content_sha256 for row in credits} != expected_credits:
            raise TargetSnapshotBindingError(
                "serialized verified-credit set is incomplete or changed"
            )

    def _remember_memory_revision(
        self,
        *,
        episode_authority_sha256: str,
        revision: int,
        writer_instance: ReversibleTargetBeliefMemory,
        writer_capability: object,
    ) -> None:
        self._assert_exact_instance()
        _require_sha256(episode_authority_sha256, "episode_authority_sha256")
        revision_value = _nonnegative_int(revision, "revision")
        self.assert_active_writer(
            episode_authority_sha256=episode_authority_sha256,
            writer_instance=writer_instance,
            writer_capability=writer_capability,
        )
        prior = self._latest_memory_revision_by_episode.get(
            episode_authority_sha256
        )
        if prior is None:
            if revision_value != 0:
                raise TargetSnapshotBindingError(
                    "initial memory revision must be zero"
                )
        elif revision_value <= prior:
            raise TargetSnapshotBindingError(
                "memory revision commitment cannot roll back or fork"
            )
        self._latest_memory_revision_by_episode[episode_authority_sha256] = (
            revision_value
        )

    def _remember_serialized_memory_state(
        self,
        *,
        episode_authority_sha256: str,
        revision: int,
        state_content_sha256: str,
        writer_instance: ReversibleTargetBeliefMemory,
        writer_capability: object,
    ) -> None:
        self._assert_exact_instance()
        _require_sha256(state_content_sha256, "state_content_sha256")
        self.assert_active_writer(
            episode_authority_sha256=episode_authority_sha256,
            writer_instance=writer_instance,
            writer_capability=writer_capability,
        )
        self.assert_latest_memory_revision(
            episode_authority_sha256=episode_authority_sha256,
            revision=revision,
        )
        prior = self._serialized_memory_state_by_episode.get(
            episode_authority_sha256
        )
        if prior is not None and prior[0] == revision and prior[1] != state_content_sha256:
            raise TargetSnapshotBindingError(
                "serialized memory state cannot fork at one revision"
            )
        self._serialized_memory_state_by_episode[episode_authority_sha256] = (
            revision,
            state_content_sha256,
        )

    def assert_latest_memory_revision(
        self,
        *,
        episode_authority_sha256: str,
        revision: int,
    ) -> None:
        self._assert_exact_instance()
        expected = self._latest_memory_revision_by_episode.get(
            episode_authority_sha256
        )
        if expected != revision:
            raise TargetSnapshotBindingError(
                "memory lacks the latest runtime state commitment: revision mismatch"
            )

    def assert_latest_memory_state(
        self,
        *,
        episode_authority_sha256: str,
        revision: int,
        state_content_sha256: str,
    ) -> None:
        self._assert_exact_instance()
        self.assert_latest_memory_revision(
            episode_authority_sha256=episode_authority_sha256,
            revision=revision,
        )
        expected = self._serialized_memory_state_by_episode.get(
            episode_authority_sha256
        )
        if expected != (revision, state_content_sha256):
            raise TargetSnapshotBindingError(
                "serialized memory lacks the latest runtime state commitment"
            )

    def claim_initial_writer(
        self,
        *,
        episode_authority_sha256: str,
        writer_instance: ReversibleTargetBeliefMemory,
        writer_capability: object,
    ) -> None:
        self._assert_exact_instance()
        _require_sha256(episode_authority_sha256, "episode_authority_sha256")
        if not isinstance(writer_instance, ReversibleTargetBeliefMemory):
            raise TypeError("writer instance must be ReversibleTargetBeliefMemory")
        if writer_capability is None:
            raise TypeError("writer capability cannot be None")
        if episode_authority_sha256 in self._active_writer_by_episode:
            raise TargetSnapshotBindingError(
                "episode already has an active target-memory writer"
            )
        self._active_writer_by_episode[episode_authority_sha256] = (
            writer_instance,
            writer_capability,
        )

    def transfer_restore_writer(
        self,
        *,
        episode_authority_sha256: str,
        writer_instance: ReversibleTargetBeliefMemory,
        writer_capability: object,
        revision: int,
        state_content_sha256: str,
    ) -> None:
        self._assert_exact_instance()
        if not isinstance(writer_instance, ReversibleTargetBeliefMemory):
            raise TypeError("writer instance must be ReversibleTargetBeliefMemory")
        self.assert_latest_memory_state(
            episode_authority_sha256=episode_authority_sha256,
            revision=revision,
            state_content_sha256=state_content_sha256,
        )
        self._active_writer_by_episode[episode_authority_sha256] = (
            writer_instance,
            writer_capability,
        )

    def assert_active_writer(
        self,
        *,
        episode_authority_sha256: str,
        writer_instance: ReversibleTargetBeliefMemory,
        writer_capability: object,
    ) -> None:
        self._assert_exact_instance()
        active = self._active_writer_by_episode.get(episode_authority_sha256)
        if (
            not isinstance(active, tuple)
            or len(active) != 2
            or active[0] is not writer_instance
            or active[1] is not writer_capability
        ):
            raise TargetSnapshotBindingError(
                "target-memory instance no longer owns the episode writer lease"
            )


class SyntheticG3TargetMemoryContextProducer:
    """Synthetic-only producer used to exercise posterior mathematics."""

    def __init__(
        self,
        context_issuer: SyntheticTargetMemoryContextIssuer,
        *,
        _producer_capability: object,
    ) -> None:
        if (
            not isinstance(context_issuer, SyntheticTargetMemoryContextIssuer)
            or _producer_capability is not context_issuer._producer_capability
        ):
            raise TypeError(
                "producer must be created by the synthetic G3 fixture builder"
            )
        self._context_issuer = context_issuer
        self._producer_capability = _producer_capability

    @property
    def synthetic_only(self) -> bool:
        return True

    @property
    def production_authority_eligible(self) -> bool:
        return False

    def __copy__(self) -> SyntheticG3TargetMemoryContextProducer:
        raise TypeError("synthetic G3 target-memory producer cannot be copied")

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> SyntheticG3TargetMemoryContextProducer:
        del memo
        raise TypeError("synthetic G3 target-memory producer cannot be copied")

    @property
    def context_issuer(self) -> SyntheticTargetMemoryContextIssuer:
        return self._context_issuer

    def _issue_context_for_tests(
        self,
        *,
        pose_provenance_sha256: str,
        pose_timestamp_ns: int,
        candidate_domain: frozenset[Cell],
    ) -> TargetMemoryContext:
        return self._context_issuer._issue_context(
            _producer_capability=self._producer_capability,
            pose_provenance_sha256=pose_provenance_sha256,
            pose_timestamp_ns=pose_timestamp_ns,
            candidate_domain=candidate_domain,
        )

    def _issue_negative_visibility_certificate_for_tests(
        self,
        context: TargetMemoryContext,
        *,
        identity: TargetEvidenceIdentity,
        target_id: str,
        visible_detection_probability: Mapping[Cell, float],
        confidence: float,
    ) -> NegativeVisibilityCertificate:
        return self._context_issuer._issue_negative_visibility_certificate(
            context,
            _producer_capability=self._producer_capability,
            identity=identity,
            target_id=target_id,
            visible_detection_probability=visible_detection_probability,
            confidence=confidence,
        )

    def _issue_positive_observation_for_tests(
        self,
        context: TargetMemoryContext,
        *,
        identity: TargetEvidenceIdentity,
        target_id: str,
        localized_distribution: Mapping[Cell, float],
        unlocalized_probability: float,
        confidence: float,
    ) -> PositiveTargetObservation:
        return self._context_issuer._issue_positive_observation(
            context,
            _producer_capability=self._producer_capability,
            identity=identity,
            target_id=target_id,
            localized_distribution=localized_distribution,
            unlocalized_probability=unlocalized_probability,
            confidence=confidence,
        )


@dataclass(frozen=True)
class TargetEvidenceIdentity:
    observation_id: str
    payload_sha256: str
    producer_sha256: str
    diversity_key: str
    tick: int

    def __post_init__(self) -> None:
        if type(self.observation_id) is not str or not self.observation_id:
            raise ValueError("observation_id must be nonempty")
        if type(self.diversity_key) is not str or not self.diversity_key:
            raise ValueError("diversity_key must be nonempty")
        _require_sha256(self.payload_sha256, "payload_sha256")
        _require_sha256(self.producer_sha256, "producer_sha256")
        _nonnegative_int(self.tick, "tick")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "payload_sha256": self.payload_sha256,
            "producer_sha256": self.producer_sha256,
            "diversity_key": self.diversity_key,
            "tick": self.tick,
        }


@dataclass(frozen=True)
class TargetCellValue:
    cell: Cell
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell))
        object.__setattr__(self, "value", _unit(self.value, "cell value"))

    def to_dict(self) -> dict[str, object]:
        return {"cell": list(self.cell), "value": self.value}


def _cell_values(
    rows: Iterable[TargetCellValue],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[TargetCellValue, ...]:
    materialized = tuple(rows)
    if any(not isinstance(row, TargetCellValue) for row in materialized):
        raise TypeError(f"{name} entries must be TargetCellValue")
    values = tuple(sorted(materialized, key=lambda row: row.cell))
    if not allow_empty and not values:
        raise ValueError(f"{name} cannot be empty")
    if len({row.cell for row in values}) != len(values):
        raise ValueError(f"{name} contains duplicate cells")
    if any(row.value <= 0.0 for row in values):
        raise ValueError(f"{name} values must be positive")
    return values


@dataclass(frozen=True)
class PositiveTargetObservation:
    identity: TargetEvidenceIdentity
    context_sha256: str
    target_id: str
    localized_distribution: tuple[TargetCellValue, ...]
    unlocalized_probability: float
    confidence: float
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TargetEvidenceIdentity):
            raise TypeError("identity must be TargetEvidenceIdentity")
        if self._issuance_capability is None:
            raise TypeError("positive observation requires registered producer capability")
        _require_sha256(self.context_sha256, "context_sha256")
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        rows = _cell_values(
            self.localized_distribution,
            name="localized_distribution",
            allow_empty=True,
        )
        unlocalized = _unit(self.unlocalized_probability, "unlocalized_probability")
        confidence = _unit(self.confidence, "confidence")
        total = math.fsum(row.value for row in rows) + unlocalized
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("positive observation distribution must sum to one")
        if confidence <= 0.0:
            raise ValueError("positive observation confidence must be positive")
        object.__setattr__(self, "localized_distribution", rows)
        object.__setattr__(self, "unlocalized_probability", unlocalized)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_positive_target_observation_v3",
            "identity": self.identity.to_dict(),
            "context_sha256": self.context_sha256,
            "target_id": self.target_id,
            "localized_distribution": [row.to_dict() for row in self.localized_distribution],
            "unlocalized_probability": self.unlocalized_probability,
            "confidence": self.confidence,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @property
    def semantic_sha256(self) -> str:
        return _sha256(
            {
                "schema": "lewm_g5_positive_semantics_v1",
                "payload_sha256": self.identity.payload_sha256,
                "producer_sha256": self.identity.producer_sha256,
                "context_sha256": self.context_sha256,
                "target_id": self.target_id,
                "localized_distribution": [row.to_dict() for row in self.localized_distribution],
                "unlocalized_probability": self.unlocalized_probability,
                "confidence": self.confidence,
            }
        )

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetEvidenceRejectedError("positive observation was mutated")


@dataclass(frozen=True)
class NegativeTargetObservation:
    identity: TargetEvidenceIdentity
    context_sha256: str
    target_id: str
    visible_detection_probability: tuple[TargetCellValue, ...]
    confidence: float
    visibility_certificate: NegativeVisibilityCertificate
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TargetEvidenceIdentity):
            raise TypeError("identity must be TargetEvidenceIdentity")
        if not isinstance(self.visibility_certificate, NegativeVisibilityCertificate):
            raise TypeError("visibility_certificate must be NegativeVisibilityCertificate")
        _require_sha256(self.context_sha256, "context_sha256")
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        rows = _cell_values(
            self.visible_detection_probability,
            name="visible_detection_probability",
            allow_empty=False,
        )
        confidence = _unit(self.confidence, "confidence")
        if confidence <= 0.0:
            raise ValueError("negative observation confidence must be positive")
        object.__setattr__(self, "visible_detection_probability", rows)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_negative_target_observation_v2",
            "identity": self.identity.to_dict(),
            "context_sha256": self.context_sha256,
            "target_id": self.target_id,
            "visible_detection_probability": [
                row.to_dict() for row in self.visible_detection_probability
            ],
            "confidence": self.confidence,
            "visibility_certificate": self.visibility_certificate.to_dict(),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @property
    def semantic_sha256(self) -> str:
        return _sha256(
            {
                "schema": "lewm_g5_negative_semantics_v1",
                "payload_sha256": self.identity.payload_sha256,
                "producer_sha256": self.identity.producer_sha256,
                "context_sha256": self.context_sha256,
                "target_id": self.target_id,
                "visible_detection_probability": [
                    row.to_dict() for row in self.visible_detection_probability
                ],
                "confidence": self.confidence,
                "visibility_certificate_sha256": self.visibility_certificate.content_sha256,
            }
        )

    def assert_integrity(self) -> None:
        self.visibility_certificate.assert_integrity()
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetEvidenceRejectedError("negative observation was mutated")


@dataclass(frozen=True)
class TargetHypothesis:
    target_id: str
    cells: frozenset[Cell]
    mass: float
    mean_xy_m: tuple[float, float]
    covariance_xy_m2: tuple[tuple[float, float], tuple[float, float]]
    positive_evidence_count: int
    negative_evidence_count: int
    evidence_diversity: int
    last_positive_tick: int | None
    age_ticks: int | None


@dataclass(frozen=True)
class TargetPosteriorSnapshot:
    target_id: str
    context_sha256: str
    config_sha256: str
    issuer_contract_sha256: str
    episode_authority_sha256: str
    physical_manifest_sha256: str
    task_object_set_sha256: str
    task_mapping_sha256: str
    exact_sim_tainted: bool
    ablation_mode: str
    target_memory_revision: int
    current_tick: int
    cell_mass: tuple[TargetCellValue, ...]
    unlocalized_mass: float
    evidence_chain_sha256: str
    positive_evidence_count: int
    negative_evidence_count: int
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        for name in (
            "context_sha256",
            "config_sha256",
            "issuer_contract_sha256",
            "episode_authority_sha256",
            "physical_manifest_sha256",
            "task_object_set_sha256",
            "task_mapping_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self._issuance_capability is None:
            raise TypeError("posterior snapshot requires an issuance capability")
        if not isinstance(self.exact_sim_tainted, bool):
            raise TypeError("exact_sim_tainted must be boolean")
        if self.ablation_mode not in {"none", "exact_sim_odometry_ablation"}:
            raise ValueError("unsupported target-memory ablation mode")
        if self.exact_sim_tainted != (
            self.ablation_mode == "exact_sim_odometry_ablation"
        ):
            raise ValueError("snapshot taint/ablation mismatch")
        _nonnegative_int(self.target_memory_revision, "target_memory_revision")
        _nonnegative_int(self.current_tick, "current_tick")
        rows = _cell_values(self.cell_mass, name="cell_mass", allow_empty=True)
        unlocalized = _unit(self.unlocalized_mass, "unlocalized_mass")
        total = math.fsum(row.value for row in rows) + unlocalized
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("target posterior mass must sum to one")
        _require_sha256(self.evidence_chain_sha256, "evidence_chain_sha256")
        _nonnegative_int(self.positive_evidence_count, "positive_evidence_count")
        _nonnegative_int(self.negative_evidence_count, "negative_evidence_count")
        object.__setattr__(self, "cell_mass", rows)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_target_posterior_snapshot_v3",
            "target_id": self.target_id,
            "context_sha256": self.context_sha256,
            "config_sha256": self.config_sha256,
            "issuer_contract_sha256": self.issuer_contract_sha256,
            "episode_authority_sha256": self.episode_authority_sha256,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "task_object_set_sha256": self.task_object_set_sha256,
            "task_mapping_sha256": self.task_mapping_sha256,
            "exact_sim_tainted": self.exact_sim_tainted,
            "ablation_mode": self.ablation_mode,
            "target_memory_revision": self.target_memory_revision,
            "current_tick": self.current_tick,
            "cell_mass": [row.to_dict() for row in self.cell_mass],
            "unlocalized_mass": self.unlocalized_mass,
            "evidence_chain_sha256": self.evidence_chain_sha256,
            "positive_evidence_count": self.positive_evidence_count,
            "negative_evidence_count": self.negative_evidence_count,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetSnapshotBindingError("target posterior snapshot was mutated")


@dataclass(frozen=True)
class ControllerClaimAttempt:
    """Controller intent, including incorrect identities; never physical credit."""

    target_id: str
    expected_object_id: str
    event_id: str
    tick: int
    target_snapshot_sha256: str
    context_sha256: str
    requested_target_json: str
    claimed_target_json: str
    identity_matches_expected: bool
    raw_event_json: str
    raw_event_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("target_id", "expected_object_id", "event_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        _nonnegative_int(self.tick, "tick")
        for name in ("target_snapshot_sha256", "context_sha256", "raw_event_sha256"):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.identity_matches_expected, bool):
            raise TypeError("identity_matches_expected must be boolean")
        for name in ("requested_target_json", "claimed_target_json", "raw_event_json"):
            value = getattr(self, name)
            if type(value) is not str:
                raise ValueError(f"{name} must be canonical JSON text")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} must be canonical JSON text") from exc
            if _canonical_json_text(parsed, name) != value:
                raise ValueError(f"{name} must be canonical JSON text")
        if _sha256(json.loads(self.raw_event_json)) != self.raw_event_sha256:
            raise ValueError("raw claim event hash mismatch")
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_controller_claim_attempt_v2",
            "target_id": self.target_id,
            "expected_object_id": self.expected_object_id,
            "event_id": self.event_id,
            "tick": self.tick,
            "target_snapshot_sha256": self.target_snapshot_sha256,
            "context_sha256": self.context_sha256,
            "requested_target_json": self.requested_target_json,
            "claimed_target_json": self.claimed_target_json,
            "identity_matches_expected": self.identity_matches_expected,
            "raw_event_json": self.raw_event_json,
            "raw_event_sha256": self.raw_event_sha256,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetClaimVerificationError("controller claim attempt was mutated")


@dataclass(frozen=True)
class PhysicalClaimEvaluationRecord:
    """Observer-only immutable result for accepted, rejected, or unverifiable input."""

    evaluation_sequence: int
    episode_authority_sha256: str
    target_id: str
    event_id: str
    status: str
    accepted: bool
    canonical_credited: bool
    verified_credit_created: bool
    reason: str
    controller_attempt_sha256: str | None
    evaluator_contract_sha256: str | None
    supplied_manifest_sha256: str | None
    task_object_set_sha256: str
    evaluation_event_sha256: str | None
    evaluation_summary_sha256: str | None
    evaluated_trace_sha256: str | None
    raw_trace_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _nonnegative_int(self.evaluation_sequence, "evaluation_sequence") == 0:
            raise ValueError("evaluation_sequence must be positive")
        _require_sha256(
            self.episode_authority_sha256,
            "episode_authority_sha256",
        )
        for name in ("target_id", "event_id", "reason"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.status not in {"accepted", "rejected", "unverifiable"}:
            raise ValueError("unsupported physical evaluation status")
        for name in ("accepted", "canonical_credited", "verified_credit_created"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.status != "accepted" and self.accepted:
            raise ValueError("only accepted status can set accepted")
        if self.verified_credit_created and not (
            self.accepted and self.canonical_credited
        ):
            raise ValueError("verified credit requires accepted canonical credit")
        _require_sha256(self.task_object_set_sha256, "task_object_set_sha256")
        _require_sha256(self.raw_trace_sha256, "raw_trace_sha256")
        for name in (
            "controller_attempt_sha256",
            "evaluator_contract_sha256",
            "supplied_manifest_sha256",
            "evaluation_event_sha256",
            "evaluation_summary_sha256",
            "evaluated_trace_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_physical_claim_evaluation_v2",
            "evaluation_sequence": self.evaluation_sequence,
            "episode_authority_sha256": self.episode_authority_sha256,
            "target_id": self.target_id,
            "event_id": self.event_id,
            "status": self.status,
            "accepted": self.accepted,
            "canonical_credited": self.canonical_credited,
            "verified_credit_created": self.verified_credit_created,
            "reason": self.reason,
            "controller_attempt_sha256": self.controller_attempt_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "supplied_manifest_sha256": self.supplied_manifest_sha256,
            "task_object_set_sha256": self.task_object_set_sha256,
            "evaluation_event_sha256": self.evaluation_event_sha256,
            "evaluation_summary_sha256": self.evaluation_summary_sha256,
            "evaluated_trace_sha256": self.evaluated_trace_sha256,
            "raw_trace_sha256": self.raw_trace_sha256,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetClaimVerificationError("physical evaluation record was mutated")


@dataclass(frozen=True)
class VerifiedClaimCredit:
    """First accepted and credited event from the canonical evaluator."""

    target_id: str
    object_id: str
    event_id: str
    tick: int
    controller_attempt_sha256: str
    evaluator_contract_sha256: str
    physical_manifest_sha256: str
    task_object_set_sha256: str
    evaluation_event_sha256: str
    evaluation_summary_sha256: str
    evaluated_trace_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("target_id", "object_id", "event_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        _nonnegative_int(self.tick, "tick")
        for name in (
            "controller_attempt_sha256",
            "evaluator_contract_sha256",
            "physical_manifest_sha256",
            "task_object_set_sha256",
            "evaluation_event_sha256",
            "evaluation_summary_sha256",
            "evaluated_trace_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_verified_claim_credit_v1",
            "target_id": self.target_id,
            "object_id": self.object_id,
            "event_id": self.event_id,
            "tick": self.tick,
            "controller_attempt_sha256": self.controller_attempt_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "task_object_set_sha256": self.task_object_set_sha256,
            "evaluation_event_sha256": self.evaluation_event_sha256,
            "evaluation_summary_sha256": self.evaluation_summary_sha256,
            "evaluated_trace_sha256": self.evaluated_trace_sha256,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TargetClaimVerificationError("verified claim credit was mutated")


class ReversibleTargetBeliefMemory:
    """Atomic sparse target memory bound to an authoritative episode."""

    def __init__(
        self,
        context: TargetMemoryContext,
        config: TargetMemoryConfig | None = None,
        *,
        context_issuer: SyntheticTargetMemoryContextIssuer,
        episode_authority: TargetEpisodeAuthority,
        _is_restore: bool = False,
        _restore_finalized: bool = False,
    ) -> None:
        if not isinstance(context, TargetMemoryContext):
            raise TypeError("context must be TargetMemoryContext")
        if not isinstance(context_issuer, SyntheticTargetMemoryContextIssuer):
            raise TypeError("context_issuer must be SyntheticTargetMemoryContextIssuer")
        if not isinstance(episode_authority, TargetEpisodeAuthority):
            raise TypeError("episode_authority must be TargetEpisodeAuthority")
        supplied_config = config or TargetMemoryConfig()
        if not isinstance(supplied_config, TargetMemoryConfig):
            raise TypeError("config must be TargetMemoryConfig")
        supplied_config.assert_integrity()
        episode_authority.assert_integrity()
        if not isinstance(_is_restore, bool) or not isinstance(_restore_finalized, bool):
            raise TypeError("_restore_finalized must be boolean")
        context_issuer.assert_issued_context(context)
        if not _restore_finalized:
            context_issuer.assert_context_matches_live_g3(context)
        if context.issuer_sha256 != context_issuer.issuer_sha256:
            raise TargetSnapshotBindingError("context issuer binding changed")
        if (
            episode_authority.context_issuer_contract_sha256
            != context_issuer.contract_sha256
        ):
            raise TargetSnapshotBindingError(
                "episode authority expected a different G3 issuer contract"
            )
        if context.frustum_contract_sha256 != context_issuer.frustum_contract_sha256:
            raise TargetSnapshotBindingError("context frustum contract changed")
        if context.physical_los_contract_sha256 != context_issuer.physical_los_contract_sha256:
            raise TargetSnapshotBindingError("context physical LOS contract changed")
        if (
            context.positive_evidence_producer_sha256
            != context_issuer.positive_evidence_producer_sha256
        ):
            raise TargetSnapshotBindingError("positive evidence producer changed")
        if (
            context.negative_visibility_producer_sha256
            != context_issuer.negative_visibility_producer_sha256
        ):
            raise TargetSnapshotBindingError("negative visibility producer changed")
        g3_frame = context_issuer._physical_memory.map_frame
        if (
            not math.isclose(
                supplied_config.cell_size_m,
                g3_frame.cell_size_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or supplied_config.origin_xy_m != g3_frame.origin_xy_m
        ):
            raise TargetSnapshotBindingError(
                "target-memory lattice differs from bound G3 map frame"
            )
        object_map = dict(episode_authority.task_object_by_target_id)
        if set(object_map) - set(supplied_config.target_ids):
            raise ValueError("episode task mapping contains an unregistered target")
        if len(context.candidate_domain) * supplied_config.posterior_mass_floor >= 1.0:
            raise ValueError("candidate domain is too large for the posterior floor")

        writer_capability = object()
        if not _is_restore:
            # Claim the single-writer lease before consuming the context so a
            # rejected fork leaves the independently issued context usable.
            context_issuer.claim_initial_writer(
                episode_authority_sha256=episode_authority.content_sha256,
                writer_instance=self,
                writer_capability=writer_capability,
            )
        context_issuer.assert_issued_context(context, consume=True)
        self._config = replace(supplied_config, target_ids=tuple(supplied_config.target_ids))
        self._context = self._clone_context(context)
        self._context_issuer = context_issuer
        self._issuer_contract = dict(context_issuer.contract_dict())
        self._issuer_contract_sha256 = _sha256(self._issuer_contract)
        self._episode_authority = self._clone_authority(episode_authority)
        self._task_object_by_target_id = object_map
        self._task_mapping_sha256 = _sha256(
            [
                {"target_id": target, "object_id": object_id}
                for target, object_id in sorted(object_map.items())
            ]
        )
        self._revision = 0
        self._current_tick = 0
        self._performance_counters: dict[str, int] = {
            "runtime_integrity_checks": 0,
            "writer_lease_checks": 0,
            "revision_checks": 0,
            "runtime_revision_commits": 0,
            "rolling_evidence_updates": 0,
            "component_cell_additions": 0,
            "component_root_merges": 0,
            "component_evidence_updates": 0,
            "component_index_rebuild_cells": 0,
            "hypothesis_component_reads": 0,
            "hypothesis_cell_reads": 0,
            "exhaustive_integrity_audits": 0,
            "evidence_chain_replay_transactions": 0,
            "posterior_replay_transactions": 0,
            "full_state_materializations": 0,
            "canonical_full_state_hashes": 0,
        }
        self._context_history: dict[str, TargetMemoryContext] = {
            self._context.content_sha256: self._clone_context(self._context)
        }
        self._authorized_context_history_sha256s = {
            self._context.content_sha256
        }
        self._cell_mass: dict[str, dict[Cell, float]] = {
            target_id: {} for target_id in self._config.target_ids
        }
        self._unlocalized_mass: dict[str, float] = {
            target_id: 1.0 for target_id in self._config.target_ids
        }
        self._positive: dict[str, PositiveTargetObservation] = {}
        self._negative: dict[str, NegativeTargetObservation] = {}
        self._evidence_transaction_order: list[tuple[str, str]] = []
        self._target_evidence_chain_sha256 = {
            target_id: self._initial_target_evidence_chain(target_id)
            for target_id in self._config.target_ids
        }
        self._positive_evidence_count = {
            target_id: 0 for target_id in self._config.target_ids
        }
        self._negative_evidence_count = {
            target_id: 0 for target_id in self._config.target_ids
        }
        self._reset_component_indexes()
        self._authorized_negative_certificate_sha256s: set[str] = set()
        self._seen_observation_ids: set[str] = set()
        self._seen_semantic_sha256s: set[str] = set()
        self._seen_payload_sha256s: set[str] = set()
        self._snapshot_capability = object()
        self._issued_snapshots: dict[int, TargetPosteriorSnapshot] = {}
        self._snapshot_history_sha256s: set[str] = set()
        self._controller_claim_attempts: dict[str, ControllerClaimAttempt] = {}
        self._controller_claim_attempt_order: list[str] = []
        self._physical_evaluations: list[PhysicalClaimEvaluationRecord] = []
        self._verified_claims: dict[str, VerifiedClaimCredit] = {}
        self._episode_finalized = False
        self._writer_capability = writer_capability
        if not _is_restore:
            self._commit_runtime_state()

    def __copy__(self) -> ReversibleTargetBeliefMemory:
        raise TypeError("reversible target-belief memory cannot be copied")

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> ReversibleTargetBeliefMemory:
        del memo
        raise TypeError("reversible target-belief memory cannot be copied")

    @staticmethod
    def _clone_context(context: TargetMemoryContext) -> TargetMemoryContext:
        return replace(context, candidate_domain=frozenset(context.candidate_domain))

    @staticmethod
    def _clone_authority(authority: TargetEpisodeAuthority) -> TargetEpisodeAuthority:
        return replace(
            authority,
            expected_task_object_ids=tuple(authority.expected_task_object_ids),
            task_object_by_target_id=tuple(authority.task_object_by_target_id),
        )

    @staticmethod
    def _clone_positive(observation: PositiveTargetObservation) -> PositiveTargetObservation:
        return replace(
            observation,
            identity=replace(observation.identity),
            localized_distribution=tuple(replace(row) for row in observation.localized_distribution),
        )

    @staticmethod
    def _clone_negative(observation: NegativeTargetObservation) -> NegativeTargetObservation:
        return replace(
            observation,
            identity=replace(observation.identity),
            visible_detection_probability=tuple(
                replace(row) for row in observation.visible_detection_probability
            ),
            visibility_certificate=replace(
                observation.visibility_certificate,
                certified_detection_probability=tuple(
                    observation.visibility_certificate.certified_detection_probability
                ),
            ),
        )

    def _initial_target_evidence_chain(self, target_id: str) -> str:
        return _sha256(
            {
                "schema": "lewm_g5_target_evidence_chain_seed_v1",
                "target_id": target_id,
                "episode_authority_sha256": self._episode_authority.content_sha256,
                "config_sha256": self._config.content_sha256,
            }
        )

    @staticmethod
    def _next_target_evidence_chain(
        previous_sha256: str,
        *,
        kind: str,
        observation: PositiveTargetObservation | NegativeTargetObservation,
    ) -> str:
        return _sha256(
            {
                "schema": "lewm_g5_target_evidence_chain_transition_v1",
                "previous_sha256": previous_sha256,
                "kind": kind,
                "target_id": observation.target_id,
                "observation_sha256": observation.content_sha256,
            }
        )

    def _advance_target_evidence_chain(
        self,
        kind: str,
        observation: PositiveTargetObservation | NegativeTargetObservation,
    ) -> None:
        target_id = observation.target_id
        self._target_evidence_chain_sha256[target_id] = (
            self._next_target_evidence_chain(
                self._target_evidence_chain_sha256[target_id],
                kind=kind,
                observation=observation,
            )
        )
        counts = (
            self._positive_evidence_count
            if kind == "positive"
            else self._negative_evidence_count
        )
        counts[target_id] += 1
        self._performance_counters["rolling_evidence_updates"] += 1

    def _derive_target_evidence_state(
        self,
    ) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
        chains = {
            target_id: self._initial_target_evidence_chain(target_id)
            for target_id in self._config.target_ids
        }
        positive_counts = {
            target_id: 0 for target_id in self._config.target_ids
        }
        negative_counts = {
            target_id: 0 for target_id in self._config.target_ids
        }
        for kind, observation_id in self._evidence_transaction_order:
            self._performance_counters[
                "evidence_chain_replay_transactions"
            ] += 1
            observation = (
                self._positive.get(observation_id)
                if kind == "positive"
                else self._negative.get(observation_id)
                if kind == "negative"
                else None
            )
            if observation is None:
                raise TargetSnapshotBindingError(
                    "evidence chain references an unknown transaction"
                )
            target_id = observation.target_id
            chains[target_id] = self._next_target_evidence_chain(
                chains[target_id],
                kind=kind,
                observation=observation,
            )
            (
                positive_counts if kind == "positive" else negative_counts
            )[target_id] += 1
        return chains, positive_counts, negative_counts

    def _reset_component_indexes(self) -> None:
        targets = self._config.target_ids
        if not hasattr(self, "_positive_ids_by_cell"):
            self._positive_ids_by_cell: dict[
                str, dict[Cell, set[str]]
            ] = {target_id: {} for target_id in targets}
            self._negative_ids_by_cell: dict[
                str, dict[Cell, set[str]]
            ] = {target_id: {} for target_id in targets}
            self._positive_evidence_metadata: dict[
                str, dict[str, tuple[str, int]]
            ] = {target_id: {} for target_id in targets}
        self._component_parent: dict[str, dict[Cell, Cell]] = {
            target_id: {} for target_id in targets
        }
        self._component_cells: dict[str, dict[Cell, set[Cell]]] = {
            target_id: {} for target_id in targets
        }
        self._component_positive_ids: dict[str, dict[Cell, set[str]]] = {
            target_id: {} for target_id in targets
        }
        self._component_negative_ids: dict[str, dict[Cell, set[str]]] = {
            target_id: {} for target_id in targets
        }
        self._component_diversity_keys: dict[str, dict[Cell, set[str]]] = {
            target_id: {} for target_id in targets
        }
        self._component_last_positive_tick: dict[
            str, dict[Cell, int | None]
        ] = {target_id: {} for target_id in targets}

    def _find_component_root(self, target_id: str, cell: Cell) -> Cell:
        parent = self._component_parent[target_id]
        root = cell
        while parent[root] != root:
            root = parent[root]
        while parent[cell] != cell:
            next_cell = parent[cell]
            parent[cell] = root
            cell = next_cell
        return root

    def _ensure_component_cell(self, target_id: str, cell: Cell) -> Cell:
        parent = self._component_parent[target_id]
        if cell not in parent:
            parent[cell] = cell
            self._component_cells[target_id][cell] = {cell}
            positive_ids = set(
                self._positive_ids_by_cell[target_id].get(cell, ())
            )
            self._component_positive_ids[target_id][cell] = positive_ids
            self._component_negative_ids[target_id][cell] = set(
                self._negative_ids_by_cell[target_id].get(cell, ())
            )
            metadata = self._positive_evidence_metadata[target_id]
            self._component_diversity_keys[target_id][cell] = {
                metadata[observation_id][0] for observation_id in positive_ids
            }
            self._component_last_positive_tick[target_id][cell] = max(
                (metadata[observation_id][1] for observation_id in positive_ids),
                default=None,
            )
            self._performance_counters["component_cell_additions"] += 1
        return self._find_component_root(target_id, cell)

    def _merge_component_roots(
        self,
        target_id: str,
        left: Cell,
        right: Cell,
    ) -> Cell:
        left_root = self._find_component_root(target_id, left)
        right_root = self._find_component_root(target_id, right)
        if left_root == right_root:
            return left_root
        cells = self._component_cells[target_id]
        if (
            len(cells[left_root]) < len(cells[right_root])
            or (
                len(cells[left_root]) == len(cells[right_root])
                and right_root < left_root
            )
        ):
            left_root, right_root = right_root, left_root
        self._component_parent[target_id][right_root] = left_root
        cells[left_root].update(cells.pop(right_root))
        self._component_positive_ids[target_id][left_root].update(
            self._component_positive_ids[target_id].pop(right_root)
        )
        self._component_negative_ids[target_id][left_root].update(
            self._component_negative_ids[target_id].pop(right_root)
        )
        self._component_diversity_keys[target_id][left_root].update(
            self._component_diversity_keys[target_id].pop(right_root)
        )
        left_tick = self._component_last_positive_tick[target_id][left_root]
        right_tick = self._component_last_positive_tick[target_id].pop(right_root)
        self._component_last_positive_tick[target_id][left_root] = max(
            (tick for tick in (left_tick, right_tick) if tick is not None),
            default=None,
        )
        self._performance_counters["component_root_merges"] += 1
        return left_root

    def _index_positive_observation(
        self,
        observation: PositiveTargetObservation,
    ) -> None:
        target_id = observation.target_id
        observation_id = observation.identity.observation_id
        self._positive_evidence_metadata[target_id][observation_id] = (
            observation.identity.diversity_key,
            observation.identity.tick,
        )
        for row in observation.localized_distribution:
            self._positive_ids_by_cell[target_id].setdefault(
                row.cell,
                set(),
            ).add(observation_id)
        mass_cells = self._cell_mass[target_id]
        touched = [
            row.cell
            for row in observation.localized_distribution
            if row.cell in mass_cells
        ]
        for cell in touched:
            self._ensure_component_cell(target_id, cell)
            x, y = cell
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in self._component_parent[target_id]:
                    self._merge_component_roots(target_id, cell, neighbor)
        roots = {
            self._find_component_root(target_id, cell) for cell in touched
        }
        for root in roots:
            self._component_positive_ids[target_id][root].add(
                observation_id
            )
            self._component_diversity_keys[target_id][root].add(
                observation.identity.diversity_key
            )
            previous = self._component_last_positive_tick[target_id][root]
            self._component_last_positive_tick[target_id][root] = max(
                observation.identity.tick,
                -1 if previous is None else previous,
            )
        self._performance_counters["component_evidence_updates"] += len(roots)

    def _index_negative_observation(
        self,
        observation: NegativeTargetObservation,
    ) -> None:
        target_id = observation.target_id
        observation_id = observation.identity.observation_id
        for row in observation.visible_detection_probability:
            self._negative_ids_by_cell[target_id].setdefault(
                row.cell,
                set(),
            ).add(observation_id)
        roots = {
            self._find_component_root(target_id, row.cell)
            for row in observation.visible_detection_probability
            if row.cell in self._component_parent[target_id]
        }
        for root in roots:
            self._component_negative_ids[target_id][root].add(
                observation_id
            )
        self._performance_counters["component_evidence_updates"] += len(roots)

    def _rebuild_evidence_cell_indexes(self) -> None:
        self._positive_ids_by_cell = {
            target_id: {} for target_id in self._config.target_ids
        }
        self._negative_ids_by_cell = {
            target_id: {} for target_id in self._config.target_ids
        }
        self._positive_evidence_metadata = {
            target_id: {} for target_id in self._config.target_ids
        }
        for observation in self._positive.values():
            target_id = observation.target_id
            observation_id = observation.identity.observation_id
            self._positive_evidence_metadata[target_id][observation_id] = (
                observation.identity.diversity_key,
                observation.identity.tick,
            )
            for row in observation.localized_distribution:
                self._positive_ids_by_cell[target_id].setdefault(
                    row.cell,
                    set(),
                ).add(observation_id)
        for observation in self._negative.values():
            target_id = observation.target_id
            observation_id = observation.identity.observation_id
            for row in observation.visible_detection_probability:
                self._negative_ids_by_cell[target_id].setdefault(
                    row.cell,
                    set(),
                ).add(observation_id)

    def _rebuild_component_indexes(self) -> None:
        self._reset_component_indexes()
        for target_id in self._config.target_ids:
            for cell in sorted(self._cell_mass[target_id]):
                self._performance_counters["component_index_rebuild_cells"] += 1
                self._ensure_component_cell(target_id, cell)
                x, y = cell
                for neighbor in ((x - 1, y), (x, y - 1)):
                    if neighbor in self._component_parent[target_id]:
                        self._merge_component_roots(target_id, cell, neighbor)

    def _assert_component_index_integrity(self) -> None:
        for target_id in self._config.target_ids:
            expected_positive_by_cell: dict[Cell, set[str]] = {}
            expected_negative_by_cell: dict[Cell, set[str]] = {}
            expected_positive_metadata: dict[str, tuple[str, int]] = {}
            for observation in self._positive.values():
                if observation.target_id != target_id:
                    continue
                observation_id = observation.identity.observation_id
                expected_positive_metadata[observation_id] = (
                    observation.identity.diversity_key,
                    observation.identity.tick,
                )
                for row in observation.localized_distribution:
                    expected_positive_by_cell.setdefault(row.cell, set()).add(
                        observation_id
                    )
            for observation in self._negative.values():
                if observation.target_id != target_id:
                    continue
                for row in observation.visible_detection_probability:
                    expected_negative_by_cell.setdefault(row.cell, set()).add(
                        observation.identity.observation_id
                    )
            if (
                self._positive_ids_by_cell[target_id]
                != expected_positive_by_cell
                or self._negative_ids_by_cell[target_id]
                != expected_negative_by_cell
                or self._positive_evidence_metadata[target_id]
                != expected_positive_metadata
            ):
                raise TargetSnapshotBindingError(
                    "component cell-evidence index changed"
                )
            mass_cells = set(self._cell_mass[target_id])
            if set(self._component_parent[target_id]) != mass_cells:
                raise TargetSnapshotBindingError(
                    "component index cell set changed"
                )
            root_keys = set(self._component_cells[target_id])
            if any(
                set(index[target_id]) != root_keys
                for index in (
                    self._component_positive_ids,
                    self._component_negative_ids,
                    self._component_diversity_keys,
                    self._component_last_positive_tick,
                )
            ):
                raise TargetSnapshotBindingError(
                    "component metadata root set changed"
                )
            actual_components = {
                frozenset(cells)
                for cells in self._component_cells[target_id].values()
            }
            actual_union = (
                set().union(*actual_components) if actual_components else set()
            )
            if actual_union != mass_cells:
                raise TargetSnapshotBindingError(
                    "component index coverage changed"
                )

            unseen = set(mass_cells)
            expected_components: set[frozenset[Cell]] = set()
            while unseen:
                start = min(unseen)
                reached = {start}
                unseen.remove(start)
                queue = deque([start])
                while queue:
                    x, y = queue.popleft()
                    for neighbor in (
                        (x - 1, y),
                        (x + 1, y),
                        (x, y - 1),
                        (x, y + 1),
                    ):
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            reached.add(neighbor)
                            queue.append(neighbor)
                expected_components.add(frozenset(reached))
            if actual_components != expected_components:
                raise TargetSnapshotBindingError(
                    "component index connectivity changed"
                )
            for component in expected_components:
                root = self._find_component_root(target_id, min(component))
                expected_positive_ids = {
                    observation.identity.observation_id
                    for observation in self._positive.values()
                    if observation.target_id == target_id
                    and any(
                        row.cell in component
                        for row in observation.localized_distribution
                    )
                }
                expected_negative_ids = {
                    observation.identity.observation_id
                    for observation in self._negative.values()
                    if observation.target_id == target_id
                    and any(
                        row.cell in component
                        for row in observation.visible_detection_probability
                    )
                }
                expected_diversity = {
                    self._positive[observation_id].identity.diversity_key
                    for observation_id in expected_positive_ids
                }
                expected_last_tick = max(
                    (
                        self._positive[observation_id].identity.tick
                        for observation_id in expected_positive_ids
                    ),
                    default=None,
                )
                if (
                    self._component_positive_ids[target_id][root]
                    != expected_positive_ids
                    or self._component_negative_ids[target_id][root]
                    != expected_negative_ids
                    or self._component_diversity_keys[target_id][root]
                    != expected_diversity
                    or self._component_last_positive_tick[target_id][root]
                    != expected_last_tick
                ):
                    raise TargetSnapshotBindingError(
                        "component evidence metadata changed"
                    )

    @property
    def config(self) -> TargetMemoryConfig:
        self._config.assert_integrity()
        return replace(
            self._config,
            target_ids=tuple(self._config.target_ids),
            origin_xy_m=tuple(self._config.origin_xy_m),
        )

    @property
    def context(self) -> TargetMemoryContext:
        self._context.assert_integrity()
        return self._clone_context(self._context)

    @property
    def episode_authority(self) -> TargetEpisodeAuthority:
        self._episode_authority.assert_integrity()
        return self._clone_authority(self._episode_authority)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def episode_finalized(self) -> bool:
        return self._episode_finalized

    @property
    def synthetic_only(self) -> bool:
        return True

    @property
    def production_authority_eligible(self) -> bool:
        return False

    @property
    def performance_counters(self) -> Mapping[str, int]:
        """Return non-authoritative work counters for scaling regression tests."""

        return dict(self._performance_counters)

    @property
    def controller_claim_attempts(self) -> tuple[ControllerClaimAttempt, ...]:
        self._assert_runtime_integrity()
        return tuple(
            replace(self._controller_claim_attempts[event_id])
            for event_id in self._controller_claim_attempt_order
        )

    @property
    def physical_claim_evaluations(self) -> tuple[PhysicalClaimEvaluationRecord, ...]:
        self._assert_foundation_integrity()
        return tuple(replace(record) for record in self._physical_evaluations)

    @property
    def verified_claims(self) -> Mapping[str, VerifiedClaimCredit]:
        self._assert_foundation_integrity()
        return {
            target_id: replace(credit)
            for target_id, credit in sorted(self._verified_claims.items())
        }

    def _commit_runtime_state(self) -> None:
        self._performance_counters["runtime_revision_commits"] += 1
        self._context_issuer._remember_memory_revision(
            episode_authority_sha256=self._episode_authority.content_sha256,
            revision=self._revision,
            writer_instance=self,
            writer_capability=self._writer_capability,
        )

    def _assert_mutable(self) -> None:
        if self._episode_finalized:
            raise TargetEvidenceRejectedError(
                "episode is finalized; evaluator state cannot feed controller state"
            )

    def _assert_runtime_integrity(
        self,
        *,
        require_live_context: bool | None = None,
        require_writer: bool = True,
    ) -> None:
        self._performance_counters["runtime_integrity_checks"] += 1
        self._config.assert_integrity()
        self._context.assert_integrity()
        self._episode_authority.assert_integrity()
        if require_writer:
            self._performance_counters["writer_lease_checks"] += 1
            self._context_issuer.assert_active_writer(
                episode_authority_sha256=self._episode_authority.content_sha256,
                writer_instance=self,
                writer_capability=self._writer_capability,
            )
        self._performance_counters["revision_checks"] += 1
        self._context_issuer.assert_latest_memory_revision(
            episode_authority_sha256=self._episode_authority.content_sha256,
            revision=self._revision,
        )
        live_required = (
            not self._episode_finalized
            if require_live_context is None
            else require_live_context
        )
        if live_required:
            self._context_issuer.assert_context_matches_live_g3(self._context)
        if (
            self._context_issuer.contract_dict() != self._issuer_contract
            or self._issuer_contract_sha256 != _sha256(self._issuer_contract)
        ):
            raise TargetSnapshotBindingError("context issuer contract was mutated")
        if (
            self._context.issuer_sha256 != self._context_issuer.issuer_sha256
            or self._context.frustum_contract_sha256
            != self._context_issuer.frustum_contract_sha256
            or self._context.physical_los_contract_sha256
            != self._context_issuer.physical_los_contract_sha256
            or self._context.positive_evidence_producer_sha256
            != self._context_issuer.positive_evidence_producer_sha256
            or self._context.negative_visibility_producer_sha256
            != self._context_issuer.negative_visibility_producer_sha256
        ):
            raise TargetSnapshotBindingError("memory context issuer changed")
        if (
            self._context.content_sha256 not in self._context_history
            or self._context.content_sha256
            not in self._authorized_context_history_sha256s
        ):
            raise TargetSnapshotBindingError(
                "current context is absent from authorized context history"
            )
        if dict(self._episode_authority.task_object_by_target_id) != (
            self._task_object_by_target_id
        ):
            raise TargetSnapshotBindingError("memory task mapping changed")
        if (
            self._episode_authority.context_issuer_contract_sha256
            != self._issuer_contract_sha256
        ):
            raise TargetSnapshotBindingError("episode G3 issuer binding changed")

    def _assert_foundation_integrity(
        self,
        *,
        require_live_context: bool | None = None,
        require_writer: bool = True,
    ) -> None:
        self._performance_counters["exhaustive_integrity_audits"] += 1
        self._assert_runtime_integrity(
            require_live_context=require_live_context,
            require_writer=require_writer,
        )
        if self._context.content_sha256 not in self._context_history:
            raise TargetSnapshotBindingError("current context is absent from context history")
        if set(self._context_history) != self._authorized_context_history_sha256s:
            raise TargetSnapshotBindingError("context-history authorization changed")
        ordered_contexts = sorted(
            self._context_history.values(),
            key=lambda row: row.context_sequence,
        )
        if len({row.context_sequence for row in ordered_contexts}) != len(ordered_contexts):
            raise TargetSnapshotBindingError("context history sequence is not unique")
        previous: TargetMemoryContext | None = None
        for historical in ordered_contexts:
            historical.assert_integrity()
            if (
                historical.issuer_sha256 != self._context.issuer_sha256
                or historical.map_frame_sha256 != self._context.map_frame_sha256
                or historical.morphology_sha256 != self._context.morphology_sha256
                or historical.camera_calibration_sha256
                != self._context.camera_calibration_sha256
                or historical.frustum_contract_sha256
                != self._context.frustum_contract_sha256
                or historical.physical_los_contract_sha256
                != self._context.physical_los_contract_sha256
                or historical.positive_evidence_producer_sha256
                != self._context.positive_evidence_producer_sha256
                or historical.negative_visibility_producer_sha256
                != self._context.negative_visibility_producer_sha256
                or historical.observation_model_checkpoint_sha256
                != self._context.observation_model_checkpoint_sha256
                or historical.exact_sim_tainted != self._context.exact_sim_tainted
                or historical.ablation_mode != self._context.ablation_mode
            ):
                raise TargetSnapshotBindingError("context history binding changed")
            transition_error = (
                None
                if previous is None
                else _context_transition_error(previous, historical)
            )
            if transition_error is not None:
                raise TargetSnapshotBindingError(
                    f"invalid context history: {transition_error}"
                )
            previous = historical
        if dict(self._episode_authority.task_object_by_target_id) != self._task_object_by_target_id:
            raise TargetSnapshotBindingError("memory task mapping changed")
        if (
            self._episode_authority.context_issuer_contract_sha256
            != self._issuer_contract_sha256
        ):
            raise TargetSnapshotBindingError("episode G3 issuer binding changed")
        if self._task_mapping_sha256 != _sha256(
            [
                {"target_id": target, "object_id": object_id}
                for target, object_id in sorted(self._task_object_by_target_id.items())
            ]
        ):
            raise TargetSnapshotBindingError("memory task mapping commitment changed")
        for observation in self._positive.values():
            observation.assert_integrity()
            self._context_issuer.assert_known_positive_observation_content(
                observation
            )
            historical = self._context_history.get(observation.context_sha256)
            if historical is None:
                raise TargetSnapshotBindingError("positive evidence context is missing")
            if {
                row.cell for row in observation.localized_distribution
            } - historical.candidate_domain:
                raise TargetSnapshotBindingError("positive evidence escaped its context")
            if (
                observation.identity.producer_sha256
                != historical.positive_evidence_producer_sha256
            ):
                raise TargetSnapshotBindingError(
                    "positive evidence producer binding changed"
                )
        for observation in self._negative.values():
            observation.assert_integrity()
            historical = self._context_history.get(observation.context_sha256)
            if historical is None:
                raise TargetSnapshotBindingError("negative evidence context is missing")
            certificate = observation.visibility_certificate
            if (
                certificate.context_sha256 != historical.content_sha256
                or certificate.physical_content_sha256
                != historical.physical_content_sha256
                or certificate.configuration_snapshot_sha256
                != historical.configuration_snapshot_sha256
                or certificate.pose_provenance_sha256
                != historical.pose_provenance_sha256
                or certificate.frustum_contract_sha256
                != historical.frustum_contract_sha256
                or certificate.physical_los_contract_sha256
                != historical.physical_los_contract_sha256
                or certificate.producer_sha256
                != historical.negative_visibility_producer_sha256
                or certificate.evidence_identity_sha256
                != _sha256(observation.identity.to_dict())
                or certificate.target_id != observation.target_id
                or certificate.confidence != observation.confidence
                or certificate.certified_detection_probability
                != tuple(
                    (row.cell, row.value)
                    for row in observation.visible_detection_probability
                )
                or certificate.certified_visible_cells - historical.candidate_domain
            ):
                raise TargetSnapshotBindingError("negative evidence certificate binding changed")
        if self._authorized_negative_certificate_sha256s != {
            row.visibility_certificate.content_sha256
            for row in self._negative.values()
        }:
            raise TargetSnapshotBindingError(
                "negative certificate authorization registry changed"
            )
        for attempt in self._controller_claim_attempts.values():
            attempt.assert_integrity()
            raw_event = json.loads(attempt.raw_event_json)
            expected_reference = {
                "namespace": "object_id",
                "value": attempt.expected_object_id,
            }
            if (
                raw_event.get("event_id") != attempt.event_id
                or raw_event.get("tick") != attempt.tick
                or _canonical_json_text(
                    raw_event.get("requested_target"),
                    "requested target",
                )
                != attempt.requested_target_json
                or _canonical_json_text(
                    raw_event.get("claimed_target"),
                    "claimed target",
                )
                != attempt.claimed_target_json
                or attempt.identity_matches_expected
                != (
                    raw_event.get("requested_target") == expected_reference
                    and raw_event.get("claimed_target") == expected_reference
                )
                or attempt.expected_object_id
                != self._task_object_by_target_id.get(attempt.target_id)
                or attempt.context_sha256 not in self._context_history
                or attempt.target_snapshot_sha256
                not in self._snapshot_history_sha256s
            ):
                raise TargetSnapshotBindingError("controller attempt binding changed")
        if (
            len(self._controller_claim_attempt_order)
            != len(self._controller_claim_attempts)
            or len(set(self._controller_claim_attempt_order))
            != len(self._controller_claim_attempt_order)
            or set(self._controller_claim_attempt_order)
            != set(self._controller_claim_attempts)
        ):
            raise TargetSnapshotBindingError("controller attempt order changed")
        ordered_attempts = [
            self._controller_claim_attempts[event_id]
            for event_id in self._controller_claim_attempt_order
        ]
        if [row.tick for row in ordered_attempts] != sorted(
            row.tick for row in ordered_attempts
        ):
            raise TargetSnapshotBindingError("controller attempt time rolls back")
        from lewm.benchmarks.go2_physical_claim_evaluator import (
            EVALUATOR_CONTRACT_SHA256,
        )

        attempt_by_sha256 = {
            attempt.content_sha256: attempt
            for attempt in self._controller_claim_attempts.values()
        }
        for record in self._physical_evaluations:
            record.assert_integrity()
            self._context_issuer.assert_known_evaluation_content(record)
            record_attempt = (
                None
                if record.controller_attempt_sha256 is None
                else attempt_by_sha256.get(record.controller_attempt_sha256)
            )
            if (
                record.episode_authority_sha256
                != self._episode_authority.content_sha256
                or record.task_object_set_sha256
                != self._episode_authority.task_object_set_sha256
                or (
                    record.controller_attempt_sha256 is not None
                    and (
                        record_attempt is None
                        or record_attempt.event_id != record.event_id
                        or record_attempt.target_id != record.target_id
                    )
                )
                or (record.controller_attempt_sha256 is None and record.status != "unverifiable")
                or (
                    record.evaluator_contract_sha256 is not None
                    and record.evaluator_contract_sha256
                    != EVALUATOR_CONTRACT_SHA256
                )
                or (
                    record.evaluated_trace_sha256 is not None
                    and (
                        record.evaluator_contract_sha256
                        != EVALUATOR_CONTRACT_SHA256
                        or record.supplied_manifest_sha256
                        != self._episode_authority.physical_manifest_sha256
                        or record.evaluation_event_sha256 is None
                        or record.evaluation_summary_sha256 is None
                        or record.evaluated_trace_sha256 is None
                    )
                )
            ):
                raise TargetSnapshotBindingError(
                    "physical evaluation authority binding changed"
                )
        canonically_evaluated_attempt_ids = {
            record.event_id
            for record in self._physical_evaluations
            if record.evaluated_trace_sha256 is not None
        }
        if canonically_evaluated_attempt_ids and (
            canonically_evaluated_attempt_ids
            != set(self._controller_claim_attempt_order)
        ):
            raise TargetSnapshotBindingError(
                "canonical evaluation omitted a controller claim attempt"
            )
        for credit in self._verified_claims.values():
            credit.assert_integrity()
            self._context_issuer.assert_known_credit_content(credit)
            attempt = self._controller_claim_attempts.get(credit.event_id)
            if (
                attempt is None
                or credit.controller_attempt_sha256 != attempt.content_sha256
                or credit.object_id != attempt.expected_object_id
                or credit.physical_manifest_sha256
                != self._episode_authority.physical_manifest_sha256
                or credit.task_object_set_sha256
                != self._episode_authority.task_object_set_sha256
                or credit.evaluator_contract_sha256
                != EVALUATOR_CONTRACT_SHA256
                or not any(
                    record.event_id == credit.event_id
                    and record.verified_credit_created
                    and record.evaluation_event_sha256
                    == credit.evaluation_event_sha256
                    for record in self._physical_evaluations
                )
            ):
                raise TargetSnapshotBindingError("verified credit binding changed")
        created_records = [
            record
            for record in self._physical_evaluations
            if record.verified_credit_created
        ]
        if len(created_records) != len(self._verified_claims):
            raise TargetSnapshotBindingError(
                "verified-credit ledger completeness changed"
            )
        for record in created_records:
            matching_credits = [
                credit
                for credit in self._verified_claims.values()
                if credit.event_id == record.event_id
                and credit.target_id == record.target_id
                and credit.evaluation_event_sha256
                == record.evaluation_event_sha256
                and credit.evaluation_summary_sha256
                == record.evaluation_summary_sha256
                and credit.evaluated_trace_sha256
                == record.evaluated_trace_sha256
            ]
            if len(matching_credits) != 1:
                raise TargetSnapshotBindingError(
                    "verified evaluation lacks exactly one credit"
                )
        self._context_issuer.assert_complete_evaluation_ledger(
            episode_authority_sha256=self._episode_authority.content_sha256,
            evaluations=self._physical_evaluations,
            credits=list(self._verified_claims.values()),
        )
        expected_ids = set(self._positive) | set(self._negative)
        expected_semantics = {
            row.semantic_sha256 for row in self._positive.values()
        } | {row.semantic_sha256 for row in self._negative.values()}
        expected_payloads = {
            row.identity.payload_sha256 for row in self._positive.values()
        } | {row.identity.payload_sha256 for row in self._negative.values()}
        if (
            expected_ids != self._seen_observation_ids
            or expected_semantics != self._seen_semantic_sha256s
            or expected_payloads != self._seen_payload_sha256s
        ):
            raise TargetSnapshotBindingError("evidence duplicate registry changed")
        (
            expected_chains,
            expected_positive_counts,
            expected_negative_counts,
        ) = self._derive_target_evidence_state()
        if (
            self._target_evidence_chain_sha256 != expected_chains
            or self._positive_evidence_count != expected_positive_counts
            or self._negative_evidence_count != expected_negative_counts
        ):
            raise TargetSnapshotBindingError(
                "runtime target evidence commitment changed"
            )
        self._assert_component_index_integrity()
        replay_mass, replay_unlocalized = self._replay_posterior()
        if (
            replay_mass != self._cell_mass
            or replay_unlocalized != self._unlocalized_mass
        ):
            raise TargetSnapshotBindingError(
                "posterior does not match causal evidence replay"
            )
        if self._physical_evaluations and not self._episode_finalized:
            raise TargetSnapshotBindingError("evaluation ledger exists before finalization")
        expected_revision = (
            len(self._positive)
            + len(self._negative)
            + len(self._context_history)
            - 1
            + len(self._controller_claim_attempts)
            + len(self._physical_evaluations)
            + int(self._episode_finalized)
        )
        if self._revision != expected_revision:
            raise TargetSnapshotBindingError("target-memory revision history changed")
        expected_tick = max(
            [0]
            + [row.identity.tick for row in self._positive.values()]
            + [row.identity.tick for row in self._negative.values()]
            + [row.tick for row in self._controller_claim_attempts.values()]
        )
        if self._current_tick != expected_tick:
            raise TargetSnapshotBindingError("target-memory current tick changed")
        for target_id in self._config.target_ids:
            if target_id not in self._cell_mass or target_id not in self._unlocalized_mass:
                raise TargetSnapshotBindingError("posterior target set changed")
            values = self._cell_mass[target_id]
            if set(values) - self._context.candidate_domain:
                raise TargetSnapshotBindingError("posterior escaped candidate domain")
            if any(
                not math.isfinite(value)
                or value < self._config.posterior_mass_floor
                for value in values.values()
            ):
                raise TargetSnapshotBindingError("posterior floor was violated")
            total = math.fsum(values.values()) + self._unlocalized_mass[target_id]
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise TargetSnapshotBindingError("posterior is not normalized")

    def _validate_common(
        self,
        observation: PositiveTargetObservation | NegativeTargetObservation,
    ) -> None:
        self._assert_mutable()
        self._assert_runtime_integrity()
        observation.assert_integrity()
        if observation.context_sha256 != self._context.content_sha256:
            raise TargetEvidenceRejectedError("target observation context is stale")
        if observation.target_id not in self._cell_mass:
            raise TargetEvidenceRejectedError("target observation identity is not registered")
        identity = observation.identity
        if identity.observation_id in self._seen_observation_ids:
            raise TargetEvidenceRejectedError("duplicate target observation id")
        if identity.payload_sha256 in self._seen_payload_sha256s:
            raise TargetEvidenceRejectedError("duplicate target observation payload")
        if observation.semantic_sha256 in self._seen_semantic_sha256s:
            raise TargetEvidenceRejectedError("semantic duplicate target observation")
        if identity.tick < self._current_tick:
            raise TargetEvidenceRejectedError("target evidence cannot move backward in time")
        rows = (
            observation.localized_distribution
            if isinstance(observation, PositiveTargetObservation)
            else observation.visible_detection_probability
        )
        if {row.cell for row in rows} - self._context.candidate_domain:
            raise TargetEvidenceRejectedError(
                "target evidence contains cells outside the registered domain"
            )
        if isinstance(observation, PositiveTargetObservation):
            self._context_issuer.assert_issued_positive_observation(observation)
            if (
                observation.identity.producer_sha256
                != self._context.positive_evidence_producer_sha256
            ):
                raise TargetEvidenceRejectedError(
                    "positive observation producer is not registered"
                )
        else:
            certificate = observation.visibility_certificate
            self._context_issuer.assert_issued_certificate(certificate)
            expected_bindings = (
                certificate.context_sha256 == self._context.content_sha256
                and certificate.physical_content_sha256
                == self._context.physical_content_sha256
                and certificate.configuration_snapshot_sha256
                == self._context.configuration_snapshot_sha256
                and certificate.pose_provenance_sha256
                == self._context.pose_provenance_sha256
                and certificate.frustum_contract_sha256
                == self._context.frustum_contract_sha256
                and certificate.physical_los_contract_sha256
                == self._context.physical_los_contract_sha256
                and certificate.producer_sha256
                == self._context.negative_visibility_producer_sha256
                and identity.producer_sha256 == certificate.producer_sha256
                and certificate.evidence_identity_sha256
                == _sha256(identity.to_dict())
                and certificate.target_id == observation.target_id
                and certificate.confidence == observation.confidence
                and certificate.certified_detection_probability
                == tuple((row.cell, row.value) for row in rows)
            )
            if not expected_bindings:
                raise TargetEvidenceRejectedError(
                    "negative visibility certificate is stale or misbound"
                )

    def _commit_identity(
        self,
        observation: PositiveTargetObservation | NegativeTargetObservation,
    ) -> None:
        self._seen_observation_ids.add(observation.identity.observation_id)
        self._seen_semantic_sha256s.add(observation.semantic_sha256)
        self._seen_payload_sha256s.add(observation.identity.payload_sha256)
        self._current_tick = observation.identity.tick
        self._revision += 1

    def _normalize_with_floor(
        self,
        mass: Mapping[Cell, float],
        unlocalized: float,
    ) -> tuple[dict[Cell, float], float]:
        clean: dict[Cell, float] = {}
        for cell, raw in mass.items():
            value = _finite(raw, "posterior cell mass")
            if value <= 0.0:
                raise TargetEvidenceRejectedError("posterior cell mass became nonpositive")
            clean[cell] = value
        unlocalized_value = _finite(unlocalized, "unlocalized mass")
        if unlocalized_value < 0.0:
            raise TargetEvidenceRejectedError("unlocalized mass became negative")
        total = math.fsum(clean.values()) + unlocalized_value
        if not math.isfinite(total) or total <= 0.0:
            raise TargetEvidenceRejectedError("target posterior normalization failed")
        normalized = {cell: value / total for cell, value in clean.items()}
        normalized_unlocalized = unlocalized_value / total
        floor = self._config.posterior_mass_floor
        deficits = {
            cell: floor - value
            for cell, value in normalized.items()
            if value < floor
        }
        deficit = math.fsum(deficits.values())
        for cell in deficits:
            normalized[cell] = floor
        if deficit > 0.0:
            take = min(deficit, normalized_unlocalized)
            normalized_unlocalized -= take
            deficit -= take
            if deficit > 0.0:
                donors = {
                    cell: value - floor
                    for cell, value in normalized.items()
                    if value > floor
                }
                available = math.fsum(donors.values())
                if available + 1e-18 < deficit:
                    raise TargetEvidenceRejectedError("posterior floor is infeasible")
                scale = deficit / available
                for cell, room in donors.items():
                    normalized[cell] -= room * scale
        correction = 1.0 - (math.fsum(normalized.values()) + normalized_unlocalized)
        if normalized_unlocalized + correction >= 0.0:
            normalized_unlocalized += correction
        elif normalized:
            donor = max(normalized, key=normalized.get)
            normalized[donor] += correction
        else:
            raise TargetEvidenceRejectedError("posterior correction failed")
        if any(value < floor or not math.isfinite(value) for value in normalized.values()):
            raise TargetEvidenceRejectedError("posterior floor enforcement failed")
        return normalized, normalized_unlocalized

    def _positive_transition(
        self,
        prior_mass: Mapping[Cell, float],
        prior_unlocalized: float,
        observation: PositiveTargetObservation,
    ) -> tuple[dict[Cell, float], float]:
        localized_probability = math.fsum(
            row.value for row in observation.localized_distribution
        )
        transfer = min(
            prior_unlocalized,
            self._config.maximum_positive_transfer
            * observation.confidence
            * localized_probability,
        )
        next_mass = dict(prior_mass)
        applied_additions: dict[Cell, float] = {}
        if transfer > 0.0:
            for row in observation.localized_distribution:
                addition = transfer * row.value / localized_probability
                if (
                    row.cell not in next_mass
                    and addition < self._config.posterior_mass_floor
                ):
                    continue
                next_mass[row.cell] = next_mass.get(row.cell, 0.0) + addition
                applied_additions[row.cell] = addition
        applied_transfer = math.fsum(applied_additions.values())
        next_unlocalized = prior_unlocalized - applied_transfer
        total = math.fsum(next_mass.values()) + next_unlocalized
        correction = 1.0 - total
        if abs(correction) > 1e-12:
            raise TargetEvidenceRejectedError("positive transfer failed normalization")
        if next_unlocalized + correction >= 0.0:
            next_unlocalized += correction
        elif applied_additions:
            donor = max(applied_additions, key=applied_additions.get)
            if next_mass[donor] + correction < prior_mass.get(
                donor,
                self._config.posterior_mass_floor,
            ):
                raise TargetEvidenceRejectedError("positive transfer correction failed")
            next_mass[donor] += correction
        else:
            raise TargetEvidenceRejectedError("positive transfer correction failed")
        if any(
            next_mass[cell] < prior_value
            for cell, prior_value in prior_mass.items()
        ):
            raise TargetEvidenceRejectedError("positive update decayed an existing mode")
        return next_mass, next_unlocalized

    def _negative_transition(
        self,
        prior_mass: Mapping[Cell, float],
        prior_unlocalized: float,
        observation: NegativeTargetObservation,
    ) -> tuple[dict[Cell, float], float]:
        next_mass = dict(prior_mass)
        floor = self._config.posterior_mass_floor
        transferred_to_unlocalized = 0.0
        for row in observation.visible_detection_probability:
            if row.cell not in next_mass:
                continue
            multiplier = max(
                self._config.negative_mass_floor_multiplier,
                1.0 - observation.confidence * row.value,
            )
            previous = next_mass[row.cell]
            updated = max(previous * multiplier, floor)
            next_mass[row.cell] = updated
            transferred_to_unlocalized += previous - updated
        next_unlocalized = prior_unlocalized + transferred_to_unlocalized
        total = math.fsum(next_mass.values()) + next_unlocalized
        correction = 1.0 - total
        if abs(correction) > 1e-12 or next_unlocalized + correction < 0.0:
            raise TargetEvidenceRejectedError("negative transfer failed normalization")
        next_unlocalized += correction
        return next_mass, next_unlocalized

    def _replay_posterior(
        self,
    ) -> tuple[dict[str, dict[Cell, float]], dict[str, float]]:
        expected_entries = {
            *(('positive', observation_id) for observation_id in self._positive),
            *(('negative', observation_id) for observation_id in self._negative),
        }
        if (
            len(self._evidence_transaction_order) != len(expected_entries)
            or len(set(self._evidence_transaction_order))
            != len(self._evidence_transaction_order)
            or set(self._evidence_transaction_order) != expected_entries
        ):
            raise TargetSnapshotBindingError("evidence transaction order changed")
        contexts = sorted(
            self._context_history.values(),
            key=lambda row: row.context_sequence,
        )
        context_index = {
            context.content_sha256: index for index, context in enumerate(contexts)
        }
        ordered_observations: list[
            PositiveTargetObservation | NegativeTargetObservation
        ] = []
        for kind, observation_id in self._evidence_transaction_order:
            self._performance_counters["posterior_replay_transactions"] += 1
            observation = (
                self._positive.get(observation_id)
                if kind == "positive"
                else self._negative.get(observation_id)
                if kind == "negative"
                else None
            )
            if observation is None:
                raise TargetSnapshotBindingError("evidence transaction kind/id changed")
            ordered_observations.append(observation)
        context_positions = [
            context_index.get(observation.context_sha256, -1)
            for observation in ordered_observations
        ]
        if (
            any(position < 0 for position in context_positions)
            or context_positions != sorted(context_positions)
            or [row.identity.tick for row in ordered_observations]
            != sorted(row.identity.tick for row in ordered_observations)
        ):
            raise TargetSnapshotBindingError("evidence transaction causality changed")
        observations_by_context: dict[
            str,
            list[PositiveTargetObservation | NegativeTargetObservation],
        ] = {context.content_sha256: [] for context in contexts}
        for observation in ordered_observations:
            observations_by_context[observation.context_sha256].append(observation)

        replay_mass: dict[str, dict[Cell, float]] = {
            target: {} for target in self._config.target_ids
        }
        replay_unlocalized = {
            target: 1.0 for target in self._config.target_ids
        }
        for context_index_value, context in enumerate(contexts):
            if context_index_value > 0:
                for target in self._config.target_ids:
                    retained = {
                        cell: value
                        for cell, value in replay_mass[target].items()
                        if cell in context.candidate_domain
                    }
                    removed = math.fsum(
                        value
                        for cell, value in replay_mass[target].items()
                        if cell not in context.candidate_domain
                    )
                    retained, unlocalized = self._normalize_with_floor(
                        retained,
                        replay_unlocalized[target] + removed,
                    )
                    replay_mass[target] = retained
                    replay_unlocalized[target] = unlocalized
            for observation in observations_by_context[context.content_sha256]:
                target = observation.target_id
                if isinstance(observation, PositiveTargetObservation):
                    mass, unlocalized = self._positive_transition(
                        replay_mass[target],
                        replay_unlocalized[target],
                        observation,
                    )
                else:
                    mass, unlocalized = self._negative_transition(
                        replay_mass[target],
                        replay_unlocalized[target],
                        observation,
                    )
                replay_mass[target] = mass
                replay_unlocalized[target] = unlocalized
        return replay_mass, replay_unlocalized

    def apply_positive(self, observation: PositiveTargetObservation) -> TargetPosteriorSnapshot:
        if not isinstance(observation, PositiveTargetObservation):
            raise TypeError("observation must be PositiveTargetObservation")
        self._validate_common(observation)
        stored = self._clone_positive(observation)
        target = stored.target_id
        next_mass, next_unlocalized = self._positive_transition(
            self._cell_mass[target],
            self._unlocalized_mass[target],
            stored,
        )
        self._context_issuer.assert_issued_positive_observation(
            observation,
            consume=True,
        )
        self._cell_mass[target] = next_mass
        self._unlocalized_mass[target] = next_unlocalized
        self._positive[stored.identity.observation_id] = stored
        self._evidence_transaction_order.append(
            ("positive", stored.identity.observation_id)
        )
        self._index_positive_observation(stored)
        self._advance_target_evidence_chain("positive", stored)
        self._commit_identity(stored)
        self._commit_runtime_state()
        return self.snapshot(target)

    def apply_negative(self, observation: NegativeTargetObservation) -> TargetPosteriorSnapshot:
        if not isinstance(observation, NegativeTargetObservation):
            raise TypeError("observation must be NegativeTargetObservation")
        self._validate_common(observation)
        stored = self._clone_negative(observation)
        target = stored.target_id
        next_mass, next_unlocalized = self._negative_transition(
            self._cell_mass[target],
            self._unlocalized_mass[target],
            stored,
        )
        # Consuming the proof is the final fallible operation before state commit.
        self._context_issuer.assert_issued_certificate(
            observation.visibility_certificate,
            consume=True,
        )
        self._cell_mass[target] = next_mass
        self._unlocalized_mass[target] = next_unlocalized
        self._negative[stored.identity.observation_id] = stored
        self._authorized_negative_certificate_sha256s.add(
            stored.visibility_certificate.content_sha256
        )
        self._evidence_transaction_order.append(
            ("negative", stored.identity.observation_id)
        )
        self._index_negative_observation(stored)
        self._advance_target_evidence_chain("negative", stored)
        self._commit_identity(stored)
        self._commit_runtime_state()
        return self.snapshot(target)

    def advance_context(self, context: TargetMemoryContext) -> None:
        """Atomically advance physical state or pose provenance without rollback."""

        if not isinstance(context, TargetMemoryContext):
            raise TypeError("context must be TargetMemoryContext")
        self._assert_mutable()
        self._assert_runtime_integrity(require_live_context=False)
        self._context_issuer.assert_issued_context(context)
        self._context_issuer.assert_context_matches_live_g3(context)
        current = self._context
        transition_error = _context_transition_error(current, context)
        if transition_error is not None:
            raise TargetEvidenceRejectedError(transition_error)
        if len(context.candidate_domain) * self._config.posterior_mass_floor >= 1.0:
            raise TargetEvidenceRejectedError("candidate domain exceeds posterior floor capacity")

        next_cell_mass: dict[str, dict[Cell, float]] = {}
        next_unlocalized: dict[str, float] = {}
        for target in self._config.target_ids:
            retained = {
                cell: value
                for cell, value in self._cell_mass[target].items()
                if cell in context.candidate_domain
            }
            removed = math.fsum(
                value
                for cell, value in self._cell_mass[target].items()
                if cell not in context.candidate_domain
            )
            retained, unlocalized = self._normalize_with_floor(
                retained,
                self._unlocalized_mass[target] + removed,
            )
            next_cell_mass[target] = retained
            next_unlocalized[target] = unlocalized
        self._context_issuer.assert_issued_context(context, consume=True)
        self._cell_mass = next_cell_mass
        self._unlocalized_mass = next_unlocalized
        self._context = self._clone_context(context)
        self._context_history[context.content_sha256] = self._clone_context(context)
        self._authorized_context_history_sha256s.add(context.content_sha256)
        self._rebuild_component_indexes()
        self._revision += 1
        self._commit_runtime_state()

    def _make_snapshot(
        self,
        target_id: str,
        capability: object,
    ) -> TargetPosteriorSnapshot:
        return TargetPosteriorSnapshot(
            target_id=target_id,
            context_sha256=self._context.content_sha256,
            config_sha256=self._config.content_sha256,
            issuer_contract_sha256=self._issuer_contract_sha256,
            episode_authority_sha256=self._episode_authority.content_sha256,
            physical_manifest_sha256=self._episode_authority.physical_manifest_sha256,
            task_object_set_sha256=self._episode_authority.task_object_set_sha256,
            task_mapping_sha256=self._task_mapping_sha256,
            exact_sim_tainted=self._context.exact_sim_tainted,
            ablation_mode=self._context.ablation_mode,
            target_memory_revision=self._revision,
            current_tick=self._current_tick,
            cell_mass=tuple(
                TargetCellValue(cell=cell, value=value)
                for cell, value in sorted(self._cell_mass[target_id].items())
            ),
            unlocalized_mass=self._unlocalized_mass[target_id],
            evidence_chain_sha256=self._target_evidence_chain_sha256[target_id],
            positive_evidence_count=self._positive_evidence_count[target_id],
            negative_evidence_count=self._negative_evidence_count[target_id],
            _issuance_capability=capability,
        )

    def snapshot(self, target_id: str) -> TargetPosteriorSnapshot:
        self._assert_runtime_integrity()
        if target_id not in self._cell_mass:
            raise KeyError(target_id)
        result = self._make_snapshot(target_id, self._snapshot_capability)
        self._issued_snapshots[id(result)] = result
        self._snapshot_history_sha256s.add(result.content_sha256)
        return result

    def assert_current_snapshot(self, snapshot: TargetPosteriorSnapshot) -> None:
        if not isinstance(snapshot, TargetPosteriorSnapshot):
            raise TypeError("snapshot must be TargetPosteriorSnapshot")
        self._assert_runtime_integrity()
        snapshot.assert_integrity()
        if (
            snapshot._issuance_capability is not self._snapshot_capability
            or self._issued_snapshots.get(id(snapshot)) is not snapshot
        ):
            raise TargetSnapshotBindingError(
                "target snapshot was not issued by this memory instance"
            )
        expected = self._make_snapshot(snapshot.target_id, self._snapshot_capability)
        if expected.content_sha256 != snapshot.content_sha256:
            raise TargetSnapshotBindingError("target snapshot is stale or misbound")

    def hypotheses(self, snapshot: TargetPosteriorSnapshot) -> tuple[TargetHypothesis, ...]:
        self.assert_current_snapshot(snapshot)
        # Posterior sparsification already applies the per-cell floor. Component
        # thresholding belongs after connectivity is known, otherwise a broad
        # high-mass mode made of individually small cells disappears.
        mass = {row.cell: row.value for row in snapshot.cell_mass}
        components = [
            (root, cells)
            for root, cells in self._component_cells[snapshot.target_id].items()
        ]
        result: list[TargetHypothesis] = []
        for root, component in components:
            self._performance_counters["hypothesis_component_reads"] += 1
            self._performance_counters["hypothesis_cell_reads"] += len(component)
            component_mass = math.fsum(mass[cell] for cell in component)
            if component_mass < self._config.component_mass_floor:
                continue
            weighted_xy: list[tuple[tuple[float, float], float]] = []
            for cell in sorted(component):
                xy = (
                    self._config.origin_xy_m[0]
                    + (cell[0] + 0.5) * self._config.cell_size_m,
                    self._config.origin_xy_m[1]
                    + (cell[1] + 0.5) * self._config.cell_size_m,
                )
                weighted_xy.append((xy, mass[cell]))
            mean_x = math.fsum(xy[0] * value for xy, value in weighted_xy) / component_mass
            mean_y = math.fsum(xy[1] * value for xy, value in weighted_xy) / component_mass
            cov_xx = math.fsum(
                value * (xy[0] - mean_x) ** 2 for xy, value in weighted_xy
            ) / component_mass
            cov_xy = math.fsum(
                value * (xy[0] - mean_x) * (xy[1] - mean_y)
                for xy, value in weighted_xy
            ) / component_mass
            cov_yy = math.fsum(
                value * (xy[1] - mean_y) ** 2 for xy, value in weighted_xy
            ) / component_mass
            positive_ids = self._component_positive_ids[snapshot.target_id][root]
            negative_ids = self._component_negative_ids[snapshot.target_id][root]
            last_tick = self._component_last_positive_tick[snapshot.target_id][root]
            result.append(
                TargetHypothesis(
                    target_id=snapshot.target_id,
                    cells=frozenset(component),
                    mass=component_mass,
                    mean_xy_m=(mean_x, mean_y),
                    covariance_xy_m2=(
                        (cov_xx + self._config.covariance_floor_m2, cov_xy),
                        (cov_xy, cov_yy + self._config.covariance_floor_m2),
                    ),
                    positive_evidence_count=len(positive_ids),
                    negative_evidence_count=len(negative_ids),
                    evidence_diversity=len(
                        self._component_diversity_keys[snapshot.target_id][root]
                    ),
                    last_positive_tick=last_tick,
                    age_ticks=(
                        None if last_tick is None else self._current_tick - last_tick
                    ),
                )
            )
        result.sort(key=lambda row: (-row.mass, min(row.cells), row.target_id))
        return tuple(result)

    def record_controller_claim_attempt(
        self,
        *,
        target_id: str,
        snapshot: TargetPosteriorSnapshot,
        raw_event: Mapping[str, object],
    ) -> ControllerClaimAttempt:
        """Record controller intent, including a wrong requested/claimed identity."""

        self._assert_mutable()
        self.assert_current_snapshot(snapshot)
        if target_id != snapshot.target_id:
            raise TargetClaimVerificationError("claim target and posterior target differ")
        expected_object_id = self._task_object_by_target_id.get(target_id)
        if expected_object_id is None:
            raise TargetClaimVerificationError("target has no registered task object")
        if not isinstance(raw_event, Mapping):
            raise TypeError("raw_event must be a mapping")
        try:
            event_json = _canonical_json_text(dict(raw_event), "raw claim event")
            event = json.loads(event_json)
        except (ValueError, json.JSONDecodeError) as exc:
            raise TargetClaimVerificationError(
                "raw claim event is not canonical JSON data"
            ) from exc
        if not isinstance(event, dict):
            raise TargetClaimVerificationError("raw claim event must be an object")
        event_id = event.get("event_id")
        tick = event.get("tick")
        if type(event_id) is not str or not event_id:
            raise TargetClaimVerificationError("raw claim event_id is invalid")
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            raise TargetClaimVerificationError("raw claim tick is invalid")
        if tick < self._current_tick:
            raise TargetClaimVerificationError("raw claim tick moves backward")
        if event_id in self._controller_claim_attempts:
            raise TargetClaimVerificationError("duplicate controller claim event_id")
        expected_reference = {"namespace": "object_id", "value": expected_object_id}
        requested = event.get("requested_target")
        claimed = event.get("claimed_target")
        attempt = ControllerClaimAttempt(
            target_id=target_id,
            expected_object_id=expected_object_id,
            event_id=event_id,
            tick=tick,
            target_snapshot_sha256=snapshot.content_sha256,
            context_sha256=self._context.content_sha256,
            requested_target_json=_canonical_json_text(requested, "requested_target"),
            claimed_target_json=_canonical_json_text(claimed, "claimed_target"),
            identity_matches_expected=(
                requested == expected_reference and claimed == expected_reference
            ),
            raw_event_json=event_json,
            raw_event_sha256=_sha256(event),
        )
        self._controller_claim_attempts[event_id] = attempt
        self._controller_claim_attempt_order.append(event_id)
        self._current_tick = tick
        self._revision += 1
        self._commit_runtime_state()
        return replace(attempt)

    def finalize_episode_for_evaluation(self) -> None:
        """Close controller mutation before the canonical evaluator can run."""

        self._assert_runtime_integrity()
        if not self._episode_finalized:
            self._episode_finalized = True
            self._revision += 1
            self._commit_runtime_state()

    def _append_evaluation(
        self,
        *,
        target_id: str,
        event_id: str,
        status: str,
        accepted: bool,
        canonical_credited: bool,
        verified_credit_created: bool,
        reason: str,
        attempt: ControllerClaimAttempt | None,
        raw_trace_sha256: str,
        supplied_manifest_sha256: str | None,
        evaluator_contract_sha256: str | None = None,
        evaluation_event_sha256: str | None = None,
        evaluation_summary_sha256: str | None = None,
        evaluated_trace_sha256: str | None = None,
        defer_state_commit: bool = False,
    ) -> PhysicalClaimEvaluationRecord:
        record = PhysicalClaimEvaluationRecord(
            evaluation_sequence=len(self._physical_evaluations) + 1,
            episode_authority_sha256=self._episode_authority.content_sha256,
            target_id=target_id,
            event_id=event_id,
            status=status,
            accepted=accepted,
            canonical_credited=canonical_credited,
            verified_credit_created=verified_credit_created,
            reason=reason,
            controller_attempt_sha256=(
                None if attempt is None else attempt.content_sha256
            ),
            evaluator_contract_sha256=evaluator_contract_sha256,
            supplied_manifest_sha256=supplied_manifest_sha256,
            task_object_set_sha256=self._episode_authority.task_object_set_sha256,
            evaluation_event_sha256=evaluation_event_sha256,
            evaluation_summary_sha256=evaluation_summary_sha256,
            evaluated_trace_sha256=evaluated_trace_sha256,
            raw_trace_sha256=raw_trace_sha256,
        )
        self._context_issuer._remember_canonical_evaluation(
            record,
            episode_authority_sha256=self._episode_authority.content_sha256,
        )
        self._physical_evaluations.append(record)
        self._revision += 1
        if not defer_state_commit:
            self._commit_runtime_state()
        return record

    def _record_unverifiable_and_raise(
        self,
        *,
        target_id: str,
        event_id: str,
        reason: str,
        attempt: ControllerClaimAttempt | None,
        raw_trace_sha256: str,
        supplied_manifest_sha256: str | None,
    ) -> None:
        self._append_evaluation(
            target_id=target_id,
            event_id=event_id,
            status="unverifiable",
            accepted=False,
            canonical_credited=False,
            verified_credit_created=False,
            reason=reason,
            attempt=attempt,
            raw_trace_sha256=raw_trace_sha256,
            supplied_manifest_sha256=supplied_manifest_sha256,
        )
        raise TargetClaimVerificationError(reason)

    def evaluate_and_record_verified_claim(
        self,
        *,
        target_id: str,
        event_id: str,
        raw_trace: Mapping[str, object],
        physical_manifest: object,
    ) -> VerifiedClaimCredit:
        """Observe a closed episode and credit only its first accepted event."""

        from lewm.benchmarks.go2_physical_claim_evaluator import (
            EVALUATED_TRACE_SCHEMA,
            EVALUATOR_CONTRACT_SHA256,
            EVENT_SCHEMA,
            SUMMARY_SCHEMA,
            evaluate_physical_claim_trace,
        )
        from lewm_worlds.manifest import manifest_sha256

        self._assert_foundation_integrity()
        if not self._episode_finalized:
            raise TargetClaimVerificationError(
                "episode must be finalized before observer evaluation"
            )
        valid_target_id = type(target_id) is str and bool(target_id)
        valid_event_id = type(event_id) is str and bool(event_id)
        ledger_target_id = target_id if valid_target_id else "<invalid-target-id>"
        ledger_event_id = event_id if valid_event_id else "<invalid-event-id>"
        fallback_raw_trace_sha256 = _sha256(
            {
                "schema": "lewm_g5_uncanonical_trace_commitment_v1",
            }
        )
        if not valid_target_id or not valid_event_id:
            reason = (
                "target_id must be nonempty"
                if not valid_target_id
                else "event_id must be nonempty"
            )
            self._append_evaluation(
                target_id=ledger_target_id,
                event_id=ledger_event_id,
                status="unverifiable",
                accepted=False,
                canonical_credited=False,
                verified_credit_created=False,
                reason=reason,
                attempt=None,
                raw_trace_sha256=fallback_raw_trace_sha256,
                supplied_manifest_sha256=None,
            )
            raise TargetClaimVerificationError(reason)
        if not isinstance(raw_trace, Mapping):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw_trace must be a mapping",
                attempt=self._controller_claim_attempts.get(event_id),
                raw_trace_sha256=fallback_raw_trace_sha256,
                supplied_manifest_sha256=None,
            )
        try:
            trace = json.loads(_canonical_json_text(dict(raw_trace), "raw claim trace"))
        except Exception:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw claim trace is not canonical JSON data",
                attempt=self._controller_claim_attempts.get(event_id),
                raw_trace_sha256=fallback_raw_trace_sha256,
                supplied_manifest_sha256=None,
            )
        raw_trace_sha256 = _sha256(trace)
        attempt = self._controller_claim_attempts.get(event_id)
        supplied_manifest_sha256: str | None = None
        try:
            supplied_manifest_sha256 = manifest_sha256(physical_manifest)
        except Exception:  # The immutable ledger must retain malformed-manifest attempts.
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="supplied physical manifest is unverifiable",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=None,
            )
        if supplied_manifest_sha256 != self._episode_authority.physical_manifest_sha256:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="supplied physical manifest differs from episode authority",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        if attempt is None:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="verified claim has no controller attempt",
                attempt=None,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        attempt.assert_integrity()
        if target_id != attempt.target_id:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="verified target differs from controller attempt",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        if not isinstance(trace, dict):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw claim trace must be an object",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        authority = self._episode_authority
        if (
            trace.get("scene_id") != authority.scene_id
            or trace.get("episode_id") != authority.episode_id
            or trace.get("physical_manifest_sha256")
            != authority.physical_manifest_sha256
            or trace.get("task_object_ids")
            != list(authority.expected_task_object_ids)
            or trace.get("task_object_set_sha256") != authority.task_object_set_sha256
        ):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw trace differs from authoritative episode task/manifest binding",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        if trace.get("evaluator_feedback_to_controller") != []:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="evaluator feedback entered the controller episode",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        raw_events = trace.get("controller_claim_attempts")
        if not isinstance(raw_events, list):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw trace lacks controller claim attempts",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        expected_raw_events = [
            json.loads(self._controller_claim_attempts[attempt_id].raw_event_json)
            for attempt_id in self._controller_claim_attempt_order
        ]
        if raw_events != expected_raw_events:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="raw trace is not the complete ordered controller-attempt ledger",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        matching_raw = [
            row
            for row in raw_events
            if isinstance(row, dict) and row.get("event_id") == event_id
        ]
        if len(matching_raw) != 1 or _sha256(matching_raw[0]) != attempt.raw_event_sha256:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="evaluated raw event differs from recorded attempt",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        try:
            evaluated = evaluate_physical_claim_trace(
                trace,
                physical_manifest,
                list(authority.expected_task_object_ids),
                authority.task_object_set_sha256,
            )
        except Exception as exc:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason=f"canonical evaluator failed: {type(exc).__name__}",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        if not isinstance(evaluated, dict) or evaluated.get("schema") != EVALUATED_TRACE_SCHEMA:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="canonical evaluator returned an invalid trace",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        trace_hash = evaluated.get("trace_content_sha256")
        trace_core = dict(evaluated)
        trace_core.pop("trace_content_sha256", None)
        if trace_hash != _sha256(trace_core):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="evaluated trace content hash mismatch",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        summary = evaluated.get("physical_claim_summary")
        if not isinstance(summary, dict) or summary.get("schema") != SUMMARY_SCHEMA:
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="evaluated trace lacks canonical summary",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        summary_hash = summary.get("content_sha256")
        summary_core = dict(summary)
        summary_core.pop("content_sha256", None)
        if (
            summary_hash != _sha256(summary_core)
            or summary.get("evaluator_contract_sha256") != EVALUATOR_CONTRACT_SHA256
            or summary.get("physical_manifest_sha256")
            != authority.physical_manifest_sha256
            or summary.get("task_object_set_sha256") != authority.task_object_set_sha256
        ):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="canonical summary authority/hash mismatch",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        evaluations = evaluated.get("physical_claim_evaluations")
        if (
            not isinstance(evaluations, list)
            or len(evaluations) != len(self._controller_claim_attempt_order)
            or any(not isinstance(row, dict) for row in evaluations)
            or [row.get("event_id") for row in evaluations]
            != self._controller_claim_attempt_order
        ):
            self._record_unverifiable_and_raise(
                target_id=target_id,
                event_id=event_id,
                reason="canonical evaluation is not the complete ordered attempt ledger",
                attempt=attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
            )
        outcomes: list[
            tuple[
                ControllerClaimAttempt,
                dict[str, object],
                str,
                str,
                bool,
                bool,
            ]
        ] = []
        for canonical_event, attempt_id in zip(
            evaluations,
            self._controller_claim_attempt_order,
            strict=True,
        ):
            canonical_attempt = self._controller_claim_attempts[attempt_id]
            event_hash = canonical_event.get("content_sha256")
            event_core = dict(canonical_event)
            event_core.pop("content_sha256", None)
            expected_reference = {
                "namespace": "object_id",
                "value": canonical_attempt.expected_object_id,
            }
            identity_ok = (
                canonical_attempt.identity_matches_expected
                and canonical_event.get("requested_target") == expected_reference
                and canonical_event.get("claimed_target") == expected_reference
                and canonical_event.get("claimed_target_object_id")
                == canonical_attempt.expected_object_id
            )
            accepted = (
                identity_ok
                and canonical_event.get("decision") == "accepted"
                and canonical_event.get("accepted") is True
                and canonical_event.get("physically_verified") is True
            )
            canonical_credited = (
                canonical_event.get("credited") is True
                and canonical_event.get("duplicate_physical_claim_not_credited")
                is False
            )
            factors = canonical_event.get("factors")
            if (
                canonical_event.get("schema") != EVENT_SCHEMA
                or event_hash != _sha256(event_core)
                or canonical_event.get("evaluator_contract_sha256")
                != EVALUATOR_CONTRACT_SHA256
                or canonical_event.get("physical_manifest_sha256")
                != authority.physical_manifest_sha256
                or canonical_event.get("task_object_set_sha256")
                != authority.task_object_set_sha256
                or canonical_event.get("tick") != canonical_attempt.tick
                or canonical_credited and not accepted
                or accepted
                and (
                    not isinstance(factors, dict)
                    or not factors
                    or any(value is not True for value in factors.values())
                )
            ):
                self._record_unverifiable_and_raise(
                    target_id=target_id,
                    event_id=event_id,
                    reason="claim evaluation event content mismatch",
                    attempt=attempt,
                    raw_trace_sha256=raw_trace_sha256,
                    supplied_manifest_sha256=supplied_manifest_sha256,
                )
            if accepted:
                status = "accepted"
                reason = (
                    "first accepted and credited canonical physical event"
                    if canonical_credited
                    else "accepted event was a duplicate canonical claim"
                )
            elif canonical_event.get("decision") == "rejected":
                status = "rejected"
                reason = "canonical physical evaluator rejected the event"
            else:
                status = "unverifiable"
                reason = "canonical physical evaluator found the event unverifiable"
            outcomes.append(
                (
                    canonical_attempt,
                    canonical_event,
                    event_hash,
                    status,
                    accepted,
                    canonical_credited,
                )
            )

        selected = next(row for row in outcomes if row[0].event_id == event_id)
        trace_already_recorded = any(
            record.evaluated_trace_sha256 == trace_hash
            for record in self._physical_evaluations
        )
        outcomes_to_record = [selected] if trace_already_recorded else outcomes
        created_credits: dict[str, VerifiedClaimCredit] = {}
        for (
            canonical_attempt,
            canonical_event,
            canonical_event_hash,
            status,
            accepted,
            canonical_credited,
        ) in outcomes_to_record:
            creates_credit = (
                not trace_already_recorded
                and accepted
                and canonical_credited
                and canonical_attempt.target_id not in self._verified_claims
                and canonical_attempt.target_id not in created_credits
            )
            credit = None
            if creates_credit:
                credit = VerifiedClaimCredit(
                    target_id=canonical_attempt.target_id,
                    object_id=canonical_attempt.expected_object_id,
                    event_id=canonical_attempt.event_id,
                    tick=canonical_attempt.tick,
                    controller_attempt_sha256=canonical_attempt.content_sha256,
                    evaluator_contract_sha256=EVALUATOR_CONTRACT_SHA256,
                    physical_manifest_sha256=authority.physical_manifest_sha256,
                    task_object_set_sha256=authority.task_object_set_sha256,
                    evaluation_event_sha256=canonical_event_hash,
                    evaluation_summary_sha256=summary_hash,
                    evaluated_trace_sha256=trace_hash,
                )
            reason = (
                "first accepted and credited canonical physical event"
                if creates_credit
                else (
                    "accepted event was duplicate or target already credited"
                    if accepted
                    else (
                        "canonical physical evaluator rejected the event"
                        if status == "rejected"
                        else "canonical physical evaluator found the event unverifiable"
                    )
                )
            )
            record = self._append_evaluation(
                target_id=canonical_attempt.target_id,
                event_id=canonical_attempt.event_id,
                status=status,
                accepted=accepted,
                canonical_credited=canonical_credited,
                verified_credit_created=creates_credit,
                reason=reason,
                attempt=canonical_attempt,
                raw_trace_sha256=raw_trace_sha256,
                supplied_manifest_sha256=supplied_manifest_sha256,
                evaluator_contract_sha256=EVALUATOR_CONTRACT_SHA256,
                evaluation_event_sha256=canonical_event_hash,
                evaluation_summary_sha256=summary_hash,
                evaluated_trace_sha256=trace_hash,
                defer_state_commit=True,
            )
            if credit is not None:
                self._context_issuer._remember_canonical_evaluation(
                    record,
                    episode_authority_sha256=(
                        self._episode_authority.content_sha256
                    ),
                    credit=credit,
                )
                self._verified_claims[credit.target_id] = credit
                created_credits[credit.target_id] = credit
        self._commit_runtime_state()

        selected_attempt, _, _, selected_status, selected_accepted, _ = selected
        selected_credit = created_credits.get(selected_attempt.target_id)
        if selected_credit is not None and selected_credit.event_id == event_id:
            return replace(selected_credit)
        if selected_status == "rejected":
            raise TargetClaimVerificationError(
                "canonical physical evaluator rejected the event"
            )
        if not selected_accepted:
            raise TargetClaimVerificationError(
                "canonical physical evaluator could not verify the event"
            )
        raise TargetClaimVerificationError(
            "accepted event did not create first verified credit"
        )

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        """Return a defensive, canonical, complete memory representation."""

        self._performance_counters["full_state_materializations"] += 1
        self._assert_foundation_integrity()
        result: dict[str, object] = {
            "schema": "lewm_g5_reversible_target_belief_memory_v3",
            "synthetic_only": True,
            "production_authority_eligible": False,
            "config": self._config.to_dict(),
            "context": self._context.to_dict(),
            "context_history": [
                row.to_dict()
                for row in sorted(
                    self._context_history.values(),
                    key=lambda item: item.context_sequence,
                )
            ],
            "context_issuer_contract": dict(self._issuer_contract),
            "episode_authority": self._episode_authority.to_dict(),
            "task_mapping_sha256": self._task_mapping_sha256,
            "revision": self._revision,
            "current_tick": self._current_tick,
            "cell_mass": {
                target: [
                    TargetCellValue(cell=cell, value=value).to_dict()
                    for cell, value in sorted(self._cell_mass[target].items())
                ]
                for target in self._config.target_ids
            },
            "unlocalized_mass": {
                target: self._unlocalized_mass[target]
                for target in self._config.target_ids
            },
            "positive_observations": [
                self._positive[key].to_dict() for key in sorted(self._positive)
            ],
            "negative_observations": [
                self._negative[key].to_dict() for key in sorted(self._negative)
            ],
            "evidence_transaction_order": [
                {"kind": kind, "observation_id": observation_id}
                for kind, observation_id in self._evidence_transaction_order
            ],
            "seen_observation_ids": sorted(self._seen_observation_ids),
            "seen_semantic_sha256s": sorted(self._seen_semantic_sha256s),
            "seen_payload_sha256s": sorted(self._seen_payload_sha256s),
            "controller_claim_attempts": [
                self._controller_claim_attempts[event_id].to_dict()
                for event_id in self._controller_claim_attempt_order
            ],
            "physical_claim_evaluations": [
                record.to_dict() for record in self._physical_evaluations
            ],
            "verified_claims": [
                self._verified_claims[key].to_dict()
                for key in sorted(self._verified_claims)
            ],
            "episode_finalized": self._episode_finalized,
            "exact_sim_tainted": self._context.exact_sim_tainted,
            "ablation_mode": self._context.ablation_mode,
        }
        if include_hash:
            self._performance_counters["canonical_full_state_hashes"] += 1
            result["state_content_sha256"] = _sha256(result)
        return result

    @property
    def content_sha256(self) -> str:
        result = self.to_dict(False)
        self._performance_counters["canonical_full_state_hashes"] += 1
        return _sha256(result)

    def serialize(self) -> bytes:
        state = self.to_dict()
        state_content_sha256 = state["state_content_sha256"]
        self._context_issuer._remember_serialized_memory_state(
            episode_authority_sha256=self._episode_authority.content_sha256,
            revision=self._revision,
            state_content_sha256=state_content_sha256,  # type: ignore[arg-type]
            writer_instance=self,
            writer_capability=self._writer_capability,
        )
        return _canonical_json_bytes(state)

    @classmethod
    def deserialize(
        cls,
        payload: bytes | str,
        *,
        context_issuer: SyntheticTargetMemoryContextIssuer,
        expected_episode_authority: TargetEpisodeAuthority,
    ) -> ReversibleTargetBeliefMemory:
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("serialized target memory is not UTF-8") from exc
        elif isinstance(payload, str):
            text = payload
        else:
            raise TypeError("serialized target memory must be bytes or text")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("serialized target memory is not JSON") from exc
        if _canonical_json_text(value, "serialized target memory") != text:
            raise ValueError("serialized target memory is not canonical JSON")
        return cls.from_mapping(
            value,
            context_issuer=context_issuer,
            expected_episode_authority=expected_episode_authority,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        context_issuer: SyntheticTargetMemoryContextIssuer,
        expected_episode_authority: TargetEpisodeAuthority,
    ) -> ReversibleTargetBeliefMemory:
        if not isinstance(context_issuer, SyntheticTargetMemoryContextIssuer):
            raise TypeError("context_issuer must be SyntheticTargetMemoryContextIssuer")
        if not isinstance(expected_episode_authority, TargetEpisodeAuthority):
            raise TypeError("expected_episode_authority must be TargetEpisodeAuthority")
        expected_episode_authority.assert_integrity()
        required = {
            "schema",
            "synthetic_only",
            "production_authority_eligible",
            "config",
            "context",
            "context_history",
            "context_issuer_contract",
            "episode_authority",
            "task_mapping_sha256",
            "revision",
            "current_tick",
            "cell_mass",
            "unlocalized_mass",
            "positive_observations",
            "negative_observations",
            "evidence_transaction_order",
            "seen_observation_ids",
            "seen_semantic_sha256s",
            "seen_payload_sha256s",
            "controller_claim_attempts",
            "physical_claim_evaluations",
            "verified_claims",
            "episode_finalized",
            "exact_sim_tainted",
            "ablation_mode",
            "state_content_sha256",
        }
        root = _strict_keys(value, required=required, name="target memory")
        if root["schema"] != "lewm_g5_reversible_target_belief_memory_v3":
            raise ValueError("unsupported target-memory schema")
        if (
            root["synthetic_only"] is not True
            or root["production_authority_eligible"] is not False
        ):
            raise PermissionError(
                "serialized synthetic target memory cannot become production eligible"
            )
        state_hash = root["state_content_sha256"]
        _require_sha256(state_hash, "state_content_sha256")
        core = dict(root)
        core.pop("state_content_sha256")
        if state_hash != _sha256(core):
            raise ValueError("serialized target-memory content hash mismatch")
        config = _parse_config(root["config"])
        parsed_context = _parse_context(root["context"])
        history_rows = _require_list(root["context_history"], "context history")
        context_history_list = [_parse_context(row) for row in history_rows]
        if not context_history_list:
            raise ValueError("serialized context history cannot be empty")
        if [row.context_sequence for row in context_history_list] != sorted(
            row.context_sequence for row in context_history_list
        ):
            raise ValueError("serialized context history is not sequence ordered")
        context_history = {
            row.content_sha256: row for row in context_history_list
        }
        if (
            len(context_history) != len(context_history_list)
            or context_history.get(parsed_context.content_sha256) != parsed_context
            or context_history_list[-1].content_sha256
            != parsed_context.content_sha256
        ):
            raise ValueError("serialized current context is not the history terminus")
        authority = _parse_authority(root["episode_authority"])
        if authority.to_dict() != expected_episode_authority.to_dict():
            raise TargetSnapshotBindingError(
                "serialized episode authority differs from independent expectation"
            )
        issuer_contract = root["context_issuer_contract"]
        if issuer_contract != context_issuer.contract_dict():
            raise TargetSnapshotBindingError("serialized context issuer contract changed")
        previous_context: TargetMemoryContext | None = None
        for historical in context_history_list:
            context_issuer.assert_known_context_content(historical)
            if (
                historical.issuer_sha256 != parsed_context.issuer_sha256
                or historical.map_frame_sha256 != parsed_context.map_frame_sha256
                or historical.morphology_sha256 != parsed_context.morphology_sha256
                or historical.camera_calibration_sha256
                != parsed_context.camera_calibration_sha256
                or historical.frustum_contract_sha256
                != parsed_context.frustum_contract_sha256
                or historical.physical_los_contract_sha256
                != parsed_context.physical_los_contract_sha256
                or historical.positive_evidence_producer_sha256
                != parsed_context.positive_evidence_producer_sha256
                or historical.negative_visibility_producer_sha256
                != parsed_context.negative_visibility_producer_sha256
                or historical.observation_model_checkpoint_sha256
                != parsed_context.observation_model_checkpoint_sha256
                or historical.exact_sim_tainted != parsed_context.exact_sim_tainted
                or historical.ablation_mode != parsed_context.ablation_mode
            ):
                raise TargetSnapshotBindingError(
                    "serialized context-history binding changed"
                )
            transition_error = (
                None
                if previous_context is None
                else _context_transition_error(previous_context, historical)
            )
            if transition_error is not None:
                raise TargetSnapshotBindingError(
                    f"serialized context transition is invalid: {transition_error}"
                )
            previous_context = historical
        task_mapping_sha256 = root["task_mapping_sha256"]
        _require_sha256(task_mapping_sha256, "task_mapping_sha256")
        expected_mapping_hash = _sha256(
            [
                {"target_id": target, "object_id": object_id}
                for target, object_id in authority.task_object_by_target_id
            ]
        )
        if task_mapping_sha256 != expected_mapping_hash:
            raise TargetSnapshotBindingError("serialized task mapping changed")
        revision = _nonnegative_int(root["revision"], "revision")  # type: ignore[arg-type]
        current_tick = _nonnegative_int(root["current_tick"], "current_tick")  # type: ignore[arg-type]
        context_issuer.assert_latest_memory_state(
            episode_authority_sha256=authority.content_sha256,
            revision=revision,
            state_content_sha256=state_hash,
        )
        if not isinstance(root["episode_finalized"], bool):
            raise ValueError("episode_finalized must be boolean")
        episode_finalized = root["episode_finalized"]
        if root["exact_sim_tainted"] != parsed_context.exact_sim_tainted:
            raise TargetSnapshotBindingError("serialized taint differs from context")
        if root["ablation_mode"] != parsed_context.ablation_mode:
            raise TargetSnapshotBindingError("serialized ablation differs from context")

        cell_mass_root = root["cell_mass"]
        unlocalized_root = root["unlocalized_mass"]
        if not isinstance(cell_mass_root, Mapping) or set(cell_mass_root) != set(config.target_ids):
            raise ValueError("serialized cell-mass target set changed")
        if not isinstance(unlocalized_root, Mapping) or set(unlocalized_root) != set(config.target_ids):
            raise ValueError("serialized unlocalized target set changed")
        cell_mass: dict[str, dict[Cell, float]] = {}
        unlocalized_mass: dict[str, float] = {}
        for target in config.target_ids:
            rows_value = cell_mass_root[target]
            if not isinstance(rows_value, list):
                raise ValueError("serialized cell mass must be a list")
            rows = tuple(_parse_cell_value(row) for row in rows_value)
            rows = _cell_values(rows, name="serialized cell mass", allow_empty=True)
            if {row.cell for row in rows} - parsed_context.candidate_domain:
                raise ValueError("serialized posterior escaped candidate domain")
            if any(row.value < config.posterior_mass_floor for row in rows):
                raise ValueError("serialized posterior violates mass floor")
            unlocalized = _unit(unlocalized_root[target], "serialized unlocalized mass")  # type: ignore[arg-type]
            if not math.isclose(
                math.fsum(row.value for row in rows) + unlocalized,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("serialized posterior is not normalized")
            cell_mass[target] = {row.cell: row.value for row in rows}
            unlocalized_mass[target] = unlocalized

        positive_rows = _require_list(root["positive_observations"], "positive observations")
        negative_rows = _require_list(root["negative_observations"], "negative observations")
        positives = [_parse_positive(row) for row in positive_rows]
        negatives = [_parse_negative(row) for row in negative_rows]
        for positive in positives:
            context_issuer.assert_known_positive_observation_content(positive)
        positive_map = {row.identity.observation_id: row for row in positives}
        negative_map = {row.identity.observation_id: row for row in negatives}
        if len(positive_map) != len(positives) or len(negative_map) != len(negatives):
            raise ValueError("serialized evidence IDs are not unique")
        if set(positive_map) & set(negative_map):
            raise ValueError("serialized evidence ID crosses evidence namespaces")
        all_observations: list[PositiveTargetObservation | NegativeTargetObservation] = [
            *positives,
            *negatives,
        ]
        order_rows = _require_list(
            root["evidence_transaction_order"],
            "evidence transaction order",
        )
        evidence_transaction_order: list[tuple[str, str]] = []
        for order_row in order_rows:
            parsed_order = _strict_keys(
                order_row,
                required={"kind", "observation_id"},
                name="evidence transaction order row",
            )
            kind = parsed_order["kind"]
            observation_id = parsed_order["observation_id"]
            if kind not in {"positive", "negative"}:
                raise ValueError("serialized evidence kind is invalid")
            if type(observation_id) is not str or not observation_id:
                raise ValueError("serialized evidence observation ID is invalid")
            evidence_transaction_order.append((kind, observation_id))
        if any(
            row.context_sha256 not in context_history
            or row.target_id not in config.target_ids
            or row.identity.tick > current_tick
            for row in all_observations
        ):
            raise ValueError("serialized evidence binding is invalid")
        for negative in negatives:
            historical = context_history[negative.context_sha256]
            certificate = negative.visibility_certificate
            context_issuer.assert_known_certificate_content(certificate)
            if (
                certificate.context_sha256 != historical.content_sha256
                or certificate.physical_content_sha256
                != historical.physical_content_sha256
                or certificate.configuration_snapshot_sha256
                != historical.configuration_snapshot_sha256
                or certificate.pose_provenance_sha256
                != historical.pose_provenance_sha256
                or certificate.frustum_contract_sha256
                != historical.frustum_contract_sha256
                or certificate.physical_los_contract_sha256
                != historical.physical_los_contract_sha256
                or certificate.producer_sha256
                != historical.negative_visibility_producer_sha256
                or certificate.evidence_identity_sha256
                != _sha256(negative.identity.to_dict())
                or certificate.target_id != negative.target_id
                or certificate.confidence != negative.confidence
                or certificate.certified_detection_probability
                != tuple(
                    (row.cell, row.value)
                    for row in negative.visible_detection_probability
                )
                or certificate.certified_visible_cells - historical.candidate_domain
            ):
                raise ValueError("serialized negative certificate binding is invalid")
        derived_ids = {row.identity.observation_id for row in all_observations}
        derived_semantics = {row.semantic_sha256 for row in all_observations}
        derived_payloads = {row.identity.payload_sha256 for row in all_observations}
        if len(derived_payloads) != len(all_observations):
            raise ValueError("serialized duplicate payload registry contains reuse")
        if _parse_string_set(root["seen_observation_ids"], "seen observation IDs") != derived_ids:
            raise ValueError("serialized observation-ID registry changed")
        if _parse_hash_set(root["seen_semantic_sha256s"], "seen semantics") != derived_semantics:
            raise ValueError("serialized semantic registry changed")
        if _parse_hash_set(root["seen_payload_sha256s"], "seen payloads") != derived_payloads:
            raise ValueError("serialized payload registry changed")

        attempt_rows = _require_list(root["controller_claim_attempts"], "claim attempts")
        attempts = [_parse_attempt(row) for row in attempt_rows]
        attempt_map = {row.event_id: row for row in attempts}
        if len(attempt_map) != len(attempts):
            raise ValueError("serialized controller event IDs are not unique")
        if any(
            row.target_id not in dict(authority.task_object_by_target_id)
            or row.expected_object_id
            != dict(authority.task_object_by_target_id).get(row.target_id)
            or row.tick > current_tick
            for row in attempts
        ):
            raise ValueError("serialized controller attempt binding is invalid")
        evaluation_rows = _require_list(
            root["physical_claim_evaluations"],
            "physical evaluations",
        )
        evaluations = [_parse_evaluation(row) for row in evaluation_rows]
        for evaluation in evaluations:
            context_issuer.assert_known_evaluation_content(evaluation)
        if [row.evaluation_sequence for row in evaluations] != list(
            range(1, len(evaluations) + 1)
        ):
            raise ValueError("serialized evaluation sequence is not contiguous")
        if any(
            row.controller_attempt_sha256 is not None
            and row.controller_attempt_sha256
            not in {attempt.content_sha256 for attempt in attempts}
            for row in evaluations
        ):
            raise ValueError("serialized evaluation lacks its controller attempt")
        credit_rows = _require_list(root["verified_claims"], "verified claims")
        credits = [_parse_credit(row) for row in credit_rows]
        for credit in credits:
            context_issuer.assert_known_credit_content(credit)
        context_issuer.assert_complete_evaluation_ledger(
            episode_authority_sha256=authority.content_sha256,
            evaluations=evaluations,
            credits=credits,
        )
        credit_map = {row.target_id: row for row in credits}
        if len(credit_map) != len(credits):
            raise ValueError("serialized target has multiple verified credits")
        for credit in credits:
            attempt = attempt_map.get(credit.event_id)
            if (
                attempt is None
                or credit.controller_attempt_sha256 != attempt.content_sha256
                or credit.object_id != attempt.expected_object_id
                or credit.physical_manifest_sha256 != authority.physical_manifest_sha256
                or credit.task_object_set_sha256 != authority.task_object_set_sha256
                or not any(
                    record.event_id == credit.event_id
                    and record.verified_credit_created
                    and record.evaluation_event_sha256 == credit.evaluation_event_sha256
                    for record in evaluations
                )
            ):
                raise ValueError("serialized verified credit binding is invalid")
        if (evaluations or credits) and not episode_finalized:
            raise ValueError("serialized evaluator state predates episode finalization")

        # All serialized data is validated before the issuer creates a rebound object.
        rebound_context = context_issuer._rehydrate_known_context(parsed_context)
        memory = cls(
            rebound_context,
            config,
            context_issuer=context_issuer,
            episode_authority=authority,
            _is_restore=True,
            _restore_finalized=episode_finalized,
        )
        memory._revision = revision
        memory._current_tick = current_tick
        memory._context_history = {
            key: memory._clone_context(row)
            for key, row in context_history.items()
        }
        memory._authorized_context_history_sha256s = set(context_history)
        memory._cell_mass = cell_mass
        memory._unlocalized_mass = unlocalized_mass
        memory._positive = positive_map
        memory._negative = negative_map
        memory._evidence_transaction_order = evidence_transaction_order
        (
            memory._target_evidence_chain_sha256,
            memory._positive_evidence_count,
            memory._negative_evidence_count,
        ) = memory._derive_target_evidence_state()
        memory._rebuild_evidence_cell_indexes()
        memory._rebuild_component_indexes()
        memory._authorized_negative_certificate_sha256s = {
            row.visibility_certificate.content_sha256 for row in negatives
        }
        memory._seen_observation_ids = derived_ids
        memory._seen_semantic_sha256s = derived_semantics
        memory._seen_payload_sha256s = derived_payloads
        memory._controller_claim_attempts = attempt_map
        memory._controller_claim_attempt_order = [row.event_id for row in attempts]
        memory._physical_evaluations = evaluations
        memory._verified_claims = credit_map
        memory._episode_finalized = episode_finalized
        memory._snapshot_history_sha256s = {
            attempt.target_snapshot_sha256 for attempt in attempts
        }
        memory._assert_foundation_integrity(require_writer=False)
        context_issuer.transfer_restore_writer(
            episode_authority_sha256=authority.content_sha256,
            writer_instance=memory,
            writer_capability=memory._writer_capability,
            revision=revision,
            state_content_sha256=state_hash,
        )
        memory._assert_foundation_integrity()
        if memory.to_dict() != dict(root):
            raise ValueError("target-memory round trip changed canonical content")
        return memory


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _parse_string_set(value: object, name: str) -> set[str]:
    rows = _require_list(value, name)
    if any(type(row) is not str or not row for row in rows):
        raise ValueError(f"{name} must contain nonempty strings")
    if rows != sorted(rows) or len(set(rows)) != len(rows):
        raise ValueError(f"{name} must be sorted and unique")
    return set(rows)


def _parse_hash_set(value: object, name: str) -> set[str]:
    rows = _parse_string_set(value, name)
    for row in rows:
        _require_sha256(row, name)
    return rows


def _check_nested_hash(
    source: Mapping[str, object],
    built: object,
    *,
    name: str,
) -> None:
    supplied = source.get("content_sha256")
    actual = getattr(built, "content_sha256")
    if supplied != actual:
        raise ValueError(f"{name} content hash mismatch")


def _parse_config(value: object) -> TargetMemoryConfig:
    keys = {
        "schema",
        "target_ids",
        "cell_size_m",
        "origin_xy_m",
        "maximum_positive_transfer",
        "negative_mass_floor_multiplier",
        "posterior_mass_floor",
        "component_mass_floor",
        "covariance_floor_m2",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="target-memory config")
    if row["schema"] != "lewm_g5_target_memory_config_v2":
        raise ValueError("unsupported target-memory config schema")
    target_ids_value = row["target_ids"]
    origin_value = row["origin_xy_m"]
    if not isinstance(target_ids_value, list) or not isinstance(origin_value, list):
        raise ValueError("serialized config tuple fields must be lists")
    built = TargetMemoryConfig(
        target_ids=tuple(target_ids_value),  # type: ignore[arg-type]
        cell_size_m=row["cell_size_m"],  # type: ignore[arg-type]
        origin_xy_m=tuple(origin_value),  # type: ignore[arg-type]
        maximum_positive_transfer=row["maximum_positive_transfer"],  # type: ignore[arg-type]
        negative_mass_floor_multiplier=row["negative_mass_floor_multiplier"],  # type: ignore[arg-type]
        posterior_mass_floor=row["posterior_mass_floor"],  # type: ignore[arg-type]
        component_mass_floor=row["component_mass_floor"],  # type: ignore[arg-type]
        covariance_floor_m2=row["covariance_floor_m2"],  # type: ignore[arg-type]
    )
    _check_nested_hash(row, built, name="target-memory config")
    return built


def _parse_authority(value: object) -> TargetEpisodeAuthority:
    keys = {
        "schema",
        "scene_id",
        "episode_id",
        "physical_manifest_sha256",
        "context_issuer_contract_sha256",
        "expected_task_object_ids",
        "task_object_by_target_id",
        "task_object_set_sha256",
        "evaluator_observer_mode",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="episode authority")
    if row["schema"] != "lewm_g5_target_episode_authority_v2":
        raise ValueError("unsupported episode-authority schema")
    ids = _require_list(row["expected_task_object_ids"], "expected task IDs")
    mappings = _require_list(row["task_object_by_target_id"], "task mapping")
    pairs: list[tuple[str, str]] = []
    for mapping in mappings:
        parsed = _strict_keys(
            mapping,
            required={"target_id", "object_id"},
            name="task mapping row",
        )
        pairs.append((parsed["target_id"], parsed["object_id"]))  # type: ignore[arg-type]
    built = TargetEpisodeAuthority(
        scene_id=row["scene_id"],  # type: ignore[arg-type]
        episode_id=row["episode_id"],  # type: ignore[arg-type]
        physical_manifest_sha256=row["physical_manifest_sha256"],  # type: ignore[arg-type]
        context_issuer_contract_sha256=row["context_issuer_contract_sha256"],  # type: ignore[arg-type]
        expected_task_object_ids=tuple(ids),  # type: ignore[arg-type]
        task_object_by_target_id=tuple(pairs),
        task_object_set_sha256=row["task_object_set_sha256"],  # type: ignore[arg-type]
        evaluator_observer_mode=row["evaluator_observer_mode"],  # type: ignore[arg-type]
    )
    _check_nested_hash(row, built, name="episode authority")
    return built


def _parse_context(value: object) -> TargetMemoryContext:
    keys = {
        "schema",
        "issuer_sha256",
        "issuance_id_sha256",
        "context_sequence",
        "pose_timestamp_ns",
        "map_frame_sha256",
        "physical_content_sha256",
        "physical_revision",
        "configuration_snapshot_sha256",
        "morphology_sha256",
        "pose_provenance_sha256",
        "camera_calibration_sha256",
        "frustum_contract_sha256",
        "physical_los_contract_sha256",
        "positive_evidence_producer_sha256",
        "negative_visibility_producer_sha256",
        "observation_model_checkpoint_sha256",
        "candidate_domain",
        "exact_sim_tainted",
        "ablation_mode",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="target-memory context")
    if row["schema"] != "lewm_g5_target_memory_context_v2":
        raise ValueError("unsupported target-memory context schema")
    domain = _require_list(row["candidate_domain"], "candidate domain")
    built = TargetMemoryContext(
        issuer_sha256=row["issuer_sha256"],  # type: ignore[arg-type]
        issuance_id_sha256=row["issuance_id_sha256"],  # type: ignore[arg-type]
        context_sequence=row["context_sequence"],  # type: ignore[arg-type]
        pose_timestamp_ns=row["pose_timestamp_ns"],  # type: ignore[arg-type]
        map_frame_sha256=row["map_frame_sha256"],  # type: ignore[arg-type]
        physical_content_sha256=row["physical_content_sha256"],  # type: ignore[arg-type]
        physical_revision=row["physical_revision"],  # type: ignore[arg-type]
        configuration_snapshot_sha256=row["configuration_snapshot_sha256"],  # type: ignore[arg-type]
        morphology_sha256=row["morphology_sha256"],  # type: ignore[arg-type]
        pose_provenance_sha256=row["pose_provenance_sha256"],  # type: ignore[arg-type]
        camera_calibration_sha256=row["camera_calibration_sha256"],  # type: ignore[arg-type]
        frustum_contract_sha256=row["frustum_contract_sha256"],  # type: ignore[arg-type]
        physical_los_contract_sha256=row["physical_los_contract_sha256"],  # type: ignore[arg-type]
        positive_evidence_producer_sha256=row["positive_evidence_producer_sha256"],  # type: ignore[arg-type]
        negative_visibility_producer_sha256=row["negative_visibility_producer_sha256"],  # type: ignore[arg-type]
        observation_model_checkpoint_sha256=row["observation_model_checkpoint_sha256"],  # type: ignore[arg-type]
        candidate_domain=frozenset(_cell(cell) for cell in domain),  # type: ignore[arg-type]
        exact_sim_tainted=row["exact_sim_tainted"],  # type: ignore[arg-type]
        ablation_mode=row["ablation_mode"],  # type: ignore[arg-type]
        _issuance_capability=object(),
    )
    _check_nested_hash(row, built, name="target-memory context")
    return built


def _parse_identity(value: object) -> TargetEvidenceIdentity:
    row = _strict_keys(
        value,
        required={
            "observation_id",
            "payload_sha256",
            "producer_sha256",
            "diversity_key",
            "tick",
        },
        name="target evidence identity",
    )
    return TargetEvidenceIdentity(
        observation_id=row["observation_id"],  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        producer_sha256=row["producer_sha256"],  # type: ignore[arg-type]
        diversity_key=row["diversity_key"],  # type: ignore[arg-type]
        tick=row["tick"],  # type: ignore[arg-type]
    )


def _parse_cell_value(value: object) -> TargetCellValue:
    row = _strict_keys(
        value,
        required={"cell", "value"},
        name="target cell value",
    )
    return TargetCellValue(
        cell=_cell(row["cell"]),  # type: ignore[arg-type]
        value=row["value"],  # type: ignore[arg-type]
    )


def _parse_positive(value: object) -> PositiveTargetObservation:
    keys = {
        "schema",
        "identity",
        "context_sha256",
        "target_id",
        "localized_distribution",
        "unlocalized_probability",
        "confidence",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="positive observation")
    if row["schema"] != "lewm_g5_positive_target_observation_v3":
        raise ValueError("unsupported positive-observation schema")
    values = _require_list(row["localized_distribution"], "localized distribution")
    built = PositiveTargetObservation(
        identity=_parse_identity(row["identity"]),
        context_sha256=row["context_sha256"],  # type: ignore[arg-type]
        target_id=row["target_id"],  # type: ignore[arg-type]
        localized_distribution=tuple(_parse_cell_value(item) for item in values),
        unlocalized_probability=row["unlocalized_probability"],  # type: ignore[arg-type]
        confidence=row["confidence"],  # type: ignore[arg-type]
        _issuance_capability=object(),
    )
    _check_nested_hash(row, built, name="positive observation")
    return built


def _parse_certificate(value: object) -> NegativeVisibilityCertificate:
    keys = {
        "schema",
        "issuer_sha256",
        "certificate_id_sha256",
        "context_sha256",
        "physical_content_sha256",
        "configuration_snapshot_sha256",
        "pose_provenance_sha256",
        "frustum_contract_sha256",
        "physical_los_contract_sha256",
        "producer_sha256",
        "evidence_identity_sha256",
        "target_id",
        "confidence",
        "certified_detection_probability",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="visibility certificate")
    if row["schema"] != "lewm_g5_negative_visibility_certificate_v2":
        raise ValueError("unsupported visibility-certificate schema")
    probability_rows = _require_list(
        row["certified_detection_probability"],
        "certified detection probability",
    )
    probabilities: list[tuple[Cell, float]] = []
    for probability_row in probability_rows:
        parsed_probability = _strict_keys(
            probability_row,
            required={"cell", "value"},
            name="certified detection row",
        )
        probabilities.append(
            (
                _cell(parsed_probability["cell"]),  # type: ignore[arg-type]
                parsed_probability["value"],  # type: ignore[arg-type]
            )
        )
    built = NegativeVisibilityCertificate(
        issuer_sha256=row["issuer_sha256"],  # type: ignore[arg-type]
        certificate_id_sha256=row["certificate_id_sha256"],  # type: ignore[arg-type]
        context_sha256=row["context_sha256"],  # type: ignore[arg-type]
        physical_content_sha256=row["physical_content_sha256"],  # type: ignore[arg-type]
        configuration_snapshot_sha256=row["configuration_snapshot_sha256"],  # type: ignore[arg-type]
        pose_provenance_sha256=row["pose_provenance_sha256"],  # type: ignore[arg-type]
        frustum_contract_sha256=row["frustum_contract_sha256"],  # type: ignore[arg-type]
        physical_los_contract_sha256=row["physical_los_contract_sha256"],  # type: ignore[arg-type]
        producer_sha256=row["producer_sha256"],  # type: ignore[arg-type]
        evidence_identity_sha256=row["evidence_identity_sha256"],  # type: ignore[arg-type]
        target_id=row["target_id"],  # type: ignore[arg-type]
        confidence=row["confidence"],  # type: ignore[arg-type]
        certified_detection_probability=tuple(probabilities),
        _certificate_capability=object(),
    )
    _check_nested_hash(row, built, name="visibility certificate")
    return built


def _parse_negative(value: object) -> NegativeTargetObservation:
    keys = {
        "schema",
        "identity",
        "context_sha256",
        "target_id",
        "visible_detection_probability",
        "confidence",
        "visibility_certificate",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="negative observation")
    if row["schema"] != "lewm_g5_negative_target_observation_v2":
        raise ValueError("unsupported negative-observation schema")
    values = _require_list(
        row["visible_detection_probability"],
        "visible detection probability",
    )
    built = NegativeTargetObservation(
        identity=_parse_identity(row["identity"]),
        context_sha256=row["context_sha256"],  # type: ignore[arg-type]
        target_id=row["target_id"],  # type: ignore[arg-type]
        visible_detection_probability=tuple(
            _parse_cell_value(item) for item in values
        ),
        confidence=row["confidence"],  # type: ignore[arg-type]
        visibility_certificate=_parse_certificate(row["visibility_certificate"]),
    )
    _check_nested_hash(row, built, name="negative observation")
    return built


def _parse_attempt(value: object) -> ControllerClaimAttempt:
    keys = {
        "schema",
        "target_id",
        "expected_object_id",
        "event_id",
        "tick",
        "target_snapshot_sha256",
        "context_sha256",
        "requested_target_json",
        "claimed_target_json",
        "identity_matches_expected",
        "raw_event_json",
        "raw_event_sha256",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="controller claim attempt")
    if row["schema"] != "lewm_g5_controller_claim_attempt_v2":
        raise ValueError("unsupported controller-attempt schema")
    built = ControllerClaimAttempt(
        target_id=row["target_id"],  # type: ignore[arg-type]
        expected_object_id=row["expected_object_id"],  # type: ignore[arg-type]
        event_id=row["event_id"],  # type: ignore[arg-type]
        tick=row["tick"],  # type: ignore[arg-type]
        target_snapshot_sha256=row["target_snapshot_sha256"],  # type: ignore[arg-type]
        context_sha256=row["context_sha256"],  # type: ignore[arg-type]
        requested_target_json=row["requested_target_json"],  # type: ignore[arg-type]
        claimed_target_json=row["claimed_target_json"],  # type: ignore[arg-type]
        identity_matches_expected=row["identity_matches_expected"],  # type: ignore[arg-type]
        raw_event_json=row["raw_event_json"],  # type: ignore[arg-type]
        raw_event_sha256=row["raw_event_sha256"],  # type: ignore[arg-type]
    )
    _check_nested_hash(row, built, name="controller claim attempt")
    return built


def _parse_evaluation(value: object) -> PhysicalClaimEvaluationRecord:
    keys = {
        "schema",
        "evaluation_sequence",
        "episode_authority_sha256",
        "target_id",
        "event_id",
        "status",
        "accepted",
        "canonical_credited",
        "verified_credit_created",
        "reason",
        "controller_attempt_sha256",
        "evaluator_contract_sha256",
        "supplied_manifest_sha256",
        "task_object_set_sha256",
        "evaluation_event_sha256",
        "evaluation_summary_sha256",
        "evaluated_trace_sha256",
        "raw_trace_sha256",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="physical evaluation")
    if row["schema"] != "lewm_g5_physical_claim_evaluation_v2":
        raise ValueError("unsupported physical-evaluation schema")
    built = PhysicalClaimEvaluationRecord(
        evaluation_sequence=row["evaluation_sequence"],  # type: ignore[arg-type]
        episode_authority_sha256=row["episode_authority_sha256"],  # type: ignore[arg-type]
        target_id=row["target_id"],  # type: ignore[arg-type]
        event_id=row["event_id"],  # type: ignore[arg-type]
        status=row["status"],  # type: ignore[arg-type]
        accepted=row["accepted"],  # type: ignore[arg-type]
        canonical_credited=row["canonical_credited"],  # type: ignore[arg-type]
        verified_credit_created=row["verified_credit_created"],  # type: ignore[arg-type]
        reason=row["reason"],  # type: ignore[arg-type]
        controller_attempt_sha256=row["controller_attempt_sha256"],  # type: ignore[arg-type]
        evaluator_contract_sha256=row["evaluator_contract_sha256"],  # type: ignore[arg-type]
        supplied_manifest_sha256=row["supplied_manifest_sha256"],  # type: ignore[arg-type]
        task_object_set_sha256=row["task_object_set_sha256"],  # type: ignore[arg-type]
        evaluation_event_sha256=row["evaluation_event_sha256"],  # type: ignore[arg-type]
        evaluation_summary_sha256=row["evaluation_summary_sha256"],  # type: ignore[arg-type]
        evaluated_trace_sha256=row["evaluated_trace_sha256"],  # type: ignore[arg-type]
        raw_trace_sha256=row["raw_trace_sha256"],  # type: ignore[arg-type]
    )
    _check_nested_hash(row, built, name="physical evaluation")
    return built


def _parse_credit(value: object) -> VerifiedClaimCredit:
    keys = {
        "schema",
        "target_id",
        "object_id",
        "event_id",
        "tick",
        "controller_attempt_sha256",
        "evaluator_contract_sha256",
        "physical_manifest_sha256",
        "task_object_set_sha256",
        "evaluation_event_sha256",
        "evaluation_summary_sha256",
        "evaluated_trace_sha256",
        "content_sha256",
    }
    row = _strict_keys(value, required=keys, name="verified claim credit")
    if row["schema"] != "lewm_g5_verified_claim_credit_v1":
        raise ValueError("unsupported verified-credit schema")
    built = VerifiedClaimCredit(
        target_id=row["target_id"],  # type: ignore[arg-type]
        object_id=row["object_id"],  # type: ignore[arg-type]
        event_id=row["event_id"],  # type: ignore[arg-type]
        tick=row["tick"],  # type: ignore[arg-type]
        controller_attempt_sha256=row["controller_attempt_sha256"],  # type: ignore[arg-type]
        evaluator_contract_sha256=row["evaluator_contract_sha256"],  # type: ignore[arg-type]
        physical_manifest_sha256=row["physical_manifest_sha256"],  # type: ignore[arg-type]
        task_object_set_sha256=row["task_object_set_sha256"],  # type: ignore[arg-type]
        evaluation_event_sha256=row["evaluation_event_sha256"],  # type: ignore[arg-type]
        evaluation_summary_sha256=row["evaluation_summary_sha256"],  # type: ignore[arg-type]
        evaluated_trace_sha256=row["evaluated_trace_sha256"],  # type: ignore[arg-type]
    )
    _check_nested_hash(row, built, name="verified claim credit")
    return built


__all__ = [
    "Cell",
    "ControllerClaimAttempt",
    "NegativeTargetObservation",
    "NegativeVisibilityCertificate",
    "PhysicalClaimEvaluationRecord",
    "PositiveTargetObservation",
    "ReversibleTargetBeliefMemory",
    "TargetCellValue",
    "TargetClaimVerificationError",
    "TargetEpisodeAuthority",
    "TargetEvidenceIdentity",
    "TargetEvidenceRejectedError",
    "TargetHypothesis",
    "TargetMemoryConfig",
    "TargetMemoryContext",
    "TargetPosteriorSnapshot",
    "TargetSnapshotBindingError",
    "VerifiedClaimCredit",
]
