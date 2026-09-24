"""Fail-closed G3 boundary for future runner-owned execution evidence.

No reviewed canonical runner currently issues executor outcomes or reset
certificates.  This module therefore exposes the exact frozen shape that a
future runner integration must satisfy, but deliberately exposes no binding,
receipt-issuance, transaction-building, or fusion API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from lewm.planning.geometry_contract import (
    DEPLOYMENT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    ConfigurationMorphology,
    REGISTERED_FOOTPRINT_RADIUS_M,
    REGISTERED_PHYSICAL_CELL_SIZE_M,
    RevisionedPhysicalMemory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_GEOMETRY_CONTRACT_SHA256 = (
    "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
)
CANONICAL_GEOMETRY_FILE_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
CANONICAL_DIRECTIONAL_POLICY_SHA256 = (
    "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc"
)
CANONICAL_PRIMITIVE_REGISTRY_SHA256 = (
    "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8"
)
CANONICAL_PLATFORM_MANIFEST_SHA256 = (
    "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189"
)

CANONICAL_BODY_FORWARD_M = 0.3700000000000001
CANONICAL_BODY_REAR_M = 0.43210313102250314
CANONICAL_BODY_HALF_WIDTH_M = 0.2668059073252429
CANONICAL_MAX_TRANSLATION_SUBSTEP_M = 0.025
CANONICAL_MAX_ANGULAR_SUBSTEP_RAD = 0.025
CANONICAL_POSE_SAMPLE_CADENCE_NS = 50_000_000
CANONICAL_COMMAND_CADENCE_NS = 100_000_000
CANONICAL_COMMAND_COUNT = 5
CANONICAL_MAX_POSE_SEQUENCE_LENGTH = 11
CANONICAL_MAX_OUTCOME_DURATION_NS = 500_000_000

# A reviewed runner authority has not been installed.  These identities must
# become fixed source hashes in a separately reviewed change before admission
# can be implemented.  ``None`` is a hard disable, not a wildcard.
CANONICAL_RUNNER_PRODUCER_SHA256: str | None = None
CANONICAL_RESET_PRODUCER_SHA256: str | None = None
CANONICAL_RUNNER_OUTCOME_PROTOCOL_SHA256: str | None = None
CANONICAL_RESET_CLEARANCE_PROTOCOL_SHA256: str | None = None

_EPS = 1e-12


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _wrapped_delta(left: float, right: float) -> float:
    return (right - left + math.pi) % (2.0 * math.pi) - math.pi


class PromotedExecutionAdmissionUnavailableError(PermissionError):
    """Raised while no reviewed canonical runner authority is installed."""


@dataclass(frozen=True, slots=True)
class CanonicalRunnerPoseSample:
    """Validation-only pose value used to freeze the future runner contract."""

    center_xy_m: tuple[float, float]
    yaw_rad: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.center_xy_m, tuple) or len(self.center_xy_m) != 2:
            raise ValueError("center_xy_m must contain two values")
        object.__setattr__(
            self,
            "center_xy_m",
            (
                _finite(self.center_xy_m[0], "body center"),
                _finite(self.center_xy_m[1], "body center"),
            ),
        )
        yaw = _finite(self.yaw_rad, "body yaw")
        object.__setattr__(
            self,
            "yaw_rad",
            (yaw + math.pi) % (2.0 * math.pi) - math.pi,
        )
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")

    def __copy__(self) -> "CanonicalRunnerPoseSample":
        raise TypeError("canonical runner pose samples are non-copyable")

    def __deepcopy__(self, memo: object) -> "CanonicalRunnerPoseSample":
        del memo
        raise TypeError("canonical runner pose samples are non-copyable")


def validate_canonical_runner_pose_sequence(
    samples: Iterable[CanonicalRunnerPoseSample],
) -> tuple[CanonicalRunnerPoseSample, ...]:
    """Validate geometry and timing only; this never creates authority."""

    rows = tuple(samples)
    if not rows or any(not isinstance(row, CanonicalRunnerPoseSample) for row in rows):
        raise ValueError("runner pose sequence is empty or malformed")
    if len(rows) > CANONICAL_MAX_POSE_SEQUENCE_LENGTH:
        raise ValueError("runner pose sequence exceeds the canonical maximum")
    duration = rows[-1].timestamp_ns - rows[0].timestamp_ns
    if duration < 0 or duration > CANONICAL_MAX_OUTCOME_DURATION_NS:
        raise ValueError("runner pose sequence duration exceeds the canonical maximum")
    for prior, current in zip(rows, rows[1:]):
        gap = current.timestamp_ns - prior.timestamp_ns
        if gap != CANONICAL_POSE_SAMPLE_CADENCE_NS:
            raise ValueError("runner pose sequence cadence changed")
        translation = math.hypot(
            current.center_xy_m[0] - prior.center_xy_m[0],
            current.center_xy_m[1] - prior.center_xy_m[1],
        )
        if translation > CANONICAL_MAX_TRANSLATION_SUBSTEP_M + _EPS:
            raise ValueError("runner translation substep exceeds 0.025 m")
        if (
            abs(_wrapped_delta(prior.yaw_rad, current.yaw_rad))
            > CANONICAL_MAX_ANGULAR_SUBSTEP_RAD + _EPS
        ):
            raise ValueError("runner angular substep exceeds 0.025 rad")
    return rows


@dataclass(frozen=True, slots=True)
class CanonicalExecutorResetContract:
    """Exact validation contract; it grants no execution-evidence authority."""

    geometry_contract_sha256: str = CANONICAL_GEOMETRY_CONTRACT_SHA256
    geometry_file_sha256: str = CANONICAL_GEOMETRY_FILE_SHA256
    directional_policy_sha256: str = CANONICAL_DIRECTIONAL_POLICY_SHA256
    primitive_registry_sha256: str = CANONICAL_PRIMITIVE_REGISTRY_SHA256
    platform_manifest_sha256: str = CANONICAL_PLATFORM_MANIFEST_SHA256
    body_forward_m: float = CANONICAL_BODY_FORWARD_M
    body_rear_m: float = CANONICAL_BODY_REAR_M
    body_half_width_m: float = CANONICAL_BODY_HALF_WIDTH_M
    reset_footprint_radius_m: float = REGISTERED_FOOTPRINT_RADIUS_M
    physical_cell_size_m: float = REGISTERED_PHYSICAL_CELL_SIZE_M
    maximum_translation_substep_m: float = CANONICAL_MAX_TRANSLATION_SUBSTEP_M
    maximum_angular_substep_rad: float = CANONICAL_MAX_ANGULAR_SUBSTEP_RAD
    pose_sample_cadence_ns: int = CANONICAL_POSE_SAMPLE_CADENCE_NS
    command_cadence_ns: int = CANONICAL_COMMAND_CADENCE_NS
    command_count: int = CANONICAL_COMMAND_COUNT
    maximum_pose_sequence_length: int = CANONICAL_MAX_POSE_SEQUENCE_LENGTH
    maximum_outcome_duration_ns: int = CANONICAL_MAX_OUTCOME_DURATION_NS
    reset_free_support_sha256: str = field(
        default_factory=lambda: ConfigurationMorphology().free_support_sha256
    )
    runner_producer_sha256: str | None = CANONICAL_RUNNER_PRODUCER_SHA256
    reset_producer_sha256: str | None = CANONICAL_RESET_PRODUCER_SHA256
    runner_outcome_protocol_sha256: str | None = (
        CANONICAL_RUNNER_OUTCOME_PROTOCOL_SHA256
    )
    reset_clearance_protocol_sha256: str | None = (
        CANONICAL_RESET_CLEARANCE_PROTOCOL_SHA256
    )
    physical_promotion_ready: bool = False
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected = {
            "geometry_contract_sha256": CANONICAL_GEOMETRY_CONTRACT_SHA256,
            "geometry_file_sha256": CANONICAL_GEOMETRY_FILE_SHA256,
            "directional_policy_sha256": CANONICAL_DIRECTIONAL_POLICY_SHA256,
            "primitive_registry_sha256": CANONICAL_PRIMITIVE_REGISTRY_SHA256,
            "platform_manifest_sha256": CANONICAL_PLATFORM_MANIFEST_SHA256,
            "body_forward_m": CANONICAL_BODY_FORWARD_M,
            "body_rear_m": CANONICAL_BODY_REAR_M,
            "body_half_width_m": CANONICAL_BODY_HALF_WIDTH_M,
            "reset_footprint_radius_m": REGISTERED_FOOTPRINT_RADIUS_M,
            "physical_cell_size_m": REGISTERED_PHYSICAL_CELL_SIZE_M,
            "maximum_translation_substep_m": CANONICAL_MAX_TRANSLATION_SUBSTEP_M,
            "maximum_angular_substep_rad": CANONICAL_MAX_ANGULAR_SUBSTEP_RAD,
            "pose_sample_cadence_ns": CANONICAL_POSE_SAMPLE_CADENCE_NS,
            "command_cadence_ns": CANONICAL_COMMAND_CADENCE_NS,
            "command_count": CANONICAL_COMMAND_COUNT,
            "maximum_pose_sequence_length": CANONICAL_MAX_POSE_SEQUENCE_LENGTH,
            "maximum_outcome_duration_ns": CANONICAL_MAX_OUTCOME_DURATION_NS,
            "reset_free_support_sha256": (
                ConfigurationMorphology().free_support_sha256
            ),
            "runner_producer_sha256": CANONICAL_RUNNER_PRODUCER_SHA256,
            "reset_producer_sha256": CANONICAL_RESET_PRODUCER_SHA256,
            "runner_outcome_protocol_sha256": (
                CANONICAL_RUNNER_OUTCOME_PROTOCOL_SHA256
            ),
            "reset_clearance_protocol_sha256": (
                CANONICAL_RESET_CLEARANCE_PROTOCOL_SHA256
            ),
            "physical_promotion_ready": False,
        }
        for name, expected_value in expected.items():
            actual = getattr(self, name)
            if isinstance(expected_value, float):
                matches = math.isclose(
                    actual,
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=_EPS,
                )
            else:
                matches = actual == expected_value
            if not matches:
                raise ValueError(f"canonical executor/reset contract changed: {name}")
        object.__setattr__(self, "content_sha256", _canonical_sha256(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_g3_canonical_executor_reset_contract_v2",
            **{
                name: getattr(self, name)
                for name in self.__slots__
                if name != "content_sha256"
            },
        }

    def assert_integrity(self) -> None:
        if self.content_sha256 != _canonical_sha256(self.to_dict()):
            raise ValueError("canonical executor/reset contract was mutated")

    def __copy__(self) -> "CanonicalExecutorResetContract":
        raise TypeError("canonical executor/reset contract is non-copyable")

    def __deepcopy__(self, memo: object) -> "CanonicalExecutorResetContract":
        del memo
        raise TypeError("canonical executor/reset contract is non-copyable")


def canonical_executor_reset_contract() -> CanonicalExecutorResetContract:
    """Reopen and verify every installed non-authoritative source identity."""

    geometry_path = (REPOSITORY_ROOT / DEPLOYMENT_GEOMETRY_CONTRACT).resolve()
    primitive_path = REPOSITORY_ROOT / "config/go2_primitive_registry.yaml"
    platform_path = REPOSITORY_ROOT / "config/go2_platform_manifest.yaml"
    if _sha256_file(geometry_path) != CANONICAL_GEOMETRY_FILE_SHA256:
        raise ValueError("canonical deployment geometry file hash changed")
    if _sha256_file(primitive_path) != CANONICAL_PRIMITIVE_REGISTRY_SHA256:
        raise ValueError("canonical primitive registry hash changed")
    if _sha256_file(platform_path) != CANONICAL_PLATFORM_MANIFEST_SHA256:
        raise ValueError("canonical platform manifest hash changed")
    geometry = load_geometry_contract(
        geometry_path,
        repository_root=REPOSITORY_ROOT,
        verify_sources=True,
    )
    swept = geometry.swept_footprint
    if (
        geometry.sha256 != CANONICAL_GEOMETRY_CONTRACT_SHA256
        or swept.directional_policy_content_sha256
        != CANONICAL_DIRECTIONAL_POLICY_SHA256
        or not math.isclose(swept.forward_m, CANONICAL_BODY_FORWARD_M)
        or not math.isclose(swept.rear_m, CANONICAL_BODY_REAR_M)
        or not math.isclose(swept.half_width_m, CANONICAL_BODY_HALF_WIDTH_M)
        or geometry.kinematic_execution.maximum_translation_substep_m
        != CANONICAL_MAX_TRANSLATION_SUBSTEP_M
        or swept.strict_collision_representation
        != "directional_polygon_at_actual_yaw"
        or swept.planning_disc_radius_m != REGISTERED_FOOTPRINT_RADIUS_M
        or geometry.configuration_space.online_cell_size_m
        != REGISTERED_PHYSICAL_CELL_SIZE_M
        or geometry.physical_promotion_ready
    ):
        raise ValueError("canonical deployment executor geometry changed")
    contract = CanonicalExecutorResetContract()
    contract.assert_integrity()
    return contract


class PromotedExecutorResetEvidenceAdapterV1:
    """Non-instantiable placeholder until a reviewed runner owns issuance."""

    __slots__ = ()

    def __new__(cls, memory: RevisionedPhysicalMemory) -> object:
        if not isinstance(memory, RevisionedPhysicalMemory):
            raise TypeError("memory must be RevisionedPhysicalMemory")
        canonical_executor_reset_contract()
        raise PromotedExecutionAdmissionUnavailableError(
            "promoted executor/reset admission is unavailable: canonical runner "
            "and reset producer identities are not installed and deployment "
            "geometry physical_promotion_ready is false"
        )

    def __copy__(self) -> "PromotedExecutorResetEvidenceAdapterV1":
        raise TypeError("promoted executor/reset adapters are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "PromotedExecutorResetEvidenceAdapterV1":
        del memo
        raise TypeError("promoted executor/reset adapters are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("promoted executor/reset adapters are non-serializable")


__all__ = [
    "CANONICAL_COMMAND_CADENCE_NS",
    "CANONICAL_COMMAND_COUNT",
    "CANONICAL_MAX_ANGULAR_SUBSTEP_RAD",
    "CANONICAL_MAX_OUTCOME_DURATION_NS",
    "CANONICAL_MAX_POSE_SEQUENCE_LENGTH",
    "CANONICAL_MAX_TRANSLATION_SUBSTEP_M",
    "CANONICAL_POSE_SAMPLE_CADENCE_NS",
    "CanonicalExecutorResetContract",
    "CanonicalRunnerPoseSample",
    "PromotedExecutionAdmissionUnavailableError",
    "PromotedExecutorResetEvidenceAdapterV1",
    "canonical_executor_reset_contract",
    "validate_canonical_runner_pose_sequence",
]
