"""Synthetic development boundary for native-0.05 m learned V4 evidence.

The runner side exposes only raw logits and calibrated query/ray geometry.  The
adapter owns thresholding and conservative projection into an exact live G3 V2
physical lattice.  Production, hardware execution, and promotion authority are
deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    TransactionReceipt,
    TransactionRejectedError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    PHYSICAL_CELL_SIZE_M,
    PROFILE_SHA256,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


Cell = tuple[int, int]
XY = tuple[float, float]
Transform2 = tuple[float, float, float]

PRODUCTION_NATIVE_V4_RUNNER = None
PRODUCTION_V4_CHECKPOINT_FILE_SHA256 = None
PRODUCTION_G2_REPORT_FILE_SHA256 = None
PRODUCTION_V4_CALIBRATION_SHA256 = None
PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER = None


class NativeLearnedProjectionError(ValueError):
    """Base error for the synthetic learned-projection boundary."""


class NativeLearnedProjectionBindingError(NativeLearnedProjectionError):
    """An artifact is foreign, stale, copied, or bound to another identity."""


class NativeLearnedProjectionRejectedError(NativeLearnedProjectionError):
    """Raw runner evidence is inadmissible under the frozen contract."""


class NativeLearnedProjectionReplayError(NativeLearnedProjectionError):
    """A single-use outcome or development transaction was replayed."""


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


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
        raise TypeError(f"{name} must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain two integers")
    return int(value[0]), int(value[1])


def _xy(value: object, name: str) -> XY:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must contain two coordinates")
    return _finite(value[0], name), _finite(value[1], name)


def _transform(value: object, name: str) -> Transform2:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"{name} must contain x, y, and yaw")
    return (
        _finite(value[0], name),
        _finite(value[1], name),
        _finite(value[2], name),
    )


def _shape(value: object, name: str) -> Cell:
    result = _cell(value, name)
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cells_json(cells: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(cells)]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class FrozenNativeLearnedProjectionCalibrationV1:
    source_shape: Cell = (128, 128)
    source_origin_forward_left_m: XY = (-1.0, -3.2)
    source_cell_size_m: float = PHYSICAL_CELL_SIZE_M
    support_offsets_cell_fraction: tuple[XY, ...] = (
        (0.5, 0.5),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
    )
    known_probability_threshold: float = 0.90
    free_given_known_probability_threshold: float = 0.90
    occupied_given_known_probability_threshold: float = 0.90
    ray_depth_min_m: float = 0.05
    ray_depth_max_m: float = 6.45
    covariance_diagonal_max: tuple[float, float, float] = (
        0.0025,
        0.0025,
        0.0004,
    )
    pose_sigma_multipliers: tuple[Transform2, ...] = (
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    camera_local_uncertainty_transforms: tuple[Transform2, ...] = (
        (0.0, 0.0, 0.0),
    )
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        shape = _shape(self.source_shape, "source_shape")
        origin = _xy(
            self.source_origin_forward_left_m,
            "source_origin_forward_left_m",
        )
        cell_size = _finite(self.source_cell_size_m, "source_cell_size_m")
        if not math.isclose(
            cell_size,
            PHYSICAL_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("frozen learned source must be native 0.05 m")
        offsets = tuple(
            _xy(offset, "support offset")
            for offset in self.support_offsets_cell_fraction
        )
        if offsets != (
            (0.5, 0.5),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
        ):
            raise ValueError("frozen V4 five-point support geometry changed")
        known = _unit(
            self.known_probability_threshold,
            "known_probability_threshold",
        )
        free = _unit(
            self.free_given_known_probability_threshold,
            "free_given_known_probability_threshold",
        )
        occupied = _unit(
            self.occupied_given_known_probability_threshold,
            "occupied_given_known_probability_threshold",
        )
        if min(known, free, occupied) <= 0.5:
            raise ValueError("frozen learned thresholds must exceed chance")
        depth_min = _finite(self.ray_depth_min_m, "ray_depth_min_m")
        depth_max = _finite(self.ray_depth_max_m, "ray_depth_max_m")
        if depth_min <= 0.0 or depth_max <= depth_min:
            raise ValueError("frozen ray depth interval is invalid")
        covariance = tuple(
            _finite(value, "covariance_diagonal_max")
            for value in self.covariance_diagonal_max
        )
        if len(covariance) != 3 or any(value < 0.0 for value in covariance):
            raise ValueError("covariance_diagonal_max must contain three limits")
        sigma = tuple(
            _transform(value, "pose sigma multiplier")
            for value in self.pose_sigma_multipliers
        )
        if sigma != (
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ):
            raise ValueError("registered seven-point pose uncertainty set changed")
        camera = tuple(
            _transform(value, "camera uncertainty transform")
            for value in self.camera_local_uncertainty_transforms
        )
        if not camera:
            raise ValueError("camera uncertainty transform set cannot be empty")
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "source_origin_forward_left_m", origin)
        object.__setattr__(self, "source_cell_size_m", cell_size)
        object.__setattr__(self, "support_offsets_cell_fraction", offsets)
        object.__setattr__(self, "known_probability_threshold", known)
        object.__setattr__(self, "free_given_known_probability_threshold", free)
        object.__setattr__(
            self,
            "occupied_given_known_probability_threshold",
            occupied,
        )
        object.__setattr__(self, "ray_depth_min_m", depth_min)
        object.__setattr__(self, "ray_depth_max_m", depth_max)
        object.__setattr__(self, "covariance_diagonal_max", covariance)
        object.__setattr__(self, "pose_sigma_multipliers", sigma)
        object.__setattr__(self, "camera_local_uncertainty_transforms", camera)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_native_learned_projection_calibration_v1",
            "source_shape": list(self.source_shape),
            "source_origin_forward_left_m": list(
                self.source_origin_forward_left_m
            ),
            "source_cell_size_m": self.source_cell_size_m,
            "support_offsets_cell_fraction": [
                list(value) for value in self.support_offsets_cell_fraction
            ],
            "known_probability_threshold": self.known_probability_threshold,
            "free_given_known_probability_threshold": (
                self.free_given_known_probability_threshold
            ),
            "occupied_given_known_probability_threshold": (
                self.occupied_given_known_probability_threshold
            ),
            "ray_depth_min_m": self.ray_depth_min_m,
            "ray_depth_max_m": self.ray_depth_max_m,
            "covariance_diagonal_max": list(self.covariance_diagonal_max),
            "pose_sigma_multipliers": [
                list(value) for value in self.pose_sigma_multipliers
            ],
            "camera_local_uncertainty_transforms": [
                list(value) for value in self.camera_local_uncertainty_transforms
            ],
            "class_semantics": {
                "unknown": "insufficient runner-owned confidence or geometry",
                "free": "all five ground queries known and clear",
                "occupied": "ordered first-surface ray hit",
                "precedence": "occupied_over_free",
            },
            "threshold_owner": "frozen_calibration_only",
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "frozen learned calibration was mutated"
            )


FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1 = (
    FrozenNativeLearnedProjectionCalibrationV1()
)


@dataclass(frozen=True)
class NativeV4SourceGeometryV1:
    shape: Cell
    origin_forward_left_m: XY
    cell_size_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape(self.shape, "source shape"))
        object.__setattr__(
            self,
            "origin_forward_left_m",
            _xy(self.origin_forward_left_m, "source origin"),
        )
        cell_size = _finite(self.cell_size_m, "source cell_size_m")
        if cell_size <= 0.0:
            raise ValueError("source cell_size_m must be positive")
        object.__setattr__(self, "cell_size_m", cell_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "origin_forward_left_m": list(self.origin_forward_left_m),
            "cell_size_m": self.cell_size_m,
        }


def canonical_ground_query_xy_body_v1(
    geometry: NativeV4SourceGeometryV1,
    cell: Sequence[int],
) -> tuple[XY, ...]:
    normalized = _cell(cell, "source cell")
    if not (
        0 <= normalized[0] < geometry.shape[0]
        and 0 <= normalized[1] < geometry.shape[1]
    ):
        raise ValueError("source cell is outside source geometry")
    x0 = geometry.origin_forward_left_m[0] + normalized[0] * geometry.cell_size_m
    y0 = geometry.origin_forward_left_m[1] + normalized[1] * geometry.cell_size_m
    return tuple(
        (
            x0 + offset[0] * geometry.cell_size_m,
            y0 + offset[1] * geometry.cell_size_m,
        )
        for offset in FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1.support_offsets_cell_fraction
    )


@dataclass(frozen=True)
class RawGroundClearCellQueriesV1:
    source_cell: Cell
    clear_to_target_logits: tuple[float, ...]
    query_in_frustum: tuple[bool, ...]
    query_xy_body_m: tuple[XY, ...]
    query_uv_px: tuple[XY, ...]
    target_distance_m: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_cell", _cell(self.source_cell, "source_cell"))
        logits = tuple(
            _finite(value, "ground clear logit")
            for value in self.clear_to_target_logits
        )
        frustum = tuple(self.query_in_frustum)
        xy = tuple(_xy(value, "ground query xy") for value in self.query_xy_body_m)
        uv = tuple(_xy(value, "ground query uv") for value in self.query_uv_px)
        distance = tuple(
            _finite(value, "ground target distance")
            for value in self.target_distance_m
        )
        if not (
            len(logits)
            == len(frustum)
            == len(xy)
            == len(uv)
            == len(distance)
            == 5
        ):
            raise ValueError("raw ground tensor requires exactly five queries per cell")
        if any(type(value) is not bool for value in frustum):
            raise TypeError("query_in_frustum entries must be booleans")
        if any(value <= 0.0 for value in distance):
            raise ValueError("ground target distances must be positive")
        object.__setattr__(self, "clear_to_target_logits", logits)
        object.__setattr__(self, "query_in_frustum", frustum)
        object.__setattr__(self, "query_xy_body_m", xy)
        object.__setattr__(self, "query_uv_px", uv)
        object.__setattr__(self, "target_distance_m", distance)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_cell": list(self.source_cell),
            "clear_to_target_logits": list(self.clear_to_target_logits),
            "query_in_frustum": list(self.query_in_frustum),
            "query_xy_body_m": [list(value) for value in self.query_xy_body_m],
            "query_uv_px": [list(value) for value in self.query_uv_px],
            "target_distance_m": list(self.target_distance_m),
        }


@dataclass(frozen=True)
class RawOrderedRayHitDepthV1:
    ray_index: int
    ray_origin_xy_body_m: XY
    ray_direction_xy_body: XY
    ordered_hit_logits: tuple[float, ...]
    ordered_depth_m: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ray_index",
            _nonnegative_int(self.ray_index, "ray_index"),
        )
        origin = _xy(self.ray_origin_xy_body_m, "ray origin")
        direction = _xy(self.ray_direction_xy_body, "ray direction")
        norm = math.hypot(*direction)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("ray direction must be unit length")
        logits = tuple(
            _finite(value, "ordered ray hit logit")
            for value in self.ordered_hit_logits
        )
        depths = tuple(
            _finite(value, "ordered ray depth") for value in self.ordered_depth_m
        )
        if not logits or len(logits) != len(depths):
            raise ValueError("ordered ray hit/depth tensors must be nonempty and aligned")
        if any(depth <= 0.0 for depth in depths) or any(
            second <= first for first, second in zip(depths, depths[1:])
        ):
            raise ValueError("ordered ray depths must be positive and strictly increasing")
        object.__setattr__(self, "ray_origin_xy_body_m", origin)
        object.__setattr__(self, "ray_direction_xy_body", direction)
        object.__setattr__(self, "ordered_hit_logits", logits)
        object.__setattr__(self, "ordered_depth_m", depths)

    def to_dict(self) -> dict[str, object]:
        return {
            "ray_index": self.ray_index,
            "ray_origin_xy_body_m": list(self.ray_origin_xy_body_m),
            "ray_direction_xy_body": list(self.ray_direction_xy_body),
            "ordered_hit_logits": list(self.ordered_hit_logits),
            "ordered_depth_m": list(self.ordered_depth_m),
        }


@dataclass(frozen=True)
class SyntheticNativeV4RawOutcomeV1:
    outcome_sequence: int
    observation_id: str
    source_derivation: str
    source_geometry: NativeV4SourceGeometryV1
    ground_clear_query_tensor: tuple[RawGroundClearCellQueriesV1, ...]
    ordered_ray_hit_depth_tensor: tuple[RawOrderedRayHitDepthV1, ...]
    runner_execution_identity_sha256: str
    inference_implementation_sha256: str
    projection_implementation_sha256: str
    access_ledger_source_sha256: str
    checkpoint_file_sha256: str
    g2_report_file_sha256: str
    calibration_sha256: str
    rgb_frame_id: str
    rgb_frame_sha256: str
    raw_outcome_file_sha256: str
    pose: PoseProvenance
    physical_map_frame: MapFrameIdentity
    configuration_map_frame: MapFrameIdentity
    physical_shape: Cell
    configuration_shape: Cell
    physical_revision: int
    configuration_revision: int
    physical_content_sha256: str
    configuration_snapshot_sha256: str
    projection_source_sha256: str
    _issuance_capability: object = field(repr=False, compare=False)
    raw_tensor_content_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.outcome_sequence, "outcome_sequence")
        for name in ("observation_id", "source_derivation", "rgb_frame_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if type(self.source_geometry) is not NativeV4SourceGeometryV1:
            raise TypeError("source_geometry has the wrong type")
        ground = tuple(self.ground_clear_query_tensor)
        rays = tuple(self.ordered_ray_hit_depth_tensor)
        if any(type(row) is not RawGroundClearCellQueriesV1 for row in ground):
            raise TypeError("ground_clear_query_tensor contains untyped rows")
        if any(type(row) is not RawOrderedRayHitDepthV1 for row in rays):
            raise TypeError("ordered_ray_hit_depth_tensor contains untyped rows")
        if not ground and not rays:
            raise ValueError("raw V4 outcome tensors cannot both be empty")
        if len({row.source_cell for row in ground}) != len(ground):
            raise ValueError("raw ground tensor repeats a source cell")
        if len({row.ray_index for row in rays}) != len(rays):
            raise ValueError("raw ray tensor repeats a ray index")
        for name in (
            "runner_execution_identity_sha256",
            "inference_implementation_sha256",
            "projection_implementation_sha256",
            "access_ledger_source_sha256",
            "checkpoint_file_sha256",
            "g2_report_file_sha256",
            "calibration_sha256",
            "rgb_frame_sha256",
            "raw_outcome_file_sha256",
            "physical_content_sha256",
            "configuration_snapshot_sha256",
            "projection_source_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.pose) is not PoseProvenance:
            raise TypeError("pose must be PoseProvenance")
        if type(self.physical_map_frame) is not MapFrameIdentity or type(
            self.configuration_map_frame
        ) is not MapFrameIdentity:
            raise TypeError("raw outcome frames have the wrong type")
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape,
            "configuration_shape",
        )
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        if self._issuance_capability is None:
            raise TypeError("raw V4 outcome requires an issuance capability")
        object.__setattr__(self, "ground_clear_query_tensor", ground)
        object.__setattr__(self, "ordered_ray_hit_depth_tensor", rays)
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        tensor_hash = _sha256(
            {
                "schema": "lewm_g3_native_v4_raw_tensor_payload_v1",
                "source_geometry": self.source_geometry.to_dict(),
                "ground_clear_query_tensor": [row.to_dict() for row in ground],
                "ordered_ray_hit_depth_tensor": [row.to_dict() for row in rays],
            }
        )
        object.__setattr__(self, "raw_tensor_content_sha256", tensor_hash)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_synthetic_native_v4_raw_outcome_v1",
            "outcome_sequence": self.outcome_sequence,
            "observation_id": self.observation_id,
            "source_derivation": self.source_derivation,
            "source_geometry": self.source_geometry.to_dict(),
            "ground_clear_query_tensor": [
                row.to_dict() for row in self.ground_clear_query_tensor
            ],
            "ordered_ray_hit_depth_tensor": [
                row.to_dict() for row in self.ordered_ray_hit_depth_tensor
            ],
            "raw_tensor_content_sha256": self.raw_tensor_content_sha256,
            "runner_execution_identity_sha256": (
                self.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": self.inference_implementation_sha256,
            "projection_implementation_sha256": (
                self.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": self.access_ledger_source_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "g2_report_file_sha256": self.g2_report_file_sha256,
            "calibration_sha256": self.calibration_sha256,
            "rgb_frame_id": self.rgb_frame_id,
            "rgb_frame_sha256": self.rgb_frame_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "pose": self.pose.to_dict(),
            "pose_sha256": self.pose.content_sha256,
            "physical_map_frame": self.physical_map_frame.to_dict(),
            "physical_map_frame_sha256": self.physical_map_frame.content_sha256,
            "configuration_map_frame": self.configuration_map_frame.to_dict(),
            "configuration_map_frame_sha256": (
                self.configuration_map_frame.content_sha256
            ),
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "physical_content_sha256": self.physical_content_sha256,
            "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
            "projection_source_sha256": self.projection_source_sha256,
            "production_eligible": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        expected_tensor = _sha256(
            {
                "schema": "lewm_g3_native_v4_raw_tensor_payload_v1",
                "source_geometry": self.source_geometry.to_dict(),
                "ground_clear_query_tensor": [
                    row.to_dict() for row in self.ground_clear_query_tensor
                ],
                "ordered_ray_hit_depth_tensor": [
                    row.to_dict() for row in self.ordered_ray_hit_depth_tensor
                ],
            }
        )
        if (
            self.raw_tensor_content_sha256 != expected_tensor
            or self.content_sha256 != _sha256(self.to_dict(False))
        ):
            raise NativeLearnedProjectionBindingError(
                "synthetic raw V4 outcome was mutated"
            )


class SyntheticNativeV4RunnerV1:
    """Exact-object synthetic issuer for raw V4 inference output."""

    __slots__ = (
        "runner_execution_identity_sha256",
        "inference_implementation_sha256",
        "projection_implementation_sha256",
        "access_ledger_source_sha256",
        "checkpoint_file_sha256",
        "g2_report_file_sha256",
        "calibration_sha256",
        "_capability",
        "_issued",
        "_consumed",
        "_sequence",
    )

    def __init__(
        self,
        *,
        runner_execution_identity_sha256: str,
        inference_implementation_sha256: str,
        projection_implementation_sha256: str,
        access_ledger_source_sha256: str,
        checkpoint_file_sha256: str,
        g2_report_file_sha256: str,
        calibration_sha256: str,
        _synthetic_test_fixture: bool = False,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError("native V4 raw runner is synthetic-only")
        for name in (
            "runner_execution_identity_sha256",
            "inference_implementation_sha256",
            "projection_implementation_sha256",
            "access_ledger_source_sha256",
            "checkpoint_file_sha256",
            "g2_report_file_sha256",
            "calibration_sha256",
        ):
            setattr(self, name, _require_sha256(locals()[name], name))
        self._capability = object()
        self._issued: dict[
            int,
            tuple[SyntheticNativeV4RawOutcomeV1, str],
        ] = {}
        self._consumed: set[int] = set()
        self._sequence = 0

    def __copy__(self) -> "SyntheticNativeV4RunnerV1":
        raise TypeError("synthetic native V4 runner is non-copyable")

    def __deepcopy__(self, memo: object) -> "SyntheticNativeV4RunnerV1":
        del memo
        raise TypeError("synthetic native V4 runner is non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("synthetic native V4 runner is non-serializable")

    def issue(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        pose: PoseProvenance,
        source_geometry: NativeV4SourceGeometryV1,
        ground_clear_query_tensor: Iterable[RawGroundClearCellQueriesV1],
        ordered_ray_hit_depth_tensor: Iterable[RawOrderedRayHitDepthV1],
        rgb_frame_id: str,
        rgb_frame_sha256: str,
        raw_outcome_file_sha256: str,
        source_derivation: str = "native_raw_v4_0p05",
    ) -> SyntheticNativeV4RawOutcomeV1:
        if type(snapshot) is not TwoResolutionConfigurationSnapshotV2:
            raise TypeError("snapshot must be TwoResolutionConfigurationSnapshotV2")
        self._sequence += 1
        outcome = SyntheticNativeV4RawOutcomeV1(
            outcome_sequence=self._sequence,
            observation_id=f"synthetic-native-v4:{self._sequence}:{rgb_frame_id}",
            source_derivation=source_derivation,
            source_geometry=source_geometry,
            ground_clear_query_tensor=tuple(ground_clear_query_tensor),
            ordered_ray_hit_depth_tensor=tuple(ordered_ray_hit_depth_tensor),
            runner_execution_identity_sha256=(
                self.runner_execution_identity_sha256
            ),
            inference_implementation_sha256=self.inference_implementation_sha256,
            projection_implementation_sha256=(
                self.projection_implementation_sha256
            ),
            access_ledger_source_sha256=self.access_ledger_source_sha256,
            checkpoint_file_sha256=self.checkpoint_file_sha256,
            g2_report_file_sha256=self.g2_report_file_sha256,
            calibration_sha256=self.calibration_sha256,
            rgb_frame_id=rgb_frame_id,
            rgb_frame_sha256=rgb_frame_sha256,
            raw_outcome_file_sha256=raw_outcome_file_sha256,
            pose=pose,
            physical_map_frame=snapshot.physical_map_frame,
            configuration_map_frame=snapshot.configuration_map_frame,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_content_sha256=snapshot.physical_content_sha256,
            configuration_snapshot_sha256=snapshot.content_sha256,
            projection_source_sha256=snapshot.projection_source_sha256,
            _issuance_capability=self._capability,
        )
        self._issued[id(outcome)] = (outcome, outcome.content_sha256)
        return outcome

    def assert_issued(
        self,
        outcome: SyntheticNativeV4RawOutcomeV1,
        *,
        consume: bool = False,
    ) -> None:
        if type(outcome) is not SyntheticNativeV4RawOutcomeV1:
            raise TypeError("outcome must be SyntheticNativeV4RawOutcomeV1")
        issued = self._issued.get(id(outcome))
        if issued is None or issued[0] is not outcome:
            raise NativeLearnedProjectionBindingError(
                "raw V4 outcome is not the exact live object issued by this runner"
            )
        outcome.assert_integrity()
        if outcome.content_sha256 != issued[1]:
            raise NativeLearnedProjectionBindingError(
                "raw V4 outcome differs from its instance-issued content"
            )
        if id(outcome) in self._consumed:
            raise NativeLearnedProjectionReplayError("raw V4 outcome was already consumed")
        if consume:
            self._consumed.add(id(outcome))


def _rotate(point: XY, yaw: float) -> XY:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
    )


def _apply_transform(point: XY, transform: Transform2) -> XY:
    rotated = _rotate(point, transform[2])
    return rotated[0] + transform[0], rotated[1] + transform[1]


def _inverse_transform(point: XY, transform: Transform2) -> XY:
    translated = point[0] - transform[0], point[1] - transform[1]
    return _rotate(translated, -transform[2])


def _compose_transform(parent: Transform2, local: Transform2) -> Transform2:
    translated = _rotate((local[0], local[1]), parent[2])
    return (
        parent[0] + translated[0],
        parent[1] + translated[1],
        parent[2] + local[2],
    )


def _polygon_area(polygon: Sequence[XY]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        math.fsum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    )


def _clip_axis(
    polygon: Sequence[XY],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> tuple[XY, ...]:
    if not polygon:
        return ()

    def inside(point: XY) -> bool:
        return (
            point[axis] >= boundary - 1e-12
            if keep_greater
            else point[axis] <= boundary + 1e-12
        )

    result: list[XY] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(delta) <= 1e-15:
                intersection = current
            else:
                ratio = (boundary - previous[axis]) / delta
                other = 1 - axis
                coordinates = [0.0, 0.0]
                coordinates[axis] = boundary
                coordinates[other] = (
                    previous[other]
                    + ratio * (current[other] - previous[other])
                )
                intersection = coordinates[0], coordinates[1]
            result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return tuple(result)


def _clip_polygon_to_square(
    polygon: Sequence[XY],
    minimum_xy: XY,
    maximum_xy: XY,
) -> tuple[XY, ...]:
    result = tuple(polygon)
    result = _clip_axis(
        result,
        axis=0,
        boundary=minimum_xy[0],
        keep_greater=True,
    )
    result = _clip_axis(
        result,
        axis=0,
        boundary=maximum_xy[0],
        keep_greater=False,
    )
    result = _clip_axis(
        result,
        axis=1,
        boundary=minimum_xy[1],
        keep_greater=True,
    )
    return _clip_axis(
        result,
        axis=1,
        boundary=maximum_xy[1],
        keep_greater=False,
    )


def _point_in_closed_square(point: XY, minimum_xy: XY, maximum_xy: XY) -> bool:
    return bool(
        minimum_xy[0] - 1e-12 <= point[0] <= maximum_xy[0] + 1e-12
        and minimum_xy[1] - 1e-12 <= point[1] <= maximum_xy[1] + 1e-12
    )


def _source_square(
    geometry: NativeV4SourceGeometryV1,
    cell: Cell,
) -> tuple[XY, XY]:
    x0 = geometry.origin_forward_left_m[0] + cell[0] * geometry.cell_size_m
    y0 = geometry.origin_forward_left_m[1] + cell[1] * geometry.cell_size_m
    return (x0, y0), (x0 + geometry.cell_size_m, y0 + geometry.cell_size_m)


def _destination_square(frame: MapFrameIdentity, cell: Cell) -> tuple[XY, ...]:
    x0 = frame.origin_xy_m[0] + cell[0] * frame.cell_size_m
    y0 = frame.origin_xy_m[1] + cell[1] * frame.cell_size_m
    size = frame.cell_size_m
    return ((x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size))


def _point_covered_by_source_union(
    point: XY,
    geometry: NativeV4SourceGeometryV1,
    source_cells: frozenset[Cell],
) -> bool:
    for cell in source_cells:
        minimum, maximum = _source_square(geometry, cell)
        if _point_in_closed_square(point, minimum, maximum):
            return True
    return False


def _polygon_covered_by_source_union(
    polygon: tuple[XY, ...],
    geometry: NativeV4SourceGeometryV1,
    source_cells: frozenset[Cell],
) -> bool:
    if not source_cells or any(
        not _point_covered_by_source_union(point, geometry, source_cells)
        for point in polygon
    ):
        return False
    covered_area = math.fsum(
        _polygon_area(
            _clip_polygon_to_square(
                polygon,
                *_source_square(geometry, cell),
            )
        )
        for cell in source_cells
    )
    return math.isclose(
        covered_area,
        _polygon_area(polygon),
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def _orientation(first: XY, second: XY, third: XY) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _point_on_segment(point: XY, start: XY, end: XY) -> bool:
    return bool(
        abs(_orientation(start, end, point)) <= 1e-12
        and min(start[0], end[0]) - 1e-12
        <= point[0]
        <= max(start[0], end[0]) + 1e-12
        and min(start[1], end[1]) - 1e-12
        <= point[1]
        <= max(start[1], end[1]) + 1e-12
    )


def _segments_intersect_closed(a: XY, b: XY, c: XY, d: XY) -> bool:
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if values[0] * values[1] < -1e-12 and values[2] * values[3] < -1e-12:
        return True
    return bool(
        (abs(values[0]) <= 1e-12 and _point_on_segment(c, a, b))
        or (abs(values[1]) <= 1e-12 and _point_on_segment(d, a, b))
        or (abs(values[2]) <= 1e-12 and _point_on_segment(a, c, d))
        or (abs(values[3]) <= 1e-12 and _point_on_segment(b, c, d))
    )


def _point_in_convex_closed(point: XY, polygon: Sequence[XY]) -> bool:
    orientations = [
        _orientation(polygon[index], polygon[(index + 1) % len(polygon)], point)
        for index in range(len(polygon))
    ]
    return bool(
        all(value >= -1e-12 for value in orientations)
        or all(value <= 1e-12 for value in orientations)
    )


def _polygons_intersect_closed(first: Sequence[XY], second: Sequence[XY]) -> bool:
    if any(_point_in_convex_closed(point, second) for point in first) or any(
        _point_in_convex_closed(point, first) for point in second
    ):
        return True
    return any(
        _segments_intersect_closed(
            first[index],
            first[(index + 1) % len(first)],
            second[other],
            second[(other + 1) % len(second)],
        )
        for index in range(len(first))
        for other in range(len(second))
    )


def _candidate_cells_for_polygon(
    frame: MapFrameIdentity,
    shape: Cell,
    polygon: Sequence[XY],
) -> frozenset[Cell]:
    minimum = frame.world_to_cell(
        (min(point[0] for point in polygon), min(point[1] for point in polygon))
    )
    maximum = frame.world_to_cell(
        (max(point[0] for point in polygon), max(point[1] for point in polygon))
    )
    return frozenset(
        (x, y)
        for x in range(minimum[0] - 1, maximum[0] + 2)
        for y in range(minimum[1] - 1, maximum[1] + 2)
        if 0 <= x < shape[0] and 0 <= y < shape[1]
    )


def _closed_point_supercover(
    frame: MapFrameIdentity,
    shape: Cell,
    point: XY,
) -> frozenset[Cell]:
    coordinates = tuple(
        (point[index] - frame.origin_xy_m[index]) / frame.cell_size_m
        for index in range(2)
    )
    axes: list[tuple[int, ...]] = []
    for coordinate in coordinates:
        rounded = int(round(coordinate))
        if math.isclose(coordinate, rounded, rel_tol=0.0, abs_tol=1e-9):
            axes.append((rounded - 1, rounded))
        else:
            axes.append((int(math.floor(coordinate)),))
    return frozenset(
        (x, y)
        for x in axes[0]
        for y in axes[1]
        if 0 <= x < shape[0] and 0 <= y < shape[1]
    )


@dataclass(frozen=True)
class NativeLearnedPhysicalProjectionReceiptV1:
    raw_outcome_sha256: str
    raw_tensor_content_sha256: str
    calibration_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_shape: Cell
    configuration_shape: Cell
    transform_uncertainty_set: tuple[Transform2, ...]
    transform_uncertainty_set_sha256: str
    free_cells: frozenset[Cell]
    occupied_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "raw_outcome_sha256",
            "raw_tensor_content_sha256",
            "calibration_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "transform_uncertainty_set_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _positive_int(self.configuration_revision, "configuration_revision")
        physical_shape = _shape(self.physical_shape, "physical_shape")
        configuration_shape = _shape(
            self.configuration_shape,
            "configuration_shape",
        )
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("projection receipt requires exact 2:1 shapes")
        transforms = tuple(
            _transform(value, "uncertainty transform")
            for value in self.transform_uncertainty_set
        )
        if not transforms:
            raise ValueError("projection receipt requires finite transforms")
        expected_transform_hash = _sha256(
            {
                "schema": "lewm_g3_native_projection_transform_set_v1",
                "transforms_xy_yaw": [list(value) for value in transforms],
            }
        )
        if self.transform_uncertainty_set_sha256 != expected_transform_hash:
            raise NativeLearnedProjectionBindingError(
                "projection transform uncertainty hash changed"
            )
        free = frozenset(_cell(value, "FREE cell") for value in self.free_cells)
        occupied = frozenset(
            _cell(value, "OCCUPIED cell") for value in self.occupied_cells
        )
        unknown = frozenset(
            _cell(value, "UNKNOWN cell") for value in self.unknown_cells
        )
        if free & occupied or free & unknown or occupied & unknown:
            raise ValueError("projection receipt labels must be disjoint")
        if any(
            not (0 <= cell[0] < physical_shape[0] and 0 <= cell[1] < physical_shape[1])
            for cell in free | occupied | unknown
        ):
            raise ValueError("projected physical cell is outside physical_shape")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "configuration_shape", configuration_shape)
        object.__setattr__(self, "transform_uncertainty_set", transforms)
        object.__setattr__(self, "free_cells", free)
        object.__setattr__(self, "occupied_cells", occupied)
        object.__setattr__(self, "unknown_cells", unknown)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_native_learned_physical_projection_receipt_v1",
            "raw_outcome_sha256": self.raw_outcome_sha256,
            "raw_tensor_content_sha256": self.raw_tensor_content_sha256,
            "calibration_sha256": self.calibration_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "physical_shape": list(self.physical_shape),
            "configuration_shape": list(self.configuration_shape),
            "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
            "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "transform_uncertainty_set": [
                list(value) for value in self.transform_uncertainty_set
            ],
            "transform_uncertainty_set_sha256": (
                self.transform_uncertainty_set_sha256
            ),
            "free_cells": _cells_json(self.free_cells),
            "occupied_cells": _cells_json(self.occupied_cells),
            "unknown_cells": _cells_json(self.unknown_cells),
            "occupied_precedes_free": True,
            "free_rule": "closed_square_covered_for_every_transform",
            "occupied_rule": "closed_point_union_supercover_all_transforms",
            "development_only": True,
            "hardware_execution_authorized": False,
            "production_promotion_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "learned projection receipt was mutated"
            )


@dataclass(frozen=True)
class QualifiedLearnedPhysicalDevelopmentAdmissionV1:
    admission_kind: str
    admission_id_sha256: str
    adapter_contract_sha256: str
    source_outcome_sha256: str
    projection_receipt_sha256: str
    physical_transaction_sha256: str
    observation_id: str
    observation_payload_sha256: str
    observation_producer_sha256: str
    pose_sha256: str
    memory_config_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision_before: int
    configuration_revision: int
    runner_execution_identity_sha256: str
    inference_implementation_sha256: str
    projection_implementation_sha256: str
    access_ledger_source_sha256: str
    checkpoint_file_sha256: str
    g2_report_file_sha256: str
    calibration_sha256: str
    rgb_frame_sha256: str
    raw_outcome_file_sha256: str
    retracts_observation_id: str | None
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.admission_kind not in {"projection", "retraction"}:
            raise ValueError("unsupported learned development admission kind")
        for name in (
            "admission_id_sha256",
            "adapter_contract_sha256",
            "source_outcome_sha256",
            "projection_receipt_sha256",
            "physical_transaction_sha256",
            "observation_payload_sha256",
            "observation_producer_sha256",
            "pose_sha256",
            "memory_config_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "runner_execution_identity_sha256",
            "inference_implementation_sha256",
            "projection_implementation_sha256",
            "access_ledger_source_sha256",
            "checkpoint_file_sha256",
            "g2_report_file_sha256",
            "calibration_sha256",
            "rgb_frame_sha256",
            "raw_outcome_file_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.observation_id) is not str or not self.observation_id:
            raise ValueError("observation_id must be nonempty")
        _nonnegative_int(self.physical_revision_before, "physical_revision_before")
        _positive_int(self.configuration_revision, "configuration_revision")
        if self.admission_kind == "projection":
            if self.retracts_observation_id is not None:
                raise ValueError("projection admission cannot retract an observation")
        elif (
            type(self.retracts_observation_id) is not str
            or not self.retracts_observation_id
        ):
            raise ValueError("retraction admission requires an exact observation ID")
        expected_id = _sha256(
            {
                "schema": "lewm_g3_qualified_learned_development_admission_id_v1",
                "kind": self.admission_kind,
                "adapter_contract_sha256": self.adapter_contract_sha256,
                "source_outcome_sha256": self.source_outcome_sha256,
                "projection_receipt_sha256": self.projection_receipt_sha256,
                "physical_transaction_sha256": self.physical_transaction_sha256,
                "observation_id": self.observation_id,
                "memory_revision_before": self.physical_revision_before,
                "retracts_observation_id": self.retracts_observation_id,
            }
        )
        if self.admission_id_sha256 != expected_id:
            raise NativeLearnedProjectionBindingError(
                "learned development admission identity changed"
            )
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_qualified_learned_development_admission_v1",
            "admission_kind": self.admission_kind,
            "admission_id_sha256": self.admission_id_sha256,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "source_outcome_sha256": self.source_outcome_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "physical_transaction_sha256": self.physical_transaction_sha256,
            "observation_id": self.observation_id,
            "observation_payload_sha256": self.observation_payload_sha256,
            "observation_producer_sha256": self.observation_producer_sha256,
            "pose_sha256": self.pose_sha256,
            "memory_config_sha256": self.memory_config_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "physical_revision_before": self.physical_revision_before,
            "configuration_revision": self.configuration_revision,
            "runner_execution_identity_sha256": (
                self.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": self.inference_implementation_sha256,
            "projection_implementation_sha256": (
                self.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": self.access_ledger_source_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "g2_report_file_sha256": self.g2_report_file_sha256,
            "calibration_sha256": self.calibration_sha256,
            "rgb_frame_sha256": self.rgb_frame_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "retracts_observation_id": self.retracts_observation_id,
            "development_only": True,
            "hardware_execution_authorized": False,
            "production_promotion_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "learned development admission was mutated"
            )


@dataclass(frozen=True)
class QualifiedLearnedPhysicalDevelopmentTransactionV1:
    admission: QualifiedLearnedPhysicalDevelopmentAdmissionV1
    pose: PoseProvenance
    projection_receipt: NativeLearnedPhysicalProjectionReceiptV1
    retracts_observation_id: str | None
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.admission) is not QualifiedLearnedPhysicalDevelopmentAdmissionV1:
            raise TypeError("development transaction admission has the wrong type")
        if type(self.pose) is not PoseProvenance:
            raise TypeError("development transaction pose has the wrong type")
        if type(self.projection_receipt) is not NativeLearnedPhysicalProjectionReceiptV1:
            raise TypeError("development transaction projection receipt has the wrong type")
        self.admission.assert_integrity()
        self.projection_receipt.assert_integrity()
        if self.pose.content_sha256 != self.admission.pose_sha256:
            raise NativeLearnedProjectionBindingError(
                "development transaction pose binding changed"
            )
        if self.projection_receipt.content_sha256 != self.admission.projection_receipt_sha256:
            raise NativeLearnedProjectionBindingError(
                "development transaction projection receipt changed"
            )
        if self.retracts_observation_id != self.admission.retracts_observation_id:
            raise NativeLearnedProjectionBindingError(
                "development transaction retraction binding changed"
            )
        if self._issuance_capability is None:
            raise TypeError("development transaction requires an issuance capability")
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def observation_id(self) -> str:
        return self.admission.observation_id

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_qualified_learned_development_transaction_v1",
            "admission": self.admission.to_dict(),
            "pose": self.pose.to_dict(),
            "projection_receipt": self.projection_receipt.to_dict(),
            "retracts_observation_id": self.retracts_observation_id,
            "development_only": True,
            "hardware_execution_authorized": False,
            "production_promotion_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        self.admission.assert_integrity()
        self.projection_receipt.assert_integrity()
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "qualified learned development transaction was mutated"
            )

    def __copy__(self) -> "QualifiedLearnedPhysicalDevelopmentTransactionV1":
        raise TypeError("qualified learned development transactions are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "QualifiedLearnedPhysicalDevelopmentTransactionV1":
        del memo
        raise TypeError("qualified learned development transactions are non-copyable")


class NativeLearnedPhysicalProjectionAdapterV1:
    """Threshold, conservatively project, and commit one synthetic V4 outcome."""

    __slots__ = (
        "_memory",
        "_projection",
        "_runner",
        "_calibration",
        "_expected",
        "_adapter_contract_sha256",
        "_capability",
        "_issued",
        "_issued_snapshots",
        "_consumed",
        "_committed_by_observation",
        "_retraction_issued",
        "_sequence",
    )

    def __init__(
        self,
        *,
        memory: RevisionedPhysicalMemory,
        projection: TwoResolutionConfigurationProjectionV2,
        runner: SyntheticNativeV4RunnerV1,
        calibration: FrozenNativeLearnedProjectionCalibrationV1,
        runner_execution_identity_sha256: str,
        inference_implementation_sha256: str,
        projection_implementation_sha256: str,
        access_ledger_source_sha256: str,
        checkpoint_file_sha256: str,
        g2_report_file_sha256: str,
        camera_transform_sha256: str,
        _synthetic_test_fixture: bool = False,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError(
                "no production native learned physical projection is configured"
            )
        if type(memory) is not RevisionedPhysicalMemory:
            raise TypeError("memory must be RevisionedPhysicalMemory")
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("projection must be TwoResolutionConfigurationProjectionV2")
        if getattr(projection, "_memory", None) is not memory:
            raise NativeLearnedProjectionBindingError(
                "projection does not own the supplied physical memory"
            )
        if type(runner) is not SyntheticNativeV4RunnerV1:
            raise TypeError("runner must be SyntheticNativeV4RunnerV1")
        if calibration is not FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1:
            raise NativeLearnedProjectionBindingError(
                "adapter requires the exact frozen synthetic calibration object"
            )
        calibration.assert_integrity()
        expected = {
            "runner_execution_identity_sha256": _require_sha256(
                runner_execution_identity_sha256,
                "runner_execution_identity_sha256",
            ),
            "inference_implementation_sha256": _require_sha256(
                inference_implementation_sha256,
                "inference_implementation_sha256",
            ),
            "projection_implementation_sha256": _require_sha256(
                projection_implementation_sha256,
                "projection_implementation_sha256",
            ),
            "access_ledger_source_sha256": _require_sha256(
                access_ledger_source_sha256,
                "access_ledger_source_sha256",
            ),
            "checkpoint_file_sha256": _require_sha256(
                checkpoint_file_sha256,
                "checkpoint_file_sha256",
            ),
            "g2_report_file_sha256": _require_sha256(
                g2_report_file_sha256,
                "g2_report_file_sha256",
            ),
            "camera_transform_sha256": _require_sha256(
                camera_transform_sha256,
                "camera_transform_sha256",
            ),
        }
        if memory.config.promoted_runtime:
            raise PermissionError("development learned projection cannot target promoted memory")
        if (
            not math.isclose(
                memory.map_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or memory.config.require_registered_lattice
            or memory.config.physical_projection_contract_sha256 != PROFILE_SHA256
            or memory.config.expected_camera_transform_sha256
            != expected["camera_transform_sha256"]
            or memory.config.pose_covariance_diagonal_limits
            != calibration.covariance_diagonal_max
        ):
            raise NativeLearnedProjectionBindingError(
                "development memory is not the native G3 V2 0.05 m contract"
            )
        runner_bindings = {
            name: getattr(runner, name)
            for name in (
                "runner_execution_identity_sha256",
                "inference_implementation_sha256",
                "projection_implementation_sha256",
                "access_ledger_source_sha256",
                "checkpoint_file_sha256",
                "g2_report_file_sha256",
            )
        }
        if runner_bindings != {
            name: expected[name] for name in runner_bindings
        } or runner.calibration_sha256 != calibration.content_sha256:
            raise NativeLearnedProjectionBindingError(
                "runner/checkpoint/G2/calibration identity differs from adapter"
            )
        self._memory = memory
        self._projection = projection
        self._runner = runner
        self._calibration = calibration
        self._expected = expected
        self._adapter_contract_sha256 = _sha256(
            {
                "schema": "lewm_g3_native_learned_projection_adapter_contract_v1",
                **expected,
                "calibration_sha256": calibration.content_sha256,
                "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
                "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
                "development_only": True,
                "hardware_execution_authorized": False,
                "production_promotion_authorized": False,
            }
        )
        self._capability = object()
        self._issued: dict[
            int,
            tuple[QualifiedLearnedPhysicalDevelopmentTransactionV1, str],
        ] = {}
        self._issued_snapshots: dict[int, TwoResolutionConfigurationSnapshotV2] = {}
        self._consumed: set[int] = set()
        self._committed_by_observation: dict[
            str,
            QualifiedLearnedPhysicalDevelopmentTransactionV1,
        ] = {}
        self._retraction_issued: set[str] = set()
        self._sequence = 0

    def __copy__(self) -> "NativeLearnedPhysicalProjectionAdapterV1":
        raise TypeError("native learned physical projection adapters are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "NativeLearnedPhysicalProjectionAdapterV1":
        del memo
        raise TypeError("native learned physical projection adapters are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("native learned physical projection adapters are non-serializable")

    @property
    def adapter_contract_sha256(self) -> str:
        return self._adapter_contract_sha256

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def _assert_outcome_bindings(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> None:
        self._calibration.assert_integrity()
        if (
            outcome.runner_execution_identity_sha256
            != self._expected["runner_execution_identity_sha256"]
            or outcome.inference_implementation_sha256
            != self._expected["inference_implementation_sha256"]
            or outcome.projection_implementation_sha256
            != self._expected["projection_implementation_sha256"]
            or outcome.access_ledger_source_sha256
            != self._expected["access_ledger_source_sha256"]
            or outcome.checkpoint_file_sha256
            != self._expected["checkpoint_file_sha256"]
            or outcome.g2_report_file_sha256
            != self._expected["g2_report_file_sha256"]
            or outcome.calibration_sha256 != self._calibration.content_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "runner/checkpoint/G2/calibration/source identity changed"
            )
        if (
            outcome.physical_map_frame is not snapshot.physical_map_frame
            or outcome.configuration_map_frame
            is not snapshot.configuration_map_frame
            or outcome.physical_map_frame.content_sha256
            != snapshot.physical_map_frame_sha256
            or outcome.configuration_map_frame.content_sha256
            != snapshot.configuration_map_frame_sha256
            or outcome.physical_shape != snapshot.physical_shape
            or outcome.configuration_shape != snapshot.configuration_shape
            or outcome.physical_shape
            != (
                2 * outcome.configuration_shape[0],
                2 * outcome.configuration_shape[1],
            )
            or outcome.physical_map_frame.origin_xy_m
            != outcome.configuration_map_frame.origin_xy_m
            or not math.isclose(
                outcome.physical_map_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                outcome.configuration_map_frame.cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome two-frame origin/shape identity changed"
            )
        if (
            outcome.physical_revision != snapshot.physical_revision
            or outcome.configuration_revision != snapshot.configuration_revision
            or outcome.physical_content_sha256 != snapshot.physical_content_sha256
            or outcome.configuration_snapshot_sha256 != snapshot.content_sha256
            or outcome.projection_source_sha256 != snapshot.projection_source_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome physical/configuration revision or source changed"
            )
        if (
            outcome.pose.source is not PoseSource.DEPLOYMENT_ODOMETRY
            or outcome.pose.frame_id != snapshot.physical_map_frame.frame_id
            or outcome.pose.camera_transform_sha256
            != self._expected["camera_transform_sha256"]
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome pose/camera identity changed"
            )
        geometry = outcome.source_geometry
        if (
            outcome.source_derivation != "native_raw_v4_0p05"
            or geometry.shape != self._calibration.source_shape
            or geometry.origin_forward_left_m
            != self._calibration.source_origin_forward_left_m
            or not math.isclose(
                geometry.cell_size_m,
                self._calibration.source_cell_size_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise NativeLearnedProjectionRejectedError(
                "raw source is not native V4 0.05 m logits; 0.10 m/upsampling rejects"
            )
        covariance = outcome.pose.covariance_xy_yaw
        if any(
            abs(covariance[row][column]) > 1e-12
            for row in range(3)
            for column in range(3)
            if row != column
        ):
            raise NativeLearnedProjectionRejectedError(
                "registered development uncertainty requires diagonal covariance"
            )
        diagonal = tuple(covariance[index][index] for index in range(3))
        if any(
            value > limit + 1e-12
            for value, limit in zip(
                diagonal,
                self._calibration.covariance_diagonal_max,
            )
        ):
            raise NativeLearnedProjectionRejectedError(
                "pose covariance exceeds frozen learned-projection envelope"
            )

    def _uncertainty_transforms(
        self,
        pose: PoseProvenance,
    ) -> tuple[Transform2, ...]:
        diagonal = tuple(pose.covariance_xy_yaw[index][index] for index in range(3))
        standard_deviation = tuple(math.sqrt(value) for value in diagonal)
        transforms: list[Transform2] = []
        for multiplier in self._calibration.pose_sigma_multipliers:
            pose_transform = (
                pose.mean_xy_yaw[0] + multiplier[0] * standard_deviation[0],
                pose.mean_xy_yaw[1] + multiplier[1] * standard_deviation[1],
                pose.mean_xy_yaw[2] + multiplier[2] * standard_deviation[2],
            )
            for camera_transform in (
                self._calibration.camera_local_uncertainty_transforms
            ):
                transforms.append(
                    _compose_transform(pose_transform, camera_transform)
                )
        unique: list[Transform2] = []
        for transform in transforms:
            if not any(
                all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
                    for left, right in zip(transform, existing)
                )
                for existing in unique
            ):
                unique.append(transform)
        return tuple(unique)

    def _free_source_cells(
        self,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> frozenset[Cell]:
        free: set[Cell] = set()
        for row in outcome.ground_clear_query_tensor:
            if not (
                0 <= row.source_cell[0] < outcome.source_geometry.shape[0]
                and 0 <= row.source_cell[1] < outcome.source_geometry.shape[1]
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ground query cell is outside native source geometry"
                )
            expected = canonical_ground_query_xy_body_v1(
                outcome.source_geometry,
                row.source_cell,
            )
            if any(
                not (
                    math.isclose(actual[0], target[0], rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(actual[1], target[1], rel_tol=0.0, abs_tol=1e-12)
                )
                for actual, target in zip(row.query_xy_body_m, expected)
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ground query geometry differs from canonical 0.05 m supports"
                )
            probabilities = tuple(_sigmoid(value) for value in row.clear_to_target_logits)
            known = tuple(
                max(probability, 1.0 - probability)
                >= self._calibration.known_probability_threshold
                for probability in probabilities
            )
            if all(
                in_frustum
                and is_known
                and probability
                >= self._calibration.free_given_known_probability_threshold
                for in_frustum, is_known, probability in zip(
                    row.query_in_frustum,
                    known,
                    probabilities,
                )
            ):
                free.add(row.source_cell)
        return frozenset(free)

    def _selected_local_hits(
        self,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> tuple[XY, ...]:
        hits: list[XY] = []
        for row in outcome.ordered_ray_hit_depth_tensor:
            if any(
                depth < self._calibration.ray_depth_min_m - 1e-12
                or depth > self._calibration.ray_depth_max_m + 1e-12
                for depth in row.ordered_depth_m
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ordered ray depth is outside frozen calibration"
                )
            survival = 1.0
            selected_depth: float | None = None
            for logit, depth in zip(row.ordered_hit_logits, row.ordered_depth_m):
                hazard = _sigmoid(logit)
                first_hit = survival * hazard
                if (
                    max(first_hit, 1.0 - first_hit)
                    >= self._calibration.known_probability_threshold
                    and first_hit
                    >= self._calibration.occupied_given_known_probability_threshold
                ):
                    selected_depth = depth
                    break
                survival *= 1.0 - hazard
            if selected_depth is not None:
                hits.append(
                    (
                        row.ray_origin_xy_body_m[0]
                        + row.ray_direction_xy_body[0] * selected_depth,
                        row.ray_origin_xy_body_m[1]
                        + row.ray_direction_xy_body[1] * selected_depth,
                    )
                )
        return tuple(hits)

    def _project(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> NativeLearnedPhysicalProjectionReceiptV1:
        transforms = self._uncertainty_transforms(outcome.pose)
        free_source = self._free_source_cells(outcome)
        physical_frame = snapshot.physical_map_frame
        shape = snapshot.physical_shape

        free_by_transform: list[frozenset[Cell]] = []
        domain: set[Cell] = set()
        all_ground_cells = frozenset(
            row.source_cell for row in outcome.ground_clear_query_tensor
        )
        for transform in transforms:
            transform_free_candidates: set[Cell] = set()
            for source_cell in free_source:
                minimum, maximum = _source_square(outcome.source_geometry, source_cell)
                polygon = tuple(
                    _apply_transform(point, transform)
                    for point in (
                        minimum,
                        (maximum[0], minimum[1]),
                        maximum,
                        (minimum[0], maximum[1]),
                    )
                )
                transform_free_candidates.update(
                    _candidate_cells_for_polygon(physical_frame, shape, polygon)
                )
            transform_free: set[Cell] = set()
            for destination_cell in transform_free_candidates:
                inverse_polygon = tuple(
                    _inverse_transform(point, transform)
                    for point in _destination_square(
                        physical_frame,
                        destination_cell,
                    )
                )
                if _polygon_covered_by_source_union(
                    inverse_polygon,
                    outcome.source_geometry,
                    free_source,
                ):
                    transform_free.add(destination_cell)
            free_by_transform.append(frozenset(transform_free))

            for source_cell in all_ground_cells:
                minimum, maximum = _source_square(outcome.source_geometry, source_cell)
                polygon = tuple(
                    _apply_transform(point, transform)
                    for point in (
                        minimum,
                        (maximum[0], minimum[1]),
                        maximum,
                        (minimum[0], maximum[1]),
                    )
                )
                for destination_cell in _candidate_cells_for_polygon(
                    physical_frame,
                    shape,
                    polygon,
                ):
                    if _polygons_intersect_closed(
                        polygon,
                        _destination_square(physical_frame, destination_cell),
                    ):
                        domain.add(destination_cell)

        free = (
            set.intersection(*(set(cells) for cells in free_by_transform))
            if free_by_transform
            else set()
        )
        occupied: set[Cell] = set()
        for local_hit in self._selected_local_hits(outcome):
            for transform in transforms:
                occupied.update(
                    _closed_point_supercover(
                        physical_frame,
                        shape,
                        _apply_transform(local_hit, transform),
                    )
                )
        domain.update(occupied)
        free.difference_update(occupied)
        unknown = domain - free - occupied
        transform_hash = _sha256(
            {
                "schema": "lewm_g3_native_projection_transform_set_v1",
                "transforms_xy_yaw": [list(value) for value in transforms],
            }
        )
        return NativeLearnedPhysicalProjectionReceiptV1(
            raw_outcome_sha256=outcome.content_sha256,
            raw_tensor_content_sha256=outcome.raw_tensor_content_sha256,
            calibration_sha256=self._calibration.content_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                snapshot.configuration_map_frame_sha256
            ),
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            transform_uncertainty_set=transforms,
            transform_uncertainty_set_sha256=transform_hash,
            free_cells=frozenset(free),
            occupied_cells=frozenset(occupied),
            unknown_cells=frozenset(unknown),
        )

    @staticmethod
    def _projection_payload_sha256(
        receipt: NativeLearnedPhysicalProjectionReceiptV1,
    ) -> str:
        return _sha256(
            {
                "schema": "lewm_g3_native_learned_projection_payload_v1",
                "projection_receipt_sha256": receipt.content_sha256,
                "free_cells": _cells_json(receipt.free_cells),
                "occupied_cells": _cells_json(receipt.occupied_cells),
                "unknown_cells": _cells_json(receipt.unknown_cells),
            }
        )

    def _build_inner_transaction(
        self,
        *,
        admission_kind: str,
        observation_id: str,
        observation_payload_sha256: str,
        observation_producer_sha256: str,
        pose: PoseProvenance,
        projection_receipt: NativeLearnedPhysicalProjectionReceiptV1,
        retracts_observation_id: str | None,
    ) -> PhysicalEvidenceTransaction:
        if admission_kind == "projection":
            evidence = tuple(
                PhysicalCellEvidence(cell=cell, label=PhysicalLabel.FREE)
                for cell in sorted(projection_receipt.free_cells)
            ) + tuple(
                PhysicalCellEvidence(cell=cell, label=PhysicalLabel.OCCUPIED)
                for cell in sorted(projection_receipt.occupied_cells)
            )
            unknown = tuple(sorted(projection_receipt.unknown_cells))
            retractions: tuple[str, ...] = ()
        elif admission_kind == "retraction":
            evidence = ()
            unknown = ()
            if retracts_observation_id is None:
                raise NativeLearnedProjectionBindingError(
                    "retraction transaction lost its target observation"
                )
            retractions = (retracts_observation_id,)
        else:
            raise NativeLearnedProjectionBindingError(
                "development transaction kind changed"
            )
        return PhysicalEvidenceTransaction(
            observation=ObservationIdentity(
                observation_id=observation_id,
                payload_sha256=observation_payload_sha256,
                producer_sha256=observation_producer_sha256,
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
            ),
            map_frame=self._memory.map_frame,
            pose=pose,
            physical_evidence=evidence,
            observed_unknown_cells=unknown,
            retract_learned_observation_ids=retractions,
            projection_contract_sha256=PROFILE_SHA256,
        )

    def _make_admission(
        self,
        *,
        admission_kind: str,
        source_outcome_sha256: str,
        projection_receipt: NativeLearnedPhysicalProjectionReceiptV1,
        inner: PhysicalEvidenceTransaction,
        configuration_revision: int,
        identities: dict[str, str],
        retracts_observation_id: str | None,
    ) -> QualifiedLearnedPhysicalDevelopmentAdmissionV1:
        observation = inner.observation
        core = {
            "schema": "lewm_g3_qualified_learned_development_admission_id_v1",
            "kind": admission_kind,
            "adapter_contract_sha256": self._adapter_contract_sha256,
            "source_outcome_sha256": source_outcome_sha256,
            "projection_receipt_sha256": projection_receipt.content_sha256,
            "physical_transaction_sha256": inner.content_sha256,
            "observation_id": observation.observation_id,
            "memory_revision_before": self._memory.revision,
            "retracts_observation_id": retracts_observation_id,
        }
        return QualifiedLearnedPhysicalDevelopmentAdmissionV1(
            admission_kind=admission_kind,
            admission_id_sha256=_sha256(core),
            adapter_contract_sha256=self._adapter_contract_sha256,
            source_outcome_sha256=source_outcome_sha256,
            projection_receipt_sha256=projection_receipt.content_sha256,
            physical_transaction_sha256=inner.content_sha256,
            observation_id=observation.observation_id,
            observation_payload_sha256=observation.payload_sha256,
            observation_producer_sha256=observation.producer_sha256,
            pose_sha256=inner.pose.content_sha256,
            memory_config_sha256=self._memory.config.content_sha256,
            physical_map_frame_sha256=self._memory.map_frame.content_sha256,
            configuration_map_frame_sha256=(
                self._projection.configuration_map_frame.content_sha256
            ),
            physical_revision_before=self._memory.revision,
            configuration_revision=configuration_revision,
            runner_execution_identity_sha256=identities[
                "runner_execution_identity_sha256"
            ],
            inference_implementation_sha256=identities[
                "inference_implementation_sha256"
            ],
            projection_implementation_sha256=identities[
                "projection_implementation_sha256"
            ],
            access_ledger_source_sha256=identities[
                "access_ledger_source_sha256"
            ],
            checkpoint_file_sha256=identities["checkpoint_file_sha256"],
            g2_report_file_sha256=identities["g2_report_file_sha256"],
            calibration_sha256=self._calibration.content_sha256,
            rgb_frame_sha256=identities["rgb_frame_sha256"],
            raw_outcome_file_sha256=identities["raw_outcome_file_sha256"],
            retracts_observation_id=retracts_observation_id,
        )

    def _register(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
        snapshot: TwoResolutionConfigurationSnapshotV2,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV1:
        self._issued[id(package)] = (package, package.content_sha256)
        self._issued_snapshots[id(package)] = snapshot
        return package

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV1:
        self._projection.assert_current_snapshot(snapshot)
        self._runner.assert_issued(outcome)
        self._assert_outcome_bindings(snapshot, outcome)
        projection_receipt = self._project(snapshot, outcome)
        payload_sha256 = self._projection_payload_sha256(projection_receipt)
        inner = self._build_inner_transaction(
            admission_kind="projection",
            observation_id=outcome.observation_id,
            observation_payload_sha256=payload_sha256,
            observation_producer_sha256=(
                outcome.runner_execution_identity_sha256
            ),
            pose=outcome.pose,
            projection_receipt=projection_receipt,
            retracts_observation_id=None,
        )
        identities = {
            "runner_execution_identity_sha256": (
                outcome.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": (
                outcome.inference_implementation_sha256
            ),
            "projection_implementation_sha256": (
                outcome.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": outcome.access_ledger_source_sha256,
            "checkpoint_file_sha256": outcome.checkpoint_file_sha256,
            "g2_report_file_sha256": outcome.g2_report_file_sha256,
            "rgb_frame_sha256": outcome.rgb_frame_sha256,
            "raw_outcome_file_sha256": outcome.raw_outcome_file_sha256,
        }
        admission = self._make_admission(
            admission_kind="projection",
            source_outcome_sha256=outcome.content_sha256,
            projection_receipt=projection_receipt,
            inner=inner,
            configuration_revision=snapshot.configuration_revision,
            identities=identities,
            retracts_observation_id=None,
        )
        package = QualifiedLearnedPhysicalDevelopmentTransactionV1(
            admission=admission,
            pose=outcome.pose,
            projection_receipt=projection_receipt,
            retracts_observation_id=None,
            _issuance_capability=self._capability,
        )
        self._runner.assert_issued(outcome, consume=True)
        return self._register(package, snapshot)

    def issue_retraction(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        committed_projection: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV1:
        self._projection.assert_current_snapshot(snapshot)
        if type(committed_projection) is not QualifiedLearnedPhysicalDevelopmentTransactionV1:
            raise TypeError("committed_projection has the wrong type")
        original = self._issued.get(id(committed_projection))
        if original is None or original[0] is not committed_projection:
            raise NativeLearnedProjectionBindingError(
                "retraction requires the exact package issued by this adapter"
            )
        committed_projection.assert_integrity()
        observation_id = committed_projection.admission.observation_id
        if (
            committed_projection.admission.admission_kind != "projection"
            or self._committed_by_observation.get(observation_id)
            is not committed_projection
            or id(committed_projection) not in self._consumed
            or observation_id not in self._memory.learned_observation_ids
        ):
            raise NativeLearnedProjectionBindingError(
                "retraction target is not an exact active committed projection"
            )
        if observation_id in self._retraction_issued:
            raise NativeLearnedProjectionReplayError(
                "a retraction was already issued for this observation"
            )
        self._sequence += 1
        retraction_observation_id = (
            f"qualified-native-v4-retract:{self._sequence}:{observation_id}"
        )
        payload_sha256 = _sha256(
            {
                "schema": "lewm_g3_qualified_learned_retraction_payload_v1",
                "retracts_observation_id": observation_id,
                "source_admission_sha256": (
                    committed_projection.admission.content_sha256
                ),
            }
        )
        inner = self._build_inner_transaction(
            admission_kind="retraction",
            observation_id=retraction_observation_id,
            observation_payload_sha256=payload_sha256,
            observation_producer_sha256=self._adapter_contract_sha256,
            pose=committed_projection.pose,
            projection_receipt=committed_projection.projection_receipt,
            retracts_observation_id=observation_id,
        )
        original_admission = committed_projection.admission
        identities = {
            "runner_execution_identity_sha256": (
                original_admission.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": (
                original_admission.inference_implementation_sha256
            ),
            "projection_implementation_sha256": (
                original_admission.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": (
                original_admission.access_ledger_source_sha256
            ),
            "checkpoint_file_sha256": original_admission.checkpoint_file_sha256,
            "g2_report_file_sha256": original_admission.g2_report_file_sha256,
            "rgb_frame_sha256": original_admission.rgb_frame_sha256,
            "raw_outcome_file_sha256": (
                original_admission.raw_outcome_file_sha256
            ),
        }
        admission = self._make_admission(
            admission_kind="retraction",
            source_outcome_sha256=original_admission.source_outcome_sha256,
            projection_receipt=committed_projection.projection_receipt,
            inner=inner,
            configuration_revision=snapshot.configuration_revision,
            identities=identities,
            retracts_observation_id=observation_id,
        )
        package = QualifiedLearnedPhysicalDevelopmentTransactionV1(
            admission=admission,
            pose=committed_projection.pose,
            projection_receipt=committed_projection.projection_receipt,
            retracts_observation_id=observation_id,
            _issuance_capability=self._capability,
        )
        self._retraction_issued.add(observation_id)
        return self._register(package, snapshot)

    def _assert_exact_package(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> TwoResolutionConfigurationSnapshotV2:
        if type(package) is not QualifiedLearnedPhysicalDevelopmentTransactionV1:
            raise TypeError("package has the wrong development transaction type")
        if (
            package._issuance_capability is not self._capability
            or self._issued.get(id(package)) is None
            or self._issued[id(package)][0] is not package
        ):
            raise NativeLearnedProjectionBindingError(
                "development transaction is not the exact live object issued here"
            )
        package.assert_integrity()
        if package.content_sha256 != self._issued[id(package)][1]:
            raise NativeLearnedProjectionBindingError(
                "development transaction differs from its issued content"
            )
        if id(package) in self._consumed:
            raise NativeLearnedProjectionReplayError(
                "development transaction was already consumed"
            )
        snapshot = self._issued_snapshots[id(package)]
        self._projection.assert_current_snapshot(snapshot)
        admission = package.admission
        if (
            admission.adapter_contract_sha256 != self._adapter_contract_sha256
            or admission.memory_config_sha256 != self._memory.config.content_sha256
            or admission.physical_map_frame_sha256
            != self._memory.map_frame.content_sha256
            or admission.configuration_map_frame_sha256
            != self._projection.configuration_map_frame.content_sha256
            or admission.physical_revision_before != self._memory.revision
            or admission.configuration_revision != snapshot.configuration_revision
            or package.development_only is not True
            or package.hardware_execution_authorized is not False
            or package.production_promotion_authorized is not False
        ):
            raise NativeLearnedProjectionBindingError(
                "development transaction authority/frame/revision binding changed"
            )
        return snapshot

    def commit(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> TransactionReceipt:
        self._assert_exact_package(package)
        admission = package.admission
        inner = self._build_inner_transaction(
            admission_kind=admission.admission_kind,
            observation_id=admission.observation_id,
            observation_payload_sha256=admission.observation_payload_sha256,
            observation_producer_sha256=admission.observation_producer_sha256,
            pose=package.pose,
            projection_receipt=package.projection_receipt,
            retracts_observation_id=package.retracts_observation_id,
        )
        if inner.content_sha256 != admission.physical_transaction_sha256:
            raise NativeLearnedProjectionBindingError(
                "privately reconstructed physical transaction changed"
            )
        try:
            receipt = self._memory.apply_transaction(inner)
        except TransactionRejectedError:
            raise
        self._consumed.add(id(package))
        if admission.admission_kind == "projection":
            self._committed_by_observation[admission.observation_id] = package
        else:
            target = admission.retracts_observation_id
            if target is not None:
                self._committed_by_observation.pop(target, None)
        return receipt


def require_production_native_learned_projection_adapter() -> object:
    if (
        PRODUCTION_NATIVE_V4_RUNNER is None
        or PRODUCTION_V4_CHECKPOINT_FILE_SHA256 is None
        or PRODUCTION_G2_REPORT_FILE_SHA256 is None
        or PRODUCTION_V4_CALIBRATION_SHA256 is None
        or PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER is None
    ):
        raise PermissionError(
            "production native learned-projection identities are unset"
        )
    return PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER


__all__ = [
    "FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1",
    "FrozenNativeLearnedProjectionCalibrationV1",
    "NativeLearnedPhysicalProjectionAdapterV1",
    "NativeLearnedPhysicalProjectionReceiptV1",
    "NativeLearnedProjectionBindingError",
    "NativeLearnedProjectionRejectedError",
    "NativeLearnedProjectionReplayError",
    "NativeV4SourceGeometryV1",
    "QualifiedLearnedPhysicalDevelopmentAdmissionV1",
    "QualifiedLearnedPhysicalDevelopmentTransactionV1",
    "RawGroundClearCellQueriesV1",
    "RawOrderedRayHitDepthV1",
    "SyntheticNativeV4RawOutcomeV1",
    "SyntheticNativeV4RunnerV1",
    "canonical_ground_query_xy_body_v1",
    "require_production_native_learned_projection_adapter",
]
