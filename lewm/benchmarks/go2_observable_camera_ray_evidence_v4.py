"""Pure contracts and rasterization for observable camera-ray evidence V4.

V4 represents only evidence supplied by the current calibrated camera:

* five clear-to-ground queries for every canonical 0.05 m source cell; and
* exact first-hit distances for visible opaque hits on the native pixel
  lattice.

Each frame also carries the compact calibrated camera origin and orthonormal
forward/right/up basis in the yaw-aligned body frame, plus the ground-plane
height in that frame.  Body-frame hit positions and arbitrary ground-query
camera coordinates are deterministic derived values, never stored label
authority.

This module performs no file I/O and imports no model, dataset, simulator, or
collision-geometry implementation. Physical-free priors, collision vetoes,
body inflation, and configuration morphology are forbidden inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np


EVIDENCE_SCHEMA = "lewm_go2_observable_camera_ray_evidence_v4"
RASTER_SCHEMA = "lewm_go2_observable_camera_ray_raster_v4"

UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2

SOURCE_SHAPE = (128, 128)
SOURCE_CELL_SIZE_M = 0.05
SOURCE_FORWARD_MIN_EDGE_M = -1.0
SOURCE_LEFT_MIN_EDGE_M = -3.2
GROUND_SUPPORT_OFFSETS_CELL_FRACTION = (
    (0.0, 0.0),
    (-0.5, -0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (0.5, 0.5),
)
GROUND_SUPPORT_COUNT = len(GROUND_SUPPORT_OFFSETS_CELL_FRACTION)

PIXEL_RAY_SHAPE = (84, 112)
CAMERA_IMAGE_SHAPE = (168, 224)
PIXEL_RAY_STRIDE_PX = 2
CAMERA_HORIZONTAL_FOV_DEG = 78.323
CAMERA_VERTICAL_FOV_DEG = 62.8370386364
CAMERA_NEAR_M = 0.05
CAMERA_BASIS_ORTHONORMAL_ATOL = 5e-5

OUTPUT_SHAPE = (64, 64)
OUTPUT_CELL_SIZE_M = 0.10
OUTPUT_FORWARD_MIN_EDGE_M = -1.0
OUTPUT_LEFT_MIN_EDGE_M = -3.2
# Hit range and calibration are authoritative float32 values.  This tolerance
# is a conservative two-centimillimetre supercover around exact cell edges.
CLOSED_BOUNDARY_ABS_TOLERANCE_M = 2e-5

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "physical_free",
        "physical_free_mask",
        "collision",
        "collision_boxes",
        "collision_obstacle_boxes",
        "collision_overlap",
        "collision_veto",
        "morphology",
        "body_inflation",
        "body_inflation_radius_m",
        "configuration_labels",
        "configuration_occupancy",
    }
)

_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "ground_support_in_frustum",
        "ground_support_clear_to_target",
        "camera_origin_body_m",
        "camera_basis_body_fru",
        "ground_plane_z_body_m",
        "pixel_hit_mask",
        "pixel_first_hit_distance_m",
    }
)


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _contains_forbidden_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    normalized = name.lower()
    return normalized in FORBIDDEN_FIELD_NAMES or any(
        token in normalized
        for token in (
            "physical_free",
            "collision",
            "morpholog",
            "body_inflat",
            "configuration_",
        )
    )


def _reject_forbidden_fields(value: Mapping[str, Any]) -> None:
    forbidden = sorted(str(name) for name in value if _contains_forbidden_name(name))
    if forbidden:
        raise ValueError(
            "observable camera-ray evidence forbids privileged fields: "
            f"{forbidden}"
        )


@dataclass(frozen=True)
class CalibratedCameraQueriesV4:
    """Deterministic camera coordinates for body-frame query points."""

    in_frustum: np.ndarray
    uv_px: np.ndarray
    target_distance_m: np.ndarray

    def __post_init__(self) -> None:
        in_frustum = np.array(self.in_frustum, dtype=bool, order="C", copy=True)
        uv = np.array(self.uv_px, dtype=np.float64, order="C", copy=True)
        distance = np.array(
            self.target_distance_m, dtype=np.float64, order="C", copy=True
        )
        if uv.shape != (*in_frustum.shape, 2) or distance.shape != in_frustum.shape:
            raise ValueError("calibrated query arrays have inconsistent shapes")
        if not np.isfinite(uv).all() or not np.isfinite(distance).all():
            raise ValueError("calibrated query arrays must be finite")
        if np.any(distance < 0.0):
            raise ValueError("calibrated query distances must be non-negative")
        in_frustum.setflags(write=False)
        uv.setflags(write=False)
        distance.setflags(write=False)
        object.__setattr__(self, "in_frustum", in_frustum)
        object.__setattr__(self, "uv_px", uv)
        object.__setattr__(self, "target_distance_m", distance)


def _validated_camera_calibration(
    camera_origin_body_m: Any,
    camera_basis_body_fru: Any,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(camera_origin_body_m, dtype=np.float64)
    basis = np.asarray(camera_basis_body_fru, dtype=np.float64)
    if origin.shape != (3,) or basis.shape != (3, 3):
        raise ValueError("camera calibration must have origin [3] and basis [3,3]")
    if not np.isfinite(origin).all() or not np.isfinite(basis).all():
        raise ValueError("camera calibration must be finite")
    if not np.allclose(
        basis @ basis.T,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError("camera basis must be orthonormal")
    if not np.allclose(
        np.cross(basis[1], basis[0]),
        basis[2],
        rtol=0.0,
        atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError("camera basis must use forward/right/up handedness")
    return origin, basis


def project_body_points_to_camera_v4(
    points_body_m: Any,
    *,
    camera_origin_body_m: Any,
    camera_basis_body_fru: Any,
) -> CalibratedCameraQueriesV4:
    """Project arbitrary yaw-body points into the fixed rectilinear camera."""

    points = np.asarray(points_body_m, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("body query points must be finite with trailing shape [3]")
    origin, basis = _validated_camera_calibration(
        camera_origin_body_m, camera_basis_body_fru
    )
    relative = points - origin
    forward = relative @ basis[0]
    right = relative @ basis[1]
    up = relative @ basis[2]
    distance = np.linalg.norm(relative, axis=-1)
    tan_h = math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5)
    tan_v = math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) * 0.5)
    safe_forward = np.where(np.abs(forward) > 1e-12, forward, 1.0)
    normalized_x = right / (safe_forward * tan_h)
    normalized_y = up / (safe_forward * tan_v)
    in_frustum = (
        (forward > CAMERA_NEAR_M)
        & (np.abs(normalized_x) <= 1.0 + 1e-12)
        & (np.abs(normalized_y) <= 1.0 + 1e-12)
    )
    height, width = CAMERA_IMAGE_SHAPE
    uv = np.stack(
        (
            (normalized_x + 1.0) * (float(width) * 0.5),
            (1.0 - normalized_y) * (float(height) * 0.5),
        ),
        axis=-1,
    )
    return CalibratedCameraQueriesV4(
        in_frustum=in_frustum,
        uv_px=uv,
        target_distance_m=distance,
    )


def calibrated_pixel_ray_directions_body_v4(
    camera_basis_body_fru: Any,
) -> np.ndarray:
    """Return the frozen stride-2 pixel-ray lattice in yaw-body coordinates."""

    _origin, basis = _validated_camera_calibration(
        np.zeros(3, dtype=np.float64), camera_basis_body_fru
    )
    height, width = CAMERA_IMAGE_SHAPE
    pixel_x = np.arange(0, width, PIXEL_RAY_STRIDE_PX, dtype=np.float64)
    pixel_y = np.arange(0, height, PIXEL_RAY_STRIDE_PX, dtype=np.float64)
    pixel_x += 0.5 * PIXEL_RAY_STRIDE_PX
    pixel_y += 0.5 * PIXEL_RAY_STRIDE_PX
    pixel_x = np.minimum(pixel_x, width - 0.5)
    pixel_y = np.minimum(pixel_y, height - 0.5)
    tan_h = math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5)
    tan_v = math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) * 0.5)
    normalized_x = (2.0 * pixel_x / width - 1.0) * tan_h
    normalized_y = (1.0 - 2.0 * pixel_y / height) * tan_v
    grid_x, grid_y = np.meshgrid(normalized_x, normalized_y, indexing="xy")
    directions = (
        basis[0][None, None, :]
        + grid_x[..., None] * basis[1][None, None, :]
        + grid_y[..., None] * basis[2][None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    if directions.shape != (*PIXEL_RAY_SHAPE, 3):
        raise AssertionError("frozen pixel-ray shape changed")
    directions.setflags(write=False)
    return directions


def project_canonical_ground_support_v4(
    *,
    camera_origin_body_m: Any,
    camera_basis_body_fru: Any,
    ground_plane_z_body_m: float,
) -> CalibratedCameraQueriesV4:
    points = canonical_ground_support_points_body_m(
        ground_z_body_m=float(ground_plane_z_body_m)
    )
    return project_body_points_to_camera_v4(
        points,
        camera_origin_body_m=camera_origin_body_m,
        camera_basis_body_fru=camera_basis_body_fru,
    )


@dataclass(frozen=True)
class ObservableCameraRayEvidenceV4:
    """One immutable current-camera evidence field in the yaw-aligned body frame."""

    camera_origin_body_m: np.ndarray
    camera_basis_body_fru: np.ndarray
    ground_plane_z_body_m: float
    ground_support_in_frustum: np.ndarray
    ground_support_clear_to_target: np.ndarray
    pixel_hit_mask: np.ndarray
    pixel_first_hit_distance_m: np.ndarray

    def __post_init__(self) -> None:
        origin = _readonly_array(
            self.camera_origin_body_m,
            dtype=np.float32,
            shape=(3,),
            name="camera_origin_body_m",
        )
        basis = _readonly_array(
            self.camera_basis_body_fru,
            dtype=np.float32,
            shape=(3, 3),
            name="camera_basis_body_fru",
        )
        basis64 = basis.astype(np.float64)
        gram = basis64 @ basis64.T
        if not np.allclose(
            gram,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
        ):
            raise ValueError("camera basis must be orthonormal")
        forward, right, up = basis64
        if not np.allclose(
            np.cross(right, forward),
            up,
            rtol=0.0,
            atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
        ):
            raise ValueError("camera basis must use forward/right/up handedness")
        ground_z = float(self.ground_plane_z_body_m)
        if not math.isfinite(ground_z):
            raise ValueError("ground_plane_z_body_m must be finite")

        support_shape = (*SOURCE_SHAPE, GROUND_SUPPORT_COUNT)
        in_frustum = _readonly_array(
            self.ground_support_in_frustum,
            dtype=bool,
            shape=support_shape,
            name="ground_support_in_frustum",
        )
        clear = _readonly_array(
            self.ground_support_clear_to_target,
            dtype=bool,
            shape=support_shape,
            name="ground_support_clear_to_target",
        )
        if np.any(clear & ~in_frustum):
            raise ValueError("out-of-frustum ground support cannot be clear")
        calibrated_ground = project_canonical_ground_support_v4(
            camera_origin_body_m=origin,
            camera_basis_body_fru=basis,
            ground_plane_z_body_m=ground_z,
        )
        if not np.array_equal(in_frustum, calibrated_ground.in_frustum):
            raise ValueError(
                "ground_support_in_frustum disagrees with frame calibration"
            )
        hit_mask = _readonly_array(
            self.pixel_hit_mask,
            dtype=bool,
            shape=PIXEL_RAY_SHAPE,
            name="pixel_hit_mask",
        )
        hit_distance = _readonly_array(
            self.pixel_first_hit_distance_m,
            dtype=np.float32,
            shape=PIXEL_RAY_SHAPE,
            name="pixel_first_hit_distance_m",
        )
        if np.any(hit_distance[~hit_mask] != 0.0):
            raise ValueError(
                "pixel distance must be canonical zero where hit_mask is false"
            )
        if np.any(hit_distance[hit_mask] <= CAMERA_NEAR_M):
            raise ValueError(
                "pixel hit distance must lie strictly beyond the near plane"
            )
        object.__setattr__(self, "camera_origin_body_m", origin)
        object.__setattr__(self, "camera_basis_body_fru", basis)
        object.__setattr__(self, "ground_plane_z_body_m", ground_z)
        object.__setattr__(self, "ground_support_in_frustum", in_frustum)
        object.__setattr__(self, "ground_support_clear_to_target", clear)
        object.__setattr__(self, "pixel_hit_mask", hit_mask)
        object.__setattr__(self, "pixel_first_hit_distance_m", hit_distance)

    @property
    def ground_source_free(self) -> np.ndarray:
        result = np.all(
            self.ground_support_in_frustum
            & self.ground_support_clear_to_target,
            axis=2,
        )
        result.setflags(write=False)
        return result

    @property
    def pixel_hit_xy_body_m(self) -> np.ndarray:
        directions = calibrated_pixel_ray_directions_body_v4(
            self.camera_basis_body_fru
        )
        points = (
            self.camera_origin_body_m.astype(np.float64)[None, None, :]
            + directions
            * self.pixel_first_hit_distance_m.astype(np.float64)[..., None]
        )
        result = np.zeros((*PIXEL_RAY_SHAPE, 2), dtype=np.float64)
        result[self.pixel_hit_mask] = points[..., :2][self.pixel_hit_mask]
        result.setflags(write=False)
        return result

    def content_sha256(self) -> str:
        digest = hashlib.sha256(EVIDENCE_SCHEMA.encode("ascii"))
        arrays = (
            np.ascontiguousarray(self.camera_origin_body_m, dtype="<f4"),
            np.ascontiguousarray(self.camera_basis_body_fru, dtype="<f4"),
            np.ascontiguousarray(
                np.asarray([self.ground_plane_z_body_m]), dtype="<f4"
            ),
            np.ascontiguousarray(self.ground_support_in_frustum, dtype=np.uint8),
            np.ascontiguousarray(
                self.ground_support_clear_to_target, dtype=np.uint8
            ),
            np.ascontiguousarray(self.pixel_hit_mask, dtype=np.uint8),
            np.ascontiguousarray(
                self.pixel_first_hit_distance_m, dtype="<f4"
            ),
        )
        for value in arrays:
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class ObservableCameraRayRasterV4:
    """Deterministic 0.05 m source and 0.10 m output evidence rasters."""

    source_free_mask: np.ndarray
    output_free_before_occupied_mask: np.ndarray
    output_occupied_mask: np.ndarray
    output_labels: np.ndarray
    evidence_sha256: str

    def __post_init__(self) -> None:
        source = _readonly_array(
            self.source_free_mask,
            dtype=bool,
            shape=SOURCE_SHAPE,
            name="source_free_mask",
        )
        free = _readonly_array(
            self.output_free_before_occupied_mask,
            dtype=bool,
            shape=OUTPUT_SHAPE,
            name="output_free_before_occupied_mask",
        )
        occupied = _readonly_array(
            self.output_occupied_mask,
            dtype=bool,
            shape=OUTPUT_SHAPE,
            name="output_occupied_mask",
        )
        labels = _readonly_array(
            self.output_labels,
            dtype=np.uint8,
            shape=OUTPUT_SHAPE,
            name="output_labels",
        )
        if not np.isin(labels, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
            raise ValueError("output_labels contains an unsupported class")
        expected = np.full(OUTPUT_SHAPE, UNKNOWN_CLASS, dtype=np.uint8)
        expected[free] = FREE_CLASS
        expected[occupied] = OCCUPIED_CLASS
        if not np.array_equal(labels, expected):
            raise ValueError("output labels disagree with FREE/OCCUPIED precedence")
        if (
            not isinstance(self.evidence_sha256, str)
            or len(self.evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.evidence_sha256
            )
        ):
            raise ValueError("evidence_sha256 must be lowercase hexadecimal")
        object.__setattr__(self, "source_free_mask", source)
        object.__setattr__(self, "output_free_before_occupied_mask", free)
        object.__setattr__(self, "output_occupied_mask", occupied)
        object.__setattr__(self, "output_labels", labels)

    def content_sha256(self) -> str:
        digest = hashlib.sha256(RASTER_SCHEMA.encode("ascii"))
        digest.update(self.evidence_sha256.encode("ascii"))
        for value in (
            np.ascontiguousarray(self.source_free_mask, dtype=np.uint8),
            np.ascontiguousarray(
                self.output_free_before_occupied_mask, dtype=np.uint8
            ),
            np.ascontiguousarray(self.output_occupied_mask, dtype=np.uint8),
            np.ascontiguousarray(self.output_labels, dtype=np.uint8),
        ):
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()


def observable_camera_ray_evidence_v4_from_mapping(
    value: Mapping[str, Any],
) -> ObservableCameraRayEvidenceV4:
    """Validate an exact external evidence payload and reject privileged fields."""

    if not isinstance(value, Mapping):
        raise TypeError("observable camera-ray evidence payload must be a mapping")
    _reject_forbidden_fields(value)
    keys = frozenset(value)
    if keys != _EVIDENCE_FIELDS:
        missing = sorted(_EVIDENCE_FIELDS - keys)
        extra = sorted(str(name) for name in keys - _EVIDENCE_FIELDS)
        raise ValueError(
            f"observable camera-ray evidence fields differ: missing={missing} extra={extra}"
        )
    if value["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("observable camera-ray evidence schema changed")
    return ObservableCameraRayEvidenceV4(
        camera_origin_body_m=value["camera_origin_body_m"],
        camera_basis_body_fru=value["camera_basis_body_fru"],
        ground_plane_z_body_m=value["ground_plane_z_body_m"],
        ground_support_in_frustum=value["ground_support_in_frustum"],
        ground_support_clear_to_target=value["ground_support_clear_to_target"],
        pixel_hit_mask=value["pixel_hit_mask"],
        pixel_first_hit_distance_m=value["pixel_first_hit_distance_m"],
    )


def canonical_ground_support_points_body_m(
    *,
    ground_z_body_m: float,
) -> np.ndarray:
    """Return the fixed source-cell center/corner ground query points."""

    ground_z = float(ground_z_body_m)
    if not math.isfinite(ground_z):
        raise ValueError("ground_z_body_m must be finite")
    forward = SOURCE_FORWARD_MIN_EDGE_M + (
        np.arange(SOURCE_SHAPE[0], dtype=np.float64) + 0.5
    ) * SOURCE_CELL_SIZE_M
    left = SOURCE_LEFT_MIN_EDGE_M + (
        np.arange(SOURCE_SHAPE[1], dtype=np.float64) + 0.5
    ) * SOURCE_CELL_SIZE_M
    forward_grid, left_grid = np.meshgrid(forward, left, indexing="ij")
    offsets = (
        np.asarray(GROUND_SUPPORT_OFFSETS_CELL_FRACTION, dtype=np.float64)
        * SOURCE_CELL_SIZE_M
    )
    result = np.empty((*SOURCE_SHAPE, GROUND_SUPPORT_COUNT, 3), dtype=np.float64)
    result[..., 0] = forward_grid[:, :, None] + offsets[None, None, :, 0]
    result[..., 1] = left_grid[:, :, None] + offsets[None, None, :, 1]
    result[..., 2] = ground_z
    result.setflags(write=False)
    return result


def _closed_axis_indices(
    value_m: float,
    *,
    minimum_edge_m: float,
    cell_size_m: float,
    cell_count: int,
) -> tuple[int, ...]:
    value = float(value_m)
    minimum = float(minimum_edge_m)
    size = float(cell_size_m)
    maximum = minimum + int(cell_count) * size
    tolerance = CLOSED_BOUNDARY_ABS_TOLERANCE_M
    if value < minimum - tolerance or value > maximum + tolerance:
        return ()
    clipped = min(max(value, minimum), maximum)
    scaled = (clipped - minimum) / size
    nearest = int(round(scaled))
    scaled_tolerance = tolerance / size
    if abs(scaled - nearest) <= scaled_tolerance:
        if nearest <= 0:
            return (0,)
        if nearest >= cell_count:
            return (cell_count - 1,)
        return (nearest - 1, nearest)
    index = int(math.floor(scaled))
    if 0 <= index < cell_count:
        return (index,)
    return ()


def rasterize_observable_camera_ray_evidence_v4(
    evidence: ObservableCameraRayEvidenceV4,
) -> ObservableCameraRayRasterV4:
    """Rasterize current-camera evidence with no privileged geometry input."""

    if not isinstance(evidence, ObservableCameraRayEvidenceV4):
        raise TypeError("evidence must be ObservableCameraRayEvidenceV4")
    source_free = np.array(evidence.ground_source_free, dtype=bool, copy=True)
    output_free = source_free.reshape(
        OUTPUT_SHAPE[0],
        2,
        OUTPUT_SHAPE[1],
        2,
    ).all(axis=(1, 3))

    output_occupied = np.zeros(OUTPUT_SHAPE, dtype=bool)
    for forward_m, left_m in evidence.pixel_hit_xy_body_m[
        evidence.pixel_hit_mask
    ]:
        rows = _closed_axis_indices(
            float(forward_m),
            minimum_edge_m=OUTPUT_FORWARD_MIN_EDGE_M,
            cell_size_m=OUTPUT_CELL_SIZE_M,
            cell_count=OUTPUT_SHAPE[0],
        )
        columns = _closed_axis_indices(
            float(left_m),
            minimum_edge_m=OUTPUT_LEFT_MIN_EDGE_M,
            cell_size_m=OUTPUT_CELL_SIZE_M,
            cell_count=OUTPUT_SHAPE[1],
        )
        for row in rows:
            for column in columns:
                output_occupied[row, column] = True

    labels = np.full(OUTPUT_SHAPE, UNKNOWN_CLASS, dtype=np.uint8)
    labels[output_free] = FREE_CLASS
    labels[output_occupied] = OCCUPIED_CLASS
    return ObservableCameraRayRasterV4(
        source_free_mask=source_free,
        output_free_before_occupied_mask=output_free,
        output_occupied_mask=output_occupied,
        output_labels=labels,
        evidence_sha256=evidence.content_sha256(),
    )


__all__ = [
    "CAMERA_BASIS_ORTHONORMAL_ATOL",
    "CAMERA_HORIZONTAL_FOV_DEG",
    "CAMERA_IMAGE_SHAPE",
    "CAMERA_NEAR_M",
    "CAMERA_VERTICAL_FOV_DEG",
    "CalibratedCameraQueriesV4",
    "CLOSED_BOUNDARY_ABS_TOLERANCE_M",
    "EVIDENCE_SCHEMA",
    "FORBIDDEN_FIELD_NAMES",
    "FREE_CLASS",
    "GROUND_SUPPORT_COUNT",
    "GROUND_SUPPORT_OFFSETS_CELL_FRACTION",
    "OCCUPIED_CLASS",
    "OUTPUT_CELL_SIZE_M",
    "OUTPUT_SHAPE",
    "ObservableCameraRayEvidenceV4",
    "ObservableCameraRayRasterV4",
    "PIXEL_RAY_SHAPE",
    "RASTER_SCHEMA",
    "SOURCE_CELL_SIZE_M",
    "SOURCE_SHAPE",
    "UNKNOWN_CLASS",
    "canonical_ground_support_points_body_m",
    "calibrated_pixel_ray_directions_body_v4",
    "observable_camera_ray_evidence_v4_from_mapping",
    "project_body_points_to_camera_v4",
    "project_canonical_ground_support_v4",
    "rasterize_observable_camera_ray_evidence_v4",
]
