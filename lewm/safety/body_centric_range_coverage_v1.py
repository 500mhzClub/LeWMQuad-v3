"""Deterministic range-coverage geometry for the Go2 development assay.

This module is deliberately independent of Genesis, Torch, learned models and
the scientific corpus.  It provides the pure geometry needed to define range
coverage before materialisation:

* world/body transforms use scalar-first ``wxyz`` quaternions;
* sensor coordinates are body FRU (``+x`` forward, ``+y`` left, ``+z`` up);
* robot boxes consume Genesis' *full* box extents and divide them by two;
* environment boxes expose half extents explicitly;
* self returns win environment returns at equal ray distance; and
* a first hit inside the near blind region remains an unsupported first hit --
  rays never pass through it to reveal geometry behind it.

Stable evaluator-facing entry points are :func:`raycast_scene`,
:func:`generate_sparse_scan_pattern`, :func:`generate_l2_scan_pattern`,
:func:`instantiate_geoms`, and :func:`closest_points_to_scene`.  The lower
level analytic intersection routines remain public for deterministic fixtures.
``generate_l2_scan_pattern`` is explicitly a deterministic low-discrepancy
approximation, not Unitree's unavailable proprietary non-repetitive sequence.

All public numerical outputs are float64 unless their type is intrinsically
discrete.  Input ordering cannot change equal-distance hit identity: robot
primitives are ordered by geometry index and environment boxes by object index.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


GEOMETRY_EPS = 1.0e-12
DISTANCE_TIE_ATOL_M = 1.0e-12

NO_HIT = 0
ENVIRONMENT_HIT = 1
ROBOT_HIT = 2
GROUND_HIT = 3

HIT_CLASS_NAMES = {
    NO_HIT: "NO_HIT",
    ENVIRONMENT_HIT: "ENVIRONMENT",
    ROBOT_HIT: "ROBOT_SELF",
    GROUND_HIT: "GROUND",
}


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical receipt representation used by focused fixtures."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _vector(value: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.shape != (length,) or not np.isfinite(output).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return output


def _directions(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.ndim == 1:
        output = output[None, :]
    if output.ndim != 2 or output.shape[1] != 3 or not np.isfinite(output).all():
        raise ValueError("ray directions must have shape [N,3] and be finite")
    norm = np.linalg.norm(output, axis=1)
    if np.any(norm <= GEOMETRY_EPS):
        raise ValueError("ray directions must be nonzero")
    return output / norm[:, None]


def normalize_quaternion_wxyz(value: Sequence[float] | np.ndarray) -> np.ndarray:
    quaternion = _vector(value, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm <= GEOMETRY_EPS:
        raise ValueError("quaternion_wxyz must be nonzero")
    return quaternion / norm


def normalize_quaternions_wxyz(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Normalize an arbitrary leading-shape array of scalar-first quaternions."""

    quaternions = np.asarray(value, dtype=np.float64)
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4 or not np.isfinite(quaternions).all():
        raise ValueError("quaternions_wxyz must have finite shape [...,4]")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= GEOMETRY_EPS):
        raise ValueError("quaternions_wxyz must be nonzero")
    return quaternions / norms


def rotation_matrix_wxyz(value: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the active local-to-world rotation for a wxyz quaternion."""

    w, x, y, z = normalize_quaternion_wxyz(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrices_wxyz(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Vectorized active local-to-world rotation matrices for wxyz inputs."""

    quaternions = normalize_quaternions_wxyz(value)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    output = np.empty(quaternions.shape[:-1] + (3, 3), dtype=np.float64)
    output[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    output[..., 0, 1] = 2.0 * (x * y - z * w)
    output[..., 0, 2] = 2.0 * (x * z + y * w)
    output[..., 1, 0] = 2.0 * (x * y + z * w)
    output[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    output[..., 1, 2] = 2.0 * (y * z - x * w)
    output[..., 2, 0] = 2.0 * (x * z - y * w)
    output[..., 2, 1] = 2.0 * (y * z + x * w)
    output[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return output


def multiply_quaternion_wxyz(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compose active rotations in scalar-first order (``first * second``)."""

    aw, ax, ay, az = normalize_quaternion_wxyz(first)
    bw, bx, by, bz = normalize_quaternion_wxyz(second)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def compose_transform(
    parent_position_xyz_m: Sequence[float] | np.ndarray,
    parent_quaternion_wxyz: Sequence[float] | np.ndarray,
    local_position_xyz_m: Sequence[float] | np.ndarray,
    local_quaternion_wxyz: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a local pose below a parent world pose."""

    parent_position = _vector(parent_position_xyz_m, 3, "parent_position_xyz_m")
    local_position = _vector(local_position_xyz_m, 3, "local_position_xyz_m")
    parent_quaternion = normalize_quaternion_wxyz(parent_quaternion_wxyz)
    position = parent_position + rotation_matrix_wxyz(parent_quaternion) @ local_position
    quaternion = multiply_quaternion_wxyz(parent_quaternion, local_quaternion_wxyz)
    return position, quaternion


def nlerp_quaternion_wxyz(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shortest-arc normalized linear interpolation in scalar-first order."""

    fraction = float(alpha)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("alpha must be finite and in [0,1]")
    left = normalize_quaternion_wxyz(first)
    right = normalize_quaternion_wxyz(second)
    if float(left @ right) < 0.0:
        right = -right
    if fraction == 0.0:
        return left
    if fraction == 1.0:
        return right
    return normalize_quaternion_wxyz((1.0 - fraction) * left + fraction * right)


def nlerp_quaternions_wxyz(
    first: Sequence[Sequence[float]] | np.ndarray,
    second: Sequence[Sequence[float]] | np.ndarray,
    alpha: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Vectorized shortest-arc nlerp with broadcast leading dimensions."""

    left = normalize_quaternions_wxyz(first)
    right = normalize_quaternions_wxyz(second)
    if left.shape != right.shape:
        raise ValueError("first and second quaternion arrays must have identical shape")
    fraction = np.asarray(alpha, dtype=np.float64)
    try:
        fraction = np.broadcast_to(fraction, left.shape[:-1])
    except ValueError as error:
        raise ValueError("alpha must broadcast to quaternion leading dimensions") from error
    if not np.isfinite(fraction).all() or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("alpha must be finite and in [0,1]")
    right = np.where(np.sum(left * right, axis=-1, keepdims=True) < 0.0, -right, right)
    blended = (1.0 - fraction[..., None]) * left + fraction[..., None] * right
    return normalize_quaternions_wxyz(blended)


def interpolate_transform(
    first_position_xyz_m: Sequence[float] | np.ndarray,
    first_quaternion_wxyz: Sequence[float] | np.ndarray,
    second_position_xyz_m: Sequence[float] | np.ndarray,
    second_quaternion_wxyz: Sequence[float] | np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    fraction = float(alpha)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("alpha must be finite and in [0,1]")
    first_position = _vector(first_position_xyz_m, 3, "first_position_xyz_m")
    second_position = _vector(second_position_xyz_m, 3, "second_position_xyz_m")
    position = (1.0 - fraction) * first_position + fraction * second_position
    quaternion = nlerp_quaternion_wxyz(first_quaternion_wxyz, second_quaternion_wxyz, fraction)
    return position, quaternion


def interpolate_transform_series(
    timestamps_s: Sequence[float] | np.ndarray,
    positions_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    quaternions_wxyz: Sequence[Sequence[float]] | np.ndarray,
    query_timestamps_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a strictly ordered transform trace at arbitrary timestamps."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    positions = np.asarray(positions_xyz_m, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    query = np.asarray(query_timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be a finite vector with at least two entries")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if positions.shape != (len(timestamps), 3) or not np.isfinite(positions).all():
        raise ValueError("positions_xyz_m must have shape [T,3]")
    if quaternions.shape != (len(timestamps), 4) or not np.isfinite(quaternions).all():
        raise ValueError("quaternions_wxyz must have shape [T,4]")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query_timestamps_s must be a finite vector")
    if len(query) and (query.min() < timestamps[0] - GEOMETRY_EPS or query.max() > timestamps[-1] + GEOMETRY_EPS):
        raise ValueError("query timestamp is outside the transform trace")

    output_position = np.empty((len(query), 3), dtype=np.float64)
    output_quaternion = np.empty((len(query), 4), dtype=np.float64)
    for query_index, value in enumerate(query):
        if value <= timestamps[0]:
            left = 0
            fraction = 0.0
        elif value >= timestamps[-1]:
            left = len(timestamps) - 2
            fraction = 1.0
        else:
            right = int(np.searchsorted(timestamps, value, side="right"))
            left = right - 1
            fraction = float((value - timestamps[left]) / (timestamps[right] - timestamps[left]))
        output_position[query_index], output_quaternion[query_index] = interpolate_transform(
            positions[left],
            quaternions[left],
            positions[left + 1],
            quaternions[left + 1],
            fraction,
        )
    return output_position, output_quaternion


def interpolate_transform_series_vectorized(
    timestamps_s: Sequence[float] | np.ndarray,
    positions_xyz_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
    query_timestamps_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one or many pose traces without looping over query timestamps.

    Poses have shape ``[T,...,3]`` and ``[T,...,4]``.  Returned arrays have
    shape ``[Q,...,3]`` and ``[Q,...,4]``.  This is the intended helper for
    producing per-ray poses for all 27 frozen collision specs.
    """

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    positions = np.asarray(positions_xyz_m, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    query = np.asarray(query_timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be a finite vector with at least two entries")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if positions.ndim < 2 or positions.shape[0] != len(timestamps) or positions.shape[-1] != 3:
        raise ValueError("positions_xyz_m must have shape [T,...,3]")
    if quaternions.shape[:-1] != positions.shape[:-1] or quaternions.shape[-1] != 4:
        raise ValueError("quaternions_wxyz must have shape [T,...,4] aligned to positions")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("pose traces must be finite")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query_timestamps_s must be a finite vector")
    if len(query) and (query.min() < timestamps[0] - GEOMETRY_EPS or query.max() > timestamps[-1] + GEOMETRY_EPS):
        raise ValueError("query timestamp is outside the transform trace")
    right = np.searchsorted(timestamps, query, side="right")
    right = np.clip(right, 1, len(timestamps) - 1)
    left = right - 1
    fraction = (query - timestamps[left]) / (timestamps[right] - timestamps[left])
    fraction = np.where(query <= timestamps[0], 0.0, fraction)
    fraction = np.where(query >= timestamps[-1], 1.0, fraction)
    expand = (len(query),) + (1,) * (positions.ndim - 2) + (1,)
    weight = fraction.reshape(expand)
    output_position = (1.0 - weight) * positions[left] + weight * positions[right]
    quaternion_weight = fraction.reshape((len(query),) + (1,) * (quaternions.ndim - 2))
    output_quaternion = nlerp_quaternions_wxyz(
        quaternions[left], quaternions[right], quaternion_weight
    )
    return output_position, output_quaternion


def transform_points(
    points_local: Sequence[Sequence[float]] | np.ndarray,
    position_world_xyz_m: Sequence[float] | np.ndarray,
    quaternion_world_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_local must have shape [N,3]")
    position = _vector(position_world_xyz_m, 3, "position_world_xyz_m")
    return points @ rotation_matrix_wxyz(quaternion_world_wxyz).T + position[None, :]


def inverse_transform_points(
    points_world: Sequence[Sequence[float]] | np.ndarray,
    position_world_xyz_m: Sequence[float] | np.ndarray,
    quaternion_world_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Transform world points into the stated local FRU frame."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must have shape [N,3]")
    position = _vector(position_world_xyz_m, 3, "position_world_xyz_m")
    return (points - position[None, :]) @ rotation_matrix_wxyz(quaternion_world_wxyz)


def transform_directions(
    directions_local: Sequence[Sequence[float]] | np.ndarray,
    quaternion_world_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Rotate local unit directions into world coordinates."""

    directions = _directions(directions_local)
    return directions @ rotation_matrix_wxyz(quaternion_world_wxyz).T


def spherical_directions_fru(
    azimuth_rad: Sequence[float] | np.ndarray,
    elevation_rad: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the azimuth-major Cartesian product of FRU spherical angles."""

    azimuth = np.asarray(azimuth_rad, dtype=np.float64)
    elevation = np.asarray(elevation_rad, dtype=np.float64)
    if azimuth.ndim != 1 or elevation.ndim != 1:
        raise ValueError("azimuth_rad and elevation_rad must be vectors")
    if not np.isfinite(azimuth).all() or not np.isfinite(elevation).all():
        raise ValueError("spherical angles must be finite")
    aa, ee = np.meshgrid(azimuth, elevation, indexing="ij")
    directions = np.stack(
        (
            np.cos(ee) * np.cos(aa),
            np.cos(ee) * np.sin(aa),
            np.sin(ee),
        ),
        axis=-1,
    )
    return directions.reshape(-1, 3).astype(np.float64, copy=False)


@dataclass(frozen=True)
class ScanPattern:
    """Frozen sensor-frame FRU rays and their planning-window timestamps."""

    identity: str
    directions_sensor_fru: np.ndarray
    timestamps_s: np.ndarray
    azimuth_rad: np.ndarray
    elevation_rad: np.ndarray
    approximation_class: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions_sensor_fru, dtype=np.float64)
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        azimuth = np.asarray(self.azimuth_rad, dtype=np.float64)
        elevation = np.asarray(self.elevation_rad, dtype=np.float64)
        count = len(timestamps)
        if not self.identity or not self.approximation_class:
            raise ValueError("scan pattern identities must be nonempty")
        if directions.shape != (count, 3):
            raise ValueError("directions_sensor_fru must have shape [N,3]")
        if azimuth.shape != (count,) or elevation.shape != (count,):
            raise ValueError("scan angles must have shape [N]")
        if not all(np.isfinite(value).all() for value in (directions, timestamps, azimuth, elevation)):
            raise ValueError("scan pattern values must be finite")
        if count and (timestamps[0] < 0.0 or np.any(np.diff(timestamps) < 0.0)):
            raise ValueError("scan timestamps must be nonnegative and nondecreasing")
        if count and not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("scan directions must be unit length")
        canonical_json_bytes(dict(self.parameters))

    @property
    def ray_count(self) -> int:
        return int(len(self.timestamps_s))

    def to_serializable(self, *, include_rays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "approximation_class": self.approximation_class,
            "identity": self.identity,
            "parameters": dict(self.parameters),
            "ray_count": self.ray_count,
            "ray_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.column_stack(
                        (
                            np.asarray(self.timestamps_s, dtype="<f8"),
                            np.asarray(self.directions_sensor_fru, dtype="<f8"),
                        )
                    ),
                    dtype="<f8",
                ).tobytes(order="C")
            ).hexdigest(),
            "schema": "body_centric_range_scan_pattern_v1",
        }
        if include_rays:
            payload["rays"] = [
                {
                    "azimuth_rad": float(self.azimuth_rad[index]),
                    "direction_sensor_fru": [float(value) for value in self.directions_sensor_fru[index]],
                    "elevation_rad": float(self.elevation_rad[index]),
                    "ray_index": index,
                    "timestamp_s": float(self.timestamps_s[index]),
                }
                for index in range(self.ray_count)
            ]
        return payload


def generate_sparse_scan_pattern(
    *,
    azimuth_bins: int = 180,
    vertical_channels_deg: Sequence[float] = (-15.0, -5.0, 5.0, 15.0),
    duration_s: float = 0.1,
    azimuth_phase_deg: float = 0.0,
    instantaneous: bool = False,
) -> ScanPattern:
    """Generate an azimuth-major spinning multi-channel baseline.

    Channels at one azimuth share a timestamp.  With ``instantaneous=True``
    all timestamps are zero, which is useful when reproducing a frozen current
    snapshot instead of an accumulated scan.
    """

    bins = int(azimuth_bins)
    duration = float(duration_s)
    phase = float(azimuth_phase_deg)
    channels = np.asarray(vertical_channels_deg, dtype=np.float64)
    if bins <= 0 or not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("azimuth_bins and duration_s must be positive")
    if channels.ndim != 1 or not len(channels) or not np.isfinite(channels).all():
        raise ValueError("vertical_channels_deg must be a nonempty finite vector")
    if not math.isfinite(phase):
        raise ValueError("azimuth_phase_deg must be finite")
    azimuth_by_bin = np.radians(-180.0 + phase + 360.0 * np.arange(bins, dtype=np.float64) / bins)
    azimuth = np.repeat(azimuth_by_bin, len(channels))
    elevation = np.tile(np.radians(channels), bins)
    directions = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    ).astype(np.float64, copy=False)
    if instantaneous:
        timestamps = np.zeros(len(directions), dtype=np.float64)
    else:
        timestamps = np.repeat(np.arange(bins, dtype=np.float64) * duration / bins, len(channels))
    return ScanPattern(
        identity=f"SPARSE_{bins}X{len(channels)}_FRU_V1",
        directions_sensor_fru=directions,
        timestamps_s=timestamps,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        approximation_class="DETERMINISTIC_SPINNING_CHANNEL_PATTERN",
        parameters={
            "azimuth_bins": bins,
            "azimuth_phase_deg": phase,
            "duration_s": duration,
            "instantaneous": bool(instantaneous),
            "vertical_channels_deg": [float(value) for value in channels],
        },
    )


def generate_l2_scan_pattern(
    *,
    point_rate_hz: float = 64_000.0,
    duration_s: float = 0.1,
    horizontal_fov_deg: tuple[float, float] = (-180.0, 180.0),
    vertical_fov_deg: tuple[float, float] = (-6.0, 90.0),
    azimuth_frequency_hz: float = 5.55,
    vertical_frequency_hz: float = 216.0,
    horizontal_phase_cycles: float = 0.0,
    vertical_phase_cycles: float = 0.0,
) -> ScanPattern:
    """Generate the frozen deterministic approximation to an L2 scan.

    The sequence matches the prospectively frozen 5.55 Hz circumferential and
    216 Hz vertical motion with a triangle-wave vertical sweep.  It exactly
    matches requested rate, FOV and mid-bin point timing, but intentionally
    makes no claim to reproduce Unitree's proprietary non-repetitive ray order.
    """

    rate = float(point_rate_hz)
    duration = float(duration_s)
    horizontal_lower, horizontal_upper = map(float, horizontal_fov_deg)
    vertical_lower, vertical_upper = map(float, vertical_fov_deg)
    azimuth_frequency = float(azimuth_frequency_hz)
    vertical_frequency = float(vertical_frequency_hz)
    if not math.isfinite(rate) or rate <= 0.0 or not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("point_rate_hz and duration_s must be positive")
    if not all(
        map(
            math.isfinite,
            (
                horizontal_lower,
                horizontal_upper,
                vertical_lower,
                vertical_upper,
                azimuth_frequency,
                vertical_frequency,
                float(horizontal_phase_cycles),
                float(vertical_phase_cycles),
            ),
        )
    ):
        raise ValueError("scan FOV and phases must be finite")
    if azimuth_frequency <= 0.0 or vertical_frequency <= 0.0:
        raise ValueError("scan frequencies must be positive")
    horizontal_span = horizontal_upper - horizontal_lower
    if horizontal_span <= 0.0 or horizontal_span > 360.0 + 1.0e-10:
        raise ValueError("horizontal FOV span must lie in (0,360]")
    if not -90.0 <= vertical_lower < vertical_upper <= 90.0:
        raise ValueError("vertical FOV must be ordered within [-90,90]")
    ray_count = int(math.floor(rate * duration + 1.0e-12))
    if ray_count <= 0:
        raise ValueError("rate-duration product must generate at least one ray")
    ray_index = np.arange(ray_count, dtype=np.float64)
    # Mid-bin timestamps are part of the prospective approximation.  The
    # vertical triangle has value zero at integer phase, one at half phase,
    # then returns to zero at the next integer phase.
    timestamps = (ray_index + 0.5) / rate
    horizontal_u = np.mod(
        azimuth_frequency * timestamps + float(horizontal_phase_cycles), 1.0
    )
    vertical_phase = np.mod(
        vertical_frequency * timestamps + float(vertical_phase_cycles), 1.0
    )
    vertical_u = 1.0 - np.abs(2.0 * vertical_phase - 1.0)
    azimuth = np.radians(horizontal_lower + horizontal_span * horizontal_u)
    elevation = np.radians(vertical_lower + (vertical_upper - vertical_lower) * vertical_u)
    directions = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    ).astype(np.float64, copy=False)
    return ScanPattern(
        identity="ASSUMED_GO2_HEAD_LIDAR_L2_APPROXIMATION_V1",
        directions_sensor_fru=directions,
        timestamps_s=timestamps,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        approximation_class="APPROXIMATED_REALISTIC_PLATFORM_SCAN",
        parameters={
            "azimuth_frequency_hz": azimuth_frequency,
            "duration_s": duration,
            "horizontal_fov_deg": [horizontal_lower, horizontal_upper],
            "horizontal_phase_cycles": float(horizontal_phase_cycles),
            "point_rate_hz": rate,
            "timestamp_rule": "(ray_index+0.5)/point_rate_hz",
            "vertical_frequency_hz": vertical_frequency,
            "vertical_fov_deg": [vertical_lower, vertical_upper],
            "vertical_phase_cycles": float(vertical_phase_cycles),
            "vertical_waveform": "TRIANGLE_MIN_AT_INTEGER_PHASE_MAX_AT_HALF_PHASE",
        },
    )


@dataclass(frozen=True)
class OrientedBox:
    """An environment OBB with explicit half extents."""

    identity: str
    center_xyz_m: tuple[float, float, float]
    half_extents_xyz_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_index: int = -1

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("environment box identity must be nonempty")
        _vector(self.center_xyz_m, 3, "center_xyz_m")
        half = _vector(self.half_extents_xyz_m, 3, "half_extents_xyz_m")
        if np.any(half <= 0.0):
            raise ValueError("half_extents_xyz_m must be positive")
        normalize_quaternion_wxyz(self.quaternion_wxyz)


@dataclass(frozen=True)
class RobotPrimitive:
    """One frozen Genesis robot collision primitive.

    ``data`` follows Genesis' primitive contract.  In particular, box entries
    are full xyz extents and are divided by two inside every geometric query.
    Capsule data is ``(radius, finite_centerline_length, ...)``.
    """

    identity: str
    kind: str
    data: tuple[float, ...]
    position_xyz_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    geom_index: int
    link_index: int
    link_name: str

    def __post_init__(self) -> None:
        if not self.identity or not self.link_name:
            raise ValueError("robot primitive and link identities must be nonempty")
        if self.kind not in {"sphere", "capsule", "box"}:
            raise ValueError(f"unsupported primitive kind {self.kind}")
        data = np.asarray(self.data, dtype=np.float64)
        minimum = {"sphere": 1, "capsule": 2, "box": 3}[self.kind]
        if data.ndim != 1 or len(data) < minimum or not np.isfinite(data).all():
            raise ValueError(f"malformed {self.kind} primitive data")
        if np.any(data[:minimum] <= 0.0):
            raise ValueError(f"{self.kind} primitive dimensions must be positive")
        _vector(self.position_xyz_m, 3, "position_xyz_m")
        normalize_quaternion_wxyz(self.quaternion_wxyz)

    @property
    def box_half_extents_xyz_m(self) -> np.ndarray:
        if self.kind != "box":
            raise AttributeError("only box primitives have half extents")
        return np.asarray(self.data[:3], dtype=np.float64) * 0.5


@dataclass(frozen=True)
class RobotPrimitiveSpec:
    """Collision primitive frozen in its parent-link frame."""

    identity: str
    kind: str
    data: tuple[float, ...]
    local_position_xyz_m: tuple[float, float, float]
    local_quaternion_wxyz: tuple[float, float, float, float]
    geom_index: int
    link_index: int
    link_name: str

    def __post_init__(self) -> None:
        # Reuse the authoritative shape/data validation without changing the
        # meaning of the local pose.
        RobotPrimitive(
            identity=self.identity,
            kind=self.kind,
            data=self.data,
            position_xyz_m=self.local_position_xyz_m,
            quaternion_wxyz=self.local_quaternion_wxyz,
            geom_index=self.geom_index,
            link_index=self.link_index,
            link_name=self.link_name,
        )


def instantiate_geoms(
    geometry_specs: Sequence[RobotPrimitiveSpec],
    link_transforms: Mapping[
        int | str,
        tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray],
    ],
    *,
    box_data_contract: str = "GENESIS_FULL_EXTENTS",
) -> tuple[RobotPrimitive, ...]:
    """Instantiate link-local collision specs at deterministic world poses.

    ``link_transforms`` may be keyed by integer link index or link name.  The
    integer key wins when both are supplied.  Boxes are accepted only under
    the corrected Genesis full-extents contract.
    """

    if box_data_contract != "GENESIS_FULL_EXTENTS":
        raise ValueError("only the GENESIS_FULL_EXTENTS box contract is supported")
    output: list[RobotPrimitive] = []
    for spec in sorted(geometry_specs, key=lambda value: (value.geom_index, value.identity)):
        if spec.link_index in link_transforms:
            link_position, link_quaternion = link_transforms[spec.link_index]
        elif spec.link_name in link_transforms:
            link_position, link_quaternion = link_transforms[spec.link_name]
        else:
            raise KeyError(f"missing transform for link {spec.link_index}:{spec.link_name}")
        position, quaternion = compose_transform(
            link_position,
            link_quaternion,
            spec.local_position_xyz_m,
            spec.local_quaternion_wxyz,
        )
        output.append(
            RobotPrimitive(
                identity=spec.identity,
                kind=spec.kind,
                data=spec.data,
                position_xyz_m=tuple(float(value) for value in position),
                quaternion_wxyz=tuple(float(value) for value in quaternion),
                geom_index=spec.geom_index,
                link_index=spec.link_index,
                link_name=spec.link_name,
            )
        )
    return tuple(output)


def _ray_oriented_box_distances_normalized(
    origin_xyz_m: np.ndarray,
    directions_world: np.ndarray,
    center_xyz_m: np.ndarray,
    half_extents_xyz_m: np.ndarray,
    quaternion_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    rotation = rotation_matrix_wxyz(quaternion_wxyz)
    local_origin = rotation.T @ (origin_xyz_m - center_xyz_m)
    local_direction = directions_world @ rotation
    lower = np.full(len(directions_world), -np.inf, dtype=np.float64)
    upper = np.full(len(directions_world), np.inf, dtype=np.float64)
    valid = np.ones(len(directions_world), dtype=bool)
    for axis in range(3):
        component = local_direction[:, axis]
        parallel = np.abs(component) <= GEOMETRY_EPS
        valid &= ~(parallel & (abs(float(local_origin[axis])) > half_extents_xyz_m[axis] + GEOMETRY_EPS))
        first = np.full(len(directions_world), -np.inf, dtype=np.float64)
        second = np.full(len(directions_world), np.inf, dtype=np.float64)
        active = ~parallel
        first[active] = (-half_extents_xyz_m[axis] - local_origin[axis]) / component[active]
        second[active] = (half_extents_xyz_m[axis] - local_origin[axis]) / component[active]
        lower = np.maximum(lower, np.minimum(first, second))
        upper = np.minimum(upper, np.maximum(first, second))
    entry = np.maximum(lower, 0.0)
    valid &= upper + GEOMETRY_EPS >= entry
    return np.where(valid, entry, np.inf)


def ray_oriented_box_distances(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    box: OrientedBox,
) -> np.ndarray:
    return _ray_oriented_box_distances_normalized(
        _vector(origin_xyz_m, 3, "origin_xyz_m"),
        _directions(directions_world),
        _vector(box.center_xyz_m, 3, "center_xyz_m"),
        _vector(box.half_extents_xyz_m, 3, "half_extents_xyz_m"),
        box.quaternion_wxyz,
    )


def _ray_sphere_distances_normalized(
    origin_xyz_m: np.ndarray,
    directions_world: np.ndarray,
    center_xyz_m: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive")
    offset = origin_xyz_m - center_xyz_m
    c = float(offset @ offset - radius * radius)
    if c <= GEOMETRY_EPS:
        return np.zeros(len(directions_world), dtype=np.float64)
    b = directions_world @ offset
    discriminant = b * b - c
    tangent_or_hit = discriminant >= -GEOMETRY_EPS
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = -b - root
    far = -b + root
    distance = np.where(near >= -GEOMETRY_EPS, np.maximum(near, 0.0), np.where(far >= -GEOMETRY_EPS, np.maximum(far, 0.0), np.inf))
    return np.where(tangent_or_hit, distance, np.inf)


def ray_sphere_distances(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    center_xyz_m: Sequence[float] | np.ndarray,
    radius_m: float,
) -> np.ndarray:
    return _ray_sphere_distances_normalized(
        _vector(origin_xyz_m, 3, "origin_xyz_m"),
        _directions(directions_world),
        _vector(center_xyz_m, 3, "center_xyz_m"),
        radius_m,
    )


def _ray_capsule_distances_normalized(
    origin_xyz_m: np.ndarray,
    directions_world: np.ndarray,
    center_xyz_m: np.ndarray,
    quaternion_wxyz: Sequence[float] | np.ndarray,
    radius_m: float,
    centerline_length_m: float,
) -> np.ndarray:
    radius = float(radius_m)
    length = float(centerline_length_m)
    if not math.isfinite(radius) or radius <= 0.0 or not math.isfinite(length) or length < 0.0:
        raise ValueError("capsule radius must be positive and length nonnegative")
    if length <= GEOMETRY_EPS:
        return _ray_sphere_distances_normalized(origin_xyz_m, directions_world, center_xyz_m, radius)

    rotation = rotation_matrix_wxyz(quaternion_wxyz)
    local_origin = rotation.T @ (origin_xyz_m - center_xyz_m)
    local_direction = directions_world @ rotation
    half = 0.5 * length
    closest_z = float(np.clip(local_origin[2], -half, half))
    closest = np.asarray([0.0, 0.0, closest_z], dtype=np.float64)
    if float(np.linalg.norm(local_origin - closest)) <= radius + GEOMETRY_EPS:
        return np.zeros(len(directions_world), dtype=np.float64)

    output = np.full(len(directions_world), np.inf, dtype=np.float64)
    radial_a = np.sum(local_direction[:, :2] ** 2, axis=1)
    radial_b = local_direction[:, 0] * local_origin[0] + local_direction[:, 1] * local_origin[1]
    radial_c = float(local_origin[0] ** 2 + local_origin[1] ** 2 - radius * radius)
    discriminant = radial_b * radial_b - radial_a * radial_c
    active = (radial_a > GEOMETRY_EPS) & (discriminant >= -GEOMETRY_EPS)
    square = np.sqrt(np.maximum(discriminant, 0.0))
    for sign in (-1.0, 1.0):
        root = np.full(len(directions_world), np.inf, dtype=np.float64)
        root[active] = (-radial_b[active] + sign * square[active]) / radial_a[active]
        z = local_origin[2] + root * local_direction[:, 2]
        valid = active & (root >= -GEOMETRY_EPS) & (z >= -half - GEOMETRY_EPS) & (z <= half + GEOMETRY_EPS)
        output = np.minimum(output, np.where(valid, np.maximum(root, 0.0), np.inf))

    for cap_z in (-half, half):
        cap_center_world = center_xyz_m + rotation @ np.asarray([0.0, 0.0, cap_z], dtype=np.float64)
        output = np.minimum(
            output,
            _ray_sphere_distances_normalized(origin_xyz_m, directions_world, cap_center_world, radius),
        )
    return output


def ray_capsule_distances(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    center_xyz_m: Sequence[float] | np.ndarray,
    quaternion_wxyz: Sequence[float] | np.ndarray,
    radius_m: float,
    centerline_length_m: float,
) -> np.ndarray:
    return _ray_capsule_distances_normalized(
        _vector(origin_xyz_m, 3, "origin_xyz_m"),
        _directions(directions_world),
        _vector(center_xyz_m, 3, "center_xyz_m"),
        quaternion_wxyz,
        radius_m,
        centerline_length_m,
    )


def ray_ground_plane_distances(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    ground_z_m: float,
) -> np.ndarray:
    origin = _vector(origin_xyz_m, 3, "origin_xyz_m")
    directions = _directions(directions_world)
    height = float(ground_z_m)
    if not math.isfinite(height):
        raise ValueError("ground_z_m must be finite")
    if abs(float(origin[2]) - height) <= GEOMETRY_EPS:
        return np.zeros(len(directions), dtype=np.float64)
    vertical = directions[:, 2]
    active = np.abs(vertical) > GEOMETRY_EPS
    distance = np.full(len(directions), np.inf, dtype=np.float64)
    distance[active] = (height - origin[2]) / vertical[active]
    return np.where(distance >= -GEOMETRY_EPS, np.maximum(distance, 0.0), np.inf)


def _ray_robot_primitive_distances_normalized(
    origin_xyz_m: np.ndarray,
    directions_world: np.ndarray,
    primitive: RobotPrimitive,
) -> np.ndarray:
    data = np.asarray(primitive.data, dtype=np.float64)
    position = _vector(primitive.position_xyz_m, 3, "position_xyz_m")
    if primitive.kind == "sphere":
        return _ray_sphere_distances_normalized(origin_xyz_m, directions_world, position, float(data[0]))
    if primitive.kind == "capsule":
        return _ray_capsule_distances_normalized(
            origin_xyz_m,
            directions_world,
            position,
            primitive.quaternion_wxyz,
            float(data[0]),
            float(data[1]),
        )
    half = data[:3] * 0.5
    return _ray_oriented_box_distances_normalized(
        origin_xyz_m,
        directions_world,
        position,
        half,
        primitive.quaternion_wxyz,
    )


def ray_robot_primitive_distances(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    primitive: RobotPrimitive,
) -> np.ndarray:
    return _ray_robot_primitive_distances_normalized(
        _vector(origin_xyz_m, 3, "origin_xyz_m"),
        _directions(directions_world),
        primitive,
    )


def _strictly_better(candidate: np.ndarray, incumbent: np.ndarray) -> np.ndarray:
    """Prefer the incumbent for equal-distance candidates of the same class."""

    return candidate < incumbent - DISTANCE_TIE_ATOL_M


@dataclass(frozen=True)
class FirstHitResult:
    directions_world: np.ndarray
    raw_distance_m: np.ndarray
    reported_distance_m: np.ndarray
    point_world_xyz_m: np.ndarray
    hit_class: np.ndarray
    hit_identity: np.ndarray
    hit_index: np.ndarray
    valid_return: np.ndarray
    near_blind: np.ndarray
    beyond_far: np.ndarray
    no_hit: np.ndarray
    self_return: np.ndarray
    self_occluded_within_range: np.ndarray
    environment_return: np.ndarray
    ground_return: np.ndarray
    free_to_far: np.ndarray

    def to_serializable(self) -> dict[str, Any]:
        rows = []
        for index in range(len(self.raw_distance_m)):
            raw = float(self.raw_distance_m[index])
            reported = float(self.reported_distance_m[index])
            point = self.point_world_xyz_m[index]
            rows.append(
                {
                    "beyond_far": bool(self.beyond_far[index]),
                    "class": HIT_CLASS_NAMES[int(self.hit_class[index])],
                    "environment_return": bool(self.environment_return[index]),
                    "free_to_far": bool(self.free_to_far[index]),
                    "ground_return": bool(self.ground_return[index]),
                    "hit_identity": None if not str(self.hit_identity[index]) else str(self.hit_identity[index]),
                    "hit_index": int(self.hit_index[index]),
                    "near_blind": bool(self.near_blind[index]),
                    "no_hit": bool(self.no_hit[index]),
                    "point_world_xyz_m": None if not np.isfinite(point).all() else [float(value) for value in point],
                    "raw_distance_m": None if not math.isfinite(raw) else raw,
                    "reported_distance_m": None if not math.isfinite(reported) else reported,
                    "self_occluded_within_range": bool(self.self_occluded_within_range[index]),
                    "self_return": bool(self.self_return[index]),
                    "valid_return": bool(self.valid_return[index]),
                }
            )
        return {"schema": "body_centric_range_first_hit_v1", "rows": rows}


def first_hits(
    origin_xyz_m: Sequence[float] | np.ndarray,
    directions_world: Sequence[Sequence[float]] | np.ndarray,
    *,
    environment_boxes: Sequence[OrientedBox] = (),
    robot_primitives: Sequence[RobotPrimitive] = (),
    ground_z_m: float | None = None,
    ground_identity: str = "ground",
    near_m: float = 0.0,
    far_m: float = 20.0,
) -> FirstHitResult:
    """Return physical first hits, retaining identities before range filtering.

    The physical first surface is selected before applying near/far limits.  A
    near-blind self/environment return therefore cannot expose a farther
    surface.  Robot hits win an environment/ground hit within
    :data:`DISTANCE_TIE_ATOL_M`.
    """

    origin = _vector(origin_xyz_m, 3, "origin_xyz_m")
    directions = _directions(directions_world)
    near = float(near_m)
    far = float(far_m)
    if not math.isfinite(near) or near < 0.0 or not math.isfinite(far) or far <= near:
        raise ValueError("range contract requires 0 <= near_m < far_m")
    if ground_z_m is not None and not ground_identity:
        raise ValueError("ground_identity must be nonempty")

    count = len(directions)
    identity_width = max(
        [1, len(ground_identity)]
        + [len(item.identity) for item in environment_boxes]
        + [len(item.identity) for item in robot_primitives]
    )
    environment_distance = np.full(count, np.inf, dtype=np.float64)
    environment_class = np.full(count, NO_HIT, dtype=np.int8)
    environment_identity = np.full(count, "", dtype=f"<U{identity_width}")
    environment_index = np.full(count, -1, dtype=np.int32)

    ordered_environment = sorted(
        environment_boxes,
        key=lambda value: (value.object_index if value.object_index >= 0 else np.iinfo(np.int32).max, value.identity),
    )
    for box in ordered_environment:
        distance = _ray_oriented_box_distances_normalized(
            origin,
            directions,
            _vector(box.center_xyz_m, 3, "center_xyz_m"),
            _vector(box.half_extents_xyz_m, 3, "half_extents_xyz_m"),
            box.quaternion_wxyz,
        )
        better = _strictly_better(distance, environment_distance)
        environment_distance[better] = distance[better]
        environment_class[better] = ENVIRONMENT_HIT
        environment_identity[better] = box.identity
        environment_index[better] = int(box.object_index)

    if ground_z_m is not None:
        distance = ray_ground_plane_distances(origin, directions, float(ground_z_m))
        better = _strictly_better(distance, environment_distance)
        environment_distance[better] = distance[better]
        environment_class[better] = GROUND_HIT
        environment_identity[better] = ground_identity
        environment_index[better] = -1

    robot_distance = np.full(count, np.inf, dtype=np.float64)
    robot_identity = np.full(count, "", dtype=f"<U{identity_width}")
    robot_index = np.full(count, -1, dtype=np.int32)
    ordered_robot = sorted(robot_primitives, key=lambda value: (value.geom_index, value.identity))
    for primitive in ordered_robot:
        distance = _ray_robot_primitive_distances_normalized(origin, directions, primitive)
        better = _strictly_better(distance, robot_distance)
        robot_distance[better] = distance[better]
        robot_identity[better] = primitive.identity
        robot_index[better] = int(primitive.geom_index)

    robot_wins = np.isfinite(robot_distance) & (
        ~np.isfinite(environment_distance)
        | (robot_distance <= environment_distance + DISTANCE_TIE_ATOL_M)
    )
    raw_distance = np.where(robot_wins, robot_distance, environment_distance)
    hit_class = np.where(robot_wins, ROBOT_HIT, environment_class).astype(np.int8)
    hit_identity = np.where(robot_wins, robot_identity, environment_identity)
    hit_index = np.where(robot_wins, robot_index, environment_index).astype(np.int32)

    finite = np.isfinite(raw_distance)
    near_blind = finite & (raw_distance < near - DISTANCE_TIE_ATOL_M)
    beyond_far = finite & (raw_distance > far + DISTANCE_TIE_ATOL_M)
    valid = finite & ~near_blind & ~beyond_far
    reported = np.where(valid, raw_distance, np.nan)
    points = np.full((count, 3), np.nan, dtype=np.float64)
    points[valid] = origin[None, :] + raw_distance[valid, None] * directions[valid]
    self_return = hit_class == ROBOT_HIT
    no_hit = ~finite
    free_to_far = no_hit | beyond_far
    return FirstHitResult(
        directions_world=directions,
        raw_distance_m=raw_distance,
        reported_distance_m=reported,
        point_world_xyz_m=points,
        hit_class=hit_class,
        hit_identity=hit_identity,
        hit_index=hit_index,
        valid_return=valid,
        near_blind=near_blind,
        beyond_far=beyond_far,
        no_hit=no_hit,
        self_return=self_return,
        self_occluded_within_range=self_return & (raw_distance <= far + DISTANCE_TIE_ATOL_M),
        environment_return=valid & (hit_class == ENVIRONMENT_HIT),
        ground_return=valid & (hit_class == GROUND_HIT),
        free_to_far=free_to_far,
    )


@dataclass(frozen=True)
class SceneRaycastResult:
    """Evaluator-facing scene raycast with exact object/link/geom identity."""

    origins_world_xyz_m: np.ndarray
    directions_world: np.ndarray
    raw_distance_m: np.ndarray
    distance_m: np.ndarray
    raw_point_world_xyz_m: np.ndarray
    point_world_xyz_m: np.ndarray
    hit_kind: np.ndarray
    object_identity: np.ndarray
    object_index: np.ndarray
    link_name: np.ndarray
    link_index: np.ndarray
    geom_identity: np.ndarray
    geom_index: np.ndarray
    valid_return: np.ndarray
    near_blind: np.ndarray
    beyond_far: np.ndarray
    no_hit: np.ndarray
    self_return: np.ndarray
    self_occluded_within_range: np.ndarray
    free_to_far: np.ndarray

    def to_serializable(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for ray_index in range(len(self.raw_distance_m)):
            raw = float(self.raw_distance_m[ray_index])
            reported = float(self.distance_m[ray_index])
            rows.append(
                {
                    "beyond_far": bool(self.beyond_far[ray_index]),
                    "direction_world": [float(value) for value in self.directions_world[ray_index]],
                    "distance_m": None if not math.isfinite(reported) else reported,
                    "free_to_far": bool(self.free_to_far[ray_index]),
                    "geom_identity": str(self.geom_identity[ray_index]) or None,
                    "geom_index": int(self.geom_index[ray_index]),
                    "hit_kind": str(self.hit_kind[ray_index]),
                    "link_index": int(self.link_index[ray_index]),
                    "link_name": str(self.link_name[ray_index]) or None,
                    "near_blind": bool(self.near_blind[ray_index]),
                    "no_hit": bool(self.no_hit[ray_index]),
                    "object_identity": str(self.object_identity[ray_index]) or None,
                    "object_index": int(self.object_index[ray_index]),
                    "origin_world_xyz_m": [float(value) for value in self.origins_world_xyz_m[ray_index]],
                    "raw_distance_m": None if not math.isfinite(raw) else raw,
                    "ray_index": ray_index,
                    "self_occluded_within_range": bool(self.self_occluded_within_range[ray_index]),
                    "self_return": bool(self.self_return[ray_index]),
                    "valid_return": bool(self.valid_return[ray_index]),
                }
            )
        return {"schema": "body_centric_range_scene_raycast_v1", "rows": rows}


def raycast_scene(
    origins: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    directions: Sequence[Sequence[float]] | np.ndarray,
    near: float,
    far: float,
    env_boxes: Sequence[OrientedBox] = (),
    robot_geoms: Sequence[RobotPrimitive] = (),
    ground: bool | float | None = True,
) -> SceneRaycastResult:
    """Raycast paired or common-origin rays against a frozen scene.

    A single ``[3]`` origin broadcasts across all directions.  An ``[N,3]``
    origin array supplies one motion-compensated origin per ray.  ``ground``
    is ``True`` for z=0, a float for another z height, or false/None to omit it.
    The physical first hit is always retained in ``raw_distance_m`` and exact
    self-return identity fields, even when near/far filtering makes
    ``distance_m`` NaN.
    """

    normalized_directions = _directions(directions)
    origin_array = np.asarray(origins, dtype=np.float64)
    if origin_array.shape == (3,):
        origin_array = np.repeat(origin_array[None, :], len(normalized_directions), axis=0)
    if (
        origin_array.shape != (len(normalized_directions), 3)
        or not np.isfinite(origin_array).all()
    ):
        raise ValueError("origins must have shape [3] or [N,3] matching directions")
    if ground is True:
        ground_z_m: float | None = 0.0
    elif ground is False or ground is None:
        ground_z_m = None
    else:
        ground_z_m = float(ground)
        if not math.isfinite(ground_z_m):
            raise ValueError("ground height must be finite")

    count = len(normalized_directions)
    scalar_fields = {
        "raw_distance_m": np.full(count, np.inf, dtype=np.float64),
        "reported_distance_m": np.full(count, np.nan, dtype=np.float64),
        "hit_class": np.full(count, NO_HIT, dtype=np.int8),
        "hit_identity": np.full(count, "", dtype=f"<U{max([1] + [len(x.identity) for x in env_boxes] + [len(x.identity) for x in robot_geoms] + [6])}"),
        "hit_index": np.full(count, -1, dtype=np.int32),
        "valid_return": np.zeros(count, dtype=bool),
        "near_blind": np.zeros(count, dtype=bool),
        "beyond_far": np.zeros(count, dtype=bool),
        "no_hit": np.ones(count, dtype=bool),
        "self_return": np.zeros(count, dtype=bool),
        "self_occluded_within_range": np.zeros(count, dtype=bool),
        "environment_return": np.zeros(count, dtype=bool),
        "ground_return": np.zeros(count, dtype=bool),
        "free_to_far": np.ones(count, dtype=bool),
    }
    reported_points = np.full((count, 3), np.nan, dtype=np.float64)
    # Exact-equality grouping retains vectorization for the common-origin dense
    # case while still supporting a unique motion-compensated pose per ray.
    groups: dict[bytes, list[int]] = {}
    for index, origin in enumerate(np.ascontiguousarray(origin_array, dtype="<f8")):
        groups.setdefault(origin.tobytes(), []).append(index)
    for indices in groups.values():
        selection = np.asarray(indices, dtype=np.int64)
        first = first_hits(
            origin_array[indices[0]],
            normalized_directions[selection],
            environment_boxes=env_boxes,
            robot_primitives=robot_geoms,
            ground_z_m=ground_z_m,
            near_m=float(near),
            far_m=float(far),
        )
        for field_name, destination in scalar_fields.items():
            destination[selection] = getattr(first, field_name)
        reported_points[selection] = first.point_world_xyz_m

    raw_distance = scalar_fields["raw_distance_m"]
    finite = np.isfinite(raw_distance)
    raw_points = np.full((count, 3), np.nan, dtype=np.float64)
    raw_points[finite] = origin_array[finite] + raw_distance[finite, None] * normalized_directions[finite]
    hit_class = scalar_fields["hit_class"]
    hit_identity = scalar_fields["hit_identity"]
    hit_index = scalar_fields["hit_index"]
    kind = np.asarray([HIT_CLASS_NAMES[int(value)] for value in hit_class], dtype="<U12")
    identity_width = max([1, 6] + [len(item.identity) for item in env_boxes] + [len(item.identity) for item in robot_geoms])
    object_identity = np.full(count, "", dtype=f"<U{identity_width}")
    object_index = np.full(count, -1, dtype=np.int32)
    geom_identity = np.full(count, "", dtype=f"<U{identity_width}")
    geom_index = np.full(count, -1, dtype=np.int32)
    link_width = max([1] + [len(item.link_name) for item in robot_geoms])
    link_name = np.full(count, "", dtype=f"<U{link_width}")
    link_index = np.full(count, -1, dtype=np.int32)
    environment_mask = np.isin(hit_class, np.asarray([ENVIRONMENT_HIT, GROUND_HIT], dtype=np.int8))
    object_identity[environment_mask] = hit_identity[environment_mask]
    object_index[hit_class == ENVIRONMENT_HIT] = hit_index[hit_class == ENVIRONMENT_HIT]
    robot_lookup = {(int(item.geom_index), item.identity): item for item in robot_geoms}
    for ray_index in np.flatnonzero(hit_class == ROBOT_HIT):
        key = (int(hit_index[ray_index]), str(hit_identity[ray_index]))
        primitive = robot_lookup[key]
        geom_identity[ray_index] = primitive.identity
        geom_index[ray_index] = primitive.geom_index
        link_name[ray_index] = primitive.link_name
        link_index[ray_index] = primitive.link_index
    return SceneRaycastResult(
        origins_world_xyz_m=origin_array,
        directions_world=normalized_directions,
        raw_distance_m=raw_distance,
        distance_m=scalar_fields["reported_distance_m"],
        raw_point_world_xyz_m=raw_points,
        point_world_xyz_m=reported_points,
        hit_kind=kind,
        object_identity=object_identity,
        object_index=object_index,
        link_name=link_name,
        link_index=link_index,
        geom_identity=geom_identity,
        geom_index=geom_index,
        valid_return=scalar_fields["valid_return"],
        near_blind=scalar_fields["near_blind"],
        beyond_far=scalar_fields["beyond_far"],
        no_hit=scalar_fields["no_hit"],
        self_return=scalar_fields["self_return"],
        self_occluded_within_range=scalar_fields["self_occluded_within_range"],
        free_to_far=scalar_fields["free_to_far"],
    )


def _paired_ray_oriented_box_distances_normalized(
    origins_world: np.ndarray,
    directions_world: np.ndarray,
    centers_world: np.ndarray,
    half_extents_xyz_m: np.ndarray,
    quaternions_world_wxyz: np.ndarray,
) -> np.ndarray:
    count = len(directions_world)
    centers = np.broadcast_to(np.asarray(centers_world, dtype=np.float64), (count, 3))
    quaternions = np.broadcast_to(np.asarray(quaternions_world_wxyz, dtype=np.float64), (count, 4))
    half = _vector(half_extents_xyz_m, 3, "half_extents_xyz_m")
    rotations = rotation_matrices_wxyz(quaternions)
    delta = origins_world - centers
    local_origin = np.einsum("ni,nij->nj", delta, rotations)
    local_direction = np.einsum("ni,nij->nj", directions_world, rotations)
    lower = np.full(count, -np.inf, dtype=np.float64)
    upper = np.full(count, np.inf, dtype=np.float64)
    valid = np.ones(count, dtype=bool)
    for axis in range(3):
        component = local_direction[:, axis]
        parallel = np.abs(component) <= GEOMETRY_EPS
        valid &= ~(parallel & (np.abs(local_origin[:, axis]) > half[axis] + GEOMETRY_EPS))
        first = np.full(count, -np.inf, dtype=np.float64)
        second = np.full(count, np.inf, dtype=np.float64)
        active = ~parallel
        first[active] = (-half[axis] - local_origin[active, axis]) / component[active]
        second[active] = (half[axis] - local_origin[active, axis]) / component[active]
        lower = np.maximum(lower, np.minimum(first, second))
        upper = np.minimum(upper, np.maximum(first, second))
    entry = np.maximum(lower, 0.0)
    valid &= upper + GEOMETRY_EPS >= entry
    return np.where(valid, entry, np.inf)


def _paired_ray_sphere_distances_normalized(
    origins_world: np.ndarray,
    directions_world: np.ndarray,
    centers_world: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive")
    centers = np.broadcast_to(np.asarray(centers_world, dtype=np.float64), origins_world.shape)
    offset = origins_world - centers
    c = np.sum(offset * offset, axis=1) - radius * radius
    inside = c <= GEOMETRY_EPS
    b = np.sum(directions_world * offset, axis=1)
    discriminant = b * b - c
    tangent_or_hit = discriminant >= -GEOMETRY_EPS
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near_distance = -b - root
    far_distance = -b + root
    distance = np.where(
        near_distance >= -GEOMETRY_EPS,
        np.maximum(near_distance, 0.0),
        np.where(
            far_distance >= -GEOMETRY_EPS,
            np.maximum(far_distance, 0.0),
            np.inf,
        ),
    )
    return np.where(inside, 0.0, np.where(tangent_or_hit, distance, np.inf))


def _paired_ray_capsule_distances_normalized(
    origins_world: np.ndarray,
    directions_world: np.ndarray,
    centers_world: np.ndarray,
    quaternions_world_wxyz: np.ndarray,
    radius_m: float,
    centerline_length_m: float,
) -> np.ndarray:
    radius = float(radius_m)
    length = float(centerline_length_m)
    if not math.isfinite(radius) or radius <= 0.0 or not math.isfinite(length) or length < 0.0:
        raise ValueError("capsule radius must be positive and length nonnegative")
    if length <= GEOMETRY_EPS:
        return _paired_ray_sphere_distances_normalized(
            origins_world, directions_world, centers_world, radius
        )
    count = len(directions_world)
    centers = np.broadcast_to(np.asarray(centers_world, dtype=np.float64), (count, 3))
    quaternions = np.broadcast_to(np.asarray(quaternions_world_wxyz, dtype=np.float64), (count, 4))
    rotations = rotation_matrices_wxyz(quaternions)
    delta = origins_world - centers
    local_origin = np.einsum("ni,nij->nj", delta, rotations)
    local_direction = np.einsum("ni,nij->nj", directions_world, rotations)
    half = 0.5 * length
    closest_z = np.clip(local_origin[:, 2], -half, half)
    closest = np.column_stack((np.zeros(count), np.zeros(count), closest_z))
    inside = np.linalg.norm(local_origin - closest, axis=1) <= radius + GEOMETRY_EPS

    output = np.full(count, np.inf, dtype=np.float64)
    radial_a = np.sum(local_direction[:, :2] ** 2, axis=1)
    radial_b = np.sum(local_direction[:, :2] * local_origin[:, :2], axis=1)
    radial_c = np.sum(local_origin[:, :2] ** 2, axis=1) - radius * radius
    discriminant = radial_b * radial_b - radial_a * radial_c
    active = (radial_a > GEOMETRY_EPS) & (discriminant >= -GEOMETRY_EPS)
    square = np.sqrt(np.maximum(discriminant, 0.0))
    for sign in (-1.0, 1.0):
        root = np.full(count, np.inf, dtype=np.float64)
        root[active] = (-radial_b[active] + sign * square[active]) / radial_a[active]
        z = local_origin[:, 2] + root * local_direction[:, 2]
        valid = active & (root >= -GEOMETRY_EPS) & (z >= -half - GEOMETRY_EPS) & (z <= half + GEOMETRY_EPS)
        output = np.minimum(output, np.where(valid, np.maximum(root, 0.0), np.inf))
    local_axis_world = rotations[..., :, 2]
    for cap_z in (-half, half):
        cap_centers = centers + cap_z * local_axis_world
        output = np.minimum(
            output,
            _paired_ray_sphere_distances_normalized(
                origins_world, directions_world, cap_centers, radius
            ),
        )
    output[inside] = 0.0
    return output


def _paired_ray_ground_distances_normalized(
    origins_world: np.ndarray,
    directions_world: np.ndarray,
    ground_z_m: float,
) -> np.ndarray:
    height = float(ground_z_m)
    if not math.isfinite(height):
        raise ValueError("ground_z_m must be finite")
    on_plane = np.abs(origins_world[:, 2] - height) <= GEOMETRY_EPS
    vertical = directions_world[:, 2]
    active = np.abs(vertical) > GEOMETRY_EPS
    distance = np.full(len(directions_world), np.inf, dtype=np.float64)
    distance[active] = (height - origins_world[active, 2]) / vertical[active]
    distance = np.where(distance >= -GEOMETRY_EPS, np.maximum(distance, 0.0), np.inf)
    return np.where(on_plane, 0.0, distance)


def raycast_moving_scene(
    origins: Sequence[Sequence[float]] | np.ndarray,
    directions: Sequence[Sequence[float]] | np.ndarray,
    near: float,
    far: float,
    env_boxes: Sequence[OrientedBox],
    robot_specs: Sequence[RobotPrimitiveSpec],
    robot_positions_world_xyz_m: np.ndarray,
    robot_quaternions_world_wxyz: np.ndarray,
    ground: bool | float | None = True,
) -> SceneRaycastResult:
    """Vectorized first hits for per-ray origins and articulated robot poses.

    Robot pose arrays have shape ``[rays, geoms, 3|4]`` and their geometry
    dimension is aligned to ``robot_specs`` before deterministic geom-index
    ordering.  Environment OBBs and ground are static.  All tie, blind-region,
    range-filtering and identity semantics are identical to :func:`first_hits`.
    """

    normalized_directions = _directions(directions)
    origin_array = np.asarray(origins, dtype=np.float64)
    if origin_array.shape != (len(normalized_directions), 3) or not np.isfinite(origin_array).all():
        raise ValueError("origins must have finite shape [rays,3]")
    specs = tuple(robot_specs)
    positions = np.asarray(robot_positions_world_xyz_m, dtype=np.float64)
    quaternions = np.asarray(robot_quaternions_world_wxyz, dtype=np.float64)
    expected_positions = (len(normalized_directions), len(specs), 3)
    expected_quaternions = (len(normalized_directions), len(specs), 4)
    if positions.shape != expected_positions or not np.isfinite(positions).all():
        raise ValueError(f"robot positions must have shape {expected_positions}")
    if quaternions.shape != expected_quaternions or not np.isfinite(quaternions).all():
        raise ValueError(f"robot quaternions must have shape {expected_quaternions}")
    normalize_quaternions_wxyz(quaternions)
    near_value = float(near)
    far_value = float(far)
    if not math.isfinite(near_value) or near_value < 0.0 or not math.isfinite(far_value) or far_value <= near_value:
        raise ValueError("range contract requires 0 <= near < far")
    if ground is True:
        ground_z_m: float | None = 0.0
    elif ground is False or ground is None:
        ground_z_m = None
    else:
        ground_z_m = float(ground)
        if not math.isfinite(ground_z_m):
            raise ValueError("ground height must be finite")

    count = len(normalized_directions)
    identity_width = max(
        [1, 6]
        + [len(item.identity) for item in env_boxes]
        + [len(item.identity) for item in specs]
    )
    environment_distance = np.full(count, np.inf, dtype=np.float64)
    environment_class = np.full(count, NO_HIT, dtype=np.int8)
    environment_identity = np.full(count, "", dtype=f"<U{identity_width}")
    environment_index = np.full(count, -1, dtype=np.int32)
    ordered_environment = sorted(
        env_boxes,
        key=lambda value: (
            value.object_index if value.object_index >= 0 else np.iinfo(np.int32).max,
            value.identity,
        ),
    )
    for box in ordered_environment:
        distance = _paired_ray_oriented_box_distances_normalized(
            origin_array,
            normalized_directions,
            np.asarray(box.center_xyz_m, dtype=np.float64),
            np.asarray(box.half_extents_xyz_m, dtype=np.float64),
            np.asarray(box.quaternion_wxyz, dtype=np.float64),
        )
        better = _strictly_better(distance, environment_distance)
        environment_distance[better] = distance[better]
        environment_class[better] = ENVIRONMENT_HIT
        environment_identity[better] = box.identity
        environment_index[better] = int(box.object_index)
    if ground_z_m is not None:
        distance = _paired_ray_ground_distances_normalized(
            origin_array, normalized_directions, ground_z_m
        )
        better = _strictly_better(distance, environment_distance)
        environment_distance[better] = distance[better]
        environment_class[better] = GROUND_HIT
        environment_identity[better] = "ground"
        environment_index[better] = -1

    robot_distance = np.full(count, np.inf, dtype=np.float64)
    robot_identity = np.full(count, "", dtype=f"<U{identity_width}")
    robot_index = np.full(count, -1, dtype=np.int32)
    spec_order = sorted(range(len(specs)), key=lambda index: (specs[index].geom_index, specs[index].identity))
    for spec_index in spec_order:
        spec = specs[spec_index]
        data = np.asarray(spec.data, dtype=np.float64)
        if spec.kind == "sphere":
            distance = _paired_ray_sphere_distances_normalized(
                origin_array,
                normalized_directions,
                positions[:, spec_index],
                float(data[0]),
            )
        elif spec.kind == "capsule":
            distance = _paired_ray_capsule_distances_normalized(
                origin_array,
                normalized_directions,
                positions[:, spec_index],
                quaternions[:, spec_index],
                float(data[0]),
                float(data[1]),
            )
        else:
            distance = _paired_ray_oriented_box_distances_normalized(
                origin_array,
                normalized_directions,
                positions[:, spec_index],
                data[:3] * 0.5,
                quaternions[:, spec_index],
            )
        better = _strictly_better(distance, robot_distance)
        robot_distance[better] = distance[better]
        robot_identity[better] = spec.identity
        robot_index[better] = int(spec.geom_index)

    robot_wins = np.isfinite(robot_distance) & (
        ~np.isfinite(environment_distance)
        | (robot_distance <= environment_distance + DISTANCE_TIE_ATOL_M)
    )
    raw_distance = np.where(robot_wins, robot_distance, environment_distance)
    hit_class = np.where(robot_wins, ROBOT_HIT, environment_class).astype(np.int8)
    hit_identity = np.where(robot_wins, robot_identity, environment_identity)
    hit_index = np.where(robot_wins, robot_index, environment_index).astype(np.int32)
    finite = np.isfinite(raw_distance)
    near_blind = finite & (raw_distance < near_value - DISTANCE_TIE_ATOL_M)
    beyond_far = finite & (raw_distance > far_value + DISTANCE_TIE_ATOL_M)
    valid = finite & ~near_blind & ~beyond_far
    reported = np.where(valid, raw_distance, np.nan)
    raw_points = np.full((count, 3), np.nan, dtype=np.float64)
    raw_points[finite] = origin_array[finite] + raw_distance[finite, None] * normalized_directions[finite]
    reported_points = np.full((count, 3), np.nan, dtype=np.float64)
    reported_points[valid] = raw_points[valid]
    self_return = hit_class == ROBOT_HIT
    no_hit = ~finite
    free_to_far = no_hit | beyond_far

    kind = np.asarray([HIT_CLASS_NAMES[int(value)] for value in hit_class], dtype="<U12")
    object_identity = np.full(count, "", dtype=f"<U{identity_width}")
    object_index = np.full(count, -1, dtype=np.int32)
    geom_identity = np.full(count, "", dtype=f"<U{identity_width}")
    geom_index = np.full(count, -1, dtype=np.int32)
    link_width = max([1] + [len(item.link_name) for item in specs])
    link_name = np.full(count, "", dtype=f"<U{link_width}")
    link_index = np.full(count, -1, dtype=np.int32)
    environment_mask = np.isin(hit_class, np.asarray([ENVIRONMENT_HIT, GROUND_HIT], dtype=np.int8))
    object_identity[environment_mask] = hit_identity[environment_mask]
    object_index[hit_class == ENVIRONMENT_HIT] = hit_index[hit_class == ENVIRONMENT_HIT]
    spec_lookup = {(int(item.geom_index), item.identity): item for item in specs}
    for ray_index in np.flatnonzero(self_return):
        spec = spec_lookup[(int(hit_index[ray_index]), str(hit_identity[ray_index]))]
        geom_identity[ray_index] = spec.identity
        geom_index[ray_index] = spec.geom_index
        link_name[ray_index] = spec.link_name
        link_index[ray_index] = spec.link_index
    return SceneRaycastResult(
        origins_world_xyz_m=origin_array,
        directions_world=normalized_directions,
        raw_distance_m=raw_distance,
        distance_m=reported,
        raw_point_world_xyz_m=raw_points,
        point_world_xyz_m=reported_points,
        hit_kind=kind,
        object_identity=object_identity,
        object_index=object_index,
        link_name=link_name,
        link_index=link_index,
        geom_identity=geom_identity,
        geom_index=geom_index,
        valid_return=valid,
        near_blind=near_blind,
        beyond_far=beyond_far,
        no_hit=no_hit,
        self_return=self_return,
        self_occluded_within_range=self_return & (raw_distance <= far_value + DISTANCE_TIE_ATOL_M),
        free_to_far=free_to_far,
    )


def _point_oriented_box_clearance(
    points_world_xyz_m: np.ndarray,
    center_xyz_m: np.ndarray,
    half_extents_xyz_m: np.ndarray,
    quaternion_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    rotation = rotation_matrix_wxyz(quaternion_wxyz)
    local = (points_world_xyz_m - center_xyz_m[None, :]) @ rotation
    q = np.abs(local) - half_extents_xyz_m[None, :]
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def point_to_primitive_clearance(
    points_world_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    primitive: RobotPrimitive,
) -> np.ndarray:
    """Signed point-to-primitive clearance for one Genesis collision shape."""

    points = np.asarray(points_world_xyz_m, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world_xyz_m must have shape [N,3] and be finite")
    position = _vector(primitive.position_xyz_m, 3, "position_xyz_m")
    data = np.asarray(primitive.data, dtype=np.float64)
    rotation = rotation_matrix_wxyz(primitive.quaternion_wxyz)
    local = (points - position[None, :]) @ rotation
    if primitive.kind == "sphere":
        return np.linalg.norm(local, axis=1) - float(data[0])
    if primitive.kind == "capsule":
        z = np.clip(local[:, 2], -float(data[1]) * 0.5, float(data[1]) * 0.5)
        centerline = np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=1)
        return np.linalg.norm(local - centerline, axis=1) - float(data[0])
    # Genesis stores full extents.  The previous upper-bound helper treated
    # these values as half extents; this qualification corrects that mismatch.
    return _point_oriented_box_clearance(
        points,
        position,
        data[:3] * 0.5,
        primitive.quaternion_wxyz,
    )


@dataclass(frozen=True)
class PrimitiveOBBClosestWitness:
    """Exact closest-witness result for one robot primitive and scene OBB.

    Positive ``signed_clearance_m`` is exact Euclidean solid separation.  Zero
    denotes touching.  Negative values conservatively identify intersection;
    sphere/capsule values use the exact axis/centre-to-box distance minus the
    radius, while intersecting OBBs use the minimum normalized SAT overlap.
    The environment witness is always on the environment OBB surface, including
    containment cases where an ordinary point-to-solid clamp would be interior.
    """

    primitive_identity: str
    environment_identity: str
    primitive_point_world_xyz_m: tuple[float, float, float]
    environment_point_world_xyz_m: tuple[float, float, float]
    signed_clearance_m: float
    separation_distance_m: float
    intersects: bool
    feature_pair: str


def _point_aabb_solid_closest_local(
    point_local: np.ndarray, half_extent: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    point = _vector(point_local, 3, "point_local")
    half = _vector(half_extent, 3, "half_extent")
    if np.any(half <= 0.0):
        raise ValueError("half_extent must be positive")
    closest = np.clip(point, -half, half)
    delta = point - closest
    distance = float(np.linalg.norm(delta))
    inside = bool(np.all(np.abs(point) <= half + GEOMETRY_EPS))
    return closest, distance, inside


def _point_aabb_surface_closest_local(
    point_local: np.ndarray, half_extent: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return a deterministic closest point on an AABB *surface*."""

    point = _vector(point_local, 3, "point_local")
    half = _vector(half_extent, 3, "half_extent")
    solid, distance, inside = _point_aabb_solid_closest_local(point, half)
    if not inside:
        return solid, distance
    slack = half - np.abs(point)
    axis = int(np.argmin(slack))
    surface = point.copy()
    # Positive wins the exact zero tie, making the result byte deterministic.
    surface[axis] = half[axis] if point[axis] >= 0.0 else -half[axis]
    return surface, float(np.linalg.norm(surface - point))


def _segment_aabb_closest_local(
    start_local: np.ndarray,
    end_local: np.ndarray,
    half_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact closest points between a finite segment and a solid AABB.

    Squared distance to an AABB is a convex piecewise quadratic along the
    segment.  The only active-set changes occur where a coordinate crosses an
    AABB slab, so enumerating those breakpoints and each interval's stationary
    point is complete and avoids iterative optimization.
    """

    start = _vector(start_local, 3, "start_local")
    end = _vector(end_local, 3, "end_local")
    half = _vector(half_extent, 3, "half_extent")
    if np.any(half <= 0.0):
        raise ValueError("half_extent must be positive")
    direction = end - start
    breaks = [0.0, 1.0]
    for axis in range(3):
        if abs(float(direction[axis])) <= GEOMETRY_EPS:
            continue
        for boundary in (-half[axis], half[axis]):
            value = float((boundary - start[axis]) / direction[axis])
            if GEOMETRY_EPS < value < 1.0 - GEOMETRY_EPS:
                breaks.append(value)
    ordered = sorted(set(breaks))
    candidates = list(ordered)
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        middle = 0.5 * (lower + upper)
        point = start + middle * direction
        coefficients: list[tuple[float, float]] = []
        for axis in range(3):
            if point[axis] < -half[axis]:
                coefficients.append((-float(direction[axis]), float(-half[axis] - start[axis])))
            elif point[axis] > half[axis]:
                coefficients.append((float(direction[axis]), float(start[axis] - half[axis])))
        denominator = sum(slope * slope for slope, _intercept in coefficients)
        if denominator > GEOMETRY_EPS:
            stationary = -sum(
                slope * intercept for slope, intercept in coefficients
            ) / denominator
            candidates.append(float(np.clip(stationary, lower, upper)))

    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for parameter in sorted(set(candidates)):
        segment_point = start + parameter * direction
        box_point = np.clip(segment_point, -half, half)
        squared = float(np.dot(segment_point - box_point, segment_point - box_point))
        key = (squared, float(parameter))
        if best is None or key < best[:2]:
            best = (squared, float(parameter), segment_point, box_point)
    assert best is not None
    return best[2], best[3], math.sqrt(max(0.0, best[0])), best[1]


def _segment_aabb_surface_witness_local(
    start_local: np.ndarray,
    end_local: np.ndarray,
    half_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest segment/AABB-surface pair, including segment containment."""

    start = _vector(start_local, 3, "start_local")
    end = _vector(end_local, 3, "end_local")
    half = _vector(half_extent, 3, "half_extent")
    direction = end - start
    candidates: list[tuple[float, float, int, np.ndarray, np.ndarray]] = []

    for ordinal, (parameter, point) in enumerate(((0.0, start), (1.0, end))):
        surface, distance = _point_aabb_surface_closest_local(point, half)
        candidates.append((distance, parameter, ordinal, point.copy(), surface))

    ordinal = 2
    for axis in range(3):
        if abs(float(direction[axis])) <= GEOMETRY_EPS:
            continue
        other = [value for value in range(3) if value != axis]
        for sign in (-1.0, 1.0):
            parameter = float((sign * half[axis] - start[axis]) / direction[axis])
            if -GEOMETRY_EPS <= parameter <= 1.0 + GEOMETRY_EPS:
                parameter = float(np.clip(parameter, 0.0, 1.0))
                point = start + parameter * direction
                if all(abs(float(point[value])) <= half[value] + GEOMETRY_EPS for value in other):
                    surface = point.copy()
                    surface[axis] = sign * half[axis]
                    candidates.append((0.0, parameter, ordinal, point, surface))
            ordinal += 1
    best = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    return best[3], best[4], float(best[0])


def _point_segment_closest(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, float, float]:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    parameter = 0.0 if denominator <= GEOMETRY_EPS else float(
        np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0)
    )
    closest = start + parameter * direction
    return closest, float(np.linalg.norm(point - closest)), parameter


def _segment_segment_closest(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact closest pair for two finite 3-D segments."""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    aa = float(np.dot(first_direction, first_direction))
    bb = float(np.dot(first_direction, second_direction))
    cc = float(np.dot(second_direction, second_direction))
    delta = first_start - second_start
    dd = float(np.dot(first_direction, delta))
    ee = float(np.dot(second_direction, delta))
    denominator = aa * cc - bb * bb
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    if aa > GEOMETRY_EPS and cc > GEOMETRY_EPS and denominator > GEOMETRY_EPS:
        first_parameter = (bb * ee - cc * dd) / denominator
        second_parameter = (aa * ee - bb * dd) / denominator
        if 0.0 <= first_parameter <= 1.0 and 0.0 <= second_parameter <= 1.0:
            first = first_start + first_parameter * first_direction
            second = second_start + second_parameter * second_direction
            candidates.append((float(np.linalg.norm(first - second)), 0, first, second))

    second, distance, _ = _point_segment_closest(first_start, second_start, second_end)
    candidates.append((distance, 1, first_start.copy(), second))
    second, distance, _ = _point_segment_closest(first_end, second_start, second_end)
    candidates.append((distance, 2, first_end.copy(), second))
    first, distance, _ = _point_segment_closest(second_start, first_start, first_end)
    candidates.append((distance, 3, first, second_start.copy()))
    first, distance, _ = _point_segment_closest(second_end, first_start, first_end)
    candidates.append((distance, 4, first, second_end.copy()))
    best = min(candidates, key=lambda row: (row[0], row[1]))
    return best[2], best[3], float(best[0])


_BOX_SIGNS = tuple(
    (x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)
)
_BOX_EDGES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if sum(_BOX_SIGNS[first][axis] != _BOX_SIGNS[second][axis] for axis in range(3)) == 1
)


def _obb_vertices(
    center: np.ndarray, rotation: np.ndarray, half_extent: np.ndarray
) -> np.ndarray:
    local = np.asarray(_BOX_SIGNS, dtype=np.float64) * half_extent[None, :]
    return local @ rotation.T + center[None, :]


def _point_obb_solid_closest(
    point_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world: np.ndarray,
    half_extent: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    local = (point_world - center_world) @ rotation_world
    closest_local, distance, inside = _point_aabb_solid_closest_local(local, half_extent)
    return closest_local @ rotation_world.T + center_world, distance, inside


def _point_obb_surface_closest(
    point_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world: np.ndarray,
    half_extent: np.ndarray,
) -> tuple[np.ndarray, float]:
    local = (point_world - center_world) @ rotation_world
    closest_local, distance = _point_aabb_surface_closest_local(local, half_extent)
    return closest_local @ rotation_world.T + center_world, distance


def _obb_sat_intersection_and_overlap(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half: np.ndarray,
) -> tuple[bool, float]:
    axes = [first_rotation[:, axis] for axis in range(3)]
    axes.extend(second_rotation[:, axis] for axis in range(3))
    axes.extend(
        np.cross(first_rotation[:, first], second_rotation[:, second])
        for first in range(3)
        for second in range(3)
    )
    delta = second_center - first_center
    minimum_overlap = math.inf
    for raw_axis in axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= 1.0e-10:
            continue
        axis = raw_axis / norm
        first_radius = float(np.sum(first_half * np.abs(first_rotation.T @ axis)))
        second_radius = float(np.sum(second_half * np.abs(second_rotation.T @ axis)))
        overlap = first_radius + second_radius - abs(float(delta @ axis))
        if overlap < -DISTANCE_TIE_ATOL_M:
            return False, 0.0
        minimum_overlap = min(minimum_overlap, max(0.0, overlap))
    return True, 0.0 if not math.isfinite(minimum_overlap) else minimum_overlap


def _obb_obb_surface_closest(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Complete vertex-face and edge-edge closest candidates for two OBBs."""

    first_vertices = _obb_vertices(first_center, first_rotation, first_half)
    second_vertices = _obb_vertices(second_center, second_rotation, second_half)
    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray, str]] = []
    ordinal = 0
    for vertex in first_vertices:
        environment, distance = _point_obb_surface_closest(
            vertex, second_center, second_rotation, second_half
        )
        candidates.append((distance, 0, ordinal, vertex.copy(), environment, "PRIMITIVE_VERTEX_ENVIRONMENT_FACE"))
        ordinal += 1
    for vertex in second_vertices:
        primitive, distance, _inside = _point_obb_solid_closest(
            vertex, first_center, first_rotation, first_half
        )
        candidates.append((distance, 1, ordinal, primitive, vertex.copy(), "ENVIRONMENT_VERTEX_PRIMITIVE_FACE"))
        ordinal += 1
    for first_edge_index, (first_a, first_b) in enumerate(_BOX_EDGES):
        for second_edge_index, (second_a, second_b) in enumerate(_BOX_EDGES):
            primitive, environment, distance = _segment_segment_closest(
                first_vertices[first_a],
                first_vertices[first_b],
                second_vertices[second_a],
                second_vertices[second_b],
            )
            candidates.append(
                (
                    distance,
                    2,
                    first_edge_index * len(_BOX_EDGES) + second_edge_index,
                    primitive,
                    environment,
                    "EDGE_EDGE",
                )
            )
    best = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    return best[3], best[4], float(best[0]), best[5]


@dataclass(frozen=True)
class BatchedBoxOBBClosestWitness:
    """Numerical closest witnesses for aligned robot-box/environment-OBB pairs.

    The batch path enumerates the same complete feature set as
    :func:`primitive_obb_closest_witness`: eight robot vertices against the
    environment faces, eight environment vertices against the robot solid,
    and every one of the 144 edge pairs.  The signed intersecting clearance is
    obtained from the same 15-axis SAT calculation.

    Separated witnesses agree with the scalar routine up to float64 round-off.
    Intersections can have several equally valid surface witnesses; callers
    that bind the scalar routine's exact intersecting witness identity must
    replace only their selected intersecting rows with the scalar result.
    """

    signed_clearance_m: np.ndarray
    primitive_point_world_xyz_m: np.ndarray
    environment_point_world_xyz_m: np.ndarray
    intersects: np.ndarray
    feature_pair: tuple[str, ...]


def _batch_box_obb_closest_witness_chunk(
    primitive_centers_world_xyz_m: np.ndarray,
    primitive_quaternions_world_wxyz: np.ndarray,
    primitive_full_extents_xyz_m: np.ndarray,
    environment_centers_world_xyz_m: np.ndarray,
    environment_quaternions_world_wxyz: np.ndarray,
    environment_half_extents_xyz_m: np.ndarray,
) -> BatchedBoxOBBClosestWitness:
    """Vectorized implementation for one memory-bounded aligned chunk."""

    count = len(primitive_centers_world_xyz_m)
    primitive_rotation = rotation_matrices_wxyz(
        primitive_quaternions_world_wxyz
    )
    environment_rotation = rotation_matrices_wxyz(
        environment_quaternions_world_wxyz
    )
    primitive_half = primitive_full_extents_xyz_m * 0.5

    primitive_local_vertices = (
        np.asarray(_BOX_SIGNS, dtype=np.float64)[None, :, :]
        * primitive_half[:, None, :]
    )
    environment_local_vertices = (
        np.asarray(_BOX_SIGNS, dtype=np.float64)[None, :, :]
        * environment_half_extents_xyz_m[:, None, :]
    )
    primitive_vertices = (
        np.einsum(
            "pvi,pji->pvj", primitive_local_vertices, primitive_rotation
        )
        + primitive_centers_world_xyz_m[:, None, :]
    )
    environment_vertices = (
        np.einsum(
            "pvi,pji->pvj", environment_local_vertices, environment_rotation
        )
        + environment_centers_world_xyz_m[:, None, :]
    )

    # Candidate class 0: robot vertices against the environment surface.
    primitive_in_environment = np.einsum(
        "pvi,pij->pvj",
        primitive_vertices - environment_centers_world_xyz_m[:, None, :],
        environment_rotation,
    )
    environment_local = np.clip(
        primitive_in_environment,
        -environment_half_extents_xyz_m[:, None, :],
        environment_half_extents_xyz_m[:, None, :],
    )
    inside = np.all(
        np.abs(primitive_in_environment)
        <= environment_half_extents_xyz_m[:, None, :] + GEOMETRY_EPS,
        axis=2,
    )
    if np.any(inside):
        inside_pair, inside_vertex = np.nonzero(inside)
        slack = environment_half_extents_xyz_m[:, None, :] - np.abs(
            primitive_in_environment
        )
        surface_axis = np.argmin(slack, axis=2)
        selected_axis = surface_axis[inside_pair, inside_vertex]
        selected_coordinate = primitive_in_environment[
            inside_pair, inside_vertex, selected_axis
        ]
        sign = np.where(selected_coordinate >= 0.0, 1.0, -1.0)
        environment_local[
            inside_pair, inside_vertex, selected_axis
        ] = sign * environment_half_extents_xyz_m[
            inside_pair, selected_axis
        ]
    primitive_vertex_environment_points = (
        np.einsum(
            "pvi,pji->pvj", environment_local, environment_rotation
        )
        + environment_centers_world_xyz_m[:, None, :]
    )
    primitive_vertex_distances = np.linalg.norm(
        primitive_vertices - primitive_vertex_environment_points, axis=2
    )

    # Candidate class 1: environment vertices against the robot solid.
    environment_in_primitive = np.einsum(
        "pvi,pij->pvj",
        environment_vertices - primitive_centers_world_xyz_m[:, None, :],
        primitive_rotation,
    )
    primitive_local = np.clip(
        environment_in_primitive,
        -primitive_half[:, None, :],
        primitive_half[:, None, :],
    )
    environment_vertex_primitive_points = (
        np.einsum("pvi,pji->pvj", primitive_local, primitive_rotation)
        + primitive_centers_world_xyz_m[:, None, :]
    )
    environment_vertex_distances = np.linalg.norm(
        environment_vertex_primitive_points - environment_vertices, axis=2
    )

    # Candidate class 2: the complete 12 x 12 edge cross-product.
    edge_indices = np.asarray(_BOX_EDGES, dtype=np.int16)
    first_start = primitive_vertices[:, edge_indices[:, 0]][:, :, None, :]
    first_end = primitive_vertices[:, edge_indices[:, 1]][:, :, None, :]
    second_start = environment_vertices[:, edge_indices[:, 0]][:, None, :, :]
    second_end = environment_vertices[:, edge_indices[:, 1]][:, None, :, :]
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    delta = first_start - second_start
    aa = np.sum(first_direction * first_direction, axis=3)
    bb = np.sum(first_direction * second_direction, axis=3)
    cc = np.sum(second_direction * second_direction, axis=3)
    dd = np.sum(first_direction * delta, axis=3)
    ee = np.sum(second_direction * delta, axis=3)
    denominator = aa * cc - bb * bb
    nonparallel = denominator > GEOMETRY_EPS
    safe_denominator = np.where(nonparallel, denominator, 1.0)
    first_parameter = (bb * ee - cc * dd) / safe_denominator
    second_parameter = (aa * ee - bb * dd) / safe_denominator
    first_interior = first_start + first_parameter[..., None] * first_direction
    second_interior = second_start + second_parameter[..., None] * second_direction
    interior_valid = (
        nonparallel
        & (first_parameter >= 0.0)
        & (first_parameter <= 1.0)
        & (second_parameter >= 0.0)
        & (second_parameter <= 1.0)
    )

    edge_squared: list[np.ndarray] = [
        np.where(
            interior_valid,
            np.sum((first_interior - second_interior) ** 2, axis=3),
            np.inf,
        )
    ]
    edge_primitive_points: list[np.ndarray] = [first_interior]
    edge_environment_points: list[np.ndarray] = [second_interior]
    endpoint_cases = (
        (first_start, second_start, second_direction, cc, True),
        (first_end, second_start, second_direction, cc, True),
        (second_start, first_start, first_direction, aa, False),
        (second_end, first_start, first_direction, aa, False),
    )
    for point, segment_start, segment_direction, squared_length, primitive_first in endpoint_cases:
        parameter = np.clip(
            np.sum((point - segment_start) * segment_direction, axis=3)
            / squared_length,
            0.0,
            1.0,
        )
        closest = segment_start + parameter[..., None] * segment_direction
        if primitive_first:
            primitive_point = np.broadcast_to(point, closest.shape)
            environment_point = closest
        else:
            primitive_point = closest
            environment_point = np.broadcast_to(point, closest.shape)
        edge_squared.append(
            np.sum((primitive_point - environment_point) ** 2, axis=3)
        )
        edge_primitive_points.append(primitive_point)
        edge_environment_points.append(environment_point)

    edge_candidate_squared = np.stack(edge_squared, axis=3)
    edge_candidate = np.argmin(edge_candidate_squared, axis=3)
    pair_index, first_edge, second_edge = np.indices(edge_candidate.shape)
    stacked_primitive = np.stack(edge_primitive_points, axis=3)
    stacked_environment = np.stack(edge_environment_points, axis=3)
    edge_primitive = stacked_primitive[
        pair_index, first_edge, second_edge, edge_candidate
    ]
    edge_environment = stacked_environment[
        pair_index, first_edge, second_edge, edge_candidate
    ]
    edge_distance = np.sqrt(
        np.take_along_axis(
            edge_candidate_squared, edge_candidate[..., None], axis=3
        )[..., 0]
    )

    all_distances = np.concatenate(
        (
            primitive_vertex_distances,
            environment_vertex_distances,
            edge_distance.reshape(count, -1),
        ),
        axis=1,
    )
    winning_candidate = np.argmin(all_distances, axis=1)
    surface_distance = all_distances[np.arange(count), winning_candidate]
    primitive_point = np.empty((count, 3), dtype=np.float64)
    environment_point = np.empty((count, 3), dtype=np.float64)
    feature = np.full(count, "EDGE_EDGE", dtype="<U36")

    primitive_vertex_winner = winning_candidate < 8
    if np.any(primitive_vertex_winner):
        row = np.flatnonzero(primitive_vertex_winner)
        vertex = winning_candidate[row]
        primitive_point[row] = primitive_vertices[row, vertex]
        environment_point[row] = primitive_vertex_environment_points[row, vertex]
        feature[row] = "PRIMITIVE_VERTEX_ENVIRONMENT_FACE"
    environment_vertex_winner = (winning_candidate >= 8) & (winning_candidate < 16)
    if np.any(environment_vertex_winner):
        row = np.flatnonzero(environment_vertex_winner)
        vertex = winning_candidate[row] - 8
        primitive_point[row] = environment_vertex_primitive_points[row, vertex]
        environment_point[row] = environment_vertices[row, vertex]
        feature[row] = "ENVIRONMENT_VERTEX_PRIMITIVE_FACE"
    edge_winner = winning_candidate >= 16
    if np.any(edge_winner):
        row = np.flatnonzero(edge_winner)
        edge = winning_candidate[row] - 16
        first_edge = edge // len(_BOX_EDGES)
        second_edge = edge % len(_BOX_EDGES)
        primitive_point[row] = edge_primitive[row, first_edge, second_edge]
        environment_point[row] = edge_environment[row, first_edge, second_edge]

    # The exact same normalized 15-axis SAT overlap used by the scalar path.
    primitive_axes = np.swapaxes(primitive_rotation, 1, 2)
    environment_axes = np.swapaxes(environment_rotation, 1, 2)
    cross_axes = np.cross(
        primitive_axes[:, :, None, :], environment_axes[:, None, :, :]
    ).reshape(count, 9, 3)
    axes = np.concatenate((primitive_axes, environment_axes, cross_axes), axis=1)
    axis_norm = np.linalg.norm(axes, axis=2)
    valid_axis = axis_norm > 1.0e-10
    unit_axis = axes / np.where(valid_axis, axis_norm, 1.0)[..., None]
    primitive_radius = np.sum(
        primitive_half[:, None, :]
        * np.abs(np.einsum("pai,pij->paj", unit_axis, primitive_rotation)),
        axis=2,
    )
    environment_radius = np.sum(
        environment_half_extents_xyz_m[:, None, :]
        * np.abs(np.einsum("pai,pij->paj", unit_axis, environment_rotation)),
        axis=2,
    )
    center_projection = np.abs(
        np.sum(
            (
                environment_centers_world_xyz_m
                - primitive_centers_world_xyz_m
            )[:, None, :]
            * unit_axis,
            axis=2,
        )
    )
    overlap = primitive_radius + environment_radius - center_projection
    intersects = np.all(
        (~valid_axis) | (overlap >= -DISTANCE_TIE_ATOL_M), axis=1
    )
    minimum_overlap = np.min(
        np.where(valid_axis, np.maximum(0.0, overlap), np.inf), axis=1
    )
    signed = np.where(intersects, -minimum_overlap, surface_distance)
    signed[np.abs(signed) <= DISTANCE_TIE_ATOL_M] = 0.0
    return BatchedBoxOBBClosestWitness(
        signed_clearance_m=signed,
        primitive_point_world_xyz_m=primitive_point,
        environment_point_world_xyz_m=environment_point,
        intersects=intersects,
        feature_pair=tuple(str(value) for value in feature),
    )


def batch_box_obb_closest_witness(
    primitive_centers_world_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    primitive_quaternions_world_wxyz: Sequence[Sequence[float]] | np.ndarray,
    primitive_full_extents_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    environment_centers_world_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    environment_quaternions_world_wxyz: Sequence[Sequence[float]] | np.ndarray,
    environment_half_extents_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    chunk_size: int = 1024,
) -> BatchedBoxOBBClosestWitness:
    """Batch aligned exact box--OBB signed clearances and closest witnesses.

    Inputs are aligned ``[pairs,3|4]`` arrays, not a Cartesian product.  Robot
    dimensions follow Genesis and are full extents; environment dimensions
    are half extents.  Chunking bounds the 12-by-12 edge-product workspace.
    """

    primitive_centers = np.asarray(
        primitive_centers_world_xyz_m, dtype=np.float64
    )
    primitive_quaternions = np.asarray(
        primitive_quaternions_world_wxyz, dtype=np.float64
    )
    primitive_extents = np.asarray(
        primitive_full_extents_xyz_m, dtype=np.float64
    )
    environment_centers = np.asarray(
        environment_centers_world_xyz_m, dtype=np.float64
    )
    environment_quaternions = np.asarray(
        environment_quaternions_world_wxyz, dtype=np.float64
    )
    environment_extents = np.asarray(
        environment_half_extents_xyz_m, dtype=np.float64
    )
    count = len(primitive_centers)
    expected_three = (count, 3)
    expected_four = (count, 4)
    for value, expected, name in (
        (primitive_centers, expected_three, "primitive centers"),
        (primitive_quaternions, expected_four, "primitive quaternions"),
        (primitive_extents, expected_three, "primitive full extents"),
        (environment_centers, expected_three, "environment centers"),
        (environment_quaternions, expected_four, "environment quaternions"),
        (environment_extents, expected_three, "environment half extents"),
    ):
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"{name} must have finite shape {expected}")
    if np.any(primitive_extents <= 0.0) or np.any(environment_extents <= 0.0):
        raise ValueError("box extents must be positive")
    normalize_quaternions_wxyz(primitive_quaternions)
    normalize_quaternions_wxyz(environment_quaternions)
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    if count == 0:
        return BatchedBoxOBBClosestWitness(
            signed_clearance_m=np.empty(0, dtype=np.float64),
            primitive_point_world_xyz_m=np.empty((0, 3), dtype=np.float64),
            environment_point_world_xyz_m=np.empty((0, 3), dtype=np.float64),
            intersects=np.empty(0, dtype=bool),
            feature_pair=(),
        )

    chunks = []
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        chunks.append(
            _batch_box_obb_closest_witness_chunk(
                primitive_centers[start:stop],
                primitive_quaternions[start:stop],
                primitive_extents[start:stop],
                environment_centers[start:stop],
                environment_quaternions[start:stop],
                environment_extents[start:stop],
            )
        )
    return BatchedBoxOBBClosestWitness(
        signed_clearance_m=np.concatenate(
            [value.signed_clearance_m for value in chunks]
        ),
        primitive_point_world_xyz_m=np.concatenate(
            [value.primitive_point_world_xyz_m for value in chunks], axis=0
        ),
        environment_point_world_xyz_m=np.concatenate(
            [value.environment_point_world_xyz_m for value in chunks], axis=0
        ),
        intersects=np.concatenate([value.intersects for value in chunks]),
        feature_pair=tuple(
            feature for value in chunks for feature in value.feature_pair
        ),
    )


def primitive_obb_closest_witness(
    primitive: RobotPrimitive,
    environment_box: OrientedBox,
) -> PrimitiveOBBClosestWitness:
    """Return the exact deterministic closest environment-OBB witness.

    Robot boxes consume Genesis full extents.  Capsules are finite along their
    local Z axis.  The result is label-independent and uses only frozen geometry.
    """

    primitive_center = _vector(primitive.position_xyz_m, 3, "primitive.position_xyz_m")
    primitive_rotation = rotation_matrix_wxyz(primitive.quaternion_wxyz)
    environment_center = _vector(environment_box.center_xyz_m, 3, "environment_box.center_xyz_m")
    environment_rotation = rotation_matrix_wxyz(environment_box.quaternion_wxyz)
    environment_half = _vector(environment_box.half_extents_xyz_m, 3, "environment_box.half_extents_xyz_m")
    data = np.asarray(primitive.data, dtype=np.float64)

    if primitive.kind == "sphere":
        local_center = (primitive_center - environment_center) @ environment_rotation
        local_solid, center_distance, inside = _point_aabb_solid_closest_local(
            local_center, environment_half
        )
        if inside:
            local_environment, _ = _point_aabb_surface_closest_local(
                local_center, environment_half
            )
        else:
            local_environment = local_solid
        environment = local_environment @ environment_rotation.T + environment_center
        if center_distance > GEOMETRY_EPS:
            direction = (environment - primitive_center) / float(
                np.linalg.norm(environment - primitive_center)
            )
            primitive_point = primitive_center + float(data[0]) * direction
        else:
            primitive_point = primitive_center.copy()
        signed = center_distance - float(data[0])
        feature = "SPHERE_ENVIRONMENT_FACE"
    elif primitive.kind == "capsule":
        axis = primitive_rotation[:, 2]
        half_length = 0.5 * float(data[1])
        start_world = primitive_center - half_length * axis
        end_world = primitive_center + half_length * axis
        start_local = (start_world - environment_center) @ environment_rotation
        end_local = (end_world - environment_center) @ environment_rotation
        segment_local, environment_local, axis_distance, _parameter = _segment_aabb_closest_local(
            start_local, end_local, environment_half
        )
        if axis_distance <= DISTANCE_TIE_ATOL_M:
            segment_local, environment_local, _surface_distance = _segment_aabb_surface_witness_local(
                start_local, end_local, environment_half
            )
        segment_world = segment_local @ environment_rotation.T + environment_center
        environment = environment_local @ environment_rotation.T + environment_center
        delta = environment - segment_world
        norm = float(np.linalg.norm(delta))
        primitive_point = (
            segment_world
            if norm <= GEOMETRY_EPS
            else segment_world + float(data[0]) * delta / norm
        )
        signed = axis_distance - float(data[0])
        feature = "CAPSULE_SEGMENT_ENVIRONMENT_FACE"
    else:
        primitive_half = data[:3] * 0.5
        primitive_point, environment, surface_distance, feature = _obb_obb_surface_closest(
            primitive_center,
            primitive_rotation,
            primitive_half,
            environment_center,
            environment_rotation,
            environment_half,
        )
        intersects, minimum_overlap = _obb_sat_intersection_and_overlap(
            primitive_center,
            primitive_rotation,
            primitive_half,
            environment_center,
            environment_rotation,
            environment_half,
        )
        signed = -minimum_overlap if intersects else surface_distance

    if abs(float(signed)) <= DISTANCE_TIE_ATOL_M:
        signed = 0.0
    intersects = bool(signed <= 0.0)
    return PrimitiveOBBClosestWitness(
        primitive_identity=primitive.identity,
        environment_identity=environment_box.identity,
        primitive_point_world_xyz_m=tuple(float(value) for value in primitive_point),
        environment_point_world_xyz_m=tuple(float(value) for value in environment),
        signed_clearance_m=float(signed),
        separation_distance_m=max(0.0, float(signed)),
        intersects=intersects,
        feature_pair=feature,
    )


def primitive_spec_at_geom_pose_obb_closest_witness(
    spec: RobotPrimitiveSpec,
    geom_position_world_xyz_m: Sequence[float] | np.ndarray,
    geom_quaternion_world_wxyz: Sequence[float] | np.ndarray,
    environment_box: OrientedBox,
) -> PrimitiveOBBClosestWitness:
    """Evaluate a spec whose local transform is already applied in a geom pose."""

    primitive = RobotPrimitive(
        identity=spec.identity,
        kind=spec.kind,
        data=spec.data,
        position_xyz_m=tuple(float(value) for value in _vector(geom_position_world_xyz_m, 3, "geom_position_world_xyz_m")),
        quaternion_wxyz=tuple(
            float(value) for value in normalize_quaternion_wxyz(geom_quaternion_world_wxyz)
        ),
        geom_index=spec.geom_index,
        link_index=spec.link_index,
        link_name=spec.link_name,
    )
    return primitive_obb_closest_witness(primitive, environment_box)


def primitive_spec_at_link_pose_obb_closest_witness(
    spec: RobotPrimitiveSpec,
    link_position_world_xyz_m: Sequence[float] | np.ndarray,
    link_quaternion_world_wxyz: Sequence[float] | np.ndarray,
    environment_box: OrientedBox,
) -> PrimitiveOBBClosestWitness:
    """Evaluate a spec by first composing its frozen link-local transform."""

    position, quaternion = compose_transform(
        link_position_world_xyz_m,
        link_quaternion_world_wxyz,
        spec.local_position_xyz_m,
        spec.local_quaternion_wxyz,
    )
    return primitive_spec_at_geom_pose_obb_closest_witness(
        spec, position, quaternion, environment_box
    )


@dataclass(frozen=True)
class PerLinkClearance:
    link_indices: np.ndarray
    link_names: tuple[str, ...]
    minimum_clearance_m: np.ndarray
    responsible_point_index: np.ndarray
    responsible_geom_index: np.ndarray
    responsible_primitive_identity: tuple[str | None, ...]

    @property
    def global_minimum_clearance_m(self) -> float:
        return float(np.min(self.minimum_clearance_m)) if len(self.minimum_clearance_m) else math.inf

    def to_serializable(self) -> dict[str, Any]:
        rows = []
        for index, link_name in enumerate(self.link_names):
            value = float(self.minimum_clearance_m[index])
            rows.append(
                {
                    "link_index": int(self.link_indices[index]),
                    "link_name": link_name,
                    "minimum_clearance_m": None if not math.isfinite(value) else value,
                    "responsible_geom_index": int(self.responsible_geom_index[index]),
                    "responsible_point_index": int(self.responsible_point_index[index]),
                    "responsible_primitive_identity": self.responsible_primitive_identity[index],
                }
            )
        return {"schema": "body_centric_range_per_link_clearance_v1", "rows": rows}


def point_cloud_per_link_clearance(
    points_world_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    robot_primitives: Sequence[RobotPrimitive],
) -> PerLinkClearance:
    """Reduce an environment-return cloud into deterministic per-link minima."""

    points = np.asarray(points_world_xyz_m, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world_xyz_m must have shape [N,3] and be finite")
    primitives = sorted(robot_primitives, key=lambda value: (value.link_index, value.geom_index, value.identity))
    link_keys = sorted({(int(value.link_index), str(value.link_name)) for value in primitives})
    link_indices = np.asarray([item[0] for item in link_keys], dtype=np.int16)
    link_names = tuple(item[1] for item in link_keys)
    minima = np.full(len(link_keys), np.inf, dtype=np.float64)
    point_indices = np.full(len(link_keys), -1, dtype=np.int32)
    geom_indices = np.full(len(link_keys), -1, dtype=np.int32)
    identities: list[str | None] = [None] * len(link_keys)
    lookup = {key: index for index, key in enumerate(link_keys)}
    if len(points):
        for primitive in primitives:
            values = point_to_primitive_clearance(points, primitive)
            point_index = int(np.argmin(values))
            value = float(values[point_index])
            link_index = lookup[(int(primitive.link_index), str(primitive.link_name))]
            if value < minima[link_index] - DISTANCE_TIE_ATOL_M:
                minima[link_index] = value
                point_indices[link_index] = point_index
                geom_indices[link_index] = int(primitive.geom_index)
                identities[link_index] = primitive.identity
    return PerLinkClearance(
        link_indices=link_indices,
        link_names=link_names,
        minimum_clearance_m=minima,
        responsible_point_index=point_indices,
        responsible_geom_index=geom_indices,
        responsible_primitive_identity=tuple(identities),
    )


def closest_points_to_scene(
    points_world_xyz_m: Sequence[Sequence[float]] | np.ndarray,
    robot_geoms: Sequence[RobotPrimitive],
    *,
    valid_environment_return: Sequence[bool] | np.ndarray | None = None,
) -> PerLinkClearance:
    """Stable wrapper for per-link reduction of environment range returns.

    Self returns must not be passed as valid environment points.  Supplying a
    boolean ``valid_environment_return`` mask makes that exclusion explicit;
    omitting it treats every supplied point as an environment return.
    """

    points = np.asarray(points_world_xyz_m, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world_xyz_m must have shape [N,3] and be finite")
    if valid_environment_return is not None:
        mask = np.asarray(valid_environment_return, dtype=bool)
        if mask.shape != (len(points),):
            raise ValueError("valid_environment_return must have shape [N]")
        points = points[mask]
    return point_cloud_per_link_clearance(points, robot_geoms)


@dataclass(frozen=True)
class TrajectoryPerLinkClearance:
    """Exact per-step/link minima accelerated by deterministic cKDTree queries."""

    link_indices: np.ndarray
    link_names: tuple[str, ...]
    minimum_clearance_m: np.ndarray
    responsible_point_index: np.ndarray
    responsible_geom_index: np.ndarray
    candidate_k: int

    def to_serializable(self) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        for step_index in range(self.minimum_clearance_m.shape[0]):
            links = []
            for link_position, link_name in enumerate(self.link_names):
                clearance = float(self.minimum_clearance_m[step_index, link_position])
                links.append(
                    {
                        "link_index": int(self.link_indices[link_position]),
                        "link_name": link_name,
                        "minimum_clearance_m": None if not math.isfinite(clearance) else clearance,
                        "responsible_geom_index": int(
                            self.responsible_geom_index[step_index, link_position]
                        ),
                        "responsible_point_index": int(
                            self.responsible_point_index[step_index, link_position]
                        ),
                    }
                )
            steps.append({"links": links, "step_index": step_index})
        return {
            "candidate_k": self.candidate_k,
            "schema": "body_centric_range_trajectory_per_link_clearance_v1",
            "steps": steps,
        }


def _primitive_bounding_radius(primitive: RobotPrimitive) -> float:
    data = np.asarray(primitive.data, dtype=np.float64)
    if primitive.kind == "sphere":
        return float(data[0])
    if primitive.kind == "capsule":
        return float(data[0] + 0.5 * data[1])
    return float(np.linalg.norm(data[:3] * 0.5))


def _trajectory_clouds(
    point_clouds_world_xyz_m: Sequence[np.ndarray] | np.ndarray,
    steps: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(point_clouds_world_xyz_m, np.ndarray):
        values = np.asarray(point_clouds_world_xyz_m, dtype=np.float64)
        if values.ndim == 2:
            candidates: Sequence[np.ndarray] = (values,) * steps
        elif values.ndim == 3 and values.shape[0] == steps:
            candidates = tuple(values[index] for index in range(steps))
        else:
            raise ValueError("point clouds must have shape [steps,points,3] or shared [points,3]")
    else:
        if len(point_clouds_world_xyz_m) != steps:
            raise ValueError("point cloud sequence must have one cloud per step")
        candidates = point_clouds_world_xyz_m
    output: list[np.ndarray] = []
    for value in candidates:
        cloud = np.asarray(value, dtype=np.float64)
        if cloud.size == 0:
            cloud = np.empty((0, 3), dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError("every point cloud must have finite shape [points,3]")
        output.append(cloud)
    return tuple(output)


def trajectory_point_cloud_per_link_clearance(
    point_clouds_world_xyz_m: Sequence[np.ndarray] | np.ndarray,
    robot_specs: Sequence[RobotPrimitiveSpec],
    robot_positions_world_xyz_m: np.ndarray,
    robot_quaternions_world_wxyz: np.ndarray,
    *,
    candidate_k: int = 64,
) -> TrajectoryPerLinkClearance:
    """Reduce trajectory clouds to exact ``[steps,links]`` clearances.

    Each primitive first evaluates the 64 center-nearest cKDTree candidates.
    It then queries every point inside ``best_clearance + bounding_radius``;
    the circumscribed-radius lower bound proves that excluded points cannot
    improve the result.  Exact signed primitive distance, not center distance,
    determines every returned minimum.  This refinement is essential for long
    capsules and boxes.
    """

    specs = tuple(robot_specs)
    positions = np.asarray(robot_positions_world_xyz_m, dtype=np.float64)
    quaternions = np.asarray(robot_quaternions_world_wxyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(specs), 3) or not np.isfinite(positions).all():
        raise ValueError("robot positions must have shape [steps,geoms,3]")
    if quaternions.shape != positions.shape[:-1] + (4,) or not np.isfinite(quaternions).all():
        raise ValueError("robot quaternions must have shape [steps,geoms,4]")
    normalize_quaternions_wxyz(quaternions)
    k_value = int(candidate_k)
    if k_value != 64:
        raise ValueError("BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1 freezes candidate_k=64")
    steps = positions.shape[0]
    clouds = _trajectory_clouds(point_clouds_world_xyz_m, steps)
    ordered_spec_indices = sorted(
        range(len(specs)),
        key=lambda index: (
            specs[index].link_index,
            specs[index].geom_index,
            specs[index].identity,
        ),
    )
    link_keys = sorted({(int(spec.link_index), str(spec.link_name)) for spec in specs})
    link_indices = np.asarray([key[0] for key in link_keys], dtype=np.int16)
    link_names = tuple(key[1] for key in link_keys)
    link_lookup = {key: index for index, key in enumerate(link_keys)}
    minimum = np.full((steps, len(link_keys)), np.inf, dtype=np.float64)
    point_index = np.full((steps, len(link_keys)), -1, dtype=np.int32)
    geom_index = np.full((steps, len(link_keys)), -1, dtype=np.int32)
    tree_cache: dict[int, cKDTree] = {}
    for step_index, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        cache_key = id(cloud)
        tree = tree_cache.get(cache_key)
        if tree is None:
            tree = cKDTree(cloud, compact_nodes=True, balanced_tree=True, copy_data=True)
            tree_cache[cache_key] = tree
        for spec_index in ordered_spec_indices:
            spec = specs[spec_index]
            primitive = RobotPrimitive(
                identity=spec.identity,
                kind=spec.kind,
                data=spec.data,
                position_xyz_m=tuple(float(value) for value in positions[step_index, spec_index]),
                quaternion_wxyz=tuple(float(value) for value in quaternions[step_index, spec_index]),
                geom_index=spec.geom_index,
                link_index=spec.link_index,
                link_name=spec.link_name,
            )
            query_k = min(k_value, len(cloud))
            _, initial_indices = tree.query(
                np.asarray(primitive.position_xyz_m, dtype=np.float64),
                k=query_k,
            )
            initial = np.unique(np.atleast_1d(initial_indices).astype(np.int64))
            initial.sort()
            initial_clearance = point_to_primitive_clearance(cloud[initial], primitive)
            initial_best = float(np.min(initial_clearance))
            search_radius = max(0.0, initial_best + _primitive_bounding_radius(primitive))
            refinement = np.asarray(
                tree.query_ball_point(
                    np.asarray(primitive.position_xyz_m, dtype=np.float64),
                    search_radius + DISTANCE_TIE_ATOL_M,
                ),
                dtype=np.int64,
            )
            candidates = np.unique(np.concatenate((initial, refinement)))
            candidates.sort()
            clearance = point_to_primitive_clearance(cloud[candidates], primitive)
            candidate_position = int(np.argmin(clearance))
            value = float(clearance[candidate_position])
            responsible_point = int(candidates[candidate_position])
            link_position = link_lookup[(int(spec.link_index), str(spec.link_name))]
            if value < minimum[step_index, link_position] - DISTANCE_TIE_ATOL_M:
                minimum[step_index, link_position] = value
                point_index[step_index, link_position] = responsible_point
                geom_index[step_index, link_position] = int(spec.geom_index)
    return TrajectoryPerLinkClearance(
        link_indices=link_indices,
        link_names=link_names,
        minimum_clearance_m=minimum,
        responsible_point_index=point_index,
        responsible_geom_index=geom_index,
        candidate_k=k_value,
    )


def _angle_in_interval_deg(angle_deg: float, limits_deg: tuple[float, float]) -> bool:
    lower, upper = map(float, limits_deg)
    if not all(map(math.isfinite, (lower, upper))):
        raise ValueError("FOV limits must be finite")
    span = upper - lower
    if abs(span) >= 360.0 - 1.0e-10:
        return True
    angle = (float(angle_deg) + 180.0) % 360.0 - 180.0
    lower = (lower + 180.0) % 360.0 - 180.0
    upper = (upper + 180.0) % 360.0 - 180.0
    if lower <= upper:
        return lower - 1.0e-10 <= angle <= upper + 1.0e-10
    return angle >= lower - 1.0e-10 or angle <= upper + 1.0e-10


@dataclass(frozen=True)
class ContinuumTargetVisibility:
    distance_m: float
    azimuth_deg: float
    elevation_deg: float
    horizontal_fov_inclusion: bool
    vertical_fov_inclusion: bool
    range_inclusion: bool
    nominal_fov_inclusion: bool
    direct_visibility_after_occlusion: bool
    target_object_observable: bool | None
    self_occluded: bool
    environment_occluded: bool
    first_hit_class: str
    first_hit_identity: str | None
    first_hit_distance_m: float | None

    def to_serializable(self) -> dict[str, Any]:
        return {
            "azimuth_deg": self.azimuth_deg,
            "direct_visibility_after_occlusion": self.direct_visibility_after_occlusion,
            "distance_m": self.distance_m,
            "elevation_deg": self.elevation_deg,
            "environment_occluded": self.environment_occluded,
            "first_hit_class": self.first_hit_class,
            "first_hit_distance_m": self.first_hit_distance_m,
            "first_hit_identity": self.first_hit_identity,
            "horizontal_fov_inclusion": self.horizontal_fov_inclusion,
            "nominal_fov_inclusion": self.nominal_fov_inclusion,
            "range_inclusion": self.range_inclusion,
            "self_occluded": self.self_occluded,
            "target_object_observable": self.target_object_observable,
            "vertical_fov_inclusion": self.vertical_fov_inclusion,
        }


def continuum_target_visibility(
    sensor_origin_world_xyz_m: Sequence[float] | np.ndarray,
    sensor_quaternion_world_wxyz: Sequence[float] | np.ndarray,
    target_world_xyz_m: Sequence[float] | np.ndarray,
    *,
    horizontal_fov_deg: tuple[float, float],
    vertical_fov_deg: tuple[float, float],
    near_m: float,
    far_m: float,
    environment_boxes: Sequence[OrientedBox] = (),
    robot_primitives: Sequence[RobotPrimitive] = (),
    ground_z_m: float | None = None,
    target_object_identity: str | None = None,
    target_tolerance_m: float = 1.0e-9,
) -> ContinuumTargetVisibility:
    """Evaluate the continuous-ray visibility of one world-space target.

    This is the dense angular continuum primitive.  It removes finite scan
    sparsity but preserves FOV, range, mounting origin, robot self-occlusion
    and environment first-surface occlusion.
    """

    origin = _vector(sensor_origin_world_xyz_m, 3, "sensor_origin_world_xyz_m")
    target = _vector(target_world_xyz_m, 3, "target_world_xyz_m")
    delta = target - origin
    distance = float(np.linalg.norm(delta))
    if distance <= GEOMETRY_EPS:
        raise ValueError("target must be distinct from sensor origin")
    direction = delta / distance
    rotation = rotation_matrix_wxyz(sensor_quaternion_world_wxyz)
    local = rotation.T @ direction
    azimuth = math.degrees(math.atan2(float(local[1]), float(local[0])))
    elevation = math.degrees(math.atan2(float(local[2]), math.hypot(float(local[0]), float(local[1]))))
    horizontal = _angle_in_interval_deg(azimuth, horizontal_fov_deg)
    vertical = float(vertical_fov_deg[0]) - 1.0e-10 <= elevation <= float(vertical_fov_deg[1]) + 1.0e-10
    range_inclusion = float(near_m) - DISTANCE_TIE_ATOL_M <= distance <= float(far_m) + DISTANCE_TIE_ATOL_M
    nominal = horizontal and vertical and range_inclusion
    hit = first_hits(
        origin,
        direction[None, :],
        environment_boxes=environment_boxes,
        robot_primitives=robot_primitives,
        ground_z_m=ground_z_m,
        near_m=0.0,
        far_m=max(float(far_m), distance + float(target_tolerance_m)),
    )
    raw = float(hit.raw_distance_m[0])
    owner = int(hit.hit_class[0])
    identity = None if not str(hit.hit_identity[0]) else str(hit.hit_identity[0])
    hit_before_target = math.isfinite(raw) and raw < distance - float(target_tolerance_m)
    self_occluded = owner == ROBOT_HIT and raw <= distance + float(target_tolerance_m)
    environment_occluded = owner in (ENVIRONMENT_HIT, GROUND_HIT) and hit_before_target
    direct = nominal and not self_occluded and not environment_occluded
    observable = None
    if target_object_identity is not None:
        observable = bool(
            nominal
            and owner == ENVIRONMENT_HIT
            and identity == target_object_identity
            and raw <= distance + float(target_tolerance_m)
        )
    return ContinuumTargetVisibility(
        distance_m=distance,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        horizontal_fov_inclusion=horizontal,
        vertical_fov_inclusion=vertical,
        range_inclusion=range_inclusion,
        nominal_fov_inclusion=nominal,
        direct_visibility_after_occlusion=direct,
        target_object_observable=observable,
        self_occluded=self_occluded,
        environment_occluded=environment_occluded,
        first_hit_class=HIT_CLASS_NAMES[owner],
        first_hit_identity=identity,
        first_hit_distance_m=None if not math.isfinite(raw) else raw,
    )


def fixture_receipt() -> Mapping[str, object]:
    """Small deterministic identity witness used by the focused test suite."""

    obstacle = OrientedBox("fixture_wall", (2.0, 0.0, 0.0), (0.05, 0.2, 0.2), object_index=7)
    self_shape = RobotPrimitive(
        identity="fixture_base",
        kind="sphere",
        data=(0.10,),
        position_xyz_m=(1.0, 0.0, 0.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        geom_index=3,
        link_index=0,
        link_name="base",
    )
    hit = first_hits(
        (0.0, 0.0, 0.0),
        spherical_directions_fru(np.asarray([0.0]), np.asarray([0.0])),
        environment_boxes=(obstacle,),
        robot_primitives=(self_shape,),
        near_m=0.05,
        far_m=10.0,
    )
    payload: dict[str, object] = {
        "coordinate_convention": "FRU_WXYZ_FLOAT64",
        "first_hit": hit.to_serializable(),
        "schema": "body_centric_range_coverage_geometry_fixture_v1",
    }
    payload["content_digest"] = canonical_digest(payload)
    return payload


def _fixture_metric_action(
    action_index: int,
    *,
    predicted_contact: bool = False,
    oracle_contact: bool = False,
    progress_m: float = 0.0,
    heading_improvement_rad: float = 0.0,
    next_actions: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "action_index": action_index,
        "controller": "route" if action_index < 12 else "lateral",
        "decision_progress_m": progress_m,
        "h3_heading_improvement_rad": heading_improvement_rad,
        "h3_progress_m": progress_m,
        "next_actions": next_actions,
        "oracle_contact": oracle_contact,
        "predicted_contact": predicted_contact,
    }


def _fixture_next_action(
    action_index: int,
    *,
    predicted_contact: bool,
    oracle_contact: bool,
) -> dict[str, object]:
    return {
        "action_index": action_index,
        "controller": "route",
        "oracle_contact": oracle_contact,
        "predicted_contact": predicted_contact,
    }


def _evaluate_named_fixtures(metrics_module: Any) -> dict[str, object]:
    required_metrics = (
        "contact_predictions",
        "h3_route_order",
        "reduce_two_ply_state",
        "tie_aware_auc",
        "tie_aware_average_precision",
        "tie_aware_spearman",
    )
    missing = [name for name in required_metrics if not callable(getattr(metrics_module, name, None))]
    if missing:
        raise TypeError(f"metrics_module lacks required functions: {missing}")

    identity = (1.0, 0.0, 0.0, 0.0)
    trunk = RobotPrimitive("trunk", "box", (1.0, 0.5, 0.4), (0.0, 0.0, 0.0), identity, 0, 0, "trunk")
    front_limb = RobotPrimitive("front_limb", "sphere", (0.1,), (0.7, 0.3, -0.2), identity, 1, 1, "front_limb")
    rear_limb = RobotPrimitive("rear_limb", "sphere", (0.1,), (-0.7, 0.3, -0.2), identity, 2, 2, "rear_limb")
    calf = RobotPrimitive("calf", "capsule", (0.08, 0.4), (0.7, 0.3, -0.6), identity, 3, 3, "calf")
    robot = (trunk, front_limb, rear_limb, calf)

    fixture_rows: dict[str, dict[str, object]] = {}

    clear_reduction = point_cloud_per_link_clearance(np.asarray([[4.0, 4.0, 4.0]]), robot)
    fixture_rows["clear full-body sweep"] = {
        "pass": bool(np.all(clear_reduction.minimum_clearance_m > 1.0)),
        "minimum_clearance_m": [float(value) for value in clear_reduction.minimum_clearance_m],
    }
    contact_cases = {
        "front trunk contact": (trunk, np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64)),
        "side trunk contact": (trunk, np.asarray([[0.0, 0.25, 0.0]], dtype=np.float64)),
        "front-limb contact": (front_limb, np.asarray([[0.8, 0.3, -0.2]], dtype=np.float64)),
        "rear-limb contact": (rear_limb, np.asarray([[-0.8, 0.3, -0.2]], dtype=np.float64)),
        "calf contact": (calf, np.asarray([[0.78, 0.3, -0.6]], dtype=np.float64)),
    }
    for fixture_name, (primitive, point) in contact_cases.items():
        clearance = float(point_to_primitive_clearance(point, primitive)[0])
        fixture_rows[fixture_name] = {
            "clearance_m": clearance,
            "pass": clearance <= DISTANCE_TIE_ATOL_M,
            "responsible_link": primitive.link_name,
        }

    near_box = OrientedBox("near_box", (0.03, 0.0, 0.0), (0.01, 0.05, 0.05), object_index=0)
    far_box = OrientedBox("far_box", (1.0, 0.0, 0.0), (0.1, 0.5, 0.5), object_index=1)
    near_result = raycast_scene(
        (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (far_box, near_box), (), False
    )
    fixture_rows["contact inside the near blind region"] = {
        "first_object": str(near_result.object_identity[0]),
        "pass": bool(
            near_result.near_blind[0]
            and not near_result.valid_return[0]
            and near_result.object_identity[0] == "near_box"
        ),
        "raw_distance_m": float(near_result.raw_distance_m[0]),
    }

    blocker = RobotPrimitive("self_blocker", "sphere", (0.2,), (1.0, 0.0, 0.0), identity, 8, 4, "head")
    self_result = raycast_scene(
        (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (far_box,), (blocker,), False
    )
    fixture_rows["contact hidden by robot self-occlusion"] = {
        "geom_identity": str(self_result.geom_identity[0]),
        "link_name": str(self_result.link_name[0]),
        "pass": bool(self_result.self_return[0] and self_result.geom_identity[0] == "self_blocker"),
    }

    narrow = OrientedBox("between", (2.0, 0.0, 0.0), (0.02, 0.02, 0.05), object_index=2)
    sparse_directions = spherical_directions_fru(np.radians(np.asarray([-2.0, 2.0])), np.asarray([0.0]))
    sparse_result = raycast_scene((0.0, 0.0, 0.0), sparse_directions, 0.05, 10.0, (narrow,), (), False)
    continuum_result = continuum_target_visibility(
        (0.0, 0.0, 0.0),
        identity,
        (1.98, 0.0, 0.0),
        horizontal_fov_deg=(-180.0, 180.0),
        vertical_fov_deg=(-45.0, 45.0),
        near_m=0.05,
        far_m=10.0,
        environment_boxes=(narrow,),
        target_object_identity="between",
    )
    fixture_rows["contact between scan samples"] = {
        "continuum_observable": bool(continuum_result.target_object_observable),
        "pass": bool(np.all(sparse_result.no_hit) and continuum_result.target_object_observable),
        "sparse_returns": int(np.sum(sparse_result.valid_return)),
    }

    visible = raycast_scene((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (far_box,), (), False)
    fixture_rows["current-state visible geometry"] = {
        "object_identity": str(visible.object_identity[0]),
        "pass": bool(visible.valid_return[0] and visible.object_identity[0] == "far_box"),
    }
    outside = OrientedBox("future_wall", (12.0, 0.0, 0.0), (0.1, 0.5, 0.5), object_index=3)
    future = OrientedBox("future_wall", (2.0, 0.0, 0.0), (0.1, 0.5, 0.5), object_index=3)
    current_future_target = raycast_scene((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (outside,), (), False)
    actual_future_target = raycast_scene((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (future,), (), False)
    fixture_rows["future-only visible geometry"] = {
        "future_object_identity": str(actual_future_target.object_identity[0]),
        "pass": bool(current_future_target.free_to_far[0] and actual_future_target.valid_return[0]),
    }

    viable = {
        "family": "fixture",
        "state_id": "one_and_zero",
        "current_actions": [
            _fixture_metric_action(
                0,
                progress_m=0.3,
                next_actions=(
                    _fixture_next_action(0, predicted_contact=False, oracle_contact=False),
                    _fixture_next_action(1, predicted_contact=True, oracle_contact=True),
                ),
            ),
            _fixture_metric_action(
                1,
                progress_m=0.2,
                next_actions=(
                    _fixture_next_action(0, predicted_contact=True, oracle_contact=True),
                    _fixture_next_action(1, predicted_contact=True, oracle_contact=True),
                ),
            ),
        ],
    }
    viability = metrics_module.reduce_two_ply_state(viable)
    fixture_rows["one safe successor action"] = {
        "pass": viability["predicted_safe_counts"][0] == 1 and viability["selected"] == 0,
        "predicted_safe_count": int(viability["predicted_safe_counts"][0]),
    }
    fixture_rows["zero safe successor actions"] = {
        "admitted": bool(viability["admitted"][1]),
        "pass": viability["predicted_safe_counts"][1] == 0 and not viability["admitted"][1],
        "predicted_safe_count": int(viability["predicted_safe_counts"][1]),
    }
    tied_predictions = metrics_module.contact_predictions([0.2, 0.200001], 0.2)
    fixture_rows["exact threshold tie"] = {
        "pass": bool(tied_predictions.tolist() == [True, False]),
        "predicted_contact": [bool(value) for value in tied_predictions],
    }
    nonviable = {
        "family": "fixture",
        "state_id": "nonviable",
        "current_actions": [
            _fixture_metric_action(
                0,
                predicted_contact=True,
                oracle_contact=True,
                next_actions=(_fixture_next_action(0, predicted_contact=True, oracle_contact=True),),
            )
        ],
    }
    abstention = metrics_module.reduce_two_ply_state(nonviable)
    fixture_rows["correct abstention"] = {
        "pass": bool(abstention["correct_abstention"] and abstention["selected"] is None),
        "selected": abstention["selected"],
    }
    h3_rows = [
        _fixture_metric_action(4, progress_m=1.0, heading_improvement_rad=0.0),
        _fixture_metric_action(2, progress_m=0.97, heading_improvement_rad=0.5),
        _fixture_metric_action(1, progress_m=0.97, heading_improvement_rad=0.5),
        _fixture_metric_action(3, progress_m=0.969, heading_improvement_rad=100.0),
    ]
    h3_order = [int(h3_rows[index]["action_index"]) for index in metrics_module.h3_route_order(h3_rows)]
    fixture_rows["deterministic H3 route selection"] = {
        "action_order": h3_order,
        "pass": h3_order == [1, 2, 4, 3],
    }

    labels = np.asarray([True, False, True, False])
    scores = np.asarray([1.0, 1.0, 0.0, 0.0])
    rank_metrics = {
        "ap": float(metrics_module.tie_aware_average_precision(labels, scores)),
        "auc": float(metrics_module.tie_aware_auc(labels, scores)),
        "spearman": float(metrics_module.tie_aware_spearman([0, 0, 1, 1], [1, 1, 0, 0])),
    }
    rank_metrics["pass"] = bool(
        rank_metrics["auc"] == 0.5
        and rank_metrics["ap"] == 0.5
        and math.isclose(rank_metrics["spearman"], -1.0)
    )

    sparse_pattern_a = generate_sparse_scan_pattern()
    sparse_pattern_b = generate_sparse_scan_pattern()
    l2_pattern_a = generate_l2_scan_pattern()
    l2_pattern_b = generate_l2_scan_pattern()
    reduction_a = clear_reduction.to_serializable()
    reduction_b = point_cloud_per_link_clearance(np.asarray([[4.0, 4.0, 4.0]]), robot).to_serializable()
    self_serialized_a = self_result.to_serializable()
    self_serialized_b = raycast_scene(
        (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), 0.05, 10.0, (far_box,), (blocker,), False
    ).to_serializable()
    requirements = {
        "deterministic occlusion": {
            "pass": canonical_json_bytes(self_serialized_a) == canonical_json_bytes(self_serialized_b)
        },
        "deterministic per-link reduction": {
            "pass": canonical_json_bytes(reduction_a) == canonical_json_bytes(reduction_b)
        },
        "deterministic point timestamps": {
            "pass": bool(
                np.array_equal(sparse_pattern_a.timestamps_s, sparse_pattern_b.timestamps_s)
                and np.array_equal(l2_pattern_a.timestamps_s, l2_pattern_b.timestamps_s)
            )
        },
        "deterministic ray generation": {
            "l2_ray_sha256": l2_pattern_a.to_serializable()["ray_sha256"],
            "pass": bool(
                np.array_equal(sparse_pattern_a.directions_sensor_fru, sparse_pattern_b.directions_sensor_fru)
                and np.array_equal(l2_pattern_a.directions_sensor_fru, l2_pattern_b.directions_sensor_fru)
            ),
            "sparse_ray_sha256": sparse_pattern_a.to_serializable()["ray_sha256"],
        },
    }
    return {
        "fixtures": fixture_rows,
        "metric_tie_handling": rank_metrics,
        "requirements": requirements,
        "scan_patterns": {
            "l2": l2_pattern_a.to_serializable(),
            "sparse": sparse_pattern_a.to_serializable(),
        },
        "schema": "body_centric_range_coverage_fixture_gate_v1",
    }


def run_fixtures(*, metrics_module: Any) -> dict[str, object]:
    """Execute every prospectively named geometry and metrics fixture."""

    first = _evaluate_named_fixtures(metrics_module)
    second = _evaluate_named_fixtures(metrics_module)
    byte_identical = canonical_json_bytes(first) == canonical_json_bytes(second)
    first["requirements"]["byte-identical receipt regeneration"] = {"pass": byte_identical}
    fixture_pass = all(bool(row["pass"]) for row in first["fixtures"].values())
    requirements_pass = all(bool(row["pass"]) for row in first["requirements"].values())
    first["pass"] = bool(fixture_pass and requirements_pass and first["metric_tie_handling"]["pass"])
    first["content_digest"] = canonical_digest(first)
    return first


__all__ = [
    "BatchedBoxOBBClosestWitness",
    "DISTANCE_TIE_ATOL_M",
    "ENVIRONMENT_HIT",
    "FirstHitResult",
    "GROUND_HIT",
    "HIT_CLASS_NAMES",
    "NO_HIT",
    "OrientedBox",
    "PerLinkClearance",
    "PrimitiveOBBClosestWitness",
    "ROBOT_HIT",
    "RobotPrimitive",
    "RobotPrimitiveSpec",
    "ScanPattern",
    "SceneRaycastResult",
    "TrajectoryPerLinkClearance",
    "canonical_digest",
    "canonical_json_bytes",
    "closest_points_to_scene",
    "batch_box_obb_closest_witness",
    "compose_transform",
    "continuum_target_visibility",
    "first_hits",
    "fixture_receipt",
    "generate_l2_scan_pattern",
    "generate_sparse_scan_pattern",
    "instantiate_geoms",
    "interpolate_transform",
    "interpolate_transform_series",
    "interpolate_transform_series_vectorized",
    "inverse_transform_points",
    "multiply_quaternion_wxyz",
    "nlerp_quaternion_wxyz",
    "nlerp_quaternions_wxyz",
    "normalize_quaternion_wxyz",
    "normalize_quaternions_wxyz",
    "point_cloud_per_link_clearance",
    "point_to_primitive_clearance",
    "primitive_obb_closest_witness",
    "primitive_spec_at_geom_pose_obb_closest_witness",
    "primitive_spec_at_link_pose_obb_closest_witness",
    "ray_capsule_distances",
    "ray_ground_plane_distances",
    "ray_oriented_box_distances",
    "ray_robot_primitive_distances",
    "raycast_scene",
    "raycast_moving_scene",
    "ray_sphere_distances",
    "rotation_matrix_wxyz",
    "rotation_matrices_wxyz",
    "spherical_directions_fru",
    "run_fixtures",
    "transform_directions",
    "transform_points",
    "trajectory_point_cloud_per_link_clearance",
]
