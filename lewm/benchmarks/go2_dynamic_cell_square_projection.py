"""Pure dynamic camera geometry for Go2 cell-square query support.

The output grid is aligned with the base's stored yaw, while the camera keeps
the roll and pitch carried by the full base quaternion.  This module performs
no file I/O and has no NumPy or torch dependency.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math


GRID_SHAPE = (64, 64)
CELL_SIZE_M = 0.10
FORWARD_MIN_EDGE_M = -1.0
LEFT_MIN_EDGE_M = -3.2
CELL_SQUARE_OFFSETS_M = (
    (0.0, 0.0),
    (-0.05, -0.05),
    (-0.05, 0.05),
    (0.05, -0.05),
    (0.05, 0.05),
)
VERTICAL_ANCHOR_Z_M = (-0.333, -0.133, 0.067, 0.267, 0.467)

HORIZONTAL_FOV_DEG = 78.323
VERTICAL_FOV_DEG = 62.8370386364
CAMERA_NEAR_M = 0.05
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
CAMERA_RPY_BODY_RAD = (0.0, 0.0, 0.0)

QUATERNION_NORM_TOLERANCE = 1e-5
QUATERNION_YAW_TOLERANCE_RAD = 1e-5

_TAN_HALF_HORIZONTAL_FOV = math.tan(
    math.radians(HORIZONTAL_FOV_DEG) * 0.5
)
_TAN_HALF_VERTICAL_FOV = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)


@dataclass(frozen=True)
class YawAlignedCamera:
    """Fixed-mount camera expressed in base-position/stored-yaw coordinates."""

    origin_xyz: tuple[float, float, float]
    forward_xyz: tuple[float, float, float]
    left_xyz: tuple[float, float, float]
    up_xyz: tuple[float, float, float]


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quaternion_xyzw(value: object) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
    ):
        raise ValueError("base quaternion must contain exactly four values")
    result = tuple(
        _finite_number(component, name=f"base quaternion component {index}")
        for index, component in enumerate(value)
    )
    qx, qy, qz, qw = result
    # Keep the left-associative four-term arithmetic used by source validation.
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError("base quaternion norm differs from one")
    return result


def _wrapped_angle_difference(first: float, second: float) -> float:
    difference = first - second
    return math.atan2(math.sin(difference), math.cos(difference))


def _rotation_world_from_body_xyzw(
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Return the standard raw-quaternion rotation without renormalizing it."""

    qx, qy, qz, qw = quaternion_xyzw
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def compose_yaw_aligned_camera(
    base_quat_world_xyzw: object,
    stored_base_yaw_rad: object,
) -> YawAlignedCamera:
    """Compose the fixed body mount while retaining full base roll and pitch."""

    quaternion = _quaternion_xyzw(base_quat_world_xyzw)
    stored_yaw = _finite_number(stored_base_yaw_rad, name="stored base yaw")
    qx, qy, qz, qw = quaternion
    quaternion_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    if (
        abs(_wrapped_angle_difference(stored_yaw, quaternion_yaw))
        > QUATERNION_YAW_TOLERANCE_RAD
    ):
        raise ValueError("stored base yaw disagrees with the base quaternion")

    rotation_world_from_body = _rotation_world_from_body_xyzw(quaternion)
    cos_yaw = math.cos(stored_yaw)
    sin_yaw = math.sin(stored_yaw)

    def world_vector_in_yaw_frame(
        vector_world: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        world_x, world_y, world_z = vector_world
        return (
            cos_yaw * world_x + sin_yaw * world_y,
            -sin_yaw * world_x + cos_yaw * world_y,
            world_z,
        )

    # The columns are the body forward/left/up axes in world coordinates.
    forward = world_vector_in_yaw_frame(
        (
            rotation_world_from_body[0][0],
            rotation_world_from_body[1][0],
            rotation_world_from_body[2][0],
        )
    )
    left = world_vector_in_yaw_frame(
        (
            rotation_world_from_body[0][1],
            rotation_world_from_body[1][1],
            rotation_world_from_body[2][1],
        )
    )
    up = world_vector_in_yaw_frame(
        (
            rotation_world_from_body[0][2],
            rotation_world_from_body[1][2],
            rotation_world_from_body[2][2],
        )
    )
    mount_forward, mount_left, mount_up = CAMERA_XYZ_BODY_M
    origin = (
        mount_forward * forward[0]
        + mount_left * left[0]
        + mount_up * up[0],
        mount_forward * forward[1]
        + mount_left * left[1]
        + mount_up * up[1],
        mount_forward * forward[2]
        + mount_left * left[2]
        + mount_up * up[2],
    )
    return YawAlignedCamera(
        origin_xyz=origin,
        forward_xyz=forward,
        left_xyz=left,
        up_xyz=up,
    )


def camera_coordinates_in_frustum(
    forward_m: object,
    left_m: object,
    up_m: object,
) -> bool:
    """Return whether camera coordinates lie in the closed rectilinear frustum."""

    forward = _finite_number(forward_m, name="camera forward coordinate")
    left = _finite_number(left_m, name="camera left coordinate")
    up = _finite_number(up_m, name="camera up coordinate")
    return (
        forward >= CAMERA_NEAR_M
        and -forward * _TAN_HALF_HORIZONTAL_FOV
        <= left
        <= forward * _TAN_HALF_HORIZONTAL_FOV
        and -forward * _TAN_HALF_VERTICAL_FOV
        <= up
        <= forward * _TAN_HALF_VERTICAL_FOV
    )


def _camera_coordinates(
    point_xyz: tuple[float, float, float],
    camera: YawAlignedCamera,
) -> tuple[float, float, float]:
    delta = tuple(
        point_xyz[axis] - camera.origin_xyz[axis] for axis in range(3)
    )
    def ordered_dot(axis: tuple[float, float, float]) -> float:
        return delta[0] * axis[0] + delta[1] * axis[1] + delta[2] * axis[2]

    return (
        ordered_dot(camera.forward_xyz),
        ordered_dot(camera.left_xyz),
        ordered_dot(camera.up_xyz),
    )


def _grid_index(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if not 0 <= value < 64:
        raise IndexError(f"{name} is outside [0,64)")
    return value


def cell_center(row: object, column: object) -> tuple[float, float]:
    """Return one registered output-cell center as ``(forward, left)``."""

    row_index = _grid_index(row, name="row")
    column_index = _grid_index(column, name="column")
    return (
        FORWARD_MIN_EDGE_M + (row_index + 0.5) * CELL_SIZE_M,
        LEFT_MIN_EDGE_M + (column_index + 0.5) * CELL_SIZE_M,
    )


def cell_square_query_visible(
    row: object,
    column: object,
    camera: YawAlignedCamera,
) -> bool:
    """Return true when any registered cell-square/height point is visible."""

    if not isinstance(camera, YawAlignedCamera):
        raise TypeError("camera must be a YawAlignedCamera")
    center_forward, center_left = cell_center(row, column)
    for offset_forward, offset_left in CELL_SQUARE_OFFSETS_M:
        point_forward = center_forward + offset_forward
        point_left = center_left + offset_left
        for point_up in VERTICAL_ANCHOR_Z_M:
            coordinates = _camera_coordinates(
                (point_forward, point_left, point_up), camera
            )
            if camera_coordinates_in_frustum(*coordinates):
                return True
    return False


def build_dynamic_cell_square_support_mask(
    base_quat_world_xyzw: object,
    stored_base_yaw_rad: object,
) -> tuple[tuple[bool, ...], ...]:
    """Build the deterministic row-major ``64 x 64`` dynamic support mask."""

    camera = compose_yaw_aligned_camera(
        base_quat_world_xyzw,
        stored_base_yaw_rad,
    )
    return tuple(
        tuple(
            cell_square_query_visible(row, column, camera)
            for column in range(GRID_SHAPE[1])
        )
        for row in range(GRID_SHAPE[0])
    )


def support_mask_sha256(mask: object) -> str:
    """Hash an exact row-major bool support mask as canonical uint8 bytes."""

    if (
        not isinstance(mask, Sequence)
        or isinstance(mask, (str, bytes, bytearray))
        or len(mask) != GRID_SHAPE[0]
    ):
        raise ValueError("support mask must have shape [64,64]")
    payload = bytearray()
    for row in mask:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes, bytearray))
            or len(row) != GRID_SHAPE[1]
        ):
            raise ValueError("support mask must have shape [64,64]")
        for value in row:
            if type(value) is not bool:
                raise TypeError("support mask values must be bool")
            payload.append(1 if value else 0)
    return hashlib.sha256(payload).hexdigest()
