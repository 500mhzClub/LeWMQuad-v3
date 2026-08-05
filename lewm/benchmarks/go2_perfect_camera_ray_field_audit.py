"""Independent perfect-camera-ray reconstruction for physical-v3 labels.

This module deliberately does not call the production physical-v3 label
builder.  It answers a narrower question: if the camera supplied exact
first-surface ray distances, could a mechanical decoder reconstruct the
64x64 UNKNOWN/FREE/OCCUPIED target?

The label contract uses two ray populations, so a conventional pixel depth
image is not, by itself, the perfect field tested here:

* five rays to the ground support of every 0.05 m physical cell determine
  whether the whole source square is visibly free; and
* the registered pinhole pixel lattice supplies exact obstacle first hits.

The zero-inflation physical-free grid and collision boxes are kept as a
separate contract-assisted arm.  The ray-only arm omits the physical-free
prior.  Reporting both prevents a privileged geometry dependency from being
mistaken for information present in the camera rays.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
CLASS_NAMES = ("unknown", "free", "occupied")
FIELD_SCHEMA = "lewm_go2_perfect_camera_ray_field_v1"
FRAME_AUDIT_SCHEMA = "lewm_go2_perfect_camera_ray_field_frame_audit_v1"
FIT_AUDIT_SCHEMA = "lewm_go2_perfect_camera_ray_field_fit_audit_v1"
GROUND_SUPPORT_OFFSETS = (
    (0.0, 0.0),
    (-0.5, -0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (0.5, 0.5),
)


def _finite_tuple(value: Sequence[float], *, length: int, name: str) -> tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    name: str,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if np.issubdtype(result.dtype, np.floating) and np.isnan(result).any():
        raise ValueError(f"{name} contains NaN")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CameraRaySpec:
    position_xyz_m: tuple[float, float, float]
    lookat_xyz_m: tuple[float, float, float]
    up_xyz: tuple[float, float, float]
    horizontal_fov_deg: float
    vertical_fov_deg: float
    near_m: float
    ground_plane_z_m: float = 0.0
    image_width_px: int = 224
    image_height_px: int = 168
    obstacle_ray_stride_px: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_xyz_m",
            _finite_tuple(self.position_xyz_m, length=3, name="camera position"),
        )
        object.__setattr__(
            self,
            "lookat_xyz_m",
            _finite_tuple(self.lookat_xyz_m, length=3, name="camera lookat"),
        )
        object.__setattr__(
            self,
            "up_xyz",
            _finite_tuple(self.up_xyz, length=3, name="camera up"),
        )
        for name in ("horizontal_fov_deg", "vertical_fov_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 180.0:
                raise ValueError(f"{name} must lie in (0, 180)")
            object.__setattr__(self, name, value)
        near = float(self.near_m)
        ground = float(self.ground_plane_z_m)
        if not math.isfinite(near) or near < 0.0:
            raise ValueError("near_m must be finite and non-negative")
        if not math.isfinite(ground):
            raise ValueError("ground_plane_z_m must be finite")
        object.__setattr__(self, "near_m", near)
        object.__setattr__(self, "ground_plane_z_m", ground)
        for name in (
            "image_width_px",
            "image_height_px",
            "obstacle_ray_stride_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @classmethod
    def from_camera_observation(cls, camera: Any) -> "CameraRaySpec":
        vertical = getattr(camera, "vertical_fov_deg", None)
        if vertical is None:
            raise ValueError("camera observation lacks a vertical FOV")
        return cls(
            position_xyz_m=tuple(camera.position_xyz_m),
            lookat_xyz_m=tuple(camera.lookat_xyz_m),
            up_xyz=tuple(camera.up_xyz),
            horizontal_fov_deg=float(camera.horizontal_fov_deg),
            vertical_fov_deg=float(vertical),
            near_m=float(camera.near_m),
            ground_plane_z_m=float(camera.ground_plane_z_m),
            image_width_px=int(camera.image_width_px),
            image_height_px=int(camera.image_height_px),
            obstacle_ray_stride_px=int(camera.obstacle_ray_stride_px),
        )


@dataclass(frozen=True)
class OrientedBox:
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        center = _finite_tuple(self.center_xyz_m, length=3, name="box center")
        size = _finite_tuple(self.size_xyz_m, length=3, name="box size")
        if any(item <= 0.0 for item in size):
            raise ValueError("box sizes must be positive")
        angles = _finite_tuple(
            (self.roll_rad, self.pitch_rad, self.yaw_rad),
            length=3,
            name="box attitude",
        )
        object.__setattr__(self, "center_xyz_m", center)
        object.__setattr__(self, "size_xyz_m", size)
        object.__setattr__(self, "roll_rad", angles[0])
        object.__setattr__(self, "pitch_rad", angles[1])
        object.__setattr__(self, "yaw_rad", angles[2])

    @classmethod
    def from_object(cls, value: Any) -> "OrientedBox":
        return cls(
            center_xyz_m=tuple(value.center_xyz_m),
            size_xyz_m=tuple(value.size_xyz_m),
            roll_rad=float(getattr(value, "roll_rad", 0.0)),
            pitch_rad=float(getattr(value, "pitch_rad", 0.0)),
            yaw_rad=float(getattr(value, "yaw_rad", 0.0)),
        )


@dataclass(frozen=True)
class OutputGridSpec:
    rows: int = 64
    cols: int = 64
    cell_size_m: float = 0.10
    forward_min_edge_m: float = -1.0
    left_min_edge_m: float = -3.2

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or isinstance(self.cols, bool):
            raise ValueError("grid dimensions must be positive integers")
        if int(self.rows) != self.rows or int(self.cols) != self.cols:
            raise ValueError("grid dimensions must be positive integers")
        if int(self.rows) <= 0 or int(self.cols) <= 0:
            raise ValueError("grid dimensions must be positive integers")
        values = _finite_tuple(
            (self.cell_size_m, self.forward_min_edge_m, self.left_min_edge_m),
            length=3,
            name="output grid geometry",
        )
        if values[0] <= 0.0:
            raise ValueError("output cell size must be positive")
        object.__setattr__(self, "rows", int(self.rows))
        object.__setattr__(self, "cols", int(self.cols))
        object.__setattr__(self, "cell_size_m", values[0])
        object.__setattr__(self, "forward_min_edge_m", values[1])
        object.__setattr__(self, "left_min_edge_m", values[2])

    @classmethod
    def from_local_grid(cls, grid: Any) -> "OutputGridSpec":
        return cls(
            rows=int(grid.rows),
            cols=int(grid.cols),
            cell_size_m=float(grid.cell_size_m),
            forward_min_edge_m=float(grid.forward_min_edge_m),
            left_min_edge_m=float(grid.left_min_edge_m),
        )

    def forward_centers_m(self) -> np.ndarray:
        return self.forward_min_edge_m + (
            np.arange(self.rows, dtype=np.float64) + 0.5
        ) * self.cell_size_m

    def left_centers_m(self) -> np.ndarray:
        return self.left_min_edge_m + (
            np.arange(self.cols, dtype=np.float64) + 0.5
        ) * self.cell_size_m


@dataclass(frozen=True)
class PhysicalWindow:
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    physical_free_mask: np.ndarray
    cell_size_m: float

    def __post_init__(self) -> None:
        x = _readonly_array(self.x_centers_m, dtype=np.float64, name="physical x")
        y = _readonly_array(self.y_centers_m, dtype=np.float64, name="physical y")
        free = _readonly_array(
            self.physical_free_mask, dtype=bool, name="physical free mask"
        )
        if x.ndim != 1 or y.ndim != 1 or free.shape != (x.size, y.size):
            raise ValueError("physical window axes and free mask do not match")
        if x.size == 0 or y.size == 0:
            raise ValueError("physical window must be nonempty")
        cell_size = float(self.cell_size_m)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("physical cell size must be positive")
        object.__setattr__(self, "x_centers_m", x)
        object.__setattr__(self, "y_centers_m", y)
        object.__setattr__(self, "physical_free_mask", free)
        object.__setattr__(self, "cell_size_m", cell_size)


@dataclass(frozen=True)
class PerfectCameraRayField:
    camera_position_xyz_m: np.ndarray
    physical_x_centers_m: np.ndarray
    physical_y_centers_m: np.ndarray
    physical_cell_size_m: float
    ground_support_in_frustum: np.ndarray
    ground_support_target_distance_m: np.ndarray
    ground_support_first_hit_distance_m: np.ndarray
    pixel_ray_directions_xyz: np.ndarray
    pixel_first_hit_distance_m: np.ndarray
    near_m: float

    def __post_init__(self) -> None:
        position = _readonly_array(
            self.camera_position_xyz_m, dtype=np.float64, name="field camera position"
        )
        x = _readonly_array(
            self.physical_x_centers_m, dtype=np.float64, name="field physical x"
        )
        y = _readonly_array(
            self.physical_y_centers_m, dtype=np.float64, name="field physical y"
        )
        in_frustum = _readonly_array(
            self.ground_support_in_frustum,
            dtype=bool,
            name="ground support frustum field",
        )
        target = _readonly_array(
            self.ground_support_target_distance_m,
            dtype=np.float64,
            name="ground support target distance field",
        )
        first = _readonly_array(
            self.ground_support_first_hit_distance_m,
            dtype=np.float64,
            name="ground support first-hit field",
        )
        pixel_directions = _readonly_array(
            self.pixel_ray_directions_xyz,
            dtype=np.float64,
            name="pixel ray directions",
        )
        pixel_first = _readonly_array(
            self.pixel_first_hit_distance_m,
            dtype=np.float64,
            name="pixel first-hit field",
        )
        expected_support = (x.size, y.size, len(GROUND_SUPPORT_OFFSETS))
        if position.shape != (3,):
            raise ValueError("field camera position must have shape [3]")
        if in_frustum.shape != expected_support:
            raise ValueError("ground support frustum field has the wrong shape")
        if target.shape != expected_support or first.shape != expected_support:
            raise ValueError("ground support distance fields have the wrong shape")
        if pixel_directions.ndim != 3 or pixel_directions.shape[2] != 3:
            raise ValueError("pixel ray directions must have shape [H, W, 3]")
        if pixel_first.shape != pixel_directions.shape[:2]:
            raise ValueError("pixel first-hit field does not match ray directions")
        if np.isneginf(target).any() or np.isneginf(first).any():
            raise ValueError("ground support distance fields contain -inf")
        if np.isneginf(pixel_first).any():
            raise ValueError("pixel first-hit field contains -inf")
        norms = np.linalg.norm(pixel_directions, axis=2)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("pixel ray directions are not unit length")
        cell_size = float(self.physical_cell_size_m)
        near = float(self.near_m)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("field physical cell size must be positive")
        if not math.isfinite(near) or near < 0.0:
            raise ValueError("field near plane must be finite and non-negative")
        object.__setattr__(self, "camera_position_xyz_m", position)
        object.__setattr__(self, "physical_x_centers_m", x)
        object.__setattr__(self, "physical_y_centers_m", y)
        object.__setattr__(self, "ground_support_in_frustum", in_frustum)
        object.__setattr__(self, "ground_support_target_distance_m", target)
        object.__setattr__(self, "ground_support_first_hit_distance_m", first)
        object.__setattr__(self, "pixel_ray_directions_xyz", pixel_directions)
        object.__setattr__(self, "pixel_first_hit_distance_m", pixel_first)
        object.__setattr__(self, "physical_cell_size_m", cell_size)
        object.__setattr__(self, "near_m", near)

    @property
    def ground_support_visible(self) -> np.ndarray:
        return self.ground_support_in_frustum & (
            self.ground_support_first_hit_distance_m
            >= self.ground_support_target_distance_m - 1e-9
        )

    def obstacle_first_hit_xy_m(self) -> np.ndarray:
        valid = np.isfinite(self.pixel_first_hit_distance_m) & (
            self.pixel_first_hit_distance_m > self.near_m
        )
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.float64)
        first_hits = self.camera_position_xyz_m[None, :] + (
            self.pixel_ray_directions_xyz[valid]
            * self.pixel_first_hit_distance_m[valid, None]
        )
        return np.unique(np.round(first_hits[:, :2], decimals=12), axis=0)

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(FIELD_SCHEMA.encode("ascii"))
        for value in (
            self.camera_position_xyz_m,
            self.physical_x_centers_m,
            self.physical_y_centers_m,
            self.ground_support_in_frustum,
            self.ground_support_target_distance_m,
            self.ground_support_first_hit_distance_m,
            self.pixel_ray_directions_xyz,
            self.pixel_first_hit_distance_m,
        ):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
        digest.update(np.float64(self.physical_cell_size_m).tobytes())
        digest.update(np.float64(self.near_m).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class RayFieldRasterization:
    contract_physical_labels: np.ndarray
    ray_only_physical_labels: np.ndarray
    contract_pre_veto_labels: np.ndarray
    ray_only_pre_veto_labels: np.ndarray
    collision_overlap: np.ndarray
    contract_labels: np.ndarray
    ray_only_labels: np.ndarray
    field_sha256: str

    def __post_init__(self) -> None:
        names = (
            "contract_physical_labels",
            "ray_only_physical_labels",
            "contract_pre_veto_labels",
            "ray_only_pre_veto_labels",
            "contract_labels",
            "ray_only_labels",
        )
        shape: tuple[int, ...] | None = None
        for name in names:
            value = _readonly_array(getattr(self, name), dtype=np.uint8, name=name)
            if not np.isin(value, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
                raise ValueError(f"{name} contains an unsupported class")
            if name.endswith("_physical_labels"):
                pass
            elif shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError("output raster shapes do not match")
            object.__setattr__(self, name, value)
        overlap = _readonly_array(
            self.collision_overlap, dtype=bool, name="collision overlap"
        )
        if shape is None or overlap.shape != shape:
            raise ValueError("collision overlap does not match output raster")
        object.__setattr__(self, "collision_overlap", overlap)
        if len(self.field_sha256) != 64:
            raise ValueError("field SHA-256 is malformed")


def _camera_basis(
    camera: CameraRaySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    position = np.asarray(camera.position_xyz_m, dtype=np.float64)
    lookat = np.asarray(camera.lookat_xyz_m, dtype=np.float64)
    up_hint = np.asarray(camera.up_xyz, dtype=np.float64)
    forward = lookat - position
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        raise ValueError("camera look direction is degenerate")
    forward /= norm
    right = np.cross(forward, up_hint)
    norm = float(np.linalg.norm(right))
    if norm <= 1e-9:
        raise ValueError("camera up is parallel to the look direction")
    right /= norm
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    return (
        position,
        forward,
        right,
        up,
        math.tan(math.radians(camera.horizontal_fov_deg) * 0.5),
        math.tan(math.radians(camera.vertical_fov_deg) * 0.5),
    )


def _box_rotation_matrix(box: OrientedBox) -> np.ndarray:
    cr, sr = math.cos(box.roll_rad), math.sin(box.roll_rad)
    cp, sp = math.cos(box.pitch_rad), math.sin(box.pitch_rad)
    cy, sy = math.cos(box.yaw_rad), math.sin(box.yaw_rad)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64
    )
    rotation_y = np.asarray(
        ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64
    )
    rotation_z = np.asarray(
        ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    return rotation_z @ rotation_y @ rotation_x


def _ray_box_entry_distances(
    origin_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    box: OrientedBox,
) -> np.ndarray:
    directions = np.asarray(directions_xyz, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions_xyz must have shape [N, 3]")
    rotation = _box_rotation_matrix(box)
    center = np.asarray(box.center_xyz_m, dtype=np.float64)
    half = 0.5 * np.asarray(box.size_xyz_m, dtype=np.float64)
    origin = rotation.T @ (np.asarray(origin_xyz, dtype=np.float64) - center)
    local = directions @ rotation
    t_min = np.full(directions.shape[0], -np.inf, dtype=np.float64)
    t_max = np.full(directions.shape[0], np.inf, dtype=np.float64)
    valid = np.ones(directions.shape[0], dtype=bool)
    for axis in range(3):
        component = local[:, axis]
        parallel = np.abs(component) <= 1e-12
        valid &= ~(parallel & (abs(origin[axis]) > half[axis] + 1e-12))
        low = np.full(component.shape, -np.inf, dtype=np.float64)
        high = np.full(component.shape, np.inf, dtype=np.float64)
        nonparallel = ~parallel
        low[nonparallel] = (-half[axis] - origin[axis]) / component[nonparallel]
        high[nonparallel] = (half[axis] - origin[axis]) / component[nonparallel]
        swap = low > high
        swapped_low = np.where(swap, high, low)
        swapped_high = np.where(swap, low, high)
        t_min = np.maximum(t_min, swapped_low)
        t_max = np.minimum(t_max, swapped_high)
    entry = np.maximum(t_min, 0.0)
    valid &= t_max + 1e-12 >= entry
    return np.where(valid, entry, np.inf)


def _nearest_box_hits(
    origin_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    boxes: Sequence[OrientedBox],
) -> np.ndarray:
    nearest = np.full(directions_xyz.shape[0], np.inf, dtype=np.float64)
    for box in boxes:
        nearest = np.minimum(
            nearest, _ray_box_entry_distances(origin_xyz, directions_xyz, box)
        )
    return nearest


def build_perfect_camera_ray_field(
    *,
    camera: CameraRaySpec,
    rendered_obstacle_boxes: Sequence[Any],
    physical_x_centers_m: np.ndarray,
    physical_y_centers_m: np.ndarray,
    physical_cell_size_m: float,
) -> PerfectCameraRayField:
    """Build exact prescribed-query and pixel-lattice first-hit fields."""

    x = np.asarray(physical_x_centers_m, dtype=np.float64)
    y = np.asarray(physical_y_centers_m, dtype=np.float64)
    cell_size = float(physical_cell_size_m)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("physical center axes must be nonempty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("physical center axes must be finite")
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("physical cell size must be positive")
    boxes = tuple(OrientedBox.from_object(box) for box in rendered_obstacle_boxes)
    position, forward, right, up, tan_h, tan_v = _camera_basis(camera)

    world_x, world_y = np.meshgrid(x, y, indexing="ij")
    offsets = np.asarray(GROUND_SUPPORT_OFFSETS, dtype=np.float64) * cell_size
    support_points = np.stack(
        (
            world_x[:, :, None] + offsets[None, None, :, 0],
            world_y[:, :, None] + offsets[None, None, :, 1],
            np.full(
                (x.size, y.size, offsets.shape[0]),
                camera.ground_plane_z_m,
                dtype=np.float64,
            ),
        ),
        axis=-1,
    )
    flat_points = support_points.reshape(-1, 3)
    relative = flat_points - position[None, :]
    forward_cam = relative @ forward
    right_cam = relative @ right
    up_cam = relative @ up
    target_distance = np.linalg.norm(relative, axis=1)
    in_frustum = (
        np.isfinite(relative).all(axis=1)
        & (forward_cam > camera.near_m)
        & (np.abs(right_cam) <= forward_cam * tan_h + 1e-12)
        & (np.abs(up_cam) <= forward_cam * tan_v + 1e-12)
    )
    support_first_hit = np.full(flat_points.shape[0], np.inf, dtype=np.float64)
    active = np.flatnonzero(in_frustum & (target_distance > 1e-12))
    if active.size:
        directions = relative[active] / target_distance[active, None]
        support_first_hit[active] = _nearest_box_hits(position, directions, boxes)

    pixel_x = (
        np.arange(0, camera.image_width_px, camera.obstacle_ray_stride_px,
                  dtype=np.float64)
        + 0.5 * camera.obstacle_ray_stride_px
    )
    pixel_y = (
        np.arange(0, camera.image_height_px, camera.obstacle_ray_stride_px,
                  dtype=np.float64)
        + 0.5 * camera.obstacle_ray_stride_px
    )
    pixel_x = np.minimum(pixel_x, camera.image_width_px - 0.5)
    pixel_y = np.minimum(pixel_y, camera.image_height_px - 0.5)
    normalized_x = (2.0 * pixel_x / camera.image_width_px - 1.0) * tan_h
    normalized_y = (1.0 - 2.0 * pixel_y / camera.image_height_px) * tan_v
    grid_x, grid_y = np.meshgrid(normalized_x, normalized_y, indexing="xy")
    pixel_directions = (
        forward[None, None, :]
        + grid_x[:, :, None] * right[None, None, :]
        + grid_y[:, :, None] * up[None, None, :]
    )
    pixel_directions /= np.linalg.norm(pixel_directions, axis=2, keepdims=True)
    pixel_first_hit = _nearest_box_hits(
        position, pixel_directions.reshape(-1, 3), boxes
    ).reshape(pixel_directions.shape[:2])

    support_shape = (x.size, y.size, offsets.shape[0])
    return PerfectCameraRayField(
        camera_position_xyz_m=position,
        physical_x_centers_m=x,
        physical_y_centers_m=y,
        physical_cell_size_m=cell_size,
        ground_support_in_frustum=in_frustum.reshape(support_shape),
        ground_support_target_distance_m=target_distance.reshape(support_shape),
        ground_support_first_hit_distance_m=support_first_hit.reshape(support_shape),
        pixel_ray_directions_xyz=pixel_directions,
        pixel_first_hit_distance_m=pixel_first_hit,
        near_m=camera.near_m,
    )


def output_world_centers(
    base_xy_yaw: Sequence[float],
    output_grid: OutputGridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    base_x, base_y, yaw = _finite_tuple(
        base_xy_yaw, length=3, name="base xy yaw"
    )
    forward, left = np.meshgrid(
        output_grid.forward_centers_m(), output_grid.left_centers_m(), indexing="ij"
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        base_x + cos_yaw * forward - sin_yaw * left,
        base_y + sin_yaw * forward + cos_yaw * left,
    )


def physical_window_for_output(
    *,
    physical_free_mask: np.ndarray,
    physical_origin_xy_m: Sequence[float],
    physical_cell_size_m: float,
    output_world_x_m: np.ndarray,
    output_world_y_m: np.ndarray,
    output_cell_size_m: float,
) -> PhysicalWindow:
    """Extract the exact world-aligned source window used by physical-v3."""

    full_free = np.asarray(physical_free_mask, dtype=bool)
    if full_free.ndim != 2 or full_free.size == 0:
        raise ValueError("physical_free_mask must be a nonempty matrix")
    origin_x, origin_y = _finite_tuple(
        physical_origin_xy_m, length=2, name="physical grid origin"
    )
    cell_size = float(physical_cell_size_m)
    output_size = float(output_cell_size_m)
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("physical cell size must be positive")
    if not math.isfinite(output_size) or output_size <= 0.0:
        raise ValueError("output cell size must be positive")
    output_x = np.asarray(output_world_x_m, dtype=np.float64)
    output_y = np.asarray(output_world_y_m, dtype=np.float64)
    if output_x.shape != output_y.shape or output_x.size == 0:
        raise ValueError("output coordinate arrays must be matching and nonempty")
    if not np.isfinite(output_x).all() or not np.isfinite(output_y).all():
        raise ValueError("output coordinate arrays must be finite")
    half_cell = 0.5 * cell_size
    output_half = 0.5 * output_size
    x_low = float(np.min(output_x)) - output_half - half_cell
    x_high = float(np.max(output_x)) + output_half + half_cell
    y_low = float(np.min(output_y)) - output_half - half_cell
    y_high = float(np.max(output_y)) + output_half + half_cell
    ix_low = int(math.floor((x_low - origin_x) / cell_size - 0.5)) - 1
    ix_high = int(math.ceil((x_high - origin_x) / cell_size - 0.5)) + 1
    iy_low = int(math.floor((y_low - origin_y) / cell_size - 0.5)) - 1
    iy_high = int(math.ceil((y_high - origin_y) / cell_size - 0.5)) + 1
    ix = np.arange(ix_low, ix_high + 1, dtype=np.int64)
    iy = np.arange(iy_low, iy_high + 1, dtype=np.int64)
    x_centers = origin_x + (ix.astype(np.float64) + 0.5) * cell_size
    y_centers = origin_y + (iy.astype(np.float64) + 0.5) * cell_size
    inside = (
        (ix[:, None] >= 0)
        & (ix[:, None] < full_free.shape[0])
        & (iy[None, :] >= 0)
        & (iy[None, :] < full_free.shape[1])
    )
    selected_free = np.zeros(inside.shape, dtype=bool)
    rows, cols = np.nonzero(inside)
    selected_free[rows, cols] = full_free[ix[rows], iy[cols]]
    return PhysicalWindow(
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        physical_free_mask=selected_free,
        cell_size_m=cell_size,
    )


def _physical_labels_from_field(
    field: PerfectCameraRayField,
    *,
    physical_free_mask: np.ndarray | None,
) -> np.ndarray:
    visible = np.all(field.ground_support_visible, axis=2)
    if physical_free_mask is not None:
        free = np.asarray(physical_free_mask, dtype=bool)
        if free.shape != visible.shape:
            raise ValueError("physical free mask does not match the ray field")
        visible &= free
    labels = np.full(visible.shape, UNKNOWN_CLASS, dtype=np.uint8)
    labels[visible] = FREE_CLASS
    return labels


def _aggregate_physical_labels(
    physical_labels: np.ndarray,
    *,
    physical_x_centers_m: np.ndarray,
    physical_y_centers_m: np.ndarray,
    output_world_x_m: np.ndarray,
    output_world_y_m: np.ndarray,
    output_yaw_rad: float,
    physical_cell_size_m: float,
    output_cell_size_m: float,
    obstacle_first_hit_xy_m: np.ndarray,
) -> np.ndarray:
    physical = np.asarray(physical_labels)
    x_centers = np.asarray(physical_x_centers_m, dtype=np.float64)
    y_centers = np.asarray(physical_y_centers_m, dtype=np.float64)
    output_x = np.asarray(output_world_x_m, dtype=np.float64)
    output_y = np.asarray(output_world_y_m, dtype=np.float64)
    if physical.shape != (x_centers.size, y_centers.size):
        raise ValueError("physical labels do not match their center axes")
    if output_x.shape != output_y.shape:
        raise ValueError("output world-coordinate arrays must match")
    if not np.isin(physical, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
        raise ValueError("physical labels contain an unsupported class")
    source_half = 0.5 * float(physical_cell_size_m)
    output_half = 0.5 * float(output_cell_size_m)
    if source_half <= 0.0 or output_half <= 0.0:
        raise ValueError("physical and output cell sizes must be positive")
    yaw = float(output_yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("output yaw must be finite")
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    output_u = np.asarray((cos_yaw, sin_yaw), dtype=np.float64)
    output_v = np.asarray((-sin_yaw, cos_yaw), dtype=np.float64)
    world_extent = output_half * (abs(cos_yaw) + abs(sin_yaw)) + source_half
    labels = np.full(output_x.shape, UNKNOWN_CLASS, dtype=np.uint8)
    for output_index in np.ndindex(output_x.shape):
        center_x = float(output_x[output_index])
        center_y = float(output_y[output_index])
        x_candidates = np.flatnonzero(
            np.abs(x_centers - center_x) <= world_extent + 1e-12
        )
        y_candidates = np.flatnonzero(
            np.abs(y_centers - center_y) <= world_extent + 1e-12
        )
        if x_candidates.size == 0 or y_candidates.size == 0:
            continue
        candidate_x, candidate_y = np.meshgrid(
            x_centers[x_candidates], y_centers[y_candidates], indexing="ij"
        )
        dx = candidate_x - center_x
        dy = candidate_y - center_y
        along_u = dx * output_u[0] + dy * output_u[1]
        along_v = dx * output_v[0] + dy * output_v[1]
        intersects = (
            (
                np.abs(along_u)
                <= output_half
                + source_half * (abs(output_u[0]) + abs(output_u[1]))
                + 1e-12
            )
            & (
                np.abs(along_v)
                <= output_half
                + source_half * (abs(output_v[0]) + abs(output_v[1]))
                + 1e-12
            )
            & (np.abs(dx) <= world_extent + 1e-12)
            & (np.abs(dy) <= world_extent + 1e-12)
        )
        support = physical[np.ix_(x_candidates, y_candidates)][intersects]
        if support.size > 0 and np.all(support == FREE_CLASS):
            labels[output_index] = FREE_CLASS

    witnesses = np.asarray(obstacle_first_hit_xy_m, dtype=np.float64)
    if witnesses.ndim != 2 or witnesses.shape[1] != 2:
        raise ValueError("obstacle first-hit witnesses must have shape [N, 2]")
    if not np.isfinite(witnesses).all():
        raise ValueError("obstacle first-hit witnesses must be finite")
    if witnesses.size:
        flat_x = output_x.ravel()
        flat_y = output_y.ravel()
        witnessed = np.zeros(flat_x.size, dtype=bool)
        for start in range(0, witnesses.shape[0], 512):
            batch = witnesses[start : start + 512]
            dx = batch[None, :, 0] - flat_x[:, None]
            dy = batch[None, :, 1] - flat_y[:, None]
            forward_delta = cos_yaw * dx + sin_yaw * dy
            left_delta = -sin_yaw * dx + cos_yaw * dy
            witnessed |= np.any(
                (np.abs(forward_delta) <= output_half + 1e-12)
                & (np.abs(left_delta) <= output_half + 1e-12),
                axis=1,
            )
        labels.ravel()[witnessed] = OCCUPIED_CLASS
    return labels


def _output_collision_overlap(
    output_x: np.ndarray,
    output_y: np.ndarray,
    *,
    output_yaw_rad: float,
    output_cell_size_m: float,
    collision_boxes: Sequence[Any],
) -> np.ndarray:
    overlap = np.zeros(output_x.shape, dtype=bool)
    output_half = 0.5 * float(output_cell_size_m)
    yaw = float(output_yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for raw_box in collision_boxes:
        box = OrientedBox.from_object(raw_box)
        rotation = _box_rotation_matrix(box)
        half = 0.5 * np.asarray(box.size_xyz_m, dtype=np.float64)
        local_corners = np.asarray(
            [
                (sx * half[0], sy * half[1], sz * half[2])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world_corners = (
            local_corners @ rotation.T
            + np.asarray(box.center_xyz_m, dtype=np.float64)[None, :]
        )
        x_low, y_low = np.min(world_corners[:, :2], axis=0)
        x_high, y_high = np.max(world_corners[:, :2], axis=0)
        center_x = 0.5 * (x_low + x_high)
        center_y = 0.5 * (y_low + y_high)
        half_x = 0.5 * (x_high - x_low)
        half_y = 0.5 * (y_high - y_low)
        dx = output_x - center_x
        dy = output_y - center_y
        along_u = cos_yaw * dx + sin_yaw * dy
        along_v = -sin_yaw * dx + cos_yaw * dy
        overlap |= (
            (
                np.abs(dx)
                <= half_x
                + output_half * (abs(cos_yaw) + abs(sin_yaw))
                + 1e-12
            )
            & (
                np.abs(dy)
                <= half_y
                + output_half * (abs(cos_yaw) + abs(sin_yaw))
                + 1e-12
            )
            & (
                np.abs(along_u)
                <= output_half
                + half_x * abs(cos_yaw)
                + half_y * abs(sin_yaw)
                + 1e-12
            )
            & (
                np.abs(along_v)
                <= output_half
                + half_x * abs(sin_yaw)
                + half_y * abs(cos_yaw)
                + 1e-12
            )
        )
    return overlap


def rasterize_perfect_camera_ray_field(
    *,
    field: PerfectCameraRayField,
    physical_free_mask: np.ndarray,
    output_world_x_m: np.ndarray,
    output_world_y_m: np.ndarray,
    output_yaw_rad: float,
    output_cell_size_m: float,
    collision_obstacle_boxes: Sequence[Any],
) -> RayFieldRasterization:
    """Rasterize exact ray evidence without calling the production builder."""

    contract_physical = _physical_labels_from_field(
        field, physical_free_mask=physical_free_mask
    )
    ray_only_physical = _physical_labels_from_field(field, physical_free_mask=None)
    witnesses = field.obstacle_first_hit_xy_m()
    common: dict[str, Any] = {
        "physical_x_centers_m": field.physical_x_centers_m,
        "physical_y_centers_m": field.physical_y_centers_m,
        "output_world_x_m": output_world_x_m,
        "output_world_y_m": output_world_y_m,
        "output_yaw_rad": output_yaw_rad,
        "physical_cell_size_m": field.physical_cell_size_m,
        "output_cell_size_m": output_cell_size_m,
        "obstacle_first_hit_xy_m": witnesses,
    }
    contract_pre_veto = _aggregate_physical_labels(contract_physical, **common)
    ray_only_pre_veto = _aggregate_physical_labels(ray_only_physical, **common)
    overlap = _output_collision_overlap(
        np.asarray(output_world_x_m, dtype=np.float64),
        np.asarray(output_world_y_m, dtype=np.float64),
        output_yaw_rad=output_yaw_rad,
        output_cell_size_m=output_cell_size_m,
        collision_boxes=collision_obstacle_boxes,
    )
    contract = np.array(contract_pre_veto, dtype=np.uint8, copy=True)
    ray_only = np.array(ray_only_pre_veto, dtype=np.uint8, copy=True)
    contract[(contract == FREE_CLASS) & overlap] = UNKNOWN_CLASS
    ray_only[(ray_only == FREE_CLASS) & overlap] = UNKNOWN_CLASS
    return RayFieldRasterization(
        contract_physical_labels=contract_physical,
        ray_only_physical_labels=ray_only_physical,
        contract_pre_veto_labels=contract_pre_veto,
        ray_only_pre_veto_labels=ray_only_pre_veto,
        collision_overlap=overlap,
        contract_labels=contract,
        ray_only_labels=ray_only,
        field_sha256=field.content_sha256(),
    )


def reconstruct_frame_from_perfect_rays(
    *,
    camera: CameraRaySpec,
    rendered_obstacle_boxes: Sequence[Any],
    collision_obstacle_boxes: Sequence[Any],
    base_xy_yaw: Sequence[float],
    physical_free_mask: np.ndarray,
    physical_origin_xy_m: Sequence[float],
    physical_cell_size_m: float,
    output_grid: OutputGridSpec = OutputGridSpec(),
) -> RayFieldRasterization:
    output_x, output_y = output_world_centers(base_xy_yaw, output_grid)
    window = physical_window_for_output(
        physical_free_mask=physical_free_mask,
        physical_origin_xy_m=physical_origin_xy_m,
        physical_cell_size_m=physical_cell_size_m,
        output_world_x_m=output_x,
        output_world_y_m=output_y,
        output_cell_size_m=output_grid.cell_size_m,
    )
    field = build_perfect_camera_ray_field(
        camera=camera,
        rendered_obstacle_boxes=rendered_obstacle_boxes,
        physical_x_centers_m=window.x_centers_m,
        physical_y_centers_m=window.y_centers_m,
        physical_cell_size_m=window.cell_size_m,
    )
    return rasterize_perfect_camera_ray_field(
        field=field,
        physical_free_mask=window.physical_free_mask,
        output_world_x_m=output_x,
        output_world_y_m=output_y,
        output_yaw_rad=float(base_xy_yaw[2]),
        output_cell_size_m=output_grid.cell_size_m,
        collision_obstacle_boxes=collision_obstacle_boxes,
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(labels == class_index))
        for class_index, name in enumerate(CLASS_NAMES)
    }


def _confusion(reference: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.count_nonzero((reference == expected) & (prediction == predicted)))
            for predicted in range(3)
        ]
        for expected in range(3)
    ]


def audit_frame_labels(
    *,
    authoritative_labels: np.ndarray,
    supervision_mask: np.ndarray,
    reconstruction: RayFieldRasterization,
    frame_key: Mapping[str, Any],
) -> dict[str, Any]:
    authoritative = np.asarray(authoritative_labels)
    supervision = np.asarray(supervision_mask)
    if authoritative.shape != (64, 64) or authoritative.dtype != np.dtype(np.uint8):
        raise ValueError("authoritative frame labels must be uint8 [64, 64]")
    if supervision.shape != (64, 64) or supervision.dtype != np.dtype(bool):
        raise ValueError("frame supervision must be bool [64, 64]")
    if not np.all(supervision):
        raise ValueError("the exact fit audit requires full-grid supervision")
    if not np.isin(authoritative, (0, 1, 2)).all():
        raise ValueError("authoritative frame labels contain an unsupported class")
    if reconstruction.contract_labels.shape != authoritative.shape:
        raise ValueError("reconstructed frame is not 64x64")
    contract_mismatch = authoritative != reconstruction.contract_labels
    ray_only_mismatch = authoritative != reconstruction.ray_only_labels
    return {
        "schema": FRAME_AUDIT_SCHEMA,
        "frame_key": dict(frame_key),
        "authoritative_labels_sha256": hashlib.sha256(
            authoritative.tobytes(order="C")
        ).hexdigest(),
        "contract_labels_sha256": hashlib.sha256(
            reconstruction.contract_labels.tobytes(order="C")
        ).hexdigest(),
        "ray_only_labels_sha256": hashlib.sha256(
            reconstruction.ray_only_labels.tobytes(order="C")
        ).hexdigest(),
        "field_sha256": reconstruction.field_sha256,
        "authoritative_class_counts": _class_counts(authoritative),
        "contract_class_counts": _class_counts(reconstruction.contract_labels),
        "ray_only_class_counts": _class_counts(reconstruction.ray_only_labels),
        "contract_confusion_reference_rows": _confusion(
            authoritative, reconstruction.contract_labels
        ),
        "ray_only_confusion_reference_rows": _confusion(
            authoritative, reconstruction.ray_only_labels
        ),
        "contract_mismatch_cell_count": int(np.count_nonzero(contract_mismatch)),
        "ray_only_mismatch_cell_count": int(np.count_nonzero(ray_only_mismatch)),
        "contract_mismatch_sample": [
            [int(row), int(column)]
            for row, column in np.argwhere(contract_mismatch)[:32]
        ],
        "ray_only_mismatch_sample": [
            [int(row), int(column)]
            for row, column in np.argwhere(ray_only_mismatch)[:32]
        ],
        "collision_veto_cell_count": int(
            np.count_nonzero(reconstruction.collision_overlap)
        ),
    }


def summarize_exact_fit(frame_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exactly 320 ordered frame reports and fail closed otherwise."""

    reports = list(frame_reports)
    if len(reports) != 320:
        raise ValueError("the perfect-ray fit audit requires exactly 320 frames")
    if any(report.get("schema") != FRAME_AUDIT_SCHEMA for report in reports):
        raise ValueError("one frame report has the wrong schema")
    keys = [report.get("frame_key") for report in reports]
    encoded_keys = [repr(sorted(dict(key).items())) for key in keys if isinstance(key, Mapping)]
    if len(encoded_keys) != 320 or len(set(encoded_keys)) != 320:
        raise ValueError("fit frame keys must be exactly unique")

    def sum_confusion(name: str) -> list[list[int]]:
        total = np.zeros((3, 3), dtype=np.int64)
        for report in reports:
            value = np.asarray(report[name])
            if value.shape != (3, 3) or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{name} is malformed")
            total += value.astype(np.int64)
        return total.tolist()

    contract_mismatches = sum(int(item["contract_mismatch_cell_count"]) for item in reports)
    ray_only_mismatches = sum(int(item["ray_only_mismatch_cell_count"]) for item in reports)
    return {
        "schema": FIT_AUDIT_SCHEMA,
        "frame_count": 320,
        "cell_count": 320 * 64 * 64,
        "contract_assisted": {
            "exact": contract_mismatches == 0,
            "mismatch_frame_count": sum(
                int(item["contract_mismatch_cell_count"]) > 0 for item in reports
            ),
            "mismatch_cell_count": contract_mismatches,
            "confusion_reference_rows": sum_confusion(
                "contract_confusion_reference_rows"
            ),
        },
        "ray_only": {
            "exact": ray_only_mismatches == 0,
            "mismatch_frame_count": sum(
                int(item["ray_only_mismatch_cell_count"]) > 0 for item in reports
            ),
            "mismatch_cell_count": ray_only_mismatches,
            "confusion_reference_rows": sum_confusion(
                "ray_only_confusion_reference_rows"
            ),
        },
        "ordered_frame_keys_sha256": hashlib.sha256(
            "\n".join(encoded_keys).encode("utf-8")
        ).hexdigest(),
        "ordered_authoritative_label_hashes_sha256": hashlib.sha256(
            "\n".join(str(item["authoritative_labels_sha256"]) for item in reports).encode(
                "ascii"
            )
        ).hexdigest(),
        "ordered_contract_label_hashes_sha256": hashlib.sha256(
            "\n".join(str(item["contract_labels_sha256"]) for item in reports).encode(
                "ascii"
            )
        ).hexdigest(),
        "ordered_ray_only_label_hashes_sha256": hashlib.sha256(
            "\n".join(str(item["ray_only_labels_sha256"]) for item in reports).encode(
                "ascii"
            )
        ).hexdigest(),
    }


__all__ = [
    "CLASS_NAMES",
    "CameraRaySpec",
    "FIELD_SCHEMA",
    "FIT_AUDIT_SCHEMA",
    "FRAME_AUDIT_SCHEMA",
    "FREE_CLASS",
    "OCCUPIED_CLASS",
    "OrientedBox",
    "OutputGridSpec",
    "PerfectCameraRayField",
    "PhysicalWindow",
    "RayFieldRasterization",
    "UNKNOWN_CLASS",
    "audit_frame_labels",
    "build_perfect_camera_ray_field",
    "output_world_centers",
    "physical_window_for_output",
    "rasterize_perfect_camera_ray_field",
    "reconstruct_frame_from_perfect_rays",
    "summarize_exact_fit",
]
