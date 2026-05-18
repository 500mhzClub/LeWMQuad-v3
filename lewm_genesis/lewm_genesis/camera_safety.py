"""Egocentric camera placement guards for wall-heavy scenes.

The renderer keeps the platform camera mount as the nominal pose, then applies
the same safety idea used in LeWMQuad-v2: cast a few frustum rays against scene
boxes and retract the optical centre along ``-forward`` only when the nominal
camera would clip into or through geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from lewm_genesis.lewm_contract import rotate_body_to_world


BODY_ENVELOPE_FORWARD_M = 0.06
DEFAULT_MAX_RETRACT_M = 0.08
DEFAULT_INSIDE_MARGIN_M = 0.02
RETRACT_MARGIN_M = 0.005


@dataclass(frozen=True)
class CameraSafetyConfig:
    safe_clearance_m: float
    near_plane_m: float
    fov_deg: float
    fov_axis: str = "horizontal"
    aspect_ratio: float = 4.0 / 3.0
    lookat_dist_m: float = 1.0
    inside_margin_m: float = DEFAULT_INSIDE_MARGIN_M
    max_retract_m: float = DEFAULT_MAX_RETRACT_M


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray
    lookat: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    rotation: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": [float(v) for v in self.position],
            "lookat": [float(v) for v in self.lookat],
            "up": [float(v) for v in self.up],
        }


# Static object kinds whose meshes do NOT occlude the camera. Empty by
# default: beacons (kind="landmark") used to be excluded so the lens could
# pass through them in tight motif scenes, but that produced training frames
# of the camera staring at the inside of a beacon mesh. We now treat
# landmarks as occluders too — the route teacher's standoff logic keeps the
# robot out of beacon cells, and the rare collector-driven beacon-cell
# occupancy correctly marks those frames invalid.
NON_OCCLUDING_KINDS: frozenset[str] = frozenset()


def occlusion_objects_only(objects: Iterable[Any]) -> tuple[Any, ...]:
    """Return the subset of ``objects`` that should be treated as occluders.

    Camera-safety raycasts and inside-geometry tests should ignore visual-only
    objects like landmarks; otherwise the camera goes "unsafe" whenever it
    enters a landmark cell even though the landmark is the thing we want it
    to see.
    """
    return tuple(
        obj for obj in objects
        if str(getattr(obj, "kind", "")) not in NON_OCCLUDING_KINDS
    )


def camera_safety_config_from_pack(pack: Any) -> CameraSafetyConfig:
    constraints = getattr(pack, "camera_constraints", {}) or {}
    camera = getattr(pack, "camera")
    resolution = tuple(getattr(camera, "native_resolution", (640, 480)))
    width = float(resolution[0]) if resolution else 640.0
    height = float(resolution[1]) if len(resolution) > 1 else 480.0
    mount_x = abs(float(getattr(camera, "xyz_body_m", (0.0, 0.0, 0.0))[0]))
    max_retract = max(DEFAULT_MAX_RETRACT_M, mount_x - BODY_ENVELOPE_FORWARD_M)
    return CameraSafetyConfig(
        safe_clearance_m=max(
            float(getattr(camera, "near_m", 0.05)),
            float(constraints.get("min_camera_clearance_m", 0.10)),
        ),
        near_plane_m=float(getattr(camera, "near_m", 0.05)),
        fov_deg=float(getattr(camera, "fov_deg", 78.323)),
        fov_axis=str(getattr(camera, "fov_axis", "horizontal")),
        aspect_ratio=max(width / max(height, 1.0), 1e-6),
        max_retract_m=max(0.0, max_retract),
    )


def camera_pose_from_base(
    base_pos_world: Iterable[float],
    base_quat_xyzw: Iterable[float],
    *,
    mount_xyz_body: Iterable[float],
    mount_rpy_body: Iterable[float],
    lookat_dist_m: float = 1.0,
) -> CameraPose:
    base_pos = np.asarray(tuple(float(v) for v in base_pos_world), dtype=np.float32)
    base_quat = np.asarray(tuple(float(v) for v in base_quat_xyzw), dtype=np.float32)
    mount_rot = _rpy_matrix(tuple(float(v) for v in mount_rpy_body))
    mount_xyz = np.asarray(tuple(float(v) for v in mount_xyz_body), dtype=np.float32)
    forward_body = mount_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up_body = mount_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_pos = base_pos + rotate_body_to_world(mount_xyz, base_quat).astype(np.float32)
    forward = _normalize(rotate_body_to_world(forward_body, base_quat).astype(np.float32))
    up = _normalize(rotate_body_to_world(up_body, base_quat).astype(np.float32))
    rotation = _camera_rotation_from_forward_up(forward, up)
    lookat = cam_pos + float(lookat_dist_m) * forward
    return CameraPose(
        position=cam_pos.astype(np.float32),
        lookat=lookat.astype(np.float32),
        up=up.astype(np.float32),
        forward=forward.astype(np.float32),
        rotation=rotation.astype(np.float32),
    )


def camera_pose_from_dict(pose: dict[str, Any]) -> CameraPose:
    position = np.asarray(pose["position"], dtype=np.float32)
    lookat = np.asarray(pose["lookat"], dtype=np.float32)
    up = _normalize(np.asarray(pose["up"], dtype=np.float32))
    forward = _normalize(lookat - position)
    rotation = _camera_rotation_from_forward_up(forward, up)
    return CameraPose(
        position=position,
        lookat=lookat,
        up=up,
        forward=forward,
        rotation=rotation,
    )


def safe_camera_pose(
    pose: CameraPose,
    objects: Iterable[Any],
    config: CameraSafetyConfig,
) -> tuple[CameraPose, dict[str, float | bool]]:
    boxes = occlusion_objects_only(objects)
    initial = camera_safety_metrics(pose, boxes, config)
    if not bool(initial["unsafe"]):
        return pose, {**initial, "retracted_m": 0.0}

    max_retract = float(config.max_retract_m)
    if max_retract <= 0.0:
        return pose, {**initial, "retracted_m": 0.0}

    candidate_distances = np.linspace(
        min(0.005, max_retract),
        max_retract,
        num=24,
        dtype=np.float32,
    )
    best_pose = pose
    best_metrics = initial
    best_retract = 0.0
    for retract in candidate_distances:
        candidate = _translated_pose(pose, -float(retract) * pose.forward, config.lookat_dist_m)
        metrics = camera_safety_metrics(candidate, boxes, config)
        best_pose = candidate
        best_metrics = metrics
        best_retract = float(retract)
        if not bool(metrics["unsafe"]):
            return candidate, {**metrics, "retracted_m": best_retract}

    return best_pose, {**best_metrics, "retracted_m": best_retract}


def safe_camera_pose_from_base(
    base_pos_world: Iterable[float],
    base_quat_xyzw: Iterable[float],
    *,
    mount_xyz_body: Iterable[float],
    mount_rpy_body: Iterable[float],
    objects: Iterable[Any],
    config: CameraSafetyConfig,
) -> tuple[CameraPose, dict[str, float | bool]]:
    pose = camera_pose_from_base(
        base_pos_world,
        base_quat_xyzw,
        mount_xyz_body=mount_xyz_body,
        mount_rpy_body=mount_rpy_body,
        lookat_dist_m=float(config.lookat_dist_m),
    )
    return safe_camera_pose(pose, objects, config)


def camera_safety_metrics(
    pose: CameraPose,
    objects: Iterable[Any],
    config: CameraSafetyConfig,
) -> dict[str, float | bool]:
    boxes = occlusion_objects_only(objects)
    inside = _camera_inside_any_box(pose.position, boxes, margin=float(config.inside_margin_m))
    clearance = _camera_clearance_to_any_box(pose.position, boxes)
    forward_hit = _ray_hit_distance_to_any_box(pose.position, pose.forward, boxes)
    frustum_hit = _frustum_min_hit_distance(pose.position, pose.rotation, boxes, config)
    safe = float(config.safe_clearance_m)
    unsafe = bool(inside or clearance < safe or frustum_hit < safe)
    return {
        "inside_geometry": inside,
        "clearance_m": clearance,
        "forward_hit_m": forward_hit,
        "frustum_min_hit_m": frustum_hit,
        "unsafe": unsafe,
    }


def depth_buffer_has_clipping(
    depth: np.ndarray,
    near_plane_m: float,
    *,
    frac_threshold: float = 0.005,
) -> bool:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.size == 0:
        return False
    at_near = depth_arr <= float(near_plane_m) * 1.05
    return float(at_near.mean()) >= float(frac_threshold)


def _translated_pose(pose: CameraPose, delta: np.ndarray, lookat_dist_m: float) -> CameraPose:
    position = pose.position + np.asarray(delta, dtype=np.float32)
    lookat = position + float(lookat_dist_m) * pose.forward
    return CameraPose(
        position=position.astype(np.float32),
        lookat=lookat.astype(np.float32),
        up=pose.up,
        forward=pose.forward,
        rotation=pose.rotation,
    )


def _frustum_min_hit_distance(
    position: np.ndarray,
    rotation: np.ndarray,
    objects: Iterable[Any],
    config: CameraSafetyConfig,
) -> float:
    min_hit = float("inf")
    for local_dir in _frustum_ray_directions(config):
        world_dir = rotation @ local_dir
        min_hit = min(min_hit, _ray_hit_distance_to_any_box(position, world_dir, objects))
    return float(min_hit)


def _frustum_ray_directions(config: CameraSafetyConfig) -> tuple[np.ndarray, ...]:
    half_angle = math.radians(float(config.fov_deg) * 0.5)
    if str(config.fov_axis).lower() == "vertical":
        tan_v = math.tan(half_angle)
        tan_h = tan_v * float(config.aspect_ratio)
    else:
        tan_h = math.tan(half_angle)
        tan_v = tan_h / max(float(config.aspect_ratio), 1e-6)
    raw = (
        (1.0, 0.0, 0.0),
        (1.0, -tan_h, +tan_v),
        (1.0, +tan_h, +tan_v),
        (1.0, -tan_h, -tan_v),
        (1.0, +tan_h, -tan_v),
        (1.0, -tan_h, 0.0),
        (1.0, +tan_h, 0.0),
        (1.0, 0.0, +tan_v),
        (1.0, 0.0, -tan_v),
    )
    return tuple(_normalize(np.asarray(direction, dtype=np.float32)) for direction in raw)


def _camera_inside_any_box(position: np.ndarray, objects: Iterable[Any], *, margin: float) -> bool:
    for obj in objects:
        local = _world_point_to_box_local(position, obj)
        hx, hy, hz = _box_half_extents(obj)
        if (
            abs(float(local[0])) < hx + float(margin)
            and abs(float(local[1])) < hy + float(margin)
            and abs(float(local[2])) < hz
        ):
            return True
    return False


def _camera_clearance_to_any_box(position: np.ndarray, objects: Iterable[Any]) -> float:
    min_clearance = float("inf")
    for obj in objects:
        local = _world_point_to_box_local(position, obj)
        hx, hy, hz = _box_half_extents(obj)
        dx = max(abs(float(local[0])) - hx, 0.0)
        dy = max(abs(float(local[1])) - hy, 0.0)
        dz = max(abs(float(local[2])) - hz, 0.0)
        min_clearance = min(min_clearance, math.sqrt(dx * dx + dy * dy + dz * dz))
    return float(min_clearance)


def _ray_hit_distance_to_any_box(
    origin: np.ndarray,
    direction: np.ndarray,
    objects: Iterable[Any],
) -> float:
    ray_dir = _normalize(direction)
    min_distance = float("inf")
    for obj in objects:
        local_origin = _world_point_to_box_local(origin, obj)
        local_direction = _world_vector_to_box_local(ray_dir, obj)
        hit = _ray_hit_distance_to_local_aabb(local_origin, local_direction, _box_half_extents(obj))
        min_distance = min(min_distance, hit)
    return float(min_distance)


def _ray_hit_distance_to_local_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    half_extents: tuple[float, float, float],
) -> float:
    t_min = -float("inf")
    t_max = float("inf")
    eps = 1e-8
    for axis, half_extent in enumerate(half_extents):
        o = float(origin[axis])
        d = float(direction[axis])
        lo = -float(half_extent)
        hi = float(half_extent)
        if abs(d) < eps:
            if o < lo or o > hi:
                return float("inf")
            continue
        t0 = (lo - o) / d
        t1 = (hi - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return float("inf")
    if t_max < 0.0:
        return float("inf")
    return float(max(0.0, t_min))


def _world_point_to_box_local(point: np.ndarray, obj: Any) -> np.ndarray:
    center = np.asarray(getattr(obj, "center_xyz_m"), dtype=np.float32)
    return _world_vector_to_box_local(np.asarray(point, dtype=np.float32) - center, obj)


def _world_vector_to_box_local(vector: np.ndarray, obj: Any) -> np.ndarray:
    yaw = -float(getattr(obj, "yaw_rad", 0.0))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = float(vector[0])
    y = float(vector[1])
    return np.array(
        [
            cos_y * x - sin_y * y,
            sin_y * x + cos_y * y,
            float(vector[2]),
        ],
        dtype=np.float32,
    )


def _box_half_extents(obj: Any) -> tuple[float, float, float]:
    sx, sy, sz = tuple(float(v) for v in getattr(obj, "size_xyz_m"))
    return sx * 0.5, sy * 0.5, sz * 0.5


def _camera_rotation_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = _normalize(forward)
    up_norm = _normalize(up)
    right = _normalize(np.cross(up_norm, fwd))
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    corrected_up = _normalize(np.cross(fwd, right))
    return np.stack([fwd, right, corrected_up], axis=1).astype(np.float32)


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return arr.copy()
    return (arr / norm).astype(np.float32)
