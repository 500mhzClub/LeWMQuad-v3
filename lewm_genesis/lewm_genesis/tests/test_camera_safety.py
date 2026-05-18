from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lewm_genesis.camera_safety import (
    CameraPose,
    CameraSafetyConfig,
    camera_safety_metrics,
    occlusion_objects_only,
    safe_camera_pose,
)


@dataclass(frozen=True)
class Box:
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    kind: str = "wall"


def _pose(x: float) -> CameraPose:
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    position = np.array([x, 0.0, 0.3], dtype=np.float32)
    return CameraPose(
        position=position,
        lookat=position + forward,
        up=up,
        forward=forward,
        rotation=np.stack([forward, right, up], axis=1),
    )


def test_safe_camera_pose_retracts_until_frustum_clears_wall() -> None:
    wall = Box(center_xyz_m=(0.35, 0.0, 0.3), size_xyz_m=(0.10, 1.0, 0.6))
    config = CameraSafetyConfig(
        safe_clearance_m=0.10,
        near_plane_m=0.05,
        fov_deg=60.0,
        max_retract_m=0.30,
    )
    initial = _pose(0.245)

    before = camera_safety_metrics(initial, (wall,), config)
    adjusted, after = safe_camera_pose(initial, (wall,), config)

    assert before["unsafe"] is True
    assert after["unsafe"] is False
    assert after["retracted_m"] > 0.0
    assert adjusted.position[0] < initial.position[0]


def test_safe_camera_pose_reports_unresolved_when_retraction_limit_is_too_small() -> None:
    wall = Box(center_xyz_m=(0.35, 0.0, 0.3), size_xyz_m=(0.10, 1.0, 0.6))
    config = CameraSafetyConfig(
        safe_clearance_m=0.10,
        near_plane_m=0.05,
        fov_deg=60.0,
        max_retract_m=0.02,
    )

    _adjusted, after = safe_camera_pose(_pose(0.245), (wall,), config)

    assert after["unsafe"] is True
    assert after["retracted_m"] > 0.0


def test_landmarks_block_camera_like_walls() -> None:
    # Landmarks are no longer transparent to the safety raycast: a beacon
    # mesh that the camera would clip into must flag unsafe (and inside the
    # geometry when the lens is fully past the front face). This is the
    # opposite of the old behaviour — beacons were treated as see-through
    # but produced training frames staring at the inside of a mesh.
    landmark = Box(
        center_xyz_m=(0.05, 0.0, 0.44),
        size_xyz_m=(0.30, 0.30, 0.88),
        kind="landmark",
    )
    config = CameraSafetyConfig(
        safe_clearance_m=0.10,
        near_plane_m=0.05,
        fov_deg=60.0,
        max_retract_m=0.20,
    )
    metrics = camera_safety_metrics(_pose(0.0), (landmark,), config)
    assert metrics["unsafe"] is True
    assert metrics["inside_geometry"] is True


def test_walls_still_block_when_landmark_is_in_the_scene() -> None:
    # Camera sits in front of a wall; a landmark exists elsewhere in the
    # scene but is well clear of the retraction path. The wall must still
    # drive a safe retraction. (Landmarks now occlude too, so we keep the
    # beacon out of the camera's path to test the wall-only resolution.)
    wall = Box(center_xyz_m=(0.35, 0.0, 0.3), size_xyz_m=(0.10, 1.0, 0.6), kind="wall")
    landmark = Box(
        center_xyz_m=(0.05, 2.0, 0.44),
        size_xyz_m=(0.30, 0.30, 0.88),
        kind="landmark",
    )
    config = CameraSafetyConfig(
        safe_clearance_m=0.10,
        near_plane_m=0.05,
        fov_deg=60.0,
        max_retract_m=0.30,
    )
    before = camera_safety_metrics(_pose(0.245), (wall, landmark), config)
    adjusted, after = safe_camera_pose(_pose(0.245), (wall, landmark), config)
    assert before["unsafe"] is True
    assert after["unsafe"] is False
    assert adjusted.position[0] < 0.245


def test_occlusion_filter_now_keeps_landmarks() -> None:
    wall = Box(center_xyz_m=(0, 0, 0.3), size_xyz_m=(0.1, 1.0, 0.6), kind="wall")
    landmark = Box(center_xyz_m=(0, 1, 0.4), size_xyz_m=(0.3, 0.3, 0.8), kind="landmark")
    obstacle = Box(center_xyz_m=(1, 0, 0.3), size_xyz_m=(0.5, 0.5, 0.6), kind="box_obstacle")
    filtered = occlusion_objects_only((wall, landmark, obstacle))
    # NON_OCCLUDING_KINDS is empty now; every static object participates in
    # the safety raycast, including landmarks.
    assert wall in filtered and obstacle in filtered and landmark in filtered
