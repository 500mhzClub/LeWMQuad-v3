"""Convert ``lewm_contract`` dataclass records into ROS 2 message instances.

The Genesis bulk rollout produces sim-agnostic ``*Record`` dataclasses; the
on-disk bag format is ROS 2 typed (rosbag2 MCAP). This module is the
serialization boundary. Importing it requires the colcon overlay (ROS 2
Jazzy + the workspace ``lewm_go2_control`` install) to be sourced.

Usage::

    msg = command_block_record_to_msg(record)
    serialized = serialize_message(msg)
    writer.write(topic, serialized, stamp_ns)

Standard ``sensor_msgs`` / ``nav_msgs`` builders are included for the
non-LeWM streams (image, IMU, joint state, camera info, odometry) that the
data spec requires alongside the LeWM contract topics.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from lewm_genesis.lewm_contract import (
    BaseStateRecord,
    CommandBlockRecord,
    EpisodeInfoRecord,
    ExecutedCommandBlockRecord,
    FootContactsRecord,
    ResetEventRecord,
)
from lewm_genesis.scene_loader import CameraMount


def _require_ros() -> None:
    try:
        import rclpy  # noqa: F401
        import lewm_go2_control.msg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when overlay missing
        raise RuntimeError(
            "ROS 2 Jazzy overlay (workspace install/setup.bash) must be sourced "
            "before importing lewm_genesis.ros_msg_adapter."
        ) from exc


def stamp_from_ns(stamp_ns: int) -> tuple[int, int]:
    """Return ``(sec, nanosec)`` for ``std_msgs/Header.stamp``."""

    sec = int(stamp_ns // 1_000_000_000)
    nanosec = int(stamp_ns % 1_000_000_000)
    return sec, nanosec


def _set_header(header: Any, stamp_ns: int, frame_id: str) -> None:
    sec, nanosec = stamp_from_ns(stamp_ns)
    header.stamp.sec = sec
    header.stamp.nanosec = nanosec
    header.frame_id = str(frame_id)


# ---------------------------------------------------------------------------
# LeWM message builders
# ---------------------------------------------------------------------------


def command_block_record_to_msg(record: CommandBlockRecord) -> Any:
    _require_ros()
    from lewm_go2_control.msg import CommandBlock

    msg = CommandBlock()
    _set_header(msg.header, record.stamp_ns, record.frame_id)
    msg.sequence_id = int(record.sequence_id)
    msg.block_size = int(record.block_size)
    msg.command_dt_s = float(record.command_dt_s)
    msg.primitive_name = str(record.primitive_name)
    msg.vx_body_mps = [float(v) for v in record.vx_body_mps]
    msg.vy_body_mps = [float(v) for v in record.vy_body_mps]
    msg.yaw_rate_radps = [float(v) for v in record.yaw_rate_radps]
    msg.event_name = str(record.event_name)
    msg.event_allowed_in_training = bool(record.event_allowed_in_training)
    return msg


def executed_command_block_record_to_msg(
    record: ExecutedCommandBlockRecord,
) -> Any:
    _require_ros()
    from lewm_go2_control.msg import ExecutedCommandBlock

    msg = ExecutedCommandBlock()
    _set_header(msg.header, record.stamp_ns, record.frame_id)
    msg.sequence_id = int(record.sequence_id)
    msg.block_size = int(record.block_size)
    msg.command_dt_s = float(record.command_dt_s)
    msg.primitive_name = str(record.primitive_name)
    msg.requested_vx_body_mps = [float(v) for v in record.requested_vx_body_mps]
    msg.requested_vy_body_mps = [float(v) for v in record.requested_vy_body_mps]
    msg.requested_yaw_rate_radps = [float(v) for v in record.requested_yaw_rate_radps]
    msg.executed_vx_body_mps = [float(v) for v in record.executed_vx_body_mps]
    msg.executed_vy_body_mps = [float(v) for v in record.executed_vy_body_mps]
    msg.executed_yaw_rate_radps = [float(v) for v in record.executed_yaw_rate_radps]
    msg.clipped = bool(record.clipped)
    msg.safety_overridden = bool(record.safety_overridden)
    msg.controller_mode = str(record.controller_mode)
    msg.backend_id = str(record.backend_id)
    return msg


def base_state_record_to_msg(record: BaseStateRecord) -> Any:
    _require_ros()
    from lewm_go2_control.msg import BaseState

    msg = BaseState()
    _set_header(msg.header, record.stamp_ns, record.frame_id)
    msg.pose_world.position.x = float(record.pose_world_xyz[0])
    msg.pose_world.position.y = float(record.pose_world_xyz[1])
    msg.pose_world.position.z = float(record.pose_world_xyz[2])
    msg.pose_world.orientation.x = float(record.pose_world_quat_xyzw[0])
    msg.pose_world.orientation.y = float(record.pose_world_quat_xyzw[1])
    msg.pose_world.orientation.z = float(record.pose_world_quat_xyzw[2])
    msg.pose_world.orientation.w = float(record.pose_world_quat_xyzw[3])
    msg.twist_world.linear.x = float(record.twist_world_linear[0])
    msg.twist_world.linear.y = float(record.twist_world_linear[1])
    msg.twist_world.linear.z = float(record.twist_world_linear[2])
    msg.twist_world.angular.x = float(record.twist_world_angular[0])
    msg.twist_world.angular.y = float(record.twist_world_angular[1])
    msg.twist_world.angular.z = float(record.twist_world_angular[2])
    msg.twist_body.linear.x = float(record.twist_body_linear[0])
    msg.twist_body.linear.y = float(record.twist_body_linear[1])
    msg.twist_body.linear.z = float(record.twist_body_linear[2])
    msg.twist_body.angular.x = float(record.twist_body_angular[0])
    msg.twist_body.angular.y = float(record.twist_body_angular[1])
    msg.twist_body.angular.z = float(record.twist_body_angular[2])
    msg.quat_world_xyzw = [float(q) for q in record.pose_world_quat_xyzw]
    msg.roll_rad = float(record.roll_rad)
    msg.pitch_rad = float(record.pitch_rad)
    msg.yaw_rad = float(record.yaw_rad)
    return msg


def foot_contacts_record_to_msg(record: FootContactsRecord) -> Any:
    _require_ros()
    from lewm_go2_control.msg import FootContacts

    msg = FootContacts()
    _set_header(msg.header, record.stamp_ns, record.frame_id)
    msg.fl_contact = bool(record.fl_contact)
    msg.fr_contact = bool(record.fr_contact)
    msg.rl_contact = bool(record.rl_contact)
    msg.rr_contact = bool(record.rr_contact)
    msg.fl_force_n = float(record.fl_force_n)
    msg.fr_force_n = float(record.fr_force_n)
    msg.rl_force_n = float(record.rl_force_n)
    msg.rr_force_n = float(record.rr_force_n)
    return msg


def reset_event_record_to_msg(record: ResetEventRecord) -> Any:
    _require_ros()
    from lewm_go2_control.msg import ResetEvent

    msg = ResetEvent()
    _set_header(msg.header, record.stamp_ns, "")
    msg.scene_id = int(record.scene_id)
    msg.episode_id = int(record.episode_id)
    msg.reset_count = int(record.reset_count)
    msg.reason = str(record.reason)
    msg.success = bool(record.success)
    msg.spawn_pose_world.position.x = float(record.spawn_pose_xyz[0])
    msg.spawn_pose_world.position.y = float(record.spawn_pose_xyz[1])
    msg.spawn_pose_world.position.z = float(record.spawn_pose_xyz[2])
    msg.spawn_pose_world.orientation.x = float(record.spawn_pose_quat_xyzw[0])
    msg.spawn_pose_world.orientation.y = float(record.spawn_pose_quat_xyzw[1])
    msg.spawn_pose_world.orientation.z = float(record.spawn_pose_quat_xyzw[2])
    msg.spawn_pose_world.orientation.w = float(record.spawn_pose_quat_xyzw[3])
    return msg


def episode_info_record_to_msg(record: EpisodeInfoRecord) -> Any:
    _require_ros()
    from lewm_go2_control.msg import EpisodeInfo

    msg = EpisodeInfo()
    _set_header(msg.header, record.stamp_ns, "")
    msg.scene_id = int(record.scene_id)
    msg.episode_id = int(record.episode_id)
    msg.episode_step = int(record.episode_step)
    msg.reset_count = int(record.reset_count)
    msg.scene_family = str(record.scene_family)
    msg.split = str(record.split)
    msg.manifest_sha256 = str(record.manifest_sha256)
    return msg


# ---------------------------------------------------------------------------
# Standard sensor_msgs / nav_msgs builders
# ---------------------------------------------------------------------------


def rgb_image_to_msg(
    rgb_hwc: np.ndarray,
    *,
    stamp_ns: int,
    frame_id: str,
    encoding: str = "rgb8",
) -> Any:
    """Pack an ``(H, W, 3)`` uint8 RGB array into ``sensor_msgs/Image``."""

    _require_ros()
    from sensor_msgs.msg import Image

    if rgb_hwc.dtype != np.uint8:
        raise ValueError(f"rgb image must be uint8; got {rgb_hwc.dtype}")
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"rgb image must be (H, W, 3); got {rgb_hwc.shape}")
    height, width, _ = rgb_hwc.shape
    msg = Image()
    _set_header(msg.header, stamp_ns, frame_id)
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = str(encoding)
    msg.is_bigendian = 0
    msg.step = int(width * 3)
    msg.data = bytes(np.ascontiguousarray(rgb_hwc).tobytes())
    return msg


def camera_info_to_msg(
    mount: CameraMount,
    *,
    stamp_ns: int,
    frame_id: str,
) -> Any:
    """Synthesize ``sensor_msgs/CameraInfo`` from the platform manifest mount.

    Mirrors the existing ``lewm_go2_control camera_info_publisher`` logic:
    pinhole model derived from horizontal FOV + native resolution.
    """

    _require_ros()
    import math

    from sensor_msgs.msg import CameraInfo

    if mount.fov_axis != "horizontal":
        raise NotImplementedError(
            f"only horizontal-fov camera info synthesis is implemented; got {mount.fov_axis}"
        )
    width, height = mount.native_resolution
    fov_rad = math.radians(float(mount.fov_deg))
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx  # square pixels assumption
    cx = width / 2.0
    cy = height / 2.0
    msg = CameraInfo()
    _set_header(msg.header, stamp_ns, frame_id)
    msg.height = int(height)
    msg.width = int(width)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def imu_to_msg(
    *,
    quat_xyzw: tuple[float, float, float, float],
    angular_vel_body: tuple[float, float, float],
    linear_accel_body: tuple[float, float, float],
    stamp_ns: int,
    frame_id: str,
) -> Any:
    """Build ``sensor_msgs/Imu`` from per-tick body-frame state."""

    _require_ros()
    from sensor_msgs.msg import Imu

    msg = Imu()
    _set_header(msg.header, stamp_ns, frame_id)
    msg.orientation.x = float(quat_xyzw[0])
    msg.orientation.y = float(quat_xyzw[1])
    msg.orientation.z = float(quat_xyzw[2])
    msg.orientation.w = float(quat_xyzw[3])
    msg.orientation_covariance = [0.0] * 9
    msg.angular_velocity.x = float(angular_vel_body[0])
    msg.angular_velocity.y = float(angular_vel_body[1])
    msg.angular_velocity.z = float(angular_vel_body[2])
    msg.angular_velocity_covariance = [0.0] * 9
    msg.linear_acceleration.x = float(linear_accel_body[0])
    msg.linear_acceleration.y = float(linear_accel_body[1])
    msg.linear_acceleration.z = float(linear_accel_body[2])
    msg.linear_acceleration_covariance = [0.0] * 9
    return msg


def joint_state_to_msg(
    *,
    joint_names: Iterable[str],
    positions: Iterable[float],
    velocities: Iterable[float],
    efforts: Iterable[float] | None,
    stamp_ns: int,
    frame_id: str = "",
) -> Any:
    """Build ``sensor_msgs/JointState``."""

    _require_ros()
    from sensor_msgs.msg import JointState

    msg = JointState()
    _set_header(msg.header, stamp_ns, frame_id)
    msg.name = [str(n) for n in joint_names]
    msg.position = [float(p) for p in positions]
    msg.velocity = [float(v) for v in velocities]
    msg.effort = [float(e) for e in efforts] if efforts is not None else []
    return msg


def odometry_from_base_state(
    record: BaseStateRecord,
    *,
    child_frame_id: str = "base_link",
) -> Any:
    """Build ``nav_msgs/Odometry`` from a ``BaseStateRecord``."""

    _require_ros()
    from nav_msgs.msg import Odometry

    msg = Odometry()
    _set_header(msg.header, record.stamp_ns, record.frame_id)
    msg.child_frame_id = str(child_frame_id)
    msg.pose.pose.position.x = float(record.pose_world_xyz[0])
    msg.pose.pose.position.y = float(record.pose_world_xyz[1])
    msg.pose.pose.position.z = float(record.pose_world_xyz[2])
    msg.pose.pose.orientation.x = float(record.pose_world_quat_xyzw[0])
    msg.pose.pose.orientation.y = float(record.pose_world_quat_xyzw[1])
    msg.pose.pose.orientation.z = float(record.pose_world_quat_xyzw[2])
    msg.pose.pose.orientation.w = float(record.pose_world_quat_xyzw[3])
    msg.twist.twist.linear.x = float(record.twist_body_linear[0])
    msg.twist.twist.linear.y = float(record.twist_body_linear[1])
    msg.twist.twist.linear.z = float(record.twist_body_linear[2])
    msg.twist.twist.angular.x = float(record.twist_body_angular[0])
    msg.twist.twist.angular.y = float(record.twist_body_angular[1])
    msg.twist.twist.angular.z = float(record.twist_body_angular[2])
    return msg
