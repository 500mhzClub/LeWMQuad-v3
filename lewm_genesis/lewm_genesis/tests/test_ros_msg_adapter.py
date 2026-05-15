"""Tests for ``lewm_genesis.ros_msg_adapter``.

Requires the ROS 2 Jazzy overlay (workspace ``install/setup.bash``) sourced.
Tests skip cleanly when the overlay is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import rclpy  # noqa: F401
    import lewm_go2_control.msg  # noqa: F401

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROS_AVAILABLE,
    reason="ROS 2 Jazzy overlay + lewm_go2_control install required",
)

from lewm_genesis.lewm_contract import (
    BaseStateRecord,
    CommandBlockRecord,
    EpisodeInfoRecord,
    ExecutedCommandBlockRecord,
    FootContactsRecord,
    ResetEventRecord,
)
from lewm_genesis.scene_loader import CameraMount

if ROS_AVAILABLE:
    from lewm_genesis.ros_msg_adapter import (
        base_state_record_to_msg,
        camera_info_to_msg,
        command_block_record_to_msg,
        episode_info_record_to_msg,
        executed_command_block_record_to_msg,
        foot_contacts_record_to_msg,
        imu_to_msg,
        joint_state_to_msg,
        odometry_from_base_state,
        reset_event_record_to_msg,
        rgb_image_to_msg,
        stamp_from_ns,
    )


def test_stamp_from_ns_splits_seconds_and_nanos():
    sec, nanosec = stamp_from_ns(1_234_567_890_123)
    assert sec == 1234
    assert nanosec == 567_890_123


def test_command_block_record_round_trips_all_fields():
    rec = CommandBlockRecord(
        sequence_id=7,
        block_size=5,
        command_dt_s=0.10,
        primitive_name="forward_slow",
        vx_body_mps=[0.2] * 5,
        vy_body_mps=[0.0] * 5,
        yaw_rate_radps=[0.0] * 5,
        event_name="",
        event_allowed_in_training=False,
        stamp_ns=1_000_000_000,
        frame_id="base_link",
    )
    msg = command_block_record_to_msg(rec)
    assert msg.sequence_id == 7
    assert msg.block_size == 5
    assert msg.command_dt_s == pytest.approx(0.10)
    assert msg.primitive_name == "forward_slow"
    # ROS float32 storage means equality must allow float32 round-trip error.
    assert list(msg.vx_body_mps) == pytest.approx([0.2] * 5, rel=1e-6)
    assert msg.header.stamp.sec == 1
    assert msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == "base_link"


def test_executed_command_block_round_trips_safety_flags():
    rec = ExecutedCommandBlockRecord(
        sequence_id=42,
        block_size=3,
        command_dt_s=0.10,
        primitive_name="forward_fast",
        requested_vx_body_mps=[0.3, 0.3, 0.3],
        requested_vy_body_mps=[0.0, 0.0, 0.0],
        requested_yaw_rate_radps=[0.0, 0.0, 0.0],
        executed_vx_body_mps=[0.25, 0.3, 0.3],
        executed_vy_body_mps=[0.0, 0.0, 0.0],
        executed_yaw_rate_radps=[0.0, 0.0, 0.0],
        clipped=True,
        safety_overridden=False,
        controller_mode="cmd_vel",
        backend_id="genesis_tier_a",
        stamp_ns=2_500_000_000,
    )
    msg = executed_command_block_record_to_msg(rec)
    assert msg.sequence_id == 42
    assert list(msg.requested_vx_body_mps) == pytest.approx([0.3, 0.3, 0.3], rel=1e-6)
    assert list(msg.executed_vx_body_mps) == pytest.approx([0.25, 0.3, 0.3], rel=1e-6)
    assert msg.clipped is True
    assert msg.controller_mode == "cmd_vel"
    assert msg.backend_id == "genesis_tier_a"


def test_base_state_record_preserves_pose_and_twist():
    rec = BaseStateRecord(
        pose_world_xyz=(1.0, 2.0, 0.4),
        pose_world_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        twist_body_linear=(0.3, 0.0, 0.0),
        twist_body_angular=(0.0, 0.0, 0.1),
        twist_world_linear=(0.3, 0.0, 0.0),
        twist_world_angular=(0.0, 0.0, 0.1),
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        stamp_ns=500_000_000,
    )
    msg = base_state_record_to_msg(rec)
    assert msg.pose_world.position.x == pytest.approx(1.0)
    assert msg.pose_world.position.z == pytest.approx(0.4)
    assert msg.pose_world.orientation.w == pytest.approx(1.0)
    assert msg.twist_body.linear.x == pytest.approx(0.3)
    assert msg.twist_world.angular.z == pytest.approx(0.1)
    assert list(msg.quat_world_xyzw) == [0.0, 0.0, 0.0, 1.0]
    assert msg.header.stamp.nanosec == 500_000_000


def test_foot_contacts_record_maps_lewm_order():
    rec = FootContactsRecord(
        fl_contact=True,
        fr_contact=False,
        rl_contact=True,
        rr_contact=False,
    )
    msg = foot_contacts_record_to_msg(rec)
    assert msg.fl_contact is True
    assert msg.fr_contact is False
    assert msg.rl_contact is True
    assert msg.rr_contact is False
    assert msg.fl_force_n == pytest.approx(0.0)


def test_reset_event_record_preserves_episode_counters():
    rec = ResetEventRecord(
        scene_id=11,
        episode_id=3,
        reset_count=3,
        reason="fall",
        success=True,
        spawn_pose_xyz=(0.5, 0.5, 0.375),
        spawn_pose_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        stamp_ns=10_000_000_000,
    )
    msg = reset_event_record_to_msg(rec)
    assert msg.scene_id == 11
    assert msg.episode_id == 3
    assert msg.reset_count == 3
    assert msg.reason == "fall"
    assert msg.success is True
    assert msg.spawn_pose_world.position.x == pytest.approx(0.5)
    assert msg.spawn_pose_world.orientation.w == pytest.approx(1.0)


def test_episode_info_record_carries_provenance():
    rec = EpisodeInfoRecord(
        scene_id=99,
        episode_id=2,
        episode_step=17,
        reset_count=2,
        scene_family="medium_enclosed_maze",
        split="train",
        manifest_sha256="deadbeef" * 8,
        stamp_ns=100,
    )
    msg = episode_info_record_to_msg(rec)
    assert msg.scene_id == 99
    assert msg.episode_id == 2
    assert msg.episode_step == 17
    assert msg.scene_family == "medium_enclosed_maze"
    assert msg.manifest_sha256 == "deadbeef" * 8


def test_rgb_image_to_msg_packs_uint8_data():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[..., 1] = 128
    msg = rgb_image_to_msg(rgb, stamp_ns=0, frame_id="camera_link")
    assert msg.height == 240
    assert msg.width == 320
    assert msg.encoding == "rgb8"
    assert msg.step == 320 * 3
    assert len(msg.data) == 240 * 320 * 3
    assert msg.data[1] == 128


def test_rgb_image_to_msg_rejects_wrong_dtype():
    bad = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        rgb_image_to_msg(bad, stamp_ns=0, frame_id="x")


def test_camera_info_synthesizes_pinhole_from_horizontal_fov():
    mount = CameraMount(
        parent_link="camera_link",
        xyz_body_m=(0.326, 0.0, 0.043),
        rpy_body_rad=(0.0, 0.0, 0.0),
        native_resolution=(640, 480),
        training_resolution=(224, 224),
        fov_axis="horizontal",
        fov_deg=78.323,
        near_m=0.05,
        far_m=200.0,
        encoding="rgb8",
    )
    msg = camera_info_to_msg(mount, stamp_ns=0, frame_id="camera_link")
    assert msg.height == 480
    assert msg.width == 640
    assert msg.distortion_model == "plumb_bob"
    # Principal point should land at image center.
    assert msg.k[2] == pytest.approx(320.0)
    assert msg.k[5] == pytest.approx(240.0)
    # fx == fy with square pixel assumption.
    assert msg.k[0] == pytest.approx(msg.k[4])


def test_imu_to_msg_carries_orientation_and_rates():
    msg = imu_to_msg(
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        angular_vel_body=(0.01, 0.02, 0.03),
        linear_accel_body=(0.0, 0.0, 9.81),
        stamp_ns=1_000_000_000,
        frame_id="imu_link",
    )
    assert msg.orientation.w == pytest.approx(1.0)
    assert msg.angular_velocity.z == pytest.approx(0.03)
    assert msg.linear_acceleration.z == pytest.approx(9.81)
    assert msg.header.frame_id == "imu_link"


def test_joint_state_to_msg_populates_fields():
    msg = joint_state_to_msg(
        joint_names=["FL_hip_joint", "FL_thigh_joint"],
        positions=[0.0, 0.5],
        velocities=[0.1, 0.2],
        efforts=[1.0, 2.0],
        stamp_ns=0,
    )
    assert list(msg.name) == ["FL_hip_joint", "FL_thigh_joint"]
    assert list(msg.position) == pytest.approx([0.0, 0.5])
    assert list(msg.velocity) == pytest.approx([0.1, 0.2])
    assert list(msg.effort) == pytest.approx([1.0, 2.0])


def test_odometry_from_base_state_copies_pose_and_twist():
    rec = BaseStateRecord(
        pose_world_xyz=(1.0, 0.0, 0.4),
        pose_world_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        twist_body_linear=(0.3, 0.0, 0.0),
        twist_body_angular=(0.0, 0.0, 0.1),
        twist_world_linear=(0.3, 0.0, 0.0),
        twist_world_angular=(0.0, 0.0, 0.1),
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
    )
    msg = odometry_from_base_state(rec)
    assert msg.child_frame_id == "base_link"
    assert msg.pose.pose.position.x == pytest.approx(1.0)
    assert msg.twist.twist.linear.x == pytest.approx(0.3)
