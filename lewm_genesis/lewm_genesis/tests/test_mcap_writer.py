"""Integration tests for ``lewm_genesis.mcap_writer``.

Writes a small per-scene MCAP, reads it back via ``rosbag2_py``, and asserts
the topic list + first-message round trip.

Requires the ROS 2 Jazzy overlay + workspace install sourced. Skips cleanly
otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import rosbag2_py  # noqa: F401
    import rclpy.serialization  # noqa: F401
    import lewm_go2_control.msg  # noqa: F401

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROS_AVAILABLE,
    reason="ROS 2 Jazzy overlay + lewm_go2_control install + rosbag2_py required",
)

from lewm_genesis.lewm_contract import (
    BaseStateRecord,
    CommandBlockRecord,
    EpisodeInfoRecord,
    ExecutedCommandBlockRecord,
    FootContactsRecord,
)
from lewm_genesis.scene_loader import (
    CameraMount,
    PhysicsTiming,
    RobotSpec,
    ScenePack,
    StaticObject,
)

if ROS_AVAILABLE:
    from lewm_genesis.mcap_writer import (
        GLOBAL_CLOCK_TOPIC,
        MCAPSceneWriter,
        topic_name,
        topic_type,
    )
    from lewm_genesis.ros_msg_adapter import (
        base_state_record_to_msg,
        command_block_record_to_msg,
        episode_info_record_to_msg,
        executed_command_block_record_to_msg,
        foot_contacts_record_to_msg,
    )


def _fake_pack(scene_id: str = "smoke_0001") -> ScenePack:
    return ScenePack(
        scene_id=scene_id,
        family="open_obstacle_field",
        split="train",
        difficulty_tier="smoke",
        manifest_sha256="deadbeef" * 8,
        physics_seed=1,
        topology_seed=2,
        visual_seed=3,
        world_bounds_xy_m=((-5.0, -5.0), (5.0, 5.0)),
        static_objects=(),
        robot=RobotSpec(
            urdf_path=Path("/dev/null"),
            spawn_xyz_m=(0.0, 0.0, 0.375),
            spawn_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            foot_links_in_lewm_order=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        ),
        camera=CameraMount(
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
        ),
        timing=PhysicsTiming(
            physics_dt_s=0.002,
            policy_dt_s=0.02,
            command_dt_s=0.10,
            action_block_size=5,
        ),
        camera_constraints={},
        source_dir=Path("/dev/null"),
    )


def test_topic_name_and_type_resolve():
    assert topic_name("base_state", 0) == "/env_00/lewm/go2/base_state"
    assert topic_name("rgb_image", 13) == "/env_13/rgb_image"
    assert topic_type("command_block") == "lewm_go2_control/msg/CommandBlock"


def test_writer_round_trips_lewm_messages(tmp_path):
    pack = _fake_pack()
    out_dir = tmp_path / "raw_rollouts"

    with MCAPSceneWriter(pack, out_dir, n_envs=2) as writer:
        # Per-env messages.
        for env_idx in range(2):
            cb = command_block_record_to_msg(
                CommandBlockRecord(
                    sequence_id=env_idx,
                    block_size=5,
                    command_dt_s=0.10,
                    primitive_name="forward_slow",
                    vx_body_mps=[0.2] * 5,
                    vy_body_mps=[0.0] * 5,
                    yaw_rate_radps=[0.0] * 5,
                    stamp_ns=100 * (env_idx + 1),
                )
            )
            writer.write_env(env_idx, "command_block", cb, stamp_ns=100 * (env_idx + 1))

            base = base_state_record_to_msg(
                BaseStateRecord(
                    pose_world_xyz=(float(env_idx), 0.0, 0.4),
                    pose_world_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    twist_body_linear=(0.0, 0.0, 0.0),
                    twist_body_angular=(0.0, 0.0, 0.0),
                    twist_world_linear=(0.0, 0.0, 0.0),
                    twist_world_angular=(0.0, 0.0, 0.0),
                    roll_rad=0.0,
                    pitch_rad=0.0,
                    yaw_rad=0.0,
                    stamp_ns=200 * (env_idx + 1),
                )
            )
            writer.write_env(env_idx, "base_state", base, stamp_ns=200 * (env_idx + 1))

        writer.write_clock(stamp_ns=10_000)

    summary_path = out_dir / pack.scene_id / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["scene_id"] == pack.scene_id
    assert summary["n_envs"] == 2
    assert summary["stats"]["total_messages"] >= 5
    assert "/env_00/lewm/go2/command_block" in summary["stats"]["per_topic_counts"]
    assert summary["stats"]["per_topic_counts"]["/clock"] == 1

    # Read the bag back and verify topic list + first message of one topic.
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(out_dir / pack.scene_id), storage_id="mcap"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)
    topic_metadata = {t.name: t.type for t in reader.get_all_topics_and_types()}
    assert "/env_00/lewm/go2/command_block" in topic_metadata
    assert (
        topic_metadata["/env_00/lewm/go2/command_block"]
        == "lewm_go2_control/msg/CommandBlock"
    )
    assert GLOBAL_CLOCK_TOPIC[0] in topic_metadata


def test_writer_refuses_unregistered_topic(tmp_path):
    pack = _fake_pack(scene_id="smoke_0002")
    with MCAPSceneWriter(pack, tmp_path / "rr", n_envs=1) as writer:
        from lewm_go2_control.msg import CommandBlock

        with pytest.raises(KeyError):
            writer.write("/not/a/registered/topic", CommandBlock(), stamp_ns=0)


def test_writer_fails_if_output_dir_exists(tmp_path):
    pack = _fake_pack(scene_id="smoke_0003")
    out_dir = tmp_path / "rr"
    (out_dir / pack.scene_id).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        MCAPSceneWriter(pack, out_dir, n_envs=1)


def test_writer_records_per_topic_counts(tmp_path):
    pack = _fake_pack(scene_id="smoke_0004")
    out_dir = tmp_path / "rr"
    with MCAPSceneWriter(pack, out_dir, n_envs=1) as writer:
        for tick in range(5):
            msg = executed_command_block_record_to_msg(
                ExecutedCommandBlockRecord(
                    sequence_id=tick,
                    block_size=1,
                    command_dt_s=0.10,
                    primitive_name="hold",
                    requested_vx_body_mps=[0.0],
                    requested_vy_body_mps=[0.0],
                    requested_yaw_rate_radps=[0.0],
                    executed_vx_body_mps=[0.0],
                    executed_vy_body_mps=[0.0],
                    executed_yaw_rate_radps=[0.0],
                    clipped=False,
                    safety_overridden=False,
                    controller_mode="cmd_vel",
                    stamp_ns=tick * 100,
                )
            )
            writer.write_env(0, "executed_command_block", msg, stamp_ns=tick * 100)

    summary = json.loads((out_dir / pack.scene_id / "summary.json").read_text())
    counts = summary["stats"]["per_topic_counts"]
    assert counts["/env_00/lewm/go2/executed_command_block"] == 5
    assert summary["stats"]["first_stamp_ns"] == 0
    assert summary["stats"]["last_stamp_ns"] == 400
