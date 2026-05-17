"""Tests for render-replay planning from converted raw_rollout JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lewm_genesis.render_replay import build_render_replay_plan


def test_render_replay_uses_canonical_env_topics_and_joint_state(tmp_path: Path):
    raw_dir = tmp_path / "raw_rollout"
    raw_dir.mkdir()
    messages_path = raw_dir / "messages.jsonl"
    records = [
        {
            "topic": "/env_00/lewm/episode_info",
            "canonical_topic": "/lewm/episode_info",
            "env_index": 0,
            "timestamp_ns": 0,
            "payload": {
                "scene_id": 123,
                "episode_id": 1,
                "episode_step": 0,
                "reset_count": 1,
                "split": "train",
                "manifest_sha256": "abc",
            },
        },
        {
            "topic": "/env_00/joint_states",
            "canonical_topic": "/joint_states",
            "env_index": 0,
            "timestamp_ns": 0,
            "payload": {
                "name": ["FL_hip_joint"],
                "position": [0.1],
                "velocity": [0.2],
                "effort": [],
            },
        },
        {
            "topic": "/env_00/lewm/go2/base_state",
            "canonical_topic": "/lewm/go2/base_state",
            "env_index": 0,
            "timestamp_ns": 0,
            "payload": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}},
                "pose_world": {"position": {"x": 1.0, "y": 2.0, "z": 0.4}},
                "quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "roll_rad": 0.0,
                "pitch_rad": 0.0,
                "yaw_rad": 0.0,
                "twist_body": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}},
            },
        },
    ]
    with messages_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    (raw_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "lewm_raw_rollout_smoke_v0",
                "messages_jsonl": str(messages_path),
                "contract_audit": {"pass": True},
                "data_quality_audit": {"pass": True, "profile": "raw_pilot"},
            }
        ),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform.yaml"
    platform_manifest.write_text(
        """
camera:
  native_resolution: [640, 480]
  training_resolution: [224, 224]
  fov_axis: horizontal
  fov_deg: 78.323
  near_m: 0.05
  far_m: 200.0
  encoding: rgb8
  parent_link: camera_link
  xyz_body_m: [0.326, 0.0, 0.043]
  rpy_body_rad: [0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )

    plan = build_render_replay_plan(
        raw_dir,
        tmp_path / "render_plan",
        platform_manifest=platform_manifest,
    )

    assert plan["frame_count"] == 1
    assert plan["source_env_count"] == 1
    frame = json.loads(Path(plan["frames_jsonl"]).read_text(encoding="utf-8").splitlines()[0])
    assert frame["source_topic"] == "/env_00/lewm/go2/base_state"
    assert frame["canonical_topic"] == "/lewm/go2/base_state"
    assert frame["env_index"] == 0
    assert frame["episode"]["scene_id"] == 123
    assert frame["joint_state"]["names"] == ["FL_hip_joint"]
    assert frame["camera_pose_world"]["position"] == pytest.approx([1.326, 2.0, 0.443])
    assert frame["camera_pose_world"]["lookat"] == pytest.approx([2.326, 2.0, 0.443])
