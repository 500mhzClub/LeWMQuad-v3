"""Tests for offline Phase A1 derived-label helpers."""

from __future__ import annotations

import math

import pytest

from lewm_worlds.labels.derived import (
    DerivedLabelComputer,
    PoseStep,
    label_to_jsonable,
    pose_steps_from_message_records,
)
from lewm_worlds.labels.topology import local_graph_type_per_node
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)


def _constraints() -> CameraValidityConstraints:
    return CameraValidityConstraints(
        min_wall_thickness_m=0.08,
        near_m=0.05,
        far_m=200.0,
        min_camera_clearance_m=0.10,
    )


def _manifest(
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
    *,
    scene_id: str = "derived_test",
    landmarks: tuple[BoxObject, ...] = (),
    walls: tuple[BoxObject, ...] = (),
) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="test",
        difficulty_tier="test",
        topology_seed=123,
        visual_seed=456,
        physics_seed=789,
        world_bounds_xy_m=((-2.0, -2.0), (4.0, 2.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=_constraints(),
        walls=walls,
    )


def _stamp(sec: int, nanosec: int = 0) -> dict:
    return {"sec": sec, "nanosec": nanosec}


def _header(ns: int) -> dict:
    return {"stamp": _stamp(ns // 1_000_000_000, ns % 1_000_000_000), "frame_id": ""}


def _base_record(ns: int, x: float, y: float, yaw: float, *, env: int = 0) -> dict:
    return {
        "topic": f"/env_{env:02d}/lewm/go2/base_state",
        "canonical_topic": "/lewm/go2/base_state",
        "env_index": env,
        "timestamp_ns": ns,
        "payload": {
            "header": _header(ns),
            "pose_world": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "yaw_rad": yaw,
            "quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_local_graph_type_classifies_corridor_turn_and_t_junction():
    straight = _manifest(
        nodes=(
            GraphNode(0, (-1.0, 0.0), 1.0),
            GraphNode(1, (0.0, 0.0), 1.0),
            GraphNode(2, (1.0, 0.0), 1.0),
        ),
        edges=(
            GraphEdge(0, 1, 1.0),
            GraphEdge(1, 2, 1.0),
        ),
    )
    assert local_graph_type_per_node(straight)[1] == "corridor"

    turn = _manifest(
        nodes=(
            GraphNode(0, (0.0, 0.0), 1.0),
            GraphNode(1, (1.0, 0.0), 1.0),
            GraphNode(2, (1.0, 1.0), 1.0),
        ),
        edges=(
            GraphEdge(0, 1, 1.0),
            GraphEdge(1, 2, 1.0),
        ),
    )
    assert local_graph_type_per_node(turn)[1] == "turn"

    t_junction = _manifest(
        nodes=(
            GraphNode(0, (0.0, 0.0), 1.0),
            GraphNode(1, (1.0, 0.0), 1.0),
            GraphNode(2, (-1.0, 0.0), 1.0),
            GraphNode(3, (0.0, 1.0), 1.0),
        ),
        edges=(
            GraphEdge(0, 1, 1.0),
            GraphEdge(0, 2, 1.0),
            GraphEdge(0, 3, 1.0),
        ),
    )
    labels = local_graph_type_per_node(t_junction)
    assert labels[0] == "t_junction"
    assert labels[1] == "dead_end"


def test_derived_label_computer_emits_topology_yaw_bfs_and_motion_labels():
    landmark = BoxObject(
        object_id="goal_red",
        kind="landmark",
        center_xyz_m=(2.0, 0.0, 0.5),
        size_xyz_m=(0.2, 0.2, 1.0),
        yaw_rad=0.0,
        material_id="landmark_red",
    )
    manifest = _manifest(
        nodes=(
            GraphNode(0, (0.0, 0.0), 1.0),
            GraphNode(1, (1.0, 0.0), 1.0),
            GraphNode(2, (2.0, 0.0), 1.0),
        ),
        edges=(GraphEdge(0, 1, 1.0), GraphEdge(1, 2, 1.0)),
        landmarks=(landmark,),
    )

    computer = DerivedLabelComputer(manifest)
    first = computer.step(
        PoseStep(
            timestamp_ns=100_000_000,
            env_idx=0,
            episode_id=4,
            episode_step=1,
            position_xy_world=(0.05, 0.0),
            yaw_world_rad=0.0,
            last_command=(0.4, 0.0, 0.0),
        )
    )
    second = computer.step(
        PoseStep(
            timestamp_ns=200_000_000,
            env_idx=0,
            episode_id=4,
            episode_step=2,
            position_xy_world=(1.0, 0.0),
            yaw_world_rad=math.pi * 0.5,
            last_command=(0.4, 0.0, 0.0),
        )
    )

    first_payload = label_to_jsonable(first)
    assert first_payload["scene_id"] == "derived_test"
    assert first_payload["cell_id"] == 0
    assert first_payload["yaw_bin"] == 4
    assert first_payload["local_graph_type"] == "dead_end"
    assert first_payload["bfs_distance_to_landmark"] == {"goal_red": 2}
    assert first_payload["landmarks"][0]["visible"] is True
    assert first_payload["landmarks"][0]["bearing_body_rad"] == pytest.approx(0.0)

    second_payload = label_to_jsonable(second)
    assert second_payload["cell_id"] == 1
    assert second_payload["yaw_bin"] == 6
    assert second_payload["local_graph_type"] == "corridor"
    assert second_payload["bfs_distance_to_landmark"] == {"goal_red": 1}
    assert second.integrated_body_motion_block[0] == pytest.approx(0.95)
    assert second.integrated_body_motion_block[1] == pytest.approx(0.0)


def test_pose_steps_from_message_records_joins_episode_and_executed_commands():
    records = [
        {
            "topic": "/env_00/lewm/go2/command_block",
            "canonical_topic": "/lewm/go2/command_block",
            "env_index": 0,
            "timestamp_ns": 0,
            "payload": {
                "header": _header(0),
                "sequence_id": 7,
                "command_dt_s": 0.1,
                "vx_body_mps": [1.0, 2.0],
                "vy_body_mps": [0.0, 0.0],
                "yaw_rate_radps": [0.1, 0.2],
            },
        },
        {
            "topic": "/env_00/lewm/episode_info",
            "canonical_topic": "/lewm/episode_info",
            "env_index": 0,
            "timestamp_ns": 100_000_000,
            "payload": {"header": _header(100_000_000), "episode_id": 3, "episode_step": 1},
        },
        _base_record(100_000_000, 0.0, 0.0, 0.0),
        {
            "topic": "/env_00/lewm/episode_info",
            "canonical_topic": "/lewm/episode_info",
            "env_index": 0,
            "timestamp_ns": 200_000_000,
            "payload": {"header": _header(200_000_000), "episode_id": 3, "episode_step": 2},
        },
        _base_record(200_000_000, 1.0, 0.0, 0.0),
        {
            "topic": "/env_00/lewm/go2/executed_command_block",
            "canonical_topic": "/lewm/go2/executed_command_block",
            "env_index": 0,
            "timestamp_ns": 300_000_000,
            "payload": {
                "header": _header(300_000_000),
                "sequence_id": 7,
                "command_dt_s": 0.1,
                "executed_vx_body_mps": [0.5, 0.6],
                "executed_vy_body_mps": [0.0, 0.0],
                "executed_yaw_rate_radps": [0.05, 0.06],
            },
        },
    ]

    poses, summary = pose_steps_from_message_records(records)

    assert summary.source_record_count == len(records)
    assert summary.base_state_count == 2
    assert summary.command_block_count == 1
    assert summary.executed_command_block_count == 1
    assert summary.missing_command_count == 0
    assert [pose.episode_step for pose in poses] == [1, 2]
    assert poses[0].last_command == pytest.approx((0.5, 0.0, 0.05))
    assert poses[1].last_command == pytest.approx((0.6, 0.0, 0.06))
