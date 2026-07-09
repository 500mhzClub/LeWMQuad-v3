from __future__ import annotations

import json
import math
from pathlib import Path
import pickle

import pytest

from lewm.planning.swept_footprint_calibration import (
    CollisionKinematicModel,
    RolloutDataset,
    StateSample,
    build_calibration_report,
    load_open_field_rollout,
    load_policy_nominal_stance,
    sha256_file,
)


SYNTHETIC_URDF = """<?xml version="1.0"?>
<robot name="synthetic">
  <link name="base">
    <collision>
      <geometry><box size="2.0 1.0 0.4"/></geometry>
    </collision>
  </link>
  <link name="arm">
    <collision>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.2"/></geometry>
    </collision>
  </link>
  <joint name="swing_joint" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def _write_model(tmp_path: Path) -> CollisionKinematicModel:
    path = tmp_path / "robot.urdf"
    path.write_text(SYNTHETIC_URDF, encoding="utf-8")
    return CollisionKinematicModel.from_urdf(path)


def _record(
    topic: str,
    timestamp_ns: int,
    payload: dict,
    *,
    env_index: int = 0,
) -> dict:
    return {
        "canonical_topic": topic,
        "env_index": env_index,
        "timestamp_ns": timestamp_ns,
        "payload": payload,
    }


def _write_open_rollout(
    tmp_path: Path,
    *,
    split: str = "train",
    family: str = "open_obstacle_field",
) -> Path:
    raw_dir = tmp_path / "raw"
    bag_dir = tmp_path / "bag"
    raw_dir.mkdir(parents=True)
    bag_dir.mkdir(parents=True)
    messages = raw_dir / "messages.jsonl"
    (raw_dir / "summary.json").write_text(
        json.dumps({"source_bag": str(bag_dir)}),
        encoding="utf-8",
    )
    (bag_dir / "summary.json").write_text(
        json.dumps(
            {
                "split": split,
                "family": family,
                "scene_id": "open_fixture",
                "n_envs": 1,
                "extra": {},
            }
        ),
        encoding="utf-8",
    )
    joint_payload = {"name": ["swing_joint"], "position": [0.0]}
    base_payload = {
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "pose_world": {"position": {"z": 0.3}},
    }
    rows = [
        _record("/lewm/go2/reset_event", 0, {"reason": "spawn"}),
        _record(
            "/lewm/go2/command_block",
            0,
            {
                "block_size": 5,
                "command_dt_s": 0.1,
                "primitive_name": "forward",
                "command_source": "fixture",
                "sequence_id": 42,
            },
        ),
        _record(
            "/lewm/episode_info",
            100_000_000,
            {"split": split, "scene_family": family},
        ),
        _record("/joint_states", 100_000_000, joint_payload),
        _record("/lewm/go2/base_state", 100_000_000, base_payload),
        # This boundary sample belongs to the first block because the second
        # command is emitted after the final state at the same timestamp.
        _record("/joint_states", 500_000_000, joint_payload),
        _record("/lewm/go2/base_state", 500_000_000, base_payload),
        _record(
            "/lewm/go2/command_block",
            500_000_000,
            {
                "block_size": 5,
                "command_dt_s": 0.1,
                "primitive_name": "hold",
                "command_source": "fixture",
                "sequence_id": 99,
            },
        ),
        _record("/joint_states", 600_000_000, joint_payload),
        _record("/lewm/go2/base_state", 600_000_000, base_payload),
    ]
    messages.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return messages


def test_urdf_forward_kinematics_and_support_envelope(tmp_path: Path) -> None:
    model = _write_model(tmp_path)

    forward = model.envelope({"swing_joint": 0.0})
    left = model.envelope({"swing_joint": math.pi / 2.0})

    assert model.root_link == "base"
    assert model.actuated_joint_names == ("swing_joint",)
    assert model.description()["collision_shape_counts"] == {"box": 1, "sphere": 1}
    assert forward.forward_m == pytest.approx(1.2)
    assert forward.rear_m == pytest.approx(1.0)
    assert forward.left_m == pytest.approx(0.5)
    assert forward.radius_m == pytest.approx(1.2, abs=1e-5)
    assert left.forward_m == pytest.approx(1.0)
    assert left.left_m == pytest.approx(1.2)
    assert left.radius_m == pytest.approx(1.2, abs=1e-5)


def test_urdf_rejects_nonprimitive_collision_geometry(tmp_path: Path) -> None:
    path = tmp_path / "mesh.urdf"
    path.write_text(
        """<robot name="bad"><link name="base"><collision><geometry>
        <mesh filename="shape.dae"/>
        </geometry></collision></link></robot>""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported geometry"):
        CollisionKinematicModel.from_urdf(path)


def test_policy_nominal_stance_accepts_recorded_list_config(tmp_path: Path) -> None:
    config = tmp_path / "cfgs.pkl"
    with config.open("wb") as handle:
        pickle.dump(
            [
                {
                    "default_joint_angles": {
                        "joint_a": 0.8,
                        "joint_b": -1.5,
                    }
                },
                {},
            ],
            handle,
        )

    stance = load_policy_nominal_stance(
        config,
        required_joint_names=("joint_a", "joint_b"),
    )

    assert stance == {"joint_a": 0.8, "joint_b": -1.5}
    with pytest.raises(ValueError, match="missing default angle"):
        load_policy_nominal_stance(config, required_joint_names=("joint_c",))


def test_rollout_loader_preserves_stream_order_and_episode_local_blocks(
    tmp_path: Path,
) -> None:
    messages = _write_open_rollout(tmp_path)

    dataset = load_open_field_rollout(
        messages,
        required_joint_names=("swing_joint",),
        workspace_root=tmp_path,
    )

    assert [sample.primitive_name for sample in dataset.samples] == [
        "forward",
        "forward",
        "hold",
    ]
    assert [sample.block_phase_s for sample in dataset.samples] == pytest.approx(
        [0.1, 0.5, 0.1]
    )
    assert [sample.is_initial_block for sample in dataset.samples] == [
        True,
        True,
        False,
    ]
    assert dataset.metadata["split"] == "train"
    assert dataset.metadata["family"] == "open_obstacle_field"
    assert {artifact.role for artifact in dataset.artifacts} == {
        "rollout_messages",
        "raw_rollout_summary",
        "source_rollout_summary",
    }


@pytest.mark.parametrize(
    ("split", "family", "message"),
    [
        ("test_id", "open_obstacle_field", "train split"),
        ("train", "large_enclosed_maze", "open_obstacle_field"),
    ],
)
def test_rollout_loader_rejects_sealed_or_nonopen_sources(
    tmp_path: Path,
    split: str,
    family: str,
    message: str,
) -> None:
    messages = _write_open_rollout(tmp_path, split=split, family=family)

    with pytest.raises(ValueError, match=message):
        load_open_field_rollout(
            messages,
            required_joint_names=("swing_joint",),
            workspace_root=tmp_path,
        )


def test_report_is_deterministic_hashed_and_flags_missing_steady_state(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path)
    messages = _write_open_rollout(tmp_path / "rollout")
    dataset = load_open_field_rollout(
        messages,
        required_joint_names=model.actuated_joint_names,
        workspace_root=tmp_path,
    )

    kwargs = dict(
        nominal_joint_positions={"swing_joint": 0.0},
        datasets=(dataset,),
        required_primitives=("forward", "hold"),
        minimum_safety_margin_m=0.03,
        output_rounding_m=0.01,
        minimum_blocks_per_primitive=1,
        minimum_samples_per_primitive=1,
        minimum_noninitial_blocks_per_primitive=1,
    )
    first = build_calibration_report(model, **kwargs)
    second = build_calibration_report(model, **kwargs)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["coverage_gate"]["per_primitive"]["forward"]["pass"] is False
    assert first["coverage_gate"]["per_primitive"]["hold"]["pass"] is True
    assert first["additional_genesis_rollout"]["required"] is True
    assert first["additional_genesis_rollout"]["primitives_missing_coverage"] == [
        "forward"
    ]
    assert first["recommendation"]["static_configuration_space_radius_m"] == 1.23
    assert first["recommendation"]["action_probe"] == {
        "forward_m": 1.23,
        "rear_m": 1.03,
        "half_width_m": 0.53,
        "probe_margin_m": 0.03,
    }
    assert first["observed_maxima_provenance"]["forward"]["source"] == (
        "nominal_stance"
    )
    assert first["observed_maxima_provenance"]["left"]["source"] == "nominal_stance"
    urdf_artifact = next(
        artifact
        for artifact in first["source_artifacts"]
        if "genesis_collision_urdf" in artifact["roles"]
    )
    assert urdf_artifact["sha256"] == sha256_file(model.urdf_path)


def test_actual_genesis_urdf_nominal_envelope_when_asset_is_available() -> None:
    root = Path(__file__).resolve().parents[2]
    urdf = (
        root
        / ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages"
        / "genesis/assets/urdf/go2/urdf/go2.urdf"
    )
    cfg = root / "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl"
    if not urdf.is_file() or not cfg.is_file():
        pytest.skip("local Genesis Go2 calibration assets are unavailable")
    model = CollisionKinematicModel.from_urdf(urdf)
    stance = load_policy_nominal_stance(
        cfg,
        required_joint_names=model.actuated_joint_names,
    )

    envelope = model.envelope(stance)

    assert model.description()["collision_primitive_count"] == 27
    assert envelope.forward_m == pytest.approx(0.34, abs=1e-9)
    assert envelope.rear_m == pytest.approx(0.3836879715, abs=1e-8)
    assert envelope.left_m == pytest.approx(0.164, abs=1e-9)
    assert envelope.right_m == pytest.approx(0.164, abs=1e-9)
    assert envelope.radius_m == pytest.approx(0.4101953279, abs=1e-8)
