"""End-to-end smoke test for ``rollout.RolloutRunner``.

Builds a real Genesis scene with the Genesis-bundled Go2 URDF, runs a few
command blocks with :class:`StancePolicy`, and verifies the resulting MCAP +
summary contain the expected per-env topics. Requires both Genesis
(``import genesis``) and the ROS 2 Jazzy overlay; skips cleanly otherwise.

Heavy enough that we keep the parameters small: 2 envs, 3 command blocks,
fall threshold disabled so the run completes without resets even with the
placeholder stance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

try:
    import genesis  # noqa: F401

    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False

try:
    import rosbag2_py  # noqa: F401
    import lewm_go2_control.msg  # noqa: F401

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (GENESIS_AVAILABLE and ROS_AVAILABLE),
    reason="Genesis + ROS 2 Jazzy overlay + lewm_go2_control install required",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_CORPUS = REPO_ROOT / ".generated" / "scene_corpus" / "acceptance"
PLATFORM_MANIFEST = REPO_ROOT / "config" / "go2_platform_manifest.yaml"
PRIMITIVE_REGISTRY = REPO_ROOT / "config" / "go2_primitive_registry.yaml"


@pytest.mark.slow
def test_rollout_smoke_two_envs_three_blocks(tmp_path):
    if not ACCEPTANCE_CORPUS.is_dir():
        pytest.skip("acceptance corpus not present")
    if not PLATFORM_MANIFEST.is_file():
        pytest.skip("platform manifest not present")

    from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
    from lewm_genesis.mcap_writer import MCAPSceneWriter
    from lewm_genesis.rollout import (
        DEFAULT_GO2_LEG_DOF_INDICES,
        RolloutConfig,
        RolloutRunner,
        StancePolicy,
    )
    from lewm_genesis.scene_builder import build_scene_from_pack
    from lewm_genesis.scene_loader import (
        find_scene_dirs,
        load_platform_manifest,
        load_scene_pack,
    )

    # Pick the first available scene.
    scenes = find_scene_dirs(ACCEPTANCE_CORPUS)
    if not scenes:
        pytest.skip("no scenes in acceptance corpus")
    platform = load_platform_manifest(PLATFORM_MANIFEST)
    # Override the URDF resolution to use the Genesis-bundled Go2 URDF for
    # the smoke test (the workspace xacro requires xacro expansion).
    import genesis  # for sys.modules / path

    bundled = (
        Path(genesis.__file__).parent / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"
    )
    if not bundled.is_file():
        pytest.skip(f"genesis-bundled go2.urdf not found at {bundled}")
    # Inject a robot.genesis_urdf override so resolve_go2_urdf picks the bundled file.
    platform.setdefault("robot", {})["genesis_urdf"] = str(bundled.relative_to(REPO_ROOT)) if bundled.is_relative_to(REPO_ROOT) else str(bundled)
    workspace_root = "/" if Path(platform["robot"]["genesis_urdf"]).is_absolute() else REPO_ROOT

    pack = load_scene_pack(
        scenes[0],
        platform_manifest=platform,
        workspace_root=workspace_root,
    )

    safety = SafetyLimits.from_manifest(platform)
    registry = PrimitiveRegistry.from_yaml(PRIMITIVE_REGISTRY)
    build = build_scene_from_pack(pack, n_envs=2, backend="cpu", show_viewer=False)

    config = RolloutConfig(
        n_blocks=3,
        # Disable falls so a stance-only smoke run completes without resets.
        fall_z_threshold_m=0.0,
        rgb_capture_per_block=False,  # skip RGB to keep the smoke fast
        seed=7,
        log_progress_every_blocks=0,
    )
    runner = RolloutRunner(build, StancePolicy(), registry, safety, config=config)

    out_dir = tmp_path / "raw_rollouts"
    writer = MCAPSceneWriter(pack, out_dir, n_envs=2)
    try:
        stats = runner.run(writer)
    finally:
        writer.close()

    assert stats["n_blocks"] == 3
    assert stats["command_ticks"] == 3 * 5
    assert stats["final_sim_time_s"] > 0

    summary = json.loads((out_dir / pack.scene_id / "summary.json").read_text())
    counts = summary["stats"]["per_topic_counts"]
    # Each env should have 3 command_block requests, 3 executed_command_blocks,
    # 15 base_state ticks, 15 episode_info ticks, 15 foot_contacts ticks.
    assert counts.get("/env_00/lewm/go2/command_block", 0) == 3
    assert counts.get("/env_01/lewm/go2/command_block", 0) == 3
    assert counts.get("/env_00/lewm/go2/executed_command_block", 0) == 3
    assert counts.get("/env_00/lewm/go2/base_state", 0) == 15
    assert counts.get("/env_00/lewm/go2/foot_contacts", 0) == 15
    assert counts.get("/env_00/lewm/episode_info", 0) == 15
    assert counts.get("/clock", 0) == 15


def test_stance_policy_returns_n_envs_by_12():
    from lewm_genesis.rollout import DEFAULT_GO2_STANCE_RAD, StancePolicy
    import numpy as np

    policy = StancePolicy()
    obs = {"base_pos_world": np.zeros((5, 3), dtype=np.float32)}
    out = policy.act(obs)
    assert out.shape == (5, 12)
    np.testing.assert_allclose(out[0], DEFAULT_GO2_STANCE_RAD)


def test_rollout_config_defaults_match_data_spec():
    from lewm_genesis.rollout import (
        DEFAULT_GO2_LEG_DOF_INDICES,
        RolloutConfig,
    )

    config = RolloutConfig()
    # 200 blocks × 5 ticks = 1000 raw steps per env, within data spec §7's 800–1200 range.
    assert config.n_blocks == 200
    # 12 leg DOFs.
    assert len(config.leg_dof_indices) == 12
    # First six DOFs are the free root joint; legs start at 6.
    assert min(config.leg_dof_indices) >= 6


def test_ppo_policy_validation_uses_robot_local_dofs():
    from lewm_genesis.rollout import (
        GENESIS_GO2_POLICY_DOF_INDICES,
        GenesisGo2PPOPolicy,
    )

    class Joint:
        def __init__(self, name, dof):
            self.name = name
            self.dofs_idx = dof

    class Robot:
        dof_start = 312

        def __init__(self):
            names = (
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            )
            self.joints = [
                Joint(name, self.dof_start + local_dof)
                for name, local_dof in zip(names, GENESIS_GO2_POLICY_DOF_INDICES)
            ]

    policy = GenesisGo2PPOPolicy.__new__(GenesisGo2PPOPolicy)
    policy.policy_joint_names = tuple(joint.name for joint in Robot().joints)
    policy.policy_dof_indices = GENESIS_GO2_POLICY_DOF_INDICES
    policy.validate_rollout_robot(Robot())
