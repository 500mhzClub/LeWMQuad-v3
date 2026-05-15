"""Unit tests for ``lewm_genesis.lewm_contract``.

These exercise the lifted logic without requiring ROS or Genesis. The parity
target is the behavior of the LeWM ROS nodes; a recorded-fixture parity check
is a follow-up task once a Gazebo audit-oracle fixture exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lewm_genesis.lewm_contract import (
    BaseStateRecord,
    EpisodeState,
    LEWM_FOOT_ORDER,
    PrimitiveRegistry,
    SafetyLimits,
    apply_safety_limits_batch,
    apply_safety_limits_single,
    base_state_from_genesis,
    expand_primitive_to_block,
    foot_contacts_record,
    make_episode_states,
    quat_to_rpy,
    reconstruct_executed_block,
    rotate_body_to_world,
    sample_command_tape,
)


# Mirrors the platform manifest values at config/go2_platform_manifest.yaml
DEFAULT_LIMITS = SafetyLimits(
    min_vx_mps=-0.3,
    max_vx_mps=0.3,
    min_vy_mps=-0.25,
    max_vy_mps=0.25,
    max_yaw_rate_radps=0.5,
    max_delta_vx_mps=0.25,
    max_delta_vy_mps=0.25,
    max_delta_yaw_rate_radps=0.35,
)


# ---------------------------------------------------------------------------
# Safety clipping
# ---------------------------------------------------------------------------


def test_absolute_clipping_caps_to_manifest_limits():
    requested = np.array([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 5.0]]], dtype=np.float32)
    previous = np.zeros((1, 3), dtype=np.float32)
    executed, clipped = apply_safety_limits_batch(
        requested, previous, DEFAULT_LIMITS, enforce_rate_limits=False
    )
    assert clipped[0]
    assert executed[0, 0, 0] == pytest.approx(0.3)
    assert executed[0, 1, 1] == pytest.approx(-0.25)
    assert executed[0, 2, 2] == pytest.approx(0.5)


def test_delta_clipping_limits_per_tick_change():
    # Step from (0,0,0) → request (0.3, 0, 0). Absolute is fine; rate is 0.25.
    requested = np.array([[[0.3, 0.0, 0.0]]], dtype=np.float32)
    previous = np.zeros((1, 3), dtype=np.float32)
    executed, clipped = apply_safety_limits_batch(
        requested, previous, DEFAULT_LIMITS, enforce_rate_limits=True
    )
    assert clipped[0]
    assert executed[0, 0, 0] == pytest.approx(0.25)


def test_rate_limit_disabled_passes_through_absolute_bounded_step():
    requested = np.array([[[0.3, 0.0, 0.0]]], dtype=np.float32)
    previous = np.zeros((1, 3), dtype=np.float32)
    executed, clipped = apply_safety_limits_batch(
        requested, previous, DEFAULT_LIMITS, enforce_rate_limits=False
    )
    assert not clipped[0]
    assert executed[0, 0, 0] == pytest.approx(0.3)


def test_batch_matches_single_env_loop():
    rng = np.random.default_rng(42)
    n_envs, T = 8, 5
    requested = rng.uniform(-0.5, 0.5, size=(n_envs, T, 3)).astype(np.float32)
    previous = rng.uniform(-0.1, 0.1, size=(n_envs, 3)).astype(np.float32)

    batched_exec, batched_clip = apply_safety_limits_batch(
        requested, previous, DEFAULT_LIMITS
    )
    for env_idx in range(n_envs):
        env_exec, env_clip = apply_safety_limits_single(
            [tuple(r) for r in requested[env_idx]],
            tuple(previous[env_idx]),
            DEFAULT_LIMITS,
        )
        np.testing.assert_allclose(batched_exec[env_idx], np.asarray(env_exec))
        assert batched_clip[env_idx] == env_clip


def test_clipped_flag_false_when_command_already_safe():
    requested = np.array([[[0.1, 0.0, 0.0], [0.15, 0.0, 0.0]]], dtype=np.float32)
    previous = np.zeros((1, 3), dtype=np.float32)
    _, clipped = apply_safety_limits_batch(requested, previous, DEFAULT_LIMITS)
    assert not clipped[0]


# ---------------------------------------------------------------------------
# Quaternion math
# ---------------------------------------------------------------------------


def test_quat_to_rpy_identity_is_zero():
    rpy = quat_to_rpy(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(rpy, np.zeros(3), atol=1e-7)


def test_quat_to_rpy_yaw_90_deg():
    # 90 deg yaw quaternion: (0, 0, sin(pi/4), cos(pi/4))
    q = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    rpy = quat_to_rpy(q)
    assert rpy[0] == pytest.approx(0.0, abs=1e-6)
    assert rpy[1] == pytest.approx(0.0, abs=1e-6)
    assert rpy[2] == pytest.approx(math.pi / 2, abs=1e-6)


def test_quat_to_rpy_batched():
    q = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        ]
    )
    rpy = quat_to_rpy(q)
    assert rpy.shape == (2, 3)
    np.testing.assert_allclose(rpy[0], np.zeros(3), atol=1e-7)
    assert rpy[1, 2] == pytest.approx(math.pi / 2, abs=1e-6)


def test_rotate_body_to_world_identity_passthrough():
    v = np.array([1.0, 2.0, 3.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    out = rotate_body_to_world(v, q)
    np.testing.assert_allclose(out, v, atol=1e-6)


def test_rotate_body_to_world_yaw_90_deg():
    # Body-forward (1,0,0) under +90 deg yaw should become world (0,1,0).
    v = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    out = rotate_body_to_world(v, q)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_rotate_body_to_world_batched_broadcasts():
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        ]
    )
    out = rotate_body_to_world(v, q)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(out[1], np.array([-1.0, 0.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# BaseState construction
# ---------------------------------------------------------------------------


def test_base_state_from_genesis_populates_world_twist_under_yaw():
    pos = np.array([0.0, 0.0, 0.4])
    q = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    body_lin = np.array([0.3, 0.0, 0.0])
    body_ang = np.array([0.0, 0.0, 0.1])
    state: BaseStateRecord = base_state_from_genesis(pos, q, body_lin, body_ang)
    assert state.twist_world_linear[0] == pytest.approx(0.0, abs=1e-6)
    assert state.twist_world_linear[1] == pytest.approx(0.3, abs=1e-6)
    assert state.yaw_rad == pytest.approx(math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# Foot contacts
# ---------------------------------------------------------------------------


def test_foot_contacts_record_preserves_lewm_order():
    rec = foot_contacts_record(
        (True, False, True, False), force_n_in_lewm_order=(1.0, 2.0, 3.0, 4.0)
    )
    assert rec.fl_contact and rec.rl_contact
    assert not rec.fr_contact and not rec.rr_contact
    assert rec.fl_force_n == 1.0
    assert rec.fr_force_n == 2.0
    assert rec.rl_force_n == 3.0
    assert rec.rr_force_n == 4.0


def test_lewm_foot_order_constant():
    assert LEWM_FOOT_ORDER == ("fl", "fr", "rl", "rr")


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------


def test_episode_state_resets_increment_monotonically():
    state = EpisodeState(scene_id=7)
    assert state.episode_id == 0
    assert state.reset_count == 0
    event_a = state.reset(reason="fall")
    event_b = state.reset(reason="oob")
    assert event_a.episode_id == 1
    assert event_b.episode_id == 2
    assert event_a.reset_count == 1
    assert event_b.reset_count == 2
    assert state.episode_step == 0


def test_episode_step_advances_and_zeros_on_reset():
    state = EpisodeState()
    for _ in range(50):
        state.step()
    assert state.episode_step == 50
    state.reset(reason="manual")
    assert state.episode_step == 0


def test_episode_info_record_reflects_state():
    state = EpisodeState(scene_id=42, scene_family="medium_enclosed_maze", split="train")
    state.step()
    state.step()
    info = state.episode_info(stamp_ns=12345)
    assert info.scene_id == 42
    assert info.episode_step == 2
    assert info.scene_family == "medium_enclosed_maze"
    assert info.split == "train"
    assert info.stamp_ns == 12345


def test_make_episode_states_are_independent():
    states = make_episode_states(4, scene_id=1)
    states[0].reset(reason="x")
    assert states[0].episode_id == 1
    assert states[1].episode_id == 0


# ---------------------------------------------------------------------------
# Primitive registry + command tape
# ---------------------------------------------------------------------------


def test_primitive_registry_loads_default_yaml(tmp_path):
    # Use the real config to lock in the schema expectations.
    registry_path = (
        # repo root is two levels up from this test file's parent package
        # (lewm_genesis/lewm_genesis/tests/) → ../../../config/...
        # Use a relative resolution from this file to avoid hard-coding.
        # We assume the test is run from a checkout.
        __file__
    )
    from pathlib import Path

    repo_root = Path(registry_path).resolve().parents[3]
    yaml_path = repo_root / "config" / "go2_primitive_registry.yaml"
    if not yaml_path.exists():
        pytest.skip(f"primitive registry not present at {yaml_path}")
    registry = PrimitiveRegistry.from_yaml(yaml_path)
    assert registry.block_size == 5
    assert registry.command_dt_s == pytest.approx(0.10)
    assert "forward_slow" in registry.primitives
    trainable = registry.trainable_velocity_names()
    assert "forward_slow" in trainable
    # lateral_left has train: false in the registry; should be excluded.
    assert "lateral_left" not in trainable


def test_expand_primitive_to_block_uses_command_values():
    registry = PrimitiveRegistry(
        block_size=5,
        command_dt_s=0.1,
        command_order=("vx_body_mps", "vy_body_mps", "yaw_rate_radps"),
        primitives={
            "forward": {
                "type": "velocity_block",
                "train": True,
                "command": {"vx_body_mps": 0.2, "vy_body_mps": 0.0, "yaw_rate_radps": 0.0},
            }
        },
        defaults={},
    )
    block = expand_primitive_to_block(registry, "forward")
    assert block.shape == (5, 3)
    np.testing.assert_allclose(block[:, 0], 0.2)
    np.testing.assert_allclose(block[:, 1:], 0.0)


def test_expand_primitive_rejects_non_velocity_blocks():
    registry = PrimitiveRegistry(
        block_size=5,
        command_dt_s=0.1,
        command_order=("vx_body_mps", "vy_body_mps", "yaw_rate_radps"),
        primitives={
            "recovery": {
                "type": "mode_event",
                "train": True,
                "event_name": "recovery_stand",
            }
        },
        defaults={},
    )
    with pytest.raises(ValueError):
        expand_primitive_to_block(registry, "recovery")


def test_sample_command_tape_shapes_and_uses_allowed_names():
    registry = PrimitiveRegistry(
        block_size=5,
        command_dt_s=0.1,
        command_order=("vx_body_mps", "vy_body_mps", "yaw_rate_radps"),
        primitives={
            "forward": {
                "type": "velocity_block",
                "train": True,
                "command": {"vx_body_mps": 0.2, "vy_body_mps": 0.0, "yaw_rate_radps": 0.0},
            },
            "back": {
                "type": "velocity_block",
                "train": True,
                "command": {"vx_body_mps": -0.2, "vy_body_mps": 0.0, "yaw_rate_radps": 0.0},
            },
        },
        defaults={"train": True},
    )
    rng = np.random.default_rng(0)
    tape, names = sample_command_tape(registry, n_envs=3, n_blocks=4, rng=rng)
    assert tape.shape == (3, 4, 5, 3)
    assert len(names) == 3 and all(len(row) == 4 for row in names)
    flat_names = {n for row in names for n in row}
    assert flat_names.issubset({"forward", "back"})


# ---------------------------------------------------------------------------
# Reconstructed executed block
# ---------------------------------------------------------------------------


def test_reconstruct_executed_block_round_trips_fields():
    requested = [(0.3, 0.0, 0.0)] * 5
    executed = [(0.25, 0.0, 0.0)] * 5
    rec = reconstruct_executed_block(
        requested,
        executed,
        sequence_id=7,
        primitive_name="forward_fast",
        command_dt_s=0.10,
        clipped=True,
    )
    assert rec.sequence_id == 7
    assert rec.block_size == 5
    assert rec.command_dt_s == pytest.approx(0.10)
    assert rec.primitive_name == "forward_fast"
    assert rec.requested_vx_body_mps == [0.3] * 5
    assert rec.executed_vx_body_mps == [0.25] * 5
    assert rec.clipped is True
    assert rec.backend_id == "genesis_tier_a"


def test_reconstruct_executed_block_rejects_length_mismatch():
    with pytest.raises(ValueError):
        reconstruct_executed_block(
            [(0.0, 0.0, 0.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            sequence_id=1,
            primitive_name="hold",
            command_dt_s=0.10,
            clipped=False,
        )
