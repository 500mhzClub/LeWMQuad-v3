"""Prospective, source-level diagnostics using only explicit synthetic inputs.

No corpus discovery, checkpoint loading, optimizer steps, experiment stages,
Genesis initialization, or writes to experiment outputs. Tests named
``known_limitation`` deliberately reproduce a frozen implementation's behavior;
their passing is NOT evidence that the underlying scientific interface passed.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from lewm.interface_semantics import (
    InterfaceError,
    sensor_channel_summary,
    validate_causal_history,
    validate_command_match,
    validate_rigid_transform,
    validate_v03_manifest,
)
from scripts import run_physical_graph_edge_handoff_qualification_v1 as PHYSICAL
from scripts import run_non_greedy_local_subgoal_jepa_planning_v1 as LOCAL
from scripts import build_dev_v03_proprio_action_manifest_v1 as HISTORY
from scripts import dev_action_slew_reconstruction_v1 as SLEW
from scripts import eval_dev_proprio_factorial_v1 as METRIC
from scripts import render_replay_v03 as LEGACY
from lewm_genesis.camera_safety import camera_pose_from_base
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_batch


def manifest():
    return {
        "walls": [{"center_xyz_m": [1, 0, 0.5], "size_xyz_m": [0.2, 2, 1]}],
        "obstacles": [], "landmarks": [],
    }


def test_explicit_scene_schema_including_an_intentionally_empty_scene():
    assert validate_v03_manifest(manifest()) == 1
    assert validate_v03_manifest({"walls": [], "obstacles": [], "landmarks": []}) == 0


@pytest.mark.parametrize("bad", [
    {}, {"objects": []}, {**manifest(), "objects": []},
    {**manifest(), "walls": [{"center_xyz_m": [1, 0, 0], "size_xyz_m": [0, 1, 1]}]},
    {**manifest(), "walls": [{"center_xyz_m": [float("nan"), 0, 0], "size_xyz_m": [1, 1, 1]}]},
    {**manifest(), "walls": [{"center_xyz_m": [1, 0], "size_xyz_m": [1, 1, 1]}]},
])
def test_scene_guard_rejects_ambiguous_or_invalid_geometry(bad):
    with pytest.raises(InterfaceError):
        validate_v03_manifest(bad)


class RecordingScene:
    def __init__(self, **kwargs):
        self.entities = []

    def add_entity(self, morph, **kwargs):
        self.entities.append((morph, kwargs))

    def add_camera(self, **kwargs):
        return kwargs

    def build(self, **kwargs):
        assert kwargs == {"n_envs": 1}


def record_legacy_scene(spec):
    gs = SimpleNamespace(
        Scene=RecordingScene,
        morphs=SimpleNamespace(
            Plane=lambda: {"kind": "plane"},
            Box=lambda **kwargs: {"kind": "box", **kwargs},
        ),
        surfaces=SimpleNamespace(Default=lambda **kwargs: kwargs),
    )
    return LEGACY.build_scene(gs, spec, fov=78.323, near=0.05, far=200,
                              res=(224, 224), textures=False)[0]


def test_legacy_geometry_submission_and_material_invariance():
    first = manifest()
    second = copy.deepcopy(first)
    second["walls"][0]["material_id"] = "changed-appearance"
    second["visual_randomization"] = {"material_overrides": [
        {"material_id": "changed-appearance", "rgba": [1, 0, 0, 1]}]}
    scene_a, scene_b = record_legacy_scene(first), record_legacy_scene(second)
    assert len(scene_a.entities) == 2  # plane plus the declared wall
    assert scene_a.entities[1][0]["pos"] == (1, 0, 0.5)
    assert [m for m, _ in scene_a.entities] == [m for m, _ in scene_b.entities]
    assert scene_a.entities[1][1] != scene_b.entities[1][1]
    # This proves submitted box geometry, not rendered texture correctness.


def test_known_limitation_legacy_builder_silently_ignores_objects_schema():
    wrong = {"objects": manifest()["walls"]}
    assert len(record_legacy_scene(wrong).entities) == 1  # floor only
    with pytest.raises(InterfaceError, match="objects schema"):
        validate_v03_manifest(wrong)


def test_analytic_wall_intervention_changes_rgb_and_collision_consistently():
    clear = {"walls": []}
    blocked = {"walls": [LOCAL.Rectangle(1, 0, 0.2, 2).as_dict()]}
    rgb_clear = LOCAL.render_rgb(clear, (0, 0, 0))
    rgb_blocked = LOCAL.render_rgb(blocked, (0, 0, 0))
    assert rgb_blocked.shape == (168, 224, 3)
    assert rgb_blocked.dtype == np.uint8
    assert not np.array_equal(rgb_clear, rgb_blocked)
    assert not LOCAL.pose_collides(clear, 1, 0)
    assert LOCAL.pose_collides(blocked, 1, 0)
    assert np.array_equal(rgb_clear, LOCAL.render_rgb(blocked, (0, 0, math.pi)))


def test_camera_mount_rotates_with_body_using_xyzw_quaternion():
    pose = camera_pose_from_base(
        [2, 3, 0.4], [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        mount_xyz_body=[0.326, 0, 0.043], mount_rpy_body=[0, 0, 0],
    )
    np.testing.assert_allclose(pose.position, [2, 3.326, 0.443], atol=1e-6)
    np.testing.assert_allclose(pose.forward, [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(pose.up, [0, 0, 1], atol=1e-6)


def physical_camera_stub(native):
    class Camera:
        def set_pose(self, **kwargs):
            self.pose = kwargs

        def render(self):
            return native

    # Bypass construction: the real constructor loads policy and simulator.
    session = PHYSICAL._GenesisPhysicalSession.__new__(PHYSICAL._GenesisPhysicalSession)
    session.ctx = SimpleNamespace(
        runner=SimpleNamespace(_as_np=np.asarray, _extract_rgb=lambda value: value),
        build=SimpleNamespace(
            robot=SimpleNamespace(get_pos=lambda: [0, 0, 0.4],
                                  get_quat=lambda: [1, 0, 0, 0]),
            camera=Camera(),
        ),
        pack=SimpleNamespace(
            camera=SimpleNamespace(xyz_body_m=(0.326, 0, 0.043), rpy_body_rad=(0, 0, 0)),
            camera_extrinsic_jitter=None, camera_constraints={}, static_objects=(),
        ),
    )
    return session


def test_known_limitation_physical_camera_basis_is_reflected_not_se3():
    native = np.zeros((480, 640, 3), dtype=np.uint8)
    native[:, :, 0] = 73
    session = physical_camera_stub(native)
    rgb, transform = session.render_rgb_and_transform()
    assert rgb.shape == (168, 224, 3) and rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous and np.all(rgb[:, :, 0] == 73)
    np.testing.assert_allclose(transform[:3, 3], [0.326, 0, 0.443], atol=1e-6)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(-1)
    with pytest.raises(InterfaceError, match="reflection"):
        validate_rigid_transform(transform)
    # An explicitly labeled FLU conversion has determinant +1. This is NOT
    # installed in the frozen runner; existing serialized bases remain intact.
    flu = transform @ np.diag([1, -1, 1, 1])
    validate_rigid_transform(flu)


def test_physical_camera_rejects_unexpected_native_resolution():
    with pytest.raises(PHYSICAL.ExperimentError, match="shape drift"):
        physical_camera_stub(np.zeros((224, 224, 3), dtype=np.uint8)).render_rgb_and_transform()


@pytest.mark.parametrize("bad", [np.eye(3), np.diag([2, 1, 1, 1]),
                                    np.diag([1, 1, 1, 0]), np.full((4, 4), np.nan)])
def test_transform_guard_rejects_non_rigid_arrays(bad):
    with pytest.raises(InterfaceError):
        validate_rigid_transform(bad)


def test_causal_history_has_explicit_time_and_reset_identity():
    validate_causal_history([100, 200, 300], [(0, 1, 0)] * 3,
                            image_timestamp_ns=300, image_episode=(0, 1, 0),
                            expected_period_ns=100)


@pytest.mark.parametrize("times,episodes,period", [
    ([], [], 100), ([100, 200], [(0, 1, 0)], 100),
    ([100, 200], [(0, 1, 0), (0, 1, 1)], 100),
    ([100, 200], [(0, 1, 0), (1, 1, 0)], 100),
    ([200, 100], [(0, 1, 0)] * 2, 100),
    ([100, 100], [(0, 1, 0)] * 2, 100),
    ([100, 400], [(0, 1, 0)] * 2, None),
    ([100, 300], [(0, 1, 0)] * 2, 100),
    ([100.0, 200.0], [(0, 1, 0)] * 2, 100),
    ([True, 200], [(0, 1, 0)] * 2, 100),
])
def test_history_guard_rejects_future_reset_gap_and_unordered_samples(times, episodes, period):
    with pytest.raises(InterfaceError):
        validate_causal_history(times, episodes, image_timestamp_ns=300,
                                image_episode=(0, 1, 0), expected_period_ns=period)


@pytest.mark.parametrize("primitive", SLEW.PRIMITIVES)
@pytest.mark.parametrize("previous", [(0, 0, 0), (0.3, 0, -0.45), (-0.2, 0, 0.45)])
def test_supported_action_reconstruction_matches_active_limiter(primitive, previous):
    platform = yaml.safe_load((Path(__file__).resolve().parents[2] /
                               "config/go2_platform_manifest.yaml").read_text())
    limits = SafetyLimits.from_manifest(platform)
    requested = np.asarray([[SLEW.COMMANDS[primitive]] * SLEW.TICKS])
    actual, clipped = apply_safety_limits_batch(requested, np.asarray([previous]), limits)
    expected, final = SLEW.reconstruct_block(primitive, previous)
    validate_command_match(expected, actual[0])
    np.testing.assert_allclose(final, actual[0, -1], atol=1e-6)
    assert clipped.shape == (1,) and clipped.dtype == np.bool_
    assert len(SLEW.flatten(expected)) == 10


@pytest.mark.parametrize("observed", [[], [[0, 0, 0]], [[0, 0]] * 5,
                                         [[0, 0, float("nan")]] * 5, [[1, 0, 0]] * 5])
def test_command_guard_rejects_missing_truncated_nonfinite_or_changed_tapes(observed):
    with pytest.raises(InterfaceError):
        validate_command_match([[0, 0, 0]] * 5, observed)


def test_action_state_carries_between_blocks_and_resets_explicitly():
    _, previous = SLEW.reconstruct_block("yaw_left", SLEW.RESET_APPLIED)
    carried, _ = SLEW.reconstruct_block("yaw_right", previous)
    reset, _ = SLEW.reconstruct_block("yaw_right", SLEW.RESET_APPLIED)
    assert carried[0][2] == pytest.approx(0.1)
    assert reset[0][2] == pytest.approx(-0.35)
    with pytest.raises(InterfaceError):
        validate_command_match(carried, reset)


def test_historical_action_interface_rejects_unsupported_lateral_state():
    with pytest.raises(SLEW.LateralMotionRejected):
        SLEW.reconstruct_block("forward_slow", [0, 0.1, 0])


@pytest.mark.parametrize("truncated", [[], [[0, 0, 0]], [[0, 0]] * 5])
def test_known_limitation_historical_zip_verifier_accepts_truncated_logs(truncated):
    expected = [[0, 0, 0]] * 5
    verified, mismatched = HISTORY._verify({(0, 0): expected}, {(0, 0): truncated})
    assert verified == {(0, 0)} and mismatched == 0
    with pytest.raises(InterfaceError):
        validate_command_match(expected, truncated)


def history_fixture(monkeypatch, *, reset_step=None, omit_step=None):
    table = {}
    for step in range(1, 36):
        table[(0, step)] = {
            "episode": (0, 1, int(reset_step is not None and step >= reset_step)),
            "requested": [0.2 if step < 16 else 0.3, 0, 0], "roll": 0, "pitch": 0,
            "gyro": [float(step), 0, 0], "q": [0.0] * 12, "dq": [0.0] * 12,
            "timestamp_ns": step * 100_000_000,
        }
    _, logged = HISTORY._reconstruct_applied(table, [0], 35)
    if omit_step is not None:
        del table[(0, omit_step)]
    monkeypatch.setattr(HISTORY, "_scene_paths", lambda scene: (None, None))
    monkeypatch.setattr(HISTORY, "_load_frames", lambda path: table)
    monkeypatch.setattr(HISTORY, "_load_logged_blocks", lambda path: logged)
    row = {"frames": [{"offset": 0, "frame_index": 15 * HISTORY.FRAMES_PER_TIMESTEP}],
           "env_index": 0, "episode_id": 1, "reset_count": 0, "pair_sha256": "synthetic",
           "role": "synthetic-development", "scene": "synthetic", "family": "fixture",
           "t": 0, "primitive": "forward_slow"}
    return HISTORY.build_scene(("synthetic", [row]))


def test_real_history_assembler_uses_past_sensor_order_and_previous_control(monkeypatch):
    _, rows, drops, _, _ = history_fixture(monkeypatch)
    assert not drops and len(rows) == 1
    row = rows[0]
    assert row["proprio_steps"] == list(range(2, 17))
    assert np.shape(row["proprio"]) == (15, 30)
    assert np.shape(row["control"]) == (15, 2)
    assert [features[3] for features in row["proprio"]] == list(range(2, 17))
    np.testing.assert_allclose(row["proprio"][0][:3], [0, 0, 0])
    assert all(command == [0.2, 0.0] for command in row["control"])
    # At the image's step the new command is 0.3, but control still ends at
    # applied[step-1] = 0.2. Future action is a separate tensor.
    assert row["action_blocks"][0][0] == pytest.approx(0.3)
    assert row["action_block_indices"] == [3, 4, 5, 6]
    assert all(len(block) == 10 for block in row["action_blocks"])
    validate_causal_history(row["proprio_timestamps_ns"], [(0, 1, 0)] * 15,
                            image_timestamp_ns=row["image_timestamp_ns"],
                            image_episode=(0, 1, 0), expected_period_ns=100_000_000)


def test_real_history_assembler_rejects_reset_crossing(monkeypatch):
    _, rows, drops, _, _ = history_fixture(monkeypatch, reset_step=10)
    assert not rows and drops["proprio_history_crosses_reset"] == 1


def test_real_history_assembler_rejects_absent_sample(monkeypatch):
    _, rows, drops, _, _ = history_fixture(monkeypatch, omit_step=7)
    assert not rows and drops["proprio_history_absent"] == 1


def test_real_history_assembler_truncates_future_actions_at_reset(monkeypatch):
    _, rows, drops, _, _ = history_fixture(monkeypatch, reset_step=21)
    assert not drops and rows[0]["action_block_indices"] == [3]


def test_validity_distinguishes_missing_zero_stationary_and_nonfinite_sensors():
    values = [[0, 0, 1, np.nan, 4], [0, 0, 2, 3, 4]]
    valid = [[False, True, True, True, False], [False, True, True, True, True]]
    rows = sensor_channel_summary(values, valid, ["missing", "stationary", "changing", "bad", "single"])
    assert [r["status"] for r in rows] == [
        "UNAVAILABLE", "CONSTANT", "VARIABLE", "NONFINITE_VALID", "INSUFFICIENT_SAMPLES"]
    assert [r["valid_samples"] for r in rows] == [0, 2, 2, 2, 1]


@pytest.mark.parametrize("values,valid,names", [
    ([[0]], [[1]], ["q"]), ([[0, 0]], [[True, True]], ["q", "q"]),
    ([[0, 0]], [[True]], ["q", "dq"]), ([[0]], [[True]], []),
])
def test_sensor_guard_requires_explicit_channel_validity(values, valid, names):
    with pytest.raises(InterfaceError):
        sensor_channel_summary(values, valid, names)


def test_target_world_body_roundtrip_and_left_positive_convention():
    body = [2, 3, math.pi / 2]
    world = [2, 4, math.pi]
    relative = PHYSICAL.body_from_world(world, body)
    np.testing.assert_allclose(relative, [1, 0, math.pi / 2], atol=1e-6)
    recovered = PHYSICAL.world_from_body(relative, body)
    np.testing.assert_allclose(recovered[:2], world[:2], atol=1e-6)
    assert math.atan2(math.sin(recovered[2] - world[2]),
                      math.cos(recovered[2] - world[2])) == pytest.approx(0, abs=1e-6)


def test_ranker_input_heading_is_target_bearing_not_arrival_heading():
    calls = []

    def stub(tokens, relative, **kwargs):
        calls.append((relative, kwargs))
        return {"candidate_names": PHYSICAL.CANDIDATE_IDS,
                "candidate_scores": [0.0] * len(PHYSICAL.CANDIDATE_IDS)}

    ranker = PHYSICAL._FrozenPhysicalRanker.__new__(PHYSICAL._FrozenPhysicalRanker)
    ranker._ranker = stub
    for heading in (-math.pi / 2, math.pi / 2):
        ranker.score(None, {"dx_m": 1, "dy_m": 1, "relative_heading_rad": heading},
                     [0, 0, 0], [[0, 0]] * 15)
    assert calls[0] == calls[1]
    assert calls[0][0] == [1, 1, math.pi / 4]


@pytest.mark.parametrize("changed,value", [
    ("oracle_admissible", False), ("entered_correct_edge", False),
    ("successor_viable", False), ("positive_port_progress", False),
    ("physics_contact", True), ("stuck", True),
])
def test_correct_handoff_requires_all_execution_conditions(changed, value):
    row = {"oracle_admissible": True, "entered_correct_edge": True,
           "successor_viable": True, "positive_port_progress": True,
           "physics_contact": False, "stuck": False}
    assert PHYSICAL._correct_candidate(row)
    assert not PHYSICAL._correct_candidate({**row, changed: value})


def test_known_limitation_occupied_iou_cannot_distinguish_perfect_from_inverted():
    target = torch.eye(4, dtype=torch.float32)[None]
    mask = torch.ones((1, 4), dtype=torch.bool)
    perfect = METRIC.occupied_metrics(target, target, mask)
    inverted = METRIC.occupied_metrics(-target, target, mask)
    constant = METRIC.occupied_metrics(torch.ones_like(target), target, mask)
    assert perfect["occupied_iou"] == inverted["occupied_iou"] == constant["occupied_iou"] == 1
    # Positive control: directional fidelity does discriminate these inputs;
    # it is a representation metric, not a navigation or occupancy metric.
    good = METRIC.row_scores(F.cosine_similarity(target, target, dim=-1), mask)[0]
    bad = METRIC.row_scores(F.cosine_similarity(-target, target, dim=-1), mask)[0]
    uninformative = METRIC.row_scores(F.cosine_similarity(torch.ones_like(target), target, dim=-1), mask)[0]
    assert good == pytest.approx(1) and bad == pytest.approx(-1)
    assert bad < uninformative < good
