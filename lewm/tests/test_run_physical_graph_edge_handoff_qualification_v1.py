from __future__ import annotations

import hashlib
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as M
from scripts import run_physical_graph_edge_handoff_qualification_v1 as R


def _fake_graph(spec: dict) -> dict:
    session = R._GenesisPhysicalSession.__new__(R._GenesisPhysicalSession)
    session.spec = spec
    session.geometry = spec["geometry"]
    return session.graph()


def _fake_snapshot(spec: dict, trace: dict[str, np.ndarray]) -> dict:
    payload = R.canonical_bytes({
        "kind": "synthetic-test-only-serialized-snapshot",
        "candidate_spec_id": spec["candidate_spec_id"],
    })
    base_pose = np.asarray(trace["base_pose_world"][0], dtype=np.float64)
    arrays = {
        "base_pose_world": base_pose,
        "base_twist_world": np.zeros(6, dtype=np.float64),
        "joint_position": np.zeros(12, dtype=np.float64),
        "joint_velocity": np.zeros(12, dtype=np.float64),
        "controller_observation": np.zeros(45, dtype=np.float64),
        "policy_last_action": np.zeros(12, dtype=np.float64),
        "previous_policy_action": np.zeros(12, dtype=np.float64),
        "previous_applied_command": np.zeros(3, dtype=np.float64),
        "command_history": np.zeros((15, 3), dtype=np.float64),
        "control_history": np.zeros((15, 2), dtype=np.float64),
        "low_level_policy_state": np.zeros(12, dtype=np.float64),
        "camera_world_transform": np.eye(4, dtype=np.float64),
    }
    digest = lambda name: R.canonical_array_sha256(arrays[name])
    return {
        "payload_bytes": payload,
        **arrays,
        "serialized_solver_state_sha256": hashlib.sha256(b"solver" + payload).hexdigest(),
        "serialized_controller_state_sha256": hashlib.sha256(b"controller" + payload).hexdigest(),
        "serialized_rng_state_sha256": hashlib.sha256(b"rng" + payload).hexdigest(),
        "policy_last_action_sha256": digest("policy_last_action"),
        "capture_timestamp_s": 0.0,
        "controller_observation_sha256": digest("controller_observation"),
        "previous_policy_action_sha256": digest("previous_policy_action"),
        "torch_cpu_rng_state_sha256": hashlib.sha256(b"torch-cpu" + payload).hexdigest(),
        "torch_device_rng_state_sha256s": [],
        "torch_device_count": 0,
        "previous_applied_command_sha256": digest("previous_applied_command"),
        "command_history_sha256": digest("command_history"),
        "control_history_sha256": digest("control_history"),
        "low_level_policy_state_sha256": digest("low_level_policy_state"),
        "solver_field_inventory": ["synthetic.solver.field"],
        "controller_field_inventory": ["synthetic.controller.field"],
        "rng_field_inventory": ["python", "numpy", "torch", "torch_devices"],
    }


def _fake_physics_trace(
    spec: dict, candidate_index: int, *, correct: bool,
) -> dict[str, np.ndarray]:
    geometry = spec["geometry"]
    edge = geometry["selected_directed_edge"]
    opening = np.asarray(edge["opening_segment_world"], dtype=np.float64)
    normal = np.asarray(edge["opening_normal_world"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    midpoint = opening.mean(axis=0)
    start = midpoint - 0.35 * normal
    end = midpoint + (0.55 if correct else -0.30) * normal
    fraction = np.linspace(0.0, 1.0, R.PHYSICS_STEPS_PER_BRANCH)
    xy = start[None, :] + fraction[:, None] * (end - start)[None, :]
    yaw = math.atan2(float(normal[1]), float(normal[0]))
    pose = np.zeros((R.PHYSICS_STEPS_PER_BRANCH, 7), dtype=np.float64)
    pose[:, :2] = xy
    pose[:, 2] = 0.35
    pose[:, 5] = math.sin(yaw / 2.0)
    pose[:, 6] = math.cos(yaw / 2.0)
    twist = np.zeros((R.PHYSICS_STEPS_PER_BRANCH, 6), dtype=np.float64)
    twist[:, :2] = (end - start)[None, :] / (
        R.PHYSICS_STEPS_PER_BRANCH * R.TRACE_DT_S
    )
    requested_ticks = np.asarray(
        R._candidate_requested_commands(candidate_index), dtype=np.float64
    )
    requested = np.repeat(requested_ticks, 50, axis=0)
    source = np.asarray([
        R._point_in_polygon(point, geometry["source_node"]["boundary_polygon_world"])
        for point in xy
    ], dtype=np.uint8)
    selected = np.asarray([
        R._point_in_polygon(point, edge["edge_region_polygon_world"])
        for point in xy
    ], dtype=np.uint8)
    wrong = np.asarray([
        any(R._point_in_polygon(point, other["edge_region_polygon_world"])
            for other in geometry["competing_directed_edges"])
        for point in xy
    ], dtype=np.uint8)
    target = np.asarray([
        R._point_in_polygon(point, geometry["target_node"]["boundary_polygon_world"])
        for point in xy
    ], dtype=np.uint8)
    return {
        "timestamp_s": np.arange(
            1, R.PHYSICS_STEPS_PER_BRANCH + 1, dtype=np.float64
        ) * R.TRACE_DT_S,
        "base_pose_world": pose,
        "base_twist_world": twist,
        "joint_position": np.zeros((R.PHYSICS_STEPS_PER_BRANCH, 12), dtype=np.float64),
        "joint_velocity": np.zeros((R.PHYSICS_STEPS_PER_BRANCH, 12), dtype=np.float64),
        "requested_command": requested,
        "post_slew_applied_command": requested.copy(),
        "physics_contact": np.zeros(R.PHYSICS_STEPS_PER_BRANCH, dtype=np.uint8),
        "source_region_member": source,
        "correct_edge_region_member": selected,
        "wrong_edge_region_member": wrong,
        "target_region_member": target,
    }


class _FakePhysicalBackend:
    """Deterministic schema-complete backend used only by the stage-flow test."""

    runtime = {
        "backend": "synthetic_test_only",
        "physical_outcomes_claimed": False,
        "physics_dt_s": R.TRACE_DT_S,
    }

    @staticmethod
    def _teacher(spec: dict) -> dict[str, np.ndarray]:
        candidate = _fake_physics_trace(spec, 2, correct=True)
        return {
            "timestamp_s": candidate["timestamp_s"],
            "base_pose_world": candidate["base_pose_world"],
            "base_twist_world": candidate["base_twist_world"],
            "joint_position": candidate["joint_position"],
            "joint_velocity": candidate["joint_velocity"],
            "applied_command": candidate["post_slew_applied_command"],
            "requested_command": candidate["requested_command"],
            "physics_contact": candidate["physics_contact"],
            "source_region_member": candidate["source_region_member"],
            "edge_region_member": candidate["correct_edge_region_member"],
            "target_region_member": candidate["target_region_member"],
        }

    def qualify(self, spec: dict) -> dict:
        teacher = self._teacher(spec)
        snapshot = _fake_snapshot(spec, teacher)
        snapshot_sha = hashlib.sha256(snapshot["payload_bytes"]).hexdigest()
        return {
            "candidate_spec_id": spec["candidate_spec_id"],
            "initial_decision_state_sha256": snapshot_sha,
            "graph": _fake_graph(spec),
            "teacher_trace": teacher,
            "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
            "snapshot": snapshot,
            "contact_instrumentation": {
                "api": "robot.get_contacts",
                "sample_period_s": R.TRACE_DT_S,
                "forbidden_net_force_api_used": False,
                "ontology_sha256": C.CONTACT_AUTHORITY["ontology_sha256"],
            },
            "runtime_evidence": {
                **self.runtime,
                "snapshot_captured_before_teacher": True,
                "teacher_restored_from_serialized_snapshot": True,
                "teacher_snapshot_sha256": snapshot_sha,
            },
        }

    def reset_fixture(self, spec: dict, payload: bytes) -> dict:
        trace = _fake_physics_trace(spec, 2, correct=True)
        snapshot_sha = hashlib.sha256(payload).hexdigest()
        rgb_sha = R.canonical_array_sha256(np.zeros((168, 224, 3), dtype=np.uint8))
        trace_digests = R._trace_digest_projection(trace)
        trial = {
            "serialized_restore_used": True,
            "clone_equivalence_used": False,
            "restored_snapshot_sha256": snapshot_sha,
            "post_restore_state_sha256": hashlib.sha256(b"restored" + payload).hexdigest(),
            "current_rgb_sha256": rgb_sha,
            "base_pose_world": [float(value) for value in trace["base_pose_world"][-1]],
            "joint_position_sha256": R.canonical_array_sha256(trace["joint_position"][-1]),
            "joint_velocity_sha256": R.canonical_array_sha256(trace["joint_velocity"][-1]),
            "controller_state_sha256": hashlib.sha256(b"controller" + payload).hexdigest(),
            "rng_state_sha256": hashlib.sha256(b"rng" + payload).hexdigest(),
            "requested_command_sequence_sha256": trace_digests["requested_command"],
            "post_slew_applied_command_sequence_sha256": trace_digests[
                "post_slew_applied_command"
            ],
            "contact_sequence_sha256": trace_digests["physics_contact"],
            "termination_reason": "H3_COMPLETE",
            "stuck": False,
        }
        return {
            "candidate_spec_id": spec["candidate_spec_id"],
            "snapshot_payload_sha256": snapshot_sha,
            "reset_trials": [
                {"metadata": {"trial_index": index, **trial}, "trace": trace}
                for index in range(2)
            ],
            "runtime_evidence": self.runtime,
        }

    def fanout(self, spec: dict, payload: bytes) -> list[dict]:
        snapshot_sha = hashlib.sha256(payload).hexdigest()
        return [
            {
                "candidate_index": index,
                "snapshot_payload_sha256": snapshot_sha,
                "trace": _fake_physics_trace(spec, index, correct=index == 2),
                "runtime_evidence": self.runtime,
            }
            for index in range(R.CANDIDATE_COUNT)
        ]

    def repeat(self, spec: dict, payload: bytes, indices: list[int]) -> list[dict]:
        snapshot_sha = hashlib.sha256(payload).hexdigest()
        return [
            {
                "selector_index": selector,
                "candidate_index": index,
                "snapshot_payload_sha256": snapshot_sha,
                "trace": _fake_physics_trace(spec, index, correct=index == 2),
                "runtime_evidence": self.runtime,
            }
            for selector, index in enumerate(indices)
        ]


class _FakeEncoder:
    def encode_singleton(self, _image: np.ndarray) -> dict:
        raw = np.zeros((768, 1024), dtype=np.float16)
        descriptor = np.zeros((768, 1024), dtype=np.float32)
        return {
            "raw_tokens": raw,
            "spatial_descriptor": descriptor,
            "preprocessed_tensor_sha256": hashlib.sha256(b"fake-preprocess").hexdigest(),
            "checkpoint_sha256": R.FROZEN_ENCODER_SHA256,
            "batch_size": 1,
        }


class _FakeRanker:
    def score(self, _tokens, _waypoint, _previous, _history) -> list[float]:
        return [1.0 if index == 2 else -float(index) for index in range(R.CANDIDATE_COUNT)]


def _candidate_trace(spec: dict, *, samples: int = 750) -> dict[str, np.ndarray]:
    geometry = spec["geometry"]
    opening = np.asarray(
        geometry["selected_directed_edge"]["opening_segment_world"], dtype=np.float64
    )
    normal = np.asarray(
        geometry["selected_directed_edge"]["opening_normal_world"], dtype=np.float64
    )
    normal /= np.linalg.norm(normal)
    midpoint = opening.mean(axis=0)
    start = midpoint - 0.35 * normal
    end = midpoint + 0.55 * normal
    alpha = np.linspace(0.0, 1.0, samples)
    xy = start[None, :] + alpha[:, None] * (end - start)[None, :]
    yaw = math.atan2(float(normal[1]), float(normal[0]))
    pose = np.zeros((samples, 7), dtype=np.float64)
    pose[:, :2] = xy
    pose[:, 2] = 0.35
    pose[:, 5] = math.sin(yaw / 2.0)
    pose[:, 6] = math.cos(yaw / 2.0)
    twist = np.zeros((samples, 6), dtype=np.float64)
    twist[:, :2] = (end - start)[None, :] / (samples * R.TRACE_DT_S)
    requested = np.repeat(
        np.asarray(R._candidate_requested_commands(2), dtype=np.float64), 50, axis=0
    )
    source = np.asarray(
        [R._point_in_polygon(value, geometry["source_node"]["boundary_polygon_world"])
         for value in xy], dtype=np.uint8,
    )
    selected = np.asarray(
        [R._point_in_polygon(
            value, geometry["selected_directed_edge"]["edge_region_polygon_world"]
        ) for value in xy], dtype=np.uint8,
    )
    wrong = np.asarray(
        [any(R._point_in_polygon(value, edge["edge_region_polygon_world"])
             for edge in geometry["competing_directed_edges"]) for value in xy],
        dtype=np.uint8,
    )
    target = np.asarray(
        [R._point_in_polygon(value, geometry["target_node"]["boundary_polygon_world"])
         for value in xy], dtype=np.uint8,
    )
    return {
        "timestamp_s": np.arange(1, samples + 1, dtype=np.float64) * R.TRACE_DT_S,
        "base_pose_world": pose,
        "base_twist_world": twist,
        "joint_position": np.zeros((samples, 12), dtype=np.float64),
        "joint_velocity": np.zeros((samples, 12), dtype=np.float64),
        "requested_command": requested,
        "post_slew_applied_command": requested.copy(),
        "physics_contact": np.zeros(samples, dtype=np.uint8),
        "source_region_member": source,
        "correct_edge_region_member": selected,
        "wrong_edge_region_member": wrong,
        "target_region_member": target,
    }


def test_signed_plane_crossing_and_candidate_outcome_are_complete() -> None:
    spec = C.build_candidate_specs()[0]
    trace = _candidate_trace(spec)
    opening = spec["geometry"]["selected_directed_edge"]
    crossing = R._first_positive_port_crossing(
        trace["base_pose_world"], opening["opening_segment_world"],
        opening["opening_normal_world"],
    )
    assert crossing is not None
    assert crossing["normal_dot_displacement_m"] > 0.0
    assert 0.0 <= crossing["lateral_fraction"] <= 1.0
    outcome = R.derive_candidate_outcome(
        spec, trace, trace["base_pose_world"][0], candidate_index=2
    )
    identity_fields = {
        "branch_id", "state_id", "role", "family", "candidate_index",
            "candidate_id", "snapshot_id", "restored_snapshot_sha256",
            "trace_index", "trace_slice", "trace_array_slice_sha256s",
            "physical_runtime_core_sha256",
    }
    assert set(outcome) == M.CANDIDATE_FANOUT_FIELDS - identity_fields
    assert outcome["entered_correct_edge"] is True
    assert outcome["entered_wrong_edge"] is False
    assert outcome["port_crossing_displacement_world_xy"] is not None
    assert outcome["beyond_port_consecutive_physics_samples"] >= 100


def test_geometry_boundary_and_tie_precedence_match_pure_authority() -> None:
    polygon = [[-10.0, -1.0], [10.0, -1.0], [10.0, 1.0], [-10.0, 1.0]]
    for point in ([0.0, 1.0 + 0.5e-6], [10.0 + 0.5e-6, 0.0], [0.0, 1.01]):
        assert R._point_in_polygon(point, polygon) is M.point_in_polygon_inclusive(
            point, polygon
        )

    segment = [[0.0, -0.5], [0.0, 0.5]]
    before, after = [-0.1, 0.5 + 0.5e-6], [0.1, 0.5 + 0.5e-6]
    pure = M.transverse_port_crossing(before, after, segment, [1.0, 0.0])
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[:, :2] = [before, after]
    poses[:, 6] = 1.0
    local = R._first_positive_port_crossing(poses, segment, [1.0, 0.0])
    assert pure is not None and local is not None
    assert local["fraction"] == pure["crossing_fraction"]
    assert local["lateral_fraction"] == pure["lateral_fraction"] == 1.0

    spec = json.loads(json.dumps(C.build_candidate_specs()[0]))
    selected = spec["geometry"]["selected_directed_edge"]
    spec["geometry"]["competing_directed_edges"] = [{
        "edge_id": "competing-edge-tie",
        "boundary_side": "TIE",
        "opening_segment_world": selected["opening_segment_world"],
        "opening_normal_world": selected["opening_normal_world"],
        "opening_width_m": selected["opening_width_m"],
        "edge_region_polygon_world": selected["edge_region_polygon_world"],
    }]
    trace = _candidate_trace(spec)
    trace["wrong_edge_region_member"] = trace["correct_edge_region_member"].copy()
    outcome = R.derive_candidate_outcome(spec, trace, trace["base_pose_world"][0], 2)
    assert outcome["entered_correct_edge"] is False
    assert outcome["entered_wrong_edge"] is True
    first = M.first_registered_port_crossing(
        trace["base_pose_world"][:, :2].tolist(),
        spec["geometry"]["selected_directed_edge"],
        spec["geometry"]["competing_directed_edges"],
    )
    assert first is not None and first["edge_id"] == "competing-edge-tie"


def test_runner_selection_projection_matches_independent_metrics() -> None:
    rows = []
    for index in range(R.CANDIDATE_COUNT):
        correct = index == 3
        rows.append({
            "candidate_index": index,
            "oracle_admissible": True,
            "entered_correct_edge": correct,
            "successor_viable": True,
            "positive_port_progress": correct,
            "physics_contact": False,
            "stuck": False,
            "entered_wrong_edge": False,
            "no_edge": not correct,
            "port_progress_m": 1.0 if correct else float(index) / 20.0,
            "lateral_error_m": 0.1,
            "angular_error_rad": 0.2,
        })
    scores = [float(index == 3) for index in range(R.CANDIDATE_COUNT)]
    projection = R._selection_metrics(rows, R._rank_scores(scores), scores)
    validated = M._selection_projection(projection, rows)
    assert validated["correct_edge_top1"] is True
    assert validated["normalized_port_regret"] == projection["normalized_port_regret"]


def test_split_assignment_order_matches_panel_canonical_state_order() -> None:
    specs = []
    for family in C.FAMILY_IDS:
        family_specs = [
            row for row in C.build_candidate_specs()
            if row["family"] == family and row["variant_index"] == 0
        ]
        assert len(family_specs) == 16
        specs.extend(family_specs)
    assignments, _digest = R._split_assignments(specs)
    assert [row["state_id"] for row in assignments] == sorted(
        row["state_id"] for row in assignments
    )


def test_teacher_record_contains_raw_reducer_fields_and_interpolated_velocity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path)
    spec = C.build_candidate_specs()[0]
    candidate = _candidate_trace(spec)
    teacher = {
        "timestamp_s": candidate["timestamp_s"],
        "base_pose_world": candidate["base_pose_world"],
        "base_twist_world": candidate["base_twist_world"],
        "joint_position": candidate["joint_position"],
        "joint_velocity": candidate["joint_velocity"],
        "applied_command": candidate["post_slew_applied_command"],
        "requested_command": candidate["requested_command"],
        "physics_contact": candidate["physics_contact"],
        "source_region_member": candidate["source_region_member"],
        "edge_region_member": candidate["correct_edge_region_member"],
        "target_region_member": candidate["target_region_member"],
    }
    edge = spec["geometry"]["selected_directed_edge"]
    crossing = R.canonical_port_crossing(
        teacher["base_pose_world"], teacher["target_region_member"],
        edge["opening_segment_world"], edge["opening_normal_world"],
        spec["geometry"]["competing_directed_edges"],
    )
    arrays = {f"teacher__{name}": value for name, value in teacher.items()}
    path = tmp_path / "teacher_traces.npz"
    payload, _spans = R._concat_traces([teacher], R.TEACHER_TRACE_MEMBERS)
    R.atomic_npz(path, **payload)
    metadata = {
        "candidate_spec": spec,
        "initial_decision_state_sha256": "1" * 64,
        "current_rgb_sha256": "2" * 64,
        "goal_reachable": True,
        "graph_edge_physically_executable": True,
        "graph": {
            "source_node_id": "source", "target_node_id": "target",
            "directed_edge_id": "selected-edge",
        },
        "teacher": {
            "crossing": crossing, "contact_free": True,
            "left_source_region": True, "positive_route_progress": True,
            "route_progress_m": 0.35, "competing_port_entered": False,
            "reached_target_node": bool(teacher["target_region_member"].any()),
            "teacher_valid": True,
        },
    }
    row = R._teacher_record(
        metadata, arrays, trace_index=0, span=(0, 750), selected=True,
        teacher_file=path,
    )
    assert set(row) == M.TEACHER_RECORD_FIELDS
    before, after, alpha = (
        crossing["sample_before"], crossing["sample_after"], crossing["fraction"]
    )
    expected_velocity = (
        (1.0 - alpha) * teacher["base_twist_world"][before, :2]
        + alpha * teacher["base_twist_world"][after, :2]
    )
    assert row["crossing_velocity_world_xy"] == pytest.approx(expected_velocity)
    assert row["first_source_exit_sample_index"] is not None
    assert isinstance(row["successor_viable"], bool)


def test_snapshot_normalizer_requires_all_controller_arrays() -> None:
    sha = "a" * 64
    value = {
        "payload_bytes": b"snapshot",
        **{name: np.zeros(shape, dtype=np.float64)
           for name, shape in R.SNAPSHOT_NUMERIC_FIELDS.items()},
        "serialized_solver_state_sha256": sha,
        "serialized_controller_state_sha256": sha,
        "serialized_rng_state_sha256": sha,
        "policy_last_action_sha256": sha,
        "capture_timestamp_s": 1.5,
        "controller_observation_sha256": sha,
        "previous_policy_action_sha256": sha,
        "torch_cpu_rng_state_sha256": sha,
        "torch_device_rng_state_sha256s": [],
        "torch_device_count": 0,
        "previous_applied_command_sha256": sha,
        "command_history_sha256": sha,
        "control_history_sha256": sha,
        "low_level_policy_state_sha256": sha,
        "solver_field_inventory": ["solver.field"],
        "controller_field_inventory": ["controller_observation_before_final_policy_act"],
        "rng_field_inventory": ["python", "torch"],
    }
    arrays, metadata = R._normalise_snapshot(value)
    assert arrays["snapshot__controller_observation"].shape == (45,)
    assert arrays["snapshot__command_history"].shape == (15, 3)
    assert arrays["snapshot__control_history"].shape == (15, 2)
    assert arrays["snapshot__low_level_policy_state"].shape == (12,)
    assert metadata["low_level_policy_state_sha256"] == sha


def test_fixed_spawn_reset_does_not_open_planner() -> None:
    calls: list[tuple] = []

    class Robot:
        def set_pos(self, *args, **kwargs): calls.append(("pos", args, kwargs))
        def set_quat(self, *args, **kwargs): calls.append(("quat", args, kwargs))
        def set_dofs_position(self, *args, **kwargs): calls.append(("q", args, kwargs))
        def set_dofs_velocity(self, *args, **kwargs): calls.append(("qd", args, kwargs))

    class Policy:
        reset_stance_rad = np.arange(12, dtype=np.float32)
        def reset(self, envs): calls.append(("policy-reset", tuple(envs)))

    runner = SimpleNamespace(
        n_envs=1, config=SimpleNamespace(randomize_spawn_pose=False),
        pack=SimpleNamespace(robot=SimpleNamespace(
            spawn_xyz_m=(1.0, 2.0, 0.375), spawn_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )),
        policy=Policy(), _stance=np.zeros(12, dtype=np.float32),
        build=SimpleNamespace(robot=Robot()), _leg_dof_idx=np.arange(12),
        _spawn_xyz_per_env=np.zeros((1, 3), dtype=np.float32),
        _spawn_quat_wxyz_per_env=np.zeros((1, 4), dtype=np.float32),
        _last_executed=np.ones((1, 3), dtype=np.float32),
        _blocks_in_episode=np.ones(1, dtype=np.int64),
        _consecutive_tipped_blocks=np.ones(1, dtype=np.int64),
        _recovery_interlock_blocks_remaining=np.ones(1, dtype=np.int64),
    )
    R._reset_robot_to_fixed_spawn_compat(runner)
    assert [row[0] for row in calls] == ["pos", "quat", "q", "qd", "policy-reset"]
    assert np.array_equal(runner._last_executed, np.zeros((1, 3)))
    assert tuple(runner._spawn_xyz_per_env[0]) == (1.0, 2.0, 0.375)


def test_ranker_goal_projection_uses_bearing_not_port_tangent() -> None:
    captured: dict = {}

    class Inner:
        def __call__(self, tokens, relative, **kwargs):
            captured["relative"] = list(relative)
            return {
                "candidate_names": list(R.CANDIDATE_IDS),
                "candidate_scores": list(range(R.CANDIDATE_COUNT)),
            }

    ranker = R._FrozenPhysicalRanker.__new__(R._FrozenPhysicalRanker)
    ranker._ranker = Inner()
    waypoint = {"dx_m": 1.0, "dy_m": -1.0, "target_tangent_heading_rad": 2.4}
    scores = ranker.score(
        np.zeros((768, 1024), dtype=np.float16), waypoint,
        [0.0, 0.0, 0.0], np.zeros((15, 2)),
    )
    assert scores == list(map(float, range(R.CANDIDATE_COUNT)))
    assert captured["relative"][2] == pytest.approx(-math.pi / 4.0)
    assert captured["relative"][2] != waypoint["target_tangent_heading_rad"]


def test_waypoint_row_separates_goal_bearing_from_target_tangent() -> None:
    row = R._waypoint_row(
        "state", "DIRECTED_EDGE_PORT", [0.0, 0.0, 0.0],
        [1.0, -1.0, 2.4], [1.0, 0.0],
    )
    assert set(row) == M.WAYPOINT_ROW_FIELDS
    assert row["relative_heading_rad"] == pytest.approx(-math.pi / 4.0)
    assert row["target_tangent_heading_rad"] == pytest.approx(2.4)


def test_heldout_fanout_is_fail_closed_before_target_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path / "material")
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path / "official")
    (R.MATERIAL_ROOT / "fanout").mkdir(parents=True)
    R.OUTPUT_ROOT.mkdir()
    monkeypatch.setattr(R, "require_stage_runtime", lambda *args, **kwargs: {})
    monkeypatch.setattr(R, "_require_initialized", lambda: ({}, {}))
    state = {
        "state_id": "heldout", "role": "DEVELOPMENT_HELDOUT", "family": R.FAMILIES[0],
        "candidate_spec_id": C.build_candidate_specs()[0]["candidate_spec_id"],
    }
    monkeypatch.setattr(R, "_panel_material", lambda: ([state], {"heldout": {}}))
    with pytest.raises(R.ExperimentError, match="before target selection freeze"):
        R.fanout_state_stage("heldout", backend=object(), fake_runtime=True)


def test_snapshot_payload_inspection_uses_exact_serialized_byte_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state_snapshots.npz"
    payload = np.frombuffer(b"one-two-three", dtype=np.uint8).copy()
    R.atomic_npz(
        path, snapshot_payload_bytes=payload,
        snapshot_offsets=np.asarray([0, len(payload)], dtype=np.int64),
    )
    authority = {
        "snapshot_payload_bytes": {
            "descr": "|u1", "digest_dtype": "uint8", "shape": ["B"],
            "hash_mode": "offset_slices", "offsets_member": "snapshot_offsets",
            "slice_digest_domain": "sha256_of_exact_serialized_snapshot_bytes",
        },
        "snapshot_offsets": {
            "descr": "<i8", "digest_dtype": "int64", "shape": [2],
            "hash_mode": "whole",
        },
    }
    inspected = R._inspect_npz_for_pure_metrics(path, authority)
    observed = inspected["members"]["snapshot_payload_bytes"]["row_or_slice_sha256s"][0]
    assert observed == hashlib.sha256(b"one-two-three").hexdigest()


def test_runtime_source_closure_rehashes_every_bound_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    first = repo / "first.py"; first.write_bytes(b"first\n")
    second = repo / "second.py"; second.write_bytes(b"second\n")
    closure_path = repo / "closure.json"
    monkeypatch.setattr(R, "REPO_ROOT", repo)
    monkeypatch.setattr(R, "DOC_PATHS", {**R.DOC_PATHS, "source_closure": closure_path})
    monkeypatch.setattr(C, "SOURCE_CLOSURE_PATHS", ("first.py", "second.py"))
    rows = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": R.sha256_file(path)}
        for path in (first, second)
    ]
    R.atomic_json(closure_path, R.attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.source_closure.v1",
        "parent_commit": R.PARENT_COMMIT, "row_count": 2, "rows": rows,
    }))
    assert R._validate_runtime_source_closure()["row_count"] == 2
    second.write_bytes(b"tampered\n")
    with pytest.raises(R.ExperimentError, match="live binding drift"):
        R._validate_runtime_source_closure()


def test_singleton_preprocess_uses_exact_png_path_and_actual_helper(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class PathOnlyArm:
        def preprocess(self, path: str):
            from PIL import Image

            with Image.open(path) as image:
                assert image.mode == "RGB" and image.size == (224, 168)
            return torch.zeros((3, 384, 512), dtype=torch.float32)

    value = R._preprocess_singleton_image(
        PathOnlyArm(), np.zeros((168, 224, 3), dtype=np.uint8), tmp_path
    )
    assert tuple(value.shape) == (3, 384, 512)
    assert value.dtype == torch.float32 and value.device.type == "cpu"

    from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm

    actual = R._preprocess_singleton_image(
        VJepa21Arm(), np.zeros((168, 224, 3), dtype=np.uint8), tmp_path
    )
    assert tuple(actual.shape) == (3, 384, 512)
    assert actual.dtype == torch.float32 and actual.is_contiguous()


def test_runtime_evidence_preserves_lexical_interpreter_and_marks_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked = tmp_path / "bound-python"
    invoked.symlink_to(Path(sys.executable))
    monkeypatch.setattr(R.sys, "executable", str(invoked))

    physical = R.require_stage_runtime("physical", fake=True)
    assert physical["python_executable"] == str(invoked)
    assert physical["fake_runtime"] is True
    digest = M.runtime_environment_sha256(physical)
    validated = M.validate_physical_runtime_environment({
        **physical,
        "runtime_core_sha256": digest,
        "qualification_runtime_sha256s": [digest] * C.TEACHER_TRACE_COUNT,
        "selected_snapshot_runtime_sha256s": [digest] * C.STATE_COUNT,
    })
    assert validated["fake_runtime"] is True

    for role in ("encoder", "ranker"):
        visual = R.require_stage_runtime(
            "visual", fake=True, visual_role=role,
        )
        assert visual["python_executable"] == str(invoked)
        assert visual["fake_runtime"] is True
        assert M.validate_visual_runtime_environment(
            visual, runtime_role=role,
        ) == visual


def test_fake_publication_flow_emits_detailed_report_and_hash_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "result"
    material = tmp_path / "material"
    root.mkdir(); material.mkdir()
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "SCIENTIFIC_LEAVES", ("contract.json", "metrics.json"))
    monkeypatch.setattr(R, "ALL_OUTPUT_LEAVES", (
        "contract.json", "metrics.json", "result.json", "result.md", "file_hashes.json",
    ))
    R.atomic_json(root / "contract.json", {"source_freeze_commit": "b" * 40})
    R.atomic_json(root / "metrics.json", {"value": 1})
    R.atomic_json(material / "material_contract.json", {"started_at_unix_s": 1.0})
    condition = [{"condition_id": value, "state_count": 16} for value in R.HELDOUT_CONDITIONS]
    targets = [{"target_id": value, "state_count": 48} for value in R.TARGET_IDS]
    metrics = {
        "primary_classification": "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO",
        "secondary_classifications": [], "next_experiment": "NEXT",
        "evidence_counts": {"selected_states": 64},
        "panel": {"coverage_rate": 0.5},
        "development": {"selected_target_id": "DIRECTED_EDGE_PORT", "target_summaries": targets},
        "heldout": {"condition_summaries": condition},
            "repeatability": {"rate": 1.0}, "command_tracking": {"passed": True},
            "runtime_environments": {
                "physical": {"fake_runtime": True},
                "encoder": {"fake_runtime": True},
                "ranker": {"fake_runtime": True},
                "any_fake_runtime": True,
            },
            "stratified": {}, "gate": {"passed": False}, "component_failures": ["coverage"],
    }
    result = R._write_publication(metrics, {"pass": True})
    assert result["primary_classification"] == "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO"
    report = (root / "result.md").read_text()
    assert "All 64 selected decision states" in report
    assert "DETERMINISTIC_KINEMATICS" in report
    manifest = R.load_json(root / "file_hashes.json")
    assert manifest["file_count_excluding_self"] == 4
    assert [row["path"] for row in manifest["files"]] == sorted(
        ["contract.json", "metrics.json", "result.json", "result.md"]
    )


def test_complete_fake_stage_flow_reaches_production_reducer_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every registered handoff at full cardinality without science.

    The backend, encoder, and ranker are deterministic test doubles, but all
    runner stages, production metric validators, the 23-leaf root, raw NPZ
    reduction, and the independent reducer/publication order remain real.
    """

    from scripts import evaluate_physical_graph_edge_handoff_qualification_v1 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v1"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v1_material"
    receipt_path = tmp_path / (
        "physical_graph_edge_handoff_qualification_v1_regeneration_receipt.json"
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", receipt_path)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=R.REPO_ROOT, text=True
    ).strip()
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_commit)
    monkeypatch.setattr(
        R, "_runtime_contract", lambda freeze: C.build_runtime_contract(freeze)
    )
    exclusion = {
        "authority_digest": hashlib.sha256(
            C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
        ).hexdigest(),
        "checked_before_simulator_creation": True,
        "scene_overlap_count": 0,
        "scene_hash_overlap_count": 0,
        "state_or_episode_overlap_count": 0,
        "seed_overlap_count": 0,
        "path_or_geometry_overlap_count": 0,
        "structured_path_overlap_count": 0,
        "all_zero": True,
    }
    monkeypatch.setattr(R, "_scene_exclusion_audit", lambda _specs: exclusion)

    pool = R.initialize_stage(fake_runtime=True)
    assert len(pool["specs"]) == C.PROSPECTIVE_POOL_COUNT == 256
    del pool
    backend = _FakePhysicalBackend()
    for pool_index in range(C.PROSPECTIVE_POOL_COUNT):
        R.qualify_pool_state_stage(
            pool_index, backend=backend, fake_runtime=True
        )
    selection = R.select_teacher_pool_stage(fake_runtime=True)
    assert len(selection["teacher_records"]) == 256
    assert len(selection["selected_specs"]) == 64
    selected_state_ids = [spec["state_id"] for spec in selection["selected_specs"]]
    for state_id in selected_state_ids:
        R.capture_selected_state_stage(
            state_id, backend=backend, fake_runtime=True
        )
    del selection, selected_state_ids

    panel = R.freeze_panel_stage(fake_runtime=True)
    assert len(panel["states"]) == 64
    assert not any(
        any(path.iterdir()) for path in [material / "fanout", material / "repeat"]
    )
    latent = R.encode_canonical_pixels_stage(
        encoder=_FakeEncoder(), fake_runtime=True
    )
    assert len(latent["records"]) == 1
    del latent

    development = [
        row for row in panel["states"] if row["role"] == "DEVELOPMENT"
    ]
    heldout = [
        row for row in panel["states"] if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    assert (len(development), len(heldout)) == (48, 16)
    for state in development:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    target = R.development_target_selection_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert target["selection_frozen"] is True
    assert len(target["state_target_rows"]) == 144
    del target

    for state in heldout:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    scores = R.heldout_ranker_scores_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert len(scores) == 64
    del scores
    for state in heldout:
        R.repeat_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assembled = R.assemble_row_evidence_stage(fake_runtime=True)
    assert assembled == {
        "candidate_trace_count": 960,
        "candidate_fanout_count": 768,
        "repeat_count": 64,
        "candidate_traces_sha256": R.sha256_file(root / "candidate_traces.npz"),
    }
    del assembled, backend, development, heldout, panel
    gc.collect()

    # Avoid streaming the 5 GB real V-JEPA checkpoint in a unit test while
    # retaining the production role order and independent reducer code path.
    external_rows = []
    for index, row in enumerate(C.EXTERNAL_ARTIFACT_BINDINGS):
        path = tmp_path / f"external-{index}.bin"
        path.write_bytes(f"{row['role']}\n".encode())
        external_rows.append({
            "role": row["role"], "kind": row["kind"], "path": str(path),
            "bytes": path.stat().st_size, "sha256": R.sha256_file(path),
        })
    source_paths = {
        E.REDUCER_SOURCE_PATH, E.CONTRACT_SOURCE_PATH, E.METRICS_SOURCE_PATH,
        E.CUSTODY_SOURCE_PATH, E.BASE_CUSTODY_SOURCE_PATH,
    }
    source_observation = {
        "head_commit": source_commit,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            relative: {
                "path": relative,
                "bytes": (R.REPO_ROOT / relative).stat().st_size,
                "sha256": R.sha256_file(R.REPO_ROOT / relative),
            }
            for relative in source_paths
        },
        "metrics_module": M.__name__,
    }

    # First prove that the runner preserves the non-scientific identity of its
    # injected implementations and that the production reducer rejects it.
    assert R.load_json(root / "panel_manifest.json")[
        "physical_runtime_environment"
    ]["fake_runtime"] is True
    assert R.load_json(root / "latent_index.json")[
        "encoder_runtime_environment"
    ]["fake_runtime"] is True
    assert R.load_json(root / "development_target_selection.json")[
        "ranker_runtime_environment"
    ]["fake_runtime"] is True
    assert "production_shaped_runtime" not in Path(R.__file__).read_text()

    recompute_child = r'''
from pathlib import Path
import sys
from scripts import run_physical_graph_edge_handoff_qualification_v1 as R
R.OUTPUT_ROOT = Path(sys.argv[1])
R.MATERIAL_ROOT = Path(sys.argv[2])
R.require_runtime_source_freeze = lambda: sys.argv[3]
R.recompute_and_persist_metrics_stage(fake_runtime=True)
'''
    subprocess.run(
        ["python3", "-c", recompute_child, str(root), str(material), source_commit],
        cwd=R.REPO_ROOT, check=True,
    )
    monkeypatch.setattr(
        M, "external_artifact_bindings", lambda _contract: external_rows
    )
    rejected_receipt = tmp_path / "fake-runtime-must-be-rejected.json"
    with pytest.raises(E.RegenerationError, match="marked fake"):
        E.verify_and_emit(
            root, rejected_receipt, metrics_module=M,
            source_freeze_observation=source_observation,
        )
    assert not rejected_receipt.exists()
    (root / "metrics.json").unlink()

    # The following production-shaped projection exists only inside this test.
    # It exercises the independent production reducer without adding any
    # fake-to-real normalization path to runner or evaluator code.
    physical_authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    physical_core = {
        "stage_id": physical_authority["stage_id"],
        "python_executable": physical_authority["real_python_executable"],
        "python_version": physical_authority["python_version"],
        "torch_version": physical_authority["torch_version"],
        "torch_hip_version": physical_authority["torch_hip_version"],
        "genesis_version": physical_authority["genesis_version"],
        "quadrants_version": physical_authority["quadrants_version"],
        "visible_device_count": physical_authority["visible_device_count"],
        "device": physical_authority["device"],
        "backend": physical_authority["backend"],
        "deterministic_environment": dict(C.DIRECT_RUNTIME_POLICY[
            "required_environment_before_simulator_creation"
        ]),
        "fake_runtime": False,
    }
    physical_digest = M.runtime_environment_sha256(physical_core)
    physical_environment = M.validate_physical_runtime_environment({
        **physical_core,
        "runtime_core_sha256": physical_digest,
        "qualification_runtime_sha256s": [physical_digest] * C.TEACHER_TRACE_COUNT,
        "selected_snapshot_runtime_sha256s": [physical_digest] * C.STATE_COUNT,
    })

    def real_visual_environment(role: str) -> dict:
        authority = C.RUNTIME_ENVIRONMENT_AUTHORITY[role]
        return M.validate_visual_runtime_environment(
                {
                    field: (
                        False if field == "fake_runtime"
                        else authority["real_python_executable"]
                        if field == "python_executable"
                        else authority[field]
                    )
                    for field in C.VISUAL_RUNTIME_ENVIRONMENT_FIELDS
                },
            runtime_role=role,
        )

    encoder_environment = real_visual_environment("encoder")
    ranker_environment = real_visual_environment("ranker")
    ranker_digest = M.runtime_environment_sha256(ranker_environment)

    def replace_document(path: Path, field: str, value: dict) -> None:
        document = R.load_json(path)
        document.pop("content_digest")
        document[field] = value
        path.unlink()
        R.atomic_json(path, R.attach_digest(document))

    replace_document(
        root / "panel_manifest.json", "physical_runtime_environment",
        physical_environment,
    )
    replace_document(
        root / "latent_index.json", "encoder_runtime_environment",
        encoder_environment,
    )
    replace_document(
        root / "development_target_selection.json", "ranker_runtime_environment",
        ranker_environment,
    )
    for name, field, digest in (
        ("candidate_fanout.jsonl", "physical_runtime_core_sha256", physical_digest),
        ("repeated_execution.jsonl", "physical_runtime_core_sha256", physical_digest),
        ("heldout_ranker_scores.jsonl", "ranker_runtime_environment_sha256", ranker_digest),
    ):
        path = root / name
        rows = R.load_jsonl(path)
        for row in rows:
            row[field] = digest
        path.unlink()
        R.atomic_jsonl(path, rows)

    child = r'''
import json
from pathlib import Path
import sys
from scripts import run_physical_graph_edge_handoff_qualification_v1 as R
from scripts import evaluate_physical_graph_edge_handoff_qualification_v1 as E
from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as M
root, material, receipt = map(Path, sys.argv[1:4])
source_commit = sys.argv[4]
external_rows = json.loads(sys.argv[5])
source_observation = json.loads(sys.argv[6])
R.OUTPUT_ROOT = root
R.MATERIAL_ROOT = material
R.EXTERNAL_REGENERATION_RECEIPT = receipt
R.require_runtime_source_freeze = lambda: source_commit
M.external_artifact_bindings = lambda _contract: json.loads(json.dumps(external_rows))
class EvaluatorAdapter:
    @staticmethod
    def verify_and_emit(output_root, output, *, metrics_module):
        return E.verify_and_emit(
            output_root, output, metrics_module=metrics_module,
            source_freeze_observation=source_observation,
        )
    @staticmethod
    def validate_existing_regeneration_receipt(output_root, output, *, metrics_module):
        return E.validate_existing_regeneration_receipt(
            output_root, output, metrics_module=metrics_module,
            source_freeze_observation=source_observation,
        )
result = R.report_stage(fake_runtime=True, evaluator_module=EvaluatorAdapter)
print(json.dumps(result, sort_keys=True))
'''
    completed = subprocess.run(
        [
            "python3", "-c", child, str(root), str(material), str(receipt_path),
            source_commit, json.dumps(external_rows, sort_keys=True),
            json.dumps(source_observation, sort_keys=True),
        ],
        cwd=R.REPO_ROOT, check=True, text=True, capture_output=True,
    )
    result = json.loads(completed.stdout)
    assert result["models_trained"] == 0
    assert receipt_path.is_file()
    assert set(path.name for path in root.iterdir()) == set(R.ALL_OUTPUT_LEAVES)
    assert len(R.load_jsonl(root / "candidate_fanout.jsonl")) == 768
    assert len(R.load_jsonl(root / "heldout_ranker_scores.jsonl")) == 64
    assert len(R.load_jsonl(root / "repeated_execution.jsonl")) == 64
