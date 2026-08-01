from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = ROOT / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
DETERMINISM_PROBE_PATH = (
    ROOT / "scripts/dev_probe_go2_counterfactual_lockstep_determinism.py"
)


def _load_collector():
    spec = importlib.util.spec_from_file_location("counterfactual_collector_v1", COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_determinism_probe():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_lockstep_determinism_probe", DETERMINISM_PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_file(path: Path, payload: bytes = b"bound\n") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return pilot.file_binding(path)


def _block(action_id: int) -> list[list[float]]:
    return [list(command) for command in pilot.CANONICAL_ACTION_BLOCKS[action_id]]


def _catalog() -> list[dict[str, Any]]:
    return [
        {"action_id": index, "name": name, "requested_block": _block(index)}
        for index, name in enumerate(pilot.CANONICAL_ACTIONS)
    ]


def _runtime_bindings(tmp_path: Path) -> dict[str, dict[str, Any]]:
    names = (
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
        "python_executable_target",
        "python_environment_config",
        "eglinfo_executable",
        "vulkaninfo_executable",
    )
    return {
        name: _bound_file(tmp_path / "runtime" / name, name.encode()) for name in names
    }


def _execution(tmp_path: Path) -> dict[str, Any]:
    return {
        "backend": "vulkan",
        "policy_device": "cpu",
        "seed": 7,
        "fall_z_threshold_m": 0.15,
        "tip_threshold_rad": 1.0471975511965976,
        "policy_steps_per_command_tick": 5,
        "python_invocation_path": str((tmp_path / "venv/bin/python").resolve()),
        "environment": dict(pilot.EXECUTION_ENVIRONMENT),
        "graphics_preflight": dict(pilot.GRAPHICS_PREFLIGHT_EXPECTATION),
    }


def _smoke_plan(tmp_path: Path) -> dict[str, Any]:
    generator = _bound_file(tmp_path / "collector.py", b"collector source\n")
    states = [
        {
            "state_id": "smoke-state-0",
            "role": "calibration",
            "family": "open_obstacle_field",
            "scene_id": "open_obstacle_field-deadbeef",
            "scene_manifest_binding": None,
            "scene_genesis_binding": None,
            "scene_generation": {
                "family": "open_obstacle_field",
                "split": "calibration_smoke",
                "plan_seed": 17,
                "scene_index": 0,
                "scene_generator_binding": generator,
            },
            "group_index": 0,
            "state_index_in_scene": 0,
            "history_action_ids": [6, 3],
            "candidate_action_ids": list(range(9)),
            "sentinel_duplicate_action_id": 6,
            "target_xy_m": [1.0, 2.0],
        }
    ]
    return {
        "schema": pilot.PLAN_SCHEMA,
        "attempt_id": "smoke-attempt-1",
        "purpose": "source_integration_smoke",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "states_per_scene": 1,
        "history_blocks": 2,
        "output_root": str((tmp_path / "output").resolve()),
        "runtime_bindings": _runtime_bindings(tmp_path),
        "execution_contract": _execution(tmp_path),
        "render_contract": dict(pilot.RENDER_CONTRACT),
        "action_catalog": _catalog(),
        "states": states,
        "expected_counts": pilot.expected_counts_from_states(states),
    }


def _bounded_two_scene_plan(tmp_path: Path) -> dict[str, Any]:
    plan = _smoke_plan(tmp_path)
    plan["purpose"] = "bounded_wm_a_pilot"
    plan["output_root"] = str((tmp_path / "bounded-output").resolve())
    states: list[dict[str, Any]] = []
    for group_index, role in enumerate(("train", "eval")):
        scene_dir = tmp_path / "preexisting-scenes" / role / f"scene-{group_index}"
        states.append(
            {
                "state_id": f"{role}-state-0",
                "role": role,
                "family": "open_obstacle_field",
                "scene_id": f"open-obstacle-{role}",
                "scene_manifest_binding": _bound_file(
                    scene_dir / "manifest.json", b"{}"
                ),
                "scene_genesis_binding": _bound_file(
                    scene_dir / "genesis_scene.json", b"{}"
                ),
                "scene_generation": None,
                "group_index": group_index,
                "state_index_in_scene": 0,
                "history_action_ids": [6, 3],
                "candidate_action_ids": list(range(9)),
                "sentinel_duplicate_action_id": None,
                "target_xy_m": [1.0, 2.0],
            }
        )
    plan["states"] = states
    plan["expected_counts"] = pilot.expected_counts_from_states(states)
    return plan


def test_smoke_plan_is_exact_role_aware_and_authority_free(tmp_path: Path) -> None:
    plan = pilot.validate_plan(_smoke_plan(tmp_path))
    assert plan["expected_counts"] == {
        "scenes": 1,
        "states": 1,
        "roles": {"calibration": 1},
        "actions": 9,
        "candidate_branches": 9,
        "sentinel_branches": 1,
        "total_branches": 10,
        "context_frames": 3,
        "target_frames": 10,
    }
    assert "authority_binding" not in plan
    assert pilot.lane_count_for_role("calibration") == 10
    assert pilot.lane_count_for_role("train") == 9
    assert pilot.lane_count_for_role("eval") == 9


def test_sentinel_allocation_is_frozen_not_state_hash() -> None:
    assert pilot.deterministic_sentinel_action_id(state_index_in_scene=0) == 6
    assert pilot.deterministic_sentinel_action_id(state_index_in_scene=1) == 4
    with pytest.raises(pilot.PilotContractError):
        pilot.deterministic_sentinel_action_id(state_index_in_scene=2)
    assert len(
        pilot.lane_layout(
            "a", role="calibration", state_index_in_scene=0
        )
    ) == 10
    assert len(
        pilot.lane_layout("a", role="train", state_index_in_scene=0)
    ) == 9


@pytest.mark.parametrize(
    "mutate",
    (
        lambda plan: plan.update(authority_binding={}),
        lambda plan: plan.update(history_blocks=3),
        lambda plan: plan["states"][0].update(target_xy_m=None),
        lambda plan: plan["states"][0].update(sentinel_duplicate_action_id=4),
        lambda plan: plan["states"][0].update(group_index=1),
        lambda plan: plan["states"][0].update(scene_manifest_binding={}),
    ),
)
def test_plan_rejects_contract_drift(tmp_path: Path, mutate: Any) -> None:
    plan = _smoke_plan(tmp_path)
    mutate(plan)
    with pytest.raises(pilot.PilotContractError):
        pilot.validate_plan(plan)


def test_plan_rejects_named_action_with_changed_command_value(
    tmp_path: Path,
) -> None:
    plan = _smoke_plan(tmp_path)
    plan["action_catalog"][0]["requested_block"][3][2] = 0.4500001
    with pytest.raises(
        pilot.PilotContractError,
        match="changed from the canonical primitive",
    ):
        pilot.validate_plan(plan)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("backend", "cpu", "exact Vulkan"),
        ("policy_device", "cuda", "exact CPU"),
    ),
)
def test_plan_rejects_execution_substrate_drift(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    plan = _smoke_plan(tmp_path)
    plan["execution_contract"][field] = value
    with pytest.raises(pilot.PilotContractError, match=message):
        pilot.validate_plan(plan)


def test_read_bound_json_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    for index, payload in enumerate((b'{"a":1,"a":2}', b'{"a":NaN}')):
        path = tmp_path / f"bad-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(pilot.PilotContractError):
            pilot.read_bound_json(
                path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_byte_count=len(payload),
                label="adversarial JSON",
            )


def test_read_bound_json_requires_caller_byte_count(tmp_path: Path) -> None:
    payload = b'{"a":1}'
    path = tmp_path / "value.json"
    path.write_bytes(payload)
    with pytest.raises(pilot.PilotContractError):
        pilot.read_bound_json(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_count=len(payload) + 1,
            label="value",
        )


def test_file_binding_rejects_leaf_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(pilot.PilotContractError):
        pilot.file_binding(link)


def _components(values: np.ndarray) -> dict[str, np.ndarray]:
    lanes = values.shape[0]
    qpos = np.zeros((lanes, 19), dtype=np.float32)
    qpos[:, 0] = values
    return {
        "qpos": qpos,
        "dofs_velocity": np.zeros((lanes, 18), dtype=np.float32),
        "base_pos_world": np.column_stack(
            (values, np.zeros(lanes), np.full(lanes, 0.4))
        ).astype(np.float32),
        "base_quat_wxyz": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (lanes, 1)
        ),
        "base_lin_vel_world": np.zeros((lanes, 3), dtype=np.float32),
        "base_ang_vel_world": np.zeros((lanes, 3), dtype=np.float32),
        "leg_joint_pos": np.zeros((lanes, 12), dtype=np.float32),
        "leg_joint_vel": np.zeros((lanes, 12), dtype=np.float32),
        "runner_last_executed": np.zeros((lanes, 3), dtype=np.float32),
        "policy_last_actions": np.zeros((lanes, 12), dtype=np.float32),
    }


def test_sync_audit_uses_cumulative_role_aware_lanes() -> None:
    values = np.concatenate((np.zeros(10), np.ones(9), np.full(9, 2.0)))
    audits = pilot.audit_prebranch_synchronization(
        _components(values),
        state_ids=["cal", "train", "eval"],
        roles=["calibration", "train", "eval"],
    )
    assert [(row["lane_start"], row["lane_count"]) for row in audits] == [
        (0, 10),
        (10, 9),
        (19, 9),
    ]
    assert all(row["passed"] for row in audits)


def test_sync_audit_detects_one_bit_of_lane_drift() -> None:
    components = _components(np.zeros(10, dtype=np.float32))
    components["qpos"][9, 0] = np.nextafter(np.float32(0), np.float32(1))
    audit = pilot.audit_prebranch_synchronization(
        components, state_ids=["state"], roles=["calibration"]
    )[0]
    assert audit["passed"] is False
    assert audit["components"]["qpos"]["exact_equal"] is False
    assert audit["components"]["qpos"]["rms_difference"] > 0.0
    assert audit["components"]["qpos"]["per_lane_max_abs_difference"][-1] > 0.0


class _Trajectory:
    def __init__(self, requested: np.ndarray) -> None:
        self.requested = requested.copy()
        self.executed = requested.copy()
        self.clipped = np.zeros(requested.shape[0], dtype=np.bool_)


class _FakeRunner:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self.policy_steps_per_command_tick = 5
        self.values = np.zeros(n_envs, dtype=np.float32)
        self.last = np.zeros((n_envs, 3), dtype=np.float32)
        self.actions = np.zeros((n_envs, 12), dtype=np.float32)
        self.time_ns = 0

    def capture(self) -> dict[str, np.ndarray]:
        result = _components(self.values)
        result["runner_last_executed"] = self.last.copy()
        result["policy_last_actions"] = self.actions.copy()
        return result

    def execute_requested_block(self, requested: np.ndarray, *, after_policy_step=None):
        for tick in range(5):
            command = requested[:, tick]
            for policy_step in range(5):
                self.values += command[:, 0] * np.float32(0.01)
                self.actions.fill(0.0)
                self.actions[:, :3] = command
                self.time_ns += 20_000_000
                if after_policy_step is not None:
                    after_policy_step(tick, policy_step)
        self.last = requested[:, -1].copy()
        return _Trajectory(requested)


class _FirstStepDivergenceRunner(_FakeRunner):
    def execute_requested_block(self, requested: np.ndarray, *, after_policy_step=None):
        for tick in range(5):
            command = requested[:, tick]
            for policy_step in range(5):
                self.values += command[:, 0] * np.float32(0.01)
                self.actions.fill(0.0)
                self.actions[:, :3] = command
                self.time_ns += 20_000_000
                if tick == 0 and policy_step == 0:
                    self.values[-1] = np.nextafter(
                        self.values[-1], np.float32(1.0)
                    )
                if after_policy_step is not None:
                    after_policy_step(tick, policy_step)
        self.last = requested[:, -1].copy()
        return _Trajectory(requested)


class _YawFakeRunner(_FakeRunner):
    def __init__(self, n_envs: int, *, yaw_rad: float) -> None:
        super().__init__(n_envs)
        self.yaw_rad = yaw_rad

    def capture(self) -> dict[str, np.ndarray]:
        result = super().capture()
        quaternion = np.asarray(
            [
                math.cos(self.yaw_rad / 2.0),
                0.0,
                0.0,
                math.sin(self.yaw_rad / 2.0),
            ],
            dtype=np.float32,
        )
        result["base_quat_wxyz"] = np.tile(quaternion, (self.n_envs, 1))
        return result


def _state(
    state_id: str, role: str, group_index: int, state_index: int
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "role": role,
        "group_index": group_index,
        "state_index_in_scene": state_index,
        "history_action_ids": [6, 3],
        "target_xy_m": [1.0, 0.0],
    }


def _render_batch(runner: _FakeRunner) -> dict[str, Any]:
    rgb = np.zeros((runner.n_envs, 224, 224, 3), dtype=np.uint8)
    for lane, value in enumerate(runner.values):
        rgb[lane].fill(int(round(float(value) * 100)) % 255)
    return {
        "stored_rgb": rgb,
        "quality": [
            {"valid": True, "invalid_reasons": []} for _ in range(runner.n_envs)
        ],
        "native_resolution": [640, 480],
        "stored_resolution": [224, 224],
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": "solid_materials_box_physics_preserved",
    }


def _render_replay(
    trial: dict[str, Any], states: list[dict[str, Any]]
) -> dict[str, Any]:
    lane_counts = [pilot.lane_count_for_role(state["role"]) for state in states]
    lane_starts = np.cumsum([0, *lane_counts[:-1]]).tolist()

    def frame(components: dict[str, np.ndarray], env_index: int) -> dict[str, Any]:
        value = float(np.asarray(components["base_pos_world"])[env_index, 0])
        rgb = np.full(
            (224, 224, 3), int(round(value * 100)) % 255, dtype=np.uint8
        )
        return {
            "stored_rgb": rgb,
            "quality": {"valid": True, "invalid_reasons": []},
            "native_resolution": [640, 480],
            "stored_resolution": [224, 224],
            "depth_rendered": True,
            "depth_persisted": False,
            "visual_mode": "solid_materials_box_physics_preserved",
            "source_base_position_xyz_m": np.asarray(
                components["base_pos_world"]
            )[env_index].copy(),
            "source_base_quaternion_wxyz": np.asarray(
                components["base_quat_wxyz"]
            )[env_index].copy(),
            "camera_pose_world": {
                "position_xyz_m": np.asarray(
                    components["base_pos_world"]
                )[env_index].copy(),
                "lookat_xyz_m": np.asarray(
                    components["base_pos_world"]
                )[env_index].copy()
                + np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "up_xyz": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            },
        }

    context_frames = {
        str(state["state_id"]): [
            frame(snapshot, int(lane_starts[group_index]))
            for snapshot in trial["history_snapshots"]
        ]
        for group_index, state in enumerate(states)
    }
    endpoint = trial["branch_endpoint"]
    branch_frames = [
        frame(endpoint, env_index) for env_index in range(sum(lane_counts))
    ]
    return {
        "context_frames": context_frames,
        "branch_frames": branch_frames,
        "native_render_calls": sum(len(rows) for rows in context_frames.values())
        + len(branch_frames),
    }


def test_sequential_replay_uses_exact_physical_poses_and_non_batched_camera() -> None:
    collector = _load_collector()
    state = _state("cal", "calibration", 0, 0)
    runner = _YawFakeRunner(10, yaw_rad=math.pi / 3.0)
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=[state],
        action_blocks=[_block(index) for index in range(9)],
        capture_components=runner.capture,
        capture_sim_time_ns=lambda: runner.time_ns,
    )

    class FakeCamera:
        _is_batched = False

        def __init__(self) -> None:
            self.poses: list[dict[str, np.ndarray]] = []

        def set_pose(self, **kwargs: np.ndarray) -> None:
            assert set(kwargs) == {"pos", "lookat", "up"}
            assert all(np.asarray(value).shape == (3,) for value in kwargs.values())
            self.poses.append(
                {
                    name: np.asarray(value, dtype=np.float32).copy()
                    for name, value in kwargs.items()
                }
            )

        def render(self, *, rgb: bool, depth: bool, force_render: bool):
            assert (rgb, depth, force_render) == (True, True, True)
            encoded = int(round((float(self.poses[-1]["pos"][0]) + 5.0) * 20.0))
            return (
                np.full((3, 4, 3), encoded, dtype=np.uint8),
                np.ones((3, 4), dtype=np.float32),
            )

    class FakePack:
        camera = type(
            "CameraContract",
            (),
            {"native_resolution": (4, 3), "training_resolution": (2, 2)},
        )()
        static_objects: tuple[Any, ...] = ()

    class FakePose:
        def __init__(self, position: np.ndarray) -> None:
            self.position = position.copy()
            self.lookat = position + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            self.up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    camera = FakeCamera()
    render_build = type(
        "RenderBuild",
        (),
        {"n_envs": 1, "camera": camera, "pack": FakePack()},
    )()
    safe_pose_inputs: list[tuple[np.ndarray, np.ndarray]] = []

    def safe_pose(position, quat_xyzw, **kwargs):
        assert kwargs["objects"] == ()
        safe_pose_inputs.append(
            (
                np.asarray(position, dtype=np.float32).copy(),
                np.asarray(quat_xyzw, dtype=np.float32).copy(),
            )
        )
        return FakePose(np.asarray(position, dtype=np.float32)), {"unsafe": False}

    stage_wall_times = collector._new_stage_wall_times()
    replay = collector._capture_sequential_render_replay(
        render_build,
        states=[state],
        trial=trial,
        safe_camera_pose_from_base=safe_pose,
        camera_safety_config_from_pack=lambda _pack: object(),
        effective_camera_mount_xyz_rpy=lambda _pack: (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        assess_rendered_frame=lambda *_args, **_kwargs: {
            "valid": True,
            "invalid_reasons": [],
        },
        stage_wall_times=stage_wall_times,
    )
    expected_sources = [
        (snapshot, 0) for snapshot in trial["history_snapshots"]
    ] + [(trial["branch_endpoint"], env_index) for env_index in range(10)]
    rendered = replay["context_frames"]["cal"] + replay["branch_frames"]
    assert replay["native_render_calls"] == 13
    assert len(camera.poses) == len(safe_pose_inputs) == len(rendered) == 13
    for record, (components, env_index), (position, quat_xyzw), camera_pose in zip(
        rendered, expected_sources, safe_pose_inputs, camera.poses, strict=True
    ):
        expected_position = np.asarray(components["base_pos_world"])[env_index]
        expected_wxyz = np.asarray(components["base_quat_wxyz"])[env_index]
        np.testing.assert_array_equal(position, expected_position)
        np.testing.assert_array_equal(quat_xyzw, expected_wxyz[[1, 2, 3, 0]])
        np.testing.assert_array_equal(camera_pose["pos"], expected_position)
        np.testing.assert_array_equal(
            record["source_base_position_xyz_m"], expected_position
        )
        np.testing.assert_array_equal(
            record["source_base_quaternion_wxyz"], expected_wxyz
        )
        encoded = int(round((float(expected_position[0]) + 5.0) * 20.0))
        assert np.all(record["stored_rgb"] == encoded)

    camera._is_batched = True
    with pytest.raises(pilot.PilotContractError, match="must be non-batched"):
        collector._capture_replayed_frame(
            render_build,
            components=trial["branch_endpoint"],
            env_index=0,
            safe_camera_pose_from_base=safe_pose,
            camera_safety_config_from_pack=lambda _pack: object(),
            effective_camera_mount_xyz_rpy=lambda _pack: (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            assess_rendered_frame=lambda *_args, **_kwargs: {
                "valid": True,
                "invalid_reasons": [],
            },
            stage_wall_times=collector._new_stage_wall_times(),
        )


def test_collector_failure_receipt_preserves_json_diagnostics() -> None:
    collector = _load_collector()
    error = pilot.PilotDiagnosticError(
        "fixture divergence",
        diagnostics={
            "phase": "common_history_policy_step",
            "sim_time_ns": 20_000_000,
            "max_abs_difference": 1.0e-7,
        },
    )
    assert collector._failure_receipt(error) == {
        "type": "PilotDiagnosticError",
        "message": "fixture divergence",
        "diagnostics": {
            "max_abs_difference": 1.0e-7,
            "phase": "common_history_policy_step",
            "sim_time_ns": 20_000_000,
        },
    }


def test_determinism_probe_binds_and_sanitizes_parallelization_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_determinism_probe()
    monkeypatch.setenv("GS_PARA_LEVEL", "ambient")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "ambient")
    plan = {"execution_contract": {"environment": pilot.EXECUTION_ENVIRONMENT}}
    selected = probe._configure_environment(plan, para_level=0)
    assert selected == {**pilot.EXECUTION_ENVIRONMENT, "GS_PARA_LEVEL": "0"}
    assert os.environ["GS_PARA_LEVEL"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_collector_uses_the_real_safety_limits_factory() -> None:
    collector = _load_collector()
    actual_safety_limits = collector._runtime_imports()["SafetyLimits"]
    calls: dict[str, Any] = {}

    class FakePolicy:
        def __init__(self, **kwargs: Any) -> None:
            calls["policy"] = kwargs

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls["config"] = kwargs

    class FakeRunner:
        _policy_steps_per_command_tick = 5
        _physics_steps_per_policy = 10

        def __init__(
            self,
            build: Any,
            policy: Any,
            registry: Any,
            safety_limits: Any,
            *,
            config: Any,
        ) -> None:
            calls["runner"] = (build, policy, registry, safety_limits, config)

    runtime = {
        "GenesisGo2PPOPolicy": FakePolicy,
        "RolloutConfig": FakeConfig,
        "RolloutRunner": FakeRunner,
        "SafetyLimits": actual_safety_limits,
    }
    plan = {
        "runtime_bindings": {
            "policy_checkpoint": {"path": "/fixture/model.pt"},
            "policy_config": {"path": "/fixture/cfgs.pkl"},
        },
        "execution_contract": {
            "policy_device": "cpu",
            "fall_z_threshold_m": 0.15,
            "tip_threshold_rad": math.pi / 3.0,
            "seed": 20260731,
        },
    }
    platform = {
        "locomotion": {
            "safety": {
                "min_vx_mps": -0.3,
                "max_vx_mps": 0.4,
                "min_vy_mps": -0.2,
                "max_vy_mps": 0.2,
                "max_yaw_rate_radps": 0.5,
                "max_command_delta_per_tick": {
                    "vx_mps": 0.1,
                    "vy_mps": 0.1,
                    "yaw_rate_radps": 0.2,
                },
            }
        }
    }
    runner = collector._build_rollout_runner(
        plan=plan,
        runtime=runtime,
        platform=platform,
        build="build",
        registry="registry",
    )
    safety_limits = calls["runner"][3]
    assert runner.policy_steps_per_command_tick == 5
    assert safety_limits.min_vx_mps == -0.3
    assert safety_limits.max_vx_mps == 0.4
    assert safety_limits.max_yaw_rate_radps == 0.5
    assert calls["policy"]["device"] == "cpu"
    assert calls["policy"]["deduplicate_exact_observation_rows"] is True

    runtime["SafetyLimits"] = type(
        "WrongSafetyLimits", (), {"from_platform_manifest": classmethod(lambda cls, value: value)}
    )
    with pytest.raises(pilot.PilotContractError, match="from_manifest factory"):
        collector._build_rollout_runner(
            plan=plan,
            runtime=runtime,
            platform=platform,
            build="build",
            registry="registry",
        )


def _test_ppo_policy(
    model: Any, *, deduplicate: bool, simulate_action_latency: bool
) -> Any:
    GenesisGo2PPOPolicy = _load_collector()._runtime_imports()[
        "GenesisGo2PPOPolicy"
    ]

    policy = GenesisGo2PPOPolicy.__new__(GenesisGo2PPOPolicy)
    policy._device = "cpu"
    policy._policy = model
    policy._last_actions = None
    policy.simulate_action_latency = simulate_action_latency
    policy.deduplicate_exact_observation_rows = deduplicate
    policy.policy_joint_names = tuple(f"joint-{index}" for index in range(12))
    policy.obs_scales = {"ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 1.0}
    policy.command_scale = np.ones(3, dtype=np.float32)
    policy.default_dof_pos_policy = np.zeros(12, dtype=np.float32)
    policy._policy_from_rollout = np.arange(12, dtype=np.int64)
    policy._rollout_from_policy = np.arange(12, dtype=np.int64)
    policy.action_scale = 1.0
    return policy


def _test_policy_observation(commands: np.ndarray) -> dict[str, np.ndarray]:
    n_envs = int(commands.shape[0])
    return {
        "command": np.asarray(commands, dtype=np.float32),
        "base_quat_xyzw": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_envs, 1)
        ),
        "base_ang_vel_world": np.zeros((n_envs, 3), dtype=np.float32),
        "joint_pos": np.zeros((n_envs, 12), dtype=np.float32),
        "joint_vel": np.zeros((n_envs, 12), dtype=np.float32),
    }


def test_counterfactual_policy_broadcasts_mixed_exact_duplicate_rows() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    calls: list[tuple[int, bool]] = []

    class RowSensitivePolicy:
        def __call__(self, observation: Any, *, stochastic_output: bool) -> Any:
            policy_input = observation["policy"]
            calls.append((int(policy_input.shape[0]), stochastic_output))
            row_offset = torch.arange(
                policy_input.shape[0],
                dtype=policy_input.dtype,
                device=policy_input.device,
            )[:, None]
            return policy_input[:, :12] + row_offset * 1.0e-6

    policy = _test_ppo_policy(
        RowSensitivePolicy(), deduplicate=True, simulate_action_latency=False
    )
    commands = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    actions = policy.act(_test_policy_observation(commands))

    assert calls == [(1, False), (1, False), (1, False)]
    assert np.array_equal(actions[0], actions[1])
    assert np.array_equal(actions[3], actions[4])
    assert not np.array_equal(actions[0], actions[2])
    assert not np.array_equal(actions[2], actions[3])


def test_ppo_policy_default_path_retains_single_batched_call() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    calls: list[tuple[int, bool]] = []

    class RowSensitivePolicy:
        def __call__(self, observation: Any, *, stochastic_output: bool) -> Any:
            policy_input = observation["policy"]
            calls.append((int(policy_input.shape[0]), stochastic_output))
            row_offset = torch.arange(
                policy_input.shape[0],
                dtype=policy_input.dtype,
                device=policy_input.device,
            )[:, None]
            return policy_input[:, :12] + row_offset * 1.0e-6

    policy = _test_ppo_policy(
        RowSensitivePolicy(), deduplicate=False, simulate_action_latency=False
    )
    actions = policy.act(
        _test_policy_observation(np.zeros((4, 3), dtype=np.float32))
    )

    assert calls == [(4, False)]
    assert not np.array_equal(actions[0], actions[1])


def test_counterfactual_policy_preserves_latency_state_across_exact_rows() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    calls: list[tuple[int, bool]] = []

    class AdvancingPolicy:
        def __call__(self, observation: Any, *, stochastic_output: bool) -> Any:
            policy_input = observation["policy"]
            calls.append((int(policy_input.shape[0]), stochastic_output))
            return torch.full(
                (policy_input.shape[0], 12),
                float(len(calls)),
                dtype=policy_input.dtype,
                device=policy_input.device,
            )

    policy = _test_ppo_policy(
        AdvancingPolicy(), deduplicate=True, simulate_action_latency=True
    )
    observation = _test_policy_observation(np.zeros((10, 3), dtype=np.float32))
    first_actions = policy.act(observation)
    second_actions = policy.act(observation)

    assert calls == [(1, False), (1, False)]
    assert np.array_equal(first_actions, np.zeros((10, 12), dtype=np.float32))
    assert np.array_equal(second_actions, np.ones((10, 12), dtype=np.float32))
    assert np.array_equal(policy._last_actions, np.full((10, 12), 2.0, dtype=np.float32))


def test_collector_rejects_unbound_genesis_parallelization_level() -> None:
    collector = _load_collector()
    build = type("Build", (), {"scene": type("Scene", (), {"_para_level": 2})()})()
    execution = {"environment": {"GS_PARA_LEVEL": "0"}}

    with pytest.raises(
        pilot.PilotContractError, match="observed 2, expected 0"
    ):
        collector._require_scene_parallelization(
            build=build, execution=execution
        )

    build.scene._para_level = 0
    collector._require_scene_parallelization(build=build, execution=execution)


def test_lockstep_trial_calibration_sentinel_and_live_render() -> None:
    runner = _FakeRunner(10)
    states = [_state("cal", "calibration", 0, 0)]
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=states,
        action_blocks=[_block(index) for index in range(9)],
        capture_components=runner.capture,
        capture_sim_time_ns=lambda: runner.time_ns,
        capture_render_batch=lambda: _render_batch(runner),
    )
    assert len(trial["trajectory_samples"]) == 25
    assert len(trial["render_batches"]) == 4
    assert trial["synchronization_audits"][0]["passed"] is True
    sentinel = trial["sentinel_audits"][0]
    assert sentinel["action_id"] == 6
    assert sentinel["physics_equal"] is True


def test_lockstep_trial_preserves_first_common_history_divergence() -> None:
    runner = _FirstStepDivergenceRunner(10)
    with pytest.raises(pilot.PilotDiagnosticError) as raised:
        pilot.execute_lockstep_trial(
            runner=runner,
            states=[_state("cal", "calibration", 0, 0)],
            action_blocks=[_block(index) for index in range(9)],
            capture_components=runner.capture,
            capture_sim_time_ns=lambda: runner.time_ns,
        )
    diagnostic = raised.value.diagnostics
    assert diagnostic["phase"] == "common_history_policy_step"
    assert diagnostic["history_index"] == 0
    assert diagnostic["command_tick_index"] == 0
    assert diagnostic["policy_step_index"] == 0
    assert diagnostic["block_policy_step_index"] == 0
    assert diagnostic["sim_time_ns"] == 20_000_000
    audit = diagnostic["synchronization_audits"][0]
    assert audit["passed"] is False
    assert audit["components"]["qpos"]["max_abs_difference"] > 0.0


def test_lockstep_trial_train_eval_have_no_sentinel() -> None:
    runner = _FakeRunner(18)
    states = [_state("train", "train", 0, 0), _state("eval", "eval", 1, 0)]
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=states,
        action_blocks=[_block(index) for index in range(9)],
        capture_components=runner.capture,
        capture_sim_time_ns=lambda: runner.time_ns,
    )
    assert trial["sentinel_audits"] == []
    assert [row["lane_count"] for row in trial["synchronization_audits"]] == [9, 9]


def test_live_render_must_not_mutate_state() -> None:
    runner = _FakeRunner(10)

    def bad_render() -> dict[str, Any]:
        runner.values[0] += 1.0
        return _render_batch(runner)

    with pytest.raises(pilot.PilotContractError, match="render mutated"):
        pilot.execute_lockstep_trial(
            runner=runner,
            states=[_state("cal", "calibration", 0, 0)],
            action_blocks=[_block(index) for index in range(9)],
            capture_components=runner.capture,
            capture_sim_time_ns=lambda: runner.time_ns,
            capture_render_batch=bad_render,
        )


def _authority(
    plan: dict[str, Any], plan_binding: dict[str, Any], source_binding: dict[str, Any]
) -> dict[str, Any]:
    sources = [
        {"name": name, "binding": pilot.file_binding(ROOT / relative)}
        for name, relative in pilot.AUTHORITY_SOURCE_PATHS
    ]
    return {
        "schema": pilot.SMOKE_AUTHORITY_SCHEMA,
        "status": "AUTHORIZED_ONE_EXACT_SOURCE_INTEGRATION_SMOKE",
        "authority_granted_by_this_document": True,
        "scientific_claim_authorized": False,
        "authorizer": {"identity": "/root/reviewer", "basis": "source-only review"},
        "issued_at": "2026-07-31T12:00:00+01:00",
        "source_commit": "1" * 40,
        "review_binding": source_binding,
        "plan_binding": plan_binding,
        "source_bindings": sources,
        "attempt": {
            "id": plan["attempt_id"],
            "root": plan["output_root"],
            "maximum_attempts": 1,
            "must_be_absent": True,
            "reservation_consumes_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
        "caps": {
            "scenes": 1,
            "states": 1,
            "candidate_branches": 9,
            "sentinel_branches": 1,
            "total_branches": 10,
            "candidate_branch_simulated_seconds": 5.0,
            "total_lane_simulated_seconds_including_common_prefix": 15.0,
            "policy_steps_per_lane": 75,
            "total_lane_policy_steps": 750,
            "total_lane_physics_steps": 7500,
            "native_render_calls": 13,
            "stored_rgb_frames": 13,
            "wall_seconds": 600.0,
        },
        "runtime_bindings": plan["runtime_bindings"],
        "execution": plan["execution_contract"],
        "network_access": False,
        "platform_gate_disposition": dict(pilot.PLATFORM_GATE_DISPOSITION),
        "external_supervisor": {
            "source_binding": next(
                row["binding"]
                for row in sources
                if row["name"] == "external_supervisor"
            ),
            "terminal_reviewer": "/root/terminal_reviewer",
        },
    }


def test_authority_and_review_are_semantic_not_hash_only(tmp_path: Path) -> None:
    plan = pilot.validate_plan(_smoke_plan(tmp_path))
    plan_binding = _bound_file(tmp_path / "plan.json", b"plan")
    source_binding = _bound_file(tmp_path / "source.py", b"source")
    authority = pilot.validate_authority(
        _authority(plan, plan_binding, source_binding),
        plan=plan,
        plan_binding=plan_binding,
    )
    review = {
        "schema": pilot.SOURCE_REVIEW_SCHEMA,
        "status": "PASS_SOURCE_ONLY_NOT_AUTHORITY",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": authority["source_commit"],
        "reviewed_source_bindings": authority["source_bindings"],
        "remaining_findings": [],
        "reviewer": {
            "identity": "/root/independent-reviewer",
            "independence_basis": "separate read-only pre-freeze review",
        },
        "reviewed_at": "2026-07-31T13:00:00+01:00",
        "review_method": ["source closure and fail-closed contract review"],
        "test_evidence": ["focused source-only tests passed"],
        "accepted_limitations": ["non-citable source-integration smoke only"],
    }
    assert pilot.validate_source_review(review, authority=authority) == review
    bad = copy.deepcopy(authority)
    bad["caps"]["stored_rgb_frames"] = 12
    with pytest.raises(pilot.PilotContractError):
        pilot.validate_authority(bad, plan=plan, plan_binding=plan_binding)


def test_authority_rejects_protected_and_wrong_source_paths_before_open(
    tmp_path: Path,
) -> None:
    plan = pilot.validate_plan(_smoke_plan(tmp_path))
    plan_binding = _bound_file(tmp_path / "plan.json", b"plan")
    review_binding = _bound_file(tmp_path / "review.json", b"review")
    authority = _authority(plan, plan_binding, review_binding)

    protected = copy.deepcopy(authority)
    protected["source_bindings"][0]["binding"]["path"] = str(
        tmp_path / "sealed_test.json"
    )
    with pytest.raises(pilot.PilotContractError, match="custody-protected"):
        pilot.validate_authority(
            protected, plan=plan, plan_binding=plan_binding
        )

    wrong = copy.deepcopy(authority)
    wrong["source_bindings"][0]["binding"]["path"] = str(
        tmp_path / "ordinary.py"
    )
    with pytest.raises(pilot.PilotContractError, match="exact reviewed path"):
        pilot.validate_authority(wrong, plan=plan, plan_binding=plan_binding)


def test_exclusive_writer_and_fresh_root(tmp_path: Path) -> None:
    root = pilot.fresh_development_output_root(
        tmp_path / "dev" / "attempt", development_root=tmp_path / "dev"
    )
    path = root / "receipt.json"
    pilot.write_json_exclusive(path, {"finite": 1.0})
    with pytest.raises(FileExistsError):
        pilot.write_json_exclusive(path, {"finite": 2.0})
    with pytest.raises(pilot.PilotContractError):
        pilot.fresh_development_output_root(root, development_root=tmp_path / "dev")


def test_group_receipt_writes_13_immutable_rgb_frames(tmp_path: Path) -> None:
    collector = _load_collector()
    plan = pilot.validate_plan(_smoke_plan(tmp_path))
    Path(plan["output_root"]).mkdir(parents=True)
    scene_dir = Path(plan["output_root"]) / "generated_scene" / "scene"
    manifest = _bound_file(scene_dir / "manifest.json", b"{}")
    genesis = _bound_file(scene_dir / "genesis_scene.json", b"{}")
    plan["states"][0]["scene_manifest_binding"] = manifest
    plan["states"][0]["scene_genesis_binding"] = genesis
    runner = _YawFakeRunner(10, yaw_rad=math.pi / 2.0)
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=plan["states"],
        action_blocks=[_block(index) for index in range(9)],
        capture_components=runner.capture,
        capture_sim_time_ns=lambda: runner.time_ns,
    )
    trial["render_replay"] = _render_replay(trial, plan["states"])
    stage_wall_times = collector._new_stage_wall_times()
    receipts, frames, quality, render_sentinel = collector._group_trial_receipts(
        plan=plan,
        states=plan["states"],
        trial=trial,
        rgb_root=Path(plan["output_root"]) / "rgb",
        stage_wall_times=stage_wall_times,
    )
    assert len(receipts) == 1
    assert len(frames) == 13
    assert len(quality) == 13
    assert render_sentinel[0]["passed"] is True
    assert receipts[0]["state"]["lane_start"] == 0
    assert receipts[0]["state"]["lane_count"] == 10
    assert receipts[0]["branches"][9]["duplicates_candidate_action_id"] == 6
    assert all(np.isfinite(row["physical_target_progress_m"]) for row in receipts[0]["branches"])
    pose = receipts[0]["context"]["prebranch_base_pose_world"]
    assert pose["quaternion_wxyz"][3] > 0.7
    assert receipts[0]["context"]["target_relative_body_xy_m"] == (
        pilot.target_world_to_body_xy(
            target_xy_m=plan["states"][0]["target_xy_m"],
            base_position_xyz_m=pose["position_xyz_m"],
            base_quaternion_wxyz=pose["quaternion_wxyz"],
        )
    )
    assert stage_wall_times["png_encode_write_hash_wall_seconds"] > 0.0


def test_world_target_to_body_xy_uses_inverse_nonzero_yaw() -> None:
    half_sqrt_two = math.sqrt(0.5)
    relative = pilot.target_world_to_body_xy(
        target_xy_m=[1.0, 3.0],
        base_position_xyz_m=[1.0, 2.0, 0.4],
        base_quaternion_wxyz=[half_sqrt_two, 0.0, 0.0, half_sqrt_two],
    )
    assert relative == pytest.approx([1.0, 0.0], abs=1.0e-12)


def test_two_preexisting_scenes_keep_absolute_input_bindings_in_receipts(
    tmp_path: Path,
) -> None:
    collector = _load_collector()
    plan = pilot.validate_plan(_bounded_two_scene_plan(tmp_path))
    Path(plan["output_root"]).mkdir(parents=True)
    observed: list[dict[str, Any]] = []
    for state in plan["states"]:
        runner = _FakeRunner(9)
        trial = pilot.execute_lockstep_trial(
            runner=runner,
            states=[state],
            action_blocks=[_block(index) for index in range(9)],
            capture_components=runner.capture,
            capture_sim_time_ns=lambda: runner.time_ns,
        )
        trial["render_replay"] = _render_replay(trial, [state])
        stage_wall_times = collector._new_stage_wall_times()
        receipts, frames, quality, render_sentinel = collector._group_trial_receipts(
            plan=plan,
            states=[state],
            trial=trial,
            rgb_root=(
                Path(plan["output_root"])
                / "scenes"
                / state["role"]
                / state["scene_id"]
                / "rgb"
            ),
            stage_wall_times=stage_wall_times,
        )
        assert len(receipts) == 1
        assert len(frames) == 12
        assert len(quality) == 12
        assert render_sentinel == []
        observed.append(receipts[0])
        for binding_name in (
            "scene_manifest_binding",
            "scene_genesis_binding",
        ):
            assert receipts[0]["state"][binding_name] == state[binding_name]
            assert Path(receipts[0]["state"][binding_name]["path"]).is_absolute()
            assert collector._scene_receipt_binding(
                state,
                binding_name=binding_name,
                output_root=Path(plan["output_root"]),
            ) == state[binding_name]
    assert [receipt["state"]["lane_start"] for receipt in observed] == [0, 9]


def test_runtime_version_receipt_is_strict_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _load_collector()
    versioned_module = lambda version: type(  # noqa: E731
        "VersionedModule", (), {"__version__": version}
    )()
    monkeypatch.setattr(collector.platform, "python_version", lambda: "3.11.9")
    monkeypatch.setitem(sys.modules, "genesis", versioned_module("0.3.3"))
    monkeypatch.setitem(sys.modules, "torch", versioned_module("2.5.1+rocm6.2"))
    monkeypatch.setitem(sys.modules, "PIL", versioned_module("10.4.0"))
    monkeypatch.setattr(collector.np, "__version__", "1.26.4")
    assert collector._capture_runtime_versions() == {
        "python": "3.11.9",
        "genesis": "0.3.3",
        "torch": "2.5.1+rocm6.2",
        "numpy": "1.26.4",
        "pillow": "10.4.0",
    }


def test_importing_collector_does_not_import_genesis_or_torch() -> None:
    before = set(sys.modules)
    _load_collector()
    added = set(sys.modules) - before
    assert "genesis" not in added
    assert "torch" not in added


def test_source_only_smoke_scene_derivation_is_deterministic() -> None:
    collector = _load_collector()
    metadata = collector.derive_source_integration_smoke_scene(
        family="open_obstacle_field", plan_seed=17
    )
    assert metadata == {
        "family": "open_obstacle_field",
        "split": "calibration_smoke",
        "plan_seed": 17,
        "scene_index": 0,
        "scene_id": "open_obstacle_field_e868c98c843f",
        "scene_seed": 8950652255921294934,
        "scene_seed_salt": 0,
        "target_landmark_id": "landmark_blue",
        "target_xy_m": [4.4, -4.4],
    }


def test_authority_source_paths_exactly_cover_source_only_smoke_imports() -> None:
    collector = _load_collector()
    assert tuple(collector.EXPECTED_SOURCE_PATHS) == pilot.AUTHORITY_SOURCE_NAMES
    bound_paths = set(collector.EXPECTED_SOURCE_PATHS.values())
    assert all((ROOT / relative).is_file() for relative in bound_paths)
    assert all(
        not ({"sealed", "heldout", "held_out"} & set(Path(relative).parts))
        for relative in bound_paths
    )

    probe = r'''
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for package_root in (root, root / "lewm_genesis", root / "lewm_worlds"):
    sys.path.insert(0, str(package_root))

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

collector = load(
    "wm_source_closure_collector",
    "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
)
collector.derive_source_integration_smoke_scene(
    family="open_obstacle_field", plan_seed=20260731
)
collector._runtime_imports()
import lewm_genesis.collectors
load(
    "wm_source_closure_checker",
    "scripts/check_go2_world_model_counterfactual_pilot_v1.py",
)
load(
    "wm_source_closure_supervisor",
    "scripts/run_go2_world_model_counterfactual_smoke_authorized_v1.py",
)

paths = {
    "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "scripts/check_go2_world_model_counterfactual_pilot_v1.py",
    "scripts/run_go2_world_model_counterfactual_smoke_authorized_v1.py",
}
for module in tuple(sys.modules.values()):
    file_name = getattr(module, "__file__", None)
    if not file_name:
        continue
    path = Path(file_name).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        continue
    if relative.suffix != ".py" or relative.parts[0] not in {
        "lewm", "lewm_genesis", "lewm_worlds", "scripts"
    }:
        continue
    paths.add(relative.as_posix())
print(json.dumps(sorted(paths)))
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "lewm_genesis"), str(ROOT / "lewm_worlds"))
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(ROOT)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_paths = set(json.loads(completed.stdout))
    assert loaded_paths == bound_paths


def write_synthetic_complete_smoke_tree(base: Path) -> Path:
    """Write a complete source-only producer tree for checker integration tests."""

    collector = _load_collector()
    root = Path(base).resolve() / "synthetic_smoke_receipt"
    plan = _smoke_plan(root.parent)
    plan["output_root"] = str(root)
    plan = pilot.validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    plan_binding = pilot.write_json_exclusive(
        Path(base).resolve()
        / "synthetic_smoke_authority"
        / "external_authorized_plan.json",
        plan,
    )
    plan_receipt_abs = pilot.write_json_exclusive(
        root / "authorized_plan.json", plan
    )
    plan_receipt_binding = collector._relative_output_binding(
        plan_receipt_abs, output_root=root
    )
    scene_dir = root / "generated_scene" / "calibration_smoke" / "scene"
    manifest_abs = _bound_file(scene_dir / "manifest.json", b"{}")
    genesis_abs = _bound_file(scene_dir / "genesis_scene.json", b"{}")
    plan["states"][0]["scene_manifest_binding"] = manifest_abs
    plan["states"][0]["scene_genesis_binding"] = genesis_abs
    runner = _FakeRunner(10)
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=plan["states"],
        action_blocks=[_block(index) for index in range(9)],
        capture_components=runner.capture,
        capture_sim_time_ns=lambda: runner.time_ns,
    )
    trial["render_replay"] = _render_replay(trial, plan["states"])
    stage_wall_times = collector._new_stage_wall_times()
    receipts, frames, quality, render_sentinel = collector._group_trial_receipts(
        plan=plan,
        states=plan["states"],
        trial=trial,
        rgb_root=root / "scenes/calibration/synthetic/rgb",
        stage_wall_times=stage_wall_times,
    )
    render_receipt = {
        "schema": collector.LIVE_RENDER_RECEIPT_SCHEMA,
        "attempt_id": plan["attempt_id"],
        "status": "RENDER_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "scene": {
            "role": "calibration",
            "scene_id": plan["states"][0]["scene_id"],
            "family": plan["states"][0]["family"],
            "scene_manifest_binding": collector._scene_receipt_binding(
                plan["states"][0],
                binding_name="scene_manifest_binding",
                output_root=root,
            ),
            "scene_genesis_binding": collector._scene_receipt_binding(
                plan["states"][0],
                binding_name="scene_genesis_binding",
                output_root=root,
            ),
        },
        "render_contract": dict(pilot.RENDER_CONTRACT),
        "native_render_calls": 13,
        "stored_rgb_frames": 13,
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": "solid_materials_box_physics_preserved",
        "visual_domain_fidelity_claimed": False,
        "frame_receipts": frames,
        "quality_audits": quality,
        "render_sentinel_audits": render_sentinel,
    }
    render_abs = pilot.write_json_exclusive(root / "live_render_receipt.json", render_receipt)
    render_binding = collector._relative_output_binding(render_abs, output_root=root)
    receipts[0]["render_receipt_binding"] = render_binding
    state_abs = pilot.write_json_exclusive(root / "state_receipt.json", receipts[0])
    state_binding = collector._relative_output_binding(state_abs, output_root=root)

    support_bindings: dict[str, dict[str, Any]] = {}
    for name in ("authority", "review", "reservation", "source"):
        absolute = _bound_file(root / f"{name}.json", b"{}\n")
        support_bindings[name] = collector._relative_output_binding(
            absolute, output_root=root
        )
    source_bindings = [
        {"name": name, "binding": support_bindings["source"]}
        for name in pilot.AUTHORITY_SOURCE_NAMES
    ]
    counts = dict(plan["expected_counts"])
    aggregate = {
        "schema": pilot.PHYSICS_RESULT_SCHEMA,
        "attempt_id": plan["attempt_id"],
        "purpose": plan["purpose"],
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "plan_binding": plan_binding,
        "plan_receipt_binding": plan_receipt_binding,
        "authority_binding": support_bindings["authority"],
        "review_binding": support_bindings["review"],
        "reservation_binding": support_bindings["reservation"],
        "caps": {
            "scenes": 1,
            "states": 1,
            "candidate_branches": 9,
            "sentinel_branches": 1,
            "total_branches": 10,
            "candidate_branch_simulated_seconds": 5.0,
            "total_lane_simulated_seconds_including_common_prefix": 15.0,
            "policy_steps_per_lane": 75,
            "total_lane_policy_steps": 750,
            "total_lane_physics_steps": 7500,
            "native_render_calls": 13,
            "stored_rgb_frames": 13,
            "wall_seconds": 600.0,
        },
        "execution_contract": dict(plan["execution_contract"]),
        "runtime_versions": {
            "python": "3.11.9",
            "genesis": "0.3.3",
            "torch": "2.5.1+rocm6.2",
            "numpy": "1.26.4",
            "pillow": "10.4.0",
        },
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": source_bindings,
        "expected_counts": counts,
        "observed_counts": counts,
        "scene_materialization": {
            "declaration": dict(plan["states"][0]["scene_generation"]),
            "scene_manifest_binding": collector._relative_output_binding(
                manifest_abs, output_root=root
            ),
            "scene_genesis_binding": collector._relative_output_binding(
                genesis_abs, output_root=root
            ),
            "scene_seed": 1,
            "scene_seed_salt": 0,
            "target_landmark_id": "landmark_blue",
        },
        "state_receipt_bindings": [state_binding],
        "render_receipt_bindings": [render_binding],
        "scene_metrics": [
            {
                "scene_id": plan["states"][0]["scene_id"],
                "family": plan["states"][0]["family"],
                "role": "calibration",
                "states": 1,
                "envs": 10,
                "physics_build_wall_seconds": 0.1,
                "physics_simulation_wall_seconds": 0.1,
                "render_scene_build_wall_seconds": 0.02,
                "native_render_wall_seconds": 0.02,
                "camera_quality_resize_wall_seconds": 0.01,
                "png_encode_write_hash_wall_seconds": 0.02,
                "lockstep_execution_wall_seconds": 0.1,
                "post_lockstep_receipt_wall_seconds": 0.05,
                "scene_pipeline_wall_seconds": 0.25,
                "scene_total_wall_seconds": 0.4,
                "native_render_calls": 13,
                "stored_rgb_frames": 13,
                "depth_rendered": True,
                "depth_persisted": False,
                "visual_mode": "solid_materials_box_physics_preserved",
            }
        ],
        "visual_domain_limitation": (
            "solid materials and box primitives preserve the physics geometry; "
            "this smoke does not establish final visual-domain fidelity"
        ),
        "collection_wall_seconds": 0.4,
        "failure": None,
    }
    result_path = root / "physics_result.json"
    pilot.write_json_exclusive(result_path, aggregate)
    return result_path


def test_synthetic_complete_smoke_tree_has_relative_output_bindings(
    tmp_path: Path,
) -> None:
    result_path = write_synthetic_complete_smoke_tree(tmp_path)
    result = json.loads(result_path.read_text())
    for key in (
        "authority_binding",
        "review_binding",
        "reservation_binding",
    ):
        assert not Path(result[key]["path"]).is_absolute()
    assert Path(result["plan_binding"]["path"]).is_absolute()
    assert not Path(result["plan_receipt_binding"]["path"]).is_absolute()
    assert result["plan_binding"]["file_sha256"] == result[
        "plan_receipt_binding"
    ]["file_sha256"]
    assert result["plan_binding"]["byte_count"] == result[
        "plan_receipt_binding"
    ]["byte_count"]
    assert not Path(result["state_receipt_bindings"][0]["path"]).is_absolute()
    assert not Path(result["render_receipt_bindings"][0]["path"]).is_absolute()
