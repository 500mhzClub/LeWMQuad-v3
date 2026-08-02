from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as producer


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_go2_world_model_counterfactual_pilot_v1.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_pilot_checker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

PRODUCER_TEST_PATH = ROOT / "lewm/tests/test_go2_world_model_counterfactual_pilot_v1.py"
PRODUCER_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_pilot_producer_test_support", PRODUCER_TEST_PATH
)
assert PRODUCER_SPEC is not None and PRODUCER_SPEC.loader is not None
producer_test_support = importlib.util.module_from_spec(PRODUCER_SPEC)
PRODUCER_SPEC.loader.exec_module(producer_test_support)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _inert(path: str) -> dict[str, Any]:
    return {"path": path, "file_sha256": hashlib.sha256(path.encode()).hexdigest(), "byte_count": 1}


def _write_json(root: Path, relative: str, value: object) -> dict[str, Any]:
    raw = _canonical(value) + b"\n"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": relative, "file_sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}


def _block(action_id: int) -> list[list[float]]:
    return [list(command) for command in producer.CANONICAL_ACTION_BLOCKS[action_id]]


def _plan(root: Path) -> dict[str, Any]:
    states = [{
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
            "scene_generator_binding": _inert("/synthetic/collector.py"),
        },
        "group_index": 0,
        "state_index_in_scene": 0,
        "history_action_ids": [6, 3],
        "candidate_action_ids": list(range(9)),
        "sentinel_duplicate_action_id": 6,
        "target_xy_m": [1.0, 2.0],
    }]
    runtime_names = (
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
    plan = {
        "schema": producer.PLAN_SCHEMA,
        "attempt_id": "smoke-attempt-1",
        "purpose": "source_integration_smoke",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": producer.BRANCH_MECHANISM,
        "states_per_scene": 1,
        "history_blocks": 2,
        "output_root": str(root.resolve()),
        "runtime_bindings": {name: _inert(f"/synthetic/runtime/{name}") for name in runtime_names},
        "execution_contract": {
            "backend": "vulkan",
            "policy_device": "cpu",
            "seed": 7,
            "fall_z_threshold_m": 0.15,
            "tip_threshold_rad": 1.0,
            "policy_steps_per_command_tick": 5,
            "python_invocation_path": "/synthetic/venv/bin/python",
            "environment": dict(producer.EXECUTION_ENVIRONMENT),
            "graphics_preflight": dict(
                producer.GRAPHICS_PREFLIGHT_EXPECTATION
            ),
        },
        "render_contract": dict(producer.RENDER_CONTRACT),
        "action_catalog": [
            {"action_id": action, "name": name, "requested_block": _block(action)}
            for action, name in enumerate(producer.CANONICAL_ACTIONS)
        ],
        "states": states,
        "expected_counts": producer.expected_counts_from_states(states),
    }
    return producer.validate_plan(plan)


def _sync() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "state_id": "smoke-state-0",
        "group_index": 0,
        "lane_start": 0,
        "lane_count": 10,
        "exact_equality_required": True,
        "passed": True,
        "prebranch_state_sha256": digest,
        "lane_state_sha256s": [digest] * 10,
        "components": {
            name: {"exact_equal": True, "max_abs_difference": 0.0, "shape_per_lane": [1]}
            for name in producer.SYNC_COMPONENTS
        },
    }


def _frame(identity: str) -> dict[str, Any]:
    path = f"rgb/{identity.replace(':', '.')}.png"
    return {
        "artifact_id": identity,
        "frame_identity": identity,
        "path": path,
        "file_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "byte_count": 100,
        "width": 224,
        "height": 224,
        "mode": "RGB",
        "format": "PNG",
        "camera_valid": True,
        "low_information": False,
        "low_info_reasons": [],
    }


def _trajectory(delta_x: float) -> list[dict[str, Any]]:
    return [
        {
            "policy_step_index": step,
            "timestamp_ns": 1_020_000_000 + step * 20_000_000,
            "base_pos_world": [delta_x * (step + 1) / 25.0, 0.0, 0.4],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_lin_vel_world": [0.0, 0.0, 0.0],
            "base_ang_vel_world": [0.0, 0.0, 0.0],
            "leg_joint_pos": [0.0] * 12,
            "leg_joint_vel": [0.0] * 12,
        }
        for step in range(25)
    ]


def _endpoint(delta_x: float) -> dict[str, Any]:
    return {
        "qpos": [0.0] * 19,
        "dofs_velocity": [0.0] * 18,
        "base_pos_world": [delta_x, 0.0, 0.4],
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "base_lin_vel_world": [0.0, 0.0, 0.0],
        "base_ang_vel_world": [0.0, 0.0, 0.0],
        "leg_joint_pos": [0.0] * 12,
        "leg_joint_vel": [0.0] * 12,
        "runner_last_executed": [0.0, 0.0, 0.0],
        "policy_last_actions": [0.0] * 12,
    }


def _fixture(
    root: Path,
    *,
    mutate_state: Callable[[dict[str, Any]], None] | None = None,
    mutate_render: Callable[[dict[str, Any]], None] | None = None,
    mutate_collection: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, str, int]:
    plan = _plan(root)
    external_plan_binding = _write_json(
        root.parent,
        f"{root.name}.external_authorized_plan.json",
        plan,
    )
    external_plan_binding["path"] = str(
        (root.parent / external_plan_binding["path"]).resolve()
    )
    plan_receipt_binding = _write_json(
        root, "receipts/authorized_plan.json", plan
    )
    scene_manifest = _inert("/synthetic/generated_scene/manifest.json")
    scene_genesis = _inert("/synthetic/generated_scene/genesis_scene.json")
    sync = _sync()
    context_ids = [producer.render_frame_identity(state_id="smoke-state-0", frame_kind="context", index=i) for i in range(3)]
    history = [_block(6), _block(3)]
    branches: list[dict[str, Any]] = []
    frames = [_frame(identity) for identity in context_ids]
    for lane in producer.lane_layout(
        "smoke-state-0", role="calibration", state_index_in_scene=0
    ):
        offset = int(lane["lane_offset"])
        action = int(lane["action_id"])
        identity = producer.render_frame_identity(
            state_id="smoke-state-0", frame_kind=str(lane["kind"]), index=offset
        )
        frame = _frame(identity)
        if str(lane["kind"]) == "sentinel":
            frame["file_sha256"] = branches[action]["frame_receipt"][
                "file_sha256"
            ]
        frames.append(frame)
        delta_x = float(action) * 0.01
        trajectory = _trajectory(delta_x)
        target_progress = math.hypot(1.0, 2.0) - math.hypot(
            1.0 - delta_x, 2.0
        )
        branches.append({
            "lane_index": offset,
            "lane_offset": offset,
            "kind": lane["kind"],
            "action_id": action,
            "action_name": producer.CANONICAL_ACTIONS[action],
            "duplicates_candidate_action_id": lane.get("duplicates_candidate_action_id"),
            "requested_block": _block(action),
            "executed_block": _block(action),
            "executed_block_sha256": producer.canonical_block_sha256(_block(action)),
            "clipped": False,
            "trajectory_policy_step_samples": trajectory,
            "endpoint_state": _endpoint(delta_x),
            "physical_fell": False,
            "physical_tipped": False,
            "physical_path_length_m": abs(delta_x),
            "physical_target_progress_m": target_progress,
            "render_frame_identity": identity,
            "frame_receipt": frame,
        })
    trajectory_sha = "b" * 64
    sentinel = {
        "state_id": "smoke-state-0",
        "group_index": 0,
        "action_id": 6,
        "candidate_lane": 6,
        "sentinel_lane": 9,
        "policy_step_count": 25,
        "exact_equality_required": True,
        "physics_equal": True,
        "candidate_trajectory_sha256": trajectory_sha,
        "sentinel_trajectory_sha256": trajectory_sha,
        "components": {
            name: {"exact_equal": True, "max_abs_difference": 0.0}
            for name in producer.SYNC_COMPONENTS
        },
    }
    render_sha = "c" * 64
    render_sentinel = {
        "state_id": "smoke-state-0",
        "group_index": 0,
        "action_id": 6,
        "candidate_lane": 6,
        "sentinel_lane": 9,
        "exact_equality_required": True,
        "stored_rgb_equal": True,
        "candidate_stored_rgb_sha256": render_sha,
        "sentinel_stored_rgb_sha256": render_sha,
        "passed": True,
    }
    replay_source_poses = [
        {
            "position_xyz_m": [0.0, 0.0, 0.4],
            "quaternion_wxyz": [1.0000001, 0.0, 0.0, 0.0],
        }
        for _ in context_ids
    ] + [
        {
            "position_xyz_m": branch["endpoint_state"]["base_pos_world"],
            "quaternion_wxyz": branch["endpoint_state"]["base_quat_wxyz"],
        }
        for branch in branches
    ]
    quality = [
        {
            "frame_identity": frame["frame_identity"],
            "native_resolution": [640, 480],
            "camera_valid": True,
            "quality": {
                "schema": checker.COUNTERFACTUAL_QUALITY_SCHEMA,
                "retained": True,
                "hard_valid": True,
                "raw_assessment_valid": True,
                "observed_reasons": [],
                "low_information": False,
                "low_info_reasons": [],
                "hard_failure_reasons": [],
                "rgb_stats": {},
                "depth_stats": {},
            },
            "replay_pose": {
                "source_base_pose_world": pose,
                "camera_pose_world": {
                    "position_xyz_m": pose["position_xyz_m"],
                    "lookat_xyz_m": [1.0, 0.0, 0.4],
                    "up_xyz": [0.0, 0.0, 1.0],
                },
            },
        }
        for frame, pose in zip(frames, replay_source_poses, strict=True)
    ]
    render_receipt = {
        "schema": "lewm_go2_world_model_counterfactual_live_render_receipt_v1",
        "attempt_id": plan["attempt_id"],
        "status": "RENDER_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "scene": {
            "role": "calibration",
            "scene_id": "open_obstacle_field-deadbeef",
            "family": "open_obstacle_field",
            "scene_manifest_binding": scene_manifest,
            "scene_genesis_binding": scene_genesis,
        },
        "render_contract": dict(producer.RENDER_CONTRACT),
        "native_render_calls": 13,
        "stored_rgb_frames": 13,
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": "solid_materials_box_physics_preserved",
        "visual_domain_fidelity_claimed": False,
        "frame_receipts": frames,
        "quality_audits": quality,
        "render_sentinel_audits": [render_sentinel],
    }
    if mutate_render is not None:
        mutate_render(render_receipt)
    render_binding = _write_json(root, "receipts/live_render.json", render_receipt)
    state_receipt = {
        "schema": producer.STATE_RECEIPT_SCHEMA,
        "attempt_id": plan["attempt_id"],
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "state": {
            "state_id": "smoke-state-0",
            "role": "calibration",
            "family": "open_obstacle_field",
            "scene_id": "open_obstacle_field-deadbeef",
            "group_index": 0,
            "state_index_in_scene": 0,
            "lane_start": 0,
            "lane_count": 10,
            "scene_manifest_binding": scene_manifest,
            "scene_genesis_binding": scene_genesis,
            "target_xy_m": [1.0, 2.0],
        },
        "context": {
            "rgb_artifact_ids": context_ids,
            "frame_identities": context_ids,
            "history_action_ids": [6, 3],
            "history_executed_blocks": history,
            "executed_block_sha256s": [producer.canonical_block_sha256(block) for block in history],
            "endpoint_command_ticks": [0, 5, 10],
            "prebranch_state_sha256": sync["prebranch_state_sha256"],
            "prebranch_base_pose_world": {
                "position_xyz_m": [0.0, 0.0, 0.4],
                "quaternion_wxyz": [1.0000001, 0.0, 0.0, 0.0],
            },
            "context_base_pose_world_sequence": [
                {
                    "position_xyz_m": [0.0, 0.0, 0.4],
                    "quaternion_wxyz": [1.0000001, 0.0, 0.0, 0.0],
                }
                for _ in range(3)
            ],
            "target_relative_body_xy_m": [1.0, 2.0],
        },
        "synchronization_audit": sync,
        "branches": branches,
        "sentinel_audit": sentinel,
        "render_sentinel_audit": render_sentinel,
        "render_receipt_binding": render_binding,
    }
    if mutate_state is not None:
        mutate_state(state_receipt)
    state_binding = _write_json(root, "receipts/state.json", state_receipt)
    caps = {
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
    }
    source_binding = _inert("/synthetic/source.py")
    collection = {
        "schema": producer.PHYSICS_RESULT_SCHEMA,
        "attempt_id": plan["attempt_id"],
        "purpose": plan["purpose"],
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": producer.BRANCH_MECHANISM,
        "plan_binding": external_plan_binding,
        "plan_receipt_binding": plan_receipt_binding,
        "authority_binding": _inert("/synthetic/authority.json"),
        "review_binding": _inert("/synthetic/review.json"),
        "reservation_binding": _inert("/synthetic/reservation.json"),
        "caps": caps,
        "execution_contract": plan["execution_contract"],
        "runtime_bindings": plan["runtime_bindings"],
        "runtime_versions": {
            "python": "3.11.9",
            "genesis": "0.3.3",
            "torch": "2.5.1+rocm6.2",
            "numpy": "1.26.4",
            "pillow": "10.4.0",
        },
        "source_bindings": [
            {"name": name, "binding": source_binding}
            for name in ("collector", "contract", "checker")
        ],
        "expected_counts": plan["expected_counts"],
        "observed_counts": plan["expected_counts"],
        "scene_materialization": {
            "declaration": plan["states"][0]["scene_generation"],
            "scene_manifest_binding": scene_manifest,
            "scene_genesis_binding": scene_genesis,
            "scene_seed": 1,
            "scene_seed_salt": 2,
            "target_landmark_id": "synthetic-target",
        },
        "state_receipt_bindings": [state_binding],
        "render_receipt_bindings": [render_binding],
        "scene_metrics": [{
            "scene_id": "open_obstacle_field-deadbeef",
            "family": "open_obstacle_field",
            "role": "calibration",
            "states": 1,
            "envs": 10,
            "physics_build_wall_seconds": 0.1,
            "physics_simulation_wall_seconds": 0.2,
            "common_prefix_step_wall_seconds": 0.1,
            "branch_step_wall_seconds": 0.1,
            "render_scene_build_wall_seconds": 0.02,
            "native_render_wall_seconds": 0.05,
            "camera_quality_resize_wall_seconds": 0.05,
            "png_encode_write_hash_wall_seconds": 0.02,
            "lockstep_execution_wall_seconds": 0.2,
            "post_lockstep_receipt_wall_seconds": 0.05,
            "scene_pipeline_wall_seconds": 0.4,
            "scene_total_wall_seconds": 0.6,
            "native_render_calls": 13,
            "stored_rgb_frames": 13,
            "depth_rendered": True,
            "depth_persisted": False,
            "visual_mode": "solid_materials_box_physics_preserved",
        }],
        "visual_domain_limitation": "synthetic solid-material smoke is not a visual-domain claim",
        "collection_wall_seconds": 1.0,
        "failure": None,
    }
    if mutate_collection is not None:
        mutate_collection(collection)
    binding = _write_json(root, "physics_result.json", collection)
    return root / binding["path"], binding["file_sha256"], binding["byte_count"]


def test_canonical_producer_smoke_is_accepted_receipt_only(tmp_path: Path) -> None:
    manifest, digest, byte_count = _fixture(tmp_path)
    report = checker.check_manifest(
        manifest, expected_file_sha256=digest, expected_byte_count=byte_count
    )
    assert report["status"] == "PASS"
    assert report["purpose"] == "source_integration_smoke"
    assert report["counts"]["total_branches"] == 10
    assert report["can_freeze_pilot_contract"] is False
    assert report["rgb_bytes_opened"] is False
    assert report["checkpoints_opened"] is False


def test_actual_producer_fixture_is_accepted_without_rgb_leaf_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = producer_test_support.write_synthetic_complete_smoke_tree(tmp_path)
    raw = manifest.read_bytes()
    opened: list[Path] = []
    original = checker._read_regular_file

    def guarded(path: Path, *, expected_bytes: int, name: str) -> bytes:
        opened.append(path)
        assert path.suffix in {".json", ".jsonl"}
        return original(path, expected_bytes=expected_bytes, name=name)

    monkeypatch.setattr(checker, "_read_regular_file", guarded)
    report = checker.check_manifest(
        manifest,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_byte_count=len(raw),
    )
    assert report["status"] == "PASS"
    assert opened
    assert all(path.is_relative_to(manifest.parent) for path in opened)
    assert report["rgb_bytes_opened"] is False


def test_duplicate_candidate_tape_is_rejected(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][1]["executed_block"] = copy.deepcopy(state["branches"][0]["executed_block"])
        state["branches"][1]["executed_block_sha256"] = state["branches"][0]["executed_block_sha256"]

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="duplicate executed command tapes"):
        checker.check_manifest(manifest, expected_file_sha256=digest, expected_byte_count=byte_count)


def test_sentinel_command_drift_is_rejected(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][-1]["executed_block"] = _block(8)
        state["branches"][-1]["executed_block_sha256"] = producer.canonical_block_sha256(_block(8))

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="sentinel command/RGB receipts"):
        checker.check_manifest(manifest, expected_file_sha256=digest, expected_byte_count=byte_count)


def test_sentinel_endpoint_drift_is_rejected(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][-1]["endpoint_state"]["base_pos_world"][0] += 0.01
        state["branches"][-1]["trajectory_policy_step_samples"][-1][
            "base_pos_world"
        ][0] += 0.01

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_trajectory_must_have_exact_policy_sample_count(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][0]["trajectory_policy_step_samples"].pop()

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="exactly 25 trajectory"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_trajectory_must_have_exact_20ms_cadence(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][0]["trajectory_policy_step_samples"][7][
            "timestamp_ns"
        ] += 1

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="ordered 20 ms samples"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_trajectory_schema_and_vector_width_are_exact(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        sample = state["branches"][0]["trajectory_policy_step_samples"][0]
        sample["undeclared"] = 1

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="unexpected=.*undeclared"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_endpoint_must_equal_final_trajectory_sample(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        state["branches"][0]["endpoint_state"]["base_pos_world"][0] = 0.25

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="does not exactly equal"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_fall_label_is_recomputed_from_trajectory(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        branch = state["branches"][0]
        branch["trajectory_policy_step_samples"][-1]["base_pos_world"][2] = 0.1
        branch["endpoint_state"]["base_pos_world"][2] = 0.1

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="physical_fell disagrees"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_tip_label_is_recomputed_from_trajectory(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        branch = state["branches"][0]
        tipped = [2.0**-0.5, 2.0**-0.5, 0.0, 0.0]
        branch["trajectory_policy_step_samples"][-1]["base_quat_wxyz"] = tipped
        branch["endpoint_state"]["base_quat_wxyz"] = tipped

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="physical_tipped disagrees"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_path_length_and_target_progress_are_recomputed(tmp_path: Path) -> None:
    def mutate_path(state: dict[str, Any]) -> None:
        state["branches"][0]["physical_path_length_m"] = 0.01

    manifest, digest, byte_count = _fixture(tmp_path / "path", mutate_state=mutate_path)
    with pytest.raises(checker.PilotReceiptError, match="physical_path_length_m disagrees"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )

    def mutate_progress(state: dict[str, Any]) -> None:
        state["branches"][0]["physical_target_progress_m"] = 0.01

    manifest, digest, byte_count = _fixture(
        tmp_path / "progress", mutate_state=mutate_progress
    )
    with pytest.raises(
        checker.PilotReceiptError, match="physical_target_progress_m disagrees"
    ):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_sentinel_component_set_is_exact(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        del state["sentinel_audit"]["components"]["qpos"]

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="component audit set changed"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_live_render_sentinel_hashes_must_equal_state_audit(tmp_path: Path) -> None:
    def mutate(render: dict[str, Any]) -> None:
        audit = dict(render["render_sentinel_audits"][0])
        audit["candidate_stored_rgb_sha256"] = "d" * 64
        audit["sentinel_stored_rgb_sha256"] = "d" * 64
        render["render_sentinel_audits"][0] = audit

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate)
    with pytest.raises(checker.PilotReceiptError, match="identities or hashes disagree"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_refill_authorization_is_rejected(tmp_path: Path) -> None:
    manifest, digest, byte_count = _fixture(
        tmp_path, mutate_collection=lambda collection: collection.update(allows_refill=True)
    )
    with pytest.raises(checker.PilotReceiptError, match="refill/overwrite"):
        checker.check_manifest(manifest, expected_file_sha256=digest, expected_byte_count=byte_count)


def test_undeclared_collection_field_is_rejected(tmp_path: Path) -> None:
    manifest, digest, byte_count = _fixture(
        tmp_path,
        mutate_collection=lambda collection: collection.update(undeclared=True),
    )
    with pytest.raises(checker.PilotReceiptError, match="unexpected=.*undeclared"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_runtime_version_receipt_rejects_whitespace(tmp_path: Path) -> None:
    def mutate(collection: dict[str, Any]) -> None:
        collection["runtime_versions"]["genesis"] = " 0.3.3"

    manifest, digest, byte_count = _fixture(tmp_path, mutate_collection=mutate)
    with pytest.raises(checker.PilotReceiptError, match="trimmed printable"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_scene_stage_timing_must_be_internally_consistent(tmp_path: Path) -> None:
    def mutate(collection: dict[str, Any]) -> None:
        collection["scene_metrics"][0]["scene_total_wall_seconds"] = 0.1

    manifest, digest, byte_count = _fixture(tmp_path, mutate_collection=mutate)
    with pytest.raises(checker.PilotReceiptError, match="timings are internally inconsistent"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_render_scene_identity_must_equal_state_scene(tmp_path: Path) -> None:
    def mutate(render: dict[str, Any]) -> None:
        render["scene"]["scene_id"] = "open_obstacle_field-other"

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate)
    with pytest.raises(checker.PilotReceiptError, match="scene identities differ"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_low_information_frame_is_retained_and_stratifiable(tmp_path: Path) -> None:
    reasons = [
        "near_forward_geometry",
        "low_rgb_texture",
        "near_wall_depth",
    ]

    def mutate(render: dict[str, Any]) -> None:
        render["frame_receipts"][0]["low_information"] = True
        render["frame_receipts"][0]["low_info_reasons"] = list(reasons)
        quality = render["quality_audits"][0]["quality"]
        quality["raw_assessment_valid"] = False
        quality["observed_reasons"] = list(reasons)
        quality["low_information"] = True
        quality["low_info_reasons"] = list(reasons)

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate)
    report = checker.check_manifest(
        manifest,
        expected_file_sha256=digest,
        expected_byte_count=byte_count,
    )
    assert report["status"] == "PASS"


def test_low_information_tag_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(render: dict[str, Any]) -> None:
        render["frame_receipts"][0]["low_information"] = True
        render["frame_receipts"][0]["low_info_reasons"] = ["near_wall_depth"]

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate)
    with pytest.raises(checker.PilotReceiptError, match="disposition failed"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_hard_quality_failure_cannot_be_relabelled_low_information(
    tmp_path: Path,
) -> None:
    def mutate(render: dict[str, Any]) -> None:
        quality = render["quality_audits"][0]["quality"]
        quality["raw_assessment_valid"] = False
        quality["observed_reasons"] = ["camera_safety_unresolved"]
        quality["hard_failure_reasons"] = ["camera_safety_unresolved"]

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate)
    with pytest.raises(checker.PilotReceiptError, match="disposition failed"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_body_target_is_recomputed_at_nonzero_yaw(tmp_path: Path) -> None:
    scale = 1.0000001
    half = 2.0 ** -0.5
    pose = {
        "position_xyz_m": [0.0, 0.0, 0.4],
        "quaternion_wxyz": [scale * half, 0.0, 0.0, scale * half],
    }

    def mutate(state: dict[str, Any]) -> None:
        state["context"]["prebranch_base_pose_world"] = copy.deepcopy(pose)
        state["context"]["context_base_pose_world_sequence"][-1] = copy.deepcopy(
            pose
        )
        state["context"]["target_relative_body_xy_m"] = [2.0, -1.0]

    def mutate_render(render: dict[str, Any]) -> None:
        render["quality_audits"][2]["replay_pose"][
            "source_base_pose_world"
        ] = copy.deepcopy(pose)

    manifest, digest, byte_count = _fixture(
        tmp_path, mutate_state=mutate, mutate_render=mutate_render
    )
    report = checker.check_manifest(
        manifest,
        expected_file_sha256=digest,
        expected_byte_count=byte_count,
    )
    assert report["status"] == "PASS"


def test_world_target_cannot_be_relabelled_as_body_target(tmp_path: Path) -> None:
    half = 2.0 ** -0.5
    quaternion = [half, 0.0, 0.0, half]

    def mutate(state: dict[str, Any]) -> None:
        state["context"]["prebranch_base_pose_world"]["quaternion_wxyz"] = [
            *quaternion
        ]
        state["context"]["context_base_pose_world_sequence"][-1][
            "quaternion_wxyz"
        ] = [*quaternion]

    def mutate_render(render: dict[str, Any]) -> None:
        render["quality_audits"][2]["replay_pose"]["source_base_pose_world"][
            "quaternion_wxyz"
        ] = [*quaternion]

    manifest, digest, byte_count = _fixture(
        tmp_path, mutate_state=mutate, mutate_render=mutate_render
    )
    with pytest.raises(checker.PilotReceiptError, match="body-frame target disagrees"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=digest,
            expected_byte_count=byte_count,
        )


def test_collapsed_candidate_physical_responses_are_rejected(tmp_path: Path) -> None:
    def mutate(state: dict[str, Any]) -> None:
        reference = state["branches"][0]
        for branch in state["branches"][1:]:
            for field in (
                "trajectory_policy_step_samples",
                "endpoint_state",
                "physical_fell",
                "physical_tipped",
                "physical_path_length_m",
                "physical_target_progress_m",
            ):
                branch[field] = copy.deepcopy(reference[field])

    manifest, digest, byte_count = _fixture(tmp_path, mutate_state=mutate)
    with pytest.raises(checker.PilotReceiptError, match="one physical response"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_collapsed_candidate_rgb_responses_are_rejected(tmp_path: Path) -> None:
    collapsed = "e" * 64

    def mutate_state(state: dict[str, Any]) -> None:
        for branch in state["branches"]:
            branch["frame_receipt"]["file_sha256"] = collapsed

    def mutate_render(render: dict[str, Any]) -> None:
        for frame in render["frame_receipts"][3:]:
            frame["file_sha256"] = collapsed

    manifest, digest, byte_count = _fixture(
        tmp_path, mutate_state=mutate_state, mutate_render=mutate_render
    )
    with pytest.raises(checker.PilotReceiptError, match="one sequential RGB response"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_branch_replay_pose_must_match_physical_endpoint(tmp_path: Path) -> None:
    def mutate_render(render: dict[str, Any]) -> None:
        original = render["quality_audits"][3]["replay_pose"][
            "source_base_pose_world"
        ]["position_xyz_m"]
        render["quality_audits"][3]["replay_pose"]["source_base_pose_world"][
            "position_xyz_m"
        ] = [0.5, *original[1:]]

    manifest, digest, byte_count = _fixture(tmp_path, mutate_render=mutate_render)
    with pytest.raises(checker.PilotReceiptError, match="physical endpoint"):
        checker.check_manifest(
            manifest, expected_file_sha256=digest, expected_byte_count=byte_count
        )


def test_duplicate_top_json_key_is_rejected(tmp_path: Path) -> None:
    manifest, _digest_value, _byte_count = _fixture(tmp_path)
    raw = manifest.read_bytes()
    malformed = raw.replace(b'{"allows_overwrite"', b'{"schema":"duplicate","allows_overwrite"', 1)
    manifest.write_bytes(malformed)
    with pytest.raises(checker.PilotReceiptError, match="duplicate JSON key"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=hashlib.sha256(malformed).hexdigest(),
            expected_byte_count=len(malformed),
        )


def test_sealed_receipt_path_rejected_before_open(tmp_path: Path) -> None:
    def mutate(collection: dict[str, Any]) -> None:
        collection["plan_receipt_binding"] = {
            "path": "sealed/plan.json",
            "file_sha256": "0" * 64,
            "byte_count": 1,
        }

    manifest, digest, byte_count = _fixture(tmp_path, mutate_collection=mutate)
    with pytest.raises(checker.PilotReceiptError, match="forbidden receipt path"):
        checker.check_manifest(manifest, expected_file_sha256=digest, expected_byte_count=byte_count)


def test_role_aware_lane_counts_preserve_frozen_sizing() -> None:
    assert checker._branch_count_for_role("calibration") == 10
    assert checker._branch_count_for_role("train") == 9
    assert checker._branch_count_for_role("eval") == 9
