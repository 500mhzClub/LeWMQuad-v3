from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
    CANONICAL_ACTION_BLOCKS,
    CALIBRATION_RECEIPT_SCHEMA,
    COLLECTION_SCHEMA,
    FAMILIES,
    GROUP_SCHEMA,
    PHYSICAL_RANK_CONTRACT_SCHEMA,
    PILOT_MANIFEST_SCHEMA,
    PLAN_SCHEMA,
    PRIMITIVE_NAMES,
    RGB_MANIFEST_SCHEMA,
    STATE_RECEIPT_SCHEMA,
    CounterfactualPilotContractError,
    PhysicalLabelsV1,
    candidate_model_inputs_v1,
    canonical_json_sha256,
    load_bound_pilot_v1,
    physical_dense_ranks_v1,
)
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    evaluator_verdict_v1,
    fit_action_specific_ridge_readouts_v1,
    fit_ridge_readout_v1,
    masked_token_descriptor_v1,
    paired_scene_cluster_comparison_v1,
    predict_action_specific_scores_v1,
    predict_ridge_readout_v1,
    selection_metrics_v1,
    task_conditioned_feature_v1,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _binding(root: Path, relative: str) -> dict[str, object]:
    raw = (root / relative).read_bytes()
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _rewrite_manifest(
    root: Path, mutate
) -> tuple[int, str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    raw = _json_bytes(manifest)
    manifest_path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _build_pilot(
    tmp_path: Path,
    *,
    overlap_scenes: bool = False,
    duplicate_executed_tape: bool = False,
    purpose: str = "bounded_wm_a_pilot",
) -> tuple[Path, int, str]:
    root = tmp_path / "pilot"
    (root / "receipts").mkdir(parents=True)
    (root / "frames").mkdir()
    attempt_id = "synthetic-pilot-v1"
    requested_blocks = [
        [list(command) for command in block] for block in CANONICAL_ACTION_BLOCKS
    ]
    action_catalog = [
        {"action_id": action, "name": name, "requested_block": requested_blocks[action]}
        for action, name in enumerate(PRIMITIVE_NAMES)
    ]
    inert = lambda path, marker: {
        "path": path,
        "file_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        "byte_count": len(marker),
    }
    source_bindings = [
        {
            "name": name,
            "binding": inert(f"/synthetic/{name}.py", f"source:{name}"),
        }
        for name in ("collector", "contract", "checker")
    ]
    runtime_bindings = {
        name: inert(f"/synthetic/runtime/{name}", f"runtime:{name}")
        for name in (
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
    }
    execution_contract = {
        "backend": "synthetic-test",
        "policy_device": "cpu",
        "seed": 7,
        "fall_z_threshold_m": 0.18,
        "tip_threshold_rad": 1.0,
        "policy_steps_per_command_tick": 5,
        "python_invocation_path": "/synthetic/python",
        "environment": {
            "EGL_DEVICE_ID": "1",
            "GS_BACKEND": "vulkan",
            "MESA_VK_DEVICE_SELECT": "1002:7551!",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        },
        "graphics_preflight": {
            "egl_device_index": 1,
            "eglinfo_expected_exit_code": 2,
            "egl_renderer_name_contains": "AMD Radeon AI PRO R9700",
            "vulkan_device_index": 0,
            "vulkan_vendor_id": "0x1002",
            "vulkan_device_id": "0x7551",
            "vulkan_device_name": "AMD Radeon AI PRO R9700",
        },
    }
    artifacts: list[dict[str, object]] = []
    role_rows: dict[str, list[dict[str, object]]] = {"train": [], "eval": []}
    plan_states: list[dict[str, object]] = []
    state_receipts: list[dict[str, object]] = []

    def add_artifact(identity: str) -> tuple[str, dict[str, object]]:
        artifact_id = f"artifact:{identity}"
        relative = f"frames/{identity.replace(':', '_')}.png"
        raw = ("synthetic:" + identity).encode()
        (root / relative).write_bytes(raw)
        receipt = {
            "artifact_id": artifact_id,
            "frame_identity": identity,
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "width": 224,
            "height": 224,
            "mode": "RGB",
            "format": "PNG",
            "camera_valid": True,
        }
        artifacts.append(dict(receipt))
        return artifact_id, receipt

    group_index = 0
    for role in ("train", "eval"):
        for family_index, family in enumerate(FAMILIES):
            scene_prefix = "train" if role == "train" or overlap_scenes else "eval"
            scene_id = f"{scene_prefix}_scene_{family_index}"
            state_id = f"{role}:{family_index}:0"
            context_ids = []
            frame_identities = []
            for endpoint in range(3):
                identity = f"{state_id}:context:{endpoint}"
                artifact_id, _ = add_artifact(identity)
                context_ids.append(artifact_id)
                frame_identities.append(identity)
            history_action_ids = [0, 1]
            history_blocks = [requested_blocks[action] for action in history_action_ids]
            state_sha = hashlib.sha256(state_id.encode()).hexdigest()
            lane_start = group_index * 9
            # Deliberately nonzero yaw and slightly non-unit float32 values.
            quaternion = [0.7071067690849304, 0.0, 0.0, 0.7071067690849304]
            target_world = [2.0, 3.0]
            prebranch_position = [1.0, 1.0, 0.30]
            target_body = [2.0, -1.0]
            components = {
                name: {
                    "exact_equal": True,
                    "max_abs_difference": 0.0,
                    "shape_per_lane": [1],
                }
                for name in (
                    "qpos",
                    "dofs_velocity",
                    "base_pos_world",
                    "base_quat_wxyz",
                    "base_lin_vel_world",
                    "base_ang_vel_world",
                    "leg_joint_pos",
                    "leg_joint_vel",
                    "runner_last_executed",
                    "policy_last_actions",
                )
            }
            branches = []
            for action, name in enumerate(PRIMITIVE_NAMES):
                identity = f"{state_id}:candidate:{action}"
                _, frame_receipt = add_artifact(identity)
                executed = requested_blocks[0] if (
                    duplicate_executed_tape and role == "train" and family_index == 0 and action == 1
                ) else requested_blocks[action]
                branches.append({
                    "lane_index": lane_start + action,
                    "lane_offset": action,
                    "kind": "candidate",
                    "action_id": action,
                    "action_name": name,
                    "requested_block": requested_blocks[action],
                    "executed_block": executed,
                    "executed_block_sha256": canonical_json_sha256(executed),
                    "clipped": False,
                    "trajectory_policy_step_samples": [{"step": 0}],
                    "endpoint_state": {"x": float(action)},
                    "physical_fell": False,
                    "physical_tipped": False,
                    "physical_path_length_m": 1.0 + action * 0.01,
                    "physical_target_progress_m": action * 0.1,
                    "render_frame_identity": identity,
                    "frame_receipt": frame_receipt,
                    "declared_oracle_dense_rank": 8 - action,
                })
            role_rows[role].append({
                "schema": GROUP_SCHEMA,
                "role": role,
                "state_id": state_id,
                "family": family,
                "scene_id": scene_id,
                "group_index": group_index,
                "state_index_in_scene": 0,
                "task": {
                    "target_present": True,
                    "relative_target_xy_body_m": target_body,
                },
                "context": {
                    "rgb_artifact_ids": context_ids,
                    "frame_identities": frame_identities,
                    "history_action_ids": history_action_ids,
                    "history_executed_blocks": history_blocks,
                    "executed_block_sha256s": [
                        canonical_json_sha256(block) for block in history_blocks
                    ],
                    "endpoint_command_ticks": [0, 5, 10],
                    "prebranch_state_sha256": state_sha,
                },
                "synchronization_audit": {
                    "state_id": state_id,
                    "group_index": group_index,
                    "lane_start": lane_start,
                    "lane_count": 9,
                    "exact_equality_required": True,
                    "passed": True,
                    "prebranch_state_sha256": state_sha,
                    "lane_state_sha256s": [state_sha] * 9,
                    "components": components,
                },
                "branches": branches,
            })
            plan_state = {
                "state_id": state_id,
                "role": role,
                "family": family,
                "scene_id": scene_id,
                "scene_manifest_binding": inert(
                    f"/synthetic/scenes/{scene_id}/manifest.json",
                    f"scene-manifest:{scene_id}",
                ),
                "scene_genesis_binding": inert(
                    f"/synthetic/scenes/{scene_id}/genesis_scene.json",
                    f"scene-genesis:{scene_id}",
                ),
                "scene_generation": None,
                "group_index": group_index,
                "state_index_in_scene": 0,
                "history_action_ids": history_action_ids,
                "candidate_action_ids": list(range(9)),
                "sentinel_duplicate_action_id": None,
                "target_xy_m": target_world,
            }
            plan_states.append(plan_state)
            receipt_context = {
                **role_rows[role][-1]["context"],
                "prebranch_base_pose_world": {
                    "position_xyz_m": prebranch_position,
                    "quaternion_wxyz": quaternion,
                },
                "context_base_pose_world_sequence": [
                    {
                        "position_xyz_m": prebranch_position,
                        "quaternion_wxyz": quaternion,
                    }
                    for _ in range(3)
                ],
                "target_relative_body_xy_m": target_body,
            }
            state_receipts.append({
                "schema": STATE_RECEIPT_SCHEMA,
                "attempt_id": attempt_id,
                "status": "PHYSICS_COMPLETE",
                "physics_validated": False,
                "citable_as_scientific_evidence": False,
                "authorizes_retry_or_resume": False,
                "state": {
                    "state_id": state_id,
                    "role": role,
                    "family": family,
                    "scene_id": scene_id,
                    "group_index": group_index,
                    "state_index_in_scene": 0,
                    "lane_start": lane_start,
                    "lane_count": 9,
                    "scene_manifest_binding": plan_state["scene_manifest_binding"],
                    "scene_genesis_binding": plan_state["scene_genesis_binding"],
                    "target_xy_m": target_world,
                },
                "context": receipt_context,
                "synchronization_audit": role_rows[role][-1]["synchronization_audit"],
                "branches": [
                    {key: value for key, value in branch.items() if key != "declared_oracle_dense_rank"}
                    | {"duplicates_candidate_action_id": None}
                    for branch in branches
                ],
                "sentinel_audit": None,
                "render_sentinel_audit": None,
                "render_receipt_binding": inert(
                    f"receipts/render_{group_index}.json",
                    f"render:{group_index}",
                ),
            })
            group_index += 1

    expected_counts = {
        "scenes": len(plan_states),
        "states": len(plan_states),
        "roles": {"eval": len(FAMILIES), "train": len(FAMILIES)},
        "actions": 9,
        "candidate_branches": len(plan_states) * 9,
        "sentinel_branches": 0,
        "total_branches": len(plan_states) * 9,
        "context_frames": len(plan_states) * 3,
        "target_frames": len(plan_states) * 9,
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "attempt_id": attempt_id,
        "purpose": "bounded_wm_a_pilot",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": "parallel_lockstep_envs_no_restore",
        "states_per_scene": 1,
        "history_blocks": 2,
        "output_root": str(root.resolve()),
        "runtime_bindings": runtime_bindings,
        "execution_contract": execution_contract,
        "render_contract": {
            "native_resolution": [640, 480],
            "stored_resolution": [224, 224],
            "rgb_format": "png",
            "depth_validation": "transient_not_persisted",
            "replay_env_mode": "single_non_batched_sequential",
            "replay_pose_source": "captured_physical_base_pose",
            "physical_scene_rendering": False,
        },
        "action_catalog": action_catalog,
        "states": plan_states,
        "expected_counts": expected_counts,
    }
    plan_raw = _json_bytes(plan)
    (root / "receipts/plan.json").write_bytes(plan_raw)
    external_plan_path = tmp_path / "authority" / "external_authorized_plan.json"
    external_plan_path.parent.mkdir(parents=True, exist_ok=True)
    external_plan_path.write_bytes(plan_raw)
    external_plan_binding = {
        "path": str(external_plan_path.resolve()),
        "file_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "byte_count": len(plan_raw),
    }
    state_receipt_bindings = []
    render_receipt_bindings = []
    for index, receipt in enumerate(state_receipts):
        relative = f"receipts/state_{index}.json"
        (root / relative).write_bytes(_json_bytes(receipt))
        state_receipt_bindings.append(_binding(root, relative))
        render_receipt_bindings.append(receipt["render_receipt_binding"])
    collection = {
        "schema": COLLECTION_SCHEMA,
        "attempt_id": attempt_id,
        "purpose": "bounded_wm_a_pilot",
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": "parallel_lockstep_envs_no_restore",
        "plan_binding": external_plan_binding,
        "plan_receipt_binding": _binding(root, "receipts/plan.json"),
        "authority_binding": inert("receipts/authority.json", "authority"),
        "review_binding": inert("receipts/review.json", "review"),
        "reservation_binding": inert("receipts/reservation.json", "reservation"),
        "execution_contract": execution_contract,
        "runtime_versions": {
            "python": "synthetic",
            "genesis": "synthetic",
            "torch": "synthetic",
            "numpy": "synthetic",
            "pillow": "synthetic",
        },
        "runtime_bindings": runtime_bindings,
        "source_bindings": source_bindings,
        "caps": {"synthetic": True},
        "expected_counts": expected_counts,
        "observed_counts": expected_counts,
        "scene_materialization": {"synthetic": True},
        "state_receipt_bindings": state_receipt_bindings,
        "render_receipt_bindings": render_receipt_bindings,
        "scene_metrics": [],
        "visual_domain_limitation": "synthetic test fixture only",
        "collection_wall_seconds": 0.0,
        "failure": None,
    }
    (root / "receipts/collection.json").write_bytes(_json_bytes(collection))

    calibration_contract = {
        "schema": PHYSICAL_RANK_CONTRACT_SCHEMA,
        "excluded_scene_ids": ["calibration_scene"],
        "progress_tolerance_m": 1e-6,
        "path_length_tolerance_m": 1e-6,
        "quantization_rule": "sign(x)*floor(abs(x)/t+0.5)",
        "lexicographic_key": [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ],
        "proxy_fields_excluded": True,
        "tolerance_derivation": {
            "schema": "lewm_go2_world_model_counterfactual_tolerance_derivation_v1",
            "method": "fixed_numerical_floor_after_exact_deterministic_repeat_gate",
            "minimum_numerical_resolution_m": 1e-6,
            "repeat_controls": 16,
            "repeated_action_ids": [index % 9 for index in range(16)],
            "all_requested_primitives_covered": True,
            "deterministic_repeat_gate_passed": True,
            "empirical_noise_scale_estimated": False,
        },
    }
    calibration_sources = [
        {
            "name": name,
            "binding": inert(f"/synthetic/{name}.py", f"calibration-source:{name}"),
        }
        for name in ("checker", "calibration_analyzer", "pilot_joiner")
    ]
    calibration_receipt = {
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "status": "SYNTHETIC_TEST_ONLY",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "calibration_id": "synthetic-calibration",
        "role": "calibration",
        "train_eval_scenes_accessed": False,
        "decision": "FREEZE_PILOT_CONTRACT",
        "calibration_collection_receipt": inert(
            "/synthetic/calibration/physics_result.json", "calibration"
        ),
        "calibration_contract": calibration_contract,
        "repeatability_analysis": {
            "repeat_controls": 16,
            "repeated_action_ids": [index % 9 for index in range(16)],
            "all_requested_primitives_covered": True,
            "interpretation": (
                "deterministic_replay_gate_not_empirical_noise_estimate"
            ),
            "empirical_noise_scale_estimated": False,
            "progress_max_abs_delta_m": 0.0,
            "path_length_max_abs_delta_m": 0.0,
            "endpoint_position_max_abs_delta_m": 0.0,
            "endpoint_quaternion_max_abs_delta": 0.0,
            "executed_command_tapes_exact": True,
            "physical_trajectories_exact": True,
            "stored_rgb_exact": True,
        },
        "physics_validation": {
            "receipt_checker_passed": True,
            "common_prefix_exact": True,
            "nine_unique_executed_tapes_per_state": True,
            "minimum_physical_rank_classes_per_state": 9,
            "maximum_physical_rank_classes_per_state": 9,
            "clipped_candidate_branches": 0,
            "physics_validated_for_branch_outcomes": True,
        },
        "visual_validation": {
            "camera_quality_receipts_passed": True,
            "endpoint_pose_replay_bound": True,
            "visual_domain_fidelity_claimed": False,
            "eligible_for_physical_branch_evaluation": True,
            "eligible_for_visual_domain_parity_claim": False,
        },
        "resource_measurements": {
            "schema": (
                "lewm_go2_world_model_counterfactual_calibration_"
                "resource_measurements_v1"
            ),
            "stored_rgb_png": {
                "context_frames": 48,
                "context_bytes": 4_800,
                "target_frames": 160,
                "target_bytes": 32_000,
                "total_frames": 208,
                "total_bytes": 36_800,
                "raw_uncompressed_rgb_ceiling_bytes": 208 * 224 * 224 * 3,
            },
            "stage_wall_seconds": {
                "collection_external_wall_seconds": 8.0,
                "physics_scene_build_wall_seconds": 0.8,
                "render_scene_build_wall_seconds": 0.8,
                "common_prefix_step_wall_seconds": 1.6,
                "branch_step_wall_seconds": 0.8,
                "native_render_wall_seconds": 1.6,
                "camera_quality_resize_wall_seconds": 0.8,
                "png_encode_write_hash_wall_seconds": 0.8,
                "post_lockstep_receipt_wall_seconds": 1.6,
                "summed_scene_total_wall_seconds": 8.0,
            },
            "outcome_counts": {
                "complete_all_nine_action_groups": 16,
                "executed_tape_distinct_groups": 16,
                "prebranch_exact_groups": 16,
                "clipped_candidate_branches": 0,
                "fallen_candidate_branches": 0,
                "tipped_candidate_branches": 0,
                "camera_invalid_frames": 0,
                "incomplete_states": 0,
            },
            "gpu_peak_memory_measurement_scope": (
                "external_terminal_required_not_observed_by_analyzer"
            ),
        },
        "analyzer_binding": calibration_sources[1]["binding"],
        "checker_binding": calibration_sources[0]["binding"],
        "source_bindings": calibration_sources,
    }
    (root / "receipts/calibration.json").write_bytes(
        _json_bytes(calibration_receipt)
    )

    (root / "rgb_manifest.json").write_bytes(_json_bytes({
        "schema": RGB_MANIFEST_SCHEMA,
        "artifacts": artifacts,
    }))
    role_contracts = {}
    for role in ("train", "eval"):
        relative = f"{role}.jsonl"
        body = b"".join(_json_bytes(row) + b"\n" for row in role_rows[role])
        (root / relative).write_bytes(body)
        role_contracts[role] = {
            "index": _binding(root, relative),
            "group_count": len(role_rows[role]),
            "branch_count": len(role_rows[role]) * 9,
            "scene_ids": sorted({str(row["scene_id"]) for row in role_rows[role]}),
        }
    manifest = {
        "schema": PILOT_MANIFEST_SCHEMA,
        "attempt_id": attempt_id,
        "purpose": purpose,
        "status": "COMPLETE",
        "physics_validated": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "evidence_scope": "physics_executed",
        "receipt_root": str(root.resolve()),
        "output_root": str(root.resolve()),
        "action_catalog": action_catalog,
        "action_contract": {
            "primitive_names": list(PRIMITIVE_NAMES),
            "command_ticks_per_block": 5,
            "executed_tape_shape": [5, 3],
            "candidate_model_input": "requested_action_id",
            "future_executed_tape_usage": "target_and_audit_only",
        },
        "calibration_contract": calibration_contract,
        "calibration_receipt": _binding(root, "receipts/calibration.json"),
        "roles": role_contracts,
        "rgb_artifact_manifest": _binding(root, "rgb_manifest.json"),
        "source_bindings": source_bindings,
        "collection_receipt": _binding(root, "receipts/collection.json"),
    }
    raw = _json_bytes(manifest)
    (root / "manifest.json").write_bytes(raw)
    return root, len(raw), hashlib.sha256(raw).hexdigest()


def test_strict_bound_pilot_loads_without_rgb_leaf_reads(tmp_path: Path) -> None:
    root, byte_count, sha256 = _build_pilot(tmp_path)
    bundle = load_bound_pilot_v1(
        root,
        expected_manifest_byte_count=byte_count,
        expected_manifest_sha256=sha256,
        allowed_parent=tmp_path,
        synthetic_test_mode=True,
    )
    assert len(bundle.groups_by_role["train"]) == len(FAMILIES)
    assert len(bundle.groups_by_role["eval"]) == len(FAMILIES)
    assert all(len(group.branches) == 9 for group in bundle.groups_by_role["train"])
    assert bundle.groups_by_role["train"][1].group_index == 1
    assert bundle.groups_by_role["train"][1].state_index_in_scene == 0
    assert bundle.groups_by_role["eval"][0].group_index == len(FAMILIES)
    assert bundle.calibration_receipt["document"]["status"] == "SYNTHETIC_TEST_ONLY"
    assert bundle.access_audit["rgb_leaf_open_count"] == 0


def test_candidate_model_inputs_exclude_future_executed_tapes(tmp_path: Path) -> None:
    root, byte_count, sha256 = _build_pilot(tmp_path)
    bundle = load_bound_pilot_v1(
        root,
        expected_manifest_byte_count=byte_count,
        expected_manifest_sha256=sha256,
        allowed_parent=tmp_path,
        synthetic_test_mode=True,
    )
    group = bundle.groups_by_role["train"][0]
    inputs = candidate_model_inputs_v1(group)
    assert [item.requested_action_id for item in inputs] == list(range(9))
    assert inputs[0].requested_block == group.branches[0].requested_block
    assert not hasattr(inputs[0], "executed_command_tape")


def test_allowed_parent_does_not_enable_synthetic_provenance(tmp_path: Path) -> None:
    root, byte_count, sha256 = _build_pilot(tmp_path)
    with pytest.raises(CounterfactualPilotContractError):
        load_bound_pilot_v1(
            root,
            expected_manifest_byte_count=byte_count,
            expected_manifest_sha256=sha256,
            allowed_parent=tmp_path,
        )


def test_canonical_action_values_and_collection_source_join_fail_closed(
    tmp_path: Path,
) -> None:
    action_root, _, _ = _build_pilot(tmp_path / "action")
    action_bytes, action_sha = _rewrite_manifest(
        action_root,
        lambda manifest: manifest["action_catalog"][0]["requested_block"][0].__setitem__(
            0, 0.21
        ),
    )
    with pytest.raises(CounterfactualPilotContractError, match="canonical registry"):
        load_bound_pilot_v1(
            action_root,
            expected_manifest_byte_count=action_bytes,
            expected_manifest_sha256=action_sha,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )

    source_root, _, _ = _build_pilot(tmp_path / "source")
    source_bytes, source_sha = _rewrite_manifest(
        source_root,
        lambda manifest: manifest["source_bindings"][0]["binding"].__setitem__(
            "file_sha256", "f" * 64
        ),
    )
    with pytest.raises(CounterfactualPilotContractError, match="source bindings changed"):
        load_bound_pilot_v1(
            source_root,
            expected_manifest_byte_count=source_bytes,
            expected_manifest_sha256=source_sha,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )


def test_joined_task_requires_verified_nonzero_yaw_body_frame(tmp_path: Path) -> None:
    root, _, _ = _build_pilot(tmp_path)
    train_path = root / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert rows[0]["task"]["relative_target_xy_body_m"] == [2.0, -1.0]
    rows[0]["task"]["relative_target_xy_body_m"] = [1.0, 2.0]
    train_path.write_bytes(b"".join(_json_bytes(row) + b"\n" for row in rows))

    def rebind(manifest: dict[str, object]) -> None:
        manifest["roles"]["train"]["index"] = _binding(root, "train.jsonl")

    byte_count, sha256 = _rewrite_manifest(root, rebind)
    with pytest.raises(CounterfactualPilotContractError, match="body-frame target"):
        load_bound_pilot_v1(
            root,
            expected_manifest_byte_count=byte_count,
            expected_manifest_sha256=sha256,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )


def test_manifest_binding_and_purpose_fail_closed(tmp_path: Path) -> None:
    root, byte_count, sha256 = _build_pilot(tmp_path)
    with pytest.raises(CounterfactualPilotContractError):
        load_bound_pilot_v1(
            root,
            expected_manifest_byte_count=byte_count,
            expected_manifest_sha256="f" * 64,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )
    smoke_root, smoke_bytes, smoke_sha = _build_pilot(
        tmp_path / "smoke", purpose="source_integration_smoke"
    )
    with pytest.raises(CounterfactualPilotContractError):
        load_bound_pilot_v1(
            smoke_root,
            expected_manifest_byte_count=smoke_bytes,
            expected_manifest_sha256=smoke_sha,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )


def test_scene_overlap_and_duplicate_tapes_fail_closed(tmp_path: Path) -> None:
    overlap_root, overlap_bytes, overlap_sha = _build_pilot(
        tmp_path / "overlap", overlap_scenes=True
    )
    with pytest.raises(
        CounterfactualPilotContractError, match="identity repeats|disjoint"
    ):
        load_bound_pilot_v1(
            overlap_root,
            expected_manifest_byte_count=overlap_bytes,
            expected_manifest_sha256=overlap_sha,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )
    duplicate_root, duplicate_bytes, duplicate_sha = _build_pilot(
        tmp_path / "duplicate", duplicate_executed_tape=True
    )
    with pytest.raises(CounterfactualPilotContractError, match="duplicate executed"):
        load_bound_pilot_v1(
            duplicate_root,
            expected_manifest_byte_count=duplicate_bytes,
            expected_manifest_sha256=duplicate_sha,
            allowed_parent=tmp_path,
            synthetic_test_mode=True,
        )


def test_physical_rank_is_lexicographic_and_proxy_blind() -> None:
    labels = [
        PhysicalLabelsV1(False, False, 1.0, 2.0, -10.0, 0.0),
        PhysicalLabelsV1(False, False, 1.0, 2.0, 10.0, 1.0),
        PhysicalLabelsV1(False, True, 100.0, 0.0, None, None),
        PhysicalLabelsV1(True, False, 100.0, 0.0, None, None),
    ]
    ranks = physical_dense_ranks_v1(
        labels,
        {"progress_tolerance_m": 0.1, "path_length_tolerance_m": 0.1},
    )
    assert ranks[0] == ranks[1]
    assert ranks[0] < ranks[2] < ranks[3]


def test_ridge_is_deterministic_and_uses_only_fit_rows() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    targets = np.asarray([0.0, 1.0, 2.0])
    first = fit_ridge_readout_v1(features, targets, ridge_lambda=0.1)
    second = fit_ridge_readout_v1(features.copy(), targets.copy(), ridge_lambda=0.1)
    assert first.identity_sha256 == second.identity_sha256
    eval_a = predict_ridge_readout_v1(first, [[3.0, 0.0]])
    eval_b = predict_ridge_readout_v1(first, [[300.0, 0.0]])
    assert eval_a.shape == eval_b.shape == (1,)
    assert first.identity_sha256 == second.identity_sha256


def test_ridge_uses_dual_solver_when_features_exceed_rows() -> None:
    features = np.arange(30, dtype=np.float64).reshape(3, 10)
    readout = fit_ridge_readout_v1(features, [0.0, 1.0, 2.0], ridge_lambda=0.1)
    assert readout.solver == "dual"
    assert np.isfinite(predict_ridge_readout_v1(readout, features)).all()


def test_action_specific_heads_learn_target_dependent_preferences() -> None:
    target_x = (-2.0, -1.0, 1.0, 2.0)
    shared_features = [
        task_conditioned_feature_v1(
            None, relative_target_xy_body_m=(coordinate, 0.0)
        )
        for coordinate in target_x
    ]
    features_by_action = [shared_features for _ in range(9)]
    targets_by_action = [
        list(target_x),
        [-coordinate for coordinate in target_x],
        *([[5.0] * len(target_x)] * 7),
    ]
    readouts = fit_action_specific_ridge_readouts_v1(
        features_by_action, targets_by_action, ridge_lambda=1e-6
    )

    def selected(target: float) -> int:
        feature = task_conditioned_feature_v1(
            None, relative_target_xy_body_m=(target, 0.0)
        )
        return int(np.argmin(predict_action_specific_scores_v1(
            readouts, [feature] * 9
        )))

    assert selected(-1.0) == 0
    assert selected(1.0) == 1


def test_four_mask_descriptor_contract() -> None:
    tokens = np.arange(4 * 3 * 2, dtype=np.float64).reshape(4, 3, 2)
    descriptor = masked_token_descriptor_v1(tokens)
    assert descriptor.shape == (16,)
    with pytest.raises(ValueError, match="shape"):
        masked_token_descriptor_v1(tokens[:3])


def test_selection_metrics_and_scene_cluster_pairing(tmp_path: Path) -> None:
    root, byte_count, sha256 = _build_pilot(tmp_path)
    bundle = load_bound_pilot_v1(
        root,
        expected_manifest_byte_count=byte_count,
        expected_manifest_sha256=sha256,
        allowed_parent=tmp_path,
        synthetic_test_mode=True,
    )
    groups = bundle.groups_by_role["eval"]
    oracle_scores = {group.state_id: list(reversed(range(9))) for group in groups}
    weak_scores = {group.state_id: list(range(9)) for group in groups}
    oracle = selection_metrics_v1(groups, oracle_scores)
    weak = selection_metrics_v1(groups, weak_scores)
    json.dumps(oracle, allow_nan=False)
    assert oracle["summary"]["normalized_rank_regret"] == 0.0
    comparison = paired_scene_cluster_comparison_v1(
        oracle["group_results"], weak["group_results"], resamples=200, seed=7
    )
    assert comparison["upper_95"] < 0.0
    assert comparison == paired_scene_cluster_comparison_v1(
        oracle["group_results"], weak["group_results"], resamples=200, seed=7
    )


def test_atomic_json_cleanup_and_path_config_serialization(tmp_path: Path) -> None:
    from scripts.dev_probe_counterfactual_action_fidelity import write_json_atomic
    from scripts.dev_probe_counterfactual_overfit_capacity import json_safe_config

    output = tmp_path / "bad.json"
    with pytest.raises(TypeError):
        write_json_atomic(output, {"not_json": Path("relative")})
    assert not output.exists()
    assert not output.with_suffix(".json.tmp").exists()
    safe = json_safe_config({"root": tmp_path, "nested": (Path("a"), 3)})
    assert safe == {"root": str(tmp_path), "nested": ["a", 3]}
    json.dumps(safe, allow_nan=False)


def test_verdict_requires_sensitivity_and_every_control() -> None:
    names = {
        "ceiling_vs_current",
        "forecast_vs_current",
        "forecast_vs_task_action",
        "forecast_vs_hold_blind",
        "forecast_vs_shuffled",
        "forecast_vs_random",
    }
    passing = {name: {"upper_95": -0.01} for name in names}
    assert evaluator_verdict_v1(passing).endswith("DEVELOPMENT_ONLY")
    insensitive = {**passing, "ceiling_vs_current": {"upper_95": 0.0}}
    assert evaluator_verdict_v1(insensitive) == "EVALUATOR_SENSITIVITY_NOT_ESTABLISHED"
    weak = {**passing, "forecast_vs_random": {"upper_95": 0.01}}
    assert evaluator_verdict_v1(weak) == "SCENE_DISJOINT_TASK_UTILITY_NOT_ESTABLISHED"
