"""Source-only and CPU-synthetic proof for Full Training V4."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v4_policy as policy
from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import OUTPUT_SHAPE
from lewm.models import shared_observable_camera_ray_jepa_v5 as shared_v5
from lewm.models import (
    shared_observable_camera_ray_jepa_v5_full_training_v4_loss as loss_adapter,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH
SOURCES = {
    "policy": ROOT / policy.POLICY_RELATIVE_PATH,
    "loss_adapter": ROOT / policy.LOSS_ADAPTER_RELATIVE_PATH,
    "preflight": ROOT / policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH,
    "preflight_verifier": ROOT / policy.PREFLIGHT_VERIFIER_RELATIVE_PATH,
    "executor": ROOT / policy.EXACT_EXECUTOR_RELATIVE_PATH,
    "trainer": ROOT / policy.EXACT_TRAINER_RELATIVE_PATH,
    "verifier": ROOT / policy.EXACT_VERIFIER_RELATIVE_PATH,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(name: str) -> str:
    return SOURCES[name].read_text(encoding="ascii")


def _load(relative: str) -> dict[str, Any]:
    return policy.parse_canonical_json(
        (ROOT / relative).read_bytes(),
        name=relative,
    )


def _calibration(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    )[None].expand(batch, -1, -1).clone()
    ground = torch.full((batch,), -0.35)
    return origin, basis, ground


def _small_model() -> shared_v5.SharedObservableCameraRayJepaV5:
    config = shared_v5.SharedObservableCameraRayJepaV5Config(
        schema=shared_v5.SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=0,
        action_dim=3,
        bev_dim=8,
        bev_size=(4, 4),
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        source_shape=(128, 128),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
        v4_pixel_ray_chunk_size=32,
        observable_camera_ray_v4_weight=1.0,
    )
    return shared_v5.SharedObservableCameraRayJepaV5(config)


def _pair(
    model: shared_v5.SharedObservableCameraRayJepaV5,
    *,
    batch: int,
) -> shared_v5.SharedTrainingPairV5:
    current = torch.randn(batch, 3, shared_v5.IMAGE_SIZE, shared_v5.IMAGE_SIZE)
    next_image = torch.randn_like(current)
    origin, basis, ground = _calibration(batch)
    action = torch.zeros(batch, model.action_dim)
    action[:, 0] = 1.0
    action[1::2, 0] = 0.0
    action[1::2, 1] = 1.0
    wrong_action = torch.roll(action, shifts=1, dims=1)
    realized = torch.zeros(batch, 3)
    realized[:, 0] = 0.05
    commanded = torch.zeros(batch, 3)
    commanded[:, 0] = 0.10
    wrong_delta = torch.zeros(batch, 3)
    wrong_delta[:, 1] = 0.20
    return model.forward_training_pair(
        current,
        next_image,
        action,
        realized,
        commanded_delta_pose_current=commanded,
        current_camera_origin_body_m=origin,
        current_camera_basis_body_fru=basis,
        current_ground_plane_z_body_m=ground,
        next_camera_origin_body_m=origin,
        next_camera_basis_body_fru=basis,
        next_ground_plane_z_body_m=ground,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_delta,
        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
    )


def _supervision(
    frame: shared_v5.SharedOnlineFrameV5,
) -> shared_v5.ObservableCameraRayV4FrameSupervisionV5:
    hazard = frame.evidence.pixel_first_hit_hazard_logits
    pixel_shape = (hazard.shape[0], hazard.shape[2], hazard.shape[3])
    hit = torch.zeros(pixel_shape, dtype=torch.bool)
    hit[:, 0, 0] = True
    hit[::2, -1, -1] = True
    distance = torch.zeros(pixel_shape, dtype=hazard.dtype)
    distance[hit] = shared_v5.DEPTH_NEAR_EDGE_M + 1.25 * shared_v5.DEPTH_BIN_SIZE_M
    in_frustum = frame.evidence.ground_query_in_frustum.detach().clone()
    parity = torch.arange(in_frustum.numel()).reshape(in_frustum.shape) % 2 == 0
    clear = in_frustum & parity
    labels = torch.zeros((hazard.shape[0], *OUTPUT_SHAPE), dtype=torch.long)
    labels[:, 12:40, 16:48] = 1
    labels[:, 28:32, 30:34] = 2
    return shared_v5.ObservableCameraRayV4FrameSupervisionV5(
        pixel_hit_mask=hit,
        pixel_first_hit_distance_m=distance,
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
        target_raster_labels=labels,
    )


def _group_balanced(rows: Iterable[tuple[str, float]]) -> float:
    groups: dict[str, list[float]] = {}
    for name, value in rows:
        groups.setdefault(name, []).append(value)
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def _synthetic_raw_manifest() -> dict[str, Any]:
    samples = [
        {
            "dataset_role": role,
            "family": family,
            "endpoint_identity_sha256": hashlib.sha256(f"{role}:{family}".encode()).hexdigest(),
            "selection_sha256": hashlib.sha256(f"select:{role}:{family}".encode()).hexdigest(),
        }
        for role in policy.DEVELOPMENT_ROLES
        for family in policy.FAMILIES
    ]
    shards = []
    global_scene_index = 0
    for role in policy.DEVELOPMENT_ROLES:
        scene_count = policy.ROLE_COUNTS[role]["scenes"]
        unique_count = policy.ROLE_COUNTS[role]["unique_endpoints"]
        per_scene, extra = divmod(unique_count, scene_count)
        for index in range(scene_count):
            family = policy.FAMILIES[index % len(policy.FAMILIES)]
            scene = f"scene_{global_scene_index:03d}_{role}"
            global_scene_index += 1
            shards.append(
                {
                    "path": (
                        "shards/"
                        + hashlib.sha256(scene.encode()).hexdigest()[:16]
                        + "/shard.json"
                    ),
                    "dataset_role": role,
                    "family": family,
                    "scene_id": scene,
                    "endpoint_count": per_scene + (index < extra),
                    "content_sha256": hashlib.sha256(scene.encode()).hexdigest(),
                }
            )
    value = {
        field: None
        for field in policy.RAW_MANIFEST_FIELDS
    }
    value.update(
        {
            "schema": policy.RAW_SUPERVISION_MANIFEST_SCHEMA,
            "status": "complete_pending_independent_audit",
            "evidence_schema": "lewm_go2_observable_camera_ray_evidence_v4",
            "raster_schema": "lewm_go2_observable_camera_ray_raster_v4",
            "roles": list(policy.DEVELOPMENT_ROLES),
            "pair_counts": {
                role: policy.ROLE_COUNTS[role]["pairs"]
                for role in policy.DEVELOPMENT_ROLES
            },
            "endpoint_instance_count": 10344,
            "unique_endpoint_counts": {
                role: policy.ROLE_COUNTS[role]["unique_endpoints"]
                for role in policy.DEVELOPMENT_ROLES
            },
            "scene_shard_count": 88,
            "ordered_pair_sha256": policy.RAW_ORDERED_PAIR_SHA256,
            "ordered_endpoint_sha256": policy.RAW_ORDERED_ENDPOINT_SHA256,
            "pair_index": {
                "path": "pairs.jsonl",
                "row_count": 5172,
                "file_sha256": "1" * 64,
            },
            "endpoint_index": {
                "path": "endpoints.jsonl",
                "row_count": 9460,
                "file_sha256": "2" * 64,
            },
            "array_layout": list(policy.RAW_ARRAY_LAYOUT),
            "shards": shards,
            "files": [],
            "input_provenance": {},
            "access_ledger": {},
            "independent_audit_precommit": {
                "scheme": "minimum_sha256_role_nul_family_nul_endpoint_identity_v1",
                "one_endpoint_per_observed_role_family": True,
                "expected_exact_record_count": 24,
                "records": samples,
                "records_sha256": policy.canonical_json_sha256(samples),
            },
            "parallel_contract": {
                "worker_start_method": "spawn",
                "maximum_workers": 6,
                "native_threads_per_worker": 1,
                "gpu_visible_to_workers": False,
                "merge_order": "role_then_scene_then_endpoint_identity",
                "worker_count_does_not_change_artifact_bytes": True,
            },
            "publication": {
                "staging": "private_sibling_directory_mode_0700",
                "commit": "single_renameat2_RENAME_NOREPLACE",
                "manifest_self_inventory": "canonical_content_sha256",
                "file_inventory": "every_regular_file_except_manifest_self",
            },
            "licenses": {
                field: False
                for field in (
                    "independent_audit_passed",
                    "dataset_use_authorized",
                    "rgb_decode_authorized",
                    "training_authorized",
                    "selection_authorized",
                    "calibration_authorized",
                    "g2_authorized",
                    "heldout_authorized",
                    "runtime_authorized",
                    "hardware_authorized",
                    "production_authorized",
                    "promotion_authorized",
                )
            },
            "content_sha256": policy.RAW_V9_MANIFEST_CONTENT_SHA256,
        }
    )
    return value


def _synthetic_raw_report() -> dict[str, Any]:
    samples = []
    for role in policy.DEVELOPMENT_ROLES:
        for family in policy.FAMILIES:
            arrays = [
                hashlib.sha256(f"{role}:{family}:{index}".encode()).hexdigest()
                for index in range(8)
            ]
            samples.append(
                {
                    "dataset_role": role,
                    "family": family,
                    "endpoint_identity_sha256": hashlib.sha256(f"{role}:{family}".encode()).hexdigest(),
                    "selection_sha256": hashlib.sha256(f"select:{role}:{family}".encode()).hexdigest(),
                    "array_byte_sha256": arrays,
                    "array_byte_sha256_set": policy.canonical_json_sha256(arrays),
                    "passes": True,
                }
            )
    value = {field: None for field in policy.RAW_REPORT_FIELDS}
    value.update(
        {
            "schema": policy.RAW_SUPERVISION_AUDIT_SCHEMA,
            "verdict": "PASS",
            "dataset_manifest_file_sha256": policy.RAW_V9_MANIFEST_FILE_SHA256,
            "dataset_manifest_content_sha256": policy.RAW_V9_MANIFEST_CONTENT_SHA256,
            "pair_count": 5172,
            "unique_endpoint_count": 9460,
            "scene_shard_count": 88,
            "sample_count": 24,
            "sample_results": samples,
            "sample_results_sha256": policy.RAW_V13_SAMPLE_RESULTS_SHA256,
            "observed_population": {
                "pair_counts": {
                    role: policy.ROLE_COUNTS[role]["pairs"]
                    for role in policy.DEVELOPMENT_ROLES
                },
                "pair_count": 5172,
                "endpoint_reference_counts": {
                    role: policy.ROLE_COUNTS[role]["endpoint_instances"]
                    for role in policy.DEVELOPMENT_ROLES
                },
                "endpoint_reference_count": 10344,
                "unique_endpoint_counts": {
                    role: policy.ROLE_COUNTS[role]["unique_endpoints"]
                    for role in policy.DEVELOPMENT_ROLES
                },
                "unique_endpoint_count": 9460,
                "role_count": 3,
                "family_counts": {role: 8 for role in policy.DEVELOPMENT_ROLES},
                "scene_shard_count": 88,
            },
            "strict_integer_cardinalities": True,
            "unaliased_descriptor_bound_dataset_leaves": True,
            "full_byte_inventory_revalidated": True,
            "pair_endpoint_joins_reconstructed": True,
            "all_stored_evidence_and_rasters_recomputed": True,
            "sample_original_geometry_recomputed": True,
            "source_file_count": 354,
            "source_inventory_before_after_sha256": "3" * 64,
            "source_payload_opens": {
                "complete_inventory_hash_passes": 2,
                "permitted_source_files_per_pass": 354,
                "sample_endpoint_count": 24,
                "rgb_byte_opens": 0,
                "rgb_decodes": 0,
                "label_shard_payload_opens": 0,
                "g2_payload_opens": 0,
                "checkpoint_model_runtime_heldout_hardware_production_opens": 0,
            },
            "authorization_v13": {
                "file_sha256": policy.RAW_CHAIN_SOURCE_BINDINGS[
                    policy.RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH
                ],
                "content_sha256": "4b179c33de00399652f4f915285ca99a4d47cfa95d31878d1f91ca7e8fd9d0e8",
                "source_map_sha256": policy.RAW_V13_AUTHORIZATION_SOURCE_MAP_SHA256,
                "phase_one_zero_target_opens": True,
                "phase_two_fixed_target_count": 14,
                "transitive_v12_target_count": 25,
                "machine_pass_reviews_parsed": 2,
            },
            "frozen_v9_terminal_artifacts": {
                "dataset_manifest_file_sha256": policy.RAW_V9_MANIFEST_FILE_SHA256,
                "dataset_manifest_content_sha256": policy.RAW_V9_MANIFEST_CONTENT_SHA256,
                "terminal_failure_file_sha256": "863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f",
                "terminal_failure_content_sha256": "aaf342f7df88796e0d03259e964ed51e42ebd1faecb33bbfe9ea9cfd0d5e2c72",
                "terminal_failure_retry_authorized": False,
                "success_report_absent": True,
            },
            "frozen_v10_terminal_artifacts": {
                "audit_authorization_file_sha256": "146e0bbf029d28fdf883bfc357b1ddbbce955f86bda00508c6091cb01db4800a",
                "audit_authorization_content_sha256": "8bab96369a5633cb82266fef6ec54964a3c25f27dc0877fde550721f3b6af981",
                "audit_authorization_source_map_sha256": "1b6ffd40b72c7d02dba24d2035ac3442af361b8b804e7df4273f5e73d1cda79b",
                "terminal_failure_file_sha256": "2c391550df540d233ded11bfcf1531dbbb29663a51918fb60e7d8cf4146d0996",
                "terminal_failure_content_sha256": "66370ec52ae06bef81ab75a47cce481830067b88ec2d579ed41a4a58a7cecc83",
                "terminal_failure_retry_authorized": False,
                "success_report_absent": True,
            },
            "frozen_v11_terminal_artifacts": {
                "auditor_block_file_sha256": "169494633f8b9bd50ceac40436e6ef1b168624b8a8c487fedb2033a9c137f3db",
                "auditor_block_content_sha256": "f45610c0db743bfd6ec655bd7d9c3f1e1f3578a3b57e7b55f7d2fcf029d76a94",
                "auditor_block_verdict": "BLOCK",
                "exact_audit_authorized": False,
                "success_report_absent": True,
            },
            "frozen_v12_terminal_artifacts": {
                "audit_authorization_file_sha256": "6b5f317119a00308390b8a32f1057f34455313eb80ec190aa9d8d27052a81575",
                "audit_authorization_content_sha256": "8db4611a321309a76a0dd81e3af0148fce788422e2008ccaef1039e3c5ae493a",
                "audit_authorization_source_map_sha256": "1fc3374101fca166fe74b34b779cf995ec46a12fbb609f5de3bc5a428d225bc2",
                "authorization_witness_file_sha256": "662e6c2f6386b8822b3bd968a4faf0bf3e2e222ff4aac9df8a99cc680c254327",
                "authorization_witness_content_sha256": "4845826d1caeedc58d01b580a8681a71730eb0ba17205bde36d3673c9052741b",
                "launch_failure_file_sha256": "cc6313b1d6e56022204ba82dc57efc6b7cc85a715f078cd865883b61cee88eb3",
                "launch_failure_content_sha256": "b9775ef4705d7505931b64c7ceaad57fb8d18da72429bb877245fb534197b2ee",
                "launch_failure_terminal": True,
                "success_report_absent": True,
                "failure_report_absent": True,
            },
            "closed_publication_transaction_v13": {
                "source_and_candidate_watches_continuous_through_rename": True,
                "retained_source_dataset_and_candidate_descriptors": True,
                "publication_and_source_ancestor_chains_watched": True,
                "single_renameat2_RENAME_NOREPLACE": True,
                "exact_owned_rename_event_sequence": True,
                "post_rename_inventory_and_quiescence": True,
            },
            "content_sha256": policy.RAW_V13_PASS_CONTENT_SHA256,
            **{field: False for field in policy.RAW_DOWNSTREAM_AUTHORITY_FIELDS},
        }
    )
    return value


def _camera_sources() -> dict[str, str]:
    return {
        path: hashlib.sha256(path.encode()).hexdigest()
        for path in policy.CAMERA_V14_PRODUCTION_SOURCE_PATHS
    }


def _camera_rows() -> list[dict[str, Any]]:
    sources = _camera_sources()
    source_rows = [
        {"role": role, "path": path, "file_sha256": sources[path]}
        for role, path in zip(
            ("policy", "trainer", "verifier", "executor"),
            policy.CAMERA_V14_PRODUCTION_SOURCE_PATHS,
        )
    ]
    source_rows_sha = policy.canonical_json_sha256(source_rows)

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode()).hexdigest()

    rows = []
    for index, (seed, fit_size) in enumerate(policy.CAMERA_LADDER_ORDER):
        output_root = (
            policy.CAMERA_V14_OUTPUT_ROOT_RELATIVE_PATH
            if index == 0
            else (
                f"{policy.CAMERA_V14_LADDER_ROOT_RELATIVE_PATH}/attempts/"
                f"seed_{seed}/n{fit_size}"
            )
        )
        attempt_root = (
            f"{output_root}/attempts/seed_{seed}/n{fit_size}"
            if index == 0
            else output_root
        )
        source_review = {
            "path": policy.CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH,
            "file_sha256": digest("source-review-file"),
            "content_sha256": digest("source-review-content"),
            "reviewer": "/root/camera_v14_review",
            "schema": policy.CAMERA_V14_SOURCE_REVIEW_SCHEMA,
            "status": "different_agent_source_review_passed",
            "verdict": "PASS",
        }
        gate = {
            "path": (
                policy.CAMERA_V14_N5_GATE_RELATIVE_PATH
                if index == 0
                else (
                    f"{policy.CAMERA_V14_LADDER_ROOT_RELATIVE_PATH}/gates/"
                    f"seed_{seed}_n{fit_size}.json"
                )
            ),
            "file_sha256": digest(f"gate-file-{index}"),
            "content_sha256": digest(f"gate-content-{index}"),
            "schema": policy.CAMERA_V14_GATE_SCHEMA,
            "gate_schema_sha256": digest("unchanged-gate-schema"),
            "status": "passed",
            "passes": True,
        }
        completion = {
            "path": f"{attempt_root}/completed.json",
            "file_sha256": digest(f"completion-file-{index}"),
            "content_sha256": digest(f"completion-content-{index}"),
        }
        checkpoint = {
            "path": f"{attempt_root}/checkpoint.pt",
            "file_sha256": digest(f"checkpoint-file-{index}"),
            "schema": "camera_v14_checkpoint_v1",
            "initialization_identity": digest(f"initialization-{index}"),
            "emitted_by_attempt_identity": f"camera_v14_seed_{seed}_n{fit_size}",
        }
        rows.append(
            {
                "schema": "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v14_ladder_row_v1",
                "row_index": index,
                "seed": seed,
                "fit_size": fit_size,
                "origin": "preexisting_v14_n5_evidence_only" if index == 0 else "new_ladder_attempt",
                "attempt_identity": f"camera_v14_seed_{seed}_n{fit_size}",
                "reservation": {
                    "path": f"{attempt_root}/reservation.json",
                    "file_sha256": digest(f"reservation-file-{index}"),
                    "content_sha256": digest(f"reservation-content-{index}"),
                },
                "output_root": output_root,
                "completion": completion,
                "production_source_bindings": source_rows,
                "source_review": source_review,
                "gate": gate,
                "checkpoint": checkpoint,
                "rung_review": {
                    "path": f"docs/camera_v14_seed_{seed}_n{fit_size}_review.json",
                    "file_sha256": digest(f"rung-review-file-{index}"),
                    "content_sha256": digest(f"rung-review-content-{index}"),
                    "reviewer": f"/root/camera_v14_rung_{index}_review",
                    "schema": "camera_v14_rung_review_v1",
                    "verdict": "PASS",
                    "production_source_bindings_sha256": source_rows_sha,
                    "source_review_file_sha256": source_review["file_sha256"],
                    "gate_file_sha256": gate["file_sha256"],
                    "completion_file_sha256": completion["file_sha256"],
                    "checkpoint_file_sha256": checkpoint["file_sha256"],
                },
                "fresh_initialization": True,
                "warm_start_used": False,
                "retry_performed": False,
                "reexecution_performed": False,
                "predecessor_checkpoint_opened": False,
                "predecessor_checkpoint_copied": False,
                "predecessor_checkpoint_loaded": False,
                "launched_by_ladder": index != 0,
                "migratable": index == 3,
            }
        )
    return rows


def _camera_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ordered_rung_count": 8,
        "preexisting_seed_20260710_n5_count": 1,
        "additional_attempt_count": 7,
        "seed_20260710_n5_reexecuted": False,
        "all_rungs_fresh_initialization": True,
        "warm_start_used": False,
        "retry_performed": False,
        "both_seed_ladders_pass": True,
        "rows_sha256": policy.canonical_json_sha256(rows),
        "gate_file_sha256": "4" * 64,
        "gate_content_sha256": "5" * 64,
        "independent_review_file_sha256": "6" * 64,
        "independent_review_content_sha256": "7" * 64,
    }


def _primary_row_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[3]
    return {
        "row_index": 3,
        "seed": 20260710,
        "fit_size": 320,
        "path": row["checkpoint"]["path"],
        "file_sha256": row["checkpoint"]["file_sha256"],
        "schema": row["checkpoint"]["schema"],
        "rung_review_file_sha256": row["rung_review"]["file_sha256"],
    }


def _preflight_authorization() -> dict[str, Any]:
    rows = _camera_rows()
    core = {
        "schema": policy.EXACT_BINDING_PREFLIGHT_AUTHORIZATION_SCHEMA,
        "status": "reviewed_authorized_for_one_payload_free_preflight",
        "authorizer": "/root/v4_preflight_authorizer",
        "reviewer": "/root/v4_preflight_authorization_review",
        "governing_amendment": {
            "path": policy.V4_AMENDMENT_RELATIVE_PATH,
            "file_sha256": policy.V4_AMENDMENT_SHA256,
        },
        "blocked_manifest": {
            "path": policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH,
            "file_sha256": "1" * 64,
            "content_sha256": "2" * 64,
        },
        "implementation_review": {
            "path": policy.IMPLEMENTATION_REVIEW_RELATIVE_PATH,
            "file_sha256": "3" * 64,
            "content_sha256": "4" * 64,
        },
        "reviewed_sources": {
            path: hashlib.sha256(path.encode()).hexdigest()
            for path in policy.IMPLEMENTATION_SOURCE_PATHS
        },
        "raw_v13_terminal_bindings": policy.RAW_V13_TERMINAL_BINDINGS,
        "camera_v14_source_bindings": _camera_sources(),
        "camera_ladder_rows": rows,
        "camera_ladder_aggregate": _camera_aggregate(rows),
        "primary_migratable_checkpoint": _primary_row_binding(rows),
        "authority": policy.PREFLIGHT_AUTHORITY,
    }
    return policy.content_value(core)


def _final_authorization() -> dict[str, Any]:
    rows = _camera_rows()
    aggregate = _camera_aggregate(rows)
    reviewed_sources = {
        path: hashlib.sha256(path.encode()).hexdigest()
        for path in policy.IMPLEMENTATION_SOURCE_PATHS
    }
    bindings = {name: "8" * 64 for name in policy.FINAL_REQUIRED_BINDING_NAMES}
    bindings.update(policy.FINAL_FROZEN_RAW_BINDINGS)
    source_names = {
        policy.POLICY_RELATIVE_PATH: "implementation_policy_source_sha256",
        policy.LOSS_ADAPTER_RELATIVE_PATH: "loss_adapter_source_sha256",
        policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH: "preflight_executor_source_sha256",
        policy.PREFLIGHT_VERIFIER_RELATIVE_PATH: "preflight_verifier_source_sha256",
        policy.EXACT_EXECUTOR_RELATIVE_PATH: "exact_executor_source_sha256",
        policy.EXACT_TRAINER_RELATIVE_PATH: "exact_trainer_source_sha256",
        policy.EXACT_VERIFIER_RELATIVE_PATH: "exact_verifier_source_sha256",
    }
    for path, name in source_names.items():
        bindings[name] = reviewed_sources[path]
    preflight = {
        "completion_file_sha256": "9" * 64,
        "completion_content_sha256": "a" * 64,
        "receipt_file_sha256": "b" * 64,
        "receipt_content_sha256": "c" * 64,
        "verification_file_sha256": "d" * 64,
        "verification_content_sha256": "e" * 64,
        "independently_verified_pass": True,
        "payload_open_count": 0,
    }
    bindings["preflight_completed_file_sha256"] = preflight["completion_file_sha256"]
    bindings["preflight_receipt_file_sha256"] = preflight["receipt_file_sha256"]
    bindings["preflight_independent_review_file_sha256"] = preflight["verification_file_sha256"]
    bindings["camera_v14_source_review_file_sha256"] = rows[0]["source_review"]["file_sha256"]
    bindings["camera_v14_source_review_content_sha256"] = rows[0]["source_review"]["content_sha256"]
    bindings["camera_v14_n5_gate_pass_file_sha256"] = rows[0]["gate"]["file_sha256"]
    bindings["camera_v14_n5_gate_pass_content_sha256"] = rows[0]["gate"]["content_sha256"]
    bindings["camera_v14_two_seed_ladder_pass_file_sha256"] = aggregate["gate_file_sha256"]
    bindings["camera_v14_two_seed_ladder_pass_content_sha256"] = aggregate["gate_content_sha256"]
    bindings["camera_v14_ladder_independent_review_file_sha256"] = aggregate["independent_review_file_sha256"]
    bindings["v4_primary_seed_20260710_n320_checkpoint_file_sha256"] = rows[3]["checkpoint"]["file_sha256"]
    bindings["implementation_independent_review_file_sha256"] = "f" * 64
    core = {
        "schema": policy.FINAL_EXACT_EXECUTION_AUTHORIZATION_SCHEMA,
        "status": "reviewed_authorized_for_one_exact_matched_training_attempt",
        "authorizer": "/root/v4_final_authorizer",
        "reviewer": "/root/v4_final_authorization_review",
        "governing_amendment": {"path": policy.V4_AMENDMENT_RELATIVE_PATH, "file_sha256": policy.V4_AMENDMENT_SHA256},
        "blocked_manifest": {"path": policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH, "file_sha256": "1" * 64, "content_sha256": "2" * 64},
        "implementation_review": {"path": policy.IMPLEMENTATION_REVIEW_RELATIVE_PATH, "file_sha256": bindings["implementation_independent_review_file_sha256"], "content_sha256": "3" * 64},
        "preflight_authorization": {"path": policy.EXACT_BINDING_PREFLIGHT_AUTHORIZATION_RELATIVE_PATH, "file_sha256": "4" * 64, "content_sha256": "5" * 64},
        "preflight_evidence": preflight,
        "reviewed_sources": reviewed_sources,
        "required_exact_bindings": bindings,
        "raw_v13_terminal_bindings": policy.RAW_V13_TERMINAL_BINDINGS,
        "camera_v14_source_bindings": _camera_sources(),
        "camera_ladder_rows": rows,
        "camera_ladder_aggregate": aggregate,
        "primary_migratable_checkpoint": _primary_row_binding(rows),
        "raw_v13_dataset_use_grant": policy.RAW_DATASET_USE_GRANT,
        "authority": policy.FINAL_EXACT_AUTHORITY,
    }
    return policy.content_value(core)


def test_frozen_parent_closure_and_v4_amendment_hashes() -> None:
    assert _sha256(ROOT / policy.V4_AMENDMENT_RELATIVE_PATH) == policy.V4_AMENDMENT_SHA256
    assert _sha256(ROOT / policy.V3_AMENDMENT_RELATIVE_PATH) == policy.V3_AMENDMENT_SHA256
    assert _sha256(ROOT / policy.CAMERA_V14_AMENDMENT_RELATIVE_PATH) == policy.CAMERA_V14_AMENDMENT_SHA256
    bindings = policy.reviewed_source_bindings()
    assert all(not path.startswith(".generated/") for path in bindings)
    assert {path: _sha256(ROOT / path) for path in bindings} == bindings
    assert len(policy.RAW_V13_AUTHORIZATION_SOURCE_ROWS) == 14


def test_blocked_manifest_has_literal_null_set_and_zero_authority() -> None:
    manifest = policy.parse_canonical_json(MANIFEST.read_bytes(), name="blocked V4 manifest")
    assert manifest == policy.content_value(policy.execution_manifest_core())
    assert manifest["status"] == "blocked_required_bindings_unset"
    assert len(policy.BLOCKED_FUTURE_BINDING_NAMES) == 32
    assert set(manifest["future_bindings"]) == set(policy.BLOCKED_FUTURE_BINDING_NAMES)
    assert all(value is None for value in manifest["future_bindings"].values())
    assert manifest["unresolved_future_bindings"] == list(policy.BLOCKED_FUTURE_BINDING_NAMES)
    assert {name: manifest[name] for name in policy.MANIFEST_AUTHORITY} == policy.MANIFEST_AUTHORITY
    assert manifest["camera_v13_evidence_accepted"] is False
    assert manifest["source_review_can_authorize_preflight"] is False
    with pytest.raises(PermissionError, match="blocked source-time"):
        policy.validate_execution_manifest(manifest, require_ready=True)
    with pytest.raises(TypeError):
        policy.execution_manifest_core(required_bindings={})


def test_source_review_is_non_authoritative_and_not_a_preflight_credential() -> None:
    source_bindings = {path: "a" * 64 for path in policy.IMPLEMENTATION_SOURCE_PATHS}
    core = policy.expected_implementation_review_core(
        reviewer="/root/full_training_v4_independent_review",
        source_bindings=source_bindings,
        author_test_file_sha256="b" * 64,
        handoff_file_sha256="c" * 64,
        blocked_manifest_file_sha256="d" * 64,
        blocked_manifest_content_sha256="e" * 64,
    )
    assert core["authority"] == policy.SOURCE_REVIEW_AUTHORITY
    assert all(value is False for value in core["authority"].values())
    assert core["reviewed_production_sources"] == source_bindings
    assert core["camera_dependency"]["ordered_ladder"] == [
        list(item) for item in policy.CAMERA_LADDER_ORDER
    ]
    with pytest.raises(PermissionError):
        policy.expected_implementation_review_core(
            reviewer=policy.IMPLEMENTATION_AUTHOR,
            source_bindings=source_bindings,
            author_test_file_sha256="b" * 64,
            handoff_file_sha256="c" * 64,
            blocked_manifest_file_sha256="d" * 64,
            blocked_manifest_content_sha256="e" * 64,
        )
    preflight = _source("preflight")
    assert "validate_implementation_review" not in preflight
    assert "IMPLEMENTATION_REVIEW_RELATIVE_PATH" not in preflight
    assert "validate_exact_binding_preflight_authorization" in preflight
    assert "--preflight-authorization-sha256" in preflight

    review_value = policy.content_value(core)
    assert policy.validate_implementation_review(review_value) == review_value
    with pytest.raises(ValueError):
        policy.validate_exact_binding_preflight_authorization(review_value)


def test_later_authorization_schemas_are_separate_and_fail_closed() -> None:
    preflight = _preflight_authorization()
    assert policy.validate_exact_binding_preflight_authorization(preflight) == preflight
    changed = copy.deepcopy(preflight)
    changed["authority"]["exact_execution_authorized"] = True
    with pytest.raises(PermissionError):
        policy.validate_exact_binding_preflight_authorization(changed)

    final = _final_authorization()
    assert policy.validate_final_exact_execution_authorization(final) == final
    changed = copy.deepcopy(final)
    changed["authority"]["retry_authorized"] = True
    with pytest.raises(PermissionError):
        policy.validate_final_exact_execution_authorization(changed)
    changed = copy.deepcopy(final)
    changed["camera_ladder_rows"][4]["warm_start_used"] = True
    with pytest.raises(PermissionError):
        policy.validate_final_exact_execution_authorization(changed)
def test_raw_v13_source_chain_is_exact_and_mutations_fail_closed() -> None:
    values = {
        "builder_review": _load(policy.RAW_BUILDER_V9_REVIEW_RELATIVE_PATH),
        "auditor_review": _load(policy.RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH),
        "authorization": _load(policy.RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH),
        "fingerprint": _load(policy.RAW_AUDITOR_V13_FINGERPRINT_RELATIVE_PATH),
    }
    result = policy.validate_raw_v13_source_chain(**values)
    assert set(result) == set(values)
    changed = copy.deepcopy(values)
    changed["authorization"]["retry_authorized"] = True
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_source_chain(**changed)
    changed = copy.deepcopy(values)
    changed["authorization"]["source_map"][0]["role"] = "role_swapped"
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_source_chain(**changed)
    changed = copy.deepcopy(values)
    changed["builder_review"]["candidate"].append(changed["builder_review"]["candidate"][0])
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_source_chain(**changed)


def test_raw_v13_manifest_and_terminal_population_are_complete() -> None:
    manifest = _synthetic_raw_manifest()
    report = _synthetic_raw_report()
    assert policy.validate_raw_v13_manifest(manifest) is manifest
    assert policy.validate_raw_v13_terminal_report(report) is report
    changed = copy.deepcopy(report)
    changed["retry_authorized"] = True
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_terminal_report(changed)
    changed = copy.deepcopy(report)
    changed["observed_population"]["endpoint_reference_counts"]["train"] -= 1
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_terminal_report(changed)
    changed = copy.deepcopy(report)
    changed["authorization_v13"]["transitive_v12_target_count"] -= 1
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_terminal_report(changed)
    changed = copy.deepcopy(report)
    changed["closed_publication_transaction_v13"] = {}
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_terminal_report(changed)
    changed = copy.deepcopy(manifest)
    changed["array_layout"][-1]["trailing_shape"] = [32, 32]
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["pair_index"]["row_count"] -= 1
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["independent_audit_precommit"]["records"].pop()
    with pytest.raises(PermissionError):
        policy.validate_raw_v13_manifest(changed)


def test_camera_v14_ladder_requires_exact_order_and_per_rung_closure() -> None:
    sources = _camera_sources()
    rows = _camera_rows()
    assert policy.validate_camera_v14_ladder_rows(rows, reviewed_source_bindings=sources) == rows
    changed = copy.deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(PermissionError):
        policy.validate_camera_v14_ladder_rows(changed, reviewed_source_bindings=sources)
    changed = copy.deepcopy(rows)
    changed[3]["migratable"] = False
    with pytest.raises(PermissionError):
        policy.validate_camera_v14_ladder_rows(changed, reviewed_source_bindings=sources)
    changed = copy.deepcopy(rows)
    changed[5]["checkpoint"]["path"] = rows[4]["checkpoint"]["path"]
    with pytest.raises(PermissionError):
        policy.validate_camera_v14_ladder_rows(changed, reviewed_source_bindings=sources)
    changed = copy.deepcopy(rows)
    changed[2]["gate"]["extra"] = False
    with pytest.raises(ValueError):
        policy.validate_camera_v14_ladder_rows(changed, reviewed_source_bindings=sources)
    changed = copy.deepcopy(rows)
    changed[1]["reservation"]["path"] = "../escape.json"
    with pytest.raises(PermissionError):
        policy.validate_camera_v14_ladder_rows(changed, reviewed_source_bindings=sources)


def test_loss_and_reduction_contract_are_exact() -> None:
    promoted = policy.JOINT_LOSS_CONTRACT["promoted_jepa"]
    assert promoted["v4_components"] == {
        "hierarchical_first_hit_nll": 0.25,
        "target_bin_offset_smooth_l1": 0.25,
        "ground_clear_distance_state_balanced_bce": 0.25,
        "derived_raster_hierarchical_bce": 0.25,
        "derived_raster_cell_nll": 0.25,
    }
    assert promoted["current_and_next_computed_separately_at_batch_size"] == 4
    assert promoted["current_next_scalar_average"] == [0.5, 0.5]
    assert promoted["microbatch_scalar_average"] == [0.25] * 4
    assert promoted["synthetic_b16_nonlinear_pooling_authorized"] is False
    assert policy.average_current_next_b4_scalars(2.0, 6.0) == 4.0

    microbatches = [
        [("A", 0.0)],
        [("A", 0.0)],
        [("A", 0.0), ("B", 100.0)],
        [("A", 0.0)],
    ]
    scalars = [_group_balanced(rows) for rows in microbatches]
    correct = policy.average_four_microbatch_scalars(scalars)
    pooled = _group_balanced(row for microbatch in microbatches for row in microbatch)
    assert correct == 12.5
    assert pooled == 50.0
    assert correct != pooled
    with pytest.raises(ValueError, match="exactly four"):
        policy.average_four_microbatch_scalars([pooled])

    leaves = [torch.tensor(float(index), requires_grad=True) for index in range(4)]
    mean = loss_adapter.average_four_microbatch_tensor_scalars_v4(leaves)
    mean.backward()
    assert float(mean.detach()) == 1.5
    assert [float(value.grad) for value in leaves] == [0.25] * 4


def test_cpu_synthetic_adapter_has_five_gradient_terms_and_separate_b4() -> None:
    torch.manual_seed(20260714)
    model = _small_model().train()
    pair = _pair(model, batch=4)
    current_supervision = _supervision(pair.current)
    next_supervision = _supervision(pair.next)
    camera = loss_adapter.observable_camera_ray_v4_loss_v4(
        model,
        pair,
        current_supervision,
        next_supervision,
    )
    joint = loss_adapter.combine_joint_losses_v4(
        model,
        pair,
        current_supervision,
        next_supervision,
    )
    for frame in (camera.current, camera.next):
        base = 0.25 * (
            frame.hierarchical_first_hit_nll
            + frame.target_bin_offset_smooth_l1
            + frame.ground_clear_distance_state_balanced_bce
            + frame.derived_raster_hierarchical_bce.total
        )
        assert torch.allclose(frame.retained_v11_base_total, base)
        assert torch.allclose(frame.total, base + 0.25 * frame.derived_raster_cell_nll)
        for term in (
            frame.hierarchical_first_hit_nll,
            frame.target_bin_offset_smooth_l1,
            frame.ground_clear_distance_state_balanced_bce,
            frame.derived_raster_hierarchical_bce.total,
            frame.derived_raster_cell_nll,
        ):
            assert term.requires_grad
    assert torch.allclose(camera.total, 0.5 * camera.current.total + 0.5 * camera.next.total)
    assert torch.allclose(joint.total, pair.jepa.total + camera.total)
    joint.total.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    wrong = _pair(model, batch=2)
    with pytest.raises(ValueError, match="synthetic-B16 pooling is forbidden"):
        loss_adapter.observable_camera_ray_v4_loss_v4(
            model,
            wrong,
            _supervision(wrong.current),
            _supervision(wrong.next),
        )


def test_pre_g2_checkpoint_has_individual_downstream_denials() -> None:
    core = policy.pre_g2_candidate_checkpoint_core(
        model_config={"model": "shared-v5"},
        deployment_state_sha256="1" * 64,
        selection={"selected_update": 8000},
        calibration={"global_threshold": 0.9},
    )
    assert core["schema"] == policy.PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA
    assert core["checkpoint_kind"] == "pre_g2_candidate"
    assert core["development_only"] is True
    assert core["g2_attempted"] is False
    assert core["g2_gate_receipt"] is None
    assert core["post_g2_qualified"] is False
    assert core["runtime_ready"] is False
    for field in (
        "heldout_authorized",
        "navigation_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
    ):
        assert core[field] is False
    assert "qualified_checkpoint.pt" not in policy.EXACT_INVENTORY
    assert "pre_g2_candidate_checkpoint.pt" in policy.EXACT_INVENTORY


def test_production_sources_are_additive_fail_closed_and_parseable() -> None:
    combined = "\n".join(_source(name) for name in SOURCES)
    assert "development_fit_v2" not in combined
    assert "qualified_checkpoint.pt" not in combined
    assert "CHECKPOINT_V5_SCHEMA" not in combined
    assert "ordered_first_hit_nll" not in combined
    assert "from lewm.benchmarks.go2_shared_jepa_v5_full_training_v3" not in combined
    assert "from lewm.models.shared_observable_camera_ray_jepa_v5_full_training_v3" not in combined
    assert "validate_raw_v13_source_chain" in _source("trainer")
    assert "validate_raw_v13_terminal_report" in _source("trainer")
    assert "validate_camera_v14_ladder_rows" in _source("trainer")
    assert "FINAL_EXACT_EXECUTION_AUTHORIZATION_RELATIVE_PATH" in _source("executor")
    assert "validate_final_exact_execution_authorization" in _source("executor")
    assert "(backward / policy.ACCUMULATION_STEPS).backward()" in _source("trainer")
    assert "(joint.total / policy.ACCUMULATION_STEPS).backward()" in _source("preflight")
    for path in SOURCES.values():
        raw = path.read_bytes()
        raw.decode("ascii")
        ast.parse(raw, filename=str(path))
    for forbidden in (
        "--backend",
        "--module",
        "--callback",
        "--test-only",
        "--fixture",
        "autocast(",
        "cuda:1",
    ):
        assert forbidden not in combined
    assert math.isfinite(policy.learning_rate(8000))
    assert json.loads(MANIFEST.read_text(encoding="ascii"))["heldout_authorized"] is False
