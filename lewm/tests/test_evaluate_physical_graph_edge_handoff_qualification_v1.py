from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as M
from scripts import evaluate_physical_graph_edge_handoff_qualification_v1 as E


def _attach(value: dict) -> dict:
    row = copy.deepcopy(value)
    row["content_digest"] = E.canonical_digest(row)
    return row


def _write_document(path: Path, value: dict) -> None:
    path.write_bytes(E.canonical_document_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(E.canonical_document_bytes(row) for row in rows))


def _source_observation(commit: str) -> dict:
    paths = {
        E.REDUCER_SOURCE_PATH,
        E.CONTRACT_SOURCE_PATH,
        E.METRICS_SOURCE_PATH,
        E.CUSTODY_SOURCE_PATH,
        E.BASE_CUSTODY_SOURCE_PATH,
    }
    return {
        "head_commit": commit,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            path: {"path": path, "bytes": 1, "sha256": "0" * 64}
            for path in paths
        },
        "metrics_module": "synthetic_physical_handoff_metrics",
    }


def _document_spec(fields: set[str], row_fields: set[str], container: str, count):
    return {
        "root": sorted(fields | {container, "content_digest"}),
        "row": sorted(row_fields),
        "container": container,
        "count": count,
    }


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "physical_graph_edge_handoff_qualification_v1"
    root.mkdir()
    receipt = tmp_path / "physical_graph_edge_handoff_qualification_v1_regeneration_receipt.json"
    commit = "a" * 40
    state_ids = [f"state-{index:03d}" for index in range(64)]
    target_ids = ["TARGET_0", "TARGET_1", "TARGET_2"]
    conditions = ["CONDITION_0", "CONDITION_1", "CONDITION_2", "CONDITION_3"]
    repeat_branches = ["BRANCH_0", "BRANCH_1"]

    v2_root = tmp_path / "occluded_goal_topological_belief_v2"
    v2_root.mkdir()
    v2_bindings = {}
    for index, leaf in enumerate(E.EXPECTED_V2_CONTEXT_LEAVES):
        raw = f"synthetic-v2-{index}\n".encode()
        (v2_root / leaf).write_bytes(raw)
        v2_bindings[leaf] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    documents: dict[str, dict] = {}
    documents["contract.json"] = _attach(
        {
            "schema": "synthetic.runtime_contract.v1",
            "source_freeze_commit": commit,
            "scientific_contract": {
                "v2_context_binding": {
                    "source_freeze_commit": "b" * 40,
                    "result_commit": "c" * 40,
                    "runtime_reuse_authorized": False,
                    "bindings": v2_bindings,
                }
            },
            "predecessor_result_binding": {
                "role": "v2_scientific_context_only",
                "kind": "frozen_result_identity_no_runtime_reuse",
                "path": str(v2_root / "result.json"),
                **v2_bindings["result.json"],
            },
        }
    )
    panel_rows = [
        {
            "state_id": state_id,
            "role": "DEVELOPMENT" if index < 48 else "DEVELOPMENT_HELDOUT",
        }
        for index, state_id in enumerate(state_ids)
    ]
    qualification_rows = [
        {"teacher_trace_index": index}
        for index in range(E.EXPECTED_TEACHER_TRACE_COUNT)
    ]
    documents["panel_manifest.json"] = _attach(
        {
            "prospective_pool_selection": {
                "qualification_rows": qualification_rows,
            },
            "states": panel_rows,
        }
    )
    documents["split_manifest.json"] = _attach(
        {"assignments": [{"state_id": value} for value in state_ids]}
    )
    documents["graph_manifest.json"] = _attach(
        {"graphs": [{"state_id": value} for value in state_ids]}
    )
    documents["state_snapshot_index.json"] = _attach(
        {"records": [{"state_id": value} for value in state_ids]}
    )
    documents["teacher_trace_index.json"] = _attach(
        {
            "records": [
                {
                    "state_id": f"pool-state-{index:03d}",
                    "teacher_trace_index": index,
                }
                for index in range(E.EXPECTED_TEACHER_TRACE_COUNT)
            ]
        }
    )
    documents["edge_port_index.json"] = _attach(
        {"records": [{"state_id": value} for value in state_ids]}
    )
    documents["waypoint_contracts.json"] = _attach(
        {
            "rows": [
                {"state_id": state_id, "target_id": target_id}
                for state_id in state_ids
                for target_id in target_ids
            ]
        }
    )
    documents["pixel_index.json"] = _attach(
        {
            "unique_pixel_count": 64,
            "records": [{"state_id": value} for value in state_ids],
        }
    )
    documents["latent_index.json"] = _attach(
        {
            "records": [
                {"canonical_pixel_index": index} for index in range(64)
            ]
        }
    )
    documents["development_target_selection.json"] = _attach(
        {
            "state_target_rows": [
                {"state_id": state_id, "target_id": target_id}
                for state_id in state_ids[:48]
                for target_id in target_ids
            ],
            "target_summaries": [{"target_id": value} for value in target_ids],
        }
    )

    candidate_rows = [
        {"state_id": state_id, "candidate_index": candidate_index}
        for state_id in state_ids
        for candidate_index in range(12)
    ]
    heldout_rows = [
        {"state_id": state_id, "condition_id": condition}
        for state_id in state_ids[48:]
        for condition in conditions
    ]
    repeat_rows = [
        {
            "state_id": state_id,
            "branch_selector_id": branch,
            "repeat_index": repeat_index,
        }
        for state_id in state_ids[48:]
        for branch in repeat_branches
        for repeat_index in range(2)
    ]

    snapshot_payload = np.arange(64, dtype=np.uint8)
    snapshot_offsets = np.arange(65, dtype=np.int64)
    snapshot_projected_members = {
        member: (np.arange(64, dtype=np.float64) + member_index).reshape(64, 1)
        for member_index, member in enumerate(
            (
                "base_pose_world",
                "base_twist_world",
                "camera_world_transform",
                "controller_observation",
                "joint_position",
                "joint_velocity",
                "policy_last_action",
                "previous_policy_action",
                "previous_applied_command",
                "command_history",
                "control_history",
                "low_level_policy_state",
            )
        )
    }
    teacher_timestamp = np.arange(
        E.EXPECTED_TEACHER_TRACE_COUNT * 2, dtype=np.uint16
    ).astype(np.uint8)
    teacher_offsets = np.arange(
        E.EXPECTED_TEACHER_TRACE_COUNT + 1, dtype=np.int64
    ) * 2
    rgb_rows = np.arange(64, dtype=np.uint8).reshape(64, 1)
    raw_tokens = np.arange(64, dtype=np.uint8).reshape(64, 1)
    spatial_descriptors = (np.arange(64, dtype=np.uint8) + 1).reshape(64, 1)
    candidate_lengths = np.array(
        [E.EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES]
        * E.EXPECTED_RESET_FIXTURE_TRACE_COUNT
        + [E.EXPECTED_PHYSICS_SAMPLES_PER_BRANCH]
        * (E.EXPECTED_CANDIDATE_TRACE_COUNT - E.EXPECTED_RESET_FIXTURE_TRACE_COUNT),
        dtype=np.int64,
    )
    candidate_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(candidate_lengths)]
    )
    candidate_timestamp = np.zeros(int(candidate_offsets[-1]), dtype=np.uint8)
    np.savez_compressed(
        root / "state_snapshots.npz",
        snapshot_payload_bytes=snapshot_payload,
        snapshot_offsets=snapshot_offsets,
        **snapshot_projected_members,
    )
    np.savez_compressed(
        root / "teacher_traces.npz",
        trace_offsets=teacher_offsets,
        timestamp_s=teacher_timestamp,
    )
    np.savez_compressed(root / "rgb_observations.npz", rgb=rgb_rows)
    np.savez_compressed(
        root / "canonical_latents.npz",
        raw_tokens=raw_tokens,
        spatial_descriptors=spatial_descriptors,
    )
    np.savez_compressed(
        root / "candidate_traces.npz",
        trace_offsets=candidate_offsets,
        timestamp_s=candidate_timestamp,
    )

    npz_authorities = {
        "state_snapshots.npz": {
            "snapshot_payload_bytes": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": ["B"],
                "hash_mode": "offset_slices", "offsets_member": "snapshot_offsets",
                "slice_digest_domain": "sha256_of_exact_serialized_snapshot_bytes",
            },
            "snapshot_offsets": {
                "descr": "<i8", "digest_dtype": "int64", "shape": [65],
                "hash_mode": "whole",
            },
            **{
                member: {
                    "descr": "<f8",
                    "digest_dtype": "float64",
                    "shape": [64, 1],
                    "hash_mode": "rows_axis0",
                }
                for member in snapshot_projected_members
            },
        },
        "teacher_traces.npz": {
            "trace_offsets": {
                "descr": "<i8", "digest_dtype": "int64",
                "shape": [E.EXPECTED_TEACHER_TRACE_COUNT + 1],
                "hash_mode": "whole",
            },
            "timestamp_s": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": ["T"],
                "hash_mode": "offset_slices", "offsets_member": "trace_offsets",
            },
        },
        "rgb_observations.npz": {
            "rgb": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": [64, 1],
                "hash_mode": "rows_axis0",
            }
        },
        "canonical_latents.npz": {
            "raw_tokens": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": ["U", 1],
                "hash_mode": "rows_axis0",
            },
            "spatial_descriptors": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": ["U", 1],
                "hash_mode": "rows_axis0",
            },
        },
        "candidate_traces.npz": {
            "trace_offsets": {
                "descr": "<i8", "digest_dtype": "int64",
                "shape": [E.EXPECTED_CANDIDATE_TRACE_COUNT + 1],
                "hash_mode": "whole",
            },
            "timestamp_s": {
                "descr": "|u1", "digest_dtype": "uint8", "shape": ["P"],
                "hash_mode": "offset_slices", "offsets_member": "trace_offsets",
            },
        },
    }

    def file_binding(leaf: str) -> dict:
        raw = (root / leaf).read_bytes()
        return {"path": leaf, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def array_digest(array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        return E._array_row_digest(
            contiguous.tobytes(order="C"), contiguous.shape, contiguous.dtype.str
        )

    snapshot_file = file_binding("state_snapshots.npz")
    teacher_file = file_binding("teacher_traces.npz")
    rgb_file = file_binding("rgb_observations.npz")
    latent_file = file_binding("canonical_latents.npz")
    candidate_file = file_binding("candidate_traces.npz")

    def slice_binding(file_row: dict, member: str, array: np.ndarray, start: int, stop: int) -> dict:
        return {
            "file_path": file_row["path"],
            "file_sha256": file_row["sha256"],
            "member": member,
            "start": start,
            "stop": stop,
            "slice_sha256": array_digest(array[start:stop]),
        }

    def raw_slice_binding(
        file_row: dict, member: str, array: np.ndarray, start: int, stop: int
    ) -> dict:
        payload = np.ascontiguousarray(array[start:stop]).tobytes(order="C")
        return {
            "file_path": file_row["path"],
            "file_sha256": file_row["sha256"],
            "member": member,
            "start": start,
            "stop": stop,
            "slice_sha256": hashlib.sha256(payload).hexdigest(),
        }

    snapshot_records = []
    snapshot_index_fields = {
        "base_pose_world_sha256": "base_pose_world",
        "base_twist_world_sha256": "base_twist_world",
        "camera_world_transform_sha256": "camera_world_transform",
        "controller_observation_sha256": "controller_observation",
        "joint_position_sha256": "joint_position",
        "joint_velocity_sha256": "joint_velocity",
        "policy_last_action_sha256": "policy_last_action",
        "previous_policy_action_sha256": "previous_policy_action",
        "previous_applied_command_sha256": "previous_applied_command",
        "command_history_sha256": "command_history",
        "control_history_sha256": "control_history",
        "low_level_policy_state_sha256": "low_level_policy_state",
    }
    for state_index, state_id in enumerate(state_ids):
        snapshot_payload_sha = hashlib.sha256(
            np.ascontiguousarray(
                snapshot_payload[state_index : state_index + 1]
            ).tobytes(order="C")
        ).hexdigest()
        trials = []
        for trial_index in range(2):
            trace_index = state_index * 2 + trial_index
            start, stop = candidate_offsets[trace_index : trace_index + 2]
            binding = slice_binding(
                candidate_file, "timestamp_s", candidate_timestamp, int(start), int(stop)
            )
            trials.append(
                {
                    "trace_index": trace_index,
                    "trace_slice": binding,
                    "trace_array_slice_sha256s": {"timestamp_s": binding["slice_sha256"]},
                    "termination_reason": "HORIZON_COMPLETE",
                    "stuck": False,
                    "restored_snapshot_sha256": snapshot_payload_sha,
                    "post_restore_state_sha256": hashlib.sha256(
                        f"post-restore-{state_index}".encode()
                    ).hexdigest(),
                }
            )
        snapshot_records.append(
            {
                "state_id": state_id,
                "snapshot_payload": raw_slice_binding(
                    snapshot_file,
                    "snapshot_payload_bytes",
                    snapshot_payload,
                    state_index,
                    state_index + 1,
                ),
                "teacher_initial_state_sha256": snapshot_payload_sha,
                **{
                    field: array_digest(snapshot_projected_members[member][state_index])
                    for field, member in snapshot_index_fields.items()
                },
                "reset_trials": trials,
                "reset_pair_comparison": {
                    "state_id": state_id,
                    "trial_indices": [0, 1],
                    "physics_sample_count": E.EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES,
                    "exact_member_equal": {"timestamp_s": True},
                    "termination_reason_equal": True,
                    "stuck_equal": True,
                    "passed": True,
                },
            }
        )
    documents["state_snapshot_index.json"] = _attach(
        {"snapshots_file": snapshot_file, "records": snapshot_records}
    )

    teacher_records = []
    for trace_index in range(E.EXPECTED_TEACHER_TRACE_COUNT):
        start, stop = teacher_offsets[trace_index : trace_index + 2]
        binding = slice_binding(
            teacher_file, "timestamp_s", teacher_timestamp, int(start), int(stop)
        )
        teacher_records.append(
            {
                "state_id": f"pool-state-{trace_index:03d}",
                "teacher_trace_index": trace_index,
                "sample_count": int(stop - start),
                "trace_slice": binding,
                "trace_array_slice_sha256s": {"timestamp_s": binding["slice_sha256"]},
            }
        )
    documents["teacher_trace_index.json"] = _attach(
        {"traces_file": teacher_file, "records": teacher_records}
    )

    rgb_hashes = [array_digest(row) for row in rgb_rows]
    canonical_pixels = sorted(set(rgb_hashes))
    pixel_to_index = {digest: index for index, digest in enumerate(canonical_pixels)}
    documents["pixel_index.json"] = _attach(
        {
            "rgb_file": rgb_file,
            "unique_pixel_count": len(canonical_pixels),
            "records": [
                {
                    "state_id": state_id,
                    "rgb_row_index": index,
                    "pixel_sha256": rgb_hashes[index],
                    "row_sha256": rgb_hashes[index],
                    "canonical_pixel_index": pixel_to_index[rgb_hashes[index]],
                }
                for index, state_id in enumerate(state_ids)
            ],
        }
    )
    raw_hashes = [array_digest(row) for row in raw_tokens]
    descriptor_hashes = [array_digest(row) for row in spatial_descriptors]
    documents["latent_index.json"] = _attach(
        {
            "latents_file": latent_file,
            "preprocessing_authority": {"synthetic": True},
            "external_encoder_source": {
                "repository_path": str(tmp_path),
                "commit": "d" * 40,
                "worktree_clean": True,
            },
            "records": [
                {
                    "canonical_pixel_index": index,
                    "pixel_sha256": canonical_pixels[index],
                    "preprocessed_tensor_sha256": hashlib.sha256(
                        f"synthetic-preprocessed-{index}".encode()
                    ).hexdigest(),
                    "raw_token_row_index": index,
                    "raw_token_sha256": raw_hashes[index],
                    "spatial_descriptor_row_index": index,
                    "spatial_descriptor_sha256": descriptor_hashes[index],
                }
                for index in range(len(canonical_pixels))
            ],
        }
    )

    for row_index, row in enumerate(candidate_rows):
        trace_index = E.EXPECTED_RESET_FIXTURE_TRACE_COUNT + row_index
        start, stop = candidate_offsets[trace_index : trace_index + 2]
        binding = slice_binding(
            candidate_file, "timestamp_s", candidate_timestamp, int(start), int(stop)
        )
        row.update(
            {
                "trace_index": trace_index,
                "trace_slice": binding,
                "trace_array_slice_sha256s": {"timestamp_s": binding["slice_sha256"]},
                "physics_sample_count": E.EXPECTED_PHYSICS_SAMPLES_PER_BRANCH,
            }
        )
    for row_index, row in enumerate(repeat_rows):
        trace_index = (
            E.EXPECTED_RESET_FIXTURE_TRACE_COUNT
            + E.EXPECTED_FANOUT_ROW_COUNT
            + row_index
        )
        start, stop = candidate_offsets[trace_index : trace_index + 2]
        binding = slice_binding(
            candidate_file, "timestamp_s", candidate_timestamp, int(start), int(stop)
        )
        row.update(
            {
                "trace_index": trace_index,
                "trace_slice": binding,
                "trace_array_slice_sha256s": {"timestamp_s": binding["slice_sha256"]},
            }
        )

    external_paths = []
    for index, role in enumerate(E.EXPECTED_EXTERNAL_ARTIFACT_ROLES):
        name = f"external-{index}.bin"
        payload = f"{role}\n".encode()
        path = tmp_path / name
        path.write_bytes(payload)
        external_paths.append(path)

    document_specs = {
        "panel_manifest": {
            **_document_spec(
                {"prospective_pool_selection"},
                {"state_id", "role"},
                "states",
                E.EXPECTED_STATE_COUNT,
            ),
            "qualification_row": ["teacher_trace_index"],
            "qualification_container": (
                "prospective_pool_selection.qualification_rows"
            ),
            "qualification_count": E.EXPECTED_TEACHER_TRACE_COUNT,
            "qualification_identity_order": ["teacher_trace_index"],
        },
        "split_manifest": _document_spec(set(), {"state_id"}, "assignments", 64),
        "graph_manifest": _document_spec(set(), {"state_id"}, "graphs", 64),
        "state_snapshot_index": _document_spec(
            {"snapshots_file"},
            {
                "state_id", "snapshot_payload", "teacher_initial_state_sha256",
                "reset_trials", "reset_pair_comparison", *snapshot_index_fields,
            },
            "records",
            64,
        ),
        "teacher_trace_index": {
            **_document_spec(
                {"traces_file"},
                {
                    "state_id", "teacher_trace_index", "sample_count",
                    "trace_slice", "trace_array_slice_sha256s",
                },
                "records",
                E.EXPECTED_TEACHER_TRACE_COUNT,
            ),
            "identity_order": ["teacher_trace_index"],
        },
        "edge_port_index": _document_spec(set(), {"state_id"}, "records", 64),
        "waypoint_contracts": _document_spec(
            set(), {"state_id", "target_id"}, "rows", 192
        ),
        "pixel_index": _document_spec(
            {"rgb_file", "unique_pixel_count"},
            {"state_id", "rgb_row_index", "pixel_sha256", "row_sha256", "canonical_pixel_index"},
            "records",
            64,
        ),
        "latent_index": _document_spec(
            {"latents_file", "preprocessing_authority", "external_encoder_source"},
            {
                "canonical_pixel_index", "pixel_sha256", "raw_token_row_index",
                "raw_token_sha256", "spatial_descriptor_row_index",
                "spatial_descriptor_sha256", "preprocessed_tensor_sha256",
            },
            "records",
            "unique_pixel_count",
        ),
        "development_target_selection": {
            "root": sorted(
                {"state_target_rows", "target_summaries", "content_digest"}
            ),
            "state_target_row": ["state_id", "target_id"],
            "state_target_container": "state_target_rows",
            "state_target_count": 144,
            "summary_row": ["target_id"],
            "summary_container": "target_summaries",
            "summary_count": 3,
        },
    }
    ledgers = {
        "candidate_fanout": {
            "fields": sorted(candidate_rows[0]),
            "count": 768,
            "identity_order": ["state_id", "candidate_index"],
        },
        "heldout_ranker_scores": {
            "fields": ["condition_id", "state_id"],
            "count": 64,
            "identity_order": ["state_id", "condition_id"],
        },
        "repeated_execution": {
            "fields": sorted(repeat_rows[0]),
            "count": 64,
            "identity_order": ["state_id", "branch_selector_id", "repeat_index"],
        },
    }
    runtime_paths = {
        f"leaf_{index}": leaf for index, leaf in enumerate(E.ALL_OUTPUT_FILES)
    }
    physical_trace_authority = _attach(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v1."
                "physical_trace_reduction_authority.v1"
            ),
            "experiment_id": E.EXPERIMENT_ID,
            "physics_dt_s": 0.002,
            "dwell_samples": 100,
            "specs": [
                {"candidate_spec_id": f"pool-spec-{index:03d}"}
                for index in range(E.EXPECTED_TEACHER_TRACE_COUNT)
            ],
        }
    )
    evidence_keys = {
        "panel_manifest",
        "split_manifest",
        "graph_manifest",
        "state_snapshot_index",
        "teacher_trace_index",
        "edge_port_index",
        "waypoint_contracts",
        "pixel_index",
        "latent_index",
        "candidate_fanout",
        "development_target_selection",
        "heldout_ranker_scores",
        "repeated_execution",
        "npz_inspections",
    }
    authority = _attach(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.reducer_authority.v1",
            "experiment_id": E.EXPERIMENT_ID,
            "evidence_keys": sorted(evidence_keys),
            "documents": document_specs,
            "ledgers": ledgers,
            "npz_authorities": npz_authorities,
            "trace_index_ranges": {
                "reset_fixture": [0, E.EXPECTED_RESET_FIXTURE_TRACE_COUNT],
                "candidate_fanout": [
                    E.EXPECTED_RESET_FIXTURE_TRACE_COUNT,
                    E.EXPECTED_RESET_FIXTURE_TRACE_COUNT + E.EXPECTED_FANOUT_ROW_COUNT,
                ],
                "repeated_execution": [
                    E.EXPECTED_RESET_FIXTURE_TRACE_COUNT + E.EXPECTED_FANOUT_ROW_COUNT,
                    E.EXPECTED_CANDIDATE_TRACE_COUNT,
                ],
            },
            "reset_fixture_physics_samples": E.EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES,
            "branch_physics_samples": E.EXPECTED_PHYSICS_SAMPLES_PER_BRANCH,
            "reset_pair_comparison_fields": sorted(
                {
                    "state_id", "trial_indices", "physics_sample_count",
                    "exact_member_equal", "termination_reason_equal", "stuck_equal",
                    "passed",
                }
            ),
            "reset_trace_pair_comparison_authority": {
                "physics_samples_per_trial": E.EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES,
                "samplewise_tolerances": {},
                "exact_members": ["timestamp_s"],
            },
            "physical_trace_reduction_authority": physical_trace_authority,
            "runtime_environment_authority": {
                "digest_domain": "synthetic",
                "physical": {"fields": ["fake_runtime"]},
                "encoder": {"fields": ["fake_runtime"]},
                "ranker": {"fields": ["fake_runtime"]},
                "official_projections": {},
            },
            "runtime_paths": runtime_paths,
            "target_ids": target_ids,
            "heldout_condition_ids": conditions,
            "repeat_branch_ids": repeat_branches,
            "external_artifact_roles": list(E.EXPECTED_EXTERNAL_ARTIFACT_ROLES),
        }
    )

    frozen_external_bindings = []
    for role, path in zip(authority["external_artifact_roles"], external_paths):
        raw = path.read_bytes()
        frozen_external_bindings.append(
            {
                "role": role,
                "kind": "frozen_checkpoint_read_only",
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    def external_artifact_bindings(_contract):
        return copy.deepcopy(frozen_external_bindings)

    expected_npz_hashes = {
        leaf: hashlib.sha256((root / leaf).read_bytes()).hexdigest()
        for leaf in E.PAYLOAD_FILES
    }

    def recompute_metrics(evidence):
        assert len(evidence["candidate_fanout"]) == 768
        assert len(evidence["heldout_ranker_scores"]) == 64
        assert len(evidence["repeated_execution"]) == 64
        assert {
            row["path"]: row["sha256"] for row in evidence["npz_inspections"]
        } == expected_npz_hashes
        return _attach(
            {
                "schema": "synthetic.physical_handoff_metrics.v1",
                "experiment_id": E.EXPERIMENT_ID,
                "fanout_rows": 768,
                "heldout_rows": 64,
                "repeat_rows": 64,
            }
        )

    module = SimpleNamespace(
        __name__="synthetic_physical_handoff_metrics",
        reducer_authority=lambda: copy.deepcopy(authority),
        external_artifact_bindings=external_artifact_bindings,
        predecessor_context_binding=lambda contract: copy.deepcopy(
            contract["scientific_contract"]["v2_context_binding"]
        ),
        validate_npz_inspections=lambda rows: {
            row["path"]: copy.deepcopy(row) for row in rows
        },
        recompute_metrics=recompute_metrics,
    )
    metrics = recompute_metrics(
        {
            "candidate_fanout": candidate_rows,
            "heldout_ranker_scores": heldout_rows,
            "repeated_execution": repeat_rows,
            "npz_inspections": [
                {"path": leaf, "sha256": value}
                for leaf, value in expected_npz_hashes.items()
            ],
        }
    )
    documents["metrics.json"] = metrics
    for leaf, value in documents.items():
        _write_document(root / leaf, value)
    _write_jsonl(root / "candidate_fanout.jsonl", candidate_rows)
    _write_jsonl(root / "heldout_ranker_scores.jsonl", heldout_rows)
    _write_jsonl(root / "repeated_execution.jsonl", repeat_rows)
    return root, receipt, module, _source_observation(commit)


def test_exact_full_rebuild_emit_and_existing_receipt(tmp_path: Path) -> None:
    root, receipt_path, module, source = _fixture(tmp_path)
    receipt = E.verify_and_emit(
        root,
        receipt_path,
        metrics_module=module,
        source_freeze_observation=source,
    )
    assert receipt["pass"] is True
    assert receipt["metrics_exact_byte_equal"] is True
    assert set(receipt["scientific_execution_counters"].values()) == {0}
    supplied = json.loads(receipt_path.read_text())
    assert "content_digest" not in supplied
    assert "self_digest" not in supplied
    assert supplied == receipt
    # Presentation is deliberately downstream of scientific reduction.  The
    # ordinary external receipt must remain exactly rebuildable after all
    # three publication leaves appear.
    result_raw = E.canonical_document_bytes(_attach({"schema": "synthetic.result.v1"}))
    report_raw = b"synthetic report\n"
    (root / "result.json").write_bytes(result_raw)
    (root / "result.md").write_bytes(report_raw)
    bindings = {
        **receipt["inputs"],
        "result.json": E._binding("result.json", result_raw),
        "result.md": E._binding("result.md", report_raw),
    }
    rows = [bindings[name] for name in sorted(bindings)]
    _write_document(
        root / "file_hashes.json",
        _attach(
            {
                "schema": "physical_graph_edge_handoff_qualification_v1.file_hashes.v1",
                "root": str(root),
                "files": rows,
                "file_count_excluding_self": len(rows),
                "bytes_excluding_self": sum(row["bytes"] for row in rows),
                "file_hashes_self_sha256_excluded": True,
            }
        ),
    )
    assert E.validate_existing_regeneration_receipt(
        root,
        receipt_path,
        metrics_module=module,
        source_freeze_observation=source,
    ) == receipt


def test_rejects_ledger_identity_and_metrics_drift(tmp_path: Path) -> None:
    root, _receipt_path, module, source = _fixture(tmp_path)
    rows = [json.loads(line) for line in (root / "candidate_fanout.jsonl").read_text().splitlines()]
    rows[1] = dict(rows[0])
    _write_jsonl(root / "candidate_fanout.jsonl", rows)
    with pytest.raises(E.RegenerationError, match="duplicate row identities"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )

    root, _receipt_path, module, source = _fixture(tmp_path / "second")
    metrics = json.loads((root / "metrics.json").read_text())
    metrics["fanout_rows"] = 767
    metrics.pop("content_digest")
    _write_document(root / "metrics.json", _attach(metrics))
    with pytest.raises(E.RegenerationError, match="persisted metrics differ"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )


def test_rejects_npz_and_checkpoint_byte_tamper(tmp_path: Path) -> None:
    root, _receipt_path, module, source = _fixture(tmp_path)
    np.savez_compressed(root / "canonical_latents.npz", x=np.arange(64, dtype=np.uint8) + 9)
    with pytest.raises(E.RegenerationError, match="NPZ member inventory drift"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )

    root, _receipt_path, module, source = _fixture(tmp_path / "second")
    checkpoint = Path(module.external_artifact_bindings({})[0]["path"])
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(E.RegenerationError, match="external artifact binding drift"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )

    root, _receipt_path, module, source = _fixture(tmp_path / "third")
    contract = json.loads((root / "contract.json").read_text())
    predecessor = Path(contract["predecessor_result_binding"]["path"])
    (predecessor.parent / "stage_b_trace.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(E.RegenerationError, match="external artifact binding drift"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )


def test_rejects_index_or_trace_hash_not_backed_by_npz(tmp_path: Path) -> None:
    root, _receipt_path, module, source = _fixture(tmp_path)
    pixel = json.loads((root / "pixel_index.json").read_text())
    pixel.pop("content_digest")
    pixel["records"][0]["row_sha256"] = "0" * 64
    _write_document(root / "pixel_index.json", _attach(pixel))
    with pytest.raises(E.RegenerationError, match="differs from RGB NPZ bytes"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )

    root, _receipt_path, module, source = _fixture(tmp_path / "second")
    rows = [
        json.loads(line)
        for line in (root / "candidate_fanout.jsonl").read_text().splitlines()
    ]
    rows[0]["trace_array_slice_sha256s"]["timestamp_s"] = "0" * 64
    _write_jsonl(root / "candidate_fanout.jsonl", rows)
    with pytest.raises(E.RegenerationError, match="trace-member hashes differ"):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )

    root, _receipt_path, module, source = _fixture(tmp_path / "third")
    snapshots = json.loads((root / "state_snapshot_index.json").read_text())
    snapshots.pop("content_digest")
    snapshots["records"][0]["base_pose_world_sha256"] = "0" * 64
    _write_document(root / "state_snapshot_index.json", _attach(snapshots))
    with pytest.raises(
        E.RegenerationError,
        match="base_pose_world_sha256 differs from snapshot NPZ bytes",
    ):
        E.build_regeneration_receipt(
            root, metrics_module=module, source_freeze_observation=source
        )


def test_system_python_help_with_stripped_pythonpath() -> None:
    script = E.REPO_ROOT / E.REDUCER_SOURCE_PATH
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        ["/usr/bin/python3", str(script), "--help"],
        cwd=E.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--validate-existing" in completed.stdout
    probe = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(E.REPO_ROOT)!r});"
                "import scripts.evaluate_physical_graph_edge_handoff_qualification_v1;"
                "print(int('torch' in sys.modules))"
            ),
        ],
        cwd=E.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "0"


def test_recomputed_metric_tuple_round_trip_is_canonical_and_finite() -> None:
    raw, value = E._canonicalise_object(
        {"claims": {"does_not_establish": ("SAFETY", "DEPLOYMENT")}},
        "tuple fixture",
    )
    assert value["claims"]["does_not_establish"] == ["SAFETY", "DEPLOYMENT"]
    assert raw == E.canonical_document_bytes(value)
    with pytest.raises(E.RegenerationError, match="non-finite"):
        E._canonicalise_object({"value": float("nan")}, "nonfinite fixture")


def test_npz_shared_symbols_and_offsets_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    np.savez(
        root / "symbolic.npz",
        first=np.zeros((2,), dtype=np.uint8),
        second=np.zeros((3,), dtype=np.uint8),
    )
    authority = {
        name: {
            "descr": "|u1",
            "digest_dtype": "uint8",
            "shape": ["U"],
            "hash_mode": "rows_axis0",
        }
        for name in ("first", "second")
    }
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(E.RegenerationError, match="symbolic dimension drift"):
            E._inspect_npz_at(root_fd, "symbolic.npz", authority, symbols={})
    finally:
        os.close(root_fd)


def test_reset_pair_is_recomputed_from_raw_trace_bytes(tmp_path: Path) -> None:
    root, _receipt_path, module, _source = _fixture(tmp_path)
    authority = module.reducer_authority()
    captured: dict[str, list[bytes]] = {}
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        inspection = E._inspect_npz_at(
            root_fd,
            "candidate_traces.npz",
            authority["npz_authorities"]["candidate_traces.npz"],
            symbols={},
            captured_reset_slices=captured,
        )
    finally:
        os.close(root_fd)
    snapshots = json.loads((root / "state_snapshot_index.json").read_text())
    result = E._validate_reset_trace_pairs(
        snapshots, inspection, captured, authority
    )
    assert result["all_pairs_passed"] is True
    mismatched_restore = copy.deepcopy(snapshots)
    mismatched_restore["records"][0]["reset_trials"][1][
        "post_restore_state_sha256"
    ] = "0" * 64
    with pytest.raises(
        E.RegenerationError,
        match="differ immediately after serialized restore",
    ):
        E._validate_reset_trace_pairs(
            mismatched_restore, inspection, captured, authority
        )
    second = bytearray(captured["timestamp_s"][1])
    second[0] ^= 1
    captured["timestamp_s"][1] = bytes(second)
    with pytest.raises(E.RegenerationError, match="differs from raw NPZ traces"):
        E._validate_reset_trace_pairs(snapshots, inspection, captured, authority)

    np.savez(
        root / "offsets.npz",
        offsets=np.array([0, 2, 2], dtype=np.int64),
        payload=np.arange(2, dtype=np.uint8),
    )
    offset_authority = {
        "offsets": {
            "descr": "<i8",
            "digest_dtype": "int64",
            "shape": [3],
            "hash_mode": "whole",
        },
        "payload": {
            "descr": "|u1",
            "digest_dtype": "uint8",
            "shape": ["P"],
            "hash_mode": "offset_slices",
            "offsets_member": "offsets",
        },
    }
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(E.RegenerationError, match="offsets are not monotone"):
            E._inspect_npz_at(root_fd, "offsets.npz", offset_authority, symbols={})
    finally:
        os.close(root_fd)


def test_reset_stuck_termination_and_terminal_rows_are_raw_reduced() -> None:
    authority = E._validate_authority(M)
    count = E.EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES
    timestamp = np.arange(count, dtype=np.float64) * 0.002
    pose = np.zeros((count, 7), dtype=np.float64)
    pose[:, 0] = np.linspace(0.0, 0.5, count)
    pose[:, 2] = 0.3
    pose[:, 6] = 1.0
    twist = np.zeros((count, 6), dtype=np.float64)
    joint = np.zeros((count, 12), dtype=np.float64)
    requested = np.zeros((count, 3), dtype=np.float64)
    requested[:, 0] = 0.1
    applied = requested.copy()
    binary = np.zeros(count, dtype=np.uint8)
    source = np.ones(count, dtype=np.uint8)
    arrays = {
        "timestamp_s": timestamp,
        "base_pose_world": pose,
        "base_twist_world": twist,
        "joint_position": joint,
        "joint_velocity": joint,
        "requested_command": requested,
        "post_slew_applied_command": applied,
        "physics_contact": binary,
        "source_region_member": source,
        "correct_edge_region_member": binary,
        "wrong_edge_region_member": binary,
        "target_region_member": binary,
    }
    payloads = {
        name: np.ascontiguousarray(value).tobytes(order="C")
        for name, value in arrays.items()
    }
    captured = {
        name: [payload] * E.EXPECTED_RESET_FIXTURE_TRACE_COUNT
        for name, payload in payloads.items()
    }
    members = {
        name: {
            "descr": value.dtype.str,
            "shape": [
                E.EXPECTED_CANDIDATE_TRACE_COUNT * count,
                *value.shape[1:],
            ],
            "row_or_slice_sha256s": [
                E._array_row_digest(payloads[name], value.shape, value.dtype.str)
            ]
            * E.EXPECTED_CANDIDATE_TRACE_COUNT,
        }
        for name, value in arrays.items()
    }
    inspection = {"members": members}
    comparison = {
        "state_id": "",
        "trial_indices": [0, 1],
        "physics_sample_count": count,
        "maximum_base_position_error_m": 0.0,
        "maximum_base_quaternion_component_error": 0.0,
        "maximum_base_twist_error": 0.0,
        "maximum_joint_position_error_rad": 0.0,
        "maximum_joint_velocity_error_rad_s": 0.0,
        "exact_member_equal": {
            member: True
            for member in authority["reset_trace_pair_comparison_authority"][
                "exact_members"
            ]
        },
        "termination_reason_equal": True,
        "stuck_equal": True,
        "endpoint_position_error_m": 0.0,
        "endpoint_heading_error_rad": 0.0,
        "passed": True,
    }
    terminal_joint_sha = E._array_row_digest(
        payloads["joint_position"][-12 * 8 :], (12,), "<f8"
    )
    records = []
    for state_index in range(E.EXPECTED_STATE_COUNT):
        state_id = f"state-{state_index:03d}"
        trials = []
        for trial_index in range(2):
            trace_index = state_index * 2 + trial_index
            trials.append(
                {
                    "termination_reason": "H3_COMPLETE",
                    "stuck": False,
                    "base_pose_world": list(pose[-1]),
                    "requested_command_sequence_sha256": members[
                        "requested_command"
                    ]["row_or_slice_sha256s"][trace_index],
                    "post_slew_applied_command_sequence_sha256": members[
                        "post_slew_applied_command"
                    ]["row_or_slice_sha256s"][trace_index],
                    "contact_sequence_sha256": members["physics_contact"][
                        "row_or_slice_sha256s"
                    ][trace_index],
                    "joint_position_sha256": terminal_joint_sha,
                    "joint_velocity_sha256": terminal_joint_sha,
                }
            )
        supplied = copy.deepcopy(comparison)
        supplied["state_id"] = state_id
        records.append(
            {
                "state_id": state_id,
                "reset_trials": trials,
                "reset_pair_comparison": supplied,
            }
        )
    result = E._validate_reset_trace_pairs(
        {"records": records}, inspection, captured, authority
    )
    assert result["all_pairs_passed"] is True

    records[0]["reset_trials"][0]["stuck"] = True
    with pytest.raises(E.RegenerationError, match="stuck metadata differs"):
        E._validate_reset_trace_pairs(
            {"records": records}, inspection, captured, authority
        )


def test_raw_candidate_physics_is_derived_from_trace_payloads() -> None:
    count = E.EXPECTED_PHYSICS_SAMPLES_PER_BRANCH
    timestamp = np.arange(count, dtype=np.float64) * 0.002
    pose = np.zeros((count, 7), dtype=np.float64)
    pose[:, 0] = np.linspace(-1.0, 1.0, count)
    pose[:, 2] = 0.30
    pose[:, 6] = 1.0
    twist = np.zeros((count, 6), dtype=np.float64)
    twist[:, 0] = (2.0 / (count - 1)) / 0.002
    requested = np.zeros((count, 3), dtype=np.float64)
    requested[:, 0] = 0.10
    applied = requested.copy()
    contact = np.zeros(count, dtype=np.uint8)
    source = (pose[:, 0] < 0.0).astype(np.uint8)
    selected = (pose[:, 0] >= 0.0).astype(np.uint8)
    wrong = np.zeros(count, dtype=np.uint8)
    target = np.zeros(count, dtype=np.uint8)
    arrays = {
        "timestamp_s": timestamp,
        "base_pose_world": pose,
        "base_twist_world": twist,
        "requested_command": requested,
        "post_slew_applied_command": applied,
        "physics_contact": contact,
        "source_region_member": source,
        "correct_edge_region_member": selected,
        "wrong_edge_region_member": wrong,
        "target_region_member": target,
    }
    captured = {
        "candidate_traces.npz": {
            name: [np.ascontiguousarray(value).tobytes(order="C")]
            for name, value in arrays.items()
        }
    }
    inspections = {
        "candidate_traces.npz": {
            "members": {
                name: {
                    "descr": value.dtype.str,
                    "shape": list(value.shape),
                }
                for name, value in arrays.items()
            }
        }
    }
    physical = {
        "physics_dt_s": 0.002,
        "dwell_samples": 100,
        "candidate_member_mapping": {
            "timestamp": "timestamp_s",
            "base_pose": "base_pose_world",
            "base_twist": "base_twist_world",
            "requested_command": "requested_command",
            "post_slew_command": "post_slew_applied_command",
            "contact": "physics_contact",
            "source_membership": "source_region_member",
            "selected_edge_membership": "correct_edge_region_member",
            "competing_edge_membership": "wrong_edge_region_member",
            "target_membership": "target_region_member",
        },
        "endpoint_indices": {"H1": 249, "H2": 499, "H3": 749},
        "stuck": {
            "command_activity_threshold": 0.05,
            "h3_translation_threshold_m": 0.02,
            "h3_heading_threshold_rad": 0.05,
        },
        "successor_viable": {
            "minimum_base_height_m": 0.20,
            "maximum_absolute_roll_rad": 0.70,
            "maximum_absolute_pitch_rad": 0.70,
        },
        "command_tracking_authority": {
            "command_ticks": 15,
            "physics_samples_per_command_tick": 50,
            "discard_initial_physics_samples_per_command_tick": 20,
            "averaged_physics_samples_per_command_tick": 30,
            "active_command_threshold": 0.05,
        },
    }
    geometry = {
        "source_boundary_polygon_world": [
            [-2.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-2.0, 1.0]
        ],
        "target_boundary_polygon_world": [
            [2.0, -1.0], [3.0, -1.0], [3.0, 1.0], [2.0, 1.0]
        ],
        "selected_edge": {
            "edge_id": "selected-edge",
            "opening_segment_world": [[0.0, -1.0], [0.0, 1.0]],
            "opening_normal_world": [1.0, 0.0],
        },
        "competing_edges": [],
    }
    raw = E._raw_candidate_projection(
        trace_index=0,
        state_index=0,
        geometry=geometry,
        initial_pose=pose[0],
        port={"directed_port_world": [0.0, 0.0, 0.0]},
        inspections=inspections,
        captured=captured,
        physical=physical,
    )
    assert raw["entered_correct_edge"] is True
    assert raw["entered_wrong_edge"] is False
    assert raw["physics_contact"] is False
    assert raw["successor_viable"] is True
    assert raw["oracle_admissible"] is True
    assert raw["left_source_region"] is True
    assert raw["first_source_exit_sample_index"] is not None
    assert raw["reached_target_node"] is False
    assert raw["first_target_entry_sample_index"] is None
    assert raw["selected_crossing"]["displacement_world_xy"][0] > 0.0
    assert raw["selected_crossing"]["direction_heading_world_rad"] == pytest.approx(0.0)
    expected_progress = 1.0 - min(abs(float(value)) for value in pose[:, 0])
    assert raw["port_progress_m"] == pytest.approx(expected_progress, abs=1e-12)

    tied_geometry = copy.deepcopy(geometry)
    tied_geometry["competing_edges"] = [
        {
            "edge_id": "competing-edge-tie",
            "opening_segment_world": geometry["selected_edge"][
                "opening_segment_world"
            ],
            "opening_normal_world": geometry["selected_edge"][
                "opening_normal_world"
            ],
        }
    ]
    tied_captured = copy.deepcopy(captured)
    tied_captured["candidate_traces.npz"]["wrong_edge_region_member"][0] = (
        tied_captured["candidate_traces.npz"]["correct_edge_region_member"][0]
    )
    tied = E._raw_candidate_projection(
        trace_index=0,
        state_index=0,
        geometry=tied_geometry,
        initial_pose=pose[0],
        port={"directed_port_world": [0.0, 0.0, 0.0]},
        inspections=inspections,
        captured=tied_captured,
        physical=physical,
    )
    assert tied["entered_correct_edge"] is False
    assert tied["entered_wrong_edge"] is True

    tampered = copy.deepcopy(captured)
    payload = bytearray(tampered["candidate_traces.npz"]["physics_contact"][0])
    payload[100] = 1
    tampered["candidate_traces.npz"]["physics_contact"][0] = bytes(payload)
    adverse = E._raw_candidate_projection(
        trace_index=0,
        state_index=0,
        geometry=geometry,
        initial_pose=pose[0],
        port={"directed_port_world": [0.0, 0.0, 0.0]},
        inspections=inspections,
        captured=tampered,
        physical=physical,
    )
    assert adverse["physics_contact"] is True
    assert adverse["oracle_admissible"] is False


def test_raw_teacher_and_port_evidence_is_derived_from_trace_payloads() -> None:
    sample_count = 200
    timestamp = np.arange(sample_count, dtype=np.float64) * 0.002
    pose = np.zeros((sample_count, 7), dtype=np.float64)
    pose[:, 0] = np.linspace(-0.5, 1.5, sample_count)
    pose[:, 2] = 0.30
    pose[:, 6] = 1.0
    twist = np.zeros((sample_count, 6), dtype=np.float64)
    twist[:, 0] = (2.0 / (sample_count - 1)) / 0.002
    requested = np.zeros((sample_count, 3), dtype=np.float64)
    requested[:, 0] = 0.10
    contact = np.zeros(sample_count, dtype=np.uint8)
    source = (pose[:, 0] < 0.0).astype(np.uint8)
    edge = (pose[:, 0] >= 0.0).astype(np.uint8)
    target = np.zeros(sample_count, dtype=np.uint8)
    arrays = {
        "timestamp_s": timestamp,
        "base_pose_world": pose,
        "base_twist_world": twist,
        "requested_command": requested,
        "applied_command": requested.copy(),
        "physics_contact": contact,
        "source_region_member": source,
        "edge_region_member": edge,
        "target_region_member": target,
    }
    captured = {
        "teacher_traces.npz": {
            name: [np.ascontiguousarray(value).tobytes(order="C")]
            * E.EXPECTED_TEACHER_TRACE_COUNT
            for name, value in arrays.items()
        }
    }
    inspections = {
        "teacher_traces.npz": {
            "members": {
                name: {"descr": value.dtype.str, "shape": list(value.shape)}
                for name, value in arrays.items()
            }
        }
    }
    opening = [[0.0, -1.0], [0.0, 1.0]]
    polygon = [[-1.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 1.0]]
    selected_edge = {
        "edge_id": "selected-edge",
        "opening_segment_world": opening,
        "opening_normal_world": [1.0, 0.0],
    }
    crossing = E._first_transverse_crossing(
        pose, opening, [1.0, 0.0], "teacher fixture", 1.0e-6
    )
    assert crossing is not None
    dwell = E._dwell_projection(pose, target, crossing, 100, 1.0e-6)
    route_progress = 0.5 - min(abs(float(row[0])) for row in pose)
    velocity = [float(twist[0, 0]), 0.0]
    first_source_exit = next(index for index, value in enumerate(source) if value == 0)
    specs = []
    rows = []
    ports = []
    for index in range(E.EXPECTED_TEACHER_TRACE_COUNT):
        state_id = f"teacher-state-{index:03d}"
        candidate_spec_id = f"teacher-spec-{index:03d}"
        specs.append(
            {
                "candidate_spec_id": candidate_spec_id,
                "state_id": state_id,
                "pool_order_index": index,
                "canonical_spec_sha256": f"{index + 1:064x}"[-64:],
                "source_boundary_polygon_world": polygon,
                "target_boundary_polygon_world": [[2.0, -1.0], [3.0, -1.0], [3.0, 1.0], [2.0, 1.0]],
                "selected_edge": selected_edge,
                "competing_edges": [],
                "teacher_route_polyline_world": [[-0.5, 0.0], [0.0, 0.0], [1.5, 0.0]],
                "spawn_se2_world": [-0.5, 0.0, 0.0],
            }
        )
        selected = index < E.EXPECTED_STATE_COUNT
        rows.append(
            {
                "candidate_spec_id": candidate_spec_id,
                "state_id": state_id,
                "canonical_spec_sha256": f"{index + 1:064x}"[-64:],
                "sample_count": sample_count,
                "contact_free": True,
                "left_source_region": True,
                "first_source_exit_sample_index": first_source_exit,
                "competing_port_entered": False,
                "crossed_directed_port": True,
                "route_progress_m": route_progress,
                "positive_route_progress": True,
                "reached_target_node": False,
                "first_target_entry_sample_index": None,
                "where_reached": "BEYOND_DIRECTED_PORT",
                "endpoint_lateral_error_m": 0.0,
                "endpoint_angular_error_rad": 0.0,
                "successor_viable": True,
                "stuck": False,
                "first_crossing_sample_index": crossing["sample_after"],
                "crossing_segment_fraction": crossing["fraction"],
                "crossing_directed_normal_dot": crossing["normal_dot_displacement_m"],
                "crossing_lateral_fraction": crossing["lateral_fraction"],
                "crossing_velocity_world_xy": velocity,
                "crossing_velocity_heading_world_rad": 0.0,
                "beyond_port_consecutive_physics_samples": dwell[
                    "beyond_port_consecutive_physics_samples"
                ],
                "target_entered_before_dwell_complete": False,
                "selected": selected,
            }
        )
        if selected:
            ports.append(
                {
                    "state_id": state_id,
                    "crossing_sample_before": crossing["sample_before"],
                    "crossing_sample_after": crossing["sample_after"],
                    "crossing_fraction": crossing["fraction"],
                    "directed_port_world": [0.0, 0.0, 0.0],
                    "teacher_crossing_velocity_world_xy": velocity,
                    "teacher_crossing_velocity_heading_world_rad": 0.0,
                }
            )
    physical = {
        "physics_dt_s": 0.002,
        "dwell_samples": 100,
        "port_crossing_tolerance_m": 1.0e-6,
        "crossing_velocity": (
            "linearly interpolate base_twist_world vx_world,vy_world at crossing alpha; "
            "heading=atan2(vy_world,vx_world), requiring nonzero planar speed"
        ),
        "teacher_member_mapping": {
            "timestamp": "timestamp_s",
            "base_pose": "base_pose_world",
            "base_twist": "base_twist_world",
            "contact": "physics_contact",
            "requested_command": "requested_command",
            "applied_command": "applied_command",
            "source_membership": "source_region_member",
            "selected_edge_membership": "edge_region_member",
            "target_membership": "target_region_member",
        },
        "successor_viable": {
            "minimum_base_height_m": 0.20,
            "maximum_absolute_roll_rad": 0.70,
            "maximum_absolute_pitch_rad": 0.70,
        },
        "stuck": {
            "command_activity_threshold": 0.05,
            "h3_translation_threshold_m": 0.02,
            "h3_heading_threshold_rad": 0.05,
        },
        "specs": specs,
    }
    documents = {
        "teacher_trace_index": {"records": rows},
        "edge_port_index": {"records": ports},
    }
    validation = E._validate_raw_teacher_evidence(
        documents, inspections, captured, physical
    )
    assert validation["teacher_trace_count"] == E.EXPECTED_TEACHER_TRACE_COUNT
    assert validation["selected_teacher_trace_count"] == E.EXPECTED_STATE_COUNT

    tampered = copy.deepcopy(captured)
    bad_twist = twist.copy()
    bad_twist[crossing["sample_after"], 1] = 1.0
    tampered["teacher_traces.npz"]["base_twist_world"][0] = (
        np.ascontiguousarray(bad_twist).tobytes(order="C")
    )
    with pytest.raises(E.RegenerationError, match="crossing velocity"):
        E._validate_raw_teacher_evidence(documents, inspections, tampered, physical)

    tampered = copy.deepcopy(captured)
    bad_source = bytearray(tampered["teacher_traces.npz"]["source_region_member"][0])
    bad_source[0] ^= 1
    tampered["teacher_traces.npz"]["source_region_member"][0] = bytes(bad_source)
    with pytest.raises(E.RegenerationError, match="membership differs"):
        E._validate_raw_teacher_evidence(documents, inspections, tampered, physical)


def test_external_encoder_checkout_and_preprocessing_hash_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "encoder-source"
    repository.mkdir()
    commit = "e" * 40
    preprocessing = {
        "input_shape": [168, 224, 3],
        "input_dtype": "uint8",
        "input_layout": "C_CONTIGUOUS_RGB",
        "transport": "synthetic exact temporary-PNG transport description",
        "temporary_png_retained": False,
        "output_type": "torch.Tensor",
        "output_shape": [3, 384, 512],
        "output_dtype": "float32",
        "finite_required": True,
        "output_projection": "detach, CPU, C-contiguous",
        "preprocessed_tensor_sha256_persisted_per_canonical_pixel": True,
    }
    latent = {
        "external_encoder_source": {
            "repository_path": str(repository),
            "commit": commit,
            "worktree_clean": True,
        },
        "preprocessing_authority": preprocessing,
        "records": [
            {"preprocessed_tensor_sha256": hashlib.sha256(b"tensor").hexdigest()}
        ],
    }

    def fake_check_output(arguments, **_kwargs):
        assert arguments[:3] == ["git", "-C", str(repository)]
        if arguments[3:] == ["rev-parse", "HEAD"]:
            return (commit + "\n").encode()
        if arguments[3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(E.subprocess, "check_output", fake_check_output)
    observed = E._validate_external_encoder_source(latent)
    assert observed["commit"] == commit
    assert observed["preprocessed_tensor_hash_count"] == 1
    assert observed["model_or_encoder_opened"] is False

    tampered = copy.deepcopy(latent)
    tampered["records"][0]["preprocessed_tensor_sha256"] = "not-a-digest"
    with pytest.raises(E.RegenerationError, match="preprocessed tensor SHA"):
        E._validate_external_encoder_source(tampered)


def test_runtime_environment_row_bindings_are_independently_joined() -> None:
    physical_digest = hashlib.sha256(b"physical-runtime").hexdigest()
    physical = {
        "fake_runtime": False,
        "runtime_core_sha256": physical_digest,
        "qualification_runtime_sha256s": [physical_digest] * 256,
        "selected_snapshot_runtime_sha256s": [physical_digest] * 64,
    }
    encoder = {"fake_runtime": False, "role": "encoder"}
    ranker = {"fake_runtime": False, "role": "ranker"}
    ranker_digest = E.canonical_digest(ranker)
    module = SimpleNamespace(
        validate_physical_runtime_environment=lambda value: value,
        validate_visual_runtime_environment=(
            lambda value, *, runtime_role: value
        ),
        runtime_environment_sha256=E.canonical_digest,
    )
    documents = {
        "panel_manifest": {"physical_runtime_environment": physical},
        "latent_index": {"encoder_runtime_environment": encoder},
        "development_target_selection": {
            "ranker_runtime_environment": ranker
        },
    }
    ledgers = {
        "candidate_fanout": [
            {"physical_runtime_core_sha256": physical_digest}
        ],
        "repeated_execution": [
            {"physical_runtime_core_sha256": physical_digest}
        ],
        "heldout_ranker_scores": [
            {"ranker_runtime_environment_sha256": ranker_digest}
        ],
    }
    observed = E._validate_runtime_environments(module, documents, ledgers)
    assert observed["fanout_and_repeat_runtime_binding_count"] == 2
    assert observed["heldout_ranker_runtime_binding_count"] == 1
    assert observed["all_observed_runtimes_real_and_exact"] is True

    tampered = copy.deepcopy(ledgers)
    tampered["candidate_fanout"][0]["physical_runtime_core_sha256"] = "0" * 64
    with pytest.raises(E.RegenerationError, match="observed physical runtime"):
        E._validate_runtime_environments(module, documents, tampered)

    fake_documents = copy.deepcopy(documents)
    fake_documents["latent_index"]["encoder_runtime_environment"][
        "fake_runtime"
    ] = True
    with pytest.raises(E.RegenerationError, match="marked fake"):
        E._validate_runtime_environments(module, fake_documents, ledgers)


def test_reducer_authority_schema_is_fail_closed() -> None:
    authority = copy.deepcopy(M.reducer_authority())
    authority.pop("content_digest")
    authority["schema"] = "physical_graph_edge_handoff_qualification_v1.stale.v0"
    authority["content_digest"] = E.canonical_digest(authority)
    module = SimpleNamespace(reducer_authority=lambda: authority)
    with pytest.raises(E.RegenerationError, match="authority identity drift"):
        E._validate_authority(module)
