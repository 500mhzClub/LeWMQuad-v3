#!/usr/bin/env python3
"""Independent custody and metric reduction for the corrected V2 run.

This module is intentionally usable with system Python.  It never imports a
model, torch, Genesis, an encoder, a ranker, or an experiment runner.  The
failed V1 material is opened only for byte custody and, after V2 has produced
fresh shards, for the registered first-eight reproduction comparison.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any, BinaryIO
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Both imports are frozen, system-Python-safe source dependencies.  They do
# not open either experiment root and contain no model/runtime imports.
from scripts import evaluate_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import evaluate_occluded_goal_topological_belief_v2 as CUSTODY


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
V1_EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
SOURCE_PARENT_COMMIT = "3dfff6caec1c3162d8123c737d04bcdd42799653"
METRICS_MODULE = "lewm.safety.physical_graph_edge_handoff_qualification_v2_metrics"

DEFAULT_V1_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1"
)
DEFAULT_V1_MATERIAL_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1_material"
)
DEFAULT_V1_CUSTODY_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1_custody_receipt.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2"
)
DEFAULT_EXTERNAL_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
)

V1_CUSTODY_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v1.custody_receipt.v1"
)
REGENERATION_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v2.regeneration_receipt.v1"
)
V1_OFFICIAL_CONTRACT_SHA256 = (
    "1ceac0e8ad8af860466537f2b462fe8105cb328daea86f9e0e2b7ed5552a6528"
)
V1_OFFICIAL_CONTRACT_BYTES = 66_418
V1_MATERIAL_FILE_COUNT = 34
V1_MATERIAL_APPARENT_BYTES = 4_299_695
V1_PAIR_COUNT = 8
V1_QUALIFIED_COUNT = 7
V1_REJECTED_POOL_INDEX = 2
V1_REJECTION_REASON = "TEACHER_PHYSICS_CONTACT"
V2_NPZ_ARCHIVE_COMMENT = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2:FRESH"
V1_TEACHER_SAMPLE_COUNTS = (500, 400, 4000, 650, 500, 450, 450, 500)
V1_INVALID_PREVIOUS_COMMAND_SHA256 = (
    "cd9cdaf845ca5fd82a869bd1f17362d780cfd1bd3a46fe76cd93575fb5ece5c6"
)
V1_PERSISTED_PREVIOUS_COMMAND_CANONICAL_SHA256 = (
    "b237b7cbc24dd1d0bc36a02fa9a0d58e12f511b0fc76d2bbd0aaa48cab0265e3"
)
V2_PERSISTED_PREVIOUS_COMMAND_RAW_SHA256 = (
    "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
)

V2_ADDITIONAL_FILES = (
    "v1_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_first_eight_reproduction.json",
)
V2_SUCCESS_FILES = (
    "contract.json",
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "state_snapshot_index.json",
    "state_snapshots.npz",
    "teacher_trace_index.json",
    "teacher_traces.npz",
    "edge_port_index.json",
    "waypoint_contracts.json",
    "pixel_index.json",
    "rgb_observations.npz",
    "latent_index.json",
    "canonical_latents.npz",
    "candidate_traces.npz",
    "candidate_fanout.jsonl",
    "development_target_selection.json",
    "heldout_ranker_scores.jsonl",
    "repeated_execution.jsonl",
    "metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
    *V2_ADDITIONAL_FILES,
)
V2_REPRODUCTION_FAILURE_FILES = (
    "contract.json",
    *V2_ADDITIONAL_FILES,
)

_SNAPSHOT_HASH_MEMBERS = {
    "controller_observation_sha256": "snapshot__controller_observation.npy",
    "policy_last_action_sha256": "snapshot__policy_last_action.npy",
    "previous_policy_action_sha256": "snapshot__previous_policy_action.npy",
    "previous_applied_command_sha256": "snapshot__previous_applied_command.npy",
    "command_history_sha256": "snapshot__command_history.npy",
    "control_history_sha256": "snapshot__control_history.npy",
    "low_level_policy_state_sha256": "snapshot__low_level_policy_state.npy",
}
_TEACHER_MEMBERS = (
    "timestamp_s",
    "base_pose_world",
    "base_twist_world",
    "joint_position",
    "joint_velocity",
    "applied_command",
    "requested_command",
    "physics_contact",
    "source_region_member",
    "edge_region_member",
    "target_region_member",
)
_EXPECTED_V1_SHARD_MEMBERS = {
    "rgb.npy",
    "snapshot_payload_bytes.npy",
    "snapshot__base_pose_world.npy",
    "snapshot__base_twist_world.npy",
    "snapshot__joint_position.npy",
    "snapshot__joint_velocity.npy",
    "snapshot__controller_observation.npy",
    "snapshot__policy_last_action.npy",
    "snapshot__previous_policy_action.npy",
    "snapshot__previous_applied_command.npy",
    "snapshot__command_history.npy",
    "snapshot__control_history.npy",
    "snapshot__low_level_policy_state.npy",
    "snapshot__camera_world_transform.npy",
    *(f"teacher__{name}.npy" for name in _TEACHER_MEMBERS),
}


RegenerationError = CUSTODY.RegenerationError
canonical_document_bytes = CUSTODY.canonical_document_bytes
canonical_json_bytes = CUSTODY.canonical_json_bytes
parse_canonical_json = CUSTODY.parse_canonical_json
_npy_header = CUSTODY._npy_header
_read_exact = CUSTODY._read_exact


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RegenerationError(f"custody input is not a single-link regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    projection = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if projection(before) != projection(after):
        raise RegenerationError(f"custody input changed while hashing: {path}")
    return digest.hexdigest()


def _root_path(path: Path | str, label: str) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.is_symlink():
        raise RegenerationError(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise RegenerationError(f"{label} is absent: {root}") from exc
    if resolved != root or not root.is_dir():
        raise RegenerationError(f"{label} traverses a symlink or is not a directory")
    return root


def _root_inventory(path: Path | str, label: str) -> dict[str, Any]:
    root = _root_path(path, label)
    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_directory = "." if current_path == root else current_path.relative_to(root).as_posix()
        current_stat = current_path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(current_stat.st_mode) or current_path.is_symlink():
            raise RegenerationError(f"{label} contains a non-directory traversal entry")
        directories.append(
            {
                "path": relative_directory,
                "allocated_bytes": int(current_stat.st_blocks * 512),
                "device": int(current_stat.st_dev),
                "inode": int(current_stat.st_ino),
                "nlink": int(current_stat.st_nlink),
            }
        )
        for name in sorted(directory_names):
            candidate = current_path / name
            info = candidate.stat(follow_symlinks=False)
            if candidate.is_symlink() or not stat.S_ISDIR(info.st_mode):
                raise RegenerationError(f"{label} contains a non-directory or symlink: {candidate}")
        for name in sorted(file_names):
            candidate = current_path / name
            info = candidate.stat(follow_symlinks=False)
            if candidate.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RegenerationError(f"{label} contains a non-single-link file: {candidate}")
            files.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "bytes": int(info.st_size),
                    "allocated_bytes": int(info.st_blocks * 512),
                    "sha256": _sha256_file(candidate),
                    "device": int(info.st_dev),
                    "inode": int(info.st_ino),
                    "nlink": int(info.st_nlink),
                }
            )
    files.sort(key=lambda row: row["path"])
    directories.sort(key=lambda row: row["path"])
    regular_allocated = sum(row["allocated_bytes"] for row in files)
    directory_allocated = sum(row["allocated_bytes"] for row in directories)
    return {
        "path": str(root),
        "file_count": len(files),
        "directory_count": len(directories),
        "regular_file_apparent_bytes": sum(row["bytes"] for row in files),
        "regular_file_allocated_bytes": regular_allocated,
        "directory_allocated_bytes": directory_allocated,
        "allocated_bytes": regular_allocated + directory_allocated,
        "files": files,
        "directories": directories,
    }


def _load_canonical_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RegenerationError(f"{label} is not a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RegenerationError(f"{label} changed while reading")
    raw = b"".join(blocks)
    value = parse_canonical_json(raw, label=label)
    if not isinstance(value, dict):
        raise RegenerationError(f"{label} must contain an object")
    if "content_digest" in value:
        V1._validate_content_digest(value, label)
    return raw, value


def _dtype_item_size(descr: str, label: str) -> int:
    return V1._dtype_item_size(descr, label)


def _read_npz_members(
    path: Path,
    label: str,
    *,
    expected_names: set[str] | None = _EXPECTED_V1_SHARD_MEMBERS,
    expected_comment: str | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RegenerationError(f"{label} is not a valid NPZ") from exc
    with archive:
        if expected_comment is not None and archive.comment != expected_comment.encode(
            "utf-8"
        ):
            raise RegenerationError(f"{label} V2 fresh-archive comment drift")
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or not names
            or (expected_names is not None and set(names) != expected_names)
            or any(not name.endswith(".npy") or "/" in name for name in names)
        ):
            raise RegenerationError(f"{label} member inventory drift")
        for name in names:
            info = archive.getinfo(name)
            if info.is_dir() or info.flag_bits & 0x1:
                raise RegenerationError(f"{label}/{name} is encrypted or not a file")
            with archive.open(info, "r") as stream:
                header, _offset = _npy_header(stream, f"{label}/{name}")
                if header["fortran_order"] is not False:
                    raise RegenerationError(f"{label}/{name} is not C-contiguous")
                shape = tuple(header["shape"])
                item_size = _dtype_item_size(str(header["descr"]), f"{label}/{name}")
                count = 1
                for extent in shape:
                    count *= int(extent)
                payload = _read_exact(stream, count * item_size, f"{label}/{name}")
                if stream.read(1):
                    raise RegenerationError(f"{label}/{name} has trailing payload bytes")
            result[name] = {
                "dtype": str(header["descr"]),
                "shape": list(shape),
                "raw_c_bytes_sha256": hashlib.sha256(payload).hexdigest(),
                "v1_canonical_array_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        {"dtype": str(header["descr"]), "layout": "C", "shape": list(shape)}
                    )
                    + b"\x00"
                    + payload
                ).hexdigest(),
                "payload": payload,
            }
    return result


def _file_row(root_report: Mapping[str, Any], relative: str) -> dict[str, Any]:
    matches = [row for row in root_report["files"] if row["path"] == relative]
    if len(matches) != 1:
        raise RegenerationError(f"custody inventory lacks exactly one file: {relative}")
    return dict(matches[0])


def _validate_v1_pair(
    pool_index: int,
    material_root: Path,
    material_report: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = f"qualification/pool-{pool_index:03d}"
    metadata_relative = f"{prefix}/metadata.json"
    payload_relative = f"{prefix}/payload.npz"
    metadata_row = _file_row(material_report, metadata_relative)
    payload_row = _file_row(material_report, payload_relative)
    _raw, metadata = _load_canonical_json(material_root / metadata_relative, metadata_relative)
    if metadata.get("experiment_id") != V1_EXPERIMENT_ID or metadata.get("pool_index") != pool_index:
        raise RegenerationError(f"V1 pool-{pool_index:03d} identity drift")
    if metadata.get("reset_or_candidate_outcome_opened") is not False:
        raise RegenerationError(f"V1 pool-{pool_index:03d} crossed the candidate-outcome boundary")
    payload_binding = metadata.get("payload")
    if not isinstance(payload_binding, Mapping) or payload_binding != {
        "bytes": payload_row["bytes"],
        "kind": "npz",
        "path": payload_relative,
        "role": "material_shard_payload",
        "sha256": payload_row["sha256"],
    }:
        raise RegenerationError(f"V1 pool-{pool_index:03d} payload binding drift")
    members = _read_npz_members(material_root / payload_relative, payload_relative)
    snapshot = metadata.get("snapshot")
    teacher = metadata.get("teacher")
    if not isinstance(snapshot, Mapping) or not isinstance(teacher, Mapping):
        raise RegenerationError(f"V1 pool-{pool_index:03d} metadata projection drift")
    snapshot_payload = members["snapshot_payload_bytes.npy"]
    raw_snapshot_sha = snapshot_payload["raw_c_bytes_sha256"]
    if (
        snapshot.get("snapshot_payload_sha256") != raw_snapshot_sha
        or metadata.get("initial_decision_state_sha256") != raw_snapshot_sha
    ):
        raise RegenerationError(f"V1 pool-{pool_index:03d} serialized snapshot binding drift")
    for metadata_field, member_name in _SNAPSHOT_HASH_MEMBERS.items():
        expected = members[member_name]["v1_canonical_array_sha256"]
        supplied = snapshot.get(metadata_field)
        if metadata_field == "previous_applied_command_sha256":
            member = members[member_name]
            if (
                member["dtype"] != "<f8"
                or member["shape"] != [3]
                or expected != V1_PERSISTED_PREVIOUS_COMMAND_CANONICAL_SHA256
                or supplied != V1_INVALID_PREVIOUS_COMMAND_SHA256
                or member["payload"] != b"\x00" * 24
            ):
                raise RegenerationError(
                    f"V1 pool-{pool_index:03d} previous-command defect evidence drift"
                )
        elif supplied != expected:
            raise RegenerationError(
                f"V1 pool-{pool_index:03d} snapshot hash drift: {metadata_field}"
            )
    if metadata.get("current_rgb_sha256") != members["rgb.npy"]["v1_canonical_array_sha256"]:
        raise RegenerationError(f"V1 pool-{pool_index:03d} RGB hash drift")
    trace_digests = teacher.get("trace_digests")
    if not isinstance(trace_digests, Mapping) or set(trace_digests) != set(_TEACHER_MEMBERS):
        raise RegenerationError(f"V1 pool-{pool_index:03d} teacher trace digest inventory drift")
    for member in _TEACHER_MEMBERS:
        if trace_digests[member] != members[f"teacher__{member}.npy"]["v1_canonical_array_sha256"]:
            raise RegenerationError(f"V1 pool-{pool_index:03d} teacher trace hash drift: {member}")
    sample_count = V1_TEACHER_SAMPLE_COUNTS[pool_index]
    if teacher.get("sample_count") != sample_count:
        raise RegenerationError(f"V1 pool-{pool_index:03d} teacher sample count drift")
    for member in _TEACHER_MEMBERS:
        if members[f"teacher__{member}.npy"]["shape"][0] != sample_count:
            raise RegenerationError(f"V1 pool-{pool_index:03d} teacher member count drift")
    qualified = pool_index != V1_REJECTED_POOL_INDEX
    reason = "QUALIFIED" if qualified else V1_REJECTION_REASON
    if metadata.get("qualified") is not qualified or metadata.get("rejection_reason") != reason:
        raise RegenerationError(f"V1 pool-{pool_index:03d} disposition drift")
    contact_member = members["teacher__physics_contact.npy"]
    if contact_member["dtype"] != "|u1" or any(
        value not in (0, 1) for value in contact_member["payload"]
    ):
        raise RegenerationError(f"V1 pool-{pool_index:03d} contact encoding drift")
    contact_count = sum(contact_member["payload"])
    if (pool_index == V1_REJECTED_POOL_INDEX) != (contact_count > 0):
        raise RegenerationError(f"V1 pool-{pool_index:03d} physical-contact evidence drift")
    pair_projection = {
        "pool_index": pool_index,
        "metadata": metadata_row,
        "payload": payload_row,
    }
    return {
        **pair_projection,
        "pair_sha256": hashlib.sha256(canonical_json_bytes(pair_projection)).hexdigest(),
        "qualified": qualified,
        "rejection_reason": reason,
        "teacher_sample_count": sample_count,
        "physics_contact_sample_count": contact_count,
        "persisted_previous_applied_command": {
            "dtype": "<f8",
            "shape": [3],
            "raw_c_bytes_sha256": members[
                "snapshot__previous_applied_command.npy"
            ]["raw_c_bytes_sha256"],
            "canonical_array_sha256": V1_PERSISTED_PREVIOUS_COMMAND_CANONICAL_SHA256,
            "invalid_metadata_sha256": V1_INVALID_PREVIOUS_COMMAND_SHA256,
        },
    }


def _assert_same_inventory(before: Mapping[str, Any], after: Mapping[str, Any], label: str) -> None:
    if before != after:
        raise RegenerationError(f"{label} changed during custody audit")


def build_v1_custody_receipt(
    official_root: Path | str = DEFAULT_V1_OUTPUT_ROOT,
    material_root: Path | str = DEFAULT_V1_MATERIAL_ROOT,
) -> dict[str, Any]:
    """Rebuild the complete ordinary receipt for the immutable V1 partial run."""

    official_path = _root_path(official_root, "V1 official root")
    material_path = _root_path(material_root, "V1 material root")
    official_before = _root_inventory(official_path, "V1 official root")
    material_before = _root_inventory(material_path, "V1 material root")
    if (
        official_before["file_count"] != 1
        or official_before["directory_count"] != 1
        or official_before["regular_file_apparent_bytes"] != V1_OFFICIAL_CONTRACT_BYTES
    ):
        raise RegenerationError("V1 official partial-root inventory drift")
    contract_row = _file_row(official_before, "contract.json")
    if (
        contract_row["bytes"] != V1_OFFICIAL_CONTRACT_BYTES
        or contract_row["sha256"] != V1_OFFICIAL_CONTRACT_SHA256
    ):
        raise RegenerationError("V1 official contract binding drift")
    _raw_contract, contract = _load_canonical_json(official_path / "contract.json", "V1 contract")
    if contract.get("experiment_id") != V1_EXPERIMENT_ID:
        raise RegenerationError("V1 official contract experiment identity drift")
    if (
        material_before["file_count"] != V1_MATERIAL_FILE_COUNT
        or material_before["regular_file_apparent_bytes"] != V1_MATERIAL_APPARENT_BYTES
    ):
        raise RegenerationError("V1 material-root inventory drift")
    expected_directories = {
        ".",
        "fanout",
        "logs",
        "qualification",
        *(f"qualification/pool-{index:03d}" for index in range(V1_PAIR_COUNT)),
        "repeat",
        "scenes",
        "selected",
    }
    if {row["path"] for row in material_before["directories"]} != expected_directories:
        raise RegenerationError("V1 material directory inventory drift")
    expected_files = {"material_contract.json", "prospective_pool.json"}
    for index in range(V1_PAIR_COUNT):
        expected_files.update(
            {
                f"qualification/pool-{index:03d}/metadata.json",
                f"qualification/pool-{index:03d}/payload.npz",
                f"logs/qualify-{index:03d}.stdout.log",
                f"logs/qualify-{index:03d}.stderr.log",
            }
        )
    if {row["path"] for row in material_before["files"]} != expected_files:
        raise RegenerationError("V1 material file inventory drift")
    for name in ("material_contract.json", "prospective_pool.json"):
        _load_canonical_json(material_path / name, f"V1 {name}")
    for index in range(V1_PAIR_COUNT):
        if _file_row(material_before, f"logs/qualify-{index:03d}.stderr.log")["bytes"] != 0:
            raise RegenerationError(f"V1 qualification {index} stderr is not empty")
    pairs = [
        _validate_v1_pair(index, material_path, material_before)
        for index in range(V1_PAIR_COUNT)
    ]
    official_after = _root_inventory(official_path, "V1 official root")
    material_after = _root_inventory(material_path, "V1 material root")
    _assert_same_inventory(official_before, official_after, "V1 official root")
    _assert_same_inventory(material_before, material_after, "V1 material root")
    return {
        "schema": V1_CUSTODY_SCHEMA,
        "experiment_id": V1_EXPERIMENT_ID,
        "source_freeze_commit": SOURCE_PARENT_COMMIT,
        "disposition": "TECHNICALLY_INVALID_PERSISTED_ARRAY_HASH_METADATA",
        "official_root": official_before,
        "material_root": material_before,
        "qualification_pairs": pairs,
        "defect_evidence": {
            "field": "snapshot.previous_applied_command_sha256",
            "failing_invariant": "persisted_array_dtype_shape_hash_must_match_reopened_npz",
            "metadata_source_dtype": "<f4",
            "persisted_dtype": "<f8",
            "shape": [3],
            "values": [0.0, 0.0, 0.0],
            "metadata_sha256": V1_INVALID_PREVIOUS_COMMAND_SHA256,
            "persisted_v1_canonical_array_sha256": V1_PERSISTED_PREVIOUS_COMMAND_CANONICAL_SHA256,
            "affected_pair_count": V1_PAIR_COUNT,
            "other_embedded_numeric_hash_mismatch_count": 0,
        },
        "scientific_boundary": {
            "teacher_qualification_rows_opened": V1_PAIR_COUNT,
            "teacher_qualified": V1_QUALIFIED_COUNT,
            "teacher_rejected": 1,
            "rejected_pool_index": V1_REJECTED_POOL_INDEX,
            "rejection_reason": V1_REJECTION_REASON,
            "prospective_pool_rows": 256,
            "selection_performed": False,
            "selected_state_count": 0,
            "reset_fixture_executions": 0,
            "candidate_fanout_executions": 0,
            "encoder_initializations": 0,
            "ranker_inference_calls": 0,
            "heldout_outcomes_opened": 0,
            "metrics_persisted": False,
            "result_persisted": False,
        },
        "repository": {
            "v1_source_freeze_commit": SOURCE_PARENT_COMMIT,
            "v1_freeze_subject": "Freeze physical graph edge handoff qualification",
        },
        "immutability": {
            "audit_mode": "read_only_same_inode_before_after",
            "official_root_unchanged_during_audit": True,
            "material_root_unchanged_during_audit": True,
            "all_files_regular_single_link": True,
            "receipt_outside_both_roots": True,
        },
    }


def _external_location(root_paths: Sequence[Path], output: Path | str) -> tuple[int, str]:
    target = Path(output)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise RegenerationError("external receipt path must be an absolute ordinary filename")
    for root in root_paths:
        try:
            target.relative_to(root)
        except ValueError:
            pass
        else:
            raise RegenerationError("external receipt must remain outside experiment roots")
    parent = target.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise RegenerationError("external receipt parent traverses a symlink")
    return os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW), target.name


def _emit_external(
    root_paths: Sequence[Path], output: Path | str, value: Mapping[str, Any]
) -> bytes:
    V1._validate_no_self_digest(value, "external receipt")
    raw = canonical_document_bytes(dict(value))
    parent_fd, leaf = _external_location(root_paths, output)
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            existing = CUSTODY._read_regular_at(parent_fd, leaf)
            if existing != raw:
                raise RegenerationError("existing external receipt bytes drift")
        else:
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        raise OSError("short external receipt write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return raw


def emit_v1_custody_receipt(
    output: Path | str = DEFAULT_V1_CUSTODY_RECEIPT,
    *,
    official_root: Path | str = DEFAULT_V1_OUTPUT_ROOT,
    material_root: Path | str = DEFAULT_V1_MATERIAL_ROOT,
    receipt: Mapping[str, Any] | None = None,
) -> bytes:
    value = (
        build_v1_custody_receipt(official_root, material_root)
        if receipt is None
        else dict(receipt)
    )
    return _emit_external(
        (Path(official_root), Path(material_root)), output, value
    )


def validate_existing_v1_custody_receipt(
    output: Path | str = DEFAULT_V1_CUSTODY_RECEIPT,
    *,
    official_root: Path | str = DEFAULT_V1_OUTPUT_ROOT,
    material_root: Path | str = DEFAULT_V1_MATERIAL_ROOT,
) -> dict[str, Any]:
    raw, supplied = _load_canonical_json(Path(output), "V1 external custody receipt")
    V1._validate_no_self_digest(supplied, "V1 external custody receipt")
    rebuilt = build_v1_custody_receipt(official_root, material_root)
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("V1 external custody receipt differs from exact rebuild")
    return rebuilt


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def raw_c_array_sha256(payload: bytes) -> str:
    """V2 persisted-array domain: exact C bytes only; shape/dtype are separate."""

    return hashlib.sha256(payload).hexdigest()


def _load_ordinary_json_at(root_fd: int, leaf: str) -> tuple[bytes, dict[str, Any]]:
    raw = CUSTODY._read_regular_at(root_fd, leaf)
    assert raw is not None
    value = parse_canonical_json(raw, label=leaf)
    if not isinstance(value, dict):
        raise RegenerationError(f"{leaf} must contain an object")
    V1._reject_nonfinite(value, label=leaf)
    V1._validate_no_self_digest(value, leaf)
    return raw, value


def _call(module: Any, name: str, *arguments: Any, **keywords: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"V2 pure metrics API is absent: {name}")
    try:
        return function(*arguments, **keywords)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(f"V2 pure metrics rejected {name}: {exc}") from exc


def _validate_v2_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "reducer_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("V2 reducer authority must be an object")
    authority = dict(authority)
    V1._validate_content_digest(authority, "V2 reducer authority")
    if (
        authority.get("schema")
        != "physical_graph_edge_handoff_qualification_v2.reducer_authority.v1"
        or authority.get("experiment_id") != EXPERIMENT_ID
    ):
        raise RegenerationError("V2 reducer authority identity drift")
    required = {
        "evidence_keys",
        "documents",
        "ledgers",
        "npz_authorities",
        "runtime_paths",
        "external_artifact_roles",
        "trace_index_ranges",
        "reset_fixture_physics_samples",
        "branch_physics_samples",
        "reset_pair_comparison_fields",
        "reset_trace_pair_comparison_authority",
        "physical_trace_reduction_authority",
        "runtime_environment_authority",
        "successful_output_leaf_count",
        "successful_output_leaves",
        "reproduction_mismatch_output_leaves",
        "persisted_array_hash_authority",
        "persisted_array_evidence",
        "new_documents",
        "v1_compatibility_identity_authority",
        "port_heading_implementation_alignment_authority",
        "candidate_port_metric_implementation_alignment_authority",
        "first_eight_reproduction_authority",
    }
    if not required.issubset(authority):
        raise RegenerationError(
            f"V2 reducer authority lacks sections: {sorted(required-set(authority))}"
        )
    expected_evidence = {
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
    if set(authority["evidence_keys"]) != expected_evidence:
        raise RegenerationError("V2 evidence-key authority drift")
    if (
        authority["successful_output_leaf_count"] != 26
        or tuple(authority["successful_output_leaves"]) != tuple(V2_SUCCESS_FILES)
        or tuple(authority["reproduction_mismatch_output_leaves"])
        != V2_REPRODUCTION_FAILURE_FILES
        or set(authority["runtime_paths"].values()) != set(V2_SUCCESS_FILES)
        or len(authority["runtime_paths"]) != 26
    ):
        raise RegenerationError("V2 output inventory authority drift")
    if set(authority["npz_authorities"]) != set(V1.PAYLOAD_FILES):
        raise RegenerationError("V2 NPZ authority inventory drift")
    if set(authority["documents"]) != {
        "panel_manifest",
        "split_manifest",
        "graph_manifest",
        "state_snapshot_index",
        "teacher_trace_index",
        "edge_port_index",
        "waypoint_contracts",
        "pixel_index",
        "latent_index",
        "development_target_selection",
    }:
        raise RegenerationError("V2 document authority inventory drift")
    if set(authority["ledgers"]) != {
        "candidate_fanout",
        "heldout_ranker_scores",
        "repeated_execution",
    }:
        raise RegenerationError("V2 ledger authority inventory drift")
    expected_counts = {
        "candidate_fanout": 768,
        "heldout_ranker_scores": 64,
        "repeated_execution": 64,
    }
    if any(
        authority["ledgers"][name].get("count") != count
        for name, count in expected_counts.items()
    ):
        raise RegenerationError("V2 ledger cardinality authority drift")
    if authority["trace_index_ranges"] != {
        "reset_fixture": [0, 128],
        "candidate_fanout": [128, 896],
        "repeated_execution": [896, 960],
    } or authority["reset_fixture_physics_samples"] != 750 or authority[
        "branch_physics_samples"
    ] != 750:
        raise RegenerationError("V2 physical trace range/duration authority drift")
    digest_authority = authority["persisted_array_hash_authority"]
    if not isinstance(digest_authority, Mapping) or (
        digest_authority.get("digest_domain")
        != "exact C-contiguous array.tobytes(order='C') only"
        or digest_authority.get("tolerance") != 0
        or digest_authority.get("npz_archive_comment_utf8")
        != V2_NPZ_ARCHIVE_COMMENT
        or digest_authority.get("npz_archive_comment_required") is not True
    ):
        raise RegenerationError("V2 raw persisted-byte digest authority drift")
    first_eight_authority = authority["first_eight_reproduction_authority"]
    if not isinstance(first_eight_authority, Mapping):
        raise RegenerationError("V2 first-eight authority is absent")
    V1._validate_content_digest(first_eight_authority, "V2 first-eight authority")
    corrected_binding = first_eight_authority.get(
        "corrected_v2_previous_command_binding"
    )
    if (
        first_eight_authority.get("content_digest")
        != "85b62dc9bac6c0b31cd031f1fba76298be3525cfb4702f05210ff39ae1322cb4"
        or not isinstance(corrected_binding, Mapping)
        or corrected_binding.get("digest_domain")
        != "exact C-contiguous persisted bytes only"
        or corrected_binding.get("dtype_str") != "<f8"
        or corrected_binding.get("per_snapshot_shape") != [3]
        or corrected_binding.get("first_eight_expected_raw_bytes_sha256")
        != V2_PERSISTED_PREVIOUS_COMMAND_RAW_SHA256
        or corrected_binding.get("other_snapshot_metadata_hash_fields_keep_frozen_v1_domains")
        is not True
    ):
        raise RegenerationError("V2 first-eight raw previous-command authority drift")
    new_documents = authority["new_documents"]
    if not isinstance(new_documents, Mapping) or set(new_documents) != {
        "v1_custody_and_nonreuse",
        "scientific_invariance_receipt",
        "v1_v2_first_eight_reproduction",
    } or any(value.get("content_digest_forbidden") is not True for value in new_documents.values()):
        raise RegenerationError("V2 ordinary receipt authority drift")
    identity = authority["v1_compatibility_identity_authority"]
    if not isinstance(identity, Mapping):
        raise RegenerationError("V2 compatibility identity authority is absent")
    V1._validate_content_digest(identity, "V2 compatibility identity authority")
    if (
        identity.get("official_experiment_id") != EXPERIMENT_ID
        or identity.get("official_schema_token")
        != "physical_graph_edge_handoff_qualification_v2"
        or identity.get("frozen_inherited_scientific_identity_salt")
        != V1_EXPERIMENT_ID
        or identity.get("v1_scientific_completion_claimed") is not False
        or identity.get("v2_execution_and_result_identity_claimed") is not True
    ):
        raise RegenerationError("V2 compatibility identity boundary drift")
    alignment = authority["port_heading_implementation_alignment_authority"]
    if not isinstance(alignment, Mapping):
        raise RegenerationError("V2 port-heading alignment authority is absent")
    V1._validate_content_digest(alignment, "V2 port-heading alignment authority")
    if (
        alignment.get("content_digest")
        != "88860aba7ef58f36d0ef78a50fcf364614528c9770be55692c17ae4d722ce37e"
        or alignment.get("disposition")
        != "INHERITED_PORT_HEADING_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
        or alignment.get("changed_output_fields") != ["directed_port_world[2]"]
        or alignment.get("exact_formula")
        != "directed_port_world[2] = atan2(opening_normal_world[1], opening_normal_world[0])"
        or alignment.get("aligns_implementation_to_frozen_v1_authority") is not True
        or alignment.get("changes_frozen_metric_gate_class_precedence_or_next_decision")
        is not False
        or alignment.get("changes_frozen_scientific_design") is not False
        or alignment.get("continuous_tuning_or_new_target_definition") is not False
        or alignment.get("uses_outcome_values_to_choose_formula_or_parameter") is not False
    ):
        raise RegenerationError("V2 port-heading alignment authority drift")
    candidate_alignment = authority[
        "candidate_port_metric_implementation_alignment_authority"
    ]
    if not isinstance(candidate_alignment, Mapping):
        raise RegenerationError("V2 candidate-port alignment authority is absent")
    V1._validate_content_digest(
        candidate_alignment, "V2 candidate-port alignment authority"
    )
    if (
        candidate_alignment.get("content_digest")
        != "28061cfb202a17d82498b5b54f7c8a98d70d836ce14544d92a33d3935cc0c73a"
        or candidate_alignment.get("disposition")
        != "INHERITED_CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
        or candidate_alignment.get("changed_output_fields")
        != ["port_progress_m", "lateral_error_m", "positive_port_progress"]
        or candidate_alignment.get("aligns_implementation_to_frozen_v1_authority")
        is not True
        or candidate_alignment.get(
            "changes_frozen_metric_gate_class_precedence_or_next_decision"
        )
        is not False
        or candidate_alignment.get("changes_frozen_scientific_design") is not False
        or candidate_alignment.get("continuous_tuning_or_new_target_definition")
        is not False
        or candidate_alignment.get("uses_outcome_values_to_choose_formula_or_parameter")
        is not False
    ):
        raise RegenerationError("V2 candidate-port alignment authority drift")
    physical = authority["physical_trace_reduction_authority"]
    if not isinstance(physical, Mapping):
        raise RegenerationError("V2 physical reduction authority is absent")
    V1._validate_content_digest(physical, "V2 physical reduction authority")
    if (
        physical.get("experiment_id") != EXPERIMENT_ID
        or physical.get("physics_dt_s") != 0.002
        or physical.get("dwell_samples") != 100
        or len(physical.get("specs", ())) != 256
    ):
        raise RegenerationError("V2 physical reduction authority drift")
    return json.loads(json.dumps(authority))


def _open_v1_custody_with_binding(module: Any) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    raw, supplied = _load_canonical_json(
        DEFAULT_V1_CUSTODY_RECEIPT, "external V1 custody receipt"
    )
    binding = {
        "path": str(DEFAULT_V1_CUSTODY_RECEIPT),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    authority = _call(module, "reducer_authority")
    expected = authority["v1_custody_and_nonreuse_authority"]["external_receipt"]
    if binding != expected:
        raise RegenerationError("external V1 custody receipt binding drift")
    validated = _call(module, "validate_external_v1_custody_receipt", supplied)
    rebuilt = build_v1_custody_receipt()
    if raw != canonical_document_bytes(rebuilt) or validated != rebuilt:
        raise RegenerationError("external V1 custody receipt is not an exact live rebuild")
    return raw, rebuilt, binding


def _validate_persisted_array_evidence(
    *,
    metadata: Mapping[str, Any],
    members: Mapping[str, Mapping[str, Any]],
    payload_binding: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    evidence = metadata.get("persisted_array_evidence")
    fields = {
        "schema",
        "experiment_id",
        "shard_kind",
        "shard_id",
        "payload_file",
        "array_count",
        "array_inventory_sha256",
        "arrays",
        "save_reopen_validation_passed",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise RegenerationError(f"{label} persisted-array evidence field drift")
    if (
        evidence.get("schema")
        != "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1"
        or evidence.get("experiment_id") != EXPERIMENT_ID
        or evidence.get("payload_file") != payload_binding
        or evidence.get("save_reopen_validation_passed") is not True
    ):
        raise RegenerationError(f"{label} persisted-array evidence identity/binding drift")
    rows = evidence.get("arrays")
    if not isinstance(rows, list) or evidence.get("array_count") != len(rows):
        raise RegenerationError(f"{label} persisted-array row count drift")
    expected_rows = [
        {
            "member": name[:-4],
            "dtype_str": members[name]["dtype"],
            "shape": members[name]["shape"],
            "c_contiguous": True,
            "array_bytes_sha256": members[name]["raw_c_bytes_sha256"],
        }
        for name in sorted(members)
    ]
    if rows != expected_rows or evidence.get("array_inventory_sha256") != hashlib.sha256(
        canonical_json_bytes(expected_rows)
    ).hexdigest():
        raise RegenerationError(f"{label} persisted-array raw-byte inventory drift")
    previous_member = members.get("snapshot__previous_applied_command.npy")
    snapshot = metadata.get("snapshot")
    snapshot_has_field = isinstance(snapshot, Mapping) and (
        "previous_applied_command_sha256" in snapshot
    )
    if previous_member is not None or snapshot_has_field:
        if (
            previous_member is None
            or not isinstance(snapshot, Mapping)
            or previous_member.get("dtype") != "<f8"
            or previous_member.get("shape") != [3]
            or snapshot.get("previous_applied_command_sha256")
            != previous_member.get("raw_c_bytes_sha256")
        ):
            raise RegenerationError(
                f"{label} previous-command raw persisted-byte binding drift"
            )
    return dict(evidence)


def _float64_values(member: Mapping[str, Any], label: str) -> list[float]:
    if member.get("dtype") != "<f8" or len(member["payload"]) % 8:
        raise RegenerationError(f"{label} must be little-endian float64")
    values = [row[0] for row in struct.iter_unpack("<d", member["payload"])]
    if any(not math.isfinite(value) for value in values):
        raise RegenerationError(f"{label} contains non-finite values")
    return values


def _yaw_xyzw(row: Sequence[float]) -> float:
    x, y, z, w = row[3:7]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _teacher_stuck_from_members(members: Mapping[str, Mapping[str, Any]], label: str) -> bool:
    pose_member = members["teacher__base_pose_world.npy"]
    command_member = members["teacher__requested_command.npy"]
    pose_shape = pose_member["shape"]
    command_shape = command_member["shape"]
    if len(pose_shape) != 2 or pose_shape[1] != 7 or command_shape != [pose_shape[0], 3]:
        raise RegenerationError(f"{label} teacher stuck input shape drift")
    poses = _float64_values(pose_member, f"{label}.pose")
    commands = _float64_values(command_member, f"{label}.command")
    first = poses[:7]
    final = poses[-7:]
    activity = max((abs(value) for value in commands), default=0.0)
    translation = math.hypot(final[0] - first[0], final[1] - first[1])
    heading = abs(_wrapped(_yaw_xyzw(final) - _yaw_xyzw(first)))
    return bool(activity > 0.05 and translation < 0.02 and heading < 0.05)


def _v1_to_v2_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _v1_to_v2_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_v1_to_v2_identity(item) for item in value]
    if isinstance(value, str):
        if value == V1_EXPERIMENT_ID:
            return EXPERIMENT_ID
        return value.replace(
            "physical_graph_edge_handoff_qualification_v1",
            "physical_graph_edge_handoff_qualification_v2",
        )
    return value


def _comparable_first_eight_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = _v1_to_v2_identity(value)
    if not isinstance(projected, dict):
        raise RegenerationError("first-eight metadata projection is not an object")
    # Stage runtime is observed physical evidence, not a storage or namespace
    # binding.  It therefore has to reproduce exactly across V1 and V2.
    for field in ("content_digest", "payload", "persisted_array_evidence"):
        projected.pop(field, None)
    snapshot = projected.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise RegenerationError("first-eight metadata snapshot projection is absent")
    snapshot = dict(snapshot)
    snapshot.pop("previous_applied_command_sha256", None)
    projected["snapshot"] = snapshot
    return projected


def _v2_material_pair(
    pool_index: int, material_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    directory = material_root / "qualification" / f"pool-{pool_index:03d}"
    metadata_path = directory / "metadata.json"
    payload_path = directory / "payload.npz"
    _raw, metadata = _load_canonical_json(metadata_path, f"V2 pool-{pool_index:03d} metadata")
    if metadata.get("experiment_id") != EXPERIMENT_ID or metadata.get("pool_index") != pool_index:
        raise RegenerationError(f"V2 pool-{pool_index:03d} identity drift")
    info = payload_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RegenerationError(f"V2 pool-{pool_index:03d} payload custody drift")
    payload_file = {
        "path": f"qualification/pool-{pool_index:03d}/payload.npz",
        "bytes": int(info.st_size),
        "sha256": _sha256_file(payload_path),
    }
    payload_binding = {"role": "material_shard_payload", **payload_file, "kind": "npz"}
    if metadata.get("payload") != payload_binding:
        raise RegenerationError(f"V2 pool-{pool_index:03d} payload binding drift")
    members = _read_npz_members(
        payload_path,
        f"V2 pool-{pool_index:03d} payload",
        # The first-eight gate must turn a complete-but-different logical
        # member inventory into a registered reproduction mismatch row.  The
        # exact V1 inventory is compared below rather than rejected here.
        expected_names=None,
        expected_comment=V2_NPZ_ARCHIVE_COMMENT,
    )
    evidence = _validate_persisted_array_evidence(
        metadata=metadata,
        members=members,
        payload_binding=payload_file,
        label=f"V2 pool-{pool_index:03d}",
    )
    return metadata, members, payload_file, evidence


def _first_eight_reproduction_rows(
    material_root: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v2_root = _root_path(material_root, "V2 material root")
    v1_report = _root_inventory(DEFAULT_V1_MATERIAL_ROOT, "V1 material root")
    v2_report = _root_inventory(v2_root, "V2 material root")
    v1_inode_set = {
        (row["device"], row["inode"]) for row in v1_report["files"]
    }
    shared_inodes = [
        row["path"]
        for row in v2_report["files"]
        if (row["device"], row["inode"]) in v1_inode_set
    ]
    if shared_inodes:
        raise RegenerationError(f"V2 material shares V1 inodes: {shared_inodes}")
    v1_payload_hashes = {
        row["sha256"]
        for row in v1_report["files"]
        if row["path"].endswith(("/metadata.json", "/payload.npz"))
    }
    copied = [row["path"] for row in v2_report["files"] if row["sha256"] in v1_payload_hashes]
    if copied:
        raise RegenerationError(f"V2 contains exact V1 shard-file copies: {copied}")
    rows: list[dict[str, Any]] = []
    for index in range(8):
        v1_directory = DEFAULT_V1_MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
        _v1_raw, v1_metadata = _load_canonical_json(
            v1_directory / "metadata.json", f"V1 pool-{index:03d} metadata"
        )
        v1_members = _read_npz_members(
            v1_directory / "payload.npz", f"V1 pool-{index:03d} payload"
        )
        v2_metadata, v2_members, _binding, _evidence = _v2_material_pair(index, v2_root)
        v1_spec = v1_metadata.get("candidate_spec")
        v2_spec = v2_metadata.get("candidate_spec")
        if not isinstance(v1_spec, Mapping) or not isinstance(v2_spec, Mapping):
            raise RegenerationError(f"pool-{index:03d} candidate spec is absent")
        identity_fields = ("candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id")
        try:
            comparable_metadata_equal = (
                _comparable_first_eight_metadata(v1_metadata)
                == _comparable_first_eight_metadata(v2_metadata)
            )
        except RegenerationError:
            comparable_metadata_equal = False
        identity_equal = bool(
            v1_spec == v2_spec
            and comparable_metadata_equal
            and all(v1_spec.get(field) == v2_spec.get(field) for field in identity_fields)
        )
        v1_names = sorted(v1_members)
        v2_names = sorted(v2_members)
        shared = sorted(set(v1_names) & set(v2_names))
        teacher_v1 = [name for name in v1_names if name.startswith("teacher__")]
        teacher_v2 = [name for name in v2_names if name.startswith("teacher__")]
        dtype_equal = all(v1_members[name]["dtype"] == v2_members[name]["dtype"] for name in shared)
        shape_equal = all(v1_members[name]["shape"] == v2_members[name]["shape"] for name in shared)
        payload_equal = all(v1_members[name]["payload"] == v2_members[name]["payload"] for name in shared)
        teacher_payload_equal = teacher_v1 == teacher_v2 and all(
            v1_members[name]["payload"] == v2_members[name]["payload"] for name in teacher_v1
        )

        def member_payload_equal(name: str) -> bool:
            return bool(
                name in v1_members
                and name in v2_members
                and v1_members[name]["payload"] == v2_members[name]["payload"]
            )

        def safely_derive_stuck(
            members: Mapping[str, Mapping[str, Any]], label: str
        ) -> bool | None:
            try:
                return _teacher_stuck_from_members(members, label)
            except (KeyError, RegenerationError):
                return None

        v1_stuck = safely_derive_stuck(v1_members, f"V1 pool-{index:03d}")
        v2_stuck = safely_derive_stuck(v2_members, f"V2 pool-{index:03d}")
        v1_snapshot = v1_metadata.get("snapshot")
        v2_snapshot = v2_metadata.get("snapshot")
        snapshot_mappings = isinstance(v1_snapshot, Mapping) and isinstance(
            v2_snapshot, Mapping
        )
        other_snapshot_hashes_exact = bool(snapshot_mappings)
        if snapshot_mappings:
            for metadata_field, member_name in _SNAPSHOT_HASH_MEMBERS.items():
                if metadata_field == "previous_applied_command_sha256":
                    continue
                member = v2_members.get(member_name)
                other_snapshot_hashes_exact = bool(
                    other_snapshot_hashes_exact
                    and member is not None
                    and v1_snapshot.get(metadata_field)
                    == v2_snapshot.get(metadata_field)
                    and v2_snapshot.get(metadata_field)
                    == member.get("v1_canonical_array_sha256")
                )
        identity_equal = bool(identity_equal and other_snapshot_hashes_exact)
        booleans = {
            "identity_equal": identity_equal,
            "snapshot_payload_sha256_equal": (
                member_payload_equal("snapshot_payload_bytes.npy")
                and snapshot_mappings
                and v1_snapshot.get("snapshot_payload_sha256")
                == v2_snapshot.get("snapshot_payload_sha256")
            ),
            "teacher_trace_member_inventory_equal": teacher_v1 == teacher_v2,
            "teacher_trace_dtypes_equal": teacher_v1 == teacher_v2 and all(
                v1_members[name]["dtype"] == v2_members[name]["dtype"] for name in teacher_v1
            ),
            "teacher_trace_shapes_equal": teacher_v1 == teacher_v2 and all(
                v1_members[name]["shape"] == v2_members[name]["shape"] for name in teacher_v1
            ),
            "teacher_trace_logical_arrays_equal": teacher_payload_equal,
            "contact_sequence_equal": member_payload_equal(
                "teacher__physics_contact.npy"
            ),
            "stuck_equal": v1_stuck is not None and v1_stuck == v2_stuck,
            "qualified_equal": v1_metadata.get("qualified") is v2_metadata.get("qualified"),
            "rejection_reason_equal": v1_metadata.get("rejection_reason")
            == v2_metadata.get("rejection_reason"),
            "all_payload_member_dtypes_equal": v1_names == v2_names and dtype_equal,
            "all_payload_member_shapes_equal": v1_names == v2_names and shape_equal,
            "shared_logical_array_members_equal": v1_names == v2_names and payload_equal,
        }
        v2_previous = v2_members.get("snapshot__previous_applied_command.npy")
        v2_previous_raw_domain = (
            v2_previous.get("raw_c_bytes_sha256")
            if v2_previous is not None
            else None
        )
        known_defect_exact = bool(
            snapshot_mappings
            and v2_previous is not None
            and v2_previous.get("dtype") == "<f8"
            and v2_previous.get("shape") == [3]
            and v1_snapshot.get("previous_applied_command_sha256")
            == V1_INVALID_PREVIOUS_COMMAND_SHA256
            and v2_snapshot.get("previous_applied_command_sha256")
            == v2_previous_raw_domain
            and v2_previous_raw_domain
            == V2_PERSISTED_PREVIOUS_COMMAND_RAW_SHA256
        )
        row = {
            "pool_index": index,
            "candidate_spec_id": str(v2_spec["candidate_spec_id"]),
            "state_id": str(v2_spec["state_id"]),
            "scene_id": str(v2_spec["scene_id"]),
            "episode_id": str(v2_spec["episode_id"]),
            "graph_id": str(v2_spec["graph_id"]),
            **booleans,
            "noncomparable_v1_known_bad_hash_fields": [
                "snapshot.previous_applied_command_sha256"
            ],
            "pass": all(booleans.values()) and known_defect_exact,
        }
        rows.append(row)
    return rows, {
        "v1_material_file_count": v1_report["file_count"],
        "v2_material_file_count": v2_report["file_count"],
        "shared_inode_count": 0,
        "exact_v1_shard_file_copy_count": 0,
        "first_eight_payloads_fresh_files": True,
    }


def _validate_all_v2_material_payloads(
    material_root: Path | str,
    *,
    expected_shard_count: int,
    expected_metadata_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reopen every V2 material NPZ and rebuild its raw-byte manifest."""

    root = _root_path(material_root, "V2 material root")
    evidence_rows: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        if metadata_path.is_symlink():
            raise RegenerationError("V2 material metadata traverses a symlink")
        relative_metadata = metadata_path.relative_to(root).as_posix()
        _raw, metadata = _load_canonical_json(
            metadata_path, f"V2 material {relative_metadata}"
        )
        if metadata.get("experiment_id") != EXPERIMENT_ID:
            raise RegenerationError(f"V2 material experiment identity drift: {relative_metadata}")
        payload = metadata.get("payload")
        if not isinstance(payload, Mapping) or set(payload) != {
            "role",
            "path",
            "bytes",
            "sha256",
            "kind",
        }:
            raise RegenerationError(f"V2 material payload binding field drift: {relative_metadata}")
        relative_payload = payload.get("path")
        if not isinstance(relative_payload, str):
            raise RegenerationError(f"V2 material payload path drift: {relative_metadata}")
        payload_path = (root / relative_payload)
        if payload_path.resolve(strict=True).parent != metadata_path.parent.resolve(strict=True):
            raise RegenerationError(f"V2 material payload/metadata directory drift: {relative_metadata}")
        info = payload_path.stat(follow_symlinks=False)
        binding = {
            "path": relative_payload,
            "bytes": int(info.st_size),
            "sha256": _sha256_file(payload_path),
        }
        if payload != {"role": "material_shard_payload", **binding, "kind": "npz"}:
            raise RegenerationError(f"V2 material payload file binding drift: {relative_metadata}")
        members = _read_npz_members(
            payload_path,
            f"V2 material {relative_payload}",
            expected_names=None,
            expected_comment=V2_NPZ_ARCHIVE_COMMENT,
        )
        evidence = _validate_persisted_array_evidence(
            metadata=metadata,
            members=members,
            payload_binding=binding,
            label=f"V2 material {relative_metadata}",
        )
        schema = metadata.get("schema")
        expected_kind = (
            schema.rsplit(".", 2)[-2]
            if isinstance(schema, str) and schema.count(".") >= 2
            else None
        )
        if (
            evidence.get("shard_id") != metadata_path.parent.name
            or evidence.get("shard_kind") != expected_kind
        ):
            raise RegenerationError(
                f"V2 material shard identity drift: {relative_metadata}"
            )
        evidence_rows.append(
            {
                "metadata_path": relative_metadata,
                "payload_path": relative_payload,
                "payload_bytes": binding["bytes"],
                "payload_sha256": binding["sha256"],
                "array_count": evidence["array_count"],
                "array_inventory_sha256": evidence["array_inventory_sha256"],
            }
        )
    if len(evidence_rows) != expected_shard_count:
        raise RegenerationError(
            f"V2 material persisted-shard count drift: expected {expected_shard_count}, "
            f"observed {len(evidence_rows)}"
        )
    observed_metadata_paths = [row["metadata_path"] for row in evidence_rows]
    if expected_metadata_paths is not None and observed_metadata_paths != sorted(
        expected_metadata_paths
    ):
        raise RegenerationError("V2 material stage/shard path inventory drift")
    return {
        "validated_shard_count": len(evidence_rows),
        "all_npz_members_reopened": True,
        "all_dtype_shape_raw_c_byte_hashes_exact": True,
        "shard_projection_sha256": hashlib.sha256(
            canonical_json_bytes(evidence_rows)
        ).hexdigest(),
    }


def _validate_internal_gates(
    *,
    module: Any,
    root_fd: int,
    material_root: Path | str,
    source_freeze_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _external_raw, external, external_binding = _open_v1_custody_with_binding(module)
    _invariance_raw, invariance = _load_ordinary_json_at(
        root_fd, "scientific_invariance_receipt.json"
    )
    _nonreuse_raw, nonreuse = _load_ordinary_json_at(
        root_fd, "v1_custody_and_nonreuse.json"
    )
    _reproduction_raw, reproduction = _load_ordinary_json_at(
        root_fd, "v1_v2_first_eight_reproduction.json"
    )
    invariance = _call(module, "validate_scientific_invariance_receipt", invariance)
    nonreuse = _call(
        module,
        "validate_v1_custody_and_nonreuse",
        nonreuse,
        source_freeze_commit=source_freeze_commit,
    )
    reproduction = _call(module, "validate_first_eight_reproduction", reproduction)
    alignment = _call(module, "port_heading_implementation_alignment_authority")
    candidate_alignment = _call(
        module, "candidate_port_metric_implementation_alignment_authority"
    )
    if (
        invariance.get("inherited_implementation_alignment_disposition")
        != alignment.get("disposition")
        or invariance.get("port_heading_alignment_authority_content_digest")
        != alignment.get("content_digest")
    ):
        raise RegenerationError("scientific invariance receipt alignment binding drift")
    if (
        invariance.get("candidate_port_metric_alignment_disposition")
        != candidate_alignment.get("disposition")
        or invariance.get(
            "candidate_port_metric_alignment_authority_content_digest"
        )
        != candidate_alignment.get("content_digest")
    ):
        raise RegenerationError(
            "scientific invariance receipt candidate-port binding drift"
        )
    if nonreuse.get("external_custody_receipt_binding") != external_binding:
        raise RegenerationError("internal nonreuse receipt external binding drift")
    if nonreuse.get("v2_source_freeze_commit") != source_freeze_commit:
        raise RegenerationError("internal nonreuse receipt source-freeze binding drift")
    projection = _call(module, "v1_custody_projection", external)
    projection_sha = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    if nonreuse.get("external_custody_projection_sha256") != projection_sha:
        raise RegenerationError("internal nonreuse receipt custody digest drift")
    rows, fresh = _first_eight_reproduction_rows(material_root)
    if reproduction.get("rows") != rows:
        raise RegenerationError("first-eight receipt differs from independent NPZ replay")
    all_pass = all(row["pass"] for row in rows)
    if reproduction.get("pass") is not all_pass or reproduction.get(
        "full_collection_authorized"
    ) is not all_pass:
        raise RegenerationError("first-eight receipt disposition differs from raw replay")
    count_fields = (
        "v1_payloads_copied_into_v2",
        "v1_hardlinks_into_v2",
        "v1_shared_inodes_with_v2",
    )
    if any(
        isinstance(nonreuse.get(field), bool)
        or not isinstance(nonreuse.get(field), int)
        or nonreuse.get(field) != 0
        for field in count_fields
    ) or nonreuse.get("v1_runtime_artifact_or_shard_reused") is not False:
        raise RegenerationError("internal V1 nonreuse claims do not match live custody")
    return invariance, nonreuse, reproduction, fresh


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise RegenerationError(
            "Git source-custody check failed: "
            + exc.output.decode("utf-8", "replace")
        ) from exc


def _repo_source_binding(relative: str, *, head: str) -> dict[str, Any]:
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
    ):
        raise RegenerationError(f"unsafe source path: {relative!r}")
    live_path = REPO_ROOT / path
    if live_path.is_symlink() or live_path.resolve(strict=True) != live_path:
        raise RegenerationError(f"source path traverses a symlink: {relative}")
    raw = live_path.read_bytes()
    if raw != _git_bytes("show", f"{head}:{relative}"):
        raise RegenerationError(f"live source differs from frozen commit: {relative}")
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _observe_source_freeze(
    contract: Mapping[str, Any], module: Any
) -> dict[str, Any]:
    head = _git_bytes("rev-parse", "HEAD").decode().strip()
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("V2 reducer requires a clean source-freeze worktree")
    if contract.get("source_freeze_commit") != head:
        raise RegenerationError("runtime contract source-freeze commit differs from HEAD")
    scientific = contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("runtime contract lacks scientific contract")
    tracked = scientific.get("tracked_source_paths")
    dependencies = scientific.get("source_dependency_paths")
    wrappers = scientific.get("v2_wrapper_dependency_paths")
    if (
        not isinstance(tracked, list)
        or len(tracked) != 15
        or not isinstance(dependencies, list)
        or not isinstance(wrappers, list)
    ):
        raise RegenerationError("V2 source path authority drift")
    closure_paths = [*tracked[7:], *dependencies, *wrappers]
    if len(closure_paths) != len(set(closure_paths)):
        raise RegenerationError("V2 source closure contains duplicate paths")
    tracked_bindings = [_repo_source_binding(path, head=head) for path in tracked]
    closure_candidates = [path for path in tracked if "source_closure" in path]
    if len(closure_candidates) != 1:
        raise RegenerationError("V2 tracked source-closure document identity drift")
    closure_raw, closure = _load_canonical_json(
        REPO_ROOT / closure_candidates[0], "V2 source closure"
    )
    if set(closure) != {"schema", "parent_commit", "row_count", "rows", "content_digest"}:
        raise RegenerationError("V2 source-closure field drift")
    if (
        closure.get("schema")
        != "physical_graph_edge_handoff_qualification_v2.source_closure.v1"
        or closure.get("parent_commit") != SOURCE_PARENT_COMMIT
        or closure.get("row_count") != len(closure_paths)
    ):
        raise RegenerationError("V2 source-closure identity/count drift")
    expected_rows = [_repo_source_binding(path, head=head) for path in closure_paths]
    if closure.get("rows") != expected_rows:
        raise RegenerationError("V2 source closure differs from exact ordered live bytes")
    return {
        "head_commit": head,
        "parent_commit": _git_bytes("rev-parse", f"{head}^").decode().strip(),
        "freeze_subject": _git_bytes("show", "-s", "--format=%s", head).decode().strip(),
        "worktree_clean": True,
        "tracked_source_count": len(tracked_bindings),
        "tracked_sources_sha256": hashlib.sha256(
            canonical_json_bytes(tracked_bindings)
        ).hexdigest(),
        "source_closure_path": closure_candidates[0],
        "source_closure_bytes": len(closure_raw),
        "source_closure_sha256": hashlib.sha256(closure_raw).hexdigest(),
        "source_closure_row_count": len(expected_rows),
        "source_closure_live_bytes_exact": True,
        "metrics_module": getattr(module, "__name__", type(module).__name__),
    }


def _validate_source_freeze_observation(
    value: Mapping[str, Any], *, expected_commit: str
) -> dict[str, Any]:
    fields = {
        "head_commit",
        "parent_commit",
        "freeze_subject",
        "worktree_clean",
        "tracked_source_count",
        "tracked_sources_sha256",
        "source_closure_path",
        "source_closure_bytes",
        "source_closure_sha256",
        "source_closure_row_count",
        "source_closure_live_bytes_exact",
        "metrics_module",
    }
    if set(value) != fields:
        raise RegenerationError("V2 source-freeze observation field drift")
    if (
        value.get("head_commit") != expected_commit
        or value.get("parent_commit") != SOURCE_PARENT_COMMIT
        or value.get("freeze_subject")
        != "Freeze corrected physical graph edge handoff qualification V2"
        or value.get("worktree_clean") is not True
        or value.get("source_closure_live_bytes_exact") is not True
        or value.get("tracked_source_count") != 15
        or value.get("source_closure_row_count") != 71
    ):
        raise RegenerationError("V2 source-freeze observation failed")
    for field in ("tracked_sources_sha256", "source_closure_sha256"):
        V1._sha256(value.get(field), f"source freeze {field}")
    return json.loads(json.dumps(value))


def _validate_success_inventory(root_fd: int) -> list[str]:
    names = sorted(os.listdir(root_fd))
    scientific = set(V2_SUCCESS_FILES) - set(V1.PUBLICATION_FILES)
    publication = set(V1.PUBLICATION_FILES)
    observed = set(names)
    if observed not in (scientific, scientific | publication):
        raise RegenerationError(
            "successful V2 root must contain exactly 23 scientific or 26 complete leaves"
        )
    for name in names:
        CUSTODY._safe_leaf(name)
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegenerationError(f"V2 official leaf is not single-link regular: {name}")
    return names


def _validate_terminal_inventory(root_fd: int) -> list[str]:
    names = sorted(os.listdir(root_fd))
    if names != sorted(V2_REPRODUCTION_FAILURE_FILES):
        raise RegenerationError("V2 reproduction-mismatch root inventory drift")
    for name in names:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegenerationError(f"V2 terminal leaf is not single-link regular: {name}")
    return names


def _scientific_input_bindings(
    raw_documents: Mapping[str, bytes],
    raw_ledgers: Mapping[str, bytes],
    npz_inspections: Mapping[str, Mapping[str, Any]],
    ordinary_receipts: Mapping[str, bytes],
) -> dict[str, dict[str, Any]]:
    bindings = {
        leaf: CUSTODY._binding(leaf, raw)
        for leaf, raw in {**raw_documents, **raw_ledgers, **ordinary_receipts}.items()
    }
    for leaf, inspection in npz_inspections.items():
        bindings[leaf] = {
            field: inspection[field] for field in ("path", "bytes", "sha256")
        }
    expected = set(V2_SUCCESS_FILES) - set(V1.PUBLICATION_FILES)
    if set(bindings) != expected:
        raise RegenerationError("V2 scientific binding inventory drift")
    return bindings


def _validate_official_previous_command_raw_hashes(
    *,
    root: Path,
    state_snapshot_index: Mapping[str, Any],
    module: Any,
) -> dict[str, Any]:
    """Reopen the assembled NPZ and bind the corrected named field to raw rows."""

    path = root / "state_snapshots.npz"
    before_sha = _sha256_file(path)
    members = _read_npz_members(
        path,
        "V2 official state_snapshots.npz raw previous-command rows",
        expected_names=None,
    )
    member = members.get("previous_applied_command.npy")
    if (
        member is None
        or member.get("dtype") != "<f8"
        or member.get("shape") != [64, 3]
        or len(member.get("payload", b"")) != 64 * 3 * 8
    ):
        raise RegenerationError(
            "official previous-command member dtype/shape/payload drift"
        )
    payload = member["payload"]
    raw_row_sha256s = [
        hashlib.sha256(payload[index * 24 : (index + 1) * 24]).hexdigest()
        for index in range(64)
    ]
    validated = _call(
        module,
        "validate_state_snapshot_previous_applied_command_hashes",
        state_snapshot_index,
        raw_row_sha256s,
    )
    if validated != raw_row_sha256s or _sha256_file(path) != before_sha:
        raise RegenerationError("official previous-command raw-row validation drift")
    return {
        "member": "previous_applied_command",
        "dtype_str": "<f8",
        "shape": [64, 3],
        "row_shape": [3],
        "row_count": 64,
        "digest_domain": "sha256 of exact C-contiguous row bytes only",
        "all_index_rows_equal_reopened_raw_row_sha256": True,
        "raw_row_sha256s_digest": hashlib.sha256(
            canonical_json_bytes(raw_row_sha256s)
        ).hexdigest(),
    }


def _validate_candidate_port_metric_alignment_from_raw(
    *,
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    material_root: Path,
    module: Any,
) -> dict[str, Any]:
    """Recompute the three aligned fields from all retained 750-sample traces."""

    panel_rows = documents["panel_manifest"]["states"]
    snapshot_payloads = captured.get("state_snapshots.npz", {}).get(
        "base_pose_world"
    )
    trace_payloads = captured.get("candidate_traces.npz", {}).get(
        "base_pose_world"
    )
    if (
        not isinstance(snapshot_payloads, list)
        or len(snapshot_payloads) != 64
        or not isinstance(trace_payloads, list)
        or len(trace_payloads) != 960
    ):
        raise RegenerationError("candidate-port raw pose capture cardinality drift")
    initial_pose_by_state = {
        row["state_id"]: V1._float64_rows(
            snapshot_payloads[index], 7, 1, f"candidate alignment snapshot[{index}]"
        )[0]
        for index, row in enumerate(panel_rows)
    }
    port_by_state = {
        row["state_id"]: row["directed_port_world"]
        for row in documents["edge_port_index"]["records"]
    }
    fields = ("port_progress_m", "lateral_error_m", "positive_port_progress")
    projection_rows: list[dict[str, Any]] = []

    def exact_metrics(state_id: str, trace_index: int) -> dict[str, Any]:
        poses = V1._float64_rows(
            trace_payloads[trace_index],
            7,
            750,
            f"candidate alignment trace[{trace_index}]",
        )
        result = _call(
            module,
            "derive_candidate_port_metrics",
            initial_pose_by_state[state_id],
            poses,
            port_by_state[state_id],
        )
        if not isinstance(result, Mapping) or set(result) != set(fields):
            raise RegenerationError("candidate-port pure formula projection drift")
        return dict(result)

    fanout_start = 128
    fanout_material: dict[str, list[dict[str, Any]]] = {}
    for state in panel_rows:
        state_id = str(state["state_id"])
        _raw, metadata = _load_canonical_json(
            material_root / "fanout" / state_id / "metadata.json",
            f"V2 fanout material {state_id}",
        )
        outcome_rows = metadata.get("outcome_rows")
        if not isinstance(outcome_rows, list) or len(outcome_rows) != 12:
            raise RegenerationError("fanout material outcome coverage drift")
        fanout_material[state_id] = [dict(row) for row in outcome_rows]
    for row_index, row in enumerate(ledgers["candidate_fanout"]):
        state_id = str(row["state_id"])
        candidate_index = int(row["candidate_index"])
        expected = exact_metrics(state_id, fanout_start + row_index)
        material_row = fanout_material[state_id][candidate_index]
        for field in fields:
            if row.get(field) != expected[field] or material_row.get(field) != expected[field]:
                raise RegenerationError(
                    f"fanout[{row_index}] {field} differs from raw canonical-port formula"
                )
        projection_rows.append(
            {
                "trace_index": fanout_start + row_index,
                "state_id": state_id,
                "candidate_index": candidate_index,
                **expected,
            }
        )

    repeat_material: dict[str, list[dict[str, Any]]] = {}
    for state in panel_rows:
        if state.get("role") != "DEVELOPMENT_HELDOUT":
            continue
        state_id = str(state["state_id"])
        _raw, metadata = _load_canonical_json(
            material_root / "repeat" / state_id / "metadata.json",
            f"V2 repeat material {state_id}",
        )
        outcome_rows = metadata.get("outcome_rows")
        if not isinstance(outcome_rows, list) or len(outcome_rows) != 4:
            raise RegenerationError("repeat material outcome coverage drift")
        repeat_material[state_id] = [dict(row) for row in outcome_rows]
    repeat_offsets: dict[str, int] = {}
    for row in ledgers["repeated_execution"]:
        state_id = str(row["state_id"])
        trace_index = int(row["trace_index"])
        material_index = repeat_offsets.get(state_id, 0)
        repeat_offsets[state_id] = material_index + 1
        expected = exact_metrics(state_id, trace_index)
        material_row = repeat_material[state_id][material_index]
        for field in fields:
            if material_row.get(field) != expected[field]:
                raise RegenerationError(
                    f"repeat trace {trace_index} {field} differs from raw canonical-port formula"
                )
        projection_rows.append(
            {
                "trace_index": trace_index,
                "state_id": state_id,
                "source_candidate_index": int(row["source_candidate_index"]),
                **expected,
            }
        )
    if len(projection_rows) != 832 or any(value != 4 for value in repeat_offsets.values()):
        raise RegenerationError("candidate-port aligned trace coverage drift")
    return {
        "fanout_trace_count": 768,
        "repeat_trace_count": 64,
        "physics_samples_per_trace": 750,
        "changed_output_fields": list(fields),
        "canonical_port_reference": "edge_port_index.directed_port_world",
        "all_material_and_official_values_exact": True,
        "projection_sha256": hashlib.sha256(
            canonical_json_bytes(projection_rows)
        ).hexdigest(),
    }


def _validate_official_document_identities(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require V2 identity on every official structured document root."""

    schema_token = "physical_graph_edge_handoff_qualification_v2"
    checked: list[str] = []
    for label, value in documents.items():
        schema = value.get("schema")
        experiment = value.get("experiment_id")
        if (
            not isinstance(schema, str)
            or schema_token not in schema
            or experiment != EXPERIMENT_ID
        ):
            raise RegenerationError(f"official {label} root does not use V2 identity")
        checked.append(label)
    return {
        "checked_document_count": len(checked),
        "checked_documents": checked,
        "official_experiment_id": EXPERIMENT_ID,
        "official_schema_token": schema_token,
        "all_official_document_roots_use_v2_identity": True,
    }


def _validate_publication(
    *,
    root: Path,
    root_fd: int,
    inventory: Sequence[str],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    expected_receipt_sha256: str,
) -> dict[str, Any] | None:
    if not set(V1.PUBLICATION_FILES).issubset(inventory):
        return None
    result_raw, result = V1._load_json_at(root_fd, "result.json")
    report_raw = CUSTODY._read_regular_at(root_fd, "result.md")
    manifest_raw, manifest = V1._load_json_at(root_fd, "file_hashes.json")
    assert report_raw is not None
    try:
        report_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError("V2 result.md is not UTF-8") from exc
    if not report_raw or not report_raw.endswith(b"\n"):
        raise RegenerationError("V2 result.md must be nonempty and LF terminated")
    expected_manifest_rows = [
        dict(scientific_bindings[name]) for name in sorted(scientific_bindings)
    ] + [
        CUSTODY._binding("result.json", result_raw),
        CUSTODY._binding("result.md", report_raw),
    ]
    expected_manifest_rows.sort(key=lambda row: row["path"])
    if (
        manifest.get("schema")
        != "physical_graph_edge_handoff_qualification_v2.file_hashes.v1"
        or manifest.get("root") != str(root)
        or manifest.get("files") != expected_manifest_rows
        or manifest.get("file_count_excluding_self") != 25
        or manifest.get("bytes_excluding_self")
        != sum(row["bytes"] for row in expected_manifest_rows)
        or manifest.get("file_hashes_self_sha256_excluded") is not True
    ):
        raise RegenerationError("V2 file_hashes.json differs from exact live bytes")
    scientific = runtime_contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V2 result lacks scientific-contract binding")
    context = scientific.get("v2_context_binding")
    if not isinstance(context, Mapping):
        raise RegenerationError("V2 result lacks predecessor context")
    expected_projection = {
        "schema": "physical_graph_edge_handoff_qualification_v2.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": runtime_contract.get("source_freeze_commit"),
        "source_baseline_commit": runtime_contract.get("source_baseline_commit"),
        "predecessor_result_commit": context.get("result_commit"),
        "development_only": True,
        "primary_classification": recomputed["primary_classification"],
        "secondary_classifications": recomputed["secondary_classifications"],
        "next_experiment": recomputed["next_experiment"],
        "selected_target_id": recomputed["development"]["selected_target_id"],
        "evidence_counts": recomputed["evidence_counts"],
        "panel_metrics": recomputed["panel"],
        "development_metrics": recomputed["development"],
        "heldout_metrics": recomputed["heldout"],
        "repeatability": recomputed["repeatability"],
        "command_tracking": recomputed["command_tracking"],
        "runtime_environments": recomputed["runtime_environments"],
        "stratified_metrics": recomputed["stratified"],
        "gate": recomputed["gate"],
        "component_failures": recomputed["component_failures"],
        "metrics_sha256": scientific_bindings["metrics.json"]["sha256"],
        "independent_reducer_receipt_sha256": expected_receipt_sha256,
        "scientific_storage_bytes": sum(
            int(binding["bytes"]) for binding in scientific_bindings.values()
        ),
        "models_trained": 0,
        "prohibited_components_trained_or_implemented": [],
    }
    for field, expected in expected_projection.items():
        if result.get(field) != expected:
            raise RegenerationError(f"V2 result.json {field} drift")
    allowed_fields = {
        *expected_projection,
        "runtime_seconds",
        "content_digest",
    }
    if set(result) != allowed_fields:
        raise RegenerationError("V2 result.json field-set drift")
    runtime_seconds = result.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) < 0
    ):
        raise RegenerationError("V2 result runtime is invalid")

    target_lines = "\n".join(
        f"- {row['target_id']}: {json.dumps(row, sort_keys=True)}"
        for row in recomputed["development"]["target_summaries"]
    )
    condition_lines = "\n".join(
        f"- {row['condition_id']}: {json.dumps(row, sort_keys=True)}"
        for row in recomputed["heldout"]["condition_summaries"]
    )
    expected_report = (
        f"# {EXPERIMENT_ID}\n\n"
        f"Primary classification: {recomputed['primary_classification']}\n\n"
        "Secondary classifications: "
        f"{', '.join(recomputed['secondary_classifications']) or 'none'}\n\n"
        f"Selected target: {recomputed['development']['selected_target_id']}\n\n"
        f"Handoff gate passed: {str(bool(recomputed['gate']['passed'])).lower()}\n\n"
        f"Next experiment: {recomputed['next_experiment']}\n\n"
        "## Physical evidence\n\n"
        f"Evidence counts: {json.dumps(recomputed['evidence_counts'], sort_keys=True)}\n\n"
        "All 64 selected decision states retained exact serialized solver, controller, "
        "command-history, and RNG snapshots. Each passed two independent 750-sample "
        "serialized-restore fixtures. All 256 prospective teacher traces, 64 actual "
        "teacher-derived directed ports, 768 candidate fanouts, and 64 repeat traces "
        "are persisted at the 2 ms physics rate.\n\n"
        f"Held-out candidate coverage: {json.dumps(recomputed['panel'], sort_keys=True)}\n\n"
        "## Development target selection\n\n"
        f"{target_lines}\n\n"
        "## Held-out conditions\n\n"
        f"{condition_lines}\n\n"
        "## Reset, repeat, and controller qualification\n\n"
        f"Repeatability: {json.dumps(recomputed['repeatability'], sort_keys=True)}\n\n"
        f"Command tracking: {json.dumps(recomputed['command_tracking'], sort_keys=True)}\n\n"
        "## Runtime environments\n\n"
        f"{json.dumps(recomputed['runtime_environments'], sort_keys=True)}\n\n"
        f"Stratified outcomes: {json.dumps(recomputed['stratified'], sort_keys=True)}\n\n"
        f"Runtime seconds: {float(runtime_seconds):.6f}; scientific storage bytes: "
        f"{result['scientific_storage_bytes']}.\n\n"
        "## Claims boundary\n\n"
        "This is a development-only physical graph-edge handoff qualification. "
        "No model was trained. The frozen V-JEPA encoder and frozen current-visual "
        "ranker were used only for registered inference. No predictor, new local "
        "ranker, memory model, safety model, novelty mechanism, beacon discovery, "
        "or online graph-construction model was trained or implemented.\n"
    ).encode("utf-8")
    if report_raw != expected_report:
        raise RegenerationError("V2 result.md differs from exact reduced presentation")
    return result


def build_regeneration_receipt(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
) -> dict[str, Any]:
    """Rebuild all V2 custody and metrics without executing science."""

    module = _load_metrics_module() if metrics_module is None else metrics_module
    authority = _validate_v2_authority(module)
    root, root_fd = V1._open_root(output_root, "V2 official root")
    material = (
        Path(material_root)
        if material_root is not None
        else root.parent / f"{root.name}_material"
    )
    try:
        observed_names = sorted(os.listdir(root_fd))
        terminal = set(observed_names) == set(V2_REPRODUCTION_FAILURE_FILES)
        inventory = (
            _validate_terminal_inventory(root_fd)
            if terminal
            else _validate_success_inventory(root_fd)
        )
        contract_raw, contract = V1._load_json_at(root_fd, "contract.json")
        source_commit = V1._contract_source_commit(contract)
        observed_source = (
            _observe_source_freeze(contract, module)
            if source_freeze_observation is None
            else dict(source_freeze_observation)
        )
        source_freeze = _validate_source_freeze_observation(
            observed_source, expected_commit=source_commit
        )
        ordinary_raw: dict[str, bytes] = {}
        for leaf in V2_ADDITIONAL_FILES:
            raw, _value = _load_ordinary_json_at(root_fd, leaf)
            ordinary_raw[leaf] = raw
        invariance, nonreuse, reproduction, fresh = _validate_internal_gates(
            module=module,
            root_fd=root_fd,
            material_root=material,
            source_freeze_commit=source_commit,
        )
        alignment_authority = _call(
            module, "port_heading_implementation_alignment_authority"
        )
        alignment_binding = {
            "disposition": alignment_authority["disposition"],
            "authority_content_digest": alignment_authority["content_digest"],
            "changed_output_fields": alignment_authority["changed_output_fields"],
            "exact_formula": alignment_authority["exact_formula"],
            "frozen_metric_gate_or_threshold_relaxed": False,
            "outcome_values_used_to_choose_formula_or_parameter": False,
        }
        candidate_alignment_authority = _call(
            module, "candidate_port_metric_implementation_alignment_authority"
        )
        candidate_alignment_binding = {
            "disposition": candidate_alignment_authority["disposition"],
            "authority_content_digest": candidate_alignment_authority[
                "content_digest"
            ],
            "changed_output_fields": candidate_alignment_authority[
                "changed_output_fields"
            ],
            "exact_formulas": candidate_alignment_authority["exact_formulas"],
            "frozen_metric_gate_or_threshold_relaxed": False,
            "outcome_values_used_to_choose_formula_or_parameter": False,
        }
        if invariance.get("pass") is not True or nonreuse.get("pass") is not True:
            raise RegenerationError("V2 pre-scientific invariance/nonreuse gate failed")

        counters = {
            "checkpoint_deserializations": 0,
            "encoder_initializations": 0,
            "model_initializations": 0,
            "ranker_inference_calls": 0,
            "training_steps": 0,
            "simulator_initializations": 0,
            "simulator_steps": 0,
            "physics_steps": 0,
            "reset_calls": 0,
            "render_calls": 0,
        }
        if terminal:
            if reproduction.get("pass") is not False or reproduction.get(
                "full_collection_authorized"
            ) is not False or reproduction.get("status") != (
                "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
            ):
                raise RegenerationError("V2 terminal root lacks a raw-proven mismatch")
            material_validation = _validate_all_v2_material_payloads(
                material,
                expected_shard_count=8,
                expected_metadata_paths=[
                    f"qualification/pool-{index:03d}/metadata.json"
                    for index in range(8)
                ],
            )
            identity_validation = _validate_official_document_identities(
                {"contract.json": contract}
            )
            receipt = {
                "schema": REGENERATION_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "pass": True,
                "mode": "INDEPENDENT_TECHNICAL_REPRODUCTION_TERMINAL_VALIDATION_ONLY",
                "technical_disposition": "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH",
                "scientific_result_produced": False,
                "reducer_authority_digest": authority["content_digest"],
                "source_freeze": source_freeze,
                "official_root_inventory": {
                    "leaf_count": 4,
                    "leaves": inventory,
                    "exact_terminal_inventory_validated": True,
                },
                "inputs": {
                    "contract.json": CUSTODY._binding("contract.json", contract_raw),
                    **{
                        leaf: CUSTODY._binding(leaf, ordinary_raw[leaf])
                        for leaf in V2_ADDITIONAL_FILES
                    },
                },
                "v1_custody_receipt_binding": dict(
                    authority["v1_custody_and_nonreuse_authority"]["external_receipt"]
                ),
                "scientific_invariance_receipt_validated": True,
                "v1_custody_and_nonreuse_validated": True,
                "first_eight_reproduction": reproduction,
                "first_eight_live_custody": fresh,
                "material_persisted_array_validation": material_validation,
                "official_identity_validation": identity_validation,
                "port_heading_implementation_alignment": alignment_binding,
                "candidate_port_metric_implementation_alignment": (
                    candidate_alignment_binding
                ),
                "no_panel_split_candidate_metric_result_or_manifest_opened": True,
                "scientific_execution_counters": counters,
            }
            V1._validate_no_self_digest(receipt, "V2 external terminal receipt")
            V1._reject_nonfinite(receipt, label="V2 external terminal receipt")
            return receipt

        if reproduction.get("pass") is not True or _call(
            module, "authorizes_full_v2_collection", reproduction
        ) is not True:
            raise RegenerationError("successful V2 root lacks the first-eight pass gate")

        raw_documents: dict[str, bytes] = {"contract.json": contract_raw}
        documents: dict[str, dict[str, Any]] = {"contract.json": contract}
        for leaf in V1.DOCUMENT_FILES:
            if leaf == "contract.json":
                continue
            raw, document = V1._load_json_at(root_fd, leaf)
            raw_documents[leaf] = raw
            documents[leaf] = document
        identity_validation = _validate_official_document_identities(documents)
        raw_ledgers: dict[str, bytes] = {}
        ledgers: dict[str, list[dict[str, Any]]] = {}
        for leaf in V1.LEDGER_FILES:
            raw, rows = V1._load_jsonl_at(root_fd, leaf)
            raw_ledgers[leaf] = raw
            ledgers[leaf] = rows
        document_name_to_leaf = {
            "panel_manifest": "panel_manifest.json",
            "split_manifest": "split_manifest.json",
            "graph_manifest": "graph_manifest.json",
            "state_snapshot_index": "state_snapshot_index.json",
            "teacher_trace_index": "teacher_trace_index.json",
            "edge_port_index": "edge_port_index.json",
            "waypoint_contracts": "waypoint_contracts.json",
            "pixel_index": "pixel_index.json",
            "latent_index": "latent_index.json",
            "development_target_selection": "development_target_selection.json",
        }
        evidence_documents = {
            name: documents[leaf] for name, leaf in document_name_to_leaf.items()
        }
        document_validation = V1._validate_document_authorities(
            evidence_documents, authority
        )
        ledger_name_to_leaf = {
            "candidate_fanout": "candidate_fanout.jsonl",
            "heldout_ranker_scores": "heldout_ranker_scores.jsonl",
            "repeated_execution": "repeated_execution.jsonl",
        }
        evidence_ledgers = {
            name: ledgers[leaf] for name, leaf in ledger_name_to_leaf.items()
        }
        ledger_validation = V1._validate_ledger_authorities(
            evidence_ledgers, evidence_documents, authority
        )
        runtime_validation = V1._validate_runtime_environments(
            module, evidence_documents, evidence_ledgers
        )
        encoder_source_validation = V1._validate_external_encoder_source(
            evidence_documents["latent_index"]
        )

        symbols: dict[str, int] = {}
        captured_reset: dict[str, list[bytes]] = {}
        captured_physical: dict[str, dict[str, list[bytes]]] = {}
        npz_inspections = {
            leaf: V1._inspect_npz_at(
                root_fd,
                leaf,
                authority["npz_authorities"][leaf],
                symbols=symbols,
                captured_reset_slices=captured_reset,
                captured_physical_slices=captured_physical,
            )
            for leaf in V1.PAYLOAD_FILES
        }
        if symbols.get("U") != evidence_documents["pixel_index"].get(
            "unique_pixel_count"
        ):
            raise RegenerationError("V2 latent U dimension differs from pixel index")
        pure_npz = _call(
            module,
            "validate_npz_inspections",
            [npz_inspections[leaf] for leaf in V1.PAYLOAD_FILES],
        )
        if not isinstance(pure_npz, Mapping) or set(pure_npz) != set(V1.PAYLOAD_FILES):
            raise RegenerationError("V2 pure NPZ projection drift")
        previous_command_raw_validation = (
            _validate_official_previous_command_raw_hashes(
                root=root,
                state_snapshot_index=evidence_documents["state_snapshot_index"],
                module=module,
            )
        )
        # The inherited V1 cross-link helper uses its legacy header+NUL digest
        # only in this one validation slot.  Feed it a private projection from
        # the already-inspected canonical rows after independently proving the
        # live V2 index contains raw-C row hashes.  No persisted evidence is
        # changed or trusted through this bridge.
        legacy_crosslink_documents = json.loads(json.dumps(evidence_documents))
        legacy_previous_rows = npz_inspections["state_snapshots.npz"]["members"][
            "previous_applied_command"
        ]["row_or_slice_sha256s"]
        legacy_snapshot_records = legacy_crosslink_documents[
            "state_snapshot_index"
        ]["records"]
        if len(legacy_previous_rows) != len(legacy_snapshot_records):
            raise RegenerationError("legacy previous-command bridge cardinality drift")
        for record, legacy_sha in zip(legacy_snapshot_records, legacy_previous_rows):
            record["previous_applied_command_sha256"] = legacy_sha
        npz_cross_links = V1._validate_npz_cross_links(
            legacy_crosslink_documents, evidence_ledgers, npz_inspections
        )
        reset_validation = V1._validate_reset_trace_pairs(
            evidence_documents["state_snapshot_index"],
            npz_inspections["candidate_traces.npz"],
            captured_reset,
            authority,
        )
        physical_authority = authority["physical_trace_reduction_authority"]
        raw_teacher = V1._validate_raw_teacher_evidence(
            evidence_documents,
            npz_inspections,
            captured_physical,
            physical_authority,
        )
        raw_candidate = V1._validate_raw_candidate_and_repeat_evidence(
            evidence_documents,
            evidence_ledgers,
            npz_inspections,
            captured_physical,
            physical_authority,
        )
        candidate_port_alignment_validation = (
            _validate_candidate_port_metric_alignment_from_raw(
                documents=evidence_documents,
                ledgers=evidence_ledgers,
                captured=captured_physical,
                material_root=material,
                module=module,
            )
        )
        external_bindings = _call(module, "external_artifact_bindings", contract)
        external_validations = [
            V1._stream_external_binding(binding, f"external artifact {index}")
            for index, binding in enumerate(external_bindings)
        ]
        if [row["role"] for row in external_validations] != authority[
            "external_artifact_roles"
        ]:
            raise RegenerationError("V2 external artifact role order drift")
        predecessor_validation = V1._validate_v2_context(contract, module)
        evidence = {
            **evidence_documents,
            **evidence_ledgers,
            "npz_inspections": [npz_inspections[leaf] for leaf in V1.PAYLOAD_FILES],
        }
        if set(evidence) != set(authority["evidence_keys"]):
            raise RegenerationError("V2 assembled scientific evidence key drift")
        recomputed = _call(module, "recompute_metrics", evidence)
        recomputed_raw, recomputed = V1._canonicalise_object(
            dict(recomputed), "V2 recomputed metrics"
        )
        V1._validate_content_digest(recomputed, "V2 recomputed metrics")
        if raw_documents["metrics.json"] != recomputed_raw:
            raise RegenerationError("V2 persisted metrics differ from exact raw-ledger rebuild")

        scientific_bindings = _scientific_input_bindings(
            raw_documents, raw_ledgers, npz_inspections, ordinary_raw
        )
        panel_states = evidence_documents["panel_manifest"].get("states")
        if not isinstance(panel_states, list):
            raise RegenerationError("V2 panel state rows are absent")
        selected_state_ids = [str(row["state_id"]) for row in panel_states]
        heldout_state_ids = [
            str(row["state_id"])
            for row in panel_states
            if row.get("role") == "DEVELOPMENT_HELDOUT"
        ]
        expected_material_metadata = [
            *(f"qualification/pool-{index:03d}/metadata.json" for index in range(256)),
            *(f"selected/{state_id}/metadata.json" for state_id in selected_state_ids),
            *(f"fanout/{state_id}/metadata.json" for state_id in selected_state_ids),
            *(f"repeat/{state_id}/metadata.json" for state_id in heldout_state_ids),
        ]
        material_validation = _validate_all_v2_material_payloads(
            material,
            expected_shard_count=400,
            expected_metadata_paths=expected_material_metadata,
        )
        receipt = {
            "schema": REGENERATION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "pass": True,
            "mode": "INDEPENDENT_PERSISTED_PHYSICAL_EVIDENCE_REDUCTION_ONLY",
            "reducer_authority_digest": authority["content_digest"],
            "source_freeze": source_freeze,
            "official_root_inventory": {
                "scientific_leaf_count": 23,
                "allowed_leaf_count": 26,
                "allowed_leaf_names_validated": True,
                "publication_all_or_absent_validated": True,
            },
            "inputs": {
                name: scientific_bindings[name] for name in sorted(scientific_bindings)
            },
            "v1_custody_receipt_binding": dict(
                authority["v1_custody_and_nonreuse_authority"]["external_receipt"]
            ),
            "scientific_invariance_receipt_validated": True,
            "v1_custody_and_nonreuse_validated": True,
            "first_eight_reproduction": reproduction,
            "first_eight_live_custody": fresh,
            "material_persisted_array_validation": material_validation,
            "official_identity_validation": identity_validation,
            "document_validation": document_validation,
            "ledger_validation": ledger_validation,
            "npz_validation": V1._compact_npz_validation(npz_inspections),
            "npz_symbols": {name: symbols[name] for name in sorted(symbols)},
            "pure_npz_validation_passed": True,
            "npz_cross_link_validation": npz_cross_links,
            "official_previous_command_raw_hash_validation": (
                previous_command_raw_validation
            ),
            "reset_pair_validation": reset_validation,
            "raw_teacher_physics_validation": raw_teacher,
            "raw_candidate_and_repeat_physics_validation": raw_candidate,
            "port_heading_implementation_alignment": {
                **alignment_binding,
                "all_edge_port_rows_independently_reduced_against_frozen_geometry": True,
                "crossing_velocity_and_heading_independently_reduced_from_raw_traces": True,
            },
            "candidate_port_metric_implementation_alignment": {
                **candidate_alignment_binding,
                "all_768_fanout_and_64_repeat_rows_independently_reduced": True,
                "raw_pose_samples_per_trace": 750,
                "canonical_port_from_edge_port_index_used": True,
                "every_other_candidate_outcome_field_independently_validated": True,
                "raw_alignment_validation": candidate_port_alignment_validation,
            },
            "predecessor_context_validation": predecessor_validation,
            "external_artifact_validation": external_validations,
            "external_encoder_source_validation": encoder_source_validation,
            "runtime_environment_validation": runtime_validation,
            "metrics_exact_byte_equal": True,
            "recomputed_metrics_sha256": hashlib.sha256(recomputed_raw).hexdigest(),
            "scientific_execution_counters": counters,
        }
        V1._validate_no_self_digest(receipt, "V2 external regeneration receipt")
        V1._reject_nonfinite(receipt, label="V2 external regeneration receipt")
        receipt_sha = hashlib.sha256(canonical_document_bytes(receipt)).hexdigest()
        _validate_publication(
            root=root,
            root_fd=root_fd,
            inventory=inventory,
            scientific_bindings=scientific_bindings,
            recomputed=recomputed,
            runtime_contract=contract,
            expected_receipt_sha256=receipt_sha,
        )
        return receipt
    finally:
        os.close(root_fd)


def verify_and_emit(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
    )
    _emit_external((Path(output_root),), output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
) -> dict[str, Any]:
    raw, supplied = _load_canonical_json(Path(output), "V2 external regeneration receipt")
    V1._validate_no_self_digest(supplied, "V2 external regeneration receipt")
    rebuilt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
    )
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("V2 external regeneration receipt differs from exact rebuild")
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("emit-v1-custody", "validate-v1-custody", "reduce-v2", "validate-v2"),
        default="reduce-v2",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.mode == "emit-v1-custody":
        output = DEFAULT_V1_CUSTODY_RECEIPT if arguments.output is None else arguments.output
        receipt = build_v1_custody_receipt()
        raw = emit_v1_custody_receipt(output, receipt=receipt)
    elif arguments.mode == "validate-v1-custody":
        output = DEFAULT_V1_CUSTODY_RECEIPT if arguments.output is None else arguments.output
        receipt = validate_existing_v1_custody_receipt(output)
        raw = canonical_document_bytes(receipt)
    elif arguments.mode == "validate-v2":
        output = DEFAULT_EXTERNAL_RECEIPT if arguments.output is None else arguments.output
        receipt = validate_existing_regeneration_receipt(arguments.output_root, output)
        raw = canonical_document_bytes(receipt)
    else:
        output = DEFAULT_EXTERNAL_RECEIPT if arguments.output is None else arguments.output
        receipt = verify_and_emit(arguments.output_root, output)
        raw = canonical_document_bytes(receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str(output),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
