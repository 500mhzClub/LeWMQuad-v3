#!/usr/bin/env python3
"""Independent persisted-evidence reducer for canonical topological-belief V2.

The reducer is intentionally system-Python and torch free.  It imports the
new pure V2 metrics module for scientific formulas and the tracked V1 reducer
only for already-audited canonical-JSON, dirfd, and row-evidence helpers.  It
never initializes an encoder, model, simulator, or training runtime.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import io
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_occluded_goal_topological_belief_v1 as V1_REDUCER


EXPERIMENT_ID = "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2"
REDUCER_SOURCE_PATH = "scripts/evaluate_occluded_goal_topological_belief_v2.py"
METRICS_SOURCE_PATH = "lewm/safety/occluded_goal_topological_belief_metrics_v2.py"
V1_REDUCER_SOURCE_PATH = "scripts/evaluate_occluded_goal_topological_belief_v1.py"
METRICS_MODULE = "lewm.safety.occluded_goal_topological_belief_metrics_v2"

DEFAULT_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v2"
)
DEFAULT_EXTERNAL_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v2_regeneration_receipt.json"
)

CONTRACT_FILE = "contract.json"
V1_REUSABLE_FILES = (
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "query_ledger.jsonl",
    "keyframe_index.json",
    "observations.npz",
)
CACHE_JSON_FILES = (
    "pixel_index.json",
    "template_to_pixel_index.json",
    "occurrence_index.json",
    "canonical_latent_index.json",
    "canonical_descriptor_index.json",
    "encoding_determinism_receipt.json",
    "cache_integrity_receipt.json",
)
CACHE_NPZ_FILES = (
    "canonical_tokens.npz",
    "canonical_descriptors.npz",
)
GRAPH_FILE = "graph_manifest.json"
QUERY_FILE = "query_ledger.jsonl"
CALIBRATION_FILE = "calibration.json"
STAGE_A_BELIEFS_FILE = "stage_a_beliefs.jsonl"
STAGE_A_METRICS_FILE = "stage_a_metrics.json"
STAGE_B_TRACE_FILE = "stage_b_trace.jsonl"
STAGE_B_METRICS_FILE = "stage_b_metrics.json"
FINAL_PUBLICATION_FILES = ("result.json", "result.md", "file_hashes.json")

RECEIPT_SCHEMA = "occluded_goal_topological_belief_v2.regeneration_receipt.v1"
EXPECTED_TEMPLATE_COUNT = 375
EXPECTED_UNIQUE_PIXEL_COUNT = 157
EXPECTED_REUSE_COUNT = 218
EXPECTED_OCCURRENCE_COUNT = 33384
EXPECTED_MULTI_TEMPLATE_GROUP_COUNT = 76
EXPECTED_TOKEN_SHAPE = (768, 1024)
EXPECTED_QUERY_COUNT = 768
EXPECTED_HELDOUT_QUERY_COUNT = 128
EXPECTED_QUERY_ROLE_COUNTS = {
    "FIT": 512,
    "CALIBRATION": 128,
    "DEVELOPMENT_HELDOUT": 128,
}
EXPECTED_CONDITION_IDS = (
    "CURRENT_FRAME_NEAREST_NODE",
    "FIXED_WINDOW_SEQUENCE",
    "MAP_FILTER",
    "TOP_K_BELIEF",
    "FULL_BELIEF",
    "ORACLE_PLACE_IDENTITY",
    "NO_ACTION_CONSISTENCY",
    "SHUFFLED_ACTION_HISTORY",
    "NO_OBSERVATION_LIKELIHOOD",
)

RegenerationError = V1_REDUCER.RegenerationError
canonical_json_bytes = V1_REDUCER.canonical_json_bytes
canonical_document_bytes = V1_REDUCER.canonical_document_bytes
canonical_digest = V1_REDUCER.canonical_digest
parse_canonical_json = V1_REDUCER.parse_canonical_json
_parse_jsonl = V1_REDUCER._parse_jsonl
_read_regular_at = V1_REDUCER._read_regular_at
_binding = V1_REDUCER._binding
_safe_leaf = V1_REDUCER._safe_leaf
_reject_nonfinite = V1_REDUCER._reject_nonfinite


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegenerationError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RegenerationError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RegenerationError(f"{label} must be lowercase SHA-256")
    return text


def _call(module: Any, name: str, *arguments: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"V2 metrics module lacks required pure API: {name}")
    try:
        return function(*arguments)
    except RegenerationError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise RegenerationError(f"V2 pure metrics API rejected evidence: {name}") from exc


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _metrics_module_name(module: Any) -> str:
    name = getattr(module, "__name__", None)
    return name if isinstance(name, str) and name else module.__class__.__name__


def _validate_content_digest(value: Mapping[str, Any], label: str) -> None:
    if "content_digest" not in value:
        raise RegenerationError(f"{label} lacks content_digest")
    core = dict(value)
    declared = core.pop("content_digest")
    if declared != canonical_digest(core):
        raise RegenerationError(f"{label} content digest drift")


def _validate_no_self_digest(value: Mapping[str, Any], label: str) -> None:
    if "content_digest" in value or "self_digest" in value or "document_sha256" in value:
        raise RegenerationError(f"{label} must not contain a self digest")


def _regular_binding_at(root_fd: int, leaf: str) -> dict[str, Any]:
    raw = _read_regular_at(root_fd, leaf)
    assert raw is not None
    return _binding(leaf, raw)


def _stream_binding_at(root_fd: int, leaf: str) -> dict[str, Any]:
    """Hash a potentially large ordinary leaf without retaining its payload."""

    _safe_leaf(leaf)
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as exc:
        raise RegenerationError(f"large reducer input cannot be opened safely: {leaf}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegenerationError(f"large reducer input is not a single-link file: {leaf}")
        digest = hashlib.sha256()
        count = 0
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            count += len(block)
        after = os.fstat(fd)
        projection = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
            row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
        )
        if projection(before) != projection(after) or count != before.st_size:
            raise RegenerationError(f"large reducer input changed while read: {leaf}")
        return {"path": leaf, "bytes": count, "sha256": digest.hexdigest()}
    finally:
        os.close(fd)


def _npy_header(stream: BinaryIO, label: str) -> tuple[dict[str, Any], int]:
    if stream.read(6) != b"\x93NUMPY":
        raise RegenerationError(f"{label} is not an NPY member")
    version = stream.read(2)
    if len(version) != 2:
        raise RegenerationError(f"{label} has a truncated NPY version")
    if version[0] == 1:
        raw_length = stream.read(2)
        if len(raw_length) != 2:
            raise RegenerationError(f"{label} has a truncated NPY header length")
        header_length = struct.unpack("<H", raw_length)[0]
    elif version[0] in {2, 3}:
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise RegenerationError(f"{label} has a truncated NPY header length")
        header_length = struct.unpack("<I", raw_length)[0]
    else:
        raise RegenerationError(f"{label} uses unsupported NPY version {tuple(version)}")
    raw_header = stream.read(header_length)
    try:
        header = ast.literal_eval(raw_header.decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise RegenerationError(f"{label} has an invalid NPY header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise RegenerationError(f"{label} NPY header field drift")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        or type(header["fortran_order"]) is not bool
        or not isinstance(header["descr"], str)
    ):
        raise RegenerationError(f"{label} NPY shape/dtype drift")
    return header, stream.tell()


def _decode_fixed_strings(header: Mapping[str, Any], payload: bytes, label: str) -> list[str]:
    shape = tuple(header["shape"])
    if len(shape) != 1:
        raise RegenerationError(f"{label} must be a one-dimensional string array")
    descr = str(header["descr"])
    if descr.startswith("<U") or descr.startswith(">U"):
        width = _integer(int(descr[2:]), f"{label} Unicode width", minimum=1)
        endian = "little" if descr.startswith("<") else "big"
        item_bytes = width * 4
        if len(payload) != shape[0] * item_bytes:
            raise RegenerationError(f"{label} Unicode payload length drift")
        rows = []
        for offset in range(0, len(payload), item_bytes):
            text = payload[offset : offset + item_bytes].decode(
                "utf-32-le" if endian == "little" else "utf-32-be"
            )
            rows.append(text.rstrip("\x00"))
        return rows
    if descr.startswith("|S") or descr.startswith("<S") or descr.startswith(">S"):
        width = _integer(int(descr[2:]), f"{label} byte width", minimum=1)
        if len(payload) != shape[0] * width:
            raise RegenerationError(f"{label} byte-string payload length drift")
        return [
            payload[offset : offset + width].rstrip(b"\x00").decode("ascii")
            for offset in range(0, len(payload), width)
        ]
    raise RegenerationError(f"{label} must use a fixed-width string dtype")


def _array_row_digest(payload: bytes, shape: Sequence[int], dtype: str) -> str:
    header = {"dtype": dtype, "layout": "C", "shape": list(shape)}
    return hashlib.sha256(canonical_json_bytes(header) + b"\x00" + payload).hexdigest()


def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    blocks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise RegenerationError(f"{label} payload is truncated")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _inspect_npz_at(
    root_fd: int,
    leaf: str,
    *,
    expected_members: set[str],
    expected_schema: str,
    identity_member: str,
    extra_string_members: set[str] = frozenset(),
    payload_member: str,
    payload_shape: tuple[int, ...],
    payload_descr: str,
    payload_dtype: str,
    payload_item_size: int,
) -> tuple[dict[str, Any], dict[str, list[str]], list[str], list[str]]:
    """Hash an NPZ and stream exact row hashes without NumPy or tensor loads."""

    fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegenerationError(f"{leaf} is not a single-link regular NPZ")
        whole_file_digest = hashlib.sha256()
        whole_file_bytes = 0
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            whole_file_digest.update(block)
            whole_file_bytes += len(block)
        if whole_file_bytes != before.st_size:
            raise RegenerationError(f"{leaf} changed while its bytes were hashed")
        os.lseek(fd, 0, os.SEEK_SET)
        binding = {
            "path": leaf,
            "bytes": whole_file_bytes,
            "sha256": whole_file_digest.hexdigest(),
        }
        with os.fdopen(os.dup(fd), "rb", closefd=True) as file_object:
            try:
                archive = zipfile.ZipFile(file_object, "r")
            except (OSError, zipfile.BadZipFile) as exc:
                raise RegenerationError(f"{leaf} is not a valid NPZ") from exc
            with archive:
                names = archive.namelist()
                expected_names = {f"{name}.npy" for name in expected_members}
                if set(names) != expected_names or len(names) != len(set(names)):
                    raise RegenerationError(f"{leaf} NPZ member set drift")
                headers: dict[str, dict[str, Any]] = {}
                string_values: dict[str, list[str]] = {}
                row_hashes: list[str] = []
                raw_row_hashes: list[str] = []
                for member in names:
                    stem = Path(member).stem
                    if Path(member).name != member:
                        raise RegenerationError(f"{leaf} contains a nested NPZ member")
                    with archive.open(member, "r") as stream:
                        header, _ = _npy_header(stream, f"{leaf}:{member}")
                        headers[stem] = header
                        if stem in {"schema", identity_member, *extra_string_members}:
                            string_values[stem] = _decode_fixed_strings(
                                header, stream.read(), f"{leaf}:{member}"
                            )
                        elif stem == payload_member:
                            if (
                                tuple(header["shape"]) != payload_shape
                                or header["fortran_order"] is not False
                                or header["descr"] != payload_descr
                            ):
                                raise RegenerationError(
                                    f"{leaf} canonical tensor header drift"
                                )
                            row_shape = payload_shape[1:]
                            row_bytes = math.prod(row_shape) * payload_item_size
                            for row_index in range(payload_shape[0]):
                                raw = _read_exact(
                                    stream,
                                    row_bytes,
                                    f"{leaf}:{payload_member}[{row_index}]",
                                )
                                raw_row_hashes.append(hashlib.sha256(raw).hexdigest())
                                row_hashes.append(
                                    _array_row_digest(raw, row_shape, payload_dtype)
                                )
                            if stream.read(1):
                                raise RegenerationError(
                                    f"{leaf}:{payload_member} has trailing payload bytes"
                                )
                payload = headers[payload_member]
                if (
                    tuple(payload["shape"]) != payload_shape
                    or payload["fortran_order"] is not False
                    or payload["descr"] != payload_descr
                ):
                    raise RegenerationError(f"{leaf} canonical tensor header drift")
                if string_values.get("schema") != [expected_schema]:
                    raise RegenerationError(f"{leaf} NPZ schema string drift")
                identity_values = string_values.get(identity_member)
                if identity_values is None:
                    raise RegenerationError(f"{leaf} lacks readable pixel identities")
                if len(identity_values) != payload_shape[0]:
                    raise RegenerationError(f"{leaf} identity/payload row-count drift")
                after = os.fstat(fd)
                projection = lambda row: (
                    row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
                    row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
                )
                if projection(before) != projection(after):
                    raise RegenerationError(f"{leaf} changed while its NPZ rows were read")
                return binding, string_values, row_hashes, raw_row_hashes
    finally:
        os.close(fd)


def _load_json_inputs(root_fd: int, leaves: Sequence[str]) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    raws: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for leaf in leaves:
        raw = _read_regular_at(root_fd, leaf)
        assert raw is not None
        value = parse_canonical_json(raw, label=leaf)
        if not isinstance(value, dict):
            raise RegenerationError(f"{leaf} must contain a JSON object")
        raws[leaf] = raw
        documents[leaf] = value
    return raws, documents


def _binding_matches(binding: Any, actual: Mapping[str, Any], label: str) -> None:
    if not isinstance(binding, Mapping):
        raise RegenerationError(f"{label} binding must be an object")
    for field in ("path", "bytes", "sha256"):
        if binding.get(field) != actual[field]:
            raise RegenerationError(f"{label} binding {field} drift")


def _authority_v1_binding(module: Any) -> dict[str, Any]:
    value = _call(module, "v1_retained_root_authority")
    if not isinstance(value, Mapping):
        raise RegenerationError("V1 retained-root authority must be an object")
    root = value.get("root")
    disposition = value.get("disposition")
    leaves = value.get("leaves")
    if (
        not isinstance(root, str)
        or not Path(root).is_absolute()
        or not isinstance(disposition, str)
        or not disposition
        or not isinstance(leaves, list)
        or len(leaves) != 12
    ):
        raise RegenerationError("V1 retained-root authority shape drift")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(leaves):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RegenerationError(f"V1 retained leaf {index} field drift")
        leaf = _string(row["path"], f"V1 retained leaf {index}")
        _safe_leaf(leaf)
        if leaf in seen:
            raise RegenerationError("V1 retained-root authority duplicates a leaf")
        seen.add(leaf)
        normalized.append(
            {
                "path": leaf,
                "bytes": _integer(row["bytes"], f"V1 {leaf} bytes", minimum=1),
                "sha256": _sha256(row["sha256"], f"V1 {leaf} sha256"),
            }
        )
    if value.get("leaf_count") != 12 or value.get("total_bytes") != sum(
        row["bytes"] for row in normalized
    ):
        raise RegenerationError("V1 retained-root count/byte authority drift")
    if value.get("inventory") != leaves:
        raise RegenerationError("V1 retained-root duplicate inventory projection drift")
    reusable = value.get("reusable_leaves")
    invalid = value.get("invalid_nonreusable_leaves")
    custody = value.get("custody_only_leaves")
    nonreusable = value.get("nonreusable_leaves")
    if tuple(reusable or ()) != V1_REUSABLE_FILES:
        raise RegenerationError("V1 reusable leaf authority drift")
    if set(invalid or ()) != {
        "latent_index.json", "latents.npz", "calibration.json",
        "stage_a_beliefs.jsonl", "stage_a_metrics.json",
    }:
        raise RegenerationError("V1 invalid/nonreusable leaf authority drift")
    if custody != ["contract.json"] or set(nonreusable or ()) != set(invalid) | {
        "contract.json"
    }:
        raise RegenerationError("V1 custody/nonreuse category authority drift")
    if set(reusable) | set(nonreusable) != seen or set(reusable) & set(nonreusable):
        raise RegenerationError("V1 retained-root categories do not partition inventory")
    normalized_value = dict(value)
    normalized_value["leaves"] = normalized
    normalized_value["inventory"] = [dict(row) for row in normalized]
    return normalized_value


def _contract_v1_binding(contract: Mapping[str, Any]) -> Any:
    scientific = contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V2 runtime contract lacks scientific_contract")
    value = scientific.get("v1_retained_root_binding")
    if value is None:
        # This fallback keeps the reducer fail closed while allowing the final
        # contract owner to nest the binding in a named source-binding section.
        bindings = scientific.get("source_bindings")
        if isinstance(bindings, Mapping):
            value = bindings.get("v1_retained_root_binding")
    if value is None:
        raise RegenerationError("V2 contract lacks the V1 retained-root binding")
    return value


def _validate_runtime_contract_binding(
    runtime: Mapping[str, Any], *, source_freeze_commit: str, v1_authority: Mapping[str, Any]
) -> None:
    fields = {
        "schema", "source_freeze_commit", "parent_commit", "scientific_contract",
        "v1_binding_verified_before_copy", "content_digest",
    }
    if set(runtime) != fields:
        raise RegenerationError("V2 runtime contract field set drift")
    _validate_content_digest(runtime, "V2 runtime contract")
    if (
        runtime.get("schema")
        != "occluded_goal_topological_belief_v2.runtime_contract.v1"
        or runtime.get("source_freeze_commit") != source_freeze_commit
    ):
        raise RegenerationError("V2 runtime contract source binding drift")
    parent = runtime.get("parent_commit")
    if not isinstance(parent, str) or len(parent) != 40 or any(
        character not in "0123456789abcdef" for character in parent
    ):
        raise RegenerationError("V2 runtime contract parent binding drift")
    scientific = runtime.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V2 runtime scientific contract must be an object")
    _validate_content_digest(scientific, "V2 scientific contract")
    if scientific.get("experiment_id") != EXPERIMENT_ID:
        raise RegenerationError("V2 scientific contract experiment identity drift")
    if runtime.get("v1_binding_verified_before_copy") != v1_authority:
        raise RegenerationError("V2 pre-copy V1 verification binding drift")


def _validate_v1_reusable_copies(
    *,
    module: Any,
    contract: Mapping[str, Any],
    v2_root_fd: int,
) -> dict[str, Any]:
    authority = _authority_v1_binding(module)
    if _contract_v1_binding(contract) != authority:
        raise RegenerationError("runtime contract V1 binding differs from frozen authority")
    root = Path(authority["root"])
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise RegenerationError("V1 retained root is absent or traverses a symlink")
    by_leaf = {row["path"]: row for row in authority["leaves"]}
    try:
        v1_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RegenerationError("V1 retained root cannot be opened read-only") from exc
    validations: list[dict[str, Any]] = []
    complete_inventory: list[dict[str, Any]] = []
    try:
        observed_names = sorted(os.listdir(v1_fd))
        if observed_names != sorted(by_leaf):
            raise RegenerationError("V1 retained root exact twelve-leaf inventory drift")
        for leaf in observed_names:
            actual = _stream_binding_at(v1_fd, leaf)
            _binding_matches(by_leaf[leaf], actual, f"V1 retained inventory {leaf}")
            complete_inventory.append(actual)
        for leaf in V1_REUSABLE_FILES:
            if leaf == "observations.npz":
                v1_binding = _stream_binding_at(v1_fd, leaf)
                v2_binding = _stream_binding_at(v2_root_fd, leaf)
            else:
                v1_binding = _regular_binding_at(v1_fd, leaf)
                v2_binding = _regular_binding_at(v2_root_fd, leaf)
            expected = by_leaf[leaf]
            _binding_matches(expected, v1_binding, f"V1 authority {leaf}")
            if v1_binding != v2_binding:
                raise RegenerationError(f"V2 reusable input is not byte-identical to V1: {leaf}")
            validations.append(dict(v2_binding))
    finally:
        os.close(v1_fd)
    return {
        "v1_root": authority["root"],
        "disposition": authority["disposition"],
        "authority_leaf_count": len(authority["leaves"]),
        "opened_v1_leaf_count": len(authority["leaves"]),
        "complete_inventory_hash_validated": True,
        "complete_inventory": complete_inventory,
        "copied_leaf_count": len(V1_REUSABLE_FILES),
        "reusable_copies_byte_identical": True,
        "reusable_files": validations,
    }


def _pixel_header_digest(payload: bytes, shape: Sequence[int], dtype: str) -> str:
    return _array_row_digest(payload, shape, dtype)


def _inspect_observations_at(
    root_fd: int,
    keyframe_index: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_content_digest(keyframe_index, "V1 keyframe index")
    image_shape = keyframe_index.get("image_shape")
    if image_shape != [168, 224, 3]:
        raise RegenerationError("V1 keyframe image shape drift")
    binding, strings, pixel_hashes, raw_hashes = _inspect_npz_at(
        root_fd,
        "observations.npz",
        expected_members={"schema", "pixel_template_ids", "image_sha256", "images"},
        expected_schema="occluded_goal_topological_belief_v1.observations.v1",
        identity_member="pixel_template_ids",
        extra_string_members={"image_sha256"},
        payload_member="images",
        payload_shape=(EXPECTED_TEMPLATE_COUNT, 168, 224, 3),
        payload_descr="|u1",
        payload_dtype="|u1",
        payload_item_size=1,
    )
    _binding_matches(
        keyframe_index.get("observations_file"), binding,
        "V1 keyframe observations file",
    )
    template_ids = strings["pixel_template_ids"]
    declared_raw_hashes = strings["image_sha256"]
    if (
        len(template_ids) != EXPECTED_TEMPLATE_COUNT
        or len(set(template_ids)) != EXPECTED_TEMPLATE_COUNT
        or len(declared_raw_hashes) != EXPECTED_TEMPLATE_COUNT
        or declared_raw_hashes != raw_hashes
    ):
        raise RegenerationError("V1 observation template/raw-hash evidence drift")
    if len(set(pixel_hashes)) != EXPECTED_UNIQUE_PIXEL_COUNT:
        raise RegenerationError("canonical pixel identity cardinality drift")
    records = keyframe_index.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_TEMPLATE_COUNT:
        raise RegenerationError("V1 keyframe template record cardinality drift")
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or (
            row.get("row_index") != index
            or row.get("pixel_template_id") != template_ids[index]
            or row.get("image_sha256") != raw_hashes[index]
        ):
            raise RegenerationError("V1 keyframe/observation row binding drift")
    return {
        "binding": binding,
        "template_ids": template_ids,
        "source_image_sha256s": raw_hashes,
        "pixel_sha256s": pixel_hashes,
        "unique_pixel_sha256s": sorted(set(pixel_hashes)),
    }


def _validate_pixel_documents(
    pixel_index: Mapping[str, Any],
    template_index: Mapping[str, Any],
    occurrence_index: Mapping[str, Any],
    keyframe_index: Mapping[str, Any],
    observation_evidence: Mapping[str, Any],
    *,
    rgb_hash_domain: str,
) -> dict[str, Any]:
    for label, document in (
        ("pixel_index", pixel_index),
        ("template_to_pixel_index", template_index),
        ("occurrence_index", occurrence_index),
    ):
        _validate_content_digest(document, label)
    pixel_fields = {
        "schema", "experiment_id", "observations_binding", "identity_rule",
        "records", "content_digest",
    }
    pixel_record_fields = {
        "pixel_index", "pixel_sha256", "canonical_template_id",
        "member_template_ids", "observation_row_indices",
    }
    template_fields = {
        "schema", "experiment_id", "pixel_index_binding", "records",
        "content_digest",
    }
    template_record_fields = {
        "template_row_index", "pixel_template_id", "pixel_sha256",
        "pixel_index", "canonical_template_id",
    }
    occurrence_fields = {
        "schema", "experiment_id", "keyframe_index_binding",
        "template_index_binding", "records", "content_digest",
    }
    occurrence_record_fields = {
        "occurrence_index", "capture_id", "episode_id", "node_id", "phase",
        "timestamp_s", "pixel_template_id", "pixel_sha256", "pixel_index",
        "query_ids",
    }
    for label, document, fields in (
        ("pixel_index", pixel_index, pixel_fields),
        ("template_to_pixel_index", template_index, template_fields),
        ("occurrence_index", occurrence_index, occurrence_fields),
    ):
        if set(document) != fields:
            raise RegenerationError(f"{label} field set drift")
        if document.get("experiment_id") != EXPERIMENT_ID:
            raise RegenerationError(f"{label} experiment identity drift")
    _binding_matches(
        pixel_index.get("observations_binding"),
        observation_evidence["binding"],
        "pixel index observations",
    )
    if pixel_index.get("identity_rule") != rgb_hash_domain:
        raise RegenerationError("pixel index hash-domain rule drift")
    if template_index.get("pixel_index_binding") != pixel_index.get(
        "content_digest"
    ):
        raise RegenerationError("template/pixel content binding drift")
    counts = {
        "template_count": EXPECTED_TEMPLATE_COUNT,
        "unique_pixel_count": EXPECTED_UNIQUE_PIXEL_COUNT,
        "reuse_count": EXPECTED_REUSE_COUNT,
        "occurrence_count": EXPECTED_OCCURRENCE_COUNT,
    }
    pixel_rows = pixel_index.get("records")
    template_rows = template_index.get("records")
    occurrence_rows = occurrence_index.get("records")
    if not isinstance(pixel_rows, list) or len(pixel_rows) != EXPECTED_UNIQUE_PIXEL_COUNT:
        raise RegenerationError("pixel index row count drift")
    if not isinstance(template_rows, list) or len(template_rows) != EXPECTED_TEMPLATE_COUNT:
        raise RegenerationError("template index row count drift")
    if not isinstance(occurrence_rows, list) or len(occurrence_rows) != EXPECTED_OCCURRENCE_COUNT:
        raise RegenerationError("occurrence index row count drift")
    pixel_by_index: dict[int, dict[str, Any]] = {}
    template_by_id: dict[str, dict[str, Any]] = {}
    flattened_templates: list[str] = []
    previous_sha = ""
    for row_index, row in enumerate(pixel_rows):
        if not isinstance(row, Mapping) or set(row) != pixel_record_fields:
            raise RegenerationError("pixel index row must be an object")
        index = _integer(row.get("pixel_index"), "pixel_index")
        pixel_sha = _sha256(row.get("pixel_sha256"), "pixel_sha256")
        if index != row_index or pixel_sha <= previous_sha:
            raise RegenerationError("pixel rows must be index-contiguous and SHA-sorted")
        previous_sha = pixel_sha
        templates = row.get("member_template_ids")
        source_indices = row.get("observation_row_indices")
        if (
            not isinstance(templates, list)
            or not templates
            or templates != sorted(templates)
            or len(templates) != len(set(templates))
            or not isinstance(source_indices, list)
            or len(source_indices) != len(templates)
        ):
            raise RegenerationError("pixel row template/source alignment drift")
        canonical = _string(row.get("canonical_template_id"), "canonical template")
        if canonical != min(templates):
            raise RegenerationError("canonical representative is not lexicographic")
        for value in source_indices:
            _integer(value, "source observation row")
        for template_id, source_index in zip(templates, source_indices):
            if (
                source_index >= EXPECTED_TEMPLATE_COUNT
                or observation_evidence["template_ids"][source_index] != template_id
                or observation_evidence["pixel_sha256s"][source_index] != pixel_sha
            ):
                raise RegenerationError(
                    "pixel group differs from exact copied observation bytes"
                )
        flattened_templates.extend(templates)
        pixel_by_index[index] = dict(row)
    if [row["pixel_sha256"] for row in pixel_rows] != observation_evidence[
        "unique_pixel_sha256s"
    ]:
        raise RegenerationError("pixel index differs from recomputed observation identities")
    if len(flattened_templates) != EXPECTED_TEMPLATE_COUNT or len(set(flattened_templates)) != EXPECTED_TEMPLATE_COUNT:
        raise RegenerationError("pixel groups do not partition all templates exactly once")
    reused = 0
    for row_index, row in enumerate(template_rows):
        if not isinstance(row, Mapping) or set(row) != template_record_fields:
            raise RegenerationError("template index row must be an object")
        template_id = _string(row.get("pixel_template_id"), "pixel template ID")
        if row.get("template_row_index") != row_index or template_id in template_by_id:
            raise RegenerationError("template rows must follow observation-row order")
        if observation_evidence["template_ids"][row_index] != template_id:
            raise RegenerationError("template row identity differs from observations")
        pixel = pixel_by_index.get(_integer(row.get("pixel_index"), "template pixel index"))
        if pixel is None or template_id not in pixel["member_template_ids"]:
            raise RegenerationError("template row does not resolve to its pixel group")
        if (
            row.get("pixel_sha256") != pixel["pixel_sha256"]
            or row.get("canonical_template_id") != pixel["canonical_template_id"]
        ):
            raise RegenerationError("template row pixel/canonical binding drift")
        reused += int(template_id != pixel["canonical_template_id"])
        template_by_id[template_id] = dict(row)
    if set(template_by_id) != set(flattened_templates) or reused != EXPECTED_REUSE_COUNT:
        raise RegenerationError("template reuse partition drift")
    captures = keyframe_index.get("captures")
    if not isinstance(captures, list) or len(captures) != EXPECTED_OCCURRENCE_COUNT:
        raise RegenerationError("V1 keyframe occurrence population drift")
    capture_by_id = {
        _string(row.get("capture_id"), "keyframe capture ID"): row
        for row in captures if isinstance(row, Mapping)
    }
    if len(capture_by_id) != EXPECTED_OCCURRENCE_COUNT:
        raise RegenerationError("V1 keyframe capture identities are not unique")
    keyframe_binding = {
        "path": "keyframe_index.json",
        "bytes": observation_evidence["keyframe_bytes"],
        "sha256": observation_evidence["keyframe_sha256"],
    }
    _binding_matches(
        occurrence_index.get("keyframe_index_binding"),
        keyframe_binding,
        "occurrence keyframe index",
    )
    if occurrence_index.get("template_index_binding") != template_index.get(
        "content_digest"
    ):
        raise RegenerationError("occurrence/template content binding drift")
    for occurrence_position, row in enumerate(occurrence_rows):
        if not isinstance(row, Mapping) or set(row) != occurrence_record_fields:
            raise RegenerationError("occurrence index row must be an object")
        capture_id = _string(row.get("capture_id"), "occurrence capture ID")
        if row.get("occurrence_index") != occurrence_position:
            raise RegenerationError("occurrence rows must be index-contiguous")
        source = capture_by_id.get(capture_id)
        if source is None:
            raise RegenerationError("occurrence row is outside V1 keyframe evidence")
        for field in (
            "capture_id", "episode_id", "node_id", "phase", "timestamp_s",
            "query_ids", "pixel_template_id",
        ):
            if row.get(field) != source.get(field):
                raise RegenerationError(f"occurrence/V1 keyframe projection drift: {field}")
        template = template_by_id.get(str(row.get("pixel_template_id")))
        if template is None or any(
            row.get(field) != template.get(field)
            for field in ("pixel_sha256", "pixel_index")
        ):
            raise RegenerationError("occurrence does not resolve through template mapping")
    return {
        **counts,
        "pixel_group_partition_validated": True,
        "template_mapping_validated": True,
        "occurrence_mapping_validated": True,
        "canonical_representatives_lexicographic": True,
    }


def _validate_canonical_indices(
    latent_index: Mapping[str, Any],
    descriptor_index: Mapping[str, Any],
    *,
    pixel_rows: Sequence[Mapping[str, Any]],
    token_binding: Mapping[str, Any],
    descriptor_binding: Mapping[str, Any],
    token_pixel_ids: Sequence[str],
    descriptor_pixel_ids: Sequence[str],
    token_row_hashes: Sequence[str],
    descriptor_row_hashes: Sequence[str],
    pixel_index_content_digest: str,
    encoding_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_content_digest(latent_index, "canonical_latent_index")
    _validate_content_digest(descriptor_index, "canonical_descriptor_index")
    expected_pixels = [str(row["pixel_sha256"]) for row in pixel_rows]
    if list(token_pixel_ids) != expected_pixels or list(descriptor_pixel_ids) != expected_pixels:
        raise RegenerationError("canonical NPZ pixel identity order drift")
    latent_fields = {
        "schema", "experiment_id", "source_freeze_commit", "pixel_index_binding",
        "encoding_determinism_receipt_binding", "tokens_file", "records",
        "content_digest",
    }
    descriptor_fields = {
        "schema", "experiment_id", "source_freeze_commit", "pixel_index_binding",
        "encoding_determinism_receipt_binding", "descriptors_file", "records",
        "content_digest",
    }
    latent_record_fields = {
        "pixel_index", "pixel_sha256", "canonical_template_id",
        "raw_token_row_index", "raw_token_sha256",
    }
    descriptor_record_fields = {
        "pixel_index", "pixel_sha256", "canonical_template_id",
        "spatial_descriptor_row_index", "spatial_descriptor_sha256",
    }
    for document, label, fields, record_fields, binding, file_field, row_index_field, row_hash_field, actual_hashes in (
        (
            latent_index, "canonical latent", latent_fields, latent_record_fields,
            token_binding, "tokens_file", "raw_token_row_index",
            "raw_token_sha256", token_row_hashes,
        ),
        (
            descriptor_index, "canonical descriptor", descriptor_fields,
            descriptor_record_fields, descriptor_binding, "descriptors_file",
            "spatial_descriptor_row_index", "spatial_descriptor_sha256",
            descriptor_row_hashes,
        ),
    ):
        if set(document) != fields or document.get("experiment_id") != EXPERIMENT_ID:
            raise RegenerationError(f"{label} index field/identity drift")
        if document.get("pixel_index_binding") != pixel_index_content_digest:
            raise RegenerationError(f"{label} pixel-index binding drift")
        _binding_matches(
            document.get("encoding_determinism_receipt_binding"),
            encoding_receipt_binding,
            f"{label} encoding receipt",
        )
        rows = document.get("records")
        if not isinstance(rows, list) or len(rows) != EXPECTED_UNIQUE_PIXEL_COUNT:
            raise RegenerationError(f"{label} index rows drift")
        file_binding = document.get(file_field)
        _binding_matches(file_binding, binding, f"{label} NPZ")
        for index, (row, pixel) in enumerate(zip(rows, pixel_rows)):
            if not isinstance(row, Mapping) or set(row) != record_fields:
                raise RegenerationError(f"{label} row must be an object")
            if (
                row.get("pixel_index") != index
                or row.get("pixel_sha256") != pixel["pixel_sha256"]
                or row.get("canonical_template_id") != pixel["canonical_template_id"]
                or row.get(row_index_field) != index
            ):
                raise RegenerationError(f"{label} row identity drift")
            digest = _sha256(row.get(row_hash_field), f"{label} row digest")
            if digest != actual_hashes[index]:
                raise RegenerationError(
                    f"{label} row digest differs from actual canonical-array bytes"
                )
    return {
        "rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "token_npz_header_validated": True,
        "descriptor_npz_header_validated": True,
        "pixel_identity_order_validated": True,
        "index_file_bindings_validated": True,
    }


def _validate_gate_receipts(
    determinism: Mapping[str, Any],
    integrity: Mapping[str, Any],
    *,
    pixel_rows: Sequence[Mapping[str, Any]],
    actual_bindings: Mapping[str, Mapping[str, Any]],
    source_freeze_commit: str,
    hash_domains: Mapping[str, Any],
    pixel_index_content_digest: str,
    template_index_content_digest: str,
    occurrence_index_content_digest: str,
    latent_index_content_digest: str,
    descriptor_index_content_digest: str,
    v1_authority: Mapping[str, Any],
    copied_v1_bindings: Sequence[Mapping[str, Any]],
    token_row_hashes: Sequence[str],
    descriptor_row_hashes: Sequence[str],
    required_gate_names: Sequence[str],
) -> dict[str, Any]:
    _validate_no_self_digest(determinism, "encoding_determinism_receipt")
    _validate_no_self_digest(integrity, "cache_integrity_receipt")
    encoding_fields = {
        "schema", "experiment_id", "source_freeze_commit", "observations_binding",
        "pixel_index_content_digest", "encoder_binding", "hash_domains", "counts",
        "pre_outcome_boundary", "passes", "comparisons",
    }
    pass_fields = {
        "pass_index", "fresh_encoder_instance_id", "ordered_pixel_sha256s",
        "records", "canonical_cache_content_digest",
    }
    pass_record_fields = {
        "pixel_index", "pixel_sha256", "canonical_template_id",
        "preprocessed_tensor_sha256", "raw_token_sha256",
        "spatial_descriptor_sha256",
    }
    comparison_fields = {
        "pixel_order_exact", "preprocessed_tensors_exact", "raw_tokens_exact",
        "spatial_descriptors_exact", "canonical_cache_content_digest_exact", "pass",
    }
    if set(determinism) != encoding_fields or determinism.get("experiment_id") != EXPERIMENT_ID:
        raise RegenerationError("encoding determinism receipt field/identity drift")
    if determinism.get("source_freeze_commit") != source_freeze_commit:
        raise RegenerationError("encoding receipt source-freeze binding drift")
    _binding_matches(
        determinism.get("observations_binding"),
        actual_bindings["observations.npz"],
        "encoding observations",
    )
    if (
        determinism.get("pixel_index_content_digest") != pixel_index_content_digest
        or determinism.get("hash_domains") != hash_domains
    ):
        raise RegenerationError("encoding receipt input/hash-domain binding drift")
    counts = determinism.get("counts")
    if counts != {
        "template_rows": EXPECTED_TEMPLATE_COUNT,
        "unique_pixel_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "reused_template_rows": EXPECTED_REUSE_COUNT,
        "pass_count": 2,
        "encoder_invocations_per_pass": EXPECTED_UNIQUE_PIXEL_COUNT,
        "singleton_batch_size": 1,
    }:
        raise RegenerationError("encoding determinism counts drift")
    expected_pre_outcome_boundary = {
        "boundary": "IMMEDIATELY_BEFORE_FIRST_CANONICAL_ENCODER_INITIALIZATION",
        "present_v2_leaf_names": sorted((CONTRACT_FILE, *V1_REUSABLE_FILES)),
        "forbidden_outcome_leaf_names": [
            CALIBRATION_FILE,
            STAGE_A_BELIEFS_FILE,
            STAGE_A_METRICS_FILE,
            STAGE_B_TRACE_FILE,
            STAGE_B_METRICS_FILE,
            *FINAL_PUBLICATION_FILES,
        ],
        "observed_forbidden_outcome_leaf_names": [],
        "calibration_outcome_documents_opened": 0,
        "heldout_outcome_documents_opened": 0,
        "external_regeneration_receipt_present": False,
    }
    if determinism.get("pre_outcome_boundary") != expected_pre_outcome_boundary:
        raise RegenerationError("encoding pre-outcome boundary evidence drift")
    expected_pixels = [str(row["pixel_sha256"]) for row in pixel_rows]
    passes = determinism.get("passes")
    if not isinstance(passes, list) or len(passes) != 2:
        raise RegenerationError("determinism receipt must contain exactly two passes")
    normalized_records: list[list[dict[str, Any]]] = []
    encoder_instance_ids: list[str] = []
    for pass_index, row in enumerate(passes, start=1):
        if not isinstance(row, Mapping) or set(row) != pass_fields or row.get("pass_index") != pass_index:
            raise RegenerationError("determinism pass identity drift")
        encoder_instance_ids.append(
            _string(row.get("fresh_encoder_instance_id"), "fresh encoder instance")
        )
        if row.get("ordered_pixel_sha256s") != expected_pixels:
            raise RegenerationError("determinism pass pixel order drift")
        records = row.get("records")
        if not isinstance(records, list) or len(records) != EXPECTED_UNIQUE_PIXEL_COUNT:
            raise RegenerationError("determinism pass record count drift")
        normalized: list[dict[str, Any]] = []
        for index, (record, pixel) in enumerate(zip(records, pixel_rows)):
            if not isinstance(record, Mapping) or set(record) != pass_record_fields:
                raise RegenerationError("determinism pass record field drift")
            if (
                record.get("pixel_index") != index
                or record.get("pixel_sha256") != pixel["pixel_sha256"]
                or record.get("canonical_template_id") != pixel["canonical_template_id"]
            ):
                raise RegenerationError("determinism pass record identity drift")
            _sha256(record.get("preprocessed_tensor_sha256"), "preprocessed tensor digest")
            raw_sha = _sha256(record.get("raw_token_sha256"), "raw token digest")
            descriptor_sha = _sha256(
                record.get("spatial_descriptor_sha256"), "descriptor digest"
            )
            if raw_sha != token_row_hashes[index] or descriptor_sha != descriptor_row_hashes[index]:
                raise RegenerationError(
                    "determinism record differs from actual canonical tensor row"
                )
            normalized.append(dict(record))
        if row.get("canonical_cache_content_digest") != canonical_digest(normalized):
            raise RegenerationError("determinism canonical-cache content digest drift")
        normalized_records.append(normalized)
    if len(set(encoder_instance_ids)) != 2 or normalized_records[0] != normalized_records[1]:
        raise RegenerationError("two fresh deterministic encoder passes differ")
    comparisons = determinism.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != comparison_fields or any(
        value is not True for value in comparisons.values()
    ):
        raise RegenerationError("two-pass comparison gate failed")

    cache_fields = {
        "schema", "experiment_id", "source_freeze_commit",
        "v1_retained_root_binding", "copied_v1_input_bindings",
        "pixel_index_content_digest", "template_index_content_digest",
        "occurrence_index_content_digest", "encoding_determinism_receipt_binding",
        "canonical_latent_index_content_digest",
        "canonical_descriptor_index_content_digest", "counts", "gates",
    }
    if set(integrity) != cache_fields or integrity.get("experiment_id") != EXPERIMENT_ID:
        raise RegenerationError("cache integrity receipt field/identity drift")
    if integrity.get("source_freeze_commit") != source_freeze_commit:
        raise RegenerationError("cache integrity source-freeze binding drift")
    if integrity.get("v1_retained_root_binding") != v1_authority:
        raise RegenerationError("cache integrity V1 authority binding drift")
    copied = integrity.get("copied_v1_input_bindings")
    if not isinstance(copied, list) or len(copied) != len(copied_v1_bindings):
        raise RegenerationError("cache integrity copied-input binding drift")
    expected_copies = {str(row["path"]): dict(row) for row in copied_v1_bindings}
    observed_copies: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(copied):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RegenerationError(
                f"cache integrity copied-input binding {index} field drift"
            )
        path = _string(row.get("path"), "cache copied-input path")
        if path in observed_copies:
            raise RegenerationError("cache integrity duplicates a copied-input binding")
        _integer(row.get("bytes"), "cache copied-input bytes", minimum=1)
        _sha256(row.get("sha256"), "cache copied-input sha256")
        observed_copies[path] = dict(row)
    if observed_copies != expected_copies:
        raise RegenerationError("cache integrity copied-input binding drift")
    _binding_matches(
        integrity.get("encoding_determinism_receipt_binding"),
        actual_bindings["encoding_determinism_receipt.json"],
        "cache integrity encoding receipt",
    )
    if (
        integrity.get("pixel_index_content_digest") != pixel_index_content_digest
        or integrity.get("template_index_content_digest") != template_index_content_digest
        or integrity.get("occurrence_index_content_digest") != occurrence_index_content_digest
        or integrity.get("canonical_latent_index_content_digest") != latent_index_content_digest
        or integrity.get("canonical_descriptor_index_content_digest") != descriptor_index_content_digest
    ):
        raise RegenerationError("cache integrity content-digest binding drift")
    if integrity.get("counts") != {
        "template_rows": EXPECTED_TEMPLATE_COUNT,
        "unique_pixel_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "reused_template_rows": EXPECTED_REUSE_COUNT,
        "occurrence_rows": EXPECTED_OCCURRENCE_COUNT,
        "canonical_token_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "canonical_descriptor_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "encoder_invocations_per_pass": EXPECTED_UNIQUE_PIXEL_COUNT,
        "multi_template_pixel_groups": EXPECTED_MULTI_TEMPLATE_GROUP_COUNT,
        "singleton_pixel_groups": (
            EXPECTED_UNIQUE_PIXEL_COUNT - EXPECTED_MULTI_TEMPLATE_GROUP_COUNT
        ),
        "templates_in_multi_template_groups": (
            EXPECTED_TEMPLATE_COUNT
            - (EXPECTED_UNIQUE_PIXEL_COUNT - EXPECTED_MULTI_TEMPLATE_GROUP_COUNT)
        ),
    }:
        raise RegenerationError("cache integrity count drift")
    gate_names = set(required_gate_names)
    independently_required = {
        "no_calibration_or_heldout_outcome_opened",
        "no_occurrence_mapped_by_gpu_slot",
        "no_template_specific_token_copy_differs",
        "canonical_regeneration_byte_identical",
    }
    if not independently_required.issubset(gate_names):
        raise RegenerationError("cache authority omits required replacement gates")
    gates = integrity.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != gate_names or any(
        value is not True for value in gates.values()
    ):
        raise RegenerationError("cache integrity gate drift")
    return {
        "two_fresh_encoder_loads": True,
        "singleton_invocations": EXPECTED_UNIQUE_PIXEL_COUNT * 2,
        "comparison_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
        "exact_byte_equality": True,
        "tolerance_used": False,
        "pre_outcome_boundary_validated": True,
        "no_gpu_slot_or_template_tensor_mapping": True,
        "canonical_regeneration_byte_identical": True,
        "cache_integrity_gate_passed": True,
    }


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegenerationError(f"source-freeze Git observation failed: {' '.join(arguments)}") from exc


def _git_text(*arguments: str) -> str:
    try:
        return _git_bytes(*arguments).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RegenerationError("source-freeze Git output is not ASCII") from exc


def _read_repository_source(relative: str) -> bytes:
    return V1_REDUCER._read_repository_source(relative)


def _observe_source_freeze(module: Any) -> dict[str, Any]:
    if _metrics_module_name(module) != METRICS_MODULE:
        raise RegenerationError("production reduction requires the exact V2 metrics module")
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or Path(module_path).absolute() != ROOT / METRICS_SOURCE_PATH:
        raise RegenerationError("V2 metrics module source path drift")
    head = _git_text("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RegenerationError("source-freeze HEAD is not canonical 40-hex")
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("V2 independent reduction requires a clean worktree")
    bindings: dict[str, dict[str, Any]] = {}
    for key, relative in (
        ("independent_reducer", REDUCER_SOURCE_PATH),
        ("metrics_module", METRICS_SOURCE_PATH),
        ("v1_pure_reducer_helpers", V1_REDUCER_SOURCE_PATH),
    ):
        raw = _read_repository_source(relative)
        if _git_bytes("show", f"{head}:{relative}") != raw:
            raise RegenerationError(f"source bytes differ from source-freeze HEAD: {relative}")
        bindings[key] = {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {
        "head_commit": head,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": bindings,
    }


def _validate_source_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"head_commit", "worktree_clean", "sources_exactly_equal_head", "sources"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RegenerationError("source-freeze observation field set drift")
    head = _string(value["head_commit"], "source-freeze HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RegenerationError("source-freeze HEAD drift")
    if value["worktree_clean"] is not True or value["sources_exactly_equal_head"] is not True:
        raise RegenerationError("source-freeze custody failed")
    sources = value["sources"]
    expected_paths = {
        "independent_reducer": REDUCER_SOURCE_PATH,
        "metrics_module": METRICS_SOURCE_PATH,
        "v1_pure_reducer_helpers": V1_REDUCER_SOURCE_PATH,
    }
    if not isinstance(sources, Mapping) or set(sources) != set(expected_paths):
        raise RegenerationError("source-freeze source binding set drift")
    for key, path in expected_paths.items():
        row = sources[key]
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RegenerationError("source-freeze source binding field drift")
        if row["path"] != path:
            raise RegenerationError("source-freeze source path drift")
        _integer(row["bytes"], "source bytes", minimum=1)
        _sha256(row["sha256"], "source sha256")
    return dict(value)


def _validate_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "reducer_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("V2 reducer authority must be an object")
    if authority.get("experiment_id") != EXPERIMENT_ID:
        raise RegenerationError("V2 reducer authority experiment drift")
    _validate_content_digest(authority, "V2 reducer authority")
    condition_ids = tuple(_call(module, "stage_a_condition_ids"))
    if condition_ids != EXPECTED_CONDITION_IDS:
        raise RegenerationError("V2 ordered Stage-A condition authority drift")
    cache_authority = _call(module, "cache_gate_authority")
    v1_authority = _call(module, "v1_retained_root_authority")
    if authority.get("cache_gate") != cache_authority:
        raise RegenerationError("V2 reducer/cache-gate authority drift")
    if authority.get("v1_retained_root") != v1_authority:
        raise RegenerationError("V2 reducer/V1-retained-root authority drift")
    if not isinstance(cache_authority, Mapping):
        raise RegenerationError("V2 cache-gate authority must be an object")
    _validate_content_digest(cache_authority, "V2 cache-gate authority")
    if cache_authority.get("receipt_self_digests") is not False:
        raise RegenerationError("V2 cache receipts must not carry self digests")
    return dict(authority)


def _root_inventory(root_fd: int) -> list[str]:
    try:
        names = sorted(os.listdir(root_fd))
    except OSError as exc:
        raise RegenerationError("official output root inventory cannot be read") from exc
    required = {
        CONTRACT_FILE, *V1_REUSABLE_FILES, *CACHE_JSON_FILES, *CACHE_NPZ_FILES,
        CALIBRATION_FILE, STAGE_A_BELIEFS_FILE, STAGE_A_METRICS_FILE,
    }
    conditional = {STAGE_B_TRACE_FILE, STAGE_B_METRICS_FILE}
    publication = set(FINAL_PUBLICATION_FILES)
    observed = set(names)
    if not required.issubset(observed):
        raise RegenerationError(f"V2 output root lacks required leaves: {sorted(required-observed)}")
    if (STAGE_B_TRACE_FILE in observed) != (STAGE_B_METRICS_FILE in observed):
        raise RegenerationError("conditional Stage-B leaves must be all-or-absent")
    allowed = required | conditional | publication
    if observed - allowed:
        raise RegenerationError(f"V2 output root contains unregistered leaves: {sorted(observed-allowed)}")
    if observed & publication and not publication.issubset(observed):
        raise RegenerationError("final publication leaves must be all-or-absent")
    return names


def build_regeneration_receipt(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise RegenerationError("--output-root must be an absolute non-symlink lexical path")
    try:
        if root.resolve(strict=True) != root:
            raise RegenerationError("official output root must not traverse symbolic links")
    except OSError as exc:
        raise RegenerationError("official V2 output root is absent") from exc
    module = metrics_module if metrics_module is not None else _load_metrics_module()
    authority = _validate_authority(module)
    cache_authority = authority["cache_gate"]
    cache_schemas = cache_authority["schemas"]
    hash_domains = cache_authority["hash_domains"]
    observed_freeze = (
        _observe_source_freeze(module)
        if source_freeze_observation is None
        else dict(source_freeze_observation)
    )
    source_freeze = _validate_source_freeze(observed_freeze)
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RegenerationError("official V2 output root cannot be opened safely") from exc
    try:
        inventory = _root_inventory(root_fd)
        json_raw, json_documents = _load_json_inputs(
            root_fd,
            (
                CONTRACT_FILE, "panel_manifest.json", "split_manifest.json",
                GRAPH_FILE, "keyframe_index.json", *CACHE_JSON_FILES,
                CALIBRATION_FILE, STAGE_A_METRICS_FILE,
            ),
        )
        query_raw = _read_regular_at(root_fd, QUERY_FILE)
        belief_raw = _read_regular_at(root_fd, STAGE_A_BELIEFS_FILE)
        trace_raw = _read_regular_at(root_fd, STAGE_B_TRACE_FILE, optional=True)
        stage_b_metrics_raw = _read_regular_at(root_fd, STAGE_B_METRICS_FILE, optional=True)
        assert query_raw is not None and belief_raw is not None
        v1_authority = _authority_v1_binding(module)
        _validate_runtime_contract_binding(
            json_documents[CONTRACT_FILE],
            source_freeze_commit=source_freeze["head_commit"],
            v1_authority=v1_authority,
        )
        v1_validation = _validate_v1_reusable_copies(
            module=module, contract=json_documents[CONTRACT_FILE], v2_root_fd=root_fd
        )
        observation_evidence = _inspect_observations_at(
            root_fd, json_documents["keyframe_index.json"]
        )
        observation_evidence["keyframe_bytes"] = len(
            json_raw["keyframe_index.json"]
        )
        observation_evidence["keyframe_sha256"] = hashlib.sha256(
            json_raw["keyframe_index.json"]
        ).hexdigest()
        token_binding, token_strings, token_row_hashes, _token_raw_hashes = _inspect_npz_at(
            root_fd, "canonical_tokens.npz",
            expected_members={"schema", "pixel_sha256", "raw_tokens"},
            expected_schema=str(cache_schemas["canonical_tokens_npz"]),
            identity_member="pixel_sha256",
            payload_member="raw_tokens",
            payload_shape=(EXPECTED_UNIQUE_PIXEL_COUNT, *EXPECTED_TOKEN_SHAPE),
            payload_descr="<f2",
            payload_dtype="<f2",
            payload_item_size=2,
        )
        descriptor_binding, descriptor_strings, descriptor_row_hashes, _descriptor_raw_hashes = _inspect_npz_at(
            root_fd, "canonical_descriptors.npz",
            expected_members={"schema", "pixel_sha256", "spatial_descriptors"},
            expected_schema=str(cache_schemas["canonical_descriptors_npz"]),
            identity_member="pixel_sha256",
            payload_member="spatial_descriptors",
            payload_shape=(EXPECTED_UNIQUE_PIXEL_COUNT, *EXPECTED_TOKEN_SHAPE),
            payload_descr="<f4",
            payload_dtype="<f4",
            payload_item_size=4,
        )
        actual_cache_bindings: dict[str, dict[str, Any]] = {
            leaf: _binding(leaf, json_raw[leaf]) for leaf in CACHE_JSON_FILES
        }
        actual_cache_bindings["observations.npz"] = observation_evidence["binding"]
        actual_cache_bindings["keyframe_index.json"] = _binding(
            "keyframe_index.json", json_raw["keyframe_index.json"]
        )
        actual_cache_bindings["canonical_tokens.npz"] = token_binding
        actual_cache_bindings["canonical_descriptors.npz"] = descriptor_binding
    finally:
        os.close(root_fd)

    graph = json_documents[GRAPH_FILE]
    keyframe = json_documents["keyframe_index.json"]
    calibration = json_documents[CALIBRATION_FILE]
    supplied_a = json_documents[STAGE_A_METRICS_FILE]
    query_rows = _parse_jsonl(query_raw, label=QUERY_FILE)
    belief_rows = _parse_jsonl(belief_raw, label=STAGE_A_BELIEFS_FILE)
    # Public pure validators remain authoritative for every exact V1 formula.
    _call(module, "validate_graph_manifest", graph)
    _call(module, "validate_query_rows", tuple(query_rows), graph)
    _call(module, "validate_belief_rows", tuple(belief_rows), tuple(query_rows), graph)
    _call(module, "validate_calibration", calibration, graph, tuple(query_rows))
    graphs = V1_REDUCER._graph_nodes(graph)
    heldout, candidates, role_counts = V1_REDUCER._validate_queries(query_rows, graphs)
    stage_a_validation = V1_REDUCER._validate_stage_a_beliefs(
        belief_rows, queries=heldout, candidates=candidates
    )
    calibration_authority = authority.get("calibration")
    if not isinstance(calibration_authority, Mapping):
        calibration_authority = _call(module, "calibration_authority")
    calibration_validation = V1_REDUCER._validate_calibration_evidence(
        calibration, query_rows=query_rows, authority=calibration_authority
    )
    cache_mapping_validation = _validate_pixel_documents(
        json_documents["pixel_index.json"],
        json_documents["template_to_pixel_index.json"],
        json_documents["occurrence_index.json"],
        keyframe,
        observation_evidence,
        rgb_hash_domain=str(hash_domains["rgb_pixel_sha256"]),
    )
    encoding_receipt_binding = actual_cache_bindings[
        "encoding_determinism_receipt.json"
    ]
    cache_index_validation = _validate_canonical_indices(
        json_documents["canonical_latent_index.json"],
        json_documents["canonical_descriptor_index.json"],
        pixel_rows=json_documents["pixel_index.json"]["records"],
        token_binding=token_binding,
        descriptor_binding=descriptor_binding,
        token_pixel_ids=token_strings["pixel_sha256"],
        descriptor_pixel_ids=descriptor_strings["pixel_sha256"],
        token_row_hashes=token_row_hashes,
        descriptor_row_hashes=descriptor_row_hashes,
        pixel_index_content_digest=str(
            json_documents["pixel_index.json"]["content_digest"]
        ),
        encoding_receipt_binding=encoding_receipt_binding,
    )
    module_cache_gate_validation = _call(
        module, "validate_cache_gate",
        json_documents["pixel_index.json"],
        json_documents["template_to_pixel_index.json"],
        json_documents["occurrence_index.json"],
        json_documents["canonical_latent_index.json"],
        json_documents["canonical_descriptor_index.json"],
        json_documents["encoding_determinism_receipt.json"],
        json_documents["cache_integrity_receipt.json"],
    )
    cache_gate_validation = _validate_gate_receipts(
        json_documents["encoding_determinism_receipt.json"],
        json_documents["cache_integrity_receipt.json"],
        pixel_rows=json_documents["pixel_index.json"]["records"],
        actual_bindings=actual_cache_bindings,
        source_freeze_commit=source_freeze["head_commit"],
        hash_domains=hash_domains,
        pixel_index_content_digest=str(
            json_documents["pixel_index.json"]["content_digest"]
        ),
        template_index_content_digest=str(
            json_documents["template_to_pixel_index.json"]["content_digest"]
        ),
        occurrence_index_content_digest=str(
            json_documents["occurrence_index.json"]["content_digest"]
        ),
        latent_index_content_digest=str(
            json_documents["canonical_latent_index.json"]["content_digest"]
        ),
        descriptor_index_content_digest=str(
            json_documents["canonical_descriptor_index.json"]["content_digest"]
        ),
        v1_authority=authority["v1_retained_root"],
        copied_v1_bindings=v1_validation["reusable_files"],
        token_row_hashes=token_row_hashes,
        descriptor_row_hashes=descriptor_row_hashes,
        required_gate_names=cache_authority["fields"]["cache_gates"],
    )
    recomputed_a = _call(
        module, "recompute_stage_a_metrics", graph, tuple(query_rows),
        tuple(belief_rows), calibration,
    )
    if not isinstance(recomputed_a, Mapping):
        raise RegenerationError("V2 Stage-A reducer must return a mapping")
    recomputed_a = dict(recomputed_a)
    _reject_nonfinite(recomputed_a, label="recomputed V2 Stage-A metrics")
    rebuilt_a = canonical_document_bytes(recomputed_a)
    if json_raw[STAGE_A_METRICS_FILE] != rebuilt_a:
        raise RegenerationError("V2 stage_a_metrics.json differs from exact recomputation")
    authorized = _call(module, "stage_a_authorizes_stage_b", recomputed_a)
    if not isinstance(authorized, bool):
        raise RegenerationError("V2 Stage-A authorization must be Boolean")
    if (trace_raw is not None) != authorized:
        raise RegenerationError("V2 conditional Stage-B presence differs from Stage-A gate")
    stage_b_validation = None
    rebuilt_b: bytes | None = None
    trace_rows: list[dict[str, Any]] | None = None
    if authorized:
        assert trace_raw is not None and stage_b_metrics_raw is not None
        trace_rows = _parse_jsonl(trace_raw, label=STAGE_B_TRACE_FILE)
        supplied_b = parse_canonical_json(stage_b_metrics_raw, label=STAGE_B_METRICS_FILE)
        if not isinstance(supplied_b, dict):
            raise RegenerationError("V2 Stage-B metrics must be an object")
        trace_authority = authority.get("stage_b_trace")
        if not isinstance(trace_authority, Mapping):
            trace_authority = _call(module, "stage_b_trace_authority")
        stage_b_validation = V1_REDUCER._validate_stage_b_trace(
            trace_rows, module=module, graph_manifest=graph,
            query_rows=query_rows, stage_a_metrics=recomputed_a,
            authority=trace_authority,
        )
        recomputed_b = _call(
            module, "recompute_stage_b_metrics", graph, tuple(query_rows),
            tuple(trace_rows), recomputed_a,
        )
        if not isinstance(recomputed_b, Mapping):
            raise RegenerationError("V2 Stage-B reducer must return a mapping")
        rebuilt_b = canonical_document_bytes(dict(recomputed_b))
        if stage_b_metrics_raw != rebuilt_b:
            raise RegenerationError("V2 stage_b_metrics.json differs from exact recomputation")

    inputs = {
        leaf: _binding(leaf, raw) for leaf, raw in json_raw.items()
    }
    inputs[QUERY_FILE] = _binding(QUERY_FILE, query_raw, rows=len(query_rows))
    inputs[STAGE_A_BELIEFS_FILE] = _binding(
        STAGE_A_BELIEFS_FILE, belief_raw, rows=len(belief_rows)
    )
    inputs["observations.npz"] = observation_evidence["binding"]
    inputs.update({"canonical_tokens.npz": token_binding, "canonical_descriptors.npz": descriptor_binding})
    inputs[STAGE_B_TRACE_FILE] = (
        None if trace_raw is None else _binding(STAGE_B_TRACE_FILE, trace_raw, rows=len(trace_rows or ()))
    )
    inputs[STAGE_B_METRICS_FILE] = (
        None if stage_b_metrics_raw is None else _binding(STAGE_B_METRICS_FILE, stage_b_metrics_raw)
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "pass": True,
        "mode": "INDEPENDENT_CANONICAL_CACHE_AND_PERSISTED_EVIDENCE_REDUCTION_ONLY",
        "metrics_module": _metrics_module_name(module),
        "source_freeze": source_freeze,
        "reducer_authority_digest": canonical_digest(authority),
        "v1_reusable_input_validation": v1_validation,
        "official_root_inventory": {
            "scientific_input_leaf_count": len(inputs) - 2 + int(trace_raw is not None) * 2,
            "allowed_leaf_names_validated": True,
        },
        "inputs": inputs,
        "cache_mapping_validation": cache_mapping_validation,
        "cache_index_validation": cache_index_validation,
        "cache_gate_validation": cache_gate_validation,
        "metrics_cache_gate_validation": module_cache_gate_validation,
        "heldout_query_validation": {
            "queries": len(heldout), "all_query_rows": len(query_rows),
            "role_counts": role_counts,
            "graphs_referenced": len({str(row["graph_id"]) for row in heldout.values()}),
        },
        "calibration_validation": calibration_validation,
        "stage_a_validation": stage_a_validation,
        "stage_a_metrics_exact_byte_equal": True,
        "recomputed_stage_a_metrics_sha256": hashlib.sha256(rebuilt_a).hexdigest(),
        "stage_a_authorizes_stage_b": authorized,
        "conditional_stage_b_present": trace_raw is not None,
        "conditional_stage_b_validation": stage_b_validation,
        "conditional_stage_b_metrics_exact_byte_equal": None if rebuilt_b is None else True,
        "recomputed_stage_b_metrics_sha256": None if rebuilt_b is None else hashlib.sha256(rebuilt_b).hexdigest(),
        "scientific_execution_counters": {
            "model_initializations": 0, "encoder_initializations": 0,
            "training_steps": 0, "inference_calls": 0, "simulator_steps": 0,
        },
    }
    _validate_no_self_digest(receipt, "external regeneration receipt")
    _reject_nonfinite(receipt, label="external regeneration receipt")
    return receipt


def _external_receipt_location(output_root: Path | str, output: Path | str) -> tuple[int, str]:
    return V1_REDUCER._external_receipt_location(output_root, output)


def emit_regeneration_receipt(
    output_root: Path | str, output: Path | str, receipt: Mapping[str, Any]
) -> bytes:
    _validate_no_self_digest(receipt, "external regeneration receipt")
    raw = canonical_document_bytes(dict(receipt))
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        try:
            fd = os.open(
                leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=parent_fd,
            )
        except FileExistsError:
            existing = _read_regular_at(parent_fd, leaf)
            if existing != raw:
                raise RegenerationError("existing external V2 receipt bytes drift")
        else:
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise OSError("short V2 receipt write")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return raw


def verify_and_emit(
    output_root: Path | str, output: Path | str, *, metrics_module: Any | None = None
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(output_root, metrics_module=metrics_module)
    emit_regeneration_receipt(output_root, output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str, output: Path | str, *, metrics_module: Any | None = None
) -> dict[str, Any]:
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        raw = _read_regular_at(parent_fd, leaf)
    finally:
        os.close(parent_fd)
    assert raw is not None
    supplied = parse_canonical_json(raw, label=str(output))
    if not isinstance(supplied, dict):
        raise RegenerationError("external V2 regeneration receipt must be an object")
    _validate_no_self_digest(supplied, "external regeneration receipt")
    rebuilt = build_regeneration_receipt(output_root, metrics_module=metrics_module)
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("external V2 receipt differs from exact rebuild")
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_RECEIPT)
    parser.add_argument("--validate-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = (
        validate_existing_regeneration_receipt(args.output_root, args.output)
        if args.validate_existing
        else verify_and_emit(args.output_root, args.output)
    )
    print(
        json.dumps(
            {
                "status": "PASS", "receipt": str(args.output),
                "document_sha256": hashlib.sha256(canonical_document_bytes(receipt)).hexdigest(),
                "stage_a_authorizes_stage_b": receipt["stage_a_authorizes_stage_b"],
            },
            sort_keys=True, separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
