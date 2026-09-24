#!/usr/bin/env python3
"""Independently reduce persisted physical graph-edge handoff evidence.

This program is deliberately usable by the system Python interpreter.  It
does not import torch, the experiment runner, an encoder, a ranker, or a
simulator.  Scientific validation and metric formulae come only from the
pure metrics module; this file owns byte custody, canonical parsing, identity
coverage, external-artifact hashing, and exact metric-byte comparison.
"""
from __future__ import annotations

import argparse
import ast
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
from typing import Any
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse only the previously frozen, torch-free custody primitives.  Importing
# this module performs no scientific or runtime-root access.
from scripts import evaluate_occluded_goal_topological_belief_v2 as CUSTODY


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
METRICS_MODULE = "lewm.safety.physical_graph_edge_handoff_qualification_v1_metrics"
REDUCER_SOURCE_PATH = "scripts/evaluate_physical_graph_edge_handoff_qualification_v1.py"
METRICS_SOURCE_PATH = "lewm/safety/physical_graph_edge_handoff_qualification_v1_metrics.py"
CONTRACT_SOURCE_PATH = "lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py"
CUSTODY_SOURCE_PATH = "scripts/evaluate_occluded_goal_topological_belief_v2.py"
BASE_CUSTODY_SOURCE_PATH = "scripts/evaluate_occluded_goal_topological_belief_v1.py"

DEFAULT_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1"
)
DEFAULT_EXTERNAL_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1_regeneration_receipt.json"
)

RECEIPT_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v1.regeneration_receipt.v1"
)

CONTRACT_FILE = "contract.json"
DOCUMENT_FILES = (
    CONTRACT_FILE,
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "state_snapshot_index.json",
    "teacher_trace_index.json",
    "edge_port_index.json",
    "waypoint_contracts.json",
    "pixel_index.json",
    "latent_index.json",
    "development_target_selection.json",
    "metrics.json",
)
LEDGER_FILES = (
    "candidate_fanout.jsonl",
    "heldout_ranker_scores.jsonl",
    "repeated_execution.jsonl",
)
PAYLOAD_FILES = (
    "state_snapshots.npz",
    "teacher_traces.npz",
    "rgb_observations.npz",
    "canonical_latents.npz",
    "candidate_traces.npz",
)
PUBLICATION_FILES = ("result.json", "result.md", "file_hashes.json")
SCIENTIFIC_FILES = DOCUMENT_FILES + LEDGER_FILES + PAYLOAD_FILES
ALL_OUTPUT_FILES = SCIENTIFIC_FILES + PUBLICATION_FILES

EXPECTED_STATE_COUNT = 64
EXPECTED_TEACHER_TRACE_COUNT = 256
EXPECTED_DEVELOPMENT_STATE_COUNT = 48
EXPECTED_HELDOUT_STATE_COUNT = 16
EXPECTED_CANDIDATES_PER_STATE = 12
EXPECTED_FANOUT_ROW_COUNT = 768
EXPECTED_HELDOUT_SCORE_ROW_COUNT = 64
EXPECTED_REPEAT_ROW_COUNT = 64
EXPECTED_RESET_FIXTURE_TRACE_COUNT = 128
EXPECTED_CANDIDATE_TRACE_COUNT = 960
EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES = 750
EXPECTED_PHYSICS_SAMPLES_PER_BRANCH = 750

_PHYSICAL_CAPTURE_MEMBERS = {
    "state_snapshots.npz": {
        "base_pose_world",
        "base_twist_world",
        "joint_position",
        "joint_velocity",
        "previous_applied_command",
    },
    "teacher_traces.npz": {
        "timestamp_s",
        "base_pose_world",
        "base_twist_world",
        "requested_command",
        "applied_command",
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "target_region_member",
    },
    "candidate_traces.npz": {
        "timestamp_s",
        "base_pose_world",
        "base_twist_world",
        "requested_command",
        "post_slew_applied_command",
        "physics_contact",
        "source_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    },
}

EXPECTED_EXTERNAL_ARTIFACT_ROLES = (
    "go2_platform_manifest",
    "go2_primitive_registry",
    "go2_ppo_policy_checkpoint",
    "go2_ppo_policy_configuration",
    "genesis_go2_urdf",
    "vjepa_encoder_checkpoint",
    "current_visual_ranker_checkpoint",
)
EXPECTED_V2_CONTEXT_LEAVES = (
    "panel_manifest.json",
    "graph_manifest.json",
    "query_ledger.jsonl",
    "stage_b_trace.jsonl",
    "stage_b_metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
)


RegenerationError = CUSTODY.RegenerationError
canonical_document_bytes = CUSTODY.canonical_document_bytes
canonical_json_bytes = CUSTODY.canonical_json_bytes
canonical_digest = CUSTODY.canonical_digest
parse_canonical_json = CUSTODY.parse_canonical_json
_parse_jsonl = CUSTODY._parse_jsonl
_read_regular_at = CUSTODY._read_regular_at
_binding = CUSTODY._binding
_safe_leaf = CUSTODY._safe_leaf
_reject_nonfinite = CUSTODY._reject_nonfinite
_npy_header = CUSTODY._npy_header
_read_exact = CUSTODY._read_exact
_array_row_digest = CUSTODY._array_row_digest


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
        raise RegenerationError(f"{label} must be a lowercase SHA-256")
    return text


def _call(module: Any, name: str, *arguments: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"pure metrics API is absent: {name}")
    try:
        return function(*arguments)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(f"pure metrics API rejected evidence in {name}: {exc}") from exc


def _call_keyword(
    module: Any, name: str, *arguments: Any, **keywords: Any
) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"pure metrics API is absent: {name}")
    try:
        return function(*arguments, **keywords)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(
            f"pure metrics API rejected evidence in {name}: {exc}"
        ) from exc


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _metrics_module_name(module: Any) -> str:
    name = getattr(module, "__name__", None)
    return name if isinstance(name, str) and name else type(module).__name__


def _validate_no_self_digest(value: Mapping[str, Any], label: str) -> None:
    forbidden = {
        "content_digest",
        "self_digest",
        "self_sha256",
        "receipt_sha256",
    }
    overlap = forbidden & set(value)
    if overlap:
        raise RegenerationError(f"{label} must not contain a self digest: {sorted(overlap)}")


def _validate_content_digest(value: Mapping[str, Any], label: str) -> None:
    declared = _sha256(value.get("content_digest"), f"{label}.content_digest")
    payload = dict(value)
    payload.pop("content_digest", None)
    # Content digests use canonical compact JSON without the document LF.
    if declared != canonical_digest(payload):
        raise RegenerationError(f"{label} content digest drift")


def _canonicalise_object(value: Any, label: str) -> tuple[bytes, dict[str, Any]]:
    """Round-trip implementation tuples into their canonical JSON value."""

    raw = canonical_document_bytes(value)
    parsed = parse_canonical_json(raw, label=label)
    if not isinstance(parsed, dict):
        raise RegenerationError(f"{label} must be a canonical JSON object")
    _reject_nonfinite(parsed, label=label)
    return raw, parsed


def _open_root(path: Path | str, label: str) -> tuple[Path, int]:
    root = Path(path)
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise RegenerationError(f"{label} must be an absolute non-symlink lexical path")
    try:
        if root.resolve(strict=True) != root:
            raise RegenerationError(f"{label} must not traverse symbolic links")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RegenerationError(f"{label} cannot be opened safely") from exc
    return root, descriptor


def _root_inventory(root_fd: int) -> list[str]:
    try:
        names = sorted(os.listdir(root_fd))
    except OSError as exc:
        raise RegenerationError("official output-root inventory cannot be read") from exc
    observed = set(names)
    scientific = set(SCIENTIFIC_FILES)
    publication = set(PUBLICATION_FILES)
    if not scientific.issubset(observed):
        raise RegenerationError(
            f"official root lacks required scientific leaves: {sorted(scientific-observed)}"
        )
    if observed & publication and not publication.issubset(observed):
        raise RegenerationError("publication leaves must be all present or all absent")
    if observed - scientific - publication:
        raise RegenerationError(
            f"official root contains unregistered leaves: {sorted(observed-scientific-publication)}"
        )
    for name in names:
        _safe_leaf(name)
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as exc:
            raise RegenerationError(f"official-root leaf cannot be inspected: {name}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegenerationError(
                f"official-root leaf must be a single-link regular file: {name}"
            )
    return names


def _load_json_at(root_fd: int, leaf: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular_at(root_fd, leaf)
    assert raw is not None
    value = parse_canonical_json(raw, label=leaf)
    if not isinstance(value, dict):
        raise RegenerationError(f"{leaf} must contain a JSON object")
    _reject_nonfinite(value, label=leaf)
    _validate_content_digest(value, leaf)
    return raw, value


def _load_jsonl_at(root_fd: int, leaf: str) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _read_regular_at(root_fd, leaf)
    assert raw is not None
    rows = _parse_jsonl(raw, label=leaf)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RegenerationError(f"{leaf} row {index} must be an object")
        _reject_nonfinite(row, label=f"{leaf}[{index}]")
    return raw, rows


def _validate_row_authority(
    *, leaf: str, rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact fields, row count, unique identities, and canonical order."""

    fields = spec.get("fields")
    identity_fields = spec.get("identity_order")
    row_count = spec.get("count")
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(value, str) and value for value in fields)
        or len(set(fields)) != len(fields)
        or not isinstance(identity_fields, list)
        or not identity_fields
        or not all(value in fields for value in identity_fields)
    ):
        raise RegenerationError(f"row authority for {leaf} is malformed")
    expected_count = _integer(row_count, f"{leaf}.row_count")
    if len(rows) != expected_count:
        raise RegenerationError(
            f"{leaf} row count drift: expected {expected_count}, observed {len(rows)}"
        )
    identities: list[tuple[Any, ...]] = []
    exact_fields = set(fields)
    for index, row in enumerate(rows):
        if set(row) != exact_fields:
            raise RegenerationError(f"{leaf} row {index} field-set drift")
        identities.append(tuple(row[field] for field in identity_fields))
    if len(set(identities)) != len(identities):
        raise RegenerationError(f"{leaf} contains duplicate row identities")
    if identities != sorted(identities):
        raise RegenerationError(f"{leaf} row identities are not in canonical order")
    return {
        "rows": len(rows),
        "identity_fields": list(identity_fields),
        "row_identity_digest": canonical_digest([list(value) for value in identities]),
        "exact_field_authority_validated": True,
        "canonical_identity_order_validated": True,
    }


def _stream_external_binding(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    expected_fields = {"path", "bytes", "sha256", "kind", "role"}
    if set(binding) != expected_fields:
        raise RegenerationError(f"{label} external binding field-set drift")
    path = Path(_string(binding["path"], f"{label}.path"))
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise RegenerationError(f"{label} path must be absolute and non-symlink")
    try:
        if path.resolve(strict=True) != path:
            raise RegenerationError(f"{label} path traverses a symbolic link")
        info_before = path.stat()
    except OSError as exc:
        raise RegenerationError(f"{label} external artifact is absent") from exc
    if not stat.S_ISREG(info_before.st_mode) or info_before.st_nlink != 1:
        raise RegenerationError(f"{label} external artifact must be a single-link regular file")
    expected_bytes = _integer(binding["bytes"], f"{label}.bytes")
    expected_sha = _sha256(binding["sha256"], f"{label}.sha256")
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            while True:
                chunk = os.read(descriptor, 4 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                observed_bytes += len(chunk)
            info_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RegenerationError(f"{label} external artifact could not be read") from exc
    stable = (
        info_before.st_dev,
        info_before.st_ino,
        info_before.st_size,
        info_before.st_mtime_ns,
    ) == (
        info_after.st_dev,
        info_after.st_ino,
        info_after.st_size,
        info_after.st_mtime_ns,
    )
    if not stable:
        raise RegenerationError(f"{label} external artifact changed while hashing")
    try:
        named_after = path.stat()
    except OSError as exc:
        raise RegenerationError(f"{label} external artifact path changed while hashing") from exc
    if (
        named_after.st_dev,
        named_after.st_ino,
        named_after.st_size,
        named_after.st_mtime_ns,
    ) != (
        info_after.st_dev,
        info_after.st_ino,
        info_after.st_size,
        info_after.st_mtime_ns,
    ):
        raise RegenerationError(f"{label} external artifact path was replaced while hashing")
    observed_sha = digest.hexdigest()
    if observed_bytes != expected_bytes or observed_sha != expected_sha:
        raise RegenerationError(f"{label} external artifact binding drift")
    return {
        "role": _string(binding["role"], f"{label}.role"),
        "kind": _string(binding["kind"], f"{label}.kind"),
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha,
    }


def _dtype_item_size(descr: str, label: str) -> int:
    """Return the byte width of a non-object scalar NPY dtype descriptor."""

    if not isinstance(descr, str) or len(descr) < 2:
        raise RegenerationError(f"{label} has an invalid dtype descriptor")
    prefix = descr[0] if descr[0] in "<>=|" else ""
    body = descr[1:] if prefix else descr
    kind, width_text = body[:1], body[1:]
    if kind == "O" or not width_text.isdigit():
        raise RegenerationError(f"{label} uses a prohibited or unsupported dtype")
    width = int(width_text)
    if width <= 0:
        raise RegenerationError(f"{label} has a zero-width dtype")
    if kind == "U":
        return width * 4
    if kind not in {"b", "i", "u", "f", "c", "S", "V", "?"}:
        raise RegenerationError(f"{label} uses an unsupported dtype kind")
    return width


def _decode_int64_offsets(payload: bytes, descr: str, label: str) -> list[int]:
    if descr not in {"<i8", "=i8", "|i8"} or len(payload) % 8:
        raise RegenerationError(f"{label} must be a little/native-endian int64 vector")
    values = [item[0] for item in struct.iter_unpack("<q", payload)]
    if not values or values[0] != 0 or any(
        left >= right for left, right in zip(values, values[1:])
    ):
        raise RegenerationError(f"{label} offsets are not monotone from zero")
    return values


def _require_finite_float_payload(payload: bytes, descr: str, label: str) -> None:
    formats = {"<f2": "<e", "<f4": "<f", "<f8": "<d"}
    format_string = formats.get(descr)
    if format_string is None:
        return
    width = int(descr[2:])
    if len(payload) % width:
        raise RegenerationError(f"{label} floating payload is misaligned")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack(format_string, payload)):
        raise RegenerationError(f"{label} contains non-finite physical evidence")


def _inspect_npz_at(
    root_fd: int,
    leaf: str,
    authority: Mapping[str, Any],
    *,
    symbols: dict[str, int],
    captured_reset_slices: dict[str, list[bytes]] | None = None,
    captured_physical_slices: dict[str, dict[str, list[bytes]]] | None = None,
) -> dict[str, Any]:
    """Inspect and hash an authority-described NPZ without NumPy.

    Each authority member has the exact fields ``descr``, ``shape`` and
    ``hash_mode``.  Hash modes are ``whole``, ``rows_axis0`` and
    ``offset_slices``; the last additionally names ``offsets_member``.  The
    returned evidence includes exact payload, row, or slice hashes for pure
    index/metrics validation, but only compact digests need be persisted in
    the external receipt.
    """

    if not isinstance(authority, Mapping):
        raise RegenerationError(f"NPZ authority field drift: {leaf}")
    member_authority = authority
    if not member_authority:
        raise RegenerationError(f"NPZ authority has no members: {leaf}")
    _safe_leaf(leaf)
    try:
        descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as exc:
        raise RegenerationError(f"NPZ cannot be opened safely: {leaf}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegenerationError(f"NPZ must be a single-link regular file: {leaf}")
        file_digest = hashlib.sha256()
        file_bytes = 0
        while True:
            block = os.read(descriptor, 4 << 20)
            if not block:
                break
            file_digest.update(block)
            file_bytes += len(block)
        if file_bytes != before.st_size:
            raise RegenerationError(f"NPZ changed or truncated while hashed: {leaf}")
        binding = {
            "path": leaf,
            "bytes": file_bytes,
            "sha256": file_digest.hexdigest(),
        }
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as file_object:
            try:
                archive = zipfile.ZipFile(file_object, "r")
            except (OSError, zipfile.BadZipFile) as exc:
                raise RegenerationError(f"invalid NPZ archive: {leaf}") from exc
            with archive:
                names = archive.namelist()
                expected_names = {f"{name}.npy" for name in member_authority}
                if set(names) != expected_names or len(names) != len(set(names)):
                    raise RegenerationError(f"NPZ member inventory drift: {leaf}")

                offsets: dict[str, list[int]] = {}
                headers: dict[str, dict[str, Any]] = {}
                for name, spec_value in member_authority.items():
                    if not isinstance(spec_value, Mapping):
                        raise RegenerationError(f"NPZ member authority must be an object: {leaf}:{name}")
                    spec = dict(spec_value)
                    allowed = {"descr", "digest_dtype", "shape", "hash_mode"}
                    if spec.get("hash_mode") == "offset_slices":
                        allowed.add("offsets_member")
                    if "slice_digest_domain" in spec:
                        allowed.add("slice_digest_domain")
                    if set(spec) != allowed:
                        raise RegenerationError(f"NPZ member authority field drift: {leaf}:{name}")
                    if "slice_digest_domain" in spec and not (
                        leaf == "state_snapshots.npz"
                        and name == "snapshot_payload_bytes"
                        and spec.get("hash_mode") == "offset_slices"
                        and spec["slice_digest_domain"]
                        == "sha256_of_exact_serialized_snapshot_bytes"
                    ):
                        raise RegenerationError(
                            f"unsupported NPZ slice digest domain: {leaf}:{name}"
                        )
                    shape = spec["shape"]
                    if (
                        not isinstance(shape, list)
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, str))
                            or (isinstance(value, int) and value < 0)
                            or (isinstance(value, str) and not value)
                            for value in shape
                        )
                    ):
                        raise RegenerationError(f"NPZ member authority shape drift: {leaf}:{name}")
                    with archive.open(f"{name}.npy", "r") as stream:
                        header, _ = _npy_header(stream, f"{leaf}:{name}.npy")
                        observed_shape = list(header["shape"])
                        if (
                            header["fortran_order"] is not False
                            or header["descr"] != spec["descr"]
                            or len(observed_shape) != len(shape)
                        ):
                            raise RegenerationError(f"NPZ member header drift: {leaf}:{name}")
                        for expected_dimension, observed_dimension in zip(shape, observed_shape):
                            if isinstance(expected_dimension, int):
                                if expected_dimension != observed_dimension:
                                    raise RegenerationError(
                                        f"NPZ member fixed dimension drift: {leaf}:{name}"
                                    )
                            else:
                                prior = symbols.setdefault(expected_dimension, observed_dimension)
                                if prior != observed_dimension:
                                    raise RegenerationError(
                                        f"NPZ symbolic dimension drift: {expected_dimension}"
                                    )
                        headers[name] = dict(header)
                        if any(
                            other.get("offsets_member") == name
                            for other in member_authority.values()
                            if isinstance(other, Mapping)
                        ):
                            item_size = _dtype_item_size(str(header["descr"]), f"{leaf}:{name}")
                            expected_bytes = math.prod(header["shape"]) * item_size
                            payload = _read_exact(stream, expected_bytes, f"{leaf}:{name}")
                            if stream.read(1):
                                raise RegenerationError(f"NPZ member has trailing bytes: {leaf}:{name}")
                            offsets[name] = _decode_int64_offsets(
                                payload, str(header["descr"]), f"{leaf}:{name}"
                            )

                inspections: dict[str, dict[str, Any]] = {}
                for name, spec_value in member_authority.items():
                    spec = dict(spec_value)
                    header = headers[name]
                    shape = tuple(header["shape"])
                    descr = str(header["descr"])
                    item_size = _dtype_item_size(descr, f"{leaf}:{name}")
                    expected_bytes = math.prod(shape) * item_size
                    with archive.open(f"{name}.npy", "r") as stream:
                        second_header, _ = _npy_header(stream, f"{leaf}:{name}.npy")
                        if second_header != header:
                            raise RegenerationError(f"NPZ member header changed: {leaf}:{name}")
                        mode = spec["hash_mode"]
                        whole = hashlib.sha256()
                        digest_dtype = _string(
                            spec["digest_dtype"], f"{leaf}:{name}.digest_dtype"
                        )
                        # Canonical array hashes use NumPy dtype.str, while the
                        # inspection separately reports the friendly dtype.
                        hash_dtype = descr
                        canonical_whole = hashlib.sha256(
                            canonical_json_bytes(
                                {
                                    "dtype": hash_dtype,
                                    "layout": "C",
                                    "shape": list(shape),
                                }
                            )
                            + b"\x00"
                        )
                        canonical_hashes: list[str] = []
                        raw_hashes: list[str] = []
                        if mode == "whole":
                            remaining = expected_bytes
                            while remaining:
                                block = stream.read(min(4 << 20, remaining))
                                if not block:
                                    raise RegenerationError(f"NPZ payload truncated: {leaf}:{name}")
                                whole.update(block)
                                canonical_whole.update(block)
                                remaining -= len(block)
                        elif mode == "rows_axis0":
                            if not shape:
                                raise RegenerationError(f"row-hashed NPZ member must have an axis: {leaf}:{name}")
                            row_shape = shape[1:]
                            row_bytes = math.prod(row_shape) * item_size
                            for index in range(shape[0]):
                                payload = _read_exact(stream, row_bytes, f"{leaf}:{name}[{index}]")
                                if leaf in {
                                    "state_snapshots.npz",
                                    "teacher_traces.npz",
                                    "candidate_traces.npz",
                                }:
                                    _require_finite_float_payload(
                                        payload, descr, f"{leaf}:{name}[{index}]"
                                    )
                                whole.update(payload)
                                canonical_whole.update(payload)
                                raw_hashes.append(hashlib.sha256(payload).hexdigest())
                                canonical_hashes.append(
                                    _array_row_digest(
                                        payload,
                                        row_shape,
                                        hash_dtype,
                                    )
                                )
                                if (
                                    captured_physical_slices is not None
                                    and name in _PHYSICAL_CAPTURE_MEMBERS.get(leaf, set())
                                ):
                                    captured_physical_slices.setdefault(leaf, {}).setdefault(
                                        name, []
                                    ).append(payload)
                        elif mode == "offset_slices":
                            offsets_member = _string(
                                spec.get("offsets_member"), f"{leaf}:{name}.offsets_member"
                            )
                            boundaries = offsets.get(offsets_member)
                            if boundaries is None or boundaries[-1] != (shape[0] if shape else 0):
                                raise RegenerationError(f"NPZ slice offsets do not cover payload: {leaf}:{name}")
                            trailing_shape = shape[1:]
                            unit_bytes = math.prod(trailing_shape) * item_size
                            for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
                                payload = _read_exact(
                                    stream,
                                    (stop - start) * unit_bytes,
                                    f"{leaf}:{name} slice {index}",
                                )
                                if leaf in {
                                    "teacher_traces.npz",
                                    "candidate_traces.npz",
                                }:
                                    _require_finite_float_payload(
                                        payload, descr, f"{leaf}:{name} slice {index}"
                                    )
                                whole.update(payload)
                                canonical_whole.update(payload)
                                slice_shape = (stop - start, *trailing_shape)
                                raw_hashes.append(hashlib.sha256(payload).hexdigest())
                                canonical_hashes.append(
                                    _array_row_digest(
                                        payload,
                                        slice_shape,
                                        hash_dtype,
                                    )
                                )
                                if (
                                    captured_reset_slices is not None
                                    and leaf == "candidate_traces.npz"
                                    and index < EXPECTED_RESET_FIXTURE_TRACE_COUNT
                                ):
                                    captured_reset_slices.setdefault(name, []).append(payload)
                                if (
                                    captured_physical_slices is not None
                                    and name in _PHYSICAL_CAPTURE_MEMBERS.get(leaf, set())
                                ):
                                    captured_physical_slices.setdefault(leaf, {}).setdefault(
                                        name, []
                                    ).append(payload)
                        else:
                            raise RegenerationError(f"unknown NPZ hash mode: {leaf}:{name}:{mode}")
                        if stream.read(1):
                            raise RegenerationError(f"NPZ member has trailing payload bytes: {leaf}:{name}")
                    whole_canonical_sha = canonical_whole.hexdigest()
                    if mode == "whole":
                        canonical_hashes = [whole_canonical_sha]
                    elif spec.get("slice_digest_domain") == (
                        "sha256_of_exact_serialized_snapshot_bytes"
                    ):
                        canonical_hashes = raw_hashes
                    inspections[name] = {
                        "descr": descr,
                        "digest_dtype": digest_dtype,
                        "shape": list(shape),
                        "c_contiguous": True,
                        "object_dtype": False,
                        "member_sha256": whole_canonical_sha,
                        "row_or_slice_sha256s": canonical_hashes,
                        "offset_values": offsets.get(name),
                    }
                after = os.fstat(descriptor)
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
                    raise RegenerationError(f"NPZ changed while inspected: {leaf}")
                try:
                    named_after = os.stat(leaf, dir_fd=root_fd, follow_symlinks=False)
                except OSError as exc:
                    raise RegenerationError(f"NPZ name changed while inspected: {leaf}") from exc
                if projection(after) != projection(named_after):
                    raise RegenerationError(f"NPZ path was replaced while inspected: {leaf}")
                return {**binding, "members": inspections}
    finally:
        os.close(descriptor)


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise RegenerationError(
            f"git {' '.join(arguments)} failed: {exc.output.decode('utf-8', 'replace')}"
        ) from exc


def _git_text(*arguments: str) -> str:
    return _git_bytes(*arguments).decode("utf-8").strip()


def _validate_external_encoder_source(
    latent_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the exact external encoder checkout bound by the latent index.

    This is source custody only: it does not import the external repository,
    deserialize a checkpoint, initialize an encoder, or open tensor outcomes.
    The pure metrics validator separately requires the frozen path/commit
    mapping from the scientific contract.
    """

    source = latent_index.get("external_encoder_source")
    if not isinstance(source, Mapping) or set(source) != {
        "repository_path",
        "commit",
        "worktree_clean",
    }:
        raise RegenerationError("latent external encoder source field-set drift")
    repository = Path(
        _string(source["repository_path"], "external encoder repository_path")
    )
    commit = _string(source["commit"], "external encoder commit")
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or source["worktree_clean"] is not True
    ):
        raise RegenerationError("latent external encoder source binding drift")
    if (
        not repository.is_absolute()
        or ".." in repository.parts
        or repository.is_symlink()
    ):
        raise RegenerationError(
            "external encoder repository must be an absolute non-symlink path"
        )
    try:
        resolved = repository.resolve(strict=True)
        info = repository.stat()
    except OSError as exc:
        raise RegenerationError("external encoder repository is absent") from exc
    if resolved != repository or not stat.S_ISDIR(info.st_mode):
        raise RegenerationError(
            "external encoder repository must be a real ordinary directory"
        )

    def run_git(*arguments: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *arguments],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RegenerationError(
                "external encoder source Git validation failed: "
                + exc.output.decode("utf-8", "replace")
            ) from exc

    observed_commit = run_git("rev-parse", "HEAD").decode("utf-8").strip()
    dirty = run_git("status", "--porcelain=v1", "--untracked-files=all")
    if observed_commit != commit or dirty != b"":
        raise RegenerationError("external encoder source checkout custody drift")

    preprocessing = latent_index.get("preprocessing_authority")
    if not isinstance(preprocessing, Mapping):
        raise RegenerationError("latent preprocessing authority must be an object")
    required_preprocessing = {
        "input_shape",
        "input_dtype",
        "input_layout",
        "transport",
        "temporary_png_retained",
        "output_type",
        "output_shape",
        "output_dtype",
        "finite_required",
        "output_projection",
        "preprocessed_tensor_sha256_persisted_per_canonical_pixel",
    }
    if set(preprocessing) != required_preprocessing:
        raise RegenerationError("latent preprocessing authority field-set drift")
    if (
        preprocessing["input_shape"] != [168, 224, 3]
        or preprocessing["input_dtype"] != "uint8"
        or preprocessing["input_layout"] != "C_CONTIGUOUS_RGB"
        or preprocessing["temporary_png_retained"] is not False
        or preprocessing["output_type"] != "torch.Tensor"
        or preprocessing["output_shape"] != [3, 384, 512]
        or preprocessing["output_dtype"] != "float32"
        or preprocessing["finite_required"] is not True
        or preprocessing[
            "preprocessed_tensor_sha256_persisted_per_canonical_pixel"
        ]
        is not True
    ):
        raise RegenerationError("latent preprocessing authority semantic drift")

    preprocessed_hashes = []
    records = latent_index.get("records")
    if not isinstance(records, list):
        raise RegenerationError("latent records must be a list")
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise RegenerationError(f"latent record {index} must be an object")
        preprocessed_hashes.append(
            _sha256(
                row.get("preprocessed_tensor_sha256"),
                f"latent record {index} preprocessed tensor SHA",
            )
        )
    return {
        "repository_path": str(repository),
        "commit": observed_commit,
        "worktree_clean": True,
        "preprocessing_authority_digest": canonical_digest(preprocessing),
        "preprocessed_tensor_hash_count": len(preprocessed_hashes),
        "preprocessed_tensor_hashes_valid": True,
        "model_or_encoder_opened": False,
    }


def _validate_runtime_environments(
    module: Any,
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate every observed runtime mapping and its row-level bindings."""

    physical = _call(
        module,
        "validate_physical_runtime_environment",
        documents["panel_manifest"].get("physical_runtime_environment"),
    )
    encoder = _call_keyword(
        module,
        "validate_visual_runtime_environment",
        documents["latent_index"].get("encoder_runtime_environment"),
        runtime_role="encoder",
    )
    ranker = _call_keyword(
        module,
        "validate_visual_runtime_environment",
        documents["development_target_selection"].get(
            "ranker_runtime_environment"
        ),
        runtime_role="ranker",
    )
    for role, row in (("physical", physical), ("encoder", encoder), ("ranker", ranker)):
        if not isinstance(row, Mapping) or row.get("fake_runtime") is not False:
            raise RegenerationError(
                f"official {role} runtime evidence is absent or marked fake"
            )

    physical_digest = _sha256(
        physical.get("runtime_core_sha256"), "physical runtime core SHA"
    )
    physical_rows = [
        *ledgers["candidate_fanout"],
        *ledgers["repeated_execution"],
    ]
    if any(
        row.get("physical_runtime_core_sha256") != physical_digest
        for row in physical_rows
    ):
        raise RegenerationError(
            "candidate or repeat row is not bound to the observed physical runtime"
        )
    ranker_digest = _call(module, "runtime_environment_sha256", ranker)
    ranker_digest = _sha256(ranker_digest, "ranker runtime environment SHA")
    if any(
        row.get("ranker_runtime_environment_sha256") != ranker_digest
        for row in ledgers["heldout_ranker_scores"]
    ):
        raise RegenerationError(
            "held-out ranker row is not bound to the observed ranker runtime"
        )
    encoder_digest = _sha256(
        _call(module, "runtime_environment_sha256", encoder),
        "encoder runtime environment SHA",
    )
    return {
        "physical_runtime_core_sha256": physical_digest,
        "qualification_runtime_binding_count": len(
            physical["qualification_runtime_sha256s"]
        ),
        "selected_snapshot_runtime_binding_count": len(
            physical["selected_snapshot_runtime_sha256s"]
        ),
        "fanout_and_repeat_runtime_binding_count": len(physical_rows),
        "encoder_runtime_environment_sha256": encoder_digest,
        "ranker_runtime_environment_sha256": ranker_digest,
        "heldout_ranker_runtime_binding_count": len(
            ledgers["heldout_ranker_scores"]
        ),
        "all_observed_runtimes_real_and_exact": True,
        "python_torch_genesis_quadrants_and_device_persisted": True,
    }


def _safe_repo_relative(relative: str) -> Path:
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
    ):
        raise RegenerationError(f"unsafe repository-relative source path: {relative!r}")
    resolved = (REPO_ROOT / path).resolve(strict=True)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise RegenerationError(f"source path escapes repository: {relative}") from exc
    if resolved != REPO_ROOT / path:
        raise RegenerationError(f"source path traverses a symbolic link: {relative}")
    return resolved


def _observe_source_freeze(module: Any) -> dict[str, Any]:
    head = _git_text("rev-parse", "HEAD")
    dirty = _git_bytes("status", "--porcelain=v1", "--untracked-files=all")
    source_paths = (
        REDUCER_SOURCE_PATH,
        CONTRACT_SOURCE_PATH,
        METRICS_SOURCE_PATH,
        CUSTODY_SOURCE_PATH,
        BASE_CUSTODY_SOURCE_PATH,
    )
    sources: dict[str, dict[str, Any]] = {}
    for relative in source_paths:
        working_path = _safe_repo_relative(relative)
        working = working_path.read_bytes()
        frozen = _git_bytes("show", f"{head}:{relative}")
        if working != frozen:
            raise RegenerationError(f"working source differs from HEAD: {relative}")
        sources[relative] = {
            "path": relative,
            "bytes": len(working),
            "sha256": hashlib.sha256(working).hexdigest(),
        }
    return {
        "head_commit": head,
        "worktree_clean": dirty == b"",
        "sources_exactly_equal_head": True,
        "sources": sources,
        "metrics_module": _metrics_module_name(module),
    }


def _validate_source_freeze(
    value: Mapping[str, Any], *, expected_commit: str
) -> dict[str, Any]:
    if set(value) != {
        "head_commit",
        "worktree_clean",
        "sources_exactly_equal_head",
        "sources",
        "metrics_module",
    }:
        raise RegenerationError("source-freeze observation field-set drift")
    if value["head_commit"] != expected_commit:
        raise RegenerationError("reducer source commit differs from runtime contract")
    if value["worktree_clean"] is not True or value["sources_exactly_equal_head"] is not True:
        raise RegenerationError("reducer requires a clean exact source freeze")
    sources = value["sources"]
    if not isinstance(sources, Mapping) or set(sources) != {
        REDUCER_SOURCE_PATH,
        CONTRACT_SOURCE_PATH,
        METRICS_SOURCE_PATH,
        CUSTODY_SOURCE_PATH,
        BASE_CUSTODY_SOURCE_PATH,
    }:
        raise RegenerationError("source-freeze source inventory drift")
    for relative, binding in sources.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "bytes", "sha256"}:
            raise RegenerationError(f"source binding field drift: {relative}")
        if binding["path"] != relative:
            raise RegenerationError(f"source path drift: {relative}")
        _integer(binding["bytes"], f"source {relative}.bytes")
        _sha256(binding["sha256"], f"source {relative}.sha256")
    return json.loads(json.dumps(value))


def _validate_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "reducer_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("reducer_authority() must return an object")
    authority = dict(authority)
    _validate_content_digest(authority, "reducer authority")
    if (
        authority.get("schema")
        != "physical_graph_edge_handoff_qualification_v1.reducer_authority.v1"
        or authority.get("experiment_id") != EXPERIMENT_ID
    ):
        raise RegenerationError("reducer authority identity drift")
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
    }
    if not required.issubset(authority):
        raise RegenerationError(
            f"reducer authority lacks required sections: {sorted(required-set(authority))}"
        )
    evidence_keys = authority["evidence_keys"]
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
    if not isinstance(evidence_keys, list) or set(evidence_keys) != expected_evidence:
        raise RegenerationError("reducer evidence-key authority drift")
    runtime_paths = authority["runtime_paths"]
    if (
        not isinstance(runtime_paths, Mapping)
        or set(runtime_paths.values()) != set(ALL_OUTPUT_FILES)
        or len(runtime_paths) != len(ALL_OUTPUT_FILES)
    ):
        raise RegenerationError("runtime output-path authority drift")
    npz_authorities = authority["npz_authorities"]
    if not isinstance(npz_authorities, Mapping) or set(npz_authorities) != set(PAYLOAD_FILES):
        raise RegenerationError("NPZ payload authority inventory drift")
    documents = authority["documents"]
    if not isinstance(documents, Mapping) or set(documents) != {
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
        raise RegenerationError("document authority inventory drift")
    expected_document_counts = {
        "panel_manifest": EXPECTED_STATE_COUNT,
        "split_manifest": EXPECTED_STATE_COUNT,
        "graph_manifest": EXPECTED_STATE_COUNT,
        "state_snapshot_index": EXPECTED_STATE_COUNT,
        "teacher_trace_index": EXPECTED_TEACHER_TRACE_COUNT,
        "edge_port_index": EXPECTED_STATE_COUNT,
        "waypoint_contracts": EXPECTED_STATE_COUNT * 3,
        "pixel_index": EXPECTED_STATE_COUNT,
        "latent_index": "unique_pixel_count",
    }
    for name, expected_count in expected_document_counts.items():
        if not isinstance(documents[name], Mapping) or documents[name].get(
            "count"
        ) != expected_count:
            raise RegenerationError(f"document authority row-count drift: {name}")
    panel_authority = documents["panel_manifest"]
    if (
        panel_authority.get("qualification_count")
        != EXPECTED_TEACHER_TRACE_COUNT
    ):
        raise RegenerationError("prospective teacher qualification count drift")
    selection_authority = documents["development_target_selection"]
    if (
        not isinstance(selection_authority, Mapping)
        or selection_authority.get("state_target_count")
        != EXPECTED_DEVELOPMENT_STATE_COUNT * 3
        or selection_authority.get("summary_count") != 3
    ):
        raise RegenerationError("development target-selection authority count drift")
    ledgers = authority["ledgers"]
    if not isinstance(ledgers, Mapping) or set(ledgers) != {
        "candidate_fanout",
        "heldout_ranker_scores",
        "repeated_execution",
    }:
        raise RegenerationError("ledger authority inventory drift")
    expected_ledger_counts = {
        "candidate_fanout": EXPECTED_FANOUT_ROW_COUNT,
        "heldout_ranker_scores": EXPECTED_HELDOUT_SCORE_ROW_COUNT,
        "repeated_execution": EXPECTED_REPEAT_ROW_COUNT,
    }
    for name, count in expected_ledger_counts.items():
        if ledgers[name].get("count") != count:
            raise RegenerationError(f"ledger authority row-count drift: {name}")
    if authority["external_artifact_roles"] != list(EXPECTED_EXTERNAL_ARTIFACT_ROLES):
        raise RegenerationError("external artifact-role authority drift")
    runtime_authority = authority["runtime_environment_authority"]
    if not isinstance(runtime_authority, Mapping) or set(runtime_authority) != {
        "digest_domain",
        "physical",
        "encoder",
        "ranker",
        "official_projections",
    }:
        raise RegenerationError("runtime environment authority field-set drift")
    for role in ("physical", "encoder", "ranker"):
        row = runtime_authority[role]
        if not isinstance(row, Mapping) or not isinstance(row.get("fields"), list):
            raise RegenerationError(f"{role} runtime environment authority drift")
    if authority["trace_index_ranges"] != {
        "reset_fixture": [0, EXPECTED_RESET_FIXTURE_TRACE_COUNT],
        "candidate_fanout": [
            EXPECTED_RESET_FIXTURE_TRACE_COUNT,
            EXPECTED_RESET_FIXTURE_TRACE_COUNT + EXPECTED_FANOUT_ROW_COUNT,
        ],
        "repeated_execution": [
            EXPECTED_RESET_FIXTURE_TRACE_COUNT + EXPECTED_FANOUT_ROW_COUNT,
            EXPECTED_CANDIDATE_TRACE_COUNT,
        ],
    }:
        raise RegenerationError("candidate trace-index range authority drift")
    if (
        authority["reset_fixture_physics_samples"]
        != EXPECTED_RESET_FIXTURE_PHYSICS_SAMPLES
        or authority["branch_physics_samples"]
        != EXPECTED_PHYSICS_SAMPLES_PER_BRANCH
    ):
        raise RegenerationError("trace duration authority drift")
    physical = authority["physical_trace_reduction_authority"]
    if not isinstance(physical, Mapping):
        raise RegenerationError("physical trace-reduction authority must be an object")
    physical = dict(physical)
    _validate_content_digest(physical, "physical trace-reduction authority")
    if (
        physical.get("schema")
        != "physical_graph_edge_handoff_qualification_v1.physical_trace_reduction_authority.v1"
        or physical.get("experiment_id") != EXPERIMENT_ID
        or physical.get("physics_dt_s") != 0.002
        or physical.get("dwell_samples") != 100
    ):
        raise RegenerationError("physical trace-reduction authority identity drift")
    geometry_rows = physical.get("specs")
    if (
        not isinstance(geometry_rows, list)
        or len(geometry_rows) != EXPECTED_TEACHER_TRACE_COUNT
        or len(
            {
                row.get("candidate_spec_id")
                for row in geometry_rows
                if isinstance(row, Mapping)
            }
        )
        != EXPECTED_TEACHER_TRACE_COUNT
    ):
        raise RegenerationError("physical trace geometry population drift")
    return json.loads(json.dumps(authority))


_DOCUMENT_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "panel_manifest": ("state_id",),
    "split_manifest": ("state_id",),
    "graph_manifest": ("state_id",),
    "state_snapshot_index": ("state_id",),
    "teacher_trace_index": ("state_id",),
    "edge_port_index": ("state_id",),
    "waypoint_contracts": ("state_id", "target_id"),
    "pixel_index": ("state_id",),
    "latent_index": ("canonical_pixel_index",),
}


def _validate_document_authorities(
    documents: Mapping[str, Mapping[str, Any]], authority: Mapping[str, Any]
) -> dict[str, Any]:
    specs = authority["documents"]
    panel_states = documents["panel_manifest"].get("states")
    if not isinstance(panel_states, list):
        raise RegenerationError("panel states must be available before authority validation")
    panel_state_ids = [row.get("state_id") for row in panel_states if isinstance(row, Mapping)]
    target_ids = authority.get("target_ids")
    if not isinstance(target_ids, list) or len(target_ids) != 3:
        raise RegenerationError("target identity authority drift")
    validations: dict[str, Any] = {}
    for name, fallback_identity_fields in _DOCUMENT_IDENTITY_FIELDS.items():
        document = documents[name]
        spec = specs[name]
        if not isinstance(spec, Mapping):
            raise RegenerationError(f"document authority must be an object: {name}")
        identity_values = spec.get("identity_order", list(fallback_identity_fields))
        if (
            not isinstance(identity_values, list)
            or not identity_values
            or not all(isinstance(field, str) and field for field in identity_values)
        ):
            raise RegenerationError(f"document identity authority drift: {name}")
        identity_fields = tuple(identity_values)
        if set(document) != set(spec.get("root", ())):
            raise RegenerationError(f"document root field-set drift: {name}")
        container = _string(spec.get("container"), f"{name}.container")
        rows = document.get(container)
        if not isinstance(rows, list):
            raise RegenerationError(f"document row container must be a list: {name}")
        count = spec.get("count")
        if count == "unique_pixel_count":
            count = documents["pixel_index"].get("unique_pixel_count")
        expected_count = _integer(count, f"{name}.count")
        if len(rows) != expected_count:
            raise RegenerationError(f"document row-count drift: {name}")
        expected_fields = set(spec.get("row", ()))
        identities: list[tuple[Any, ...]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise RegenerationError(f"document row field-set drift: {name}[{index}]")
            identities.append(tuple(row[field] for field in identity_fields))
        if len(set(identities)) != len(identities):
            raise RegenerationError(f"document row identities duplicate: {name}")
        if name == "waypoint_contracts":
            expected_identities = [
                (state_id, target_id)
                for state_id in panel_state_ids
                for target_id in target_ids
            ]
        elif name == "teacher_trace_index" and expected_count != len(panel_state_ids):
            pool = documents["panel_manifest"].get("prospective_pool_selection")
            qualification_rows = (
                pool.get("qualification_rows") if isinstance(pool, Mapping) else None
            )
            if not isinstance(qualification_rows, list):
                raise RegenerationError(
                    "teacher identities lack prospective qualification projection"
                )
            expected_identities = [
                tuple(row.get(field) for field in identity_fields)
                for row in qualification_rows
                if isinstance(row, Mapping)
            ]
        elif name == "latent_index":
            expected_identities = [(index,) for index in range(expected_count)]
        else:
            expected_identities = [(state_id,) for state_id in panel_state_ids]
        if identities != expected_identities:
            raise RegenerationError(f"document row identity order drift: {name}")
        validations[name] = {
            "rows": len(rows),
            "identity_fields": list(identity_fields),
            "row_identity_digest": canonical_digest([list(value) for value in identities]),
            "exact_field_authority_validated": True,
            "canonical_identity_order_validated": True,
        }

    panel_spec = specs["panel_manifest"]
    if "qualification_row" in panel_spec:
        pool = documents["panel_manifest"].get("prospective_pool_selection")
        if not isinstance(pool, Mapping):
            raise RegenerationError("panel lacks prospective qualification evidence")
        qualification_rows = pool.get("qualification_rows")
        expected_count = _integer(
            panel_spec.get("qualification_count"),
            "panel qualification_count",
        )
        expected_fields = set(panel_spec.get("qualification_row", ()))
        identity_fields = panel_spec.get("qualification_identity_order")
        if (
            not isinstance(qualification_rows, list)
            or len(qualification_rows) != expected_count
            or not isinstance(identity_fields, list)
            or not identity_fields
        ):
            raise RegenerationError("panel qualification cardinality/authority drift")
        identities: list[tuple[Any, ...]] = []
        for index, row in enumerate(qualification_rows):
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise RegenerationError(f"panel qualification row {index} field drift")
            identities.append(tuple(row[field] for field in identity_fields))
        if len(set(identities)) != expected_count:
            raise RegenerationError("panel qualification identities are not unique")
        validations["panel_qualification"] = {
            "rows": expected_count,
            "identity_fields": list(identity_fields),
            "row_identity_digest": canonical_digest(
                [list(identity) for identity in identities]
            ),
            "exact_field_authority_validated": True,
            "unique_identity_coverage_validated": True,
        }

    selection = documents["development_target_selection"]
    selection_spec = specs["development_target_selection"]
    if set(selection) != set(selection_spec.get("root", ())):
        raise RegenerationError("development target-selection root field drift")
    nested_validations: dict[str, Any] = {}
    development_state_ids = [
        row["state_id"] for row in panel_states if row.get("role") == "DEVELOPMENT"
    ]
    for prefix, fallback_identity_fields in (
        ("state_target", ("state_id", "target_id")),
        ("summary", ("target_id",)),
    ):
        container = _string(
            selection_spec.get(f"{prefix}_container"),
            f"development_target_selection.{prefix}_container",
        )
        rows = selection.get(container)
        expected_count = _integer(
            selection_spec.get(f"{prefix}_count"),
            f"development_target_selection.{prefix}_count",
        )
        expected_fields = set(selection_spec.get(f"{prefix}_row", ()))
        identity_values = selection_spec.get(
            f"{prefix}_identity_order", list(fallback_identity_fields)
        )
        if not isinstance(identity_values, list) or not identity_values:
            raise RegenerationError(
                f"development selection {prefix} identity authority drift"
            )
        identity_fields = tuple(identity_values)
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise RegenerationError(f"development selection {prefix} row-count drift")
        identities = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise RegenerationError(
                    f"development selection {prefix} field drift at row {index}"
                )
            identities.append(tuple(row[field] for field in identity_fields))
        expected_identities = (
            [
                (state_id, target_id)
                for state_id in development_state_ids
                for target_id in target_ids
            ]
            if prefix == "state_target"
            else [(target_id,) for target_id in target_ids]
        )
        if len(set(identities)) != len(identities) or identities != expected_identities:
            raise RegenerationError(f"development selection {prefix} identity drift")
        nested_validations[prefix] = {
            "rows": len(rows),
            "row_identity_digest": canonical_digest([list(value) for value in identities]),
        }
    validations["development_target_selection"] = nested_validations
    return validations


def _validate_ledger_authorities(
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
    documents: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    panel_states = documents["panel_manifest"]["states"]
    state_ids = [row["state_id"] for row in panel_states]
    heldout_state_ids = [
        row["state_id"] for row in panel_states if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    condition_ids = authority.get("heldout_condition_ids")
    repeat_branch_ids = authority.get("repeat_branch_ids")
    if not isinstance(condition_ids, list) or len(condition_ids) != 4:
        raise RegenerationError("heldout condition authority drift")
    if not isinstance(repeat_branch_ids, list) or len(repeat_branch_ids) != 2:
        raise RegenerationError("repeat branch authority drift")
    validations = {
        name: _validate_row_authority(
            leaf=f"{name}.jsonl", rows=rows, spec=authority["ledgers"][name]
        )
        for name, rows in ledgers.items()
    }
    observed_fanout = [
        (row["state_id"], row["candidate_index"])
        for row in ledgers["candidate_fanout"]
    ]
    expected_fanout = [
        (state_id, candidate_index)
        for state_id in state_ids
        for candidate_index in range(EXPECTED_CANDIDATES_PER_STATE)
    ]
    if observed_fanout != expected_fanout:
        raise RegenerationError("candidate fanout lacks exact state/candidate cross-product")
    observed_scores = [
        (row["state_id"], row["condition_id"])
        for row in ledgers["heldout_ranker_scores"]
    ]
    expected_scores = [
        (state_id, condition_id)
        for state_id in heldout_state_ids
        for condition_id in condition_ids
    ]
    if observed_scores != expected_scores:
        raise RegenerationError("heldout scores lack exact state/condition cross-product")
    observed_repeats = [
        (row["state_id"], row["branch_selector_id"], row["repeat_index"])
        for row in ledgers["repeated_execution"]
    ]
    expected_repeats = [
        (state_id, branch_id, repeat_index)
        for state_id in heldout_state_ids
        for branch_id in repeat_branch_ids
        for repeat_index in range(2)
    ]
    if observed_repeats != expected_repeats:
        raise RegenerationError("repeat ledger lacks exact state/branch/repeat cross-product")
    return validations


def _require_file_binding(
    value: Any, inspection: Mapping[str, Any], label: str
) -> None:
    expected = {field: inspection[field] for field in ("path", "bytes", "sha256")}
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RegenerationError(f"{label} does not bind the inspected NPZ bytes")


def _require_slice_binding(
    value: Any,
    inspection: Mapping[str, Any],
    *,
    member: str,
    slice_index: int,
    start: int,
    stop: int,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "file_path",
        "file_sha256",
        "member",
        "start",
        "stop",
        "slice_sha256",
    }:
        raise RegenerationError(f"{label} slice-binding field drift")
    digests = inspection["members"][member]["row_or_slice_sha256s"]
    if not 0 <= slice_index < len(digests):
        raise RegenerationError(f"{label} slice index is outside NPZ evidence")
    expected = {
        "file_path": inspection["path"],
        "file_sha256": inspection["sha256"],
        "member": member,
        "start": start,
        "stop": stop,
        "slice_sha256": digests[slice_index],
    }
    if dict(value) != expected:
        raise RegenerationError(f"{label} slice binding differs from NPZ bytes")


def _require_trace_hash_map(
    value: Any,
    inspection: Mapping[str, Any],
    *,
    trace_index: int,
    label: str,
) -> None:
    expected = {
        member: details["row_or_slice_sha256s"][trace_index]
        for member, details in inspection["members"].items()
        if member != "trace_offsets"
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RegenerationError(f"{label} trace-member hashes differ from NPZ bytes")


def _float64_rows(
    payload: bytes, width: int, sample_count: int, label: str
) -> list[tuple[float, ...]]:
    expected = sample_count * width * 8
    if len(payload) != expected:
        raise RegenerationError(f"{label} float64 payload length drift")
    values = struct.unpack(f"<{sample_count * width}d", payload)
    if any(not math.isfinite(value) for value in values):
        raise RegenerationError(f"{label} contains non-finite trace values")
    return [
        tuple(values[offset : offset + width])
        for offset in range(0, len(values), width)
    ]


def _uint8_values(payload: bytes, sample_count: int, label: str) -> tuple[int, ...]:
    if len(payload) != sample_count:
        raise RegenerationError(f"{label} uint8 payload length drift")
    if any(value not in (0, 1) for value in payload):
        raise RegenerationError(f"{label} is not binary physics evidence")
    return tuple(payload)


def _segment_intersection_fraction(
    start: Sequence[float],
    stop: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    px, py = float(start[0]), float(start[1])
    rx, ry = float(stop[0]) - px, float(stop[1]) - py
    qx, qy = float(first[0]), float(first[1])
    sx, sy = float(second[0]) - qx, float(second[1]) - qy
    cross = rx * sy - ry * sx
    if abs(cross) <= 1.0e-12:
        return None
    qpx, qpy = qx - px, qy - py
    along_motion = (qpx * sy - qpy * sx) / cross
    along_segment = (qpx * ry - qpy * rx) / cross
    if (
        -1.0e-12 <= along_motion <= 1.0 + 1.0e-12
        and -1.0e-12 <= along_segment <= 1.0 + 1.0e-12
    ):
        return float(along_motion)
    return None


def _normalised_opening(
    segment: Sequence[Sequence[float]], normal: Sequence[float], label: str
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    if len(segment) != 2 or any(len(point) != 2 for point in segment) or len(normal) != 2:
        raise RegenerationError(f"{label} opening geometry shape drift")
    first = (float(segment[0][0]), float(segment[0][1]))
    second = (float(segment[1][0]), float(segment[1][1]))
    nx, ny = float(normal[0]), float(normal[1])
    if any(not math.isfinite(value) for value in (*first, *second, nx, ny)):
        raise RegenerationError(f"{label} opening geometry is non-finite")
    width = math.hypot(second[0] - first[0], second[1] - first[1])
    norm = math.hypot(nx, ny)
    if width <= 1.0e-12 or norm <= 1.0e-12:
        raise RegenerationError(f"{label} opening geometry is degenerate")
    return first, second, (nx / norm, ny / norm), width


def _point_in_polygon_inclusive(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    tolerance: float,
    label: str,
) -> bool:
    """Independently reproduce the frozen boundary-inclusive polygon rule."""

    if (
        len(point) < 2
        or len(polygon) < 3
        or tolerance < 0.0
        or not math.isfinite(tolerance)
    ):
        raise RegenerationError(f"{label} polygon-membership authority drift")
    vertices: list[tuple[float, float]] = []
    for index, vertex in enumerate(polygon):
        if len(vertex) != 2:
            raise RegenerationError(f"{label} polygon vertex {index} shape drift")
        values = (float(vertex[0]), float(vertex[1]))
        if any(not math.isfinite(value) for value in values):
            raise RegenerationError(f"{label} polygon is non-finite")
        vertices.append(values)
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise RegenerationError(f"{label} point is non-finite")
    inside = False
    for index, start in enumerate(vertices):
        stop = vertices[(index + 1) % len(vertices)]
        dx, dy = stop[0] - start[0], stop[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 0.0:
            raise RegenerationError(f"{label} polygon has a zero-length edge")
        projection = max(
            0.0,
            min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_squared),
        )
        nearest_x = start[0] + projection * dx
        nearest_y = start[1] + projection * dy
        if math.hypot(x - nearest_x, y - nearest_y) <= tolerance:
            return True
        if (start[1] > y) != (stop[1] > y):
            crossing_x = start[0] + (y - start[1]) * dx / dy
            if x < crossing_x:
                inside = not inside
    return inside


def _validate_membership_against_geometry(
    poses: Sequence[Sequence[float]],
    observed: Sequence[int],
    polygon: Sequence[Sequence[float]],
    tolerance: float,
    label: str,
) -> None:
    expected = tuple(
        int(_point_in_polygon_inclusive(pose, polygon, tolerance, label))
        for pose in poses
    )
    if tuple(observed) != expected:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(observed, expected))
            if left != right
        )
        raise RegenerationError(
            f"{label} membership differs from raw pose/geometry at sample {mismatch}"
        )


def _first_transverse_crossing(
    poses: Sequence[Sequence[float]],
    segment: Sequence[Sequence[float]],
    normal: Sequence[float],
    label: str,
    tolerance: float = 0.0,
) -> dict[str, Any] | None:
    first, second, _normalised, width = _normalised_opening(segment, normal, label)
    raw_normal_length = math.hypot(float(normal[0]), float(normal[1]))
    if abs(raw_normal_length - 1.0) > tolerance:
        raise RegenerationError(f"{label} opening normal is not unit length")
    # The frozen formula first proves unit length under the registered
    # tolerance and then uses the persisted normal components themselves.
    # Do not silently renormalize a near-unit public geometry vector.
    unit_normal = (float(normal[0]), float(normal[1]))
    midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    tangent = (
        (second[0] - first[0]) / width,
        (second[1] - first[1]) / width,
    )
    for sample_after in range(1, len(poses)):
        before = poses[sample_after - 1]
        after = poses[sample_after]
        d0 = (
            (float(before[0]) - first[0]) * unit_normal[0]
            + (float(before[1]) - first[1]) * unit_normal[1]
        )
        d1 = (
            (float(after[0]) - first[0]) * unit_normal[0]
            + (float(after[1]) - first[1]) * unit_normal[1]
        )
        denominator = d1 - d0
        if not (d0 <= tolerance and d1 > tolerance and denominator > 0.0):
            continue
        raw_fraction = -d0 / denominator
        if raw_fraction < -tolerance or raw_fraction > 1.0 + tolerance:
            continue
        fraction = min(1.0, max(0.0, raw_fraction))
        dx = float(after[0]) - float(before[0])
        dy = float(after[1]) - float(before[1])
        normal_dot = dx * unit_normal[0] + dy * unit_normal[1]
        if normal_dot <= 0.0:
            continue
        point = (
            float(poses[sample_after - 1][0]) + fraction * dx,
            float(poses[sample_after - 1][1]) + fraction * dy,
        )
        lateral_fraction = (
            (point[0] - first[0]) * tangent[0]
            + (point[1] - first[1]) * tangent[1]
        ) / width
        lateral_tolerance = tolerance / width
        if (
            lateral_fraction < -lateral_tolerance
            or lateral_fraction > 1.0 + lateral_tolerance
        ):
            continue
        lateral_fraction = min(1.0, max(0.0, lateral_fraction))
        lateral_m = (
            lateral_fraction - 0.5
        )
        lateral_m *= width
        return {
            "sample_before": sample_after - 1,
            "sample_after": sample_after,
            "fraction": fraction,
            "point_world_xy": [point[0], point[1]],
            "normal_dot_displacement_m": normal_dot,
            "displacement_world_xy": [dx, dy],
            "direction_heading_world_rad": math.atan2(dy, dx),
            "lateral_coordinate_m": lateral_m,
            "lateral_fraction": lateral_fraction,
            "opening_width_m": width,
            "unit_normal": [unit_normal[0], unit_normal[1]],
            "midpoint_world_xy": [midpoint[0], midpoint[1]],
            "tangent": [tangent[0], tangent[1]],
        }
    return None


def _first_source_boundary_exit(
    poses: Sequence[Sequence[float]],
    source_membership: Sequence[int],
    selected_edge_membership: Sequence[int],
    polygon: Sequence[Sequence[float]],
    label: str,
) -> dict[str, Any] | None:
    if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
        raise RegenerationError(f"{label} source polygon is degenerate")
    if not (
        len(poses) == len(source_membership) == len(selected_edge_membership)
    ):
        raise RegenerationError(f"{label} source-exit evidence is misaligned")
    for sample_after in range(1, len(poses)):
        if not (
            source_membership[sample_after - 1] == 1
            and source_membership[sample_after] == 0
            and selected_edge_membership[sample_after] == 1
        ):
            continue
        fractions = [
            fraction
            for first, second in zip(polygon, [*polygon[1:], polygon[0]])
            if (
                fraction := _segment_intersection_fraction(
                    poses[sample_after - 1], poses[sample_after], first, second
                )
            )
            is not None
        ]
        if not fractions:
            raise RegenerationError(
                f"{label} membership exit has no geometric boundary crossing"
            )
        fraction = min(fractions)
        dx = float(poses[sample_after][0]) - float(poses[sample_after - 1][0])
        dy = float(poses[sample_after][1]) - float(poses[sample_after - 1][1])
        return {
            "sample_before": sample_after - 1,
            "sample_after": sample_after,
            "fraction": fraction,
            "point_world_xy": [
                float(poses[sample_after - 1][0]) + fraction * dx,
                float(poses[sample_after - 1][1]) + fraction * dy,
            ],
            "motion_heading_world_rad": math.atan2(dy, dx),
        }
    return None


def _first_membership_transition(
    values: Sequence[int], before: int, after: int, label: str
) -> int | None:
    if not values or any(value not in {0, 1} for value in values):
        raise RegenerationError(f"{label} membership evidence is invalid")
    return next(
        (
            index
            for index in range(1, len(values))
            if values[index - 1] == before and values[index] == after
        ),
        None,
    )


def _quaternion_roll_pitch_yaw(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 4:
        raise RegenerationError("quaternion shape drift")
    x, y, z, w = (float(item) for item in value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, pitch_argument) if abs(
        pitch_argument
    ) >= 1.0 else math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _yaw_from_xyzw(value: Sequence[float]) -> float:
    x, y, z, w = value
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_difference(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _maximum_component_error(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    indices: Sequence[int] | None = None,
) -> float:
    selected = indices if indices is not None else range(len(left[0]))
    return max(
        abs(left_row[index] - right_row[index])
        for left_row, right_row in zip(left, right)
        for index in selected
    )


def _validate_reset_trace_pairs(
    state_snapshot_index: Mapping[str, Any],
    candidate_inspection: Mapping[str, Any],
    captured: Mapping[str, Sequence[bytes]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_authority = authority["reset_trace_pair_comparison_authority"]
    if not isinstance(comparison_authority, Mapping):
        raise RegenerationError("reset trace-pair comparison authority drift")
    sample_count = _integer(
        comparison_authority.get("physics_samples_per_trial"),
        "reset physics_samples_per_trial",
        minimum=1,
    )
    if sample_count != authority["reset_fixture_physics_samples"]:
        raise RegenerationError("reset trace comparison duration authority drift")
    exact_members = comparison_authority.get("exact_members")
    tolerances = comparison_authority.get("samplewise_tolerances")
    if (
        not isinstance(exact_members, list)
        or not exact_members
        or not isinstance(tolerances, Mapping)
    ):
        raise RegenerationError("reset trace comparison member authority drift")
    candidate_members = candidate_inspection["members"]
    required_members = set(exact_members)
    numeric_member_by_projection = {
        "maximum_base_position_error_m": ("base_pose_world", range(0, 3)),
        "maximum_base_quaternion_component_error": ("base_pose_world", range(3, 7)),
        "maximum_base_twist_error": ("base_twist_world", None),
        "maximum_joint_position_error_rad": ("joint_position", None),
        "maximum_joint_velocity_error_rad_s": ("joint_velocity", None),
    }
    tolerance_key_by_projection = {
        "maximum_base_position_error_m": "base_pose_world_position_xyz",
        "maximum_base_quaternion_component_error": "base_pose_world_quaternion_xyzw",
        "maximum_base_twist_error": "base_twist_world",
        "maximum_joint_position_error_rad": "joint_position",
        "maximum_joint_velocity_error_rad_s": "joint_velocity",
    }
    active_numeric = {
        projection: details
        for projection, details in numeric_member_by_projection.items()
        if tolerance_key_by_projection[projection] in tolerances
    }
    required_members.update(member for member, _indices in active_numeric.values())
    raw_reset_members = {
        "base_pose_world",
        "requested_command",
        "post_slew_applied_command",
        "physics_contact",
        "joint_position",
        "joint_velocity",
    }
    raw_reset_evidence = raw_reset_members.issubset(candidate_members) and (
        raw_reset_members.issubset(captured)
    )
    if bool(raw_reset_members & set(candidate_members)) != raw_reset_evidence:
        raise RegenerationError("reset raw outcome authority is only partially present")
    if raw_reset_evidence:
        required_members.update(raw_reset_members)
    if not required_members.issubset(candidate_members) or not required_members.issubset(captured):
        raise RegenerationError("reset comparison lacks required candidate-trace members")
    for member in required_members:
        if len(captured[member]) != EXPECTED_RESET_FIXTURE_TRACE_COUNT:
            raise RegenerationError(f"reset capture cardinality drift: {member}")

    expected_fields = authority["reset_pair_comparison_fields"]
    if not isinstance(expected_fields, list) or not expected_fields:
        raise RegenerationError("reset comparison projection field authority drift")
    rows = state_snapshot_index["records"]
    projections: list[dict[str, Any]] = []
    for state_index, state_row in enumerate(rows):
        first_index = state_index * 2
        second_index = first_index + 1
        projection: dict[str, Any] = {
            "state_id": state_row["state_id"],
            "trial_indices": [0, 1],
            "physics_sample_count": sample_count,
        }
        numeric_rows: dict[str, tuple[list[tuple[float, ...]], list[tuple[float, ...]]]] = {}
        for projection_name, (member, indices) in active_numeric.items():
            trailing_shape = candidate_members[member]["shape"][1:]
            if len(trailing_shape) != 1 or not isinstance(trailing_shape[0], int):
                raise RegenerationError(f"reset numeric trace shape drift: {member}")
            width = trailing_shape[0]
            left = _float64_rows(
                captured[member][first_index], width, sample_count, f"{member} trial 0"
            )
            right = _float64_rows(
                captured[member][second_index], width, sample_count, f"{member} trial 1"
            )
            numeric_rows[member] = (left, right)
            projection[projection_name] = _maximum_component_error(left, right, indices)
        exact_equal = {
            member: captured[member][first_index] == captured[member][second_index]
            for member in exact_members
        }
        projection["exact_member_equal"] = exact_equal

        trials = state_row["reset_trials"]
        if not isinstance(trials, list) or len(trials) != 2:
            raise RegenerationError("reset pair lacks two trial metadata rows")
        post_restore_present = [
            "post_restore_state_sha256" in trial for trial in trials
        ]
        if any(post_restore_present) != all(post_restore_present):
            raise RegenerationError(
                "reset pair has only partial post-restore state custody"
            )
        if all(post_restore_present):
            post_restore_digests = [
                _sha256(
                    trial["post_restore_state_sha256"],
                    "reset post-restore state SHA",
                )
                for trial in trials
            ]
            if post_restore_digests[0] != post_restore_digests[1]:
                raise RegenerationError(
                    "reset trials differ immediately after serialized restore"
                )
        expected_termination = comparison_authority.get(
            "completion_termination_reason"
        )
        if expected_termination is not None and (
            not isinstance(expected_termination, str)
            or not expected_termination
            or any(
                trial.get("termination_reason") != expected_termination
                for trial in trials
            )
        ):
            raise RegenerationError("reset trial did not terminate at the frozen horizon")
        projection["termination_reason_equal"] = (
            trials[0].get("termination_reason") == trials[1].get("termination_reason")
        )
        projection["stuck_equal"] = trials[0].get("stuck") == trials[1].get("stuck")

        if raw_reset_evidence:
            stuck_authority = authority["physical_trace_reduction_authority"].get(
                "stuck"
            )
            if not isinstance(stuck_authority, Mapping):
                raise RegenerationError("reset stuck authority is absent")
            for trial_index, trace_index in enumerate((first_index, second_index)):
                poses = _float64_rows(
                    captured["base_pose_world"][trace_index],
                    7,
                    sample_count,
                    f"reset[{state_index}][{trial_index}].base_pose_world",
                )
                commands = _float64_rows(
                    captured["requested_command"][trace_index],
                    3,
                    sample_count,
                    f"reset[{state_index}][{trial_index}].requested_command",
                )
                endpoint = _relative_endpoint(poses[0], poses[-1])
                activity = max(abs(value) for command in commands for value in command)
                raw_stuck = bool(
                    activity > float(stuck_authority["command_activity_threshold"])
                    and math.hypot(endpoint[0], endpoint[1])
                    < float(stuck_authority["h3_translation_threshold_m"])
                    and abs(endpoint[2])
                    < float(stuck_authority["h3_heading_threshold_rad"])
                )
                if trials[trial_index].get("stuck") is not raw_stuck:
                    raise RegenerationError("reset stuck metadata differs from raw trace")
                _require_vector_close(
                    trials[trial_index].get("base_pose_world"),
                    poses[-1],
                    f"reset[{state_index}][{trial_index}].base_pose_world",
                )
                for field, member in (
                    ("requested_command_sequence_sha256", "requested_command"),
                    (
                        "post_slew_applied_command_sequence_sha256",
                        "post_slew_applied_command",
                    ),
                    ("contact_sequence_sha256", "physics_contact"),
                ):
                    expected_digest = candidate_members[member][
                        "row_or_slice_sha256s"
                    ][trace_index]
                    if trials[trial_index].get(field) != expected_digest:
                        raise RegenerationError(
                            f"reset {field} differs from raw candidate-trace bytes"
                        )
                for field, member in (
                    ("joint_position_sha256", "joint_position"),
                    ("joint_velocity_sha256", "joint_velocity"),
                ):
                    details = candidate_members[member]
                    shape = details["shape"]
                    if len(shape) != 2 or shape[1] != 12:
                        raise RegenerationError(f"reset {member} trace shape drift")
                    payload = captured[member][trace_index]
                    final_digest = _array_row_digest(
                        payload[-12 * 8 :],
                        (12,),
                        details["descr"],
                    )
                    if trials[trial_index].get(field) != final_digest:
                        raise RegenerationError(
                            f"reset {field} differs from raw final trace row"
                        )

        if "base_pose_world" in numeric_rows:
            left_pose, right_pose = numeric_rows["base_pose_world"]
            left_endpoint, right_endpoint = left_pose[-1], right_pose[-1]
            projection["endpoint_position_error_m"] = math.sqrt(
                math.fsum(
                    (left_endpoint[index] - right_endpoint[index]) ** 2
                    for index in range(3)
                )
            )
            projection["endpoint_heading_error_rad"] = _angle_difference(
                _yaw_from_xyzw(left_endpoint[3:7]),
                _yaw_from_xyzw(right_endpoint[3:7]),
            )

        passed = all(exact_equal.values()) and projection["termination_reason_equal"] and projection["stuck_equal"]
        for projection_name in active_numeric:
            passed = passed and projection[projection_name] <= float(
                tolerances[tolerance_key_by_projection[projection_name]]
            )
        if "endpoint_position_error_m" in projection:
            passed = passed and projection["endpoint_position_error_m"] <= float(
                comparison_authority["endpoint_position_tolerance_m"]
            )
            passed = passed and projection["endpoint_heading_error_rad"] <= float(
                comparison_authority["endpoint_heading_tolerance_rad"]
            )
        projection["passed"] = bool(passed)
        if set(projection) != set(expected_fields):
            raise RegenerationError("recomputed reset comparison field-set drift")
        supplied = state_row.get("reset_pair_comparison")
        if not isinstance(supplied, Mapping) or dict(supplied) != projection:
            raise RegenerationError(
                f"reset pair comparison differs from raw NPZ traces: {state_row['state_id']}"
            )
        if projection["passed"] is not True:
            raise RegenerationError(
                f"reset pair comparison gate failed: {state_row['state_id']}"
            )
        projections.append(projection)
    return {
        "state_pairs": len(projections),
        "trace_slices": len(projections) * 2,
        "physics_samples_per_trace": sample_count,
        "comparison_projection_digest": canonical_digest(projections),
        "all_pairs_passed": True,
    }


def _captured_float_rows(
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    inspections: Mapping[str, Mapping[str, Any]],
    leaf: str,
    member: str,
    index: int,
    sample_count: int,
) -> list[tuple[float, ...]]:
    try:
        payload = captured[leaf][member][index]
        details = inspections[leaf]["members"][member]
    except (KeyError, IndexError, TypeError) as exc:
        raise RegenerationError(
            f"raw physical capture is absent: {leaf}:{member}[{index}]"
        ) from exc
    shape = details["shape"]
    if details["descr"] != "<f8" or not isinstance(shape, list) or not shape:
        raise RegenerationError(f"raw physical float authority drift: {leaf}:{member}")
    trailing = shape[1:]
    width = math.prod(trailing) if trailing else 1
    return _float64_rows(payload, width, sample_count, f"{leaf}:{member}[{index}]")


def _captured_binary(
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    inspections: Mapping[str, Mapping[str, Any]],
    leaf: str,
    member: str,
    index: int,
    sample_count: int,
) -> tuple[int, ...]:
    try:
        payload = captured[leaf][member][index]
        details = inspections[leaf]["members"][member]
    except (KeyError, IndexError, TypeError) as exc:
        raise RegenerationError(
            f"raw physical capture is absent: {leaf}:{member}[{index}]"
        ) from exc
    if details["descr"] != "|u1" or details["shape"][1:] != []:
        raise RegenerationError(f"raw physical binary authority drift: {leaf}:{member}")
    return _uint8_values(payload, sample_count, f"{leaf}:{member}[{index}]")


def _require_close(
    observed: Any, expected: float, label: str, tolerance: float = 1.0e-9
) -> None:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise RegenerationError(f"{label} must be numeric")
    value = float(observed)
    if not math.isfinite(value) or abs(value - expected) > tolerance:
        raise RegenerationError(
            f"{label} differs from raw physical evidence: {value!r} != {expected!r}"
        )


def _require_vector_close(
    observed: Any,
    expected: Sequence[float],
    label: str,
    tolerance: float = 1.0e-9,
) -> None:
    if not isinstance(observed, list) or len(observed) != len(expected):
        raise RegenerationError(f"{label} vector shape drift")
    for index, (left, right) in enumerate(zip(observed, expected)):
        _require_close(left, float(right), f"{label}[{index}]", tolerance)


def _crossing_order(
    value: Mapping[str, Any], edge_id: str | None = None
) -> tuple[int, float, str]:
    return (
        int(value["sample_after"]),
        float(value["fraction"]),
        str(value.get("edge_id") if edge_id is None else edge_id),
    )


def _first_competing_crossing(
    poses: Sequence[Sequence[float]],
    edges: Sequence[Mapping[str, Any]],
    label: str,
    tolerance: float = 0.0,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, Mapping) or set(edge) != {
            "edge_id",
            "opening_segment_world",
            "opening_normal_world",
        }:
            raise RegenerationError(f"{label} competing-edge geometry drift")
        crossing = _first_transverse_crossing(
            poses,
            edge["opening_segment_world"],
            edge["opening_normal_world"],
            f"{label}:{edge['edge_id']}",
            tolerance,
        )
        if crossing is not None:
            crossing["edge_id"] = _string(edge["edge_id"], f"{label}.edge_id")
            candidates.append(crossing)
    return min(candidates, key=_crossing_order) if candidates else None


def _dwell_projection(
    poses: Sequence[Sequence[float]],
    target_membership: Sequence[int],
    crossing: Mapping[str, Any],
    required_samples: int,
    tolerance: float,
) -> dict[str, Any]:
    after = int(crossing["sample_after"])
    midpoint = crossing["midpoint_world_xy"]
    normal = crossing["unit_normal"]
    beyond = [
        (float(row[0]) - float(midpoint[0])) * float(normal[0])
        + (float(row[1]) - float(midpoint[1])) * float(normal[1])
        for row in poses[after:]
    ]
    target_offsets = [
        index for index, value in enumerate(target_membership[after:]) if value
    ]
    reached_offset = target_offsets[0] if target_offsets else None
    consecutive = 0
    for value in beyond:
        if value <= tolerance:
            break
        consecutive += 1
    sustained = consecutive >= required_samples
    reached = reached_offset is not None
    target_early = bool(
        reached and reached_offset + 1 < required_samples
    )
    return {
        "valid": sustained or target_early,
        "sustained": sustained,
        "target_reached": reached,
        "target_entered_before_dwell_complete": target_early,
        "beyond_port_consecutive_physics_samples": consecutive,
    }


def _relative_endpoint(
    initial_pose: Sequence[float], endpoint_pose: Sequence[float]
) -> list[float]:
    _initial_roll, _initial_pitch, initial_yaw = _quaternion_roll_pitch_yaw(
        initial_pose[3:7]
    )
    _endpoint_roll, _endpoint_pitch, endpoint_yaw = _quaternion_roll_pitch_yaw(
        endpoint_pose[3:7]
    )
    dx = float(endpoint_pose[0]) - float(initial_pose[0])
    dy = float(endpoint_pose[1]) - float(initial_pose[1])
    cosine, sine = math.cos(initial_yaw), math.sin(initial_yaw)
    return [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        math.atan2(
            math.sin(endpoint_yaw - initial_yaw),
            math.cos(endpoint_yaw - initial_yaw),
        ),
    ]


def _validate_timestamps(rows: Sequence[Sequence[float]], dt: float, label: str) -> None:
    values = [row[0] for row in rows]
    if len(values) < 2 or any(
        abs((right - left) - dt) > 1.0e-12
        for left, right in zip(values, values[1:])
    ):
        raise RegenerationError(f"{label} is not sampled at the frozen physics rate")


def _crossing_velocity(
    twists: Sequence[Sequence[float]],
    crossing: Mapping[str, Any],
    rule: str,
) -> list[float]:
    before = int(crossing["sample_before"])
    after = int(crossing["sample_after"])
    fraction = float(crossing["fraction"])
    if rule in {
        "linear interpolation of base_twist_world XY at crossing fraction",
        (
            "linearly interpolate base_twist_world vx_world,vy_world at crossing alpha; "
            "heading=atan2(vy_world,vx_world), requiring nonzero planar speed"
        ),
    }:
        return [
            float(twists[before][index])
            + fraction * (float(twists[after][index]) - float(twists[before][index]))
            for index in range(2)
        ]
    if rule == "base_twist_world XY at crossing sample_after":
        return [float(twists[after][0]), float(twists[after][1])]
    raise RegenerationError("crossing-velocity reduction rule is absent or unsupported")


def _validate_raw_teacher_evidence(
    documents: Mapping[str, Mapping[str, Any]],
    inspections: Mapping[str, Mapping[str, Any]],
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = physical.get("teacher_member_mapping")
    geometry_rows = physical.get("specs")
    if not isinstance(mapping, Mapping) or not isinstance(geometry_rows, list):
        raise RegenerationError("teacher raw-reduction authority drift")
    required_mapping = {
        "timestamp",
        "base_pose",
        "base_twist",
        "contact",
        "requested_command",
        "applied_command",
        "source_membership",
        "selected_edge_membership",
        "target_membership",
    }
    if set(mapping) != required_mapping:
        raise RegenerationError("teacher member mapping drift")
    rows = documents["teacher_trace_index"]["records"]
    if len(rows) != EXPECTED_TEACHER_TRACE_COUNT or len(geometry_rows) != len(rows):
        raise RegenerationError("teacher raw-reduction population drift")
    dwell_samples = _integer(physical.get("dwell_samples"), "teacher dwell samples", minimum=1)
    dt = float(physical.get("physics_dt_s"))
    velocity_rule = _string(
        physical.get("crossing_velocity"), "crossing_velocity"
    )
    port_by_state = {
        row["state_id"]: row for row in documents["edge_port_index"]["records"]
    }
    raw_projections: list[dict[str, Any]] = []
    selected_count = 0
    for trace_index, (row, geometry) in enumerate(zip(rows, geometry_rows)):
        if not isinstance(geometry, Mapping) or set(geometry) != {
            "candidate_spec_id",
            "state_id",
            "pool_order_index",
            "canonical_spec_sha256",
            "source_boundary_polygon_world",
            "target_boundary_polygon_world",
            "selected_edge",
            "competing_edges",
            "teacher_route_polyline_world",
            "spawn_se2_world",
        }:
            raise RegenerationError(f"teacher geometry row field drift: {trace_index}")
        if geometry["pool_order_index"] != trace_index:
            raise RegenerationError("teacher geometry pool order drift")
        for field in ("candidate_spec_id", "state_id", "canonical_spec_sha256"):
            if row.get(field) != geometry.get(field):
                raise RegenerationError(f"teacher geometry identity drift: {field}")
        sample_count = _integer(row.get("sample_count"), "teacher sample count", minimum=2)
        timestamps = _captured_float_rows(
            captured, inspections, "teacher_traces.npz", mapping["timestamp"],
            trace_index, sample_count,
        )
        poses = _captured_float_rows(
            captured, inspections, "teacher_traces.npz", mapping["base_pose"],
            trace_index, sample_count,
        )
        twists = _captured_float_rows(
            captured, inspections, "teacher_traces.npz", mapping["base_twist"],
            trace_index, sample_count,
        )
        requested_commands = _captured_float_rows(
            captured,
            inspections,
            "teacher_traces.npz",
            mapping["requested_command"],
            trace_index,
            sample_count,
        )
        contact = _captured_binary(
            captured, inspections, "teacher_traces.npz", mapping["contact"],
            trace_index, sample_count,
        )
        source = _captured_binary(
            captured, inspections, "teacher_traces.npz", mapping["source_membership"],
            trace_index, sample_count,
        )
        edge_member = _captured_binary(
            captured, inspections, "teacher_traces.npz",
            mapping["selected_edge_membership"], trace_index, sample_count,
        )
        target = _captured_binary(
            captured, inspections, "teacher_traces.npz", mapping["target_membership"],
            trace_index, sample_count,
        )
        _validate_timestamps(timestamps, dt, f"teacher trace {trace_index}")
        geometry_tolerance = float(physical["port_crossing_tolerance_m"])
        _validate_membership_against_geometry(
            poses,
            source,
            geometry["source_boundary_polygon_world"],
            geometry_tolerance,
            f"teacher[{trace_index}].source_region",
        )
        _validate_membership_against_geometry(
            poses,
            target,
            geometry["target_boundary_polygon_world"],
            geometry_tolerance,
            f"teacher[{trace_index}].target_region",
        )
        selected_edge = geometry["selected_edge"]
        if not isinstance(selected_edge, Mapping) or set(selected_edge) != {
            "edge_id", "opening_segment_world", "opening_normal_world"
        }:
            raise RegenerationError("selected teacher-edge geometry drift")
        selected_crossing = _first_transverse_crossing(
            poses,
            selected_edge["opening_segment_world"],
            selected_edge["opening_normal_world"],
            f"teacher[{trace_index}].selected_edge",
            float(physical["port_crossing_tolerance_m"]),
        )
        competing = _first_competing_crossing(
            poses,
            geometry["competing_edges"],
            f"teacher[{trace_index}]",
            float(physical["port_crossing_tolerance_m"]),
        )
        competing_first = bool(
            competing is not None
            and (
                selected_crossing is None
                or _crossing_order(competing)
                <= _crossing_order(selected_crossing, str(selected_edge["edge_id"]))
            )
        )
        dwell = (
            _dwell_projection(
                poses,
                target,
                selected_crossing,
                dwell_samples,
                float(physical["port_crossing_tolerance_m"]),
            )
            if selected_crossing is not None
            else {
                "valid": False,
                "sustained": False,
                "target_reached": bool(any(target)),
                "target_entered_before_dwell_complete": False,
                "beyond_port_consecutive_physics_samples": 0,
            }
        )
        crossed = bool(
            selected_crossing is not None and not competing_first and dwell["valid"]
        )
        selected_source_exit = _first_source_boundary_exit(
            poses,
            source,
            edge_member,
            geometry["source_boundary_polygon_world"],
            f"teacher[{trace_index}]",
        )
        first_source_exit = _first_membership_transition(
            source, 1, 0, f"teacher[{trace_index}].source"
        )
        if source[0] != 1 or target[0] != 0:
            raise RegenerationError(
                "teacher trace does not begin in source and outside target"
            )
        left_source = first_source_exit is not None
        first_target_entry = _first_membership_transition(
            target, 0, 1, f"teacher[{trace_index}].target"
        )
        contact_free = not any(contact)
        _first, _second, normal, _width = _normalised_opening(
            selected_edge["opening_segment_world"],
            selected_edge["opening_normal_world"],
            f"teacher[{trace_index}].selected_edge",
        )
        opening_midpoint = (
            (_first[0] + _second[0]) / 2.0,
            (_first[1] + _second[1]) / 2.0,
        )
        start_route_distance = math.hypot(
            float(poses[0][0]) - opening_midpoint[0],
            float(poses[0][1]) - opening_midpoint[1],
        )
        route_progress = start_route_distance - min(
            math.hypot(
                float(pose[0]) - opening_midpoint[0],
                float(pose[1]) - opening_midpoint[1],
            )
            for pose in poses
        )
        reached_target = bool(any(target))
        endpoint_pose = poses[-1]
        endpoint_roll, endpoint_pitch, endpoint_yaw = _quaternion_roll_pitch_yaw(
            endpoint_pose[3:7]
        )
        opening_tangent = (
            (_second[0] - _first[0]) / _width,
            (_second[1] - _first[1]) / _width,
        )
        endpoint_lateral_error = abs(
            (float(endpoint_pose[0]) - opening_midpoint[0]) * opening_tangent[0]
            + (float(endpoint_pose[1]) - opening_midpoint[1]) * opening_tangent[1]
        )
        endpoint_angular_error = _angle_difference(
            endpoint_yaw, math.atan2(normal[1], normal[0])
        )
        successor_authority = physical["successor_viable"]
        successor_viable = bool(
            float(endpoint_pose[2])
            >= float(successor_authority["minimum_base_height_m"])
            and abs(endpoint_roll)
            <= float(successor_authority["maximum_absolute_roll_rad"])
            and abs(endpoint_pitch)
            <= float(successor_authority["maximum_absolute_pitch_rad"])
            and not bool(contact[-1])
        )
        body_endpoint = _relative_endpoint(poses[0], endpoint_pose)
        activity = max(abs(value) for command in requested_commands for value in command)
        stuck_authority = physical["stuck"]
        stuck = bool(
            activity > float(stuck_authority["command_activity_threshold"])
            and math.hypot(body_endpoint[0], body_endpoint[1])
            < float(stuck_authority["h3_translation_threshold_m"])
            and abs(body_endpoint[2])
            < float(stuck_authority["h3_heading_threshold_rad"])
        )
        if row.get("contact_free") is not contact_free:
            raise RegenerationError("teacher contact-free summary differs from raw trace")
        if row.get("left_source_region") is not left_source:
            raise RegenerationError("teacher source-leave summary differs from raw trace")
        if row.get("first_source_exit_sample_index") != first_source_exit:
            raise RegenerationError("teacher source-exit index differs from raw trace")
        if row.get("competing_port_entered") is not (competing is not None):
            raise RegenerationError("teacher competing-port summary differs from raw trace")
        if row.get("crossed_directed_port") is not crossed:
            raise RegenerationError("teacher selected-port summary differs from raw trace")
        _require_close(row.get("route_progress_m"), route_progress, "teacher route_progress_m")
        if row.get("positive_route_progress") is not (route_progress > 0.0):
            raise RegenerationError("teacher positive-progress summary differs from raw trace")
        if row.get("reached_target_node") is not reached_target:
            raise RegenerationError("teacher target-membership summary differs from raw trace")
        if row.get("first_target_entry_sample_index") != first_target_entry:
            raise RegenerationError("teacher target-entry index differs from raw trace")
        expected_where = "TARGET_NODE" if reached_target else "BEYOND_DIRECTED_PORT"
        if row.get("where_reached") != expected_where:
            raise RegenerationError("teacher where_reached differs from raw trace")
        _require_close(
            row.get("endpoint_lateral_error_m"),
            endpoint_lateral_error,
            "teacher endpoint_lateral_error_m",
        )
        _require_close(
            row.get("endpoint_angular_error_rad"),
            endpoint_angular_error,
            "teacher endpoint_angular_error_rad",
        )
        if row.get("successor_viable") is not successor_viable:
            raise RegenerationError("teacher successor viability differs from raw trace")
        if row.get("stuck") is not stuck:
            raise RegenerationError("teacher stuck outcome differs from raw trace")
        if crossed:
            assert selected_crossing is not None
            if selected_source_exit is None:
                raise RegenerationError("teacher selected crossing lacks source-boundary exit")
            if (
                selected_source_exit["sample_after"] != selected_crossing["sample_after"]
                or abs(
                    selected_source_exit["fraction"] - selected_crossing["fraction"]
                )
                > 1.0e-9
            ):
                raise RegenerationError("teacher source exit and selected port disagree")
            velocity = _crossing_velocity(twists, selected_crossing, velocity_rule)
            _require_close(
                row.get("crossing_segment_fraction"),
                selected_crossing["fraction"],
                "teacher crossing fraction",
            )
            if row.get("first_crossing_sample_index") != selected_crossing["sample_after"]:
                raise RegenerationError("teacher crossing sample differs from raw trace")
            _require_close(
                row.get("crossing_directed_normal_dot"),
                selected_crossing["normal_dot_displacement_m"],
                "teacher crossing normal dot",
            )
            _require_close(
                row.get("crossing_lateral_fraction"),
                selected_crossing["lateral_fraction"],
                "teacher crossing lateral fraction",
            )
            _require_vector_close(
                row.get("crossing_velocity_world_xy"), velocity,
                "teacher crossing velocity",
            )
            if math.hypot(*velocity) <= 0.0:
                raise RegenerationError("teacher crossing velocity is zero")
            velocity_heading = math.atan2(velocity[1], velocity[0])
            _require_close(
                row.get("crossing_velocity_heading_world_rad"), velocity_heading,
                "teacher crossing velocity heading",
            )
            if row.get("beyond_port_consecutive_physics_samples") != dwell[
                "beyond_port_consecutive_physics_samples"
            ]:
                raise RegenerationError("teacher dwell count differs from raw trace")
            if row.get("target_entered_before_dwell_complete") is not dwell[
                "target_entered_before_dwell_complete"
            ]:
                raise RegenerationError("teacher early-target summary differs from raw trace")
        else:
            null_fields = (
                "first_crossing_sample_index", "crossing_segment_fraction",
                "crossing_directed_normal_dot", "crossing_lateral_fraction",
                "crossing_velocity_world_xy", "crossing_velocity_heading_world_rad",
            )
            if any(row.get(field) is not None for field in null_fields):
                raise RegenerationError("uncrossed teacher fabricates raw crossing evidence")
            if row.get("beyond_port_consecutive_physics_samples") != 0:
                raise RegenerationError("uncrossed teacher fabricates dwell evidence")
        if row.get("selected"):
            selected_count += 1
            port = port_by_state.get(row["state_id"])
            if port is None or not crossed or selected_crossing is None:
                raise RegenerationError("selected teacher lacks reducible edge-port evidence")
            if (
                port.get("crossing_sample_before") != selected_crossing["sample_before"]
                or port.get("crossing_sample_after") != selected_crossing["sample_after"]
            ):
                raise RegenerationError("edge-port sample identity differs from teacher trace")
            _require_close(
                port.get("crossing_fraction"), selected_crossing["fraction"],
                "edge-port crossing fraction",
            )
            _opening_first, _opening_second, opening_normal, _opening_width = (
                _normalised_opening(
                    selected_edge["opening_segment_world"],
                    selected_edge["opening_normal_world"],
                    f"teacher[{trace_index}].selected_edge",
                )
            )
            port_pose = [
                selected_crossing["point_world_xy"][0],
                selected_crossing["point_world_xy"][1],
                math.atan2(opening_normal[1], opening_normal[0]),
            ]
            _require_vector_close(port.get("directed_port_world"), port_pose, "directed port")
            velocity = _crossing_velocity(twists, selected_crossing, velocity_rule)
            _require_vector_close(
                port.get("teacher_crossing_velocity_world_xy"), velocity,
                "edge-port teacher velocity",
            )
            _require_close(
                port.get("teacher_crossing_velocity_heading_world_rad"),
                math.atan2(velocity[1], velocity[0]),
                "edge-port teacher velocity heading",
            )
        raw_projections.append(
            {
                "candidate_spec_id": row["candidate_spec_id"],
                "contact_free": contact_free,
                "left_source_region": left_source,
                "crossed_directed_port": crossed,
                "competing_port_entered": competing is not None,
                "route_progress_m": route_progress,
                "reached_target_node": reached_target,
            }
        )
    if selected_count != EXPECTED_STATE_COUNT:
        raise RegenerationError("raw teacher selected-state count drift")
    return {
        "teacher_trace_count": len(raw_projections),
        "selected_teacher_trace_count": selected_count,
        "raw_projection_digest": canonical_digest(raw_projections),
        "all_teacher_labels_regenerated": True,
    }


def _raw_candidate_projection(
    *,
    trace_index: int,
    state_index: int,
    geometry: Mapping[str, Any],
    initial_pose: Sequence[float],
    port: Mapping[str, Any],
    inspections: Mapping[str, Mapping[str, Any]],
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = physical["candidate_member_mapping"]
    if not isinstance(mapping, Mapping) or set(mapping) != {
        "timestamp", "base_pose", "base_twist", "requested_command",
        "post_slew_command", "contact", "source_membership",
        "selected_edge_membership", "competing_edge_membership",
        "target_membership",
    }:
        raise RegenerationError("candidate member mapping drift")
    count = EXPECTED_PHYSICS_SAMPLES_PER_BRANCH
    timestamps = _captured_float_rows(
        captured, inspections, "candidate_traces.npz", mapping["timestamp"],
        trace_index, count,
    )
    poses = _captured_float_rows(
        captured, inspections, "candidate_traces.npz", mapping["base_pose"],
        trace_index, count,
    )
    twists = _captured_float_rows(
        captured, inspections, "candidate_traces.npz", mapping["base_twist"],
        trace_index, count,
    )
    requested = _captured_float_rows(
        captured, inspections, "candidate_traces.npz", mapping["requested_command"],
        trace_index, count,
    )
    applied = _captured_float_rows(
        captured, inspections, "candidate_traces.npz", mapping["post_slew_command"],
        trace_index, count,
    )
    contact = _captured_binary(
        captured, inspections, "candidate_traces.npz", mapping["contact"],
        trace_index, count,
    )
    source = _captured_binary(
        captured, inspections, "candidate_traces.npz", mapping["source_membership"],
        trace_index, count,
    )
    selected_member = _captured_binary(
        captured, inspections, "candidate_traces.npz",
        mapping["selected_edge_membership"], trace_index, count,
    )
    competing_member = _captured_binary(
        captured, inspections, "candidate_traces.npz",
        mapping["competing_edge_membership"], trace_index, count,
    )
    target = _captured_binary(
        captured, inspections, "candidate_traces.npz", mapping["target_membership"],
        trace_index, count,
    )
    _validate_timestamps(
        timestamps, float(physical["physics_dt_s"]), f"candidate trace {trace_index}"
    )
    if source[0] != 1 or target[0] != 0:
        raise RegenerationError("candidate trace does not begin in source and outside target")
    geometry_tolerance = float(physical.get("port_crossing_tolerance_m", 0.0))
    _validate_membership_against_geometry(
        poses,
        source,
        geometry["source_boundary_polygon_world"],
        geometry_tolerance,
        f"candidate[{trace_index}].source_region",
    )
    _validate_membership_against_geometry(
        poses,
        target,
        geometry["target_boundary_polygon_world"],
        geometry_tolerance,
        f"candidate[{trace_index}].target_region",
    )
    first_source_exit = _first_membership_transition(
        source, 1, 0, f"candidate[{trace_index}].source"
    )
    first_target_entry = _first_membership_transition(
        target, 0, 1, f"candidate[{trace_index}].target"
    )
    selected_edge = geometry["selected_edge"]
    selected_crossing = _first_transverse_crossing(
        poses,
        selected_edge["opening_segment_world"],
        selected_edge["opening_normal_world"],
        f"candidate[{trace_index}].selected_edge",
        float(physical.get("port_crossing_tolerance_m", 0.0)),
    )
    competing = _first_competing_crossing(
        poses,
        geometry["competing_edges"],
        f"candidate[{trace_index}]",
        float(physical.get("port_crossing_tolerance_m", 0.0)),
    )
    competing_first = bool(
        competing is not None
        and (
            selected_crossing is None
            or _crossing_order(competing)
            <= _crossing_order(selected_crossing, str(selected_edge["edge_id"]))
        )
    )
    dwell = (
        _dwell_projection(
            poses,
            target,
            selected_crossing,
            int(physical["dwell_samples"]),
            float(physical.get("port_crossing_tolerance_m", 0.0)),
        )
        if selected_crossing is not None
        else {
            "valid": False,
            "sustained": False,
            "target_reached": bool(any(target)),
            "target_entered_before_dwell_complete": False,
            "beyond_port_consecutive_physics_samples": 0,
        }
    )
    entered_correct = bool(
        selected_crossing is not None and not competing_first and dwell["valid"]
    )
    entered_wrong = competing_first
    no_edge = not entered_correct and not entered_wrong
    if entered_correct and not any(selected_member):
        raise RegenerationError("correct-edge geometry event lacks selected-edge membership")
    if entered_wrong and not any(competing_member):
        raise RegenerationError("wrong-edge geometry event lacks competing-edge membership")

    endpoint_indices = physical.get(
        "endpoint_indices", {"H1": 249, "H2": 499, "H3": 749}
    )
    if endpoint_indices != {"H1": 249, "H2": 499, "H3": 749}:
        raise RegenerationError("candidate endpoint sample authority drift")
    endpoints = {
        horizon: _relative_endpoint(initial_pose, poses[index])
        for horizon, index in endpoint_indices.items()
    }
    endpoint_pose = poses[endpoint_indices["H3"]]
    roll, pitch, yaw = _quaternion_roll_pitch_yaw(endpoint_pose[3:7])
    endpoint_world = [float(endpoint_pose[0]), float(endpoint_pose[1]), yaw]
    physics_contact = bool(any(contact))
    h3_contact = bool(contact[-1])
    h3_solver_finite = True
    successor_authority = physical["successor_viable"]
    successor_viable = bool(
        h3_solver_finite
        and float(endpoint_pose[2])
        >= float(successor_authority["minimum_base_height_m"])
        and abs(roll)
        <= float(successor_authority["maximum_absolute_roll_rad"])
        and abs(pitch)
        <= float(successor_authority["maximum_absolute_pitch_rad"])
        and not h3_contact
    )
    activity = max(abs(value) for row in requested for value in row)
    stuck_authority = physical["stuck"]
    stuck = bool(
        activity > float(stuck_authority["command_activity_threshold"])
        and math.hypot(endpoints["H3"][0], endpoints["H3"][1])
        < float(stuck_authority["h3_translation_threshold_m"])
        and abs(endpoints["H3"][2])
        < float(stuck_authority["h3_heading_threshold_rad"])
    )
    oracle_admissible = bool(
        not physics_contact and h3_solver_finite and successor_viable and not stuck
    )
    port_pose = port["directed_port_world"]
    start_distance = math.hypot(
        float(initial_pose[0]) - float(port_pose[0]),
        float(initial_pose[1]) - float(port_pose[1]),
    )
    minimum_distance = min(
        math.hypot(float(pose[0]) - float(port_pose[0]), float(pose[1]) - float(port_pose[1]))
        for pose in poses
    )
    port_progress = start_distance - minimum_distance
    tangent = (math.cos(float(port_pose[2])), math.sin(float(port_pose[2])))
    lateral_error = abs(
        -(float(endpoint_pose[0]) - float(port_pose[0])) * tangent[1]
        + (float(endpoint_pose[1]) - float(port_pose[1])) * tangent[0]
    )
    angular_error = _angle_difference(yaw, float(port_pose[2]))

    command_tracking_authority = physical.get("command_tracking_authority")
    if not isinstance(command_tracking_authority, Mapping):
        raise RegenerationError("raw command-tracking authority is absent")
    tick_count = _integer(
        command_tracking_authority.get("command_ticks"),
        "command tracking tick count",
        minimum=1,
    )
    samples_per_tick = _integer(
        command_tracking_authority.get("physics_samples_per_command_tick"),
        "command tracking physics samples per tick",
        minimum=1,
    )
    if tick_count * samples_per_tick != count:
        raise RegenerationError("command tracking dimensions do not cover the trace")
    transient_samples = _integer(
        command_tracking_authority.get(
            "discard_initial_physics_samples_per_command_tick"
        ),
        "command tracking transient samples",
    )
    if transient_samples >= samples_per_tick:
        raise RegenerationError("command tracking transient consumes the tick")
    if command_tracking_authority.get(
        "averaged_physics_samples_per_command_tick"
    ) != samples_per_tick - transient_samples:
        raise RegenerationError("command tracking averaging window drift")
    command_rows: list[dict[str, Any]] = []
    for tick in range(tick_count):
        start, stop = tick * samples_per_tick, (tick + 1) * samples_per_tick
        requested_tick = requested[start:stop]
        applied_tick = applied[start:stop]
        if any(value != requested_tick[0] for value in requested_tick):
            raise RegenerationError("requested command changes inside a command tick")
        if any(value != applied_tick[0] for value in applied_tick):
            raise RegenerationError("post-slew command changes inside a command tick")
        achieved_body: list[tuple[float, float, float]] = []
        for pose, twist in zip(
            poses[start + transient_samples : stop],
            twists[start + transient_samples : stop],
        ):
            yaw_now = _quaternion_roll_pitch_yaw(pose[3:7])[2]
            cosine, sine = math.cos(yaw_now), math.sin(yaw_now)
            achieved_body.append(
                (
                    cosine * twist[0] + sine * twist[1],
                    -sine * twist[0] + cosine * twist[1],
                    twist[5],
                )
            )
        mean_achieved = [
            math.fsum(row[index] for row in achieved_body) / len(achieved_body)
            for index in range(3)
        ]
        command_rows.append(
            {
                "command_tick_index": tick,
                "requested_command": list(requested_tick[0]),
                "post_slew_command": list(applied_tick[0]),
                "mean_achieved_body_velocity": mean_achieved,
            }
        )
    return {
        "poses": poses,
        "requested": requested,
        "applied": applied,
        "entered_correct_edge": entered_correct,
        "entered_wrong_edge": entered_wrong,
        "no_edge": no_edge,
        "selected_crossing": selected_crossing,
        "competing_crossing": competing,
        "competing_first": competing_first,
        "dwell": dwell,
        "endpoints": endpoints,
        "endpoint_world": endpoint_world,
        "h3_base_height_m": float(endpoint_pose[2]),
        "h3_roll_rad": roll,
        "h3_pitch_rad": pitch,
        "h3_solver_finite": h3_solver_finite,
        "h3_disallowed_contact": h3_contact,
        "physics_contact": physics_contact,
        "stuck": stuck,
        "successor_viable": successor_viable,
        "oracle_admissible": oracle_admissible,
        "port_progress_m": port_progress,
        "lateral_error_m": lateral_error,
        "angular_error_rad": angular_error,
        "positive_port_progress": port_progress > 0.0,
        "left_source_region": first_source_exit is not None,
        "first_source_exit_sample_index": first_source_exit,
        "reached_target_node": first_target_entry is not None,
        "first_target_entry_sample_index": first_target_entry,
        "source_at_endpoint": bool(source[-1]),
        "selected_edge_at_endpoint": bool(selected_member[-1]),
        "competing_edge_at_endpoint": bool(competing_member[-1]),
        "target_at_endpoint": bool(target[-1]),
        "command_tracking_rows": command_rows,
    }


def _compare_raw_branch_row(
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    physical: Mapping[str, Any],
    geometry: Mapping[str, Any],
    label: str,
) -> None:
    for field in (
        "entered_correct_edge", "entered_wrong_edge", "no_edge", "h3_solver_finite",
        "h3_disallowed_contact", "physics_contact", "stuck", "successor_viable",
        "oracle_admissible", "positive_port_progress", "left_source_region",
        "reached_target_node",
    ):
        if row.get(field) is not raw[field]:
            raise RegenerationError(f"{label}.{field} differs from raw physical trace")
    for field in (
        "first_source_exit_sample_index",
        "first_target_entry_sample_index",
    ):
        if row.get(field) != raw[field]:
            raise RegenerationError(f"{label}.{field} differs from raw physical trace")
    for field in (
        "h3_base_height_m", "h3_roll_rad", "h3_pitch_rad", "port_progress_m",
        "lateral_error_m", "angular_error_rad",
    ):
        _require_close(row.get(field), raw[field], f"{label}.{field}")
    for horizon, field in (
        ("H1", "h1_endpoint_body"),
        ("H2", "h2_endpoint_body"),
        ("H3", "h3_endpoint_body"),
    ):
        _require_vector_close(row.get(field), raw["endpoints"][horizon], f"{label}.{field}")
    _require_vector_close(
        row.get("h3_endpoint_world"), raw["endpoint_world"], f"{label}.h3_endpoint_world"
    )

    selected = raw["selected_crossing"]
    selected_fields = (
        "port_crossing_sample_before", "port_crossing_sample_after",
        "port_crossing_fraction", "port_crossing_directed_normal_dot",
        "port_crossing_lateral_fraction",
    )
    if raw["entered_correct_edge"]:
        assert selected is not None
        expected_selected = (
            selected["sample_before"], selected["sample_after"], selected["fraction"],
            selected["normal_dot_displacement_m"], selected["lateral_fraction"],
        )
        for field, expected in zip(selected_fields, expected_selected):
            if isinstance(expected, int):
                if row.get(field) != expected:
                    raise RegenerationError(f"{label}.{field} differs from raw trace")
            else:
                _require_close(row.get(field), expected, f"{label}.{field}")
        _require_vector_close(
            row.get("port_crossing_displacement_world_xy"),
            selected["displacement_world_xy"],
            f"{label}.port_crossing_displacement_world_xy",
        )
        _require_close(
            row.get("port_crossing_direction_heading_world_rad"),
            selected["direction_heading_world_rad"],
            f"{label}.port_crossing_direction_heading_world_rad",
        )
        if row.get("beyond_port_consecutive_physics_samples") != raw["dwell"][
            "beyond_port_consecutive_physics_samples"
        ]:
            raise RegenerationError(f"{label} selected-port dwell differs from raw trace")
        if row.get("target_entered_before_dwell_complete") is not raw["dwell"][
            "target_entered_before_dwell_complete"
        ]:
            raise RegenerationError(f"{label} early target differs from raw trace")
        if row.get("competing_port_crossing_first") is not False:
            raise RegenerationError(f"{label} competing-first disposition drift")
    else:
        if any(
            row.get(field) is not None
            for field in (
                *selected_fields,
                "port_crossing_displacement_world_xy",
                "port_crossing_direction_heading_world_rad",
            )
        ):
            raise RegenerationError(f"{label} fabricates selected-port crossing evidence")
        if row.get("beyond_port_consecutive_physics_samples") != 0:
            raise RegenerationError(f"{label} fabricates selected-port dwell evidence")

    competing = raw["competing_crossing"]
    wrong_fields = (
        "first_wrong_edge_id", "wrong_port_crossing_sample_before",
        "wrong_port_crossing_sample_after", "wrong_port_crossing_fraction",
        "wrong_port_crossing_directed_normal_dot", "wrong_port_crossing_lateral_fraction",
    )
    if raw["entered_wrong_edge"]:
        assert competing is not None
        expected_wrong = (
            competing["edge_id"], competing["sample_before"], competing["sample_after"],
            competing["fraction"], competing["normal_dot_displacement_m"],
            competing["lateral_fraction"],
        )
        for field, expected in zip(wrong_fields, expected_wrong):
            if isinstance(expected, (str, int)):
                if row.get(field) != expected:
                    raise RegenerationError(f"{label}.{field} differs from raw trace")
            else:
                _require_close(row.get(field), expected, f"{label}.{field}")
        _require_vector_close(
            row.get("wrong_port_crossing_displacement_world_xy"),
            competing["displacement_world_xy"],
            f"{label}.wrong_port_crossing_displacement_world_xy",
        )
        _require_close(
            row.get("wrong_port_crossing_direction_heading_world_rad"),
            competing["direction_heading_world_rad"],
            f"{label}.wrong_port_crossing_direction_heading_world_rad",
        )
        if row.get("competing_port_crossing_first") is not True:
            raise RegenerationError(f"{label} wrong-edge precedence drift")
    elif any(
        row.get(field) is not None
        for field in (
            *wrong_fields,
            "wrong_port_crossing_displacement_world_xy",
            "wrong_port_crossing_direction_heading_world_rad",
        )
    ):
        raise RegenerationError(f"{label} fabricates wrong-port crossing evidence")

    expected_requested = [value["requested_command"] for value in raw["command_tracking_rows"]]
    expected_applied = [value["post_slew_command"] for value in raw["command_tracking_rows"]]
    if row.get("requested_commands") != expected_requested:
        raise RegenerationError(f"{label} requested plan differs from raw trace")
    if row.get("post_slew_applied_commands") != expected_applied:
        raise RegenerationError(f"{label} post-slew plan differs from raw trace")
    tracking_rows = row.get("command_tracking_rows")
    if not isinstance(tracking_rows, list) or len(tracking_rows) != 15:
        raise RegenerationError(f"{label} command tracking row-count drift")
    tracking_authority = physical["command_tracking_authority"]
    threshold = float(tracking_authority["active_command_threshold"])
    for observed, expected in zip(tracking_rows, raw["command_tracking_rows"]):
        if not isinstance(observed, Mapping):
            raise RegenerationError(f"{label} command tracking row is not an object")
        for field in ("command_tick_index", "requested_command", "post_slew_command"):
            if observed.get(field) != expected[field]:
                raise RegenerationError(f"{label} command tracking {field} drift")
        _require_vector_close(
            observed.get("mean_achieved_body_velocity"),
            expected["mean_achieved_body_velocity"],
            f"{label}.mean_achieved_body_velocity",
        )
        if observed.get("active_vx") is not (
            abs(expected["post_slew_command"][0]) > threshold
        ) or observed.get("active_yaw") is not (
            abs(expected["post_slew_command"][2]) > threshold
        ):
            raise RegenerationError(f"{label} command activity differs from raw trace")

    selected_edge_id = geometry["selected_edge"]["edge_id"]
    expected_endpoint_edge = (
        selected_edge_id
        if (
            raw["target_at_endpoint"]
            or raw["entered_correct_edge"]
            or raw["selected_edge_at_endpoint"]
        )
        else (
            raw["competing_crossing"]["edge_id"]
            if raw["entered_wrong_edge"] and raw["competing_crossing"] is not None
            else None
        )
    )
    if row.get("endpoint_edge_id") != expected_endpoint_edge:
        raise RegenerationError(f"{label} endpoint edge differs from raw memberships")
    expected_endpoint_node = (
        "target" if raw["target_at_endpoint"] else ("source" if raw["source_at_endpoint"] else None)
    )
    if row.get("endpoint_node_id") != expected_endpoint_node:
        raise RegenerationError(f"{label} endpoint node differs from raw memberships")


def _validate_raw_candidate_and_repeat_evidence(
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
    inspections: Mapping[str, Mapping[str, Any]],
    captured: Mapping[str, Mapping[str, Sequence[bytes]]],
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    geometry_rows = physical["specs"]
    geometry_by_state = {
        row["state_id"]: row for row in geometry_rows if isinstance(row, Mapping)
    }
    panel_rows = documents["panel_manifest"]["states"]
    snapshot_capture = captured.get("state_snapshots.npz", {})
    initial_poses: dict[str, tuple[float, ...]] = {}
    for state_index, state in enumerate(panel_rows):
        payloads = snapshot_capture.get("base_pose_world")
        if not isinstance(payloads, list) or len(payloads) != EXPECTED_STATE_COUNT:
            raise RegenerationError("snapshot base-pose raw capture drift")
        initial_poses[state["state_id"]] = _float64_rows(
            payloads[state_index], 7, 1, f"snapshot base pose {state_index}"
        )[0]
    port_by_state = {
        row["state_id"]: row for row in documents["edge_port_index"]["records"]
    }
    raw_fanout: dict[str, list[dict[str, Any]]] = {}
    projections: list[dict[str, Any]] = []
    fanout_start = EXPECTED_RESET_FIXTURE_TRACE_COUNT
    for row_index, row in enumerate(ledgers["candidate_fanout"]):
        state_id = row["state_id"]
        geometry = geometry_by_state.get(state_id)
        if geometry is None:
            raise RegenerationError("fanout state lacks physical geometry authority")
        raw = _raw_candidate_projection(
            trace_index=fanout_start + row_index,
            state_index=row_index // EXPECTED_CANDIDATES_PER_STATE,
            geometry=geometry,
            initial_pose=initial_poses[state_id],
            port=port_by_state[state_id],
            inspections=inspections,
            captured=captured,
            physical=physical,
        )
        _compare_raw_branch_row(row, raw, physical, geometry, f"fanout[{row_index}]")
        raw_fanout.setdefault(state_id, []).append(raw)
        projections.append(
            {
                "branch_id": row["branch_id"],
                "entered_correct_edge": raw["entered_correct_edge"],
                "entered_wrong_edge": raw["entered_wrong_edge"],
                "physics_contact": raw["physics_contact"],
                "stuck": raw["stuck"],
                "successor_viable": raw["successor_viable"],
                "port_progress_m": raw["port_progress_m"],
            }
        )

    fanout_by_state = {
        state_id: sorted(
            [row for row in ledgers["candidate_fanout"] if row["state_id"] == state_id],
            key=lambda row: row["candidate_index"],
        )
        for state_id in raw_fanout
    }
    repeat_start = fanout_start + EXPECTED_FANOUT_ROW_COUNT
    repeat_projections: list[dict[str, Any]] = []
    candidate_inspection = inspections["candidate_traces.npz"]
    applied_member = physical["candidate_member_mapping"]["post_slew_command"]
    applied_hashes = candidate_inspection["members"][applied_member][
        "row_or_slice_sha256s"
    ]
    for repeat_index, row in enumerate(ledgers["repeated_execution"]):
        state_id = row["state_id"]
        geometry = geometry_by_state[state_id]
        raw = _raw_candidate_projection(
            trace_index=repeat_start + repeat_index,
            state_index=next(
                index for index, value in enumerate(panel_rows) if value["state_id"] == state_id
            ),
            geometry=geometry,
            initial_pose=initial_poses[state_id],
            port=port_by_state[state_id],
            inspections=inspections,
            captured=captured,
            physical=physical,
        )
        source_index = row["source_candidate_index"]
        source_row = fanout_by_state[state_id][source_index]
        source_raw = raw_fanout[state_id][source_index]
        if row.get("repeat_correct_edge_execution") is not raw["entered_correct_edge"]:
            raise RegenerationError("repeat correctness differs from raw trace")
        _require_vector_close(
            row.get("repeat_endpoint_body"), raw["endpoints"]["H3"],
            f"repeat[{repeat_index}].repeat_endpoint_body",
        )
        if row.get("physics_contact") is not raw["physics_contact"] or row.get(
            "stuck"
        ) is not raw["stuck"]:
            raise RegenerationError("repeat adverse outcome differs from raw trace")
        expected_endpoint_edge = (
            geometry["selected_edge"]["edge_id"]
            if (
                raw["target_at_endpoint"]
                or raw["entered_correct_edge"]
                or raw["selected_edge_at_endpoint"]
            )
            else (
                raw["competing_crossing"]["edge_id"]
                if raw["entered_wrong_edge"] and raw["competing_crossing"]
                else None
            )
        )
        if row.get("repeat_endpoint_edge_id") != expected_endpoint_edge:
            raise RegenerationError("repeat endpoint edge differs from raw trace")
        if row.get("repeat_applied_command_sequence_sha256") != applied_hashes[
            repeat_start + repeat_index
        ]:
            raise RegenerationError("repeat applied-command digest differs from NPZ")
        if row.get("source_applied_command_sequence_sha256") != applied_hashes[
            fanout_start
            + next(
                index
                for index, candidate in enumerate(ledgers["candidate_fanout"])
                if candidate["branch_id"] == source_row["branch_id"]
            )
        ]:
            raise RegenerationError("repeat source-command digest differs from NPZ")
        position_error = math.hypot(
            source_raw["endpoints"]["H3"][0] - raw["endpoints"]["H3"][0],
            source_raw["endpoints"]["H3"][1] - raw["endpoints"]["H3"][1],
        )
        heading_error = _angle_difference(
            source_raw["endpoints"]["H3"][2], raw["endpoints"]["H3"][2]
        )
        _require_close(
            row.get("endpoint_position_error_m"), position_error,
            f"repeat[{repeat_index}].endpoint_position_error_m",
        )
        _require_close(
            row.get("endpoint_heading_error_rad"), heading_error,
            f"repeat[{repeat_index}].endpoint_heading_error_rad",
        )
        repeat_projections.append(
            {
                "repeat_id": row["repeat_id"],
                "correct": raw["entered_correct_edge"],
                "contact": raw["physics_contact"],
                "stuck": raw["stuck"],
                "endpoint_position_error_m": position_error,
                "endpoint_heading_error_rad": heading_error,
            }
        )
    return {
        "fanout_trace_count": len(projections),
        "repeat_trace_count": len(repeat_projections),
        "fanout_raw_projection_digest": canonical_digest(projections),
        "repeat_raw_projection_digest": canonical_digest(repeat_projections),
        "all_candidate_and_repeat_labels_regenerated": True,
    }


def _validate_npz_cross_links(
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
    inspections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently join every public payload slice to its persisted bytes."""

    snapshots = inspections["state_snapshots.npz"]
    teachers = inspections["teacher_traces.npz"]
    rgb = inspections["rgb_observations.npz"]
    latents = inspections["canonical_latents.npz"]
    candidate = inspections["candidate_traces.npz"]

    _require_file_binding(
        documents["state_snapshot_index"]["snapshots_file"],
        snapshots,
        "state_snapshot_index.snapshots_file",
    )
    _require_file_binding(
        documents["teacher_trace_index"]["traces_file"],
        teachers,
        "teacher_trace_index.traces_file",
    )
    _require_file_binding(
        documents["pixel_index"]["rgb_file"], rgb, "pixel_index.rgb_file"
    )
    _require_file_binding(
        documents["latent_index"]["latents_file"],
        latents,
        "latent_index.latents_file",
    )

    snapshot_offsets = snapshots["members"]["snapshot_offsets"]["offset_values"]
    teacher_offsets = teachers["members"]["trace_offsets"]["offset_values"]
    candidate_offsets = candidate["members"]["trace_offsets"]["offset_values"]
    if (
        not isinstance(snapshot_offsets, list)
        or len(snapshot_offsets) != EXPECTED_STATE_COUNT + 1
    ):
        raise RegenerationError("snapshot offset cardinality drift")
    if (
        not isinstance(teacher_offsets, list)
        or len(teacher_offsets) != len(
            documents["teacher_trace_index"]["records"]
        )
        + 1
    ):
        raise RegenerationError("teacher offset cardinality drift")
    if (
        not isinstance(candidate_offsets, list)
        or len(candidate_offsets) != EXPECTED_CANDIDATE_TRACE_COUNT + 1
    ):
        raise RegenerationError("candidate trace offset cardinality drift")
    expected_trace_lengths = [EXPECTED_PHYSICS_SAMPLES_PER_BRANCH] * (
        EXPECTED_CANDIDATE_TRACE_COUNT
    )
    if [
        stop - start for start, stop in zip(candidate_offsets, candidate_offsets[1:])
    ] != expected_trace_lengths:
        raise RegenerationError(
            "candidate trace offsets do not encode 960 exact 750-sample branches"
        )

    snapshot_records = documents["state_snapshot_index"]["records"]
    reset_trace_indices: list[int] = []
    snapshot_row_hash_fields = {
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
    extended_snapshot_members_present = set(snapshot_row_hash_fields.values()).issubset(
        snapshots["members"]
    )
    extended_snapshot_fields_present = all(
        "teacher_initial_state_sha256" in row
        and set(snapshot_row_hash_fields).issubset(row)
        and all(
            isinstance(trial, Mapping) and "restored_snapshot_sha256" in trial
            for trial in row.get("reset_trials", ())
        )
        for row in snapshot_records
    )
    if extended_snapshot_members_present is not extended_snapshot_fields_present:
        raise RegenerationError("snapshot raw-byte/hash authority is only partially present")
    extended_snapshot_evidence = extended_snapshot_members_present
    for state_index, row in enumerate(snapshot_records):
        _require_slice_binding(
            row["snapshot_payload"],
            snapshots,
            member="snapshot_payload_bytes",
            slice_index=state_index,
            start=snapshot_offsets[state_index],
            stop=snapshot_offsets[state_index + 1],
            label=f"snapshot[{state_index}]",
        )
        payload_sha = snapshots["members"]["snapshot_payload_bytes"][
            "row_or_slice_sha256s"
        ][state_index]
        if extended_snapshot_evidence:
            if row.get("teacher_initial_state_sha256") != payload_sha:
                raise RegenerationError(
                    f"snapshot[{state_index}] teacher initial state differs from serialized bytes"
                )
            for field, member in snapshot_row_hash_fields.items():
                if row.get(field) != snapshots["members"][member][
                    "row_or_slice_sha256s"
                ][state_index]:
                    raise RegenerationError(
                        f"snapshot[{state_index}].{field} differs from snapshot NPZ bytes"
                    )
        trials = row["reset_trials"]
        if not isinstance(trials, list) or len(trials) != 2:
            raise RegenerationError(f"snapshot[{state_index}] reset trial coverage drift")
        for trial_index, trial in enumerate(trials):
            trace_index = state_index * 2 + trial_index
            if trial.get("trace_index") != trace_index:
                raise RegenerationError("reset fixture trace range/order drift")
            if (
                extended_snapshot_evidence
                and trial.get("restored_snapshot_sha256") != payload_sha
            ):
                raise RegenerationError(
                    "reset fixture restored snapshot differs from retained serialized bytes"
                )
            _require_slice_binding(
                trial.get("trace_slice"),
                candidate,
                member="timestamp_s",
                slice_index=trace_index,
                start=candidate_offsets[trace_index],
                stop=candidate_offsets[trace_index + 1],
                label=f"reset fixture trace {trace_index}",
            )
            _require_trace_hash_map(
                trial.get("trace_array_slice_sha256s"),
                candidate,
                trace_index=trace_index,
                label=f"reset fixture trace {trace_index}",
            )
            pixel_record = documents["pixel_index"]["records"][state_index]
            if (
                "current_rgb_sha256" in trial
                and trial.get("current_rgb_sha256") != pixel_record.get("pixel_sha256")
            ):
                raise RegenerationError(
                    "reset fixture current RGB differs from the retained state image"
                )
            reset_trace_indices.append(trace_index)
    if reset_trace_indices != list(range(EXPECTED_RESET_FIXTURE_TRACE_COUNT)):
        raise RegenerationError("reset fixture trace identity coverage drift")

    teacher_records = documents["teacher_trace_index"]["records"]
    for trace_index, row in enumerate(teacher_records):
        start, stop = teacher_offsets[trace_index : trace_index + 2]
        if row.get("sample_count") != stop - start:
            raise RegenerationError(f"teacher trace {trace_index} sample count drift")
        _require_slice_binding(
            row.get("trace_slice"),
            teachers,
            member="timestamp_s",
            slice_index=trace_index,
            start=start,
            stop=stop,
            label=f"teacher trace {trace_index}",
        )
        _require_trace_hash_map(
            row.get("trace_array_slice_sha256s"),
            teachers,
            trace_index=trace_index,
            label=f"teacher trace {trace_index}",
        )

    fanout_start = EXPECTED_RESET_FIXTURE_TRACE_COUNT
    for row_index, row in enumerate(ledgers["candidate_fanout"]):
        trace_index = fanout_start + row_index
        if (
            row.get("trace_index") != trace_index
            or row.get("physics_sample_count") != EXPECTED_PHYSICS_SAMPLES_PER_BRANCH
        ):
            raise RegenerationError("candidate-fanout trace range/count drift")
        _require_slice_binding(
            row.get("trace_slice"),
            candidate,
            member="timestamp_s",
            slice_index=trace_index,
            start=candidate_offsets[trace_index],
            stop=candidate_offsets[trace_index + 1],
            label=f"candidate fanout trace {trace_index}",
        )
        _require_trace_hash_map(
            row.get("trace_array_slice_sha256s"),
            candidate,
            trace_index=trace_index,
            label=f"candidate fanout trace {trace_index}",
        )

    repeat_start = fanout_start + EXPECTED_FANOUT_ROW_COUNT
    for row_index, row in enumerate(ledgers["repeated_execution"]):
        trace_index = repeat_start + row_index
        if row.get("trace_index") != trace_index:
            raise RegenerationError("repeat trace range/order drift")
        _require_slice_binding(
            row.get("trace_slice"),
            candidate,
            member="timestamp_s",
            slice_index=trace_index,
            start=candidate_offsets[trace_index],
            stop=candidate_offsets[trace_index + 1],
            label=f"repeated execution trace {trace_index}",
        )
        _require_trace_hash_map(
            row.get("trace_array_slice_sha256s"),
            candidate,
            trace_index=trace_index,
            label=f"repeated execution trace {trace_index}",
        )
    if repeat_start + len(ledgers["repeated_execution"]) != EXPECTED_CANDIDATE_TRACE_COUNT:
        raise RegenerationError("candidate trace ranges do not exhaust 960 slices")

    rgb_hashes = rgb["members"]["rgb"]["row_or_slice_sha256s"]
    pixel_records = documents["pixel_index"]["records"]
    if len(rgb_hashes) != len(pixel_records):
        raise RegenerationError("RGB row/hash cardinality drift")
    canonical_pixels = sorted(set(rgb_hashes))
    if documents["pixel_index"].get("unique_pixel_count") != len(canonical_pixels):
        raise RegenerationError("pixel unique-count drift")
    pixel_to_index = {digest: index for index, digest in enumerate(canonical_pixels)}
    for row_index, (row, digest) in enumerate(zip(pixel_records, rgb_hashes)):
        if (
            row.get("rgb_row_index") != row_index
            or row.get("pixel_sha256") != digest
            or row.get("row_sha256") != digest
            or row.get("canonical_pixel_index") != pixel_to_index[digest]
        ):
            raise RegenerationError(f"pixel row {row_index} differs from RGB NPZ bytes")

    latent_records = documents["latent_index"]["records"]
    raw_hashes = latents["members"]["raw_tokens"]["row_or_slice_sha256s"]
    descriptor_hashes = latents["members"]["spatial_descriptors"][
        "row_or_slice_sha256s"
    ]
    if not (
        len(latent_records) == len(canonical_pixels) == len(raw_hashes) == len(descriptor_hashes)
    ):
        raise RegenerationError("canonical latent row coverage drift")
    for index, row in enumerate(latent_records):
        # The production row authority requires this field.  Retain support
        # for deliberately reduced synthetic custody authorities used by the
        # system-Python unit tests.
        if "preprocessed_tensor_sha256" in row:
            _sha256(
                row["preprocessed_tensor_sha256"],
                f"latent row {index} preprocessed tensor SHA",
            )
        if (
            row.get("canonical_pixel_index") != index
            or row.get("pixel_sha256") != canonical_pixels[index]
            or row.get("raw_token_row_index") != index
            or row.get("raw_token_sha256") != raw_hashes[index]
            or row.get("spatial_descriptor_row_index") != index
            or row.get("spatial_descriptor_sha256") != descriptor_hashes[index]
        ):
            raise RegenerationError(f"latent row {index} differs from canonical NPZ bytes")

    return {
        "snapshot_slices": len(snapshot_records),
        "teacher_slices": len(teacher_records),
        "candidate_trace_slices": EXPECTED_CANDIDATE_TRACE_COUNT,
        "reset_fixture_trace_slices": EXPECTED_RESET_FIXTURE_TRACE_COUNT,
        "fanout_trace_slices": len(ledgers["candidate_fanout"]),
        "repeat_trace_slices": len(ledgers["repeated_execution"]),
        "rgb_rows": len(rgb_hashes),
        "unique_pixel_rows": len(canonical_pixels),
        "canonical_latent_rows": len(latent_records),
        "all_slice_hashes_exact": True,
    }


def _compact_npz_validation(inspections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for leaf, inspection in inspections.items():
        members = inspection["members"]
        output[leaf] = {
            "path": inspection["path"],
            "bytes": inspection["bytes"],
            "sha256": inspection["sha256"],
            "member_count": len(members),
            "member_inspection_digest": canonical_digest(members),
            "row_or_slice_hash_count": sum(
                len(value["row_or_slice_sha256s"]) for value in members.values()
            ),
        }
    return output


def _contract_source_commit(contract: Mapping[str, Any]) -> str:
    commit = contract.get("source_freeze_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RegenerationError("runtime contract source-freeze binding drift")
    return commit


def _validate_publication_if_present(
    root: Path,
    root_fd: int,
    inventory: Sequence[str],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    *,
    recomputed_metrics: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    enforce_semantics: bool,
) -> dict[str, Any] | None:
    if not set(PUBLICATION_FILES).issubset(inventory):
        return None
    result_raw, result = _load_json_at(root_fd, "result.json")
    report_raw = _read_regular_at(root_fd, "result.md")
    assert report_raw is not None
    try:
        report_text = report_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError("result.md is not UTF-8") from exc
    if not report_text or not report_raw.endswith(b"\n"):
        raise RegenerationError("result.md must be nonempty and LF terminated")
    _manifest_raw, manifest = _load_json_at(root_fd, "file_hashes.json")
    if set(manifest) != {
        "schema",
        "root",
        "files",
        "file_count_excluding_self",
        "bytes_excluding_self",
        "file_hashes_self_sha256_excluded",
        "content_digest",
    }:
        raise RegenerationError("file_hashes.json root field-set drift")
    if (
        manifest["schema"]
        != "physical_graph_edge_handoff_qualification_v1.file_hashes.v1"
        or manifest["root"] != str(root)
        or manifest["file_hashes_self_sha256_excluded"] is not True
    ):
        raise RegenerationError("file_hashes.json identity/self-exclusion drift")
    expected = {
        **{name: dict(value) for name, value in scientific_bindings.items()},
        "result.json": _binding("result.json", result_raw),
        "result.md": _binding("result.md", report_raw),
    }
    expected_rows = [expected[name] for name in sorted(expected)]
    if (
        manifest["files"] != expected_rows
        or manifest["file_count_excluding_self"] != len(expected_rows)
        or manifest["bytes_excluding_self"]
        != sum(row["bytes"] for row in expected_rows)
    ):
        raise RegenerationError("file_hashes.json differs from exact live bytes")
    if not enforce_semantics:
        return result

    result_fields = {
        "schema",
        "experiment_id",
        "source_commit",
        "source_baseline_commit",
        "predecessor_result_commit",
        "development_only",
        "primary_classification",
        "secondary_classifications",
        "next_experiment",
        "selected_target_id",
        "evidence_counts",
        "panel_metrics",
        "development_metrics",
        "heldout_metrics",
        "repeatability",
        "command_tracking",
        "runtime_environments",
        "stratified_metrics",
        "gate",
        "component_failures",
        "metrics_sha256",
        "independent_reducer_receipt_sha256",
        "runtime_seconds",
        "scientific_storage_bytes",
        "models_trained",
        "prohibited_components_trained_or_implemented",
        "content_digest",
    }
    if set(result) != result_fields:
        raise RegenerationError("result.json root field-set drift")
    scientific_contract = runtime_contract.get("scientific_contract")
    if not isinstance(scientific_contract, Mapping):
        raise RegenerationError("result validation lacks scientific contract")
    v2_context = scientific_contract.get("v2_context_binding")
    if not isinstance(v2_context, Mapping):
        raise RegenerationError("result validation lacks predecessor context")
    expected_projection = {
        "schema": "physical_graph_edge_handoff_qualification_v1.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": runtime_contract.get("source_freeze_commit"),
        "source_baseline_commit": runtime_contract.get("source_baseline_commit"),
        "predecessor_result_commit": v2_context.get("result_commit"),
        "development_only": True,
        "primary_classification": recomputed_metrics["primary_classification"],
        "secondary_classifications": recomputed_metrics[
            "secondary_classifications"
        ],
        "next_experiment": recomputed_metrics["next_experiment"],
        "selected_target_id": recomputed_metrics["development"][
            "selected_target_id"
        ],
        "evidence_counts": recomputed_metrics["evidence_counts"],
        "panel_metrics": recomputed_metrics["panel"],
        "development_metrics": recomputed_metrics["development"],
        "heldout_metrics": recomputed_metrics["heldout"],
        "repeatability": recomputed_metrics["repeatability"],
        "command_tracking": recomputed_metrics["command_tracking"],
        "runtime_environments": recomputed_metrics["runtime_environments"],
        "stratified_metrics": recomputed_metrics["stratified"],
        "gate": recomputed_metrics["gate"],
        "component_failures": recomputed_metrics["component_failures"],
        "metrics_sha256": scientific_bindings["metrics.json"]["sha256"],
        "scientific_storage_bytes": sum(
            int(binding["bytes"]) for binding in scientific_bindings.values()
        ),
        "models_trained": 0,
        "prohibited_components_trained_or_implemented": [],
    }
    for field, expected_value in expected_projection.items():
        if result.get(field) != expected_value:
            raise RegenerationError(f"result.json {field} differs from reduced evidence")
    runtime_seconds = result.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) < 0.0
    ):
        raise RegenerationError("result.json runtime_seconds is invalid")
    _sha256(
        result.get("independent_reducer_receipt_sha256"),
        "result.json independent reducer receipt SHA",
    )

    target_lines = "\n".join(
        f"- {row['target_id']}: {json.dumps(row, sort_keys=True)}"
        for row in recomputed_metrics["development"]["target_summaries"]
    )
    condition_lines = "\n".join(
        f"- {row['condition_id']}: {json.dumps(row, sort_keys=True)}"
        for row in recomputed_metrics["heldout"]["condition_summaries"]
    )
    expected_report = (
        f"# {EXPERIMENT_ID}\n\n"
        f"Primary classification: {recomputed_metrics['primary_classification']}\n\n"
        "Secondary classifications: "
        f"{', '.join(recomputed_metrics['secondary_classifications']) or 'none'}\n\n"
        "Selected target: "
        f"{recomputed_metrics['development']['selected_target_id']}\n\n"
        f"Handoff gate passed: {str(bool(recomputed_metrics['gate']['passed'])).lower()}\n\n"
        f"Next experiment: {recomputed_metrics['next_experiment']}\n\n"
        "## Physical evidence\n\n"
        "Evidence counts: "
        f"{json.dumps(recomputed_metrics['evidence_counts'], sort_keys=True)}\n\n"
        "All 64 selected decision states retained exact serialized solver, controller, "
        "command-history, and RNG snapshots. Each passed two independent 750-sample "
        "serialized-restore fixtures. All 256 prospective teacher traces, 64 actual "
        "teacher-derived directed ports, 768 candidate fanouts, and 64 repeat traces "
        "are persisted at the 2 ms physics rate.\n\n"
        "Held-out candidate coverage: "
        f"{json.dumps(recomputed_metrics['panel'], sort_keys=True)}\n\n"
        "## Development target selection\n\n"
        f"{target_lines}\n\n"
        "## Held-out conditions\n\n"
        f"{condition_lines}\n\n"
        "## Reset, repeat, and controller qualification\n\n"
        "Repeatability: "
        f"{json.dumps(recomputed_metrics['repeatability'], sort_keys=True)}\n\n"
        "Command tracking: "
        f"{json.dumps(recomputed_metrics['command_tracking'], sort_keys=True)}\n\n"
        "## Runtime environments\n\n"
        f"{json.dumps(recomputed_metrics['runtime_environments'], sort_keys=True)}\n\n"
        "Stratified outcomes: "
        f"{json.dumps(recomputed_metrics['stratified'], sort_keys=True)}\n\n"
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
        raise RegenerationError("result.md differs from exact reduced presentation")
    return result


def _validate_v2_context(contract: Mapping[str, Any], module: Any) -> dict[str, Any]:
    """Hash every frozen V2 context leaf named by the runtime contract.

    This is byte custody only: no V2 outcome document or ledger is interpreted,
    and no invalid V1 artifact is opened.  The pure runtime-contract validator
    separately establishes that this nested authority equals the preregistered
    one.
    """

    scientific = contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("runtime contract lacks its scientific contract")
    context = scientific.get("v2_context_binding")
    if not isinstance(context, Mapping):
        raise RegenerationError("runtime contract lacks frozen V2 context binding")
    projected = _call(module, "predecessor_context_binding", contract)
    if not isinstance(projected, Mapping) or dict(projected) != dict(context):
        raise RegenerationError("pure predecessor-context projection drift")
    if context.get("runtime_reuse_authorized") is not False:
        raise RegenerationError("V2 runtime evidence must remain nonreusable")
    for commit_field in ("source_freeze_commit", "result_commit"):
        value = context.get(commit_field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RegenerationError(f"V2 {commit_field} binding drift")
    bindings = context.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_V2_CONTEXT_LEAVES):
        raise RegenerationError("V2 context leaf authority drift")
    predecessor = contract.get("predecessor_result_binding")
    if not isinstance(predecessor, Mapping) or set(predecessor) != {
        "role",
        "path",
        "bytes",
        "sha256",
        "kind",
    }:
        raise RegenerationError("V2 predecessor-result binding field drift")
    predecessor_path = Path(_string(predecessor["path"], "predecessor result path"))
    if predecessor_path.name != "result.json":
        raise RegenerationError("V2 predecessor result must bind result.json")
    result_authority = bindings["result.json"]
    if not isinstance(result_authority, Mapping) or set(result_authority) != {
        "bytes",
        "sha256",
    }:
        raise RegenerationError("V2 result context binding field drift")
    if (
        predecessor["bytes"] != result_authority["bytes"]
        or predecessor["sha256"] != result_authority["sha256"]
    ):
        raise RegenerationError("V2 predecessor/result context bindings disagree")

    root = predecessor_path.parent
    validations: list[dict[str, Any]] = []
    for leaf in EXPECTED_V2_CONTEXT_LEAVES:
        row = bindings[leaf]
        if not isinstance(row, Mapping) or set(row) != {"bytes", "sha256"}:
            raise RegenerationError(f"V2 context binding field drift: {leaf}")
        validations.append(
            _stream_external_binding(
                {
                    "role": f"v2_context:{leaf}",
                    "kind": "frozen_v2_context_leaf_read_only",
                    "path": str(root / leaf),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                },
                f"v2_context[{leaf}]",
            )
        )
    return {
        "root": str(root),
        "source_freeze_commit": context["source_freeze_commit"],
        "result_commit": context["result_commit"],
        "runtime_reuse_authorized": False,
        "bound_leaf_count": len(validations),
        "bound_total_bytes": sum(row["bytes"] for row in validations),
        "bindings": validations,
    }


def build_regeneration_receipt(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild all metrics and custody evidence without executing science."""

    module = metrics_module if metrics_module is not None else _load_metrics_module()
    authority = _validate_authority(module)
    root, root_fd = _open_root(output_root, "official qualification root")
    try:
        inventory = _root_inventory(root_fd)
        raw_documents: dict[str, bytes] = {}
        documents_by_leaf: dict[str, dict[str, Any]] = {}
        for leaf in DOCUMENT_FILES:
            raw, document = _load_json_at(root_fd, leaf)
            raw_documents[leaf] = raw
            documents_by_leaf[leaf] = document
        raw_ledgers: dict[str, bytes] = {}
        ledgers_by_leaf: dict[str, list[dict[str, Any]]] = {}
        for leaf in LEDGER_FILES:
            raw, rows = _load_jsonl_at(root_fd, leaf)
            raw_ledgers[leaf] = raw
            ledgers_by_leaf[leaf] = rows

        contract = documents_by_leaf[CONTRACT_FILE]
        source_commit = _contract_source_commit(contract)
        observed_freeze = (
            _observe_source_freeze(module)
            if source_freeze_observation is None
            else dict(source_freeze_observation)
        )
        source_freeze = _validate_source_freeze(
            observed_freeze, expected_commit=source_commit
        )
        v2_context_validation = _validate_v2_context(contract, module)

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
            name: documents_by_leaf[leaf]
            for name, leaf in document_name_to_leaf.items()
        }
        document_validation = _validate_document_authorities(
            evidence_documents, authority
        )
        encoder_source_validation = (
            _validate_external_encoder_source(evidence_documents["latent_index"])
            if _metrics_module_name(module) == METRICS_MODULE
            else {"synthetic_fixture_not_scientific": True}
        )
        ledger_name_to_leaf = {
            "candidate_fanout": "candidate_fanout.jsonl",
            "heldout_ranker_scores": "heldout_ranker_scores.jsonl",
            "repeated_execution": "repeated_execution.jsonl",
        }
        evidence_ledgers = {
            name: ledgers_by_leaf[leaf]
            for name, leaf in ledger_name_to_leaf.items()
        }
        ledger_validation = _validate_ledger_authorities(
            evidence_ledgers, evidence_documents, authority
        )
        runtime_environment_validation = (
            _validate_runtime_environments(
                module, evidence_documents, evidence_ledgers
            )
            if _metrics_module_name(module) == METRICS_MODULE
            else {"synthetic_fixture_not_scientific": True}
        )

        symbols: dict[str, int] = {}
        captured_reset_slices: dict[str, list[bytes]] = {}
        captured_physical_slices: dict[str, dict[str, list[bytes]]] = {}
        npz_inspections = {
            leaf: _inspect_npz_at(
                root_fd,
                leaf,
                authority["npz_authorities"][leaf],
                symbols=symbols,
                captured_reset_slices=captured_reset_slices,
                captured_physical_slices=captured_physical_slices,
            )
            for leaf in PAYLOAD_FILES
        }
        if symbols.get("U") != evidence_documents["pixel_index"].get(
            "unique_pixel_count"
        ):
            raise RegenerationError("canonical latent U dimension differs from pixel index")
        pure_npz_validation = _call(
            module,
            "validate_npz_inspections",
            [npz_inspections[leaf] for leaf in PAYLOAD_FILES],
        )
        if (
            not isinstance(pure_npz_validation, Mapping)
            or set(pure_npz_validation) != set(PAYLOAD_FILES)
        ):
            raise RegenerationError("pure NPZ inspection projection drift")
        npz_cross_link_validation = _validate_npz_cross_links(
            evidence_documents, evidence_ledgers, npz_inspections
        )
        reset_pair_validation = _validate_reset_trace_pairs(
            evidence_documents["state_snapshot_index"],
            npz_inspections["candidate_traces.npz"],
            captured_reset_slices,
            authority,
        )
        physical_authority = authority["physical_trace_reduction_authority"]
        required_physical_sections = {
            "teacher_member_mapping",
            "candidate_member_mapping",
            "crossing_velocity",
            "command_tracking_authority",
            "specs",
            "physics_dt_s",
            "dwell_samples",
            "stuck",
            "successor_viable",
        }
        if required_physical_sections.issubset(physical_authority):
            raw_teacher_validation = _validate_raw_teacher_evidence(
                evidence_documents,
                npz_inspections,
                captured_physical_slices,
                physical_authority,
            )
            raw_candidate_validation = _validate_raw_candidate_and_repeat_evidence(
                evidence_documents,
                evidence_ledgers,
                npz_inspections,
                captured_physical_slices,
                physical_authority,
            )
        elif _metrics_module_name(module) == METRICS_MODULE:
            raise RegenerationError(
                "production physical trace-reduction authority is incomplete: "
                f"{sorted(required_physical_sections-set(physical_authority))}"
            )
        else:
            # Synthetic custody fixtures exercise raw byte/hash/reset machinery
            # without pretending to be production physical evidence.
            raw_teacher_validation = {"synthetic_fixture_not_scientific": True}
            raw_candidate_validation = {"synthetic_fixture_not_scientific": True}

        external_bindings = _call(module, "external_artifact_bindings", contract)
        if (
            not isinstance(external_bindings, list)
            or len(external_bindings) != len(EXPECTED_EXTERNAL_ARTIFACT_ROLES)
        ):
            raise RegenerationError("external artifact binding projection drift")
        external_validations = [
            _stream_external_binding(binding, f"external_artifact_bindings[{index}]")
            for index, binding in enumerate(external_bindings)
        ]
        if [row["role"] for row in external_validations] != authority[
            "external_artifact_roles"
        ]:
            raise RegenerationError("external checkpoint role order drift")

        evidence = {
            **evidence_documents,
            **evidence_ledgers,
            "npz_inspections": [npz_inspections[leaf] for leaf in PAYLOAD_FILES],
        }
        if set(evidence) != set(authority["evidence_keys"]):
            raise RegenerationError("assembled evidence key-set drift")
        recomputed = _call(module, "recompute_metrics", evidence)
        if not isinstance(recomputed, Mapping):
            raise RegenerationError("recompute_metrics() must return a metrics object")
        recomputed_raw, recomputed = _canonicalise_object(
            dict(recomputed), "recomputed metrics"
        )
        _validate_content_digest(recomputed, "recomputed metrics")
        metrics_raw = raw_documents["metrics.json"]
        if metrics_raw != recomputed_raw:
            raise RegenerationError("persisted metrics differ from exact independent rebuild")

        input_bindings: dict[str, dict[str, Any]] = {}
        for leaf, raw in {**raw_documents, **raw_ledgers}.items():
            input_bindings[leaf] = _binding(leaf, raw)
        for leaf, inspection in npz_inspections.items():
            input_bindings[leaf] = {
                field: inspection[field] for field in ("path", "bytes", "sha256")
            }
        if set(input_bindings) != set(SCIENTIFIC_FILES):
            raise RegenerationError("scientific input binding inventory drift")
        publication_result = _validate_publication_if_present(
            root,
            root_fd,
            inventory,
            input_bindings,
            # Use the canonical parsed form after the exact byte comparison so
            # tuple/list implementation details cannot create a false
            # presentation mismatch.
            recomputed_metrics=documents_by_leaf["metrics.json"],
            runtime_contract=contract,
            enforce_semantics=_metrics_module_name(module) == METRICS_MODULE,
        )

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "pass": True,
            "mode": "INDEPENDENT_PERSISTED_PHYSICAL_EVIDENCE_REDUCTION_ONLY",
            "reducer_authority_digest": authority["content_digest"],
            "source_freeze": source_freeze,
            "official_root_inventory": {
                "scientific_leaf_count": len(SCIENTIFIC_FILES),
                # Keep the external receipt byte-stable when the three
                # presentation leaves are added after scientific reduction.
                # ``inventory`` is still validated above; the registered
                # successful-root cardinality is the invariant recorded here.
                "allowed_leaf_count": len(ALL_OUTPUT_FILES),
                "allowed_leaf_names_validated": True,
                "publication_all_or_absent_validated": True,
                "publication_contract_validated_if_present": True,
            },
            "inputs": {name: input_bindings[name] for name in sorted(input_bindings)},
            "document_validation": document_validation,
            "ledger_validation": ledger_validation,
            "npz_validation": _compact_npz_validation(npz_inspections),
            "npz_symbols": {name: symbols[name] for name in sorted(symbols)},
            "pure_npz_validation_passed": True,
            "npz_cross_link_validation": npz_cross_link_validation,
            "reset_pair_validation": reset_pair_validation,
            "raw_teacher_physics_validation": raw_teacher_validation,
            "raw_candidate_and_repeat_physics_validation": raw_candidate_validation,
            "v2_context_validation": v2_context_validation,
            "external_artifact_validation": external_validations,
            "external_encoder_source_validation": encoder_source_validation,
            "runtime_environment_validation": runtime_environment_validation,
            "metrics_exact_byte_equal": True,
            "recomputed_metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
            "scientific_execution_counters": {
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
            },
        }
        if publication_result is not None and _metrics_module_name(module) == METRICS_MODULE:
            expected_receipt_sha = hashlib.sha256(
                canonical_document_bytes(receipt)
            ).hexdigest()
            if publication_result.get(
                "independent_reducer_receipt_sha256"
            ) != expected_receipt_sha:
                raise RegenerationError(
                    "result.json reducer-receipt SHA differs from exact external receipt"
                )
        _validate_no_self_digest(receipt, "external regeneration receipt")
        _reject_nonfinite(receipt, label="external regeneration receipt")
        return receipt
    finally:
        os.close(root_fd)


def _external_receipt_location(
    output_root: Path | str, output: Path | str
) -> tuple[int, str]:
    return CUSTODY._external_receipt_location(output_root, output)


def emit_regeneration_receipt(
    output_root: Path | str, output: Path | str, receipt: Mapping[str, Any]
) -> bytes:
    _validate_no_self_digest(receipt, "external regeneration receipt")
    raw = canonical_document_bytes(dict(receipt))
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            existing = _read_regular_at(parent_fd, leaf)
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


def verify_and_emit(
    output_root: Path | str,
    output: Path | str,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
    )
    emit_regeneration_receipt(output_root, output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str,
    output: Path | str,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        raw = _read_regular_at(parent_fd, leaf)
    finally:
        os.close(parent_fd)
    assert raw is not None
    supplied = parse_canonical_json(raw, label=str(output))
    if not isinstance(supplied, dict):
        raise RegenerationError("external regeneration receipt must be an object")
    _validate_no_self_digest(supplied, "external regeneration receipt")
    rebuilt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
    )
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("external regeneration receipt differs from exact rebuild")
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
                "status": "PASS",
                "receipt": str(args.output),
                "document_sha256": hashlib.sha256(
                    canonical_document_bytes(receipt)
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
