#!/usr/bin/env python3
"""Independent custody and reduction for snapshot-semantic PGEHQ V3.

The public evaluator runs under system Python.  It does not import torch,
Genesis, a model, an encoder, a ranker, or an experiment runner.  Historical
V1/V2 protocol-4 snapshot artifacts are inspected only through a restricted,
read-only worker running the frozen physical environment.  The worker returns
canonical semantic bytes; this process independently parses and hashes those
bytes before accepting any semantic identity.

The combined V1/V2 custody receipt and the V3 regeneration receipt are
ordinary canonical JSON documents outside every experiment root.  Neither has
a self digest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import pickletools
import stat
import struct
import subprocess
import sys
from typing import Any
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Frozen system-Python-safe dependency.  Importing it opens no experiment
# artifact and imports no runtime/model stack.
from scripts import evaluate_physical_graph_edge_handoff_qualification_v2 as V2E


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3"
V1_EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
V2_EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
SOURCE_PARENT_COMMIT = "10117870fe00bbfd8709932cab7b6e9df3be9bfc"
SOURCE_BASELINE_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
METRICS_MODULE = (
    "lewm.safety.physical_graph_edge_handoff_qualification_v3_metrics"
)
SEMANTICS_MODULE = (
    "lewm.safety.physical_graph_edge_handoff_snapshot_semantics_v1"
)

DEFAULT_V1_OFFICIAL_ROOT = Path(
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
DEFAULT_V2_OFFICIAL_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2"
)
DEFAULT_V2_MATERIAL_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2_material"
)
DEFAULT_V3_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v3"
)
DEFAULT_V3_MATERIAL_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v3_material"
)
DEFAULT_HISTORICAL_CUSTODY_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1_v2_custody_receipt.json"
)
DEFAULT_EXTERNAL_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v3_regeneration_receipt.json"
)

HISTORICAL_WORKER_INTERPRETER = Path(
    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
    "genesis_rocm_0_4_6_v1/bin/python"
)
HISTORICAL_CUSTODY_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v1_v2.custody_receipt.v1"
)
REGENERATION_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v3.regeneration_receipt.v1"
)
WORKER_MAGIC = b"PGEHQ-V3-HISTORICAL-SEMANTIC-WORKER\x00\x01"
SEMANTIC_MAGIC = b"PGEHQ-SNAPSHOT-SEMANTICS\x00\x01"

V3_SUCCESS_FILES = (
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
    "v1_v2_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_v3_first_eight_reproduction.json",
    "snapshot_equivalence_index.json",
    "snapshot_behavioural_probes.npz",
)
V3_MISMATCH_FILES = (
    "contract.json",
    "v1_v2_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_v3_first_eight_reproduction.json",
)
MISMATCH_DISPOSITION = (
    "V1_V2_V3_SEMANTIC_OR_BEHAVIOURAL_REPRODUCTION_MISMATCH"
)

_ROOT_INVENTORY_FIELDS = {
    "path", "file_count", "directory_count", "regular_file_apparent_bytes",
    "regular_file_allocated_bytes", "directory_allocated_bytes",
    "allocated_bytes", "files", "directories", "manifest_sha256",
}
_ROOT_FILE_FIELDS = {
    "path", "bytes", "allocated_bytes", "sha256", "device", "inode", "nlink",
}
_ROOT_DIRECTORY_FIELDS = {"path", "allocated_bytes", "device", "inode", "nlink"}
_HISTORICAL_VERSION_FIELDS = {
    "metadata_binding", "payload_binding", "artifact_file_sha256",
    "artifact_bytes", "snapshot_semantic_digest_v1", "semantic_payload_bytes",
    "semantic_payload_sha256", "semantic_evidence",
}
_HISTORICAL_PAIR_FIELDS = {
    "pool_index", "candidate_spec_id", "state_id", "scene_id", "episode_id",
    "graph_id", "v1", "v2", "payload_member_inventory_equal",
    "payload_member_dtypes_equal", "non_snapshot_member_shapes_equal",
    "non_snapshot_members_equal", "artifact_file_sha256_equal",
    "snapshot_semantic_digest_v1_equal", "semantic_evidence_equal",
    "qualified_equal", "rejection_reason_equal", "contact_sequence_equal",
    "stuck_equal", "pass",
}
_V2_TERMINAL_FIELDS = {
    "disposition", "official_leaf_count", "material_pair_count",
    "full_collection_authorized",
    "candidate_ranker_development_heldout_outcomes_opened",
    "external_regeneration_receipt_present", "first_eight_receipt_binding",
}

RegenerationError = V2E.RegenerationError
canonical_document_bytes = V2E.canonical_document_bytes
canonical_json_bytes = V2E.canonical_json_bytes
parse_canonical_json = V2E.parse_canonical_json


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _call(module: Any, name: str, *arguments: Any, **keywords: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"V3 pure metrics API is absent: {name}")
    try:
        return function(*arguments, **keywords)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(f"V3 pure metrics rejected {name}: {exc}") from exc


def _hex64(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegenerationError(f"{label} is not lowercase SHA-256")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or path.is_symlink():
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
        raise RegenerationError(f"{label} changed while reading")
    return b"".join(blocks)


def _load_canonical_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular(path, label)
    value = parse_canonical_json(raw, label=label)
    if not isinstance(value, dict):
        raise RegenerationError(f"{label} must contain a JSON object")
    return raw, value


def _binding(path: Path, label: str) -> dict[str, Any]:
    raw = _read_regular(path, label)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _manifest_sha256(value: Mapping[str, Any]) -> str:
    projection = dict(value)
    projection.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_document_bytes(projection)).hexdigest()


def _root_inventory(path: Path | str, label: str) -> dict[str, Any]:
    value = V2E._root_inventory(path, label)
    value["manifest_sha256"] = _manifest_sha256(value)
    return value


def _assert_inventory_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], label: str
) -> None:
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise RegenerationError(f"{label} changed during the read-only audit")


def _assert_no_shared_inodes(*inventories: Mapping[str, Any]) -> int:
    owners: dict[tuple[int, int], str] = {}
    shared = 0
    for inventory in inventories:
        root = str(inventory["path"])
        for row in inventory["files"]:
            identity = (int(row["device"]), int(row["inode"]))
            prior = owners.setdefault(identity, f"{root}/{row['path']}")
            if prior != f"{root}/{row['path']}":
                shared += 1
    if shared:
        raise RegenerationError("historical roots share file inodes")
    return shared


def _npz_snapshot_payload(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    members = V2E._read_npz_members(path, label, expected_names=None)
    member = members.get("snapshot_payload_bytes.npy")
    if not isinstance(member, Mapping):
        raise RegenerationError(f"{label} lacks snapshot_payload_bytes.npy")
    if member.get("dtype") != "|u1" or not isinstance(member.get("shape"), list):
        raise RegenerationError(f"{label} snapshot payload dtype/shape drift")
    shape = member["shape"]
    payload = member.get("payload")
    if (
        len(shape) != 1
        or not isinstance(shape[0], int)
        or shape[0] < 1
        or not isinstance(payload, bytes)
        or len(payload) != shape[0]
    ):
        raise RegenerationError(f"{label} snapshot payload extent drift")
    return payload, dict(member)


# ---------------------------------------------------------------------------
# Restricted historical protocol-4 worker
# ---------------------------------------------------------------------------


def _preflight_outer_pickle(payload: bytes, authority: Mapping[str, Any]) -> None:
    allowed = authority.get("allowed_opcodes")
    protocol = authority.get("outer_pickle_protocol")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise RegenerationError("historical deserializer opcode authority drift")
    allowed_set = set(allowed)
    expected_globals = [
        ["scripts.run_go2_oracle_branch_pilot_v1", "BranchSnapshot"],
        ["numpy._core.multiarray", "_reconstruct"],
        ["numpy", "ndarray"],
        ["numpy", "dtype"],
        ["lewm_genesis.lewm_contract", "EpisodeState"],
        ["torch._utils", "_rebuild_tensor_v2"],
        ["torch.storage", "_load_from_bytes"],
        ["collections", "OrderedDict"],
    ]
    if authority.get("allowed_globals") != expected_globals:
        raise RegenerationError("historical deserializer global authority drift")
    observed_protocol: int | None = None
    try:
        operations = list(pickletools.genops(payload))
    except Exception as exc:
        raise RegenerationError("historical snapshot outer pickle is malformed") from exc
    if (
        not operations
        or operations[-1][0].name != "STOP"
        or operations[-1][2] + 1 != len(payload)
    ):
        raise RegenerationError("historical snapshot outer pickle lacks terminal STOP")
    for opcode, argument, _position in operations:
        if opcode.name not in allowed_set:
            raise RegenerationError(
                f"historical snapshot uses forbidden outer opcode: {opcode.name}"
            )
        if opcode.name == "PROTO":
            if observed_protocol is not None:
                raise RegenerationError("historical snapshot has repeated PROTO")
            observed_protocol = int(argument)
    if observed_protocol != protocol:
        raise RegenerationError("historical snapshot outer pickle protocol drift")


def _worker_stub(name: str, module: str, fields: Sequence[str]) -> type:
    declared = tuple(fields)
    # Deliberately no methods: BUILD handling lives in the restricted
    # unpickler, not on an artifact-selected class.
    cls = type(name, (), {"__slots__": declared})
    cls.__module__ = module
    cls.__qualname__ = name
    return cls


def _historical_worker(payload_path: Path) -> int:
    """Restricted worker entry; called only by the portable parent process."""

    # These imports exist only in the isolated historical worker.
    import collections
    import numpy as np
    import torch

    semantics = importlib.import_module(SEMANTICS_MODULE)
    branch_fields = (
        "solver_state", "step_index", "last_actions", "harness", "rng",
        "counters", "goal", "identity", "boundary", "digest",
    )
    episode_fields = (
        "scene_id", "episode_id", "reset_count", "episode_step",
        "scene_family", "split", "manifest_sha256",
    )
    branch = _worker_stub(
        "BranchSnapshot", "scripts.run_go2_oracle_branch_pilot_v1", branch_fields
    )
    episode = _worker_stub(
        "EpisodeState", "lewm_genesis.lewm_contract", episode_fields
    )
    allowed: dict[tuple[str, str], Any] = {
        ("scripts.run_go2_oracle_branch_pilot_v1", "BranchSnapshot"): branch,
        ("lewm_genesis.lewm_contract", "EpisodeState"): episode,
        ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch.storage", "_load_from_bytes"): torch.storage._load_from_bytes,
        ("collections", "OrderedDict"): collections.OrderedDict,
    }

    structured_stubs = {
        branch: branch_fields,
        episode: episode_fields,
    }

    class RestrictedUnpickler(pickle._Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return allowed[(module, name)]
            except KeyError as exc:
                raise pickle.UnpicklingError(
                    f"forbidden historical global: {module}.{name}"
                ) from exc

        def persistent_load(self, persistent_id: Any) -> Any:
            raise pickle.UnpicklingError(
                f"forbidden historical persistent ID: {type(persistent_id).__name__}"
            )

        def load_build(self) -> None:
            state = self.stack.pop()
            instance = self.stack[-1]
            declared = structured_stubs.get(type(instance))
            if declared is None:
                self.stack.append(state)
                return pickle._Unpickler.load_build(self)
            if isinstance(state, tuple) and len(state) == 2 and state[0] is None:
                state = state[1]
            if not isinstance(state, dict) or set(state) != set(declared):
                raise pickle.UnpicklingError(
                    f"{type(instance).__name__} state-field inventory drift"
                )
            for field in declared:
                object.__setattr__(instance, field, state[field])

    RestrictedUnpickler.dispatch = pickle._Unpickler.dispatch.copy()
    RestrictedUnpickler.dispatch[pickle.BUILD[0]] = RestrictedUnpickler.load_build

    with np.load(payload_path, allow_pickle=False) as archive:
        if set(archive.files) == set():
            raise pickle.UnpicklingError("historical NPZ has no members")
        artifact = archive["snapshot_payload_bytes"]
        if artifact.dtype.str != "|u1" or artifact.ndim != 1:
            raise pickle.UnpicklingError("historical serialized snapshot member drift")
        artifact_bytes = artifact.tobytes(order="C")
    stream = io.BytesIO(artifact_bytes)
    value = RestrictedUnpickler(stream).load()
    if stream.read(1):
        raise pickle.UnpicklingError("trailing outer pickle bytes")
    canonical = semantics.canonical_semantic_snapshot(value)
    evidence = semantics.semantic_snapshot_evidence(value)
    if evidence.get("snapshot_semantic_digest_v1") != hashlib.sha256(canonical).hexdigest():
        raise pickle.UnpicklingError("serializer evidence digest drift")
    evidence_raw = canonical_document_bytes(evidence)
    output = (
        WORKER_MAGIC
        + len(evidence_raw).to_bytes(8, "big")
        + evidence_raw
        + len(canonical).to_bytes(8, "big")
        + canonical
    )
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    return 0


def _run_historical_worker(
    payload_path: Path,
    *,
    deserializer_authority: Mapping[str, Any],
    timeout_seconds: int = 120,
) -> tuple[bytes, dict[str, Any]]:
    artifact, _member = _npz_snapshot_payload(payload_path, str(payload_path))
    _preflight_outer_pickle(artifact, deserializer_authority)
    expected_interpreter = deserializer_authority.get("worker_interpreter")
    if expected_interpreter != str(HISTORICAL_WORKER_INTERPRETER):
        raise RegenerationError("historical worker interpreter authority drift")
    if not HISTORICAL_WORKER_INTERPRETER.is_file():
        raise RegenerationError("historical worker interpreter is absent")
    environment = {
        "HOME": os.environ.get("HOME", "/home/andrewknowles"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            str(HISTORICAL_WORKER_INTERPRETER),
            str(Path(__file__).resolve()),
            "--historical-worker",
            str(payload_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RegenerationError(f"historical semantic worker failed: {message}")
    raw = completed.stdout
    if not raw.startswith(WORKER_MAGIC):
        raise RegenerationError("historical semantic worker framing drift")
    cursor = len(WORKER_MAGIC)
    if len(raw) < cursor + 8:
        raise RegenerationError("historical semantic worker output is truncated")
    evidence_size = int.from_bytes(raw[cursor : cursor + 8], "big")
    cursor += 8
    evidence_end = cursor + evidence_size
    if evidence_end + 8 > len(raw):
        raise RegenerationError("historical semantic evidence is truncated")
    evidence_raw = raw[cursor:evidence_end]
    cursor = evidence_end
    canonical_size = int.from_bytes(raw[cursor : cursor + 8], "big")
    cursor += 8
    if cursor + canonical_size != len(raw):
        raise RegenerationError("historical semantic payload extent drift")
    canonical = raw[cursor:]
    evidence = parse_canonical_json(evidence_raw, label="historical semantic evidence")
    if not isinstance(evidence, dict):
        raise RegenerationError("historical semantic evidence is not an object")
    return canonical, evidence


# ---------------------------------------------------------------------------
# Portable independent semantic-byte parser
# ---------------------------------------------------------------------------


class _SemanticParser:
    """Parse the frozen framed language without importing NumPy or torch."""

    _SCALAR_TAGS = {b"none", b"bool", b"int", b"float64", b"utf8", b"bytes", b"numpy-scalar"}
    _NUMPY_ITEM_SIZES = {
        "|u1": 1, "|i1": 1, "<u2": 2, ">u2": 2, "<i2": 2, ">i2": 2,
        "<u4": 4, ">u4": 4, "<i4": 4, ">i4": 4, "<u8": 8, ">u8": 8,
        "<i8": 8, ">i8": 8, "<f2": 2, ">f2": 2, "<f4": 4, ">f4": 4,
        "<f8": 8, ">f8": 8, "<c8": 8, ">c8": 8, "<c16": 16, ">c16": 16,
        "|b1": 1,
    }
    _TORCH_ITEM_SIZES = {
        "torch.uint8": 1,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.bool": 1,
        "torch.complex64": 8,
        "torch.complex128": 16,
    }

    def __init__(self, payload: bytes, authority: Mapping[str, Any]) -> None:
        if not payload.startswith(SEMANTIC_MAGIC):
            raise RegenerationError("semantic payload magic drift")
        self.payload = payload
        self.authority = authority
        self.defined_ids: list[int] = []
        self.reference_ids: list[int] = []
        self.active_ids: set[int] = set()
        self.reference_manifest: list[dict[str, Any]] = []
        self.reference_edge_manifest: list[dict[str, Any]] = []
        self.storage_ids: set[int] = set()
        self.numpy_rows: list[dict[str, Any]] = []
        self.torch_rows: list[dict[str, Any]] = []
        self.structured_rows: list[dict[str, Any]] = []
        self.nonfinite_rows: list[dict[str, Any]] = []
        self.type_counts: Counter[str] = Counter()
        self.current_object_id: int | None = None

    @staticmethod
    def _u64(payload: bytes, offset: int, label: str) -> tuple[int, int]:
        if offset + 8 > len(payload):
            raise RegenerationError(f"truncated semantic uint64: {label}")
        return int.from_bytes(payload[offset : offset + 8], "big"), offset + 8

    @staticmethod
    def _frame(payload: bytes, offset: int, label: str) -> tuple[bytes, bytes, int, bytes]:
        if offset >= len(payload):
            raise RegenerationError(f"truncated semantic frame: {label}")
        tag_size = payload[offset]
        offset += 1
        if tag_size < 1 or offset + tag_size + 8 > len(payload):
            raise RegenerationError(f"invalid semantic frame header: {label}")
        tag = payload[offset : offset + tag_size]
        offset += tag_size
        size = int.from_bytes(payload[offset : offset + 8], "big")
        offset += 8
        end = offset + size
        if end > len(payload):
            raise RegenerationError(f"truncated semantic frame payload: {label}")
        raw_frame = payload[offset - 9 - tag_size : end]
        return tag, payload[offset:end], end, raw_frame

    @classmethod
    def _single_frame(cls, payload: bytes, label: str) -> tuple[bytes, bytes, bytes]:
        tag, body, end, raw = cls._frame(payload, 0, label)
        if end != len(payload):
            raise RegenerationError(f"semantic frame has trailing bytes: {label}")
        return tag, body, raw

    @staticmethod
    def _strict_utf8(payload: bytes, label: str) -> str:
        try:
            value = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RegenerationError(f"invalid semantic UTF-8: {label}") from exc
        if value.encode("utf-8") != payload:
            raise RegenerationError(f"noncanonical semantic UTF-8: {label}")
        return value

    @staticmethod
    def _shape(payload: bytes, label: str) -> list[int]:
        count, cursor = _SemanticParser._u64(payload, 0, label)
        if cursor + count * 8 != len(payload):
            raise RegenerationError(f"semantic shape extent drift: {label}")
        return [
            int.from_bytes(payload[cursor + 8 * index : cursor + 8 * (index + 1)], "big")
            for index in range(count)
        ]

    @staticmethod
    def _integer(payload: bytes, label: str) -> int:
        if not payload:
            raise RegenerationError(f"empty semantic integer: {label}")
        sign = payload[0]
        magnitude_bytes = payload[1:]
        if sign == 0:
            if magnitude_bytes:
                raise RegenerationError(f"noncanonical semantic zero: {label}")
            return 0
        if sign not in (1, 2) or not magnitude_bytes or magnitude_bytes[0] == 0:
            raise RegenerationError(f"noncanonical semantic integer: {label}")
        magnitude = int.from_bytes(magnitude_bytes, "big")
        return magnitude if sign == 1 else -magnitude

    @classmethod
    def _integer_vector(cls, payload: bytes, label: str) -> list[int]:
        count, cursor = cls._u64(payload, 0, label)
        rows: list[int] = []
        for index in range(count):
            tag, body, cursor, _raw = cls._frame(payload, cursor, f"{label}/{index}")
            if tag != b"int":
                raise RegenerationError(f"semantic integer vector tag drift: {label}")
            rows.append(cls._integer(body, f"{label}/{index}"))
        if cursor != len(payload):
            raise RegenerationError(f"semantic integer vector trailing bytes: {label}")
        return rows

    @staticmethod
    def _product(shape: Sequence[int]) -> int:
        result = 1
        for extent in shape:
            result *= extent
        return result

    def _scalar(self, tag: bytes, body: bytes, path: str, *, mapping_key: bool = False) -> Any:
        if tag == b"none":
            if body:
                raise RegenerationError(f"semantic none payload drift: {path}")
            self.type_counts["builtins.NoneType"] += 1
            return None
        if tag == b"bool":
            if body not in (b"\x00", b"\x01"):
                raise RegenerationError(f"semantic bool payload drift: {path}")
            self.type_counts["builtins.bool"] += 1
            return body == b"\x01"
        if tag == b"int":
            self.type_counts["builtins.int"] += 1
            return self._integer(body, path)
        if tag == b"float64":
            if len(body) != 8 or not math.isfinite(struct.unpack(">d", body)[0]):
                raise RegenerationError(f"semantic float64 payload drift: {path}")
            self.type_counts["builtins.float"] += 1
            return struct.unpack(">d", body)[0]
        if tag == b"utf8":
            self.type_counts["builtins.str"] += 1
            return self._strict_utf8(body, path)
        if tag == b"bytes":
            self.type_counts["builtins.bytes"] += 1
            return body
        if tag == b"numpy-scalar" and not mapping_key:
            first, dtype_body, cursor, _ = self._frame(body, 0, f"{path}/dtype")
            second, data, cursor, _ = self._frame(body, cursor, f"{path}/data")
            if first != b"dtype" or second != b"data" or cursor != len(body):
                raise RegenerationError(f"semantic NumPy scalar framing drift: {path}")
            dtype = self._strict_utf8(dtype_body, f"{path}/dtype")
            size = self._NUMPY_ITEM_SIZES.get(dtype)
            if size is None or len(data) != size:
                raise RegenerationError(f"semantic NumPy scalar dtype/size drift: {path}")
            numpy_name = {
                "|u1": "numpy.uint8", "|i1": "numpy.int8", "|b1": "numpy.bool_",
                "<u2": "numpy.uint16", ">u2": "numpy.uint16",
                "<i2": "numpy.int16", ">i2": "numpy.int16",
                "<u4": "numpy.uint32", ">u4": "numpy.uint32",
                "<i4": "numpy.int32", ">i4": "numpy.int32",
                "<u8": "numpy.uint64", ">u8": "numpy.uint64",
                "<i8": "numpy.int64", ">i8": "numpy.int64",
                "<f2": "numpy.float16", ">f2": "numpy.float16",
                "<f4": "numpy.float32", ">f4": "numpy.float32",
                "<f8": "numpy.float64", ">f8": "numpy.float64",
                "<c8": "numpy.complex64", ">c8": "numpy.complex64",
                "<c16": "numpy.complex128", ">c16": "numpy.complex128",
            }.get(dtype)
            if numpy_name is None:
                raise RegenerationError(f"semantic NumPy scalar type drift: {path}")
            self.type_counts[numpy_name] += 1
            return (dtype, data)
        raise RegenerationError(f"unsupported semantic scalar tag {tag!r}: {path}")

    def _sequence_frames(self, body: bytes, cursor: int, count: int, path: str) -> int:
        for index in range(count):
            cursor = self._value(body, cursor, f"{path}/{index}")
        return cursor

    def _nested_fields(self, body: bytes, expected: Sequence[bytes], path: str) -> dict[bytes, bytes]:
        cursor = 0
        result: dict[bytes, bytes] = {}
        for expected_tag in expected:
            tag, value, cursor, _raw = self._frame(body, cursor, f"{path}/{expected_tag.decode()}")
            if tag != expected_tag or tag in result:
                raise RegenerationError(f"semantic nested field order drift: {path}")
            result[tag] = value
        if cursor != len(body):
            raise RegenerationError(f"semantic nested field trailing bytes: {path}")
        return result

    def _body(self, tag: bytes, body: bytes, path: str) -> None:
        if tag in (b"list", b"tuple"):
            count, cursor = self._u64(body, 0, path)
            cursor = self._sequence_frames(body, cursor, count, path)
            if cursor != len(body):
                raise RegenerationError(f"semantic sequence trailing bytes: {path}")
            return
        if tag in (b"set", b"frozenset"):
            count, cursor = self._u64(body, 0, path)
            prior: bytes | None = None
            for index in range(count):
                scalar_tag, scalar_body, cursor, raw = self._frame(body, cursor, f"{path}/{index}")
                if scalar_tag not in self._SCALAR_TAGS - {b"numpy-scalar"}:
                    raise RegenerationError(f"semantic set member tag drift: {path}")
                self._scalar(scalar_tag, scalar_body, f"{path}/{index}", mapping_key=True)
                if prior is not None and raw <= prior:
                    raise RegenerationError(f"semantic set ordering/uniqueness drift: {path}")
                prior = raw
            if cursor != len(body):
                raise RegenerationError(f"semantic set trailing bytes: {path}")
            return
        if tag == b"mapping":
            mapping_tag, mapping_body, cursor, _ = self._frame(body, 0, f"{path}/mapping-type")
            if mapping_tag != b"mapping-type":
                raise RegenerationError(f"semantic mapping-type field drift: {path}")
            mapping_type = self._strict_utf8(mapping_body, f"{path}/mapping-type")
            if mapping_type not in ("builtins.dict", "collections.OrderedDict"):
                raise RegenerationError(f"semantic mapping type drift: {path}")
            count, cursor = self._u64(body, cursor, f"{path}/count")
            prior: bytes | None = None
            for index in range(count):
                entry_tag, entry_body, cursor, _ = self._frame(body, cursor, f"{path}/entry-{index}")
                if entry_tag != b"entry":
                    raise RegenerationError(f"semantic mapping entry tag drift: {path}")
                key_tag, key_body, key_end, key_raw = self._frame(entry_body, 0, f"{path}/key-{index}")
                if key_tag not in self._SCALAR_TAGS - {b"numpy-scalar"}:
                    raise RegenerationError(f"semantic mapping key tag drift: {path}")
                key = self._scalar(key_tag, key_body, f"{path}/key-{index}", mapping_key=True)
                if prior is not None and key_raw <= prior:
                    raise RegenerationError(f"semantic mapping key ordering drift: {path}")
                prior = key_raw
                child = (
                    f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}"
                    if isinstance(key, str)
                    else f"{path}/@{hashlib.sha256(key_raw).hexdigest()}"
                )
                value_end = self._value(entry_body, key_end, child)
                if value_end != len(entry_body):
                    raise RegenerationError(f"semantic mapping entry trailing bytes: {path}")
            if cursor != len(body):
                raise RegenerationError(f"semantic mapping trailing bytes: {path}")
            return
        if tag == b"numpy":
            fields = self._nested_fields(
                body,
                (b"storage-id", b"storage-offset-bytes", b"dtype", b"shape", b"stride-bytes", b"logical-c-bytes"),
                path,
            )
            storage, end = self._u64(fields[b"storage-id"], 0, f"{path}/storage-id")
            offset, offset_end = self._u64(fields[b"storage-offset-bytes"], 0, f"{path}/storage-offset")
            if end != len(fields[b"storage-id"]) or offset_end != len(fields[b"storage-offset-bytes"]):
                raise RegenerationError(f"semantic NumPy scalar field extent drift: {path}")
            dtype = self._strict_utf8(fields[b"dtype"], f"{path}/dtype")
            shape = self._shape(fields[b"shape"], f"{path}/shape")
            strides = self._integer_vector(fields[b"stride-bytes"], f"{path}/strides")
            size = self._NUMPY_ITEM_SIZES.get(dtype)
            logical = fields[b"logical-c-bytes"]
            if size is None or len(shape) != len(strides) or len(logical) != self._product(shape) * size:
                raise RegenerationError(f"semantic NumPy dtype/shape/stride extent drift: {path}")
            self.storage_ids.add(storage)
            if self.current_object_id is None:
                raise RegenerationError(f"semantic NumPy object context absent: {path}")
            row = {"path": path, "object_id": self.current_object_id, "storage_id": storage, "storage_offset_bytes": offset, "dtype_str": dtype, "shape": shape, "strides_bytes": strides, "logical_sha256": hashlib.sha256(logical).hexdigest()}
            self._validate_nonfinite_numpy(row, logical)
            self.numpy_rows.append(row)
            return
        if tag == b"torch":
            fields = self._nested_fields(
                body,
                (b"storage-id", b"storage-offset-bytes", b"dtype", b"shape", b"logical-stride", b"device-rule", b"device-class", b"logical-cpu-c-bytes"),
                path,
            )
            storage, end = self._u64(fields[b"storage-id"], 0, f"{path}/storage-id")
            offset, offset_end = self._u64(fields[b"storage-offset-bytes"], 0, f"{path}/storage-offset")
            if end != len(fields[b"storage-id"]) or offset_end != len(fields[b"storage-offset-bytes"]):
                raise RegenerationError(f"semantic torch scalar field extent drift: {path}")
            dtype = self._strict_utf8(fields[b"dtype"], f"{path}/dtype")
            shape = self._shape(fields[b"shape"], f"{path}/shape")
            strides = self._integer_vector(fields[b"logical-stride"], f"{path}/strides")
            device_rule = self._strict_utf8(fields[b"device-rule"], f"{path}/device-rule")
            device_class = self._strict_utf8(fields[b"device-class"], f"{path}/device-class")
            size = self._TORCH_ITEM_SIZES.get(dtype)
            logical = fields[b"logical-cpu-c-bytes"]
            if size is None or len(shape) != len(strides) or len(logical) != self._product(shape) * size:
                raise RegenerationError(f"semantic torch dtype/shape/stride extent drift: {path}")
            expected_rule = "CPU_OR_ACCELERATOR_CLASS_WITH_PATH_ASSOCIATION_V1"
            if device_rule != expected_rule or device_class not in ("cpu", "accelerator"):
                raise RegenerationError(f"semantic torch device authority drift: {path}")
            self.storage_ids.add(storage)
            if size and offset % size:
                raise RegenerationError(f"semantic torch storage byte offset alignment drift: {path}")
            if self.current_object_id is None:
                raise RegenerationError(f"semantic torch object context absent: {path}")
            self.torch_rows.append({"path": path, "object_id": self.current_object_id, "storage_id": storage, "storage_offset_bytes": offset, "dtype": dtype, "shape": shape, "logical_stride": strides, "semantic_device_class": device_class, "logical_sha256": hashlib.sha256(logical).hexdigest()})
            return
        if tag == b"structured":
            type_tag, type_body, cursor, _ = self._frame(body, 0, f"{path}/type")
            if type_tag != b"type":
                raise RegenerationError(f"semantic structured type field drift: {path}")
            type_name = self._strict_utf8(type_body, f"{path}/type")
            count, cursor = self._u64(body, cursor, f"{path}/field-count")
            authority = self.authority.get("structured_type_fields_in_order")
            expected = authority.get(type_name) if isinstance(authority, Mapping) else None
            if not isinstance(expected, list) or count != len(expected):
                raise RegenerationError(f"semantic structured type authority drift: {path}")
            observed: list[str] = []
            for index in range(count):
                field_tag, field_body, cursor, _ = self._frame(body, cursor, f"{path}/field-{index}")
                if field_tag != b"field":
                    raise RegenerationError(f"semantic structured field tag drift: {path}")
                name_tag, name_body, name_end, _ = self._frame(field_body, 0, f"{path}/field-name")
                if name_tag != b"name":
                    raise RegenerationError(f"semantic structured name tag drift: {path}")
                name = self._strict_utf8(name_body, f"{path}/field-name")
                observed.append(name)
                child_path = f"{path}/{name.replace('~', '~0').replace('/', '~1')}"
                value_end = self._value(field_body, name_end, child_path)
                if value_end != len(field_body):
                    raise RegenerationError(f"semantic structured field trailing bytes: {path}")
            if cursor != len(body) or observed != expected:
                raise RegenerationError(f"semantic structured field order drift: {path}")
            self.structured_rows.append({"path": path, "type": type_name, "declared_fields": observed})
            return
        raise RegenerationError(f"unsupported semantic body tag {tag!r}: {path}")

    def _validate_nonfinite_numpy(self, row: Mapping[str, Any], logical: bytes) -> None:
        dtype = str(row["dtype_str"])
        if dtype not in ("<f4", ">f4", "<f8", ">f8"):
            return
        endian = "<" if dtype[0] == "<" else ">"
        code = "f" if dtype.endswith("f4") else "d"
        size = 4 if code == "f" else 8
        values = struct.iter_unpack(endian + code, logical)
        positive = negative = nan = 0
        for (value,) in values:
            if math.isnan(value):
                nan += 1
            elif value == math.inf:
                positive += 1
            elif value == -math.inf:
                negative += 1
        if not (positive or negative or nan):
            return
        sentinel = self.authority.get("nonfinite_sentinel_authority")
        rows = sentinel.get("allowed_numpy_arrays") if isinstance(sentinel, Mapping) else None
        matches = [item for item in rows or [] if item.get("path") == row["path"]]
        observed = {
            "path": row["path"],
            "dtype_str": dtype,
            "shape": row["shape"],
            "strides_bytes": row["strides_bytes"],
            "positive_infinity_count": positive,
            "negative_infinity_count": negative,
            "nan_count": nan,
        }
        if len(matches) != 1 or observed != matches[0]:
            raise RegenerationError(f"semantic nonfinite authority drift: {row['path']}")
        self.nonfinite_rows.append(observed)

    def _body_type(self, tag: bytes, body: bytes, path: str) -> str:
        if tag == b"list":
            return "builtins.list"
        if tag == b"tuple":
            return "builtins.tuple"
        if tag == b"set":
            return "builtins.set"
        if tag == b"frozenset":
            return "builtins.frozenset"
        if tag == b"numpy":
            return "numpy.ndarray"
        if tag == b"torch":
            return "torch.Tensor"
        if tag == b"mapping":
            nested, value, _end, _raw = self._frame(body, 0, f"{path}/mapping-type")
            if nested != b"mapping-type":
                raise RegenerationError(f"semantic mapping type frame drift: {path}")
            return self._strict_utf8(value, f"{path}/mapping-type")
        if tag == b"structured":
            nested, value, _end, _raw = self._frame(body, 0, f"{path}/type")
            if nested != b"type":
                raise RegenerationError(f"semantic structured type frame drift: {path}")
            return self._strict_utf8(value, f"{path}/type")
        raise RegenerationError(f"unknown semantic reference type tag {tag!r}: {path}")

    def _value(self, payload: bytes, offset: int, path: str) -> int:
        tag, body, end, _raw = self._frame(payload, offset, path)
        if tag in self._SCALAR_TAGS:
            self._scalar(tag, body, path)
            return end
        if tag == b"REF":
            reference, cursor = self._u64(body, 0, path)
            if cursor != len(body) or reference not in self.defined_ids:
                raise RegenerationError(f"semantic REF target drift: {path}")
            self.reference_ids.append(reference)
            self.reference_edge_manifest.append(
                {"path": path, "reference_id": reference, "cycle": reference in self.active_ids}
            )
            return end
        if tag == b"DEF":
            object_id, cursor = self._u64(body, 0, path)
            if object_id != len(self.defined_ids):
                raise RegenerationError(f"semantic DEF order drift: {path}")
            self.defined_ids.append(object_id)
            body_tag, body_payload, body_end, _ = self._frame(body, cursor, path)
            if body_end != len(body):
                raise RegenerationError(f"semantic DEF trailing bytes: {path}")
            type_name = self._body_type(body_tag, body_payload, path)
            self.type_counts[type_name] += 1
            self.reference_manifest.append(
                {"object_id": object_id, "path": path, "type": type_name}
            )
            prior_object = self.current_object_id
            self.current_object_id = object_id
            self.active_ids.add(object_id)
            try:
                self._body(body_tag, body_payload, path)
            finally:
                self.active_ids.remove(object_id)
                self.current_object_id = prior_object
            return end
        raise RegenerationError(f"unsupported semantic value tag {tag!r}: {path}")

    def inspect(self) -> dict[str, Any]:
        end = self._value(self.payload, len(SEMANTIC_MAGIC), "$root")
        if end != len(self.payload):
            raise RegenerationError("semantic payload has trailing bytes")
        storage_ids = sorted(self.storage_ids)
        if storage_ids != list(range(len(storage_ids))):
            raise RegenerationError("semantic storage ID range drift")
        storage_manifest = []
        for storage_id in storage_ids:
            members = [
                row for row in [*self.numpy_rows, *self.torch_rows]
                if row["storage_id"] == storage_id
            ]
            members.sort(key=lambda row: row["object_id"])
            storage_manifest.append(
                {
                    "storage_id": storage_id,
                    "kind": "numpy" if members and members[0] in self.numpy_rows else "torch",
                    "owner_object_id": members[0]["object_id"],
                    "member_object_ids": [row["object_id"] for row in members],
                    "member_paths": [row["path"] for row in members],
                }
            )
        structured_inventory: dict[str, list[str]] = {}
        for row in self.structured_rows:
            prior = structured_inventory.setdefault(row["type"], row["declared_fields"])
            if prior != row["declared_fields"]:
                raise RegenerationError("semantic structured inventory is inconsistent")
        return {
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "bytes": len(self.payload),
            "referenceable_object_count": len(self.defined_ids),
            "reference_alias_edge_count": len(self.reference_ids),
            "reference_cycle_edge_count": sum(
                int(row["cycle"]) for row in self.reference_edge_manifest
            ),
            "reference_manifest": self.reference_manifest,
            "reference_edge_manifest": self.reference_edge_manifest,
            "storage_group_count": len(storage_ids),
            "storage_manifest": storage_manifest,
            "numpy_array_count": len(self.numpy_rows),
            "torch_tensor_count": len(self.torch_rows),
            "structured_object_count": len(self.structured_rows),
            "numpy_rows": self.numpy_rows,
            "torch_rows": self.torch_rows,
            "structured_rows": self.structured_rows,
            "structured_type_inventory": [
                {"type": name, "declared_fields": fields}
                for name, fields in sorted(structured_inventory.items())
            ],
            "tensor_device_manifest": [
                {
                    "path": row["path"],
                    "object_id": row["object_id"],
                    "semantic_device_class": row["semantic_device_class"],
                }
                for row in self.torch_rows
            ],
            "nonfinite_sentinel_inventory": sorted(
                self.nonfinite_rows, key=lambda row: row["path"]
            ),
            "type_inventory": [
                {"type": name, "count": int(count)}
                for name, count in sorted(self.type_counts.items())
            ],
        }


def inspect_semantic_snapshot(
    payload: bytes, serializer_authority: Mapping[str, Any]
) -> dict[str, Any]:
    return _SemanticParser(payload, serializer_authority).inspect()


def _validate_semantic_worker_result(
    canonical: bytes,
    evidence: Mapping[str, Any],
    *,
    metrics: Any,
    serializer_authority: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _call(metrics, "validate_snapshot_semantic_evidence", evidence)
    _call(metrics, "validate_snapshot_semantic_bytes", canonical, validated)
    inspected = inspect_semantic_snapshot(canonical, serializer_authority)
    direct_fields = (
        "referenceable_object_count",
        "reference_alias_edge_count",
        "reference_cycle_edge_count",
        "reference_manifest",
        "reference_edge_manifest",
        "storage_manifest",
        "structured_type_inventory",
        "nonfinite_sentinel_inventory",
        "type_inventory",
    )
    for field in direct_fields:
        if inspected[field] != validated[field]:
            raise RegenerationError(
                f"portable semantic parser/evidence mismatch: {field}"
            )
    allowed_sentinels = serializer_authority.get("nonfinite_sentinel_authority", {}).get(
        "allowed_numpy_arrays"
    )
    if validated["nonfinite_sentinel_inventory"] != allowed_sentinels:
        raise RegenerationError("historical semantic sentinel coverage is incomplete")
    production = serializer_authority.get("production_pool_000_observed")
    if not isinstance(production, Mapping):
        raise RegenerationError("historical semantic production authority is absent")
    type_counts = {
        item["type"]: item["count"] for item in validated["type_inventory"]
    }
    if (
        inspected["numpy_array_count"] != production.get("numpy_array_count")
        or inspected["torch_tensor_count"] != production.get("torch_tensor_count")
        or type_counts.get("builtins.set") != production.get("set_count")
        or inspected["reference_alias_edge_count"]
        != production.get("reference_alias_count")
        or sum(
            int(len(item["member_object_ids"]) > 1)
            for item in inspected["storage_manifest"]
        )
        != production.get("shared_storage_alias_count")
    ):
        raise RegenerationError("historical semantic production inventory drift")
    device_projection = []
    for index, item in enumerate(validated["tensor_device_manifest"]):
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "object_id",
            "semantic_device_class",
            "source_device_type",
            "source_device_index",
        }:
            raise RegenerationError(
                f"semantic tensor-device manifest field drift: {index}"
            )
        device_class = item["semantic_device_class"]
        source_type = item["source_device_type"]
        source_index = item["source_device_index"]
        if (
            device_class not in ("cpu", "accelerator")
            or not isinstance(source_type, str)
            or not source_type
            or (device_class == "cpu") != (source_type == "cpu")
            or (
                source_index is not None
                and (
                    not isinstance(source_index, int)
                    or isinstance(source_index, bool)
                    or source_index < 0
                )
            )
        ):
            raise RegenerationError(
                f"semantic tensor-device evidence value drift: {index}"
            )
        device_projection.append(
            {
                "path": item["path"],
                "object_id": item["object_id"],
                "semantic_device_class": device_class,
            }
        )
    if inspected["tensor_device_manifest"] != device_projection:
        raise RegenerationError("portable semantic tensor-device projection drift")
    if (
        inspected["sha256"] != validated["snapshot_semantic_digest_v1"]
        or inspected["bytes"] != validated["canonical_semantic_byte_count"]
    ):
        raise RegenerationError("portable semantic byte identity drift")
    return {
        "snapshot_semantic_digest_v1": inspected["sha256"],
        "semantic_payload_bytes": inspected["bytes"],
        "semantic_payload_sha256": inspected["sha256"],
        "semantic_evidence": validated,
    }


def reconstruct_historical_snapshot_semantics(
    payload_path: Path | str, *, metrics: Any | None = None
) -> dict[str, Any]:
    """Return independently verified semantic identity for one historical shard.

    The input is opened read-only.  No simulator or experiment runner is
    imported or called.
    """

    module = _load_metrics_module() if metrics is None else metrics
    serializer_authority = _call(module, "semantic_snapshot_serializer_authority")
    deserializer_authority = _call(
        module, "historical_snapshot_deserializer_authority"
    )
    path = Path(payload_path)
    artifact, _member = _npz_snapshot_payload(path, str(path))
    canonical, evidence = _run_historical_worker(
        path, deserializer_authority=deserializer_authority
    )
    semantic = _validate_semantic_worker_result(
        canonical,
        evidence,
        metrics=module,
        serializer_authority=serializer_authority,
    )
    return {
        "artifact_file_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_bytes": len(artifact),
        **semantic,
    }


# ---------------------------------------------------------------------------
# Complete immutable V1/V2 custody receipt
# ---------------------------------------------------------------------------


def _file_row(inventory: Mapping[str, Any], relative: str) -> dict[str, Any]:
    matches = [row for row in inventory["files"] if row["path"] == relative]
    if len(matches) != 1:
        raise RegenerationError(f"root inventory lacks exactly one file: {relative}")
    return dict(matches[0])


def _historical_version(
    *,
    version: str,
    pool_index: int,
    material_root: Path,
    inventory: Mapping[str, Any],
    metrics: Any,
    serializer_authority: Mapping[str, Any],
    deserializer_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    prefix = f"qualification/pool-{pool_index:03d}"
    metadata_relative = f"{prefix}/metadata.json"
    payload_relative = f"{prefix}/payload.npz"
    metadata_binding = _file_row(inventory, metadata_relative)
    payload_binding = _file_row(inventory, payload_relative)
    _raw, metadata = _load_canonical_object(
        material_root / metadata_relative,
        f"{version} pool-{pool_index:03d} metadata",
    )
    expected_experiment = V1_EXPERIMENT_ID if version == "V1" else V2_EXPERIMENT_ID
    if (
        metadata.get("experiment_id") != expected_experiment
        or metadata.get("pool_index") != pool_index
        or metadata.get("reset_or_candidate_outcome_opened") is not False
    ):
        raise RegenerationError(
            f"{version} pool-{pool_index:03d} historical identity/boundary drift"
        )
    supplied_payload = metadata.get("payload")
    if not isinstance(supplied_payload, Mapping) or (
        supplied_payload.get("path") != payload_relative
        or supplied_payload.get("bytes") != payload_binding["bytes"]
        or supplied_payload.get("sha256") != payload_binding["sha256"]
    ):
        raise RegenerationError(
            f"{version} pool-{pool_index:03d} payload binding drift"
        )
    expected_comment = (
        None if version == "V1" else "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2:FRESH"
    )
    members = V2E._read_npz_members(
        material_root / payload_relative,
        f"{version} pool-{pool_index:03d} payload",
        expected_names=None,
        expected_comment=expected_comment,
    )
    artifact_member = members.get("snapshot_payload_bytes.npy")
    if not isinstance(artifact_member, Mapping):
        raise RegenerationError(
            f"{version} pool-{pool_index:03d} snapshot artifact absent"
        )
    artifact = artifact_member["payload"]
    if not isinstance(artifact, bytes):
        raise RegenerationError(
            f"{version} pool-{pool_index:03d} snapshot artifact encoding drift"
        )
    artifact_sha = hashlib.sha256(artifact).hexdigest()
    snapshot = metadata.get("snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("snapshot_payload_sha256") != artifact_sha
        or metadata.get("initial_decision_state_sha256") != artifact_sha
    ):
        raise RegenerationError(
            f"{version} pool-{pool_index:03d} artifact binding drift"
        )
    canonical, evidence = _run_historical_worker(
        material_root / payload_relative,
        deserializer_authority=deserializer_authority,
    )
    semantic = _validate_semantic_worker_result(
        canonical,
        evidence,
        metrics=metrics,
        serializer_authority=serializer_authority,
    )
    version_row = {
        "metadata_binding": metadata_binding,
        "payload_binding": payload_binding,
        "artifact_file_sha256": artifact_sha,
        "artifact_bytes": len(artifact),
        **semantic,
    }
    return version_row, metadata, members


def _historical_pair(
    pool_index: int,
    *,
    v1_material_root: Path,
    v1_inventory: Mapping[str, Any],
    v2_material_root: Path,
    v2_inventory: Mapping[str, Any],
    metrics: Any,
    serializer_authority: Mapping[str, Any],
    deserializer_authority: Mapping[str, Any],
) -> dict[str, Any]:
    v1, v1_metadata, v1_members = _historical_version(
        version="V1",
        pool_index=pool_index,
        material_root=v1_material_root,
        inventory=v1_inventory,
        metrics=metrics,
        serializer_authority=serializer_authority,
        deserializer_authority=deserializer_authority,
    )
    v2, v2_metadata, v2_members = _historical_version(
        version="V2",
        pool_index=pool_index,
        material_root=v2_material_root,
        inventory=v2_inventory,
        metrics=metrics,
        serializer_authority=serializer_authority,
        deserializer_authority=deserializer_authority,
    )
    identity_fields = (
        "candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id"
    )
    v1_spec = v1_metadata.get("candidate_spec")
    v2_spec = v2_metadata.get("candidate_spec")
    if not isinstance(v1_spec, Mapping) or not isinstance(v2_spec, Mapping):
        raise RegenerationError(
            f"historical pool-{pool_index:03d} candidate-spec evidence absent"
        )
    identity = {field: v1_spec.get(field) for field in identity_fields}
    if any(
        not isinstance(value, str) or not value or v2_spec.get(field) != value
        for field, value in identity.items()
    ):
        raise RegenerationError(
            f"historical pool-{pool_index:03d} scientific identity drift"
        )
    v1_names = set(v1_members)
    v2_names = set(v2_members)
    inventory_equal = v1_names == v2_names
    dtype_equal = inventory_equal and all(
        v1_members[name]["dtype"] == v2_members[name]["dtype"] for name in v1_names
    )
    shape_equal_except_artifact = inventory_equal and all(
        name == "snapshot_payload_bytes.npy"
        or v1_members[name]["shape"] == v2_members[name]["shape"]
        for name in v1_names
    )
    non_snapshot_equal = inventory_equal and all(
        name == "snapshot_payload_bytes.npy"
        or (
            v1_members[name]["dtype"] == v2_members[name]["dtype"]
            and v1_members[name]["shape"] == v2_members[name]["shape"]
            and v1_members[name]["payload"] == v2_members[name]["payload"]
        )
        for name in v1_names
    )
    semantic_equal = (
        v1["snapshot_semantic_digest_v1"]
        == v2["snapshot_semantic_digest_v1"]
    )
    semantic_evidence_equal = (
        canonical_json_bytes(v1["semantic_evidence"])
        == canonical_json_bytes(v2["semantic_evidence"])
    )
    qualified_equal = v1_metadata.get("qualified") == v2_metadata.get("qualified")
    rejection_equal = (
        v1_metadata.get("rejection_reason") == v2_metadata.get("rejection_reason")
    )
    contact_equal = (
        v1_members["teacher__physics_contact.npy"]["payload"]
        == v2_members["teacher__physics_contact.npy"]["payload"]
    )
    stuck_equal = V2E._teacher_stuck_from_members(
        v1_members, f"V1 pool-{pool_index:03d}"
    ) == V2E._teacher_stuck_from_members(
        v2_members, f"V2 pool-{pool_index:03d}"
    )
    if not all(
        (
            inventory_equal,
            dtype_equal,
            shape_equal_except_artifact,
            non_snapshot_equal,
            semantic_equal,
            semantic_evidence_equal,
            qualified_equal,
            rejection_equal,
            contact_equal,
            stuck_equal,
        )
    ):
        raise RegenerationError(
            f"historical pool-{pool_index:03d} substantive V1/V2 drift"
        )
    return {
        "pool_index": pool_index,
        **identity,
        "v1": v1,
        "v2": v2,
        "payload_member_inventory_equal": inventory_equal,
        "payload_member_dtypes_equal": dtype_equal,
        "non_snapshot_member_shapes_equal": shape_equal_except_artifact,
        "non_snapshot_members_equal": non_snapshot_equal,
        "artifact_file_sha256_equal": (
            v1["artifact_file_sha256"] == v2["artifact_file_sha256"]
        ),
        "snapshot_semantic_digest_v1_equal": semantic_equal,
        "semantic_evidence_equal": semantic_evidence_equal,
        "qualified_equal": qualified_equal,
        "rejection_reason_equal": rejection_equal,
        "contact_sequence_equal": contact_equal,
        "stuck_equal": stuck_equal,
        "pass": True,
    }


def _validate_v2_terminal(
    official_root: Path,
    official_inventory: Mapping[str, Any],
    material_inventory: Mapping[str, Any],
    *,
    metrics: Any,
) -> dict[str, Any]:
    expected_leaves = {
        "contract.json",
        "scientific_invariance_receipt.json",
        "v1_custody_and_nonreuse.json",
        "v1_v2_first_eight_reproduction.json",
    }
    if (
        official_inventory["file_count"] != 4
        or {row["path"] for row in official_inventory["files"]} != expected_leaves
        or official_inventory["regular_file_apparent_bytes"] != 102_527
        or material_inventory["file_count"] != 34
        or material_inventory["regular_file_apparent_bytes"] != 4_382_459
    ):
        raise RegenerationError("V2 terminal root inventory drift")
    custody_authority = _call(metrics, "v1_v2_custody_and_nonreuse_authority")
    expected_binding = custody_authority.get("v2_terminal_root_binding")
    if not isinstance(expected_binding, Mapping):
        raise RegenerationError("V2 terminal binding authority is absent")
    expected_rows = expected_binding.get("files")
    if not isinstance(expected_rows, list):
        raise RegenerationError("V2 terminal leaf authority is absent")
    actual_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in official_inventory["files"]
    ]
    if actual_rows != expected_rows:
        raise RegenerationError("V2 terminal exact leaf binding drift")
    _raw, first_eight = _load_canonical_object(
        official_root / "v1_v2_first_eight_reproduction.json",
        "V2 terminal first-eight receipt",
    )
    if (
        first_eight.get("status") != "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
        or first_eight.get("technical_disposition")
        != "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
        or first_eight.get("full_collection_authorized") is not False
        or first_eight.get("pass") is not False
        or first_eight.get("row_count") != 8
        or first_eight.get("candidate_ranker_development_heldout_outcomes_opened")
        != 0
    ):
        raise RegenerationError("V2 terminal disposition drift")
    external_v2 = official_root.parent / (
        "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
    )
    if external_v2.exists() or external_v2.is_symlink():
        raise RegenerationError("prohibited V2 regeneration receipt appeared")
    return {
        "disposition": "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH",
        "official_leaf_count": 4,
        "material_pair_count": 8,
        "full_collection_authorized": False,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
        "external_regeneration_receipt_present": False,
        "first_eight_receipt_binding": _binding(
            official_root / "v1_v2_first_eight_reproduction.json",
            "V2 terminal first-eight receipt",
        ),
    }


def _validate_root_inventory_document(
    value: Any, *, expected_path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROOT_INVENTORY_FIELDS:
        raise RegenerationError(f"{label} root inventory field drift")
    row = dict(value)
    if row["path"] != str(expected_path):
        raise RegenerationError(f"{label} root path drift")
    files = row["files"]
    directories = row["directories"]
    if not isinstance(files, list) or not isinstance(directories, list):
        raise RegenerationError(f"{label} root inventory rows are not lists")
    if (
        row["file_count"] != len(files)
        or row["directory_count"] != len(directories)
        or [item.get("path") for item in files] != sorted(item.get("path") for item in files)
        or [item.get("path") for item in directories]
        != sorted(item.get("path") for item in directories)
    ):
        raise RegenerationError(f"{label} root inventory count/order drift")
    file_identities: set[tuple[int, int]] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != _ROOT_FILE_FIELDS:
            raise RegenerationError(f"{label} file[{index}] field drift")
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or item["path"].startswith("/")
            or ".." in Path(item["path"]).parts
            or not isinstance(item["bytes"], int)
            or isinstance(item["bytes"], bool)
            or item["bytes"] < 0
            or not isinstance(item["allocated_bytes"], int)
            or isinstance(item["allocated_bytes"], bool)
            or item["allocated_bytes"] < 0
            or item["nlink"] != 1
        ):
            raise RegenerationError(f"{label} file[{index}] value drift")
        _hex64(item["sha256"], f"{label} file[{index}] sha256")
        identity = (item["device"], item["inode"])
        if identity in file_identities:
            raise RegenerationError(f"{label} file inode is repeated")
        file_identities.add(identity)
    for index, item in enumerate(directories):
        if not isinstance(item, Mapping) or set(item) != _ROOT_DIRECTORY_FIELDS:
            raise RegenerationError(f"{label} directory[{index}] field drift")
        if item["nlink"] < 1 or item["allocated_bytes"] < 0:
            raise RegenerationError(f"{label} directory[{index}] value drift")
    file_apparent = sum(item["bytes"] for item in files)
    file_allocated = sum(item["allocated_bytes"] for item in files)
    directory_allocated = sum(item["allocated_bytes"] for item in directories)
    if (
        row["regular_file_apparent_bytes"] != file_apparent
        or row["regular_file_allocated_bytes"] != file_allocated
        or row["directory_allocated_bytes"] != directory_allocated
        or row["allocated_bytes"] != file_allocated + directory_allocated
        or row["manifest_sha256"] != _manifest_sha256(row)
    ):
        raise RegenerationError(f"{label} root aggregate/manifest drift")
    return row


def _validate_historical_version_document(
    value: Any, *, label: str, metrics: Any
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _HISTORICAL_VERSION_FIELDS:
        raise RegenerationError(f"{label} historical version field drift")
    row = dict(value)
    for binding_name in ("metadata_binding", "payload_binding"):
        binding = row[binding_name]
        if not isinstance(binding, Mapping) or set(binding) != _ROOT_FILE_FIELDS:
            raise RegenerationError(f"{label} {binding_name} field drift")
        _hex64(binding["sha256"], f"{label} {binding_name} sha256")
    artifact_sha = _hex64(row["artifact_file_sha256"], f"{label} artifact SHA")
    semantic_sha = _hex64(
        row["snapshot_semantic_digest_v1"], f"{label} semantic SHA"
    )
    if (
        not isinstance(row["artifact_bytes"], int)
        or isinstance(row["artifact_bytes"], bool)
        or row["artifact_bytes"] <= 0
        or not isinstance(row["semantic_payload_bytes"], int)
        or isinstance(row["semantic_payload_bytes"], bool)
        or row["semantic_payload_bytes"] <= len(SEMANTIC_MAGIC)
        or row["semantic_payload_sha256"] != semantic_sha
        or artifact_sha == semantic_sha
    ):
        raise RegenerationError(f"{label} historical version identity drift")
    evidence = _call(
        metrics, "validate_snapshot_semantic_evidence", row["semantic_evidence"]
    )
    if (
        evidence["snapshot_semantic_digest_v1"] != semantic_sha
        or evidence["canonical_semantic_byte_count"]
        != row["semantic_payload_bytes"]
    ):
        raise RegenerationError(f"{label} semantic evidence cross-link drift")
    return row


def validate_historical_custody_receipt_document(
    value: Any, *, metrics: Any | None = None
) -> dict[str, Any]:
    """Strictly validate the ordinary combined receipt without opening roots."""

    module = _load_metrics_module() if metrics is None else metrics
    row = _call(module, "validate_external_v1_v2_custody_receipt", value)
    V2E.V1._validate_no_self_digest(row, "historical custody receipt")
    roots = {
        "v1_official_root": DEFAULT_V1_OFFICIAL_ROOT,
        "v1_material_root": DEFAULT_V1_MATERIAL_ROOT,
        "v2_official_root": DEFAULT_V2_OFFICIAL_ROOT,
        "v2_material_root": DEFAULT_V2_MATERIAL_ROOT,
    }
    validated_roots = {
        name: _validate_root_inventory_document(
            row[name], expected_path=path, label=name
        )
        for name, path in roots.items()
    }
    all_file_inodes: set[tuple[int, int]] = set()
    for name, inventory in validated_roots.items():
        for item in inventory["files"]:
            identity = (item["device"], item["inode"])
            if identity in all_file_inodes:
                raise RegenerationError(f"historical roots share an inode: {name}")
            all_file_inodes.add(identity)
    terminal = row["v2_terminal_evidence"]
    if not isinstance(terminal, Mapping) or set(terminal) != _V2_TERMINAL_FIELDS:
        raise RegenerationError("V2 terminal evidence field drift")
    if terminal != {
        "disposition": "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH",
        "official_leaf_count": 4,
        "material_pair_count": 8,
        "full_collection_authorized": False,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
        "external_regeneration_receipt_present": False,
        "first_eight_receipt_binding": terminal["first_eight_receipt_binding"],
    }:
        raise RegenerationError("V2 terminal evidence value drift")
    first_eight_binding = terminal["first_eight_receipt_binding"]
    if (
        not isinstance(first_eight_binding, Mapping)
        or set(first_eight_binding) != {"path", "bytes", "sha256"}
        or first_eight_binding["path"]
        != str(DEFAULT_V2_OFFICIAL_ROOT / "v1_v2_first_eight_reproduction.json")
        or first_eight_binding["bytes"] != 6797
        or first_eight_binding["sha256"]
        != "d3437d2560af6005b871b94bed6ef3f7f0f185497984725eae41de80224069c6"
    ):
        raise RegenerationError("V2 first-eight terminal binding drift")
    pairs = row["first_eight_pairs"]
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise RegenerationError("historical first-eight pair count drift")
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping) or set(item) != _HISTORICAL_PAIR_FIELDS:
            raise RegenerationError(f"historical pair[{index}] field drift")
        if item["pool_index"] != index:
            raise RegenerationError(f"historical pair[{index}] order drift")
        v1 = _validate_historical_version_document(
            item["v1"], label=f"pair[{index}].v1", metrics=module
        )
        v2 = _validate_historical_version_document(
            item["v2"], label=f"pair[{index}].v2", metrics=module
        )
        expected = {
            "artifact_file_sha256_equal": (
                v1["artifact_file_sha256"] == v2["artifact_file_sha256"]
            ),
            "snapshot_semantic_digest_v1_equal": (
                v1["snapshot_semantic_digest_v1"]
                == v2["snapshot_semantic_digest_v1"]
            ),
            "semantic_evidence_equal": (
                canonical_json_bytes(v1["semantic_evidence"])
                == canonical_json_bytes(v2["semantic_evidence"])
            ),
        }
        if any(item[key] is not expected_value for key, expected_value in expected.items()):
            raise RegenerationError(f"historical pair[{index}] digest projection drift")
        required_true = _HISTORICAL_PAIR_FIELDS - {
            "pool_index", "candidate_spec_id", "state_id", "scene_id",
            "episode_id", "graph_id", "v1", "v2",
            "artifact_file_sha256_equal",
        }
        if (
            item["artifact_file_sha256_equal"] is not False
            or any(item[field] is not True for field in required_true)
        ):
            raise RegenerationError(f"historical pair[{index}] substantive gate failed")
    expected_scientific = {
        "v1_teacher_qualification_rows_opened": 8,
        "v1_teacher_qualified": 7,
        "v1_teacher_rejected": 1,
        "v2_fresh_teacher_qualification_rows_opened": 8,
        "v2_teacher_qualified": 7,
        "v2_teacher_rejected": 1,
        "v2_pool_002_physics_contact_reproduced": True,
        "selection_performed": False,
        "reset_fixture_executions": 0,
        "candidate_fanout_executions": 0,
        "encoder_initializations": 0,
        "ranker_inference_calls": 0,
        "heldout_outcomes_opened": 0,
        "metrics_persisted": False,
        "result_persisted": False,
    }
    expected_repository = {
        "v1_source_freeze_commit": V2E.SOURCE_PARENT_COMMIT,
        "v2_source_freeze_commit": SOURCE_PARENT_COMMIT,
        "v1_freeze_subject": "Freeze physical graph edge handoff qualification",
        "v2_freeze_subject": "Freeze corrected physical graph edge handoff qualification V2",
    }
    expected_immutability = {
        "audit_mode": "read_only_same_inode_before_after",
        "v1_official_root_unchanged_during_audit": True,
        "v1_material_root_unchanged_during_audit": True,
        "v2_official_root_unchanged_during_audit": True,
        "v2_material_root_unchanged_during_audit": True,
        "all_files_regular_single_link": True,
        "receipt_outside_all_experiment_roots": True,
    }
    expected_nonreuse = {
        "historical_roots_shared_inode_count": 0,
        "v1_v2_artifact_file_sha256_equal_count": 0,
        "v1_v2_snapshot_semantic_digest_v1_equal_count": 8,
        "raw_artifact_inequality_is_descriptive": True,
        "historical_payload_copy_into_v3_count": 0,
        "historical_hardlink_into_v3_count": 0,
        "historical_runtime_artifact_or_shard_reused": False,
        "historical_deserializer_invocations": 16,
        "model_initializations": 0,
        "training_runs": 0,
        "simulator_initializations": 0,
        "runner_calls": 0,
        "encoder_calls": 0,
        "ranker_calls": 0,
    }
    if (
        row["scientific_boundary"] != expected_scientific
        or row["repository"] != expected_repository
        or row["immutability"] != expected_immutability
        or row["nonreuse"] != expected_nonreuse
    ):
        raise RegenerationError("historical custody boundary/nonreuse projection drift")
    return dict(row)


def build_historical_custody_receipt(
    *,
    v1_official_root: Path | str = DEFAULT_V1_OFFICIAL_ROOT,
    v1_material_root: Path | str = DEFAULT_V1_MATERIAL_ROOT,
    v1_custody_receipt: Path | str = DEFAULT_V1_CUSTODY_RECEIPT,
    v2_official_root: Path | str = DEFAULT_V2_OFFICIAL_ROOT,
    v2_material_root: Path | str = DEFAULT_V2_MATERIAL_ROOT,
    require_v3_roots_absent: bool = False,
) -> dict[str, Any]:
    """Rebuild the ordinary, complete V1/V2 historical custody receipt."""

    metrics = _load_metrics_module()
    serializer_authority = _call(metrics, "semantic_snapshot_serializer_authority")
    deserializer_authority = _call(metrics, "historical_snapshot_deserializer_authority")
    v1_official_path = V2E._root_path(v1_official_root, "V1 official root")
    v1_material_path = V2E._root_path(v1_material_root, "V1 material root")
    v2_official_path = V2E._root_path(v2_official_root, "V2 official root")
    v2_material_path = V2E._root_path(v2_material_root, "V2 material root")
    if require_v3_roots_absent:
        for path in (DEFAULT_V3_OUTPUT_ROOT, DEFAULT_V3_MATERIAL_ROOT):
            if path.exists() or path.is_symlink():
                raise RegenerationError(
                    "V3 root exists before historical custody receipt generation"
                )
    v1_receipt = V2E.validate_existing_v1_custody_receipt(
        v1_custody_receipt,
        official_root=v1_official_path,
        material_root=v1_material_path,
    )
    if not isinstance(v1_receipt, Mapping):
        raise RegenerationError("V1 custody receipt validation did not return a mapping")
    v1_binding = _binding(Path(v1_custody_receipt), "V1 external custody receipt")
    v1_official_before = _root_inventory(v1_official_path, "V1 official root")
    v1_material_before = _root_inventory(v1_material_path, "V1 material root")
    v2_official_before = _root_inventory(v2_official_path, "V2 official root")
    v2_material_before = _root_inventory(v2_material_path, "V2 material root")
    shared = _assert_no_shared_inodes(
        v1_official_before,
        v1_material_before,
        v2_official_before,
        v2_material_before,
    )
    terminal = _validate_v2_terminal(
        v2_official_path,
        v2_official_before,
        v2_material_before,
        metrics=metrics,
    )
    pairs = [
        _historical_pair(
            index,
            v1_material_root=v1_material_path,
            v1_inventory=v1_material_before,
            v2_material_root=v2_material_path,
            v2_inventory=v2_material_before,
            metrics=metrics,
            serializer_authority=serializer_authority,
            deserializer_authority=deserializer_authority,
        )
        for index in range(8)
    ]
    if not all(
        row["snapshot_semantic_digest_v1_equal"]
        and row["semantic_evidence_equal"]
        and row["pass"]
        for row in pairs
    ):
        raise RegenerationError("historical semantic equivalence failed")
    v1_official_after = _root_inventory(v1_official_path, "V1 official root")
    v1_material_after = _root_inventory(v1_material_path, "V1 material root")
    v2_official_after = _root_inventory(v2_official_path, "V2 official root")
    v2_material_after = _root_inventory(v2_material_path, "V2 material root")
    for before, after, label in (
        (v1_official_before, v1_official_after, "V1 official root"),
        (v1_material_before, v1_material_after, "V1 material root"),
        (v2_official_before, v2_official_after, "V2 official root"),
        (v2_material_before, v2_material_after, "V2 material root"),
    ):
        _assert_inventory_equal(before, after, label)
    result = {
        "schema": HISTORICAL_CUSTODY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "generated_before_v3_simulator_creation": True,
        "v1_external_custody_receipt_binding": v1_binding,
        "v1_official_root": v1_official_before,
        "v1_material_root": v1_material_before,
        "v2_official_root": v2_official_before,
        "v2_material_root": v2_material_before,
        "v2_terminal_evidence": terminal,
        "first_eight_pairs": pairs,
        "scientific_boundary": {
            "v1_teacher_qualification_rows_opened": 8,
            "v1_teacher_qualified": 7,
            "v1_teacher_rejected": 1,
            "v2_fresh_teacher_qualification_rows_opened": 8,
            "v2_teacher_qualified": 7,
            "v2_teacher_rejected": 1,
            "v2_pool_002_physics_contact_reproduced": True,
            "selection_performed": False,
            "reset_fixture_executions": 0,
            "candidate_fanout_executions": 0,
            "encoder_initializations": 0,
            "ranker_inference_calls": 0,
            "heldout_outcomes_opened": 0,
            "metrics_persisted": False,
            "result_persisted": False,
        },
        "repository": {
            "v1_source_freeze_commit": V2E.SOURCE_PARENT_COMMIT,
            "v2_source_freeze_commit": SOURCE_PARENT_COMMIT,
            "v1_freeze_subject": "Freeze physical graph edge handoff qualification",
            "v2_freeze_subject": "Freeze corrected physical graph edge handoff qualification V2",
        },
        "immutability": {
            "audit_mode": "read_only_same_inode_before_after",
            "v1_official_root_unchanged_during_audit": True,
            "v1_material_root_unchanged_during_audit": True,
            "v2_official_root_unchanged_during_audit": True,
            "v2_material_root_unchanged_during_audit": True,
            "all_files_regular_single_link": True,
            "receipt_outside_all_experiment_roots": True,
        },
        "nonreuse": {
            "historical_roots_shared_inode_count": shared,
            "v1_v2_artifact_file_sha256_equal_count": sum(
                int(row["artifact_file_sha256_equal"]) for row in pairs
            ),
            "v1_v2_snapshot_semantic_digest_v1_equal_count": sum(
                int(row["snapshot_semantic_digest_v1_equal"]) for row in pairs
            ),
            "raw_artifact_inequality_is_descriptive": True,
            "historical_payload_copy_into_v3_count": 0,
            "historical_hardlink_into_v3_count": 0,
            "historical_runtime_artifact_or_shard_reused": False,
            "historical_deserializer_invocations": 16,
            "model_initializations": 0,
            "training_runs": 0,
            "simulator_initializations": 0,
            "runner_calls": 0,
            "encoder_calls": 0,
            "ranker_calls": 0,
        },
    }
    return validate_historical_custody_receipt_document(result, metrics=metrics)


def emit_historical_custody_receipt(
    output: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
    *,
    receipt: Mapping[str, Any] | None = None,
    require_v3_roots_absent: bool = True,
) -> bytes:
    value = (
        build_historical_custody_receipt(
            require_v3_roots_absent=require_v3_roots_absent
        )
        if receipt is None
        else dict(receipt)
    )
    V2E.V1._validate_no_self_digest(value, "historical custody receipt")
    roots = (
        DEFAULT_V1_OFFICIAL_ROOT,
        DEFAULT_V1_MATERIAL_ROOT,
        DEFAULT_V2_OFFICIAL_ROOT,
        DEFAULT_V2_MATERIAL_ROOT,
        DEFAULT_V3_OUTPUT_ROOT,
        DEFAULT_V3_MATERIAL_ROOT,
    )
    return V2E._emit_external(roots, output, value)


def validate_existing_historical_custody_receipt(
    output: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    raw, supplied = _load_canonical_object(
        Path(output), "combined V1/V2 historical custody receipt"
    )
    V2E.V1._validate_no_self_digest(supplied, "historical custody receipt")
    rebuilt = build_historical_custody_receipt(require_v3_roots_absent=False)
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError(
            "combined V1/V2 historical custody receipt differs from exact rebuild"
        )
    binding = _binding(Path(output), "combined V1/V2 historical custody receipt")
    metrics = _load_metrics_module()
    _call(
        metrics,
        "validate_external_v1_v2_custody_receipt",
        supplied,
        expected_binding=binding,
    )
    return validate_historical_custody_receipt_document(supplied, metrics=metrics)


# ---------------------------------------------------------------------------
# Independent behavioural-probe NPZ reduction
# ---------------------------------------------------------------------------


def _load_npz_arrays(
    path: Path,
    authority: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen exact NPY payloads and return arrays plus a compact inspection."""

    import numpy as np

    expected_names = {f"{name}.npy" for name in authority}
    members = V2E._read_npz_members(
        path, label, expected_names=expected_names, expected_comment=None
    )
    arrays: dict[str, Any] = {}
    inspection_rows: list[dict[str, Any]] = []
    for member in authority:
        item = members[f"{member}.npy"]
        expected = authority[member]
        if item["dtype"] != expected["descr"] or item["shape"] != expected["shape"]:
            raise RegenerationError(f"{label}/{member} dtype/shape drift")
        dtype = np.dtype(item["dtype"])
        if dtype.str != item["dtype"]:
            raise RegenerationError(f"{label}/{member} dtype is noncanonical")
        array = np.frombuffer(item["payload"], dtype=dtype).reshape(item["shape"])
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise RegenerationError(f"{label}/{member} is nonfinite")
        arrays[member] = array
        inspection_rows.append(
            {
                "member": member,
                "dtype_str": dtype.str,
                "shape": list(array.shape),
                "array_bytes_sha256": hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return arrays, {
        "binding": _binding(path, label),
        "member_count": len(inspection_rows),
        "members": inspection_rows,
    }


def _yaw_xyzw(values: Sequence[float]) -> float:
    x, y, z, w = [float(value) for value in values]
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _derive_probe_stuck(trace: Mapping[str, Any], authority: Mapping[str, Any]) -> bool:
    import numpy as np

    requested = np.asarray(trace["requested_command"])
    poses = np.asarray(trace["base_pose_world"])
    stuck = authority["stuck"]
    return bool(
        float(np.max(np.abs(requested)))
        > float(stuck["command_activity_threshold"])
        and float(np.linalg.norm(poses[-1, :2] - poses[0, :2]))
        < float(stuck["h3_translation_threshold_m"])
        and abs(_wrapped(_yaw_xyzw(poses[-1, 3:]) - _yaw_xyzw(poses[0, 3:])))
        < float(stuck["h3_heading_threshold_rad"])
    )


def _probe_trace(
    arrays: Mapping[str, Any],
    trace_index: int,
    *,
    behavioural_authority: Mapping[str, Any],
    physical_outcome_authority: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(arrays["trace_offsets"][trace_index])
    stop = int(arrays["trace_offsets"][trace_index + 1])
    if stop - start != int(behavioural_authority["physics_samples_per_trial"]):
        raise RegenerationError("behavioural trace slice length drift")
    result = {
        member: arrays[member][start:stop]
        for member in behavioural_authority["npz_members"]
        if member
        not in {
            "trace_offsets", "version_code", "pool_index", "trial_index",
            "stuck", "termination_code", "final_snapshot_semantic_digest_bytes",
        }
    }
    if int(arrays["termination_code"][trace_index]) != 1:
        raise RegenerationError("behavioural trace termination-code drift")
    result["termination_reason"] = "H3_COMPLETE"
    result["stuck"] = _derive_probe_stuck(result, physical_outcome_authority)
    if bool(int(arrays["stuck"][trace_index])) is not result["stuck"]:
        raise RegenerationError("behavioural trace stuck projection drift")
    digest_bytes = arrays["final_snapshot_semantic_digest_bytes"][trace_index]
    if list(digest_bytes.shape) != [32]:
        raise RegenerationError("behavioural final semantic digest extent drift")
    result["final_snapshot_semantic_digest_v1"] = bytes(
        digest_bytes.tolist()
    ).hex()
    _hex64(
        result["final_snapshot_semantic_digest_v1"],
        "behavioural final semantic digest",
    )
    return result


def _compare_probe_traces(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    behavioural_authority: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    pair = behavioural_authority["pair_comparison"]
    dt = float(behavioural_authority["physics_dt_s"])
    expected_requested = np.broadcast_to(
        np.asarray(
            behavioural_authority["requested_command_trace_value"],
            dtype=np.float64,
        ),
        (int(behavioural_authority["physics_samples_per_trial"]), 3),
    )
    for label, trace in (("left", left), ("right", right)):
        timestamps = np.asarray(trace["timestamp_s"])
        requested = np.asarray(trace["requested_command"])
        if (
            timestamps.dtype.str != "<f8"
            or list(timestamps.shape)
            != [int(behavioural_authority["physics_samples_per_trial"])]
            or not bool(np.isfinite(timestamps).all())
            or not bool(
                np.allclose(
                    np.diff(timestamps), dt, rtol=0.0, atol=1.0e-12
                )
            )
        ):
            raise RegenerationError(f"{label} behavioural timestamp cadence drift")
        if (
            requested.dtype.str != "<f8"
            or list(requested.shape) != list(expected_requested.shape)
            or not bool(np.array_equal(requested, expected_requested))
        ):
            raise RegenerationError(
                f"{label} behavioural requested-command trace drift"
            )
    exact = {
        member: bool(np.array_equal(left[member], right[member]))
        for member in pair["exact_members"]
    }
    maxima = {
        "base_pose_world_position_xyz": float(
            np.max(np.abs(left["base_pose_world"][:, :3] - right["base_pose_world"][:, :3]))
        ),
        "base_pose_world_quaternion_xyzw": float(
            np.max(np.abs(left["base_pose_world"][:, 3:] - right["base_pose_world"][:, 3:]))
        ),
        "base_twist_world": float(
            np.max(np.abs(left["base_twist_world"] - right["base_twist_world"]))
        ),
        "joint_position": float(
            np.max(np.abs(left["joint_position"] - right["joint_position"]))
        ),
        "joint_velocity": float(
            np.max(np.abs(left["joint_velocity"] - right["joint_velocity"]))
        ),
        "controller_observation": float(
            np.max(
                np.abs(
                    left["controller_observation"]
                    - right["controller_observation"]
                )
            )
        ),
        "policy_output": float(
            np.max(np.abs(left["policy_output"] - right["policy_output"]))
        ),
    }
    samplewise = {
        name: value
        <= float(
            pair["samplewise_tolerances"].get(
                name,
                behavioural_authority["controller_policy_samplewise_tolerance"],
            )
        )
        for name, value in maxima.items()
    }
    endpoint_position = float(
        np.linalg.norm(left["base_pose_world"][-1, :3] - right["base_pose_world"][-1, :3])
    )
    endpoint_heading = abs(
        _wrapped(
            _yaw_xyzw(left["base_pose_world"][-1, 3:])
            - _yaw_xyzw(right["base_pose_world"][-1, 3:])
        )
    )
    result = {
        "exact_member_equal": exact,
        "samplewise_max_abs_error": maxima,
        "samplewise_pass": samplewise,
        "endpoint_position_error_m": endpoint_position,
        "endpoint_heading_error_rad": endpoint_heading,
        "termination_reason_equal": (
            left["termination_reason"] == right["termination_reason"]
        ),
        "stuck_equal": left["stuck"] == right["stuck"],
        "final_snapshot_semantic_digest_equal": (
            left["final_snapshot_semantic_digest_v1"]
            == right["final_snapshot_semantic_digest_v1"]
        ),
    }
    result["pass"] = bool(
        all(exact.values())
        and all(samplewise.values())
        and endpoint_position <= float(pair["endpoint_position_tolerance_m"])
        and endpoint_heading <= float(pair["endpoint_heading_tolerance_rad"])
        and result["termination_reason_equal"]
        and result["stuck_equal"]
    )
    return result


def inspect_behavioural_probe_npz(
    path: Path | str,
    *,
    metrics: Any | None = None,
    first_eight_only: bool = False,
) -> dict[str, Any]:
    """Independently reopen and reduce the raw behavioural probe cube."""

    module = _load_metrics_module() if metrics is None else metrics
    behavioural = _call(module, "behavioural_probe_authority")
    physical = _call(module, "physical_trace_reduction_authority")
    if not isinstance(physical, Mapping) or "stuck" not in physical:
        raise RegenerationError("physical stuck authority is absent")
    authority = (
        behavioural["first_eight_material_npz_members"]
        if first_eight_only
        else behavioural["npz_members"]
    )
    arrays, inspection = _load_npz_arrays(
        Path(path), authority, label="snapshot behavioural probes"
    )
    validator = (
        "validate_first_eight_behavioural_probe_npz_arrays"
        if first_eight_only
        else "validate_behavioural_probe_npz_arrays"
    )
    _call(module, validator, arrays)
    trace_count = 48 if first_eight_only else int(behavioural["trace_count"])
    traces = [
        _probe_trace(
            arrays,
            index,
            behavioural_authority=behavioural,
            physical_outcome_authority=physical,
        )
        for index in range(trace_count)
    ]
    within_version: list[dict[str, Any]] = []
    cross_version: list[dict[str, Any]] = []
    pool_stop = 8 if first_eight_only else 256
    for pool_index in range(pool_stop):
        versions = ("V1", "V2", "V3") if pool_index < 8 else ("V3",)
        indices_by_version: dict[str, list[int]] = {}
        for version in versions:
            if pool_index < 8:
                start = pool_index * 6 + ("V1", "V2", "V3").index(version) * 2
            else:
                start = 48 + (pool_index - 8) * 2
            indices_by_version[version] = [start, start + 1]
            independent = _compare_probe_traces(
                traces[start], traces[start + 1], behavioural_authority=behavioural
            )
            pure = _call(
                module,
                "compare_behavioural_probe_traces",
                traces[start],
                traces[start + 1],
            )
            if canonical_json_bytes(independent) != canonical_json_bytes(pure):
                raise RegenerationError(
                    "independent/pure within-version behavioural comparison drift"
                )
            within_version.append(
                {
                    "pool_index": pool_index,
                    "version": version,
                    "trace_indices": [start, start + 1],
                    "comparison": independent,
                }
            )
        if pool_index < 8:
            for left_version, right_version in (("V1", "V2"), ("V1", "V3"), ("V2", "V3")):
                comparisons = []
                for left_index in indices_by_version[left_version]:
                    for right_index in indices_by_version[right_version]:
                        independent = _compare_probe_traces(
                            traces[left_index],
                            traces[right_index],
                            behavioural_authority=behavioural,
                        )
                        pure = _call(
                            module,
                            "compare_behavioural_probe_traces",
                            traces[left_index],
                            traces[right_index],
                        )
                        if canonical_json_bytes(independent) != canonical_json_bytes(pure):
                            raise RegenerationError(
                                "independent/pure cross-version behavioural comparison drift"
                            )
                        comparisons.append(
                            {
                                "left_trace_index": left_index,
                                "right_trace_index": right_index,
                                "comparison": independent,
                            }
                        )
                cross_version.append(
                    {
                        "pool_index": pool_index,
                        "versions": [left_version, right_version],
                        "comparisons": comparisons,
                        "pass": all(item["comparison"]["pass"] for item in comparisons),
                    }
                )
    projection_sha = _call(
        module,
        "behavioural_probe_npz_projection_sha256",
        arrays,
        first_eight_only=first_eight_only,
    )
    return {
        **inspection,
        "first_eight_only": first_eight_only,
        "trace_count": trace_count,
        "physics_sample_count": int(arrays["trace_offsets"][-1]),
        "projection_sha256": projection_sha,
        "within_version_comparisons": within_version,
        "cross_version_comparisons": cross_version,
        "all_within_version_pass": all(
            row["comparison"]["pass"] for row in within_version
        ),
        "all_historical_cross_version_pass": all(
            row["pass"] for row in cross_version
        ),
    }


# ---------------------------------------------------------------------------
# Final-root custody and independent scientific reduction
# ---------------------------------------------------------------------------


_DOCUMENT_LEAVES = {
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
_LEDGER_LEAVES = {
    "candidate_fanout": "candidate_fanout.jsonl",
    "heldout_ranker_scores": "heldout_ranker_scores.jsonl",
    "repeated_execution": "repeated_execution.jsonl",
}
_PAYLOAD_LEAVES = (
    "state_snapshots.npz",
    "teacher_traces.npz",
    "rgb_observations.npz",
    "canonical_latents.npz",
    "candidate_traces.npz",
)
_SCIENTIFIC_NPZ_LEAVES = (
    *_PAYLOAD_LEAVES,
    "snapshot_behavioural_probes.npz",
)
_NEW_ORDINARY_LEAVES = (
    "v1_v2_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_v3_first_eight_reproduction.json",
)
_PUBLICATION_LEAVES = ("result.json", "result.md", "file_hashes.json")


def _validate_v3_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "reducer_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("V3 reducer authority is not an object")
    row = dict(authority)
    V2E.V1._validate_content_digest(row, "V3 reducer authority")
    if (
        row.get("schema")
        != "physical_graph_edge_handoff_qualification_v3.reducer_authority.v1"
        or row.get("experiment_id") != EXPERIMENT_ID
        or row.get("successful_output_leaf_count") != 28
        or row.get("successful_output_leaves") != list(V3_SUCCESS_FILES)
        or row.get("reproduction_mismatch_output_leaves")
        != list(V3_MISMATCH_FILES)
    ):
        raise RegenerationError("V3 reducer authority identity/inventory drift")
    required = {
        "evidence_keys", "documents", "ledgers", "npz_authorities",
        "runtime_paths", "external_artifact_roles",
        "physical_trace_reduction_authority",
        "runtime_environment_authority", "new_documents",
        "snapshot_semantic_serializer_authority",
        "historical_snapshot_deserializer_authority",
        "snapshot_behavioural_probe_authority",
        "qualification_shard_augmentation_authority",
        "v1_v2_custody_and_nonreuse_authority",
        "first_eight_reproduction_authority",
    }
    if not required.issubset(row):
        raise RegenerationError(
            "V3 reducer authority lacks sections: "
            f"{sorted(required - set(row))}"
        )
    if set(row["runtime_paths"].values()) != set(V3_SUCCESS_FILES):
        raise RegenerationError("V3 runtime path authority drift")
    if set(row["npz_authorities"]) != set(_PAYLOAD_LEAVES):
        raise RegenerationError("V3 inherited NPZ authority drift")
    if set(row["documents"]) != set(_DOCUMENT_LEAVES):
        raise RegenerationError("V3 document authority drift")
    if set(row["ledgers"]) != set(_LEDGER_LEAVES):
        raise RegenerationError("V3 ledger authority drift")
    if set(row["new_documents"]) != {
        "v1_v2_custody_and_nonreuse",
        "scientific_invariance_receipt",
        "v1_v2_v3_first_eight_reproduction",
        "snapshot_equivalence_index",
        "snapshot_behavioural_probes",
    }:
        raise RegenerationError("V3 new-evidence authority drift")
    return json.loads(json.dumps(row))


def _open_official_root(path: Path | str) -> tuple[Path, int]:
    return V2E.V1._open_root(path, "V3 official root")


def _validate_official_inventory(root_fd: int) -> tuple[list[str], bool, bool]:
    names = sorted(os.listdir(root_fd))
    observed = set(names)
    success_scientific = set(V3_SUCCESS_FILES) - set(_PUBLICATION_LEAVES)
    terminal = observed == set(V3_MISMATCH_FILES)
    complete = observed == set(V3_SUCCESS_FILES)
    if not terminal and observed not in (success_scientific, set(V3_SUCCESS_FILES)):
        raise RegenerationError(
            "V3 root is neither exact four-leaf terminal nor exact "
            "25/28-leaf successful inventory"
        )
    for name in names:
        V2E.CUSTODY._safe_leaf(name)
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegenerationError(
                f"V3 official leaf is not a single-link regular file: {name}"
            )
    return names, terminal, complete


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise RegenerationError(
            "V3 Git custody check failed: "
            + exc.output.decode("utf-8", "replace")
        ) from exc


def _repo_source_binding(relative: str, *, head: str) -> dict[str, Any]:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != relative
    ):
        raise RegenerationError(f"unsafe V3 source path: {relative!r}")
    path = REPO_ROOT / candidate
    info = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or path.resolve(strict=True) != path
    ):
        raise RegenerationError(f"V3 source is not a direct regular file: {relative}")
    raw = _read_regular(path, f"V3 source {relative}")
    if raw != _git_bytes("show", f"{head}:{relative}"):
        raise RegenerationError(f"live V3 source differs from freeze: {relative}")
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _observe_source_freeze(
    runtime_contract: Mapping[str, Any], module: Any
) -> dict[str, Any]:
    head = _git_bytes("rev-parse", "HEAD").decode().strip()
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("V3 reducer requires a clean source-freeze worktree")
    if runtime_contract.get("source_freeze_commit") != head:
        raise RegenerationError("V3 runtime contract differs from HEAD")
    scientific = runtime_contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V3 runtime contract lacks scientific contract")
    tracked = scientific.get("tracked_source_paths")
    closure = scientific.get("v3_wrapper_dependency_paths")
    if (
        not isinstance(tracked, list)
        or not isinstance(closure, list)
        or len(tracked) < 8
        or len(closure) == 0
    ):
        raise RegenerationError("V3 source-path authority drift")
    closure_paths = [*tracked[7:], *closure]
    if len(closure_paths) != len(set(closure_paths)):
        raise RegenerationError("V3 source closure contains duplicate paths")
    tracked_bindings = [_repo_source_binding(path, head=head) for path in tracked]
    closure_candidates = [path for path in tracked if "source_closure" in path]
    if len(closure_candidates) != 1:
        raise RegenerationError("V3 source-closure document identity drift")
    closure_raw, closure_document = _load_canonical_object(
        REPO_ROOT / closure_candidates[0], "V3 source closure"
    )
    expected_fields = {
        "schema", "parent_commit", "row_count", "rows", "content_digest"
    }
    if set(closure_document) != expected_fields:
        raise RegenerationError("V3 source-closure field drift")
    V2E.V1._validate_content_digest(closure_document, "V3 source closure")
    expected_rows = [_repo_source_binding(path, head=head) for path in closure_paths]
    if (
        closure_document.get("schema")
        != "physical_graph_edge_handoff_qualification_v3.source_closure.v1"
        or closure_document.get("parent_commit") != SOURCE_PARENT_COMMIT
        or closure_document.get("row_count") != len(expected_rows)
        or closure_document.get("rows") != expected_rows
    ):
        raise RegenerationError("V3 source closure differs from ordered live bytes")
    return {
        "head_commit": head,
        "parent_commit": _git_bytes("rev-parse", f"{head}^").decode().strip(),
        "freeze_subject": _git_bytes(
            "show", "-s", "--format=%s", head
        ).decode().strip(),
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
        "head_commit", "parent_commit", "freeze_subject", "worktree_clean",
        "tracked_source_count", "tracked_sources_sha256",
        "source_closure_path", "source_closure_bytes",
        "source_closure_sha256", "source_closure_row_count",
        "source_closure_live_bytes_exact", "metrics_module",
    }
    if set(value) != fields:
        raise RegenerationError("V3 source-freeze observation field drift")
    if (
        value["head_commit"] != expected_commit
        or value["parent_commit"] != SOURCE_PARENT_COMMIT
        or value["freeze_subject"]
        != "Freeze semantic physical graph edge handoff qualification V3"
        or value["worktree_clean"] is not True
        or value["source_closure_live_bytes_exact"] is not True
        or not isinstance(value["tracked_source_count"], int)
        or value["tracked_source_count"] < 8
        or not isinstance(value["source_closure_row_count"], int)
        or value["source_closure_row_count"] < 1
    ):
        raise RegenerationError("V3 source-freeze observation failed")
    _hex64(value["tracked_sources_sha256"], "tracked source projection")
    _hex64(value["source_closure_sha256"], "source closure")
    return json.loads(json.dumps(value))


def _load_jsonl_at(root_fd: int, leaf: str) -> tuple[bytes, list[dict[str, Any]]]:
    return V2E.V1._load_jsonl_at(root_fd, leaf)


def _load_internal_gates(
    *,
    root_fd: int,
    module: Any,
    source_freeze_commit: str,
    historical_receipt_path: Path,
    material_root: Path,
) -> tuple[
    dict[str, bytes], dict[str, Any], dict[str, Any], dict[str, Any],
    dict[str, Any], dict[str, Any],
]:
    ordinary_raw: dict[str, bytes] = {}
    ordinary: dict[str, dict[str, Any]] = {}
    for leaf in _NEW_ORDINARY_LEAVES:
        raw, value = V2E._load_ordinary_json_at(root_fd, leaf)
        V2E.V1._validate_no_self_digest(value, f"V3 {leaf}")
        ordinary_raw[leaf] = raw
        ordinary[leaf] = value
    historical = validate_existing_historical_custody_receipt(
        historical_receipt_path
    )
    historical_binding = _binding(
        historical_receipt_path, "combined historical custody receipt"
    )
    _call(
        module,
        "validate_external_v1_v2_custody_receipt",
        historical,
        expected_binding=historical_binding,
    )
    invariance = _call(
        module,
        "validate_scientific_invariance_receipt",
        ordinary["scientific_invariance_receipt.json"],
    )
    nonreuse = _call(
        module,
        "validate_v1_v2_custody_and_nonreuse",
        ordinary["v1_v2_custody_and_nonreuse.json"],
        source_freeze_commit=source_freeze_commit,
    )
    reproduction = _call(
        module,
        "validate_first_eight_reproduction",
        ordinary["v1_v2_v3_first_eight_reproduction.json"],
        historical_custody_receipt_binding=historical_binding,
    )
    if invariance.get("pass") is not True or nonreuse.get("pass") is not True:
        raise RegenerationError("V3 invariance or historical nonreuse gate failed")
    material_binding = reproduction.get(
        "first_eight_behavioural_probe_material_binding"
    )
    if not isinstance(material_binding, Mapping):
        raise RegenerationError("first-eight probe material binding is absent")
    probe_path = material_root / str(material_binding.get("path"))
    live_probe_binding = _binding(probe_path, "first-eight probe material")
    if (
        material_binding.get("path")
        != "reproduction/first_eight_behavioural_probes.npz"
        or material_binding.get("bytes") != live_probe_binding["bytes"]
        or material_binding.get("sha256") != live_probe_binding["sha256"]
    ):
        raise RegenerationError("first-eight probe material binding drift")
    first_probe = inspect_behavioural_probe_npz(
        probe_path, metrics=module, first_eight_only=True
    )
    if first_probe["projection_sha256"] != reproduction[
        "first_eight_behavioural_probe_projection_sha256"
    ]:
        raise RegenerationError("first-eight raw behavioural gate evidence drift")
    # The pure evidence cross-link reopens equivalent arrays independently in
    # this process.  No simulator or model is involved.
    first_arrays, _inspection = _load_npz_arrays(
        probe_path,
        _call(module, "behavioural_probe_authority")[
            "first_eight_material_npz_members"
        ],
        label="first-eight behavioural probes",
    )
    _call(
        module,
        "validate_first_eight_reproduction_evidence",
        reproduction,
        first_arrays,
    )
    _validate_first_probe_gate_disposition(reproduction, first_probe)
    return (
        ordinary_raw, invariance, nonreuse, reproduction, historical,
        first_probe,
    )


def _validate_first_probe_gate_disposition(
    reproduction: Mapping[str, Any], first_probe: Mapping[str, Any]
) -> None:
    """Apply success/terminal policy after exact raw-evidence reconstruction."""

    passed = reproduction.get("pass")
    within_passed = first_probe.get("all_within_version_pass")
    cross_passed = first_probe.get("all_historical_cross_version_pass")
    if type(passed) is not bool or type(within_passed) is not bool or type(
        cross_passed
    ) is not bool:
        raise RegenerationError("first-eight gate Boolean projection drift")
    if passed:
        if (
            reproduction.get("status") != "PASS"
            or reproduction.get("technical_disposition") is not None
            or reproduction.get("full_collection_authorized") is not True
            or within_passed is not True
            or cross_passed is not True
        ):
            raise RegenerationError(
                "first-eight successful raw behavioural gate evidence drift"
            )
        return
    if (
        reproduction.get("status") != MISMATCH_DISPOSITION
        or reproduction.get("technical_disposition") != MISMATCH_DISPOSITION
        or reproduction.get("full_collection_authorized") is not False
        or (
            reproduction.get("semantic_all_pass") is True
            and reproduction.get("behavioural_all_pass") is True
        )
    ):
        raise RegenerationError("first-eight terminal gate disposition drift")


def _material_npz_arrays(
    payload_path: Path,
    metadata: Mapping[str, Any],
    *,
    material_root: Path,
    module: Any,
    archive_comment: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen one shard without pickle and validate every exact C-byte row."""

    import numpy as np

    evidence = metadata.get("persisted_array_evidence")
    if not isinstance(evidence, Mapping):
        raise RegenerationError(f"{label} lacks persisted-array evidence")
    rows = evidence.get("arrays")
    if not isinstance(rows, list) or not rows:
        raise RegenerationError(f"{label} persisted-array rows are absent")
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "member", "dtype_str", "shape", "c_contiguous",
            "array_bytes_sha256",
        }:
            raise RegenerationError(f"{label} array row field drift: {index}")
        name = row.get("member")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise RegenerationError(f"{label} unsafe array member: {index}")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise RegenerationError(f"{label} array member order/uniqueness drift")
    members = V2E._read_npz_members(
        payload_path,
        label,
        expected_names={f"{name}.npy" for name in names},
        expected_comment=archive_comment,
    )
    arrays: dict[str, Any] = {}
    rebuilt_rows: list[dict[str, Any]] = []
    for name in names:
        member = members[f"{name}.npy"]
        dtype = np.dtype(member["dtype"])
        if dtype.str != member["dtype"]:
            raise RegenerationError(f"{label}/{name} dtype is noncanonical")
        array = np.frombuffer(member["payload"], dtype=dtype).reshape(
            member["shape"]
        )
        arrays[name] = array
        rebuilt_rows.append(
            {
                "member": name,
                "dtype_str": dtype.str,
                "shape": list(array.shape),
                "c_contiguous": True,
                "array_bytes_sha256": hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    payload_live = _binding(payload_path, label)
    relative_payload = payload_path.relative_to(material_root).as_posix()
    payload_binding = {
        "path": relative_payload,
        "bytes": payload_live["bytes"],
        "sha256": payload_live["sha256"],
    }
    payload = metadata.get("payload")
    if payload != {
        "role": "material_shard_payload",
        **payload_binding,
        "kind": "npz",
    }:
        raise RegenerationError(f"{label} material payload binding drift")
    if (
        evidence.get("payload_file") != payload_binding
        or evidence.get("arrays") != rebuilt_rows
        or evidence.get("array_count") != len(rebuilt_rows)
        or evidence.get("array_inventory_sha256")
        != hashlib.sha256(canonical_json_bytes(rebuilt_rows)).hexdigest()
        or evidence.get("save_reopen_validation_passed") is not True
    ):
        raise RegenerationError(f"{label} raw persisted-array evidence drift")
    _call(
        module,
        "validate_persisted_array_evidence",
        evidence,
        reopened_arrays=arrays,
    )
    snapshot = metadata.get("snapshot")
    if isinstance(snapshot, Mapping) and "previous_applied_command_sha256" in snapshot:
        _call(
            module,
            "validate_snapshot_previous_applied_command_binding",
            snapshot,
            evidence,
            reopened_arrays=arrays,
        )
    return arrays, {
        "payload_path": relative_payload,
        "payload_bytes": payload_binding["bytes"],
        "payload_sha256": payload_binding["sha256"],
        "array_count": len(rebuilt_rows),
        "array_inventory_sha256": evidence["array_inventory_sha256"],
    }


def _historical_semantic_manifests_equal(
    *,
    pool_index: int,
    metadata: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> bool:
    evidence = metadata.get("snapshot_semantic_evidence")
    probes = metadata.get("behavioural_probes")
    pairs = historical.get("first_eight_pairs")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(probes, Mapping)
        or not isinstance(pairs, list)
        or len(pairs) != 8
    ):
        raise RegenerationError("historical semantic manifest projection is absent")
    pair = pairs[pool_index]
    if not isinstance(pair, Mapping) or pair.get("pool_index") != pool_index:
        raise RegenerationError("historical semantic pair order drift")
    result = True
    for version in ("V1", "V2"):
        historical_version = pair.get(version.lower())
        version_probe = probes.get(version)
        if not isinstance(historical_version, Mapping) or not isinstance(
            version_probe, Mapping
        ):
            raise RegenerationError("historical semantic version projection drift")
        identity = version_probe.get("snapshot_identity")
        if not isinstance(identity, Mapping):
            raise RegenerationError("historical probe snapshot identity is absent")
        if (
            identity.get("artifact_file_sha256")
            != historical_version.get("artifact_file_sha256")
            or identity.get("snapshot_semantic_digest_v1")
            != historical_version.get("snapshot_semantic_digest_v1")
        ):
            raise RegenerationError("historical shard identity cross-link drift")
        result = bool(
            result and historical_version.get("semantic_evidence") == evidence
        )
    return result


def _qualification_equivalence_record(
    *,
    pool_index: int,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    module: Any,
    historical: Mapping[str, Any],
    serializer_authority: Mapping[str, Any],
    behavioural_authority: Mapping[str, Any],
    physical_authority: Mapping[str, Any],
) -> dict[str, Any]:
    augmentation = {
        field: metadata[field]
        for field in (
            "snapshot_semantic_evidence", "snapshot_identity",
            "behavioural_probes",
        )
    }
    _call(
        module,
        "validate_qualification_shard_augmentation",
        augmentation,
        pool_index=pool_index,
        reopened_arrays=arrays,
    )
    canonical = bytes(memoryview(arrays["snapshot_semantic_bytes"]).cast("B"))
    semantic = _validate_semantic_worker_result(
        canonical,
        augmentation["snapshot_semantic_evidence"],
        metrics=module,
        serializer_authority=serializer_authority,
    )
    if semantic["snapshot_semantic_digest_v1"] != augmentation[
        "snapshot_identity"
    ]["snapshot_semantic_digest_v1"]:
        raise RegenerationError("qualification semantic root identity drift")
    probes = augmentation["behavioural_probes"]
    versions = ("V1", "V2", "V3") if pool_index < 8 else ("V3",)
    traces: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, dict[str, Any]] = {}
    trace_members = [
        member
        for member in behavioural_authority["npz_members"]
        if member
        not in {
            "trace_offsets", "version_code", "pool_index", "trial_index",
            "stuck", "termination_code",
            "final_snapshot_semantic_digest_bytes",
        }
    ]
    for version in versions:
        version_row = probes[version]
        identities[version] = dict(version_row["snapshot_identity"])
        traces[version] = []
        for trial_index in (0, 1):
            trace: dict[str, Any] = {}
            for member in trace_members:
                trace[member] = arrays[
                    f"probe__{version}__{trial_index}__{member}"
                ]
            digest_array = arrays[
                f"probe__{version}__{trial_index}__"
                "final_snapshot_semantic_digest_bytes"
            ]
            trace["final_snapshot_semantic_digest_v1"] = bytes(
                memoryview(digest_array).cast("B")
            ).hex()
            trace["termination_reason"] = version_row[
                "trial_termination_reasons"
            ][trial_index]
            trace["stuck"] = _derive_probe_stuck(trace, physical_authority)
            traces[version].append(trace)
    semantic_manifests_equal = (
        _historical_semantic_manifests_equal(
            pool_index=pool_index, metadata=metadata, historical=historical
        )
        if pool_index < 8
        else True
    )
    return _call(
        module,
        "build_snapshot_equivalence_record",
        pool_index=pool_index,
        versions=identities,
        behavioural_probe_traces=traces,
        semantic_manifests_equal=semantic_manifests_equal,
    )


def _validate_v3_material_payloads(
    material_root: Path | str,
    *,
    module: Any,
    historical: Mapping[str, Any],
    expected_metadata_paths: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = V2E._root_path(material_root, "V3 material root")
    live_authority = _call(module, "reducer_authority")
    try:
        archive_comment = live_authority["persisted_array_hash_authority"][
            "npz_archive_comment_utf8"
        ]
    except (KeyError, TypeError) as exc:
        raise RegenerationError("V3 material archive-comment authority drift") from exc
    if not isinstance(archive_comment, str) or not archive_comment:
        raise RegenerationError("V3 material archive-comment authority is invalid")
    serializer_authority = _call(module, "semantic_snapshot_serializer_authority")
    behavioural_authority = _call(module, "behavioural_probe_authority")
    physical_authority = _call(module, "physical_trace_reduction_authority")
    observed_paths = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("metadata.json"))
    ]
    expected = sorted(expected_metadata_paths)
    if observed_paths != expected:
        raise RegenerationError("V3 material stage/shard path inventory drift")
    rows: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for relative in observed_paths:
        metadata_path = root / relative
        raw, metadata = _load_canonical_object(
            metadata_path, f"V3 material {relative}"
        )
        if metadata.get("experiment_id") != EXPERIMENT_ID:
            raise RegenerationError(f"V3 material identity drift: {relative}")
        payload = metadata.get("payload")
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("path"), str
        ):
            raise RegenerationError(f"V3 material payload binding absent: {relative}")
        payload_path = root / str(payload["path"])
        if payload_path.resolve(strict=True).parent != metadata_path.parent.resolve(
            strict=True
        ):
            raise RegenerationError(f"V3 payload/metadata directory drift: {relative}")
        arrays, validation = _material_npz_arrays(
            payload_path,
            metadata,
            material_root=root,
            module=module,
            archive_comment=archive_comment,
            label=f"V3 material {relative}",
        )
        row = {
            "metadata_path": relative,
            "metadata_bytes": len(raw),
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            **validation,
        }
        if relative.startswith("qualification/pool-"):
            try:
                pool_index = int(metadata["pool_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RegenerationError("qualification pool identity drift") from exc
            record = _qualification_equivalence_record(
                pool_index=pool_index,
                metadata=metadata,
                arrays=arrays,
                module=module,
                historical=historical,
                serializer_authority=serializer_authority,
                behavioural_authority=behavioural_authority,
                physical_authority=physical_authority,
            )
            equivalence.append(record)
            row["snapshot_semantic_digest_v1"] = record["versions"]["V3"][
                "snapshot_semantic_digest_v1"
            ]
            row["behavioural_probe_version_count"] = len(record["versions"])
        rows.append(row)
    if len(equivalence) not in (8, 256) or [
        row["pool_index"] for row in equivalence
    ] != list(range(len(equivalence))):
        raise RegenerationError("V3 qualification/equivalence coverage drift")
    return {
        "validated_shard_count": len(rows),
        "qualification_shard_count": len(equivalence),
        "all_npz_members_reopened": True,
        "all_dtype_shape_raw_c_byte_hashes_exact": True,
        "all_v3_npz_archive_comments_exact": True,
        "semantic_payloads_independently_parsed": len(equivalence),
        "shard_projection_sha256": hashlib.sha256(
            canonical_json_bytes(rows)
        ).hexdigest(),
    }, equivalence


def _validate_semantic_equivalence_evidence(
    *,
    root_fd: int,
    root: Path,
    material_root: Path,
    module: Any,
    historical_binding: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    equivalence_records: Sequence[Mapping[str, Any]],
    terminal: bool,
) -> dict[str, Any]:
    first_path = material_root / (
        "reproduction/first_eight_behavioural_probes.npz"
    )
    behavioural = _call(module, "behavioural_probe_authority")
    first_arrays, first_inspection = _load_npz_arrays(
        first_path,
        behavioural["first_eight_material_npz_members"],
        label="first-eight behavioural probe material",
    )
    first_projection = _call(
        module,
        "behavioural_probe_npz_projection_sha256",
        first_arrays,
        first_eight_only=True,
    )
    material_binding = {
        "path": "reproduction/first_eight_behavioural_probes.npz",
        "bytes": first_inspection["binding"]["bytes"],
        "sha256": first_inspection["binding"]["sha256"],
    }
    rebuilt_reproduction = _call(
        module,
        "build_first_eight_reproduction",
        list(equivalence_records[:8]),
        historical_custody_receipt_binding=historical_binding,
        first_eight_behavioural_probe_material_binding=material_binding,
        first_eight_behavioural_probe_projection_sha256=first_projection,
    )
    if canonical_document_bytes(rebuilt_reproduction) != canonical_document_bytes(
        reproduction
    ):
        raise RegenerationError(
            "first-eight receipt differs from raw shard/trace reconstruction"
        )
    _call(
        module,
        "validate_first_eight_reproduction_evidence",
        reproduction,
        first_arrays,
    )
    if terminal:
        if (
            reproduction.get("pass") is not False
            or reproduction.get("full_collection_authorized") is not False
            or reproduction.get("status") != MISMATCH_DISPOSITION
            or len(equivalence_records) != 8
        ):
            raise RegenerationError("V3 terminal semantic gate disposition drift")
        return {
            "terminal": True,
            "first_eight_row_count": 8,
            "first_eight_receipt_exact_rebuild": True,
            "first_eight_material_binding": material_binding,
            "first_eight_projection_sha256": first_projection,
            "technical_disposition": MISMATCH_DISPOSITION,
            "scientific_result_authorized": False,
        }
    if (
        reproduction.get("pass") is not True
        or _call(module, "authorizes_full_v3_collection", reproduction) is not True
        or len(equivalence_records) != 256
    ):
        raise RegenerationError("successful V3 root lacks semantic/probe pass gate")
    index_raw, index = V2E.V1._load_json_at(
        root_fd, "snapshot_equivalence_index.json"
    )
    probe_path = root / "snapshot_behavioural_probes.npz"
    final_arrays, final_inspection = _load_npz_arrays(
        probe_path,
        behavioural["npz_members"],
        label="official snapshot behavioural probes",
    )
    _call(
        module,
        "validate_final_behavioural_probe_assembly",
        final_arrays,
        first_arrays,
    )
    validated_index = _call(
        module,
        "validate_snapshot_equivalence_evidence",
        index,
        final_arrays,
        first_arrays,
    )
    if validated_index.get("records") != list(equivalence_records):
        raise RegenerationError(
            "snapshot equivalence index differs from all 256 material shards"
        )
    final_projection = _call(
        module,
        "behavioural_probe_npz_projection_sha256",
        final_arrays,
        first_eight_only=False,
    )
    official_binding = {
        "path": "snapshot_behavioural_probes.npz",
        "bytes": final_inspection["binding"]["bytes"],
        "sha256": final_inspection["binding"]["sha256"],
    }
    if (
        validated_index.get("behavioural_probe_npz_binding")
        != official_binding
        or validated_index.get("behavioural_probe_npz_projection_sha256")
        != final_projection
        or validated_index.get("first_eight_material_probe_binding")
        != material_binding
        or validated_index.get("first_eight_material_probe_projection_sha256")
        != first_projection
    ):
        raise RegenerationError("snapshot equivalence payload binding drift")
    return {
        "terminal": False,
        "first_eight_row_count": 8,
        "complete_row_count": 256,
        "historical_row_count": 8,
        "v3_only_row_count": 248,
        "first_eight_receipt_exact_rebuild": True,
        "snapshot_equivalence_index_binding": {
            "path": "snapshot_equivalence_index.json",
            "bytes": len(index_raw),
            "sha256": hashlib.sha256(index_raw).hexdigest(),
        },
        "snapshot_behavioural_probe_binding": official_binding,
        "snapshot_behavioural_probe_projection_sha256": final_projection,
        "first_eight_material_binding": material_binding,
        "first_eight_projection_sha256": first_projection,
        "all_semantic_and_behavioural_rows_pass": True,
        "scientific_result_authorized": True,
    }


def _validate_historical_nonreuse(
    *,
    historical: Mapping[str, Any],
    v3_root: Path,
    v3_material_root: Path,
) -> dict[str, Any]:
    historical_files: list[dict[str, Any]] = []
    for key in (
        "v1_official_root", "v1_material_root",
        "v2_official_root", "v2_material_root",
    ):
        value = historical.get(key)
        if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
            raise RegenerationError("historical root manifest is absent")
        historical_files.extend(value["files"])
    v3_official = _root_inventory(v3_root, "V3 official root")
    v3_material = _root_inventory(v3_material_root, "V3 material root")
    historical_inodes = {
        (int(row["device"]), int(row["inode"])) for row in historical_files
    }
    v3_files = [*v3_official["files"], *v3_material["files"]]
    shared_inodes = [
        row for row in v3_files
        if (int(row["device"]), int(row["inode"])) in historical_inodes
    ]
    if shared_inodes:
        raise RegenerationError("V3 shares a file inode with historical evidence")
    historical_payload_shas = {
        str(row["sha256"])
        for row in historical_files
        if str(row["path"]).endswith("payload.npz")
    }
    v3_payload_rows = [
        row for row in v3_material["files"]
        if str(row["path"]).endswith("payload.npz")
    ]
    copied = [
        row for row in v3_payload_rows
        if str(row["sha256"]) in historical_payload_shas
    ]
    if copied:
        raise RegenerationError("V3 material contains a historical NPZ file copy")
    scientific_official_rows = [
        row for row in v3_official["files"]
        if row["path"] not in _PUBLICATION_LEAVES
    ]
    return {
        "historical_file_count": len(historical_files),
        "v3_scientific_official_file_count": len(scientific_official_rows),
        "v3_material_file_count": v3_material["file_count"],
        "historical_v3_shared_inode_count": 0,
        "historical_payload_whole_file_copy_count": 0,
        "v3_material_payload_count": len(v3_payload_rows),
        "historical_runtime_artifact_or_shard_reused": False,
        "v3_scientific_official_manifest_sha256": hashlib.sha256(
            canonical_document_bytes(scientific_official_rows)
        ).hexdigest(),
        "v3_material_manifest_sha256": v3_material["manifest_sha256"],
    }


def _official_identity_validation(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    token = "physical_graph_edge_handoff_qualification_v3"
    checked: list[str] = []
    for label, value in documents.items():
        schema = value.get("schema")
        if (
            not isinstance(schema, str)
            or token not in schema
            or value.get("experiment_id") != EXPERIMENT_ID
        ):
            raise RegenerationError(f"official {label} lacks exact V3 identity")
        checked.append(label)
    return {
        "checked_document_count": len(checked),
        "checked_documents": checked,
        "official_experiment_id": EXPERIMENT_ID,
        "official_schema_token": token,
        "all_official_document_roots_use_v3_identity": True,
    }


def _official_binding(leaf: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": leaf,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_v3_scientific_evidence(
    *,
    root: Path,
    root_fd: int,
    module: Any,
    authority: Mapping[str, Any],
    contract_raw: bytes,
    contract: Mapping[str, Any],
    ordinary_raw: Mapping[str, bytes],
    semantic_validation: Mapping[str, Any],
    material_root: Path,
    historical: Mapping[str, Any],
    invariance: Mapping[str, Any],
    nonreuse: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    raw_documents: dict[str, bytes] = {"contract.json": contract_raw}
    documents: dict[str, dict[str, Any]] = {"contract.json": dict(contract)}
    for _name, leaf in _DOCUMENT_LEAVES.items():
        raw, value = V2E.V1._load_json_at(root_fd, leaf)
        raw_documents[leaf] = raw
        documents[leaf] = value
    identity_validation = _official_identity_validation(documents)
    raw_ledgers: dict[str, bytes] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for name, leaf in _LEDGER_LEAVES.items():
        raw, rows = _load_jsonl_at(root_fd, leaf)
        raw_ledgers[leaf] = raw
        ledgers[name] = rows
    evidence_documents = {
        name: documents[leaf] for name, leaf in _DOCUMENT_LEAVES.items()
    }
    document_validation = V2E.V1._validate_document_authorities(
        evidence_documents, authority
    )
    ledger_validation = V2E.V1._validate_ledger_authorities(
        ledgers, evidence_documents, authority
    )
    runtime_validation = V2E.V1._validate_runtime_environments(
        module, evidence_documents, ledgers
    )
    encoder_source_validation = V2E.V1._validate_external_encoder_source(
        evidence_documents["latent_index"]
    )

    symbols: dict[str, int] = {}
    captured_reset: dict[str, list[bytes]] = {}
    captured_physical: dict[str, dict[str, list[bytes]]] = {}
    npz_authorities = dict(authority["npz_authorities"])
    behavioural_npz_authority = _call(module, "behavioural_probe_authority").get(
        "npz_members"
    )
    if not isinstance(behavioural_npz_authority, Mapping):
        raise RegenerationError("V3 behavioural NPZ member authority is absent")
    npz_authorities["snapshot_behavioural_probes.npz"] = (
        behavioural_npz_authority
    )
    if set(npz_authorities) != set(_SCIENTIFIC_NPZ_LEAVES):
        raise RegenerationError("V3 scientific NPZ authority inventory drift")
    npz_inspections = {
        leaf: V2E.V1._inspect_npz_at(
            root_fd,
            leaf,
            npz_authorities[leaf],
            symbols=symbols,
            captured_reset_slices=captured_reset,
            captured_physical_slices=captured_physical,
        )
        for leaf in _SCIENTIFIC_NPZ_LEAVES
    }
    if symbols.get("U") != evidence_documents["pixel_index"].get(
        "unique_pixel_count"
    ):
        raise RegenerationError("V3 latent U dimension differs from pixel index")
    pure_npz = _call(
        module,
        "validate_npz_inspections",
        [npz_inspections[leaf] for leaf in _SCIENTIFIC_NPZ_LEAVES],
    )
    if not isinstance(pure_npz, Mapping) or set(pure_npz) != set(
        _SCIENTIFIC_NPZ_LEAVES
    ):
        raise RegenerationError("V3 pure inherited NPZ projection drift")
    inherited_npz_inspections = {
        leaf: npz_inspections[leaf] for leaf in _PAYLOAD_LEAVES
    }
    previous_raw = V2E._validate_official_previous_command_raw_hashes(
        root=root,
        state_snapshot_index=evidence_documents["state_snapshot_index"],
        module=module,
    )
    # The frozen inherited cross-link helper names its legacy canonical-array
    # hash in one slot.  The official V3 document remains untouched and has
    # already passed the raw-C-byte validator above.
    bridge_documents = json.loads(json.dumps(evidence_documents))
    legacy_rows = npz_inspections["state_snapshots.npz"]["members"][
        "previous_applied_command"
    ]["row_or_slice_sha256s"]
    snapshot_records = bridge_documents["state_snapshot_index"]["records"]
    if len(legacy_rows) != len(snapshot_records):
        raise RegenerationError("V3 legacy previous-command bridge drift")
    for record, legacy_sha in zip(snapshot_records, legacy_rows):
        record["previous_applied_command_sha256"] = legacy_sha
    npz_cross_links = V2E.V1._validate_npz_cross_links(
        bridge_documents, ledgers, inherited_npz_inspections
    )
    reset_validation = V2E.V1._validate_reset_trace_pairs(
        evidence_documents["state_snapshot_index"],
        npz_inspections["candidate_traces.npz"],
        captured_reset,
        authority,
    )
    raw_teacher = V2E.V1._validate_raw_teacher_evidence(
        evidence_documents,
        inherited_npz_inspections,
        captured_physical,
        authority["physical_trace_reduction_authority"],
    )
    raw_candidate = V2E.V1._validate_raw_candidate_and_repeat_evidence(
        evidence_documents,
        ledgers,
        inherited_npz_inspections,
        captured_physical,
        authority["physical_trace_reduction_authority"],
    )
    candidate_alignment = V2E._validate_candidate_port_metric_alignment_from_raw(
        documents=evidence_documents,
        ledgers=ledgers,
        captured=captured_physical,
        material_root=material_root,
        module=module,
    )
    external_bindings = _call(module, "external_artifact_bindings", contract)
    external_validations = [
        V2E.V1._stream_external_binding(binding, f"external artifact {index}")
        for index, binding in enumerate(external_bindings)
    ]
    if [row["role"] for row in external_validations] != authority[
        "external_artifact_roles"
    ]:
        raise RegenerationError("V3 external artifact role order drift")
    predecessor = V2E.V1._validate_v2_context(contract, module)
    index_raw, snapshot_equivalence_index = V2E.V1._load_json_at(
        root_fd, "snapshot_equivalence_index.json"
    )
    behavioural_authority = _call(module, "behavioural_probe_authority")
    final_behavioural_arrays, _final_probe_inspection = _load_npz_arrays(
        root / "snapshot_behavioural_probes.npz",
        behavioural_authority["npz_members"],
        label="V3 official behavioural probes for metric recomputation",
    )
    first_behavioural_arrays, _first_probe_inspection = _load_npz_arrays(
        material_root / "reproduction/first_eight_behavioural_probes.npz",
        behavioural_authority["first_eight_material_npz_members"],
        label="V3 first-eight behavioural probes for metric recomputation",
    )
    evidence = {
        **evidence_documents,
        **ledgers,
        "npz_inspections": [
            npz_inspections[leaf] for leaf in _SCIENTIFIC_NPZ_LEAVES
        ],
        "external_historical_custody_receipt": dict(historical),
        "v1_v2_custody_and_nonreuse": dict(nonreuse),
        "scientific_invariance_receipt": dict(invariance),
        "v1_v2_v3_first_eight_reproduction": dict(reproduction),
        "snapshot_equivalence_index": snapshot_equivalence_index,
        "snapshot_behavioural_probe_arrays": final_behavioural_arrays,
        "first_eight_behavioural_probe_arrays": first_behavioural_arrays,
    }
    expected_evidence_keys = authority.get(
        "v3_recompute_evidence_keys", authority["evidence_keys"]
    )
    if set(evidence) != set(expected_evidence_keys):
        raise RegenerationError("V3 assembled scientific evidence-key drift")
    recomputed = _call(module, "recompute_metrics", evidence)
    recomputed_raw, recomputed = V2E.V1._canonicalise_object(
        dict(recomputed), "V3 recomputed metrics"
    )
    V2E.V1._validate_content_digest(recomputed, "V3 recomputed metrics")
    persisted_metrics_raw = raw_documents.get("metrics.json")
    if persisted_metrics_raw is None:
        persisted_metrics_raw, persisted_metrics = V2E.V1._load_json_at(
            root_fd, "metrics.json"
        )
        raw_documents["metrics.json"] = persisted_metrics_raw
        documents["metrics.json"] = persisted_metrics
    if persisted_metrics_raw != recomputed_raw:
        raise RegenerationError(
            "V3 persisted metrics differ from exact raw-ledger rebuild"
        )

    scientific_bindings: dict[str, dict[str, Any]] = {}
    for leaf, raw in {**raw_documents, **raw_ledgers}.items():
        scientific_bindings[leaf] = _official_binding(leaf, raw)
    for leaf in _PAYLOAD_LEAVES:
        scientific_bindings[leaf] = {
            key: npz_inspections[leaf][key]
            for key in ("path", "bytes", "sha256")
        }
    for leaf, raw in ordinary_raw.items():
        scientific_bindings[leaf] = _official_binding(leaf, raw)
    scientific_bindings["snapshot_equivalence_index.json"] = _official_binding(
        "snapshot_equivalence_index.json", index_raw
    )
    probe_binding = semantic_validation["snapshot_behavioural_probe_binding"]
    scientific_bindings["snapshot_behavioural_probes.npz"] = dict(probe_binding)
    expected_scientific = set(V3_SUCCESS_FILES) - set(_PUBLICATION_LEAVES)
    if set(scientific_bindings) != expected_scientific:
        raise RegenerationError(
            "V3 scientific input binding inventory drift: "
            f"{sorted(expected_scientific - set(scientific_bindings))}"
        )
    validation = {
        "official_identity_validation": identity_validation,
        "document_validation": document_validation,
        "ledger_validation": ledger_validation,
        "npz_validation": V2E.V1._compact_npz_validation(npz_inspections),
        "npz_symbols": {name: symbols[name] for name in sorted(symbols)},
        "pure_npz_validation_passed": True,
        "npz_cross_link_validation": npz_cross_links,
        "official_previous_command_raw_hash_validation": previous_raw,
        "reset_pair_validation": reset_validation,
        "raw_teacher_physics_validation": raw_teacher,
        "raw_candidate_and_repeat_physics_validation": raw_candidate,
        "candidate_port_metric_alignment_validation": candidate_alignment,
        "predecessor_context_validation": predecessor,
        "external_artifact_validation": external_validations,
        "external_encoder_source_validation": encoder_source_validation,
        "runtime_environment_validation": runtime_validation,
        "metrics_exact_byte_equal": True,
        "recomputed_metrics_sha256": hashlib.sha256(recomputed_raw).hexdigest(),
    }
    return recomputed, scientific_bindings, validation


def _material_metadata_paths_for_success(
    panel_manifest: Mapping[str, Any]
) -> list[str]:
    states = panel_manifest.get("states")
    if not isinstance(states, list) or len(states) != 64:
        raise RegenerationError("V3 panel state population is absent")
    state_ids: list[str] = []
    heldout: list[str] = []
    for row in states:
        if not isinstance(row, Mapping) or not isinstance(row.get("state_id"), str):
            raise RegenerationError("V3 panel state identity drift")
        state_ids.append(row["state_id"])
        if row.get("role") == "DEVELOPMENT_HELDOUT":
            heldout.append(row["state_id"])
    if len(set(state_ids)) != 64 or len(heldout) != 16:
        raise RegenerationError("V3 panel state role/identity cardinality drift")
    return [
        *(f"qualification/pool-{index:03d}/metadata.json" for index in range(256)),
        *(f"selected/{state_id}/metadata.json" for state_id in state_ids),
        *(f"fanout/{state_id}/metadata.json" for state_id in state_ids),
        *(f"repeat/{state_id}/metadata.json" for state_id in heldout),
    ]


def _receipt_counters() -> dict[str, int]:
    return {
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
        "runner_calls": 0,
    }


def _validate_publication_if_present(
    *,
    root: Path,
    root_fd: int,
    complete: bool,
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    expected_receipt_sha256: str,
    semantic_validation: Mapping[str, Any],
    historical_binding: Mapping[str, Any],
    module: Any,
) -> dict[str, Any] | None:
    if not complete:
        return None
    validator = getattr(module, "validate_result_publication_projection", None)
    report_builder = getattr(module, "build_result_report_bytes", None)
    if not callable(validator) or not callable(report_builder):
        raise RegenerationError(
            "frozen V3 result/publication projection API is absent"
        )
    result_raw, result = V2E.V1._load_json_at(root_fd, "result.json")
    report_raw = V2E.CUSTODY._read_regular_at(root_fd, "result.md")
    manifest_raw, manifest = V2E.V1._load_json_at(root_fd, "file_hashes.json")
    assert report_raw is not None
    try:
        report_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError("V3 result.md is not UTF-8") from exc
    if not report_raw or not report_raw.endswith(b"\n"):
        raise RegenerationError("V3 result.md must be nonempty and LF terminated")
    expected_rows = [dict(value) for value in scientific_bindings.values()]
    expected_rows.extend(
        (
            _official_binding("result.json", result_raw),
            _official_binding("result.md", report_raw),
        )
    )
    expected_rows.sort(key=lambda row: row["path"])
    if (
        manifest.get("schema")
        != "physical_graph_edge_handoff_qualification_v3.file_hashes.v1"
        or manifest.get("root") != str(root)
        or manifest.get("files") != expected_rows
        or manifest.get("file_count_excluding_self") != 27
        or manifest.get("bytes_excluding_self")
        != sum(int(row["bytes"]) for row in expected_rows)
        or manifest.get("file_hashes_self_sha256_excluded") is not True
    ):
        raise RegenerationError("V3 file_hashes.json differs from exact live bytes")
    V2E.V1._validate_content_digest(manifest, "V3 file hashes")
    scientific_storage_bytes = sum(
        int(binding["bytes"]) for binding in scientific_bindings.values()
    )
    if result.get("scientific_storage_bytes") != scientific_storage_bytes:
        raise RegenerationError(
            "V3 result scientific storage differs from exact input bytes"
        )
    projection = _call(
        module,
        "validate_result_publication_projection",
        result,
        recomputed_metrics=recomputed,
        scientific_bindings=scientific_bindings,
        runtime_contract=runtime_contract,
        independent_reducer_receipt_sha256=expected_receipt_sha256,
        snapshot_equivalence=semantic_validation,
        historical_custody_receipt_binding=historical_binding,
    )
    expected_report = _call(module, "build_result_report_bytes", projection)
    if not isinstance(expected_report, bytes) or report_raw != expected_report:
        raise RegenerationError("V3 result.md differs from exact reduced report")
    return {
        "result_binding": _official_binding("result.json", result_raw),
        "report_binding": _official_binding("result.md", report_raw),
        "manifest_binding": _official_binding("file_hashes.json", manifest_raw),
        "result_projection_validated": True,
        "report_exact_byte_equal": True,
        "manifest_exact_live_bytes": True,
    }


def build_regeneration_receipt(
    output_root: Path | str = DEFAULT_V3_OUTPUT_ROOT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    """Rebuild V3 custody, raw metrics, and semantic gates without execution."""

    module = _load_metrics_module() if metrics_module is None else metrics_module
    authority = _validate_v3_authority(module)
    root, root_fd = _open_official_root(output_root)
    material = (
        Path(material_root)
        if material_root is not None
        else root.parent / f"{root.name}_material"
    )
    historical_path = Path(historical_custody_receipt)
    try:
        inventory, terminal, complete = _validate_official_inventory(root_fd)
        contract_raw, contract = V2E.V1._load_json_at(root_fd, "contract.json")
        if (
            contract.get("schema")
            != "physical_graph_edge_handoff_qualification_v3.runtime_contract.v1"
            or contract.get("experiment_id") != EXPERIMENT_ID
        ):
            raise RegenerationError("V3 runtime contract identity drift")
        source_commit = contract.get("source_freeze_commit")
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise RegenerationError("V3 runtime source-freeze binding drift")
        actual_historical_binding = _binding(
            historical_path, "combined historical custody receipt"
        )
        if contract.get("historical_custody_receipt_binding") != actual_historical_binding:
            raise RegenerationError("V3 runtime historical custody binding drift")
        # Both pure accessors validate the complete runtime-contract bytes.
        _call(module, "external_artifact_bindings", contract)
        _call(module, "predecessor_context_binding", contract)
        observed_source = (
            _observe_source_freeze(contract, module)
            if source_freeze_observation is None
            else dict(source_freeze_observation)
        )
        source_freeze = _validate_source_freeze_observation(
            observed_source, expected_commit=source_commit
        )
        (
            ordinary_raw,
            invariance,
            nonreuse,
            reproduction,
            historical,
            first_probe,
        ) = _load_internal_gates(
            root_fd=root_fd,
            module=module,
            source_freeze_commit=source_commit,
            historical_receipt_path=historical_path,
            material_root=material,
        )
        if terminal:
            expected_material = [
                f"qualification/pool-{index:03d}/metadata.json"
                for index in range(8)
            ]
        else:
            _panel_raw, panel = V2E.V1._load_json_at(root_fd, "panel_manifest.json")
            expected_material = _material_metadata_paths_for_success(panel)
        material_validation, equivalence_records = _validate_v3_material_payloads(
            material,
            module=module,
            historical=historical,
            expected_metadata_paths=expected_material,
        )
        semantic_validation = _validate_semantic_equivalence_evidence(
            root_fd=root_fd,
            root=root,
            material_root=material,
            module=module,
            historical_binding=actual_historical_binding,
            reproduction=reproduction,
            equivalence_records=equivalence_records,
            terminal=terminal,
        )
        historical_nonreuse = _validate_historical_nonreuse(
            historical=historical,
            v3_root=root,
            v3_material_root=material,
        )
        counters = _receipt_counters()
        base = {
            "schema": REGENERATION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "pass": True,
            "reducer_authority_digest": authority["content_digest"],
            "source_freeze": source_freeze,
            "historical_custody_receipt_binding": actual_historical_binding,
            "historical_custody_projection_sha256": _call(
                module, "historical_custody_projection_sha256", historical
            ),
            "scientific_invariance_receipt_validated": invariance["pass"] is True,
            "v1_v2_custody_and_nonreuse_validated": nonreuse["pass"] is True,
            "first_eight_reproduction": reproduction,
            "first_eight_raw_probe_validation": first_probe,
            "snapshot_equivalence_validation": semantic_validation,
            "material_persisted_array_validation": material_validation,
            "historical_nonreuse_validation": historical_nonreuse,
            "scientific_execution_counters": counters,
        }
        if terminal:
            receipt = {
                **base,
                "mode": "INDEPENDENT_SEMANTIC_REPRODUCTION_TERMINAL_VALIDATION_ONLY",
                "technical_disposition": MISMATCH_DISPOSITION,
                "scientific_result_produced": False,
                "official_root_inventory": {
                    "leaf_count": 4,
                    "leaves": inventory,
                    "exact_terminal_inventory_validated": True,
                },
                "inputs": {
                    "contract.json": _official_binding("contract.json", contract_raw),
                    **{
                        leaf: _official_binding(leaf, ordinary_raw[leaf])
                        for leaf in _NEW_ORDINARY_LEAVES
                    },
                },
                "no_panel_candidate_ranker_metric_result_or_manifest_opened": True,
            }
            V2E.V1._validate_no_self_digest(receipt, "V3 terminal receipt")
            V2E.V1._reject_nonfinite(receipt, label="V3 terminal receipt")
            return receipt

        recomputed, scientific_bindings, scientific_validation = (
            _validate_v3_scientific_evidence(
                root=root,
                root_fd=root_fd,
                module=module,
                authority=authority,
                contract_raw=contract_raw,
                contract=contract,
                ordinary_raw=ordinary_raw,
                semantic_validation=semantic_validation,
                material_root=material,
                historical=historical,
                invariance=invariance,
                nonreuse=nonreuse,
                reproduction=reproduction,
            )
        )
        receipt = {
            **base,
            "mode": "INDEPENDENT_PERSISTED_PHYSICAL_AND_SEMANTIC_EVIDENCE_REDUCTION_ONLY",
            "technical_disposition": None,
            "scientific_result_produced": True,
            "official_root_inventory": {
                "scientific_leaf_count": 25,
                "allowed_complete_leaf_count": 28,
                "allowed_leaf_names_validated": True,
                "publication_all_or_absent_validated": True,
            },
            "inputs": {
                name: scientific_bindings[name]
                for name in sorted(scientific_bindings)
            },
            "scientific_validation": scientific_validation,
            "metrics_exact_byte_equal": True,
            "recomputed_metrics_sha256": scientific_validation[
                "recomputed_metrics_sha256"
            ],
        }
        V2E.V1._validate_no_self_digest(receipt, "V3 regeneration receipt")
        V2E.V1._reject_nonfinite(receipt, label="V3 regeneration receipt")
        receipt_sha = hashlib.sha256(canonical_document_bytes(receipt)).hexdigest()
        publication = _validate_publication_if_present(
            root=root,
            root_fd=root_fd,
            complete=complete,
            scientific_bindings=scientific_bindings,
            recomputed=recomputed,
            runtime_contract=contract,
            expected_receipt_sha256=receipt_sha,
            semantic_validation=semantic_validation,
            historical_binding=actual_historical_binding,
            module=module,
        )
        # Publication is deliberately not part of the receipt identity; it is
        # validated after rebuilding the same ordinary receipt bytes.
        del publication
        return receipt
    finally:
        os.close(root_fd)


def verify_and_emit(
    output_root: Path | str = DEFAULT_V3_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
        historical_custody_receipt=historical_custody_receipt,
    )
    if receipt.get("scientific_result_produced") is not True:
        raise RegenerationError(
            "terminal V3 evidence is validation-only; external receipt emission "
            "is forbidden"
        )
    root_path = Path(output_root)
    material_path = (
        Path(material_root)
        if material_root is not None
        else root_path.parent / f"{root_path.name}_material"
    )
    V2E._emit_external((root_path, material_path), output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str = DEFAULT_V3_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    raw, supplied = _load_canonical_object(
        Path(output), "V3 external regeneration receipt"
    )
    V2E.V1._validate_no_self_digest(supplied, "V3 external regeneration receipt")
    rebuilt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
        historical_custody_receipt=historical_custody_receipt,
    )
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError(
            "V3 external regeneration receipt differs from exact rebuild"
        )
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate PGEHQ V3 evidence and publication"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_V3_OUTPUT_ROOT)
    parser.add_argument("--material-root", type=Path, default=DEFAULT_V3_MATERIAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_RECEIPT)
    parser.add_argument("--historical-worker", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.historical_worker is not None:
        return _historical_worker(arguments.historical_worker)
    if arguments.output.exists() or arguments.output.is_symlink():
        receipt = validate_existing_regeneration_receipt(
            arguments.root,
            arguments.output,
            material_root=arguments.material_root,
        )
    else:
        receipt = verify_and_emit(
            arguments.root,
            arguments.output,
            material_root=arguments.material_root,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
