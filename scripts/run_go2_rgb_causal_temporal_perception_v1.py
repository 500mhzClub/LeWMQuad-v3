#!/usr/bin/env python3
"""Run the one bounded RGB causal-temporal perception falsification.

Importing this module is source-only: Torch, image decoders, generated inputs,
and tensor checkpoints are deferred until exact authority validation and a
mode-0700 attempt reservation have both succeeded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_rgb_causal_temporal_perception_v1_contract",
    _CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load temporal probe contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

PREFLIGHT_ENVIRONMENT_KEY = "LEWM_CAUSAL_TEMPORAL_PERCEPTION_V1_PREFLIGHT_JSON"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONFLICTING_ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)
SYNTHETIC_FAILURE_BOUNDARIES = (
    "ledger_before_header",
    "ledger_after_durable_header",
    "schedule",
    "gate",
    "n320_checkpoint",
    "raw_authority",
    "raw_indexes",
    "model_preparation",
    "training",
    "evaluation",
    "result_publication",
    "completion_publication",
)


def _failure_boundary(name: str) -> None:
    """Production no-op; tests monkeypatch this exact closed boundary set."""
    if name not in SYNTHETIC_FAILURE_BOUNDARIES:
        raise ValueError(f"unknown temporal V1 failure boundary: {name}")


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular(path: Path, *, expected_sha256: str | None = None) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"input is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError(f"input hash changed: {path}")
    return raw


def _read_pre_ledger_prefix(path: Path) -> bytes | None:
    """Observe a writer prefix without relying on the general read helper."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError(
            "O_NOFOLLOW is required for temporal V1 prefix custody"
        )
    try:
        before = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError("pre-ledger prefix is not a regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    if not (
        _fingerprint(before)
        == _fingerprint(opened_before)
        == _fingerprint(opened_after)
        == _fingerprint(after)
    ):
        raise RuntimeError("pre-ledger prefix changed while read")
    return b"".join(chunks)


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    return value, raw


def _binding(
    relative: str,
    value: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    return contract.artifact_binding(
        relative,
        raw,
        content_sha256=str(value["content_sha256"]),
    )


@dataclass
class OperationProgress:
    """Exact operations for terminal causal-temporal V1 receipts."""

    stage: str = "post_reservation_ledger_creation"
    update: int | None = None
    microbatch: int | None = None
    checkpoint_update: int | None = None
    role: str | None = None
    counts: dict[str, Any] = field(
        default_factory=contract.empty_partial_operation_counts
    )

    def enter(
        self,
        stage: str,
        *,
        update: int | None = None,
        microbatch: int | None = None,
        checkpoint_update: int | None = None,
        role: str | None = None,
    ) -> None:
        self.stage = stage
        self.update = update
        self.microbatch = microbatch
        self.checkpoint_update = checkpoint_update
        self.role = role

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in contract.PARTIAL_OPERATION_INTEGER_FIELDS:
            raise KeyError(f"unknown partial operation counter: {name}")
        if type(amount) is not int or amount <= 0:
            raise ValueError("partial operation increment must be positive")
        self.counts[name] += amount

    def snapshot(self) -> dict[str, Any]:
        return contract.validate_partial_operation_counts({
            key: list(value) if isinstance(value, list) else value
            for key, value in self.counts.items()
        })

    def location(self) -> dict[str, Any]:
        return {
            "name": self.stage,
            "update": self.update,
            "microbatch": self.microbatch,
            "checkpoint_update": self.checkpoint_update,
            "role": self.role,
        }


class _PreLedgerInitializationError(RuntimeError):
    """Carry the original failure and exact pre-ledger custody phase."""

    def __init__(
        self,
        *,
        boundary: str,
        error: BaseException,
        durable_header_raw: bytes | None,
        unaccepted_header_prefix_raw: bytes | None = None,
    ) -> None:
        super().__init__(str(error))
        self.boundary = boundary
        self.error = error
        self.durable_header_raw = durable_header_raw
        self.unaccepted_header_prefix_raw = unaccepted_header_prefix_raw


class PartialAccessLedger:
    """Fsynced hash-chained evidence around every generated-input read."""

    RELATIVE_PATH = "partial_access.jsonl"

    def __init__(
        self,
        output_root: Path,
        *,
        reservation: Mapping[str, Any],
        reservation_raw: bytes,
        header: Mapping[str, Any],
        header_raw: bytes,
        repository_root: Path = ROOT,
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            raise PermissionError(
                "O_NOFOLLOW is required for temporal V1 input custody"
            )
        self.output_root = output_root
        self.repository_root = repository_root.resolve()
        self.path = output_root / self.RELATIVE_PATH
        self.records = [dict(header)]
        self.raw_parts = [header_raw]
        self.previous_content_sha256 = str(header["content_sha256"])
        self.next_open_id = 1
        self.closed = False
        self.reservation_binding = _binding(
            "reservation.json", reservation, reservation_raw
        )
        expected_header = self.header_value(
            reservation=reservation,
            reservation_raw=reservation_raw,
        )
        if (
            dict(header) != expected_header
            or header_raw
            != contract.canonical_json_bytes(expected_header) + b"\n"
            or _read_regular(
                self.path,
                expected_sha256=hashlib.sha256(header_raw).hexdigest(),
            )
            != header_raw
        ):
            raise PermissionError("partial-access ledger header changed")
        self.descriptor = os.open(
            self.path,
            os.O_WRONLY
            | os.O_APPEND
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(self.descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(header_raw):
            os.close(self.descriptor)
            raise PermissionError("partial-access ledger creation changed")

    @staticmethod
    def header_value(
        *,
        reservation: Mapping[str, Any],
        reservation_raw: bytes,
    ) -> dict[str, Any]:
        reservation_binding = _binding(
            "reservation.json", reservation, reservation_raw
        )
        return contract.with_content_sha256({
            "schema": contract.PARTIAL_ACCESS_RECORD_SCHEMA,
            "sequence": 0,
            "previous_record_content_sha256": None,
            "record_type": "LEDGER_OPENED",
            "attempt_identity": reservation["attempt_identity"],
            "reservation": reservation_binding,
        })

    def _record_value(self, core: Mapping[str, Any]) -> dict[str, Any]:
        return contract.with_content_sha256({
            "schema": contract.PARTIAL_ACCESS_RECORD_SCHEMA,
            "sequence": len(self.records),
            "previous_record_content_sha256":
                self.previous_content_sha256,
            **dict(core),
        })

    def _append(self, core: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("partial-access ledger is closed")
        value = self._record_value(core)
        raw = contract.canonical_json_bytes(value) + b"\n"
        view = memoryview(raw)
        while view:
            written = os.write(self.descriptor, view)
            if written <= 0:
                raise OSError("partial-access ledger append made no progress")
            view = view[written:]
        os.fsync(self.descriptor)
        self.records.append(value)
        self.raw_parts.append(raw)
        self.previous_content_sha256 = str(value["content_sha256"])
        return value

    @staticmethod
    def _error(value: BaseException) -> dict[str, str]:
        message = str(value)
        return {
            "type": type(value).__name__,
            "message": message,
            "message_sha256":
                hashlib.sha256(message.encode("utf-8")).hexdigest(),
        }

    def read_regular(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_byte_count: int | None = None,
        content_sha256: str | None = None,
        kind: str,
        stage: str,
        role: str | None,
        purpose: str,
    ) -> bytes:
        if not contract.is_sha256(expected_sha256):
            raise ValueError("ledgered input hash is malformed")
        if (
            expected_byte_count is not None
            and (
                type(expected_byte_count) is not int
                or expected_byte_count <= 0
            )
        ):
            raise ValueError("ledgered input byte count is malformed")
        if content_sha256 is not None and not contract.is_sha256(content_sha256):
            raise ValueError("ledgered input content hash is malformed")
        lexical_path = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = lexical_path.relative_to(
                self.repository_root
            ).as_posix()
        except ValueError as error:
            raise PermissionError("runtime input escaped repository root") from error
        contract.safe_relative_path(relative, name="ledgered runtime input")
        if (
            any(
                relative == root or relative.startswith(root + "/")
                for root in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
            )
        ):
            raise PermissionError(
                "rejected or prior runtime output access is forbidden"
            )
        resolved_parent = path.parent.resolve(strict=True)
        resolved_path = resolved_parent / path.name
        if lexical_path != resolved_path:
            raise PermissionError(
                "runtime input path is noncanonical or has a symlinked parent"
            )
        open_id = self.next_open_id
        self.next_open_id += 1
        expected_binding = {
            "path": relative,
            "file_sha256": expected_sha256,
            "content_sha256": content_sha256,
            "byte_count": expected_byte_count,
        }
        self._append({
            "record_type": "OPEN_ATTEMPTED",
            "open_id": open_id,
            "stage": stage,
            "kind": kind,
            "role": role,
            "purpose": purpose,
            "expected_binding": expected_binding,
        })

        descriptor: int | None = None
        descriptor_opened = False
        read_completed = False
        raw = b""
        partial_byte_count = 0
        observed_binding: dict[str, Any] | None = None
        try:
            boundary = {
                "bound_schedule": "schedule",
                "n320_gate": "gate",
                "n320_checkpoint": "n320_checkpoint",
                "raw_authority_manifest": "raw_authority",
                "raw_authority_audit": "raw_authority",
                "raw_pairs_index": "raw_indexes",
                "raw_endpoints_index": "raw_indexes",
            }.get(kind)
            if boundary is not None:
                _failure_boundary(boundary)
            before = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise PermissionError(f"runtime input is not regular: {relative}")
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            descriptor_opened = True
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_before.st_mode)
                or _fingerprint(before) != _fingerprint(opened_before)
            ):
                raise PermissionError(
                    f"runtime input identity changed before read: {relative}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                partial_byte_count += len(chunk)
            opened_after = os.fstat(descriptor)
            after = path.stat(follow_symlinks=False)
            if not (
                _fingerprint(before)
                == _fingerprint(opened_before)
                == _fingerprint(opened_after)
                == _fingerprint(after)
            ):
                raise RuntimeError(
                    f"runtime input changed while read: {relative}"
                )
            raw = b"".join(chunks)
            if len(raw) != partial_byte_count:
                raise RuntimeError("runtime input partial-byte accounting changed")
            read_completed = True
            observed_binding = {
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
            if (
                observed_binding["file_sha256"] != expected_sha256
                or (
                    expected_byte_count is not None
                    and len(raw) != expected_byte_count
                )
            ):
                raise PermissionError(
                    f"runtime input binding changed: {relative}"
                )
        except BaseException as error:
            self._append({
                "record_type": "OPEN_OUTCOME",
                "open_id": open_id,
                "stage": stage,
                "kind": kind,
                "outcome": (
                    "REJECTED_BINDING"
                    if read_completed
                    else "READ_FAILED"
                    if descriptor_opened
                    else "OPEN_FAILED"
                ),
                "descriptor_opened": descriptor_opened,
                "read_completed": read_completed,
                "binding_accepted": False,
                "observed_binding": observed_binding,
                "partial_byte_count": partial_byte_count,
                "error": self._error(error),
            })
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self._append({
            "record_type": "OPEN_OUTCOME",
            "open_id": open_id,
            "stage": stage,
            "kind": kind,
            "outcome": "ACCEPTED",
            "descriptor_opened": True,
            "read_completed": True,
            "binding_accepted": True,
            "observed_binding": observed_binding,
            "partial_byte_count": partial_byte_count,
            "error": None,
        })
        return raw

    def append_terminal(
        self,
        *,
        record_type: str,
        stage: Mapping[str, Any],
        operation_counts: Mapping[str, Any],
        error: BaseException | None,
    ) -> None:
        self._append({
            "record_type": record_type,
            "stage": dict(stage),
            "operation_counts": dict(operation_counts),
            "error": None if error is None else self._error(error),
        })

    def close(self) -> None:
        if self.closed:
            return
        os.fsync(self.descriptor)
        os.close(self.descriptor)
        self.closed = True

    def runtime_opens(self) -> list[dict[str, Any]]:
        attempted: dict[int, dict[str, Any]] = {}
        outcomes: dict[int, dict[str, Any]] = {}
        for record in self.records:
            record_type = record["record_type"]
            if record_type == "OPEN_ATTEMPTED":
                attempted[int(record["open_id"])] = record
            elif record_type == "OPEN_OUTCOME":
                outcomes[int(record["open_id"])] = record
        if set(attempted) != set(outcomes):
            raise RuntimeError("partial-access ledger has an unpaired open")
        result: list[dict[str, Any]] = []
        for open_id in sorted(attempted):
            attempt = attempted[open_id]
            outcome = outcomes[open_id]
            result.append({
                "open_id": open_id,
                "stage": attempt["stage"],
                "kind": attempt["kind"],
                "role": attempt["role"],
                "purpose": attempt["purpose"],
                "expected_binding": attempt["expected_binding"],
                "outcome": outcome["outcome"],
                "descriptor_opened": outcome["descriptor_opened"],
                "read_completed": outcome["read_completed"],
                "binding_accepted": outcome["binding_accepted"],
                "observed_binding": outcome["observed_binding"],
                "partial_byte_count": outcome["partial_byte_count"],
                "error": outcome["error"],
            })
        return result

    def binding(self) -> dict[str, Any]:
        self.close()
        raw = b"".join(self.raw_parts)
        observed = _read_regular(
            self.path, expected_sha256=hashlib.sha256(raw).hexdigest()
        )
        if observed != raw:
            raise PermissionError("partial-access ledger bytes changed")
        opens = self.runtime_opens()
        return {
            "path": self.RELATIVE_PATH,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "records_content_sha256":
                contract.canonical_json_sha256(self.records),
            "record_count": len(self.records),
            "last_record_content_sha256": self.previous_content_sha256,
            "attempted_open_count": len(opens),
            "descriptor_opened_count":
                sum(row["descriptor_opened"] for row in opens),
            "read_completed_count":
                sum(row["read_completed"] for row in opens),
            "accepted_open_count":
                sum(row["binding_accepted"] for row in opens),
            "rejected_or_failed_open_count":
                sum(not row["binding_accepted"] for row in opens),
        }


def _initialize_partial_access_ledger(
    output_root: Path,
    *,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    repository_root: Path = ROOT,
) -> PartialAccessLedger:
    """Publish one durable header, then separately accept its live ledger."""
    header = PartialAccessLedger.header_value(
        reservation=reservation,
        reservation_raw=reservation_raw,
    )
    header_raw = contract.canonical_json_bytes(header) + b"\n"
    header_path = output_root / PartialAccessLedger.RELATIVE_PATH
    try:
        _failure_boundary("ledger_before_header")
        _write_exclusive(
            header_path,
            header_raw,
            mode=0o600,
        )
    except BaseException as error:
        observed_prefix_raw = _read_pre_ledger_prefix(header_path)
        raise _PreLedgerInitializationError(
            boundary=(
                "before_header_publication"
                if observed_prefix_raw is None
                else "during_header_publication_unaccepted_prefix"
            ),
            error=error,
            durable_header_raw=None,
            unaccepted_header_prefix_raw=observed_prefix_raw,
        ) from error
    try:
        _failure_boundary("ledger_after_durable_header")
        return PartialAccessLedger(
            output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            header=header,
            header_raw=header_raw,
            repository_root=repository_root,
        )
    except BaseException as error:
        raise _PreLedgerInitializationError(
            boundary="after_durable_header_before_constructor_acceptance",
            error=error,
            durable_header_raw=header_raw,
        ) from error


def _publish_readonly_atomic(path: Path, raw: bytes) -> None:
    """Publish complete immutable bytes without overwriting any prior path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.publishing")
    _write_exclusive(temporary, raw, mode=0o444)
    try:
        os.link(
            temporary,
            path,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=False,
        )
        os.unlink(temporary)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Leave a mode-0444 publication remnant as evidence if final linking
        # fails.  The enclosing attempt is terminal and cannot be retried.
        raise
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o444:
        raise PermissionError("atomic sidecar did not publish mode 0444")


def _load_authority_pre_reservation(
    review_sha256: str,
    authorization_sha256: str,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, str],
]:
    sources = contract.current_source_bindings(ROOT)
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="source review"),
        expected_sources=sources,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_sha256,
    )
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="execution authorization"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    return review, review_raw, authorization, authorization_raw, sources


def _source_authority_receipt(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "source_binding_count": len(sources),
        "source_bindings_sha256": contract.canonical_json_sha256(sources),
        "source_review": contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=str(review["content_sha256"]),
        ),
        "execution_authorization": contract.artifact_binding(
            contract.AUTHORIZATION_RELATIVE_PATH,
            authorization_raw,
            content_sha256=str(authorization["content_sha256"]),
        ),
        "generated_runtime_input_open_count": 0,
        "torch_imported": False,
    }


def _validate_preflight(
    *,
    expected_sha256: str,
    launcher_source_sha256: str,
    expected_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("probe runner requires python -I -B")
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch was imported before attempt reservation")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("probe runner requires HIP_VISIBLE_DEVICES=0")
    conflicting = [
        name for name in CONFLICTING_ACCELERATOR_ENVIRONMENT if name in os.environ
    ]
    threads = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if conflicting or any(value != "1" for value in threads.values()):
        raise PermissionError("accelerator or native-thread environment changed")
    encoded = os.environ.get(PREFLIGHT_ENVIRONMENT_KEY)
    if type(encoded) is not str:
        raise PermissionError("isolated no-tensor preflight receipt is absent")
    try:
        raw = encoded.encode("ascii") + b"\n"
    except UnicodeEncodeError as error:
        raise PermissionError("preflight receipt is not ASCII") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("preflight receipt file hash changed")
    value = contract.parse_canonical_json(raw, name="hardware preflight receipt")
    fields = {
        "schema",
        "status",
        "launcher_process_id",
        "source_authority",
        "preflight_child_process_id",
        "visible_device_count",
        "visible_device_index",
        "visible_device_name",
        "total_memory_bytes",
        "torch_version",
        "hip_version",
        "tensor_allocation_count",
        "payload_open_count",
        "torch_device_api_call_count",
        "launcher_source_sha256",
        "immediate_exec_required",
        "intervening_gpu_query_count",
        "content_sha256",
    }
    name = value.get("visible_device_name")
    if (
        set(value) != fields
        or value["schema"] != f"{contract.SCHEMA_PREFIX}_hardware_preflight_v1"
        or value["status"] != "PASS_EXACTLY_ONE_VISIBLE_DISCRETE_R9700"
        or value["launcher_process_id"] != os.getpid()
        or value["source_authority"] != dict(expected_source_authority)
        or type(value["preflight_child_process_id"]) is not int
        or value["preflight_child_process_id"] <= 0
        or value["visible_device_count"] != 1
        or value["visible_device_index"] != 0
        or type(name) is not str
        or "r9700" not in name.casefold().replace(" ", "")
        or type(value["total_memory_bytes"]) is not int
        or value["total_memory_bytes"] < 32_000_000_000
        or type(value["torch_version"]) is not str
        or not value["torch_version"]
        or type(value["hip_version"]) is not str
        or not value["hip_version"]
        or value["tensor_allocation_count"] != 0
        or value["payload_open_count"] != 0
        or value["torch_device_api_call_count"] != 3
        or value["launcher_source_sha256"] != launcher_source_sha256
        or value["immediate_exec_required"] is not True
        or value["intervening_gpu_query_count"] != 0
    ):
        raise PermissionError("hardware preflight receipt changed")
    return value


def _reserve(
    output_root: Path,
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("the sole temporal probe attempt is already consumed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    attempt_identity = contract.canonical_json_sha256({
        "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
        "review": review_binding,
        "authorization": authorization_binding,
        "science_contract_sha256":
            contract.canonical_json_sha256(contract.science_contract()),
    })
    reservation_core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_0700_BEFORE_TORCH_OR_RUNTIME_INPUTS",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt_identity,
        "independent_source_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "preflight": dict(preflight),
        "science_contract": contract.science_contract(),
        "lifecycle_contract": contract.lifecycle_contract(),
        "output_root_absent_before_reservation": True,
        "output_root_mode": "0700",
        "torch_imported_before_reservation": False,
        "runtime_input_opened_before_reservation": False,
        "reservation_consumes_attempt": True,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    os.mkdir(output_root, mode=0o700)
    try:
        if (
            stat.S_IMODE(output_root.stat(follow_symlinks=False).st_mode)
            != 0o700
        ):
            raise PermissionError("attempt output root was not reserved mode 0700")
        return _publish_json(output_root / "reservation.json", reservation_core)
    except BaseException as error:
        failure_error: BaseException | None = None
        try:
            _publish_json(output_root / "reservation_failed.json", {
                "schema": contract.FAILURE_SCHEMA,
                "status": "TERMINAL_RESERVATION_COMMIT_FAILURE",
                "stage": "reservation_commit",
                "attempt_identity": attempt_identity,
                "error": {"type": type(error).__name__, "message": str(error)},
                "torch_imported": False,
                "runtime_input_opened": False,
                "retry_authorized": False,
                "authority": dict(contract.DOWNSTREAM_DENIALS),
            })
        except BaseException as terminal_error:
            failure_error = terminal_error
        finally:
            _seal_terminal(output_root)
        if failure_error is not None:
            raise RuntimeError(
                "reservation commit and terminalization both failed"
            ) from failure_error
        raise


def _load_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, Any, Any, Any]:
    """First Torch-capable import point; caller must already own reservation."""
    model_source_sha256 = sources.get(contract.MODEL_RELATIVE_PATH)
    matched_source_sha256 = sources.get(
        contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    )
    if (
        not contract.is_sha256(model_source_sha256)
        or not contract.is_sha256(matched_source_sha256)
    ):
        raise PermissionError("reviewed runtime source binding is incomplete")
    matched_path = ROOT / contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    matched_raw = _read_regular(
        matched_path,
        expected_sha256=matched_source_sha256,
    )
    if not matched_raw:
        raise PermissionError("matched V1 reusable runner is empty")
    matched = _load_path(
        "_lewm_causal_temporal_perception_matched_v1_loader",
        matched_path,
    )
    base_runtime = matched._load_runtime()

    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_multires_temporal_v1
            as temporal,
        )
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail_depth,
        )
    finally:
        sys.path[:] = original_path
    expected_model = ROOT / contract.MODEL_RELATIVE_PATH
    observed_model = Path(temporal.__file__)
    if (
        observed_model.is_symlink()
        or expected_model.is_symlink()
        or observed_model.resolve() != expected_model.resolve()
    ):
        raise PermissionError("imported temporal model source changed")
    _read_regular(
        observed_model,
        expected_sha256=model_source_sha256,
    )
    loss_adapter = SimpleNamespace(
        observable_camera_ray_v4_loss_v4=(
            tail_depth.observable_camera_ray_v4_tail_depth_loss_v4
        )
    )
    runtime = SimpleNamespace(
        **{
            **vars(base_runtime),
            "loss_adapter": loss_adapter,
        }
    )
    schedule_adapter = _load_path(
        "_lewm_causal_temporal_perception_v2_schedule_adapter",
        ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
    )
    return matched, runtime, temporal, schedule_adapter


def _runtime_binding_index(
    authorization: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    runtime_inputs = authorization["runtime_inputs"]
    bindings = (
        runtime_inputs["schedule"],
        runtime_inputs["camera"]["gate"],
        runtime_inputs["camera"]["checkpoint"],
        runtime_inputs["raw"]["manifest"],
        runtime_inputs["raw"]["audit"],
    )
    return {str(binding["path"]): dict(binding) for binding in bindings}


def _runtime_kind(relative: str) -> str:
    fixed = {
        contract.SCHEDULE_RELATIVE_PATH: "bound_schedule",
        contract.N320_GATE_RELATIVE_PATH: "n320_gate",
        contract.N320_CHECKPOINT_RELATIVE_PATH: "n320_checkpoint",
        contract.RAW_MANIFEST_RELATIVE_PATH: "raw_authority_manifest",
        contract.RAW_AUDIT_RELATIVE_PATH: "raw_authority_audit",
        f"{contract.RAW_ROOT_RELATIVE_PATH}/pairs.jsonl": "raw_pairs_index",
        f"{contract.RAW_ROOT_RELATIVE_PATH}/endpoints.jsonl":
            "raw_endpoints_index",
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith(contract.RAW_ROOT_RELATIVE_PATH + "/"):
        return "raw_supervision"
    return "development_rgb"


def _install_ledgered_matched_reader(
    matched: Any,
    *,
    ledger: PartialAccessLedger,
    progress: OperationProgress,
    authorization: Mapping[str, Any],
) -> Any:
    original = matched._read_regular
    fixed_bindings = _runtime_binding_index(authorization)

    def ledgered(
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        if not contract.is_sha256(expected_sha256):
            raise PermissionError("matched loader omitted an input binding")
        relative = (
            path.parent.resolve() / path.name
        ).relative_to(ROOT.resolve()).as_posix()
        binding = fixed_bindings.get(relative, {})
        return ledger.read_regular(
            path,
            expected_sha256=str(expected_sha256),
            expected_byte_count=binding.get("byte_count"),
            content_sha256=binding.get("content_sha256"),
            kind=_runtime_kind(relative),
            stage=progress.stage,
            role=progress.role,
            purpose=(
                "terminal_rehash"
                if "rehash" in progress.stage
                else "runtime_load"
            ),
        )

    matched._read_regular = ledgered
    return original


def _read_bound(
    path: Path,
    binding: Mapping[str, Any],
) -> bytes:
    validated = contract.validate_binding(
        binding,
        path=path.relative_to(ROOT).as_posix(),
    )
    raw = _read_regular(path, expected_sha256=validated["file_sha256"])
    if len(raw) != validated["byte_count"]:
        raise PermissionError(f"bound byte count changed: {path}")
    return raw


def _rehash_deferred_runtime_and_authority(
    *,
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    ledger: PartialAccessLedger,
    progress: OperationProgress,
) -> dict[str, Any]:
    runtime_inputs = authorization["runtime_inputs"]
    deferred = (
        runtime_inputs["camera"]["gate"],
        runtime_inputs["camera"]["checkpoint"],
        runtime_inputs["schedule"],
    )
    runtime_records: list[dict[str, Any]] = []
    for binding in deferred:
        progress.enter("deferred_runtime_rehash", role="authority")
        raw = ledger.read_regular(
            ROOT / binding["path"],
            expected_sha256=str(binding["file_sha256"]),
            expected_byte_count=int(binding["byte_count"]),
            content_sha256=str(binding["content_sha256"]),
            kind=_runtime_kind(str(binding["path"])),
            stage=progress.stage,
            role=progress.role,
            purpose="terminal_rehash",
        )
        runtime_records.append({
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    authority_records: list[dict[str, Any]] = []
    for kind in ("independent_source_review", "execution_authorization"):
        binding = reservation[kind]
        raw = _read_regular(
            ROOT / binding["path"],
            expected_sha256=binding["file_sha256"],
        )
        if len(raw) != binding["byte_count"]:
            raise PermissionError(f"{kind} byte count changed")
        authority_records.append({
            "kind": kind,
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    return {
        "deferred_runtime_records": runtime_records,
        "authority_records": authority_records,
        "all_rehashed": True,
    }


def _load_schedule_phase_a(
    schedule_adapter: Any,
    authorization: Mapping[str, Any],
    ledger: PartialAccessLedger,
    progress: OperationProgress,
) -> Any:
    binding = authorization["runtime_inputs"]["schedule"]
    progress.enter("schedule_phase_a", role="authority")
    raw = ledger.read_regular(
        ROOT / binding["path"],
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        content_sha256=str(binding["content_sha256"]),
        kind="bound_schedule",
        stage=progress.stage,
        role=progress.role,
        purpose="runtime_load",
    )
    return schedule_adapter.validate_bound_schedule_phase_a(
        raw=raw,
        binding=binding,
    )


def _finalize_schedule_train_identity(
    schedule_adapter: Any,
    state: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    progress: OperationProgress,
) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    progress.enter("schedule_train_identity_finalization", role="train")
    indices, binding, record = schedule_adapter.finalize_train_identity(
        state=state,
        ordered_train_pair_ids=[
            str(item["content_sha256"]) for item in train_pairs
        ],
    )
    if (
        type(indices) is not list
        or len(indices) != contract.MAXIMUM_PRESENTATIONS
        or type(binding) is not dict
        or type(record) is not dict
    ):
        raise PermissionError(
            "temporal V1 schedule adapter return contract changed"
        )
    return indices, binding, record


def _state_sha(runtime: Any, state_or_model: Any) -> str:
    state = (
        state_or_model.state_dict()
        if hasattr(state_or_model, "state_dict")
        else state_or_model
    )
    return runtime.model_module.tensor_state_dict_sha256(state)


def _subset_sha(runtime: Any, model: Any, prefixes: Sequence[str]) -> str:
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(tuple(prefixes))
    }
    if not state:
        raise RuntimeError("fixed state subset is empty")
    return _state_sha(runtime, state)


def _receipt_dict(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        observed = to_dict()
    elif is_dataclass(value):
        observed = asdict(value)
    elif type(value) is dict:
        observed = dict(value)
    else:
        raise TypeError("migration receipt is not structured")
    if type(observed) is not dict:
        raise TypeError("migration receipt did not normalize to a dict")
    return observed


def _validate_migration_receipt(
    runtime: Any,
    temporal: Any,
    model: Any,
    fit: Any,
    value: object,
) -> dict[str, Any]:
    receipt = _receipt_dict(value)
    fields = {
        "schema",
        "model_family",
        "base_initialization_seed",
        "decoder_initialization_seed",
        "temporal_initialization_seed",
        "initialization_input_role",
        "n320_checkpoint_file_sha256",
        "n320_checkpoint_content_sha256",
        "fit_model_state_sha256",
        "shared_encoder_state_sha256",
        "pixel_head_state_sha256",
        "ground_head_state_sha256",
        "decoder_state_sha256",
        "temporal_state_sha256",
        "evidence_head_state_sha256",
        "copied_state_keys",
        "copied_state_entry_count",
        "copied_predecessor_dense_decoder_entry_count",
        "copied_temporal_entry_count",
        "temporal_output_projection_exact_zero",
        "canonical_ground_support_exact",
        "hard_sync_count",
        "caller_cpu_rng_restored",
        "rejected_adaptation_checkpoint_open_count",
        "torch_version",
    }
    copied = receipt.get("copied_state_keys")
    expected_copied = sorted((
        *(f"encoder.{name}" for name in model.encoder.state_dict()),
        *(
            f"evidence_head.pixel_head.{name}"
            for name in model.evidence_head.pixel_head.state_dict()
        ),
        *(
            f"evidence_head.ground_head.{name}"
            for name in model.evidence_head.ground_head.state_dict()
        ),
    ))
    if (
        set(receipt) != fields
        or receipt["schema"] != temporal.INITIALIZATION_SCHEMA
        or receipt["model_family"] != temporal.MODEL_FAMILY
        or receipt["base_initialization_seed"]
        != contract.BASE_INITIALIZATION_SEED
        or receipt["decoder_initialization_seed"]
        != contract.DECODER_INITIALIZATION_SEED
        or receipt["temporal_initialization_seed"]
        != contract.TEMPORAL_INITIALIZATION_SEED
        or receipt["initialization_input_role"]
        != "n320_fit_initialization_only"
        or receipt["n320_checkpoint_file_sha256"]
        != contract.RUNTIME_FILE_SHA256[contract.N320_CHECKPOINT_RELATIVE_PATH]
        or receipt["n320_checkpoint_content_sha256"]
        != contract.RUNTIME_CONTENT_SHA256[
            contract.N320_CHECKPOINT_RELATIVE_PATH
        ]
        or receipt["fit_model_state_sha256"] != _state_sha(runtime, fit)
        or receipt["shared_encoder_state_sha256"]
        != _state_sha(runtime, model.encoder)
        or receipt["pixel_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.pixel_head)
        or receipt["ground_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.ground_head)
        or receipt["decoder_state_sha256"]
        != _state_sha(runtime, model.evidence_head.dense_decoder)
        or receipt["temporal_state_sha256"]
        != _state_sha(runtime, model.evidence_head.temporal_residual)
        or receipt["evidence_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head)
        or type(copied) is not list
        or copied != expected_copied
        or len(copied) != 84
        or len(set(copied)) != 84
        or receipt["copied_state_entry_count"] != 84
        or receipt["copied_predecessor_dense_decoder_entry_count"] != 0
        or receipt["copied_temporal_entry_count"] != 0
        or receipt["temporal_output_projection_exact_zero"] is not True
        or int(runtime.torch.count_nonzero(
            model.evidence_head.temporal_residual.output_projection.weight
        ).item()) != 0
        or any(
            "dense_decoder" in name or "temporal_residual" in name
            for name in copied
        )
        or receipt["canonical_ground_support_exact"] is not True
        or receipt["hard_sync_count"] != 1
        or receipt["caller_cpu_rng_restored"] is not True
        or receipt["rejected_adaptation_checkpoint_open_count"] != 0
        or receipt["torch_version"] != str(runtime.torch.__version__)
        or not bool(getattr(model, "_n320_initialization_complete", False))
        or _state_sha(runtime, model.encoder)
        != _state_sha(runtime, model.target_encoder)
    ):
        raise PermissionError("N320 temporal initialization receipt changed")
    return receipt


def _prepare_model(
    runtime: Any,
    temporal: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, list[Any], list[Any], list[Any], dict[str, Any]]:
    caller_rng = runtime.torch.random.get_rng_state().clone()
    model, raw_migration = (
        temporal.SharedObservableCameraRayJepaV5MultiresTemporalV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=contract.RUNTIME_FILE_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
            n320_checkpoint_content_sha256=contract.RUNTIME_CONTENT_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
        )
    )
    if not bool(runtime.torch.equal(
        caller_rng, runtime.torch.random.get_rng_state()
    )):
        raise RuntimeError("N320 model initialization changed caller CPU RNG")
    migration = _validate_migration_receipt(
        runtime, temporal, model, fit, raw_migration
    )
    if (
        getattr(temporal, "MODEL_FAMILY", None)
        != "shared_observable_camera_ray_jepa_v5_multires_temporal_v1"
    ):
        raise PermissionError("temporal model runtime identity changed")
    declared_trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_trainable = {
        name for name, _parameter in model.named_parameters()
        if name.startswith(contract.TRAINABLE_PARAMETER_PREFIXES)
    }
    if declared_trainable != expected_trainable:
        raise PermissionError("constructor trainable partition changed")
    model = model.to(device)
    model.requires_grad_(False)
    groups: dict[str, list[tuple[str, Any]]] = {
        "evidence_head": [],
        "encoder": [],
        "frozen": [],
    }
    for name in model.state_dict():
        contract.parameter_partition(name)
    for name, parameter in model.named_parameters():
        component = contract.parameter_partition(name)
        if component in ("evidence_head", "encoder"):
            parameter.requires_grad_(True)
            groups[component].append((name, parameter))
        else:
            groups["frozen"].append((name, parameter))
    counts = {
        name: sum(parameter.numel() for _, parameter in groups[name])
        for name in ("evidence_head", "encoder")
    }
    tensor_counts = {
        name: len(groups[name]) for name in ("evidence_head", "encoder")
    }
    if (
        counts != contract.EXPECTED_PARAMETER_COUNTS
        or tensor_counts != contract.EXPECTED_PARAMETER_TENSOR_COUNTS
        or not groups["frozen"]
        or any(parameter.requires_grad for _, parameter in groups["frozen"])
    ):
        raise PermissionError("temporal trainable/frozen partition changed")
    names = {
        name: [parameter_name for parameter_name, _ in values]
        for name, values in groups.items()
    }
    partition = {
        "parameter_counts": counts,
        "parameter_tensor_counts": tensor_counts,
        "parameter_names_sha256": {
            name: contract.canonical_json_sha256(values)
            for name, values in names.items()
        },
        "migration": migration,
        "model_runtime_version": contract.MODEL_RUNTIME_VERSION,
        "initial_state_sha256": _state_sha(runtime, model),
    }
    return (
        model,
        [parameter for _, parameter in groups["evidence_head"]],
        [parameter for _, parameter in groups["encoder"]],
        [parameter for _, parameter in groups["frozen"]],
        partition,
    )


def _assert_frozen_grads_none(frozen: Sequence[Any]) -> None:
    if any(parameter.grad is not None for parameter in frozen):
        raise RuntimeError("a frozen parameter acquired a gradient")


def _gradient_group_norm(
    runtime: Any,
    parameters: Sequence[Any],
    group: str,
    *,
    maximum: float | None = None,
) -> float:
    if len(parameters) != contract.EXPECTED_PARAMETER_TENSOR_COUNTS[group]:
        raise RuntimeError(f"{group} gradient tensor count changed")
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError(f"{group} parameter has no gradient")
    if not bool(runtime.torch.stack([
        runtime.torch.isfinite(gradient).all() for gradient in gradients
    ]).all().item()):
        raise FloatingPointError(f"{group} gradient became nonfinite")
    squared = runtime.torch.stack([
        gradient.detach().float().square().sum() for gradient in gradients
    ]).sum()
    norm = math.sqrt(float(squared.detach().cpu()))
    if (
        not math.isfinite(norm)
        or (
            maximum is not None
            and norm > maximum + contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
        )
    ):
        raise RuntimeError(f"{group} gradient norm is invalid")
    return norm


def _camera_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    forward = batch["forward"]
    current, next_frame = model.forward_camera_pair(
        previous_image=forward["current_image"],
        current_image=forward["next_image"],
        previous_camera_origin_body_m=(
            forward["current_camera_origin_body_m"]
        ),
        previous_camera_basis_body_fru=(
            forward["current_camera_basis_body_fru"]
        ),
        previous_ground_plane_z_body_m=(
            forward["current_ground_plane_z_body_m"]
        ),
        current_camera_origin_body_m=(
            forward["next_camera_origin_body_m"]
        ),
        current_camera_basis_body_fru=(
            forward["next_camera_basis_body_fru"]
        ),
        current_ground_plane_z_body_m=(
            forward["next_ground_plane_z_body_m"]
        ),
    )
    overlap = runtime.torch.ones_like(
        current.bev[:, :1], dtype=runtime.torch.bool
    )
    return runtime.model_module.SharedTrainingPairV5(
        current=current,
        next=next_frame,
        predicted_next_bev=next_frame.bev,
        stop_gradient_target_next_bev=next_frame.bev.detach(),
        commanded_warped_current_bev=current.bev,
        commanded_overlap_mask=overlap,
        realized_warped_current_bev=current.bev,
        realized_overlap_mask=overlap,
        jepa=None,
    )


def _visual_only_batch(
    trainer: Any,
    pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: Any,
    *,
    role: str,
    arm: str,
    stage: str,
) -> dict[str, Any]:
    """Materialize only the preregistered RGB, geometry, and supervision."""

    selected = [pairs[index] for index in indices]
    if any(item["dataset_role"] != role for item in selected):
        raise PermissionError("visual-only batch crossed dataset roles")
    current = [
        trainer.inputs.frame(
            str(item["current_endpoint_sha256"]),
            role=role,
            arm=arm,
            stage=stage,
        )
        for item in selected
    ]
    next_ = [
        trainer.inputs.frame(
            str(item["next_endpoint_sha256"]),
            role=role,
            arm=arm,
            stage=stage,
        )
        for item in selected
    ]

    def stack(frames: Sequence[Mapping[str, Any]], name: str) -> Any:
        return trainer.r.torch.stack([item[name] for item in frames]).to(device)

    return {
        "forward": {
            "current_image": stack(current, "image"),
            "next_image": stack(next_, "image"),
            "current_camera_origin_body_m": stack(
                current, "camera_origin"
            ).float(),
            "current_camera_basis_body_fru": stack(
                current, "camera_basis"
            ).float(),
            "current_ground_plane_z_body_m": stack(
                current, "ground"
            ).float(),
            "next_camera_origin_body_m": stack(
                next_, "camera_origin"
            ).float(),
            "next_camera_basis_body_fru": stack(
                next_, "camera_basis"
            ).float(),
            "next_ground_plane_z_body_m": stack(
                next_, "ground"
            ).float(),
        },
        "current_supervision": trainer.supervision(current, device),
        "next_supervision": trainer.supervision(next_, device),
    }


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("probe scalar became nonfinite")
    return result


def _camera_components(loss: Any) -> dict[str, float]:
    result = {"camera_total": _scalar(loss.total)}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update({
            f"{side}_hierarchical_first_hit_nll":
                _scalar(frame.hierarchical_first_hit_nll),
            f"{side}_tail_depth_p95_cvar":
                _scalar(frame.tail_depth_p95_cvar),
            f"{side}_ground_clear_distance_state_balanced_bce":
                _scalar(frame.ground_clear_distance_state_balanced_bce),
            f"{side}_derived_raster_hierarchical_bce":
                _scalar(frame.derived_raster_hierarchical_bce.total),
            f"{side}_derived_raster_cell_nll":
                _scalar(frame.derived_raster_cell_nll),
        })
    return result


def _snapshot(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    frozen_sha256: str,
    initial_state_sha256: str,
    migration: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    state_sha256 = _state_sha(runtime, state)
    frozen_observed = _state_sha(runtime, {
        name: value
        for name, value in state.items()
        if name.startswith(contract.FROZEN_STATE_PREFIXES)
    })
    if frozen_observed != frozen_sha256:
        raise RuntimeError("frozen state changed before snapshot")
    semantic = {
        "schema": contract.SNAPSHOT_SCHEMA,
        "update": update,
        "model_family":
            "shared_observable_camera_ray_jepa_v5_multires_temporal_v1",
        "model_config": model.model_config.to_dict(),
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
        "initial_state_sha256": initial_state_sha256,
        "migration": dict(migration),
        "schedule_prefix_indices_sha256":
            contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[update],
        "development_only": True,
        "resume_authorized": False,
        "runtime_ready": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    content_sha256 = contract.canonical_json_sha256(semantic)
    buffer = io.BytesIO()
    runtime.torch.save({
        **semantic,
        "content_sha256": content_sha256,
        "model_state_dict": state,
    }, buffer)
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
    }


def _selection_temporal_index(
    selection_pairs: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[str]],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    """Build the one reset-safe predecessor map from frozen pair identity."""

    pair_fields = {
        "schema",
        "dataset_role",
        "global_row",
        "scene_id",
        "family",
        "episode_id",
        "env_index",
        "reset_count",
        "source_split",
        "frames_jsonl_sha256",
        "scene_manifest_sha256",
        "primitive",
        "relative_se2_current_frame",
        "current_endpoint_sha256",
        "next_endpoint_sha256",
        "label_shard_path_metadata_only",
        "label_shard_sha256",
        "label_shard_row",
        "sidecar_row_identity_sha256",
        "content_sha256",
    }
    primitives = {
        "arc_left",
        "arc_right",
        "backward",
        "forward_fast",
        "forward_medium",
        "forward_slow",
        "hold",
        "yaw_left",
        "yaw_right",
    }
    current_ids: set[str] = set()
    next_to_pair: dict[str, Mapping[str, Any]] = {}
    family_by_id: dict[str, str] = {}
    context_by_id: dict[
        str, tuple[str, str, int, str, int, str, str]
    ] = {}
    family_by_scene: dict[str, str] = {}
    global_rows: set[int] = set()
    pair_content_ids: set[str] = set()
    mapping_rows: list[dict[str, str]] = []
    for pair in selection_pairs:
        if type(pair) is not dict or set(pair) != pair_fields:
            raise PermissionError("temporal pair schema fields changed")
        if (
            pair.get("schema")
            != "lewm_go2_shared_jepa_v5_raw_supervision_pair_v1"
            or pair.get("dataset_role") != "checkpoint_selection"
        ):
            raise PermissionError("temporal evaluator crossed dataset roles")
        family = pair["family"]
        if family not in contract.FAMILIES:
            raise PermissionError("temporal evaluator saw an unknown family")
        current_id = pair["current_endpoint_sha256"]
        next_id = pair["next_endpoint_sha256"]
        episode_id = pair["episode_id"]
        env_index = pair["env_index"]
        reset_count = pair["reset_count"]
        global_row = pair["global_row"]
        scene_id = pair["scene_id"]
        source_split = pair["source_split"]
        frames_sha256 = pair["frames_jsonl_sha256"]
        scene_manifest_sha256 = pair["scene_manifest_sha256"]
        primitive = pair["primitive"]
        label_path = pair["label_shard_path_metadata_only"]
        label_row = pair["label_shard_row"]
        relative_se2 = pair["relative_se2_current_frame"]
        content_sha256 = pair["content_sha256"]
        if (
            not contract.is_sha256(current_id)
            or not contract.is_sha256(next_id)
            or current_id == next_id
            or type(episode_id) is not str
            or not episode_id
            or type(env_index) is not int
            or env_index < 0
            or type(reset_count) is not int
            or reset_count < 0
            or type(global_row) is not int
            or global_row < 0
            or type(scene_id) is not str
            or not scene_id
            or type(source_split) is not str
            or not source_split
            or not contract.is_sha256(frames_sha256)
            or not contract.is_sha256(scene_manifest_sha256)
            or primitive not in primitives
            or type(label_path) is not str
            or not label_path
            or type(label_row) is not int
            or label_row < 0
            or not contract.is_sha256(pair["label_shard_sha256"])
            or not contract.is_sha256(
                pair["sidecar_row_identity_sha256"]
            )
            or type(relative_se2) is not list
            or len(relative_se2) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in relative_se2
            )
            or not contract.is_sha256(content_sha256)
        ):
            raise PermissionError(
                "temporal pair stream or endpoint identity changed"
            )
        core = dict(pair)
        core.pop("content_sha256")
        if contract.canonical_json_sha256(core) != content_sha256:
            raise PermissionError("temporal pair content identity changed")
        if global_row in global_rows or content_sha256 in pair_content_ids:
            raise PermissionError("temporal pair source identity repeated")
        global_rows.add(global_row)
        pair_content_ids.add(content_sha256)

        stream_context = (
            family,
            scene_id,
            env_index,
            episode_id,
            reset_count,
            frames_sha256,
            scene_manifest_sha256,
        )
        for endpoint_id in (current_id, next_id):
            previous_context = context_by_id.setdefault(
                endpoint_id, stream_context
            )
            if previous_context != stream_context:
                raise PermissionError(
                    "temporal endpoint crossed scene, episode, reset, or stream"
                )
        previous_scene_family = family_by_scene.setdefault(scene_id, family)
        if previous_scene_family != family:
            raise PermissionError("temporal scene crossed families")
        previous = next_to_pair.get(next_id)
        if previous is not None:
            raise PermissionError("selection target has multiple predecessors")
        next_to_pair[next_id] = pair
        current_ids.add(current_id)
        for endpoint_id in (current_id, next_id):
            previous_family = family_by_id.setdefault(endpoint_id, family)
            if previous_family != family:
                raise PermissionError("selection endpoint crossed families")
        mapping_rows.append({
            "family": family,
            "previous_endpoint_sha256": current_id,
            "current_endpoint_sha256": next_id,
            "stream": f"{env_index}:{episode_id}:{reset_count}",
        })

    next_ids = set(next_to_pair)
    all_ids = current_ids | next_ids
    cold_ids = all_ids - next_ids
    both_ids = current_ids & next_ids
    observed = {
        "pairs": len(selection_pairs),
        "unique_endpoints": len(all_ids),
        "warm_endpoints": len(next_ids),
        "cold_endpoints": len(cold_ids),
        "both_roles": len(both_ids),
        "ambiguous_predecessors": 0,
        "scenes": len(family_by_scene),
    }
    expected = {
        key: contract.SELECTION_ROLE_COUNTS[key]
        for key in observed
    }
    if observed != expected:
        raise PermissionError(
            f"temporal selection population changed: {observed!r}"
        )

    ids_by_family = {
        family: sorted(
            endpoint_id
            for endpoint_id, endpoint_family in family_by_id.items()
            if endpoint_family == family
        )
        for family in contract.FAMILIES
    }
    if (
        sum(len(values) for values in ids_by_family.values()) != len(all_ids)
        or any(len(values) < 2 for values in ids_by_family.values())
        or set(family_by_scene.values()) != set(contract.FAMILIES)
    ):
        raise PermissionError("temporal family population changed")
    receipt = {
        **observed,
        "mapping_sha256": contract.canonical_json_sha256(
            sorted(
                mapping_rows,
                key=lambda row: (
                    row["family"],
                    row["current_endpoint_sha256"],
                    row["previous_endpoint_sha256"],
                ),
            )
        ),
        "fixed_lag_seconds": 0.5,
        "fixed_lag_ticks": 5,
        "reset_safe": True,
        "pair_content_identity_verified": True,
        "connected_endpoint_stream_consistency_verified": True,
    }
    return ids_by_family, next_to_pair, receipt


def _slice_batch_dataclass(value: Any, index: slice) -> Any:
    if not is_dataclass(value):
        raise TypeError("warm metric value must be a dataclass")
    values: dict[str, Any] = {}
    for item in fields(value):
        member = getattr(value, item.name)
        values[item.name] = member[index] if hasattr(member, "shape") else member
    return type(value)(**values)


def _temporal_physical_metrics(
    runtime: Any,
    trainer: Any,
    model: Any,
    pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    arm: str,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """Run the unchanged physical math over pair-aware RGB observations."""

    torch = runtime.torch
    correct = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    wrong = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    warm_correct = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    warm_wrong = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    ids_by_family, predecessor_by_target, population = (
        _selection_temporal_index(pairs)
    )
    loss_sum = 0.0
    frame_count = 0
    warm_frame_count = 0

    def packet(
        endpoint_id: str,
        *,
        family: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        current = trainer.inputs.frame(
            endpoint_id,
            role="checkpoint_selection",
            arm=arm,
            stage=stage,
        )
        pair = predecessor_by_target.get(endpoint_id)
        if pair is None:
            return current, current
        if str(pair["family"]) != family:
            raise PermissionError("temporal predecessor crossed families")
        previous = trainer.inputs.frame(
            str(pair["current_endpoint_sha256"]),
            role="checkpoint_selection",
            arm=arm,
            stage=stage,
        )
        return previous, current

    def stack(frames: Sequence[Mapping[str, Any]], name: str) -> Any:
        return torch.stack([item[name] for item in frames]).to(device)

    with torch.no_grad():
        for family, ids in ids_by_family.items():
            wrong_ids = ids[1:] + ids[:1]
            for start in range(0, len(ids), contract.MICROBATCH_SIZE):
                target_ids = ids[
                    start : start + contract.MICROBATCH_SIZE
                ]
                mapped_ids = wrong_ids[
                    start : start + contract.MICROBATCH_SIZE
                ]
                target_packets = [
                    packet(item, family=family) for item in target_ids
                ]
                mapped_packets = [
                    packet(item, family=family) for item in mapped_ids
                ]
                target_previous = [item[0] for item in target_packets]
                target_current = [item[1] for item in target_packets]
                mapped_previous = [item[0] for item in mapped_packets]
                mapped_current = [item[1] for item in mapped_packets]
                history_valid = torch.tensor(
                    [
                        endpoint_id in predecessor_by_target
                        for endpoint_id in target_ids
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                origin = stack(target_current, "camera_origin").float()
                basis = stack(target_current, "camera_basis").float()
                ground = stack(target_current, "ground").float()
                supervision = trainer.supervision(target_current, device)
                targets = runtime.derive_targets(
                    pixel_hit_mask=supervision.pixel_hit_mask,
                    pixel_first_hit_distance_m=(
                        supervision.pixel_first_hit_distance_m
                    ),
                    ground_support_in_frustum=(
                        supervision.ground_support_in_frustum
                    ),
                    ground_support_clear_to_target=(
                        supervision.ground_support_clear_to_target
                    ),
                )
                observations = (
                    (target_previous, target_current),
                    (mapped_previous, mapped_current),
                )
                outputs = []
                for previous_frames, current_frames in observations:
                    online = model.forward_temporal_frame(
                        stack(previous_frames, "image"),
                        stack(current_frames, "image"),
                        origin,
                        basis,
                        ground,
                        history_valid,
                    )
                    soft = runtime.soft_rasterize(
                        online.evidence,
                        camera_origin_body_m=origin,
                        camera_basis_body_fru=basis,
                        pixel_ray_chunk_size=(
                            model.model_config.v4_pixel_ray_chunk_size
                        ),
                    )
                    outputs.append((online, soft))

                for accumulator_set, warm_set, output in zip(
                    (correct, wrong),
                    (warm_correct, warm_wrong),
                    outputs,
                    strict=True,
                ):
                    online, soft = output
                    for scope in ("aggregate", family):
                        accumulator_set[scope].update(
                            raw_output=online.evidence,
                            targets=targets,
                            soft_raster=soft,
                            target_raster_labels=(
                                supervision.target_raster_labels
                            ),
                            families=[family] * len(target_ids),
                        )
                    for index, is_warm in enumerate(
                        history_valid.detach().cpu().tolist()
                    ):
                        if not is_warm:
                            continue
                        selected = slice(index, index + 1)
                        for scope in ("aggregate", family):
                            warm_set[scope].update(
                                raw_output=_slice_batch_dataclass(
                                    online.evidence, selected
                                ),
                                targets=_slice_batch_dataclass(
                                    targets, selected
                                ),
                                soft_raster=_slice_batch_dataclass(
                                    soft, selected
                                ),
                                target_raster_labels=(
                                    supervision.target_raster_labels[selected]
                                ),
                                families=[family],
                            )

                camera = (
                    runtime.loss_adapter.observable_camera_ray_v4_loss_v4(
                        model,
                        trainer._single_frame_pair(outputs[0][0]),
                        supervision,
                        supervision,
                        require_b4=False,
                    )
                )
                loss_sum += float(camera.total.cpu()) * len(target_ids)
                frame_count += len(target_ids)
                warm_frame_count += int(history_valid.sum().item())

    if (
        frame_count != contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        or warm_frame_count
        != contract.SELECTION_ROLE_COUNTS["warm_endpoints"]
    ):
        raise PermissionError("temporal evaluator frame counts changed")
    metrics = {
        scope: trainer._flatten_physical(
            correct[scope].finalize(), wrong[scope].finalize()
        )
        for scope in contract.SCOPES
    }
    warm_metrics = {
        scope: trainer._flatten_physical(
            warm_correct[scope].finalize(),
            warm_wrong[scope].finalize(),
        )
        for scope in contract.SCOPES
    }
    return metrics, warm_metrics, loss_sum / frame_count, population


def _evaluate(
    runtime: Any,
    trainer: Any,
    model: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
    frozen_sha256: str,
) -> dict[str, Any]:
    before = _state_sha(runtime, model)
    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha256:
        raise RuntimeError("frozen state changed before inline evaluation")
    model.eval()
    physical, warm_physical, camera_loss, temporal_population = (
        _temporal_physical_metrics(
            runtime,
            trainer,
            model,
            selection_pairs,
            device,
            arm="causal_temporal_perception_v1",
            stage=f"inline_checkpoint_selection_update_{update}",
        )
    )
    model.train()
    after = _state_sha(runtime, model)
    frozen_after = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    if before != after or frozen_after != frozen_sha256:
        raise RuntimeError("inline evaluation mutated model state")
    evaluation = contract.evaluate_physical_scopes(physical)
    return {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "temporal_population": temporal_population,
        "scopes": physical,
        "warm_scopes_informational_only": warm_physical,
        "aggregate_complete_v4_tail_depth_loss": float(camera_loss),
        "evaluation": evaluation,
        "integrity_pass": True,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "frozen_state_sha256_before_and_after": frozen_sha256,
        "state_mutation_count": 0,
    }


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    continuation = contract.checkpoint_control_decision(
        update=update,
        evaluation=metric["evaluation"],
        integrity_pass=metric["integrity_pass"],
    )
    core = {
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "status": "PUBLISHED_0444_AFTER_INLINE_EVALUATION_BEFORE_CONTROL",
        "update": update,
        "checkpoint": dict(checkpoint),
        "metric": dict(metric),
        "inline_evaluation_count": 1,
        "state_mutation_count": 0,
        "publication_order": [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_sidecar",
            "control_branch",
        ],
        "continuation": continuation,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    value = contract.with_content_sha256(core)
    raw = contract.canonical_json_bytes(value) + b"\n"
    relative = contract.metric_sidecar_relative_path(update)
    _publish_readonly_atomic(output_root / relative, raw)
    contract.validate_metric_sidecar(value, update=update)
    return _binding(relative, value, raw), continuation


def _train(
    runtime: Any,
    trainer: Any,
    model: Any,
    head: Sequence[Any],
    encoder: Sequence[Any],
    frozen: Sequence[Any],
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: Any,
    output_root: Path,
    partition: Mapping[str, Any],
    progress: OperationProgress,
) -> dict[str, Any]:
    frozen_sha256 = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    initial_state_sha256 = _state_sha(runtime, model)
    progress.counts["training_entered"] = True
    progress.enter("training_optimizer_construction", role="train")
    _failure_boundary("training")
    progress.increment("optimizer_construction_attempt_count")
    optimizer = runtime.torch.optim.AdamW(
        [
            {
                "params": list(head),
                "lr": contract.learning_rates(1)[0],
                "group_name": "evidence_head",
            },
            {
                "params": list(encoder),
                "lr": contract.learning_rates(1)[1],
                "group_name": "encoder",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    )
    progress.increment("optimizer_construction_completion_count")
    trace: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for update in range(1, contract.MAXIMUM_UPDATE + 1):
        progress.enter("training_update", update=update, role="train")
        progress.increment("optimizer_update_attempt_count")
        head_lr, encoder_lr = contract.learning_rates(update)
        optimizer.param_groups[0]["lr"] = head_lr
        optimizer.param_groups[1]["lr"] = encoder_lr
        _assert_frozen_grads_none(frozen)
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        update_indices = indices[start : start + contract.EFFECTIVE_BATCH_SIZE]
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise PermissionError("fixed presentation schedule ended early")
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            microbatch_indices = update_indices[
                low : low + contract.MICROBATCH_SIZE
            ]
            progress.enter(
                "training_batch_materialization",
                update=update,
                microbatch=microbatch,
                role="train",
            )
            progress.increment("microbatch_attempt_count")
            progress.increment(
                "pair_index_presentations_attempted",
                len(microbatch_indices),
            )
            batch = _visual_only_batch(
                trainer,
                train_pairs,
                microbatch_indices,
                device,
                role="train",
                arm="causal_temporal_perception_v1",
                stage="camera_gradient",
            )
            progress.increment(
                "pair_index_presentations_materialized",
                len(microbatch_indices),
            )
            progress.enter(
                "training_camera_forward",
                update=update,
                microbatch=microbatch,
                role="train",
            )
            pair = _camera_pair(runtime, model, batch)
            progress.enter(
                "training_camera_objective",
                update=update,
                microbatch=microbatch,
                role="train",
            )
            progress.increment("camera_objective_attempt_count")
            camera = runtime.loss_adapter.observable_camera_ray_v4_loss_v4(
                model,
                pair,
                batch["current_supervision"],
                batch["next_supervision"],
            )
            progress.increment("camera_objective_completion_count")
            if not bool(runtime.torch.isfinite(camera.total).item()):
                raise FloatingPointError("probe backward scalar became nonfinite")
            progress.increment("finite_camera_objective_count")
            progress.enter(
                "training_backward",
                update=update,
                microbatch=microbatch,
                role="train",
            )
            progress.increment("backward_attempt_count")
            (camera.total / contract.MICROBATCHES_PER_UPDATE).backward()
            progress.increment("backward_completion_count")
            for name, value in _camera_components(camera).items():
                sums[name] = (
                    sums.get(name, 0.0)
                    + value / contract.MICROBATCHES_PER_UPDATE
                )
            progress.increment("microbatch_completion_count")
        _assert_frozen_grads_none(frozen)
        head_pre = _gradient_group_norm(runtime, head, "evidence_head")
        encoder_pre = _gradient_group_norm(runtime, encoder, "encoder")
        progress.enter("training_head_clip", update=update, role="train")
        progress.increment("head_clip_attempt_count")
        head_clip = runtime.torch.nn.utils.clip_grad_norm_(head, max_norm=1.0)
        progress.increment("head_clip_completion_count")
        progress.enter("training_encoder_clip", update=update, role="train")
        progress.increment("encoder_clip_attempt_count")
        encoder_clip = runtime.torch.nn.utils.clip_grad_norm_(
            encoder, max_norm=1.0
        )
        progress.increment("encoder_clip_completion_count")
        if (
            not bool(runtime.torch.isfinite(head_clip).item())
            or not bool(runtime.torch.isfinite(encoder_clip).item())
        ):
            raise FloatingPointError("probe clip norm became nonfinite")
        head_post = _gradient_group_norm(
            runtime, head, "evidence_head", maximum=1.0
        )
        encoder_post = _gradient_group_norm(
            runtime, encoder, "encoder", maximum=1.0
        )
        progress.enter("training_optimizer_step", update=update, role="train")
        progress.increment("optimizer_step_attempt_count")
        optimizer.step()
        progress.increment("optimizer_step_completion_count")
        progress.increment("complete_optimizer_updates")
        _assert_frozen_grads_none(frozen)
        trace.append({
            "schema": f"{contract.SCHEMA_PREFIX}_trace_row_v1",
            "update": update,
            "presentation_indices_sha256":
                contract.canonical_json_sha256(list(update_indices)),
            "head_learning_rate": head_lr,
            "encoder_learning_rate": encoder_lr,
            "microbatch_count": contract.MICROBATCHES_PER_UPDATE,
            "camera_objective_count": contract.MICROBATCHES_PER_UPDATE,
            "backward_call_count": contract.MICROBATCHES_PER_UPDATE,
            "optimizer_step_count": update,
            "head_clip_invocation_count": update,
            "encoder_clip_invocation_count": update,
            "head_gradient_norm_before_clip": head_pre,
            "encoder_gradient_norm_before_clip": encoder_pre,
            "head_clip_return_norm": _scalar(head_clip),
            "encoder_clip_return_norm": _scalar(encoder_clip),
            "head_gradient_norm_after_clip": head_post,
            "encoder_gradient_norm_after_clip": encoder_post,
            "losses": sums,
            "jepa_objective_count": 0,
            "jepa_backward_count": 0,
            "ema_update_count": 0,
        })
        if update not in contract.CHECKPOINT_UPDATES:
            continue
        if _subset_sha(
            runtime, model, contract.FROZEN_STATE_PREFIXES
        ) != frozen_sha256:
            raise RuntimeError("frozen state changed during probe training")
        progress.enter(
            "checkpoint_snapshot",
            update=update,
            checkpoint_update=update,
            role="checkpoint_selection",
        )
        snapshot = _snapshot(
            runtime,
            model,
            output_root,
            update=update,
            frozen_sha256=frozen_sha256,
            initial_state_sha256=initial_state_sha256,
            migration=partition["migration"],
        )
        progress.increment("checkpoint_snapshot_completion_count")
        snapshots.append(snapshot)
        progress.enter(
            "checkpoint_selection_evaluation",
            update=update,
            checkpoint_update=update,
            role="checkpoint_selection",
        )
        progress.increment("checkpoint_selection_evaluation_attempt_count")
        progress.counts[
            "checkpoint_selection_evaluation_updates_attempted"
        ].append(update)
        _failure_boundary("evaluation")
        metric = _evaluate(
            runtime,
            trainer,
            model,
            selection_pairs,
            device,
            update=update,
            frozen_sha256=frozen_sha256,
        )
        progress.increment("checkpoint_selection_evaluation_completion_count")
        progress.counts[
            "checkpoint_selection_evaluation_updates_completed"
        ].append(update)
        metrics.append(metric)
        progress.enter(
            "checkpoint_metric_sidecar_publication",
            update=update,
            checkpoint_update=update,
            role="checkpoint_selection",
        )
        sidecar, control = _publish_metric_sidecar(
            output_root,
            update=update,
            checkpoint=snapshot,
            metric=metric,
        )
        progress.increment("metric_sidecar_publication_count")
        sidecars.append(sidecar)
        # The branch occurs only after the immutable sidecar is visible.
        controls.append(control)
        if update in (100, 400):
            if control["action"] != contract.CONTROL_CONTINUE:
                raise RuntimeError("informational checkpoint stopped the probe")
        elif control["action"] not in (
            contract.CONTROL_PASS,
            contract.CONTROL_FAIL,
        ):
            raise RuntimeError("terminal probe control is invalid")
    if [row["update"] for row in metrics] != list(contract.CHECKPOINT_UPDATES):
        raise RuntimeError("probe did not evaluate the exact checkpoint set")
    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha256:
        raise RuntimeError("frozen state changed at probe terminal")
    return {
        "trace": trace,
        "metrics": metrics,
        "snapshots": snapshots,
        "sidecars": sidecars,
        "controls": controls,
        "terminal_control": controls[-1],
        "frozen_state_sha256": frozen_sha256,
        "final_state_sha256": _state_sha(runtime, model),
        "operation_counts": contract.operation_counts(
            contract.MAXIMUM_UPDATE, contract.CHECKPOINT_UPDATES
        ),
        "partial_operation_counts": progress.snapshot(),
    }


def _publish_training_records(
    output_root: Path,
    training: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = training["trace"]
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {
        "path": "training_trace.jsonl",
        "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
        "content_sha256": contract.canonical_json_sha256(trace),
        "byte_count": len(trace_raw),
        "row_count": len(trace),
    }
    value, raw = _publish_json(output_root / "checkpoint_metrics.json", {
        "schema": f"{contract.SCHEMA_PREFIX}_checkpoint_metrics_v1",
        "status": "COLLATED_FROM_THREE_IMMUTABLE_INLINE_SIDECARS",
        "checkpoint_updates": list(contract.CHECKPOINT_UPDATES),
        "rows": list(training["metrics"]),
        "sidecars": list(training["sidecars"]),
        "controls": list(training["controls"]),
        "inline_evaluation_count": 3,
        "observer_evaluation_rerun_count": 0,
        "threshold_equality_passes": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    })
    return trace_binding, _binding("checkpoint_metrics.json", value, raw)


def _terminal_inventory(
    output_root: Path,
    *,
    exclude: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise PermissionError("terminal output contains a symlink")
    excluded = set(exclude)
    files = sorted(
        item.relative_to(output_root).as_posix()
        for item in entries
        if item.is_file()
        and item.relative_to(output_root).as_posix() not in excluded
    )
    directories = [
        ".",
        *sorted(
            item.relative_to(output_root).as_posix()
            for item in entries
            if item.is_dir()
        ),
    ]
    return files, directories


def _terminal_file_bindings(
    output_root: Path,
    *,
    exclude: Sequence[str] = (),
) -> list[dict[str, Any]]:
    files, _directories = _terminal_inventory(output_root, exclude=exclude)
    result: list[dict[str, Any]] = []
    for relative in files:
        path = output_root / relative
        raw = (
            _read_pre_ledger_prefix(path)
            if relative == PartialAccessLedger.RELATIVE_PATH
            else _read_regular(path)
        )
        if raw is None:
            raise PermissionError("terminal file disappeared while binding")
        result.append({
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "mode": f"{stat.S_IMODE(path.stat(follow_symlinks=False).st_mode):04o}",
        })
    return result


def _seal_terminal(output_root: Path) -> dict[str, Any]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise PermissionError("cannot seal a symlinked terminal")
    for path in (item for item in entries if item.is_file()):
        os.chmod(path, 0o444, follow_symlinks=False)
    directories = sorted(
        (item for item in entries if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        os.chmod(path, 0o555, follow_symlinks=False)
    os.chmod(output_root, 0o555, follow_symlinks=False)
    files, observed_directories = _terminal_inventory(output_root)
    if any(
        stat.S_IMODE((output_root / relative).stat().st_mode) != 0o444
        for relative in files
    ):
        raise PermissionError("terminal file sealing failed")
    if any(
        stat.S_IMODE(
            (output_root if relative == "." else output_root / relative).stat().st_mode
        )
        != 0o555
        for relative in observed_directories
    ):
        raise PermissionError("terminal directory sealing failed")
    return {
        "files": files,
        "directories_including_root": observed_directories,
        "file_mode": "0444",
        "directory_mode": "0555",
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    ledger: PartialAccessLedger,
    progress: OperationProgress,
    *,
    error: BaseException,
) -> None:
    operation_counts = progress.snapshot()
    try:
        if not ledger.closed:
            ledger.append_terminal(
                record_type="ATTEMPT_TERMINATING",
                stage=progress.location(),
                operation_counts=operation_counts,
                error=error,
            )
        ledger_binding = ledger.binding()
        runtime_opens = ledger.runtime_opens()
        ledger_records = contract.parse_partial_access_ledger(
            b"".join(ledger.raw_parts)
        )
        if (
            ledger_binding["records_content_sha256"]
            != contract.canonical_json_sha256(ledger_records)
        ):
            raise PermissionError("partial-access ledger summary changed")
        prefix = _terminal_file_bindings(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        _files, directories = _terminal_inventory(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        core = {
            "schema": contract.FAILURE_SCHEMA,
            "status":
                "TERMINAL_CAUSAL_TEMPORAL_V1_OPERATIONAL_OR_INTEGRITY_"
                "FAILURE_NO_RETRY",
            "attempt_identity": reservation["attempt_identity"],
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "partial_access_ledger": ledger_binding,
            "runtime_opens": runtime_opens,
            "runtime_opens_sha256":
                contract.canonical_json_sha256(runtime_opens),
            "failure_stage": progress.location(),
            "operation_counts": operation_counts,
            "published_prefix": prefix,
            "published_prefix_sha256":
                contract.canonical_json_sha256(prefix),
            "directories_including_root": directories,
            "error": PartialAccessLedger._error(error),
            "scientific_result": None,
            "scientific_result_status":
                "NOT_OBSERVED_TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE",
            "retry_authorized": False,
            "g2_navigation_or_heldout_attempted": False,
            "prior_runtime_output_open_count": 0,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
            "terminalization": {
                "failure_publication": "exclusive_atomic_fsync",
                "terminal_file_mode": "0444",
                "terminal_directory_mode": "0555",
                "seal_after_publication": True,
            },
        }
        value = contract.with_content_sha256(core)
        contract.validate_failure_receipt(
            value,
            reservation_binding=_binding(
                "reservation.json", reservation, reservation_raw
            ),
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _publish_readonly_atomic(output_root / "failed.json", raw)
    finally:
        ledger.close()
        _seal_terminal(output_root)


def _terminal_pre_ledger_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: OperationProgress,
    *,
    failure: _PreLedgerInitializationError,
) -> None:
    """Publish a distinct receipt without claiming a complete live ledger."""
    reservation_binding = _binding(
        "reservation.json", reservation, reservation_raw
    )
    operation_counts = progress.snapshot()
    if operation_counts != contract.empty_partial_operation_counts():
        raise PermissionError("pre-ledger failure performed an operation")
    try:
        header: dict[str, Any] | None = None
        header_prefix: dict[str, Any] | None = None
        ledger_status = "NOT_PUBLISHED"
        if failure.durable_header_raw is not None:
            header_raw = failure.durable_header_raw
            observed = _read_pre_ledger_prefix(
                output_root / PartialAccessLedger.RELATIVE_PATH
            )
            if observed != header_raw:
                raise PermissionError("pre-ledger durable header bytes changed")
            header = contract.validate_pre_ledger_header(
                header_raw,
                reservation_binding=reservation_binding,
                attempt_identity=str(reservation["attempt_identity"]),
            )
            ledger_status = "DURABLE_NOT_CONSTRUCTOR_ACCEPTED"
        elif failure.unaccepted_header_prefix_raw is not None:
            prefix_raw = failure.unaccepted_header_prefix_raw
            observed = _read_pre_ledger_prefix(
                output_root / PartialAccessLedger.RELATIVE_PATH
            )
            if observed != prefix_raw:
                raise PermissionError("pre-ledger header prefix bytes changed")
            expected_header = PartialAccessLedger.header_value(
                reservation=reservation,
                reservation_raw=reservation_raw,
            )
            expected_header_raw = (
                contract.canonical_json_bytes(expected_header) + b"\n"
            )
            header_prefix = {
                "path": PartialAccessLedger.RELATIVE_PATH,
                "file_sha256": hashlib.sha256(prefix_raw).hexdigest(),
                "byte_count": len(prefix_raw),
                "matches_expected_header": prefix_raw == expected_header_raw,
                "constructor_accepted": False,
                "complete_ledger": False,
            }
            ledger_status = "UNACCEPTED_HEADER_PREFIX"
        prefix = _terminal_file_bindings(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        _files, directories = _terminal_inventory(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        core = {
            "schema": contract.PRE_LEDGER_FAILURE_SCHEMA,
            "status":
                "TERMINAL_CAUSAL_TEMPORAL_V1_POST_RESERVATION_PRE_LEDGER_"
                "FAILURE_NO_RETRY",
            "attempt_identity": reservation["attempt_identity"],
            "reservation": reservation_binding,
            "ledger_state": {
                "status": ledger_status,
                "header": header,
                "header_prefix": header_prefix,
                "runtime_input_open_count": 0,
                "standard_ledger_complete": False,
                "standard_failure_validator_applicable": False,
            },
            "failure_stage": {
                "name": "partial_access_ledger_initialization",
                "boundary": failure.boundary,
            },
            "operation_counts": operation_counts,
            "published_prefix": prefix,
            "published_prefix_sha256":
                contract.canonical_json_sha256(prefix),
            "directories_including_root": directories,
            "error": PartialAccessLedger._error(failure.error),
            "scientific_result": None,
            "scientific_result_status":
                "NOT_OBSERVED_TERMINAL_PRE_LEDGER_FAILURE",
            "retry_authorized": False,
            "g2_navigation_or_heldout_attempted": False,
            "prior_runtime_output_open_count": 0,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
            "terminalization": {
                "failure_publication": "exclusive_atomic_fsync",
                "terminal_file_mode": "0444",
                "terminal_directory_mode": "0555",
                "seal_after_publication": True,
            },
        }
        value = contract.with_content_sha256(core)
        contract.validate_pre_ledger_failure_receipt(
            value,
            reservation_binding=reservation_binding,
            attempt_identity=str(reservation["attempt_identity"]),
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _publish_readonly_atomic(output_root / "failed.json", raw)
    finally:
        _seal_terminal(output_root)


def _execute_after_reservation(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    preflight: Mapping[str, Any],
    output_root: Path,
) -> int:
    del preflight
    progress = OperationProgress()
    progress.enter("partial_access_ledger_initialization")
    try:
        ledger = _initialize_partial_access_ledger(
            output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
        )
    except _PreLedgerInitializationError as failure:
        _terminal_pre_ledger_failure(
            output_root,
            reservation,
            reservation_raw,
            progress,
            failure=failure,
        )
        raise failure.error
    matched: Any | None = None
    original_matched_reader: Any | None = None
    try:
        progress.enter("post_reservation_source_and_authority_rehash")
        if contract.current_source_bindings(ROOT) != dict(sources):
            raise PermissionError("reviewed source changed across reservation")
        observed_review = contract.validate_review(
            contract.parse_canonical_json(review_raw, name="source review rehash"),
            expected_sources=sources,
        )
        review_binding = contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=str(observed_review["content_sha256"]),
        )
        observed_authorization = contract.validate_authorization(
            contract.parse_canonical_json(
                authorization_raw, name="execution authorization rehash"
            ),
            review_binding=review_binding,
            reviewer=str(observed_review["reviewer"]),
        )
        if (
            observed_review != dict(review)
            or observed_authorization != dict(authorization)
        ):
            raise PermissionError("authority changed across reservation")

        progress.enter("deferred_torch_and_reusable_v1_loader_import")
        matched, runtime, temporal, schedule_adapter = (
            _load_post_reservation_stack(sources)
        )
        original_matched_reader = _install_ledgered_matched_reader(
            matched,
            ledger=ledger,
            progress=progress,
            authorization=authorization,
        )
        schedule_state = _load_schedule_phase_a(
            schedule_adapter,
            authorization,
            ledger,
            progress,
        )

        runtime_authority = authorization["runtime_inputs"]
        adapted_authorization = {
            "raw": runtime_authority["raw"],
            "camera": runtime_authority["camera"],
        }
        progress.enter("raw_authority_and_index_validation", role="authority")
        inputs = matched.RawInputs(runtime, adapted_authorization)
        for endpoint in inputs.endpoints.values():
            image_path = Path(str(endpoint["image_path_metadata_only"]))
            if image_path.is_absolute():
                try:
                    image_path = image_path.relative_to(ROOT)
                except ValueError as error:
                    raise PermissionError(
                        "development RGB path is outside the repository"
                    ) from error
            relative_image_path = image_path.as_posix()
            matched.contract.safe_relative_path(
                relative_image_path, name="development RGB path"
            )
            endpoint["image_path_metadata_only"] = relative_image_path
        trainer = matched.Trainer(runtime, inputs, output_root, reservation)
        progress.enter("reserved_runtime_device_validation")
        device, hardware = trainer.device()
        if (
            hardware["visible_device_count"] != 1
            or "r9700" not in hardware["name"].casefold().replace(" ", "")
        ):
            raise PermissionError("reserved runtime device differs from preflight")
        train_pairs = inputs.role_pairs("train")
        selection_pairs = inputs.role_pairs("checkpoint_selection")
        _ids, _predecessors, temporal_population = (
            _selection_temporal_index(selection_pairs)
        )
        del _ids, _predecessors
        indices, schedule_binding, schedule_record = (
            _finalize_schedule_train_identity(
                schedule_adapter,
                schedule_state,
                train_pairs,
                progress,
            )
        )
        progress.enter("deferred_n320_direct_reconstruction", role="authority")
        fit, gate, camera_binding = matched._camera_model_after_reservation(
            runtime, adapted_authorization
        )
        progress.enter("model_preparation", role="train")
        _failure_boundary("model_preparation")
        model, head, encoder, frozen, partition = _prepare_model(
            runtime, temporal, fit, device
        )
        partition["temporal_population_pre_model"] = temporal_population
        del fit
        original_from_numpy = runtime.torch.from_numpy

        def from_numpy_with_scalar(value: Any) -> Any:
            if isinstance(value, runtime.np.generic):
                return runtime.torch.as_tensor(value)
            return original_from_numpy(value)

        runtime.torch.from_numpy = from_numpy_with_scalar
        with warnings.catch_warnings(record=True) as determinism_warnings:
            warnings.simplefilter("once")
            runtime.torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                training = _train(
                    runtime,
                    trainer,
                    model,
                    head,
                    encoder,
                    frozen,
                    train_pairs,
                    selection_pairs,
                    indices,
                    device,
                    output_root,
                    partition,
                    progress,
                )
            finally:
                runtime.torch.from_numpy = original_from_numpy
                runtime.torch.use_deterministic_algorithms(
                    True, warn_only=False
                )
        expected_warning_prefix = (
            "grid_sampler_2d_backward_cuda does not have a deterministic "
            "implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'."
        )
        if not determinism_warnings or any(
            item.category is not UserWarning
            or not str(item.message).startswith(expected_warning_prefix)
            for item in determinism_warnings
        ):
            raise RuntimeError(
                "training emitted an unexpected determinism warning set"
            )
        progress.enter("training_record_publication")
        trace_binding, metrics_binding = _publish_training_records(
            output_root, training
        )

        progress.enter("all_consumed_input_rehash", role="authority")
        consumed = inputs.rehash_consumed()
        observed_roles = {
            role for row in consumed["records"] for role in row["roles"]
        }
        if (
            not {"train", "checkpoint_selection"}.issubset(observed_roles)
            or not observed_roles.issubset(
                {"authority", "index", "train", "checkpoint_selection"}
            )
            or contract.current_source_bindings(ROOT) != dict(sources)
        ):
            raise PermissionError("probe consumed an unauthorized role or source")
        final_rehash = _rehash_deferred_runtime_and_authority(
            authorization=authorization,
            reservation=reservation,
            ledger=ledger,
            progress=progress,
        )
        progress.enter("partial_access_ledger_finalization")
        ledger.append_terminal(
            record_type="RUNTIME_INPUT_ACCESS_FINALIZED",
            stage=progress.location(),
            operation_counts=progress.snapshot(),
            error=None,
        )
        ledger_binding = ledger.binding()
        runtime_opens = ledger.runtime_opens()
        progress.enter("access_publication")
        access, access_raw = _publish_json(output_root / "access.json", {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ALL_CONSUMED_DEVELOPMENT_INPUTS_REHASHED",
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "roles_opened": ["train", "checkpoint_selection"],
            "probability_calibration_open_count": 0,
            "n320": {
                "gate_content_sha256": gate["content_sha256"],
                "checkpoint": camera_binding,
                "initialization_only": True,
            },
            "schedule": {
                "binding": schedule_binding,
                "adapter_record": schedule_record,
            },
            "consumed": consumed,
            "partial_access_ledger": ledger_binding,
            "runtime_opens": runtime_opens,
            "runtime_opens_sha256":
                contract.canonical_json_sha256(runtime_opens),
            "deferred_runtime_and_authority_rehash": final_rehash,
            "reviewed_sources": {
                "count": len(sources),
                "bindings": dict(sources),
                "all_rehashed": True,
            },
            "rejected_adaptation_checkpoint_open_count": 0,
            "g2_navigation_or_heldout_open_count": 0,
            "prior_runtime_output_open_count": 0,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        terminal = training["terminal_control"]
        passed = terminal["action"] == contract.CONTROL_PASS
        progress.enter("result_publication")
        _failure_boundary("result_publication")
        result, result_raw = _publish_json(output_root / "result.json", {
            "schema": contract.RESULT_SCHEMA,
            "status": (
                "PASS_BOUNDED_FALSIFICATION_SEPARATE_QUALIFICATION_PREREG_ONLY"
                if passed
                else "FAIL_BOUNDED_FALSIFICATION_MECHANISM_TERMINATED"
            ),
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "access": _binding("access.json", access, access_raw),
            "terminal_control": terminal,
            "snapshots": list(training["snapshots"]),
            "checkpoint_metrics": metrics_binding,
            "training_trace": trace_binding,
            "partition": partition,
            "state": {
                "initial_state_sha256": partition["initial_state_sha256"],
                "frozen_state_sha256": training["frozen_state_sha256"],
                "final_state_sha256": training["final_state_sha256"],
            },
            "operation_counts": training["operation_counts"],
            "partial_operation_counts": progress.snapshot(),
            "probe_pass_authorizes":
                "separate_bounded_perception_qualification_preregistration_only"
                if passed
                else "nothing",
            "checkpoint_qualified": False,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        progress.enter("completion_publication")
        _failure_boundary("completion_publication")
        files, directories = _terminal_inventory(output_root)
        completed, _ = _publish_json(output_root / "completed.json", {
            "schema": contract.COMPLETION_SCHEMA,
            "status": "TERMINAL_PASS" if passed else "TERMINAL_FAIL",
            "attempt_identity": reservation["attempt_identity"],
            "result": _binding("result.json", result, result_raw),
            "terminal_control": terminal,
            "operation_counts": training["operation_counts"],
            "partial_operation_counts": progress.snapshot(),
            "partial_access_ledger": ledger_binding,
            "exact_precompletion_files": files,
            "exact_terminal_files": sorted([*files, "completed.json"]),
            "exact_terminal_directories_including_root": directories,
            "all_inputs_rehashed": True,
            "all_terminal_files_sealed_read_only": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        if completed["operation_counts"] != contract.operation_counts(
            1_000, contract.CHECKPOINT_UPDATES
        ):
            raise RuntimeError("terminal operation counts changed")
        if original_matched_reader is not None:
            matched._read_regular = original_matched_reader
            original_matched_reader = None
        progress.enter("terminal_sealing")
        _seal_terminal(output_root)
        return 0 if passed else 2
    except BaseException as error:
        if matched is not None and original_matched_reader is not None:
            matched._read_regular = original_matched_reader
            original_matched_reader = None
        _terminal_failure(
            output_root,
            reservation,
            reservation_raw,
            ledger,
            progress,
            error=error,
        )
        raise
    finally:
        if matched is not None and original_matched_reader is not None:
            matched._read_regular = original_matched_reader


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
    preflight_file_sha256: str,
) -> int:
    # Immutable pre-reservation order: authority first, preflight second,
    # namespace reservation third.  No call above imports Torch or opens a
    # generated runtime input.
    review, review_raw, authorization, authorization_raw, sources = (
        _load_authority_pre_reservation(
            review_file_sha256,
            authorization_file_sha256,
        )
    )
    preflight = _validate_preflight(
        expected_sha256=preflight_file_sha256,
        launcher_source_sha256=sources[contract.LAUNCHER_RELATIVE_PATH],
        expected_source_authority=_source_authority_receipt(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
        ),
    )
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(
        output_root,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
        preflight=preflight,
    )
    return _execute_after_reservation(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
        reservation=reservation,
        reservation_raw=reservation_raw,
        preflight=preflight,
        output_root=output_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if (
        not args.run
        or not contract.is_sha256(args.review_sha256)
        or not contract.is_sha256(args.authorization_sha256)
        or not contract.is_sha256(args.preflight_sha256)
    ):
        parser.error("--run and all three exact SHA-256 arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
        preflight_file_sha256=args.preflight_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
