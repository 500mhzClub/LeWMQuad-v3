#!/usr/bin/env python3
"""Authority-free, pre-reservation GPU visibility gate for Camera V15.

Importing this module is stdlib-only and performs no runtime enumeration or
filesystem access.  The production CLI owns one fixed external receipt path;
there is deliberately no caller-selected output path.
"""
from __future__ import annotations

import sys

# A natural CLI invocation reaches the repository policy import before dispatch
# can relaunch with ``-I -B``.  Suppress bytecode before that first repo import.
sys.dont_write_bytecode = True

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15 as policy,
)


CANONICAL_RECEIPT_PATH = policy.GPU_VISIBILITY_RECEIPT_PATH
MAX_RECEIPT_BYTES = 1 << 20
ZERO_ACCESS_EVIDENCE = {
    "canonical_data_open_count": 0,
    "rgb_open_or_decode_count": 0,
    "checkpoint_open_count": 0,
    "v14_output_open_count": 0,
    "v15_output_open_count": 0,
    "generated_path_open_count": 0,
    "tensor_allocation_count": 0,
    "kernel_launch_count": 0,
    "model_construction_count": 0,
    "optimizer_construction_count": 0,
    "attempt_reservation_count": 0,
    "repository_mutation_count": 0,
}
NO_AUTHORITY = {
    "scientific_authority": False,
    "attempt_authority": False,
    "data_authority": False,
    "training_authority": False,
    "checkpoint_authority": False,
    "metric_authority": False,
    "later_rung_authority": False,
    "navigation_authority": False,
    "production_authority": False,
}


class VisibilityGateError(PermissionError):
    """A pre-reservation visibility predicate failed closed."""

    def __init__(self, disposition: str, reason_code: str) -> None:
        if disposition not in policy.GPU_VISIBILITY_DISPOSITIONS:
            raise ValueError("unknown V15 GPU visibility disposition")
        self.disposition = disposition
        self.reason_code = reason_code
        super().__init__(f"{disposition}: {reason_code}")


def _expect_plain_dict(
    value: object,
    expected_keys: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected_keys:
        raise PermissionError(f"V15 GPU visibility {name} schema changed")
    return value


def _exception_class(error: BaseException) -> str:
    value = type(error).__name__
    return value if value.isidentifier() else "RuntimeFailure"


def _selector_observation(environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES"),
        "conflicting_selectors": {
            name: environment.get(name)
            for name in policy.GPU_VISIBILITY_UNSET_SELECTORS
        },
        "hsa_override_gfx_version": environment.get("HSA_OVERRIDE_GFX_VERSION"),
    }


def _native_environment_observation(
    environment: Mapping[str, str],
) -> dict[str, str | None]:
    return {
        name: environment.get(name)
        for name in policy.GPU_VISIBILITY_THREAD_ENVIRONMENT
    }


def _selectors_match(value: Mapping[str, Any]) -> bool:
    conflicting = value.get("conflicting_selectors")
    return (
        value.get("hip_visible_devices") == "0"
        and type(conflicting) is dict
        and set(conflicting) == set(policy.GPU_VISIBILITY_UNSET_SELECTORS)
        and all(item is None for item in conflicting.values())
        and value.get("hsa_override_gfx_version") is None
    )


def _native_environment_matches(value: Mapping[str, Any]) -> bool:
    return (
        type(value) is dict
        and set(value) == set(policy.GPU_VISIBILITY_THREAD_ENVIRONMENT)
        and all(item == "1" for item in value.values())
    )


def _base_observation(environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "disposition": "gpu_runtime_unavailable",
        "reason_code": "unclassified_runtime_failure",
        "sanitized_exception_class": None,
        "selector_observation": _selector_observation(environment),
        "native_thread_observation": {
            "environment": _native_environment_observation(environment),
            "torch_intra_op": None,
            "torch_inter_op": None,
        },
        "runtime_observation": {
            "runtime_available": None,
            "enumeration_completed": False,
            "visible_device_count": None,
            "ordered_devices": None,
            "gpu1_absent": None,
            "raphael_absent": None,
        },
    }


def observe_visibility(
    *,
    environment: Mapping[str, str] | None = None,
    torch_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Classify selectors, threads and read-only runtime enumeration.

    The function calls only PyTorch configuration/introspection and CUDA
    availability/count/name APIs.  It never constructs a tensor, model,
    optimizer or experiment input.
    """

    selected_environment = os.environ if environment is None else environment
    observation = _base_observation(selected_environment)
    selectors = observation["selector_observation"]
    native = observation["native_thread_observation"]
    if not _selectors_match(selectors):
        observation.update(
            disposition="gpu_selector_mismatch",
            reason_code="frozen_accelerator_selector_contract_changed",
        )
        return observation
    if not _native_environment_matches(native["environment"]):
        observation.update(
            disposition="native_thread_mismatch",
            reason_code="frozen_native_thread_environment_changed",
        )
        return observation

    loader = (
        (lambda: importlib.import_module("torch"))
        if torch_loader is None
        else torch_loader
    )
    try:
        torch_module = loader()
    except BaseException as error:
        observation.update(
            disposition="gpu_runtime_unavailable",
            reason_code="torch_runtime_import_failed",
            sanitized_exception_class=_exception_class(error),
        )
        return observation

    try:
        torch_module.set_num_threads(1)
        torch_module.set_num_interop_threads(1)
        native["torch_intra_op"] = int(torch_module.get_num_threads())
        native["torch_inter_op"] = int(torch_module.get_num_interop_threads())
    except BaseException as error:
        observation.update(
            disposition="native_thread_mismatch",
            reason_code="torch_thread_configuration_or_observation_failed",
            sanitized_exception_class=_exception_class(error),
        )
        return observation
    if native["torch_intra_op"] != 1 or native["torch_inter_op"] != 1:
        observation.update(
            disposition="native_thread_mismatch",
            reason_code="frozen_torch_thread_count_changed",
        )
        return observation

    runtime = observation["runtime_observation"]
    try:
        available = torch_module.cuda.is_available()
        if type(available) is not bool:
            raise TypeError("CUDA availability result is not boolean")
        runtime["runtime_available"] = available
    except BaseException as error:
        observation.update(
            disposition="gpu_runtime_unavailable",
            reason_code="gpu_runtime_availability_check_failed",
            sanitized_exception_class=_exception_class(error),
        )
        return observation
    if not runtime["runtime_available"]:
        observation.update(
            disposition="gpu_runtime_unavailable",
            reason_code="gpu_runtime_reported_unavailable",
        )
        return observation

    try:
        visible_count = torch_module.cuda.device_count()
        if type(visible_count) is not int or visible_count < 0:
            raise TypeError("CUDA device count is not a nonnegative integer")
        devices = [
            {
                "logical_ordinal": ordinal,
                "name": torch_module.cuda.get_device_name(ordinal),
            }
            for ordinal in range(visible_count)
        ]
        if any(type(row["name"]) is not str for row in devices):
            raise TypeError("CUDA device name is not a string")
    except BaseException as error:
        observation.update(
            disposition="gpu_runtime_unavailable",
            reason_code="gpu_runtime_enumeration_failed",
            sanitized_exception_class=_exception_class(error),
        )
        return observation

    runtime.update(
        enumeration_completed=True,
        visible_device_count=visible_count,
        ordered_devices=devices,
        gpu1_absent=visible_count == 1,
        raphael_absent=not any(
            "raphael" in row["name"].casefold() for row in devices
        ),
    )
    if visible_count != 1:
        observation.update(
            disposition="gpu_device_count_mismatch",
            reason_code="visible_device_count_is_not_exactly_one",
        )
        return observation
    if (
        devices
        != [
            {
                "logical_ordinal": policy.EXPECTED_GPU_LOGICAL_ORDINAL,
                "name": policy.EXPECTED_GPU_DEVICE_NAME,
            }
        ]
        or runtime["raphael_absent"] is not True
    ):
        observation.update(
            disposition="gpu_device_identity_mismatch",
            reason_code="logical_gpu0_identity_changed",
        )
        return observation

    observation.update(
        disposition="pass_exactly_one_r9700",
        reason_code="exact_visibility_predicate_passed",
    )
    return observation


def require_passing_observation(observation: Mapping[str, Any]) -> None:
    disposition = observation.get("disposition")
    reason = observation.get("reason_code")
    if disposition != "pass_exactly_one_r9700":
        raise VisibilityGateError(
            str(disposition),
            str(reason),
        )
    if observation.get("sanitized_exception_class") is not None:
        raise PermissionError("passing V15 visibility observation has an exception")
    expected_runtime = {
        "runtime_available": True,
        "enumeration_completed": True,
        "visible_device_count": 1,
        "ordered_devices": [
            {
                "logical_ordinal": policy.EXPECTED_GPU_LOGICAL_ORDINAL,
                "name": policy.EXPECTED_GPU_DEVICE_NAME,
            }
        ],
        "gpu1_absent": True,
        "raphael_absent": True,
    }
    if (
        not _selectors_match(observation.get("selector_observation", {}))
        or not _native_environment_matches(
            observation.get("native_thread_observation", {}).get("environment", {})
        )
        or observation.get("native_thread_observation", {}).get("torch_intra_op")
        != 1
        or observation.get("native_thread_observation", {}).get("torch_inter_op")
        != 1
        or observation.get("runtime_observation") != expected_runtime
    ):
        raise PermissionError("passing V15 visibility observation fields changed")


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PermissionError("Linux boot-ID source is not a regular file")
        raw = os.read(descriptor, 128)
        if os.read(descriptor, 1):
            raise PermissionError("Linux boot-ID source is oversized")
    finally:
        os.close(descriptor)
    value = raw.decode("ascii").strip()
    if len(value) != 36 or value.count("-") != 4:
        raise PermissionError("Linux boot ID is malformed")
    return value


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def source_review_receipt_binding(
    review: Mapping[str, Any],
    file_sha256: str,
) -> dict[str, Any]:
    binding = policy.source_review_binding(review, file_sha256)
    return {
        **binding,
        "reviewer": review["reviewer"],
        "status": review["status"],
        "source_closure_approved": review["source_closure_approved"],
        "exact_attempt_authorized": review["exact_attempt_authorized"],
        "successor_sources": dict(review["successor_sources"]),
        "successor_proofs": dict(review["successor_proofs"]),
    }


def _is_git_object_id(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def current_reviewed_git_commit(review: Mapping[str, Any]) -> str:
    """Return HEAD only when it contains the complete reviewed closure."""

    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "HEAD"],
        cwd=policy.ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    lines = completed.stdout.decode("ascii", errors="strict").splitlines()
    if completed.returncode != 0 or len(lines) != 2:
        raise PermissionError("V15 reviewed Git closure is unavailable")
    git_root, commit = lines
    if (
        Path(git_root).resolve() != policy.ROOT.resolve()
        or not _is_git_object_id(commit)
    ):
        raise PermissionError("V15 reviewed Git repository binding changed")
    bindings = {
        **review["successor_sources"],
        **review["successor_proofs"],
    }
    for relative, binding in bindings.items():
        shown = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=policy.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            shown.returncode != 0
            or hashlib.sha256(shown.stdout).hexdigest()
            != binding["file_sha256"]
        ):
            raise PermissionError(
                f"V15 reviewed closure is not contained by Git HEAD: {relative}"
            )
    for relative, expected_sha256 in (
        (
            policy.V15_AMENDMENT_RELATIVE_PATH,
            policy.V15_AMENDMENT_FILE_SHA256,
        ),
        (
            policy.V15_TERMINAL_V14_PROOF_CLARIFICATION_RELATIVE_PATH,
            policy.V15_TERMINAL_V14_PROOF_CLARIFICATION_FILE_SHA256,
        ),
    ):
        authority = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=policy.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            authority.returncode != 0
            or hashlib.sha256(authority.stdout).hexdigest() != expected_sha256
        ):
            raise PermissionError(
                f"V15 source authority is not contained by Git HEAD: {relative}"
            )
    return commit


def build_receipt(
    *,
    observation: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_file_sha256: str,
    repository_commit: str,
    hostname: str | None = None,
    boot_id: str | None = None,
    process_id: int | None = None,
    utc_timestamp: str | None = None,
    monotonic_seconds: float | None = None,
) -> dict[str, Any]:
    disposition = observation.get("disposition")
    if disposition not in policy.GPU_VISIBILITY_DISPOSITIONS:
        raise ValueError("V15 GPU visibility observation disposition changed")
    core = {
        "schema": policy.GPU_VISIBILITY_RECEIPT_SCHEMA,
        "status": "passed" if disposition == "pass_exactly_one_r9700" else "failed",
        "disposition": disposition,
        "authority": dict(NO_AUTHORITY),
        "amendment": {
            "path": policy.V15_AMENDMENT_RELATIVE_PATH,
            "file_sha256": policy.V15_AMENDMENT_FILE_SHA256,
        },
        "source_review": source_review_receipt_binding(
            source_review,
            source_review_file_sha256,
        ),
        "repository": {
            "root": str(policy.ROOT.resolve()),
            "git_commit": repository_commit,
        },
        "selector_observation": dict(observation["selector_observation"]),
        "native_thread_observation": dict(
            observation["native_thread_observation"]
        ),
        "runtime_observation": dict(observation["runtime_observation"]),
        "expected_device": {
            "logical_ordinal": policy.EXPECTED_GPU_LOGICAL_ORDINAL,
            "exact_name": policy.EXPECTED_GPU_DEVICE_NAME,
            "gpu1_absent_required": True,
            "raphael_absent_required": True,
        },
        "host": {
            "hostname": socket.gethostname() if hostname is None else hostname,
            "linux_boot_id": _boot_id() if boot_id is None else boot_id,
            "process_id": os.getpid() if process_id is None else process_id,
            "utc_timestamp": _utc_now() if utc_timestamp is None else utc_timestamp,
            "monotonic_seconds": (
                time.monotonic()
                if monotonic_seconds is None
                else monotonic_seconds
            ),
        },
        "zero_access_evidence": dict(ZERO_ACCESS_EVIDENCE),
        "reason": {
            "code": observation["reason_code"],
            "sanitized_exception_class": observation[
                "sanitized_exception_class"
            ],
        },
    }
    return {**core, "content_sha256": policy.canonical_json_sha256(core)}


def _rename_noreplace(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PermissionError("V15 visibility publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = int(
        renameat2(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(destination_name),
            1,
        )
    )
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination_name)


def publish_receipt(receipt: Mapping[str, Any]) -> tuple[str, str]:
    """Durably publish canonical bytes through a private no-clobber sibling."""

    raw = policy.canonical_json_bytes(receipt) + b"\n"
    destination = CANONICAL_RECEIPT_PATH
    parent = destination.parent
    private_name = (
        f".{destination.name}.private-{os.getpid()}-{secrets.token_hex(16)}"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    private_fd: int | None = None
    published = False
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("canonical V15 visibility receipt already exists")
        private_fd = os.open(private_name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(private_fd, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(private_fd, view)
            if written <= 0:
                raise OSError("short V15 visibility receipt write")
            view = view[written:]
        os.fsync(private_fd)
        metadata = os.fstat(private_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(raw)
        ):
            raise PermissionError("private V15 visibility receipt is insecure")
        os.close(private_fd)
        private_fd = None
        _rename_noreplace(parent_fd, private_name, destination.name)
        published = True
        os.fsync(parent_fd)
    finally:
        if private_fd is not None:
            os.close(private_fd)
        if not published:
            try:
                os.unlink(private_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return hashlib.sha256(raw).hexdigest(), str(receipt["content_sha256"])


def _read_receipt_bytes() -> bytes:
    path = CANONICAL_RECEIPT_PATH
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > MAX_RECEIPT_BYTES
        ):
            raise PermissionError("canonical V15 visibility receipt is insecure")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                raise PermissionError("canonical V15 visibility receipt was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise PermissionError("canonical V15 visibility receipt grew while read")
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(named):
            raise PermissionError("canonical V15 visibility receipt changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_utc(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise PermissionError("V15 visibility UTC timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError("V15 visibility UTC timestamp is malformed") from error
    if parsed.tzinfo != timezone.utc:
        raise PermissionError("V15 visibility UTC timestamp is not UTC")
    return parsed


def validate_receipt_value(
    receipt: object,
    *,
    expected_source_review: Mapping[str, Any],
    expected_source_review_file_sha256: str,
    expected_repository_commit: str,
    hostname: str | None = None,
    boot_id: str | None = None,
    utc_now: datetime | None = None,
    monotonic_now: float | None = None,
) -> dict[str, Any]:
    value = _expect_plain_dict(
        receipt,
        {
            "schema",
            "status",
            "disposition",
            "authority",
            "amendment",
            "source_review",
            "repository",
            "selector_observation",
            "native_thread_observation",
            "runtime_observation",
            "expected_device",
            "host",
            "zero_access_evidence",
            "reason",
            "content_sha256",
        },
        name="receipt",
    )
    core = dict(value)
    declared = core.pop("content_sha256")
    if (
        value["schema"] != policy.GPU_VISIBILITY_RECEIPT_SCHEMA
        or value["status"] != "passed"
        or value["disposition"] != "pass_exactly_one_r9700"
        or not policy.is_sha256(declared)
        or policy.canonical_json_sha256(core) != declared
        or value["authority"] != NO_AUTHORITY
        or value["zero_access_evidence"] != ZERO_ACCESS_EVIDENCE
        or value["amendment"]
        != {
            "path": policy.V15_AMENDMENT_RELATIVE_PATH,
            "file_sha256": policy.V15_AMENDMENT_FILE_SHA256,
        }
        or value["source_review"]
        != source_review_receipt_binding(
            expected_source_review,
            expected_source_review_file_sha256,
        )
        or value["repository"]
        != {
            "root": str(policy.ROOT.resolve()),
            "git_commit": expected_repository_commit,
        }
        or value["expected_device"]
        != {
            "logical_ordinal": policy.EXPECTED_GPU_LOGICAL_ORDINAL,
            "exact_name": policy.EXPECTED_GPU_DEVICE_NAME,
            "gpu1_absent_required": True,
            "raphael_absent_required": True,
        }
        or value["reason"]
        != {
            "code": "exact_visibility_predicate_passed",
            "sanitized_exception_class": None,
        }
    ):
        raise PermissionError("V15 GPU visibility receipt authority fields changed")
    require_passing_observation(
        {
            "disposition": value["disposition"],
            "reason_code": value["reason"]["code"],
            "sanitized_exception_class": value["reason"][
                "sanitized_exception_class"
            ],
            "selector_observation": value["selector_observation"],
            "native_thread_observation": value["native_thread_observation"],
            "runtime_observation": value["runtime_observation"],
        }
    )
    host = _expect_plain_dict(
        value["host"],
        {
            "hostname",
            "linux_boot_id",
            "process_id",
            "utc_timestamp",
            "monotonic_seconds",
        },
        name="host",
    )
    expected_hostname = socket.gethostname() if hostname is None else hostname
    expected_boot = _boot_id() if boot_id is None else boot_id
    if (
        host["hostname"] != expected_hostname
        or host["linux_boot_id"] != expected_boot
        or type(host["process_id"]) is not int
        or host["process_id"] <= 0
        or type(host["monotonic_seconds"]) not in {int, float}
        or isinstance(host["monotonic_seconds"], bool)
    ):
        raise PermissionError("V15 GPU visibility host or boot binding changed")
    receipt_utc = _parse_utc(host["utc_timestamp"])
    current_utc = datetime.now(timezone.utc) if utc_now is None else utc_now
    current_monotonic = time.monotonic() if monotonic_now is None else monotonic_now
    utc_age = (current_utc - receipt_utc).total_seconds()
    monotonic_age = current_monotonic - float(host["monotonic_seconds"])
    if (
        not 0.0 <= utc_age <= policy.GPU_VISIBILITY_MAX_AGE_SECONDS
        or not 0.0 <= monotonic_age <= policy.GPU_VISIBILITY_MAX_AGE_SECONDS
    ):
        raise PermissionError("V15 GPU visibility receipt is stale or future-dated")
    return value


def validate_fixed_receipt(
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
    expected_source_review: Mapping[str, Any],
    expected_source_review_file_sha256: str,
    expected_repository_commit: str,
    **clock_overrides: Any,
) -> dict[str, Any]:
    if not policy.is_sha256(expected_file_sha256) or not policy.is_sha256(
        expected_content_sha256
    ):
        raise ValueError("V15 GPU visibility caller hashes are malformed")
    raw = _read_receipt_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise PermissionError("V15 GPU visibility receipt file hash changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("V15 GPU visibility receipt JSON is malformed") from error
    if raw != policy.canonical_json_bytes(value) + b"\n":
        raise PermissionError("V15 GPU visibility receipt is not canonical JSON plus LF")
    if value.get("content_sha256") != expected_content_sha256:
        raise PermissionError("V15 GPU visibility receipt content hash changed")
    return validate_receipt_value(
        value,
        expected_source_review=expected_source_review,
        expected_source_review_file_sha256=expected_source_review_file_sha256,
        expected_repository_commit=expected_repository_commit,
        **clock_overrides,
    )


def _publication_failure_receipt(
    *,
    source_review: Mapping[str, Any],
    source_review_file_sha256: str,
    repository_commit: str,
    error: BaseException,
) -> dict[str, Any]:
    observation = _base_observation(
        {
            "HIP_VISIBLE_DEVICES": "0",
            **{
                name: "1"
                for name in policy.GPU_VISIBILITY_THREAD_ENVIRONMENT
            },
        }
    )
    observation.update(
        disposition="gpu_visibility_receipt_publication_failure",
        reason_code="canonical_external_receipt_publication_failed",
        sanitized_exception_class=_exception_class(error),
    )
    return build_receipt(
        observation=observation,
        source_review=source_review,
        source_review_file_sha256=source_review_file_sha256,
        repository_commit=repository_commit,
    )


def run_diagnostic(
    source_review_file_sha256: str,
    *,
    publisher: Callable[[Mapping[str, Any]], tuple[str, str]] = publish_receipt,
    torch_loader: Callable[[], Any] | None = None,
    repository_commit: str | None = None,
) -> tuple[dict[str, Any], bool]:
    policy.preflight_v15_source_authority_documents()
    review, _ = policy.preflight_source_review(
        policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_file_sha256,
    )
    commit = (
        current_reviewed_git_commit(review)
        if repository_commit is None
        else repository_commit
    )
    observation = observe_visibility(torch_loader=torch_loader)
    receipt = build_receipt(
        observation=observation,
        source_review=review,
        source_review_file_sha256=source_review_file_sha256,
        repository_commit=commit,
    )
    try:
        publisher(receipt)
    except BaseException as error:
        if observation["disposition"] == "pass_exactly_one_r9700":
            return (
                _publication_failure_receipt(
                    source_review=review,
                    source_review_file_sha256=source_review_file_sha256,
                    repository_commit=commit,
                    error=error,
                ),
                False,
            )
        raise
    return receipt, observation["disposition"] == "pass_exactly_one_r9700"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review-sha256", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if not policy.is_sha256(args.source_review_sha256):
        raise ValueError("V15 source-review SHA-256 is malformed")
    return args


def _isolated_child(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    for name in policy.GPU_VISIBILITY_UNSET_SELECTORS:
        environment.pop(name, None)
    environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
    for name in policy.GPU_VISIBILITY_THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
        cwd=policy.ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def dispatch(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        return _isolated_child(raw_argv)
    args = parse_args(raw_argv)
    receipt, passed = run_diagnostic(args.source_review_sha256)
    print(policy.canonical_json_bytes(receipt).decode("ascii"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(dispatch())


__all__ = [
    "CANONICAL_RECEIPT_PATH",
    "MAX_RECEIPT_BYTES",
    "NO_AUTHORITY",
    "VisibilityGateError",
    "ZERO_ACCESS_EVIDENCE",
    "build_receipt",
    "current_reviewed_git_commit",
    "observe_visibility",
    "publish_receipt",
    "require_passing_observation",
    "run_diagnostic",
    "source_review_receipt_binding",
    "validate_fixed_receipt",
    "validate_receipt_value",
]
