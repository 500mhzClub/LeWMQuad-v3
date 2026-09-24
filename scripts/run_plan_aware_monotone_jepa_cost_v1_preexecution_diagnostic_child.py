#!/usr/bin/env python3
"""Import-safe outer wrapper for the preexecution-only diagnostic child.

This file intentionally imports only the Python standard library.  It opens no
experiment panel, archive payload, checkpoint, tensor, model, metric, or route
outcome.  Its sole purpose is to preserve stderr/traceback and structured
exception custody even when importing the full evaluator fails before its
``main`` function exists.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import faulthandler
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from typing import Any, Mapping


EVALUATOR = Path(__file__).with_name(
    "evaluate_plan_aware_monotone_jepa_cost_v1.py"
).absolute()
REAL_MODE = "diagnose-preexecution-child"
CORRECTION_REAL_MODE = "diagnose-preexecution-correction-1-child"
CORRECTION_FINALIZER_MODE = "finalize-preexecution-forensic-correction-1"
CORRECTION_CHECKER_MODE = "check-preexecution-forensic-correction-1"
SYNTHETIC_MODE = "diagnose-preexecution-synthetic-child"
CORRECTION_DIAGNOSTIC_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_correction_1"
)
CORRECTION_PHASE_MODES = (
    CORRECTION_FINALIZER_MODE,
    CORRECTION_CHECKER_MODE,
)
SYNTHETIC_FIXTURES = (
    "PASS",
    "RAISE",
    "EXIT_NONZERO",
    "SIGTERM",
    "MISSING_PATH",
    "UNICODE_RAISE",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.pop("content_digest", None)
    output["content_digest"] = hashlib.sha256(
        _canonical_json_bytes(output)
    ).hexdigest()
    return output


def _write_preopened_json(fd: int, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    os.write(fd, payload)
    os.fsync(fd)


def _heartbeat(fd: int, event: str) -> None:
    row = _attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "preexecution_wrapper_heartbeat.v1"
            ),
            "sequence": _heartbeat.sequence,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    _heartbeat.sequence += 1
    os.write(fd, _canonical_json_bytes(row) + b"\n")
    os.fsync(fd)


# Sequence zero is durably written by the launcher before ``Popen``.
_heartbeat.sequence = 1


SYNTHETIC_TECHNICAL_STAGE_IDS = (
    "CREATE_TECHNICAL_RESERVATION_DIRECTORY",
    "ACQUIRE_TECHNICAL_LOCK",
    "REGISTER_PROCESS_STATE",
    "ENTER_PREEXECUTION_BOUNDARY",
    "CLEANUP_TECHNICAL_RESOURCES",
)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_last_stage(
    root: Path, *, producer_role: str, stage_id: str, event: str
) -> None:
    value = _attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1.preexecution_last_stage.v1"
            ),
            "producer_role": producer_role,
            "stage_id": stage_id,
            "event": event,
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    path = root / "receipts/last_stage.json"
    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, _canonical_json_bytes(value) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append_synthetic_technical_stage(
    path: Path, *, sequence: int, stage_id: str, event: str
) -> None:
    if stage_id not in SYNTHETIC_TECHNICAL_STAGE_IDS or event not in {
        "STARTED",
        "COMPLETED",
    }:
        raise RuntimeError("synthetic technical lifecycle stage drift")
    row = _attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "preexecution_synthetic_technical_stage.v1"
            ),
            "sequence": sequence,
            "stage_id": stage_id,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    with path.open("ab", buffering=0) as stream:
        stream.write(_canonical_json_bytes(row) + b"\n")
        os.fsync(stream.fileno())


def _prepare_synthetic_technical_resources(root: Path) -> dict[str, Any]:
    """Create synthetic-only reservation/lock/state/PREEXEC resources."""

    lifecycle = root / "receipts/synthetic_technical_lifecycle.jsonl"
    reservation = root / "technical_reservation"
    lock_path = reservation / "technical.lock"
    state_path = reservation / "process_state.json"
    marker_path = reservation / "PREEXECUTION_ONLY.marker"
    sequence = 0

    def stage(stage_id: str, action: Any) -> None:
        nonlocal sequence
        _append_synthetic_technical_stage(
            lifecycle, sequence=sequence, stage_id=stage_id, event="STARTED"
        )
        sequence += 1
        action()
        _append_synthetic_technical_stage(
            lifecycle, sequence=sequence, stage_id=stage_id, event="COMPLETED"
        )
        sequence += 1

    def create_reservation() -> None:
        reservation.mkdir()
        _fsync_directory(reservation.parent)

    stage("CREATE_TECHNICAL_RESERVATION_DIRECTORY", create_reservation)
    lock_stream = lock_path.open("xb", buffering=0)

    def lock() -> None:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fsync(lock_stream.fileno())
        _fsync_directory(reservation)

    stage("ACQUIRE_TECHNICAL_LOCK", lock)

    # _write_preopened_json intentionally does not own the descriptor.  Close
    # it deterministically after the register stage.
    state_fd: list[int] = []

    def register_owned() -> None:
        fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        state_fd.append(fd)
        try:
            _write_preopened_json(
                fd,
                _attach_digest(
                    {
                        "schema": (
                            "plan_aware_monotone_jepa_cost_v1."
                            "preexecution_synthetic_process_state.v1"
                        ),
                        "pid": os.getpid(),
                        "process_group_id": os.getpgrp(),
                        "technical_only": True,
                    }
                ),
            )
        finally:
            os.close(fd)
            state_fd.clear()
        _fsync_directory(reservation)

    stage("REGISTER_PROCESS_STATE", register_owned)

    def mark_preexecution() -> None:
        fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, b"PREEXECUTION_ONLY_DIAGNOSTIC\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(reservation)

    stage("ENTER_PREEXECUTION_BOUNDARY", mark_preexecution)
    return {
        "lifecycle": lifecycle,
        "reservation": reservation,
        "lock_path": lock_path,
        "state_path": state_path,
        "marker_path": marker_path,
        "lock_stream": lock_stream,
        "sequence": sequence,
    }


def _cleanup_synthetic_technical_resources(state: Mapping[str, Any]) -> None:
    lifecycle = Path(state["lifecycle"])
    sequence = int(state["sequence"])
    _append_synthetic_technical_stage(
        lifecycle,
        sequence=sequence,
        stage_id="CLEANUP_TECHNICAL_RESOURCES",
        event="STARTED",
    )
    lock_stream = state["lock_stream"]
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    finally:
        lock_stream.close()
    for key in ("marker_path", "state_path", "lock_path"):
        Path(state[key]).unlink(missing_ok=True)
    Path(state["reservation"]).rmdir()
    _fsync_directory(Path(state["reservation"]).parent)
    _append_synthetic_technical_stage(
        lifecycle,
        sequence=sequence + 1,
        stage_id="CLEANUP_TECHNICAL_RESOURCES",
        event="COMPLETED",
    )


def _load_read_guard_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("diagnostic read-guard manifest is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("diagnostic read-guard manifest must be an object")
    declared = value.get("content_digest")
    core = dict(value)
    core.pop("content_digest", None)
    if declared != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise RuntimeError("diagnostic read-guard manifest digest mismatch")
    required = {
        "schema",
        "experiment_id",
        "forbidden_path_prefixes",
        "scientific_forbidden_bindings",
        "admitted_exact_read_only_technical_receipts",
        "admission_does_not_apply_to_parent_or_sibling_paths",
        "admission_precedence",
        "admitted_open_modes",
        "admitted_write_create_truncate_append_or_update",
        "installed_by_stdlib_wrapper_before_evaluator_runpy",
        "forbidden_open_event_action",
        "expected_event_rows",
        "content_digest",
    }
    if (
        set(value) != required
        or not isinstance(value["forbidden_path_prefixes"], list)
        or not isinstance(value["scientific_forbidden_bindings"], dict)
        or not isinstance(
            value["admitted_exact_read_only_technical_receipts"], list
        )
        or value["admission_does_not_apply_to_parent_or_sibling_paths"] is not True
        or value["admission_precedence"]
        != "EXACT_ADMITTED_PATH_CHECK_BEFORE_FORBIDDEN_PREFIX_CHECK"
        or value["admitted_open_modes"] != ["r", "rb"]
        or value["admitted_write_create_truncate_append_or_update"] is not False
        or value["installed_by_stdlib_wrapper_before_evaluator_runpy"] is not True
        or value["expected_event_rows"] != 0
    ):
        raise RuntimeError("diagnostic read-guard manifest schema drift")
    forbidden = tuple(
        Path(str(item)).absolute().resolve(strict=False)
        for item in value["forbidden_path_prefixes"]
    )
    bindings = value["scientific_forbidden_bindings"]
    if (
        not bindings
        or any(not isinstance(key, str) or not key for key in bindings)
        or any(
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not any(
                Path(path).resolve(strict=False) == root
                or Path(path).resolve(strict=False).is_relative_to(root)
                for root in forbidden
            )
            for path in bindings.values()
        )
    ):
        raise RuntimeError("diagnostic scientific binding coverage drift")
    return value


def _install_read_guard(manifest: Mapping[str, Any], events_fd: int) -> None:
    forbidden = tuple(
        Path(str(value)).absolute().resolve(strict=False)
        for value in manifest["forbidden_path_prefixes"]
    )
    admitted = {
        Path(str(value)).absolute().resolve(strict=False)
        for value in manifest["admitted_exact_read_only_technical_receipts"]
    }
    sequence = {"value": 0}

    def disposition(path: Path, mode: Any, flags: Any) -> str:
        return _read_guard_disposition(
            forbidden=forbidden,
            admitted=admitted,
            path=path,
            mode=mode,
            flags=flags,
        )

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        raw = arguments[0]
        if isinstance(raw, int) or not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(raw)).absolute().resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        decision = disposition(path, mode, flags)
        if decision != "DENY":
            return
        row = _attach_digest(
            {
                "schema": (
                    "plan_aware_monotone_jepa_cost_v1."
                    "preexecution_read_guard_event.v1"
                ),
                "sequence": sequence["value"],
                "event": "FORBIDDEN_OPEN_ATTEMPT",
                "path": str(path),
            }
        )
        sequence["value"] += 1
        os.write(events_fd, _canonical_json_bytes(row) + b"\n")
        os.fsync(events_fd)
        raise PermissionError(
            f"PREEXECUTION_ONLY_DIAGNOSTIC forbids scientific input: {path}"
        )

    sys.addaudithook(audit)


def _read_guard_disposition(
    *,
    forbidden: tuple[Path, ...],
    admitted: set[Path],
    path: Path,
    mode: Any,
    flags: Any,
) -> str:
    """Pure decision core used by the pre-import audit hook and tests."""

    write_mode = isinstance(mode, str) and any(
        marker in mode for marker in ("w", "a", "x", "+")
    )
    write_flags = isinstance(flags, int) and bool(
        flags
        & (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )
    )
    if path in admitted:
        return "DENY" if write_mode or write_flags else "ALLOW_ADMITTED_READ"
    for root in forbidden:
        if path == root:
            return "DENY"
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return "DENY"
    return "ALLOW_UNPROTECTED"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(REAL_MODE, CORRECTION_REAL_MODE, SYNTHETIC_MODE),
        required=True,
    )
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--launcher-start-time-ticks", type=int)
    parser.add_argument("--fixture-id", choices=SYNTHETIC_FIXTURES)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--traceback-fd", type=int, required=True)
    parser.add_argument("--exception-fd", type=int, required=True)
    parser.add_argument("--heartbeat-fd", type=int, required=True)
    parser.add_argument("--read-guard-manifest", type=Path, required=True)
    parser.add_argument("--read-guard-events-fd", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode in {REAL_MODE, CORRECTION_REAL_MODE}:
        if (
            args.launcher_pid is None
            or args.launcher_pid <= 0
            or args.launcher_start_time_ticks is None
            or args.launcher_start_time_ticks <= 0
            or args.fixture_id is not None
        ):
            parser.error("real diagnostic mode requires launcher PID/start only")
    elif (
        args.fixture_id is None
        or args.launcher_pid is not None
        or args.launcher_start_time_ticks is not None
    ):
        parser.error("synthetic mode requires only a fixture ID")
    root = args.diagnostic_root.absolute()
    if any(
        value < 3
        for value in (
            args.traceback_fd,
            args.exception_fd,
            args.heartbeat_fd,
            args.read_guard_events_fd,
        )
    ) or len(
        {
            args.traceback_fd,
            args.exception_fd,
            args.heartbeat_fd,
            args.read_guard_events_fd,
        }
    ) != 4:
        parser.error("diagnostic custody FDs must be distinct inherited descriptors")
    for fd in (
        args.traceback_fd,
        args.exception_fd,
        args.heartbeat_fd,
        args.read_guard_events_fd,
    ):
        try:
            os.fstat(fd)
        except OSError:
            parser.error(f"diagnostic custody FD is not open: {fd}")
    expected_fd_paths = {
        args.traceback_fd: root / "streams/child.traceback",
        args.exception_fd: root / "receipts/child_exception.json",
        args.heartbeat_fd: root / "receipts/heartbeat.jsonl",
        args.read_guard_events_fd: root / "receipts/read_guard_events.jsonl",
    }
    for fd, expected in expected_fd_paths.items():
        try:
            observed = Path(f"/proc/self/fd/{fd}").resolve(strict=True)
        except OSError:
            parser.error(f"diagnostic custody FD cannot be resolved: {fd}")
        if observed != expected:
            parser.error(f"diagnostic custody FD path drift: {fd}")
    args.diagnostic_root = root
    args.read_guard_manifest = args.read_guard_manifest.absolute()
    if args.read_guard_manifest != root / "receipts/read_guard_manifest.json":
        parser.error("diagnostic read-guard manifest path drift")
    return args


def _bootstrap_custody_fds(argv: list[str]) -> dict[str, int]:
    """Recover preopened custody FDs before full argparse can fail."""

    flags = {
        "traceback_fd": "--traceback-fd",
        "exception_fd": "--exception-fd",
        "heartbeat_fd": "--heartbeat-fd",
        "read_guard_events_fd": "--read-guard-events-fd",
    }
    output: dict[str, int] = {}
    for key, flag in flags.items():
        positions = [index for index, value in enumerate(argv) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise RuntimeError(f"bootstrap custody flag is absent or repeated: {flag}")
        raw = argv[positions[0] + 1]
        if not raw.isdecimal() or (len(raw) > 1 and raw.startswith("0")):
            raise RuntimeError(f"bootstrap custody FD is not canonical: {flag}")
        value = int(raw)
        if value < 3:
            raise RuntimeError(f"bootstrap custody FD is reserved: {flag}")
        os.fstat(value)
        output[key] = value
    if len(set(output.values())) != len(output):
        raise RuntimeError("bootstrap custody FDs are not distinct")
    return output


def _bootstrap_diagnostic_root(argv: list[str]) -> Path:
    positions = [
        index for index, value in enumerate(argv) if value == "--diagnostic-root"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("bootstrap diagnostic root is absent or repeated")
    root = Path(argv[positions[0] + 1])
    if not root.is_absolute():
        raise RuntimeError("bootstrap diagnostic root is not absolute")
    return root


def _capture_bootstrap_exception(
    *,
    fds: Mapping[str, int],
    diagnostic_root: Path,
    exc: BaseException,
    traceback_stream: Any,
) -> None:
    traceback.print_exc(file=traceback_stream)
    traceback_stream.flush()
    os.fsync(traceback_stream.fileno())
    record = _attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "preexecution_child_exception.v1"
            ),
            "mode": "WRAPPER_BOOTSTRAP",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback_stream": "streams/child.traceback",
            "structured_capture_available": True,
        }
    )
    _write_preopened_json(int(fds["exception_fd"]), record)
    _heartbeat(int(fds["heartbeat_fd"]), "WRAPPER_ARGPARSE_EXCEPTION")
    _atomic_last_stage(
        diagnostic_root,
        producer_role="PREEXECUTION_DIAGNOSTIC_WRAPPER",
        stage_id="WRAPPER_ARGPARSE_EXCEPTION",
        event="COMPLETED",
    )


def _capture_bootstrap_setup_exception(
    *,
    fds: Mapping[str, int],
    diagnostic_root: Path,
    exc: BaseException,
) -> None:
    """Preserve setup failures without relying on buffered logging objects."""

    traceback_payload = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).encode("utf-8", errors="backslashreplace")
    os.write(int(fds["traceback_fd"]), traceback_payload)
    os.fsync(int(fds["traceback_fd"]))
    record = _attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "preexecution_child_exception.v1"
            ),
            "mode": "WRAPPER_BOOTSTRAP",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback_stream": "streams/child.traceback",
            "structured_capture_available": True,
        }
    )
    _write_preopened_json(int(fds["exception_fd"]), record)
    _heartbeat(int(fds["heartbeat_fd"]), "WRAPPER_ARGPARSE_EXCEPTION")
    _atomic_last_stage(
        diagnostic_root,
        producer_role="PREEXECUTION_DIAGNOSTIC_WRAPPER",
        stage_id="WRAPPER_ARGPARSE_EXCEPTION",
        event="COMPLETED",
    )


def _inner_argv(args: argparse.Namespace) -> list[str]:
    if args.mode in {REAL_MODE, CORRECTION_REAL_MODE}:
        return [
            str(EVALUATOR),
            args.mode,
            "--launcher-pid",
            str(args.launcher_pid),
            "--launcher-start-time-ticks",
            str(args.launcher_start_time_ticks),
            "--diagnostic-root",
            str(args.diagnostic_root),
        ]
    return [
        str(EVALUATOR),
        SYNTHETIC_MODE,
        "--fixture-id",
        str(args.fixture_id),
        "--diagnostic-root",
        str(args.diagnostic_root),
    ]


def _correction_phase_mode(argv: list[str]) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == "--mode"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    value = argv[positions[0] + 1]
    return value if value in CORRECTION_PHASE_MODES else None


def _correction_phase_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=CORRECTION_PHASE_MODES, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.diagnostic_root != CORRECTION_DIAGNOSTIC_ROOT:
        parser.error("correction terminal phase diagnostic root drift")
    return args


def _correction_phase_contract() -> Any:
    repo_root = EVALUATOR.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module(
        "lewm.safety."
        "plan_aware_monotone_jepa_cost_v1_forensic_correction_1_contract"
    )


def _correction_phase_process_identity(correction: Any, *, phase: str) -> dict[str, Any]:
    argv_payload = Path("/proc/self/cmdline").read_bytes()
    argv = [
        item.decode("utf-8", errors="surrogateescape")
        for item in argv_payload.split(b"\0")
        if item
    ]
    stat_payload = Path("/proc/self/stat").read_text(encoding="utf-8")
    try:
        _prefix, suffix = stat_payload.rsplit(")", 1)
        fields = suffix.strip().split()
        process_group_id = int(fields[2])
        start_time_ticks = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("correction terminal phase /proc stat drift") from exc
    role = (
        correction.CORRECTION_FINALIZER_ROLE
        if phase == "FINALIZER"
        else correction.CORRECTION_CHECKER_ROLE
    )
    return correction.BASE.validate_process_identity(
        {
            "pid": os.getpid(),
            "process_group_id": process_group_id,
            "start_time_ticks": start_time_ticks,
            "argv": argv,
            "argv_sha256": correction.BASE.canonical_json_sha256(argv),
            "executable": str(Path("/proc/self/exe").resolve(strict=True)),
            "role": role,
        }
    )


def _correction_phase_coordinator(correction: Any) -> dict[str, Any]:
    raw_pid = os.environ.get("LEWM_CORRECTION_COORDINATOR_PID", "")
    raw_start = os.environ.get(
        "LEWM_CORRECTION_COORDINATOR_START_TIME_TICKS", ""
    )
    if (
        not raw_pid.isdecimal()
        or not raw_start.isdecimal()
        or raw_pid.startswith("0")
        or raw_start.startswith("0")
    ):
        raise RuntimeError("correction terminal coordinator environment drift")
    return correction.require_live_correction_terminal_coordinator(
        pid=int(raw_pid), start_time_ticks=int(raw_start)
    )


def _correction_phase_inner_receipt(
    correction: Any, *, phase: str
) -> dict[str, Any]:
    inner_argv = (
        correction.expected_correction_finalizer_inner_argv()
        if phase == "FINALIZER"
        else correction.expected_correction_checker_inner_argv()
    )
    saved_argv = sys.argv
    capture = io.StringIO()
    try:
        sys.argv = [str(value) for value in inner_argv]
        with contextlib.redirect_stdout(capture):
            try:
                runpy.run_path(str(EVALUATOR), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
    finally:
        sys.argv = saved_argv
    payload = capture.getvalue().encode("utf-8")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("correction phase inner receipt JSON drift") from exc
    if (
        type(value) is not dict
        or payload != correction._authority_bytes(value)
    ):
        raise RuntimeError("correction phase inner receipt canonical-byte drift")
    correction.BASE.validate_self_digest(value)
    return value


def _run_correction_phase_wrapper(argv: list[str]) -> int:
    started_ns = time.monotonic_ns()
    correction: Any | None = None
    phase = "FINALIZER"
    identity: dict[str, Any] | None = None
    coordinator: dict[str, Any] | None = None
    environment_custody: dict[str, Any] | None = None
    heartbeat_started: dict[str, Any] | None = None
    current_stage = "STARTED"
    try:
        args = _correction_phase_arguments(argv)
        phase = "FINALIZER" if args.mode == CORRECTION_FINALIZER_MODE else "CHECKER"
        # Give the root-owned coordinator a deterministic /proc identity window.
        time.sleep(0.15)
        correction = _correction_phase_contract()
        identity = _correction_phase_process_identity(correction, phase=phase)
        coordinator = _correction_phase_coordinator(correction)
        environment_custody = correction.build_correction_phase_environment_custody(
            dict(os.environ)
        )
        heartbeat_started = correction.build_correction_phase_heartbeat_row(
            phase=phase,
            pid=os.getpid(),
            sequence=0,
            event="WRAPPER_STARTED",
            monotonic_ns=time.monotonic_ns(),
        )
        phase_receipt = _correction_phase_inner_receipt(
            correction, phase=phase
        )
        current_stage = "COMPLETE"
        marker = correction.build_correction_phase_last_stage(
            phase=phase,
            pid=os.getpid(),
            stage_id=current_stage,
            event="COMPLETED",
            monotonic_ns=time.monotonic_ns(),
        )
        heartbeat_completed = correction.build_correction_phase_heartbeat_row(
            phase=phase,
            pid=os.getpid(),
            sequence=1,
            event="WRAPPER_COMPLETED",
            monotonic_ns=time.monotonic_ns(),
        )
        ended_ns = time.monotonic_ns()
        envelope = correction.build_correction_phase_wrapper_envelope(
            phase=phase,
            phase_process_identity=identity,
            coordinator_process_identity=coordinator,
            environment_custody=environment_custody,
            observed_cwd=os.getcwd(),
            wrapper_heartbeat=[heartbeat_started, heartbeat_completed],
            started_monotonic_ns=started_ns,
            ended_monotonic_ns=ended_ns,
            last_stage_marker=marker,
            phase_receipt=phase_receipt,
            structured_exception=None,
            envelope_source="STDLIB_PHASE_WRAPPER",
        )
    except BaseException as exc:
        traceback_bytes = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).encode("utf-8", errors="backslashreplace")
        try:
            if any(
                value is None
                for value in (
                    correction,
                    identity,
                    coordinator,
                    environment_custody,
                    heartbeat_started,
                )
            ):
                raise RuntimeError(
                    "correction phase failed before wrapper envelope authority"
                )
            assert correction is not None
            assert identity is not None
            assert coordinator is not None
            assert environment_custody is not None
            assert heartbeat_started is not None
            structured = correction.build_correction_phase_structured_exception(
                phase=phase,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback_bytes=traceback_bytes,
                capture_source="STDLIB_PHASE_WRAPPER",
            )
            marker = correction.build_correction_phase_last_stage(
                phase=phase,
                pid=os.getpid(),
                stage_id=current_stage,
                event="FAILED",
                monotonic_ns=time.monotonic_ns(),
            )
            heartbeat_failed = correction.build_correction_phase_heartbeat_row(
                phase=phase,
                pid=os.getpid(),
                sequence=1,
                event="WRAPPER_FAILED",
                monotonic_ns=time.monotonic_ns(),
            )
            ended_ns = time.monotonic_ns()
            envelope = correction.build_correction_phase_wrapper_envelope(
                phase=phase,
                phase_process_identity=identity,
                coordinator_process_identity=coordinator,
                environment_custody=environment_custody,
                observed_cwd=os.getcwd(),
                wrapper_heartbeat=[heartbeat_started, heartbeat_failed],
                started_monotonic_ns=started_ns,
                ended_monotonic_ns=ended_ns,
                last_stage_marker=marker,
                phase_receipt=None,
                structured_exception=structured,
                envelope_source="STDLIB_PHASE_WRAPPER",
            )
            sys.stdout.buffer.write(correction._authority_bytes(envelope))
            sys.stdout.buffer.flush()
        except BaseException as custody_exc:
            traceback_bytes += "".join(
                traceback.format_exception(
                    type(custody_exc), custody_exc, custody_exc.__traceback__
                )
            ).encode("utf-8", errors="backslashreplace")
        sys.stderr.buffer.write(traceback_bytes)
        sys.stderr.buffer.flush()
        return 1
    sys.stdout.buffer.write(correction._authority_bytes(envelope))
    sys.stdout.buffer.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if _correction_phase_mode(raw_argv) is not None:
        return _run_correction_phase_wrapper(raw_argv)
    bootstrap_fds = _bootstrap_custody_fds(raw_argv)
    bootstrap_root = _bootstrap_diagnostic_root(raw_argv)
    # Hold before any logging/faulthandler setup so even a setup or argparse
    # failure remains live long enough for the parent to bind /proc identity.
    time.sleep(0.15)
    traceback_stream: Any | None = None
    try:
        traceback_stream = os.fdopen(
            os.dup(bootstrap_fds["traceback_fd"]),
            "w",
            encoding="utf-8",
            buffering=1,
        )
        faulthandler.enable(file=traceback_stream, all_threads=True)
        _heartbeat(bootstrap_fds["heartbeat_fd"], "WRAPPER_BOOTSTRAP_STARTED")
        _atomic_last_stage(
            bootstrap_root,
            producer_role="PREEXECUTION_DIAGNOSTIC_WRAPPER",
            stage_id="WRAPPER_BOOTSTRAP_STARTED",
            event="COMPLETED",
        )
    except BaseException as exc:
        _capture_bootstrap_setup_exception(
            fds=bootstrap_fds,
            diagnostic_root=bootstrap_root,
            exc=exc,
        )
        if traceback_stream is not None:
            traceback_stream.close()
        raise
    try:
        args = _arguments(raw_argv)
    except BaseException as exc:
        _capture_bootstrap_exception(
            fds=bootstrap_fds,
            diagnostic_root=bootstrap_root,
            exc=exc,
            traceback_stream=traceback_stream,
        )
        raise
    try:
        read_guard_manifest = _load_read_guard_manifest(args.read_guard_manifest)
        _install_read_guard(read_guard_manifest, args.read_guard_events_fd)
    except BaseException as exc:
        if os.fstat(bootstrap_fds["exception_fd"]).st_size == 0:
            _capture_bootstrap_exception(
                fds=bootstrap_fds,
                diagnostic_root=bootstrap_root,
                exc=exc,
                traceback_stream=traceback_stream,
            )
        raise
    _heartbeat(args.heartbeat_fd, "WRAPPER_STARTED")
    _heartbeat(args.heartbeat_fd, "WRAPPER_ARGPARSE_COMPLETED")
    _atomic_last_stage(
        args.diagnostic_root,
        producer_role="PREEXECUTION_DIAGNOSTIC_WRAPPER",
        stage_id="WRAPPER_ARGPARSE_COMPLETED",
        event="COMPLETED",
    )
    # Give the parent a deterministic window to bind PID/start/argv/executable
    # before any deliberately fast synthetic failure.
    time.sleep(0.15)
    technical_state: dict[str, Any] | None = None
    try:
        if args.mode == SYNTHETIC_MODE:
            technical_state = _prepare_synthetic_technical_resources(
                args.diagnostic_root
            )
        sys.argv = _inner_argv(args)
        evaluator = EVALUATOR
        if args.mode == SYNTHETIC_MODE and args.fixture_id == "MISSING_PATH":
            evaluator = EVALUATOR.with_name("intentionally_missing_evaluator.py")
        _heartbeat(
            args.heartbeat_fd, "BEFORE_EVALUATOR_IMPORT_AND_ARGPARSE_HANDOFF"
        )
        _atomic_last_stage(
            args.diagnostic_root,
            producer_role="PREEXECUTION_DIAGNOSTIC_WRAPPER",
            stage_id="BEFORE_EVALUATOR_IMPORT_AND_ARGPARSE_HANDOFF",
            event="COMPLETED",
        )
        try:
            runpy.run_path(str(evaluator), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code is None or code == 0:
                _heartbeat(args.heartbeat_fd, "WRAPPER_COMPLETED")
                return 0
            record = _attach_digest(
                {
                    "schema": (
                        "plan_aware_monotone_jepa_cost_v1."
                        "preexecution_child_exception.v1"
                    ),
                    "mode": args.mode,
                    "exception_type": "SystemExit",
                    "exception_message": str(code),
                    "traceback_stream": "streams/child.traceback",
                    "structured_capture_available": True,
                }
            )
            _write_preopened_json(args.exception_fd, record)
            traceback.print_exc(file=traceback_stream)
            traceback_stream.flush()
            os.fsync(traceback_stream.fileno())
            _heartbeat(args.heartbeat_fd, "WRAPPER_SYSTEM_EXIT")
            raise
        except BaseException as exc:
            traceback.print_exc(file=traceback_stream)
            traceback_stream.flush()
            os.fsync(traceback_stream.fileno())
            record = _attach_digest(
                {
                    "schema": (
                        "plan_aware_monotone_jepa_cost_v1."
                        "preexecution_child_exception.v1"
                    ),
                    "mode": args.mode,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback_stream": "streams/child.traceback",
                    "structured_capture_available": True,
                }
            )
            _write_preopened_json(args.exception_fd, record)
            _heartbeat(args.heartbeat_fd, "WRAPPER_EXCEPTION")
            raise
    finally:
        if technical_state is not None:
            _cleanup_synthetic_technical_resources(technical_state)
    _heartbeat(args.heartbeat_fd, "WRAPPER_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
