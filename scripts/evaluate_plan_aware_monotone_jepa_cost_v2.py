"""Direct three-phase evaluator for PLAN_AWARE_MONOTONE_JEPA_COST_V2.

The scientific implementation is deliberately composed from the frozen V1
functions.  This module owns only the V2 attempt boundary, prospective
conditional routing, immutable-payload handoff, and the two non-producer
entrypoints.  In particular, it never enters the V1 correction-replay path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lewm.safety import plan_aware_monotone_jepa_cost_v2_contract as V2


class V2EvaluationError(RuntimeError):
    """Raised when execution differs from the prospective V2 authority."""


class V2Phase1WorkerProcessError(V2EvaluationError):
    """Carries a durably persisted failed-worker custody row to its helper."""

    def __init__(
        self,
        message: str,
        *,
        exit_custody: Mapping[str, Any],
        exit_custody_binding: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.exit_custody = dict(exit_custody)
        self.exit_custody_binding = dict(exit_custody_binding)


class V2TerminalAttestorProcessError(V2EvaluationError):
    """Carries the exact failed terminal-attestor lifecycle to its supervisor."""

    def __init__(
        self,
        message: str,
        *,
        failure_stage: str,
        lifecycle: Mapping[str, Any],
        primary_error: BaseException,
        cleanup_errors: Sequence[BaseException] = (),
    ) -> None:
        super().__init__(message)
        self.failure_stage = failure_stage
        self.lifecycle = dict(lifecycle)
        self.primary_error = primary_error
        self.failure_errors = (primary_error, *tuple(cleanup_errors))


_ACTIVE_ATTEMPT_ROOT: Path | None = None
_ACTIVE_ATTEMPT_ROOT_FD: int | None = None
_ACTIVE_ATTEMPT_PARENT_FD: int | None = None
_ACTIVE_ATTEMPT_ROOT_FD_CUSTODY: dict[str, Any] | None = None
_ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY: dict[str, Any] | None = None
_ACTIVE_HELPER_WORKER_PIDS: set[int] = set()
_ACTIVE_HELPER_WORKER_PIDS_LOCK = threading.RLock()
_STATIC_AUDIT_LOCK = threading.RLock()
_STATIC_AUDIT_HOOK_INSTALLED = False
_STATIC_AUDIT_CAPTURE: dict[str, Any] | None = None


_PROHIBITED_REPLAY_SYMBOLS = frozenset(
    {
        "_runtime_execution_correction_2_custody",
        "_publish_execution_correction_2_stage_a_replay_gate",
        "_publish_execution_correction_2_stage_b_replay_gate",
        "execution_correction_2_failed_archive_custody",
    }
)


def _v1() -> Any:
    """Late import: unreachable before the scientific attempt boundary."""

    from scripts import evaluate_plan_aware_monotone_jepa_cost_v1 as module

    return module


def _materializer_v1() -> Any:
    """Late import of frozen V1 numerical/materialisation primitives only."""

    from scripts import materialize_plan_aware_proprio_predictor_substitution_v1 as module

    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=str(V2.EVALUATOR_PATH),
        description="Direct plan-aware monotone JEPA cost V2 evaluator",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser(V2.PUBLIC_EXECUTE_MODE)
    for mode in V2.INTERNAL_MODES:
        child = subparsers.add_parser(mode)
        child.add_argument("--attempt-root", required=True, type=Path)
        if mode == V2.PHASE1_TERMINAL_CUSTODY_MODE:
            child.add_argument(
                "--supervisor-observation-content-digest", required=True
            )
        if mode == V2.PHASE3_PUBLISHER_MODE:
            child.add_argument(
                "--publication-attempt-number", required=True, type=int
            )
            child.add_argument("--publication-attempt-id", required=True)
            child.add_argument(
                "--presentation-retry-authority-content-digest"
            )
    context_worker = subparsers.add_parser(
        V2.PHASE1_CONTEXT_STATE_WORKER_MODE
    )
    context_worker.add_argument("--attempt-root", required=True, type=Path)
    context_worker.add_argument("--state-index", required=True, type=int)
    prediction_worker = subparsers.add_parser(
        V2.PHASE1_PREDICTION_SOURCE_WORKER_MODE
    )
    prediction_worker.add_argument("--attempt-root", required=True, type=Path)
    prediction_worker.add_argument("--source-id", required=True)
    prediction_worker.add_argument("--ablation")
    supervisor = subparsers.add_parser(V2.PHASE1_SUPERVISOR_MODE)
    supervisor.add_argument("--attempt-id", required=True)
    pre_root = subparsers.add_parser(
        V2.PRE_ROOT_PHASE1_FAILURE_CUSTODY_MODE
    )
    pre_root.add_argument("--custody-path", required=True, type=Path)
    pre_root.add_argument(
        "--supervisor-observation-content-digest", required=True
    )
    terminal = subparsers.add_parser(
        V2.TERMINAL_PUBLICATION_SUPERVISOR_MODE
    )
    terminal.add_argument("--attempt-root", required=True, type=Path)
    terminal.add_argument(
        "--resume-boundary",
        required=True,
        choices=V2.TERMINAL_PUBLICATION_RESUME_BOUNDARIES,
    )
    terminal.add_argument(
        "--publication-attempt-number", required=True, type=int
    )
    terminal.add_argument("--publication-attempt-id", required=True)
    terminal.add_argument(
        "--tracked-attempt-number", required=True, type=int
    )
    terminal.add_argument("--tracked-attempt-id", required=True)
    terminal.add_argument(
        "--phase-1-custody-content-digest", required=True
    )
    terminal.add_argument("--phase-2-custody-content-digest")
    terminal.add_argument("--phase-3-custody-content-digest")
    terminal.add_argument(
        "--presentation-retry-authority-content-digest"
    )
    terminal.add_argument("--tracked-retry-authority-content-digest")
    return parser


def _validated_attempt_root(mode: str, value: Path) -> Path:
    root = value.absolute()
    V2.expected_internal_argv(mode, root)
    return root


def _phase_for_mode(mode: str) -> str:
    if mode in (
        V2.PUBLIC_EXECUTE_MODE,
        V2.PHASE1_PROPRIO_HELPER_MODE,
        V2.PHASE1_STAGE_C_HELPER_MODE,
        V2.PHASE1_TERMINAL_CUSTODY_MODE,
    ):
        return V2.PHASE_IDS[0]
    if mode == V2.PHASE2_VALIDATOR_MODE:
        return V2.PHASE_IDS[1]
    if mode == V2.PHASE3_PUBLISHER_MODE:
        return V2.PHASE_IDS[2]
    raise V2EvaluationError(f"unknown V2 mode: {mode}")


def _process_identity(*, expected_executable: Path | None = None) -> dict[str, Any]:
    stat_text = Path("/proc/self/stat").read_text(encoding="utf-8")
    closing = stat_text.rfind(")")
    if closing < 0:
        raise V2EvaluationError("/proc/self/stat is malformed")
    fields = stat_text[closing + 2 :].split()
    if len(fields) < 20:
        raise V2EvaluationError("/proc/self/stat is incomplete")
    argv = [
        item.decode("utf-8", errors="strict")
        for item in Path("/proc/self/cmdline").read_bytes().split(b"\0")
        if item
    ]
    executable_authority = (
        V2.PYTHON_EXECUTABLE
        if expected_executable is None
        else expected_executable
    )
    executable = Path(sys.executable).absolute()
    if executable != executable_authority:
        raise V2EvaluationError("V2 interpreter identity drift")
    return {
        "pid": os.getpid(),
        "start_time_ticks": int(fields[19]),
        "ppid": int(fields[1]),
        "argv": argv,
        "cwd": os.getcwd(),
        "executable": str(executable),
    }


def _process_identity_for_pid(
    pid: int,
    *,
    expected_argv: Sequence[str],
    expected_executable: Path | None = None,
) -> dict[str, Any]:
    executable_authority = (
        V2.PYTHON_EXECUTABLE
        if expected_executable is None
        else expected_executable
    )
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    cwd_path = Path(f"/proc/{pid}/cwd")
    exe_path = Path(f"/proc/{pid}/exe")
    for _ in range(500):
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
            cmdline = [
                item.decode("utf-8", errors="strict")
                for item in cmdline_path.read_bytes().split(b"\0")
                if item
            ]
            cwd = os.readlink(cwd_path)
            executable = os.readlink(exe_path)
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        closing = stat_text.rfind(")")
        fields = stat_text[closing + 2 :].split()
        if closing < 0 or len(fields) < 20:
            raise V2EvaluationError("child /proc identity is malformed")
        if cmdline != list(expected_argv):
            raise V2EvaluationError("child /proc argv drift")
        if cwd != str(V2.REPO_ROOT):
            raise V2EvaluationError("child /proc cwd drift")
        if not os.path.samefile(executable, executable_authority):
            raise V2EvaluationError("child /proc executable drift")
        return {
            "pid": pid,
            "start_time_ticks": int(fields[19]),
            "ppid": int(fields[1]),
            "argv": cmdline,
            "cwd": cwd,
            "executable": str(executable_authority),
        }
    raise V2EvaluationError("child exited before exact process identity capture")


def _environment_custody() -> tuple[list[str], str]:
    environment = {str(key): str(value) for key, value in os.environ.items()}
    return sorted(environment), V2.canonical_json_sha256(environment)


def _static_audit_hook(event: str, arguments: tuple[Any, ...]) -> None:
    if event != "open":
        return
    with _STATIC_AUDIT_LOCK:
        capture = _STATIC_AUDIT_CAPTURE
        if capture is None:
            return
        if len(arguments) != 3:
            raise V2EvaluationError("CPython open audit argument drift")
        path, mode, flags = arguments
        if type(path) is not str or not path or "\0" in path:
            raise V2EvaluationError("non-text CPython open audit path")
        if mode is not None and type(mode) is not str:
            raise V2EvaluationError("CPython open audit mode drift")
        if type(flags) is not int:
            raise V2EvaluationError("CPython open audit flags drift")
        started = time.monotonic_ns()
        correlation = V2.current_contract_openat_audit_correlation()
        completed = time.monotonic_ns()
        capture["rows"].append(
            {
                "sequence": len(capture["rows"]),
                "audit_event_name": "open",
                "audit_path": path,
                "audit_mode": mode,
                "audit_flags": flags,
                "audit_cwd": capture["cwd"],
                "correlation_token": (
                    None if correlation is None else correlation["token"]
                ),
                "correlation_context": correlation,
                "open_started_monotonic_ns": started,
                "open_completed_monotonic_ns": completed,
            }
        )


def _ensure_static_audit_hook() -> None:
    global _STATIC_AUDIT_HOOK_INSTALLED
    with _STATIC_AUDIT_LOCK:
        if not _STATIC_AUDIT_HOOK_INSTALLED:
            sys.addaudithook(_static_audit_hook)
            _STATIC_AUDIT_HOOK_INSTALLED = True


def _begin_static_audit(
    *, stage_id: str, process_identity: Mapping[str, Any]
) -> int:
    global _STATIC_AUDIT_CAPTURE
    _ensure_static_audit_hook()
    installed = time.monotonic_ns()
    V2.begin_contract_openat_audit_ledger(
        stage_id=stage_id,
        process_identity=process_identity,
        audit_hook_installed_monotonic_ns=installed,
    )
    with _STATIC_AUDIT_LOCK:
        if _STATIC_AUDIT_CAPTURE is not None:
            raise V2EvaluationError("nested static-input audit capture")
        _STATIC_AUDIT_CAPTURE = {
            "stage_id": stage_id,
            "process_identity": copy.deepcopy(dict(process_identity)),
            "cwd": process_identity["cwd"],
            "installed_monotonic_ns": installed,
            "rows": [],
        }
    return installed


def _finish_static_audit(
    *,
    stage_id: str,
    process_identity: Mapping[str, Any],
    attempt_root: Path,
    installed_monotonic_ns: int,
) -> dict[str, Any]:
    global _STATIC_AUDIT_CAPTURE
    completed = time.monotonic_ns()
    with _STATIC_AUDIT_LOCK:
        capture = _STATIC_AUDIT_CAPTURE
        _STATIC_AUDIT_CAPTURE = None
    if (
        capture is None
        or capture["stage_id"] != stage_id
        or capture["process_identity"] != process_identity
        or capture["installed_monotonic_ns"] != installed_monotonic_ns
    ):
        raise V2EvaluationError("static-input audit capture drift")
    ledger = V2.finish_contract_openat_audit_ledger(
        stage_id=stage_id,
        process_identity=process_identity,
        audit_stream_completed_monotonic_ns=completed,
    )
    stream = V2.observe_static_input_open_events(
        stage_id=stage_id,
        process_identity=process_identity,
        attempt_root=attempt_root,
        audit_hook_installed_monotonic_ns=installed_monotonic_ns,
        audit_stream_completed_monotonic_ns=completed,
        raw_audit_open_rows=capture["rows"],
        contract_openat_audit_ledger=ledger,
        dropped_event_count=0,
    )
    return V2.validate_static_input_audit_stream_receipt(stream)


def _raw_static_audit_rows() -> list[dict[str, Any]]:
    """Snapshot the current process-local audit stream without ending it."""

    with _STATIC_AUDIT_LOCK:
        capture = _STATIC_AUDIT_CAPTURE
        if capture is None:
            raise V2EvaluationError("static-input audit capture is not active")
        return copy.deepcopy(capture["rows"])


def _first_split_audit_row() -> dict[str, Any]:
    expected = str(
        V2.REPO_ROOT / V2.STATIC_INPUT_BINDINGS["panel"]["split"]["path"]
    )
    matches = []
    for row in _raw_static_audit_rows():
        raw = row["audit_path"]
        classified = (
            raw
            if Path(raw).is_absolute()
            else str(Path(row["audit_cwd"]) / raw)
        )
        if classified == expected and row["correlation_token"] is None:
            matches.append(row)
    if len(matches) != 1:
        raise V2EvaluationError("first split open audit cardinality drift")
    return matches[0]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _activate_attempt_io(
    *,
    attempt_root: Path,
    root_fd: int,
    root_fd_custody: Mapping[str, Any],
    parent_fd: int | None,
    reopened_root_fd_custody: Mapping[str, Any] | None = None,
) -> None:
    global _ACTIVE_ATTEMPT_ROOT
    global _ACTIVE_ATTEMPT_ROOT_FD
    global _ACTIVE_ATTEMPT_PARENT_FD
    global _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY
    global _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY
    root = attempt_root.absolute()
    custody = V2.validate_attempt_root_fd_custody(
        root_fd_custody,
        root_fd=root_fd,
        parent_fd=parent_fd,
        reverify_live=True,
    )
    if custody["attempt_root"] != str(root):
        raise V2EvaluationError("attempt-root FD/canonical path drift")
    reopened = None
    if reopened_root_fd_custody is not None:
        if reopened_root_fd_custody.get("schema") == (
            V2.TERMINAL_PUBLICATION_SUPERVISOR_ROOT_REOPEN_CUSTODY_SCHEMA
        ):
            reopened = (
                V2.validate_terminal_publication_supervisor_root_reopen_custody(
                    reopened_root_fd_custody,
                    attempt_root_fd_custody=custody,
                    root_fd=root_fd,
                    parent_fd=parent_fd,
                    reverify_live=True,
                )
            )
        else:
            reopened = V2.validate_reopened_attempt_root_fd_custody(
                reopened_root_fd_custody,
                attempt_root_fd_custody=custody,
                root_fd=root_fd,
                parent_fd=parent_fd,
                reverify_live=True,
            )
    _ACTIVE_ATTEMPT_ROOT = root
    _ACTIVE_ATTEMPT_ROOT_FD = root_fd
    _ACTIVE_ATTEMPT_PARENT_FD = parent_fd
    _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY = custody
    _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY = reopened


def _close_attempt_io() -> None:
    global _ACTIVE_ATTEMPT_ROOT
    global _ACTIVE_ATTEMPT_ROOT_FD
    global _ACTIVE_ATTEMPT_PARENT_FD
    global _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY
    global _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY
    descriptors = []
    if _ACTIVE_ATTEMPT_ROOT_FD is not None:
        descriptors.append(_ACTIVE_ATTEMPT_ROOT_FD)
    if (
        _ACTIVE_ATTEMPT_PARENT_FD is not None
        and _ACTIVE_ATTEMPT_PARENT_FD not in descriptors
    ):
        descriptors.append(_ACTIVE_ATTEMPT_PARENT_FD)
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _ACTIVE_ATTEMPT_ROOT = None
    _ACTIVE_ATTEMPT_ROOT_FD = None
    _ACTIVE_ATTEMPT_PARENT_FD = None
    _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY = None
    _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY = None


def _require_attempt_io(
    attempt_root: Path,
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    root = attempt_root.absolute()
    if (
        _ACTIVE_ATTEMPT_ROOT != root
        or _ACTIVE_ATTEMPT_ROOT_FD is None
        or _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY is None
    ):
        raise V2EvaluationError("attempt-root FD capability is not active")
    if _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY is None:
        V2.validate_attempt_root_fd_custody(
            _ACTIVE_ATTEMPT_ROOT_FD_CUSTODY,
            root_fd=_ACTIVE_ATTEMPT_ROOT_FD,
            parent_fd=_ACTIVE_ATTEMPT_PARENT_FD,
            reverify_live=True,
        )
    else:
        if _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY.get("schema") == (
            V2.TERMINAL_PUBLICATION_SUPERVISOR_ROOT_REOPEN_CUSTODY_SCHEMA
        ):
            V2.validate_terminal_publication_supervisor_root_reopen_custody(
                _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY,
                attempt_root_fd_custody=_ACTIVE_ATTEMPT_ROOT_FD_CUSTODY,
                root_fd=_ACTIVE_ATTEMPT_ROOT_FD,
                parent_fd=_ACTIVE_ATTEMPT_PARENT_FD,
                reverify_live=True,
            )
        else:
            V2.validate_reopened_attempt_root_fd_custody(
                _ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY,
                attempt_root_fd_custody=_ACTIVE_ATTEMPT_ROOT_FD_CUSTODY,
                root_fd=_ACTIVE_ATTEMPT_ROOT_FD,
                parent_fd=_ACTIVE_ATTEMPT_PARENT_FD,
                reverify_live=True,
            )
    return (
        _ACTIVE_ATTEMPT_ROOT_FD,
        copy.deepcopy(_ACTIVE_ATTEMPT_ROOT_FD_CUSTODY),
        copy.deepcopy(_ACTIVE_REOPENED_ATTEMPT_ROOT_FD_CUSTODY),
    )


def _attempt_relative_path(path: Path) -> tuple[Path, str]:
    if _ACTIVE_ATTEMPT_ROOT is None:
        raise V2EvaluationError("attempt-root FD capability is not active")
    absolute = path.absolute()
    try:
        relative = str(absolute.relative_to(_ACTIVE_ATTEMPT_ROOT))
    except ValueError as exc:
        raise V2EvaluationError("path is outside the active attempt root") from exc
    if not relative or relative == ".":
        raise V2EvaluationError("attempt runtime leaf path is empty")
    return _ACTIVE_ATTEMPT_ROOT, relative


def _read_attempt_bytes(path: Path) -> bytes:
    root, relative = _attempt_relative_path(path)
    root_fd, custody, reopened = _require_attempt_io(root)
    return V2.read_v2_attempt_bytes_no_follow(
        attempt_root_fd_custody=custody,
        root_fd=root_fd,
        relative_path=relative,
        reopened_attempt_root_fd_custody=reopened,
    )


def _read_root_fd_bytes_bootstrap(root_fd: int, relative_path: str) -> bytes:
    """Read the root-custody receipt before its own validator is available."""

    parts = Path(relative_path).parts
    if (
        not parts
        or Path(relative_path).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or str(Path(relative_path)) != relative_path
    ):
        raise V2EvaluationError("unsafe attempt bootstrap path")
    directory_fd = os.dup(root_fd)
    descriptor = -1
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            metadata = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != V2.RUNTIME_DIRECTORY_MODE
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
            ):
                os.close(next_fd)
                raise V2EvaluationError("attempt bootstrap directory drift")
            os.close(directory_fd)
            directory_fd = next_fd
        before = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or stat.S_IMODE(opened.st_mode) != V2.RUNTIME_FILE_MODE
            or opened.st_uid != os.getuid()
            or opened.st_gid != os.getgid()
            or opened.st_nlink != 1
        ):
            raise V2EvaluationError("attempt bootstrap leaf custody drift")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise V2EvaluationError("attempt bootstrap leaf changed during read")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def _load_attempt_root_custody_for_reopen(
    attempt_root: Path,
) -> dict[str, Any]:
    """Bootstrap only the immutable historical root receipt, no-follow."""

    root = attempt_root.absolute()
    if (
        root.parent != V2.OUTPUT_ROOT.parent
        or not root.name.startswith(f".{V2.OUTPUT_ROOT.name}.attempt-")
    ):
        raise V2EvaluationError("attempt-root bootstrap grammar drift")
    descriptor = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    parent_metadata: os.stat_result | None = None
    root_metadata: os.stat_result | None = None

    def stable_directory_identity(
        value: os.stat_result,
    ) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IMODE(value.st_mode),
            value.st_uid,
            value.st_gid,
        )

    try:
        components = root.parts[1:]
        for index, part in enumerate(components):
            next_fd = -1
            try:
                before = os.stat(
                    part, dir_fd=descriptor, follow_symlinks=False
                )
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                opened = os.fstat(next_fd)
                after = os.stat(
                    part, dir_fd=descriptor, follow_symlinks=False
                )
                final_component = index == len(components) - 1
                if (
                    not all(
                        stat.S_ISDIR(value.st_mode)
                        for value in (before, opened, after)
                    )
                    or stable_directory_identity(before)
                    != stable_directory_identity(opened)
                    or stable_directory_identity(opened)
                    != stable_directory_identity(after)
                    or final_component
                    and (
                        stat.S_IMODE(opened.st_mode)
                        != V2.ATTEMPT_ROOT_DIRECTORY_MODE
                        or opened.st_uid != os.getuid()
                        or opened.st_gid != os.getgid()
                    )
                ):
                    raise V2EvaluationError(
                        "attempt-root bootstrap component drift"
                    )
            except BaseException:
                if next_fd >= 0:
                    os.close(next_fd)
                raise
            if final_component:
                parent_metadata = os.fstat(descriptor)
                root_metadata = opened
            os.close(descriptor)
            descriptor = next_fd
        relative = V2.RUNTIME_PATHS["attempt_root_fd_custody"]
        value = _canonical_object_from_bytes(
            _read_root_fd_bytes_bootstrap(descriptor, relative),
            label=relative,
        )
        custody = V2.validate_attempt_root_fd_custody(
            value, reverify_live=False
        )
        if parent_metadata is None or root_metadata is None:
            raise V2EvaluationError("attempt-root bootstrap identity absent")
        historical_parent = custody["parent_stat"]
        historical_root = custody["root_stat"]
        if (
            custody["attempt_root"] != str(root)
            or stable_directory_identity(parent_metadata)
            != tuple(
                historical_parent[key]
                for key in ("device", "inode", "mode", "uid", "gid")
            )
            or stable_directory_identity(root_metadata)
            != tuple(
                historical_root[key]
                for key in ("device", "inode", "mode", "uid", "gid")
            )
        ):
            raise V2EvaluationError("historical attempt-root custody drift")
        return custody
    finally:
        os.close(descriptor)


def _load_attempt_json_bootstrap(
    attempt_root: Path, relative_path: str
) -> dict[str, Any]:
    historical = _load_attempt_root_custody_for_reopen(attempt_root)
    root = attempt_root.absolute()
    descriptor = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        for part in root.parts[1:]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_fd)
                raise V2EvaluationError("bootstrap JSON root component drift")
            os.close(descriptor)
            descriptor = next_fd
        metadata = os.fstat(descriptor)
        expected = historical["root_stat"]
        if any(
            metadata_value != expected[key]
            for key, metadata_value in (
                ("device", metadata.st_dev),
                ("inode", metadata.st_ino),
                ("mode", stat.S_IMODE(metadata.st_mode)),
                ("uid", metadata.st_uid),
                ("gid", metadata.st_gid),
            )
        ):
            raise V2EvaluationError("bootstrap JSON root identity drift")
        return _canonical_object_from_bytes(
            _read_root_fd_bytes_bootstrap(descriptor, relative_path),
            label=relative_path,
        )
    finally:
        os.close(descriptor)


def _activate_reopened_attempt_io(
    *,
    attempt_root: Path,
    phase_id: str,
    phase_process_identity: Mapping[str, Any],
    publication_attempt_number: int | None = None,
    publication_attempt_id: str | None = None,
    presentation_retry_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    historical = _load_attempt_root_custody_for_reopen(attempt_root)
    parent_fd, root_fd, reopened = V2.reopen_attempt_root_anchored(
        attempt_root_fd_custody=historical,
        phase_id=phase_id,
        phase_process_identity=phase_process_identity,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=presentation_retry_authority,
    )
    try:
        _activate_attempt_io(
            attempt_root=attempt_root,
            root_fd=root_fd,
            root_fd_custody=historical,
            parent_fd=parent_fd,
            reopened_root_fd_custody=reopened,
        )
    except BaseException:
        os.close(root_fd)
        os.close(parent_fd)
        raise
    return reopened


def _canonical_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2EvaluationError(f"invalid canonical JSON: {label}") from exc
    if type(value) is not dict or payload != V2.canonical_json_bytes(value) + b"\n":
        raise V2EvaluationError(f"non-canonical JSON object: {label}")
    return value


def _activate_inherited_attempt_io(
    *,
    attempt_root: Path,
    child_process_identity: Mapping[str, Any],
    expected_parent_argv: Sequence[str],
) -> dict[str, Any]:
    raw_fd = os.environ.get(V2.ATTEMPT_ROOT_FD_ENVIRONMENT_NAME)
    if raw_fd is None or not raw_fd.isdigit():
        raise V2EvaluationError("inherited attempt-root FD is absent")
    root_fd = int(raw_fd)
    custody_path = V2.RUNTIME_PATHS["attempt_root_fd_custody"]
    custody = V2.validate_attempt_root_fd_custody(
        _canonical_object_from_bytes(
            _read_root_fd_bytes_bootstrap(root_fd, custody_path),
            label=custody_path,
        ),
        root_fd=root_fd,
        reverify_live=True,
    )
    _activate_attempt_io(
        attempt_root=attempt_root,
        root_fd=root_fd,
        root_fd_custody=custody,
        parent_fd=None,
    )
    parent = _process_identity_for_pid(
        int(child_process_identity["ppid"]), expected_argv=expected_parent_argv
    )
    inherited = V2.build_inherited_attempt_root_fd_custody(
        attempt_root_fd_custody=custody,
        parent_process_identity=parent,
        child_process_identity=child_process_identity,
        inherited_root_fd=root_fd,
        fd_was_passed_explicitly=True,
        environment_fd_value=raw_fd,
    )
    return V2.build_child_attempt_root_fd_binding(
        attempt_root_fd_custody=custody,
        inherited_attempt_root_fd_custody=inherited,
    )


def _attempt_fd_alias(attempt_root: Path) -> Path:
    root_fd, _custody, _reopened = _require_attempt_io(attempt_root)
    return Path(f"/proc/self/fd/{root_fd}")


def _open_attempt_parent(relative_path: str, *, create: bool) -> tuple[int, str]:
    if _ACTIVE_ATTEMPT_ROOT_FD is None:
        raise V2EvaluationError("attempt-root FD capability is not active")
    parts = Path(relative_path).parts
    if (
        not parts
        or Path(relative_path).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or str(Path(relative_path)) != relative_path
    ):
        raise V2EvaluationError("unsafe attempt-relative path")
    directory_fd = os.dup(_ACTIVE_ATTEMPT_ROOT_FD)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=V2.RUNTIME_DIRECTORY_MODE, dir_fd=directory_fd)
                os.fsync(directory_fd)
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            metadata = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != V2.RUNTIME_DIRECTORY_MODE
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
            ):
                os.close(next_fd)
                raise V2EvaluationError("attempt runtime directory custody drift")
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd, parts[-1]
    except BaseException:
        os.close(directory_fd)
        raise


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if mode != V2.RUNTIME_FILE_MODE:
        raise V2EvaluationError("attempt runtime file mode drift")
    root, relative = _attempt_relative_path(path)
    root_fd, custody, reopened = _require_attempt_io(root)
    V2.write_v2_attempt_bytes_exclusive_fsync(
        attempt_root_fd_custody=custody,
        root_fd=root_fd,
        relative_path=relative,
        payload=payload,
        content_digest=None,
        rows=None,
        reopened_attempt_root_fd_custody=reopened,
    )


def _replace_bytes_atomic(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if mode != V2.RUNTIME_FILE_MODE:
        raise V2EvaluationError("attempt runtime file mode drift")
    root, relative = _attempt_relative_path(path)
    root_fd, custody, reopened = _require_attempt_io(root)
    parent_fd, leaf = _open_attempt_parent(relative, create=True)
    temporary = f".{leaf}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            V2.RUNTIME_FILE_MODE,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise V2EvaluationError("short atomic attempt write")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != V2.RUNTIME_FILE_MODE
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise V2EvaluationError("atomic attempt leaf custody drift")
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            leaf,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    if reopened is None:
        V2.validate_attempt_root_fd_custody(
            custody, root_fd=root_fd, reverify_live=True
        )
    else:
        V2.validate_reopened_attempt_root_fd_custody(
            reopened,
            attempt_root_fd_custody=custody,
            root_fd=root_fd,
            reverify_live=True,
        )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, V2.canonical_json_bytes(value) + b"\n")


def _write_last_stage(
    attempt_root: Path,
    phase_id: str,
    stage: str,
    *,
    publication_runtime_paths: Mapping[str, str] | None = None,
) -> None:
    if phase_id not in V2.PHASE_IDS or stage not in V2.PHASE_STAGE_IDS[phase_id]:
        raise V2EvaluationError("last-stage authority drift")
    if phase_id == V2.PHASE_IDS[0]:
        relative_path = V2.RUNTIME_PATHS["phase_1_producer_last_stage"]
    elif phase_id == V2.PHASE_IDS[1]:
        relative_path = V2.RUNTIME_PATHS["phase_2_last_stage"]
    elif publication_runtime_paths is not None:
        relative_path = publication_runtime_paths["last_stage"]
    else:
        raise V2EvaluationError("Phase-3 last-stage path is not selected")
    value = V2.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v2.last_stage.v1",
            "phase_id": phase_id,
            "stage": stage,
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    _replace_bytes_atomic(
        attempt_root / relative_path,
        V2.canonical_json_bytes(value) + b"\n",
        mode=0o600,
    )


def _artifact_binding(path: Path, *, attempt_root: Path, rows: int | None) -> dict[str, Any]:
    payload = _read_attempt_bytes(path)
    try:
        relative = str(path.relative_to(attempt_root))
    except ValueError as exc:
        raise V2EvaluationError("artifact is outside the V2 attempt root") from exc
    content_digest: str | None = None
    if path.suffix == ".json":
        parsed = json.loads(payload.decode("utf-8"))
        if isinstance(parsed, Mapping):
            content_digest = parsed.get("content_digest")
    return V2.build_artifact_binding(
        path=relative,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes_count=len(payload),
        content_digest=content_digest,
        rows=rows,
    )


def _scientific_authority_bindings(
    attempt_root: Path, keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    return {
        key: _artifact_binding(
            attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[key]["path"],
            attempt_root=attempt_root,
            rows=V2.SCIENTIFIC_ARTIFACT_AUTHORITY[key]["rows"],
        )
        for key in keys
    }


def _new_attempt_root(
    producer_process_identity: Mapping[str, Any],
    attempt_id: str,
) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    observation = V2.observe_fresh_v2_namespace(attempt_id)
    receipt = V2.build_namespace_and_nonreuse_receipt(
        attempt_id=attempt_id,
        precreation_observation=observation,
    )
    root = Path(receipt["attempt_root"])
    parent_fd, root_fd, root_custody = V2.create_fresh_attempt_root_anchored(
        namespace_and_nonreuse_receipt=receipt,
        producer_process_identity=producer_process_identity,
    )
    _activate_attempt_io(
        attempt_root=root,
        root_fd=root_fd,
        root_fd_custody=root_custody,
        parent_fd=parent_fd,
    )
    return attempt_id, root, receipt, root_custody


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(V2.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _scientific_counter_state() -> dict[str, int]:
    return {key: 0 for key in V2.SCIENTIFIC_COUNTER_FIELDS}


def _require_metadata_only_regular_file(
    path: Path, *, expected_bytes: int | None
) -> None:
    """Validate one frozen input using metadata syscalls, never a content open."""

    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.is_absolute() else absolute.parts
    for index, part in enumerate(parts):
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError as exc:
            raise V2EvaluationError(
                f"frozen static-input path is absent: {absolute}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise V2EvaluationError(
                f"frozen static-input path crosses a symlink: {cursor}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise V2EvaluationError(
                f"frozen static-input parent is not a directory: {cursor}"
            )
    if not stat.S_ISREG(metadata.st_mode):
        raise V2EvaluationError(
            f"frozen static-input path is not a regular file: {absolute}"
        )
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise V2EvaluationError(
            f"frozen static-input byte metadata drift: {absolute}"
        )


def _validate_static_input_metadata_without_content_open() -> bool:
    """Walk the frozen binding maps and lstat every bound input file."""

    checked: dict[Path, int | None] = {}

    def visit(value: Any, *, base: Path) -> None:
        if isinstance(value, Mapping):
            local_base = base
            root_value = value.get("root")
            if isinstance(root_value, str) and root_value:
                root_path = Path(root_value)
                local_base = (
                    root_path if root_path.is_absolute() else base / root_path
                )
                try:
                    root_metadata = os.lstat(local_base)
                except FileNotFoundError as exc:
                    raise V2EvaluationError(
                        f"frozen static-input root is absent: {local_base}"
                    ) from exc
                if (
                    stat.S_ISLNK(root_metadata.st_mode)
                    or not stat.S_ISDIR(root_metadata.st_mode)
                ):
                    raise V2EvaluationError(
                        f"frozen static-input root metadata drift: {local_base}"
                    )
            path_value = value.get("path")
            if isinstance(path_value, str) and path_value:
                candidate = Path(path_value)
                if not candidate.is_absolute():
                    candidate = local_base / candidate
                byte_value = value.get("bytes")
                expected_bytes = (
                    byte_value if type(byte_value) is int else None
                )
                prior = checked.get(candidate)
                if candidate in checked and prior != expected_bytes:
                    raise V2EvaluationError(
                        f"conflicting frozen byte metadata: {candidate}"
                    )
                checked[candidate] = expected_bytes
            for key, item in value.items():
                if key not in ("root", "path"):
                    visit(item, base=local_base)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ):
            for item in value:
                visit(item, base=base)

    visit(V2.STATIC_INPUT_BINDINGS, base=V2.REPO_ROOT)
    if not checked:
        raise V2EvaluationError("frozen static-input metadata set is empty")
    for path, expected_bytes in sorted(
        checked.items(), key=lambda item: str(item[0])
    ):
        _require_metadata_only_regular_file(
            path, expected_bytes=expected_bytes
        )
    return True


def _validate_interpreter_and_resource_paths() -> bool:
    """Validate executable/resource metadata without reading their contents."""

    for executable in (
        V2.PYTHON_EXECUTABLE,
        V2.CPU_WORKER_PYTHON_EXECUTABLE,
    ):
        try:
            target = executable.resolve(strict=True)
            metadata = os.stat(target)
        except (FileNotFoundError, OSError) as exc:
            raise V2EvaluationError(
                f"V2 interpreter is unavailable: {executable}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or not os.access(
            target, os.X_OK
        ):
            raise V2EvaluationError(
                f"V2 interpreter metadata drift: {executable}"
            )
    evaluator_metadata = os.lstat(V2.EVALUATOR_PATH)
    repo_metadata = os.lstat(V2.REPO_ROOT)
    if (
        stat.S_ISLNK(evaluator_metadata.st_mode)
        or not stat.S_ISREG(evaluator_metadata.st_mode)
        or stat.S_ISLNK(repo_metadata.st_mode)
        or not stat.S_ISDIR(repo_metadata.st_mode)
    ):
        raise V2EvaluationError("V2 evaluator/resource metadata drift")
    return True


def _build_static_stage_custody(
    *,
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    static_input_audit_stream: Mapping[str, Any],
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase1_preexecution_receipt: Mapping[str, Any] | None = None,
    phase1_boundary_entry_receipt: Mapping[str, Any] | None = None,
    first_scientific_open_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = V2.build_static_input_stage_custody(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        static_input_audit_stream=static_input_audit_stream,
        stage_started_monotonic_ns=started_monotonic_ns,
        stage_ended_monotonic_ns=ended_monotonic_ns,
        phase1_preexecution_receipt=phase1_preexecution_receipt,
        phase1_boundary_entry_receipt=phase1_boundary_entry_receipt,
        first_scientific_open_receipt=first_scientific_open_receipt,
    )
    return V2.validate_static_input_stage_custody(receipt)


def _observe_static_snapshot(
    *,
    stage_id: str,
    snapshot_kind: str,
    process_identity: Mapping[str, Any],
    attempt_root: Path,
) -> dict[str, Any]:
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    return V2.observe_static_input_snapshot(
        stage_id=stage_id,
        snapshot_kind=snapshot_kind,
        stage_process_identity=process_identity,
        attempt_root_fd_custody=historical,
        root_fd=root_fd,
        reopened_attempt_root_fd_custody=reopened,
    )


def _static_stage_runtime_key(stage_id: str) -> str:
    return {
        "PHASE1_STAGE_A": "static_input_stage_a_custody",
        "PHASE1_STAGE_B_HELPER": "static_input_stage_b_helper_custody",
        "PHASE1_STAGE_C_HELPER": "static_input_stage_c_helper_custody",
        "PHASE2_REPLAY": "static_input_phase2_replay_custody",
    }[stage_id]


def _monotonic_ns_after(value: int) -> int:
    observed = time.monotonic_ns()
    while observed <= value:
        observed = time.monotonic_ns()
    return observed


def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = _read_attempt_bytes(path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise V2EvaluationError(f"invalid UTF-8 JSONL: {path}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V2EvaluationError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        if type(value) is not dict:
            raise V2EvaluationError(
                f"non-object JSONL row at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _load_canonical_json(path: Path) -> dict[str, Any]:
    payload = _read_attempt_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2EvaluationError(f"invalid UTF-8 JSON: {path}") from exc
    if type(value) is not dict:
        raise V2EvaluationError(f"JSON object required: {path}")
    if payload != V2.canonical_json_bytes(value) + b"\n":
        raise V2EvaluationError(f"non-canonical JSON bytes: {path}")
    return value


def _verify_artifact_binding(
    path: Path,
    binding: Mapping[str, Any],
    *,
    attempt_root: Path,
) -> bytes:
    validated = V2.validate_artifact_binding(binding)
    expected = attempt_root / validated["path"]
    if path != expected:
        raise V2EvaluationError("artifact binding resolved to the wrong path")
    payload = _read_attempt_bytes(path)
    if (
        len(payload) != validated["bytes"]
        or hashlib.sha256(payload).hexdigest() != validated["sha256"]
    ):
        raise V2EvaluationError(f"artifact bytes differ from binding: {path}")
    if validated["content_digest"] is not None:
        value = _load_canonical_json(path)
        if value.get("content_digest") != validated["content_digest"]:
            raise V2EvaluationError(f"artifact content digest differs: {path}")
        V2.validate_self_digest(value)
    return payload


def _identity_is_live(identity: Mapping[str, Any]) -> bool:
    pid = identity.get("pid")
    ticks = identity.get("start_time_ticks")
    if type(pid) is not int or type(ticks) is not int:
        raise V2EvaluationError("process identity is malformed")
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    closing = text.rfind(")")
    fields = text[closing + 2 :].split()
    return closing >= 0 and len(fields) >= 20 and int(fields[19]) == ticks


def _assert_no_replay_symbols(callables: Sequence[Callable[..., Any]]) -> None:
    for function in callables:
        names = frozenset(function.__code__.co_names)
        overlap = names & _PROHIBITED_REPLAY_SYMBOLS
        if overlap:
            raise V2EvaluationError(
                f"V2 scientific callable references replay symbols: {sorted(overlap)}"
            )


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=V2.REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _require_clean_source_freeze() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain"):
        raise V2EvaluationError("V2 scientific execution requires a clean source freeze")
    return head


def _route_roles(v1: Any) -> dict[str, str]:
    authority_path = V2.REPO_ROOT / v1.CONTRACT.TRACKED_ROUTE_ROLE_AUTHORITY_PATH
    return v1.CONTRACT.load_route_role_map(authority_path)


def _optimizer_update_count(training_receipt: Mapping[str, Any]) -> int:
    histories = training_receipt.get("training_history")
    if not isinstance(histories, Mapping):
        raise V2EvaluationError("training receipt lacks histories")
    updates = 0
    for history in histories.values():
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise V2EvaluationError("training history is malformed")
        for row in history:
            if not isinstance(row, Mapping) or type(row.get("optimizer_steps")) is not int:
                raise V2EvaluationError("training optimizer-step custody drift")
            updates += int(row["optimizer_steps"])
    return updates


def _run_stage_a(
    *,
    attempt_root: Path,
    source_freeze: str,
    counters: dict[str, int],
    v1: Any,
    ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Run the frozen prospective Stage A without any correction replay."""

    reused = (
        v1.split_ids,
        v1.route_line_index,
        v1.tensor_index,
        v1.state_manifest_map,
        v1.authorised_dataset,
        v1.run_training_smoke,
        v1.train_rankers,
        v1.write_evaluation_contract,
        v1._predecessor_raw_goal_score_maps,
        v1.score_dataset,
        v1._stage_a_ledger_rows,
        v1._write_training_ledgers,
        v1._stage_a_decisions,
    )
    _assert_no_replay_symbols(reused)
    v1_output_root = _attempt_fd_alias(attempt_root)

    line_index = v1.route_line_index()
    contexts = v1._index_by_state(v1.CONTEXT_INDEX)
    goals = v1._index_by_state(v1.GOAL_INDEX)
    manifests = v1.state_manifest_map()
    tensor_rows = v1.tensor_index()
    route_roles = _route_roles(v1)
    counters["route_labels_opened"] = v1.STATE_COUNT * v1.CANDIDATE_COUNT

    fit = v1.authorised_dataset(
        v1.FIT,
        ids=ids,
        line_index=line_index,
        contexts=contexts,
        goals=goals,
        route_roles=route_roles,
    )
    counters["fit_states_opened"] = len(fit)
    counters["fit_rows_opened"] = sum(len(rows) for rows in fit.values())
    smoke = v1.run_training_smoke(
        fit, tensor_rows, ids=ids, output_root=v1_output_root
    )
    checkpoints, training_receipt, models = v1.train_rankers(
        fit, tensor_rows, output_root=v1_output_root
    )
    counters["model_initializations"] = 2
    counters["optimizer_updates"] = _optimizer_update_count(training_receipt)
    counters["training_epochs"] = sum(
        len(history) for history in training_receipt["training_history"].values()
    )
    evaluation_contract = v1.write_evaluation_contract(
        output_root=v1_output_root,
        source_freeze_commit=source_freeze,
        checkpoints=checkpoints,
        line_index=line_index,
        ids=ids,
        manifests=manifests,
    )

    calibration = v1.authorised_dataset(
        v1.CALIBRATION,
        ids=ids,
        line_index=line_index,
        contexts=contexts,
        goals=goals,
        route_roles=route_roles,
    )
    counters["calibration_states_opened"] = len(calibration)
    counters["calibration_rows_opened"] = sum(
        len(rows) for rows in calibration.values()
    )
    heldout = v1.authorised_dataset(
        v1.HELDOUT,
        ids=ids,
        line_index=line_index,
        contexts=contexts,
        goals=goals,
        route_roles=route_roles,
    )
    counters["heldout_states_opened"] = len(heldout)
    counters["heldout_rows_opened"] = sum(len(rows) for rows in heldout.values())
    datasets = {
        v1.FIT: fit,
        v1.CALIBRATION: calibration,
        v1.HELDOUT: heldout,
    }

    raw_maps, raw_rows, raw_reduction = v1._predecessor_raw_goal_score_maps(
        datasets, heldout_barrier_open=True
    )
    if len(raw_rows) != V2.STAGE_ROW_AUTHORITY["stage_a_raw_cost_rows"]:
        raise V2EvaluationError("Stage-A raw comparator cardinality drift")
    score_maps_by_role: dict[str, Any] = {}
    evidence_by_role: dict[str, Any] = {}
    summaries_by_role: dict[str, Any] = {}
    future_maps = evaluation_contract["future_latent_derangement_by_state"]
    for role in v1.SPLIT_ROLES:
        score_maps, evidence = v1.score_dataset(
            datasets[role],
            models=models,
            tensor_rows=tensor_rows,
            latent_source="TRUE",
            future_derangements={state: future_maps[state] for state in datasets[role]},
        )
        v1._merge_score_map(score_maps, raw_maps[role])
        score_maps_by_role[role] = score_maps
        evidence_by_role[role] = evidence
        summaries_by_role[role] = v1.summarize_score_maps(
            datasets[role], score_maps
        )
    stage_a_rows = v1._stage_a_ledger_rows(
        datasets, score_maps_by_role, evidence_by_role
    )
    if len(stage_a_rows) != V2.STAGE_ROW_AUTHORITY["stage_a_rows"]:
        raise V2EvaluationError("Stage-A ledger cardinality drift")
    _write_bytes_exclusive(
        attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_a_rows"]["path"],
        _canonical_jsonl(stage_a_rows),
    )
    _write_bytes_exclusive(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_a_raw_cost_rows"]["path"],
        _canonical_jsonl(raw_rows),
    )
    v1._write_training_ledgers(
        attempt=v1_output_root,
        datasets=datasets,
        training_receipt=training_receipt,
    )
    decisions = v1._stage_a_decisions(summaries_by_role[v1.HELDOUT])
    counters["route_cost_checkpoints_opened"] = 2
    counters["true_future_latents_opened"] = v1.STATE_COUNT * v1.CANDIDATE_COUNT
    counters["scientific_score_rows"] = len(stage_a_rows) + len(raw_rows)
    return {
        "ids": ids,
        "manifests": manifests,
        "tensor_rows": tensor_rows,
        "datasets": datasets,
        "models": models,
        "checkpoints": checkpoints,
        "training_receipt": training_receipt,
        "training_smoke": smoke,
        "evaluation_contract": evaluation_contract,
        "summaries": summaries_by_role,
        "stage_a_rows": stage_a_rows,
        "raw_rows": raw_rows,
        "raw_reduction": raw_reduction,
        "decisions": decisions,
    }


def _scientific_decision(
    stage_a_decisions: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    v1 = _v1()
    stage_b_decisions = (
        None
        if stage_b is None
        else {
            key: copy.deepcopy(stage_b[key])
            for key in _STAGE_B_DECISION_KEYS
        }
    )
    stage_c_decisions = (
        None
        if stage_c is None
        else {
            key: copy.deepcopy(stage_c[key])
            for key in _STAGE_C_DECISION_KEYS
        }
    )
    primary, secondaries, next_experiment = v1._primary_and_secondaries(
        stage_a_decisions, stage_b_decisions, stage_c_decisions
    )
    result = V2.build_scientific_decision(
        stage_a_decisions=stage_a_decisions,
        stage_b_decisions=stage_b_decisions,
        stage_c_decisions=stage_c_decisions,
    )
    if (
        result["primary_classification"] != primary
        or result["secondary_classifications"] != secondaries
        or result["next_experiment"] != next_experiment
    ):
        raise V2EvaluationError(
            "V2 decision builder differs from frozen _primary_and_secondaries"
        )
    return result


def _publish_v2_stage_b_gate(
    *,
    attempt_root: Path,
    source_freeze: str,
    evaluation_contract: Mapping[str, Any],
    stage_a_metrics: Mapping[str, Any],
    stage_a_decisions: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Persist the V2-native gate without entering the replay-era publisher."""

    del source_freeze
    preexecution = _load_canonical_json(
        V2.runtime_path("preexecution_receipt", attempt_root)
    )
    evidence = V2.build_stage_a_gate_evidence(
        preexecution_receipt=preexecution,
        evaluation_contract=evaluation_contract,
        stage_a_rows=_rows_from_jsonl(
            attempt_root
            / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_a_rows"]["path"]
        ),
        stage_a_raw_cost_rows=_rows_from_jsonl(
            attempt_root
            / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
                "stage_a_raw_cost_rows"
            ]["path"]
        ),
        stage_a_summaries=stage_a_metrics,
        stage_a_decisions=stage_a_decisions,
        prerequisite_artifact_bindings=_scientific_authority_bindings(
            attempt_root, V2.STAGE_A_GATE_INPUT_ARTIFACT_KEYS
        ),
    )
    V2.validate_stage_a_gate_evidence(
        evidence, preexecution_receipt=preexecution
    )
    evidence_path = attempt_root / V2.RUNTIME_PATHS["stage_a_gate_evidence"]
    _write_json_exclusive(evidence_path, evidence)
    gate = V2.build_stage_b_gate_receipt(
        preexecution_receipt=preexecution,
        stage_a_gate_evidence=evidence,
        stage_a_gate_evidence_binding=_artifact_binding(
            evidence_path, attempt_root=attempt_root, rows=None
        ),
    )
    gate_path = V2.runtime_path("stage_b_gate_receipt", attempt_root)
    _write_json_exclusive(gate_path, gate)
    V2.validate_stage_b_gate_receipt(
        gate, preexecution_receipt=preexecution
    )
    return gate_path, gate


def _publish_v2_stage_c_gate(
    *,
    attempt_root: Path,
    source_freeze: str,
    stage_b_gate_path: Path,
    evaluation_contract: Mapping[str, Any],
    stage_b_decisions: Mapping[str, Any],
    heldout_summaries: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Persist the V2-native Stage-C gate from the frozen decision evidence."""

    del source_freeze, evaluation_contract
    preexecution = _load_canonical_json(
        V2.runtime_path("preexecution_receipt", attempt_root)
    )
    stage_b_gate = _load_canonical_json(stage_b_gate_path)
    stage_b_rows = _rows_from_jsonl(
        attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_b_rows"]["path"]
    )
    evidence = V2.build_stage_b_gate_evidence(
        preexecution_receipt=preexecution,
        stage_b_gate_receipt=stage_b_gate,
        stage_b_rows=stage_b_rows,
        stage_b_summaries=heldout_summaries,
        stage_b_decisions=stage_b_decisions,
        prerequisite_artifact_bindings=_scientific_authority_bindings(
            attempt_root, V2.STAGE_B_GATE_INPUT_ARTIFACT_KEYS
        ),
    )
    V2.validate_stage_b_gate_evidence(
        evidence,
        preexecution_receipt=preexecution,
        stage_b_gate_receipt=stage_b_gate,
    )
    evidence_path = attempt_root / V2.RUNTIME_PATHS["stage_b_gate_evidence"]
    _write_json_exclusive(evidence_path, evidence)
    gate = V2.build_stage_c_gate_receipt(
        preexecution_receipt=preexecution,
        stage_b_gate_receipt=stage_b_gate,
        stage_b_gate_evidence=evidence,
        stage_b_gate_evidence_binding=_artifact_binding(
            evidence_path, attempt_root=attempt_root, rows=None
        ),
    )
    gate_path = V2.runtime_path("stage_c_gate_receipt", attempt_root)
    _write_json_exclusive(gate_path, gate)
    V2.validate_stage_c_gate_receipt(
        gate, preexecution_receipt=preexecution
    )
    return gate_path, gate


def _load_stage_b_helper_outputs(
    *,
    attempt_root: Path,
    gate_path: Path,
    gate: Mapping[str, Any],
    helper_receipt: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int | None, int | None], dict[str, Any]], dict[str, Any]]:
    v1 = _v1()
    validated_helper = V2.validate_phase1_helper_receipt(helper_receipt)
    if validated_helper["mode"] != V2.PHASE1_PROPRIO_HELPER_MODE:
        raise V2EvaluationError("Stage-B helper receipt mode drift")
    v1_root = _attempt_fd_alias(attempt_root)
    materializer = _materializer_v1()
    contexts = materializer._load_context_index(
        v1_root, str(gate["content_digest"])
    )
    if len(contexts) != v1.STATE_COUNT:
        raise V2EvaluationError("Stage-B context index is incomplete")
    context_index = v1._validate_proprio_context_receipts(
        attempt=v1_root,
        gate_digest=str(gate["content_digest"]),
        contexts=contexts,
    )
    combined: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    indexes: dict[str, Any] = {}
    for source_id in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        rows, value = v1._load_prediction_index(
            attempt=v1_root,
            source_id=source_id,
            gate_digest=str(gate["content_digest"]),
        )
        if set(combined) & set(rows):
            raise V2EvaluationError("Stage-B prediction indexes overlap")
        combined.update(rows)
        path = attempt_root / "stage_b/predictions" / source_id / "index.json"
        indexes[source_id] = {
            "binding": _artifact_binding(path, attempt_root=attempt_root, rows=1_728),
            "content_digest": value["content_digest"],
        }
    return combined, {
        "stage_b_gate": _artifact_binding(
            gate_path, attempt_root=attempt_root, rows=None
        ),
        "stage_b_gate_content_digest": gate["content_digest"],
        "helper_receipt": validated_helper,
        "context_index": _artifact_binding(
            attempt_root / "stage_b/proprio_context/index.json",
            attempt_root=attempt_root,
            rows=48,
        ),
        "context_index_content_digest": context_index["content_digest"],
        "prediction_indexes": indexes,
        "correction_replay_receipt": None,
        "failed_output_files_reused": 0,
    }


def _load_stage_c_helper_outputs(
    *,
    attempt_root: Path,
    stage_b_gate_digest: str,
    stage_c_gate_digest: str,
    helper_receipt: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int | None, int | None], dict[str, Any]], dict[str, Any]]:
    v1 = _v1()
    validated_helper = V2.validate_phase1_helper_receipt(helper_receipt)
    if validated_helper["mode"] != V2.PHASE1_STAGE_C_HELPER_MODE:
        raise V2EvaluationError("Stage-C helper receipt mode drift")
    v1_root = _attempt_fd_alias(attempt_root)
    rows: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    indexes: dict[str, Any] = {}
    for source_id in v1.STAGE_C_SOURCE_IDS:
        source_rows, value = v1._load_prediction_index(
            attempt=v1_root,
            source_id=source_id,
            gate_digest=stage_b_gate_digest,
            stage_c_gate_digest=stage_c_gate_digest,
        )
        if set(rows) & set(source_rows):
            raise V2EvaluationError("Stage-C prediction indexes overlap")
        rows.update(source_rows)
        path = attempt_root / "stage_b/predictions" / source_id / "index.json"
        indexes[source_id] = {
            "binding": _artifact_binding(path, attempt_root=attempt_root, rows=1_728),
            "content_digest": value["content_digest"],
        }
    return rows, {
        "helper_receipt": validated_helper,
        "prediction_indexes": indexes,
        "correction_replay_receipt": None,
        "failed_output_files_reused": 0,
    }


def _run_conditional_tree(
    *,
    attempt_root: Path,
    source_freeze: str,
    stage_a: Mapping[str, Any],
    counters: dict[str, int],
    helper_runner: Callable[[str, Path], Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Execute B/C exactly when the frozen prospective gates authorize them."""

    decisions = stage_a["decisions"]
    if not decisions["true_future_gate"]["pass"]:
        return None, None
    v1 = _v1()
    v1_root = _attempt_fd_alias(attempt_root)
    gate_path, gate = _publish_v2_stage_b_gate(
        attempt_root=attempt_root,
        source_freeze=source_freeze,
        evaluation_contract=stage_a["evaluation_contract"],
        stage_a_metrics=stage_a["summaries"],
        stage_a_decisions=decisions,
    )
    helper_receipt = helper_runner(V2.PHASE1_PROPRIO_HELPER_MODE, attempt_root)
    conditional_rows, materialisation = _load_stage_b_helper_outputs(
        attempt_root=attempt_root,
        gate_path=gate_path,
        gate=gate,
        helper_receipt=helper_receipt,
    )
    all_tensors = dict(stage_a["tensor_rows"])
    if set(all_tensors) & set(conditional_rows):
        raise V2EvaluationError("P1/PR tensor identities overlap predecessor tensors")
    all_tensors.update(conditional_rows)
    sources = ("R1", "RR", "P1", "PR")
    score_maps, evidence, summaries = v1._score_conditional_sources(
        attempt=v1_root,
        datasets=stage_a["datasets"],
        models=stage_a["models"],
        tensor_rows=all_tensors,
        sources=sources,
    )
    rows = v1._conditional_ledger_rows(
        datasets=stage_a["datasets"],
        score_maps_by_role=score_maps,
        evidence_by_role=evidence,
        sources=sources,
        schema="plan_aware_evaluation_row_v1",
    )
    if len(rows) != V2.STAGE_ROW_AUTHORITY["stage_b_rows_if_executed"]:
        raise V2EvaluationError("Stage-B ledger cardinality drift")
    stage_b_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_b_rows"]["path"]
    _write_bytes_exclusive(stage_b_path, _canonical_jsonl(rows))
    stage_b_decisions = v1._stage_b_decisions(
        stage_a_decisions=decisions,
        stage_a_heldout=stage_a["summaries"][v1.HELDOUT],
        summaries=summaries[v1.HELDOUT],
    )
    stage_b = {
        "schema": "plan_aware_stage_b_predictor_substitution_v2",
        "row_count": len(rows),
        "source_specific_rows": True,
        "sources": list(sources),
        "summaries": summaries,
        **stage_b_decisions,
        "gate_receipt": _artifact_binding(
            gate_path, attempt_root=attempt_root, rows=None
        ),
        "gate_content_digest": gate["content_digest"],
        "helper_materialisation_custody": materialisation,
        "ledger": _artifact_binding(
            stage_b_path, attempt_root=attempt_root, rows=len(rows)
        ),
        "ranker_refit_or_recalibration": False,
        "predictor_training_steps": 0,
        "fresh_states_or_candidates": 0,
    }
    counters["predicted_latents_opened"] += 3_456
    counters["predictor_checkpoints_opened"] += 2
    counters["scientific_score_rows"] += len(rows)
    if not stage_b_decisions["proprioception_gate"]["pass"]:
        return stage_b, None

    stage_c_gate_path, stage_c_gate = _publish_v2_stage_c_gate(
        attempt_root=attempt_root,
        source_freeze=source_freeze,
        stage_b_gate_path=gate_path,
        evaluation_contract=stage_a["evaluation_contract"],
        stage_b_decisions=stage_b_decisions,
        heldout_summaries=summaries[v1.HELDOUT],
    )
    stage_c_helper = helper_runner(V2.PHASE1_STAGE_C_HELPER_MODE, attempt_root)
    stage_c_tensor_rows, stage_c_materialisation = _load_stage_c_helper_outputs(
        attempt_root=attempt_root,
        stage_b_gate_digest=str(gate["content_digest"]),
        stage_c_gate_digest=str(stage_c_gate["content_digest"]),
        helper_receipt=stage_c_helper,
    )
    all_stage_c_tensors = {**all_tensors, **stage_c_tensor_rows}
    stage_c_maps, stage_c_evidence, stage_c_summaries = v1._score_conditional_sources(
        attempt=v1_root,
        datasets=stage_a["datasets"],
        models=stage_a["models"],
        tensor_rows=all_stage_c_tensors,
        sources=v1.STAGE_C_SOURCE_IDS,
    )
    stage_c_rows = v1._conditional_ledger_rows(
        datasets=stage_a["datasets"],
        score_maps_by_role=stage_c_maps,
        evidence_by_role=stage_c_evidence,
        sources=v1.STAGE_C_SOURCE_IDS,
        schema="plan_aware_attribution_row_v1",
        matched_route_scores_by_role=score_maps,
        matched_condition="KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR",
    )
    if len(stage_c_rows) != V2.STAGE_ROW_AUTHORITY["stage_c_rows_if_executed"]:
        raise V2EvaluationError("Stage-C attribution cardinality drift")
    stage_c_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_c_rows"]["path"]
    _write_bytes_exclusive(stage_c_path, _canonical_jsonl(stage_c_rows))
    _pr_rows, pr_index = v1._load_prediction_index(
        attempt=v1_root,
        source_id="PR_PROPRIO_ROLLOUT",
        gate_digest=str(gate["content_digest"]),
    )
    fidelity_indexes = {
        "PR": pr_index,
        **{
            source: v1._load_prediction_index(
                attempt=v1_root,
                source_id=source,
                gate_digest=str(gate["content_digest"]),
                stage_c_gate_digest=str(stage_c_gate["content_digest"]),
            )[1]
            for source in v1.STAGE_C_SOURCE_IDS
        },
    }
    fidelity_rows, action_rows = v1._stage_c_fidelity_and_action_rows(
        attempt=v1_root,
        datasets=stage_a["datasets"],
        predecessor_tensor_rows=stage_a["tensor_rows"],
        prediction_indexes=fidelity_indexes,
    )
    if len(fidelity_rows) != V2.STAGE_ROW_AUTHORITY["stage_c_direct_fidelity_rows_if_executed"]:
        raise V2EvaluationError("Stage-C fidelity cardinality drift")
    if len(action_rows) != V2.STAGE_ROW_AUTHORITY["stage_c_action_sensitivity_rows_if_executed"]:
        raise V2EvaluationError("Stage-C action-sensitivity cardinality drift")
    fidelity_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
        "stage_c_direct_fidelity_rows"
    ]["path"]
    action_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
        "stage_c_action_sensitivity_rows"
    ]["path"]
    _write_bytes_exclusive(fidelity_path, _canonical_jsonl(fidelity_rows))
    _write_bytes_exclusive(action_path, _canonical_jsonl(action_rows))
    fidelity = v1._aggregate_stage_c_fidelity(fidelity_rows)
    action_sensitivity = v1._action_sensitivity_evidence(
        attempt=v1_root,
        rows=action_rows,
        prediction_indexes=fidelity_indexes,
    )
    stage_c_decisions = v1._stage_c_decisions(
        stage_b=stage_b,
        summaries=stage_c_summaries,
        action_sensitivity=action_sensitivity,
    )
    stage_c = {
        "schema": "plan_aware_stage_c_attribution_v2",
        "row_count": len(stage_c_rows),
        "sources": list(v1.STAGE_C_SOURCE_IDS),
        "summaries": stage_c_summaries,
        **stage_c_decisions,
        "route_score_changes": v1._aggregate_stage_c_route_score_changes(stage_c_rows),
        "direct_future_fidelity_h1_h3": fidelity,
        "direct_fidelity_rows": len(fidelity_rows),
        "direct_fidelity_ledger": _artifact_binding(
            fidelity_path, attempt_root=attempt_root, rows=len(fidelity_rows)
        ),
        "candidate_action_sensitivity": action_sensitivity,
        "action_sensitivity_ledger": _artifact_binding(
            action_path, attempt_root=attempt_root, rows=len(action_rows)
        ),
        "gate_receipt": _artifact_binding(
            stage_c_gate_path, attempt_root=attempt_root, rows=None
        ),
        "gate_content_digest": stage_c_gate["content_digest"],
        "helper_materialisation_custody": stage_c_materialisation,
        "ledger": _artifact_binding(
            stage_c_path, attempt_root=attempt_root, rows=len(stage_c_rows)
        ),
        "ranker_refit_or_recalibration": False,
        "predictor_training_steps": 0,
    }
    counters["predicted_latents_opened"] += 5_184
    counters["predictor_checkpoints_opened"] += 3
    counters["scientific_score_rows"] += (
        len(stage_c_rows) + len(fidelity_rows) + len(action_rows)
    )
    return stage_b, stage_c


def _candidate_selection_rows(
    *,
    stage_a_summaries: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    v1 = _v1()
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}

    def merge(group: Mapping[str, Any]) -> None:
        for role in v1.SPLIT_ROLES:
            for source_id, summary in group[role].items():
                population = summary["populations"][
                    v1.METRICS.ORACLE_VIABILITY_ADMISSIBLE
                ]
                for row in population["per_state"]:
                    identity = (role, str(row["state_id"]))
                    target = by_identity.setdefault(
                        identity,
                        {
                            "schema": (
                                "plan_aware_monotone_jepa_cost_v2."
                                "candidate_selection_row.v1"
                            ),
                            "split": role,
                            "state_id": str(row["state_id"]),
                            "family": str(row["family"]),
                            "selected_candidate_indices": {},
                        },
                    )
                    raw_selected = row["selected_candidate_index"]
                    selected = (
                        None if raw_selected is None else int(raw_selected)
                    )
                    selections = target["selected_candidate_indices"]
                    if source_id in selections and selections[source_id] != selected:
                        raise V2EvaluationError(
                            f"candidate selection changed across stages: {identity}:{source_id}"
                        )
                    selections[source_id] = selected

    merge(stage_a_summaries)
    if stage_b is not None:
        merge(stage_b["summaries"])
    if stage_c is not None:
        merge(stage_c["summaries"])
    rows = [
        by_identity[key]
        for key in sorted(
            by_identity,
            key=lambda item: (
                tuple(v1.SPLIT_ROLES).index(item[0]),
                v1.numeric_state_key(item[1]),
            ),
        )
    ]
    if len(rows) != 48:
        raise V2EvaluationError("candidate-selection row cardinality drift")
    return rows


_STAGE_B_DECISION_KEYS = (
    "predicted_gate",
    "incremental_over_kinematics",
    "proprioception_gate",
    "factorial_contrasts",
    "direct_proprioception_contrasts",
    "paired_bootstrap",
    "selected_candidate_changes",
    "predicted_source_absolute_preservation",
    "all_predicted_substitutions_fail_materially",
)
_STAGE_C_DECISION_KEYS = (
    "substitution_gate",
    "classification",
    "dependence_attribution",
    "route_metric_changes",
    "candidate_action_sensitivity",
    "control_only_explanation_excluded",
)


def _decision_component(
    *,
    stage_a_decisions: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
    scientific_decision: Mapping[str, Any],
) -> dict[str, Any]:
    return V2.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v2.gate_decisions.v1",
            "stage_a": copy.deepcopy(dict(stage_a_decisions)),
            "stage_b": (
                None
                if stage_b is None
                else {key: copy.deepcopy(stage_b[key]) for key in _STAGE_B_DECISION_KEYS}
            ),
            "stage_c": (
                None
                if stage_c is None
                else {key: copy.deepcopy(stage_c[key]) for key in _STAGE_C_DECISION_KEYS}
            ),
            "scientific_decision": copy.deepcopy(dict(scientific_decision)),
            "forced_stage_a_pass": False,
            "correction_replay_control_flow_used": False,
        }
    )


def _write_scientific_components(
    *,
    attempt_root: Path,
    attempt_id: str,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
    scientific_decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v1 = _v1()
    evaluation_lifecycle = V2.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v2.evaluation_lifecycle.v1",
            "heldout_opened_after_checkpoint_publication": True,
            "heldout_opened_after_evaluation_contract": True,
            "calibration_used_for_model_selection": False,
            "predictor_training_steps": 0,
            "raw_goal_cosine_executions": 0,
            "pass": True,
        }
    )
    evaluation_lifecycle_path = attempt_root / "receipts/evaluation.json"
    _write_json_exclusive(evaluation_lifecycle_path, evaluation_lifecycle)
    selection_rows = _candidate_selection_rows(
        stage_a_summaries=stage_a["summaries"],
        stage_b=stage_b,
        stage_c=stage_c,
    )
    selections = V2.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v2.candidate_selections.v1",
            "attempt_id": attempt_id,
            "rows": selection_rows,
            "row_count": len(selection_rows),
        }
    )
    selection_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
        "candidate_selections"
    ]["path"]
    _write_json_exclusive(selection_path, selections)

    gates = _decision_component(
        stage_a_decisions=stage_a["decisions"],
        stage_b=stage_b,
        stage_c=stage_c,
        scientific_decision=scientific_decision,
    )
    gates_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
        "gate_decisions"
    ]["path"]
    _write_json_exclusive(gates_path, gates)

    raw_comparisons = v1._matched_raw_cost_comparisons(
        stage_a["summaries"], stage_b
    )
    metrics = V2.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v2.scientific_metrics.v1",
            "attempt_id": attempt_id,
            "stage_a": copy.deepcopy(stage_a["summaries"]),
            "stage_a_decisions": copy.deepcopy(stage_a["decisions"]),
            "stage_b": copy.deepcopy(stage_b),
            "stage_c": copy.deepcopy(stage_c),
            "fit_optimization": copy.deepcopy(
                stage_a["training_receipt"]["fit_optimization"]
            ),
            "stage_a_raw_cost_rereduced": copy.deepcopy(stage_a["raw_reduction"]),
            "stage_a_raw_cost_matched_comparisons": raw_comparisons,
            "scientific_decision": copy.deepcopy(dict(scientific_decision)),
            "scientific_custody": {
                "training_smoke": _artifact_binding(
                    attempt_root / "receipts/training_smoke.json",
                    attempt_root=attempt_root,
                    rows=None,
                ),
                "training": _artifact_binding(
                    attempt_root / "receipts/training.json",
                    attempt_root=attempt_root,
                    rows=None,
                ),
                "evaluation_contract": _artifact_binding(
                    attempt_root / "receipts/evaluation_contract.json",
                    attempt_root=attempt_root,
                    rows=None,
                ),
                "evaluation_lifecycle": _artifact_binding(
                    evaluation_lifecycle_path,
                    attempt_root=attempt_root,
                    rows=None,
                ),
                "trained_route_ranker_checkpoints": copy.deepcopy(
                    stage_a["checkpoints"]
                ),
            },
            "historical_failed_output_metrics_opened": 0,
            "historical_failed_output_metrics_reused": 0,
            "raw_goal_cosine_inference_executions": 0,
        }
    )
    metrics_path = attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["metrics"]["path"]
    _write_json_exclusive(metrics_path, metrics)
    return selections, gates, metrics


def _scientific_artifact_bindings(
    *,
    attempt_root: Path,
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, authority in V2.SCIENTIFIC_ARTIFACT_AUTHORITY.items():
        required = authority["required"] is True
        if authority["required"] == "STAGE_B_EXECUTED":
            required = stage_b is not None
        elif authority["required"] == "STAGE_C_EXECUTED":
            required = stage_c is not None
        if not required:
            result[key] = None
            continue
        path = attempt_root / authority["path"]
        result[key] = _artifact_binding(
            path,
            attempt_root=attempt_root,
            rows=authority["rows"],
        )
    return result


def _runtime_file_counts(attempt_root: Path) -> tuple[int, int]:
    root_fd, _custody, _reopened = _require_attempt_io(attempt_root)

    def visit(directory_fd: int) -> tuple[int, int]:
        count = 0
        byte_count = 0
        with os.scandir(directory_fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise V2EvaluationError("attempt namespace contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    child_count, child_bytes = visit(child_fd)
                finally:
                    os.close(child_fd)
                count += child_count
                byte_count += child_bytes
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise V2EvaluationError("attempt namespace contains a hard link")
                count += 1
                byte_count += metadata.st_size
            else:
                raise V2EvaluationError("attempt namespace contains a special file")
        return count, byte_count

    return visit(root_fd)


def _scientific_component_values(
    *,
    attempt_root: Path,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
    evaluation_receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    selections: Mapping[str, Any],
    gates: Mapping[str, Any],
    runtime_storage: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "preexecution_receipt": _load_canonical_json(
            V2.runtime_path("preexecution_receipt", attempt_root)
        ),
        "foreground_runtime_import_custody": _load_canonical_json(
            V2.runtime_path(
                "foreground_runtime_import_custody", attempt_root
            )
        ),
        "attempt_root_fd_custody": _load_canonical_json(
            V2.runtime_path("attempt_root_fd_custody", attempt_root)
        ),
        "scientific_attempt_boundary_entry": _load_canonical_json(
            V2.runtime_path(
                "scientific_attempt_boundary_entry", attempt_root
            )
        ),
        "first_scientific_open_initiation": _load_canonical_json(
            V2.runtime_path(
                "first_scientific_open_initiation", attempt_root
            )
        ),
        "first_scientific_open_receipt": _load_canonical_json(
            V2.runtime_path("first_scientific_open_receipt", attempt_root)
        ),
        "training_smoke_receipt": copy.deepcopy(
            stage_a["training_smoke"]
        ),
        "training_receipt": copy.deepcopy(stage_a["training_receipt"]),
        "evaluation_contract": copy.deepcopy(stage_a["evaluation_contract"]),
        "evaluation_receipt": copy.deepcopy(dict(evaluation_receipt)),
        "metrics": copy.deepcopy(dict(metrics)),
        "candidate_selections": copy.deepcopy(dict(selections)),
        "gate_decisions": copy.deepcopy(dict(gates)),
        "runtime_storage_custody": copy.deepcopy(dict(runtime_storage)),
        "static_input_stage_a_custody": _load_canonical_json(
            V2.runtime_path("static_input_stage_a_custody", attempt_root)
        ),
        "stage_b_gate_receipt": None,
        "proprio_helper_manifest": None,
        "proprio_helper_receipt": None,
        "proprio_helper_exit_custody": None,
        "proprio_worker_lifecycle_manifest": None,
        "static_input_stage_b_helper_custody": None,
        "stage_c_gate_receipt": None,
        "stage_c_helper_manifest": None,
        "stage_c_helper_receipt": None,
        "stage_c_helper_exit_custody": None,
        "stage_c_worker_lifecycle_manifest": None,
        "static_input_stage_c_helper_custody": None,
    }
    if stage_b is not None:
        for key in (
            "stage_b_gate_receipt",
            "proprio_helper_manifest",
            "proprio_helper_receipt",
            "proprio_helper_exit_custody",
            "proprio_worker_lifecycle_manifest",
            "static_input_stage_b_helper_custody",
        ):
            values[key] = _load_canonical_json(
                attempt_root
                / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[key]["path"]
            )
    if stage_c is not None:
        for key in (
            "stage_c_gate_receipt",
            "stage_c_helper_manifest",
            "stage_c_helper_receipt",
            "stage_c_helper_exit_custody",
            "stage_c_worker_lifecycle_manifest",
            "static_input_stage_c_helper_custody",
        ):
            values[key] = _load_canonical_json(
                attempt_root
                / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[key]["path"]
            )
    if tuple(values) != V2.SCIENTIFIC_COMPONENT_KEYS:
        raise V2EvaluationError("scientific component order drift")
    return values


def _internal_environment() -> dict[str, str]:
    environment = {str(key): str(value) for key, value in os.environ.items()}
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"):
        environment.pop(key, None)
    environment.update(
        {str(key): str(value) for key, value in V2.PHASE_ENVIRONMENT_AUTHORITY[
            "numerical_thread_environment"
        ].items()}
    )
    return environment


def _process_group_members(process_group: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            closing = text.rfind(")")
            fields = text[closing + 2 :].split()
            pid = int(stat_path.parent.name)
            if closing < 0 or len(fields) < 20 or int(fields[2]) != process_group:
                continue
            argv = [
                item.decode("utf-8", errors="surrogateescape")
                for item in (stat_path.parent / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            rows.append(
                {
                    "pid": pid,
                    "start_time_ticks": int(fields[19]),
                    "argv": argv,
                }
            )
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return sorted(rows, key=lambda row: (row["pid"], row["start_time_ticks"]))


def _active_v2_role_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluator = str(V2.EVALUATOR_PATH)
    modes = set(V2.ALL_EVALUATOR_SUBCOMMAND_MODES)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [
                item.decode("utf-8", errors="strict")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if evaluator not in argv or not any(item in modes for item in argv):
                continue
            stat_text = (proc / "stat").read_text(encoding="ascii")
            closing = stat_text.rfind(")")
            fields = stat_text[closing + 2 :].split()
            rows.append(
                {
                    "pid": int(proc.name),
                    "start_time_ticks": int(fields[19]),
                    "argv": argv,
                }
            )
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    return sorted(rows, key=lambda row: (row["pid"], row["start_time_ticks"]))


def _producer_resource_rows(
    producer_identity: Mapping[str, Any], *, attempt_root: Path
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Conservatively observe residual resources owned by the exact child."""

    if _identity_is_live(producer_identity):
        raise V2EvaluationError("producer remains live during resource scan")
    pid = int(producer_identity["pid"])
    gpu: list[str] = []
    kfd: list[str] = []
    open_files: list[str] = []
    fd_root = Path("/proc") / str(pid) / "fd"
    if fd_root.exists():
        for entry in fd_root.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            row = f"pid={pid} fd={entry.name} target={target}"
            if target == "/dev/kfd":
                kfd.append(row)
            elif target.startswith(("/dev/dri/", "/dev/nvidia")):
                gpu.append(row)
            if target == str(attempt_root) or target.startswith(
                str(attempt_root) + os.sep
            ):
                open_files.append(row)
    locks: list[str] = []
    try:
        lock_lines = Path("/proc/locks").read_text(encoding="ascii").splitlines()
    except OSError:
        lock_lines = []
    for line in lock_lines:
        fields = line.split()
        if len(fields) > 4 and fields[4] == str(pid):
            locks.append(line)
    return tuple(sorted(set(rows)) for rows in (gpu, kfd, locks, open_files))


def _stream_bindings(
    paths: Mapping[str, str], payloads: Mapping[str, bytes]
) -> dict[str, dict[str, Any]]:
    return {
        key: V2.build_stream_binding(paths[key], payloads[key])
        for key in ("stdout", "stderr", "traceback", "exception", "last_stage")
    }


def _register_helper_worker_pid(pid: int) -> None:
    with _ACTIVE_HELPER_WORKER_PIDS_LOCK:
        if pid in _ACTIVE_HELPER_WORKER_PIDS:
            raise V2EvaluationError("duplicate active helper-worker pid")
        _ACTIVE_HELPER_WORKER_PIDS.add(pid)


def _unregister_helper_worker_pid(pid: int) -> None:
    with _ACTIVE_HELPER_WORKER_PIDS_LOCK:
        _ACTIVE_HELPER_WORKER_PIDS.discard(pid)


def _observe_helper_group_members(
    *, helper_pid: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with _ACTIVE_HELPER_WORKER_PIDS_LOCK:
        members = _process_group_members(helper_pid)
        supervised = set(_ACTIVE_HELPER_WORKER_PIDS)
        unbound = [
            dict(row)
            for row in members
            if row["pid"] != helper_pid and row["pid"] not in supervised
        ]
    return members, unbound


def _require_no_active_helper_worker_pids() -> None:
    with _ACTIVE_HELPER_WORKER_PIDS_LOCK:
        if _ACTIVE_HELPER_WORKER_PIDS:
            raise V2EvaluationError(
                "helper worker registry is not empty after bounded joins"
            )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> list[str]:
    actions: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return actions
    actions.append("SIGTERM_PROCESS_GROUP")
    try:
        process.wait(timeout=V2.PROCESS_LIFECYCLE_POLICY["term_grace_seconds"])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            actions.append("SIGKILL_PROCESS_GROUP")
        try:
            process.wait(
                timeout=V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
            )
        except subprocess.TimeoutExpired:
            pass
    return actions


def _terminate_nested_worker(process: subprocess.Popen[bytes]) -> list[str]:
    """Bound one worker while leaving its enclosing helper group intact."""

    actions: list[str] = []
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return actions
    actions.append("SIGTERM_WORKER_PID")
    try:
        process.wait(timeout=V2.PROCESS_LIFECYCLE_POLICY["term_grace_seconds"])
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        else:
            actions.append("SIGKILL_WORKER_PID")
        try:
            process.wait(
                timeout=V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
            )
        except subprocess.TimeoutExpired:
            pass
    return actions


def _bounded_process_drain(
    process: subprocess.Popen[bytes], *, timeout_seconds: float
) -> tuple[bytes, bytes, bool]:
    """Drain once within a fixed bound; never wait on descendant-held pipes."""

    try:
        stdout, stderr = process.communicate(timeout=max(0.0, timeout_seconds))
        return stdout, stderr, True
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output if isinstance(exc.output, bytes) else b""
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            process.wait(
                timeout=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                )
            )
        except subprocess.TimeoutExpired:
            pass
        return stdout, stderr, False


def _cleanup_remaining_process_group(process_group: int) -> list[str]:
    if not _process_group_members(process_group):
        return []
    actions: list[str] = []
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return actions
    actions.append("SIGTERM_PROCESS_GROUP")
    deadline = time.monotonic() + float(
        V2.PROCESS_LIFECYCLE_POLICY["term_grace_seconds"]
    )
    while _process_group_members(process_group) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_members(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            actions.append("SIGKILL_PROCESS_GROUP")
        deadline = time.monotonic() + float(
            V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
        )
        while _process_group_members(process_group) and time.monotonic() < deadline:
            time.sleep(0.01)
    return actions


def _worker_environment(*, cpu: bool, root_fd: int) -> dict[str, str]:
    environment = _internal_environment()
    authority = V2.build_frozen_helper_runtime_import_authority()[
        "environment_policy"
    ]["per_interpreter"]["cpu_child" if cpu else "gpu_child"]
    environment["PYTHONNOUSERSITE"] = "1"
    environment["VIRTUAL_ENV"] = str(authority["VIRTUAL_ENV"])
    existing_path = environment.get("PATH", "")
    environment["PATH"] = (
        str(authority["PATH_prepend"])
        + (os.pathsep + existing_path if existing_path else "")
    )
    environment[V2.ATTEMPT_ROOT_FD_ENVIRONMENT_NAME] = str(root_fd)
    return environment


def _structured_failure(
    *, label: str, message: str, traceback_bytes: bytes
) -> dict[str, str]:
    return {
        "type": label,
        "message": message,
        "traceback_sha256": hashlib.sha256(traceback_bytes).hexdigest(),
    }


def _persist_phase1_terminal_source_streams(
    *, error: BaseException | None, traceback_bytes: bytes
) -> None:
    if _ACTIVE_ATTEMPT_ROOT is None:
        return
    attempt_root = _ACTIVE_ATTEMPT_ROOT
    _write_bytes_exclusive(
        V2.runtime_path("phase_1_producer_traceback", attempt_root),
        traceback_bytes,
    )
    exception_payload = b""
    if error is not None:
        exception_payload = V2.canonical_json_bytes(
            _structured_failure(
                label=type(error).__name__,
                message=str(error),
                traceback_bytes=traceback_bytes,
            )
        ) + b"\n"
    _write_bytes_exclusive(
        V2.runtime_path("phase_1_producer_exception", attempt_root),
        exception_payload,
    )


def _worker_exit_custody_binding(
    exit_custody: Mapping[str, Any], *, attempt_root: Path
) -> dict[str, Any]:
    paths = V2.phase1_worker_runtime_paths(
        mode=str(exit_custody["mode"]),
        attempt_root=attempt_root,
        parent_helper_process_identity=exit_custody[
            "parent_helper_process_identity"
        ],
        state_index=exit_custody.get("state_index"),
        state_id=exit_custody.get("state_id"),
        source_id=exit_custody.get("source_id"),
        ablation=exit_custody.get("ablation"),
    )
    return _artifact_binding(
        attempt_root / paths["exit_custody"],
        attempt_root=attempt_root,
        rows=None,
    )


def _persist_worker_failure_manifest(
    *,
    helper_mode: str,
    attempt_root: Path,
    parent_helper_process_identity: Mapping[str, Any],
    worker_exit_custodies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = [
        _worker_exit_custody_binding(row, attempt_root=attempt_root)
        for row in worker_exit_custodies
    ]
    manifest = V2.build_phase1_worker_failure_manifest(
        helper_mode=helper_mode,
        attempt_root=attempt_root,
        parent_helper_process_identity=parent_helper_process_identity,
        worker_exit_custodies=worker_exit_custodies,
        worker_exit_custody_bindings=bindings,
    )
    _write_json_exclusive(
        attempt_root / V2.phase1_worker_failure_manifest_path(helper_mode),
        manifest,
    )
    return V2.validate_phase1_worker_failure_manifest(manifest)


def _run_phase1_worker_process(
    *,
    mode: str,
    attempt_root: Path,
    parent_helper_process_identity: Mapping[str, Any],
    state_index: int | None = None,
    state_id: str | None = None,
    source_id: str | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    command = V2.expected_phase1_worker_argv(
        mode,
        attempt_root,
        state_index=state_index,
        source_id=source_id,
        ablation=ablation,
    )
    cpu = mode == V2.PHASE1_CONTEXT_STATE_WORKER_MODE
    executable = (
        V2.CPU_WORKER_PYTHON_EXECUTABLE if cpu else V2.PYTHON_EXECUTABLE
    )
    root_fd, root_custody, _reopened = _require_attempt_io(attempt_root)
    environment = _worker_environment(cpu=cpu, root_fd=root_fd)
    paths = V2.phase1_worker_runtime_paths(
        mode=mode,
        attempt_root=attempt_root,
        parent_helper_process_identity=parent_helper_process_identity,
        state_index=state_index,
        state_id=state_id,
        source_id=source_id,
        ablation=ablation,
    )
    started = time.monotonic_ns()
    deadline = time.monotonic() + float(
        V2.PROCESS_LIFECYCLE_POLICY[
            "nested_worker_wall_clock_timeout_seconds"
        ]
    )
    process: subprocess.Popen[bytes] | None = None
    identity: dict[str, Any] | None = None
    child_root_binding: dict[str, Any] | None = None
    stdout = b""
    stderr = b""
    local_traceback = b""
    deadline_exceeded = False
    communication_attempted = False
    communicate_completed = False
    process_reaped = False
    cleanup_actions: list[str] = []
    cleanup_errors: list[BaseException] = []
    registered_pid: int | None = None
    try:
        if os.getpgrp() != int(parent_helper_process_identity["pid"]):
            raise V2EvaluationError(
                "worker parent does not own the shared helper process group"
            )
        with _ACTIVE_HELPER_WORKER_PIDS_LOCK:
            process = subprocess.Popen(
                command,
                cwd=V2.REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=False,
                pass_fds=(root_fd,),
            )
            registered_pid = process.pid
            _register_helper_worker_pid(process.pid)
        communication_attempted = True
        identity = _process_identity_for_pid(
            process.pid,
            expected_argv=command,
            expected_executable=executable,
        )
        inherited = V2.build_inherited_attempt_root_fd_custody(
            attempt_root_fd_custody=root_custody,
            parent_process_identity=parent_helper_process_identity,
            child_process_identity=identity,
            inherited_root_fd=root_fd,
            fd_was_passed_explicitly=True,
            environment_fd_value=str(root_fd),
        )
        child_root_binding = V2.build_child_attempt_root_fd_binding(
            attempt_root_fd_custody=root_custody,
            inherited_attempt_root_fd_custody=inherited,
        )
        try:
            stdout, stderr = process.communicate(
                timeout=max(0.0, deadline - time.monotonic())
            )
            communicate_completed = True
        except subprocess.TimeoutExpired:
            deadline_exceeded = True
            cleanup_actions.extend(_terminate_nested_worker(process))
            stdout, stderr, communicate_completed = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
    except BaseException:
        local_traceback = traceback.format_exc().encode("utf-8")
        if process is not None and process.poll() is None:
            cleanup_actions.extend(_terminate_nested_worker(process))
        if process is not None:
            communication_attempted = True
            stdout, stderr, communicate_completed = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
    finally:
        if process is not None and process.poll() is None:
            cleanup_actions.extend(_terminate_nested_worker(process))
        if process is not None:
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            process_reaped = process.poll() is not None
    ended = time.monotonic_ns()
    popen_succeeded = process is not None
    spawned_pid = None if process is None else process.pid
    returncode = None if process is None else process.returncode
    process_absent = (
        True
        if spawned_pid is None
        else process_reaped and not Path(f"/proc/{spawned_pid}").exists()
    )
    parent_group = int(parent_helper_process_identity["pid"])
    if popen_succeeded:
        remaining, unbound_remaining = _observe_helper_group_members(
            helper_pid=parent_group
        )
    else:
        remaining, unbound_remaining = [], []
    if registered_pid is not None:
        _unregister_helper_worker_pid(registered_pid)
        registered_pid = None
    worker_receipt: dict[str, Any] | None = None
    if (
        identity is not None
        and child_root_binding is not None
        and returncode == 0
        and not cleanup_actions
        and not local_traceback
    ):
        try:
            parsed = _canonical_object_from_bytes(
                stdout, label="worker stdout"
            )
            worker_receipt = V2.validate_phase1_worker_receipt(parsed)
        except BaseException:
            local_traceback = traceback.format_exc().encode("utf-8")
    static_audit_stream = (
        None
        if worker_receipt is None
        else worker_receipt["static_input_audit_stream"]
    )
    failed = (
        not popen_succeeded
        or identity is None
        or child_root_binding is None
        or returncode != 0
        or deadline_exceeded
        or cleanup_actions != []
        or unbound_remaining != []
        or bool(local_traceback)
        or not communicate_completed
        or not process_reaped
        or not process_absent
        or worker_receipt is None
    )
    traceback_bytes = (
        b""
        if not failed
        else (
            local_traceback
            or stderr
            or b"worker process failed without stderr\n"
        )
    )
    _write_bytes_exclusive(attempt_root / paths["stdout"], stdout)
    _write_bytes_exclusive(attempt_root / paths["stderr"], stderr)
    _write_bytes_exclusive(attempt_root / paths["traceback"], traceback_bytes)
    exception = (
        _structured_failure(
            label=(
                "V2Phase1WorkerIdentityFailure"
                if popen_succeeded and identity is None
                else "V2Phase1WorkerFailure"
            ),
            message=(
                f"{mode} worker failed or emitted invalid custody; "
                f"returncode={returncode}"
            ),
            traceback_bytes=traceback_bytes,
        )
        if failed
        else None
    )
    if failed:
        worker_receipt = None
    exception_binding = None
    if exception is not None:
        exception_bytes = V2.canonical_json_bytes(exception) + b"\n"
        _write_bytes_exclusive(
            attempt_root / paths["exception"], exception_bytes
        )
        exception_binding = V2.build_stream_binding(
            paths["exception"], exception_bytes
        )
    identity_error = (
        exception if popen_succeeded and identity is None else None
    )
    exit_custody = V2.build_phase1_worker_exit_custody(
        worker_receipt=worker_receipt,
        mode=mode,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        parent_helper_process_identity=parent_helper_process_identity,
        worker_process_identity=identity,
        spawned_pid=spawned_pid,
        parent_helper_process_group_id=(
            parent_group if popen_succeeded else None
        ),
        spawned_process_group_id=(
            parent_group if popen_succeeded else None
        ),
        popen_succeeded=popen_succeeded,
        identity_observation_error=identity_error,
        process_absent_after_exit=process_absent,
        static_input_audit_stream=static_audit_stream,
        state_index=state_index,
        state_id=state_id,
        source_id=source_id,
        ablation=ablation,
        stdout_binding=V2.build_stream_binding(paths["stdout"], stdout),
        stderr_binding=V2.build_stream_binding(paths["stderr"], stderr),
        traceback_binding=V2.build_stream_binding(
            paths["traceback"], traceback_bytes
        ),
        structured_exception=exception,
        structured_exception_binding=exception_binding,
        last_stage=("FAILED" if failed else "COMPLETE"),
        returncode=returncode,
        termination_signal=(
            -returncode
            if type(returncode) is int and returncode < 0
            else None
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        environment_names=sorted(environment),
        environment_sha256=V2.canonical_json_sha256(environment),
        cleanup_actions=cleanup_actions,
        process_group_members_after_cleanup=remaining,
        unbound_process_group_members_after_cleanup=unbound_remaining,
        deadline_exceeded=deadline_exceeded,
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
    )
    V2.validate_phase1_worker_exit_custody(exit_custody)
    _write_json_exclusive(
        attempt_root / paths["exit_custody"], exit_custody
    )
    exit_custody_binding = _artifact_binding(
        attempt_root / paths["exit_custody"],
        attempt_root=attempt_root,
        rows=None,
    )
    if exit_custody["pass"] is not True or worker_receipt is None:
        raise V2Phase1WorkerProcessError(
            (
                "Phase-1 worker failed: "
                f"{mode}:{state_index}:{source_id}:{ablation}"
            ),
            exit_custody=exit_custody,
            exit_custody_binding=exit_custody_binding,
        )
    return exit_custody


def _run_phase1_helper_process(mode: str, attempt_root: Path) -> dict[str, Any]:
    if mode not in (
        V2.PHASE1_PROPRIO_HELPER_MODE,
        V2.PHASE1_STAGE_C_HELPER_MODE,
    ):
        raise V2EvaluationError("invalid Phase-1 helper mode")
    command = V2.expected_internal_argv(mode, attempt_root)
    observer_identity = _process_identity()
    environment = _internal_environment()
    root_fd, root_custody, _reopened = _require_attempt_io(attempt_root)
    environment[V2.ATTEMPT_ROOT_FD_ENVIRONMENT_NAME] = str(root_fd)
    timeout_seconds = float(
        V2.PROCESS_LIFECYCLE_POLICY[
            "scientific_helper_wall_clock_timeout_seconds"
        ]
    )
    started_monotonic_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    process: subprocess.Popen[bytes] | None = None
    identity: dict[str, Any] | None = None
    child_root_binding: dict[str, Any] | None = None
    stdout = b""
    stderr = b""
    local_traceback = b""
    deadline_exceeded = False
    communication_attempted = False
    communicate_completed = False
    process_reaped = False
    cleanup_actions: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=V2.REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(root_fd,),
        )
        identity = _process_identity_for_pid(
            process.pid, expected_argv=command
        )
        inherited = V2.build_inherited_attempt_root_fd_custody(
            attempt_root_fd_custody=root_custody,
            parent_process_identity=observer_identity,
            child_process_identity=identity,
            inherited_root_fd=root_fd,
            fd_was_passed_explicitly=True,
            environment_fd_value=str(root_fd),
        )
        child_root_binding = V2.build_child_attempt_root_fd_binding(
            attempt_root_fd_custody=root_custody,
            inherited_attempt_root_fd_custody=inherited,
        )
        communication_attempted = True
        remaining = max(0.0, deadline - time.monotonic())
        try:
            stdout, stderr = process.communicate(timeout=remaining)
            communicate_completed = True
        except subprocess.TimeoutExpired:
            deadline_exceeded = True
            cleanup_actions.extend(_terminate_process_group(process))
            stdout, stderr, communicate_completed = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
    except BaseException:
        local_traceback = traceback.format_exc().encode("utf-8")
        if process is not None and process.poll() is None:
            cleanup_actions.extend(_terminate_process_group(process))
        if process is not None:
            stdout, stderr, communicate_completed = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
            communication_attempted = True
    finally:
        if process is not None and process.poll() is None:
            cleanup_actions.extend(_terminate_process_group(process))
        if process is not None:
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            process_reaped = process.poll() is not None
    if process is not None:
        cleanup_actions.extend(_cleanup_remaining_process_group(process.pid))
    ended_monotonic_ns = time.monotonic_ns()
    prefix = (
        "proprio_helper"
        if mode == V2.PHASE1_PROPRIO_HELPER_MODE
        else "stage_c_helper"
    )
    stdout_path = V2.runtime_path(f"{prefix}_stdout", attempt_root)
    stderr_path = V2.runtime_path(f"{prefix}_stderr", attempt_root)
    traceback_path = V2.runtime_path(f"{prefix}_traceback", attempt_root)
    popen_succeeded = process is not None
    spawned_pid = None if process is None else process.pid
    returncode = None if process is None else process.returncode
    remaining = [] if process is None else _process_group_members(process.pid)
    process_absent = (
        True
        if spawned_pid is None
        else process_reaped and not Path(f"/proc/{spawned_pid}").exists()
    )
    helper_receipt: dict[str, Any] | None = None
    if (
        identity is not None
        and child_root_binding is not None
        and returncode == 0
        and not cleanup_actions
        and not remaining
        and not local_traceback
    ):
        try:
            value = _canonical_object_from_bytes(
                stdout, label="Phase-1 helper stdout"
            )
            helper_receipt = V2.validate_phase1_helper_receipt(value)
        except BaseException:
            local_traceback = traceback.format_exc().encode("utf-8")
    worker_failure_manifest = None
    worker_failure_manifest_binding = None
    failure_manifest_relative = V2.phase1_worker_failure_manifest_path(mode)
    try:
        worker_failure_manifest = _load_canonical_json(
            attempt_root / failure_manifest_relative
        )
    except FileNotFoundError:
        pass
    except BaseException:
        local_traceback = traceback.format_exc().encode("utf-8")
        worker_failure_manifest = None
    else:
        worker_failure_manifest = (
            V2.validate_phase1_worker_failure_manifest(
                worker_failure_manifest
            )
        )
        worker_failure_manifest_binding = _artifact_binding(
            attempt_root / failure_manifest_relative,
            attempt_root=attempt_root,
            rows=None,
        )
    failed = (
        not popen_succeeded
        or identity is None
        or child_root_binding is None
        or returncode != 0
        or deadline_exceeded
        or cleanup_actions != []
        or remaining != []
        or bool(local_traceback)
        or not communicate_completed
        or not process_reaped
        or not process_absent
        or helper_receipt is None
    )
    traceback_bytes = (
        b""
        if not failed
        else (
            local_traceback
            or stderr
            or b"helper process failed without stderr\n"
        )
    )
    _write_bytes_exclusive(stdout_path, stdout)
    _write_bytes_exclusive(stderr_path, stderr)
    _write_bytes_exclusive(traceback_path, traceback_bytes)
    terminated_by = (
        -returncode if type(returncode) is int and returncode < 0 else None
    )
    exception = (
        _structured_failure(
            label=(
                "V2Phase1HelperIdentityFailure"
                if identity is None
                else "V2Phase1HelperFailure"
            ),
            message=(
                f"{mode} failed or emitted an invalid canonical receipt; "
                f"returncode={returncode}"
            ),
            traceback_bytes=traceback_bytes,
        )
        if failed
        else None
    )
    if failed:
        helper_receipt = None
    else:
        _write_json_exclusive(
            V2.runtime_path(f"{prefix}_receipt", attempt_root),
            helper_receipt,
        )
    identity_error = exception if identity is None else None
    environment_names = sorted(environment)
    exit_custody = V2.build_phase1_helper_exit_custody(
        mode=mode,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        process_identity=identity,
        spawned_pid=spawned_pid,
        spawned_process_group_id=spawned_pid,
        popen_succeeded=popen_succeeded,
        identity_observation_error=identity_error,
        observer_process_identity=observer_identity,
        process_absent_after_exit=process_absent,
        static_input_audit_stream=(
            None
            if helper_receipt is None
            else helper_receipt["static_input_audit_stream"]
        ),
        helper_receipt=helper_receipt,
        worker_failure_manifest=worker_failure_manifest,
        worker_failure_manifest_binding=worker_failure_manifest_binding,
        stdout_binding=V2.build_stream_binding(
            V2.RUNTIME_PATHS[f"{prefix}_stdout"], stdout
        ),
        stderr_binding=V2.build_stream_binding(
            V2.RUNTIME_PATHS[f"{prefix}_stderr"], stderr
        ),
        traceback_binding=V2.build_stream_binding(
            V2.RUNTIME_PATHS[f"{prefix}_traceback"], traceback_bytes
        ),
        structured_exception=exception,
        last_stage=("FAILED" if failed else "COMPLETE"),
        returncode=returncode,
        termination_signal=terminated_by,
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        environment_names=environment_names,
        environment_sha256=V2.canonical_json_sha256(environment),
        cleanup_actions=cleanup_actions,
        process_group_members_after_cleanup=remaining,
        deadline_exceeded=deadline_exceeded,
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
    )
    exit_path = V2.runtime_path(f"{prefix}_exit_custody", attempt_root)
    _write_json_exclusive(exit_path, exit_custody)
    V2.validate_phase1_helper_exit_custody(exit_custody)
    if exit_custody["pass"] is not True or helper_receipt is None:
        tail = stderr[-8_000:].decode("utf-8", errors="replace")
        raise V2EvaluationError(
            f"Phase-1 helper failed ({returncode}): {mode}: {tail}"
        )
    return helper_receipt


def _write_evaluation_lifecycle_receipt(attempt_root: Path) -> None:
    # The exact receipt is written by _write_scientific_components.  Keeping
    # this assertion separate makes the checkpoint validator's prerequisite
    # explicit before Phase 2 is ever launched.
    try:
        _read_attempt_bytes(attempt_root / "receipts/evaluation.json")
    except FileNotFoundError:
        return
    raise V2EvaluationError("evaluation lifecycle receipt already exists")


def capture_phase_1_terminal_custody(
    attempt_root: Path, supervisor_observation_content_digest: str
) -> dict[str, Any]:
    identity = _process_identity()
    expected = V2.expected_internal_argv(
        V2.PHASE1_TERMINAL_CUSTODY_MODE,
        attempt_root,
        supervisor_observation_content_digest=(
            supervisor_observation_content_digest
        ),
    )
    if identity["argv"] != expected or identity["cwd"] != str(V2.REPO_ROOT):
        raise V2EvaluationError("Phase-1 terminal attestor identity drift")
    custody = V2.build_phase1_terminal_custody_from_supervisor_bundle(
        attempt_root=attempt_root,
        supervisor_observation_content_digest=(
            supervisor_observation_content_digest
        ),
        terminal_attestor_process_identity=identity,
    )
    custody_binding = V2.write_phase1_terminal_custody_exclusive_fsync(
        phase1_custody=custody,
        terminal_attestor_process_identity=identity,
    )
    if custody["pass"] is not True:
        return custody
    receipt, _binding = (
        V2.write_phase1_terminal_publication_receipt_exclusive_fsync(
            phase1_custody=custody,
            phase1_custody_binding=custody_binding,
            terminal_attestor_process_identity=identity,
        )
    )
    return receipt


def capture_pre_root_phase_1_failure_custody(
    custody_path: Path, supervisor_observation_content_digest: str
) -> dict[str, Any]:
    identity = _process_identity()
    expected = V2.expected_pre_root_phase1_failure_custody_argv(
        supervisor_observation_content_digest
    )
    if (
        identity["argv"] != expected
        or identity["cwd"] != str(V2.REPO_ROOT)
        or custody_path.absolute() != V2.PRE_ROOT_FAILURE_CUSTODY_PATH
    ):
        raise V2EvaluationError("pre-root Phase-1 attestor identity drift")
    observation, _payloads = (
        V2.load_pre_root_phase1_supervisor_observation_bundle_anchored(
            supervisor_observation_content_digest=(
                supervisor_observation_content_digest
            ),
            terminal_attestor_process_identity=identity,
        )
    )
    custody = V2.build_pre_root_phase1_failure_custody(
        supervisor_observation=observation,
        terminal_attestor_process_identity=identity,
    )
    V2.write_pre_root_phase1_failure_custody_exclusive_fsync(
        custody=custody
    )
    return custody


def _run_terminal_attestor(argv: Sequence[str]) -> dict[str, Any]:
    started = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    identity: dict[str, Any] | None = None
    identity_error: dict[str, Any] | None = None
    stdout = b""
    stderr = b""
    communication_attempted = False
    communicate_completed = False
    cleanup_actions: list[str] = []
    primary_error: BaseException | None = None
    failure_stage = "TERMINAL_ATTESTOR_IDENTITY_OR_LIFECYCLE"
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=V2.REPO_ROOT,
            env=_internal_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except BaseException as exc:
        lifecycle = {
            "popen_succeeded": False,
            "spawned_pid": None,
            "spawned_process_group_id": None,
            "process_identity": None,
            "identity_observation_error": None,
            "returncode": None,
            "termination_signal": None,
            "started_monotonic_ns": started,
            "ended_monotonic_ns": time.monotonic_ns(),
            "communication_attempted": False,
            "communicate_completed": False,
            "process_reaped": False,
            "cleanup_actions": [],
            "process_group_members_after_cleanup": [],
            "process_absent_after_exit": False,
        }
        raise V2TerminalAttestorProcessError(
            "Phase-1 terminal attestor Popen failed",
            failure_stage="TERMINAL_ATTESTOR_POPEN",
            lifecycle=lifecycle,
            primary_error=exc,
        ) from exc

    try:
        identity = _process_identity_for_pid(
            process.pid, expected_argv=argv
        )
    except BaseException as exc:
        primary_error = exc
        identity_error = _terminal_supervisor_exception(exc)

    if primary_error is None:
        communication_attempted = True
        try:
            stdout, stderr = process.communicate()
            communicate_completed = True
        except BaseException as exc:
            primary_error = exc
    if primary_error is not None:
        try:
            for action in _terminate_process_group(process):
                if action not in cleanup_actions:
                    cleanup_actions.append(action)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        try:
            drained_stdout, drained_stderr, drained = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
            stdout += drained_stdout
            stderr += drained_stderr
            communication_attempted = True
            communicate_completed = drained
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
    try:
        for action in _cleanup_remaining_process_group(process.pid):
            if action not in cleanup_actions:
                cleanup_actions.append(action)
    except BaseException as cleanup_error:
        cleanup_errors.append(cleanup_error)
    members = _process_group_members(process.pid)
    returncode = process.returncode
    process_reaped = returncode is not None
    termination_signal = (
        -returncode
        if type(returncode) is int and returncode < 0
        else None
    )
    process_absent = (
        identity is not None
        and process_reaped
        and not _identity_is_live(identity)
        and members == []
    )
    lifecycle = {
        "popen_succeeded": True,
        "spawned_pid": process.pid,
        "spawned_process_group_id": process.pid,
        "process_identity": identity,
        "identity_observation_error": identity_error,
        "returncode": returncode,
        "termination_signal": termination_signal,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": time.monotonic_ns(),
        "communication_attempted": communication_attempted,
        "communicate_completed": communicate_completed,
        "process_reaped": process_reaped,
        "cleanup_actions": cleanup_actions,
        "process_group_members_after_cleanup": members,
        "process_absent_after_exit": process_absent,
    }
    if primary_error is not None or cleanup_errors or cleanup_actions or members:
        if primary_error is not None:
            error = primary_error
            trailing_errors = cleanup_errors
        elif cleanup_errors:
            error = cleanup_errors[0]
            trailing_errors = cleanup_errors[1:]
        else:
            error = V2EvaluationError(
                "Phase-1 terminal attestor required forced cleanup"
            )
            trailing_errors = []
        raise V2TerminalAttestorProcessError(
            "Phase-1 terminal attestor lifecycle failed",
            failure_stage=failure_stage,
            lifecycle=lifecycle,
            primary_error=error,
            cleanup_errors=trailing_errors,
        ) from error
    if returncode != 0:
        error = V2EvaluationError(
            "Phase-1 terminal custody persistence failed: "
            + stderr.decode("utf-8", errors="replace")
        )
        raise V2TerminalAttestorProcessError(
            str(error),
            failure_stage="TERMINAL_CUSTODY_PERSISTENCE",
            lifecycle=lifecycle,
            primary_error=error,
        ) from error
    if stderr != b"":
        error = V2EvaluationError(
            "Phase-1 terminal attestor emitted unexpected stderr"
        )
        raise V2TerminalAttestorProcessError(
            str(error),
            failure_stage="TERMINAL_ATTESTOR_OUTPUT_VALIDATION",
            lifecycle=lifecycle,
            primary_error=error,
        ) from error
    try:
        return _canonical_object_from_bytes(
            stdout, label="terminal attestor stdout"
        )
    except BaseException as exc:
        raise V2TerminalAttestorProcessError(
            "Phase-1 terminal attestor output validation failed",
            failure_stage="TERMINAL_ATTESTOR_OUTPUT_VALIDATION",
            lifecycle=lifecycle,
            primary_error=exc,
        ) from exc


def _pre_root_last_stage_payload(
    *, supervisor_identity: Mapping[str, Any], last_stage: str
) -> bytes:
    value = V2.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v2."
                "pre_root_supervisor_last_stage.v1"
            ),
            "stage": last_stage,
            "supervisor_pid": supervisor_identity["pid"],
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    return V2.canonical_json_bytes(value) + b"\n"


def _persist_rooted_phase1_supervisor_failure(
    *,
    attempt_id: str,
    prelaunch_namespace_custody: Mapping[str, Any],
    prelaunch_handoff_memfd_custody: Mapping[str, Any],
    supervisor_process_identity: Mapping[str, Any],
    producer_process_identity: Mapping[str, Any] | None,
    attempt_root_fd_custody: Mapping[str, Any] | None,
    supervisor_root_reopen_custody: Mapping[str, Any] | None,
    producer_environment_names: Sequence[str],
    producer_environment_sha256: str,
    supervisor_environment_names: Sequence[str],
    supervisor_environment_sha256: str,
    process: subprocess.Popen[bytes],
    captured_stdout: bytes,
    captured_stderr: bytes,
    failure_stage: str,
    failure_errors: Sequence[BaseException],
    supervisor_started_monotonic_ns: int,
    popen_started_monotonic_ns: int,
    cleanup_actions: Sequence[str],
    communication_attempted: bool,
    communicate_completed: bool,
    terminal_attestor_lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Durably classify every post-root technical failure as no-retry."""

    errors = list(failure_errors)
    if not errors:
        errors.append(V2EvaluationError("rooted Phase-1 failure lacked cause"))
    actions = list(cleanup_actions)
    try:
        for action in _cleanup_remaining_process_group(process.pid):
            if action not in actions:
                actions.append(action)
    except BaseException as cleanup_error:
        errors.append(cleanup_error)
    returncode = process.poll()
    process_reaped = returncode is not None
    termination_signal = (
        -returncode
        if type(returncode) is int and returncode < 0
        else None
    )
    group_members = _process_group_members(process.pid)
    active_roles = _active_v2_role_rows()
    attempt_root = Path(prelaunch_namespace_custody["attempt_root"])
    if producer_process_identity is not None and not _identity_is_live(
        producer_process_identity
    ):
        gpu, kfd, locks, open_files = _producer_resource_rows(
            producer_process_identity, attempt_root=attempt_root
        )
    else:
        gpu, kfd, locks, open_files = [], [], [], []
    failure_ended = time.monotonic_ns()
    post_cleanup = V2.observe_phase1_post_cleanup_start_namespace(
        attempt_id=attempt_id,
        supervisor_process_identity=supervisor_process_identity,
    )
    exception_rows = [
        {
            "stage": failure_stage,
            "exception": _terminal_supervisor_exception(error),
        }
        for error in errors
    ]
    observation = V2.observe_rooted_phase1_supervisor_failure(
        attempt_id=attempt_id,
        prelaunch_namespace_custody=prelaunch_namespace_custody,
        prelaunch_handoff_memfd_custody=(
            prelaunch_handoff_memfd_custody
        ),
        supervisor_process_identity=supervisor_process_identity,
        producer_process_identity=producer_process_identity,
        attempt_root_fd_custody=attempt_root_fd_custody,
        supervisor_root_reopen_custody=(
            supervisor_root_reopen_custody
        ),
        producer_environment_names=producer_environment_names,
        producer_environment_sha256=producer_environment_sha256,
        supervisor_environment_names=supervisor_environment_names,
        supervisor_environment_sha256=supervisor_environment_sha256,
        spawned_pid=process.pid,
        spawned_process_group_id=process.pid,
        captured_stdout=captured_stdout,
        captured_stderr=captured_stderr,
        failure_stage=failure_stage,
        failure_exceptions=exception_rows,
        returncode=returncode,
        termination_signal=termination_signal,
        supervisor_started_monotonic_ns=supervisor_started_monotonic_ns,
        popen_started_monotonic_ns=popen_started_monotonic_ns,
        ended_monotonic_ns=failure_ended,
        cleanup_actions=actions,
        process_group_members_after_cleanup=group_members,
        active_v2_role_rows_after_cleanup=active_roles,
        gpu_process_rows_after_cleanup=gpu,
        kfd_rows_after_cleanup=kfd,
        lock_rows_after_cleanup=locks,
        open_file_rows_after_cleanup=open_files,
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
        post_cleanup_start_namespace_observation=post_cleanup,
        terminal_attestor_lifecycle=terminal_attestor_lifecycle,
    )
    observation = V2.validate_rooted_phase1_supervisor_failure_observation(
        observation
    )
    custody = V2.build_rooted_phase1_supervisor_failure_custody(
        failure_observation=observation
    )
    custody = V2.validate_rooted_phase1_supervisor_failure_custody(custody)
    V2.write_rooted_phase1_supervisor_failure_custody_exclusive_fsync(
        failure_custody=custody,
        supervisor_process_identity=supervisor_process_identity,
    )
    return V2.validate_rooted_phase1_supervisor_failure_custody(custody)


def supervise_phase_1_execution(attempt_id: str) -> dict[str, Any]:
    supervisor_started = time.monotonic_ns()
    supervisor = _process_identity()
    if (
        supervisor["argv"] != V2.expected_phase1_supervisor_argv(attempt_id)
        or supervisor["cwd"] != str(V2.REPO_ROOT)
    ):
        raise V2EvaluationError("Phase-1 supervisor identity drift")
    prelaunch = V2.observe_phase1_supervisor_prelaunch_namespace(attempt_id)
    amendment = V2.validate_technical_correction_amendment_authority(
        prelaunch["technical_correction_amendment_authority"]
    )
    failed_nonreuse = (
        V2.validate_failed_technical_startup_nonreuse_custody(
            prelaunch["failed_technical_startup_nonreuse_custody"],
            reverify_live=True,
        )
    )
    if (
        amendment != failed_nonreuse["amendment_authority"]
        or amendment
        != prelaunch["technical_correction_amendment_authority"]
        or failed_nonreuse["content_digest"]
        != prelaunch[
            "failed_technical_startup_nonreuse_content_digest"
        ]
    ):
        raise V2EvaluationError(
            "Phase-1 corrected-startup nonreuse custody drift"
        )
    handoff_fd, handoff_custody = (
        V2.create_phase1_supervisor_prelaunch_handoff_memfd(
            attempt_id=attempt_id,
            prelaunch_namespace_custody=prelaunch,
            supervisor_process_identity=supervisor,
        )
    )
    producer_environment = V2.build_supervised_phase1_environment(
        base_environment=_internal_environment(),
        attempt_id=attempt_id,
        prelaunch_handoff_memfd_custody=handoff_custody,
    )
    producer_environment_names = sorted(producer_environment)
    producer_environment_sha256 = V2.canonical_json_sha256(
        producer_environment
    )
    supervisor_environment_names, supervisor_environment_sha256 = (
        _environment_custody()
    )
    process: subprocess.Popen[bytes] | None = None
    producer_identity: dict[str, Any] | None = None
    identity_error: BaseException | None = None
    lifecycle_error: BaseException | None = None
    stdout = b""
    stderr = b""
    popen_started = time.monotonic_ns()
    popen_returned = popen_started
    identity_observed = popen_started
    communication_attempted = False
    communicate_completed = False
    process_reaped = False
    cleanup_actions: list[str] = []
    try:
        process = subprocess.Popen(
            list(V2.PUBLIC_EXECUTE_ARGV),
            cwd=V2.REPO_ROOT,
            env=producer_environment,
            pass_fds=V2.supervised_phase1_prelaunch_pass_fds(
                handoff_custody
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        popen_returned = time.monotonic_ns()
        try:
            producer_identity = _process_identity_for_pid(
                process.pid, expected_argv=V2.PUBLIC_EXECUTE_ARGV
            )
        except BaseException as exc:
            identity_error = exc
        identity_observed = time.monotonic_ns()
        communication_attempted = True
        stdout, stderr = process.communicate()
        communicate_completed = True
        process_reaped = process.returncode is not None
    except BaseException as exc:
        lifecycle_error = exc
        if process is not None:
            cleanup_actions.extend(_terminate_process_group(process))
            drained_stdout, drained_stderr, drained = _bounded_process_drain(
                process,
                timeout_seconds=float(
                    V2.PROCESS_LIFECYCLE_POLICY["kill_grace_seconds"]
                ),
            )
            stdout += drained_stdout
            stderr += drained_stderr
            communication_attempted = True
            communicate_completed = drained
            process_reaped = process.returncode is not None
    finally:
        try:
            os.close(handoff_fd)
        except OSError:
            pass
    if process is not None:
        cleanup_actions.extend(_cleanup_remaining_process_group(process.pid))
    ended = time.monotonic_ns()
    post_cleanup = V2.observe_phase1_post_cleanup_start_namespace(
        attempt_id=attempt_id,
        supervisor_process_identity=supervisor,
    )
    attempt_root = Path(prelaunch["attempt_root"])
    if producer_identity is None and not post_cleanup[
        "attempt_root_absent_after_cleanup"
    ]:
        try:
            historical_probe = _load_attempt_root_custody_for_reopen(
                attempt_root
            )
            producer_identity = copy.deepcopy(
                historical_probe["producer_process_identity"]
            )
        except BaseException as exc:
            if identity_error is None:
                identity_error = exc
    returncode = None if process is None else process.returncode
    termination_signal = (
        -returncode
        if type(returncode) is int and returncode < 0
        else None
    )
    group_members = (
        [] if process is None else _process_group_members(process.pid)
    )
    active_roles = _active_v2_role_rows()
    if producer_identity is not None and not _identity_is_live(
        producer_identity
    ):
        gpu, kfd, locks, open_files = _producer_resource_rows(
            producer_identity, attempt_root=attempt_root
        )
    else:
        gpu, kfd, locks, open_files = [], [], [], []

    historical: dict[str, Any] | None = None
    reopened: dict[str, Any] | None = None
    rooted = not post_cleanup["attempt_root_absent_after_cleanup"]
    if rooted and process is not None:
        V2.require_rooted_phase1_supervisor_failure_custody_absent(
            attempt_id
        )

        def fail_rooted(
            failure_stage: str,
            *failure_errors: BaseException,
            terminal_attestor_lifecycle: Mapping[str, Any] | None = None,
        ) -> None:
            custody = _persist_rooted_phase1_supervisor_failure(
                attempt_id=attempt_id,
                prelaunch_namespace_custody=prelaunch,
                prelaunch_handoff_memfd_custody=handoff_custody,
                supervisor_process_identity=supervisor,
                producer_process_identity=producer_identity,
                attempt_root_fd_custody=historical,
                supervisor_root_reopen_custody=reopened,
                producer_environment_names=producer_environment_names,
                producer_environment_sha256=producer_environment_sha256,
                supervisor_environment_names=supervisor_environment_names,
                supervisor_environment_sha256=supervisor_environment_sha256,
                process=process,
                captured_stdout=stdout,
                captured_stderr=stderr,
                failure_stage=failure_stage,
                failure_errors=failure_errors,
                supervisor_started_monotonic_ns=supervisor_started,
                popen_started_monotonic_ns=popen_started,
                cleanup_actions=cleanup_actions,
                communication_attempted=communication_attempted,
                communicate_completed=communicate_completed,
                terminal_attestor_lifecycle=terminal_attestor_lifecycle,
            )
            raise V2EvaluationError(
                "rooted Phase-1 supervisor failure is UNKNOWN_NO_RETRY; "
                f"custody={custody['content_digest']}"
            ) from failure_errors[0]

        try:
            historical = _load_attempt_root_custody_for_reopen(attempt_root)
        except BaseException as exc:
            fail_rooted("ATTEMPT_ROOT_CUSTODY_LOAD", exc)
        if producer_identity is None:
            producer_identity = copy.deepcopy(
                historical["producer_process_identity"]
            )
        try:
            parent_fd, root_fd, reopened = (
                V2.reopen_phase1_attempt_root_for_supervisor(
                    attempt_root_fd_custody=historical,
                    supervisor_process_identity=supervisor,
                    producer_process_identity=producer_identity,
                    producer_absent_after_exit=True,
                )
            )
        except BaseException as exc:
            fail_rooted("ATTEMPT_ROOT_REOPEN", exc)

        source_payloads: dict[str, bytes] = {}
        source_error: BaseException | None = None
        close_errors: list[BaseException] = []
        try:
            for target, source in (
                ("traceback", "phase_1_producer_traceback"),
                ("exception", "phase_1_producer_exception"),
                ("last_stage", "phase_1_producer_last_stage"),
            ):
                source_payloads[target] = V2.read_v2_attempt_bytes_no_follow(
                    attempt_root_fd_custody=historical,
                    reopened_attempt_root_fd_custody=reopened,
                    root_fd=root_fd,
                    relative_path=V2.RUNTIME_PATHS[source],
                )
        except BaseException as exc:
            source_error = exc
        if source_error is not None:
            for descriptor in (root_fd, parent_fd):
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    close_errors.append(close_error)
            fail_rooted(
                "PRODUCER_SOURCE_STREAM_LOAD", source_error, *close_errors
            )

        payloads = {"stdout": stdout, "stderr": stderr, **source_payloads}
        paths = V2.phase1_terminal_observation_runtime_paths()
        try:
            observation = V2.build_phase1_supervisor_observation(
                attempt_id=attempt_id,
                prelaunch_namespace_custody=prelaunch,
                prelaunch_handoff_memfd_custody=handoff_custody,
                supervisor_process_identity=supervisor,
                producer_process_identity=producer_identity,
                attempt_root_fd_custody=historical,
                supervisor_root_reopen_custody=reopened,
                producer_environment_names=producer_environment_names,
                producer_environment_sha256=producer_environment_sha256,
                supervisor_environment_names=supervisor_environment_names,
                supervisor_environment_sha256=supervisor_environment_sha256,
                spawned_pid=process.pid,
                spawned_process_group_id=process.pid,
                stream_bindings=_stream_bindings(paths, payloads),
                returncode=returncode,
                termination_signal=termination_signal,
                supervisor_started_monotonic_ns=supervisor_started,
                popen_started_monotonic_ns=popen_started,
                popen_returned_monotonic_ns=popen_returned,
                identity_observed_monotonic_ns=identity_observed,
                ended_monotonic_ns=ended,
                cleanup_actions=cleanup_actions,
                process_group_members_after_cleanup=group_members,
                active_v2_role_rows_after_cleanup=active_roles,
                gpu_process_rows_after_cleanup=gpu,
                kfd_rows_after_cleanup=kfd,
                lock_rows_after_cleanup=locks,
                open_file_rows_after_cleanup=open_files,
                process_absent_after_exit=not _identity_is_live(
                    producer_identity
                ),
                communication_attempted=communication_attempted,
                communicate_completed=communicate_completed,
                process_reaped=process_reaped,
                cancellation_requested=lifecycle_error is not None,
            )
        except BaseException as exc:
            for descriptor in (root_fd, parent_fd):
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    close_errors.append(close_error)
            fail_rooted(
                "SUPERVISOR_OBSERVATION_CONSTRUCTION",
                exc,
                *close_errors,
            )

        bundle_error: BaseException | None = None
        try:
            V2.write_phase1_supervisor_observation_bundle_exclusive_fsync(
                supervisor_observation=observation,
                stream_payloads=payloads,
                parent_fd=parent_fd,
                root_fd=root_fd,
            )
        except BaseException as exc:
            bundle_error = exc
        for descriptor in (root_fd, parent_fd):
            try:
                os.close(descriptor)
            except BaseException as close_error:
                close_errors.append(close_error)
        if bundle_error is not None or close_errors:
            fail_rooted(
                "SUPERVISOR_BUNDLE_PUBLICATION",
                *(
                    ([bundle_error] if bundle_error is not None else [])
                    + close_errors
                ),
            )

        try:
            terminal = _run_terminal_attestor(
                V2.expected_internal_argv(
                    V2.PHASE1_TERMINAL_CUSTODY_MODE,
                    attempt_root,
                    supervisor_observation_content_digest=observation[
                        "content_digest"
                    ],
                )
            )
        except V2TerminalAttestorProcessError as exc:
            fail_rooted(
                exc.failure_stage,
                *exc.failure_errors,
                terminal_attestor_lifecycle=exc.lifecycle,
            )
        if terminal.get("pass") is not True:
            raise V2EvaluationError(
                "Phase-1 terminal custody records producer failure"
            )
        return terminal

    error = lifecycle_error or identity_error or V2EvaluationError(
        f"Phase-1 producer failed before usable root: returncode={returncode}"
    )
    traceback_payload = stderr or (
        "".join(traceback.format_exception(error)).encode("utf-8")
    )
    structured = _structured_failure(
        label=type(error).__name__,
        message=str(error),
        traceback_bytes=traceback_payload,
    )
    last_stage = "PHASE1_PRE_ROOT_FAILURE"
    pre_root_payloads = {
        "stdout": stdout,
        "stderr": stderr,
        "traceback": traceback_payload,
        "exception": V2.canonical_json_bytes(structured) + b"\n",
        "last_stage": _pre_root_last_stage_payload(
            supervisor_identity=supervisor, last_stage=last_stage
        ),
    }
    pre_root_paths = V2.pre_root_phase1_supervisor_observation_runtime_paths()
    observation = V2.build_pre_root_phase1_supervisor_observation(
        attempt_id=attempt_id,
        prelaunch_namespace_custody=prelaunch,
        prelaunch_handoff_memfd_custody=handoff_custody,
        supervisor_process_identity=supervisor,
        producer_process_identity=producer_identity,
        spawned_pid=None if process is None else process.pid,
        spawned_process_group_id=None if process is None else process.pid,
        popen_succeeded=process is not None,
        identity_observation_error=(
            structured if process is not None and producer_identity is None else None
        ),
        producer_environment_names=producer_environment_names,
        producer_environment_sha256=producer_environment_sha256,
        supervisor_environment_names=supervisor_environment_names,
        supervisor_environment_sha256=supervisor_environment_sha256,
        stream_bindings=_stream_bindings(pre_root_paths, pre_root_payloads),
        structured_exception=structured,
        last_stage=last_stage,
        returncode=returncode,
        termination_signal=termination_signal,
        supervisor_started_monotonic_ns=supervisor_started,
        popen_started_monotonic_ns=popen_started,
        ended_monotonic_ns=ended,
        cleanup_actions=cleanup_actions,
        process_group_members_after_cleanup=group_members,
        active_v2_role_rows_after_cleanup=active_roles,
        gpu_process_rows_after_cleanup=gpu,
        kfd_rows_after_cleanup=kfd,
        lock_rows_after_cleanup=locks,
        open_file_rows_after_cleanup=open_files,
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
        post_cleanup_start_namespace_observation=post_cleanup,
    )
    V2.write_pre_root_phase1_supervisor_observation_bundle_exclusive_fsync(
        supervisor_observation=observation,
        stream_payloads=pre_root_payloads,
    )
    terminal = _run_terminal_attestor(
        V2.expected_pre_root_phase1_failure_custody_argv(
            observation["content_digest"]
        )
    )
    raise V2EvaluationError(
        "Phase-1 failed before a usable attempt-root boundary; "
        f"custody={terminal.get('content_digest')}"
    )


def _load_successful_phase_custody(
    attempt_root: Path, phase_index: int
) -> dict[str, Any]:
    key = f"phase_{phase_index + 1}_custody"
    value = _load_canonical_json(V2.runtime_path(key, attempt_root))
    custody = V2.validate_phase_custody(value)
    if (
        custody["phase_id"] != V2.PHASE_IDS[phase_index]
        or custody["pass"] is not True
    ):
        raise V2EvaluationError("prior phase custody did not pass")
    if _identity_is_live(custody["process_identity"]):
        raise V2EvaluationError("prior V2 phase process is still live")
    return custody


def _rebind_scientific_artifacts(
    *, attempt_root: Path, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], int, int, int]:
    rebound: dict[str, Any] = {}
    files_opened = 0
    bytes_opened = 0
    ledger_rows = 0
    for key, authority in V2.SCIENTIFIC_ARTIFACT_AUTHORITY.items():
        expected = payload["artifact_bindings"][key]
        if expected is None:
            rebound[key] = None
            continue
        path = attempt_root / authority["path"]
        raw = _verify_artifact_binding(
            path, expected, attempt_root=attempt_root
        )
        rebuilt = _artifact_binding(
            path, attempt_root=attempt_root, rows=authority["rows"]
        )
        if rebuilt != expected:
            raise V2EvaluationError(f"artifact rebound differs: {key}")
        rebound[key] = rebuilt
        files_opened += 1
        bytes_opened += len(raw)
        if path.suffix == ".jsonl":
            rows = _rows_from_jsonl(path)
            if authority["rows"] is not None and len(rows) != authority["rows"]:
                raise V2EvaluationError(f"artifact row count differs: {key}")
            ledger_rows += len(rows)
    return rebound, files_opened, bytes_opened, ledger_rows


def _validate_phase1_helper_artifacts(
    *, attempt_root: Path, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = (
        "proprio_helper"
        if mode == V2.PHASE1_PROPRIO_HELPER_MODE
        else "stage_c_helper"
    )
    manifest = _load_canonical_json(
        V2.runtime_path(f"{prefix}_manifest", attempt_root)
    )
    V2.validate_phase1_helper_manifest(manifest)
    receipt = _load_canonical_json(
        V2.runtime_path(f"{prefix}_receipt", attempt_root)
    )
    V2.validate_phase1_helper_receipt(receipt)
    custody = _load_canonical_json(
        V2.runtime_path(f"{prefix}_exit_custody", attempt_root)
    )
    V2.validate_phase1_helper_exit_custody(custody)
    if (
        custody["pass"] is not True
        or custody["helper_receipt"] != receipt
        or receipt["output_manifest_binding"]
        != _artifact_binding(
            V2.runtime_path(f"{prefix}_manifest", attempt_root),
            attempt_root=attempt_root,
            rows=(96 if mode == V2.PHASE1_PROPRIO_HELPER_MODE else 144),
        )
    ):
        raise V2EvaluationError("Phase-1 helper custody cross-binding drift")
    return manifest, receipt


def _regenerate_phase1_scientific_counters(
    *,
    v1: Any,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    training_receipt: Mapping[str, Any],
    stage_a_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    stage_b_rows: Sequence[Mapping[str, Any]],
    stage_c_rows: Sequence[Mapping[str, Any]],
    fidelity_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    candidate_selection_rows: Sequence[Mapping[str, Any]],
    proprio_helper_manifest: Mapping[str, Any] | None,
    proprio_helper_receipt: Mapping[str, Any] | None,
    stage_c_helper_manifest: Mapping[str, Any] | None,
    stage_c_helper_receipt: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Independently derive the complete Phase-1 counter record."""

    counters = V2.zero_scientific_counters()
    counters.update(
        {
            "fit_states_opened": len(datasets[v1.FIT]),
            "fit_rows_opened": sum(
                len(rows) for rows in datasets[v1.FIT].values()
            ),
            "calibration_states_opened": len(datasets[v1.CALIBRATION]),
            "calibration_rows_opened": sum(
                len(rows) for rows in datasets[v1.CALIBRATION].values()
            ),
            "heldout_states_opened": len(datasets[v1.HELDOUT]),
            "heldout_rows_opened": sum(
                len(rows) for rows in datasets[v1.HELDOUT].values()
            ),
            "route_labels_opened": len(stage_a_rows),
            "true_future_latents_opened": len(stage_a_rows),
            "route_cost_checkpoints_opened": len(
                training_receipt["checkpoint_bindings"]
            ),
            "model_initializations": len(
                training_receipt["checkpoint_bindings"]
            ),
            "optimizer_updates": _optimizer_update_count(training_receipt),
            "training_epochs": sum(
                len(history)
                for history in training_receipt["training_history"].values()
            ),
            "scientific_score_rows": sum(
                len(rows)
                for rows in (
                    stage_a_rows,
                    raw_rows,
                    stage_b_rows,
                    stage_c_rows,
                    fidelity_rows,
                    action_rows,
                )
            ),
            "candidate_selections": len(candidate_selection_rows),
            "scientific_metrics": 1,
            "scientific_payloads": 1,
        }
    )
    if proprio_helper_manifest is not None:
        if proprio_helper_receipt is None:
            raise V2EvaluationError("Stage-B counter receipt is absent")
        counters["predicted_latents_opened"] += int(
            proprio_helper_manifest["logical_records"]
        )
        counters["predictor_checkpoints_opened"] += int(
            proprio_helper_receipt["counters"][
                "predictor_checkpoints_opened"
            ]
        )
    elif proprio_helper_receipt is not None:
        raise V2EvaluationError("Stage-B counter manifest is absent")
    if stage_c_helper_manifest is not None:
        if stage_c_helper_receipt is None:
            raise V2EvaluationError("Stage-C counter receipt is absent")
        counters["predicted_latents_opened"] += int(
            stage_c_helper_manifest["logical_records"]
        )
        counters["predictor_checkpoints_opened"] += int(
            stage_c_helper_receipt["counters"][
                "predictor_checkpoints_opened"
            ]
        )
    elif stage_c_helper_receipt is not None:
        raise V2EvaluationError("Stage-C counter manifest is absent")
    if tuple(counters) != V2.SCIENTIFIC_COUNTER_FIELDS:
        raise V2EvaluationError("regenerated scientific counter order drift")
    return counters


def _regenerate_scientific_payload(
    *, attempt_root: Path, original: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """Rebuild every decision-bearing component without training or inference."""

    payload = V2.validate_immutable_scientific_payload(original)
    v1 = _v1()
    v1_root = _attempt_fd_alias(attempt_root)
    artifacts, opened, byte_count, ledger_count = _rebind_scientific_artifacts(
        attempt_root=attempt_root, payload=payload
    )
    metrics = _load_canonical_json(
        attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["metrics"]["path"]
    )
    selections = _load_canonical_json(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["candidate_selections"]["path"]
    )
    gates = _load_canonical_json(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["gate_decisions"]["path"]
    )
    for value in (metrics, selections, gates):
        V2.validate_self_digest(value)

    stage_a_rows = _rows_from_jsonl(
        attempt_root / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_a_rows"]["path"]
    )
    raw_rows = _rows_from_jsonl(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_a_raw_cost_rows"]["path"]
    )
    if len(raw_rows) != V2.STAGE_ROW_AUTHORITY["stage_a_raw_cost_rows"]:
        raise V2EvaluationError("raw Stage-A comparator row count drift")
    v1._validate_score_row_arithmetic(stage_a_rows, stage="STAGE_A")
    v1._validate_persisted_action_plans(stage_a_rows)
    datasets = v1._datasets_from_stage_rows(stage_a_rows)
    target_rows = _rows_from_jsonl(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["route_only_target_ledger"][
            "path"
        ]
    )
    regenerated_target_rows = v1._training_target_rows(datasets)
    if regenerated_target_rows != target_rows:
        raise V2EvaluationError("route-only target ledger does not regenerate")
    training_rows = _rows_from_jsonl(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["training_ledger"]["path"]
    )
    training_receipt = _load_canonical_json(
        attempt_root
        / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["training_receipt"]["path"]
    )
    regenerated_training_rows = [
        {
            "schema": "plan_aware_training_epoch_row_v1",
            "condition": condition,
            **copy.deepcopy(dict(row)),
        }
        for condition, history in training_receipt["training_history"].items()
        for row in history
    ]
    if regenerated_training_rows != training_rows:
        raise V2EvaluationError("training epoch ledger does not regenerate")
    _raw_maps, regenerated_raw_rows, regenerated_raw_reduction = (
        v1._predecessor_raw_goal_score_maps(
            datasets, heldout_barrier_open=True
        )
    )
    if regenerated_raw_rows != raw_rows:
        raise V2EvaluationError(
            "predecessor -cost_h3 rows do not exactly regenerate"
        )
    v1._validate_raw_cost_merged_scores(stage_a_rows, regenerated_raw_rows)
    if (
        metrics["stage_a_raw_cost_rereduced"]
        != regenerated_raw_reduction
    ):
        raise V2EvaluationError("raw-cost reduction custody does not regenerate")
    tensor_rows = v1.tensor_index()
    v1._validate_persisted_latent_bindings(
        stage_a_rows, tensor_rows=tensor_rows
    )
    stage_a_summaries = v1._replay_stage_a_metrics(stage_a_rows)
    if stage_a_summaries != metrics["stage_a"]:
        raise V2EvaluationError("Stage-A metrics do not regenerate")
    stage_a_decisions = v1._stage_a_decisions(
        stage_a_summaries[v1.HELDOUT]
    )
    if stage_a_decisions != metrics["stage_a_decisions"]:
        raise V2EvaluationError("Stage-A decisions do not regenerate")

    stage_b: dict[str, Any] | None = None
    stage_c: dict[str, Any] | None = None
    stage_b_rows: list[dict[str, Any]] = []
    stage_c_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    proprio_helper_manifest: dict[str, Any] | None = None
    proprio_helper_receipt: dict[str, Any] | None = None
    stage_c_helper_manifest: dict[str, Any] | None = None
    stage_c_helper_receipt: dict[str, Any] | None = None
    if payload["stage_execution"]["stage_b"]["executed"]:
        stage_b_gate_path = V2.runtime_path(
            "stage_b_gate_receipt", attempt_root
        )
        stage_b_gate = _load_canonical_json(stage_b_gate_path)
        V2.validate_stage_b_gate_receipt(stage_b_gate)
        if stage_b_gate["stage_a_decisions"] != stage_a_decisions:
            raise V2EvaluationError("Stage-B gate does not bind regenerated Stage A")
        proprio_helper_manifest, proprio_helper_receipt = (
            _validate_phase1_helper_artifacts(
                attempt_root=attempt_root,
                mode=V2.PHASE1_PROPRIO_HELPER_MODE,
            )
        )
        prediction_rows, stage_b_materialisation = (
            _load_stage_b_helper_outputs(
                attempt_root=attempt_root,
                gate_path=stage_b_gate_path,
                gate=stage_b_gate,
                helper_receipt=proprio_helper_receipt,
            )
        )
        stage_b_rows = _rows_from_jsonl(
            attempt_root
            / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_b_rows"]["path"]
        )
        v1._validate_score_row_arithmetic(stage_b_rows, stage="CONDITIONAL")
        v1._validate_persisted_action_plans(stage_b_rows)
        v1._validate_persisted_latent_bindings(
            stage_b_rows, tensor_rows={**tensor_rows, **prediction_rows}
        )
        stage_b_summaries = v1._replay_conditional_metrics(
            stage_b_rows, expected_sources=("R1", "RR", "P1", "PR")
        )
        persisted_stage_b = metrics["stage_b"]
        if stage_b_summaries != persisted_stage_b["summaries"]:
            raise V2EvaluationError("Stage-B metrics do not regenerate")
        stage_b_decisions = v1._stage_b_decisions(
            stage_a_decisions=stage_a_decisions,
            stage_a_heldout=stage_a_summaries[v1.HELDOUT],
            summaries=stage_b_summaries[v1.HELDOUT],
        )
        if any(
            stage_b_decisions[key] != persisted_stage_b[key]
            for key in _STAGE_B_DECISION_KEYS
        ):
            raise V2EvaluationError("Stage-B decisions do not regenerate")
        stage_b = {
            "schema": "plan_aware_stage_b_predictor_substitution_v2",
            "row_count": len(stage_b_rows),
            "source_specific_rows": True,
            "sources": ["R1", "RR", "P1", "PR"],
            "summaries": stage_b_summaries,
            **stage_b_decisions,
            "gate_receipt": _artifact_binding(
                stage_b_gate_path, attempt_root=attempt_root, rows=None
            ),
            "gate_content_digest": stage_b_gate["content_digest"],
            "helper_materialisation_custody": stage_b_materialisation,
            "ledger": artifacts["stage_b_rows"],
            "ranker_refit_or_recalibration": False,
            "predictor_training_steps": 0,
            "fresh_states_or_candidates": 0,
        }
        if stage_b != persisted_stage_b:
            raise V2EvaluationError(
                "complete Stage-B object does not independently regenerate"
            )

        if payload["stage_execution"]["stage_c"]["executed"]:
            stage_c_gate_path = V2.runtime_path(
                "stage_c_gate_receipt", attempt_root
            )
            stage_c_gate = _load_canonical_json(stage_c_gate_path)
            V2.validate_stage_c_gate_receipt(stage_c_gate)
            if stage_c_gate["stage_b_decisions"] != stage_b_decisions:
                raise V2EvaluationError(
                    "Stage-C gate does not bind regenerated Stage B"
                )
            stage_c_helper_manifest, stage_c_helper_receipt = (
                _validate_phase1_helper_artifacts(
                    attempt_root=attempt_root,
                    mode=V2.PHASE1_STAGE_C_HELPER_MODE,
                )
            )
            stage_c_prediction_rows, stage_c_materialisation = (
                _load_stage_c_helper_outputs(
                    attempt_root=attempt_root,
                    stage_b_gate_digest=str(
                        stage_b_gate["content_digest"]
                    ),
                    stage_c_gate_digest=str(
                        stage_c_gate["content_digest"]
                    ),
                    helper_receipt=stage_c_helper_receipt,
                )
            )
            stage_c_rows = _rows_from_jsonl(
                attempt_root
                / V2.SCIENTIFIC_ARTIFACT_AUTHORITY["stage_c_rows"]["path"]
            )
            fidelity_rows = _rows_from_jsonl(
                attempt_root
                / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
                    "stage_c_direct_fidelity_rows"
                ]["path"]
            )
            action_rows = _rows_from_jsonl(
                attempt_root
                / V2.SCIENTIFIC_ARTIFACT_AUTHORITY[
                    "stage_c_action_sensitivity_rows"
                ]["path"]
            )
            v1._validate_score_row_arithmetic(
                stage_c_rows, stage="CONDITIONAL"
            )
            v1._validate_persisted_action_plans(stage_c_rows)
            stage_c_summaries = v1._replay_conditional_metrics(
                stage_c_rows, expected_sources=v1.STAGE_C_SOURCE_IDS
            )
            persisted_stage_c = metrics["stage_c"]
            if stage_c_summaries != persisted_stage_c["summaries"]:
                raise V2EvaluationError("Stage-C metrics do not regenerate")
            prediction_indexes = {
                "PR": v1._load_prediction_index(
                    attempt=v1_root,
                    source_id="PR_PROPRIO_ROLLOUT",
                    gate_digest=stage_b_gate["content_digest"],
                )[1]
            }
            for source_id in v1.STAGE_C_SOURCE_IDS:
                _source_rows, index = v1._load_prediction_index(
                    attempt=v1_root,
                    source_id=source_id,
                    gate_digest=stage_b_gate["content_digest"],
                    stage_c_gate_digest=stage_c_gate["content_digest"],
                )
                prediction_indexes[source_id] = index
            v1._validate_persisted_latent_bindings(
                stage_c_rows,
                tensor_rows={
                    **tensor_rows,
                    **prediction_rows,
                    **stage_c_prediction_rows,
                },
            )
            fidelity = v1._aggregate_stage_c_fidelity(fidelity_rows)
            action_sensitivity = v1._action_sensitivity_evidence(
                attempt=v1_root,
                rows=action_rows,
                prediction_indexes=prediction_indexes,
            )
            stage_c_decisions = v1._stage_c_decisions(
                stage_b=stage_b,
                summaries=stage_c_summaries,
                action_sensitivity=action_sensitivity,
            )
            if any(
                stage_c_decisions[key] != persisted_stage_c[key]
                for key in _STAGE_C_DECISION_KEYS
            ):
                raise V2EvaluationError("Stage-C decisions do not regenerate")
            if (
                fidelity != persisted_stage_c["direct_future_fidelity_h1_h3"]
                or action_sensitivity
                != persisted_stage_c["candidate_action_sensitivity"]
                or v1._aggregate_stage_c_route_score_changes(stage_c_rows)
                != persisted_stage_c["route_score_changes"]
            ):
                raise V2EvaluationError("Stage-C evidence does not regenerate")
            route_score_changes = v1._aggregate_stage_c_route_score_changes(
                stage_c_rows
            )
            stage_c = {
                "schema": "plan_aware_stage_c_attribution_v2",
                "row_count": len(stage_c_rows),
                "sources": list(v1.STAGE_C_SOURCE_IDS),
                "summaries": stage_c_summaries,
                **stage_c_decisions,
                "route_score_changes": route_score_changes,
                "direct_future_fidelity_h1_h3": fidelity,
                "direct_fidelity_rows": len(fidelity_rows),
                "direct_fidelity_ledger": artifacts[
                    "stage_c_direct_fidelity_rows"
                ],
                "candidate_action_sensitivity": action_sensitivity,
                "action_sensitivity_ledger": artifacts[
                    "stage_c_action_sensitivity_rows"
                ],
                "gate_receipt": _artifact_binding(
                    stage_c_gate_path,
                    attempt_root=attempt_root,
                    rows=None,
                ),
                "gate_content_digest": stage_c_gate["content_digest"],
                "helper_materialisation_custody": stage_c_materialisation,
                "ledger": artifacts["stage_c_rows"],
                "ranker_refit_or_recalibration": False,
                "predictor_training_steps": 0,
            }
            if stage_c != persisted_stage_c:
                raise V2EvaluationError(
                    "complete Stage-C object does not independently regenerate"
                )

    decision = _scientific_decision(stage_a_decisions, stage_b, stage_c)
    if decision != payload["scientific_decision"]:
        raise V2EvaluationError("scientific decision does not regenerate")
    regenerated_selections = _candidate_selection_rows(
        stage_a_summaries=stage_a_summaries,
        stage_b=stage_b,
        stage_c=stage_c,
    )
    regenerated_selection_component = V2.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v2."
                "candidate_selections.v1"
            ),
            "attempt_id": payload["attempt_id"],
            "rows": regenerated_selections,
            "row_count": len(regenerated_selections),
        }
    )
    if regenerated_selection_component != selections:
        raise V2EvaluationError("candidate selections do not regenerate")
    regenerated_gates = _decision_component(
        stage_a_decisions=stage_a_decisions,
        stage_b=stage_b,
        stage_c=stage_c,
        scientific_decision=decision,
    )
    if regenerated_gates != gates:
        raise V2EvaluationError("gate-decision component does not regenerate")
    evaluation_contract = _load_canonical_json(
        attempt_root / "receipts/evaluation_contract.json"
    )
    proxy = {
        "checkpoint_bindings": {
            "trained_route_rankers": evaluation_contract[
                "checkpoint_bindings"
            ]
        }
    }
    v1._validate_training_checkpoint_custody(v1_root, proxy)
    checkpoint_digests = {
        "no_latent": evaluation_contract["checkpoint_bindings"][
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL"
        ]["parameter_digest"],
        "latent_true_future": evaluation_contract["checkpoint_bindings"][
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"
        ]["parameter_digest"],
    }
    if checkpoint_digests != payload["checkpoint_state_digests"]:
        raise V2EvaluationError("checkpoint-state digests do not regenerate")

    regenerated_metrics = V2.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v2.scientific_metrics.v1"
            ),
            "attempt_id": payload["attempt_id"],
            "stage_a": stage_a_summaries,
            "stage_a_decisions": stage_a_decisions,
            "stage_b": stage_b,
            "stage_c": stage_c,
            "fit_optimization": training_receipt["fit_optimization"],
            "stage_a_raw_cost_rereduced": regenerated_raw_reduction,
            "stage_a_raw_cost_matched_comparisons": (
                v1._matched_raw_cost_comparisons(stage_a_summaries, stage_b)
            ),
            "scientific_decision": decision,
            "scientific_custody": {
                "training_smoke": artifacts["training_smoke_receipt"],
                "training": artifacts["training_receipt"],
                "evaluation_contract": artifacts["evaluation_contract"],
                "evaluation_lifecycle": artifacts["evaluation_receipt"],
                "trained_route_ranker_checkpoints": copy.deepcopy(
                    evaluation_contract["checkpoint_bindings"]
                ),
            },
            "historical_failed_output_metrics_opened": 0,
            "historical_failed_output_metrics_reused": 0,
            "raw_goal_cosine_inference_executions": 0,
        }
    )
    if regenerated_metrics != metrics:
        raise V2EvaluationError(
            "scientific metrics component does not independently regenerate"
        )

    regenerated_scientific_counters = _regenerate_phase1_scientific_counters(
        v1=v1,
        datasets=datasets,
        training_receipt=training_receipt,
        stage_a_rows=stage_a_rows,
        raw_rows=raw_rows,
        stage_b_rows=stage_b_rows,
        stage_c_rows=stage_c_rows,
        fidelity_rows=fidelity_rows,
        action_rows=action_rows,
        candidate_selection_rows=regenerated_selections,
        proprio_helper_manifest=proprio_helper_manifest,
        proprio_helper_receipt=proprio_helper_receipt,
        stage_c_helper_manifest=stage_c_helper_manifest,
        stage_c_helper_receipt=stage_c_helper_receipt,
    )
    if regenerated_scientific_counters != payload["scientific_counters"]:
        raise V2EvaluationError("Phase-1 scientific counters do not regenerate")

    component_values: dict[str, Any] = {}
    for key in V2.SCIENTIFIC_COMPONENT_KEYS:
        binding = artifacts[key]
        component_values[key] = (
            None
            if binding is None
            else _load_canonical_json(attempt_root / binding["path"])
        )
    component_values["metrics"] = regenerated_metrics
    component_values["candidate_selections"] = (
        regenerated_selection_component
    )
    component_values["gate_decisions"] = regenerated_gates
    regenerated_stage_execution = V2.build_stage_execution(
        stage_a_gate_pass=decision["stage_a_gate_pass"],
        proprio_contribution_supported=decision[
            "proprio_contribution_supported"
        ],
    )
    if regenerated_stage_execution != payload["stage_execution"]:
        raise V2EvaluationError("conditional stage execution does not regenerate")
    regenerated_components = V2.build_scientific_component_payloads(
        attempt_id=payload["attempt_id"],
        stage_execution=regenerated_stage_execution,
        scientific_decision=decision,
        artifact_bindings=artifacts,
        component_values=component_values,
    )
    if regenerated_components != payload["scientific_component_payloads"]:
        raise V2EvaluationError("scientific components do not regenerate")
    regenerated_invariance = V2.build_scientific_invariance_receipt()
    V2.validate_scientific_invariance_receipt(regenerated_invariance)
    if regenerated_invariance != payload["scientific_invariance_receipt"]:
        raise V2EvaluationError("frozen scientific authority does not regenerate")
    regenerated = V2.build_immutable_scientific_payload(
        attempt_id=payload["attempt_id"],
        source_freeze_commit=payload["source_freeze_commit"],
        scientific_invariance_receipt=regenerated_invariance,
        namespace_and_nonreuse_receipt=payload[
            "namespace_and_nonreuse_receipt"
        ],
        artifact_bindings=artifacts,
        checkpoint_state_digests=checkpoint_digests,
        scientific_decision=decision,
        scientific_counters=regenerated_scientific_counters,
        scientific_component_payloads=regenerated_components,
    )
    counters = V2.expected_phase2_regeneration_counters(payload)
    if (
        counters["artifact_bindings_rehashed"] != opened
        or counters["artifact_bound_bytes_rehashed"] != byte_count
        or counters["candidate_selection_rows_regenerated"]
        != len(regenerated_selections)
        or counters["stage_a_rows_regenerated"] != len(stage_a_rows)
        or counters["stage_a_raw_cost_rows_regenerated"] != len(raw_rows)
        or ledger_count
        != sum(
            counters[key]
            for key in (
                "target_ledger_rows_revalidated",
                "training_ledger_rows_revalidated",
                "stage_a_rows_regenerated",
                "stage_a_raw_cost_rows_regenerated",
                "stage_b_rows_regenerated",
                "stage_c_rows_regenerated",
                "stage_c_direct_fidelity_rows_regenerated",
                "stage_c_action_sensitivity_rows_regenerated",
            )
        )
    ):
        raise V2EvaluationError("Phase-2 exact regeneration counters differ")
    replay_values = {
        "route_target_regenerated_binding": artifacts[
            "route_only_target_ledger"
        ],
        "training_ledger_regenerated_binding": artifacts[
            "training_ledger"
        ],
        "stage_a_rows_regenerated_binding": artifacts["stage_a_rows"],
        "raw_cost_regenerated_binding": artifacts[
            "stage_a_raw_cost_rows"
        ],
        "stage_b_rows_regenerated_binding": artifacts["stage_b_rows"],
        "stage_c_rows_regenerated_binding": artifacts["stage_c_rows"],
        "stage_c_fidelity_regenerated_binding": artifacts[
            "stage_c_direct_fidelity_rows"
        ],
        "stage_c_action_regenerated_binding": artifacts[
            "stage_c_action_sensitivity_rows"
        ],
        "regenerated_checkpoint_state_digests": checkpoint_digests,
        "regenerated_training_receipt": training_receipt,
        "regenerated_scientific_metrics": regenerated_metrics,
        "regenerated_candidate_selections": (
            regenerated_selection_component
        ),
        "regenerated_gate_decisions": regenerated_gates,
        "regenerated_scientific_decision": decision,
        "regenerated_scientific_counters": regenerated_scientific_counters,
    }
    return regenerated, counters, replay_values


def execute_phase_1() -> dict[str, Any]:
    phase_started_monotonic_ns = time.monotonic_ns()
    identity = _process_identity()
    if identity["argv"] != list(V2.PUBLIC_EXECUTE_ARGV):
        raise V2EvaluationError("public V2 execute argv drift")
    if identity["cwd"] != str(V2.REPO_ROOT):
        raise V2EvaluationError("public V2 execute cwd drift")
    prelaunch_handoff = V2.consume_supervised_phase1_prelaunch_handoff(
        environment=os.environ,
        producer_process_identity=identity,
    )
    attempt_id = prelaunch_handoff["attempt_id"]
    source_freeze = _require_clean_source_freeze()
    invariance = V2.build_scientific_invariance_receipt()
    V2.validate_scientific_invariance_receipt(invariance)
    attempt_id, attempt_root, nonreuse, root_fd_custody = _new_attempt_root(
        identity, attempt_id
    )
    root_fd, live_root_custody, _reopened = _require_attempt_io(
        attempt_root
    )
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "STARTED")
    prelaunch_handoff_binding = (
        V2.write_phase1_supervisor_prelaunch_handoff_receipt_exclusive_fsync(
            handoff_receipt=prelaunch_handoff,
            attempt_root_fd_custody=live_root_custody,
            root_fd=root_fd,
        )
    )
    counters = _scientific_counter_state()
    resources_validated = _validate_interpreter_and_resource_paths()
    environment_names, environment_sha256 = _environment_custody()
    foreground_runtime_import_custody = (
        V2.observe_phase1_foreground_runtime_import_custody(
            producer_process_identity=identity,
            environment_names=environment_names,
            environment_sha256=environment_sha256,
        )
    )
    _write_json_exclusive(
        V2.runtime_path("foreground_runtime_import_custody", attempt_root),
        foreground_runtime_import_custody,
    )
    stage_a_audit_installed = _begin_static_audit(
        stage_id="PHASE1_STAGE_A", process_identity=identity
    )
    stage_a_pre_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_A",
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    preexecution = V2.build_preexecution_receipt(
        source_freeze_commit=source_freeze,
        attempt_id=attempt_id,
        supervisor_prelaunch_namespace_custody=prelaunch_handoff[
            "prelaunch_namespace_custody"
        ],
        supervisor_prelaunch_handoff_receipt=prelaunch_handoff,
        supervisor_prelaunch_handoff_binding=prelaunch_handoff_binding,
        supervised_attempt_id_environment_value=attempt_id,
        scientific_invariance_receipt=invariance,
        namespace_and_nonreuse_receipt=nonreuse,
        attempt_root_fd_custody=root_fd_custody,
        foreground_runtime_import_custody=(
            foreground_runtime_import_custody
        ),
        producer_process_identity=identity,
        environment_names=environment_names,
        environment_sha256=environment_sha256,
        completed_monotonic_ns=time.monotonic_ns(),
        source_worktree_clean=True,
        static_input_metadata_snapshot=stage_a_pre_snapshot,
        interpreter_and_resource_paths_validated=resources_validated,
    )
    V2.validate_preexecution_receipt(preexecution)
    _write_json_exclusive(
        V2.runtime_path("preexecution_receipt", attempt_root), preexecution
    )
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "PREEXECUTION_VALIDATED")

    boundary_entered_monotonic_ns = _monotonic_ns_after(
        preexecution["completed_monotonic_ns"]
    )
    boundary_entry = V2.build_scientific_attempt_boundary_entry_receipt(
        preexecution_receipt=preexecution,
        boundary_entered_monotonic_ns=boundary_entered_monotonic_ns,
    )
    root_fd, live_root_custody, _reopened = _require_attempt_io(attempt_root)
    boundary_binding, boundary_persisted_monotonic_ns = (
        V2.write_scientific_attempt_boundary_entry_exclusive_fsync(
            boundary_entry_receipt=boundary_entry,
            preexecution_receipt=preexecution,
            attempt_root_fd_custody=live_root_custody,
            root_fd=root_fd,
        )
    )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[0],
        "SCIENTIFIC_ATTEMPT_BOUNDARY_ENTRY_DURABLE",
    )
    v1 = _v1()
    initiation_entered_monotonic_ns = _monotonic_ns_after(
        boundary_persisted_monotonic_ns
    )
    first_open_initiation = (
        V2.build_first_scientific_open_initiation_receipt(
            boundary_entry_receipt=boundary_entry,
            preexecution_receipt=preexecution,
            boundary_entry_artifact_binding=boundary_binding,
            boundary_entry_persisted_monotonic_ns=(
                boundary_persisted_monotonic_ns
            ),
            initiation_entered_monotonic_ns=(
                initiation_entered_monotonic_ns
            ),
            first_scientific_input_binding=(
                V2.STATIC_INPUT_BINDINGS["panel"]["split"]
            ),
        )
    )
    initiation_binding, initiation_persisted_monotonic_ns = (
        V2.write_first_scientific_open_initiation_exclusive_fsync(
            initiation_receipt=first_open_initiation,
            boundary_entry_receipt=boundary_entry,
            preexecution_receipt=preexecution,
            attempt_root_fd_custody=live_root_custody,
            root_fd=root_fd,
        )
    )
    ids = v1.split_ids()
    first_scientific_open_event = (
        V2.observe_first_scientific_open_event_in_flight(
            process_identity=identity,
            first_open_raw_audit_row=_first_split_audit_row(),
            initiation_receipt=first_open_initiation,
            boundary_entry_receipt=boundary_entry,
            preexecution_receipt=preexecution,
            initiation_persisted_monotonic_ns=(
                initiation_persisted_monotonic_ns
            ),
        )
    )
    first_scientific_open_completed_monotonic_ns = _monotonic_ns_after(
        first_scientific_open_event["open_completed_monotonic_ns"]
    )
    first_open = V2.build_first_scientific_open_receipt(
        boundary_entry_receipt=boundary_entry,
        preexecution_receipt=preexecution,
        initiation_receipt=first_open_initiation,
        initiation_artifact_binding=initiation_binding,
        initiation_persisted_monotonic_ns=(
            initiation_persisted_monotonic_ns
        ),
        first_open_completed_monotonic_ns=(
            first_scientific_open_completed_monotonic_ns
        ),
        first_scientific_input_binding=V2.STATIC_INPUT_BINDINGS["panel"][
            "split"
        ],
        first_scientific_open_event=first_scientific_open_event,
    )
    V2.validate_first_scientific_open_receipt(
        first_open,
        boundary_entry_receipt=boundary_entry,
        preexecution_receipt=preexecution,
    )
    _write_json_exclusive(
        V2.runtime_path("first_scientific_open_receipt", attempt_root),
        first_open,
    )
    V2.validate_attempt_accounting(
        V2.build_attempt_accounting("SCIENTIFIC_STARTED")
    )
    _write_last_stage(
        attempt_root, V2.PHASE_IDS[0], "FIRST_SCIENTIFIC_OPEN_COMPLETE"
    )

    stage_a_started_monotonic_ns = _monotonic_ns_after(
        first_scientific_open_completed_monotonic_ns
    )
    stage_a = _run_stage_a(
        attempt_root=attempt_root,
        source_freeze=source_freeze,
        counters=counters,
        v1=v1,
        ids=ids,
    )
    stage_a_ended_monotonic_ns = _monotonic_ns_after(
        stage_a_started_monotonic_ns
    )
    stage_a_post_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_A",
        snapshot_kind="POST_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    stage_a_audit_stream = _finish_static_audit(
        stage_id="PHASE1_STAGE_A",
        process_identity=identity,
        attempt_root=attempt_root,
        installed_monotonic_ns=stage_a_audit_installed,
    )
    stage_a_static_custody = _build_static_stage_custody(
        pre_snapshot=stage_a_pre_snapshot,
        post_snapshot=stage_a_post_snapshot,
        static_input_audit_stream=stage_a_audit_stream,
        started_monotonic_ns=stage_a_started_monotonic_ns,
        ended_monotonic_ns=stage_a_ended_monotonic_ns,
        phase1_preexecution_receipt=preexecution,
        phase1_boundary_entry_receipt=boundary_entry,
        first_scientific_open_receipt=first_open,
    )
    _write_json_exclusive(
        V2.runtime_path("static_input_stage_a_custody", attempt_root),
        stage_a_static_custody,
    )
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "TRAINING_COMPLETE")
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "STAGE_A_COMPLETE")
    stage_b, stage_c = _run_conditional_tree(
        attempt_root=attempt_root,
        source_freeze=source_freeze,
        stage_a=stage_a,
        counters=counters,
        helper_runner=_run_phase1_helper_process,
    )
    _write_last_stage(
        attempt_root, V2.PHASE_IDS[0], "STAGE_B_COMPLETE_OR_SKIPPED"
    )
    _write_last_stage(
        attempt_root, V2.PHASE_IDS[0], "STAGE_C_COMPLETE_OR_SKIPPED"
    )
    decision = _scientific_decision(stage_a["decisions"], stage_b, stage_c)
    _write_evaluation_lifecycle_receipt(attempt_root)
    selections, gates, metrics = _write_scientific_components(
        attempt_root=attempt_root,
        attempt_id=attempt_id,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        scientific_decision=decision,
    )
    counters["candidate_selections"] = 48
    counters["scientific_metrics"] = 1
    counters["scientific_payloads"] = 1
    checkpoint_digests = {
        "no_latent": stage_a["checkpoints"][
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL"
        ]["parameter_digest"],
        "latent_true_future": stage_a["checkpoints"][
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"
        ]["parameter_digest"],
    }
    file_count, byte_count = _runtime_file_counts(attempt_root)
    payload_frozen_monotonic_ns = time.monotonic_ns()
    runtime_storage = V2.build_runtime_storage_custody(
        attempt_root=attempt_root,
        producer_process_identity=identity,
        started_monotonic_ns=phase_started_monotonic_ns,
        payload_frozen_monotonic_ns=payload_frozen_monotonic_ns,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        technical_file_count_before_payload=file_count,
        technical_bytes_before_payload=byte_count,
        environment_names=environment_names,
        environment_sha256=environment_sha256,
    )
    _write_json_exclusive(
        V2.runtime_path("runtime_storage_custody", attempt_root),
        runtime_storage,
    )
    artifacts = _scientific_artifact_bindings(
        attempt_root=attempt_root,
        stage_b=stage_b,
        stage_c=stage_c,
    )
    evaluation_receipt = _load_canonical_json(
        attempt_root / "receipts/evaluation.json"
    )
    components = V2.build_scientific_component_payloads(
        attempt_id=attempt_id,
        stage_execution=V2.build_stage_execution(
            stage_a_gate_pass=decision["stage_a_gate_pass"],
            proprio_contribution_supported=decision[
                "proprio_contribution_supported"
            ],
        ),
        scientific_decision=decision,
        artifact_bindings=artifacts,
        component_values=_scientific_component_values(
            attempt_root=attempt_root,
            stage_a=stage_a,
            stage_b=stage_b,
            stage_c=stage_c,
            evaluation_receipt=evaluation_receipt,
            metrics=metrics,
            selections=selections,
            gates=gates,
            runtime_storage=runtime_storage,
        ),
    )
    payload = V2.build_immutable_scientific_payload(
        attempt_id=attempt_id,
        source_freeze_commit=source_freeze,
        scientific_invariance_receipt=invariance,
        namespace_and_nonreuse_receipt=nonreuse,
        artifact_bindings=artifacts,
        checkpoint_state_digests=checkpoint_digests,
        scientific_decision=decision,
        scientific_counters=counters,
        scientific_component_payloads=components,
    )
    payload_bytes = V2.immutable_scientific_payload_bytes(payload)
    _write_bytes_exclusive(
        V2.runtime_path("immutable_scientific_payload", attempt_root),
        payload_bytes,
    )
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "IMMUTABLE_PAYLOAD_FROZEN")
    _write_last_stage(attempt_root, V2.PHASE_IDS[0], "COMPLETE")
    _persist_phase1_terminal_source_streams(error=None, traceback_bytes=b"")
    return payload


def _verified_helper_runtime_import_custody(mode: str) -> dict[str, Any]:
    return V2.observe_phase1_helper_runtime_import_custody(mode=mode)


def capture_proprio_context_state(
    attempt_root: Path, state_index: int
) -> dict[str, Any]:
    identity = _process_identity(
        expected_executable=V2.CPU_WORKER_PYTHON_EXECUTABLE
    )
    expected_parent = V2.expected_internal_argv(
        V2.PHASE1_PROPRIO_HELPER_MODE, attempt_root
    )
    child_root_binding = _activate_inherited_attempt_io(
        attempt_root=attempt_root,
        child_process_identity=identity,
        expected_parent_argv=expected_parent,
    )
    installed = _begin_static_audit(
        stage_id="PHASE1_STAGE_B_HELPER", process_identity=identity
    )
    pre_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_B_HELPER",
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    operation_started = _monotonic_ns_after(
        pre_snapshot["ended_monotonic_ns"]
    )
    gate = V2.validate_stage_b_gate_receipt(
        _load_canonical_json(
            V2.runtime_path("stage_b_gate_receipt", attempt_root)
        )
    )
    materializer = _materializer_v1()
    record = materializer._capture_prefix_state(
        state_index,
        output_root=_attempt_fd_alias(attempt_root),
        gate_digest=str(gate["content_digest"]),
    )
    state_id = str(record["state_id"])
    output_path = (
        attempt_root
        / "stage_b/proprio_context/receipts"
        / f"{state_id}.json"
    )
    operation_ended = time.monotonic_ns()
    post_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_B_HELPER",
        snapshot_kind="POST_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    audit_stream = _finish_static_audit(
        stage_id="PHASE1_STAGE_B_HELPER",
        process_identity=identity,
        attempt_root=attempt_root,
        installed_monotonic_ns=installed,
    )
    parent = _process_identity_for_pid(
        identity["ppid"], expected_argv=expected_parent
    )
    process_custody = V2.build_static_input_process_custody(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        static_input_audit_stream=audit_stream,
        parent_process_identity=parent,
        process_started_monotonic_ns=operation_started,
        process_ended_monotonic_ns=operation_ended,
    )
    return V2.build_phase1_worker_receipt(
        mode=V2.PHASE1_CONTEXT_STATE_WORKER_MODE,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        parent_helper_process_identity=parent,
        worker_process_identity=identity,
        scientific_output_binding=_artifact_binding(
            output_path, attempt_root=attempt_root, rows=None
        ),
        static_input_audit_stream=audit_stream,
        static_input_process_custody=process_custody,
        state_index=state_index,
        state_id=state_id,
    )


def predict_frozen_source(
    attempt_root: Path, source_id: str, ablation: str | None
) -> dict[str, Any]:
    identity = _process_identity()
    parent_mode = (
        V2.PHASE1_STAGE_C_HELPER_MODE
        if ablation is not None
        else V2.PHASE1_PROPRIO_HELPER_MODE
    )
    expected_parent = V2.expected_internal_argv(parent_mode, attempt_root)
    child_root_binding = _activate_inherited_attempt_io(
        attempt_root=attempt_root,
        child_process_identity=identity,
        expected_parent_argv=expected_parent,
    )
    stage_id = (
        "PHASE1_STAGE_C_HELPER"
        if ablation is not None
        else "PHASE1_STAGE_B_HELPER"
    )
    installed = _begin_static_audit(
        stage_id=stage_id, process_identity=identity
    )
    pre_snapshot = _observe_static_snapshot(
        stage_id=stage_id,
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    operation_started = _monotonic_ns_after(
        pre_snapshot["ended_monotonic_ns"]
    )
    gate = V2.validate_stage_b_gate_receipt(
        _load_canonical_json(
            V2.runtime_path("stage_b_gate_receipt", attempt_root)
        )
    )
    materializer = _materializer_v1()
    donor_mapping = None
    stage_c_gate_digest = None
    if ablation is not None:
        stage_c_gate = V2.validate_stage_c_gate_receipt(
            _load_canonical_json(
                V2.runtime_path("stage_c_gate_receipt", attempt_root)
            )
        )
        stage_c_gate_digest = str(stage_c_gate["content_digest"])
        evaluation_contract = _load_canonical_json(
            attempt_root / "receipts/evaluation_contract.json"
        )
        io_root = _attempt_fd_alias(attempt_root)
        contexts = materializer._load_context_index(
            io_root, str(gate["content_digest"])
        )
        states = {
            str(row["state_id"]): row
            for row in materializer._state_manifest_rows()
        }
        donor_mapping = materializer._stage_c_mapping(
            ablation=ablation,
            evaluation_contract=evaluation_contract,
            states=states,
            contexts=contexts,
        )
    output = materializer._predict_source(
        source_id,
        output_root=_attempt_fd_alias(attempt_root),
        gate_digest=str(gate["content_digest"]),
        stage_c_gate_digest=stage_c_gate_digest,
        ablation=ablation,
        donor_mapping=donor_mapping,
    )
    output_source = source_id if ablation is None else ablation
    if output["source_id"] != output_source:
        raise V2EvaluationError("prediction worker output source drift")
    operation_ended = time.monotonic_ns()
    post_snapshot = _observe_static_snapshot(
        stage_id=stage_id,
        snapshot_kind="POST_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    audit_stream = _finish_static_audit(
        stage_id=stage_id,
        process_identity=identity,
        attempt_root=attempt_root,
        installed_monotonic_ns=installed,
    )
    parent = _process_identity_for_pid(
        identity["ppid"], expected_argv=expected_parent
    )
    process_custody = V2.build_static_input_process_custody(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        static_input_audit_stream=audit_stream,
        parent_process_identity=parent,
        process_started_monotonic_ns=operation_started,
        process_ended_monotonic_ns=operation_ended,
    )
    return V2.build_phase1_worker_receipt(
        mode=V2.PHASE1_PREDICTION_SOURCE_WORKER_MODE,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        parent_helper_process_identity=parent,
        worker_process_identity=identity,
        scientific_output_binding=_artifact_binding(
            attempt_root
            / "stage_b/predictions"
            / str(output_source)
            / "index.json",
            attempt_root=attempt_root,
            rows=1_728,
        ),
        static_input_audit_stream=audit_stream,
        static_input_process_custody=process_custody,
        source_id=source_id,
        ablation=ablation,
    )


def _write_helper_stage_custody(
    *,
    mode: str,
    attempt_root: Path,
    process_identity: Mapping[str, Any],
    parent_process_identity: Mapping[str, Any],
    installed_monotonic_ns: int,
    pre_snapshot: Mapping[str, Any],
    process_started_monotonic_ns: int,
    process_ended_monotonic_ns: int,
    worker_lifecycle_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_id = (
        "PHASE1_STAGE_B_HELPER"
        if mode == V2.PHASE1_PROPRIO_HELPER_MODE
        else "PHASE1_STAGE_C_HELPER"
    )
    post_snapshot = _observe_static_snapshot(
        stage_id=stage_id,
        snapshot_kind="POST_STAGE",
        process_identity=process_identity,
        attempt_root=attempt_root,
    )
    audit_stream = _finish_static_audit(
        stage_id=stage_id,
        process_identity=process_identity,
        attempt_root=attempt_root,
        installed_monotonic_ns=installed_monotonic_ns,
    )
    process_custody = V2.build_static_input_process_custody(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        static_input_audit_stream=audit_stream,
        parent_process_identity=parent_process_identity,
        process_started_monotonic_ns=process_started_monotonic_ns,
        process_ended_monotonic_ns=process_ended_monotonic_ns,
    )
    custody = V2.build_static_input_multi_process_stage_custody(
        helper_process_custody=process_custody,
        worker_lifecycle_manifest=worker_lifecycle_manifest,
    )
    _write_json_exclusive(
        V2.runtime_path(_static_stage_runtime_key(stage_id), attempt_root),
        custody,
    )
    return audit_stream, process_custody


def materialize_proprio_substitution(attempt_root: Path) -> dict[str, Any]:
    identity = _process_identity()
    child_root_binding = _activate_inherited_attempt_io(
        attempt_root=attempt_root,
        child_process_identity=identity,
        expected_parent_argv=V2.PUBLIC_EXECUTE_ARGV,
    )
    installed = _begin_static_audit(
        stage_id="PHASE1_STAGE_B_HELPER", process_identity=identity
    )
    pre_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_B_HELPER",
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    started = _monotonic_ns_after(pre_snapshot["ended_monotonic_ns"])
    parent_identity = child_root_binding[
        "inherited_attempt_root_fd_custody"
    ]["parent_process_identity"]
    runtime_import_custody = _verified_helper_runtime_import_custody(
        V2.PHASE1_PROPRIO_HELPER_MODE
    )
    materializer = _materializer_v1()
    gate = V2.validate_stage_b_gate_receipt(
        _load_canonical_json(
            V2.runtime_path("stage_b_gate_receipt", attempt_root)
        )
    )
    state_rows = materializer._state_manifest_rows()
    if len(state_rows) != 48:
        raise V2EvaluationError("proprio helper state cardinality drift")
    context_custodies: dict[int, dict[str, Any]] = {}
    context_worker_failed = False
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {
            executor.submit(
                _run_phase1_worker_process,
                mode=V2.PHASE1_CONTEXT_STATE_WORKER_MODE,
                attempt_root=attempt_root,
                parent_helper_process_identity=identity,
                state_index=index,
                state_id=str(row["state_id"]),
            ): index
            for index, row in enumerate(state_rows)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                context_custodies[index] = future.result()
            except V2Phase1WorkerProcessError as exc:
                context_custodies[index] = dict(exc.exit_custody)
                context_worker_failed = True
    ordered_context_custodies = [
        context_custodies[index] for index in range(48)
    ]
    if context_worker_failed:
        _persist_worker_failure_manifest(
            helper_mode=V2.PHASE1_PROPRIO_HELPER_MODE,
            attempt_root=attempt_root,
            parent_helper_process_identity=identity,
            worker_exit_custodies=ordered_context_custodies,
        )
        raise V2EvaluationError("proprio context worker failed")
    io_root = _attempt_fd_alias(attempt_root)
    materializer._write_context_index(
        io_root, str(gate["content_digest"])
    )
    worker_rows = list(ordered_context_custodies)
    for source_id in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        try:
            worker_rows.append(
                _run_phase1_worker_process(
                    mode=V2.PHASE1_PREDICTION_SOURCE_WORKER_MODE,
                    attempt_root=attempt_root,
                    parent_helper_process_identity=identity,
                    source_id=source_id,
                )
            )
        except V2Phase1WorkerProcessError as exc:
            worker_rows.append(dict(exc.exit_custody))
            _persist_worker_failure_manifest(
                helper_mode=V2.PHASE1_PROPRIO_HELPER_MODE,
                attempt_root=attempt_root,
                parent_helper_process_identity=identity,
                worker_exit_custodies=worker_rows,
            )
            raise V2EvaluationError(
                f"proprio prediction worker failed: {source_id}"
            ) from exc
    _require_no_active_helper_worker_pids()
    worker_manifest = V2.build_phase1_worker_lifecycle_manifest(
        helper_mode=V2.PHASE1_PROPRIO_HELPER_MODE,
        attempt_root=attempt_root,
        parent_helper_process_identity=identity,
        worker_exit_custodies=worker_rows,
    )
    worker_manifest_path = V2.runtime_path(
        "proprio_worker_lifecycle_manifest", attempt_root
    )
    _write_json_exclusive(worker_manifest_path, worker_manifest)
    worker_manifest_binding = _artifact_binding(
        worker_manifest_path, attempt_root=attempt_root, rows=50
    )
    prediction_bindings = {
        source: _artifact_binding(
            attempt_root / "stage_b/predictions" / source / "index.json",
            attempt_root=attempt_root,
            rows=1_728,
        )
        for source in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT")
    }
    helper_manifest = V2.build_phase1_helper_manifest(
        mode=V2.PHASE1_PROPRIO_HELPER_MODE,
        context_index_binding=_artifact_binding(
            attempt_root / "stage_b/proprio_context/index.json",
            attempt_root=attempt_root,
            rows=48,
        ),
        prediction_index_bindings=prediction_bindings,
        worker_lifecycle_manifest_binding=worker_manifest_binding,
    )
    manifest_path = V2.runtime_path("proprio_helper_manifest", attempt_root)
    _write_json_exclusive(manifest_path, helper_manifest)
    process_ended = time.monotonic_ns()
    audit_stream, process_custody = _write_helper_stage_custody(
        mode=V2.PHASE1_PROPRIO_HELPER_MODE,
        attempt_root=attempt_root,
        process_identity=identity,
        parent_process_identity=parent_identity,
        installed_monotonic_ns=installed,
        pre_snapshot=pre_snapshot,
        process_started_monotonic_ns=started,
        process_ended_monotonic_ns=process_ended,
        worker_lifecycle_manifest=worker_manifest,
    )
    return V2.build_phase1_helper_receipt(
        mode=V2.PHASE1_PROPRIO_HELPER_MODE,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        runtime_import_custody=runtime_import_custody,
        process_identity=identity,
        output_manifest_binding=_artifact_binding(
            manifest_path, attempt_root=attempt_root, rows=96
        ),
        worker_lifecycle_manifest_binding=worker_manifest_binding,
        static_input_audit_stream=audit_stream,
        static_input_process_custody=process_custody,
        counters=copy.deepcopy(
            V2.PHASE1_HELPER_COUNTER_AUTHORITY[
                V2.PHASE1_PROPRIO_HELPER_MODE
            ]
        ),
    )


def materialize_stage_c_attribution(attempt_root: Path) -> dict[str, Any]:
    identity = _process_identity()
    child_root_binding = _activate_inherited_attempt_io(
        attempt_root=attempt_root,
        child_process_identity=identity,
        expected_parent_argv=V2.PUBLIC_EXECUTE_ARGV,
    )
    installed = _begin_static_audit(
        stage_id="PHASE1_STAGE_C_HELPER", process_identity=identity
    )
    pre_snapshot = _observe_static_snapshot(
        stage_id="PHASE1_STAGE_C_HELPER",
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    started = _monotonic_ns_after(pre_snapshot["ended_monotonic_ns"])
    parent_identity = child_root_binding[
        "inherited_attempt_root_fd_custody"
    ]["parent_process_identity"]
    runtime_import_custody = _verified_helper_runtime_import_custody(
        V2.PHASE1_STAGE_C_HELPER_MODE
    )
    materializer = _materializer_v1()
    gate = V2.validate_stage_b_gate_receipt(
        _load_canonical_json(
            V2.runtime_path("stage_b_gate_receipt", attempt_root)
        )
    )
    stage_c_gate = V2.validate_stage_c_gate_receipt(
        _load_canonical_json(
            V2.runtime_path("stage_c_gate_receipt", attempt_root)
        )
    )
    evaluation_contract = _load_canonical_json(
        attempt_root / "receipts/evaluation_contract.json"
    )
    io_root = _attempt_fd_alias(attempt_root)
    contexts = materializer._load_context_index(
        io_root, str(gate["content_digest"])
    )
    states = {
        str(row["state_id"]): row
        for row in materializer._state_manifest_rows()
    }
    donor_rows = []
    for ablation in materializer.STAGE_C_ABLATIONS:
        mapping = materializer._stage_c_mapping(
            ablation=ablation,
            evaluation_contract=evaluation_contract,
            states=states,
            contexts=contexts,
        )
        donor_rows.extend(
            {
                "ablation": ablation,
                "recipient_state_id": recipient,
                "donor_state_id": mapping[recipient],
            }
            for recipient in sorted(mapping, key=lambda value: int(value.rsplit("-", 1)[1]))
        )
    donor_document = V2.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v2."
                "stage_c_donor_mappings.v1"
            ),
            "stage_b_gate_content_digest": gate["content_digest"],
            "stage_c_gate_content_digest": stage_c_gate["content_digest"],
            "row_count": len(donor_rows),
            "rows": donor_rows,
            "outcome_informed_selection": False,
        }
    )
    donor_path = V2.runtime_path("stage_c_donor_mappings", attempt_root)
    _write_json_exclusive(donor_path, donor_document)
    worker_rows = []
    for ablation in materializer.STAGE_C_ABLATIONS:
        try:
            worker_rows.append(
                _run_phase1_worker_process(
                    mode=V2.PHASE1_PREDICTION_SOURCE_WORKER_MODE,
                    attempt_root=attempt_root,
                    parent_helper_process_identity=identity,
                    source_id="PR_PROPRIO_ROLLOUT",
                    ablation=ablation,
                )
            )
        except V2Phase1WorkerProcessError as exc:
            worker_rows.append(dict(exc.exit_custody))
            _persist_worker_failure_manifest(
                helper_mode=V2.PHASE1_STAGE_C_HELPER_MODE,
                attempt_root=attempt_root,
                parent_helper_process_identity=identity,
                worker_exit_custodies=worker_rows,
            )
            raise V2EvaluationError(
                f"Stage-C prediction worker failed: {ablation}"
            ) from exc
    _require_no_active_helper_worker_pids()
    worker_manifest = V2.build_phase1_worker_lifecycle_manifest(
        helper_mode=V2.PHASE1_STAGE_C_HELPER_MODE,
        attempt_root=attempt_root,
        parent_helper_process_identity=identity,
        worker_exit_custodies=worker_rows,
    )
    worker_manifest_path = V2.runtime_path(
        "stage_c_worker_lifecycle_manifest", attempt_root
    )
    _write_json_exclusive(worker_manifest_path, worker_manifest)
    worker_manifest_binding = _artifact_binding(
        worker_manifest_path, attempt_root=attempt_root, rows=3
    )
    prediction_bindings = {
        ablation: _artifact_binding(
            attempt_root / "stage_b/predictions" / ablation / "index.json",
            attempt_root=attempt_root,
            rows=1_728,
        )
        for ablation in materializer.STAGE_C_ABLATIONS
    }
    donor_binding = _artifact_binding(
        donor_path, attempt_root=attempt_root, rows=144
    )
    helper_manifest = V2.build_phase1_helper_manifest(
        mode=V2.PHASE1_STAGE_C_HELPER_MODE,
        context_index_binding=None,
        prediction_index_bindings=prediction_bindings,
        worker_lifecycle_manifest_binding=worker_manifest_binding,
        donor_mapping_binding=donor_binding,
    )
    manifest_path = V2.runtime_path("stage_c_helper_manifest", attempt_root)
    _write_json_exclusive(manifest_path, helper_manifest)
    process_ended = time.monotonic_ns()
    audit_stream, process_custody = _write_helper_stage_custody(
        mode=V2.PHASE1_STAGE_C_HELPER_MODE,
        attempt_root=attempt_root,
        process_identity=identity,
        parent_process_identity=parent_identity,
        installed_monotonic_ns=installed,
        pre_snapshot=pre_snapshot,
        process_started_monotonic_ns=started,
        process_ended_monotonic_ns=process_ended,
        worker_lifecycle_manifest=worker_manifest,
    )
    return V2.build_phase1_helper_receipt(
        mode=V2.PHASE1_STAGE_C_HELPER_MODE,
        attempt_root=attempt_root,
        child_attempt_root_fd_binding=child_root_binding,
        runtime_import_custody=runtime_import_custody,
        process_identity=identity,
        output_manifest_binding=_artifact_binding(
            manifest_path, attempt_root=attempt_root, rows=144
        ),
        worker_lifecycle_manifest_binding=worker_manifest_binding,
        static_input_audit_stream=audit_stream,
        static_input_process_custody=process_custody,
        counters=copy.deepcopy(
            V2.PHASE1_HELPER_COUNTER_AUTHORITY[
                V2.PHASE1_STAGE_C_HELPER_MODE
            ]
        ),
        donor_mapping_binding=donor_binding,
    )


def _persist_terminal_child_bytes(
    attempt_root: Path, relative_path: str, payload: bytes
) -> None:
    path = attempt_root / relative_path
    try:
        _write_bytes_exclusive(path, payload)
    except FileExistsError:
        if _read_attempt_bytes(path) != payload:
            raise V2EvaluationError(
                f"terminal child artifact already differs: {relative_path}"
            )


def _terminal_child_paths(
    *, phase_id: str, publication_attempt_number: int | None,
    publication_attempt_id: str | None,
) -> dict[str, str]:
    if phase_id == V2.PHASE_IDS[1]:
        return {
            "stdout": V2.RUNTIME_PATHS["phase_2_stdout"],
            "stderr": V2.RUNTIME_PATHS["phase_2_stderr"],
            "traceback": V2.RUNTIME_PATHS["phase_2_traceback"],
            "exception": V2.RUNTIME_PATHS["phase_2_exception"],
            "last_stage": V2.RUNTIME_PATHS["phase_2_last_stage"],
            "handoff": V2.RUNTIME_PATHS["phase_2_supervisor_handoff"],
            "produced": V2.RUNTIME_PATHS[
                "independent_validation_receipt"
            ],
        }
    if (
        phase_id != V2.PHASE_IDS[2]
        or publication_attempt_number is None
        or publication_attempt_id is None
    ):
        raise V2EvaluationError("terminal child path phase drift")
    paths = V2.publication_attempt_runtime_paths(
        publication_attempt_number, publication_attempt_id
    )
    return {**paths, "produced": paths["manifest"]}


def _terminal_child_last_stage(
    *, attempt_root: Path, relative_path: str, phase_id: str,
    child_process_identity: Mapping[str, Any] | None,
    spawned_pid: int | None,
) -> str:
    path = attempt_root / relative_path
    try:
        value = _load_canonical_json(path)
    except FileNotFoundError:
        value = V2.attach_self_digest(
            {
                "schema": (
                    "plan_aware_monotone_jepa_cost_v2."
                    "supervisor_fallback_last_stage.v1"
                ),
                "phase_id": phase_id,
                "stage": "STARTED",
                "child_pid": spawned_pid,
                "child_process_identity_observed": (
                    child_process_identity is not None
                ),
                "observer_pid": os.getpid(),
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        _persist_terminal_child_bytes(
            attempt_root,
            relative_path,
            V2.canonical_json_bytes(value) + b"\n",
        )
    stage = value.get("stage")
    if stage not in V2.PHASE_STAGE_IDS[phase_id]:
        raise V2EvaluationError("terminal child last-stage drift")
    return str(stage)


def _load_terminal_child_handoff_if_present(
    *, attempt_root: Path, relative_path: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        receipt = V2.validate_terminal_publication_child_handoff_receipt(
            _load_canonical_json(attempt_root / relative_path)
        )
    except FileNotFoundError:
        return None, None
    return receipt, _artifact_binding(
        attempt_root / relative_path,
        attempt_root=attempt_root,
        rows=None,
    )


def _terminal_supervisor_exception(
    error: BaseException,
) -> dict[str, str]:
    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback_text,
        "traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
    }


def _persist_pre_phase2_supervisor_failure(
    *,
    phase_1_custody: Mapping[str, Any],
    supervisor_process_identity: Mapping[str, Any],
    publication_attempt_id: str,
    tracked_attempt_id: str,
    failure_stage: str,
    error: BaseException,
    failure_started_monotonic_ns: int,
    terminal_publication_supervisor_root_reopen_custody: (
        Mapping[str, Any] | None
    ) = None,
    terminal_publication_supervisor_preflight: (
        Mapping[str, Any] | None
    ) = None,
    popen_succeeded: bool = False,
    spawned_pid: int | None = None,
    spawned_process_group_id: int | None = None,
    child_process_identity: Mapping[str, Any] | None = None,
    identity_observation_error: Mapping[str, Any] | None = None,
    returncode: int | None = None,
    termination_signal: int | None = None,
    cleanup_actions: Sequence[str] = (),
    process_group_members_after_cleanup: Sequence[
        Mapping[str, Any]
    ] = (),
    communication_attempted: bool = False,
    communicate_completed: bool = False,
    process_reaped: bool = False,
) -> dict[str, Any]:
    observation = V2.observe_terminal_publication_supervisor_failure(
        phase_1_custody=phase_1_custody,
        supervisor_process_identity=supervisor_process_identity,
        publication_attempt_id=publication_attempt_id,
        tracked_attempt_id=tracked_attempt_id,
        failure_stage=failure_stage,
        structured_exception=_terminal_supervisor_exception(error),
        failure_started_monotonic_ns=failure_started_monotonic_ns,
        failure_observed_monotonic_ns=time.monotonic_ns(),
        terminal_publication_supervisor_root_reopen_custody=(
            terminal_publication_supervisor_root_reopen_custody
        ),
        terminal_publication_supervisor_preflight=(
            terminal_publication_supervisor_preflight
        ),
        popen_succeeded=popen_succeeded,
        spawned_pid=spawned_pid,
        spawned_process_group_id=spawned_process_group_id,
        child_process_identity=child_process_identity,
        identity_observation_error=identity_observation_error,
        returncode=returncode,
        termination_signal=termination_signal,
        cleanup_actions=cleanup_actions,
        process_group_members_after_cleanup=(
            process_group_members_after_cleanup
        ),
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
    )
    custody = V2.build_terminal_publication_supervisor_failure_custody(
        phase_1_custody=phase_1_custody,
        failure_observation=observation,
    )
    V2.write_terminal_publication_supervisor_failure_custody_exclusive_fsync(
        failure_custody=custody,
        supervisor_process_identity=supervisor_process_identity,
    )
    return V2.validate_terminal_publication_supervisor_failure_custody(
        custody
    )


def _persist_raw_phase3_supervisor_failure(
    *,
    attempt_root: Path,
    phase_2_custody: Mapping[str, Any],
    supervisor_root_reopen_custody: Mapping[str, Any],
    publication_attempt_number: int,
    publication_attempt_id: str,
    presentation_retry_authority: Mapping[str, Any] | None,
    supervisor_resume_preflight: Mapping[str, Any] | None,
    failure_stage: str,
    failure_errors: Sequence[BaseException],
    handoff_memfd_custody: Mapping[str, Any] | None,
    handoff_receipt: Mapping[str, Any] | None,
    handoff_binding: Mapping[str, Any] | None,
    child_process_identity: Mapping[str, Any] | None,
    process: subprocess.Popen[bytes] | None,
    identity_observation_error: Mapping[str, Any] | None,
    captured_stdout: bytes,
    captured_stderr: bytes,
    captured_traceback: bytes,
    started_monotonic_ns: int,
    environment_names: Sequence[str],
    environment_sha256: str,
    cleanup_actions: Sequence[str],
    process_group_members_after_cleanup: Sequence[Mapping[str, Any]],
    communication_attempted: bool,
    communicate_completed: bool,
    process_reaped: bool,
) -> dict[str, Any]:
    """Freeze and preserve an uncanonicalized Phase-3 supervisor failure."""

    errors = list(failure_errors)
    if not errors:
        errors.append(V2EvaluationError("raw Phase-3 failure lacked cause"))
    root_fd, historical, active_reopen = _require_attempt_io(attempt_root)
    if active_reopen != supervisor_root_reopen_custody:
        raise V2EvaluationError("raw Phase-3 root custody drift")
    returncode = None if process is None else process.returncode
    termination_signal = (
        -returncode
        if type(returncode) is int and returncode < 0
        else None
    )
    observation = V2.observe_raw_phase3_supervisor_failure(
        phase_2_custody=phase_2_custody,
        terminal_publication_supervisor_root_reopen_custody=(
            supervisor_root_reopen_custody
        ),
        root_fd=root_fd,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=presentation_retry_authority,
        terminal_publication_supervisor_resume_preflight=(
            supervisor_resume_preflight
        ),
        failure_stage=failure_stage,
        failure_exceptions=[
            {
                "stage": failure_stage,
                "exception": _terminal_supervisor_exception(error),
            }
            for error in errors
        ],
        terminal_publication_child_handoff_memfd_custody=(
            handoff_memfd_custody
        ),
        terminal_publication_child_handoff_receipt=handoff_receipt,
        terminal_publication_child_handoff_binding=handoff_binding,
        child_process_identity=child_process_identity,
        spawned_pid=None if process is None else process.pid,
        spawned_process_group_id=None if process is None else process.pid,
        popen_succeeded=process is not None,
        identity_observation_error=identity_observation_error,
        captured_stdout=captured_stdout,
        captured_stderr=captured_stderr,
        captured_traceback=captured_traceback,
        returncode=returncode,
        termination_signal=termination_signal,
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        environment_names=environment_names,
        environment_sha256=environment_sha256,
        cleanup_actions=cleanup_actions,
        process_group_members_after_cleanup=(
            process_group_members_after_cleanup
        ),
        communication_attempted=communication_attempted,
        communicate_completed=communicate_completed,
        process_reaped=process_reaped,
    )
    observation = V2.validate_raw_phase3_supervisor_failure_observation(
        observation
    )
    custody = V2.build_raw_phase3_supervisor_failure_custody(
        failure_observation=observation
    )
    custody = V2.validate_raw_phase3_supervisor_failure_custody(custody)
    persisted = (
        V2.write_raw_phase3_supervisor_failure_custody_exclusive_fsync(
            failure_custody=custody,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=active_reopen,
            root_fd=root_fd,
            supervisor_process_identity=active_reopen[
                "supervisor_process_identity"
            ],
        )
    )
    if (
        V2.validate_raw_phase3_supervisor_failure_custody(
            persisted["failure_custody"]
        )
        != custody
    ):
        raise V2EvaluationError("raw Phase-3 persisted custody drift")
    V2.validate_raw_phase3_failure_preservation_receipt(
        persisted["preservation_receipt"]
    )
    return persisted


def _run_terminal_publication_child(
    *,
    phase_id: str,
    attempt_root: Path,
    phase_1_custody: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any] | None,
    supervisor_root_reopen_custody: Mapping[str, Any],
    supervisor_preflight: Mapping[str, Any] | None,
    supervisor_resume_preflight: Mapping[str, Any] | None,
    presentation_retry_authority: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    is_phase_2 = phase_id == V2.PHASE_IDS[1]
    supervisor_identity = supervisor_root_reopen_custody[
        "supervisor_process_identity"
    ]
    handoff_started = time.monotonic_ns()
    handoff_fd = -1
    handoff_memfd_custody: dict[str, Any] | None = None
    handoff_failure_stage = "CHILD_HANDOFF_CONSTRUCTION"
    environment = _internal_environment()
    environment_names = sorted(environment)
    environment_sha256 = V2.canonical_json_sha256(environment)
    publication_number = None
    publication_id = None
    try:
        if is_phase_2:
            command = V2.expected_internal_argv(
                V2.PHASE2_VALIDATOR_MODE, attempt_root
            )
            publication_number = None
            publication_id = None
        elif phase_id == V2.PHASE_IDS[2]:
            publication_number = supervisor_root_reopen_custody[
                "publication_attempt_number"
            ]
            publication_id = supervisor_root_reopen_custody[
                "publication_attempt_id"
            ]
            command = V2.expected_internal_argv(
                V2.PHASE3_PUBLISHER_MODE,
                attempt_root,
                publication_attempt_number=publication_number,
                publication_attempt_id=publication_id,
                presentation_retry_authority_content_digest=(
                    None
                    if presentation_retry_authority is None
                    else presentation_retry_authority["content_digest"]
                ),
            )
        else:
            raise V2EvaluationError("terminal child phase drift")
        paths = _terminal_child_paths(
            phase_id=phase_id,
            publication_attempt_number=publication_number,
            publication_attempt_id=publication_id,
        )
        handoff_fd, handoff_memfd_custody = (
            V2.create_terminal_publication_child_handoff_memfd(
                phase_id=phase_id,
                phase_1_custody=phase_1_custody,
                phase_2_custody=phase_2_custody,
                terminal_publication_supervisor_root_reopen_custody=(
                    supervisor_root_reopen_custody
                ),
                terminal_publication_supervisor_preflight=(
                    supervisor_preflight
                ),
                terminal_publication_supervisor_resume_preflight=(
                    supervisor_resume_preflight
                ),
                presentation_retry_authority=presentation_retry_authority,
            )
        )
        environment = V2.build_terminal_publication_child_environment(
            base_environment=_internal_environment(),
            handoff_memfd_custody=handoff_memfd_custody,
        )
        environment_names = sorted(environment)
        environment_sha256 = V2.canonical_json_sha256(environment)
        handoff_failure_stage = "CHILD_HANDOFF_PERSISTENCE"
        handoff_pass_fds = (
            V2.terminal_publication_child_handoff_pass_fds(
                handoff_memfd_custody
            )
        )
    except BaseException as error:
        if handoff_fd >= 0:
            try:
                os.close(handoff_fd)
            except OSError:
                pass
        if is_phase_2:
            return (
                _persist_pre_phase2_supervisor_failure(
                    phase_1_custody=phase_1_custody,
                    supervisor_process_identity=supervisor_identity,
                    publication_attempt_id=(
                        supervisor_root_reopen_custody[
                            "publication_attempt_id"
                        ]
                    ),
                    tracked_attempt_id=supervisor_root_reopen_custody[
                        "tracked_attempt_id"
                    ],
                    failure_stage="CHILD_HANDOFF",
                    error=error,
                    failure_started_monotonic_ns=handoff_started,
                    terminal_publication_supervisor_root_reopen_custody=(
                        supervisor_root_reopen_custody
                    ),
                    terminal_publication_supervisor_preflight=(
                        supervisor_preflight
                    ),
                ),
                None,
            )
        if phase_2_custody is None or publication_number is None or (
            publication_id is None
        ):
            raise V2EvaluationError(
                "raw Phase-3 handoff failure lacks phase authority"
            ) from error
        persisted = _persist_raw_phase3_supervisor_failure(
            attempt_root=attempt_root,
            phase_2_custody=phase_2_custody,
            supervisor_root_reopen_custody=(
                supervisor_root_reopen_custody
            ),
            publication_attempt_number=publication_number,
            publication_attempt_id=publication_id,
            presentation_retry_authority=presentation_retry_authority,
            supervisor_resume_preflight=supervisor_resume_preflight,
            failure_stage=handoff_failure_stage,
            failure_errors=[error],
            handoff_memfd_custody=(
                handoff_memfd_custody
                if handoff_failure_stage == "CHILD_HANDOFF_PERSISTENCE"
                else None
            ),
            handoff_receipt=None,
            handoff_binding=None,
            child_process_identity=None,
            process=None,
            identity_observation_error=None,
            captured_stdout=b"",
            captured_stderr=b"",
            captured_traceback="".join(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            ).encode("utf-8"),
            started_monotonic_ns=handoff_started,
            environment_names=environment_names,
            environment_sha256=environment_sha256,
            cleanup_actions=[],
            process_group_members_after_cleanup=[],
            communication_attempted=False,
            communicate_completed=False,
            process_reaped=False,
        )
        return persisted["preservation_receipt"], None
    environment_names = sorted(environment)
    environment_sha256 = V2.canonical_json_sha256(environment)
    process: subprocess.Popen[bytes] | None = None
    identity: dict[str, Any] | None = None
    stdout = b""
    stderr = b""
    local_traceback = b""
    cleanup_actions: list[str] = []
    communication_attempted = False
    communicate_completed = False
    process_reaped = False
    lifecycle_error: BaseException | None = None
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command,
            cwd=V2.REPO_ROOT,
            env=environment,
            pass_fds=handoff_pass_fds,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        identity = _process_identity_for_pid(
            process.pid, expected_argv=command
        )
        communication_attempted = True
        stdout, stderr = process.communicate()
        communicate_completed = True
        process_reaped = process.returncode is not None
    except BaseException as error:
        lifecycle_error = error
        local_traceback = traceback.format_exc().encode("utf-8")
        if process is not None:
            communication_attempted = True
            try:
                if process.poll() is None:
                    cleanup_actions.extend(_terminate_process_group(process))
            except BaseException as cleanup_error:
                if lifecycle_error is None:
                    lifecycle_error = cleanup_error
            try:
                drained_stdout, drained_stderr, drained = (
                    _bounded_process_drain(
                        process,
                        timeout_seconds=float(
                            V2.PROCESS_LIFECYCLE_POLICY[
                                "kill_grace_seconds"
                            ]
                        ),
                    )
                )
            except BaseException as drain_error:
                if lifecycle_error is None:
                    lifecycle_error = drain_error
            else:
                stdout += drained_stdout
                stderr += drained_stderr
                communicate_completed = drained
            process_reaped = process.returncode is not None
    finally:
        try:
            os.close(handoff_fd)
        except OSError:
            pass
    if process is not None:
        try:
            if process.poll() is None:
                cleanup_actions.extend(_terminate_process_group(process))
            cleanup_actions.extend(
                _cleanup_remaining_process_group(process.pid)
            )
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        except BaseException as cleanup_error:
            if lifecycle_error is None:
                lifecycle_error = cleanup_error
                local_traceback = "".join(
                    traceback.format_exception(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    )
                ).encode("utf-8")
        process_reaped = process.returncode is not None
    ended = time.monotonic_ns()
    popen_succeeded = process is not None
    spawned_pid = None if process is None else process.pid
    returncode = None if process is None else process.returncode
    try:
        remaining = (
            [] if process is None else _process_group_members(process.pid)
        )
    except BaseException as residual_error:
        remaining = []
        if lifecycle_error is None:
            lifecycle_error = residual_error
            local_traceback = "".join(
                traceback.format_exception(
                    type(residual_error),
                    residual_error,
                    residual_error.__traceback__,
                )
            ).encode("utf-8")
    lifecycle_incomplete = (
        process is not None
        and (
            identity is None
            or not communicate_completed
            or not process_reaped
            or cleanup_actions != []
            or remaining != []
        )
    )
    if is_phase_2 and (
        lifecycle_error is not None or process is None or lifecycle_incomplete
    ):
        failure_error = lifecycle_error or V2EvaluationError(
            "Phase-2 child lifecycle did not reach a clean reaped state"
        )
        identity_error = (
            _terminal_supervisor_exception(failure_error)
            if process is not None and identity is None
            else None
        )
        return (
            _persist_pre_phase2_supervisor_failure(
                phase_1_custody=phase_1_custody,
                supervisor_process_identity=supervisor_identity,
                publication_attempt_id=supervisor_root_reopen_custody[
                    "publication_attempt_id"
                ],
                tracked_attempt_id=supervisor_root_reopen_custody[
                    "tracked_attempt_id"
                ],
                failure_stage=(
                    "CHILD_POPEN"
                    if process is None
                    else "CHILD_IDENTITY_OR_LIFECYCLE"
                ),
                error=failure_error,
                failure_started_monotonic_ns=started,
                terminal_publication_supervisor_root_reopen_custody=(
                    supervisor_root_reopen_custody
                ),
                terminal_publication_supervisor_preflight=(
                    supervisor_preflight
                ),
                popen_succeeded=process is not None,
                spawned_pid=spawned_pid,
                spawned_process_group_id=spawned_pid,
                child_process_identity=identity,
                identity_observation_error=identity_error,
                returncode=returncode,
                termination_signal=(
                    -returncode
                    if type(returncode) is int and returncode < 0
                    else None
                ),
                cleanup_actions=cleanup_actions,
                process_group_members_after_cleanup=remaining,
                communication_attempted=communication_attempted,
                communicate_completed=communicate_completed,
                process_reaped=process_reaped,
            ),
            None,
        )
    if not is_phase_2 and (
        lifecycle_error is not None or process is None or lifecycle_incomplete
    ):
        if phase_2_custody is None or publication_number is None or (
            publication_id is None
        ):
            raise V2EvaluationError(
                "raw Phase-3 lifecycle failure lacks phase authority"
            )
        failure_error = lifecycle_error or V2EvaluationError(
            "Phase-3 child lifecycle did not reach a clean reaped state"
        )
        raw_errors: list[BaseException] = [failure_error]
        handoff_receipt = None
        handoff_binding = None
        try:
            handoff_receipt, handoff_binding = (
                _load_terminal_child_handoff_if_present(
                    attempt_root=attempt_root,
                    relative_path=paths["handoff"],
                )
            )
        except BaseException as handoff_error:
            raw_errors.append(handoff_error)
        identity_error = (
            _terminal_supervisor_exception(failure_error)
            if process is not None and identity is None
            else None
        )
        persisted = _persist_raw_phase3_supervisor_failure(
            attempt_root=attempt_root,
            phase_2_custody=phase_2_custody,
            supervisor_root_reopen_custody=(
                supervisor_root_reopen_custody
            ),
            publication_attempt_number=publication_number,
            publication_attempt_id=publication_id,
            presentation_retry_authority=presentation_retry_authority,
            supervisor_resume_preflight=supervisor_resume_preflight,
            failure_stage=(
                "CHILD_POPEN"
                if process is None
                else "CHILD_IDENTITY_OR_LIFECYCLE"
            ),
            failure_errors=raw_errors,
            handoff_memfd_custody=handoff_memfd_custody,
            handoff_receipt=handoff_receipt,
            handoff_binding=handoff_binding,
            child_process_identity=identity,
            process=process,
            identity_observation_error=identity_error,
            captured_stdout=stdout,
            captured_stderr=stderr,
            captured_traceback=(
                local_traceback
                or "".join(
                    traceback.format_exception(
                        type(failure_error),
                        failure_error,
                        failure_error.__traceback__,
                    )
                ).encode("utf-8")
            ),
            started_monotonic_ns=started,
            environment_names=environment_names,
            environment_sha256=environment_sha256,
            cleanup_actions=cleanup_actions,
            process_group_members_after_cleanup=remaining,
            communication_attempted=communication_attempted,
            communicate_completed=communicate_completed,
            process_reaped=process_reaped,
        )
        return persisted["preservation_receipt"], None
    traceback_bytes = (
        b""
        if (
            popen_succeeded
            and identity is not None
            and returncode == 0
            and cleanup_actions == []
            and remaining == []
            and local_traceback == b""
            and stderr == b""
            and communicate_completed
            and process_reaped
        )
        else (
            local_traceback
            or stderr
            or b"terminal child failed without traceback\n"
        )
    )
    failed = traceback_bytes != b""
    structured_exception = (
        None
        if not failed
        else _structured_failure(
            label=(
                "V2TerminalChildPopenFailure"
                if not popen_succeeded
                else (
                    "V2TerminalChildIdentityFailure"
                    if identity is None
                    else "V2TerminalChildFailure"
                )
            ),
            message=(
                f"{phase_id} terminal child failed; "
                f"returncode={returncode}"
            ),
            traceback_bytes=traceback_bytes,
        )
    )
    exit_construction_started = time.monotonic_ns()
    exit_failure_stage = "CHILD_EXIT_OBSERVATION_CONSTRUCTION"
    handoff_receipt = None
    handoff_binding = None
    try:
        _persist_terminal_child_bytes(attempt_root, paths["stdout"], stdout)
        _persist_terminal_child_bytes(attempt_root, paths["stderr"], stderr)
        _persist_terminal_child_bytes(
            attempt_root, paths["traceback"], traceback_bytes
        )
        _persist_terminal_child_bytes(
            attempt_root,
            paths["exception"],
            (
                b""
                if structured_exception is None
                else V2.canonical_json_bytes(structured_exception) + b"\n"
            ),
        )
        last_stage = _terminal_child_last_stage(
            attempt_root=attempt_root,
            relative_path=paths["last_stage"],
            phase_id=phase_id,
            child_process_identity=identity,
            spawned_pid=spawned_pid,
        )
        handoff_receipt, handoff_binding = (
            _load_terminal_child_handoff_if_present(
                attempt_root=attempt_root, relative_path=paths["handoff"]
            )
        )
        produced_binding = None
        try:
            produced_binding = _artifact_binding(
                attempt_root / paths["produced"],
                attempt_root=attempt_root,
                rows=(3 if phase_id == V2.PHASE_IDS[2] else None),
            )
        except FileNotFoundError:
            pass
        identity_exception = (
            structured_exception
            if popen_succeeded and identity is None
            else None
        )
        root_fd, _historical, active_reopen = _require_attempt_io(
            attempt_root
        )
        if active_reopen != supervisor_root_reopen_custody:
            raise V2EvaluationError("terminal supervisor root custody drift")
        observation = V2.observe_terminal_publication_child_exit(
            phase_id=phase_id,
            terminal_publication_supervisor_root_reopen_custody=(
                supervisor_root_reopen_custody
            ),
            root_fd=root_fd,
            terminal_publication_child_handoff_memfd_custody=(
                handoff_memfd_custody
            ),
            terminal_publication_child_handoff_receipt=handoff_receipt,
            terminal_publication_child_handoff_binding=handoff_binding,
            child_process_identity=identity,
            spawned_pid=spawned_pid,
            spawned_process_group_id=spawned_pid,
            popen_succeeded=popen_succeeded,
            identity_observation_error=identity_exception,
            stdout_binding=V2.build_stream_binding(paths["stdout"], stdout),
            stderr_binding=V2.build_stream_binding(paths["stderr"], stderr),
            traceback_binding=V2.build_stream_binding(
                paths["traceback"], traceback_bytes
            ),
            structured_exception=structured_exception,
            last_stage=last_stage,
            returncode=returncode,
            termination_signal=(
                -returncode
                if type(returncode) is int and returncode < 0
                else None
            ),
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
            environment_names=environment_names,
            environment_sha256=environment_sha256,
            cleanup_actions=cleanup_actions,
            process_group_members_after_cleanup=remaining,
            communication_attempted=communication_attempted,
            communicate_completed=communicate_completed,
            process_reaped=process_reaped,
            produced_artifact_binding=produced_binding,
            prior_phase_custody=(
                phase_1_custody
                if phase_id == V2.PHASE_IDS[1]
                else phase_2_custody
            ),
            publication_attempt_number=publication_number,
            publication_attempt_id=publication_id,
            presentation_retry_authority=presentation_retry_authority,
            terminal_publication_supervisor_resume_preflight=(
                supervisor_resume_preflight
            ),
        )
        prior = (
            phase_1_custody
            if phase_id == V2.PHASE_IDS[1]
            else phase_2_custody
        )
        exit_failure_stage = "CHILD_EXIT_OBSERVATION_PERSISTENCE"
        V2.write_terminal_publication_child_exit_observation_exclusive_fsync(
            child_exit_observation=observation,
            prior_phase_custody=prior,
            root_fd=root_fd,
        )
        custody = None
        if (
            phase_id == V2.PHASE_IDS[1]
            or observation["observation_complete"] is True
        ):
            exit_failure_stage = "PHASE_CUSTODY_CONSTRUCTION"
            custody = V2.build_terminal_publication_phase_custody(
                child_exit_observation=observation,
                prior_phase_custody=prior,
            )
            exit_failure_stage = "PHASE_CUSTODY_PERSISTENCE"
            V2.write_terminal_publication_phase_custody_exclusive_fsync(
                phase_custody=custody,
                prior_phase_custody=prior,
                root_fd=root_fd,
            )
        return observation, custody
    except BaseException as error:
        if is_phase_2:
            return (
                _persist_pre_phase2_supervisor_failure(
                    phase_1_custody=phase_1_custody,
                    supervisor_process_identity=supervisor_identity,
                    publication_attempt_id=(
                        supervisor_root_reopen_custody[
                            "publication_attempt_id"
                        ]
                    ),
                    tracked_attempt_id=supervisor_root_reopen_custody[
                        "tracked_attempt_id"
                    ],
                    failure_stage="CHILD_EXIT_OBSERVATION_CONSTRUCTION",
                    error=error,
                    failure_started_monotonic_ns=(
                        exit_construction_started
                    ),
                    terminal_publication_supervisor_root_reopen_custody=(
                        supervisor_root_reopen_custody
                    ),
                    terminal_publication_supervisor_preflight=(
                        supervisor_preflight
                    ),
                    popen_succeeded=True,
                    spawned_pid=spawned_pid,
                    spawned_process_group_id=spawned_pid,
                    child_process_identity=identity,
                    identity_observation_error=None,
                    returncode=returncode,
                    termination_signal=(
                        -returncode
                        if type(returncode) is int and returncode < 0
                        else None
                    ),
                    cleanup_actions=cleanup_actions,
                    process_group_members_after_cleanup=remaining,
                    communication_attempted=communication_attempted,
                    communicate_completed=communicate_completed,
                    process_reaped=process_reaped,
                ),
                None,
            )
        if phase_2_custody is None or publication_number is None or (
            publication_id is None
        ):
            raise V2EvaluationError(
                "raw Phase-3 exit failure lacks phase authority"
            ) from error
        persisted = _persist_raw_phase3_supervisor_failure(
            attempt_root=attempt_root,
            phase_2_custody=phase_2_custody,
            supervisor_root_reopen_custody=(
                supervisor_root_reopen_custody
            ),
            publication_attempt_number=publication_number,
            publication_attempt_id=publication_id,
            presentation_retry_authority=presentation_retry_authority,
            supervisor_resume_preflight=supervisor_resume_preflight,
            failure_stage=exit_failure_stage,
            failure_errors=[error],
            handoff_memfd_custody=handoff_memfd_custody,
            handoff_receipt=handoff_receipt,
            handoff_binding=handoff_binding,
            child_process_identity=identity,
            process=process,
            identity_observation_error=None,
            captured_stdout=stdout,
            captured_stderr=stderr,
            captured_traceback=traceback_bytes,
            started_monotonic_ns=exit_construction_started,
            environment_names=environment_names,
            environment_sha256=environment_sha256,
            cleanup_actions=cleanup_actions,
            process_group_members_after_cleanup=remaining,
            communication_attempted=communication_attempted,
            communicate_completed=communicate_completed,
            process_reaped=process_reaped,
        )
        return persisted["preservation_receipt"], None


def _load_terminal_phase_custody(
    *,
    attempt_root: Path,
    phase_index: int,
    publication_attempt_number: int,
    publication_attempt_id: str,
    expected_content_digest: str,
) -> dict[str, Any]:
    if phase_index == 0:
        relative_path = V2.RUNTIME_PATHS["phase_1_custody"]
    elif phase_index == 1:
        relative_path = V2.RUNTIME_PATHS["phase_2_custody"]
    elif phase_index == 2:
        relative_path = V2.publication_attempt_runtime_paths(
            publication_attempt_number, publication_attempt_id
        )["custody"]
    else:
        raise V2EvaluationError("terminal phase-custody index drift")
    if _ACTIVE_ATTEMPT_ROOT is None:
        value = _load_attempt_json_bootstrap(attempt_root, relative_path)
    else:
        value = _load_canonical_json(attempt_root / relative_path)
    custody = V2.validate_phase_custody(value)
    disposition = V2.validate_terminal_scientific_attempt_disposition(
        custody["scientific_attempt_disposition"]
    )
    if (
        custody["phase_id"] != V2.PHASE_IDS[phase_index]
        or custody["content_digest"] != expected_content_digest
        or custody["pass"] is not True
        or disposition["later_phase_continuation_authorized"] is not True
    ):
        raise V2EvaluationError("terminal prior-phase custody did not pass")
    if phase_index == 0 and disposition["disposition"] != (
        "SCIENCE_EXECUTED_PENDING_VALIDATION"
    ):
        raise V2EvaluationError("Phase-1 disposition is not pending validation")
    if phase_index in (1, 2) and disposition["disposition"] != (
        "SCIENTIFIC_ATTEMPT_COMPLETE"
    ):
        raise V2EvaluationError("validated science disposition is incomplete")
    return custody


def _load_terminal_retry_authorities(
    *,
    attempt_root: Path,
    publication_attempt_number: int,
    publication_attempt_id: str,
    presentation_retry_content_digest: str | None,
    tracked_attempt_number: int,
    tracked_attempt_id: str,
    tracked_retry_content_digest: str | None,
    preloaded_presentation_retry_authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    presentation_retry = None
    if publication_attempt_number > 1:
        paths = V2.publication_attempt_runtime_paths(
            publication_attempt_number, publication_attempt_id
        )
        presentation_retry = V2.validate_presentation_retry_authority(
            _load_canonical_json(attempt_root / paths["retry_authority"]),
            expected_attempt=publication_attempt_number,
        )
        if presentation_retry["content_digest"] != (
            presentation_retry_content_digest
        ):
            raise V2EvaluationError("presentation retry-authority digest drift")
    elif presentation_retry_content_digest is not None:
        raise V2EvaluationError("initial presentation has retry authority")
    if presentation_retry != preloaded_presentation_retry_authority:
        raise V2EvaluationError(
            "presentation retry authority changed across root reopen"
        )

    tracked_retry = None
    if tracked_attempt_number > 1:
        paths = V2.tracked_publication_attempt_runtime_paths(
            tracked_attempt_number, tracked_attempt_id
        )
        tracked_retry = V2.validate_tracked_publication_retry_authority(
            _load_canonical_json(attempt_root / paths["retry_authority"])
        )
        if (
            tracked_retry["content_digest"] != tracked_retry_content_digest
            or tracked_retry["next_tracked_attempt_number"]
            != tracked_attempt_number
            or tracked_retry["next_tracked_attempt_id"]
            != tracked_attempt_id
        ):
            raise V2EvaluationError("tracked retry-authority digest drift")
    elif tracked_retry_content_digest is not None:
        raise V2EvaluationError("initial tracked publication has retry authority")
    return presentation_retry, tracked_retry


def _load_presentation_retry_authority_before_reopen(
    *,
    attempt_root: Path,
    publication_attempt_number: int,
    publication_attempt_id: str,
    expected_content_digest: str | None,
) -> dict[str, Any] | None:
    """Bootstrap the full retry authority before the root-reopen gate."""

    if publication_attempt_number == 1:
        if expected_content_digest is not None:
            raise V2EvaluationError("initial presentation has retry authority")
        return None
    if expected_content_digest is None:
        raise V2EvaluationError("presentation retry lacks authority digest")
    paths = V2.publication_attempt_runtime_paths(
        publication_attempt_number, publication_attempt_id
    )
    authority = V2.validate_presentation_retry_authority(
        _load_attempt_json_bootstrap(
            attempt_root, paths["retry_authority"]
        ),
        expected_attempt=publication_attempt_number,
    )
    if authority["content_digest"] != expected_content_digest:
        raise V2EvaluationError("presentation retry-authority digest drift")
    return authority


def _build_terminal_published_payloads(
    *,
    attempt_root: Path,
    publication_attempt_number: int,
    publication_attempt_id: str,
    presentation_retry_authority: Mapping[str, Any] | None,
    immutable_payload: Mapping[str, Any],
    independent_validation_receipt: Mapping[str, Any],
    phase_1_custody: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any],
    phase_3_custody: Mapping[str, Any],
) -> dict[str, bytes]:
    paths = V2.publication_attempt_runtime_paths(
        publication_attempt_number, publication_attempt_id
    )
    deterministic = V2.build_presentation_payloads(
        immutable_payload=immutable_payload,
        independent_validation_receipt=independent_validation_receipt,
    )
    runtime_pairs = (
        (str(V2.TRACKED_RESULT_PATH), paths["result"]),
        (str(V2.TRACKED_REPORT_PATH), paths["report"]),
        (str(V2.TRACKED_EVIDENCE_PATH), paths["evidence"]),
    )
    for tracked_path, runtime_path in runtime_pairs:
        if _read_attempt_bytes(attempt_root / runtime_path) != deterministic[
            tracked_path
        ]:
            raise V2EvaluationError(
                f"Phase-3 runtime presentation differs: {tracked_path}"
            )
    result = json.loads(deterministic[str(V2.TRACKED_RESULT_PATH)])
    evidence = json.loads(deterministic[str(V2.TRACKED_EVIDENCE_PATH)])
    report = deterministic[str(V2.TRACKED_REPORT_PATH)].decode("utf-8")
    manifest = V2.validate_presentation_manifest(
        _load_canonical_json(attempt_root / paths["manifest"]),
        presentation_result=result,
        presentation_report_markdown=report,
        presentation_evidence=evidence,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=presentation_retry_authority,
    )
    if phase_3_custody["produced_artifact_binding"] != _artifact_binding(
        attempt_root / paths["manifest"],
        attempt_root=attempt_root,
        rows=3,
    ) or manifest["rows"] != 3:
        raise V2EvaluationError("Phase-3 custody does not bind its manifest")
    receipt = V2.build_publication_receipt(
        immutable_payload=immutable_payload,
        independent_validation_receipt=independent_validation_receipt,
        presentation_result=result,
        presentation_report_markdown=report,
        presentation_evidence=evidence,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
        publication_attempt_number=publication_attempt_number,
        presentation_retry_authority=presentation_retry_authority,
    )
    published = {
        **deterministic,
        str(V2.TRACKED_PUBLICATION_RECEIPT_PATH): (
            V2.canonical_json_bytes(receipt) + b"\n"
        ),
    }
    return V2.validate_published_presentation_payloads(
        published,
        immutable_payload=immutable_payload,
        independent_validation_receipt=independent_validation_receipt,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
    )


def _preserve_failed_presentation_attempt(
    *,
    attempt_root: Path,
    observer_process_identity: Mapping[str, Any],
    immutable_payload: Mapping[str, Any],
    independent_validation_receipt: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any],
    phase_3_exit_observation: Mapping[str, Any],
    phase_3_custody: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    if reopened is None:
        raise V2EvaluationError("presentation failure lacks reopened root")
    number = phase_3_exit_observation["publication_attempt_number"]
    attempt_id = phase_3_exit_observation["publication_attempt_id"]
    inventory = V2.observe_presentation_attempt_namespace_inventory(
        attempt_root_fd_custody=historical,
        reopened_attempt_root_fd_custody=reopened,
        root_fd=root_fd,
        observer_process_identity=observer_process_identity,
        publication_attempt_number=number,
        publication_attempt_id=attempt_id,
        inventory_stage="FAILED_BEFORE_FAILURE_CUSTODY_WRITE",
    )
    failure = V2.build_presentation_attempt_failure_custody(
        immutable_payload=immutable_payload,
        independent_validation_receipt=independent_validation_receipt,
        phase_2_custody=phase_2_custody,
        failed_phase_3_exit_observation=phase_3_exit_observation,
        failed_phase_3_custody=phase_3_custody,
        pre_failure_namespace_inventory=inventory,
        failure_reason_id=(
            "PHASE_3_START_OR_CLEANUP_STATE_UNKNOWN"
            if phase_3_custody is None
            else "PHASE_3_PROCESS_FAILURE"
        ),
    )
    paths = V2.publication_attempt_runtime_paths(number, attempt_id)
    _write_bytes_exclusive(
        attempt_root / paths["failure_custody"],
        V2.presentation_attempt_failure_custody_bytes(failure),
    )
    preservation = V2.preserve_failed_presentation_attempt_namespace(
        failure_custody=failure,
        attempt_root_fd_custody=historical,
        reopened_attempt_root_fd_custody=reopened,
        root_fd=root_fd,
        observer_process_identity=observer_process_identity,
    )
    V2.write_presentation_failure_preservation_receipt_exclusive_fsync(
        preservation_receipt=preservation,
        attempt_root_fd_custody=historical,
        reopened_attempt_root_fd_custody=reopened,
        root_fd=root_fd,
    )
    return preservation


def _canonical_mapping_binding(
    *, relative_path: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    payload = V2.canonical_json_bytes(value) + b"\n"
    return V2.build_artifact_binding(
        path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes_count=len(payload),
        content_digest=value["content_digest"],
        rows=None,
    )


def _postcommit_commit_success_from_retry(
    tracked_retry_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    retry = V2.validate_tracked_publication_retry_authority(
        tracked_retry_authority
    )
    if retry["retry_scope"] != "POSTCOMMIT_OBSERVATION_ONLY":
        raise V2EvaluationError("postcommit resume lacks observation-only authority")
    failure = V2.validate_tracked_publication_failure_custody(
        retry["prior_failure_custodies"][-1]
    )
    if failure["failure_stage"] == "POSTCOMMIT_HEAD_OBSERVATION":
        commit_success = failure["git_commit_success_custody"]
        commit_success_binding = failure[
            "git_commit_success_custody_binding"
        ]
    elif failure["failure_stage"] == "POSTCOMMIT_VALIDATION":
        observation = failure["postcommit_head_observation"]
        commit_success = observation["git_commit_success_custody"]
        commit_success_binding = observation[
            "git_commit_success_custody_binding"
        ]
    else:
        raise V2EvaluationError("postcommit retry failure-stage drift")
    return (
        V2.validate_git_commit_success_custody(commit_success),
        V2.validate_artifact_binding(
            commit_success_binding,
            expected_path=commit_success_binding["path"],
            expected_rows=None,
        ),
    )


def _observe_and_validate_postcommit(
    *,
    attempt_root: Path,
    candidate: Mapping[str, Any],
    git_commit_success_custody: Mapping[str, Any],
    git_commit_success_custody_binding: Mapping[str, Any],
    published_payloads: Mapping[str, bytes],
    immutable_payload: Mapping[str, Any],
    independent_validation_receipt: Mapping[str, Any],
    phase_1_custody: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any],
    phase_3_custody: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    if reopened is None:
        raise V2EvaluationError("postcommit observation lacks reopened root")
    observation, _head_stdout, _head_stderr = (
        V2.observe_postcommit_head_after_successful_commit(
            candidate_custody=candidate,
            git_commit_success_custody=git_commit_success_custody,
            git_commit_success_custody_binding=(
                git_commit_success_custody_binding
            ),
            repo_root=V2.REPO_ROOT,
        )
    )
    if observation.get("schema") == (
        V2.TRACKED_PUBLICATION_FAILURE_CUSTODY_SCHEMA
    ):
        V2.write_tracked_publication_failure_custody_exclusive_fsync(
            failure_custody=observation,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
        )
        return V2.validate_tracked_publication_failure_custody(observation)

    observation_binding = (
        V2.write_postcommit_head_observation_exclusive_fsync(
            observation=observation,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
        )
    )
    commit_success = V2.validate_git_commit_success_custody(
        git_commit_success_custody
    )
    staging = commit_success["staging_receipt"]
    git_index_receipt = commit_success["git_index_receipt"]
    original_paths = V2.tracked_publication_attempt_runtime_paths(
        commit_success["tracked_attempt_number"],
        commit_success["tracked_attempt_id"],
    )
    staging_binding = _canonical_mapping_binding(
        relative_path=original_paths["staging_receipt"], value=staging
    )
    try:
        return V2.validate_presentation_result_commit(
            repo_root=V2.REPO_ROOT,
            source_freeze_commit=immutable_payload["source_freeze_commit"],
            published_payloads=published_payloads,
            immutable_payload=immutable_payload,
            independent_validation_receipt=independent_validation_receipt,
            phase_1_custody=phase_1_custody,
            phase_2_custody=phase_2_custody,
            phase_3_custody=phase_3_custody,
            tracked_publication_candidate_custody=candidate,
            tracked_publication_staging_receipt=staging,
            tracked_publication_git_index_receipt=git_index_receipt,
            tracked_publication_candidate_custody_binding=candidate_binding,
            tracked_publication_staging_receipt_binding=staging_binding,
            tracked_publication_git_index_receipt_binding=commit_success[
                "git_index_receipt_binding"
            ],
            postcommit_head_observation=observation,
            postcommit_head_observation_binding=observation_binding,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
        )
    except BaseException as exc:
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        failure = V2.build_postcommit_validation_failure_custody(
            candidate_custody=candidate,
            postcommit_head_observation=observation,
            postcommit_head_observation_binding=observation_binding,
            structured_exception={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback_text,
                "traceback_sha256": hashlib.sha256(
                    traceback_text.encode("utf-8")
                ).hexdigest(),
            },
        )
        V2.write_tracked_publication_failure_custody_exclusive_fsync(
            failure_custody=failure,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
        )
        return V2.validate_tracked_publication_failure_custody(failure)


def _commit_tracked_publication(
    *,
    attempt_root: Path,
    candidate: Mapping[str, Any],
    staging: Mapping[str, Any],
    git_index_receipt: Mapping[str, Any],
    git_index_receipt_binding: Mapping[str, Any],
    published_payloads: Mapping[str, bytes],
    immutable_payload: Mapping[str, Any],
    independent_validation_receipt: Mapping[str, Any],
    phase_1_custody: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any],
    phase_3_custody: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    staging_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    if reopened is None:
        raise V2EvaluationError("Git publication lacks reopened root")
    git_stdout = b""
    git_stderr = b""
    returncode: int | None = None
    spawn_error = None
    structured_exception = None
    try:
        completed = subprocess.run(
            V2.expected_tracked_publication_git_commit_argv(),
            cwd=V2.REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        git_stdout = completed.stdout
        git_stderr = completed.stderr
        returncode = completed.returncode
    except OSError as exc:
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        spawn_error = {"type": type(exc).__name__, "message": str(exc)}
        structured_exception = {
            **spawn_error,
            "traceback": traceback_text,
            "traceback_sha256": hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest(),
        }
    if returncode != 0:
        failure = V2.build_git_commit_failure_custody(
            staging_receipt=staging,
            candidate_custody=candidate,
            git_index_receipt=git_index_receipt,
            git_index_receipt_binding=git_index_receipt_binding,
            git_stdout=git_stdout,
            git_stderr=git_stderr,
            returncode=returncode,
            spawn_error=spawn_error,
            structured_exception=structured_exception,
            repo_root=V2.REPO_ROOT,
        )
        V2.write_tracked_publication_failure_custody_exclusive_fsync(
            failure_custody=failure,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
            git_stdout=git_stdout,
            git_stderr=git_stderr,
        )
        return failure
    commit_success = V2.build_git_commit_success_custody(
        staging_receipt=staging,
        candidate_custody=candidate,
        git_index_receipt=git_index_receipt,
        git_index_receipt_binding=git_index_receipt_binding,
        git_stdout=git_stdout,
        git_stderr=git_stderr,
        returncode=0,
    )
    commit_success_binding = (
        V2.write_git_commit_success_custody_exclusive_fsync(
            commit_success_custody=commit_success,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
        )
    )
    return _observe_and_validate_postcommit(
        attempt_root=attempt_root,
        candidate=candidate,
        git_commit_success_custody=commit_success,
        git_commit_success_custody_binding=commit_success_binding,
        published_payloads=published_payloads,
        immutable_payload=immutable_payload,
        independent_validation_receipt=independent_validation_receipt,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
        candidate_binding=candidate_binding,
    )


def _publish_terminal_tracked_result(
    *,
    attempt_root: Path,
    phase_1_custody: Mapping[str, Any],
    phase_2_custody: Mapping[str, Any],
    phase_3_custody: Mapping[str, Any],
    publication_attempt_number: int,
    publication_attempt_id: str,
    presentation_retry_authority: Mapping[str, Any] | None,
    tracked_attempt_number: int,
    tracked_attempt_id: str,
    tracked_retry_authority: Mapping[str, Any] | None,
    supervisor_resume_preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    if reopened is None:
        raise V2EvaluationError("tracked publication lacks reopened root")
    identity = _process_identity()
    V2.observe_presentation_attempt_namespace_inventory(
        attempt_root_fd_custody=historical,
        reopened_attempt_root_fd_custody=reopened,
        root_fd=root_fd,
        observer_process_identity=identity,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        inventory_stage="SUCCESSFUL_PHASE3_COMPLETE",
    )
    payload = V2.validate_immutable_scientific_payload(
        _load_canonical_json(
            V2.runtime_path("immutable_scientific_payload", attempt_root)
        )
    )
    validation = V2.validate_independent_payload_validation_receipt(
        _load_canonical_json(
            V2.runtime_path("independent_validation_receipt", attempt_root)
        ),
        immutable_payload=payload,
    )
    published = _build_terminal_published_payloads(
        attempt_root=attempt_root,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=presentation_retry_authority,
        immutable_payload=payload,
        independent_validation_receipt=validation,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
    )
    candidate_resume_preflight = (
        supervisor_resume_preflight
        if reopened["resume_boundary"] in (
            "AFTER_SUCCESSFUL_PHASE_3",
            "GIT_ONLY_FROM_PRESERVED_STAGING",
            "POSTCOMMIT_OBSERVATION_ONLY",
        )
        else None
    )
    candidate = V2.build_tracked_publication_candidate_custody(
        published_payloads=published,
        immutable_payload=payload,
        independent_validation_receipt=validation,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
        tracked_attempt_number=tracked_attempt_number,
        tracked_attempt_id=tracked_attempt_id,
        tracked_publication_retry_authority=tracked_retry_authority,
        terminal_publication_supervisor_resume_preflight=(
            candidate_resume_preflight
        ),
    )
    tracked_paths = V2.tracked_publication_attempt_runtime_paths(
        tracked_attempt_number, tracked_attempt_id
    )
    retry_binding = (
        None
        if tracked_retry_authority is None
        else _artifact_binding(
            attempt_root / tracked_paths["retry_authority"],
            attempt_root=attempt_root,
            rows=None,
        )
    )
    candidate_binding = (
        V2.write_tracked_publication_candidate_custody_exclusive_fsync(
            candidate_custody=candidate,
            published_payloads=published,
            immutable_payload=payload,
            independent_validation_receipt=validation,
            phase_1_custody=phase_1_custody,
            phase_2_custody=phase_2_custody,
            phase_3_custody=phase_3_custody,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
            tracked_publication_retry_authority_binding=retry_binding,
        )
    )
    if reopened["resume_boundary"] == "POSTCOMMIT_OBSERVATION_ONLY":
        if tracked_retry_authority is None:
            raise V2EvaluationError("postcommit resume lacks retry authority")
        commit_success, commit_success_binding = (
            _postcommit_commit_success_from_retry(tracked_retry_authority)
        )
        return _observe_and_validate_postcommit(
            attempt_root=attempt_root,
            candidate=candidate,
            git_commit_success_custody=commit_success,
            git_commit_success_custody_binding=commit_success_binding,
            published_payloads=published,
            immutable_payload=payload,
            independent_validation_receipt=validation,
            phase_1_custody=phase_1_custody,
            phase_2_custody=phase_2_custody,
            phase_3_custody=phase_3_custody,
            candidate_binding=candidate_binding,
        )
    staging, staging_binding = (
        V2.stage_tracked_publication_payloads_transactional(
            candidate_custody=candidate,
            published_payloads=published,
            immutable_payload=payload,
            independent_validation_receipt=validation,
            phase_1_custody=phase_1_custody,
            phase_2_custody=phase_2_custody,
            phase_3_custody=phase_3_custody,
            candidate_custody_binding=candidate_binding,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=reopened,
            root_fd=root_fd,
            repo_root=V2.REPO_ROOT,
        )
    )
    if staging.get("schema") == V2.TRACKED_PUBLICATION_FAILURE_CUSTODY_SCHEMA:
        return V2.validate_tracked_publication_failure_custody(staging)
    if staging.get("schema") == (
        V2.TRACKED_PUBLICATION_REUSED_STAGING_RECEIPT_SCHEMA
    ):
        git_index_receipt = staging["prior_git_index_receipt"]
        git_index_binding = staging["prior_git_index_receipt_binding"]
    else:
        git_index_receipt, git_index_binding = (
            V2.stage_tracked_publication_git_index_transactional(
                staging_receipt=staging,
                candidate_custody=candidate,
                staging_receipt_binding=staging_binding,
                attempt_root_fd_custody=historical,
                reopened_attempt_root_fd_custody=reopened,
                root_fd=root_fd,
                repo_root=V2.REPO_ROOT,
            )
        )
        if git_index_receipt.get("schema") == (
            V2.TRACKED_PUBLICATION_FAILURE_CUSTODY_SCHEMA
        ):
            return V2.validate_tracked_publication_failure_custody(
                git_index_receipt
            )
    return _commit_tracked_publication(
        attempt_root=attempt_root,
        candidate=candidate,
        staging=staging,
        git_index_receipt=git_index_receipt,
        git_index_receipt_binding=git_index_binding,
        published_payloads=published,
        immutable_payload=payload,
        independent_validation_receipt=validation,
        phase_1_custody=phase_1_custody,
        phase_2_custody=phase_2_custody,
        phase_3_custody=phase_3_custody,
        candidate_binding=candidate_binding,
        staging_binding=staging_binding,
    )


def supervise_v2_terminal_publication(
    *,
    attempt_root: Path,
    resume_boundary: str,
    publication_attempt_number: int,
    publication_attempt_id: str,
    tracked_attempt_number: int,
    tracked_attempt_id: str,
    phase_1_custody_content_digest: str,
    phase_2_custody_content_digest: str | None,
    phase_3_custody_content_digest: str | None,
    presentation_retry_authority_content_digest: str | None,
    tracked_retry_authority_content_digest: str | None,
) -> dict[str, Any]:
    root = attempt_root.absolute()
    identity = _process_identity()
    expected_argv = V2.expected_terminal_publication_supervisor_argv(
        root,
        resume_boundary=resume_boundary,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        tracked_attempt_number=tracked_attempt_number,
        tracked_attempt_id=tracked_attempt_id,
        phase_1_custody_content_digest=phase_1_custody_content_digest,
        phase_2_custody_content_digest=phase_2_custody_content_digest,
        phase_3_custody_content_digest=phase_3_custody_content_digest,
        presentation_retry_authority_content_digest=(
            presentation_retry_authority_content_digest
        ),
        tracked_retry_authority_content_digest=(
            tracked_retry_authority_content_digest
        ),
    )
    if identity["argv"] != expected_argv or identity["cwd"] != str(V2.REPO_ROOT):
        raise V2EvaluationError("terminal publication supervisor identity drift")
    phase_1 = _load_terminal_phase_custody(
        attempt_root=root,
        phase_index=0,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        expected_content_digest=phase_1_custody_content_digest,
    )
    historical = phase_1["preexecution_receipt"]["attempt_root_fd_custody"]
    preloaded_presentation_retry = (
        _load_presentation_retry_authority_before_reopen(
            attempt_root=root,
            publication_attempt_number=publication_attempt_number,
            publication_attempt_id=publication_attempt_id,
            expected_content_digest=(
                presentation_retry_authority_content_digest
            ),
        )
    )
    if resume_boundary == "AFTER_PHASE_1":
        V2.require_terminal_publication_supervisor_failure_custody_absent(
            historical["attempt_id"]
        )
    root_reopen_started = time.monotonic_ns()
    parent_fd = -1
    root_fd = -1
    try:
        parent_fd, root_fd, reopened = (
            V2.reopen_attempt_root_for_terminal_publication_supervisor(
                attempt_root_fd_custody=historical,
                supervisor_process_identity=identity,
                resume_boundary=resume_boundary,
                publication_attempt_number=publication_attempt_number,
                publication_attempt_id=publication_attempt_id,
                tracked_attempt_number=tracked_attempt_number,
                tracked_attempt_id=tracked_attempt_id,
                phase_1_custody_content_digest=(
                    phase_1_custody_content_digest
                ),
                phase_2_custody_content_digest=(
                    phase_2_custody_content_digest
                ),
                phase_3_custody_content_digest=(
                    phase_3_custody_content_digest
                ),
                presentation_retry_authority=(
                    preloaded_presentation_retry
                ),
                presentation_retry_authority_content_digest=(
                    presentation_retry_authority_content_digest
                ),
                tracked_retry_authority_content_digest=(
                    tracked_retry_authority_content_digest
                ),
            )
        )
        _activate_attempt_io(
            attempt_root=root,
            root_fd=root_fd,
            root_fd_custody=historical,
            parent_fd=parent_fd,
            reopened_root_fd_custody=reopened,
        )
    except BaseException as error:
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if resume_boundary == "AFTER_PHASE_1":
            return _persist_pre_phase2_supervisor_failure(
                phase_1_custody=phase_1,
                supervisor_process_identity=identity,
                publication_attempt_id=publication_attempt_id,
                tracked_attempt_id=tracked_attempt_id,
                failure_stage="ROOT_REOPEN",
                error=error,
                failure_started_monotonic_ns=root_reopen_started,
            )
        raise
    initial_preflight_started = time.monotonic_ns()
    try:
        publication = V2.validate_phase1_terminal_publication_receipt(
            _load_canonical_json(
                V2.runtime_path(
                    "phase_1_terminal_publication_receipt", root
                )
            ),
            phase1_custody=phase_1,
        )
        presentation_retry, tracked_retry = _load_terminal_retry_authorities(
            attempt_root=root,
            publication_attempt_number=publication_attempt_number,
            publication_attempt_id=publication_attempt_id,
            presentation_retry_content_digest=(
                presentation_retry_authority_content_digest
            ),
            tracked_attempt_number=tracked_attempt_number,
            tracked_attempt_id=tracked_attempt_id,
            tracked_retry_content_digest=(
                tracked_retry_authority_content_digest
            ),
            preloaded_presentation_retry_authority=(
                preloaded_presentation_retry
            ),
        )
    except BaseException as error:
        if resume_boundary == "AFTER_PHASE_1":
            return _persist_pre_phase2_supervisor_failure(
                phase_1_custody=phase_1,
                supervisor_process_identity=identity,
                publication_attempt_id=publication_attempt_id,
                tracked_attempt_id=tracked_attempt_id,
                failure_stage="INITIAL_PREFLIGHT",
                error=error,
                failure_started_monotonic_ns=initial_preflight_started,
                terminal_publication_supervisor_root_reopen_custody=(
                    reopened
                ),
            )
        raise

    phase_2: dict[str, Any] | None = None
    phase_3: dict[str, Any] | None = None
    resume_preflight = None
    preflight = None
    if resume_boundary == "AFTER_PHASE_1":
        try:
            preflight = V2.observe_terminal_publication_supervisor_preflight(
                producer_phase_exit_custody=phase_1,
                phase1_terminal_publication_receipt=publication,
                terminal_publication_supervisor_root_reopen_custody=(
                    reopened
                ),
                root_fd=root_fd,
            )
            V2.write_terminal_publication_supervisor_preflight_exclusive_fsync(
                supervisor_preflight=preflight,
                producer_phase_exit_custody=phase_1,
                terminal_publication_supervisor_root_reopen_custody=(
                    reopened
                ),
                root_fd=root_fd,
            )
        except BaseException as error:
            return _persist_pre_phase2_supervisor_failure(
                phase_1_custody=phase_1,
                supervisor_process_identity=identity,
                publication_attempt_id=publication_attempt_id,
                tracked_attempt_id=tracked_attempt_id,
                failure_stage="INITIAL_PREFLIGHT",
                error=error,
                failure_started_monotonic_ns=initial_preflight_started,
                terminal_publication_supervisor_root_reopen_custody=(
                    reopened
                ),
            )
        phase_2_exit, phase_2 = _run_terminal_publication_child(
            phase_id=V2.PHASE_IDS[1],
            attempt_root=root,
            phase_1_custody=phase_1,
            phase_2_custody=None,
            supervisor_root_reopen_custody=reopened,
            supervisor_preflight=preflight,
            supervisor_resume_preflight=None,
            presentation_retry_authority=None,
        )
        if phase_2 is None:
            return phase_2_exit
        phase_2_disposition = (
            V2.validate_terminal_scientific_attempt_disposition(
                phase_2["scientific_attempt_disposition"]
            )
        )
        if (
            phase_2["pass"] is not True
            or phase_2_disposition["disposition"]
            != "SCIENTIFIC_ATTEMPT_COMPLETE"
        ):
            return phase_2
    else:
        if phase_2_custody_content_digest is None:
            raise V2EvaluationError("terminal resume lacks Phase-2 digest")
        phase_2 = _load_terminal_phase_custody(
            attempt_root=root,
            phase_index=1,
            publication_attempt_number=publication_attempt_number,
            publication_attempt_id=publication_attempt_id,
            expected_content_digest=phase_2_custody_content_digest,
        )
        if resume_boundary in (
            "AFTER_SUCCESSFUL_PHASE_3",
            "GIT_ONLY_FROM_PRESERVED_STAGING",
            "POSTCOMMIT_OBSERVATION_ONLY",
        ):
            if phase_3_custody_content_digest is None:
                raise V2EvaluationError("terminal resume lacks Phase-3 digest")
            phase_3 = _load_terminal_phase_custody(
                attempt_root=root,
                phase_index=2,
                publication_attempt_number=publication_attempt_number,
                publication_attempt_id=publication_attempt_id,
                expected_content_digest=phase_3_custody_content_digest,
            )
        resume_preflight = (
            V2.observe_terminal_publication_supervisor_resume_preflight(
                terminal_publication_supervisor_root_reopen_custody=reopened,
                root_fd=root_fd,
                phase_1_custody=phase_1,
                phase_2_custody=phase_2,
                phase_3_custody=phase_3,
                presentation_retry_authority=presentation_retry,
                tracked_publication_retry_authority=tracked_retry,
            )
        )
        V2.write_terminal_publication_supervisor_resume_preflight_exclusive_fsync(
            resume_preflight=resume_preflight,
            phase_1_custody=phase_1,
            root_fd=root_fd,
        )

    if resume_boundary in ("AFTER_PHASE_1", "AFTER_PHASE_2_VALIDATION"):
        if phase_2 is None:
            raise V2EvaluationError("Phase-3 launch lacks Phase-2 custody")
        phase_3_exit, phase_3 = _run_terminal_publication_child(
            phase_id=V2.PHASE_IDS[2],
            attempt_root=root,
            phase_1_custody=phase_1,
            phase_2_custody=phase_2,
            supervisor_root_reopen_custody=reopened,
            supervisor_preflight=(
                preflight
                if resume_boundary == "AFTER_PHASE_1"
                else None
            ),
            supervisor_resume_preflight=resume_preflight,
            presentation_retry_authority=presentation_retry,
        )
        if phase_3 is None and phase_3_exit.get("schema") == (
            V2.RAW_PHASE3_FAILURE_PRESERVATION_RECEIPT_SCHEMA
        ):
            return V2.validate_raw_phase3_failure_preservation_receipt(
                phase_3_exit
            )
        if phase_3 is None or phase_3["pass"] is not True:
            payload = V2.validate_immutable_scientific_payload(
                _load_canonical_json(
                    V2.runtime_path("immutable_scientific_payload", root)
                )
            )
            validation = V2.validate_independent_payload_validation_receipt(
                _load_canonical_json(
                    V2.runtime_path(
                        "independent_validation_receipt", root
                    )
                ),
                immutable_payload=payload,
            )
            return _preserve_failed_presentation_attempt(
                attempt_root=root,
                observer_process_identity=identity,
                immutable_payload=payload,
                independent_validation_receipt=validation,
                phase_2_custody=phase_2,
                phase_3_exit_observation=phase_3_exit,
                phase_3_custody=phase_3,
            )
    if phase_2 is None or phase_3 is None:
        raise V2EvaluationError("terminal publication lacks complete custody chain")
    V2.validate_phase_custody_sequence(phase_1, phase_2, phase_3)
    return _publish_terminal_tracked_result(
        attempt_root=root,
        phase_1_custody=phase_1,
        phase_2_custody=phase_2,
        phase_3_custody=phase_3,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=presentation_retry,
        tracked_attempt_number=tracked_attempt_number,
        tracked_attempt_id=tracked_attempt_id,
        tracked_retry_authority=tracked_retry,
        supervisor_resume_preflight=resume_preflight,
    )


def validate_immutable_payload(attempt_root: Path) -> dict[str, Any]:
    identity = _process_identity()
    expected_argv = V2.expected_internal_argv(
        V2.PHASE2_VALIDATOR_MODE, attempt_root
    )
    if identity["argv"] != expected_argv or identity["cwd"] != str(V2.REPO_ROOT):
        raise V2EvaluationError("Phase-2 process identity drift")
    handoff_receipt = V2.consume_terminal_publication_child_handoff(
        environment=os.environ,
        child_process_identity=identity,
    )
    handoff = handoff_receipt["payload"]
    if handoff["phase_id"] != V2.PHASE_IDS[1]:
        raise V2EvaluationError("Phase-2 supervisor handoff phase drift")
    reopened = _activate_reopened_attempt_io(
        attempt_root=attempt_root,
        phase_id=V2.PHASE_IDS[1],
        phase_process_identity=identity,
    )
    root_fd, historical, live_reopened = _require_attempt_io(attempt_root)
    handoff_binding = (
        V2.write_terminal_publication_child_handoff_receipt_exclusive_fsync(
            handoff_receipt=handoff_receipt,
            attempt_root_fd_custody=historical,
            reopened_attempt_root_fd_custody=live_reopened,
            root_fd=root_fd,
        )
    )
    phase_1_custody = V2.validate_phase_custody(
        handoff["phase_1_custody"]
    )
    phase_1_publication = V2.validate_phase1_terminal_publication_receipt(
        _load_canonical_json(
            V2.runtime_path(
                "phase_1_terminal_publication_receipt", attempt_root
            )
        ),
        phase1_custody=phase_1_custody,
    )
    prior_absence = V2.observe_phase2_prior_process_absence_custody(
        producer_phase_exit_custody=phase_1_custody,
        phase1_terminal_publication_receipt=phase_1_publication,
        terminal_publication_child_handoff_receipt=handoff_receipt,
        terminal_publication_child_handoff_binding=handoff_binding,
        validator_process_identity=identity,
    )
    validator_started_monotonic_ns = _monotonic_ns_after(
        prior_absence["preflight_completed_monotonic_ns"]
    )
    _write_last_stage(attempt_root, V2.PHASE_IDS[1], "STARTED")
    pre_snapshot = _observe_static_snapshot(
        stage_id="PHASE2_REPLAY",
        snapshot_kind="PRE_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    audit_installed = _begin_static_audit(
        stage_id="PHASE2_REPLAY", process_identity=identity
    )
    stage_started = _monotonic_ns_after(pre_snapshot["ended_monotonic_ns"])
    payload_path = V2.runtime_path("immutable_scientific_payload", attempt_root)
    original = V2.validate_immutable_scientific_payload(
        _load_canonical_json(payload_path)
    )
    payload_binding = _artifact_binding(
        payload_path, attempt_root=attempt_root, rows=None
    )
    if phase_1_custody["produced_artifact_binding"] != payload_binding:
        raise V2EvaluationError("Phase-1 custody does not bind immutable payload")
    if live_reopened != reopened or _ACTIVE_ATTEMPT_PARENT_FD is None:
        raise V2EvaluationError("Phase-2 reopened root custody drift")
    artifact_rehash = V2.observe_phase2_artifact_rehash_receipt(
        immutable_payload=original,
        validator_process_identity=identity,
        reopened_attempt_root_fd_custody=reopened,
        parent_fd=_ACTIVE_ATTEMPT_PARENT_FD,
        root_fd=root_fd,
    )
    regenerated, counters, replay_values = _regenerate_scientific_payload(
        attempt_root=attempt_root, original=original
    )
    stage_ended = time.monotonic_ns()
    audit_stream = _finish_static_audit(
        stage_id="PHASE2_REPLAY",
        process_identity=identity,
        attempt_root=attempt_root,
        installed_monotonic_ns=audit_installed,
    )
    post_snapshot = _observe_static_snapshot(
        stage_id="PHASE2_REPLAY",
        snapshot_kind="POST_STAGE",
        process_identity=identity,
        attempt_root=attempt_root,
    )
    static_custody = _build_static_stage_custody(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        static_input_audit_stream=audit_stream,
        started_monotonic_ns=stage_started,
        ended_monotonic_ns=stage_ended,
    )
    _write_json_exclusive(
        V2.runtime_path("static_input_phase2_replay_custody", attempt_root),
        static_custody,
    )
    _write_last_stage(
        attempt_root, V2.PHASE_IDS[1], "COMPONENTS_REGENERATED"
    )
    if V2.immutable_scientific_payload_bytes(regenerated) != (
        V2.immutable_scientific_payload_bytes(original)
    ):
        raise V2EvaluationError("independent payload bytes differ")
    _write_last_stage(attempt_root, V2.PHASE_IDS[1], "PAYLOAD_BYTE_EQUAL")
    replay_components = V2.build_phase2_replay_components(
        immutable_payload=original,
        validator_process_identity=identity,
        static_input_stage_custody=static_custody,
        artifact_rehash_receipt=artifact_rehash,
        **replay_values,
    )
    regeneration_evidence = V2.build_phase2_regeneration_evidence(
        immutable_payload=original,
        independently_regenerated_payload=regenerated,
        validator_process_identity=identity,
        phase2_replay_components=replay_components,
    )
    receipt = V2.build_independent_payload_validation_receipt(
        immutable_payload=original,
        independently_regenerated_payload=regenerated,
        producer_phase_exit_custody=phase_1_custody,
        phase2_prior_process_absence_custody=prior_absence,
        producer_process_identity=phase_1_custody["process_identity"],
        validator_process_identity=identity,
        validator_started_monotonic_ns=validator_started_monotonic_ns,
        regenerated_artifact_bindings=regenerated["artifact_bindings"],
        regeneration_counters=counters,
        phase2_regeneration_evidence=regeneration_evidence,
    )
    V2.validate_independent_payload_validation_receipt(
        receipt, immutable_payload=original
    )
    receipt_path = V2.runtime_path(
        "independent_validation_receipt", attempt_root
    )
    _write_json_exclusive(receipt_path, receipt)
    _write_last_stage(attempt_root, V2.PHASE_IDS[1], "SCIENCE_COMPLETE")
    _write_last_stage(attempt_root, V2.PHASE_IDS[1], "COMPLETE")
    return receipt


def publish_validated_presentation(
    attempt_root: Path,
    publication_attempt_number: int,
    publication_attempt_id: str,
    presentation_retry_authority_content_digest: str | None,
) -> dict[str, Any]:
    paths = V2.publication_attempt_runtime_paths(
        publication_attempt_number, publication_attempt_id
    )
    identity = _process_identity()
    expected_argv = V2.expected_internal_argv(
        V2.PHASE3_PUBLISHER_MODE,
        attempt_root,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority_content_digest=(
            presentation_retry_authority_content_digest
        ),
    )
    if identity["argv"] != expected_argv or identity["cwd"] != str(V2.REPO_ROOT):
        raise V2EvaluationError("Phase-3 process identity drift")
    handoff_receipt = V2.consume_terminal_publication_child_handoff(
        environment=os.environ,
        child_process_identity=identity,
    )
    handoff = handoff_receipt["payload"]
    if handoff["phase_id"] != V2.PHASE_IDS[2]:
        raise V2EvaluationError("Phase-3 supervisor handoff phase drift")
    retry_authority = handoff["presentation_retry_authority"]
    if publication_attempt_number > 1:
        retry_authority = V2.validate_presentation_retry_authority(
            retry_authority,
            expected_attempt=publication_attempt_number,
        )
        if retry_authority["content_digest"] != (
            presentation_retry_authority_content_digest
        ):
            raise V2EvaluationError("Phase-3 retry-authority digest drift")
    elif (
        presentation_retry_authority_content_digest is not None
        or retry_authority is not None
    ):
        raise V2EvaluationError("initial Phase 3 has retry authority")
    _activate_reopened_attempt_io(
        attempt_root=attempt_root,
        phase_id=V2.PHASE_IDS[2],
        phase_process_identity=identity,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=retry_authority,
    )
    root_fd, historical, reopened = _require_attempt_io(attempt_root)
    V2.write_terminal_publication_child_handoff_receipt_exclusive_fsync(
        handoff_receipt=handoff_receipt,
        attempt_root_fd_custody=historical,
        reopened_attempt_root_fd_custody=reopened,
        root_fd=root_fd,
    )
    phase_1_custody = V2.validate_phase_custody(
        handoff["phase_1_custody"]
    )
    phase_2_custody = V2.validate_phase_custody(
        handoff["phase_2_custody"]
    )
    if (
        phase_2_custody["prior_phase_custody_binding"]
        != {
            "phase_id": phase_1_custody["phase_id"],
            "content_digest": phase_1_custody["content_digest"],
        }
        or phase_1_custody["ended_monotonic_ns"]
        >= phase_2_custody["started_monotonic_ns"]
        or phase_2_custody["ended_monotonic_ns"] >= time.monotonic_ns()
    ):
        raise V2EvaluationError("Phase-3 prior custody sequence drift")
    payload = V2.validate_immutable_scientific_payload(
        _load_canonical_json(
            V2.runtime_path("immutable_scientific_payload", attempt_root)
        )
    )
    validation = V2.validate_independent_payload_validation_receipt(
        _load_canonical_json(
            V2.runtime_path("independent_validation_receipt", attempt_root)
        ),
        immutable_payload=payload,
    )
    validation_binding = _artifact_binding(
        V2.runtime_path("independent_validation_receipt", attempt_root),
        attempt_root=attempt_root,
        rows=None,
    )
    if phase_2_custody["produced_artifact_binding"] != validation_binding:
        raise V2EvaluationError(
            "Phase-2 custody does not bind validation receipt"
        )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[2],
        "STARTED",
        publication_runtime_paths=paths,
    )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[2],
        "VALIDATION_RECEIPT_LOADED",
        publication_runtime_paths=paths,
    )
    result = V2.build_presentation_result(
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    evidence = V2.build_presentation_evidence(
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    report = V2.build_presentation_report_markdown(
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    V2.validate_presentation_result(
        result,
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    V2.validate_presentation_evidence(
        evidence,
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    V2.validate_presentation_report_markdown(
        report,
        immutable_payload=payload,
        independent_validation_receipt=validation,
    )
    _write_json_exclusive(
        attempt_root / paths["result"], result
    )
    _write_bytes_exclusive(
        attempt_root / paths["report"],
        report.encode("utf-8"),
    )
    _write_json_exclusive(
        attempt_root / paths["evidence"], evidence
    )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[2],
        "PRESENTATION_DERIVED",
        publication_runtime_paths=paths,
    )
    manifest = V2.build_presentation_manifest(
        presentation_result=result,
        presentation_report_markdown=report,
        presentation_evidence=evidence,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=retry_authority,
    )
    runtime_manifest = V2.build_runtime_presentation_manifest(
        immutable_payload=payload,
        independent_validation_receipt=validation,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=retry_authority,
    )
    if manifest != runtime_manifest:
        raise V2EvaluationError("runtime presentation manifest differs")
    V2.validate_presentation_manifest(
        manifest,
        presentation_result=result,
        presentation_report_markdown=report,
        presentation_evidence=evidence,
        publication_attempt_number=publication_attempt_number,
        publication_attempt_id=publication_attempt_id,
        presentation_retry_authority=retry_authority,
    )
    _write_bytes_exclusive(
        attempt_root / paths["manifest"],
        V2.presentation_manifest_bytes(manifest),
    )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[2],
        "PRESENTATION_MANIFEST_FROZEN",
        publication_runtime_paths=paths,
    )
    _write_last_stage(
        attempt_root,
        V2.PHASE_IDS[2],
        "COMPLETE",
        publication_runtime_paths=paths,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    previous_umask = os.umask(0o077)
    try:
        args = build_parser().parse_args(argv)
        try:
            if args.mode == V2.PHASE1_SUPERVISOR_MODE:
                value = supervise_phase_1_execution(args.attempt_id)
            elif args.mode == V2.PRE_ROOT_PHASE1_FAILURE_CUSTODY_MODE:
                value = capture_pre_root_phase_1_failure_custody(
                    args.custody_path,
                    args.supervisor_observation_content_digest,
                )
            elif args.mode == V2.PUBLIC_EXECUTE_MODE:
                value = execute_phase_1()
            elif args.mode == V2.PHASE1_CONTEXT_STATE_WORKER_MODE:
                root = Path(args.attempt_root).absolute()
                V2.expected_phase1_worker_argv(
                    args.mode, root, state_index=args.state_index
                )
                value = capture_proprio_context_state(root, args.state_index)
            elif args.mode == V2.PHASE1_PREDICTION_SOURCE_WORKER_MODE:
                root = Path(args.attempt_root).absolute()
                V2.expected_phase1_worker_argv(
                    args.mode,
                    root,
                    source_id=args.source_id,
                    ablation=args.ablation,
                )
                value = predict_frozen_source(
                    root, args.source_id, args.ablation
                )
            elif args.mode == V2.TERMINAL_PUBLICATION_SUPERVISOR_MODE:
                value = supervise_v2_terminal_publication(
                    attempt_root=Path(args.attempt_root).absolute(),
                    resume_boundary=args.resume_boundary,
                    publication_attempt_number=(
                        args.publication_attempt_number
                    ),
                    publication_attempt_id=args.publication_attempt_id,
                    tracked_attempt_number=args.tracked_attempt_number,
                    tracked_attempt_id=args.tracked_attempt_id,
                    phase_1_custody_content_digest=(
                        args.phase_1_custody_content_digest
                    ),
                    phase_2_custody_content_digest=(
                        args.phase_2_custody_content_digest
                    ),
                    phase_3_custody_content_digest=(
                        args.phase_3_custody_content_digest
                    ),
                    presentation_retry_authority_content_digest=(
                        args.presentation_retry_authority_content_digest
                    ),
                    tracked_retry_authority_content_digest=(
                        args.tracked_retry_authority_content_digest
                    ),
                )
            else:
                root = _validated_attempt_root(args.mode, args.attempt_root)
                if args.mode == V2.PHASE1_TERMINAL_CUSTODY_MODE:
                    value = capture_phase_1_terminal_custody(
                        root,
                        args.supervisor_observation_content_digest,
                    )
                elif args.mode == V2.PHASE3_PUBLISHER_MODE:
                    value = publish_validated_presentation(
                        root,
                        args.publication_attempt_number,
                        args.publication_attempt_id,
                        args.presentation_retry_authority_content_digest,
                    )
                else:
                    dispatch = {
                        V2.PHASE1_PROPRIO_HELPER_MODE: (
                            materialize_proprio_substitution
                        ),
                        V2.PHASE1_STAGE_C_HELPER_MODE: (
                            materialize_stage_c_attribution
                        ),
                        V2.PHASE2_VALIDATOR_MODE: validate_immutable_payload,
                    }
                    value = dispatch[args.mode](root)
        except BaseException as exc:
            if args.mode == V2.PUBLIC_EXECUTE_MODE and _ACTIVE_ATTEMPT_ROOT:
                trace = "".join(traceback.format_exception(exc)).encode(
                    "utf-8"
                )
                try:
                    _persist_phase1_terminal_source_streams(
                        error=exc, traceback_bytes=trace
                    )
                except BaseException as persistence_error:
                    raise persistence_error from exc
            raise
        sys.stdout.buffer.write(V2.canonical_json_bytes(value) + b"\n")
        sys.stdout.buffer.flush()
        return 0
    finally:
        _close_attempt_io()
        os.umask(previous_umask)


if __name__ == "__main__":
    try:
        _exit_code = main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise SystemExit(1) from None
    raise SystemExit(_exit_code)
