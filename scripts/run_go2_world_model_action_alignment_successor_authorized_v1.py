#!/usr/bin/env python3
"""Supervise the bounded fixed same-mechanism alignment continuation V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_exact_environment_before_worker_import() -> None:
    expected = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONSAFEPATH": "1",
        "OMP_NUM_THREADS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    if dict(os.environ) != expected:
        raise SystemExit("supervisor requires the exact allowlisted GPU environment")


if __name__ == "__main__":
    _require_exact_environment_before_worker_import()

from scripts import (  # noqa: E402
    execute_go2_world_model_action_alignment_successor_v1 as worker,
)
from scripts import (  # noqa: E402
    run_go2_world_model_existing_pool_three_arm_authorized_v1 as device_guard,
)


TERMINAL_SCHEMA = f"{worker.SCHEMA_PREFIX}_supervision_terminal_v1"
SUCCESS_STATUS = (
    "PASS_COMPLETE_ACTION_ALIGNMENT_FIXED_SAME_MECHANISM_CONTINUATION_V1"
)
FAILURE_STATUS = (
    "TERMINAL_ACTION_ALIGNMENT_FIXED_SAME_MECHANISM_CONTINUATION_V1_FAILURE"
)


class AlignmentSupervisionError(RuntimeError):
    """The external one-shot supervision contract failed closed."""


class ReservationError(AlignmentSupervisionError):
    def __init__(self, message: str, *, root_created: bool) -> None:
        super().__init__(message)
        self.root_created = root_created


def _run_once(
    argv: Sequence[str], *, timeout: float, environment: Mapping[str, str]
) -> dict[str, Any]:
    if timeout <= 0.0:
        raise AlignmentSupervisionError("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv), cwd=REPO_ROOT, env=dict(environment),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise AlignmentSupervisionError("supervised command exceeded wall cap") from error
    receipt = {
        "argv": list(argv),
        "elapsed_seconds": time.monotonic() - started,
        "exit_code": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_byte_count": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_byte_count": len(stderr),
        "stdout_tail": stdout.decode("utf-8", errors="replace")[-4000:],
        "stderr_tail": stderr.decode("utf-8", errors="replace")[-4000:],
    }
    if process.returncode != 0:
        raise AlignmentSupervisionError(
            f"supervised command exited {process.returncode}: {receipt}"
        )
    return receipt


def _reserve(
    authority: Mapping[str, Any], authority_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_root = worker.ATTEMPT_ROOT.parent
    if campaign_root.exists() or campaign_root.is_symlink():
        raise AlignmentSupervisionError("one-shot successor namespace is occupied")
    development_root = campaign_root.parent
    if development_root.is_symlink() or not development_root.is_dir():
        raise AlignmentSupervisionError("development root changed")
    temporary_campaign: Path | None = None
    root_created = False
    try:
        temporary_campaign = Path(
            tempfile.mkdtemp(prefix=".action_alignment_reservation_", dir=development_root)
        )
        temporary_attempt = temporary_campaign / worker.ATTEMPT_ROOT.name
        temporary_attempt.mkdir()
        reservation = worker.expected_reservation(
            authority, authority_binding, supervisor_nonce=secrets.token_hex(32)
        )
        temporary_binding = worker.write_immutable_json(
            temporary_attempt / "reservation.json", reservation
        )
        binding = {**temporary_binding, "path": str(worker.ATTEMPT_ROOT / "reservation.json")}
        descriptor = os.open(temporary_attempt, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary_campaign, campaign_root)
        temporary_campaign = None
        root_created = True
        descriptor = os.open(development_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if worker.file_binding(worker.ATTEMPT_ROOT / "reservation.json") != binding:
            raise AlignmentSupervisionError("materialized reservation changed")
        worker.exact_root_inventory({"reservation.json"})
        return reservation, binding
    except BaseException as error:
        if temporary_campaign is not None:
            try:
                if temporary_campaign.parent == development_root and temporary_campaign.name.startswith(".action_alignment_reservation_"):
                    shutil.rmtree(temporary_campaign)
            except BaseException:
                pass
        raise ReservationError("reservation creation failed", root_created=root_created) from error


def _instantiate(
    template: Sequence[str], replacements: Mapping[str, str]
) -> list[str]:
    return [replacements.get(item, item) for item in template]


def _inventory() -> list[str]:
    if not worker.ATTEMPT_ROOT.is_dir() or worker.ATTEMPT_ROOT.is_symlink():
        return []
    observed = []
    with os.scandir(worker.ATTEMPT_ROOT) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise AlignmentSupervisionError("attempt root contains a non-file")
            observed.append(entry.name)
    return sorted(observed)


def _terminal(
    *, status: str, authority: Mapping[str, Any], authority_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any] | None,
    result_binding: Mapping[str, Any] | None,
    checker_binding: Mapping[str, Any] | None,
    worker_process: Mapping[str, Any] | None,
    checker_process: Mapping[str, Any] | None,
    wall_elapsed_seconds: float, inventory: Sequence[str],
    error: BaseException | None, error_traceback: str | None,
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "attempt_id": worker.ATTEMPT_ID,
        "attempt_consumed": True,
        "development_verdict_emitted": status == SUCCESS_STATUS,
        "citable_as_original_factual_learnability_claim": False,
        "citable_as_planning_usefulness_evidence": False,
        "authority_binding": dict(authority_binding),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "reservation_binding": dict(reservation_binding) if reservation_binding else None,
        "result_binding": dict(result_binding) if result_binding else None,
        "checker_binding": dict(checker_binding) if checker_binding else None,
        "worker_process": dict(worker_process) if worker_process else None,
        "checker_process": dict(checker_process) if checker_process else None,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "caps": authority["caps"],
        "root_inventory_before_terminal": list(inventory),
        "bound_non_u700_input_identity_hash_reads_performed_by_supervisor": True,
        "bound_predecessor_evidence_identity_hash_reads_performed_by_supervisor": True,
        "source_test_and_runtime_identity_hash_reads_performed_by_supervisor": True,
        "u700_continuation_snapshot_content_read_by_supervisor": False,
        "runtime_tensor_or_array_deserialized_by_supervisor": False,
        "gpu_work_performed_by_supervisor": False,
        "network_access_used_by_supervisor": False,
        "retry": False, "resume": False, "refill": False, "overwrite": False,
        "recovery": False, "integrity_replacement": False,
        "further_continuation": False,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "traceback": error_traceback,
    }


def supervise(
    authority_path: Path, *, expected_authority_sha256: str,
    expected_authority_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    wall_started = time.monotonic()
    authority, authority_binding = worker.load_and_validate_authority(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
    )
    if Path(authority["execution"]["supervisor_path"]) != Path(__file__).resolve():
        raise AlignmentSupervisionError("authority selected another supervisor")
    device_guard._require_idle_authorized_device()
    reservation_binding: dict[str, Any] | None = None
    try:
        reservation, reservation_binding = _reserve(authority, authority_binding)
    except ReservationError as error:
        if not error.root_created:
            raise
        terminal = _terminal(
            status=FAILURE_STATUS, authority=authority, authority_binding=authority_binding,
            reservation_binding=(worker.file_binding(worker.ATTEMPT_ROOT / "reservation.json") if (worker.ATTEMPT_ROOT / "reservation.json").is_file() else None),
            result_binding=None, checker_binding=None, worker_process=None,
            checker_process=None, wall_elapsed_seconds=time.monotonic() - wall_started,
            inventory=_inventory(), error=error, error_traceback=traceback.format_exc(),
        )
        binding = worker.write_immutable_json(worker.ATTEMPT_ROOT / "terminal_supervision.json", terminal)
        return terminal, binding

    result_binding: dict[str, Any] | None = None
    checker_binding: dict[str, Any] | None = None
    worker_process: dict[str, Any] | None = None
    checker_process: dict[str, Any] | None = None
    failure: BaseException | None = None
    failure_traceback: str | None = None
    status = FAILURE_STATUS
    print(json.dumps({"status": "RESERVED_AND_STARTING", "attempt": worker.ATTEMPT_ID}), flush=True)
    try:
        worker_command = _instantiate(
            reservation["worker_command_template"],
            {
                "<SUPERVISOR_BOUND_RESERVATION_SHA256>": reservation_binding["file_sha256"],
                "<SUPERVISOR_BOUND_RESERVATION_BYTE_COUNT>": str(reservation_binding["byte_count"]),
            },
        )
        worker_process = _run_once(
            worker_command,
            timeout=worker.MAXIMUM_WALL_SECONDS - (time.monotonic() - wall_started) - 15.0,
            environment=worker.EXACT_CHILD_ENVIRONMENT,
        )
        worker.exact_root_inventory(worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER)
        result_binding = worker.file_binding(worker.ATTEMPT_ROOT / "result.json")
        checker_command = _instantiate(
            reservation["checker_command_template"],
            {
                "<WORKER_RESULT_SHA256>": result_binding["file_sha256"],
                "<WORKER_RESULT_BYTE_COUNT>": str(result_binding["byte_count"]),
            },
        )
        checker_process = _run_once(
            checker_command,
            timeout=worker.MAXIMUM_WALL_SECONDS - (time.monotonic() - wall_started) - 15.0,
            environment=worker.EXACT_CHILD_ENVIRONMENT,
        )
        worker.exact_root_inventory(
            worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER | {"receipt_check.json"}
        )
        checker_binding = worker.file_binding(worker.ATTEMPT_ROOT / "receipt_check.json")
        status = SUCCESS_STATUS
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()
    inventory = _inventory()
    if status == SUCCESS_STATUS and set(inventory) != worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER | {"receipt_check.json"}:
        failure = AlignmentSupervisionError("success inventory changed")
        failure_traceback = None
        status = FAILURE_STATUS
    elapsed = time.monotonic() - wall_started
    if status == SUCCESS_STATUS and elapsed >= worker.MAXIMUM_WALL_SECONDS:
        failure = AlignmentSupervisionError("terminal wall cap reached")
        failure_traceback = None
        status = FAILURE_STATUS
    terminal = _terminal(
        status=status, authority=authority, authority_binding=authority_binding,
        reservation_binding=reservation_binding, result_binding=result_binding,
        checker_binding=checker_binding, worker_process=worker_process,
        checker_process=checker_process, wall_elapsed_seconds=elapsed,
        inventory=inventory, error=failure, error_traceback=failure_traceback,
    )
    binding = worker.write_immutable_json(worker.ATTEMPT_ROOT / "terminal_supervision.json", terminal)
    if status == SUCCESS_STATUS:
        worker.exact_root_inventory(
            worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER
            | {"receipt_check.json", "terminal_supervision.json"}
        )
    return terminal, binding


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    terminal, binding = supervise(
        arguments.authority,
        expected_authority_sha256=arguments.expected_authority_sha256,
        expected_authority_byte_count=arguments.expected_authority_byte_count,
    )
    print(json.dumps({"status": terminal["status"], "terminal_supervision": binding}, sort_keys=True))
    return 0 if terminal["status"] == SUCCESS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
