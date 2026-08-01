#!/usr/bin/env python3
"""Externally supervise the exact one-shot V3 action localization.

This source grants no authority.  It validates one separately committed
authority, reserves the fresh output root, launches the CPU-only worker and
receipt-only checker once under a hard wall ceiling, and closes the attempt
with an immutable terminal receipt.  Any reservation consumes the attempt.
"""
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


def _require_explicit_empty_gpu_visibility_before_worker_import() -> None:
    names = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    if any(name not in os.environ or os.environ[name] != "" for name in names):
        raise SystemExit(
            "supervisor requires all GPU visibility variables explicitly empty"
        )


if __name__ == "__main__":
    _require_explicit_empty_gpu_visibility_before_worker_import()

from scripts import (  # noqa: E402
    extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as worker,
)


TERMINAL_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "supervision_terminal_v1"
)
SUCCESS_STATUS = "PASS_COMPLETE_READ_ONLY_LOCALIZATION"
FAILURE_STATUS = "TERMINAL_LOCALIZATION_FAILURE"


class LocalizationSupervisionError(RuntimeError):
    """The exact authorized supervision contract failed closed."""


class LocalizationReservationError(LocalizationSupervisionError):
    """Reservation failed, possibly after this invocation created the root."""

    def __init__(self, message: str, *, attempt_root_created: bool) -> None:
        super().__init__(message)
        self.attempt_root_created = attempt_root_created


def _run_once(
    argv: Sequence[str], *, timeout: float, environment: Mapping[str, str]
) -> dict[str, Any]:
    if timeout <= 0.0:
        raise LocalizationSupervisionError("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
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
        raise LocalizationSupervisionError(
            "supervised command exceeded the hard wall ceiling"
        ) from error
    elapsed = time.monotonic() - started
    receipt = {
        "argv": list(argv),
        "elapsed_seconds": elapsed,
        "exit_code": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_byte_count": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_byte_count": len(stderr),
    }
    if process.returncode != 0:
        raise LocalizationSupervisionError(
            f"supervised command exited with status {process.returncode}: {receipt}"
        )
    return receipt


def _reserve(
    authority: Mapping[str, Any], authority_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_root = worker.ATTEMPT_ROOT.parent
    if (
        campaign_root.exists()
        or campaign_root.is_symlink()
        or worker.ATTEMPT_ROOT.exists()
        or worker.ATTEMPT_ROOT.is_symlink()
    ):
        raise LocalizationSupervisionError(
            "one-shot localization namespace is already occupied"
        )
    development_root = campaign_root.parent
    if development_root.is_symlink() or not development_root.is_dir():
        raise LocalizationSupervisionError("development output root changed")
    root_created = False
    temporary_campaign: Path | None = None
    try:
        temporary_campaign = Path(
            tempfile.mkdtemp(
                prefix=".world_model_action_localization_reservation_",
                dir=development_root,
            )
        )
        temporary_attempt = temporary_campaign / worker.ATTEMPT_ROOT.name
        temporary_attempt.mkdir()
        nonce = secrets.token_hex(32)
        reservation = worker.expected_reservation(
            authority, authority_binding, supervisor_nonce=nonce
        )
        temporary_binding = worker.write_immutable_json(
            temporary_attempt / "reservation.json", reservation
        )
        binding = {
            **temporary_binding,
            "path": str(worker.ATTEMPT_ROOT / "reservation.json"),
        }
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
            raise LocalizationSupervisionError(
                "atomically materialized reservation binding changed"
            )
        worker.exact_root_inventory({"reservation.json"})
        return reservation, binding
    except BaseException as error:
        if temporary_campaign is not None:
            try:
                if (
                    temporary_campaign.parent == development_root
                    and temporary_campaign.name.startswith(
                        ".world_model_action_localization_reservation_"
                    )
                ):
                    shutil.rmtree(temporary_campaign)
            except BaseException:
                pass
        raise LocalizationReservationError(
            "reservation creation failed", attempt_root_created=root_created
        ) from error


def _instantiate_worker_command(
    reservation: Mapping[str, Any], reservation_binding: Mapping[str, Any]
) -> list[str]:
    result = []
    for item in reservation["worker_command_template"]:
        if item == "<SUPERVISOR_BOUND_RESERVATION_SHA256>":
            result.append(reservation_binding["file_sha256"])
        elif item == "<SUPERVISOR_BOUND_RESERVATION_BYTE_COUNT>":
            result.append(str(reservation_binding["byte_count"]))
        else:
            result.append(item)
    return result


def _instantiate_checker_command(
    reservation: Mapping[str, Any], result_binding: Mapping[str, Any]
) -> list[str]:
    result = []
    for item in reservation["checker_command_template"]:
        if item == "<WORKER_RESULT_SHA256>":
            result.append(result_binding["file_sha256"])
        elif item == "<WORKER_RESULT_BYTE_COUNT>":
            result.append(str(result_binding["byte_count"]))
        else:
            result.append(item)
    return result


def _terminal_document(
    *,
    status: str,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any] | None,
    result_binding: Mapping[str, Any] | None,
    checker_binding: Mapping[str, Any] | None,
    worker_process: Mapping[str, Any] | None,
    checker_process: Mapping[str, Any] | None,
    wall_elapsed_seconds: float,
    inventory: Sequence[str],
    error: BaseException | None,
    error_traceback: str | None,
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "scientific_verdict_emitted": False,
        "attempt_id": worker.ATTEMPT_ID,
        "attempt_consumed": True,
        "authority_binding": dict(authority_binding),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "reservation_binding": (
            dict(reservation_binding) if reservation_binding else None
        ),
        "result_binding": dict(result_binding) if result_binding else None,
        "checker_binding": dict(checker_binding) if checker_binding else None,
        "worker_process": dict(worker_process) if worker_process else None,
        "checker_process": dict(checker_process) if checker_process else None,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "caps": authority["caps"],
        "root_inventory_before_terminal": list(inventory),
        "snapshot_opened_by_supervisor": False,
        "validation_index_opened_by_supervisor": False,
        "pack_or_rgb_opened_by_supervisor": False,
        "network_access_used_by_supervisor": False,
        "retry": False,
        "resume": False,
        "refill": False,
        "overwrite": False,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "traceback": error_traceback,
    }


def supervise(
    authority_path: Path,
    *,
    expected_authority_sha256: str,
    expected_authority_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    wall_started = time.monotonic()
    authority, authority_binding = worker.load_and_validate_authority(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
    )
    if Path(authority["execution"]["supervisor_path"]) != Path(__file__).resolve():
        raise LocalizationSupervisionError("authority selected another supervisor")
    try:
        reservation, reservation_binding = _reserve(authority, authority_binding)
    except LocalizationReservationError as error:
        if not error.attempt_root_created:
            raise
        partial_binding: dict[str, Any] | None = None
        reservation_path = worker.ATTEMPT_ROOT / "reservation.json"
        try:
            if reservation_path.is_file() and not reservation_path.is_symlink():
                partial_binding = worker.file_binding(reservation_path)
        except BaseException:
            partial_binding = None
        partial_inventory = []
        if worker.ATTEMPT_ROOT.is_dir() and not worker.ATTEMPT_ROOT.is_symlink():
            with os.scandir(worker.ATTEMPT_ROOT) as entries:
                partial_inventory = sorted(entry.name for entry in entries)
        terminal = _terminal_document(
            status=FAILURE_STATUS,
            authority=authority,
            authority_binding=authority_binding,
            reservation_binding=partial_binding,
            result_binding=None,
            checker_binding=None,
            worker_process=None,
            checker_process=None,
            wall_elapsed_seconds=time.monotonic() - wall_started,
            inventory=partial_inventory,
            error=error,
            error_traceback=traceback.format_exc(),
        )
        terminal_binding = worker.write_immutable_json(
            worker.ATTEMPT_ROOT / "terminal_supervision.json", terminal
        )
        return terminal, terminal_binding
    result_binding: dict[str, Any] | None = None
    checker_binding: dict[str, Any] | None = None
    worker_process: dict[str, Any] | None = None
    checker_process: dict[str, Any] | None = None
    failure: BaseException | None = None
    failure_traceback: str | None = None
    status = FAILURE_STATUS
    try:
        worker_process = _run_once(
            _instantiate_worker_command(reservation, reservation_binding),
            timeout=(
                worker.MAXIMUM_WALL_SECONDS
                - (time.monotonic() - wall_started)
                - 15.0
            ),
            environment=worker.EXACT_CHILD_ENVIRONMENT,
        )
        worker.exact_root_inventory({"reservation.json", "localization.json"})
        result_binding = worker.file_binding(
            worker.ATTEMPT_ROOT / "localization.json"
        )
        checker_process = _run_once(
            _instantiate_checker_command(reservation, result_binding),
            timeout=(
                worker.MAXIMUM_WALL_SECONDS
                - (time.monotonic() - wall_started)
                - 15.0
            ),
            environment=worker.EXACT_CHILD_ENVIRONMENT,
        )
        worker.exact_root_inventory(
            {"reservation.json", "localization.json", "receipt_check.json"}
        )
        checker_binding = worker.file_binding(
            worker.ATTEMPT_ROOT / "receipt_check.json"
        )
        status = SUCCESS_STATUS
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()

    inventory: list[str] = []
    if worker.ATTEMPT_ROOT.is_dir() and not worker.ATTEMPT_ROOT.is_symlink():
        with os.scandir(worker.ATTEMPT_ROOT) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    failure = failure or LocalizationSupervisionError(
                        "attempt root contains a non-file"
                    )
                    status = FAILURE_STATUS
                inventory.append(entry.name)
    expected_success_inventory = {
        "reservation.json",
        "localization.json",
        "receipt_check.json",
    }
    if status == SUCCESS_STATUS and set(inventory) != expected_success_inventory:
        failure = LocalizationSupervisionError(
            f"success inventory changed before terminal: {sorted(inventory)}"
        )
        failure_traceback = None
        status = FAILURE_STATUS
    observed_wall = time.monotonic() - wall_started
    if status == SUCCESS_STATUS and observed_wall >= worker.MAXIMUM_WALL_SECONDS:
        failure = LocalizationSupervisionError(
            "supervision reached the hard wall cap"
        )
        failure_traceback = None
        status = FAILURE_STATUS
    terminal = _terminal_document(
        status=status,
        authority=authority,
        authority_binding=authority_binding,
        reservation_binding=reservation_binding,
        result_binding=result_binding,
        checker_binding=checker_binding,
        worker_process=worker_process,
        checker_process=checker_process,
        wall_elapsed_seconds=observed_wall,
        inventory=sorted(inventory),
        error=failure,
        error_traceback=failure_traceback,
    )
    terminal_binding = worker.write_immutable_json(
        worker.ATTEMPT_ROOT / "terminal_supervision.json", terminal
    )
    if status == SUCCESS_STATUS:
        worker.exact_root_inventory(
            {
                "reservation.json",
                "localization.json",
                "receipt_check.json",
                "terminal_supervision.json",
            }
        )
    return terminal, terminal_binding


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
    print(
        json.dumps(
            {"status": terminal["status"], "terminal_supervision": binding},
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == SUCCESS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
