#!/usr/bin/env python3
"""Run frozen Phase 2D training manifests with bounded parallelism."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import verify_manifest_files, write_json  # noqa: E402
from lewm.benchmarks.phase2d_gate import phase2d_smoke_gate_report_from_path  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _selected_records(summary: dict, selected: set[str] | None) -> list[tuple[str, dict]]:
    records = summary.get("manifests", {})
    if not isinstance(records, dict):
        raise ValueError("summary must contain a manifests object")
    result = []
    for key in sorted(records):
        if selected is not None and key not in selected:
            continue
        record = records[key]
        if not isinstance(record, dict):
            raise ValueError(f"manifest record must be an object: {key}")
        result.append((key, record))
    if selected is not None:
        missing = sorted(selected - {key for key, _record in result})
        if missing:
            raise ValueError(f"requested manifest keys not found: {missing}")
    return result


def _verify_run_manifest(path: Path) -> dict:
    manifest = _load_json(path)
    verification = verify_manifest_files(manifest)
    checkpoint = manifest.get("config", {}).get("expected_checkpoint_path")
    if not checkpoint:
        raise ValueError(f"manifest lacks config.expected_checkpoint_path: {path}")
    return {
        "manifest": manifest,
        "verification": verification,
        "checkpoint_path": Path(str(checkpoint)),
    }


def _run_one(
    key: str,
    record: dict,
    *,
    log_dir: Path,
    threads_per_job: int,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    manifest_path = Path(str(record["manifest_path"]))
    verified = _verify_run_manifest(manifest_path)
    checkpoint_path = verified["checkpoint_path"]
    log_path = log_dir / f"{key}.log"
    if not verified["verification"]["passes"]:
        return {
            "key": key,
            "manifest_path": str(manifest_path.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "log_path": str(log_path.resolve()),
            "status": "manifest_verification_failed",
            "verification": verified["verification"],
            "return_code": None,
        }
    if checkpoint_path.exists() and not overwrite:
        return {
            "key": key,
            "manifest_path": str(manifest_path.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "log_path": str(log_path.resolve()),
            "status": "skipped_existing_checkpoint",
            "verification": verified["verification"],
            "return_code": 0,
        }

    command = shlex.split(str(verified["manifest"]["run_command"]))
    env = os.environ.copy()
    thread_text = str(int(threads_per_job))
    env.update(
        {
            "OMP_NUM_THREADS": thread_text,
            "OPENBLAS_NUM_THREADS": thread_text,
            "MKL_NUM_THREADS": thread_text,
            "NUMEXPR_NUM_THREADS": thread_text,
            "PYTHONUNBUFFERED": "1",
        }
    )
    base = {
        "key": key,
        "manifest_path": str(manifest_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "log_path": str(log_path.resolve()),
        "command": command,
        "thread_environment": {
            name: env[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "verification": verified["verification"],
    }
    if dry_run:
        return {**base, "status": "dry_run", "return_code": None}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    start_time = time.monotonic()
    with log_path.open("w") as log:
        log.write(
            json.dumps(
                {
                    "event": "start",
                    "key": key,
                    "started_at_utc": started,
                    "command": command,
                    "thread_environment": base["thread_environment"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        ended = _utc_now()
        log.write(
            json.dumps(
                {
                    "event": "end",
                    "key": key,
                    "ended_at_utc": ended,
                    "return_code": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return {
        **base,
        "status": "completed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "elapsed_seconds": time.monotonic() - start_time,
    }


def _run_succeeded(run: dict) -> bool:
    return run.get("status") in {"completed", "skipped_existing_checkpoint", "dry_run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", action="append", help="Run one manifest key.")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads-per-job", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--required-smoke-gate-report",
        type=Path,
        help=(
            "Required for non-dry full launches. Must be a smoke training JSON "
            "whose final validation gate passes."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep launching queued manifests after a failed run.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.threads_per_job < 1:
        parser.error("--threads-per-job must be positive")
    if not args.dry_run and args.required_smoke_gate_report is None:
        parser.error("--required-smoke-gate-report is required unless --dry-run")

    summary = _load_json(args.summary)
    selected = None if not args.only else set(args.only)
    records = _selected_records(summary, selected)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    smoke_gate = (
        phase2d_smoke_gate_report_from_path(args.required_smoke_gate_report)
        if args.required_smoke_gate_report is not None
        else None
    )
    report = {
        "schema": "jepa_phase2d_training_matrix_runner_report_v0",
        "summary_path": str(args.summary.resolve()),
        "started_at_utc": _utc_now(),
        "jobs": args.jobs,
        "threads_per_job": args.threads_per_job,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "continue_on_failure": args.continue_on_failure,
        "required_smoke_gate": smoke_gate,
        "runs": [],
        "not_started": [],
    }
    write_json(args.report, report)
    if smoke_gate is not None and not smoke_gate["passed"]:
        report["ended_at_utc"] = _utc_now()
        report["passed"] = False
        report["aborted_before_launch"] = True
        report["abort_reason"] = "required_smoke_gate_failed"
        write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        next_record = 0
        running: dict[concurrent.futures.Future, str] = {}
        stop_launching = False

        def launch_next() -> None:
            nonlocal next_record
            key, record = records[next_record]
            future = executor.submit(
                _run_one,
                key,
                record,
                log_dir=args.log_dir,
                threads_per_job=args.threads_per_job,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            running[future] = key
            next_record += 1

        while next_record < len(records) and len(running) < args.jobs:
            launch_next()

        while running:
            done, _pending = concurrent.futures.wait(
                running,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                key = running.pop(future)
                try:
                    run = future.result()
                except Exception as error:  # pragma: no cover - defensive runner guard.
                    run = {
                        "key": key,
                        "status": "runner_exception",
                        "return_code": None,
                        "error": repr(error),
                    }
                report["runs"].append(run)
                if not args.continue_on_failure and not _run_succeeded(run):
                    stop_launching = True
                    report["aborted_after_failure"] = True
                    report.setdefault("first_failure_key", key)
                    report.setdefault("first_failure_status", run.get("status"))
                while (
                    not stop_launching
                    and next_record < len(records)
                    and len(running) < args.jobs
                ):
                    launch_next()
            write_json(args.report, report)

    report["ended_at_utc"] = _utc_now()
    report["not_started"] = [
        {
            "key": key,
            "manifest_path": str(Path(str(record["manifest_path"])).resolve()),
            "checkpoint_path": str(Path(str(record["checkpoint_path"])).resolve()),
            "status": "not_started_after_failure",
        }
        for key, record in records[next_record:]
    ]
    report["passed"] = all(_run_succeeded(run) for run in report["runs"]) and not report[
        "not_started"
    ]
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
