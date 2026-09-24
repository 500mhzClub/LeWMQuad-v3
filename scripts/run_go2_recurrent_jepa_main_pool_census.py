#!/usr/bin/env python3
"""Run the train/validation recurrent-JEPA temporal census once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (  # noqa: E402
    SCHEMA,
    _open_absolute_directory,
    run_census,
)


OUTPUT_DIRECTORY = (
    ROOT / ".generated" / "go2_recurrent_jepa_main_pool_census_v2"
)
OUTPUT_PATH = OUTPUT_DIRECTORY / "receipt.json"
OUTPUT_DIRECTORY_NAME = OUTPUT_DIRECTORY.name
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _write_exclusive(directory_fd: int, payload: dict[str, Any]) -> None:
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open("receipt.json", flags, 0o644, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _reserve_output() -> int:
    generated_fd = _open_absolute_directory(ROOT / ".generated")
    try:
        os.mkdir(OUTPUT_DIRECTORY_NAME, 0o755, dir_fd=generated_fd)
        directory_fd = os.open(
            OUTPUT_DIRECTORY_NAME,
            _DIR_FLAGS,
            dir_fd=generated_fd,
        )
        os.fsync(generated_fd)
        return directory_fd
    finally:
        os.close(generated_fd)


def _failure_receipt(error: BaseException) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "decision": "STOP_MAIN_POOL_H4_CENSUS_EXECUTION_FAILURE",
        "failure_class": type(error).__name__,
        "access": {
            "counts_complete": False,
            "completed_source_count": "unknown",
            "note": "The interrupted/failed execution did not return aggregate access counters.",
        },
        "authority": "Execution failure grants no scientific or runtime authority.",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel metadata parsers (default: min(8, CPU count))",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory_fd = _reserve_output()
    last_reported = 0

    def progress(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed == total or completed - last_reported >= 50:
            print(f"scanned {completed}/{total} metadata scenes", flush=True)
            last_reported = completed

    try:
        receipt = run_census(ROOT, workers=args.workers, progress=progress)
    except BaseException as error:
        _write_exclusive(output_directory_fd, _failure_receipt(error))
        os.close(output_directory_fd)
        print(
            json.dumps(
                {
                    "decision": "STOP_MAIN_POOL_H4_CENSUS_EXECUTION_FAILURE",
                    "failure_class": type(error).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 3
    _write_exclusive(output_directory_fd, receipt)
    os.close(output_directory_fd)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "source_count": receipt["totals"]["source_count"],
                "row_count": receipt["totals"]["row_count"],
                "sliding_h6": receipt["totals"]["sliding_h6"],
                "packed_h6": receipt["totals"]["packed_h6"],
                "failed_predicate_count": len(receipt["failed_predicates"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if receipt["decision"] == "MAIN_POOL_H4_METADATA_FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
