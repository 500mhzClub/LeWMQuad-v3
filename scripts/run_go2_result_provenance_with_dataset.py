#!/usr/bin/env python3
"""Replay a saved Go2 benchmark argv with small provenance-preserving edits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, default=None)
    parser.add_argument("--dataset-states", default="EXPLORE,SEEK,SERVO")
    parser.add_argument("--post-claim-policy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--set-arg",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help="Set or replace an arbitrary benchmark option in the replayed argv.",
    )
    parser.add_argument(
        "--add-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help="Add a boolean benchmark flag to the replayed argv if it is not already present.",
    )
    parser.add_argument("--child-log", type=Path, default=None)
    parser.add_argument(
        "--remove-runtime-contract",
        action="store_true",
        help="Remove contract flags so dataset-capture-only argv is not rejected.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.template_result.read_text(encoding="utf-8"))
    argv = [str(item) for item in payload.get("provenance", {}).get("argv", [])]
    if not argv:
        raise SystemExit(f"{args.template_result} has no provenance.argv")

    if args.remove_runtime_contract:
        _remove_flag(argv, "--generalized-runtime-contract")
        _remove_flag(argv, "--fully-learned-runtime-contract")
    _remove_option(argv, "--learned-local-dataset-output")
    _remove_option(argv, "--learned-local-dataset-states")
    _set_value(argv, "--output", str(args.output))
    if args.post_claim_policy_checkpoint is not None:
        _set_value(
            argv,
            "--learned-local-post-claim-policy-checkpoint",
            str(args.post_claim_policy_checkpoint),
        )
    if args.dataset_output is not None:
        _set_value(argv, "--learned-local-dataset-output", str(args.dataset_output))
        _set_value(argv, "--learned-local-dataset-states", str(args.dataset_states))
    for item in args.set_arg:
        if "=" not in str(item):
            raise SystemExit(f"--set-arg must be FLAG=VALUE: {item}")
        flag, value = str(item).split("=", 1)
        if not str(flag).startswith("--"):
            raise SystemExit(f"--set-arg flag must start with --: {flag}")
        _set_value(argv, str(flag), str(value))
    for flag in args.add_flag:
        if not str(flag).startswith("--"):
            raise SystemExit(f"--add-flag flag must start with --: {flag}")
        _add_flag(argv, str(flag))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset_output is not None:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, *argv]
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return 0
    if args.child_log is not None:
        args.child_log.parent.mkdir(parents=True, exist_ok=True)
        with args.child_log.open("w", encoding="utf-8") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    return subprocess.run(cmd, check=False).returncode


def _remove_flag(argv: list[str], flag: str) -> None:
    argv[:] = [item for item in argv if item != flag]


def _remove_option(argv: list[str], flag: str) -> None:
    while flag in argv:
        idx = argv.index(flag)
        del argv[idx]
        if idx < len(argv):
            del argv[idx]


def _set_value(argv: list[str], flag: str, value: str) -> None:
    _remove_option(argv, flag)
    argv.extend([flag, value])


def _add_flag(argv: list[str], flag: str) -> None:
    if flag not in argv:
        argv.append(flag)


if __name__ == "__main__":
    raise SystemExit(main())
