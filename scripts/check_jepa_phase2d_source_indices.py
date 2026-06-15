#!/usr/bin/env python3
"""Audit scored task-aligned source indices before Phase 2D generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_source_indices import (  # noqa: E402
    audit_phase2d_source_indices,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("source argument must use SPLIT=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_named_path, default=[])
    parser.add_argument(
        "--require-local-target-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.source:
        parser.error("at least one --source SPLIT=PATH argument is required")

    report = audit_phase2d_source_indices(
        dict(args.source),
        require_local_target_frame=args.require_local_target_frame,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        write_json(args.output, report)
    return 0 if report["ready_for_counterfactual_generation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
