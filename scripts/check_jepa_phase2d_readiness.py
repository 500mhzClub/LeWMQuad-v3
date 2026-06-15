#!/usr/bin/env python3
"""Guard Phase 2D validation/test access behind frozen verified manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_readiness import phase2d_run_readiness  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("cell manifest must use CELL=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cell-manifest", action="append", type=_named_path, default=[])
    parser.add_argument(
        "--requested-stage",
        choices=("validation", "test_id", "test_hard"),
        required=True,
    )
    parser.add_argument("--test-id-report-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = phase2d_run_readiness(
        split_manifest_path=args.split_manifest,
        cell_manifest_paths=dict(args.cell_manifest),
        requested_stage=args.requested_stage,
        test_id_report_manifest_path=args.test_id_report_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
