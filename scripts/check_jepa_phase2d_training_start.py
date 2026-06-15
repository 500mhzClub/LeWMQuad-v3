#!/usr/bin/env python3
"""Guard Phase 2D training start behind frozen split-manifest checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_readiness import (  # noqa: E402
    phase2d_training_start_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument(
        "--cell",
        choices=("C0", "C1", "C2", "state_only", "action_only"),
        required=True,
    )
    parser.add_argument(
        "--run-class",
        choices=("pilot", "confirmatory"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = phase2d_training_start_readiness(
        split_manifest_path=args.split_manifest,
        cell=args.cell,
        requested_run_class=args.run_class,
        train_data_path=args.train_data,
        validation_data_path=args.validation_data,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
