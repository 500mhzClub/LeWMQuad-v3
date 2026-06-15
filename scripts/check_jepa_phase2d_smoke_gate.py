#!/usr/bin/env python3
"""Check whether a Phase 2D smoke report permits full training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_gate import (  # noqa: E402
    phase2d_smoke_gate_report_from_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gate = phase2d_smoke_gate_report_from_path(args.report)
    if args.output is not None:
        write_json(args.output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
