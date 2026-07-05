#!/usr/bin/env python3
"""Audit rendered Phase 2D counterfactual roots before spatial dataset joins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_readiness import canonical_split_name  # noqa: E402
from lewm.benchmarks.phase2d_render_readiness import (  # noqa: E402
    audit_phase2d_render_readiness,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("argument must use SPLIT=PATH")
    return canonical_split_name(name), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", action="append", type=_named_path, default=[])
    parser.add_argument("--render-root", action="append", type=_named_path, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.plan_root:
        parser.error("at least one --plan-root SPLIT=PATH argument is required")
    if not args.render_root:
        parser.error("at least one --render-root SPLIT=PATH argument is required")

    report = audit_phase2d_render_readiness(
        plan_roots=dict(args.plan_root),
        render_roots=dict(args.render_root),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        write_json(args.output, report)
    return 0 if report["ready_for_spatial_future_join"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
