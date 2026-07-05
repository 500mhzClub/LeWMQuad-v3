#!/usr/bin/env python3
"""Create or verify an immutable Phase 2D split lineage manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2d_readiness import (  # noqa: E402
    build_phase2d_split_manifest,
    verify_phase2d_split_manifest,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("split must use NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", type=_named_path, default=[])
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify is not None:
        report = verify_phase2d_split_manifest(json.loads(args.verify.read_text()))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passes"] else 1

    if args.output is None or not args.split:
        parser.error("--output and at least one --split are required")
    if len(dict(args.split)) != len(args.split):
        parser.error("split names must be unique")
    manifest = build_phase2d_split_manifest(
        dict(args.split),
        hash_images=args.hash_images,
    )
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["confirmatory_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
