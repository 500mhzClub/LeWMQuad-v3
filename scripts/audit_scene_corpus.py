#!/usr/bin/env python3
"""Audit every scene in a built corpus against route-teacher reachability.

Run before launching mass data generation to catch fragmented scenes that
would silently waste rollout cycles. Exits non-zero if any scene fails.

Usage::

    python3 scripts/audit_scene_corpus.py [CORPUS_DIR]

``CORPUS_DIR`` defaults to ``.generated/scene_corpus/acceptance``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a checkout without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_worlds"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_genesis"))

from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_validation import audit_scene_reachability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_dir",
        nargs="?",
        default=".generated/scene_corpus/acceptance",
        help="Corpus root containing corpus.json and <split>/<family>/<scene_id>/manifest.json",
    )
    parser.add_argument(
        "--cell-size-m", type=float, default=0.05, help="Planning-grid cell size"
    )
    parser.add_argument(
        "--inflation-m", type=float, default=0.20, help="Body-radius inflation"
    )
    parser.add_argument(
        "--standoff-m", type=float, default=0.85, help="Standoff distance from beacon"
    )
    parser.add_argument(
        "--spawn-clearance-floor-m",
        type=float,
        default=0.20,
        help="Minimum wall clearance for spawn candidate cells",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        print(f"corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str, str, str]] = []
    total = 0
    for manifest_path in sorted(corpus_dir.rglob("manifest.json")):
        total += 1
        manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
        report = audit_scene_reachability(
            manifest,
            cell_size_m=args.cell_size_m,
            inflation_m=args.inflation_m,
            standoff_m=args.standoff_m,
            spawn_clearance_floor_m=args.spawn_clearance_floor_m,
        )
        rel = manifest_path.parent.relative_to(corpus_dir)
        split = rel.parts[0] if rel.parts else "?"
        if not report.is_valid:
            failures.append((split, manifest.family, manifest.scene_id, report.failure_reason))
            print(
                f"[FAIL] {split:9s} {manifest.family:25s} {manifest.scene_id:50s} "
                f"{report.failure_reason}"
            )
        else:
            print(
                f"[ OK ] {split:9s} {manifest.family:25s} {manifest.scene_id:50s} "
                f"canon={report.spawn_candidates_in_canonical}"
            )

    print()
    print(f"audited {total} scenes; {len(failures)} failure(s)")
    if failures:
        print()
        print(
            "Rebuild the plan with plan_corpus(..., validate=True) (or "
            "smoke_corpus_plan(validate=True) / standard_corpus_plan(validate=True))"
            " to reroll fragmented seeds before scenes hit disk."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
