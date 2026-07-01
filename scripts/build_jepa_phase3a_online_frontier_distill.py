#!/usr/bin/env python3
"""Build Phase 3A rows with online frontier-marker policy targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_explore_claim import (  # noqa: E402
    egocentric_explore_claim_score,
)
from lewm.benchmarks.phase3a_positive_control import read_jsonl, write_jsonl  # noqa: E402


def _rewrite_rows(rows: list[dict]) -> list[dict]:
    rewritten = []
    for row in rows:
        item = dict(row)
        labels = dict(item["consequence_labels"])
        score = float(egocentric_explore_claim_score(item))
        labels["target_original_utility"] = float(labels["target_utility"])
        labels["target_online_frontier_marker_utility"] = score
        labels["target_utility"] = score
        labels["utility_mode"] = "online_frontier_marker_distill"
        item["consequence_labels"] = labels
        item["utility_mode"] = "online_frontier_marker_distill"
        rewritten.append(item)
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_source = args.input_dir / "train_phase3a_positive_control.jsonl"
    validation_source = args.input_dir / "validation_phase3a_positive_control.jsonl"
    train_rows = _rewrite_rows(read_jsonl(train_source))
    validation_rows = _rewrite_rows(read_jsonl(validation_source))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_phase3a_positive_control.jsonl"
    validation_path = args.output_dir / "validation_phase3a_positive_control.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "schema": "jepa_phase3a_online_frontier_distill_manifest_v0",
        "input_dir": str(args.input_dir.resolve()),
        "train_data": str(train_path.resolve()),
        "validation_data": str(validation_path.resolve()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
    }
    manifest_path = args.output_dir / "phase3a_online_frontier_distill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
