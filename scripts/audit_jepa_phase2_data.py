#!/usr/bin/env python3
"""Audit Phase 2 spatial-future data, controls, split integrity, and provenance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import artifact_record, write_json  # noqa: E402
from lewm.benchmarks.phase2_data import (  # noqa: E402
    confirmatory_data_gate,
    load_spatial_future_rows,
    pairwise_split_overlap,
    phase2_dataset_audit,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("dataset must use NAME=PATH")
    return name, Path(path)


def build_audit(named_paths: dict[str, Path], *, legacy_batch_size: int) -> dict:
    rows = {}
    datasets = {}
    for name, path in sorted(named_paths.items()):
        loaded, load_audit = load_spatial_future_rows(path, mode="all")
        rows[name] = loaded
        audit = phase2_dataset_audit(loaded, legacy_batch_size=legacy_batch_size)
        audit["file"] = artifact_record(path)
        audit["load_audit"] = load_audit
        datasets[name] = audit
    overlap = pairwise_split_overlap(rows)
    foundation_checks = {
        "all_split_pairs_scene_disjoint": all(
            not value["scene_ids"] for value in overlap.values()
        ),
        "all_split_pairs_source_disjoint": all(
            not value["source_keys"] for value in overlap.values()
        ),
        "hard_negative_constructor_has_zero_identical_negatives": all(
            step["identical_negative_count"] == 0
            for audit in datasets.values()
            for step in audit["hard_negative_by_step"]
        ),
    }
    confirmatory_checks = {
        "all_sources_have_81_candidates": all(
            audit["source_states"] > 0
            and audit["full_81_candidate_sources"] == audit["source_states"]
            for audit in datasets.values()
        ),
        "non_hold_hard_negative_coverage_at_least_70pct": all(
            step["eligible_non_hold_valid_coverage"] >= 0.70
            for audit in datasets.values()
            for step in audit["hard_negative_by_step"][:1]
        ),
        "eligible_first_action_minimum_share_at_least_5pct": all(
            not step["eligible_positive_action_counts"]
            or min(step["eligible_positive_action_counts"].values())
            / sum(step["eligible_positive_action_counts"].values())
            >= 0.05
            for audit in datasets.values()
            for step in audit["hard_negative_by_step"][:1]
        ),
    }
    registered_confirmatory_gate = confirmatory_data_gate(
        rows,
        lineage_verified=False,
    )
    return {
        "schema": "jepa_phase2_data_and_control_audit_v0",
        "datasets": datasets,
        "pairwise_split_overlap": overlap,
        "foundation_checks": foundation_checks,
        "foundation_gate_passed": all(foundation_checks.values()),
        "confirmatory_data_checks": confirmatory_checks,
        "registered_confirmatory_gate": registered_confirmatory_gate,
        "confirmatory_data_gate_passed": registered_confirmatory_gate["passed"],
        "limitations": [
            "minimum-scene and unopened-test gates require separate split manifests",
            "hard-negative eligibility is an audit until the Phase 2D trainer consumes it",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_named_path,
        required=True,
        help="Named spatial future dataset, for example train=path.jsonl",
    )
    parser.add_argument("--legacy-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    named_paths = dict(args.dataset)
    if len(named_paths) != len(args.dataset):
        raise SystemExit("dataset names must be unique")
    report = build_audit(named_paths, legacy_batch_size=args.legacy_batch_size)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output.resolve()),
                "foundation_checks": report["foundation_checks"],
                "foundation_gate_passed": report["foundation_gate_passed"],
                "confirmatory_data_checks": report["confirmatory_data_checks"],
                "confirmatory_data_gate_passed": report[
                    "confirmatory_data_gate_passed"
                ],
                "datasets": {
                    name: {
                        "rows": audit["rows"],
                        "complete_valid_rows": audit["complete_valid_rows"],
                        "rows_with_any_valid_transition": audit[
                            "rows_with_any_valid_transition"
                        ],
                        "source_states": audit["source_states"],
                        "full_81_candidate_sources": audit[
                            "full_81_candidate_sources"
                        ],
                    }
                    for name, audit in report["datasets"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["confirmatory_data_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
