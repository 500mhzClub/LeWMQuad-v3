#!/usr/bin/env python3
"""Audit Phase 2H action-utility labels and action-only validation baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2_data import (  # noqa: E402
    load_spatial_future_rows,
    pairwise_split_overlap,
)
from lewm.benchmarks.phase2h_action_utility import (  # noqa: E402
    phase2h_action_utility_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    args = parser.parse_args()

    if args.max_train_rows < 0:
        parser.error("--max-train-rows must be non-negative")
    if args.max_validation_rows < 0:
        parser.error("--max-validation-rows must be non-negative")

    train_rows, train_load_audit = load_spatial_future_rows(
        args.train_data,
        mode="all",
        max_rows=args.max_train_rows,
    )
    validation_rows, validation_load_audit = load_spatial_future_rows(
        args.validation_data,
        mode="all",
        max_rows=args.max_validation_rows,
    )
    overlap = pairwise_split_overlap(
        {"train": train_rows, "validation": validation_rows}
    )
    if any(
        value["scene_ids"] or value["source_keys"]
        for value in overlap.values()
    ):
        raise SystemExit(f"train/validation overlap is prohibited: {overlap}")

    report = phase2h_action_utility_audit(
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    report["train_data"] = {
        "path": str(args.train_data.resolve()),
        "load_audit": train_load_audit,
    }
    report["validation_data"] = {
        "path": str(args.validation_data.resolve()),
        "load_audit": validation_load_audit,
    }
    report["split_overlap"] = overlap
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output.resolve()),
                "train_valid_utility_rows": report["train_label_audit"][
                    "valid_utility_rows"
                ],
                "validation_valid_utility_rows": report[
                    "validation_label_audit"
                ]["valid_utility_rows"],
                "validation_action_only_baselines": [
                    {
                        "baseline": item["baseline"],
                        "top1_match_rate": item.get("top1_match_rate"),
                        "first_primitive_match_rate": item.get(
                            "first_primitive_match_rate"
                        ),
                        "mean_target_utility_regret": item.get(
                            "mean_target_utility_regret"
                        ),
                    }
                    for item in report["validation_action_only_baselines"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
