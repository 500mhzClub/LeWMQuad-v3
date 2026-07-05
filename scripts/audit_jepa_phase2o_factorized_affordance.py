#!/usr/bin/env python3
"""Audit Phase 2O factorized primitive affordance target coverage."""
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
from lewm.benchmarks.phase2m_primitive_affordance import primitive_vocabulary  # noqa: E402
from lewm.benchmarks.phase2o_factorized_affordance import (  # noqa: E402
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    build_factorized_primitive_affordance_examples,
    factorized_affordance_dataset_audit,
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
    if any(value["scene_ids"] or value["source_keys"] for value in overlap.values()):
        raise SystemExit(f"train/validation overlap is prohibited: {overlap}")

    primitive_names = primitive_vocabulary(train_rows)
    train_examples = build_factorized_primitive_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
    )
    validation_examples = build_factorized_primitive_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
    )
    report = {
        "schema": "jepa_phase2o_factorized_affordance_audit_v0",
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "confirmatory_result": False,
        "train_data": {
            "path": str(args.train_data.resolve()),
            "load_audit": train_load_audit,
            "factorized_affordance_audit": factorized_affordance_dataset_audit(
                train_examples,
                split_name="train",
            ),
        },
        "validation_data": {
            "path": str(args.validation_data.resolve()),
            "load_audit": validation_load_audit,
            "factorized_affordance_audit": factorized_affordance_dataset_audit(
                validation_examples,
                split_name="validation",
            ),
        },
        "primitive_names": list(primitive_names),
        "split_overlap": overlap,
        "limitations": [
            "target coverage audit only",
            "train and validation evidence only",
            "test_id and test_hard remain unopened",
            "factor targets are generator-derived supervision",
        ],
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output.resolve()),
                "target_version": report["target_version"],
                "train_all_factors_complete": report["train_data"][
                    "factorized_affordance_audit"
                ]["all_factors_complete"],
                "train_core_factors_complete": report["train_data"][
                    "factorized_affordance_audit"
                ]["core_factors_complete"],
                "validation_all_factors_complete": report["validation_data"][
                    "factorized_affordance_audit"
                ]["all_factors_complete"],
                "validation_core_factors_complete": report["validation_data"][
                    "factorized_affordance_audit"
                ]["core_factors_complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
