#!/usr/bin/env python3
"""Audit the Phase 2Q true-factor primitive-affordance selection ceiling."""
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
from lewm.benchmarks.phase2q_factorized_ceiling import (  # noqa: E402
    DEFAULT_PHASE2Q_SELECTION_KWARGS,
    phase2q_factorized_affordance_ceiling_audit,
)
from scripts.check_jepa_phase2m_primitive_affordance_gate import check_gate  # noqa: E402


def _selection_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "safe_threshold": args.safe_threshold,
        "unsafe_threshold": args.unsafe_threshold,
        "task_gain_weight": args.task_gain_weight,
        "p05_clearance_weight": args.p05_clearance_weight,
        "minimum_clearance_weight": args.minimum_clearance_weight,
        "unsafe_penalty_weight": args.unsafe_penalty_weight,
        "heading_weight": args.heading_weight,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--safe-threshold",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["safe_threshold"],
    )
    parser.add_argument(
        "--unsafe-threshold",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["unsafe_threshold"],
    )
    parser.add_argument(
        "--task-gain-weight",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["task_gain_weight"],
    )
    parser.add_argument(
        "--p05-clearance-weight",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["p05_clearance_weight"],
    )
    parser.add_argument(
        "--minimum-clearance-weight",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["minimum_clearance_weight"],
    )
    parser.add_argument(
        "--unsafe-penalty-weight",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["unsafe_penalty_weight"],
    )
    parser.add_argument(
        "--heading-weight",
        type=float,
        default=DEFAULT_PHASE2Q_SELECTION_KWARGS["heading_weight"],
    )
    parser.add_argument("--min-primitive-match-rate", type=float, default=0.50)
    parser.add_argument("--max-selected-primitive-excess", type=float, default=0.20)
    parser.add_argument(
        "--require-gate-pass",
        action="store_true",
        help="Exit non-zero if the true-factor ceiling does not pass the gate.",
    )
    args = parser.parse_args()

    if args.max_train_rows < 0:
        parser.error("--max-train-rows must be non-negative")
    if args.max_validation_rows < 0:
        parser.error("--max-validation-rows must be non-negative")
    if not 0.0 <= args.safe_threshold <= 1.0:
        parser.error("--safe-threshold must lie in [0, 1]")
    if not 0.0 <= args.unsafe_threshold <= 1.0:
        parser.error("--unsafe-threshold must lie in [0, 1]")
    if not 0.0 <= args.min_primitive_match_rate <= 1.0:
        parser.error("--min-primitive-match-rate must lie in [0, 1]")
    if not 0.0 <= args.max_selected_primitive_excess <= 1.0:
        parser.error("--max-selected-primitive-excess must lie in [0, 1]")

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

    report = phase2q_factorized_affordance_ceiling_audit(
        train_rows=train_rows,
        validation_rows=validation_rows,
        seed=args.seed,
        selection_kwargs=_selection_kwargs(args),
    )
    report["train_data"]["path"] = str(args.train_data.resolve())
    report["train_data"]["load_audit"] = train_load_audit
    report["validation_data"]["path"] = str(args.validation_data.resolve())
    report["validation_data"]["load_audit"] = validation_load_audit
    report["split_overlap"] = overlap
    gate = check_gate(
        report,
        min_primitive_match_rate=args.min_primitive_match_rate,
        max_selected_primitive_excess=args.max_selected_primitive_excess,
    )
    report["validation_gate"] = gate
    write_json(args.output, report)
    compact = {
        "schema": report["schema"],
        "output": str(args.output.resolve()),
        "gate_passed": gate["passed"],
        "failure_reasons": gate["failure_reasons"],
        "observed": gate["observed"],
        "primitive_action_only_baseline": gate["primitive_action_only_baseline"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 1 if args.require_gate_pass and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
