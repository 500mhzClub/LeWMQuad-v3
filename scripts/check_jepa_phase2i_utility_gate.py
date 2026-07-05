#!/usr/bin/env python3
"""Check the bounded Phase 2I utility-selection promotion gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_gate(
    report: dict,
    *,
    min_top1_match_rate: float,
    min_first_primitive_match_rate: float,
) -> dict:
    final_validation = report.get("final_validation")
    summary = (
        final_validation or {}
    ).get("action_utility_selection_summary")
    failure_reasons = []
    observed = {
        "top1_match_rate": None,
        "first_primitive_match_rate": None,
        "mean_target_utility_regret": None,
    }
    if not isinstance(summary, dict):
        failure_reasons.append("missing_action_utility_selection_summary")
    else:
        observed = {
            "top1_match_rate": summary.get("top1_match_rate"),
            "first_primitive_match_rate": summary.get("first_primitive_match_rate"),
            "mean_target_utility_regret": summary.get("mean_target_utility_regret"),
        }
        for key, value in observed.items():
            if not _finite(value):
                failure_reasons.append(f"nonfinite_{key}")
        if _finite(observed["top1_match_rate"]) and (
            float(observed["top1_match_rate"]) < min_top1_match_rate
        ):
            failure_reasons.append("top1_match_rate_below_threshold")
        if _finite(observed["first_primitive_match_rate"]) and (
            float(observed["first_primitive_match_rate"])
            < min_first_primitive_match_rate
        ):
            failure_reasons.append("first_primitive_match_rate_below_threshold")

    baselines = report.get("baseline_reference") or []
    baseline_observed = []
    for baseline in baselines:
        if not isinstance(baseline, dict):
            continue
        baseline_observed.append(
            {
                "baseline": baseline.get("baseline"),
                "top1_match_rate": baseline.get("top1_match_rate"),
                "first_primitive_match_rate": baseline.get(
                    "first_primitive_match_rate"
                ),
                "mean_target_utility_regret": baseline.get(
                    "mean_target_utility_regret"
                ),
            }
        )
    finite_top1 = [
        float(item["top1_match_rate"])
        for item in baseline_observed
        if _finite(item["top1_match_rate"])
    ]
    finite_first = [
        float(item["first_primitive_match_rate"])
        for item in baseline_observed
        if _finite(item["first_primitive_match_rate"])
    ]
    finite_regret = [
        float(item["mean_target_utility_regret"])
        for item in baseline_observed
        if _finite(item["mean_target_utility_regret"])
    ]
    if summary is not None and finite_top1 and _finite(observed["top1_match_rate"]):
        if float(observed["top1_match_rate"]) <= max(finite_top1):
            failure_reasons.append("top1_match_rate_not_above_action_only_baselines")
    if summary is not None and finite_first and _finite(
        observed["first_primitive_match_rate"]
    ):
        if float(observed["first_primitive_match_rate"]) <= max(finite_first):
            failure_reasons.append(
                "first_primitive_match_rate_not_above_action_only_baselines"
            )
    if summary is not None and finite_regret and _finite(
        observed["mean_target_utility_regret"]
    ):
        if float(observed["mean_target_utility_regret"]) >= min(finite_regret):
            failure_reasons.append("regret_not_below_action_only_baselines")

    return {
        "schema": "jepa_phase2i_utility_gate_report_v0",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "observed": observed,
        "baseline_reference": baseline_observed,
        "thresholds": {
            "min_top1_match_rate": min_top1_match_rate,
            "min_first_primitive_match_rate": min_first_primitive_match_rate,
        },
        "report_schema": report.get("schema"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-top1-match-rate", type=float, default=0.25)
    parser.add_argument("--min-first-primitive-match-rate", type=float, default=0.50)
    args = parser.parse_args()

    if not 0.0 <= args.min_top1_match_rate <= 1.0:
        parser.error("--min-top1-match-rate must lie in [0, 1]")
    if not 0.0 <= args.min_first_primitive_match_rate <= 1.0:
        parser.error("--min-first-primitive-match-rate must lie in [0, 1]")

    report = json.loads(args.report.read_text())
    result = check_gate(
        report,
        min_top1_match_rate=args.min_top1_match_rate,
        min_first_primitive_match_rate=args.min_first_primitive_match_rate,
    )
    result["report_path"] = str(args.report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
