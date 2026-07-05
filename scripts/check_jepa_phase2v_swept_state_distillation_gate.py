#!/usr/bin/env python3
"""Check the bounded Phase 2V swept-state distillation promotion gate."""
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
    min_primitive_match_rate: float,
    max_selected_primitive_excess: float,
    max_swept_state_mean_absolute_error: float,
    max_swept_state_max_feature_mae: float,
) -> dict:
    final_validation = report.get("final_validation")
    summary = (
        final_validation or {}
    ).get("primitive_affordance_selection_summary")
    baseline = (
        report.get("primitive_action_only_baseline") or {}
    ).get("selection_summary")
    swept_state = (final_validation or {}).get("swept_state_error_summary")
    failure_reasons = []
    observed = {
        "primitive_match_rate": None,
        "mean_target_utility_regret": None,
        "selected_max_primitive_fraction": None,
        "oracle_max_primitive_fraction": None,
        "swept_state_mean_absolute_error": None,
        "swept_state_mean_squared_error": None,
        "swept_state_max_feature_mae": None,
        "swept_state_worst_feature_name": None,
    }
    baseline_observed = {
        "primitive_match_rate": None,
        "mean_target_utility_regret": None,
        "selected_max_primitive_fraction": None,
    }
    if not isinstance(summary, dict):
        failure_reasons.append("missing_primitive_affordance_selection_summary")
    else:
        observed.update(
            {
                "primitive_match_rate": summary.get("primitive_match_rate"),
                "mean_target_utility_regret": summary.get(
                    "mean_target_utility_regret"
                ),
                "selected_max_primitive_fraction": summary.get(
                    "selected_max_primitive_fraction"
                ),
                "oracle_max_primitive_fraction": summary.get(
                    "oracle_max_primitive_fraction"
                ),
            }
        )
        for key in (
            "primitive_match_rate",
            "mean_target_utility_regret",
            "selected_max_primitive_fraction",
            "oracle_max_primitive_fraction",
        ):
            if not _finite(observed[key]):
                failure_reasons.append(f"nonfinite_{key}")
        if _finite(observed["primitive_match_rate"]) and (
            float(observed["primitive_match_rate"]) < min_primitive_match_rate
        ):
            failure_reasons.append("primitive_match_rate_below_threshold")
        if _finite(observed["selected_max_primitive_fraction"]) and _finite(
            observed["oracle_max_primitive_fraction"]
        ):
            max_allowed = min(
                1.0,
                float(observed["oracle_max_primitive_fraction"])
                + max_selected_primitive_excess,
            )
            if float(observed["selected_max_primitive_fraction"]) > max_allowed:
                failure_reasons.append(
                    "selected_primitive_distribution_more_collapsed_than_oracle"
                )

    if not isinstance(baseline, dict):
        failure_reasons.append("missing_primitive_action_only_baseline_summary")
    else:
        baseline_observed = {
            "primitive_match_rate": baseline.get("primitive_match_rate"),
            "mean_target_utility_regret": baseline.get("mean_target_utility_regret"),
            "selected_max_primitive_fraction": baseline.get(
                "selected_max_primitive_fraction"
            ),
        }
        for key, value in baseline_observed.items():
            if not _finite(value):
                failure_reasons.append(f"nonfinite_baseline_{key}")

    if not isinstance(swept_state, dict):
        failure_reasons.append("missing_swept_state_error_summary")
    else:
        observed.update(
            {
                "swept_state_mean_absolute_error": swept_state.get(
                    "mean_absolute_error"
                ),
                "swept_state_mean_squared_error": swept_state.get(
                    "mean_squared_error"
                ),
                "swept_state_max_feature_mae": swept_state.get(
                    "max_feature_mean_absolute_error"
                ),
                "swept_state_worst_feature_name": swept_state.get(
                    "worst_feature_name"
                ),
            }
        )
        if swept_state.get("finite_predictions") is not True:
            failure_reasons.append("nonfinite_swept_state_predictions")
        if swept_state.get("finite_targets") is not True:
            failure_reasons.append("nonfinite_swept_state_targets")
        if not _finite(observed["swept_state_mean_absolute_error"]):
            failure_reasons.append("nonfinite_swept_state_mean_absolute_error")
        elif (
            float(observed["swept_state_mean_absolute_error"])
            > max_swept_state_mean_absolute_error
        ):
            failure_reasons.append(
                "swept_state_mean_absolute_error_above_threshold"
            )
        if not _finite(observed["swept_state_max_feature_mae"]):
            failure_reasons.append("nonfinite_swept_state_max_feature_mae")
        elif (
            float(observed["swept_state_max_feature_mae"])
            > max_swept_state_max_feature_mae
        ):
            failure_reasons.append("swept_state_max_feature_mae_above_threshold")

    if (
        isinstance(summary, dict)
        and isinstance(baseline, dict)
        and _finite(observed["primitive_match_rate"])
        and _finite(baseline_observed["primitive_match_rate"])
        and float(observed["primitive_match_rate"])
        <= float(baseline_observed["primitive_match_rate"])
    ):
        failure_reasons.append("primitive_match_rate_not_above_action_only_baseline")
    if (
        isinstance(summary, dict)
        and isinstance(baseline, dict)
        and _finite(observed["mean_target_utility_regret"])
        and _finite(baseline_observed["mean_target_utility_regret"])
        and float(observed["mean_target_utility_regret"])
        >= float(baseline_observed["mean_target_utility_regret"])
    ):
        failure_reasons.append("regret_not_below_action_only_baseline")

    return {
        "schema": "jepa_phase2v_swept_state_distillation_gate_report_v0",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "observed": observed,
        "primitive_action_only_baseline": baseline_observed,
        "thresholds": {
            "min_primitive_match_rate": min_primitive_match_rate,
            "max_selected_primitive_excess": max_selected_primitive_excess,
            "max_swept_state_mean_absolute_error": (
                max_swept_state_mean_absolute_error
            ),
            "max_swept_state_max_feature_mae": max_swept_state_max_feature_mae,
        },
        "report_schema": report.get("schema"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-primitive-match-rate", type=float, default=0.50)
    parser.add_argument("--max-selected-primitive-excess", type=float, default=0.20)
    parser.add_argument("--max-swept-state-mean-absolute-error", type=float, default=0.20)
    parser.add_argument("--max-swept-state-max-feature-mae", type=float, default=0.40)
    args = parser.parse_args()

    if not 0.0 <= args.min_primitive_match_rate <= 1.0:
        parser.error("--min-primitive-match-rate must lie in [0, 1]")
    if not 0.0 <= args.max_selected_primitive_excess <= 1.0:
        parser.error("--max-selected-primitive-excess must lie in [0, 1]")
    if args.max_swept_state_mean_absolute_error < 0.0:
        parser.error("--max-swept-state-mean-absolute-error must be non-negative")
    if args.max_swept_state_max_feature_mae < 0.0:
        parser.error("--max-swept-state-max-feature-mae must be non-negative")

    report = json.loads(args.report.read_text())
    result = check_gate(
        report,
        min_primitive_match_rate=args.min_primitive_match_rate,
        max_selected_primitive_excess=args.max_selected_primitive_excess,
        max_swept_state_mean_absolute_error=(
            args.max_swept_state_mean_absolute_error
        ),
        max_swept_state_max_feature_mae=args.max_swept_state_max_feature_mae,
    )
    result["report_path"] = str(args.report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
