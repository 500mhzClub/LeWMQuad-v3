#!/usr/bin/env python3
"""Check the Phase 3A positive-control JEPA promotion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(report: dict, *, min_primitive_match_rate: float = 0.50) -> dict:
    validation = report["final_validation"]
    control_surface = (
        "decision_rollout_controls"
        if "decision_rollout_controls" in validation
        else "rollout_controls"
    )
    controls = validation[control_surface]
    representation = controls["representation"]
    primitive = validation["primitive_selection"]
    prior = report["action_only_prior"]
    failures = []
    if representation["collapse_warning"]:
        failures.append("collapse_warning")
    per_horizon_gate_checks = []
    for step in controls["per_horizon_step"]:
        step_number = int(step["step"])
        action_advantage_passed = (
            step["meaningful_real_action_beats_zero"]
            and step["meaningful_real_action_beats_shuffled"]
        )
        if step_number == 1:
            persistence_or_action_advantage_passed = step[
                "free_running_beats_persistence"
            ]
        else:
            persistence_or_action_advantage_passed = (
                step["free_running_beats_persistence"] or action_advantage_passed
            )
        if not persistence_or_action_advantage_passed:
            if step_number == 1:
                failures.append(f"step_{step_number}_persistence_not_beaten")
            else:
                failures.append(
                    f"step_{step_number}_persistence_or_action_advantage_below_threshold"
                )
        if not step["meaningful_real_action_beats_zero"]:
            failures.append(f"step_{step_number}_zero_action_advantage_below_threshold")
        if not step["meaningful_real_action_beats_shuffled"]:
            failures.append(
                f"step_{step_number}_hard_negative_advantage_below_threshold"
            )
        per_horizon_gate_checks.append(
            {
                "step": step_number,
                "first_horizon_requires_persistence": step_number == 1,
                "later_horizon_persistence_or_action_advantage": step_number > 1,
                "persistence_or_action_advantage_passed": persistence_or_action_advantage_passed,
            }
        )
    if primitive["primitive_match_rate"] < min_primitive_match_rate:
        failures.append("primitive_match_rate_below_threshold")
    if primitive["primitive_match_rate"] <= prior["primitive_match_rate"]:
        failures.append("primitive_match_rate_not_above_action_only_prior")
    if primitive["mean_target_utility_regret"] >= prior["mean_target_utility_regret"]:
        failures.append("regret_not_below_action_only_prior")
    return {
        "schema": "jepa_phase3a_positive_control_gate_v1",
        "passed": not failures,
        "failure_reasons": failures,
        "observed": {
            "control_surface": control_surface,
            "primitive_match_rate": primitive["primitive_match_rate"],
            "mean_target_utility_regret": primitive["mean_target_utility_regret"],
            "action_only_prior_match_rate": prior["primitive_match_rate"],
            "action_only_prior_regret": prior["mean_target_utility_regret"],
            "collapse_warning": representation["collapse_warning"],
            "per_horizon_step": controls["per_horizon_step"],
            "per_horizon_gate_checks": per_horizon_gate_checks,
        },
        "thresholds": {
            "min_primitive_match_rate": min_primitive_match_rate,
            "action_advantage_over_target_change": 0.10,
            "free_running_vs_persistence_mse_ratio": 1.0,
            "first_horizon_requires_persistence": True,
            "later_horizon_requires_persistence_or_real_action_advantage": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-primitive-match-rate", type=float, default=0.50)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    gate = evaluate_gate(
        report,
        min_primitive_match_rate=args.min_primitive_match_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
