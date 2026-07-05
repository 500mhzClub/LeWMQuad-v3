#!/usr/bin/env python3
"""Check the Phase 3A no-beacon explore-then-claim promotion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(
    report: dict,
    *,
    min_explore_sources: int = 16,
    min_discover_sources: int = 4,
    min_claim_sources: int = 4,
    min_explore_match_rate: float = 0.50,
    max_explore_regret: float = 1.0,
    min_discover_marker_rate: float = 0.75,
    min_claim_rate: float = 0.50,
    max_claim_sequence_regret: float = 2.0,
    min_claim_top5_claimed_rate: float = 0.75,
    min_egocentric_claim_rate: float = 0.75,
    max_egocentric_claim_sequence_regret: float = 2.0,
    min_egocentric_claim_top5_claimed_rate: float = 0.75,
) -> dict:
    memory = report["summaries"]["memory"]["phases"]
    no_memory = report["summaries"]["no_memory"]["phases"]
    egocentric_marker_memory = report.get("summaries", {}).get(
        "egocentric_marker_memory",
        {},
    ).get("phases", {})
    model_online_marker_memory = (
        report.get("score_summaries", {})
        .get("memory", {})
        .get("online_marker_memory_score", {})
        .get("phases", {})
    )
    model_spatial_marker_memory = (
        report.get("score_summaries", {})
        .get("memory", {})
        .get("spatial_marker_memory_score", {})
        .get("phases", {})
    )
    comparisons = report["comparisons"]["memory_minus_no_memory"]
    aggregate = report["memory_training_aggregate"]
    no_memory_aggregate = report["no_memory_training_aggregate"]
    validation_audit = report.get("validation_audit", {})
    failures = []
    baseline_failures = []
    model_online_failures = []
    model_spatial_failures = []
    enforce_aggregate_comparison = (
        min_explore_sources > 0 or min_discover_sources > 0
    )

    if validation_audit.get("current_goal_beacon_counts") != {"False": validation_audit.get("rows")}:
        failures.append("validation_current_goal_beacon_not_fully_disabled")
    if validation_audit.get("history_goal_beacon_counts") != {"False": validation_audit.get("rows")}:
        failures.append("validation_history_goal_beacon_not_fully_disabled")

    explore = memory["explore_unseen"]
    if int(explore["source_states"]) < min_explore_sources:
        failures.append("explore_unseen_source_states_below_threshold")
    if int(explore["source_states"]) > 0 and (
        explore["primitive_match_rate"] is None
        or float(explore["primitive_match_rate"]) < min_explore_match_rate
    ):
        failures.append("explore_unseen_primitive_match_rate_below_threshold")
    if int(explore["source_states"]) > 0 and (
        explore["mean_target_utility_regret"] is None
        or float(explore["mean_target_utility_regret"]) > max_explore_regret
    ):
        failures.append("explore_unseen_regret_above_threshold")

    discover = memory["discover_visible_marker"]
    if int(discover["source_states"]) < min_discover_sources:
        failures.append("discover_visible_marker_source_states_below_threshold")
    if int(discover["source_states"]) > 0 and (
        discover["selected_future_goal_marker_seen_rate"] is None
        or float(discover["selected_future_goal_marker_seen_rate"])
        < min_discover_marker_rate
    ):
        failures.append("discover_visible_marker_selection_rate_below_threshold")

    claim = memory["claim_after_marker_seen"]
    if int(claim["source_states"]) < min_claim_sources:
        failures.append("claim_after_marker_seen_source_states_below_threshold")
    if int(claim["source_states"]) > 0 and (
        claim["selected_goal_claimed_rate"] is None
        or float(claim["selected_goal_claimed_rate"]) < min_claim_rate
    ):
        failures.append("claim_after_marker_seen_selection_rate_below_threshold")
    if int(claim["source_states"]) > 0 and (
        claim["mean_selected_sequence_target_utility_regret"] is None
        or float(claim["mean_selected_sequence_target_utility_regret"])
        > max_claim_sequence_regret
    ):
        failures.append("claim_after_marker_seen_sequence_regret_above_threshold")
    claim_top5 = claim.get("topk_claimed_rate", {}).get("5")
    if int(claim["source_states"]) > 0 and (
        claim_top5 is None or float(claim_top5) < min_claim_top5_claimed_rate
    ):
        failures.append("claim_after_marker_seen_top5_claimed_rate_below_threshold")

    if enforce_aggregate_comparison:
        if (
            aggregate["primitive_match_rate"]
            <= no_memory_aggregate["primitive_match_rate"]
        ):
            failures.append("aggregate_memory_not_above_no_memory_match_rate")
        if (
            aggregate["mean_target_utility_regret"]
            >= no_memory_aggregate["mean_target_utility_regret"]
        ):
            failures.append("aggregate_memory_not_below_no_memory_regret")
    if (
        comparisons["explore_unseen"]["mean_target_utility_regret_delta"] is not None
        and comparisons["explore_unseen"]["mean_target_utility_regret_delta"] > 0.0
    ):
        failures.append("explore_unseen_memory_regret_worse_than_no_memory")
    if (
        comparisons["discover_visible_marker"]["selected_future_goal_marker_seen_rate_delta"]
        is not None
        and comparisons["discover_visible_marker"][
            "selected_future_goal_marker_seen_rate_delta"
        ]
        < 0.0
    ):
        failures.append("discover_visible_marker_memory_selects_marker_less_often")
    if (
        comparisons["claim_after_marker_seen"]["selected_goal_claimed_rate_delta"]
        is not None
        and comparisons["claim_after_marker_seen"][
            "selected_goal_claimed_rate_delta"
        ]
        <= 0.0
    ):
        failures.append("claim_after_marker_seen_memory_does_not_improve_claim_rate")
    if (
        comparisons["claim_after_marker_seen"]["mean_selected_sequence_regret_delta"]
        is not None
        and comparisons["claim_after_marker_seen"][
            "mean_selected_sequence_regret_delta"
        ]
        >= 0.0
    ):
        failures.append("claim_after_marker_seen_memory_does_not_reduce_sequence_regret")

    egocentric_claim = egocentric_marker_memory.get("claim_after_marker_seen", {})
    if not egocentric_claim:
        baseline_failures.append("egocentric_marker_memory_claim_phase_missing")
    else:
        if int(egocentric_claim["source_states"]) < min_claim_sources:
            baseline_failures.append(
                "egocentric_marker_memory_claim_source_states_below_threshold"
            )
        if int(egocentric_claim["source_states"]) > 0 and (
            egocentric_claim["selected_goal_claimed_rate"] is None
            or float(egocentric_claim["selected_goal_claimed_rate"])
            < min_egocentric_claim_rate
        ):
            baseline_failures.append(
                "egocentric_marker_memory_claim_rate_below_threshold"
            )
        if int(egocentric_claim["source_states"]) > 0 and (
            egocentric_claim["mean_selected_sequence_target_utility_regret"] is None
            or float(egocentric_claim["mean_selected_sequence_target_utility_regret"])
            > max_egocentric_claim_sequence_regret
        ):
            baseline_failures.append(
                "egocentric_marker_memory_claim_sequence_regret_above_threshold"
            )
        egocentric_claim_top5 = egocentric_claim.get("topk_claimed_rate", {}).get("5")
        if int(egocentric_claim["source_states"]) > 0 and (
            egocentric_claim_top5 is None
            or float(egocentric_claim_top5)
            < min_egocentric_claim_top5_claimed_rate
        ):
            baseline_failures.append(
                "egocentric_marker_memory_claim_top5_claimed_rate_below_threshold"
            )

    model_online_claim = model_online_marker_memory.get("claim_after_marker_seen", {})
    if not model_online_claim:
        model_online_failures.append("online_marker_memory_score_claim_phase_missing")
    else:
        if int(model_online_claim["source_states"]) < min_claim_sources:
            model_online_failures.append(
                "online_marker_memory_score_claim_source_states_below_threshold"
            )
        if int(model_online_claim["source_states"]) > 0 and (
            model_online_claim["selected_goal_claimed_rate"] is None
            or float(model_online_claim["selected_goal_claimed_rate"])
            < min_egocentric_claim_rate
        ):
            model_online_failures.append(
                "online_marker_memory_score_claim_rate_below_threshold"
            )
        if int(model_online_claim["source_states"]) > 0 and (
            model_online_claim["mean_selected_sequence_target_utility_regret"] is None
            or float(model_online_claim["mean_selected_sequence_target_utility_regret"])
            > max_egocentric_claim_sequence_regret
        ):
            model_online_failures.append(
                "online_marker_memory_score_claim_sequence_regret_above_threshold"
            )
        model_online_claim_top5 = model_online_claim.get("topk_claimed_rate", {}).get(
            "5"
        )
        if int(model_online_claim["source_states"]) > 0 and (
            model_online_claim_top5 is None
            or float(model_online_claim_top5)
            < min_egocentric_claim_top5_claimed_rate
        ):
            model_online_failures.append(
                "online_marker_memory_score_claim_top5_claimed_rate_below_threshold"
            )

    model_spatial_claim = model_spatial_marker_memory.get("claim_after_marker_seen", {})
    if not model_spatial_claim:
        model_spatial_failures.append("spatial_marker_memory_score_claim_phase_missing")
    else:
        if int(model_spatial_claim["source_states"]) < min_claim_sources:
            model_spatial_failures.append(
                "spatial_marker_memory_score_claim_source_states_below_threshold"
            )
        if int(model_spatial_claim["source_states"]) > 0 and (
            model_spatial_claim["selected_goal_claimed_rate"] is None
            or float(model_spatial_claim["selected_goal_claimed_rate"])
            < min_egocentric_claim_rate
        ):
            model_spatial_failures.append(
                "spatial_marker_memory_score_claim_rate_below_threshold"
            )
        if int(model_spatial_claim["source_states"]) > 0 and (
            model_spatial_claim["mean_selected_sequence_target_utility_regret"] is None
            or float(model_spatial_claim["mean_selected_sequence_target_utility_regret"])
            > max_egocentric_claim_sequence_regret
        ):
            model_spatial_failures.append(
                "spatial_marker_memory_score_claim_sequence_regret_above_threshold"
            )
        model_spatial_claim_top5 = model_spatial_claim.get("topk_claimed_rate", {}).get(
            "5"
        )
        if int(model_spatial_claim["source_states"]) > 0 and (
            model_spatial_claim_top5 is None
            or float(model_spatial_claim_top5)
            < min_egocentric_claim_top5_claimed_rate
        ):
            model_spatial_failures.append(
                "spatial_marker_memory_score_claim_top5_claimed_rate_below_threshold"
            )

    return {
        "schema": "jepa_phase3a_explore_claim_gate_v0",
        "passed": not failures,
        "failure_reasons": failures,
        "non_privileged_egocentric_marker_memory": {
            "passed": not baseline_failures,
            "failure_reasons": baseline_failures,
            "claim_after_marker_seen": egocentric_claim,
            "thresholds": {
                "min_claim_sources": min_claim_sources,
                "min_claim_rate": min_egocentric_claim_rate,
                "max_claim_sequence_regret": max_egocentric_claim_sequence_regret,
                "min_claim_top5_claimed_rate": (
                    min_egocentric_claim_top5_claimed_rate
                ),
            },
        },
        "model_online_marker_memory_score": {
            "passed": not model_online_failures,
            "failure_reasons": model_online_failures,
            "claim_after_marker_seen": model_online_claim,
            "thresholds": {
                "min_claim_sources": min_claim_sources,
                "min_claim_rate": min_egocentric_claim_rate,
                "max_claim_sequence_regret": max_egocentric_claim_sequence_regret,
                "min_claim_top5_claimed_rate": (
                    min_egocentric_claim_top5_claimed_rate
                ),
            },
        },
        "model_spatial_marker_memory_score": {
            "passed": not model_spatial_failures,
            "failure_reasons": model_spatial_failures,
            "claim_after_marker_seen": model_spatial_claim,
            "thresholds": {
                "min_claim_sources": min_claim_sources,
                "min_claim_rate": min_egocentric_claim_rate,
                "max_claim_sequence_regret": max_egocentric_claim_sequence_regret,
                "min_claim_top5_claimed_rate": (
                    min_egocentric_claim_top5_claimed_rate
                ),
            },
        },
        "observed": {
            "aggregate_memory_match_rate": aggregate["primitive_match_rate"],
            "aggregate_no_memory_match_rate": no_memory_aggregate[
                "primitive_match_rate"
            ],
            "aggregate_memory_regret": aggregate["mean_target_utility_regret"],
            "aggregate_no_memory_regret": no_memory_aggregate[
                "mean_target_utility_regret"
            ],
            "aggregate_memory_sequence_regret": aggregate[
                "mean_selected_sequence_target_utility_regret"
            ],
            "aggregate_no_memory_sequence_regret": no_memory_aggregate[
                "mean_selected_sequence_target_utility_regret"
            ],
            "explore_unseen": explore,
            "discover_visible_marker": discover,
            "claim_after_marker_seen": claim,
            "egocentric_marker_memory_claim_after_marker_seen": egocentric_claim,
            "online_marker_memory_score_claim_after_marker_seen": model_online_claim,
            "spatial_marker_memory_score_claim_after_marker_seen": model_spatial_claim,
            "memory_minus_no_memory": comparisons,
            "validation_audit": {
                "rows": validation_audit.get("rows"),
                "current_goal_beacon_counts": validation_audit.get(
                    "current_goal_beacon_counts"
                ),
                "history_goal_beacon_counts": validation_audit.get(
                    "history_goal_beacon_counts"
                ),
            },
        },
        "thresholds": {
            "min_explore_sources": min_explore_sources,
            "min_discover_sources": min_discover_sources,
            "min_claim_sources": min_claim_sources,
            "min_explore_match_rate": min_explore_match_rate,
            "max_explore_regret": max_explore_regret,
            "min_discover_marker_rate": min_discover_marker_rate,
            "min_claim_rate": min_claim_rate,
            "max_claim_sequence_regret": max_claim_sequence_regret,
            "min_claim_top5_claimed_rate": min_claim_top5_claimed_rate,
            "min_egocentric_claim_rate": min_egocentric_claim_rate,
            "max_egocentric_claim_sequence_regret": (
                max_egocentric_claim_sequence_regret
            ),
            "min_egocentric_claim_top5_claimed_rate": (
                min_egocentric_claim_top5_claimed_rate
            ),
            "memory_must_beat_no_memory_aggregate": enforce_aggregate_comparison,
            "memory_must_beat_no_memory_on_claim_rate": True,
            "memory_must_reduce_claim_sequence_regret": True,
            "no_hidden_goal_beacons": True,
            "non_privileged_egocentric_marker_memory_reported": True,
            "model_online_marker_memory_score_reported": True,
            "model_spatial_marker_memory_score_reported": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-explore-sources", type=int, default=16)
    parser.add_argument("--min-discover-sources", type=int, default=4)
    parser.add_argument("--min-claim-sources", type=int, default=4)
    parser.add_argument("--min-explore-match-rate", type=float, default=0.50)
    parser.add_argument("--max-explore-regret", type=float, default=1.0)
    parser.add_argument("--min-discover-marker-rate", type=float, default=0.75)
    parser.add_argument("--min-claim-rate", type=float, default=0.50)
    parser.add_argument("--max-claim-sequence-regret", type=float, default=2.0)
    parser.add_argument("--min-claim-top5-claimed-rate", type=float, default=0.75)
    parser.add_argument("--min-egocentric-claim-rate", type=float, default=0.75)
    parser.add_argument(
        "--max-egocentric-claim-sequence-regret",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--min-egocentric-claim-top5-claimed-rate",
        type=float,
        default=0.75,
    )
    args = parser.parse_args()

    gate = evaluate_gate(
        json.loads(args.report.read_text()),
        min_explore_sources=args.min_explore_sources,
        min_discover_sources=args.min_discover_sources,
        min_claim_sources=args.min_claim_sources,
        min_explore_match_rate=args.min_explore_match_rate,
        max_explore_regret=args.max_explore_regret,
        min_discover_marker_rate=args.min_discover_marker_rate,
        min_claim_rate=args.min_claim_rate,
        max_claim_sequence_regret=args.max_claim_sequence_regret,
        min_claim_top5_claimed_rate=args.min_claim_top5_claimed_rate,
        min_egocentric_claim_rate=args.min_egocentric_claim_rate,
        max_egocentric_claim_sequence_regret=(
            args.max_egocentric_claim_sequence_regret
        ),
        min_egocentric_claim_top5_claimed_rate=(
            args.min_egocentric_claim_top5_claimed_rate
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
