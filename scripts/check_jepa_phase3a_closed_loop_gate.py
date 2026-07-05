#!/usr/bin/env python3
"""Check the Phase 3A closed-loop 2D navigation gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(
    report: dict,
    *,
    min_claimed_episodes: int,
    min_marker_seen_episodes: int,
    max_collision_steps: int,
    require_learned_value_field: bool,
    forbid_fixed_marker_target: bool,
    require_extractor_head: bool,
) -> dict:
    summaries = report.get("episode_summaries", [])
    collision_steps = sum(int(item.get("collision_steps", 0)) for item in summaries)
    marker_seen_episodes = sum(
        1 for item in summaries if bool(item.get("marker_seen_ever", False))
    )
    claimed_episodes = int(report.get("claimed_episodes", 0))
    failures = []

    if str(report.get("schema")) != "jepa_phase3a_closed_loop_demo_report_v0":
        failures.append("unexpected_report_schema")
    if claimed_episodes < min_claimed_episodes:
        failures.append("claimed_episodes_below_threshold")
    if marker_seen_episodes < min_marker_seen_episodes:
        failures.append("marker_seen_episodes_below_threshold")
    if collision_steps > max_collision_steps:
        failures.append("collision_steps_above_threshold")
    if require_learned_value_field and report.get("score_source") != (
        "latent_recurrent_learned_value_field_planner"
    ):
        failures.append("score_source_not_learned_value_field")
    if forbid_fixed_marker_target and bool(report.get("latent_value_fixed_marker_target")):
        failures.append("fixed_marker_target_enabled")
    if require_extractor_head and not report.get("latent_value_extractor_head"):
        failures.append("latent_value_extractor_head_missing")

    return {
        "schema": "jepa_phase3a_closed_loop_gate_v0",
        "passed": not failures,
        "failure_reasons": failures,
        "observed": {
            "claimed_episodes": claimed_episodes,
            "episodes_attempted": int(report.get("episodes_attempted", len(summaries))),
            "marker_seen_episodes": marker_seen_episodes,
            "collision_steps": collision_steps,
            "score_source": report.get("score_source"),
            "latent_value_fixed_marker_target": report.get(
                "latent_value_fixed_marker_target"
            ),
            "latent_value_extractor_head": report.get("latent_value_extractor_head"),
            "latent_value_target_top_k": report.get("latent_value_target_top_k"),
            "latent_value_marker_target_top_k": report.get(
                "latent_value_marker_target_top_k"
            ),
            "latent_value_sparse_target_top_k": report.get(
                "latent_value_sparse_target_top_k"
            ),
        },
        "thresholds": {
            "min_claimed_episodes": min_claimed_episodes,
            "min_marker_seen_episodes": min_marker_seen_episodes,
            "max_collision_steps": max_collision_steps,
            "require_learned_value_field": require_learned_value_field,
            "forbid_fixed_marker_target": forbid_fixed_marker_target,
            "require_extractor_head": require_extractor_head,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-claimed-episodes", type=int, default=16)
    parser.add_argument("--min-marker-seen-episodes", type=int, default=16)
    parser.add_argument("--max-collision-steps", type=int, default=0)
    parser.add_argument("--allow-non-learned-value-field", action="store_true")
    parser.add_argument("--allow-fixed-marker-target", action="store_true")
    parser.add_argument("--require-extractor-head", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    result = evaluate_gate(
        report,
        min_claimed_episodes=args.min_claimed_episodes,
        min_marker_seen_episodes=args.min_marker_seen_episodes,
        max_collision_steps=args.max_collision_steps,
        require_learned_value_field=not args.allow_non_learned_value_field,
        forbid_fixed_marker_target=not args.allow_fixed_marker_target,
        require_extractor_head=bool(args.require_extractor_head),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
