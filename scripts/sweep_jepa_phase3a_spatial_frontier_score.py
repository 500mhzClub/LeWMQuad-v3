#!/usr/bin/env python3
"""Sweep Phase 3A learned spatial-frontier score constants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_explore_claim import (  # noqa: E402
    summarize_explore_claim_predictions,
)
from lewm.benchmarks.phase3a_positive_control import read_jsonl  # noqa: E402
from scripts.report_jepa_phase3a_explore_claim import (  # noqa: E402
    load_model,
    predict_model_scores,
)


def _variant_grid() -> list[dict]:
    return [
        {
            "name": "p2_n035_t050_w025",
            "collision_penalty": 2.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p4_n035_t050_w025",
            "collision_penalty": 4.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p4_n035_t050_w025_tau050",
            "collision_penalty": 4.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.5,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p4_n035_t050_w025_tau050_u050_w025",
            "collision_penalty": 4.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.5,
            "marker_update_threshold": 0.50,
            "marker_update_width": 0.25,
        },
        {
            "name": "p4_n035_t050_w025_tau050_u060_w020",
            "collision_penalty": 4.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.5,
            "marker_update_threshold": 0.60,
            "marker_update_width": 0.20,
        },
        {
            "name": "p4_n035_t050_w025_tau025",
            "collision_penalty": 4.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.25,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p8_n035_t050_w025",
            "collision_penalty": 8.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p8_n035_t050_w025_tau050",
            "collision_penalty": 8.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.5,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p8_n035_t050_w025_tau050_u050_w025",
            "collision_penalty": 8.0,
            "novelty_reward": 0.35,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 0.5,
            "marker_update_threshold": 0.50,
            "marker_update_width": 0.25,
        },
        {
            "name": "p8_n025_t050_w025",
            "collision_penalty": 8.0,
            "novelty_reward": 0.25,
            "marker_gate_threshold": 0.50,
            "marker_gate_width": 0.25,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p8_n025_t060_w020",
            "collision_penalty": 8.0,
            "novelty_reward": 0.25,
            "marker_gate_threshold": 0.60,
            "marker_gate_width": 0.20,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
        {
            "name": "p12_n025_t060_w020",
            "collision_penalty": 12.0,
            "novelty_reward": 0.25,
            "marker_gate_threshold": 0.60,
            "marker_gate_width": 0.20,
            "marker_score_temperature": 1.0,
            "marker_update_threshold": 0.0,
            "marker_update_width": 1.0,
        },
    ]


def _compact_phase(summary: dict, phase: str) -> dict:
    item = summary["phases"][phase]
    return {
        "source_states": item["source_states"],
        "selected_goal_claimed_rate": item["selected_goal_claimed_rate"],
        "selected_future_goal_marker_seen_rate": item[
            "selected_future_goal_marker_seen_rate"
        ],
        "mean_selected_sequence_target_utility_regret": item[
            "mean_selected_sequence_target_utility_regret"
        ],
        "mean_target_utility_regret": item["mean_target_utility_regret"],
        "primitive_match_rate": item["primitive_match_rate"],
        "sequence_match_rate": item["sequence_match_rate"],
        "top10_claimed_rate": item["topk_claimed_rate"]["10"],
        "top10_oracle_sequence_rate": item["topk_oracle_sequence_rate"]["10"],
        "selected_primitive_counts": item["selected_primitive_counts"],
    }


def _variant_score(summary: dict) -> tuple[float, float, float]:
    claim = summary["phases"]["claim_after_marker_seen"]
    discover = summary["phases"]["discover_visible_marker"]
    explore = summary["phases"]["explore_unseen"]
    return (
        float(claim["selected_goal_claimed_rate"]),
        -float(claim["mean_selected_sequence_target_utility_regret"]),
        -float(discover["mean_selected_sequence_target_utility_regret"])
        - float(explore["mean_selected_sequence_target_utility_regret"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows = read_jsonl(args.validation_data)
    model, _ = load_model(args.checkpoint, device=device)

    results = []
    for variant in _variant_grid():
        model.spatial_frontier_collision_penalty = float(variant["collision_penalty"])
        model.spatial_frontier_novelty_reward = float(variant["novelty_reward"])
        model.spatial_frontier_marker_gate_threshold = float(
            variant["marker_gate_threshold"]
        )
        model.spatial_frontier_marker_gate_width = float(variant["marker_gate_width"])
        model.spatial_marker_memory_score_temperature = float(
            variant["marker_score_temperature"]
        )
        model.spatial_frontier_marker_update_threshold = float(
            variant["marker_update_threshold"]
        )
        model.spatial_frontier_marker_update_width = float(
            variant["marker_update_width"]
        )
        scores = predict_model_scores(
            model,
            rows,
            source_states_per_batch=args.source_states_per_batch,
            device=device,
        )
        summary = summarize_explore_claim_predictions(
            rows,
            scores["spatial_frontier_memory_score"],
        )
        results.append(
            {
                "variant": variant,
                "sort_key": _variant_score(summary),
                "phases": {
                    phase: _compact_phase(summary, phase)
                    for phase in (
                        "explore_unseen",
                        "discover_visible_marker",
                        "claim_after_marker_seen",
                    )
                },
            }
        )

    results.sort(key=lambda item: tuple(item["sort_key"]), reverse=True)
    report = {
        "schema": "jepa_phase3a_spatial_frontier_score_sweep_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "device": str(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
