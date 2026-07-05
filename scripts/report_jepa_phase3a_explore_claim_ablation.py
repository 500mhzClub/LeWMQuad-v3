#!/usr/bin/env python3
"""Run input ablations for Phase 3A no-beacon explore/claim scoring."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_explore_claim import (  # noqa: E402
    egocentric_explore_claim_predictions,
    summarize_explore_claim_predictions,
)
from lewm.benchmarks.phase3a_positive_control import read_jsonl  # noqa: E402
from lewm.benchmarks.phase3a_training import source_key  # noqa: E402
from scripts.report_jepa_phase3a_explore_claim import (  # noqa: E402
    load_model,
    predict_model_scores,
)


def _remove_green_marker(observation: list) -> list:
    """Suppress bright green marker evidence in a channel-major RGB crop."""

    tensor = torch.tensor(observation, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected RGB observation with shape C,H,W, got {tensor.shape}")
    red, green, blue = tensor[0], tensor[1], tensor[2]
    mask = (green > 0.45) & ((green - torch.maximum(red, blue)) > 0.25)
    tensor[:, mask] = 1.0
    return tensor.tolist()


def _remove_marker_from_row(row: dict) -> dict:
    row = copy.deepcopy(row)
    row["start_observation_rgb"] = _remove_green_marker(row["start_observation_rgb"])
    row["history_observations_rgb"] = [
        _remove_green_marker(item)
        for item in row.get("history_observations_rgb", [])
    ]
    for observation in row.get("future_observations", []):
        observation["observation_rgb"] = _remove_green_marker(
            observation["observation_rgb"]
        )
    return row


def _no_history(row: dict) -> dict:
    row = copy.deepcopy(row)
    row["history_observations_rgb"] = []
    row["history_actions"] = []
    row["history_primitive_sequence"] = []
    return row


def _shuffled_history_actions(row: dict) -> dict:
    row = copy.deepcopy(row)
    actions = list(row.get("history_actions", []))
    names = list(row.get("history_primitive_sequence", []))
    if len(actions) > 1:
        row["history_actions"] = actions[1:] + actions[:1]
    if len(names) > 1:
        row["history_primitive_sequence"] = names[1:] + names[:1]
    return row


def _identity(row: dict) -> dict:
    return copy.deepcopy(row)


def _candidate_action_shuffled_rows(rows: list[dict]) -> list[dict]:
    """Rotate candidate action sequences within each source group.

    Labels and future outcomes stay attached to their original rows. Only the
    action inputs seen by the model are mismatched, so performance is still
    summarized against the original rows.
    """

    transformed = [copy.deepcopy(row) for row in rows]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        rotated = indices[1:] + indices[:1]
        for target_index, source_index in zip(indices, rotated):
            transformed[target_index]["active_blocks"] = copy.deepcopy(
                rows[source_index]["active_blocks"]
            )
            transformed[target_index]["primitive_sequence"] = copy.deepcopy(
                rows[source_index]["primitive_sequence"]
            )
    return transformed


def _transform_rows(rows: list[dict], transform: str) -> list[dict]:
    row_transforms: dict[str, Callable[[dict], dict]] = {
        "identity": _identity,
        "no_history": _no_history,
        "shuffled_history_actions": _shuffled_history_actions,
        "marker_color_removed": _remove_marker_from_row,
    }
    if transform == "candidate_actions_shuffled":
        return _candidate_action_shuffled_rows(rows)
    if transform not in row_transforms:
        raise ValueError(f"unknown transform: {transform}")
    return [row_transforms[transform](row) for row in rows]


def _phase_table(summary: dict) -> dict[str, dict[str, float]]:
    return {
        phase: {
            "source_states": metrics["source_states"],
            "primitive_match_rate": metrics["primitive_match_rate"],
            "sequence_regret": metrics[
                "mean_selected_sequence_target_utility_regret"
            ],
            "selected_new_free_cells": metrics["mean_selected_new_free_cells"],
            "marker_seen_rate": metrics["selected_future_goal_marker_seen_rate"],
            "claim_rate": metrics["selected_goal_claimed_rate"],
        }
        for phase, metrics in summary["phases"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.validation_data)
    device = torch.device(args.device)
    model, report = load_model(args.checkpoint, device=device)

    results: dict[str, object] = {
        "schema": "jepa_phase3a_explore_claim_ablation_report_v0",
        "validation_data": str(args.validation_data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_completed_steps": report.get("completed_steps"),
        "checkpoint_device": report.get("device"),
        "evaluation_device": str(device),
        "online_frontier_marker": summarize_explore_claim_predictions(
            rows,
            egocentric_explore_claim_predictions(rows),
        ),
        "ablations": {},
    }

    for transform in (
        "identity",
        "no_history",
        "shuffled_history_actions",
        "marker_color_removed",
        "candidate_actions_shuffled",
    ):
        print(f"scoring {transform}", flush=True)
        transformed = _transform_rows(rows, transform)
        scores = predict_model_scores(
            model,
            transformed,
            source_states_per_batch=args.source_states_per_batch,
            device=device,
        )
        results["ablations"][transform] = {
            "input_transform": transform,
            "spatial_frontier_memory_score": summarize_explore_claim_predictions(
                rows,
                scores["spatial_frontier_memory_score"],
            ),
            "candidate_score": summarize_explore_claim_predictions(
                rows,
                scores["candidate_score"],
            ),
        }

    print("scoring spatial_memory_disabled", flush=True)
    original_weight = model.spatial_frontier_memory_score_weight
    model.spatial_frontier_memory_score_weight = 0.0
    scores = predict_model_scores(
        model,
        rows,
        source_states_per_batch=args.source_states_per_batch,
        device=device,
    )
    model.spatial_frontier_memory_score_weight = original_weight
    results["ablations"]["spatial_memory_disabled"] = {
        "input_transform": "identity",
        "candidate_score": summarize_explore_claim_predictions(
            rows,
            scores["candidate_score"],
        ),
    }

    results["phase_tables"] = {
        "online_frontier_marker": _phase_table(results["online_frontier_marker"]),
        **{
            name: _phase_table(
                item.get("spatial_frontier_memory_score", item["candidate_score"])
            )
            for name, item in results["ablations"].items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for name, table in results["phase_tables"].items():
        print(name)
        for phase, metrics in table.items():
            print(
                " ",
                phase,
                "prim",
                round(metrics["primitive_match_rate"], 3),
                "regret",
                round(metrics["sequence_regret"], 3),
                "marker",
                round(metrics["marker_seen_rate"], 3),
                "claim",
                round(metrics["claim_rate"], 3),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
