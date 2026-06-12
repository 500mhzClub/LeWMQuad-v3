#!/usr/bin/env python3
"""Evaluate random, logged-action, and train-set action-prior controls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def _candidate_map(row: dict) -> dict[str, dict]:
    return {
        candidate["primitive_name"]: candidate
        for candidate in row["counterfactual_candidates"]
    }


def _metrics(rows: list[dict], selected_names: list[str]) -> dict[str, float]:
    regret_sum = 0.0
    collision_sum = 0
    optimal_sum = 0
    progress_sum = 0.0
    progress_count = 0
    for row, selected_name in zip(rows, selected_names, strict=True):
        selected = _candidate_map(row)[selected_name]
        best_cost = float(row["counterfactual_best_cost"])
        regret = float(selected["cost"]) - best_cost
        regret_sum += regret
        collision_sum += bool(selected["collided"])
        optimal_sum += regret <= 1e-8
        progress = selected["target_progress_m"]
        if progress is not None:
            progress_sum += float(progress)
            progress_count += 1
    count = len(rows)
    return {
        "mean_regret": regret_sum / count,
        "optimal_rate": optimal_sum / count,
        "selected_collision_rate": collision_sum / count,
        "mean_target_progress_m": progress_sum / max(progress_count, 1),
        "target_progress_rows": progress_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_rows = _load_rows(args.train)
    eval_rows = _load_rows(args.eval)
    if not train_rows or not eval_rows:
        raise SystemExit("train and eval scored indexes must be non-empty")

    cost_sums: dict[str, float] = defaultdict(float)
    cost_counts: dict[str, int] = defaultdict(int)
    for row in train_rows:
        for candidate in row["counterfactual_candidates"]:
            name = str(candidate["primitive_name"])
            cost_sums[name] += float(candidate["cost"])
            cost_counts[name] += 1
    mean_train_cost = {
        name: cost_sums[name] / cost_counts[name] for name in sorted(cost_sums)
    }
    prior_primitive = min(mean_train_cost, key=lambda name: (mean_train_cost[name], name))

    random_regret = 0.0
    random_collision = 0.0
    random_progress = 0.0
    random_progress_rows = 0
    for row in eval_rows:
        candidates = row["counterfactual_candidates"]
        best_cost = float(row["counterfactual_best_cost"])
        random_regret += sum(float(item["cost"]) - best_cost for item in candidates) / len(
            candidates
        )
        random_collision += sum(bool(item["collided"]) for item in candidates) / len(
            candidates
        )
        progresses = [
            float(item["target_progress_m"])
            for item in candidates
            if item["target_progress_m"] is not None
        ]
        if progresses:
            random_progress += sum(progresses) / len(progresses)
            random_progress_rows += 1

    logged_names = [str(row["primitive_name"]) for row in eval_rows]
    prior_names = [prior_primitive] * len(eval_rows)
    payload = {
        "schema": "task_aligned_counterfactual_baselines_v0",
        "train": str(args.train),
        "eval": str(args.eval),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "mean_train_cost_by_primitive": mean_train_cost,
        "action_only_prior_primitive": prior_primitive,
        "action_only_prior": _metrics(eval_rows, prior_names),
        "logged_action": _metrics(eval_rows, logged_names),
        "random_action": {
            "mean_regret": random_regret / len(eval_rows),
            "selected_collision_rate": random_collision / len(eval_rows),
            "mean_target_progress_m": random_progress / max(random_progress_rows, 1),
            "target_progress_rows": random_progress_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
