#!/usr/bin/env python3
"""Sweep action-selection rules over a trained frozen-feature candidate scorer."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import PosixPath
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.action_ranker import TaskAlignedCandidateScorer  # noqa: E402
from train_task_aligned_candidate_scorer import (  # noqa: E402
    _controls,
    _load_dataset,
    _predict,
    _selection_metrics,
)


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _evaluate_selection(
    dataset: dict,
    predictions: dict[str, torch.Tensor],
    *,
    progress_weight: float,
    collision_penalty: float,
    clearance_penalty: float,
    clearance_target_m: float,
    heading_weight: float,
    collision_threshold: float | None,
) -> dict[str, float]:
    score = (
        -progress_weight * predictions["progress"]
        + heading_weight * predictions["heading"]
        + collision_penalty * predictions["collision_probability"]
        + clearance_penalty * (clearance_target_m - predictions["clearance"]).clamp_min(0.0)
    )
    if collision_threshold is not None:
        unsafe = predictions["collision_probability"] > collision_threshold
        all_unsafe = unsafe.all(dim=1, keepdim=True)
        score = torch.where(unsafe & ~all_unsafe, score.new_full((), 1e6), score)
    return _selection_metrics(dataset, score.argmin(dim=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-space", choices=("raw", "proj", "spatial", "history"), default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-weights", default="1,2,4,8")
    parser.add_argument("--collision-penalties", default="1,2,4,8")
    parser.add_argument("--collision-thresholds", default="none,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--clearance-penalty", type=float, default=1.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--heading-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.serialization.add_safe_globals([PosixPath])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    latent_space = args.latent_space or str(checkpoint["latent_space"])
    train = _load_dataset(args.train_data, latent_space)
    evaluation = _load_dataset(args.eval_data, latent_space)
    head = TaskAlignedCandidateScorer(
        latent_dim=int(checkpoint["latent_dim"]),
        cmd_dim=int(checkpoint["cmd_dim"]),
        hidden=int(checkpoint["hidden"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    predictions = _predict(
        head,
        evaluation,
        checkpoint["regression_stats"],
        batch_size=args.batch_size,
        device=device,
    )
    controls = _controls(train, evaluation)
    progress_weights = _parse_floats(args.progress_weights)
    collision_penalties = _parse_floats(args.collision_penalties)
    thresholds: list[float | None] = [
        None if part.strip().lower() == "none" else float(part.strip())
        for part in args.collision_thresholds.split(",")
        if part.strip()
    ]
    results = []
    for progress_weight, collision_penalty, threshold in product(
        progress_weights,
        collision_penalties,
        thresholds,
    ):
        metrics = _evaluate_selection(
            evaluation,
            predictions,
            progress_weight=progress_weight,
            collision_penalty=collision_penalty,
            clearance_penalty=args.clearance_penalty,
            clearance_target_m=args.clearance_target_m,
            heading_weight=args.heading_weight,
            collision_threshold=threshold,
        )
        action_only = controls["action_only"]
        random = controls["random"]
        minimum = bool(
            metrics["mean_regret"] < action_only["mean_regret"]
            and metrics["selected_collision_rate"] < action_only["selected_collision_rate"]
            and metrics["mean_target_progress_m"] > action_only["mean_target_progress_m"]
        )
        promotion = bool(
            minimum
            and metrics["regret_ratio_vs_random"] <= 0.5
            and metrics["selected_collision_rate"] <= 0.05
            and metrics["mean_target_progress_m"] > random["mean_target_progress_m"]
        )
        results.append(
            {
                "progress_weight": progress_weight,
                "collision_penalty": collision_penalty,
                "collision_threshold": threshold,
                "metrics": metrics,
                "minimum_validity_passed": minimum,
                "promotion_gate_passed": promotion,
            }
        )
    best_by_score = min(
        results,
        key=lambda row: (
            row["metrics"]["mean_regret"]
            + row["metrics"]["selected_collision_rate"]
            - row["metrics"]["mean_target_progress_m"]
        ),
    )
    best_by_progress = max(
        results,
        key=lambda row: (
            row["metrics"]["mean_target_progress_m"],
            -row["metrics"]["selected_collision_rate"],
            -row["metrics"]["mean_regret"],
        ),
    )
    passing = [row for row in results if row["minimum_validity_passed"]]
    report = {
        "schema": "task_aligned_selection_grid_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "latent_space": latent_space,
        "eval_data": str(args.eval_data.resolve()),
        "controls": controls,
        "best_by_score": best_by_score,
        "best_by_progress": best_by_progress,
        "minimum_passing_count": len(passing),
        "promotion_passing_count": sum(row["promotion_gate_passed"] for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
