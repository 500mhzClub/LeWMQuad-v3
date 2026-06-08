#!/usr/bin/env python3
"""Train and gate a multi-task candidate scorer on frozen LeWM features."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.models.action_ranker import (  # noqa: E402
    TaskAlignedCandidateScorer,
    action_ranker_loss,
    task_aligned_candidate_loss,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_task_aligned_candidate_scorer")


def _load_dataset(path: Path, latent_space: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "schema": str(data["schema"]),
            "source_checkpoint": str(data["source_checkpoint"]),
            "primitive_names": [str(name) for name in data["primitive_names"]],
            "actions": torch.from_numpy(data["primitive_actions"]).float(),
            "start": torch.from_numpy(data[f"start_{latent_space}"]).float(),
            "goal": torch.from_numpy(data[f"goal_{latent_space}"]).float(),
            "goal_present": torch.from_numpy(data["goal_present"]).bool(),
            "collision": torch.from_numpy(data["collision"]).bool(),
            "progress": torch.from_numpy(data["progress"]).float(),
            "heading": torch.from_numpy(data["heading"]).float(),
            "clearance": torch.from_numpy(data["clearance"]).float(),
            "cost": torch.from_numpy(data["cost"]).float(),
            "scene_ids": [str(value) for value in data["scene_id"]],
            "families": [str(value) for value in data["family"]],
            "decision_types": [str(value) for value in data["decision_types"]],
            "logged_primitives": [str(value) for value in data["logged_primitive"]],
        }


def _regression_stats(train: dict) -> dict[str, tuple[float, float]]:
    goal_mask = train["goal_present"].unsqueeze(1).expand_as(train["progress"])
    stats = {}
    for name in ("progress", "heading"):
        values = train[name][goal_mask]
        stats[name] = (float(values.mean()), max(float(values.std()), 1e-6))
    values = train["clearance"]
    stats["clearance"] = (float(values.mean()), max(float(values.std()), 1e-6))
    return stats


def _normalize(value: torch.Tensor, stats: tuple[float, float]) -> torch.Tensor:
    mean, std = stats
    return (value - mean) / std


def _denormalize(value: torch.Tensor, stats: tuple[float, float]) -> torch.Tensor:
    mean, std = stats
    return value * std + mean


def _score_all(
    head: TaskAlignedCandidateScorer,
    dataset: dict,
    index: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    start = dataset["start"][index].to(device)
    goal = dataset["goal"][index].to(device)
    goal_present = dataset["goal_present"][index].to(device)
    actions = dataset["actions"].to(device)
    groups, latent_dim = start.shape
    primitives, cmd_dim = actions.shape
    return head(
        start[:, None, :].expand(groups, primitives, latent_dim),
        goal[:, None, :].expand(groups, primitives, latent_dim),
        goal_present[:, None].expand(groups, primitives),
        actions[None, :, :].expand(groups, primitives, cmd_dim),
    )


def _run_epoch(
    head: TaskAlignedCandidateScorer,
    optimizer: torch.optim.Optimizer | None,
    dataset: dict,
    stats: dict[str, tuple[float, float]],
    *,
    batch_size: int,
    collision_pos_weight: torch.Tensor,
    ranking_weight: float,
    ranking_temperature: float,
    progress_weight: float,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    heading_weight: float,
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    head.train(train)
    count = int(dataset["start"].shape[0])
    order = torch.randperm(count) if train else torch.arange(count)
    totals = {
        "loss": 0.0,
        "collision": 0.0,
        "progress": 0.0,
        "heading": 0.0,
        "clearance": 0.0,
        "ranking": 0.0,
    }
    for offset in range(0, count, batch_size):
        index = order[offset : offset + batch_size]
        predictions = _score_all(head, dataset, index, device)
        targets = {
            "collision": dataset["collision"][index].to(device),
            "progress": _normalize(dataset["progress"][index].to(device), stats["progress"]),
            "heading": _normalize(dataset["heading"][index].to(device), stats["heading"]),
            "clearance": _normalize(dataset["clearance"][index].to(device), stats["clearance"]),
        }
        loss, components = task_aligned_candidate_loss(
            predictions,
            targets,
            goal_present=dataset["goal_present"][index].to(device),
            collision_pos_weight=collision_pos_weight,
        )
        ranking = loss.new_zeros(())
        if ranking_weight > 0.0:
            goal_present = dataset["goal_present"][index].to(device).unsqueeze(1)
            predicted_progress = _denormalize(predictions["progress"], stats["progress"])
            predicted_heading = _denormalize(predictions["heading"], stats["heading"])
            predicted_clearance = _denormalize(
                predictions["clearance"],
                stats["clearance"],
            )
            predicted_task_cost = torch.where(
                goal_present,
                -progress_weight * predicted_progress + heading_weight * predicted_heading,
                torch.zeros_like(predicted_progress),
            )
            predicted_cost = (
                predicted_task_cost
                + collision_penalty * predictions["collision_logit"].sigmoid()
                + clearance_penalty
                * (clearance_target_m - predicted_clearance).clamp_min(0.0)
            )
            ranking = action_ranker_loss(
                predicted_cost,
                dataset["cost"][index].to(device),
                temperature=ranking_temperature,
                regression_weight=0.0,
            )
            loss = loss + ranking_weight * ranking
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_count = len(index)
        totals["loss"] += float(loss.detach()) * batch_count
        for name, value in components.items():
            totals[name] += float(value) * batch_count
        totals["ranking"] += float(ranking.detach()) * batch_count
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _predict(
    head: TaskAlignedCandidateScorer,
    dataset: dict,
    stats: dict[str, tuple[float, float]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    head.eval()
    outputs: dict[str, list[torch.Tensor]] = {
        "collision_probability": [],
        "progress": [],
        "heading": [],
        "clearance": [],
    }
    count = int(dataset["start"].shape[0])
    for offset in range(0, count, batch_size):
        index = torch.arange(offset, min(offset + batch_size, count))
        predictions = _score_all(head, dataset, index, device)
        outputs["collision_probability"].append(
            predictions["collision_logit"].sigmoid().cpu()
        )
        for name in ("progress", "heading", "clearance"):
            outputs[name].append(_denormalize(predictions[name], stats[name]).cpu())
    return {name: torch.cat(values) for name, values in outputs.items()}


def _selection_metrics(
    dataset: dict,
    selected_index: torch.Tensor,
) -> dict[str, float]:
    selected_index = selected_index.long().unsqueeze(1)
    selected_cost = dataset["cost"].gather(1, selected_index).squeeze(1)
    oracle_cost = dataset["cost"].min(dim=1).values
    random_cost = dataset["cost"].mean(dim=1)
    regret = selected_cost - oracle_cost
    random_regret = random_cost - oracle_cost
    selected_collision = dataset["collision"].gather(1, selected_index).squeeze(1)
    selected_progress = dataset["progress"].gather(1, selected_index).squeeze(1)
    goal_present = dataset["goal_present"]
    return {
        "mean_regret": float(regret.mean()),
        "mean_random_regret": float(random_regret.mean()),
        "regret_ratio_vs_random": float(regret.mean() / random_regret.mean().clamp_min(1e-8)),
        "optimal_rate": float((regret <= 1e-8).float().mean()),
        "selected_collision_rate": float(selected_collision.float().mean()),
        "mean_target_progress_m": float(selected_progress[goal_present].mean()),
        "target_progress_rows": int(goal_present.sum()),
    }


def _evaluate(
    head: TaskAlignedCandidateScorer,
    dataset: dict,
    stats: dict[str, tuple[float, float]],
    *,
    batch_size: int,
    progress_weight: float,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    heading_weight: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    predictions = _predict(head, dataset, stats, batch_size=batch_size, device=device)
    goal_present = dataset["goal_present"].unsqueeze(1)
    task_cost = torch.where(
        goal_present,
        -progress_weight * predictions["progress"] + heading_weight * predictions["heading"],
        torch.zeros_like(predictions["progress"]),
    )
    predicted_cost = (
        task_cost
        + collision_penalty * predictions["collision_probability"]
        + clearance_penalty
        * (clearance_target_m - predictions["clearance"]).clamp_min(0.0)
    )
    return _selection_metrics(dataset, predicted_cost.argmin(dim=1)), predictions


def _controls(train: dict, evaluation: dict) -> dict[str, dict[str, float] | str]:
    mean_train_cost = train["cost"].mean(dim=0)
    action_only_index = int(mean_train_cost.argmin())
    action_only = _selection_metrics(
        evaluation,
        torch.full((len(evaluation["scene_ids"]),), action_only_index),
    )
    name_to_index = {
        name: index for index, name in enumerate(evaluation["primitive_names"])
    }
    logged = _selection_metrics(
        evaluation,
        torch.tensor([name_to_index[name] for name in evaluation["logged_primitives"]]),
    )
    random = {
        "mean_regret": float(
            (evaluation["cost"].mean(dim=1) - evaluation["cost"].min(dim=1).values).mean()
        ),
        "selected_collision_rate": float(evaluation["collision"].float().mean()),
        "mean_target_progress_m": float(
            evaluation["progress"][evaluation["goal_present"]].mean()
        ),
    }
    return {
        "action_only_primitive": evaluation["primitive_names"][action_only_index],
        "action_only": action_only,
        "logged": logged,
        "random": random,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--latent-space",
        choices=("raw", "proj", "spatial", "history"),
        default="raw",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--progress-weight", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=3.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--clearance-penalty", type=float, default=0.5)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-temperature", type=float, default=0.1)
    parser.add_argument("--gate-max-regret-ratio", type=float, default=0.5)
    parser.add_argument("--gate-max-collision-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    train = _load_dataset(args.train_data, args.latent_space)
    evaluation = _load_dataset(args.eval_data, args.latent_space)
    if train["schema"] != "task_aligned_feature_dataset_v0" or evaluation["schema"] != train["schema"]:
        raise SystemExit("unsupported or mismatched task-aligned feature dataset schema")
    if train["primitive_names"] != evaluation["primitive_names"]:
        raise SystemExit("train/eval primitive order differs")
    if not torch.equal(train["actions"], evaluation["actions"]):
        raise SystemExit("train/eval primitive actions differ")
    stats = _regression_stats(train)
    positives = train["collision"].float().sum()
    negatives = train["collision"].numel() - positives
    collision_pos_weight = (negatives / positives.clamp_min(1.0)).to(device)

    head = TaskAlignedCandidateScorer(
        latent_dim=int(train["start"].shape[-1]),
        cmd_dim=int(train["actions"].shape[-1]),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    controls = _controls(train, evaluation)
    best_score = float("inf")
    best_epoch = -1
    best_metrics = None
    best_state = None
    for epoch in range(args.epochs):
        train_loss = _run_epoch(
            head,
            optimizer,
            train,
            stats,
            batch_size=args.batch_size,
            collision_pos_weight=collision_pos_weight,
            ranking_weight=args.ranking_weight,
            ranking_temperature=args.ranking_temperature,
            progress_weight=args.progress_weight,
            collision_penalty=args.collision_penalty,
            clearance_target_m=args.clearance_target_m,
            clearance_penalty=args.clearance_penalty,
            heading_weight=args.heading_weight,
            device=device,
        )
        metrics, _predictions = _evaluate(
            head,
            evaluation,
            stats,
            batch_size=args.batch_size,
            progress_weight=args.progress_weight,
            collision_penalty=args.collision_penalty,
            clearance_target_m=args.clearance_target_m,
            clearance_penalty=args.clearance_penalty,
            heading_weight=args.heading_weight,
            device=device,
        )
        selection_score = (
            metrics["mean_regret"]
            + metrics["selected_collision_rate"]
            - metrics["mean_target_progress_m"]
        )
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(
                "ep=%d loss=%.4f regret=%.4f collision=%.3f progress=%+.4f",
                epoch,
                train_loss["loss"],
                metrics["mean_regret"],
                metrics["selected_collision_rate"],
                metrics["mean_target_progress_m"],
            )

    assert best_state is not None and best_metrics is not None
    action_only = controls["action_only"]
    random = controls["random"]
    minimum_validity_passed = bool(
        best_metrics["mean_regret"] < action_only["mean_regret"]
        and best_metrics["selected_collision_rate"] < action_only["selected_collision_rate"]
        and best_metrics["mean_target_progress_m"] > action_only["mean_target_progress_m"]
    )
    promotion_gate_passed = bool(
        minimum_validity_passed
        and best_metrics["regret_ratio_vs_random"] <= args.gate_max_regret_ratio
        and best_metrics["selected_collision_rate"] <= args.gate_max_collision_rate
        and best_metrics["mean_target_progress_m"] > random["mean_target_progress_m"]
    )
    payload = {
        "head_state_dict": best_state,
        "latent_space": args.latent_space,
        "latent_dim": int(train["start"].shape[-1]),
        "cmd_dim": int(train["actions"].shape[-1]),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "primitive_names": train["primitive_names"],
        "primitive_actions": train["actions"],
        "source_checkpoint": train["source_checkpoint"],
        "regression_stats": stats,
        "best_epoch": best_epoch,
        "best_eval_metrics": best_metrics,
        "controls": controls,
        "minimum_validity_passed": minimum_validity_passed,
        "promotion_gate_passed": promotion_gate_passed,
        "config": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "schema": "task_aligned_candidate_scorer_report_v0",
        "checkpoint": str(args.output.resolve()),
        "seed": args.seed,
        "latent_space": args.latent_space,
        "best_epoch": best_epoch,
        "best_eval_metrics": best_metrics,
        "controls": controls,
        "minimum_validity_passed": minimum_validity_passed,
        "promotion_gate_passed": promotion_gate_passed,
        "eval_scene_count": len(set(evaluation["scene_ids"])),
        "eval_rows": len(evaluation["scene_ids"]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
