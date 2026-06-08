#!/usr/bin/env python3
"""Train and gate a goal-conditioned first-action ranker on frozen LeWM features."""
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
    GoalActionRanker,
    action_ranker_loss,
    first_action_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_first_action_ranker")


def _load_dataset(
    path: Path,
    *,
    latent_space: str,
) -> dict[str, torch.Tensor | list[str] | str]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "schema": str(data["schema"]),
            "source_checkpoint": str(data["source_checkpoint"]),
            "primitive_names": [str(name) for name in data["primitive_names"]],
            "scene_ids": [str(scene_id) for scene_id in data["scene_id"]],
            "actions": torch.from_numpy(data["primitive_actions"]).float(),
            "start": torch.from_numpy(data[f"start_{latent_space}"]).float(),
            "goal": torch.from_numpy(data[f"goal_{latent_space}"]).float(),
            "distance": torch.from_numpy(data["after_distance_m"]).float(),
            "collision": torch.from_numpy(data["collision"]).bool(),
        }


def _score_all(
    head: GoalActionRanker,
    start: torch.Tensor,
    goal: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    groups, latent_dim = start.shape
    primitives, cmd_dim = actions.shape
    return head(
        start[:, None, :].expand(groups, primitives, latent_dim),
        goal[:, None, :].expand(groups, primitives, latent_dim),
        actions[None, :, :].expand(groups, primitives, cmd_dim),
    )


def _run_epoch(
    head: GoalActionRanker,
    optimizer: torch.optim.Optimizer | None,
    dataset: dict,
    *,
    batch_size: int,
    collision_penalty: float,
    temperature: float,
    regression_weight: float,
    device: torch.device,
) -> float:
    train = optimizer is not None
    head.train(train)
    count = int(dataset["start"].shape[0])
    order = torch.randperm(count) if train else torch.arange(count)
    actions = dataset["actions"].to(device)
    total_loss = 0.0
    for offset in range(0, count, batch_size):
        index = order[offset : offset + batch_size]
        start = dataset["start"][index].to(device)
        goal = dataset["goal"][index].to(device)
        distance = dataset["distance"][index].to(device)
        collision = dataset["collision"][index].to(device)
        scores = _score_all(head, start, goal, actions)
        target_cost = distance + collision_penalty * collision.float()
        loss = action_ranker_loss(
            scores,
            target_cost,
            temperature=temperature,
            regression_weight=regression_weight,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * len(index)
    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate(head: GoalActionRanker, dataset: dict, device: torch.device) -> dict[str, float]:
    head.eval()
    scores = _score_all(
        head,
        dataset["start"].to(device),
        dataset["goal"].to(device),
        dataset["actions"].to(device),
    )
    return first_action_metrics(
        scores,
        dataset["distance"].to(device),
        dataset["collision"].to(device),
    )


@torch.no_grad()
def _action_only_baseline(
    train: dict,
    evaluation: dict,
    *,
    collision_penalty: float,
) -> dict[str, float]:
    """Score every state with the same train-set mean cost per primitive."""
    train_cost = train["distance"] + collision_penalty * train["collision"].float()
    action_scores = train_cost.mean(dim=0, keepdim=True)
    scores = action_scores.expand(evaluation["distance"].shape[0], -1)
    return first_action_metrics(scores, evaluation["distance"], evaluation["collision"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-space", choices=("raw", "proj"), default="proj")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument("--collision-penalty", type=float, default=1.0)
    parser.add_argument("--gate-regret-ratio", type=float, default=0.5)
    parser.add_argument("--gate-max-collision-rate", type=float, default=0.05)
    parser.add_argument("--min-eval-scenes", type=int, default=32)
    parser.add_argument("--min-eval-groups", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train = _load_dataset(args.train_data, latent_space=args.latent_space)
    evaluation = _load_dataset(args.eval_data, latent_space=args.latent_space)
    if train["schema"] != "first_action_dataset_v0" or evaluation["schema"] != train["schema"]:
        raise SystemExit("unsupported or mismatched first-action dataset schema")
    if train["primitive_names"] != evaluation["primitive_names"]:
        raise SystemExit("train/eval primitive order differs")
    if not torch.equal(train["actions"], evaluation["actions"]):
        raise SystemExit("train/eval primitive actions differ")

    latent_dim = int(train["start"].shape[-1])
    cmd_dim = int(train["actions"].shape[-1])
    head = GoalActionRanker(
        latent_dim=latent_dim,
        cmd_dim=cmd_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_regret = float("inf")
    best_epoch = -1
    best_state = None
    best_metrics = None
    for epoch in range(args.epochs):
        train_loss = _run_epoch(
            head,
            optimizer,
            train,
            batch_size=args.batch_size,
            collision_penalty=args.collision_penalty,
            temperature=args.temperature,
            regression_weight=args.regression_weight,
            device=device,
        )
        eval_loss = _run_epoch(
            head,
            None,
            evaluation,
            batch_size=args.batch_size,
            collision_penalty=args.collision_penalty,
            temperature=args.temperature,
            regression_weight=args.regression_weight,
            device=device,
        )
        metrics = _evaluate(head, evaluation, device)
        if metrics["mean_first_regret_m"] < best_regret:
            best_regret = metrics["mean_first_regret_m"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(
                "ep=%d train_loss=%.4f eval_loss=%.4f first_rho=%+.3f regret=%.4fm ratio=%.3f",
                epoch,
                train_loss,
                eval_loss,
                metrics["mean_first_spearman"],
                metrics["mean_first_regret_m"],
                metrics["regret_ratio_vs_random"],
            )

    assert best_state is not None and best_metrics is not None
    action_only_metrics = _action_only_baseline(
        train,
        evaluation,
        collision_penalty=args.collision_penalty,
    )
    eval_scene_count = len(set(evaluation["scene_ids"]))
    eval_group_count = len(evaluation["scene_ids"])
    gate_passed = bool(
        best_metrics["regret_ratio_vs_random"] <= args.gate_regret_ratio
        and best_metrics["mean_first_spearman"] > 0.0
        and best_metrics["selected_collision_rate"] <= args.gate_max_collision_rate
        and best_metrics["mean_first_regret_m"] < action_only_metrics["mean_first_regret_m"]
        and eval_scene_count >= args.min_eval_scenes
        and eval_group_count >= args.min_eval_groups
    )
    payload = {
        "head_state_dict": best_state,
        "latent_dim": latent_dim,
        "cmd_dim": cmd_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "primitive_names": train["primitive_names"],
        "primitive_actions": train["actions"],
        "source_checkpoint": train["source_checkpoint"],
        "latent_space": args.latent_space,
        "train_data": str(args.train_data.resolve()),
        "eval_data": str(args.eval_data.resolve()),
        "best_epoch": best_epoch,
        "best_eval_metrics": best_metrics,
        "action_only_baseline_metrics": action_only_metrics,
        "gate_regret_ratio": args.gate_regret_ratio,
        "gate_max_collision_rate": args.gate_max_collision_rate,
        "min_eval_scenes": args.min_eval_scenes,
        "min_eval_groups": args.min_eval_groups,
        "eval_scene_count": eval_scene_count,
        "eval_group_count": eval_group_count,
        "gate_passed": gate_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "schema": "first_action_ranker_report_v0",
        "checkpoint": str(args.output.resolve()),
        "latent_space": args.latent_space,
        "best_epoch": best_epoch,
        "best_eval_metrics": best_metrics,
        "action_only_baseline_metrics": action_only_metrics,
        "gate_regret_ratio": args.gate_regret_ratio,
        "gate_max_collision_rate": args.gate_max_collision_rate,
        "min_eval_scenes": args.min_eval_scenes,
        "min_eval_groups": args.min_eval_groups,
        "eval_scene_count": eval_scene_count,
        "eval_group_count": eval_group_count,
        "gate_passed": gate_passed,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
