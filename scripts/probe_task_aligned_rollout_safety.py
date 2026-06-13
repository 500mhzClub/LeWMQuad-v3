#!/usr/bin/env python3
"""Test whether frozen LeWM rollouts improve task-aligned safety decoding."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.action_ranker import task_aligned_candidate_loss  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_task_aligned_rollout_safety")


class CandidateOutcomeProbe(nn.Module):
    """Multi-task probe over an already assembled candidate descriptor."""

    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.collision_head = nn.Linear(hidden // 2, 1)
        self.progress_head = nn.Linear(hidden // 2, 1)
        self.heading_head = nn.Linear(hidden // 2, 1)
        self.clearance_head = nn.Linear(hidden // 2, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "collision_logit": self.collision_head(hidden).squeeze(-1),
            "progress": self.progress_head(hidden).squeeze(-1),
            "heading": self.heading_head(hidden).squeeze(-1),
            "clearance": self.clearance_head(hidden).squeeze(-1),
        }


def _load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "schema": str(data["schema"]),
            "source_checkpoint": str(data["source_checkpoint"]),
            "primitive_names": [str(name) for name in data["primitive_names"]],
            "actions": torch.from_numpy(data["primitive_actions"]).float(),
            "start": torch.from_numpy(data["start_raw"]).float(),
            "goal": torch.from_numpy(data["goal_raw"]).float(),
            "goal_present": torch.from_numpy(data["goal_present"]).bool(),
            "collision": torch.from_numpy(data["collision"]).bool(),
            "progress": torch.from_numpy(data["progress"]).float(),
            "heading": torch.from_numpy(data["heading"]).float(),
            "clearance": torch.from_numpy(data["clearance"]).float(),
            "cost": torch.from_numpy(data["cost"]).float(),
            "scene_ids": [str(value) for value in data["scene_id"]],
            "logged_primitives": [str(value) for value in data["logged_primitive"]],
        }


def _sample_dataset(dataset: dict, max_rows: int, seed: int) -> dict:
    if max_rows <= 0 or len(dataset["scene_ids"]) <= max_rows:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(len(dataset["scene_ids"]), generator=generator)[:max_rows]
    sampled = {}
    row_count = len(dataset["scene_ids"])
    for key, value in dataset.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == row_count:
            sampled[key] = value[index]
        elif isinstance(value, list) and len(value) == row_count:
            sampled[key] = [value[i] for i in index.tolist()]
        else:
            sampled[key] = value
    return sampled


@torch.no_grad()
def _predict_rollouts(
    model,
    dataset: dict,
    *,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return raw predicted latents with shape (decision, primitive, horizon, dim)."""
    starts = dataset["start"]
    actions = dataset["actions"]
    groups, latent_dim = starts.shape
    primitives, cmd_dim = actions.shape
    outputs = []
    for offset in range(0, groups, batch_size):
        batch_start = starts[offset : offset + batch_size].to(device)
        count = batch_start.shape[0]
        flat_start = (
            batch_start[:, None, :]
            .expand(count, primitives, latent_dim)
            .reshape(count * primitives, latent_dim)
        )
        action_seq = (
            actions.to(device)[None, :, None, :]
            .expand(count, primitives, horizon, cmd_dim)
            .reshape(count * primitives, horizon, cmd_dim)
        )
        predicted = model.plan_rollout_raw(flat_start, action_seq)
        outputs.append(
            predicted.reshape(count, primitives, horizon, latent_dim).cpu().float()
        )
        if offset == 0 or offset + batch_size >= groups:
            logger.info("rollout rows %d/%d", min(offset + batch_size, groups), groups)
    return torch.cat(outputs, dim=0)


def _candidate_features(
    dataset: dict,
    rollouts: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Assemble the current-latent baseline plus optional rollout innovations."""
    start = dataset["start"]
    goal = dataset["goal"]
    actions = dataset["actions"]
    groups, latent_dim = start.shape
    primitives, cmd_dim = actions.shape
    start = start[:, None, :].expand(groups, primitives, latent_dim)
    goal = goal[:, None, :].expand(groups, primitives, latent_dim)
    features = [
        start,
        goal,
        start - goal,
        start * goal,
        dataset["goal_present"][:, None, None]
        .expand(groups, primitives, 1)
        .to(start.dtype),
        actions[None, :, :].expand(groups, primitives, cmd_dim),
    ]
    previous = start
    for step in range(horizon):
        predicted = rollouts[:, :, step, :]
        features.append(predicted - previous)
        previous = predicted
    return torch.cat(features, dim=-1)


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


def _binary_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probability = probability.flatten().float()
    target = target.flatten().bool()
    positives = int(target.sum())
    negatives = int((~target).sum())
    order = torch.argsort(probability, descending=True)
    sorted_target = target[order].float()
    true_positive = sorted_target.cumsum(0)
    false_positive = (1.0 - sorted_target).cumsum(0)
    recall = true_positive / max(positives, 1)
    precision = true_positive / torch.arange(
        1, len(sorted_target) + 1, dtype=torch.float32
    )
    average_precision = float(precision[target[order]].mean()) if positives else 0.0
    tpr = torch.cat((torch.zeros(1), recall, torch.ones(1)))
    fpr = torch.cat(
        (
            torch.zeros(1),
            false_positive / max(negatives, 1),
            torch.ones(1),
        )
    )
    return {
        "collision_auroc": float(torch.trapz(tpr, fpr)),
        "collision_average_precision": average_precision,
        "collision_brier": float((probability - target.float()).square().mean()),
    }


def _selection_metrics(dataset: dict, selected_index: torch.Tensor) -> dict[str, float]:
    selected_index = selected_index.long().unsqueeze(1)
    selected_cost = dataset["cost"].gather(1, selected_index).squeeze(1)
    oracle_cost = dataset["cost"].min(dim=1).values
    random_cost = dataset["cost"].mean(dim=1)
    regret = selected_cost - oracle_cost
    random_regret = random_cost - oracle_cost
    collision = dataset["collision"].gather(1, selected_index).squeeze(1)
    progress = dataset["progress"].gather(1, selected_index).squeeze(1)
    return {
        "mean_regret": float(regret.mean()),
        "regret_ratio_vs_random": float(regret.mean() / random_regret.mean().clamp_min(1e-8)),
        "optimal_rate": float((regret <= 1e-8).float().mean()),
        "selected_collision_rate": float(collision.float().mean()),
        "mean_target_progress_m": float(progress[dataset["goal_present"]].mean()),
    }


def _controls(train: dict, evaluation: dict) -> dict:
    action_index = int(train["cost"].mean(dim=0).argmin())
    action_only = _selection_metrics(
        evaluation,
        torch.full((len(evaluation["scene_ids"]),), action_index),
    )
    return {
        "action_only_primitive": evaluation["primitive_names"][action_index],
        "action_only": action_only,
        "random": {
            "selected_collision_rate": float(evaluation["collision"].float().mean()),
            "mean_target_progress_m": float(
                evaluation["progress"][evaluation["goal_present"]].mean()
            ),
        },
    }


def _run_epoch(
    probe: CandidateOutcomeProbe,
    features: torch.Tensor,
    dataset: dict,
    stats: dict[str, tuple[float, float]],
    *,
    optimizer: torch.optim.Optimizer | None,
    collision_pos_weight: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    training = optimizer is not None
    probe.train(training)
    order = torch.randperm(len(features)) if training else torch.arange(len(features))
    total = 0.0
    for offset in range(0, len(order), batch_size):
        index = order[offset : offset + batch_size]
        predictions = probe(features[index].to(device))
        targets = {
            "collision": dataset["collision"][index].to(device),
            "progress": _normalize(dataset["progress"][index].to(device), stats["progress"]),
            "heading": _normalize(dataset["heading"][index].to(device), stats["heading"]),
            "clearance": _normalize(dataset["clearance"][index].to(device), stats["clearance"]),
        }
        loss, _components = task_aligned_candidate_loss(
            predictions,
            targets,
            goal_present=dataset["goal_present"][index].to(device),
            collision_pos_weight=collision_pos_weight,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(index)
    return total / len(features)


@torch.no_grad()
def _evaluate(
    probe: CandidateOutcomeProbe,
    features: torch.Tensor,
    dataset: dict,
    stats: dict[str, tuple[float, float]],
    *,
    batch_size: int,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    progress_weight: float,
    heading_weight: float,
    device: torch.device,
) -> dict:
    probe.eval()
    values = {name: [] for name in ("collision", "progress", "heading", "clearance")}
    for offset in range(0, len(features), batch_size):
        predictions = probe(features[offset : offset + batch_size].to(device))
        values["collision"].append(predictions["collision_logit"].sigmoid().cpu())
        for name in ("progress", "heading", "clearance"):
            values[name].append(_denormalize(predictions[name], stats[name]).cpu())
    values = {name: torch.cat(parts) for name, parts in values.items()}
    goal_present = dataset["goal_present"].unsqueeze(1)
    task_cost = torch.where(
        goal_present,
        -progress_weight * values["progress"] + heading_weight * values["heading"],
        torch.zeros_like(values["progress"]),
    )
    predicted_cost = (
        task_cost
        + collision_penalty * values["collision"]
        + clearance_penalty * (clearance_target_m - values["clearance"]).clamp_min(0.0)
    )
    metrics = _selection_metrics(dataset, predicted_cost.argmin(dim=1))
    metrics.update(_binary_metrics(values["collision"], dataset["collision"]))
    clearance_error = values["clearance"] - dataset["clearance"]
    metrics["clearance_mae_m"] = float(clearance_error.abs().mean())
    metrics["clearance_rmse_m"] = float(clearance_error.square().mean().sqrt())
    return metrics


def _train_variant(
    name: str,
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
    train: dict,
    evaluation: dict,
    stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    torch.manual_seed(args.seed)
    probe = CandidateOutcomeProbe(
        input_dim=train_features.shape[-1],
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = train["collision"].float().sum()
    negatives = train["collision"].numel() - positives
    collision_pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
    best_selection_score = float("inf")
    best_selection_epoch = -1
    best_selection_metrics = None
    best_selection_state = None
    best_collision_ap = float("-inf")
    best_collision_epoch = -1
    best_collision_metrics = None
    best_clearance_mae = float("inf")
    best_clearance_epoch = -1
    best_clearance_metrics = None
    final_metrics = None
    for epoch in range(args.epochs):
        loss = _run_epoch(
            probe,
            train_features,
            train,
            stats,
            optimizer=optimizer,
            collision_pos_weight=collision_pos_weight,
            batch_size=args.batch_size,
            device=device,
        )
        metrics = _evaluate(
            probe,
            eval_features,
            evaluation,
            stats,
            batch_size=args.batch_size,
            collision_penalty=args.collision_penalty,
            clearance_target_m=args.clearance_target_m,
            clearance_penalty=args.clearance_penalty,
            progress_weight=args.progress_weight,
            heading_weight=args.heading_weight,
            device=device,
        )
        final_metrics = metrics
        selection_score = (
            metrics["mean_regret"]
            + metrics["selected_collision_rate"]
            - metrics["mean_target_progress_m"]
        )
        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_selection_epoch = epoch
            best_selection_metrics = metrics
            best_selection_state = copy.deepcopy(probe.state_dict())
        if metrics["collision_average_precision"] > best_collision_ap:
            best_collision_ap = metrics["collision_average_precision"]
            best_collision_epoch = epoch
            best_collision_metrics = metrics
        if metrics["clearance_mae_m"] < best_clearance_mae:
            best_clearance_mae = metrics["clearance_mae_m"]
            best_clearance_epoch = epoch
            best_clearance_metrics = metrics
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info(
                "%s ep=%d loss=%.4f regret_ratio=%.3f collision=%.3f AP=%.3f",
                name,
                epoch,
                loss,
                metrics["regret_ratio_vs_random"],
                metrics["selected_collision_rate"],
                metrics["collision_average_precision"],
            )
    assert (
        best_selection_metrics is not None
        and best_selection_state is not None
        and best_collision_metrics is not None
        and best_clearance_metrics is not None
        and final_metrics is not None
    )
    return {
        "best_selection_epoch": best_selection_epoch,
        "best_selection_metrics": best_selection_metrics,
        "best_collision_epoch": best_collision_epoch,
        "best_collision_metrics": best_collision_metrics,
        "best_clearance_epoch": best_clearance_epoch,
        "best_clearance_metrics": best_clearance_metrics,
        "final_epoch_metrics": final_metrics,
    }, best_selection_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--progress-weight", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=3.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--clearance-penalty", type=float, default=0.5)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    train = _sample_dataset(_load_dataset(args.train_data), args.max_train_rows, args.seed)
    evaluation = _sample_dataset(
        _load_dataset(args.eval_data), args.max_eval_rows, args.seed + 1
    )
    if train["schema"] != "task_aligned_feature_dataset_v0":
        raise SystemExit("unsupported task-aligned feature dataset schema")
    if train["primitive_names"] != evaluation["primitive_names"]:
        raise SystemExit("train/eval primitive order differs")
    if train["source_checkpoint"] != str(args.checkpoint.resolve()):
        raise SystemExit("train features and requested checkpoint differ")
    if evaluation["source_checkpoint"] != train["source_checkpoint"]:
        raise SystemExit("train/eval source checkpoints differ")

    model, _config = load_model(
        SimpleNamespace(
            checkpoint=args.checkpoint.resolve(),
            max_seq_len=None,
            sigreg_lambda=None,
        ),
        device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info(
        "predicting horizon-%d rollouts for train=%d eval=%d on %s",
        args.horizon,
        len(train["scene_ids"]),
        len(evaluation["scene_ids"]),
        device,
    )
    train_rollouts = _predict_rollouts(
        model,
        train,
        horizon=args.horizon,
        batch_size=args.rollout_batch_size,
        device=device,
    )
    eval_rollouts = _predict_rollouts(
        model,
        evaluation,
        horizon=args.horizon,
        batch_size=args.rollout_batch_size,
        device=device,
    )
    del model

    stats = _regression_stats(train)
    controls = _controls(train, evaluation)
    reports = {}
    states = {}
    for horizon in range(args.horizon + 1):
        name = "current_raw" if horizon == 0 else f"rollout_raw_h{horizon}"
        logger.info("training %s", name)
        train_features = _candidate_features(train, train_rollouts, horizon)
        eval_features = _candidate_features(evaluation, eval_rollouts, horizon)
        reports[name], states[name] = _train_variant(
            name,
            train_features,
            eval_features,
            train,
            evaluation,
            stats,
            args,
            device,
        )
        del train_features, eval_features

    baseline = reports["current_raw"]
    comparisons = {}
    for name, report in reports.items():
        collision_metrics = report["best_collision_metrics"]
        clearance_metrics = report["best_clearance_metrics"]
        selection_metrics = report["best_selection_metrics"]
        comparisons[name] = {
            "collision_auroc_delta_vs_current": (
                collision_metrics["collision_auroc"]
                - baseline["best_collision_metrics"]["collision_auroc"]
            ),
            "collision_ap_delta_vs_current": (
                collision_metrics["collision_average_precision"]
                - baseline["best_collision_metrics"]["collision_average_precision"]
            ),
            "clearance_mae_delta_vs_current_m": (
                clearance_metrics["clearance_mae_m"]
                - baseline["best_clearance_metrics"]["clearance_mae_m"]
            ),
            "selected_collision_delta_vs_current": (
                selection_metrics["selected_collision_rate"]
                - baseline["best_selection_metrics"]["selected_collision_rate"]
            ),
            "regret_ratio_delta_vs_current": (
                selection_metrics["regret_ratio_vs_random"]
                - baseline["best_selection_metrics"]["regret_ratio_vs_random"]
            ),
        }
    final_name = f"rollout_raw_h{args.horizon}"
    final = reports[final_name]
    rollout_decodability_passed = bool(
        final["best_collision_metrics"]["collision_average_precision"]
        >= baseline["best_collision_metrics"]["collision_average_precision"] + 0.02
        and final["best_clearance_metrics"]["clearance_mae_m"]
        <= baseline["best_clearance_metrics"]["clearance_mae_m"] - 0.005
    )
    final_selection = final["best_selection_metrics"]
    promotion_gate_passed = bool(
        final_selection["regret_ratio_vs_random"] <= 0.5
        and final_selection["selected_collision_rate"] <= 0.05
        and final_selection["mean_target_progress_m"]
        > controls["random"]["mean_target_progress_m"]
    )
    report = {
        "schema": "task_aligned_rollout_safety_probe_report_v0",
        "source_checkpoint": train["source_checkpoint"],
        "train_data": str(args.train_data.resolve()),
        "eval_data": str(args.eval_data.resolve()),
        "train_rows": len(train["scene_ids"]),
        "eval_rows": len(evaluation["scene_ids"]),
        "train_scene_count": len(set(train["scene_ids"])),
        "eval_scene_count": len(set(evaluation["scene_ids"])),
        "horizon": args.horizon,
        "variants": reports,
        "comparisons": comparisons,
        "controls": controls,
        "rollout_decodability_gate": {
            "requires_collision_ap_delta": 0.02,
            "requires_clearance_mae_improvement_m": 0.005,
            "passed": rollout_decodability_passed,
        },
        "promotion_gate": {
            "requires_regret_ratio_at_most": 0.5,
            "requires_collision_rate_at_most": 0.05,
            "requires_progress_above_random": controls["random"]["mean_target_progress_m"],
            "passed": promotion_gate_passed,
        },
        "limitations": [
            "collision labels are the existing unvalidated inflated-grid proxy",
            "rollout features are deterministic transforms of current raw latent plus action",
            "candidate labels describe one primitive block even for horizon-two features",
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": report["schema"],
            "variant_state_dicts": states,
            "report": report,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
