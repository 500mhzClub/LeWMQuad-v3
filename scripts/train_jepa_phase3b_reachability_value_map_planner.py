#!/usr/bin/env python3
"""Train a Phase 3B value-map planner conditioned on reachability features."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    ACTION_NAMES,
    read_jsonl,
)
from lewm.models.phase3b_reachability import (  # noqa: E402
    Phase3BReachabilityConditionedValueMapPlannerHead,
    Phase3BReachabilityHead,
    reachability_feature_tensor,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _egocentric_memory_tensor_to_dict,
    _infer_scene_seed,
    _select_egocentric_learned_value_map_action,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_action import (  # noqa: E402
    _load_value_extractor_head,
)
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _build_examples as _build_memory_examples,
    _load_latent_memory_updater,
    _load_value_field_head,
)
from scripts.train_jepa_phase3a_value_map_planner import (  # noqa: E402
    ValueMapExamples,
    _build_value_map_examples,
    _filter_value_map_examples,
    _planner_action_logits,
)


@dataclass(frozen=True)
class ConditionedValueMapExamples:
    memories: torch.Tensor
    target_fields: torch.Tensor
    sparse_probabilities: torch.Tensor
    reachability_features: torch.Tensor
    value_maps: torch.Tensor
    actions: torch.Tensor
    sparse_labels: torch.Tensor


def _copy_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
        "radius": int(memory.get("radius", 0)),
    }


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def _load_reachability_head(
    path: Path,
    *,
    fallback_memory_size: int,
    device: torch.device,
) -> tuple[Phase3BReachabilityHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    head = Phase3BReachabilityHead(
        memory_size=int(config.get("memory_size", fallback_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        memory_channels=int(config.get("memory_channels", 3)),
        architecture=str(config.get("architecture", "conv")),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    return head, report


@torch.no_grad()
def _condition_examples(
    examples: ValueMapExamples,
    *,
    reachability_head: Phase3BReachabilityHead,
    batch_size: int,
    device: torch.device,
) -> ConditionedValueMapExamples:
    loader = DataLoader(
        TensorDataset(examples.memories),
        batch_size=batch_size,
        shuffle=False,
    )
    features = []
    for (memory,) in loader:
        memory = memory.to(device)
        predictions = reachability_head(memory)
        features.append(
            reachability_feature_tensor(
                predictions,
                memory_size=int(reachability_head.memory_size),
            )
            .detach()
            .cpu()
        )
    return ConditionedValueMapExamples(
        memories=examples.memories,
        target_fields=examples.target_fields,
        sparse_probabilities=examples.sparse_probabilities,
        reachability_features=torch.cat(features, dim=0),
        value_maps=examples.value_maps,
        actions=examples.actions,
        sparse_labels=examples.sparse_labels,
    )


@torch.no_grad()
def _evaluate(
    planner_head: Phase3BReachabilityConditionedValueMapPlannerHead,
    examples: ConditionedValueMapExamples,
    *,
    batch_size: int,
    positive_weight: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    device: torch.device,
) -> dict:
    planner_head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.target_fields,
        examples.sparse_probabilities,
        examples.reachability_features,
        examples.value_maps,
        examples.actions,
        examples.sparse_labels,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    mse_total = 0.0
    matches = 0
    sparse_total = 0
    sparse_matches = 0
    broad_total = 0
    broad_matches = 0
    action_counts = {name: 0 for name in ACTION_NAMES}
    predicted_counts = {name: 0 for name in ACTION_NAMES}
    for batch in loader:
        (
            memory,
            target,
            sparse_prob,
            reachability,
            value_map,
            action,
            sparse_label,
        ) = batch
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        reachability = reachability.to(device)
        value_map = value_map.to(device)
        logits = planner_head(memory, target, sparse_prob, reachability)
        weight = 1.0 + float(positive_weight) * value_map
        loss = F.binary_cross_entropy_with_logits(logits, value_map, weight=weight)
        probs = logits.sigmoid().detach().cpu()
        memory_cpu = memory.detach().cpu()
        action_cpu = action.detach().cpu()
        sparse_cpu = sparse_label.detach().cpu()
        loss_total += float(loss.item()) * int(memory.shape[0])
        mse_total += float(F.mse_loss(probs, value_map.detach().cpu()).item()) * int(
            memory.shape[0]
        )
        total += int(memory.shape[0])
        batch_matches = []
        for item_index in range(int(memory_cpu.shape[0])):
            memory_dict = _egocentric_memory_tensor_to_dict(
                memory_cpu[item_index],
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            predicted_action, _mode = _select_egocentric_learned_value_map_action(
                _copy_memory(memory_dict),
                probs[item_index, 0],
            )
            expected_action = ACTION_NAMES[int(action_cpu[item_index].item())]
            match = predicted_action == expected_action
            batch_matches.append(match)
            action_counts[expected_action] += 1
            predicted_counts[predicted_action] += 1
        match_tensor = torch.tensor(batch_matches, dtype=torch.bool)
        matches += int(match_tensor.sum().item())
        sparse_total += int(sparse_cpu.sum().item())
        sparse_matches += int((match_tensor & sparse_cpu).sum().item())
        broad_mask = ~sparse_cpu
        broad_total += int(broad_mask.sum().item())
        broad_matches += int((match_tensor & broad_mask).sum().item())
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "mse": mse_total / max(total, 1),
        "action_match": matches / max(total, 1),
        "sparse_examples": sparse_total,
        "sparse_action_match": sparse_matches / max(sparse_total, 1),
        "broad_examples": broad_total,
        "broad_action_match": broad_matches / max(broad_total, 1),
        "action_counts": action_counts,
        "predicted_action_counts": predicted_counts,
    }


def _build_conditioned_value_examples(
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_episodes: int | None,
    max_steps: int,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    reachability_head: Phase3BReachabilityHead,
    thresholds: dict[str, float],
    target_top_k: int,
    sparse_target_top_k: int,
    value_iterations: int,
    value_gamma: float,
    explicit_frontier_targets: bool,
    fixed_marker_targets: bool,
    include_marker_start_groups: bool,
    batch_size: int,
    device: torch.device,
) -> ConditionedValueMapExamples:
    memory_examples = _build_memory_examples(
        rows,
        scene_seed=scene_seed,
        width=width,
        height=height,
        view_size=view_size,
        memory_size=memory_size,
        max_episodes=max_episodes,
        max_steps=max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        blocked_threshold=thresholds["blocked"],
        free_threshold=thresholds["free"],
        marker_threshold=thresholds["marker"],
        output_channels=1,
        include_marker_start_groups=include_marker_start_groups,
        rollout_value_field_head=None,
        rollout_target_threshold=thresholds["target"],
        rollout_target_top_k=target_top_k,
        rollout_fixed_marker_target=False,
        device=device,
    )
    value_examples = _build_value_map_examples(
        memory_examples.memories,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        blocked_threshold=thresholds["blocked"],
        free_threshold=thresholds["free"],
        marker_threshold=thresholds["marker"],
        target_threshold=thresholds["target"],
        target_top_k=target_top_k,
        extractor_threshold=thresholds["extractor"],
        sparse_target_top_k=sparse_target_top_k,
        explicit_frontier_targets=explicit_frontier_targets,
        fixed_marker_targets=fixed_marker_targets,
        value_iterations=value_iterations,
        value_gamma=value_gamma,
        batch_size=batch_size,
        device=device,
    )
    return _condition_examples(
        value_examples,
        reachability_head=reachability_head,
        batch_size=batch_size,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
    parser.add_argument("--latent-value-field-head", type=Path, required=True)
    parser.add_argument("--latent-value-extractor-head", type=Path, required=True)
    parser.add_argument("--reachability-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--include-marker-start-train-groups", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--architecture", choices=("conv", "dilated"), default="conv")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--value-iterations", type=int, default=64)
    parser.add_argument("--value-gamma", type=float, default=0.94)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--sparse-target-top-k", type=int, default=1)
    parser.add_argument("--explicit-frontier-targets", action="store_true")
    parser.add_argument("--fixed-marker-targets", action="store_true")
    parser.add_argument(
        "--example-filter",
        choices=("all", "broad", "sparse"),
        default="all",
    )
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--save-best-metric",
        choices=(
            "action_match",
            "broad_action_match",
            "sparse_action_match",
            "loss",
            "mse",
        ),
        default="action_match",
    )
    parser.add_argument("--seed", type=int, default=20260661)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if args.positive_weight <= 0.0:
        raise SystemExit("--positive-weight must be positive")
    if args.action_loss_weight < 0.0:
        raise SystemExit("--action-loss-weight must be non-negative")
    if args.value_iterations < 1:
        raise SystemExit("--value-iterations must be positive")
    if not 0.0 < args.value_gamma <= 1.0:
        raise SystemExit("--value-gamma must be in (0, 1]")
    if not 0.0 <= args.target_threshold <= 1.0:
        raise SystemExit("--target-threshold must be in [0, 1]")
    if args.target_top_k < 1:
        raise SystemExit("--target-top-k must be positive")
    if not 0.0 <= args.extractor_threshold <= 1.0:
        raise SystemExit("--extractor-threshold must be in [0, 1]")
    if args.sparse_target_top_k < 1:
        raise SystemExit("--sparse-target-top-k must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_seed = _infer_scene_seed(args.train_data)
    validation_seed = _infer_scene_seed(args.validation_data)
    if train_seed is None or validation_seed is None:
        raise SystemExit("could not infer train/validation scene seeds")
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)
    base_model, base_report = load_model(args.base_checkpoint, device=device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    latent_map_head, latent_map_report = _load_latent_map_head(
        args.latent_map_head,
        base_model=base_model,
        device=device,
    )
    latent_memory_updater, latent_memory_report = _load_latent_memory_updater(
        args.latent_memory_updater,
        model=base_model,
        device=device,
    )
    value_field_head, value_field_report = _load_value_field_head(
        args.latent_value_field_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    extractor_head, extractor_report = _load_value_extractor_head(
        args.latent_value_extractor_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    reachability_head, reachability_report = _load_reachability_head(
        args.reachability_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    if int(latent_memory_updater.memory_size) != int(args.memory_size):
        raise SystemExit(
            "--memory-size must match latent memory updater size "
            f"({latent_memory_updater.memory_size})"
        )
    if int(reachability_head.memory_size) != int(args.memory_size):
        raise SystemExit(
            "--memory-size must match reachability head size "
            f"({reachability_head.memory_size})"
        )
    thresholds = {
        "blocked": float(args.latent_memory_blocked_threshold),
        "free": float(args.latent_memory_free_threshold),
        "marker": float(args.latent_memory_marker_threshold),
        "target": float(args.target_threshold),
        "extractor": float(args.extractor_threshold),
    }
    train_examples = _build_conditioned_value_examples(
        train_rows,
        scene_seed=train_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_train_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        reachability_head=reachability_head,
        thresholds=thresholds,
        target_top_k=args.target_top_k,
        sparse_target_top_k=args.sparse_target_top_k,
        value_iterations=args.value_iterations,
        value_gamma=args.value_gamma,
        explicit_frontier_targets=bool(args.explicit_frontier_targets),
        fixed_marker_targets=bool(args.fixed_marker_targets),
        include_marker_start_groups=bool(args.include_marker_start_train_groups),
        batch_size=args.batch_size,
        device=device,
    )
    train_examples = _condition_examples(
        _filter_value_map_examples(
            ValueMapExamples(
                memories=train_examples.memories,
                target_fields=train_examples.target_fields,
                sparse_probabilities=train_examples.sparse_probabilities,
                value_maps=train_examples.value_maps,
                actions=train_examples.actions,
                sparse_labels=train_examples.sparse_labels,
            ),
            mode=args.example_filter,
        ),
        reachability_head=reachability_head,
        batch_size=args.batch_size,
        device=device,
    )
    validation_examples = _build_conditioned_value_examples(
        validation_rows,
        scene_seed=validation_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_validation_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        reachability_head=reachability_head,
        thresholds=thresholds,
        target_top_k=args.target_top_k,
        sparse_target_top_k=args.sparse_target_top_k,
        value_iterations=args.value_iterations,
        value_gamma=args.value_gamma,
        explicit_frontier_targets=bool(args.explicit_frontier_targets),
        fixed_marker_targets=bool(args.fixed_marker_targets),
        include_marker_start_groups=False,
        batch_size=args.batch_size,
        device=device,
    )
    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.target_fields,
        train_examples.sparse_probabilities,
        train_examples.reachability_features,
        train_examples.value_maps,
        train_examples.actions,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    planner_head = Phase3BReachabilityConditionedValueMapPlannerHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
        reachability_channels=4,
        architecture=args.architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        planner_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = None
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, target, sparse_prob, reachability, value_map, action = next(
                iterator
            )
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, sparse_prob, reachability, value_map, action = next(
                iterator
            )
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        reachability = reachability.to(device)
        value_map = value_map.to(device)
        action = action.to(device)
        logits = planner_head(memory, target, sparse_prob, reachability)
        weight = 1.0 + float(args.positive_weight) * value_map
        loss = F.binary_cross_entropy_with_logits(logits, value_map, weight=weight)
        if args.action_loss_weight > 0.0:
            loss = loss + float(args.action_loss_weight) * F.cross_entropy(
                _planner_action_logits(logits),
                action,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                planner_head,
                validation_examples,
                batch_size=args.batch_size,
                positive_weight=args.positive_weight,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            metric_value = float(metrics[args.save_best_metric])
            if args.save_best_metric in {"loss", "mse"}:
                score = -metric_value
            else:
                score = metric_value
            if args.save_best and (best_score is None or score > best_score):
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in planner_head.state_dict().items()
                }

    final_validation = _evaluate(
        planner_head,
        validation_examples,
        batch_size=args.batch_size,
        positive_weight=args.positive_weight,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        planner_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation
    report = {
        "schema": "jepa_phase3b_reachability_value_map_planner_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "latent_value_field_head": str(args.latent_value_field_head.resolve()),
        "latent_value_field_completed_steps": value_field_report.get("completed_steps"),
        "latent_value_extractor_head": str(args.latent_value_extractor_head.resolve()),
        "latent_value_extractor_completed_steps": extractor_report.get(
            "completed_steps"
        ),
        "reachability_head": str(args.reachability_head.resolve()),
        "reachability_completed_steps": reachability_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "validation_examples": int(len(validation_examples.actions)),
        "completed_steps": args.optimization_steps,
        "final_validation": final_validation,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "logs": logs,
        "args": {key: _json_safe_arg(value) for key, value in vars(args).items()},
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
            "reachability_channels": 4,
            "architecture": args.architecture,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "planner_head_state_dict": planner_head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
