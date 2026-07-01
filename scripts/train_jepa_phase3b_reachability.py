#!/usr/bin/env python3
"""Train a Phase 3B reachability head from recurrent egocentric memory."""
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
    ACTION_INDEX,
    read_jsonl,
    step_state,
)
from lewm.benchmarks.phase3b_reachability import (  # noqa: E402
    Phase3BReachabilityTargetBatch,
    build_reachability_target,
    reachability_prediction_losses,
    stack_reachability_targets,
)
from lewm.models.phase3b_reachability import Phase3BReachabilityHead  # noqa: E402
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _infer_scene_seed,
    _latent_local_evidence,
    _select_egocentric_learned_value_map_action,
    _select_egocentric_value_field_action,
    _state_from_dict,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _groups_from_rows,
    _load_latent_memory_updater,
)


@dataclass(frozen=True)
class ReachabilityExamples:
    memories: torch.Tensor
    targets: Phase3BReachabilityTargetBatch
    actions: torch.Tensor


def _copy_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
        "radius": int(memory.get("radius", 0)),
    }


@torch.no_grad()
def _build_examples(
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
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    target_mode: str,
    target_gamma: float,
    include_marker_start_groups: bool,
    device: torch.device,
) -> ReachabilityExamples:
    groups = _groups_from_rows(
        rows,
        include_marker_start_groups=include_marker_start_groups,
    )
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    targets = []
    actions = []
    for group_index, group in enumerate(groups):
        template = group[0]
        scene = _goal_scene_from_row(
            template,
            seed=scene_seed,
            width=width,
            height=height,
        )
        state = _state_from_dict(template["start_state"])
        recurrent_memory = torch.zeros(
            1,
            3,
            memory_size,
            memory_size,
            dtype=torch.float32,
            device=device,
        )
        last_action = "hold"
        last_collision = False
        for _step in range(max_steps):
            local_evidence = _latent_local_evidence(
                model=base_model,
                latent_map_head=latent_map_head,
                scene=scene,
                state=state,
                view_size=view_size,
                current_goal_marker=True,
                device=device,
            )
            local_evidence = _center_local_evidence(
                local_evidence,
                memory_size=memory_size,
            )
            logits = latent_memory_updater(
                recurrent_memory,
                local_evidence,
                _action_index_tensor(last_action, device=device),
                torch.tensor(
                    [float(last_collision)],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            recurrent_memory = logits.sigmoid().detach()
            memory_tensor = recurrent_memory[0].detach().cpu()
            memory_dict = _egocentric_memory_tensor_to_dict(
                memory_tensor,
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            target = build_reachability_target(
                memory_dict,
                memory_size=memory_size,
                target_mode=target_mode,
                gamma=target_gamma,
            )
            action, _mode = _select_egocentric_learned_value_map_action(
                _copy_memory(memory_dict),
                target.target_value[0],
            )
            memories.append(memory_tensor)
            targets.append(target)
            actions.append(ACTION_INDEX[action])
            rollout_action, _mode = _select_egocentric_value_field_action(memory_dict)
            next_state, collision = step_state(scene, state, rollout_action)
            last_action = rollout_action
            last_collision = bool(collision)
            state = next_state
            if (state.x, state.y) == scene.goal:
                break
        if (group_index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "built_groups": group_index + 1,
                        "examples": len(actions),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not memories:
        raise SystemExit("no Phase 3B reachability examples were generated")
    return ReachabilityExamples(
        memories=torch.stack(memories),
        targets=stack_reachability_targets(targets),
        actions=torch.tensor(actions, dtype=torch.long),
    )


def _target_batch_from_tensors(
    reachable_mask: torch.Tensor,
    current_distance: torch.Tensor,
    target_distance: torch.Tensor,
    target_value: torch.Tensor,
    target_mask: torch.Tensor,
    frontier_mask: torch.Tensor,
) -> Phase3BReachabilityTargetBatch:
    return Phase3BReachabilityTargetBatch(
        reachable_mask=reachable_mask,
        current_distance=current_distance,
        target_distance=target_distance,
        target_value=target_value,
        target_mask=target_mask,
        frontier_mask=frontier_mask,
    )


@torch.no_grad()
def _evaluate(
    head: Phase3BReachabilityHead,
    examples: ReachabilityExamples,
    *,
    batch_size: int,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    loss_weights: dict[str, float],
    device: torch.device,
) -> dict:
    head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.targets.reachable_mask,
        examples.targets.current_distance,
        examples.targets.target_distance,
        examples.targets.target_value,
        examples.targets.target_mask,
        examples.targets.frontier_mask,
        examples.actions,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_totals = {
        "loss": 0.0,
        "reachable_loss": 0.0,
        "current_distance_loss": 0.0,
        "target_distance_loss": 0.0,
        "target_value_loss": 0.0,
    }
    reachable_correct = 0
    reachable_true_positive = 0
    reachable_predicted_positive = 0
    reachable_target_positive = 0
    action_matches = 0
    target_top1_matches = 0
    target_value_abs_error = 0.0
    target_value_count = 0
    current_distance_abs_error = 0.0
    current_distance_count = 0
    target_distance_abs_error = 0.0
    target_distance_count = 0
    for batch in loader:
        (
            memory,
            reachable_mask,
            current_distance,
            target_distance,
            target_value,
            target_mask,
            frontier_mask,
            action,
        ) = batch
        memory = memory.to(device)
        target_batch = _target_batch_from_tensors(
            reachable_mask.to(device),
            current_distance.to(device),
            target_distance.to(device),
            target_value.to(device),
            target_mask.to(device),
            frontier_mask.to(device),
        )
        predictions = head(memory)
        losses = reachability_prediction_losses(
            predictions,
            target_batch,
            **loss_weights,
        )
        item_count = int(memory.shape[0])
        for key, value in losses.items():
            loss_totals[key] += float(value.item()) * item_count
        total += item_count
        reachable_probs = predictions["reachable_logits"].sigmoid().detach().cpu()
        target_value_probs = predictions["target_value_logits"].sigmoid().detach().cpu()
        current_distance_prediction = (
            F.softplus(predictions["current_distance"]).detach().cpu()
        )
        target_distance_prediction = (
            F.softplus(predictions["target_distance"]).detach().cpu()
        )
        memory_cpu = memory.detach().cpu()
        reachable_cpu = reachable_mask.detach().cpu().bool()
        target_value_cpu = target_value.detach().cpu()
        current_distance_cpu = current_distance.detach().cpu()
        target_distance_cpu = target_distance.detach().cpu()
        action_cpu = action.detach().cpu()
        predicted_reachable = reachable_probs >= 0.5
        reachable_correct += int((predicted_reachable == reachable_cpu).sum().item())
        reachable_true_positive += int(
            (predicted_reachable & reachable_cpu).sum().item()
        )
        reachable_predicted_positive += int(predicted_reachable.sum().item())
        reachable_target_positive += int(reachable_cpu.sum().item())
        current_mask = reachable_cpu
        target_distance_mask = target_value_cpu > 0.0
        current_distance_diff = (
            current_distance_prediction[current_mask]
            - current_distance_cpu[current_mask]
        )
        current_distance_abs_error += float(current_distance_diff.abs().sum().item())
        current_distance_count += int(current_mask.sum().item())
        target_distance_abs_error += float(
            (
                target_distance_prediction[target_distance_mask]
                - target_distance_cpu[target_distance_mask]
            )
            .abs()
            .sum()
            .item()
        )
        target_distance_count += int(target_distance_mask.sum().item())
        target_value_abs_error += float(
            (target_value_probs - target_value_cpu).abs().sum().item()
        )
        target_value_count += int(target_value_cpu.numel())
        for item_index in range(item_count):
            memory_dict = _egocentric_memory_tensor_to_dict(
                memory_cpu[item_index],
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            predicted_action, _mode = _select_egocentric_learned_value_map_action(
                _copy_memory(memory_dict),
                target_value_probs[item_index, 0],
            )
            if ACTION_INDEX[predicted_action] == int(action_cpu[item_index]):
                action_matches += 1
            target_flat = target_value_cpu[item_index, 0].flatten()
            if float(target_flat.max().item()) > 0.0:
                target_indices = {
                    int(index)
                    for index in (target_flat == target_flat.max())
                    .nonzero(as_tuple=False)
                    .flatten()
                }
                predicted_index = int(target_value_probs[item_index, 0].argmax().item())
                if predicted_index in target_indices:
                    target_top1_matches += 1
    cells = max(total * int(examples.targets.reachable_mask.shape[-1]) ** 2, 1)
    return {
        **{key: value / max(total, 1) for key, value in loss_totals.items()},
        "examples": total,
        "reachable_accuracy": reachable_correct / cells,
        "reachable_precision": (
            reachable_true_positive / max(reachable_predicted_positive, 1)
        ),
        "reachable_recall": reachable_true_positive / max(reachable_target_positive, 1),
        "current_distance_mae": (
            current_distance_abs_error / max(current_distance_count, 1)
        ),
        "target_distance_mae": (
            target_distance_abs_error / max(target_distance_count, 1)
        ),
        "target_value_mae": target_value_abs_error / max(target_value_count, 1),
        "target_top1_match": target_top1_matches / max(total, 1),
        "action_match": action_matches / max(total, 1),
        "reachable_positive_cells": reachable_target_positive,
        "reachable_predicted_positive_cells": reachable_predicted_positive,
    }


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
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
    parser.add_argument(
        "--target-mode",
        choices=("marker", "frontier", "marker_or_frontier"),
        default="marker_or_frontier",
    )
    parser.add_argument("--target-gamma", type=float, default=0.94)
    parser.add_argument("--reachable-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-distance-loss-weight", type=float, default=0.25)
    parser.add_argument("--target-distance-loss-weight", type=float, default=0.25)
    parser.add_argument("--target-value-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--save-best-metric",
        choices=(
            "action_match",
            "target_top1_match",
            "reachable_recall",
            "target_value_mae",
            "loss",
        ),
        default="action_match",
    )
    parser.add_argument("--seed", type=int, default=20260660)
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
    if not 0.0 < args.target_gamma <= 1.0:
        raise SystemExit("--target-gamma must be in (0, 1]")
    for name in (
        "reachable_loss_weight",
        "current_distance_loss_weight",
        "target_distance_loss_weight",
        "target_value_loss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")

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
    if int(latent_memory_updater.memory_size) != int(args.memory_size):
        raise SystemExit(
            "--memory-size must match latent memory updater size "
            f"({latent_memory_updater.memory_size})"
        )
    train_examples = _build_examples(
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
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_mode=args.target_mode,
        target_gamma=args.target_gamma,
        include_marker_start_groups=args.include_marker_start_train_groups,
        device=device,
    )
    validation_examples = _build_examples(
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
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_mode=args.target_mode,
        target_gamma=args.target_gamma,
        include_marker_start_groups=False,
        device=device,
    )
    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.targets.reachable_mask,
        train_examples.targets.current_distance,
        train_examples.targets.target_distance,
        train_examples.targets.target_value,
        train_examples.targets.target_mask,
        train_examples.targets.frontier_mask,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    head = Phase3BReachabilityHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
        architecture=args.architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_weights = {
        "reachable_weight": float(args.reachable_loss_weight),
        "current_distance_weight": float(args.current_distance_loss_weight),
        "target_distance_weight": float(args.target_distance_loss_weight),
        "target_value_weight": float(args.target_value_loss_weight),
    }
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_metrics = None
    best_step = None
    best_score = None
    for step in range(1, args.optimization_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        (
            memory,
            reachable_mask,
            current_distance,
            target_distance,
            target_value,
            target_mask,
            frontier_mask,
        ) = batch
        memory = memory.to(device)
        target_batch = _target_batch_from_tensors(
            reachable_mask.to(device),
            current_distance.to(device),
            target_distance.to(device),
            target_value.to(device),
            target_mask.to(device),
            frontier_mask.to(device),
        )
        predictions = head(memory)
        losses = reachability_prediction_losses(
            predictions,
            target_batch,
            **loss_weights,
        )
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                head,
                validation_examples,
                batch_size=args.batch_size,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                loss_weights=loss_weights,
                device=device,
            )
            entry = {
                "step": step,
                "train_loss": float(loss.item()),
                **metrics,
            }
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            metric_value = float(metrics[args.save_best_metric])
            if args.save_best_metric in {"loss", "target_value_mae"}:
                score = -metric_value
            else:
                score = metric_value
            if args.save_best and (best_score is None or score > best_score):
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }

    final_validation = _evaluate(
        head,
        validation_examples,
        batch_size=args.batch_size,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        loss_weights=loss_weights,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation
    report = {
        "schema": "jepa_phase3b_reachability_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(train_examples.memories.shape[0]),
        "validation_examples": int(validation_examples.memories.shape[0]),
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
            "architecture": args.architecture,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
