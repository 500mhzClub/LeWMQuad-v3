#!/usr/bin/env python3
"""Train a learned target-field head for Phase 3A recurrent value planning."""
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
    ACTION_NAMES,
    read_jsonl,
    step_state,
)
from lewm.benchmarks.phase3a_training import source_key  # noqa: E402
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AEgocentricMemoryUpdate,
    Phase3AEgocentricValueFieldHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _group_validation_sources,
    _infer_scene_seed,
    _is_egocentric_frontier_cell,
    _latent_local_evidence,
    _select_egocentric_learned_value_field_action,
    _select_egocentric_value_field_action,
    _state_from_dict,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402


@dataclass(frozen=True)
class ValueFieldExamples:
    memories: torch.Tensor
    targets: torch.Tensor
    actions: torch.Tensor
    marker_targets: torch.Tensor


def _concat_examples(first: ValueFieldExamples, second: ValueFieldExamples) -> ValueFieldExamples:
    return ValueFieldExamples(
        memories=torch.cat([first.memories, second.memories], dim=0),
        targets=torch.cat([first.targets, second.targets], dim=0),
        actions=torch.cat([first.actions, second.actions], dim=0),
        marker_targets=torch.cat([first.marker_targets, second.marker_targets], dim=0),
    )


def _load_latent_memory_updater(
    path: Path,
    *,
    model: nn.Module,
    device: torch.device,
) -> tuple[Phase3AEgocentricMemoryUpdate, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    updater = Phase3AEgocentricMemoryUpdate(
        memory_size=int(config.get("memory_size", model.spatial_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        memory_channels=int(config.get("memory_channels", 3)),
        evidence_channels=int(config.get("evidence_channels", 3)),
        action_dim=int(config.get("action_dim", len(ACTION_NAMES))),
        use_geometric_prior=bool(config.get("use_geometric_prior", True)),
        learned_transition_hidden_dim=config.get("learned_transition_hidden_dim"),
    ).to(device)
    updater.load_state_dict(checkpoint["updater_state_dict"])
    updater.eval()
    return updater, report


def _load_value_field_head(
    path: Path,
    *,
    fallback_memory_size: int,
    device: torch.device,
) -> tuple[Phase3AEgocentricValueFieldHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    head = Phase3AEgocentricValueFieldHead(
        memory_size=int(config.get("memory_size", fallback_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        memory_channels=int(config.get("memory_channels", 3)),
        output_channels=int(config.get("output_channels", 1)),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    return head, report


def _copy_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
        "radius": int(memory.get("radius", 0)),
    }


def _marker_target_cells_for_memory(memory: dict) -> set[tuple[int, int]]:
    working = _copy_memory(memory)
    current = (0, 0)
    working["free"].add(current)
    working["blocked"].discard(current)
    marker = working.get("marker")
    if marker is not None and marker in working["free"]:
        return {marker}
    return set()


def _frontier_target_cells_for_memory(memory: dict) -> set[tuple[int, int]]:
    working = _copy_memory(memory)
    current = (0, 0)
    working["free"].add(current)
    working["blocked"].discard(current)
    if _is_egocentric_frontier_cell(working, current):
        return {current}
    targets = {
        cell
        for cell in working["free"]
        if cell != current and _is_egocentric_frontier_cell(working, cell)
    }
    return targets or {current}


def _target_tensor_from_memory(
    memory: dict,
    *,
    memory_size: int,
    output_channels: int,
) -> torch.Tensor:
    radius = memory_size // 2
    tensor = torch.zeros(output_channels, memory_size, memory_size, dtype=torch.float32)
    if output_channels == 1:
        target_cells = (
            _marker_target_cells_for_memory(memory)
            or _frontier_target_cells_for_memory(memory)
        )
        channels_and_cells = [(0, target_cells)]
    else:
        channels_and_cells = [
            (0, _marker_target_cells_for_memory(memory)),
            (1, _frontier_target_cells_for_memory(memory)),
        ]
    for channel, cells in channels_and_cells:
        for ahead, lateral in cells:
            row = radius - int(ahead)
            col = radius + int(lateral)
            if 0 <= row < memory_size and 0 <= col < memory_size:
                tensor[channel, row, col] = 1.0
    return tensor


def _groups_from_rows(rows: list[dict], *, include_marker_start_groups: bool) -> list[list[dict]]:
    if include_marker_start_groups:
        grouped = {}
        for row in rows:
            grouped.setdefault(source_key(row), []).append(row)
        return [grouped[key] for key in sorted(grouped)]
    return _group_validation_sources(rows)


@torch.no_grad()
def _select_rollout_value_field_action(
    head: Phase3AEgocentricValueFieldHead,
    recurrent_memory: torch.Tensor,
    memory_dict: dict,
    *,
    target_threshold: float,
    target_top_k: int,
    fixed_marker_target: bool,
) -> str:
    target_fields = head(recurrent_memory).sigmoid()[0].detach().cpu()
    marker = memory_dict.get("marker")
    if (
        int(target_fields.shape[0]) >= 2
        and marker is not None
        and marker in memory_dict["free"]
    ):
        channel = 0
    elif int(target_fields.shape[0]) >= 2:
        channel = 1
    else:
        channel = 0
    action, _mode = _select_egocentric_learned_value_field_action(
        memory_dict,
        target_fields[channel],
        threshold=target_threshold,
        top_k=target_top_k,
        fixed_marker_target=fixed_marker_target,
    )
    return action


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
    latent_memory_updater: Phase3AEgocentricMemoryUpdate,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    output_channels: int,
    include_marker_start_groups: bool,
    rollout_value_field_head: Phase3AEgocentricValueFieldHead | None,
    rollout_target_threshold: float,
    rollout_target_top_k: int,
    rollout_fixed_marker_target: bool,
    device: torch.device,
) -> ValueFieldExamples:
    groups = _groups_from_rows(
        rows,
        include_marker_start_groups=include_marker_start_groups,
    )
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    targets = []
    actions = []
    marker_targets = []
    for group_index, group in enumerate(groups):
        template = group[0]
        scene = _goal_scene_from_row(template, seed=scene_seed, width=width, height=height)
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
                torch.tensor([float(last_collision)], dtype=torch.float32, device=device),
            )
            recurrent_memory = logits.sigmoid().detach()
            memory_dict = _egocentric_memory_tensor_to_dict(
                recurrent_memory[0].detach().cpu(),
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            action, _mode = _select_egocentric_value_field_action(memory_dict)
            marker_target_cells = _marker_target_cells_for_memory(memory_dict)
            memories.append(recurrent_memory[0].detach().cpu())
            targets.append(
                _target_tensor_from_memory(
                    memory_dict,
                    memory_size=memory_size,
                    output_channels=output_channels,
                )
            )
            actions.append(ACTION_INDEX[action])
            marker_targets.append(bool(marker_target_cells))
            rollout_action = action
            if rollout_value_field_head is not None:
                rollout_action = _select_rollout_value_field_action(
                    rollout_value_field_head,
                    recurrent_memory,
                    memory_dict,
                    target_threshold=rollout_target_threshold,
                    target_top_k=rollout_target_top_k,
                    fixed_marker_target=rollout_fixed_marker_target,
                )
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
        raise SystemExit("no value-field examples were generated")
    return ValueFieldExamples(
        memories=torch.stack(memories),
        targets=torch.stack(targets),
        actions=torch.tensor(actions, dtype=torch.long),
        marker_targets=torch.tensor(marker_targets, dtype=torch.bool),
    )


@torch.no_grad()
def _evaluate(
    head: Phase3AEgocentricValueFieldHead,
    examples: ValueFieldExamples,
    *,
    batch_size: int,
    positive_weight: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    output_channels: int,
    device: torch.device,
) -> dict:
    head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.targets,
        examples.actions,
        examples.marker_targets,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    total_loss = 0.0
    top1_matches = 0
    action_matches = 0
    target_pixels = 0
    pred_pixels = 0
    true_positive_pixels = 0
    marker_total = 0
    marker_top1_matches = 0
    marker_action_matches = 0
    frontier_total = 0
    frontier_top1_matches = 0
    frontier_action_matches = 0
    pos_weight = torch.tensor([positive_weight], dtype=torch.float32, device=device).view(
        1,
        1,
        1,
        1,
    )
    for memory, target, action, marker_target in loader:
        memory = memory.to(device)
        target = target.to(device)
        logits = head(memory)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        probs = logits.sigmoid().detach().cpu()
        memory_cpu = memory.detach().cpu()
        target_cpu = target.detach().cpu()
        total_loss += float(loss.item()) * int(memory.shape[0])
        total += int(memory.shape[0])
        marker_target_cpu = marker_target.detach().cpu()
        for item_index in range(int(memory_cpu.shape[0])):
            memory_dict = _egocentric_memory_tensor_to_dict(
                memory_cpu[item_index],
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            marker = memory_dict.get("marker")
            if output_channels >= 2 and marker is not None and marker in memory_dict["free"]:
                channel = 0
            elif output_channels >= 2:
                channel = 1
            else:
                channel = 0
            target_flat = target_cpu[item_index, channel].flatten()
            pred_flat = probs[item_index, channel].flatten()
            if int(target_flat.sum().item()) > 0:
                target_indices = set(
                    int(index)
                    for index in (target_flat >= 0.5).nonzero(as_tuple=False).flatten()
                )
                if int(pred_flat.argmax().item()) in target_indices:
                    top1_matches += 1
                    if bool(marker_target_cpu[item_index]):
                        marker_top1_matches += 1
                    else:
                        frontier_top1_matches += 1
            predicted_action, _mode = _select_egocentric_learned_value_field_action(
                memory_dict,
                probs[item_index, channel],
                threshold=target_threshold,
                top_k=target_top_k,
            )
            if ACTION_INDEX[predicted_action] == int(action[item_index]):
                action_matches += 1
                if bool(marker_target_cpu[item_index]):
                    marker_action_matches += 1
                else:
                    frontier_action_matches += 1
            if bool(marker_target_cpu[item_index]):
                marker_total += 1
            else:
                frontier_total += 1
            pred_binary = pred_flat >= target_threshold
            target_binary = target_flat >= 0.5
            target_pixels += int(target_binary.sum().item())
            pred_pixels += int(pred_binary.sum().item())
            true_positive_pixels += int((pred_binary & target_binary).sum().item())
    return {
        "examples": total,
        "loss": total_loss / max(total, 1),
        "target_top1_match": top1_matches / max(total, 1),
        "action_match": action_matches / max(total, 1),
        "marker_examples": marker_total,
        "marker_target_top1_match": marker_top1_matches / max(marker_total, 1),
        "marker_action_match": marker_action_matches / max(marker_total, 1),
        "frontier_examples": frontier_total,
        "frontier_target_top1_match": frontier_top1_matches / max(frontier_total, 1),
        "frontier_action_match": frontier_action_matches / max(frontier_total, 1),
        "target_recall": true_positive_pixels / max(target_pixels, 1),
        "target_precision": true_positive_pixels / max(pred_pixels, 1),
        "target_pixels": target_pixels,
        "predicted_pixels": pred_pixels,
    }


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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=32.0)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--marker-sample-weight", type=float, default=1.0)
    parser.add_argument("--dagger-rollout-value-field-head", type=Path, default=None)
    parser.add_argument("--dagger-rollout-target-threshold", type=float, default=0.5)
    parser.add_argument("--dagger-rollout-target-top-k", type=int, default=16)
    parser.add_argument("--dagger-rollout-fixed-marker-target", action="store_true")
    parser.add_argument("--save-best-action-match", action="store_true")
    parser.add_argument(
        "--save-best-metric",
        choices=(
            "action_match",
            "marker_action_match",
            "target_top1_match",
            "marker_target_top1_match",
        ),
        default="action_match",
    )
    parser.add_argument("--seed", type=int, default=20260653)
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
    if not 0.0 <= args.target_threshold <= 1.0:
        raise SystemExit("--target-threshold must be in [0, 1]")
    if args.target_top_k < 1:
        raise SystemExit("--target-top-k must be positive")
    if args.marker_sample_weight <= 0.0:
        raise SystemExit("--marker-sample-weight must be positive")
    if not 0.0 <= args.dagger_rollout_target_threshold <= 1.0:
        raise SystemExit("--dagger-rollout-target-threshold must be in [0, 1]")
    if args.dagger_rollout_target_top_k < 1:
        raise SystemExit("--dagger-rollout-target-top-k must be positive")

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
    dagger_rollout_value_field_head = None
    dagger_rollout_value_field_report = None
    if args.dagger_rollout_value_field_head is not None:
        dagger_rollout_value_field_head, dagger_rollout_value_field_report = (
            _load_value_field_head(
                args.dagger_rollout_value_field_head,
                fallback_memory_size=args.memory_size,
                device=device,
            )
        )
        if int(dagger_rollout_value_field_head.memory_size) != int(args.memory_size):
            raise SystemExit(
                "--memory-size must match DAgger rollout value-field checkpoint "
                f"({dagger_rollout_value_field_head.memory_size})"
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
        output_channels=args.output_channels,
        include_marker_start_groups=args.include_marker_start_train_groups,
        rollout_value_field_head=None,
        rollout_target_threshold=args.dagger_rollout_target_threshold,
        rollout_target_top_k=args.dagger_rollout_target_top_k,
        rollout_fixed_marker_target=False,
        device=device,
    )
    if dagger_rollout_value_field_head is not None:
        train_rollout_examples = _build_examples(
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
            output_channels=args.output_channels,
            include_marker_start_groups=args.include_marker_start_train_groups,
            rollout_value_field_head=dagger_rollout_value_field_head,
            rollout_target_threshold=args.dagger_rollout_target_threshold,
            rollout_target_top_k=args.dagger_rollout_target_top_k,
            rollout_fixed_marker_target=bool(args.dagger_rollout_fixed_marker_target),
            device=device,
        )
        train_examples = _concat_examples(train_examples, train_rollout_examples)
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
        output_channels=args.output_channels,
        include_marker_start_groups=False,
        rollout_value_field_head=None,
        rollout_target_threshold=args.dagger_rollout_target_threshold,
        rollout_target_top_k=args.dagger_rollout_target_top_k,
        rollout_fixed_marker_target=False,
        device=device,
    )
    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.targets,
        train_examples.marker_targets,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    head = Phase3AEgocentricValueFieldHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
        output_channels=args.output_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pos_weight = torch.tensor(
        [args.positive_weight],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 1, 1)
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_metrics = None
    best_step = None
    best_score = (-1.0, -1.0, -1.0, -1.0, float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, target, marker_target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, marker_target = next(iterator)
        memory = memory.to(device)
        target = target.to(device)
        marker_target = marker_target.to(device)
        logits = head(memory)
        loss_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        loss_per_example = loss_per_pixel.mean(dim=(1, 2, 3))
        sample_weights = torch.where(
            marker_target,
            torch.full_like(loss_per_example, float(args.marker_sample_weight)),
            torch.ones_like(loss_per_example),
        )
        loss = (loss_per_example * sample_weights).sum() / sample_weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                head,
                validation_examples,
                batch_size=args.batch_size,
                positive_weight=args.positive_weight,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                target_threshold=args.target_threshold,
                target_top_k=args.target_top_k,
                output_channels=args.output_channels,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            primary = float(metrics[args.save_best_metric])
            score = (
                primary,
                float(metrics["action_match"]),
                float(metrics["marker_action_match"]),
                float(metrics["target_top1_match"]),
                -float(metrics["loss"]),
            )
            if args.save_best_action_match and score > best_score:
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
        positive_weight=args.positive_weight,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_threshold=args.target_threshold,
        target_top_k=args.target_top_k,
        output_channels=args.output_channels,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best_action_match and best_state is not None:
        head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation
    report = {
        "schema": "jepa_phase3a_value_field_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "dagger_rollout_value_field_head": (
            str(args.dagger_rollout_value_field_head.resolve())
            if args.dagger_rollout_value_field_head
            else None
        ),
        "dagger_rollout_value_field_completed_steps": (
            dagger_rollout_value_field_report.get("completed_steps")
            if dagger_rollout_value_field_report
            else None
        ),
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
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
            "output_channels": args.output_channels,
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
