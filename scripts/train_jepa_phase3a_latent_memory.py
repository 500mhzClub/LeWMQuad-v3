#!/usr/bin/env python3
"""Train a recurrent Phase 3A egocentric memory updater."""
from __future__ import annotations

import argparse
import json
import random
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
    GridScene,
    GridState,
    _goal_marker_visible,
    read_jsonl,
    render_observation,
    step_state,
)
from lewm.benchmarks.phase3a_training import source_key  # noqa: E402
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AEgocentricMemoryUpdate,
    Phase3ALatentMapHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _goal_scene_from_row,
    _group_validation_sources,
    _infer_scene_seed,
    _roll_egocentric_frontier_memory,
    _select_egocentric_frontier_action,
    _state_from_dict,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402


@dataclass(frozen=True)
class MemoryExamples:
    previous_memory: torch.Tensor
    local_evidence: torch.Tensor
    actions: torch.Tensor
    collisions: torch.Tensor
    targets: torch.Tensor
    teacher_actions: torch.Tensor
    marker_visible: torch.Tensor


def _empty_memory(*, memory_size: int) -> dict:
    return {
        "free": set(),
        "blocked": set(),
        "marker": None,
        "radius": memory_size // 2,
    }


def _copy_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
        "radius": int(memory.get("radius", 0)),
    }


def _memory_from_tensor(tensor: torch.Tensor) -> dict:
    if tensor.ndim != 3:
        raise ValueError(f"expected tensor shape (C, S, S), got {tensor.shape}")
    _channels, memory_size, _width = tensor.shape
    radius = memory_size // 2
    memory = _empty_memory(memory_size=memory_size)
    for row in range(memory_size):
        for col in range(memory_size):
            ahead = radius - row
            lateral = col - radius
            if float(tensor[0, row, col]) >= 0.5:
                memory["blocked"].add((ahead, lateral))
            elif float(tensor[1, row, col]) >= 0.5:
                memory["free"].add((ahead, lateral))
            if float(tensor[2, row, col]) >= 0.5:
                memory["marker"] = (ahead, lateral)
    return memory


def _memory_to_tensor(memory: dict, *, memory_size: int) -> torch.Tensor:
    radius = memory_size // 2
    tensor = torch.zeros(3, memory_size, memory_size, dtype=torch.float32)
    for ahead, lateral in memory["blocked"]:
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            tensor[0, row, col] = 1.0
    for ahead, lateral in memory["free"]:
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            tensor[1, row, col] = 1.0
    marker = memory.get("marker")
    if marker is not None:
        ahead, lateral = marker
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            tensor[2, row, col] = 1.0
    return tensor


def _random_memory_tensor(
    rng: random.Random,
    *,
    memory_size: int,
) -> torch.Tensor:
    tensor = torch.zeros(3, memory_size, memory_size, dtype=torch.float32)
    radius = memory_size // 2
    tensor[1, radius, radius] = 1.0
    cell_count = rng.randint(memory_size, memory_size * 3)
    for _ in range(cell_count):
        row = rng.randrange(memory_size)
        col = rng.randrange(memory_size)
        if row == radius and col == radius:
            continue
        channel = 0 if rng.random() < 0.28 else 1
        tensor[0, row, col] = 0.0
        tensor[1, row, col] = 0.0
        tensor[channel, row, col] = 1.0
    if rng.random() < 0.35:
        free_cells = (tensor[1] >= 0.5).nonzero(as_tuple=False)
        if int(free_cells.shape[0]) > 0:
            marker_cell = free_cells[rng.randrange(int(free_cells.shape[0]))]
            tensor[2, int(marker_cell[0]), int(marker_cell[1])] = 1.0
    return tensor


def _synthetic_transition_examples(
    *,
    count: int,
    memory_size: int,
    seed: int,
) -> MemoryExamples:
    if count < 1:
        empty = torch.empty(0, 3, memory_size, memory_size)
        return MemoryExamples(
            previous_memory=empty,
            local_evidence=empty.clone(),
            actions=torch.empty(0, dtype=torch.long),
            collisions=torch.empty(0, dtype=torch.float32),
            targets=empty.clone(),
            teacher_actions=torch.empty(0, dtype=torch.long),
            marker_visible=torch.empty(0, dtype=torch.float32),
        )
    rng = random.Random(seed)
    previous = []
    evidence = []
    actions = []
    collisions = []
    targets = []
    for index in range(count):
        action_id = index % len(ACTION_NAMES)
        action = ACTION_NAMES[action_id]
        collision = bool(action == "forward" and rng.random() < 0.2)
        previous_tensor = _random_memory_tensor(rng, memory_size=memory_size)
        memory = _memory_from_tensor(previous_tensor)
        _roll_egocentric_frontier_memory(memory, action, collision=collision)
        previous.append(previous_tensor)
        evidence.append(torch.zeros_like(previous_tensor))
        actions.append(action_id)
        collisions.append(float(collision))
        targets.append(_memory_to_tensor(memory, memory_size=memory_size))
    return MemoryExamples(
        previous_memory=torch.stack(previous),
        local_evidence=torch.stack(evidence),
        actions=torch.tensor(actions, dtype=torch.long),
        collisions=torch.tensor(collisions, dtype=torch.float32),
        targets=torch.stack(targets),
        teacher_actions=torch.zeros(count, dtype=torch.long),
        marker_visible=torch.zeros(count, dtype=torch.float32),
    )


def _concat_examples(first: MemoryExamples, second: MemoryExamples) -> MemoryExamples:
    if len(second.actions) == 0:
        return first
    return MemoryExamples(
        previous_memory=torch.cat([first.previous_memory, second.previous_memory], dim=0),
        local_evidence=torch.cat([first.local_evidence, second.local_evidence], dim=0),
        actions=torch.cat([first.actions, second.actions], dim=0),
        collisions=torch.cat([first.collisions, second.collisions], dim=0),
        targets=torch.cat([first.targets, second.targets], dim=0),
        teacher_actions=torch.cat([first.teacher_actions, second.teacher_actions], dim=0),
        marker_visible=torch.cat([first.marker_visible, second.marker_visible], dim=0),
    )


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def _tensor_to_memory(
    tensor: torch.Tensor,
    *,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
) -> dict:
    if tensor.ndim != 3:
        raise ValueError(f"expected tensor shape (C, S, S), got {tensor.shape}")
    channels, memory_size, _width = tensor.shape
    if channels != 3:
        raise ValueError(f"expected 3 memory channels, got {channels}")
    radius = memory_size // 2
    blocked_probs = tensor[0]
    free_probs = tensor[1]
    marker_probs = tensor[2]
    memory = _empty_memory(memory_size=memory_size)
    for row in range(memory_size):
        for col in range(memory_size):
            ahead = radius - row
            lateral = col - radius
            blocked = (
                float(blocked_probs[row, col]) >= blocked_threshold
                and float(blocked_probs[row, col]) > float(free_probs[row, col])
            )
            free = (
                float(free_probs[row, col]) >= free_threshold
                and float(free_probs[row, col]) >= float(blocked_probs[row, col])
            )
            if blocked:
                memory["blocked"].add((ahead, lateral))
            elif free:
                memory["free"].add((ahead, lateral))
    marker_index = int(marker_probs.flatten().argmax().item())
    marker_score = float(marker_probs.flatten()[marker_index])
    if marker_score >= marker_threshold:
        row = marker_index // memory_size
        col = marker_index % memory_size
        memory["marker"] = (radius - row, col - radius)
    return memory


def _center_local_evidence(local_evidence: torch.Tensor, *, memory_size: int) -> torch.Tensor:
    view_size = int(local_evidence.shape[-1])
    if view_size > memory_size:
        raise ValueError("view_size cannot exceed memory_size")
    output = torch.zeros(3, memory_size, memory_size, dtype=torch.float32)
    start = memory_size // 2 - view_size // 2
    output[:, start : start + view_size, start : start + view_size] = local_evidence
    return output


@torch.no_grad()
def _latent_local_evidence(
    base_model: nn.Module,
    latent_map_head: Phase3ALatentMapHead,
    observation: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    vision = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    tokens = base_model.encoder(vision)
    return latent_map_head(tokens).sigmoid()[0].detach().cpu()


def _apply_local_evidence(
    memory: dict,
    local_evidence: torch.Tensor,
    *,
    blocked_threshold: float,
    marker_threshold: float,
) -> None:
    view_size = int(local_evidence.shape[-1])
    radius = view_size // 2
    blocked_probs = local_evidence[0]
    free_probs = local_evidence[1]
    marker_probs = local_evidence[2]
    marker_index = int(marker_probs.flatten().argmax().item())
    marker_score = float(marker_probs.flatten()[marker_index])
    if marker_score >= marker_threshold:
        row = marker_index // view_size
        col = marker_index % view_size
        memory["marker"] = (radius - row, col - radius)
    for row in range(view_size):
        for col in range(view_size):
            cell = (radius - row, col - radius)
            blocked = (
                float(blocked_probs[row, col]) >= blocked_threshold
                and float(blocked_probs[row, col]) > float(free_probs[row, col])
            )
            if blocked:
                memory["blocked"].add(cell)
                memory["free"].discard(cell)
            else:
                memory["free"].add(cell)
                memory["blocked"].discard(cell)


def _build_examples_for_group(
    group: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_steps: int,
    base_model: nn.Module,
    latent_map_head: Phase3ALatentMapHead,
    blocked_threshold: float,
    marker_threshold: float,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor, int, float]]:
    template = group[0]
    scene = _goal_scene_from_row(template, seed=scene_seed, width=width, height=height)
    state = _state_from_dict(template["start_state"])
    previous_decision_memory = _empty_memory(memory_size=memory_size)
    last_action = "hold"
    last_collision = False
    examples = []
    for _step in range(max_steps):
        observation = render_observation(
            scene,
            state,
            view_size=view_size,
            include_goal_beacon=False,
            show_goal_marker=True,
        )
        local_evidence = _latent_local_evidence(
            base_model,
            latent_map_head,
            observation,
            device=device,
        )
        target_memory = _copy_memory(previous_decision_memory)
        _roll_egocentric_frontier_memory(
            target_memory,
            last_action,
            collision=bool(last_collision),
        )
        _apply_local_evidence(
            target_memory,
            local_evidence,
            blocked_threshold=blocked_threshold,
            marker_threshold=marker_threshold,
        )
        teacher_action = _select_egocentric_frontier_action(_copy_memory(target_memory))
        examples.append(
            (
                _memory_to_tensor(previous_decision_memory, memory_size=memory_size),
                _center_local_evidence(local_evidence, memory_size=memory_size),
                ACTION_INDEX[last_action],
                float(last_collision),
                _memory_to_tensor(target_memory, memory_size=memory_size),
                ACTION_INDEX[teacher_action],
                float(_goal_marker_visible(scene, state, view_size=view_size)),
            )
        )
        next_state, collision = step_state(scene, state, teacher_action)
        previous_decision_memory = target_memory
        last_action = teacher_action
        last_collision = bool(collision)
        state = next_state
        if (state.x, state.y) == scene.goal:
            break
    return examples


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
    latent_map_head: Phase3ALatentMapHead,
    blocked_threshold: float,
    marker_threshold: float,
    device: torch.device,
    include_marker_start_groups: bool,
) -> MemoryExamples:
    if include_marker_start_groups:
        grouped = {}
        for row in rows:
            grouped.setdefault(source_key(row), []).append(row)
        groups = [grouped[key] for key in sorted(grouped)]
    else:
        groups = _group_validation_sources(rows)
    if max_episodes is not None:
        groups = groups[:max_episodes]
    raw_examples = []
    for index, group in enumerate(groups):
        raw_examples.extend(
            _build_examples_for_group(
                group,
                scene_seed=scene_seed,
                width=width,
                height=height,
                view_size=view_size,
                memory_size=memory_size,
                max_steps=max_steps,
                base_model=base_model,
                latent_map_head=latent_map_head,
                blocked_threshold=blocked_threshold,
                marker_threshold=marker_threshold,
                device=device,
            )
        )
        if (index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "built_groups": index + 1,
                        "examples": len(raw_examples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not raw_examples:
        raise SystemExit("no memory examples were generated")
    previous, evidence, actions, collisions, targets, teacher_actions, marker_visible = zip(
        *raw_examples
    )
    return MemoryExamples(
        previous_memory=torch.stack(list(previous)),
        local_evidence=torch.stack(list(evidence)),
        actions=torch.tensor(actions, dtype=torch.long),
        collisions=torch.tensor(collisions, dtype=torch.float32),
        targets=torch.stack(list(targets)),
        teacher_actions=torch.tensor(teacher_actions, dtype=torch.long),
        marker_visible=torch.tensor(marker_visible, dtype=torch.float32),
    )


def _loss(logits: torch.Tensor, targets: torch.Tensor, *, marker_weight: float) -> torch.Tensor:
    pos_weight = logits.new_tensor([4.0, 1.0, marker_weight]).view(1, 3, 1, 1)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


@torch.no_grad()
def _evaluate(
    updater: Phase3AEgocentricMemoryUpdate,
    examples: MemoryExamples,
    *,
    batch_size: int,
    marker_weight: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    device: torch.device,
) -> dict:
    updater.eval()
    dataset = TensorDataset(
        examples.previous_memory,
        examples.local_evidence,
        examples.actions,
        examples.collisions,
        examples.targets,
        examples.teacher_actions,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total = 0
    cell_correct = torch.zeros(3, dtype=torch.long)
    cell_total = 0
    action_matches = 0
    marker_present = 0
    marker_top1 = 0
    for previous, evidence, actions, collisions, targets, teacher_actions in loader:
        previous = previous.to(device)
        evidence = evidence.to(device)
        actions = actions.to(device)
        collisions = collisions.to(device)
        targets = targets.to(device)
        logits = updater(previous, evidence, actions, collisions)
        loss = _loss(logits, targets, marker_weight=marker_weight)
        probs = logits.sigmoid().detach().cpu()
        target_cpu = targets.detach().cpu()
        predictions = probs >= 0.5
        target_binary = target_cpu >= 0.5
        cell_correct += (predictions == target_binary).sum(dim=(0, 2, 3)).to(torch.long)
        cell_total += int(target_binary[:, 0].numel())
        total_loss += float(loss.item()) * previous.shape[0]
        total += int(previous.shape[0])
        for item_index in range(int(probs.shape[0])):
            predicted_memory = _tensor_to_memory(
                probs[item_index],
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            predicted_action = _select_egocentric_frontier_action(predicted_memory)
            if ACTION_INDEX[predicted_action] == int(teacher_actions[item_index]):
                action_matches += 1
            marker_target = target_cpu[item_index, 2].flatten()
            if float(marker_target.max()) >= 0.5:
                marker_present += 1
                if int(probs[item_index, 2].flatten().argmax()) == int(marker_target.argmax()):
                    marker_top1 += 1
    return {
        "examples": total,
        "loss": total_loss / max(total, 1),
        "blocked_accuracy": int(cell_correct[0]) / max(cell_total, 1),
        "free_accuracy": int(cell_correct[1]) / max(cell_total, 1),
        "marker_accuracy": int(cell_correct[2]) / max(cell_total, 1),
        "teacher_action_match": action_matches / max(total, 1),
        "marker_top1_when_present": marker_top1 / max(marker_present, 1),
        "marker_present_examples": marker_present,
    }


def _load_latent_map_head(
    path: Path,
    *,
    base_model: nn.Module,
    device: torch.device,
) -> tuple[Phase3ALatentMapHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    head = Phase3ALatentMapHead(
        view_size=int(config.get("view_size", base_model.view_size)),
        latent_dim=int(config.get("latent_dim", base_model.latent_dim)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        output_channels=int(config.get("output_channels", 3)),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    return head, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument(
        "--extra-train-data",
        type=Path,
        action="append",
        default=[],
        help="additional train JSONL files to build memory examples from",
    )
    parser.add_argument(
        "--extra-validation-data",
        type=Path,
        action="append",
        default=[],
        help="additional validation JSONL files to build memory examples from",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=None)
    parser.add_argument(
        "--include-marker-start-train-groups",
        action="store_true",
        help="build train traces from all source groups, not only marker-unseen starts",
    )
    parser.add_argument(
        "--synthetic-transition-examples",
        type=int,
        default=0,
        help="add random supervised action/collision memory-transition examples",
    )
    parser.add_argument(
        "--transition-pretrain-steps",
        type=int,
        default=0,
        help="train on synthetic transition examples first, then fine-tune on real traces",
    )
    parser.add_argument(
        "--save-best-action-match",
        action="store_true",
        help="save the checkpoint with best validation teacher-action match",
    )
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learned-transition-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--no-geometric-prior",
        action="store_true",
        help="learn the action/collision memory transition instead of using a fixed roll",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-map-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.9)
    parser.add_argument("--memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--memory-marker-threshold", type=float, default=0.5)
    parser.add_argument("--marker-loss-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260646)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.max_train_episodes is not None and args.max_train_episodes < 1:
        raise SystemExit("--max-train-episodes must be positive")
    if args.max_validation_episodes is not None and args.max_validation_episodes < 1:
        raise SystemExit("--max-validation-episodes must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.synthetic_transition_examples < 0:
        raise SystemExit("--synthetic-transition-examples must be non-negative")
    if args.transition_pretrain_steps < 0:
        raise SystemExit("--transition-pretrain-steps must be non-negative")
    if args.transition_pretrain_steps > 0 and args.synthetic_transition_examples < 1:
        raise SystemExit(
            "--transition-pretrain-steps requires --synthetic-transition-examples"
        )
    if (
        args.learned_transition_hidden_dim is not None
        and args.learned_transition_hidden_dim < 0
    ):
        raise SystemExit("--learned-transition-hidden-dim must be non-negative")

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
    print(
        json.dumps(
            {
                "stage": "build_train_examples",
                "device": str(device),
                "train_seed": train_seed,
                "validation_seed": validation_seed,
            },
            sort_keys=True,
        ),
        flush=True,
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
        blocked_threshold=args.latent_map_blocked_threshold,
        marker_threshold=args.latent_map_marker_threshold,
        device=device,
        include_marker_start_groups=args.include_marker_start_train_groups,
    )
    for extra_train_path in args.extra_train_data:
        extra_train_seed = _infer_scene_seed(extra_train_path)
        if extra_train_seed is None:
            raise SystemExit(
                f"could not infer extra train scene seed from {extra_train_path}"
            )
        extra_train_examples = _build_examples(
            read_jsonl(extra_train_path),
            scene_seed=extra_train_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            max_episodes=args.max_train_episodes,
            max_steps=args.max_steps,
            base_model=base_model,
            latent_map_head=latent_map_head,
            blocked_threshold=args.latent_map_blocked_threshold,
            marker_threshold=args.latent_map_marker_threshold,
            device=device,
            include_marker_start_groups=args.include_marker_start_train_groups,
        )
        train_examples = _concat_examples(train_examples, extra_train_examples)
    synthetic_examples = _synthetic_transition_examples(
        count=args.synthetic_transition_examples,
        memory_size=args.memory_size,
        seed=args.seed + 99173,
    )
    if args.synthetic_transition_examples > 0 and args.transition_pretrain_steps == 0:
        train_examples = _concat_examples(train_examples, synthetic_examples)
    print(
        json.dumps(
            {
                "stage": "build_validation_examples",
                "train_examples": len(train_examples.actions),
                "synthetic_transition_examples": args.synthetic_transition_examples,
                "transition_pretrain_steps": args.transition_pretrain_steps,
            },
            sort_keys=True,
        ),
        flush=True,
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
        blocked_threshold=args.latent_map_blocked_threshold,
        marker_threshold=args.latent_map_marker_threshold,
        device=device,
        include_marker_start_groups=False,
    )
    for extra_validation_path in args.extra_validation_data:
        extra_validation_seed = _infer_scene_seed(extra_validation_path)
        if extra_validation_seed is None:
            raise SystemExit(
                "could not infer extra validation scene seed from "
                f"{extra_validation_path}"
            )
        extra_validation_examples = _build_examples(
            read_jsonl(extra_validation_path),
            scene_seed=extra_validation_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            max_episodes=args.max_validation_episodes,
            max_steps=args.max_steps,
            base_model=base_model,
            latent_map_head=latent_map_head,
            blocked_threshold=args.latent_map_blocked_threshold,
            marker_threshold=args.latent_map_marker_threshold,
            device=device,
            include_marker_start_groups=False,
        )
        validation_examples = _concat_examples(
            validation_examples,
            extra_validation_examples,
        )
    updater = Phase3AEgocentricMemoryUpdate(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        use_geometric_prior=not args.no_geometric_prior,
        learned_transition_hidden_dim=args.learned_transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        updater.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    marker_weight = (
        float(args.memory_size * args.memory_size)
        if args.marker_loss_weight is None
        else float(args.marker_loss_weight)
    )
    logs = []
    best_state = None
    best_metrics = None
    best_step = None
    best_score = (-1.0, float("-inf"))

    def _run_phase(
        *,
        phase: str,
        examples: MemoryExamples,
        steps: int,
        global_step: int,
    ) -> int:
        nonlocal best_state, best_metrics, best_step, best_score
        if steps < 1:
            return global_step
        train_dataset = TensorDataset(
            examples.previous_memory,
            examples.local_evidence,
            examples.actions,
            examples.collisions,
            examples.targets,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        iterator = iter(train_loader)
        for phase_step in range(1, steps + 1):
            try:
                previous, evidence, actions, collisions, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                previous, evidence, actions, collisions, targets = next(iterator)
            previous = previous.to(device)
            evidence = evidence.to(device)
            actions = actions.to(device)
            collisions = collisions.to(device)
            targets = targets.to(device)
            logits = updater(previous, evidence, actions, collisions)
            loss = _loss(logits, targets, marker_weight=marker_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_every == 0 or phase_step == steps:
                metrics = _evaluate(
                    updater,
                    validation_examples,
                    batch_size=args.batch_size,
                    marker_weight=marker_weight,
                    blocked_threshold=args.latent_map_blocked_threshold,
                    free_threshold=args.memory_free_threshold,
                    marker_threshold=args.memory_marker_threshold,
                    device=device,
                )
                entry = {
                    "phase": phase,
                    "phase_step": phase_step,
                    "step": global_step,
                    "train_loss": float(loss.item()),
                    **metrics,
                }
                logs.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)
                score = (
                    float(metrics["teacher_action_match"]),
                    -float(metrics["loss"]),
                )
                if args.save_best_action_match and score > best_score:
                    best_score = score
                    best_step = global_step
                    best_metrics = dict(metrics)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in updater.state_dict().items()
                    }
        return global_step

    total_steps = 0
    if args.transition_pretrain_steps > 0:
        total_steps = _run_phase(
            phase="transition_pretrain",
            examples=synthetic_examples,
            steps=args.transition_pretrain_steps,
            global_step=total_steps,
        )
    total_steps = _run_phase(
        phase="real_trace",
        examples=train_examples,
        steps=args.optimization_steps,
        global_step=total_steps,
    )

    final_validation = _evaluate(
        updater,
        validation_examples,
        batch_size=args.batch_size,
        marker_weight=marker_weight,
        blocked_threshold=args.latent_map_blocked_threshold,
        free_threshold=args.memory_free_threshold,
        marker_threshold=args.memory_marker_threshold,
        device=device,
    )
    selected_validation = final_validation
    selected_step = total_steps
    if args.save_best_action_match and best_state is not None:
        updater.load_state_dict(best_state)
        selected_validation = best_metrics or final_validation
        selected_step = best_step or total_steps
    report = {
        "schema": "jepa_phase3a_latent_memory_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "extra_train_data": [str(path.resolve()) for path in args.extra_train_data],
        "extra_validation_data": [
            str(path.resolve()) for path in args.extra_validation_data
        ],
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "validation_examples": int(len(validation_examples.actions)),
        "completed_steps": total_steps,
        "real_trace_steps": args.optimization_steps,
        "transition_pretrain_steps": args.transition_pretrain_steps,
        "final_validation": final_validation,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "logs": logs,
        "args": {
            key: _json_safe_arg(value)
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
            "evidence_channels": 3,
            "action_dim": len(ACTION_NAMES),
            "use_geometric_prior": not args.no_geometric_prior,
            "learned_transition_hidden_dim": (
                updater.learned_transition_hidden_dim
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "updater_state_dict": updater.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
