#!/usr/bin/env python3
"""Train a learned action head for Phase 3A value-field planning."""
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
    read_jsonl,
    step_state,
)
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AValueFieldActionHead,
    Phase3AValueFieldExtractorHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _infer_scene_seed,
    _is_egocentric_frontier_cell,
    _goal_scene_from_row,
    _latent_local_evidence,
    _select_egocentric_learned_value_field_action,
    _select_odom_frontier_action,
    _state_from_dict,
    _update_odom_frontier_memory,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _build_examples,
    _groups_from_rows,
    _load_latent_memory_updater,
    _load_value_field_head,
)


@dataclass(frozen=True)
class ActionExamples:
    memories: torch.Tensor
    target_fields: torch.Tensor
    sparse_probabilities: torch.Tensor
    actions: torch.Tensor
    sparse_labels: torch.Tensor


def _json_safe_arg(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    return value


def _concat_action_examples(*items: ActionExamples) -> ActionExamples:
    return ActionExamples(
        memories=torch.cat([item.memories for item in items], dim=0),
        target_fields=torch.cat([item.target_fields for item in items], dim=0),
        sparse_probabilities=torch.cat(
            [item.sparse_probabilities for item in items],
            dim=0,
        ),
        actions=torch.cat([item.actions for item in items], dim=0),
        sparse_labels=torch.cat([item.sparse_labels for item in items], dim=0),
    )


def _filter_action_examples(
    examples: ActionExamples,
    *,
    mode: str,
) -> ActionExamples:
    if mode == "all":
        return examples
    if mode == "broad":
        mask = ~examples.sparse_labels
    elif mode == "sparse":
        mask = examples.sparse_labels
    else:
        raise ValueError(f"unknown example filter: {mode!r}")
    if int(mask.sum().item()) == 0:
        raise SystemExit(f"example filter {mode!r} removed all examples")
    return ActionExamples(
        memories=examples.memories[mask],
        target_fields=examples.target_fields[mask],
        sparse_probabilities=examples.sparse_probabilities[mask],
        actions=examples.actions[mask],
        sparse_labels=examples.sparse_labels[mask],
    )


def _load_value_extractor_head(
    path: Path,
    *,
    fallback_memory_size: int,
    device: torch.device,
) -> tuple[Phase3AValueFieldExtractorHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    extractor = Phase3AValueFieldExtractorHead(
        memory_size=int(config.get("memory_size", fallback_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        memory_channels=int(config.get("memory_channels", 3)),
    ).to(device)
    extractor.load_state_dict(checkpoint["extractor_state_dict"])
    extractor.eval()
    return extractor, report


def _load_value_action_head(
    path: Path,
    *,
    fallback_memory_size: int,
    device: torch.device,
) -> tuple[Phase3AValueFieldActionHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    action_head = Phase3AValueFieldActionHead(
        memory_size=int(config.get("memory_size", fallback_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        memory_channels=int(config.get("memory_channels", 3)),
        action_dim=int(config.get("action_dim", len(ACTION_NAMES))),
    ).to(device)
    action_head.load_state_dict(checkpoint["action_head_state_dict"])
    action_head.eval()
    return action_head, report


def _action_memory_input(
    recurrent_memory: torch.Tensor,
    local_evidence: torch.Tensor,
    *,
    include_local_evidence_channels: bool,
) -> torch.Tensor:
    if include_local_evidence_channels:
        return torch.cat([recurrent_memory, local_evidence], dim=1)
    return recurrent_memory


def _pad_local_evidence_channels(memories: torch.Tensor) -> torch.Tensor:
    if int(memories.shape[1]) != 3:
        return memories
    return torch.cat([memories, torch.zeros_like(memories)], dim=1)


@torch.no_grad()
def _action_head_rollout_action(
    action_head: Phase3AValueFieldActionHead,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    recurrent_memory: torch.Tensor,
    action_memory: torch.Tensor,
    memory_dict: dict,
) -> str:
    target_fields = value_field_head(recurrent_memory).sigmoid()
    marker = memory_dict.get("marker")
    if (
        int(target_fields.shape[1]) >= 2
        and marker is not None
        and marker in memory_dict["free"]
    ):
        target_field = target_fields[:, 0:1]
    elif int(target_fields.shape[1]) >= 2:
        target_field = target_fields[:, 1:2]
    else:
        target_field = target_fields[:, 0:1]
    sparse_probability = extractor_head(recurrent_memory).sigmoid()
    logits = action_head(action_memory, target_field, sparse_probability)
    return ACTION_NAMES[int(logits.argmax(dim=1).item())]


@torch.no_grad()
def _build_rollout_memories(
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
    rollout_action_head: Phase3AValueFieldActionHead,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    include_marker_start_groups: bool,
    include_local_evidence_channels: bool,
    device: torch.device,
) -> torch.Tensor:
    groups = _groups_from_rows(
        rows,
        include_marker_start_groups=include_marker_start_groups,
    )
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
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
            action_memory = _action_memory_input(
                recurrent_memory,
                local_evidence,
                include_local_evidence_channels=include_local_evidence_channels,
            )
            memories.append(action_memory[0].detach().cpu())
            rollout_action = _action_head_rollout_action(
                rollout_action_head,
                value_field_head,
                extractor_head,
                recurrent_memory,
                action_memory,
                memory_dict,
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
                        "built_dagger_groups": group_index + 1,
                        "dagger_examples": len(memories),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not memories:
        raise SystemExit("no DAgger rollout memories were generated")
    return torch.stack(memories)


@torch.no_grad()
def _build_trace_memories(
    trace_paths: list[Path],
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    latent_map_marker_threshold: float,
    failed_only: bool,
    post_latent_marker_only: bool,
    include_local_evidence_channels: bool,
    repeat: int,
    device: torch.device,
) -> torch.Tensor:
    groups = _groups_from_rows(rows, include_marker_start_groups=False)
    memories = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        episodes = trace.get("episodes", [])
        for episode_index, episode in enumerate(episodes):
            if episode_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has episode index {episode_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            if failed_only and bool(episode.get("claimed", False)):
                continue
            group = groups[episode_index]
            template = group[0]
            scene = _goal_scene_from_row(
                template,
                seed=scene_seed,
                width=width,
                height=height,
            )
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
            latent_marker_seen_ever = False
            for item in episode.get("trajectory", []):
                state = _state_from_dict(item["state"])
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
                latent_marker_seen_ever = latent_marker_seen_ever or (
                    float(local_evidence[0, 2].max().detach().cpu())
                    >= latent_map_marker_threshold
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
                if not (post_latent_marker_only and not latent_marker_seen_ever):
                    action_memory = _action_memory_input(
                        recurrent_memory,
                        local_evidence,
                        include_local_evidence_channels=(
                            include_local_evidence_channels
                        ),
                    )
                    memories.append(action_memory[0].detach().cpu())
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not memories:
        raise SystemExit("no trace memories were generated")
    stacked = torch.stack(memories)
    if repeat > 1:
        stacked = stacked.repeat((repeat, 1, 1, 1))
    return stacked


@torch.no_grad()
def _build_trace_action_examples(
    trace_paths: list[Path],
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    latent_map_marker_threshold: float,
    fixed_marker_targets: bool,
    teacher_source: str,
    failed_only: bool,
    post_latent_marker_only: bool,
    current_marker_only: bool,
    marker_return_only: bool,
    repeat: int,
    include_local_evidence_channels: bool,
    device: torch.device,
) -> ActionExamples:
    groups = _groups_from_rows(rows, include_marker_start_groups=False)
    memories = []
    target_fields = []
    sparse_probabilities = []
    actions = []
    sparse_labels = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        episodes = trace.get("episodes", [])
        for episode_index, episode in enumerate(episodes):
            if episode_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has episode index {episode_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            if failed_only and bool(episode.get("claimed", False)):
                continue
            group = groups[episode_index]
            template = group[0]
            scene = _goal_scene_from_row(
                template,
                seed=scene_seed,
                width=width,
                height=height,
            )
            recurrent_memory = torch.zeros(
                1,
                3,
                memory_size,
                memory_size,
                dtype=torch.float32,
                device=device,
            )
            odom_frontier_memory = {
                "free": set(),
                "blocked": set(),
                "marker": None,
                "radius": max(memory_size // 2, 0),
            }
            last_action = "hold"
            last_collision = False
            latent_marker_seen_ever = False
            for item in episode.get("trajectory", []):
                state = _state_from_dict(item["state"])
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
                current_marker_seen = (
                    float(local_evidence[0, 2].max().detach().cpu())
                    >= latent_map_marker_threshold
                )
                latent_marker_seen_ever = latent_marker_seen_ever or current_marker_seen
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
                if teacher_source == "odom_frontier":
                    _update_odom_frontier_memory(
                        odom_frontier_memory,
                        scene=scene,
                        state=state,
                        view_size=view_size,
                        current_goal_marker=True,
                    )
                    target_action = _select_odom_frontier_action(
                        odom_frontier_memory,
                        state,
                    )
                elif teacher_source == "local_oracle":
                    target_action = str(item.get("oracle_action", ""))
                    if target_action not in ACTION_INDEX:
                        raise SystemExit(
                            f"{trace_path} episode {episode_index} step "
                            f"{item.get('step')} has invalid oracle_action "
                            f"{target_action!r}"
                        )
                else:
                    raise ValueError(
                        f"unknown trace action teacher source: {teacher_source!r}"
                    )
                selection_mode = str(item.get("selection_mode", ""))
                if not (
                    (post_latent_marker_only and not latent_marker_seen_ever)
                    or (current_marker_only and not current_marker_seen)
                    or (
                        marker_return_only
                        and "marker_action_return" not in selection_mode
                    )
                ):
                    memory_dict = _egocentric_memory_tensor_to_dict(
                        recurrent_memory[0].detach().cpu(),
                        blocked_threshold=blocked_threshold,
                        free_threshold=free_threshold,
                        marker_threshold=marker_threshold,
                    )
                    target_fields_gpu = value_field_head(recurrent_memory).sigmoid()
                    marker = memory_dict.get("marker")
                    if (
                        int(target_fields_gpu.shape[1]) >= 2
                        and marker is not None
                        and marker in memory_dict["free"]
                    ):
                        target_probs_gpu = target_fields_gpu[:, 0:1]
                    elif int(target_fields_gpu.shape[1]) >= 2:
                        target_probs_gpu = target_fields_gpu[:, 1:2]
                    else:
                        target_probs_gpu = target_fields_gpu[:, 0:1]
                    if (
                        fixed_marker_targets
                        and marker is not None
                        and marker in memory_dict["free"]
                    ):
                        row = memory_size // 2 - int(marker[0])
                        col = memory_size // 2 + int(marker[1])
                        if 0 <= row < memory_size and 0 <= col < memory_size:
                            fixed_target = torch.zeros_like(target_probs_gpu)
                            fixed_target[:, :, row, col] = 1.0
                            target_probs_gpu = fixed_target
                    sparse_prob = extractor_head(recurrent_memory).sigmoid()
                    action_memory = _action_memory_input(
                        recurrent_memory,
                        local_evidence,
                        include_local_evidence_channels=(
                            include_local_evidence_channels
                        ),
                    )
                    memories.append(action_memory[0].detach().cpu())
                    target_fields.append(target_probs_gpu[0].detach().cpu())
                    sparse_probabilities.append(float(sparse_prob.item()))
                    actions.append(ACTION_INDEX[target_action])
                    sparse_labels.append(True)
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not actions:
        raise SystemExit("no trace action examples were generated")
    examples = ActionExamples(
        memories=torch.stack(memories),
        target_fields=torch.stack(target_fields),
        sparse_probabilities=torch.tensor(sparse_probabilities, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=torch.tensor(sparse_labels, dtype=torch.bool),
    )
    if repeat <= 1:
        return examples
    return ActionExamples(
        memories=examples.memories.repeat((repeat, 1, 1, 1)),
        target_fields=examples.target_fields.repeat((repeat, 1, 1, 1)),
        sparse_probabilities=examples.sparse_probabilities.repeat(repeat),
        actions=examples.actions.repeat(repeat),
        sparse_labels=examples.sparse_labels.repeat(repeat),
    )


@torch.no_grad()
def _build_action_examples(
    memories: torch.Tensor,
    *,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    fixed_marker_targets: bool,
    batch_size: int,
    device: torch.device,
) -> ActionExamples:
    dataset = TensorDataset(memories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_memories = []
    all_targets = []
    all_sparse_probs = []
    actions = []
    sparse_labels = []
    for (memory_batch,) in loader:
        memory_batch = memory_batch.to(device)
        recurrent_batch = memory_batch[:, :3]
        target_fields = value_field_head(recurrent_batch).sigmoid()
        sparse_probs = extractor_head(recurrent_batch).sigmoid()
        memory_cpu = memory_batch.detach().cpu()
        recurrent_cpu = recurrent_batch.detach().cpu()
        targets_cpu = target_fields.detach().cpu()
        sparse_cpu = sparse_probs.detach().cpu()
        for item_index in range(int(memory_cpu.shape[0])):
            memory_dict = _egocentric_memory_tensor_to_dict(
                recurrent_cpu[item_index],
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            marker = memory_dict.get("marker")
            if (
                int(targets_cpu.shape[1]) >= 2
                and marker is not None
                and marker in memory_dict["free"]
            ):
                channel = 0
            elif int(targets_cpu.shape[1]) >= 2:
                channel = 1
            else:
                channel = 0
            sparse = float(sparse_cpu[item_index].item()) >= extractor_threshold
            top_k = sparse_target_top_k if sparse else target_top_k
            target_probs = targets_cpu[item_index, channel]
            fixed_marker_target = (
                fixed_marker_targets
                and marker is not None
                and marker in memory_dict["free"]
            )
            if fixed_marker_target:
                memory_size = int(target_probs.shape[0])
                radius = memory_size // 2
                row = radius - int(marker[0])
                col = radius + int(marker[1])
                if 0 <= row < memory_size and 0 <= col < memory_size:
                    target_probs = torch.zeros_like(target_probs)
                    target_probs[row, col] = 1.0
            action, _mode = _select_egocentric_learned_value_field_action(
                memory_dict,
                target_probs,
                threshold=target_threshold,
                top_k=top_k,
                fixed_marker_target=fixed_marker_target,
            )
            all_memories.append(memory_cpu[item_index])
            all_targets.append(target_probs.view(1, *target_probs.shape))
            all_sparse_probs.append(sparse_cpu[item_index].view(1))
            actions.append(ACTION_INDEX[action])
            sparse_labels.append(sparse)
    return ActionExamples(
        memories=torch.stack(all_memories),
        target_fields=torch.stack(all_targets),
        sparse_probabilities=torch.stack(all_sparse_probs).view(-1),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=torch.tensor(sparse_labels, dtype=torch.bool),
    )


def _memory_tensor_from_dict(memory: dict, *, memory_size: int) -> torch.Tensor:
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
        row = radius - int(marker[0])
        col = radius + int(marker[1])
        if 0 <= row < memory_size and 0 <= col < memory_size:
            tensor[2, row, col] = 1.0
    return tensor


def _target_tensor_from_cells(
    cells: set[tuple[int, int]],
    *,
    memory_size: int,
) -> torch.Tensor:
    radius = memory_size // 2
    tensor = torch.zeros(1, memory_size, memory_size, dtype=torch.float32)
    for ahead, lateral in cells:
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            tensor[0, row, col] = 1.0
    return tensor


def _synthetic_memory(rng: random.Random, *, memory_size: int) -> dict:
    radius = memory_size // 2
    free = {(0, 0)}
    blocked: set[tuple[int, int]] = set()
    current = (0, 0)
    for _ in range(rng.randint(6, 32)):
        neighbors = [
            item
            for item in (
                (current[0] + 1, current[1]),
                (current[0] - 1, current[1]),
                (current[0], current[1] + 1),
                (current[0], current[1] - 1),
            )
            if abs(item[0]) <= radius and abs(item[1]) <= radius
        ]
        current = rng.choice(neighbors)
        free.add(current)
    for cell in list(free):
        for neighbor in (
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ):
            if (
                abs(neighbor[0]) <= radius
                and abs(neighbor[1]) <= radius
                and neighbor not in free
                and rng.random() < 0.22
            ):
                blocked.add(neighbor)
    blocked.discard((0, 0))
    return {
        "free": free,
        "blocked": blocked,
        "marker": None,
        "radius": radius,
    }


def _build_synthetic_action_examples(
    count: int,
    *,
    memory_size: int,
    target_threshold: float,
    target_top_k: int,
    sparse_target_top_k: int,
    fixed_marker_targets: bool,
    seed: int,
) -> ActionExamples:
    rng = random.Random(seed)
    memories = []
    targets = []
    sparse_probs = []
    actions = []
    sparse_labels = []
    attempts = 0
    while len(actions) < count:
        attempts += 1
        if attempts > count * 20:
            raise SystemExit("could not build enough synthetic action examples")
        memory = _synthetic_memory(rng, memory_size=memory_size)
        marker_mode = rng.random() < 0.45 and len(memory["free"]) > 1
        if marker_mode:
            marker = rng.choice(sorted(memory["free"] - {(0, 0)}))
            memory["marker"] = marker
            target_cells = {marker}
            top_k = sparse_target_top_k
            sparse = True
        else:
            target_cells = {
                cell
                for cell in memory["free"]
                if _is_egocentric_frontier_cell(memory, cell)
            }
            if not target_cells:
                continue
            top_k = target_top_k
            sparse = False
        target = _target_tensor_from_cells(target_cells, memory_size=memory_size)
        action, _mode = _select_egocentric_learned_value_field_action(
            memory,
            target[0],
            threshold=target_threshold,
            top_k=top_k,
            fixed_marker_target=bool(fixed_marker_targets and marker_mode),
        )
        memories.append(_memory_tensor_from_dict(memory, memory_size=memory_size))
        targets.append(target)
        sparse_probs.append(1.0 if sparse else 0.0)
        actions.append(ACTION_INDEX[action])
        sparse_labels.append(sparse)
    return ActionExamples(
        memories=torch.stack(memories),
        target_fields=torch.stack(targets),
        sparse_probabilities=torch.tensor(sparse_probs, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=torch.tensor(sparse_labels, dtype=torch.bool),
    )


@torch.no_grad()
def _evaluate(
    action_head: Phase3AValueFieldActionHead,
    examples: ActionExamples,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    action_head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.target_fields,
        examples.sparse_probabilities,
        examples.actions,
        examples.sparse_labels,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    matches = 0
    sparse_total = 0
    sparse_matches = 0
    broad_total = 0
    broad_matches = 0
    counts = {name: 0 for name in ACTION_NAMES}
    pred_counts = {name: 0 for name in ACTION_NAMES}
    for memory, target, sparse_prob, action, sparse_label in loader:
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        action = action.to(device)
        logits = action_head(memory, target, sparse_prob)
        loss = F.cross_entropy(logits, action)
        pred = logits.argmax(dim=1)
        match = pred == action
        total += int(memory.shape[0])
        loss_total += float(loss.item()) * int(memory.shape[0])
        matches += int(match.sum().item())
        for action_index in action.detach().cpu().tolist():
            counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in pred.detach().cpu().tolist():
            pred_counts[ACTION_NAMES[int(action_index)]] += 1
        sparse_cpu = sparse_label.detach().cpu()
        match_cpu = match.detach().cpu()
        sparse_total += int(sparse_cpu.sum().item())
        sparse_matches += int((match_cpu & sparse_cpu).sum().item())
        broad_mask = ~sparse_cpu
        broad_total += int(broad_mask.sum().item())
        broad_matches += int((match_cpu & broad_mask).sum().item())
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "action_match": matches / max(total, 1),
        "sparse_examples": sparse_total,
        "sparse_action_match": sparse_matches / max(sparse_total, 1),
        "broad_examples": broad_total,
        "broad_action_match": broad_matches / max(broad_total, 1),
        "action_counts": counts,
        "predicted_action_counts": pred_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
    parser.add_argument("--latent-value-field-head", type=Path, required=True)
    parser.add_argument("--latent-value-extractor-head", type=Path, required=True)
    parser.add_argument("--dagger-rollout-action-head", type=Path, default=None)
    parser.add_argument(
        "--init-action-checkpoint",
        type=Path,
        default=None,
        help="optional value-action checkpoint to initialize from",
    )
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.5)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--sparse-target-top-k", type=int, default=1)
    parser.add_argument("--fixed-marker-targets", action="store_true")
    parser.add_argument(
        "--example-filter",
        choices=("all", "broad", "sparse"),
        default="all",
    )
    parser.add_argument(
        "--trace-memory-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "closed-loop trace JSON to replay into action-head training "
            "examples; may be passed multiple times"
        ),
    )
    parser.add_argument(
        "--trace-memory-source",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--trace-memory-failed-only", action="store_true")
    parser.add_argument("--trace-memory-post-latent-marker-only", action="store_true")
    parser.add_argument("--trace-memory-current-marker-only", action="store_true")
    parser.add_argument("--trace-memory-marker-return-only", action="store_true")
    parser.add_argument("--trace-memory-repeat", type=int, default=1)
    parser.add_argument(
        "--include-local-evidence-channels",
        action="store_true",
        help=(
            "concatenate centered learned local evidence channels to recurrent "
            "memory before the action head"
        ),
    )
    parser.add_argument(
        "--trace-action-teacher-source",
        choices=("none", "odom_frontier", "local_oracle"),
        default="none",
    )
    parser.add_argument("--synthetic-examples", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=20260656)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260655)
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
    if not 0.0 <= args.latent_map_marker_threshold <= 1.0:
        raise SystemExit("--latent-map-marker-threshold must be in [0, 1]")
    if not 0.0 <= args.target_threshold <= 1.0:
        raise SystemExit("--target-threshold must be in [0, 1]")
    if args.target_top_k < 1:
        raise SystemExit("--target-top-k must be positive")
    if not 0.0 <= args.extractor_threshold <= 1.0:
        raise SystemExit("--extractor-threshold must be in [0, 1]")
    if args.sparse_target_top_k < 1:
        raise SystemExit("--sparse-target-top-k must be positive")
    if args.synthetic_examples < 0:
        raise SystemExit("--synthetic-examples must be non-negative")
    if args.trace_memory_repeat < 1:
        raise SystemExit("--trace-memory-repeat must be positive")
    if args.trace_action_teacher_source != "none" and not args.trace_memory_data:
        raise SystemExit(
            "--trace-action-teacher-source requires --trace-memory-data"
        )
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit("--label-smoothing must be in [0, 1)")

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
    dagger_rollout_action_head = None
    dagger_rollout_action_report = None
    init_action_report = None
    if args.dagger_rollout_action_head is not None:
        dagger_rollout_action_head, dagger_rollout_action_report = (
            _load_value_action_head(
                args.dagger_rollout_action_head,
                fallback_memory_size=args.memory_size,
                device=device,
            )
        )
    for name, size in (
        ("latent memory updater", latent_memory_updater.memory_size),
        ("value field head", value_field_head.memory_size),
        ("value extractor head", extractor_head.memory_size),
    ):
        if int(size) != int(args.memory_size):
            raise SystemExit(f"--memory-size must match {name} size ({size})")
    if (
        dagger_rollout_action_head is not None
        and int(dagger_rollout_action_head.memory_size) != int(args.memory_size)
    ):
        raise SystemExit(
            "--memory-size must match DAgger rollout action head size "
            f"({dagger_rollout_action_head.memory_size})"
        )

    train_memory_examples = _build_examples(
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
        output_channels=1,
        include_marker_start_groups=args.include_marker_start_train_groups,
        rollout_value_field_head=None,
        rollout_target_threshold=args.target_threshold,
        rollout_target_top_k=args.target_top_k,
        rollout_fixed_marker_target=False,
        device=device,
    )
    if args.include_local_evidence_channels:
        train_memory_examples = type(train_memory_examples)(
            memories=_pad_local_evidence_channels(train_memory_examples.memories),
            targets=train_memory_examples.targets,
            actions=train_memory_examples.actions,
            marker_targets=train_memory_examples.marker_targets,
        )
    if dagger_rollout_action_head is not None:
        dagger_memories = _build_rollout_memories(
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
            rollout_action_head=dagger_rollout_action_head,
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            include_marker_start_groups=args.include_marker_start_train_groups,
            include_local_evidence_channels=bool(args.include_local_evidence_channels),
            device=device,
        )
        train_memory_examples = type(train_memory_examples)(
            memories=torch.cat([train_memory_examples.memories, dagger_memories], dim=0),
            targets=torch.cat(
                [
                    train_memory_examples.targets,
                    torch.zeros(
                        len(dagger_memories),
                        *train_memory_examples.targets.shape[1:],
                        dtype=train_memory_examples.targets.dtype,
                    ),
                ],
                dim=0,
            ),
            actions=torch.cat(
                [
                    train_memory_examples.actions,
                    torch.zeros(len(dagger_memories), dtype=train_memory_examples.actions.dtype),
                ],
                dim=0,
            ),
            marker_targets=torch.cat(
                [
                    train_memory_examples.marker_targets,
                    torch.zeros(len(dagger_memories), dtype=torch.bool),
                ],
                dim=0,
            ),
        )
    train_memories = train_memory_examples.memories
    trace_memory_count = 0
    if args.trace_memory_data and args.trace_action_teacher_source == "none":
        trace_rows = train_rows if args.trace_memory_source == "train" else validation_rows
        trace_seed = train_seed if args.trace_memory_source == "train" else validation_seed
        trace_memories = _build_trace_memories(
            list(args.trace_memory_data),
            trace_rows,
            scene_seed=trace_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            latent_map_marker_threshold=args.latent_map_marker_threshold,
            failed_only=bool(args.trace_memory_failed_only),
            post_latent_marker_only=bool(args.trace_memory_post_latent_marker_only),
            include_local_evidence_channels=bool(args.include_local_evidence_channels),
            repeat=int(args.trace_memory_repeat),
            device=device,
        )
        trace_memory_count = int(trace_memories.shape[0])
        train_memories = torch.cat([train_memories, trace_memories], dim=0)
    validation_memory_examples = _build_examples(
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
        output_channels=1,
        include_marker_start_groups=False,
        rollout_value_field_head=None,
        rollout_target_threshold=args.target_threshold,
        rollout_target_top_k=args.target_top_k,
        rollout_fixed_marker_target=False,
        device=device,
    )
    if args.include_local_evidence_channels:
        validation_memory_examples = type(validation_memory_examples)(
            memories=_pad_local_evidence_channels(validation_memory_examples.memories),
            targets=validation_memory_examples.targets,
            actions=validation_memory_examples.actions,
            marker_targets=validation_memory_examples.marker_targets,
        )
    train_examples = _build_action_examples(
        train_memories,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_threshold=args.target_threshold,
        target_top_k=args.target_top_k,
        extractor_threshold=args.extractor_threshold,
        sparse_target_top_k=args.sparse_target_top_k,
        fixed_marker_targets=bool(args.fixed_marker_targets),
        batch_size=args.batch_size,
        device=device,
    )
    validation_examples = _build_action_examples(
        validation_memory_examples.memories,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_threshold=args.target_threshold,
        target_top_k=args.target_top_k,
        extractor_threshold=args.extractor_threshold,
        sparse_target_top_k=args.sparse_target_top_k,
        fixed_marker_targets=bool(args.fixed_marker_targets),
        batch_size=args.batch_size,
        device=device,
    )
    if args.synthetic_examples > 0:
        synthetic_examples = _build_synthetic_action_examples(
            args.synthetic_examples,
            memory_size=args.memory_size,
            target_threshold=args.target_threshold,
            target_top_k=args.target_top_k,
            sparse_target_top_k=args.sparse_target_top_k,
            fixed_marker_targets=bool(args.fixed_marker_targets),
            seed=args.synthetic_seed,
        )
        train_examples = _concat_action_examples(train_examples, synthetic_examples)
    train_examples = _filter_action_examples(
        train_examples,
        mode=str(args.example_filter),
    )
    validation_examples = _filter_action_examples(
        validation_examples,
        mode=str(args.example_filter),
    )
    trace_action_count = 0
    if args.trace_action_teacher_source != "none":
        trace_rows = train_rows if args.trace_memory_source == "train" else validation_rows
        trace_seed = train_seed if args.trace_memory_source == "train" else validation_seed
        trace_action_examples = _build_trace_action_examples(
            list(args.trace_memory_data),
            trace_rows,
            scene_seed=trace_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            value_field_head=value_field_head,
            extractor_head=extractor_head,
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            latent_map_marker_threshold=args.latent_map_marker_threshold,
            fixed_marker_targets=bool(args.fixed_marker_targets),
            teacher_source=str(args.trace_action_teacher_source),
            failed_only=bool(args.trace_memory_failed_only),
            post_latent_marker_only=bool(
                args.trace_memory_post_latent_marker_only
            ),
            current_marker_only=bool(args.trace_memory_current_marker_only),
            marker_return_only=bool(args.trace_memory_marker_return_only),
            repeat=int(args.trace_memory_repeat),
            include_local_evidence_channels=bool(args.include_local_evidence_channels),
            device=device,
        )
        trace_action_count = int(len(trace_action_examples.actions))
        train_examples = _concat_action_examples(
            train_examples,
            trace_action_examples,
        )

    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.target_fields,
        train_examples.sparse_probabilities,
        train_examples.actions,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    action_memory_channels = 6 if args.include_local_evidence_channels else 3
    action_head = Phase3AValueFieldActionHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=action_memory_channels,
        action_dim=len(ACTION_NAMES),
    ).to(device)
    if args.init_action_checkpoint is not None:
        try:
            init_checkpoint = torch.load(
                args.init_action_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            init_checkpoint = torch.load(
                args.init_action_checkpoint,
                map_location=device,
            )
        init_action_report = init_checkpoint.get("report", {})
        init_config = init_action_report.get("model_config", {})
        expected = {
            "memory_size": int(args.memory_size),
            "hidden_dim": int(args.hidden_dim),
            "memory_channels": int(action_memory_channels),
            "action_dim": len(ACTION_NAMES),
        }
        actual = {
            "memory_size": int(init_config.get("memory_size", args.memory_size)),
            "hidden_dim": int(init_config.get("hidden_dim", args.hidden_dim)),
            "memory_channels": int(init_config.get("memory_channels", 3)),
            "action_dim": int(init_config.get("action_dim", len(ACTION_NAMES))),
        }
        if actual != expected:
            raise SystemExit(
                "--init-action-checkpoint config does not match requested "
                f"model config ({actual} != {expected})"
            )
        action_head.load_state_dict(init_checkpoint["action_head_state_dict"])
    optimizer = torch.optim.AdamW(
        action_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = (-1.0, -1.0, float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, target, sparse_prob, action = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, sparse_prob, action = next(iterator)
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        action = action.to(device)
        logits = action_head(memory, target, sparse_prob)
        loss = F.cross_entropy(
            logits,
            action,
            label_smoothing=float(args.label_smoothing),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                action_head,
                validation_examples,
                batch_size=args.batch_size,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["action_match"]),
                float(metrics["sparse_action_match"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in action_head.state_dict().items()
                }

    final_validation = _evaluate(
        action_head,
        validation_examples,
        batch_size=args.batch_size,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        action_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation

    report = {
        "schema": "jepa_phase3a_value_action_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "latent_value_field_head": str(args.latent_value_field_head.resolve()),
        "latent_value_field_completed_steps": value_field_report.get("completed_steps"),
        "latent_value_extractor_head": str(args.latent_value_extractor_head.resolve()),
        "latent_value_extractor_completed_steps": extractor_report.get("completed_steps"),
        "init_action_checkpoint": (
            str(args.init_action_checkpoint.resolve())
            if args.init_action_checkpoint
            else None
        ),
        "init_action_completed_steps": (
            init_action_report.get("completed_steps")
            if init_action_report
            else None
        ),
        "init_action_selected_step": (
            init_action_report.get("selected_step")
            if init_action_report
            else None
        ),
        "dagger_rollout_action_head": (
            str(args.dagger_rollout_action_head.resolve())
            if args.dagger_rollout_action_head
            else None
        ),
        "dagger_rollout_action_completed_steps": (
            dagger_rollout_action_report.get("completed_steps")
            if dagger_rollout_action_report
            else None
        ),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "trace_memory_data": [
            str(path.resolve()) for path in args.trace_memory_data
        ],
        "trace_memory_source": str(args.trace_memory_source),
        "trace_memory_failed_only": bool(args.trace_memory_failed_only),
        "trace_memory_post_latent_marker_only": bool(
            args.trace_memory_post_latent_marker_only
        ),
        "trace_memory_current_marker_only": bool(
            args.trace_memory_current_marker_only
        ),
        "trace_memory_marker_return_only": bool(
            args.trace_memory_marker_return_only
        ),
        "trace_memory_repeat": int(args.trace_memory_repeat),
        "trace_memory_examples": int(trace_memory_count),
        "trace_action_teacher_source": str(args.trace_action_teacher_source),
        "trace_action_examples": int(trace_action_count),
        "include_local_evidence_channels": bool(args.include_local_evidence_channels),
        "train_synthetic_examples": int(args.synthetic_examples),
        "fixed_marker_targets": bool(args.fixed_marker_targets),
        "example_filter": str(args.example_filter),
        "validation_examples": int(len(validation_examples.actions)),
        "completed_steps": args.optimization_steps,
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
            "memory_channels": int(action_memory_channels),
            "action_dim": len(ACTION_NAMES),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_head_state_dict": action_head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
