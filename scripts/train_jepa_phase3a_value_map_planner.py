#!/usr/bin/env python3
"""Train a dense value-map planner for Phase 3A recurrent navigation."""
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
    Phase3AValueMapPlannerHead,
    Phase3AValueMapRouterHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _break_turn_oscillation_action,
    _candidate_rows,
    _center_local_evidence,
    _egocentric_neighbors,
    _egocentric_has_contiguous_side_wall,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _infer_scene_seed,
    _is_egocentric_frontier_cell,
    _latent_local_evidence,
    _select_egocentric_learned_value_map_action,
    _select_odom_frontier_lookahead_action,
    _selection_for_single_action,
    _state_from_dict,
    _update_odom_frontier_memory,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_action import (  # noqa: E402
    _build_synthetic_action_examples,
    _load_value_extractor_head,
    _memory_tensor_from_dict,
    _synthetic_memory,
)
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _build_examples,
    _load_latent_memory_updater,
    _load_value_field_head,
)


@dataclass(frozen=True)
class ValueMapExamples:
    memories: torch.Tensor
    target_fields: torch.Tensor
    sparse_probabilities: torch.Tensor
    value_maps: torch.Tensor
    actions: torch.Tensor
    sparse_labels: torch.Tensor


@dataclass(frozen=True)
class RouterExamples:
    memories: torch.Tensor
    route_labels: torch.Tensor


def _concat_value_map_examples(*items: ValueMapExamples) -> ValueMapExamples:
    return ValueMapExamples(
        memories=torch.cat([item.memories for item in items], dim=0),
        target_fields=torch.cat([item.target_fields for item in items], dim=0),
        sparse_probabilities=torch.cat(
            [item.sparse_probabilities for item in items],
            dim=0,
        ),
        value_maps=torch.cat([item.value_maps for item in items], dim=0),
        actions=torch.cat([item.actions for item in items], dim=0),
        sparse_labels=torch.cat([item.sparse_labels for item in items], dim=0),
    )


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def _concat_router_examples(*items: RouterExamples) -> RouterExamples:
    return RouterExamples(
        memories=torch.cat([item.memories for item in items], dim=0),
        route_labels=torch.cat([item.route_labels for item in items], dim=0),
    )


def _router_examples_from_memories(
    memories: torch.Tensor,
    *,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
) -> RouterExamples:
    labels = []
    for memory_tensor in memories:
        memory = _egocentric_memory_tensor_to_dict(
            memory_tensor,
            blocked_threshold=blocked_threshold,
            free_threshold=free_threshold,
            marker_threshold=marker_threshold,
        )
        labels.append(float(_egocentric_has_contiguous_side_wall(memory)))
    return RouterExamples(
        memories=memories,
        route_labels=torch.tensor(labels, dtype=torch.float32),
    )


def _without_side_wall(memory: dict) -> dict:
    clean = _copy_memory(memory)
    for sign in (-1, 1):
        if all((0, sign * offset) in clean["blocked"] for offset in range(1, 4)):
            clean["blocked"].discard((0, sign * 2))
            clean["free"].add((0, sign * 2))
    return clean


def _with_side_wall(memory: dict, *, rng: random.Random) -> dict:
    side = rng.choice((-1, 1))
    routed = _copy_memory(memory)
    routed["free"].add((0, 0))
    for offset in range(1, 4):
        cell = (0, side * offset)
        routed["blocked"].add(cell)
        routed["free"].discard(cell)
    return routed


def _build_synthetic_router_examples(
    count: int,
    *,
    memory_size: int,
    seed: int,
) -> RouterExamples:
    if count <= 0:
        return RouterExamples(
            memories=torch.empty(0, 3, memory_size, memory_size, dtype=torch.float32),
            route_labels=torch.empty(0, dtype=torch.float32),
        )
    rng = random.Random(seed)
    memories = []
    labels = []
    attempts = 0
    while len(labels) < count:
        attempts += 1
        if attempts > count * 40:
            raise SystemExit("could not build enough synthetic router examples")
        memory = _synthetic_memory(rng, memory_size=memory_size)
        positive = len(labels) % 2 == 0
        memory = _with_side_wall(memory, rng=rng) if positive else _without_side_wall(memory)
        label = _egocentric_has_contiguous_side_wall(memory)
        if bool(label) != bool(positive):
            continue
        memories.append(_memory_tensor_from_dict(memory, memory_size=memory_size))
        labels.append(float(label))
    return RouterExamples(
        memories=torch.stack(memories),
        route_labels=torch.tensor(labels, dtype=torch.float32),
    )


@torch.no_grad()
def _build_counterfactual_router_examples(
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_episodes: int | None,
    max_steps: int,
    horizon: int,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    primary_planner_head: Phase3AValueMapPlannerHead,
    primary_ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    primary_ensemble_mode: str,
    fallback_planner_head: Phase3AValueMapPlannerHead,
    fallback_after_step: int | None,
    utility_margin: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    include_marker_start_groups: bool,
    device: torch.device,
) -> RouterExamples:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

    groups = _groups_from_rows(
        rows,
        include_marker_start_groups=include_marker_start_groups,
    )
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    labels = []
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
        history_states = [_state_from_dict(item) for item in template["history_states"]]
        history_actions = [str(item) for item in template["history_primitive_sequence"]]
        for step in range(max_steps):
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
            primary_action = _value_map_rollout_action(
                primary_planner_head,
                primary_ensemble_heads,
                primary_ensemble_mode,
                value_field_head,
                extractor_head,
                recurrent_memory,
                memory_dict,
            )
            fallback_action = _value_map_rollout_action(
                fallback_planner_head,
                (),
                primary_ensemble_mode,
                value_field_head,
                extractor_head,
                recurrent_memory,
                memory_dict,
            )
            use_fallback = False
            if primary_action != fallback_action and memory_dict.get("marker") is None:
                candidate_rows = _candidate_rows(
                    scene=scene,
                    source_index=group_index,
                    state=state,
                    history_states=history_states,
                    history_actions=history_actions,
                    horizon=horizon,
                    view_size=view_size,
                    current_goal_marker=True,
                )
                primary_selected, _primary_oracle = _selection_for_single_action(
                    candidate_rows,
                    primary_action,
                )
                fallback_selected, _fallback_oracle = _selection_for_single_action(
                    candidate_rows,
                    fallback_action,
                )
                use_fallback = (
                    float(fallback_selected["utility"])
                    > float(primary_selected["utility"]) + float(utility_margin)
                )
            memories.append(recurrent_memory[0].detach().cpu())
            labels.append(float(use_fallback))
            rollout_action = fallback_action if use_fallback else primary_action
            if (
                not use_fallback
                and fallback_after_step is not None
                and step >= fallback_after_step
            ):
                rollout_action = fallback_action
            next_state, collision = step_state(scene, state, rollout_action)
            history_states.append(state)
            history_actions.append(rollout_action)
            last_action = rollout_action
            last_collision = bool(collision)
            state = next_state
            if (state.x, state.y) == scene.goal:
                break
        if (group_index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "built_counterfactual_router_groups": group_index + 1,
                        "router_examples": len(labels),
                        "router_positive_examples": int(sum(labels)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not memories:
        raise SystemExit("no counterfactual router examples were generated")
    return RouterExamples(
        memories=torch.stack(memories),
        route_labels=torch.tensor(labels, dtype=torch.float32),
    )


@torch.no_grad()
def _evaluate_router(
    router_head: Phase3AValueMapRouterHead,
    examples: RouterExamples,
    *,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> dict:
    dataset = TensorDataset(examples.memories, examples.route_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    positives = 0
    negatives = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    loss_total = 0.0
    for memory, label in loader:
        memory = memory.to(device)
        label = label.to(device)
        logits = router_head(memory)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        probs = logits.sigmoid()
        predicted = probs >= threshold
        expected = label >= 0.5
        total += int(label.numel())
        positives += int(expected.sum().item())
        negatives += int((~expected).sum().item())
        correct += int((predicted == expected).sum().item())
        true_positive += int((predicted & expected).sum().item())
        false_positive += int((predicted & ~expected).sum().item())
        false_negative += int((~predicted & expected).sum().item())
        loss_total += float(loss.item()) * int(label.numel())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "examples": total,
        "positive_examples": positives,
        "negative_examples": negatives,
        "loss": loss_total / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _train_router_head(
    train_examples: RouterExamples,
    validation_examples: RouterExamples,
    *,
    memory_size: int,
    hidden_dim: int,
    optimization_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    positive_weight: float | None,
    threshold: float,
    save_best: bool,
    log_every: int,
    device: torch.device,
) -> tuple[Phase3AValueMapRouterHead, dict]:
    router_head = Phase3AValueMapRouterHead(
        memory_size=memory_size,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router_head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_loader = DataLoader(
        TensorDataset(train_examples.memories, train_examples.route_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    if positive_weight is None:
        positives = float(train_examples.route_labels.sum().item())
        negatives = float(train_examples.route_labels.numel()) - positives
        positive_weight = negatives / max(positives, 1.0)
    pos_weight_tensor = torch.tensor([float(positive_weight)], device=device)
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = (-1.0, -1.0, -1.0, float("inf"))
    for step in range(1, optimization_steps + 1):
        try:
            memory, label = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, label = next(iterator)
        memory = memory.to(device)
        label = label.to(device)
        logits = router_head(memory)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            label,
            pos_weight=pos_weight_tensor,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == optimization_steps:
            metrics = _evaluate_router(
                router_head,
                validation_examples,
                batch_size=batch_size,
                threshold=threshold,
                device=device,
            )
            log_item = {
                **metrics,
                "step": step,
                "train_loss": float(loss.item()),
            }
            logs.append(log_item)
            print(json.dumps(log_item, sort_keys=True), flush=True)
            score = (
                float(metrics["recall"]),
                float(metrics["precision"]),
                float(metrics["accuracy"]),
                -float(metrics["loss"]),
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in router_head.state_dict().items()
                }
    final_validation = _evaluate_router(
        router_head,
        validation_examples,
        batch_size=batch_size,
        threshold=threshold,
        device=device,
    )
    selected_step = optimization_steps
    selected_validation = final_validation
    if save_best and best_state is not None:
        router_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation
    report = {
        "train_examples": int(train_examples.route_labels.numel()),
        "train_positive_examples": int(train_examples.route_labels.sum().item()),
        "validation_examples": int(validation_examples.route_labels.numel()),
        "validation_positive_examples": int(validation_examples.route_labels.sum().item()),
        "positive_weight": float(positive_weight),
        "threshold": float(threshold),
        "completed_steps": int(optimization_steps),
        "selected_step": int(selected_step),
        "final_validation": final_validation,
        "selected_validation": selected_validation,
        "logs": logs,
        "model_config": {
            "memory_size": int(memory_size),
            "hidden_dim": int(hidden_dim),
            "memory_channels": 3,
        },
    }
    return router_head, report


def _filter_value_map_examples(
    examples: ValueMapExamples,
    *,
    mode: str,
) -> ValueMapExamples:
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
    return ValueMapExamples(
        memories=examples.memories[mask],
        target_fields=examples.target_fields[mask],
        sparse_probabilities=examples.sparse_probabilities[mask],
        value_maps=examples.value_maps[mask],
        actions=examples.actions[mask],
        sparse_labels=examples.sparse_labels[mask],
    )


def _load_value_map_planner_head(
    path: Path,
    *,
    fallback_memory_size: int,
    device: torch.device,
) -> tuple[Phase3AValueMapPlannerHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    planner_head = Phase3AValueMapPlannerHead(
        memory_size=int(config.get("memory_size", fallback_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        memory_channels=int(config.get("memory_channels", 3)),
        architecture=str(config.get("architecture", "conv")),
        refinement_steps=int(config.get("refinement_steps", 8)),
    ).to(device)
    planner_head.load_state_dict(checkpoint["planner_head_state_dict"])
    planner_head.eval()
    return planner_head, report


def _cell_to_row_col(
    cell: tuple[int, int],
    *,
    memory_size: int,
) -> tuple[int, int] | None:
    radius = memory_size // 2
    row = radius - int(cell[0])
    col = radius + int(cell[1])
    if 0 <= row < memory_size and 0 <= col < memory_size:
        return row, col
    return None


def _copy_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
        "radius": int(memory.get("radius", 0)),
    }


def _target_cells_from_probs(
    memory: dict,
    target_probs: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
) -> set[tuple[int, int]]:
    memory_size = int(target_probs.shape[0])
    free = set(memory["free"])
    free.add((0, 0))
    scored: list[tuple[float, tuple[int, int]]] = []
    for cell in free:
        row_col = _cell_to_row_col(cell, memory_size=memory_size)
        if row_col is not None:
            row, col = row_col
            scored.append((float(target_probs[row, col]), cell))
    if not scored:
        return set()
    scored.sort(key=lambda item: item[0], reverse=True)
    targets = {
        cell
        for score, cell in scored[: max(1, int(top_k))]
        if score >= threshold
    }
    return targets or {scored[0][1]}


def _unknown_frontier_targets_at_current(memory: dict) -> set[tuple[int, int]]:
    candidates = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    return {
        cell
        for cell in candidates
        if cell not in memory["free"] and cell not in memory["blocked"]
    }


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


def _teacher_value_map_from_targets(
    memory: dict,
    targets: set[tuple[int, int]],
    *,
    memory_size: int,
    iterations: int,
    gamma: float,
) -> torch.Tensor:
    working = _copy_memory(memory)
    current = (0, 0)
    working["free"].add(current)
    working["blocked"].discard(current)
    passable = set(working["free"])
    if current in targets and _is_egocentric_frontier_cell(working, current):
        targets = _unknown_frontier_targets_at_current(working) or targets
    targets = {
        cell
        for cell in targets
        if _cell_to_row_col(cell, memory_size=memory_size) is not None
        and cell not in working["blocked"]
    }
    passable.update(targets)
    if not targets:
        return torch.zeros(1, memory_size, memory_size, dtype=torch.float32)
    values = {cell: (1.0 if cell in targets else 0.0) for cell in passable}
    for _ in range(max(1, iterations)):
        updated = {}
        for cell in passable:
            if cell in targets:
                updated[cell] = 1.0
            else:
                updated[cell] = max(
                    (
                        gamma * values.get(neighbor, 0.0)
                        for neighbor in _egocentric_neighbors(cell)
                    ),
                    default=0.0,
                )
        values = updated
    tensor = torch.zeros(1, memory_size, memory_size, dtype=torch.float32)
    for cell, value in values.items():
        row_col = _cell_to_row_col(cell, memory_size=memory_size)
        if row_col is not None:
            row, col = row_col
            tensor[0, row, col] = float(value)
    return tensor


def _teacher_value_map_from_probs(
    memory: dict,
    target_probs: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
    iterations: int,
    gamma: float,
) -> torch.Tensor:
    targets = _target_cells_from_probs(
        memory,
        target_probs,
        threshold=threshold,
        top_k=top_k,
    )
    return _teacher_value_map_from_targets(
        memory,
        targets,
        memory_size=int(target_probs.shape[0]),
        iterations=iterations,
        gamma=gamma,
    )


@torch.no_grad()
def _value_map_rollout_action(
    planner_head: Phase3AValueMapPlannerHead,
    ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    ensemble_mode: str,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    recurrent_memory: torch.Tensor,
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
    value_probs = planner_head(
        recurrent_memory,
        target_field,
        sparse_probability,
    ).sigmoid()
    if ensemble_heads:
        ensemble_probs = [value_probs]
        for ensemble_head in ensemble_heads:
            ensemble_probs.append(
                ensemble_head(
                    recurrent_memory,
                    target_field,
                    sparse_probability,
                ).sigmoid()
            )
        stacked_probs = torch.stack(ensemble_probs, dim=0)
        if ensemble_mode == "max":
            value_probs = stacked_probs.max(dim=0).values
        else:
            value_probs = stacked_probs.mean(dim=0)
    action, _mode = _select_egocentric_learned_value_map_action(
        _copy_memory(memory_dict),
        value_probs[0, 0].detach().cpu(),
    )
    return action


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
    rollout_planner_head: Phase3AValueMapPlannerHead,
    rollout_ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    rollout_ensemble_mode: str,
    rollout_fallback_head: Phase3AValueMapPlannerHead | None,
    rollout_fallback_after_step: int | None,
    rollout_turn_oscillation_breaker: bool,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    include_marker_start_groups: bool,
    device: torch.device,
) -> torch.Tensor:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

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
        trajectory: list[dict] = []
        for step in range(max_steps):
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
            memories.append(recurrent_memory[0].detach().cpu())
            active_planner_head = rollout_planner_head
            active_ensemble_heads = rollout_ensemble_heads
            if (
                rollout_fallback_head is not None
                and (
                    rollout_fallback_after_step is None
                    or step >= rollout_fallback_after_step
                )
            ):
                active_planner_head = rollout_fallback_head
                active_ensemble_heads = ()
            rollout_action = _value_map_rollout_action(
                active_planner_head,
                active_ensemble_heads,
                rollout_ensemble_mode,
                value_field_head,
                extractor_head,
                recurrent_memory,
                memory_dict,
            )
            if rollout_turn_oscillation_breaker:
                replacement_action = _break_turn_oscillation_action(
                    rollout_action,
                    memory_dict,
                    trajectory,
                    state,
                )
                if replacement_action is not None:
                    rollout_action = replacement_action
            next_state, collision = step_state(scene, state, rollout_action)
            trajectory.append(
                {
                    "state": {
                        "x": int(state.x),
                        "y": int(state.y),
                        "yaw": int(state.yaw),
                    },
                    "next_state": {
                        "x": int(next_state.x),
                        "y": int(next_state.y),
                        "yaw": int(next_state.yaw),
                    },
                    "selected_action": rollout_action,
                    "collision": bool(collision),
                }
            )
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
    min_first_marker_step: int | None,
    pre_latent_marker_only: bool,
    repeat: int,
    device: torch.device,
) -> torch.Tensor:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

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
            if not _keep_trace_episode(
                episode,
                failed_only=failed_only,
                min_first_marker_step=min_first_marker_step,
            ):
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
                if not (pre_latent_marker_only and latent_marker_seen_ever):
                    memories.append(recurrent_memory[0].detach().cpu())
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not memories:
        raise SystemExit("no trace memories were generated")
    stacked = torch.stack(memories)
    if repeat > 1:
        stacked = stacked.repeat((repeat, 1, 1, 1))
    return stacked


def _episode_claims_from_report(path: Path) -> dict[int, bool]:
    report = json.loads(path.read_text())
    claims: dict[int, bool] = {}
    for fallback_index, summary in enumerate(report.get("episode_summaries", [])):
        episode_index = int(summary.get("episode_index", fallback_index))
        claims[episode_index] = bool(summary.get("claimed", False))
    if not claims:
        raise SystemExit(f"{path} does not contain episode_summaries")
    return claims


@torch.no_grad()
def _build_trace_outcome_router_examples(
    trace_paths: list[Path],
    source_paths: list[Path],
    primary_report_paths: list[Path],
    fallback_report_paths: list[Path],
    *,
    fallback_rows: list[dict],
    fallback_scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_episodes: int | None,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    latent_map_marker_threshold: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    pre_memory_marker_only: bool,
    same_outcome_negative: bool,
    repeat: int,
    device: torch.device,
) -> tuple[RouterExamples, dict]:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

    if not trace_paths:
        raise SystemExit("trace-outcome router training requires trace paths")
    if source_paths and len(source_paths) != len(trace_paths):
        raise SystemExit(
            "trace-outcome source paths must be passed once for each trace path"
        )
    if len(primary_report_paths) != len(trace_paths):
        raise SystemExit(
            "trace-outcome primary reports must be passed once for each trace path"
        )
    if len(fallback_report_paths) != len(trace_paths):
        raise SystemExit(
            "trace-outcome fallback reports must be passed once for each trace path"
        )

    memories = []
    labels = []
    stats = {
        "trace_outcome_paths": len(trace_paths),
        "trace_outcome_episodes": 0,
        "trace_outcome_primary_only_episodes": 0,
        "trace_outcome_fallback_only_episodes": 0,
        "trace_outcome_same_outcome_episodes": 0,
        "trace_outcome_skipped_marker_memory_states": 0,
    }
    for path_index, trace_path in enumerate(trace_paths):
        rows = (
            read_jsonl(source_paths[path_index])
            if source_paths
            else fallback_rows
        )
        scene_seed = (
            _infer_scene_seed(source_paths[path_index])
            if source_paths
            else fallback_scene_seed
        )
        if scene_seed is None:
            source_name = source_paths[path_index] if source_paths else trace_path
            raise SystemExit(f"could not infer trace-outcome scene seed from {source_name}")
        groups = _groups_from_rows(rows, include_marker_start_groups=False)
        primary_claims = _episode_claims_from_report(primary_report_paths[path_index])
        fallback_claims = _episode_claims_from_report(fallback_report_paths[path_index])
        trace = json.loads(trace_path.read_text())
        episodes = trace.get("episodes", [])
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        for episode_index, episode in enumerate(episodes):
            source_index = int(episode.get("source_episode_index", episode_index))
            if source_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has source episode index {source_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            primary_claimed = bool(primary_claims.get(source_index, False))
            fallback_claimed = bool(fallback_claims.get(source_index, False))
            if primary_claimed == fallback_claimed:
                stats["trace_outcome_same_outcome_episodes"] += 1
                if not same_outcome_negative:
                    continue
            label = float(fallback_claimed and not primary_claimed)
            if label >= 0.5:
                stats["trace_outcome_fallback_only_episodes"] += 1
            else:
                stats["trace_outcome_primary_only_episodes"] += 1
            stats["trace_outcome_episodes"] += 1
            template = groups[source_index][0]
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
                keep_state = True
                if pre_memory_marker_only:
                    memory_dict = _egocentric_memory_tensor_to_dict(
                        recurrent_memory[0].detach().cpu(),
                        blocked_threshold=blocked_threshold,
                        free_threshold=free_threshold,
                        marker_threshold=marker_threshold,
                    )
                    marker = memory_dict.get("marker")
                    keep_state = not (
                        marker is not None and marker in memory_dict["free"]
                    )
                if keep_state:
                    memories.append(recurrent_memory[0].detach().cpu())
                    labels.append(label)
                else:
                    stats["trace_outcome_skipped_marker_memory_states"] += 1
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not memories:
        raise SystemExit("no trace-outcome router examples were generated")
    stacked_memories = torch.stack(memories)
    stacked_labels = torch.tensor(labels, dtype=torch.float32)
    if repeat > 1:
        stacked_memories = stacked_memories.repeat((repeat, 1, 1, 1))
        stacked_labels = stacked_labels.repeat((repeat,))
    stats["trace_outcome_examples"] = int(stacked_labels.numel())
    stats["trace_outcome_positive_examples"] = int(stacked_labels.sum().item())
    return (
        RouterExamples(memories=stacked_memories, route_labels=stacked_labels),
        stats,
    )


@torch.no_grad()
def _build_trace_action_preference_router_examples(
    trace_paths: list[Path],
    source_paths: list[Path],
    *,
    fallback_rows: list[dict],
    fallback_scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_episodes: int | None,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    primary_planner_head: Phase3AValueMapPlannerHead,
    primary_ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    primary_ensemble_mode: str,
    fallback_planner_head: Phase3AValueMapPlannerHead,
    latent_map_marker_threshold: float,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    pre_memory_marker_only: bool,
    repeat: int,
    device: torch.device,
) -> tuple[RouterExamples, dict]:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

    if not trace_paths:
        raise SystemExit("trace-action-preference router training requires trace paths")
    if source_paths and len(source_paths) != len(trace_paths):
        raise SystemExit(
            "trace-action-preference source paths must be passed once for each trace path"
        )

    memories = []
    labels = []
    stats = {
        "trace_action_preference_paths": len(trace_paths),
        "trace_action_preference_episodes": 0,
        "trace_action_preference_examples": 0,
        "trace_action_preference_positive_examples": 0,
        "trace_action_preference_primary_teacher_matches": 0,
        "trace_action_preference_fallback_teacher_matches": 0,
        "trace_action_preference_both_same_action": 0,
        "trace_action_preference_neither_teacher_matches": 0,
        "trace_action_preference_skipped_marker_memory_states": 0,
        "trace_action_preference_skipped_missing_teacher": 0,
    }
    for path_index, trace_path in enumerate(trace_paths):
        rows = read_jsonl(source_paths[path_index]) if source_paths else fallback_rows
        scene_seed = (
            _infer_scene_seed(source_paths[path_index])
            if source_paths
            else fallback_scene_seed
        )
        if scene_seed is None:
            source_name = source_paths[path_index] if source_paths else trace_path
            raise SystemExit(
                f"could not infer trace-action-preference scene seed from {source_name}"
            )
        groups = _groups_from_rows(rows, include_marker_start_groups=False)
        trace = json.loads(trace_path.read_text())
        episodes = trace.get("episodes", [])
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        for episode_index, episode in enumerate(episodes):
            source_index = int(episode.get("source_episode_index", episode_index))
            if source_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has source episode index {source_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            stats["trace_action_preference_episodes"] += 1
            template = groups[source_index][0]
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
                teacher_action = str(item.get("oracle_action", ""))
                if teacher_action not in ACTION_INDEX:
                    stats["trace_action_preference_skipped_missing_teacher"] += 1
                    last_action = str(item["selected_action"])
                    last_collision = bool(item["collision"])
                    continue
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
                memory_dict = _egocentric_memory_tensor_to_dict(
                    recurrent_memory[0].detach().cpu(),
                    blocked_threshold=blocked_threshold,
                    free_threshold=free_threshold,
                    marker_threshold=marker_threshold,
                )
                marker = memory_dict.get("marker")
                if pre_memory_marker_only and marker is not None and marker in memory_dict["free"]:
                    stats["trace_action_preference_skipped_marker_memory_states"] += 1
                    last_action = str(item["selected_action"])
                    last_collision = bool(item["collision"])
                    continue
                primary_action = _value_map_rollout_action(
                    primary_planner_head,
                    primary_ensemble_heads,
                    primary_ensemble_mode,
                    value_field_head,
                    extractor_head,
                    recurrent_memory,
                    memory_dict,
                )
                fallback_action = _value_map_rollout_action(
                    fallback_planner_head,
                    (),
                    primary_ensemble_mode,
                    value_field_head,
                    extractor_head,
                    recurrent_memory,
                    memory_dict,
                )
                primary_match = primary_action == teacher_action
                fallback_match = fallback_action == teacher_action
                if primary_match:
                    stats["trace_action_preference_primary_teacher_matches"] += 1
                if fallback_match:
                    stats["trace_action_preference_fallback_teacher_matches"] += 1
                if primary_action == fallback_action:
                    label = 0.0
                    stats["trace_action_preference_both_same_action"] += 1
                elif fallback_match and not primary_match:
                    label = 1.0
                else:
                    label = 0.0
                    if not primary_match and not fallback_match:
                        stats["trace_action_preference_neither_teacher_matches"] += 1
                memories.append(recurrent_memory[0].detach().cpu())
                labels.append(label)
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not memories:
        raise SystemExit("no trace-action-preference router examples were generated")
    stacked_memories = torch.stack(memories)
    stacked_labels = torch.tensor(labels, dtype=torch.float32)
    if repeat > 1:
        stacked_memories = stacked_memories.repeat((repeat, 1, 1, 1))
        stacked_labels = stacked_labels.repeat((repeat,))
    stats["trace_action_preference_examples"] = int(stacked_labels.numel())
    stats["trace_action_preference_positive_examples"] = int(stacked_labels.sum().item())
    return (
        RouterExamples(memories=stacked_memories, route_labels=stacked_labels),
        stats,
    )


def _oracle_action_value_map(action: str, *, memory_size: int) -> torch.Tensor:
    action_cells = {
        "forward": (1, 0),
        "turn_left": (0, 1),
        "turn_right": (0, -1),
        "hold": (0, 0),
    }
    if action not in action_cells:
        raise ValueError(f"unknown oracle action: {action!r}")
    value_map = torch.zeros(1, memory_size, memory_size, dtype=torch.float32)
    row_col = _cell_to_row_col(action_cells[action], memory_size=memory_size)
    if row_col is not None:
        row, col = row_col
        value_map[0, row, col] = 1.0
    return value_map


def _first_marker_step_from_trace_episode(episode: dict) -> int | None:
    for item in episode.get("trajectory", []):
        if bool(item.get("current_marker_seen", False)) or bool(
            item.get("marker_seen_ever", False)
        ):
            return int(item.get("step", 0))
    return None


def _keep_trace_episode(
    episode: dict,
    *,
    failed_only: bool,
    min_first_marker_step: int | None,
) -> bool:
    failed = not bool(episode.get("claimed", False))
    late = False
    if min_first_marker_step is not None:
        first_marker_step = _first_marker_step_from_trace_episode(episode)
        late = (
            first_marker_step is not None
            and int(first_marker_step) >= int(min_first_marker_step)
        )
    if failed_only:
        return failed or late
    if min_first_marker_step is not None:
        return late
    return True


@torch.no_grad()
def _build_trace_action_value_map_examples(
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
    teacher_source: str,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
    failed_only: bool,
    min_first_marker_step: int | None,
    pre_latent_marker_only: bool,
    repeat: int,
    device: torch.device,
) -> ValueMapExamples:
    from scripts.train_jepa_phase3a_value_field import _groups_from_rows

    groups = _groups_from_rows(rows, include_marker_start_groups=False)
    memories = []
    target_fields = []
    sparse_probabilities = []
    value_maps = []
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
            if not _keep_trace_episode(
                episode,
                failed_only=failed_only,
                min_first_marker_step=min_first_marker_step,
            ):
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
            odom_frontier_memory = {"free": set(), "blocked": set(), "marker": None}
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
                if not (pre_latent_marker_only and latent_marker_seen_ever):
                    if teacher_source == "local_oracle":
                        target_action = str(item.get("oracle_action", ""))
                        if target_action not in ACTION_INDEX:
                            raise SystemExit(
                                f"{trace_path} episode {episode_index} step "
                                f"{item.get('step')} has invalid oracle_action "
                                f"{target_action!r}"
                            )
                    elif teacher_source == "trace_selected":
                        target_action = str(item["selected_action"])
                    elif teacher_source == "odom_lookahead":
                        _update_odom_frontier_memory(
                            odom_frontier_memory,
                            scene=scene,
                            state=state,
                            view_size=view_size,
                            current_goal_marker=True,
                        )
                        target_action = _select_odom_frontier_lookahead_action(
                            odom_frontier_memory,
                            scene=scene,
                            state=state,
                            view_size=view_size,
                            horizon=odom_lookahead_horizon,
                            beam_width=odom_lookahead_beam_width,
                        )
                    else:
                        raise ValueError(
                            f"unknown trace action teacher source: {teacher_source!r}"
                        )
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
                    sparse_prob = extractor_head(recurrent_memory).sigmoid()
                    memories.append(recurrent_memory[0].detach().cpu())
                    target_fields.append(target_probs_gpu[0].detach().cpu())
                    sparse_probabilities.append(float(sparse_prob.item()))
                    value_maps.append(
                        _oracle_action_value_map(
                            target_action,
                            memory_size=memory_size,
                        )
                    )
                    actions.append(ACTION_INDEX[target_action])
                    sparse_labels.append(True)
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not actions:
        raise SystemExit("no trace action value-map examples were generated")
    examples = ValueMapExamples(
        memories=torch.stack(memories),
        target_fields=torch.stack(target_fields),
        sparse_probabilities=torch.tensor(sparse_probabilities, dtype=torch.float32),
        value_maps=torch.stack(value_maps),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=torch.tensor(sparse_labels, dtype=torch.bool),
    )
    if repeat <= 1:
        return examples
    return ValueMapExamples(
        memories=examples.memories.repeat((repeat, 1, 1, 1)),
        target_fields=examples.target_fields.repeat((repeat, 1, 1, 1)),
        sparse_probabilities=examples.sparse_probabilities.repeat(repeat),
        value_maps=examples.value_maps.repeat((repeat, 1, 1, 1)),
        actions=examples.actions.repeat(repeat),
        sparse_labels=examples.sparse_labels.repeat(repeat),
    )


@torch.no_grad()
def _build_value_map_examples(
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
    explicit_frontier_targets: bool,
    fixed_marker_targets: bool,
    value_iterations: int,
    value_gamma: float,
    batch_size: int,
    device: torch.device,
) -> ValueMapExamples:
    dataset = TensorDataset(memories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_memories = []
    all_targets = []
    all_sparse_probs = []
    value_maps = []
    actions = []
    sparse_labels = []
    for (memory_batch,) in loader:
        memory_batch = memory_batch.to(device)
        target_fields = value_field_head(memory_batch).sigmoid()
        sparse_probs = extractor_head(memory_batch).sigmoid()
        memory_cpu = memory_batch.detach().cpu()
        targets_cpu = target_fields.detach().cpu()
        sparse_cpu = sparse_probs.detach().cpu()
        for item_index in range(int(memory_cpu.shape[0])):
            memory_dict = _egocentric_memory_tensor_to_dict(
                memory_cpu[item_index],
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
            if (
                fixed_marker_targets
                and marker is not None
                and marker in memory_dict["free"]
            ):
                memory_size = int(target_probs.shape[0])
                radius = memory_size // 2
                row = radius - int(marker[0])
                col = radius + int(marker[1])
                if 0 <= row < memory_size and 0 <= col < memory_size:
                    target_probs = torch.zeros_like(target_probs)
                    target_probs[row, col] = 1.0
            if explicit_frontier_targets and not sparse:
                value_map = _teacher_value_map_from_targets(
                    memory_dict,
                    _frontier_target_cells_for_memory(memory_dict),
                    memory_size=int(target_probs.shape[0]),
                    iterations=value_iterations,
                    gamma=value_gamma,
                )
            else:
                value_map = _teacher_value_map_from_probs(
                    memory_dict,
                    target_probs,
                    threshold=target_threshold,
                    top_k=top_k,
                    iterations=value_iterations,
                    gamma=value_gamma,
                )
            action, _mode = _select_egocentric_learned_value_map_action(
                _copy_memory(memory_dict),
                value_map[0],
            )
            all_memories.append(memory_cpu[item_index])
            all_targets.append(target_probs.view(1, *target_probs.shape))
            all_sparse_probs.append(sparse_cpu[item_index].view(1))
            value_maps.append(value_map)
            actions.append(ACTION_INDEX[action])
            sparse_labels.append(sparse)
    return ValueMapExamples(
        memories=torch.stack(all_memories),
        target_fields=torch.stack(all_targets),
        sparse_probabilities=torch.stack(all_sparse_probs).view(-1),
        value_maps=torch.stack(value_maps),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=torch.tensor(sparse_labels, dtype=torch.bool),
    )


def _build_synthetic_value_map_examples(
    count: int,
    *,
    memory_size: int,
    target_threshold: float,
    target_top_k: int,
    sparse_target_top_k: int,
    value_iterations: int,
    value_gamma: float,
    fixed_marker_targets: bool,
    seed: int,
) -> ValueMapExamples:
    action_examples = _build_synthetic_action_examples(
        count,
        memory_size=memory_size,
        target_threshold=target_threshold,
        target_top_k=target_top_k,
        sparse_target_top_k=sparse_target_top_k,
        fixed_marker_targets=fixed_marker_targets,
        seed=seed,
    )
    value_maps = []
    actions = []
    for memory_tensor, target_field, sparse_probability in zip(
        action_examples.memories,
        action_examples.target_fields,
        action_examples.sparse_probabilities,
    ):
        memory = _egocentric_memory_tensor_to_dict(
            memory_tensor,
            blocked_threshold=0.5,
            free_threshold=0.5,
            marker_threshold=0.5,
        )
        top_k = (
            sparse_target_top_k
            if float(sparse_probability.item()) >= 0.5
            else target_top_k
        )
        value_map = _teacher_value_map_from_probs(
            memory,
            target_field[0],
            threshold=target_threshold,
            top_k=top_k,
            iterations=value_iterations,
            gamma=value_gamma,
        )
        action, _mode = _select_egocentric_learned_value_map_action(
            _copy_memory(memory),
            value_map[0],
        )
        value_maps.append(value_map)
        actions.append(ACTION_INDEX[action])
    return ValueMapExamples(
        memories=action_examples.memories,
        target_fields=action_examples.target_fields,
        sparse_probabilities=action_examples.sparse_probabilities,
        value_maps=torch.stack(value_maps),
        actions=torch.tensor(actions, dtype=torch.long),
        sparse_labels=action_examples.sparse_labels,
    )


def _planner_action_logits(value_logits: torch.Tensor) -> torch.Tensor:
    if value_logits.ndim != 4 or int(value_logits.shape[1]) != 1:
        raise ValueError(
            "value_logits must have shape (B, 1, S, S), got "
            f"{tuple(value_logits.shape)}"
        )
    _batch, _channels, height, width = value_logits.shape
    if height != width or height % 2 == 0:
        raise ValueError(f"value map must be odd square, got {(height, width)}")
    radius = height // 2
    forward = value_logits[:, 0, radius - 1, radius]
    left_lateral = value_logits[:, 0, radius, radius + 1]
    left_reverse = value_logits[:, 0, radius + 1, radius]
    left = torch.maximum(left_lateral, left_reverse)
    right = value_logits[:, 0, radius, radius - 1]
    hold = value_logits[:, 0, radius, radius]
    return torch.stack([forward, left, right, hold], dim=1)


@torch.no_grad()
def _evaluate(
    planner_head: Phase3AValueMapPlannerHead,
    examples: ValueMapExamples,
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
    for memory, target, sparse_prob, value_map, action, sparse_label in loader:
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        value_map = value_map.to(device)
        logits = planner_head(memory, target, sparse_prob)
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
                memory_dict,
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
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
    parser.add_argument("--latent-value-field-head", type=Path, required=True)
    parser.add_argument("--latent-value-extractor-head", type=Path, required=True)
    parser.add_argument(
        "--init-planner-checkpoint",
        type=Path,
        default=None,
        help="optional value-map planner checkpoint to initialize from",
    )
    parser.add_argument("--dagger-rollout-value-map-planner-head", type=Path, default=None)
    parser.add_argument(
        "--dagger-rollout-value-map-ensemble-head",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dagger-rollout-value-map-ensemble-mode",
        choices=("mean", "max"),
        default="mean",
    )
    parser.add_argument(
        "--dagger-rollout-value-map-fallback-head",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dagger-rollout-value-map-fallback-after-step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dagger-rollout-value-map-turn-oscillation-breaker",
        action="store_true",
        help=(
            "when collecting DAgger rollout memories, apply the same repeated "
            "left/right turn oscillation breaker used by closed-loop export"
        ),
    )
    parser.add_argument(
        "--trace-memory-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "closed-loop trace JSON to replay into recurrent-memory training "
            "examples; may be passed multiple times"
        ),
    )
    parser.add_argument(
        "--trace-memory-source-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional JSONL source rows paired one-for-one with "
            "--trace-memory-data; use this when replaying traces from "
            "multiple generated splits"
        ),
    )
    parser.add_argument(
        "--trace-memory-source",
        choices=("train", "validation"),
        default="validation",
        help="which source rows/scene seed the trace episode indices correspond to",
    )
    parser.add_argument(
        "--trace-memory-failed-only",
        action="store_true",
        help="only replay unclaimed episodes from --trace-memory-data",
    )
    parser.add_argument(
        "--trace-memory-min-first-marker-step",
        type=int,
        default=None,
        help=(
            "include episodes whose first real marker sighting is at or after "
            "this step; with --trace-memory-failed-only this is additive, so "
            "failed episodes and late-sighting episodes are both replayed"
        ),
    )
    parser.add_argument(
        "--trace-memory-pre-latent-marker-only",
        action="store_true",
        help=(
            "only keep trace memories before latent local marker evidence has "
            "been observed in that episode"
        ),
    )
    parser.add_argument("--trace-memory-repeat", type=int, default=1)
    parser.add_argument(
        "--trace-action-teacher-source",
        choices=("none", "local_oracle", "trace_selected", "odom_lookahead"),
        default="none",
        help=(
            "optionally add trace states as direct action-supervised value-map "
            "examples"
        ),
    )
    parser.add_argument(
        "--trace-odom-lookahead-horizon",
        type=int,
        default=9,
        help="lookahead depth for --trace-action-teacher-source=odom_lookahead",
    )
    parser.add_argument(
        "--trace-odom-lookahead-beam-width",
        type=int,
        default=32,
        help="beam width for --trace-action-teacher-source=odom_lookahead",
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
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument(
        "--planner-architecture",
        choices=("conv", "dilated", "recurrent"),
        default="conv",
    )
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--value-iterations", type=int, default=64)
    parser.add_argument("--value-gamma", type=float, default=0.94)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.5)
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
    parser.add_argument("--synthetic-examples", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=20260657)
    parser.add_argument("--train-router-only", action="store_true")
    parser.add_argument(
        "--router-label-source",
        choices=(
            "side_wall",
            "counterfactual",
            "trace_outcome",
            "trace_action_preference",
        ),
        default="side_wall",
    )
    parser.add_argument(
        "--router-trace-primary-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "primary-controller report paired with --trace-memory-data for "
            "--router-label-source trace_outcome"
        ),
    )
    parser.add_argument(
        "--router-trace-fallback-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "fallback-controller report paired with --trace-memory-data for "
            "--router-label-source trace_outcome"
        ),
    )
    parser.add_argument(
        "--router-validation-trace-memory-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "validation closed-loop traces for "
            "--router-label-source trace_outcome"
        ),
    )
    parser.add_argument(
        "--router-validation-trace-memory-source-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional JSONL source rows paired one-for-one with "
            "--router-validation-trace-memory-data"
        ),
    )
    parser.add_argument(
        "--router-validation-trace-primary-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "primary-controller report paired with "
            "--router-validation-trace-memory-data"
        ),
    )
    parser.add_argument(
        "--router-validation-trace-fallback-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "fallback-controller report paired with "
            "--router-validation-trace-memory-data"
        ),
    )
    parser.add_argument(
        "--router-trace-pre-memory-marker-only",
        action="store_true",
        help=(
            "for trace_outcome router labels, keep only states where recurrent "
            "marker memory is not yet available to the marker planner"
        ),
    )
    parser.add_argument(
        "--router-trace-same-outcome-negative",
        action="store_true",
        help=(
            "for trace_outcome router labels, include episodes where primary "
            "and fallback have the same claimed/not-claimed outcome as "
            "negative examples"
        ),
    )
    parser.add_argument("--router-synthetic-examples", type=int, default=0)
    parser.add_argument("--router-counterfactual-horizon", type=int, default=4)
    parser.add_argument("--router-counterfactual-utility-margin", type=float, default=0.0)
    parser.add_argument("--router-positive-weight", type=float, default=None)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260657)
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
    if args.refinement_steps < 1:
        raise SystemExit("--refinement-steps must be positive")
    if args.positive_weight <= 0.0:
        raise SystemExit("--positive-weight must be positive")
    if args.action_loss_weight < 0.0:
        raise SystemExit("--action-loss-weight must be non-negative")
    if args.value_iterations < 1:
        raise SystemExit("--value-iterations must be positive")
    if not 0.0 < args.value_gamma <= 1.0:
        raise SystemExit("--value-gamma must be in (0, 1]")
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
    if args.trace_odom_lookahead_horizon < 1:
        raise SystemExit("--trace-odom-lookahead-horizon must be positive")
    if args.trace_odom_lookahead_beam_width < 1:
        raise SystemExit("--trace-odom-lookahead-beam-width must be positive")
    if (
        args.trace_memory_min_first_marker_step is not None
        and args.trace_memory_min_first_marker_step < 0
    ):
        raise SystemExit("--trace-memory-min-first-marker-step must be non-negative")
    if args.trace_memory_source_data and (
        len(args.trace_memory_source_data) != len(args.trace_memory_data)
    ):
        raise SystemExit(
            "--trace-memory-source-data must be passed once for each "
            "--trace-memory-data path"
        )
    if args.router_validation_trace_memory_source_data and (
        len(args.router_validation_trace_memory_source_data)
        != len(args.router_validation_trace_memory_data)
    ):
        raise SystemExit(
            "--router-validation-trace-memory-source-data must be passed once "
            "for each --router-validation-trace-memory-data path"
        )
    if (
        args.train_router_only
        and args.router_label_source
        not in {"counterfactual", "trace_outcome", "trace_action_preference"}
        and (
        args.extra_train_data or args.extra_validation_data
        )
    ):
        raise SystemExit(
            "--extra-train-data/--extra-validation-data are not supported with "
            "--train-router-only unless --router-label-source counterfactual "
            "or trace_outcome/trace_action_preference"
        )
    if args.trace_action_teacher_source != "none" and not args.trace_memory_data:
        raise SystemExit(
            "--trace-action-teacher-source requires --trace-memory-data"
        )
    if args.router_synthetic_examples < 0:
        raise SystemExit("--router-synthetic-examples must be non-negative")
    if args.router_counterfactual_horizon < 1:
        raise SystemExit("--router-counterfactual-horizon must be positive")
    if args.router_counterfactual_utility_margin < 0.0:
        raise SystemExit("--router-counterfactual-utility-margin must be non-negative")
    if (
        args.router_positive_weight is not None
        and args.router_positive_weight <= 0.0
    ):
        raise SystemExit("--router-positive-weight must be positive")
    if not 0.0 < args.router_threshold < 1.0:
        raise SystemExit("--router-threshold must be in (0, 1)")
    if (
        args.dagger_rollout_value_map_fallback_after_step is not None
        and args.dagger_rollout_value_map_fallback_after_step < 0
    ):
        raise SystemExit(
            "--dagger-rollout-value-map-fallback-after-step must be non-negative"
        )
    if (
        args.dagger_rollout_value_map_planner_head is None
        and (
            args.dagger_rollout_value_map_ensemble_head
            or args.dagger_rollout_value_map_fallback_head is not None
        )
    ):
        raise SystemExit(
            "--dagger-rollout-value-map-planner-head is required when using "
            "DAgger rollout ensemble or fallback heads"
        )
    if args.train_router_only and args.router_label_source == "counterfactual":
        if args.dagger_rollout_value_map_planner_head is None:
            raise SystemExit(
                "--router-label-source counterfactual requires "
                "--dagger-rollout-value-map-planner-head"
            )
        if args.dagger_rollout_value_map_fallback_head is None:
            raise SystemExit(
                "--router-label-source counterfactual requires "
                "--dagger-rollout-value-map-fallback-head"
            )
        if args.router_synthetic_examples > 0:
            raise SystemExit(
                "--router-synthetic-examples is only supported with "
                "--router-label-source side_wall"
            )
    if args.train_router_only and args.router_label_source == "trace_outcome":
        if not args.trace_memory_data:
            raise SystemExit(
                "--router-label-source trace_outcome requires --trace-memory-data"
            )
        if not args.router_validation_trace_memory_data:
            raise SystemExit(
                "--router-label-source trace_outcome requires "
                "--router-validation-trace-memory-data"
            )
        if len(args.router_trace_primary_report) != len(args.trace_memory_data):
            raise SystemExit(
                "--router-trace-primary-report must be passed once for each "
                "--trace-memory-data path"
            )
        if len(args.router_trace_fallback_report) != len(args.trace_memory_data):
            raise SystemExit(
                "--router-trace-fallback-report must be passed once for each "
                "--trace-memory-data path"
            )
        if (
            len(args.router_validation_trace_primary_report)
            != len(args.router_validation_trace_memory_data)
        ):
            raise SystemExit(
                "--router-validation-trace-primary-report must be passed once "
                "for each --router-validation-trace-memory-data path"
            )
        if (
            len(args.router_validation_trace_fallback_report)
            != len(args.router_validation_trace_memory_data)
        ):
            raise SystemExit(
                "--router-validation-trace-fallback-report must be passed once "
                "for each --router-validation-trace-memory-data path"
            )
        if args.router_synthetic_examples > 0:
            raise SystemExit(
                "--router-synthetic-examples is only supported with "
                "--router-label-source side_wall"
            )
    if args.train_router_only and args.router_label_source == "trace_action_preference":
        if not args.trace_memory_data:
            raise SystemExit(
                "--router-label-source trace_action_preference requires "
                "--trace-memory-data"
            )
        if not args.router_validation_trace_memory_data:
            raise SystemExit(
                "--router-label-source trace_action_preference requires "
                "--router-validation-trace-memory-data"
            )
        if args.dagger_rollout_value_map_planner_head is None:
            raise SystemExit(
                "--router-label-source trace_action_preference requires "
                "--dagger-rollout-value-map-planner-head"
            )
        if args.dagger_rollout_value_map_fallback_head is None:
            raise SystemExit(
                "--router-label-source trace_action_preference requires "
                "--dagger-rollout-value-map-fallback-head"
            )
        if args.router_synthetic_examples > 0:
            raise SystemExit(
                "--router-synthetic-examples is only supported with "
                "--router-label-source side_wall"
            )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
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
    dagger_rollout_planner_head = None
    dagger_rollout_planner_report = None
    init_planner_report = None
    if args.dagger_rollout_value_map_planner_head is not None:
        dagger_rollout_planner_head, dagger_rollout_planner_report = (
            _load_value_map_planner_head(
                args.dagger_rollout_value_map_planner_head,
                fallback_memory_size=args.memory_size,
                device=device,
            )
        )
    dagger_rollout_ensemble_heads = []
    dagger_rollout_ensemble_reports = []
    for ensemble_path in args.dagger_rollout_value_map_ensemble_head:
        ensemble_head, ensemble_report = _load_value_map_planner_head(
            ensemble_path,
            fallback_memory_size=args.memory_size,
            device=device,
        )
        dagger_rollout_ensemble_heads.append(ensemble_head)
        dagger_rollout_ensemble_reports.append(ensemble_report)
    dagger_rollout_fallback_head = None
    dagger_rollout_fallback_report = None
    if args.dagger_rollout_value_map_fallback_head is not None:
        dagger_rollout_fallback_head, dagger_rollout_fallback_report = (
            _load_value_map_planner_head(
                args.dagger_rollout_value_map_fallback_head,
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
        dagger_rollout_planner_head is not None
        and int(dagger_rollout_planner_head.memory_size) != int(args.memory_size)
    ):
        raise SystemExit(
            "--memory-size must match DAgger rollout value-map planner size "
            f"({dagger_rollout_planner_head.memory_size})"
        )
    for ensemble_head in dagger_rollout_ensemble_heads:
        if int(ensemble_head.memory_size) != int(args.memory_size):
            raise SystemExit(
                "--memory-size must match DAgger rollout ensemble value-map "
                f"planner size ({ensemble_head.memory_size})"
            )
    if (
        dagger_rollout_fallback_head is not None
        and int(dagger_rollout_fallback_head.memory_size) != int(args.memory_size)
    ):
        raise SystemExit(
            "--memory-size must match DAgger rollout fallback value-map planner "
            f"size ({dagger_rollout_fallback_head.memory_size})"
        )

    def _build_counterfactual_router_set(
        base_rows: list[dict],
        *,
        base_seed: int,
        extra_paths: list[Path],
        max_episodes: int | None,
        include_marker_start_groups: bool,
    ) -> RouterExamples:
        assert dagger_rollout_planner_head is not None
        assert dagger_rollout_fallback_head is not None
        examples = _build_counterfactual_router_examples(
            base_rows,
            scene_seed=base_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            max_episodes=max_episodes,
            max_steps=args.max_steps,
            horizon=args.router_counterfactual_horizon,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            value_field_head=value_field_head,
            extractor_head=extractor_head,
            primary_planner_head=dagger_rollout_planner_head,
            primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
            primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
            fallback_planner_head=dagger_rollout_fallback_head,
            fallback_after_step=args.dagger_rollout_value_map_fallback_after_step,
            utility_margin=args.router_counterfactual_utility_margin,
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            include_marker_start_groups=include_marker_start_groups,
            device=device,
        )
        for extra_path in extra_paths:
            extra_seed = _infer_scene_seed(extra_path)
            if extra_seed is None:
                raise SystemExit(
                    f"could not infer extra router scene seed from {extra_path}"
                )
            examples = _concat_router_examples(
                examples,
                _build_counterfactual_router_examples(
                    read_jsonl(extra_path),
                    scene_seed=extra_seed,
                    width=args.width_cells,
                    height=args.height_cells,
                    view_size=args.view_size,
                    memory_size=args.memory_size,
                    max_episodes=max_episodes,
                    max_steps=args.max_steps,
                    horizon=args.router_counterfactual_horizon,
                    base_model=base_model,
                    latent_map_head=latent_map_head,
                    latent_memory_updater=latent_memory_updater,
                    value_field_head=value_field_head,
                    extractor_head=extractor_head,
                    primary_planner_head=dagger_rollout_planner_head,
                    primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                    primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                    fallback_planner_head=dagger_rollout_fallback_head,
                    fallback_after_step=(
                        args.dagger_rollout_value_map_fallback_after_step
                    ),
                    utility_margin=args.router_counterfactual_utility_margin,
                    blocked_threshold=args.latent_memory_blocked_threshold,
                    free_threshold=args.latent_memory_free_threshold,
                    marker_threshold=args.latent_memory_marker_threshold,
                    include_marker_start_groups=include_marker_start_groups,
                    device=device,
                ),
            )
        return examples

    if args.train_router_only and args.router_label_source == "counterfactual":
        train_router_examples = _build_counterfactual_router_set(
            train_rows,
            base_seed=train_seed,
            extra_paths=args.extra_train_data,
            max_episodes=args.max_train_episodes,
            include_marker_start_groups=args.include_marker_start_train_groups,
        )
        validation_router_examples = _build_counterfactual_router_set(
            validation_rows,
            base_seed=validation_seed,
            extra_paths=args.extra_validation_data,
            max_episodes=args.max_validation_episodes,
            include_marker_start_groups=False,
        )
        router_head, router_report = _train_router_head(
            train_router_examples,
            validation_router_examples,
            memory_size=args.memory_size,
            hidden_dim=args.hidden_dim,
            optimization_steps=args.optimization_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            positive_weight=args.router_positive_weight,
            threshold=args.router_threshold,
            save_best=bool(args.save_best),
            log_every=args.log_every,
            device=device,
        )
        report = {
            "schema": "jepa_phase3a_value_map_router_training_report_v0",
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_completed_steps": base_report.get("completed_steps"),
            "latent_map_head": str(args.latent_map_head.resolve()),
            "latent_map_completed_steps": latent_map_report.get("completed_steps"),
            "latent_memory_updater": str(args.latent_memory_updater.resolve()),
            "latent_memory_completed_steps": latent_memory_report.get(
                "completed_steps"
            ),
            "train_data": str(args.train_data.resolve()),
            "validation_data": str(args.validation_data.resolve()),
            "train_seed": train_seed,
            "validation_seed": validation_seed,
            "memory_size": args.memory_size,
            "router_label_source": args.router_label_source,
            "router_synthetic_examples": int(args.router_synthetic_examples),
            "router_counterfactual_horizon": int(
                args.router_counterfactual_horizon
            ),
            "router_counterfactual_utility_margin": float(
                args.router_counterfactual_utility_margin
            ),
            "args": {
                key: _json_safe_arg(value)
                for key, value in vars(args).items()
            },
            **router_report,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_head_state_dict": router_head.state_dict(),
                "report": report,
            },
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

    if args.train_router_only and args.router_label_source == "trace_outcome":
        train_router_examples, train_trace_stats = _build_trace_outcome_router_examples(
            list(args.trace_memory_data),
            list(args.trace_memory_source_data),
            list(args.router_trace_primary_report),
            list(args.router_trace_fallback_report),
            fallback_rows=train_rows,
            fallback_scene_seed=train_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            max_episodes=args.max_train_episodes,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            latent_map_marker_threshold=args.latent_map_marker_threshold,
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            pre_memory_marker_only=bool(args.router_trace_pre_memory_marker_only),
            same_outcome_negative=bool(args.router_trace_same_outcome_negative),
            repeat=int(args.trace_memory_repeat),
            device=device,
        )
        validation_router_examples, validation_trace_stats = (
            _build_trace_outcome_router_examples(
                list(args.router_validation_trace_memory_data),
                list(args.router_validation_trace_memory_source_data),
                list(args.router_validation_trace_primary_report),
                list(args.router_validation_trace_fallback_report),
                fallback_rows=validation_rows,
                fallback_scene_seed=validation_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                max_episodes=args.max_validation_episodes,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                latent_map_marker_threshold=args.latent_map_marker_threshold,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                pre_memory_marker_only=bool(args.router_trace_pre_memory_marker_only),
                same_outcome_negative=bool(args.router_trace_same_outcome_negative),
                repeat=1,
                device=device,
            )
        )
        router_head, router_report = _train_router_head(
            train_router_examples,
            validation_router_examples,
            memory_size=args.memory_size,
            hidden_dim=args.hidden_dim,
            optimization_steps=args.optimization_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            positive_weight=args.router_positive_weight,
            threshold=args.router_threshold,
            save_best=bool(args.save_best),
            log_every=args.log_every,
            device=device,
        )
        report = {
            "schema": "jepa_phase3a_value_map_router_training_report_v0",
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_completed_steps": base_report.get("completed_steps"),
            "latent_map_head": str(args.latent_map_head.resolve()),
            "latent_map_completed_steps": latent_map_report.get("completed_steps"),
            "latent_memory_updater": str(args.latent_memory_updater.resolve()),
            "latent_memory_completed_steps": latent_memory_report.get(
                "completed_steps"
            ),
            "train_data": str(args.train_data.resolve()),
            "validation_data": str(args.validation_data.resolve()),
            "train_seed": train_seed,
            "validation_seed": validation_seed,
            "memory_size": args.memory_size,
            "router_label_source": args.router_label_source,
            "router_trace_pre_memory_marker_only": bool(
                args.router_trace_pre_memory_marker_only
            ),
            "router_trace_same_outcome_negative": bool(
                args.router_trace_same_outcome_negative
            ),
            "train_trace_outcome": train_trace_stats,
            "validation_trace_outcome": validation_trace_stats,
            "args": {
                key: _json_safe_arg(value)
                for key, value in vars(args).items()
            },
            **router_report,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_head_state_dict": router_head.state_dict(),
                "report": report,
            },
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

    if (
        args.train_router_only
        and args.router_label_source == "trace_action_preference"
    ):
        assert dagger_rollout_planner_head is not None
        assert dagger_rollout_fallback_head is not None
        train_router_examples, train_trace_stats = (
            _build_trace_action_preference_router_examples(
                list(args.trace_memory_data),
                list(args.trace_memory_source_data),
                fallback_rows=train_rows,
                fallback_scene_seed=train_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                max_episodes=args.max_train_episodes,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                primary_planner_head=dagger_rollout_planner_head,
                primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                fallback_planner_head=dagger_rollout_fallback_head,
                latent_map_marker_threshold=args.latent_map_marker_threshold,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                pre_memory_marker_only=bool(args.router_trace_pre_memory_marker_only),
                repeat=int(args.trace_memory_repeat),
                device=device,
            )
        )
        validation_router_examples, validation_trace_stats = (
            _build_trace_action_preference_router_examples(
                list(args.router_validation_trace_memory_data),
                list(args.router_validation_trace_memory_source_data),
                fallback_rows=validation_rows,
                fallback_scene_seed=validation_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                max_episodes=args.max_validation_episodes,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                primary_planner_head=dagger_rollout_planner_head,
                primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                fallback_planner_head=dagger_rollout_fallback_head,
                latent_map_marker_threshold=args.latent_map_marker_threshold,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                pre_memory_marker_only=bool(args.router_trace_pre_memory_marker_only),
                repeat=1,
                device=device,
            )
        )
        router_head, router_report = _train_router_head(
            train_router_examples,
            validation_router_examples,
            memory_size=args.memory_size,
            hidden_dim=args.hidden_dim,
            optimization_steps=args.optimization_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            positive_weight=args.router_positive_weight,
            threshold=args.router_threshold,
            save_best=bool(args.save_best),
            log_every=args.log_every,
            device=device,
        )
        report = {
            "schema": "jepa_phase3a_value_map_router_training_report_v0",
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_completed_steps": base_report.get("completed_steps"),
            "latent_map_head": str(args.latent_map_head.resolve()),
            "latent_map_completed_steps": latent_map_report.get("completed_steps"),
            "latent_memory_updater": str(args.latent_memory_updater.resolve()),
            "latent_memory_completed_steps": latent_memory_report.get(
                "completed_steps"
            ),
            "train_data": str(args.train_data.resolve()),
            "validation_data": str(args.validation_data.resolve()),
            "train_seed": train_seed,
            "validation_seed": validation_seed,
            "memory_size": args.memory_size,
            "router_label_source": args.router_label_source,
            "router_trace_pre_memory_marker_only": bool(
                args.router_trace_pre_memory_marker_only
            ),
            "train_trace_action_preference": train_trace_stats,
            "validation_trace_action_preference": validation_trace_stats,
            "args": {
                key: _json_safe_arg(value)
                for key, value in vars(args).items()
            },
            **router_report,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_head_state_dict": router_head.state_dict(),
                "report": report,
            },
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

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
    train_memories = train_memory_examples.memories
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
        train_memories = torch.cat([train_memories, extra_train_examples.memories], dim=0)
    if dagger_rollout_planner_head is not None and not (
        args.train_router_only and args.router_label_source == "counterfactual"
    ):
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
            rollout_planner_head=dagger_rollout_planner_head,
            rollout_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
            rollout_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
            rollout_fallback_head=dagger_rollout_fallback_head,
            rollout_fallback_after_step=(
                args.dagger_rollout_value_map_fallback_after_step
            ),
            rollout_turn_oscillation_breaker=bool(
                args.dagger_rollout_value_map_turn_oscillation_breaker
            ),
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            include_marker_start_groups=args.include_marker_start_train_groups,
            device=device,
        )
        train_memories = torch.cat([train_memories, dagger_memories], dim=0)
    trace_memory_count = 0
    if args.trace_memory_data and args.trace_action_teacher_source == "none":
        if args.trace_memory_source_data:
            trace_memory_parts = []
            for trace_path, source_path in zip(
                args.trace_memory_data,
                args.trace_memory_source_data,
                strict=True,
            ):
                trace_seed = _infer_scene_seed(source_path)
                if trace_seed is None:
                    raise SystemExit(
                        "could not infer trace-memory source scene seed "
                        f"from {source_path}"
                    )
                trace_memory_parts.append(
                    _build_trace_memories(
                        [trace_path],
                        read_jsonl(source_path),
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
                        min_first_marker_step=args.trace_memory_min_first_marker_step,
                        pre_latent_marker_only=bool(
                            args.trace_memory_pre_latent_marker_only
                        ),
                        repeat=int(args.trace_memory_repeat),
                        device=device,
                    )
                )
            trace_memories = torch.cat(trace_memory_parts, dim=0)
        else:
            trace_rows = (
                train_rows if args.trace_memory_source == "train" else validation_rows
            )
            trace_seed = (
                train_seed if args.trace_memory_source == "train" else validation_seed
            )
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
                min_first_marker_step=args.trace_memory_min_first_marker_step,
                pre_latent_marker_only=bool(args.trace_memory_pre_latent_marker_only),
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
    validation_memories = validation_memory_examples.memories
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
        validation_memories = torch.cat(
            [validation_memories, extra_validation_examples.memories],
            dim=0,
        )
    if args.train_router_only:
        if args.router_label_source == "counterfactual":
            assert dagger_rollout_planner_head is not None
            assert dagger_rollout_fallback_head is not None
            train_router_examples = _build_counterfactual_router_examples(
                train_rows,
                scene_seed=train_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                max_episodes=args.max_train_episodes,
                max_steps=args.max_steps,
                horizon=args.router_counterfactual_horizon,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                primary_planner_head=dagger_rollout_planner_head,
                primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                fallback_planner_head=dagger_rollout_fallback_head,
                fallback_after_step=args.dagger_rollout_value_map_fallback_after_step,
                utility_margin=args.router_counterfactual_utility_margin,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                include_marker_start_groups=args.include_marker_start_train_groups,
                device=device,
            )
            for extra_train_path in args.extra_train_data:
                extra_train_seed = _infer_scene_seed(extra_train_path)
                if extra_train_seed is None:
                    raise SystemExit(
                        "could not infer extra train scene seed from "
                        f"{extra_train_path}"
                    )
                train_router_examples = _concat_router_examples(
                    train_router_examples,
                    _build_counterfactual_router_examples(
                        read_jsonl(extra_train_path),
                        scene_seed=extra_train_seed,
                        width=args.width_cells,
                        height=args.height_cells,
                        view_size=args.view_size,
                        memory_size=args.memory_size,
                        max_episodes=args.max_train_episodes,
                        max_steps=args.max_steps,
                        horizon=args.router_counterfactual_horizon,
                        base_model=base_model,
                        latent_map_head=latent_map_head,
                        latent_memory_updater=latent_memory_updater,
                        value_field_head=value_field_head,
                        extractor_head=extractor_head,
                        primary_planner_head=dagger_rollout_planner_head,
                        primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                        primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                        fallback_planner_head=dagger_rollout_fallback_head,
                        fallback_after_step=(
                            args.dagger_rollout_value_map_fallback_after_step
                        ),
                        utility_margin=args.router_counterfactual_utility_margin,
                        blocked_threshold=args.latent_memory_blocked_threshold,
                        free_threshold=args.latent_memory_free_threshold,
                        marker_threshold=args.latent_memory_marker_threshold,
                        include_marker_start_groups=(
                            args.include_marker_start_train_groups
                        ),
                        device=device,
                    ),
                )
            validation_router_examples = _build_counterfactual_router_examples(
                validation_rows,
                scene_seed=validation_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                max_episodes=args.max_validation_episodes,
                max_steps=args.max_steps,
                horizon=args.router_counterfactual_horizon,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                primary_planner_head=dagger_rollout_planner_head,
                primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                fallback_planner_head=dagger_rollout_fallback_head,
                fallback_after_step=args.dagger_rollout_value_map_fallback_after_step,
                utility_margin=args.router_counterfactual_utility_margin,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                include_marker_start_groups=False,
                device=device,
            )
            for extra_validation_path in args.extra_validation_data:
                extra_validation_seed = _infer_scene_seed(extra_validation_path)
                if extra_validation_seed is None:
                    raise SystemExit(
                        "could not infer extra validation scene seed from "
                        f"{extra_validation_path}"
                    )
                validation_router_examples = _concat_router_examples(
                    validation_router_examples,
                    _build_counterfactual_router_examples(
                        read_jsonl(extra_validation_path),
                        scene_seed=extra_validation_seed,
                        width=args.width_cells,
                        height=args.height_cells,
                        view_size=args.view_size,
                        memory_size=args.memory_size,
                        max_episodes=args.max_validation_episodes,
                        max_steps=args.max_steps,
                        horizon=args.router_counterfactual_horizon,
                        base_model=base_model,
                        latent_map_head=latent_map_head,
                        latent_memory_updater=latent_memory_updater,
                        value_field_head=value_field_head,
                        extractor_head=extractor_head,
                        primary_planner_head=dagger_rollout_planner_head,
                        primary_ensemble_heads=tuple(dagger_rollout_ensemble_heads),
                        primary_ensemble_mode=args.dagger_rollout_value_map_ensemble_mode,
                        fallback_planner_head=dagger_rollout_fallback_head,
                        fallback_after_step=(
                            args.dagger_rollout_value_map_fallback_after_step
                        ),
                        utility_margin=args.router_counterfactual_utility_margin,
                        blocked_threshold=args.latent_memory_blocked_threshold,
                        free_threshold=args.latent_memory_free_threshold,
                        marker_threshold=args.latent_memory_marker_threshold,
                        include_marker_start_groups=False,
                        device=device,
                    ),
                )
        else:
            train_router_examples = _router_examples_from_memories(
                train_memories,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
            )
            if args.router_synthetic_examples > 0:
                train_router_examples = _concat_router_examples(
                    train_router_examples,
                    _build_synthetic_router_examples(
                        args.router_synthetic_examples,
                        memory_size=args.memory_size,
                        seed=args.synthetic_seed,
                    ),
                )
            validation_router_examples = _router_examples_from_memories(
                validation_memory_examples.memories,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
            )
        router_head, router_report = _train_router_head(
            train_router_examples,
            validation_router_examples,
            memory_size=args.memory_size,
            hidden_dim=args.hidden_dim,
            optimization_steps=args.optimization_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            positive_weight=args.router_positive_weight,
            threshold=args.router_threshold,
            save_best=bool(args.save_best),
            log_every=args.log_every,
            device=device,
        )
        report = {
            "schema": "jepa_phase3a_value_map_router_training_report_v0",
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
            "memory_size": args.memory_size,
            "router_label_source": args.router_label_source,
            "router_synthetic_examples": int(args.router_synthetic_examples),
            "router_counterfactual_horizon": int(args.router_counterfactual_horizon),
            "router_counterfactual_utility_margin": float(
                args.router_counterfactual_utility_margin
            ),
            "args": {
                key: _json_safe_arg(value)
                for key, value in vars(args).items()
            },
            **router_report,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_head_state_dict": router_head.state_dict(),
                "report": report,
            },
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    train_examples = _build_value_map_examples(
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
        explicit_frontier_targets=bool(args.explicit_frontier_targets),
        fixed_marker_targets=bool(args.fixed_marker_targets),
        value_iterations=args.value_iterations,
        value_gamma=args.value_gamma,
        batch_size=args.batch_size,
        device=device,
    )
    validation_examples = _build_value_map_examples(
        validation_memories,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        target_threshold=args.target_threshold,
        target_top_k=args.target_top_k,
        extractor_threshold=args.extractor_threshold,
        sparse_target_top_k=args.sparse_target_top_k,
        explicit_frontier_targets=bool(args.explicit_frontier_targets),
        fixed_marker_targets=bool(args.fixed_marker_targets),
        value_iterations=args.value_iterations,
        value_gamma=args.value_gamma,
        batch_size=args.batch_size,
        device=device,
    )
    if args.synthetic_examples > 0:
        synthetic_examples = _build_synthetic_value_map_examples(
            args.synthetic_examples,
            memory_size=args.memory_size,
            target_threshold=args.target_threshold,
            target_top_k=args.target_top_k,
            sparse_target_top_k=args.sparse_target_top_k,
            value_iterations=args.value_iterations,
            value_gamma=args.value_gamma,
            fixed_marker_targets=bool(args.fixed_marker_targets),
            seed=args.synthetic_seed,
        )
        train_examples = _concat_value_map_examples(train_examples, synthetic_examples)
    trace_action_count = 0
    trace_action_examples = None
    if args.trace_action_teacher_source != "none":
        if args.trace_memory_source_data:
            for trace_path, source_path in zip(
                args.trace_memory_data,
                args.trace_memory_source_data,
                strict=True,
            ):
                trace_seed = _infer_scene_seed(source_path)
                if trace_seed is None:
                    raise SystemExit(
                        "could not infer trace-memory source scene seed "
                        f"from {source_path}"
                    )
                paired_examples = _build_trace_action_value_map_examples(
                    [trace_path],
                    read_jsonl(source_path),
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
                    teacher_source=str(args.trace_action_teacher_source),
                    odom_lookahead_horizon=args.trace_odom_lookahead_horizon,
                    odom_lookahead_beam_width=args.trace_odom_lookahead_beam_width,
                    failed_only=bool(args.trace_memory_failed_only),
                    min_first_marker_step=args.trace_memory_min_first_marker_step,
                    pre_latent_marker_only=bool(args.trace_memory_pre_latent_marker_only),
                    repeat=int(args.trace_memory_repeat),
                    device=device,
                )
                trace_action_examples = (
                    paired_examples
                    if trace_action_examples is None
                    else _concat_value_map_examples(
                        trace_action_examples,
                        paired_examples,
                    )
                )
            assert trace_action_examples is not None
        else:
            trace_rows = (
                train_rows if args.trace_memory_source == "train" else validation_rows
            )
            trace_seed = (
                train_seed if args.trace_memory_source == "train" else validation_seed
            )
            trace_action_examples = _build_trace_action_value_map_examples(
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
                teacher_source=str(args.trace_action_teacher_source),
                odom_lookahead_horizon=args.trace_odom_lookahead_horizon,
                odom_lookahead_beam_width=args.trace_odom_lookahead_beam_width,
                failed_only=bool(args.trace_memory_failed_only),
                min_first_marker_step=args.trace_memory_min_first_marker_step,
                pre_latent_marker_only=bool(args.trace_memory_pre_latent_marker_only),
                repeat=int(args.trace_memory_repeat),
                device=device,
            )
        trace_action_count = int(len(trace_action_examples.actions))
    train_examples = _filter_value_map_examples(
        train_examples,
        mode=str(args.example_filter),
    )
    validation_examples = _filter_value_map_examples(
        validation_examples,
        mode=str(args.example_filter),
    )
    if trace_action_examples is not None:
        train_examples = _concat_value_map_examples(
            train_examples,
            trace_action_examples,
        )

    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.target_fields,
        train_examples.sparse_probabilities,
        train_examples.value_maps,
        train_examples.actions,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    planner_head = Phase3AValueMapPlannerHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
        architecture=args.planner_architecture,
        refinement_steps=args.refinement_steps,
    ).to(device)
    if args.init_planner_checkpoint is not None:
        try:
            init_checkpoint = torch.load(
                args.init_planner_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            init_checkpoint = torch.load(
                args.init_planner_checkpoint,
                map_location=device,
            )
        init_planner_report = init_checkpoint.get("report", {})
        init_config = init_planner_report.get("model_config", {})
        expected = {
            "memory_size": int(args.memory_size),
            "hidden_dim": int(args.hidden_dim),
            "memory_channels": 3,
            "architecture": str(args.planner_architecture),
            "refinement_steps": int(args.refinement_steps),
        }
        actual = {
            "memory_size": int(init_config.get("memory_size", args.memory_size)),
            "hidden_dim": int(init_config.get("hidden_dim", args.hidden_dim)),
            "memory_channels": int(init_config.get("memory_channels", 3)),
            "architecture": str(
                init_config.get("architecture", args.planner_architecture)
            ),
            "refinement_steps": int(
                init_config.get("refinement_steps", args.refinement_steps)
            ),
        }
        if actual != expected:
            raise SystemExit(
                "--init-planner-checkpoint config does not match requested "
                f"model config ({actual} != {expected})"
            )
        planner_head.load_state_dict(init_checkpoint["planner_head_state_dict"])
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
    best_score = (-1.0, -1.0, -1.0, float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, target, sparse_prob, value_map, action = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, sparse_prob, value_map, action = next(iterator)
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        value_map = value_map.to(device)
        action = action.to(device)
        logits = planner_head(memory, target, sparse_prob)
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
            score = (
                float(metrics["action_match"]),
                float(metrics["sparse_action_match"]),
                float(metrics["broad_action_match"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
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
        "schema": "jepa_phase3a_value_map_planner_training_report_v0",
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
        "init_planner_checkpoint": (
            str(args.init_planner_checkpoint.resolve())
            if args.init_planner_checkpoint
            else None
        ),
        "init_planner_completed_steps": (
            init_planner_report.get("completed_steps")
            if init_planner_report
            else None
        ),
        "init_planner_selected_step": (
            init_planner_report.get("selected_step")
            if init_planner_report
            else None
        ),
        "dagger_rollout_value_map_planner_head": (
            str(args.dagger_rollout_value_map_planner_head.resolve())
            if args.dagger_rollout_value_map_planner_head
            else None
        ),
        "dagger_rollout_value_map_planner_completed_steps": (
            dagger_rollout_planner_report.get("completed_steps")
            if dagger_rollout_planner_report
            else None
        ),
        "dagger_rollout_value_map_ensemble_heads": [
            str(path.resolve()) for path in args.dagger_rollout_value_map_ensemble_head
        ],
        "dagger_rollout_value_map_ensemble_completed_steps": [
            report.get("completed_steps")
            for report in dagger_rollout_ensemble_reports
        ],
        "dagger_rollout_value_map_ensemble_mode": (
            args.dagger_rollout_value_map_ensemble_mode
        ),
        "dagger_rollout_value_map_fallback_head": (
            str(args.dagger_rollout_value_map_fallback_head.resolve())
            if args.dagger_rollout_value_map_fallback_head
            else None
        ),
        "dagger_rollout_value_map_fallback_completed_steps": (
            dagger_rollout_fallback_report.get("completed_steps")
            if dagger_rollout_fallback_report
            else None
        ),
        "dagger_rollout_value_map_fallback_after_step": (
            args.dagger_rollout_value_map_fallback_after_step
        ),
        "dagger_rollout_value_map_turn_oscillation_breaker": bool(
            args.dagger_rollout_value_map_turn_oscillation_breaker
        ),
        "train_data": str(args.train_data.resolve()),
        "extra_train_data": [
            str(path.resolve()) for path in args.extra_train_data
        ],
        "validation_data": str(args.validation_data.resolve()),
        "extra_validation_data": [
            str(path.resolve()) for path in args.extra_validation_data
        ],
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "trace_memory_data": [
            str(path.resolve()) for path in args.trace_memory_data
        ],
        "trace_memory_source_data": [
            str(path.resolve()) for path in args.trace_memory_source_data
        ],
        "trace_memory_source": str(args.trace_memory_source),
        "trace_memory_failed_only": bool(args.trace_memory_failed_only),
        "trace_memory_min_first_marker_step": (
            None
            if args.trace_memory_min_first_marker_step is None
            else int(args.trace_memory_min_first_marker_step)
        ),
        "trace_memory_pre_latent_marker_only": bool(
            args.trace_memory_pre_latent_marker_only
        ),
        "trace_memory_repeat": int(args.trace_memory_repeat),
        "trace_memory_examples": int(trace_memory_count),
        "trace_action_teacher_source": str(args.trace_action_teacher_source),
        "trace_action_examples": int(trace_action_count),
        "train_synthetic_examples": int(args.synthetic_examples),
        "explicit_frontier_targets": bool(args.explicit_frontier_targets),
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
            "memory_channels": 3,
            "architecture": args.planner_architecture,
            "refinement_steps": args.refinement_steps,
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
