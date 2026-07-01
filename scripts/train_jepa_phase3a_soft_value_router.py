#!/usr/bin/env python3
"""Train a learned router from value-map readout to soft-value readout."""
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
from lewm.benchmarks.phase3a_training import source_key  # noqa: E402
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AValueMapRouterHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _break_turn_oscillation_action,
    _candidate_rows,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _infer_scene_seed,
    _latent_local_evidence,
    _latent_soft_value_map,
    _select_egocentric_learned_value_map_action,
    _selection_for_single_action,
    _sparse_target_tensor_from_memory,
    _state_from_dict,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_action_correction import (  # noqa: E402
    _json_safe_arg,
    _planned_value_map_action,
)
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_action import (  # noqa: E402
    _load_value_action_head,
    _load_value_extractor_head,
)
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _load_latent_memory_updater,
    _load_value_field_head,
)
from scripts.train_jepa_phase3a_value_map_planner import (  # noqa: E402
    _load_value_map_planner_head,
)


@dataclass(frozen=True)
class SoftRouterExamples:
    memories: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    baseline_actions: torch.Tensor
    soft_actions: torch.Tensor
    oracle_actions: torch.Tensor
    baseline_utilities: torch.Tensor
    soft_utilities: torch.Tensor
    oracle_utilities: torch.Tensor


@torch.no_grad()
def _rollout_claimed(
    template: dict,
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    max_steps: int,
    mode: str,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    value_map_planner_head: nn.Module,
    value_action_head: nn.Module,
    turn_oscillation_breaker: bool,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    latent_map_marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    soft_value_iterations: int,
    soft_value_gamma: float,
    device: torch.device,
) -> bool:
    scene = _goal_scene_from_row(template, seed=scene_seed, width=width, height=height)
    state = _state_from_dict(template["start_state"])
    recurrent_memory = torch.zeros(
        1,
        3,
        int(latent_memory_updater.memory_size),
        int(latent_memory_updater.memory_size),
        dtype=torch.float32,
        device=device,
    )
    last_action = "hold"
    last_collision = False
    latent_marker_seen_ever = False
    trajectory: list[dict] = []
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
            memory_size=int(recurrent_memory.shape[-1]),
        )
        latent_marker_seen_ever = latent_marker_seen_ever or (
            float(local_evidence[0, 2].max().detach().cpu())
            >= latent_map_marker_threshold
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
        planned_action, _planned_mode, target_probs_gpu, sparse_prob = (
            _planned_value_map_action(
                recurrent_memory_tensor=recurrent_memory,
                recurrent_memory=memory_dict,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                value_map_planner_head=value_map_planner_head,
                value_action_head=value_action_head,
                latent_marker_seen_ever=latent_marker_seen_ever,
                marker_action_return=True,
                marker_action_require_local_evidence=False,
                target_threshold=target_threshold,
                target_top_k=target_top_k,
                extractor_threshold=extractor_threshold,
                sparse_target_top_k=sparse_target_top_k,
                device=device,
            )
        )
        if mode == "soft":
            target_top_k_for_state = (
                sparse_target_top_k
                if float(sparse_prob.item()) >= float(extractor_threshold)
                else target_top_k
            )
            sparse_target_probs_gpu = _sparse_target_tensor_from_memory(
                memory_dict,
                target_probs_gpu[0, 0].detach().cpu(),
                threshold=target_threshold,
                top_k=target_top_k_for_state,
                device=device,
            )
            soft_value_probs = _latent_soft_value_map(
                recurrent_memory,
                sparse_target_probs_gpu,
                iterations=soft_value_iterations,
                gamma=soft_value_gamma,
            )
            planned_action, _soft_mode = _select_egocentric_learned_value_map_action(
                memory_dict,
                soft_value_probs[0, 0].detach().cpu(),
            )
        elif mode != "baseline":
            raise ValueError(f"unknown rollout mode: {mode!r}")
        if turn_oscillation_breaker:
            replacement_action = _break_turn_oscillation_action(
                planned_action,
                memory_dict,
                trajectory,
                state,
            )
            if replacement_action is not None:
                planned_action = replacement_action
        next_state, collision = step_state(scene, state, planned_action)
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
                "selected_action": planned_action,
                "collision": bool(collision),
            }
        )
        last_action = planned_action
        last_collision = bool(collision)
        state = next_state
        if (state.x, state.y) == scene.goal:
            return True
    return False


def _group_sources(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(source_key(row), []).append(row)
    return [grouped[key] for key in sorted(grouped)]


@torch.no_grad()
def _build_soft_router_examples(
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    horizon: int,
    max_episodes: int | None,
    max_steps: int,
    base_model: nn.Module,
    latent_map_head: nn.Module,
    latent_memory_updater: nn.Module,
    value_field_head: nn.Module,
    extractor_head: nn.Module,
    value_map_planner_head: nn.Module,
    value_action_head: nn.Module,
    turn_oscillation_breaker: bool,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    latent_map_marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    soft_value_iterations: int,
    soft_value_gamma: float,
    utility_margin: float,
    positive_weight: float,
    utility_weight: float,
    episode_label_source: str,
    max_candidates_per_state: int | None,
    device: torch.device,
) -> SoftRouterExamples:
    groups = _group_sources(rows)
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    labels = []
    weights = []
    baseline_actions = []
    soft_actions = []
    oracle_actions = []
    baseline_utilities = []
    soft_utilities = []
    oracle_utilities = []
    for group_index, group in enumerate(groups):
        template = group[0]
        episode_route_label: bool | None = None
        if episode_label_source == "episode_outcome":
            baseline_claimed = _rollout_claimed(
                template,
                scene_seed=scene_seed,
                width=width,
                height=height,
                view_size=view_size,
                max_steps=max_steps,
                mode="baseline",
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                value_map_planner_head=value_map_planner_head,
                value_action_head=value_action_head,
                turn_oscillation_breaker=turn_oscillation_breaker,
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
                latent_map_marker_threshold=latent_map_marker_threshold,
                target_threshold=target_threshold,
                target_top_k=target_top_k,
                extractor_threshold=extractor_threshold,
                sparse_target_top_k=sparse_target_top_k,
                soft_value_iterations=soft_value_iterations,
                soft_value_gamma=soft_value_gamma,
                device=device,
            )
            soft_claimed = _rollout_claimed(
                template,
                scene_seed=scene_seed,
                width=width,
                height=height,
                view_size=view_size,
                max_steps=max_steps,
                mode="soft",
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                value_field_head=value_field_head,
                extractor_head=extractor_head,
                value_map_planner_head=value_map_planner_head,
                value_action_head=value_action_head,
                turn_oscillation_breaker=turn_oscillation_breaker,
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
                latent_map_marker_threshold=latent_map_marker_threshold,
                target_threshold=target_threshold,
                target_top_k=target_top_k,
                extractor_threshold=extractor_threshold,
                sparse_target_top_k=sparse_target_top_k,
                soft_value_iterations=soft_value_iterations,
                soft_value_gamma=soft_value_gamma,
                device=device,
            )
            if soft_claimed and not baseline_claimed:
                episode_route_label = True
            elif baseline_claimed and not soft_claimed:
                episode_route_label = False
        elif episode_label_source != "immediate_utility":
            raise ValueError(f"unknown episode label source: {episode_label_source!r}")
        scene = _goal_scene_from_row(template, seed=scene_seed, width=width, height=height)
        state = _state_from_dict(template["start_state"])
        history_states = [_state_from_dict(item) for item in template["history_states"]]
        history_actions = [str(item) for item in template["history_primitive_sequence"]]
        recurrent_memory = torch.zeros(
            1,
            3,
            int(latent_memory_updater.memory_size),
            int(latent_memory_updater.memory_size),
            dtype=torch.float32,
            device=device,
        )
        last_action = "hold"
        last_collision = False
        latent_marker_seen_ever = False
        trajectory: list[dict] = []
        for step in range(max_steps):
            rows_for_state = _candidate_rows(
                scene=scene,
                source_index=step,
                state=state,
                history_states=history_states,
                history_actions=history_actions,
                horizon=horizon,
                view_size=view_size,
                current_goal_marker=True,
            )
            if max_candidates_per_state is not None:
                by_first: dict[str, list[dict]] = {}
                for row in rows_for_state:
                    by_first.setdefault(str(row["primitive_sequence"][0]), []).append(row)
                rows_for_state = [
                    item
                    for first_action in ACTION_NAMES
                    for item in by_first.get(first_action, [])[:max_candidates_per_state]
                ] or rows_for_state
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
                memory_size=int(recurrent_memory.shape[-1]),
            )
            latent_marker_seen_ever = latent_marker_seen_ever or (
                float(local_evidence[0, 2].max().detach().cpu())
                >= latent_map_marker_threshold
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
            baseline_action, _baseline_mode, target_probs_gpu, sparse_prob = (
                _planned_value_map_action(
                    recurrent_memory_tensor=recurrent_memory,
                    recurrent_memory=memory_dict,
                    value_field_head=value_field_head,
                    extractor_head=extractor_head,
                    value_map_planner_head=value_map_planner_head,
                    value_action_head=value_action_head,
                    latent_marker_seen_ever=latent_marker_seen_ever,
                    marker_action_return=True,
                    marker_action_require_local_evidence=False,
                    target_threshold=target_threshold,
                    target_top_k=target_top_k,
                    extractor_threshold=extractor_threshold,
                    sparse_target_top_k=sparse_target_top_k,
                    device=device,
                )
            )
            target_top_k_for_state = (
                sparse_target_top_k
                if float(sparse_prob.item()) >= float(extractor_threshold)
                else target_top_k
            )
            sparse_target_probs_gpu = _sparse_target_tensor_from_memory(
                memory_dict,
                target_probs_gpu[0, 0].detach().cpu(),
                threshold=target_threshold,
                top_k=target_top_k_for_state,
                device=device,
            )
            soft_value_probs = _latent_soft_value_map(
                recurrent_memory,
                sparse_target_probs_gpu,
                iterations=soft_value_iterations,
                gamma=soft_value_gamma,
            )
            soft_action, _soft_mode = _select_egocentric_learned_value_map_action(
                memory_dict,
                soft_value_probs[0, 0].detach().cpu(),
            )
            if turn_oscillation_breaker:
                replacement_action = _break_turn_oscillation_action(
                    baseline_action,
                    memory_dict,
                    trajectory,
                    state,
                )
                if replacement_action is not None:
                    baseline_action = replacement_action
                replacement_action = _break_turn_oscillation_action(
                    soft_action,
                    memory_dict,
                    trajectory,
                    state,
                )
                if replacement_action is not None:
                    soft_action = replacement_action
            baseline_selected, oracle = _selection_for_single_action(
                rows_for_state,
                baseline_action,
            )
            soft_selected, _soft_oracle = _selection_for_single_action(
                rows_for_state,
                soft_action,
            )
            baseline_utility = float(baseline_selected["utility"])
            soft_utility = float(soft_selected["utility"])
            oracle_utility = float(oracle["utility"])
            label = (
                bool(episode_route_label)
                if episode_route_label is not None
                else soft_utility > baseline_utility + float(utility_margin)
            )
            utility_delta = max(0.0, soft_utility - baseline_utility)
            memories.append(recurrent_memory[0].detach().cpu())
            labels.append(float(label))
            weights.append(
                1.0
                + (float(positive_weight) if label else 0.0)
                + float(utility_weight) * utility_delta
            )
            baseline_actions.append(ACTION_INDEX[baseline_action])
            soft_actions.append(ACTION_INDEX[soft_action])
            oracle_actions.append(ACTION_INDEX[str(oracle["action"])])
            baseline_utilities.append(baseline_utility)
            soft_utilities.append(soft_utility)
            oracle_utilities.append(oracle_utility)
            next_state, collision = step_state(scene, state, baseline_action)
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
                    "selected_action": baseline_action,
                    "collision": bool(collision),
                }
            )
            last_action = baseline_action
            last_collision = bool(collision)
            state = next_state
            history_states.append(state)
            history_actions.append(baseline_action)
            if (state.x, state.y) == scene.goal:
                break
        if (group_index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "built_router_groups": group_index + 1,
                        "router_examples": len(labels),
                        "router_positive_examples": int(sum(labels)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not labels:
        raise SystemExit("no router examples were generated")
    return SoftRouterExamples(
        memories=torch.stack(memories),
        labels=torch.tensor(labels, dtype=torch.float32),
        weights=torch.tensor(weights, dtype=torch.float32),
        baseline_actions=torch.tensor(baseline_actions, dtype=torch.long),
        soft_actions=torch.tensor(soft_actions, dtype=torch.long),
        oracle_actions=torch.tensor(oracle_actions, dtype=torch.long),
        baseline_utilities=torch.tensor(baseline_utilities, dtype=torch.float32),
        soft_utilities=torch.tensor(soft_utilities, dtype=torch.float32),
        oracle_utilities=torch.tensor(oracle_utilities, dtype=torch.float32),
    )


@torch.no_grad()
def _evaluate_router(
    router_head: Phase3AValueMapRouterHead,
    examples: SoftRouterExamples,
    *,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    router_head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.labels,
        examples.weights,
        examples.baseline_actions,
        examples.soft_actions,
        examples.oracle_actions,
        examples.baseline_utilities,
        examples.soft_utilities,
        examples.oracle_utilities,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    label_matches = 0
    positives = 0
    predicted_positives = 0
    true_positive = 0
    chosen_action_matches = 0
    baseline_action_matches = 0
    soft_action_matches = 0
    chosen_utility_total = 0.0
    baseline_utility_total = 0.0
    soft_utility_total = 0.0
    oracle_utility_total = 0.0
    for (
        memory,
        label,
        weight,
        baseline_action,
        soft_action,
        oracle_action,
        baseline_utility,
        soft_utility,
        oracle_utility,
    ) in loader:
        memory = memory.to(device)
        label = label.to(device)
        weight = weight.to(device)
        baseline_action = baseline_action.to(device)
        soft_action = soft_action.to(device)
        oracle_action = oracle_action.to(device)
        baseline_utility = baseline_utility.to(device)
        soft_utility = soft_utility.to(device)
        oracle_utility = oracle_utility.to(device)
        logits = router_head(memory).view(-1)
        loss_items = F.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        )
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        probability = logits.sigmoid()
        route = probability >= float(threshold)
        label_bool = label >= 0.5
        chosen_action = torch.where(route, soft_action, baseline_action)
        chosen_utility = torch.where(route, soft_utility, baseline_utility)
        total += int(memory.shape[0])
        loss_total += float(loss.item()) * int(memory.shape[0])
        label_matches += int((route == label_bool).sum().item())
        positives += int(label_bool.sum().item())
        predicted_positives += int(route.sum().item())
        true_positive += int((route & label_bool).sum().item())
        chosen_action_matches += int((chosen_action == oracle_action).sum().item())
        baseline_action_matches += int((baseline_action == oracle_action).sum().item())
        soft_action_matches += int((soft_action == oracle_action).sum().item())
        chosen_utility_total += float(chosen_utility.sum().item())
        baseline_utility_total += float(baseline_utility.sum().item())
        soft_utility_total += float(soft_utility.sum().item())
        oracle_utility_total += float(oracle_utility.sum().item())
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "label_match": label_matches / max(total, 1),
        "positive_examples": positives,
        "predicted_positive_examples": predicted_positives,
        "positive_recall": true_positive / max(positives, 1),
        "positive_precision": true_positive / max(predicted_positives, 1),
        "chosen_action_match": chosen_action_matches / max(total, 1),
        "baseline_action_match": baseline_action_matches / max(total, 1),
        "soft_action_match": soft_action_matches / max(total, 1),
        "chosen_utility": chosen_utility_total / max(total, 1),
        "baseline_utility": baseline_utility_total / max(total, 1),
        "soft_utility": soft_utility_total / max(total, 1),
        "oracle_utility": oracle_utility_total / max(total, 1),
        "route_rate": predicted_positives / max(total, 1),
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
    parser.add_argument("--latent-value-map-planner-head", type=Path, required=True)
    parser.add_argument("--latent-value-action-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--max-candidates-per-state", type=int, default=None)
    parser.add_argument("--turn-oscillation-breaker", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--utility-margin", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument(
        "--episode-label-source",
        choices=("immediate_utility", "episode_outcome"),
        default="immediate_utility",
    )
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--sparse-target-top-k", type=int, default=1)
    parser.add_argument("--soft-value-iterations", type=int, default=64)
    parser.add_argument("--soft-value-gamma", type=float, default=0.97)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.horizon < 1:
        raise SystemExit("--horizon must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if not 0.0 < args.label_threshold < 1.0:
        raise SystemExit("--label-threshold must be in (0, 1)")
    if args.positive_weight < 0.0:
        raise SystemExit("--positive-weight must be non-negative")
    if args.utility_weight < 0.0:
        raise SystemExit("--utility-weight must be non-negative")
    if not 0.0 <= args.latent_map_marker_threshold <= 1.0:
        raise SystemExit("--latent-map-marker-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_blocked_threshold <= 1.0:
        raise SystemExit("--latent-memory-blocked-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_free_threshold <= 1.0:
        raise SystemExit("--latent-memory-free-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_marker_threshold <= 1.0:
        raise SystemExit("--latent-memory-marker-threshold must be in [0, 1]")
    if args.target_top_k < 1:
        raise SystemExit("--target-top-k must be positive")
    if args.sparse_target_top_k < 1:
        raise SystemExit("--sparse-target-top-k must be positive")
    if args.soft_value_iterations < 1:
        raise SystemExit("--soft-value-iterations must be positive")
    if not 0.0 < args.soft_value_gamma <= 1.0:
        raise SystemExit("--soft-value-gamma must be in (0, 1]")
    if args.max_candidates_per_state is not None and args.max_candidates_per_state < 1:
        raise SystemExit("--max-candidates-per-state must be positive")

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
    value_map_planner_head, planner_report = _load_value_map_planner_head(
        args.latent_value_map_planner_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    value_action_head, action_report = _load_value_action_head(
        args.latent_value_action_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    for name, size in (
        ("latent memory updater", latent_memory_updater.memory_size),
        ("value field head", value_field_head.memory_size),
        ("value extractor head", extractor_head.memory_size),
        ("value-map planner head", value_map_planner_head.memory_size),
        ("value-action head", value_action_head.memory_size),
    ):
        if int(size) != int(args.memory_size):
            raise SystemExit(f"--memory-size must match {name} size ({size})")

    common_build_args = {
        "width": args.width_cells,
        "height": args.height_cells,
        "view_size": args.view_size,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "base_model": base_model,
        "latent_map_head": latent_map_head,
        "latent_memory_updater": latent_memory_updater,
        "value_field_head": value_field_head,
        "extractor_head": extractor_head,
        "value_map_planner_head": value_map_planner_head,
        "value_action_head": value_action_head,
        "turn_oscillation_breaker": bool(args.turn_oscillation_breaker),
        "blocked_threshold": args.latent_memory_blocked_threshold,
        "free_threshold": args.latent_memory_free_threshold,
        "marker_threshold": args.latent_memory_marker_threshold,
        "latent_map_marker_threshold": args.latent_map_marker_threshold,
        "target_threshold": args.target_threshold,
        "target_top_k": args.target_top_k,
        "extractor_threshold": args.extractor_threshold,
        "sparse_target_top_k": args.sparse_target_top_k,
        "soft_value_iterations": args.soft_value_iterations,
        "soft_value_gamma": args.soft_value_gamma,
        "utility_margin": args.utility_margin,
        "positive_weight": args.positive_weight,
        "utility_weight": args.utility_weight,
        "episode_label_source": args.episode_label_source,
        "max_candidates_per_state": args.max_candidates_per_state,
        "device": device,
    }
    train_examples = _build_soft_router_examples(
        train_rows,
        scene_seed=train_seed,
        max_episodes=args.max_train_episodes,
        **common_build_args,
    )
    validation_examples = _build_soft_router_examples(
        validation_rows,
        scene_seed=validation_seed,
        max_episodes=args.max_validation_episodes,
        **common_build_args,
    )

    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.labels,
        train_examples.weights,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    router_head = Phase3AValueMapRouterHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = (-1.0, -1.0, -1.0, -float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, label, weight = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, label, weight = next(iterator)
        memory = memory.to(device)
        label = label.to(device)
        weight = weight.to(device)
        logits = router_head(memory).view(-1)
        loss_items = F.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        )
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate_router(
                router_head,
                validation_examples,
                threshold=args.label_threshold,
                batch_size=args.batch_size,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["chosen_utility"]),
                float(metrics["chosen_action_match"]),
                float(metrics["positive_precision"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in router_head.state_dict().items()
                }

    final_validation = _evaluate_router(
        router_head,
        validation_examples,
        threshold=args.label_threshold,
        batch_size=args.batch_size,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        router_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation

    report = {
        "schema": "jepa_phase3a_soft_value_router_training_report_v0",
        "router_label_source": args.episode_label_source,
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
        "latent_value_map_planner_head": str(
            args.latent_value_map_planner_head.resolve()
        ),
        "latent_value_map_planner_completed_steps": planner_report.get(
            "completed_steps"
        ),
        "latent_value_action_head": str(args.latent_value_action_head.resolve()),
        "latent_value_action_completed_steps": action_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.labels)),
        "train_positive_examples": int(train_examples.labels.sum().item()),
        "validation_examples": int(len(validation_examples.labels)),
        "validation_positive_examples": int(validation_examples.labels.sum().item()),
        "completed_steps": args.optimization_steps,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "logs": logs,
        "args": {
            key: _json_safe_arg(value)
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "router_head_state_dict": router_head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
