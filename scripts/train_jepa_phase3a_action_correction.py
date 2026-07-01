#!/usr/bin/env python3
"""Train a closed-loop action-correction head for Phase 3A navigation."""
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
    Phase3AActionCorrectionHead,
    Phase3AValueFieldActionHead,
    Phase3AValueFieldExtractorHead,
    Phase3AValueMapPlannerHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _apply_action_correction,
    _break_turn_oscillation_action,
    _candidate_rows,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _group_validation_sources,
    _infer_scene_seed,
    _latent_local_evidence,
    _select_egocentric_learned_value_map_action,
    _select_odom_frontier_action,
    _select_odom_frontier_lookahead_action,
    _selection_for_single_action,
    _state_from_dict,
    _update_odom_frontier_memory,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
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
class CorrectionExamples:
    memories: torch.Tensor
    target_fields: torch.Tensor
    sparse_probabilities: torch.Tensor
    planned_actions: torch.Tensor
    marker_seen: torch.Tensor
    actions: torch.Tensor
    weights: torch.Tensor
    regrets: torch.Tensor


def _json_safe_arg(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    return value


def _group_sources(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(source_key(row), []).append(row)
    return [grouped[key] for key in sorted(grouped)]


def _concat_examples(*items: CorrectionExamples) -> CorrectionExamples:
    return CorrectionExamples(
        memories=torch.cat([item.memories for item in items], dim=0),
        target_fields=torch.cat([item.target_fields for item in items], dim=0),
        sparse_probabilities=torch.cat(
            [item.sparse_probabilities for item in items],
            dim=0,
        ),
        planned_actions=torch.cat([item.planned_actions for item in items], dim=0),
        marker_seen=torch.cat([item.marker_seen for item in items], dim=0),
        actions=torch.cat([item.actions for item in items], dim=0),
        weights=torch.cat([item.weights for item in items], dim=0),
        regrets=torch.cat([item.regrets for item in items], dim=0),
    )


@torch.no_grad()
def _planned_value_map_action(
    *,
    recurrent_memory_tensor: torch.Tensor,
    recurrent_memory: dict,
    value_field_head: nn.Module,
    extractor_head: Phase3AValueFieldExtractorHead,
    value_map_planner_head: Phase3AValueMapPlannerHead,
    value_action_head: Phase3AValueFieldActionHead | None,
    latent_marker_seen_ever: bool,
    marker_action_return: bool,
    marker_action_require_local_evidence: bool,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    device: torch.device,
) -> tuple[str, str, torch.Tensor, torch.Tensor]:
    target_fields_gpu = value_field_head(recurrent_memory_tensor).sigmoid()
    target_fields = target_fields_gpu[0].detach().cpu()
    marker = recurrent_memory.get("marker")
    if (
        int(target_fields.shape[0]) >= 2
        and marker is not None
        and marker in recurrent_memory["free"]
    ):
        target_probs_gpu = target_fields_gpu[:, 0:1]
    elif int(target_fields.shape[0]) >= 2:
        target_probs_gpu = target_fields_gpu[:, 1:2]
    else:
        target_probs_gpu = target_fields_gpu[:, 0:1]
    sparse_prob = extractor_head(recurrent_memory_tensor).sigmoid()
    _top_k = (
        sparse_target_top_k
        if float(sparse_prob.item()) >= extractor_threshold
        else target_top_k
    )
    _ = (target_threshold, _top_k, device)
    value_probs = value_map_planner_head(
        recurrent_memory_tensor,
        target_probs_gpu,
        sparse_prob,
    ).sigmoid()
    planned_action, planned_mode = _select_egocentric_learned_value_map_action(
        recurrent_memory,
        value_probs[0, 0].detach().cpu(),
    )
    if (
        marker_action_return
        and marker is not None
        and marker in recurrent_memory["free"]
        and value_action_head is not None
        and (not marker_action_require_local_evidence or latent_marker_seen_ever)
    ):
        action_logits = value_action_head(
            recurrent_memory_tensor,
            target_probs_gpu,
            sparse_prob,
        )
        planned_action = ACTION_NAMES[int(action_logits.argmax(dim=1).item())]
        planned_mode = "marker_action_return"
    return planned_action, planned_mode, target_probs_gpu, sparse_prob


def _keep_correction_example(
    *,
    example_filter: str,
    planned_mode: str,
    planned_action: str,
    target_action: str,
    latent_marker_seen_ever: bool,
    current_marker_seen: bool,
) -> bool:
    if example_filter == "all":
        return True
    if example_filter == "current_marker_seen":
        return bool(current_marker_seen)
    if example_filter == "marker_return":
        return "marker_action_return" in planned_mode
    if example_filter == "latent_marker_seen":
        return bool(latent_marker_seen_ever)
    if example_filter == "pre_latent_marker":
        return not bool(latent_marker_seen_ever)
    if example_filter == "correction_needed":
        return target_action != planned_action
    if example_filter == "pre_latent_marker_correction_needed":
        return (not bool(latent_marker_seen_ever)) and target_action != planned_action
    raise ValueError(f"unknown correction example filter: {example_filter}")


def _correction_memory_input(
    recurrent_memory: torch.Tensor,
    local_evidence: torch.Tensor,
    *,
    include_local_evidence_channels: bool,
) -> torch.Tensor:
    if include_local_evidence_channels:
        return torch.cat([recurrent_memory, local_evidence], dim=1)
    return recurrent_memory


@torch.no_grad()
def _build_correction_examples(
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
    extractor_head: Phase3AValueFieldExtractorHead,
    value_map_planner_head: Phase3AValueMapPlannerHead,
    value_action_head: Phase3AValueFieldActionHead | None,
    marker_action_return: bool,
    marker_action_require_local_evidence: bool,
    turn_oscillation_breaker: bool,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    latent_map_marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    regret_weight: float,
    mismatch_weight: float,
    max_candidates_per_state: int | None,
    example_filter: str,
    teacher_source: str,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
    include_local_evidence_channels: bool,
    device: torch.device,
) -> CorrectionExamples:
    groups = _group_sources(rows)
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    target_fields = []
    sparse_probabilities = []
    planned_actions = []
    marker_seen_values = []
    actions = []
    weights = []
    regrets = []
    for group_index, group in enumerate(groups):
        template = group[0]
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
        odom_frontier_memory = {
            "free": set(),
            "blocked": set(),
            "marker": None,
            "radius": max(int(latent_memory_updater.memory_size) // 2, 0),
        }
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
            current_marker_seen = (
                float(local_evidence[0, 2].max().detach().cpu())
                >= latent_map_marker_threshold
            )
            latent_marker_seen_ever = latent_marker_seen_ever or current_marker_seen
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
            planned_action, planned_mode, target_probs_gpu, sparse_prob = (
                _planned_value_map_action(
                    recurrent_memory_tensor=recurrent_memory,
                    recurrent_memory=memory_dict,
                    value_field_head=value_field_head,
                    extractor_head=extractor_head,
                    value_map_planner_head=value_map_planner_head,
                    value_action_head=value_action_head,
                    latent_marker_seen_ever=latent_marker_seen_ever,
                    marker_action_return=marker_action_return,
                    marker_action_require_local_evidence=(
                        marker_action_require_local_evidence
                    ),
                    target_threshold=target_threshold,
                    target_top_k=target_top_k,
                    extractor_threshold=extractor_threshold,
                    sparse_target_top_k=sparse_target_top_k,
                    device=device,
                )
            )
            if turn_oscillation_breaker:
                replacement_action = _break_turn_oscillation_action(
                    planned_action,
                    memory_dict,
                    trajectory,
                    state,
                )
                if replacement_action is not None:
                    planned_action = replacement_action
                    planned_mode = f"{planned_mode}_turnbreak"
            selected, oracle = _selection_for_single_action(rows_for_state, planned_action)
            if teacher_source == "local_oracle":
                target_action = str(oracle["action"])
                target_utility = float(oracle["utility"])
            elif teacher_source == "odom_frontier":
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
                target_selected, _target_oracle = _selection_for_single_action(
                    rows_for_state,
                    target_action,
                )
                target_utility = float(target_selected["utility"])
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
                target_selected, _target_oracle = _selection_for_single_action(
                    rows_for_state,
                    target_action,
                )
                target_utility = float(target_selected["utility"])
            else:
                raise ValueError(f"unknown correction teacher source: {teacher_source!r}")
            regret = max(0.0, target_utility - float(selected["utility"]))
            mismatch = target_action != planned_action
            if _keep_correction_example(
                example_filter=example_filter,
                planned_mode=planned_mode,
                planned_action=planned_action,
                target_action=target_action,
                latent_marker_seen_ever=latent_marker_seen_ever,
                current_marker_seen=current_marker_seen,
            ):
                correction_memory = _correction_memory_input(
                    recurrent_memory,
                    local_evidence,
                    include_local_evidence_channels=include_local_evidence_channels,
                )
                memories.append(correction_memory[0].detach().cpu())
                target_fields.append(target_probs_gpu[0].detach().cpu())
                sparse_probabilities.append(float(sparse_prob.item()))
                planned_actions.append(ACTION_INDEX[planned_action])
                marker_seen_values.append(float(latent_marker_seen_ever))
                actions.append(ACTION_INDEX[target_action])
                weights.append(
                    1.0
                    + (mismatch_weight if mismatch else 0.0)
                    + float(regret_weight) * regret
                )
                regrets.append(regret)
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
            history_states.append(state)
            history_actions.append(planned_action)
            if (state.x, state.y) == scene.goal:
                break
        if (group_index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "built_correction_groups": group_index + 1,
                        "correction_examples": len(actions),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not actions:
        raise SystemExit("no correction examples were generated")
    return CorrectionExamples(
        memories=torch.stack(memories),
        target_fields=torch.stack(target_fields),
        sparse_probabilities=torch.tensor(sparse_probabilities, dtype=torch.float32),
        planned_actions=torch.tensor(planned_actions, dtype=torch.long),
        marker_seen=torch.tensor(marker_seen_values, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        weights=torch.tensor(weights, dtype=torch.float32),
        regrets=torch.tensor(regrets, dtype=torch.float32),
    )


@torch.no_grad()
def _build_trace_correction_examples(
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
    extractor_head: Phase3AValueFieldExtractorHead,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    latent_map_marker_threshold: float,
    target_threshold: float,
    target_top_k: int,
    extractor_threshold: float,
    sparse_target_top_k: int,
    mismatch_weight: float,
    regret_weight: float,
    teacher_source: str,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
    planned_source: str,
    value_map_planner_head: Phase3AValueMapPlannerHead,
    value_action_head: Phase3AValueFieldActionHead | None,
    marker_action_return: bool,
    marker_action_require_local_evidence: bool,
    episode_indices: tuple[int, ...],
    failed_only: bool,
    pre_latent_marker_only: bool,
    post_latent_marker_only: bool,
    current_marker_only: bool,
    marker_return_only: bool,
    correction_needed_only: bool,
    repeat: int,
    include_local_evidence_channels: bool,
    device: torch.device,
) -> CorrectionExamples:
    groups = _group_validation_sources(rows)
    memories = []
    target_fields = []
    sparse_probabilities = []
    planned_actions = []
    marker_seen_values = []
    actions = []
    weights = []
    regrets = []
    episode_index_filter = set(int(index) for index in episode_indices)
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        episodes = trace.get("episodes", [])
        for episode_index, episode in enumerate(episodes):
            if episode_index_filter and episode_index not in episode_index_filter:
                continue
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
                if teacher_source == "local_oracle":
                    target_action = str(item.get("oracle_action", ""))
                    if target_action not in ACTION_INDEX:
                        raise SystemExit(
                            f"{trace_path} episode {episode_index} step "
                            f"{item.get('step')} has invalid oracle_action "
                            f"{target_action!r}"
                        )
                elif teacher_source == "odom_frontier":
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
                elif teacher_source == "trace_selected":
                    target_action = str(item["selected_action"])
                else:
                    raise ValueError(
                        f"unknown trace correction teacher source: {teacher_source!r}"
                    )
                if planned_source == "trace_selected":
                    planned_action = str(item["selected_action"])
                elif planned_source == "trace_pre_correction":
                    planned_action = str(
                        item.get("soft_value_router_action")
                        or item.get("marker_action_return_action")
                        or item.get("latent_value_action_candidate_action")
                        or item["selected_action"]
                    )
                elif planned_source == "value_map_planner":
                    planned_action, _planned_mode, target_probs_gpu, sparse_prob = (
                        _planned_value_map_action(
                            recurrent_memory_tensor=recurrent_memory,
                            recurrent_memory=memory_dict,
                            value_field_head=value_field_head,
                            extractor_head=extractor_head,
                            value_map_planner_head=value_map_planner_head,
                            value_action_head=value_action_head,
                            latent_marker_seen_ever=latent_marker_seen_ever,
                            marker_action_return=marker_action_return,
                            marker_action_require_local_evidence=(
                                marker_action_require_local_evidence
                            ),
                            target_threshold=target_threshold,
                            target_top_k=target_top_k,
                            extractor_threshold=extractor_threshold,
                            sparse_target_top_k=sparse_target_top_k,
                            device=device,
                        )
                    )
                else:
                    raise ValueError(
                        f"unknown trace correction planned source: {planned_source!r}"
                    )
                selection_mode = str(item.get("selection_mode", ""))
                if pre_latent_marker_only and latent_marker_seen_ever:
                    keep = False
                elif post_latent_marker_only and not latent_marker_seen_ever:
                    keep = False
                elif current_marker_only and not current_marker_seen:
                    keep = False
                elif marker_return_only and "marker_action_return" not in selection_mode:
                    keep = False
                elif correction_needed_only and target_action == planned_action:
                    keep = False
                else:
                    keep = True
                if keep:
                    mismatch = target_action != planned_action
                    correction_memory = _correction_memory_input(
                        recurrent_memory,
                        local_evidence,
                        include_local_evidence_channels=(
                            include_local_evidence_channels
                        ),
                    )
                    memories.append(correction_memory[0].detach().cpu())
                    target_fields.append(target_probs_gpu[0].detach().cpu())
                    sparse_probabilities.append(float(sparse_prob.item()))
                    planned_actions.append(ACTION_INDEX[planned_action])
                    marker_seen_values.append(float(latent_marker_seen_ever))
                    actions.append(ACTION_INDEX[target_action])
                    weights.append(1.0 + (mismatch_weight if mismatch else 0.0))
                    regrets.append(0.0)
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not actions:
        raise SystemExit("no trace correction examples were generated")
    examples = CorrectionExamples(
        memories=torch.stack(memories),
        target_fields=torch.stack(target_fields),
        sparse_probabilities=torch.tensor(sparse_probabilities, dtype=torch.float32),
        planned_actions=torch.tensor(planned_actions, dtype=torch.long),
        marker_seen=torch.tensor(marker_seen_values, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        weights=torch.tensor(weights, dtype=torch.float32),
        regrets=torch.tensor(regrets, dtype=torch.float32),
    )
    if repeat <= 1:
        return examples
    return CorrectionExamples(
        memories=examples.memories.repeat((repeat, 1, 1, 1)),
        target_fields=examples.target_fields.repeat((repeat, 1, 1, 1)),
        sparse_probabilities=examples.sparse_probabilities.repeat(repeat),
        planned_actions=examples.planned_actions.repeat(repeat),
        marker_seen=examples.marker_seen.repeat(repeat),
        actions=examples.actions.repeat(repeat),
        weights=examples.weights.repeat(repeat),
        regrets=examples.regrets.repeat(repeat),
    )


@torch.no_grad()
def _evaluate(
    correction_head: Phase3AActionCorrectionHead,
    examples: CorrectionExamples,
    *,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    correction_head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.target_fields,
        examples.sparse_probabilities,
        examples.planned_actions,
        examples.marker_seen,
        examples.actions,
        examples.weights,
        examples.regrets,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    head_matches = 0
    threshold_matches = 0
    planned_matches = 0
    correction_total = 0
    correction_matches = 0
    threshold_correction_matches = 0
    keep_total = 0
    keep_matches = 0
    override_total = 0
    action_counts = {name: 0 for name in ACTION_NAMES}
    planned_counts = {name: 0 for name in ACTION_NAMES}
    predicted_counts = {name: 0 for name in ACTION_NAMES}
    threshold_counts = {name: 0 for name in ACTION_NAMES}
    for (
        memory,
        target,
        sparse_prob,
        planned_action,
        marker_seen,
        action,
        weight,
        _regret,
    ) in loader:
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        planned_action = planned_action.to(device)
        marker_seen = marker_seen.to(device)
        action = action.to(device)
        weight = weight.to(device)
        logits = correction_head(
            memory,
            target,
            sparse_prob,
            planned_action,
            marker_seen,
        )
        loss_items = F.cross_entropy(logits, action, reduction="none")
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        probabilities = logits.softmax(dim=1)
        confidence, pred = probabilities.max(dim=1)
        threshold_pred = torch.where(
            confidence >= float(threshold),
            pred,
            planned_action,
        )
        match = pred == action
        threshold_match = threshold_pred == action
        planned_match = planned_action == action
        correction_mask = planned_action != action
        keep_mask = planned_action == action
        total += int(memory.shape[0])
        loss_total += float(loss.item()) * int(memory.shape[0])
        head_matches += int(match.sum().item())
        threshold_matches += int(threshold_match.sum().item())
        planned_matches += int(planned_match.sum().item())
        correction_total += int(correction_mask.sum().item())
        correction_matches += int((match & correction_mask).sum().item())
        threshold_correction_matches += int(
            (threshold_match & correction_mask).sum().item()
        )
        keep_total += int(keep_mask.sum().item())
        keep_matches += int((threshold_match & keep_mask).sum().item())
        override_total += int((threshold_pred != planned_action).sum().item())
        for action_index in action.detach().cpu().tolist():
            action_counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in planned_action.detach().cpu().tolist():
            planned_counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in pred.detach().cpu().tolist():
            predicted_counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in threshold_pred.detach().cpu().tolist():
            threshold_counts[ACTION_NAMES[int(action_index)]] += 1
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "planned_action_match": planned_matches / max(total, 1),
        "head_action_match": head_matches / max(total, 1),
        "threshold_action_match": threshold_matches / max(total, 1),
        "correction_examples": correction_total,
        "correction_head_match": correction_matches / max(correction_total, 1),
        "correction_threshold_match": (
            threshold_correction_matches / max(correction_total, 1)
        ),
        "keep_examples": keep_total,
        "keep_threshold_match": keep_matches / max(keep_total, 1),
        "override_rate": override_total / max(total, 1),
        "action_counts": action_counts,
        "planned_action_counts": planned_counts,
        "predicted_action_counts": predicted_counts,
        "threshold_action_counts": threshold_counts,
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
    parser.add_argument("--latent-value-action-head", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="optional action-correction checkpoint to initialize from",
    )
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
    parser.add_argument("--marker-action-return", action="store_true")
    parser.add_argument("--marker-action-require-local-evidence", action="store_true")
    parser.add_argument("--turn-oscillation-breaker", action="store_true")
    parser.add_argument(
        "--example-filter",
        choices=(
            "all",
            "current_marker_seen",
            "marker_return",
            "latent_marker_seen",
            "pre_latent_marker",
            "correction_needed",
            "pre_latent_marker_correction_needed",
        ),
        default="all",
        help=(
            "which closed-loop rollout states to train on; the rollout itself "
            "always follows the same planned controller"
        ),
    )
    parser.add_argument(
        "--teacher-source",
        choices=("local_oracle", "odom_frontier", "odom_lookahead"),
        default="local_oracle",
        help=(
            "target action source for correction labels; odom teachers are used "
            "only for training labels and are not embedded in the correction head"
        ),
    )
    parser.add_argument("--odom-lookahead-horizon", type=int, default=9)
    parser.add_argument("--odom-lookahead-beam-width", type=int, default=32)
    parser.add_argument(
        "--trace-correction-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "closed-loop trace JSON to replay into correction examples; "
            "may be passed multiple times"
        ),
    )
    parser.add_argument(
        "--trace-correction-source-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional JSONL source rows paired one-for-one with "
            "--trace-correction-data; use this when replaying traces from "
            "multiple generated splits"
        ),
    )
    parser.add_argument(
        "--trace-correction-source",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument(
        "--trace-correction-teacher-source",
        choices=("local_oracle", "odom_frontier", "odom_lookahead", "trace_selected"),
        default=None,
        help=(
            "target action source for --trace-correction-data; defaults to "
            "--teacher-source"
        ),
    )
    parser.add_argument(
        "--trace-correction-planned-source",
        choices=("trace_selected", "trace_pre_correction", "value_map_planner"),
        default="trace_selected",
        help=(
            "planned action source for trace examples; trace_pre_correction "
            "uses candidate fields logged before action correction, and "
            "value_map_planner recomputes the current learned planner action "
            "on each trace state"
        ),
    )
    parser.add_argument(
        "--trace-correction-episode-index",
        type=int,
        action="append",
        default=[],
        help=(
            "optional zero-based trace episode index to include; may be passed "
            "multiple times"
        ),
    )
    parser.add_argument("--trace-correction-failed-only", action="store_true")
    parser.add_argument("--trace-correction-pre-latent-marker-only", action="store_true")
    parser.add_argument("--trace-correction-post-latent-marker-only", action="store_true")
    parser.add_argument("--trace-correction-current-marker-only", action="store_true")
    parser.add_argument("--trace-correction-marker-return-only", action="store_true")
    parser.add_argument("--trace-correction-needed-only", action="store_true")
    parser.add_argument("--trace-correction-repeat", type=int, default=1)
    parser.add_argument(
        "--include-local-evidence-channels",
        action="store_true",
        help=(
            "concatenate centered learned local evidence channels to recurrent "
            "memory before the correction head"
        ),
    )
    parser.add_argument(
        "--trace-correction-only",
        action="store_true",
        help=(
            "train only on --trace-correction-data examples; the normal "
            "validation examples are still built for checkpoint selection"
        ),
    )
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--correction-threshold", type=float, default=0.0)
    parser.add_argument("--mismatch-weight", type=float, default=4.0)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--sparse-target-top-k", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.horizon < 1:
        raise SystemExit("--horizon must be positive")
    if args.odom_lookahead_horizon < 1:
        raise SystemExit("--odom-lookahead-horizon must be positive")
    if args.odom_lookahead_beam_width < 1:
        raise SystemExit("--odom-lookahead-beam-width must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.correction_threshold <= 1.0:
        raise SystemExit("--correction-threshold must be in [0, 1]")
    if args.mismatch_weight < 0.0:
        raise SystemExit("--mismatch-weight must be non-negative")
    if args.regret_weight < 0.0:
        raise SystemExit("--regret-weight must be non-negative")
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
    if (
        args.max_candidates_per_state is not None
        and args.max_candidates_per_state < 1
    ):
        raise SystemExit("--max-candidates-per-state must be positive")
    if args.trace_correction_repeat < 1:
        raise SystemExit("--trace-correction-repeat must be positive")
    if any(index < 0 for index in args.trace_correction_episode_index):
        raise SystemExit("--trace-correction-episode-index must be non-negative")
    if args.trace_correction_source_data and (
        len(args.trace_correction_source_data) != len(args.trace_correction_data)
    ):
        raise SystemExit(
            "--trace-correction-source-data must be passed once for each "
            "--trace-correction-data path"
        )
    if (
        args.trace_correction_pre_latent_marker_only
        and args.trace_correction_post_latent_marker_only
    ):
        raise SystemExit(
            "--trace-correction-pre-latent-marker-only and "
            "--trace-correction-post-latent-marker-only are mutually exclusive"
        )
    if args.marker_action_return and args.latent_value_action_head is None:
        raise SystemExit("--marker-action-return requires --latent-value-action-head")
    if args.trace_correction_only and not args.trace_correction_data:
        raise SystemExit("--trace-correction-only requires --trace-correction-data")
    trace_correction_teacher_source = (
        args.trace_correction_teacher_source or args.teacher_source
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
    value_map_planner_head, planner_report = _load_value_map_planner_head(
        args.latent_value_map_planner_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    value_action_head = None
    value_action_report = None
    if args.latent_value_action_head is not None:
        value_action_head, value_action_report = _load_value_action_head(
            args.latent_value_action_head,
            fallback_memory_size=args.memory_size,
            device=device,
        )
    for name, size in (
        ("latent memory updater", latent_memory_updater.memory_size),
        ("value field head", value_field_head.memory_size),
        ("value extractor head", extractor_head.memory_size),
        ("value-map planner head", value_map_planner_head.memory_size),
    ):
        if int(size) != int(args.memory_size):
            raise SystemExit(f"--memory-size must match {name} size ({size})")
    if (
        value_action_head is not None
        and int(value_action_head.memory_size) != int(args.memory_size)
    ):
        raise SystemExit(
            "--memory-size must match value action head size "
            f"({value_action_head.memory_size})"
        )

    train_examples = None
    if not args.trace_correction_only:
        train_examples = _build_correction_examples(
            train_rows,
            scene_seed=train_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            horizon=args.horizon,
            max_episodes=args.max_train_episodes,
            max_steps=args.max_steps,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            value_field_head=value_field_head,
            extractor_head=extractor_head,
            value_map_planner_head=value_map_planner_head,
            value_action_head=value_action_head,
            marker_action_return=bool(args.marker_action_return),
            marker_action_require_local_evidence=bool(
                args.marker_action_require_local_evidence
            ),
            turn_oscillation_breaker=bool(args.turn_oscillation_breaker),
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            latent_map_marker_threshold=args.latent_map_marker_threshold,
            target_threshold=args.target_threshold,
            target_top_k=args.target_top_k,
            extractor_threshold=args.extractor_threshold,
            sparse_target_top_k=args.sparse_target_top_k,
            regret_weight=args.regret_weight,
            mismatch_weight=args.mismatch_weight,
            max_candidates_per_state=args.max_candidates_per_state,
            example_filter=args.example_filter,
            teacher_source=args.teacher_source,
            odom_lookahead_horizon=args.odom_lookahead_horizon,
            odom_lookahead_beam_width=args.odom_lookahead_beam_width,
            include_local_evidence_channels=bool(
                args.include_local_evidence_channels
            ),
            device=device,
        )
    trace_correction_count = 0
    if args.trace_correction_data:
        if args.trace_correction_source_data:
            trace_examples = None
            for trace_path, source_path in zip(
                args.trace_correction_data,
                args.trace_correction_source_data,
                strict=True,
            ):
                trace_seed = _infer_scene_seed(source_path)
                if trace_seed is None:
                    raise SystemExit(
                        "could not infer trace-correction source scene seed "
                        f"from {source_path}"
                    )
                paired_examples = _build_trace_correction_examples(
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
                    target_threshold=args.target_threshold,
                    target_top_k=args.target_top_k,
                    extractor_threshold=args.extractor_threshold,
                    sparse_target_top_k=args.sparse_target_top_k,
                    mismatch_weight=args.mismatch_weight,
                    regret_weight=args.regret_weight,
                    teacher_source=trace_correction_teacher_source,
                    odom_lookahead_horizon=args.odom_lookahead_horizon,
                    odom_lookahead_beam_width=args.odom_lookahead_beam_width,
                    planned_source=args.trace_correction_planned_source,
                    value_map_planner_head=value_map_planner_head,
                    value_action_head=value_action_head,
                    marker_action_return=bool(args.marker_action_return),
                    marker_action_require_local_evidence=bool(
                        args.marker_action_require_local_evidence
                    ),
                    episode_indices=tuple(args.trace_correction_episode_index),
                    failed_only=bool(args.trace_correction_failed_only),
                    pre_latent_marker_only=bool(
                        args.trace_correction_pre_latent_marker_only
                    ),
                    post_latent_marker_only=bool(
                        args.trace_correction_post_latent_marker_only
                    ),
                    current_marker_only=bool(args.trace_correction_current_marker_only),
                    marker_return_only=bool(args.trace_correction_marker_return_only),
                    correction_needed_only=bool(args.trace_correction_needed_only),
                    repeat=int(args.trace_correction_repeat),
                    include_local_evidence_channels=bool(
                        args.include_local_evidence_channels
                    ),
                    device=device,
                )
                trace_examples = (
                    paired_examples
                    if trace_examples is None
                    else _concat_examples(trace_examples, paired_examples)
                )
            assert trace_examples is not None
        else:
            trace_rows = (
                train_rows if args.trace_correction_source == "train" else validation_rows
            )
            trace_seed = (
                train_seed if args.trace_correction_source == "train" else validation_seed
            )
            trace_examples = _build_trace_correction_examples(
                list(args.trace_correction_data),
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
                target_threshold=args.target_threshold,
                target_top_k=args.target_top_k,
                extractor_threshold=args.extractor_threshold,
                sparse_target_top_k=args.sparse_target_top_k,
                mismatch_weight=args.mismatch_weight,
                regret_weight=args.regret_weight,
                teacher_source=trace_correction_teacher_source,
                odom_lookahead_horizon=args.odom_lookahead_horizon,
                odom_lookahead_beam_width=args.odom_lookahead_beam_width,
                planned_source=args.trace_correction_planned_source,
                value_map_planner_head=value_map_planner_head,
                value_action_head=value_action_head,
                marker_action_return=bool(args.marker_action_return),
                marker_action_require_local_evidence=bool(
                    args.marker_action_require_local_evidence
                ),
                episode_indices=tuple(args.trace_correction_episode_index),
                failed_only=bool(args.trace_correction_failed_only),
                pre_latent_marker_only=bool(
                    args.trace_correction_pre_latent_marker_only
                ),
                post_latent_marker_only=bool(
                    args.trace_correction_post_latent_marker_only
                ),
                current_marker_only=bool(args.trace_correction_current_marker_only),
                marker_return_only=bool(args.trace_correction_marker_return_only),
                correction_needed_only=bool(args.trace_correction_needed_only),
                repeat=int(args.trace_correction_repeat),
                include_local_evidence_channels=bool(
                    args.include_local_evidence_channels
                ),
                device=device,
            )
        trace_correction_count = int(len(trace_examples.actions))
        if train_examples is None:
            train_examples = trace_examples
        else:
            train_examples = _concat_examples(train_examples, trace_examples)
    if train_examples is None:
        raise SystemExit("no training examples were generated")
    validation_examples = _build_correction_examples(
        validation_rows,
        scene_seed=validation_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        horizon=args.horizon,
        max_episodes=args.max_validation_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        value_field_head=value_field_head,
        extractor_head=extractor_head,
        value_map_planner_head=value_map_planner_head,
        value_action_head=value_action_head,
        marker_action_return=bool(args.marker_action_return),
        marker_action_require_local_evidence=bool(
            args.marker_action_require_local_evidence
        ),
        turn_oscillation_breaker=bool(args.turn_oscillation_breaker),
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        latent_map_marker_threshold=args.latent_map_marker_threshold,
        target_threshold=args.target_threshold,
        target_top_k=args.target_top_k,
        extractor_threshold=args.extractor_threshold,
        sparse_target_top_k=args.sparse_target_top_k,
        regret_weight=args.regret_weight,
        mismatch_weight=args.mismatch_weight,
        max_candidates_per_state=args.max_candidates_per_state,
        example_filter=args.example_filter,
        teacher_source=args.teacher_source,
        odom_lookahead_horizon=args.odom_lookahead_horizon,
        odom_lookahead_beam_width=args.odom_lookahead_beam_width,
        include_local_evidence_channels=bool(args.include_local_evidence_channels),
        device=device,
    )

    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.target_fields,
        train_examples.sparse_probabilities,
        train_examples.planned_actions,
        train_examples.marker_seen,
        train_examples.actions,
        train_examples.weights,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    correction_head = Phase3AActionCorrectionHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=(6 if args.include_local_evidence_channels else 3),
        action_dim=len(ACTION_NAMES),
    ).to(device)
    init_report = None
    if args.init_checkpoint is not None:
        try:
            init_checkpoint = torch.load(
                args.init_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        init_report = init_checkpoint.get("report", {})
        init_config = init_report.get("model_config", {})
        init_memory_size = int(init_config.get("memory_size", args.memory_size))
        init_hidden_dim = int(init_config.get("hidden_dim", args.hidden_dim))
        init_memory_channels = int(
            init_config.get("memory_channels", correction_head.memory_channels)
        )
        if init_memory_size != int(args.memory_size):
            raise SystemExit(
                "--init-checkpoint memory size does not match --memory-size "
                f"({init_memory_size} != {args.memory_size})"
            )
        if init_hidden_dim != int(args.hidden_dim):
            raise SystemExit(
                "--init-checkpoint hidden dim does not match --hidden-dim "
                f"({init_hidden_dim} != {args.hidden_dim})"
            )
        if init_memory_channels != int(correction_head.memory_channels):
            raise SystemExit(
                "--init-checkpoint memory channels do not match "
                f"({init_memory_channels} != {correction_head.memory_channels})"
            )
        correction_head.load_state_dict(init_checkpoint["correction_head_state_dict"])
    optimizer = torch.optim.AdamW(
        correction_head.parameters(),
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
            memory, target, sparse_prob, planned_action, marker_seen, action, weight = (
                next(iterator)
            )
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, sparse_prob, planned_action, marker_seen, action, weight = (
                next(iterator)
            )
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        planned_action = planned_action.to(device)
        marker_seen = marker_seen.to(device)
        action = action.to(device)
        weight = weight.to(device)
        logits = correction_head(
            memory,
            target,
            sparse_prob,
            planned_action,
            marker_seen,
        )
        loss_items = F.cross_entropy(
            logits,
            action,
            reduction="none",
            label_smoothing=args.label_smoothing,
        )
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                correction_head,
                validation_examples,
                threshold=args.correction_threshold,
                batch_size=args.batch_size,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["threshold_action_match"]),
                float(metrics["correction_threshold_match"]),
                float(metrics["keep_threshold_match"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in correction_head.state_dict().items()
                }

    final_validation = _evaluate(
        correction_head,
        validation_examples,
        threshold=args.correction_threshold,
        batch_size=args.batch_size,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        correction_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation

    report = {
        "schema": "jepa_phase3a_action_correction_training_report_v0",
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
        "init_checkpoint": (
            str(args.init_checkpoint.resolve()) if args.init_checkpoint else None
        ),
        "init_checkpoint_completed_steps": (
            init_report.get("completed_steps") if init_report else None
        ),
        "init_checkpoint_selected_step": (
            init_report.get("selected_step") if init_report else None
        ),
        "latent_value_action_head": (
            str(args.latent_value_action_head.resolve())
            if args.latent_value_action_head
            else None
        ),
        "latent_value_action_completed_steps": (
            value_action_report.get("completed_steps")
            if value_action_report
            else None
        ),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "validation_examples": int(len(validation_examples.actions)),
        "example_filter": args.example_filter,
        "teacher_source": args.teacher_source,
        "trace_correction_data": [
            str(path.resolve()) for path in args.trace_correction_data
        ],
        "trace_correction_source_data": [
            str(path.resolve()) for path in args.trace_correction_source_data
        ],
        "trace_correction_source": str(args.trace_correction_source),
        "trace_correction_teacher_source": str(trace_correction_teacher_source),
        "trace_correction_planned_source": str(args.trace_correction_planned_source),
        "trace_correction_episode_indices": [
            int(index) for index in args.trace_correction_episode_index
        ],
        "trace_correction_failed_only": bool(args.trace_correction_failed_only),
        "trace_correction_pre_latent_marker_only": bool(
            args.trace_correction_pre_latent_marker_only
        ),
        "trace_correction_post_latent_marker_only": bool(
            args.trace_correction_post_latent_marker_only
        ),
        "trace_correction_current_marker_only": bool(
            args.trace_correction_current_marker_only
        ),
        "trace_correction_marker_return_only": bool(
            args.trace_correction_marker_return_only
        ),
        "trace_correction_needed_only": bool(args.trace_correction_needed_only),
        "trace_correction_repeat": int(args.trace_correction_repeat),
        "trace_correction_examples": int(trace_correction_count),
        "include_local_evidence_channels": bool(
            args.include_local_evidence_channels
        ),
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
            "memory_channels": int(correction_head.memory_channels),
            "action_dim": len(ACTION_NAMES),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "correction_head_state_dict": correction_head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
