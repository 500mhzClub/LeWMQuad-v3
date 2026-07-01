#!/usr/bin/env python3
"""Export a closed-loop Phase 3A novelty-then-claim rollout demo."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    ACTION_INDEX,
    ACTION_NAMES,
    GridScene,
    GridState,
    YAW_TO_VEC,
    _explore_then_claim_labels,
    _goal_distances,
    _goal_marker_visible,
    _state_dict,
    _utility,
    _visible_cells,
    action_vector,
    generate_scene,
    read_jsonl,
    render_observation,
    step_state,
)
from lewm.benchmarks.phase3a_explore_claim import (  # noqa: E402
    egocentric_explore_claim_score,
)
from lewm.benchmarks.phase3a_marker_memory import (  # noqa: E402
    egocentric_marker_memory_delta,
    egocentric_marker_memory_score,
    marker_position_in_observation,
    remembered_marker_position,
)
from lewm.benchmarks.phase3a_training import (  # noqa: E402
    Phase3AMaterializedDataset,
    source_key,
)
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AActionCorrectionHead,
    Phase3AEgocentricMemoryUpdate,
    Phase3AEgocentricMemoryPolicy,
    Phase3AEgocentricValueFieldHead,
    Phase3AValueFieldActionHead,
    Phase3AValueFieldExtractorHead,
    Phase3AValueMapPlannerHead,
    Phase3AValueMapRouterHead,
    Phase3ALatentMapHead,
)
from lewm.models.phase3b_reachability import (  # noqa: E402
    Phase3BReachabilityConditionedValueMapPlannerHead,
    Phase3BReachabilityHead,
    reachability_feature_tensor,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _state_from_dict(value: dict) -> GridState:
    return GridState(x=int(value["x"]), y=int(value["y"]), yaw=int(value["yaw"]))


def _rgb_image(observation: list, *, size: int) -> Image.Image:
    from PIL import Image

    tensor = np.asarray(observation, dtype=np.float32)
    rgb = np.clip(np.moveaxis(tensor, 0, -1) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _goal_scene_from_row(row: dict, *, seed: int, width: int, height: int) -> GridScene:
    base_scene_id = str(row["base_scene_id"])
    scene_index = int(base_scene_id.rsplit("_", 1)[-1])
    base_scene = generate_scene(
        split=str(row["split"]),
        scene_index=scene_index,
        seed=seed,
        width=width,
        height=height,
    )
    goal = (int(row["goal"]["x"]), int(row["goal"]["y"]))
    palette = {
        key: tuple(float(channel) for channel in value)
        for key, value in row.get("render_palette", {}).items()
    } or base_scene.render_palette
    return GridScene(
        scene_id=str(row["scene_id"]),
        family=base_scene.family,
        grid=base_scene.grid,
        goal=goal,
        distances=_goal_distances(base_scene.grid, goal),
        render_palette=palette,
    )


def _infer_scene_seed(data_path: Path) -> int | None:
    split = None
    if data_path.name.startswith("validation"):
        split = "validation"
    elif data_path.name.startswith("train"):
        split = "train"
    if split is None:
        return None
    manifest_paths = (
        data_path.parent / "phase3a_positive_control_manifest.json",
        data_path.parent / "phase3a_odom_frontier_distill_manifest.json",
    )
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        direct_seed = manifest.get(f"{split}_seed")
        if direct_seed is not None:
            return int(direct_seed)
        audit = manifest.get(f"{split}_audit", {})
        audit_seed = audit.get("seed")
        if audit_seed is not None:
            return int(audit_seed)
    return None


def _candidate_rows(
    *,
    scene: GridScene,
    source_index: int,
    state: GridState,
    history_states: list[GridState],
    history_actions: list[str],
    horizon: int,
    view_size: int,
    current_goal_marker: bool,
) -> list[dict]:
    history_observations = [
        render_observation(
            scene,
            history_state,
            view_size=view_size,
            include_goal_beacon=False,
            show_goal_marker=True,
        )
        for history_state in history_states
    ]
    history_goal_seen = any(
        _goal_marker_visible(scene, item, view_size=view_size)
        for item in history_states
    )
    current_goal_seen = current_goal_marker and _goal_marker_visible(
        scene,
        state,
        view_size=view_size,
    )
    observed_free_cells = set()
    for history_state in history_states:
        observed_free_cells.update(
            _visible_cells(scene, history_state, view_size=view_size, free_only=True)
        )
    observed_free_cells.update(
        _visible_cells(scene, state, view_size=view_size, free_only=True)
    )
    start_observation = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    rows = []
    for candidate_index, sequence in enumerate(product(ACTION_NAMES, repeat=horizon)):
        future_state = state
        collisions = 0
        future_observations = []
        future_states = []
        future_goal_seen = False
        candidate_observed = set(observed_free_cells)
        new_free_cells_total = 0
        for action in sequence:
            previous_distance = scene.distance_to_goal(future_state.x, future_state.y)
            future_state, collided = step_state(scene, future_state, action)
            next_distance = scene.distance_to_goal(future_state.x, future_state.y)
            step_progress = (
                0.0
                if previous_distance is None or next_distance is None
                else float(previous_distance - next_distance)
            )
            collisions += int(collided)
            future_states.append(_state_dict(future_state))
            visible_free = _visible_cells(
                scene,
                future_state,
                view_size=view_size,
                free_only=True,
            )
            newly_visible = visible_free - candidate_observed
            candidate_observed.update(visible_free)
            new_free_cells_total += len(newly_visible)
            step_goal_seen = _goal_marker_visible(
                scene,
                future_state,
                view_size=view_size,
            )
            future_goal_seen = future_goal_seen or step_goal_seen
            future_observations.append(
                {
                    "observation_rgb": render_observation(
                        scene,
                        future_state,
                        view_size=view_size,
                        include_goal_beacon=False,
                        show_goal_marker=True,
                    ),
                    "observation_valid": True,
                    "collision": collided,
                    "step_progress_cells": step_progress,
                    "goal_distance_cells": next_distance,
                    "goal_marker_visible": step_goal_seen,
                    "newly_observed_free_cells": len(newly_visible),
                    "cumulative_new_free_cells": new_free_cells_total,
                    "goal_claimed": (future_state.x, future_state.y) == scene.goal,
                }
            )
        labels = _explore_then_claim_labels(
            _utility(scene, state, future_state, collisions),
            history_goal_seen=history_goal_seen,
            current_goal_seen=current_goal_seen,
            future_goal_seen=future_goal_seen,
            new_free_cells=new_free_cells_total,
            collisions=collisions,
            discovery_bonus=False,
            reached_bonus=False,
        )
        rows.append(
            {
                "schema": "jepa_phase3a_closed_loop_candidate_v0",
                "split": "closed_loop",
                "scene_id": scene.scene_id,
                "base_scene_id": scene.scene_id,
                "family": scene.family,
                "render_palette": {
                    key: list(value)
                    for key, value in scene.render_palette.items()
                },
                "source_index": source_index,
                "base_source_index": source_index,
                "goal_variant_index": 0,
                "goal_variants_per_source": 1,
                "candidate_index": candidate_index,
                "start_state": _state_dict(state),
                "goal": {"x": scene.goal[0], "y": scene.goal[1]},
                "history_steps": len(history_states),
                "history_states": [_state_dict(item) for item in history_states],
                "history_primitive_sequence": list(history_actions),
                "history_actions": [list(action_vector(action)) for action in history_actions],
                "history_observations_rgb": history_observations,
                "history_goal_beacon": False,
                "current_goal_beacon": False,
                "history_goal_marker": True,
                "current_goal_marker": current_goal_marker,
                "future_goal_marker": True,
                "history_policy": "closed_loop",
                "utility_mode": "novelty_then_claim",
                "history_goal_marker_seen": history_goal_seen,
                "current_goal_marker_seen": current_goal_seen,
                "observed_free_cells_before_candidate": len(observed_free_cells),
                "start_observation_rgb": start_observation,
                "primitive_sequence": list(sequence),
                "active_blocks": [list(action_vector(action)) for action in sequence],
                "future_states": future_states,
                "future_observations": future_observations,
                "complete_valid_future_sequence": True,
                "consequence_labels": labels,
            }
        )
    return rows


def _subsample_candidates_by_first(
    rows: list[dict],
    max_candidates_per_state: int | None,
) -> list[dict]:
    if max_candidates_per_state is None:
        return rows
    by_first: dict[str, list[dict]] = {}
    for row in rows:
        first = str(row["primitive_sequence"][0])
        by_first.setdefault(first, []).append(row)
    actions = sorted(by_first)
    per_action = max(1, max_candidates_per_state // max(len(actions), 1))
    remaining = max_candidates_per_state - per_action * len(actions)
    subsampled: list[dict] = []
    for action in actions:
        cap = per_action + int(remaining > 0)
        remaining -= int(remaining > 0)
        subsampled.extend(by_first[action][:cap])
    return subsampled


@torch.no_grad()
def _select_action(
    model: torch.nn.Module,
    rows: list[dict],
    *,
    score_source: str,
    device: torch.device,
) -> tuple[dict, dict]:
    if score_source == "egocentric_explore_claim_score":
        scores = torch.tensor(
            [egocentric_explore_claim_score(row) for row in rows],
            dtype=torch.float32,
        )
        return _select_action_from_scores(rows, scores)

    batch = Phase3AMaterializedDataset(rows).materialize_batch(range(len(rows))).to(device)
    if score_source == "palette_online_frontier_marker_score":
        if remembered_marker_position(rows[0]) is not None:
            scores = torch.tensor(
                [egocentric_marker_memory_score(row) for row in rows],
                dtype=torch.float32,
            )
        else:
            scores = _exact_online_frontier_scores(model, batch).detach().cpu()
        return _select_action_from_scores(rows, scores)

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
        structured_marker_memory_start_delta_targets=batch.marker_memory_start_delta_targets,
        categorical_marker_memory_valid_mask=batch.marker_memory_start_cell_valid_mask,
        categorical_marker_memory_cell_targets=batch.marker_memory_start_cell_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )
    if score_source == "egocentric_marker_memory_score":
        scores = torch.tensor(
            [egocentric_marker_memory_score(row) for row in rows],
            dtype=torch.float32,
        )
    elif score_source == "egocentric_marker_bounded_frontier_score":
        marker_valid, _ahead, _lateral = egocentric_marker_memory_delta(rows[0])
        if marker_valid:
            scores = torch.tensor(
                [egocentric_marker_memory_score(row) for row in rows],
                dtype=torch.float32,
            )
        else:
            scores = output["spatial_frontier_memory_score_prediction"].detach().cpu()
    elif score_source == "online_frontier_marker_score":
        scores = output["online_frontier_marker_score_prediction"].detach().cpu()
    else:
        key = (
            "spatial_frontier_memory_score_prediction"
            if score_source == "spatial_frontier_memory_score"
            else "candidate_score_prediction"
        )
        scores = output[key].detach().cpu()
    return _select_action_from_scores(rows, scores)


@torch.no_grad()
def _exact_online_frontier_scores(
    model: torch.nn.Module,
    batch,
) -> torch.Tensor:
    observed, blocked = model.online_frontier_start_maps(
        batch.history_vision,
        batch.history_actions,
        batch.vision[:, 0],
        batch_size=batch.actions.shape[0],
        device=batch.actions.device,
        dtype=batch.actions.dtype,
    )
    scores = batch.actions.new_zeros(batch.actions.shape[0])
    footprint = model.current_view_footprint_like(observed)
    center = observed.shape[-1] // 2
    for step in range(batch.actions.shape[1]):
        ahead_blocked = blocked[:, center - 1, center].clamp(0.0, 1.0)
        scores = (
            scores
            - model.spatial_frontier_collision_penalty
            * batch.actions[:, step, 0]
            * ahead_blocked
        )
        observed, blocked = model.roll_online_frontier_maps(
            observed,
            blocked,
            batch.actions[:, step : step + 1],
        )
        novel = (footprint - observed).clamp(0.0, 1.0)
        scores = (
            scores
            + model.spatial_frontier_novelty_reward
            * novel.flatten(start_dim=1).sum(dim=-1)
        )
        observed = torch.maximum(observed, footprint)
    return scores


def _select_action_from_scores(
    rows: list[dict],
    scores: torch.Tensor,
) -> tuple[dict, dict]:
    selected_index = int(scores.argmax().item())
    oracle_index = max(
        range(len(rows)),
        key=lambda index: float(rows[index]["consequence_labels"]["target_utility"]),
    )
    selected = rows[selected_index]
    oracle = rows[oracle_index]
    return (
        {
            "row": selected,
            "score": float(scores[selected_index]),
            "action": str(selected["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in selected["primitive_sequence"]),
            "utility": float(selected["consequence_labels"]["target_utility"]),
        },
        {
            "row": oracle,
            "action": str(oracle["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in oracle["primitive_sequence"]),
            "utility": float(oracle["consequence_labels"]["target_utility"]),
        },
    )


def _observation_tensor(observation: list, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def _action_tensor(action: str, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[list(action_vector(action))]],
        dtype=torch.float32,
        device=device,
    )


def _world_from_ego(state: GridState, ahead: int, lateral: int) -> tuple[int, int]:
    forward = YAW_TO_VEC[state.yaw]
    left = YAW_TO_VEC[(state.yaw - 1) % 4]
    return (
        state.x + forward[0] * ahead + left[0] * lateral,
        state.y + forward[1] * ahead + left[1] * lateral,
    )


def _color_distance_sq(
    color: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return sum((color[index] - target[index]) ** 2 for index in range(3))


def _is_blocked_observation_color(
    color: tuple[float, float, float],
    palette: dict[str, tuple[float, float, float]] | None,
) -> bool:
    if palette is not None:
        wall = palette.get("wall")
        outside = palette.get("outside")
        if wall is not None and _color_distance_sq(color, wall) <= 1e-8:
            return True
        if outside is not None and _color_distance_sq(color, outside) <= 1e-8:
            return True
        return False
    return max(color) < 0.25


def _update_odom_frontier_memory(
    memory: dict,
    *,
    scene: GridScene,
    state: GridState,
    view_size: int,
    current_goal_marker: bool,
) -> None:
    observation = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    marker_color = None
    if scene.render_palette is not None:
        marker_color = scene.render_palette.get("goal")
    marker = marker_position_in_observation(
        observation,
        marker_color=marker_color,
    )
    if marker is not None:
        memory["marker"] = _world_from_ego(state, marker[0], marker[1])

    red, green, blue = observation
    radius = view_size // 2
    for row in range(view_size):
        for col in range(view_size):
            ahead = radius - row
            lateral = col - radius
            cell = _world_from_ego(state, ahead, lateral)
            color = (
                float(red[row][col]),
                float(green[row][col]),
                float(blue[row][col]),
            )
            if _is_blocked_observation_color(color, scene.render_palette):
                memory["blocked"].add(cell)
                memory["free"].discard(cell)
            else:
                memory["free"].add(cell)
                memory["blocked"].discard(cell)


@torch.no_grad()
def _update_latent_odom_frontier_memory(
    memory: dict,
    *,
    model: torch.nn.Module,
    latent_map_head: Phase3ALatentMapHead,
    scene: GridScene,
    state: GridState,
    view_size: int,
    current_goal_marker: bool,
    blocked_threshold: float,
    marker_threshold: float,
    device: torch.device,
) -> None:
    observation = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    vision = _observation_tensor(observation, device=device)
    tokens = model.encoder(vision)
    logits = latent_map_head(tokens)
    probs = logits.sigmoid()[0].detach().cpu()
    blocked_probs = probs[0]
    free_probs = probs[1]
    marker_probs = probs[2]
    marker_index = int(marker_probs.flatten().argmax().item())
    marker_score = float(marker_probs.flatten()[marker_index])
    radius = view_size // 2
    if marker_score >= marker_threshold:
        marker_row = marker_index // view_size
        marker_col = marker_index % view_size
        ahead = radius - marker_row
        lateral = marker_col - radius
        memory["marker"] = _world_from_ego(state, ahead, lateral)
    for row in range(view_size):
        for col in range(view_size):
            ahead = radius - row
            lateral = col - radius
            cell = _world_from_ego(state, ahead, lateral)
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


@torch.no_grad()
def _update_latent_egocentric_frontier_memory(
    memory: dict,
    *,
    model: torch.nn.Module,
    latent_map_head: Phase3ALatentMapHead,
    scene: GridScene,
    state: GridState,
    view_size: int,
    current_goal_marker: bool,
    blocked_threshold: float,
    marker_threshold: float,
    device: torch.device,
) -> None:
    observation = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    vision = _observation_tensor(observation, device=device)
    tokens = model.encoder(vision)
    logits = latent_map_head(tokens)
    probs = logits.sigmoid()[0].detach().cpu()
    blocked_probs = probs[0]
    free_probs = probs[1]
    marker_probs = probs[2]
    marker_index = int(marker_probs.flatten().argmax().item())
    marker_score = float(marker_probs.flatten()[marker_index])
    radius = view_size // 2
    if marker_score >= marker_threshold:
        marker_row = marker_index // view_size
        marker_col = marker_index % view_size
        memory["marker"] = (
            radius - marker_row,
            marker_col - radius,
        )
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
    _clip_egocentric_memory(memory)


def _center_local_evidence(
    local_evidence: torch.Tensor,
    *,
    memory_size: int,
) -> torch.Tensor:
    view_size = int(local_evidence.shape[-1])
    if view_size > memory_size:
        raise ValueError("local evidence view cannot exceed memory size")
    output = torch.zeros(
        local_evidence.shape[0],
        local_evidence.shape[1],
        memory_size,
        memory_size,
        dtype=local_evidence.dtype,
        device=local_evidence.device,
    )
    start = memory_size // 2 - view_size // 2
    output[:, :, start : start + view_size, start : start + view_size] = local_evidence
    return output


@torch.no_grad()
def _latent_local_evidence(
    *,
    model: torch.nn.Module,
    latent_map_head: Phase3ALatentMapHead,
    scene: GridScene,
    state: GridState,
    view_size: int,
    current_goal_marker: bool,
    device: torch.device,
) -> torch.Tensor:
    observation = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    vision = _observation_tensor(observation, device=device)
    tokens = model.encoder(vision)
    return latent_map_head(tokens).sigmoid()


def _egocentric_memory_tensor_to_dict(
    memory_tensor: torch.Tensor,
    *,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
) -> dict:
    if memory_tensor.ndim != 3:
        raise ValueError(f"expected (C, S, S) memory tensor, got {memory_tensor.shape}")
    channels, memory_size, _width = memory_tensor.shape
    if channels != 3:
        raise ValueError(f"expected 3 memory channels, got {channels}")
    radius = memory_size // 2
    blocked_probs = memory_tensor[0]
    free_probs = memory_tensor[1]
    marker_probs = memory_tensor[2]
    memory = {
        "free": set(),
        "blocked": set(),
        "marker": None,
        "radius": radius,
    }
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
        marker_row = marker_index // memory_size
        marker_col = marker_index % memory_size
        memory["marker"] = (
            radius - marker_row,
            marker_col - radius,
        )
    return memory


def _action_index_tensor(action: str, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([ACTION_NAMES.index(action)], dtype=torch.long, device=device)


def _action_correction_memory_input(
    correction_head: Phase3AActionCorrectionHead,
    recurrent_memory_tensor: torch.Tensor,
    local_evidence: torch.Tensor,
) -> torch.Tensor:
    recurrent_channels = int(recurrent_memory_tensor.shape[1])
    if int(correction_head.memory_channels) == recurrent_channels:
        return recurrent_memory_tensor
    if int(correction_head.memory_channels) == recurrent_channels + int(
        local_evidence.shape[1]
    ):
        return torch.cat([recurrent_memory_tensor, local_evidence], dim=1)
    raise ValueError(
        "action correction head expects incompatible memory channels "
        f"({correction_head.memory_channels}) for recurrent "
        f"{recurrent_channels} and local evidence {int(local_evidence.shape[1])}"
    )


def _value_action_memory_input(
    action_head: Phase3AValueFieldActionHead,
    recurrent_memory_tensor: torch.Tensor,
    local_evidence: torch.Tensor,
) -> torch.Tensor:
    recurrent_channels = int(recurrent_memory_tensor.shape[1])
    if int(action_head.memory_channels) == recurrent_channels:
        return recurrent_memory_tensor
    if int(action_head.memory_channels) == recurrent_channels + int(
        local_evidence.shape[1]
    ):
        return torch.cat([recurrent_memory_tensor, local_evidence], dim=1)
    raise ValueError(
        "value action head expects incompatible memory channels "
        f"({action_head.memory_channels}) for recurrent {recurrent_channels} "
        f"and local evidence {int(local_evidence.shape[1])}"
    )


@torch.no_grad()
def _apply_action_correction(
    correction_head: Phase3AActionCorrectionHead,
    planned_action: str,
    recurrent_memory_tensor: torch.Tensor,
    target_probs_gpu: torch.Tensor,
    sparse_prob: torch.Tensor,
    *,
    latent_marker_seen_ever: bool,
    threshold: float,
) -> tuple[str, float]:
    planned_index = _action_index_tensor(planned_action, device=recurrent_memory_tensor.device)
    marker_seen = torch.tensor(
        [float(latent_marker_seen_ever)],
        dtype=torch.float32,
        device=recurrent_memory_tensor.device,
    )
    logits = correction_head(
        recurrent_memory_tensor,
        target_probs_gpu,
        sparse_prob,
        planned_index,
        marker_seen,
    )
    probabilities = logits.softmax(dim=1)
    confidence, action_index = probabilities.max(dim=1)
    if float(confidence.item()) < threshold:
        return planned_action, float(confidence.item())
    return ACTION_NAMES[int(action_index.item())], float(confidence.item())


def _load_action_correction_head(
    path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[Phase3AActionCorrectionHead, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    head = Phase3AActionCorrectionHead(
        memory_size=int(config.get("memory_size", model.spatial_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        memory_channels=int(config.get("memory_channels", 3)),
        action_dim=int(config.get("action_dim", len(ACTION_NAMES))),
    ).to(device)
    head.load_state_dict(checkpoint["correction_head_state_dict"])
    head.eval()
    return head, report


def _roll_egocentric_cell(
    cell: tuple[int, int],
    action: str,
    *,
    collision: bool,
) -> tuple[int, int]:
    ahead, lateral = cell
    if action == "forward":
        if collision:
            return cell
        return ahead - 1, lateral
    if action == "turn_left":
        return lateral, -ahead
    if action == "turn_right":
        return -lateral, ahead
    return cell


def _clip_egocentric_memory(memory: dict) -> None:
    radius = int(memory.get("radius", 0))
    if radius <= 0:
        return
    memory["free"] = {
        cell
        for cell in memory["free"]
        if abs(cell[0]) <= radius and abs(cell[1]) <= radius
    }
    memory["blocked"] = {
        cell
        for cell in memory["blocked"]
        if abs(cell[0]) <= radius and abs(cell[1]) <= radius
    }
    marker = memory.get("marker")
    if marker is not None and (
        abs(marker[0]) > radius or abs(marker[1]) > radius
    ):
        memory["marker"] = None


def _roll_egocentric_frontier_memory(
    memory: dict,
    action: str,
    *,
    collision: bool,
) -> None:
    memory["free"] = {
        _roll_egocentric_cell(cell, action, collision=collision)
        for cell in memory["free"]
    }
    memory["blocked"] = {
        _roll_egocentric_cell(cell, action, collision=collision)
        for cell in memory["blocked"]
    }
    marker = memory.get("marker")
    if marker is not None:
        memory["marker"] = _roll_egocentric_cell(
            marker,
            action,
            collision=collision,
        )
    if action == "forward" and collision:
        memory["blocked"].add((1, 0))
        memory["free"].discard((1, 0))
    memory["free"].add((0, 0))
    memory["blocked"].discard((0, 0))
    _clip_egocentric_memory(memory)


def _neighbor_cells(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in YAW_TO_VEC]


def _is_frontier_cell(memory: dict, cell: tuple[int, int]) -> bool:
    if cell not in memory["free"]:
        return False
    return any(
        neighbor not in memory["free"] and neighbor not in memory["blocked"]
        for neighbor in _neighbor_cells(cell)
    )


def _bfs_first_step(
    memory: dict,
    *,
    start: tuple[int, int],
    target: tuple[int, int] | None = None,
    frontier: bool = False,
) -> tuple[int, int] | None:
    free = memory["free"]
    if start not in free:
        free.add(start)
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found = None
    while queue:
        cell = queue.popleft()
        if target is not None and cell == target:
            found = cell
            break
        if frontier and cell != start and _is_frontier_cell(memory, cell):
            found = cell
            break
        for neighbor in _neighbor_cells(cell):
            if neighbor in parent or neighbor not in free:
                continue
            parent[neighbor] = cell
            queue.append(neighbor)
    if found is None:
        return None
    step = found
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    return step if step != start else None


def _bfs_frontier_candidates(
    memory: dict,
    *,
    start: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int] | None, int]]:
    free = memory["free"]
    if start not in free:
        free.add(start)
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distance: dict[tuple[int, int], int] = {start: 0}
    candidates: list[tuple[tuple[int, int], tuple[int, int] | None, int]] = []
    while queue:
        cell = queue.popleft()
        if _is_frontier_cell(memory, cell):
            step = cell
            while parent[step] is not None and parent[step] != start:
                step = parent[step]
            candidates.append(
                (
                    cell,
                    step if step != start else None,
                    distance[cell],
                )
            )
        for neighbor in _neighbor_cells(cell):
            if neighbor in parent or neighbor not in free:
                continue
            parent[neighbor] = cell
            distance[neighbor] = distance[cell] + 1
            queue.append(neighbor)
    return candidates


def _desired_yaw_for_step(
    start: tuple[int, int],
    step: tuple[int, int],
) -> int | None:
    dx = step[0] - start[0]
    dy = step[1] - start[1]
    for yaw, vector in enumerate(YAW_TO_VEC):
        if vector == (dx, dy):
            return yaw
    return None


def _action_toward_yaw(current_yaw: int, desired_yaw: int | None) -> str:
    if desired_yaw is None:
        return "hold"
    delta = (desired_yaw - current_yaw) % 4
    if delta == 0:
        return "forward"
    if delta == 1:
        return "turn_right"
    if delta == 3:
        return "turn_left"
    return "turn_left"


def _turn_cost_to_yaw(current_yaw: int, desired_yaw: int | None) -> int:
    if desired_yaw is None:
        return 0
    delta = (desired_yaw - current_yaw) % 4
    if delta == 0:
        return 0
    if delta in (1, 3):
        return 1
    return 2


def _unknown_neighbor_count(memory: dict, cell: tuple[int, int]) -> int:
    return sum(
        neighbor not in memory["free"] and neighbor not in memory["blocked"]
        for neighbor in _neighbor_cells(cell)
    )


def _unknown_view_gain(
    memory: dict,
    cell: tuple[int, int],
    *,
    radius: int,
) -> int:
    cx, cy = cell
    gain = 0
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            target = (x, y)
            if target not in memory["free"] and target not in memory["blocked"]:
                gain += 1
    return gain


def _frontier_action_at_current(
    memory: dict,
    state: GridState,
    *,
    strategy: str = "nearest",
    gain_radius: int = 3,
) -> str | None:
    current = (state.x, state.y)
    candidates = []
    for yaw, vector in enumerate(YAW_TO_VEC):
        neighbor = (current[0] + vector[0], current[1] + vector[1])
        if neighbor in memory["free"] or neighbor in memory["blocked"]:
            continue
        candidates.append(yaw)
    if not candidates:
        return None
    if strategy == "gain":
        candidates.sort(
            key=lambda yaw: (
                _unknown_view_gain(
                    memory,
                    (
                        current[0] + YAW_TO_VEC[yaw][0],
                        current[1] + YAW_TO_VEC[yaw][1],
                    ),
                    radius=gain_radius,
                ),
                -_turn_cost_to_yaw(state.yaw, yaw),
                yaw == state.yaw,
            ),
            reverse=True,
        )
        return _action_toward_yaw(state.yaw, candidates[0])
    if state.yaw in candidates:
        return "forward"
    left_yaw = (state.yaw - 1) % 4
    right_yaw = (state.yaw + 1) % 4
    if left_yaw in candidates:
        return "turn_left"
    if right_yaw in candidates:
        return "turn_right"
    return "turn_left"


def _select_gain_frontier_action(
    memory: dict,
    state: GridState,
    *,
    gain_radius: int,
    distance_penalty: float,
    turn_penalty: float,
    neighbor_weight: float,
) -> str | None:
    current = (state.x, state.y)
    candidates = _bfs_frontier_candidates(memory, start=current)
    if not candidates:
        return None
    best: tuple[float, int, int, tuple[int, int], tuple[int, int] | None] | None = None
    for cell, first_step, distance in candidates:
        if first_step is None:
            desired_yaw = None
            turn_cost = 0
        else:
            desired_yaw = _desired_yaw_for_step(current, first_step)
            turn_cost = _turn_cost_to_yaw(state.yaw, desired_yaw)
        score = (
            float(_unknown_view_gain(memory, cell, radius=gain_radius))
            + neighbor_weight * float(_unknown_neighbor_count(memory, cell))
            - distance_penalty * float(distance)
            - turn_penalty * float(turn_cost)
        )
        ranked = (score, -distance, -turn_cost, cell, first_step)
        if best is None or ranked > best:
            best = ranked
    assert best is not None
    _score, _distance_key, _turn_key, target_cell, first_step = best
    if first_step is None:
        return _frontier_action_at_current(
            memory,
            state,
            strategy="gain",
            gain_radius=gain_radius,
        )
    return _action_toward_yaw(
        state.yaw,
        _desired_yaw_for_step(current, first_step),
    )


def _copy_odom_frontier_memory(memory: dict) -> dict:
    return {
        "free": set(memory["free"]),
        "blocked": set(memory["blocked"]),
        "marker": memory.get("marker"),
    }


def _odom_memory_path_distance(
    memory: dict,
    start: tuple[int, int],
    target: tuple[int, int] | None,
) -> int | None:
    if target is None:
        return None
    free = memory["free"]
    if start not in free or target not in free:
        return None
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    visited = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == target:
            return distance
        for neighbor in _neighbor_cells(cell):
            if neighbor in visited or neighbor not in free:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _select_odom_frontier_lookahead_action(
    memory: dict,
    *,
    scene: GridScene,
    state: GridState,
    view_size: int,
    horizon: int,
    beam_width: int,
) -> str:
    if horizon <= 0 or beam_width <= 0:
        return _select_odom_frontier_action(memory, state)
    base_known_count = len(memory["free"]) + len(memory["blocked"])
    base_marker = memory.get("marker")
    nodes = [
        (
            0.0,
            None,
            state,
            _copy_odom_frontier_memory(memory),
            {(state.x, state.y, state.yaw)},
        )
    ]
    best = None
    primitive_actions = ("forward", "turn_left", "turn_right")
    for depth in range(horizon):
        expanded = []
        for score, first_action, node_state, node_memory, visited in nodes:
            for action in primitive_actions:
                next_state, collision = step_state(scene, node_state, action)
                next_memory = _copy_odom_frontier_memory(node_memory)
                before_known = len(next_memory["free"]) + len(next_memory["blocked"])
                _update_odom_frontier_memory(
                    next_memory,
                    scene=scene,
                    state=next_state,
                    view_size=view_size,
                    current_goal_marker=True,
                )
                after_known = len(next_memory["free"]) + len(next_memory["blocked"])
                gain = max(0, after_known - before_known)
                next_first_action = action if first_action is None else first_action
                pose = (next_state.x, next_state.y, next_state.yaw)
                repeated_pose = pose in visited
                next_score = (
                    score
                    + float(gain)
                    - (0.3 if action != "forward" else 0.0)
                    - (3.0 if collision else 0.0)
                    - (1.5 if repeated_pose else 0.0)
                )
                if (next_state.x, next_state.y) == scene.goal:
                    next_score += 10000.0 + float(horizon - depth)
                marker = next_memory.get("marker") or base_marker
                if marker is not None:
                    path_distance = _odom_memory_path_distance(
                        next_memory,
                        (next_state.x, next_state.y),
                        marker,
                    )
                    manhattan_distance = (
                        abs(next_state.x - marker[0]) + abs(next_state.y - marker[1])
                    )
                    distance_proxy = (
                        manhattan_distance + 8
                        if path_distance is None
                        else min(path_distance, manhattan_distance + 8)
                    )
                    next_score += (
                        800.0
                        - 8.0 * float(distance_proxy)
                        - 0.5 * float(depth)
                    )
                else:
                    next_score += 0.05 * float(after_known - base_known_count)
                next_visited = set(visited)
                next_visited.add(pose)
                expanded.append(
                    (
                        next_score,
                        next_first_action,
                        next_state,
                        next_memory,
                        next_visited,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0], reverse=True)
        nodes = expanded[:beam_width]
        if best is None or nodes[0][0] > best[0]:
            best = nodes[0]
    if best is None or best[1] is None:
        return _select_odom_frontier_action(memory, state)
    return str(best[1])


def _select_odom_frontier_action(
    memory: dict,
    state: GridState,
    *,
    strategy: str = "nearest",
    gain_radius: int = 3,
    distance_penalty: float = 0.6,
    turn_penalty: float = 0.25,
    neighbor_weight: float = 2.0,
) -> str:
    current = (state.x, state.y)
    memory["free"].add(current)
    marker = memory.get("marker")
    if marker is not None and marker in memory["free"]:
        if marker == current:
            return "hold"
        next_step = _bfs_first_step(memory, start=current, target=marker)
        if next_step is not None:
            return _action_toward_yaw(
                state.yaw,
                _desired_yaw_for_step(current, next_step),
            )
    if strategy == "gain":
        action = _select_gain_frontier_action(
            memory,
            state,
            gain_radius=gain_radius,
            distance_penalty=distance_penalty,
            turn_penalty=turn_penalty,
            neighbor_weight=neighbor_weight,
        )
        if action is not None:
            return action
    if _is_frontier_cell(memory, current):
        action = _frontier_action_at_current(memory, state)
        if action is not None:
            return action
    next_step = _bfs_first_step(memory, start=current, frontier=True)
    if next_step is not None:
        return _action_toward_yaw(
            state.yaw,
            _desired_yaw_for_step(current, next_step),
        )
    return "turn_left"


def _egocentric_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    ahead, lateral = cell
    return [
        (ahead + 1, lateral),
        (ahead, lateral + 1),
        (ahead, lateral - 1),
        (ahead - 1, lateral),
    ]


def _is_egocentric_frontier_cell(memory: dict, cell: tuple[int, int]) -> bool:
    if cell not in memory["free"]:
        return False
    return any(
        neighbor not in memory["free"] and neighbor not in memory["blocked"]
        for neighbor in _egocentric_neighbors(cell)
    )


def _egocentric_bfs_first_step(
    memory: dict,
    *,
    target: tuple[int, int] | None = None,
    frontier: bool = False,
) -> tuple[int, int] | None:
    start = (0, 0)
    memory["free"].add(start)
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found = None
    while queue:
        cell = queue.popleft()
        if target is not None and cell == target:
            found = cell
            break
        if frontier and cell != start and _is_egocentric_frontier_cell(memory, cell):
            found = cell
            break
        for neighbor in _egocentric_neighbors(cell):
            if neighbor in parent or neighbor not in memory["free"]:
                continue
            parent[neighbor] = cell
            queue.append(neighbor)
    if found is None:
        return None
    step = found
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    return step if step != start else None


def _action_toward_egocentric_step(step: tuple[int, int] | None) -> str:
    if step is None:
        return "hold"
    ahead, lateral = step
    if ahead > 0 and abs(ahead) >= abs(lateral):
        return "forward"
    if lateral > 0:
        return "turn_left"
    if lateral < 0:
        return "turn_right"
    return "turn_left"


def _egocentric_frontier_action_at_current(memory: dict) -> str | None:
    candidates = [
        ((1, 0), "forward"),
        ((0, 1), "turn_left"),
        ((0, -1), "turn_right"),
        ((-1, 0), "turn_left"),
    ]
    for neighbor, action in candidates:
        if neighbor not in memory["free"] and neighbor not in memory["blocked"]:
            return action
    return None


def _select_egocentric_frontier_action(memory: dict) -> str:
    current = (0, 0)
    memory["free"].add(current)
    memory["blocked"].discard(current)
    marker = memory.get("marker")
    if marker is not None and marker in memory["free"]:
        if marker == current:
            return "hold"
        next_step = _egocentric_bfs_first_step(memory, target=marker)
        if next_step is not None:
            return _action_toward_egocentric_step(next_step)
    if _is_egocentric_frontier_cell(memory, current):
        action = _egocentric_frontier_action_at_current(memory)
        if action is not None:
            return action
    next_step = _egocentric_bfs_first_step(memory, frontier=True)
    if next_step is not None:
        return _action_toward_egocentric_step(next_step)
    return "turn_left"


def _value_field_first_step(
    memory: dict,
    *,
    targets: set[tuple[int, int]],
    iterations: int = 64,
    gamma: float = 0.97,
) -> tuple[int, int] | None:
    free = set(memory["free"])
    free.add((0, 0))
    if not targets:
        return None
    reachable_targets = targets & free
    if not reachable_targets:
        return None
    floor = -1.0e6
    values = {
        cell: (1.0 if cell in reachable_targets else floor)
        for cell in free
    }
    for _ in range(max(1, iterations)):
        updated = {}
        for cell in free:
            neighbor_value = max(
                (values.get(neighbor, floor) for neighbor in _egocentric_neighbors(cell)),
                default=floor,
            )
            target_value = 1.0 if cell in reachable_targets else floor
            updated[cell] = max(target_value, gamma * neighbor_value - 0.01)
        values = updated
    candidates = [
        neighbor
        for neighbor in ((1, 0), (0, 1), (0, -1), (-1, 0))
        if neighbor in free
    ]
    if not candidates:
        return None
    best_step = max(candidates, key=lambda cell: values.get(cell, floor))
    if values.get(best_step, floor) <= floor / 2:
        return None
    return best_step


def _select_egocentric_value_field_action(memory: dict) -> tuple[str, str]:
    current = (0, 0)
    memory["free"].add(current)
    memory["blocked"].discard(current)
    marker = memory.get("marker")
    if marker is not None and marker in memory["free"]:
        if marker == current:
            return "hold", "latent_recurrent_value_marker"
        next_step = _value_field_first_step(memory, targets={marker})
        if next_step is not None:
            return (
                _action_toward_egocentric_step(next_step),
                "latent_recurrent_value_marker",
            )
    if _is_egocentric_frontier_cell(memory, current):
        action = _egocentric_frontier_action_at_current(memory)
        if action is not None:
            return action, "latent_recurrent_value_current_frontier"
    targets = {
        cell
        for cell in memory["free"]
        if cell != current and _is_egocentric_frontier_cell(memory, cell)
    }
    next_step = _value_field_first_step(memory, targets=targets)
    if next_step is not None:
        return (
            _action_toward_egocentric_step(next_step),
            "latent_recurrent_value_frontier",
        )
    return "turn_left", "latent_recurrent_value_fallback"


def _select_egocentric_learned_value_field_action(
    memory: dict,
    target_probs: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
    fixed_marker_target: bool = False,
) -> tuple[str, str]:
    if target_probs.ndim != 2:
        raise ValueError(f"expected target_probs shape (S, S), got {target_probs.shape}")
    memory_size = int(target_probs.shape[0])
    radius = memory_size // 2
    free = set(memory["free"])
    free.add((0, 0))
    marker = memory.get("marker")
    if fixed_marker_target and marker is not None and marker in free:
        if marker == (0, 0):
            return "hold", "latent_recurrent_learned_value_fixed_marker"
        next_step = _value_field_first_step(memory, targets={marker})
        if next_step is not None:
            return (
                _action_toward_egocentric_step(next_step),
                "latent_recurrent_learned_value_fixed_marker",
            )
    scored: list[tuple[float, tuple[int, int]]] = []
    for cell in free:
        ahead, lateral = cell
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            scored.append((float(target_probs[row, col]), cell))
    if not scored:
        return "turn_left", "latent_recurrent_learned_value_no_targets"
    scored.sort(key=lambda item: item[0], reverse=True)
    targets = {
        cell
        for score, cell in scored[: max(1, int(top_k))]
        if score >= threshold
    }
    if not targets:
        targets = {scored[0][1]}
    if (0, 0) in targets and _is_egocentric_frontier_cell(memory, (0, 0)):
        action = _egocentric_frontier_action_at_current(memory)
        if action is not None:
            return action, "latent_recurrent_learned_value_current_target"
    next_step = _value_field_first_step(memory, targets=targets)
    if next_step is not None:
        return (
            _action_toward_egocentric_step(next_step),
            "latent_recurrent_learned_value_field",
        )
    return "turn_left", "latent_recurrent_learned_value_fallback"


def _egocentric_cell_value(value_probs: torch.Tensor, cell: tuple[int, int]) -> float:
    memory_size = int(value_probs.shape[0])
    radius = memory_size // 2
    row = radius - int(cell[0])
    col = radius + int(cell[1])
    if 0 <= row < memory_size and 0 <= col < memory_size:
        return float(value_probs[row, col])
    return -1.0


def _select_egocentric_learned_value_map_action(
    memory: dict,
    value_probs: torch.Tensor,
) -> tuple[str, str]:
    """Select a primitive by locally reading a learned dense value map."""

    if value_probs.ndim != 2:
        raise ValueError(f"expected value_probs shape (S, S), got {value_probs.shape}")
    current = (0, 0)
    memory["free"].add(current)
    memory["blocked"].discard(current)
    marker = memory.get("marker")
    if marker == current:
        return "hold", "latent_recurrent_learned_value_map_marker"
    center_value = _egocentric_cell_value(value_probs, current)
    candidates = [
        ((1, 0), "forward"),
        ((0, 1), "turn_left"),
        ((0, -1), "turn_right"),
        ((-1, 0), "turn_left"),
    ]
    scored = [
        (_egocentric_cell_value(value_probs, cell), action, cell)
        for cell, action in candidates
        if cell not in memory["blocked"]
    ]
    if not scored:
        return "turn_left", "latent_recurrent_learned_value_map_fallback"
    if (
        _is_egocentric_frontier_cell(memory, current)
        and center_value >= max(score for score, _action, _cell in scored)
    ):
        frontier_action = _egocentric_frontier_action_at_current(memory)
        if frontier_action is not None:
            return frontier_action, "latent_recurrent_learned_value_map_current_frontier"
    _score, action, _cell = max(
        scored,
        key=lambda item: (
            item[0],
            1 if item[1] == "forward" else 0,
            1 if item[1] == "turn_left" else 0,
        ),
    )
    return action, "latent_recurrent_learned_value_map"


def _egocentric_has_contiguous_side_wall(memory: dict, *, min_run: int = 3) -> bool:
    blocked = set(memory["blocked"])
    for sign in (-1, 1):
        if all((0, sign * offset) in blocked for offset in range(1, min_run + 1)):
            return True
    return False


def _break_turn_oscillation_action(
    planned_action: str,
    memory: dict,
    trajectory: list[dict],
    state: GridState,
) -> str | None:
    """Force progress out of repeated in-place left/right turn oscillations."""

    turn_actions = {"turn_left", "turn_right"}
    if planned_action not in turn_actions or (1, 0) in memory["blocked"]:
        return None
    if len(trajectory) < 2:
        return None
    previous = trajectory[-2:]
    previous_actions = [str(item["selected_action"]) for item in previous]
    if any(action not in turn_actions for action in previous_actions):
        return None
    if previous_actions[0] == previous_actions[1]:
        return None
    if planned_action == previous_actions[-1]:
        return None
    if any(bool(item["collision"]) for item in previous):
        return None
    current_xy = (int(state.x), int(state.y))
    for item in previous:
        item_xy = (int(item["state"]["x"]), int(item["state"]["y"]))
        next_xy = (int(item["next_state"]["x"]), int(item["next_state"]["y"]))
        if item_xy != current_xy or next_xy != current_xy:
            return None
    return "forward"


def _break_repeated_state_action(
    planned_action: str,
    memory: dict,
    trajectory: list[dict],
    state: GridState,
    *,
    selection_mode: str = "",
    lookback: int = 24,
) -> str | None:
    """Avoid reusing the same primitive from a recently repeated pose."""

    pose = (int(state.x), int(state.y), int(state.yaw))
    tried_actions = [
        str(item["selected_action"])
        for item in trajectory[-lookback:]
        if (
            int(item["state"]["x"]),
            int(item["state"]["y"]),
            int(item["state"]["yaw"]),
        )
        == pose
    ]
    if (
        "marker_action_return" in selection_mode
        and planned_action in {"turn_left", "turn_right"}
        and (1, 0) not in memory["blocked"]
    ):
        xy = (int(state.x), int(state.y))
        xy_turn_actions = {
            str(item["selected_action"])
            for item in trajectory[-lookback:]
            if (int(item["state"]["x"]), int(item["state"]["y"])) == xy
            and str(item["selected_action"]) in {"turn_left", "turn_right"}
        }
        if len(xy_turn_actions | {planned_action}) >= 2:
            return "forward"
    if planned_action not in tried_actions:
        return None
    turn_candidates = ["turn_left", "turn_right"]
    if planned_action == "turn_left":
        turn_candidates = ["turn_right", "turn_left"]
    elif planned_action == "turn_right":
        turn_candidates = ["turn_left", "turn_right"]
    candidates = (
        ["turn_left", "turn_right", "forward"]
        if planned_action == "forward"
        else ["forward", *turn_candidates]
    )
    for candidate in candidates:
        if candidate == planned_action:
            continue
        if candidate == "forward" and (1, 0) in memory["blocked"]:
            continue
        if candidate not in tried_actions:
            return candidate
    for candidate in candidates:
        if candidate != planned_action and not (
            candidate == "forward" and (1, 0) in memory["blocked"]
        ):
            return candidate
    return None


def _latent_soft_value_map(
    memory: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    iterations: int,
    gamma: float,
) -> torch.Tensor:
    if memory.ndim != 4 or int(memory.shape[1]) < 2:
        raise ValueError(f"expected memory shape (B, >=2, S, S), got {memory.shape}")
    if target_probs.ndim != 4 or int(target_probs.shape[1]) != 1:
        raise ValueError(
            f"expected target_probs shape (B, 1, S, S), got {target_probs.shape}"
        )
    blocked = memory[:, 0:1].clamp(0.0, 1.0)
    free = memory[:, 1:2].clamp(0.0, 1.0)
    passable = (free * (1.0 - blocked)).clamp(0.0, 1.0)
    radius = int(memory.shape[-1]) // 2
    passable[:, :, radius, radius] = 1.0
    values = target_probs.clamp(0.0, 1.0)
    for _ in range(max(1, int(iterations))):
        north = F.pad(values[:, :, 1:, :], (0, 0, 0, 1))
        south = F.pad(values[:, :, :-1, :], (0, 0, 1, 0))
        east = F.pad(values[:, :, :, :-1], (1, 0, 0, 0))
        west = F.pad(values[:, :, :, 1:], (0, 1, 0, 0))
        neighbor_values = torch.maximum(
            torch.maximum(north, south),
            torch.maximum(east, west),
        )
        values = torch.maximum(values, passable * float(gamma) * neighbor_values)
    return values


def _sparse_target_tensor(
    target_probs: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
) -> torch.Tensor:
    if target_probs.ndim != 4 or int(target_probs.shape[1]) != 1:
        raise ValueError(
            f"expected target_probs shape (B, 1, S, S), got {target_probs.shape}"
        )
    batch = int(target_probs.shape[0])
    flat = target_probs.flatten(start_dim=1)
    k = min(max(1, int(top_k)), int(flat.shape[1]))
    values, indices = torch.topk(flat, k=k, dim=1)
    keep = values >= float(threshold)
    keep[:, 0] = True
    sparse = torch.zeros_like(flat)
    sparse.scatter_(1, indices, values * keep.to(dtype=values.dtype))
    return sparse.view_as(target_probs)


def _sparse_target_tensor_from_memory(
    memory: dict,
    target_probs: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    if target_probs.ndim != 2:
        raise ValueError(f"expected target_probs shape (S, S), got {target_probs.shape}")
    memory_size = int(target_probs.shape[0])
    radius = memory_size // 2
    free = set(memory["free"])
    free.add((0, 0))
    scored = []
    for ahead, lateral in free:
        row = radius - int(ahead)
        col = radius + int(lateral)
        if 0 <= row < memory_size and 0 <= col < memory_size:
            scored.append((float(target_probs[row, col]), row, col))
    if not scored:
        scored = [(float(target_probs[radius, radius]), radius, radius)]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [
        item
        for item in scored[: max(1, int(top_k))]
        if float(item[0]) >= float(threshold)
    ] or [scored[0]]
    tensor = torch.zeros(1, 1, memory_size, memory_size, dtype=torch.float32)
    for score, row, col in selected:
        tensor[0, 0, row, col] = float(score)
    return tensor.to(device=device)


def _selection_for_single_action(
    rows: list[dict],
    action: str,
) -> tuple[dict, dict]:
    matching = [
        row for row in rows if str(row["primitive_sequence"][0]) == action
    ]
    selected_row = max(
        matching or rows,
        key=lambda row: float(row["consequence_labels"]["target_utility"]),
    )
    oracle_row = max(
        rows,
        key=lambda row: float(row["consequence_labels"]["target_utility"]),
    )
    return (
        {
            "row": selected_row,
            "score": 0.0,
            "action": action,
            "sequence": (action,),
            "utility": float(selected_row["consequence_labels"]["target_utility"]),
        },
        {
            "row": oracle_row,
            "action": str(oracle_row["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in oracle_row["primitive_sequence"]),
            "utility": float(oracle_row["consequence_labels"]["target_utility"]),
        },
    )


@torch.no_grad()
def _initialize_persistent_memory(
    model: torch.nn.Module,
    *,
    scene: GridScene,
    state: GridState,
    history_states: list[GridState],
    history_actions: list[str],
    view_size: int,
    current_goal_marker: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history_observations = [
        render_observation(
            scene,
            history_state,
            view_size=view_size,
            include_goal_beacon=False,
            show_goal_marker=True,
        )
        for history_state in history_states
    ]
    history_vision = torch.tensor(
        history_observations,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    history_action_tensor = torch.tensor(
        [list(action_vector(action)) for action in history_actions],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    start_vision = _observation_tensor(
        render_observation(
            scene,
            state,
            view_size=view_size,
            include_goal_beacon=False,
            show_goal_marker=current_goal_marker,
        ),
        device=device,
    )
    return model.spatial_frontier_memory_start_maps(
        history_vision,
        history_action_tensor,
        start_vision,
        batch_size=1,
        device=device,
        dtype=torch.float32,
    )


@torch.no_grad()
def _select_action_from_persistent_memory(
    model: torch.nn.Module,
    rows: list[dict],
    memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict, dict]:
    batch = Phase3AMaterializedDataset(rows).materialize_batch(range(len(rows))).to(device)
    marker_belief, marker_mass, observed, _free, blocked = memory
    count = batch.actions.shape[0]
    scores = model.spatial_frontier_memory_score(
        marker_belief.expand(count, -1),
        marker_mass.expand(count),
        observed.expand(count, -1, -1),
        blocked.expand(count, -1, -1),
        batch.actions,
    ).detach().cpu()
    selected_index = int(scores.argmax().item())
    oracle_index = max(
        range(len(rows)),
        key=lambda index: float(rows[index]["consequence_labels"]["target_utility"]),
    )
    selected = rows[selected_index]
    oracle = rows[oracle_index]
    return (
        {
            "row": selected,
            "score": float(scores[selected_index]),
            "action": str(selected["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in selected["primitive_sequence"]),
            "utility": float(selected["consequence_labels"]["target_utility"]),
        },
        {
            "row": oracle,
            "action": str(oracle["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in oracle["primitive_sequence"]),
            "utility": float(oracle["consequence_labels"]["target_utility"]),
        },
    )


@torch.no_grad()
def _select_action_from_persistent_marker_memory(
    model: torch.nn.Module,
    rows: list[dict],
    memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict, dict]:
    batch = Phase3AMaterializedDataset(rows).materialize_batch(range(len(rows))).to(device)
    marker_belief, marker_mass, _observed, _free, _blocked = memory
    count = batch.actions.shape[0]
    scores, _ = model.spatial_marker_memory_score(
        marker_belief.expand(count, -1),
        marker_mass.expand(count),
        batch.actions,
    )
    scores = scores.detach().cpu()
    selected_index = int(scores.argmax().item())
    oracle_index = max(
        range(len(rows)),
        key=lambda index: float(rows[index]["consequence_labels"]["target_utility"]),
    )
    selected = rows[selected_index]
    oracle = rows[oracle_index]
    return (
        {
            "row": selected,
            "score": float(scores[selected_index]),
            "action": str(selected["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in selected["primitive_sequence"]),
            "utility": float(selected["consequence_labels"]["target_utility"]),
        },
        {
            "row": oracle,
            "action": str(oracle["primitive_sequence"][0]),
            "sequence": tuple(str(item) for item in oracle["primitive_sequence"]),
            "utility": float(oracle["consequence_labels"]["target_utility"]),
        },
    )


@torch.no_grad()
def _advance_persistent_memory(
    model: torch.nn.Module,
    memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    action: str,
    scene: GridScene,
    next_state: GridState,
    view_size: int,
    current_goal_marker: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_observation = render_observation(
        scene,
        next_state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    return model.step_spatial_frontier_memory(
        *memory,
        _action_tensor(action, device=device),
        _observation_tensor(next_observation, device=device),
    )


def _group_validation_sources(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_key(row)].append(row)
    groups = [grouped[key] for key in sorted(grouped)]
    return [
        group
        for group in groups
        if not any(bool(row.get("history_goal_marker_seen", False)) for row in group)
        and not any(bool(row.get("current_goal_marker_seen", False)) for row in group)
    ] or groups


def _run_episode(
    model: torch.nn.Module,
    group: list[dict],
    *,
    seed: int,
    width: int,
    height: int,
    view_size: int,
    horizon: int,
    max_steps: int,
    execute_block_steps: int,
    history_window: int,
    max_candidates_per_state: int | None,
    score_source: str,
    odom_frontier_strategy: str,
    odom_frontier_gain_radius: int,
    odom_frontier_distance_penalty: float,
    odom_frontier_turn_penalty: float,
    odom_frontier_neighbor_weight: float,
    odom_frontier_lookahead_horizon: int,
    odom_frontier_lookahead_beam_width: int,
    latent_map_head: Phase3ALatentMapHead | None,
    latent_memory_updater: Phase3AEgocentricMemoryUpdate | None,
    latent_policy_head: Phase3AEgocentricMemoryPolicy | None,
    latent_value_field_head: Phase3AEgocentricValueFieldHead | None,
    latent_value_extractor_head: Phase3AValueFieldExtractorHead | None,
    latent_value_action_head: Phase3AValueFieldActionHead | None,
    latent_reachability_head: Phase3BReachabilityHead | None,
    latent_reachability_value_map_planner_head: (
        Phase3BReachabilityConditionedValueMapPlannerHead | None
    ),
    latent_pre_marker_action_correction_head: Phase3AActionCorrectionHead | None,
    latent_pre_marker_action_correction_threshold: float,
    latent_pre_marker_action_correction_initial_threshold: float | None,
    latent_pre_marker_action_correction_initial_max_step: int | None,
    latent_pre_marker_action_correction_min_step: int | None,
    latent_pre_marker_action_correction_max_step: int | None,
    latent_action_correction_head: Phase3AActionCorrectionHead | None,
    latent_action_correction_threshold: float,
    latent_action_correction_mode: str,
    latent_action_correction_min_step: int | None,
    latent_action_correction_max_step: int | None,
    latent_value_map_planner_head: Phase3AValueMapPlannerHead | None,
    latent_marker_value_map_planner_head: Phase3AValueMapPlannerHead | None,
    latent_value_map_ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    latent_value_map_ensemble_mode: str,
    latent_value_map_readout: str,
    latent_value_map_fallback_head: Phase3AValueMapPlannerHead | None,
    latent_value_map_fallback_ensemble_heads: tuple[Phase3AValueMapPlannerHead, ...],
    latent_value_map_fallback_after_step: int | None,
    latent_value_map_router_head: Phase3AValueMapRouterHead | None,
    latent_value_map_router_threshold: float,
    latent_soft_value_router_head: Phase3AValueMapRouterHead | None,
    latent_soft_value_router_threshold: float,
    latent_soft_value_router_mode: str,
    latent_value_map_side_wall_fallback: bool,
    latent_value_map_fixed_marker_return: bool,
    latent_value_map_marker_action_return: bool,
    latent_value_map_current_marker_action_return: bool,
    latent_value_map_current_marker_action_threshold: float,
    latent_value_map_current_marker_local_threshold: float | None,
    latent_value_map_marker_action_require_local_evidence: bool,
    latent_value_map_turn_oscillation_breaker: bool,
    latent_value_map_state_loop_breaker: bool,
    latent_map_blocked_threshold: float,
    latent_map_marker_threshold: float,
    latent_memory_blocked_threshold: float,
    latent_memory_free_threshold: float,
    latent_memory_marker_threshold: float,
    latent_memory_merge_current_marker_evidence: bool,
    latent_value_target_threshold: float,
    latent_value_target_top_k: int,
    latent_value_marker_target_threshold: float | None,
    latent_value_marker_target_top_k: int | None,
    latent_value_extractor_threshold: float,
    latent_value_sparse_target_top_k: int,
    latent_value_action_fallback_threshold: float | None,
    latent_soft_value_iterations: int,
    latent_soft_value_gamma: float,
    latent_value_fixed_marker_target: bool,
    persistent_marker_claim_threshold: float,
    persistent_marker_require_seen: bool,
    current_goal_marker: bool,
    device: torch.device,
) -> dict:
    template = group[0]
    scene = _goal_scene_from_row(template, seed=seed, width=width, height=height)
    state = _state_from_dict(template["start_state"])
    history_states = [_state_from_dict(item) for item in template["history_states"]]
    history_actions = [str(item) for item in template["history_primitive_sequence"]]
    trajectory = []
    claimed = (state.x, state.y) == scene.goal
    marker_seen_ever = any(
        _goal_marker_visible(scene, item, view_size=view_size)
        for item in history_states
    )
    latent_marker_seen_ever = False
    odom_frontier_memory = {
        "free": set(),
        "blocked": set(),
        "marker": None,
        "radius": max(int(getattr(model, "spatial_memory_size", view_size)) // 2, 0),
    }
    recurrent_memory_tensor = None
    recurrent_last_action = "hold"
    recurrent_last_collision = False
    router_fallback_active = False
    side_wall_fallback_active = False
    if score_source in (
        "latent_recurrent_egocentric_frontier_planner",
        "latent_recurrent_policy_planner",
        "latent_recurrent_value_field_planner",
        "latent_recurrent_learned_value_field_planner",
        "latent_recurrent_learned_value_action_planner",
        "latent_recurrent_learned_value_map_planner",
        "latent_recurrent_soft_value_map_planner",
    ):
        memory_size = int(getattr(model, "spatial_memory_size", view_size))
        recurrent_memory_tensor = torch.zeros(
            1,
            3,
            memory_size,
            memory_size,
            dtype=torch.float32,
            device=device,
        )
    persistent_memory = None
    if score_source in (
        "persistent_spatial_frontier_memory_score",
        "persistent_marker_bounded_frontier_score",
    ):
        persistent_memory = _initialize_persistent_memory(
            model,
            scene=scene,
            state=state,
            history_states=history_states,
            history_actions=history_actions,
            view_size=view_size,
            current_goal_marker=current_goal_marker,
            device=device,
        )
    step = 0
    while step < max_steps:
        current_marker_seen = _goal_marker_visible(scene, state, view_size=view_size)
        marker_seen_ever = marker_seen_ever or current_marker_seen
        if history_window > 0:
            candidate_history_states = history_states[-history_window:]
            candidate_history_actions = history_actions[-history_window:]
        else:
            candidate_history_states = history_states
            candidate_history_actions = history_actions
        rows = _candidate_rows(
            scene=scene,
            source_index=step,
            state=state,
            history_states=candidate_history_states,
            history_actions=candidate_history_actions,
            horizon=horizon,
            view_size=view_size,
            current_goal_marker=current_goal_marker,
        )
        rows = _subsample_candidates_by_first(rows, max_candidates_per_state)
        selection_mode = score_source
        persistent_marker_mass = None
        pre_marker_action_correction_allowed = False
        pre_marker_action_correction_confidence = None
        pre_marker_action_correction_action = None
        action_correction_allowed = False
        action_correction_confidence = None
        action_correction_action = None
        latent_value_action_candidate_action = None
        latent_value_action_candidate_confidence = None
        latent_current_marker_probability = None
        marker_action_return_action = None
        marker_action_return_confidence = None
        soft_value_router_allowed = False
        soft_value_router_probability = None
        soft_value_router_action = None
        soft_value_router_pre_action = None
        soft_value_router_applied = False
        recurrent_memory_marker = None
        recurrent_memory_marker_in_free = None
        recurrent_memory_marker_probability = None
        recurrent_memory_marker_free_probability = None
        recurrent_memory_marker_blocked_probability = None
        recurrent_memory_free_count = None
        recurrent_memory_blocked_count = None
        if score_source in (
            "odom_frontier_marker_planner",
            "latent_odom_frontier_planner",
            "latent_egocentric_frontier_planner",
            "latent_recurrent_egocentric_frontier_planner",
            "latent_recurrent_policy_planner",
            "latent_recurrent_value_field_planner",
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        ):
            if score_source in (
                "latent_recurrent_egocentric_frontier_planner",
                "latent_recurrent_policy_planner",
                "latent_recurrent_value_field_planner",
                "latent_recurrent_learned_value_field_planner",
                "latent_recurrent_learned_value_action_planner",
                "latent_recurrent_learned_value_map_planner",
                "latent_recurrent_reachability_value_map_planner",
                "latent_recurrent_soft_value_map_planner",
            ):
                if latent_map_head is None or latent_memory_updater is None:
                    raise RuntimeError("latent map head and memory updater are required")
                if score_source == "latent_recurrent_policy_planner" and latent_policy_head is None:
                    raise RuntimeError("latent policy head is required")
                if (
                    score_source
                    in (
                        "latent_recurrent_learned_value_field_planner",
                        "latent_recurrent_learned_value_action_planner",
                        "latent_recurrent_learned_value_map_planner",
                        "latent_recurrent_reachability_value_map_planner",
                        "latent_recurrent_soft_value_map_planner",
                    )
                    and latent_value_field_head is None
                ):
                    raise RuntimeError("latent value-field head is required")
                if (
                    score_source == "latent_recurrent_learned_value_action_planner"
                    and latent_value_action_head is None
                ):
                    raise RuntimeError("latent value action head is required")
                if (
                    score_source == "latent_recurrent_learned_value_map_planner"
                    and latent_value_map_planner_head is None
                ):
                    raise RuntimeError("latent value-map planner head is required")
                if score_source == "latent_recurrent_reachability_value_map_planner":
                    if latent_reachability_head is None:
                        raise RuntimeError("latent reachability head is required")
                    if latent_reachability_value_map_planner_head is None:
                        raise RuntimeError(
                            "latent reachability value-map planner head is required"
                        )
                if recurrent_memory_tensor is None:
                    raise RuntimeError("recurrent memory tensor was not initialized")
                local_evidence = _latent_local_evidence(
                    model=model,
                    latent_map_head=latent_map_head,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    current_goal_marker=current_goal_marker,
                    device=device,
                )
                local_evidence = _center_local_evidence(
                    local_evidence,
                    memory_size=int(recurrent_memory_tensor.shape[-1]),
                )
                latent_current_marker_probability = float(
                    local_evidence[0, 2].max().detach().cpu()
                )
                latent_current_marker_seen = (
                    latent_current_marker_probability >= latent_map_marker_threshold
                )
                latent_marker_seen_ever = (
                    latent_marker_seen_ever or latent_current_marker_seen
                )
                logits = latent_memory_updater(
                    recurrent_memory_tensor,
                    local_evidence,
                    _action_index_tensor(recurrent_last_action, device=device),
                    torch.tensor(
                        [float(recurrent_last_collision)],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                recurrent_memory_tensor = logits.sigmoid().detach()
                if latent_memory_merge_current_marker_evidence:
                    recurrent_memory_tensor = recurrent_memory_tensor.clone()
                    recurrent_memory_tensor[:, 1:2] = torch.maximum(
                        recurrent_memory_tensor[:, 1:2],
                        local_evidence[:, 1:2],
                    )
                    recurrent_memory_tensor[:, 2:3] = torch.maximum(
                        recurrent_memory_tensor[:, 2:3],
                        local_evidence[:, 2:3],
                    )
                if score_source == "latent_recurrent_policy_planner":
                    assert latent_policy_head is not None
                    policy_logits = latent_policy_head(recurrent_memory_tensor)
                    planned_action = ACTION_NAMES[
                        int(policy_logits.argmax(dim=1).item())
                    ]
                    selection_mode = "latent_recurrent_policy"
                else:
                    recurrent_memory = _egocentric_memory_tensor_to_dict(
                        recurrent_memory_tensor[0].detach().cpu(),
                        blocked_threshold=latent_memory_blocked_threshold,
                        free_threshold=latent_memory_free_threshold,
                        marker_threshold=latent_memory_marker_threshold,
                    )
                    recurrent_memory_free_count = len(recurrent_memory["free"])
                    recurrent_memory_blocked_count = len(recurrent_memory["blocked"])
                    memory_tensor_cpu = recurrent_memory_tensor[0].detach().cpu()
                    marker = recurrent_memory.get("marker")
                    if marker is not None:
                        recurrent_memory_marker = [
                            int(marker[0]),
                            int(marker[1]),
                        ]
                        recurrent_memory_marker_in_free = (
                            marker in recurrent_memory["free"]
                        )
                        radius = int(memory_tensor_cpu.shape[-1]) // 2
                        marker_row = radius - int(marker[0])
                        marker_col = radius + int(marker[1])
                        if (
                            0 <= marker_row < int(memory_tensor_cpu.shape[-2])
                            and 0 <= marker_col < int(memory_tensor_cpu.shape[-1])
                        ):
                            recurrent_memory_marker_blocked_probability = float(
                                memory_tensor_cpu[0, marker_row, marker_col].item()
                            )
                            recurrent_memory_marker_free_probability = float(
                                memory_tensor_cpu[1, marker_row, marker_col].item()
                            )
                            recurrent_memory_marker_probability = float(
                                memory_tensor_cpu[2, marker_row, marker_col].item()
                            )
                    if score_source == "latent_recurrent_value_field_planner":
                        planned_action, selection_mode = _select_egocentric_value_field_action(
                            recurrent_memory,
                        )
                    elif score_source in (
                        "latent_recurrent_learned_value_field_planner",
                        "latent_recurrent_learned_value_action_planner",
                        "latent_recurrent_learned_value_map_planner",
                        "latent_recurrent_soft_value_map_planner",
                    ):
                        assert latent_value_field_head is not None
                        target_fields_gpu = latent_value_field_head(
                            recurrent_memory_tensor,
                        ).sigmoid()
                        target_fields = target_fields_gpu[0].detach().cpu()
                        marker = recurrent_memory.get("marker")
                        if (
                            int(target_fields.shape[0]) >= 2
                            and marker is not None
                            and marker in recurrent_memory["free"]
                        ):
                            target_probs = target_fields[0]
                            target_probs_gpu = target_fields_gpu[:, 0:1]
                        elif int(target_fields.shape[0]) >= 2:
                            target_probs = target_fields[1]
                            target_probs_gpu = target_fields_gpu[:, 1:2]
                        else:
                            target_probs = target_fields[0]
                            target_probs_gpu = target_fields_gpu[:, 0:1]
                        target_threshold = latent_value_target_threshold
                        target_top_k = latent_value_target_top_k
                        if (
                            latent_value_fixed_marker_target
                            and marker is not None
                            and marker in recurrent_memory["free"]
                        ):
                            memory_size = int(target_probs.shape[0])
                            radius = memory_size // 2
                            row = radius - int(marker[0])
                            col = radius + int(marker[1])
                            if 0 <= row < memory_size and 0 <= col < memory_size:
                                fixed_target = torch.zeros_like(target_probs_gpu)
                                fixed_target[:, :, row, col] = 1.0
                                target_probs_gpu = fixed_target
                                target_probs = fixed_target[0, 0].detach().cpu()
                        if marker is not None and marker in recurrent_memory["free"]:
                            if latent_value_marker_target_threshold is not None:
                                target_threshold = latent_value_marker_target_threshold
                            if latent_value_marker_target_top_k is not None:
                                target_top_k = latent_value_marker_target_top_k
                        sparse_prob = torch.zeros(
                            1,
                            dtype=torch.float32,
                            device=device,
                        )
                        if latent_value_extractor_head is not None:
                            sparse_prob = latent_value_extractor_head(
                                recurrent_memory_tensor,
                            ).sigmoid()
                            if float(sparse_prob.item()) >= latent_value_extractor_threshold:
                                target_top_k = latent_value_sparse_target_top_k
                        if score_source == "latent_recurrent_learned_value_action_planner":
                            assert latent_value_action_head is not None
                            action_logits = latent_value_action_head(
                                _value_action_memory_input(
                                    latent_value_action_head,
                                    recurrent_memory_tensor,
                                    local_evidence,
                                ),
                                target_probs_gpu,
                                sparse_prob,
                            )
                            action_probs = action_logits.softmax(dim=1)
                            action_confidence = float(action_probs.max(dim=1).values.item())
                            latent_value_action_candidate_action = ACTION_NAMES[
                                int(action_logits.argmax(dim=1).item())
                            ]
                            latent_value_action_candidate_confidence = action_confidence
                            if (
                                latent_value_action_fallback_threshold is not None
                                and action_confidence
                                < latent_value_action_fallback_threshold
                            ):
                                planned_action, _fallback_mode = (
                                    _select_egocentric_learned_value_field_action(
                                        recurrent_memory,
                                        target_probs,
                                        threshold=target_threshold,
                                        top_k=target_top_k,
                                        fixed_marker_target=(
                                            latent_value_fixed_marker_target
                                        ),
                                    )
                                )
                                selection_mode = (
                                    "latent_recurrent_learned_value_action_fallback"
                                )
                            else:
                                planned_action = latent_value_action_candidate_action
                                selection_mode = "latent_recurrent_learned_value_action"
                        elif score_source == "latent_recurrent_learned_value_map_planner":
                            if (
                                latent_policy_head is not None
                                and (marker is None or marker not in recurrent_memory["free"])
                            ):
                                policy_logits = latent_policy_head(recurrent_memory_tensor)
                                planned_action = ACTION_NAMES[
                                    int(policy_logits.argmax(dim=1).item())
                                ]
                                selection_mode = (
                                    "latent_recurrent_learned_value_map_frontier_policy"
                                )
                            else:
                                assert latent_value_map_planner_head is not None
                                planner_head = latent_value_map_planner_head
                                marker_planner = (
                                    latent_marker_value_map_planner_head is not None
                                    and marker is not None
                                    and marker in recurrent_memory["free"]
                                    and (
                                        not latent_value_map_marker_action_require_local_evidence
                                        or latent_marker_seen_ever
                                    )
                                )
                                if marker_planner:
                                    planner_head = latent_marker_value_map_planner_head
                                if (
                                    latent_value_map_router_head is not None
                                    and latent_value_map_fallback_head is not None
                                    and not marker_planner
                                ):
                                    route_probability = float(
                                        latent_value_map_router_head(
                                            recurrent_memory_tensor
                                        ).sigmoid().item()
                                    )
                                    if (
                                        route_probability
                                        >= latent_value_map_router_threshold
                                    ):
                                        router_fallback_active = True
                                if (
                                    latent_value_map_side_wall_fallback
                                    and latent_value_map_fallback_head is not None
                                    and not marker_planner
                                    and _egocentric_has_contiguous_side_wall(
                                        recurrent_memory
                                    )
                                ):
                                    side_wall_fallback_active = True
                                fallback_planner = (
                                    latent_value_map_fallback_head is not None
                                    and not marker_planner
                                    and (
                                        router_fallback_active
                                        or side_wall_fallback_active
                                        or latent_value_map_fallback_after_step is None
                                        or step >= latent_value_map_fallback_after_step
                                    )
                                )
                                if fallback_planner:
                                    planner_head = latent_value_map_fallback_head
                                value_logits = planner_head(
                                    recurrent_memory_tensor,
                                    target_probs_gpu,
                                    sparse_prob,
                                )
                                value_probs = value_logits.sigmoid()
                                if (
                                    fallback_planner
                                    and latent_value_map_fallback_ensemble_heads
                                ):
                                    ensemble_probs = [value_probs]
                                    for ensemble_head in (
                                        latent_value_map_fallback_ensemble_heads
                                    ):
                                        ensemble_probs.append(
                                            ensemble_head(
                                                recurrent_memory_tensor,
                                                target_probs_gpu,
                                                sparse_prob,
                                            ).sigmoid()
                                        )
                                    stacked_probs = torch.stack(ensemble_probs, dim=0)
                                    if latent_value_map_ensemble_mode == "max":
                                        value_probs = stacked_probs.max(dim=0).values
                                    else:
                                        value_probs = stacked_probs.mean(dim=0)
                                if (
                                    latent_value_map_ensemble_heads
                                    and not marker_planner
                                    and not fallback_planner
                                ):
                                    ensemble_probs = [value_probs]
                                    for ensemble_head in latent_value_map_ensemble_heads:
                                        ensemble_probs.append(
                                            ensemble_head(
                                                recurrent_memory_tensor,
                                                target_probs_gpu,
                                                sparse_prob,
                                            ).sigmoid()
                                        )
                                    stacked_probs = torch.stack(ensemble_probs, dim=0)
                                    if latent_value_map_ensemble_mode == "max":
                                        value_probs = stacked_probs.max(dim=0).values
                                    else:
                                        value_probs = stacked_probs.mean(dim=0)
                                if latent_value_map_readout == "field":
                                    planned_action, selection_mode = (
                                        _select_egocentric_learned_value_field_action(
                                            recurrent_memory,
                                            value_probs[0, 0].detach().cpu(),
                                            threshold=target_threshold,
                                            top_k=target_top_k,
                                            fixed_marker_target=False,
                                        )
                                    )
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_field_readout"
                                    )
                                else:
                                    planned_action, selection_mode = (
                                        _select_egocentric_learned_value_map_action(
                                            recurrent_memory,
                                            value_probs[0, 0].detach().cpu(),
                                        )
                                    )
                                if fallback_planner:
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_router_fallback"
                                        if router_fallback_active
                                        else (
                                            "latent_recurrent_learned_value_map_side_wall_fallback"
                                            if side_wall_fallback_active
                                            else (
                                                "latent_recurrent_learned_value_map_fallback_ensemble"
                                                if latent_value_map_fallback_ensemble_heads
                                                else "latent_recurrent_learned_value_map_fallback_head"
                                            )
                                        )
                                    )
                                elif latent_value_map_ensemble_heads and not marker_planner:
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_ensemble_"
                                        f"{latent_value_map_ensemble_mode}"
                                    )
                                if marker_planner:
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_marker_head"
                                    )
                                if (
                                    latent_value_map_marker_action_return
                                    and marker is not None
                                    and marker in recurrent_memory["free"]
                                    and latent_value_action_head is not None
                                    and (
                                        not latent_value_map_marker_action_require_local_evidence
                                        or latent_marker_seen_ever
                                    )
                                ):
                                    action_logits = latent_value_action_head(
                                        _value_action_memory_input(
                                            latent_value_action_head,
                                            recurrent_memory_tensor,
                                            local_evidence,
                                        ),
                                        target_probs_gpu,
                                        sparse_prob,
                                    )
                                    action_probs = action_logits.softmax(dim=1)
                                    marker_action_return_confidence = float(
                                        action_probs.max(dim=1).values.item()
                                    )
                                    marker_action_return_action = ACTION_NAMES[
                                        int(action_logits.argmax(dim=1).item())
                                    ]
                                    latent_value_action_candidate_action = (
                                        marker_action_return_action
                                    )
                                    latent_value_action_candidate_confidence = (
                                        marker_action_return_confidence
                                    )
                                    planned_action = marker_action_return_action
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_marker_action_return"
                                    )
                                if (
                                    latent_value_map_current_marker_action_return
                                    and (
                                        latent_current_marker_probability
                                        >= (
                                            latent_value_map_current_marker_local_threshold
                                            if latent_value_map_current_marker_local_threshold
                                            is not None
                                            else latent_map_marker_threshold
                                        )
                                    )
                                    and latent_value_action_head is not None
                                ):
                                    action_logits = latent_value_action_head(
                                        _value_action_memory_input(
                                            latent_value_action_head,
                                            recurrent_memory_tensor,
                                            local_evidence,
                                        ),
                                        target_probs_gpu,
                                        sparse_prob,
                                    )
                                    action_probs = action_logits.softmax(dim=1)
                                    current_marker_action_confidence = float(
                                        action_probs.max(dim=1).values.item()
                                    )
                                    current_marker_action = ACTION_NAMES[
                                        int(action_logits.argmax(dim=1).item())
                                    ]
                                    latent_value_action_candidate_action = (
                                        current_marker_action
                                    )
                                    latent_value_action_candidate_confidence = (
                                        current_marker_action_confidence
                                    )
                                    if (
                                        current_marker_action_confidence
                                        >= latent_value_map_current_marker_action_threshold
                                    ):
                                        planned_action = current_marker_action
                                        selection_mode = (
                                            "latent_recurrent_learned_value_map_"
                                            "current_marker_action_return"
                                        )
                                if (
                                    latent_value_map_fixed_marker_return
                                    and marker is not None
                                    and marker in recurrent_memory["free"]
                                ):
                                    planned_action, _fixed_mode = (
                                        _select_egocentric_learned_value_field_action(
                                            recurrent_memory,
                                            target_probs,
                                            threshold=target_threshold,
                                            top_k=target_top_k,
                                            fixed_marker_target=True,
                                        )
                                    )
                                    selection_mode = (
                                        "latent_recurrent_learned_value_map_fixed_marker_return"
                                    )
                                soft_value_router_allowed = (
                                    latent_soft_value_router_mode == "all"
                                    or (
                                        latent_soft_value_router_mode
                                        == "latent_marker_seen"
                                        and latent_marker_seen_ever
                                    )
                                    or (
                                        latent_soft_value_router_mode
                                        == "marker_memory"
                                        and marker is not None
                                        and marker in recurrent_memory["free"]
                                    )
                                )
                                if (
                                    latent_soft_value_router_head is not None
                                    and soft_value_router_allowed
                                ):
                                    soft_route_probability = float(
                                        latent_soft_value_router_head(
                                            recurrent_memory_tensor
                                        ).sigmoid().item()
                                    )
                                    soft_value_router_probability = (
                                        soft_route_probability
                                    )
                                    if (
                                        soft_route_probability
                                        >= latent_soft_value_router_threshold
                                    ):
                                        sparse_target_probs_gpu = (
                                            _sparse_target_tensor_from_memory(
                                                recurrent_memory,
                                                target_probs,
                                                threshold=target_threshold,
                                                top_k=target_top_k,
                                                device=device,
                                            )
                                        )
                                        soft_value_probs = _latent_soft_value_map(
                                            recurrent_memory_tensor,
                                            sparse_target_probs_gpu,
                                            iterations=latent_soft_value_iterations,
                                            gamma=latent_soft_value_gamma,
                                        )
                                        soft_value_router_pre_action = planned_action
                                        planned_action, _soft_mode = (
                                            _select_egocentric_learned_value_map_action(
                                                recurrent_memory,
                                                soft_value_probs[0, 0].detach().cpu(),
                                            )
                                        )
                                        soft_value_router_action = planned_action
                                        soft_value_router_applied = True
                                        selection_mode = (
                                            f"{selection_mode}_soft_value_router"
                                        )
                                if latent_value_map_turn_oscillation_breaker:
                                    replacement_action = (
                                        _break_turn_oscillation_action(
                                            planned_action,
                                            recurrent_memory,
                                            trajectory,
                                            state,
                                        )
                                    )
                                    if replacement_action is not None:
                                        planned_action = replacement_action
                                        selection_mode = (
                                            f"{selection_mode}_turn_oscillation_breaker"
                                        )
                                if latent_value_map_state_loop_breaker:
                                    replacement_action = (
                                        _break_repeated_state_action(
                                            planned_action,
                                            recurrent_memory,
                                            trajectory,
                                            state,
                                            selection_mode=selection_mode,
                                        )
                                    )
                                    if replacement_action is not None:
                                        planned_action = replacement_action
                                        selection_mode = (
                                            f"{selection_mode}_state_loop_breaker"
                                        )
                        elif (
                            score_source
                            == "latent_recurrent_reachability_value_map_planner"
                        ):
                            assert latent_reachability_head is not None
                            assert latent_reachability_value_map_planner_head is not None
                            reachability_predictions = latent_reachability_head(
                                recurrent_memory_tensor
                            )
                            reachability_features = reachability_feature_tensor(
                                reachability_predictions,
                                memory_size=int(model.spatial_memory_size),
                            )
                            value_logits = latent_reachability_value_map_planner_head(
                                recurrent_memory_tensor,
                                target_probs_gpu,
                                sparse_prob,
                                reachability_features,
                            )
                            value_probs = value_logits.sigmoid()
                            if latent_value_map_readout == "field":
                                planned_action, _field_mode = (
                                    _select_egocentric_learned_value_field_action(
                                        recurrent_memory,
                                        value_probs[0, 0].detach().cpu(),
                                        threshold=target_threshold,
                                        top_k=target_top_k,
                                        fixed_marker_target=False,
                                    )
                                )
                                selection_mode = (
                                    "latent_recurrent_reachability_value_map_"
                                    "field_readout"
                                )
                            else:
                                planned_action, _map_mode = (
                                    _select_egocentric_learned_value_map_action(
                                        recurrent_memory,
                                        value_probs[0, 0].detach().cpu(),
                                    )
                                )
                                selection_mode = (
                                    "latent_recurrent_reachability_value_map"
                                )
                            if latent_value_map_turn_oscillation_breaker:
                                replacement_action = _break_turn_oscillation_action(
                                    planned_action,
                                    recurrent_memory,
                                    trajectory,
                                    state,
                                )
                                if replacement_action is not None:
                                    planned_action = replacement_action
                                    selection_mode = (
                                        f"{selection_mode}_turn_oscillation_breaker"
                                    )
                            if latent_value_map_state_loop_breaker:
                                replacement_action = _break_repeated_state_action(
                                    planned_action,
                                    recurrent_memory,
                                    trajectory,
                                    state,
                                    selection_mode=selection_mode,
                                )
                                if replacement_action is not None:
                                    planned_action = replacement_action
                                    selection_mode = (
                                        f"{selection_mode}_state_loop_breaker"
                                    )
                        elif score_source == "latent_recurrent_soft_value_map_planner":
                            sparse_target_probs_gpu = _sparse_target_tensor_from_memory(
                                recurrent_memory,
                                target_probs,
                                threshold=target_threshold,
                                top_k=target_top_k,
                                device=device,
                            )
                            value_probs = _latent_soft_value_map(
                                recurrent_memory_tensor,
                                sparse_target_probs_gpu,
                                iterations=latent_soft_value_iterations,
                                gamma=latent_soft_value_gamma,
                            )
                            planned_action, selection_mode = (
                                _select_egocentric_learned_value_map_action(
                                    recurrent_memory,
                                    value_probs[0, 0].detach().cpu(),
                                )
                            )
                            selection_mode = "latent_recurrent_soft_value_map"
                        else:
                            planned_action, selection_mode = (
                                _select_egocentric_learned_value_field_action(
                                    recurrent_memory,
                                    target_probs,
                                    threshold=target_threshold,
                                    top_k=target_top_k,
                                    fixed_marker_target=latent_value_fixed_marker_target,
                                )
                            )
                        pre_marker_action_correction_allowed = (
                            latent_pre_marker_action_correction_head is not None
                            and not latent_marker_seen_ever
                            and (
                                latent_pre_marker_action_correction_min_step is None
                                or step >= latent_pre_marker_action_correction_min_step
                            )
                            and (
                                latent_pre_marker_action_correction_max_step is None
                                or step <= latent_pre_marker_action_correction_max_step
                            )
                        )
                        if pre_marker_action_correction_allowed:
                            pre_marker_correction_threshold = (
                                latent_pre_marker_action_correction_threshold
                            )
                            if (
                                latent_pre_marker_action_correction_initial_threshold
                                is not None
                                and (
                                    latent_pre_marker_action_correction_initial_max_step
                                    is None
                                    or step
                                    <= latent_pre_marker_action_correction_initial_max_step
                                )
                            ):
                                pre_marker_correction_threshold = (
                                    latent_pre_marker_action_correction_initial_threshold
                                )
                            corrected_action, _correction_confidence = (
                                _apply_action_correction(
                                    latent_pre_marker_action_correction_head,
                                    planned_action,
                                    _action_correction_memory_input(
                                        latent_pre_marker_action_correction_head,
                                        recurrent_memory_tensor,
                                        local_evidence,
                                    ),
                                    target_probs_gpu,
                                    sparse_prob,
                                    latent_marker_seen_ever=latent_marker_seen_ever,
                                    threshold=pre_marker_correction_threshold,
                                )
                            )
                            pre_marker_action_correction_confidence = (
                                _correction_confidence
                            )
                            pre_marker_action_correction_action = corrected_action
                            if corrected_action != planned_action:
                                planned_action = corrected_action
                                selection_mode = (
                                    f"{selection_mode}_pre_marker_action_correction"
                                )
                        correction_allowed = (
                            (
                                latent_action_correction_min_step is None
                                or step >= latent_action_correction_min_step
                            )
                            and (
                                latent_action_correction_max_step is None
                                or step <= latent_action_correction_max_step
                            )
                            and (
                                latent_action_correction_mode == "all"
                                or (
                                    latent_action_correction_mode == "marker_return"
                                    and "marker_action_return" in selection_mode
                                )
                                or (
                                    latent_action_correction_mode == "latent_marker_seen"
                                    and latent_marker_seen_ever
                                )
                                or (
                                    latent_action_correction_mode == "marker_seen"
                                    and marker_seen_ever
                                )
                                or (
                                    latent_action_correction_mode
                                    == "current_marker_seen"
                                    and current_marker_seen
                                )
                                or (
                                    latent_action_correction_mode
                                    == "current_marker_seen_value_map"
                                    and current_marker_seen
                                    and selection_mode
                                    == "latent_recurrent_learned_value_map"
                                )
                                or (
                                    latent_action_correction_mode == "pre_latent_marker"
                                    and not latent_marker_seen_ever
                                )
                            )
                        )
                        if (
                            latent_action_correction_head is not None
                            and correction_allowed
                        ):
                            action_correction_allowed = True
                            corrected_action, correction_confidence = (
                                _apply_action_correction(
                                    latent_action_correction_head,
                                    planned_action,
                                    _action_correction_memory_input(
                                        latent_action_correction_head,
                                        recurrent_memory_tensor,
                                        local_evidence,
                                    ),
                                    target_probs_gpu,
                                    sparse_prob,
                                    latent_marker_seen_ever=latent_marker_seen_ever,
                                    threshold=latent_action_correction_threshold,
                                )
                            )
                            action_correction_confidence = correction_confidence
                            action_correction_action = corrected_action
                            if corrected_action != planned_action:
                                planned_action = corrected_action
                                selection_mode = (
                                    f"{selection_mode}_action_correction"
                                )
                                if latent_value_map_turn_oscillation_breaker:
                                    replacement_action = (
                                        _break_turn_oscillation_action(
                                            planned_action,
                                            recurrent_memory,
                                            trajectory,
                                            state,
                                        )
                                    )
                                    if replacement_action is not None:
                                        planned_action = replacement_action
                                        selection_mode = (
                                            f"{selection_mode}"
                                            "_post_correction_turn_oscillation_breaker"
                                        )
                                if latent_value_map_state_loop_breaker:
                                    replacement_action = (
                                        _break_repeated_state_action(
                                            planned_action,
                                            recurrent_memory,
                                            trajectory,
                                            state,
                                            selection_mode=selection_mode,
                                        )
                                    )
                                    if replacement_action is not None:
                                        planned_action = replacement_action
                                        selection_mode = (
                                            f"{selection_mode}"
                                            "_post_correction_state_loop_breaker"
                                        )
                    else:
                        planned_action = _select_egocentric_frontier_action(
                            recurrent_memory,
                        )
                        selection_mode = (
                            "latent_recurrent_egocentric_marker"
                            if recurrent_memory.get("marker") is not None
                            else "latent_recurrent_egocentric_frontier"
                        )
                selected, oracle = _selection_for_single_action(rows, planned_action)
            elif score_source == "latent_egocentric_frontier_planner":
                if latent_map_head is None:
                    raise RuntimeError("latent map head is required")
                _update_latent_egocentric_frontier_memory(
                    odom_frontier_memory,
                    model=model,
                    latent_map_head=latent_map_head,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    current_goal_marker=current_goal_marker,
                    blocked_threshold=latent_map_blocked_threshold,
                    marker_threshold=latent_map_marker_threshold,
                    device=device,
                )
                planned_action = _select_egocentric_frontier_action(
                    odom_frontier_memory,
                )
                selection_mode = (
                    "latent_egocentric_marker"
                    if odom_frontier_memory.get("marker") is not None
                    else "latent_egocentric_frontier"
                )
                selected, oracle = _selection_for_single_action(rows, planned_action)
            elif score_source == "latent_odom_frontier_planner":
                if latent_map_head is None:
                    raise RuntimeError("latent map head is required")
                _update_latent_odom_frontier_memory(
                    odom_frontier_memory,
                    model=model,
                    latent_map_head=latent_map_head,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    current_goal_marker=current_goal_marker,
                    blocked_threshold=latent_map_blocked_threshold,
                    marker_threshold=latent_map_marker_threshold,
                    device=device,
                )
                if odom_frontier_strategy == "lookahead":
                    planned_action = _select_odom_frontier_lookahead_action(
                        odom_frontier_memory,
                        scene=scene,
                        state=state,
                        view_size=view_size,
                        horizon=odom_frontier_lookahead_horizon,
                        beam_width=odom_frontier_lookahead_beam_width,
                    )
                else:
                    planned_action = _select_odom_frontier_action(
                        odom_frontier_memory,
                        state,
                        strategy=odom_frontier_strategy,
                        gain_radius=odom_frontier_gain_radius,
                        distance_penalty=odom_frontier_distance_penalty,
                        turn_penalty=odom_frontier_turn_penalty,
                        neighbor_weight=odom_frontier_neighbor_weight,
                    )
                selection_mode = (
                    "odom_marker"
                    if odom_frontier_memory.get("marker") is not None
                    else f"odom_frontier_{odom_frontier_strategy}"
                )
                selected, oracle = _selection_for_single_action(rows, planned_action)
            else:
                _update_odom_frontier_memory(
                    odom_frontier_memory,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    current_goal_marker=current_goal_marker,
                )
                if odom_frontier_strategy == "lookahead":
                    planned_action = _select_odom_frontier_lookahead_action(
                        odom_frontier_memory,
                        scene=scene,
                        state=state,
                        view_size=view_size,
                        horizon=odom_frontier_lookahead_horizon,
                        beam_width=odom_frontier_lookahead_beam_width,
                    )
                else:
                    planned_action = _select_odom_frontier_action(
                        odom_frontier_memory,
                        state,
                        strategy=odom_frontier_strategy,
                        gain_radius=odom_frontier_gain_radius,
                        distance_penalty=odom_frontier_distance_penalty,
                        turn_penalty=odom_frontier_turn_penalty,
                        neighbor_weight=odom_frontier_neighbor_weight,
                    )
                selection_mode = (
                    "odom_marker"
                    if odom_frontier_memory.get("marker") is not None
                    else f"odom_frontier_{odom_frontier_strategy}"
                )
                selected, oracle = _selection_for_single_action(rows, planned_action)
        elif persistent_memory is None:
            selected, oracle = _select_action(
                model,
                rows,
                score_source=score_source,
                device=device,
            )
        elif score_source == "persistent_marker_bounded_frontier_score" and (
            float(persistent_memory[1].max().detach().cpu())
            < persistent_marker_claim_threshold
            or (persistent_marker_require_seen and not marker_seen_ever)
        ):
            persistent_marker_mass = float(persistent_memory[1].max().detach().cpu())
            selection_mode = (
                "bounded_frontier_seen_gate"
                if persistent_marker_require_seen and not marker_seen_ever
                else "bounded_frontier"
            )
            selected, oracle = _select_action(
                model,
                rows,
                score_source="spatial_frontier_memory_score",
                device=device,
            )
        elif score_source == "persistent_marker_bounded_frontier_score":
            persistent_marker_mass = float(persistent_memory[1].max().detach().cpu())
            selection_mode = "persistent_marker"
            selected, oracle = _select_action_from_persistent_marker_memory(
                model,
                rows,
                persistent_memory,
                device=device,
            )
        else:
            persistent_marker_mass = float(persistent_memory[1].max().detach().cpu())
            selected, oracle = _select_action_from_persistent_memory(
                model,
                rows,
                persistent_memory,
                device=device,
            )
        block_actions = selected["sequence"][:execute_block_steps]
        for block_step, action in enumerate(block_actions):
            if step >= max_steps:
                break
            current_marker_seen = _goal_marker_visible(
                scene,
                state,
                view_size=view_size,
            )
            marker_seen_ever = marker_seen_ever or current_marker_seen
            next_state, collision = step_state(scene, state, action)
            trajectory.append(
                {
                    "step": step,
                    "block_step": block_step,
                    "state": _state_dict(state),
                    "next_state": _state_dict(next_state),
                    "goal_distance": scene.distance_to_goal(state.x, state.y),
                    "current_marker_seen": current_marker_seen,
                    "marker_seen_ever": marker_seen_ever,
                    "latent_marker_seen_ever": latent_marker_seen_ever,
                    "selected_action": action,
                    "selected_sequence": list(selected["sequence"]),
                    "selected_score": selected["score"],
                    "selection_mode": selection_mode,
                    "persistent_marker_mass": persistent_marker_mass,
                    "pre_marker_action_correction_allowed": (
                        pre_marker_action_correction_allowed
                    ),
                    "pre_marker_action_correction_confidence": (
                        pre_marker_action_correction_confidence
                    ),
                    "pre_marker_action_correction_action": (
                        pre_marker_action_correction_action
                    ),
                    "action_correction_allowed": action_correction_allowed,
                    "action_correction_confidence": action_correction_confidence,
                    "action_correction_action": action_correction_action,
                    "latent_value_action_candidate_action": (
                        latent_value_action_candidate_action
                    ),
                    "latent_value_action_candidate_confidence": (
                        latent_value_action_candidate_confidence
                    ),
                    "latent_current_marker_probability": (
                        latent_current_marker_probability
                    ),
                    "marker_action_return_action": marker_action_return_action,
                    "marker_action_return_confidence": marker_action_return_confidence,
                    "soft_value_router_allowed": soft_value_router_allowed,
                    "soft_value_router_probability": soft_value_router_probability,
                    "soft_value_router_pre_action": soft_value_router_pre_action,
                    "soft_value_router_action": soft_value_router_action,
                    "soft_value_router_applied": soft_value_router_applied,
                    "recurrent_memory_marker": recurrent_memory_marker,
                    "recurrent_memory_marker_in_free": recurrent_memory_marker_in_free,
                    "recurrent_memory_marker_probability": (
                        recurrent_memory_marker_probability
                    ),
                    "recurrent_memory_marker_free_probability": (
                        recurrent_memory_marker_free_probability
                    ),
                    "recurrent_memory_marker_blocked_probability": (
                        recurrent_memory_marker_blocked_probability
                    ),
                    "recurrent_memory_free_count": recurrent_memory_free_count,
                    "recurrent_memory_blocked_count": recurrent_memory_blocked_count,
                    "selected_utility": selected["utility"],
                    "oracle_action": oracle["action"],
                    "oracle_sequence": list(oracle["sequence"]),
                    "oracle_utility": oracle["utility"],
                    "collision": collision,
                }
            )
            history_states.append(state)
            history_actions.append(action)
            state = next_state
            if persistent_memory is not None:
                persistent_memory = _advance_persistent_memory(
                    model,
                    persistent_memory,
                    action=action,
                    scene=scene,
                    next_state=state,
                    view_size=view_size,
                    current_goal_marker=current_goal_marker,
                    device=device,
                )
            if score_source == "latent_egocentric_frontier_planner":
                _roll_egocentric_frontier_memory(
                    odom_frontier_memory,
                    action,
                    collision=bool(collision),
                )
            if score_source in (
                "latent_recurrent_egocentric_frontier_planner",
                "latent_recurrent_policy_planner",
                "latent_recurrent_value_field_planner",
                "latent_recurrent_learned_value_field_planner",
                "latent_recurrent_learned_value_action_planner",
                "latent_recurrent_learned_value_map_planner",
                "latent_recurrent_soft_value_map_planner",
            ):
                recurrent_last_action = action
                recurrent_last_collision = bool(collision)
            claimed = (state.x, state.y) == scene.goal
            step += 1
            if claimed:
                break
        if claimed:
            break
    return {
        "scene": scene,
        "start": _state_dict(_state_from_dict(template["start_state"])),
        "goal": {"x": scene.goal[0], "y": scene.goal[1]},
        "final_state": _state_dict(state),
        "claimed": claimed,
        "marker_seen_ever": marker_seen_ever,
        "latent_marker_seen_ever": latent_marker_seen_ever,
        "steps": len(trajectory),
        "trajectory": trajectory,
    }


def _draw_episode_frame(
    episode: dict,
    *,
    frame_index: int,
    width: int,
    height: int,
    view_size: int,
    current_goal_marker: bool,
) -> Image.Image:
    from PIL import Image, ImageDraw

    scene: GridScene = episode["scene"]
    step = episode["trajectory"][min(frame_index, len(episode["trajectory"]) - 1)]
    state = _state_from_dict(step["state"])
    cell = min(30, (height - 180) // scene.height, (width // 2 - 48) // scene.width)
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title = _load_font(30, bold=True)
    font = _load_font(18)
    small = _load_font(15)
    draw.text((28, 22), "Phase 3A Closed-Loop JEPA Memory Rollout", font=title, fill=(24, 28, 34))
    status = "CLAIMED" if episode["claimed"] else "running / not claimed"
    draw.text((28, 62), f"score: learned spatial frontier memory | result: {status}", font=font, fill=(60, 68, 76))

    grid_x, grid_y = 30, 112
    for y, row in enumerate(scene.grid):
        for x, value in enumerate(row):
            fill = (34, 36, 38) if value == "#" else (224, 226, 216)
            if (x, y) == scene.goal:
                fill = (46, 190, 78)
            if x == state.x and y == state.y:
                fill = (45, 86, 205)
            draw.rectangle(
                (grid_x + x * cell, grid_y + y * cell, grid_x + (x + 1) * cell, grid_y + (y + 1) * cell),
                fill=fill,
                outline=(180, 184, 188),
            )
    draw.text((grid_x, grid_y + scene.height * cell + 12), "world map: blue agent, green beacon", font=small, fill=(80, 86, 94))

    obs = render_observation(
        scene,
        state,
        view_size=view_size,
        include_goal_beacon=False,
        show_goal_marker=current_goal_marker,
    )
    obs_img = _rgb_image(obs, size=260)
    obs_x, obs_y = width // 2 + 40, 116
    canvas.paste(obs_img, (obs_x, obs_y))
    draw.rectangle((obs_x, obs_y, obs_x + 260, obs_y + 260), outline=(40, 44, 50), width=3)
    draw.text((obs_x, obs_y + 276), "current observation, no breadcrumb channel", font=small, fill=(70, 76, 84))

    text_x, text_y = width // 2 + 40, 430
    lines = [
        f"step {step['step']} / {episode['steps']}",
        f"state ({state.x}, {state.y}) yaw {state.yaw}  distance {step['goal_distance']}",
        f"marker visible now: {step['current_marker_seen']}  seen ever: {step['marker_seen_ever']}",
        f"selected: {' -> '.join(step['selected_sequence'])}",
        f"oracle:   {' -> '.join(step['oracle_sequence'])}",
        f"score {step['selected_score']:+.2f}  utility {step['selected_utility']:+.2f}",
        f"collision: {step['collision']}",
    ]
    for index, line in enumerate(lines):
        draw.text((text_x, text_y + index * 28), line, font=font, fill=(30, 36, 42))
    return canvas


def _episode_mode_counts(episode: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in episode["trajectory"]:
        mode = str(step.get("selection_mode", "unknown"))
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _export_episode_mp4(
    episode: dict,
    output: Path,
    *,
    fps: int,
    seconds_per_step: float,
    width: int,
    height: int,
    view_size: int,
    current_goal_marker: bool,
) -> None:
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    frames_per_step = max(1, int(round(fps * seconds_per_step)))
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for step_index in range(max(episode["steps"], 1)):
            frame = _draw_episode_frame(
                episode,
                frame_index=step_index,
                width=width,
                height=height,
                view_size=view_size,
                current_goal_marker=current_goal_marker,
            )
            for _ in range(frames_per_step):
                writer.append_data(np.asarray(frame))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "scene-generation seed; defaults to the train/validation seed in "
            "the dataset manifest when available"
        ),
    )
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=28)
    parser.add_argument("--max-episodes", type=int, default=24)
    parser.add_argument("--selected-episode-index", type=int, default=None)
    parser.add_argument("--execute-block-steps", type=int, default=1)
    parser.add_argument(
        "--history-window",
        type=int,
        default=6,
        help="Number of recent closed-loop history states to expose; 0 keeps all.",
    )
    parser.add_argument(
        "--max-candidates-per-state",
        type=int,
        default=None,
        help=(
            "subsample a fixed set of candidate blocks per first action at each "
            "closed-loop state; useful for matching compact distillation data"
        ),
    )
    parser.add_argument(
        "--score-source",
        choices=(
            "spatial_frontier_memory_score",
            "persistent_spatial_frontier_memory_score",
            "persistent_marker_bounded_frontier_score",
            "egocentric_marker_memory_score",
            "egocentric_marker_bounded_frontier_score",
            "egocentric_explore_claim_score",
            "online_frontier_marker_score",
            "palette_online_frontier_marker_score",
            "odom_frontier_marker_planner",
            "latent_odom_frontier_planner",
            "latent_egocentric_frontier_planner",
            "latent_recurrent_egocentric_frontier_planner",
            "latent_recurrent_policy_planner",
            "latent_recurrent_value_field_planner",
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
            "candidate_score",
        ),
        default="spatial_frontier_memory_score",
    )
    parser.add_argument(
        "--odom-frontier-strategy",
        choices=("nearest", "gain", "lookahead"),
        default="nearest",
        help=(
            "frontier target selection for odom_frontier_marker_planner and "
            "latent_odom_frontier_planner"
        ),
    )
    parser.add_argument(
        "--odom-frontier-gain-radius",
        type=int,
        default=3,
        help="unknown-cell radius used by --odom-frontier-strategy=gain",
    )
    parser.add_argument(
        "--odom-frontier-distance-penalty",
        type=float,
        default=0.6,
        help="path-distance penalty used by --odom-frontier-strategy=gain",
    )
    parser.add_argument(
        "--odom-frontier-turn-penalty",
        type=float,
        default=0.25,
        help="first-turn penalty used by --odom-frontier-strategy=gain",
    )
    parser.add_argument(
        "--odom-frontier-neighbor-weight",
        type=float,
        default=2.0,
        help="unknown-neighbor bonus used by --odom-frontier-strategy=gain",
    )
    parser.add_argument(
        "--odom-frontier-lookahead-horizon",
        type=int,
        default=9,
        help="primitive lookahead depth used by --odom-frontier-strategy=lookahead",
    )
    parser.add_argument(
        "--odom-frontier-lookahead-beam-width",
        type=int,
        default=32,
        help="beam width used by --odom-frontier-strategy=lookahead",
    )
    parser.add_argument(
        "--latent-map-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_latent_map.py",
    )
    parser.add_argument(
        "--latent-memory-updater",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_latent_memory.py",
    )
    parser.add_argument(
        "--latent-policy-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_latent_policy.py",
    )
    parser.add_argument(
        "--latent-value-field-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_value_field.py",
    )
    parser.add_argument(
        "--latent-value-extractor-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_value_extractor.py",
    )
    parser.add_argument(
        "--latent-value-action-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_value_action.py",
    )
    parser.add_argument(
        "--latent-value-map-planner-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3a_value_map_planner.py",
    )
    parser.add_argument(
        "--latent-reachability-head",
        type=Path,
        default=None,
        help="checkpoint from train_jepa_phase3b_reachability.py",
    )
    parser.add_argument(
        "--latent-reachability-value-map-planner-head",
        type=Path,
        default=None,
        help=(
            "checkpoint from "
            "train_jepa_phase3b_reachability_value_map_planner.py"
        ),
    )
    parser.add_argument(
        "--latent-marker-value-map-planner-head",
        type=Path,
        default=None,
        help=(
            "optional value-map planner checkpoint to use once recurrent marker "
            "memory exists"
        ),
    )
    parser.add_argument(
        "--latent-value-map-ensemble-head",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional value-map planner checkpoint to ensemble before marker "
            "memory is available; may be passed multiple times"
        ),
    )
    parser.add_argument(
        "--latent-value-map-ensemble-mode",
        choices=("mean", "max"),
        default="mean",
    )
    parser.add_argument(
        "--latent-value-map-readout",
        choices=("local", "field"),
        default="local",
        help=(
            "how to convert a learned value-map planner output into an action; "
            "local reads immediate neighbors, field follows a value-field path "
            "to the strongest free cells"
        ),
    )
    parser.add_argument(
        "--latent-value-map-fallback-head",
        type=Path,
        default=None,
        help=(
            "optional value-map planner checkpoint to use after "
            "--latent-value-map-fallback-after-step when no marker memory exists"
        ),
    )
    parser.add_argument(
        "--latent-value-map-fallback-ensemble-head",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional value-map planner checkpoint to ensemble with "
            "--latent-value-map-fallback-head after the fallback step"
        ),
    )
    parser.add_argument(
        "--latent-value-map-fallback-after-step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--latent-value-map-router-head",
        type=Path,
        default=None,
        help=(
            "optional router checkpoint that latches --latent-value-map-fallback-head "
            "from recurrent egocentric memory before marker memory exists"
        ),
    )
    parser.add_argument(
        "--latent-value-map-router-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--latent-soft-value-router-head",
        type=Path,
        default=None,
        help=(
            "optional learned router checkpoint that switches the learned "
            "value-map planner to the soft-value readout from recurrent memory"
        ),
    )
    parser.add_argument(
        "--latent-soft-value-router-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--latent-soft-value-router-mode",
        choices=("all", "latent_marker_seen", "marker_memory"),
        default="all",
        help=(
            "where to allow --latent-soft-value-router-head: all states, once "
            "local latent marker evidence has ever appeared, or only while "
            "marker memory is active"
        ),
    )
    parser.add_argument(
        "--latent-value-map-side-wall-fallback",
        action="store_true",
        help=(
            "latch --latent-value-map-fallback-head when recurrent egocentric "
            "memory contains a contiguous side wall next to the agent"
        ),
    )
    parser.add_argument(
        "--latent-value-map-fixed-marker-return",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, use fixed "
            "value-field return once marker memory exists"
        ),
    )
    parser.add_argument(
        "--latent-value-map-marker-action-return",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, use "
            "--latent-value-action-head once marker memory exists"
        ),
    )
    parser.add_argument(
        "--latent-value-map-current-marker-action-return",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, use "
            "--latent-value-action-head when learned local marker evidence is "
            "visible in the current view"
        ),
    )
    parser.add_argument(
        "--latent-value-map-current-marker-action-threshold",
        type=float,
        default=0.0,
        help=(
            "minimum value-action softmax confidence for "
            "--latent-value-map-current-marker-action-return"
        ),
    )
    parser.add_argument(
        "--latent-value-map-current-marker-local-threshold",
        type=float,
        default=None,
        help=(
            "optional learned local marker probability threshold for "
            "--latent-value-map-current-marker-action-return; defaults to "
            "--latent-map-marker-threshold"
        ),
    )
    parser.add_argument(
        "--latent-value-map-marker-action-require-local-evidence",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, only switch to "
            "learned marker-return modes after the latent local-evidence marker "
            "channel has crossed --latent-map-marker-threshold at least once"
        ),
    )
    parser.add_argument(
        "--latent-value-map-turn-oscillation-breaker",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, replace repeated "
            "in-place left/right oscillations with forward when latent memory "
            "does not mark the forward cell blocked"
        ),
    )
    parser.add_argument(
        "--latent-value-map-state-loop-breaker",
        action="store_true",
        help=(
            "for latent_recurrent_learned_value_map_planner, avoid taking the "
            "same primitive again from a recently repeated pose"
        ),
    )
    parser.add_argument(
        "--latent-action-correction-head",
        type=Path,
        default=None,
        help=(
            "optional learned action-correction checkpoint applied after the "
            "learned value planner selects a primitive"
        ),
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-head",
        type=Path,
        default=None,
        help=(
            "optional learned action-correction checkpoint applied only before "
            "latent marker evidence has been observed"
        ),
    )
    parser.add_argument(
        "--latent-action-correction-threshold",
        type=float,
        default=0.0,
        help="minimum correction-head softmax confidence required to override",
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-threshold",
        type=float,
        default=0.0,
        help=(
            "minimum pre-marker correction-head softmax confidence required "
            "to override"
        ),
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-initial-threshold",
        type=float,
        default=None,
        help=(
            "optional alternate pre-marker correction threshold for the first "
            "N steps; use with --latent-pre-marker-action-correction-initial-max-step"
        ),
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-initial-max-step",
        type=int,
        default=None,
        help=(
            "inclusive zero-based final episode step that uses "
            "--latent-pre-marker-action-correction-initial-threshold"
        ),
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-min-step",
        type=int,
        default=None,
        help=(
            "optional inclusive zero-based episode step floor for applying "
            "--latent-pre-marker-action-correction-head"
        ),
    )
    parser.add_argument(
        "--latent-pre-marker-action-correction-max-step",
        type=int,
        default=None,
        help=(
            "optional inclusive zero-based episode step limit for applying "
            "--latent-pre-marker-action-correction-head"
        ),
    )
    parser.add_argument(
        "--latent-action-correction-max-step",
        type=int,
        default=None,
        help=(
            "optional inclusive zero-based episode step limit for applying "
            "--latent-action-correction-head"
        ),
    )
    parser.add_argument(
        "--latent-action-correction-min-step",
        type=int,
        default=None,
        help=(
            "optional inclusive zero-based episode step floor for applying "
            "--latent-action-correction-head"
        ),
    )
    parser.add_argument(
        "--latent-action-correction-mode",
        choices=(
            "all",
            "marker_return",
            "latent_marker_seen",
            "marker_seen",
            "current_marker_seen",
            "current_marker_seen_value_map",
            "pre_latent_marker",
        ),
        default="all",
        help=(
            "where to apply --latent-action-correction-head: every learned "
            "value-map action, only learned marker-return actions, once "
            "latent marker memory has been observed, or before latent marker "
            "memory has been observed"
        ),
    )
    parser.add_argument(
        "--latent-map-blocked-threshold",
        type=float,
        default=0.5,
        help="blocked probability threshold for latent-map odometry memory",
    )
    parser.add_argument(
        "--latent-map-marker-threshold",
        type=float,
        default=0.5,
        help="marker probability threshold for latent-map odometry memory",
    )
    parser.add_argument(
        "--latent-memory-blocked-threshold",
        type=float,
        default=0.5,
        help="blocked probability threshold for recurrent egocentric memory",
    )
    parser.add_argument(
        "--latent-memory-free-threshold",
        type=float,
        default=0.5,
        help="free probability threshold for recurrent egocentric memory",
    )
    parser.add_argument(
        "--latent-memory-marker-threshold",
        type=float,
        default=0.5,
        help="marker probability threshold for recurrent egocentric memory",
    )
    parser.add_argument(
        "--latent-memory-merge-current-marker-evidence",
        action="store_true",
        help=(
            "merge learned current local free/marker evidence into the "
            "recurrent memory tensor before value/readout heads"
        ),
    )
    parser.add_argument(
        "--latent-value-target-threshold",
        type=float,
        default=0.5,
        help="target probability threshold for learned value-field targets",
    )
    parser.add_argument(
        "--latent-value-target-top-k",
        type=int,
        default=16,
        help="maximum learned target cells to keep for value propagation",
    )
    parser.add_argument(
        "--latent-value-marker-target-threshold",
        type=float,
        default=None,
        help="optional learned target threshold override once marker memory exists",
    )
    parser.add_argument(
        "--latent-value-marker-target-top-k",
        type=int,
        default=None,
        help="optional learned target top-k override once marker memory exists",
    )
    parser.add_argument(
        "--latent-value-extractor-threshold",
        type=float,
        default=0.5,
        help="sparse-extraction probability threshold for the learned extractor",
    )
    parser.add_argument(
        "--latent-value-sparse-target-top-k",
        type=int,
        default=1,
        help="target top-k to use when the learned extractor selects sparse mode",
    )
    parser.add_argument(
        "--latent-value-action-fallback-threshold",
        type=float,
        default=None,
        help=(
            "for latent_recurrent_learned_value_action_planner, fall back to "
            "structured value-field action extraction below this action confidence"
        ),
    )
    parser.add_argument(
        "--latent-soft-value-iterations",
        type=int,
        default=64,
        help="soft value-iteration steps for latent_recurrent_soft_value_map_planner",
    )
    parser.add_argument(
        "--latent-soft-value-gamma",
        type=float,
        default=0.97,
        help="discount for latent_recurrent_soft_value_map_planner",
    )
    parser.add_argument(
        "--latent-value-fixed-marker-target",
        action="store_true",
        help=(
            "use the learned target field for frontier exploration, but route to "
            "the recurrent marker cell directly once marker memory is available"
        ),
    )
    parser.add_argument(
        "--persistent-marker-claim-threshold",
        type=float,
        default=0.25,
        help=(
            "marker mass required before persistent-marker/bounded-frontier "
            "selection switches from frontier exploration to marker claiming"
        ),
    )
    parser.add_argument(
        "--persistent-marker-require-seen",
        action="store_true",
        help=(
            "for persistent-marker/bounded-frontier scoring, keep using the bounded "
            "frontier scorer until the marker has appeared in the observation history"
        ),
    )
    parser.add_argument(
        "--spatial-frontier-marker-update-threshold",
        type=float,
        default=None,
        help="override the checkpoint marker-memory write threshold for evaluation",
    )
    parser.add_argument(
        "--spatial-frontier-marker-update-width",
        type=float,
        default=None,
        help="override the checkpoint marker-memory write ramp width for evaluation",
    )
    parser.add_argument(
        "--spatial-marker-memory-score-temperature",
        type=float,
        default=None,
        help="override the checkpoint marker belief temperature for claim scoring",
    )
    parser.add_argument(
        "--exact-online-memory-size",
        type=int,
        default=None,
        help=(
            "override spatial memory size for exact online frontier diagnostics; "
            "useful for checking whether the finite map support is the ceiling"
        ),
    )
    parser.add_argument("--hide-current-marker", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds-per-step", type=float, default=0.8)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="write only the JSON report; avoids optional MP4 dependencies",
    )
    parser.add_argument(
        "--progress-every-episodes",
        type=int,
        default=0,
        help=(
            "print a compact progress line every N completed episodes; useful for "
            "long closed-loop evaluation sweeps"
        ),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help=(
            "optional path for full per-step episode traces; the main report keeps "
            "only compact episode summaries"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.execute_block_steps < 1:
        raise SystemExit("--execute-block-steps must be positive")
    if args.progress_every_episodes < 0:
        raise SystemExit("--progress-every-episodes must be non-negative")
    if args.history_window < 0:
        raise SystemExit("--history-window must be non-negative")
    if args.selected_episode_index is not None and args.selected_episode_index < 0:
        raise SystemExit("--selected-episode-index must be non-negative")
    if (
        args.latent_value_map_fallback_after_step is not None
        and args.latent_value_map_fallback_after_step < 0
    ):
        raise SystemExit("--latent-value-map-fallback-after-step must be non-negative")
    if not 0.0 < args.latent_value_map_router_threshold < 1.0:
        raise SystemExit("--latent-value-map-router-threshold must be in (0, 1)")
    if not 0.0 < args.latent_soft_value_router_threshold < 1.0:
        raise SystemExit("--latent-soft-value-router-threshold must be in (0, 1)")
    if (
        args.latent_value_map_router_head is not None
        and args.latent_value_map_fallback_head is None
    ):
        raise SystemExit(
            "--latent-value-map-router-head requires --latent-value-map-fallback-head"
        )
    if args.latent_value_map_marker_action_return and args.latent_value_action_head is None:
        raise SystemExit(
            "--latent-value-map-marker-action-return requires --latent-value-action-head"
        )
    if (
        args.latent_value_map_current_marker_action_return
        and args.latent_value_action_head is None
    ):
        raise SystemExit(
            "--latent-value-map-current-marker-action-return requires "
            "--latent-value-action-head"
        )
    if not 0.0 <= args.latent_value_map_current_marker_action_threshold <= 1.0:
        raise SystemExit(
            "--latent-value-map-current-marker-action-threshold must be in [0, 1]"
        )
    if (
        args.latent_value_map_current_marker_local_threshold is not None
        and not 0.0 <= args.latent_value_map_current_marker_local_threshold <= 1.0
    ):
        raise SystemExit(
            "--latent-value-map-current-marker-local-threshold must be in [0, 1]"
        )
    if args.max_candidates_per_state is not None and args.max_candidates_per_state < 1:
        raise SystemExit("--max-candidates-per-state must be positive")
    if args.persistent_marker_claim_threshold < 0.0:
        raise SystemExit("--persistent-marker-claim-threshold must be non-negative")
    if args.score_source in (
        "latent_odom_frontier_planner",
        "latent_egocentric_frontier_planner",
        "latent_recurrent_egocentric_frontier_planner",
        "latent_recurrent_policy_planner",
        "latent_recurrent_value_field_planner",
        "latent_recurrent_learned_value_field_planner",
        "latent_recurrent_learned_value_action_planner",
        "latent_recurrent_learned_value_map_planner",
        "latent_recurrent_reachability_value_map_planner",
        "latent_recurrent_soft_value_map_planner",
    ) and args.latent_map_head is None:
        raise SystemExit("--latent-map-head is required for latent score sources")
    if (
        args.score_source in (
            "latent_recurrent_egocentric_frontier_planner",
            "latent_recurrent_policy_planner",
            "latent_recurrent_value_field_planner",
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and args.latent_memory_updater is None
    ):
        raise SystemExit(
            "--latent-memory-updater is required for "
            "latent recurrent score sources"
        )
    if (
        args.score_source == "latent_recurrent_policy_planner"
        and args.latent_policy_head is None
    ):
        raise SystemExit(
            "--latent-policy-head is required for latent_recurrent_policy_planner"
        )
    if (
        args.score_source
        in (
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and args.latent_value_field_head is None
    ):
        raise SystemExit(
            "--latent-value-field-head is required for "
            "latent recurrent learned value planners"
        )
    if (
        args.score_source == "latent_recurrent_reachability_value_map_planner"
        and args.latent_reachability_head is None
    ):
        raise SystemExit(
            "--latent-reachability-head is required for "
            "latent_recurrent_reachability_value_map_planner"
        )
    if (
        args.score_source == "latent_recurrent_reachability_value_map_planner"
        and args.latent_reachability_value_map_planner_head is None
    ):
        raise SystemExit(
            "--latent-reachability-value-map-planner-head is required for "
            "latent_recurrent_reachability_value_map_planner"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_action_planner"
        and args.latent_value_extractor_head is None
    ):
        raise SystemExit(
            "--latent-value-extractor-head is required for "
            "latent_recurrent_learned_value_action_planner"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_action_planner"
        and args.latent_value_action_head is None
    ):
        raise SystemExit(
            "--latent-value-action-head is required for "
            "latent_recurrent_learned_value_action_planner"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_map_planner"
        and args.latent_value_map_planner_head is None
    ):
        raise SystemExit(
            "--latent-value-map-planner-head is required for "
            "latent_recurrent_learned_value_map_planner"
        )
    if not 0.0 <= args.latent_map_blocked_threshold <= 1.0:
        raise SystemExit("--latent-map-blocked-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_map_marker_threshold <= 1.0:
        raise SystemExit("--latent-map-marker-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_blocked_threshold <= 1.0:
        raise SystemExit("--latent-memory-blocked-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_free_threshold <= 1.0:
        raise SystemExit("--latent-memory-free-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_marker_threshold <= 1.0:
        raise SystemExit("--latent-memory-marker-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_value_target_threshold <= 1.0:
        raise SystemExit("--latent-value-target-threshold must be in [0, 1]")
    if args.latent_value_target_top_k < 1:
        raise SystemExit("--latent-value-target-top-k must be positive")
    if (
        args.latent_value_marker_target_threshold is not None
        and not 0.0 <= args.latent_value_marker_target_threshold <= 1.0
    ):
        raise SystemExit("--latent-value-marker-target-threshold must be in [0, 1]")
    if (
        args.latent_value_marker_target_top_k is not None
        and args.latent_value_marker_target_top_k < 1
    ):
        raise SystemExit("--latent-value-marker-target-top-k must be positive")
    if not 0.0 <= args.latent_value_extractor_threshold <= 1.0:
        raise SystemExit("--latent-value-extractor-threshold must be in [0, 1]")
    if args.latent_value_sparse_target_top_k < 1:
        raise SystemExit("--latent-value-sparse-target-top-k must be positive")
    if (
        args.latent_value_action_fallback_threshold is not None
        and not 0.0 <= args.latent_value_action_fallback_threshold <= 1.0
    ):
        raise SystemExit("--latent-value-action-fallback-threshold must be in [0, 1]")
    if args.latent_soft_value_iterations < 1:
        raise SystemExit("--latent-soft-value-iterations must be positive")
    if not 0.0 < args.latent_soft_value_gamma <= 1.0:
        raise SystemExit("--latent-soft-value-gamma must be in (0, 1]")
    if not 0.0 <= args.latent_action_correction_threshold <= 1.0:
        raise SystemExit("--latent-action-correction-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_pre_marker_action_correction_threshold <= 1.0:
        raise SystemExit(
            "--latent-pre-marker-action-correction-threshold must be in [0, 1]"
        )
    if (
        args.latent_pre_marker_action_correction_initial_threshold is not None
        and not (
            0.0
            <= args.latent_pre_marker_action_correction_initial_threshold
            <= 1.0
        )
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-initial-threshold must be "
            "in [0, 1]"
        )
    if (
        args.latent_pre_marker_action_correction_initial_max_step is not None
        and args.latent_pre_marker_action_correction_initial_max_step < 0
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-initial-max-step must be "
            "non-negative"
        )
    if (
        args.latent_pre_marker_action_correction_initial_max_step is not None
        and args.latent_pre_marker_action_correction_initial_threshold is None
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-initial-max-step requires "
            "--latent-pre-marker-action-correction-initial-threshold"
        )
    if (
        args.latent_action_correction_max_step is not None
        and args.latent_action_correction_max_step < 0
    ):
        raise SystemExit("--latent-action-correction-max-step must be non-negative")
    if (
        args.latent_action_correction_min_step is not None
        and args.latent_action_correction_min_step < 0
    ):
        raise SystemExit("--latent-action-correction-min-step must be non-negative")
    if (
        args.latent_action_correction_min_step is not None
        and args.latent_action_correction_max_step is not None
        and args.latent_action_correction_min_step
        > args.latent_action_correction_max_step
    ):
        raise SystemExit(
            "--latent-action-correction-min-step must be <= "
            "--latent-action-correction-max-step"
        )
    if (
        args.latent_pre_marker_action_correction_min_step is not None
        and args.latent_pre_marker_action_correction_min_step < 0
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-min-step must be non-negative"
        )
    if (
        args.latent_pre_marker_action_correction_max_step is not None
        and args.latent_pre_marker_action_correction_max_step < 0
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-max-step must be non-negative"
        )
    if (
        args.latent_pre_marker_action_correction_min_step is not None
        and args.latent_pre_marker_action_correction_max_step is not None
        and args.latent_pre_marker_action_correction_min_step
        > args.latent_pre_marker_action_correction_max_step
    ):
        raise SystemExit(
            "--latent-pre-marker-action-correction-min-step must be <= "
            "--latent-pre-marker-action-correction-max-step"
        )

    scene_seed = args.seed
    if scene_seed is None:
        scene_seed = _infer_scene_seed(args.validation_data)
    if scene_seed is None:
        raise SystemExit(
            "--seed is required when no train/validation seed can be inferred "
            "from the dataset manifest"
        )
    rows = read_jsonl(args.validation_data)
    model, report = load_model(args.checkpoint, device=torch.device(args.device))
    latent_map_head = None
    latent_map_report = None
    if args.latent_map_head is not None:
        try:
            latent_checkpoint = torch.load(
                args.latent_map_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            latent_checkpoint = torch.load(
                args.latent_map_head,
                map_location=torch.device(args.device),
            )
        latent_map_report = latent_checkpoint.get("report", {})
        latent_config = latent_map_report.get("model_config", {})
        latent_map_head = Phase3ALatentMapHead(
            view_size=int(latent_config.get("view_size", model.view_size)),
            latent_dim=int(latent_config.get("latent_dim", model.latent_dim)),
            hidden_dim=int(latent_config.get("hidden_dim", 96)),
            output_channels=int(latent_config.get("output_channels", 3)),
        ).to(torch.device(args.device))
        latent_map_head.load_state_dict(latent_checkpoint["head_state_dict"])
        latent_map_head.eval()
    latent_memory_updater = None
    latent_memory_report = None
    if args.latent_memory_updater is not None:
        try:
            memory_checkpoint = torch.load(
                args.latent_memory_updater,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            memory_checkpoint = torch.load(
                args.latent_memory_updater,
                map_location=torch.device(args.device),
            )
        latent_memory_report = memory_checkpoint.get("report", {})
        memory_config = latent_memory_report.get("model_config", {})
        latent_memory_updater = Phase3AEgocentricMemoryUpdate(
            memory_size=int(memory_config.get("memory_size", model.spatial_memory_size)),
            hidden_dim=int(memory_config.get("hidden_dim", 96)),
            memory_channels=int(memory_config.get("memory_channels", 3)),
            evidence_channels=int(memory_config.get("evidence_channels", 3)),
            action_dim=int(memory_config.get("action_dim", len(ACTION_NAMES))),
            use_geometric_prior=bool(memory_config.get("use_geometric_prior", True)),
            learned_transition_hidden_dim=memory_config.get(
                "learned_transition_hidden_dim",
            ),
        ).to(torch.device(args.device))
        latent_memory_updater.load_state_dict(memory_checkpoint["updater_state_dict"])
        latent_memory_updater.eval()
    latent_policy_head = None
    latent_policy_report = None
    if args.latent_policy_head is not None:
        try:
            policy_checkpoint = torch.load(
                args.latent_policy_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            policy_checkpoint = torch.load(
                args.latent_policy_head,
                map_location=torch.device(args.device),
            )
        latent_policy_report = policy_checkpoint.get("report", {})
        policy_config = latent_policy_report.get("model_config", {})
        latent_policy_head = Phase3AEgocentricMemoryPolicy(
            memory_size=int(policy_config.get("memory_size", model.spatial_memory_size)),
            hidden_dim=int(policy_config.get("hidden_dim", 256)),
            memory_channels=int(policy_config.get("memory_channels", 3)),
            action_dim=int(policy_config.get("action_dim", len(ACTION_NAMES))),
            architecture=str(policy_config.get("architecture", "mlp")),
        ).to(torch.device(args.device))
        latent_policy_head.load_state_dict(policy_checkpoint["policy_state_dict"])
        latent_policy_head.eval()
    latent_value_field_head = None
    latent_value_field_report = None
    if args.latent_value_field_head is not None:
        try:
            value_checkpoint = torch.load(
                args.latent_value_field_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            value_checkpoint = torch.load(
                args.latent_value_field_head,
                map_location=torch.device(args.device),
            )
        latent_value_field_report = value_checkpoint.get("report", {})
        value_config = latent_value_field_report.get("model_config", {})
        latent_value_field_head = Phase3AEgocentricValueFieldHead(
            memory_size=int(value_config.get("memory_size", model.spatial_memory_size)),
            hidden_dim=int(value_config.get("hidden_dim", 64)),
            memory_channels=int(value_config.get("memory_channels", 3)),
            output_channels=int(value_config.get("output_channels", 1)),
        ).to(torch.device(args.device))
        latent_value_field_head.load_state_dict(value_checkpoint["head_state_dict"])
        latent_value_field_head.eval()
    latent_value_extractor_head = None
    latent_value_extractor_report = None
    if args.latent_value_extractor_head is not None:
        try:
            extractor_checkpoint = torch.load(
                args.latent_value_extractor_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            extractor_checkpoint = torch.load(
                args.latent_value_extractor_head,
                map_location=torch.device(args.device),
            )
        latent_value_extractor_report = extractor_checkpoint.get("report", {})
        extractor_config = latent_value_extractor_report.get("model_config", {})
        latent_value_extractor_head = Phase3AValueFieldExtractorHead(
            memory_size=int(
                extractor_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(extractor_config.get("hidden_dim", 32)),
            memory_channels=int(extractor_config.get("memory_channels", 3)),
        ).to(torch.device(args.device))
        latent_value_extractor_head.load_state_dict(
            extractor_checkpoint["extractor_state_dict"]
        )
        latent_value_extractor_head.eval()
    latent_value_action_head = None
    latent_value_action_report = None
    if args.latent_value_action_head is not None:
        try:
            action_checkpoint = torch.load(
                args.latent_value_action_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            action_checkpoint = torch.load(
                args.latent_value_action_head,
                map_location=torch.device(args.device),
            )
        latent_value_action_report = action_checkpoint.get("report", {})
        action_config = latent_value_action_report.get("model_config", {})
        latent_value_action_head = Phase3AValueFieldActionHead(
            memory_size=int(action_config.get("memory_size", model.spatial_memory_size)),
            hidden_dim=int(action_config.get("hidden_dim", 64)),
            memory_channels=int(action_config.get("memory_channels", 3)),
            action_dim=int(action_config.get("action_dim", len(ACTION_NAMES))),
        ).to(torch.device(args.device))
        latent_value_action_head.load_state_dict(
            action_checkpoint["action_head_state_dict"]
        )
        latent_value_action_head.eval()
    latent_action_correction_head = None
    latent_action_correction_report = None
    if args.latent_action_correction_head is not None:
        latent_action_correction_head, latent_action_correction_report = (
            _load_action_correction_head(
                args.latent_action_correction_head,
                model=model,
                device=torch.device(args.device),
            )
        )
    latent_pre_marker_action_correction_head = None
    latent_pre_marker_action_correction_report = None
    if args.latent_pre_marker_action_correction_head is not None:
        (
            latent_pre_marker_action_correction_head,
            latent_pre_marker_action_correction_report,
        ) = _load_action_correction_head(
            args.latent_pre_marker_action_correction_head,
            model=model,
            device=torch.device(args.device),
        )
    latent_value_map_planner_head = None
    latent_value_map_planner_report = None
    if args.latent_value_map_planner_head is not None:
        try:
            planner_checkpoint = torch.load(
                args.latent_value_map_planner_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            planner_checkpoint = torch.load(
                args.latent_value_map_planner_head,
                map_location=torch.device(args.device),
            )
        latent_value_map_planner_report = planner_checkpoint.get("report", {})
        planner_config = latent_value_map_planner_report.get("model_config", {})
        latent_value_map_planner_head = Phase3AValueMapPlannerHead(
            memory_size=int(planner_config.get("memory_size", model.spatial_memory_size)),
            hidden_dim=int(planner_config.get("hidden_dim", 96)),
            memory_channels=int(planner_config.get("memory_channels", 3)),
            architecture=str(planner_config.get("architecture", "conv")),
            refinement_steps=int(planner_config.get("refinement_steps", 8)),
        ).to(torch.device(args.device))
        latent_value_map_planner_head.load_state_dict(
            planner_checkpoint["planner_head_state_dict"]
        )
        latent_value_map_planner_head.eval()
    latent_reachability_head = None
    latent_reachability_report = None
    if args.latent_reachability_head is not None:
        try:
            reachability_checkpoint = torch.load(
                args.latent_reachability_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            reachability_checkpoint = torch.load(
                args.latent_reachability_head,
                map_location=torch.device(args.device),
            )
        latent_reachability_report = reachability_checkpoint.get("report", {})
        reachability_config = latent_reachability_report.get("model_config", {})
        latent_reachability_head = Phase3BReachabilityHead(
            memory_size=int(
                reachability_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(reachability_config.get("hidden_dim", 96)),
            memory_channels=int(reachability_config.get("memory_channels", 3)),
            architecture=str(reachability_config.get("architecture", "conv")),
        ).to(torch.device(args.device))
        latent_reachability_head.load_state_dict(
            reachability_checkpoint["head_state_dict"]
        )
        latent_reachability_head.eval()
    latent_reachability_value_map_planner_head = None
    latent_reachability_value_map_planner_report = None
    if args.latent_reachability_value_map_planner_head is not None:
        try:
            reachability_planner_checkpoint = torch.load(
                args.latent_reachability_value_map_planner_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            reachability_planner_checkpoint = torch.load(
                args.latent_reachability_value_map_planner_head,
                map_location=torch.device(args.device),
            )
        latent_reachability_value_map_planner_report = (
            reachability_planner_checkpoint.get("report", {})
        )
        reachability_planner_config = (
            latent_reachability_value_map_planner_report.get("model_config", {})
        )
        latent_reachability_value_map_planner_head = (
            Phase3BReachabilityConditionedValueMapPlannerHead(
                memory_size=int(
                    reachability_planner_config.get(
                        "memory_size",
                        model.spatial_memory_size,
                    )
                ),
                hidden_dim=int(reachability_planner_config.get("hidden_dim", 96)),
                memory_channels=int(
                    reachability_planner_config.get("memory_channels", 3)
                ),
                reachability_channels=int(
                    reachability_planner_config.get("reachability_channels", 4)
                ),
                architecture=str(
                    reachability_planner_config.get("architecture", "conv")
                ),
            ).to(torch.device(args.device))
        )
        latent_reachability_value_map_planner_head.load_state_dict(
            reachability_planner_checkpoint["planner_head_state_dict"]
        )
        latent_reachability_value_map_planner_head.eval()
    latent_marker_value_map_planner_head = None
    latent_marker_value_map_planner_report = None
    if args.latent_marker_value_map_planner_head is not None:
        try:
            marker_planner_checkpoint = torch.load(
                args.latent_marker_value_map_planner_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            marker_planner_checkpoint = torch.load(
                args.latent_marker_value_map_planner_head,
                map_location=torch.device(args.device),
            )
        latent_marker_value_map_planner_report = marker_planner_checkpoint.get(
            "report",
            {},
        )
        marker_planner_config = latent_marker_value_map_planner_report.get(
            "model_config",
            {},
        )
        latent_marker_value_map_planner_head = Phase3AValueMapPlannerHead(
            memory_size=int(
                marker_planner_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(marker_planner_config.get("hidden_dim", 96)),
            memory_channels=int(marker_planner_config.get("memory_channels", 3)),
            architecture=str(marker_planner_config.get("architecture", "conv")),
            refinement_steps=int(marker_planner_config.get("refinement_steps", 8)),
        ).to(torch.device(args.device))
        latent_marker_value_map_planner_head.load_state_dict(
            marker_planner_checkpoint["planner_head_state_dict"]
        )
        latent_marker_value_map_planner_head.eval()
    latent_value_map_ensemble_heads = []
    latent_value_map_ensemble_reports = []
    for ensemble_path in args.latent_value_map_ensemble_head:
        try:
            ensemble_checkpoint = torch.load(
                ensemble_path,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            ensemble_checkpoint = torch.load(
                ensemble_path,
                map_location=torch.device(args.device),
            )
        ensemble_report = ensemble_checkpoint.get("report", {})
        ensemble_config = ensemble_report.get("model_config", {})
        ensemble_head = Phase3AValueMapPlannerHead(
            memory_size=int(
                ensemble_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(ensemble_config.get("hidden_dim", 96)),
            memory_channels=int(ensemble_config.get("memory_channels", 3)),
            architecture=str(ensemble_config.get("architecture", "conv")),
            refinement_steps=int(ensemble_config.get("refinement_steps", 8)),
        ).to(torch.device(args.device))
        ensemble_head.load_state_dict(ensemble_checkpoint["planner_head_state_dict"])
        ensemble_head.eval()
        latent_value_map_ensemble_heads.append(ensemble_head)
        latent_value_map_ensemble_reports.append(ensemble_report)
    latent_value_map_fallback_head = None
    latent_value_map_fallback_report = None
    if args.latent_value_map_fallback_head is not None:
        try:
            fallback_checkpoint = torch.load(
                args.latent_value_map_fallback_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            fallback_checkpoint = torch.load(
                args.latent_value_map_fallback_head,
                map_location=torch.device(args.device),
            )
        latent_value_map_fallback_report = fallback_checkpoint.get("report", {})
        fallback_config = latent_value_map_fallback_report.get("model_config", {})
        latent_value_map_fallback_head = Phase3AValueMapPlannerHead(
            memory_size=int(
                fallback_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(fallback_config.get("hidden_dim", 96)),
            memory_channels=int(fallback_config.get("memory_channels", 3)),
            architecture=str(fallback_config.get("architecture", "conv")),
            refinement_steps=int(fallback_config.get("refinement_steps", 8)),
        ).to(torch.device(args.device))
        latent_value_map_fallback_head.load_state_dict(
            fallback_checkpoint["planner_head_state_dict"]
        )
        latent_value_map_fallback_head.eval()
    latent_value_map_fallback_ensemble_heads = []
    latent_value_map_fallback_ensemble_reports = []
    for fallback_ensemble_path in args.latent_value_map_fallback_ensemble_head:
        try:
            fallback_ensemble_checkpoint = torch.load(
                fallback_ensemble_path,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            fallback_ensemble_checkpoint = torch.load(
                fallback_ensemble_path,
                map_location=torch.device(args.device),
            )
        fallback_ensemble_report = fallback_ensemble_checkpoint.get("report", {})
        fallback_ensemble_config = fallback_ensemble_report.get("model_config", {})
        fallback_ensemble_head = Phase3AValueMapPlannerHead(
            memory_size=int(
                fallback_ensemble_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(fallback_ensemble_config.get("hidden_dim", 96)),
            memory_channels=int(fallback_ensemble_config.get("memory_channels", 3)),
            architecture=str(fallback_ensemble_config.get("architecture", "conv")),
            refinement_steps=int(fallback_ensemble_config.get("refinement_steps", 8)),
        ).to(torch.device(args.device))
        fallback_ensemble_head.load_state_dict(
            fallback_ensemble_checkpoint["planner_head_state_dict"]
        )
        fallback_ensemble_head.eval()
        latent_value_map_fallback_ensemble_heads.append(fallback_ensemble_head)
        latent_value_map_fallback_ensemble_reports.append(fallback_ensemble_report)
    latent_value_map_router_head = None
    latent_value_map_router_report = None
    if args.latent_value_map_router_head is not None:
        try:
            router_checkpoint = torch.load(
                args.latent_value_map_router_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            router_checkpoint = torch.load(
                args.latent_value_map_router_head,
                map_location=torch.device(args.device),
            )
        latent_value_map_router_report = router_checkpoint.get("report", {})
        router_config = latent_value_map_router_report.get("model_config", {})
        latent_value_map_router_head = Phase3AValueMapRouterHead(
            memory_size=int(
                router_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(router_config.get("hidden_dim", 32)),
            memory_channels=int(router_config.get("memory_channels", 3)),
        ).to(torch.device(args.device))
        latent_value_map_router_head.load_state_dict(
            router_checkpoint["router_head_state_dict"]
        )
        latent_value_map_router_head.eval()
    latent_soft_value_router_head = None
    latent_soft_value_router_report = None
    if args.latent_soft_value_router_head is not None:
        try:
            soft_router_checkpoint = torch.load(
                args.latent_soft_value_router_head,
                map_location=torch.device(args.device),
                weights_only=False,
            )
        except TypeError:
            soft_router_checkpoint = torch.load(
                args.latent_soft_value_router_head,
                map_location=torch.device(args.device),
            )
        latent_soft_value_router_report = soft_router_checkpoint.get("report", {})
        soft_router_config = latent_soft_value_router_report.get("model_config", {})
        latent_soft_value_router_head = Phase3AValueMapRouterHead(
            memory_size=int(
                soft_router_config.get("memory_size", model.spatial_memory_size)
            ),
            hidden_dim=int(soft_router_config.get("hidden_dim", 32)),
            memory_channels=int(soft_router_config.get("memory_channels", 3)),
        ).to(torch.device(args.device))
        latent_soft_value_router_head.load_state_dict(
            soft_router_checkpoint["router_head_state_dict"]
        )
        latent_soft_value_router_head.eval()
    if args.spatial_frontier_marker_update_threshold is not None:
        model.spatial_frontier_marker_update_threshold = float(
            args.spatial_frontier_marker_update_threshold
        )
    if args.spatial_frontier_marker_update_width is not None:
        model.spatial_frontier_marker_update_width = float(
            args.spatial_frontier_marker_update_width
        )
    if args.spatial_marker_memory_score_temperature is not None:
        if args.spatial_marker_memory_score_temperature <= 0.0:
            raise SystemExit("--spatial-marker-memory-score-temperature must be positive")
        model.spatial_marker_memory_score_temperature = float(
            args.spatial_marker_memory_score_temperature
        )
    if args.exact_online_memory_size is not None:
        if args.exact_online_memory_size < args.view_size:
            raise SystemExit("--exact-online-memory-size must be >= --view-size")
        if args.exact_online_memory_size % 2 == 0:
            raise SystemExit("--exact-online-memory-size must be odd")
        model.spatial_memory_size = int(args.exact_online_memory_size)
    elif (
        args.score_source in (
            "latent_recurrent_egocentric_frontier_planner",
            "latent_recurrent_policy_planner",
            "latent_recurrent_value_field_planner",
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and latent_memory_updater is not None
    ):
        model.spatial_memory_size = int(latent_memory_updater.memory_size)
    if (
        args.score_source in (
            "latent_recurrent_egocentric_frontier_planner",
            "latent_recurrent_policy_planner",
            "latent_recurrent_value_field_planner",
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and latent_memory_updater is not None
        and int(model.spatial_memory_size) != int(latent_memory_updater.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the recurrent memory "
            f"checkpoint size ({latent_memory_updater.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_policy_planner"
        and latent_policy_head is not None
        and int(model.spatial_memory_size) != int(latent_policy_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent policy "
            f"checkpoint size ({latent_policy_head.memory_size})"
        )
    if (
        args.score_source
        in (
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_reachability_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and latent_value_field_head is not None
        and int(model.spatial_memory_size) != int(latent_value_field_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent value-field "
            f"checkpoint size ({latent_value_field_head.memory_size})"
        )
    if (
        args.score_source
        in (
            "latent_recurrent_learned_value_field_planner",
            "latent_recurrent_learned_value_action_planner",
            "latent_recurrent_learned_value_map_planner",
            "latent_recurrent_soft_value_map_planner",
        )
        and latent_value_extractor_head is not None
        and int(model.spatial_memory_size)
        != int(latent_value_extractor_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent value extractor "
            f"checkpoint size ({latent_value_extractor_head.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_action_planner"
        and latent_value_action_head is not None
        and int(model.spatial_memory_size) != int(latent_value_action_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent value action "
            f"checkpoint size ({latent_value_action_head.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_map_planner"
        and latent_value_map_planner_head is not None
        and int(model.spatial_memory_size)
        != int(latent_value_map_planner_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent value-map planner "
            f"checkpoint size ({latent_value_map_planner_head.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_reachability_value_map_planner"
        and latent_reachability_head is not None
        and int(model.spatial_memory_size) != int(latent_reachability_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent reachability "
            f"checkpoint size ({latent_reachability_head.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_reachability_value_map_planner"
        and latent_reachability_value_map_planner_head is not None
        and int(model.spatial_memory_size)
        != int(latent_reachability_value_map_planner_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent reachability "
            "value-map planner checkpoint size "
            f"({latent_reachability_value_map_planner_head.memory_size})"
        )
    if (
        args.score_source == "latent_recurrent_learned_value_map_planner"
        and latent_marker_value_map_planner_head is not None
        and int(model.spatial_memory_size)
        != int(latent_marker_value_map_planner_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent marker value-map "
            f"planner checkpoint size ({latent_marker_value_map_planner_head.memory_size})"
        )
    if args.score_source == "latent_recurrent_learned_value_map_planner":
        for ensemble_head in latent_value_map_ensemble_heads:
            if int(model.spatial_memory_size) != int(ensemble_head.memory_size):
                raise SystemExit(
                    "--exact-online-memory-size must match every latent value-map "
                    f"ensemble checkpoint size ({ensemble_head.memory_size})"
                )
        for fallback_ensemble_head in latent_value_map_fallback_ensemble_heads:
            if int(model.spatial_memory_size) != int(
                fallback_ensemble_head.memory_size
            ):
                raise SystemExit(
                    "--exact-online-memory-size must match every latent value-map "
                    "fallback ensemble checkpoint size "
                    f"({fallback_ensemble_head.memory_size})"
                )
        if (
            latent_value_map_fallback_head is not None
            and int(model.spatial_memory_size)
            != int(latent_value_map_fallback_head.memory_size)
        ):
            raise SystemExit(
                "--exact-online-memory-size must match the latent value-map "
                f"fallback checkpoint size ({latent_value_map_fallback_head.memory_size})"
            )
        if (
            latent_value_map_router_head is not None
            and int(model.spatial_memory_size)
            != int(latent_value_map_router_head.memory_size)
        ):
            raise SystemExit(
                "--exact-online-memory-size must match the latent value-map "
                f"router checkpoint size ({latent_value_map_router_head.memory_size})"
            )
        if (
            latent_soft_value_router_head is not None
            and int(model.spatial_memory_size)
            != int(latent_soft_value_router_head.memory_size)
        ):
            raise SystemExit(
                "--exact-online-memory-size must match the latent soft-value "
                f"router checkpoint size ({latent_soft_value_router_head.memory_size})"
            )
    if (
        latent_action_correction_head is not None
        and int(model.spatial_memory_size)
        != int(latent_action_correction_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent action correction "
            f"checkpoint size ({latent_action_correction_head.memory_size})"
        )
    if (
        latent_pre_marker_action_correction_head is not None
        and int(model.spatial_memory_size)
        != int(latent_pre_marker_action_correction_head.memory_size)
    ):
        raise SystemExit(
            "--exact-online-memory-size must match the latent pre-marker action "
            "correction checkpoint size "
            f"({latent_pre_marker_action_correction_head.memory_size})"
        )
    groups = _group_validation_sources(rows)
    episodes = []
    best = None
    selected_episode = None
    for episode_index, group in enumerate(groups[: args.max_episodes]):
        if (
            args.selected_episode_index is not None
            and episode_index != args.selected_episode_index
        ):
            continue
        episode = _run_episode(
            model,
            group,
            seed=scene_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            horizon=args.horizon,
            max_steps=args.max_steps,
            execute_block_steps=min(args.execute_block_steps, args.horizon),
            history_window=args.history_window,
            max_candidates_per_state=args.max_candidates_per_state,
            score_source=args.score_source,
            odom_frontier_strategy=args.odom_frontier_strategy,
            odom_frontier_gain_radius=args.odom_frontier_gain_radius,
            odom_frontier_distance_penalty=args.odom_frontier_distance_penalty,
            odom_frontier_turn_penalty=args.odom_frontier_turn_penalty,
            odom_frontier_neighbor_weight=args.odom_frontier_neighbor_weight,
            odom_frontier_lookahead_horizon=args.odom_frontier_lookahead_horizon,
            odom_frontier_lookahead_beam_width=(
                args.odom_frontier_lookahead_beam_width
            ),
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            latent_policy_head=latent_policy_head,
            latent_value_field_head=latent_value_field_head,
            latent_value_extractor_head=latent_value_extractor_head,
            latent_value_action_head=latent_value_action_head,
            latent_reachability_head=latent_reachability_head,
            latent_reachability_value_map_planner_head=(
                latent_reachability_value_map_planner_head
            ),
            latent_pre_marker_action_correction_head=(
                latent_pre_marker_action_correction_head
            ),
            latent_pre_marker_action_correction_threshold=(
                args.latent_pre_marker_action_correction_threshold
            ),
            latent_pre_marker_action_correction_initial_threshold=(
                args.latent_pre_marker_action_correction_initial_threshold
            ),
            latent_pre_marker_action_correction_initial_max_step=(
                args.latent_pre_marker_action_correction_initial_max_step
            ),
            latent_pre_marker_action_correction_min_step=(
                args.latent_pre_marker_action_correction_min_step
            ),
            latent_pre_marker_action_correction_max_step=(
                args.latent_pre_marker_action_correction_max_step
            ),
            latent_action_correction_head=latent_action_correction_head,
            latent_action_correction_threshold=args.latent_action_correction_threshold,
            latent_action_correction_mode=args.latent_action_correction_mode,
            latent_action_correction_min_step=args.latent_action_correction_min_step,
            latent_action_correction_max_step=args.latent_action_correction_max_step,
            latent_value_map_planner_head=latent_value_map_planner_head,
            latent_marker_value_map_planner_head=latent_marker_value_map_planner_head,
            latent_value_map_ensemble_heads=tuple(latent_value_map_ensemble_heads),
            latent_value_map_ensemble_mode=args.latent_value_map_ensemble_mode,
            latent_value_map_readout=args.latent_value_map_readout,
            latent_value_map_fallback_head=latent_value_map_fallback_head,
            latent_value_map_fallback_ensemble_heads=tuple(
                latent_value_map_fallback_ensemble_heads
            ),
            latent_value_map_fallback_after_step=(
                args.latent_value_map_fallback_after_step
            ),
            latent_value_map_router_head=latent_value_map_router_head,
            latent_value_map_router_threshold=args.latent_value_map_router_threshold,
            latent_soft_value_router_head=latent_soft_value_router_head,
            latent_soft_value_router_threshold=args.latent_soft_value_router_threshold,
            latent_soft_value_router_mode=args.latent_soft_value_router_mode,
            latent_value_map_side_wall_fallback=bool(
                args.latent_value_map_side_wall_fallback
            ),
            latent_value_map_fixed_marker_return=bool(
                args.latent_value_map_fixed_marker_return
            ),
            latent_value_map_marker_action_return=bool(
                args.latent_value_map_marker_action_return
            ),
            latent_value_map_current_marker_action_return=bool(
                args.latent_value_map_current_marker_action_return
            ),
            latent_value_map_current_marker_action_threshold=(
                args.latent_value_map_current_marker_action_threshold
            ),
            latent_value_map_current_marker_local_threshold=(
                args.latent_value_map_current_marker_local_threshold
            ),
            latent_value_map_marker_action_require_local_evidence=bool(
                args.latent_value_map_marker_action_require_local_evidence
            ),
            latent_value_map_turn_oscillation_breaker=bool(
                args.latent_value_map_turn_oscillation_breaker
            ),
            latent_value_map_state_loop_breaker=bool(
                args.latent_value_map_state_loop_breaker
            ),
            latent_map_blocked_threshold=args.latent_map_blocked_threshold,
            latent_map_marker_threshold=args.latent_map_marker_threshold,
            latent_memory_blocked_threshold=args.latent_memory_blocked_threshold,
            latent_memory_free_threshold=args.latent_memory_free_threshold,
            latent_memory_marker_threshold=args.latent_memory_marker_threshold,
            latent_memory_merge_current_marker_evidence=bool(
                args.latent_memory_merge_current_marker_evidence
            ),
            latent_value_target_threshold=args.latent_value_target_threshold,
            latent_value_target_top_k=args.latent_value_target_top_k,
            latent_value_marker_target_threshold=(
                args.latent_value_marker_target_threshold
            ),
            latent_value_marker_target_top_k=args.latent_value_marker_target_top_k,
            latent_value_extractor_threshold=args.latent_value_extractor_threshold,
            latent_value_sparse_target_top_k=args.latent_value_sparse_target_top_k,
            latent_value_action_fallback_threshold=(
                args.latent_value_action_fallback_threshold
            ),
            latent_soft_value_iterations=args.latent_soft_value_iterations,
            latent_soft_value_gamma=args.latent_soft_value_gamma,
            latent_value_fixed_marker_target=bool(
                args.latent_value_fixed_marker_target
            ),
            persistent_marker_claim_threshold=(
                args.persistent_marker_claim_threshold
            ),
            persistent_marker_require_seen=args.persistent_marker_require_seen,
            current_goal_marker=not args.hide_current_marker,
            device=torch.device(args.device),
        )
        episode_record = {
            key: value
            for key, value in episode.items()
            if key != "scene"
        }
        episode_record["source_episode_index"] = episode_index
        episodes.append(episode_record)
        if (
            args.progress_every_episodes > 0
            and len(episodes) % args.progress_every_episodes == 0
        ):
            claimed_so_far = sum(1 for item in episodes if item["claimed"])
            print(
                json.dumps(
                    {
                        "attempted": len(episodes),
                        "claimed": claimed_so_far,
                        "episode_index": int(episode_index),
                        "last_claimed": bool(episode["claimed"]),
                        "last_steps": int(episode["steps"]),
                        "score_source": args.score_source,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.selected_episode_index == episode_index:
            selected_episode = episode
        if best is None:
            best = episode
        elif episode["claimed"] and not best["claimed"]:
            best = episode
        elif episode["claimed"] == best["claimed"]:
            best_distance = GridScene.distance_to_goal(
                best["scene"],
                best["final_state"]["x"],
                best["final_state"]["y"],
            )
            distance = scene_distance = episode["scene"].distance_to_goal(
                episode["final_state"]["x"],
                episode["final_state"]["y"],
            )
            if distance is not None and (
                best_distance is None or distance < best_distance
            ):
                best = episode
    assert best is not None
    if args.selected_episode_index is not None:
        if selected_episode is None:
            raise SystemExit("--selected-episode-index exceeds attempted episodes")
        best = selected_episode
    if not args.skip_video:
        _export_episode_mp4(
            best,
            args.output,
            fps=args.fps,
            seconds_per_step=args.seconds_per_step,
            width=args.video_width,
            height=args.video_height,
            view_size=args.view_size,
            current_goal_marker=not args.hide_current_marker,
        )
    report_data = {
        "schema": "jepa_phase3a_closed_loop_demo_report_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_completed_steps": report.get("completed_steps"),
        "validation_data": str(args.validation_data.resolve()),
        "scene_seed": scene_seed,
        "score_source": args.score_source,
        "odom_frontier_strategy": args.odom_frontier_strategy,
        "odom_frontier_gain_radius": args.odom_frontier_gain_radius,
        "odom_frontier_distance_penalty": args.odom_frontier_distance_penalty,
        "odom_frontier_turn_penalty": args.odom_frontier_turn_penalty,
        "odom_frontier_neighbor_weight": args.odom_frontier_neighbor_weight,
        "odom_frontier_lookahead_horizon": args.odom_frontier_lookahead_horizon,
        "odom_frontier_lookahead_beam_width": (
            args.odom_frontier_lookahead_beam_width
        ),
        "latent_map_head": (
            str(args.latent_map_head.resolve()) if args.latent_map_head else None
        ),
        "latent_map_completed_steps": (
            latent_map_report.get("completed_steps") if latent_map_report else None
        ),
        "latent_memory_updater": (
            str(args.latent_memory_updater.resolve())
            if args.latent_memory_updater
            else None
        ),
        "latent_memory_completed_steps": (
            latent_memory_report.get("completed_steps")
            if latent_memory_report
            else None
        ),
        "latent_policy_head": (
            str(args.latent_policy_head.resolve()) if args.latent_policy_head else None
        ),
        "latent_policy_completed_steps": (
            latent_policy_report.get("completed_steps")
            if latent_policy_report
            else None
        ),
        "latent_value_field_head": (
            str(args.latent_value_field_head.resolve())
            if args.latent_value_field_head
            else None
        ),
        "latent_value_field_completed_steps": (
            latent_value_field_report.get("completed_steps")
            if latent_value_field_report
            else None
        ),
        "latent_value_extractor_head": (
            str(args.latent_value_extractor_head.resolve())
            if args.latent_value_extractor_head
            else None
        ),
        "latent_value_extractor_completed_steps": (
            latent_value_extractor_report.get("completed_steps")
            if latent_value_extractor_report
            else None
        ),
        "latent_value_action_head": (
            str(args.latent_value_action_head.resolve())
            if args.latent_value_action_head
            else None
        ),
        "latent_value_action_completed_steps": (
            latent_value_action_report.get("completed_steps")
            if latent_value_action_report
            else None
        ),
        "latent_action_correction_head": (
            str(args.latent_action_correction_head.resolve())
            if args.latent_action_correction_head
            else None
        ),
        "latent_pre_marker_action_correction_head": (
            str(args.latent_pre_marker_action_correction_head.resolve())
            if args.latent_pre_marker_action_correction_head
            else None
        ),
        "latent_action_correction_completed_steps": (
            latent_action_correction_report.get("completed_steps")
            if latent_action_correction_report
            else None
        ),
        "latent_pre_marker_action_correction_completed_steps": (
            latent_pre_marker_action_correction_report.get("completed_steps")
            if latent_pre_marker_action_correction_report
            else None
        ),
        "latent_action_correction_selected_step": (
            latent_action_correction_report.get("selected_step")
            if latent_action_correction_report
            else None
        ),
        "latent_pre_marker_action_correction_selected_step": (
            latent_pre_marker_action_correction_report.get("selected_step")
            if latent_pre_marker_action_correction_report
            else None
        ),
        "latent_action_correction_selected_validation": (
            latent_action_correction_report.get("selected_validation")
            if latent_action_correction_report
            else None
        ),
        "latent_pre_marker_action_correction_selected_validation": (
            latent_pre_marker_action_correction_report.get("selected_validation")
            if latent_pre_marker_action_correction_report
            else None
        ),
        "latent_action_correction_threshold": (
            args.latent_action_correction_threshold
        ),
        "latent_pre_marker_action_correction_threshold": (
            args.latent_pre_marker_action_correction_threshold
        ),
        "latent_pre_marker_action_correction_initial_threshold": (
            args.latent_pre_marker_action_correction_initial_threshold
        ),
        "latent_pre_marker_action_correction_initial_max_step": (
            args.latent_pre_marker_action_correction_initial_max_step
        ),
        "latent_pre_marker_action_correction_min_step": (
            args.latent_pre_marker_action_correction_min_step
        ),
        "latent_pre_marker_action_correction_max_step": (
            args.latent_pre_marker_action_correction_max_step
        ),
        "latent_action_correction_mode": args.latent_action_correction_mode,
        "latent_action_correction_min_step": (
            args.latent_action_correction_min_step
        ),
        "latent_action_correction_max_step": (
            args.latent_action_correction_max_step
        ),
        "latent_value_map_planner_head": (
            str(args.latent_value_map_planner_head.resolve())
            if args.latent_value_map_planner_head
            else None
        ),
        "latent_value_map_planner_completed_steps": (
            latent_value_map_planner_report.get("completed_steps")
            if latent_value_map_planner_report
            else None
        ),
        "latent_reachability_head": (
            str(args.latent_reachability_head.resolve())
            if args.latent_reachability_head
            else None
        ),
        "latent_reachability_completed_steps": (
            latent_reachability_report.get("completed_steps")
            if latent_reachability_report
            else None
        ),
        "latent_reachability_value_map_planner_head": (
            str(args.latent_reachability_value_map_planner_head.resolve())
            if args.latent_reachability_value_map_planner_head
            else None
        ),
        "latent_reachability_value_map_planner_completed_steps": (
            latent_reachability_value_map_planner_report.get("completed_steps")
            if latent_reachability_value_map_planner_report
            else None
        ),
        "latent_marker_value_map_planner_head": (
            str(args.latent_marker_value_map_planner_head.resolve())
            if args.latent_marker_value_map_planner_head
            else None
        ),
        "latent_value_map_ensemble_heads": [
            str(path.resolve()) for path in args.latent_value_map_ensemble_head
        ],
        "latent_value_map_ensemble_mode": args.latent_value_map_ensemble_mode,
        "latent_value_map_readout": args.latent_value_map_readout,
        "latent_value_map_ensemble_completed_steps": [
            report.get("completed_steps")
            for report in latent_value_map_ensemble_reports
        ],
        "latent_value_map_fallback_head": (
            str(args.latent_value_map_fallback_head.resolve())
            if args.latent_value_map_fallback_head
            else None
        ),
        "latent_value_map_fallback_ensemble_heads": [
            str(path.resolve())
            for path in args.latent_value_map_fallback_ensemble_head
        ],
        "latent_value_map_fallback_after_step": (
            args.latent_value_map_fallback_after_step
        ),
        "latent_value_map_router_head": (
            str(args.latent_value_map_router_head.resolve())
            if args.latent_value_map_router_head
            else None
        ),
        "latent_value_map_router_threshold": args.latent_value_map_router_threshold,
        "latent_value_map_router_selected_step": (
            latent_value_map_router_report.get("selected_step")
            if latent_value_map_router_report
            else None
        ),
        "latent_value_map_router_label_source": (
            latent_value_map_router_report.get("router_label_source")
            if latent_value_map_router_report
            else None
        ),
        "latent_value_map_router_selected_validation": (
            latent_value_map_router_report.get("selected_validation")
            if latent_value_map_router_report
            else None
        ),
        "latent_soft_value_router_head": (
            str(args.latent_soft_value_router_head.resolve())
            if args.latent_soft_value_router_head
            else None
        ),
        "latent_soft_value_router_threshold": args.latent_soft_value_router_threshold,
        "latent_soft_value_router_mode": args.latent_soft_value_router_mode,
        "latent_soft_value_router_selected_step": (
            latent_soft_value_router_report.get("selected_step")
            if latent_soft_value_router_report
            else None
        ),
        "latent_soft_value_router_label_source": (
            latent_soft_value_router_report.get("router_label_source")
            if latent_soft_value_router_report
            else None
        ),
        "latent_soft_value_router_selected_validation": (
            latent_soft_value_router_report.get("selected_validation")
            if latent_soft_value_router_report
            else None
        ),
        "latent_value_map_side_wall_fallback": bool(
            args.latent_value_map_side_wall_fallback
        ),
        "latent_value_map_fallback_completed_steps": (
            latent_value_map_fallback_report.get("completed_steps")
            if latent_value_map_fallback_report
            else None
        ),
        "latent_value_map_fallback_ensemble_completed_steps": [
            report.get("completed_steps")
            for report in latent_value_map_fallback_ensemble_reports
        ],
        "latent_value_map_fixed_marker_return": bool(
            args.latent_value_map_fixed_marker_return
        ),
        "latent_value_map_marker_action_return": bool(
            args.latent_value_map_marker_action_return
        ),
        "latent_value_map_current_marker_action_return": bool(
            args.latent_value_map_current_marker_action_return
        ),
        "latent_value_map_current_marker_action_threshold": (
            args.latent_value_map_current_marker_action_threshold
        ),
        "latent_value_map_current_marker_local_threshold": (
            args.latent_value_map_current_marker_local_threshold
        ),
        "latent_value_map_marker_action_require_local_evidence": bool(
            args.latent_value_map_marker_action_require_local_evidence
        ),
        "latent_value_map_turn_oscillation_breaker": bool(
            args.latent_value_map_turn_oscillation_breaker
        ),
        "latent_value_map_state_loop_breaker": bool(
            args.latent_value_map_state_loop_breaker
        ),
        "latent_marker_value_map_planner_completed_steps": (
            latent_marker_value_map_planner_report.get("completed_steps")
            if latent_marker_value_map_planner_report
            else None
        ),
        "latent_map_blocked_threshold": args.latent_map_blocked_threshold,
        "latent_map_marker_threshold": args.latent_map_marker_threshold,
        "latent_memory_blocked_threshold": args.latent_memory_blocked_threshold,
        "latent_memory_free_threshold": args.latent_memory_free_threshold,
        "latent_memory_marker_threshold": args.latent_memory_marker_threshold,
        "latent_memory_merge_current_marker_evidence": bool(
            args.latent_memory_merge_current_marker_evidence
        ),
        "latent_value_target_threshold": args.latent_value_target_threshold,
        "latent_value_target_top_k": args.latent_value_target_top_k,
        "latent_value_marker_target_threshold": (
            args.latent_value_marker_target_threshold
        ),
        "latent_value_marker_target_top_k": args.latent_value_marker_target_top_k,
        "latent_value_extractor_threshold": args.latent_value_extractor_threshold,
        "latent_value_sparse_target_top_k": args.latent_value_sparse_target_top_k,
        "latent_value_action_fallback_threshold": (
            args.latent_value_action_fallback_threshold
        ),
        "latent_soft_value_iterations": args.latent_soft_value_iterations,
        "latent_soft_value_gamma": args.latent_soft_value_gamma,
        "latent_value_fixed_marker_target": bool(
            args.latent_value_fixed_marker_target
        ),
        "history_window": args.history_window,
        "max_candidates_per_state": args.max_candidates_per_state,
        "selected_episode_index": args.selected_episode_index,
        "persistent_marker_claim_threshold": (
            args.persistent_marker_claim_threshold
        ),
        "persistent_marker_require_seen": bool(args.persistent_marker_require_seen),
        "spatial_frontier_marker_update_threshold": (
            model.spatial_frontier_marker_update_threshold
        ),
        "spatial_frontier_marker_update_width": (
            model.spatial_frontier_marker_update_width
        ),
        "spatial_marker_memory_score_temperature": (
            model.spatial_marker_memory_score_temperature
        ),
        "spatial_memory_size": model.spatial_memory_size,
        "current_goal_marker_visible": not args.hide_current_marker,
        "episodes_attempted": len(episodes),
        "claimed_episodes": sum(1 for item in episodes if item["claimed"]),
        "episode_summaries": [
            {
                "episode_index": int(item.get("source_episode_index", index)),
                "claimed": bool(item["claimed"]),
                "marker_seen_ever": bool(item["marker_seen_ever"]),
                "latent_marker_seen_ever": bool(item["latent_marker_seen_ever"]),
                "steps": int(item["steps"]),
                "final_goal_distance": (
                    abs(int(item["goal"]["x"]) - int(item["final_state"]["x"]))
                    + abs(int(item["goal"]["y"]) - int(item["final_state"]["y"]))
                ),
                "final_state": {
                    "x": int(item["final_state"]["x"]),
                    "y": int(item["final_state"]["y"]),
                    "yaw": int(item["final_state"]["yaw"]),
                },
                "goal": {
                    "x": int(item["goal"]["x"]),
                    "y": int(item["goal"]["y"]),
                },
                "collision_steps": sum(
                    1 for step in item["trajectory"] if bool(step["collision"])
                ),
                "selection_mode_counts": _episode_mode_counts(item),
            }
            for index, item in enumerate(episodes)
        ],
        "selected_episode": {
            key: value for key, value in best.items() if key != "scene"
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n")
    if args.trace_output is not None:
        trace_data = {
            "schema": "jepa_phase3a_closed_loop_trace_v0",
            "report_output": str(args.report_output.resolve()),
            "episodes": episodes,
        }
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(
            json.dumps(trace_data, indent=2, sort_keys=True) + "\n"
        )
    if args.skip_video:
        print(
            f"wrote {args.report_output}; "
            f"claimed {report_data['claimed_episodes']}/{len(episodes)} episodes"
        )
    else:
        print(
            f"wrote {args.output} and {args.report_output}; "
            f"claimed {report_data['claimed_episodes']}/{len(episodes)} episodes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
