"""Phase 3A foundational JEPA positive-control navigation world.

The generator deliberately stays small and deterministic. It creates a pixel
observation task with privileged labels and same-source counterfactual action
branches, so JEPA objective changes can be tested before returning to Go2.
"""
from __future__ import annotations

import json
import math
import random
import colorsys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

PHASE3A_ROW_SCHEMA = "jepa_phase3a_positive_control_row_v2"
PHASE3A_AUDIT_SCHEMA = "jepa_phase3a_positive_control_audit_v1"

ACTION_NAMES = ("forward", "turn_left", "turn_right", "hold")
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
YAW_TO_VEC = ((0, -1), (1, 0), (0, 1), (-1, 0))
HISTORY_POLICIES = ("turning", "explore")
UTILITY_MODES = ("goal_progress", "explore_then_claim", "novelty_then_claim")
COLOR_PALETTE_MODES = ("fixed", "scene_random")
DEFAULT_RENDER_PALETTE = {
    "outside": (0.05, 0.05, 0.06),
    "wall": (0.18, 0.18, 0.18),
    "free": (0.72, 0.72, 0.66),
    "goal": (0.10, 0.85, 0.18),
    "agent": (0.10, 0.20, 0.90),
}


@dataclass(frozen=True)
class GridState:
    """Discrete agent state in the positive-control world."""

    x: int
    y: int
    yaw: int


@dataclass(frozen=True)
class GridScene:
    """One generated 2D navigation scene."""

    scene_id: str
    family: str
    grid: tuple[str, ...]
    goal: tuple[int, int]
    distances: tuple[tuple[int | None, ...], ...]
    render_palette: dict[str, tuple[float, float, float]] | None = None

    @property
    def width(self) -> int:
        return len(self.grid[0])

    @property
    def height(self) -> int:
        return len(self.grid)

    def is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != "#"

    def distance_to_goal(self, x: int, y: int) -> int | None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return self.distances[y][x]


def action_vector(name: str) -> tuple[float, ...]:
    """Return a stable one-hot action vector."""

    try:
        index = ACTION_INDEX[name]
    except KeyError as error:
        raise ValueError(f"unknown Phase 3A action: {name}") from error
    return tuple(1.0 if item == index else 0.0 for item in range(len(ACTION_NAMES)))


def _rgb_from_hsv(
    hue: float,
    saturation: float,
    value: float,
) -> tuple[float, float, float]:
    return tuple(float(channel) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def render_palette_for_scene(
    *,
    mode: str,
    seed: int,
    scene_index: int,
) -> dict[str, tuple[float, float, float]]:
    """Return a deterministic rendering palette for one scene."""

    if mode not in COLOR_PALETTE_MODES:
        raise ValueError(f"mode must be one of {COLOR_PALETTE_MODES}")
    if mode == "fixed":
        return dict(DEFAULT_RENDER_PALETTE)
    rng = random.Random(seed + scene_index * 1000003 + 9176)
    goal_hue = rng.choice((0.00, 0.08, 0.58, 0.72, 0.86)) + rng.uniform(-0.025, 0.025)
    floor_hue = rng.random()
    wall_hue = (floor_hue + 0.32 + rng.uniform(-0.10, 0.10)) % 1.0
    agent_hue = (goal_hue + 0.45) % 1.0
    return {
        "outside": _rgb_from_hsv((wall_hue + 0.08) % 1.0, 0.35, 0.05),
        "wall": _rgb_from_hsv(wall_hue, rng.uniform(0.35, 0.75), rng.uniform(0.14, 0.32)),
        "free": _rgb_from_hsv(floor_hue, rng.uniform(0.18, 0.45), rng.uniform(0.58, 0.82)),
        "goal": _rgb_from_hsv(goal_hue % 1.0, rng.uniform(0.78, 1.0), rng.uniform(0.76, 0.95)),
        "agent": _rgb_from_hsv(agent_hue, 0.85, 0.90),
    }


def _goal_distances(grid: Sequence[str], goal: tuple[int, int]) -> tuple[tuple[int | None, ...], ...]:
    width = len(grid[0])
    height = len(grid)
    distances: list[list[int | None]] = [[None for _ in range(width)] for _ in range(height)]
    queue: deque[tuple[int, int]] = deque([goal])
    distances[goal[1]][goal[0]] = 0
    while queue:
        x, y = queue.popleft()
        base = distances[y][x]
        assert base is not None
        for dx, dy in YAW_TO_VEC:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != "#":
                if distances[ny][nx] is None:
                    distances[ny][nx] = base + 1
                    queue.append((nx, ny))
    return tuple(tuple(row) for row in distances)


def _candidate_goal(grid: Sequence[str], rng: random.Random) -> tuple[int, int]:
    free = [
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value != "#" and x not in (0, len(row) - 1) and y not in (0, len(grid) - 1)
    ]
    return rng.choice(free)


def generate_scene(
    *,
    split: str,
    scene_index: int,
    seed: int,
    width: int = 13,
    height: int = 13,
    obstacle_probability: float = 0.16,
    minimum_reachable_free_cells: int = 32,
    color_palette_mode: str = "fixed",
) -> GridScene:
    """Generate one deterministic grid scene with a reachable goal basin."""

    if width < 7 or height < 7:
        raise ValueError("width and height must be at least 7")
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("width and height must be odd so view symmetry is stable")
    if not (0.0 <= obstacle_probability < 0.5):
        raise ValueError("obstacle_probability must be in [0, 0.5)")

    for attempt in range(128):
        rng = random.Random(seed + scene_index * 104729 + attempt * 1009)
        rows: list[str] = []
        for y in range(height):
            chars = []
            for x in range(width):
                boundary = x in (0, width - 1) or y in (0, height - 1)
                patterned_wall = (
                    not boundary
                    and x % 4 == 0
                    and y % 3 != 1
                    and rng.random() < 0.65
                )
                random_wall = (
                    not boundary
                    and not patterned_wall
                    and rng.random() < obstacle_probability
                )
                chars.append("#" if boundary or patterned_wall or random_wall else ".")
            rows.append("".join(chars))
        goal = _candidate_goal(rows, rng)
        goal_row = list(rows[goal[1]])
        goal_row[goal[0]] = "."
        rows[goal[1]] = "".join(goal_row)
        distances = _goal_distances(rows, goal)
        reachable = sum(distance is not None for row in distances for distance in row)
        if reachable >= minimum_reachable_free_cells:
            family = "phase3a_structured_2d_maze"
            return GridScene(
                scene_id=f"{split}_phase3a_scene_{scene_index:04d}",
                family=family,
                grid=tuple(rows),
                goal=goal,
                distances=distances,
                render_palette=render_palette_for_scene(
                    mode=color_palette_mode,
                    seed=seed,
                    scene_index=scene_index,
                ),
            )
    raise RuntimeError("failed to generate a reachable Phase 3A scene")


def sample_source_states(
    scene: GridScene,
    *,
    count: int,
    seed: int,
    minimum_goal_distance: int = 3,
    maximum_goal_distance: int | None = None,
) -> tuple[GridState, ...]:
    """Sample deterministic source states with non-trivial goal distance."""

    if count < 1:
        raise ValueError("count must be positive")
    if maximum_goal_distance is not None and maximum_goal_distance < minimum_goal_distance:
        raise ValueError("maximum_goal_distance must be >= minimum_goal_distance")
    candidates = [
        (x, y)
        for y, row in enumerate(scene.distances)
        for x, distance in enumerate(row)
        if distance is not None
        and distance >= minimum_goal_distance
        and (
            maximum_goal_distance is None
            or distance <= maximum_goal_distance
        )
    ]
    if len(candidates) < count:
        distance_range = (
            f">={minimum_goal_distance}"
            if maximum_goal_distance is None
            else f"{minimum_goal_distance}..{maximum_goal_distance}"
        )
        raise ValueError(
            f"scene {scene.scene_id} has only {len(candidates)} source candidates "
            f"at goal distance {distance_range}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    states = []
    for index, (x, y) in enumerate(candidates[:count]):
        states.append(GridState(x=x, y=y, yaw=(seed + index) % 4))
    return tuple(states)


def step_state(scene: GridScene, state: GridState, action: str) -> tuple[GridState, bool]:
    """Apply one primitive and return ``(next_state, collision)``."""

    if action == "hold":
        return state, False
    if action == "turn_left":
        return GridState(state.x, state.y, (state.yaw - 1) % 4), False
    if action == "turn_right":
        return GridState(state.x, state.y, (state.yaw + 1) % 4), False
    if action != "forward":
        raise ValueError(f"unknown Phase 3A action: {action}")
    dx, dy = YAW_TO_VEC[state.yaw]
    nx, ny = state.x + dx, state.y + dy
    if not scene.is_free(nx, ny):
        return state, True
    return GridState(nx, ny, state.yaw), False


def render_observation(
    scene: GridScene,
    state: GridState,
    *,
    view_size: int = 9,
    include_goal_beacon: bool = True,
    show_goal_marker: bool = True,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Render a small ego-aligned RGB observation as channel-major floats."""

    if view_size < 3 or view_size % 2 == 0:
        raise ValueError("view_size must be an odd integer >= 3")
    radius = view_size // 2
    forward = YAW_TO_VEC[state.yaw]
    left = YAW_TO_VEC[(state.yaw - 1) % 4]
    rgb = [
        [[0.0 for _ in range(view_size)] for _ in range(view_size)]
        for _ in range(3)
    ]
    colors = scene.render_palette or DEFAULT_RENDER_PALETTE
    for row in range(view_size):
        ahead = radius - row
        for col in range(view_size):
            lateral = col - radius
            wx = state.x + forward[0] * ahead + left[0] * lateral
            wy = state.y + forward[1] * ahead + left[1] * lateral
            if wx == state.x and wy == state.y:
                color = colors["agent"]
            elif show_goal_marker and (wx, wy) == scene.goal:
                color = colors["goal"]
            elif not (0 <= wx < scene.width and 0 <= wy < scene.height):
                color = colors["outside"]
            elif scene.grid[wy][wx] == "#":
                color = colors["wall"]
            else:
                color = colors["free"]
            for channel in range(3):
                rgb[channel][row][col] = color[channel]
    if include_goal_beacon:
        dx = scene.goal[0] - state.x
        dy = scene.goal[1] - state.y
        ahead = dx * forward[0] + dy * forward[1]
        lateral = dx * left[0] + dy * left[1]
        max_span = max(scene.width - 1, scene.height - 1, 1)
        distance = scene.distance_to_goal(state.x, state.y)
        normalized_distance = (
            1.0
            if distance is None
            else min(float(distance) / float(max_span), 1.0)
        )
        beacon = (
            0.5 + 0.5 * max(min(float(ahead) / float(max_span), 1.0), -1.0),
            0.5 + 0.5 * max(min(float(lateral) / float(max_span), 1.0), -1.0),
            normalized_distance,
        )
        for channel, value in enumerate(beacon):
            rgb[channel][0][0] = value
    return tuple(tuple(tuple(row) for row in channel) for channel in rgb)


def _visible_cells(
    scene: GridScene,
    state: GridState,
    *,
    view_size: int,
    free_only: bool = False,
) -> frozenset[tuple[int, int]]:
    """Return world cells inside the ego-aligned observation crop."""

    if view_size < 3 or view_size % 2 == 0:
        raise ValueError("view_size must be an odd integer >= 3")
    radius = view_size // 2
    forward = YAW_TO_VEC[state.yaw]
    left = YAW_TO_VEC[(state.yaw - 1) % 4]
    cells = set()
    for row in range(view_size):
        ahead = radius - row
        for col in range(view_size):
            lateral = col - radius
            wx = state.x + forward[0] * ahead + left[0] * lateral
            wy = state.y + forward[1] * ahead + left[1] * lateral
            if not (0 <= wx < scene.width and 0 <= wy < scene.height):
                continue
            if free_only and not scene.is_free(wx, wy):
                continue
            cells.add((wx, wy))
    return frozenset(cells)


def _goal_marker_visible(scene: GridScene, state: GridState, *, view_size: int) -> bool:
    """Return whether the visual goal marker would be inside the crop."""

    return scene.goal in _visible_cells(scene, state, view_size=view_size)


def _state_dict(state: GridState) -> dict:
    return {"x": state.x, "y": state.y, "yaw": state.yaw}


def _utility(scene: GridScene, start: GridState, final: GridState, collisions: int) -> dict:
    start_distance = scene.distance_to_goal(start.x, start.y)
    final_distance = scene.distance_to_goal(final.x, final.y)
    if start_distance is None or final_distance is None:
        progress = 0.0
    else:
        progress = float(start_distance - final_distance)
    reached_goal = (final.x, final.y) == scene.goal
    utility = progress - 2.0 * float(collisions) + (5.0 if reached_goal else 0.0)
    return {
        "target_progress_cells": progress,
        "collision_count": int(collisions),
        "collision": collisions > 0,
        "reached_goal": reached_goal,
        "start_goal_distance_cells": start_distance,
        "final_goal_distance_cells": final_distance,
        "target_utility": utility,
        "safe_recoverable": collisions == 0 and final_distance is not None,
    }


def _explore_then_claim_labels(
    base: dict,
    *,
    history_goal_seen: bool,
    current_goal_seen: bool,
    future_goal_seen: bool,
    new_free_cells: int,
    collisions: int,
    discovery_bonus: bool = True,
    reached_bonus: bool = True,
) -> dict:
    """Return labels for self-exploration until a visual goal marker is found."""

    goal_known_before_candidate = history_goal_seen or current_goal_seen
    reached_goal = bool(base["reached_goal"])
    if goal_known_before_candidate:
        exploration_utility = float(base["target_utility"]) + 0.10 * float(
            new_free_cells
        )
    else:
        exploration_utility = (
            0.35 * float(new_free_cells)
            - 2.0 * float(collisions)
            + (4.0 if discovery_bonus and future_goal_seen else 0.0)
            + (6.0 if reached_bonus and reached_goal else 0.0)
        )
    labels = dict(base)
    labels.update(
        {
            "utility_mode": (
                "explore_then_claim"
                if discovery_bonus or reached_bonus
                else "novelty_then_claim"
            ),
            "goal_known_before_candidate": goal_known_before_candidate,
            "history_goal_marker_seen": history_goal_seen,
            "current_goal_marker_seen": current_goal_seen,
            "future_goal_marker_seen": future_goal_seen,
            "target_new_free_cells": int(new_free_cells),
            "target_goal_progress_utility": float(base["target_utility"]),
            "target_exploration_utility": exploration_utility,
            "target_utility": exploration_utility,
        }
    )
    return labels


def _candidate_sequences(horizon: int) -> tuple[tuple[str, ...], ...]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return tuple(product(ACTION_NAMES, repeat=horizon))


def _goal_variant_scenes(
    scene: GridScene,
    source_state: GridState,
    *,
    count: int,
    seed: int,
    minimum_goal_distance: int = 3,
    maximum_goal_distance: int | None = None,
) -> tuple[GridScene, ...]:
    """Return same-grid goal variants that are reachable from one source."""

    if count < 1:
        raise ValueError("goal_variants_per_source must be positive")
    if maximum_goal_distance is not None and maximum_goal_distance < minimum_goal_distance:
        raise ValueError("maximum_goal_distance must be >= minimum_goal_distance")
    if count == 1:
        return (scene,)
    candidates = [
        (x, y)
        for y, row in enumerate(scene.grid)
        for x, value in enumerate(row)
        if value != "#"
        and (x, y) != (source_state.x, source_state.y)
        and x not in (0, scene.width - 1)
        and y not in (0, scene.height - 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    variants = []
    seen = set()
    for goal in candidates:
        if goal in seen:
            continue
        distances = _goal_distances(scene.grid, goal)
        source_distance = distances[source_state.y][source_state.x]
        if source_distance is None or source_distance < minimum_goal_distance:
            continue
        if maximum_goal_distance is not None and source_distance > maximum_goal_distance:
            continue
        seen.add(goal)
        variant_index = len(variants)
        variants.append(
            GridScene(
                scene_id=f"{scene.scene_id}_goal_{variant_index:02d}",
                family=scene.family,
                grid=scene.grid,
                goal=goal,
                distances=distances,
                render_palette=scene.render_palette,
            )
        )
        if len(variants) == count:
            break
    if len(variants) < count:
        raise ValueError(
            f"scene {scene.scene_id} source {source_state} only has "
            f"{len(variants)} reachable goal variants"
        )
    return tuple(variants)


def rows_for_scene(
    scene: GridScene,
    *,
    split: str,
    source_states_per_scene: int,
    seed: int,
    horizon: int = 2,
    view_size: int = 9,
    history_steps: int = 0,
    current_goal_beacon: bool = True,
    history_goal_beacon: bool = True,
    current_goal_marker: bool = True,
    history_goal_marker: bool = True,
    future_goal_marker: bool = True,
    goal_variants_per_source: int = 1,
    history_policy: str = "turning",
    utility_mode: str = "goal_progress",
    minimum_source_goal_distance: int = 3,
    maximum_source_goal_distance: int | None = None,
    minimum_goal_variant_distance: int = 3,
    maximum_goal_variant_distance: int | None = None,
    color_palette_mode: str = "fixed",
) -> list[dict]:
    """Return one row per same-source candidate action sequence."""

    if history_steps < 0:
        raise ValueError("history_steps must be non-negative")
    if history_policy not in HISTORY_POLICIES:
        raise ValueError(f"history_policy must be one of {HISTORY_POLICIES}")
    if utility_mode not in UTILITY_MODES:
        raise ValueError(f"utility_mode must be one of {UTILITY_MODES}")
    states = sample_source_states(
        scene,
        count=source_states_per_scene,
        seed=seed,
        minimum_goal_distance=minimum_source_goal_distance,
        maximum_goal_distance=maximum_source_goal_distance,
    )
    rows = []
    sequences = _candidate_sequences(horizon)
    history_primitives = (
        ("turn_left", "turn_right", "hold")
        if history_policy == "turning"
        else ACTION_NAMES
    )
    for source_index, sampled_state in enumerate(states):
        rng = random.Random(seed + source_index * 7919 + history_steps * 104729)
        history_state = sampled_state
        history_actions = []
        history_states = []
        for _ in range(history_steps):
            history_states.append(_state_dict(history_state))
            history_action = rng.choice(history_primitives)
            history_actions.append(history_action)
            history_state, _ = step_state(scene, history_state, history_action)
        source_state = history_state
        goal_variant_scenes = _goal_variant_scenes(
            scene,
            source_state,
            count=goal_variants_per_source,
            seed=seed + source_index * 15485863,
            minimum_goal_distance=minimum_goal_variant_distance,
            maximum_goal_distance=maximum_goal_variant_distance,
        )
        for goal_variant_index, goal_scene in enumerate(goal_variant_scenes):
            variant_source_index = (
                source_index * goal_variants_per_source + goal_variant_index
            )
            variant_history_observations = [
                render_observation(
                    goal_scene,
                    GridState(
                        x=int(history_state_dict["x"]),
                        y=int(history_state_dict["y"]),
                        yaw=int(history_state_dict["yaw"]),
                    ),
                    view_size=view_size,
                    include_goal_beacon=history_goal_beacon,
                    show_goal_marker=history_goal_marker,
                )
                for history_state_dict in history_states
            ]
            history_grid_states = [
                GridState(
                    x=int(history_state_dict["x"]),
                    y=int(history_state_dict["y"]),
                    yaw=int(history_state_dict["yaw"]),
                )
                for history_state_dict in history_states
            ]
            history_goal_seen = history_goal_marker and any(
                _goal_marker_visible(goal_scene, state, view_size=view_size)
                for state in history_grid_states
            )
            current_goal_seen = (
                current_goal_marker
                and _goal_marker_visible(goal_scene, source_state, view_size=view_size)
            )
            observed_free_cells = set()
            for state in history_grid_states:
                observed_free_cells.update(
                    _visible_cells(
                        goal_scene,
                        state,
                        view_size=view_size,
                        free_only=True,
                    )
                )
            observed_free_cells.update(
                _visible_cells(
                    goal_scene,
                    source_state,
                    view_size=view_size,
                    free_only=True,
                )
            )
            start_observation = render_observation(
                goal_scene,
                source_state,
                view_size=view_size,
                include_goal_beacon=current_goal_beacon,
                show_goal_marker=current_goal_marker,
            )
            for candidate_index, sequence in enumerate(sequences):
                state = source_state
                collisions = 0
                future_observations = []
                future_states = []
                candidate_observed_free_cells = set(observed_free_cells)
                future_goal_seen = False
                new_free_cells_total = 0
                for action in sequence:
                    previous_distance = goal_scene.distance_to_goal(state.x, state.y)
                    state, collided = step_state(goal_scene, state, action)
                    next_distance = goal_scene.distance_to_goal(state.x, state.y)
                    if previous_distance is None or next_distance is None:
                        step_progress = 0.0
                    else:
                        step_progress = float(previous_distance - next_distance)
                    collisions += int(collided)
                    future_states.append(_state_dict(state))
                    visible_free_cells = _visible_cells(
                        goal_scene,
                        state,
                        view_size=view_size,
                        free_only=True,
                    )
                    newly_visible = visible_free_cells - candidate_observed_free_cells
                    candidate_observed_free_cells.update(visible_free_cells)
                    new_free_cells_total += len(newly_visible)
                    step_goal_seen = (
                        future_goal_marker
                        and _goal_marker_visible(
                            goal_scene,
                            state,
                            view_size=view_size,
                        )
                    )
                    future_goal_seen = future_goal_seen or step_goal_seen
                    future_observations.append(
                        {
                            "observation_rgb": render_observation(
                                goal_scene,
                                state,
                                view_size=view_size,
                                show_goal_marker=future_goal_marker,
                            ),
                            "observation_valid": True,
                            "collision": collided,
                            "step_progress_cells": step_progress,
                            "goal_distance_cells": next_distance,
                            "goal_marker_visible": step_goal_seen,
                            "newly_observed_free_cells": len(newly_visible),
                            "cumulative_new_free_cells": new_free_cells_total,
                            "goal_claimed": (state.x, state.y) == goal_scene.goal,
                        }
                    )
                labels = _utility(goal_scene, source_state, state, collisions)
                labels["utility_mode"] = "goal_progress"
                if utility_mode in ("explore_then_claim", "novelty_then_claim"):
                    labels = _explore_then_claim_labels(
                        labels,
                        history_goal_seen=history_goal_seen,
                        current_goal_seen=current_goal_seen,
                        future_goal_seen=future_goal_seen,
                        new_free_cells=new_free_cells_total,
                        collisions=collisions,
                        discovery_bonus=utility_mode == "explore_then_claim",
                        reached_bonus=utility_mode == "explore_then_claim",
                    )
                rows.append(
                    {
                        "schema": PHASE3A_ROW_SCHEMA,
                        "split": split,
                        "scene_id": goal_scene.scene_id,
                        "base_scene_id": scene.scene_id,
                        "family": scene.family,
                        "source_index": variant_source_index,
                        "base_source_index": source_index,
                        "goal_variant_index": goal_variant_index,
                        "goal_variants_per_source": goal_variants_per_source,
                        "candidate_index": candidate_index,
                        "start_state": _state_dict(source_state),
                        "goal": {"x": goal_scene.goal[0], "y": goal_scene.goal[1]},
                        "render_palette": {
                            key: list(value)
                            for key, value in (goal_scene.render_palette or DEFAULT_RENDER_PALETTE).items()
                        },
                        "history_steps": history_steps,
                        "history_states": history_states,
                        "history_primitive_sequence": list(history_actions),
                        "history_actions": [
                            list(action_vector(action)) for action in history_actions
                        ],
                        "history_observations_rgb": variant_history_observations,
                        "history_goal_beacon": history_goal_beacon,
                        "current_goal_beacon": current_goal_beacon,
                        "history_goal_marker": history_goal_marker,
                        "current_goal_marker": current_goal_marker,
                        "future_goal_marker": future_goal_marker,
                        "history_policy": history_policy,
                        "utility_mode": utility_mode,
                        "history_goal_marker_seen": history_goal_seen,
                        "current_goal_marker_seen": current_goal_seen,
                        "observed_free_cells_before_candidate": len(
                            observed_free_cells
                        ),
                        "start_observation_rgb": start_observation,
                        "primitive_sequence": list(sequence),
                        "active_blocks": [
                            list(action_vector(action)) for action in sequence
                        ],
                        "future_states": future_states,
                        "future_observations": future_observations,
                        "complete_valid_future_sequence": True,
                        "consequence_labels": labels,
                    }
                )
    return rows


def generate_phase3a_rows(
    *,
    split: str,
    scene_count: int,
    source_states_per_scene: int,
    seed: int,
    horizon: int = 2,
    view_size: int = 9,
    width: int = 13,
    height: int = 13,
    history_steps: int = 0,
    current_goal_beacon: bool = True,
    history_goal_beacon: bool = True,
    current_goal_marker: bool = True,
    history_goal_marker: bool = True,
    future_goal_marker: bool = True,
    goal_variants_per_source: int = 1,
    history_policy: str = "turning",
    utility_mode: str = "goal_progress",
    minimum_source_goal_distance: int = 3,
    maximum_source_goal_distance: int | None = None,
    minimum_goal_variant_distance: int = 3,
    maximum_goal_variant_distance: int | None = None,
    color_palette_mode: str = "fixed",
) -> tuple[list[dict], dict]:
    """Generate a complete Phase 3A split and return rows plus audit."""

    if scene_count < 1:
        raise ValueError("scene_count must be positive")
    rows = []
    scene_summaries = []
    for scene_index in range(scene_count):
        scene = generate_scene(
            split=split,
            scene_index=scene_index,
            seed=seed,
            width=width,
            height=height,
            color_palette_mode=color_palette_mode,
        )
        scene_rows = rows_for_scene(
            scene,
            split=split,
            source_states_per_scene=source_states_per_scene,
            seed=seed + scene_index * 1009,
            horizon=horizon,
            view_size=view_size,
            history_steps=history_steps,
            current_goal_beacon=current_goal_beacon,
            history_goal_beacon=history_goal_beacon,
            current_goal_marker=current_goal_marker,
            history_goal_marker=history_goal_marker,
            future_goal_marker=future_goal_marker,
            goal_variants_per_source=goal_variants_per_source,
            history_policy=history_policy,
            utility_mode=utility_mode,
            minimum_source_goal_distance=minimum_source_goal_distance,
            maximum_source_goal_distance=maximum_source_goal_distance,
            minimum_goal_variant_distance=minimum_goal_variant_distance,
            maximum_goal_variant_distance=maximum_goal_variant_distance,
            color_palette_mode=color_palette_mode,
        )
        rows.extend(scene_rows)
        scene_summaries.append(
            {
                "scene_id": scene.scene_id,
                "goal": {"x": scene.goal[0], "y": scene.goal[1]},
                "reachable_free_cells": sum(
                    distance is not None for row in scene.distances for distance in row
                ),
            }
        )
    audit = phase3a_dataset_audit(rows)
    audit.update(
        {
            "split": split,
            "seed": seed,
            "scene_count": scene_count,
            "source_states_per_scene": source_states_per_scene,
            "horizon": horizon,
            "view_size": view_size,
            "width": width,
            "height": height,
            "history_steps": history_steps,
            "current_goal_beacon": current_goal_beacon,
            "history_goal_beacon": history_goal_beacon,
            "current_goal_marker": current_goal_marker,
            "history_goal_marker": history_goal_marker,
            "future_goal_marker": future_goal_marker,
            "goal_variants_per_source": goal_variants_per_source,
            "history_policy": history_policy,
            "utility_mode": utility_mode,
            "minimum_source_goal_distance": minimum_source_goal_distance,
            "maximum_source_goal_distance": maximum_source_goal_distance,
            "minimum_goal_variant_distance": minimum_goal_variant_distance,
            "maximum_goal_variant_distance": maximum_goal_variant_distance,
            "color_palette_mode": color_palette_mode,
            "scenes": scene_summaries,
        }
    )
    return rows, audit


def _source_key(row: dict) -> tuple[str, int]:
    return str(row["scene_id"]), int(row["source_index"])


def phase3a_source_oracles(rows: Sequence[dict]) -> dict[tuple[str, int], dict]:
    """Return oracle first-primitive and utility statistics per source state."""

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_source_key(row)].append(row)
    result = {}
    for key, group in grouped.items():
        best_by_first: dict[str, float] = {}
        for row in group:
            first = str(row["primitive_sequence"][0])
            utility = float(row["consequence_labels"]["target_utility"])
            best_by_first[first] = max(utility, best_by_first.get(first, -math.inf))
        oracle_primitive, oracle_utility = max(
            sorted(best_by_first.items()),
            key=lambda item: item[1],
        )
        result[key] = {
            "oracle_first_primitive": oracle_primitive,
            "oracle_utility": oracle_utility,
            "best_utility_by_first_primitive": best_by_first,
        }
    return result


def phase3a_action_only_prior(
    train_rows: Sequence[dict],
    validation_rows: Sequence[dict],
) -> dict:
    """Evaluate train-derived action-only priors on validation rows."""

    train_scores: dict[tuple[str, ...], list[float]] = defaultdict(list)
    train_scores_by_first: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        sequence = tuple(str(action) for action in row["primitive_sequence"])
        utility = float(row["consequence_labels"]["target_utility"])
        train_scores[sequence].append(utility)
        train_scores_by_first[sequence[0]].append(utility)
    mean_scores = {
        "|".join(sequence): sum(values) / len(values)
        for sequence, values in sorted(train_scores.items())
    }
    mean_scores_by_first = {
        action: sum(values) / len(values)
        for action, values in sorted(train_scores_by_first.items())
    }
    selected_sequence = max(
        sorted(train_scores),
        key=lambda sequence: sum(train_scores[sequence]) / len(train_scores[sequence]),
    )
    selected_first = max(
        sorted(mean_scores_by_first),
        key=lambda action: mean_scores_by_first[action],
    )
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in validation_rows:
        grouped[_source_key(row)].append(row)
    matches = 0
    regret = 0.0
    sequence_regret = 0.0
    selected_utilities = []
    selected_sequence_utilities = []
    for group in grouped.values():
        oracle_row = max(
            sorted(group, key=lambda item: tuple(item["primitive_sequence"])),
            key=lambda item: float(item["consequence_labels"]["target_utility"]),
        )
        best_by_first: dict[str, float] = {}
        for row in group:
            first = str(row["primitive_sequence"][0])
            utility = float(row["consequence_labels"]["target_utility"])
            best_by_first[first] = max(utility, best_by_first.get(first, utility))
        selected_row = next(
            row
            for row in group
            if tuple(str(action) for action in row["primitive_sequence"])
            == selected_sequence
        )
        selected_utility = float(best_by_first[selected_first])
        selected_sequence_utility = float(
            selected_row["consequence_labels"]["target_utility"]
        )
        oracle_utility = float(oracle_row["consequence_labels"]["target_utility"])
        selected_utilities.append(selected_utility)
        selected_sequence_utilities.append(selected_sequence_utility)
        matches += int(selected_first == str(oracle_row["primitive_sequence"][0]))
        regret += oracle_utility - selected_utility
        sequence_regret += oracle_utility - selected_sequence_utility
    count = max(len(grouped), 1)
    return {
        "schema": "jepa_phase3a_action_only_prior_v2",
        "selected_primitive_sequence": list(selected_sequence),
        "selected_first_primitive": selected_first,
        "train_mean_utility_by_sequence": mean_scores,
        "train_mean_utility_by_first_primitive": mean_scores_by_first,
        "validation_source_states": len(grouped),
        "primitive_match_rate": matches / count,
        "mean_target_utility_regret": regret / count,
        "mean_selected_sequence_target_utility_regret": sequence_regret / count,
        "mean_selected_utility": sum(selected_utilities) / count,
        "mean_selected_sequence_utility": sum(selected_sequence_utilities) / count,
    }


def phase3a_dataset_audit(rows: Sequence[dict]) -> dict:
    """Return deterministic data-contract counts for a Phase 3A split."""

    if not rows:
        raise ValueError("Phase 3A rows must not be empty")
    source_counts = Counter(_source_key(row) for row in rows)
    candidate_counts = Counter(source_counts.values())
    primitive_counts = Counter(str(row["primitive_sequence"][0]) for row in rows)
    history_lengths = Counter(int(row.get("history_steps", 0)) for row in rows)
    current_beacon_counts = Counter(
        bool(row.get("current_goal_beacon", True)) for row in rows
    )
    history_beacon_counts = Counter(
        bool(row.get("history_goal_beacon", True)) for row in rows
    )
    current_marker_counts = Counter(
        bool(row.get("current_goal_marker", True)) for row in rows
    )
    history_marker_counts = Counter(
        bool(row.get("history_goal_marker", True)) for row in rows
    )
    future_marker_counts = Counter(
        bool(row.get("future_goal_marker", True)) for row in rows
    )
    goal_variant_counts = Counter(
        int(row.get("goal_variants_per_source", 1)) for row in rows
    )
    history_policy_counts = Counter(
        str(row.get("history_policy", "turning")) for row in rows
    )
    utility_mode_counts = Counter(
        str(
            row.get("consequence_labels", {}).get(
                "utility_mode",
                row.get("utility_mode", "goal_progress"),
            )
        )
        for row in rows
    )
    history_goal_seen_counts = Counter(
        bool(row.get("history_goal_marker_seen", False)) for row in rows
    )
    current_goal_seen_counts = Counter(
        bool(row.get("current_goal_marker_seen", False)) for row in rows
    )
    future_goal_seen_counts = Counter(
        bool(row["consequence_labels"].get("future_goal_marker_seen", False))
        for row in rows
    )
    collisions = sum(int(row["consequence_labels"]["collision"]) for row in rows)
    utilities = [float(row["consequence_labels"]["target_utility"]) for row in rows]
    return {
        "schema": PHASE3A_AUDIT_SCHEMA,
        "rows": len(rows),
        "source_states": len(source_counts),
        "candidate_rows_per_source_histogram": dict(sorted(candidate_counts.items())),
        "first_primitive_counts": dict(sorted(primitive_counts.items())),
        "history_step_counts": dict(sorted(history_lengths.items())),
        "current_goal_beacon_counts": {
            str(key): value for key, value in sorted(current_beacon_counts.items())
        },
        "history_goal_beacon_counts": {
            str(key): value for key, value in sorted(history_beacon_counts.items())
        },
        "current_goal_marker_counts": {
            str(key): value for key, value in sorted(current_marker_counts.items())
        },
        "history_goal_marker_counts": {
            str(key): value for key, value in sorted(history_marker_counts.items())
        },
        "future_goal_marker_counts": {
            str(key): value for key, value in sorted(future_marker_counts.items())
        },
        "goal_variants_per_source_counts": dict(sorted(goal_variant_counts.items())),
        "history_policy_counts": dict(sorted(history_policy_counts.items())),
        "utility_mode_counts": dict(sorted(utility_mode_counts.items())),
        "history_goal_marker_seen_counts": {
            str(key): value for key, value in sorted(history_goal_seen_counts.items())
        },
        "current_goal_marker_seen_counts": {
            str(key): value for key, value in sorted(current_goal_seen_counts.items())
        },
        "future_goal_marker_seen_counts": {
            str(key): value for key, value in sorted(future_goal_seen_counts.items())
        },
        "collision_rows": collisions,
        "collision_fraction": collisions / len(rows),
        "mean_target_utility": sum(utilities) / len(utilities),
        "min_target_utility": min(utilities),
        "max_target_utility": max(utilities),
        "schemas": dict(Counter(str(row.get("schema")) for row in rows)),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows as newline-delimited JSON with stable key order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""

    return [json.loads(line) for line in path.open()]
