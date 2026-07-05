"""Non-privileged egocentric marker-memory helpers for Phase 3A."""
from __future__ import annotations

from typing import Mapping, Sequence

from .phase3a_positive_control import ACTION_NAMES


def egocentric_marker_memory_predictions(rows: Sequence[Mapping]) -> list[float]:
    """Score candidates with pixel marker memory plus egocentric action odometry.

    This baseline uses no simulator state, map coordinates, goal beacon, or goal
    label at inference. It detects the visual goal marker in previous RGB crops,
    rolls that marker location forward through the observed action history, then
    scores candidate action sequences by whether they end at the remembered
    marker location.
    """

    return [_egocentric_marker_memory_score(row) for row in rows]


def egocentric_marker_memory_score(row: Mapping) -> float:
    """Return one non-privileged marker-memory score for a candidate row."""

    return _egocentric_marker_memory_score(row)


def egocentric_marker_memory_delta(row: Mapping) -> tuple[bool, int, int]:
    """Return whether a remembered marker exists and its final ego delta.

    The returned ``(ahead, lateral)`` location is expressed after applying the
    row's candidate action sequence. A claimed marker is therefore ``(0, 0)``.
    """

    marker = remembered_marker_position(row)
    if marker is None:
        return False, 0, 0
    ahead, lateral = marker
    for action in primitive_sequence_from_blocks(row.get("active_blocks", [])):
        ahead, lateral = advance_marker_egocentric(ahead, lateral, action)
    return True, ahead, lateral


def marker_memory_target(row: Mapping) -> tuple[bool, tuple[float, float], float]:
    """Return normalized delta and claim target for the learned memory head."""

    valid, ahead, lateral = egocentric_marker_memory_delta(row)
    normalizer = max(float(_view_size(row)), 1.0)
    claimed = float(valid and abs(ahead) + abs(lateral) == 0)
    return valid, (float(ahead) / normalizer, float(lateral) / normalizer), claimed


def marker_memory_start_target(row: Mapping) -> tuple[bool, tuple[float, float]]:
    """Return normalized remembered marker delta before candidate actions."""

    marker = remembered_marker_position(row)
    normalizer = max(float(_view_size(row)), 1.0)
    if marker is None:
        return False, (0.0, 0.0)
    ahead, lateral = marker
    return True, (float(ahead) / normalizer, float(lateral) / normalizer)


def marker_memory_start_cell_target(
    row: Mapping,
    *,
    radius: int = 2,
) -> tuple[bool, int]:
    """Return a categorical start-memory cell target within a square support."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    marker = remembered_marker_position(row)
    if marker is None:
        return False, 0
    ahead, lateral = marker
    if abs(ahead) > radius or abs(lateral) > radius:
        return False, 0
    width = 2 * radius + 1
    return True, (ahead + radius) * width + (lateral + radius)


def remembered_marker_position(row: Mapping) -> tuple[int, int] | None:
    """Return the marker position in the candidate-start ego frame, if observed."""

    history = row.get("history_observations_rgb", [])
    actions = primitive_sequence_from_blocks(row.get("history_actions", []))
    marker_color = _marker_color(row)
    for index in range(len(history) - 1, -1, -1):
        marker = marker_position_in_observation(
            history[index],
            marker_color=marker_color,
        )
        if marker is None:
            continue
        ahead, lateral = marker
        for action in actions[index:]:
            ahead, lateral = advance_marker_egocentric(ahead, lateral, action)
        return ahead, lateral
    marker = marker_position_in_observation(
        row.get("start_observation_rgb"),
        marker_color=marker_color,
    )
    if marker is None:
        return None
    return marker


def marker_position_in_observation(
    observation: object,
    *,
    marker_color: tuple[float, float, float] | None = None,
) -> tuple[int, int] | None:
    """Detect the goal marker in one channel-major RGB observation."""

    if not isinstance(observation, Sequence) or len(observation) != 3:
        return None
    red, green, blue = observation
    if not (
        isinstance(red, Sequence)
        and isinstance(green, Sequence)
        and isinstance(blue, Sequence)
        and len(red) == len(green) == len(blue)
        and len(red) > 0
    ):
        return None
    view_size = len(red)
    radius = view_size // 2
    best: tuple[float, int, int] | None = None
    for row_index in range(view_size):
        if not (
            isinstance(red[row_index], Sequence)
            and isinstance(green[row_index], Sequence)
            and isinstance(blue[row_index], Sequence)
        ):
            continue
        for col_index in range(len(green[row_index])):
            r = float(red[row_index][col_index])
            g = float(green[row_index][col_index])
            b = float(blue[row_index][col_index])
            if marker_color is None:
                marker_score = g - max(r, b)
                if g < 0.7 or r > 0.35 or b > 0.45:
                    continue
            else:
                distance = (
                    (r - marker_color[0]) ** 2
                    + (g - marker_color[1]) ** 2
                    + (b - marker_color[2]) ** 2
                )
                if distance > 1e-4:
                    continue
                marker_score = -distance
            if best is None or marker_score > best[0]:
                best = (marker_score, row_index, col_index)
    if best is None:
        return None
    _, row_index, col_index = best
    return radius - row_index, col_index - radius


def primitive_sequence_from_blocks(blocks: object) -> tuple[str, ...]:
    """Decode one-hot action blocks into Phase 3A primitive names."""

    if not isinstance(blocks, Sequence):
        return ()
    names = []
    for block in blocks:
        if isinstance(block, str):
            names.append(block)
            continue
        if not isinstance(block, Sequence) or len(block) == 0:
            continue
        index = max(
            range(min(len(block), len(ACTION_NAMES))),
            key=lambda item: float(block[item]),
        )
        names.append(ACTION_NAMES[index])
    return tuple(names)


def advance_marker_egocentric(ahead: int, lateral: int, action: str) -> tuple[int, int]:
    """Roll a remembered marker through one egocentric primitive."""

    if action == "forward":
        return ahead - 1, lateral
    if action == "turn_left":
        return lateral, -ahead
    if action == "turn_right":
        return -lateral, ahead
    return ahead, lateral


def _egocentric_marker_memory_score(row: Mapping) -> float:
    valid, ahead, lateral = egocentric_marker_memory_delta(row)
    if not valid:
        return 0.0
    distance = abs(ahead) + abs(lateral)
    claimed = distance == 0
    return (100.0 if claimed else 0.0) - float(distance)


def _view_size(row: Mapping) -> int:
    observation = row.get("start_observation_rgb")
    if isinstance(observation, Sequence) and len(observation) == 3:
        channel = observation[0]
        if isinstance(channel, Sequence):
            return len(channel)
    history = row.get("history_observations_rgb", [])
    if isinstance(history, Sequence) and history:
        item = history[0]
        if isinstance(item, Sequence) and len(item) == 3 and isinstance(item[0], Sequence):
            return len(item[0])
    return 1


def _marker_color(row: Mapping) -> tuple[float, float, float] | None:
    palette = row.get("render_palette")
    if not isinstance(palette, Mapping):
        return None
    goal = palette.get("goal")
    if not isinstance(goal, Sequence) or len(goal) != 3:
        return None
    return tuple(float(channel) for channel in goal)
