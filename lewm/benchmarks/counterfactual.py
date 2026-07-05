"""Counterfactual action-sequence labels for JEPA navigation experiments.

The benchmark deliberately exposes separate safety, clearance, progress,
heading, and recoverability labels. A single scalar cost may be useful for a
specific planner, but it is not the benchmark contract: models should be
compared on the consequences that navigation actually requires.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import numpy as np


class CounterfactualGrid(Protocol):
    """Minimum privileged-oracle surface required by the benchmark."""

    def is_free(self, xy: tuple[float, float]) -> bool: ...

    def configuration_clearance_m(self, xy: tuple[float, float]) -> float: ...

    def astar(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]): ...


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class CandidateTrajectory:
    """Privileged labels for one action sequence from one matched start state."""

    primitive_sequence: tuple[str, ...]
    endpoint: Pose2D
    path_length_m: float
    start_configuration_clearance_m: float
    end_configuration_clearance_m: float
    minimum_swept_configuration_clearance_m: float
    p05_swept_configuration_clearance_m: float
    clearance_gain_m: float
    starts_grid_unsafe: bool
    enters_grid_unsafe: bool
    ends_grid_unsafe: bool
    unsafe_sample_fraction: float
    target_initial_distance_m: float | None
    target_final_distance_m: float | None
    target_progress_m: float | None
    target_heading_error_rad: float | None
    target_recoverable: bool | None
    sampled_trajectory_xyyaw: tuple[tuple[float, float, float], ...] | None = None

    def to_jsonable(self) -> dict:
        payload = asdict(self)
        payload["primitive_sequence"] = list(self.primitive_sequence)
        if payload["sampled_trajectory_xyyaw"] is None:
            del payload["sampled_trajectory_xyyaw"]
        return payload


def _wrap_angle_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _bounded_clearance(value: float, cap_m: float) -> float:
    if math.isnan(value):
        raise ValueError("configuration clearance query returned NaN")
    if math.isinf(value):
        return float(cap_m if value > 0.0 else -cap_m)
    return float(max(-cap_m, min(cap_m, value)))


def integrate_action_blocks(
    *,
    action_blocks: Sequence[np.ndarray],
    start: Pose2D,
    command_dt_s: float,
) -> tuple[Pose2D, ...]:
    """Return the kinematic pose after each complete action block."""

    if not action_blocks:
        raise ValueError("action_blocks must not be empty")
    if command_dt_s <= 0.0:
        raise ValueError("command_dt_s must be positive")

    x, y, yaw = float(start.x_m), float(start.y_m), float(start.yaw_rad)
    endpoints = []
    for action_block in action_blocks:
        block = np.asarray(action_block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != 3:
            raise ValueError(f"action block must have shape (T, 3), got {block.shape}")
        for vx_body, vy_body, yaw_rate in block:
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            x += (
                float(vx_body) * cos_yaw - float(vy_body) * sin_yaw
            ) * command_dt_s
            y += (
                float(vx_body) * sin_yaw + float(vy_body) * cos_yaw
            ) * command_dt_s
            yaw = _wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
        endpoints.append(Pose2D(x_m=x, y_m=y, yaw_rad=yaw))
    return tuple(endpoints)


def simulate_candidate_trajectory(
    *,
    primitive_sequence: Sequence[str],
    action_blocks: Sequence[np.ndarray],
    start: Pose2D,
    command_dt_s: float,
    grid: CounterfactualGrid,
    target_xy: tuple[float, float] | None,
    sweep_step_m: float = 0.025,
    sweep_step_yaw_rad: float = 0.05,
    clearance_cap_m: float = 10.0,
    include_trajectory: bool = False,
) -> CandidateTrajectory:
    """Simulate a multi-block candidate and label its complete swept path.

    The simulator is intentionally kinematic and privileged. It establishes a
    reproducible decision benchmark before the bounded physics-replay labels
    are added. ``enters_grid_unsafe`` excludes starts that are already inside
    the inflated grid, avoiding the v1 failure where every stationary action
    was incorrectly treated as a new collision.
    """

    if len(primitive_sequence) != len(action_blocks):
        raise ValueError("primitive_sequence and action_blocks must have equal length")
    if not primitive_sequence:
        raise ValueError("candidate action sequence must not be empty")
    if command_dt_s <= 0.0:
        raise ValueError("command_dt_s must be positive")
    if sweep_step_m <= 0.0 or sweep_step_yaw_rad <= 0.0:
        raise ValueError("sweep steps must be positive")

    x, y, yaw = float(start.x_m), float(start.y_m), float(start.yaw_rad)
    samples: list[tuple[float, float, float]] = [(x, y, yaw)]
    clearances = [
        _bounded_clearance(grid.configuration_clearance_m((x, y)), clearance_cap_m)
    ]
    unsafe = [not grid.is_free((x, y)) or clearances[0] < 0.0]
    path_length_m = 0.0

    for action_block in action_blocks:
        block = np.asarray(action_block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != 3:
            raise ValueError(f"action block must have shape (T, 3), got {block.shape}")
        for vx_body, vy_body, yaw_rate in block:
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            next_x = x + (
                float(vx_body) * cos_yaw - float(vy_body) * sin_yaw
            ) * command_dt_s
            next_y = y + (
                float(vx_body) * sin_yaw + float(vy_body) * cos_yaw
            ) * command_dt_s
            yaw_delta = float(yaw_rate) * command_dt_s
            next_yaw = _wrap_angle_pi(yaw + yaw_delta)
            translation = math.hypot(next_x - x, next_y - y)
            path_length_m += translation
            substeps = max(
                1,
                int(math.ceil(translation / sweep_step_m)),
                int(math.ceil(abs(yaw_delta) / sweep_step_yaw_rad)),
            )
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                sample_x = x + (next_x - x) * fraction
                sample_y = y + (next_y - y) * fraction
                sample_yaw = _wrap_angle_pi(yaw + yaw_delta * fraction)
                clearance = _bounded_clearance(
                    grid.configuration_clearance_m((sample_x, sample_y)),
                    clearance_cap_m,
                )
                samples.append((sample_x, sample_y, sample_yaw))
                clearances.append(clearance)
                unsafe.append(
                    not grid.is_free((sample_x, sample_y)) or clearance < 0.0
                )
            x, y, yaw = next_x, next_y, next_yaw

    start_clearance = clearances[0]
    end_clearance = clearances[-1]
    start_unsafe = unsafe[0]
    enters_unsafe = any(
        not unsafe[index - 1] and unsafe[index] for index in range(1, len(unsafe))
    )
    endpoint = Pose2D(x_m=x, y_m=y, yaw_rad=yaw)

    if target_xy is None:
        initial_distance = None
        final_distance = None
        progress = None
        heading_error = None
        recoverable = None
    else:
        initial_distance = math.dist((start.x_m, start.y_m), target_xy)
        final_distance = math.dist((x, y), target_xy)
        progress = initial_distance - final_distance
        target_bearing = math.atan2(target_xy[1] - y, target_xy[0] - x)
        heading_error = abs(_wrap_angle_pi(target_bearing - yaw))
        recoverable = bool(
            not unsafe[-1] and grid.astar((x, y), target_xy) is not None
        )

    return CandidateTrajectory(
        primitive_sequence=tuple(str(value) for value in primitive_sequence),
        endpoint=endpoint,
        path_length_m=float(path_length_m),
        start_configuration_clearance_m=float(start_clearance),
        end_configuration_clearance_m=float(end_clearance),
        minimum_swept_configuration_clearance_m=float(min(clearances)),
        p05_swept_configuration_clearance_m=float(np.quantile(clearances, 0.05)),
        clearance_gain_m=float(end_clearance - start_clearance),
        starts_grid_unsafe=bool(start_unsafe),
        enters_grid_unsafe=bool(enters_unsafe),
        ends_grid_unsafe=bool(unsafe[-1]),
        unsafe_sample_fraction=float(np.mean(unsafe)),
        target_initial_distance_m=(
            float(initial_distance) if initial_distance is not None else None
        ),
        target_final_distance_m=(
            float(final_distance) if final_distance is not None else None
        ),
        target_progress_m=float(progress) if progress is not None else None,
        target_heading_error_rad=(
            float(heading_error) if heading_error is not None else None
        ),
        target_recoverable=recoverable,
        sampled_trajectory_xyyaw=tuple(samples) if include_trajectory else None,
    )


def oracle_sort_key(candidate: CandidateTrajectory) -> tuple:
    """Return a transparent lexicographic oracle ordering.

    Safety and recoverability are hard priorities. Among equally valid
    candidates, target progress, swept clearance, heading, and path length are
    used in that order. The benchmark still reports every component separately.
    """

    has_target = candidate.target_progress_m is not None
    task_gain = (
        candidate.target_progress_m if has_target else candidate.clearance_gain_m
    )
    heading_error = (
        candidate.target_heading_error_rad
        if candidate.target_heading_error_rad is not None
        else 0.0
    )
    unrecoverable = (
        not candidate.target_recoverable
        if candidate.target_recoverable is not None
        else False
    )
    return (
        candidate.enters_grid_unsafe,
        candidate.ends_grid_unsafe,
        unrecoverable,
        -float(task_gain),
        -candidate.p05_swept_configuration_clearance_m,
        float(heading_error),
        candidate.path_length_m,
        candidate.primitive_sequence,
    )
