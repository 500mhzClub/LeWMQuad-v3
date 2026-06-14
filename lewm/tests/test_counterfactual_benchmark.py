from __future__ import annotations

import numpy as np
import pytest

from lewm.benchmarks.counterfactual import (
    Pose2D,
    integrate_action_blocks,
    oracle_sort_key,
    simulate_candidate_trajectory,
)


class _ToyGrid:
    def __init__(self, wall_x_m: float = 0.4):
        self.wall_x_m = wall_x_m

    def is_free(self, xy: tuple[float, float]) -> bool:
        return xy[0] < self.wall_x_m

    def configuration_clearance_m(self, xy: tuple[float, float]) -> float:
        return self.wall_x_m - xy[0]

    def astar(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]):
        return object() if self.is_free(start_xy) and self.is_free(goal_xy) else None


def _block(vx: float = 0.0, yaw_rate: float = 0.0) -> np.ndarray:
    return np.asarray([[vx, 0.0, yaw_rate]] * 5, dtype=np.float32)


def test_simulate_candidate_labels_complete_swept_path() -> None:
    candidate = simulate_candidate_trajectory(
        primitive_sequence=("forward", "forward"),
        action_blocks=(_block(vx=0.5), _block(vx=0.5)),
        start=Pose2D(0.0, 0.0, 0.0),
        command_dt_s=0.1,
        grid=_ToyGrid(wall_x_m=0.4),
        target_xy=(0.35, 0.0),
        sweep_step_m=0.01,
    )

    assert candidate.path_length_m == pytest.approx(0.5)
    assert candidate.enters_grid_unsafe
    assert candidate.ends_grid_unsafe
    assert candidate.minimum_swept_configuration_clearance_m < 0.0
    assert candidate.target_progress_m is not None


def test_integrate_action_blocks_returns_each_block_endpoint() -> None:
    endpoints = integrate_action_blocks(
        action_blocks=(_block(vx=0.5), _block(yaw_rate=np.pi)),
        start=Pose2D(0.0, 0.0, 0.0),
        command_dt_s=0.1,
    )

    assert len(endpoints) == 2
    assert endpoints[0].x_m == pytest.approx(0.25)
    assert endpoints[0].y_m == pytest.approx(0.0)
    assert endpoints[1].x_m == pytest.approx(0.25)
    assert endpoints[1].yaw_rad == pytest.approx(np.pi / 2)


def test_starting_unsafe_does_not_count_as_entering_unsafe() -> None:
    candidate = simulate_candidate_trajectory(
        primitive_sequence=("hold",),
        action_blocks=(_block(),),
        start=Pose2D(0.45, 0.0, 0.0),
        command_dt_s=0.1,
        grid=_ToyGrid(wall_x_m=0.4),
        target_xy=None,
    )

    assert candidate.starts_grid_unsafe
    assert not candidate.enters_grid_unsafe
    assert candidate.ends_grid_unsafe


def test_oracle_prioritizes_safety_before_progress() -> None:
    safe = simulate_candidate_trajectory(
        primitive_sequence=("hold",),
        action_blocks=(_block(),),
        start=Pose2D(0.0, 0.0, 0.0),
        command_dt_s=0.1,
        grid=_ToyGrid(wall_x_m=0.4),
        target_xy=(1.0, 0.0),
    )
    unsafe_progress = simulate_candidate_trajectory(
        primitive_sequence=("forward",),
        action_blocks=(_block(vx=1.0),),
        start=Pose2D(0.0, 0.0, 0.0),
        command_dt_s=0.1,
        grid=_ToyGrid(wall_x_m=0.4),
        target_xy=(1.0, 0.0),
    )

    assert unsafe_progress.target_progress_m > safe.target_progress_m
    assert oracle_sort_key(safe) < oracle_sort_key(unsafe_progress)
