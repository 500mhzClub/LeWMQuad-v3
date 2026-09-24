from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from scripts import diagnose_go2_swept_progress_selection_v1 as diagnostic


@dataclass(frozen=True)
class _Report:
    feasible: bool


class _RecordingChecker:
    def __init__(self, *, collision_y_m: float) -> None:
        self.collision_y_m = collision_y_m
        self.sweeps: list[tuple[object, object, float, float]] = []

    def interpolated_sweep(
        self,
        start: object,
        end: object,
        *,
        maximum_corner_step_m: float,
        maximum_yaw_step_rad: float,
    ) -> tuple[tuple[float, object], ...]:
        self.sweeps.append(
            (start, end, maximum_corner_step_m, maximum_yaw_step_rad)
        )
        return ((0.0, start), (1.0, end))

    def pose_feasibility(self, pose: object) -> _Report:
        return _Report(feasible=pose.y_m < self.collision_y_m)


def _row(
    state_index: int,
    *,
    family: str,
    action: str,
    prefix: int,
    immediate_feasible: bool = True,
) -> dict[str, object]:
    return {
        "dataset_role": diagnostic.SELECTION_ROLE,
        "role_state_index": state_index,
        "family": family,
        "action_index": diagnostic.labels.ACTION_ORDER.index(action),
        "action": action,
        "immediate_feasible": immediate_feasible,
        "swept_progress_prefix_length": prefix,
    }


def _state_rows(
    state_index: int,
    *,
    family: str,
    prefixes: dict[str, int],
    infeasible: set[str] = frozenset(),
) -> list[dict[str, object]]:
    return [
        _row(
            state_index,
            family=family,
            action=action,
            prefix=prefixes[action],
            immediate_feasible=action not in infeasible,
        )
        for action in diagnostic.NON_HOLD_ACTIONS
    ]


def test_swept_progress_uses_exact_segments_and_stops_at_first_collision() -> None:
    checker = _RecordingChecker(collision_y_m=3.35)
    post = diagnostic.labels.Pose2D(2.0, 3.0, math.pi / 2.0)

    prefix = diagnostic.swept_progress_prefix_v1(
        checker, post, immediate_feasible=True
    )

    assert prefix == 3
    assert len(checker.sweeps) == 4
    for index, (start, end, corner_step, yaw_step) in enumerate(checker.sweeps):
        assert start.x_m == pytest.approx(2.0)
        assert end.x_m == pytest.approx(2.0)
        assert start.y_m == pytest.approx(3.0 + index * 0.1)
        assert end.y_m == pytest.approx(3.0 + (index + 1) * 0.1)
        assert start.yaw_rad == pytest.approx(math.pi / 2.0)
        assert end.yaw_rad == pytest.approx(math.pi / 2.0)
        assert corner_step == diagnostic.labels.MAXIMUM_CORNER_STEP_M
        assert yaw_step == diagnostic.labels.MAXIMUM_YAW_STEP_RAD


def test_infeasible_immediate_primitive_forces_zero_without_progress_sweep() -> None:
    checker = _RecordingChecker(collision_y_m=100.0)
    post = diagnostic.labels.Pose2D(0.0, 0.0, 0.0)

    assert (
        diagnostic.swept_progress_prefix_v1(
            checker, post, immediate_feasible=False
        )
        == 0
    )
    assert checker.sweeps == []


def test_aggregation_counts_informative_states_histograms_and_actions() -> None:
    actions = diagnostic.NON_HOLD_ACTIONS
    rows = [
        *_state_rows(
            0,
            family="family_a",
            prefixes={action: (3 if index == 0 else 1) for index, action in enumerate(actions)},
        ),
        *_state_rows(
            1,
            family="family_a",
            prefixes={action: 2 for action in actions},
        ),
        *_state_rows(
            2,
            family="family_b",
            prefixes={action: 0 for action in actions},
            infeasible=set(actions),
        ),
    ]

    census = diagnostic.aggregate_selection_rows_v1(
        rows, families=("family_a", "family_b")
    )

    aggregate = census["aggregate"]
    assert aggregate["state_count"] == 3
    assert aggregate["informative_state_count"] == 1
    assert aggregate["rejection_counts"] == {
        "zero_best_prefix": 1,
        "positive_but_no_action_difference": 1,
    }
    assert aggregate["prefix_histogram_0_through_15"][0] == 8
    assert aggregate["prefix_histogram_0_through_15"][1] == 7
    assert aggregate["prefix_histogram_0_through_15"][2] == 8
    assert aggregate["prefix_histogram_0_through_15"][3] == 1
    first = aggregate["actions"][actions[0]]
    assert first["immediate_feasible_count"] == 2
    assert first["positive_prefix_count"] == 2
    assert first["variation_participation_count"] == 1
    assert first["prefix_histogram_0_through_15"][0] == 1
    assert first["prefix_histogram_0_through_15"][2] == 1
    assert first["prefix_histogram_0_through_15"][3] == 1
    family_a = census["families"]["family_a"]
    assert family_a["state_count"] == 2
    assert family_a["informative_state_count"] == 1
    assert census["families"]["family_b"]["informative_state_count"] == 0
