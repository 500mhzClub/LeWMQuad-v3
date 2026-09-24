"""Focused behaviour and provenance tests for the closed-loop oracle assay."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_lewm_closed_loop_mpc as benchmark  # noqa: E402


class _FakeRobot:
    def __init__(self, xy: tuple[float, float], yaw: float) -> None:
        self.pos = np.asarray([xy[0], xy[1], 0.3], dtype=np.float32)
        self.quat = benchmark._quat_wxyz_from_yaw(yaw)

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return self.quat

    def set_pos(self, value, **_kwargs) -> None:
        self.pos = np.asarray(value, dtype=np.float32)[0].copy()

    def set_quat(self, value, **_kwargs) -> None:
        self.quat = np.asarray(value, dtype=np.float32)[0].copy()


class _FakeBuild:
    def __init__(self, xy: tuple[float, float], yaw: float) -> None:
        self.robot = _FakeRobot(xy, yaw)


def test_kinematic_endpoint_matches_sequential_execution(monkeypatch) -> None:
    blocks = {
        "forward": np.asarray(
            [[0.35, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32
        ),
        "turn": np.asarray(
            [[0.0, 0.0, 0.7], [0.0, 0.0, 0.4]], dtype=np.float32
        ),
        "arc": np.asarray(
            [[0.2, 0.0, -0.3], [0.2, 0.0, -0.3]], dtype=np.float32
        ),
    }
    monkeypatch.setattr(
        benchmark,
        "expand_primitive_to_block",
        lambda registry, name: registry[name],
    )
    sequence = ("forward", "turn", "arc")
    start_xy = (0.2, -0.4)
    start_yaw = 0.35
    command_dt_s = 0.1

    expected_xy = benchmark._kinematic_endpoint(
        sequence,
        blocks,
        None,
        start_xy,
        start_yaw,
        command_dt_s,
    )
    build = _FakeBuild(start_xy, start_yaw)
    for primitive_name in sequence:
        benchmark._execute_kinematic_primitive(
            build,
            blocks,
            primitive_name,
            command_dt_s=command_dt_s,
            grid=None,
        )

    actual_pos, _actual_quat = benchmark._current_pose(build)
    assert np.allclose(actual_pos[:2], expected_xy, rtol=0.0, atol=1e-7)


def test_candidate_bank_order_is_exact_for_horizon_one_and_two() -> None:
    names = ["hold", "forward", "left"]
    blocks = {
        name: np.full(3, index, dtype=np.float32)
        for index, name in enumerate(names)
    }
    horizon_one, actions_one = benchmark._candidate_action_tensor(
        blocks,
        names,
        1,
        max_candidates=None,
        rng=random.Random(3),
        device=torch.device("cpu"),
    )
    horizon_two, actions_two = benchmark._candidate_action_tensor(
        blocks,
        names,
        2,
        max_candidates=None,
        rng=random.Random(3),
        device=torch.device("cpu"),
    )

    assert horizon_one == [("hold",), ("forward",), ("left",)]
    assert horizon_two == [
        ("hold", "hold"),
        ("hold", "forward"),
        ("hold", "left"),
        ("forward", "hold"),
        ("forward", "forward"),
        ("forward", "left"),
        ("left", "hold"),
        ("left", "forward"),
        ("left", "left"),
    ]
    assert tuple(actions_one.shape) == (3, 1, 3)
    assert tuple(actions_two.shape) == (9, 2, 3)


def test_shuffle_is_reproducible_nonidentity_and_moves_score_sources() -> None:
    first = benchmark._deterministic_score_permutation(7, seed=7, block_index=0)
    replay = benchmark._deterministic_score_permutation(7, seed=7, block_index=0)
    assert np.array_equal(first, replay)
    assert sorted(first.tolist()) == list(range(7))
    assert not np.array_equal(first, np.arange(7, dtype=np.int64))

    costs = np.arange(7, dtype=np.float64)
    sequences = [(f"action_{index}",) for index in range(7)]
    shuffled_ranking = benchmark._oracle_ranking(costs[first], sequences)
    selected_slot = int(shuffled_ranking["best_candidate_index"])
    assert int(first[selected_slot]) == 0
    assert selected_slot != 0
    assert benchmark.ORACLE_SHUFFLE_NAME == (
        "deterministic_candidate_score_permutation"
    )
    assert benchmark.ORACLE_SHUFFLE_VERSION == 1


def test_horizon_one_ties_use_order_but_not_false_action_disagreement() -> None:
    tolerance = benchmark.ORACLE_COST_TIE_TOLERANCE_M
    sequences = [("hold",), ("forward",), ("left",)]
    costs = np.asarray([1.0, 0.5 + 0.5 * tolerance, 0.5], dtype=np.float64)

    ranking = benchmark._oracle_ranking(costs, sequences)
    assert ranking["best_candidate_index"] == 1
    assert ranking["best_cost_m"] == 0.5
    assert ranking["optimal_candidate_indices"] == [1, 2]
    assert ranking["optimal_first_primitives"] == ["forward", "left"]
    assert ranking["tie_break"] == "lowest_candidate_index_within_tolerance"

    forward = benchmark._oracle_first_action_assessment(
        costs,
        sequences,
        "forward",
        oracle_best_cost_m=ranking["best_cost_m"],
    )
    left = benchmark._oracle_first_action_assessment(
        costs,
        sequences,
        "left",
        oracle_best_cost_m=ranking["best_cost_m"],
    )
    hold = benchmark._oracle_first_action_assessment(
        costs,
        sequences,
        "hold",
        oracle_best_cost_m=ranking["best_cost_m"],
    )
    assert forward["disagreement"] is False
    assert left["disagreement"] is False
    assert hold["disagreement"] is True


def test_horizon_two_tied_candidates_expose_all_optimal_first_actions() -> None:
    sequences = [
        ("hold", "forward"),
        ("forward", "hold"),
        ("forward", "forward"),
        ("left", "hold"),
    ]
    costs = np.asarray([0.25, 0.25, 0.7, 0.8], dtype=np.float64)
    ranking = benchmark._oracle_ranking(costs, sequences)

    assert ranking["best_candidate_index"] == 0
    assert ranking["optimal_candidate_count"] == 2
    assert ranking["optimal_first_primitives"] == ["hold", "forward"]
    forward = benchmark._oracle_first_action_assessment(
        costs,
        sequences,
        "forward",
        oracle_best_cost_m=ranking["best_cost_m"],
    )
    assert forward == {
        "best_candidate_index": 1,
        "best_cost_m": 0.25,
        "regret_m": 0.0,
        "disagreement": False,
    }


def test_oracle_assay_provenance_binds_replay_inputs_and_mode_scope() -> None:
    sequences = [("hold", "forward"), ("forward", "hold")]
    provenance = benchmark._oracle_assay_provenance(
        seed=41,
        max_candidates=2,
        sequences=sequences,
        mode="kinematic",
    )

    assert provenance["name"] == "privileged_kinematic_endpoint_distance"
    assert provenance["version"] == 1
    assert provenance["validity"] == "exact_execution_oracle_in_kinematic_mode"
    assert provenance["candidate_bank"] == {
        "count": 2,
        "max_candidates": 2,
        "ordered_sequences": [["hold", "forward"], ["forward", "hold"]],
    }
    assert provenance["planning_grid"] == {
        "cell_size_m": 0.05,
        "inflation_m": 0.20,
    }
    assert provenance["tie_tolerance_m"] == 1e-9
    assert provenance["shuffle"]["seed"] == 41
    assert provenance["shuffle"]["name"] == (
        "deterministic_candidate_score_permutation"
    )
    assert provenance["shuffle"]["version"] == 1

    physical = benchmark._oracle_assay_provenance(
        seed=41,
        max_candidates=None,
        sequences=sequences,
        mode="physical",
    )
    assert physical["validity"] == (
        "privileged_nominal_geometric_controller_not_a_physical_outcome_oracle"
    )


def test_oracle_ranking_rejects_invalid_inputs() -> None:
    sequences = [("hold",), ("forward",)]
    for values in (
        np.asarray([], dtype=np.float64),
        np.asarray([[0.0, 1.0]], dtype=np.float64),
        np.asarray([0.0, math.inf], dtype=np.float64),
    ):
        try:
            benchmark._oracle_ranking(values, sequences)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid oracle costs were accepted: {values!r}")
