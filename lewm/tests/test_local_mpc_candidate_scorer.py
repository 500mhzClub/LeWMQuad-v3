"""Focused tests for the optional ``LocalMPC`` candidate-scoring seam."""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.planning.local_mpc import GoalSpec, LocalMPC, PlannerState  # noqa: E402


class _NeverCalledModel:
    def __getattr__(self, name):
        raise AssertionError(f"model path must be bypassed, attempted {name}")


def _fixture():
    sequences = [("hold", "hold"), ("forward", "hold"), ("left", "forward")]
    action_tensor = torch.zeros(3, 2, 3)
    state = PlannerState(
        image=torch.zeros(3, 4, 4),
        action_history=["left"],
        metadata={"pose_xy": (1.0, 2.0)},
    )
    goal = GoalSpec(torch.ones(3, 4, 4), subgoal_node_id=7)
    return sequences, action_tensor, state, goal


def test_optional_scorer_selects_candidate_and_records_copy_safe_trace():
    sequences, action_tensor, state, goal = _fixture()
    scores = np.array([4.0, 0.25, 2.0], dtype=np.float64)
    received = {}

    def oracle_scorer(got_state, got_goal, got_sequences, got_actions):
        received.update(
            state=got_state,
            goal=got_goal,
            sequences=got_sequences,
            actions=got_actions,
        )
        return scores

    controller = LocalMPC(
        _NeverCalledModel(),
        sequences,
        action_tensor,
        candidate_scorer=oracle_scorer,
    )

    assert controller.last_decision is None
    assert controller.choose(state, goal) == ("forward", 0.25)
    trace = controller.last_decision
    assert trace is not None
    assert received == {
        "state": state,
        "goal": goal,
        "sequences": tuple(sequences),
        "actions": action_tensor,
    }
    assert trace.scorer == "oracle_scorer"
    assert trace.candidate_costs == (4.0, 0.25, 2.0)
    assert trace.sequences == tuple(sequences)
    assert trace.best_index == 1
    assert trace.best_sequence == ("forward", "hold")
    assert trace.primitive == "forward"
    assert trace.cost == 0.25

    scores[1] = 99.0
    sequences[1] = ("mutated", "mutated")
    assert trace.candidate_costs == (4.0, 0.25, 2.0)
    assert trace.best_sequence == ("forward", "hold")
    with pytest.raises(FrozenInstanceError):
        trace.cost = 9.0


def test_candidate_costs_uses_optional_scorer_without_creating_decision_trace():
    sequences, action_tensor, state, goal = _fixture()
    scores = torch.tensor([1.5, -2.0, 3.25], dtype=torch.float64)
    controller = LocalMPC(
        _NeverCalledModel(),
        sequences,
        action_tensor,
        candidate_scorer=lambda *_args: scores,
    )

    costs, names = controller.candidate_costs(state, goal)

    assert costs.dtype == np.float64
    assert np.array_equal(costs, scores.numpy())
    assert names == ["hold", "forward", "left"]
    assert controller.last_decision is None


@pytest.mark.parametrize(
    ("scores", "error", "match"),
    [
        (torch.tensor(1.0), ValueError, "expected shape"),
        (torch.zeros(3, 1), ValueError, "expected shape"),
        (torch.zeros(2), ValueError, "expected shape"),
        (torch.tensor([0.0, float("nan"), 1.0]), ValueError, "non-finite"),
        (torch.tensor([0.0, float("inf"), 1.0]), ValueError, "non-finite"),
        (torch.tensor([False, True, False]), TypeError, "real numeric"),
        (np.array([1.0 + 0.0j, 2.0, 3.0]), TypeError, "real numeric"),
        (None, TypeError, "real numeric"),
    ],
)
def test_optional_scorer_rejects_invalid_candidate_costs(scores, error, match):
    sequences, action_tensor, state, goal = _fixture()
    controller = LocalMPC(
        _NeverCalledModel(),
        sequences,
        action_tensor,
        candidate_scorer=lambda *_args: scores,
    )

    with pytest.raises(error, match=match):
        controller.choose(state, goal)


def test_optional_scorer_rejects_candidate_bank_length_mismatch():
    sequences, action_tensor, state, goal = _fixture()
    controller = LocalMPC(
        _NeverCalledModel(),
        sequences[:-1],
        action_tensor,
        candidate_scorer=lambda *_args: torch.zeros(3),
    )

    with pytest.raises(ValueError, match="sequences and action tensor disagree"):
        controller.choose(state, goal)
