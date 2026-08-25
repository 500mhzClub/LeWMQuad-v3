import math

import numpy as np

from lewm.safety import body_centric_range_coverage_metrics_v1 as M


def _decision(*, top3: float = 1.0) -> dict:
    return {
        "states_retaining_admitted_action": 1,
        "correct_abstentions": 1,
        "selected_h3_route_progress_m": 0.2,
        "normalized_regret": 0.1,
        "best_admissible_top3": top3,
    }


def _next(action_index: int, oracle: bool, predicted: bool) -> dict:
    return {
        "action_index": action_index,
        "controller": "route" if action_index < 12 else "lateral",
        "oracle_contact": oracle,
        "predicted_contact": predicted,
    }


def _current(
    action_index: int,
    *,
    oracle: bool = False,
    predicted: bool = False,
    progress: float = 0.0,
    heading: float = 0.0,
    next_actions: list[dict] | None = None,
) -> dict:
    return {
        "action_index": action_index,
        "controller": "route" if action_index < 12 else "lateral",
        "oracle_contact": oracle,
        "predicted_contact": predicted,
        "h3_progress_m": None if action_index >= 12 else progress,
        "h3_heading_improvement_rad": None if action_index >= 12 else heading,
        "decision_progress_m": progress,
        "next_actions": next_actions,
    }


def test_tie_aware_rank_metrics_are_order_invariant():
    labels = np.asarray([True, False, True, False])
    scores = np.asarray([1.0, 1.0, 0.0, 0.0])
    permutation = np.asarray([1, 0, 3, 2])
    assert M.tie_aware_auc(labels, scores) == 0.5
    assert M.tie_aware_average_precision(labels, scores) == 0.5
    assert M.tie_aware_auc(labels[permutation], scores[permutation]) == 0.5
    assert M.tie_aware_average_precision(labels[permutation], scores[permutation]) == 0.5
    assert math.isclose(
        M.tie_aware_spearman([0, 0, 1, 1], [1, 1, 0, 0]), -1.0
    )


def test_threshold_ties_and_unsupported_are_contact_risk():
    predicted = M.contact_predictions(
        [0.2, 0.200001, 100.0, np.inf],
        0.2,
        unsupported=[False, False, True, False],
    )
    assert predicted.tolist() == [True, False, True, True]


def test_threshold_frontier_prefers_more_conservative_exact_tie():
    # The negative row is unsupported and therefore positive at every
    # threshold.  Thresholds at max(clearance) and just above it have exactly
    # the same first six criteria, so +threshold must choose the latter.
    result = M.enumerate_threshold_frontier(
        [True, True, False],
        [0.1, 0.2, np.inf],
        lambda _threshold: _decision(),
        unsupported=[False, False, True],
    )
    assert result["selected_threshold_m"] > 0.2
    assert result["selected"]["lexicographic_key"][-1] == result["selected_threshold_m"]


def test_threshold_frontier_uses_top3_before_conservative_threshold():
    result = M.enumerate_threshold_frontier(
        [True, True, False],
        [0.1, 0.2, np.inf],
        lambda threshold: _decision(top3=1.0 if threshold == 0.2 else 0.5),
        unsupported=[False, False, True],
    )
    assert result["selected_threshold_m"] == 0.2
    assert result["selected"]["decision"]["best_admissible_top3"] == 1.0


def test_h3_route_order_obeys_point_zero_three_band_heading_then_index():
    rows = [
        _current(4, progress=1.0, heading=0.0),
        _current(2, progress=0.97, heading=0.5),
        _current(1, progress=0.97, heading=0.5),
        _current(3, progress=0.969, heading=100.0),
    ]
    order = M.h3_route_order(rows)
    ordered_actions = [rows[index]["action_index"] for index in order]
    assert ordered_actions[:3] == [1, 2, 4]
    assert ordered_actions[-1] == 3


def test_lateral_fallback_sorts_safe_count_then_action_index():
    rows = [_current(12), _current(13)]
    admitted = {12: True, 13: True}
    assert M.select_action(rows, admitted, {12: 1, 13: 2}) == 13
    assert M.select_action(rows, admitted, {12: 2, 13: 2}) == 12


def test_two_ply_reconstructs_one_and_zero_safe_action_and_abstention():
    viable = {
        "state_id": "viable",
        "family": "family-a",
        "current_actions": [
            _current(
                0,
                progress=0.3,
                next_actions=[
                    _next(0, oracle=False, predicted=False),
                    _next(1, oracle=True, predicted=True),
                ],
            ),
            _current(
                1,
                progress=0.2,
                next_actions=[
                    _next(0, oracle=True, predicted=True),
                    _next(1, oracle=True, predicted=True),
                ],
            ),
        ],
    }
    nonviable = {
        "state_id": "nonviable",
        "family": "family-a",
        "current_actions": [
            _current(
                0,
                predicted=True,
                next_actions=[_next(0, oracle=True, predicted=True)],
            )
        ],
    }
    first = M.reduce_two_ply_state(viable)
    assert first["true_safe_counts"] == {0: 1, 1: 0}
    assert first["predicted_safe_counts"] == {0: 1, 1: 0}
    assert first["selected"] == 0
    assert first["selected_safe_margin_ge_1"]
    assert not first["selected_safe_margin_ge_2"]

    second = M.reduce_two_ply_state(nonviable)
    assert not second["oracle_viable"]
    assert second["selected"] is None
    assert second["correct_abstention"]

    combined = M.reduce_two_ply_states([viable, nonviable])
    assert combined["oracle_viable_states"] == 1
    assert combined["states_retaining_admitted_action"] == 1
    assert combined["correct_abstentions"] == 1
    assert combined["safe_action_count"]["zero_vs_nonzero_accuracy"] == 1.0
    assert combined["safe_action_count"]["margin_ge_1"] == 1
    assert combined["safe_action_count"]["margin_ge_2"] == 0
    assert combined["per_family"]["family-a"]["states"] == 2


def test_missing_successor_set_is_conservative_and_cannot_be_admitted():
    state = {
        "state_id": "missing",
        "family": "family-a",
        "current_actions": [_current(0, predicted=False, next_actions=None)],
    }
    result = M.reduce_two_ply_state(state)
    assert result["predicted_safe_counts"][0] == -1
    assert not result["admitted"][0]
    assert result["selected"] is None
