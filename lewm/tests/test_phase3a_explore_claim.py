from __future__ import annotations

from lewm.benchmarks.phase3a_explore_claim import (
    action_sequence_prior_predictions,
    compare_explore_claim_summaries,
    egocentric_explore_claim_predictions,
    egocentric_marker_memory_delta,
    egocentric_marker_memory_predictions,
    explore_claim_phase,
    summarize_explore_claim_predictions,
)
from lewm.benchmarks.phase3a_positive_control import action_vector


def _row(
    *,
    source_index: int,
    primitive: str,
    utility: float,
    new_free: float,
    future_marker: bool = False,
    known_before: bool = False,
    claimed: bool = False,
) -> dict:
    return {
        "scene_id": "scene-a",
        "source_index": source_index,
        "primitive_sequence": [primitive, "hold"],
        "consequence_labels": {
            "target_utility": utility,
            "target_new_free_cells": new_free,
            "future_goal_marker_seen": future_marker,
            "goal_known_before_candidate": known_before,
            "goal_claimed": claimed,
        },
    }


def test_explore_claim_phase_splits_source_groups() -> None:
    assert (
        explore_claim_phase(
            [
                _row(source_index=0, primitive="forward", utility=1.0, new_free=3),
                _row(source_index=0, primitive="hold", utility=0.0, new_free=0),
            ]
        )
        == "explore_unseen"
    )
    assert (
        explore_claim_phase(
            [
                _row(source_index=1, primitive="forward", utility=1.0, new_free=1),
                _row(
                    source_index=1,
                    primitive="turn_left",
                    utility=5.0,
                    new_free=0,
                    future_marker=True,
                ),
            ]
        )
        == "discover_visible_marker"
    )
    assert (
        explore_claim_phase(
            [
                _row(
                    source_index=2,
                    primitive="forward",
                    utility=4.0,
                    new_free=0,
                    known_before=True,
                    claimed=True,
                ),
                _row(
                    source_index=2,
                    primitive="turn_right",
                    utility=1.0,
                    new_free=1,
                    known_before=True,
                ),
            ]
        )
        == "claim_after_marker_seen"
    )


def test_explore_claim_summary_compares_predictions_by_phase() -> None:
    rows = [
        _row(source_index=0, primitive="forward", utility=1.0, new_free=3),
        _row(source_index=0, primitive="turn_left", utility=0.0, new_free=0),
        _row(
            source_index=1,
            primitive="forward",
            utility=0.5,
            new_free=1,
        ),
        _row(
            source_index=1,
            primitive="turn_left",
            utility=5.0,
            new_free=0,
            future_marker=True,
        ),
        _row(
            source_index=2,
            primitive="forward",
            utility=4.0,
            new_free=0,
            known_before=True,
            claimed=True,
        ),
        _row(
            source_index=2,
            primitive="turn_right",
            utility=1.0,
            new_free=1,
            known_before=True,
        ),
    ]
    weaker = summarize_explore_claim_predictions(
        rows,
        [0.9, 0.1, 0.8, 0.2, 0.1, 0.7],
    )
    stronger = summarize_explore_claim_predictions(
        rows,
        [0.9, 0.1, 0.2, 0.8, 0.7, 0.1],
    )
    deltas = compare_explore_claim_summaries(
        {
            "memory": stronger,
            "no_memory": weaker,
        }
    )

    assert stronger["source_states"] == 3
    assert stronger["phases"]["explore_unseen"]["primitive_match_rate"] == 1.0
    assert (
        stronger["phases"]["discover_visible_marker"][
            "selected_future_goal_marker_seen_rate"
        ]
        == 1.0
    )
    assert stronger["phases"]["claim_after_marker_seen"]["selected_goal_claimed_rate"] == 1.0
    weaker_claim = weaker["phases"]["claim_after_marker_seen"]
    assert weaker_claim["mean_selected_prediction"] == 0.7
    assert weaker_claim["mean_oracle_prediction"] == 0.1
    assert weaker_claim["mean_best_claim_prediction"] == 0.1
    assert weaker_claim["mean_selected_minus_oracle_prediction"] == 0.6
    assert weaker_claim["mean_selected_minus_best_claim_prediction"] == 0.6
    assert weaker_claim["selected_prediction_above_oracle_rate"] == 1.0
    assert weaker_claim["selected_prediction_above_best_claim_rate"] == 1.0
    assert (
        deltas["memory_minus_no_memory"]["discover_visible_marker"][
            "primitive_match_rate_delta"
        ]
        == 1.0
    )
    assert (
        deltas["memory_minus_no_memory"]["claim_after_marker_seen"][
            "mean_target_utility_regret_delta"
        ]
        == -3.0
    )


def test_action_sequence_prior_predictions_use_train_mean_utility() -> None:
    train_rows = [
        _row(source_index=0, primitive="forward", utility=1.0, new_free=0),
        _row(source_index=1, primitive="forward", utility=3.0, new_free=0),
        _row(source_index=2, primitive="turn_left", utility=0.5, new_free=0),
    ]
    validation_rows = [
        _row(source_index=0, primitive="forward", utility=0.0, new_free=0),
        _row(source_index=0, primitive="turn_left", utility=0.0, new_free=0),
        _row(source_index=0, primitive="turn_right", utility=0.0, new_free=0),
    ]

    assert action_sequence_prior_predictions(train_rows, validation_rows) == [
        2.0,
        0.5,
        0.0,
    ]


def test_egocentric_marker_memory_scores_remembered_marker_without_sim_state() -> None:
    empty = [[[0.72 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    marker_ahead = [[row.copy() for row in channel] for channel in empty]
    marker_ahead[0][0][1] = 0.10
    marker_ahead[1][0][1] = 0.85
    marker_ahead[2][0][1] = 0.18
    forward = {
        "history_observations_rgb": [marker_ahead],
        "history_actions": [],
        "start_observation_rgb": empty,
        "active_blocks": [list(action_vector("forward"))],
    }
    hold = {
        "history_observations_rgb": [marker_ahead],
        "history_actions": [],
        "start_observation_rgb": empty,
        "active_blocks": [list(action_vector("hold"))],
    }

    forward_score, hold_score = egocentric_marker_memory_predictions([forward, hold])
    forward_valid, forward_ahead, forward_lateral = egocentric_marker_memory_delta(
        forward
    )
    hold_valid, hold_ahead, hold_lateral = egocentric_marker_memory_delta(hold)

    assert forward_score > hold_score
    assert (forward_valid, forward_ahead, forward_lateral) == (True, 0, 0)
    assert (hold_valid, hold_ahead, hold_lateral) == (True, 1, 0)


def test_egocentric_explore_claim_scores_frontier_before_marker() -> None:
    free = [[[0.72 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    forward = {
        "history_observations_rgb": [],
        "history_actions": [],
        "history_goal_beacon": False,
        "current_goal_beacon": False,
        "start_observation_rgb": free,
        "active_blocks": [list(action_vector("forward"))],
    }
    hold = {
        **forward,
        "active_blocks": [list(action_vector("hold"))],
    }
    turn_left = {
        **forward,
        "active_blocks": [list(action_vector("turn_left"))],
    }

    forward_score, hold_score, turn_score = egocentric_explore_claim_predictions(
        [forward, hold, turn_left]
    )

    assert forward_score > hold_score
    assert forward_score > turn_score


def test_egocentric_explore_claim_switches_to_marker_claiming() -> None:
    free = [[[0.72 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    marker_ahead = [[row.copy() for row in channel] for channel in free]
    marker_ahead[0][0][1] = 0.10
    marker_ahead[1][0][1] = 0.85
    marker_ahead[2][0][1] = 0.18
    forward = {
        "history_observations_rgb": [marker_ahead],
        "history_actions": [],
        "history_goal_beacon": False,
        "current_goal_beacon": False,
        "start_observation_rgb": free,
        "active_blocks": [list(action_vector("forward"))],
    }
    hold = {
        **forward,
        "active_blocks": [list(action_vector("hold"))],
    }

    forward_score, hold_score = egocentric_explore_claim_predictions([forward, hold])

    assert forward_score > hold_score
