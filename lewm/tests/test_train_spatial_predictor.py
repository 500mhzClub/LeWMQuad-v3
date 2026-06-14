from __future__ import annotations

from scripts.train_jepa_spatial_predictor import _selection_metrics


def test_spatial_selection_metrics_group_candidates_by_source_state() -> None:
    base_labels = {
        "target_progress_m": 0.0,
        "enters_grid_unsafe": False,
        "ends_grid_unsafe": False,
        "target_recoverable": True,
    }
    rows = [
        {
            "source_index": 4,
            "goal_present": True,
            "goal_frame": "goal.png",
            "is_oracle_candidate": False,
            "consequence_labels": base_labels,
        },
        {
            "source_index": 4,
            "goal_present": True,
            "goal_frame": "goal.png",
            "is_oracle_candidate": True,
            "consequence_labels": {**base_labels, "target_progress_m": 0.2},
        },
    ]

    metrics = _selection_metrics(rows, [2.0, 1.0])

    assert metrics["target_groups"] == 1
    assert metrics["safe_positive_progress_rate"] == 1.0
    assert metrics["oracle_sequence_match_rate"] == 1.0
    assert metrics["minimum_candidates_per_target_group"] == 2
    assert metrics["conditional_on_complete_valid_future_observations"]
