from __future__ import annotations

import numpy as np

from lewm.benchmarks.traversability_metrics import (
    TraversabilityThresholds,
    evaluate_traversability,
    planned_path_collision_rate,
    select_conservative_thresholds,
)


def _fixture():
    labels = np.array([[[1, 1, 2, 2, 0]]], dtype=np.int64)
    probabilities = np.array(
        [
            [
                [[0.01, 0.02, 0.02, 0.10, 0.90]],
                [[0.98, 0.96, 0.01, 0.05, 0.05]],
                [[0.01, 0.02, 0.97, 0.85, 0.05]],
            ]
        ],
        dtype=np.float64,
    )
    distances = np.array([[[0.5, 1.0, 1.5, 2.5, 3.0]]])
    return probabilities, labels, distances


def test_metrics_separate_planner_admission_and_obstacle_detection() -> None:
    probabilities, labels, distances = _fixture()
    metrics = evaluate_traversability(
        probabilities,
        labels,
        distances,
        thresholds=TraversabilityThresholds(0.9, 0.1, 0.1),
    )

    assert metrics.planner_admitted_free_precision == 1.0
    assert metrics.useful_traversable_recall == 1.0
    assert metrics.obstacle_exclusion_recall_within_range == 1.0
    assert metrics.obstacle_detection_recall_within_range == 1.0
    assert metrics.unknown_admission_rate == 0.0


def test_threshold_selection_prefers_recall_among_passing_candidates() -> None:
    probabilities, labels, distances = _fixture()
    selection = select_conservative_thresholds(
        probabilities,
        labels,
        distances,
        free_probability_candidates=(0.9, 0.97),
        occupied_probability_candidates=(0.1,),
        unknown_probability_candidates=(0.1,),
    )

    assert selection.passing_candidate_count == 2
    assert selection.thresholds.free_probability_min == 0.9
    assert selection.metrics.useful_traversable_recall == 1.0


def test_path_collision_rate_treats_unknown_as_unsafe() -> None:
    labels = np.array([[[1, 1, 2], [1, 0, 1]]], dtype=np.int64)
    paths = (
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 2)),
        ((0, 1, 0), (0, 1, 1)),
    )

    assert planned_path_collision_rate(paths, labels) == 2 / 3
