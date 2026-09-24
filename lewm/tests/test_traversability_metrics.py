from __future__ import annotations

import numpy as np
import pytest

from lewm.benchmarks import traversability_metrics as metrics_module
from lewm.benchmarks.traversability_metrics import (
    ThresholdSelection,
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


def _select_conservative_thresholds_reference(
    probabilities: np.ndarray,
    labels: np.ndarray,
    distances_m: np.ndarray,
    *,
    free_probability_candidates,
    occupied_probability_candidates,
    unknown_probability_candidates,
    occupied_detection_probability_candidates=(0.5,),
    evaluation_mask=None,
    minimum_free_precision: float = 0.99,
    minimum_obstacle_exclusion_recall: float = 0.95,
    minimum_obstacle_detection_recall: float = 0.95,
    obstacle_range_m: float = 2.0,
) -> ThresholdSelection:
    """Deliberately simple pre-optimization selector used as an exact oracle."""

    evaluated = []
    compatible = []
    passing = []
    for free_min in free_probability_candidates:
        for occupied_max in occupied_probability_candidates:
            for unknown_max in unknown_probability_candidates:
                for occupied_detection_min in occupied_detection_probability_candidates:
                    thresholds = TraversabilityThresholds(
                        free_probability_min=float(free_min),
                        occupied_probability_max=float(occupied_max),
                        unknown_probability_max=float(unknown_max),
                        occupied_detection_min=float(occupied_detection_min),
                    )
                    metrics = evaluate_traversability(
                        probabilities,
                        labels,
                        distances_m,
                        thresholds=thresholds,
                        evaluation_mask=evaluation_mask,
                        obstacle_range_m=obstacle_range_m,
                    )
                    item = (thresholds, metrics)
                    evaluated.append(item)
                    if (
                        thresholds.occupied_detection_min
                        <= thresholds.occupied_probability_max
                    ):
                        continue
                    compatible.append(item)
                    if (
                        metrics.admitted_free_count > 0
                        and metrics.planner_admitted_free_precision
                        >= float(minimum_free_precision)
                        and metrics.obstacle_exclusion_recall_within_range
                        >= float(minimum_obstacle_exclusion_recall)
                        and metrics.obstacle_detection_recall_within_range
                        >= float(minimum_obstacle_detection_recall)
                    ):
                        passing.append(item)
    if not evaluated:
        raise ValueError("threshold candidate grid is empty")
    if not compatible:
        raise ValueError(
            "threshold candidate grid has no disjoint admission/detection tuple"
        )
    pool = passing if passing else compatible
    thresholds, metrics = max(
        pool,
        key=lambda item: (
            item[1].useful_traversable_recall,
            item[1].planner_admitted_free_precision,
            item[1].obstacle_detection_recall_within_range,
            item[0].occupied_detection_min,
            -item[0].free_probability_min,
        ),
    )
    return ThresholdSelection(
        thresholds=thresholds,
        metrics=metrics,
        candidate_count=len(evaluated),
        passing_candidate_count=len(passing),
    )


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


@pytest.mark.parametrize(
    ("seed", "mask_mode"),
    (
        (3, "none"),
        (11, "random"),
        (19, "empty"),
        (29, "unknown_only"),
        (37, "occupied_only"),
    ),
)
def test_threshold_selection_matches_simple_reference_on_randomized_inputs(
    seed: int,
    mask_mode: str,
) -> None:
    rng = np.random.default_rng(seed)
    probabilities = np.moveaxis(
        rng.dirichlet((0.7, 1.1, 0.9), size=(3, 4, 5)),
        -1,
        1,
    )
    labels = rng.integers(0, 3, size=(3, 4, 5), dtype=np.int64)
    distances = rng.uniform(0.0, 3.5, size=labels.shape)
    probabilities[0, :, 0, 0] = (0.1, 0.7, 0.2)
    labels[0, 0, 0] = 1
    distances[0, 0, 0] = 1.25

    if mask_mode == "none":
        evaluation_mask = None
    elif mask_mode == "random":
        evaluation_mask = rng.random(labels.shape) >= 0.35
    elif mask_mode == "empty":
        evaluation_mask = np.zeros(labels.shape, dtype=bool)
    elif mask_mode == "unknown_only":
        labels.fill(0)
        evaluation_mask = np.ones(labels.shape, dtype=bool)
    elif mask_mode == "occupied_only":
        labels.fill(2)
        evaluation_mask = np.ones(labels.shape, dtype=bool)
    else:  # pragma: no cover - parameterization is exhaustive
        raise AssertionError(mask_mode)

    kwargs = {
        "free_probability_candidates": (0.7, 0.4, 0.7, 0.9),
        "occupied_probability_candidates": (0.2, 0.05, 0.4, 0.2),
        "unknown_probability_candidates": (0.1, 0.35, 0.1),
        "occupied_detection_probability_candidates": (0.1, 0.3, 0.5, 0.3),
        "evaluation_mask": evaluation_mask,
        "minimum_free_precision": 0.6,
        "minimum_obstacle_exclusion_recall": 0.4,
        "minimum_obstacle_detection_recall": 0.2,
        "obstacle_range_m": 1.5,
    }

    expected = _select_conservative_thresholds_reference(
        probabilities,
        labels,
        distances,
        **kwargs,
    )
    actual = select_conservative_thresholds(
        probabilities,
        labels,
        distances,
        **kwargs,
    )

    assert actual == expected


def test_threshold_selection_prepares_invariant_metrics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities, labels, distances = _fixture()
    calls = {"validate": 0, "ece": 0}
    original_validate = metrics_module._validate_inputs
    original_ece = metrics_module._binary_ece

    def counted_validate(*args, **kwargs):
        calls["validate"] += 1
        return original_validate(*args, **kwargs)

    def counted_ece(*args, **kwargs):
        calls["ece"] += 1
        return original_ece(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "_validate_inputs", counted_validate)
    monkeypatch.setattr(metrics_module, "_binary_ece", counted_ece)

    selection = select_conservative_thresholds(
        probabilities,
        labels,
        distances,
        free_probability_candidates=(0.5, 0.9),
        occupied_probability_candidates=(0.05, 0.1),
        unknown_probability_candidates=(0.05, 0.1),
        occupied_detection_probability_candidates=(0.2, 0.5),
    )

    assert selection.candidate_count == 16
    assert calls == {"validate": 1, "ece": 1}


def test_threshold_selection_defaults_to_legacy_detection_threshold() -> None:
    probabilities, labels, distances = _fixture()
    selection = select_conservative_thresholds(
        probabilities,
        labels,
        distances,
        free_probability_candidates=(0.9,),
        occupied_probability_candidates=(0.1,),
        unknown_probability_candidates=(0.1,),
    )

    assert selection.candidate_count == 1
    assert selection.thresholds.occupied_detection_min == 0.5


def test_threshold_selection_prefers_highest_equally_detecting_threshold() -> None:
    probabilities, labels, distances = _fixture()
    selection = select_conservative_thresholds(
        probabilities,
        labels,
        distances,
        free_probability_candidates=(0.9,),
        occupied_probability_candidates=(0.1,),
        unknown_probability_candidates=(0.1,),
        occupied_detection_probability_candidates=(0.2, 0.8, 0.5),
    )

    assert selection.passing_candidate_count == 3
    assert selection.metrics.obstacle_detection_recall_within_range == 1.0
    assert selection.thresholds.occupied_detection_min == 0.8


def test_threshold_selection_rejects_only_overlapping_operating_points() -> None:
    probabilities, labels, distances = _fixture()
    with pytest.raises(ValueError, match="no disjoint admission/detection tuple"):
        select_conservative_thresholds(
            probabilities,
            labels,
            distances,
            free_probability_candidates=(0.9,),
            occupied_probability_candidates=(0.2,),
            unknown_probability_candidates=(0.1,),
            occupied_detection_probability_candidates=(0.1, 0.2),
        )


def test_path_collision_rate_treats_unknown_as_unsafe() -> None:
    labels = np.array([[[1, 1, 2], [1, 0, 1]]], dtype=np.int64)
    paths = (
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 2)),
        ((0, 1, 0), (0, 1, 1)),
    )

    assert planned_path_collision_rate(paths, labels) == 2 / 3
