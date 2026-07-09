"""Conservative offline gates for body-inflated occupancy predictions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2


@dataclass(frozen=True)
class TraversabilityThresholds:
    free_probability_min: float
    occupied_probability_max: float
    unknown_probability_max: float
    occupied_detection_min: float = 0.5

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class TraversabilityMetrics:
    admitted_free_count: int
    true_free_count: int
    true_occupied_count: int
    true_occupied_within_range_count: int
    planner_admitted_free_precision: float
    useful_traversable_recall: float
    obstacle_exclusion_recall_within_range: float
    obstacle_detection_recall_within_range: float
    unknown_admission_rate: float
    free_probability_brier: float
    free_probability_ece: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    thresholds: TraversabilityThresholds
    metrics: TraversabilityMetrics
    candidate_count: int
    passing_candidate_count: int


def _validate_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
    distances_m: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    distances_m = np.asarray(distances_m, dtype=np.float64)
    if probabilities.ndim < 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (N, 3, ...)")
    expected = probabilities.shape[:1] + probabilities.shape[2:]
    if labels.shape != expected or distances_m.shape != expected:
        raise ValueError("labels/distances must match probability spatial shape")
    if not np.isfinite(probabilities).all() or not np.isfinite(distances_m).all():
        raise ValueError("probabilities and distances must be finite")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("class probabilities must sum to one")
    if not np.isin(labels, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
        raise ValueError("labels contain an unsupported class")
    valid = np.ones(expected, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != expected:
        raise ValueError("mask must match labels")
    return probabilities, labels, distances_m, valid


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int,
) -> float:
    if probabilities.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(probabilities.size)
    error = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(targets[selected].mean())
        error += float(selected.sum()) / total * abs(confidence - accuracy)
    return float(error)


def evaluate_traversability(
    probabilities: np.ndarray,
    labels: np.ndarray,
    distances_m: np.ndarray,
    *,
    thresholds: TraversabilityThresholds,
    evaluation_mask: np.ndarray | None = None,
    obstacle_range_m: float = 2.0,
    calibration_bins: int = 15,
) -> TraversabilityMetrics:
    """Evaluate the exact probabilities admitted to deterministic planning."""

    thresholds.validate()
    if not math.isfinite(obstacle_range_m) or obstacle_range_m <= 0.0:
        raise ValueError("obstacle_range_m must be positive")
    probabilities, labels, distances_m, valid = _validate_inputs(
        probabilities,
        labels,
        distances_m,
        evaluation_mask,
    )
    unknown_prob = probabilities[:, UNKNOWN_CLASS]
    free_prob = probabilities[:, FREE_CLASS]
    occupied_prob = probabilities[:, OCCUPIED_CLASS]
    admitted = (
        valid
        & (free_prob >= thresholds.free_probability_min)
        & (occupied_prob <= thresholds.occupied_probability_max)
        & (unknown_prob <= thresholds.unknown_probability_max)
    )
    true_free = valid & (labels == FREE_CLASS)
    true_occupied = valid & (labels == OCCUPIED_CLASS)
    true_unknown = valid & (labels == UNKNOWN_CLASS)
    near_occupied = true_occupied & (distances_m <= float(obstacle_range_m))
    detected_occupied = valid & (
        occupied_prob >= thresholds.occupied_detection_min
    )
    known = valid & (labels != UNKNOWN_CLASS)
    known_free_prob = free_prob[known]
    known_free_target = (labels[known] == FREE_CLASS).astype(np.float64)
    return TraversabilityMetrics(
        admitted_free_count=int(admitted.sum()),
        true_free_count=int(true_free.sum()),
        true_occupied_count=int(true_occupied.sum()),
        true_occupied_within_range_count=int(near_occupied.sum()),
        planner_admitted_free_precision=_safe_ratio(
            int((admitted & true_free).sum()), int(admitted.sum())
        ),
        useful_traversable_recall=_safe_ratio(
            int((admitted & true_free).sum()), int(true_free.sum())
        ),
        obstacle_exclusion_recall_within_range=_safe_ratio(
            int((near_occupied & ~admitted).sum()), int(near_occupied.sum())
        ),
        obstacle_detection_recall_within_range=_safe_ratio(
            int((near_occupied & detected_occupied).sum()), int(near_occupied.sum())
        ),
        unknown_admission_rate=_safe_ratio(
            int((admitted & true_unknown).sum()), int(true_unknown.sum())
        ),
        free_probability_brier=(
            0.0
            if known_free_prob.size == 0
            else float(np.square(known_free_prob - known_free_target).mean())
        ),
        free_probability_ece=_binary_ece(
            known_free_prob,
            known_free_target,
            bins=int(calibration_bins),
        ),
    )


def select_conservative_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    distances_m: np.ndarray,
    *,
    free_probability_candidates: Sequence[float],
    occupied_probability_candidates: Sequence[float],
    unknown_probability_candidates: Sequence[float],
    evaluation_mask: np.ndarray | None = None,
    minimum_free_precision: float = 0.99,
    minimum_obstacle_exclusion_recall: float = 0.95,
    minimum_obstacle_detection_recall: float = 0.95,
    obstacle_range_m: float = 2.0,
) -> ThresholdSelection:
    """Choose the highest-recall threshold tuple satisfying safety gates."""

    evaluated: list[tuple[TraversabilityThresholds, TraversabilityMetrics]] = []
    passing: list[tuple[TraversabilityThresholds, TraversabilityMetrics]] = []
    for free_min in free_probability_candidates:
        for occupied_max in occupied_probability_candidates:
            for unknown_max in unknown_probability_candidates:
                thresholds = TraversabilityThresholds(
                    free_probability_min=float(free_min),
                    occupied_probability_max=float(occupied_max),
                    unknown_probability_max=float(unknown_max),
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
    pool = passing if passing else evaluated
    thresholds, metrics = max(
        pool,
        key=lambda item: (
            item[1].useful_traversable_recall,
            item[1].planner_admitted_free_precision,
            item[1].obstacle_detection_recall_within_range,
            -item[0].free_probability_min,
        ),
    )
    return ThresholdSelection(
        thresholds=thresholds,
        metrics=metrics,
        candidate_count=len(evaluated),
        passing_candidate_count=len(passing),
    )


def planned_path_collision_rate(
    paths: Iterable[Sequence[tuple[int, int, int]]],
    labels: np.ndarray,
) -> float:
    """Fraction of proposed paths touching ground-truth occupied/unknown cells."""

    labels = np.asarray(labels, dtype=np.int64)
    path_list = list(paths)
    if not path_list:
        return 0.0
    unsafe = 0
    for path in path_list:
        if not path:
            unsafe += 1
            continue
        if any(
            not (0 <= sample < labels.shape[0] and 0 <= row < labels.shape[1] and 0 <= col < labels.shape[2])
            or labels[sample, row, col] != FREE_CLASS
            for sample, row, col in path
        ):
            unsafe += 1
    return float(unsafe / len(path_list))


__all__ = [
    "FREE_CLASS",
    "OCCUPIED_CLASS",
    "ThresholdSelection",
    "TraversabilityMetrics",
    "TraversabilityThresholds",
    "UNKNOWN_CLASS",
    "evaluate_traversability",
    "planned_path_collision_rate",
    "select_conservative_thresholds",
]
