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


@dataclass(frozen=True)
class _TraversabilityEvaluationContext:
    unknown_probability: np.ndarray
    free_probability: np.ndarray
    occupied_probability: np.ndarray
    valid: np.ndarray
    true_free: np.ndarray
    true_unknown: np.ndarray
    near_occupied: np.ndarray
    true_free_count: int
    true_occupied_count: int
    true_occupied_within_range_count: int
    true_unknown_count: int
    free_probability_brier: float
    free_probability_ece: float


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


def _prepare_evaluation_context(
    probabilities: np.ndarray,
    labels: np.ndarray,
    distances_m: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None,
    obstacle_range_m: float,
    calibration_bins: int,
) -> _TraversabilityEvaluationContext:
    probabilities, labels, distances_m, valid = _validate_inputs(
        probabilities,
        labels,
        distances_m,
        evaluation_mask,
    )
    unknown_probability = probabilities[:, UNKNOWN_CLASS]
    free_probability = probabilities[:, FREE_CLASS]
    occupied_probability = probabilities[:, OCCUPIED_CLASS]
    true_free = valid & (labels == FREE_CLASS)
    true_occupied = valid & (labels == OCCUPIED_CLASS)
    true_unknown = valid & (labels == UNKNOWN_CLASS)
    near_occupied = true_occupied & (distances_m <= float(obstacle_range_m))
    known = valid & (labels != UNKNOWN_CLASS)
    known_free_probability = free_probability[known]
    known_free_target = (labels[known] == FREE_CLASS).astype(np.float64)
    true_free_count = int(true_free.sum())
    true_occupied_count = int(true_occupied.sum())
    true_occupied_within_range_count = int(near_occupied.sum())
    true_unknown_count = int(true_unknown.sum())
    return _TraversabilityEvaluationContext(
        unknown_probability=unknown_probability,
        free_probability=free_probability,
        occupied_probability=occupied_probability,
        valid=valid,
        true_free=true_free,
        true_unknown=true_unknown,
        near_occupied=near_occupied,
        true_free_count=true_free_count,
        true_occupied_count=true_occupied_count,
        true_occupied_within_range_count=true_occupied_within_range_count,
        true_unknown_count=true_unknown_count,
        free_probability_brier=(
            0.0
            if known_free_probability.size == 0
            else float(
                np.square(known_free_probability - known_free_target).mean()
            )
        ),
        free_probability_ece=_binary_ece(
            known_free_probability,
            known_free_target,
            bins=int(calibration_bins),
        ),
    )


def _evaluate_prepared_context(
    context: _TraversabilityEvaluationContext,
    thresholds: TraversabilityThresholds,
) -> TraversabilityMetrics:
    admitted = (
        context.valid
        & (context.free_probability >= thresholds.free_probability_min)
        & (context.occupied_probability <= thresholds.occupied_probability_max)
        & (context.unknown_probability <= thresholds.unknown_probability_max)
    )
    admitted_free_count = int(admitted.sum())
    admitted_true_free_count = int((admitted & context.true_free).sum())
    return TraversabilityMetrics(
        admitted_free_count=admitted_free_count,
        true_free_count=context.true_free_count,
        true_occupied_count=context.true_occupied_count,
        true_occupied_within_range_count=context.true_occupied_within_range_count,
        planner_admitted_free_precision=_safe_ratio(
            admitted_true_free_count,
            admitted_free_count,
        ),
        useful_traversable_recall=_safe_ratio(
            admitted_true_free_count,
            context.true_free_count,
        ),
        obstacle_exclusion_recall_within_range=_safe_ratio(
            int((context.near_occupied & ~admitted).sum()),
            context.true_occupied_within_range_count,
        ),
        obstacle_detection_recall_within_range=_safe_ratio(
            int(
                (
                    context.near_occupied
                    & (
                        context.occupied_probability
                        >= thresholds.occupied_detection_min
                    )
                ).sum()
            ),
            context.true_occupied_within_range_count,
        ),
        unknown_admission_rate=_safe_ratio(
            int((admitted & context.true_unknown).sum()),
            context.true_unknown_count,
        ),
        free_probability_brier=context.free_probability_brier,
        free_probability_ece=context.free_probability_ece,
    )


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
    occupied_detection_probability_candidates: Sequence[float] = (0.5,),
    evaluation_mask: np.ndarray | None = None,
    minimum_free_precision: float = 0.99,
    minimum_obstacle_exclusion_recall: float = 0.95,
    minimum_obstacle_detection_recall: float = 0.95,
    obstacle_range_m: float = 2.0,
) -> ThresholdSelection:
    """Choose the highest-recall threshold tuple satisfying safety gates."""

    evaluated: list[tuple[TraversabilityThresholds, TraversabilityMetrics]] = []
    compatible: list[tuple[TraversabilityThresholds, TraversabilityMetrics]] = []
    passing: list[tuple[TraversabilityThresholds, TraversabilityMetrics]] = []
    context: _TraversabilityEvaluationContext | None = None
    for free_min in free_probability_candidates:
        for occupied_max in occupied_probability_candidates:
            for unknown_max in unknown_probability_candidates:
                for occupied_detection_min in (
                    occupied_detection_probability_candidates
                ):
                    thresholds = TraversabilityThresholds(
                        free_probability_min=float(free_min),
                        occupied_probability_max=float(occupied_max),
                        unknown_probability_max=float(unknown_max),
                        occupied_detection_min=float(occupied_detection_min),
                    )
                    thresholds.validate()
                    if context is None:
                        if (
                            not math.isfinite(obstacle_range_m)
                            or obstacle_range_m <= 0.0
                        ):
                            raise ValueError("obstacle_range_m must be positive")
                        context = _prepare_evaluation_context(
                            probabilities,
                            labels,
                            distances_m,
                            evaluation_mask=evaluation_mask,
                            obstacle_range_m=float(obstacle_range_m),
                            calibration_bins=15,
                        )
                    metrics = _evaluate_prepared_context(context, thresholds)
                    item = (thresholds, metrics)
                    evaluated.append(item)
                    # Admission and detection comparisons are both inclusive.
                    # A compatible operating point cannot classify one cell as
                    # both planner-free and occupied evidence.
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
