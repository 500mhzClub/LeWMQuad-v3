"""Pure metrics for ``PLAN_AWARE_MONOTONE_JEPA_COST_V1``.

This module is intentionally unable to discover scientific artifacts.  It
accepts scalar rows and already-computed scores, performs deterministic route
reductions, and returns JSON-compatible dictionaries.  In particular, it does
not open a panel, checkpoint, latent tensor, or outcome ledger.

The ranker score convention is *higher is better*.  Contact and successor
viability are used only to form the three frozen evaluation populations and to
describe a selected candidate.  They never enter the route preference or any
training utility.  The route preference reuses the predecessor's exact
completion, distance-progress, then heading-progress tuple to construct a
population-conditioned margin-Borda utility; completion is not a separate head
or regression target.  Pairwise and rank metrics use that same Borda utility.
Progress correlation and normalized progress regret remain separate diagnostics.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
from typing import Any

import numpy as np
from scipy import stats

from lewm.safety import jepa_local_waypoint_planning_cost_metrics_v1 as PREDECESSOR


EXPERIMENT_ID = "PLAN_AWARE_MONOTONE_JEPA_COST_V1"

POPULATION_IDS = PREDECESSOR.POPULATION_IDS
(
    ALL_CANDIDATES,
    ORACLE_CONTACT_FREE,
    ORACLE_VIABILITY_ADMISSIBLE,
) = POPULATION_IDS
FAMILY_IDS = PREDECESSOR.FAMILY_IDS

DISTANCE_PREFERENCE_MARGIN_M = PREDECESSOR.DISTANCE_PREFERENCE_MARGIN_M
HEADING_PREFERENCE_MARGIN_RAD = PREDECESSOR.HEADING_PREFERENCE_MARGIN_RAD
SCORE_TIE_TOLERANCE = PREDECESSOR.COST_TIE_TOLERANCE
ROUTE_UTILITY_TIE_TOLERANCE = 1e-12
NUMERIC_EPSILON = PREDECESSOR.NUMERIC_EPSILON

BOOTSTRAP_DRAWS = PREDECESSOR.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = PREDECESSOR.BOOTSTRAP_SEED

DERANGEMENT_NAMESPACE = "SHA256(contract_digest_utf8,NUL,state_id_utf8)/CYCLIC_SHIFT_V1"

TRUE_GATE_THRESHOLDS = {
    "pairwise_accuracy_minimum": 0.75,
    "spearman_minimum": 0.60,
    "normalized_regret_maximum": 0.20,
    "best_route_top3_minimum": 0.75,
    "selected_progress_ratio_minimum": 0.80,
}
DERANGEMENT_MATERIALITY_THRESHOLDS = {
    "pairwise_accuracy_loss_minimum": 0.05,
    "selected_progress_ratio_loss_minimum": 0.10,
    "normalized_regret_worsening_minimum": 0.05,
    "pairwise_accuracy_opposing_improvement_minimum": 0.05,
    "selected_progress_ratio_opposing_improvement_minimum": 0.10,
    "normalized_regret_opposing_improvement_minimum": 0.05,
}
ADVERSE_DOWNRANKING_OUTCOMES = (
    "immediate_contact",
    "successor_nonviable",
    "stuck",
)
INCREMENTAL_ROUTE_VALUE_THRESHOLDS = {
    "selected_progress_gain_m_minimum": 0.02,
    "oracle_normalized_progress_gain_minimum": 0.05,
    "normalized_regret_reduction_minimum": 0.03,
    "families_improved_or_tied_minimum": 3,
    "population_progress_loss_maximum": 0.02,
}
PREDICTED_GATE_THRESHOLDS = {
    "pairwise_accuracy_minimum": 0.70,
    "normalized_regret_maximum": 0.25,
    "best_route_top3_minimum": 0.75,
    "selected_progress_ratio_minimum": 0.75,
    "true_selected_progress_retention_minimum": 0.85,
}
PROPRIO_CONTRIBUTION_THRESHOLDS = {
    "pairwise_accuracy_gain": 0.05,
    "selected_progress_ratio_gain": 0.05,
    "normalized_regret_reduction": 0.03,
    "best_route_top3_gain": 0.125,
    "minimum_trigger_count": 2,
}
SUBSTITUTION_THRESHOLDS = {
    "visual_pairwise_loss_strict_maximum": 0.03,
    "visual_progress_loss_strict_maximum": 0.05,
    "proprio_pairwise_loss_minimum": 0.10,
    "proprio_progress_loss_minimum": 0.15,
    "proprio_regret_worsening_minimum": 0.10,
    "differential_pairwise_loss_minimum": 0.05,
    "differential_progress_loss_minimum": 0.10,
    "differential_regret_worsening_minimum": 0.05,
}

PRIMARY_CLASSIFICATIONS = (
    "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL",
    "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO",
    "KINEMATIC_BASELINE_DOMINANT",
    "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
)


class PlanAwareMetricsError(ValueError):
    """Raised for incomplete, ambiguous, or non-finite metric inputs."""


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PlanAwareMetricsError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PlanAwareMetricsError(f"{name} must be finite")
    return result


def _optional_finite(value: Any, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise PlanAwareMetricsError(f"{name} must be Boolean")
    return bool(value)


def _candidate_index(row: Mapping[str, Any]) -> int:
    value = row.get("candidate_index")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PlanAwareMetricsError("candidate_index must be an integer")
    result = int(value)
    if result < 0:
        raise PlanAwareMetricsError("candidate_index must be non-negative")
    return result


def _progress(row: Mapping[str, Any]) -> float:
    for key in ("p_d", "p_d_m", "distance_progress_m", "route_progress_m"):
        if key in row:
            return _finite(row[key], name=key)
    raise PlanAwareMetricsError("candidate lacks route distance progress")


def _heading(row: Mapping[str, Any]) -> float:
    for key in ("p_theta", "p_theta_rad", "heading_progress_rad"):
        if key in row:
            return _finite(row[key], name=key)
    raise PlanAwareMetricsError("candidate lacks route heading progress")


def _completed(row: Mapping[str, Any]) -> bool:
    if "completed" not in row:
        raise PlanAwareMetricsError("candidate lacks completed route component")
    return _strict_bool(row["completed"], name="completed")


def _contact(row: Mapping[str, Any]) -> bool:
    for key in ("immediate_contact_h1", "oracle_contact", "contact"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanAwareMetricsError("candidate lacks immediate-contact population label")


def _successor_viable(row: Mapping[str, Any]) -> bool:
    if "successor_safe_action_count" in row:
        value = row["successor_safe_action_count"]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PlanAwareMetricsError("successor_safe_action_count must be an integer")
        if not 0 <= int(value) <= 9:
            raise PlanAwareMetricsError("successor_safe_action_count must be in [0,9]")
        return int(value) > 0
    for key in ("successor_viable", "successor_has_safe_action"):
        if key in row:
            return _strict_bool(row[key], name=key)
    raise PlanAwareMetricsError("candidate lacks successor-viability label")


def _stuck(row: Mapping[str, Any]) -> bool:
    for key in ("stuck", "oracle_stuck"):
        if key in row:
            return _strict_bool(row[key], name=key)
    return False


def route_only_preference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    distance_margin_m: float = DISTANCE_PREFERENCE_MARGIN_M,
    heading_margin_rad: float = HEADING_PREFERENCE_MARGIN_RAD,
) -> int:
    """Return +1 for a route-preferred left candidate and -1 for right.

    This comparator contains no contact, viability, stuck, or prior aggregate-
    scorer-utility target.  It reuses the predecessor's prospectively frozen
    local route tuple: completion first, distance second, and heading only
    inside the distance margin.
    """

    distance_margin = _finite(distance_margin_m, name="distance_margin_m")
    heading_margin = _finite(heading_margin_rad, name="heading_margin_rad")
    if distance_margin < 0.0 or heading_margin < 0.0:
        raise PlanAwareMetricsError("route margins must be non-negative")
    left_completed = _completed(left)
    right_completed = _completed(right)
    if left_completed != right_completed:
        return 1 if left_completed else -1
    distance_delta = _progress(left) - _progress(right)
    if abs(distance_delta) > distance_margin:
        return 1 if distance_delta > 0.0 else -1
    heading_delta = _heading(left) - _heading(right)
    if abs(heading_delta) > heading_margin:
        return 1 if heading_delta > 0.0 else -1
    return 0


def route_only_order(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return input positions in the deterministic realised route order."""

    rows = list(candidates)
    indices = [_candidate_index(row) for row in rows]
    if len(indices) != len(set(indices)):
        raise PlanAwareMetricsError("candidate indices must be unique")
    remaining = list(range(len(rows)))
    order: list[int] = []
    while remaining:
        preferred_completion = any(_completed(rows[position]) for position in remaining)
        completion_eligible = [
            position
            for position in remaining
            if _completed(rows[position]) == preferred_completion
        ]
        best_distance = max(
            _progress(rows[position]) for position in completion_eligible
        )
        distance_eligible = [
            position
            for position in completion_eligible
            if best_distance - _progress(rows[position])
            <= DISTANCE_PREFERENCE_MARGIN_M
        ]
        best_heading = max(_heading(rows[position]) for position in distance_eligible)
        heading_eligible = [
            position
            for position in distance_eligible
            if best_heading - _heading(rows[position])
            <= HEADING_PREFERENCE_MARGIN_RAD
        ]
        chosen = min(heading_eligible, key=lambda position: indices[position])
        order.append(chosen)
        remaining.remove(chosen)
    return order


def route_only_margin_borda_utility(
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return route-only Borda utility ``(wins + .5 ties)/(N-1)``."""

    rows = list(candidates)
    if not rows:
        return np.empty(0, dtype=np.float64)
    if len(rows) == 1:
        return np.ones(1, dtype=np.float64)
    result = np.zeros(len(rows), dtype=np.float64)
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            preference = route_only_preference(rows[left], rows[right])
            if preference > 0:
                result[left] += 1.0
            elif preference < 0:
                result[right] += 1.0
            else:
                result[left] += 0.5
                result[right] += 0.5
    return result / (len(rows) - 1)


def _score_order(
    scores: np.ndarray,
    candidate_indices: Sequence[int],
    *,
    tolerance: float,
) -> list[int]:
    remaining = list(range(len(scores)))
    result: list[int] = []
    while remaining:
        maximum = max(float(scores[position]) for position in remaining)
        tied = [
            position
            for position in remaining
            if maximum - float(scores[position]) <= tolerance
        ]
        chosen = min(tied, key=lambda position: candidate_indices[position])
        result.append(chosen)
        remaining.remove(chosen)
    return result


def route_ordering_metrics(
    candidates: Sequence[Mapping[str, Any]],
    scores: Any,
    *,
    score_tie_tolerance: float = SCORE_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate one already-filtered state using higher-is-better scores."""

    rows = list(candidates)
    numeric = np.asarray(scores, dtype=np.float64)
    if numeric.shape != (len(rows),) or (numeric.size and not np.isfinite(numeric).all()):
        raise PlanAwareMetricsError("scores must be one finite scalar per candidate")
    tolerance = _finite(score_tie_tolerance, name="score_tie_tolerance")
    if tolerance < 0.0:
        raise PlanAwareMetricsError("score tie tolerance must be non-negative")
    indices = [_candidate_index(row) for row in rows]
    if len(indices) != len(set(indices)):
        raise PlanAwareMetricsError("candidate indices must be unique")
    if not rows:
        return {
            "candidates": 0,
            "ordered_pairs": 0,
            "pairwise_correct_credit": 0.0,
            "pairwise_accuracy": None,
            "spearman_rho": None,
            "route_progress_spearman": None,
            "kendall_tau_b": None,
            "oracle_best_candidate_index": None,
            "oracle_best_candidate_indices": [],
            "selected_candidate_index": None,
            "best_route_top1": None,
            "best_route_top3": None,
            "mean_reciprocal_rank": None,
            "mrr": None,
            "mean_best_route_rank": None,
            "mean_rank": None,
            "score_minimum": None,
            "score_maximum": None,
            "score_spread": None,
            "score_pair_count": 0,
            "score_tie_count": 0,
            "score_tie_rate": None,
            "tie_rate": None,
            "ranked_candidate_indices": [],
            "ideal_route_borda_order": [],
            "oracle_realised_order": [],
            "route_borda_utility": {},
        }

    # The population-conditioned margin-Borda utility is the frozen learning
    # target and therefore also the ordering authority at evaluation.  Direct
    # completion/distance/heading preferences are only the primitives from
    # which this utility is constructed.
    utilities = route_only_margin_borda_utility(rows)
    correct = 0.0
    ordered_pairs = 0
    score_pairs = 0
    score_ties = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            score_pairs += 1
            delta = float(numeric[left] - numeric[right])
            if abs(delta) <= tolerance:
                score_ties += 1
            utility_delta = float(utilities[left] - utilities[right])
            if abs(utility_delta) <= ROUTE_UTILITY_TIE_TOLERANCE:
                continue
            ordered_pairs += 1
            if abs(delta) <= tolerance:
                correct += 0.5
            elif (utility_delta > 0.0 and delta > 0.0) or (
                utility_delta < 0.0 and delta < 0.0
            ):
                correct += 1.0

    progress = np.asarray([_progress(row) for row in rows], dtype=np.float64)
    if (
        len(rows) >= 2
        and np.unique(numeric).size >= 2
        and np.unique(progress).size >= 2
    ):
        spearman = float(stats.spearmanr(numeric, progress).statistic)
        kendall = float(stats.kendalltau(numeric, progress, variant="b").statistic)
        spearman_value = spearman if math.isfinite(spearman) else None
        kendall_value = kendall if math.isfinite(kendall) else None
    else:
        spearman_value = kendall_value = None

    ideal_borda_order = _score_order(
        utilities,
        indices,
        tolerance=ROUTE_UTILITY_TIE_TOLERANCE,
    )
    ranked = _score_order(numeric, indices, tolerance=tolerance)
    best_utility = float(np.max(utilities))
    oracle_best_positions = [
        position
        for position, utility in enumerate(utilities)
        if best_utility - float(utility) <= ROUTE_UTILITY_TIE_TOLERANCE
    ]
    oracle_best_position = ideal_borda_order[0]
    best_rank = 1 + min(ranked.index(position) for position in oracle_best_positions)
    score_minimum = float(np.min(numeric))
    score_maximum = float(np.max(numeric))
    return {
        "candidates": len(rows),
        "ordered_pairs": ordered_pairs,
        "pairwise_correct_credit": correct,
        "pairwise_accuracy": correct / ordered_pairs if ordered_pairs else None,
        "spearman_rho": spearman_value,
        "route_progress_spearman": spearman_value,
        "kendall_tau_b": kendall_value,
        "oracle_best_candidate_index": indices[oracle_best_position],
        "oracle_best_candidate_indices": sorted(
            indices[position] for position in oracle_best_positions
        ),
        "selected_candidate_index": indices[ranked[0]],
        "best_route_top1": best_rank == 1,
        "best_route_top3": best_rank <= 3,
        "mean_reciprocal_rank": 1.0 / best_rank,
        "mrr": 1.0 / best_rank,
        "mean_best_route_rank": float(best_rank),
        "mean_rank": float(best_rank),
        "best_route_rank": best_rank,
        "score_minimum": score_minimum,
        "score_maximum": score_maximum,
        "score_spread": score_maximum - score_minimum,
        "score_pair_count": score_pairs,
        "score_tie_count": score_ties,
        "score_tie_rate": score_ties / score_pairs if score_pairs else None,
        "tie_rate": score_ties / score_pairs if score_pairs else None,
        "ranked_candidate_indices": [indices[position] for position in ranked],
        "ideal_route_borda_order": [
            indices[position] for position in ideal_borda_order
        ],
        # Backward-compatible field name with prospectively corrected Borda
        # semantics; no predecessor metric is imported from this value.
        "oracle_realised_order": [
            indices[position] for position in ideal_borda_order
        ],
        "route_borda_utility": {
            str(indices[position]): float(utilities[position])
            for position in range(len(rows))
        },
    }


def _aligned_scores(
    candidates: Sequence[Mapping[str, Any]], scores: Any
) -> np.ndarray:
    rows = list(candidates)
    if isinstance(scores, Mapping):
        by_index = {int(key): value for key, value in scores.items()}
        expected = {_candidate_index(row) for row in rows}
        if set(by_index) != expected:
            raise PlanAwareMetricsError("score-map candidate identities do not match")
        value = [by_index[_candidate_index(row)] for row in rows]
    else:
        value = scores
    numeric = np.asarray(value, dtype=np.float64)
    if numeric.shape != (len(rows),) or (numeric.size and not np.isfinite(numeric).all()):
        raise PlanAwareMetricsError("scores must align with candidates")
    return numeric


def evaluate_state_population(
    candidates: Sequence[Mapping[str, Any]],
    scores: Any,
    *,
    state_id: str,
    family: str,
    role: str,
    source_id: str,
    population_id: str,
) -> dict[str, Any]:
    """Evaluate ranking and selected outcomes for one population of one state."""

    if not all(isinstance(value, str) and value for value in (state_id, family, role, source_id)):
        raise PlanAwareMetricsError("state, family, role, and source IDs must be nonempty")
    rows = list(candidates)
    all_scores = _aligned_scores(rows, scores)
    try:
        filtered = PREDECESSOR.filter_population(rows, population_id)
    except PREDECESSOR.PlanningCostMetricsError as exc:
        raise PlanAwareMetricsError(str(exc)) from exc
    position_by_candidate = {
        _candidate_index(row): position for position, row in enumerate(rows)
    }
    filtered_scores = np.asarray(
        [all_scores[position_by_candidate[_candidate_index(row)]] for row in filtered],
        dtype=np.float64,
    )
    ordering = route_ordering_metrics(filtered, filtered_scores)
    by_index = {_candidate_index(row): row for row in filtered}
    selected_index = ordering["selected_candidate_index"]
    if selected_index is None:
        selected_progress = selected_heading = 0.0
        selected_combined_utility = None
        selected_completed = False
        selected_contact = selected_nonviable = selected_stuck = False
        best_progress = 0.0
        regret = None
    else:
        selected = by_index[int(selected_index)]
        selected_progress = _progress(selected)
        selected_heading = _heading(selected)
        selected_completed = _completed(selected)
        selected_contact = _contact(selected)
        selected_nonviable = not _successor_viable(selected)
        selected_stuck = _stuck(selected)
        selected_combined_utility = float(
            ordering["route_borda_utility"][str(int(selected_index))]
        )
        progress_values = np.asarray([_progress(row) for row in filtered], np.float64)
        # Route-progress regret remains a separate diagnostic from the Borda
        # ideal used by pairwise/top-k/rank metrics.
        best_progress = float(np.max(progress_values))
        span = float(np.max(progress_values) - np.min(progress_values))
        if span <= SCORE_TIE_TOLERANCE:
            if abs(best_progress - selected_progress) > SCORE_TIE_TOLERANCE:
                raise PlanAwareMetricsError("zero progress span has unequal route progress")
            regret = 0.0
        else:
            # Retained exactly for comparability with the predecessor metric.
            regret = (best_progress - selected_progress) / span
    abstained = selected_index is None
    return {
        "schema": "plan_aware_monotone_state_population_metrics_v1",
        "state_id": state_id,
        "family": family,
        "role": role,
        "source_id": source_id,
        "population_id": population_id,
        **ordering,
        "abstained": abstained,
        "abstention": abstained,
        "selected_completed": selected_completed,
        "waypoint_completion": selected_completed,
        "selected_route_progress_m": selected_progress,
        "selected_heading_progress_rad": selected_heading,
        "selected_combined_route_utility": selected_combined_utility,
        "selected_immediate_contact_h1": selected_contact,
        "selected_nonviable_successor": selected_nonviable,
        "selected_stuck": selected_stuck,
        "oracle_best_route_progress_m": best_progress,
        "normalized_regret": regret,
    }


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def aggregate_state_metrics(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate complete per-state metrics without candidate-row weighting."""

    rows = list(states)
    if not rows:
        return {
            "states": 0,
            "candidates": 0,
            "ordered_pairs": 0,
            "pairwise_correct_credit": 0.0,
            "pairwise_accuracy": None,
            "spearman_rho": None,
            "route_progress_spearman": None,
            "kendall_tau_b": None,
            "best_route_top1_rate": None,
            "best_route_top3_rate": None,
            "best_route_top1": None,
            "best_route_top3": None,
            "mean_reciprocal_rank": None,
            "mrr": None,
            "mean_best_route_rank": None,
            "mean_rank": None,
            "score_spread": None,
            "score_pair_count": 0,
            "score_tie_count": 0,
            "score_tie_rate": None,
            "tie_rate": None,
            "normalized_regret": None,
            "selected_route_progress_m_sum": 0.0,
            "selected_route_progress_m_mean": 0.0,
            "selected_heading_progress_rad_sum": 0.0,
            "selected_heading_progress_rad_mean": 0.0,
            "selected_combined_route_utility_sum": 0.0,
            "selected_combined_route_utility_mean": None,
            "selected_combined_route_utility_count": 0,
            "oracle_best_route_progress_m_sum": 0.0,
            "selected_progress_ratio": 0.0,
            "selected_immediate_contacts_h1": 0,
            "selected_nonviable_successors": 0,
            "selected_stuck": 0,
            "selected_completions": 0,
            "waypoint_completion_rate": 0.0,
            "abstentions": 0,
            "all_comparable_score_pairs_tied": True,
            "all_nonabstaining_score_spreads_within_tolerance": True,
            "complete_family_collapse": True,
        }
    state_ids = [str(row["state_id"]) for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise PlanAwareMetricsError("aggregate state IDs must be unique")
    ordered_pairs = sum(int(row["ordered_pairs"]) for row in rows)
    correct = sum(float(row["pairwise_correct_credit"]) for row in rows)
    score_pairs = sum(int(row["score_pair_count"]) for row in rows)
    score_ties = sum(int(row["score_tie_count"]) for row in rows)
    selected_sum = float(sum(float(row["selected_route_progress_m"]) for row in rows))
    selected_heading_sum = float(
        sum(float(row["selected_heading_progress_rad"]) for row in rows)
    )
    selected_utilities = [
        float(row["selected_combined_route_utility"])
        for row in rows
        if row["selected_combined_route_utility"] is not None
    ]
    best_sum = float(sum(float(row["oracle_best_route_progress_m"]) for row in rows))
    pairwise = correct / ordered_pairs if ordered_pairs else None
    top3 = _mean_optional(
        [None if row["best_route_top3"] is None else float(row["best_route_top3"]) for row in rows]
    )
    nonabstaining = [row for row in rows if not bool(row["abstained"])]
    all_comparable_score_pairs_tied = bool(
        score_pairs > 0 and score_ties == score_pairs
    )
    all_nonabstaining_score_spreads_within_tolerance = bool(
        nonabstaining
        and all(
            row["score_spread"] is not None
            and float(row["score_spread"]) <= SCORE_TIE_TOLERANCE
            for row in nonabstaining
        )
    )
    positive_signal = bool(
        (pairwise is not None and pairwise > 0.5)
        or (top3 is not None and top3 > 0.0)
        or selected_sum > 0.0
    )
    spearman = _mean_optional([row["spearman_rho"] for row in rows])
    top1 = _mean_optional(
        [None if row["best_route_top1"] is None else float(row["best_route_top1"]) for row in rows]
    )
    mrr = _mean_optional([row["mean_reciprocal_rank"] for row in rows])
    tie_rate = score_ties / score_pairs if score_pairs else None
    completions = sum(bool(row["selected_completed"]) for row in rows)
    return {
        "states": len(rows),
        "candidates": sum(int(row["candidates"]) for row in rows),
        "ordered_pairs": ordered_pairs,
        "pairwise_correct_credit": correct,
        "pairwise_accuracy": pairwise,
        "spearman_rho": spearman,
        "route_progress_spearman": spearman,
        "kendall_tau_b": _mean_optional([row["kendall_tau_b"] for row in rows]),
        "best_route_top1_rate": top1,
        "best_route_top1": top1,
        "best_route_top3_rate": top3,
        "best_route_top3": top3,
        "mean_reciprocal_rank": mrr,
        "mrr": mrr,
        "mean_best_route_rank": _mean_optional([row["mean_best_route_rank"] for row in rows]),
        "mean_rank": _mean_optional([row["mean_best_route_rank"] for row in rows]),
        "score_spread": _mean_optional([row["score_spread"] for row in rows]),
        "score_pair_count": score_pairs,
        "score_tie_count": score_ties,
        "score_tie_rate": tie_rate,
        "tie_rate": tie_rate,
        "normalized_regret": _mean_optional([row["normalized_regret"] for row in rows]),
        "selected_route_progress_m_sum": selected_sum,
        "selected_route_progress_m_mean": selected_sum / len(rows),
        "selected_heading_progress_rad_sum": selected_heading_sum,
        "selected_heading_progress_rad_mean": selected_heading_sum / len(rows),
        "selected_combined_route_utility_sum": float(sum(selected_utilities)),
        "selected_combined_route_utility_mean": (
            float(np.mean(selected_utilities)) if selected_utilities else None
        ),
        "selected_combined_route_utility_count": len(selected_utilities),
        "oracle_best_route_progress_m_sum": best_sum,
        "selected_progress_ratio": selected_sum / max(abs(best_sum), NUMERIC_EPSILON),
        "selected_immediate_contacts_h1": sum(
            bool(row["selected_immediate_contact_h1"]) for row in rows
        ),
        "selected_nonviable_successors": sum(
            bool(row["selected_nonviable_successor"]) for row in rows
        ),
        "selected_stuck": sum(bool(row["selected_stuck"]) for row in rows),
        "selected_completions": completions,
        "waypoint_completion_rate": completions / len(rows),
        "abstentions": sum(bool(row["abstained"]) for row in rows),
        "all_comparable_score_pairs_tied": all_comparable_score_pairs_tied,
        "all_nonabstaining_score_spreads_within_tolerance": (
            all_nonabstaining_score_spreads_within_tolerance
        ),
        "complete_family_collapse": (
            not nonabstaining
            or ordered_pairs == 0
            or not positive_signal
            or all_comparable_score_pairs_tied
            or all_nonabstaining_score_spreads_within_tolerance
        ),
    }


def adverse_cross_group_downranking_metrics(
    candidates_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    scores_by_state: Mapping[str, Any],
    *,
    expected_families: Sequence[str] = FAMILY_IDS,
    score_tie_tolerance: float = SCORE_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Describe whether scores down-rank adverse candidates in ALL_CANDIDATES.

    Only within-state adverse/nonadverse cross-group pairs are compared.  One
    credit is awarded when the nonadverse candidate scores higher, half for a
    tie, and zero when the adverse candidate scores higher.  Outcome fields are
    evaluation labels only: this descriptive metric is not a route target or a
    classification gate.
    """

    if list(candidates_by_state) != list(scores_by_state):
        raise PlanAwareMetricsError("candidate and score state order must match exactly")
    families = tuple(expected_families)
    if not families or len(families) != len(set(families)):
        raise PlanAwareMetricsError("expected families must be nonempty and unique")
    tolerance = _finite(score_tie_tolerance, name="score_tie_tolerance")
    if tolerance < 0.0:
        raise PlanAwareMetricsError("score tie tolerance must be non-negative")

    totals = {
        outcome: {
            "overall": {"pair_count": 0, "correct_credit": 0.0},
            "per_family": {
                family: {"pair_count": 0, "correct_credit": 0.0}
                for family in families
            },
        }
        for outcome in ADVERSE_DOWNRANKING_OUTCOMES
    }

    for state_id, candidate_values in candidates_by_state.items():
        rows = list(candidate_values)
        observed_families = {str(row.get("family", "")) for row in rows}
        if len(observed_families) != 1 or "" in observed_families:
            raise PlanAwareMetricsError(f"{state_id}: family is not state-invariant")
        family = next(iter(observed_families))
        if family not in families:
            raise PlanAwareMetricsError(f"{state_id}: unexpected family {family!r}")
        scores = _aligned_scores(rows, scores_by_state[state_id])
        adverse_flags = {
            "immediate_contact": [_contact(row) for row in rows],
            "successor_nonviable": [not _successor_viable(row) for row in rows],
            "stuck": [_stuck(row) for row in rows],
        }
        for outcome, flags in adverse_flags.items():
            adverse = [index for index, flag in enumerate(flags) if flag]
            nonadverse = [index for index, flag in enumerate(flags) if not flag]
            for adverse_index in adverse:
                for nonadverse_index in nonadverse:
                    delta = float(scores[nonadverse_index] - scores[adverse_index])
                    credit = 0.5 if abs(delta) <= tolerance else float(delta > 0.0)
                    totals[outcome]["overall"]["pair_count"] += 1
                    totals[outcome]["overall"]["correct_credit"] += credit
                    totals[outcome]["per_family"][family]["pair_count"] += 1
                    totals[outcome]["per_family"][family]["correct_credit"] += credit

    def finalize(value: Mapping[str, Any]) -> dict[str, Any]:
        pair_count = int(value["pair_count"])
        credit = float(value["correct_credit"])
        return {
            "pair_count": pair_count,
            "correct_credit": credit,
            "pairwise_accuracy": credit / pair_count if pair_count else None,
        }

    return {
        "schema": "plan_aware_adverse_cross_group_downranking_v1",
        "population_id": ALL_CANDIDATES,
        "score_direction": "higher_is_better",
        "pair_definition": "within_state_adverse_vs_nonadverse_cross_group",
        "descriptive_only": True,
        "used_as_score_or_route_target": False,
        "used_as_classification_gate": False,
        "outcomes": {
            outcome: {
                "overall": finalize(values["overall"]),
                "per_family": {
                    family: finalize(values["per_family"][family])
                    for family in families
                },
            }
            for outcome, values in totals.items()
        },
    }


def summarize_scores(
    candidates_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    scores_by_state: Mapping[str, Any],
    source_id: str,
    *,
    expected_families: Sequence[str] = FAMILY_IDS,
) -> dict[str, Any]:
    """Evaluate higher-is-better scores under all three frozen populations."""

    if list(candidates_by_state) != list(scores_by_state):
        raise PlanAwareMetricsError("candidate and score state order must match exactly")
    families = tuple(expected_families)
    if not families or len(families) != len(set(families)):
        raise PlanAwareMetricsError("expected families must be nonempty and unique")
    populations: dict[str, Any] = {}
    for population_id in POPULATION_IDS:
        per_state: list[dict[str, Any]] = []
        for state_id, candidates_value in candidates_by_state.items():
            candidates = list(candidates_value)
            observed_families = {str(row.get("family", "")) for row in candidates}
            observed_roles = {str(row.get("role", "")) for row in candidates}
            if len(observed_families) != 1 or "" in observed_families:
                raise PlanAwareMetricsError(f"{state_id}: family is not state-invariant")
            if len(observed_roles) != 1 or "" in observed_roles:
                raise PlanAwareMetricsError(f"{state_id}: role is not state-invariant")
            family = next(iter(observed_families))
            if family not in families:
                raise PlanAwareMetricsError(f"{state_id}: unexpected family {family!r}")
            per_state.append(
                evaluate_state_population(
                    candidates,
                    scores_by_state[state_id],
                    state_id=str(state_id),
                    family=family,
                    role=next(iter(observed_roles)),
                    source_id=source_id,
                    population_id=population_id,
                )
            )
        per_family = {
            family: aggregate_state_metrics(
                [row for row in per_state if row["family"] == family]
            )
            for family in families
        }
        per_role = {
            role: aggregate_state_metrics(
                [row for row in per_state if row["role"] == role]
            )
            for role in sorted({str(row["role"]) for row in per_state})
        }
        collapsed = [
            family
            for family, aggregate in per_family.items()
            if aggregate["complete_family_collapse"]
        ]
        overall = aggregate_state_metrics(per_state)
        populations[population_id] = {
            "per_state": per_state,
            "aggregate": overall,
            "overall": overall,
            "per_family": per_family,
            "per_role": per_role,
            "collapsed_families": collapsed,
            "no_family_complete_collapse": not collapsed,
        }
    return {
        "schema": "plan_aware_monotone_score_summary_v1",
        "score_direction": "higher_is_better",
        "source_id": source_id,
        "populations": populations,
        "descriptive_adverse_downranking": adverse_cross_group_downranking_metrics(
            candidates_by_state,
            scores_by_state,
            expected_families=families,
        ),
    }


def _population(source: Mapping[str, Any], population_id: str) -> Mapping[str, Any]:
    try:
        population = source["populations"][population_id]
        aggregate = population["aggregate"]
    except (KeyError, TypeError) as exc:
        raise PlanAwareMetricsError(f"source lacks {population_id} aggregate") from exc
    if not isinstance(aggregate, Mapping):
        raise PlanAwareMetricsError("population aggregate must be a mapping")
    return population


def _aggregate(source: Mapping[str, Any], population_id: str) -> Mapping[str, Any]:
    return _population(source, population_id)["aggregate"]


def _metric(aggregate: Mapping[str, Any], key: str) -> float:
    if key not in aggregate or aggregate[key] is None:
        raise PlanAwareMetricsError(f"aggregate metric {key!r} is unavailable")
    return _finite(aggregate[key], name=key)


def _metric_or_none(aggregate: Mapping[str, Any], key: str) -> float | None:
    """Preserve a legitimate unavailable aggregate while rejecting schema drift."""

    if key not in aggregate:
        raise PlanAwareMetricsError(f"aggregate metric {key!r} is missing")
    return _optional_finite(aggregate[key], name=key)


def _difference_or_none(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _minimum_criterion(value: float | None, threshold: float) -> bool:
    return bool(value is not None and value >= threshold)


def _maximum_criterion(value: float | None, threshold: float) -> bool:
    return bool(value is not None and value <= threshold)


def principal_metric_deltas(
    candidate_source: Mapping[str, Any],
    comparator_source: Mapping[str, Any],
    *,
    population_id: str = ORACLE_VIABILITY_ADMISSIBLE,
) -> dict[str, float | None]:
    """Return candidate-minus-comparator effects, all signed positive=favorable."""

    candidate = _aggregate(candidate_source, population_id)
    comparator = _aggregate(comparator_source, population_id)
    candidate_best = _metric(candidate, "oracle_best_route_progress_m_sum")
    comparator_best = _metric(comparator, "oracle_best_route_progress_m_sum")
    if not math.isclose(candidate_best, comparator_best, rel_tol=0.0, abs_tol=1e-12):
        raise PlanAwareMetricsError("paired sources disagree on oracle-best progress")
    candidate_selected = _metric(candidate, "selected_route_progress_m_sum")
    comparator_selected = _metric(comparator, "selected_route_progress_m_sum")
    candidate_states = int(candidate.get("states", 0))
    comparator_states = int(comparator.get("states", 0))
    if candidate_states != comparator_states or candidate_states <= 0:
        raise PlanAwareMetricsError("paired population state counts must match and be positive")
    candidate_pairwise = _metric_or_none(candidate, "pairwise_accuracy")
    comparator_pairwise = _metric_or_none(comparator, "pairwise_accuracy")
    candidate_spearman = _metric_or_none(candidate, "spearman_rho")
    comparator_spearman = _metric_or_none(comparator, "spearman_rho")
    candidate_kendall = _metric_or_none(candidate, "kendall_tau_b")
    comparator_kendall = _metric_or_none(comparator, "kendall_tau_b")
    candidate_regret = _metric_or_none(candidate, "normalized_regret")
    comparator_regret = _metric_or_none(comparator, "normalized_regret")
    candidate_top3 = _metric_or_none(candidate, "best_route_top3_rate")
    comparator_top3 = _metric_or_none(comparator, "best_route_top3_rate")
    return {
        "pairwise_accuracy_gain": _difference_or_none(
            candidate_pairwise, comparator_pairwise
        ),
        "spearman_gain": _difference_or_none(
            candidate_spearman, comparator_spearman
        ),
        "kendall_gain": _difference_or_none(candidate_kendall, comparator_kendall),
        "normalized_regret_reduction": _difference_or_none(
            comparator_regret, candidate_regret
        ),
        "best_route_top3_gain": _difference_or_none(
            candidate_top3, comparator_top3
        ),
        "selected_progress_gain_m": (candidate_selected - comparator_selected)
        / candidate_states,
        "selected_progress_ratio_gain": _metric(candidate, "selected_progress_ratio")
        - _metric(comparator, "selected_progress_ratio"),
        "oracle_normalized_progress_gain": (candidate_selected - comparator_selected)
        / max(abs(candidate_best), NUMERIC_EPSILON),
        "selected_immediate_contact_reduction": _metric(
            comparator, "selected_immediate_contacts_h1"
        )
        - _metric(candidate, "selected_immediate_contacts_h1"),
        "selected_nonviable_reduction": _metric(
            comparator, "selected_nonviable_successors"
        )
        - _metric(candidate, "selected_nonviable_successors"),
    }


def paired_principal_bootstrap(
    candidate_source: Mapping[str, Any],
    comparator_source: Mapping[str, Any],
    *,
    population_id: str = ORACLE_VIABILITY_ADMISSIBLE,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return descriptive paired-state bootstrap intervals for principal effects."""

    candidate_rows = _population(candidate_source, population_id).get("per_state")
    comparator_rows = _population(comparator_source, population_id).get("per_state")
    if not isinstance(candidate_rows, list) or not isinstance(comparator_rows, list):
        raise PlanAwareMetricsError("paired bootstrap requires per-state rows")
    candidate = {str(row["state_id"]): row for row in candidate_rows}
    comparator = {str(row["state_id"]): row for row in comparator_rows}
    if list(candidate) != list(comparator):
        raise PlanAwareMetricsError("paired per-state identities/order differ")
    families = {state: str(candidate[state]["family"]) for state in candidate}
    if any(str(comparator[state]["family"]) != families[state] for state in candidate):
        raise PlanAwareMetricsError("paired per-state families differ")

    def bootstrap(
        candidate_values: Mapping[str, float],
        comparator_values: Mapping[str, float],
        metric: str,
    ) -> dict[str, Any]:
        return PREDECESSOR.paired_state_bootstrap(
            candidate_values,
            comparator_values,
            {state: families[state] for state in candidate_values},
            comparison_id=(
                f"{candidate_source.get('source_id')}_MINUS_"
                f"{comparator_source.get('source_id')}/{population_id}/{metric}"
            ),
            draws=draws,
            seed=seed,
        )

    joint_pairwise = [
        state
        for state in candidate
        if candidate[state]["pairwise_accuracy"] is not None
        and comparator[state]["pairwise_accuracy"] is not None
    ]
    joint_regret = [
        state
        for state in candidate
        if candidate[state]["normalized_regret"] is not None
        and comparator[state]["normalized_regret"] is not None
    ]
    output = {
        "selected_progress_gain_m": bootstrap(
            {state: float(candidate[state]["selected_route_progress_m"]) for state in candidate},
            {state: float(comparator[state]["selected_route_progress_m"]) for state in candidate},
            "SELECTED_PROGRESS_M",
        ),
        "best_route_top3_gain": bootstrap(
            {state: float(candidate[state]["best_route_top3"] or False) for state in candidate},
            {state: float(comparator[state]["best_route_top3"] or False) for state in candidate},
            "BEST_ROUTE_TOP3",
        ),
    }
    if joint_pairwise:
        output["pairwise_accuracy_gain"] = bootstrap(
            {state: float(candidate[state]["pairwise_accuracy"]) for state in joint_pairwise},
            {state: float(comparator[state]["pairwise_accuracy"]) for state in joint_pairwise},
            "PAIRWISE_ACCURACY",
        )
    else:
        output["pairwise_accuracy_gain"] = None
    if joint_regret:
        # Swap arguments so positive means regret reduction.
        output["normalized_regret_reduction"] = bootstrap(
            {state: float(comparator[state]["normalized_regret"]) for state in joint_regret},
            {state: float(candidate[state]["normalized_regret"]) for state in joint_regret},
            "NORMALIZED_REGRET_REDUCTION",
        )
    else:
        output["normalized_regret_reduction"] = None
    return {
        "schema": "plan_aware_monotone_paired_principal_bootstrap_v1",
        "population_id": population_id,
        "draws": draws,
        "seed": seed,
        "metrics": output,
    }


def within_state_future_derangement(
    contract_digest: str,
    state_id: str,
    candidate_indices: Sequence[int],
) -> dict[int, int]:
    """Map each destination candidate to a complete-trajectory donor.

    SHA-256 of ``contract_digest UTF-8, NUL, state_id UTF-8`` selects a nonzero
    cyclic shift in the sorted candidate bank.  This produces one deterministic,
    bijective, fixed-point-free within-state map and exactly matches the
    evaluator's prospective map.  H1--H3 must be moved as one bundle by
    :func:`derange_future_trajectories`.
    """

    if not isinstance(contract_digest, str) or len(contract_digest) != 64:
        raise PlanAwareMetricsError("contract_digest must be a SHA-256 hex digest")
    try:
        bytes.fromhex(contract_digest)
    except ValueError as exc:
        raise PlanAwareMetricsError("contract_digest must be hexadecimal") from exc
    if not isinstance(state_id, str) or not state_id:
        raise PlanAwareMetricsError("state_id must be nonempty")
    candidates = []
    for value in candidate_indices:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PlanAwareMetricsError("candidate indices must be integers")
        candidate = int(value)
        if not 0 <= candidate < 2**32:
            raise PlanAwareMetricsError("candidate index must fit uint32")
        candidates.append(candidate)
    if len(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise PlanAwareMetricsError("derangement requires at least two unique candidates")
    ordered = sorted(candidates)
    digest = hashlib.sha256(
        contract_digest.encode("utf-8") + b"\x00" + state_id.encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(digest[:8], "big", signed=False) % (len(ordered) - 1)
    return {
        destination: ordered[(position + shift) % len(ordered)]
        for position, destination in enumerate(ordered)
    }


def derange_future_trajectories(
    trajectories_by_candidate: Mapping[int, Sequence[Any]],
    *,
    contract_digest: str,
    state_id: str,
) -> tuple[dict[int, tuple[Any, Any, Any]], dict[str, Any]]:
    """Reassign complete H1--H3 bundles and return an integrity receipt."""

    trajectories = {int(key): tuple(value) for key, value in trajectories_by_candidate.items()}
    if any(len(value) != 3 for value in trajectories.values()):
        raise PlanAwareMetricsError("every future trajectory must contain exactly H1--H3")
    mapping = within_state_future_derangement(
        contract_digest, state_id, list(trajectories)
    )
    output = {
        destination: tuple(copy.deepcopy(trajectories[donor]))
        for destination, donor in mapping.items()
    }
    receipt = {
        "schema": "plan_aware_complete_future_derangement_receipt_v1",
        "namespace": DERANGEMENT_NAMESPACE,
        "contract_digest": contract_digest,
        "state_id": state_id,
        "destination_to_donor": {str(key): value for key, value in sorted(mapping.items())},
        "candidate_count": len(mapping),
        "bijection": set(mapping) == set(mapping.values()),
        "fixed_point_count": sum(key == value for key, value in mapping.items()),
        "complete_h1_h3_bundle_reassigned": True,
    }
    return output, receipt


def derangement_damage(
    matched_source: Mapping[str, Any],
    deranged_source: Mapping[str, Any],
    *,
    population_id: str = ORACLE_VIABILITY_ADMISSIBLE,
) -> dict[str, float | bool | None]:
    """Return gated principal damage plus frozen descriptive route losses."""

    matched = _aggregate(matched_source, population_id)
    deranged = _aggregate(deranged_source, population_id)
    damage: dict[str, float | bool | None] = {
        "pairwise_accuracy_loss": _difference_or_none(
            _metric_or_none(matched, "pairwise_accuracy"),
            _metric_or_none(deranged, "pairwise_accuracy"),
        ),
        "selected_progress_ratio_loss": _metric(matched, "selected_progress_ratio")
        - _metric(deranged, "selected_progress_ratio"),
        "selected_progress_m_loss": _metric(
            matched, "selected_route_progress_m_mean"
        )
        - _metric(deranged, "selected_route_progress_m_mean"),
        "normalized_regret_worsening": _difference_or_none(
            _metric_or_none(deranged, "normalized_regret"),
            _metric_or_none(matched, "normalized_regret"),
        ),
        "best_route_top3_loss": _difference_or_none(
            _metric_or_none(matched, "best_route_top3_rate"),
            _metric_or_none(deranged, "best_route_top3_rate"),
        ),
    }
    damage["pairwise_accuracy_material_opposing_reversal"] = bool(
        damage["pairwise_accuracy_loss"] is not None
        and damage["pairwise_accuracy_loss"]
        <= -DERANGEMENT_MATERIALITY_THRESHOLDS[
            "pairwise_accuracy_opposing_improvement_minimum"
        ]
        + SCORE_TIE_TOLERANCE
    )
    damage["selected_progress_ratio_material_opposing_reversal"] = bool(
        damage["selected_progress_ratio_loss"]
        <= -DERANGEMENT_MATERIALITY_THRESHOLDS[
            "selected_progress_ratio_opposing_improvement_minimum"
        ]
        + SCORE_TIE_TOLERANCE
    )
    damage["normalized_regret_material_opposing_reversal"] = bool(
        damage["normalized_regret_worsening"] is not None
        and damage["normalized_regret_worsening"]
        <= -DERANGEMENT_MATERIALITY_THRESHOLDS[
            "normalized_regret_opposing_improvement_minimum"
        ]
        + SCORE_TIE_TOLERANCE
    )
    damage["no_material_principal_reversal"] = not any(
        (
            damage["pairwise_accuracy_material_opposing_reversal"],
            damage["selected_progress_ratio_material_opposing_reversal"],
            damage["normalized_regret_material_opposing_reversal"],
        )
    )
    return damage


def evaluate_future_derangement_materiality(
    matched_source: Mapping[str, Any],
    deranged_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the prospective true-future anti-shortcut materiality gate."""

    damage = derangement_damage(matched_source, deranged_source)
    triggers = {
        "pairwise_accuracy_loss": _minimum_criterion(
            damage["pairwise_accuracy_loss"],
            DERANGEMENT_MATERIALITY_THRESHOLDS[
                "pairwise_accuracy_loss_minimum"
            ],
        ),
        "selected_progress_ratio_loss": damage["selected_progress_ratio_loss"]
        >= DERANGEMENT_MATERIALITY_THRESHOLDS[
            "selected_progress_ratio_loss_minimum"
        ],
        "normalized_regret_worsening": _minimum_criterion(
            damage["normalized_regret_worsening"],
            DERANGEMENT_MATERIALITY_THRESHOLDS[
                "normalized_regret_worsening_minimum"
            ],
        ),
    }
    passed = bool(any(triggers.values()) and damage["no_material_principal_reversal"])
    return {
        "schema": "plan_aware_true_future_derangement_materiality_gate_v1",
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "damage": damage,
        "thresholds": dict(DERANGEMENT_MATERIALITY_THRESHOLDS),
        "triggers": triggers,
        "pass": passed,
    }


def evaluate_true_future_gate(
    true_source: Mapping[str, Any],
    derangement_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every Stage-A true-future plan-aware gate conjunctively."""

    aggregate = _aggregate(true_source, ORACLE_VIABILITY_ADMISSIBLE)
    population = _population(true_source, ORACLE_VIABILITY_ADMISSIBLE)
    derangement_pass = _strict_bool(derangement_gate.get("pass"), name="derangement.pass")
    observed = {
        "pairwise_accuracy": _metric_or_none(aggregate, "pairwise_accuracy"),
        "spearman_rho": _metric_or_none(aggregate, "spearman_rho"),
        "normalized_regret": _metric_or_none(aggregate, "normalized_regret"),
        "best_route_top3": _metric_or_none(aggregate, "best_route_top3_rate"),
        "selected_progress_ratio": _metric_or_none(
            aggregate, "selected_progress_ratio"
        ),
    }
    criteria = {
        "pairwise_accuracy": _minimum_criterion(
            observed["pairwise_accuracy"],
            TRUE_GATE_THRESHOLDS["pairwise_accuracy_minimum"],
        ),
        "spearman_rho": _minimum_criterion(
            observed["spearman_rho"], TRUE_GATE_THRESHOLDS["spearman_minimum"]
        ),
        "normalized_regret": _maximum_criterion(
            observed["normalized_regret"],
            TRUE_GATE_THRESHOLDS["normalized_regret_maximum"],
        ),
        "best_route_top3": _minimum_criterion(
            observed["best_route_top3"],
            TRUE_GATE_THRESHOLDS["best_route_top3_minimum"],
        ),
        "selected_progress_ratio": _minimum_criterion(
            observed["selected_progress_ratio"],
            TRUE_GATE_THRESHOLDS["selected_progress_ratio_minimum"],
        ),
        "no_family_complete_collapse": bool(
            population.get("no_family_complete_collapse")
        ),
        "future_derangement_material": derangement_pass,
    }
    passed = all(criteria.values())
    return {
        "schema": "plan_aware_true_future_gate_v1",
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "thresholds": dict(TRUE_GATE_THRESHOLDS),
        "observed_metrics": observed,
        "criteria": criteria,
        "pass": passed,
        "classification": (
            "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL"
            if passed
            else "TRUE_FUTURE_PLAN_AWARE_COST_NO_SIGNAL"
        ),
    }


def _adverse_not_worse(
    candidate_source: Mapping[str, Any], comparator_source: Mapping[str, Any]
) -> bool:
    candidate = _aggregate(candidate_source, ALL_CANDIDATES)
    comparator = _aggregate(comparator_source, ALL_CANDIDATES)
    return bool(
        _metric(candidate, "selected_immediate_contacts_h1")
        <= _metric(comparator, "selected_immediate_contacts_h1")
        and _metric(candidate, "selected_nonviable_successors")
        <= _metric(comparator, "selected_nonviable_successors")
    )


def _rr_adverse_not_worse(
    rollout_source: Mapping[str, Any], one_step_source: Mapping[str, Any]
) -> bool:
    """Apply the complete Section-19 RR adverse-selection parity check.

    Unlike the Section-17 incremental and Section-20 proprioception checks, the
    RR-vs-R1 gate also includes selected stuck outcomes in ALL_CANDIDATES.
    """

    rollout = _aggregate(rollout_source, ALL_CANDIDATES)
    one_step = _aggregate(one_step_source, ALL_CANDIDATES)
    return bool(
        _adverse_not_worse(rollout_source, one_step_source)
        and _metric(rollout, "selected_stuck")
        <= _metric(one_step, "selected_stuck")
    )


def _incremental_against(
    candidate_source: Mapping[str, Any], comparator_source: Mapping[str, Any]
) -> dict[str, Any]:
    primary = principal_metric_deltas(candidate_source, comparator_source)
    family_candidate = _population(candidate_source, ORACLE_VIABILITY_ADMISSIBLE)[
        "per_family"
    ]
    family_comparator = _population(comparator_source, ORACLE_VIABILITY_ADMISSIBLE)[
        "per_family"
    ]
    if set(family_candidate) != set(family_comparator):
        raise PlanAwareMetricsError("paired sources have different family sets")
    family_effects = {
        family: _metric(family_candidate[family], "selected_route_progress_m_sum")
        - _metric(family_comparator[family], "selected_route_progress_m_sum")
        for family in family_candidate
    }
    families_improved_or_tied = sum(
        value >= -SCORE_TIE_TOLERANCE for value in family_effects.values()
    )
    population_effects = {
        population_id: principal_metric_deltas(
            candidate_source, comparator_source, population_id=population_id
        )["oracle_normalized_progress_gain"]
        for population_id in POPULATION_IDS
    }
    criteria = {
        "progress_material": bool(
            primary["selected_progress_gain_m"]
            >= INCREMENTAL_ROUTE_VALUE_THRESHOLDS[
                "selected_progress_gain_m_minimum"
            ]
            or primary["oracle_normalized_progress_gain"]
            >= INCREMENTAL_ROUTE_VALUE_THRESHOLDS[
                "oracle_normalized_progress_gain_minimum"
            ]
        ),
        "normalized_regret_material": _minimum_criterion(
            primary["normalized_regret_reduction"],
            INCREMENTAL_ROUTE_VALUE_THRESHOLDS[
                "normalized_regret_reduction_minimum"
            ],
        ),
        "three_of_four_families_improve_or_tie": bool(
            len(family_effects) == 4
            and families_improved_or_tied
            >= INCREMENTAL_ROUTE_VALUE_THRESHOLDS[
                "families_improved_or_tied_minimum"
            ]
        ),
        "no_population_loses_more_than_two_percent_progress": all(
            value
            >= -INCREMENTAL_ROUTE_VALUE_THRESHOLDS[
                "population_progress_loss_maximum"
            ]
            for value in population_effects.values()
        ),
        "all_candidate_adverse_outcomes_not_worse": _adverse_not_worse(
            candidate_source, comparator_source
        ),
    }
    return {
        "comparator_id": comparator_source.get("source_id"),
        "principal_deltas": primary,
        "per_family_selected_progress_delta_m": family_effects,
        "families_improved_or_tied": families_improved_or_tied,
        "per_population_oracle_normalized_progress_gain": population_effects,
        "criteria": criteria,
        "pass": all(criteria.values()),
    }


def evaluate_incremental_route_value(
    candidate_source: Mapping[str, Any],
    *,
    kinematic_source: Mapping[str, Any],
    no_latent_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the exact incremental route-value test against required baselines."""

    comparisons = {
        "KINEMATIC": _incremental_against(candidate_source, kinematic_source)
    }
    if no_latent_source is not None:
        comparisons["NO_LATENT"] = _incremental_against(
            candidate_source, no_latent_source
        )
    passed = all(comparison["pass"] for comparison in comparisons.values())
    true_mode = no_latent_source is not None
    classification = (
        "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE"
        if passed and true_mode
        else "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS"
        if passed
        else "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_SUPPORTED"
        if true_mode
        else "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED"
    )
    return {
        "schema": "plan_aware_incremental_route_value_gate_v1",
        "thresholds": dict(INCREMENTAL_ROUTE_VALUE_THRESHOLDS),
        "comparisons": comparisons,
        "pass": passed,
        "classification": classification,
    }


def evaluate_predicted_gate(
    *,
    true_gate: Mapping[str, Any],
    true_source: Mapping[str, Any],
    one_step_source: Mapping[str, Any],
    rollout_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the Stage-B RGB-rollout plan-aware gate."""

    true_pass = _strict_bool(true_gate.get("pass"), name="true_gate.pass")
    one = _aggregate(one_step_source, ORACLE_VIABILITY_ADMISSIBLE)
    rollout_population = _population(rollout_source, ORACLE_VIABILITY_ADMISSIBLE)
    rollout = rollout_population["aggregate"]
    true = _aggregate(true_source, ORACLE_VIABILITY_ADMISSIBLE)
    retained = _metric(rollout, "selected_route_progress_m_sum") / max(
        abs(_metric(true, "selected_route_progress_m_sum")), NUMERIC_EPSILON
    )
    rollout_pairwise = _metric_or_none(rollout, "pairwise_accuracy")
    one_pairwise = _metric_or_none(one, "pairwise_accuracy")
    pairwise_improvement = _difference_or_none(
        rollout_pairwise, one_pairwise
    )
    progress_improvement = _metric(
        rollout, "selected_route_progress_m_sum"
    ) > _metric(one, "selected_route_progress_m_sum")
    rollout_regret = _metric_or_none(rollout, "normalized_regret")
    one_regret = _metric_or_none(one, "normalized_regret")
    regret_improvement = bool(
        rollout_regret is not None
        and one_regret is not None
        and rollout_regret < one_regret
    )
    rollout_top3 = _metric_or_none(rollout, "best_route_top3_rate")
    rollout_progress_ratio = _metric_or_none(rollout, "selected_progress_ratio")
    criteria = {
        "true_gate_passes": true_pass,
        "rollout_pairwise_accuracy": _minimum_criterion(
            rollout_pairwise,
            PREDICTED_GATE_THRESHOLDS["pairwise_accuracy_minimum"],
        ),
        "rollout_normalized_regret": _maximum_criterion(
            rollout_regret,
            PREDICTED_GATE_THRESHOLDS["normalized_regret_maximum"],
        ),
        "rollout_best_route_top3": _minimum_criterion(
            rollout_top3,
            PREDICTED_GATE_THRESHOLDS["best_route_top3_minimum"],
        ),
        "rollout_selected_progress_ratio": _minimum_criterion(
            rollout_progress_ratio,
            PREDICTED_GATE_THRESHOLDS["selected_progress_ratio_minimum"],
        ),
        "retains_true_selected_progress": retained
        >= PREDICTED_GATE_THRESHOLDS[
            "true_selected_progress_retention_minimum"
        ],
        "pairwise_strictly_improves_over_one_step": bool(
            pairwise_improvement is not None and pairwise_improvement > 0.0
        ),
        "progress_or_regret_improves_over_one_step": (
            progress_improvement or regret_improvement
        ),
        "no_family_complete_collapse": bool(
            rollout_population.get("no_family_complete_collapse")
        ),
        "all_candidate_adverse_outcomes_not_worse": _rr_adverse_not_worse(
            rollout_source, one_step_source
        ),
    }
    passed = all(criteria.values())
    return {
        "schema": "plan_aware_predicted_cost_gate_v1",
        "thresholds": dict(PREDICTED_GATE_THRESHOLDS),
        "observed_metrics": {
            "rollout_pairwise_accuracy": rollout_pairwise,
            "rollout_normalized_regret": rollout_regret,
            "rollout_best_route_top3": rollout_top3,
            "rollout_selected_progress_ratio": rollout_progress_ratio,
            "pairwise_accuracy_delta_vs_one_step": pairwise_improvement,
            "normalized_regret_one_step": one_regret,
        },
        "true_selected_progress_retention": retained,
        "rollout_minus_one_step": principal_metric_deltas(
            rollout_source, one_step_source
        ),
        "criteria": criteria,
        "pass": passed,
        "classification": (
            "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"
            if passed
            else "TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL"
        ),
    }


def factorial_rollout_contrasts(
    *,
    rgb_one_step: Mapping[str, Any],
    rgb_rollout: Mapping[str, Any],
    proprio_one_step: Mapping[str, Any],
    proprio_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    """Return BR=RR-R1, BP=PR-P1, and interaction J=BP-BR."""

    br = principal_metric_deltas(rgb_rollout, rgb_one_step)
    bp = principal_metric_deltas(proprio_rollout, proprio_one_step)
    common = sorted(set(br) & set(bp))
    return {
        "schema": "plan_aware_factorial_rollout_contrasts_v1",
        "BR_rgb_rollout_minus_rgb_one_step": br,
        "BP_proprio_rollout_minus_proprio_one_step": bp,
        "J_BP_minus_BR": {
            key: (
                None
                if bp[key] is None or br[key] is None
                else float(bp[key] - br[key])
            )
            for key in common
        },
    }


def evaluate_proprioceptive_route_contribution(
    *,
    rgb_rollout: Mapping[str, Any],
    proprio_rollout: Mapping[str, Any],
    rgb_one_step: Mapping[str, Any] | None = None,
    proprio_one_step: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the PR-over-RR two-of-four contribution gate."""

    effects = principal_metric_deltas(proprio_rollout, rgb_rollout)
    triggers = {
        "pairwise_accuracy": _minimum_criterion(
            effects["pairwise_accuracy_gain"],
            PROPRIO_CONTRIBUTION_THRESHOLDS["pairwise_accuracy_gain"],
        ),
        "selected_progress_ratio": _minimum_criterion(
            effects["selected_progress_ratio_gain"],
            PROPRIO_CONTRIBUTION_THRESHOLDS["selected_progress_ratio_gain"],
        ),
        "normalized_regret": _minimum_criterion(
            effects["normalized_regret_reduction"],
            PROPRIO_CONTRIBUTION_THRESHOLDS["normalized_regret_reduction"],
        ),
        "best_route_top3": _minimum_criterion(
            effects["best_route_top3_gain"],
            PROPRIO_CONTRIBUTION_THRESHOLDS["best_route_top3_gain"],
        ),
    }
    adverse = _adverse_not_worse(proprio_rollout, rgb_rollout)
    passed = bool(
        sum(triggers.values())
        >= PROPRIO_CONTRIBUTION_THRESHOLDS["minimum_trigger_count"]
        and adverse
    )
    contrasts = None
    if (rgb_one_step is None) != (proprio_one_step is None):
        raise PlanAwareMetricsError("factorial contrast requires both one-step sources")
    if rgb_one_step is not None and proprio_one_step is not None:
        contrasts = factorial_rollout_contrasts(
            rgb_one_step=rgb_one_step,
            rgb_rollout=rgb_rollout,
            proprio_one_step=proprio_one_step,
            proprio_rollout=proprio_rollout,
        )
    return {
        "schema": "plan_aware_proprioceptive_route_contribution_gate_v1",
        "thresholds": dict(PROPRIO_CONTRIBUTION_THRESHOLDS),
        "PR_minus_RR": effects,
        "triggers": triggers,
        "trigger_count": sum(triggers.values()),
        "all_candidate_adverse_outcomes_not_worse": adverse,
        "factorial_contrasts": contrasts,
        "pass": passed,
        "classification": (
            "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION"
            if passed
            else "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED"
        ),
    }


def _differential_damage(
    left: Mapping[str, float | bool | None],
    right: Mapping[str, float | bool | None],
) -> dict[str, Any]:
    def differential(key: str) -> float | None:
        left_value = left[key]
        right_value = right[key]
        if left_value is None or right_value is None:
            return None
        return float(left_value) - float(right_value)

    values = {
        "pairwise_accuracy_loss": differential("pairwise_accuracy_loss"),
        "selected_progress_ratio_loss": differential(
            "selected_progress_ratio_loss"
        ),
        "normalized_regret_worsening": differential(
            "normalized_regret_worsening"
        ),
    }
    no_reversal = all(
        value is not None and value >= -SCORE_TIE_TOLERANCE
        for value in values.values()
    )
    material = bool(
        no_reversal
        and (
            _minimum_criterion(
                values["pairwise_accuracy_loss"],
                SUBSTITUTION_THRESHOLDS[
                    "differential_pairwise_loss_minimum"
                ],
            )
            or _minimum_criterion(
                values["selected_progress_ratio_loss"],
                SUBSTITUTION_THRESHOLDS[
                    "differential_progress_loss_minimum"
                ],
            )
            or _minimum_criterion(
                values["normalized_regret_worsening"],
                SUBSTITUTION_THRESHOLDS[
                    "differential_regret_worsening_minimum"
                ],
            )
        )
    )
    return {**values, "no_principal_reversal": no_reversal, "material": material}


def evaluate_substitution_tendency(
    *,
    proprio_contribution_gate: Mapping[str, Any],
    matched_proprio_rollout: Mapping[str, Any],
    visual_deranged: Mapping[str, Any],
    proprio_deranged: Mapping[str, Any],
    control_deranged: Mapping[str, Any],
    candidate_action_sensitivity_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply Stage-C modality substitution/attribution gates.

    ``candidate_action_sensitivity_evidence`` is a frozen prior diagnostic, not
    inferred here.  It must name its authority and carry raw evidence alongside
    an exact Boolean ``passed`` value.
    """

    contribution_pass = _strict_bool(
        proprio_contribution_gate.get("pass"), name="proprio_contribution.pass"
    )
    authority = candidate_action_sensitivity_evidence.get("authority")
    if not isinstance(authority, str) or not authority:
        raise PlanAwareMetricsError("action-sensitivity evidence requires authority")
    if "raw_evidence" not in candidate_action_sensitivity_evidence:
        raise PlanAwareMetricsError("action-sensitivity evidence requires raw_evidence")
    action_pass = _strict_bool(
        candidate_action_sensitivity_evidence.get("passed"),
        name="candidate_action_sensitivity.passed",
    )
    visual = derangement_damage(matched_proprio_rollout, visual_deranged)
    proprio = derangement_damage(matched_proprio_rollout, proprio_deranged)
    control = derangement_damage(matched_proprio_rollout, control_deranged)
    visual_negligible = bool(
        visual["pairwise_accuracy_loss"] is not None
        and visual["pairwise_accuracy_loss"]
        < SUBSTITUTION_THRESHOLDS["visual_pairwise_loss_strict_maximum"]
        and visual["selected_progress_ratio_loss"] is not None
        and visual["selected_progress_ratio_loss"]
        < SUBSTITUTION_THRESHOLDS["visual_progress_loss_strict_maximum"]
    )
    proprio_material = bool(
        _minimum_criterion(
            proprio["pairwise_accuracy_loss"],
            SUBSTITUTION_THRESHOLDS["proprio_pairwise_loss_minimum"],
        )
        or _minimum_criterion(
            proprio["selected_progress_ratio_loss"],
            SUBSTITUTION_THRESHOLDS["proprio_progress_loss_minimum"],
        )
        or _minimum_criterion(
            proprio["normalized_regret_worsening"],
            SUBSTITUTION_THRESHOLDS["proprio_regret_worsening_minimum"],
        )
    )
    visual_material = bool(
        _minimum_criterion(
            visual["pairwise_accuracy_loss"],
            SUBSTITUTION_THRESHOLDS["proprio_pairwise_loss_minimum"],
        )
        or _minimum_criterion(
            visual["selected_progress_ratio_loss"],
            SUBSTITUTION_THRESHOLDS["proprio_progress_loss_minimum"],
        )
        or _minimum_criterion(
            visual["normalized_regret_worsening"],
            SUBSTITUTION_THRESHOLDS["proprio_regret_worsening_minimum"],
        )
    )
    proprio_over_visual = _differential_damage(proprio, visual)
    proprio_over_control = _differential_damage(proprio, control)
    visual_over_proprio = _differential_damage(visual, proprio)
    criteria = {
        "PR_materially_exceeds_RR": contribution_pass,
        "visual_derangement_negligible": visual_negligible,
        "proprio_derangement_material": proprio_material,
        "proprio_materially_more_damaging_than_visual": proprio_over_visual[
            "material"
        ],
        "candidate_action_sensitivity_retained": action_pass,
        "not_explained_by_control_derangement_alone": proprio_over_control[
            "material"
        ],
    }
    passed = all(criteria.values())
    if passed:
        attribution = "PROPRIOCEPTIVE_SUBSTITUTION_TENDENCY"
    elif visual_over_proprio["material"]:
        attribution = "VISUAL_ROUTE_DEPENDENCE"
    elif visual_material and proprio_material:
        attribution = "MULTIMODAL_ROUTE_DEPENDENCE"
    else:
        attribution = "PROPRIOCEPTIVE_SUBSTITUTION_NOT_SUPPORTED"
    return {
        "schema": "plan_aware_proprioceptive_substitution_gate_v1",
        "population_id": ORACLE_VIABILITY_ADMISSIBLE,
        "thresholds": dict(SUBSTITUTION_THRESHOLDS),
        "damage": {
            "visual": visual,
            "proprio": proprio,
            "control": control,
            "proprio_minus_visual": proprio_over_visual,
            "proprio_minus_control": proprio_over_control,
            "visual_minus_proprio": visual_over_proprio,
        },
        "candidate_action_sensitivity_evidence": copy.deepcopy(
            dict(candidate_action_sensitivity_evidence)
        ),
        "criteria": criteria,
        "pass": passed,
        "classification": (
            "PROPRIOCEPTIVE_SUBSTITUTION_TENDENCY"
            if passed
            else "PROPRIOCEPTIVE_SUBSTITUTION_NOT_SUPPORTED"
        ),
        "dependence_attribution": attribution,
    }


# Evaluator-facing names kept deliberately short.  The descriptive names above
# remain the implementation authority, while these wrappers make the gate API
# parallel to ``evaluate_true_future_gate`` and ``evaluate_predicted_gate``.
def evaluate_incremental_value(
    candidate_source: Mapping[str, Any],
    *,
    kinematic_source: Mapping[str, Any],
    no_latent_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_incremental_route_value(
        candidate_source,
        kinematic_source=kinematic_source,
        no_latent_source=no_latent_source,
    )


def evaluate_proprio_gate(
    *,
    rgb_rollout: Mapping[str, Any],
    proprio_rollout: Mapping[str, Any],
    rgb_one_step: Mapping[str, Any] | None = None,
    proprio_one_step: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_proprioceptive_route_contribution(
        rgb_rollout=rgb_rollout,
        proprio_rollout=proprio_rollout,
        rgb_one_step=rgb_one_step,
        proprio_one_step=proprio_one_step,
    )


def evaluate_substitution_gate(**kwargs: Any) -> dict[str, Any]:
    return evaluate_substitution_tendency(**kwargs)


def classify_primary(
    *,
    true_gate: Mapping[str, Any],
    predicted_gate: Mapping[str, Any],
    true_incremental_gate: Mapping[str, Any],
    all_predicted_substitutions_fail_materially: bool,
) -> dict[str, Any]:
    """Apply the frozen primary precedence and return exactly one class."""

    true_pass = _strict_bool(true_gate.get("pass"), name="true_gate.pass")
    predicted_pass = _strict_bool(
        predicted_gate.get("pass"), name="predicted_gate.pass"
    )
    true_incremental = _strict_bool(
        true_incremental_gate.get("pass"), name="true_incremental_gate.pass"
    )
    substitutions_fail = _strict_bool(
        all_predicted_substitutions_fail_materially,
        name="all_predicted_substitutions_fail_materially",
    )
    if not true_pass:
        classification = "PLAN_AWARE_JEPA_COST_NO_SIGNAL"
    elif not true_incremental:
        classification = "KINEMATIC_BASELINE_DOMINANT"
    elif predicted_pass:
        classification = "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"
    elif substitutions_fail:
        classification = "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO"
    else:
        raise PlanAwareMetricsError(
            "primary classification is unresolved: predicted substitutions have "
            "not all failed materially"
        )
    return {
        "schema": "plan_aware_monotone_primary_classification_v1",
        "inputs": {
            "true_gate": true_pass,
            "predicted_gate": predicted_pass,
            "true_incremental_gate": true_incremental,
            "all_predicted_substitutions_fail_materially": substitutions_fail,
        },
        "classification": classification,
    }


__all__ = [
    "ALL_CANDIDATES",
    "ADVERSE_DOWNRANKING_OUTCOMES",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DERANGEMENT_MATERIALITY_THRESHOLDS",
    "DERANGEMENT_NAMESPACE",
    "EXPERIMENT_ID",
    "FAMILY_IDS",
    "INCREMENTAL_ROUTE_VALUE_THRESHOLDS",
    "ORACLE_CONTACT_FREE",
    "ORACLE_VIABILITY_ADMISSIBLE",
    "POPULATION_IDS",
    "PREDICTED_GATE_THRESHOLDS",
    "PRIMARY_CLASSIFICATIONS",
    "PROPRIO_CONTRIBUTION_THRESHOLDS",
    "PlanAwareMetricsError",
    "ROUTE_UTILITY_TIE_TOLERANCE",
    "SCORE_TIE_TOLERANCE",
    "SUBSTITUTION_THRESHOLDS",
    "TRUE_GATE_THRESHOLDS",
    "aggregate_state_metrics",
    "adverse_cross_group_downranking_metrics",
    "classify_primary",
    "derange_future_trajectories",
    "derangement_damage",
    "evaluate_future_derangement_materiality",
    "evaluate_incremental_value",
    "evaluate_incremental_route_value",
    "evaluate_predicted_gate",
    "evaluate_proprioceptive_route_contribution",
    "evaluate_proprio_gate",
    "evaluate_state_population",
    "evaluate_substitution_tendency",
    "evaluate_substitution_gate",
    "evaluate_true_future_gate",
    "factorial_rollout_contrasts",
    "paired_principal_bootstrap",
    "principal_metric_deltas",
    "route_only_margin_borda_utility",
    "route_only_order",
    "route_only_preference",
    "route_ordering_metrics",
    "summarize_scores",
    "within_state_future_derangement",
]
