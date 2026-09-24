"""Pure reducers for NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1.

The functions in this module accept in-memory rows only.  They perform no file,
checkpoint, simulator, archive, process, or device access.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import math
from typing import Any

import numpy as np

from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as CONTRACT


STAGE_A = "STAGE_A"
CONDITIONAL_STAGE_B = "CONDITIONAL_STAGE_B"
POPULATION_IDS = (
    "ALL_CANDIDATES",
    "ORACLE_CONTACT_FREE",
    "ORACLE_VIABILITY_ADMISSIBLE",
)
SCORE_ROW_REQUIRED_FIELDS = (
    "stage_id",
    "state_id",
    "family",
    "split_role",
    "candidate_index",
    "model_id",
    "source_id",
    "score",
    "geodesic_progress_m",
    "remaining_geodesic_m",
    "heading_error_to_next_shortest_segment_rad",
    "euclidean_progress_m",
    "oracle_admissible",
    "immediate_contact",
    "committed_prefix_contact",
    "successor_viable",
    "stuck",
    "dead_end",
    "completed",
)
FINITE_NUMERIC_FIELDS = (
    "score",
    "geodesic_progress_m",
    "euclidean_progress_m",
    "remaining_geodesic_m",
    "heading_error_to_next_shortest_segment_rad",
)
BOOLEAN_FIELDS = (
    "oracle_admissible",
    "immediate_contact",
    "committed_prefix_contact",
    "successor_viable",
    "stuck",
    "dead_end",
    "completed",
)
TARGET_TIE_TOLERANCE = 0.0
SCORE_TIE_TOLERANCE = 1.0e-12


class NonGreedyMetricsError(ValueError):
    """Raised on incomplete, contradictory, or non-finite score rows."""


def score_row_authority() -> dict[str, Any]:
    """Return the exact heldout score-ledger identity authority."""

    def stage(
        *, stage_id: str, file_name: str, pairs: Sequence[tuple[str, str]]
    ) -> dict[str, Any]:
        return {
            "stage_id": stage_id,
            "file_name": file_name,
            "required_fields": list(SCORE_ROW_REQUIRED_FIELDS),
            "finite_numeric_fields": list(FINITE_NUMERIC_FIELDS),
            "state_ids": list(CONTRACT.DEVELOPMENT_HELDOUT_STATE_IDS),
            "model_source_pairs": [
                {"model_id": model_id, "source_id": source_id}
                for model_id, source_id in pairs
            ],
            "candidate_count": CONTRACT.CANDIDATE_COUNT,
        }

    return {
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.score_row_authority.v1",
        "stages": {
            "stage_a": stage(
                stage_id=STAGE_A,
                file_name="stage_a_scores.jsonl",
                pairs=CONTRACT.STAGE_A_MODEL_SOURCE_PAIRS,
            ),
            "conditional_stage_b": stage(
                stage_id=CONDITIONAL_STAGE_B,
                file_name="conditional_stage_b_scores.jsonl",
                pairs=CONTRACT.STAGE_B_MODEL_SOURCE_PAIRS,
            ),
        },
    }


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise NonGreedyMetricsError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise NonGreedyMetricsError(f"{name} must be finite")
    return output


def _optional_finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def _at_least(value: Any, threshold: float) -> bool:
    observed = _optional_finite(value)
    return observed is not None and observed >= float(threshold)


def _at_most(value: Any, threshold: float) -> bool:
    observed = _optional_finite(value)
    return observed is not None and observed <= float(threshold)


def _difference(left: Any, right: Any) -> float | None:
    left_value = _optional_finite(left)
    right_value = _optional_finite(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _state_family(state_id: str) -> str:
    try:
        parts = state_id.split("-")
        family_index = int(parts[1])
        family_rank = int(parts[2])
    except (AttributeError, ValueError, IndexError) as exc:
        raise NonGreedyMetricsError(f"malformed state_id {state_id!r}") from exc
    if (
        len(parts) != 3
        or state_id != f"ngls-{family_index:02d}-{family_rank:02d}"
        or family_index not in range(len(CONTRACT.FAMILY_IDS))
        or family_rank not in range(20, 24)
    ):
        raise NonGreedyMetricsError(f"state_id is outside heldout authority: {state_id}")
    return CONTRACT.FAMILY_IDS[family_index]


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, authority_key: str
) -> list[dict[str, Any]]:
    authority = score_row_authority()["stages"][authority_key]
    output = [copy.deepcopy(dict(row)) for row in rows]
    pairs = [
        (str(pair["model_id"]), str(pair["source_id"]))
        for pair in authority["model_source_pairs"]
    ]
    expected_identities = {
        (state_id, model_id, source_id, candidate_index)
        for state_id in authority["state_ids"]
        for model_id, source_id in pairs
        for candidate_index in range(int(authority["candidate_count"]))
    }
    observed: set[tuple[str, str, str, int]] = set()
    outcome_by_candidate: dict[tuple[str, int], tuple[Any, ...]] = {}
    for row in output:
        if set(row) != set(SCORE_ROW_REQUIRED_FIELDS):
            raise NonGreedyMetricsError("score row key set drift")
        if row["stage_id"] != authority["stage_id"]:
            raise NonGreedyMetricsError("score row stage_id drift")
        state_id = row["state_id"]
        if not isinstance(state_id, str) or state_id not in authority["state_ids"]:
            raise NonGreedyMetricsError("score row state_id drift")
        family = _state_family(state_id)
        if row["family"] != family or row["split_role"] != "DEVELOPMENT_HELDOUT":
            raise NonGreedyMetricsError("score row family/split authority drift")
        candidate_index = row["candidate_index"]
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index not in range(CONTRACT.CANDIDATE_COUNT)
        ):
            raise NonGreedyMetricsError("candidate_index outside frozen bank")
        pair = (row["model_id"], row["source_id"])
        if pair not in pairs:
            raise NonGreedyMetricsError("unregistered model/source pair")
        for field in FINITE_NUMERIC_FIELDS:
            row[field] = _finite(row[field], name=field)
        if row["remaining_geodesic_m"] < 0.0:
            raise NonGreedyMetricsError("remaining geodesic distance cannot be negative")
        if row["heading_error_to_next_shortest_segment_rad"] < 0.0:
            raise NonGreedyMetricsError("heading error cannot be negative")
        for field in BOOLEAN_FIELDS:
            if not isinstance(row[field], (bool, np.bool_)):
                raise NonGreedyMetricsError(f"{field} must be Boolean")
            row[field] = bool(row[field])
        if row["oracle_admissible"] != (
            not row["committed_prefix_contact"] and row["successor_viable"]
        ):
            raise NonGreedyMetricsError("oracle admissibility does not match full-prefix outcomes")
        if row["immediate_contact"] and not row["committed_prefix_contact"]:
            raise NonGreedyMetricsError(
                "immediate contact must imply committed-prefix contact"
            )
        identity = (state_id, str(pair[0]), str(pair[1]), candidate_index)
        if identity in observed:
            raise NonGreedyMetricsError("duplicate score row identity")
        observed.add(identity)
        outcome = tuple(
            row[field]
            for field in (
                "family",
                "split_role",
                "geodesic_progress_m",
                "remaining_geodesic_m",
                "heading_error_to_next_shortest_segment_rad",
                "euclidean_progress_m",
                *BOOLEAN_FIELDS,
            )
        )
        candidate_key = (state_id, candidate_index)
        prior = outcome_by_candidate.setdefault(candidate_key, outcome)
        if prior != outcome:
            raise NonGreedyMetricsError("candidate outcome differs across model/source rows")
    if observed != expected_identities:
        missing = len(expected_identities - observed)
        extra = len(observed - expected_identities)
        raise NonGreedyMetricsError(
            f"score identity cube incomplete (missing={missing}, extra={extra})"
        )
    return output


def _target_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    """Lower is better; completion is deliberately absent/descriptive only."""

    return (
        -float(row["geodesic_progress_m"]),
        float(row["remaining_geodesic_m"]),
        float(row["heading_error_to_next_shortest_segment_rad"]),
        int(row["candidate_index"]),
    )


def _score_key(row: Mapping[str, Any]) -> tuple[float, int]:
    return (-float(row["score"]), int(row["candidate_index"]))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def _kendall_tau_b(target_order: Sequence[int], scores: Sequence[float]) -> float | None:
    concordant = discordant = target_ties = score_ties = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            target_delta = target_order[left] - target_order[right]
            score_delta = scores[left] - scores[right]
            if target_delta == 0:
                target_ties += 1
            elif abs(score_delta) <= SCORE_TIE_TOLERANCE:
                score_ties += 1
            elif target_delta * score_delta < 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + target_ties)
        * (concordant + discordant + score_ties)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def state_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_id: str = "ORACLE_VIABILITY_ADMISSIBLE",
) -> dict[str, Any]:
    """Reduce one state and one registered model/source pair."""

    values = [copy.deepcopy(dict(row)) for row in rows]
    if len(values) != CONTRACT.CANDIDATE_COUNT:
        raise NonGreedyMetricsError("state metrics require exactly 12 candidates")
    if len({int(row["candidate_index"]) for row in values}) != CONTRACT.CANDIDATE_COUNT:
        raise NonGreedyMetricsError("state candidate identities are incomplete")
    if len({(row["state_id"], row["model_id"], row["source_id"]) for row in values}) != 1:
        raise NonGreedyMetricsError("state metrics rows do not share one identity")
    if population_id not in POPULATION_IDS:
        raise NonGreedyMetricsError(f"unknown population {population_id!r}")
    raw_selected = min(values, key=_score_key)
    if population_id == "ALL_CANDIDATES":
        admissible = values
    elif population_id == "ORACLE_CONTACT_FREE":
        admissible = [row for row in values if not bool(row["committed_prefix_contact"])]
    else:
        admissible = [row for row in values if bool(row["oracle_admissible"])]
    if not admissible:
        return {
            "state_id": values[0]["state_id"],
            "family": values[0]["family"],
            "model_id": values[0]["model_id"],
            "source_id": values[0]["source_id"],
            "admissible_candidate_count": 0,
            "selected_candidate_index": None,
            "oracle_best_candidate_index": None,
            "pairwise_accuracy": None,
            "pairwise_correct_credit": 0.0,
            "pair_count": 0,
            "spearman": None,
            "kendall": None,
            "best_route_top1": None,
            "best_route_top3": None,
            "reciprocal_rank": None,
            "oracle_best_rank": None,
            "selected_geodesic_progress_m": None,
            "selected_euclidean_progress_m": None,
            "oracle_best_geodesic_progress_m": None,
            "oracle_progress_fraction": None,
            "normalized_regret": None,
            "score_spread": None,
            "population_id": population_id,
            "raw_selected_contact": bool(raw_selected["immediate_contact"]),
            "raw_selected_nonviable": not bool(raw_selected["successor_viable"]),
            "raw_selected_stuck": bool(raw_selected["stuck"]),
            "raw_selected_dead_end": bool(raw_selected["dead_end"]),
        }

    target_ordered = sorted(admissible, key=_target_key)
    score_ordered = sorted(admissible, key=_score_key)
    target_position = {
        int(row["candidate_index"]): index for index, row in enumerate(target_ordered)
    }
    score_position = {
        int(row["candidate_index"]): index for index, row in enumerate(score_ordered)
    }
    pair_count = 0
    pairwise_credit = 0.0
    for left in range(len(admissible)):
        for right in range(left + 1, len(admissible)):
            pair_count += 1
            left_row, right_row = admissible[left], admissible[right]
            target_prefers_left = _target_key(left_row) < _target_key(right_row)
            score_delta = float(left_row["score"]) - float(right_row["score"])
            if abs(score_delta) <= SCORE_TIE_TOLERANCE:
                pairwise_credit += 0.5
            elif (score_delta > 0.0) == target_prefers_left:
                pairwise_credit += 1.0
    target_ranks = [target_position[int(row["candidate_index"])] + 1 for row in admissible]
    scores = [float(row["score"]) for row in admissible]
    spearman = _correlation(
        [-float(value) for value in target_ranks],
        _average_ranks(scores).tolist(),
    )
    kendall = _kendall_tau_b(target_ranks, scores)
    selected = score_ordered[0]
    oracle_best = target_ordered[0]
    oracle_best_rank = score_position[int(oracle_best["candidate_index"])] + 1
    selected_progress = float(selected["geodesic_progress_m"])
    oracle_progress = float(oracle_best["geodesic_progress_m"])
    target_progresses = [float(row["geodesic_progress_m"]) for row in target_ordered]
    progress_range = max(target_progresses) - min(target_progresses)
    regret = oracle_progress - selected_progress
    normalized_regret = 0.0 if progress_range == 0.0 else regret / progress_range
    if oracle_progress > 0.0:
        oracle_fraction = selected_progress / oracle_progress
    else:
        oracle_fraction = 1.0 if _target_key(selected) == _target_key(oracle_best) else 0.0
    oracle_fraction = float(min(1.0, max(0.0, oracle_fraction)))
    score_values = [float(row["score"]) for row in admissible]
    return {
        "state_id": values[0]["state_id"],
        "family": values[0]["family"],
        "model_id": values[0]["model_id"],
        "source_id": values[0]["source_id"],
        "admissible_candidate_count": len(admissible),
        "selected_candidate_index": int(selected["candidate_index"]),
        "oracle_best_candidate_index": int(oracle_best["candidate_index"]),
        "pairwise_accuracy": pairwise_credit / pair_count if pair_count else None,
        "pairwise_correct_credit": pairwise_credit,
        "pair_count": pair_count,
        "spearman": spearman,
        "kendall": kendall,
        "best_route_top1": oracle_best_rank == 1,
        "best_route_top3": oracle_best_rank <= 3,
        "reciprocal_rank": 1.0 / oracle_best_rank,
        "oracle_best_rank": oracle_best_rank,
        "selected_geodesic_progress_m": selected_progress,
        "selected_euclidean_progress_m": float(selected["euclidean_progress_m"]),
        "oracle_best_geodesic_progress_m": oracle_progress,
        "oracle_progress_fraction": oracle_fraction,
        "normalized_regret": float(max(0.0, normalized_regret)),
        "score_spread": max(score_values) - min(score_values),
        "population_id": population_id,
        "raw_selected_contact": bool(raw_selected["immediate_contact"]),
        "raw_selected_nonviable": not bool(raw_selected["successor_viable"]),
        "raw_selected_stuck": bool(raw_selected["stuck"]),
        "raw_selected_dead_end": bool(raw_selected["dead_end"]),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise NonGreedyMetricsError("aggregate metrics require states")
    pair_count = sum(int(row["pair_count"]) for row in values)
    credit = sum(float(row["pairwise_correct_credit"]) for row in values)
    scored = [row for row in values if row["oracle_best_rank"] is not None]
    output = {
        "state_count": len(values),
        "states_with_admissible_candidates": len(scored),
        "pair_count": pair_count,
        "pairwise_accuracy": credit / pair_count if pair_count else None,
        "spearman": _mean(values, "spearman"),
        "kendall": _mean(values, "kendall"),
        "best_route_top1": _mean(values, "best_route_top1"),
        "best_route_top3": _mean(values, "best_route_top3"),
        "mean_reciprocal_rank": _mean(values, "reciprocal_rank"),
        "mean_oracle_best_rank": _mean(values, "oracle_best_rank"),
        "selected_geodesic_progress_m": _mean(values, "selected_geodesic_progress_m"),
        "selected_euclidean_progress_m": _mean(values, "selected_euclidean_progress_m"),
        "oracle_best_geodesic_progress_m": _mean(
            values, "oracle_best_geodesic_progress_m"
        ),
        "oracle_progress_fraction": _mean(values, "oracle_progress_fraction"),
        "normalized_regret": _mean(values, "normalized_regret"),
        "mean_score_spread": _mean(values, "score_spread"),
        "complete_score_collapse": bool(
            not scored
            or all(float(row["score_spread"]) <= SCORE_TIE_TOLERANCE for row in scored)
        ),
        "descriptive_selected_adverse_counts": {
            "contact": sum(bool(row["raw_selected_contact"]) for row in values),
            "nonviable": sum(bool(row["raw_selected_nonviable"]) for row in values),
            "stuck": sum(bool(row["raw_selected_stuck"]) for row in values),
            "dead_end": sum(bool(row["raw_selected_dead_end"]) for row in values),
        },
    }
    return output


def source_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(dict(row))
    if len(by_state) != len(CONTRACT.DEVELOPMENT_HELDOUT_STATE_IDS):
        raise NonGreedyMetricsError("source metrics require all 16 heldout states")
    populations: dict[str, Any] = {}
    for population_id in POPULATION_IDS:
        states = [
            state_metrics(by_state[state_id], population_id=population_id)
            for state_id in sorted(by_state)
        ]
        families: dict[str, Any] = {}
        family_floor_checks: dict[str, Any] = {}
        for family in CONTRACT.FAMILY_IDS:
            family_rows = [row for row in states if row["family"] == family]
            if len(family_rows) != 4:
                raise NonGreedyMetricsError("source metrics require four states per family")
            families[family] = aggregate_metrics(family_rows)
            family_value = families[family]
            floor = CONTRACT.FAMILY_PERFORMANCE_FLOOR
            checks = {
                "pairwise_accuracy_at_least_half": (
                    _at_least(
                        family_value["pairwise_accuracy"],
                        floor["pairwise_accuracy_minimum_each_family"],
                    )
                ),
                "oracle_progress_fraction_at_least_half": (
                    _at_least(
                        family_value["oracle_progress_fraction"],
                        floor["oracle_progress_fraction_minimum_each_family"],
                    )
                ),
                "score_not_completely_collapsed": not family_value[
                    "complete_score_collapse"
                ],
            }
            family_floor_checks[family] = {"checks": checks, "pass": all(checks.values())}
        aggregate = aggregate_metrics(states)
        aggregate["family_performance_floor_checks"] = family_floor_checks
        aggregate["family_performance_floor_pass"] = all(
            value["pass"] for value in family_floor_checks.values()
        )
        aggregate["no_family_complete_collapse"] = not any(
            value["complete_score_collapse"] for value in families.values()
        )
        populations[population_id] = {
            "aggregate": aggregate,
            "families": families,
            "states": states,
        }
    viability = populations["ORACLE_VIABILITY_ADMISSIBLE"]
    return {
        "model_id": viability["states"][0]["model_id"],
        "source_id": viability["states"][0]["source_id"],
        "populations": populations,
        "aggregate": viability["aggregate"],
        "families": viability["families"],
        "states": viability["states"],
    }


def _pair_key(model_id: str, source_id: str) -> str:
    return f"{model_id}::{source_id}"


def _comparison(signal: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    signal_agg = signal["aggregate"]
    reference_agg = reference["aggregate"]
    return {
        "pairwise_accuracy_gain": _difference(
            signal_agg["pairwise_accuracy"], reference_agg["pairwise_accuracy"]
        ),
        "normalized_regret_reduction": _difference(
            reference_agg["normalized_regret"], signal_agg["normalized_regret"]
        ),
        "oracle_progress_fraction_gain": _difference(
            signal_agg["oracle_progress_fraction"],
            reference_agg["oracle_progress_fraction"],
        ),
    }


def _incremental_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = CONTRACT.STAGE_A_GATE["incremental_criteria"]
    checks = {
        "pairwise_accuracy_gain": _at_least(
            comparison["pairwise_accuracy_gain"],
            thresholds["pairwise_accuracy_gain_minimum"],
        ),
        "normalized_regret_reduction": _at_least(
            comparison["normalized_regret_reduction"],
            thresholds["normalized_regret_reduction_minimum"],
        ),
        "oracle_progress_fraction_gain": _at_least(
            comparison["oracle_progress_fraction_gain"],
            thresholds["oracle_progress_fraction_gain_minimum"],
        ),
    }
    return {
        "comparison": copy.deepcopy(dict(comparison)),
        "checks": checks,
        "criteria_passed": sum(checks.values()),
        "pass": sum(checks.values())
        >= CONTRACT.STAGE_A_GATE["incremental_minimum_criteria"],
    }


def _derangement_comparison(
    signal: Mapping[str, Any], deranged: Mapping[str, Any]
) -> dict[str, Any]:
    signal_agg = signal["aggregate"]
    deranged_agg = deranged["aggregate"]
    values = {
        "pairwise_accuracy_drop": _difference(
            signal_agg["pairwise_accuracy"], deranged_agg["pairwise_accuracy"]
        ),
        "normalized_regret_increase": _difference(
            deranged_agg["normalized_regret"], signal_agg["normalized_regret"]
        ),
        "oracle_progress_fraction_drop": _difference(
            signal_agg["oracle_progress_fraction"],
            deranged_agg["oracle_progress_fraction"],
        ),
    }
    thresholds = CONTRACT.STAGE_A_GATE["candidate_future_derangement_material_if_any"]
    checks = {
        "pairwise_accuracy_drop": _at_least(
            values["pairwise_accuracy_drop"],
            thresholds["pairwise_accuracy_drop_minimum"],
        ),
        "normalized_regret_increase": _at_least(
            values["normalized_regret_increase"],
            thresholds["normalized_regret_increase_minimum"],
        ),
        "oracle_progress_fraction_drop": _at_least(
            values["oracle_progress_fraction_drop"],
            thresholds["oracle_progress_fraction_drop_minimum"],
        ),
    }
    return {**values, "checks": checks, "material": any(checks.values())}


def _stage_a_gate(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pair = lambda model, source: metrics[_pair_key(model, source)]
    true = pair(CONTRACT.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "TRUE_FUTURE")
    kinematic = pair("DETERMINISTIC_KINEMATICS", "KINEMATIC")
    current = pair(CONTRACT.CURRENT_VISUAL_REACTIVE_RANKER, "CURRENT_VISUAL")
    candidate_deranged = pair(
        "FUTURE_TRAJECTORY_DERANGEMENT", "TRUE_FUTURE_DERANGED_CANDIDATE"
    )
    time_deranged = pair(
        "FUTURE_TIME_ORDER_DERANGEMENT", "TRUE_FUTURE_DERANGED_TIME"
    )
    aggregate = true["aggregate"]
    thresholds = CONTRACT.STAGE_A_GATE["absolute"]
    absolute_checks = {
        "pairwise_accuracy": _at_least(
            aggregate["pairwise_accuracy"], thresholds["pairwise_accuracy_minimum"]
        ),
        "spearman": _at_least(
            aggregate["spearman"], thresholds["spearman_minimum"]
        ),
        "normalized_regret": _at_most(
            aggregate["normalized_regret"], thresholds["normalized_regret_maximum"]
        ),
        "best_route_top3": _at_least(
            aggregate["best_route_top3"], thresholds["best_route_top3_minimum"]
        ),
        "oracle_progress_fraction": _at_least(
            aggregate["oracle_progress_fraction"],
            thresholds["oracle_progress_fraction_minimum"],
        ),
        "no_family_complete_collapse": aggregate["no_family_complete_collapse"] is True,
        "family_performance_floor": aggregate["family_performance_floor_pass"] is True,
    }
    kinematic_gate = _incremental_gate(_comparison(true, kinematic))
    current_gate = _incremental_gate(_comparison(true, current))
    candidate_derangement = _derangement_comparison(true, candidate_deranged)
    time_derangement = _derangement_comparison(true, time_deranged)
    absolute_pass = all(absolute_checks.values())
    passed = (
        absolute_pass
        and kinematic_gate["pass"]
        and current_gate["pass"]
        and candidate_derangement["material"]
    )
    if passed:
        classification = "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_SIGNAL"
    elif not absolute_pass:
        classification = "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL"
    elif not kinematic_gate["pass"] or not current_gate["pass"]:
        classification = "NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT"
    else:
        classification = "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL"
    return {
        "absolute_checks": absolute_checks,
        "incremental_over_kinematic": kinematic_gate,
        "incremental_over_current_visual": current_gate,
        "candidate_future_derangement": candidate_derangement,
        "time_order_derangement_descriptive": time_derangement,
        "pass": passed,
        "classification": classification,
        "next_decision": CONTRACT.NEXT_DECISION_BY_CLASSIFICATION[classification],
    }


def _stage_b_gate(
    metrics: Mapping[str, Mapping[str, Any]],
    *, stage_a_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    r1 = metrics[_pair_key(CONTRACT.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "R1")]
    rr = metrics[_pair_key(CONTRACT.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "RR")]
    true = stage_a_metrics[
        _pair_key(CONTRACT.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "TRUE_FUTURE")
    ]
    rr_agg, r1_agg, true_agg = rr["aggregate"], r1["aggregate"], true["aggregate"]
    thresholds = CONTRACT.STAGE_B_GATE["rr_absolute"]

    def absolute(source: Mapping[str, Any]) -> dict[str, Any]:
        aggregate = source["aggregate"]
        selected_progress = _optional_finite(aggregate["selected_geodesic_progress_m"])
        true_progress = _optional_finite(true_agg["selected_geodesic_progress_m"])
        if selected_progress is None or true_progress is None:
            true_fraction = None
        elif true_progress > 0.0:
            true_fraction = selected_progress / true_progress
        else:
            true_fraction = 1.0 if selected_progress == true_progress else 0.0
        checks = {
            "pairwise_accuracy": _at_least(
                aggregate["pairwise_accuracy"], thresholds["pairwise_accuracy_minimum"]
            ),
            "normalized_regret": _at_most(
                aggregate["normalized_regret"], thresholds["normalized_regret_maximum"]
            ),
            "best_route_top3": _at_least(
                aggregate["best_route_top3"], thresholds["best_route_top3_minimum"]
            ),
            "oracle_progress_fraction": _at_least(
                aggregate["oracle_progress_fraction"],
                thresholds["oracle_progress_fraction_minimum"],
            ),
            "true_selected_progress_fraction": _at_least(
                true_fraction, thresholds["true_selected_progress_fraction_minimum"]
            ),
            "no_family_complete_collapse": aggregate["no_family_complete_collapse"] is True,
            "family_performance_floor": aggregate["family_performance_floor_pass"] is True,
        }
        return {
            "checks": checks,
            "true_selected_progress_fraction": true_fraction,
            "pass": all(checks.values()),
        }

    r1_absolute = absolute(r1)
    rr_absolute = absolute(rr)
    delta = _comparison(rr, r1)
    relative = CONTRACT.STAGE_B_GATE["rr_over_r1"]
    relative_checks = {
        "pairwise_accuracy_gain": _at_least(
            delta["pairwise_accuracy_gain"],
            relative["pairwise_accuracy_gain_minimum"],
        ),
        "normalized_regret_reduction": _at_least(
            delta["normalized_regret_reduction"],
            relative["and_one_of"]["normalized_regret_reduction_minimum"],
        ),
        "oracle_progress_fraction_gain": _at_least(
            delta["oracle_progress_fraction_gain"],
            relative["and_one_of"]["oracle_progress_fraction_gain_minimum"],
        ),
    }
    relative_pass = relative_checks["pairwise_accuracy_gain"] and (
        relative_checks["normalized_regret_reduction"]
        or relative_checks["oracle_progress_fraction_gain"]
    )
    delta.update(
        {
            "selected_geodesic_progress_m_gain": _difference(
                rr_agg["selected_geodesic_progress_m"],
                r1_agg["selected_geodesic_progress_m"],
            ),
            "best_route_top3_gain": _difference(
                rr_agg["best_route_top3"], r1_agg["best_route_top3"]
            ),
        }
    )
    rr_vs_true = {
        "pairwise_accuracy_gap": _difference(
            rr_agg["pairwise_accuracy"], true_agg["pairwise_accuracy"]
        ),
        "normalized_regret_gap": _difference(
            rr_agg["normalized_regret"], true_agg["normalized_regret"]
        ),
        "oracle_progress_fraction_gap": _difference(
            rr_agg["oracle_progress_fraction"], true_agg["oracle_progress_fraction"]
        ),
        "selected_geodesic_progress_m_gap": _difference(
            rr_agg["selected_geodesic_progress_m"],
            true_agg["selected_geodesic_progress_m"],
        ),
        "best_route_top3_gap": _difference(
            rr_agg["best_route_top3"], true_agg["best_route_top3"]
        ),
    }
    passed = rr_absolute["pass"] and relative_pass
    if passed:
        classification = "NON_GREEDY_TWO_STEP_JEPA_PLANNING_SIGNAL"
    elif r1_absolute["pass"]:
        classification = "NON_GREEDY_JEPA_PLANNING_SIGNAL_NO_ROLLOUT_ADVANTAGE"
    else:
        classification = "NON_GREEDY_TRUE_FUTURE_SIGNAL_PREDICTOR_NO_GO"

    r1_states = {row["state_id"]: row for row in r1["states"]}
    rr_states = {row["state_id"]: row for row in rr["states"]}
    changed = [
        state_id
        for state_id in sorted(r1_states)
        if r1_states[state_id]["selected_candidate_index"]
        != rr_states[state_id]["selected_candidate_index"]
    ]
    family_effects = {
        family: {
            "pairwise_accuracy_gain": _difference(
                rr["families"][family]["pairwise_accuracy"],
                r1["families"][family]["pairwise_accuracy"],
            ),
            "normalized_regret_reduction": _difference(
                r1["families"][family]["normalized_regret"],
                rr["families"][family]["normalized_regret"],
            ),
            "oracle_progress_fraction_gain": _difference(
                rr["families"][family]["oracle_progress_fraction"],
                r1["families"][family]["oracle_progress_fraction"],
            ),
        }
        for family in CONTRACT.FAMILY_IDS
    }
    return {
        "r1_absolute": r1_absolute,
        "rr_absolute": rr_absolute,
        "rr_over_r1": {"comparison": delta, "checks": relative_checks, "pass": relative_pass},
        "rr_versus_true_future": rr_vs_true,
        "paired_selection_changes": {
            "changed_state_ids": changed,
            "changed_state_count": len(changed),
            "changed_state_fraction": len(changed) / len(r1_states),
        },
        "per_family_effects": family_effects,
        "pass": passed,
        "classification": classification,
        "next_decision": CONTRACT.NEXT_DECISION_BY_CLASSIFICATION[classification],
    }


def _reduce_stage_a(
    stage_a_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_a = _validate_rows(stage_a_rows, authority_key="stage_a")
    grouped_a: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stage_a:
        grouped_a[(row["model_id"], row["source_id"])].append(row)
    metrics_a = {
        _pair_key(model_id, source_id): source_metrics(grouped_a[(model_id, source_id)])
        for model_id, source_id in CONTRACT.STAGE_A_MODEL_SOURCE_PAIRS
    }
    gate_a = _stage_a_gate(metrics_a)
    return metrics_a, gate_a


def recompute_stage_a_metrics_and_gate(
    stage_a_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the provisional Stage-A reduction without requiring conditional B."""

    metrics_a, gate_a = _reduce_stage_a(stage_a_rows)
    return CONTRACT.attach_content_digest(
        {
            "schema": "non_greedy_local_subgoal_jepa_planning_v1.stage_a_metrics.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "stage_a": {"sources": metrics_a, "gate": gate_a},
            "decision": {
                "stage_a_authorizes_predictor_substitution": bool(gate_a["pass"]),
                "primary_classification": gate_a["classification"],
                "secondary_classifications": [],
                "next_experiment": gate_a["next_decision"],
            },
        }
    )


def recompute_metrics_and_gates(
    stage_a_rows: Sequence[Mapping[str, Any]],
    conditional_stage_b_rows_or_none: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Rebuild the canonical final metrics mapping from exact heldout ledgers."""

    metrics_a, gate_a = _reduce_stage_a(stage_a_rows)

    stage_b_present = conditional_stage_b_rows_or_none is not None
    if stage_b_present != bool(gate_a["pass"]):
        raise NonGreedyMetricsError(
            "conditional Stage-B presence contradicts recomputed Stage-A gate"
        )
    metrics_b: dict[str, Any] | None = None
    gate_b: dict[str, Any] | None = None
    if conditional_stage_b_rows_or_none is not None:
        stage_b = _validate_rows(
            conditional_stage_b_rows_or_none, authority_key="conditional_stage_b"
        )
        grouped_b: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in stage_b:
            grouped_b[(row["model_id"], row["source_id"])].append(row)
        metrics_b = {
            _pair_key(model_id, source_id): source_metrics(grouped_b[(model_id, source_id)])
            for model_id, source_id in CONTRACT.STAGE_B_MODEL_SOURCE_PAIRS
        }
        gate_b = _stage_b_gate(metrics_b, stage_a_metrics=metrics_a)
    final_classification = (
        gate_a["classification"] if gate_b is None else gate_b["classification"]
    )
    secondary_classifications = []
    if gate_a["candidate_future_derangement"]["material"]:
        secondary_classifications.append("CANDIDATE_FUTURE_DERANGEMENT_MATERIAL")
    if gate_a["time_order_derangement_descriptive"]["material"]:
        secondary_classifications.append("FUTURE_TIME_ORDER_DERANGEMENT_MATERIAL")
    next_decision = CONTRACT.NEXT_DECISION_BY_CLASSIFICATION[final_classification]
    output = {
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.metrics.v1",
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "score_row_authority": score_row_authority(),
        "stage_a": {"sources": metrics_a, "gate": gate_a},
        "conditional_stage_b": (
            None if metrics_b is None else {"sources": metrics_b, "gate": gate_b}
        ),
        "decision": {
            "stage_a_authorizes_predictor_substitution": bool(gate_a["pass"]),
            "primary_classification": final_classification,
            "secondary_classifications": secondary_classifications,
            "next_experiment": next_decision,
        },
        "final_classification": final_classification,
        "next_decision": next_decision,
        "descriptive_outcomes_are_not_admissibility_training_targets": True,
    }
    return CONTRACT.attach_content_digest(output)


__all__ = [
    "BOOLEAN_FIELDS",
    "CONDITIONAL_STAGE_B",
    "FINITE_NUMERIC_FIELDS",
    "NonGreedyMetricsError",
    "SCORE_ROW_REQUIRED_FIELDS",
    "STAGE_A",
    "aggregate_metrics",
    "recompute_metrics_and_gates",
    "recompute_stage_a_metrics_and_gate",
    "score_row_authority",
    "source_metrics",
    "state_metrics",
]
