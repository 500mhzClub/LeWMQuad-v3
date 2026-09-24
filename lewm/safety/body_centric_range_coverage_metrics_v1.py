"""Pure metrics and two-ply reducers for range-coverage qualification V1.

The module deliberately performs no file I/O, simulation, model inference, or
sensor materialisation.  A caller supplies already-materialised clearances and
state/action records.  Clearance is contact-positive when it is less than or
equal to the calibrated threshold.  Unsupported or non-finite observations
are always treated as contact risk; they are never interpreted as free space.

The state reducer accepts this compact schema::

    {
        "state_id": "...",
        "family": "...",
        "current_actions": [
            {
                "action_index": 0,
                "controller": "route",
                "applied_action": [vx, vy, yaw_rate],  # optional dedup key
                "oracle_contact": False,
                "predicted_contact": False,
                "h3_progress_m": 0.1,
                "h3_heading_improvement_rad": 0.0,
                "decision_progress_m": 0.1,
                "next_actions": [
                    {
                        "action_index": 0,
                        "controller": "route",
                        "applied_action": [...],
                        "oracle_contact": False,
                        "predicted_contact": False,
                    },
                ],
            },
        ],
    }

``current_rows`` is accepted as an alias for ``current_actions``.  Oracle
contact may be named ``current_contact``/``contact`` for compatibility with
the frozen corpus rows.  Predictions must always be explicit.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import math
import resource
import time
from typing import Any

import numpy as np


H3_PROGRESS_TIE_M = 0.03
H3_TIE_ABS_TOLERANCE_M = 1e-12
MIN_CALIBRATION_RECALL = 0.95
MAX_CALIBRATION_FNR = 0.05


def _one_dimensional(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _same_length(reference: np.ndarray, value: np.ndarray, *, name: str) -> None:
    if len(value) != len(reference):
        raise ValueError(f"{name} has {len(value)} rows, expected {len(reference)}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with simultaneous, order-free ties."""

    values = _one_dimensional(values, name="values", dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("rank values must not contain NaN")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        # Positions are one based; the inclusive rank interval is
        # [start + 1, stop].
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def tie_aware_auc(labels: Any, scores: Any) -> float | None:
    """Mann--Whitney AUC with average ranks for tied scores."""

    truth = _one_dimensional(labels, name="labels", dtype=bool)
    value = _one_dimensional(scores, name="scores", dtype=np.float64)
    _same_length(truth, value, name="scores")
    if np.isnan(value).any():
        raise ValueError("AUC scores must not contain NaN")
    positive = int(truth.sum())
    negative = len(truth) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = _average_ranks(value)
    numerator = float(ranks[truth].sum()) - positive * (positive + 1) / 2.0
    return numerator / (positive * negative)


def tie_aware_average_precision(labels: Any, scores: Any) -> float | None:
    """Threshold AP where an equal-score group enters simultaneously.

    This definition is invariant to input ordering inside a score tie.  At
    each distinct descending score, the group's recall increment is weighted
    by precision after the complete group has entered.
    """

    truth = _one_dimensional(labels, name="labels", dtype=bool)
    value = _one_dimensional(scores, name="scores", dtype=np.float64)
    _same_length(truth, value, name="scores")
    if np.isnan(value).any():
        raise ValueError("AP scores must not contain NaN")
    positive = int(truth.sum())
    if positive == 0:
        return None
    order = np.argsort(-value, kind="mergesort")
    selected = 0
    true_selected = 0
    result = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and value[order[stop]] == value[order[start]]:
            stop += 1
        group_true = int(truth[order[start:stop]].sum())
        selected += stop - start
        true_selected += group_true
        result += (group_true / positive) * (true_selected / selected)
        start = stop
    return float(result)


def tie_aware_spearman(left: Any, right: Any) -> float:
    """Spearman correlation using average ranks on both sides."""

    left_array = _one_dimensional(left, name="left", dtype=np.float64)
    right_array = _one_dimensional(right, name="right", dtype=np.float64)
    _same_length(left_array, right_array, name="right")
    if not len(left_array):
        return 0.0
    left_rank = _average_ranks(left_array)
    right_rank = _average_ranks(right_array)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


# Short aliases are useful in evaluators while retaining explicit public names.
auc = tie_aware_auc
average_precision = tie_aware_average_precision
spearman = tie_aware_spearman


def conservative_risk_scores(clearance_m: Any, unsupported: Any | None = None) -> np.ndarray:
    """Convert clearance to contact-risk score; unsupported becomes +inf."""

    clearance = _one_dimensional(clearance_m, name="clearance_m", dtype=np.float64)
    unsupported_array = (
        np.zeros(len(clearance), dtype=bool)
        if unsupported is None
        else _one_dimensional(unsupported, name="unsupported", dtype=bool)
    )
    _same_length(clearance, unsupported_array, name="unsupported")
    unsupported_array = unsupported_array | ~np.isfinite(clearance)
    score = -clearance
    score[unsupported_array] = np.inf
    return score


def contact_predictions(
    clearance_m: Any,
    threshold_m: float,
    unsupported: Any | None = None,
) -> np.ndarray:
    """Predict contact, including equality and unsupported observations."""

    threshold = float(threshold_m)
    if not math.isfinite(threshold):
        raise ValueError("contact threshold must be finite")
    clearance = _one_dimensional(clearance_m, name="clearance_m", dtype=np.float64)
    unsupported_array = (
        np.zeros(len(clearance), dtype=bool)
        if unsupported is None
        else _one_dimensional(unsupported, name="unsupported", dtype=bool)
    )
    _same_length(clearance, unsupported_array, name="unsupported")
    # Non-finite values carry no defensible free-space evidence, including
    # +inf produced by an empty point cloud.
    unsupported_array = unsupported_array | ~np.isfinite(clearance)
    return unsupported_array | (clearance <= threshold)


def contact_confusion(labels: Any, predicted_contact: Any) -> dict[str, int | float | None]:
    truth = _one_dimensional(labels, name="labels", dtype=bool)
    predicted = _one_dimensional(
        predicted_contact, name="predicted_contact", dtype=bool
    )
    _same_length(truth, predicted, name="predicted_contact")
    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())
    positive = tp + fn
    negative = tn + fp
    recall = None if positive == 0 else tp / positive
    fnr = None if positive == 0 else fn / positive
    retention = None if negative == 0 else tn / negative
    return {
        "rows": len(truth),
        "positives": positive,
        "negatives": negative,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "fnr": fnr,
        "negative_retention": retention,
    }


def contact_summary(
    labels: Any,
    clearance_m: Any,
    threshold_m: float,
    unsupported: Any | None = None,
) -> dict[str, int | float | None]:
    truth = _one_dimensional(labels, name="labels", dtype=bool)
    clearance = _one_dimensional(clearance_m, name="clearance_m", dtype=np.float64)
    _same_length(truth, clearance, name="clearance_m")
    unsupported_array = (
        np.zeros(len(truth), dtype=bool)
        if unsupported is None
        else _one_dimensional(unsupported, name="unsupported", dtype=bool)
    )
    _same_length(truth, unsupported_array, name="unsupported")
    unsupported_array = unsupported_array | ~np.isfinite(clearance)
    prediction = contact_predictions(clearance, threshold_m, unsupported_array)
    result = contact_confusion(truth, prediction)
    risk = conservative_risk_scores(clearance, unsupported_array)
    result.update(
        {
            "auc": tie_aware_auc(truth, risk),
            "average_precision": tie_aware_average_precision(truth, risk),
            "threshold_m": float(threshold_m),
            "unsupported_rows": int(unsupported_array.sum()),
            "supported_rows": int((~unsupported_array).sum()),
        }
    )
    return result


def current_successor_contact_summaries(
    *,
    current_labels: Any,
    current_clearance_m: Any,
    successor_labels: Any,
    successor_clearance_m: Any,
    threshold_m: float,
    current_unsupported: Any | None = None,
    successor_unsupported: Any | None = None,
) -> dict[str, dict[str, int | float | None]]:
    current_labels_array = _one_dimensional(
        current_labels, name="current_labels", dtype=bool
    )
    successor_labels_array = _one_dimensional(
        successor_labels, name="successor_labels", dtype=bool
    )
    current_clearance_array = _one_dimensional(
        current_clearance_m, name="current_clearance_m", dtype=np.float64
    )
    successor_clearance_array = _one_dimensional(
        successor_clearance_m, name="successor_clearance_m", dtype=np.float64
    )
    current_unsupported_array = (
        np.zeros(len(current_labels_array), dtype=bool)
        if current_unsupported is None
        else _one_dimensional(
            current_unsupported, name="current_unsupported", dtype=bool
        )
    )
    successor_unsupported_array = (
        np.zeros(len(successor_labels_array), dtype=bool)
        if successor_unsupported is None
        else _one_dimensional(
            successor_unsupported, name="successor_unsupported", dtype=bool
        )
    )
    return {
        "current": contact_summary(
            current_labels_array,
            current_clearance_array,
            threshold_m,
            current_unsupported_array,
        ),
        "successor": contact_summary(
            successor_labels_array,
            successor_clearance_array,
            threshold_m,
            successor_unsupported_array,
        ),
        "combined": contact_summary(
            np.concatenate((current_labels_array, successor_labels_array)),
            np.concatenate((current_clearance_array, successor_clearance_array)),
            threshold_m,
            np.concatenate(
                (current_unsupported_array, successor_unsupported_array)
            ),
        ),
    }


def grouped_contact_summaries(
    labels: Any,
    clearance_m: Any,
    groups: Sequence[str],
    threshold_m: float,
    unsupported: Any | None = None,
) -> dict[str, dict[str, int | float | None]]:
    truth = _one_dimensional(labels, name="labels", dtype=bool)
    clearance = _one_dimensional(clearance_m, name="clearance_m", dtype=np.float64)
    if len(groups) != len(truth):
        raise ValueError("groups must have one entry per contact row")
    unsupported_array = (
        np.zeros(len(truth), dtype=bool)
        if unsupported is None
        else _one_dimensional(unsupported, name="unsupported", dtype=bool)
    )
    output: dict[str, dict[str, int | float | None]] = {}
    group_array = np.asarray(groups, dtype=str)
    for group in sorted(set(group_array.tolist())):
        mask = group_array == group
        output[group] = contact_summary(
            truth[mask], clearance[mask], threshold_m, unsupported_array[mask]
        )
    return output


def _calibration_thresholds(clearance_m: np.ndarray) -> np.ndarray:
    finite = np.unique(clearance_m[np.isfinite(clearance_m)])
    if not len(finite):
        # Threshold is observationally irrelevant when every row is
        # unsupported; zero is a deterministic signed-clearance convention.
        return np.asarray([0.0], dtype=np.float64)
    return np.concatenate(
        (
            [np.nextafter(finite[0], -np.inf)],
            finite,
            [np.nextafter(finite[-1], np.inf)],
        )
    )


def _required_decision_value(decision: Mapping[str, Any], name: str) -> float:
    aliases = {
        "viable_retained": (
            "states_retaining_admitted_action",
            "states_retaining_viable_action",
        ),
        "nonviable_abstentions": ("correct_abstentions",),
        "h3_progress": ("selected_h3_route_progress_m", "route_progress_m"),
        "normalized_regret": (
            "normalized_regret",
            "normalized_viability_regret",
        ),
        "top3": ("best_admissible_top3", "best_viability_admissible_top3"),
    }
    for key in aliases[name]:
        if key in decision:
            return float(decision[key])
    raise KeyError(f"decision summary lacks {name}: expected {aliases[name]}")


def enumerate_threshold_frontier(
    labels: Any,
    clearance_m: Any,
    decision_at_threshold: Callable[[float], Mapping[str, Any]],
    *,
    unsupported: Any | None = None,
    minimum_recall: float = MIN_CALIBRATION_RECALL,
    maximum_fnr: float = MAX_CALIBRATION_FNR,
) -> dict[str, Any]:
    """Enumerate and select the prospectively ordered calibration frontier."""

    truth = _one_dimensional(labels, name="labels", dtype=bool)
    clearance = _one_dimensional(clearance_m, name="clearance_m", dtype=np.float64)
    _same_length(truth, clearance, name="clearance_m")
    if not truth.any():
        raise ValueError("calibration requires at least one contact-positive row")
    if truth.all():
        raise ValueError("calibration requires at least one contact-negative row")
    unsupported_array = (
        np.zeros(len(truth), dtype=bool)
        if unsupported is None
        else _one_dimensional(unsupported, name="unsupported", dtype=bool)
    )
    _same_length(truth, unsupported_array, name="unsupported")
    thresholds = _calibration_thresholds(clearance)
    frontier: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        contact = contact_summary(
            truth, clearance, threshold_value, unsupported_array
        )
        decision = dict(decision_at_threshold(threshold_value))
        recall = contact["recall"]
        fnr = contact["fnr"]
        is_eligible = bool(
            recall is not None
            and fnr is not None
            and recall >= float(minimum_recall)
            and fnr <= float(maximum_fnr)
        )
        key: tuple[float, ...] | None = None
        if is_eligible:
            retention = contact["negative_retention"]
            if retention is None:
                raise ValueError("negative retention is undefined")
            key = (
                float(retention),
                _required_decision_value(decision, "viable_retained"),
                _required_decision_value(decision, "nonviable_abstentions"),
                _required_decision_value(decision, "h3_progress"),
                -_required_decision_value(decision, "normalized_regret"),
                _required_decision_value(decision, "top3"),
                threshold_value,
            )
        row = {
            "threshold_m": threshold_value,
            "eligible": is_eligible,
            "contact": contact,
            "decision": decision,
            "lexicographic_key": None if key is None else list(key),
        }
        frontier.append(row)
        if is_eligible:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no threshold satisfies contact recall/FNR calibration")
    selected = max(
        eligible, key=lambda row: tuple(row["lexicographic_key"])
    )
    return {
        "frontier_points": len(frontier),
        "eligible_points": len(eligible),
        "selection_key": [
            "negative_retention",
            "viable_retained",
            "nonviable_abstentions",
            "h3_progress",
            "negative_normalized_regret",
            "best_admissible_top3",
            "conservative_threshold_m",
        ],
        "selected_threshold_m": selected["threshold_m"],
        "selected": selected,
        "frontier": frontier,
    }


def action_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if "action_identity" in row:
        identity = row["action_identity"]
        return tuple(identity) if isinstance(identity, (list, tuple)) else (identity,)
    controller = str(row.get("controller", "route"))
    if row.get("applied_action") is not None:
        return (
            controller,
            *(round(float(value), 7) for value in row["applied_action"]),
        )
    return (controller, int(row["action_index"]))


def unique_action_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        identity = action_identity(row)
        if identity not in observed:
            observed.add(identity)
            output.append(row)
    return output


def _oracle_contact(row: Mapping[str, Any]) -> bool:
    for key in ("oracle_contact", "current_contact", "contact"):
        if key in row:
            return bool(row[key])
    raise KeyError("action row lacks oracle contact")


def _predicted_contact(row: Mapping[str, Any]) -> bool:
    if "predicted_contact" not in row:
        raise KeyError("action row lacks predicted_contact")
    return bool(row["predicted_contact"])


def h3_route_order(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return row positions under the frozen 0.03 m H3 tie band."""

    remaining = list(range(len(rows)))
    output: list[int] = []
    while remaining:
        greatest = max(float(rows[index]["h3_progress_m"]) for index in remaining)
        tied = [
            index
            for index in remaining
            if greatest - float(rows[index]["h3_progress_m"])
            <= H3_PROGRESS_TIE_M + H3_TIE_ABS_TOLERANCE_M
        ]
        chosen = min(
            tied,
            key=lambda index: (
                -float(rows[index]["h3_heading_improvement_rad"]),
                int(rows[index]["action_index"]),
            ),
        )
        output.append(chosen)
        remaining.remove(chosen)
    return output


def lateral_fallback_order(
    rows: Sequence[Mapping[str, Any]], safe_counts: Mapping[int, int]
) -> list[int]:
    lateral = [
        index for index, row in enumerate(rows) if int(row["action_index"]) >= 12
    ]
    return sorted(
        lateral,
        key=lambda index: (
            -int(safe_counts[int(rows[index]["action_index"])]),
            int(rows[index]["action_index"]),
        ),
    )


def ordered_admitted_actions(
    rows: Sequence[Mapping[str, Any]],
    admitted: Mapping[int, bool],
    safe_counts: Mapping[int, int],
) -> list[int]:
    route = [
        row
        for row in rows
        if int(row["action_index"]) < 12
        and bool(admitted.get(int(row["action_index"]), False))
    ]
    if route:
        return [int(route[index]["action_index"]) for index in h3_route_order(route)]
    lateral = [
        row
        for row in rows
        if int(row["action_index"]) >= 12
        and bool(admitted.get(int(row["action_index"]), False))
    ]
    ordered = lateral_fallback_order(lateral, safe_counts)
    return [int(lateral[index]["action_index"]) for index in ordered]


def select_action(
    rows: Sequence[Mapping[str, Any]],
    admitted: Mapping[int, bool],
    safe_counts: Mapping[int, int],
) -> int | None:
    ordered = ordered_admitted_actions(rows, admitted, safe_counts)
    return None if not ordered else ordered[0]


def safe_action_count_summary(truth: Any, predicted: Any) -> dict[str, int | float]:
    target = _one_dimensional(truth, name="truth", dtype=np.int64)
    estimate = _one_dimensional(predicted, name="predicted", dtype=np.int64)
    _same_length(target, estimate, name="predicted")
    if np.any(target < 0) or np.any(estimate < 0):
        raise ValueError("safe-action counts must be non-negative")
    if not len(target):
        return {
            "rows": 0,
            "mae": 0.0,
            "spearman": 0.0,
            "exact_count_accuracy": 0.0,
            "zero_vs_nonzero_accuracy": 0.0,
            "false_zero_rate": 0.0,
            "false_nonzero_rate": 0.0,
            "margin_ge_1": 0,
            "margin_ge_2": 0,
            "margin_ge_3": 0,
        }
    true_nonzero = target > 0
    predicted_nonzero = estimate > 0
    return {
        "rows": len(target),
        "mae": float(np.abs(target - estimate).mean()),
        "spearman": tie_aware_spearman(target, estimate),
        "exact_count_accuracy": float((target == estimate).mean()),
        "zero_vs_nonzero_accuracy": float((true_nonzero == predicted_nonzero).mean()),
        "false_zero_rate": (
            float((~predicted_nonzero[true_nonzero]).mean())
            if true_nonzero.any()
            else 0.0
        ),
        "false_nonzero_rate": (
            float((predicted_nonzero[~true_nonzero]).mean())
            if (~true_nonzero).any()
            else 0.0
        ),
        "margin_ge_1": int((estimate >= 1).sum()),
        "margin_ge_2": int((estimate >= 2).sum()),
        "margin_ge_3": int((estimate >= 3).sum()),
    }


def _current_actions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = state.get("current_actions", state.get("current_rows"))
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("state current_actions/current_rows must be a sequence")
    return unique_action_rows(rows)


def _successor_rows(
    state: Mapping[str, Any], current: Mapping[str, Any]
) -> list[Mapping[str, Any]] | None:
    if current.get("next_actions") is not None:
        return unique_action_rows(current["next_actions"])
    source = state.get("successor_actions", state.get("successor_rows"))
    if source is None:
        return None
    action_index = int(current["action_index"])
    if isinstance(source, Mapping):
        value = source.get(action_index, source.get(str(action_index)))
        if value is None:
            return None
        if isinstance(value, Mapping) and "next_actions" in value:
            value = value["next_actions"]
        return unique_action_rows(value)
    for row in source:
        if int(row["current_action_index"]) == action_index:
            return unique_action_rows(row["next_actions"])
    return None


def reduce_two_ply_state(state: Mapping[str, Any]) -> dict[str, Any]:
    current = _current_actions(state)
    by_index = {int(row["action_index"]): row for row in current}
    if len(by_index) != len(current):
        raise ValueError("unique current actions repeat an action index")
    true_current: dict[int, bool] = {}
    predicted_current: dict[int, bool] = {}
    true_count: dict[int, int] = {}
    predicted_count: dict[int, int] = {}
    true_viable: dict[int, bool] = {}
    admitted: dict[int, bool] = {}
    count_truth: list[int] = []
    count_prediction: list[int] = []
    for action_index, row in by_index.items():
        true_current[action_index] = _oracle_contact(row)
        predicted_current[action_index] = _predicted_contact(row)
        next_rows = _successor_rows(state, row)
        if next_rows is None:
            true_count[action_index] = -1
            predicted_count[action_index] = -1
        else:
            true_count[action_index] = sum(not _oracle_contact(item) for item in next_rows)
            predicted_count[action_index] = sum(
                not _predicted_contact(item) for item in next_rows
            )
        true_viable[action_index] = (
            not true_current[action_index] and true_count[action_index] >= 1
        )
        admitted[action_index] = (
            not predicted_current[action_index]
            and predicted_count[action_index] >= 1
        )
        # Safe-next-action-count reconstruction is defined for every current
        # candidate whose frozen successor set is present, independently of
        # whether the current tick itself contacts.  Current-tick contact is a
        # separate admission condition and must not narrow this denominator.
        if true_count[action_index] >= 0:
            count_truth.append(true_count[action_index])
            count_prediction.append(predicted_count[action_index])

    choice = select_action(current, admitted, predicted_count)
    oracle_choice = select_action(current, true_viable, true_count)
    oracle_viable = any(true_viable.values())
    retained = any(
        admitted[index] and true_viable[index] for index in by_index
    )
    selected_contact = choice is not None and true_current[choice]
    selected_nonviable = (
        choice is not None
        and not true_current[choice]
        and true_count[choice] == 0
    )
    progress = (
        0.0 if choice is None else float(by_index[choice]["decision_progress_m"])
    )
    oracle_progress = (
        0.0
        if oracle_choice is None
        else float(by_index[oracle_choice]["decision_progress_m"])
    )
    ranked = ordered_admitted_actions(current, admitted, predicted_count)
    top1_hit = bool(oracle_viable and oracle_choice in ranked[:1])
    top3_hit = bool(oracle_viable and oracle_choice in ranked[:3])
    selected_predicted_count = -1 if choice is None else predicted_count[choice]
    return {
        "state_id": str(state["state_id"]),
        "family": str(state["family"]),
        "unique_current_actions": len(current),
        "oracle_viable": oracle_viable,
        "oracle_viable_action_count": sum(true_viable.values()),
        "retained": retained,
        "admitted_count": sum(admitted.values()),
        "selected": choice,
        "oracle_selected": oracle_choice,
        "selected_immediate_contact": bool(selected_contact),
        "selected_nonviable_successor": bool(selected_nonviable),
        "false_abstention": bool(oracle_viable and choice is None),
        "correct_abstention": bool(not oracle_viable and choice is None),
        "unsafe_movement": bool(not oracle_viable and choice is not None),
        "falsely_viable_candidates": (
            int(sum(admitted.values())) if not oracle_viable else 0
        ),
        "progress_m": progress,
        "oracle_progress_m": oracle_progress,
        "normalized_regret_numerator": (
            max(0.0, oracle_progress - progress) if oracle_viable else 0.0
        ),
        "normalized_regret_denominator": (
            max(abs(oracle_progress), 1e-6) if oracle_viable else 0.0
        ),
        "best_admissible_top1_hit": top1_hit,
        "best_admissible_top3_hit": top3_hit,
        "predicted_selected_safe_count": selected_predicted_count,
        "selected_safe_margin_ge_1": selected_predicted_count >= 1,
        "selected_safe_margin_ge_2": selected_predicted_count >= 2,
        "selected_safe_margin_ge_3": selected_predicted_count >= 3,
        "true_current_contact": true_current,
        "predicted_current_contact": predicted_current,
        "true_safe_counts": true_count,
        "predicted_safe_counts": predicted_count,
        "true_viable": true_viable,
        "admitted": admitted,
        "ranked_admitted_actions": ranked,
        "count_truth": count_truth,
        "count_prediction": count_prediction,
    }


def _reduce_state_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    viable = [row for row in rows if row["oracle_viable"]]
    nonviable = [row for row in rows if not row["oracle_viable"]]
    count_truth = [value for row in rows for value in row["count_truth"]]
    count_prediction = [value for row in rows for value in row["count_prediction"]]
    progress = sum(float(row["progress_m"]) for row in viable)
    oracle_progress = sum(float(row["oracle_progress_m"]) for row in viable)
    regret_numerator = sum(
        float(row["normalized_regret_numerator"]) for row in viable
    )
    regret_denominator = sum(
        float(row["normalized_regret_denominator"]) for row in viable
    )
    top_denominator = len(viable)
    result = {
        "states": len(rows),
        "oracle_viable_states": len(viable),
        "states_retaining_admitted_action": sum(bool(row["retained"]) for row in viable),
        "selected_immediate_contacts": sum(
            bool(row["selected_immediate_contact"]) for row in rows
        ),
        "selected_oracle_nonviable_successors": sum(
            bool(row["selected_nonviable_successor"]) for row in rows
        ),
        "false_abstentions": sum(bool(row["false_abstention"]) for row in viable),
        "oracle_nonviable_states": len(nonviable),
        "correct_abstentions": sum(
            bool(row["correct_abstention"]) for row in nonviable
        ),
        "unsafe_movement_decisions": sum(
            bool(row["unsafe_movement"]) for row in nonviable
        ),
        "falsely_viable_candidates": sum(
            int(row["falsely_viable_candidates"]) for row in nonviable
        ),
        "selected_h3_route_progress_m": progress,
        "oracle_h3_route_progress_m": oracle_progress,
        "oracle_progress_fraction": progress / max(abs(oracle_progress), 1e-9),
        "normalized_regret": regret_numerator / max(regret_denominator, 1e-9),
        "best_admissible_top1": (
            sum(bool(row["best_admissible_top1_hit"]) for row in viable)
            / max(1, top_denominator)
        ),
        "best_admissible_top3": (
            sum(bool(row["best_admissible_top3_hit"]) for row in viable)
            / max(1, top_denominator)
        ),
        "selected_safe_margin_ge_1": sum(
            bool(row["selected_safe_margin_ge_1"]) for row in rows
        ),
        "selected_safe_margin_ge_2": sum(
            bool(row["selected_safe_margin_ge_2"]) for row in rows
        ),
        "selected_safe_margin_ge_3": sum(
            bool(row["selected_safe_margin_ge_3"]) for row in rows
        ),
        "safe_action_count": safe_action_count_summary(
            count_truth, count_prediction
        ),
    }
    result["family_collapse"] = bool(
        result["oracle_viable_states"] > 0
        and result["states_retaining_admitted_action"] == 0
    )
    return result


def reduce_two_ply_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_state = [reduce_two_ply_state(state) for state in states]
    result = _reduce_state_results(per_state)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_state:
        grouped[str(row["family"])].append(row)
    per_family = {
        family: _reduce_state_results(grouped[family]) for family in sorted(grouped)
    }
    result["per_family"] = per_family
    result["per_state"] = per_state
    result["no_family_collapse"] = all(
        not row["family_collapse"] for row in per_family.values()
    )
    return result


def benchmark_complete_prematerialized_decision(
    states: Sequence[Mapping[str, Any]],
    reducer: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] = reduce_two_ply_states,
    *,
    warmups: int = 10,
    iterations: int = 100,
) -> dict[str, int | float | str | bool]:
    """Benchmark one complete pre-materialised held-out decision reduction."""

    if not states:
        raise ValueError("benchmark requires at least one state")
    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations positive")
    current_actions = sum(len(_current_actions(state)) for state in states)
    next_action_sets = 0
    next_actions = 0
    for state in states:
        for row in _current_actions(state):
            successors = _successor_rows(state, row)
            if successors is not None:
                next_action_sets += 1
                next_actions += len(successors)
    samples: list[float] = []
    for iteration in range(warmups + iterations):
        started = time.perf_counter_ns()
        reducer(states)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if iteration >= warmups:
            samples.append(elapsed_ms)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "schema": "body_centric_range_coverage_complete_set_benchmark_v1",
        "device": "CPU",
        "scope": (
            "complete pre-materialised set: all current actions, all available "
            "next-action sets, safe counts, threshold decisions, and H3 selection"
        ),
        "includes_ray_generation": False,
        "includes_future_trajectory_acquisition": False,
        "complete_set_each_iteration": True,
        "states": len(states),
        "current_actions": current_actions,
        "next_action_sets": next_action_sets,
        "next_actions": next_actions,
        "warmups": warmups,
        "iterations": iterations,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "misses_50ms": int((values > 50).sum()),
        "misses_80ms": int((values > 80).sum()),
        "misses_100ms": int((values > 100).sum()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_vram_bytes": 0,
    }
