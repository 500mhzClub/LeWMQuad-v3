"""Exploration-phase summaries for Phase 3A no-beacon navigation."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from .phase3a_marker_memory import (
    advance_marker_egocentric,
    egocentric_marker_memory_delta,
    egocentric_marker_memory_score,
    egocentric_marker_memory_predictions,
    primitive_sequence_from_blocks,
    remembered_marker_position,
)
from .phase3a_training import source_key


EXPLORE_CLAIM_PHASES = (
    "explore_unseen",
    "discover_visible_marker",
    "claim_after_marker_seen",
)

TOP_K_VALUES = (1, 3, 5, 10)


def _label(row: Mapping, name: str, default: object = 0.0) -> object:
    labels = row.get("consequence_labels", {})
    if isinstance(labels, Mapping) and name in labels:
        return labels[name]
    return row.get(name, default)


def explore_claim_phase(rows: Sequence[Mapping]) -> str:
    """Classify a same-source candidate group by the available goal evidence."""

    if not rows:
        raise ValueError("rows must not be empty")
    known_before = any(
        bool(_label(row, "goal_known_before_candidate", False)) for row in rows
    )
    if known_before:
        return "claim_after_marker_seen"
    future_marker_available = any(
        bool(_label(row, "future_goal_marker_seen", False)) for row in rows
    )
    if future_marker_available:
        return "discover_visible_marker"
    return "explore_unseen"


def summarize_explore_claim_predictions(
    rows: Sequence[Mapping],
    utility_predictions: Sequence[float],
) -> dict:
    """Summarize action selection by exploration phase.

    Selection is computed over every candidate in each source-state group. The
    phase is assigned to the whole group, avoiding invalid comparisons where a
    candidate subset would hide alternatives from the policy.
    """

    if len(rows) != len(utility_predictions):
        raise ValueError("rows and utility_predictions must have the same length")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(dict(row))].append(index)

    phase_records: dict[str, list[dict]] = defaultdict(list)
    for key, indices in grouped.items():
        group_rows = [rows[index] for index in indices]
        phase = explore_claim_phase(group_rows)
        ranked_indices = sorted(
            indices,
            key=lambda item: float(utility_predictions[item]),
            reverse=True,
        )
        predicted_index = ranked_indices[0]
        oracle_index = max(
            indices,
            key=lambda item: float(_label(rows[item], "target_utility", 0.0)),
        )
        claim_indices = [
            index
            for index in indices
            if bool(_label(rows[index], "goal_claimed", False))
            or bool(_label(rows[index], "reached_goal", False))
        ]
        best_claim_index = (
            max(
                claim_indices,
                key=lambda item: float(_label(rows[item], "target_utility", 0.0)),
            )
            if claim_indices
            else None
        )
        selected_primitive = str(rows[predicted_index]["primitive_sequence"][0])
        oracle_primitive = str(rows[oracle_index]["primitive_sequence"][0])
        selected_sequence = tuple(str(item) for item in rows[predicted_index]["primitive_sequence"])
        oracle_sequence = tuple(str(item) for item in rows[oracle_index]["primitive_sequence"])
        selected_sequence_utility = float(_label(rows[predicted_index], "target_utility", 0.0))
        oracle_utility = float(_label(rows[oracle_index], "target_utility", 0.0))
        selected_prediction = float(utility_predictions[predicted_index])
        oracle_prediction = float(utility_predictions[oracle_index])
        best_claim_prediction = (
            float(utility_predictions[best_claim_index])
            if best_claim_index is not None
            else None
        )
        best_by_first: dict[str, float] = {}
        for index in indices:
            primitive = str(rows[index]["primitive_sequence"][0])
            utility = float(_label(rows[index], "target_utility", 0.0))
            best_by_first[primitive] = max(utility, best_by_first.get(primitive, utility))
        topk_claim_rates = {
            str(k): any(
                bool(_label(rows[index], "goal_claimed", False))
                or bool(_label(rows[index], "reached_goal", False))
                for index in ranked_indices[:k]
            )
            for k in TOP_K_VALUES
        }
        topk_oracle_rates = {
            str(k): oracle_index in set(ranked_indices[:k]) for k in TOP_K_VALUES
        }
        topk_best_claim_rates = {
            str(k): (
                best_claim_index in set(ranked_indices[:k])
                if best_claim_index is not None
                else False
            )
            for k in TOP_K_VALUES
        }
        phase_records[phase].append(
            {
                "source_key": f"{key[0]}:{key[1]}",
                "selected_primitive": selected_primitive,
                "oracle_primitive": oracle_primitive,
                "primitive_match": selected_primitive == oracle_primitive,
                "selected_sequence": selected_sequence,
                "oracle_sequence": oracle_sequence,
                "sequence_match": selected_sequence == oracle_sequence,
                "selected_prediction": selected_prediction,
                "oracle_prediction": oracle_prediction,
                "best_claim_prediction": best_claim_prediction,
                "selected_minus_oracle_prediction": (
                    selected_prediction - oracle_prediction
                ),
                "selected_minus_best_claim_prediction": (
                    selected_prediction - best_claim_prediction
                    if best_claim_prediction is not None
                    else None
                ),
                "selected_prediction_above_oracle": (
                    selected_prediction > oracle_prediction
                ),
                "selected_prediction_above_best_claim": (
                    selected_prediction > best_claim_prediction
                    if best_claim_prediction is not None
                    else None
                ),
                "oracle_rank": ranked_indices.index(oracle_index) + 1,
                "best_claim_rank": (
                    ranked_indices.index(best_claim_index) + 1
                    if best_claim_index is not None
                    else None
                ),
                "best_claim_available": best_claim_index is not None,
                "primitive_regret": oracle_utility - best_by_first[selected_primitive],
                "sequence_regret": oracle_utility - selected_sequence_utility,
                "selected_target_utility": selected_sequence_utility,
                "oracle_target_utility": oracle_utility,
                "selected_new_free_cells": float(
                    _label(rows[predicted_index], "target_new_free_cells", 0.0)
                ),
                "oracle_new_free_cells": float(
                    _label(rows[oracle_index], "target_new_free_cells", 0.0)
                ),
                "selected_future_goal_marker_seen": bool(
                    _label(rows[predicted_index], "future_goal_marker_seen", False)
                ),
                "oracle_future_goal_marker_seen": bool(
                    _label(rows[oracle_index], "future_goal_marker_seen", False)
                ),
                "selected_goal_claimed": bool(
                    _label(rows[predicted_index], "goal_claimed", False)
                    or _label(rows[predicted_index], "reached_goal", False)
                ),
                "oracle_goal_claimed": bool(
                    _label(rows[oracle_index], "goal_claimed", False)
                    or _label(rows[oracle_index], "reached_goal", False)
                ),
                "topk_claimed": topk_claim_rates,
                "topk_oracle_sequence": topk_oracle_rates,
                "topk_best_claim_sequence": topk_best_claim_rates,
            }
        )

    return {
        "schema": "jepa_phase3a_explore_claim_selection_summary_v0",
        "source_states": len(grouped),
        "phases": {
            phase: _summarize_phase_records(phase_records.get(phase, []))
            for phase in EXPLORE_CLAIM_PHASES
        },
    }


def action_sequence_prior_predictions(
    train_rows: Sequence[Mapping],
    validation_rows: Sequence[Mapping],
) -> list[float]:
    """Score validation candidates from train-set mean utility per action sequence."""

    totals: dict[tuple[str, ...], float] = defaultdict(float)
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for row in train_rows:
        sequence = tuple(str(item) for item in row["primitive_sequence"])
        totals[sequence] += float(_label(row, "target_utility", 0.0))
        counts[sequence] += 1
    means = {
        sequence: totals[sequence] / float(counts[sequence])
        for sequence in sorted(totals)
    }
    return [
        means.get(tuple(str(item) for item in row["primitive_sequence"]), 0.0)
        for row in validation_rows
    ]


def egocentric_explore_claim_predictions(rows: Sequence[Mapping]) -> list[float]:
    """Score candidates with non-privileged online exploration memory.

    The score uses only RGB history/current observations plus executed history
    and candidate actions. Before a visual marker has been remembered it acts as
    a frontier/novelty score over egocentric observed cells. Once the marker is
    remembered, it switches to the marker-claiming score.
    """

    return [_egocentric_explore_claim_score(row) for row in rows]


def egocentric_explore_claim_score(row: Mapping) -> float:
    """Return one online frontier-plus-marker score for a candidate row."""

    return _egocentric_explore_claim_score(row)


def _egocentric_explore_claim_score(row: Mapping) -> float:
    if remembered_marker_position(row) is not None:
        return egocentric_marker_memory_score(row)
    seen, blocked = _egocentric_observed_cells(row)
    radius = _view_radius(row)
    if radius < 1:
        return 0.0
    score = 0.0
    for action in primitive_sequence_from_blocks(row.get("active_blocks", [])):
        if action == "forward" and (1, 0) in blocked:
            score -= 2.0
        else:
            seen, blocked = _roll_frontier_cells(seen, blocked, action, radius)
        visible = _view_footprint(radius)
        novel = visible - seen
        score += 0.35 * float(len(novel))
        seen = seen | visible
    return score


def _egocentric_observed_cells(
    row: Mapping,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    history = row.get("history_observations_rgb", [])
    history_actions = primitive_sequence_from_blocks(row.get("history_actions", []))
    radius = _view_radius(row)
    seen: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    if isinstance(history, Sequence):
        for index, observation in enumerate(history):
            item_seen, item_blocked = _observation_cells(
                observation,
                has_beacon=bool(row.get("history_goal_beacon", True)),
            )
            seen.update(item_seen)
            blocked.update(item_blocked)
            if index < len(history_actions):
                seen, blocked = _roll_frontier_cells(
                    seen,
                    blocked,
                    history_actions[index],
                    radius,
                )
    current_seen, current_blocked = _observation_cells(
        row.get("start_observation_rgb"),
        has_beacon=bool(row.get("current_goal_beacon", True)),
    )
    seen.update(current_seen)
    blocked.update(current_blocked)
    return seen, blocked


def _observation_cells(
    observation: object,
    *,
    has_beacon: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if not isinstance(observation, Sequence) or len(observation) != 3:
        return set(), set()
    red, green, blue = observation
    if not (
        isinstance(red, Sequence)
        and isinstance(green, Sequence)
        and isinstance(blue, Sequence)
        and len(red) == len(green) == len(blue)
    ):
        return set(), set()
    radius = len(red) // 2
    seen: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    for row_index in range(len(red)):
        if not (
            isinstance(red[row_index], Sequence)
            and isinstance(green[row_index], Sequence)
            and isinstance(blue[row_index], Sequence)
        ):
            continue
        for col_index in range(len(red[row_index])):
            if has_beacon and row_index == 0 and col_index == 0:
                continue
            ahead = radius - row_index
            lateral = col_index - radius
            cell = (ahead, lateral)
            seen.add(cell)
            r = float(red[row_index][col_index])
            g = float(green[row_index][col_index])
            b = float(blue[row_index][col_index])
            if max(r, g, b) < 0.25:
                blocked.add(cell)
    return seen, blocked


def _roll_cells(cells: set[tuple[int, int]], action: str) -> set[tuple[int, int]]:
    return {advance_marker_egocentric(ahead, lateral, action) for ahead, lateral in cells}


def _roll_frontier_cells(
    seen: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    action: str,
    radius: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if action == "forward" and (1, 0) in blocked:
        return set(seen), set(blocked)
    footprint = _view_footprint(radius)
    return (
        _roll_cells(seen, action) & footprint,
        _roll_cells(blocked, action) & footprint,
    )


def _view_radius(row: Mapping) -> int:
    observation = row.get("start_observation_rgb")
    if isinstance(observation, Sequence) and len(observation) == 3:
        channel = observation[0]
        if isinstance(channel, Sequence):
            return len(channel) // 2
    return 0


def _view_footprint(radius: int) -> set[tuple[int, int]]:
    return {
        (ahead, lateral)
        for ahead in range(-radius, radius + 1)
        for lateral in range(-radius, radius + 1)
    }


def compare_explore_claim_summaries(summaries: Mapping[str, dict]) -> dict:
    """Return compact per-phase deltas for a set of named summaries."""

    comparisons = {}
    if "memory" in summaries and "no_memory" in summaries:
        comparisons["memory_minus_no_memory"] = _phase_deltas(
            summaries["memory"],
            summaries["no_memory"],
        )
    if "memory" in summaries and "action_sequence_prior" in summaries:
        comparisons["memory_minus_action_sequence_prior"] = _phase_deltas(
            summaries["memory"],
            summaries["action_sequence_prior"],
        )
    if "no_memory" in summaries and "action_sequence_prior" in summaries:
        comparisons["no_memory_minus_action_sequence_prior"] = _phase_deltas(
            summaries["no_memory"],
            summaries["action_sequence_prior"],
        )
    if "memory" in summaries and "egocentric_marker_memory" in summaries:
        comparisons["memory_minus_egocentric_marker_memory"] = _phase_deltas(
            summaries["memory"],
            summaries["egocentric_marker_memory"],
        )
    return comparisons


def _summarize_phase_records(records: Sequence[Mapping]) -> dict:
    count = len(records)
    if count == 0:
        return {
            "source_states": 0,
            "primitive_match_rate": None,
            "mean_target_utility_regret": None,
            "mean_selected_sequence_target_utility_regret": None,
            "mean_selected_target_utility": None,
            "mean_oracle_target_utility": None,
            "mean_selected_new_free_cells": None,
            "mean_oracle_new_free_cells": None,
            "selected_future_goal_marker_seen_rate": None,
            "oracle_future_goal_marker_seen_rate": None,
            "selected_goal_claimed_rate": None,
            "oracle_goal_claimed_rate": None,
            "best_claim_available_rate": None,
            "mean_oracle_rank": None,
            "mean_best_claim_rank": None,
            "mean_selected_prediction": None,
            "mean_oracle_prediction": None,
            "mean_best_claim_prediction": None,
            "mean_selected_minus_oracle_prediction": None,
            "mean_selected_minus_best_claim_prediction": None,
            "selected_prediction_above_oracle_rate": None,
            "selected_prediction_above_best_claim_rate": None,
            "sequence_match_rate": None,
            "topk_claimed_rate": {str(k): None for k in TOP_K_VALUES},
            "topk_oracle_sequence_rate": {str(k): None for k in TOP_K_VALUES},
            "topk_best_claim_sequence_rate": {str(k): None for k in TOP_K_VALUES},
            "selected_primitive_counts": {},
            "oracle_primitive_counts": {},
            "selected_sequence_counts": {},
            "oracle_sequence_counts": {},
            "examples": [],
        }
    selected_counts: dict[str, int] = defaultdict(int)
    oracle_counts: dict[str, int] = defaultdict(int)
    selected_sequence_counts: dict[str, int] = defaultdict(int)
    oracle_sequence_counts: dict[str, int] = defaultdict(int)
    for record in records:
        selected_counts[str(record["selected_primitive"])] += 1
        oracle_counts[str(record["oracle_primitive"])] += 1
        selected_sequence_counts[_sequence_key(record["selected_sequence"])] += 1
        oracle_sequence_counts[_sequence_key(record["oracle_sequence"])] += 1

    def mean(name: str) -> float:
        return sum(float(record[name]) for record in records) / float(count)

    def rate(name: str) -> float:
        return sum(float(bool(record[name])) for record in records) / float(count)

    def nullable_mean(name: str) -> float | None:
        values = [
            float(record[name])
            for record in records
            if record.get(name) is not None
        ]
        if not values:
            return None
        return sum(values) / float(len(values))

    def nullable_rate(name: str) -> float | None:
        values = [
            bool(record[name])
            for record in records
            if record.get(name) is not None
        ]
        if not values:
            return None
        return sum(float(value) for value in values) / float(len(values))

    def topk_rate(field: str) -> dict[str, float]:
        return {
            str(k): sum(
                float(bool(record[field][str(k)])) for record in records
            )
            / float(count)
            for k in TOP_K_VALUES
        }

    return {
        "source_states": count,
        "primitive_match_rate": rate("primitive_match"),
        "sequence_match_rate": rate("sequence_match"),
        "mean_target_utility_regret": mean("primitive_regret"),
        "mean_selected_sequence_target_utility_regret": mean("sequence_regret"),
        "mean_selected_target_utility": mean("selected_target_utility"),
        "mean_oracle_target_utility": mean("oracle_target_utility"),
        "mean_selected_new_free_cells": mean("selected_new_free_cells"),
        "mean_oracle_new_free_cells": mean("oracle_new_free_cells"),
        "selected_future_goal_marker_seen_rate": rate(
            "selected_future_goal_marker_seen"
        ),
        "oracle_future_goal_marker_seen_rate": rate("oracle_future_goal_marker_seen"),
        "selected_goal_claimed_rate": rate("selected_goal_claimed"),
        "oracle_goal_claimed_rate": rate("oracle_goal_claimed"),
        "best_claim_available_rate": rate("best_claim_available"),
        "mean_oracle_rank": mean("oracle_rank"),
        "mean_best_claim_rank": nullable_mean("best_claim_rank"),
        "mean_selected_prediction": mean("selected_prediction"),
        "mean_oracle_prediction": mean("oracle_prediction"),
        "mean_best_claim_prediction": nullable_mean("best_claim_prediction"),
        "mean_selected_minus_oracle_prediction": mean(
            "selected_minus_oracle_prediction"
        ),
        "mean_selected_minus_best_claim_prediction": nullable_mean(
            "selected_minus_best_claim_prediction"
        ),
        "selected_prediction_above_oracle_rate": rate(
            "selected_prediction_above_oracle"
        ),
        "selected_prediction_above_best_claim_rate": nullable_rate(
            "selected_prediction_above_best_claim"
        ),
        "topk_claimed_rate": topk_rate("topk_claimed"),
        "topk_oracle_sequence_rate": topk_rate("topk_oracle_sequence"),
        "topk_best_claim_sequence_rate": topk_rate("topk_best_claim_sequence"),
        "selected_primitive_counts": dict(sorted(selected_counts.items())),
        "oracle_primitive_counts": dict(sorted(oracle_counts.items())),
        "selected_sequence_counts": dict(sorted(selected_sequence_counts.items())),
        "oracle_sequence_counts": dict(sorted(oracle_sequence_counts.items())),
        "examples": _diagnostic_examples(records),
    }


def _phase_deltas(left: Mapping, right: Mapping) -> dict:
    deltas = {}
    for phase in EXPLORE_CLAIM_PHASES:
        left_phase = left["phases"][phase]
        right_phase = right["phases"][phase]
        deltas[phase] = {
            "source_states": left_phase["source_states"],
            "primitive_match_rate_delta": _nullable_delta(
                left_phase["primitive_match_rate"],
                right_phase["primitive_match_rate"],
            ),
            "mean_target_utility_regret_delta": _nullable_delta(
                left_phase["mean_target_utility_regret"],
                right_phase["mean_target_utility_regret"],
            ),
            "selected_future_goal_marker_seen_rate_delta": _nullable_delta(
                left_phase["selected_future_goal_marker_seen_rate"],
                right_phase["selected_future_goal_marker_seen_rate"],
            ),
            "selected_goal_claimed_rate_delta": _nullable_delta(
                left_phase["selected_goal_claimed_rate"],
                right_phase["selected_goal_claimed_rate"],
            ),
            "sequence_match_rate_delta": _nullable_delta(
                left_phase["sequence_match_rate"],
                right_phase["sequence_match_rate"],
            ),
            "mean_selected_sequence_regret_delta": _nullable_delta(
                left_phase["mean_selected_sequence_target_utility_regret"],
                right_phase["mean_selected_sequence_target_utility_regret"],
            ),
        }
    return deltas


def _nullable_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _sequence_key(sequence: object) -> str:
    return "|".join(str(item) for item in sequence)


def _diagnostic_examples(records: Sequence[Mapping], *, limit: int = 5) -> list[dict]:
    ranked = sorted(
        records,
        key=lambda record: (
            float(record["sequence_regret"]),
            float(record["oracle_target_utility"]),
        ),
        reverse=True,
    )
    examples = []
    for record in ranked[:limit]:
        examples.append(
            {
                "source_key": record["source_key"],
                "selected_sequence": list(record["selected_sequence"]),
                "oracle_sequence": list(record["oracle_sequence"]),
                "selected_prediction": record["selected_prediction"],
                "oracle_prediction": record["oracle_prediction"],
                "best_claim_prediction": record["best_claim_prediction"],
                "selected_minus_oracle_prediction": (
                    record["selected_minus_oracle_prediction"]
                ),
                "selected_minus_best_claim_prediction": (
                    record["selected_minus_best_claim_prediction"]
                ),
                "oracle_rank": record["oracle_rank"],
                "best_claim_rank": record["best_claim_rank"],
                "selected_target_utility": record["selected_target_utility"],
                "oracle_target_utility": record["oracle_target_utility"],
                "sequence_regret": record["sequence_regret"],
                "selected_goal_claimed": record["selected_goal_claimed"],
                "oracle_goal_claimed": record["oracle_goal_claimed"],
                "topk_claimed": record["topk_claimed"],
                "topk_oracle_sequence": record["topk_oracle_sequence"],
                "topk_best_claim_sequence": record["topk_best_claim_sequence"],
            }
        )
    return examples
