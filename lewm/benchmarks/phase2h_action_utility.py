"""Phase 2H source-local action-utility label and baseline audits."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from .phase2_data import action_name, source_key
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION, action_utility_target


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _primitive_sequence(row: Mapping) -> tuple[str, ...]:
    return tuple(str(value) for value in row.get("primitive_sequence", ()))


def _group_valid_utility_rows(rows: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        utility, valid = action_utility_target(row)
        if not valid:
            continue
        grouped[source_key(row)].append(
            {
                "row": row,
                "row_index": row_index,
                "utility": float(utility),
                "sequence": _primitive_sequence(row),
                "first_primitive": action_name(row, 0),
            }
        )
    return dict(grouped)


def utility_label_audit(rows: Sequence[dict], *, split_name: str) -> dict:
    """Return coverage and source-local spread of the utility target."""

    valid_groups = _group_valid_utility_rows(rows)
    all_source_keys = {source_key(row) for row in rows}
    group_records = []
    for key, records in sorted(valid_groups.items()):
        utilities = [float(record["utility"]) for record in records]
        best = max(utilities)
        top_tie_count = sum(value == best for value in utilities)
        group_records.append(
            {
                "scene_id": key[0],
                "source_index": key[1],
                "candidate_rows": len(records),
                "utility_min": min(utilities),
                "utility_max": best,
                "utility_range": best - min(utilities),
                "utility_mean": sum(utilities) / len(utilities),
                "top_tie_count": top_tie_count,
                "top_tie_fraction": top_tie_count / len(records),
                "oracle_first_primitive": records[utilities.index(best)][
                    "first_primitive"
                ],
                "oracle_sequence": list(records[utilities.index(best)]["sequence"]),
            }
        )
    ranges = [record["utility_range"] for record in group_records]
    candidate_counts = [record["candidate_rows"] for record in group_records]
    tie_fractions = [record["top_tie_fraction"] for record in group_records]
    return {
        "schema": "jepa_phase2h_action_utility_label_audit_v0",
        "split": split_name,
        "target_version": ACTION_UTILITY_TARGET_VERSION,
        "input_rows": len(rows),
        "source_states": len(all_source_keys),
        "valid_utility_rows": sum(len(records) for records in valid_groups.values()),
        "valid_utility_source_states": len(valid_groups),
        "valid_row_fraction": (
            sum(len(records) for records in valid_groups.values()) / len(rows)
            if rows
            else 0.0
        ),
        "valid_source_state_fraction": (
            len(valid_groups) / len(all_source_keys) if all_source_keys else 0.0
        ),
        "mean_valid_candidate_rows_per_source": _mean(
            [float(value) for value in candidate_counts]
        ),
        "minimum_valid_candidate_rows_per_source": min(candidate_counts, default=0),
        "maximum_valid_candidate_rows_per_source": max(candidate_counts, default=0),
        "mean_utility_range_per_source": _mean(ranges),
        "minimum_utility_range_per_source": min(ranges, default=None),
        "maximum_utility_range_per_source": max(ranges, default=None),
        "mean_top_tie_fraction": _mean(tie_fractions),
        "source_records": group_records,
    }


def fit_action_utility_priors(rows: Sequence[dict]) -> dict:
    """Fit source-independent mean utility priors from training rows."""

    by_sequence: dict[tuple[str, ...], list[float]] = defaultdict(list)
    by_first: dict[str, list[float]] = defaultdict(list)
    global_values = []
    for row in rows:
        utility, valid = action_utility_target(row)
        if not valid:
            continue
        sequence = _primitive_sequence(row)
        first = action_name(row, 0)
        by_sequence[sequence].append(float(utility))
        by_first[first].append(float(utility))
        global_values.append(float(utility))
    if not global_values:
        raise ValueError("cannot fit action-utility priors without valid targets")
    sequence_means = {
        " / ".join(key): {
            "mean_utility": sum(values) / len(values),
            "count": len(values),
        }
        for key, values in sorted(by_sequence.items())
    }
    first_means = {
        key: {
            "mean_utility": sum(values) / len(values),
            "count": len(values),
        }
        for key, values in sorted(by_first.items())
    }
    return {
        "schema": "jepa_phase2h_action_utility_priors_v0",
        "target_version": ACTION_UTILITY_TARGET_VERSION,
        "valid_training_rows": len(global_values),
        "global_mean_utility": sum(global_values) / len(global_values),
        "sequence_mean_utility": sequence_means,
        "first_primitive_mean_utility": first_means,
    }


def _prior_score(record: dict, priors: Mapping, *, baseline: str) -> tuple[float, bool]:
    global_mean = float(priors["global_mean_utility"])
    if baseline == "full_sequence_mean":
        key = " / ".join(record["sequence"])
        entry = priors["sequence_mean_utility"].get(key)
    elif baseline == "first_primitive_mean":
        entry = priors["first_primitive_mean_utility"].get(
            record["first_primitive"]
        )
    else:
        raise ValueError(f"unsupported action-only baseline: {baseline}")
    if entry is None:
        return global_mean, False
    return float(entry["mean_utility"]), True


def evaluate_action_only_baseline(
    rows: Sequence[dict],
    priors: Mapping,
    *,
    split_name: str,
    baseline: str,
) -> dict:
    """Evaluate a source-independent action prior on source-local selection."""

    grouped = _group_valid_utility_rows(rows)
    selection_records = []
    for key, records in sorted(grouped.items()):
        if len(records) < 2:
            continue
        targets = [float(record["utility"]) for record in records]
        scores_and_seen = [
            _prior_score(record, priors, baseline=baseline) for record in records
        ]
        scores = [score for score, _seen in scores_and_seen]
        selected_index = max(range(len(records)), key=lambda index: scores[index])
        oracle_index = max(range(len(records)), key=lambda index: targets[index])
        selected = records[selected_index]
        oracle = records[oracle_index]
        selected_target = targets[selected_index]
        oracle_target = targets[oracle_index]
        selection_records.append(
            {
                "schema": "jepa_phase2h_action_only_selection_record_v0",
                "split": split_name,
                "baseline": baseline,
                "scene_id": key[0],
                "source_index": key[1],
                "candidate_rows": len(records),
                "selected_row_index": int(selected["row_index"]),
                "oracle_row_index": int(oracle["row_index"]),
                "selected_sequence": list(selected["sequence"]),
                "oracle_sequence": list(oracle["sequence"]),
                "selected_first_primitive": selected["first_primitive"],
                "oracle_first_primitive": oracle["first_primitive"],
                "selected_predicted_utility": float(scores[selected_index]),
                "oracle_predicted_utility": float(scores[oracle_index]),
                "selected_target_utility": selected_target,
                "oracle_target_utility": oracle_target,
                "target_utility_regret": oracle_target - selected_target,
                "top1_match": selected_index == oracle_index,
                "first_primitive_match": (
                    selected["first_primitive"] == oracle["first_primitive"]
                ),
                "selected_prior_seen_in_train": bool(
                    scores_and_seen[selected_index][1]
                ),
                "oracle_prior_seen_in_train": bool(scores_and_seen[oracle_index][1]),
                "uniform_random_top1_rate": 1.0 / len(records),
                "uniform_random_first_primitive_match_rate": (
                    sum(
                        record["first_primitive"] == oracle["first_primitive"]
                        for record in records
                    )
                    / len(records)
                ),
            }
        )
    if not selection_records:
        return {
            "schema": "jepa_phase2h_action_only_selection_summary_v0",
            "split": split_name,
            "baseline": baseline,
            "source_state_count": 0,
            "selection_records": [],
        }

    def mean_key(key: str) -> float:
        return sum(float(record[key]) for record in selection_records) / len(
            selection_records
        )

    return {
        "schema": "jepa_phase2h_action_only_selection_summary_v0",
        "split": split_name,
        "baseline": baseline,
        "target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_state_count": len(selection_records),
        "mean_candidate_rows": mean_key("candidate_rows"),
        "top1_match_rate": mean_key("top1_match"),
        "first_primitive_match_rate": mean_key("first_primitive_match"),
        "mean_target_utility_regret": mean_key("target_utility_regret"),
        "mean_selected_target_utility": mean_key("selected_target_utility"),
        "mean_oracle_target_utility": mean_key("oracle_target_utility"),
        "selected_prior_seen_rate": mean_key("selected_prior_seen_in_train"),
        "oracle_prior_seen_rate": mean_key("oracle_prior_seen_in_train"),
        "uniform_random_expected_top1_rate": mean_key("uniform_random_top1_rate"),
        "uniform_random_expected_first_primitive_match_rate": mean_key(
            "uniform_random_first_primitive_match_rate"
        ),
        "selection_records": selection_records,
    }


def phase2h_action_utility_audit(
    *,
    train_rows: Sequence[dict],
    validation_rows: Sequence[dict],
) -> dict:
    """Build the Phase 2H train/validation utility-label audit."""

    priors = fit_action_utility_priors(train_rows)
    baselines = [
        evaluate_action_only_baseline(
            validation_rows,
            priors,
            split_name="validation",
            baseline=baseline,
        )
        for baseline in ("full_sequence_mean", "first_primitive_mean")
    ]
    return {
        "schema": "jepa_phase2h_action_utility_audit_v0",
        "target_version": ACTION_UTILITY_TARGET_VERSION,
        "train_label_audit": utility_label_audit(train_rows, split_name="train"),
        "validation_label_audit": utility_label_audit(
            validation_rows,
            split_name="validation",
        ),
        "train_action_priors": priors,
        "validation_action_only_baselines": baselines,
        "decision_rule": (
            "If action-only baselines meet or exceed the Phase 2G utility "
            "selection threshold, audit target confounding before training a "
            "source-conditioned model. Otherwise proceed to a source-conditioned "
            "affordance/utility state pilot."
        ),
        "limitations": [
            "train and validation evidence only",
            "test_id and test_hard remain unopened",
            "source-independent priors test label/action bias, not deployable policy quality",
        ],
    }
