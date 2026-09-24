#!/usr/bin/env python3
"""One-shot, evidence-only attentive metric reconciliation.

The scientific path is deliberately small.  It verifies the seven immutable
artifacts, inventories the already-written JSON evidence, and stops before
metric reduction when the evidence is incomplete.  The two reducers below are
pure NumPy/Python functions exercised only by synthetic focused tests unless
every frozen recoverability gate is met.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.oracle import go2_attentive_metric_reconciliation_v1_contract as C


ATTEMPT_SCHEMA = "go2_attentive_metric_reconciliation_v1_attempt_v1"
ATTEMPT_SELF_KEY = "attentive_metric_reconciliation_attempt_digest"
TERMINAL_SCHEMA = "go2_attentive_metric_reconciliation_v1_terminal_v1"
TERMINAL_SELF_KEY = "attentive_metric_reconciliation_terminal_digest"
RESULT_SCHEMA = "go2_attentive_metric_reconciliation_v1_repaired_result_v1"
RESULT_SELF_KEY = "attentive_metric_reconciliation_repaired_result_digest"


class MetricReconciliationError(RuntimeError):
    """The one-shot evidence or reconciliation result changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MetricReconciliationError(message)


def _signed(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    result[key] = C.digest(payload)
    return result


def _publish_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(C.canonical_bytes(value) + b"\n")
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o444)


def _read_signed(path: Path, key: str, label: str) -> dict[str, Any]:
    return C.validate_signed(C.read_json(path, label), key, label)


def _finite(number: Any) -> bool:
    return (isinstance(number, (int, float)) and not isinstance(number, bool)
            and math.isfinite(float(number)))


def _json_number(number: float) -> float | None:
    return float(number) if math.isfinite(float(number)) else None


def _probability(prediction: Mapping[str, Any], key: str) -> float:
    probability = prediction.get(key)
    logit = prediction.get(f"{key}_logit")
    require(probability is not None or logit is not None,
            f"{key} probability or logit is absent")
    if probability is None:
        value = float(logit)
        probability = (1.0 / (1.0 + math.exp(-value)) if value >= 0.0
                       else math.exp(value) / (1.0 + math.exp(value)))
    probability = float(probability)
    require(math.isfinite(probability) and 0.0 <= probability <= 1.0,
            f"{key} probability is invalid")
    if logit is not None:
        value = float(logit)
        derived = (1.0 / (1.0 + math.exp(-value)) if value >= 0.0
                   else math.exp(value) / (1.0 + math.exp(value)))
        require(math.isclose(probability, derived, rel_tol=1e-12,
                             abs_tol=1e-12),
                f"{key} logit and probability disagree")
    return probability


def _normalise_rows(rows: Sequence[Mapping[str, Any]], *,
                    project_component_targets_to_float32: bool
                    ) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        target = source.get("target")
        prediction = source.get("prediction")
        require(isinstance(target, Mapping) and isinstance(prediction, Mapping),
                f"row {index} target or prediction is absent")
        target_values = {}
        for key in ("progress", "safety", "completion"):
            value = float(target[key])
            if project_component_targets_to_float32:
                value = float(np.float32(value))
            require(math.isfinite(value), f"row {index} target is non-finite")
            target_values[key] = value
        target_values["utility"] = float(target["utility"])
        prediction_values = {
            "progress": float(prediction["progress"]),
            "safety": _probability(prediction, "safety"),
            "completion": _probability(prediction, "completion"),
            "utility": float(prediction["utility"]),
        }
        require(all(math.isfinite(value) for value in target_values.values())
                and all(math.isfinite(value)
                        for value in prediction_values.values()),
                f"row {index} contains a non-finite scalar")
        result.append({
            "training_view_row_digest": source.get(
                "training_view_row_digest", f"fixture-row-{index}"),
            "branch_identity_digest": source.get(
                "branch_identity_digest", f"fixture-branch-{index}"),
            "state_id": str(source["state_id"]),
            "family": str(source["family"]),
            "stratum": str(source.get("stratum", "fixture")),
            "candidate_index": int(source["candidate_index"]),
            "target": target_values,
            "prediction": prediction_values,
        })
    return result


# ------------------------------------------------------ Consumer A: frozen --
def _a_average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _a_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(left) != len(right):
        return float("nan")
    x = _a_average_ranks(np.asarray(left, dtype=np.float64))
    y = _a_average_ranks(np.asarray(right, dtype=np.float64))
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def _a_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = np.asarray(labels, dtype=np.float64) > 0.5
    count_positive = int(positive.sum())
    count_negative = int((~positive).sum())
    if not count_positive or not count_negative:
        return float("nan")
    ranks = _a_average_ranks(np.asarray(scores, dtype=np.float64)) + 1.0
    return float((ranks[positive].sum()
                  - count_positive * (count_positive + 1) / 2)
                 / (count_positive * count_negative))


def _a_ece(target: np.ndarray, predicted: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    weighted = 0.0
    total = 0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = ((predicted >= lower)
                & (predicted < upper if upper < 1.0 else predicted <= upper))
        count = int(mask.sum())
        if count:
            weighted += count * abs(float(predicted[mask].mean())
                                    - float(target[mask].mean()))
            total += count
    return weighted / total if total else float("nan")


def _state_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["state_id"]), []).append(index)
    return groups


def _a_composite(rows: Sequence[Mapping[str, Any]], truth: np.ndarray,
                 scores: np.ndarray) -> dict[str, Any]:
    absolute: list[float] = []
    normalised: list[float] = []
    selected: list[float] = []
    ranks: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    top_ties: list[float] = []
    spreads: list[float] = []
    correct = considered = score_ties = all_pairs = 0
    groups = _state_groups(rows)
    for indices in groups.values():
        actual = truth[indices]
        predicted = scores[indices]
        order = np.argsort(-predicted, kind="mergesort")
        chosen = int(order[0])
        best = actual == actual.max()
        regret = float(actual.max() - actual[chosen])
        spread = float(actual.max() - actual.min())
        absolute.append(regret)
        normalised.append(0.0 if spread <= 0 else regret / spread)
        selected.append(float(actual[chosen]))
        top1.append(float(best[chosen]))
        top3.append(float(np.any(best[order[:3]])))
        top_ties.append(float(np.sum(np.abs(
            predicted - predicted.max()) <= C.TIE_TOLERANCE) > 1))
        spreads.append(float(predicted.max() - predicted.min()))
        rank = _a_spearman(actual, predicted)
        if math.isfinite(rank):
            ranks.append(rank)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                gap = float(actual[left] - actual[right])
                predicted_gap = float(predicted[left] - predicted[right])
                all_pairs += 1
                score_ties += int(abs(predicted_gap) <= C.TIE_TOLERANCE)
                if abs(gap) <= C.TIE_TOLERANCE:
                    continue
                considered += 1
                correct += int(predicted_gap * gap > 0.0)
    return {
        "absolute_rank_regret": float(np.mean(absolute)) if absolute else None,
        "normalised_rank_regret": float(np.mean(normalised)) if normalised else None,
        "realised_selected_utility": float(np.mean(selected)) if selected else None,
        "pairwise_ordering_accuracy": correct / considered if considered else None,
        "pairs_considered": considered,
        "ranking_spearman": float(np.mean(ranks)) if ranks else None,
        "top1_recovery": float(np.mean(top1)) if top1 else None,
        "top3_recovery": float(np.mean(top3)) if top3 else None,
        "top_score_tie_rate": float(np.mean(top_ties)) if top_ties else None,
        "all_pair_tie_rate": score_ties / all_pairs if all_pairs else None,
        "candidate_score_spread": {
            "mean": float(np.mean(spreads)) if spreads else None,
            "median": float(np.median(spreads)) if spreads else None,
            "min": float(np.min(spreads)) if spreads else None,
            "max": float(np.max(spreads)) if spreads else None,
        },
        "states": len(groups),
    }


def _consumer_a_core(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = {key: np.asarray([row["target"][key] for row in rows],
                              dtype=np.float64)
              for key in ("progress", "safety", "completion", "utility")}
    predicted = {key: np.asarray([row["prediction"][key] for row in rows],
                                 dtype=np.float64)
                 for key in ("progress", "safety", "completion", "utility")}
    result = {
        "rows": len(rows),
        "progress": {
            "mae": float(np.mean(np.abs(target["progress"] - predicted["progress"]))),
            "rmse": float(np.sqrt(np.mean((target["progress"]
                                             - predicted["progress"]) ** 2))),
            "spearman": _json_number(_a_spearman(
                target["progress"], predicted["progress"])),
        },
        "safety": {
            "mae": float(np.mean(np.abs(target["safety"] - predicted["safety"]))),
            "rmse": float(np.sqrt(np.mean((target["safety"]
                                             - predicted["safety"]) ** 2))),
            "auc": _json_number(_a_auc(
                (target["safety"] > 0.0).astype(np.float64), predicted["safety"])),
            "ece": _json_number(_a_ece(target["safety"], predicted["safety"])),
        },
        "completion": {
            "prevalence": float(np.mean(target["completion"])),
            "mae": float(np.mean(np.abs(target["completion"]
                                          - predicted["completion"]))),
            "brier": float(np.mean((target["completion"]
                                      - predicted["completion"]) ** 2)),
            "auc": _json_number(_a_auc(target["completion"],
                                         predicted["completion"])),
            "ece": _json_number(_a_ece(target["completion"],
                                         predicted["completion"])),
        },
        "composite": _a_composite(rows, target["utility"],
                                    predicted["utility"]),
    }
    return result


def consumer_a(rows: Sequence[Mapping[str, Any]], *,
               project_component_targets_to_float32: bool = True
               ) -> dict[str, Any]:
    normalised = _normalise_rows(
        rows, project_component_targets_to_float32=
        project_component_targets_to_float32)
    result = _consumer_a_core(normalised)
    families = sorted({row["family"] for row in normalised})
    result["per_family"] = {
        family: _consumer_a_core([
            row for row in normalised if row["family"] == family])
        for family in families
    }
    return result


# -------------------------------------------- Consumer B: independent f64 --
def _b_mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _b_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0)


def _b_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for position in range(start, stop):
            result[ordered[position]] = rank
        start = stop
    return result


def _b_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x, y = _b_ranks(left), _b_ranks(right)
    x_mean, y_mean = _b_mean(x), _b_mean(y)
    x = [value - x_mean for value in x]
    y = [value - y_mean for value in y]
    denominator = math.sqrt(math.fsum(value * value for value in x)
                            * math.fsum(value * value for value in y))
    return (math.fsum(a * b for a, b in zip(x, y, strict=True)) / denominator
            if denominator > 0.0 else None)


def _b_auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    positive = [value > 0.5 for value in labels]
    count_positive = sum(positive)
    count_negative = len(labels) - count_positive
    if not count_positive or not count_negative:
        return None
    ranks = [value + 1.0 for value in _b_ranks(scores)]
    rank_sum = math.fsum(rank for rank, is_positive in zip(
        ranks, positive, strict=True) if is_positive)
    return ((rank_sum - count_positive * (count_positive + 1) / 2.0)
            / (count_positive * count_negative))


def _b_ece(target: Sequence[float], predicted: Sequence[float]) -> float | None:
    weighted = 0.0
    total = 0
    for bin_index in range(10):
        lower, upper = bin_index / 10.0, (bin_index + 1) / 10.0
        indices = [index for index, value in enumerate(predicted)
                   if value >= lower and (value < upper or
                                           (bin_index == 9 and value <= upper))]
        if indices:
            weighted += len(indices) * abs(
                _b_mean([predicted[index] for index in indices])
                - _b_mean([target[index] for index in indices]))
            total += len(indices)
    return weighted / total if total else None


def _b_composite(rows: Sequence[Mapping[str, Any]], truth: Sequence[float],
                 scores: Sequence[float]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["state_id"])].append(index)
    absolute: list[float] = []
    normalised: list[float] = []
    selected: list[float] = []
    ranks: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    top_ties: list[float] = []
    spreads: list[float] = []
    correct = considered = score_ties = all_pairs = 0
    for indices in groups.values():
        actual = [truth[index] for index in indices]
        predicted = [scores[index] for index in indices]
        order = sorted(range(len(indices)), key=lambda index: (-predicted[index], index))
        chosen = order[0]
        maximum, minimum = max(actual), min(actual)
        regret = maximum - actual[chosen]
        absolute.append(regret)
        normalised.append(0.0 if maximum <= minimum
                          else regret / (maximum - minimum))
        selected.append(actual[chosen])
        best = {index for index, value in enumerate(actual) if value == maximum}
        top1.append(float(chosen in best))
        top3.append(float(any(index in best for index in order[:3])))
        top_ties.append(float(sum(
            abs(value - max(predicted)) <= C.TIE_TOLERANCE
            for value in predicted) > 1))
        spreads.append(max(predicted) - min(predicted))
        rank = _b_spearman(actual, predicted)
        if rank is not None:
            ranks.append(rank)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                actual_gap = actual[left] - actual[right]
                score_gap = predicted[left] - predicted[right]
                all_pairs += 1
                score_ties += int(abs(score_gap) <= C.TIE_TOLERANCE)
                if abs(actual_gap) <= C.TIE_TOLERANCE:
                    continue
                considered += 1
                correct += int(actual_gap * score_gap > 0.0)
    return {
        "absolute_rank_regret": _b_mean(absolute) if absolute else None,
        "normalised_rank_regret": _b_mean(normalised) if normalised else None,
        "realised_selected_utility": _b_mean(selected) if selected else None,
        "pairwise_ordering_accuracy": correct / considered if considered else None,
        "pairs_considered": considered,
        "ranking_spearman": _b_mean(ranks) if ranks else None,
        "top1_recovery": _b_mean(top1) if top1 else None,
        "top3_recovery": _b_mean(top3) if top3 else None,
        "top_score_tie_rate": _b_mean(top_ties) if top_ties else None,
        "all_pair_tie_rate": score_ties / all_pairs if all_pairs else None,
        "candidate_score_spread": {
            "mean": _b_mean(spreads) if spreads else None,
            "median": _b_median(spreads) if spreads else None,
            "min": min(spreads) if spreads else None,
            "max": max(spreads) if spreads else None,
        },
        "states": len(groups),
    }


def _consumer_b_core(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = {key: [float(row["target"][key]) for row in rows]
              for key in ("progress", "safety", "completion", "utility")}
    predicted = {key: [float(row["prediction"][key]) for row in rows]
                 for key in ("progress", "safety", "completion", "utility")}
    errors = {key: [predicted[key][index] - target[key][index]
                    for index in range(len(rows))]
              for key in ("progress", "safety", "completion")}
    return {
        "rows": len(rows),
        "progress": {
            "mae": _b_mean([abs(value) for value in errors["progress"]]),
            "rmse": math.sqrt(_b_mean([
                value * value for value in errors["progress"]])),
            "spearman": _b_spearman(target["progress"], predicted["progress"]),
        },
        "safety": {
            "mae": _b_mean([abs(value) for value in errors["safety"]]),
            "rmse": math.sqrt(_b_mean([
                value * value for value in errors["safety"]])),
            "auc": _b_auc([float(value > 0.0) for value in target["safety"]],
                           predicted["safety"]),
            "ece": _b_ece(target["safety"], predicted["safety"]),
        },
        "completion": {
            "prevalence": _b_mean(target["completion"]),
            "mae": _b_mean([abs(value) for value in errors["completion"]]),
            "brier": _b_mean([value * value
                               for value in errors["completion"]]),
            "auc": _b_auc(target["completion"], predicted["completion"]),
            "ece": _b_ece(target["completion"], predicted["completion"]),
        },
        "composite": _b_composite(rows, target["utility"],
                                    predicted["utility"]),
    }


def consumer_b(rows: Sequence[Mapping[str, Any]], *,
               project_component_targets_to_float32: bool = True
               ) -> dict[str, Any]:
    normalised = _normalise_rows(
        rows, project_component_targets_to_float32=
        project_component_targets_to_float32)
    result = _consumer_b_core(normalised)
    result["per_family"] = {
        family: _consumer_b_core([
            row for row in normalised if row["family"] == family])
        for family in sorted({row["family"] for row in normalised})
    }
    return result


def compare_consumers(left: Any, right: Any, path: str = "") -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0

    def visit(a: Any, b: Any, current: str) -> None:
        nonlocal maximum_absolute, maximum_relative
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                differences.append({"path": current, "kind": "key_set"})
                return
            for key in sorted(a):
                visit(a[key], b[key], f"{current}.{key}" if current else str(key))
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({"path": current, "kind": "length"})
                return
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, f"{current}[{index}]")
            return
        if (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(b, (int, float)) and not isinstance(b, bool)):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    differences.append({"path": current, "kind": "discrete",
                                        "left": a, "right": b})
                return
            difference = abs(float(a) - float(b))
            relative = difference / max(abs(float(a)), abs(float(b)), 1e-300)
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not math.isclose(float(a), float(b),
                                abs_tol=C.TOLERANCES["absolute"],
                                rel_tol=C.TOLERANCES["relative"]):
                differences.append({"path": current, "kind": "float",
                                    "left": float(a), "right": float(b),
                                    "absolute_difference": difference,
                                    "relative_difference": relative})
            return
        if a != b:
            differences.append({"path": current, "kind": "discrete",
                                "left": a, "right": b})

    visit(left, right, path)
    return {
        "discrete_identities_and_counts_exact": not any(
            row["kind"] != "float" for row in differences),
        "float_outputs_within_tolerance": not any(
            row["kind"] == "float" for row in differences),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "differences": differences,
        "passed": not differences,
    }


def align_baseline_predictions(
        attentive_rows: Sequence[Mapping[str, Any]],
        baseline_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    identity = lambda row: (str(row["state_id"]), int(row["candidate_index"]),
                            str(row["training_view_row_digest"]))
    baseline = {identity(row): row for row in baseline_rows}
    require(len(baseline) == len(baseline_rows),
            "baseline predictions contain duplicate row identities")
    expected = [identity(row) for row in attentive_rows]
    require(set(expected) == set(baseline),
            "baseline prediction identity set changed")
    return [baseline[key] for key in expected]


# ---------------------------------------------------------- evidence audit --
def inspect_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    value = C.validate_signed(evidence, "calibration_evidence_digest",
                              "calibration evidence")
    rows = value.get("rows")
    require(set(value) == {
        "schema", "status", "complete", "execution_bindings",
        "evaluation_authorisation_digest", "final_checkpoint_sha256",
        "final_state_digest", "row_count", "training_view_row_order_digest",
        "training_view_row_identity_set_digest", "branch_identity_set_digest",
        "rows", "calibration_evaluation_session_count",
        "model_forward_batch_count", "raw_latent_persisted",
        "predictor_material_accessed", "calibration_evidence_digest",
    }, "closed evidence top-level schema changed")
    require(value.get("schema") == C.FROZEN_EVIDENCE["schema"]
            and value.get("status") == C.FROZEN_EVIDENCE["status"]
            and value.get("complete") is True
            and value.get("execution_bindings")
            == C.FROZEN_EVIDENCE_EXECUTION_BINDINGS
            and value.get("calibration_evaluation_session_count") == 1
            and value.get("raw_latent_persisted") is False
            and value.get("predictor_material_accessed") is False,
            "closed evidence execution flags changed")
    require(value.get("row_count") == C.FROZEN_EVIDENCE["rows"]
            and isinstance(rows, list)
            and len(rows) == C.FROZEN_EVIDENCE["rows"],
            "closed evidence row count changed")
    require(value.get("training_view_row_order_digest")
            == C.digest([row.get("training_view_row_digest") for row in rows])
            == C.FROZEN_EVIDENCE["training_view_row_order_digest"],
            "closed evidence row order changed")
    require(value.get("training_view_row_identity_set_digest")
            == C.digest(sorted(row.get("training_view_row_digest") for row in rows))
            == C.FROZEN_EVIDENCE["training_view_row_identity_set_digest"],
            "closed evidence row identity set changed")
    require(value.get("branch_identity_set_digest")
            == C.digest(sorted(row.get("branch_identity_digest") for row in rows))
            == C.FROZEN_EVIDENCE["branch_identity_set_digest"],
            "closed evidence branch identity set changed")
    require(value.get("model_forward_batch_count")
            == C.FROZEN_EVIDENCE["model_forward_batch_count"]
            and value.get("evaluation_authorisation_digest")
            == C.FROZEN_EVIDENCE["evaluation_authorisation_digest"]
            and value.get("final_checkpoint_sha256")
            == C.FROZEN_EVIDENCE["final_checkpoint_sha256"]
            and value.get("final_state_digest")
            == C.FROZEN_EVIDENCE["final_state_digest"],
            "closed evidence execution binding changed")

    missing = defaultdict(int)
    projected_progress = projected_safety = 0
    maximum_progress = maximum_safety = 0.0
    float32_predictions = 0
    families: dict[str, set[str]] = defaultdict(set)
    family_rows: dict[str, int] = defaultdict(int)
    state_candidates: dict[str, set[int]] = defaultdict(set)
    state_families: dict[str, set[str]] = defaultdict(set)
    state_strata: dict[str, set[str]] = defaultdict(set)
    training_digests: set[str] = set()
    branch_digests: set[str] = set()
    state_candidate_pairs: set[tuple[str, int]] = set()
    schemas_exact = targets_finite = predictions_finite = True
    probabilities_valid = completion_binary = True
    first_projection: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping)
                and isinstance(row.get("target"), Mapping)
                and isinstance(row.get("prediction"), Mapping),
                f"evidence row {index} is malformed")
        schemas_exact &= (set(row) == {
            "training_view_row_digest", "branch_identity_digest", "state_id",
            "family", "stratum", "candidate_index", "target", "prediction"}
            and set(row["target"]) == {
                "progress", "safety", "completion", "utility"}
            and set(row["prediction"]) == {
                "progress", "safety", "completion", "utility"})
        for key in ("action_blocks", "goal_binding_input", "split_role",
                    "scene_id", "state_identity_digest"):
            if key not in row:
                missing[key] += 1
        if not isinstance(row.get("no_latent_prediction"), Mapping):
            missing["no_latent_prediction"] += 1
        row_targets_finite = all(_finite(row["target"].get(key)) for key in (
            "progress", "safety", "completion", "utility"))
        row_predictions_finite = all(_finite(row["prediction"].get(key)) for key in (
            "progress", "safety", "completion", "utility"))
        targets_finite &= row_targets_finite
        predictions_finite &= row_predictions_finite
        require(row_targets_finite and row_predictions_finite,
                f"evidence row {index} contains a non-finite scalar")
        probabilities_valid &= all(
            0.0 <= float(row["prediction"][key]) <= 1.0
            for key in ("safety", "completion"))
        completion_binary &= float(row["target"]["completion"]) in (0.0, 1.0)
        for key in ("progress", "safety", "completion", "utility"):
            prediction = float(row["prediction"][key])
            float32_predictions += int(float(np.float32(prediction)) == prediction)
        progress = float(row["target"]["progress"])
        safety = float(row["target"]["safety"])
        progress_delta = abs(progress - float(np.float32(progress)))
        safety_delta = abs(safety - float(np.float32(safety)))
        projected_progress += int(progress_delta > 0.0)
        projected_safety += int(safety_delta > 0.0)
        maximum_progress = max(maximum_progress, progress_delta)
        maximum_safety = max(maximum_safety, safety_delta)
        if first_projection is None and (progress_delta or safety_delta):
            first_projection = {
                "row_index": index,
                "training_view_row_digest": row["training_view_row_digest"],
                "branch_identity_digest": row["branch_identity_digest"],
                "state_id": row["state_id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "candidate_index": row["candidate_index"],
                "stored_progress": progress,
                "direct_float32_progress": float(np.float32(progress)),
                "progress_prediction": float(row["prediction"]["progress"]),
                "replay_progress_absolute_error": abs(
                    float(row["prediction"]["progress"]) - progress),
                "direct_progress_absolute_error": abs(
                    float(row["prediction"]["progress"])
                    - float(np.float32(progress))),
                "stored_safety": safety,
                "direct_float32_safety": float(np.float32(safety)),
            }
        family = str(row["family"])
        state_id = str(row["state_id"])
        candidate = int(row["candidate_index"])
        training_digests.add(str(row["training_view_row_digest"]))
        branch_digests.add(str(row["branch_identity_digest"]))
        state_candidate_pairs.add((state_id, candidate))
        state_candidates[state_id].add(candidate)
        state_families[state_id].add(family)
        state_strata[state_id].add(str(row["stratum"]))
        families[family].add(state_id)
        family_rows[family] += 1
    inventory = {
        "unique_training_view_row_digests": len(training_digests),
        "unique_branch_identity_digests": len(branch_digests),
        "unique_states": len(state_candidates),
        "unique_state_candidate_pairs": len(state_candidate_pairs),
        "states_with_exact_candidates_0_through_11": sum(
            candidates == set(range(12)) for candidates in state_candidates.values()),
        "states_with_single_observed_family": sum(
            len(values) == 1 for values in state_families.values()),
        "states_with_single_observed_stratum": sum(
            len(values) == 1 for values in state_strata.values()),
        "row_target_prediction_key_schemas_exact": schemas_exact,
        "all_targets_finite": targets_finite,
        "all_predictions_finite": predictions_finite,
        "all_safety_and_completion_predictions_are_probabilities":
            probabilities_valid,
        "all_completion_targets_binary": completion_binary,
        "rows_missing_action_blocks": missing["action_blocks"],
        "rows_missing_goal_binding_input": missing["goal_binding_input"],
        "rows_missing_no_latent_prediction": missing["no_latent_prediction"],
        "rows_missing_split_role": missing["split_role"],
        "rows_missing_scene_id": missing["scene_id"],
        "rows_missing_state_identity_digest": missing["state_identity_digest"],
        "rows_missing_safety_logit": sum(
            "safety_logit" not in row["prediction"] for row in rows),
        "rows_missing_completion_logit": sum(
            "completion_logit" not in row["prediction"] for row in rows),
        "rows_with_attentive_safety_probability": sum(
            "safety" in row["prediction"] for row in rows),
        "rows_with_attentive_completion_probability": sum(
            "completion" in row["prediction"] for row in rows),
        "attentive_prediction_scalars_exactly_float32_representable":
            float32_predictions,
        "observed_family_count": len(families),
        "observed_states_per_family": (
            next(iter({len(states) for states in families.values()}))
            if len({len(states) for states in families.values()}) == 1 else None),
        "observed_rows_per_family": (
            next(iter(set(family_rows.values())))
            if len(set(family_rows.values())) == 1 else None),
        "frozen_calibration_manifest_or_state_family_mapping_retained": False,
        "direct_metric_table_retained": False,
        "direct_online_accumulators_retained": False,
        "direct_online_dtype_precision_declaration_retained": False,
        "progress_targets_changed_by_online_float32_projection":
            projected_progress,
        "safety_targets_changed_by_online_float32_projection": projected_safety,
        "maximum_progress_target_projection_delta": maximum_progress,
        "maximum_safety_target_projection_delta": maximum_safety,
    }
    require(inventory == C.EXPECTED_EVIDENCE_INVENTORY,
            "closed evidence completeness inventory changed")
    witness = C.SOURCE_RECONSTRUCTION["first_divergent_row"]
    require(first_projection is not None
            and all(first_projection.get(key) == value for key, value in {
                "row_index": witness["row_index"],
                "training_view_row_digest": witness["training_view_row_digest"],
                "branch_identity_digest": witness["branch_identity_digest"],
                "state_id": witness["state_id"], "family": witness["family"],
                "stratum": witness["stratum"],
                "candidate_index": witness["candidate_index"],
                "stored_progress": witness["stored_target"],
                "direct_float32_progress": witness[
                    "direct_float32_projected_target"],
                "progress_prediction": witness["prediction"],
                "replay_progress_absolute_error": witness[
                    "replay_absolute_error"],
                "direct_progress_absolute_error": witness[
                    "direct_absolute_error"],
            }.items()), "first component-target divergence changed")
    return {
        "inventory": inventory,
        "observed_families": {
            family: {"states": len(families[family]),
                     "rows": family_rows[family]}
            for family in sorted(families)},
        "family_assignment_matches_frozen_manifest": None,
        "family_assignment_manifest_verdict": (
            "UNVERIFIABLE_FROM_THE_SEVEN_AUTHORISED_ARTIFACTS"),
        "prediction_representation": {
            "progress": "RAW_FLOAT32_MODEL_SCALAR_PERSISTED_AS_JSON_NUMBER",
            "safety": "POST_SIGMOID_PROBABILITY_NO_LOGIT_RETAINED",
            "completion": "POST_SIGMOID_PROBABILITY_NO_LOGIT_RETAINED",
            "utility": "COMPOSITE_FLOAT32_MODEL_SCALAR_PERSISTED_AS_JSON_NUMBER",
            "source_defined_dtype_path_is_unambiguous": True,
            "receipt_contains_explicit_dtype_declaration": False,
        },
        "first_component_target_projection_difference": first_projection,
        "source_reconstruction": dict(C.SOURCE_RECONSTRUCTION),
        "exact_24_by_12_structure": (
            inventory["unique_states"] == 24
            and inventory["unique_state_candidate_pairs"] == 288
            and inventory["states_with_exact_candidates_0_through_11"] == 24),
    }


def recoverability(inventory: Mapping[str, Any]) -> dict[str, bool | None]:
    gates = {
        "all_288_rows_and_24x12_identity_structure_exact":
            inventory["unique_training_view_row_digests"] == 288
            and inventory["unique_branch_identity_digests"] == 288
            and inventory["unique_states"] == 24
            and inventory["unique_state_candidate_pairs"] == 288
            and inventory["states_with_exact_candidates_0_through_11"] == 24
            and inventory["states_with_single_observed_family"] == 24
            and inventory["states_with_single_observed_stratum"] == 24
            and inventory["row_target_prediction_key_schemas_exact"] is True,
        "true_component_and_utility_targets_finite":
            inventory["all_targets_finite"] is True,
        "attentive_probabilities_and_utility_predictions_finite":
            inventory["all_predictions_finite"] is True
            and inventory[
                "all_safety_and_completion_predictions_are_probabilities"] is True,
        "complete_action_blocks_present":
            inventory["rows_missing_action_blocks"] == 0,
        "complete_goal_binding_present":
            inventory["rows_missing_goal_binding_input"] == 0,
        "row_aligned_no_latent_predictions_present":
            inventory["rows_missing_no_latent_prediction"] == 0,
        "complete_split_scene_state_and_family_manifest_provenance_present":
            all(inventory[key] == 0 for key in (
                "rows_missing_split_role", "rows_missing_scene_id",
                "rows_missing_state_identity_digest"))
            and inventory[
                "frozen_calibration_manifest_or_state_family_mapping_retained"]
            is True,
        "source_and_evidence_sufficient_to_reconstruct_direct_and_replay_paths":
            True,
        "consumer_a_and_b_discrete_outputs_exact": None,
        "consumer_a_and_b_float_outputs_within_frozen_tolerance": None,
        "direct_vs_replay_first_metric_divergence_exactly_localised": True,
        "baseline_identity_reorder_verified": None,
        "all_original_gate_inputs_reconstructable": None,
    }
    require(set(gates) == set(C.RECOVERABILITY_GATES),
            "recoverability gate schema changed")
    return gates


def original_gate_replay(metrics: Mapping[str, Any], baseline: Mapping[str, Any],
                         *, completion_fit_nondegenerate: bool,
                         completion_calibration_nondegenerate: bool
                         ) -> dict[str, bool]:
    pairwise = metrics["composite"]["pairwise_ordering_accuracy"]
    baseline_pairwise = baseline["composite"]["pairwise_ordering_accuracy"]
    gain = Decimal(str(pairwise)) - Decimal(str(baseline_pairwise))
    return {
        "progress_spearman": metrics["progress"]["spearman"] is not None
            and metrics["progress"]["spearman"] >= 0.50,
        "safety_auc": metrics["safety"]["auc"] is not None
            and metrics["safety"]["auc"] >= 0.75,
        "safety_ece": metrics["safety"]["ece"] is not None
            and metrics["safety"]["ece"] <= 0.10,
        "completion_auc": metrics["completion"]["auc"] is not None
            and metrics["completion"]["auc"] >= 0.75,
        "completion_ece": metrics["completion"]["ece"] is not None
            and metrics["completion"]["ece"] <= 0.10,
        "pairwise_ordering_accuracy": pairwise is not None and pairwise >= 0.65,
        "latent_over_baseline_pairwise_gain": gain >= Decimal("0.05"),
        "completion_labels_nondegenerate_in_fit_and_calibration":
            completion_fit_nondegenerate and completion_calibration_nondegenerate,
    }


def reconcile_complete_evidence(
        rows: Sequence[Mapping[str, Any]],
        baseline_rows: Sequence[Mapping[str, Any]],
        direct_metric_tree: Mapping[str, Any], *,
        completion_fit_nondegenerate: bool,
        family_assignment_manifest: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build a repaired result only for a separately supplied complete fixture.

    The immutable scientific evidence cannot call this function: every row is
    missing the action, goal, and row-aligned baseline fields required here.
    """
    require(len(rows) == 288 and all(
        isinstance(row.get("action_blocks"), Sequence)
        and not isinstance(row.get("action_blocks"), (str, bytes))
        and len(row["action_blocks"]) == 4
        and all(isinstance(block, Sequence)
                and not isinstance(block, (str, bytes))
                and len(block) == 10
                and all(_finite(value) for value in block)
                for block in row["action_blocks"])
        and isinstance(row.get("goal_binding_input"), Sequence)
        and not isinstance(row.get("goal_binding_input"), (str, bytes))
        and len(row["goal_binding_input"]) == 3
        and all(_finite(value) for value in row["goal_binding_input"])
        and isinstance(row.get("target"), Mapping)
        and _finite(row["target"].get("completion"))
        and float(row["target"].get("completion", -1.0)) in (0.0, 1.0)
        and _finite(row["target"].get("safety"))
        and 0.0 <= float(row["target"].get("safety", -1.0)) <= 1.0
        and "split_role" in row and "scene_id" in row
        and "state_identity_digest" in row for row in rows),
        "complete evidence provenance is absent")
    identities = [(str(row["state_id"]), int(row["candidate_index"]),
                   str(row["training_view_row_digest"]),
                   str(row["branch_identity_digest"])) for row in rows]
    require(len(set(identities)) == 288, "complete evidence identities changed")
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    require(len(by_state) == 24
            and all({int(row["candidate_index"]) for row in state_rows}
                    == set(range(12)) and len(state_rows) == 12
                    for state_rows in by_state.values()),
            "complete evidence is not the exact 24 by 12 design")
    require(set(family_assignment_manifest) == set(by_state),
            "frozen family-assignment manifest identity set changed")
    families: dict[str, int] = defaultdict(int)
    for state_id, state_rows in by_state.items():
        manifest = family_assignment_manifest[state_id]
        expected = {
            "family": str(state_rows[0]["family"]),
            "stratum": str(state_rows[0]["stratum"]),
            "scene_id": str(state_rows[0]["scene_id"]),
            "state_identity_digest": str(state_rows[0]["state_identity_digest"]),
        }
        require(dict(manifest) == expected
                and all(str(row[key]) == value
                        for row in state_rows for key, value in expected.items())
                and all(row["split_role"] == "calibration"
                        for row in state_rows),
                "complete state provenance disagrees with its manifest")
        families[expected["family"]] += 1
    require(len(families) == 8 and set(families.values()) == {3},
            "complete family design changed")
    require(len(baseline_rows) == 288,
            "row-aligned no-latent predictions are incomplete")
    aligned_baseline = align_baseline_predictions(rows, baseline_rows)
    for attentive, baseline in zip(rows, aligned_baseline, strict=True):
        require(all(baseline.get(key) == attentive.get(key) for key in (
            "training_view_row_digest", "branch_identity_digest", "state_id",
            "family", "stratum", "candidate_index", "target"))
            and isinstance(baseline.get("prediction"), Mapping)
            and all(_finite(baseline["prediction"].get(key)) for key in (
                "progress", "safety", "completion", "utility")),
            "row-aligned no-latent prediction evidence changed")
    direct_a = consumer_a(rows, project_component_targets_to_float32=True)
    direct_b = consumer_b(rows, project_component_targets_to_float32=True)
    replay_a = consumer_a(rows, project_component_targets_to_float32=False)
    replay_b = consumer_b(rows, project_component_targets_to_float32=False)
    direct_agreement = compare_consumers(direct_a, direct_b)
    replay_agreement = compare_consumers(replay_a, replay_b)
    retained_direct_agreement = compare_consumers(direct_metric_tree, direct_a)
    require(direct_agreement["passed"] and replay_agreement["passed"]
            and retained_direct_agreement["passed"],
            "the two consumers or retained direct tree disagree")
    baseline = consumer_b(aligned_baseline,
                          project_component_targets_to_float32=True)
    completion_values = {float(row["target"]["completion"]) for row in rows}
    gates = original_gate_replay(
        direct_b, baseline,
        completion_fit_nondegenerate=completion_fit_nondegenerate,
        completion_calibration_nondegenerate=completion_values == {0.0, 1.0})
    payload = {
        "schema": RESULT_SCHEMA,
        "status": C.STATUS,
        "result_label": C.REPAIRED_RESULT_LABEL,
        "classification": C.RECOVERABLE_CLASSIFICATION,
        "consumer_a_direct": direct_a,
        "consumer_b_direct": direct_b,
        "consumer_a_replay": replay_a,
        "consumer_b_replay": replay_b,
        "baseline": baseline,
        "consumer_agreement": {
            "direct": direct_agreement, "replay": replay_agreement,
            "retained_direct": retained_direct_agreement,
        },
        "original_gate_replay": gates,
        "all_original_gates_pass": all(gates.values()),
        "evidence_only": True,
    }
    return _signed(payload, RESULT_SELF_KEY)


# ---------------------------------------------------------- one-shot flow --
def issue_contract(root: Path = C.ROOT) -> dict[str, Any]:
    source = C.source_closure(root)
    lineage = C.lineage_binding(root)
    storage = C.storage_binding(root)
    contract = C.build_contract(source, lineage, storage)
    target = C.runtime_root(root)
    target.mkdir(parents=False, exist_ok=False)
    _publish_once(target / "contract.json", contract)
    return contract


def load_contract(root: Path = C.ROOT) -> dict[str, Any]:
    return C.validate_contract(C.read_json(
        C.runtime_root(root) / "contract.json", "reconciliation contract"))


def _attempt(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "status": C.STATUS,
        "attempt_number": 1,
        "maximum_attempts": 1,
        "contract_digest": contract[C.CONTRACT_SELF_KEY],
        "scientific_attempt_lineage": dict(C.SCIENTIFIC_ATTEMPT_LINEAGE),
        "read_only_evidence_consumer": True,
        "consumer_a_executions": 0,
        "consumer_b_executions": 0,
        "torch_tensor_model_training_predictor_access": 0,
        "retry_resume_or_replacement_authorised": False,
    }
    return _signed(payload, ATTEMPT_SELF_KEY)


def _terminal(contract: Mapping[str, Any], attempt: Mapping[str, Any],
              inspection: Mapping[str, Any], gates: Mapping[str, bool | None]
              ) -> dict[str, Any]:
    failed = [key for key, value in gates.items() if value is False]
    not_reached = [key for key, value in gates.items() if value is None]
    require(failed, "incomplete evidence unexpectedly passed every gate")
    payload = {
        "schema": TERMINAL_SCHEMA,
        "status": C.STATUS,
        "classification": C.EXPECTED_PRIMARY_TERMINAL,
        "contract_digest": contract[C.CONTRACT_SELF_KEY],
        "attempt_digest": attempt[ATTEMPT_SELF_KEY],
        "scientific_attempt_lineage": dict(C.SCIENTIFIC_ATTEMPT_LINEAGE),
        "original_artifact_set_digest":
            contract["lineage"]["artifact_set_digest"],
        "original_artifact_bindings": contract["lineage"]["artifacts"],
        "original_artifact_total_byte_count": sum(
            row["byte_count"] for row in contract["lineage"]["artifacts"].values()),
        "reconciliation_source_commit":
            contract["source_closure"]["source_repository_commit"],
        "reconciliation_source_closure_digest":
            contract["source_closure"][C.SOURCE_SELF_KEY],
        "frozen_no_latent_baseline_binding":
            dict(C.FROZEN_NO_LATENT_BASELINE_BINDING),
        "evidence_inventory": inspection["inventory"],
        "observed_families": inspection["observed_families"],
        "family_assignment_matches_frozen_manifest": None,
        "family_assignment_manifest_verdict":
            inspection["family_assignment_manifest_verdict"],
        "prediction_representation": inspection["prediction_representation"],
        "precision_verdict": {
            "prediction_scalars_checked": 1_152,
            "all_prediction_scalars_exact_float32_round_trip": True,
            "prediction_storage_can_change_ordering_auc_ties_regret": False,
            "component_target_storage": (
                "persisted binary64 source scalars versus direct float32 target "
                "projection changes progress/safety errors and exact safety ties"),
            "composite_true_utility": "stored binary64 scalar in both reducers",
        },
        "metric_path_specification": dict(C.METRIC_PATH_SPECIFICATION),
        "source_reconstruction": inspection["source_reconstruction"],
        "first_component_target_projection_difference":
            inspection["first_component_target_projection_difference"],
        "recoverability_gates": dict(gates),
        "section_2_failed_completeness_gates": failed,
        "downstream_checks_not_reached": {
            key: "NOT_REACHED_BECAUSE_SECTION_2_EVIDENCE_COMPLETENESS_FAILED"
            for key in not_reached},
        "fatal_missing_evidence": {
            "complete_action_blocks": False,
            "complete_goal_binding": False,
            "row_aligned_no_latent_component_and_utility_predictions": False,
            "complete_split_scene_state_and_family_manifest_provenance": False,
        },
        "evidence_completeness_verdict": False,
        "metric_recoverability": {
            "attentive_metric_math": (
                "SOURCE_AND_EVIDENCE_RECONSTRUCTABLE_BUT_NOT_PUBLISHABLE_"
                "AFTER_SECTION_2_STOP"),
            "unavailable_row_level_no_latent_outputs": [
                "progress", "safety", "completion", "composite_utility",
                "selected_candidate_identity", "candidate_order", "ties",
                "score_spread", "rank_regret", "pairwise_ordering",
                "row_alignment",
            ],
            "baseline_comparison_and_gate_table_published": False,
        },
        "direct_replay_localisation": {
            "first_source_representation_divergence":
                "COMPONENT_TARGET_FLOAT32_PROJECTION_BEFORE_DIRECT_REDUCTION",
            "first_source_reconstructed_aggregate_metric_leaf":
                C.SOURCE_RECONSTRUCTION["first_overall_metric_leaf"],
            "tie_induced_pair_count_example":
                C.SOURCE_RECONSTRUCTION["first_tie_induced_state"],
            "source_level_delta_checklist": {
                "component_target_float32_projection_compared": True,
                "progress_and_safety_changed_target_counts_bound": True,
                "overall_family_stratum_tree_digests_bound": True,
                "first_row_and_first_metric_leaf_bound": True,
                "first_tie_induced_pair_count_bound": True,
                "direct_online_accumulator_replay_possible": False,
            },
            "cast_difference_proven_as_sole_cause": False,
            "reason": (
                "the source and evidence reconstruct the cast difference, but "
                "the direct metric table and online accumulators were not retained"),
        },
        "consumers_scientifically_executed": False,
        "consumer_a_executions": 0,
        "consumer_b_executions": 0,
        "consumer_a_metric_tree_digest": None,
        "consumer_b_metric_tree_digest": None,
        "consumer_agreement": None,
        "synthetic_fixture_suite": {
            "runtime_executed": False,
            "implementation_and_fixtures_bound_by_committed_source_closure": True,
        },
        "attentive_metric_table_published": False,
        "original_gate_replay_published": False,
        "original_gate_verdicts": None,
        "strong_mixed_no_readout_interpretation": None,
        "readout_result_label": None,
        "repaired_result_published": False,
        "reconciliation_execution_counters": {
            "checkpoint_deserialisations": 0,
            "scorer_forwards": 0,
            "calibration_batches": 0,
            "training_or_optimizer_updates": 0,
            "predictor_files_or_shards_opened": 0,
            "final_corpus_rows_opened_or_generated": 0,
        },
        "torch_import_or_load_model_forward_training_predictor_access": 0,
        "original_seven_artifacts_modified": False,
        "retry_resume_replacement_training_or_predictor_authorised": False,
        "nothing_running": True,
    }
    return _signed(payload, TERMINAL_SELF_KEY)


def run_once(root: Path = C.ROOT) -> dict[str, Any]:
    runtime = C.runtime_root(root)
    require(runtime.is_dir() and not runtime.is_symlink(),
            "reconciliation namespace is absent")
    require({path.name for path in runtime.iterdir()} == {"contract.json"},
            "reconciliation namespace is already consumed or contaminated")
    contract = load_contract(root)
    require(C.source_closure(root) == contract["source_closure"],
            "reconciliation source changed after contract issuance")
    require(C.lineage_binding(root) == contract["lineage"],
            "scientific artifact lineage changed after contract issuance")
    evidence_path = C.predecessor_root(root) / "calibration_evidence.json"
    evidence = _read_signed(evidence_path, "calibration_evidence_digest",
                            "calibration evidence")
    inspection = inspect_evidence(evidence)
    gates = recoverability(inspection["inventory"])
    # The exact immutable evidence fails four literal completeness gates.  No
    # scientific reducer call, metric table, gate replay, or result is allowed.
    require(not all(gates.values()),
            "complete evidence requires the separately reviewed reducer path")
    attempt = _attempt(contract)
    _publish_once(runtime / "attempt.json", attempt)
    terminal = _terminal(contract, attempt, inspection, gates)
    validate_terminal_payload(terminal, contract, attempt, inspection)
    _publish_once(runtime / "terminal.json", terminal)
    return terminal


def validate_terminal_payload(terminal: Mapping[str, Any],
                              contract: Mapping[str, Any],
                              attempt: Mapping[str, Any],
                              inspection: Mapping[str, Any]) -> dict[str, Any]:
    result = C.validate_signed(terminal, TERMINAL_SELF_KEY,
                               "reconciliation terminal")
    gates = recoverability(inspection["inventory"])
    expected = _terminal(contract, attempt, inspection, gates)
    require(result == expected, "reconciliation terminal semantics changed")
    return result


def validate_terminal(root: Path = C.ROOT) -> dict[str, Any]:
    runtime = C.runtime_root(root)
    require(runtime.is_dir() and not runtime.is_symlink(),
            "reconciliation namespace is absent")
    require({path.name for path in runtime.iterdir()}
            == {"contract.json", "attempt.json", "terminal.json"},
            "reconciliation terminal inventory changed")
    for name in ("contract.json", "attempt.json", "terminal.json"):
        path = runtime / name
        require(path.is_file() and not path.is_symlink()
                and stat.S_IMODE(path.stat().st_mode) == 0o444,
                f"reconciliation artifact changed: {name}")
    contract = load_contract(root)
    require(C.source_closure(root) == contract["source_closure"],
            "reconciliation source changed")
    require(C.lineage_binding(root) == contract["lineage"],
            "scientific artifact lineage changed")
    attempt = _read_signed(runtime / "attempt.json", ATTEMPT_SELF_KEY,
                           "reconciliation attempt")
    require(attempt == _attempt(contract), "reconciliation attempt changed")
    evidence = _read_signed(C.predecessor_root(root) / "calibration_evidence.json",
                            "calibration_evidence_digest",
                            "calibration evidence")
    inspection = inspect_evidence(evidence)
    terminal = _read_signed(runtime / "terminal.json", TERMINAL_SELF_KEY,
                            "reconciliation terminal")
    return validate_terminal_payload(terminal, contract, attempt, inspection)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("issue-contract", "run", "validate"))
    arguments = parser.parse_args(argv)
    if arguments.stage == "issue-contract":
        value = issue_contract()
    elif arguments.stage == "run":
        value = run_once()
    else:
        value = validate_terminal()
    print(json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
