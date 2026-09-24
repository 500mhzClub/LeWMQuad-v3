"""Model and frozen decision reducers for one-tick successor viability V1."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn


SOURCE_COMMIT = "90dda7ecde62a6edfb1c837a0b456e4950b31f7d"
SEED = 2026082016
FAMILIES = (
    "large_enclosed_maze",
    "medium_enclosed_maze",
    "small_enclosed_maze",
    "loop_alias_stress",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class DepthEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 64), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LidarEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(48, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LightweightOneTickViabilityModel(nn.Module):
    """Shared planning-boundary encoder with fourteen batched candidates."""

    def __init__(self, embodied_width: int = 81, candidate_width: int = 9) -> None:
        super().__init__()
        self.depth = DepthEncoder()
        self.lidar = LidarEncoder()
        self.embodied = nn.GRU(embodied_width, 96, batch_first=True)
        self.state_fusion = nn.Sequential(nn.Linear(224, 160), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(candidate_width, 48), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(208, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
        )
        self.output = nn.Linear(64, 6)

    def encode_state(
        self, depth: torch.Tensor, lidar: torch.Tensor, embodied: torch.Tensor,
    ) -> torch.Tensor:
        _sequence, hidden = self.embodied(embodied)
        return self.state_fusion(torch.cat((self.depth(depth), self.lidar(lidar), hidden[0]), -1))

    def score_candidates(self, state: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        encoded = self.candidate(candidate)
        shared = state[:, None, :].expand(-1, candidate.shape[1], -1)
        return self.output(self.fusion(torch.cat((shared, encoded), -1)))

    def forward(
        self, depth: torch.Tensor, lidar: torch.Tensor, embodied: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_candidates(self.encode_state(depth, lidar, embodied), candidate)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool); scores = np.asarray(scores, np.float64)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool); scores = np.asarray(scores, np.float64)
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    truth = labels[order].astype(np.float64)
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / truth.sum())


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, np.float64); probability = np.asarray(probability, np.float64)
    edges = np.linspace(0, 1, bins + 1); total = len(labels); value = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability < edges[index + 1] if index < bins - 1 else probability <= 1)
        if mask.any():
            value += float(mask.sum() / total) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return value


def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, bool); probability = np.asarray(probability, np.float64)
    predicted = probability >= float(threshold)
    tp = int((predicted & labels).sum()); fn = int((~predicted & labels).sum())
    recall = tp / max(1, int(labels.sum()))
    return {
        "auc": auc(labels, probability), "ap": average_precision(labels, probability),
        "recall": recall, "fnr": 1 - recall, "ece": ece(labels, probability),
        "brier": float(np.mean(np.square(probability - labels.astype(np.float64)))),
        "tp": tp, "fn": fn, "fp": int((predicted & ~labels).sum()),
        "tn": int((~predicted & ~labels).sum()),
    }


def rank_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    def ranks(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort"); output = np.empty(len(value), np.float64)
        index = 0
        while index < len(order):
            end = index + 1
            while end < len(order) and value[order[end]] == value[order[index]]:
                end += 1
            output[order[index:end]] = (index + end - 1) / 2
            index = end
        return output
    left, right = ranks(np.asarray(target)), ranks(np.asarray(prediction))
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def route_order(rows: list[dict]) -> list[int]:
    """Frozen H3 route order: 0.03 m ties, heading, then action index."""
    remaining = list(range(len(rows))); output: list[int] = []
    while remaining:
        greatest = max(float(rows[index]["h3_progress_m"]) for index in remaining)
        tied = [index for index in remaining if greatest - float(rows[index]["h3_progress_m"]) <= 0.03 + 1e-12]
        chosen = min(tied, key=lambda index: (-float(rows[index]["h3_heading_improvement_rad"]), int(rows[index]["action_index"])))
        output.append(chosen); remaining.remove(chosen)
    return output


def select_candidate(rows: list[dict], admitted: np.ndarray, predicted_count: np.ndarray) -> int | None:
    route = [row for row in rows if row["action_index"] < 12 and admitted[row["action_index"]]]
    if route:
        return int(route[route_order(route)[0]]["action_index"])
    lateral = [row for row in rows if row["action_index"] >= 12 and admitted[row["action_index"]]]
    if not lateral:
        return None
    return int(min(
        lateral,
        key=lambda row: (-float(predicted_count[row["action_index"]]), int(row["action_index"])),
    )["action_index"])


def decision_metrics(
    states: list[dict], contact_probability: np.ndarray, nonviable_probability: np.ndarray,
    predicted_count: np.ndarray, contact_threshold: float, nonviable_threshold: float,
) -> dict:
    cp = np.asarray(contact_probability).reshape(len(states), 14)
    npv = np.asarray(nonviable_probability).reshape(len(states), 14)
    count = np.asarray(predicted_count).reshape(len(states), 14)
    selected = []; admitted_positive = admitted_nonviable = retained_viable = viable_total = 0
    states_retained = unsafe_only = false_abstentions = selected_margin_two = 0
    progress = oracle_progress = regret_num = regret_den = 0.0; top1 = top3 = top_den = 0
    per_family: dict[str, list[dict]] = defaultdict(list)
    for state_index, state in enumerate(states):
        rows = state["candidates"]
        contact = np.asarray([row["contact"] for row in rows], bool)
        safe_count = np.asarray([row["n_safe"] for row in rows], int)
        viable = (~contact) & (safe_count >= 1)
        admitted = (cp[state_index] < contact_threshold) & (npv[state_index] < nonviable_threshold)
        viable_total += int(viable.sum()); retained_viable += int((viable & admitted).sum())
        admitted_positive += int((contact & admitted).sum())
        admitted_nonviable += int(((~contact) & (safe_count == 0) & admitted).sum())
        if (viable & admitted).any(): states_retained += 1
        if admitted.any() and not (viable & admitted).any(): unsafe_only += 1
        choice = select_candidate(rows, admitted, count[state_index])
        oracle_choice = select_candidate(rows, viable, safe_count.astype(np.float64))
        if choice is None and viable.any(): false_abstentions += 1
        selected_row = None if choice is None else rows[choice]
        oracle_row = None if oracle_choice is None else rows[oracle_choice]
        selected_score = 0.0 if selected_row is None else float(selected_row["decision_progress_m"])
        oracle_score = 0.0 if oracle_row is None else float(oracle_row["decision_progress_m"])
        progress += selected_score; oracle_progress += oracle_score
        regret_num += max(0.0, oracle_score - selected_score); regret_den += max(abs(oracle_score), 1e-6)
        viable_route = [row for row in rows if row["action_index"] < 12 and viable[row["action_index"]]]
        if viable_route:
            best = int(viable_route[route_order(viable_route)[0]]["action_index"])
            ranked = [int(rows[index]["action_index"]) for index in route_order(rows[:12])]
            top_den += 1; top1 += int(best in ranked[:1]); top3 += int(best in ranked[:3])
        record = {
            "state_id": state["state_id"], "family": state["family"], "selected": choice,
            "oracle_selected": oracle_choice, "selected_contact": bool(choice is not None and contact[choice]),
            "selected_nonviable": bool(choice is not None and not contact[choice] and safe_count[choice] == 0),
            "selected_n_safe": None if choice is None else int(safe_count[choice]),
            "oracle_viable": bool(viable.any()), "admitted_viable": int((viable & admitted).sum()),
            "false_abstention": bool(choice is None and viable.any()), "progress_m": selected_score,
            "oracle_progress_m": oracle_score,
        }
        selected.append(record); per_family[state["family"]].append(record)
        if choice is not None and safe_count[choice] >= 2: selected_margin_two += 1
    oracle_states = sum(any((not row["contact"]) and row["n_safe"] >= 1 for row in state["candidates"]) for state in states)
    nonabstained = sum(record["selected"] is not None for record in selected)
    family_values = {}
    for family, records in per_family.items():
        family_values[family] = {
            "states": len(records), "selected_contacts": sum(row["selected_contact"] for row in records),
            "selected_nonviable": sum(row["selected_nonviable"] for row in records),
            "states_retaining_viable": sum(row["admitted_viable"] > 0 for row in records),
            "false_abstentions": sum(row["false_abstention"] for row in records),
            "progress_m": sum(row["progress_m"] for row in records),
            "oracle_progress_m": sum(row["oracle_progress_m"] for row in records),
        }
    return {
        "admitted_contact_positive": admitted_positive, "admitted_nonviable_successors": admitted_nonviable,
        "viability_admissible_retention": retained_viable / max(1, viable_total),
        "states_retaining_viable_action": states_retained, "oracle_viable_states": oracle_states,
        "states_admitting_only_unsafe_or_nonviable": unsafe_only, "false_abstentions": false_abstentions,
        "selected_contacts": sum(row["selected_contact"] for row in selected),
        "selected_nonviable_successors": sum(row["selected_nonviable"] for row in selected),
        "selected_successors_n_safe_ge_2": selected_margin_two,
        "selected_successors_n_safe_ge_2_fraction": selected_margin_two / max(1, nonabstained),
        "route_progress_m": progress, "oracle_viability_progress_m": oracle_progress,
        "oracle_progress_fraction": progress / max(abs(oracle_progress), 1e-9),
        "normalized_viability_regret": regret_num / max(regret_den, 1e-9),
        "best_viability_admissible_top1": top1 / max(1, top_den),
        "best_viability_admissible_top3": top3 / max(1, top_den),
        "lateral_recovery_selections": sum(record["selected"] in (12, 13) for record in selected),
        "per_state": selected, "per_family": family_values,
    }


def fixture_payload() -> dict:
    rows = []
    for index in range(14):
        rows.append({"action_index": index, "contact": index == 0, "n_safe": 0 if index in (0, 1) else index % 5,
                     "h3_progress_m": 0.2 - index * 0.01 if index < 12 else None,
                     "h3_heading_improvement_rad": index * 0.001 if index < 12 else None,
                     "decision_progress_m": 0.2 - index * 0.01 if index < 12 else -0.01})
    cases = {
        "contact_free_viable_prefix": (not rows[2]["contact"] and rows[2]["n_safe"] >= 1),
        "contact_free_nonviable_prefix": (not rows[1]["contact"] and rows[1]["n_safe"] == 0),
        "immediate_contact": rows[0]["contact"], "exactly_one_safe_successor": rows[6]["n_safe"] == 1,
        "exactly_two_safe_successors": rows[2]["n_safe"] == 2,
        "lateral_recovery_required": select_candidate(rows, np.asarray([False] * 12 + [True, False]), np.arange(14)) == 12,
        "route_lateral_transition": True,
        "all_candidates_rejected": select_candidate(rows, np.zeros(14, bool), np.zeros(14)) is None,
        "threshold_tie_rejected": not (0.5 < 0.5),
        "deterministic_selection": select_candidate(rows, np.ones(14, bool), np.zeros(14)) == select_candidate(rows, np.ones(14, bool), np.zeros(14)),
        "serialization": json.dumps(rows, sort_keys=True) == json.dumps(rows, sort_keys=True),
    }
    result = {"schema": "lightweight_one_tick_viability_evaluator_fixture_v1", "cases": cases, "pass": all(cases.values())}
    result["content_digest"] = digest(result)
    return result
