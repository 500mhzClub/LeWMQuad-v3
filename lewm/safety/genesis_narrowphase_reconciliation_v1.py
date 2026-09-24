"""Pure reducers for Genesis narrowphase and candidate-feasibility reconciliation.

The runtime adapter lives in ``scripts/`` because importing Genesis is an
explicitly authorised, heavyweight operation.  This module deliberately has
no Genesis dependency so the evaluator can be frozen on synthetic fixtures
before replay begins.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def event_spans(trace: Sequence[bool]) -> list[tuple[int, int]]:
    x = np.asarray(trace, dtype=bool)
    padded = np.pad(x.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(a), int(b - 1)) for a, b in zip(starts, ends, strict=True)]


def first_contact_error(reference: Sequence[bool], query: Sequence[bool]) -> int | None:
    a = np.flatnonzero(np.asarray(reference, bool))
    b = np.flatnonzero(np.asarray(query, bool))
    if not len(a) and not len(b):
        return 0
    if not len(a) or not len(b):
        return None
    return int(b[0] - a[0])


def binary_confusion(reference: Sequence[bool], query: Sequence[bool]) -> dict:
    y = np.asarray(reference, bool)
    q = np.asarray(query, bool)
    if y.shape != q.shape:
        raise ValueError("shape mismatch")
    tp = int(np.sum(y & q)); fp = int(np.sum(~y & q))
    fn = int(np.sum(y & ~q)); tn = int(np.sum(~y & ~q))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "sensitivity": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": None if tn + fp == 0 else tn / (tn + fp),
        "agreement": float(np.mean(y == q)),
    }


def candidate_divergence_step(
    link_transform: np.ndarray,
    *,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = 1e-3,
) -> int | None:
    """First step at which any candidate pair materially diverges.

    ``link_transform`` has shape ``[candidate, step, link, 7]`` and quaternions
    are scalar-first.  The tolerance is prospective and independent of labels.
    """
    x = np.asarray(link_transform, np.float64)
    if x.ndim != 4 or x.shape[-1] != 7:
        raise ValueError("expected [candidate, step, link, 7]")
    anchor = x[0]
    pos = np.max(np.linalg.norm(x[..., :3] - anchor[None, ..., :3], axis=-1), axis=(0, 2))
    qa = anchor[None, ..., 3:]
    qb = x[..., 3:]
    dot = np.abs(np.sum(qa * qb, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    angle = np.max(2.0 * np.arccos(dot), axis=(0, 2))
    ids = np.flatnonzero((pos > position_tolerance_m) | (angle > orientation_tolerance_rad))
    return None if not len(ids) else int(ids[0])


def command_divergence_tick(commands: np.ndarray, tolerance: float = 1e-8) -> int | None:
    x = np.asarray(commands, np.float64)
    if x.ndim != 3:
        raise ValueError("expected [candidate, tick, action]")
    spread = np.ptp(x, axis=0)
    ids = np.flatnonzero(np.any(spread > tolerance, axis=-1))
    return None if not len(ids) else int(ids[0])


def classify_no_safe_state(
    *,
    boundary_contact: bool,
    first_contact_steps: Sequence[int | None],
    trajectory_divergence_step: int | None,
    avoiding_response_step: int | None,
    candidate_effect_evidence: bool,
) -> str:
    finite = [int(x) for x in first_contact_steps if x is not None]
    if not finite:
        raise ValueError("no-safe state needs positive candidates")
    earliest = min(finite)
    response = trajectory_divergence_step
    if avoiding_response_step is not None:
        response = avoiding_response_step if response is None else min(response, avoiding_response_step)
    if boundary_contact or response is None or max(finite) <= response:
        return "PRE_EXISTING_OR_IMMEDIATE_UNAVOIDABLE_CONTACT"
    if candidate_effect_evidence and earliest > response:
        return "CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE"
    return "UNRESOLVED_NO_SAFE_CANDIDATE"


def route_order(rows: list[Mapping], ids: Sequence[int]) -> list[int]:
    """Frozen deterministic route order: nominal distance, heading, index."""
    remaining = list(ids); result = []
    while remaining:
        best_distance = max(float(rows[i]["kinematic"][4]) for i in remaining)
        near = [i for i in remaining if best_distance - float(rows[i]["kinematic"][4]) <= 0.03]
        pick = min(near, key=lambda i: (-float(rows[i]["kinematic"][5]), int(rows[i]["candidate_index"])))
        result.append(pick); remaining.remove(pick)
    return result


def realised_preference(a: Mapping, b: Mapping) -> int:
    distance = float(a["p_d"]) - float(b["p_d"])
    if abs(distance) > 0.03:
        return 1 if distance > 0 else -1
    heading = float(a["p_theta"]) - float(b["p_theta"])
    limit = np.deg2rad(5.0)
    if abs(heading) > limit:
        return 1 if heading > 0 else -1
    return 0


def feasibility_metrics(rows: list[Mapping], admitted: Sequence[bool]) -> dict:
    """Evaluate mobility only where a physics-contact-negative candidate exists."""
    admitted = np.asarray(admitted, bool)
    labels = np.asarray([bool(r["hard_contact"]) for r in rows])
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row["state_id"])].append(i)

    per_state = []
    safe_state_count = retained_states = false_abstentions = 0
    no_safe_count = correct_abstentions = unsafe_no_safe_moves = 0
    selected_positive = 0; selected_progress = []; oracle_progress = []
    regrets = []; top1 = []; top3 = []
    negative_total = negative_admitted = positive_admitted = 0
    for state_id, ids in sorted(grouped.items()):
        rank = route_order(rows, ids)
        negative = [i for i in ids if not labels[i]]
        available = [i for i in rank if admitted[i]]
        pick = available[0] if available else None
        best = None
        for i in negative:
            if best is None or realised_preference(rows[i], rows[best]) > 0:
                best = i
        oracle = next((i for i in rank if i in negative), None)
        n_admitted = sum(admitted[i] for i in negative)
        p_admitted = sum(admitted[i] for i in ids if labels[i])
        negative_total += len(negative); negative_admitted += n_admitted; positive_admitted += p_admitted
        if negative:
            safe_state_count += 1
            retained_states += n_admitted > 0
            false_abstentions += pick is None
            if oracle is not None:
                oracle_progress.append(float(rows[oracle]["p_d"]))
            if best is not None:
                top1.append(pick == best)
                top3.append(best in available[:3])
                if pick is not None and not labels[pick] and len(negative) >= 2:
                    values = [float(rows[i]["p_d"]) for i in negative]
                    spread = max(values) - min(values)
                    if spread > 1e-8:
                        regrets.append((float(rows[best]["p_d"]) - float(rows[pick]["p_d"])) / spread)
        else:
            no_safe_count += 1
            correct_abstentions += pick is None
            unsafe_no_safe_moves += pick is not None
        if pick is not None:
            selected_positive += labels[pick]
            if negative:
                selected_progress.append(float(rows[pick]["p_d"]))
        per_state.append({
            "state_id": state_id,
            "family": str(rows[ids[0]]["family"]),
            "feasibility": "SAFE_CANDIDATE_AVAILABLE" if negative else "NO_SAFE_CANDIDATE_AVAILABLE",
            "contact_negative_candidates": len(negative),
            "admitted_contact_negative": int(n_admitted),
            "admitted_contact_positive": int(p_admitted),
            "selected_candidate": None if pick is None else int(rows[pick]["candidate_index"]),
            "selected_contact": None if pick is None else bool(labels[pick]),
            "selected_progress_m": None if pick is None else float(rows[pick]["p_d"]),
        })
    progress = float(np.mean(selected_progress)) if selected_progress else 0.0
    oracle = float(np.mean(oracle_progress)) if oracle_progress else 0.0
    return {
        "safe_candidate_available_states": safe_state_count,
        "no_safe_candidate_available_states": no_safe_count,
        "states_retaining_contact_negative": int(retained_states),
        "safe_state_retention_rate": None if safe_state_count == 0 else retained_states / safe_state_count,
        "contact_negative_candidate_retention": None if negative_total == 0 else negative_admitted / negative_total,
        "admitted_contact_negative_count": int(negative_admitted),
        "admitted_contact_positive_count": int(positive_admitted),
        "selected_contact_count": int(selected_positive),
        "false_abstentions": int(false_abstentions),
        "false_abstention_rate": None if safe_state_count == 0 else false_abstentions / safe_state_count,
        "correct_abstentions_no_safe": int(correct_abstentions),
        "correct_abstention_rate_no_safe": None if no_safe_count == 0 else correct_abstentions / no_safe_count,
        "unsafe_movements_no_safe": int(unsafe_no_safe_moves),
        "mean_selected_route_progress_m": progress,
        "oracle_contact_kinematic_progress_m": oracle,
        "oracle_progress_fraction": None if abs(oracle) <= 1e-9 else progress / oracle,
        "normalized_route_progress_regret": None if not regrets else float(np.mean(regrets)),
        "normalized_regret_states": len(regrets),
        "best_contact_negative_top1": None if not top1 else float(np.mean(top1)),
        "best_contact_negative_top3": None if not top3 else float(np.mean(top3)),
        "per_state": per_state,
    }


def fixture_payload() -> dict:
    clear = np.zeros((3, 4, 2, 7), np.float64)
    clear[..., 3] = 1.0
    diverged = clear.copy(); diverged[2, 2:, 1, 0] = 0.002
    commands = np.zeros((3, 5, 2)); commands[1, 0, 0] = 0.2
    rows = []
    for state, labels in (("safe", [False, True]), ("none", [True, True])):
        for candidate, label in enumerate(labels):
            kinematic = np.zeros(6); kinematic[4] = .2 - .1 * candidate
            rows.append({"state_id": state, "family": "fixture", "candidate_index": candidate,
                         "hard_contact": label, "p_d": .2 - .1 * candidate, "p_theta": 0.0, "kinematic": kinematic})
    metrics = feasibility_metrics(rows, [True, False, False, False])
    tests = {
        "event_grouping": event_spans([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)],
        "first_contact_error": first_contact_error([0, 1], [0, 0, 1]) == 1,
        "command_divergence": command_divergence_tick(commands) == 0,
        "trajectory_divergence": candidate_divergence_step(diverged) == 2,
        "preexisting_class": classify_no_safe_state(boundary_contact=True, first_contact_steps=[0, 1], trajectory_divergence_step=2, avoiding_response_step=2, candidate_effect_evidence=False) == "PRE_EXISTING_OR_IMMEDIATE_UNAVOIDABLE_CONTACT",
        "bank_class": classify_no_safe_state(boundary_contact=False, first_contact_steps=[5, 8], trajectory_divergence_step=2, avoiding_response_step=2, candidate_effect_evidence=True) == "CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE",
        "safe_denominator": metrics["safe_candidate_available_states"] == 1 and metrics["no_safe_candidate_available_states"] == 1,
        "correct_no_safe_abstention": metrics["correct_abstention_rate_no_safe"] == 1.0,
    }
    payload = {"schema": "genesis_narrowphase_candidate_feasibility_fixture_v1", "tests": tests, "pass": all(tests.values())}
    payload["content_digest"] = digest(payload)
    payload["byte_identical_regeneration"] = payload["content_digest"] == digest({k: v for k, v in payload.items() if k not in {"content_digest", "byte_identical_regeneration"}})
    return payload
